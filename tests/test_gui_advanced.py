"""Advanced mode, through the GUI server with no browser and no GPU.

1. A walk of the API table fails on any POST without a command: every endpoint is exercised, its reply carries the
   ``gpuwm-cm1`` line it ran (and a dry run of the same request returns the same line and changes no file), and
   each line, run from a shell against a copy of the run directory taken just before, gives the same files.
2. The mode survives a server restart and a second browser, and toggling it changes no file in any run directory
   (every run directory is hashed before and after).
3. Editing the namelist imports it again, the reply carries the six-category report, and the diff is stored.
4. A guided run opened in advanced mode shows its fitted namelist, its coverage report and fit.json, and Continue
   or extend (resume, from a chosen restart to a later storm time) works on it.
5. Every screen of section 6 is reachable in advanced mode and every guided screen from it; the only mode-only
   features are the gallery layout and the budget buttons (a static check of the pages; a browser walk of the
   same screens stays in the development repository).

The real host-only commands (import, edit, check, cartesian --plan, estimate, jobscript, preset) run where the
native namelist parser is available (a checkout builds it); the fake runner stands in for the GPU.  CPU only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable

import pytest

from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.gui import commands as C
from cm1import_support import namelist_tool
from gui_support import ROOT, SUPERCELL, Client, import_supercell, start_server, stop_server, wait_for

STATIC = ROOT / "src/cm1core/gui/static"
GPUINFO = ROOT / "tests/data/gpuinfo"
PY = sys.executable


@pytest.fixture(autouse=True)
def _environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", os.fspath(GPUINFO / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", os.fspath(GPUINFO / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setenv("GPUWM_CM1_HOST_FIXTURE", os.fspath(GPUINFO / "host-64gb-1tb.json"))
    monkeypatch.setenv("GPUWM_CM1_CACHE_DIR", os.fspath(tmp_path / "cache"))
    monkeypatch.setenv(rates.CACHE_ENV, os.fspath(tmp_path / "cache"))
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    monkeypatch.setenv("GM1_FAKE_MEASURE_S", "0.1")
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


def _mixed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))
    from cm1core.gui.jobs import mixed_runner

    return start_server(tmp_path / "runs", tmp_path / "config", runner=mixed_runner())


# ------------------------------------------------------------------ files, hashes and the shell


def _digest(root: Path) -> dict[str, str]:
    """Every file under the run directories of ``root`` (the server's own .gm1-gui folder aside)."""

    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file() and ".gm1-gui" not in path.parts}


def _shim(tmp_path: Path, kind: str) -> Path:
    """``gpuwm-cm1`` on PATH as the server's runner runs it: the fake runner, or the real CLI with a fake GPU."""

    bin_dir = tmp_path / f"bin-{kind}"
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "gpuwm-cm1"
    if kind == "fake":
        shim.write_text(f"#!/bin/sh\nexec {PY} -m cm1core.gui.fake_runner \"$@\"\n")
    else:
        shim.write_text(
            "#!/bin/bash\n"
            f'fake() {{ exec {PY} -m cm1core.gui.fake_runner "$@"; }}\n'
            'case "$1" in run|resume) fake "$@";; esac\n'
            'if [ "$1" = estimate ]; then for a in "$@"; do [ "$a" = --measure ] && fake "$@"; done; fi\n'
            f'exec {PY} -m cm1core "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _replay(lines: list[str], bin_dir: Path, cwd: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "PYTHONPATH": str(ROOT / "src"),
           "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(["bash", "-c", "\n".join(lines)], env=env, capture_output=True, text=True, timeout=300,
                          cwd=cwd)


#: Values that record when, where or how fast something ran, not what the command made.
_MOMENT = re.compile(r"(_utc$|^utc$|elapsed|wall_s$|^pid|_ticks$|^started|^time$|seconds_measured|^host$|"
                     r"^measured_at|^when|_at_utc$|^made)", re.I)
_MOMENT_HOST = {"ram_available_bytes", "disk_free_bytes", "disk_total_bytes", "ram_total_bytes"}


def _steady(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _steady(item) for key, item in value.items()
                if not _MOMENT.search(str(key)) and key not in _MOMENT_HOST}
    if isinstance(value, list):
        return [_steady(item) for item in value]
    return value


#: Files a run writes as it goes (times, process ids, rates), and a reproduce bundle (a tarball with its own file
#: times): compared by name, not by content.
_LIVE = re.compile(r"^(segments/|cm1out/|quicklook/|control/|reproduce/)|^run\.json$|^commands\.log$")


def _files(rundir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted(rundir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(rundir).as_posix()
        if rel == "commands.log":
            continue  # the server writes it; a shell replay does not
        if _LIVE.search(rel):
            out[rel] = "(written as it ran)"
            continue
        data = path.read_bytes()
        if path.suffix == ".sh":
            # A job script's header says when it was written.
            data = re.sub(rb"(?m)^# Written \S+ by .*$", b"# Written (time) by ...", data)
        if path.suffix == ".json":
            try:
                out[rel] = _steady(json.loads(data))
                continue
            except ValueError:
                pass
        out[rel] = hashlib.sha256(data).hexdigest()
    return out


class Walk:
    """Each POST as the page makes it, then the same line from a shell against the run directory as it was."""

    def __init__(self, server, client: Client, root: Path, bin_dir: Path, scratch: Path) -> None:
        self.server = server
        self.client = client
        self.root = root
        self.bin_dir = bin_dir
        self.scratch = scratch
        self.seen: dict[str, dict[str, Any]] = {}
        self.count = 0

    def settle(self, run: str | None) -> None:
        if run:
            wait_for(lambda: not self.server.jobs.launched(run), timeout=120)

    def post(self, action: str, path: str, body: dict[str, Any], *, run: str | None,
             replay: bool = True, expect_ok: bool = True, lines: Callable[[dict], list[str]] | None = None,
             same: Callable[[Path, Path], None] | None = None) -> dict[str, Any]:
        endpoint = C.endpoint_for([part for part in path.split("/")[2:]])
        assert endpoint is not None and endpoint.action == action, (action, path)
        rundir = self.root / run if run else None
        # The dry run: the same line, and no file in any run directory changes.
        before_all = _digest(self.root)
        dry = self.client.post(path, {**body, "dry_run": True})
        assert dry.status == 200, (action, dry.json)
        assert dry.json["dry_run"] is True and dry.json["command"].startswith("gpuwm-cm1 "), (action, dry.json)
        assert _digest(self.root) == before_all, f"{action}: a dry run changed a file"
        # The action itself, from a copy of the run directory as it stood.
        self.count += 1
        snapshot = self.scratch / f"before-{self.count:02d}-{action}"
        existed = rundir is not None and rundir.exists()
        if existed:
            shutil.copytree(rundir, snapshot, symlinks=True)
        reply = self.client.post(path, body)
        assert reply.status == 200, (action, reply.json)
        data = reply.json
        assert isinstance(data.get("command"), str) and data["command"].startswith("gpuwm-cm1 "), (action, data)
        assert data["command"] == dry.json["command"], (action, dry.json["command"], data["command"])
        if expect_ok:
            assert data.get("ok") is not False, (action, data)
        self.seen[action] = data
        self.settle(run)
        if not replay or rundir is None:
            return data
        # Replay: the run directory as it was, the command from a shell, then the two compared.
        made = self.scratch / f"after-{self.count:02d}-{action}"
        shutil.move(rundir, made)
        if existed:
            shutil.copytree(snapshot, rundir, symlinks=True)
        script = lines(data) if lines else data["command"].splitlines()
        done = _replay(script, self.bin_dir, self.scratch)
        # The shell gives the exit code the GUI's call got (0, or a refusal's 2 where the machine refuses it).
        want_rc = data["rc"] if isinstance(data.get("rc"), int) else 0
        assert done.returncode == want_rc, (action, script, done.stdout[-2000:], done.stderr[-2000:])
        if same is not None:
            same(rundir, made)
        else:
            got, want = _files(rundir), _files(made)
            assert got == want, (action, sorted(set(got) ^ set(want)),
                                 [key for key in got if key in want and got[key] != want[key]])
        # Go on from the directory the GUI made, as the session did.
        shutil.rmtree(rundir)
        shutil.move(made, rundir)
        return data


def _launch_lines(data: dict[str, Any]) -> list[str]:
    """A run as commands.log records it: the sidecars in the background, the run, then wait."""

    sidecars = data.get("sidecars") or []
    return [*sidecars, data["command"], *(["wait"] if sidecars else [])]


def _same_run(rundir: Path, made: Path) -> None:
    """A run replayed: the same setup files, and the same kinds of files in each segment."""

    got, want = _files(rundir), _files(made)
    assert {k: v for k, v in got.items() if not _LIVE.search(k)} == \
        {k: v for k, v in want.items() if not _LIVE.search(k)}

    def shape(files: dict[str, Any]) -> set[str]:
        return {"/".join(key.split("/")[:3]) if key.startswith("segments/") else key.split("/")[0] for key in files}

    assert shape(got) == shape(want), (shape(got) ^ shape(want))


# ------------------------------------------------------------------ 1. the API table, walked


def test_every_post_of_the_api_table_returns_its_command_and_replays_from_a_shell(tmp_path, monkeypatch):
    """Every POST returns its command and replays from a shell: the host-only and setup actions with the real commands (fake GPU)."""

    server, client = _mixed(tmp_path, monkeypatch)
    root = Path(server.root.path)
    scratch = tmp_path / "walk"
    scratch.mkdir()
    walk = Walk(server, client, root, _shim(tmp_path, "mixed"), scratch)
    try:
        text = SUPERCELL.read_text(encoding="utf-8")
        walk.post("import", "/api/import", {"name": "sc1", "namelist": {"text": text}}, run="sc1")
        walk.post("check", "/api/runs/sc1/check", {}, run="sc1")
        # The plan reads ArWen's audited sources; a Python without gpuwm 2.7.4 refuses it, with its command.
        walk.post("geometry", "/api/runs/sc1/geometry", {}, run="sc1",
                  expect_ok=importlib.util.find_spec("gpuwm") is not None)
        walk.post("estimate", "/api/runs/sc1/estimate", {}, run="sc1")
        walk.post("edit", "/api/runs/sc1/edit",
                  {"set": {"tapfrq": 600.0}, "settings": {"initialization.zc_m": 1500.0}}, run="sc1")
        walk.post("estimate", "/api/runs/sc1/estimate", {}, run="sc1")
        walk.post("recommended", "/api/runs/sc1/recommended", {}, run="sc1")
        walk.post("namelist", "/api/runs/sc1/namelist", {"text": text.replace(" tapfrq =   900.0,", " tapfrq =   300.0,")},
                  run="sc1")
        walk.post("namelist", "/api/runs/sc1/namelist", {"revert": True}, run="sc1")
        walk.post("acknowledge", "/api/runs/sc1/acknowledge", {"output_keys": True}, run="sc1")
        sounding = tmp_path / "input_sounding"
        sounding.write_text("1000.0 300.0 14.0\n")
        walk.post("sounding", "/api/runs/sc1/sounding", {"sounding": {"path": os.fspath(sounding)}}, run="sc1")
        walk.post("jobscript", "/api/runs/sc1/jobscript",
                  {"scheduler": "pbs", "time": "02:00:00", "cpus": 4, "mem": "16G", "until_s": 9000, "write": True,
                   "setup": ["module load cuda", "export OMP_NUM_THREADS=1"], "python": PY},
                  run="sc1")
        walk.post("measure", "/api/runs/sc1/measure", {}, run="sc1",
                  same=lambda got, want: None)          # its record is the rate cache, outside the run
        walk.post("fit", "/api/fit", {"preset": "supercell-1km", "budget": "10m"}, run=None)
        walk.post("preset-new", "/api/presets/new", {"preset": "supercell-1km", "budget": "10m", "name": "p1"},
                  run="p1")
        walk.post("guided-start", "/api/guided/start", {"preset": "supercell-1km", "budget": "10m", "name": "g1"},
                  run="g1", lines=lambda data: [data["preset_command"], *_launch_lines(
                      {"command": data["run_command"], "sidecars": data.get("sidecars")})],
                  same=_same_run)
        walk.post("mode", "/api/mode", {"mode": "advanced"}, run=None)
        # The replies name what they ran; the guided Start's two lines, and the job script's options.
        assert "--until 9000" in walk.seen["jobscript"]["command"] and "--write" in walk.seen["jobscript"]["command"]
        assert "--setup 'module load cuda' --setup 'export OMP_NUM_THREADS=1'" in walk.seen["jobscript"]["command"]
        assert f"--python {PY}" in walk.seen["jobscript"]["command"]
        written = (root / "sc1" / "job-pbs.sh").read_text()
        assert "\nmodule load cuda\nexport OMP_NUM_THREADS=1\n" in written and PY in written
        assert walk.seen["guided-start"]["command"].splitlines()[0].startswith("gpuwm-cm1 preset new supercell-1km")
        assert walk.seen["namelist"]["command"].endswith(f"--from {root / 'sc1' / 'namelist.input.original'} --json")
        logged = (root / "sc1" / "commands.log").read_text()
        for action in ("import", "check", "geometry", "estimate", "edit", "namelist", "acknowledge", "sounding",
                       "jobscript", "measure"):
            assert walk.seen[action]["command"] in logged, action
    finally:
        stop_server(server)
    assert set(walk.seen) >= MIXED_WALK, sorted(MIXED_WALK - set(walk.seen))


#: The actions each of the two replay walks covers; together they must be the whole API table (checked below).
MIXED_WALK = {"import", "check", "geometry", "estimate", "edit", "recommended", "namelist", "acknowledge", "sounding",
              "jobscript", "measure", "fit", "preset-new", "guided-start", "mode"}
FAKE_WALK = {"import", "start", "fields", "slice", "section", "column", "resume", "pause", "stop", "force-stop"}
#: The Results screen's actions (Convert again, Reproduce bundle), walked where the API table has them.
RESULTS_WALK = {"convert", "reproduce"}
#: Films and looks drawn on request (the fake walk).
LOOKS_WALK = {"film", "look"}
#: The WRF column route (open, column, use), walked in tests/test_gui_wrf.py with the same Walk.
WRF_WALK = {"wrf-open", "wrf-column", "wrf-use"}
#: The Use my own sounding card (check, options), walked in tests/test_gui_own_sounding.py with the same Walk.
SOUNDING_WALK = {"sounding-check", "sounding-options"}
#: The Downscale (3D) screen (open, check, new), walked in tests/test_gui_chain.py with the same Walk.
CHAIN_WALK = {"chain-open", "chain-check", "chain-new"}


def _posts() -> set[str]:
    return {endpoint.action for endpoint in C.API if endpoint.method == "POST"}


def test_the_run_control_and_picture_actions_return_their_command_and_replay(tmp_path, monkeypatch):
    """Every POST returns its command and replays: the actions that launch or control a run and draw pictures (the fake runner throughout)."""

    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    root = Path(server.root.path)
    scratch = tmp_path / "walk"
    scratch.mkdir()
    walk = Walk(server, client, root, _shim(tmp_path, "fake"), scratch)
    try:
        text = SUPERCELL.read_text(encoding="utf-8").replace(" rstfrq = -3600.0,", " rstfrq =  1800.0,")
        walk.post("import", "/api/import", {"name": "r", "namelist": {"text": text}}, run="r")
        walk.post("start", "/api/runs/r/start", {}, run="r", lines=_launch_lines, same=_same_run)
        walk.post("fields", "/api/runs/r/fields", {}, run="r")
        walk.post("slice", "/api/runs/r/slice", {"field": "w", "height_m": 5000.0}, run="r")
        walk.post("section", "/api/runs/r/section", {"field": "w", "from_km": [10.0, 60.0], "to_km": [110.0, 60.0]},
                  run="r")
        walk.post("column", "/api/runs/r/column", {"at_km": [60.0, 60.0]}, run="r")
        # A film of a look, and a look drawn once per run, drawn now
        walk.post("film", "/api/runs/r/film", {"look": "swath", "fps": 8}, run="r")
        walk.post("look", "/api/runs/r/look", {"look": "environment"}, run="r")
        restarts = client.get("/api/runs/r/restarts").json
        assert restarts["can_resume"] and [row["number"] for row in restarts["restarts"]][:1] == [4]
        walk.post("resume", "/api/runs/r/resume", {"from_restart": 2, "until_s": 5400.0}, run="r",
                  lines=_launch_lines, same=_same_run)
        assert walk.seen["resume"]["command"].endswith("--from-restart 2 --until 5400")
        if "convert" in _posts():
            walk.post("convert", "/api/runs/r/convert", {}, run="r", same=_same_run)
        if "reproduce" in _posts():
            walk.post("reproduce", "/api/runs/r/reproduce", {}, run="r")
        # Stop, pause and force stop act on a run that is going: a slow one, controlled from a copy as it runs.
        monkeypatch.setenv("GM1_FAKE_STEP_S", "0.05")
        import_supercell(client, "s")
        assert client.post("/api/runs/s/start").json["ok"]
        client.events("s", lambda e: e.get("event") == "progress" and (e["data"].get("step") or 0) >= 2, timeout=60)
        rundir = root / "s"
        for action in ("pause", "stop"):
            # From a shell, against a copy taken while the run goes on (the run the copy names is still alive),
            # then from the page against the run itself: the same control file in both.
            copy_dir = scratch / f"copy-{action}"
            shutil.copytree(rundir, copy_dir, symlinks=True)
            line = C.display([action, str(copy_dir)])
            done = _replay([line], walk.bin_dir, scratch)
            assert done.returncode == 0, done.stderr
            reply = client.post(f"/api/runs/s/{action}").json
            assert reply["ok"] and reply["command"] == C.display([action, str(rundir)]), reply
            assert reply["command"].replace(str(rundir), str(copy_dir)) == line
            walk.seen[action] = reply
            assert (rundir / "control" / action).is_file() and (copy_dir / "control" / action).is_file()
        forced = client.post("/api/runs/s/force-stop").json
        assert forced["command"] == C.display(["stop", str(rundir), "--force"]), forced
        walk.seen["force-stop"] = forced
        wait_for(lambda: not server.jobs.launched("s"), timeout=60)
    finally:
        stop_server(server)
    expected = FAKE_WALK | (RESULTS_WALK & _posts()) | LOOKS_WALK
    assert set(walk.seen) >= expected, sorted(expected - set(walk.seen))


def test_the_two_walks_cover_every_post_in_the_api_table():
    """A new endpoint fails this test until it joins a replay walk."""

    posts = _posts()
    walked = (MIXED_WALK | FAKE_WALK | (RESULTS_WALK & posts) | LOOKS_WALK | (WRF_WALK & posts) | (CHAIN_WALK & posts)
              | (SOUNDING_WALK & posts))
    assert posts == walked, sorted(posts ^ walked)


def test_a_post_that_fails_before_it_has_a_command_names_its_template(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        import_supercell(client, "t")
        bad = client.post("/api/runs/t/resume", {"from_restart": "x"})
        assert bad.status == 400 and bad.json["command_template"].startswith("gpuwm-cm1 resume RUNDIR")
        refused = client.post("/api/runs/t/jobscript", {"scheduler": "lsf"})
        assert refused.status == 400 and refused.json["command_template"].startswith("gpuwm-cm1 jobscript RUNDIR")
        not_a_flag = client.post("/api/runs/t/stop", {"dry_run": "yes"})
        assert not_a_flag.status == 400 and "dry_run" in not_a_flag.json["message"]
        # A dry run that the action would refuse says so, with the line it would have run.
        dup = client.post("/api/import", {"name": "t", "namelist": {"text": "&param1\n/\n"}, "dry_run": True})
        assert dup.status == 409 and dup.json["command_template"].startswith("gpuwm-cm1 import")
    finally:
        stop_server(server)


def test_the_job_script_takes_every_option_and_refuses_lines_it_cannot_hold(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        import_supercell(client, "j")
        rundir = tmp_path / "runs" / "j"
        shown = client.post("/api/runs/j/jobscript", {
            "scheduler": "slurm", "time": "01:00:00", "partition": "gpu", "account": "abc", "qos": "normal",
            "mem": "32G", "cpus": 8, "gpu_type": "a100", "until_s": 5400, "python": "/opt/gm1/bin/python",
            "setup": "module load cuda\n\n  source /opt/gm1/bin/activate  \n", "dry_run": True}).json
        assert shown["command"] == C.display(C.jobscript_args(
            rundir, "slurm", time="01:00:00", partition="gpu", account="abc", qos="normal", mem="32G", cpus=8,
            gpu_type="a100", until_s=5400, setup=["module load cuda", "source /opt/gm1/bin/activate"],
            python="/opt/gm1/bin/python"))
        made = client.post("/api/runs/j/jobscript", {"scheduler": "slurm", "setup": ["module load cuda"]}).json
        assert made["ok"] and made["command"].endswith("--setup 'module load cuda' --json")
        assert "\nmodule load cuda\n" in made["script"]
        for bad in ({"setup": ["--until 1"]}, {"setup": ["x" * 301]}, {"setup": ["a\tb"]}, {"setup": [1]},
                    {"setup": ["ok"] * 21}, {"python": "~/gm1/bin/python"}, {"python": "/opt/my python"},
                    {"python": "-c"}):
            refused = client.post("/api/runs/j/jobscript", {"scheduler": "slurm", **bad})
            assert refused.status == 400 and refused.json["command_template"].startswith("gpuwm-cm1 jobscript"), bad
    finally:
        stop_server(server)


def test_continue_counts_later_end_times_from_the_runs_own_line_and_offers_each_once(tmp_path):
    """Gone on from restart 1 of a 2 h run to T+1:00: its line is at 1 h, the old line's later restarts are a
    branch, so "60 minutes longer" is timax and offered once, and "30 minutes longer" is T+1:30."""

    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        text = SUPERCELL.read_text(encoding="utf-8").replace(" rstfrq = -3600.0,", " rstfrq =  1800.0,")
        assert client.post("/api/import", {"name": "c", "namelist": {"text": text}}).json["ok"]
        assert client.post("/api/runs/c/start").json["ok"]
        wait_for(lambda: client.get("/api/runs/c/status").json["state"] == "finished", timeout=60)
        wait_for(lambda: not server.jobs.launched("c"), timeout=60)
        assert client.post("/api/runs/c/resume", {"from_restart": 1, "until_s": 3600.0}).json["ok"]
        wait_for(lambda: (lambda s: s["state"] == "finished" and s["segment"] == "002")(
            client.get("/api/runs/c/status").json), timeout=60)
        wait_for(lambda: not server.jobs.launched("c"), timeout=60)
        d = client.get("/api/runs/c/restarts").json
        assert d["can_resume"] and d["reached_s"] == 3600.0
        ends = [option["until_s"] for option in d["until_options"]]
        assert ends == [7200.0, 5400.0], d["until_options"]
        newest = [row for row in d["restarts"] if row["newest"]]
        assert len(newest) == 1 and newest[0]["segment"] == 2 and newest[0]["on_line"]
        # The old line's restart with the same number is a branch, reached with --from-segment.
        branch = [row for row in d["restarts"] if not row["on_line"] and row["number"] == newest[0]["number"]]
        assert branch and branch[0]["needs_segment"] and "--from-segment 1" in branch[0]["command"]
    finally:
        stop_server(server)


def test_a_dry_run_launches_nothing_and_logs_nothing(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        import_supercell(client, "d")
        log_before = (tmp_path / "runs" / "d" / "commands.log").read_text()
        for action in ("start", "measure", "estimate", "check", "stop", "pause", "force-stop"):
            reply = client.post(f"/api/runs/d/{action}", {"dry_run": True}).json
            assert reply["dry_run"] and reply["command"].startswith("gpuwm-cm1 "), (action, reply)
        started = client.post("/api/runs/d/start", {"dry_run": True}).json
        assert started["sidecars"] and all(" --follow &" in line for line in started["sidecars"])
        assert not server.jobs.launched("d")
        assert (tmp_path / "runs" / "d" / "commands.log").read_text() == log_before
        assert not (tmp_path / "runs" / "d" / "segments").exists()
        assert client.get("/api/runs/d/status").json["state"] == "imported"
        # Asking for an upload's line (a Save, an import, a sounding) names its staged copy and writes nothing,
        # however often the page asks while the text is typed.
        uploads = tmp_path / "runs" / ".gm1-gui" / "uploads"
        staged_before = sorted(path.relative_to(uploads) for path in uploads.rglob("*")) if uploads.exists() else []
        text = (tmp_path / "runs" / "d" / "namelist.input").read_text()
        for n in range(3):
            edited = text.replace(" tapfrq =   900.0,", f" tapfrq =   {600 + n}.0,")
            saved = client.post("/api/runs/d/namelist", {"text": edited, "dry_run": True}).json
            assert saved["dry_run"] and " --from " in saved["command"], saved
            named = Path(shlex.split(saved["command"])[4])
            assert named.parent.name == hashlib.sha256(edited.encode()).hexdigest()[:24] and not named.exists()
        assert client.post("/api/import", {"name": "d2", "namelist": {"text": text + "\n"}, "dry_run": True}).json["dry_run"]
        assert client.post("/api/runs/d/sounding", {"sounding": {"text": "1000.0 300.0 14.0\n"},
                                                    "dry_run": True}).json["dry_run"]
        assert (sorted(path.relative_to(uploads) for path in uploads.rglob("*")) if uploads.exists() else []) == \
            staged_before
        assert not (tmp_path / "runs" / "d2").exists()
        # The mode's dry run stores nothing (the import stored advanced once, as a first namelist import does).
        stored = (tmp_path / "config" / "gui.json").read_text()
        mode = client.post("/api/mode", {"mode": "guided", "dry_run": True}).json
        assert mode["command"] == "gpuwm-cm1 gui --mode guided"
        assert (tmp_path / "config" / "gui.json").read_text() == stored
        assert client.get("/api/mode").json["mode"] == "advanced"
    finally:
        stop_server(server)


def test_an_upload_is_staged_by_its_content_so_the_shown_command_is_the_one_that_runs(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        body = {"name": "u", "namelist": {"text": SUPERCELL.read_text(encoding="utf-8")}}
        shown = client.post("/api/import", {**body, "dry_run": True}).json["command"]
        ran = client.post("/api/import", body).json["command"]
        assert shown == ran
        staged = Path(shlex.split(ran)[2])
        assert staged.read_bytes() == SUPERCELL.read_bytes()
        assert staged.parent.name == hashlib.sha256(SUPERCELL.read_bytes()).hexdigest()[:24]
    finally:
        stop_server(server)


# ------------------------------------------------------------------ 2. the mode


def test_the_mode_survives_a_restart_and_a_second_browser_and_toggling_changes_no_run_directory(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    root = tmp_path / "runs"
    try:
        import_supercell(client, "one")
        import_supercell(client, "two", folder="study")
        assert client.post("/api/runs/one/start").json["ok"]
        wait_for(lambda: client.get("/api/runs/one/status").json["state"] == "finished", timeout=60)
        wait_for(lambda: not server.jobs.launched("one"), timeout=60)
        before = _digest(root)
        for mode in ("guided", "advanced", "guided", "advanced"):
            reply = client.post("/api/mode", {"mode": mode}).json
            assert reply["mode"] == mode and reply["command"] == f"gpuwm-cm1 gui --mode {mode}"
            # Every page read the mode drives changes nothing either.
            for path in ("/api/runs", "/api/runs/one/results", "/api/runs/one/setup", "/api/runs/one/words",
                         "/api/runs/one/restarts", "/api/runs/one/commands", "/api/runs/study%2Ftwo/setup"):
                assert client.get(path).status == 200, path
        assert _digest(root) == before
        second = Client(server.port, server.token)
        assert second.get("/api/mode").json == {"mode": "advanced", "source": "stored", "modes": ["guided", "advanced"]}
    finally:
        stop_server(server)
    assert json.loads((tmp_path / "config" / "gui.json").read_text())["mode"] == "advanced"
    server, client = start_server(root, tmp_path / "config")
    try:
        assert client.get("/api/session").json["mode"] == "advanced"
        assert _digest(root) == before
    finally:
        stop_server(server)


# ------------------------------------------------------------------ 3. the namelist editor


def test_editing_the_namelist_imports_it_again_and_returns_the_six_category_report(tmp_path, monkeypatch):
    server, client = _mixed(tmp_path, monkeypatch)
    root = Path(server.root.path)
    try:
        import_supercell(client, "e")
        rundir = root / "e"
        text = (rundir / "namelist.input").read_text()
        edited = text.replace(" stretch_x =      0,", " stretch_x =      1,")
        assert edited != text
        reply = client.post("/api/runs/e/namelist", {"text": edited}).json
        assert reply["command"].startswith(f"gpuwm-cm1 edit {rundir} --from ")
        report = reply["report"]
        assert [section["id"] for section in report["sections"]] == [
            "refused", "not_read", "differs", "not_checked", "not_needed", "same"]
        assert report["exit"] == 2 and report["counts"]["refused"] >= 1
        refused = report["sections"][0]["rows"]
        assert any("stretch_x" in (row.get("keys") or [row["key"]]) and row.get("fix") for row in refused)
        # The diff against the user's own file is stored in the run folder, and the reply carries it.
        stored = (rundir / "namelist.diff").read_text()
        assert "-" in stored and "stretch_x =      1" in stored and reply["stored_diff"] == stored
        assert (rundir / "namelist.input.original").read_text() == text
        assert json.loads((rundir / "import-report.json").read_text())["exit"] == 2
        # Word by word: every CM1 r21.1 word, with the report's category beside it, and one --set per change.
        words = client.get("/api/runs/e/words").json
        assert words["total"] == 583 and words["editable"]
        flat = {word["key"]: word for group in words["groups"] for word in group["words"]}
        assert flat["stretch_x"]["value"] == "1" and flat["stretch_x"]["category"] == "refused"
        assert flat["ptype"]["category"] == "same" and flat["ptype"]["in_file"]
        missing = [word for word in flat.values() if not word["in_file"]]
        assert missing and all(word["default"] is not None for word in missing)
        # How r21.1 comes by each value (records/CM1-R21-NAMELIST-DEFAULTS.json): dx has no value in its code
        # (it reads as 0 with gfortran and ifort); alphahl's is its declaration's, module_mp_nssl_2mom.F:747.
        assert flat["dx"]["basis"] == "static-zero" and flat["dx"]["default"] == "0.0"
        assert flat["alphahl"]["basis"] == "declaration-initializer" and flat["alphahl"]["default"] == "1.0"
        assert flat["alphahl"]["source"] == "src/module_mp_nssl_2mom.F:747"
        fixed = client.post("/api/runs/e/edit", {"set": {"stretch_x": "0"}}).json
        assert fixed["ok"] and fixed["command"] == f"gpuwm-cm1 edit {rundir} --set stretch_x=0 --json"
        assert fixed["report"]["exit"] == 0 and fixed["report"]["counts"]["refused"] == 0
        # Back to the user's file: the command names the kept original, and the diff is empty again.
        back = client.post("/api/runs/e/namelist", {"revert": True}).json
        assert back["ok"] and (rundir / "namelist.input").read_text() == text and not back["stored_diff"]
    finally:
        stop_server(server)


def test_the_namelist_reader_keeps_each_value_as_written():
    from cm1core.gui.advanced import read_commands_log, read_words

    order, groups = read_words(" &param1\n dx = 1000.0, dy=2000.0 ! comment\n timax = 7200.0,\n /\n"
                               " &param9\n output_format = 1, output_filetype=2 /\n &nope\n x = 'a/b', y = 3\n /\n")
    assert order == ["param1", "param9", "nope"]
    assert groups["param1"] == {"dx": "1000.0", "dy": "2000.0", "timax": "7200.0"}
    assert groups["param9"] == {"output_format": "1", "output_filetype": "2"}
    assert groups["nope"] == {"x": "'a/b'", "y": "3"}
    entries = read_commands_log("# 2026-09-24T01:02:03Z start: the run\nnice -n 19 env A\ngpuwm-cm1 run R\nwait\n"
                                "# 2026-09-24T01:03:03Z stop\ngpuwm-cm1 stop R\n")
    assert [(e["action"], e["note"], e["lines"]) for e in entries] == [
        ("start", "the run", ["nice -n 19 env A", "gpuwm-cm1 run R", "wait"]), ("stop", "", ["gpuwm-cm1 stop R"])]


# ------------------------------------------------------------------ 4. a guided run in advanced mode


def test_a_guided_run_opened_in_advanced_shows_its_fit_namelist_and_report_and_continues(tmp_path, monkeypatch):
    server, client = _mixed(tmp_path, monkeypatch)
    root = Path(server.root.path)
    try:
        assert client.get("/api/mode").json["mode"] == "guided"
        started = client.post("/api/guided/start", {"preset": "supercell-1km", "budget": "10m", "name": "g"}).json
        assert started["ok"], started
        rundir = root / "g"
        wait_for(lambda: client.get("/api/runs/g/status").json["state"] == "finished", timeout=120)
        wait_for(lambda: not server.jobs.launched("g"), timeout=60)
        assert client.post("/api/mode", {"mode": "advanced"}).json["mode"] == "advanced"
        setup = client.get("/api/runs/g/setup").json
        assert setup["mode"] == "advanced"
        fit = setup["fit"]
        assert fit["preset"] == "supercell-1km" and fit["budget"] == "10m" and "sentences" in fit
        assert json.loads((rundir / "fit.json").read_text())["preset"] == "supercell-1km"
        nml = client.get("/api/runs/g/namelist").json
        assert nml["text"] == (rundir / "namelist.input").read_text() and not nml["editable"]
        report = client.get("/api/runs/g/report").json["report"]
        assert report["exit"] == 0 and len(report["sections"]) == 6
        words = client.get("/api/runs/g/words").json
        assert not words["editable"] and words["why_fixed"]
        # Continue or extend: from the run's one restart (its end), an hour longer.
        restarts = client.get("/api/runs/g/restarts").json
        assert restarts["can_resume"] and restarts["restarts"][0]["newest"]
        chosen = restarts["restarts"][0]
        until = [option["until_s"] for option in restarts["until_options"] if option["label"] == "60 minutes longer"][0]
        dry = client.post("/api/runs/g/resume", {"from_restart": chosen["number"], "until_s": until,
                                                 "dry_run": True}).json
        resumed = client.post("/api/runs/g/resume", {"from_restart": chosen["number"], "until_s": until}).json
        assert resumed["ok"] and resumed["command"] == dry["command"] == C.display(
            C.resume_args(rundir, chosen["number"], until))
        wait_for(lambda: (lambda s: s["state"] == "finished" and s["segment"] == "002")(
            client.get("/api/runs/g/status").json), timeout=120)
        status = client.get("/api/runs/g/status").json
        assert status["model_time_s"] == until
        assert resumed["command"] in (rundir / "commands.log").read_text()
        # The job script continues it on a cluster to the same time.
        job = client.post("/api/runs/g/jobscript", {"scheduler": "slurm", "until_s": until + 1800}).json
        assert job["ok"] and "--continue" in job["script"] and job["command"].endswith(f"--until {until + 1800:g} --json")
        wait_for(lambda: not server.jobs.launched("g"), timeout=60)
    finally:
        stop_server(server)


def test_start_from_a_preset_makes_a_fitted_run_folder_that_opens_in_the_editor(tmp_path, monkeypatch):
    server, client = _mixed(tmp_path, monkeypatch)
    root = Path(server.root.path)
    try:
        client.post("/api/mode", {"mode": "advanced"})
        made = client.post("/api/presets/new", {"preset": "supercell-250m", "budget": "10m", "name": "p"}).json
        assert made["ok"] and made["url"] == "p", made
        assert made["command"] == C.display(C.preset_new_args("supercell-250m", "10m", root, "p"))
        rundir = root / "p"
        assert (rundir / "fit.json").is_file() and not (rundir / "segments").exists()
        setup = client.get("/api/runs/p/setup").json
        assert setup["editable"] and setup["fit"]["preset"] == "supercell-250m"
        nml = client.get("/api/runs/p/namelist").json
        assert nml["editable"] and nml["original"] is not None
        again = client.post("/api/presets/new", {"preset": "supercell-250m", "budget": "10m", "name": "p"})
        assert again.status == 409 and again.json["command"].startswith("gpuwm-cm1 preset new")
        assert "preset-new" in (rundir / "commands.log").read_text()
        assert client.get("/api/mode").json["mode"] == "advanced"
    finally:
        stop_server(server)


# ------------------------------------------------------------------ 5. reachability, and one set of features


_REGISTER = re.compile(r'register\("([a-z-]+)"')
#: Section 6's ten screens, as the router names them.
SECTION_6 = ("runs", "new-run", "check", "sounding", "domain", "settings", "estimate", "run", "results", "system")
#: Built with the Run, Results, Runs and System screens, alongside this package.
RUN_SCREENS = ("results", "system")


def _pages() -> dict[str, str]:
    return {path.relative_to(STATIC).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted(STATIC.rglob("*.js")) if "mock" not in path.parts}


def test_every_section_6_screen_is_registered_and_every_guided_screen_is_linked_from_advanced():
    pages = _pages()
    registered = {name for text in pages.values() for name in _REGISTER.findall(text)}
    missing = [name for name in SECTION_6 if name not in registered]
    assert [name for name in missing if name not in RUN_SCREENS] == []
    router = pages["js/router.js"]
    # Advanced mode's rail: the run pages, including Continue and Commands.
    for name in ("check", "namelist", "sounding", "domain", "settings", "estimate", "run", "results", "continue",
                 "commands"):
        assert f'name: "{name}"' in router, name
    advanced = "\n".join(text for path, text in pages.items() if path.startswith(("js/advanced/", "js/app.js",
                                                                                   "js/setup/newrun.js")))
    # Every guided screen has a link from an advanced page: the gallery (and from it the budget screen),
    # Watching and Results.
    for route in ("#/guided\"", "#/guided/watch/", "#/guided/results/"):
        assert route in advanced, route
    assert "guided/fit/" in pages["guided/gallery.js"] or "fit/" in pages["guided/gallery.js"]
    if missing:
        pytest.xfail(f"{', '.join(missing)} not built yet (Run, Results, Runs and System screens); "
                     "this check covers them once they are")


def test_guided_pages_call_nothing_advanced_mode_cannot():
    """One set of features: each action a guided page takes is one an advanced page takes too, except
    guided Start, which is Start from a preset (preset new) and Start (run) in one press."""

    pages = _pages()
    table = {endpoint.path.replace("/api/runs/RUN/", ""): endpoint.action for endpoint in C.API
             if endpoint.method == "POST"}
    run_action = re.compile(r'\$\{api\.runPath\([^)]*\)\}/([a-z-]+)')
    root_action = re.compile(r'"(/api/[a-z/-]+)"')
    # A Run screen's buttons named by their action: act("stop", ...), actionButton(id, "stop", ...).
    named = re.compile(r'\bact\("([a-z-]+)"|actionButton\(\s*[\w.]+\s*,\s*"([a-z-]+)"')

    def actions(texts: list[str]) -> set[str]:
        found = set()
        for text in texts:
            names = run_action.findall(text) + root_action.findall(text)
            names += [a or b for a, b in named.findall(text)]
            for name in names:
                if name in table:
                    found.add(table[name])
        return found

    guided = actions([text for path, text in pages.items() if path.startswith("guided/")])
    advanced = actions([text for path, text in pages.items() if not path.startswith("guided/")])
    # The gallery and the budget screen's actions; Watching's Stop and Results' Run it longer are the guided pages'
    # own calls or the shared Run and Results screens' (the same actions, drawn for guided mode).
    assert {"fit", "guided-start"} <= guided, guided
    only_guided = guided - advanced - {"guided-start"}
    assert only_guided == set(), only_guided
    # Guided Start is Start from a preset and Start, which advanced mode has as two presses.
    assert {"preset-new", "start", "fit", "resume", "stop", "pause", "force-stop", "jobscript", "edit",
            "namelist"} <= advanced, advanced


def test_advanced_pages_keep_the_word_rules():
    """Every button, tooltip and error in both modes is plain and factual (no marketing words, no tornado
    wording); advanced mode shows CM1's own names, so they are allowed here."""

    from cm1core.gui.copy_lint import js_strings, lint_text

    files = sorted((STATIC / "js" / "advanced").glob("*.js")) + [STATIC / "js" / "command.js"]
    problems = [problem for path in files for text in js_strings(path.read_text(encoding="utf-8"))
                for problem in lint_text(text, where=path.name, allow_cm1=True)]
    source = (ROOT / "src/cm1core/gui/advanced.py").read_text(encoding="utf-8")
    for text in re.findall(r'"([^"\n]{12,})"', source):
        problems += lint_text(text, where="advanced.py", allow_cm1=True)
    assert problems == []
