"""Where the GUI release's packages meet: import, run directory, estimate, run admission and cm1out.

Each package was verified on its own branch; these hold the contracts
between them on the merged tree:

- ``gpuwm-cm1 import`` writes the run directory ``gpuwm-cm1 run`` and
  ``status`` and the GUI server read;
- ``gpuwm-cm1 run`` admits a run against the GPU's free memory with the
  estimator's own model, so the Estimate screen and Start agree;
- ``gpuwm-cm1 cm1out`` converts the frames of the run's lineage,
  not those of a branch the run has left.

CPU only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from cm1import_support import SUPERCELL, namelist_tool


@pytest.fixture()
def parser_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


# ------------------------------------------------ import -> estimate -> run admission


def test_start_admits_a_run_with_the_numbers_estimate_shows(tmp_path: Path, parser_tool) -> None:
    # The host plan is the real door's `cartesian --plan --integrate`, which admits only the
    # audited ArWen (gpuwm) install; without gpuwm it refuses before pricing.
    pytest.importorskip("gpuwm")
    from cm1core.cli import main
    from cm1core.cm1run.estimate import price
    from cm1core.cm1run.launch import StartRequest, estimate_needs, host_plan, plan_segment
    from cm1core.cm1run.rundir import RunDirectory
    from cm1core.cm1run.status import run_status
    import tomllib

    rundir = tmp_path / "stock"
    assert main(["import", os.fspath(SUPERCELL), "--out", os.fspath(rundir)]) == 0
    run = RunDirectory(rundir)
    assert run_status(run).state == "ready"
    record = run.load()
    plan = plan_segment(run, record, StartRequest())
    payload = host_plan(run, record, run.deck, StartRequest())
    needs = estimate_needs(tomllib.loads(run.deck.read_text(encoding="utf-8")), payload, plan)
    estimate = price(rundir, None)
    assert needs.gpu_bytes == estimate.gpu_need_bytes
    assert needs.frame_bytes == estimate.frame_bytes
    assert plan.frames_expected == estimate.frames and plan.restarts_expected == estimate.restarts
    # The stock deck: 0.58 M cells, about 1.1 GB (records/GM1-GPU-RATES.json's memory model, every
    # measured run within 7.7 %); section 8's first fit (1.2 GB + 461 B/cell) said 1.47 GB.
    assert 1.0e9 < needs.gpu_bytes < 1.25e9


# ------------------------------------------------ cm1out follows the run's lineage


def _at(carriers: dict, frame: dict, t: float, value: float) -> tuple[dict, dict]:
    return (dict(carriers, w3d=np.full_like(carriers["w3d"], np.float32(value))),
            dict(frame, time_s=t, step=int(t / 6), domainloc=(12.5 * t, 3.0 * t)))


def _run_directory(tmp_path: Path) -> Path:
    from cm1core.cm1run.rundir import RUN_SCHEMA, write_json_atomic
    from cm1out_support import REF

    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(REF / "namelist.input", run / "namelist.input")
    write_json_atomic(run / "run.json", {"schema": RUN_SCHEMA, "name": "run", "segments": []})
    return run


def _segment(run: Path, number: int, frames: list[tuple[int, float, float]], *,
             resumed_from: tuple[int, float] | None, row: bool = True) -> None:
    from cm1core.cm1run.rundir import RunDirectory
    from cm1out_support import REF, frame_from_cm1, publish

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    root = run / "segments" / f"{number:03d}" / "frames"
    root.mkdir(parents=True)
    for index, t, value in frames:
        c, f = _at(carriers, frame, t, value)
        publish(root, index, c, **f)
    if not row:
        return

    def add(record: dict) -> None:
        record["segments"].append({
            "number": number, "directory": f"segments/{number:03d}",
            "resumed_from": None if resumed_from is None else {
                "segment": resumed_from[0], "model_time_s": resumed_from[1], "restart": 1},
        })

    RunDirectory(run).update(add)


def test_a_resume_from_an_earlier_restart_drops_the_left_branchs_files(tmp_path: Path, parser_tool) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    from cm1core.cm1out import convert

    run = _run_directory(tmp_path)
    _segment(run, 1, [(0, 0.0, 1.0), (1, 60.0, 1.0), (2, 120.0, 1.0), (3, 180.0, 1.0)], resumed_from=None)
    convert(run)
    assert sorted(p.name for p in (run / "cm1out").glob("cm1out_0*.nc")) == [
        "cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc", "cm1out_000004.nc"]
    # Resumed from the 60 s restart and stopped after its 120 s frame: 180 s is
    # the old branch's, and the run no longer has it.
    _segment(run, 2, [(2, 120.0, 2.0)], resumed_from=(1, 60.0))
    summary = convert(run)
    assert summary.frames_converted == 1 and summary.frames_total == 3
    assert sorted(p.name for p in (run / "cm1out").glob("cm1out_0*.nc")) == [
        "cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc"]
    for name, expected in (("cm1out_000001.nc", 1.0), ("cm1out_000002.nc", 1.0), ("cm1out_000003.nc", 2.0)):
        with netCDF4.Dataset(run / "cm1out" / name) as ds:
            assert float(ds["w"][0].max()) == expected, name
    with netCDF4.Dataset(run / "cm1out" / "cm1out_stats.nc") as ds:
        assert list(ds["time"][:]) == [0.0, 60.0, 120.0]
        assert list(ds["wmax"][:]) == [1.0, 1.0, 2.0]
    assert convert(run).frames_converted == 0


def test_a_start_over_writes_one_file_cm1out_nc_again(tmp_path: Path, parser_tool) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    from cm1core.cm1out import convert

    run = _run_directory(tmp_path)
    _segment(run, 1, [(0, 0.0, 1.0), (1, 60.0, 1.0), (2, 120.0, 1.0)], resumed_from=None)
    convert(run, filetype=1, stats=False)
    _segment(run, 2, [(0, 0.0, 3.0), (1, 60.0, 3.0)], resumed_from=None)
    convert(run, filetype=1, stats=False)
    with netCDF4.Dataset(run / "cm1out" / "cm1out.nc") as ds:
        assert list(ds["time"][:]) == [0.0, 60.0]
        assert [float(ds["w"][i].max()) for i in range(2)] == [3.0, 3.0]


def test_a_segment_not_yet_recorded_leaves_the_output_alone(tmp_path: Path, parser_tool) -> None:
    pytest.importorskip("netCDF4")
    from cm1core.cm1out import convert

    run = _run_directory(tmp_path)
    _segment(run, 1, [(0, 0.0, 1.0), (1, 60.0, 1.0)], resumed_from=None)
    convert(run)
    # `gpuwm-cm1 run` has made segments/002 but not yet written its run.json row.
    (run / "segments" / "002").mkdir()
    summary = convert(run)
    assert summary.frames_converted == 0
    assert sorted(p.name for p in (run / "cm1out").glob("cm1out_0*.nc")) == [
        "cm1out_000001.nc", "cm1out_000002.nc"]
    state = json.loads((run / "cm1out" / ".cm1out-state.json").read_text(encoding="utf-8"))
    assert sorted(state["frames"]) == ["0.0", "60.0"]


# ------------------------------------------------ the GUI server with the real gpuwm-cm1


def test_the_server_imports_and_estimates_with_the_real_commands(tmp_path: Path, parser_tool,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Import and Estimate through the GUI server run the real `gpuwm-cm1 import` and `estimate` (no GPU).

    Before the release merge the real runner had neither command and answered
    501; now the run the import makes is listed, its report read and its
    estimate written into the run directory.
    """

    from cm1core.gui.jobs import real_runner
    from gui_support import start_server, stop_server

    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", "none")
    root = tmp_path / "runs"
    server, client = start_server(root, tmp_path / "config", runner=real_runner())
    try:
        assert {"import", "estimate", "run", "resume", "cm1out", "status"} <= set(
            client.get("/api/session").json["commands_available"])
        answer = client.post("/api/import", {"name": "stock", "namelist": {"path": os.fspath(SUPERCELL)}})
        assert answer.status == 200, answer.raw
        reply = answer.json
        assert reply["ok"] and reply["verdict"] == "runnable", reply
        assert reply["command"].startswith("gpuwm-cm1 import ") and reply["command"].endswith(" --json")
        rundir = root / "stock"
        assert (rundir / "run.json").is_file() and (rundir / "deck.toml").is_file()
        listed = {row["id"]: row for row in client.get("/api/runs").json["runs"]}
        assert listed["stock"]["status"]["import_exit_code"] == 0
        assert client.get("/api/runs/stock/report").json["report"]["verdict"] == "runs"
        estimate = client.post("/api/runs/stock/estimate").json
        assert estimate["ok"], estimate
        assert (rundir / "estimate.json").is_file() and (rundir / "geometry.json").is_file()
        status = client.get("/api/runs/stock/status").json
        assert status["state"] == "ready" and "start" in status["actions"]
    finally:
        stop_server(server)


def test_a_resumes_sidecars_start_once_its_own_segment_exists(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """`cm1out --follow` ends with the newest segment's end record, so a sidecar
    launched beside a resume before the resume has made its segment would find the
    previous segment's end and exit at once (seen on the RTX 3080 merge smoke: the
    netCDF sidecar converted nothing of segments 002 and 003).  The server starts a
    resume's sidecars once the engine's own segment exists, as the job script does."""

    from gui_support import SUPERCELL as GUI_SUPERCELL, start_server, stop_server, wait_for

    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.004")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    root = tmp_path / "runs"
    server, client = start_server(root, tmp_path / "config")
    try:
        text = GUI_SUPERCELL.read_text().replace(" rstfrq = -3600.0,", " rstfrq =  1800.0,")
        assert client.post("/api/import", {"name": "p", "namelist": {"text": text}}).json["ok"]
        started = client.post("/api/runs/p/start").json
        assert started["ok"] and len(started["sidecars"]) == 2
        # A first start launches its sidecars at once: `--follow` waits for a run with no segment yet.
        assert sorted(row.role for row in server.jobs.launched("p", alive_only=False)) == [
            "cm1out", "quicklook", "run"]
        client.events("p", lambda e: e.get("event") == "progress" and e["data"].get("step"), timeout=30)
        assert client.post("/api/runs/p/pause").json["ok"]
        client.events("p", lambda e: e.get("event") == "end", timeout=30)
        wait_for(lambda: client.get("/api/runs/p/status").json["state"] == "paused")
        wait_for(lambda: not server.jobs.launched("p"), timeout=30)

        monkeypatch.setenv("GM1_FAKE_LAUNCH_DELAY_S", "1.0")  # the resume takes a second to make its segment
        resumed = client.post("/api/runs/p/resume").json
        assert resumed["ok"] and len(resumed["sidecars"]) == 2
        assert [row.role for row in server.jobs.launched("p")] == ["resume"]
        assert not (root / "p" / "segments" / "002").exists()
        wait_for(lambda: (root / "p" / "segments" / "002").is_dir(), timeout=30)
        # Both sidecars are launched one after the other: wait for the pair, not the first of them (under
        # load the first answer held only cm1out).
        sidecars = wait_for(lambda: (lambda rows: rows if len(rows) >= 2 else None)(
            [row for row in server.jobs.launched("p", alive_only=False)
             if row.role in ("cm1out", "quicklook") and row.segments_before == 1]), timeout=10)
        assert sorted(row.role for row in sidecars) == ["cm1out", "quicklook"]
        client.events("p", lambda e: e.get("event") == "end" and e["data"]["segment"] == "002", timeout=60)
    finally:
        stop_server(server)


def test_a_run_stopped_before_its_first_restart_still_offers_continue(tmp_path: Path) -> None:
    """Continue after a Stop that came before the resumed segment's first restart: `gpuwm-cm1
    resume` continues from the earlier segment's restart on the run's lineage, so the GUI's
    status offers it (seen on the RTX 3080 merge smoke, where segment 002 stopped at 930 s)."""

    from cm1core.cm1run.rundir import RUN_SCHEMA, write_json_atomic
    from cm1core.gui.runstate import read_status

    run = tmp_path / "run"
    restart = run / "segments" / "001" / "restarts" / "restart-000001"
    restart.mkdir(parents=True)
    write_json_atomic(restart / "restart-publication.json", {"model_time_s": 600.0, "step": 100})
    (run / "segments" / "002").mkdir(parents=True)
    for number, reason in ((1, "completed"), (2, "stopped")):
        records = [{"schema": "gm1.run-events/v1", "seq": 0, "kind": "admitted", "time_utc": "2026-09-23T21:00:00Z"},
                   {"schema": "gm1.run-events/v1", "seq": 1, "kind": "end", "reason": reason, "rc": 0,
                    "time_utc": "2026-09-23T21:01:00Z"}]
        if number == 1:
            records.insert(1, {"schema": "gm1.run-events/v1", "seq": 1, "kind": "restart", "number": 1,
                               "model_time_s": 600.0, "time_utc": "2026-09-23T21:00:30Z"})
            records[-1]["seq"] = 2
        (run / "segments" / f"{number:03d}" / "events.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    write_json_atomic(run / "run.json", {"schema": RUN_SCHEMA, "name": "run", "segments": [
        {"number": 1, "directory": "segments/001", "resumed_from": None},
        {"number": 2, "directory": "segments/002",
         "resumed_from": {"segment": 1, "restart": 1, "model_time_s": 600.0}},
    ]})
    status = read_status(run)
    assert status.state == "stopped" and status.detail["last_restart"] is None
    assert status.detail["restartable"] is True and "resume" in status.detail["actions"]


# ------------------------------------------------ the Run and Results buttons and their Show command


def _action_button_calls(source: str) -> list[tuple[str, str]]:
    """(action, call text) for each actionButton(id, "action", ...) call, strings skipped when matching parens."""

    import re

    calls = []
    for match in re.finditer(r'\bactionButton\(\s*[\w.]+\s*,\s*"([a-z-]+)"', source):
        depth, quote, i = 0, None, match.start() + len("actionButton")
        while i < len(source):
            char = source[i]
            if quote:
                if char == "\\":
                    i += 1
                elif char == quote:
                    quote = None
            elif char in "\"'`":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        calls.append((match.group(1), source[match.start():i + 1]))
    return calls


def test_run_and_results_buttons_whose_line_depends_on_inputs_show_the_exact_line_before_the_press() -> None:
    """Advanced mode: each GUI action maps to a CLI command, shown before the press. The Continue (restart,
    end time) and Convert (format) panels build their request from inputs, so their Show command is the live one (a
    dry run that follows the inputs), not the default line."""

    from gui_support import ROOT

    js = ROOT / "src" / "cm1core" / "gui" / "static" / "js" / "run"
    parts = (js / "parts.js").read_text(encoding="utf-8")
    button = parts[parts.index("export function actionButton("):parts.index("// ---- the picture viewer")]
    assert "if (opts.watch)" in button and "liveCommand(" in button and "dryRun" not in button
    dependent = []
    for name in ("run.js", "results.js"):
        for action, call in _action_button_calls((js / name).read_text(encoding="utf-8")):
            if "body:" in call:
                assert "watch: [" in call, (name, action)
                dependent.append(action)
    # Make the film (look, format, pictures per second, blended pictures) follows its inputs too
    assert sorted(dependent) == ["convert", "film", "resume", "resume"], dependent
    results = (js / "results.js").read_text(encoding="utf-8")
    assert 'runRoute("continue", id)' in results  # the Continue panel links the Continue or extend page


def test_a_dry_run_of_continue_and_convert_gives_the_chosen_line_and_changes_nothing(
        tmp_path: Path, parser_tool, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import shlex

    from gui_support import ROOT, start_server, stop_server, wait_for

    gpuinfo = ROOT / "tests" / "data" / "gpuinfo"
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", os.fspath(gpuinfo / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", os.fspath(gpuinfo / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setenv("GPUWM_CM1_CACHE_DIR", os.fspath(tmp_path / "cache"))
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        text = SUPERCELL.read_text(encoding="utf-8").replace(" rstfrq = -3600.0,", " rstfrq =  1800.0,")
        assert client.post("/api/import", {"name": "r", "namelist": {"text": text}}).json["ok"]
        assert client.post("/api/runs/r/start").json["ok"]
        wait_for(lambda: client.get("/api/runs/r/status").json["state"] == "finished", timeout=120)
        rundir = tmp_path / "runs" / "r"
        wait_for(lambda: "cm1out_stats.nc" in {row["name"] for row in client.get("/api/runs/r/results").json["cm1out"]},
                 timeout=60)
        # the sidecars (cm1out, quick-looks) end after the run's end record; hash once they have exited
        wait_for(lambda: not server.jobs.launched("r"), timeout=60)
        known = client.get("/api/runs/r/results").json["commands"]
        restart = client.get("/api/runs/r/restarts").json["restarts"][-1]["number"]

        def tree() -> dict[str, str]:
            return {os.fspath(p.relative_to(rundir)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(rundir.rglob("*")) if p.is_file()}

        launched = len(server.jobs.launched(alive_only=False))
        before = tree()
        resume = client.post("/api/runs/r/resume", {"from_restart": restart, "until_s": 9000.0, "dry_run": True}).json
        assert resume["dry_run"] is True and resume["action"] == "resume"
        assert resume["command"].startswith(f"gpuwm-cm1 resume {shlex.quote(os.fspath(rundir))} ")
        assert f" --from-restart {restart} --until 9000" in resume["command"]
        assert resume["command"] != known["resume"]
        convert = client.post("/api/runs/r/convert", {"format": "grads", "dry_run": True}).json
        assert convert["dry_run"] is True and convert["command"] == known["convert"] + " --format grads"
        assert tree() == before and len(server.jobs.launched(alive_only=False)) == launched
    finally:
        stop_server(server)
