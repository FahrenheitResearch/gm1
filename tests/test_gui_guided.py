"""Guided mode through the GUI server, on the fake runner: the gallery, the fit, one Start, Watching,
Stop and Results, and the replay: the server's commands.log, replayed from a shell, makes the same run
directory.

The fit and the run folder are made by the real ``gpuwm-cm1 preset`` code on a fake GPU (the WSL2 RTX 3080 fixture:
4.8 GB in use, no process list); the run itself is the fake runner's.  CPU only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.gui import guided as G
from gui_support import start_server, stop_server, wait_for
from cm1import_support import namelist_tool

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "gpuinfo"


@pytest.fixture(autouse=True)
def _fake_gpu(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(FIXTURES / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(FIXTURES / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setenv("GPUWM_CM1_HOST_FIXTURE", str(FIXTURES / "host-64gb-1tb.json"))
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.004")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.2")
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


@pytest.fixture()
def live(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        yield server, client, Path(server.root.path)
    finally:
        stop_server(server)


def test_a_first_launch_opens_guided_and_the_gallery_lists_every_preset(live):
    server, client, root = live
    assert client.get("/api/mode").json["mode"] == "guided"
    copy = client.get("/api/copy").json
    assert copy["guided"]["strings"]["gallery.title"]["text"] == "Pick a storm"
    # The four favourites, then a hurricane run's own two (its wind and its pressure).
    assert [look["id"] for look in copy["looks"]["looks"]] == ["reflectivity", "spin-near-ground",
                                                                 "updraft-5km", "swath", "wind-lowest",
                                                                 "pressure-lowest"]
    listed = client.get("/api/presets")
    assert listed.status == 200
    body = listed.json
    assert body["command"] == "gpuwm-cm1 preset list --json"
    rows = {row["id"]: row for row in body["presets"]}
    assert list(rows) == [preset.id for preset in P.load_all()]
    assert rows["supercell-1km"]["offered"] is True
    for row in rows.values():
        picture = row["picture"]
        if picture["available"]:
            assert picture["url"] == f"/api/presets/{picture['preset']}/picture"
            got = client.get(picture["url"])
            assert got.status == 200 and got.headers["content-type"] == "image/png"
            assert got.raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get("/api/presets/../picture").status in (400, 404)
    assert client.get("/api/presets/no-such/picture").status == 404


def test_fit_returns_its_command_and_what_start_would_run(live):
    server, client, root = live
    reply = client.post("/api/fit", {"preset": "supercell-250m", "budget": "10m"})
    assert reply.status == 200, reply.raw
    body = reply.json
    assert body["ok"] and body["offered"] and body["action"] == "fit"
    assert body["command"] == f"gpuwm-cm1 preset fit supercell-250m --budget 10m --root {root} --json"
    fit = body["fit"]
    assert fit["facts"]["dx_m"] == 500.0 and fit["summary"].startswith("A 120 × 120 km area at 500 m")
    name = body["start"]["name"]
    assert body["start"]["commands"] == [
        f"gpuwm-cm1 preset new supercell-250m --budget 10m --root {root} --name {name} --json",
        f"gpuwm-cm1 run {root / name}",
    ]
    not_here = client.post("/api/fit", {"preset": "supercell-100m", "budget": "overnight"}).json
    assert not_here["ok"] and not not_here["offered"]
    # Offered again (earlier versions withheld it); this 10 GB card under a Windows desktop has too little memory free.
    assert not_here["fit"]["reason_kind"] == "gpu" and "of free GPU memory" in not_here["message"], not_here
    assert client.post("/api/fit", {"preset": "supercell-1km", "budget": "2h"}).status == 400
    assert client.post("/api/fit", {"budget": "10m"}).status == 400
    unknown = client.post("/api/fit", {"preset": "no-such", "budget": "10m"})
    assert unknown.status == 400 and "no-such" in unknown.json["message"]


def test_customize_choices_go_to_the_fit_and_start_as_custom_flags(live):
    """The budget screen's Customize panel: {custom: {KNOB: VALUE}} becomes --custom KNOB=VALUE on the fit and on
    both lines Start would run; an unknown name is refused before any command runs."""

    server, client, root = live
    custom = {"grid": 500, "rotation": "on", "parcels": True}
    reply = client.post("/api/fit", {"preset": "supercell-1km", "budget": "1h", "custom": custom})
    assert reply.status == 200, reply.raw
    body = reply.json
    flags = "--custom grid=500 --custom rotation=on --custom parcels=on"
    assert body["command"] == f"gpuwm-cm1 preset fit supercell-1km --budget 1h {flags} --root {root} --json"
    assert body["offered"], body["message"]
    fit = body["fit"]
    assert [entry["knob"] for entry in fit["customized"]] == ["grid", "rotation", "parcels"]
    assert fit["pinned"] == ["grid"] and fit["facts"]["dx_m"] == 500.0
    name = body["start"]["name"]
    # Named with the spacing the run really has, not the preset's.
    assert name.startswith("supercell-500m-custom-")
    assert body["start"]["commands"][0] == (f"gpuwm-cm1 preset new supercell-1km --budget 1h --root {root} "
                                            f"--name {name} {flags} --json")
    # The cache tells a customized fit from the preset's own.
    plain = client.post("/api/fit", {"preset": "supercell-1km", "budget": "1h"}).json
    assert plain["fit"]["customized"] == [] and plain["fit"]["facts"]["dx_m"] == 1000.0
    again = client.post("/api/fit", {"preset": "supercell-1km", "budget": "1h", "custom": custom}).json
    assert again.get("cached") and again["fit"]["facts"]["dx_m"] == 500.0
    unknown = client.post("/api/fit", {"preset": "supercell-1km", "budget": "1h", "custom": {"colour": "red"}})
    assert unknown.status == 400 and unknown.json["message"] == "There is no setting called colour."
    for bad in ({"grid": [500]}, {"grid": None}, {"grid": "5 0 0"}, ["grid=500"]):
        assert client.post("/api/fit", {"preset": "supercell-1km", "budget": "1h", "custom": bad}).status == 400
    # Any grid spacing is a choice; the only ones turned down cannot be a grid.
    unoffered = client.post("/api/fit", {"preset": "supercell-1km", "budget": "1h", "custom": {"grid": 20000}})
    assert unoffered.status == 400 and "fewer than 8 cells across" in unoffered.json["message"]
    # The value the message names as taken comes with it, for the screen's one-tap button.
    assert unoffered.json["suggest"] == {"grid": 15000}
    thin = client.post("/api/fit", {"preset": "supercell-1km", "budget": "1h", "custom": {"levels": 50}})
    assert thin.status == 400 and thin.json["suggest"] == {"levels": 78.0}


def test_any_grid_spacing_is_fitted_through_the_api(live):
    """The Customize panel's slider and number box: {custom: {grid: 67}} is --custom grid=67, and the fit says the
    grid, the area and the cells exactly, the time step that lands on the pictures, and what to know."""

    server, client, root = live
    reply = client.post("/api/fit", {"preset": "supercell-1km", "budget": "1h", "custom": {"grid": 67}})
    assert reply.status == 200, reply.raw
    body = reply.json
    assert body["command"] == f"gpuwm-cm1 preset fit supercell-1km --budget 1h --custom grid=67 --root {root} --json"
    assert body["offered"], body["message"]
    fit = body["fit"]
    facts = fit["facts"]
    assert facts["dx_m"] == 67.0 and facts["dt_s"] == 0.390625      # a moving domain: a short binary fraction
    assert facts["exact"] == (
        f"67 m grid, {facts['nx'] * 0.067:.1f} × {facts['ny'] * 0.067:.1f} km, {facts['nx']} × {facts['ny']} cells")
    assert fit["customized"][0]["sentence"].startswith(
        "Grid spacing 67 m instead of 1 km (and a 0.390625 s time step")
    assert fit["pinned"] == ["grid"] and any("finer than any storm GM1 ships" in line for line in fit["warnings"])
    assert fit["estimate"]["largest_square_side_cells"] > 0
    # The time with no limit is one of the times.
    open_ended = client.post("/api/fit", {"preset": "supercell-1km", "budget": "any", "custom": {"grid": 67}})
    assert open_ended.status == 200 and open_ended.json["fit"]["budget_s"] is None
    assert open_ended.json["fit"]["budget_words"] == "as long as it takes"


def test_a_customized_start_makes_a_custom_run_and_advanced_makes_its_folder_without_starting(live):
    server, client, root = live
    custom = {"pictures": 10, "bumps": "on"}
    started = client.post("/api/guided/start", {"preset": "supercell-1km", "budget": "10m", "custom": custom})
    assert started.status == 200, started.raw
    name = started.json["url"]
    assert name.startswith("supercell-1km-custom-")
    assert "--custom pictures=10 --custom bumps=on" in started.json["commands"][0]
    rundir = root / name
    run = json.loads((rundir / "run.json").read_text())
    assert run["case"] == "Supercell, quick look, 1 km (customized)"
    assert json.loads((rundir / "fit.json").read_text())["facts"]["output_interval_s"] == 600.0
    wait_for(lambda: client.get(f"/api/runs/{name}/status").json["state"] in ("finished", "stopped"), timeout=120)
    results = client.get(f"/api/runs/{name}/results?mode=guided").json
    guided = results.get("guided") or results
    fit_block = guided.get("fit") or {}
    if fit_block:
        assert [entry["knob"] for entry in fit_block["customized"]] == ["pictures", "bumps"]
    made = client.post("/api/presets/new", {"preset": "supercell-1km", "budget": "10m", "custom": {"grid": 500},
                                            "environment": {"instability": 0.4, "shear": 0.6}})
    assert made.status == 200, made.raw
    folder = root / made.json["run"]
    assert made.json["run"].startswith("supercell-500m-custom-")
    assert "--instability 0.4 --shear 0.6 --hodograph curved --custom grid=500" in made.json["command"]
    assert (folder / "namelist.input").is_file() and (folder / "input_sounding").is_file()
    status = client.get(f"/api/runs/{made.json['run']}/status").json
    assert status["state"] not in ("running", "waiting")
    assert "isnd" in (folder / "namelist.diff").read_text()


def test_a_choice_at_the_presets_own_value_is_not_a_change_and_a_greyed_one_goes_to_advanced(live):
    server, client, root = live
    # Holding the preset's own grid pins it but changes nothing: the run is named and labelled as the preset.
    same = client.post("/api/fit", {"preset": "supercell-1km", "budget": "1h", "custom": {"grid": 1000}}).json
    assert same["offered"] and same["fit"]["customized"] == [] and same["fit"]["pinned"] == ["grid"]
    assert "-custom-" not in same["start"]["name"]
    # A choice the storm cannot take says so with a full stop, and the fix is the choice, not another storm.
    refused = client.post("/api/fit", {"preset": "bubble-cloud-250m", "budget": "1h", "custom": {"parcels": "on"}})
    assert refused.status == 400
    assert refused.json["message"].endswith(".") and "Follow air parcels" in refused.json["message"]
    assert refused.json["fix"] == "Change the setting it names, or press Reset to the preset."
    # "All settings (Advanced)" with a greyed choice: the folder is made, then that choice set in one edit.
    made = client.post("/api/presets/new", {"preset": "supercell-1km", "budget": "10m", "greyed": {"scheme": "wsm6"}})
    assert made.status == 200, made.raw
    folder = root / made.json["run"]
    assert "rain and hail by WSM6" in made.json["message"]
    assert "ptype" in (folder / "namelist.diff").read_text() or "ptype = 7" in (folder / "namelist.input").read_text()
    run = json.loads((folder / "run.json").read_text())
    assert run["case"].endswith("(edited)") and run["guided"]["edited"] is True
    bad = client.post("/api/presets/new", {"preset": "supercell-1km", "budget": "10m", "greyed": {"scheme": "kessler"}})
    assert bad.status == 400


def _session(client, root: Path) -> tuple[str, dict]:
    """Gallery -> fit -> Start -> (first frame) Stop -> end, as the guided pages drive it."""

    assert client.get("/api/presets").status == 200
    fit = client.post("/api/fit", {"preset": "supercell-1km", "budget": "10m"}).json
    name = fit["start"]["name"]
    started = client.post("/api/guided/start", {"preset": "supercell-1km", "budget": "10m", "name": name})
    assert started.status == 200, started.raw
    body = started.json
    assert body["ok"] and body["action"] == "guided-start" and body["url"] == name
    lines = body["command"].splitlines()
    assert lines == [f"gpuwm-cm1 preset new supercell-1km --budget 10m --root {root} --name {name} --json",
                     f"gpuwm-cm1 run {root / name}"]
    assert body["commands"][0] == lines[0] and body["commands"][-1] == lines[1]
    rundir = root / name
    assert (rundir / "fit.json").is_file() and (rundir / "namelist.input.original").is_file()
    wait_for(lambda: (client.get(f"/api/runs/{name}/status").json.get("last_frame") or 0) >= 1, timeout=60)
    stopped = client.post(f"/api/runs/{name}/stop").json
    assert stopped["ok"] and stopped["command"] == f"gpuwm-cm1 stop {rundir}"
    wait_for(lambda: client.get(f"/api/runs/{name}/status").json["state"] in ("stopped", "finished"), timeout=60)
    return name, body


def test_one_guided_session_start_stop_and_results(live, tmp_path):
    server, client, root = live
    name, started = _session(client, root)
    rundir = root / name
    # The first launch was guided because no run existed; now one does, so a reload (or the next launch) must
    # still open guided: the guided Start stored the mode.
    assert client.get("/api/mode").json == {**client.get("/api/mode").json, "mode": "guided", "source": "stored"}
    status = client.get(f"/api/runs/{name}/status").json
    # One restart at the end (the preset's rstfrq < 0), so a run stopped before it has none to continue from.
    assert status["state"] == "stopped" and "resume" not in status["actions"] and "results" in status["actions"]
    runs = client.get("/api/runs").json["runs"]
    assert runs[0]["case"] == "Supercell, quick look, 1 km"
    # Results in guided mode: the four looks first, and their panels filled with this run's numbers.
    wait_for(lambda: client.get(f"/api/runs/{name}/results?mode=guided").json["looks"], timeout=30)
    results = client.get(f"/api/runs/{name}/results?mode=guided").json
    assert [look["id"] for look in results["looks"][:4]] == ["reflectivity", "spin-near-ground", "updraft-5km", "swath"]
    assert all(look["favourite"] for look in results["looks"][:4])
    guided = results["guided"]
    spin = guided["looks"]["spin-near-ground"]["text"]
    assert "at the lowest model level (250 m)" in spin
    assert "At 1 km grid spacing, features smaller than about 6 km are not resolved." in spin
    assert guided["this_run"] == ("This run: 1 km grid spacing, a 120 × 120 km area, 2 h of storm time, "
                                  "a saved time every 5 min.")
    assert guided["case"] == "Supercell, quick look, 1 km" and guided["fit"]["budget"] == "10m"
    longer = guided["run_longer"]
    assert longer["command"] == f"gpuwm-cm1 resume {rundir} --until {int(longer['until_s'])}"
    assert results["commands"]["show_everything"] == "gpuwm-cm1 gui --mode advanced"
    advanced = client.get(f"/api/runs/{name}/results?mode=advanced").json
    assert "guided" not in advanced
    # The same run in Advanced: its fitted namelist, its report and fit.json are all there.
    detail = client.get(f"/api/runs/{name}").json
    assert detail["files"]["fit.json"] and detail["files"]["import-report.json"] and detail["files"]["commands.log"]
    log = (rundir / "commands.log").read_text()
    assert "preset-new" in log and "gpuwm-cm1 preset new supercell-1km" in log
    assert f"gpuwm-cm1 run {rundir}" in log and f"gpuwm-cm1 stop {rundir}" in log


def test_a_guided_start_never_overrides_a_mode_the_user_chose(live):
    server, client, root = live
    assert client.post("/api/mode", {"mode": "advanced"}).status == 200
    started = client.post("/api/guided/start", {"preset": "supercell-1km", "budget": "10m", "name": "chosen"})
    assert started.status == 200, started.raw
    assert client.get("/api/mode").json["mode"] == "advanced"
    wait_for(lambda: client.get("/api/runs/chosen/status").json["state"] in ("finished", "stopped"), timeout=120)


def test_guided_start_refuses_a_name_in_use_and_a_run_that_does_not_fit(live):
    server, client, root = live
    (root / "taken").mkdir()
    (root / "taken" / "run.json").write_text(json.dumps({"schema": "gm1.cm1-run/v1", "segments": []}))
    taken = client.post("/api/guided/start", {"preset": "supercell-1km", "budget": "10m", "name": "taken"})
    assert taken.status == 409 and taken.json["fix"] == "Pick another name."
    too_big = client.post("/api/guided/start", {"preset": "supercell-100m", "budget": "overnight", "name": "big"})
    assert too_big.status == 409
    assert "of free GPU memory" in too_big.json["message"] and too_big.json["command"].startswith("gpuwm-cm1 preset new")
    assert not (root / "big").exists()
    bad = client.post("/api/guided/start", {"preset": "supercell-1km", "budget": "10m", "name": "../x"})
    assert bad.status == 400


@pytest.mark.parametrize("fixture, blocked, words", [
    (None, "no-gpu", "No NVIDIA GPU answered on this computer"),
    ("a100-80gb-old-driver.json", "old-driver", "Your NVIDIA driver is version 550.54; GM1 needs version 580"),
])
def test_guided_mode_offers_nothing_when_no_gpu_here_can_run_it(live, monkeypatch, fixture, blocked, words):
    """The fit prices a run with no GPU (for an RTX 3080), but guided Start would run it here and fail at once:
    the card is greyed with why and what to do, and Start refuses before any folder is made."""

    server, client, root = live
    if fixture is None:
        monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(root / "no-such-nvidia-smi"))
    else:
        monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(FIXTURES / fixture))
    reply = client.post("/api/fit", {"preset": "supercell-1km", "budget": "10m"})
    assert reply.status == 200, reply.raw
    body = reply.json
    assert body["fit"]["status"] == "fits"                    # the fit itself still prices it
    assert body["offered"] is False and body["blocked"] == blocked and "start" not in body
    assert body["message"].startswith(words)
    assert body["fix"]
    started = client.post("/api/guided/start", {"preset": "supercell-1km", "budget": "10m", "name": "here"})
    assert started.status == 409 and started.json["message"].startswith(words)
    assert started.json["command"].startswith("gpuwm-cm1 preset new supercell-1km")
    assert not (root / "here").exists()


def test_two_starts_with_one_name_make_one_run_and_one_log_line(live):
    """A double click (or a second tab) sends Start twice with the same name: one run folder is made, and the
    other Start's failed command never reaches its commands.log (the log must replay as the session ran)."""

    import threading

    server, client, root = live
    replies: list = []

    def press() -> None:
        replies.append(client.post("/api/guided/start", {"preset": "supercell-1km", "budget": "10m",
                                                         "name": "twice"}))

    pressers = [threading.Thread(target=press) for _ in range(2)]
    for presser in pressers:
        presser.start()
    for presser in pressers:
        presser.join(120)
    assert sorted(reply.status for reply in replies) == [200, 409], [reply.raw[:300] for reply in replies]
    wait_for(lambda: client.get("/api/runs/twice/status").json["state"] in ("finished", "stopped"), timeout=120)
    log = (root / "twice" / "commands.log").read_text()
    assert log.count("gpuwm-cm1 preset new") == 1, log


# ------------------------------------------------------------------ commands.log replays


def shape(files: dict[str, bytes]) -> set[str]:
    """The run folder's layout: top-level entries, and each segment's own entries (frames/, restarts/, ...)."""

    out = set()
    for path in files:
        if path in ("run.json", "commands.log"):
            continue
        parts = path.split("/")
        out.add("/".join(parts[:3]) if parts[0] == "segments" else parts[0])
    return out


def _files(rundir: Path) -> dict[str, bytes]:
    return {path.relative_to(rundir).as_posix(): path.read_bytes()
            for path in sorted(rundir.rglob("*")) if path.is_file()}


#: fit.json fields that record the machine at the moment of the fit, not what the fit chose.
MOMENT = {"elapsed_s"}
MOMENT_HOST = {"ram_available_bytes", "disk_free_bytes", "disk_total_bytes", "ram_total_bytes"}


def _fit_without_moment(data: bytes) -> dict:
    record = json.loads(data)
    for key in MOMENT:
        record.pop(key, None)
    for key in MOMENT_HOST:
        (record.get("host") or {}).pop(key, None)
    return record


def test_the_commands_log_replayed_from_a_shell_makes_the_same_run_directory(live, tmp_path):
    server, client, root = live
    name, _ = _session(client, root)
    rundir = root / name
    wait_for(lambda: not server.jobs.launched(name), timeout=60)      # the sidecars end with the run
    script = (rundir / "commands.log").read_text()
    expected = tmp_path / "expected"
    shutil.copytree(rundir, expected)
    shutil.rmtree(rundir)
    # gpuwm-cm1 on PATH is the fake runner, as the server ran it; the script is the log as written.
    shims = tmp_path / "bin"
    shims.mkdir()
    shim = shims / "gpuwm-cm1"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable} -m cm1core.gui.fake_runner \"$@\"\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{shims}{os.pathsep}{os.environ['PATH']}", "PYTHONPATH": str(ROOT / "src")}
    replay = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=300,
                            cwd=tmp_path)
    assert replay.returncode == 0, replay.stderr[-2000:]
    got, want = _files(rundir), _files(expected)
    # The files the commands determine are the same, byte for byte.
    setup = ["namelist.input", "namelist.input.original", "settings.toml", "deck.toml", "import-report.json",
             "import-report.txt"]
    for path in setup:
        assert got[path] == want[path], path
    assert _fit_without_moment(got["fit.json"]) == _fit_without_moment(want["fit.json"])
    # run.json records when and where it ran; commands.log is the script itself (the server writes it).
    # The run is the same segment with the same kinds of files, except what the Stop's timing decides: the
    # session stopped after its first frame (its control/stop request, moved into the segment once the run ended,
    # as the real supervisor archives it), and a replayed Stop meets a finished run, which has run on to its one
    # restart at the end (rstfrq < 0) and every frame.
    assert shape(want) - shape(got) == {"segments/001/control-stop.ended"}
    assert shape(got) - shape(want) == {"segments/001/restarts"}
    assert (expected / "segments/001/control-stop.ended").is_file()
    assert not (expected / "control" / "stop").exists()
    assert len(list((rundir / "segments/001/frames").iterdir())) > len(list((expected / "segments/001/frames").iterdir()))
    ends = [json.loads(line) for line in (rundir / "segments/001/events.jsonl").read_text().splitlines()]
    assert ends[-1]["kind"] == "end" and ends[-1]["reason"] == "completed"
    assert "Nothing to stop" in replay.stdout + replay.stderr
