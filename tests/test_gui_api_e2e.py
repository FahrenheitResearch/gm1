"""The API end to end with the fake runner.

import -> check -> geometry -> estimate -> start -> SSE progress -> end -> results, through HTTP on 127.0.0.1
exactly as the pages call it; then stop, pause, resume, force stop, a run that vanishes, a reconnecting stream,
and commands.log.  The fake runner (cm1core.gui.fake_runner) writes the run-directory contract and a
gm1.run-events/v1 log that passes the schema; no GPU, no model.  CPU only.
"""

from __future__ import annotations

from pathlib import Path
import shlex

import pytest

from cm1core.gui import auth
from cm1core.run_events import validate_file
from gui_support import import_supercell, start_server, stop_server, wait_for


@pytest.fixture()
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    yield server, client, tmp_path / "runs"
    stop_server(server)


def _ended(event):
    return event.get("event") == "end"


def _commands_logged(rundir: Path) -> list[str]:
    return [line for line in (rundir / "commands.log").read_text().splitlines()
            if line and not line.startswith("#")]


def test_import_check_geometry_estimate_start_events_end_results(live):
    server, client, root = live
    rundir = root / "sc1"
    replies = {}

    # import: the namelist text is uploaded; the server writes it where the command can read it
    replies["import"] = imported = import_supercell(client)
    assert imported["ok"] and imported["verdict"] == "runnable" and imported["rc"] == 0
    assert imported["run"] == "sc1" and imported["url"] == "sc1"
    argv = shlex.split(imported["command"])
    assert argv[:2] == ["gpuwm-cm1", "import"] and argv[argv.index("--out") + 1] == str(rundir)
    for name in ("run.json", "namelist.input", "namelist.input.original", "import-report.json", "deck.toml"):
        assert (rundir / name).is_file(), name
    runs = client.get("/api/runs").json["runs"]
    assert [(row["id"], row["status"]["state"]) for row in runs] == [("sc1", "imported")]
    report = client.get("/api/runs/sc1/report").json
    assert report["report"]["exit_code"] == 0

    # check, geometry, estimate: host-only commands, each returns the line it ran
    replies["check"] = checked = client.post("/api/runs/sc1/check").json
    assert checked["ok"] and checked["command"] == f"gpuwm-cm1 check {rundir / 'deck.toml'} --json"
    replies["geometry"] = geometry = client.post("/api/runs/sc1/geometry").json
    assert geometry["ok"] and geometry["geometry"]["status"] == "evaluated"
    assert shlex.split(geometry["command"])[1:4] == ["cartesian", str(rundir / "deck.toml"), "--plan"]
    replies["estimate"] = estimate = client.post("/api/runs/sc1/estimate").json
    assert estimate["ok"] and estimate["result"]["gpu"]["fits"] is True
    assert estimate["command"] == f"gpuwm-cm1 estimate {rundir} --json"
    assert (rundir / "estimate.json").is_file()
    assert client.get("/api/runs/sc1/status").json["state"] == "ready"

    # start: the run in its own session, and the two sidecars
    replies["start"] = started = client.post("/api/runs/sc1/start").json
    assert started["ok"], started
    assert started["command"] == f"gpuwm-cm1 run {rundir}"
    assert [shlex.split(line)[6] for line in started["sidecars"]] == ["cm1out", "quicklook"]

    # SSE: every record of the log, and status changes, until the end record
    events = client.events("sc1", _ended, timeout=60)
    kinds = [event["event"] for event in events]
    assert kinds[0] in ("segment", "status")
    records = [event for event in events if event["event"] not in ("status", "segment")]
    assert [event["event"] for event in records][:2] == ["admitted", "startup"]
    assert kinds.count("progress") >= 2 and kinds.count("frame") == 9 and kinds[-1] == "end"
    assert [event["id"] for event in records] == [f"1:{n}" for n in range(len(records))]
    assert [event["data"]["seq"] for event in records] == list(range(len(records)))
    progress = [event["data"] for event in records if event["event"] == "progress"]
    assert progress[-1]["step"] == progress[-1]["total_steps"] == 1200
    assert events[-1]["data"]["reason"] == "completed" and events[-1]["data"]["segment"] == "001"
    statuses = [event["data"]["label"] for event in events if event["event"] == "status"]
    assert any(label.startswith(("Running", "Starting")) for label in statuses), statuses
    assert validate_file(rundir / "segments" / "001" / "events.jsonl") == []

    # end: Finished, and the sidecars finish their last frame
    assert wait_for(lambda: client.get("/api/runs/sc1/status").json["state"] == "finished")
    wait_for(lambda: not server.jobs.launched("sc1"), timeout=30)

    # results: the cm1out files, the quick-looks, copy lines, the receipt and disk use
    results = client.get("/api/runs/sc1/results?mode=advanced").json
    names = [row["name"] for row in results["cm1out"]]
    assert names == [f"cm1out_{n:06d}.nc" for n in range(1, 10)] + ["cm1out_stats.nc"]
    assert results["receipt"]["status"] == "completed"
    assert {look["id"] for look in results["looks"]} >= {"reflectivity", "spin-near-ground", "updraft-5km", "swath"}
    assert all(len(look["frames"]) == 9 for look in results["looks"])
    assert any(line["label"] == "ncview" for line in results["copy"])
    assert results["disk"]["total_bytes"] > 0 and results["disk"]["frames_bytes"] > 0
    picture = client.get(results["looks"][0]["frames"][-1]["url"])
    assert picture.status == 200 and picture.headers["content-type"] == "image/png"
    assert picture.raw.startswith(b"\x89PNG\r\n\x1a\n")
    guided = client.get("/api/runs/sc1/results?mode=guided").json
    assert [look["id"] for look in guided["looks"]][:4] == [
        "reflectivity", "spin-near-ground", "updraft-5km", "swath"]

    # every POST carried its command, and commands.log holds each in order, replayable
    for action, reply in replies.items():
        assert reply["command"].startswith("gpuwm-cm1 "), action
    logged = _commands_logged(rundir)
    assert logged[:4] == [replies[a]["command"] for a in ("import", "check", "geometry", "estimate")]
    assert logged[-1] == "wait" and logged[-2] == started["command"]
    assert logged[4:6] == started["sidecars"]
    # the GUI's own files stay out of the run directory
    assert not any(path.suffix == ".log" and path.name != "run.log" and path.name != "commands.log"
                   for path in rundir.rglob("*"))


def test_stop_ends_the_run_at_the_next_step(live, monkeypatch):
    server, client, root = live
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.02")
    import_supercell(client, "slow")
    assert client.post("/api/runs/slow/start").json["ok"]
    client.events("slow", lambda e: e.get("event") == "progress" and e["data"].get("step"), timeout=30)
    stopped = client.post("/api/runs/slow/stop").json
    assert stopped["ok"] and stopped["command"] == f"gpuwm-cm1 stop {root / 'slow'}"
    assert stopped["escalates_to"] == f"gpuwm-cm1 stop {root / 'slow'} --terminate"
    end = client.events("slow", _ended, timeout=30)[-1]["data"]
    assert end["reason"] == "stopped" and end["request"] == "stop" and end["rc"] == 0
    assert wait_for(lambda: client.get("/api/runs/slow/status").json["state"] == "stopped")
    assert f"gpuwm-cm1 stop {root / 'slow'}" in _commands_logged(root / "slow")
    # stopping a run that has ended writes nothing, so the next start meets no stale control file
    again = client.post("/api/runs/slow/stop").json
    assert again["ok"] and "Nothing to stop" in again["message"]
    assert again["command"] == f"gpuwm-cm1 stop {root / 'slow'}"


def test_pause_then_resume_continues_in_a_new_segment(live, monkeypatch, tmp_path):
    server, client, root = live
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.004")
    text = (Path(__file__).parent / "data/cm1import/r21.1-supercell.namelist.input").read_text()
    text = text.replace(" rstfrq = -3600.0,", " rstfrq =  1800.0,")
    assert " rstfrq =  1800.0," in text
    assert client.post("/api/import", {"name": "p", "namelist": {"text": text}}).json["ok"]
    assert client.post("/api/runs/p/start").json["ok"]
    client.events("p", lambda e: e.get("event") == "progress" and e["data"].get("step"), timeout=30)
    paused = client.post("/api/runs/p/pause").json
    assert paused["ok"] and paused["command"] == f"gpuwm-cm1 pause {root / 'p'}"
    events = client.events("p", _ended, timeout=30)
    end = events[-1]["data"]
    assert end["reason"] == "paused" and end["request"] == "pause"
    assert events[-2]["event"] == "restart" or any(e["event"] == "restart" for e in events)
    status = wait_for(lambda: (lambda s: s if s["state"] == "paused" else None)(
        client.get("/api/runs/p/status").json))
    assert "resume" in status["actions"]
    wait_for(lambda: not server.jobs.launched("p"), timeout=30)

    resumed = client.post("/api/runs/p/resume", {"from_restart": status["last_restart"]}).json
    assert resumed["ok"], resumed
    assert resumed["command"] == f"gpuwm-cm1 resume {root / 'p'} --from-restart {status['last_restart']}"
    events = client.events("p", lambda e: _ended(e) and e["data"]["segment"] == "002", timeout=60)
    segments = [e["data"]["segment"] for e in events if e["event"] == "segment"]
    assert segments == ["001", "002"]
    assert events[-1]["data"]["reason"] == "completed"
    admitted = [e["data"] for e in events if e["event"] == "admitted"]
    assert admitted[-1]["resumed_from"] and admitted[-1]["start_step"] > 0
    # ids carry the segment, so a reconnect resumes in the right log
    assert events[-1]["id"].startswith("2:")
    assert validate_file(root / "p" / "segments" / "002" / "events.jsonl") == []


def test_a_reconnecting_stream_resumes_after_the_last_event(live):
    server, client, root = live
    import_supercell(client, "r")
    assert client.post("/api/runs/r/start").json["ok"]
    full = [e for e in client.events("r", _ended, timeout=60) if "id" in e]
    later = client.events("r", _ended, timeout=30, last_event_id=full[4]["id"])
    ids = [e["id"] for e in later if "id" in e]
    assert ids == [e["id"] for e in full[5:]]


def test_the_stream_works_with_the_cookie_alone(live):
    server, client, root = live
    import_supercell(client, "c")
    assert client.post("/api/runs/c/start").json["ok"]
    jar = f"{auth.cookie_name(server.port)}={server.token}"
    events = client.events("c", _ended, timeout=60, cookie=jar)
    assert events[-1]["data"]["reason"] == "completed"


def test_force_stop_kills_and_the_run_reads_stopped_without_a_final_record(live, monkeypatch):
    server, client, root = live
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.05")
    import_supercell(client, "k")
    assert client.post("/api/runs/k/start").json["ok"]
    client.events("k", lambda e: e.get("event") == "progress" and (e["data"].get("step") or 0) >= 5, timeout=60)
    killed = client.post("/api/runs/k/force-stop").json
    assert killed["ok"] and killed["command"] == f"gpuwm-cm1 stop {root / 'k'} --force"
    status = wait_for(lambda: (lambda s: s if s["state"] == "stopped" else None)(
        client.get("/api/runs/k/status").json), timeout=15)
    assert status["label"] == "Stopped without a final record (last frame 0)"
    # the sidecars follow the run, so they end with it
    wait_for(lambda: not server.jobs.launched("k"), timeout=15)
    assert not (root / "k" / "segments" / "001" / "frames" / ".incomplete").exists()


def test_a_run_that_vanishes_reads_stopped_without_a_final_record(live, monkeypatch):
    server, client, root = live
    monkeypatch.setenv("GM1_FAKE_OUTCOME", "vanish")
    import_supercell(client, "v")
    assert client.post("/api/runs/v/start").json["ok"]
    status = wait_for(lambda: (lambda s: s if s["state"] == "stopped" else None)(
        client.get("/api/runs/v/status").json), timeout=30)
    assert "without a final record" in status["label"]
    wait_for(lambda: not server.jobs.launched("v"), timeout=15)


def test_a_run_that_fails_reads_failed(live, monkeypatch):
    server, client, root = live
    monkeypatch.setenv("GM1_FAKE_OUTCOME", "courant-stop")
    import_supercell(client, "f")
    assert client.post("/api/runs/f/start").json["ok"]
    end = client.events("f", _ended, timeout=60)[-1]["data"]
    assert end["reason"] == "courant-stop"
    status = client.get("/api/runs/f/status").json
    assert status["state"] == "failed" and "Courant" in status["label"]


def test_start_twice_is_refused_while_running(live, monkeypatch):
    server, client, root = live
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.02")
    import_supercell(client, "twice")
    assert client.post("/api/runs/twice/start").json["ok"]
    second = client.post("/api/runs/twice/start")
    assert second.status == 409 and "already" in second.json["message"]
    assert second.json["command"] == f"gpuwm-cm1 run {root / 'twice'}"
    client.post("/api/runs/twice/stop")
    client.events("twice", _ended, timeout=30)


def test_a_refused_namelist_imports_but_does_not_start(live):
    server, client, root = live
    text = (Path(__file__).parent / "data/cm1import/r21.1-supercell.namelist.input").read_text()
    text = text.replace(" stretch_x =      0,", " stretch_x =      1,")
    assert "stretch_x =      1" in text
    reply = client.post("/api/import", {"name": "bad", "namelist": {"text": text}}).json
    assert reply["verdict"] == "refused" and reply["rc"] == 2 and not reply["ok"]
    assert "Check" in reply["fix"]
    started = client.post("/api/runs/bad/start")
    assert started.status == 409 and "cannot run" in started.json["message"]


def test_import_errors_say_what_happened_and_what_to_do(live, tmp_path):
    server, client, root = live
    import_supercell(client, "dup")
    dup = client.post("/api/import", {"name": "dup", "namelist": {"text": "&param1\n/\n"}})
    assert dup.status == 409 and dup.json["fix"]
    missing = client.post("/api/import", {"name": "m", "namelist": {"path": str(tmp_path / "nope")}})
    assert missing.status == 400 and "no file" in missing.json["message"].lower()
    bad_name = client.post("/api/import", {"name": "../x", "namelist": {"text": "x"}})
    assert bad_name.status == 400
    not_json = client.request("POST", "/api/import", raw_body=b"name=x",
                              headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert not_json.status == 415
    path_import = client.post("/api/import", {"name": "bypath", "folder": "study",
                                              "namelist": {"path": str(Path(__file__).parent /
                                                                   "data/cm1import/r21.1-supercell.namelist.input")}})
    assert path_import.json["ok"] and path_import.json["run"] == "study/bypath"
    assert path_import.json["url"] == "study%2Fbypath"
    assert client.get("/api/runs/study%2Fbypath/status").json["state"] == "imported"
    # a run never goes inside another run's folder (it would be hidden from the Runs list)
    nested = client.post("/api/import", {"name": "inner", "folder": "dup", "namelist": {"text": "&param1\n/\n"}})
    assert nested.status == 409 and "inside another run" in nested.json["message"] and nested.json["fix"]
    assert not (root / "dup" / "inner").exists()
    (root / "afile").write_text("x")
    onto_file = client.post("/api/import", {"name": "y", "folder": "afile", "namelist": {"text": "&param1\n/\n"}})
    assert onto_file.status == 409 and "not a folder" in onto_file.json["message"]


def test_the_api_table_lists_every_action_with_its_command(live):
    server, client, root = live
    endpoints = client.get("/api/api").json["endpoints"]
    posts = {row["action"]: row for row in endpoints if row["method"] == "POST"}
    assert set(posts) == {"mode", "import", "check", "geometry", "estimate", "start", "resume", "stop",
                          "pause", "force-stop",
                          # picture requests
                          "fields", "slice", "section", "column",
                          # the setup screens
                          "edit", "recommended", "namelist", "acknowledge", "sounding", "jobscript", "measure",
                          # guided mode
                          "fit", "guided-start",
                          # the Results screen
                          "convert", "reproduce",
                          # advanced mode: Start from a preset
                          "preset-new",
                          # A film of a look, a look drawn on request
                          "film", "look",
                          # An environment from a WRF or ArWen run
                          "wrf-open", "wrf-column", "wrf-use",
                          # A sounding file of the user's own
                          "sounding-check", "sounding-options",
                          # Downscaling a WRF or ArWen run (3D)
                          "chain-open", "chain-check", "chain-new"}
    assert all(row["command"].startswith("gpuwm-cm1 ") for row in posts.values())
    session = client.get("/api/session").json
    assert session["runner"] == "fake" and set(session["commands_available"]) >= {"import", "run", "resume"}


def test_the_real_runner_names_commands_this_build_lacks(tmp_path):
    from cm1core.gui.jobs import real_runner

    server, client = start_server(tmp_path / "runs", tmp_path / "config", runner=real_runner())
    try:
        available = client.get("/api/session").json["commands_available"]
        assert {"gui", "stop", "pause", "check", "cartesian"} <= set(available)
        if "import" in available:
            pytest.skip("this build has gpuwm-cm1 import")
        answer = client.post("/api/import", {"name": "x", "namelist": {"text": "&param1\n/\n"}})
        assert answer.status == 501
        assert "no 'import' command" in answer.json["message"] and answer.json["fix"]
        assert not (tmp_path / "runs" / "x").exists()
    finally:
        stop_server(server)


def test_the_real_runner_checks_a_real_deck(tmp_path):
    """The same endpoint with the real gpuwm-cm1: check runs as a host-only subprocess and its JSON comes back."""

    from cm1core.gui.jobs import real_runner

    pytest.importorskip("numpy")      # the real check reads the deck with numpy (a gpuwm dependency)
    deck = Path(__file__).resolve().parents[1] / "tests/data/decks/stock-imported.gpuwm-cm1.toml"
    rundir = tmp_path / "runs" / "stock"
    rundir.mkdir(parents=True)
    (rundir / "deck.toml").write_bytes(deck.read_bytes())
    (rundir / "run.json").write_text('{"schema": "gm1.cm1-run/v1", "segments": []}')
    server, client = start_server(tmp_path / "runs", tmp_path / "config", runner=real_runner())
    try:
        reply = client.post("/api/runs/stock/check").json
        assert reply["ok"], reply["output"]["stderr_tail"]
        assert reply["command"] == f"gpuwm-cm1 check {rundir / 'deck.toml'} --json"
        assert reply["result"]["config"]["boundaries"]["west"] == 2
        assert reply["output"]["argv"][:2] == ["gpuwm-cm1", "check"]
        assert _commands_logged(rundir) == [reply["command"]]
    finally:
        stop_server(server)


# ---------------------------------------------------------------- Stop, and the run after it

def test_a_stop_escalation_never_reaches_the_next_run(live, monkeypatch):
    """Stop arms a SIGTERM for the run it was pressed on.  Starting again before it fires must not let it
    signal the new run (it did: the second run ended rc 143 by SIGTERM with no request)."""

    import time

    import cm1core.gui.api as api_module

    server, client, root = live
    monkeypatch.setattr(api_module, "ESCALATE_AFTER_S", 1.0)
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.01")
    import_supercell(client, "again")
    assert client.post("/api/runs/again/start").json["ok"]
    client.events("again", lambda e: e.get("event") == "progress" and e["data"].get("step"), timeout=30)
    stopped = client.post("/api/runs/again/stop").json
    assert stopped["ok"] and stopped["escalates_to"]
    assert wait_for(lambda: client.get("/api/runs/again/status").json["state"] == "stopped", timeout=20)
    wait_for(lambda: not server.jobs.launched("again"), timeout=20)
    assert client.post("/api/runs/again/start").json["ok"]
    time.sleep(2.5)                                         # well past the first Stop's escalation
    status = client.get("/api/runs/again/status").json
    assert (status["segment"], status["state"]) == ("002", "running"), status
    assert "stop_requested_s_ago" not in status
    assert f"gpuwm-cm1 stop {root / 'again'} --terminate" not in _commands_logged(root / "again")
    client.post("/api/runs/again/stop")
    end = client.events("again", lambda e: _ended(e) and e["data"]["segment"] == "002", timeout=30)[-1]["data"]
    assert (end["reason"], end["request"], end["signal"]) == ("stopped", "stop", None)


def test_the_escalation_still_stops_a_run_that_ignores_stop(live, monkeypatch):
    """The same timer does act on the run it was aimed at: a run still going after it gets SIGTERM."""

    import cm1core.gui.api as api_module

    server, client, root = live
    monkeypatch.setattr(api_module, "ESCALATE_AFTER_S", 1.0)
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.02")
    monkeypatch.setenv("GM1_FAKE_IGNORE_CONTROL", "1")     # a run that does not read control/
    import_supercell(client, "deaf")
    assert client.post("/api/runs/deaf/start").json["ok"]
    client.events("deaf", lambda e: e.get("event") == "progress" and e["data"].get("step"), timeout=30)
    stopped = client.post("/api/runs/deaf/stop").json
    assert stopped["ok"] and stopped["escalates_to"]
    assert client.post("/api/runs/deaf/stop").json["ok"]    # pressed twice: still one escalation
    end = client.events("deaf", _ended, timeout=30)[-1]["data"]
    assert (end["reason"], end["signal"], end["rc"]) == ("stopped", "SIGTERM", 143)
    assert _commands_logged(root / "deaf").count(f"gpuwm-cm1 stop {root / 'deaf'} --terminate") == 1


def test_stop_before_the_run_writes_anything_still_stops_it(live, monkeypatch):
    """Right after Start the status reads Starting and offers Stop, before the run has written a record.  Stop
    then writes control/stop for it (it said "Nothing to stop" and the run went on); a run that meets the file
    at start-up refuses to begin, as run_control.require_control_directory does."""

    server, client, root = live
    monkeypatch.setenv("GM1_FAKE_LAUNCH_DELAY_S", "1.5")
    import_supercell(client, "early")
    started = client.post("/api/runs/early/start").json
    assert started["ok"] and started["status"]["label"] == "Starting"
    assert "stop" in started["status"]["actions"]
    stopped = client.post("/api/runs/early/stop").json
    assert stopped["ok"] and stopped["message"].startswith("Asked the run to stop"), stopped
    assert (root / "early" / "control" / "stop").is_file()
    wait_for(lambda: not server.jobs.launched("early"), timeout=30)
    status = client.get("/api/runs/early/status").json
    assert status["state"] != "finished" and not (root / "early" / "segments").exists(), status
    # the next Start clears the old request and runs to the end
    monkeypatch.setenv("GM1_FAKE_LAUNCH_DELAY_S", "0")
    assert client.post("/api/runs/early/start").json["ok"]
    assert client.events("early", _ended, timeout=60)[-1]["data"]["reason"] == "completed"


def test_a_run_that_fails_at_once_takes_its_sidecars_with_it(live):
    server, client, root = live
    import_supercell(client, "nodeck")
    (root / "nodeck" / "deck.toml").unlink()
    started = client.post("/api/runs/nodeck/start").json
    assert not started["ok"] and started["rc"] == 2 and "deck.toml" in started["output"]["log_tail"]
    # the sidecars were ended with it rather than waiting for a run that never began
    assert not server.jobs.launched("nodeck")


def test_results_copy_lines_cover_one_file_netcdf_and_the_grads_descriptors(tmp_path):
    """The output formats: cm1out.nc (output_filetype = 1) gets ncview and xarray lines too, and the GrADS lines open
    the history descriptors, the statistics and the parcels, not the grid's metadata descriptor."""

    from cm1core.gui.api import results_copy_lines

    lines = results_copy_lines(tmp_path, ["cm1out.nc", "cm1out_stats.nc", "cm1out_s.ctl", "cm1out_s.dat",
                                          "cm1out_metadata.ctl", "cm1out_stats.ctl", "cm1out_parcel.ctl"])
    assert [line["label"] for line in lines] == ["ncview", "Python (xarray)", "GrADS, parcels", "GrADS",
                                                 "GrADS, statistics"]
    assert lines[0]["text"] == f"ncview {tmp_path / 'cm1out.nc'}"
    assert "xr.open_dataset(" in lines[1]["text"]
    per_time = results_copy_lines(tmp_path, ["cm1out_000001.nc", "cm1out_000002.nc", "cm1out.nc"])
    assert [line["label"] for line in per_time] == ["ncview", "Python (xarray)"]
    assert "open_mfdataset" in per_time[1]["text"]
