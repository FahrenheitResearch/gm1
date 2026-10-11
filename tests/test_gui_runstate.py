"""A run's status from its run directory alone, and the stop/pause commands.  CPU only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cm1core.gui import control
from cm1core.gui.files import PathRefused, Root
from cm1core.gui.runstate import EventTail, proc_start_ticks, process_alive, read_status


def _utc(seconds_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat(timespec="milliseconds")


def _run(tmp_path: Path, records: list[dict], row: dict | None = None, name: str = "r") -> Path:
    rundir = tmp_path / name
    segment = rundir / "segments" / "001"
    segment.mkdir(parents=True)
    run = {"schema": "gm1.cm1-run/v1", "segments": [row] if row else []}
    (rundir / "run.json").write_text(json.dumps(run))
    with (segment / "events.jsonl").open("w") as stream:
        for seq, record in enumerate(records):
            stream.write(json.dumps({"schema": "gm1.run-events/v1", "seq": seq, **record}) + "\n")
    return rundir


def _admitted(host: str, pid: int, ago: float = 30.0) -> dict:
    return {"kind": "admitted", "host": host, "pid": pid, "time_utc": _utc(ago), "duration_s": 7200.0,
            "steps": 1200, "start_step": 0, "grid": {"nx": 120, "ny": 120, "nz": 40, "dx_m": 1000.0}}


def _progress(pid: int, step: int, ago: float, phase: str = "stepping") -> dict:
    return {"kind": "progress", "pid": pid, "time_utc": _utc(ago), "phase": phase, "step": step,
            "total_steps": 1200, "model_time_s": step * 6.0, "duration_s": 7200.0, "eta_s": 10.0}


def test_a_live_local_run_reads_running_with_its_percent(tmp_path):
    pid = os.getpid()
    rundir = _run(tmp_path, [_admitted("here", pid), _progress(pid, 600, 1.0)],
                  {"index": 1, "pid": pid, "pid_start_ticks": proc_start_ticks(pid), "host": "here"})
    status = read_status(rundir, host="here")
    assert (status.state, status.label, status.percent) == ("running", "Running 50%", 50.0)
    assert set(status.detail["actions"]) >= {"stop", "pause", "force-stop"}


def test_a_reused_pid_is_not_the_run(tmp_path):
    pid = os.getpid()
    rundir = _run(tmp_path, [_admitted("here", pid), _progress(pid, 600, 1.0)],
                  {"index": 1, "pid": pid, "pid_start_ticks": (proc_start_ticks(pid) or 0) + 1, "host": "here"})
    status = read_status(rundir, host="here")
    assert status.state == "stopped" and status.label == "Stopped without a final record"


def test_a_dead_local_run_without_an_end_record(tmp_path):
    child = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    pid = int(child.stdout)
    frame = {"kind": "frame", "pid": pid, "time_utc": _utc(5), "index": 3, "model_time_s": 2700.0}
    rundir = _run(tmp_path, [_admitted("here", pid), frame, _progress(pid, 460, 4)])
    status = read_status(rundir, host="here")
    assert status.label == "Stopped without a final record (last frame 3)"
    assert status.detail["final_record"] is False


def test_a_run_on_another_host_is_stale_after_60_s(tmp_path):
    fresh = _run(tmp_path, [_admitted("gpu7.cluster.example", 4242), _progress(4242, 100, 20)], name="fresh")
    old = _run(tmp_path, [_admitted("gpu7.cluster.example", 4242), _progress(4242, 100, 300)], name="old")
    assert read_status(fresh, host="here").state == "running"
    status = read_status(old, host="here")
    assert status.state == "stale" and status.label == "Stale (no word for 5 min)"
    assert "force-stop" not in status.detail["actions"] and "stop" in status.detail["actions"]


def test_waiting_for_the_gpu(tmp_path):
    pid = os.getpid()
    waiting = {**_progress(pid, 0, 1.0, "waiting-for-gpu"), "step": None}
    rundir = _run(tmp_path, [_admitted("here", pid), waiting])
    assert read_status(rundir, host="here").label == "Waiting for GPU"


@pytest.mark.parametrize("reason, state, label", [
    ("completed", "finished", "Finished"), ("stopped", "stopped", "Stopped"), ("paused", "paused", "Paused"),
    ("error", "failed", "Failed"),
])
def test_the_end_record_decides(tmp_path, reason, state, label):
    end = {"kind": "end", "pid": 1, "time_utc": _utc(), "reason": reason, "model_time_s": 3600.0}
    rundir = _run(tmp_path, [_admitted("elsewhere", 1), end])
    status = read_status(rundir, host="here")
    assert (status.state, status.label, status.percent) == (state, label, 50.0)


def test_setup_states(tmp_path):
    rundir = tmp_path / "setup"
    rundir.mkdir()
    (rundir / "run.json").write_text("{}")
    assert read_status(rundir).label == "Imported"
    (rundir / "import-report.json").write_text(json.dumps({"exit_code": 2}))
    assert read_status(rundir).label == "Imported (refused: see Check)"
    (rundir / "import-report.json").write_text(json.dumps({"exit_code": 0}))
    (rundir / "estimate.json").write_text("{}")
    status = read_status(rundir)
    assert status.state == "ready" and "start" in status.detail["actions"]


def test_a_torn_last_line_is_left_for_the_next_read(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_bytes(b'{"seq": 0, "kind": "admitted"}\n{"seq": 1, "ki')
    tail = EventTail(log)
    assert [record["seq"] for record in tail.read_new()] == [0]
    with log.open("ab") as stream:
        stream.write(b'nd": "progress"}\n')
    assert [record["seq"] for record in tail.read_new()] == [1]
    assert tail.read_new() == []


def test_stop_and_pause_write_control_files_only_for_a_live_run(tmp_path):
    pid = os.getpid()
    rundir = _run(tmp_path, [_admitted("x", pid), _progress(pid, 10, 1.0)], {"index": 1, "pid": pid, "host": "x"})
    result = control.request(rundir, "pause")
    assert result.ok and (rundir / "control" / "pause").is_file()
    ended = _run(tmp_path, [_admitted("x", 1), {"kind": "end", "reason": "completed", "time_utc": _utc()}],
                 name="done")
    result = control.request(ended, "stop")
    assert result.ok and result.wrote is None and not (ended / "control").exists()
    assert not control.request(tmp_path / "nowhere", "stop").ok


def test_the_stop_command_line(tmp_path, capsys):
    """The command line's stop and pause are the run-directory commands (cm1run).

    gui.control.add_commands steps aside when they are registered, so the
    command a Stop or Pause button shows is the one a terminal runs.  The
    server's own in-process request (above) answers "Nothing to stop" for a
    finished run; the command line refuses it with exit 2 and the same fact.
    """

    from cm1core.cli import build_parser, main
    from cm1core.cm1run import doors

    choices = next(a.choices for a in build_parser()._actions if isinstance(getattr(a, "choices", None), dict))
    for name in ("stop", "pause"):
        assert choices[name].get_default("handler").__module__ == doors.__name__, name
    assert main(["stop", str(tmp_path / "nowhere")]) == 2
    assert "not a run directory" in capsys.readouterr().err
    ended = _run(tmp_path, [{"kind": "end", "reason": "completed", "time_utc": _utc()}], name="done")
    assert main(["stop", str(ended), "--json"]) == 2
    assert "nothing to stop" in json.loads(capsys.readouterr().err)["reason"]
    assert main(["pause", str(ended)]) == 2
    assert "nothing to pause" in capsys.readouterr().err


def test_process_alive_ignores_zombies_and_wrong_start_times():
    assert process_alive(os.getpid())
    assert process_alive(os.getpid(), proc_start_ticks(os.getpid()))
    assert not process_alive(os.getpid(), -1)
    assert not process_alive(None) and not process_alive(0)


def test_run_ids_stay_under_the_root(tmp_path):
    root = Root(tmp_path)
    assert root.run_path("a") == tmp_path.resolve() / "a"
    assert root.run_path("study/a") == tmp_path.resolve() / "study" / "a"
    for bad in ("..", "../x", "a/b/c", ".hidden", "a b", "/abs", "", "x" * 65):
        with pytest.raises(PathRefused):
            root.run_path(bad)
    (tmp_path / "link").symlink_to("/")
    with pytest.raises(PathRefused):
        root.run_path("link/etc")
