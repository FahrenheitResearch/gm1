"""Run status and host-aware crash detection.

On this host a run's processes are checked by pid and ``/proc`` start time.
On another host ``/proc`` is never read: the heartbeat's age, then the
scheduler, decide between Running, Stale and "Stopped without a final record".
"""

from __future__ import annotations

import builtins
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from cm1run_support import make_run

from cm1core.cm1run import rundir as rundir_module
from cm1core.cm1run import status as status_module
from cm1core.cm1run.rundir import RunDirectory, process_identity, write_json_atomic
from cm1core.cm1run.status import run_status


def _utc(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat(timespec="milliseconds")


def _record(kind: str, seq: int, seconds_ago: float, **fields) -> dict:
    return {"schema": "gm1.run-events/v1", "seq": seq, "kind": kind, "time_utc": _utc(seconds_ago),
            "elapsed_s": 1.0, "pid": 4242, **fields}


def _segment(run: Path, *, host: str, heartbeat_ago: float, state: str = "running",
             scheduler: dict | None = None, supervisor: dict | None = None, child: dict | None = None,
             frames: int = 3, end: dict | None = None) -> Path:
    """A segment as a run on ``host`` leaves it, its last record ``heartbeat_ago`` seconds old."""

    segment = run / "segments" / "001"
    (segment / "frames").mkdir(parents=True)
    (segment / "restarts").mkdir()
    records = [_record("admitted", 0, heartbeat_ago + 30, steps=60, duration_s=360.0)]
    for index in range(frames):
        frame = segment / "frames" / f"frame-{index:06d}"
        frame.mkdir()
        records.append(_record("frame", len(records), heartbeat_ago + 20 - index, index=index,
                               step=index * 10, model_time_s=index * 60.0, directory=str(frame),
                               bytes=1, restart_written=None))
    records.append(_record("progress", len(records), heartbeat_ago, phase="stepping", heartbeat=False,
                           step=(frames - 1) * 10 + 5, total_steps=60, model_time_s=((frames - 1) * 10 + 5) * 6.0,
                           duration_s=360.0, s_per_step_recent=1.0, s_per_step_mean=1.0,
                           model_s_per_wall_s=6.0, eta_s=100.0, last_io=None, gpu=None))
    if end is not None:
        records.append(_record("end", len(records), heartbeat_ago, **end))
    events = segment / "events.jsonl"
    events.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    moment = time.time() - heartbeat_ago
    os.utime(events, (moment, moment))
    write_json_atomic(segment / "segment.json", {
        "schema": "gm1.cm1-run-segment/v1", "number": 1, "state": state, "host": host,
        "boot_id": "another-boot", "heartbeat_utc": _utc(heartbeat_ago + 30),
        "supervisor": supervisor or {"pid": 999999, "start_ticks": 1, "host": host, "boot_id": "another-boot"},
        "child": child or {"pid": 999998, "start_ticks": 2, "host": host, "boot_id": "another-boot"},
        "scheduler": scheduler, "gpu": {"name": "A100"},
    })
    record = json.loads((run / "run.json").read_text())
    record["segments"] = [{"number": 1, "directory": "segments/001", "kind": "fresh", "host": host,
                           "scheduler": scheduler, "until_s": 360.0}]
    write_json_atomic(run / "run.json", record)
    return segment


@pytest.fixture()
def no_proc(monkeypatch: pytest.MonkeyPatch):
    """Fail the test on any read of /proc and on any signal."""

    real_open = builtins.open
    real_io_open = io.open

    def guarded(opener):
        def wrapped(file, *args, **kwargs):
            if str(file).startswith("/proc"):
                raise AssertionError(f"status read {file} for a run on another host")
            return opener(file, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(builtins, "open", guarded(real_open))
    monkeypatch.setattr(io, "open", guarded(real_io_open))

    def trap(*_args, **_kwargs):
        raise AssertionError("status used a process check for a run on another host")

    monkeypatch.setattr(rundir_module, "process_start_ticks", trap)
    monkeypatch.setattr(rundir_module, "boot_id", trap)
    monkeypatch.setattr(os, "kill", trap)
    yield


@pytest.fixture(autouse=True)
def fresh_scheduler_cache():
    status_module._SCHEDULER_CACHE.clear()
    yield
    status_module._SCHEDULER_CACHE.clear()


def _fake_command(tmp_path: Path, name: str, script: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + script, encoding="ascii")
    path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return path


def test_another_hosts_run_with_an_old_heartbeat_is_stale_without_reading_proc(
    tmp_path: Path, no_proc, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    run = make_run(tmp_path)
    _segment(run, host="gpu17.cluster.example", heartbeat_ago=90.0)
    answer = run_status(RunDirectory(run), this_host="login-node")
    assert answer.state == "stale" and answer.label == "Stale"
    assert answer.local is False and answer.host == "gpu17.cluster.example"
    assert 85.0 <= answer.heartbeat_age_s <= 120.0
    assert "gpu17.cluster.example" in answer.detail and "90 s ago" in answer.detail
    assert {row["id"] for row in answer.actions} == {"stop", "abandon"}


def test_another_hosts_run_with_a_recent_heartbeat_is_running(tmp_path: Path, no_proc) -> None:
    run = make_run(tmp_path)
    _segment(run, host="gpu17.cluster.example", heartbeat_ago=8.0)
    answer = run_status(RunDirectory(run), this_host="login-node")
    assert answer.state == "running" and answer.label == "Running 42%"
    assert answer.model_time_s == 150.0 and answer.duration_s == 360.0


def test_a_job_squeue_no_longer_lists_is_stopped_without_a_final_record(
    tmp_path: Path, no_proc, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_command(tmp_path, "squeue",
                  'echo "slurm_load_jobs error: Invalid job id specified" >&2\nexit 1\n', monkeypatch)
    run = make_run(tmp_path)
    _segment(run, host="gpu17.cluster.example", heartbeat_ago=90.0, scheduler={"kind": "slurm", "job_id": "123456"})
    answer = run_status(RunDirectory(run), this_host="login-node")
    assert answer.state == "lost"
    assert answer.label == "Stopped without a final record (last frame 2, 120 s)"
    assert answer.scheduler_state == "gone" and "123456" in answer.detail
    assert {row["id"] for row in answer.actions} >= {"start-over"}


def test_squeue_with_empty_output_means_the_job_has_left(tmp_path: Path, no_proc, monkeypatch) -> None:
    _fake_command(tmp_path, "squeue", "exit 0\n", monkeypatch)
    run = make_run(tmp_path)
    _segment(run, host="gpu17.cluster.example", heartbeat_ago=90.0, scheduler={"kind": "slurm", "job_id": "7"})
    assert run_status(RunDirectory(run), this_host="login-node").state == "lost"


def test_a_job_squeue_still_lists_stays_stale_and_says_so(tmp_path: Path, no_proc, monkeypatch) -> None:
    _fake_command(tmp_path, "squeue", 'echo RUNNING\n', monkeypatch)
    run = make_run(tmp_path)
    _segment(run, host="gpu17.cluster.example", heartbeat_ago=90.0, scheduler={"kind": "slurm", "job_id": "123456"})
    answer = run_status(RunDirectory(run), this_host="login-node")
    assert answer.state == "stale" and answer.scheduler_state == "RUNNING"
    assert "lists job 123456 as RUNNING" in answer.detail


def test_a_pbs_job_qstat_does_not_know_is_stopped_without_a_final_record(
    tmp_path: Path, no_proc, monkeypatch,
) -> None:
    _fake_command(tmp_path, "qstat", 'echo "qstat: Unknown Job Id 99.pbs" >&2\nexit 153\n', monkeypatch)
    run = make_run(tmp_path)
    _segment(run, host="gpu3.cluster.example", heartbeat_ago=300.0, scheduler={"kind": "pbs", "job_id": "99.pbs"})
    assert run_status(RunDirectory(run), this_host="login-node").state == "lost"


def test_a_finished_pbs_job_is_gone(tmp_path: Path, no_proc, monkeypatch) -> None:
    _fake_command(tmp_path, "qstat", 'echo "Job Id: 99.pbs"\necho "    job_state = F"\n', monkeypatch)
    run = make_run(tmp_path)
    _segment(run, host="gpu3.cluster.example", heartbeat_ago=300.0, scheduler={"kind": "pbs", "job_id": "99.pbs"})
    assert run_status(RunDirectory(run), this_host="login-node").state == "lost"


def test_an_end_record_decides_whatever_the_heartbeat(tmp_path: Path, no_proc) -> None:
    run = make_run(tmp_path)
    _segment(run, host="gpu17.cluster.example", heartbeat_ago=86400.0, end={
        "reason": "completed", "rc": 0, "receipt": None, "step": 60, "model_time_s": 360.0,
        "frames_published": 3, "restarts_published": 1, "signal": None, "request": None, "message": None,
    })
    answer = run_status(RunDirectory(run), this_host="login-node")
    assert answer.state == "finished" and answer.model_time_s == 360.0


def test_this_hosts_run_is_checked_by_pid_and_start_time(tmp_path: Path) -> None:
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        identity = process_identity(sleeper.pid)
        host = identity["host"]
        run = make_run(tmp_path, name="alive")
        _segment(run, host=host, heartbeat_ago=500.0, supervisor=identity, child=identity)
        answer = run_status(RunDirectory(run))
        # Alive by pid and start time: running, even with an old record (said in the detail).
        assert answer.state == "running" and answer.local is True
        assert "no record for" in answer.detail

        reused = make_run(tmp_path, name="reused")
        wrong = dict(identity, start_ticks=identity["start_ticks"] + 12345)
        _segment(reused, host=host, heartbeat_ago=5.0, supervisor=wrong, child=wrong)
        answer = run_status(RunDirectory(reused))
        # The pid exists, but it is another process: the run is gone.
        assert answer.state == "lost" and answer.label.startswith("Stopped without a final record")

        waiting = make_run(tmp_path, name="waiting")
        segment = _segment(waiting, host=host, heartbeat_ago=2.0, state="waiting-for-gpu",
                           supervisor=identity, child={}, frames=0)
        (segment / "events.jsonl").unlink()
        state = json.loads((segment / "segment.json").read_text())
        state["child"] = None
        state["queue"] = {"ahead": 1, "holder": {"rundir": "/runs/other", "pid": 1}}
        write_json_atomic(segment / "segment.json", state)
        answer = run_status(RunDirectory(waiting))
        assert answer.state == "waiting" and answer.label == "Waiting for GPU"
        assert "other" in answer.detail and "1 run(s) ahead" in answer.detail
    finally:
        sleeper.kill()
        sleeper.wait()


def test_a_ready_run_and_a_view_only_directory(tmp_path: Path) -> None:
    run = make_run(tmp_path, name="ready")
    answer = run_status(RunDirectory(run))
    assert answer.state == "ready" and [row["id"] for row in answer.actions] == ["start", "job-script"]
    assert answer.actions[0]["command"] == f"gpuwm-cm1 run {run} --detach"
    plain = tmp_path / "plain"
    (plain / "frames").mkdir(parents=True)
    listed = status_module.list_runs(tmp_path)
    by_name = {row.name: row for row in listed}
    assert by_name["ready"].state == "ready"
    assert by_name["plain"].view_only is True


def test_status_wait_answers_once_the_run_has_left_the_active_states(tmp_path: Path, monkeypatch, capsys) -> None:
    """``gpuwm-cm1 status RUNDIR --wait``: the guides' "wait for it to finish or pause"."""

    import argparse
    import time as time_module

    from cm1core.cm1run import doors
    from cm1core.cm1run import status as status_module

    states = iter(["running", "running", "waiting", "paused"])
    seen = []

    def fake_status(run):  # noqa: ANN001 - the RunDirectory the door made
        state = next(states)
        seen.append(state)
        return status_module.RunStatus(rundir=str(tmp_path), name="r", state=state,
                                       label=status_module.LABELS[state])

    monkeypatch.setattr(status_module, "run_status", fake_status)
    monkeypatch.setattr("cm1core.cm1run.rundir.RunDirectory", lambda path: path)
    monkeypatch.setattr(time_module, "sleep", lambda seconds: None)
    arguments = argparse.Namespace(rundirs=[tmp_path], root=None, json=True, wait=True)
    assert doors.status_door(arguments) == 0
    assert seen == ["running", "running", "waiting", "paused"]
    assert '"state": "paused"' in capsys.readouterr().out
