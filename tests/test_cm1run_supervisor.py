"""``gpuwm-cm1 run | resume | stop | pause``: the supervisor, the queue and the control files.

Every run here is the real supervisor in its own process (started from a
"shell": its own session, no GUI parent), with ``cm1run_fake_door`` as the
integration.  The door stand-in uses the shipped RunEventLog and RunControl,
so stop and pause are read by the runtime's own code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import time

from cm1run_support import (
    environment,
    events_of,
    finish,
    gm1,
    make_run,
    progress_at_least,
    start,
    status,
    wait_for,
)

from cm1core.cm1run.rundir import RunDirectory
from cm1core.run_events import validate_file


def _frames_digest(path: Path) -> dict[int, str]:
    run = RunDirectory(path)
    return {
        frame.index: hashlib.sha256((frame.path / "state.bin").read_bytes()).hexdigest()
        for frame in run.frames()
    }


def test_a_run_from_a_shell_completes_and_records_itself(tmp_path: Path) -> None:
    run = make_run(tmp_path, rstfrq=-3600.0)
    env = environment(tmp_path)
    completed = gm1("run", str(run), env=env)
    assert completed.returncode == 0, completed.stderr
    events = run / "segments" / "001" / "events.jsonl"
    assert validate_file(events) == []
    kinds = [record["kind"] for record in events_of(run)]
    assert kinds[0] == "admitted" and kinds[-1] == "end"
    answer = status(run, env)
    assert answer["state"] == "finished" and answer["label"] == "Finished"
    assert answer["frames"] == 7 and answer["percent"] == 100.0
    record = json.loads((run / "run.json").read_text())
    row = record["segments"][0]
    assert row["kind"] == "fresh" and row["reason"] == "completed" and row["rc"] == 0
    assert row["restart_interval_s"] == 360.0 and "one restart at the end" in row["restart_rule"]
    assert record["gpu"]["uuid"] == "GPU-fake-0000-0000"
    state = json.loads((run / "segments" / "001" / "segment.json").read_text())
    assert state["state"] == "ended" and state["child"]["start_ticks"] is not None
    # The segment's deck is deck.toml with only the run table's schedule changed.
    deck = (run / "segments" / "001" / "deck.toml").read_text()
    assert "restart_interval_s = 360.0" in deck and "irst" not in deck
    assert os.listdir(run / "control") == []
    # Nothing left to run: --continue says so and exits 0; a plain run refuses.
    again = gm1("run", str(run), "--continue", env=env)
    assert again.returncode == 0 and "nothing left" in again.stderr
    refused = gm1("run", str(run), env=env)
    assert refused.returncode == 2 and "already run" in refused.stderr and "--start-over" in refused.stderr


def test_a_shell_run_stops_on_the_control_file_alone(tmp_path: Path) -> None:
    run = make_run(tmp_path, duration=1800.0, restart=600.0)
    env = environment(tmp_path, FAKE_STEP_S="0.03")
    process = start("run", str(run), env=env, log=tmp_path / "run.log")
    wait_for(lambda: progress_at_least(run, 20), what="stepping")
    stopped = gm1("stop", str(run), "--json", env=env)
    assert stopped.returncode == 0, stopped.stderr
    assert json.loads(stopped.stdout)["command"].startswith("gpuwm-cm1 stop ")
    assert finish(process) == 0
    end = events_of(run)[-1]
    assert end["kind"] == "end" and end["reason"] == "stopped" and end["request"] == "stop"
    assert end["step"] < 300
    answer = status(run, env)
    assert answer["state"] == "stopped" and "stop request" in answer["detail"]
    # The consumed request is filed with the segment; the next start is clean.
    assert os.listdir(run / "control") == []
    assert (run / "segments" / "001" / "control-stop.consumed").exists()
    assert not list((run / "segments" / "001").rglob(".incomplete/*"))
    # It stopped before its first restart (600 s), so it can only start over.
    actions = {row["id"] for row in answer["actions"]}
    assert "start-over" in actions and "continue" not in actions


def test_pause_then_resume_equals_the_uninterrupted_run(tmp_path: Path) -> None:
    straight = make_run(tmp_path, name="straight", duration=720.0, restart=240.0)
    paused = make_run(tmp_path, name="paused", duration=720.0, restart=240.0)
    env = environment(tmp_path, FAKE_STEP_S="0.03")
    assert gm1("run", str(straight), env=env).returncode == 0
    process = start("run", str(paused), env=env, log=tmp_path / "paused.log")
    wait_for(lambda: progress_at_least(paused, 5), what="stepping")
    asked = gm1("pause", str(paused), env=env)
    assert asked.returncode == 0 and "at 240 s of storm time" in asked.stdout, asked.stdout
    assert finish(process) == 0
    answer = status(paused, env)
    assert answer["state"] == "paused", answer
    assert answer["last_restart"]["model_time_s"] < 720.0
    resumed = gm1("resume", str(paused), env=env)
    assert resumed.returncode == 0, resumed.stderr
    assert status(paused, env)["state"] == "finished"
    row = json.loads((paused / "run.json").read_text())["segments"][1]
    assert row["kind"] == "resume" and row["resumed_from"]["segment"] == 1
    deck = (paused / "segments" / "002" / "deck.toml").read_text()
    assert "irst = 1" in deck and f"rstnum = {row['resumed_from']['restart']}" in deck
    assert _frames_digest(paused) == _frames_digest(straight)
    assert [frame.index for frame in RunDirectory(paused).frames()] == list(range(13))


def test_resume_extends_a_finished_run_and_checks_the_end_time(tmp_path: Path) -> None:
    run = make_run(tmp_path, duration=360.0, restart=120.0)
    env = environment(tmp_path)
    assert gm1("run", str(run), env=env).returncode == 0
    nothing = gm1("resume", str(run), env=env)
    assert nothing.returncode == 2 and "--until" in nothing.stderr
    off_grid = gm1("resume", str(run), "--until", "400", env=env)
    assert off_grid.returncode == 2 and "480" in off_grid.stderr
    extended = gm1("resume", str(run), "--until", "8min", env=env)
    assert extended.returncode == 0, extended.stderr
    answer = status(run, env)
    assert answer["state"] == "finished" and answer["model_time_s"] == 480.0
    assert [frame.index for frame in RunDirectory(run).frames()] == list(range(9))


def test_a_second_run_on_the_gpu_waits_in_line_then_runs(tmp_path: Path) -> None:
    first = make_run(tmp_path, name="first", duration=600.0, restart=600.0)
    second = make_run(tmp_path, name="second", duration=120.0, restart=120.0)
    third = make_run(tmp_path, name="third", duration=120.0, restart=120.0)
    env = environment(tmp_path, FAKE_STEP_S="0.04")
    one = start("run", str(first), env=env, log=tmp_path / "first.log")
    wait_for(lambda: progress_at_least(first, 2), what="the first run stepping")
    two = start("run", str(second), env=env, log=tmp_path / "second.log")
    waiting = wait_for(
        lambda: (lambda answer: answer if answer["state"] == "waiting" and answer["queue"] else None)(
            status(second, env)),
        what="the second run waiting",
    )
    assert waiting["label"] == "Waiting for GPU"
    assert "first" in waiting["detail"]
    three = start("run", str(third), env=env, log=tmp_path / "third.log")
    wait_for(lambda: (status(third, env).get("queue") or {}).get("ahead") == 1, what="third in line")
    assert finish(one) == 0 and finish(two) == 0 and finish(three) == 0
    starts = {}
    for name, path in (("first", first), ("second", second), ("third", third)):
        assert status(path, env)["state"] == "finished"
        starts[name] = json.loads((path / "segments" / "001" / "segment.json").read_text())["queue"]["acquired_utc"]
    assert starts["first"] < starts["second"] < starts["third"]
    # The first run was over before the second took the GPU.
    first_end = events_of(first)[-1]["time_utc"]
    assert first_end <= starts["second"]


def test_stop_while_waiting_ends_before_anything_runs(tmp_path: Path) -> None:
    first = make_run(tmp_path, name="first", duration=600.0, restart=600.0)
    second = make_run(tmp_path, name="second")
    env = environment(tmp_path, FAKE_STEP_S="0.04")
    one = start("run", str(first), env=env, log=tmp_path / "first.log")
    wait_for(lambda: progress_at_least(first, 2), what="stepping")
    two = start("run", str(second), env=env, log=tmp_path / "second.log")
    wait_for(lambda: status(second, env)["state"] == "waiting", what="waiting")
    assert gm1("stop", str(second), env=env).returncode == 0
    assert finish(two) == 0
    answer = status(second, env)
    assert answer["state"] == "stopped" and "before it started" in answer["detail"]
    assert validate_file(second / "segments" / "001" / "events.jsonl") == []
    assert not (second / "segments" / "001" / "frames").exists()
    assert gm1("stop", str(first), env=env).returncode == 0
    assert finish(one) == 0
    # Nothing ran, so starting it again needs no --start-over.
    assert gm1("run", str(second), env=env).returncode == 0
    assert status(second, env)["state"] == "finished"


def test_a_stop_made_before_the_runtime_reads_its_directory_is_handed_on(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    env = environment(tmp_path, FAKE_STARTUP_S="1.5")
    process = start("run", str(run), env=env, log=tmp_path / "run.log")
    wait_for(lambda: (run / "segments" / "001" / "segment.json").exists()
             and json.loads((run / "segments" / "001" / "segment.json").read_text()).get("child"),
             what="the child")
    (run / "control" / "stop").touch()
    assert finish(process) == 0
    records = events_of(run)
    end = records[-1]
    # The runtime never saw the request before it admitted its directory (it
    # would have refused to start); it got it right after, and stopped.
    assert not any(record["kind"] == "refused" for record in records)
    assert end["reason"] == "stopped" and end["request"] == "stop" and end["step"] < 30
    assert status(run, env)["state"] == "stopped"


def test_a_request_left_by_an_earlier_run_is_set_aside(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    (run / "control" / "stop").touch()
    old = time.time() - 3600
    os.utime(run / "control" / "stop", (old, old))
    env = environment(tmp_path)
    completed = gm1("run", str(run), env=env)
    assert completed.returncode == 0 and "set aside" in completed.stderr
    assert status(run, env)["state"] == "finished"
    assert list((run / "logs").glob("control-stop.stale-*"))


def test_a_crash_reads_as_stopped_without_a_final_record_and_continues(tmp_path: Path) -> None:
    run = make_run(tmp_path, duration=360.0, restart=120.0)
    env = environment(tmp_path, FAKE_CRASH_AT_STEP="45")
    completed = gm1("run", str(run), env=env)
    assert completed.returncode == 128 + signal.SIGKILL
    answer = status(run, env)
    assert answer["state"] == "lost"
    # Killed at step 45 (270 s): frames every 60 s, so the last is frame 4 at 240 s.
    assert answer["label"] == "Stopped without a final record (last frame 4, 240 s)"
    assert "signal 9" in answer["detail"]
    assert {row["id"] for row in answer["actions"]} >= {"continue"}
    env.pop("FAKE_CRASH_AT_STEP")
    resumed = gm1("resume", str(run), env=env)
    assert resumed.returncode == 0, resumed.stderr
    assert status(run, env)["state"] == "finished"
    assert [frame.index for frame in RunDirectory(run).frames()] == list(range(7))


def test_sigterm_to_the_supervisor_stops_cleanly_then_a_second_ends_it(tmp_path: Path) -> None:
    run = make_run(tmp_path, duration=1800.0, restart=600.0)
    env = environment(tmp_path, FAKE_STEP_S="0.03")
    process = start("run", str(run), env=env, log=tmp_path / "run.log")
    wait_for(lambda: progress_at_least(run, 10), what="stepping")
    process.send_signal(signal.SIGTERM)
    assert finish(process) == 0
    end = events_of(run)[-1]
    assert end["reason"] == "stopped" and end["request"] == "stop" and end["rc"] == 0

    again = make_run(tmp_path, name="again", duration=1800.0, restart=600.0)
    env_slow = environment(tmp_path, FAKE_STEP_S="0.03", FAKE_STARTUP_S="0")
    process = start("run", str(again), env=env_slow, log=tmp_path / "again.log")
    wait_for(lambda: progress_at_least(again, 10), what="stepping")
    # Make the control file unreadable to the stand-in's next poll by
    # signalling twice at once: the second is forwarded as SIGTERM.
    process.send_signal(signal.SIGTERM)
    time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    rc = finish(process)
    end = events_of(again)[-1]
    assert end["reason"] == "stopped"
    assert rc in (0, 143)


def test_force_stop_kills_and_removes_unfinished_writes(tmp_path: Path) -> None:
    run = make_run(tmp_path, duration=1800.0, restart=600.0)
    env = environment(tmp_path, FAKE_STEP_S="0.05")
    process = start("run", str(run), env=env, log=tmp_path / "run.log")
    wait_for(lambda: progress_at_least(run, 5), what="stepping")
    staging = run / "segments" / "001" / "frames" / ".incomplete" / "frame-999999"
    staging.mkdir(parents=True)
    forced = gm1("stop", str(run), "--force", "--json", env=env)
    assert forced.returncode == 0, forced.stderr
    result = json.loads(forced.stdout)
    assert set(result["killed"]) == {"child", "supervisor"}
    assert result["staging_removed"]
    finish(process)
    assert not staging.parent.exists()
    answer = status(run, env)
    assert answer["state"] == "stopped" and "Force stopped" in answer["detail"]


def test_the_gpu_memory_check_refuses_with_the_numbers(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    env = environment(tmp_path, FAKE_SMI_USED_MIB="9000", FAKE_PRICED_PEAK_BYTES=str(3 * 10**9))
    completed = gm1("run", str(run), env=env)
    assert completed.returncode == 2
    answer = status(run, env)
    assert answer["state"] == "failed"
    assert "GPU memory" in answer["detail"] and "free" in answer["detail"]
    records = events_of(run)
    assert [record["kind"] for record in records] == ["refused", "end"]
    assert records[0]["code"] == "GM1-RUN-GPU-MEMORY"
    # --ignore-estimate starts it anyway, and says so.
    forced = gm1("run", str(run), "--ignore-estimate", env=env)
    assert forced.returncode == 0 and "not enforced" in forced.stderr


def test_the_disk_check_refuses_before_a_segment_exists(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    env = environment(tmp_path, FAKE_FRAME_BYTES=str(10**15))
    completed = gm1("run", str(run), "--json", env=env)
    assert completed.returncode == 2
    refusal = json.loads(completed.stderr)
    assert refusal["code"] == "GM1-RUN-DISK" and "GB" in refusal["reason"]
    assert not (run / "segments").exists()
    assert status(run, env)["state"] == "ready"


def test_a_refused_configuration_is_reported_from_the_plan(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    env = environment(tmp_path, FAKE_PLAN_REFUSE="stretch_x = 1 is not admitted")
    completed = gm1("run", str(run), env=env)
    assert completed.returncode == 2 and "stretch_x = 1 is not admitted" in completed.stderr


def test_import_refusals_and_acknowledgments_gate_the_start(tmp_path: Path) -> None:
    refused = make_run(tmp_path, name="refused", import_block={
        "result": "refused", "refused": ["ptype = 26 is NSSL 2-moment, which gpuwm-cm1 does not have"],
        "acknowledgments_needed": []})
    env = environment(tmp_path)
    answer = status(refused, env)
    assert answer["state"] == "imported" and "NSSL" in answer["detail"]
    completed = gm1("run", str(refused), env=env)
    assert completed.returncode == 2 and "NSSL" in completed.stderr

    ack = make_run(tmp_path, name="ack", import_block={
        "result": "acknowledge", "refused": [],
        "acknowledgments_needed": [{"id": "ptype-3", "text": "Thompson has run only a 120-s column"}]})
    assert status(ack, env)["state"] == "imported"
    completed = gm1("run", str(ack), env=env)
    assert completed.returncode == 2 and "acknowledgment" in completed.stderr
    record = json.loads((ack / "run.json").read_text())
    record["acknowledgments"] = ["ptype-3"]
    (ack / "run.json").write_text(json.dumps(record))
    assert status(ack, env)["state"] == "ready"
    assert gm1("run", str(ack), env=env).returncode == 0


def test_detach_returns_at_once_and_the_run_goes_on(tmp_path: Path) -> None:
    run = make_run(tmp_path, duration=600.0, restart=600.0)
    env = environment(tmp_path, FAKE_STEP_S="0.03")
    began = time.monotonic()
    completed = gm1("run", str(run), "--detach", "--json", env=env)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert time.monotonic() - began < 20
    assert result["segment"] == 1 and result["command"].startswith("gpuwm-cm1 run ")
    assert "--detach" not in result["command"]
    assert Path(result["log"]).parent == run / "logs"
    wait_for(lambda: status(run, env)["state"] == "finished", timeout=60, what="the detached run")
    # A second start while it runs would have been refused; after it, resume says there is nothing left.
    assert "nothing" in gm1("run", str(run), "--continue", env=env).stderr


def test_a_running_run_refuses_a_second_start(tmp_path: Path) -> None:
    run = make_run(tmp_path, duration=1800.0, restart=600.0)
    env = environment(tmp_path, FAKE_STEP_S="0.03")
    process = start("run", str(run), env=env, log=tmp_path / "run.log")
    wait_for(lambda: progress_at_least(run, 3), what="stepping")
    second = gm1("run", str(run), "--start-over", env=env)
    assert second.returncode == 2 and "already" in second.stderr
    assert gm1("stop", str(run), env=env).returncode == 0
    assert finish(process) == 0
    # stop on a run that is not running is refused with the reason.
    late = gm1("stop", str(run), env=env)
    assert late.returncode == 2 and "nothing to stop" in late.stderr


def test_the_chosen_gpu_is_the_childs_only_card(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    env = environment(tmp_path)
    missing = gm1("run", str(run), "--gpu", "3", env=env)
    assert missing.returncode == 2 and "there is no GPU '3'" in missing.stderr
    assert gm1("run", str(run), "--gpu", "0", env=env).returncode == 0
    state = json.loads((run / "segments" / "002" / "segment.json").read_text()) if (
        run / "segments" / "002").exists() else json.loads((run / "segments" / "001" / "segment.json").read_text())
    assert state["gpu"]["cuda_visible_devices"] == "GPU-fake-0000-0000"
    assert state["gpu"]["chosen_by"] == "--gpu"


def test_run_scheduler_job_is_recorded(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    env = environment(tmp_path, SLURM_JOB_ID="4242", SLURM_CLUSTER_NAME="test")
    assert gm1("run", str(run), env=env).returncode == 0
    state = json.loads((run / "segments" / "001" / "segment.json").read_text())
    assert state["scheduler"]["kind"] == "slurm" and state["scheduler"]["job_id"] == "4242"
    assert json.loads((run / "run.json").read_text())["scheduler"]["job_id"] == "4242"


def test_a_pause_that_falls_on_the_last_restart_reads_finished(tmp_path: Path) -> None:
    # rstfrq <= 0: the run writes one restart, at its end, so a pause means "at the end".
    run = make_run(tmp_path, duration=600.0, restart=600.0, rstfrq=-3600.0)
    env = environment(tmp_path, FAKE_STEP_S="0.03")
    process = start("run", str(run), env=env, log=tmp_path / "run.log")
    wait_for(lambda: progress_at_least(run, 5), what="stepping")
    answer = status(run, env)
    assert answer["next_restart_s"] == 600.0 and answer["until_s"] == 600.0
    pause = next(row for row in answer["actions"] if row["id"] == "pause")
    assert pause["label"] == "Pause at next restart (600 s)"
    paused = gm1("pause", str(run), env=env)
    assert paused.returncode == 0
    # Said up front: with one restart, at the end, a pause means the run finishes.
    assert "its last, at the end (600 s): it finishes there" in paused.stdout
    assert finish(process) == 0
    # The runtime completes the final step rather than pause on it (the real
    # door's end record says completed, with a warning that the pause is unused).
    records = events_of(run)
    assert records[-1]["reason"] == "completed"
    assert any(record["kind"] == "warning" and record["code"] == "GM1-RUN-CONTROL-PAUSE-UNAVAILABLE"
               for record in records)
    answer = status(run, env)
    assert answer["state"] == "finished" and answer["detail"] == "Finished at 600 s of storm time."


def test_the_gpu_lock_outlives_a_supervisor_that_leaves_before_its_run(tmp_path: Path, monkeypatch) -> None:
    import fcntl
    import subprocess
    import sys

    from cm1core.cm1run.launch import GpuQueue

    monkeypatch.setenv("GPUWM_GPU_LOCK_ROOT", str(tmp_path / "locks"))
    queue = GpuQueue("GPU-test-lock", {"rundir": str(tmp_path)})
    queue.join()
    assert queue.try_acquire()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=(queue.fd,))
    try:
        queue.release()  # the supervisor leaves; the integration is still running

        def held() -> bool:
            fd = os.open(queue.path, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            finally:
                os.close(fd)
            return False

        assert held(), "releasing the supervisor's copy freed the card while its run still held it"
        child.kill()
        child.wait()
        assert not held()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_force_stop_takes_the_integrations_helpers_with_it(tmp_path: Path) -> None:
    run = make_run(tmp_path, duration=1800.0, restart=600.0)
    helper = tmp_path / "helper.pid"
    env = environment(tmp_path, FAKE_STEP_S="0.05", FAKE_HELPER_PID_FILE=str(helper))
    process = start("run", str(run), env=env, log=tmp_path / "run.log")
    wait_for(lambda: progress_at_least(run, 5) and helper.exists(), what="stepping with a helper")
    pid = int(helper.read_text())
    os.kill(pid, 0)
    forced = gm1("stop", str(run), "--force", "--json", env=env)
    assert forced.returncode == 0, forced.stderr
    finish(process)

    def gone() -> bool:
        try:
            state = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return True
        return state[state.rfind(")") + 2:].split()[0] in ("Z", "X")

    wait_for(gone, timeout=10, what="the helper to be killed with the integration")
