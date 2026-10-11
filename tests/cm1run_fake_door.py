"""A stand-in for ``gpuwm-cm1 cartesian --integrate`` for the run-directory tests.

``gpuwm-cm1 run`` starts ``python -m <module> cartesian DECK ...`` with the
module named by ``GPUWM_CM1_RUN_CHILD_MODULE``.  This module answers the same
arguments without a GPU: it reads the segment's deck, writes a real
``gm1.run-events/v1`` log with :class:`cm1core.run_events.RunEventLog`, reads
the control directory with the shipped :class:`cm1core.run_control.RunControl`
(so stop and pause are the runtime's own code), publishes frames and restarts
through ``.incomplete/`` renames, and resumes from a restart's receipt, so the
supervisor, queue, status and control code run exactly as they do on a GPU.

A frame's ``state.bin`` is a function of the step alone, so a paused and
resumed run's frames equal an uninterrupted run's.

Environment knobs: ``FAKE_STEP_S`` (seconds per step, default 0.02),
``FAKE_CRASH_AT_STEP`` (SIGKILL itself after that step), ``FAKE_PLAN_REFUSE``
(``--plan`` refuses), ``FAKE_PRICED_PEAK_BYTES`` and ``FAKE_FRAME_BYTES``
(what ``--plan`` reports), ``FAKE_STARTUP_S`` (seconds before admission),
``FAKE_HELPER_PID_FILE`` (start a helper process in the integration's group,
as the runtime's native publishers are, and write its pid there).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from cm1core.run_control import PAUSE, STOP, RunControl, require_control_directory
from cm1core.run_events import RunEventLog, host_name


class _Clock:
    completed_steps = 0
    model_seconds = 0.0


class _Runtime:
    def __init__(self) -> None:
        self.clock = _Clock()
        self.profiler = None


def state_bytes(step: int) -> bytes:
    return hashlib.sha256(f"state@{step}".encode()).digest() * 64


def _publish(root: Path, name: str, files: dict[str, bytes]) -> Path:
    staging = root / ".incomplete" / name
    staging.mkdir(parents=True)
    for filename, data in files.items():
        (staging / filename).write_bytes(data)
    target = root / name
    os.replace(staging, target)
    return target


def _plan(arguments: argparse.Namespace) -> int:
    import tomllib

    table = tomllib.loads(arguments.config.read_text(encoding="utf-8"))["run"]
    if int(table.get("irst", 0)) == 1:
        # As the door's resume_selection does, before any pricing.
        chosen = None if arguments.restart_from is None else (
            arguments.restart_from / f"restart-{int(table['rstnum']):06d}")
        if chosen is None or not (chosen / "restart-publication.json").is_file():
            print(json.dumps({"status": "refused", "code": "CM1-CARTESIAN-RESTART-SELECTION", "gate": "fake",
                              "reason": "run.irst = 1 and no restart directory holds the selected restart",
                              "remedy": "pass --restart-from"}), file=sys.stderr)
            return 2
    if os.environ.get("FAKE_PLAN_REFUSE"):
        print(json.dumps({"status": "refused", "code": "CM1-CFG-FAKE", "gate": "fake",
                          "reason": os.environ["FAKE_PLAN_REFUSE"], "remedy": "change it"}), file=sys.stderr)
        return 2
    print(json.dumps({
        "schema": "gpuwm-cm1.cartesian-plan/v1",
        "status": "priced-no-device-opened",
        "config": {"tile_selection": {"priced_peak_bytes": int(os.environ.get("FAKE_PRICED_PEAK_BYTES", "1000"))}},
        "history_frame": {"payload_bytes": int(os.environ.get("FAKE_FRAME_BYTES", "1000"))},
    }))
    return 0


def main(argv: list[str] | None = None) -> int:
    import tomllib

    parser = argparse.ArgumentParser()
    parser.add_argument("door")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--integrate", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--history-dir", type=Path)
    parser.add_argument("--restart-dir", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--history-envelope")
    parser.add_argument("--history-initial", action="store_true")
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--sounding", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--history-publisher", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.plan:
        return _plan(arguments)
    deck = tomllib.loads(arguments.config.read_text(encoding="utf-8"))
    table = deck["run"]
    dt = float(table["dt_s"])
    total = round(float(table["duration_s"]) / dt)
    every_frame = round(float(table["history_interval_s"]) / dt)
    every_restart = round(float(table["restart_interval_s"]) / dt)
    step_s = float(os.environ.get("FAKE_STEP_S", "0.02"))
    crash_at = int(os.environ.get("FAKE_CRASH_AT_STEP", "-1"))
    time.sleep(float(os.environ.get("FAKE_STARTUP_S", "0")))

    events = RunEventLog(arguments.events)
    ended = {"done": False}

    def end(reason: str, rc: int, **fields) -> int:
        if not ended["done"]:
            ended["done"] = True
            events.emit("end", reason=reason, rc=rc, receipt=os.fspath(arguments.output),
                        step=runtime.clock.completed_steps, model_time_s=runtime.clock.model_seconds,
                        frames_published=published["frames"], restarts_published=published["restarts"],
                        signal=fields.get("signal"), request=fields.get("request"), message=fields.get("message"))
            events.close()
            arguments.output.write_text(json.dumps({"status": reason, "rc": rc}), encoding="utf-8")
        return rc

    runtime = _Runtime()
    published = {"frames": 0, "restarts": 0}
    try:
        control_dir = require_control_directory(arguments.control_dir)
    except ValueError as error:
        events.emit("refused", code="CM1-CARTESIAN-CONTROL-DIRECTORY", gate="fake", reason=str(error),
                    remedy="pass a control directory with no request in it", option="control-dir")
        return end("refused", 2, message=str(error))

    start_step, next_frame, next_restart = 0, 1, 1
    if int(table.get("irst", 0)) == 1:
        chosen = arguments.restart_from / f"restart-{int(table['rstnum']):06d}"
        receipt = json.loads((chosen / "restart-publication.json").read_text(encoding="utf-8"))
        start_step = int(receipt["step"])
        next_frame = int(receipt["next_history_index"])
        next_restart = int(receipt["restart_number"]) + 1
        if not (chosen / "state.bin").read_bytes() == state_bytes(start_step):
            return end("error", 1, message="restart state does not match its step")
    runtime.clock.completed_steps = start_step
    runtime.clock.model_seconds = start_step * dt
    arguments.history_dir.mkdir()
    arguments.restart_dir.mkdir()
    frames_expected = sum(1 for step in range(start_step + 1, total + 1) if step % every_frame == 0)
    restarts_expected = sum(1 for step in range(start_step + 1, total + 1) if step % every_restart == 0)
    events.emit(
        "admitted", config=os.fspath(arguments.config), config_sha256="0" * 64,
        grid={"nx": 8, "ny": 8, "nz": 8, "dx_m": 1000.0, "dy_m": 1000.0, "stretch_z": 0},
        microphysics="fake", steps=total, start_step=start_step, dt_s=dt, nsound=4,
        duration_s=float(table["duration_s"]), history_interval_s=float(table["history_interval_s"]),
        frames_expected=frames_expected + (1 if arguments.history_initial else 0),
        restart_interval_s=float(table["restart_interval_s"]), restarts_expected=restarts_expected,
        history_directory=os.fspath(arguments.history_dir), restart_directory=os.fspath(arguments.restart_dir),
        receipt=os.fspath(arguments.output), control_directory=os.fspath(control_dir),
        history_envelope=arguments.history_envelope or "on", history_initial=arguments.history_initial,
        host=host_name(),
    )
    events.start_heartbeat(0.2)
    control = RunControl(control_dir, events, progress_interval_s=0.2)
    if os.environ.get("FAKE_HELPER_PID_FILE"):
        # A helper process in the integration's group, as the runtime's native
        # publishers are: a force stop must not leave it behind.
        import subprocess

        helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        Path(os.environ["FAKE_HELPER_PID_FILE"]).write_text(str(helper.pid), encoding="ascii")

    def on_term(_number, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, on_term)

    def frame(step: int, index: int) -> None:
        path = _publish(arguments.history_dir, f"frame-{index:06d}", {
            "state.bin": state_bytes(step),
            "history-publication.json": json.dumps({"step": step, "model_time_s": step * dt}).encode(),
        })
        published["frames"] += 1
        events.emit("frame", index=index, step=step, model_time_s=step * dt, directory=os.fspath(path),
                    bytes=len(state_bytes(step)), restart_written=None, initial_condition=index == 0)

    def restart(step: int, number: int, next_index: int) -> None:
        path = _publish(arguments.restart_dir, f"restart-{number:06d}", {
            "state.bin": state_bytes(step),
            "restart-publication.json": json.dumps({
                "schema": "fake", "restart_number": number, "step": step, "model_time_s": step * dt,
                "next_history_index": next_index,
            }).encode(),
        })
        published["restarts"] += 1
        events.emit("restart", number=number, step=step, model_time_s=step * dt, directory=os.fspath(path),
                    bytes=len(state_bytes(step)), next_history_index=next_index)

    try:
        control.begin(runtime, total_steps=total, duration_s=float(table["duration_s"]),
                      model_start_s=start_step * dt, startup={"stages": {}, "seconds": 0.0})
        if control.stop_requested:
            return end("stopped", 0, request=STOP)
        if arguments.history_initial:
            frame(0, 0)
        for step in range(start_step + 1, total + 1):
            time.sleep(step_s)
            runtime.clock.completed_steps = step
            runtime.clock.model_seconds = step * dt
            if step == crash_at:
                os.kill(os.getpid(), signal.SIGKILL)
            request = control.after_step(step_s)
            restarted = False
            if step % every_frame == 0:
                frame(step, next_frame)
                next_frame += 1
            if step % every_restart == 0:
                restart(step, next_restart, next_frame)
                next_restart += 1
                restarted = True
            # As cuda_cartesian_runtime does: neither request ends the final
            # step, which completes the requested duration, and a pause whose
            # only restart left is the final one is refused (it completes).
            if step == total:
                continue
            if request == STOP:
                return end("stopped", 0, request=STOP)
            if request == PAUSE:
                if restarted:
                    return end("paused", 0, request=PAUSE)
                if not any(later % every_restart == 0 for later in range(step + 1, total)):
                    control.refuse_pause("its next restart is the one at the end of the run, "
                                         "so it will complete rather than pause")
        control.finishing()
        return end("completed", 0)
    except SystemExit as exit_:
        if exit_.code == 143:
            return end("stopped", 143, signal="SIGTERM")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
