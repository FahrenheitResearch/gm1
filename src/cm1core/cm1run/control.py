"""Stop, pause, terminate and force-stop a run.

``stop`` and ``pause`` write a file into ``RUNDIR/control``; the runtime
checks the directory once per large step (:mod:`cm1core.run_control`), so a
request works from any machine that sees the directory: the GUI, a login
node, another terminal.  No parent process is involved.

``terminate`` (SIGTERM) and ``force`` (SIGKILL, then the staging a killed
writer leaves) signal processes, so they work only on the host the run is
on; each checks the recorded pid *and* its ``/proc`` start time first.
``abandon`` marks a segment whose processes are gone (on another host) as
ended, without signalling anything.

Importing this module imports neither CuPy nor numpy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import time
from typing import Any

from .rundir import (
    RunDirectory,
    Segment,
    fsync_directory,
    host_name,
    local_process_alive,
    utc_now,
)
from .status import run_status

STOP = "stop"
PAUSE = "pause"
REQUESTS = (STOP, PAUSE)


class ControlError(RuntimeError):
    """A control action that cannot be done; the message says why and what to do."""


def _command(*words: str) -> str:
    return shlex.join(["gpuwm-cm1", *words])


def request(path: Path | str, name: str) -> dict[str, Any]:
    """Create ``control/<name>`` for an active run and say what will happen."""

    if name not in REQUESTS:
        raise ValueError(f"{name!r} is not one of {REQUESTS}")
    run = RunDirectory(path)
    status = run_status(run)
    if status.state not in ("waiting", "running", "stale"):
        raise ControlError(
            f"{status.name} is {status.label.lower()}; there is nothing to {name}."
        )
    run.control.mkdir(exist_ok=True)
    target = run.control / name
    payload = json.dumps({"request": name, "requested_utc": utc_now(), "host": host_name(),
                          "pid": os.getpid()}, sort_keys=True)
    handle = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(handle, (payload + "\n").encode("utf-8"))
        os.fsync(handle)
    finally:
        os.close(handle)
    fsync_directory(run.control)
    if name == STOP:
        when = (
            "It stops before it starts." if status.state == "waiting"
            else "It stops at the end of the current step, after any write in progress finishes."
        )
    else:
        when = (
            "It will pause after its first restart once it starts." if status.state == "waiting"
            else "It pauses right after its next restart is written."
        )
        following, until = status.next_restart_s, status.until_s
        if following is not None and until is not None and float(following) >= float(until):
            # One restart at the end (a namelist with rstfrq < 0), or the last one is next.
            when += f" Its next restart is its last, at the end ({float(until):g} s): it finishes there."
        elif following is not None and status.state != "waiting":
            when = f"It pauses right after its next restart is written, at {float(following):g} s of storm time."
    return {
        "action": name,
        "file": os.fspath(target),
        "state": status.state,
        "message": f"{name.capitalize()} requested. {when}",
        "command": _command(name, os.fspath(run.path)),
    }


def _latest(run: RunDirectory) -> Segment:
    segment = run.latest_segment()
    if segment is None:
        raise ControlError(f"{run.path} has not been started; there is nothing to stop.")
    return segment


def _require_local(run: RunDirectory, segment: Segment) -> dict[str, Any]:
    state = segment.state()
    host = state.get("host")
    if host is not None and host != host_name():
        job = state.get("scheduler") or {}
        cancel = {
            "slurm": f" or cancel job {job.get('job_id')} with scancel",
            "pbs": f" or cancel job {job.get('job_id')} with qdel",
        }.get(job.get("kind"), "")
        raise ControlError(
            f"The run is on {host}; signals work only there. Use "
            f"{_command('stop', os.fspath(run.path))} (it works from any machine){cancel}."
        )
    return state


def _signal(identity: dict[str, Any] | None, number: int, *, group: bool = False) -> bool:
    """Signal the recorded process (checked by pid *and* start time) on this host.

    ``group``: the whole process group when the process leads one, as the
    integration does (``gpuwm-cm1 run`` starts it in its own group), so the
    native publishers it started go with it.
    """

    if not local_process_alive(identity):
        return False
    pid = int(identity["pid"])
    try:
        if group and os.getpgid(pid) == pid:
            os.killpg(pid, number)
        else:
            os.kill(pid, number)
    except ProcessLookupError:
        return False
    return True


def terminate(path: Path | str) -> dict[str, Any]:
    """SIGTERM the run's process (local runs): it ends ``stopped`` with rc 143."""

    run = RunDirectory(path)
    segment = _latest(run)
    state = _require_local(run, segment)
    target = "child" if local_process_alive(state.get("child")) else "supervisor"
    if not _signal(state.get(target), signal.SIGTERM):
        raise ControlError(f"{run.name()} has no running process on this host.")
    return {
        "action": "terminate", "signalled": target, "pid": state[target]["pid"],
        "message": "Sent SIGTERM. The run discards the write in progress and records how it ended.",
        "command": _command("stop", os.fspath(run.path), "--terminate"),
    }


def remove_staging(segment: Segment) -> list[str]:
    """Delete the ``.incomplete/`` directories a killed writer leaves in a segment."""

    removed = []
    for staging in segment.staging():
        shutil.rmtree(staging, ignore_errors=True)
        removed.append(os.fspath(staging))
    return removed


def archive_requests(run: RunDirectory, segment: Segment, *, note: str) -> list[str]:
    """Move stop/pause files out of ``control/`` into the segment, so the next start is clean."""

    moved = []
    for name in REQUESTS:
        source = run.control / name
        if not source.exists():
            continue
        target = segment.path / f"control-{name}.{note}"
        counter = 1
        while target.exists():
            counter += 1
            target = segment.path / f"control-{name}.{note}.{counter}"
        try:
            os.replace(source, target)
            moved.append(os.fspath(target))
        except OSError:
            continue
    if moved:
        fsync_directory(run.control)
    return moved


def _mark_ended(run: RunDirectory, segment: Segment, reason: str, rc: int | None) -> None:
    ended = utc_now()

    def mutate(state: dict[str, Any]) -> None:
        state["state"] = "ended"
        state["ended_utc"] = ended
        state["reason"] = reason
        state["rc"] = rc

    segment.write_state(mutate)

    def update(record: dict[str, Any]) -> None:
        for row in record.get("segments") or []:
            if row.get("number") == segment.number:
                row.update({"ended_utc": ended, "reason": reason, "rc": rc})

    run.update(update)


def force_stop(path: Path | str, *, wait_s: float = 10.0) -> dict[str, Any]:
    """SIGKILL the run's processes (local runs), then remove the staging they leave.

    Frames and restarts already renamed into place are complete and kept; a
    write that was in progress is lost with its ``.incomplete/`` directory.
    """

    run = RunDirectory(path)
    segment = _latest(run)
    state = _require_local(run, segment)
    killed = []
    for role in ("child", "supervisor"):
        # The integration's group: a native publisher it started would
        # otherwise live on and write into the staging removed below.
        if _signal(state.get(role), signal.SIGKILL, group=role == "child"):
            killed.append(role)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline and any(
        local_process_alive(state.get(role)) for role in ("child", "supervisor")
    ):
        time.sleep(0.1)
    alive = [role for role in ("child", "supervisor") if local_process_alive(state.get(role))]
    if alive:
        raise ControlError(f"{', '.join(alive)} did not exit after SIGKILL; check the process table")
    removed = remove_staging(segment)
    if segment.state().get("state") != "ended" or killed:
        _mark_ended(run, segment, "force-stopped", -int(signal.SIGKILL) if killed else None)
    archived = archive_requests(run, segment, note="force-stopped")
    return {
        "action": "force-stop",
        "killed": killed,
        "staging_removed": removed,
        "control_archived": archived,
        "message": "Force stopped. Frames and restarts written before it are kept.",
        "command": _command("stop", os.fspath(run.path), "--force"),
    }


def abandon(path: Path | str) -> dict[str, Any]:
    """Mark the latest segment as ended without signalling anything.

    For a run whose processes are known to be gone but that left no final
    record where this machine can confirm it (another host, no scheduler
    command here).  Refused while a local process of the run is alive.
    """

    run = RunDirectory(path)
    segment = _latest(run)
    state = segment.state()
    if state.get("host") in (None, host_name()) and any(
        local_process_alive(state.get(role)) for role in ("child", "supervisor")
    ):
        raise ControlError("The run is still running on this host; stop it instead.")
    _mark_ended(run, segment, "abandoned", None)
    archived = archive_requests(run, segment, note="abandoned")
    return {
        "action": "abandon", "segment": segment.number, "control_archived": archived,
        "message": "The run is no longer tracked as running. Its frames and restarts are kept.",
        "command": _command("stop", os.fspath(run.path), "--abandon"),
    }


__all__ = [
    "ControlError",
    "PAUSE",
    "REQUESTS",
    "STOP",
    "abandon",
    "archive_requests",
    "force_stop",
    "remove_staging",
    "request",
    "terminate",
]
