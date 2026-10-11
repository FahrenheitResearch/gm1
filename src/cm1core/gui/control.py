"""``gpuwm-cm1 stop RUNDIR`` and ``gpuwm-cm1 pause RUNDIR``.

The writer's side of the run's control directory (:mod:`cm1core.run_control`):

``stop RUNDIR``
    Creates ``RUNDIR/control/stop``.  The run finishes the publication in
    flight, writes its receipt and exits 0 at the next large step.  Works on
    any host that shares the directory (a login node, a Slurm job).
``pause RUNDIR``
    Creates ``RUNDIR/control/pause``: the run stops right after its next
    restart is durable, so ``gpuwm-cm1 resume`` continues it.
``stop RUNDIR --terminate``
    SIGTERM to a run on this computer (its process group when the run leads
    one): the runtime's fallback path, which discards the publication in
    flight and exits 143 with an ``end`` record.
``stop RUNDIR --force``
    SIGKILL, then removes the ``.incomplete`` staging directories the killed
    process left.  The log then has no ``end`` record, and the run reads
    "Stopped without a final record".

A run that has already ended is left alone: no control file is created, so
the next ``resume`` does not meet a stale one.  Importing this module
imports nothing outside the standard library and :mod:`.runstate`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from .runstate import _utc_seconds, proc_start_ticks, process_alive, read_status, segment_dirs

CONTROL_DIR = "control"
STAGING = ".incomplete"


@dataclass
class ControlResult:
    ok: bool
    message: str
    fix: str = ""
    rc: int = 0
    wrote: str | None = None
    signalled: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "message": self.message, "fix": self.fix, "rc": self.rc,
                "wrote": self.wrote, "signalled": self.signalled}


def _running(rundir: Path) -> tuple[bool, dict[str, object]]:
    status = read_status(rundir)
    return status.state in ("running", "waiting", "stale"), status.as_dict()


def _ended_since(status: dict[str, object], launched_at: float | None) -> bool:
    """The latest segment has its end record, written after ``launched_at``."""

    end = status.get("end")
    if not isinstance(end, dict) or launched_at is None:
        return False
    written = _utc_seconds(end.get("time_utc"))
    return written is not None and written >= launched_at - 1.0


def request(rundir: Path, which: str, *, assume_live: bool = False,
            launched_at: float | None = None) -> ControlResult:
    """Create ``control/stop`` or ``control/pause`` for a run that has not ended.

    ``assume_live``: the caller holds the run's process and knows it is alive
    (the GUI server that launched it), even though the run directory has no
    record of it yet.  The file is then written for the run to read when its
    control loop starts; a run that has not yet opened its control directory
    refuses to start over it (run_control.require_control_directory).
    ``launched_at`` is when that process was started: once the run has
    written its end record after it, the process is only exiting, and
    nothing is written (a stray file would meet the next start).
    """

    if which not in ("stop", "pause"):
        raise ValueError(which)
    rundir = Path(rundir)
    if not (rundir / "run.json").is_file():
        return ControlResult(False, f"{rundir} is not a run folder (no run.json).",
                             "Pass the folder gpuwm-cm1 import or run created.", rc=2)
    live, status = _running(rundir)
    if assume_live and not live and _ended_since(status, launched_at):
        assume_live = False
    if not live and not assume_live:
        return ControlResult(True, f"Nothing to {which}: the run is {str(status['label']).lower()}.")
    directory = rundir / CONTROL_DIR
    directory.mkdir(exist_ok=True)
    path = directory / which
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid {os.getpid()}\n")
    words = ("It stops at the next time step and keeps everything written so far."
             if which == "stop" else
             "It stops right after its next restart is saved; Continue picks it up from there.")
    return ControlResult(True, f"Asked the run to {which}. {words}", wrote=str(path))


def _local_pid(rundir: Path) -> tuple[int | None, int | None, str]:
    status = read_status(rundir)
    detail = status.detail
    if status.state not in ("running", "waiting"):
        return None, None, f"the run is {status.label.lower()}"
    if not detail.get("local"):
        return None, None, f"the run is on {detail.get('host')}, not this computer"
    pid = detail.get("pid")
    if not pid:
        return None, None, "the run has not recorded its process yet"
    return int(pid), proc_start_ticks(int(pid)), ""


def _signal(pid: int, sig: int) -> None:
    try:
        group = os.getpgid(pid)
    except OSError:
        group = None
    if group is not None and group == pid:
        os.killpg(group, sig)
    else:
        os.kill(pid, sig)


def terminate(rundir: Path) -> ControlResult:
    pid, _, why = _local_pid(Path(rundir))
    if pid is None:
        return ControlResult(True, f"Nothing to stop now: {why}.")
    try:
        _signal(pid, signal.SIGTERM)
    except OSError as error:
        return ControlResult(False, f"Could not signal process {pid}: {error}.",
                             "Check that the run belongs to your user.", rc=2)
    return ControlResult(True, f"Sent the stop signal to process {pid}. It discards the save in "
                               "progress and writes its final record.", signalled=pid)


def force_stop(rundir: Path, wait_s: float = 5.0) -> ControlResult:
    rundir = Path(rundir)
    pid, ticks, why = _local_pid(rundir)
    if pid is None:
        result = ControlResult(True, f"Nothing to force-stop: {why}.")
    else:
        try:
            _signal(pid, signal.SIGKILL)
        except OSError as error:
            return ControlResult(False, f"Could not kill process {pid}: {error}.",
                                 "Check that the run belongs to your user.", rc=2)
        deadline = time.monotonic() + wait_s
        while process_alive(pid, ticks) and time.monotonic() < deadline:
            time.sleep(0.05)
        result = ControlResult(True, f"Killed process {pid}.", signalled=pid)
    removed = []
    segments = segment_dirs(rundir)
    if segments and (pid is not None or not _running(rundir)[0]):
        for sub in ("frames", "restarts"):
            staging = segments[-1] / sub / STAGING
            if staging.is_dir():
                shutil.rmtree(staging, ignore_errors=True)
                removed.append(str(staging))
    if removed:
        result.message += f" Removed {len(removed)} unfinished save folder(s)."
    return result


# ---- the command line

def add_commands(commands: "argparse._SubParsersAction") -> None:
    """Register ``stop`` and ``pause`` unless the parser already has them.

    The run-directory commands register the same two names with the same
    spellings (``stop RUNDIR [--terminate | --force]``, ``pause RUNDIR``);
    registered first, theirs stand and these step aside.
    """

    if {"stop", "pause"} & set(commands.choices):
        return
    command = commands.add_parser(
        "stop", help="stop a run at its next time step (control/stop); --terminate: SIGTERM; --force: SIGKILL")
    command.add_argument("rundir", type=Path)
    how = command.add_mutually_exclusive_group()
    how.add_argument("--terminate", action="store_true",
                     help="send SIGTERM to the run on this computer (the save in progress is discarded)")
    how.add_argument("--force", action="store_true",
                     help="send SIGKILL and remove the unfinished save folders it leaves")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=stop_door)

    command = commands.add_parser(
        "pause", help="stop a run right after its next restart is saved (control/pause)")
    command.add_argument("rundir", type=Path)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=pause_door)


def _report(result: ControlResult, as_json: bool) -> int:
    if as_json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        stream = sys.stdout if result.ok else sys.stderr
        print(result.message, file=stream)
        if result.fix:
            print(result.fix, file=stream)
    return result.rc


def stop_door(arguments: argparse.Namespace) -> int:
    if arguments.force:
        result = force_stop(arguments.rundir)
    elif arguments.terminate:
        result = terminate(arguments.rundir)
    else:
        result = request(arguments.rundir, "stop")
    return _report(result, arguments.json)


def pause_door(arguments: argparse.Namespace) -> int:
    return _report(request(arguments.rundir, "pause"), arguments.json)


__all__ = [
    "CONTROL_DIR",
    "ControlResult",
    "add_commands",
    "force_stop",
    "pause_door",
    "request",
    "stop_door",
    "terminate",
]
