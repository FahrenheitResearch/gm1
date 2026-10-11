"""The ``gpuwm-cm1 run | resume | status | stop | pause | jobscript`` doors.

Every GUI action on a run is one of these commands, and each JSON answer
carries the ``command`` that reproduces it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

EXIT_REFUSED = 2


def _emit(arguments: argparse.Namespace, payload: dict[str, Any], text: str) -> None:
    if getattr(arguments, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif text:
        print(text)


def _refused(arguments: argparse.Namespace, command: str, code: str, reason: str, remedy: str) -> int:
    if getattr(arguments, "json", False):
        print(json.dumps({"status": "refused", "code": code, "reason": reason, "remedy": remedy},
                         indent=2, sort_keys=True), file=sys.stderr)
    else:
        print(f"gpuwm-cm1 {command}: {reason}\n  What to do: {remedy}", file=sys.stderr)
    return EXIT_REFUSED


def _start_request(arguments: argparse.Namespace, command: str):
    from .launch import StartRequest, parse_seconds

    until = None if arguments.until is None else parse_seconds(arguments.until)
    return StartRequest(
        command=command,
        start_over=bool(getattr(arguments, "start_over", False)),
        continue_run=bool(getattr(arguments, "continue_run", False)),
        from_restart=getattr(arguments, "from_restart", None),
        from_segment=getattr(arguments, "from_segment", None),
        until_s=until,
        gpu=arguments.gpu,
        wait=not arguments.no_wait,
        ignore_estimate=arguments.ignore_estimate,
        library=arguments.library,
        history_publisher=arguments.history_publisher,
    )


def _start_words(arguments: argparse.Namespace, command: str) -> list[str]:
    """The command's words after ``gpuwm-cm1``, without ``--detach`` (what the supervisor runs)."""

    words = [command, os.fspath(Path(arguments.rundir).expanduser().absolute())]
    if getattr(arguments, "continue_run", False):
        words.append("--continue")
    if getattr(arguments, "start_over", False):
        words.append("--start-over")
    if getattr(arguments, "from_restart", None) is not None:
        words += ["--from-restart", str(arguments.from_restart)]
    if getattr(arguments, "from_segment", None) is not None:
        words += ["--from-segment", str(arguments.from_segment)]
    if arguments.until is not None:
        words += ["--until", str(arguments.until)]
    if arguments.gpu is not None:
        words += ["--gpu", str(arguments.gpu)]
    if arguments.no_wait:
        words.append("--no-wait")
    if arguments.ignore_estimate:
        words.append("--ignore-estimate")
    if arguments.library is not None:
        words += ["--library", os.fspath(Path(arguments.library).expanduser().absolute())]
    if arguments.history_publisher is not None:
        words += ["--history-publisher", os.fspath(Path(arguments.history_publisher).expanduser().absolute())]
    return words


def _start(arguments: argparse.Namespace, command: str) -> int:
    from .launch import LaunchRefusal, spawn_detached, supervise
    from .rundir import RunDirectoryError

    try:
        request = _start_request(arguments, command)
    except ValueError as error:
        return _refused(arguments, command, "GM1-RUN-ARGUMENT", str(error), "correct the argument")
    words = _start_words(arguments, command)
    try:
        if arguments.detach:
            result = spawn_detached(arguments.rundir, words, request)
            _emit(arguments, result, (
                f"{result.get('message', '')}\n"
                + (f"Supervisor pid {result['pid']}; its output: {result['log']}\n" if result.get("pid") else "")
                + f"Follow it with: {shlex.join(['gpuwm-cm1', 'status', words[1]])}"
            ).strip())
            if not result.get("started") and result.get("rc"):
                return int(result["rc"])
            return 0
        return supervise(arguments.rundir, request)
    except LaunchRefusal as refusal:
        return _refused(arguments, command, refusal.code, refusal.reason, refusal.remedy)
    except RunDirectoryError as error:
        return _refused(arguments, command, "GM1-RUN-DIRECTORY", str(error),
                        "pass a run directory made by gpuwm-cm1 import")


def run_door(arguments: argparse.Namespace) -> int:
    return _start(arguments, "run")


def resume_door(arguments: argparse.Namespace) -> int:
    return _start(arguments, "resume")


def status_door(arguments: argparse.Namespace) -> int:
    import time

    from .rundir import RunDirectory, RunDirectoryError
    from .status import ACTIVE_STATES, format_status, list_runs, run_status

    def collect() -> list[Any]:
        found = []
        if arguments.root is not None:
            found.extend(list_runs(arguments.root))
        for path in arguments.rundirs:
            found.append(run_status(RunDirectory(path)))
        return found

    try:
        statuses = collect()
        # --wait: answer once no run named here is waiting for the GPU, running or stale (a script's, or a
        # guide's, "wait for it to finish or pause").
        while getattr(arguments, "wait", False) and any(status.state in ACTIVE_STATES for status in statuses):
            time.sleep(2.0)
            statuses = collect()
    except RunDirectoryError as error:
        return _refused(arguments, "status", "GM1-RUN-DIRECTORY", str(error),
                        "pass a run directory made by gpuwm-cm1 import, or --root with a folder of them")
    if not statuses and arguments.root is None and not arguments.rundirs:
        return _refused(arguments, "status", "GM1-RUN-ARGUMENT", "no run directory given",
                        "pass one or more run directories, or --root with a folder of them")
    if arguments.json:
        payload: Any = [status.as_dict() for status in statuses]
        if len(arguments.rundirs) == 1 and arguments.root is None:
            payload = payload[0]
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print("\n".join(format_status(status) for status in statuses) or "No runs.")
    return 0


def stop_door(arguments: argparse.Namespace) -> int:
    from .control import ControlError, abandon, force_stop, request, terminate
    from .rundir import RunDirectoryError

    try:
        if arguments.force:
            result = force_stop(arguments.rundir)
        elif arguments.terminate:
            result = terminate(arguments.rundir)
        elif arguments.abandon:
            result = abandon(arguments.rundir)
        else:
            result = request(arguments.rundir, "stop")
    except (ControlError, RunDirectoryError) as error:
        return _refused(arguments, "stop", "GM1-RUN-CONTROL", str(error), "see gpuwm-cm1 status RUNDIR")
    _emit(arguments, result, result["message"])
    return 0


def pause_door(arguments: argparse.Namespace) -> int:
    from .control import ControlError, request
    from .rundir import RunDirectoryError

    try:
        result = request(arguments.rundir, "pause")
    except (ControlError, RunDirectoryError) as error:
        return _refused(arguments, "pause", "GM1-RUN-CONTROL", str(error), "see gpuwm-cm1 status RUNDIR")
    _emit(arguments, result, result["message"])
    return 0


def jobscript_door(arguments: argparse.Namespace) -> int:
    from .jobscript import JobScriptError, render, write
    from .launch import parse_seconds
    from .rundir import RunDirectory, RunDirectoryError

    try:
        options = dict(
            time_limit=arguments.time, partition=arguments.partition, account=arguments.account,
            qos=arguments.qos, cpus=arguments.cpus, memory=arguments.mem,
            until_s=None if arguments.until is None else parse_seconds(arguments.until),
            gpu_type=arguments.gpu_type, setup=arguments.setup, python=arguments.python,
            library=None if arguments.library is None else os.fspath(Path(arguments.library).absolute()),
            history_publisher=None if arguments.history_publisher is None else os.fspath(
                Path(arguments.history_publisher).absolute()),
        )
        if arguments.write:
            target = write(arguments.rundir, arguments.scheduler, **options)
            text = target.read_text(encoding="utf-8")
        else:
            text = render(arguments.rundir, arguments.scheduler, **options)
            RunDirectory(arguments.rundir).logs.mkdir(exist_ok=True)
            target = None
    except (JobScriptError, RunDirectoryError, ValueError) as error:
        return _refused(arguments, "jobscript", "GM1-JOBSCRIPT", str(error), "correct the argument")
    words = ["gpuwm-cm1", "jobscript", os.fspath(Path(arguments.rundir).absolute()),
             "--scheduler", arguments.scheduler]
    if arguments.json:
        _emit(arguments, {"script": text, "file": None if target is None else os.fspath(target),
                          "command": shlex.join(words)}, "")
    elif target is not None:
        submit = "sbatch" if arguments.scheduler == "slurm" else "qsub"
        print(f"Wrote {target}\nSubmit it with: {submit} {shlex.quote(os.fspath(target))}")
    else:
        sys.stdout.write(text)
    return 0


# ----------------------------------------------------------------- parser


def _start_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("rundir", type=Path, help="the run directory")
    command.add_argument(
        "--until", metavar="T",
        help="storm time to end at: seconds, or e.g. 150min or 2.5h (default: the namelist's timax)",
    )
    command.add_argument(
        "--gpu", metavar="N|UUID",
        help="the GPU to use: nvidia-smi's index or a UUID (default: the first visible GPU)",
    )
    command.add_argument(
        "--no-wait", action="store_true",
        help="refuse instead of waiting in line when the GPU is running another job",
    )
    command.add_argument(
        "--ignore-estimate", action="store_true",
        help="start even when the disk or GPU memory estimate says it does not fit (recorded)",
    )
    command.add_argument(
        "--detach", action="store_true",
        help="start in the background and return; closing the terminal does not stop the run",
    )
    command.add_argument(
        "--library", type=Path,
        help="explicit Rust idealized initializer (source checkouts); kept for later segments",
    )
    command.add_argument(
        "--history-publisher", type=Path,
        help="explicit cm1-forecast-publish (source checkouts); kept for later segments",
    )
    command.add_argument("--json", action="store_true", help="answer in JSON")


def add_commands(commands: Any) -> None:
    command = commands.add_parser(
        "run",
        help="start a run directory's integration on the GPU (waits in line for a busy GPU)",
        description=(
            "Start the run in RUNDIR: check it, wait for the GPU, integrate to timax (or --until), "
            "writing frames and restarts into RUNDIR/segments/NNN. Stop it with gpuwm-cm1 stop RUNDIR."
        ),
    )
    _start_options(command)
    command.add_argument(
        "--continue", dest="continue_run", action="store_true",
        help="continue from the last restart if the run has one, otherwise start it (for job scripts)",
    )
    command.add_argument(
        "--start-over", action="store_true",
        help="start again from the initial state; earlier segments are kept",
    )
    command.set_defaults(handler=run_door)

    command = commands.add_parser(
        "resume",
        help="continue a paused, stopped or finished run from a restart",
        description=(
            "Continue the run in RUNDIR from a restart (default: the newest) to timax, or to --until "
            "to extend it, in a new segment."
        ),
    )
    _start_options(command)
    command.add_argument("--from-restart", type=int, metavar="N", help="restart number (default: the newest)")
    command.add_argument(
        "--from-segment", type=int, metavar="K",
        help="the segment holding restart N, when two branches hold the same number",
    )
    command.set_defaults(handler=resume_door)

    command = commands.add_parser("status", help="what one or more runs are doing")
    command.add_argument("rundirs", type=Path, nargs="*", help="run directories")
    command.add_argument("--root", type=Path, help="also list every run directory in this folder")
    command.add_argument("--wait", action="store_true",
                         help="answer once no run named here is waiting for the GPU or running (it has finished, "
                              "paused, stopped or failed)")
    command.add_argument("--json", action="store_true", help="answer in JSON")
    command.set_defaults(handler=status_door)

    command = commands.add_parser(
        "stop", help="stop a run at the end of its current step (works from any machine)",
    )
    command.add_argument("rundir", type=Path)
    how = command.add_mutually_exclusive_group()
    how.add_argument("--terminate", action="store_true",
                     help="send SIGTERM now (this host only): the write in progress is discarded")
    how.add_argument("--force", action="store_true",
                     help="SIGKILL (this host only), then remove the unfinished writes it leaves")
    how.add_argument("--abandon", action="store_true",
                     help="mark a run whose process is known to be gone as ended; signals nothing")
    command.add_argument("--json", action="store_true", help="answer in JSON")
    command.set_defaults(handler=stop_door)

    command = commands.add_parser("pause", help="stop a run right after its next restart is written")
    command.add_argument("rundir", type=Path)
    command.add_argument("--json", action="store_true", help="answer in JSON")
    command.set_defaults(handler=pause_door)

    command = commands.add_parser(
        "jobscript", help="print a Slurm or PBS script that runs RUNDIR on a GPU node, with CPU sidecars",
    )
    command.add_argument("rundir", type=Path)
    command.add_argument("--scheduler", required=True, metavar="{slurm,pbs}")
    command.add_argument("--time", default="04:00:00", help="time limit, HH:MM:SS (default 04:00:00)")
    command.add_argument("--partition", help="Slurm partition or PBS queue")
    command.add_argument("--account", help="allocation account")
    command.add_argument("--qos", help="Slurm QOS")
    command.add_argument("--cpus", type=int, default=4, help="CPUs for the run and its two sidecars (default 4)")
    command.add_argument("--mem", help="host memory, e.g. 32G (default: from the grid size)")
    command.add_argument("--gpu-type", help="a GPU type the scheduler knows, e.g. a100")
    command.add_argument("--until", metavar="T", help="storm time to run to (default: timax)")
    command.add_argument("--setup", action="append", default=[], metavar="LINE",
                         help="a shell line to run first, e.g. 'module load cuda' (repeatable)")
    command.add_argument("--python", help="the Python of the environment on the node (default: this one)")
    command.add_argument("--library", type=Path, help=argparse.SUPPRESS)
    command.add_argument("--history-publisher", type=Path, help=argparse.SUPPRESS)
    command.add_argument("--write", action="store_true", help="write RUNDIR/job-<scheduler>.sh instead of printing")
    command.add_argument("--json", action="store_true", help="answer in JSON")
    command.set_defaults(handler=jobscript_door)


__all__ = [
    "add_commands",
    "jobscript_door",
    "pause_door",
    "resume_door",
    "run_door",
    "status_door",
    "stop_door",
]
