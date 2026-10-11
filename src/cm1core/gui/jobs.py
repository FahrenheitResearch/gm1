"""Running and launching ``gpuwm-cm1`` commands for the GUI.

- A :class:`Runner` turns a command's arguments into an argv.  The real one
  is ``python -m cm1core ARGS``; the fake one (:mod:`.fake_runner`) takes the
  same arguments and writes the same run-directory contract with no GPU, for
  tests and screen work.  The command shown to the user is the same either way.
- Host-only commands run to completion with no GPU visible, at most
  :data:`HOST_CONCURRENCY` at a time, with a timeout.
- run and resume are launched in their own session (``start_new_session``),
  so closing the GUI never stops them; under WSL each also gets a holder that
  keeps the distribution running until it ends
  (:mod:`cm1core.cm1run.wsl_keepalive`).  The sidecars also run at nice 19
  with no GPU.  Their output goes to ``ROOT/.gm1-gui/logs/``, never into the run
  directory, so a run started here and one started from a shell leave the
  same files.
- Every command that is run is appended to ``RUNDIR/commands.log`` as a
  line a shell can replay.
- A request may ask for its command alone ("Show command" before a press):
  inside :meth:`Jobs.dry_run`, the point where a command would run raises
  :class:`DryRun` with the exact line instead, so nothing runs, nothing is
  launched and nothing is logged.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import sysconfig
import threading
import time
from typing import Any, Iterator, Sequence

from . import commands as C
from .runstate import segment_dirs

HOST_TIMEOUT_S = 180.0
HOST_CONCURRENCY = 2
OUTPUT_TAIL = 8000
COMMANDS_LOG = "commands.log"
#: Environment for host-only calls: no GPU, and no thread pools that would
#: take every core of a shared desktop for a sub-second check.
HOST_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "RAYON_NUM_THREADS": "2",
}
SIDECAR_ENV = {**HOST_ENV, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The server's environment for a child, able to import this cm1core.

    Children run in the run directory, so a relative ``PYTHONPATH`` entry
    (a source checkout started as ``PYTHONPATH=src``) is made absolute, and
    a checkout's ``src`` is added when cm1core is not installed.
    """

    env = dict(os.environ)
    entries = [str(Path(entry).resolve()) for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    parent = str(Path(__file__).resolve().parents[2])
    installed = {str(Path(path).resolve()) for path in (
        sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")) if path}
    if parent not in installed and parent not in entries:
        entries.insert(0, parent)
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(entries)
    env.update(extra or {})
    return env


def opens_gpu(args: Sequence[str]) -> bool:
    """Whether a command runs on the GPU: run, resume, and estimate --measure."""

    args = [str(arg) for arg in args]
    return bool(args) and (args[0] in ("run", "resume") or (args[0] == "estimate" and "--measure" in args))


@dataclass(frozen=True)
class Runner:
    kind: str                          # "real", "fake", or "mixed" (real host calls, fake GPU work)
    prefix: tuple[str, ...]
    #: For "mixed": what runs the commands that would open the GPU.
    gpu_prefix: tuple[str, ...] | None = None

    def argv(self, args: Sequence[str]) -> list[str]:
        prefix = self.gpu_prefix if self.gpu_prefix is not None and opens_gpu(args) else self.prefix
        return [*prefix, *[str(arg) for arg in args]]

    def available(self) -> frozenset[str]:
        """The subcommands this runner has (the real one: this version's CLI)."""

        if self.kind == "fake":
            from .fake_runner import COMMANDS

            return frozenset(COMMANDS)
        return _cli_commands()


_CLI_COMMANDS: list[frozenset[str]] = []


def _cli_commands() -> frozenset[str]:
    if not _CLI_COMMANDS:
        from ..cli import build_parser

        parser = build_parser()
        names: set[str] = set()
        for action in parser._actions:  # noqa: SLF001 - argparse has no public accessor
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                names.update(choices)
        _CLI_COMMANDS.append(frozenset(names))
    return _CLI_COMMANDS[0]


def real_runner() -> Runner:
    return Runner("real", (sys.executable, "-m", "cm1core"))


def fake_runner() -> Runner:
    return Runner("fake", (sys.executable, "-m", "cm1core.gui.fake_runner"))


def mixed_runner() -> Runner:
    """The real host-only commands (import, edit, check, estimate, jobscript) and a fake GPU.

    For screen work and tests on a machine whose GPU is busy or absent: every
    setup screen shows what the real commands say, and Start and Measure speed
    run the fake runner, which writes the same run-directory contract.
    """

    return Runner("mixed", (sys.executable, "-m", "cm1core"),
                  (sys.executable, "-m", "cm1core.gui.fake_runner"))


@dataclass
class CommandResult:
    args: list[str]
    command: str
    rc: int
    stdout: str
    stderr: str
    elapsed_s: float
    json: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "argv": ["gpuwm-cm1", *self.args],
            "rc": self.rc,
            "elapsed_s": round(self.elapsed_s, 3),
            "stdout_tail": self.stdout[-OUTPUT_TAIL:],
            "stderr_tail": self.stderr[-OUTPUT_TAIL:],
        }


@dataclass
class Launched:
    run_id: str
    rundir: Path
    role: str                            # run, resume, cm1out or quicklook
    command: str
    process: subprocess.Popen
    log: Path
    started: float = field(default_factory=time.time)
    #: A sidecar's run, and how many segments the run folder had before it was launched.
    engine: "Launched | None" = None
    segments_before: int = 0

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    @property
    def orphaned(self) -> bool:
        """A sidecar whose run has exited without starting a segment: it has nothing to follow."""

        return (self.engine is not None and not self.engine.alive and self.alive
                and len(segment_dirs(self.rundir)) <= self.segments_before)


class Unavailable(RuntimeError):
    """This version of gpuwm-cm1 has no such command yet."""


class DryRun(Exception):
    """A request asked for its command only; raised where the command would run, before anything changes."""

    def __init__(self, command: str, **extra: Any) -> None:
        super().__init__(command)
        self.command = command
        self.extra = extra


def _parse_json(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    # Some commands print a line or two before their JSON document.
    start = text.find("{")
    if start >= 0:
        try:
            return json.loads(text[start:])
        except ValueError:
            return None
    return None


class Jobs:
    """The server's commands: run, launch, reap and log them."""

    def __init__(self, runner: Runner, server_dir: Path) -> None:
        self.runner = runner
        self.server_dir = Path(server_dir)
        self._host_slots = threading.BoundedSemaphore(HOST_CONCURRENCY)
        self._lock = threading.Lock()
        self._launched: list[Launched] = []
        self._run_locks: dict[str, threading.Lock] = {}
        self._stopping = threading.Event()
        self._dry = threading.local()
        self._reaper = threading.Thread(target=self._reap_loop, name="gm1-gui-reaper", daemon=True)
        self._reaper.start()

    # ---- dry runs ("Show command" before a press)

    @contextmanager
    def dry_run(self, on: bool = True) -> Iterator[None]:
        """Within this block (on this thread), commands are built and checked but not run: see :class:`DryRun`."""

        previous = getattr(self._dry, "on", False)
        self._dry.on = bool(on)
        try:
            yield
        finally:
            self._dry.on = previous

    def dry_active(self) -> bool:
        return bool(getattr(self._dry, "on", False))

    def stop_if_dry(self, command: str | Sequence[str], **extra: Any) -> None:
        """Raise :class:`DryRun` for ``command`` (a line, or a command's arguments) inside a dry run."""

        if self.dry_active():
            line = command if isinstance(command, str) else C.display(command)
            raise DryRun(line, **extra)

    # ---- bookkeeping

    def run_lock(self, run_id: str) -> threading.Lock:
        with self._lock:
            return self._run_locks.setdefault(run_id, threading.Lock())

    def require(self, name: str) -> None:
        if name not in self.runner.available():
            raise Unavailable(
                f"This version of gpuwm-cm1 has no '{name}' command yet.")

    def launched(self, run_id: str | None = None, alive_only: bool = True) -> list[Launched]:
        with self._lock:
            rows = list(self._launched)
        return [row for row in rows
                if (run_id is None or row.run_id == run_id) and (row.alive or not alive_only)]

    def engine_alive(self, run_id: str) -> bool:
        return any(row.role in ("run", "resume") for row in self.launched(run_id))

    @staticmethod
    def end(row: Launched, grace_s: float = 2.0) -> None:
        """End a process this server launched, with its whole session (SIGTERM, then SIGKILL)."""

        if not row.alive:
            return
        try:
            os.killpg(row.process.pid, signal.SIGTERM)
        except OSError:
            return
        try:
            row.process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(row.process.pid, signal.SIGKILL)
            except OSError:
                pass

    def end_orphans(self, run_id: str | None = None) -> None:
        """End the sidecars of runs that exited before they began (a sidecar never outlives its run)."""

        for row in self.launched(run_id):
            if row.orphaned:
                self.end(row)

    def _reap_loop(self) -> None:
        while not self._stopping.wait(1.0):
            with self._lock:
                for row in self._launched:
                    row.process.poll()
                cutoff = time.time() - 3600.0
                self._launched = [row for row in self._launched
                                  if row.process.returncode is None or row.started > cutoff]
            try:
                self.end_orphans()
            except Exception:  # noqa: BLE001 - the reaper never stops over one run folder
                pass

    def close(self) -> None:
        """Stop reaping.  Launched runs keep running: they are in their own sessions."""

        self._stopping.set()

    # ---- commands.log

    def log_command(self, rundir: Path, action: str, line: str, note: str = "") -> None:
        if self.dry_active():
            return
        try:
            if not rundir.is_dir():
                return
            text = f"# {utc_now()} {action}{(': ' + note) if note else ''}\n{line}\n"
            descriptor = os.open(rundir / COMMANDS_LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(descriptor, text.encode("utf-8"))
            finally:
                os.close(descriptor)
        except OSError:
            pass

    # ---- host-only commands

    def host(self, args: Sequence[str], *, timeout: float = HOST_TIMEOUT_S,
             cwd: Path | None = None) -> CommandResult:
        args = [str(arg) for arg in args]
        if args[0] not in C.HOST_COMMANDS:
            raise ValueError(f"{args[0]} is not a host-only command")
        if args[0] == "quicklook" and "--follow" in args:
            raise ValueError("quicklook --follow is a sidecar, launched beside a run, not a host-only call")
        self.require(args[0])
        self.stop_if_dry(args)
        env = child_env(HOST_ENV)
        started = time.monotonic()
        with self._host_slots:
            try:
                completed = subprocess.run(
                    self.runner.argv(args), stdin=subprocess.DEVNULL, capture_output=True,
                    text=True, timeout=timeout, env=env, cwd=cwd, check=False)
                rc, out, err = completed.returncode, completed.stdout, completed.stderr
            except subprocess.TimeoutExpired as error:
                rc = 124
                out = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
                err = (error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")) + \
                    f"\nStopped after {timeout:.0f} s."
            except OSError as error:
                rc, out, err = 127, "", f"Could not start the command: {error}"
        return CommandResult(args, C.display(args), rc, out, err,
                             time.monotonic() - started, _parse_json(out))

    def host_lines(self, args: Sequence[str], on_line: Any, *, timeout: float = HOST_TIMEOUT_S,
                   cwd: Path | None = None) -> CommandResult:
        """A host-only command whose output is read a line at a time, as it is written.

        ``on_line(text)`` is called for every stdout line (``estimate --each`` writes one priced example
        per line), so a screen can show each result as it lands.  Otherwise as :meth:`host`: the same
        environment, the same slots, the same timeout (the command is killed when it is reached).
        """

        args = [str(arg) for arg in args]
        if args[0] not in C.HOST_COMMANDS:
            raise ValueError(f"{args[0]} is not a host-only command")
        self.require(args[0])
        self.stop_if_dry(args)
        env = child_env(HOST_ENV)
        started = time.monotonic()
        lines: list[str] = []
        errors: list[str] = []
        with self._host_slots:
            try:
                process = subprocess.Popen(
                    self.runner.argv(args), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, env=env, cwd=cwd)
            except OSError as error:
                return CommandResult(args, C.display(args), 127, "", f"Could not start the command: {error}",
                                     time.monotonic() - started)
            drain = threading.Thread(target=lambda: errors.append(process.stderr.read()), daemon=True)
            drain.start()
            timer = threading.Timer(timeout, process.kill)
            timer.start()
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    lines.append(line)
                    try:
                        on_line(line)
                    except Exception:  # noqa: BLE001 - one bad line never stops the others
                        pass
                rc = process.wait()
            finally:
                timed_out = not timer.is_alive() and process.returncode not in (0, None) and \
                    time.monotonic() - started >= timeout
                timer.cancel()
            drain.join(timeout=5.0)
        err = "".join(errors)
        if timed_out:
            rc, err = 124, err + f"\nStopped after {timeout:.0f} s."
        out = "".join(lines)
        return CommandResult(args, C.display(args), rc, out, err, time.monotonic() - started, None)

    # ---- launched commands

    def launch(self, run_id: str, rundir: Path, args: Sequence[str], *, role: str,
               engine: Launched | None = None, segments_before: int = 0) -> Launched:
        args = [str(arg) for arg in args]
        measure = args[0] == "estimate" and "--measure" in args
        if args[0] not in C.LAUNCH_COMMANDS and not measure:
            raise ValueError(f"{args[0]} is not a command the GUI launches")
        self.require(args[0])
        self.stop_if_dry(args)
        sidecar = args[0] in C.SIDECARS
        argv = self.runner.argv(args)
        if sidecar and shutil.which("nice"):
            argv = ["nice", "-n", "19", *argv]
        env = child_env(SIDECAR_ENV if sidecar else None)
        logs = self.server_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        log = logs / f"{run_id.replace('/', '__')}-{stamp}-{role}.log"
        with log.open("ab") as stream:
            process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                cwd=rundir, env=env, start_new_session=True, close_fds=True)
        row = Launched(run_id, rundir, role, C.display(args), process, log,
                       engine=engine, segments_before=segments_before)
        with self._lock:
            self._launched.append(row)
        if args[0] in ("run", "resume") or sidecar:
            self._hold_wsl(row)
        return row

    @staticmethod
    def _hold_wsl(row: Launched) -> None:
        """Under WSL, keep Ubuntu running while ``row`` runs, so closing the GM1 window never ends it.

        Best effort (:mod:`cm1core.cm1run.wsl_keepalive`): a holder that cannot start is a warning in the
        command's log and on the server's terminal, never a failed run.
        """

        def report(text: str) -> None:
            try:
                with row.log.open("a", encoding="utf-8") as stream:
                    stream.write(f"{utc_now()} {text}\n")
                print(f"gpuwm-cm1 gui: {row.run_id}: {text}", file=sys.stderr, flush=True)
            except (OSError, ValueError):
                pass

        from ..cm1run.wsl_keepalive import hold

        hold(row.process.pid, report=report)


__all__ = [
    "COMMANDS_LOG",
    "CommandResult",
    "DryRun",
    "HOST_ENV",
    "Jobs",
    "Launched",
    "Runner",
    "Unavailable",
    "fake_runner",
    "mixed_runner",
    "opens_gpu",
    "real_runner",
    "utc_now",
]
