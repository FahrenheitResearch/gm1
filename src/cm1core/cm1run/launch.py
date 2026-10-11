"""``gpuwm-cm1 run`` and ``resume``: the supervisor of one run segment.

``gpuwm-cm1 run RUNDIR`` is a small
process that owns one segment of a run from start to end:

1. it refuses a run directory that is already running, a namelist the import
   refused or one still waiting for an acknowledgment, and a run whose frames
   and restarts would not fit on the disk (``cartesian --plan``, host only);
2. it creates ``segments/NNN`` with the segment's deck (``deck.toml`` with
   the restart schedule, and ``irst``/``rstnum``/``duration_s`` on a resume);
3. it waits its turn for the GPU in a per-GPU queue (a ticket per waiting
   run, then an exclusive ``flock`` on the lock file ``gpuwm``'s runs use for
   the same card), writing a heartbeat every 5 s while it waits: the run's
   status reads ``Waiting for GPU``;
4. with the GPU, it checks the memory the run needs against what the card
   has free, then starts ``gpuwm-cm1 cartesian --integrate --events
   ... --control-dir RUNDIR/control --history-envelope off`` (and
   ``--history-initial`` on a fresh start) as its child, which inherits the
   GPU lock, so the lock is held exactly as long as either process lives;
5. it follows the child's event log, prints a line per frame and restart and
   a progress line every 30 s, forwards SIGTERM and Ctrl-C to it, and records
   how the segment ended in ``segment.json`` and ``run.json``.

Stop and pause go through ``RUNDIR/control`` and the runtime reads them
itself: a run started from a shell stops on the control file alone.  A stop
or pause made before the runtime has admitted its control directory (while
the run waits for the GPU or is starting) is held by the supervisor and
handed on once the runtime is reading the directory.

``--detach`` checks the same gates, starts the supervisor in its own session
with its output in ``RUNDIR/logs``, and returns: closing the terminal or the
GUI never stops a run.  Under WSL it also keeps the distribution running until
the supervisor ends (:mod:`.wsl_keepalive`), so closing the last Ubuntu window
does not end the run either.

Importing this module imports neither CuPy nor numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, TextIO

from .events import EventTail, SegmentEvents
from .rundir import (
    DECK,
    RestartInfo,
    RunDirectory,
    Segment,
    derive_deck,
    host_name,
    installation,
    local_process_alive,
    process_identity,
    read_json_or_none,
    utc_now,
    write_json_atomic,
    write_text_atomic,
)

#: Test hook: the module the supervisor runs as its child instead of cm1core
#: (``python -m <module> cartesian DECK ...``), so the CPU suite drives the
#: real supervisor, queue and control files with a stand-in integration.
CHILD_MODULE_ENV = "GPUWM_CM1_RUN_CHILD_MODULE"
#: An explicit nvidia-smi (a fixture in tests; a full path on odd installs).
NVIDIA_SMI_ENV = "GPUWM_CM1_NVIDIA_SMI"
#: The directory of the per-GPU lock files, shared with gpuwm.supervisor's
#: GPUFileLock (same variable, same default, same file name for a UUID).
LOCK_ROOT_ENV = "GPUWM_GPU_LOCK_ROOT"

HEARTBEAT_S = 5.0
QUEUE_POLL_S = 0.5
#: Before the runtime admits its control directory the supervisor moves any
#: new request aside this often, so the runtime never sees one it would refuse.
PREADMISSION_POLL_S = 0.1
#: A request file older than the supervisor by more than this was left by an
#: earlier run; it is set aside rather than honoured.
STALE_REQUEST_GRACE_S = 5.0
PROGRESS_EVERY_S = 30.0
#: After a signal has asked the run to stop through its control file, the
#: child gets SIGTERM if it has not ended within this long.
SIGNAL_STOP_GRACE_S = 60.0
#: How long a start waits for a finished segment's supervisor to let go of
#: the run directory (it files the segment's end right after the end record).
OWNER_FINISHING_S = 15.0

#: GPU memory and disk use the estimator's model (cm1run.estimate,
#: records/GM1-GPU-RATES.json) on the host plan's own priced peak and frame
#: payload, so a run the Estimate screen says fits is not refused here.
#: Needed <= this share of the memory actually free (the estimator's rule).
GPU_FREE_SHARE = 0.95
#: A restart's payload, 280 B per cell (measured; the estimator's figure when
#: it cannot build the restart profile); the host plan does not price it.
RESTART_BYTES_PER_CELL = 280.0
#: Refused when the disk estimate exceeds free space minus 10 % (4.4).
DISK_FREE_SHARE = 0.90

CONTROL_REFUSAL_CODE = "CM1-CARTESIAN-CONTROL-DIRECTORY"


class LaunchRefusal(RuntimeError):
    """The run was not started; ``reason`` says why and ``remedy`` what to do."""

    def __init__(self, code: str, reason: str, remedy: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.remedy = remedy

    def as_dict(self) -> dict[str, str]:
        return {"status": "refused", "code": self.code, "reason": self.reason, "remedy": self.remedy}


# ================================================================== the GPU


@dataclass(frozen=True)
class GpuDevice:
    index: int
    uuid: str
    name: str
    memory_total_mib: int | None
    memory_used_mib: int | None
    driver: str | None

    @property
    def memory_free_bytes(self) -> int | None:
        if self.memory_total_mib is None or self.memory_used_mib is None:
            return None
        return max(0, self.memory_total_mib - self.memory_used_mib) * 1024 * 1024

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "uuid": self.uuid, "name": self.name,
            "memory_total_mib": self.memory_total_mib, "memory_used_mib": self.memory_used_mib,
            "driver": self.driver,
        }


class GpuQueryError(RuntimeError):
    pass


def nvidia_smi() -> str | None:
    explicit = os.environ.get(NVIDIA_SMI_ENV)
    if explicit:
        return explicit
    found = shutil.which("nvidia-smi")
    if found:
        return found
    wsl = "/usr/lib/wsl/lib/nvidia-smi"
    return wsl if os.access(wsl, os.X_OK) else None


def _mib(text: str) -> int | None:
    try:
        return int(float(text))
    except ValueError:
        return None


def query_gpus() -> list[GpuDevice]:
    """Every GPU nvidia-smi lists here, with its memory in use now."""

    smi = nvidia_smi()
    if smi is None:
        raise GpuQueryError("nvidia-smi is not on this machine's PATH")
    try:
        completed = subprocess.run(
            [smi, "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GpuQueryError(f"nvidia-smi could not run: {error}") from error
    if completed.returncode != 0:
        raise GpuQueryError(f"nvidia-smi failed: {(completed.stderr or completed.stdout).strip()}")
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6 or not fields[0].isdigit():
            continue
        devices.append(GpuDevice(
            index=int(fields[0]), uuid=fields[1], name=fields[2],
            memory_total_mib=_mib(fields[3]), memory_used_mib=_mib(fields[4]), driver=fields[5] or None,
        ))
    if not devices:
        raise GpuQueryError("nvidia-smi lists no GPU")
    return devices


@dataclass(frozen=True)
class GpuChoice:
    """The card a run uses, the key of its queue, and what the child is shown."""

    device: GpuDevice | None
    key: str
    cuda_visible_devices: str | None
    how: str

    def as_dict(self) -> dict[str, Any]:
        return {
            **({} if self.device is None else self.device.as_dict()),
            "queue_key": self.key,
            "cuda_visible_devices": self.cuda_visible_devices,
            "chosen_by": self.how,
        }


def choose_gpu(spec: str | None, env: Mapping[str, str] | None = None) -> GpuChoice:
    """``--gpu N`` (nvidia-smi index) or ``--gpu GPU-uuid``; by default the first visible card.

    A job scheduler that sets ``CUDA_VISIBLE_DEVICES`` chooses the card:
    its first entry is used when ``--gpu`` is not given.
    """

    env = os.environ if env is None else env
    visible = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    wanted = spec if spec not in (None, "") else (visible.split(",")[0].strip() if visible else None)
    how = "--gpu" if spec not in (None, "") else ("CUDA_VISIBLE_DEVICES" if visible else "the first GPU")
    try:
        devices = query_gpus()
    except GpuQueryError as error:
        key = f"{host_name()}:cuda-visible-devices={visible or 'unset'}:{wanted or 0}"
        return GpuChoice(None, key, wanted if spec not in (None, "") else None,
                         f"{how}; the card could not be named ({error})")
    device = None
    if wanted is None:
        device = devices[0]
    elif wanted.isdigit():
        device = next((row for row in devices if row.index == int(wanted)), None)
    else:
        device = next((row for row in devices if row.uuid == wanted or row.uuid.startswith(wanted)), None)
    if device is None:
        listed = "; ".join(f"{row.index}: {row.name} ({row.uuid})" for row in devices)
        raise LaunchRefusal(
            "GM1-RUN-GPU",
            f"there is no GPU {wanted!r} on this machine ({how}); it has {listed}",
            "pass --gpu with one of the listed indices or UUIDs",
        )
    return GpuChoice(device, device.uuid, device.uuid, how)


# ================================================================ the queue


def lock_root() -> Path:
    configured = os.environ.get(LOCK_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(tempfile.gettempdir()) / "gpuwm" / "locks"


def lock_path(key: str) -> Path:
    """gpuwm.supervisor.default_lock_path: the same file for the same card."""

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return lock_root() / f"gpu-{digest}.lock"


class GpuQueue:
    """The per-GPU queue: tickets in arrival order, then an exclusive ``flock``.

    Each waiting run holds a ticket file named by its arrival time; only the
    oldest live ticket tries the lock, so runs start in the order they asked.
    A ticket whose process is gone is removed by whoever sees it.  The lock
    itself is the flock gpuwm's own runs take on the same card, so a GM1 run
    and a gpuwm run never share a GPU either.
    """

    def __init__(self, key: str, info: Mapping[str, Any]) -> None:
        self.key = key
        self.path = lock_path(key)
        self.queue_dir = self.path.with_name(self.path.stem + ".queue")
        self.info = dict(info)
        self.ticket: Path | None = None
        self.fd: int | None = None

    def join(self) -> None:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        identity = process_identity(os.getpid())
        name = f"{time.time_ns():020d}-{os.getpid()}.json"
        ticket = self.queue_dir / name
        write_json_atomic(ticket, {**self.info, "identity": identity, "joined_utc": utc_now()})
        self.ticket = ticket

    def ahead(self) -> list[dict[str, Any]]:
        """Live tickets older than this one (dead ones are removed)."""

        if self.ticket is None:
            return []
        waiting = []
        try:
            names = sorted(os.listdir(self.queue_dir))
        except OSError:
            return []
        for name in names:
            if name >= self.ticket.name:
                break
            if not name.endswith(".json"):
                continue
            path = self.queue_dir / name
            payload = read_json_or_none(path)
            identity = payload.get("identity") if isinstance(payload, dict) else None
            if not local_process_alive(identity):
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            waiting.append(payload)
        return waiting

    def holder(self) -> dict[str, Any] | None:
        """Who holds the lock, as the holder wrote it (gpuwm's owner record format)."""

        try:
            data = self.path.read_bytes()
        except OSError:
            return None
        try:
            payload = json.loads(data[1:].decode("utf-8")) if len(data) > 1 else None
        except (UnicodeDecodeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def try_acquire(self) -> bool:
        if self.fd is not None:
            return True
        if self.ahead():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            if error.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                return False
            raise
        owner = json.dumps({
            "gpu_uuid": self.key, "pid": os.getpid(), "run_id": self.info.get("run_id"),
            "acquired_at_utc": utc_now(), "rundir": self.info.get("rundir"),
            "segment": self.info.get("segment"), "host": host_name(),
        }, sort_keys=True).encode("utf-8")
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, b"\0")
        os.ftruncate(fd, 1)
        os.write(fd, owner)
        self.fd = fd
        self.leave()
        return True

    def leave(self) -> None:
        if self.ticket is not None:
            try:
                self.ticket.unlink()
            except OSError:
                pass
            self.ticket = None

    def release(self) -> None:
        """Drop this process's hold on the lock.

        Only ``close``: the child inherited the same open file description,
        and ``LOCK_UN`` on it would release the card for the child too.  A
        supervisor that leaves early (an error while it follows the run)
        must leave the card locked until the integration itself exits.
        """

        self.leave()
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)


# ============================================================ the segment


def parse_seconds(text: str | float | int) -> float:
    """``9000``, ``9000s``, ``150min``, ``2.5h`` as seconds."""

    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    value = str(text).strip().lower().replace(" ", "")
    for suffix, scale in (("min", 60.0), ("h", 3600.0), ("m", 60.0), ("s", 1.0)):
        if value.endswith(suffix):
            number = value[: -len(suffix)]
            break
    else:
        number, scale = value, 1.0
    try:
        seconds = float(number) * scale
    except ValueError:
        raise ValueError(f"{text!r} is not a time; write seconds, or e.g. 150min or 2.5h") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{text!r} is not a positive time")
    return seconds


@dataclass
class SegmentPlan:
    """What one segment will integrate, and how its deck differs from ``deck.toml``."""

    kind: str  # "fresh" or "resume"
    until_s: float
    restart_interval_s: float
    restart_rule: str
    start_model_time_s: float
    frames_expected: int
    restarts_expected: int
    history_initial: bool
    overrides: dict[str, dict[str, Any]]
    resumed_from: RestartInfo | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "until_s": self.until_s,
            "restart_interval_s": self.restart_interval_s, "restart_rule": self.restart_rule,
            "start_model_time_s": self.start_model_time_s,
            "frames_expected": self.frames_expected, "restarts_expected": self.restarts_expected,
            "history_initial": self.history_initial, "overrides": self.overrides,
            "resumed_from": None if self.resumed_from is None else {
                "segment": self.resumed_from.segment, "restart": self.resumed_from.number,
                "model_time_s": self.resumed_from.model_time_s,
                "directory": os.fspath(self.resumed_from.path),
            },
        }


@dataclass
class StartRequest:
    command: str = "run"  # "run" or "resume"
    start_over: bool = False
    continue_run: bool = False
    from_restart: int | None = None
    from_segment: int | None = None
    until_s: float | None = None
    gpu: str | None = None
    wait: bool = True
    ignore_estimate: bool = False
    library: Path | None = None
    history_publisher: Path | None = None
    progress_every_s: float = PROGRESS_EVERY_S


def _steps(seconds: float, dt: float) -> int | None:
    count = seconds / dt
    nearest = round(count)
    return int(nearest) if abs(count - nearest) <= 1e-9 * max(1.0, abs(count)) else None


def _events_after(start_step: int, every: int, total: int) -> int:
    first = (start_step // every + 1) * every
    return 0 if first > total else (total - first) // every + 1


def _near(value: float, grid: float, above: float) -> list[float]:
    """Two or three valid end times near ``value`` on ``grid`` and after ``above``."""

    base = math.floor(value / grid) * grid
    options = sorted({candidate for candidate in (base, base + grid, base + 2 * grid) if candidate > above})
    return options[:3]


def plan_segment(run: RunDirectory, record: Mapping[str, Any], request: StartRequest) -> SegmentPlan | None:
    """The next segment: a fresh start or a resume, its end time and restart cadence.

    ``None`` when ``run --continue`` finds nothing left to integrate.
    """

    import tomllib

    try:
        deck = tomllib.loads(run.deck.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LaunchRefusal(
            "GM1-RUN-NO-DECK", f"{run.path} has no {DECK}: the namelist was not translated",
            "fix the refused settings the import report lists and import the namelist again",
        ) from None
    table = deck.get("run") or {}
    dt = float(table["dt_s"])
    tapfrq = float(table["history_interval_s"])
    timax = float(table["duration_s"])
    deck_restart = float(table["restart_interval_s"])
    rstfrq = (record.get("cm1") or {}).get("rstfrq")
    namelist_restarts = rstfrq is None or float(rstfrq) > 0

    restarts = run.restarts()
    started = any(segment.frames() or segment.restarts() for segment in run.segments())
    resume = request.command == "resume"
    if request.command == "run":
        # --continue goes on with the run as its frames show it: a restart on
        # the latest segment's lineage, never one of a branch it has left.
        if request.continue_run and run.lineage_restarts():
            resume = True
        elif started and not request.start_over and not request.continue_run:
            raise LaunchRefusal(
                "GM1-RUN-STARTED",
                f"{run.name()} has already run ({len(run.segments())} segment(s)); starting it "
                "again from the beginning is a separate choice",
                f"continue it with gpuwm-cm1 resume {shlex.quote(os.fspath(run.path))}, or start "
                "over (the earlier segments are kept) with --start-over",
            )

    restart: RestartInfo | None = None
    start_time = 0.0
    start_step = 0
    if resume:
        restart = run.find_restart(request.from_restart, request.from_segment)
        if restart is None:
            available = ", ".join(
                f"{row.number} ({'?' if row.model_time_s is None else format(row.model_time_s, 'g')} s, "
                f"segment {row.segment})" for row in restarts
            ) or "none"
            raise LaunchRefusal(
                "GM1-RUN-NO-RESTART",
                "there is no restart to continue from"
                + ("" if request.from_restart is None else f" numbered {request.from_restart}")
                + ("" if request.from_segment is None else f" in segment {request.from_segment}")
                + (" on the run's latest segment and the segments it continues"
                   if request.from_restart is None and request.from_segment is None else "")
                + f"; the run's restarts: {available}",
                "start it with gpuwm-cm1 run, or pick one of the listed restarts with --from-restart N "
                "(and --from-segment K for one on another branch)",
            )
        start_time = float(restart.model_time_s or 0.0)
        start_step = int(restart.step if restart.step is not None else _steps(start_time, dt) or 0)

    if request.until_s is not None:
        until = float(request.until_s)
    elif resume:
        row = run.segment_record(restart.segment) or {}
        until = max(timax, float(row.get("until_s") or 0.0))
    else:
        until = timax
    if resume and until <= start_time:
        if request.continue_run and request.until_s is None:
            return None
        raise LaunchRefusal(
            "GM1-RUN-NOTHING-LEFT",
            f"the run has reached {start_time:g} s, its end"
            + ("" if request.until_s is not None else f" (timax = {timax:g} s)"),
            "pass --until with a later storm time to extend it, e.g. --until "
            f"{_near(start_time + 3600.0, tapfrq, start_time)[0]:g}",
        )
    total = _steps(until, dt)
    frame_steps = _steps(tapfrq, dt)
    if total is None or frame_steps is None or total % frame_steps:
        options = ", ".join(f"{value:g}" for value in _near(until, tapfrq, start_time))
        raise LaunchRefusal(
            "GM1-RUN-END-TIME",
            f"a run ends on a frame: {until:g} s is not a whole number of tapfrq = {tapfrq:g} s "
            f"frames (dt = {dt:g} s)",
            f"choose an end time such as {options} s",
        )
    # A downscaled run's [chain] table (gpuwm-cm1 chain new) is also kept in run.json, so a re-import
    # (an edit on the Settings screen) that rewrites deck.toml without it cannot turn the run into an
    # undriven one: the segment deck takes the table back, and the door's contract check then refuses
    # an edit that no longer fits the prepared history (GM1-CHAIN-PREPARED).
    chain_table = (record.get("chain") or {}).get("table")
    chained = "chain" in deck or bool(chain_table)
    if chained and resume:
        raise LaunchRefusal(
            "GM1-CHAIN-RESTART",
            "a run driven by a WRF or ArWen history is not continued from a restart: the restart "
            "carries its state but not its boundary interval or its prepared tables",
            "start it over (gpuwm-cm1 run RUNDIR --start-over), or make a new downscaled run "
            "that covers the longer time",
        )
    if chained:
        # records/GM1-ARWEN-CHAIN.md 2 and 7.6: a chained run is not resumed, so it writes no
        # restart (the door would refuse the resume, GM1-CHAIN-RESTART).
        restart_every = until
        rule = "none: a run driven by a WRF or ArWen history is not continued from a restart"
        restart_steps = total
    elif namelist_restarts:
        restart_every = deck_restart
        rule = f"every {restart_every:g} s (the namelist's rstfrq)"
        restart_steps = _steps(restart_every, dt)
        if restart_steps is None or total % restart_steps:
            grid = restart_every * tapfrq / math.gcd(int(round(restart_every)), int(round(tapfrq)))
            options = ", ".join(f"{value:g}" for value in _near(until, grid, start_time))
            raise LaunchRefusal(
                "GM1-RUN-END-TIME",
                f"the run's last restart must be its end state: {until:g} s is not a whole number "
                f"of rstfrq = {restart_every:g} s",
                f"choose an end time such as {options} s, or change rstfrq in the namelist",
            )
    else:
        restart_every = until
        rule = (
            f"one restart at the end ({until:g} s): the namelist's rstfrq = {float(rstfrq):g} "
            "writes none, and a GUI run keeps one so it can always be continued"
        )
        restart_steps = total
    frames = _events_after(start_step, frame_steps, total) + (0 if resume else 1)
    restarts_expected = 0 if chained else _events_after(start_step, restart_steps, total)
    overrides: dict[str, dict[str, Any]] = {
        "run": {"duration_s": float(until), "restart_interval_s": float(restart_every)},
    }
    if resume:
        overrides["run"].update({"irst": 1, "rstnum": int(restart.number)})
    if chain_table and "chain" not in deck:
        overrides["chain"] = dict(chain_table)
    return SegmentPlan(
        kind="resume" if resume else "fresh", until_s=float(until),
        restart_interval_s=float(restart_every), restart_rule=rule,
        start_model_time_s=start_time, frames_expected=frames,
        restarts_expected=restarts_expected, history_initial=not resume,
        overrides=overrides, resumed_from=restart,
    )


def segment_deck_text(run: RunDirectory, plan: SegmentPlan, segment_name: str) -> str:
    changed = ", ".join(
        f"{table}.{key} = {value}" for table, keys in plan.overrides.items() for key, value in keys.items()
    )
    header = (
        f"# Segment {segment_name} of this run: deck.toml with {changed}.",
        f"# Restarts: {plan.restart_rule}.",
        "# Written by gpuwm-cm1 run; never hand-edited.",
    )
    return derive_deck(run.deck.read_text(encoding="utf-8"), plan.overrides, header=header)


# ================================================================ admission


@dataclass
class Needs:
    cells: int
    component: str | None
    gpu_bytes: int
    gpu_basis: str
    disk_bytes: int
    disk_basis: str
    locked_host_bytes: int
    frame_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def child_environment(cuda_visible_devices: str | None = None) -> dict[str, str]:
    """The child's environment: this package importable, the chosen card only."""

    env = dict(os.environ)
    package_parent = Path(__file__).resolve().parents[2]
    if not {"site-packages", "dist-packages"} & set(package_parent.parts):
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.fspath(package_parent) + (os.pathsep + existing if existing else "")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _sounding_arguments(run: RunDirectory, record: Mapping[str, Any]) -> list[str]:
    sounding = record.get("sounding") or {}
    isnd = (record.get("cm1") or {}).get("isnd")
    if run.sounding.is_file() and (sounding.get("read") or isnd in (7, 17)):
        return ["--sounding", os.fspath(run.sounding)]
    return []


def host_plan(run: RunDirectory, record: Mapping[str, Any], deck: Path, request: StartRequest,
              *, restart_from: Path | None = None) -> dict[str, Any]:
    """``gpuwm-cm1 cartesian DECK --plan``: the configuration check and the priced plan.

    Host only (CUDA_VISIBLE_DEVICES=""): no device is opened.  A resume's deck
    names ``irst = 1``, so the plan is given the restart directory it reads,
    as the run itself will be.
    """

    with tempfile.TemporaryDirectory(prefix="gm1-plan-") as scratch:
        # The integration's plan, not the timing harness's: the door admits
        # the output directories (new paths it does not create under --plan),
        # the restart a resume reads and the whole schedule, CUDA-cold.
        command = [
            sys.executable, "-m", os.environ.get(CHILD_MODULE_ENV) or "cm1core", "cartesian", os.fspath(deck),
            "--output", os.fspath(Path(scratch) / "plan.json"), "--plan", "--json", "--integrate",
            "--history-dir", os.fspath(Path(scratch) / "frames"),
            "--restart-dir", os.fspath(Path(scratch) / "restarts"),
            "--history-envelope", "off",
            *_sounding_arguments(run, record),
        ]
        if restart_from is not None:
            command += ["--restart-from", os.fspath(restart_from)]
        if request.library is not None:
            command += ["--library", os.fspath(request.library)]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=300, check=False,
                env=child_environment(""), cwd=run.path,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise LaunchRefusal("GM1-RUN-PLAN", f"the run could not be priced: {error}",
                                "run gpuwm-cm1 doctor") from error
    if completed.returncode != 0:
        detail = None
        try:
            detail = json.loads(completed.stderr)
        except ValueError:
            pass
        if isinstance(detail, dict) and detail.get("reason"):
            raise LaunchRefusal(
                str(detail.get("code") or "GM1-RUN-CHECK"),
                f"the configuration is refused: {detail['reason']}",
                str(detail.get("remedy") or "edit the namelist and import it again"),
            )
        raise LaunchRefusal(
            "GM1-RUN-CHECK",
            f"the configuration check failed (exit {completed.returncode}): "
            + ((completed.stderr or completed.stdout).strip()[-800:] or "no message"),
            "edit the namelist and import it again",
        )
    try:
        return json.loads(completed.stdout)
    except ValueError as error:
        raise LaunchRefusal("GM1-RUN-PLAN", f"the plan was not readable: {error}",
                            "run gpuwm-cm1 doctor") from error


def estimate_needs(deck: Mapping[str, Any], plan_payload: Mapping[str, Any], plan: SegmentPlan) -> Needs:
    """GPU memory, disk and locked host memory for one segment, in the estimator's model.

    The host plan (``cartesian --plan --integrate``) gives the door's priced
    registry peak at the run's own closure tile and the frame's payload;
    :func:`cm1core.cm1run.estimate.device_need` and the record's disk ratios
    turn them into what ``gpuwm-cm1 estimate`` shows.  The counts are this
    segment's (a resume writes only what is left).
    """

    from .estimate import device_need
    from .rates import load_record

    grid = deck.get("grid") or {}
    cells = int(grid.get("nx", 0)) * int(grid.get("ny", 0)) * int(grid.get("nz", 0))
    component = str((deck.get("physics") or {}).get("component") or "")
    config = plan_payload.get("config") or {}
    priced = int(((config.get("tile_selection") or {}).get("priced_peak_bytes")) or 0)
    gpu, gpu_basis = device_need(priced, cells, component)
    frame = int(((plan_payload.get("history_frame") or {}).get("payload_bytes")) or 0)
    disk_model = load_record()["disk"]
    netcdf_ratio = float(disk_model["netcdf_over_raw"])
    quicklooks = int(disk_model["quicklook_bytes_per_frame"])
    per_frame = int(frame * (1.0 + netcdf_ratio)) + quicklooks
    restart = int(RESTART_BYTES_PER_CELL * cells)
    disk = plan.frames_expected * per_frame + plan.restarts_expected * restart
    return Needs(
        cells=cells, component=component or None, gpu_bytes=int(gpu), gpu_basis=gpu_basis,
        disk_bytes=int(disk),
        disk_basis=(
            f"{plan.frames_expected} frame(s) x ({frame / 1e9:.3f} GB raw + {netcdf_ratio:g} x that as cm1out "
            f"netCDF + quick-looks) + {plan.restarts_expected} restart(s) x {RESTART_BYTES_PER_CELL:.0f} B/cell"
        ),
        locked_host_bytes=frame + restart, frame_bytes=frame,
    )


def check_disk(run: RunDirectory, needs: Needs) -> dict[str, Any]:
    free = shutil.disk_usage(run.path).free
    record = {"needed_bytes": needs.disk_bytes, "free_bytes": free, "basis": needs.disk_basis}
    if needs.disk_bytes > DISK_FREE_SHARE * free:
        raise LaunchRefusal(
            "GM1-RUN-DISK",
            f"the run writes about {needs.disk_bytes / 1e9:.1f} GB and {run.path} has "
            f"{free / 1e9:.1f} GB free ({needs.disk_basis})",
            "free some disk, put the run on a larger disk, or write frames less often (tapfrq)",
        )
    return record


def check_gpu_memory(choice: GpuChoice, needs: Needs) -> dict[str, Any]:
    """The memory the run needs against what the card has free *now*."""

    if choice.device is None:
        return {"checked": False, "reason": choice.how}
    try:
        device = next((row for row in query_gpus() if row.uuid == choice.device.uuid), None)
    except GpuQueryError as error:
        return {"checked": False, "reason": str(error)}
    free = None if device is None else device.memory_free_bytes
    if free is None:
        return {"checked": False, "reason": "nvidia-smi did not report the card's memory"}
    record = {"checked": True, "needed_bytes": needs.gpu_bytes, "free_bytes": free,
              "basis": needs.gpu_basis, "memory_used_mib": device.memory_used_mib}
    if needs.gpu_bytes > GPU_FREE_SHARE * free:
        raise LaunchRefusal(
            "GM1-RUN-GPU-MEMORY",
            f"the run needs about {needs.gpu_bytes / 1e9:.1f} GB of GPU memory and {device.name} "
            f"has {free / 1e9:.1f} GB free now ({needs.gpu_basis})",
            "close what else is using the card, use a smaller domain, or pass "
            "--ignore-estimate if you know the estimate is too high",
        )
    return record


def scheduler_from_environment(env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    env = os.environ if env is None else env
    if env.get("SLURM_JOB_ID"):
        return {"kind": "slurm", "job_id": env["SLURM_JOB_ID"],
                "cluster": env.get("SLURM_CLUSTER_NAME"), "node": host_name()}
    if env.get("PBS_JOBID"):
        return {"kind": "pbs", "job_id": env["PBS_JOBID"], "server": env.get("PBS_SERVER"),
                "node": host_name()}
    return None


# ============================================================== supervisor


class _Out:
    """Supervisor lines: the terminal (or the detached log) and the segment's run.log."""

    def __init__(self, stream: TextIO | None) -> None:
        self.stream = stream
        self.log: Any = None

    def __call__(self, line: str) -> None:
        text = f"gpuwm-cm1 run: {line}"
        if self.stream is not None:
            try:
                print(text, file=self.stream, flush=True)
            except (OSError, ValueError):
                self.stream = None
        if self.log is not None:
            try:
                self.log.write((f"[{utc_now()}] {text}\n").encode("utf-8"))
                self.log.flush()
            except (OSError, ValueError):
                pass


@dataclass
class Prepared:
    record: dict[str, Any]
    plan: SegmentPlan
    deck_text: str
    needs: Needs | None
    disk: dict[str, Any] | None
    library: Path | None
    history_publisher: Path | None
    notes: list[str] = field(default_factory=list)


def _previous_tools(run: RunDirectory) -> tuple[Path | None, Path | None]:
    """The explicit native paths the latest segment used, if they still exist."""

    for number in reversed(run.segment_numbers()):
        row = run.segment_record(number) or {}
        library = row.get("library")
        publisher = row.get("history_publisher")
        if library or publisher:
            return (
                Path(library) if library and Path(library).is_file() else None,
                Path(publisher) if publisher and Path(publisher).is_file() else None,
            )
    return None, None


def prepare(run: RunDirectory, request: StartRequest) -> Prepared | None:
    """Every check that needs no GPU; ``None`` when there is nothing left to run."""

    from .status import run_status

    record = run.load()
    status = run_status(run)
    if status.state in ("waiting", "running"):
        raise LaunchRefusal(
            "GM1-RUN-ACTIVE", f"{status.name} is already {status.label.lower()} (segment {status.segment})",
            f"watch it with gpuwm-cm1 status {shlex.quote(os.fspath(run.path))}, or stop it first",
        )
    if status.state == "stale":
        raise LaunchRefusal(
            "GM1-RUN-STALE",
            f"{status.name} was running on {status.host} and has not reported for "
            f"{status.heartbeat_age_s or 0:.0f} s; it may still be running there",
            "check the job there; if it has ended, mark it with gpuwm-cm1 stop RUNDIR --abandon",
        )
    imported = record.get("import") or {}
    if imported.get("result") == "refused" or not run.deck.is_file():
        reasons = imported.get("refused") or ["the namelist was not translated"]
        raise LaunchRefusal(
            "GM1-RUN-IMPORT-REFUSED", "the namelist cannot run as written: " + "; ".join(reasons[:3]),
            "fix the settings the import report lists and import the namelist again",
        )
    acknowledged = {str(item) for item in record.get("acknowledgments") or []}
    missing = [
        item for item in imported.get("acknowledgments_needed") or []
        if str(item.get("id") if isinstance(item, Mapping) else item) not in acknowledged
    ]
    if missing:
        raise LaunchRefusal(
            "GM1-RUN-ACKNOWLEDGE",
            f"{len(missing)} setting(s) need your acknowledgment first: "
            + "; ".join(str(item.get("text") if isinstance(item, Mapping) else item) for item in missing[:3]),
            "acknowledge them on the Check screen, or import again with --acknowledge-output-keys",
        )
    plan = plan_segment(run, record, request)
    if plan is None:
        return None
    library, publisher = request.library, request.history_publisher
    if library is None or publisher is None:
        previous_library, previous_publisher = _previous_tools(run)
        library = library or previous_library
        publisher = publisher or previous_publisher
    request.library, request.history_publisher = library, publisher
    deck_text = segment_deck_text(run, plan, "NNN")
    needs = disk = None
    notes: list[str] = []
    import tomllib

    deck = tomllib.loads(deck_text)
    with tempfile.TemporaryDirectory(prefix="gm1-deck-") as scratch:
        pending = Path(scratch) / DECK
        pending.write_text(deck_text, encoding="utf-8")
        payload = host_plan(
            run, record, pending, request,
            restart_from=None if plan.resumed_from is None else plan.resumed_from.path.parent,
        )
    needs = estimate_needs(deck, payload, plan)
    if request.ignore_estimate:
        notes.append("the disk and GPU memory estimates were not enforced (--ignore-estimate)")
    else:
        disk = check_disk(run, needs)
    return Prepared(record=record, plan=plan, deck_text=deck_text, needs=needs, disk=disk,
                    library=library, history_publisher=publisher, notes=notes)


def _chained(deck_text: str) -> bool:
    """Whether the segment's deck is driven by a WRF or ArWen history (``[chain]``)."""

    import tomllib

    try:
        return "chain" in tomllib.loads(deck_text)
    except ValueError:
        return False


def child_command(run: RunDirectory, segment: Segment, prepared: Prepared) -> list[str]:
    module = os.environ.get(CHILD_MODULE_ENV) or "cm1core"
    plan = prepared.plan
    command = [
        sys.executable, "-m", module, "cartesian", os.fspath(segment.deck),
        "--output", os.fspath(segment.receipt), "--integrate",
        "--history-dir", os.fspath(segment.frames_dir),
        *([] if plan.restarts_expected == 0 and _chained(prepared.deck_text)
          else ["--restart-dir", os.fspath(segment.restarts_dir)]),
        "--events", os.fspath(segment.events),
        "--control-dir", os.fspath(run.control),
        "--history-envelope", "off",
        *_sounding_arguments(run, prepared.record),
    ]
    if plan.history_initial:
        command.append("--history-initial")
    if plan.resumed_from is not None:
        command += ["--restart-from", os.fspath(plan.resumed_from.path.parent)]
    if prepared.library is not None:
        command += ["--library", os.fspath(prepared.library)]
    if prepared.history_publisher is not None:
        command += ["--history-publisher", os.fspath(prepared.history_publisher)]
    return command


class Supervisor:
    """One segment, from the queue to its end record."""

    def __init__(self, run: RunDirectory, request: StartRequest, *, stream: TextIO | None = None) -> None:
        self.run = run
        self.request = request
        self.say = _Out(sys.stderr if stream is None else stream)
        self.started = time.time()
        self.signal_received: int | None = None
        self.signal_count = 0
        self.child: subprocess.Popen[bytes] | None = None
        self._forwarded = False
        self.segment: Segment | None = None
        self.queue: GpuQueue | None = None
        self.active_fd: int | None = None
        self.log_handle: Any = None

    # ---- signals

    def _on_signal(self, number: int, _frame: Any) -> None:
        # Only flags here: the loops act on them.  The first signal asks the
        # run to stop through its control file (a clean stop at the next
        # step boundary; a scheduler's SIGTERM reaches the child directly
        # anyway, and a second SIGTERM could cut its cleanup short); a second
        # signal, or SIGNAL_STOP_GRACE_S without an exit, sends SIGTERM.
        self.signal_received = number
        self.signal_count += 1

    def _forward(self) -> None:
        child = self.child
        if child is not None and not self._forwarded and child.poll() is None:
            self._forwarded = True
            try:
                child.send_signal(signal.SIGTERM)
            except OSError:
                pass

    # ---- the directory's single owner

    def _own_directory(self) -> None:
        from .status import ACTIVE_STATES, run_status

        fd = os.open(self.run.active_lock, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + OWNER_FINISHING_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                status = run_status(self.run)
                # A segment whose end record is written has a supervisor that
                # is still filing it for a moment: wait for that, not for a run.
                if status.state not in ACTIVE_STATES and time.monotonic() < deadline:
                    time.sleep(0.1)
                    continue
            os.close(fd)
            raise LaunchRefusal(
                "GM1-RUN-ACTIVE",
                f"{status.name} is already {status.label.lower()}"
                + ("" if status.segment is None else f" (segment {status.segment})")
                + "; another gpuwm-cm1 process owns this run directory",
                f"watch it with gpuwm-cm1 status {shlex.quote(os.fspath(self.run.path))}, or stop it first",
            )
        self.active_fd = fd

    # ---- control requests before the runtime reads the directory

    def _set_aside_stale_requests(self) -> list[str]:
        noted = []
        for name in ("stop", "pause"):
            path = self.run.control / name
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < self.started - STALE_REQUEST_GRACE_S:
                self.run.logs.mkdir(exist_ok=True)
                target = self.run.logs / f"control-{name}.stale-{int(mtime)}"
                os.replace(path, target)
                noted.append(f"set aside a {name} request left by an earlier run ({target.name})")
        return noted

    def _hold_requests(self, held: set[str]) -> None:
        """Move new request files aside until the runtime admits the directory."""

        assert self.segment is not None
        for name in ("stop", "pause"):
            path = self.run.control / name
            if path.exists():
                try:
                    os.replace(path, self.segment.path / f"held-{name}")
                except OSError:
                    continue
                if name not in held:
                    self.say(f"{name} requested; it is handed to the run once it is reading its control directory")
                held.add(name)

    def _release_requests(self, held: set[str]) -> None:
        assert self.segment is not None
        for name in sorted(held):
            source = self.segment.path / f"held-{name}"
            try:
                os.replace(source, self.run.control / name)
            except OSError:
                (self.run.control / name).touch()
        held.clear()

    # ---- records

    def _row(self, **fields: Any) -> None:
        number = self.segment.number

        def mutate(record: dict[str, Any]) -> None:
            for row in record.setdefault("segments", []):
                if row.get("number") == number:
                    row.update(fields)
                    return
            record["segments"].append({"number": number, **fields})

        self.run.update(mutate)

    def _end_before_start(self, *, reason: str, rc: int, message: str, request: str | None = None,
                          signal_name: str | None = None, refusal: LaunchRefusal | None = None,
                          stopped_before_start: bool = False) -> int:
        """The segment ends before a runtime started: its log gets the end record."""

        from ..run_events import RunEventLog

        segment = self.segment
        try:
            log = RunEventLog(segment.events)
        except FileExistsError:
            log = None
        if log is not None:
            if refusal is not None:
                log.emit("refused", code=refusal.code, gate="gpuwm-cm1 run", reason=refusal.reason,
                         remedy=refusal.remedy, option=None)
            log.emit("end", reason=reason, rc=int(rc), receipt=None, step=None, model_time_s=None,
                     frames_published=0, restarts_published=0, signal=signal_name, request=request,
                     message=message)
            log.close()
        ended = utc_now()

        def mutate(state: dict[str, Any]) -> None:
            state.update({"state": "ended", "ended_utc": ended, "rc": int(rc), "reason": reason,
                          "stopped_before_start": stopped_before_start})

        segment.write_state(mutate)
        self._row(ended_utc=ended, rc=int(rc), reason=reason)
        from .control import archive_requests

        archive_requests(self.run, segment, note="before-start")
        for name in ("stop", "pause"):
            held = segment.path / f"held-{name}"
            if held.exists():
                os.replace(held, segment.path / f"control-{name}.before-start")
        self.say(message)
        return int(rc)

    # ---- the whole segment

    def run_segment(self) -> int:
        previous = {
            number: signal.signal(number, self._on_signal)
            for number in (signal.SIGTERM, signal.SIGINT)
        }
        previous_hup = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            return self._run_segment()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            signal.signal(signal.SIGHUP, previous_hup)
            if self.queue is not None:
                self.queue.release()
            if self.log_handle is not None:
                self.say.log = None
                self.log_handle.close()
            if self.active_fd is not None:
                # Close only (see GpuQueue.release): a child still running
                # keeps the run directory owned through its inherited copy.
                os.close(self.active_fd)
                self.active_fd = None

    def _run_segment(self) -> int:
        run = self.run
        self._own_directory()
        prepared = prepare(run, self.request)
        if prepared is None:
            self.say(f"{run.name()} has nothing left to run; it is finished")
            return 0
        if self.signal_received is not None:
            self.say("stopped before it started; nothing ran")
            return 128 + self.signal_received
        plan = prepared.plan
        for note in self._set_aside_stale_requests() + prepared.notes:
            self.say(note)
        run.control.mkdir(exist_ok=True)
        run.logs.mkdir(exist_ok=True)
        choice = choose_gpu(self.request.gpu)
        scheduler = scheduler_from_environment()
        segment = run.new_segment()
        self.segment = segment
        write_text_atomic(segment.deck, prepared.deck_text.replace("Segment NNN", f"Segment {segment.name}", 1))
        log_handle = open(segment.log, "ab")
        self.log_handle = log_handle
        self.say.log = log_handle
        supervisor = process_identity(os.getpid())
        segment.write_state(lambda state: state.update({
            "schema": "gm1.cm1-run-segment/v1",
            "number": segment.number,
            "state": "waiting-for-gpu",
            "created_utc": utc_now(),
            "heartbeat_utc": utc_now(),
            "host": supervisor["host"],
            "boot_id": supervisor["boot_id"],
            "supervisor": supervisor,
            "child": None,
            "gpu": choice.as_dict(),
            "scheduler": scheduler,
            "plan": plan.as_dict(),
            "needs": None if prepared.needs is None else prepared.needs.as_dict(),
            "disk": prepared.disk,
        }))
        self._row(
            directory=os.fspath(segment.path.relative_to(run.path)), kind=plan.kind,
            resumed_from=plan.as_dict()["resumed_from"], until_s=plan.until_s,
            restart_interval_s=plan.restart_interval_s, restart_rule=plan.restart_rule,
            host=supervisor["host"], boot_id=supervisor["boot_id"], pid=supervisor["pid"],
            pid_start_ticks=supervisor["start_ticks"], started_utc=utc_now(),
            gpu=choice.as_dict(), scheduler=scheduler, gm1=installation(),
            library=None if prepared.library is None else os.fspath(prepared.library),
            history_publisher=None if prepared.history_publisher is None else os.fspath(prepared.history_publisher),
            ended_utc=None, rc=None, reason=None,
        )
        run.update(lambda record: record.update({
            "host": supervisor["host"], "gpu": choice.as_dict(), "scheduler": scheduler,
        }))
        where = choice.device.name if choice.device is not None else "the GPU"
        self.say(
            f"segment {segment.name}: {plan.kind} run to {plan.until_s:g} s"
            + ("" if plan.resumed_from is None else
               f" from restart {plan.resumed_from.number} ({plan.start_model_time_s:g} s)")
            + f" on {where}; restarts {plan.restart_rule}"
        )

        # ---- the queue
        self.queue = GpuQueue(choice.key, {
            "rundir": os.fspath(run.path), "segment": segment.number, "host": supervisor["host"],
            "run_id": f"gm1-run|{run.path}|{segment.name}",
        })
        self.queue.join()
        held: set[str] = set()
        announced = False
        last_beat = 0.0
        while not self.queue.try_acquire():
            if self.signal_received is not None:
                return self._end_before_start(
                    reason="stopped", rc=128 + self.signal_received, signal_name=signal.Signals(self.signal_received).name,
                    message="stopped while waiting for the GPU; nothing ran", stopped_before_start=True,
                )
            self._hold_requests(held)
            if "stop" in held:
                return self._end_before_start(
                    reason="stopped", rc=0, request="stop",
                    message="stopped while waiting for the GPU; nothing ran", stopped_before_start=True,
                )
            if not self.request.wait:
                refusal = LaunchRefusal(
                    "GM1-RUN-GPU-BUSY", f"{where} is running another job",
                    "start it again without --no-wait to wait in line",
                )
                return self._end_before_start(reason="refused", rc=2, message=refusal.reason, refusal=refusal)
            now = time.monotonic()
            if now - last_beat >= HEARTBEAT_S:
                last_beat = now
                ahead = len(self.queue.ahead())
                holder = self.queue.holder()
                segment.write_state(lambda state: state.update({
                    "heartbeat_utc": utc_now(),
                    "queue": {"lock": os.fspath(self.queue.path), "ahead": ahead, "holder": holder},
                }))
                if not announced:
                    announced = True
                    who = (holder or {}).get("rundir") or f"process {(holder or {}).get('pid', '?')}"
                    self.say(f"waiting for {where}: it is running {who}")
            time.sleep(QUEUE_POLL_S)

        # ---- the GPU is ours: its memory, then the child
        try:
            memory = {"checked": False, "reason": "--ignore-estimate"} if self.request.ignore_estimate else (
                check_gpu_memory(choice, prepared.needs)
            )
        except LaunchRefusal as refusal:
            return self._end_before_start(reason="refused", rc=2, message=refusal.reason, refusal=refusal)
        self._hold_requests(held)
        if "stop" in held and self.signal_received is None:
            return self._end_before_start(reason="stopped", rc=0, request="stop",
                                          message="stopped before it started; nothing ran",
                                          stopped_before_start=True)
        if self.signal_received is not None:
            return self._end_before_start(
                reason="stopped", rc=128 + self.signal_received,
                signal_name=signal.Signals(self.signal_received).name,
                message="stopped before it started; nothing ran", stopped_before_start=True,
            )
        command = child_command(run, segment, prepared)
        env = child_environment(choice.cuda_visible_devices)
        fds = tuple(fd for fd in (self.queue.fd, self.active_fd) if fd is not None)
        segment.write_state(lambda state: state.update({
            "state": "starting", "heartbeat_utc": utc_now(), "gpu_memory": memory,
            "queue": {"lock": os.fspath(self.queue.path), "acquired_utc": utc_now()},
            "command": command,
        }))
        self.child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
            env=env, cwd=run.path, pass_fds=fds, process_group=0,
        )
        child = process_identity(self.child.pid)
        segment.write_state(lambda state: state.update({"child": child}))
        self._row(child_pid=child["pid"], child_start_ticks=child["start_ticks"], command=command)
        self.say(f"started: {shlex.join(command)}")
        return self._follow(prepared, held)

    def _follow(self, prepared: Prepared, held: set[str]) -> int:
        segment = self.segment
        tail = EventTail(segment.events)
        summary = SegmentEvents()
        admitted = False
        running_state = False
        last_progress = time.monotonic()
        last_beat = time.monotonic()
        signals_handled = 0
        stop_asked_at: float | None = None
        while True:
            if self.signal_count > signals_handled:
                signals_handled = self.signal_count
                if stop_asked_at is None:
                    stop_asked_at = time.monotonic()
                    try:
                        (self.run.control / "stop").touch()
                    except OSError:
                        self._forward()
                    self.say(
                        f"{signal.Signals(self.signal_received).name}: stop requested; the run stops "
                        "at the end of its current step (send it again to end it at once)"
                    )
                else:
                    self._forward()
            if stop_asked_at is not None and time.monotonic() - stop_asked_at > SIGNAL_STOP_GRACE_S:
                self._forward()
            returncode = self.child.poll()
            records = tail.poll()
            summary.add(records)
            for record in records:
                self._report(record, summary)
            if not admitted:
                if summary.admitted is not None:
                    admitted = True
                    if held:
                        self._release_requests(held)
                else:
                    self._hold_requests(held)
                    # The runtime writes its own heartbeat from admission on;
                    # until then (imports, configuration checks on a slow
                    # shared filesystem) the supervisor keeps the segment
                    # from reading Stale on another host.  Only until then:
                    # a wedged runtime must still go stale.
                    if time.monotonic() - last_beat >= HEARTBEAT_S:
                        last_beat = time.monotonic()
                        segment.write_state(lambda state: state.update({"heartbeat_utc": utc_now()}))
            if admitted and not running_state and summary.last_progress is not None:
                running_state = True
                segment.write_state(lambda state: state.update({"state": "running"}))
            now = time.monotonic()
            if summary.last_progress is not None and now - last_progress >= self.request.progress_every_s:
                last_progress = now
                self._progress_line(summary.last_progress)
            if returncode is not None:
                break
            time.sleep(PREADMISSION_POLL_S if not admitted else 0.5)
        for record in tail.poll():
            summary.add([record])
            self._report(record, summary)
        return self._finish(summary, held)

    def _report(self, record: Mapping[str, Any], summary: SegmentEvents) -> None:
        kind = record.get("kind")
        if kind == "admitted":
            self.say(f"admitted: {record.get('steps')} steps of {record.get('dt_s')} s, "
                     f"{record.get('frames_expected')} frame(s), {record.get('restarts_expected')} restart(s)")
        elif kind == "frame":
            self.say(f"frame {record.get('index')} at {record.get('model_time_s'):g} s")
        elif kind == "restart":
            self.say(f"restart {record.get('number')} at {record.get('model_time_s'):g} s")
        elif kind == "warning":
            self.say(f"warning: {record.get('message')}")
        elif kind == "refused":
            from .events import plain_record

            plain = plain_record(record)
            self.say(f"refused: {plain.get('reason')} -- {plain.get('remedy')}")

    def _progress_line(self, progress: Mapping[str, Any]) -> None:
        step, total = progress.get("step"), progress.get("total_steps")
        if step is None:
            self.say(f"{progress.get('phase', 'starting')}")
            return
        words = f"step {step}" + (f"/{total} ({100.0 * step / total:.0f}%)" if total else "")
        if progress.get("model_time_s") is not None:
            words += f", {progress['model_time_s']:g} s of storm time"
        if progress.get("s_per_step_recent"):
            words += f", {progress['s_per_step_recent']:.3f} s/step"
        if progress.get("eta_s") is not None:
            words += f", about {progress['eta_s']:.0f} s left"
        self.say(words)

    def _finish(self, summary: SegmentEvents, held: set[str]) -> int:
        segment = self.segment
        returncode = int(self.child.returncode)
        end = summary.end
        reason = end.get("reason") if end else "no-final-record"
        override = None
        if (
            end is not None and end.get("reason") in ("refused", "error")
            and (summary.refused or {}).get("code") == CONTROL_REFUSAL_CODE
        ):
            # Stale requests were set aside before the start, so a request
            # the runtime refused was made in the instant before it admitted
            # its control directory: the user asked it to stop, and nothing ran.
            override = "stopped"
        ended = utc_now()

        def mutate(state: dict[str, Any]) -> None:
            state.update({"state": "ended", "ended_utc": ended, "rc": returncode, "reason": reason})
            if override:
                state["reason_override"] = override

        segment.write_state(mutate)
        self._row(ended_utc=ended, rc=returncode, reason=override or reason)
        from .control import archive_requests

        note = "consumed" if end is not None and end.get("request") else "unused"
        archive_requests(self.run, segment, note=note)
        for name in list(held):
            source = segment.path / f"held-{name}"
            if source.exists():
                os.replace(source, segment.path / f"control-{name}.before-start")
        if end is None:
            how = f"signal {-returncode}" if returncode < 0 else f"exit code {returncode}"
            self.say(f"segment {segment.name} ended ({how}) without a final record; see {segment.log}")
        else:
            self.say(
                f"segment {segment.name} ended: {override or end.get('reason')} at "
                f"{(end.get('model_time_s') or 0):g} s, rc {end.get('rc')}; "
                f"{end.get('frames_published', 0)} frame(s), {end.get('restarts_published', 0)} restart(s)"
            )
        return returncode if returncode >= 0 else 128 - returncode


def supervise(path: Path | str, request: StartRequest, *, stream: TextIO | None = None) -> int:
    return Supervisor(RunDirectory(path), request, stream=stream).run_segment()


# =================================================================== detach


def spawn_detached(path: Path | str, arguments: list[str], request: StartRequest, *,
                   wait_s: float = 15.0) -> dict[str, Any]:
    """Check the gates here, then start the supervisor in its own session and return.

    ``arguments`` are the command's words after ``gpuwm-cm1`` without
    ``--detach``.  The supervisor's output goes to ``RUNDIR/logs``.  Waits up
    to ``wait_s`` for the new segment to appear so the caller can show it.
    """

    run = RunDirectory(path)
    prepared = prepare(run, request)
    command = shlex.join(["gpuwm-cm1", *arguments])
    if prepared is None:
        return {"started": False, "message": f"{run.name()} has nothing left to run.", "command": command}
    run.logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log = run.logs / f"supervisor-{stamp}-{os.getpid()}.log"
    before = set(run.segment_numbers())
    with open(log, "ab") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "cm1core", *arguments],
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=child_environment(None), cwd=run.path, close_fds=True,
        )
    # Under WSL, Ubuntu stays up until the supervisor ends, whichever window closes (best effort).
    from .wsl_keepalive import hold

    holder = hold(process.pid)
    deadline = time.monotonic() + wait_s
    segment = None
    while time.monotonic() < deadline:
        new = sorted(set(run.segment_numbers()) - before)
        if new and run.segment(new[-1]).state_path.exists():
            segment = run.segment(new[-1])
            break
        if process.poll() is not None:
            break
        time.sleep(0.1)
    result: dict[str, Any] = {
        "started": process.poll() is None or segment is not None,
        "pid": process.pid,
        "log": os.fspath(log),
        "command": command,
        "segment": None if segment is None else segment.number,
    }
    if process.poll() is not None and segment is None:
        result["rc"] = process.returncode
        try:
            result["message"] = log.read_text(encoding="utf-8", errors="replace").strip()[-2000:]
        except OSError:
            result["message"] = ""
    else:
        from .status import run_status

        status = run_status(run)
        result["state"] = status.state
        result["message"] = f"{status.name}: {status.label}. {status.detail}".strip()
    warning = holder.warning() if holder is not None and result["started"] else None
    if warning:
        result["warning"] = warning
        result["message"] = f"{result['message']}\n{warning}".strip()
    return result


__all__ = [
    "CHILD_MODULE_ENV",
    "GpuChoice",
    "GpuDevice",
    "GpuQueue",
    "GpuQueryError",
    "LaunchRefusal",
    "LOCK_ROOT_ENV",
    "NVIDIA_SMI_ENV",
    "Needs",
    "SegmentPlan",
    "StartRequest",
    "Supervisor",
    "check_disk",
    "check_gpu_memory",
    "child_command",
    "choose_gpu",
    "estimate_needs",
    "lock_path",
    "parse_seconds",
    "plan_segment",
    "prepare",
    "query_gpus",
    "spawn_detached",
    "supervise",
]
