"""The run directory, ``gm1.cm1-run/v1``.

::

    RUNDIR/
      namelist.input.original   the user's file, byte copy (sha256 in run.json)
      namelist.input            the run's namelist
      input_sounding            isnd 7/17 only
      import-report.json/.txt   coverage report
      settings.toml             values CM1 keeps in Fortran, output choices
      deck.toml                 generated gpuwm-cm1 document; never hand-edited
      estimate.json geometry.json
      run.json                  this module's record: version, host, GPU,
                                scheduler job, acknowledgments, segments[]
      control/                  stop, pause   (checked once per large step)
      segments/NNN/             deck.toml segment.json events.jsonl run.log
                                receipt.json frames/ restarts/
      cm1out/ quicklook/ reproduce/ logs/

A *segment* is one process's integration.  The door refuses to reuse a
history directory or a receipt, so every start and every resume gets its own
``segments/NNN``.  ``run.json`` lists the segments; each segment's
``segment.json`` holds its live state (waiting, running, ended), the process
identities crash detection checks, and the heartbeat of a run that is still
waiting for its GPU.

A resumed segment continues from a restart of an earlier one.  The frames
and restarts that form the run are the latest segment's and, before its
resume point, its ancestors' (:meth:`RunDirectory.frames`).

Importing this module imports neither CuPy nor numpy.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

RUN_SCHEMA = "gm1.cm1-run/v1"
SEGMENT_SCHEMA = "gm1.cm1-run-segment/v1"

RUN_JSON = "run.json"
NAMELIST_ORIGINAL = "namelist.input.original"
NAMELIST = "namelist.input"
SOUNDING = "input_sounding"
IMPORT_REPORT_JSON = "import-report.json"
IMPORT_REPORT_TEXT = "import-report.txt"
SETTINGS = "settings.toml"
DECK = "deck.toml"
ESTIMATE = "estimate.json"
GEOMETRY = "geometry.json"
CONTROL = "control"
SEGMENTS = "segments"
CM1OUT = "cm1out"
QUICKLOOK = "quicklook"
REPRODUCE = "reproduce"
LOGS = "logs"
COMMANDS_LOG = "commands.log"
#: Held (flock) by the one supervisor that owns this run directory.
ACTIVE_LOCK = ".run.lock"
#: Held (flock) around every read-modify-write of run.json.
RECORD_LOCK = ".run.json.lock"

SEGMENT_FORMAT = "{number:03d}"
SEGMENT_JSON = "segment.json"
SEGMENT_DECK = "deck.toml"
SEGMENT_EVENTS = "events.jsonl"
SEGMENT_LOG = "run.log"
SEGMENT_RECEIPT = "receipt.json"
SEGMENT_FRAMES = "frames"
SEGMENT_RESTARTS = "restarts"

# The door's own names (cuda_cartesian_runtime.py:151-156, cartesian_restart.py:192-196,
# cartesian_history.py:116-123), repeated here so reading a run directory never
# imports the runtime.
FRAME_STAGING = ".incomplete"
FRAME_PATTERN = re.compile(r"^frame-(\d{6})$")
RESTART_PATTERN = re.compile(r"^restart-(\d{6})$")
RESTART_RECEIPT = "restart-publication.json"
FRAME_RECEIPT = "history-publication.json"

#: Segment states a supervisor writes.
SEGMENT_STATES = ("waiting-for-gpu", "starting", "running", "ended")

#: The ``import`` block's results: the run may start, it needs the user's
#: acknowledgment first, or it cannot run as written.
IMPORT_RESULTS = ("runnable", "acknowledge", "refused")


class RunDirectoryError(ValueError):
    """A path that is not a usable run directory; the message says what to do."""


# ---------------------------------------------------------------- small helpers


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_utc(text: Any) -> float | None:
    """Seconds since the epoch of an ISO-8601 time this package wrote, or None."""

    if not isinstance(text, str) or not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    try:
        handle = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON beside ``path`` and rename it into place (never a torn file)."""

    path = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    fsync_directory(path.parent)


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_or_none(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


@contextmanager
def flocked(path: Path) -> Iterator[int]:
    """An exclusive ``flock`` on ``path`` (created if absent) for the block."""

    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield handle
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


# ------------------------------------------------------------ host identity


def host_name() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def boot_id() -> str | None:
    """This boot's id: a pid from before a reboot names another process."""

    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip() or None
    except OSError:
        return None


def process_start_ticks(pid: int) -> int | None:
    """Field 22 of ``/proc/<pid>/stat``: when the process started, in clock ticks.

    A pid alone is reused; the pid with its start time names one process.
    ``None`` when there is no such process (or no ``/proc``).
    """

    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii", errors="replace")
    except (OSError, ValueError):
        return None
    # The command name (field 2) is in parentheses and may hold spaces.
    closing = text.rfind(")")
    if closing < 0:
        return None
    fields = text[closing + 2:].split()
    # fields[0] is field 3 (state); field 22 is fields[19].
    try:
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def process_state(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii", errors="replace")
    except (OSError, ValueError):
        return None
    closing = text.rfind(")")
    return text[closing + 2:].split()[0] if closing >= 0 else None


def process_identity(pid: int) -> dict[str, Any]:
    """What crash detection needs to recognise this process later."""

    return {
        "pid": int(pid),
        "start_ticks": process_start_ticks(pid),
        "host": host_name(),
        "boot_id": boot_id(),
    }


def local_process_alive(identity: Mapping[str, Any] | None) -> bool:
    """Whether the process ``identity`` names is still running on this host.

    Reads ``/proc``: call it only for a process recorded on this host.  A
    zombie (exited, not yet reaped) is not alive.
    """

    if not identity or not identity.get("pid"):
        return False
    recorded_boot = identity.get("boot_id")
    if recorded_boot is not None and boot_id() not in (None, recorded_boot):
        return False
    ticks = process_start_ticks(int(identity["pid"]))
    if ticks is None:
        return False
    if identity.get("start_ticks") is not None and int(identity["start_ticks"]) != ticks:
        return False
    return process_state(int(identity["pid"])) not in ("Z", "X")


def installation() -> dict[str, Any]:
    """Which gpuwm-cm1 wrote a record: version, and the wheel or checkout it came from."""

    from .. import _version

    package = Path(__file__).resolve().parents[1]
    record: dict[str, Any] = {"version": _version.__version__, "package": os.fspath(package)}
    try:
        from importlib import metadata

        distribution = metadata.distribution("gpuwm-cm1")
        files = distribution.files or ()
        located = next(
            (Path(distribution.locate_file(item)) for item in files
             if str(item).endswith(".dist-info/RECORD")), None,
        )
        if located is not None and located.is_file() and Path(distribution.locate_file("cm1core")).resolve() == package:
            record["install"] = "wheel"
            record["distribution_version"] = distribution.version
            # RECORD pins every installed file's hash: the installed wheel's identity.
            record["record_sha256"] = sha256_file(located)
            return record
    except Exception:  # noqa: BLE001 - provenance is best effort
        pass
    record["install"] = "checkout"
    head = package.parents[1] / ".git"
    commit = _git_head(head)
    if commit is not None:
        record["git_commit"] = commit
    return record


def _git_head(git: Path) -> str | None:
    try:
        if git.is_file():  # a worktree's .git file names the real directory
            text = git.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                git = (git.parent / text.split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        for root in (git, _common_dir(git)):
            if root is None:
                continue
            candidate = root / ref
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
            packed = root / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(" " + ref):
                        return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


def _common_dir(git: Path) -> Path | None:
    try:
        text = (git / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return (git / text).resolve()


# ----------------------------------------------------------------- records


@dataclass(frozen=True)
class FrameInfo:
    """One published frame of a run."""

    index: int
    model_time_s: float | None
    step: int | None
    path: Path
    segment: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "model_time_s": self.model_time_s, "step": self.step,
            "path": os.fspath(self.path), "segment": self.segment,
        }


@dataclass(frozen=True)
class RestartInfo:
    """One durable restart: renamed into place with its publication receipt."""

    number: int
    model_time_s: float | None
    step: int | None
    next_history_index: int | None
    path: Path
    segment: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number, "model_time_s": self.model_time_s, "step": self.step,
            "next_history_index": self.next_history_index, "path": os.fspath(self.path),
            "segment": self.segment,
        }


class Segment:
    """``segments/NNN``: one process's integration of the run."""

    def __init__(self, run: "RunDirectory", number: int) -> None:
        self.run = run
        self.number = int(number)
        self.path = run.segments_root / SEGMENT_FORMAT.format(number=self.number)

    def __repr__(self) -> str:
        return f"Segment({os.fspath(self.path)!r})"

    @property
    def name(self) -> str:
        return SEGMENT_FORMAT.format(number=self.number)

    @property
    def state_path(self) -> Path:
        return self.path / SEGMENT_JSON

    @property
    def deck(self) -> Path:
        return self.path / SEGMENT_DECK

    @property
    def events(self) -> Path:
        return self.path / SEGMENT_EVENTS

    @property
    def log(self) -> Path:
        return self.path / SEGMENT_LOG

    @property
    def receipt(self) -> Path:
        return self.path / SEGMENT_RECEIPT

    @property
    def frames_dir(self) -> Path:
        return self.path / SEGMENT_FRAMES

    @property
    def restarts_dir(self) -> Path:
        return self.path / SEGMENT_RESTARTS

    def state(self) -> dict[str, Any]:
        """``segment.json``; empty if it has not been written (or is unreadable)."""

        payload = read_json_or_none(self.state_path)
        return payload if isinstance(payload, dict) else {}

    def write_state(self, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """Read, change and atomically rewrite ``segment.json`` (one writer: its supervisor)."""

        state = self.state() or {"schema": SEGMENT_SCHEMA, "number": self.number}
        mutate(state)
        write_json_atomic(self.state_path, state)
        return state

    def frame_events(self) -> dict[int, Mapping[str, Any]]:
        from .events import read_records

        return {
            int(record["index"]): record
            for record in read_records(self.events)
            if record.get("kind") == "frame" and isinstance(record.get("index"), int)
        }

    def frames(self) -> list[FrameInfo]:
        """Frames renamed into place in this segment, in index order."""

        found: list[FrameInfo] = []
        try:
            entries = sorted(os.scandir(self.frames_dir), key=lambda entry: entry.name)
        except OSError:
            return found
        recorded = None
        for entry in entries:
            match = FRAME_PATTERN.match(entry.name)
            if match is None or not entry.is_dir(follow_symlinks=False):
                continue
            index = int(match.group(1))
            if recorded is None:
                recorded = self.frame_events()
            record = recorded.get(index)
            model_time = step = None
            if record is not None:
                model_time, step = record.get("model_time_s"), record.get("step")
            else:
                receipt = read_json_or_none(Path(entry.path) / FRAME_RECEIPT)
                if isinstance(receipt, dict):
                    model_time, step = receipt.get("model_time_s"), receipt.get("step")
            found.append(FrameInfo(
                index=index,
                model_time_s=None if model_time is None else float(model_time),
                step=None if step is None else int(step),
                path=Path(entry.path), segment=self.number,
            ))
        return found

    def restarts(self) -> list[RestartInfo]:
        """Durable restarts of this segment, in number order."""

        found: list[RestartInfo] = []
        try:
            entries = sorted(os.scandir(self.restarts_dir), key=lambda entry: entry.name)
        except OSError:
            return found
        for entry in entries:
            match = RESTART_PATTERN.match(entry.name)
            if match is None or not entry.is_dir(follow_symlinks=False):
                continue
            receipt = read_json_or_none(Path(entry.path) / RESTART_RECEIPT)
            if not isinstance(receipt, dict):
                continue
            found.append(RestartInfo(
                number=int(match.group(1)),
                model_time_s=_number(receipt.get("model_time_s")),
                step=_integer(receipt.get("step")),
                next_history_index=_integer(receipt.get("next_history_index")),
                path=Path(entry.path), segment=self.number,
            ))
        return found

    def staging(self) -> list[Path]:
        """Staging directories a killed writer can leave (``.incomplete/``)."""

        return [
            directory / FRAME_STAGING
            for directory in (self.frames_dir, self.restarts_dir)
            if (directory / FRAME_STAGING).exists()
        ]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def segment_of(row: Mapping[str, Any]) -> int | None:
    """The segment a drawn or converted time came from: its record's ``segment``, else the one its frame's key
    (``segments/NNN/frames/frame-XXXXXX``) names; None when neither says."""

    segment = row.get("segment")
    if isinstance(segment, int) and not isinstance(segment, bool):
        return segment
    parts = str(row.get("frame") or "").split("/")
    if len(parts) >= 2 and parts[0] == SEGMENTS and parts[1].isdigit():
        return int(parts[1])
    return None


def off_the_line(line: Mapping[int, float | None], row: Mapping[str, Any], time_s: Any = None) -> bool:
    """Whether a drawn or converted time (its frame's key, segment and time) belongs to a branch the run has left,
    by the run's :meth:`RunDirectory.recorded_line`: its segment is not on the line, or it is past the time the
    segment's descendant resumed from.  A row that names no segment or time is never taken to be off it."""

    if not line:
        return False
    segment = segment_of(row)
    value = row.get("time")
    try:
        t = float(time_s if value is None else value)
    except (TypeError, ValueError):
        return False
    if segment is None or not math.isfinite(t):
        return False
    if segment not in line:
        return True
    until = line[segment]
    return until is not None and t > until + 1e-6


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


class RunDirectory:
    """One run directory; every path of the contract, and its records."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().absolute()

    def __repr__(self) -> str:
        return f"RunDirectory({os.fspath(self.path)!r})"

    # ---- paths

    @property
    def run_json(self) -> Path:
        return self.path / RUN_JSON

    @property
    def namelist(self) -> Path:
        return self.path / NAMELIST

    @property
    def namelist_original(self) -> Path:
        return self.path / NAMELIST_ORIGINAL

    @property
    def sounding(self) -> Path:
        return self.path / SOUNDING

    @property
    def deck(self) -> Path:
        return self.path / DECK

    @property
    def control(self) -> Path:
        return self.path / CONTROL

    @property
    def segments_root(self) -> Path:
        return self.path / SEGMENTS

    @property
    def logs(self) -> Path:
        return self.path / LOGS

    @property
    def estimate(self) -> Path:
        return self.path / ESTIMATE

    @property
    def import_report(self) -> Path:
        return self.path / IMPORT_REPORT_JSON

    @property
    def active_lock(self) -> Path:
        return self.path / ACTIVE_LOCK

    # ---- the record

    def is_run(self) -> bool:
        return self.run_json.is_file()

    def load(self) -> dict[str, Any]:
        try:
            payload = read_json(self.run_json)
        except FileNotFoundError:
            raise RunDirectoryError(
                f"{self.path} has no {RUN_JSON}; it is not a run directory. Create one with "
                "gpuwm-cm1 import NAMELIST --out DIR"
            ) from None
        except (OSError, ValueError) as error:
            raise RunDirectoryError(f"{self.run_json} cannot be read: {error}") from None
        if not isinstance(payload, dict) or payload.get("schema") != RUN_SCHEMA:
            raise RunDirectoryError(
                f"{self.run_json} is not a {RUN_SCHEMA} record "
                f"(schema {payload.get('schema') if isinstance(payload, dict) else None!r})"
            )
        return payload

    def update(self, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """Read-modify-write ``run.json`` under its lock, renamed into place."""

        with flocked(self.path / RECORD_LOCK):
            payload = self.load()
            mutate(payload)
            write_json_atomic(self.run_json, payload)
        return payload

    def name(self) -> str:
        try:
            return str(self.load().get("name") or self.path.name)
        except RunDirectoryError:
            return self.path.name

    # ---- segments

    def segment_numbers(self) -> list[int]:
        try:
            names = os.listdir(self.segments_root)
        except OSError:
            return []
        return sorted(int(name) for name in names if re.fullmatch(r"\d{3,}", name))

    def segment(self, number: int) -> Segment:
        return Segment(self, number)

    def segments(self) -> list[Segment]:
        return [Segment(self, number) for number in self.segment_numbers()]

    def latest_segment(self) -> Segment | None:
        numbers = self.segment_numbers()
        return Segment(self, numbers[-1]) if numbers else None

    def new_segment(self) -> Segment:
        """Create the next ``segments/NNN`` (exclusively: two starts never share one)."""

        self.segments_root.mkdir(parents=True, exist_ok=True)
        number = (self.segment_numbers() or [0])[-1] + 1
        while True:
            segment = Segment(self, number)
            try:
                segment.path.mkdir()
            except FileExistsError:
                number += 1
                continue
            return segment

    def segment_record(self, number: int) -> dict[str, Any] | None:
        try:
            rows = self.load().get("segments") or []
        except RunDirectoryError:
            return None
        for row in rows:
            if isinstance(row, dict) and row.get("number") == number:
                return row
        return None

    # ---- what forms the run

    def lineage(self, segment: Segment | None = None) -> list[tuple[Segment, float | None]]:
        """The latest segment (or ``segment``) and its ancestors, oldest first.

        Each entry is ``(segment, until_s)``: frames of an ancestor count up
        to the model time its descendant resumed from; the last entry's
        ``until_s`` is ``None`` (all of its frames).
        """

        segment = segment or self.latest_segment()
        if segment is None:
            return []
        chain: list[tuple[Segment, float | None]] = [(segment, None)]
        seen = {segment.number}
        current = segment
        while True:
            record = self.segment_record(current.number) or current.state()
            source = record.get("resumed_from") if isinstance(record, dict) else None
            if not isinstance(source, dict) or not isinstance(source.get("segment"), int):
                break
            parent = Segment(self, int(source["segment"]))
            if parent.number in seen or not parent.path.is_dir():
                break
            seen.add(parent.number)
            chain.append((parent, _number(source.get("model_time_s"))))
            current = parent
        chain.reverse()
        return chain

    def recorded_line(self) -> dict[int, float | None]:
        """The run's line as run.json records it: the newest segment it records and each one that segment resumed
        from, by number, each with the model time its descendant resumed from (``None``: all of its frames).

        Unlike :meth:`lineage`, a segment whose folder was deleted stays on the line: only the run's own records
        say that it left a branch (a resume from an earlier restart, or a start over), never files that are
        gone.  Empty when run.json records no segment.
        """

        try:
            rows = self.load().get("segments") or []
        except RunDirectoryError:
            return {}
        numbers = [_integer(row.get("number")) for row in rows if isinstance(row, dict)]
        numbers = [number for number in numbers if number is not None]
        if not numbers:
            return {}
        current = max(numbers)
        line: dict[int, float | None] = {current: None}
        while True:
            record = self.segment_record(current) or Segment(self, current).state()
            source = record.get("resumed_from") if isinstance(record, dict) else None
            parent = _integer(source.get("segment")) if isinstance(source, dict) else None
            if parent is None or parent in line:
                break
            line[parent] = _number(source.get("model_time_s"))
            current = parent
        return line

    def frame_line(self) -> list[tuple[Segment, float | None]]:
        """The segments whose frames (and parcel tracks) form the run, oldest first, each with the model time they
        count up to (``None``: all of them): the line run.json records (:meth:`recorded_line`), each of its
        segments whose folder is there.

        A segment whose folder was deleted stays on that line, so the frames of a branch the run left are never
        taken back for the run's own because the folder of the segment that replaced them is gone.  While the
        newest segment's folder is not recorded yet (a start, for a moment), or when run.json records none, this
        is :meth:`lineage`.
        """

        line = self.recorded_line()
        latest = self.latest_segment()
        if not line or (latest is not None and latest.number > max(line)):
            return self.lineage()
        return [(Segment(self, number), until) for number, until in sorted(line.items())
                if Segment(self, number).path.is_dir()]

    def frames(self) -> list[FrameInfo]:
        """The run's frames in model-time order, along its line (:meth:`frame_line`)."""

        by_index: dict[int, FrameInfo] = {}
        for segment, until in self.frame_line():
            for frame in segment.frames():
                if until is not None and frame.model_time_s is not None and frame.model_time_s > until:
                    continue
                by_index[frame.index] = frame
        return [by_index[index] for index in sorted(by_index)]

    def restarts(self) -> list[RestartInfo]:
        """Every durable restart in every segment, oldest segment first."""

        found: list[RestartInfo] = []
        for segment in self.segments():
            found.extend(segment.restarts())
        return found

    def lineage_restarts(self) -> list[RestartInfo]:
        """The restarts of the run as it stands: along the latest segment's lineage.

        An ancestor's restarts count up to the model time its descendant
        resumed from, as its frames do.  A branch the run has left (after a
        resume from an earlier restart, or a start over) is not continued by
        default: it is a different trajectory from the one the frames show.
        """

        found: list[RestartInfo] = []
        for segment, until in self.lineage():
            for restart in segment.restarts():
                if until is not None and restart.model_time_s is not None and restart.model_time_s > until:
                    continue
                found.append(restart)
        return found

    def find_restart(self, number: int | None = None, segment: int | None = None) -> RestartInfo | None:
        """Restart ``number`` (default: the newest) in ``segment`` (default: the run's lineage).

        Without ``segment`` the restart comes from the latest segment's
        lineage (:meth:`lineage_restarts`), so "continue" goes on with the
        trajectory the run's frames show.  Numbers continue across a resume
        (CM1's rstnum), so along one lineage they are unique; a number the
        lineage does not hold is looked up in the most recent segment that
        has it, and ``segment`` picks another branch explicitly.
        """

        if segment is not None:
            candidates = [row for row in self.restarts() if row.segment == int(segment)]
            if number is not None:
                candidates = [row for row in candidates if row.number == int(number)]
            return max(candidates, key=lambda row: row.number) if candidates else None
        lineage = self.lineage_restarts()
        if number is None:
            return lineage[-1] if lineage else None
        along = [row for row in lineage if row.number == int(number)]
        if along:
            return along[-1]
        candidates = [row for row in self.restarts() if row.number == int(number)]
        return max(candidates, key=lambda row: row.segment) if candidates else None

    def disk_bytes(self) -> int:
        total = 0
        for root, _dirs, files in os.walk(self.path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    continue
        return total


def is_run_directory(path: Path | str) -> bool:
    return (Path(path) / RUN_JSON).is_file()


# ------------------------------------------------------------ the segment deck


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"{value!r} is not a TOML scalar this writer emits")


_TABLE = re.compile(r"^\s*\[([A-Za-z0-9_.\-]+)\]\s*(#.*)?$")
_KEY = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s*=")


def derive_deck(
    text: str,
    overrides: Mapping[str, Mapping[str, Any]],
    *,
    header: Sequence[str] = (),
) -> str:
    """``text`` with ``overrides`` (``{table: {key: value}}``) set, line by line.

    Only the named keys change; every other line is kept byte for byte.  The
    result is parsed back and checked: the overridden keys hold their values
    and nothing else moved.  ``header`` lines (comments) are put first.
    """

    import tomllib

    original = tomllib.loads(text)
    lines = text.splitlines()
    pending = {table: dict(keys) for table, keys in overrides.items()}
    output: list[str] = []
    table: str | None = None

    def flush(name: str | None) -> None:
        if name is None or name not in pending:
            return
        # Keys not present in the table go at its end, before trailing blank lines.
        rows = [f"{key} = {_toml_scalar(value)}" for key, value in pending.pop(name).items()]
        trailing = []
        while output and not output[-1].strip():
            trailing.append(output.pop())
        output.extend(rows)
        output.extend(reversed(trailing))

    for line in lines:
        header_match = _TABLE.match(line)
        if header_match is not None:
            flush(table)
            table = header_match.group(1)
            output.append(line)
            continue
        key_match = _KEY.match(line)
        if table is not None and key_match is not None and key_match.group(1) in pending.get(table, {}):
            key = key_match.group(1)
            output.append(f"{key} = {_toml_scalar(pending[table].pop(key))}")
            continue
        output.append(line)
    flush(table)
    for name, keys in pending.items():
        if keys:
            output.append("")
            output.append(f"[{name}]")
            output.extend(f"{key} = {_toml_scalar(value)}" for key, value in keys.items())
    result = "\n".join([*header, *output]) + "\n"
    derived = tomllib.loads(result)
    expected = json.loads(json.dumps(original))
    for name, keys in overrides.items():
        expected.setdefault(name, {}).update(keys)
    if derived != expected:
        raise ValueError("the derived deck does not parse to the document with only the overrides changed")
    return result


# ------------------------------------------------------------ creating a run


#: CM1 settings the run supervisor reads from run.json (the importer's
#: effective namelist), so starting or resuming a run never re-parses it.
CM1_SETTINGS = ("timax", "tapfrq", "rstfrq", "dtl", "isnd", "ptype", "imoist", "nx", "ny", "nz", "dx", "dy", "dz")


def import_summary(result: Any) -> dict[str, Any]:
    """Whether an imported namelist may run: ``runnable``, ``acknowledge`` or ``refused``.

    This is the minimum ``gpuwm-cm1 run`` needs.  The coverage report
    (``import-report.json``) is the full account and overwrites this block
    when it is written.
    """

    refused: list[str] = []
    acknowledge: list[dict[str, str]] = []
    choice = result.component
    if choice.status == "refused":
        refused.append(choice.reason)
    elif choice.status == "acknowledge":
        acknowledge.append({"id": f"ptype-{choice.ptype}", "text": choice.reason})
    refused.extend(result.cm1_stops)
    translation = result.translation
    if translation is not None:
        for row in translation.unrepresentable:
            refused.append(f"{row.get('cm1_key')} = {row.get('effective')!r}: {row.get('note')}")
    for reading in result.r22:
        if getattr(reading, "status", "same") == "same":
            continue
        if getattr(reading, "klass", "") == "physics":
            refused.append(reading.message)
        else:
            acknowledge.append({"id": f"r22-{reading.key}", "text": reading.message})
    for unknown in result.unknown:
        acknowledge.append({
            "id": f"unknown-{unknown.key}",
            "text": f"{unknown.key} = {unknown.text} is not a CM1 r21.1 setting and is not read",
        })
    if translation is None and not refused:
        refused.append("the namelist could not be translated")
    status = "refused" if refused else ("acknowledge" if acknowledge else "runnable")
    return {
        "result": status,
        "component": choice.component,
        "scheme": choice.scheme,
        "refused": refused,
        "acknowledgments_needed": acknowledge,
        "by": "cm1core.cm1run.rundir.import_summary (the coverage report replaces it)",
    }


def create_run_directory(
    namelist: Path | str,
    out: Path | str,
    *,
    sounding: Path | str | None = None,
    name: str | None = None,
    gm1_only: Mapping[str, Any] | None = None,
    tool: Any = None,
) -> RunDirectory:
    """A new run directory from a CM1 namelist: the byte copy, the deck, ``run.json``.

    ``out`` must not exist, or be an empty directory.  The namelist is read
    as CM1 r21.1 would read it (:func:`cm1core.cm1import.import_namelist`);
    a namelist that cannot run still gets a directory, whose ``run.json``
    says why, so the report can be shown and the namelist edited.
    """

    from ..cm1import import import_namelist

    source = Path(namelist).expanduser()
    target = Path(out).expanduser().absolute()
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise RunDirectoryError(f"{target} already exists and is not empty; choose a new folder")
    run = RunDirectory(target)
    result = import_namelist(source, tool=tool, gm1_only=gm1_only)
    effective = result.effective
    isnd = int(effective.get("isnd", 0))
    needs_sounding = isnd in (7, 17)
    if needs_sounding and sounding is None:
        raise RunDirectoryError(
            f"isnd = {isnd} reads an input_sounding file; pass --sounding FILE "
            "(a WRF-format sounding, as SounderPy's to_file('cm1') writes)"
        )
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, run.namelist_original)
    shutil.copyfile(source, run.namelist)
    sounding_record = None
    if sounding is not None:
        shutil.copyfile(Path(sounding).expanduser(), run.sounding)
        sounding_record = {
            "file": SOUNDING,
            "sha256": sha256_file(run.sounding),
            "source_path": os.fspath(Path(sounding).expanduser().absolute()),
            "read": needs_sounding,
        }
    run.control.mkdir(exist_ok=True)
    toml = result.toml()
    deck_record = None
    if toml is not None:
        write_text_atomic(run.deck, toml)
        deck_record = {"file": DECK, "sha256": sha256_file(run.deck)}
    record = {
        "schema": RUN_SCHEMA,
        "name": name or target.name,
        "created_utc": utc_now(),
        "gm1": installation(),
        "namelist": {
            "original": NAMELIST_ORIGINAL,
            "original_sha256": sha256_file(run.namelist_original),
            "sha256": sha256_file(run.namelist),
            "source_path": os.fspath(source.absolute()),
            "version": result.version,
        },
        "sounding": sounding_record,
        "cm1": {key: effective.get(key) for key in CM1_SETTINGS},
        "import": import_summary(result),
        "acknowledgments": [],
        "deck": deck_record,
        "host": None,
        "gpu": None,
        "scheduler": None,
        "segments": [],
    }
    write_json_atomic(run.run_json, record)
    return run


#: The coverage report's verdicts (``import-report.json``) as run.json's
#: ``import.result`` words, which ``run`` and ``status`` read.
REPORT_RESULTS = {"runs": "runnable", "runs-after-acknowledgment": "acknowledge", "refused": "refused"}
#: What each acknowledgment ``gpuwm-cm1 import`` can be given lets the run do.
ACKNOWLEDGMENT_TEXT = {
    "output-keys": "run without the output settings GM1 does not read (import with --acknowledge-output-keys)",
    "thompson": "run Thompson microphysics (ptype = 3), checked only in a 120-s column "
                "(import with --acknowledge-thompson)",
}


def import_block_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """``run.json``'s ``import`` block from the coverage report (``import-report.json``).

    The report is the full account; this is the part ``gpuwm-cm1 run`` and
    ``status`` act on: whether the namelist runs, why not, and which
    acknowledgments it needs.
    """

    refused: list[str] = []
    for section in report.get("sections") or []:
        if not isinstance(section, Mapping) or section.get("id") != "refused":
            continue
        for row in section.get("rows") or []:
            if not isinstance(row, Mapping):
                continue
            key, value = row.get("key"), row.get("value")
            text = str(row.get("text") or row.get("plain") or "refused")
            refused.append(f"{key} = {value}: {text}" if key and value is not None else text)
    if report.get("verdict") == "refused" and not refused:
        refused.append(str(report.get("error") or report.get("verdict_text") or "the namelist cannot run"))
    acknowledgments = report.get("acknowledgments") or {}
    component = report.get("component") or {}
    return {
        "result": REPORT_RESULTS.get(str(report.get("verdict")), "refused"),
        "verdict_text": report.get("verdict_text"),
        "exit": report.get("exit"),
        "component": component.get("component"),
        "scheme": component.get("scheme"),
        "refused": refused,
        "acknowledgments_needed": [
            {"id": str(name), "text": ACKNOWLEDGMENT_TEXT.get(str(name), str(name))}
            for name in acknowledgments.get("needed") or []
        ],
        "report": IMPORT_REPORT_JSON,
        "by": "cm1core.cm1import.door (import-report.json is the full account)",
    }


def record_import(
    path: Path | str,
    report: Mapping[str, Any],
    *,
    source: Path | str,
    effective: Mapping[str, Any] | None,
    sounding: Path | str | None = None,
    sounding_source: Path | str | None = None,
    name: str | None = None,
) -> RunDirectory:
    """Write or update ``run.json`` for a directory ``gpuwm-cm1 import`` has just filled.

    The importer writes the namelist copies, the coverage report and,
    when the namelist runs, ``deck.toml``; this adds the record that makes
    the folder a run directory for ``run``, ``status``, ``estimate`` and the
    GUI.  A re-import keeps the run's name and creation time and replaces
    what the import decides: the namelist's digests, the CM1 settings,
    the ``import`` block, the acknowledgments given and the deck's digest.
    """

    run = RunDirectory(path)
    source = Path(source).expanduser()
    effective = effective or {}
    isnd = effective.get("isnd")
    sounding_path = None if sounding is None else Path(sounding)
    sounding_record = None
    if sounding_path is not None and sounding_path.is_file():
        sounding_record = {
            "file": SOUNDING if sounding_path.parent.resolve() == run.path.resolve() else os.fspath(sounding_path),
            "sha256": sha256_file(sounding_path),
            "source_path": os.fspath(Path(sounding_source or sounding_path).expanduser().absolute()),
            "read": isnd in (7, 17),
        }
    acknowledgments = report.get("acknowledgments") or {}
    changes = {
        "gm1": installation(),
        "imported_utc": utc_now(),
        "namelist": {
            "original": NAMELIST_ORIGINAL,
            "original_sha256": sha256_file(run.namelist_original),
            "sha256": sha256_file(run.namelist),
            "source_path": os.fspath(source.absolute()),
            "version": report.get("version"),
        },
        "sounding": sounding_record,
        "cm1": {key: effective.get(key) for key in CM1_SETTINGS},
        "import": import_block_from_report(report),
        "acknowledgments": [str(item) for item in acknowledgments.get("given") or []],
        "deck": {"file": DECK, "sha256": sha256_file(run.deck)} if run.deck.is_file() else None,
    }
    if run.is_run():
        def mutate(record: dict[str, Any]) -> None:
            record.update(changes)
            mark_edited(run.path, record)

        run.update(mutate)
        return run
    record = {
        "schema": RUN_SCHEMA,
        "name": name or run.path.name,
        "created_utc": utc_now(),
        **changes,
        "host": None,
        "gpu": None,
        "scheduler": None,
        "segments": [],
    }
    write_json_atomic(run.run_json, record)
    return run


def mark_edited(path: Path, record: dict[str, Any]) -> None:
    """A guided run whose namelist was edited after its fit says so: its case ends in "(edited)".

    Edited means ``namelist.input`` is no longer the namelist ``fit.json`` fitted (an edit back to it drops the
    word), so Watching, Results and the Runs list never show an edited run as the preset it came from.
    """

    guided = record.get("guided")
    if not isinstance(guided, dict):
        return
    try:
        fitted = json.loads((Path(path) / "fit.json").read_text(encoding="utf-8")).get("namelist_sha256")
    except (OSError, ValueError, AttributeError):
        return
    now = (record.get("namelist") or {}).get("sha256")
    if not fitted or not now:
        return
    edited = now != fitted
    case = str(record.get("case") or "")
    base = case[: -len(" (edited)")] if case.endswith(" (edited)") else case
    if base:
        record["case"] = f"{base} (edited)" if edited else base
    guided["edited"] = edited


__all__ = [
    "CM1_SETTINGS",
    "mark_edited",
    "FrameInfo",
    "RUN_SCHEMA",
    "RestartInfo",
    "RunDirectory",
    "RunDirectoryError",
    "SEGMENT_SCHEMA",
    "Segment",
    "create_run_directory",
    "derive_deck",
    "import_block_from_report",
    "import_summary",
    "is_run_directory",
    "local_process_alive",
    "off_the_line",
    "process_identity",
    "process_start_ticks",
    "record_import",
    "segment_of",
    "utc_now",
    "write_json_atomic",
]
