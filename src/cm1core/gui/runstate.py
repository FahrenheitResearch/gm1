"""A run's status, read from its run directory.

Nothing here talks to a process: the run directory is the contract.

- ``run.json`` (``gm1.cm1-run/v1``) names the run and lists its segments,
  each with the pid, its start time (``/proc/PID/stat`` field 22, in clock
  ticks), the host and a scheduler job id when there is one.
- ``segments/NNN/events.jsonl`` is the segment's ``gm1.run-events/v1`` log.
  Every record carries the writer's pid, and ``admitted`` its host, so the
  log alone is enough when ``run.json`` has no segment row.
- ``import-report.json`` and ``estimate.json`` say how far setup got.

The states are the Runs screen's (section 6): imported, ready, waiting,
running, paused, finished, stopped, failed, stale, and view-only for a plain
``cartesian`` output directory.  A process that is gone without an ``end``
record reads "Stopped without a final record (last frame N)".  On another
host, a log whose last record is older than 60 s reads Stale.

A run supervisor (``gpuwm-cm1 run``, from a terminal, a job script or the
GUI) also keeps ``segments/NNN/segment.json``: waiting for the GPU in line,
its heartbeat while it waits, the child's identity, the scheduler job, and
how the segment ended when the runtime could not say (force stopped).  A
stale run on another host whose scheduler job is no longer listed reads
"Stopped without a final record", as ``gpuwm-cm1 status`` says
(:func:`cm1core.cm1run.status.scheduler_job_state`).

Each log is read once: a :class:`LogDigest` per log keeps what the screens
need (the admitted record, start-up, the latest progress, every frame,
restart and warning, the end) and reads only the bytes written since the
last look, so listing many runs, and a Run screen polling one, stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
from typing import Any, Callable, Iterable

from .files import RUN_FILE, cm1_output, is_view_only, read_json

RUN_SCHEMA = "gm1.cm1-run/v1"
EVENTS_NAME = "events.jsonl"
SEGMENTS_DIR = "segments"
SEGMENT_RE = re.compile(r"^\d{3,}$")
#: A log with no end record whose last record is older than this, on another
#: host, reads Stale (run_events.STALE_AFTER_S).
STALE_AFTER_S = 60.0
#: A new segment folder with no record and no run.json row yet reads
#: Starting for this long, then "Stopped before it wrote anything".
STARTING_GRACE_S = 30.0
TAIL_BYTES = 64 * 1024
#: One poll of a growing log reads at most this much.
READ_CHUNK = 4 * 1024 * 1024
SEGMENT_STATE = "segment.json"
#: How many logs' digests are kept (least recently read dropped first).  The Runs list reads every segment's
#: log on each poll, so this is well above the segments of a busy runs folder: past it, each poll would read the
#: dropped logs again from their start.
DIGESTS_KEPT = 2048
#: The warnings a run's screens say in one plain line.
WARNING_WORDS = {
    "GM1-SWATHS-RESTART-HERE": "The swaths start again at {clock}: the save point this run continued from did not "
                               "keep them.",
}

END_LABELS = {
    "completed": ("finished", "Finished"),
    "stopped": ("stopped", "Stopped"),
    "paused": ("paused", "Paused"),
    "courant-stop": ("failed", "Failed: the wind got too fast for the time step (Courant stop)"),
    "refused": ("failed", "Failed: refused before it started"),
    "error": ("failed", "Failed"),
}


def this_host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def segment_dirs(rundir: Path) -> list[Path]:
    base = rundir / SEGMENTS_DIR
    try:
        entries = [entry for entry in os.scandir(base) if entry.is_dir() and SEGMENT_RE.match(entry.name)]
    except OSError:
        return []
    return [Path(entry.path) for entry in sorted(entries, key=lambda entry: int(entry.name))]


def events_path(rundir: Path, segment: Path | None = None) -> Path | None:
    """The newest segment's log, or a view-only directory's own."""

    if segment is not None:
        return segment / EVENTS_NAME
    segments = segment_dirs(rundir)
    if segments:
        return segments[-1] / EVENTS_NAME
    if (rundir / EVENTS_NAME).is_file():
        return rundir / EVENTS_NAME
    return None


def _records(data: bytes) -> list[dict[str, Any]]:
    records = []
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def read_head_tail(path: Path, tail_bytes: int = TAIL_BYTES) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float | None]:
    """(first record, the complete records of the last ``tail_bytes``, mtime)."""

    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            mtime = os.fstat(stream.fileno()).st_mtime
            first_line = stream.readline()
            start = max(0, size - tail_bytes)
            stream.seek(start)
            tail = stream.read(tail_bytes)
    except OSError:
        return None, [], None
    if not first_line.endswith(b"\n"):
        first_line = b""
    first = _records(first_line)
    if start > 0:
        # The first chunk line is probably torn: drop it.
        tail = tail.split(b"\n", 1)[1] if b"\n" in tail else b""
    if not tail.endswith(b"\n"):
        tail = tail.rsplit(b"\n", 1)[0] if b"\n" in tail else b""
    return (first[0] if first else None), _records(tail), mtime


def _utc_seconds(text: Any) -> float | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def proc_start_ticks(pid: int) -> int | None:
    """Field 22 of ``/proc/PID/stat``: the process start time in clock ticks."""

    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii", errors="replace")
    except (OSError, ValueError):
        return None
    fields = text.rsplit(")", 1)[-1].split()
    try:
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def process_alive(pid: Any, start_ticks: Any = None) -> bool:
    """A live, non-zombie process with this pid (and start time, when given)."""

    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return False
    fields = text.rsplit(")", 1)[-1].split()
    if not fields or fields[0] in ("Z", "X", "x"):
        return False
    if start_ticks is not None:
        try:
            return int(fields[19]) == int(start_ticks)
        except (IndexError, TypeError, ValueError):
            return False
    return True


@dataclass
class RunStatus:
    state: str
    label: str
    percent: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "label": self.label, "percent": self.percent, **self.detail}


def _segment_row(run: dict[str, Any], segment: Path | None) -> dict[str, Any]:
    rows = run.get("segments") if isinstance(run.get("segments"), list) else []
    if segment is None:
        return {}
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        named = str(row.get("directory") or row.get("dir") or "")
        numbers = {str(row.get("index")), str(row.get("number"))}
        if named.rstrip("/").endswith(segment.name) or numbers & {segment.name, segment.name.lstrip("0")}:
            return row
    return {}


def _as_dict(value: Any) -> dict[str, Any]:
    """A JSON object as a dict; anything else (a hand-edited or torn file) as an empty one."""

    return value if isinstance(value, dict) else {}


def segment_rows(rundir: Path, run: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per segment: its number, kind, host, times, wall seconds and how it ended (run.json, segment.json)."""

    if run is None:
        run = read_json(rundir / RUN_FILE, default={})
    run = run if isinstance(run, dict) else {}
    out = []
    now = time.time()
    for segment in segment_dirs(rundir):
        row = _segment_row(run, segment)
        state = _as_dict(read_json(segment / SEGMENT_STATE, default={}))
        started = _utc_seconds(row.get("started_utc") or state.get("created_utc"))
        ended = _utc_seconds(row.get("ended_utc") or state.get("ended_utc"))
        digest = log_digest(segment / EVENTS_NAME)
        admitted = (digest.admitted if digest is not None else None) or {}
        if started is not None and ended is not None:
            wall: float | None = max(0.0, ended - started)
        elif digest is not None and digest.end is not None and isinstance(digest.end.get("elapsed_s"), (int, float)):
            wall = float(digest.end["elapsed_s"])
        elif started is not None and digest is not None and digest.end is None:
            # Still going: the time since it started; gone quiet (crashed, or a lost node): up to its last word.
            last_word = _utc_seconds((digest.last or {}).get("time_utc"))
            quiet = last_word is not None and now - last_word > STALE_AFTER_S
            wall = max(0.0, (last_word if quiet else now) - started)
        elif digest is not None and digest.last is not None and isinstance(digest.last.get("elapsed_s"), (int, float)):
            wall = float(digest.last["elapsed_s"])
        else:
            wall = None
        resumed = row.get("resumed_from") or admitted.get("resumed_from")
        # The supervisor's row names the restart ({segment, restart, model_time_s, directory}); the runtime's own admitted
        # record, the only word on a run without that row, names the restart's folder.
        resumed = resumed if isinstance(resumed, dict) or resumed is None else {"directory": str(resumed)}
        gpu = row.get("gpu") or state.get("gpu")
        out.append({
            "segment": segment.name,
            # The supervisor records "fresh" or "resume"; the screens say start and continue.
            "kind": {"fresh": "start", "resume": "continue"}.get(str(row.get("kind")), None)
            or ("continue" if resumed else "start"),
            "resumed_from": resumed,
            "until_s": row.get("until_s"),
            "host": row.get("host") or state.get("host") or admitted.get("host"),
            "gpu": (gpu.get("device") or {}).get("name") if isinstance(gpu, dict) and isinstance(gpu.get("device"), dict)
            else (gpu.get("name") if isinstance(gpu, dict) else None),
            "scheduler": _as_dict(row.get("scheduler") or state.get("scheduler")) or None,
            "started_utc": row.get("started_utc") or state.get("created_utc"),
            "ended_utc": row.get("ended_utc") or state.get("ended_utc"),
            "wall_s": None if wall is None else round(wall, 1),
            "reason": (digest.end or {}).get("reason") if digest and digest.end else row.get("reason"),
            "frames": 0 if digest is None else len(digest.frames),
            "restarts": 0 if digest is None else len(digest.restarts),
            "model_time_s": None if digest is None else digest.model_time(),
        })
    return out


def import_exit_code(rundir: Path) -> int | None:
    report = read_json(rundir / "import-report.json", default=None)
    if isinstance(report, dict):
        # "exit" is the coverage report's (cm1import.coverage, gpuwm-cm1 import).
        for key in ("exit", "exit_code", "rc"):
            if isinstance(report.get(key), int):
                return int(report[key])
    return None


class LogDigest:
    """What one segment's log says so far, kept up to date by reading only its new bytes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self._tail = EventTail(self.path)
        self._clear()

    def _clear(self) -> None:
        self.records = 0
        self.first: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None
        self.admitted: dict[str, Any] | None = None
        self.startup: dict[str, Any] | None = None
        self.progress: dict[str, Any] | None = None
        self.frames: list[dict[str, Any]] = []
        self.restarts: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.stats: list[dict[str, Any]] = []
        self.refused: dict[str, Any] | None = None
        self.end: dict[str, Any] | None = None
        self.mtime: float | None = None
        self._generation = self._tail.generation

    def update(self) -> "LogDigest":
        with self.lock:
            while True:
                batch = self._tail.read_new()
                if self._tail.generation != self._generation:
                    self._clear()
                for record in batch:
                    self._add(record)
                if not batch or not self._tail.pending():
                    break
            try:
                self.mtime = self.path.stat().st_mtime
            except OSError:
                self.mtime = None
        return self

    def _add(self, record: dict[str, Any]) -> None:
        self.records += 1
        if self.first is None:
            self.first = record
        self.last = record
        kind = record.get("kind")
        if kind == "admitted":
            self.admitted = record
        elif kind == "startup":
            self.startup = record
        elif kind == "progress":
            self.progress = record
        elif kind == "frame":
            self.frames.append({key: record.get(key) for key in (
                "index", "model_time_s", "step", "bytes", "time_utc", "initial_condition", "restart_written")})
        elif kind == "restart":
            self.restarts.append({key: record.get(key) for key in (
                "number", "model_time_s", "step", "bytes", "time_utc")})
        elif kind == "warning":
            self.warnings.append({key: record.get(key) for key in ("code", "message", "time_utc")})
        elif kind == "stats":
            self.stats.append({key: record.get(key) for key in ("model_time_s", "step", "values")})
            del self.stats[:-400]
        elif kind == "refused":
            from ..cm1run.events import plain_record

            self.refused = dict(plain_record(record))
        elif kind == "end":
            from ..cm1run.events import plain_record

            self.end = dict(plain_record(record))

    def model_time(self) -> float | None:
        for record in (self.end, self.progress):
            if record is not None and isinstance(record.get("model_time_s"), (int, float)):
                return float(record["model_time_s"])
        if self.frames and isinstance(self.frames[-1].get("model_time_s"), (int, float)):
            return float(self.frames[-1]["model_time_s"])
        return None


_DIGESTS: "OrderedDict[str, LogDigest]" = OrderedDict()
_DIGESTS_LOCK = threading.Lock()


def log_digest(path: Path | None) -> LogDigest | None:
    """The up-to-date digest of the log at ``path``, or None when there is no log there."""

    if path is None or not Path(path).is_file():
        return None
    key = os.fspath(Path(path).absolute())
    with _DIGESTS_LOCK:
        digest = _DIGESTS.get(key)
        if digest is None:
            digest = LogDigest(Path(key))
            _DIGESTS[key] = digest
        _DIGESTS.move_to_end(key)
        while len(_DIGESTS) > DIGESTS_KEPT:
            _DIGESTS.popitem(last=False)
    return digest.update()


def warning_words(warning: dict[str, Any]) -> str:
    """One plain line for a run's warning record."""

    code = str(warning.get("code") or "")
    message = str(warning.get("message") or "")
    template = WARNING_WORDS.get(code)
    if template is None:
        return message or code
    match = re.search(r"\bt=(\d+(?:\.\d+)?) ?s\b", message)
    if match:
        minutes = int(round(float(match.group(1)) / 60.0))
        clock = f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d} min"
    else:
        clock = "the save point"
    return template.format(clock=clock)


def _scheduler_answer(scheduler: Any) -> str:
    if not isinstance(scheduler, dict):
        return "unknown"
    try:
        from ..cm1run.status import scheduler_job_state

        return scheduler_job_state(scheduler.get("kind"), scheduler.get("job_id"))
    except Exception:  # noqa: BLE001 - a status read never fails on the scheduler
        return "unknown"


def read_status(rundir: Path, *, now: float | None = None, host: str | None = None,
                launched_alive: bool = False,
                scheduler: Callable[[Any], str] | None = None) -> RunStatus:
    """The status of the run in ``rundir``.

    ``launched_alive``: the caller (the GUI server) launched a process for this
    run that is still alive, which covers the second or two before the run
    writes its first record.  ``scheduler``: asks whether a batch job is still
    listed (tests pass a stand-in); by default ``squeue``/``qstat`` when the
    run is on another host, has gone quiet and names its job.
    """

    now = time.time() if now is None else now
    host = this_host() if host is None else host
    ask = _scheduler_answer if scheduler is None else scheduler
    view_only = is_view_only(rundir)
    run = read_json(rundir / RUN_FILE, default={}) if not view_only else {}
    run = run if isinstance(run, dict) else {}
    segments = segment_dirs(rundir)
    segment = segments[-1] if segments else None
    log = events_path(rundir, segment)
    detail: dict[str, Any] = {
        "segment": None if segment is None else segment.name,
        "segments": len(segments),
        "view_only": view_only,
    }
    if view_only:
        detail["actions"] = []
        if cm1_output(rundir) is not None:
            # CM1's own netCDF output, opened for pictures: nothing runs here, and nothing is to come.  Its grid
            # and its last saved time are what its pictures say (quicklook/index.json).
            detail["cm1_output"] = True
            detail.update(_cm1_output_detail(rundir))
            return RunStatus("finished", "CM1 output", None, detail)

    digest = log_digest(log)
    admitted = None if digest is None else digest.admitted
    progress = None if digest is None else digest.progress
    frame = None if digest is None or not digest.frames else digest.frames[-1]
    restart = None if digest is None or not digest.restarts else digest.restarts[-1]
    end = None if digest is None else digest.end
    last = None if digest is None else digest.last
    has_records = bool(digest is not None and digest.records)
    mtime = None if digest is None else digest.mtime

    row = _segment_row(run, segment)
    state_file = (read_json(segment / SEGMENT_STATE, default={}) or {}) if segment is not None else {}
    state_file = state_file if isinstance(state_file, dict) else {}
    run_host = row.get("host") or state_file.get("host") or (admitted or {}).get("host") or run.get("host")
    local = run_host is None or run_host == host
    child = state_file.get("child") if isinstance(state_file.get("child"), dict) else None
    supervisor = state_file.get("supervisor") if isinstance(state_file.get("supervisor"), dict) else None
    pid = (child or {}).get("pid") or row.get("child_pid") or row.get("pid") or (last or {}).get("pid")
    # The newest record's own time (the file's mtime when it has none), or the supervisor's heartbeat
    # while it waits in line for the GPU and no record exists yet.
    beats = [moment for moment in (_utc_seconds((last or {}).get("time_utc")) or mtime,
                                   _utc_seconds(state_file.get("heartbeat_utc"))) if moment is not None]
    last_time = max(beats) if beats else None
    quiet_s = None if last_time is None else max(0.0, now - last_time)
    scheduler_job = row.get("scheduler") or state_file.get("scheduler") or run.get("scheduler")

    duration = (progress or {}).get("duration_s") or (admitted or {}).get("duration_s")
    model_time = None if digest is None else digest.model_time()
    percent = None
    if isinstance(model_time, (int, float)) and isinstance(duration, (int, float)) and duration > 0:
        percent = max(0.0, min(100.0, 100.0 * float(model_time) / float(duration)))
    elif progress and progress.get("step") is not None and progress.get("total_steps"):
        start_step = (admitted or {}).get("start_step") or 0
        total = float(progress["total_steps"])
        percent = max(0.0, min(100.0, 100.0 * (float(progress["step"]) - start_step) / total)) if total else None

    detail.update({
        "host": run_host,
        "local": local,
        "pid": pid,
        "model_time_s": model_time,
        "duration_s": duration,
        "eta_s": (progress or {}).get("eta_s") if end is None else None,
        "s_per_step_mean": (progress or {}).get("s_per_step_mean"),
        "model_s_per_wall_s": (progress or {}).get("model_s_per_wall_s"),
        "phase": (progress or {}).get("phase"),
        "last_frame": None if frame is None else frame.get("index"),
        "last_restart": None if restart is None else restart.get("number"),
        "frames": 0 if digest is None else len(digest.frames),
        "last_record_age_s": None if quiet_s is None else round(quiet_s, 1),
        "grid": (admitted or {}).get("grid"),
        "microphysics": (admitted or {}).get("microphysics"),
        "end": end,
        "scheduler": scheduler_job if isinstance(scheduler_job, dict) else None,
        "import_exit_code": import_exit_code(rundir),
        "has_estimate": (rundir / "estimate.json").is_file(),
    })

    def finish(state: str, label: str) -> RunStatus:
        if (state in ("paused", "stopped", "finished") and detail.get("last_restart") is None
                and not view_only):
            # A segment stopped before its first restart is still continued from
            # an earlier segment's (cm1run: the restarts on the run's lineage).
            detail["restartable"] = _lineage_restartable(rundir)
        if state in ("paused", "stopped", "finished") and not view_only and adaptive_timestep(rundir):
            # An ended run on the adaptive time step: no Continue, and the pages say why.
            detail["no_resume_why"] = ADAPTIVE_NO_CONTINUE
        detail["actions"] = [] if view_only else _actions(state, detail)
        return RunStatus(state, label, None if percent is None else round(percent, 1), detail)

    def lost() -> RunStatus:
        where = "" if frame is None else f" (last frame {frame.get('index')})"
        detail["final_record"] = False
        return finish("stopped", f"Stopped without a final record{where}")

    if end is not None:
        reason = str(state_file.get("reason_override") or end.get("reason"))
        state, label = END_LABELS.get(reason, ("failed", "Failed"))
        plan = _as_dict(state_file.get("plan"))
        until = plan.get("until_s", row.get("until_s"))
        until = until if isinstance(until, (int, float)) else duration
        if (reason == "paused" and isinstance(model_time, (int, float)) and isinstance(until, (int, float))
                and float(model_time) >= float(until) - 1e-6):
            # The next restart was the segment's last one: it paused at its end (gpuwm-cm1 status says the same).
            state, label = "finished", "Finished"
            detail["paused_at_end"] = True
        if state == "failed" and end.get("message"):
            detail["message"] = end.get("message")
        return finish(state, label)

    if state_file.get("state") == "ended":
        # The supervisor saw its process end with no end record, or it was force stopped.
        if state_file.get("reason") in ("force-stopped", "abandoned"):
            detail["final_record"] = False
            return finish("stopped", "Force stopped" if state_file.get("reason") == "force-stopped" else "Stopped")
        return lost()

    alive = False
    known = has_records or bool(row) or bool(state_file)
    if known:
        if local:
            identities = [(ident.get("pid"), ident.get("start_ticks")) for ident in (child, supervisor) if ident]
            if not identities and pid:
                identities = [(pid, row.get("pid_start_ticks"))]
            alive = any(process_alive(p, ticks) for p, ticks in identities if p)
        else:
            alive = quiet_s is not None and quiet_s <= STALE_AFTER_S
    alive = alive or launched_alive

    if not known and not launched_alive:
        if segments:
            try:
                young = now - segment.stat().st_mtime < STARTING_GRACE_S
            except OSError:
                young = False
            if young:
                return finish("running", "Starting")
            return finish("stopped", "Stopped before it wrote anything")
        code = detail["import_exit_code"]
        if code == 2:
            return finish("imported", "Imported (refused: see Check)")
        if code == 3:
            return finish("imported", "Imported (needs your OK: see Check)")
        if detail["has_estimate"]:
            return finish("ready", "Ready")
        return finish("imported", "Imported")

    if not alive:
        if not local:
            answer = ask(scheduler_job)
            detail["scheduler_state"] = answer
            if answer == "gone":
                return lost()
            minutes = (quiet_s or 0.0) / 60.0
            return finish("stale", f"Stale (no word for {minutes:.0f} min)")
        return lost()

    phase = (progress or {}).get("phase")
    queue = state_file.get("queue") if isinstance(state_file.get("queue"), dict) else {}
    if phase == "waiting-for-gpu" or (not has_records and state_file.get("state") == "waiting-for-gpu"):
        detail["queue"] = {"ahead": queue.get("ahead"),
                           "holder": (queue.get("holder") or {}).get("rundir") if isinstance(queue.get("holder"),
                                                                                             dict) else None}
        return finish("waiting", "Waiting for GPU")
    if progress is None or progress.get("step") is None or phase == "starting":
        return finish("running", "Starting")
    if percent is None:
        return finish("running", "Running")
    return finish("running", f"Running {percent:.0f}%")


#: Why a run on CM1's adaptive time step (adapt_dt = 1, as CM1's hurricane namelist has it) is never offered Continue or
#: "Run it longer": the runtime refuses to resume one (gm1_runtime: the completed steps of a varying dt do not
#: determine the storm time, and the restart does not carry it), so the pages say so in place of a button that
#: would fail.
ADAPTIVE_NO_CONTINUE = ("This run's time step changes with the storm, and a saved point does not keep the storm "
                        "time of such a run, so it cannot be continued or run longer. Start it again from the "
                        "beginning.")


def adaptive_timestep(rundir: Path) -> bool:
    """Whether the run's deck takes CM1's adaptive large step (``dynamics.adaptive_timestep`` in ``deck.toml``)."""

    try:
        import tomllib

        with (rundir / "deck.toml").open("rb") as handle:
            deck = tomllib.load(handle)
    except (OSError, ValueError):
        return False
    dynamics = deck.get("dynamics")
    return isinstance(dynamics, dict) and dynamics.get("adaptive_timestep") is True


def _lineage_restartable(rundir: Path) -> bool:
    """Whether ``gpuwm-cm1 resume`` finds a restart on the run's lineage (cm1run.rundir)."""

    try:
        from ..cm1run.rundir import RunDirectory

        return bool(RunDirectory(rundir).lineage_restarts())
    except Exception:  # noqa: BLE001 - a status read never fails on this
        return False


def _actions(state: str, detail: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if state in ("imported", "ready") and detail.get("import_exit_code") in (0, None):
        actions += ["check", "geometry", "estimate", "start"]
    if state in ("imported",) and detail.get("import_exit_code") == 3:
        actions += ["check"]
    if state in ("running", "waiting"):
        actions += ["stop", "pause"]
        if detail.get("local"):
            actions += ["force-stop"]
    if state == "stale":
        actions += ["stop"]
    if state in ("paused", "stopped", "finished") and not detail.get("no_resume_why") and (
            detail.get("last_restart") is not None or detail.get("restartable")):
        actions += ["resume"]
    if detail.get("last_frame") is not None:
        actions += ["results"]
    return actions


class EventTail:
    """New complete records of one growing log, poll by poll."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self._buffer = b""
        self._inode: tuple[int, int] | None = None
        #: Counts the times the file was replaced or truncated and read again from its start.
        self.generation = 0

    def read_new(self) -> list[dict[str, Any]]:
        try:
            status = self.path.stat()
        except OSError:
            return []
        identity = (status.st_dev, status.st_ino)
        if self._inode is not None and (identity != self._inode or status.st_size < self.offset):
            self.offset, self._buffer = 0, b""
            self.generation += 1
        self._inode = identity
        if status.st_size <= self.offset:
            return []
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read(READ_CHUNK)
        except OSError:
            return []
        self.offset += len(data)
        self._buffer += data
        complete, _, self._buffer = self._buffer.rpartition(b"\n")
        return _records(complete)

    def pending(self) -> bool:
        """More bytes are on disk than have been read."""

        try:
            return self.path.stat().st_size > self.offset
        except OSError:
            return False


def _cm1_output_detail(rundir: Path) -> dict[str, Any]:
    """The grid and the last saved time of CM1 output opened for pictures: what the files hold (cm1-output.json,
    written when it was opened), else what its pictures' index says.  The last saved time is the files' own from
    the start, not that of the last picture drawn so far."""

    from .files import CM1_OUTPUT_FILE

    out: dict[str, Any] = {}
    opened = read_json(rundir / CM1_OUTPUT_FILE, default=None)
    if isinstance(opened, dict):
        grid = opened.get("grid")
        if isinstance(grid, dict) and all(isinstance(grid.get(k), (int, float)) for k in ("nx", "ny", "nz", "dx_m")):
            out["grid"] = {"nx": int(grid["nx"]), "ny": int(grid["ny"]), "nz": int(grid["nz"]),
                           "dx_m": float(grid["dx_m"])}
        saved = opened.get("saved_times_s")
        if isinstance(saved, list) and saved and all(isinstance(t, (int, float)) for t in saved):
            out["model_time_s"] = out["duration_s"] = float(max(saved))
    index = read_json(rundir / "quicklook" / "index.json", default=None)
    if not isinstance(index, dict):
        return out
    run = index.get("run") if isinstance(index.get("run"), dict) else {}
    found = re.match(r"(\d+) × (\d+) × (\d+)", str(run.get("grid") or ""))
    if found and isinstance(run.get("dx_m"), (int, float)):
        out["grid"] = {"nx": int(found.group(1)), "ny": int(found.group(2)), "nz": int(found.group(3)),
                       "dx_m": float(run["dx_m"])}
    times = [f.get("time_s") for f in index.get("frames") or [] if isinstance(f, dict)]
    times = [float(t) for t in times if isinstance(t, (int, float))]
    if times:
        out["model_time_s"] = out["duration_s"] = max(max(times), float(out.get("model_time_s") or 0.0))
        out["frames"] = len(times)
    return out


def summarize(rundir: Path, status: RunStatus, run_id: str) -> dict[str, Any]:
    """One Runs-table row (section 6, screen 1)."""

    run = _as_dict(read_json(rundir / RUN_FILE, default={}))
    grid = status.detail.get("grid") or run.get("grid") or {}
    return {
        "id": run_id,
        "name": run.get("name") or rundir.name,
        "case": run.get("case") or run.get("preset") or None,
        "grid": grid or None,
        "status": status.as_dict(),
        "created_utc": run.get("created_utc"),
        "host": status.detail.get("host") or run.get("host"),
    }


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    try:
        data = path.read_bytes()
    except OSError:
        return []
    complete, _, _ = data.rpartition(b"\n")
    return _records(complete)


__all__ = [
    "END_LABELS",
    "EVENTS_NAME",
    "EventTail",
    "LogDigest",
    "RUN_SCHEMA",
    "RunStatus",
    "SEGMENTS_DIR",
    "STALE_AFTER_S",
    "events_path",
    "import_exit_code",
    "iter_records",
    "log_digest",
    "proc_start_ticks",
    "process_alive",
    "read_head_tail",
    "read_status",
    "segment_dirs",
    "segment_rows",
    "summarize",
    "this_host",
    "warning_words",
]
