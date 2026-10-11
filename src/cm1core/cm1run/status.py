"""What a run is doing, from its directory alone.

The status of a run is read from files, so the GUI, a terminal on the login
node and a job script all see the same thing, and a GUI that was closed and
opened again loses nothing.

Crash detection depends on where the run is:

* **On this host** (the segment's ``host`` is this machine's name): the
  recorded processes are checked by pid *and* ``/proc/<pid>/stat`` start
  time, so a reused pid is not mistaken for the run.  A run whose processes
  are gone and whose log has no ``end`` record "stopped without a final
  record".
* **On another host** (a compute node, another machine sharing the
  directory): ``/proc`` is never read, because a pid there means nothing
  here.  The heartbeat is the age of the newest record (the runtime writes
  one at least every 10 s); older than :data:`STALE_AFTER_S` reads
  ``Stale``.  When the segment names a scheduler job and ``squeue`` (Slurm)
  or ``qstat`` (PBS) is on this machine, a job the scheduler no longer lists
  turns ``Stale`` into "Stopped without a final record".

Importing this module imports neither CuPy nor numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping

from .events import SegmentEvents, summarize
from .rundir import (
    RunDirectory,
    RunDirectoryError,
    host_name,
    local_process_alive,
    parse_utc,
)

#: A log (or a waiting supervisor's heartbeat) older than this, with no end
#: record, is stale.  The runtime writes a record at least every 10 s
#: (run_events.HEARTBEAT_LIMIT_S), a waiting supervisor every 5 s.
STALE_AFTER_S = 60.0

#: state -> the word the Runs table shows (section 6, screen 1).
LABELS = {
    "imported": "Imported",
    "ready": "Ready",
    "waiting": "Waiting for GPU",
    "running": "Running",
    "paused": "Paused",
    "finished": "Finished",
    "stopped": "Stopped",
    "failed": "Failed",
    "stale": "Stale",
    "lost": "Stopped without a final record",
}
STATES = tuple(LABELS)
#: States in which a process may still be integrating (or waiting to).
ACTIVE_STATES = ("waiting", "running", "stale")

CLI = "gpuwm-cm1"


@dataclass
class RunStatus:
    """One run's status, as the Runs table, the Run screen and ``status --json`` show it."""

    rundir: str
    name: str
    state: str
    label: str
    detail: str = ""
    percent: float | None = None
    step: int | None = None
    total_steps: int | None = None
    model_time_s: float | None = None
    duration_s: float | None = None
    eta_s: float | None = None
    s_per_step: float | None = None
    model_s_per_wall_s: float | None = None
    phase: str | None = None
    segment: int | None = None
    segments: int = 0
    host: str | None = None
    local: bool | None = None
    heartbeat_age_s: float | None = None
    frames: int = 0
    last_frame: Mapping[str, Any] | None = None
    last_restart: Mapping[str, Any] | None = None
    scheduler: Mapping[str, Any] | None = None
    scheduler_state: str | None = None
    queue: Mapping[str, Any] | None = None
    gpu: Mapping[str, Any] | None = None
    end: Mapping[str, Any] | None = None
    stop_requested_age_s: float | None = None
    pause_requested: bool = False
    #: The segment's planned end and restart cadence, and when a pause would
    #: take effect (the next restart; at the end when only one is written).
    until_s: float | None = None
    restart_interval_s: float | None = None
    next_restart_s: float | None = None
    grid: Mapping[str, Any] | None = None
    actions: list[dict[str, str]] = field(default_factory=list)
    view_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


# --------------------------------------------------------------- schedulers


_SCHEDULER_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
#: A login node's scheduler is asked about one job at most this often.
SCHEDULER_CACHE_S = 20.0


def scheduler_job_state(kind: str | None, job_id: str | None, *, now: float | None = None) -> str:
    """``gone``, ``unknown``, or the scheduler's state word for a job (``RUNNING``, ``R``...).

    ``unknown`` when the job or its scheduler's command is not available here,
    or the command failed in a way that says nothing about the job.
    """

    if not kind or not job_id:
        return "unknown"
    moment = time.monotonic() if now is None else now
    cached = _SCHEDULER_CACHE.get((kind, str(job_id)))
    if cached is not None and moment - cached[0] < SCHEDULER_CACHE_S:
        return cached[1]
    if kind == "slurm":
        state = _squeue(str(job_id))
    elif kind == "pbs":
        state = _qstat(str(job_id))
    else:
        state = "unknown"
    _SCHEDULER_CACHE[(kind, str(job_id))] = (moment, state)
    return state


def _run(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _squeue(job_id: str) -> str:
    squeue = shutil.which("squeue")
    if squeue is None:
        return "unknown"
    completed = _run([squeue, "--noheader", "--jobs", job_id, "--format=%T"])
    if completed is None:
        return "unknown"
    text = (completed.stdout or "").strip()
    if completed.returncode == 0:
        # squeue lists pending and running jobs; a finished one is simply absent.
        return text.splitlines()[0].strip() if text else "gone"
    if "invalid job id" in (completed.stderr or "").lower():
        return "gone"
    return "unknown"


def _qstat(job_id: str) -> str:
    qstat = shutil.which("qstat")
    if qstat is None:
        return "unknown"
    completed = _run([qstat, "-f", job_id])
    if completed is None:
        return "unknown"
    if completed.returncode != 0:
        message = (completed.stderr or "") + (completed.stdout or "")
        return "gone" if "unknown job id" in message.lower() or "job has finished" in message.lower() else "unknown"
    for line in (completed.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("job_state"):
            state = stripped.split("=", 1)[-1].strip()
            # F (finished, PBS Pro with -x) and C (completed, Torque) are gone.
            return "gone" if state in ("F", "C") else state
    return "unknown"


# ------------------------------------------------------------------ helpers


def _age(moment: float | None, now: float) -> float | None:
    return None if moment is None else max(0.0, now - moment)


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _command(*words: str) -> str:
    return shlex.join([CLI, *words])


def _actions(run: RunDirectory, state: str, *, local: bool, restartable: bool,
             next_restart_s: float | None = None) -> list[dict[str, str]]:
    path = os.fspath(run.path)
    pause = "Pause at next restart" + ("" if next_restart_s is None else f" ({next_restart_s:g} s)")
    table = {
        "start": ("Start", _command("run", path, "--detach")),
        "stop": ("Stop", _command("stop", path)),
        "pause": (pause, _command("pause", path)),
        "terminate": ("Terminate (SIGTERM)", _command("stop", path, "--terminate")),
        "force-stop": ("Force stop", _command("stop", path, "--force")),
        "continue": ("Continue from last restart", _command("resume", path, "--detach")),
        "extend": ("Continue and extend", _command("resume", path, "--until", "T", "--detach")),
        "start-over": ("Start over", _command("run", path, "--start-over", "--detach")),
        "abandon": ("Stop tracking this run", _command("stop", path, "--abandon")),
        "job-script": ("Copy job script", _command("jobscript", path, "--scheduler", "slurm")),
    }
    wanted: list[str] = []
    if state == "ready":
        wanted = ["start", "job-script"]
    elif state == "waiting":
        wanted = ["stop"] + (["force-stop"] if local else [])
    elif state == "running":
        wanted = ["stop", "pause"] + (["terminate", "force-stop"] if local else [])
    elif state == "stale":
        wanted = ["stop", "abandon"]
    elif state in ("paused", "stopped", "failed", "lost"):
        wanted = (["continue", "extend"] if restartable else []) + ["start-over", "job-script"]
    elif state == "finished":
        wanted = (["extend"] if restartable else []) + ["start-over"]
    return [{"id": key, "label": table[key][0], "command": table[key][1]} for key in wanted]


def _grid(record: Mapping[str, Any]) -> dict[str, Any] | None:
    cm1 = record.get("cm1") or {}
    if not cm1:
        return None
    return {key: cm1.get(key) for key in ("nx", "ny", "nz", "dx", "dy", "dz") if key in cm1}


def _fill_progress(status: RunStatus, events: SegmentEvents) -> None:
    progress = events.last_progress or {}
    frame = events.last_frame
    status.step = progress.get("step")
    status.total_steps = progress.get("total_steps")
    status.model_time_s = progress.get("model_time_s")
    status.duration_s = progress.get("duration_s")
    status.eta_s = progress.get("eta_s")
    status.s_per_step = progress.get("s_per_step_recent") or progress.get("s_per_step_mean")
    status.model_s_per_wall_s = progress.get("model_s_per_wall_s")
    status.phase = progress.get("phase")
    status.gpu = progress.get("gpu") or status.gpu
    if events.admitted is not None:
        status.duration_s = status.duration_s or events.admitted.get("duration_s")
        status.total_steps = status.total_steps or events.admitted.get("steps")
    if frame is not None and (status.model_time_s is None or frame.get("model_time_s", 0) > status.model_time_s):
        status.model_time_s = frame.get("model_time_s")
        status.step = frame.get("step")
    if events.end is not None:
        status.step = events.end.get("step", status.step)
        status.model_time_s = events.end.get("model_time_s", status.model_time_s)
    if status.step is not None and status.total_steps:
        status.percent = round(100.0 * float(status.step) / float(status.total_steps), 1)
    status.last_frame = frame
    status.last_restart = events.last_restart


def _last_frame_words(run: RunDirectory) -> str:
    frames = run.frames()
    if not frames:
        return "no frame was published"
    last = frames[-1]
    when = "" if last.model_time_s is None else f", {last.model_time_s:g} s"
    return f"last frame {last.index}{when}"


def _ended(status: RunStatus, events: SegmentEvents, segment_state: Mapping[str, Any]) -> None:
    end = events.end or {}
    reason = end.get("reason")
    status.end = dict(end)
    override = segment_state.get("reason_override")
    if override == "stopped" and reason in ("refused", "error"):
        reason = "stopped"
    message = end.get("message")
    if reason == "completed":
        status.state = "finished"
        status.detail = (
            f"Finished at {status.model_time_s:g} s of storm time." if status.model_time_s is not None
            else "Finished."
        )
    elif reason == "stopped":
        status.state = "stopped"
        why = {
            "stop": "a stop request",
            None: "a signal" if end.get("signal") else (message or "a stop rule"),
        }.get(end.get("request"), "a request")
        if segment_state.get("stopped_before_start"):
            status.detail = "Stopped before it started; nothing ran."
        else:
            status.detail = f"Stopped by {why} at {status.model_time_s or 0:g} s."
    elif reason == "paused":
        restart = events.last_restart
        until = (segment_state.get("plan") or {}).get("until_s")
        if until is not None and status.model_time_s is not None and status.model_time_s >= float(until):
            # The next restart was the last one: the run paused at its end.
            status.state = "finished"
            status.detail = f"Finished at {status.model_time_s:g} s of storm time (paused on its last restart)."
        else:
            status.state = "paused"
            status.detail = (
                f"Paused after restart {restart.get('number')} at {restart.get('model_time_s'):g} s."
                if restart else "Paused."
            )
    elif reason == "courant-stop":
        status.state = "failed"
        status.detail = f"The Courant number limit stopped the run at step {end.get('step')}. {message or ''}".strip()
    elif reason == "refused":
        status.state = "failed"
        refused = events.refused or {}
        status.detail = f"Refused before it started: {refused.get('reason') or message or 'see the log'}"
    else:
        status.state = "failed"
        status.detail = f"Ended with an error (exit code {end.get('rc')}): {message or 'see run.log'}"


def _label(status: RunStatus, *, last_frame_words: str = "") -> str:
    if status.state == "running":
        if status.phase in (None, "starting") and not status.step:
            return "Starting"
        if status.percent is not None:
            return f"Running {status.percent:.0f}%"
        return "Running"
    if status.state == "lost":
        return f"Stopped without a final record ({last_frame_words})" if last_frame_words else LABELS["lost"]
    return LABELS[status.state]


# ------------------------------------------------------------------- status


def run_status(
    path: Path | str | RunDirectory,
    *,
    now: float | None = None,
    this_host: str | None = None,
    scheduler: Callable[[str | None, str | None], str] | None = None,
) -> RunStatus:
    """The status of one run directory."""

    run = path if isinstance(path, RunDirectory) else RunDirectory(path)
    now = time.time() if now is None else now
    this_host = host_name() if this_host is None else this_host
    ask = scheduler_job_state if scheduler is None else scheduler
    record = run.load()
    status = RunStatus(
        rundir=os.fspath(run.path), name=str(record.get("name") or run.path.name),
        state="ready", label=LABELS["ready"], grid=_grid(record),
    )
    segments = run.segments()
    status.segments = len(segments)
    imported = record.get("import") or {}
    if not segments:
        acknowledged = {str(item) for item in record.get("acknowledgments") or []}
        needed = [
            item for item in imported.get("acknowledgments_needed") or []
            if str(item.get("id") if isinstance(item, Mapping) else item) not in acknowledged
        ]
        if not run.deck.is_file() or imported.get("result") == "refused":
            status.state = "imported"
            reasons = imported.get("refused") or ["the namelist was not translated"]
            status.detail = f"Cannot run as written: {reasons[0]}" + (
                f" (and {len(reasons) - 1} more)" if len(reasons) > 1 else ""
            )
        elif needed:
            status.state = "imported"
            status.detail = f"Needs your acknowledgment of {len(needed)} setting(s) before it can run."
        else:
            status.detail = "Ready to start."
        status.label = _label(status)
        status.actions = _actions(run, status.state, local=True, restartable=False)
        return status

    segment = segments[-1]
    state = segment.state()
    row = run.segment_record(segment.number) or {}
    events = summarize(segment.events)
    status.segment = segment.number
    status.host = state.get("host") or row.get("host")
    status.scheduler = state.get("scheduler") or row.get("scheduler")
    status.gpu = state.get("gpu") or row.get("gpu")
    status.queue = state.get("queue")
    status.local = status.host is None or status.host == this_host
    _fill_progress(status, events)
    plan = state.get("plan") or {}
    status.until_s = plan.get("until_s", row.get("until_s"))
    status.restart_interval_s = plan.get("restart_interval_s", row.get("restart_interval_s"))
    if status.until_s is not None and status.restart_interval_s:
        now_s = status.model_time_s or float(plan.get("start_model_time_s") or 0.0)
        every = float(status.restart_interval_s)
        status.next_restart_s = min(float(status.until_s), (int(now_s // every) + 1) * every)
    status.frames = len(run.frames())
    stop_mtime = _mtime(run.control / "stop")
    status.stop_requested_age_s = _age(stop_mtime, now)
    status.pause_requested = (run.control / "pause").exists()
    heartbeat = max(
        (moment for moment in (
            events.last_time, parse_utc(state.get("heartbeat_utc")), _mtime(segment.events),
        ) if moment is not None),
        default=None,
    )
    status.heartbeat_age_s = _age(heartbeat, now)
    # What "Continue" would resume from: the latest segment's lineage (rundir.find_restart).
    restartable = bool(run.lineage_restarts())

    if events.end is not None:
        _ended(status, events, state)
    elif state.get("state") == "ended":
        # The supervisor saw its process end without an end record, or the
        # segment was force-stopped or abandoned.
        reason = state.get("reason")
        if reason == "force-stopped":
            status.state = "stopped"
            status.detail = "Force stopped. The frames and restarts written before it are kept."
        elif reason == "abandoned":
            status.state = "stopped"
            status.detail = "No longer tracked (marked as ended by hand)."
        else:
            status.state = "lost"
            exit_code = state.get("rc")
            how = f"exit code {exit_code}" if isinstance(exit_code, int) and exit_code >= 0 else (
                f"signal {-exit_code}" if isinstance(exit_code, int) else "an unknown cause"
            )
            status.detail = f"The run process ended ({how}) without a final record; {_last_frame_words(run)}."
    elif not state and events.records == 0 and (
        (_age(_mtime(segment.path), now) or 0.0) <= STALE_AFTER_S
    ):
        # A supervisor creates segments/NNN a moment before it writes
        # segment.json with the identities the checks below need.
        status.state = "running"
        status.phase = "starting"
        status.detail = "Starting."
    elif status.local:
        # This host: pid and /proc start time, never a pid alone.
        child_alive = local_process_alive(state.get("child"))
        supervisor_alive = local_process_alive(state.get("supervisor"))
        if child_alive:
            status.state = "running"
            status.detail = _running_detail(status, events)
        elif supervisor_alive and state.get("state") in ("waiting-for-gpu", None):
            status.state = "waiting"
            status.detail = _waiting_detail(status)
        elif supervisor_alive:
            status.state = "running"
            status.phase = status.phase or "starting"
            status.detail = "Starting."
        else:
            status.state = "lost"
            status.detail = f"The run's processes are gone and its log has no final record; {_last_frame_words(run)}."
    else:
        # Another host: /proc means nothing here.  Heartbeat, then the scheduler.
        age = status.heartbeat_age_s
        if age is not None and age <= STALE_AFTER_S:
            if state.get("state") == "waiting-for-gpu" and events.records == 0:
                status.state = "waiting"
                status.detail = _waiting_detail(status)
            else:
                status.state = "running"
                status.detail = _running_detail(status, events)
        else:
            job = status.scheduler or {}
            answer = ask(job.get("kind"), job.get("job_id"))
            status.scheduler_state = answer
            if answer == "gone":
                status.state = "lost"
                status.detail = (
                    f"Job {job.get('job_id')} is no longer listed by the scheduler and the log has "
                    f"no final record; {_last_frame_words(run)}."
                )
            else:
                status.state = "stale"
                seen = "never" if age is None else f"{age:.0f} s ago"
                listed = (
                    f" The scheduler lists job {job.get('job_id')} as {answer}." if answer not in ("unknown",)
                    else ""
                )
                status.detail = f"Running on {status.host}; its last record was {seen}.{listed}"
    status.label = _label(status, last_frame_words=_last_frame_words(run) if status.state == "lost" else "")
    status.actions = _actions(run, status.state, local=bool(status.local), restartable=restartable,
                              next_restart_s=status.next_restart_s)
    return status


def _running_detail(status: RunStatus, events: SegmentEvents) -> str:
    if status.model_time_s is None or not status.duration_s:
        return "Starting." if events.admitted is None or not status.step else "Running."
    words = f"{status.model_time_s:g} s of {status.duration_s:g} s of storm time"
    if status.eta_s is not None:
        words += f", about {max(status.eta_s, 0) / 60:.0f} min left" if status.eta_s >= 90 else f", about {status.eta_s:.0f} s left"
    if status.local and status.heartbeat_age_s is not None and status.heartbeat_age_s > STALE_AFTER_S:
        words += f"; no record for {status.heartbeat_age_s:.0f} s"
    return words + "."


def _waiting_detail(status: RunStatus) -> str:
    queue = status.queue or {}
    holder = queue.get("holder") or {}
    ahead = queue.get("ahead")
    words = "Waiting for the GPU"
    gpu = (status.gpu or {}).get("name")
    if gpu:
        words += f" ({gpu})"
    if holder.get("rundir"):
        words += f"; it is running {Path(str(holder['rundir'])).name}"
    elif holder.get("pid"):
        words += f"; process {holder['pid']} holds it"
    if isinstance(ahead, int) and ahead > 0:
        words += f"; {ahead} run(s) ahead in line"
    return words + "."


def view_only_status(path: Path | str, *, now: float | None = None) -> RunStatus | None:
    """A plain ``cartesian`` output directory (no run.json): view only.

    Its log, if it has one, is read with the heartbeat rule, since no process
    identity was recorded for it.
    """

    root = Path(path)
    logs = sorted(root.glob("*.jsonl"))
    if not logs and not (root / "frames").is_dir() and not any(root.glob("frame-*")):
        return None
    now = time.time() if now is None else now
    status = RunStatus(rundir=os.fspath(root), name=root.name, state="stopped", label="", view_only=True)
    if logs:
        events = summarize(logs[0])
        _fill_progress(status, events)
        status.heartbeat_age_s = _age(events.last_time, now)
        if events.end is not None:
            _ended(status, events, {})
        elif status.heartbeat_age_s is not None and status.heartbeat_age_s <= STALE_AFTER_S:
            status.state = "running"
        else:
            status.state = "stale"
            status.detail = "No run.json and no recent record; opened view-only."
    else:
        status.state = "finished"
        status.detail = "A cartesian output directory without a run.json; opened view-only."
    status.label = _label(status)
    return status


def list_runs(root: Path | str) -> list[RunStatus]:
    """Every run directory directly under ``root`` (and ``root`` itself if it is one)."""

    root = Path(root).expanduser().absolute()
    found: list[RunStatus] = []
    candidates = [root] + sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    for candidate in candidates:
        run = RunDirectory(candidate)
        if run.is_run():
            try:
                found.append(run_status(run))
            except (RunDirectoryError, OSError) as error:
                found.append(RunStatus(
                    rundir=os.fspath(candidate), name=candidate.name, state="failed", label="Unreadable",
                    detail=str(error),
                ))
        elif candidate != root:
            view = view_only_status(candidate)
            if view is not None:
                found.append(view)
    return found


def format_status(status: RunStatus) -> str:
    """One run as two plain lines for a terminal."""

    time_words = ""
    if status.model_time_s is not None and status.duration_s:
        time_words = f"  {status.model_time_s:g} / {status.duration_s:g} s"
    where = f"  on {status.host}" if status.host and not status.local else ""
    line = f"{status.name}: {status.label}{time_words}{where}"
    if status.detail:
        line += f"\n  {status.detail}"
    return line


__all__ = [
    "ACTIVE_STATES",
    "LABELS",
    "RunStatus",
    "STALE_AFTER_S",
    "STATES",
    "format_status",
    "list_runs",
    "run_status",
    "scheduler_job_state",
    "view_only_status",
]
