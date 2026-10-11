"""Stop and pause a running integration through files in a control directory.

``gpuwm-cm1 cartesian --integrate --control-dir DIR`` checks ``DIR`` once per
completed large step.  The check is one ``stat()`` of the directory: creating
a file in it changes its modification time, and then it is listed.  It is also
listed at least every :data:`LIST_AT_LEAST_EVERY_S` of wall time whatever the
``stat()`` said: on NFS a ``stat()`` can return the directory's cached
attributes for up to a minute (acdirmax), so a ``stop`` written from a login
node would go unseen that long, while opening the directory to list it
revalidates them (close-to-open consistency).

``DIR/stop``
    Stop at the next large-step boundary.  A publication already in flight is
    finished, not discarded (it is a completed model state), no new frame or
    restart is taken, and the run writes its receipt with the status
    ``stopped-at-run-control-request`` and exits 0.  If the stop is seen on a
    step at which a frame or restart falls due, that event is published
    first: its state has already been computed.
``DIR/pause``
    Keep stepping to the next scheduled restart, publish it, wait until it is
    durable, and stop there with the status ``paused-at-run-control-request``
    so ``--restart-from`` (``run.irst = 1``, ``run.rstnum = N``) continues
    the run.  A run with no restart left to write cannot pause; it says so
    once (a ``warning`` event and a stderr line) and keeps running, and
    ``stop`` still works.

``stop`` wins when both exist.  SIGTERM is the fallback for a process that is
not reading the directory (still compiling, or wedged): it cancels the
publication in flight, discards its staging and exits 143.

The same object keeps the ``--events`` progress records coming from the
stepping thread (:mod:`cm1core.run_events`).  Importing this module imports
neither CuPy nor numpy.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from .run_events import PROGRESS_INTERVAL_S, RunEventLog

RUN_CONTROL_SCHEMA = "gm1.run-control/v1"
STOP = "stop"
PAUSE = "pause"
CONTROL_FILES = (STOP, PAUSE)
#: A directory modified this recently is listed on every poll: a filesystem
#: with coarse timestamps can create a second file within the same tick as
#: the first, leaving the modification time unchanged.
_RECENT_CHANGE_S = 2.0
#: The directory is listed at least this often (wall seconds, on the control's
#: clock) even when its modification time has not moved; see the module text.
LIST_AT_LEAST_EVERY_S = 5.0
_RECENT_STEPS = 32


def _utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="milliseconds")


class RunControl:
    """The run-control directory and the stepping thread's progress records.

    ``directory`` may be ``None`` (events only) and ``events`` may be ``None``
    (control only).  The integration calls :meth:`begin` once before its
    first large step, :meth:`after_step` after every one and :meth:`finishing`
    after its last; it never calls anything else.
    """

    def __init__(self, directory: Path | str | None, events: RunEventLog | None = None, *,
                 progress_interval_s: float = PROGRESS_INTERVAL_S,
                 clock: Callable[[], float] = time.monotonic,
                 stream: Any = None) -> None:
        self.directory = None if directory is None else Path(directory)
        self.events = events
        self.progress_interval_s = float(progress_interval_s)
        self._clock = clock
        self._stream = stream
        self._stat_key: tuple[int, int] | None = None
        self._listed_at = 0.0
        self.polls = 0
        self.listings = 0
        self.request: str | None = None
        self.request_seen: dict[str, Any] | None = None
        self.pause_refused = False
        self.pending_pause_seen: dict[str, Any] | None = None
        # Stepping statistics of this integration (a resumed run's are its own).
        self.step_count = 0
        self.step_total_s = 0.0
        self.recent: deque[float] = deque(maxlen=_RECENT_STEPS)
        self._started: float | None = None
        self._last_progress = 0.0
        self._model_start_s: float | None = None
        self._total_steps: int | None = None
        self._duration_s: float | None = None
        self.runtime: Any = None
        self.stopped: dict[str, Any] | None = None

    # ---- the control directory

    def poll(self) -> str | None:
        """One ``stat()`` of the directory; the latched request, if any."""

        if self.directory is None or self.request == STOP:
            return self.request
        self.polls += 1
        try:
            stat = os.stat(self.directory)
        except OSError:
            return self.request
        key = (stat.st_mtime_ns, stat.st_ino)
        now = time.time()
        tick = self._clock()
        if (
            key == self._stat_key and now - stat.st_mtime > _RECENT_CHANGE_S
            and tick - self._listed_at < LIST_AT_LEAST_EVERY_S
        ):
            return self.request
        self._stat_key = key
        self._listed_at = tick
        self.listings += 1
        try:
            present = {name for name in os.listdir(self.directory) if name in CONTROL_FILES}
        except OSError:
            return self.request
        for name in (STOP, PAUSE):
            if name in present and self.request != STOP and self.request != name:
                self._seen(name)
                break
        return self.request

    def _seen(self, name: str) -> None:
        path = self.directory / name
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            mtime = None
        runtime = self.runtime
        self.request = name
        self.request_seen = {
            "request": name,
            "file": os.fspath(path),
            "file_mtime_utc": None if mtime is None else _utc(mtime),
            "seen_utc": _utc(time.time()),
            "seen_after_file_s": None if mtime is None else max(0.0, time.time() - mtime),
            "seen_at_step": None if runtime is None else int(runtime.clock.completed_steps),
            "seen_at_model_time_s": (
                None if runtime is None else float(runtime.clock.model_seconds)
            ),
        }
        if name == PAUSE:
            self.pending_pause_seen = dict(self.request_seen)
        self._notice(
            f"gm1 run control: {name} requested ({path}) at step "
            f"{self.request_seen['seen_at_step']}; "
            + ("stopping at this step boundary" if name == STOP
               else "stopping after the next restart is durable")
        )

    @property
    def stop_requested(self) -> bool:
        return self.request == STOP

    @property
    def pause_requested(self) -> bool:
        return self.request == PAUSE

    def refuse_pause(self, reason: str) -> None:
        """Say once that a pause cannot be honoured, and keep running."""

        if self.pause_refused:
            return
        self.pause_refused = True
        self._notice(f"gm1 run control: pause cannot be honoured: {reason}; the run continues")
        if self.events is not None:
            self.events.emit(
                "warning", code="GM1-RUN-CONTROL-PAUSE-UNAVAILABLE",
                message=f"pause cannot be honoured: {reason}; the run continues, and "
                "a stop file still stops it",
            )

    def _notice(self, line: str) -> None:
        try:
            print(line, file=self._stream if self._stream is not None else sys.stderr, flush=True)
        except (OSError, ValueError):
            pass

    # ---- the integration's three calls

    def begin(self, runtime: Any, *, total_steps: int | None, duration_s: float,
              model_start_s: float, startup: dict[str, Any] | None = None) -> None:
        self.runtime = runtime
        self._total_steps = total_steps
        self._duration_s = float(duration_s)
        self._model_start_s = float(model_start_s)
        self._started = self._clock()
        self._last_progress = self._started
        if self.events is not None:
            if startup is not None:
                self.events.emit("startup", **startup)
            self.events.set_phase("stepping")
            self.events.progress(self.progress_fields())
        # A request made while the run compiled is seen before the first step.
        self.poll()

    def after_step(self, elapsed_s: float) -> str | None:
        self.step_count += 1
        self.step_total_s += float(elapsed_s)
        self.recent.append(float(elapsed_s))
        if self.events is not None:
            now = self._clock()
            if now - self._last_progress >= self.progress_interval_s:
                self._last_progress = now
                self.events.progress(self.progress_fields())
        return self.poll()

    def finishing(self) -> None:
        if self.events is not None:
            self.events.set_phase("finishing")
            self.events.progress(self.progress_fields())

    # ---- records

    def progress_fields(self) -> dict[str, Any]:
        from .runtime_profiler import io_summary, progress_fields

        runtime = self.runtime
        clock = runtime.clock
        fields = progress_fields(
            step=int(clock.completed_steps), total_steps=self._total_steps,
            model_time_s=float(clock.model_seconds), duration_s=self._duration_s,
            model_start_s=self._model_start_s,
            wall_s=0.0 if self._started is None else self._clock() - self._started,
            mean_s=self.step_total_s / self.step_count if self.step_count else None,
            recent_s=sum(self.recent) / len(self.recent) if self.recent else None,
        )
        profiler = getattr(runtime, "profiler", None)
        last_io = None if profiler is None else getattr(profiler, "last_io", None)
        fields["last_io"] = None if last_io is None else io_summary(last_io)
        gpu = None if profiler is None else profiler.gpu_reading()
        if gpu is None:
            # Without NVML: the device's own memory counters, which every
            # CUDA runtime has (called on the stepping thread, in its context).
            try:
                free, total = runtime.cp.cuda.runtime.memGetInfo()
                gpu = {"memory_used_bytes": int(total - free), "memory_total_bytes": int(total)}
            except Exception:  # noqa: BLE001 - telemetry is optional
                gpu = None
        fields["gpu"] = gpu
        return fields

    def stop_record(self, *, restart_number: int | None, frames_published: int,
                    frames_requested: int) -> dict[str, Any]:
        """The receipt block of a run this directory stopped or paused."""

        runtime = self.runtime
        seen = self.request_seen if self.request == STOP else self.pending_pause_seen
        return {
            "schema": RUN_CONTROL_SCHEMA,
            "request": self.request,
            **({} if seen is None else {
                key: value for key, value in seen.items() if key != "request"
            }),
            "ended_at_step": int(runtime.clock.completed_steps),
            "ended_at_model_time_s": float(runtime.clock.model_seconds),
            "restart_number": restart_number,
            "frames_requested": frames_requested,
            "frames_published": frames_published,
            "authority": (
                "gpuwm-cm1 run control: the control "
                "directory is checked once per completed large step"
            ),
        }

    def record(self) -> dict[str, Any]:
        return {
            "schema": RUN_CONTROL_SCHEMA,
            "directory": None if self.directory is None else os.fspath(self.directory),
            "polls": self.polls,
            "listings": self.listings,
            "request": self.request,
            "request_seen": self.request_seen,
            "pause_refused": self.pause_refused,
            "events": None if self.events is None else os.fspath(self.events.path),
            "progress_interval_s": self.progress_interval_s,
        }


def require_control_directory(directory: Path | str) -> Path:
    """Admit a control directory: created if absent, and holding no request.

    A ``stop`` or ``pause`` file already present would end the run before its
    first step, and the operator who left it there is not the one who will
    read the result.  Raises ``ValueError`` naming the file.
    """

    path = Path(directory)
    if path.exists() and not path.is_dir():
        raise ValueError(f"{path} exists and is not a directory")
    path.mkdir(parents=True, exist_ok=True)
    stale = [name for name in CONTROL_FILES if (path / name).exists()]
    if stale:
        raise ValueError(
            f"{', '.join(os.fspath(path / name) for name in stale)} already exists; "
            "the run would stop before its first step"
        )
    return path


__all__ = [
    "CONTROL_FILES",
    "LIST_AT_LEAST_EVERY_S",
    "PAUSE",
    "RUN_CONTROL_SCHEMA",
    "RunControl",
    "STOP",
    "require_control_directory",
]
