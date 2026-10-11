"""One output event's publications, committed on a worker thread.

CM1 writes its history and restart files synchronously (``cm1.F:3517-3520``
and ``cm1.F:4246``): the model does not advance while ``writeout`` or
``restart_write`` runs.  On this lane that means the GPU idles through every
SHA-256, every write, the publisher subprocess and every fsync of a
multi-gibibyte checkpoint.  None of that work reads the device.  So an event
is split: the device words are copied once into a host arena on the stepping
thread, and the rest of the publication is handed to one worker thread while
stepping continues.

The contract this module keeps:

* Queue depth one.  :meth:`BackgroundPublisher.submit` refuses while an event
  is still in flight, and the integration joins the previous event before it
  snapshots the next one, so the reused host arena is never read and written
  at once, frames and restarts reach disk in CM1's model order, and nothing
  is ever reported published before it is.
* Parts in turn.  A frame and a restart taken at the same step are two parts
  of one event, each already in its own host arena.  A part that says it
  awaits its turn (:attr:`PublicationPart.awaits_turn`) is committed on its
  own thread beside the earlier parts, so a restart is not written only after
  the frame has been, but it renames nothing into place until every earlier
  part of its event has (:meth:`PublicationCancel.await_turn`), and it is
  stopped and discarded if one of them failed.  The rename order, and what an
  event that succeeds or is cancelled leaves on disk, are what committing the
  parts one after another produces; the event just finishes when its slowest
  part does, not after the sum of them.  That equivalence does not extend to
  running out of a resource: the parts' writes and publisher payloads now
  overlap, so on a nearly full disk or under memory pressure the frame can hit
  ENOSPC or the OOM killer first, and then the frame is lost with its restart,
  where the parts written one after the other would have published the frame
  and failed only the restart.  ``BackgroundPublisher(concurrent_parts=False)``
  keeps them one after another (``integrate(concurrent_event_parts=False)``;
  ``cm1core cartesian --event-parts in-turn``).
* A worker error is re-raised by :meth:`BackgroundPublisher.join` on the
  stepping thread, which is where every other failure of the run surfaces.
* Every part of an event is atomic on disk: it writes inside a dot-prefixed
  staging directory and renames into place only after the last fsync.  A
  failed or cancelled part discards its staging directory, so nothing that
  was not completely written is left looking complete.
* The worker is a non-daemon thread, and :meth:`BackgroundPublisher.abort`
  cancels it and waits for it, so a run cannot return or raise with a worker
  still writing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time
from typing import Any


class PublicationCancelled(RuntimeError):
    """Raised inside a worker whose publication was cancelled."""


class TerminationRequested(SystemExit):
    """SIGTERM, raised on the stepping thread while an integration runs.

    A ``SystemExit`` so an uncaught one ends the process with the status a
    SIGTERM-killed process reports (128 + 15), after the integration has
    cancelled its in-flight publication and discarded the staging.
    """

    def __init__(self) -> None:
        super().__init__(143)


@contextmanager
def termination_as_exception() -> Iterator[None]:
    """Turn SIGTERM into :class:`TerminationRequested` for the enclosed block.

    Only from the main thread, and only over the default disposition: a
    process that installed its own SIGTERM handler keeps it.  Without this a
    SIGTERM ends the interpreter at once, the worker with it, and a staging
    directory -- never a published one -- is left behind.
    """

    import signal

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    try:
        previous = signal.getsignal(signal.SIGTERM)
    except (AttributeError, ValueError):
        yield
        return
    if previous is not signal.SIG_DFL:
        yield
        return

    def handler(signum: int, frame: Any) -> None:
        raise TerminationRequested()

    signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class PublicationCancel:
    """A cancellation flag that also stops the publisher subprocesses it holds.

    One per event.  A part committed beside the earlier parts of its event is
    handed its own view (made by :func:`run_event`): it shares the event's flag,
    so cancelling the event stops every part and kills every part's publisher,
    and it also knows the earlier parts, so it can wait for their publication
    (:meth:`await_turn`) and stops as soon as one of them has failed.  Handed
    to a part committed after the earlier ones have finished -- every part on
    the stepping thread, every part that does not await its turn -- the event's
    own object has no earlier parts to know about, and :meth:`await_turn` is
    just :meth:`check`.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes: list[Any] = []
        self._views: list[PublicationCancel] = []
        self._earlier: tuple[_PartOutcome, ...] = ()
        self._root: PublicationCancel = self
        self.turn_awaited = False
        self.turn_wait_seconds = 0.0

    def _view(self, earlier: list[_PartOutcome]) -> "PublicationCancel":
        """A part's view: this event's flag and lock, its own processes."""

        root = self._root
        view = PublicationCancel.__new__(PublicationCancel)
        view._event, view._lock, view._root = root._event, root._lock, root
        view._processes, view._views = [], []
        view._earlier = tuple(earlier)
        view.turn_awaited, view.turn_wait_seconds = False, 0.0
        with root._lock:
            root._views.append(view)
        return view

    def _earlier_failure(self) -> "_PartOutcome | None":
        return next((earlier for earlier in self._earlier if earlier.failed), None)

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or self._earlier_failure() is not None

    def check(self) -> None:
        if self._event.is_set():
            raise PublicationCancelled(
                "the publication was cancelled before it completed; its staging "
                "directory is discarded and nothing was renamed into place"
            )
        failed = self._earlier_failure()
        if failed is not None:
            raise EarlierPartFailed(
                f"{failed.label}, earlier in this event, was not published, and a "
                "part is renamed into place only after every earlier part of its "
                "event is, as committing them one after another would have left "
                "it; this part's staging directory is discarded"
            )

    def await_turn(self) -> None:
        """Wait until every earlier part of the event is finished; then :meth:`check`.

        A commit calls this immediately before it renames anything into place.
        It raises if the event was cancelled or an earlier part failed, so a
        part never appears on disk ahead of, or without, an earlier part of
        its event.  The wait is not interruptible, and need not be: it only
        ever runs off the main thread, where no signal handler raises, or on
        the stepping thread after the earlier parts have finished.
        """

        if self._earlier:
            started = time.perf_counter()
            for earlier in self._earlier:
                earlier.settled.wait()
            self.turn_wait_seconds += time.perf_counter() - started
        self.turn_awaited = True
        self.check()

    def attach(self, process: Any) -> None:
        with self._lock:
            self._processes.append(process)
        if self.cancelled:
            _kill(process)

    def detach(self, process: Any) -> None:
        with self._lock:
            self._processes[:] = [held for held in self._processes if held is not process]

    def cancel(self) -> None:
        """Cancel the whole event: every part stops and every publisher is killed."""

        root = self._root
        with root._lock:
            root._event.set()
            processes = [
                *root._processes,
                *(process for view in root._views for process in view._processes),
            ]
        for process in processes:
            _kill(process)

    def _stop_after_failure(self) -> None:
        """An earlier part failed: kill this part's publisher, if one is running."""

        with self._lock:
            processes = list(self._processes)
        for process in processes:
            _kill(process)


class EarlierPartFailed(PublicationCancelled):
    """Raised inside a part whose event had an earlier part fail first."""


@dataclass
class _PartOutcome:
    """What a later part of the same event may know about an earlier one."""

    label: str
    settled: threading.Event = field(default_factory=threading.Event)
    failed: bool = False


def _kill(process: Any) -> None:
    """Kill the publisher and anything it started.

    The publisher runs in its own session (``start_new_session``), so its
    process group is its own pid; killing the group also stops a wrapper's
    children, which would otherwise hold the output pipes open.
    """

    import os
    import signal

    try:
        os.killpg(int(process.pid), signal.SIGKILL)
        return
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        process.kill()
    except (OSError, ProcessLookupError):
        pass


@dataclass
class PublicationPart:
    """One artifact of an event: a frame or a restart.

    ``commit`` runs on the worker and returns what the stepping thread will
    report; ``discard`` removes whatever ``commit`` left in staging when it
    did not finish; ``io`` is the profiler event the snapshot opened, or
    ``None``; ``detail`` is the snapshot itself, for callers that need a
    value it captured before the commit has run.

    ``awaits_turn`` is a promise about ``commit``: it reads nothing an earlier
    part of the event writes, and it calls ``cancel.await_turn()`` before it
    renames anything into place.  The worker may then start it beside the
    earlier parts instead of after them.  ``turn_wait_seconds`` is how long
    such a part waited there for them (``None`` for a part that never ran
    beside an earlier one).
    """

    label: str
    commit: Callable[[PublicationCancel], Any]
    discard: Callable[[], None]
    io: Any = None
    snapshot_seconds: float = 0.0
    detail: Any = None
    result: Any = None
    started: float | None = None
    finished: float | None = None
    awaits_turn: bool = False
    turn_wait_seconds: float | None = None


@dataclass
class EventPublication:
    """Every part of one output event, committed in the order given."""

    parts: list[PublicationPart]
    context: Any = None
    error: BaseException | None = None
    join_started: float | None = None
    join_finished: float | None = None
    background: bool = True
    aborted: bool = False
    completed: list[PublicationPart] = field(default_factory=list)
    #: A ``SystemExit`` (SIGTERM, as :class:`TerminationRequested`) that
    #: arrived while :meth:`BackgroundPublisher.abort` waited for the worker.
    #: The wait cannot end early, so it is kept here for the caller to raise
    #: once the worker has settled, instead of being dropped.
    termination: BaseException | None = None

    @property
    def labels(self) -> list[str]:
        return [part.label for part in self.parts]


_TAKEN_PARTS = "gm1_publication_parts_taken"


def attach_taken_parts(error: BaseException, parts: list[PublicationPart]) -> None:
    """Carry parts already taken off the device on the error that stopped a snapshot.

    A frame whose words are in host memory is a completed state even when
    something after it -- a nest child's snapshot -- fails; the caller that
    catches ``error`` publishes these before re-raising it, as the one-call
    publication published the parent frame before it tried the child.
    """

    try:
        setattr(error, _TAKEN_PARTS, [*taken_parts(error), *parts])
    except (AttributeError, TypeError):  # pragma: no cover - every BaseException has a __dict__
        pass


def taken_parts(error: BaseException) -> list[PublicationPart]:
    """The parts :func:`attach_taken_parts` put on ``error``, in order."""

    return list(getattr(error, _TAKEN_PARTS, ()))


def run_event(event: EventPublication, cancel: PublicationCancel,
              clock: Callable[[], float] = time.perf_counter, *,
              concurrent: bool = False) -> None:
    """Commit every part in order; on the first failure discard and stop.

    With ``concurrent`` -- the background worker, a thread no signal handler
    raises in -- each part after the first that awaits its turn is started at
    once on a thread of its own, and the rest run in order on this thread.
    The rename order is the ordered one: a part is renamed into place only
    after every earlier part has been, a failure stops every later part and
    discards what it staged, ``completed`` holds the parts published, in
    order, and ``error`` is the earliest failed part's error.  A later part's
    own error, when it failed before an earlier one did, is added to that
    error as a note; a later part that was only stopped (cancelled with the
    event, or after an earlier part failed) adds none.  Every thread started
    here has finished when this returns.

    Which parts fail is not always the ordered run's: the parts write and
    hold their publisher payloads at the same time, so when the disk or host
    memory runs out an earlier part can fail where, committed alone first, it
    would have been published, and every later part is then discarded with it.
    """

    parts = event.parts
    if not concurrent or not any(part.awaits_turn for part in parts[1:]):
        _run_in_order(event, cancel, clock)
        return
    outcomes = [_PartOutcome(part.label) for part in parts]
    views = [cancel._view(outcomes[:index]) for index in range(len(parts))]
    errors: list[BaseException | None] = [None] * len(parts)

    def commit(index: int) -> None:
        part, view, outcome = parts[index], views[index], outcomes[index]
        part.started = clock()
        try:
            view.check()
            part.result = part.commit(view)
            if index and part.awaits_turn and not view.turn_awaited:
                raise RuntimeError(
                    f"{part.label} was committed beside the earlier parts of its "
                    "event but never awaited its turn, so it may have been "
                    "renamed into place ahead of them"
                )
        except BaseException as error:  # noqa: BLE001 - re-raised at the join
            part.finished = clock()
            errors[index] = error
            outcome.failed = True
            for later in views[index + 1:]:
                later._stop_after_failure()
            try:
                part.discard()
            except BaseException as cleanup:  # noqa: BLE001
                error.add_note(f"discarding the staging of {part.label} also failed: {cleanup!r}")
        else:
            part.finished = clock()
        finally:
            if index and part.awaits_turn:
                part.turn_wait_seconds = view.turn_wait_seconds
            outcome.settled.set()

    helpers: list[threading.Thread] = []
    # Parts not yet started, here or on a helper; they run in order below.
    here = list(range(len(parts)))
    try:
        for index in range(1, len(parts)):
            if not parts[index].awaits_turn:
                continue
            helper = threading.Thread(
                target=commit, args=(index,), daemon=False,
                name=f"gm1-publication-{parts[index].label}",
            )
            try:
                helper.start()
            except Exception:  # noqa: BLE001 - no thread: commit it here, in order
                continue
            helpers.append(helper)
            here.remove(index)
        while here:
            index = here[0]
            for earlier in outcomes[:index]:
                earlier.settled.wait()
            here.pop(0)
            if any(earlier.failed for earlier in outcomes[:index]):
                # Never started, as in the ordered run: nothing to discard.
                outcomes[index].failed = True
                outcomes[index].settled.set()
                continue
            commit(index)
    except BaseException as error:  # noqa: BLE001 - recorded, then re-raised
        if event.error is None:
            event.error = error
        raise
    finally:
        # Anything that stopped this thread early leaves no helper waiting
        # for a part that will never run.
        for index in here:
            outcomes[index].failed = True
            outcomes[index].settled.set()
        for helper in helpers:
            helper.join()
    for part, error in zip(parts, errors):
        if error is None:
            if event.error is None and part.finished is not None:
                event.completed.append(part)
        elif event.error is None:
            event.error = error
        elif not isinstance(error, PublicationCancelled):
            # A later part stopped by the event's cancel or by an earlier
            # part's failure (EarlierPartFailed is a PublicationCancelled)
            # did not fail on its own; only its own failure is news.
            event.error.add_note(f"{part.label} also failed: {error!r}")


def _run_in_order(event: EventPublication, cancel: PublicationCancel,
                  clock: Callable[[], float]) -> None:
    for part in event.parts:
        part.started = clock()
        try:
            cancel.check()
            part.result = part.commit(cancel)
        except BaseException as error:  # noqa: BLE001 - re-raised at the join
            part.finished = clock()
            event.error = error
            try:
                part.discard()
            except BaseException as cleanup:  # noqa: BLE001
                error.add_note(f"discarding the staging of {part.label} also failed: {cleanup!r}")
            return
        part.finished = clock()
        event.completed.append(part)


# How long a submit whose Thread.start() raised waits to learn whether the
# thread exists after all.  A started worker's first statement sets a flag,
# and it needs only the GIL, which that wait releases.
_THREAD_START_GRACE_SECONDS = 2.0


def _began_within(began: threading.Event, seconds: float) -> tuple[bool, BaseException | None]:
    """Whether ``began`` is set within ``seconds``, and any interrupt meanwhile."""

    deadline = time.monotonic() + seconds
    later: BaseException | None = None
    while not began.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            began.wait(remaining)
        except BaseException as interrupt:  # noqa: BLE001 - raised by the caller
            later = interrupt
    return began.is_set(), later


class BackgroundPublisher:
    """At most one output event's publication in flight, on one worker thread.

    The worker commits the event's parts that await their turn beside each
    other, on threads it starts and joins itself (:func:`run_event`), so the
    event is still one unit to submit, join and abort.  With
    ``concurrent_parts=False`` it commits them one after another instead:
    the event then takes the sum of its parts' writes, and the parts neither
    compete for the disk nor hold their publisher payloads at the same time.

    Completion is a flag the worker sets when :func:`run_event` returns, and
    every wait is on that flag, never on ``Thread.join`` alone: on CPython
    3.12 a ``join`` interrupted by a signal handler's exception releases the
    still-running thread's state lock and marks it stopped (the bpo-45274
    handling), after which ``is_alive()`` is False while the worker is still
    writing.  ``join`` is only called once the flag is set.
    """

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter,
                 concurrent_parts: bool = True) -> None:
        if not isinstance(concurrent_parts, bool):
            raise TypeError("concurrent_parts must be a bool")
        self.clock = clock
        self.concurrent_parts = concurrent_parts
        self._thread: threading.Thread | None = None
        self._event: EventPublication | None = None
        self._cancel: PublicationCancel | None = None
        self._done: threading.Event | None = None
        self.events_submitted = 0
        self.join_wait_seconds = 0.0
        self.maximum_join_wait_seconds = 0.0

    @property
    def in_flight(self) -> EventPublication | None:
        return self._event

    @property
    def settled(self) -> bool:
        """An event is in flight and its worker has already finished it.

        Joining it then costs the stepping thread nothing, so a caller can
        poll this between steps and report -- or raise -- a publication as
        soon as it is durable instead of at the next output event.
        """

        done = self._done
        return self._event is not None and done is not None and done.is_set()

    def submit(self, event: EventPublication) -> None:
        if self._event is not None:
            raise RuntimeError(
                f"a publication of {self._event.labels} is still in flight; queue "
                "depth is one, so the previous event must be joined before the "
                "next is submitted or two events would share one host arena"
            )
        cancel = PublicationCancel()
        done = threading.Event()
        began = threading.Event()
        clock = self.clock
        concurrent = self.concurrent_parts

        def work() -> None:
            began.set()
            try:
                run_event(event, cancel, clock, concurrent=concurrent)
            finally:
                done.set()

        thread = threading.Thread(
            target=work, name=f"gm1-publication-{'+'.join(event.labels)}", daemon=False,
        )
        counted = False
        try:
            self._event, self._cancel, self._thread, self._done = event, cancel, thread, done
            thread.start()
            self.events_submitted += 1
            counted = True
        except BaseException:
            # start() raises either before the thread exists (it could not
            # start one) or after, when a signal handler's exception lands in
            # its wait for the new thread -- and then the new thread may not
            # have an ident yet, so ``ident`` cannot tell the two apart.  A
            # thread that exists sets ``began`` as soon as it runs, so the
            # event stays in flight exactly when that happens, and is then
            # joined or aborted rather than left writing unjoined.
            started, later = _began_within(began, _THREAD_START_GRACE_SECONDS)
            if not started:
                # Never started: nothing ran and there is nothing to join.
                self._event = self._cancel = self._thread = self._done = None
            elif not counted:
                self.events_submitted += 1
            if later is not None:
                raise later
            raise

    def join(self, *, raise_error: bool = True) -> EventPublication | None:
        """Wait for the event in flight; re-raise its error on this thread.

        Interruptible: a KeyboardInterrupt while waiting propagates and leaves
        the event in flight, so the caller can :meth:`abort` it.  With
        ``raise_error=False`` the settled event is returned and its ``error``
        is the caller's to raise.
        """

        event, thread, done = self._event, self._thread, self._done
        if event is None or thread is None or done is None:
            return None
        if event.join_started is None:
            event.join_started = self.clock()
        done.wait()
        thread.join()
        return self._settle(event, raise_error=raise_error)

    def abort(self) -> EventPublication | None:
        """Cancel the event in flight and wait until its worker has finished.

        Cancellation is prompt -- the arena writer checks it every write
        window and the publisher subprocess is killed -- so this wait is
        short, and it is not abandoned on a further interrupt: returning with
        the worker still running is exactly the unjoined exit this prevents.
        Never raises; the settled event says what was finished and what not,
        and a SIGTERM (any ``SystemExit``) that arrived during the wait is
        kept on it as ``termination`` for the caller to raise.  A worker held
        in an uninterruptible system call (an fsync on a hung NFS server) keeps
        this waiting however many signals arrive: only SIGKILL ends that
        process, as it would end the synchronous writer in the same call.
        """

        event, thread, cancel, done = self._event, self._thread, self._cancel, self._done
        if event is None or thread is None or cancel is None or done is None:
            return None
        if event.join_started is None:
            event.join_started = self.clock()
        event.aborted = True
        cancel.cancel()
        termination: BaseException | None = None
        while not done.is_set():
            try:
                done.wait()
            except (KeyboardInterrupt, SystemExit) as interrupt:  # noqa: PERF203 - keep waiting for the worker
                if isinstance(interrupt, SystemExit) and termination is None:
                    termination = interrupt
                cancel.cancel()
        while True:
            try:
                thread.join()
                break
            except (KeyboardInterrupt, SystemExit) as interrupt:  # noqa: PERF203 - run_event has returned
                if isinstance(interrupt, SystemExit) and termination is None:
                    termination = interrupt
                continue
        if termination is not None:
            event.termination = termination
        return self._settle(event, raise_error=False)

    def _settle(self, event: EventPublication, *, raise_error: bool = True) -> EventPublication:
        event.join_finished = self.clock()
        wait = event.join_finished - (event.join_started or event.join_finished)
        self.join_wait_seconds += wait
        self.maximum_join_wait_seconds = max(self.maximum_join_wait_seconds, wait)
        self._event = self._cancel = self._thread = self._done = None
        if raise_error and event.error is not None:
            raise event.error
        return event

    @property
    def event_parts(self) -> str:
        """How this worker commits an event's parts: a token for the receipt."""

        return EVENT_PARTS_BESIDE if self.concurrent_parts else EVENT_PARTS_IN_TURN

    def record(self) -> dict[str, Any]:
        return {
            "queue_depth": 1,
            "event_parts": self.event_parts,
            "event_parts_note": EVENT_PARTS_NOTES[self.event_parts],
            "events_submitted": self.events_submitted,
            "join_wait_seconds": self.join_wait_seconds,
            "maximum_join_wait_seconds": self.maximum_join_wait_seconds,
        }


#: How the parts of one background event reach disk: the receipt's
#: ``publication.event_parts`` token (``--event-parts`` on the command line)
#: and the sentence beside it.
EVENT_PARTS_BESIDE = "beside"
EVENT_PARTS_IN_TURN = "in-turn"
EVENT_PARTS_NOTES = {
    EVENT_PARTS_BESIDE: (
        "committed beside each other when they await their turn (a frame and a "
        "restart of the same step, each in its own host arena); each is renamed "
        "into place only after every earlier part of its event, and not at all "
        "if one of those failed"
    ),
    EVENT_PARTS_IN_TURN: (
        "committed one after another in the order taken (a frame, then the "
        "restart of the same step); a part that fails stops every later part"
    ),
}


def blocking_overlap(part: PublicationPart, event: EventPublication) -> float:
    """Seconds the stepping thread spent joined while this part committed.

    Parts committed beside each other would count the same seconds of one
    join twice, so a second of the join is this part's only when no earlier
    part of the event was still committing then -- which is how the one
    worker committing them in order would have spent it.
    """

    if (
        part.started is None or part.finished is None
        or event.join_started is None or event.join_finished is None
    ):
        return 0.0
    start = max(part.started, event.join_started)
    stop = min(part.finished, event.join_finished)
    if stop <= start:
        return 0.0
    pieces = [(start, stop)]
    for earlier in event.parts:
        if earlier is part:
            break
        if earlier.started is None or earlier.finished is None:
            continue
        pieces = [
            piece for low, high in pieces for piece in (
                ((low, high),) if earlier.finished <= low or high <= earlier.started
                else ((low, earlier.started), (earlier.finished, high))
            ) if piece[1] > piece[0]
        ]
    return sum(high - low for low, high in pieces)


__all__ = (
    "EVENT_PARTS_BESIDE",
    "EVENT_PARTS_IN_TURN",
    "EVENT_PARTS_NOTES",
    "BackgroundPublisher",
    "EarlierPartFailed",
    "EventPublication",
    "PublicationCancel",
    "PublicationCancelled",
    "PublicationPart",
    "TerminationRequested",
    "attach_taken_parts",
    "blocking_overlap",
    "run_event",
    "taken_parts",
    "termination_as_exception",
)
