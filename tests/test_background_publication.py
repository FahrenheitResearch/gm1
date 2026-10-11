"""Background publication of history frames and restarts, proved on the host.

No CUDA device is opened.  The runtime is ``test_cartesian_restart``'s host
double, whose ``integrate``, frame and restart publication are the shipped
methods; the publisher is the real ``cm1-forecast-publish`` built from this
tree, sometimes behind a wrapper that slows it down, limits its file size or
never returns, so that a failure or a signal lands while an event is really
in flight on the worker thread.

What is claimed:

* the same bytes whether an event is committed on the worker or on the
  stepping thread, over many events that reuse one host arena, and for a
  nest child's frames and placements taken with their parent frame;
* a run resumed from a restart written in the background -- a directory that
  no longer holds the raw arena -- is bit-identical to the continuous run;
* nothing reaches ``progress`` or the receipt before its directory is
  complete, and a finished publication is reported at the next step boundary;
* the raw restart arena is deleted only when the envelope header proves it;
* the stepping thread pays for the snapshot, not the publication;
* a frame and a restart of the same step are written beside each other, not
  one after the other, yet renamed into place in ``cm1.F``'s order, a failed
  frame stops the restart beside it, cancelling stops both, and the join
  that waits for them counts each second once;
* a worker exception, a full disk in the arena writer or in the publisher,
  SIGINT (once, twice and from another process) and SIGTERM each end the run
  with only complete directories published, no staging left behind and no
  worker thread alive.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from cm1core import cuda_cartesian_runtime as runtime_module
from cm1core.background_publication import (
    BackgroundPublisher,
    EarlierPartFailed,
    EventPublication,
    PublicationCancelled,
    PublicationPart,
    TerminationRequested,
    blocking_overlap,
)
from cm1core.cartesian_history import _run_checked
from cm1core import cartesian_restart as restart_module
from cm1core.cartesian_restart import (
    ARENA_FILENAME as RESTART_ARENA_FILENAME,
    RESTART_DIRECTORY_FORMAT,
    read_cartesian_restart,
    verify_restart_envelope_carries_arena,
    write_cartesian_restart,
)
from cm1core.forecast_arena import ForecastArenaWrite
from cm1core.cuda_cartesian_runtime import FRAME_DIRECTORY_FORMAT, FRAME_STAGING_DIRECTORY
from cm1core.runtime_profiler import RuntimeProfiler

from test_cartesian_restart import (  # noqa: E402 - the shared host double
    CpuCartesianRuntime,
    _config,
    forecast_publisher,  # noqa: F401 - the session fixture
)

ROOT = Path(__file__).resolve().parents[1]
RESTART_FILES = [
    "restart-field-table.tsv", "restart-publication.json",
    "restart-scheduler-scalars.tsv", "restart.cm1fcp",
]
FRAME_FILES = [
    "history-carriers.f32", "history-field-table.tsv", "history-frame-location.json",
    "history-publication.json", "history-scheduler-scalars.tsv", "history.cm1fhs",
]


# ------------------------------------------------------------------ helpers


def _wrapper(tmp_path: Path, real: Path, name: str, body: str) -> Path:
    """An executable publisher that answers ``--abi`` at once and runs ``body``."""

    script = tmp_path / name
    script.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        if [ "$1" = "--abi" ]; then exec "{real}" --abi; fi
        {body}
        exec "{real}" "$@"
    """))
    script.chmod(0o755)
    return script


def _slow(tmp_path: Path, real: Path, seconds: float, kind: str = "checkpoint") -> Path:
    return _wrapper(
        tmp_path, real, f"slow-{kind}-publisher",
        f'if [ "$1" = "{kind}" ]; then sleep {seconds}; fi',
    )


def _publication_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("gm1-publication")]


def _staging_left(*roots: Path) -> list[str]:
    return sorted(
        os.fspath(path) for root in roots if root.exists()
        for path in root.rglob("*") if FRAME_STAGING_DIRECTORY in path.parts
    )


def _state_after(steps: int, seed: int) -> dict[str, bytes]:
    reference = CpuCartesianRuntime(_config(), seed=seed)
    for _ in range(steps):
        reference.step()
    return reference.raw_state()


def _resume(directory: Path, publisher: Path, into: Path, seed: int) -> CpuCartesianRuntime:
    resumed = CpuCartesianRuntime(_config(), seed=seed)
    resumed.restore_restart(read_cartesian_restart(directory, publisher=publisher, into=into))
    return resumed


def _tree(root: Path) -> dict[str, bytes]:
    prefix = os.fspath(root.resolve()).encode()
    return {
        os.fspath(path.relative_to(root)): path.read_bytes().replace(prefix, b"<ROOT>")
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _integrate(runtime, root: Path, publisher: Path, *, duration=4.0, history=1.0,
               restart=2.0, restart_publisher=None, **keywords):
    """dt is 0.5 s: frames at steps 2, 4, 6, 8 and restarts at steps 4 and 8."""

    return runtime.integrate(
        duration_s=duration, history_interval_s=history,
        history_root=root / "frames", history_publisher=publisher,
        restart_root=root / "restarts", restart_interval_s=restart,
        restart_publisher=publisher if restart_publisher is None else restart_publisher,
        **keywords,
    )


# ------------------------------------------------------------ the mechanism


def test_the_worker_is_queue_depth_one_and_raises_its_error_at_the_join() -> None:
    gate = threading.Event()
    order: list[str] = []
    discarded: list[str] = []

    def part(label, action):
        return PublicationPart(label=label, commit=action, discard=lambda: discarded.append(label))

    def slow(cancel):
        gate.wait(10)
        order.append("frame")
        return "frame-receipt"

    def broken(cancel):
        order.append("restart")
        raise OSError(28, "No space left on device")

    def never(cancel):
        order.append("never")

    publisher = BackgroundPublisher()
    event = EventPublication([part("frame", slow), part("restart", broken), part("after", never)])
    publisher.submit(event)
    with pytest.raises(RuntimeError, match="queue depth is one"):
        publisher.submit(EventPublication([part("next", never)]))
    gate.set()
    with pytest.raises(OSError, match="No space left"):
        publisher.join()
    # In order, stopping at the first failure, whose staging alone is discarded.
    assert order == ["frame", "restart"]
    assert discarded == ["restart"]
    assert [p.label for p in event.completed] == ["frame"]
    assert event.parts[0].result == "frame-receipt"
    assert publisher.in_flight is None and not _publication_threads()
    assert publisher.join() is None
    # The join wait is what the stepping thread paid, attributed per part.
    assert event.join_finished >= event.join_started
    assert 0.0 <= blocking_overlap(event.parts[0], event) <= event.join_finished - event.join_started


def test_abort_kills_the_publisher_group_and_discards_the_staging(tmp_path) -> None:
    pid_file = tmp_path / "pid"
    hung = tmp_path / "hung-publisher"
    hung.write_text(f"#!/bin/bash\necho $$ > {pid_file}\nexec sleep 60\n")
    hung.chmod(0o755)
    discarded = []

    def commit(cancel):
        return _run_checked([hung], "hung publication", cancel=cancel)

    publisher = BackgroundPublisher()
    event = EventPublication([
        PublicationPart("restart", commit, discard=lambda: discarded.append("restart"))
    ])
    publisher.submit(event)
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text().strip():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    started = time.monotonic()
    settled = publisher.abort()
    assert time.monotonic() - started < 5.0
    assert settled is event and event.aborted
    assert isinstance(event.error, PublicationCancelled)
    assert discarded == ["restart"] and not _publication_threads()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


# ------------------------------------------- a frame and a restart, one step


def test_parts_that_await_their_turn_are_committed_beside_each_other_and_renamed_in_order() -> None:
    """The restart starts while the frame is still committing, and still renames after it.

    The ordered worker would start the restart only once the frame had
    returned, and the frame here returns only once the restart has started.
    """

    restart_started = threading.Event()
    renamed: list[str] = []

    def frame(cancel):
        assert restart_started.wait(10), "the restart was not started beside the frame"
        time.sleep(0.4)  # still writing after the restart is ready to rename
        cancel.await_turn()
        renamed.append("frame")
        return "frame-receipt"

    def restart(cancel):
        restart_started.set()
        cancel.await_turn()
        renamed.append("restart")
        return "restart-receipt"

    publisher = BackgroundPublisher()
    event = EventPublication([
        PublicationPart("frame", frame, discard=lambda: None, awaits_turn=True),
        PublicationPart("restart", restart, discard=lambda: None, awaits_turn=True),
    ])
    publisher.submit(event)
    assert publisher.join() is event and event.error is None
    assert renamed == ["frame", "restart"]
    assert [part.result for part in event.completed] == ["frame-receipt", "restart-receipt"]
    frame_part, restart_part = event.parts
    assert restart_part.started < frame_part.finished
    assert frame_part.turn_wait_seconds is None
    assert restart_part.turn_wait_seconds >= 0.3
    assert publisher.in_flight is None and not _publication_threads()
    # One join waited for both: each of its seconds is counted once.
    waited = event.join_finished - event.join_started
    attributed = [blocking_overlap(part, event) for part in event.parts]
    assert sum(attributed) <= waited + 1e-9
    # The earliest part keeps its whole overlap with the join; the restart
    # keeps only the seconds of it when the frame was not committing.

    def overlap(low, high):
        return max(0.0, min(high, event.join_finished) - max(low, event.join_started))

    assert attributed[0] == pytest.approx(overlap(frame_part.started, frame_part.finished), abs=1e-9)
    assert attributed[1] == pytest.approx(
        overlap(restart_part.started, restart_part.finished)
        - overlap(max(restart_part.started, frame_part.started),
                  min(restart_part.finished, frame_part.finished)),
        abs=1e-9,
    )
    assert attributed[1] < attributed[0]


def test_a_failed_earlier_part_stops_the_part_beside_it_and_a_later_failure_spares_the_earlier() -> None:
    discarded: list[str] = []

    def part(label, commit):
        return PublicationPart(
            label, commit, discard=lambda: discarded.append(label), awaits_turn=True,
        )

    # The frame fails while the restart is still writing: the restart stops at
    # its next write window -- not after its whole publication -- and is
    # discarded; the frame's error is the event's, with no note for the
    # restart it stopped.
    writing = threading.Event()

    def failing_frame(cancel):
        assert writing.wait(10)
        raise OSError(28, "No space left on device")

    def long_restart(cancel):
        writing.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            cancel.check()
            time.sleep(0.01)
        cancel.await_turn()
        return "published after a failed frame"

    publisher = BackgroundPublisher()
    event = EventPublication([part("frame", failing_frame), part("restart", long_restart)])
    began = time.monotonic()
    publisher.submit(event)
    with pytest.raises(OSError, match="No space left") as caught:
        publisher.join()
    assert time.monotonic() - began < 5.0
    assert not getattr(caught.value, "__notes__", [])
    assert sorted(discarded) == ["frame", "restart"] and event.completed == []
    assert event.parts[1].result is None and not _publication_threads()

    # A restart that was already waiting for its turn is not renamed either.
    discarded.clear()
    waiting = threading.Event()

    def later_failing_frame(cancel):
        assert waiting.wait(10)
        time.sleep(0.1)
        raise OSError(5, "Input/output error")

    def turn_restart(cancel):
        waiting.set()
        with pytest.raises(EarlierPartFailed, match="frame, earlier in this event"):
            cancel.await_turn()
        raise EarlierPartFailed("re-raised for the worker")

    event = EventPublication([part("frame", later_failing_frame), part("restart", turn_restart)])
    publisher.submit(event)
    with pytest.raises(OSError, match="Input/output") as caught:
        publisher.join()
    assert not getattr(caught.value, "__notes__", [])
    assert sorted(discarded) == ["frame", "restart"] and event.completed == []

    # The restart fails first, while the frame still writes: the frame is
    # published, as it was committed first in the ordered run, and the
    # restart's error is the event's.
    discarded.clear()
    restart_failed = threading.Event()

    def surviving_frame(cancel):
        assert restart_failed.wait(10)
        time.sleep(0.1)
        cancel.await_turn()
        return "frame-receipt"

    def failing_restart(cancel):
        try:
            raise OSError(28, "No space left on device")
        finally:
            restart_failed.set()

    event = EventPublication([part("frame", surviving_frame), part("restart", failing_restart)])
    publisher.submit(event)
    with pytest.raises(OSError, match="No space left"):
        publisher.join()
    assert [p.label for p in event.completed] == ["frame"] and discarded == ["restart"]

    # Both fail on their own: the earlier error is raised, the later is noted.
    discarded.clear()

    def frame_after(cancel):
        assert restart_failed.wait(10)
        time.sleep(0.1)
        raise RuntimeError("frame failure")

    restart_failed.clear()
    event = EventPublication([part("frame", frame_after), part("restart", failing_restart)])
    publisher.submit(event)
    with pytest.raises(RuntimeError, match="frame failure") as caught:
        publisher.join()
    assert any("restart also failed" in note for note in caught.value.__notes__)
    assert sorted(discarded) == ["frame", "restart"] and not _publication_threads()


def test_a_part_that_never_awaits_its_turn_is_refused() -> None:
    """A commit that renames without waiting could land ahead of an earlier part."""

    publisher = BackgroundPublisher()
    discarded: list[str] = []
    event = EventPublication([
        PublicationPart("frame", lambda cancel: (time.sleep(0.2), cancel.await_turn()),
                        discard=lambda: None, awaits_turn=True),
        PublicationPart("restart", lambda cancel: "renamed without waiting",
                        discard=lambda: discarded.append("restart"), awaits_turn=True),
    ])
    publisher.submit(event)
    with pytest.raises(RuntimeError, match="never awaited its turn"):
        publisher.join()
    assert discarded == ["restart"] and [p.label for p in event.completed] == ["frame"]


def test_abort_kills_the_publisher_of_every_part_committed_beside_another(tmp_path) -> None:
    discarded: list[str] = []
    pids = {}

    def hung(label):
        pid_file = tmp_path / f"{label}.pid"
        script = tmp_path / f"hung-{label}"
        script.write_text(f"#!/bin/bash\necho $$ > {pid_file}\nexec sleep 60\n")
        script.chmod(0o755)
        pids[label] = pid_file

        def commit(cancel):
            _run_checked([script], f"hung {label}", cancel=cancel)
            cancel.await_turn()

        return PublicationPart(
            label, commit, discard=lambda: discarded.append(label), awaits_turn=True,
        )

    publisher = BackgroundPublisher()
    event = EventPublication([hung("frame"), hung("restart")])
    publisher.submit(event)
    deadline = time.monotonic() + 10
    while not all(p.exists() and p.read_text().strip() for p in pids.values()):
        assert time.monotonic() < deadline, "both publishers were not running at once"
        time.sleep(0.01)
    started = time.monotonic()
    settled = publisher.abort()
    assert time.monotonic() - started < 5.0
    assert settled is event and isinstance(event.error, PublicationCancelled)
    # The restart was cancelled with the frame; it did not fail on its own.
    assert not getattr(event.error, "__notes__", [])
    assert sorted(discarded) == ["frame", "restart"] and not _publication_threads()
    for pid_file in pids.values():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


def test_a_later_part_stopped_by_an_abort_adds_no_note_but_its_own_failure_does() -> None:
    """Only a later part's own failure is noted on the earliest part's error."""

    def part(label, commit):
        return PublicationPart(label, commit, discard=lambda: None, awaits_turn=True)

    def waiting(ready):
        def commit(cancel):
            ready.set()
            while True:
                cancel.check()
                time.sleep(0.01)
        return commit

    # The frame fails on its own, and the event is aborted before the
    # restart beside it reaches its next write window: the restart then
    # raises the abort's PublicationCancelled, which is not news.  The
    # restart's write windows are ``cancel.check()`` calls; it keeps writing
    # past the frame's failure (EarlierPartFailed) until the abort lands,
    # which ``check`` reports ahead of an earlier part's failure.
    frame_failed, restart_writing = threading.Event(), threading.Event()

    def failing_frame(cancel):
        assert restart_writing.wait(10)
        try:
            raise OSError(28, "No space left on device")
        finally:
            frame_failed.set()

    def restart_until_aborted(cancel):
        restart_writing.set()
        while True:
            try:
                cancel.check()
            except EarlierPartFailed:
                time.sleep(0.01)

    publisher = BackgroundPublisher()
    event = EventPublication([part("frame", failing_frame), part("restart", restart_until_aborted)])
    publisher.submit(event)
    assert frame_failed.wait(10)
    settled = publisher.abort()
    assert isinstance(settled.error, OSError) and not getattr(settled.error, "__notes__", [])

    # Both cancelled while writing: the frame's PublicationCancelled, no note.
    frame_writing, restart_writing = threading.Event(), threading.Event()
    event = EventPublication([part("frame", waiting(frame_writing)), part("restart", waiting(restart_writing))])
    publisher.submit(event)
    assert frame_writing.wait(10) and restart_writing.wait(10)
    settled = publisher.abort()
    assert type(settled.error) is PublicationCancelled
    assert not getattr(settled.error, "__notes__", []) and not _publication_threads()

    # The restart fails on its own and the abort then stops the frame: the
    # frame's cancellation is the event's error, and the restart's failure
    # is still reported on it.
    frame_writing, restart_failed = threading.Event(), threading.Event()

    def failing_restart(cancel):
        try:
            raise OSError(5, "Input/output error")
        finally:
            restart_failed.set()

    event = EventPublication([part("frame", waiting(frame_writing)), part("restart", failing_restart)])
    publisher.submit(event)
    assert frame_writing.wait(10) and restart_failed.wait(10)
    settled = publisher.abort()
    assert type(settled.error) is PublicationCancelled
    assert any("restart also failed: OSError" in note for note in settled.error.__notes__)


def test_the_worker_commits_the_parts_in_turn_when_told_not_to_write_them_beside_each_other() -> None:
    """``concurrent_parts=False``: the restart starts only after the frame is renamed."""

    with pytest.raises(TypeError, match="concurrent_parts must be a bool"):
        BackgroundPublisher(concurrent_parts=0)
    log: list[str] = []

    def commit(label):
        def run(cancel):
            log.append(f"start {label}")
            time.sleep(0.1)
            cancel.await_turn()
            log.append(f"rename {label}")
            return label
        return run

    publisher = BackgroundPublisher(concurrent_parts=False)
    event = EventPublication([
        PublicationPart("frame", commit("frame"), discard=lambda: None, awaits_turn=True),
        PublicationPart("restart", commit("restart"), discard=lambda: None, awaits_turn=True),
    ])
    publisher.submit(event)
    assert publisher.join() is event and event.error is None
    assert log == ["start frame", "rename frame", "start restart", "rename restart"]
    assert event.parts[1].started >= event.parts[0].finished
    assert all(part.turn_wait_seconds is None for part in event.parts)
    assert publisher.record()["event_parts"] == "in-turn"
    assert publisher.record()["event_parts_note"].startswith("committed one after another")
    assert BackgroundPublisher().record()["event_parts"] == "beside"
    assert BackgroundPublisher().record()["event_parts_note"].startswith("committed beside each other")


def _timed_publisher(tmp_path: Path, real: Path, log: Path, body: str) -> Path:
    """A publisher that logs when each kind starts and ends, around ``body``."""

    script = tmp_path / "timed-publisher"
    script.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        if [ "$1" = "--abi" ]; then exec "{real}" --abi; fi
        echo "start $1 $(date +%s.%N)" >> "{log}"
        {body}
        "{real}" "$@"
        status=$?
        echo "end $1 $(date +%s.%N)" >> "{log}"
        exit $status
    """))
    script.chmod(0o755)
    return script


def _renames(monkeypatch) -> list[str]:
    renamed: list[str] = []
    replace = os.replace

    def recording(source, destination, *args, **kwargs):
        replace(source, destination, *args, **kwargs)
        if FRAME_STAGING_DIRECTORY in Path(source).parts:
            renamed.append(Path(destination).name)

    monkeypatch.setattr(runtime_module.os, "replace", recording)
    return renamed


def test_a_restart_on_a_frame_step_is_written_beside_the_frame_and_renamed_after_it(
    forecast_publisher, tmp_path, monkeypatch
) -> None:
    """One event, a frame and a restart: it takes its slowest part, not their sum.

    The frame's publisher sleeps ``FRAME_SLEEP`` and the restart's
    ``RESTART_SLEEP``.  Committed one after the other the run's last join
    would wait at least their sum, the frame's write included; beside each
    other it waits for the frame alone, and the restart, ready first, still
    waits for the frame's rename before its own.  Every wall-clock bound is
    the sum or the difference of the two sleeps: the publishers' own work on
    this tiny grid is milliseconds, and the bounds hold with over a second
    to spare on a loaded host.
    """

    FRAME_SLEEP, RESTART_SLEEP = 4.0, 1.5
    log = tmp_path / "publisher.log"
    timed = _timed_publisher(
        tmp_path, forecast_publisher, log,
        f'if [ "$1" = "history" ]; then sleep {FRAME_SLEEP}; else sleep {RESTART_SLEEP}; fi',
    )
    renamed = _renames(monkeypatch)
    runtime = CpuCartesianRuntime(_config(), seed=61)
    runtime.profiler = RuntimeProfiler(
        "summary", cupy=runtime.cp, stream=runtime.stream, progress_interval_s=0.0, nvml=False,
    )
    receipt = _integrate(
        runtime, tmp_path / "run", timed, duration=2.0, history=2.0, restart=2.0,
    )
    times = {}
    for line in log.read_text().splitlines():
        edge, kind, stamp = line.split()
        times[edge, kind] = float(stamp)
    # Beside: the restart's publisher started before the frame's had finished.
    assert times["start", "checkpoint"] < times["end", "history"]
    # In cm1.F's order on disk although the restart was ready first.
    assert renamed == ["frame-000001", "restart-000001"]
    assert times["end", "checkpoint"] < times["end", "history"]
    # The publishers themselves overlapped: the event spanned less than the
    # sum of their sleeps, on the publishers' own clock.
    span = max(times["end", "history"], times["end", "checkpoint"]) - min(
        times["start", "history"], times["start", "checkpoint"]
    )
    assert span < FRAME_SLEEP + RESTART_SLEEP
    publication = receipt["publication"]
    assert publication["events_submitted"] == 1
    assert publication["event_parts"] == "beside"
    # The join waited for the slower part, not the sum: one after the other
    # it could not have been shorter than both sleeps.
    assert publication["join_wait_seconds"] < FRAME_SLEEP + RESTART_SLEEP
    io = receipt["performance"]["io"]
    turn = {row["stage"]: row for row in io["restart"]["stages"]}["background.turn_wait"]
    # The restart, ready RESTART_SLEEP in, waited for the frame's rename at
    # FRAME_SLEEP; a second of the difference is left for the publishers' work.
    assert turn["seconds"] > FRAME_SLEEP - RESTART_SLEEP - 1.0
    assert "background.turn_wait" not in {row["stage"] for row in io["history"]["stages"]}
    # The restart's own work excludes what it waited for its turn: with the
    # wait counted it could not end before the frame's sleep ...
    assert io["restart"]["seconds"] < FRAME_SLEEP
    # ... and the one join is counted once across the two parts.
    blocked = io["history"]["blocking_seconds"] + io["restart"]["blocking_seconds"]
    assert blocked <= publication["join_wait_seconds"] + 0.5
    for kind, name, files in (("frames", "frame-000001", FRAME_FILES),
                              ("restarts", "restart-000001", RESTART_FILES)):
        assert sorted(p.name for p in (tmp_path / "run" / kind / name).iterdir()) == files
    assert not _staging_left(tmp_path / "run") and not _publication_threads()
    resumed = _resume(
        tmp_path / "run" / "restarts" / "restart-000001", forecast_publisher,
        tmp_path / "restored", seed=61,
    )
    assert resumed.raw_state() == _state_after(4, seed=61)


def test_a_frame_that_fails_stops_the_restart_of_its_step_even_when_the_restart_is_ready_first(
    forecast_publisher, tmp_path, monkeypatch
) -> None:
    """A restart never names a next frame index whose frame was not written."""

    log = tmp_path / "publisher.log"
    failing = _timed_publisher(
        tmp_path, forecast_publisher, log,
        'if [ "$1" = "history" ]; then sleep 3; echo "injected frame failure" >&2; exit 3; fi',
    )
    renamed = _renames(monkeypatch)
    runtime = CpuCartesianRuntime(_config(), seed=62)
    with pytest.raises(RuntimeError, match="forecast history publication failed") as caught:
        _integrate(runtime, tmp_path / "run", failing, duration=2.0, history=2.0, restart=2.0)
    assert not getattr(caught.value, "__notes__", None) or not any(
        "restart-000001 also failed" in note for note in caught.value.__notes__
    )
    starts = [line.split()[:2] for line in log.read_text().splitlines()]
    assert ["start", "checkpoint"] in starts and ["start", "history"] in starts
    assert renamed == []
    assert not (tmp_path / "run" / "restarts").exists()
    assert not (tmp_path / "run" / "frames").exists()
    assert not _staging_left(tmp_path / "run") and not _publication_threads()


def test_a_run_told_to_write_the_parts_in_turn_starts_the_restart_after_the_frame(
    forecast_publisher, tmp_path, monkeypatch
) -> None:
    """``integrate(concurrent_event_parts=False)``: the restart's publisher runs after the frame's."""

    log = tmp_path / "publisher.log"
    timed = _timed_publisher(
        tmp_path, forecast_publisher, log, 'if [ "$1" = "history" ]; then sleep 1; fi',
    )
    runtime = CpuCartesianRuntime(_config(), seed=63)
    with pytest.raises(TypeError, match="concurrent_event_parts must be a bool"):
        _integrate(runtime, tmp_path / "refused", timed, duration=2.0, history=2.0,
                   restart=2.0, concurrent_event_parts=1)
    assert not (tmp_path / "refused").exists()
    renamed = _renames(monkeypatch)
    receipt = _integrate(
        runtime, tmp_path / "run", timed, duration=2.0, history=2.0, restart=2.0,
        concurrent_event_parts=False,
    )
    times = {}
    for line in log.read_text().splitlines():
        edge, kind, stamp = line.split()
        times[edge, kind] = float(stamp)
    assert times["start", "checkpoint"] > times["end", "history"]
    assert renamed == ["frame-000001", "restart-000001"]
    assert receipt["publication"]["event_parts"] == "in-turn"
    assert receipt["publication"]["event_parts_note"].startswith("committed one after another")
    assert not _staging_left(tmp_path / "run") and not _publication_threads()
    resumed = _resume(
        tmp_path / "run" / "restarts" / "restart-000001", forecast_publisher,
        tmp_path / "restored", seed=63,
    )
    assert resumed.raw_state() == _state_after(4, seed=63)


def test_the_door_admits_event_parts_by_value_and_the_command_refuses_a_bad_one(
    tmp_path, capsys
) -> None:
    """``cm1core cartesian --event-parts {beside,in-turn}`` is how an operator opts out."""

    from cm1core.cartesian_door import require_event_parts
    from cm1core.cli import main
    from cm1core.errors import CM1Refusal

    assert require_event_parts("beside") is True
    assert require_event_parts("in-turn") is False
    for value in ("sequential", "", "Beside", None, True, 1):
        with pytest.raises(CM1Refusal) as caught:
            require_event_parts(value)
        assert caught.value.detail.code == "CM1-CARTESIAN-EVENT-PARTS"
        assert caught.value.detail.option == "event-parts"
        assert "'beside'" in caught.value.detail.remedy and "'in-turn'" in caught.value.detail.remedy
    output = tmp_path / "run.json"
    status = main([
        "cartesian", str(ROOT / "examples" / "cartesian-uniform-grid.toml"),
        "--output", str(output), "--integrate", "--event-parts", "sequential",
    ])
    assert status == 2
    assert "--event-parts 'sequential'" in capsys.readouterr().err
    assert not output.exists()


def test_the_publication_benchmark_times_a_restart_alone_and_on_a_frame_step(
    forecast_publisher, tmp_path
) -> None:
    """``tools/benchmark_publication_io.py --kind same-step`` on tiny payloads."""

    envelopes = {}
    for parts in ("concurrent", "sequential"):
        # The history publisher sleeps, so the restart beside it starts
        # before it has finished, and in turn only after.
        slow = _slow(tmp_path, forecast_publisher, 0.5, kind="history")
        workdir = tmp_path / f"bench-{parts}"
        completed = subprocess.run(
            [sys.executable, os.fspath(ROOT / "tools" / "benchmark_publication_io.py"),
             "--mode", "background", "--kind", "same-step", "--gib", "0.002", "--carriers", "3",
             "--restart-gib", "0.004", "--restart-carriers", "4", "--next-event-s", "0",
             "--parts", parts,
             "--workdir", os.fspath(workdir), "--publisher", os.fspath(slow)],
            capture_output=True, text=True, check=False,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["schema"] == "gm1.publication-io-benchmark/v4"
        assert result["next_event_seconds"] == 0.0 and result["parts"] == parts
        alone, same = result["scenarios"]["restart_only"], result["scenarios"]["same_step"]
        assert set(alone["parts"]) == {"checkpoint"} and set(same["parts"]) == {"history", "checkpoint"}
        assert alone["parts_committed_beside_each_other"] is False
        assert same["parts_committed_beside_each_other"] is (parts == "concurrent")
        assert same["event_parts"] == ("beside" if parts == "concurrent" else "in-turn")
        assert result["restart_envelope_identical_across_scenarios"] is True
        envelopes[parts] = same["parts"]["checkpoint"]["envelope_sha256"]
        for scenario in (alone, same):
            assert scenario["stepping_thread_blocked_seconds"] == pytest.approx(
                scenario["snapshot_seconds"] + scenario["join_wait_seconds"]
            )
            memory = scenario["mem_available"]
            assert memory["samples"] >= 2 and 0 < memory["lowest_bytes"] <= memory["at_submit_bytes"]
            assert memory["drop_from_submit_bytes"] == memory["at_submit_bytes"] - memory["lowest_bytes"]
        assert same["parts"]["checkpoint"]["raw_arena_retained"] is False
        assert same["parts"]["history"]["raw_arena_retained"] is True
        assert result["frame_adds_to_restart_blocked_seconds"] == pytest.approx(
            same["stepping_thread_blocked_seconds"] - alone["stepping_thread_blocked_seconds"]
        )
        assert result["frame_adds_to_restart_mem_available_drop_bytes"] == (
            same["mem_available"]["drop_from_submit_bytes"]
            - alone["mem_available"]["drop_from_submit_bytes"]
        )
        assert not workdir.exists()
    assert envelopes["concurrent"] == envelopes["sequential"]


# ------------------------------------------------------- bytes and resumes


def test_background_and_synchronous_runs_publish_the_same_bytes(
    forecast_publisher, tmp_path
) -> None:
    """Six frames and three restarts, one host arena of each kind reused throughout.

    Both runs publish under the same root, one after the other, because the
    field tables name the arena by absolute path and the receipts hash those
    tables: equal bytes then means every byte, receipts included.
    """

    trees = {}
    states = {}
    for mode, keywords in (
        ("background", {}),
        # The worker told to commit a frame and its step's restart in turn.
        ("in-turn", {"concurrent_event_parts": False}),
        ("synchronous", {"background_publication": False}),
    ):
        root = tmp_path / "run"
        runtime = CpuCartesianRuntime(_config(u_move_m_s=12.5, v_move_m_s=3.0), seed=31)
        receipt = _integrate(
            runtime, root, forecast_publisher, duration=3.0, history=0.5, restart=1.0,
            **keywords,
        )
        assert receipt["publication"]["mode"] == ("synchronous" if mode == "synchronous" else "background")
        assert [row["index"] for row in receipt["frames"]] == [1, 2, 3, 4, 5, 6]
        assert [row["number"] for row in receipt["restarts"]] == [1, 2, 3]
        assert receipt["last_restart_publication"]["step"] == 6
        trees[mode] = _tree(root)
        states[mode] = runtime.raw_state()
        for number in (1, 2, 3):
            directory = root / "restarts" / RESTART_DIRECTORY_FORMAT.format(number=number)
            assert sorted(p.name for p in directory.iterdir()) == RESTART_FILES
        for index in range(1, 7):
            directory = root / "frames" / FRAME_DIRECTORY_FORMAT.format(index=index)
            # A history frame keeps its raw arena: every frame reader maps it.
            assert sorted(p.name for p in directory.iterdir()) == FRAME_FILES
        assert not _staging_left(root)
        root.rename(tmp_path / mode)
    for mode in ("in-turn", "synchronous"):
        assert states[mode] == states["background"]
        assert trees[mode].keys() == trees["background"].keys()
        differing = [name for name in trees[mode] if trees[mode][name] != trees["background"][name]]
        assert differing == [], mode


def _nested(seed: int) -> CpuCartesianRuntime:
    """A parent double carrying the shipped NestDriver publication over a child double.

    The child steps in lockstep with the parent (a time ratio of one), standing
    in for the child steps the real driver takes inside each parent step, so
    every child frame and child restart must be the child's state at its parent
    event's step.
    """

    from types import SimpleNamespace

    import numpy as np

    from cm1core.nest_geometry import NestGeometry
    from cm1core.nest_live import NestDriver

    geometry = NestGeometry(
        parent_nx=9, parent_ny=11, parent_dx_m=500.0, parent_dy_m=500.0,
        refinement_ratio=3, time_refinement_ratio=1, parent_i0=3, parent_j0=4,
        parent_span_x=3, parent_span_y=4, spec_zone=1, relax_zone=2, feedback=1,
    )
    parent = CpuCartesianRuntime(_config(), seed=seed)
    child = CpuCartesianRuntime(_config(nx=geometry.nx, ny=geometry.ny), seed=seed + 1)
    driver = object.__new__(NestDriver)
    driver.parent, driver.child, driver.geometry = parent, child, geometry
    driver.cp, driver.parcels, driver.child_frames = np, None, []
    driver.stream, driver.mover = parent.stream, None
    driver.parent_steps = driver.child_steps = driver.feedback_writes = 0
    driver.forcing = SimpleNamespace(
        refreshes=0, held_relaxations=0, acoustic_small_steps=0, scalar_specifications=0,
    )
    parent.nest_driver = driver
    stepping = parent.step

    def step():
        stepping()
        child.step()

    parent.step = step
    return parent


def test_a_nest_child_frame_is_taken_with_its_parent_frame_and_published_with_it(
    forecast_publisher, tmp_path
) -> None:
    trees = {}
    for background in (True, False):
        root = tmp_path / "run"
        parent = _nested(seed=21)
        receipt = _integrate(
            parent, root, forecast_publisher, background_publication=background,
        )
        child_rows = parent.nest_driver.child_frames
        assert [row["index"] for row in child_rows] == [1, 2, 3, 4]
        assert [row["step"] for row in child_rows] == [2, 4, 6, 8]
        assert [row["step"] for row in receipt["frames"]] == [2, 4, 6, 8]
        child_root = root / "frames" / "nest-child"
        for row in child_rows:
            directory = Path(row["directory"])
            assert directory.parent == child_root
            assert sorted(p.name for p in directory.iterdir()) == FRAME_FILES
            placement = json.loads(Path(row["placement_path"]).read_text())
            assert (placement["index"], placement["step"]) == (row["index"], row["step"])
        assert receipt["last_frame_publication"]["nest_child_frame"] == child_rows[-1]
        # The child's own restart is published inside each parent restart
        # (cm1core.nest_restart), in the same event and the same rename.
        for number, step in ((1, 4), (2, 8)):
            restart = root / "restarts" / f"restart-{number:06d}"
            assert sorted(p.name for p in restart.iterdir()) == sorted(RESTART_FILES + ["nest-child"])
            assert sorted(p.name for p in (restart / "nest-child").iterdir()) == RESTART_FILES
            declared = json.loads((restart / "restart-publication.json").read_text())["nest_child"]
            child_receipt = json.loads((restart / "nest-child" / "restart-publication.json").read_text())
            assert (declared["parent_step"], declared["child_step"], child_receipt["step"]) == (step, step, step)
            assert declared["profile_sha256"] == child_receipt["profile_sha256"]
        assert receipt["last_restart_publication"]["nest_child_restart"]["step"] == 8
        assert not _staging_left(root) and not _publication_threads()
        trees[background] = _tree(root)
        root.rename(tmp_path / ("background" if background else "synchronous"))
    assert trees[True] == trees[False]


def test_a_run_resumed_from_a_background_restart_is_bit_identical_to_the_continuous_run(
    forecast_publisher, tmp_path
) -> None:
    whole = CpuCartesianRuntime(_config(), seed=41)
    _integrate(whole, tmp_path / "whole", forecast_publisher, duration=3.0, history=0.5, restart=1.5)

    stopped = CpuCartesianRuntime(_config(), seed=41)
    _integrate(stopped, tmp_path / "part1", forecast_publisher, duration=1.5, history=0.5, restart=1.5)
    restart = tmp_path / "part1" / "restarts" / "restart-000001"
    assert sorted(p.name for p in restart.iterdir()) == RESTART_FILES
    assert not (restart / RESTART_ARENA_FILENAME).exists()

    resumed = _resume(restart, forecast_publisher, tmp_path / "restored", seed=41)
    receipt = _integrate(
        resumed, tmp_path / "part2", forecast_publisher, duration=3.0, history=0.5,
        restart=1.5, first_history_index=4, first_restart_number=2,
    )
    assert [row["index"] for row in receipt["frames"]] == [4, 5, 6]
    assert resumed.raw_state() == whole.raw_state()
    for index in (4, 5, 6):
        name = FRAME_DIRECTORY_FORMAT.format(index=index)
        for file in ("history-carriers.f32", "history.cm1fhs"):
            assert (tmp_path / "part2" / "frames" / name / file).read_bytes() == (
                tmp_path / "whole" / "frames" / name / file
            ).read_bytes(), (name, file)
    assert (tmp_path / "part2" / "restarts" / "restart-000002" / "restart.cm1fcp").read_bytes() == (
        tmp_path / "whole" / "restarts" / "restart-000002" / "restart.cm1fcp"
    ).read_bytes()


def test_the_stepping_thread_pays_for_the_snapshot_not_the_publication(
    forecast_publisher, tmp_path
) -> None:
    """A checkpoint that takes a second to publish costs the steps almost nothing.

    Each step sleeps, standing in for device work, so the next event is
    reached after the background publication has finished; the synchronous
    run pays the whole publication inside every event instead.
    """

    slow = _slow(tmp_path, forecast_publisher, 1.0)
    rows = {}
    for background in (True, False):
        runtime = CpuCartesianRuntime(_config(), seed=7)
        runtime.profiler = RuntimeProfiler(
            "summary", cupy=runtime.cp, stream=runtime.stream, progress_interval_s=0.0, nvml=False,
        )
        stepping = runtime.step

        def step(stepping=stepping):
            stepping()
            time.sleep(0.75)

        runtime.step = step
        started = time.perf_counter()
        receipt = _integrate(
            runtime, tmp_path / f"blocking-{background}", forecast_publisher,
            duration=4.0, history=2.0, restart=2.0, restart_publisher=slow,
            background_publication=background,
        )
        rows[background] = (receipt["performance"]["io"]["restart"], time.perf_counter() - started)
    background_io, background_wall = rows[True]
    synchronous_io, synchronous_wall = rows[False]
    assert background_io["events"] == synchronous_io["events"] == 2
    assert background_io["seconds"] > 2.0 and synchronous_io["seconds"] > 2.0
    # Synchronous: the stepping thread is blocked for the whole publication.
    assert synchronous_io["blocking_seconds"] == pytest.approx(synchronous_io["seconds"])
    # Background: only the last event, which the run end joins, is waited for.
    assert background_io["blocking_seconds"] < background_io["seconds"] - 0.8
    assert background_wall < synchronous_wall - 0.8


def test_progress_and_the_receipt_name_only_directories_already_complete(
    forecast_publisher, tmp_path
) -> None:
    """Nothing is reported before it is on disk, and a finished event is not held back.

    ``progress`` runs on the stepping thread; at each call the frame it names,
    and the restart when the row says one was written, must already be the
    complete published directory.  Steps are slowed so each publication
    finishes within the step after its event, and at least one is reported
    there -- at the first step boundary after it is durable -- rather than at
    the next event.
    """

    runtime = CpuCartesianRuntime(_config(), seed=13)
    stepping = runtime.step

    def step():
        stepping()
        time.sleep(0.3)

    runtime.step = step
    seen = []

    def progress(row):
        frame = Path(row["directory"])
        restart = None
        if "restart_written" in row:
            restart = tmp_path / "restarts" / RESTART_DIRECTORY_FORMAT.format(
                number=row["restart_written"]
            )
        seen.append((
            dict(row), int(runtime.clock.completed_steps),
            sorted(p.name for p in frame.iterdir()),
            None if restart is None else sorted(p.name for p in restart.iterdir()),
        ))

    receipt = _integrate(runtime, tmp_path, forecast_publisher, progress=progress)
    assert [row["index"] for row, _, _, _ in seen] == [1, 2, 3, 4]
    assert all(files == FRAME_FILES for _, _, files, _ in seen)
    assert [row.get("restart_written") for row, _, _, _ in seen] == [None, 1, None, 2]
    assert all(files in (None, RESTART_FILES) for _, _, _, files in seen)
    # Each event is two steps after the last: reported before the next one.
    assert any(steps < row["step"] + 2 for row, steps, _, _ in seen[:-1])
    assert all(steps >= row["step"] for row, steps, _, _ in seen)
    assert receipt["last_restart_publication"]["step"] == 8
    assert receipt["publication"]["events_submitted"] == 4
    assert not _staging_left(tmp_path) and not _publication_threads()


def test_a_checkpoint_that_does_not_hold_the_arena_words_is_refused(
    forecast_publisher, tmp_path, monkeypatch
) -> None:
    """The raw arena is deleted only when the envelope header proves every carrier."""

    runtime = CpuCartesianRuntime(_config(), seed=12)
    runtime.step()
    receipt = write_cartesian_restart(runtime, tmp_path / "restart", publisher=forecast_publisher)
    envelope = tmp_path / "restart" / "restart.cm1fcp"
    contract = receipt["forecast_envelope"]["contract_sha256"]
    recorded = receipt["arena"]
    arena = ForecastArenaWrite(
        tmp_path / "gone.f32", recorded["sha256"], int(recorded["nbytes"]),
        dict(recorded["carrier_sha256"]), dict(recorded["carrier_offset_bytes"]),
    )
    assert verify_restart_envelope_carries_arena(
        envelope, arena, contract_sha256=contract,
    ) == recorded["envelope_verification"]
    name = sorted(arena.carrier_sha256)[3]
    tampered = ForecastArenaWrite(
        arena.path, arena.sha256, arena.nbytes,
        {**arena.carrier_sha256, name: "0" * 64}, arena.carrier_offset_bytes,
    )
    with pytest.raises(RuntimeError, match=f"{name} has envelope SHA-256"):
        verify_restart_envelope_carries_arena(envelope, tampered, contract_sha256=contract)
    with pytest.raises(RuntimeError, match="contract"):
        verify_restart_envelope_carries_arena(envelope, arena, contract_sha256="1" * 64)
    truncated = tmp_path / "truncated.cm1fcp"
    truncated.write_bytes(envelope.read_bytes()[:-4])
    with pytest.raises(RuntimeError, match="prefix \\+ header \\+ payload"):
        verify_restart_envelope_carries_arena(truncated, arena, contract_sha256=contract)
    history = tmp_path / "not-a-checkpoint"
    history.write_bytes(b"CM1FHS1\0" + bytes(64))
    with pytest.raises(RuntimeError, match="CM1FCP1 magic"):
        verify_restart_envelope_carries_arena(history, arena, contract_sha256=contract)

    # In a run, a failed verification fails the run and publishes no restart.
    def refuse(envelope, arena, *, contract_sha256):
        raise RuntimeError("injected verification failure")

    monkeypatch.setattr(restart_module, "verify_restart_envelope_carries_arena", refuse)
    run = CpuCartesianRuntime(_config(), seed=12)
    with pytest.raises(RuntimeError, match="injected verification failure"):
        _integrate(run, tmp_path / "run", forecast_publisher)
    assert not (tmp_path / "run" / "restarts").exists()
    frames = sorted((tmp_path / "run" / "frames").iterdir())
    assert [p.name for p in frames] == ["frame-000001", "frame-000002"]
    assert not _staging_left(tmp_path / "run") and not _publication_threads()


# ---------------------------------------------------------- failure injection


def test_a_worker_exception_is_raised_at_the_next_join_and_publishes_nothing_partial(
    forecast_publisher, tmp_path, monkeypatch
) -> None:
    real = runtime_module.commit_cartesian_restart

    def broken(snapshot, *, io=None, cancel=None):
        snapshot.directory.mkdir(parents=True)
        (snapshot.directory / "restart-carriers.f32").write_bytes(b"\0" * 64)
        raise RuntimeError("injected worker failure")

    monkeypatch.setattr(runtime_module, "commit_cartesian_restart", broken)
    runtime = CpuCartesianRuntime(_config(), seed=3)
    with pytest.raises(RuntimeError, match="injected worker failure"):
        _integrate(runtime, tmp_path, forecast_publisher)
    # Raised on the stepping thread at the first step boundary after the
    # worker failed -- step 5, or the join of the next event at step 6 -- not
    # where it happened: stepping went on while the worker ran.
    assert runtime.clock.completed_steps in (5, 6)
    frames = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert frames == ["frame-000001", "frame-000002"]
    # No restart was published, so the empty root is removed as well, exactly
    # as a refused run removes it, and a retry at the same path is admitted.
    assert not (tmp_path / "restarts").exists()
    assert not _staging_left(tmp_path) and not _publication_threads()
    assert real is not broken


def test_a_full_disk_while_the_worker_writes_the_restart_arena(
    forecast_publisher, tmp_path
) -> None:
    """The kernel refuses the arena write (RLIMIT_FSIZE); the run fails cleanly.

    The 262 KiB restart arena crosses the limit, the 57 KiB history arenas and
    envelopes do not, so the failure is the arena writer's own ``write`` on
    the worker thread.
    """

    runtime = CpuCartesianRuntime(_config(), seed=4)
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (150_000, hard))
    try:
        with pytest.raises(OSError) as caught:
            _integrate(runtime, tmp_path, forecast_publisher)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
    assert "File too large" in str(caught.value) or caught.value.errno in (27, 28)
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["frame-000001", "frame-000002"]
    # No restart was published, so the empty root is removed as well, exactly
    # as a refused run removes it, and a retry at the same path is admitted.
    assert not (tmp_path / "restarts").exists()
    assert not _staging_left(tmp_path) and not _publication_threads()


def test_a_full_disk_inside_the_publisher_fails_the_run_and_leaves_no_staging(
    forecast_publisher, tmp_path
) -> None:
    """The real publisher, under a file-size limit its envelope write crosses."""

    full = _wrapper(
        tmp_path, forecast_publisher, "full-disk-publisher",
        'if [ "$1" = "checkpoint" ]; then trap "" XFSZ; ulimit -f 128; fi',
    )
    runtime = CpuCartesianRuntime(_config(), seed=4)
    with pytest.raises(RuntimeError, match="forecast checkpoint publication failed"):
        _integrate(
            runtime, tmp_path, forecast_publisher, restart_publisher=full,
        )
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["frame-000001", "frame-000002"]
    # No restart was published, so the empty root is removed as well, exactly
    # as a refused run removes it, and a retry at the same path is admitted.
    assert not (tmp_path / "restarts").exists()
    assert not _staging_left(tmp_path) and not _publication_threads()


def _interrupt_after(runtime, step: int, action) -> None:
    stepping = runtime.step

    def interrupted():
        stepping()
        if runtime.clock.completed_steps == step:
            action()

    runtime.step = interrupted


def test_sigint_during_a_background_restart_finishes_it_then_stops(
    forecast_publisher, tmp_path, capsys
) -> None:
    """The restart in flight is a completed state: it is finished, then the run ends.

    The wait is announced first, naming what it waits for: without a word the
    operator's next Ctrl-C is the one that discards the restart.
    """

    slow = _slow(tmp_path, forecast_publisher, 1.0)
    runtime = CpuCartesianRuntime(_config(), seed=5)
    # Event two (step 4) writes a frame and a restart; the SIGINT arrives on
    # the first step after it, while the publisher is still sleeping.
    _interrupt_after(runtime, 5, lambda: signal.raise_signal(signal.SIGINT))
    with pytest.raises(KeyboardInterrupt) as caught:
        _integrate(runtime, tmp_path, forecast_publisher, restart_publisher=slow)
    assert any("completed before the run ended" in note for note in caught.value.__notes__)
    assert (
        "gm1: waiting for frame-000002, restart-000001 to finish publishing "
        "(Ctrl-C again cancels it and discards its staging)"
    ) in capsys.readouterr().err.splitlines()
    restart = tmp_path / "restarts" / "restart-000001"
    assert sorted(p.name for p in restart.iterdir()) == RESTART_FILES
    assert json.loads((restart / "restart-publication.json").read_text())["step"] == 4
    resumed = _resume(restart, forecast_publisher, tmp_path / "restored", seed=5)
    assert resumed.raw_state() == _state_after(4, seed=5)
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["frame-000001", "frame-000002"]
    assert not _staging_left(tmp_path) and not _publication_threads()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_a_second_sigint_cancels_the_restart_in_flight_and_discards_its_staging(
    forecast_publisher, tmp_path, capsys
) -> None:
    slow = _slow(tmp_path, forecast_publisher, 30.0)
    runtime = CpuCartesianRuntime(_config(), seed=6)

    def twice():
        threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT)).start()
        signal.raise_signal(signal.SIGINT)

    _interrupt_after(runtime, 5, twice)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt) as caught:
        _integrate(runtime, tmp_path, forecast_publisher, restart_publisher=slow)
    # Cancelled, not waited out: the wrapper would have slept thirty seconds.
    assert time.monotonic() - started < 15.0
    assert any("was cancelled" in note for note in caught.value.__notes__), caught.value.__notes__
    err = capsys.readouterr().err.splitlines()
    assert any(line.startswith("gm1: waiting for ") for line in err), err
    assert "gm1: cancelling the publication of frame-000002, restart-000001; its staging is discarded" in err
    # No restart was published, so the empty root is removed as well, exactly
    # as a refused run removes it, and a retry at the same path is admitted.
    assert not (tmp_path / "restarts").exists()
    # The first interrupt waited for the event in flight, so its frame -- the
    # part committed before the restart -- was finished; the second cancelled
    # the restart.
    frames = sorted((tmp_path / "frames").iterdir())
    assert [p.name for p in frames] == ["frame-000001", "frame-000002"]
    assert all(sorted(p.name for p in f.iterdir()) == FRAME_FILES for f in frames)
    assert not _staging_left(tmp_path) and not _publication_threads()


def test_sigterm_cancels_the_publication_in_flight(forecast_publisher, tmp_path, capsys) -> None:
    slow = _slow(tmp_path, forecast_publisher, 30.0)
    runtime = CpuCartesianRuntime(_config(), seed=6)
    _interrupt_after(runtime, 5, lambda: signal.raise_signal(signal.SIGTERM))
    started = time.monotonic()
    with pytest.raises(TerminationRequested) as caught:
        _integrate(runtime, tmp_path, forecast_publisher, restart_publisher=slow)
    assert caught.value.code == 143
    assert time.monotonic() - started < 15.0
    err = capsys.readouterr().err.splitlines()
    assert (
        "gm1: SIGTERM: cancelling the publication of frame-000002, restart-000001; "
        "its staging is discarded"
    ) in err, err
    assert not any(line.startswith("gm1: waiting for ") for line in err), err
    # No restart was published, so the empty root is removed as well, exactly
    # as a refused run removes it, and a retry at the same path is admitted.
    assert not (tmp_path / "restarts").exists()
    # Frame two shares the cancelled event: it is either finished or absent,
    # never a directory under its published name that is still being written.
    frames = sorted((tmp_path / "frames").iterdir())
    assert [p.name for p in frames] in (["frame-000001"], ["frame-000001", "frame-000002"])
    assert all(sorted(p.name for p in f.iterdir()) == FRAME_FILES for f in frames)
    assert not _staging_left(tmp_path) and not _publication_threads()
    # The handler was the integration's own, and it is gone with it.
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


_CHILD = """
import sys, time
from pathlib import Path
sys.path[:0] = [{src!r}, {tests!r}]
from test_cartesian_restart import CpuCartesianRuntime, _config
runtime = CpuCartesianRuntime(_config(), seed=8)
stepping = runtime.step
def step():
    stepping()
    time.sleep(0.05)
runtime.step = step
root = Path({root!r})
runtime.integrate(
    duration_s=60.0, history_interval_s=1.0, history_root=root / "frames",
    history_publisher={publisher!r}, restart_root=root / "restarts",
    restart_interval_s=2.0, restart_publisher={slow!r},
)
"""


def test_sigint_from_outside_the_process_leaves_only_complete_directories(
    forecast_publisher, tmp_path
) -> None:
    """A real SIGINT, sent by another process while a restart is being written."""

    slow = _slow(tmp_path, forecast_publisher, 1.5)
    root = tmp_path / "run"
    script = tmp_path / "child.py"
    script.write_text(_CHILD.format(
        src=os.fspath(ROOT / "src"), tests=os.fspath(ROOT / "tests"), root=os.fspath(root),
        publisher=os.fspath(forecast_publisher), slow=os.fspath(slow),
    ))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    child = subprocess.Popen(
        [sys.executable, os.fspath(script)], env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    staged = root / "restarts" / FRAME_STAGING_DIRECTORY / "restart-000001"
    deadline = time.monotonic() + 60
    try:
        while not staged.exists():
            assert child.poll() is None, child.communicate()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        child.send_signal(signal.SIGINT)
        _, stderr = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    assert child.returncode != 0
    assert "KeyboardInterrupt" in stderr
    assert not _staging_left(root)
    restarts = sorted((root / "restarts").iterdir())
    assert [p.name for p in restarts] == ["restart-000001"]
    assert sorted(p.name for p in restarts[0].iterdir()) == RESTART_FILES
    resumed = _resume(restarts[0], forecast_publisher, tmp_path / "restored", seed=8)
    assert resumed.raw_state() == _state_after(4, seed=8)
    frames = sorted((root / "frames").iterdir())
    assert frames and all(sorted(p.name for p in f.iterdir()) == FRAME_FILES for f in frames)


# ------------------------------------------------------- review follow-ups


def test_one_call_publications_stream_the_device_words_and_hold_no_host_copy(
    forecast_publisher, tmp_path, monkeypatch
) -> None:
    """The synchronous writers keep the release head's host memory: no payload copy.

    ``_publish_frame``, ``_publish_restart``, ``write_cartesian_restart`` and a
    nest child's ``publish_child_frame`` commit at once on the calling thread,
    so their words are streamed from the device in transfer chunks as they
    always were; a full-size host arena would sit beside the publisher's own
    copy of the payload.  None is reserved, and the files are the ones
    ``integrate`` publishes from its host arenas.
    """

    from cm1core.cartesian_history import cartesian_history_profile
    from cm1core.cartesian_restart import cartesian_restart_profile
    from cm1core.forecast_arena import HostArena

    # One root for both, as in the background/synchronous comparison: the
    # field tables name their arena by absolute path.
    direct = tmp_path / "run"
    runtime = CpuCartesianRuntime(_config(), seed=51)
    _integrate(
        runtime, direct, forecast_publisher, duration=1.0, history=1.0, restart=1.0,
        background_publication=False,
    )
    expected = _tree(direct)
    reference = direct.rename(tmp_path / "integrated")

    def refuse(self, cupy, nbytes):
        raise AssertionError(f"a one-call publication reserved a {nbytes}-byte host copy")

    monkeypatch.setattr(HostArena, "reserve", refuse)
    runtime = CpuCartesianRuntime(_config(), seed=51)
    runtime.step()
    runtime.step()
    (direct / "frames").mkdir(parents=True)
    (direct / "restarts").mkdir()
    runtime._publish_frame(
        direct / "frames", 1, publisher=forecast_publisher,
        profile=cartesian_history_profile(runtime.config),
    )
    runtime._publish_restart(
        direct / "restarts", 1, publisher=forecast_publisher,
        profile=cartesian_restart_profile(runtime.config), next_history_index=2,
    )
    for home in (direct / "frames", direct / "restarts"):
        (home / FRAME_STAGING_DIRECTORY).rmdir()
    assert _tree(direct) == expected

    # write_cartesian_restart: straight into its directory, the same envelope.
    receipt = write_cartesian_restart(
        runtime, tmp_path / "in-place", publisher=forecast_publisher,
        restart_number=1, next_history_index=2,
    )
    assert (tmp_path / "in-place" / "restart.cm1fcp").read_bytes() == (
        reference / "restarts" / "restart-000001" / "restart.cm1fcp"
    ).read_bytes()
    assert receipt["arena"]["retained"] is False

    # A nest child's frame, published the one-call way after its parent.
    parent = _nested(seed=52)
    parent.step()
    (tmp_path / "nested").mkdir()
    receipt, _ = parent._publish_frame(
        tmp_path / "nested", 1, publisher=forecast_publisher,
        profile=cartesian_history_profile(parent.config),
    )
    child = Path(receipt["nest_child_frame"]["directory"])
    assert child == tmp_path / "nested" / "nest-child" / "frame-000001"
    assert sorted(p.name for p in child.iterdir()) == FRAME_FILES


def test_a_failed_nest_child_snapshot_still_publishes_its_parent_frame(
    forecast_publisher, tmp_path, monkeypatch
) -> None:
    """The parent frame is a completed state; the release head renamed it into place first.

    The child's snapshot fails at the second event.  In both modes the parent
    frame of that event is published, then the child's error ends the run.
    The one-call ``_publish_frame`` does the same.
    """

    from cm1core.cartesian_history import cartesian_history_profile
    from cm1core.nest_live import NestDriver

    real = NestDriver.snapshot_child_frame

    def failing(self, root, index, **keywords):
        if index == 2:
            raise RuntimeError("injected child snapshot failure")
        return real(self, root, index, **keywords)

    monkeypatch.setattr(NestDriver, "snapshot_child_frame", failing)
    for background in (True, False):
        root = tmp_path / ("background" if background else "synchronous")
        parent = _nested(seed=53)
        with pytest.raises(RuntimeError, match="injected child snapshot failure") as caught:
            _integrate(parent, root, forecast_publisher, background_publication=background)
        frames = sorted(p.name for p in (root / "frames").iterdir() if p.name.startswith("frame-"))
        assert frames == ["frame-000001", "frame-000002"], background
        assert all(
            sorted(p.name for p in (root / "frames" / name).iterdir()) == FRAME_FILES
            for name in frames
        )
        children = sorted(p.name for p in (root / "frames" / "nest-child").iterdir()
                          if p.name.startswith("frame-"))
        assert children == ["frame-000001"]
        assert any("frame-000002" in note and "were published" in note
                   for note in caught.value.__notes__), caught.value.__notes__
        assert not _staging_left(root) and not _publication_threads()

    parent = _nested(seed=54)
    parent.step()
    (tmp_path / "one-call").mkdir()
    for index in (1, 2):
        if index == 2:
            parent.step()
            with pytest.raises(RuntimeError, match="injected child snapshot failure"):
                parent._publish_frame(
                    tmp_path / "one-call", index, publisher=forecast_publisher,
                    profile=cartesian_history_profile(parent.config),
                )
        else:
            parent._publish_frame(
                tmp_path / "one-call", index, publisher=forecast_publisher,
                profile=cartesian_history_profile(parent.config),
            )
    assert sorted(p.name for p in (tmp_path / "one-call").iterdir() if p.name.startswith("frame-")) == [
        "frame-000001", "frame-000002",
    ]


def test_an_event_whose_worker_cannot_start_is_published_on_the_stepping_thread(
    forecast_publisher, tmp_path, monkeypatch
) -> None:
    """``Thread.start`` fails at the second event: its frame and restart still land."""

    real = threading.Thread.start
    started = []

    def start(self):
        if self.name.startswith("gm1-publication"):
            started.append(self.name)
            if len(started) == 2:
                raise RuntimeError("can't start new thread")
        return real(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    runtime = CpuCartesianRuntime(_config(), seed=55)
    with pytest.raises(RuntimeError, match="can't start new thread") as caught:
        _integrate(runtime, tmp_path, forecast_publisher)
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["frame-000001", "frame-000002"]
    restarts = sorted((tmp_path / "restarts").iterdir())
    assert [p.name for p in restarts] == ["restart-000001"]
    assert sorted(p.name for p in restarts[0].iterdir()) == RESTART_FILES
    assert any("were published" in note for note in caught.value.__notes__)
    resumed = _resume(restarts[0], forecast_publisher, tmp_path / "restored", seed=55)
    assert resumed.raw_state() == _state_after(4, seed=55)
    assert not _staging_left(tmp_path) and not _publication_threads()


def test_a_signal_inside_thread_start_after_the_worker_exists_leaves_it_in_flight() -> None:
    """``start`` raised, but the thread runs: the event is in flight and is joined."""

    real = threading.Thread.start
    release = threading.Event()

    def start(self):
        real(self)
        raise KeyboardInterrupt  # as if delivered during start's own wait

    publisher = BackgroundPublisher()
    event = EventPublication([
        PublicationPart("restart", lambda cancel: release.wait(10) and "done", discard=lambda: None)
    ])
    original = threading.Thread.start
    threading.Thread.start = start
    try:
        with pytest.raises(KeyboardInterrupt):
            publisher.submit(event)
    finally:
        threading.Thread.start = original
    assert publisher.in_flight is event and publisher.events_submitted == 1
    release.set()
    assert publisher.join() is event and event.parts[0].result == "done"
    assert not _publication_threads()

    # And a start that raised before any thread existed leaves nothing in flight.
    def refused(self):
        raise RuntimeError("can't start new thread")

    threading.Thread.start = refused
    try:
        began = time.monotonic()
        with pytest.raises(RuntimeError):
            publisher.submit(EventPublication([PublicationPart("frame", lambda c: None, lambda: None)]))
    finally:
        threading.Thread.start = original
    assert time.monotonic() - began < 10.0
    assert publisher.in_flight is None and publisher.events_submitted == 1


def test_abort_keeps_a_sigterm_that_arrives_while_it_waits(tmp_path) -> None:
    """A SIGTERM during the cancel wait is returned on the event, not swallowed."""

    from cm1core.background_publication import termination_as_exception

    publisher = BackgroundPublisher()

    def slow(cancel):
        time.sleep(1.0)  # an uninterruptible write: it does not look at cancel
        cancel.check()

    event = EventPublication([PublicationPart("restart", slow, discard=lambda: None)])
    with termination_as_exception():
        publisher.submit(event)
        threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGTERM)).start()
        settled = publisher.abort()
    assert settled is event and isinstance(event.termination, TerminationRequested)
    assert isinstance(event.error, PublicationCancelled)
    assert not _publication_threads()

    # The integration's drain hands it back so the run still ends as 143.
    publisher = BackgroundPublisher()
    event = EventPublication([PublicationPart("restart", slow, discard=lambda: None)])
    failure = KeyboardInterrupt()
    notices = io.StringIO()
    with termination_as_exception():
        publisher.submit(event)
        threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGTERM)).start()
        termination = runtime_module.CudaCartesianRuntime._drain_publication(
            publisher, failure, notices=notices,
        )
    assert isinstance(termination, TerminationRequested) and termination.code == 143
    assert notices.getvalue().splitlines() == [
        "gm1: waiting for restart to finish publishing (Ctrl-C again cancels it and "
        "discards its staging)",
        "gm1: cancelling the publication of restart; its staging is discarded",
    ]
    assert any("was cancelled" in note for note in failure.__notes__)
    assert publisher.in_flight is None and not _publication_threads()


_ORPHAN = """
import sys
sys.path[:0] = [{src!r}]
from cm1core.background_publication import PublicationCancel
from cm1core.cartesian_history import _run_cancellable
_run_cancellable([{hung!r}], None, PublicationCancel())
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux's")
def test_a_publisher_in_its_own_session_dies_with_a_killed_run(tmp_path) -> None:
    """SIGKILL the run: the publisher it started does not keep writing, orphaned."""

    pid_file = tmp_path / "pid"
    hung = tmp_path / "hung-publisher"
    hung.write_text(f"#!/bin/bash\necho $$ > {pid_file}\nexec sleep 60\n")
    hung.chmod(0o755)
    script = tmp_path / "run.py"
    script.write_text(_ORPHAN.format(src=os.fspath(ROOT / "src"), hung=os.fspath(hung)))
    run = subprocess.Popen([sys.executable, os.fspath(script)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    try:
        deadline = time.monotonic() + 30
        while not pid_file.exists() or not pid_file.read_text().strip():
            assert run.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        publisher = int(pid_file.read_text())
        # Its own session, so a signal to the run's process group misses it.
        assert os.getsid(publisher) == publisher != os.getsid(run.pid)
        run.kill()
        run.wait(timeout=10)

        def alive(pid: int) -> bool:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError, IndexError):
                return False
            return state != "Z"

        deadline = time.monotonic() + 10
        while alive(publisher):
            assert time.monotonic() < deadline, "the publisher outlived its killed run"
            time.sleep(0.02)
    finally:
        if run.poll() is None:
            run.kill()
            run.wait()
        if pid_file.exists() and pid_file.read_text().strip():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
