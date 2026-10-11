"""One ``--follow`` writer per run folder at a time (cm1out and the quick-looks).

The GUI starts a run's sidecars again with each segment (Continue, Run it
longer), and a job script or a user may start one by hand.  After a Stop and an
immediate Continue, the previous segment's sidecar can still be finishing its
last frames when the next one starts.  Both would then read and rewrite the
same state file, and cm1out would append to the same one-file-for-every-time
outputs (``output_filetype = 1``).  So a follower takes an advisory lock in the
run folder first; a second one waits, with its stop request honoured, until the
first has finished, then carries on from the files the first one left.

Linux only (``fcntl``), as GM1 is; where ``fcntl`` or the folder is not
available the follower runs unlocked, as it did before.  Standard library only.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import time
from typing import Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - not a platform GM1 runs on
    fcntl = None  # type: ignore[assignment]


class FollowerStopped(RuntimeError):
    """Stopped while another follower still held the run folder."""


def lock_path(target: Path | str, name: str) -> Path:
    """``TARGET/.NAME-follow.lock`` (hidden: an import and the run directory checks pass over it)."""

    return Path(target) / f".{name}-follow.lock"


@contextmanager
def follow_lock(target: Path | str, name: str, *, should_stop: Callable[[], bool] | None = None,
                log: Callable[[str], None] | None = None, poll_s: float = 0.5) -> Iterator[bool]:
    """Hold ``name``'s follower lock on the run folder ``target`` for the ``with`` block.

    Yields True while the lock is held, False when it could not be taken
    (no ``fcntl``, or a folder that cannot hold the lock file), in which case
    the follower runs as it always did.  Raises :class:`FollowerStopped` when
    ``should_stop`` turns true while it waits.
    """

    should_stop = should_stop or (lambda: False)
    folder = Path(target)
    fd = None
    if fcntl is not None and folder.is_dir():
        try:
            fd = os.open(lock_path(folder, name), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            fd = None
    if fd is None:
        yield False
        return
    try:
        told = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if should_stop():
                    raise FollowerStopped(
                        f"stopped while another {name} --follow was still writing {folder}; it finishes the run's "
                        "frames") from None
                if not told and log is not None:
                    log(f"another {name} --follow is still writing {folder}; waiting for it to finish")
                told = True
                time.sleep(poll_s)
        yield True
    finally:
        os.close(fd)  # closing the descriptor releases the lock


__all__ = ["FollowerStopped", "follow_lock", "lock_path"]
