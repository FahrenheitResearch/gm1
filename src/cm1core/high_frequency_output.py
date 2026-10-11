"""CM1 ``hifrq.F``'s high-frequency output, as a targeted subdomain.

Authority
---------
``src/hifrq.F:21`` ``writeout_hifrq``, driven by ``dohifrq`` and ``hifrqfrq``
(``src/cm1.F:2645-2647``; ``src/cm1.F:3228-3263`` is the cadence test and
``:4125-4166`` the write and the next-time advance).  In r21.1 those two words
are **not** in any namelist: ``cm1.F:2645`` hard-codes ``dohifrq = .false.``
and ``:2647`` hard-codes ``hifrqfrq = 3600.0``, under the comment "New for
cm1r19.6 (not thoroughly tested)".  Turning it on means editing and rebuilding
CM1.

What GM1 adds
-------------
**GM1 EXTENSION (window).**  ``hifrq.F`` writes the whole domain.  A tornado
evolves on seconds inside a few kilometres of a domain that may be hundreds of
kilometres across, so writing the whole domain every few seconds is the cost
that makes people not do it.  This module writes a declared
``(k0:k1, j0:j1, i0:i1)`` window instead, so the frequent output is the size
of the vortex and the full domain keeps its own rare cadence.

**GM1 EXTENSION (selector).**  ``dohifrq``/``hifrqfrq`` become configuration
rather than a source edit, under CM1's own names.

Format
------
The same raw little-endian float32 arena and TSV field table the full history
frame uses (:mod:`cm1core.cartesian_history`), so one reader reads both.  The
window's own extents travel in the table, because a windowed frame whose
offsets are not stated is unreadable.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

HIFRQ_SCHEMA = "gpuwm-cm1.high-frequency-window/v1"
HIFRQ_ARENA_FILENAME = "hifrq-carriers.f32"
HIFRQ_TABLE_FILENAME = "hifrq-field-table.tsv"
HIFRQ_RECEIPT_FILENAME = "hifrq-window.json"

SOURCE_ANCHORS: Mapping[str, str] = {
    "writeout_hifrq": "src/hifrq.F:21-1790",
    "selectors": "src/cm1.F:2645-2647",
    "cadence_test": "src/cm1.F:3228-3263",
    "write_and_advance": "src/cm1.F:4125-4166",
    "window": "GM1 EXTENSION: hifrq.F writes the whole domain",
}


@dataclass(frozen=True, slots=True)
class HighFrequencyWindow:
    """A closed-open index window into the physical scalar rectangle.

    Indices are zero-based into the ``(nz, ny, nx)`` physical rectangle, which
    is the array a reader of a GM1 history frame already holds.  They are
    refused against the grid rather than clipped: a window silently clipped to
    the domain would publish a different volume from the one the document
    names, and every position in the output would be offset.
    """

    i0: int
    i1: int
    j0: int
    j1: int
    k0: int
    k1: int

    def __post_init__(self) -> None:
        for name in ("i0", "i1", "j0", "j1", "k0", "k1"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        for lo, hi in (("i0", "i1"), ("j0", "j1"), ("k0", "k1")):
            if getattr(self, hi) <= getattr(self, lo):
                raise ValueError(f"{hi} must exceed {lo}; an empty window writes nothing")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.k1 - self.k0, self.j1 - self.j0, self.i1 - self.i0)

    @property
    def words(self) -> int:
        return math.prod(self.shape)

    def require_within(self, *, nx: int, ny: int, nz: int) -> None:
        for hi, extent, name in ((self.i1, nx, "i1"), (self.j1, ny, "j1"),
                                 (self.k1, nz, "k1")):
            if hi > extent:
                raise ValueError(
                    f"high-frequency window {name}={hi} exceeds the grid "
                    f"({nx}x{ny}x{nz}); the window would name cells this run "
                    "does not integrate"
                )

    def slices(self) -> tuple[slice, slice, slice]:
        return (slice(self.k0, self.k1), slice(self.j0, self.j1), slice(self.i0, self.i1))

    def record(self) -> dict[str, Any]:
        return {
            "i0": self.i0, "i1": self.i1, "j0": self.j0, "j1": self.j1,
            "k0": self.k0, "k1": self.k1,
            "shape_zyx": list(self.shape), "words": self.words,
        }


def full_domain_window(*, nx: int, ny: int, nz: int) -> HighFrequencyWindow:
    """``hifrq.F``'s own extent: the whole domain."""

    return HighFrequencyWindow(0, nx, 0, ny, 0, nz)


def high_frequency_cadence(*, dt_s: float, duration_s: float, interval_s: float) -> int:
    """Steps between high-frequency writes, or a refusal by value.

    ``cm1.F:3229`` tests ``(mtime+dbldt) >= (hifrqtim - 0.1*dbldt)``, which
    fires on the first step at or past the requested time.  GM1 requires the
    interval to be a whole number of large steps instead, for the same reason
    :func:`cm1core.gm1_runtime.history_cadence` does: a cadence that is not an
    exact step count writes frames at times the document does not name.
    """

    from .gm1_runtime import _exact_step_count

    total = _exact_step_count("duration_s", duration_s, dt_s)
    steps = _exact_step_count("hifrqfrq_s", interval_s, dt_s)
    if steps > total:
        raise ValueError(
            f"hifrqfrq_s={interval_s} is {steps} large steps but the run is "
            f"{total}; no high-frequency frame would ever be written"
        )
    return steps


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_high_frequency_window(
    directory: Path | str,
    fields: Mapping[str, Any],
    window: HighFrequencyWindow,
    *,
    order: Sequence[str],
    nx: int,
    ny: int,
    nz: int,
    model_time_s: float,
    step: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one window frame: arena, field table, receipt.

    ``extra`` is merged into the receipt *before* it is written, so anything
    the caller must say about the frame -- what kind it is, what has not been
    validated -- is on disk and not only in the returned dict.  A marker that
    exists only in memory is a marker nobody reading the artifact will see.

    ``fields`` are host ``(nz, ny, nx)`` float32 rectangles; ``order`` fixes
    the arena order so the table and the bytes cannot disagree.
    """

    import numpy as np

    window.require_within(nx=nx, ny=ny, nz=nz)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    cut = window.slices()
    payload = bytearray()
    rows: list[str] = ["name\toffset_bytes\twords\tshape_zyx\tsha256"]
    offset = 0
    for name in order:
        if name not in fields:
            raise ValueError(f"high-frequency field {name!r} was declared and not supplied")
        block = np.ascontiguousarray(fields[name], dtype=np.float32)
        if block.shape != (nz, ny, nx):
            raise ValueError(
                f"high-frequency field {name!r} must be the (nz, ny, nx) rectangle"
            )
        cutout = np.ascontiguousarray(block[cut])
        raw = cutout.tobytes(order="C")
        rows.append(
            f"{name}\t{offset}\t{cutout.size}\t"
            f"{'x'.join(str(v) for v in cutout.shape)}\t"
            f"{hashlib.sha256(raw).hexdigest()}"
        )
        payload += raw
        offset += len(raw)
    arena = root / HIFRQ_ARENA_FILENAME
    with arena.open("xb", buffering=0) as stream:
        stream.write(bytes(payload))
        os.fsync(stream.fileno())
    table = root / HIFRQ_TABLE_FILENAME
    with table.open("xb", buffering=0) as stream:
        stream.write(("\n".join(rows) + "\n").encode("utf-8"))
        os.fsync(stream.fileno())
    receipt = {
        "schema": HIFRQ_SCHEMA,
        "step": int(step),
        "model_time_s": float(model_time_s),
        "window": window.record(),
        "fields": list(order),
        "arena_bytes": offset,
        "arena_sha256": hashlib.sha256(bytes(payload)).hexdigest(),
        "cm1_authority": dict(SOURCE_ANCHORS),
        "beyond_cm1": "hifrq.F writes the whole domain; this writes the declared window",
    }
    if extra:
        receipt.update(dict(extra))
    import json

    path = root / HIFRQ_RECEIPT_FILENAME
    with path.open("xb", buffering=0) as stream:
        stream.write((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(stream.fileno())
    _fsync_directory(root)
    return receipt


__all__ = [
    "HIFRQ_ARENA_FILENAME", "HIFRQ_RECEIPT_FILENAME", "HIFRQ_SCHEMA",
    "HIFRQ_TABLE_FILENAME", "HighFrequencyWindow", "SOURCE_ANCHORS",
    "full_domain_window", "high_frequency_cadence", "write_high_frequency_window",
]
