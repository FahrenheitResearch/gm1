"""Chunked device-to-host publication of explicitly named float32 carriers.

This is the installed-package form of the raw arena writer that previously
lived only in ``tools/run_real_gm1_child_one_step.py`` against hard-coded
300x300x75 literals.  A caller supplies the ordered carrier-name to shape
mapping for its own grid; nothing here knows a fixture, storm or domain.

The arena is exactly the concatenated little-endian binary32 words the Rust
``cm1-forecast-publish`` boundary reads back by ``(path, offset)``.  Every
carrier is proven contiguous, correctly shaped and correctly typed before
the output file is created, so a refused publication leaves no partial file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any

from .cuda_memory_inventory import shape_bytes


FORECAST_ARENA_SCHEMA = "gm1.forecast-raw-arena/v1"
FORECAST_ARENA_REPRESENTATION = (
    "concatenated raw IEEE-754 binary32 little-endian carriers in the declared "
    "mapping order; no header, padding or alignment between carriers"
)
# The same bounded host staging window the admitted child post-step runner
# uses (8 MiB per chunk); it bounds host RAM, never the carrier count.
FORECAST_ARENA_TRANSFER_CHUNK_WORDS = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ForecastArenaWrite:
    """One completed raw arena, with the per-carrier digests Rust re-derives."""

    path: Path
    sha256: str
    nbytes: int
    carrier_sha256: Mapping[str, str]
    carrier_offset_bytes: Mapping[str, int]

    def record(self) -> dict[str, Any]:
        return {
            "schema": FORECAST_ARENA_SCHEMA,
            "path": os.fspath(self.path),
            "representation": FORECAST_ARENA_REPRESENTATION,
            "sha256": self.sha256,
            "nbytes": self.nbytes,
            "transfer_chunk_words": FORECAST_ARENA_TRANSFER_CHUNK_WORDS,
            "carrier_sha256": dict(self.carrier_sha256),
            "carrier_offset_bytes": dict(self.carrier_offset_bytes),
        }


def forecast_arena_layout(shapes: Mapping[str, tuple[int, ...]]) -> Mapping[str, int]:
    """Price named float32 carriers into byte offsets without touching CUDA."""

    if not shapes:
        raise ValueError("a forecast arena needs at least one named carrier")
    offsets: dict[str, int] = {}
    total = 0
    for name, shape in shapes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("arena carrier names must be nonempty strings")
        if not shape or any(type(extent) is not int or extent < 1 for extent in shape):
            raise ValueError(
                f"carrier {name} declares shape {tuple(shape)!r}; a zero or "
                "non-integer extent cannot address any binary32 word"
            )
        offsets[name] = total
        total += shape_bytes(shape)
    return MappingProxyType(offsets)


def forecast_arena_nbytes(shapes: Mapping[str, tuple[int, ...]]) -> int:
    """Total published payload bytes for the declared carrier mapping."""

    return sum(shape_bytes(shape) for shape in shapes.values()) if shapes else 0


def require_forecast_arena_carriers(
    carriers: Mapping[str, Any], shapes: Mapping[str, tuple[int, ...]], *, cupy: Any
) -> None:
    """Refuse, by carrier name, anything the raw arena cannot represent.

    This runs before the output file is created.  A wrong shape or a strided
    view is a publication defect, not a transfer defect: the arena has no
    per-carrier header, so the first bad carrier silently renames every word
    that follows it.
    """

    if tuple(carriers) != tuple(shapes):
        raise ValueError(
            f"arena carrier order {tuple(carriers)!r} differs from the declared "
            f"layout order {tuple(shapes)!r}; each carrier after the first "
            "difference would be republished under another carrier's name"
        )
    float32 = cupy.dtype("float32")
    for name, shape in shapes.items():
        value = carriers[name]
        if not isinstance(value, cupy.ndarray):
            raise ValueError(
                f"carrier {name} is {type(value).__name__}, not a CUDA array; "
                "there are no device words to copy under that name"
            )
        if value.dtype != float32:
            raise ValueError(
                f"carrier {name} is {value.dtype}, not float32; the forecast "
                "envelope would decode its words as a different scalar type"
            )
        if tuple(value.shape) != tuple(shape):
            raise ValueError(
                f"carrier {name} has shape {tuple(value.shape)!r}; the declared "
                f"arena shape is {tuple(shape)!r}, and writing it would shift "
                "every following carrier's byte offset"
            )
        if not bool(value.flags.c_contiguous):
            raise ValueError(
                f"carrier {name} is a strided device view; copying its words "
                "in address order would publish halo and neighbouring-carrier "
                f"values under the {name} name"
            )


# Chunks smaller than this are hashed on the writing thread: handing them to a
# worker costs more than the digest.  hashlib releases the GIL above 2047
# bytes, so larger chunks really are digested in parallel with the write.
_PARALLEL_HASH_MIN_BYTES = 1 << 20


class _PinnedChunk:
    """One reusable page-locked host window for the chunked device reads.

    ``cupy.asnumpy`` allocates a fresh pageable array per chunk and copies
    into it through the driver's own bounce buffer.  A page-locked window the
    size of one transfer chunk is allocated once per arena and every chunk is
    read straight into it, so the words land in exactly the same host bytes
    either way; only the copy path differs.  A storage module without pinned
    allocation (a host double) keeps the ``asnumpy`` path.
    """

    def __init__(self, cupy: Any, words: int) -> None:
        self.view = None
        allocate = getattr(getattr(cupy, "cuda", None), "alloc_pinned_memory", None)
        if allocate is None or words <= 0:
            return
        import numpy

        try:
            self._memory = allocate(4 * words)
            self.view = numpy.frombuffer(self._memory, dtype=numpy.float32, count=words)
        except Exception:  # noqa: BLE001 - pinned memory is an optimisation only
            self.view = None

    def read(self, cupy: Any, device: Any) -> Any:
        count = int(device.size)
        view = self.view
        if view is not None and count <= view.size and hasattr(device, "get"):
            window = view[:count]
            device.get(out=window)
            return window
        return cupy.asnumpy(device)


def write_forecast_arena_new(
    path: Path, carriers: Mapping[str, Any], *, shapes: Mapping[str, tuple[int, ...]],
    cupy: Any, io: Any = None, fsync: bool = True, cancel: Any = None,
) -> ForecastArenaWrite:
    """Stream declared device carriers to one new raw file in bounded chunks.

    Each chunk is read into one reusable host window, written, and fed to the
    aggregate and the per-carrier SHA-256 on two worker threads while the
    write proceeds; the digests absorb the same bytes in the same order as a
    serial loop would, so every recorded hash is unchanged.  ``io``, when
    given, receives the device-read, write, hash and fsync stage timings.
    ``fsync`` and ``cancel`` mean what they mean for
    :func:`write_forecast_arena_snapshot_new`; ``cancel.check()`` runs before
    every chunk.
    """

    from concurrent.futures import ThreadPoolExecutor

    offsets = forecast_arena_layout(shapes)
    expected_nbytes = forecast_arena_nbytes(shapes)
    require_forecast_arena_carriers(carriers, shapes, cupy=cupy)
    path = Path(path)
    aggregate = hashlib.sha256()
    carrier_hashes: dict[str, str] = {}
    total = 0
    clock = time.perf_counter
    largest = max(
        (min(FORECAST_ARENA_TRANSFER_CHUNK_WORDS, int(carriers[name].size)) for name in shapes),
        default=0,
    )
    timing = io is not None
    busy = [0.0, 0.0]

    def digest(hasher: Any, raw: memoryview, slot: int) -> None:
        if timing:
            started = clock()
            hasher.update(raw)
            busy[slot] += clock() - started
        else:
            hasher.update(raw)

    staged = clock() if timing else 0.0
    pinned = _PinnedChunk(cupy, largest)
    if timing:
        io.add("arena.host_window", clock() - staged, 4 * largest if pinned.view is not None else 0)
    read_s = write_s = wait_s = 0.0
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gm1-arena-sha256") as pool:
        with path.open("xb", buffering=0) as stream:
            for name in shapes:
                flat = carriers[name].reshape(-1)
                carrier = hashlib.sha256()
                carrier_read = carrier_write = carrier_wait = 0.0
                for start in range(0, int(flat.size), FORECAST_ARENA_TRANSFER_CHUNK_WORDS):
                    if cancel is not None:
                        cancel.check()
                    stop = min(start + FORECAST_ARENA_TRANSFER_CHUNK_WORDS, int(flat.size))
                    if timing:
                        io.range_push("arena.d2h")
                        tick = clock()
                    host = pinned.read(cupy, flat[start:stop])
                    host = host.astype("<f4", copy=False)
                    raw = memoryview(host).cast("B")
                    if timing:
                        tock = clock()
                        io.range_pop()
                        io.range_push("arena.write")
                        carrier_read += tock - tick
                        tick = tock
                    parallel = len(raw) >= _PARALLEL_HASH_MIN_BYTES
                    if parallel:
                        pending = (
                            pool.submit(digest, aggregate, raw, 0),
                            pool.submit(digest, carrier, raw, 1),
                        )
                    written = 0
                    while written != len(raw):
                        count = stream.write(raw[written:])
                        if count is None or count <= 0:
                            raise OSError("raw arena write made no forward progress")
                        written += count
                    if timing:
                        tock = clock()
                        io.range_pop()
                        io.range_push("arena.hash_wait")
                        carrier_write += tock - tick
                        tick = tock
                    if parallel:
                        for future in pending:
                            future.result()
                    else:
                        digest(aggregate, raw, 0)
                        digest(carrier, raw, 1)
                    if timing:
                        carrier_wait += clock() - tick
                        io.range_pop()
                    total += len(raw)
                carrier_hashes[name] = carrier.hexdigest()
                if timing:
                    read_s += carrier_read
                    write_s += carrier_write
                    wait_s += carrier_wait
                    io.carrier(
                        name, 4 * int(flat.size), d2h_s=carrier_read,
                        write_s=carrier_write, hash_wait_s=carrier_wait,
                    )
            if fsync:
                if timing:
                    io.range_push("arena.fsync")
                tick = clock() if timing else 0.0
                os.fsync(stream.fileno())
                if timing:
                    io.add("arena.fsync", clock() - tick, total)
                    io.range_pop()
    if timing:
        io.add("arena.d2h", read_s, total)
        io.add("arena.write", write_s, total)
        io.add("arena.hash_wait", wait_s, total)
        io.add("arena.sha256_busy", busy[0] + busy[1], 2 * total)
    if total != expected_nbytes or path.stat().st_size != expected_nbytes:
        raise RuntimeError(
            f"raw arena byte count is {total}; expected exactly {expected_nbytes}"
        )
    return ForecastArenaWrite(
        path, aggregate.hexdigest(), total,
        MappingProxyType(carrier_hashes), offsets,
    )


# ------------------------------------------------------- host snapshots
#
# A publication split in two: the device words are copied once into host
# memory at the output event, and everything after that -- the digests, the
# file, the publisher -- reads the host copy and never the device.  The
# stepping thread pays only for the copy; the rest can run beside the next
# large steps on a worker thread.

# Bytes written (and fed to the two digests) per call when writing a snapshot.
# Only the size of each write call; the file and every digest are the bytes a
# single call would produce.  Bounded so a cancelled publication stops within
# one window rather than after a multi-gibibyte carrier.
FORECAST_ARENA_WRITE_WINDOW_BYTES = 64 * 1024 * 1024


class HostArena:
    """One reusable host buffer that an event's carrier words are copied into.

    Page-locked when the storage module can allocate it that way (CuPy's
    ``cuda.alloc_pinned_memory``), so the device copy is a direct transfer;
    pageable otherwise, and for host doubles.  It is sized once and reused by
    every later event of the same kind, which is what makes it cheap: pinning
    gigabytes takes longer than copying them.  At most one publication may
    hold it at a time, and :class:`~cm1core.background_publication` enforces
    that by joining the previous event before the next snapshot.

    ``prefault`` touches every page of a pageable arena when it is reserved,
    which page-locking does anyway: otherwise the first event's copy also
    pays gigabytes of first-touch page faults on the stepping thread.
    """

    def __init__(self, *, pinned: bool = True, prefault: bool = True) -> None:
        self.want_pinned = bool(pinned)
        self.prefault = bool(prefault)
        self.pinned = False
        self.capacity_bytes = 0
        self._memory: Any = None
        self._words: Any = None

    def reserve(self, cupy: Any, nbytes: int) -> float:
        """Hold at least ``nbytes``; return the seconds spent allocating."""

        nbytes = int(nbytes)
        if nbytes % 4 != 0 or nbytes <= 0:
            raise ValueError(
                f"a host arena holds whole binary32 words; {nbytes} bytes is not one"
            )
        if self._words is not None and self.capacity_bytes >= nbytes:
            return 0.0
        import numpy

        started = time.perf_counter()
        self.release(cupy)
        allocate = getattr(getattr(cupy, "cuda", None), "alloc_pinned_memory", None)
        if self.want_pinned and allocate is not None:
            try:
                memory = allocate(nbytes)
                self._words = numpy.frombuffer(memory, dtype="<f4", count=nbytes // 4)
                self._memory = memory
                self.pinned = True
            except Exception:  # noqa: BLE001 - page-locking is an optimisation only
                self._memory = self._words = None
        if self._words is None:
            self._words = numpy.empty(nbytes // 4, dtype="<f4")
            if self.prefault:
                self._words.fill(0.0)
            self.pinned = False
        self.capacity_bytes = nbytes
        return time.perf_counter() - started

    def words(self, cupy: Any, nbytes: int) -> Any:
        """The first ``nbytes // 4`` words, allocating on first use."""

        self.reserve(cupy, nbytes)
        return self._words[: int(nbytes) // 4]

    def release(self, cupy: Any = None) -> None:
        pinned = self.pinned
        self._memory = self._words = None
        self.capacity_bytes = 0
        self.pinned = False
        if pinned:
            pool = getattr(cupy, "get_default_pinned_memory_pool", None)
            if callable(pool):
                try:
                    pool().free_all_blocks()
                except Exception:  # noqa: BLE001 - returning pages is best effort
                    pass

    def record(self) -> dict[str, Any]:
        return {"capacity_bytes": self.capacity_bytes, "page_locked": self.pinned}


class DeviceArenaWords:
    """Declared carriers left on the device for a commit on the same thread.

    What a snapshot without a host arena holds: the carriers themselves, not a
    copy.  The one-call publications (``write_cartesian_restart``,
    ``publish_cartesian_history``, a runtime's ``_publish_frame``) commit at
    once on the stepping thread, before the state can change, so their arena
    is streamed from the device in 8 MiB windows exactly as
    :func:`write_forecast_arena_new` always did, and host memory never holds a
    second full copy of the payload beside the publisher's own.  ``release``
    runs once, when the words have been written or the publication is
    abandoned, and frees whatever the caller staged the carriers in.
    """

    def __init__(self, carriers: Mapping[str, Any], cupy: Any,
                 release: Callable[[], None] | None = None) -> None:
        self._carriers: Mapping[str, Any] | None = MappingProxyType(dict(carriers))
        self.cupy = cupy
        self._release = release
        self._thread = threading.get_ident()

    def take(self) -> Mapping[str, Any]:
        if threading.get_ident() != self._thread:
            raise RuntimeError(
                "these carriers were left on the device for a commit on the thread "
                "that took them, and the runtime may have stepped on since; a "
                "publication committed on another thread needs a host arena"
            )
        if self._carriers is None:
            raise RuntimeError(
                "these device carriers were already written or released; writing "
                "them again would read words a later step may have replaced"
            )
        return self._carriers

    def release(self) -> None:
        self._carriers = None
        release, self._release = self._release, None
        if release is not None:
            release()


@dataclass(frozen=True, slots=True)
class ForecastArenaSnapshot:
    """The declared carriers' words, in arena order.

    Held in host memory (``words``), or -- with ``device`` -- still on the
    device, for a commit that runs at once on the thread that took them.
    """

    words: Any
    shapes: Mapping[str, tuple[int, ...]]
    offsets: Mapping[str, int]
    nbytes: int
    carrier_d2h_seconds: Mapping[str, float]
    device: DeviceArenaWords | None = None

    @property
    def on_device(self) -> bool:
        return self.device is not None

    def release(self) -> None:
        """Drop device carriers not yet written; nothing for a host snapshot."""

        if self.device is not None:
            self.device.release()


def snapshot_forecast_arena(
    carriers: Mapping[str, Any], *, shapes: Mapping[str, tuple[int, ...]],
    cupy: Any, host: HostArena | None = None, io: Any = None,
    release: Callable[[], None] | None = None,
) -> ForecastArenaSnapshot:
    """Copy declared device carriers once into host memory, in arena order.

    The same refusals as :func:`write_forecast_arena_new`, before any word is
    copied.  Each carrier is read in :data:`FORECAST_ARENA_TRANSFER_CHUNK_WORDS`
    chunks -- the transfer the receipts have always described -- straight into
    its window of ``host``.  The caller has synchronized the stream that
    produced the words.

    Without a ``host`` nothing is copied: the snapshot keeps the carriers on
    the device (:class:`DeviceArenaWords`), and its write streams them.  That
    is for a commit that follows at once on this thread; ``release``, if
    given, is called once the words are written or abandoned.  With a
    ``host`` the caller frees its staging itself, and ``release`` is unused.
    """

    import numpy

    offsets = forecast_arena_layout(shapes)
    nbytes = forecast_arena_nbytes(shapes)
    require_forecast_arena_carriers(carriers, shapes, cupy=cupy)
    if host is None:
        return ForecastArenaSnapshot(
            None, MappingProxyType(dict(shapes)), offsets, nbytes, MappingProxyType({}),
            device=DeviceArenaWords(carriers, cupy, release),
        )
    timing = io is not None
    clock = time.perf_counter
    if timing:
        io.range_push("arena.host_window")
    allocated = host.reserve(cupy, nbytes)
    if timing:
        io.range_pop()
        if allocated:
            io.add("arena.host_window", allocated, host.capacity_bytes)
    words = host.words(cupy, nbytes)
    seconds: dict[str, float] = {}
    total = 0.0
    for name, shape in shapes.items():
        flat = carriers[name].reshape(-1)
        count = int(flat.size)
        first = offsets[name] // 4
        if timing:
            io.range_push("arena.d2h")
        tick = clock()
        for start in range(0, count, FORECAST_ARENA_TRANSFER_CHUNK_WORDS):
            stop = min(start + FORECAST_ARENA_TRANSFER_CHUNK_WORDS, count)
            window = words[first + start:first + stop]
            chunk = flat[start:stop]
            if hasattr(chunk, "get"):
                # Straight into the arena window, as _PinnedChunk.read does.
                chunk.get(out=window)
            else:
                numpy.copyto(window, cupy.asnumpy(chunk))
        elapsed = clock() - tick
        if timing:
            io.range_pop()
        seconds[name] = elapsed
        total += elapsed
    if timing:
        io.add("arena.d2h", total, nbytes)
    return ForecastArenaSnapshot(
        words, MappingProxyType(dict(shapes)), offsets, nbytes, MappingProxyType(seconds),
    )


def write_forecast_arena_snapshot_new(
    path: Path, snapshot: ForecastArenaSnapshot, *, io: Any = None,
    fsync: bool = True, cancel: Any = None,
) -> ForecastArenaWrite:
    """Write one host snapshot to a new raw file with the arena's digests.

    Byte for byte the file :func:`write_forecast_arena_new` writes from the
    same carriers, with the same aggregate and per-carrier SHA-256: the words
    are the same words in the same order, and a digest does not depend on how
    its input is split between ``update`` calls.  ``fsync=False`` is for an
    arena that is an input to the publisher and is deleted once the envelope
    holding the same words is on disk: syncing a file that is about to be
    unlinked buys no durability.  ``cancel.check()`` is called before every
    write window.  A snapshot whose words are still on the device is streamed
    by :func:`write_forecast_arena_new` itself and then released.
    """

    from concurrent.futures import ThreadPoolExecutor

    if snapshot.device is not None:
        carriers = snapshot.device.take()
        try:
            return write_forecast_arena_new(
                path, carriers, shapes=snapshot.shapes,
                cupy=snapshot.device.cupy, io=io, fsync=fsync, cancel=cancel,
            )
        finally:
            carriers = None
            snapshot.device.release()
    path = Path(path)
    aggregate = hashlib.sha256()
    carrier_hashes: dict[str, str] = {}
    raw = memoryview(snapshot.words).cast("B")
    if len(raw) != snapshot.nbytes:
        raise RuntimeError(
            f"the host snapshot holds {len(raw)} bytes; its layout declares {snapshot.nbytes}"
        )
    timing = io is not None
    clock = time.perf_counter
    busy = [0.0, 0.0]

    def digest(hasher: Any, piece: memoryview, slot: int) -> None:
        if timing:
            started = clock()
            hasher.update(piece)
            busy[slot] += clock() - started
        else:
            hasher.update(piece)

    total = 0
    write_s = wait_s = 0.0
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gm1-arena-sha256") as pool:
        with path.open("xb", buffering=0) as stream:
            for name, shape in snapshot.shapes.items():
                begin = snapshot.offsets[name]
                end = begin + shape_bytes(shape)
                carrier = hashlib.sha256()
                carrier_write = carrier_wait = 0.0
                for start in range(begin, end, FORECAST_ARENA_WRITE_WINDOW_BYTES):
                    if cancel is not None:
                        cancel.check()
                    piece = raw[start:min(start + FORECAST_ARENA_WRITE_WINDOW_BYTES, end)]
                    if timing:
                        io.range_push("arena.write")
                        tick = clock()
                    parallel = len(piece) >= _PARALLEL_HASH_MIN_BYTES
                    if parallel:
                        pending = (
                            pool.submit(digest, aggregate, piece, 0),
                            pool.submit(digest, carrier, piece, 1),
                        )
                    written = 0
                    while written != len(piece):
                        count = stream.write(piece[written:])
                        if count is None or count <= 0:
                            raise OSError("raw arena write made no forward progress")
                        written += count
                    if timing:
                        tock = clock()
                        io.range_pop()
                        io.range_push("arena.hash_wait")
                        carrier_write += tock - tick
                        tick = tock
                    if parallel:
                        for future in pending:
                            future.result()
                    else:
                        digest(aggregate, piece, 0)
                        digest(carrier, piece, 1)
                    if timing:
                        carrier_wait += clock() - tick
                        io.range_pop()
                    total += len(piece)
                carrier_hashes[name] = carrier.hexdigest()
                if timing:
                    write_s += carrier_write
                    wait_s += carrier_wait
                    io.carrier(
                        name, end - begin, d2h_s=snapshot.carrier_d2h_seconds.get(name, 0.0),
                        write_s=carrier_write, hash_wait_s=carrier_wait,
                    )
            if fsync:
                if timing:
                    io.range_push("arena.fsync")
                tick = clock() if timing else 0.0
                os.fsync(stream.fileno())
                if timing:
                    io.add("arena.fsync", clock() - tick, total)
                    io.range_pop()
    if timing:
        io.add("arena.write", write_s, total)
        io.add("arena.hash_wait", wait_s, total)
        io.add("arena.sha256_busy", busy[0] + busy[1], 2 * total)
    if total != snapshot.nbytes or path.stat().st_size != snapshot.nbytes:
        raise RuntimeError(
            f"raw arena byte count is {total}; expected exactly {snapshot.nbytes}"
        )
    return ForecastArenaWrite(
        path, aggregate.hexdigest(), total,
        MappingProxyType(carrier_hashes), snapshot.offsets,
    )


__all__ = (
    "FORECAST_ARENA_REPRESENTATION",
    "FORECAST_ARENA_SCHEMA",
    "FORECAST_ARENA_TRANSFER_CHUNK_WORDS",
    "FORECAST_ARENA_WRITE_WINDOW_BYTES",
    "DeviceArenaWords",
    "ForecastArenaSnapshot",
    "ForecastArenaWrite",
    "HostArena",
    "forecast_arena_layout",
    "forecast_arena_nbytes",
    "require_forecast_arena_carriers",
    "snapshot_forecast_arena",
    "write_forecast_arena_new",
    "write_forecast_arena_snapshot_new",
)
