"""CPU arena-layout and refusal checks; no CUDA transfer is claimed here."""

from types import SimpleNamespace

import hashlib
import numpy as np
import pytest

from cm1core import forecast_arena
from cm1core.forecast_arena import (
    FORECAST_ARENA_TRANSFER_CHUNK_WORDS,
    HostArena,
    forecast_arena_layout,
    forecast_arena_nbytes,
    require_forecast_arena_carriers,
    snapshot_forecast_arena,
    write_forecast_arena_new,
    write_forecast_arena_snapshot_new,
)


def _storage():
    """Only array storage and the host copy are emulated, never a device."""
    return SimpleNamespace(ndarray=np.ndarray, dtype=np.dtype, asnumpy=np.asarray)


def _words(count, *, start=0):
    return (np.arange(start, start + count, dtype=np.float32) * np.float32(0.25)).copy()


def test_layout_offsets_and_totals_are_the_concatenated_carrier_payloads():
    shapes = {"th3d": (4, 3, 2), "q3d": (6, 4, 3, 2), "terrain_msl": (3, 2)}
    offsets = forecast_arena_layout(shapes)
    assert dict(offsets) == {"th3d": 0, "q3d": 96, "terrain_msl": 672}
    assert forecast_arena_nbytes(shapes) == 96 + 576 + 24
    with pytest.raises(ValueError, match="binary32 word"):
        forecast_arena_layout({"th3d": (4, 0, 2)})
    with pytest.raises(ValueError, match="at least one named carrier"):
        forecast_arena_layout({})


def test_written_arena_is_exact_carrier_bytes_with_per_carrier_digests(tmp_path):
    shapes = {"th3d": (4, 3, 2), "terrain_msl": (3, 2), "rf": (5, 3, 2)}
    carriers, expected = {}, b""
    for index, (name, shape) in enumerate(shapes.items()):
        value = _words(int(np.prod(shape)), start=index * 1000).reshape(shape)
        carriers[name], expected = value, expected + value.tobytes()
    path = tmp_path / "history-carriers.f32"
    write = write_forecast_arena_new(path, carriers, shapes=shapes, cupy=_storage())
    assert path.read_bytes() == expected
    assert write.nbytes == len(expected) == forecast_arena_nbytes(shapes)
    assert write.sha256 == hashlib.sha256(expected).hexdigest()
    assert dict(write.carrier_offset_bytes) == dict(forecast_arena_layout(shapes))
    for name, value in carriers.items():
        assert write.carrier_sha256[name] == hashlib.sha256(value.tobytes()).hexdigest()
    assert write.record()["transfer_chunk_words"] == FORECAST_ARENA_TRANSFER_CHUNK_WORDS
    with pytest.raises(FileExistsError):
        write_forecast_arena_new(path, carriers, shapes=shapes, cupy=_storage())


def test_a_carrier_larger_than_one_transfer_chunk_is_streamed_word_exact(tmp_path):
    shape = (FORECAST_ARENA_TRANSFER_CHUNK_WORDS + 7,)
    carriers = {"prs": _words(shape[0])}
    path = tmp_path / "chunked.f32"
    write = write_forecast_arena_new(path, carriers, shapes={"prs": shape}, cupy=_storage())
    assert write.nbytes == 4 * shape[0]
    assert path.read_bytes() == carriers["prs"].tobytes()
    assert write.carrier_sha256["prs"] == hashlib.sha256(carriers["prs"].tobytes()).hexdigest()


def test_wrong_shape_type_order_or_strided_carrier_refuses_by_name_before_the_file(tmp_path):
    shapes = {"th3d": (4, 3, 2), "rf": (5, 3, 2)}
    good = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    padded = np.zeros((7, 5, 4), dtype=np.float32)
    cases = (
        ("rf", {**good, "rf": np.zeros((5, 3, 3), dtype=np.float32)}, "declared"),
        ("rf", {**good, "rf": np.zeros((5, 3, 2), dtype=np.float64)}, "float32"),
        ("rf", {**good, "rf": padded[1:6, 1:4, 1:3]}, "strided device view"),
        ("rf", {**good, "rf": [0.0] * 30}, "not a CUDA array"),
    )
    for index, (name, carriers, message) in enumerate(cases):
        path = tmp_path / f"refused-{index}.f32"
        with pytest.raises(ValueError, match=message) as error:
            write_forecast_arena_new(path, carriers, shapes=shapes, cupy=_storage())
        assert name in str(error.value)
        assert not path.exists()
    reordered = {"rf": good["rf"], "th3d": good["th3d"]}
    path = tmp_path / "reordered.f32"
    with pytest.raises(ValueError, match="differs from the declared"):
        write_forecast_arena_new(path, reordered, shapes=shapes, cupy=_storage())
    assert not path.exists()


def test_contiguity_gate_accepts_the_materialized_copy_of_a_strided_view():
    padded = np.arange(7 * 5 * 4, dtype=np.float32).reshape(7, 5, 4)
    view = padded[1:6, 1:4, 1:3]
    shapes = {"rf": (5, 3, 2)}
    with pytest.raises(ValueError, match="strided device view"):
        require_forecast_arena_carriers({"rf": view}, shapes, cupy=_storage())
    require_forecast_arena_carriers({"rf": np.ascontiguousarray(view)}, shapes, cupy=_storage())


class _DeviceWords(np.ndarray):
    """A host array that answers ``get(out=...)`` the way a CuPy array does."""

    reads: list = []

    def get(self, out=None, stream=None):
        _DeviceWords.reads.append(("get", self.size, out is not None))
        np.copyto(out, self.view(np.ndarray), casting="no")
        return out


def _pinned_storage(allocations):
    def alloc_pinned_memory(nbytes):
        allocations.append(nbytes)
        return bytearray(nbytes)

    def asnumpy(value):
        _DeviceWords.reads.append(("asnumpy", value.size, False))
        return np.array(value, copy=True).view(np.ndarray)

    return SimpleNamespace(
        ndarray=np.ndarray, dtype=np.dtype, asnumpy=asnumpy,
        cuda=SimpleNamespace(alloc_pinned_memory=alloc_pinned_memory),
    )


class _Stages:
    def __init__(self):
        self.stages, self.carriers = {}, []

    def add(self, stage, seconds, nbytes=0):
        row = self.stages.setdefault(stage, [0.0, 0])
        row[0] += seconds
        row[1] += nbytes

    def carrier(self, name, nbytes, **seconds):
        self.carriers.append((name, nbytes, sorted(seconds)))

    def range_push(self, name):
        self.depth = getattr(self, "depth", 0) + 1

    def range_pop(self):
        self.depth -= 1


def test_the_pinned_window_and_parallel_digests_publish_the_serial_bytes_and_hashes(tmp_path):
    """One reusable page-locked window, two digest threads: same file, same SHA-256s.

    The carriers straddle the transfer chunk and the parallel-digest threshold
    so every branch of the loop -- a partial final chunk, a chunk hashed on
    worker threads and a small carrier hashed inline -- is exercised, and each
    digest is checked against hashlib over the carrier's own bytes.
    """

    words = FORECAST_ARENA_TRANSFER_CHUNK_WORDS
    shapes = {"q3d": (2, words + 13), "zs": (3, 5), "w3d": (words // 3,)}
    carriers, expected = {}, b""
    for index, (name, shape) in enumerate(shapes.items()):
        value = _words(int(np.prod(shape)), start=17 * index).reshape(shape).view(_DeviceWords)
        carriers[name] = value
        expected += value.view(np.ndarray).tobytes()
    allocations = []
    _DeviceWords.reads = []
    stages = _Stages()
    path = tmp_path / "pinned.f32"
    write = write_forecast_arena_new(
        path, carriers, shapes=shapes, cupy=_pinned_storage(allocations), io=stages,
    )
    assert path.read_bytes() == expected
    assert write.sha256 == hashlib.sha256(expected).hexdigest()
    for name, value in carriers.items():
        assert write.carrier_sha256[name] == hashlib.sha256(value.view(np.ndarray).tobytes()).hexdigest()
    # One window, sized to one transfer chunk, and every read went into it.
    assert allocations == [4 * words]
    assert {kind for kind, _, _ in _DeviceWords.reads} == {"get"}
    assert sum(size for _, size, _ in _DeviceWords.reads) == sum(int(np.prod(s)) for s in shapes.values())
    assert set(stages.stages) >= {"arena.d2h", "arena.write", "arena.fsync", "arena.hash_wait", "arena.sha256_busy"}
    assert stages.stages["arena.write"][1] == len(expected)
    assert stages.stages["arena.sha256_busy"][1] == 2 * len(expected)
    assert [name for name, _, _ in stages.carriers] == list(shapes)
    assert stages.depth == 0  # every NVTX sub-step range was closed


def test_a_storage_without_pinned_memory_keeps_the_asnumpy_path(tmp_path):
    shapes = {"th3d": (4, 3, 2)}
    carriers = {"th3d": _words(24).reshape(4, 3, 2)}
    path = tmp_path / "plain.f32"
    write = write_forecast_arena_new(path, carriers, shapes=shapes, cupy=_storage(), io=_Stages())
    assert path.read_bytes() == carriers["th3d"].tobytes()
    assert write.sha256 == hashlib.sha256(carriers["th3d"].tobytes()).hexdigest()


class _Cancelled(RuntimeError):
    pass


def test_a_host_snapshot_writes_the_streaming_writers_bytes_and_digests(tmp_path, monkeypatch):
    """Copy once into a reused host arena, write later: the same file and SHA-256s.

    The snapshot and its write are the two halves background publication runs
    on different threads.  The carriers straddle the transfer chunk, and the
    write window is shrunk to a ragged size so pieces cross the parallel-digest
    threshold both ways; the file, the aggregate and every per-carrier digest
    must still be the one-pass streaming writer's.  A second event reuses the
    arena without allocating, and a cancel stops the write at a window edge.
    """

    words = FORECAST_ARENA_TRANSFER_CHUNK_WORDS
    shapes = {"q3d": (2, words + 13), "zs": (3, 5), "w3d": (words // 3,)}

    def carriers_from(seed):
        return {
            name: _words(int(np.prod(shape)), start=seed + 17 * index).reshape(shape).view(_DeviceWords)
            for index, (name, shape) in enumerate(shapes.items())
        }

    carriers = carriers_from(0)
    streamed = write_forecast_arena_new(
        tmp_path / "streamed.f32", carriers, shapes=shapes, cupy=_storage(),
    )
    window = 1_500_007
    monkeypatch.setattr(forecast_arena, "FORECAST_ARENA_WRITE_WINDOW_BYTES", window)
    allocations = []
    storage = _pinned_storage(allocations)
    host = HostArena()
    _DeviceWords.reads = []
    snapshot_stages = _Stages()
    snapshot = snapshot_forecast_arena(
        carriers, shapes=shapes, cupy=storage, host=host, io=snapshot_stages,
    )
    nbytes = forecast_arena_nbytes(shapes)
    assert allocations == [nbytes] and host.pinned and host.capacity_bytes == nbytes
    # Every chunk went straight into the arena with get(out=...).
    assert {kind for kind, _, _ in _DeviceWords.reads} == {"get"}
    assert set(snapshot_stages.stages) == {"arena.host_window", "arena.d2h"}
    assert snapshot_stages.depth == 0
    # The device words may change after the snapshot: the write reads the copy.
    for value in carriers.values():
        value[...] = np.float32(-1.0)
    checks = []
    stages = _Stages()
    path = tmp_path / "snapshot.f32"
    write = write_forecast_arena_snapshot_new(
        path, snapshot, io=stages, fsync=False,
        cancel=SimpleNamespace(check=lambda: checks.append(None)),
    )
    assert path.read_bytes() == (tmp_path / "streamed.f32").read_bytes()
    assert (write.sha256, write.nbytes) == (streamed.sha256, streamed.nbytes)
    assert dict(write.carrier_sha256) == dict(streamed.carrier_sha256)
    assert dict(write.carrier_offset_bytes) == dict(streamed.carrier_offset_bytes)
    assert {**write.record(), "path": None} == {**streamed.record(), "path": None}
    assert len(checks) == sum(-(-4 * int(np.prod(shape)) // window) for shape in shapes.values())
    assert "arena.fsync" not in stages.stages and stages.stages["arena.write"][1] == nbytes
    assert [name for name, _, _ in stages.carriers] == list(shapes)
    assert stages.depth == 0
    # A second event: new words, the same arena, no allocation.
    second = carriers_from(5000)
    expected = b"".join(value.view(np.ndarray).tobytes() for value in second.values())
    again = snapshot_forecast_arena(second, shapes=shapes, cupy=storage, host=host)
    assert allocations == [nbytes]
    synced = _Stages()
    write = write_forecast_arena_snapshot_new(tmp_path / "second.f32", again, io=synced)
    assert (tmp_path / "second.f32").read_bytes() == expected
    assert write.sha256 == hashlib.sha256(expected).hexdigest()
    assert "arena.fsync" in synced.stages

    def cancel_on_second_window():
        checks.append(None)
        if len(checks) > 1:
            raise _Cancelled("cancelled")

    checks.clear()
    with pytest.raises(_Cancelled):
        write_forecast_arena_snapshot_new(
            tmp_path / "cancelled.f32", again,
            cancel=SimpleNamespace(check=cancel_on_second_window),
        )
    assert (tmp_path / "cancelled.f32").stat().st_size == window


def test_the_cupy_host_stand_in_snapshots_the_bytes_the_streaming_writer_publishes(tmp_path):
    """The same halves through ``tools.oracle.host_cupy``, the repository's CuPy stand-in.

    Its ``cuda.alloc_pinned_memory`` backs the arena and its ``ndarray.get(out=...)``
    fills it, the calls a CUDA runtime makes; the streaming writer goes through
    the same module's pinned window.  Every byte and digest must agree.
    """

    from tools.oracle import host_cupy

    cupy = host_cupy.build()
    words = FORECAST_ARENA_TRANSFER_CHUNK_WORDS
    shapes = {"ua": (3, words // 2 + 5), "prs": (2, 7, 11), "tkea": (words + 1,)}
    carriers = {
        name: _words(int(np.prod(shape)), start=101 * index).reshape(shape).view(host_cupy.ndarray)
        for index, (name, shape) in enumerate(shapes.items())
    }
    streamed = write_forecast_arena_new(tmp_path / "streamed.f32", carriers, shapes=shapes, cupy=cupy)
    host = HostArena()
    host_cupy.COUNTS.clear()
    snapshot = snapshot_forecast_arena(carriers, shapes=shapes, cupy=cupy, host=host)
    assert host.pinned and host_cupy.COUNTS["device_to_host_read"] == sum(
        -(-int(np.prod(shape)) // words) for shape in shapes.values()
    )
    write = write_forecast_arena_snapshot_new(tmp_path / "snapshot.f32", snapshot)
    assert (tmp_path / "snapshot.f32").read_bytes() == (tmp_path / "streamed.f32").read_bytes()
    assert (write.sha256, dict(write.carrier_sha256)) == (streamed.sha256, dict(streamed.carrier_sha256))
    host.release(cupy)
    assert host.capacity_bytes == 0 and not host.pinned


def test_words_left_on_the_device_are_streamed_once_by_the_thread_that_took_them(tmp_path):
    """Without a host arena nothing is copied: the write streams the carriers.

    The one-call publications commit at once on the stepping thread, so their
    snapshot keeps the carriers on the device and the write is the streaming
    writer's own -- the same file and digests, no payload-sized host buffer.
    A write from another thread is refused (the runtime may have stepped on),
    without releasing the words; the caller's release runs exactly once.
    """

    import threading

    words = FORECAST_ARENA_TRANSFER_CHUNK_WORDS
    shapes = {"q3d": (2, words + 13), "zs": (3, 5)}
    carriers = {
        name: _words(int(np.prod(shape)), start=7 * index).reshape(shape)
        for index, (name, shape) in enumerate(shapes.items())
    }
    streamed = write_forecast_arena_new(
        tmp_path / "streamed.f32", carriers, shapes=shapes, cupy=_storage(),
    )
    released = []
    snapshot = snapshot_forecast_arena(
        carriers, shapes=shapes, cupy=_storage(), release=lambda: released.append(True),
    )
    assert snapshot.on_device and snapshot.words is None

    refused = []

    def elsewhere():
        try:
            write_forecast_arena_snapshot_new(tmp_path / "elsewhere.f32", snapshot)
        except RuntimeError as error:
            refused.append(str(error))

    worker = threading.Thread(target=elsewhere)
    worker.start()
    worker.join()
    assert refused and "needs a host arena" in refused[0]
    assert not (tmp_path / "elsewhere.f32").exists() and released == []

    stages = _Stages()
    checks = []
    write = write_forecast_arena_snapshot_new(
        tmp_path / "device.f32", snapshot, io=stages, fsync=False,
        cancel=SimpleNamespace(check=lambda: checks.append(None)),
    )
    assert (tmp_path / "device.f32").read_bytes() == (tmp_path / "streamed.f32").read_bytes()
    assert (write.sha256, dict(write.carrier_sha256)) == (streamed.sha256, dict(streamed.carrier_sha256))
    assert "arena.fsync" not in stages.stages and "arena.d2h" in stages.stages
    assert len(checks) == sum(-(-int(np.prod(shape)) // words) for shape in shapes.values())
    assert released == [True]
    snapshot.release()
    assert released == [True]
    with pytest.raises(RuntimeError, match="already written or released"):
        write_forecast_arena_snapshot_new(tmp_path / "again.f32", snapshot)
