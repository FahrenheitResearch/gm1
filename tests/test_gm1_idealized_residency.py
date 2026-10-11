from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.gm1_idealized_residency import (
    DeviceBufferRegistry,
    IdealizedSource,
    SourceRequest,
)
from cm1core.grid_geometry import CartesianGridGeometry


class SourceOwner:
    def __init__(self, _library, _sounding, *, iinit, geometry):
        nz, ny, nx = geometry.shape_zyx
        self.closed = False
        self.groups = {
            "terrain_open": {
                "th0": np.broadcast_to(
                    np.arange(nz + 2, dtype=np.float32)[:, None, None],
                    (nz + 2, ny + 6, nx + 6),
                ).copy(),
                "u3d": np.arange((nz + 2) * (ny + 6) * (nx + 7), dtype=np.float32).reshape(nz + 2, ny + 6, nx + 7),
                "pp3d": np.arange((nz + 2) * (ny + 2) * (nx + 2), dtype=np.float32).reshape(nz + 2, ny + 2, nx + 2),
                "rf0": np.arange((nz + 2) * (ny + 6) * (nx + 6), dtype=np.float32).reshape(nz + 2, ny + 6, nx + 6),
            },
            "moist": {
                "q3d": np.arange(6 * (nz + 2) * (ny + 6) * (nx + 6), dtype=np.float32).reshape(6, nz + 2, ny + 6, nx + 6),
            },
        }

    def close(self):
        self.closed = True


class Stream:
    def __init__(self):
        self.synchronized = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def synchronize(self):
        self.synchronized += 1


class Allocator:
    float32 = np.float32

    def __init__(self):
        self.calls = 0

    def asarray(self, host, *, dtype, order):
        self.calls += 1
        return np.array(host, dtype=dtype, order=order, copy=True)

    def empty(self, shape, *, dtype):
        self.calls += 1
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, *, dtype):
        self.calls += 1
        return np.zeros(shape, dtype=dtype)


def source():
    return IdealizedSource(
        "unused.dll", b"source-owned sounding", CartesianGridGeometry(11, 13, 17, 75, 75, 75),
        owner_factory=SourceOwner,
    )


def test_native_halo_families_and_species_crop_preserve_exact_words():
    with source() as initial:
        for key, selection in (
            ("terrain_open.u3d", (slice(1, 18), slice(3, 16), slice(3, 15))),
            ("terrain_open.pp3d", (slice(1, 18), slice(1, 14), slice(1, 12))),
            ("moist.q3d", (slice(None), slice(1, 18), slice(3, 16), slice(3, 14))),
            ("terrain_open.rf0", (slice(1, 19), slice(3, 16), slice(3, 14))),
        ):
            actual = initial.field(key, layout="physical")
            expected = initial.field(key)[selection]
            assert np.shares_memory(actual, expected)
            assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_profile_compression_refuses_horizontal_difference_including_signed_zero():
    with source() as initial:
        actual = initial.field("terrain_open.th0", layout="profile")
        assert np.array_equal(actual, np.arange(1, 18, dtype=np.float32))
        assert not actual.flags.writeable
    with source() as initial:
        field = initial.field("terrain_open.th0")
        field[1, 3:16, 3:14] = np.float32(0.0)
        field[1, 5, 5] = np.float32(-0.0)
        with pytest.raises(ValueError, match="varies horizontally"):
            initial.field("terrain_open.th0", layout="profile")


def test_budget_refusal_precedes_allocator_and_mutable_copies_are_distinct():
    allocator, stream = Allocator(), Stream()
    registry = DeviceBufferRegistry(allocator, stream, budget_bytes=512)
    with source() as initial:
        with pytest.raises(MemoryError, match="before any CUDA"):
            registry.upload(initial, [SourceRequest("q3d", "moist.q3d", "physical")])
        assert allocator.calls == 0
        requests = [
            SourceRequest("a", "terrain_open.th0", "profile"),
            SourceRequest("current", "terrain_open.th0", "profile"),
        ]
        registry = DeviceBufferRegistry(allocator, stream, budget_bytes=1024)
        values = registry.upload(initial, requests)
        assert not np.shares_memory(values["a"], values["current"])
        with pytest.raises(ValueError, match="read-only"):
            registry.alias_read_only("bad", "a")
    assert stream.synchronized == 1
    assert initial._owner.closed
    with pytest.raises(RuntimeError, match="closed"):
        initial.field("terrain_open.th0")


def test_readonly_alias_retains_budget_until_last_name_is_released():
    allocator, stream = Allocator(), Stream()
    registry = DeviceBufferRegistry(allocator, stream, budget_bytes=512)
    with source() as initial:
        registry.upload(initial, [SourceRequest("base", "terrain_open.th0", "profile", True)])
    same = registry.alias_read_only("same-base", "base")
    assert same is registry.buffers["base"]
    registry.release(["base"])
    assert registry.registered_allocation_bytes == 512
    with pytest.raises(ValueError, match="already registered"):
        registry.allocate("base", (17,))
    with pytest.raises(ValueError, match="unique"):
        registry.release(["same-base", "same-base"])
    registry.release(["same-base"])
    assert registry.registered_allocation_bytes == 0
    registry.allocate("base", (17,))
    assert registry.registered_allocation_bytes == 512


def test_float64_and_integer_workspace_bytes_use_their_actual_itemsize():
    allocator, stream = Allocator(), Stream()
    registry = DeviceBufferRegistry(allocator, stream, budget_bytes=1536)
    values = registry.allocate("mass", (100,), dtype=np.float64)
    assert values.dtype == np.float64
    assert registry.registered_allocation_bytes == 1024
    counter = registry.allocate("invalid", (1,), dtype=np.uint32, zero=True)
    assert counter.dtype == np.uint32
    assert registry.registered_allocation_bytes == 1536
