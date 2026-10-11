from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from cm1core.cuda_memory_inventory import (
    shape_bytes,
    snapshot_cuda_allocations,
)


class DeviceView:
    """Ownership metadata only; attempts to read payloads fail the test."""

    def __init__(self, owner, *, offset=0, words=4, device=0):
        self.data = SimpleNamespace(mem=owner, ptr=owner.ptr + offset)
        self.device = SimpleNamespace(id=device)
        self.shape = (words,)
        self.nbytes = words * 4
        self.dtype = "float32"

    @property
    def __cuda_array_interface__(self):
        raise AssertionError("the inventory must not request/synchronize array exports")


@dataclass
class Handoff:
    current: object
    prior: object = None


def test_overlapping_views_and_repeated_handoffs_count_owner_once():
    owner = SimpleNamespace(ptr=4096, size=1024)
    current = DeviceView(owner, words=128)
    interior = DeviceView(owner, offset=64, words=32)
    prior = Handoff(current)
    prior.prior = prior  # A handoff cycle must not cause recursion or recounting.
    result = snapshot_cuda_allocations(
        {"live": Handoff(interior, prior), "other": {"same": current}}, label="rk2"
    )
    assert result["allocation_count"] == 1
    assert result["allocation_bytes"] == 1024
    assert result["distinct_view_count"] == 2
    assert result["distinct_view_bytes"] == 640
    assert result["roots"]["live"]["exclusive_bytes"] == 0
    assert result["roots"]["other"]["exclusive_bytes"] == 0
    assert result["allocations"][0]["roots"] == ["live", "other"]


def test_device_identity_prevents_equal_address_dedup_and_exposes_dead_root():
    owner = SimpleNamespace(ptr=4096, size=1024)
    a = DeviceView(owner, device=0)
    b = DeviceView(owner, device=1)
    result = snapshot_cuda_allocations({"old": a, "live": b}, label="final")
    assert result["allocation_bytes"] == 2048
    assert result["roots"]["old"]["exclusive_bytes"] == 1024
    assert result["roots"]["live"]["exclusive_bytes"] == 1024


def test_owner_metadata_conflict_and_unknown_owner_refuse_false_totals():
    a = DeviceView(SimpleNamespace(ptr=4096, size=1024))
    b = DeviceView(SimpleNamespace(ptr=4096, size=2048))
    with pytest.raises(ValueError, match="conflicting"):
        snapshot_cuda_allocations({"live": [a, b]}, label="bad")
    del a.data.mem
    with pytest.raises(TypeError, match="ownership is unavailable"):
        snapshot_cuda_allocations({"live": a}, label="bad")


def test_shape_count_does_not_allocate_requested_grid():
    assert shape_bytes((237, 400, 400)) == 151_680_000
    assert shape_bytes((6, 237, 400, 400)) == 910_080_000
    with pytest.raises(ValueError, match="nonnegative"):
        shape_bytes((237, -400, 400))
