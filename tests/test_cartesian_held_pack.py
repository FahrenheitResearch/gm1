from contextlib import nullcontext

import numpy as np
import pytest

from cm1core.cuda_cartesian_dynamics import pack_cartesian_held_momentum
from cm1core.grid_geometry import CartesianGridGeometry


@pytest.mark.parametrize("nz", (8, 237))
def test_held_pack_exact_physical_faces_and_zero_common_fringes_with_guarded_allocations(nz):
    g = CartesianGridGeometry(13, 11, nz, 125, 125, 125)
    held, padded, owners, expected = {}, {}, {}, {}
    guard = np.uint32(0xDEADBEEF)
    for name, stagger in (("uten", "u"), ("vten", "v"), ("wten", "w")):
        shape = g.physical_shape(stagger)
        target_shape = g.padded_shape(stagger, halo_xy=3, halo_z=1)
        n, m = int(np.prod(shape)), int(np.prod(target_shape))
        source = np.full(n + 64, guard, dtype=np.uint32)
        source[:n] = (np.arange(n, dtype=np.uint32) & np.uint32(0x007FFFFF)) | np.uint32(0x3F000000)
        source[0] = np.uint32(0x80000000)
        target = np.full(m + 64, guard, dtype=np.uint32)
        held[name] = source[:n].view(np.float32).reshape(shape)
        padded[name] = target[:m].view(np.float32).reshape(target_shape)
        words = np.zeros(target_shape, dtype=np.uint32)
        words[g.physical_slices(stagger, halo_xy=3, halo_z=1)] = source[:n].reshape(shape)
        owners[name], expected[name] = (source, target), words
    pack_cartesian_held_momentum(np, held, padded, g, stream=nullcontext())
    for name, (source, target) in owners.items():
        assert np.array_equal(padded[name].view(np.uint32), expected[name])
        assert np.all(source[-64:] == guard)
        assert np.all(target[-64:] == guard)


def test_layout_error_or_overlap_refuses_before_any_destination_mutation():
    g = CartesianGridGeometry(13, 11, 8, 125, 125, 125)
    held = {name: np.ones(g.physical_shape(stagger), dtype=np.float32) for name, stagger in (("uten", "u"), ("vten", "v"), ("wten", "w"))}
    padded = {name: np.full(g.padded_shape(stagger, halo_xy=3, halo_z=1), 99, dtype=np.float32) for name, stagger in (("uten", "u"), ("vten", "v"), ("wten", "w"))}
    bad = dict(held)
    bad["vten"] = bad["vten"][:-1]
    with pytest.raises(ValueError, match="layouts"):
        pack_cartesian_held_momentum(np, bad, padded, g, stream=nullcontext())
    assert all(np.all(value == 99) for value in padded.values())
    bad = dict(held)
    bad["uten"] = padded["uten"].reshape(-1)[:held["uten"].size].reshape(held["uten"].shape)
    with pytest.raises(ValueError, match="overlap"):
        pack_cartesian_held_momentum(np, bad, padded, g, stream=nullcontext())
    assert all(np.all(value == 99) for value in padded.values())
