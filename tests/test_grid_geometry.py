from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from cm1core.grid_geometry import CartesianGridGeometry


@pytest.mark.parametrize("stagger", ("mass", "u", "v", "w"))
@pytest.mark.parametrize("halo_xy", (1, 3))
def test_source_family_extraction_excludes_halos_and_keeps_extra_faces(
    stagger: str, halo_xy: int
) -> None:
    grid = CartesianGridGeometry(17, 23, 237, 75.0, 90.0, 75.0)
    actual = np.full(grid.padded_shape(stagger, halo_xy=halo_xy, halo_z=1), -999)
    expected_shape = {
        "mass": (237, 23, 17), "u": (237, 23, 18),
        "v": (237, 24, 17), "w": (238, 23, 17),
    }[stagger]
    nz, ny, nx = expected_shape
    expected = np.arange(nz * ny * nx).reshape(expected_shape)
    # Independent known CM1 storage offset, rather than using the extraction
    # helper to put the data where that same helper will look for it.
    actual[1:1 + nz, halo_xy:halo_xy + ny, halo_xy:halo_xy + nx] = expected
    extracted = actual[grid.physical_slices(stagger, halo_xy=halo_xy, halo_z=1)]
    np.testing.assert_array_equal(extracted, expected)
    assert extracted.shape == expected_shape
    assert extracted[-1, -1, -1] == expected.size - 1


def test_uniform_extent_and_legacy_geometry_are_exact_and_immutable() -> None:
    legacy = CartesianGridGeometry(300, 300, 75, 50.0, 50.0, 200.0)
    assert legacy.shape_zyx == legacy.mass_shape == (75, 300, 300)
    assert legacy.u_shape == (75, 300, 301)
    assert legacy.v_shape == (75, 301, 300)
    assert legacy.w_shape == (76, 300, 300)
    assert legacy.padded_shape("w", halo_xy=3, halo_z=1) == (78, 306, 306)
    assert legacy.physical_slices("u", halo_xy=3, halo_z=1) == (
        slice(1, 76), slice(3, 303), slice(3, 304)
    )
    assert legacy.spacing_m == (legacy.dx, legacy.dy, legacy.dz) == (50.0, 50.0, 200.0)
    assert legacy.ztop_m == 15000.0
    fine = CartesianGridGeometry(400, 400, 237, 75, 75, 75)
    assert fine.ztop_m == 17775.0
    assert fine.record()["nz"] == 237
    assert fine.surface_shape == (400, 400)
    with pytest.raises(FrozenInstanceError):
        fine.nz = 75


@pytest.mark.parametrize(
    "change",
    ({"nx": True}, {"ny": 1.5}, {"nz": 0}, {"nx": 1 << 32},
     {"dx_m": 0}, {"dy_m": float("nan")}, {"dz_m": float("inf")},
     {"dz_m": -75}, {"dx_m": True}, {"dy_m": "75"}, {"dz_m": 1e308}),
)
def test_invalid_dimensions_or_spacing_refuse(change: dict[str, object]) -> None:
    grid = CartesianGridGeometry(17, 23, 237, 75, 75, 75)
    with pytest.raises((TypeError, ValueError)):
        replace(grid, **change)


def test_stencil_and_unknown_storage_family_refuse() -> None:
    grid = CartesianGridGeometry(7, 9, 4, 75, 75, 75)
    grid.require_stencil(radius_xy=3, minimum_nz=4)
    with pytest.raises(ValueError, match="horizontal.*stencil"):
        replace(grid, nx=6).require_stencil(radius_xy=3, minimum_nz=4)
    with pytest.raises(ValueError, match="vertical stencil"):
        replace(grid, nz=3).require_stencil(radius_xy=3, minimum_nz=4)
    with pytest.raises(ValueError, match="unknown C-grid stagger"):
        grid.physical_shape("pressure")
    with pytest.raises(ValueError, match="halo_xy"):
        grid.physical_slices(halo_xy=-1, halo_z=1)
    with pytest.raises(TypeError):
        grid.physical_slices()
