from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cm1core.layout import (
    CORE_LAYOUTS,
    GRID_COORDINATES,
    STRUCTURAL_DYCORE_NAMES,
    TARGET_HALO,
    GridShape,
    Halo,
    NativeRange,
    Stagger,
    layout,
)


def test_c_grid_interior_shapes_are_explicit_zyx() -> None:
    grid = GridShape(120, 120, 40)
    assert grid.interior(Stagger.SCALAR) == (40, 120, 120)
    assert grid.interior(Stagger.X_FACE) == (40, 120, 121)
    assert grid.interior(Stagger.Y_FACE) == (40, 121, 120)
    assert grid.interior(Stagger.Z_FACE) == (41, 120, 120)


def test_halo_width_cannot_be_negative() -> None:
    with pytest.raises(ValueError):
        Halo(3, 3, -1)


def test_native_allocation_bounds_and_confirmed_slices_are_exact() -> None:
    grid = GridShape(16, 16, 16)

    scalar = grid.geometry(Stagger.SCALAR, TARGET_HALO)
    assert scalar.allocation.x == NativeRange(-2, 19)
    assert scalar.allocation.y == NativeRange(-2, 19)
    assert scalar.allocation.z == NativeRange(0, 17)
    assert scalar.allocated_shape_zyx == (18, 22, 22)
    assert scalar.confirmed_slices_zyx == (
        slice(1, 17),
        slice(3, 19),
        slice(3, 19),
    )

    x_face = grid.geometry(Stagger.X_FACE, TARGET_HALO)
    y_face = grid.geometry(Stagger.Y_FACE, TARGET_HALO)
    z_face = grid.geometry(Stagger.Z_FACE, TARGET_HALO)
    assert x_face.allocated_shape_zyx == (18, 22, 23)
    assert y_face.allocated_shape_zyx == (18, 23, 22)
    assert z_face.allocated_shape_zyx == (19, 22, 22)
    assert x_face.confirmed_interior.x == NativeRange(1, 17)
    assert y_face.confirmed_interior.y == NativeRange(1, 17)
    assert z_face.confirmed_interior.z == NativeRange(1, 17)


def test_horizontal_coordinate_allocations_preserve_extra_faces() -> None:
    grid = GridShape(16, 16, 16)
    bounds = {
        row.native_name: row.native_bounds(grid, TARGET_HALO)
        for row in GRID_COORDINATES
    }
    assert bounds == {
        "xh": NativeRange(-2, 19),
        "xf": NativeRange(-2, 20),
        "yh": NativeRange(-2, 19),
        "yf": NativeRange(-2, 20),
    }


def test_structural_inventory_excludes_unresolved_moisture_and_sgs() -> None:
    assert STRUCTURAL_DYCORE_NAMES == (
        "ua",
        "u3d",
        "va",
        "v3d",
        "wa",
        "w3d",
        "ppi",
        "pp3d",
        "tha",
        "th3d",
    )
    assert {"qa", "q3d", "tkea", "tke3d"}.isdisjoint(STRUCTURAL_DYCORE_NAMES)


def test_a_and_current_3d_families_have_distinct_live_policies() -> None:
    for carrier, current in (
        ("ua", "u3d"),
        ("va", "v3d"),
        ("wa", "w3d"),
        ("ppi", "pp3d"),
        ("tha", "th3d"),
        ("qa", "q3d"),
        ("tkea", "tke3d"),
    ):
        assert layout(carrier).live_region != layout(current).live_region
        assert layout(carrier).stagger == layout(current).stagger


def test_code_layouts_map_exact_native_names_to_state_record() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "src" / "cm1core" / "records" / "CM1-STATE-MAP.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    names = {row["state_name"] for row in rows}
    assert all("|" not in name for name in names)
    assert {field.native_name for field in CORE_LAYOUTS} <= names
    assert {"pi0", "rho0", "th0", "xh", "xf", "nrk", "acoustic_substep"} <= names
