from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cm1core.gm1_restart_grid_interp import (
    ARWEN_COMPONENT_POLICY_SEAM,
    CM1_A_TO_CURRENT_COPY,
    CM1_DIRECT_RESTART_READ,
    CM1_FRESH_ZERO_AFTER_GRID_INTERP,
    CM1_REINITIALIZE_AFTER_GRID_INTERP,
    build_cm1_restart_axis_map,
    build_gm1_fixed_child_restart_maps,
    cm1_do_interp_point,
    validate_cm1_restart_interp_authority,
)
from authority_trees import requires_native_reference


ROOT = Path(__file__).resolve().parents[1]
# CM1_AUTHORITY_ROOT arrives here as a symlink (authority_trees.link_supplied_assets),
# and the validator refuses a symlinked root by design; hand it the tree itself.
AUTHORITY = (ROOT / "work/native-reference/cm1r21.1").resolve()


def test_import_is_cuda_cold() -> None:
    command = (
        "import sys; import cm1core.gm1_restart_grid_interp; "
        "assert not any(name == 'cupy' or name.startswith('cupy.') "
        "for name in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


@requires_native_reference
def test_authority_and_fixed_child_maps_are_exact() -> None:
    assert validate_cm1_restart_interp_authority(AUTHORITY) == AUTHORITY.resolve()
    maps = build_gm1_fixed_child_restart_maps()
    assert maps.receipt["status"] == "PASS_EXACT_CM1_RESTART_GRID_MAPS"
    assert maps.scalar_sigma.below_support_count == 5
    assert maps.w_sigmaf.below_support_count == 0
    np.testing.assert_array_equal(maps.scalar_sigma.lower[:5], 0)
    np.testing.assert_array_equal(maps.scalar_sigma.upper[:5], 0)
    np.testing.assert_array_equal(
        maps.scalar_sigma.fraction[:5].view(np.uint32), np.uint32(0)
    )
    assert float(maps.scalar_x.fraction[0]) == 0.25
    assert float(maps.u_face_x.fraction[1]) == 1.0
    assert int(maps.u_face_x.lower[1]) == 149
    assert int(maps.u_face_x.upper[1]) == 150


def test_below_support_duplicates_first_plane_not_slope() -> None:
    first = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    second = np.array([[110.0, 120.0], [130.0, 140.0]], dtype=np.float32)
    result = cm1_do_interp_point(
        first,
        first,
        ix=0,
        iy=0,
        rx=np.float32(0.25),
        ry=np.float32(0.75),
        rz=np.float32(0.55),
    )
    expected = cm1_do_interp_point(
        first,
        first,
        ix=0,
        iy=0,
        rx=np.float32(0.25),
        ry=np.float32(0.75),
        rz=np.float32(0.0),
    )
    invented_slope = cm1_do_interp_point(
        first,
        second,
        ix=0,
        iy=0,
        rx=np.float32(0.25),
        ry=np.float32(0.75),
        rz=np.float32(0.55),
    )
    assert result.view(np.uint32) == expected.view(np.uint32)
    assert result.view(np.uint32) != invented_slope.view(np.uint32)


def test_weight_rhs_is_real_before_double_assignment() -> None:
    low = np.array(
        [[487758.21875, -140339.359375], [-394395.75, -988219.9375]],
        dtype=np.float32,
    )
    high = np.array(
        [[512957.9375, -844848.0625], [-20023.923828125, -391277.8125]],
        dtype=np.float32,
    )
    result = cm1_do_interp_point(
        low,
        high,
        ix=0,
        iy=0,
        rx=np.float32(0.840822160243988),
        ry=np.float32(0.9504758715629578),
        rz=np.float32(0.3188745677471161),
    )
    # Exact source-order word when the REAL RHS is rounded before assignment
    # to DOUBLE PRECISION w1..w8.  Prematurely widening rx/ry/rz gives c92896dd.
    assert result.view(np.uint32) == np.uint32(0xC92896DC)
    assert result.view(np.uint32) != np.uint32(0xC92896DD)


def test_cross_language_rust_vector_is_raw_bit_exact() -> None:
    bits = np.array(
        [
            0x3F800001,
            0x40000001,
            0x40400001,
            0x40800001,
            0x40A00001,
            0x40C00001,
            0x40E00001,
            0x41000001,
        ],
        dtype=np.uint32,
    )
    value = bits.view(np.float32)
    low = np.array([[value[0], value[1]], [value[2], value[6]]], dtype=np.float32)
    high = np.array([[value[3], value[4]], [value[5], value[7]]], dtype=np.float32)
    result = cm1_do_interp_point(
        low,
        high,
        ix=0,
        iy=0,
        rx=np.float32(0.25),
        ry=np.float32(0.75),
        rz=np.float32(0.55),
    )
    assert result.view(np.uint32) == np.uint32(0x40983334)


def test_strict_less_than_exact_match_uses_fraction_one() -> None:
    old = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    new = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    mapping = build_cm1_restart_axis_map(
        old, new, allow_constant_extension=False
    )
    assert (int(mapping.lower[1]), int(mapping.upper[1])) == (0, 1)
    assert mapping.fraction[1].view(np.uint32) == np.float32(1.0).view(np.uint32)


def test_source_order_carrier_disposition_refuses_old_29_remap_assumption() -> None:
    assert CM1_DIRECT_RESTART_READ == ("ua", "va", "wa", "ppi", "ppx", "tha", "qa")
    assert dict(CM1_A_TO_CURRENT_COPY) == {
        "u3d": "ua",
        "v3d": "va",
        "w3d": "wa",
        "pp3d": "ppi",
        "th3d": "tha",
        "q3d": "qa",
    }
    assert set(CM1_FRESH_ZERO_AFTER_GRID_INTERP) == {
        "held.qpten",
        "held.qtten",
        "held.qvten",
        "held.qcten",
    }
    assert CM1_REINITIALIZE_AFTER_GRID_INTERP == ("closure.tkea",)
    assert len(ARWEN_COMPONENT_POLICY_SEAM) == 11


def test_tampered_or_outside_maps_fail_closed(tmp_path: Path) -> None:
    authority = tmp_path / "cm1r21.1"
    source = authority / "src"
    source.mkdir(parents=True)
    (source / "restart_read.F").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs"):
        validate_cm1_restart_interp_authority(authority)
    with pytest.raises(ValueError, match="exceed"):
        build_cm1_restart_axis_map(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-0.1, 0.5], dtype=np.float32),
            allow_constant_extension=False,
        )
