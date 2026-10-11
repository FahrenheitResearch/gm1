from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import cm1_marks

from authority_trees import requires_cm1_tarball, requires_native_reference, requires_work
from cm1core import cuda_gm1_tilestream_solve3_pressure as pressure
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_binding(*, nx: int = 17, ny: int = 15, nk: int = 3):
    words = np.arange(nk * ny * nx, dtype=np.uint32).reshape(nk, ny, nx)
    ppx = np.ascontiguousarray((words ^ np.uint32(0x3F123456)).view(np.float32))
    pp3d = np.ascontiguousarray(
        ((words * np.uint32(2654435761)) | np.uint32(0x3E000000)).view(np.float32)
    )
    # Keep the deliberately varied raw words finite for the admission guard.
    ppx = np.ascontiguousarray(np.nan_to_num(ppx, nan=0.0, posinf=1.0, neginf=-1.0))
    pp3d = np.ascontiguousarray(np.nan_to_num(pp3d, nan=0.0, posinf=1.0, neginf=-1.0))
    return pressure.Gm1TiledSolve3PressureBinding.create(
        ppx=ppx,
        pp3d=pp3d,
        source_identity_sha256="1" * 64,
        post_wsm6_receipt_sha256="2" * 64,
        post_wsm6_semantic_sha256="3" * 64,
        post_wsm6_output_sha256="4" * 64,
        allow_compact_validation=True,
    )


@requires_cm1_tarball
@requires_native_reference
def test_authority_and_selected_source_anchor_are_exact() -> None:
    assert _sha(ROOT / "upstream/cm1-21.1/cm1r21.1.tar.gz") == (
        pressure.AUTHORITY_TARBALL_SHA256
    )
    source = ROOT / "work/native-reference/cm1r21.1/src/solve3.F"
    assert _sha(source) == pressure.AUTHORITY_SOLVE3_SHA256
    lines = source.read_text(encoding="ascii").splitlines()
    selected = "\n".join(lines[269:283])
    assert "psolver.eq.3" in selected
    # ppx gains pp3d (the psolver=3 pressure carry).
    assert cm1_marks.has(selected, "895d466ea7508bb9/33")
    assert "call bcp" not in selected.lower()
    assert pressure.AUTHORITY_ANCHORS == ("src/solve3.F:270-283",)


@requires_work("gm1-t300-child-auxiliary-v1")
def test_auxiliary_physical_ppx_pin_is_source_initialized() -> None:
    from cm1core.cuda_gm1_child_nm_def import load_gm1_child_auxiliary

    auxiliary = load_gm1_child_auxiliary(ROOT / "work/gm1-t300-child-auxiliary-v1")
    ppx = np.ascontiguousarray(
        auxiliary.field("terrain_open", "ppx")[1:76, 1:301, 1:301]
    )
    ppi = np.ascontiguousarray(
        auxiliary.field("terrain_open", "ppi")[1:76, 1:301, 1:301]
    )
    assert np.array_equal(ppx.view(np.uint32), ppi.view(np.uint32))
    assert hashlib.sha256(ppx.tobytes(order="C")).hexdigest() == (
        pressure.GM1_SOURCE_INITIALIZED_PHYSICAL_PPX_SHA256
    )
    assert auxiliary.manifest_sha256 == pressure.GM1_AUXILIARY_MANIFEST_SHA256
    assert auxiliary.terrain_open_sha256 == pressure.GM1_AUXILIARY_TERRAIN_OPEN_SHA256


def test_fill_cpu_scatter_partitions_every_physical_cell_once() -> None:
    binding = _compact_binding()
    expected = np.add(binding.ppx, binding.pp3d, dtype=np.float32)
    plan = build_gm1_open_tile_plan(nx=binding.nx, ny=binding.ny, tile_nx=8, tile_ny=7)
    counts = np.zeros((binding.ny, binding.nx), dtype=np.uint8)
    for spec in plan.specs:
        shapes = pressure.tiled_solve3_pressure_buffer_shapes(
            tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
        )
        tile = {
            name: np.empty(shape, dtype=np.float32) for name, shape in shapes.items()
        }
        fill = pressure.fill_gm1_tiled_solve3_pressure_source(binding, spec, tile)
        active_nk, active_ny, active_nx = fill["active_shape"]
        assert active_nk == binding.nk
        before_padding = tile["ppx"].copy()
        pressure.cpu_gm1_tiled_solve3_pressure_inplace(
            tile["ppx"],
            tile["pp3d"],
            active_nx=active_nx,
            active_ny=active_ny,
        )
        assert np.array_equal(
            tile["ppx"][:, active_ny:, :].view(np.uint32),
            before_padding[:, active_ny:, :].view(np.uint32),
        )
        assert np.array_equal(
            tile["ppx"][:, :active_ny, active_nx:].view(np.uint32),
            before_padding[:, :active_ny, active_nx:].view(np.uint32),
        )
        scattered = pressure.scatter_gm1_tiled_solve3_pressure_output(
            binding, spec, tile["ppx"]
        )
        assert scattered["padding_scattered"] is False
        y0, y1, x0, x1 = spec.owned_rect("mass")
        counts[y0:y1, x0:x1] += 1
    assert np.all(counts == 1)
    assert np.array_equal(binding.ppx.view(np.uint32), expected.view(np.uint32))


def test_alias_nonfinite_and_stale_real_identity_refuse() -> None:
    base = np.zeros((3, 9, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="must not alias"):
        pressure.Gm1TiledSolve3PressureBinding.create(
            ppx=base,
            pp3d=base,
            source_identity_sha256="1" * 64,
            post_wsm6_receipt_sha256="2" * 64,
            post_wsm6_semantic_sha256="3" * 64,
            post_wsm6_output_sha256="4" * 64,
            allow_compact_validation=True,
        )
    bad = base.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pressure.Gm1TiledSolve3PressureBinding.create(
            ppx=base.copy(),
            pp3d=bad,
            source_identity_sha256="1" * 64,
            post_wsm6_receipt_sha256="2" * 64,
            post_wsm6_semantic_sha256="3" * 64,
            post_wsm6_output_sha256="4" * 64,
            allow_compact_validation=True,
        )
    with pytest.raises(ValueError, match=r"\[75,300,300\]"):
        pressure.Gm1TiledSolve3PressureBinding.create(
            ppx=base.copy(),
            pp3d=base.copy(),
            source_identity_sha256="1" * 64,
            post_wsm6_receipt_sha256="2" * 64,
            post_wsm6_semantic_sha256="3" * 64,
            post_wsm6_output_sha256="4" * 64,
        )


def test_kernel_is_single_f32_add_and_has_no_lifecycle_side_effects() -> None:
    source = pressure.CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE
    assert "__fadd_rn(ppx[o], pp3d[o])" in source
    assert "atomic" not in source
    assert "qa" not in source
    assert "rho" not in source
    assert "bcp" not in source
    assert hashlib.sha256(source.encode("ascii")).hexdigest() == (
        pressure.CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE_SHA256
    )
    assert not hasattr(pressure, "launch_cuda_gm1_tiled_solve3_current_to_a")
    assert "no ppx boundary completion" in (
        pressure.GM1_TILESTREAM_SOLVE3_PRESSURE_CLAIM_LIMIT
    )


def test_compiler_requests_distinct_ptx_and_cubin_images() -> None:
    source = Path(pressure.__file__).read_text(encoding="utf-8")
    assert source.count('method="ptx"') == 1
    assert source.count('method="cubin"') == 1


def test_five_meter_inventory_is_incremental_and_pressure_global_stays_refused() -> (
    None
):
    inventory = pressure.five_meter_tiled_solve3_pressure_inventory()
    one_ppx = 75 * 192 * 192 * 4
    assert inventory["incremental_host_per_buffer_bytes"] == one_ppx
    assert inventory["incremental_device_per_buffer_bytes"] == one_ppx
    assert inventory["incremental_double_host_bytes"] == 2 * one_ppx
    assert inventory["incremental_double_device_bytes"] == 2 * one_ppx
    assert inventory["post_wsm6_combined_double_host_bytes"] == 555_319_296
    assert inventory["post_wsm6_combined_double_device_bytes"] == 799_211_520
    assert inventory["full_domain_ppx_duplicate_bytes"] == 0
    assert inventory["global_pressure_workspace_executed"] is False
    assert inventory["operation_b_current_to_a_available"] is False
    assert inventory["five_meter_meteorological_product"] is False
