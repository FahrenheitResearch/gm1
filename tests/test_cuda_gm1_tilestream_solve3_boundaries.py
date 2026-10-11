from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import cm1_marks

from authority_trees import requires_cm1_tarball, requires_native_reference
from cm1core import cuda_gm1_tilestream_solve3_boundaries as boundaries
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(*, nx: int = 17, ny: int = 15, nk: int = 3):
    k, j, i = np.indices((nk, ny, nx), dtype=np.int32)
    ppx = np.ascontiguousarray(
        np.float32(-0.002)
        + np.multiply(k + 2 * j - i, np.float32(1.0e-6), dtype=np.float32)
    )
    rho = np.ascontiguousarray(
        np.float32(1.0) + np.multiply(k - j + i, np.float32(2.0e-5), dtype=np.float32)
    )
    ppx_padded = np.full((nk + 2, ny + 2, nx + 2), np.float32(17.0))
    rho_padded = np.zeros((nk + 2, ny + 2, nx + 2), dtype=np.float32)
    return boundaries.Gm1TiledSolve3BoundaryBinding.create(
        ppx=ppx,
        rho=rho,
        ppx_padded=ppx_padded,
        rho_padded=rho_padded,
        source_identity_sha256="1" * 64,
        operation_a_receipt_file_sha256="2" * 64,
        operation_a_receipt_semantic_sha256="3" * 64,
        allow_compact_validation=True,
    )


def _fake_launch(spec, binding):
    receipt = {
        "schema": boundaries.CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SCHEMA,
        "status": "ENQUEUED_PPX_THEN_RHO_OPEN_BOUNDARY_TILE",
        "tile": spec.index,
        "source_identity_sha256": binding.source_identity_sha256,
        "operation_a_receipt_file_sha256": (binding.operation_a_receipt_file_sha256),
        "stream_pointer": 1234,
    }
    receipt["operation_token_sha256"] = hashlib.sha256(
        boundaries._canonical(receipt)
    ).hexdigest()
    return boundaries.CudaGm1TiledSolve3BoundaryLaunch(
        object(), object(), MappingProxyType(receipt)
    )


@requires_cm1_tarball
@requires_native_reference
def test_authority_seals_literal_ppx_then_rho_bcp_order() -> None:
    assert _sha(ROOT / "upstream/cm1-21.1/cm1r21.1.tar.gz") == (
        boundaries.AUTHORITY_TARBALL_SHA256
    )
    solve3 = ROOT / "work/native-reference/cm1r21.1/src/solve3.F"
    bc = ROOT / "work/native-reference/cm1r21.1/src/bc.F"
    assert _sha(solve3) == boundaries.AUTHORITY_SOLVE3_SHA256
    assert _sha(bc) == boundaries.AUTHORITY_BC_SHA256
    solve_lines = solve3.read_text(encoding="ascii").splitlines()
    selected = "\n".join(solve_lines[269:312])
    assert selected.index("call bcp(ppx)") < selected.index("call bcp(rho)")
    bc_lines = bc.read_text(encoding="ascii").splitlines()
    bcp = "\n".join(bc_lines[371:520])
    assert "subroutine bcp(p)" in bcp
    assert cm1_marks.has(bcp, "09de682ccf0ca14c/17")
    assert cm1_marks.has(bcp, "fe1c5c14d27ad07c/21")
    assert cm1_marks.has(bcp, "6b26669d129e5fb9/17")
    assert cm1_marks.has(bcp, "6755bc254bbd817d/21")


def test_cpu_tiled_boundary_matches_resident_and_emits_preheld_token() -> None:
    binding = _binding()
    ppx_expected = binding.ppx_padded.copy(order="C")
    rho_expected = binding.rho_padded.copy(order="C")
    ppx_expected[1:-1, 1:-1, 1:-1] = binding.ppx
    rho_expected[1:-1, 1:-1, 1:-1] = binding.rho
    boundaries.cpu_gm1_tiled_solve3_boundary_inplace(
        {"ppx": ppx_expected[1:-1], "rho": rho_expected[1:-1]},
        active_nx=binding.nx,
        active_ny=binding.ny,
        west=True,
        east=True,
        south=True,
        north=True,
    )
    vertical_before = (
        binding.ppx_padded[[0, -1]].copy(),
        binding.rho_padded[[0, -1]].copy(),
    )
    plan = build_gm1_open_tile_plan(nx=binding.nx, ny=binding.ny, tile_nx=8, tile_ny=7)
    launches = []
    for spec in plan.specs:
        shapes = boundaries.tiled_solve3_boundary_buffer_shapes(
            tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
        )
        tile = {
            name: np.empty(shape, dtype=np.float32) for name, shape in shapes.items()
        }
        fill = boundaries.fill_gm1_tiled_solve3_boundary_source(binding, spec, tile)
        _, active_ny, active_nx = fill["active_shape"]
        edges = fill["open_edges"]
        boundaries.cpu_gm1_tiled_solve3_boundary_inplace(
            tile,
            active_nx=active_nx,
            active_ny=active_ny,
            **edges,
        )
        scatter = boundaries.scatter_gm1_tiled_solve3_boundary_output(
            binding, spec, tile
        )
        assert scatter["vertical_halo_mutated"] is False
        launches.append(_fake_launch(spec, binding))
    assert np.array_equal(
        binding.ppx_padded.view(np.uint32), ppx_expected.view(np.uint32)
    )
    assert np.array_equal(
        binding.rho_padded.view(np.uint32), rho_expected.view(np.uint32)
    )
    assert np.array_equal(binding.ppx_padded[[0, -1]], vertical_before[0])
    assert np.array_equal(binding.rho_padded[[0, -1]], vertical_before[1])
    token = boundaries.finish_gm1_tiled_solve3_preheld_boundaries(
        binding, plan, launches
    )
    assert token.receipt["status"] == (
        "ADMITTED_PPX_RHO_BOUNDARIES_READY_FOR_HELD_RECAPTURE_ONLY"
    )
    assert token.receipt["ready_for"] == "held-rate recapture admission only"
    assert token.receipt["operation_b_current_to_a_available"] is False
    assert token.receipt["mpi_performed"] is False


def test_finish_refuses_missing_or_rotated_launch_tokens() -> None:
    binding = _binding()
    plan = build_gm1_open_tile_plan(nx=binding.nx, ny=binding.ny, tile_nx=8, tile_ny=7)
    with pytest.raises(ValueError, match="every exact tile"):
        boundaries.finish_gm1_tiled_solve3_preheld_boundaries(binding, plan, [])
    launches = [_fake_launch(spec, binding) for spec in plan.specs]
    changed = dict(launches[0].receipt)
    changed["operation_token_sha256"] = "0" * 64
    launches[0] = boundaries.CudaGm1TiledSolve3BoundaryLaunch(
        object(), object(), MappingProxyType(changed)
    )
    with pytest.raises(ValueError, match="operation token changed"):
        boundaries.finish_gm1_tiled_solve3_preheld_boundaries(binding, plan, launches)


def test_alias_nonfinite_and_stale_real_geometry_refuse() -> None:
    binding = _binding()
    with pytest.raises(ValueError, match="must not alias"):
        boundaries.Gm1TiledSolve3BoundaryBinding.create(
            ppx=binding.ppx,
            rho=binding.ppx,
            ppx_padded=binding.ppx_padded.copy(),
            rho_padded=binding.rho_padded.copy(),
            source_identity_sha256="1" * 64,
            operation_a_receipt_file_sha256="2" * 64,
            operation_a_receipt_semantic_sha256="3" * 64,
            allow_compact_validation=True,
        )
    bad = binding.rho.copy()
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        boundaries.Gm1TiledSolve3BoundaryBinding.create(
            ppx=binding.ppx.copy(),
            rho=bad,
            ppx_padded=binding.ppx_padded.copy(),
            rho_padded=binding.rho_padded.copy(),
            source_identity_sha256="1" * 64,
            operation_a_receipt_file_sha256="2" * 64,
            operation_a_receipt_semantic_sha256="3" * 64,
            allow_compact_validation=True,
        )
    with pytest.raises(ValueError, match=r"\[75,300,300\]"):
        boundaries.Gm1TiledSolve3BoundaryBinding.create(
            ppx=binding.ppx.copy(),
            rho=binding.rho.copy(),
            ppx_padded=binding.ppx_padded.copy(),
            rho_padded=binding.rho_padded.copy(),
            source_identity_sha256="1" * 64,
            operation_a_receipt_file_sha256="2" * 64,
            operation_a_receipt_semantic_sha256="3" * 64,
        )


def test_cuda_source_preserves_carrier_and_direction_order() -> None:
    source = Path(boundaries.__file__).read_text(encoding="utf-8")
    assert 'for carrier in (checked["ppx"], checked["rho"])' in source
    assert source.index("compiled.kernels[0](") < source.index("compiled.kernels[1](")
    cuda = boundaries.CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE
    assert "atomic" not in cuda
    assert "qpten" not in cuda
    assert "tkea" not in cuda
    assert hashlib.sha256(cuda.encode("ascii")).hexdigest() == (
        boundaries.CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE_SHA256
    )
    assert 'method="ptx"' in source and 'method="cubin"' in source


def test_five_meter_inventory_keeps_boundary_buffers_local() -> None:
    inventory = boundaries.five_meter_tiled_solve3_boundary_inventory()
    each = 2 * 75 * 194 * 194 * 4
    assert inventory["host_per_buffer_bytes"] == each
    assert inventory["device_per_buffer_bytes"] == each
    assert inventory["double_host_bytes"] == 2 * each
    assert inventory["double_device_bytes"] == 2 * each
    assert inventory["full_domain_padded_ppx_rho_duplicate_bytes"] == 0
    assert inventory["sequential_after_operation_a"] is True
    assert inventory["global_pressure_workspace_executed"] is False
    assert inventory["operation_b_current_to_a_available"] is False
    assert inventory["five_meter_meteorological_product"] is False
