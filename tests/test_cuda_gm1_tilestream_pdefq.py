from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.cuda_gm1_tilestream_pdefq import (
    CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_BYTES,
    CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_SHA256,
    GM1_TILESTREAM_PDEFQ_CLAIM_LIMIT,
    Gm1TiledPdefqSourceBinding,
    cpu_gm1_tiled_wsm6_pdefq_pdscheme1_inplace,
    fill_gm1_tiled_pdefq_source,
    five_meter_tiled_pdefq_inventory,
    scatter_gm1_tiled_pdefq_output,
    tiled_pdefq_buffer_shapes,
)
from cm1core.cuda_stretch_z1_terrain import cpu_build_stretch_z1_terrain
from cm1core.cuda_wsm6_pdefq import cpu_wsm6_pdefq_pdscheme1_inplace
from cm1core.stretch_z1_child_15km import build_stretch_z1_child_15km_grid


def _compact_source() -> tuple[Gm1TiledPdefqSourceBinding, np.ndarray]:
    nx, ny, nk = 20, 18, 75
    k, j, i = np.indices((nk, ny, nx), dtype=np.int32)
    terrain = np.ascontiguousarray(
        np.float32(210.0)
        + np.multiply(j[0], np.float32(0.375), dtype=np.float32)
        - np.multiply(i[0], np.float32(0.21875), dtype=np.float32),
        dtype=np.float32,
    )
    rho = np.ascontiguousarray(
        np.float32(0.8)
        + np.multiply(k + 1, np.float32(0.001953125), dtype=np.float32)
        + np.multiply(i + 2 * j, np.float32(0.00006103515625), dtype=np.float32)
    )
    q3d = np.empty((6, nk, ny, nx), dtype=np.float32)
    for species in range(6):
        q3d[species] = np.float32(2.0e-5 * (species + 1)) + np.multiply(
            (13 * i + 7 * j + 3 * k + species) % 17 - 8,
            np.float32(2.5e-6),
            dtype=np.float32,
        )
    q3d[0, 0, ::3, ::4] = np.float32(-2.0e-7)
    q3d[1:, 1, ::2, ::5] = np.float32(5.0e-15)
    grid = build_stretch_z1_child_15km_grid()
    binding = Gm1TiledPdefqSourceBinding.create(
        q3d=np.ascontiguousarray(q3d),
        rho=rho,
        terrain_msl=terrain,
        ruh=np.ones((nx,), dtype=np.float32),
        rvh=np.ones((ny,), dtype=np.float32),
        sigmaf=np.ascontiguousarray(grid.sigmaf),
        source_identity_sha256="1" * 64,
        allow_compact_validation=True,
    )
    asq = np.array(
        [-0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625], dtype=np.float64
    )
    return binding, asq


def test_cpu_tiled_carry_matches_full_domain_pdefq_raw_bits() -> None:
    tiled, tiled_asq = _compact_source()
    full_q = tiled.q3d.copy()
    full_asq = tiled_asq.copy()
    static = cpu_build_stretch_z1_terrain(
        tiled.terrain_msl, allow_compact_validation=True
    )
    full = {
        "q3d": full_q,
        "rho": tiled.rho.copy(),
        "ruh": tiled.ruh.copy(),
        "rvh": tiled.rvh.copy(),
        "rmh": static.arrays["rmh"].copy(),
        "asq": full_asq,
    }
    full_result = cpu_wsm6_pdefq_pdscheme1_inplace(
        full, ni=tiled.nx, nj=tiled.ny, nk=tiled.nk, dx=50.0, dy=50.0, dz=100.0
    )
    plan = build_gm1_open_tile_plan(
        nx=tiled.nx, ny=tiled.ny, tile_nx=8, tile_ny=7
    )
    tiled_result = cpu_gm1_tiled_wsm6_pdefq_pdscheme1_inplace(
        tiled, plan, tiled_asq, dx=50.0, dy=50.0, dz=100.0
    )
    assert np.array_equal(tiled.q3d.view(np.uint32), full_q.view(np.uint32))
    assert np.array_equal(tiled_asq.view(np.uint64), full_asq.view(np.uint64))
    assert np.array_equal(
        tiled_result["budget_columns"].view(np.uint64),
        full_result["budget_columns"].view(np.uint64),
    )
    assert np.all(tiled.q3d >= np.float32(0.0))
    assert np.all(np.isfinite(tiled_asq))
    assert np.any(tiled_asq < 0.0)


def test_tile_fill_and_scatter_publish_only_contract_owned_physical_cells() -> None:
    binding, _asq = _compact_source()
    plan = build_gm1_open_tile_plan(
        nx=binding.nx, ny=binding.ny, tile_nx=8, tile_ny=7
    )
    spec = plan.specs[-1]
    shapes = tiled_pdefq_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    target = {name: np.empty(shape, dtype=np.float32) for name, shape in shapes.items()}
    original = binding.q3d[5].copy()
    fill = fill_gm1_tiled_pdefq_source(binding, spec, target, species=5)
    y0, y1, x0, x1 = spec.owned_rect("mass")
    height, width = y1 - y0, x1 - x0
    assert fill["owned_yx"] == [y0, y1, x0, x1]
    assert np.array_equal(
        target["q3d"][:, :height, :width].view(np.uint32),
        original[:, y0:y1, x0:x1].view(np.uint32),
    )
    assert np.all(target["q3d"][:, height:, :] == np.float32(0.0))
    target["q3d"][:, :height, :width] = np.float32(0.25)
    scatter = scatter_gm1_tiled_pdefq_output(binding, spec, target["q3d"], species=5)
    assert scatter["owned_yx"] == [y0, y1, x0, x1]
    assert np.all(binding.q3d[5, :, y0:y1, x0:x1] == np.float32(0.25))
    expected = original.copy()
    expected[:, y0:y1, x0:x1] = np.float32(0.25)
    assert np.array_equal(
        binding.q3d[5].view(np.uint32), expected.view(np.uint32)
    )


def test_source_binding_refuses_nonuniform_metrics_and_wrong_profile() -> None:
    binding, _asq = _compact_source()
    ruh = binding.ruh.copy()
    ruh[3] = np.float32(1.0001)
    with pytest.raises(ValueError, match="uniform metrics"):
        Gm1TiledPdefqSourceBinding.create(
            q3d=binding.q3d.copy(),
            rho=binding.rho,
            terrain_msl=binding.terrain_msl,
            ruh=ruh,
            rvh=binding.rvh,
            sigmaf=binding.sigmaf,
            source_identity_sha256="1" * 64,
            allow_compact_validation=True,
        )
    sigmaf = binding.sigmaf.copy()
    sigmaf[5] = np.nextafter(sigmaf[5], np.float32(np.inf))
    with pytest.raises(ValueError, match="sealed stretch-z1"):
        Gm1TiledPdefqSourceBinding.create(
            q3d=binding.q3d.copy(),
            rho=binding.rho,
            terrain_msl=binding.terrain_msl,
            ruh=binding.ruh,
            rvh=binding.rvh,
            sigmaf=sigmaf,
            source_identity_sha256="1" * 64,
            allow_compact_validation=True,
        )


def test_kernel_and_five_meter_inventory_are_sealed_without_global_duplicates() -> None:
    assert hashlib.sha256(CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_BYTES).hexdigest() == (
        CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_SHA256
    )
    source = CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_BYTES.decode("utf-8")
    assert "budj=__dsub_rn(__dadd_rn(budj,a2),a1)" in source
    assert "reset_west?0.0:budgets" in source
    inventory = five_meter_tiled_pdefq_inventory()
    assert inventory["domain"] == [75, 3000, 3000]
    assert inventory["tile_count"] == 256
    assert inventory["per_host_or_device_buffer_bytes"] == 22_267_392
    assert inventory["double_host_buffers_bytes"] == 44_534_784
    assert inventory["double_device_buffers_bytes"] == 44_534_784
    assert inventory["global_budget_f64_bytes"] == 144_000
    assert inventory["global_sigmaf_f32_bytes"] == 312
    assert inventory["full_domain_rmh_duplicate_bytes"] == 0
    assert inventory["full_domain_q3d_duplicate_bytes"] == 0
    assert inventory["pressure_workspace_executed"] is False
    assert inventory["wsm6_executed"] is False
    assert "no qa mutation, WSM6, solve3, pressure" in GM1_TILESTREAM_PDEFQ_CLAIM_LIMIT
