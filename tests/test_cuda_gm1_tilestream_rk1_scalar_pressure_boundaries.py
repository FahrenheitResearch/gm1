from __future__ import annotations

import hashlib
from types import MappingProxyType

import numpy as np
import pytest

from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.cuda_gm1_tilestream_post_acoustic_getdivx import (
    Gm1PostSoundVelocityToken,
)
from cm1core.cuda_gm1_rk1_post_scalar_momentum_boundaries import (
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_SCHEMA,
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_STATUS,
    Gm1Rk1PostScalarMomentumBoundaryToken,
)
from cm1core.cuda_gm1_tilestream_rk1_scalar_pressure_boundaries import (
    AUTHORITY_BC_SHA256,
    AUTHORITY_SOLVE2_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CM1_NUMQ,
    CM1_OPEN_NGXY,
    CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_RK1_BOUNDARY_KERNEL_NAMES,
    CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE,
    CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE_SHA256,
    Gm1TiledRk1ScalarPressureBoundaryBinding,
    _canonical,
    _corner_sources,
    _operation_token,
    _verify_bcp,
    _verify_bcs,
    _vertical_halo_sha,
    checkpoint_gm1_rk1_post_sound_pressure_density,
    cpu_gm1_tiled_rk1_scalar_pressure_boundaries_inplace,
    fill_gm1_tiled_rk1_scalar_pressure_boundary_source,
    five_meter_rk1_scalar_pressure_boundary_inventory,
    scatter_gm1_tiled_rk1_scalar_pressure_boundary_output,
    tiled_rk1_scalar_pressure_boundary_buffer_shapes,
)
from cm1core.cuda_gm1_tilestream_scalar_moist_rk1 import (
    CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS,
    Gm1TiledScalarMoistRk1Token,
)
from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)


def _velocity_token(*, stream_pointer: int = 7123) -> Gm1PostSoundVelocityToken:
    receipt: dict[str, object] = {
        "schema": "gm1.post-sound-averaged-velocity-token/v1",
        "status": "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT",
        "nrk": 1,
        "stage_identity_sha256": "a" * 64,
        "solve1_tke_rhs_token_sha256": "b" * 64,
        "psolver_receipt_sha256": "c" * 64,
        "stream_pointer": stream_pointer,
        "host_checkpoint_transport": True,
        "velocity_shapes": {},
        "velocity_sha256": {},
        "sound_output_semantics": "small-step-averaged velocities before getdivx",
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "authority": {},
        "claim_limit": "test exact type",
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1PostSoundVelocityToken(MappingProxyType(receipt))


def _physical(*, nk: int, ny: int, nx: int) -> dict[str, np.ndarray]:
    words = nk * ny * nx
    base = np.arange(words, dtype=np.float32).reshape(nk, ny, nx)
    return {
        "rho": np.ascontiguousarray(np.float32(1.0) + base * np.float32(1.0e-6)),
        "pp3d": np.ascontiguousarray(base * np.float32(2.0e-7) - np.float32(0.01)),
        "th3d": np.ascontiguousarray(base * np.float32(3.0e-6)),
        "q3d": np.ascontiguousarray(
            np.stack(
                [
                    base * np.float32(1.0e-8 * (species + 1))
                    for species in range(CM1_NUMQ)
                ]
            )
        ),
    }


def _padded(*, nk: int, ny: int, nx: int) -> dict[str, np.ndarray]:
    ng = CM1_OPEN_NGXY
    result = {
        "rho": np.empty((nk + 2, ny + 2, nx + 2), dtype=np.float32),
        "pp3d": np.empty((nk + 2, ny + 2, nx + 2), dtype=np.float32),
        "th3d": np.empty((nk + 2, ny + 2 * ng, nx + 2 * ng), dtype=np.float32),
        "q3d": np.empty((CM1_NUMQ, nk + 2, ny + 2 * ng, nx + 2 * ng), dtype=np.float32),
    }
    for index, value in enumerate(result.values(), start=1):
        value.flat[:] = np.arange(value.size, dtype=np.float32) * np.float32(
            index * 1.0e-7
        ) + np.float32(index * 0.125)
    return result


def _direct_binding(
    physical: dict[str, np.ndarray], padded: dict[str, np.ndarray]
) -> Gm1TiledRk1ScalarPressureBoundaryBinding:
    nk, ny, nx = physical["rho"].shape
    return Gm1TiledRk1ScalarPressureBoundaryBinding(
        MappingProxyType(physical),
        MappingProxyType(padded),
        nx,
        ny,
        nk,
        7123,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "f" * 64,
        MappingProxyType(
            {name: _vertical_halo_sha(value) for name, value in padded.items()}
        ),
        MappingProxyType(
            {
                name: _corner_sources(padded[name], ny=ny, nx=nx)
                for name in ("th3d", "q3d")
            }
        ),
        True,
    )


def test_authority_and_cuda_source_are_pinned() -> None:
    assert AUTHORITY_TARBALL_SHA256 == (
        "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
    )
    assert AUTHORITY_SOLVE2_SHA256 == (
        "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
    )
    assert AUTHORITY_BC_SHA256 == (
        "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
    )
    assert (
        hashlib.sha256(
            CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE.encode("ascii")
        ).hexdigest()
        == CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE_SHA256
    )
    assert all(
        name in CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE
        for name in CUDA_GM1_TILESTREAM_RK1_BOUNDARY_KERNEL_NAMES
    )


def test_pressure_checkpoint_binds_exact_post_sound_words() -> None:
    physical = _physical(nk=3, ny=8, nx=9)
    token = checkpoint_gm1_rk1_post_sound_pressure_density(
        _velocity_token(),
        {"rho": physical["rho"], "pp3d": physical["pp3d"]},
        allow_compact_validation=True,
    )
    assert token.receipt["schema"] == (
        CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_SCHEMA
    )
    assert token.receipt["selected_rk1_pressure_density_recompute_expected"] is False
    assert token.receipt["output_sha256"] == {
        "rho": hashlib.sha256(physical["rho"].tobytes()).hexdigest(),
        "pp3d": hashlib.sha256(physical["pp3d"].tobytes()).hexdigest(),
    }
    assert token.receipt["operation_token_sha256"] == _operation_token(token.receipt)
    bad = physical["rho"].copy()
    bad[0, 0, 0] = np.float32(0.0)
    with pytest.raises(ValueError, match="positive"):
        checkpoint_gm1_rk1_post_sound_pressure_density(
            _velocity_token(),
            {"rho": bad, "pp3d": physical["pp3d"]},
            allow_compact_validation=True,
        )


def test_binding_requires_exact_post_scalar_momentum_chronology() -> None:
    nk, ny, nx = 3, 8, 9
    physical = _physical(nk=nk, ny=ny, nx=nx)
    padded = _padded(nk=nk, ny=ny, nx=nx)
    velocity = _velocity_token()
    pressure = checkpoint_gm1_rk1_post_sound_pressure_density(
        velocity,
        {"rho": physical["rho"], "pp3d": physical["pp3d"]},
        allow_compact_validation=True,
    )
    flux_receipt: dict[str, object] = {
        "schema": "gm1.tilestream-post-acoustic-mass-flux-token/v1",
        "status": "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX",
        "nrk": 1,
        "stage_identity_sha256": "a" * 64,
        "post_sound_velocity_token_sha256": velocity.receipt["operation_token_sha256"],
        "stream_pointer": 7123,
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
    }
    flux_receipt["operation_token_sha256"] = _operation_token(flux_receipt)
    flux = Gm1TiledSolve2PostAcousticFluxToken(MappingProxyType(flux_receipt))
    scalar_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS,
        "completed_nrk": 1,
        "stream_pointer": 7123,
        "post_acoustic_flux_token_sha256": flux_receipt["operation_token_sha256"],
        "output_shapes": {
            "th3d": [nk, ny, nx],
            "q3d": [CM1_NUMQ, nk, ny, nx],
        },
        "output_sha256": {
            "th3d": hashlib.sha256(physical["th3d"].tobytes()).hexdigest(),
            "q3d": hashlib.sha256(physical["q3d"].tobytes()).hexdigest(),
        },
    }
    scalar_receipt["operation_token_sha256"] = _operation_token(scalar_receipt)
    scalar = Gm1TiledScalarMoistRk1Token(MappingProxyType(scalar_receipt))
    momentum_receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_STATUS,
        "completed_nrk": 1,
        "stage_identity_sha256": "a" * 64,
        "post_sound_momentum_carrier_token_sha256": "9" * 64,
        "post_acoustic_flux_token_sha256": flux_receipt["operation_token_sha256"],
        "scalar_moist_rk1_token_sha256": scalar_receipt["operation_token_sha256"],
        "stream_pointer": 7123,
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
    }
    momentum_receipt["operation_token_sha256"] = _operation_token(momentum_receipt)
    momentum = Gm1Rk1PostScalarMomentumBoundaryToken(MappingProxyType(momentum_receipt))
    binding = Gm1TiledRk1ScalarPressureBoundaryBinding.create(
        pressure,
        scalar,
        flux,
        momentum,
        physical=physical,
        padded=padded,
        allow_compact_validation=True,
    )
    assert (
        binding.momentum_boundary_token_sha256
        == momentum.receipt["operation_token_sha256"]
    )
    wrong = dict(momentum_receipt)
    wrong["scalar_moist_rk1_token_sha256"] = "9" * 64
    wrong["operation_token_sha256"] = _operation_token(wrong)
    with pytest.raises(ValueError, match="momentum boundary identity or lineage"):
        Gm1TiledRk1ScalarPressureBoundaryBinding.create(
            pressure,
            scalar,
            flux,
            Gm1Rk1PostScalarMomentumBoundaryToken(MappingProxyType(wrong)),
            physical=physical,
            padded=padded,
            allow_compact_validation=True,
        )


def test_tiled_cpu_publication_is_raw_bit_exact_to_resident_source_order() -> None:
    nk, ny, nx = 4, 16, 17
    physical = _physical(nk=nk, ny=ny, nx=nx)
    initial = _padded(nk=nk, ny=ny, nx=nx)
    tiled = {name: value.copy() for name, value in initial.items()}
    resident = {name: value.copy() for name, value in initial.items()}

    resident_tiles = {
        "rho": resident["rho"][1:-1].copy(),
        "pp3d": resident["pp3d"][1:-1].copy(),
        "th3d": resident["th3d"][1:-1].copy(),
        "q3d": resident["q3d"][:, 1:-1].copy(),
    }
    resident_tiles["rho"][:, 1:-1, 1:-1] = physical["rho"]
    resident_tiles["pp3d"][:, 1:-1, 1:-1] = physical["pp3d"]
    ng = CM1_OPEN_NGXY
    resident_tiles["th3d"][:, ng:-ng, ng:-ng] = physical["th3d"]
    resident_tiles["q3d"][:, :, ng:-ng, ng:-ng] = physical["q3d"]
    cpu_gm1_tiled_rk1_scalar_pressure_boundaries_inplace(
        resident_tiles,
        active_nx=nx,
        active_ny=ny,
        west=True,
        east=True,
        south=True,
        north=True,
    )
    resident["rho"][1:-1] = resident_tiles["rho"]
    resident["pp3d"][1:-1] = resident_tiles["pp3d"]
    resident["th3d"][1:-1] = resident_tiles["th3d"]
    resident["q3d"][:, 1:-1] = resident_tiles["q3d"]

    binding = _direct_binding(physical, tiled)
    plan = build_gm1_open_tile_plan(nx=nx, ny=ny, tile_nx=8, tile_ny=8, halo=3)
    shapes = tiled_rk1_scalar_pressure_boundary_buffer_shapes(
        tile_nx=8, tile_ny=8, nk=nk
    )
    for spec in plan.specs:
        tile = {
            name: np.empty(shape, dtype=np.float32) for name, shape in shapes.items()
        }
        source = fill_gm1_tiled_rk1_scalar_pressure_boundary_source(binding, spec, tile)
        flags = source["open_edges"]
        cpu_gm1_tiled_rk1_scalar_pressure_boundaries_inplace(
            tile,
            active_nx=source["active_shape"][2],
            active_ny=source["active_shape"][1],
            west=flags["west"],
            east=flags["east"],
            south=flags["south"],
            north=flags["north"],
        )
        scatter_gm1_tiled_rk1_scalar_pressure_boundary_output(binding, spec, tile)

    for name in tiled:
        assert np.array_equal(
            tiled[name].view(np.uint32), resident[name].view(np.uint32)
        ), name
    _verify_bcp(np, tiled["rho"], physical["rho"], nk=nk, ny=ny, nx=nx)
    _verify_bcp(np, tiled["pp3d"], physical["pp3d"], nk=nk, ny=ny, nx=nx)
    _verify_bcs(
        np,
        tiled["th3d"],
        physical["th3d"],
        binding.scalar_corner_sources["th3d"],
        nk=nk,
        ny=ny,
        nx=nx,
    )
    _verify_bcs(
        np,
        tiled["q3d"],
        physical["q3d"],
        binding.scalar_corner_sources["q3d"],
        nk=nk,
        ny=ny,
        nx=nx,
    )


def test_memory_inventory_stays_bounded_and_refuses_global_pressure_claim() -> None:
    inventory = five_meter_rk1_scalar_pressure_boundary_inventory()
    expected_shapes = tiled_rk1_scalar_pressure_boundary_buffer_shapes(
        tile_nx=192, tile_ny=192, nk=75
    )
    expected = 4 * sum(
        np.prod(shape, dtype=np.int64) for shape in expected_shapes.values()
    )
    assert inventory["one_device_tile_bytes"] == expected
    assert inventory["boundary_operator_tile_streamable"] is True
    assert inventory["global_psolver_tile_streamable"] is False
    assert inventory["completed_broader_rk_stage"] is False
    assert inventory["five_meter_meteorological_product"] is False
    detached = dict(inventory)
    token = detached.pop("inventory_sha256")
    assert token == hashlib.sha256(_canonical(detached)).hexdigest()
