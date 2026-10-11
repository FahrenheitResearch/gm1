from __future__ import annotations

import hashlib
from types import MappingProxyType

import numpy as np
import pytest

from cm1core.cuda_gm1_rk1_post_scalar_momentum_boundaries import (
    AUTHORITY_BC_SHA256,
    AUTHORITY_INIT_TERRAIN_SHA256,
    AUTHORITY_SOLVE2_SHA256,
    CM1_OPEN_NGXY,
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SCHEMA,
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE,
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE_SHA256,
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_SCHEMA,
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_STATUS,
    CudaGm1Rk1PostScalarMomentumBoundaryLaunch,
    Gm1Rk1PostScalarMomentumBoundaryBinding,
    Gm1Rk1PostSoundMomentumCarrierToken,
    _canonical,
    _operation_token,
    checkpoint_gm1_rk1_post_sound_momentum_carriers,
    cpu_gm1_rk1_post_scalar_momentum_boundaries_inplace,
    finish_cuda_gm1_rk1_post_scalar_momentum_boundaries,
    five_meter_rk1_post_scalar_momentum_boundary_inventory,
    rk1_post_scalar_momentum_boundary_shapes,
)
from cm1core.cuda_gm1_tilestream_post_acoustic_getdivx import (
    Gm1PostSoundVelocityToken,
)
from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from cm1core.cuda_gm1_tilestream_scalar_moist_rk1 import (
    Gm1TiledScalarMoistRk1Token,
)


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _arrays(*, nx: int, ny: int, nk: int) -> dict[str, np.ndarray]:
    shapes = rk1_post_scalar_momentum_boundary_shapes(nx=nx, ny=ny, nk=nk)
    result: dict[str, np.ndarray] = {}
    for index, (name, shape) in enumerate(shapes.items(), start=1):
        value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        result[name] = np.ascontiguousarray(
            value * np.float32(index * 1.0e-7) + np.float32(index * 0.25)
        )
    return result


def _tokens(
    arrays: dict[str, np.ndarray], *, nx: int, ny: int, nk: int, stream: int = 9123
) -> tuple[
    Gm1PostSoundVelocityToken,
    Gm1Rk1PostSoundMomentumCarrierToken,
    Gm1TiledSolve2PostAcousticFluxToken,
    Gm1TiledScalarMoistRk1Token,
]:
    ng = CM1_OPEN_NGXY
    physical = {
        "u3d": np.ascontiguousarray(
            arrays["u3d"][1 : nk + 1, ng : ng + ny, ng : ng + nx + 1]
        ),
        "v3d": np.ascontiguousarray(
            arrays["v3d"][1 : nk + 1, ng : ng + ny + 1, ng : ng + nx]
        ),
        "w3d": np.ascontiguousarray(
            arrays["w3d"][1 : nk + 2, ng : ng + ny, ng : ng + nx]
        ),
    }
    velocity: dict[str, object] = {
        "schema": "gm1.post-sound-averaged-velocity-token/v1",
        "status": "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT",
        "nrk": 1,
        "stage_identity_sha256": "a" * 64,
        "solve1_tke_rhs_token_sha256": "b" * 64,
        "psolver_receipt_sha256": "c" * 64,
        "stream_pointer": stream,
        "host_checkpoint_transport": True,
        "velocity_shapes": {
            "u3d": [nk, ny, nx + 1],
            "v3d": [nk, ny + 1, nx],
            "w3d": [nk + 1, ny, nx],
        },
        "velocity_sha256": {name: _sha(value) for name, value in physical.items()},
        "sound_output_semantics": "small-step-averaged velocities before getdivx",
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "authority": {},
        "claim_limit": "exact compact test token",
    }
    velocity["operation_token_sha256"] = _operation_token(velocity)
    velocity_token = Gm1PostSoundVelocityToken(MappingProxyType(velocity))
    carrier_token = checkpoint_gm1_rk1_post_sound_momentum_carriers(
        velocity_token,
        physical,
        allow_compact_validation=True,
    )
    flux: dict[str, object] = {
        "schema": "gm1.tilestream-post-acoustic-mass-flux-token/v1",
        "status": "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX",
        "nrk": 1,
        "stage_identity_sha256": "a" * 64,
        "solve1_tke_rhs_token_sha256": "b" * 64,
        "post_sound_velocity_token_sha256": velocity["operation_token_sha256"],
        "tile_plan_sha256": "d" * 64,
        "tile_count": 1,
        "tile_operation_tokens_sha256": "e" * 64,
        "stream_pointer": stream,
        "host_scatter_exact_once": True,
        "output_shapes": {},
        "output_sha256": {},
        "output_words": 0,
        "source_order": [],
        "ready_for": "typed TKE advw/RK1 operator only",
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "structural_refusals": [],
        "authority": {},
        "claim_limit": "exact compact test token",
    }
    flux["operation_token_sha256"] = _operation_token(flux)
    flux_token = Gm1TiledSolve2PostAcousticFluxToken(MappingProxyType(flux))
    scalar: dict[str, object] = {
        "schema": "gm1.tilestream-scalar-moist-rk1-token/v1",
        "status": "ADMITTED_LIVE_TILESTREAM_SCALAR_MOIST_RK1",
        "completed_nrk": 1,
        "stream_pointer": stream,
        "solve1_scalar_held_token_sha256": "f" * 64,
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "theta_divergence_token_sha256": "1" * 64,
        "output_shapes": {
            "th3d": [nk, ny, nx],
            "q3d": [6, nk, ny, nx],
        },
        "output_sha256": {"th3d": "2" * 64, "q3d": "3" * 64},
    }
    scalar["operation_token_sha256"] = _operation_token(scalar)
    return (
        velocity_token,
        carrier_token,
        flux_token,
        Gm1TiledScalarMoistRk1Token(MappingProxyType(scalar)),
    )


def test_authority_and_cuda_source_are_pinned() -> None:
    assert AUTHORITY_SOLVE2_SHA256 == (
        "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
    )
    assert AUTHORITY_BC_SHA256 == (
        "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
    )
    assert AUTHORITY_INIT_TERRAIN_SHA256 == (
        "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c"
    )
    assert (
        hashlib.sha256(
            CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE.encode("ascii")
        ).hexdigest()
        == CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE_SHA256
    )
    assert "__fadd_rn" in CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE
    assert "__fmul_rn" in CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE


def test_cpu_replay_preserves_physical_uv_and_rebuilds_w_faces() -> None:
    nx, ny, nk = 8, 7, 4
    arrays = _arrays(nx=nx, ny=ny, nk=nk)
    ng = CM1_OPEN_NGXY
    u_before = arrays["u3d"][1 : nk + 1, ng : ng + ny, ng : ng + nx + 1].copy()
    v_before = arrays["v3d"][1 : nk + 1, ng : ng + ny + 1, ng : ng + nx].copy()
    cpu_gm1_rk1_post_scalar_momentum_boundaries_inplace(arrays, nx=nx, ny=ny, nk=nk)
    assert np.array_equal(
        arrays["u3d"][1 : nk + 1, ng : ng + ny, ng : ng + nx + 1], u_before
    )
    assert np.array_equal(
        arrays["v3d"][1 : nk + 1, ng : ng + ny + 1, ng : ng + nx], v_before
    )
    assert (
        np.count_nonzero(
            arrays["w3d"][nk + 1, ng - 1 : ng + ny + 1, ng - 1 : ng + nx + 1].view(
                np.uint32
            )
        )
        == 0
    )
    i, j = 3, 2
    x, y = i + ng - 1, j + ng - 1
    mi, mj = i - 1, j - 1
    expected = np.float32(
        np.float32(0.5)
        * np.float32(
            np.float32(
                np.float32(arrays["u3d"][1, y, x] + arrays["u3d"][1, y, x + 1])
                * arrays["dzdx"][mj, mi]
            )
            + np.float32(
                np.float32(arrays["v3d"][1, y, x] + arrays["v3d"][1, y + 1, x])
                * arrays["dzdy"][mj, mi]
            )
        )
        * arrays["gz"][mj, mi]
    )
    assert arrays["w3d"][1, y, x].view(np.uint32) == expected.view(np.uint32)


def test_binding_and_finish_require_post_scalar_lineage() -> None:
    nx, ny, nk = 8, 7, 4
    arrays = _arrays(nx=nx, ny=ny, nk=nk)
    velocity, carrier, flux, scalar = _tokens(arrays, nx=nx, ny=ny, nk=nk)
    binding = Gm1Rk1PostScalarMomentumBoundaryBinding.create(
        velocity,
        carrier,
        flux,
        scalar,
        arrays,
        allow_compact_validation=True,
    )
    cpu_gm1_rk1_post_scalar_momentum_boundaries_inplace(arrays, nx=nx, ny=ny, nk=nk)
    launch_receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SCHEMA,
        "abi": "gm1.cuda-rk1-post-scalar-momentum-boundary-f32/v1",
        "status": "ENQUEUED_LIVE_RK1_POST_SCALAR_MOMENTUM_BOUNDARIES",
        "completed_nrk": 1,
        "stage_identity_sha256": binding.stage_identity_sha256,
        "post_sound_velocity_token_sha256": binding.velocity_token_sha256,
        "post_sound_momentum_carrier_token_sha256": (
            binding.momentum_carrier_token_sha256
        ),
        "post_acoustic_flux_token_sha256": binding.flux_token_sha256,
        "scalar_moist_rk1_token_sha256": binding.scalar_token_sha256,
        "stream_pointer": binding.stream_pointer,
        "same_stream_after_scalar_rk": True,
        "host_sync_before_launch": True,
        "source_order": [],
        "domain": {},
        "metric_halo_policy": "test",
        "kernel": {},
        "authority": {},
        "claim_limit": "test",
    }
    launch_receipt["operation_token_sha256"] = _operation_token(launch_receipt)
    launch = CudaGm1Rk1PostScalarMomentumBoundaryLaunch(
        object(), object(), MappingProxyType(launch_receipt)
    )
    token = finish_cuda_gm1_rk1_post_scalar_momentum_boundaries(binding, launch)
    assert token.receipt["schema"] == (
        CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_SCHEMA
    )
    assert token.receipt["status"] == (
        CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_STATUS
    )
    assert token.receipt["component_key"] == "momentum_boundary"
    assert token.receipt["comparison"]["mismatch_words"] == 0
    assert token.receipt["completed_broader_rk_stage"] is False
    wrong = dict(scalar.receipt)
    wrong["post_acoustic_flux_token_sha256"] = "9" * 64
    wrong["operation_token_sha256"] = _operation_token(wrong)
    with pytest.raises(ValueError, match="lineage"):
        Gm1Rk1PostScalarMomentumBoundaryBinding.create(
            velocity,
            carrier,
            flux,
            Gm1TiledScalarMoistRk1Token(MappingProxyType(wrong)),
            _arrays(nx=nx, ny=ny, nk=nk),
            allow_compact_validation=True,
        )


def test_memory_inventory_refuses_missing_five_meter_sound_token() -> None:
    inventory = five_meter_rk1_post_scalar_momentum_boundary_inventory()
    assert inventory["incremental_device_workspace_bytes"] == 0
    assert inventory["metric_halo_allocation_bytes"] == 0
    assert inventory["global_psolver_tile_streamable"] is False
    assert inventory["live_post_sound_token_available_at_five_meters"] is False
    assert inventory["completed_broader_rk_stage"] is False
    detached = dict(inventory)
    token = detached.pop("inventory_sha256")
    assert token == hashlib.sha256(_canonical(detached)).hexdigest()
