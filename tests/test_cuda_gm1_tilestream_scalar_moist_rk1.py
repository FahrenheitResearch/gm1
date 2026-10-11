from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import numpy as np
import pytest

from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
)
from cm1core.cuda_gm1_tilestream_scalar_post_sound_getdiv import (
    CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA,
    Gm1TiledScalarThetaDivergenceToken,
)
from cm1core.cuda_gm1_tilestream_solve1_scalar_held import (
    CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS,
    Gm1TiledSolve1ScalarHeldToken,
)
from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from cm1core import cuda_gm1_tilestream_scalar_moist_rk1 as rk1


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _tokenize(receipt: dict[str, object]) -> MappingProxyType:
    receipt["operation_token_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    return MappingProxyType(receipt)


def _tokens(*, plan_sha: str = "d" * 64) -> tuple[
    Gm1TiledSolve1ScalarHeldToken,
    Gm1TiledSolve2PostAcousticFluxToken,
    Gm1TiledScalarThetaDivergenceToken,
]:
    held = Gm1TiledSolve1ScalarHeldToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_SCHEMA,
                "status": CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS,
                "completed_nrk": 0,
                "stream_pointer": 19,
                "tile_plan_sha256": plan_sha,
            }
        )
    )
    flux = Gm1TiledSolve2PostAcousticFluxToken(
        _tokenize(
            {
                "schema": CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX",
                "nrk": 1,
                "stream_pointer": 19,
                "tile_plan_sha256": plan_sha,
                "stage_identity_sha256": "a" * 64,
                "post_sound_velocity_token_sha256": "b" * 64,
            }
        )
    )
    theta = Gm1TiledScalarThetaDivergenceToken(
        _tokenize(
            {
                "schema": (
                    CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA
                ),
                "status": "ADMITTED_LIVE_RK1_SCALAR_THETA_DIVERGENCE",
                "nrk": 1,
                "stream_pointer": 19,
                "tile_plan_sha256": plan_sha,
                "stage_identity_sha256": "a" * 64,
                "post_sound_velocity_token_sha256": "b" * 64,
            }
        )
    )
    return held, flux, theta


def test_exact_downstream_token_contract_is_frozen() -> None:
    assert rk1.CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA == (
        "gm1.tilestream-scalar-moist-rk1-token/v1"
    )
    assert rk1.CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS == (
        "ADMITTED_LIVE_TILESTREAM_SCALAR_MOIST_RK1"
    )
    assert rk1.SCALAR_ORDER == ("theta", "qv", "qc", "qr", "qi", "qs", "qg")
    token = rk1.Gm1TiledScalarMoistRk1Token(
        MappingProxyType(
            {
                "schema": rk1.CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA,
                "status": rk1.CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS,
                "completed_nrk": 1,
                "stream_pointer": 19,
                "output_sha256": {"th3d": "a" * 64, "q3d": "b" * 64},
                "output_shapes": {
                    "th3d": [75, 300, 300],
                    "q3d": [6, 75, 300, 300],
                },
            }
        )
    )
    assert tuple(token.receipt["output_sha256"]) == ("th3d", "q3d")
    assert tuple(token.receipt["output_shapes"]) == ("th3d", "q3d")


def test_prerequisites_require_one_sound_stage_and_stream() -> None:
    held, flux, theta = _tokens()
    got = rk1._require_tokens(held, flux, theta)
    assert got[0]["stream_pointer"] == 19
    changed = dict(theta.receipt)
    changed["post_sound_velocity_token_sha256"] = "c" * 64
    changed.pop("operation_token_sha256")
    foreign = Gm1TiledScalarThetaDivergenceToken(_tokenize(changed))
    with pytest.raises(ValueError, match="different sound stages"):
        rk1._require_tokens(held, flux, foreign)


def test_tile_source_binds_ring_patch_and_raw_h2d_bytes() -> None:
    held, flux, theta = _tokens()
    plan = build_gm1_open_tile_plan(nx=12, ny=12, tile_nx=6, tile_ny=6)
    spec = plan.specs[0]
    shapes = rk1.scalar_moist_rk1_device_tile_shapes(
        tile_nx=6, tile_ny=6, nk=75
    )
    arrays = {
        name: np.zeros(shapes[name], dtype="<f4")
        for name in rk1._DEVICE_INPUT_NAMES
    }
    fill = {
        "schema": "gm1.tilestream-scalar-moist-rk1-host-fill/v1",
        "scalar": "theta",
        "tile": spec.record(),
    }
    fill["operation_token_sha256"] = hashlib.sha256(_canonical(fill)).hexdigest()
    save = {
        "schema": "gm1.hybrid-tilestream-scalar-ring-save/v1",
        "tile": spec.index,
        "ring_schedule_sha256": "c" * 64,
    }
    patch = {
        "schema": "gm1.hybrid-tilestream-scalar-ring-patch/v1",
        "tile": spec.index,
        "source_stage_values_restored": True,
        "canonical_state_mutated_by_patch": False,
    }
    first = rk1.bind_gm1_tiled_scalar_moist_rk1_tile_source(
        held,
        flux,
        theta,
        arrays,
        spec=spec,
        scalar="theta",
        fill_receipt=fill,
        ring_save_receipt=save,
        ring_patch_receipt=patch,
    )
    assert first.receipt["h2d_raw_bit_verified"] is False
    admitted = rk1.bind_gm1_tiled_scalar_moist_rk1_tile_source(
        held,
        flux,
        theta,
        arrays,
        spec=spec,
        scalar="theta",
        fill_receipt=fill,
        ring_save_receipt=save,
        ring_patch_receipt=patch,
        device_h2d_inventory_sha256=first.receipt["host_inventory_sha256"],
    )
    assert admitted.receipt["h2d_raw_bit_verified"] is True
    assert admitted.receipt["source_stage_values_restored"] is True
    with pytest.raises(ValueError, match="H2D bytes differ"):
        rk1.bind_gm1_tiled_scalar_moist_rk1_tile_source(
            held,
            flux,
            theta,
            arrays,
            spec=spec,
            scalar="theta",
            fill_receipt=fill,
            ring_save_receipt=save,
            ring_patch_receipt=patch,
            device_h2d_inventory_sha256="e" * 64,
        )


def test_finish_refuses_non_oracle_domain_before_output_access() -> None:
    plan = build_gm1_open_tile_plan(nx=12, ny=12, tile_nx=6, tile_ny=6)
    held, flux, theta = _tokens(plan_sha=plan.ownership_sha256)
    with pytest.raises(ValueError, match="admitted 300 grid"):
        rk1.finish_cuda_gm1_tiled_scalar_moist_rk1(
            held,
            flux,
            theta,
            {},
            {},
            plan=plan,
            outputs={},
        )


def test_five_meter_memory_and_no_product_boundary_are_exact() -> None:
    inventory = rk1.five_meter_scalar_moist_rk1_inventory()
    assert inventory["compute_tile"] == [198, 198]
    assert inventory["tile_count"] == 256
    assert inventory["host_buffer_bytes_each"] == 183_248_880
    assert inventory["double_pinned_host_buffer_bytes"] == 366_497_760
    assert inventory["device_buffer_bytes_each"] == 196_629_600
    assert inventory["double_device_buffer_bytes"] == 393_259_200
    assert inventory["inventory_sha256"] == (
        "64f7eefc773465b5b89e8562cb8616a5db54b8fcd6203db0903341f438cb7781"
    )
    assert inventory["five_meter_solve1_scalar_held_token_available"] is False
    assert inventory["five_meter_post_acoustic_flux_token_available"] is False
    assert inventory["five_meter_theta_divergence_token_available"] is False
    assert inventory["global_psolver_tile_streamable"] is False
    assert inventory["five_meter_meteorological_product"] is False


def test_source_order_and_authority_are_literal() -> None:
    assert rk1.AUTHORITY_SOLVE2_SHA256 == (
        "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
    )
    assert "src/solve2.F:1430-1533" in rk1.AUTHORITY_ANCHORS
    assert rk1.SOURCE_ORDER[-1].startswith("publish every physical cell")
    assert "before the separate bcp/bcs boundary phase" in rk1.SOURCE_ORDER[-1]
    assert "five-metre meteorological product remain refused" in (
        rk1.GM1_TILESTREAM_SCALAR_MOIST_RK1_CLAIM_LIMIT
    )
