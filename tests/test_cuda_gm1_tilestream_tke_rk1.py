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
from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
    Gm1TiledSolve2PostAcousticFluxToken,
)
from cm1core.cuda_gm1_tilestream_tke_rk1 import (
    AUTHORITY_ADV_ROUTINES_SHA256,
    AUTHORITY_ADV_SHA256,
    AUTHORITY_ANCHORS,
    AUTHORITY_SOLVE2_SHA256,
    CUDA_GM1_TILESTREAM_TKE_RK1_SOURCE_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA,
    Gm1TiledTkeRk1Token,
    bind_gm1_tiled_tke_rk1_tile_source,
    five_meter_tiled_tke_rk1_inventory,
    require_gm1_tiled_tke_rk2,
)
from cm1core.cuda_tke_evolution import tke_evolution_shapes


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


def _tokens() -> tuple[Gm1TiledSolve1TkeRhsToken, Gm1TiledSolve2PostAcousticFluxToken]:
    solve1 = Gm1TiledSolve1TkeRhsToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS",
                "completed_nrk": 0,
                "stream_pointer": 17,
            }
        )
    )
    flux = Gm1TiledSolve2PostAcousticFluxToken(
        _tokenize(
            {
                "schema": CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX",
                "nrk": 1,
                "stream_pointer": 17,
                "stage_identity_sha256": "a" * 64,
                "solve1_tke_rhs_token_sha256": solve1.receipt[
                    "operation_token_sha256"
                ],
            }
        )
    )
    return solve1, flux


def test_rk1_source_token_binds_exact_host_and_h2d_bytes() -> None:
    solve1, flux = _tokens()
    spec = build_gm1_open_tile_plan(
        nx=12, ny=12, tile_nx=6, tile_ny=6
    ).specs[0]
    arrays = {
        name: np.zeros(shape, dtype="<f4")
        for name, shape in tke_evolution_shapes(spec.cnx, spec.cny, 75).items()
    }
    first = bind_gm1_tiled_tke_rk1_tile_source(
        solve1, flux, arrays, spec=spec
    )
    assert first.receipt["h2d_raw_bit_verified"] is False
    token = bind_gm1_tiled_tke_rk1_tile_source(
        solve1,
        flux,
        arrays,
        spec=spec,
        device_h2d_inventory_sha256=first.receipt["host_inventory_sha256"],
    )
    assert token.receipt["schema"] == (
        CUDA_GM1_TILESTREAM_TKE_RK1_SOURCE_TOKEN_SCHEMA
    )
    assert token.receipt["h2d_raw_bit_verified"] is True
    assert token.receipt["source_halo"] == 3
    assert token.receipt["post_acoustic_flux_token_sha256"] == (
        flux.receipt["operation_token_sha256"]
    )


def test_rk2_refuses_until_a_new_evolved_flux_token_exists() -> None:
    receipt = {
        "schema": CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILESTREAM_TKE_ADVW_RK1",
        "completed_nrk": 1,
        "stream_pointer": 17,
    }
    token = Gm1TiledTkeRk1Token(_tokenize(receipt))
    with pytest.raises(RuntimeError, match="nrk=2"):
        require_gm1_tiled_tke_rk2(token, object())


def test_authority_and_five_meter_refusal_are_explicit() -> None:
    assert AUTHORITY_SOLVE2_SHA256 == (
        "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
    )
    assert AUTHORITY_ADV_SHA256 == (
        "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f"
    )
    assert AUTHORITY_ADV_ROUTINES_SHA256 == (
        "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00"
    )
    assert "src/solve2.F:1708-1803" in AUTHORITY_ANCHORS
    inventory = five_meter_tiled_tke_rk1_inventory()
    assert inventory["tile_count"] == 256
    assert inventory["tke_advw_rk1_tile_streamable"] is True
    assert inventory["global_psolver_tile_streamable"] is False
    assert inventory["rk1_post_acoustic_flux_available_at_5m"] is False
    assert inventory["rk2_available"] is False
    assert inventory["five_meter_meteorological_product"] is False

