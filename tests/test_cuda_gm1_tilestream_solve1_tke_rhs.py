from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import pytest

from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    AUTHORITY_ANCHORS,
    AUTHORITY_SOLVE1_SHA256,
    AUTHORITY_TURBTEND_SHA256,
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE,
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_KERNEL_NAME,
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_SHA256,
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
    _vertical_disposition,
    five_meter_tiled_solve1_tke_rhs_inventory,
    require_gm1_tiled_solve2_tke_rk1,
    launch_cuda_gm1_tiled_solve1_tke_rhs,
)
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _solve1_token() -> Gm1TiledSolve1TkeRhsToken:
    receipt = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS",
        "stream_pointer": 91,
    }
    receipt["operation_token_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    return Gm1TiledSolve1TkeRhsToken(MappingProxyType(receipt))


def test_local_source_is_exact_selected_solve1_expression_and_pinned() -> None:
    assert hashlib.sha256(
        CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE.encode("ascii")
    ).hexdigest() == CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_SHA256
    assert CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_KERNEL_NAME in (
        CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE
    )
    for expression in (
        "float buoyancy=M(-khv[p],nm[p])",
        "float source=S(buoyancy,dissten[p])",
        "float deformation=fmaxf(0.0f,A(defv[p],defh[p]))",
        "tketen[p]=A(source,M(kmv[p],deformation))",
    ):
        assert expression in CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE
    for forbidden in ("rru", "rrv", "rrw", "divx", "psolver", "tke3d[p]="):
        assert forbidden not in CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE


def test_official_source_authority_and_cadence_are_explicit() -> None:
    assert AUTHORITY_SOLVE1_SHA256 == (
        "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
    )


def test_tke_vertical_disposition_matches_exact_closure_modes() -> None:
    assert _vertical_disposition(False) == "sealed-stretched-child"
    assert _vertical_disposition(True) == (
        "source-audited-flat-idealized-uniform-200m"
    )
    assert AUTHORITY_TURBTEND_SHA256 == (
        "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de"
    )
    assert AUTHORITY_ANCHORS == (
        "src/solve1.F:314-345",
        "src/solve1.F:362-369",
        "src/turbtend.F:700-1044",
    )


def test_solve2_rk1_refuses_missing_live_post_acoustic_flux_token() -> None:
    with pytest.raises(RuntimeError, match="post-acoustic rru/rrv/rrw/divx"):
        require_gm1_tiled_solve2_tke_rk1(_solve1_token(), object())


def test_five_meter_inventory_reuses_closure_without_product_claim() -> None:
    inventory = five_meter_tiled_solve1_tke_rhs_inventory()
    assert inventory["domain"] == [75, 3000, 3000]
    assert inventory["tile"] == [192, 192]
    assert inventory["tile_count"] == 256
    assert inventory["incremental_global_device_buffer_bytes"] == 0
    assert inventory["post_acoustic_flux_token_available"] is False
    assert inventory["completed_rk_stage_available"] is False
    assert inventory["five_meter_meteorological_product"] is False


def test_uniform_idealized_admission_is_explicit_and_exact_before_cuda() -> None:
    spec = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96).specs[0]
    with pytest.raises(ValueError, match="exact 300x300x75"):
        launch_cuda_gm1_tiled_solve1_tke_rhs(
            object(),
            {},
            spec=spec,
            sigma=object(),
            nk=75,
            dx=50.0,
            dy=50.0,
            dz=100.0,
            ztop=15000.0,
            dt=0.25,
            allow_uniform_idealized_validation=True,
        )
    with pytest.raises(ValueError, match="exact 300x300x75"):
        launch_cuda_gm1_tiled_solve1_tke_rhs(
            object(),
            {},
            spec=spec,
            sigma=object(),
            nk=75,
            dx=50.0,
            dy=50.0,
            dz=200.0,
            ztop=15000.0,
            dt=0.25,
            allow_compact_validation=True,
            allow_uniform_idealized_validation=True,
        )
