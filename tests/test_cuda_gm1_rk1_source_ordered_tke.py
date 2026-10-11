from __future__ import annotations

from types import MappingProxyType

import pytest

from cm1core.cuda_gm1_rk1_source_ordered_tke import (
    CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA,
    CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS,
    Gm1Rk1SourceOrderedTkeToken,
    _operation_token,
    bind_gm1_rk1_source_ordered_tke,
)
from cm1core.cuda_gm1_tilestream_rk1_scalar_pressure_boundaries import (
    CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS,
    Gm1TiledRk1ScalarPressureBoundaryToken,
)
from cm1core.cuda_gm1_tilestream_tke_rk1 import Gm1TiledTkeRk1Token
from cm1core.cuda_gm1_tilestream_tke_stage_state import (
    CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
    Gm1TiledTkeHeldStageToken,
)


def _tokens() -> tuple[
    Gm1TiledRk1ScalarPressureBoundaryToken,
    Gm1TiledTkeRk1Token,
    Gm1TiledTkeHeldStageToken,
]:
    flux = "1" * 64
    stage = "2" * 64
    stream = 9123
    boundary: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS,
        "completed_nrk": 1,
        "stage_identity_sha256": stage,
        "post_acoustic_flux_token_sha256": flux,
        "stream_pointer": stream,
    }
    boundary["operation_token_sha256"] = _operation_token(boundary)
    tke: dict[str, object] = {
        "schema": "gm1.tilestream-tke-rk1-token/v1",
        "status": "ADMITTED_LIVE_TILESTREAM_TKE_ADVW_RK1",
        "completed_nrk": 1,
        "stage_identity_sha256": stage,
        "post_acoustic_flux_token_sha256": flux,
        "stream_pointer": stream,
        "output_tke3d_sha256": "3" * 64,
    }
    tke["operation_token_sha256"] = _operation_token(tke)
    held: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TKE_HELD_STATE_AFTER_RK1",
        "completed_nrk": 1,
        "rk1_stage_identity_sha256": stage,
        "tke_rk1_token_sha256": tke["operation_token_sha256"],
        "current_tke3d_sha256": tke["output_tke3d_sha256"],
        "stream_pointer": stream,
    }
    held["operation_token_sha256"] = _operation_token(held)
    return (
        Gm1TiledRk1ScalarPressureBoundaryToken(MappingProxyType(boundary)),
        Gm1TiledTkeRk1Token(MappingProxyType(tke)),
        Gm1TiledTkeHeldStageToken(MappingProxyType(held)),
    )


def test_binds_exact_post_boundary_tke_component_without_broad_stage() -> None:
    token = bind_gm1_rk1_source_ordered_tke(*_tokens())
    assert isinstance(token, Gm1Rk1SourceOrderedTkeToken)
    assert token.receipt["schema"] == CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA
    assert token.receipt["status"] == CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS
    assert token.receipt["component_key"] == "tke_rk"
    assert token.receipt["source_order_component_complete"] is True
    assert token.receipt["completed_broader_rk_stage"] is False
    assert token.receipt["operation_token_sha256"] == _operation_token(token.receipt)


def test_refuses_tke_that_precedes_or_differs_from_boundary_lineage() -> None:
    boundary, tke, held = _tokens()
    wrong = dict(boundary.receipt)
    wrong["post_acoustic_flux_token_sha256"] = "9" * 64
    wrong["operation_token_sha256"] = _operation_token(wrong)
    with pytest.raises(ValueError, match="lineage"):
        bind_gm1_rk1_source_ordered_tke(
            Gm1TiledRk1ScalarPressureBoundaryToken(MappingProxyType(wrong)),
            tke,
            held,
        )


def test_refuses_duck_typed_tke_token() -> None:
    boundary, tke, held = _tokens()
    with pytest.raises(TypeError, match="exact"):
        bind_gm1_rk1_source_ordered_tke(boundary, tke.receipt, held)  # type: ignore[arg-type]
