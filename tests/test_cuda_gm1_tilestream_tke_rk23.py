from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import pytest

from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.cuda_gm1_tilestream_completed_rk_stage import (
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
    GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
    Gm1TiledCompletedRkStageToken,
)
from cm1core.cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
)
from cm1core.cuda_gm1_rk1_source_ordered_tke import (
    CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA,
    CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS,
    GM1_RK1_SOURCE_ORDERED_TKE_CLAIM_LIMIT,
    Gm1Rk1SourceOrderedTkeToken,
)
from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from cm1core.cuda_gm1_tilestream_tke_rk23 import (
    COMPLETED_RK_STAGE_STATUS,
    CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA,
    GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT,
    Gm1TiledTkeRk23Token,
    bind_gm1_tiled_tke_rk23_tile_source,
    require_gm1_tiled_tke_rk23_token,
)
from cm1core.cuda_gm1_tilestream_tke_stage_state import (
    CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
    Gm1TiledTkeHeldStageToken,
)


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


def _held() -> Gm1TiledTkeHeldStageToken:
    return Gm1TiledTkeHeldStageToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TKE_HELD_STATE_AFTER_RK1",
                "completed_nrk": 1,
                "next_nrk": 2,
                "stream_pointer": 17,
                "closure_tkea_baseline_token_sha256": "e" * 64,
                "fixed_tkea_sha256": "a" * 64,
                "held_tketen_sha256": "b" * 64,
                "current_tke3d_sha256": "c" * 64,
                "tke_rk1_token_sha256": "d" * 64,
            }
        )
    )


def _completed(
    *, completed_nrk: int, tke_sha: str
) -> Gm1TiledCompletedRkStageToken:
    return Gm1TiledCompletedRkStageToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
                "status": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
                "completed_nrk": completed_nrk,
                "next_nrk": completed_nrk + 1 if completed_nrk < 3 else None,
                "stream_pointer": 17,
                "stage_identity_sha256": str(5 + completed_nrk) * 64,
                "component_operation_token_sha256": {
                    "tke_rk": tke_sha,
                    "all_non_tke_components": "f" * 64,
                },
                "source_order_complete": True,
                "global_psolver_tile_streamable": False,
                "five_meter_meteorological_product": False,
                "full_forecast": False,
                "claim_limit": GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
            }
        )
    )


def _source_ordered(held: Gm1TiledTkeHeldStageToken) -> Gm1Rk1SourceOrderedTkeToken:
    return Gm1Rk1SourceOrderedTkeToken(
        _tokenize(
            {
                "schema": CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA,
                "status": CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS,
                "completed_nrk": 1,
                "next_nrk": 2,
                "stage_identity_sha256": "6" * 64,
                "stream_pointer": 17,
                "scalar_pressure_boundary_token_sha256": "0" * 64,
                "tke_rk1_token_sha256": held.receipt["tke_rk1_token_sha256"],
                "tke_held_state_token_sha256": held.receipt[
                    "operation_token_sha256"
                ],
                "output_tke3d_sha256": held.receipt["current_tke3d_sha256"],
                "component_key": "tke_rk",
                "source_order": [],
                "source_order_component_complete": True,
                "completed_broader_rk_stage": False,
                "missing_components": [],
                "global_psolver_tile_streamable": False,
                "mpi_performed": False,
                "open_history_mutated": False,
                "full_forecast": False,
                "five_meter_meteorological_product": False,
                "authority": {},
                "claim_limit": GM1_RK1_SOURCE_ORDERED_TKE_CLAIM_LIMIT,
            }
        )
    )


def _flux(
    *, nrk: int, completed: Gm1TiledCompletedRkStageToken
) -> Gm1TiledSolve2PostAcousticFluxToken:
    return Gm1TiledSolve2PostAcousticFluxToken(
        _tokenize(
            {
                "schema": CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX",
                "nrk": nrk,
                "stream_pointer": 17,
                "stage_identity_sha256": str(8 - nrk) * 64,
                "completed_rk_stage_token_sha256": completed.receipt[
                    "operation_token_sha256"
                ],
                "output_sha256": {
                    "rru": "1" * 64,
                    "rrv": "2" * 64,
                    "rrw": "3" * 64,
                    "divx": "4" * 64,
                },
            }
        )
    )


def _spec():
    return build_gm1_open_tile_plan(
        nx=12, ny=12, tile_nx=6, tile_ny=6
    ).specs[0]


def test_rk2_accepts_only_exact_broad_tke_lineage_before_payload() -> None:
    held = _held()
    source_ordered = _source_ordered(held)
    completed = _completed(
        completed_nrk=1,
        tke_sha=str(source_ordered.receipt["operation_token_sha256"]),
    )
    flux = _flux(nrk=2, completed=completed)
    with pytest.raises(ValueError, match="host tile inventory"):
        bind_gm1_tiled_tke_rk23_tile_source(
            held,
            completed,
            flux,
            source_ordered,
            {},
            {},
            nrk=2,
            spec=_spec(),
        )


def test_rk2_refuses_foreign_or_wrong_tke_component_token() -> None:
    held = _held()
    source_ordered = _source_ordered(held)
    completed = _completed(completed_nrk=1, tke_sha="e" * 64)
    flux = _flux(nrk=2, completed=completed)
    with pytest.raises(ValueError, match="TKE lineage"):
        bind_gm1_tiled_tke_rk23_tile_source(
            held, completed, flux, source_ordered, {}, {}, nrk=2, spec=_spec()
        )
    with pytest.raises(TypeError, match="exact source-ordered"):
        bind_gm1_tiled_tke_rk23_tile_source(
            held, completed, flux, object(), {}, {}, nrk=2, spec=_spec()
        )


def test_rk2_refuses_flux_not_linked_to_completed_stage() -> None:
    held = _held()
    source_ordered = _source_ordered(held)
    completed = _completed(
        completed_nrk=1,
        tke_sha=str(source_ordered.receipt["operation_token_sha256"]),
    )
    receipt = dict(_flux(nrk=2, completed=completed).receipt)
    receipt["completed_rk_stage_token_sha256"] = "0" * 64
    receipt.pop("operation_token_sha256")
    flux = Gm1TiledSolve2PostAcousticFluxToken(_tokenize(receipt))
    with pytest.raises(ValueError, match="stage lineage"):
        bind_gm1_tiled_tke_rk23_tile_source(
            held, completed, flux, source_ordered, {}, {}, nrk=2, spec=_spec()
        )


def test_rk3_requires_exact_completed_tke_rk2_lineage() -> None:
    held = _held()
    prior = Gm1TiledTkeRk23Token(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TILESTREAM_TKE_RK_STAGE",
                "completed_nrk": 2,
                "held_tke_state_token_sha256": held.receipt[
                    "operation_token_sha256"
                ],
                "stream_pointer": 17,
                "output_tke3d_sha256": "9" * 64,
            }
        )
    )
    completed = _completed(
        completed_nrk=2,
        tke_sha=str(prior.receipt["operation_token_sha256"]),
    )
    flux = _flux(nrk=3, completed=completed)
    with pytest.raises(ValueError, match="host tile inventory"):
        bind_gm1_tiled_tke_rk23_tile_source(
            held, completed, flux, prior, {}, {}, nrk=3, spec=_spec()
        )
    with pytest.raises(TypeError, match="exact completed tiled TKE RK2"):
        bind_gm1_tiled_tke_rk23_tile_source(
            held, completed, flux, object(), {}, {}, nrk=3, spec=_spec()
        )


def test_shared_completed_stage_status_is_not_redeclared_differently() -> None:
    assert COMPLETED_RK_STAGE_STATUS == (
        CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS
    )


def _finished_token(*, nrk: int = 3, stream: int = 17) -> Gm1TiledTkeRk23Token:
    return Gm1TiledTkeRk23Token(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TILESTREAM_TKE_RK_STAGE",
                "completed_nrk": nrk,
                "held_tke_state_token_sha256": "1" * 64,
                "prior_tke_stage_token_sha256": "2" * 64,
                "completed_prior_rk_stage_token_sha256": "3" * 64,
                "post_acoustic_flux_token_sha256": "4" * 64,
                "stage_identity_sha256": "5" * 64,
                "tile_plan_sha256": "6" * 64,
                "tile_count": 16,
                "tile_source_tokens_sha256": "7" * 64,
                "tile_operation_tokens_sha256": "8" * 64,
                "stream_pointer": stream,
                "host_scatter_exact_once": True,
                "output_tke3d_shape": [76, 300, 300],
                "output_tke3d_words": 76 * 300 * 300,
                "output_tke3d_sha256": "9" * 64,
                "final_bcw_applied": True,
                "ready_for": (
                    "broader RK2 completion, then fresh nrk=3 sound/getdivx"
                    if nrk == 2
                    else "solve3 TKE commit only after exact completed broader RK3"
                ),
                "structural_refusals": [
                    "TKE-only assertion of broader RK completion",
                    "reuse of an earlier-stage post-acoustic flux token",
                    "solve3 tkea commit before exact completed broader RK3",
                    "global psolver tilestream claim",
                    "MPI exchange or open-history production",
                    "full forecast or five-metre meteorological product",
                ],
                "claim_limit": GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT,
            }
        )
    )


@pytest.mark.parametrize("nrk", (2, 3))
def test_finished_tke_token_validator_is_stage_and_stream_exact(nrk: int) -> None:
    token = _finished_token(nrk=nrk)
    receipt = require_gm1_tiled_tke_rk23_token(
        token,
        completed_nrk=nrk,
        stream_pointer=17,
    )
    assert receipt["completed_nrk"] == nrk
    assert receipt["final_bcw_applied"] is True
    with pytest.raises(ValueError, match="another stream"):
        require_gm1_tiled_tke_rk23_token(token, stream_pointer=18)
    with pytest.raises(ValueError, match="another RK stage"):
        require_gm1_tiled_tke_rk23_token(token, completed_nrk=3 if nrk == 2 else 2)


def test_finished_tke_token_validator_refuses_duck_and_relabeled_output() -> None:
    token = _finished_token()
    with pytest.raises(TypeError, match="exact Gm1TiledTkeRk23Token"):
        require_gm1_tiled_tke_rk23_token({"receipt": token.receipt})
    changed = dict(token.receipt)
    changed["final_bcw_applied"] = False
    changed.pop("operation_token_sha256")
    with pytest.raises(ValueError, match="identity, output, or claim"):
        require_gm1_tiled_tke_rk23_token(
            Gm1TiledTkeRk23Token(_tokenize(changed))
        )
