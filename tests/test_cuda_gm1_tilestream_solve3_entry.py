from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import pytest

from cm1core.cuda_gm1_tilestream_completed_rk_stage import (
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
    GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
    Gm1TiledCompletedRkStageToken,
)
from cm1core.cuda_gm1_tilestream_solve3_boundaries import (
    Gm1TiledSolve3PreHeldBoundaryToken,
)
from cm1core.cuda_gm1_tilestream_solve3_entry import (
    AUTHORITY_ANCHORS,
    CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_STATUS,
    GM1_TILESTREAM_SOLVE3_ENTRY_CLAIM_LIMIT,
    Gm1TiledSolve3EntryToken,
    require_gm1_tiled_solve3_entry_token,
    solve3_entry_contract,
)
from cm1core.cuda_gm1_tilestream_solve3_held import Gm1TiledSolve3HeldToken
from cm1core.cuda_gm1_tilestream_solve3_momentum_a import (
    Gm1TiledSolve3PhysicalMomentumAToken,
)
from cm1core.cuda_gm1_tilestream_solve3_padded_momentum_a import (
    Gm1TiledSolve3PaddedMomentumAToken,
)
from cm1core.cuda_gm1_tilestream_tke_rk23 import (
    CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA,
    Gm1TiledTkeRk23Token,
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


def _fixtures() -> dict[str, object]:
    stream = 17
    stage = "8" * 64
    source = "a" * 64
    tke_held = Gm1TiledTkeHeldStageToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TKE_HELD_STATE_AFTER_RK1",
                "completed_nrk": 1,
                "stream_pointer": stream,
                "closure_tkea_baseline_token_sha256": "7" * 64,
            }
        )
    )
    tke_rk3 = Gm1TiledTkeRk23Token(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TILESTREAM_TKE_RK_STAGE",
                "completed_nrk": 3,
                "held_tke_state_token_sha256": tke_held.receipt[
                    "operation_token_sha256"
                ],
                "stream_pointer": stream,
                "stage_identity_sha256": stage,
                "output_tke3d_sha256": "9" * 64,
                "final_bcw_applied": True,
            }
        )
    )
    completed = Gm1TiledCompletedRkStageToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
                "status": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
                "completed_nrk": 3,
                "next_nrk": None,
                "stream_pointer": stream,
                "stage_identity_sha256": stage,
                "component_operation_token_sha256": {
                    "tke_rk": tke_rk3.receipt["operation_token_sha256"],
                    "all_non_tke_components": "b" * 64,
                },
                "source_order_complete": True,
                "global_psolver_tile_streamable": False,
                "five_meter_meteorological_product": False,
                "full_forecast": False,
                "claim_limit": GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
            }
        )
    )
    preheld = Gm1TiledSolve3PreHeldBoundaryToken(
        _tokenize(
            {
                "schema": "gm1.tilestream-solve3-preheld-boundary-token/v1",
                "status": (
                    "ADMITTED_PPX_RHO_BOUNDARIES_READY_FOR_HELD_RECAPTURE_ONLY"
                ),
                "source_identity_sha256": source,
                "stream_pointer": stream,
                "ready_for": "held-rate recapture admission only",
                "operation_b_current_to_a_available": False,
                "mpi_performed": False,
            }
        )
    )
    held = Gm1TiledSolve3HeldToken(
        _tokenize(
            {
                "schema": "gm1.tilestream-solve3-held-token/v1",
                "status": (
                    "ADMITTED_HELD_RATES_READY_FOR_MOMENTUM_CURRENT_TO_A_ONLY"
                ),
                "preheld_operation_token_sha256": preheld.receipt[
                    "operation_token_sha256"
                ],
                "source_identity_sha256": source,
                "stream_pointer": stream,
                "ready_for": "momentum current-to-A admission only",
                "tke_commit_performed": False,
            }
        )
    )
    physical = Gm1TiledSolve3PhysicalMomentumAToken(
        _tokenize(
            {
                "schema": "gm1.tilestream-solve3-physical-momentum-a-token/v1",
                "status": "ADMITTED_PHYSICAL_MOMENTUM_A_ONLY",
                "held_operation_token_sha256": held.receipt[
                    "operation_token_sha256"
                ],
                "source_identity_sha256": source,
                "stream_pointer": stream,
                "ready_for": (
                    "horizontal bc2d and padded momentum-A completion only"
                ),
                "tke_commit_available": False,
            }
        )
    )
    padded = Gm1TiledSolve3PaddedMomentumAToken(
        _tokenize(
            {
                "schema": "gm1.tilestream-solve3-padded-momentum-a-token/v1",
                "status": "ADMITTED_PADDED_MOMENTUM_A_ONLY",
                "physical_operation_token_sha256": physical.receipt[
                    "operation_token_sha256"
                ],
                "source_identity_sha256": source,
                "stream_pointer": stream,
                "ready_for": "independent TKE-commit admission audit only",
                "tke_commit_available": False,
            }
        )
    )
    lineage = {
        "completed_rk3": completed.receipt["operation_token_sha256"],
        "tke_held_state": tke_held.receipt["operation_token_sha256"],
        "tke_rk3": tke_rk3.receipt["operation_token_sha256"],
        "preheld_boundary": preheld.receipt["operation_token_sha256"],
        "held_recapture": held.receipt["operation_token_sha256"],
        "physical_momentum_a": physical.receipt["operation_token_sha256"],
        "padded_momentum_a": padded.receipt["operation_token_sha256"],
    }
    entry = Gm1TiledSolve3EntryToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_SCHEMA,
                "status": CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_STATUS,
                "completed_nrk": 3,
                "stream_pointer": stream,
                "stage_identity_sha256": stage,
                "source_identity_sha256": source,
                "lineage_operation_token_sha256": lineage,
                "source_order": [
                    "completed broader solve2 RK3",
                    "completed TKE advw/RK3/bcw",
                    "solve3 ppx/rho preheld boundaries",
                    "solve3 held-rate recapture",
                    "solve3 physical terrain momentum current-to-A",
                    "solve3 horizontal bc2d and padded momentum current-to-A",
                ],
                "solve3_entry_source_order_complete": True,
                "historical_replay_consumed": False,
                "same_stream_live_chain": True,
                "ready_for": "live solve3 TKE current-to-A only",
                "global_psolver_tile_streamable": False,
                "full_forecast": False,
                "five_meter_meteorological_product": False,
                "claim_limit": GM1_TILESTREAM_SOLVE3_ENTRY_CLAIM_LIMIT,
            }
        )
    )
    return {
        "token": entry,
        "completed_rk3_token": completed,
        "tke_held_token": tke_held,
        "tke_rk3_token": tke_rk3,
        "preheld_boundary_token": preheld,
        "held_recapture_token": held,
        "physical_momentum_token": physical,
        "padded_momentum_token": padded,
    }


def test_exact_live_rk3_to_solve3_lineage_is_accepted() -> None:
    fixture = _fixtures()
    receipt = require_gm1_tiled_solve3_entry_token(**fixture)
    assert receipt["schema"] == CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_SCHEMA
    assert receipt["historical_replay_consumed"] is False
    assert receipt["same_stream_live_chain"] is True
    assert list(receipt["lineage_operation_token_sha256"]) == [
        "completed_rk3",
        "tke_held_state",
        "tke_rk3",
        "preheld_boundary",
        "held_recapture",
        "physical_momentum_a",
        "padded_momentum_a",
    ]


def test_foreign_entry_and_tampered_lineage_are_refused() -> None:
    fixture = _fixtures()
    fixture["token"] = object()
    with pytest.raises(TypeError, match="exact Gm1TiledSolve3EntryToken"):
        require_gm1_tiled_solve3_entry_token(**fixture)
    fixture = _fixtures()
    receipt = dict(fixture["token"].receipt)
    receipt["lineage_operation_token_sha256"] = {
        **receipt["lineage_operation_token_sha256"],
        "tke_rk3": "0" * 64,
    }
    receipt.pop("operation_token_sha256")
    fixture["token"] = Gm1TiledSolve3EntryToken(_tokenize(receipt))
    with pytest.raises(ValueError, match="lineage inventory"):
        require_gm1_tiled_solve3_entry_token(**fixture)


def test_broader_rk3_must_bind_exact_tke_rk3() -> None:
    fixture = _fixtures()
    completed = dict(fixture["completed_rk3_token"].receipt)
    completed["component_operation_token_sha256"] = {
        **completed["component_operation_token_sha256"],
        "tke_rk": "0" * 64,
    }
    completed.pop("operation_token_sha256")
    fixture["completed_rk3_token"] = Gm1TiledCompletedRkStageToken(
        _tokenize(completed)
    )
    with pytest.raises(ValueError, match="does not bind"):
        require_gm1_tiled_solve3_entry_token(**fixture)


def test_contract_is_validator_only_and_detached_evidence_is_terminal() -> None:
    contract = solve3_entry_contract()
    assert contract["producer_available"] is False
    assert contract["detached_evidence_admissible"] is False
    assert "src/solve3.F:563-578" in AUTHORITY_ANCHORS
