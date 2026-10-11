"""Fail-closed live transition from completed RK3 into CM1 ``solve3``.

Historical solve3 tile receipts prove individual operators, but they cannot be
relabelled as a live continuation of a newly completed RK3 state.  This module
freezes the exact cross-component token boundary that an eventual live
aggregator must emit: completed broader RK3 and TKE RK3 on one side, then the
source-ordered preheld, held-recapture, physical momentum-A, and padded
momentum-A chain on the other.

No producer exists here.  The validator lets downstream TKE current-to-A work
depend on one exact type without accepting detached evidence or duck-typed
receipts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream_completed_rk_stage import (
    GM1_COMPLETED_RK_STAGE_TKE_COMPONENT_KEY,
    Gm1TiledCompletedRkStageToken,
    require_gm1_tiled_completed_rk_stage_token,
)
from .cuda_gm1_tilestream_solve3_boundaries import (
    Gm1TiledSolve3PreHeldBoundaryToken,
)
from .cuda_gm1_tilestream_solve3_held import Gm1TiledSolve3HeldToken
from .cuda_gm1_tilestream_solve3_momentum_a import (
    Gm1TiledSolve3PhysicalMomentumAToken,
)
from .cuda_gm1_tilestream_solve3_padded_momentum_a import (
    Gm1TiledSolve3PaddedMomentumAToken,
)
from .cuda_gm1_tilestream_tke_rk23 import (
    CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA,
    Gm1TiledTkeRk23Token,
)
from .cuda_gm1_tilestream_tke_stage_state import (
    CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
    Gm1TiledTkeHeldStageToken,
)


CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_SCHEMA = (
    "gm1.tilestream-live-solve3-entry-token/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_STATUS = (
    "ADMITTED_LIVE_SOLVE3_ENTRY_AFTER_COMPLETED_RK3"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE3_SHA256 = (
    "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"
)
AUTHORITY_ANCHORS = (
    "src/solve3.F:19-337",
    "src/solve3.F:339-559",
    "src/solve3.F:563-578",
)
GM1_TILESTREAM_SOLVE3_ENTRY_CLAIM_LIMIT = (
    "typed live source-order identity joining completed broader RK3 and TKE "
    "RK3 to the serial preheld-boundary, held-recapture, physical momentum-A, "
    "and padded momentum-A chain on one CUDA stream. The type emits no chain, "
    "performs no TKE current-to-A copy, reciprocal-density rebuild, pressure "
    "solve, MPI exchange, open-history mutation, checkpoint, full forecast, "
    "or five-metre meteorological product"
)
_LINEAGE_KEYS = (
    "completed_rk3",
    "tke_held_state",
    "tke_rk3",
    "preheld_boundary",
    "held_recapture",
    "physical_momentum_a",
    "padded_momentum_a",
)
_SOURCE_ORDER = (
    "completed broader solve2 RK3",
    "completed TKE advw/RK3/bcw",
    "solve3 ppx/rho preheld boundaries",
    "solve3 held-rate recapture",
    "solve3 physical terrain momentum current-to-A",
    "solve3 horizontal bc2d and padded momentum current-to-A",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _lower_sha(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex sha256")
    return value


def _receipt(token: object, token_type: type[Any], name: str) -> Mapping[str, Any]:
    if not isinstance(token, token_type):
        raise TypeError(f"{name} must be the exact {token_type.__name__} type")
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError(f"{name} receipt must be a mapping")
    if receipt.get("operation_token_sha256") != _operation_token(receipt):
        raise ValueError(f"{name} operation token changed")
    return receipt


def _require_tke_held(token: object) -> Mapping[str, Any]:
    receipt = _receipt(token, Gm1TiledTkeHeldStageToken, "tke_held_token")
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TKE_HELD_STATE_AFTER_RK1"
        or receipt.get("completed_nrk") != 1
    ):
        raise ValueError("held TKE state identity changed")
    _lower_sha(
        "fresh TKE baseline token",
        receipt.get("closure_tkea_baseline_token_sha256"),
    )
    return receipt


def _require_tke_rk3(
    token: object, held: Mapping[str, Any]
) -> Mapping[str, Any]:
    receipt = _receipt(token, Gm1TiledTkeRk23Token, "tke_rk3_token")
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILESTREAM_TKE_RK_STAGE"
        or receipt.get("completed_nrk") != 3
        or receipt.get("held_tke_state_token_sha256")
        != held.get("operation_token_sha256")
        or receipt.get("stream_pointer") != held.get("stream_pointer")
        or receipt.get("final_bcw_applied") is not True
    ):
        raise ValueError("completed TKE RK3 identity, held state, or stream changed")
    _lower_sha("TKE RK3 stage identity", receipt.get("stage_identity_sha256"))
    _lower_sha("TKE RK3 output", receipt.get("output_tke3d_sha256"))
    return receipt


def _require_preheld(token: object, stream_pointer: int) -> Mapping[str, Any]:
    receipt = _receipt(
        token, Gm1TiledSolve3PreHeldBoundaryToken, "preheld_boundary_token"
    )
    if (
        receipt.get("schema") != "gm1.tilestream-solve3-preheld-boundary-token/v1"
        or receipt.get("status")
        != "ADMITTED_PPX_RHO_BOUNDARIES_READY_FOR_HELD_RECAPTURE_ONLY"
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("ready_for") != "held-rate recapture admission only"
        or receipt.get("operation_b_current_to_a_available") is not False
        or receipt.get("mpi_performed") is not False
    ):
        raise ValueError("preheld boundary identity, stream, or scope changed")
    _lower_sha("solve3 source identity", receipt.get("source_identity_sha256"))
    return receipt


def _require_held_recapture(
    token: object, preheld: Mapping[str, Any], stream_pointer: int
) -> Mapping[str, Any]:
    receipt = _receipt(token, Gm1TiledSolve3HeldToken, "held_recapture_token")
    if (
        receipt.get("schema") != "gm1.tilestream-solve3-held-token/v1"
        or receipt.get("status")
        != "ADMITTED_HELD_RATES_READY_FOR_MOMENTUM_CURRENT_TO_A_ONLY"
        or receipt.get("preheld_operation_token_sha256")
        != preheld.get("operation_token_sha256")
        or receipt.get("source_identity_sha256")
        != preheld.get("source_identity_sha256")
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("ready_for") != "momentum current-to-A admission only"
        or receipt.get("tke_commit_performed") is not False
    ):
        raise ValueError("held recapture identity, lineage, stream, or scope changed")
    return receipt


def _require_physical_momentum(
    token: object, held: Mapping[str, Any], stream_pointer: int
) -> Mapping[str, Any]:
    receipt = _receipt(
        token, Gm1TiledSolve3PhysicalMomentumAToken, "physical_momentum_token"
    )
    if (
        receipt.get("schema")
        != "gm1.tilestream-solve3-physical-momentum-a-token/v1"
        or receipt.get("status") != "ADMITTED_PHYSICAL_MOMENTUM_A_ONLY"
        or receipt.get("held_operation_token_sha256")
        != held.get("operation_token_sha256")
        or receipt.get("source_identity_sha256")
        != held.get("source_identity_sha256")
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("ready_for")
        != "horizontal bc2d and padded momentum-A completion only"
        or receipt.get("tke_commit_available") is not False
    ):
        raise ValueError("physical momentum-A identity, lineage, or stream changed")
    return receipt


def _require_padded_momentum(
    token: object, physical: Mapping[str, Any], stream_pointer: int
) -> Mapping[str, Any]:
    receipt = _receipt(
        token, Gm1TiledSolve3PaddedMomentumAToken, "padded_momentum_token"
    )
    if (
        receipt.get("schema")
        != "gm1.tilestream-solve3-padded-momentum-a-token/v1"
        or receipt.get("status") != "ADMITTED_PADDED_MOMENTUM_A_ONLY"
        or receipt.get("physical_operation_token_sha256")
        != physical.get("operation_token_sha256")
        or receipt.get("source_identity_sha256")
        != physical.get("source_identity_sha256")
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("ready_for") != "independent TKE-commit admission audit only"
        or receipt.get("tke_commit_available") is not False
    ):
        raise ValueError("padded momentum-A identity, lineage, or stream changed")
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3EntryToken:
    """Reserved exact output type for the future live transition aggregator."""

    receipt: Mapping[str, object]


def require_gm1_tiled_solve3_entry_token(
    token: object,
    *,
    completed_rk3_token: Gm1TiledCompletedRkStageToken,
    tke_held_token: Gm1TiledTkeHeldStageToken,
    tke_rk3_token: Gm1TiledTkeRk23Token,
    preheld_boundary_token: Gm1TiledSolve3PreHeldBoundaryToken,
    held_recapture_token: Gm1TiledSolve3HeldToken,
    physical_momentum_token: Gm1TiledSolve3PhysicalMomentumAToken,
    padded_momentum_token: Gm1TiledSolve3PaddedMomentumAToken,
) -> Mapping[str, object]:
    """Validate one exact live solve3 entry without emitting it."""

    held_tke = _require_tke_held(tke_held_token)
    tke_rk3 = _require_tke_rk3(tke_rk3_token, held_tke)
    stream_pointer = int(tke_rk3["stream_pointer"])
    completed = require_gm1_tiled_completed_rk_stage_token(
        completed_rk3_token,
        completed_nrk=3,
        stream_pointer=stream_pointer,
        stage_identity_sha256=str(tke_rk3["stage_identity_sha256"]),
    )
    if completed["component_operation_token_sha256"].get(
        GM1_COMPLETED_RK_STAGE_TKE_COMPONENT_KEY
    ) != tke_rk3.get("operation_token_sha256"):
        raise ValueError("completed broader RK3 does not bind the exact TKE RK3")
    preheld = _require_preheld(preheld_boundary_token, stream_pointer)
    held = _require_held_recapture(held_recapture_token, preheld, stream_pointer)
    physical = _require_physical_momentum(
        physical_momentum_token, held, stream_pointer
    )
    padded = _require_padded_momentum(
        padded_momentum_token, physical, stream_pointer
    )

    if not isinstance(token, Gm1TiledSolve3EntryToken):
        raise TypeError("token must be the exact Gm1TiledSolve3EntryToken type")
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("solve3 entry receipt must be a mapping")
    expected_lineage = {
        "completed_rk3": completed["operation_token_sha256"],
        "tke_held_state": held_tke["operation_token_sha256"],
        "tke_rk3": tke_rk3["operation_token_sha256"],
        "preheld_boundary": preheld["operation_token_sha256"],
        "held_recapture": held["operation_token_sha256"],
        "physical_momentum_a": physical["operation_token_sha256"],
        "padded_momentum_a": padded["operation_token_sha256"],
    }
    lineage = receipt.get("lineage_operation_token_sha256")
    if not isinstance(lineage, Mapping) or dict(lineage) != expected_lineage:
        raise ValueError("solve3 entry lineage inventory changed")
    if tuple(lineage) != _LINEAGE_KEYS:
        raise ValueError("solve3 entry lineage order changed")
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_STATUS
        or receipt.get("completed_nrk") != 3
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("stage_identity_sha256")
        != tke_rk3.get("stage_identity_sha256")
        or receipt.get("source_identity_sha256")
        != preheld.get("source_identity_sha256")
        or receipt.get("source_order") != list(_SOURCE_ORDER)
        or receipt.get("solve3_entry_source_order_complete") is not True
        or receipt.get("historical_replay_consumed") is not False
        or receipt.get("same_stream_live_chain") is not True
        or receipt.get("ready_for") != "live solve3 TKE current-to-A only"
        or receipt.get("global_psolver_tile_streamable") is not False
        or receipt.get("full_forecast") is not False
        or receipt.get("five_meter_meteorological_product") is not False
        or receipt.get("claim_limit") != GM1_TILESTREAM_SOLVE3_ENTRY_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("solve3 entry identity, source order, or claim changed")
    return MappingProxyType(dict(receipt))


def solve3_entry_contract() -> Mapping[str, object]:
    """Publish the frozen boundary without pretending an emitter exists."""

    return MappingProxyType(
        {
            "schema": CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_SCHEMA,
            "status": CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_STATUS,
            "lineage_keys": list(_LINEAGE_KEYS),
            "source_order": list(_SOURCE_ORDER),
            "producer_available": False,
            "detached_evidence_admissible": False,
            "authority": {
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "solve3_sha256": AUTHORITY_SOLVE3_SHA256,
                "anchors": list(AUTHORITY_ANCHORS),
            },
            "claim_limit": GM1_TILESTREAM_SOLVE3_ENTRY_CLAIM_LIMIT,
        }
    )


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_SOLVE3_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE3_ENTRY_TOKEN_STATUS",
    "GM1_TILESTREAM_SOLVE3_ENTRY_CLAIM_LIMIT",
    "Gm1TiledSolve3EntryToken",
    "require_gm1_tiled_solve3_entry_token",
    "solve3_entry_contract",
)
