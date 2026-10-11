"""Bind CM1 RK1 TKE only after the source-ordered scalar boundaries.

The numerical TKE operator already has its own exact tile token.  This module
adds the chronology that token intentionally cannot prove: for CM1 r21.1 the
RK1 ``bcp/bcs`` publication precedes TKE ``advw``/RK/clip/``bcw``.  The result
is one narrow ``tke_rk`` component token for a future broader-stage
aggregator.  It does not emit that broader stage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .cuda_gm1_tilestream_rk1_scalar_pressure_boundaries import (
    CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS,
    Gm1TiledRk1ScalarPressureBoundaryToken,
)
from .cuda_gm1_tilestream_tke_rk1 import (
    AUTHORITY_ADV_ROUTINES_SHA256,
    AUTHORITY_ADV_SHA256,
    AUTHORITY_SOLVE2_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA,
    Gm1TiledTkeRk1Token,
)
from .cuda_gm1_tilestream_tke_stage_state import (
    CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
    Gm1TiledTkeHeldStageToken,
)


CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA = (
    "gm1.rk1-source-ordered-tke-token/v1"
)
CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS = (
    "ADMITTED_LIVE_RK1_SOURCE_ORDERED_TKE_COMPONENT"
)
SOURCE_ORDER = (
    "completed RK1 momentum then scalar/pressure boundary tokens",
    "copy held solve1 tketen into the TKE advw tendency",
    "advw(stag=4) with same-stage post-acoustic rru/rrv/rrw/divx",
    "tke3d=tkea+(dt/3)*tendency then clip below 1e-10",
    "bcw(tke3d,1) then bind fixed tkea/held tketen/current tke3d",
)
GM1_RK1_SOURCE_ORDERED_TKE_CLAIM_LIMIT = (
    "one live source-ordered CM1 r21.1 RK1 TKE component after the exact "
    "scalar/pressure boundary token, binding the tiled advw/RK1 token and "
    "held tkea/tketen/tke3d state on one stream. It does not prove the "
    "pre-acoustic momentum RK component, emit a completed broader RK stage, "
    "run RK2/RK3 or solve3, tile a global pressure solver, perform MPI/open "
    "history work, or establish a forecast or five-metre product"
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


def _require_receipt(
    token: object,
    token_type: type,
    *,
    schema: str,
    status: str,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(token, token_type):
        raise TypeError(f"{label} must use the exact {token_type.__name__} type")
    receipt = token.receipt  # type: ignore[attr-defined]
    if (
        receipt.get("schema") != schema
        or receipt.get("status") != status
        or receipt.get("completed_nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError(f"{label} identity or cadence changed")
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1Rk1SourceOrderedTkeToken:
    receipt: Mapping[str, object]


def bind_gm1_rk1_source_ordered_tke(
    scalar_boundary_token: Gm1TiledRk1ScalarPressureBoundaryToken,
    tke_rk1_token: Gm1TiledTkeRk1Token,
    held_tke_token: Gm1TiledTkeHeldStageToken,
) -> Gm1Rk1SourceOrderedTkeToken:
    """Bind the exact post-boundary TKE chronology without broadening scope."""

    boundary = _require_receipt(
        scalar_boundary_token,
        Gm1TiledRk1ScalarPressureBoundaryToken,
        schema=CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA,
        status=CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS,
        label="scalar_boundary_token",
    )
    tke = _require_receipt(
        tke_rk1_token,
        Gm1TiledTkeRk1Token,
        schema=CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA,
        status="ADMITTED_LIVE_TILESTREAM_TKE_ADVW_RK1",
        label="tke_rk1_token",
    )
    held = _require_receipt(
        held_tke_token,
        Gm1TiledTkeHeldStageToken,
        schema=CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
        status="ADMITTED_LIVE_TKE_HELD_STATE_AFTER_RK1",
        label="held_tke_token",
    )
    if (
        boundary.get("post_acoustic_flux_token_sha256")
        != tke.get("post_acoustic_flux_token_sha256")
        or held.get("tke_rk1_token_sha256")
        != tke.get("operation_token_sha256")
        or held.get("current_tke3d_sha256") != tke.get("output_tke3d_sha256")
        or boundary.get("stage_identity_sha256")
        != tke.get("stage_identity_sha256")
        or held.get("rk1_stage_identity_sha256")
        != tke.get("stage_identity_sha256")
        or len(
            {
                boundary.get("stream_pointer"),
                tke.get("stream_pointer"),
                held.get("stream_pointer"),
            }
        )
        != 1
    ):
        raise ValueError("post-boundary TKE token lineage or stream changed")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS,
        "completed_nrk": 1,
        "next_nrk": 2,
        "stage_identity_sha256": tke["stage_identity_sha256"],
        "stream_pointer": tke["stream_pointer"],
        "scalar_pressure_boundary_token_sha256": boundary[
            "operation_token_sha256"
        ],
        "tke_rk1_token_sha256": tke["operation_token_sha256"],
        "tke_held_state_token_sha256": held["operation_token_sha256"],
        "output_tke3d_sha256": tke["output_tke3d_sha256"],
        "component_key": "tke_rk",
        "source_order": list(SOURCE_ORDER),
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
        "missing_components": [
            "typed full pre-acoustic momentum RK component",
            "broader completed-RK-stage aggregator",
        ],
        "global_psolver_tile_streamable": False,
        "mpi_performed": False,
        "open_history_mutated": False,
        "full_forecast": False,
        "five_meter_meteorological_product": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "adv_sha256": AUTHORITY_ADV_SHA256,
            "adv_routines_sha256": AUTHORITY_ADV_ROUTINES_SHA256,
            "anchors": [
                "src/solve2.F:1662-1803",
                "src/adv.F:2778-3610",
                "src/bc.F:942-1143",
            ],
        },
        "claim_limit": GM1_RK1_SOURCE_ORDERED_TKE_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1Rk1SourceOrderedTkeToken(MappingProxyType(receipt))


__all__ = (
    "CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA",
    "CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS",
    "GM1_RK1_SOURCE_ORDERED_TKE_CLAIM_LIMIT",
    "Gm1Rk1SourceOrderedTkeToken",
    "SOURCE_ORDER",
    "bind_gm1_rk1_source_ordered_tke",
)
