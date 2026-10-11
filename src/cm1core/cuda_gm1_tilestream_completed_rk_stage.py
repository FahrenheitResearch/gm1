"""Type boundary for one completed tile-streamed CM1 RK stage.

A completed CM1 ``solve2`` stage is broader than any one momentum, scalar,
moisture, pressure, boundary, or TKE operator.  This module owns only the
shared token type and validator.  The source-order aggregator constructs the
token after every required component supplies an exact live operation token
on one CUDA stream; downstream stages consume and revalidate this exact type.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA = (
    "gm1.tilestream-completed-rk-stage-token/v1"
)
CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_COMPLETED_RK_STAGE"
)
GM1_COMPLETED_RK_STAGE_TKE_COMPONENT_KEY = "tke_rk"
GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT = (
    "typed identity for one source-ordered live CM1 r21.1 tile-streamed RK "
    "stage only after momentum, thermodynamic, moisture, pressure/density, "
    "boundary, and TKE component tokens are aggregated on one stream. This "
    "type alone emits no stage, tiles no global pressure solver, performs no "
    "MPI exchange, creates no checkpoint or forecast, and does not admit the "
    "five-metre meteorological product"
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


@dataclass(frozen=True, slots=True)
class Gm1TiledCompletedRkStageToken:
    """Exact token class emitted only by a broad source-order aggregator."""

    receipt: Mapping[str, object]


def require_gm1_tiled_completed_rk_stage_token(
    token: object,
    *,
    completed_nrk: int,
    stream_pointer: int | None = None,
    stage_identity_sha256: str | None = None,
) -> Mapping[str, object]:
    """Validate the immutable common contract of a completed-stage token."""

    if not isinstance(token, Gm1TiledCompletedRkStageToken):
        raise TypeError(
            "token must be the exact Gm1TiledCompletedRkStageToken type"
        )
    if (
        isinstance(completed_nrk, bool)
        or not isinstance(completed_nrk, int)
        or completed_nrk not in (1, 2, 3)
    ):
        raise ValueError("completed_nrk must be 1, 2, or 3")
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("completed-stage receipt must be a mapping")
    components = receipt.get("component_operation_token_sha256")
    if not isinstance(components, Mapping) or not components:
        raise ValueError("completed-stage component token inventory is missing")
    checked_components: dict[str, str] = {}
    for key, value in components.items():
        if not isinstance(key, str) or not key:
            raise ValueError("completed-stage component token key is invalid")
        checked_components[key] = _lower_sha(
            f"component_operation_token_sha256[{key!r}]", value
        )
    if GM1_COMPLETED_RK_STAGE_TKE_COMPONENT_KEY not in checked_components:
        raise ValueError("completed-stage token does not bind the live TKE RK token")
    expected_next_nrk = completed_nrk + 1 if completed_nrk < 3 else None
    if (
        receipt.get("schema")
        != CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS
        or receipt.get("completed_nrk") != completed_nrk
        or receipt.get("next_nrk") != expected_next_nrk
        or receipt.get("source_order_complete") is not True
        or receipt.get("global_psolver_tile_streamable") is not False
        or receipt.get("five_meter_meteorological_product") is not False
        or receipt.get("full_forecast") is not False
        or receipt.get("claim_limit")
        != GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("completed-stage identity, cadence, or claim changed")
    actual_stream = receipt.get("stream_pointer")
    if (
        isinstance(actual_stream, bool)
        or not isinstance(actual_stream, int)
        or actual_stream <= 0
    ):
        raise ValueError("completed-stage stream pointer is invalid")
    actual_stage = _lower_sha(
        "completed-stage stage_identity_sha256",
        receipt.get("stage_identity_sha256"),
    )
    if stream_pointer is not None and actual_stream != stream_pointer:
        raise ValueError("completed-stage token belongs to another CUDA stream")
    if stage_identity_sha256 is not None and actual_stage != _lower_sha(
        "stage_identity_sha256", stage_identity_sha256
    ):
        raise ValueError("completed-stage token belongs to another RK state")
    return MappingProxyType(dict(receipt))


__all__ = (
    "CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS",
    "GM1_COMPLETED_RK_STAGE_TKE_COMPONENT_KEY",
    "GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT",
    "Gm1TiledCompletedRkStageToken",
    "require_gm1_tiled_completed_rk_stage_token",
)
