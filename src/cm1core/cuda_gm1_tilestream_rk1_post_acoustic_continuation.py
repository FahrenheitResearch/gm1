"""Exact same-residency handoff from RK1 sound to tiled post-sound work.

The global acoustic solve is owned by :mod:`cuda_gm1_tilestream_rk1_acoustic`.
This module binds its live handoff to the exact solve1 TKE object needed by the
existing tiled getdivx/TKE chronology.  It performs no kernel launch and emits
no mass-flux, scalar-RK, TKE-RK, or completed-stage token.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
    Gm1PostSoundVelocityToken,
)
from .cuda_gm1_tilestream_pre_acoustic_momentum_rk import (
    FinishedGm1TiledPreAcousticMomentumRk,
    Gm1TiledPreAcousticMomentumRkBinding,
    Gm1TiledPreAcousticMomentumRkToken,
    Gm1TiledRk1AcousticContinuationToken,
)
from .cuda_gm1_tilestream_rk1_acoustic import Gm1TiledRk1AcousticLiveHandoff
from .cuda_gm1_tilestream_rk1_pre_acoustic_lineage import (
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
    CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS,
)
from .cuda_gm1_tilestream_solve1_full_held import (
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
    Gm1TiledSolve1FullHeldToken,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
)


GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_SCHEMA = (
    "gm1.tilestream-rk1-post-acoustic-live-binding/v1"
)
GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_STATUS = (
    "BOUND_LIVE_RK1_POST_ACOUSTIC_SAME_RESIDENCY"
)
GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_CLAIM_LIMIT = (
    "exact same-process lineage and residency after the admitted real 300x300x75 "
    "RK1 global acoustic solve; no getdiv/getdivx, theta divergence, scalar or "
    "moisture RK, TKE RK, boundary completion, completed RK stage, solve3, "
    "checkpoint, forecast, or five-metre product is emitted here"
)


def _json_default(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _require_receipt(
    name: str,
    token: object,
    *,
    exact_type: type[Any],
    schema: str,
    status: str,
) -> Mapping[str, object]:
    if type(token) is not exact_type:
        raise TypeError(f"{name} must be exact {exact_type.__name__}")
    receipt = token.receipt
    operation = receipt.get("operation_token_sha256")
    if (
        receipt.get("schema") != schema
        or receipt.get("status") != status
        or not isinstance(operation, str)
        or len(operation) != 64
        or any(character not in "0123456789abcdef" for character in operation)
        or operation != _operation_token(receipt)
    ):
        raise ValueError(f"{name} identity changed")
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1TiledRk1PostAcousticContinuationBinding:
    acoustic_handoff: Gm1TiledRk1AcousticLiveHandoff
    full_held_token: Gm1TiledSolve1FullHeldToken
    solve1_tke_token: Gm1TiledSolve1TkeRhsToken
    pre_acoustic_result: FinishedGm1TiledPreAcousticMomentumRk
    pre_acoustic_token: Gm1TiledPreAcousticMomentumRkToken
    acoustic_token: Gm1TiledRk1AcousticContinuationToken
    post_sound_velocity_token: Gm1PostSoundVelocityToken
    arrays: Mapping[str, Any]
    metrics: Mapping[str, Any]
    vertical: Any
    stream: Any
    bridge: Any
    terrain_metrics: Any
    stage: Any
    receipt: Mapping[str, object]


def bind_gm1_tiled_rk1_post_acoustic_continuation(
    handoff: Gm1TiledRk1AcousticLiveHandoff,
) -> Gm1TiledRk1PostAcousticContinuationBinding:
    """Bind exact sound output to the original full-held/TKE live objects."""

    if type(handoff) is not Gm1TiledRk1AcousticLiveHandoff:
        raise TypeError("handoff must be exact Gm1TiledRk1AcousticLiveHandoff")
    pre_result = handoff.pre_acoustic_result
    if type(pre_result) is not FinishedGm1TiledPreAcousticMomentumRk:
        raise TypeError("acoustic handoff lost exact pre-acoustic result")
    full_held_token = handoff.full_held_token
    solve1_tke_token = getattr(handoff, "solve1_tke_token", None)
    pre_token = handoff.pre_acoustic_token
    acoustic_token = handoff.acoustic_token
    velocity_token = handoff.post_sound_velocity_token
    full = _require_receipt(
        "full_held_token",
        full_held_token,
        exact_type=Gm1TiledSolve1FullHeldToken,
        schema=CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
        status=CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
    )
    solve1 = _require_receipt(
        "solve1_tke_token",
        solve1_tke_token,
        exact_type=Gm1TiledSolve1TkeRhsToken,
        schema=CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
        status="ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS",
    )
    pre = _require_receipt(
        "pre_acoustic_token",
        pre_token,
        exact_type=Gm1TiledPreAcousticMomentumRkToken,
        schema=CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
        status=CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
    )
    acoustic = _require_receipt(
        "acoustic_token",
        acoustic_token,
        exact_type=Gm1TiledRk1AcousticContinuationToken,
        schema=CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA,
        status=CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS,
    )
    velocity = _require_receipt(
        "post_sound_velocity_token",
        velocity_token,
        exact_type=Gm1PostSoundVelocityToken,
        schema=CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
        status="ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT",
    )
    operation = {
        "full": full["operation_token_sha256"],
        "solve1": solve1["operation_token_sha256"],
        "pre": pre["operation_token_sha256"],
        "acoustic": acoustic["operation_token_sha256"],
        "velocity": velocity["operation_token_sha256"],
    }
    if (
        type(pre_result.binding) is not Gm1TiledPreAcousticMomentumRkBinding
        or pre_result.full_held_token is not full_held_token
        or pre_result.solve1_tke_token is not solve1_tke_token
        or pre_result.binding.full_held_token is not full_held_token
        or pre_result.binding.solve1_tke_token is not solve1_tke_token
        or pre_result.token is not pre_token
        or handoff.arrays is not pre_result.arrays
        or handoff.metrics is not pre_result.metrics
        or handoff.vertical is not pre_result.vertical
        or handoff.stream is not pre_result.stream
        or handoff.bridge is not pre_result.bridge
        or handoff.terrain_metrics is not pre_result.terrain_metrics
    ):
        raise ValueError("post-acoustic live object identity or residency changed")
    stream_pointer = int(handoff.stream.ptr)
    if (
        full.get("solve1_tke_operation_token_sha256") != operation["solve1"]
        or pre.get("solve1_full_held_token_sha256") != operation["full"]
        or pre.get("solve1_tke_rhs_token_sha256") != operation["solve1"]
        or acoustic.get("pre_acoustic_momentum_rk_token_sha256") != operation["pre"]
        or acoustic.get("post_sound_velocity_token_sha256") != operation["velocity"]
        or velocity.get("pre_acoustic_momentum_rk_token_sha256") != operation["pre"]
        or velocity.get("solve1_full_held_token_sha256") != operation["full"]
        or velocity.get("solve1_tke_rhs_token_sha256") != operation["solve1"]
    ):
        raise ValueError("post-acoustic token lineage changed")
    stage_identity = pre.get("stage_identity_sha256")
    if (
        acoustic.get("stage_identity_sha256") != stage_identity
        or velocity.get("stage_identity_sha256") != stage_identity
        or any(
            receipt.get("stream_pointer") != stream_pointer
            for receipt in (full, solve1, pre, acoustic, velocity)
        )
    ):
        raise ValueError("post-acoustic stage identity or stream changed")

    receipt: dict[str, object] = {
        "schema": GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_SCHEMA,
        "status": GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_STATUS,
        "nrk": 1,
        "stage_identity_sha256": stage_identity,
        "stream_pointer": stream_pointer,
        "solve1_full_held_token_sha256": operation["full"],
        "solve1_tke_rhs_token_sha256": operation["solve1"],
        "pre_acoustic_momentum_rk_token_sha256": operation["pre"],
        "acoustic_continuation_token_sha256": operation["acoustic"],
        "post_sound_velocity_token_sha256": operation["velocity"],
        "same_live_residency": True,
        "getdiv_getdivx_performed": False,
        "theta_divergence_performed": False,
        "completed_rk_stage": False,
        "claim_limit": GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledRk1PostAcousticContinuationBinding(
        handoff,
        full_held_token,
        solve1_tke_token,
        pre_result,
        pre_token,
        acoustic_token,
        velocity_token,
        handoff.arrays,
        handoff.metrics,
        handoff.vertical,
        handoff.stream,
        handoff.bridge,
        handoff.terrain_metrics,
        handoff.stage,
        MappingProxyType(receipt),
    )


__all__ = (
    "GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_CLAIM_LIMIT",
    "GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_SCHEMA",
    "GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_STATUS",
    "Gm1TiledRk1PostAcousticContinuationBinding",
    "bind_gm1_tiled_rk1_post_acoustic_continuation",
)
