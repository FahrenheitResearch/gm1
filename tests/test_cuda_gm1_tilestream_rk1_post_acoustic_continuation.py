from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

from cm1core import cuda_gm1_tilestream_rk1_post_acoustic_continuation as post
from cm1core.cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
    Gm1PostSoundVelocityToken,
)
from cm1core.cuda_gm1_tilestream_pre_acoustic_momentum_rk import (
    FinishedGm1TiledPreAcousticMomentumRk,
    Gm1TiledPreAcousticMomentumRkBinding,
    Gm1TiledPreAcousticMomentumRkToken,
    Gm1TiledRk1AcousticContinuationToken,
)
from cm1core.cuda_gm1_tilestream_rk1_pre_acoustic_lineage import (
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
    CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS,
)
from cm1core.cuda_gm1_tilestream_solve1_full_held import (
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
    Gm1TiledSolve1FullHeldToken,
)
from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _seal(receipt: dict[str, object]) -> MappingProxyType:
    receipt["operation_token_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return MappingProxyType(receipt)


def _live_handoff() -> post.Gm1TiledRk1AcousticLiveHandoff:
    stream = SimpleNamespace(ptr=73)
    stage = "a" * 64
    solve1 = Gm1TiledSolve1TkeRhsToken(
        _seal(
            {
                "schema": CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS",
                "completed_nrk": 0,
                "stream_pointer": 73,
            }
        )
    )
    full = Gm1TiledSolve1FullHeldToken(
        _seal(
            {
                "schema": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
                "status": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
                "completed_nrk": 0,
                "stream_pointer": 73,
                "solve1_tke_operation_token_sha256": solve1.receipt[
                    "operation_token_sha256"
                ],
            }
        )
    )
    pre_token = Gm1TiledPreAcousticMomentumRkToken(
        _seal(
            {
                "schema": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
                "status": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
                "nrk": 1,
                "stream_pointer": 73,
                "stage_identity_sha256": stage,
                "solve1_full_held_token_sha256": full.receipt[
                    "operation_token_sha256"
                ],
                "solve1_tke_rhs_token_sha256": solve1.receipt[
                    "operation_token_sha256"
                ],
            }
        )
    )
    velocity = Gm1PostSoundVelocityToken(
        _seal(
            {
                "schema": CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
                "status": "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT",
                "nrk": 1,
                "stream_pointer": 73,
                "stage_identity_sha256": stage,
                "pre_acoustic_momentum_rk_token_sha256": pre_token.receipt[
                    "operation_token_sha256"
                ],
                "solve1_full_held_token_sha256": full.receipt[
                    "operation_token_sha256"
                ],
                "solve1_tke_rhs_token_sha256": solve1.receipt[
                    "operation_token_sha256"
                ],
            }
        )
    )
    acoustic = Gm1TiledRk1AcousticContinuationToken(
        _seal(
            {
                "schema": CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA,
                "status": CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS,
                "nrk": 1,
                "stream_pointer": 73,
                "stage_identity_sha256": stage,
                "pre_acoustic_momentum_rk_token_sha256": pre_token.receipt[
                    "operation_token_sha256"
                ],
                "post_sound_velocity_token_sha256": velocity.receipt[
                    "operation_token_sha256"
                ],
            }
        )
    )
    arrays, metrics, vertical, bridge, terrain = (
        MappingProxyType({"arrays": object()}),
        MappingProxyType({"metrics": object()}),
        object(),
        object(),
        object(),
    )
    pre_binding = Gm1TiledPreAcousticMomentumRkBinding(
        full,
        solve1,
        arrays,
        metrics,
        terrain,  # type: ignore[arg-type]
        vertical,  # type: ignore[arg-type]
        stream,
        bridge,  # type: ignore[arg-type]
        object(),
        stage,
        MappingProxyType({}),
    )
    pre_result = FinishedGm1TiledPreAcousticMomentumRk(
        pre_token,
        pre_binding,
        full,
        solve1,
        arrays,
        metrics,
        terrain,  # type: ignore[arg-type]
        vertical,  # type: ignore[arg-type]
        stream,
        bridge,  # type: ignore[arg-type]
        MappingProxyType({}),
        object(),  # type: ignore[arg-type]
    )
    return post.Gm1TiledRk1AcousticLiveHandoff(
        ready_event=object(),
        stage=object(),  # type: ignore[arg-type]
        arrays=arrays,
        metrics=metrics,
        vertical=vertical,
        stream=stream,
        bridge=bridge,
        terrain_metrics=terrain,  # type: ignore[arg-type]
        full_held_token=full,
        solve1_tke_token=solve1,
        pre_acoustic_result=pre_result,
        pre_acoustic_token=pre_token,
        acoustic_token=acoustic,
        post_sound_velocity_token=velocity,
    )


def test_exact_live_handoff_binds_every_upstream_token_by_identity(
) -> None:
    handoff = _live_handoff()
    binding = post.bind_gm1_tiled_rk1_post_acoustic_continuation(handoff)
    assert binding.acoustic_handoff is handoff
    assert binding.full_held_token is handoff.full_held_token
    assert binding.solve1_tke_token is handoff.solve1_tke_token
    assert binding.pre_acoustic_result is handoff.pre_acoustic_result
    assert binding.arrays is handoff.arrays
    assert binding.metrics is handoff.metrics
    assert binding.stream is handoff.stream
    assert binding.receipt["status"] == post.GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_STATUS
    assert binding.receipt["getdiv_getdivx_performed"] is False
    assert binding.receipt["completed_rk_stage"] is False


def test_substituted_solve1_object_refuses_even_with_valid_receipt(
) -> None:
    handoff = _live_handoff()
    substituted = replace(
        handoff,
        solve1_tke_token=Gm1TiledSolve1TkeRhsToken(
            handoff.solve1_tke_token.receipt
        ),
    )
    with pytest.raises(ValueError, match="object identity or residency"):
        post.bind_gm1_tiled_rk1_post_acoustic_continuation(substituted)


def test_foreign_handoff_and_completed_stage_claim_refuse(
) -> None:
    with pytest.raises(TypeError, match="exact Gm1TiledRk1AcousticLiveHandoff"):
        post.bind_gm1_tiled_rk1_post_acoustic_continuation(object())
    claim = post.GM1_RK1_POST_ACOUSTIC_LIVE_BINDING_CLAIM_LIMIT
    assert "no getdiv/getdivx" in claim
    assert "completed RK stage" in claim
    assert "five-metre product" in claim
