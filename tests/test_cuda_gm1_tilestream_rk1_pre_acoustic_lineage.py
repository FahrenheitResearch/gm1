from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from types import SimpleNamespace

import pytest

from cm1core.cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
    Gm1PostSoundVelocityToken,
)
from cm1core.cuda_gm1_tilestream_rk1_pre_acoustic_lineage import (
    CUDA_GM1_RK1_PRE_ACOUSTIC_LINEAGE_AUDIT_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
    CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS,
    CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_SCHEMA,
    CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_STATUS,
    GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS,
    GM1_RK1_PRE_ACOUSTIC_SOURCE_ORDER,
    audit_gm1_rk1_pre_acoustic_source_contract,
    refuse_historical_gm1_post_sound_velocity_as_full_lineage,
    require_gm1_rk1_pre_acoustic_acoustic_lineage,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: dict[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _historical_token() -> Gm1PostSoundVelocityToken:
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
        "status": "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT",
        "nrk": 1,
        "stage_identity_sha256": "1" * 64,
        "solve1_tke_rhs_token_sha256": "2" * 64,
        "psolver_receipt_sha256": "3" * 64,
        "stream_pointer": 17,
        "host_checkpoint_transport": True,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1PostSoundVelocityToken(MappingProxyType(receipt))


def test_source_contract_is_explicitly_non_runtime_and_non_token() -> None:
    audit = audit_gm1_rk1_pre_acoustic_source_contract()
    assert audit["schema"] == CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_SCHEMA
    assert audit["status"] == CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_STATUS
    assert audit["source_order"] == list(GM1_RK1_PRE_ACOUSTIC_SOURCE_ORDER)
    assert audit["required_output_sha256_keys"] == list(
        GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS
    )
    assert audit["historical_post_sound_velocity_token_accepted"] is False
    assert audit["positive_runtime_validator_implemented"] is True
    assert audit["positive_runtime_token_types_required"] is True
    assert audit["completed_stage_token_emitted"] is False
    assert "operation_token_sha256" not in audit
    assert "component_operation_token_sha256" not in audit


def test_historical_post_sound_token_is_validated_then_refused() -> None:
    with pytest.raises(RuntimeError, match="binds only solve1_tke_rhs"):
        refuse_historical_gm1_post_sound_velocity_as_full_lineage(
            _historical_token()
        )


def test_mutated_historical_token_is_rejected_before_lineage_refusal() -> None:
    token = _historical_token()
    changed = dict(token.receipt)
    changed["nrk"] = 2
    changed["operation_token_sha256"] = _operation_token(changed)
    mutated = Gm1PostSoundVelocityToken(MappingProxyType(changed))
    with pytest.raises(ValueError, match="identity or cadence changed"):
        refuse_historical_gm1_post_sound_velocity_as_full_lineage(mutated)


def test_foreign_historical_token_shape_is_not_duck_admitted() -> None:
    with pytest.raises(TypeError, match="exact historical"):
        refuse_historical_gm1_post_sound_velocity_as_full_lineage(
            _historical_token().receipt
        )


def test_positive_lineage_gate_refuses_historical_acoustic_token() -> None:
    with pytest.raises(RuntimeError, match="binds only solve1_tke_rhs"):
        require_gm1_rk1_pre_acoustic_acoustic_lineage(
            object(), _historical_token()
        )


def test_positive_lineage_gate_refuses_foreign_types_after_exact_module_lands() -> None:
    with pytest.raises(TypeError, match="exact Gm1TiledPreAcousticMomentumRkToken"):
        require_gm1_rk1_pre_acoustic_acoustic_lineage(object(), object())


def test_positive_lineage_gate_accepts_only_exact_reserved_types(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass(frozen=True)
    class Gm1TiledPreAcousticMomentumRkToken:
        receipt: MappingProxyType

    @dataclass(frozen=True)
    class Gm1TiledRk1AcousticContinuationToken:
        receipt: MappingProxyType

    fake_module = SimpleNamespace(
        Gm1TiledPreAcousticMomentumRkToken=Gm1TiledPreAcousticMomentumRkToken,
        Gm1TiledRk1AcousticContinuationToken=Gm1TiledRk1AcousticContinuationToken,
    )
    import cm1core.cuda_gm1_tilestream_rk1_pre_acoustic_lineage as lineage

    real_import = lineage.importlib.import_module

    def fake_import(name: str):
        if name == "cm1core.cuda_gm1_tilestream_pre_acoustic_momentum_rk":
            return fake_module
        return real_import(name)

    monkeypatch.setattr(lineage.importlib, "import_module", fake_import)
    output_shapes = {
        "uten": [75, 300, 301],
        "vten": [75, 301, 300],
        "wten": [76, 300, 300],
        "rru": [75, 300, 301],
        "rrv": [75, 301, 300],
        "rrw": [76, 300, 300],
        "divx": [75, 300, 300],
    }
    pre_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
        "nrk": 1,
        "stream_pointer": 31,
        "stage_identity_sha256": "4" * 64,
        "solve1_full_held_token_sha256": "5" * 64,
        "output_sha256": {name: str(index) * 64 for index, name in enumerate(output_shapes, 1)},
        "output_shapes": output_shapes,
    }
    pre_receipt["operation_token_sha256"] = _operation_token(pre_receipt)
    pre = Gm1TiledPreAcousticMomentumRkToken(MappingProxyType(pre_receipt))
    acoustic_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS,
        "nrk": 1,
        "stream_pointer": 31,
        "stage_identity_sha256": "4" * 64,
        "pre_acoustic_momentum_rk_token_sha256": pre_receipt[
            "operation_token_sha256"
        ],
        "post_sound_velocity_token_sha256": "9" * 64,
    }
    acoustic_receipt["operation_token_sha256"] = _operation_token(acoustic_receipt)
    acoustic = Gm1TiledRk1AcousticContinuationToken(
        MappingProxyType(acoustic_receipt)
    )
    audit = require_gm1_rk1_pre_acoustic_acoustic_lineage(pre, acoustic)
    assert audit["schema"] == CUDA_GM1_RK1_PRE_ACOUSTIC_LINEAGE_AUDIT_SCHEMA
    assert audit["source_order_complete_through_global_acoustic"] is True
    assert audit["completed_stage_token_emitted"] is False
    assert "operation_token_sha256" not in audit

    changed = dict(acoustic_receipt)
    changed["stream_pointer"] = 32
    changed["operation_token_sha256"] = _operation_token(changed)
    foreign = Gm1TiledRk1AcousticContinuationToken(MappingProxyType(changed))
    with pytest.raises(ValueError, match="lineage, stage, or stream changed"):
        require_gm1_rk1_pre_acoustic_acoustic_lineage(pre, foreign)
