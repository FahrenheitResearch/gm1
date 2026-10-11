from __future__ import annotations

from types import MappingProxyType

import pytest

from cm1core import cuda_gm1_tilestream_completed_rk_stage as completed


def _token(*, nrk: int = 1) -> completed.Gm1TiledCompletedRkStageToken:
    record: dict[str, object] = {
        "schema": completed.CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
        "status": completed.CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
        "completed_nrk": nrk,
        "next_nrk": nrk + 1 if nrk < 3 else None,
        "stream_pointer": 1234,
        "stage_identity_sha256": "1" * 64,
        "component_operation_token_sha256": {
            "momentum_rk": "2" * 64,
            "tke_rk": "3" * 64,
        },
        "source_order_complete": True,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "full_forecast": False,
        "claim_limit": completed.GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = completed._operation_token(record)
    return completed.Gm1TiledCompletedRkStageToken(MappingProxyType(record))


def test_exact_token_common_contract_is_admitted() -> None:
    token = _token()
    receipt = completed.require_gm1_tiled_completed_rk_stage_token(
        token,
        completed_nrk=1,
        stream_pointer=1234,
        stage_identity_sha256="1" * 64,
    )
    assert receipt["next_nrk"] == 2
    assert receipt["component_operation_token_sha256"]["tke_rk"] == "3" * 64


def test_foreign_or_duck_token_is_refused() -> None:
    with pytest.raises(TypeError, match="exact Gm1TiledCompletedRkStageToken"):
        completed.require_gm1_tiled_completed_rk_stage_token(
            {"receipt": _token().receipt}, completed_nrk=1
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_order_complete", False, "identity, cadence, or claim"),
        ("five_meter_meteorological_product", True, "identity, cadence, or claim"),
        ("global_psolver_tile_streamable", True, "identity, cadence, or claim"),
        ("stream_pointer", 0, "stream pointer"),
    ),
)
def test_changed_scope_or_cadence_is_refused(
    field: str, value: object, message: str
) -> None:
    record = dict(_token().receipt)
    record[field] = value
    record["operation_token_sha256"] = completed._operation_token(record)
    token = completed.Gm1TiledCompletedRkStageToken(MappingProxyType(record))
    with pytest.raises(ValueError, match=message):
        completed.require_gm1_tiled_completed_rk_stage_token(
            token, completed_nrk=1
        )


def test_tampered_operation_token_or_missing_tke_lineage_is_refused() -> None:
    record = dict(_token().receipt)
    record["operation_token_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity, cadence, or claim"):
        completed.require_gm1_tiled_completed_rk_stage_token(
            completed.Gm1TiledCompletedRkStageToken(MappingProxyType(record)),
            completed_nrk=1,
        )
    record = dict(_token().receipt)
    record["component_operation_token_sha256"] = {"momentum_rk": "2" * 64}
    record["operation_token_sha256"] = completed._operation_token(record)
    with pytest.raises(ValueError, match="live TKE RK token"):
        completed.require_gm1_tiled_completed_rk_stage_token(
            completed.Gm1TiledCompletedRkStageToken(MappingProxyType(record)),
            completed_nrk=1,
        )


def test_rk3_has_no_next_rk_stage() -> None:
    receipt = completed.require_gm1_tiled_completed_rk_stage_token(
        _token(nrk=3), completed_nrk=3
    )
    assert receipt["next_nrk"] is None


def test_module_deliberately_exposes_no_token_emitter() -> None:
    assert not hasattr(completed, "finish_cuda_gm1_tiled_completed_rk_stage")
    assert "emits no stage" in completed.GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT
