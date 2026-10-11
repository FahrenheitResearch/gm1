from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from authority_trees import requires_native_reference
import cm1core.cuda_gm1_tilestream_rk23_acoustic as acoustic
from cm1core.cuda_gm1_child_psolver3_open import (
    CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA,
    CudaGm1ChildPsolver3OpenLaunch,
)
from cm1core.cuda_gm1_tilestream_completed_rk_stage import (
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
    GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
    Gm1TiledCompletedRkStageToken,
)


ROOT = Path(__file__).resolve().parents[1]


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


class FakeVertical:
    pass


class FakeBridge:
    pass


class FakeTerrainMetrics:
    pass


def _prior(*, completed_nrk: int, stream_pointer: int = 73):
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
        "completed_nrk": completed_nrk,
        "next_nrk": completed_nrk + 1 if completed_nrk < 3 else None,
        "stream_pointer": stream_pointer,
        "stage_identity_sha256": f"{completed_nrk}" * 64,
        "component_operation_token_sha256": {
            "momentum_rk": "4" * 64,
            "tke_rk": "5" * 64,
        },
        "source_order_complete": True,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "full_forecast": False,
        "claim_limit": GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledCompletedRkStageToken(MappingProxyType(receipt))


def _pre_receipt(
    prior: Gm1TiledCompletedRkStageToken, *, requested_nrk: int
) -> dict[str, object]:
    source = acoustic.audit_gm1_rk23_live_acoustic_source_contract(requested_nrk)
    hashes = {
        name: f"{index:x}" * 64
        for index, name in enumerate(acoustic.GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS, 6)
    }
    receipt: dict[str, object] = {
        "schema": acoustic.CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_SCHEMA,
        "status": acoustic.CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_STATUS,
        "requested_nrk": requested_nrk,
        "required_completed_nrk": requested_nrk - 1,
        "prior_completed_rk_stage_token_sha256": prior.receipt[
            "operation_token_sha256"
        ],
        "solve1_full_held_token_sha256": "f" * 64,
        "stage_identity_sha256": f"{requested_nrk + 6:x}" * 64,
        "stream_pointer": prior.receipt["stream_pointer"],
        "source_contract_sha256": source["source_contract_sha256"],
        "current_velocity_boundary_replay": True,
        "fresh_current_stage_mass_flux": True,
        "historical_rk1_post_sound_token_reused": False,
        "output_sha256": hashes,
        "output_shapes": dict(acoustic.GM1_RK23_PRE_ACOUSTIC_OUTPUT_SHAPES),
        "completed_rk_stage": False,
        "claim_limit": acoustic.GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return receipt


def _pre_result(
    *,
    requested_nrk: int,
    arrays: MappingProxyType | None = None,
    metrics: MappingProxyType | None = None,
    stream: object | None = None,
):
    prior = _prior(completed_nrk=requested_nrk - 1)
    stream = stream or SimpleNamespace(ptr=prior.receipt["stream_pointer"])
    arrays = arrays or MappingProxyType({"live": object()})
    metrics = metrics or MappingProxyType({"sealed": object()})
    vertical = FakeVertical()
    bridge = FakeBridge()
    terrain = FakeTerrainMetrics()
    token_receipt = _pre_receipt(prior, requested_nrk=requested_nrk)
    token = acoustic.Gm1TiledRk23PreAcousticMomentumToken(
        MappingProxyType(token_receipt)
    )
    binding_receipt: dict[str, object] = {
        "schema": acoustic.CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_SCHEMA,
        "status": acoustic.CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_STATUS,
        "requested_nrk": requested_nrk,
        "prior_completed_rk_stage_token_sha256": prior.receipt[
            "operation_token_sha256"
        ],
        "stage_identity_sha256": token_receipt["stage_identity_sha256"],
        "stream_pointer": prior.receipt["stream_pointer"],
    }
    binding_receipt["operation_token_sha256"] = _operation_token(binding_receipt)
    binding = acoustic.Gm1TiledRk23PreAcousticMomentumBinding(
        prior,
        arrays,
        metrics,
        terrain,  # type: ignore[arg-type]
        vertical,  # type: ignore[arg-type]
        stream,
        bridge,  # type: ignore[arg-type]
        str(token_receipt["stage_identity_sha256"]),
        MappingProxyType(binding_receipt),
    )
    return acoustic.FinishedGm1TiledRk23PreAcousticMomentum(
        token,
        binding,
        prior,
        requested_nrk,
        arrays,
        metrics,
        terrain,  # type: ignore[arg-type]
        vertical,  # type: ignore[arg-type]
        stream,
        bridge,  # type: ignore[arg-type]
        MappingProxyType(dict(token_receipt["output_sha256"])),  # type: ignore[arg-type]
    )


def _upstream_receipt(requested_nrk: int) -> dict[str, object]:
    schedule = acoustic.audit_gm1_rk23_live_acoustic_source_contract(requested_nrk)[
        "schedule"
    ]
    return {
        "schema": CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA,
        "domain": {
            "ni": 300,
            "nj": 300,
            "nk": 75,
            "dx_m": 50.0,
            "dy_m": 50.0,
            "ztop_m": 15000.0,
            "compact_validation": False,
        },
        "cadence": [
            {
                "nrk": requested_nrk,
                "nloop": schedule["sound_substeps"],
                "dts_f32": schedule["sound_dt_seconds_f32"],
                "open_consumptions": schedule["sound_substeps"],
                "pressure_fills": int(schedule["sound_substeps"]) - 1,
                "radiation_productions": 1,
            }
        ],
        "kernels": {
            "core_source_sha256": CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
            "boundary_source_sha256": (
                CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256
            ),
        },
        "source_valid_padded_rho0": True,
        "scalar_dz_fallback": False,
        "uniform_idealized_validation": False,
    }


def _upstream(requested_nrk: int, event: object | None = None):
    return CudaGm1ChildPsolver3OpenLaunch(
        event or SimpleNamespace(synchronize=lambda: None),
        None,  # type: ignore[arg-type]
        None,
        None,
        (),
        (),
        _upstream_receipt(requested_nrk),
    )


@pytest.fixture(autouse=True)
def exact_resident_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acoustic, "Gm1ChildStretchedMetrics", FakeVertical)
    monkeypatch.setattr(acoustic, "CudaMoistStageBridgeState", FakeBridge)
    monkeypatch.setattr(
        acoustic, "CudaGm1ChildTerrainMetricsLaunch", FakeTerrainMetrics
    )


@requires_native_reference
def test_source_hashes_schedules_and_fail_closed_scope_are_exact() -> None:
    authority = ROOT / "work/native-reference/cm1r21.1/src"
    assert hashlib.sha256((authority / "solve2.F").read_bytes()).hexdigest() == (
        acoustic.AUTHORITY_SOLVE2_SHA256
    )
    assert hashlib.sha256((authority / "sound.F").read_bytes()).hexdigest() == (
        acoustic.AUTHORITY_SOUND_SHA256
    )
    assert hashlib.sha256((authority / "bc.F").read_bytes()).hexdigest() == (
        acoustic.AUTHORITY_BC_SHA256
    )
    expected = {
        2: {
            "rk_stage_dt_seconds_f32": 0.25,
            "sound_substeps": 3,
            "sound_dt_seconds_f32": 0.0833333358168602,
        },
        3: {
            "rk_stage_dt_seconds_f32": 0.5,
            "sound_substeps": 6,
            "sound_dt_seconds_f32": 0.0833333358168602,
        },
    }
    for nrk in (2, 3):
        audit = acoustic.audit_gm1_rk23_live_acoustic_source_contract(nrk)
        assert audit["schedule"] == expected[nrk]
        assert audit["required_completed_nrk"] == nrk - 1
        assert audit["historical_rk1_token_admissible"] is False
        assert audit["source_audit_carries_runtime_completed_stage"] is False
        assert audit["source_audit_carries_runtime_pre_acoustic_result"] is False
        assert audit["runtime_pre_acoustic_producer_module"] == (
            "cm1core.cuda_gm1_tilestream_rk23_pre_acoustic_momentum"
        )
        assert audit["continuation_token_emitted"] is False
        assert audit["completed_stage_token_emitted"] is False
    assert acoustic.LIVE_RK23_ACOUSTIC_ABLATIONS.stage_family_fills is False


def test_fresh_pre_token_is_exact_ordered_self_hashed_and_prior_bound() -> None:
    prior = _prior(completed_nrk=1)
    receipt = _pre_receipt(prior, requested_nrk=2)
    token = acoustic.Gm1TiledRk23PreAcousticMomentumToken(MappingProxyType(receipt))
    assert (
        acoustic._require_pre_acoustic_token(
            token,
            requested_nrk=2,
            prior_completed_stage_token=prior,
        )
        == receipt
    )

    reordered = dict(receipt)
    reordered["output_sha256"] = dict(
        reversed(list(receipt["output_sha256"].items()))  # type: ignore[union-attr]
    )
    reordered["operation_token_sha256"] = _operation_token(reordered)
    with pytest.raises(ValueError, match="identity or chronology"):
        acoustic._require_pre_acoustic_token(
            acoustic.Gm1TiledRk23PreAcousticMomentumToken(MappingProxyType(reordered)),
            requested_nrk=2,
            prior_completed_stage_token=prior,
        )
    with pytest.raises(TypeError, match="exact RK23 pre-acoustic"):
        acoustic._require_pre_acoustic_token(
            SimpleNamespace(receipt=receipt),
            requested_nrk=2,
            prior_completed_stage_token=prior,
        )


@pytest.mark.parametrize(("requested_nrk", "nloop"), ((2, 3), (3, 6)))
def test_launch_uses_exact_prior_same_stream_and_rk_cadence(
    requested_nrk: int,
    nloop: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_launch(arrays: object, metrics: object, **kwargs: object):
        captured.update(kwargs)
        captured["arrays"] = arrays
        captured["metrics"] = metrics
        return _upstream(requested_nrk)

    monkeypatch.setattr(
        acoustic, "launch_cuda_gm1_child_psolver3_open_stage", fake_launch
    )
    pre = _pre_result(requested_nrk=requested_nrk)
    launch = acoustic.launch_cuda_gm1_tilestream_rk23_acoustic(pre)
    assert captured["arrays"] is pre.arrays
    assert captured["metrics"] is pre.metrics
    assert captured["nrk"] == requested_nrk
    assert captured["dt"] == 0.5
    assert captured["nsound"] == 6
    assert captured["dx"] == captured["dy"] == 50.0
    assert captured["stream"] is pre.stream
    assert captured["ablations"] == acoustic.LIVE_RK23_ACOUSTIC_ABLATIONS
    assert captured["allow_compact_validation"] is False
    assert captured["allow_uniform_idealized_validation"] is False
    assert launch.prior_completed_stage_token is pre.prior_completed_stage_token
    assert launch.receipt["schedule"]["sound_substeps"] == nloop
    assert launch.receipt["completed_rk_stage"] is False


def test_launch_refuses_wrong_prior_stream_historical_or_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="exact FinishedGm1"):
        acoustic.launch_cuda_gm1_tilestream_rk23_acoustic(  # type: ignore[arg-type]
            SimpleNamespace()
        )
    changed_stream = _pre_result(requested_nrk=2, stream=SimpleNamespace(ptr=74))
    with pytest.raises(ValueError, match="stream differs"):
        acoustic._require_pre_acoustic_result(changed_stream)

    pre = _pre_result(requested_nrk=2)
    receipt = dict(pre.token.receipt)
    receipt["historical_rk1_post_sound_token_reused"] = True
    receipt["operation_token_sha256"] = _operation_token(receipt)
    changed = acoustic.FinishedGm1TiledRk23PreAcousticMomentum(
        acoustic.Gm1TiledRk23PreAcousticMomentumToken(MappingProxyType(receipt)),
        pre.binding,
        pre.prior_completed_stage_token,
        pre.requested_nrk,
        pre.arrays,
        pre.metrics,
        pre.terrain_metrics,
        pre.vertical,
        pre.stream,
        pre.bridge,
        pre.output_sha256,
    )
    with pytest.raises(ValueError, match="identity or chronology"):
        acoustic._require_pre_acoustic_result(changed)

    upstream = _upstream(2)
    upstream.receipt["cadence"][0]["nloop"] = 6
    with pytest.raises(ValueError, match="source, domain, or cadence"):
        acoustic._require_upstream_receipt(upstream, requested_nrk=2)


@pytest.mark.parametrize("requested_nrk", (2, 3))
def test_finish_hashes_exact_outputs_and_hands_off_live_carriers(
    requested_nrk: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    event = SimpleNamespace(sync_count=0)

    def synchronize() -> None:
        event.sync_count += 1

    event.synchronize = synchronize
    upstream = _upstream(requested_nrk, event)
    monkeypatch.setattr(
        acoustic,
        "launch_cuda_gm1_child_psolver3_open_stage",
        lambda *_args, **_kwargs: upstream,
    )
    tiny_shapes = MappingProxyType(
        {name: [1] for name in acoustic.GM1_RK23_ACOUSTIC_OUTPUT_KEYS}
    )
    monkeypatch.setattr(acoustic, "GM1_RK23_ACOUSTIC_OUTPUT_SHAPES", tiny_shapes)

    class NamedValue:
        def __init__(self, name: str) -> None:
            self.name = name

    arrays = MappingProxyType(
        {name: NamedValue(name) for name in acoustic.GM1_RK23_ACOUSTIC_OUTPUT_KEYS}
    )
    monkeypatch.setattr(
        acoustic,
        "_host_output",
        lambda _cupy, value, _slices: np.array(
            [float(acoustic.GM1_RK23_ACOUSTIC_OUTPUT_KEYS.index(value.name))],
            dtype=np.float32,
        ),
    )
    real_import = acoustic.importlib.import_module
    monkeypatch.setattr(
        acoustic.importlib,
        "import_module",
        lambda name: SimpleNamespace() if name == "cupy" else real_import(name),
    )
    pre = _pre_result(requested_nrk=requested_nrk, arrays=arrays)
    launch = acoustic.launch_cuda_gm1_tilestream_rk23_acoustic(pre)
    observed: list[acoustic.Gm1TiledRk23AcousticLiveHandoff] = []

    def continue_live(handoff: acoustic.Gm1TiledRk23AcousticLiveHandoff):
        assert event.sync_count == 1
        assert handoff.stage is upstream
        assert handoff.prior_completed_stage_token is pre.prior_completed_stage_token
        assert handoff.arrays is pre.arrays
        assert handoff.metrics is pre.metrics
        assert handoff.vertical is pre.vertical
        assert handoff.stream is pre.stream
        assert handoff.bridge is pre.bridge
        assert handoff.terrain_metrics is pre.terrain_metrics
        assert handoff.pre_acoustic_result is pre
        assert handoff.pre_acoustic_token is pre.token
        observed.append(handoff)
        return "same-residency-continuation"

    result = acoustic.finish_cuda_gm1_tilestream_rk23_acoustic(
        launch, continuation=continue_live
    )
    assert observed == [result.handoff]
    assert result.continuation_result == "same-residency-continuation"
    velocity = result.handoff.post_sound_velocity_token.receipt
    continuation = result.handoff.acoustic_token.receipt
    assert velocity["requested_nrk"] == requested_nrk
    assert continuation["requested_nrk"] == requested_nrk
    assert (
        continuation["post_sound_velocity_token_sha256"]
        == velocity["operation_token_sha256"]
    )
    assert tuple(continuation["output_sha256"]) == (
        acoustic.GM1_RK23_ACOUSTIC_OUTPUT_KEYS
    )
    assert result.execution_receipt["host_sync_performed"] is True
    assert result.execution_receipt["uninterrupted_pre_sync_enqueue_claimed"] is False
    assert result.execution_receipt["completed_rk_stage"] is False

    lineage = acoustic.require_gm1_tiled_rk23_acoustic_continuation(
        pre.prior_completed_stage_token,
        pre.token,
        result.handoff.post_sound_velocity_token,
        result.handoff.acoustic_token,
    )
    assert lineage["requested_nrk"] == requested_nrk
    assert lineage["completed_stage_token_emitted"] is False

    tampered = dict(continuation)
    tampered["schedule"] = dict(tampered["schedule"])
    tampered["schedule"]["sound_substeps"] = 2
    tampered["operation_token_sha256"] = _operation_token(tampered)
    with pytest.raises(ValueError, match="lineage or cadence"):
        acoustic.require_gm1_tiled_rk23_acoustic_continuation(
            pre.prior_completed_stage_token,
            pre.token,
            result.handoff.post_sound_velocity_token,
            acoustic.Gm1TiledRk23AcousticContinuationToken(MappingProxyType(tampered)),
        )


@pytest.mark.parametrize("requested_nrk", (2, 3))
def test_runtime_refuses_without_fresh_pre_and_emits_no_completed_stage(
    requested_nrk: int,
) -> None:
    prior = _prior(completed_nrk=requested_nrk - 1)
    with pytest.raises(RuntimeError, match="exact fresh same-stage"):
        acoustic.refuse_gm1_rk23_acoustic_without_fresh_pre_result(
            prior, requested_nrk=requested_nrk
        )
    with pytest.raises(ValueError, match="completed-stage identity"):
        acoustic.refuse_gm1_rk23_acoustic_without_fresh_pre_result(
            _prior(completed_nrk=1), requested_nrk=3
        )
    assert not hasattr(acoustic, "prepare_cuda_gm1_tilestream_rk23_pre_acoustic")
    assert not hasattr(acoustic, "finish_cuda_gm1_tiled_completed_rk_stage")
    assert "completed current stage" in acoustic.GM1_RK23_ACOUSTIC_CLAIM_LIMIT
