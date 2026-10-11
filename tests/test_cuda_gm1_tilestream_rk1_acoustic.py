from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from authority_trees import requires_native_reference
import cm1core.cuda_gm1_tilestream_rk1_acoustic as acoustic
from cm1core.cuda_gm1_child_psolver3_open import (
    CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA,
    CudaGm1ChildPsolver3OpenLaunch,
)
from cm1core.cuda_gm1_tilestream_rk1_pre_acoustic_lineage import (
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
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


@dataclass(frozen=True)
class FakePreToken:
    receipt: MappingProxyType


@dataclass(frozen=True)
class FakeAcousticToken:
    receipt: MappingProxyType


@dataclass(frozen=True)
class FakePreLaunch:
    receipt: MappingProxyType


class FakeVertical:
    pass


class FakeBridge:
    pass


class FakeTerrainMetrics:
    pass


class FakeMomentumLaunch:
    pass


@dataclass(frozen=True)
class FakeFullHeldToken:
    receipt: MappingProxyType


@dataclass(frozen=True)
class FakeSolve1TkeToken:
    receipt: MappingProxyType


@dataclass(frozen=True)
class FakeBinding:
    full_held_token: FakeFullHeldToken
    solve1_tke_token: FakeSolve1TkeToken
    arrays: MappingProxyType
    metrics: MappingProxyType
    terrain_metrics: FakeTerrainMetrics
    vertical: FakeVertical
    stream: object
    bridge: FakeBridge
    receipt: MappingProxyType


@dataclass(frozen=True)
class FakeFinishedPre:
    token: FakePreToken
    binding: FakeBinding
    full_held_token: FakeFullHeldToken
    solve1_tke_token: FakeSolve1TkeToken
    arrays: MappingProxyType
    metrics: MappingProxyType
    terrain_metrics: FakeTerrainMetrics
    vertical: FakeVertical
    stream: object
    bridge: FakeBridge
    output_sha256: MappingProxyType
    upstream: FakeMomentumLaunch


def _pre_receipt(*, stream_pointer: int = 73) -> dict[str, object]:
    shapes = {
        "uten": [75, 300, 301],
        "vten": [75, 301, 300],
        "wten": [76, 300, 300],
        "rru": [75, 300, 301],
        "rrv": [75, 301, 300],
        "rrw": [76, 300, 300],
        "divx": [75, 300, 300],
    }
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
        "nrk": 1,
        "stream_pointer": stream_pointer,
        "stage_identity_sha256": "1" * 64,
        "solve1_full_held_token_sha256": "2" * 64,
        "output_sha256": {
            name: f"{index:x}" * 64 for index, name in enumerate(shapes, 3)
        },
        "output_shapes": shapes,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return receipt


def _upstream_receipt() -> dict[str, object]:
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
                "nrk": 1,
                "nloop": 2,
                "dts_f32": 0.0833333358168602,
                "open_consumptions": 2,
                "pressure_fills": 1,
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


def _upstream(event: object | None = None) -> CudaGm1ChildPsolver3OpenLaunch:
    return CudaGm1ChildPsolver3OpenLaunch(
        event or SimpleNamespace(synchronize=lambda: None),
        None,  # type: ignore[arg-type]
        None,
        None,
        (),
        (),
        _upstream_receipt(),
    )


def _pre_result(
    *,
    arrays: MappingProxyType | None = None,
    metrics: MappingProxyType | None = None,
    vertical: FakeVertical | None = None,
    stream: object | None = None,
) -> FakeFinishedPre:
    receipt = _pre_receipt()
    arrays = arrays or MappingProxyType({"live": object()})
    metrics = metrics or MappingProxyType({"sealed": object()})
    vertical = vertical or FakeVertical()
    stream = stream or SimpleNamespace(ptr=73)
    terrain_metrics = FakeTerrainMetrics()
    bridge = FakeBridge()
    solve1_tke_receipt: dict[str, object] = {
        "schema": acoustic.CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS",
        "completed_nrk": 0,
        "stream_pointer": int(stream.ptr),
        "output_tketen_shape": [76, 300, 300],
        "output_tketen_sha256": "c" * 64,
    }
    solve1_tke_receipt["operation_token_sha256"] = _operation_token(solve1_tke_receipt)
    held_receipt: dict[str, object] = {
        "schema": acoustic.CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
        "status": acoustic.CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
        "completed_nrk": 0,
        "stream_pointer": int(stream.ptr),
        "solve1_tke_operation_token_sha256": solve1_tke_receipt[
            "operation_token_sha256"
        ],
    }
    held_receipt["operation_token_sha256"] = _operation_token(held_receipt)
    receipt["solve1_full_held_token_sha256"] = held_receipt["operation_token_sha256"]
    receipt["operation_token_sha256"] = _operation_token(receipt)
    token = FakePreToken(MappingProxyType(receipt))
    full_held_token = FakeFullHeldToken(MappingProxyType(held_receipt))
    solve1_tke_token = FakeSolve1TkeToken(MappingProxyType(solve1_tke_receipt))
    binding_receipt = MappingProxyType(
        {"binding_sha256": receipt.get("binding_sha256")}
    )
    binding = FakeBinding(
        full_held_token,
        solve1_tke_token,
        arrays,
        metrics,
        terrain_metrics,
        vertical,
        stream,
        bridge,
        binding_receipt,
    )
    return FakeFinishedPre(
        token,
        binding,
        full_held_token,
        solve1_tke_token,
        arrays,
        metrics,
        terrain_metrics,
        vertical,
        stream,
        bridge,
        MappingProxyType(dict(receipt["output_sha256"])),
        FakeMomentumLaunch(),
    )


@pytest.fixture
def exact_token_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        acoustic,
        "_runtime_types",
        lambda: (
            FakePreToken,
            FakeAcousticToken,
            FakeBinding,
            FakePreLaunch,
            FakeFinishedPre,
        ),
    )
    monkeypatch.setattr(acoustic, "Gm1TiledSolve1FullHeldToken", FakeFullHeldToken)
    monkeypatch.setattr(acoustic, "Gm1TiledSolve1TkeRhsToken", FakeSolve1TkeToken)
    monkeypatch.setattr(acoustic, "Gm1ChildStretchedMetrics", FakeVertical)
    monkeypatch.setattr(acoustic, "CudaMoistStageBridgeState", FakeBridge)
    monkeypatch.setattr(
        acoustic, "CudaGm1ChildTerrainMetricsLaunch", FakeTerrainMetrics
    )
    monkeypatch.setattr(acoustic, "CudaMomentumAdvection5Launch", FakeMomentumLaunch)


def test_source_contract_pins_official_order_and_defers_family_fills() -> None:
    audit = acoustic.audit_gm1_rk1_live_acoustic_source_contract()
    assert audit["status"] == "SOURCE_AUDITED_RUNTIME_TOKEN_REQUIRED"
    assert audit["selected_profile"] == {
        "domain_zyx": [75, 300, 300],
        "dx_m": 50.0,
        "dy_m": 50.0,
        "dt_seconds": 0.5,
        "nrk": 1,
        "nrkmax": 3,
        "psolver": 3,
        "nsound": 6,
        "nloop": 2,
        "dts_f32": 0.0833333358168602,
        "terrain": True,
        "stretch_z": 1,
        "lateral_boundaries": [2, 2, 2, 2],
        "mpi": False,
    }
    switches = audit["operator_switches"]
    assert switches["radiation_history_production"] is True
    assert switches["stage_family_fills"] is False
    assert audit["continuation_boundary"] == (
        "after radiation history; before getdiv/getdivx"
    )
    assert audit["completed_stage_token_emitted"] is False
    assert acoustic.LIVE_ACOUSTIC_ABLATIONS.stage_family_fills is False
    assert acoustic.LIVE_ACOUSTIC_ABLATIONS.radiation_history_production is True


@requires_native_reference
def test_source_contract_hashes_match_official_cm1_r21_1_bytes() -> None:
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


def test_pre_token_is_exact_typed_ordered_and_self_hashed(
    exact_token_types: None,
) -> None:
    receipt = _pre_receipt()
    assert (
        acoustic._require_pre_acoustic_token(FakePreToken(MappingProxyType(receipt)))
        == receipt
    )

    reordered = dict(receipt)
    reordered["output_sha256"] = dict(reversed(list(receipt["output_sha256"].items())))
    reordered["operation_token_sha256"] = _operation_token(reordered)
    with pytest.raises(ValueError, match="output inventory changed"):
        acoustic._require_pre_acoustic_token(FakePreToken(MappingProxyType(reordered)))

    with pytest.raises(TypeError, match="exact Gm1Tiled"):
        acoustic._require_pre_acoustic_token(receipt)


def test_runtime_gate_is_explicitly_unavailable_without_reserved_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = acoustic.importlib.import_module

    def refuse_reserved(name: str):
        if name == acoustic._TOKEN_MODULE:
            raise ModuleNotFoundError("reserved module absent", name=name)
        return real_import(name)

    monkeypatch.setattr(acoustic.importlib, "import_module", refuse_reserved)
    with pytest.raises(RuntimeError, match="exact pre-acoustic token module"):
        acoustic._runtime_types()


def test_launch_uses_same_stream_native_cadence_and_no_late_fills(
    exact_token_types: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_launch(
        arrays: object, metrics: object, **kwargs: object
    ) -> CudaGm1ChildPsolver3OpenLaunch:
        captured.update(kwargs)
        captured["arrays"] = arrays
        captured["metrics"] = metrics
        return _upstream()

    monkeypatch.setattr(
        acoustic, "launch_cuda_gm1_child_psolver3_open_stage", fake_launch
    )
    stream = SimpleNamespace(ptr=73)
    arrays = MappingProxyType({"live": object()})
    metrics = MappingProxyType({"sealed": object()})
    pre = _pre_result(
        arrays=arrays, metrics=metrics, vertical=FakeVertical(), stream=stream
    )
    launch = acoustic.launch_cuda_gm1_tilestream_rk1_acoustic(
        pre,
    )
    assert captured["arrays"] is arrays
    assert captured["metrics"] is metrics
    assert captured["ni"] == 300
    assert captured["nj"] == 300
    assert captured["nk"] == 75
    assert captured["nrk"] == 1
    assert captured["dt"] == 0.5
    assert captured["nsound"] == 6
    assert captured["dx"] == 50.0
    assert captured["dy"] == 50.0
    assert captured["stream"] is stream
    assert captured["ablations"] == acoustic.LIVE_ACOUSTIC_ABLATIONS
    assert captured["ablations"].stage_family_fills is False
    assert launch.receipt["stage_family_fills_deferred"] is True
    assert launch.receipt["getdiv_getdivx_performed"] is False


def test_launch_refuses_stream_or_operator_cadence_mutation(
    exact_token_types: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_only = FakePreToken(MappingProxyType(_pre_receipt()))
    with pytest.raises(TypeError, match="exact FinishedGm1Tiled"):
        acoustic.launch_cuda_gm1_tilestream_rk1_acoustic(token_only)

    pre = _pre_result(stream=SimpleNamespace(ptr=74))
    with pytest.raises(ValueError, match="full-held lineage changed"):
        acoustic._require_pre_acoustic_result(pre)

    pre = _pre_result()
    changed_hashes = dict(pre.output_sha256)
    changed_hashes["divx"] = "a" * 64
    changed = FakeFinishedPre(
        pre.token,
        pre.binding,
        pre.full_held_token,
        pre.solve1_tke_token,
        pre.arrays,
        pre.metrics,
        pre.terrain_metrics,
        pre.vertical,
        SimpleNamespace(ptr=73),
        pre.bridge,
        MappingProxyType(changed_hashes),
        pre.upstream,
    )
    with pytest.raises(ValueError, match="output hashes differ"):
        acoustic._require_pre_acoustic_result(changed)

    changed = _upstream_receipt()
    changed["cadence"][0]["nloop"] = 3
    upstream = _upstream()
    upstream.receipt.clear()
    upstream.receipt.update(changed)
    with pytest.raises(ValueError, match="source, domain, or RK1 cadence"):
        acoustic._require_upstream_receipt(upstream)


def test_upstream_receipt_admits_only_explicit_uniform_idealized_disposition() -> None:
    receipt = _upstream_receipt()
    receipt["domain"]["compact_validation"] = True
    receipt["uniform_idealized_validation"] = True
    upstream = _upstream()
    upstream.receipt.clear()
    upstream.receipt.update(receipt)

    acoustic._require_upstream_receipt(
        upstream, allow_uniform_idealized_validation=True
    )
    with pytest.raises(ValueError, match="source, domain, or RK1 cadence"):
        acoustic._require_upstream_receipt(upstream)


def test_finish_hashes_then_invokes_same_residency_handoff(
    exact_token_types: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    event = SimpleNamespace(sync_count=0)

    def synchronize() -> None:
        event.sync_count += 1

    event.synchronize = synchronize
    upstream = _upstream(event)
    monkeypatch.setattr(
        acoustic,
        "launch_cuda_gm1_child_psolver3_open_stage",
        lambda *_args, **_kwargs: upstream,
    )
    tiny_shapes = MappingProxyType(
        {name: [1] for name in acoustic.GM1_RK1_LIVE_ACOUSTIC_OUTPUT_KEYS}
    )
    monkeypatch.setattr(acoustic, "GM1_RK1_LIVE_ACOUSTIC_OUTPUT_SHAPES", tiny_shapes)

    class NamedValue:
        def __init__(self, name: str) -> None:
            self.name = name

    arrays = MappingProxyType(
        {name: NamedValue(name) for name in acoustic.GM1_RK1_LIVE_ACOUSTIC_OUTPUT_KEYS}
    )
    monkeypatch.setattr(
        acoustic,
        "_host_output",
        lambda _cupy, value, _slices: np.array(
            [float(acoustic.GM1_RK1_LIVE_ACOUSTIC_OUTPUT_KEYS.index(value.name))],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        acoustic,
        "require_gm1_rk1_pre_acoustic_acoustic_lineage",
        lambda pre_token, acoustic_token: MappingProxyType(
            {
                "schema": "gm1.rk1-pre-acoustic-acoustic-lineage-audit/v1",
                "status": "VERIFIED_LIVE_PRE_ACOUSTIC_ACOUSTIC_LINEAGE_ONLY",
                "pre": pre_token.receipt["operation_token_sha256"],
                "acoustic": acoustic_token.receipt["operation_token_sha256"],
            }
        ),
    )
    real_import = acoustic.importlib.import_module
    monkeypatch.setattr(
        acoustic.importlib,
        "import_module",
        lambda name: SimpleNamespace() if name == "cupy" else real_import(name),
    )
    stream = SimpleNamespace(ptr=73)
    metrics = MappingProxyType({"sealed": object()})
    vertical = FakeVertical()
    pre = _pre_result(arrays=arrays, metrics=metrics, vertical=vertical, stream=stream)
    launch = acoustic.launch_cuda_gm1_tilestream_rk1_acoustic(
        pre,
    )
    observed: list[acoustic.Gm1TiledRk1AcousticLiveHandoff] = []

    def continue_live(
        handoff: acoustic.Gm1TiledRk1AcousticLiveHandoff,
    ) -> str:
        assert event.sync_count == 1
        assert handoff.ready_event is event
        assert handoff.stage is upstream
        assert handoff.arrays is arrays
        assert handoff.metrics is metrics
        assert handoff.vertical is vertical
        assert handoff.stream is stream
        assert handoff.terrain_metrics is pre.terrain_metrics
        assert handoff.full_held_token is pre.full_held_token
        assert handoff.solve1_tke_token is pre.solve1_tke_token
        assert handoff.pre_acoustic_result is pre
        assert handoff.pre_acoustic_token is pre.token
        observed.append(handoff)
        return "live-chain-enqueued"

    result = acoustic.finish_cuda_gm1_tilestream_rk1_acoustic(
        launch, continuation=continue_live
    )
    assert observed == [result.handoff]
    assert result.continuation_result == "live-chain-enqueued"
    assert tuple(result.host_velocity) == ("u3d", "v3d", "w3d")
    assert tuple(
        result.handoff.post_sound_velocity_token.receipt["velocity_sha256"]
    ) == (
        "u3d",
        "v3d",
        "w3d",
    )
    token = result.handoff.acoustic_token.receipt
    assert tuple(token) == (
        "schema",
        "status",
        "nrk",
        "stage_identity_sha256",
        "pre_acoustic_momentum_rk_token_sha256",
        "post_sound_velocity_token_sha256",
        "stream_pointer",
        "operation_token_sha256",
    )
    execution = result.execution_receipt
    assert execution["host_sync_performed"] is True
    assert execution["uninterrupted_pre_sync_enqueue_claimed"] is False
    assert execution["completed_rk_stage"] is False
    assert execution["pre_acoustic_acoustic_lineage_audit"]["status"] == (
        "VERIFIED_LIVE_PRE_ACOUSTIC_ACOUSTIC_LINEAGE_ONLY"
    )


def test_contract_never_claims_completed_stage_or_five_metre_product() -> None:
    claim = acoustic.GM1_RK1_LIVE_ACOUSTIC_CLAIM_LIMIT
    assert "completed RK stage" in claim
    assert "five-metre scaling" in claim
    assert (
        "before getdiv/getdivx"
        in (
            acoustic.audit_gm1_rk1_live_acoustic_source_contract()[
                "continuation_boundary"
            ]
        )
    )
