from __future__ import annotations

import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import pytest

import cm1core.cuda_gm1_tilestream_rk23_acoustic as acoustic
import cm1core.cuda_gm1_tilestream_rk23_pre_acoustic_momentum as pre
from cm1core.cuda_gm1_child_stretched_momentum import (
    GM1_CHILD_STRETCHED_MOMENTUM_SCHEMA,
    CudaMomentumAdvection5Launch,
)
from cm1core.cuda_gm1_tilestream_completed_rk_stage import (
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
    GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
    Gm1TiledCompletedRkStageToken,
)
from cm1core.cuda_gm1_tilestream_pre_acoustic_momentum_rk import (
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
    Gm1TiledPreAcousticMomentumRkToken,
)
from cm1core.cuda_gm1_tilestream_solve1_full_held import (
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
    Gm1TiledSolve1FullHeldToken,
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


def _full_held(*, stream_pointer: int = 73):
    shapes = {
        "uten": [75, 300, 301],
        "vten": [75, 301, 300],
        "wten": [76, 300, 300],
        "thten1": [75, 300, 300],
        "qten": [6, 75, 300, 300],
    }
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
        "completed_nrk": 0,
        "stream_pointer": stream_pointer,
        "solve1_tke_operation_token_sha256": "1" * 64,
        "output_shapes": shapes,
        "output_sha256": {
            name: f"{index:x}" * 64 for index, name in enumerate(shapes, 2)
        },
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledSolve1FullHeldToken(MappingProxyType(receipt))


def _rk1_pre(full_held: Gm1TiledSolve1FullHeldToken):
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
        "nrk": 1,
        "stream_pointer": full_held.receipt["stream_pointer"],
        "stage_identity_sha256": "a" * 64,
        "solve1_full_held_token_sha256": full_held.receipt["operation_token_sha256"],
        "completed_rk_stage": False,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledPreAcousticMomentumRkToken(MappingProxyType(receipt))


def _rk23_pre(
    full_held: Gm1TiledSolve1FullHeldToken,
    prior_completed: Gm1TiledCompletedRkStageToken,
    *,
    nrk: int,
):
    source = acoustic.audit_gm1_rk23_live_acoustic_source_contract(nrk)
    receipt: dict[str, object] = {
        "schema": acoustic.CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_SCHEMA,
        "status": acoustic.CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_STATUS,
        "requested_nrk": nrk,
        "required_completed_nrk": nrk - 1,
        "prior_completed_rk_stage_token_sha256": prior_completed.receipt[
            "operation_token_sha256"
        ],
        "solve1_full_held_token_sha256": full_held.receipt["operation_token_sha256"],
        "stage_identity_sha256": "b" * 64,
        "stream_pointer": full_held.receipt["stream_pointer"],
        "source_contract_sha256": source["source_contract_sha256"],
        "current_velocity_boundary_replay": True,
        "fresh_current_stage_mass_flux": True,
        "historical_rk1_post_sound_token_reused": False,
        "output_sha256": {
            name: f"{index:x}" * 64
            for index, name in enumerate(acoustic.GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS, 7)
        },
        "output_shapes": dict(acoustic.GM1_RK23_PRE_ACOUSTIC_OUTPUT_SHAPES),
        "completed_rk_stage": False,
        "claim_limit": acoustic.GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return acoustic.Gm1TiledRk23PreAcousticMomentumToken(MappingProxyType(receipt))


def _completed(
    *, nrk: int, pre_operation: str, stream_pointer: int = 73
) -> Gm1TiledCompletedRkStageToken:
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
        "completed_nrk": nrk,
        "next_nrk": nrk + 1,
        "stream_pointer": stream_pointer,
        "stage_identity_sha256": f"{nrk}" * 64,
        "component_operation_token_sha256": {
            "pre_acoustic_momentum_rk": pre_operation,
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


def test_prior_component_requires_exact_rk1_then_exact_rk2_types() -> None:
    held = _full_held()
    rk1 = _rk1_pre(held)
    completed1 = _completed(nrk=1, pre_operation=rk1.receipt["operation_token_sha256"])
    assert (
        pre._require_prior_pre_component(
            rk1,
            requested_nrk=2,
            completed=completed1.receipt,
            full_held=held.receipt,
        )["nrk"]
        == 1
    )
    with pytest.raises(TypeError, match="RK2 requires exact RK1"):
        pre._require_prior_pre_component(
            SimpleNamespace(receipt=rk1.receipt),
            requested_nrk=2,
            completed=completed1.receipt,
            full_held=held.receipt,
        )

    rk2 = _rk23_pre(held, completed1, nrk=2)
    completed2 = _completed(nrk=2, pre_operation=rk2.receipt["operation_token_sha256"])
    assert (
        pre._require_prior_pre_component(
            rk2,
            requested_nrk=3,
            completed=completed2.receipt,
            full_held=held.receipt,
        )["requested_nrk"]
        == 2
    )
    with pytest.raises(TypeError, match="RK3 requires exact RK2"):
        pre._require_prior_pre_component(
            rk1,
            requested_nrk=3,
            completed=completed2.receipt,
            full_held=held.receipt,
        )


class FakeArray:
    def __init__(self, shape: tuple[int, ...], ptr: int) -> None:
        self.shape = shape
        self.dtype = "float32"
        self.flags = SimpleNamespace(c_contiguous=True)
        self.data = SimpleNamespace(ptr=ptr)
        self.size = 1

    def __getitem__(self, _key: object):
        return self


class FakeEvent:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def record(self, _stream: object) -> None:
        self.log.append("event-record")

    def synchronize(self) -> None:
        self.log.append("event-sync")


class FakeStream:
    ptr = 73

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeVertical:
    target_arch = "86"
    identity = MappingProxyType({"grid_sha256": "7" * 64})


class FakeTerrain:
    receipt = MappingProxyType({"schema": "terrain"})


class FakeBridge:
    def __init__(self, stream: FakeStream, log: list[str]) -> None:
        self.ni, self.nj, self.nk, self.base3d = 300, 300, 75, True
        self.stream = stream
        self.terms_ready = True
        self.buoyancy_applied = True
        self.post_acoustic_ready = True
        self.divx = FakeArray((75, 300, 300), 900)
        self.compiled = SimpleNamespace(target_arch="86")
        self.open_compiled = SimpleNamespace(target_arch="86", kernels=())
        self.log = log


def _fake_live(monkeypatch: pytest.MonkeyPatch, *, requested_nrk: int):
    import numpy as np

    log: list[str] = []
    stream = FakeStream()
    held = _full_held()
    prior_pre = _rk1_pre(held)
    if requested_nrk == 3:
        completed1 = _completed(
            nrk=1, pre_operation=prior_pre.receipt["operation_token_sha256"]
        )
        prior_pre = _rk23_pre(held, completed1, nrk=2)
    completed = _completed(
        nrk=requested_nrk - 1,
        pre_operation=prior_pre.receipt["operation_token_sha256"],
    )
    arrays = {
        name: FakeArray((1,), index + 10)
        for index, name in enumerate(
            (
                "u3d",
                "v3d",
                "w3d",
                "pp3d",
                "uten",
                "vten",
                "wten",
                "rru",
                "rrv",
                "rrw",
            )
        )
    }
    metrics = {
        name: FakeArray((1,), index + 100)
        for index, name in enumerate(("gz", "dzdx", "dzdy", "rho0_padded"))
    }
    bridge = FakeBridge(stream, log)
    vertical = FakeVertical()
    terrain = FakeTerrain()
    base = {
        "th0": FakeArray((75, 300, 300), 201),
        "qv0": FakeArray((75, 300, 300), 202),
        "pi0": FakeArray((75, 300, 300), 203),
        "th3d": FakeArray((75, 300, 300), 204),
        "q3d": FakeArray((6, 75, 300, 300), 205),
    }
    held_outputs = {name: np.zeros((1,), dtype=np.float32) for name in pre._HELD_SHAPES}

    class FakeCupy:
        ndarray = FakeArray
        float32 = "float32"

        @staticmethod
        def dtype(value: str) -> str:
            return value

        @staticmethod
        def uint32(value: int) -> int:
            return value

        @staticmethod
        def asarray(value: object):
            return value

        @staticmethod
        def copyto(_target: object, _source: object) -> None:
            log.append("held-seed")

        cuda = SimpleNamespace(Event=lambda disable_timing=True: FakeEvent(log))

    real_import = pre.importlib.import_module
    monkeypatch.setattr(
        pre.importlib,
        "import_module",
        lambda name: FakeCupy if name == "cupy" else real_import(name),
    )
    monkeypatch.setattr(pre, "CudaGm1ChildTerrainMetricsLaunch", FakeTerrain)
    monkeypatch.setattr(pre, "Gm1ChildStretchedMetrics", FakeVertical)
    monkeypatch.setattr(pre, "CudaMoistStageBridgeState", FakeBridge)
    monkeypatch.setattr(pre, "_require_device_arrays", lambda *_a, **_k: arrays)
    monkeypatch.setattr(pre, "_require_device_metrics", lambda *_a, **_k: metrics)
    monkeypatch.setattr(pre, "_HELD_SHAPES", {name: (1,) for name in held_outputs})
    monkeypatch.setattr(
        pre,
        "_OUTPUT_SHAPES",
        {name: (1,) for name in pre.GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS},
    )
    monkeypatch.setattr(pre, "_host_f32", lambda _np, value, **_kwargs: value)
    held_sha_by_id = {
        id(held_outputs[name]): held.receipt["output_sha256"][name]
        for name in held_outputs
    }
    monkeypatch.setattr(
        pre,
        "_raw_sha_host",
        lambda value: held_sha_by_id.get(
            id(value), hashlib.sha256(value.tobytes()).hexdigest()
        ),
    )
    monkeypatch.setattr(
        pre,
        "_require_full_held_token",
        lambda token, **_kwargs: token.receipt,
    )
    monkeypatch.setattr(
        pre,
        "_pointer_sha256",
        lambda values: hashlib.sha256(
            _canonical({name: value.data.ptr for name, value in values.items()})
        ).hexdigest(),
    )

    def begin(value: FakeBridge) -> None:
        log.append("begin-stage")
        value.terms_ready = value.buoyancy_applied = value.post_acoustic_ready = False

    def terms(value: FakeBridge, *_args: object, **_kwargs: object):
        log.append("eqtset2")
        value.terms_ready = True
        return FakeEvent(log)

    def buoyancy(value: FakeBridge, *_args: object, **_kwargs: object):
        log.append("buoyancy")
        value.buoyancy_applied = True
        return FakeEvent(log)

    monkeypatch.setattr(pre, "begin_cuda_moist_stage", begin)
    monkeypatch.setattr(pre, "launch_cuda_eqtset2_stage_terms", terms)
    monkeypatch.setattr(pre, "launch_cuda_moist_buoyancy", buoyancy)

    def bc2d(*_args: object, **_kwargs: object) -> None:
        log.append("bc2d")

    def bcwsfc(*_args: object, **_kwargs: object) -> None:
        log.append("bcwsfc")

    momentum_rk = SimpleNamespace(
        target_arch="86", kernels=(lambda *_a, **_k: None, bc2d)
    )
    terrain_boundary = SimpleNamespace(target_arch="86", kernels=(None, None, bcwsfc))

    def momentum(*_args: object, **kwargs: object):
        log.append(f"momentum-rk{kwargs['nrk']}")
        receipt = {
            "schema": GM1_CHILD_STRETCHED_MOMENTUM_SCHEMA,
            "rrw_open_boundary": {
                "schema": pre.RRW_OPEN_BOUNDARY_SCHEMA,
                "performed": True,
                "flag": 0,
            },
        }
        return CudaMomentumAdvection5Launch(
            bridge.divx, FakeEvent(log), SimpleNamespace(), receipt
        )

    monkeypatch.setattr(pre, "launch_cuda_gm1_child_stretched_momentum5", momentum)
    return SimpleNamespace(
        log=log,
        stream=stream,
        held=held,
        prior_pre=prior_pre,
        completed=completed,
        arrays=arrays,
        metrics=metrics,
        bridge=bridge,
        vertical=vertical,
        terrain=terrain,
        base=base,
        held_outputs=held_outputs,
        momentum_rk=momentum_rk,
        terrain_boundary=terrain_boundary,
    )


@pytest.mark.parametrize("requested_nrk", (2, 3))
def test_runnable_launch_orders_next_stage_boundary_and_momentum(
    requested_nrk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _fake_live(monkeypatch, requested_nrk=requested_nrk)
    launch = pre.launch_cuda_gm1_tilestream_rk23_pre_acoustic_momentum(
        live.completed,
        live.prior_pre,
        live.held,
        live.held_outputs,
        live.arrays,
        live.metrics,
        live.terrain,
        live.vertical,
        live.bridge,
        requested_nrk=requested_nrk,
        **live.base,
        stream=live.stream,
        momentum_rk_compiled=live.momentum_rk,
        terrain_boundary_compiled=live.terrain_boundary,
        momentum_compiled=SimpleNamespace(),
    )
    assert live.log == [
        "begin-stage",
        "eqtset2",
        "held-seed",
        "held-seed",
        "held-seed",
        "bcwsfc",
        "bc2d",
        "bc2d",
        "bc2d",
        "bc2d",
        f"momentum-rk{requested_nrk}",
        "buoyancy",
        "event-record",
    ]
    assert launch.binding.arrays is not live.arrays
    assert dict(launch.binding.arrays) == live.arrays
    assert launch.binding.stream is live.stream
    assert launch.receipt["current_velocity_boundary_replay"] is True
    assert launch.receipt["fresh_current_stage_mass_flux"] is True
    assert launch.receipt["completed_rk_stage"] is False
    assert launch.binding.receipt["uniform_idealized_validation"] is False
    assert launch.receipt["uniform_idealized_validation"] is False


def test_launch_refuses_incomplete_prior_bridge_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _fake_live(monkeypatch, requested_nrk=2)
    live.bridge.post_acoustic_ready = False
    with pytest.raises(ValueError, match="bridge is not complete"):
        pre.launch_cuda_gm1_tilestream_rk23_pre_acoustic_momentum(
            live.completed,
            live.prior_pre,
            live.held,
            live.held_outputs,
            live.arrays,
            live.metrics,
            live.terrain,
            live.vertical,
            live.bridge,
            requested_nrk=2,
            **live.base,
            stream=live.stream,
            momentum_rk_compiled=live.momentum_rk,
            terrain_boundary_compiled=live.terrain_boundary,
            momentum_compiled=SimpleNamespace(),
        )
    assert live.log == []


def test_finish_emits_exact_acoustic_input_type_and_no_completed_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    live = _fake_live(monkeypatch, requested_nrk=2)
    launch = pre.launch_cuda_gm1_tilestream_rk23_pre_acoustic_momentum(
        live.completed,
        live.prior_pre,
        live.held,
        live.held_outputs,
        live.arrays,
        live.metrics,
        live.terrain,
        live.vertical,
        live.bridge,
        requested_nrk=2,
        **live.base,
        stream=live.stream,
        momentum_rk_compiled=live.momentum_rk,
        terrain_boundary_compiled=live.terrain_boundary,
        momentum_compiled=SimpleNamespace(),
    )
    monkeypatch.setattr(
        pre,
        "_physical_host",
        lambda *_args, **_kwargs: np.array([1.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        pre,
        "_raw_sha_host",
        lambda value: hashlib.sha256(value.tobytes()).hexdigest(),
    )
    result = pre.finish_cuda_gm1_tilestream_rk23_pre_acoustic_momentum(launch)
    assert type(result) is acoustic.FinishedGm1TiledRk23PreAcousticMomentum
    assert type(result.token) is acoustic.Gm1TiledRk23PreAcousticMomentumToken
    assert result.binding is launch.binding
    assert result.arrays is launch.binding.arrays
    assert result.stream is live.stream
    assert result.token.receipt["requested_nrk"] == 2
    assert (
        result.token.receipt["solve1_full_held_token_sha256"]
        == live.held.receipt["operation_token_sha256"]
    )
    assert result.token.receipt["completed_rk_stage"] is False
    assert tuple(result.token.receipt["output_sha256"]) == (
        acoustic.GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS
    )


def test_module_does_not_emit_completed_stage() -> None:
    assert not hasattr(pre, "finish_cuda_gm1_tiled_completed_rk_stage")
    assert "Acoustic" in pre.GM1_RK23_PRE_ACOUSTIC_CLAIM_LIMIT


def test_executable_stage_shell_composes_existing_exact_results_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[object] = []
    prior, prior_pre, held = object(), object(), object()
    held_outputs, arrays, metrics = {}, {}, {}
    terrain, vertical, bridge, stream = object(), object(), object(), object()
    pre_launch, pre_result, acoustic_launch, finished = (
        object(),
        object(),
        object(),
        object(),
    )

    def launch_pre(*args: object, **kwargs: object) -> object:
        log.append(("launch-pre", args, kwargs))
        return pre_launch

    def finish_pre(value: object) -> object:
        log.append(("finish-pre", value))
        return pre_result

    def launch_acoustic(value: object, **kwargs: object) -> object:
        log.append(("launch-acoustic", value, kwargs))
        return acoustic_launch

    def finish_acoustic(value: object, **kwargs: object) -> object:
        log.append(("finish-acoustic", value, kwargs))
        return finished

    monkeypatch.setattr(
        pre, "launch_cuda_gm1_tilestream_rk23_pre_acoustic_momentum", launch_pre
    )
    monkeypatch.setattr(
        pre, "finish_cuda_gm1_tilestream_rk23_pre_acoustic_momentum", finish_pre
    )
    monkeypatch.setattr(
        pre, "launch_cuda_gm1_tilestream_rk23_acoustic", launch_acoustic
    )
    monkeypatch.setattr(
        pre, "finish_cuda_gm1_tilestream_rk23_acoustic", finish_acoustic
    )

    def continuation(_handoff: object) -> None:
        return None

    result = pre.execute_cuda_gm1_tilestream_rk23_acoustic_stage(
        prior,
        prior_pre,
        held,
        held_outputs,
        arrays,
        metrics,
        terrain,
        vertical,
        bridge,
        requested_nrk=2,
        th0="th0",
        qv0="qv0",
        pi0="pi0",
        th3d="th3d",
        q3d="q3d",
        stream=stream,
        momentum_rk_compiled="rk",
        terrain_boundary_compiled="terrain",
        momentum_compiled="momentum",
        acoustic_compiled="sound",
        acoustic_open_compiled="open",
        acoustic_lateral_compiled="lateral",
        continuation=continuation,
    )

    assert result is finished
    assert [entry[0] for entry in log] == [
        "launch-pre",
        "finish-pre",
        "launch-acoustic",
        "finish-acoustic",
    ]
    assert log[0][1][:3] == (prior, prior_pre, held)
    assert log[1][1] is pre_launch
    assert log[2][1] is pre_result
    assert log[3][1] is acoustic_launch
    assert log[3][2]["continuation"] is continuation
