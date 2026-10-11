from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import cm1core.cuda_gm1_child_step_composer as composer


def test_packaged_child_core_source_is_sealed() -> None:
    assert (
        hashlib.sha256(composer.GM1_CHILD_CORE_SOURCE.encode()).hexdigest()
        == composer.GM1_CHILD_CORE_SOURCE_SHA256
    )
    assert (
        hashlib.sha256(composer.GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE.encode()).hexdigest()
        == composer.GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE_SHA256
    )
    assert (
        hashlib.sha256(composer.GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE.encode()).hexdigest()
        == composer.GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE_SHA256
    )
    assert (
        hashlib.sha256(
            composer.GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE.encode()
        ).hexdigest()
        == composer.GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE_SHA256
    )


def test_composer_has_no_uniform_z_operator_fallbacks() -> None:
    source = inspect.getsource(composer)
    forbidden = (
        "compile_cuda_momentum_advection5",
        "launch_cuda_terrain_open_momentum_advection5",
        "compile_cuda_psolver3_terrain_open",
        "launch_cuda_psolver3_terrain_open_stage",
        "compile_cuda_advection5",
        "launch_cuda_scalar_moist_rk_stage",
        "launch_cuda_post_acoustic_flux_bridge",
        "launch_cuda_gm1_closure_stage",
        "launch_cuda_gm1_scalar_turbulence",
    )
    assert not {name for name in forbidden if name in source}


def test_source_order_anchors_match_corrected_r21_1_ranges() -> None:
    assert composer.SOURCE_ANCHORS["solve1"] == (
        "src/solve1.F:317-383,414-454,917-1001,1115-1240,1696-1777"
    )
    assert composer.SOURCE_ANCHORS["solve2"] == (
        "src/solve2.F:490-627,1277-1803,1973-2030"
    )
    assert composer.SOURCE_ANCHORS["solve3"] == "src/solve3.F:270-337,548-680"
    assert composer.SOURCE_ANCHORS["live_nm_def_and_coefficients"] == (
        "src/turb.F:1326-1413"
    )


def test_solve1_resolution_refuses_before_unfrozen_scalar_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(composer, "GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_COMMIT", "")
    with pytest.raises(RuntimeError, match="stretched scalar-turbulence commit"):
        composer._load_gm1_child_solve1_operator_functions()


def test_solve1_resolution_checks_both_operator_source_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(composer, "GM1_CHILD_STRETCHED_NM_DEF_COMMIT", "nm-commit")
    monkeypatch.setattr(
        composer, "GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_COMMIT", "scalar-commit"
    )
    nm_launch = object()
    scalar_launch = object()
    modules = {
        "cm1core.cuda_gm1_child_stretched_nm_def": SimpleNamespace(
            __file__="nm.py",
            CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256=(
                composer.GM1_CHILD_STRETCHED_NM_DEF_SOURCE_SHA256
            ),
            launch_cuda_gm1_child_stretched_nm_def_refresh=nm_launch,
        ),
        "cm1core.cuda_gm1_child_stretched_scalar_turbulence": SimpleNamespace(
            __file__="scalar.py",
            CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256=(
                composer.GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256
            ),
            launch_cuda_gm1_child_stretched_scalar_turbulence=scalar_launch,
        ),
    }
    monkeypatch.setattr(composer.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        composer,
        "_hash_file",
        lambda path: (
            composer.GM1_CHILD_STRETCHED_NM_DEF_MODULE_SHA256
            if path.name == "nm.py"
            else composer.GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_MODULE_SHA256
        ),
    )
    functions = composer._load_gm1_child_solve1_operator_functions()
    assert functions.launch_nm_def is nm_launch
    assert functions.launch_scalar_turbulence is scalar_launch


def test_frozen_solve1_modules_resolve_from_committed_bytes() -> None:
    functions = composer._load_gm1_child_solve1_operator_functions()
    assert callable(functions.launch_nm_def)
    assert callable(functions.launch_scalar_turbulence)


def test_begin_consumes_prepared_closure_without_rediagnosing_it() -> None:
    source = inspect.getsource(composer.begin_cuda_gm1_child_large_step)
    assert "operators.launch_nm_def" not in source
    assert "launch_cuda_sgs_tke" not in source
    # The initial step's prepared closure is the corrected stretched-z arena
    # the run actually loaded, reported from that artifact rather than from a
    # constant naming the superseded uniform-z publication.
    closure_export = SimpleNamespace(payload_sha256="7c" * 32)
    initial = SimpleNamespace(
        step_index=0,
        last_completed_closure_refresh=None,
        closure_export=closure_export,
    )
    prepared = composer._require_gm1_child_prepared_closure(initial)
    assert prepared["payload_sha256"] == closure_export.payload_sha256
    assert prepared["schema"] == composer.GM1_CHILD_CLOSURE_SCHEMA
    assert prepared["schema"].endswith("/v3")

    receipt = {"schema": composer.GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SCHEMA}
    refresh = SimpleNamespace(step_index=0, receipt=receipt)
    continued = SimpleNamespace(
        step_index=1,
        last_completed_closure_refresh=refresh,
        closure_export=closure_export,
    )
    assert composer._require_gm1_child_prepared_closure(continued) is receipt
    stale = SimpleNamespace(
        step_index=2,
        last_completed_closure_refresh=refresh,
        closure_export=closure_export,
    )
    with pytest.raises(ValueError, match="immediately preceding"):
        composer._require_gm1_child_prepared_closure(stale)


def _finish_state() -> composer.CudaGm1ChildStepState:
    state = object.__new__(composer.CudaGm1ChildStepState)
    stream = SimpleNamespace(ptr=7301)
    state.selection = SimpleNamespace(nx=3, ny=4, nz=5, dt_s=0.5)
    state.next_stage = 4
    state.solve1_ready = True
    state.step_index = 0
    state.time_seconds = 0.0
    state.stream = stream
    state.mass1_f64 = 123.0
    state.post_final = SimpleNamespace(next_step_index=0, poisoned=False)
    state.scalar_rk = SimpleNamespace(state=SimpleNamespace(completed=True, next_nrk=4))
    state.closure = SimpleNamespace(
        completed_stage=3,
        refresh_index=1,
        tke={"tkea": object(), "tke3d": object()},
        sgs_tke_compiled=object(),
    )
    state.retained = []
    state.last_completed_closure_refresh = None
    state.last_receipt = None
    return state


def test_finish_composes_v2_callbacks_and_resets_next_step_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _finish_state()
    calls: list[str] = []
    manifest_sha = "a" * 64
    final_sync = SimpleNamespace(operation_token_sha256="b" * 64)
    closure_result = SimpleNamespace(operation_token_sha256="c" * 64)

    monkeypatch.setattr(
        composer,
        "require_child_post_final_rk_operator_sources",
        lambda _operators: calls.append("manifest") or manifest_sha,
    )
    monkeypatch.setattr(
        composer,
        "sync_cuda_gm1_child_final_dynamics_to_product",
        lambda _state: calls.append("sync") or final_sync,
    )
    monkeypatch.setattr(
        composer,
        "_launch_cuda_gm1_child_apmasscon",
        lambda _state: calls.append("apmasscon") or SimpleNamespace(receipt={}),
    )
    monkeypatch.setattr(
        composer,
        "launch_cuda_gm1_child_solve3_rho_boundary",
        lambda *_args, **_kwargs: calls.append("rho") or SimpleNamespace(receipt={}),
    )
    monkeypatch.setattr(
        composer,
        "launch_cuda_gm1_child_stretched_tke_commit",
        lambda *_args, **_kwargs: calls.append("tke")
        or SimpleNamespace(receipt={"schema": "tke"}),
    )

    def closure_refresh(*_args: object, **_kwargs: object) -> object:
        calls.append("closure")
        state.last_completed_closure_refresh = closure_result
        state.closure.refresh_index = 2
        return closure_result

    monkeypatch.setattr(
        composer,
        "launch_cuda_gm1_child_completed_closure_refresh",
        closure_refresh,
    )

    def post_final(_post: object, **kwargs: object) -> object:
        calls.append("post-final")
        cadence = kwargs["cadence"]
        assert cadence.step_index == cadence.apmasscon_target_step_index == 0
        assert kwargs["final_dynamics_sync"] is final_sync
        kwargs["apmasscon_hook"](0, state.stream)
        kwargs["rho_boundary_hook"](0, state.stream)
        tke = kwargs["tke_commit_hook"](0, state.stream)
        closure = kwargs["closure_refresh_hook"](0, state.stream, tke.receipt)
        return SimpleNamespace(
            completed_step_index=1,
            operator_manifest_sha256=manifest_sha,
            stream_pointer=state.stream.ptr,
            closure_refresh=closure,
            ready_event=object(),
            receipt={"schema": "gm1.cuda-child-post-final-rk/v2"},
        )

    monkeypatch.setattr(composer, "launch_cuda_gm1_child_post_final_rk", post_final)

    def reset_scalar(scalar: object) -> None:
        calls.append("scalar-reset")
        scalar.completed = False
        scalar.next_nrk = 1

    monkeypatch.setattr(composer, "begin_cuda_scalar_moist_rk_step", reset_scalar)
    receipt = composer.finish_cuda_gm1_child_large_step(
        state, operator_sources={"sealed": "d" * 64}
    )
    assert calls == [
        "manifest",
        "sync",
        "post-final",
        "apmasscon",
        "rho",
        "tke",
        "closure",
        "scalar-reset",
    ]
    assert state.step_index == 1
    assert state.time_seconds == 0.5
    assert state.next_stage == 1
    assert state.solve1_ready is False
    assert state.scalar_rk.state.next_nrk == 1
    assert state.closure.refresh_index == 2
    assert receipt["publication"]["rust_pair_directory"] == "step-00000001"
    assert receipt["operator_manifest_sha256"] == manifest_sha


def test_finish_refuses_bad_manifest_before_final_product_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _finish_state()
    synchronized = False

    def bad_manifest(_operators: object) -> str:
        raise ValueError("operator manifest drift")

    def sync(_state: object) -> object:
        nonlocal synchronized
        synchronized = True
        return object()

    monkeypatch.setattr(
        composer, "require_child_post_final_rk_operator_sources", bad_manifest
    )
    monkeypatch.setattr(
        composer, "sync_cuda_gm1_child_final_dynamics_to_product", sync
    )
    with pytest.raises(ValueError, match="operator manifest drift"):
        composer.finish_cuda_gm1_child_large_step(state, operator_sources={})
    assert synchronized is False


def test_sgs_momentum_accumulators_are_cleared_on_every_large_step() -> None:
    closure = SimpleNamespace(
        stress={
            "uten": np.full((2,), np.float32(1.0)),
            "vten": np.full((2,), np.float32(-2.0)),
            "wten": np.full((2,), np.float32(3.0)),
        }
    )
    composer._clear_gm1_child_stress_tendencies(closure, np)
    assert all(np.array_equal(value, np.zeros_like(value)) for value in closure.stress.values())
    for value in closure.stress.values():
        value.fill(np.float32(9.0))
    composer._clear_gm1_child_stress_tendencies(closure, np)
    assert all(np.array_equal(value, np.zeros_like(value)) for value in closure.stress.values())


def test_prior_step_launch_retention_does_not_grow_across_steps() -> None:
    retained: list[object] = []
    closure_retained: list[object] = []
    state = SimpleNamespace(
        retained=retained,
        closure=SimpleNamespace(retained=closure_retained),
        last_final_dynamics_sync=object(),
    )
    for _step in range(2):
        state.retained.extend(object() for _ in range(7))
        state.closure.retained.extend(object() for _ in range(7))
        composer._reset_gm1_child_step_retention(state)
        assert len(state.retained) == 0
        assert len(state.closure.retained) == 0
        assert state.last_final_dynamics_sync is None
    assert state.retained is retained
    assert state.closure.retained is closure_retained


def test_apmasscon_callback_binds_live_eos_and_mass_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields = {
        name: object()
        for name in ("rho", "th3d", "q3d", "pp3d", "prs")
    }
    fields["q3d"] = [object()]
    captured: dict[str, object] = {}

    def fake_launch(arrays: dict[str, object], **kwargs: object) -> str:
        captured["arrays"] = arrays
        captured.update(kwargs)
        return "apmasscon-launch"

    monkeypatch.setattr(composer, "launch_cuda_apmasscon", fake_launch)
    state = SimpleNamespace(
        selection=SimpleNamespace(nx=3, ny=4, nz=5),
        product=SimpleNamespace(field=fields.__getitem__),
        prepared=SimpleNamespace(
            arrays={"uh": np.arange(9), "vh": np.arange(10)},
            metrics={"rmh": object()},
        ),
        pi0_3d=object(),
        th0_3d=object(),
        mass1_f64=123.5,
        stream=object(),
        apmasscon_compiled=object(),
    )
    assert composer._launch_cuda_gm1_child_apmasscon(state) == "apmasscon-launch"
    arrays = captured["arrays"]
    assert isinstance(arrays, dict)
    assert set(arrays) == {
        "rho",
        "ruh",
        "rvh",
        "rmh",
        "pi0",
        "th0",
        "th3d",
        "qv",
        "pp3d",
        "prs",
    }
    assert captured["mass1"] == 123.5


def test_final_rk3_dynamics_sync_overwrites_poisoned_product_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        ptr = 731

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeEvent:
        def __init__(self, *, disable_timing: bool) -> None:
            assert disable_timing
            self.recorded_stream: FakeStream | None = None

        def record(self, stream: FakeStream) -> None:
            self.recorded_stream = stream

    stream = FakeStream()
    fake_cupy = SimpleNamespace(
        copyto=np.copyto,
        cuda=SimpleNamespace(Event=FakeEvent),
    )
    monkeypatch.setattr(
        composer.importlib,
        "import_module",
        lambda name: fake_cupy if name == "cupy" else __import__(name),
    )

    nx = ny = nz = 2
    prepared = {
        "u3d": np.arange(4 * 8 * 9, dtype=np.float32).reshape(4, 8, 9),
        "v3d": np.arange(4 * 9 * 8, dtype=np.float32).reshape(4, 9, 8) + 1000,
        "w3d": np.arange(5 * 8 * 8, dtype=np.float32).reshape(5, 8, 8) + 2000,
        "pp3d": np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4) + 3000,
    }
    product_fields = {
        "u3d": np.full((2, 2, 3), np.float32(-101)),
        "v3d": np.full((2, 3, 2), np.float32(-102)),
        "w3d": np.full((3, 2, 2), np.float32(-103)),
        "pp3d": np.full((2, 2, 2), np.float32(-104)),
    }
    product = SimpleNamespace(
        stream=stream,
        field=product_fields.__getitem__,
    )
    state = object.__new__(composer.CudaGm1ChildStepState)
    state.selection = SimpleNamespace(nx=nx, ny=ny, nz=nz)
    state.prepared = SimpleNamespace(arrays=prepared)
    state.product = product
    state.stream = stream
    state.solve1_ready = True
    state.next_stage = 4
    state.closure = SimpleNamespace(completed_stage=3)
    state.last_final_dynamics_sync = None
    state.step_index = 7
    state.retained = []
    state.ready_event = None

    result = composer.sync_cuda_gm1_child_final_dynamics_to_product(state)
    ng = composer.CM1_OPEN_NGXY
    np.testing.assert_array_equal(
        product_fields["u3d"], prepared["u3d"][1:3, ng : ng + 2, ng : ng + 3]
    )
    np.testing.assert_array_equal(
        product_fields["v3d"], prepared["v3d"][1:3, ng : ng + 3, ng : ng + 2]
    )
    np.testing.assert_array_equal(
        product_fields["w3d"], prepared["w3d"][1:4, ng : ng + 2, ng : ng + 2]
    )
    np.testing.assert_array_equal(
        product_fields["pp3d"], prepared["pp3d"][1:3, 1:3, 1:3]
    )
    descriptor = dict(result.receipt)
    assert descriptor.pop("operation_token_sha256") == result.operation_token_sha256
    assert composer._canonical_receipt_token(descriptor) == result.operation_token_sha256
    assert result.receipt["producer_source_sha256"] == (
        composer.GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE_SHA256
    )
    assert result.ready_event.recorded_stream is stream
    assert state.retained == [result]
    with pytest.raises(ValueError, match="already synchronized"):
        composer.sync_cuda_gm1_child_final_dynamics_to_product(state)


def test_solve3_rho_boundary_publishes_exact_physical_slice_and_open_halo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        ptr = 811

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeEvent:
        def __init__(self, *, disable_timing: bool) -> None:
            assert disable_timing

        def record(self, stream: FakeStream) -> None:
            assert stream.ptr == 811

    order: list[str] = []

    def we(_grid: object, _block: object, args: tuple[object, ...], **_kw: object) -> None:
        order.append("we")
        padded = args[0]
        padded[1:3, :, 0] = padded[1:3, :, 1]
        padded[1:3, :, 3] = padded[1:3, :, 2]

    def sn(_grid: object, _block: object, args: tuple[object, ...], **_kw: object) -> None:
        order.append("sn")
        padded = args[0]
        padded[1:3, 0, :] = padded[1:3, 1, :]
        padded[1:3, 3, :] = padded[1:3, 2, :]

    stream = FakeStream()
    fake_cupy = SimpleNamespace(
        copyto=np.copyto,
        uint32=np.uint32,
        cuda=SimpleNamespace(Event=FakeEvent),
    )
    monkeypatch.setattr(
        composer.importlib,
        "import_module",
        lambda name: fake_cupy if name == "cupy" else __import__(name),
    )
    physical = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 1
    padded = np.zeros((4, 4, 4), dtype=np.float32)
    state = object.__new__(composer.CudaGm1ChildStepState)
    state.selection = SimpleNamespace(nx=2, ny=2, nz=2)
    state.step_index = 0
    state.stream = stream
    state.prepared = SimpleNamespace(arrays={"rho": padded})
    state.product = SimpleNamespace(field={"rho": physical}.__getitem__)
    state.open_compiled = SimpleNamespace(kernels=(None, None, None, we, sn))
    state.retained = []

    result = composer.launch_cuda_gm1_child_solve3_rho_boundary(
        state, step_index=0, stream=stream
    )
    np.testing.assert_array_equal(padded[1:3, 1:3, 1:3], physical)
    np.testing.assert_array_equal(padded[1:3, 0, :], padded[1:3, 1, :])
    np.testing.assert_array_equal(padded[1:3, 3, :], padded[1:3, 2, :])
    assert order == ["we", "sn"]
    assert result.receipt["producer_source_sha256"] == (
        composer.GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE_SHA256
    )


def test_completed_step_closure_refresh_prepares_exact_persistent_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        ptr = 919

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeEvent:
        def __init__(self, *, disable_timing: bool) -> None:
            assert disable_timing

        def record(self, stream: FakeStream) -> None:
            assert stream.ptr == 919

    stream = FakeStream()
    face = (3, 1, 1)
    nm_outputs = {
        name: np.full(face, np.float32(value))
        for name, value in (("nm", 1), ("defv", 2), ("defh", 3))
    }
    workspace = {
        name: np.zeros(face, dtype=np.float32)
        for name in ("s11", "s22", "s33", "s12", "s13", "s23")
    }
    nm_launch = SimpleNamespace(
        arrays=nm_outputs,
        workspace=workspace,
        receipt={"schema": "nm-def"},
        compiled_calcdef=object(),
        compiled_calcnm=object(),
    )

    def fake_nm_def(*_args: object, **kwargs: object) -> object:
        assert kwargs["refresh_index"] == 1
        return nm_launch

    def fake_sgs(arrays: dict[str, object], **_kwargs: object) -> object:
        assert np.count_nonzero(arrays["dissten"]) == 0
        assert np.count_nonzero(arrays["tketen"]) == 0
        arrays["dissten"].fill(np.float32(4))
        arrays["tketen"].fill(np.float32(5))
        return SimpleNamespace(receipt={"schema": "sgs"})

    fake_cupy = SimpleNamespace(
        copyto=np.copyto,
        float32=np.float32,
        cuda=SimpleNamespace(Event=FakeEvent),
    )
    monkeypatch.setattr(
        composer.importlib,
        "import_module",
        lambda name: fake_cupy if name == "cupy" else __import__(name),
    )
    monkeypatch.setattr(
        composer,
        "_load_gm1_child_solve1_operator_functions",
        lambda: SimpleNamespace(launch_nm_def=fake_nm_def),
    )
    monkeypatch.setattr(composer, "launch_cuda_sgs_tke", fake_sgs)

    persistent_names = (
        "grdscl",
        "lenscl",
        "dissten",
        "kmh",
        "kmv",
        "khh",
        "khv",
        "tkea",
        "tketen",
    )
    sgs = {name: np.ones(face, dtype=np.float32) for name in persistent_names}
    sgs.update(nm_outputs)
    closure = SimpleNamespace(
        sgs=sgs,
        stress={},
        tke={"tke3d": np.zeros(face, dtype=np.float32)},
        refresh_index=1,
        completed_stage=3,
        retained=[],
        sgs_compiled=object(),
        nm_def_compiled_calcdef=None,
        nm_def_compiled_calcnm=None,
        last_nm_def=None,
    )
    fields = {
        name: np.zeros((1,), dtype=np.float32)
        for name in (
            "c1",
            "c2",
            "mf",
            "pi0",
            "th0",
            "prs",
            "pp3d",
            "th3d",
            "q3d",
            "u3d",
            "v3d",
            "w3d",
            "rho",
            "rf",
            "gz",
            "rgz",
            "gzu",
            "rgzu",
            "gzv",
            "rgzv",
            "gx",
            "gy",
        )
    }
    state = object.__new__(composer.CudaGm1ChildStepState)
    state.selection = SimpleNamespace(
        nx=1, ny=1, nz=2, dx_m=50.0, dy_m=50.0, dz_m=100.0
    )
    state.step_index = 0
    state.stream = stream
    state.static_metrics = object()
    state.vertical = SimpleNamespace(ztop_f32=15000.0)
    state.product = SimpleNamespace(field=fields.__getitem__)
    state.closure = closure
    state.last_completed_closure_refresh = None
    state.last_final_dynamics_sync = SimpleNamespace(step_index=0)
    state.ready_event = None
    state.retained = []

    tke_receipt = {"schema": "tke-commit", "completed_nrk": 3}
    result = composer.launch_cuda_gm1_child_completed_closure_refresh(
        state,
        step_index=0,
        stream=stream,
        tke_commit_receipt=tke_receipt,
    )
    assert tuple(result.arrays) == (
        "nm",
        "defv",
        "defh",
        "grdscl",
        "lenscl",
        "dissten",
        "kmh",
        "kmv",
        "khh",
        "khv",
        "tkea",
        "tke3d",
        "tketen",
    )
    assert result.receipt["refresh_index"] == 1
    assert result.receipt["completed_step_index"] == 1
    assert closure.refresh_index == 2
    assert closure.completed_stage == 0
    descriptor = dict(result.receipt)
    assert descriptor.pop("operation_token_sha256") == result.operation_token_sha256
    assert composer._canonical_receipt_token(descriptor) == result.operation_token_sha256
    assert np.all(result.arrays["dissten"] == np.float32(4))
    assert np.all(result.arrays["tketen"] == np.float32(5))
