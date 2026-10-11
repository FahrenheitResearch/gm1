from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_gm1_tilestream_sgs_startup_ramp as live
from cm1core import cuda_gm1_tilestream_solve1_full_held as full_held
from cm1core import cuda_gm1_tilestream_stretched_closure as closure
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.sgs_startup_ramp import (
    SgsStartupRampSelection,
    cpu_sgs_startup_ramp_inplace,
)


ROOT = Path(__file__).resolve().parents[1]


def _selection(rtime: float = 0.0) -> SgsStartupRampSelection:
    return SgsStartupRampSelection(
        ramp_sgs=1,
        idoles=True,
        sgsmodel=1,
        rtime_seconds=rtime,
        ramp_time_seconds=1800.0,
    )


@pytest.mark.parametrize("rtime", [-10.0, 0.0, 900.0, 1800.0, 2400.0])
def test_uniform_live_coefficients_are_exact_cpu_operator_words(rtime: float) -> None:
    finished = live.build_gm1_sgs_startup_ramp_cpu_coefficients(
        selection=_selection(rtime),
        nx=4,
        ny=3,
        nk=2,
        dx=50.0,
        dy=50.0,
        allow_compact_validation=True,
    )
    direct = {
        "cm0": np.full((1, 1), np.float32(0.10), dtype=np.float32),
        **{
            name: np.zeros((2, 1, 1), dtype=np.float32)
            for name in ("cme", "ce1", "ce2", "csm")
        },
    }
    cpu_sgs_startup_ramp_inplace(
        direct, ni=1, nj=1, nk=1, selection=_selection(rtime)
    )
    for name in ("cme", "ce1", "ce2", "csm"):
        actual = np.asarray(np.float32(getattr(finished, name))).view(np.uint32)
        expected = direct[name][0, 0, 0].view(np.uint32)
        assert actual == expected, name
        assert finished.receipt["coefficient_raw_bits"][name] == int(expected)
    assert finished.receipt["selected_profile"] == {
        "cm1setup": 1,
        "sgsmodel": 1,
        "tconfig": 1,
        "cm0_raw_bits": 1036831949,
        "uniform_cm0": True,
    }
    assert finished.receipt["operation_token_sha256"] == live._operation_token(
        finished.receipt
    )


def test_live_selection_geometry_and_simultaneous_families_fail_closed() -> None:
    with pytest.raises(ValueError, match="300x300x75"):
        live.build_gm1_sgs_startup_ramp_cpu_coefficients(
            selection=_selection(), nx=4, ny=3, nk=2
        )
    with pytest.raises(ValueError, match="before RK stage 1"):
        live.build_gm1_sgs_startup_ramp_cpu_coefficients(
            selection=_selection(), completed_nrk=1
        )
    with pytest.raises(ValueError, match="sgsmodel=1"):
        live.build_gm1_sgs_startup_ramp_cpu_coefficients(
            selection=replace(_selection(), sgsmodel=2)
        )
    with pytest.raises(ValueError, match="cm1setup=1/tconfig=1"):
        live.build_gm1_sgs_startup_ramp_cpu_coefficients(
            selection=_selection(), cm1setup=4
        )
    with pytest.raises(ValueError, match="cm0=c_m"):
        live.build_gm1_sgs_startup_ramp_cpu_coefficients(
            selection=_selection(), cm0=0.09
        )
    for kwargs in (
        {"mpi": True},
        {"adaptive_dt": True},
        {"simultaneous_lspgrad": 1},
        {"simultaneous_rterm": 1},
        {"simultaneous_rayleigh_variant": True},
    ):
        pattern = "serial fixed-dt" if "mpi" in kwargs or "adaptive_dt" in kwargs else "simultaneous"
        with pytest.raises(ValueError, match=pattern):
            live.build_gm1_sgs_startup_ramp_cpu_coefficients(
                selection=_selection(), **kwargs
            )


class _Flags:
    c_contiguous = True


class _DeviceArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.dtype = np.dtype(np.float32)
        self.flags = _Flags()


class _Event:
    def __init__(self, *, disable_timing: bool) -> None:
        assert disable_timing is True
        self.recorded_on: _Stream | None = None

    def record(self, stream: "_Stream") -> None:
        self.recorded_on = stream


class _Stream:
    def __init__(self, ptr: int = 4419) -> None:
        self.ptr = ptr
        self.waited: list[_Event] = []

    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def wait_event(self, event: _Event) -> None:
        self.waited.append(event)


class _Kernel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((*args, kwargs))


@pytest.fixture
def fake_cupy(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    module = SimpleNamespace(
        ndarray=_DeviceArray,
        float32=np.float32,
        uint32=np.uint32,
        cuda=SimpleNamespace(Event=_Event),
    )
    monkeypatch.setitem(sys.modules, "cupy", module)
    return module


def test_prepared_binding_enqueues_exact_kernel_scalars_and_source_order(
    fake_cupy: SimpleNamespace,
) -> None:
    del fake_cupy
    stream = _Stream()
    kernel = _Kernel()
    compiled = live.CompiledCudaGm1TiledSgsStartupRamp(
        object(), kernel, live.CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_SHA256, "fake"
    )
    binding = live.prepare_cuda_gm1_tilestream_sgs_startup_ramp(
        selection=_selection(900.0), stream=stream, compiled=compiled
    )
    assert binding.ready_event.recorded_on is stream
    assert binding.receipt["selection"]["turbfac_raw_bits"] == int(
        np.asarray(np.float32(0.5)).view(np.uint32)
    )
    spec = build_gm1_open_tile_plan(
        nx=300, ny=300, tile_nx=96, tile_ny=96
    ).specs[5]
    mass = (75, spec.cny, spec.cnx)
    face = (76, spec.cny, spec.cnx)
    work = {
        name: _DeviceArray(mass if name == "grdscl" else face)
        for name in live._WORKSPACE_NAMES
    }
    receipt = live.enqueue_cuda_gm1_tilestream_sgs_startup_ramp(
        binding, work, spec=spec, nk=75, dx=50.0, dy=50.0, stream=stream
    )
    assert stream.waited == [binding.ready_event]
    assert len(kernel.calls) == 1
    args = kernel.calls[0][2]
    assert int(args[10]) == spec.cnx
    assert int(args[11]) == spec.cny
    assert int(args[12]) == 75
    assert np.asarray(args[13], dtype=np.float32).view(np.uint32) == (
        binding.receipt["coefficient_raw_bits"]["cme"]
    )
    assert np.asarray(args[14], dtype=np.float32).view(np.uint32) == (
        binding.receipt["coefficient_raw_bits"]["csm"]
    )
    assert receipt["same_stream_before_fresh_tke_copy"] is True
    assert receipt["binding_operation_token_sha256"] == binding.receipt[
        "operation_token_sha256"
    ]
    assert receipt["active_cuda_raw_bit_oracle"] is False

    tampered = replace(
        binding,
        coefficients=MappingProxyType(
            {**dict(binding.coefficients), "cme": float(np.float32(0.2))}
        ),
    )
    with pytest.raises(ValueError, match="coefficient inventory or raw bits"):
        live.enqueue_cuda_gm1_tilestream_sgs_startup_ramp(
            tampered, work, spec=spec, nk=75, dx=50.0, dy=50.0, stream=stream
        )


def _closure_launch_receipt(
    spec: object, *, stream_pointer: int, ramp: bool
) -> MappingProxyType:
    tile = spec.record()
    receipt: dict[str, object] = {
        "tile": tile,
        "stream_pointer": stream_pointer,
        "reduction_operation_token_sha256": "r" * 64,
        "closure_lifecycle": "fresh-initial-tkea",
        "vertical_disposition": "sealed-stretched-child",
    }
    if ramp:
        ramp_receipt: dict[str, object] = {
            "schema": live.CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_SCHEMA,
            "status": live.CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_STATUS,
            "completed_nrk": 0,
            "tile": tile,
            "stream_pointer": stream_pointer,
            "binding_operation_token_sha256": "a" * 64,
            "cpu_operator_operation_token_sha256": "b" * 64,
            "selection": {"ramp_sgs": 1, "sgsmodel": 1, "turbfac_raw_bits": 0},
            "selected_profile": {"cm1setup": 1, "sgsmodel": 1, "tconfig": 1},
            "coefficient_raw_bits": {"cme": 1, "ce1": 2, "ce2": 3, "csm": 4},
            "retained_tke_admitted": False,
        }
        ramp_receipt["operation_token_sha256"] = live._operation_token(ramp_receipt)
        receipt["sgs_startup_ramp"] = ramp_receipt
        receipt["sgs_startup_ramp_active"] = True
    receipt["operation_token_sha256"] = closure._operation_token(receipt)
    return MappingProxyType(receipt)


def test_closure_finish_aggregates_active_only_and_rejects_mixed_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96)
    reduction_receipt = {
        "schema": closure.CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_REDUCTION_SCHEMA,
        "stream_pointer": 97,
    }
    reduction_receipt["operation_token_sha256"] = closure._operation_token(
        reduction_receipt
    )

    class _Counter:
        @staticmethod
        def get() -> np.ndarray:
            return np.zeros((1,), dtype=np.uint32)

    class _FinishEvent(_Event):
        def synchronize(self) -> None:
            return None

    stream = _Stream(97)
    reduction = closure.Gm1TiledStretchedClosureReduction(
        _Counter(), object(), 97, MappingProxyType(reduction_receipt)
    )
    launches = [
        closure.CudaGm1TiledStretchedClosureLaunch(
            object(),
            {},
            {},
            object(),
            _closure_launch_receipt(spec, stream_pointer=97, ramp=True),
        )
        for spec in plan.specs
    ]
    fake = SimpleNamespace(cuda=SimpleNamespace(Event=_FinishEvent))
    monkeypatch.setattr(
        closure.importlib,
        "import_module",
        lambda name: fake if name == "cupy" else __import__(name),
    )
    # The launch receipt binds the reduction's canonical token, not a fixture label.
    normalized = []
    for launch in launches:
        record = dict(launch.receipt)
        record["reduction_operation_token_sha256"] = reduction_receipt[
            "operation_token_sha256"
        ]
        record["operation_token_sha256"] = closure._operation_token(record)
        normalized.append(replace(launch, receipt=MappingProxyType(record)))
    token = closure.finish_cuda_gm1_tiled_stretched_closure(
        reduction, normalized, plan=plan, stream=stream
    )
    assert token.receipt["sgs_startup_ramp_active"] is True
    assert token.receipt["sgs_startup_ramp_binding_operation_token_sha256"] == "a" * 64
    assert token.receipt["sgs_startup_ramp_cpu_operator_operation_token_sha256"] == "b" * 64
    assert token.receipt["sgs_startup_ramp_active_cuda_raw_bit_oracle"] is False
    assert token.receipt["source_order"] == list(
        closure.SOURCE_ORDER_WITH_ACTIVE_SGS_STARTUP_RAMP
    )
    mixed = list(normalized)
    mixed[-1] = replace(
        mixed[-1],
        receipt=_closure_launch_receipt(
            plan.specs[-1], stream_pointer=97, ramp=False
        ),
    )
    changed = dict(mixed[-1].receipt)
    changed["reduction_operation_token_sha256"] = reduction_receipt[
        "operation_token_sha256"
    ]
    changed["operation_token_sha256"] = closure._operation_token(changed)
    mixed[-1] = replace(mixed[-1], receipt=MappingProxyType(changed))
    with pytest.raises(ValueError, match="mix active and disabled"):
        closure.finish_cuda_gm1_tiled_stretched_closure(
            reduction, mixed, plan=plan, stream=stream
        )


def test_default_abi_is_omitted_and_live_source_order_is_strict() -> None:
    parameter = inspect.signature(
        closure.launch_cuda_gm1_tiled_stretched_closure
    ).parameters["sgs_startup_ramp_binding"]
    assert parameter.default is None
    source = inspect.getsource(closure.launch_cuda_gm1_tiled_stretched_closure)
    positions = [
        source.index("compiled.fresh_tke.kernels[1]("),
        source.index("enqueue_cuda_gm1_tilestream_sgs_startup_ramp("),
        source.index("compiled.fresh_tke.kernels[2]("),
    ]
    assert positions == sorted(positions)
    assert "sgs_startup_ramp_active" not in closure.SOURCE_ORDER
    assert "sgs_startup_ramp" not in {
        "schema": closure.CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_SCHEMA,
        "source_order": list(closure.SOURCE_ORDER),
    }
    assert "active SGS startup ramp has no admitted simultaneous" in inspect.getsource(
        full_held.launch_cuda_gm1_tiled_solve1_full_held
    )
    assert "restored retained TKE" in source


def test_matrix_and_validation_receipt_seal_only_cpu_typed_binding() -> None:
    matrix = json.loads(
        (ROOT / "src/cm1core/records/CM1-DYNAMICS-BRANCH-MATRIX.json").read_bytes()
    )
    profile = matrix["admitted_profiles"][
        "sgs-startup-ramp-all-bounded-branches-cpu-operator"
    ]
    typed = profile["typed_fresh_closure_binding"]
    assert profile["admission"] == "operator-only"
    assert typed["admission"] == "cpu-tested-source-order-binding-only"
    assert typed["ramp_sgs0_receipt_shape_unchanged"] is True
    assert typed["retained_tke_admitted"] is False
    assert typed["active_cuda_raw_bit_oracle"] is False
    assert typed["coupled_trajectory_admitted"] is False

    receipt = json.loads(
        (
            ROOT
            / "evidence/validation/cm1-sgs-startup-ramp-live-source-order-binding-v1/validation-receipt.json"
        ).read_bytes()
    )
    assert receipt["status"] == "PASS_CPU_TYPED_SOURCE_ORDER_BINDING_ONLY"
    assert receipt["execution"]["gpu_used"] is False
    assert receipt["result"]["ramp_sgs0_default_receipt_shape_unchanged"] is True
    assert receipt["result"]["active_cuda_raw_bit_oracle"] is False
    for artifact in receipt["artifacts"].values():
        path = ROOT / artifact["path"]
        raw = path.read_bytes()
        assert len(raw) == artifact["bytes"]
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
