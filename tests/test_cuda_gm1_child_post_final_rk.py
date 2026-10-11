from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import cm1core.cuda_gm1_child_post_final_rk as post
import cm1core.eqtset2_wsm6_bridge as bridge_module
from cm1core.arwen_wsm6_component import (
    ARWEN_WSM6_COMPONENT_ID,
    WSM6_COMPONENT_PERSISTENCE,
    WSM6_SURFACE_SCRATCH,
)
from cm1core.cuda_gm1_child_post_final_rk import (
    CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_ABI,
    CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_SCHEMA,
    CudaGm1ChildPostFinalRkState,
    Gm1ChildPostFinalRkCadence,
    child_post_final_rk_metadata,
    child_post_final_rk_metadata_sha256,
    child_post_final_rk_operator_source_pins,
    create_cuda_gm1_child_post_final_rk_state,
    launch_cuda_gm1_child_post_final_rk,
    require_child_post_final_rk_cadence,
    require_child_post_final_rk_operator_sources,
)
from cm1core.cuda_apmasscon import CUDA_APMASSCON_SCHEMA, CudaApMassConLaunch
from cm1core.cuda_gm1_child_stretched_sgs_tke import (
    CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI,
    CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA,
    GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT,
    SOURCE_ANCHORS as SGS_TKE_SOURCE_ANCHORS,
    CudaChildStretchedLaunch,
)
from cm1core.cuda_wsm6_pdefq import (
    CompiledWsm6Pdefq,
    CudaWsm6PdefqLaunch,
    Wsm6PdefqWorkspace,
)
from cm1core.eqtset2_wsm6_bridge import (
    EQTSET2_HELD_FIELDS,
    Eqtset2Wsm6Bridge,
    Eqtset2Wsm6Carriers,
)
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]
REAL_RECEIPT = (
    ROOT / "evidence" / "validation" / "gm1-child-post-final-rk-v2" / "receipt.json"
)
HISTORICAL_V1_RECEIPT = (
    ROOT / "evidence" / "validation" / "gm1-child-post-final-rk-v1" / "receipt.json"
)


class _DeviceArray(np.ndarray):
    @property
    def device(self) -> SimpleNamespace:
        return SimpleNamespace(id=0)


def _device(values: object, *, dtype: object = np.float32) -> _DeviceArray:
    return np.asarray(values, dtype=dtype).copy(order="C").view(_DeviceArray)


class _FakeEvent:
    def __init__(self, *, disable_timing: bool) -> None:
        self.disable_timing = disable_timing
        self.recorded_stream = None

    def record(self, stream: object) -> None:
        self.recorded_stream = stream


class _FakeStream:
    def __init__(self, owner: "_FakeCupy", pointer: int) -> None:
        self.owner = owner
        self.ptr = pointer
        self.previous = None

    def __enter__(self):
        self.previous = self.owner.current_stream
        self.owner.current_stream = self
        return self

    def __exit__(self, *_args):
        self.owner.current_stream = self.previous


class _FakeCupy:
    ndarray = _DeviceArray
    float32 = np.float32
    float64 = np.float64
    uint32 = np.uint32
    add = staticmethod(np.add)
    subtract = staticmethod(np.subtract)
    multiply = staticmethod(np.multiply)
    divide = staticmethod(np.divide)
    reciprocal = staticmethod(np.reciprocal)
    maximum = staticmethod(np.maximum)
    power = staticmethod(np.power)
    copyto = staticmethod(np.copyto)
    all = staticmethod(np.all)
    min = staticmethod(np.min)
    isfinite = staticmethod(np.isfinite)
    array_equal = staticmethod(np.array_equal)

    def __init__(self, pointer: int = 9107) -> None:
        self.stream = _FakeStream(self, pointer)
        self.current_stream = self.stream
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(getDevice=lambda: 0),
            get_current_stream=lambda: self.current_stream,
            Event=_FakeEvent,
        )

    @staticmethod
    def empty_like(value: _DeviceArray) -> _DeviceArray:
        return np.empty_like(value, subok=True)

    @staticmethod
    def asnumpy(value: _DeviceArray) -> np.ndarray:
        return np.asarray(value).copy()


class _FakeKernel:
    def __init__(self, apply=None) -> None:
        self.calls = 0
        self.apply = apply

    def __call__(self, _grid, _block, args, *, stream) -> None:
        assert stream is not None
        self.calls += 1
        if self.apply is not None:
            self.apply(args)


class _FakeComponent:
    component_id = ARWEN_WSM6_COMPONENT_ID

    def __init__(self, *, nx: int, ny: int, nz: int, dt_s: float) -> None:
        self.nx, self.ny, self.nz, self.dt_s = nx, ny, nz, dt_s
        self.calls = 0
        volume, surface = (nz, ny, nx), (ny, nx)
        self.persistence = {
            name: _device(
                np.zeros(
                    surface if name in WSM6_SURFACE_SCRATCH else volume,
                    dtype=np.float32,
                )
            )
            for name in WSM6_COMPONENT_PERSISTENCE
        }

    def apply(self, carriers, *, refl_10cm_due: bool):
        assert type(refl_10cm_due) is bool
        self.calls += 1
        if hasattr(self, "sequence"):
            self.sequence.append("wsm6")
        np.add(carriers.qc, np.float32(2.0e-7), out=carriers.qc)
        return SimpleNamespace(
            stream_pointer=9107,
            call_index=self.calls,
            reflectivity_10cm=None,
        )

    def persistent_arrays(self):
        return self.persistence

    def receipt_dict(self):
        return {"schema": "fake-wsm6/v1", "call_index": self.calls}

    def receipt_sha256(self):
        raw = (json.dumps(self.receipt_dict(), sort_keys=True) + "\n").encode()
        return hashlib.sha256(raw).hexdigest()


class _Product:
    def __init__(self, fields):
        self.fields = fields

    def field(self, name):
        return self.fields[name]


class _Transition:
    def __init__(self, bridge, carriers):
        self.bridge = bridge
        self.carriers = carriers
        self.completed_transitions = 0

    def apply(self, **kwargs):
        result = self.bridge.apply(self.carriers, **kwargs)
        self.completed_transitions += 1
        return result


def _operators() -> dict[str, str]:
    values = {name: "0" * 64 for name in post.GM1_BASELINE_OPERATOR_KEYS}
    values.update(child_post_final_rk_operator_source_pins())
    return values


def _state(fake_cupy: _FakeCupy, *, ni: int = 2, nj: int = 2, nk: int = 3):
    mass, qshape, zface = (nk, nj, ni), (6, nk, nj, ni), (nk + 1, nj, ni)
    xface, yface = (nk, nj, ni + 1), (nk, nj + 1, ni)
    fields = {
        "q3d": _device(np.full(qshape, 2.0e-5, dtype=np.float32)),
        "qa": _device(np.full(qshape, 8.0e-5, dtype=np.float32)),
        "rho": _device(np.full(mass, 1.1, dtype=np.float32)),
        "prs": _device(np.full(mass, 90000.0, dtype=np.float32)),
        "rr": _device(np.full(mass, 1.0 / 1.1, dtype=np.float32)),
        "rf": _device(np.full(zface, 1.0 / 1.1, dtype=np.float32)),
        "ppi": _device(np.zeros(mass, dtype=np.float32)),
        "pp3d": _device(np.full(mass, 0.001, dtype=np.float32)),
        "ppx": _device(np.zeros(mass, dtype=np.float32)),
        "tha": _device(np.zeros(mass, dtype=np.float32)),
        "th3d": _device(np.full(mass, 0.15, dtype=np.float32)),
        "pi0": _device(np.full(mass, 0.95, dtype=np.float32)),
        "th0": _device(np.full(mass, 300.0, dtype=np.float32)),
        "c1": _device(np.full(zface, 0.5, dtype=np.float32)),
        "c2": _device(np.full(zface, 0.5, dtype=np.float32)),
        "total_geopotential_zf": _device(np.arange(np.prod(zface)).reshape(zface)),
        "u3d": _device(np.arange(np.prod(xface)).reshape(xface) + 1),
        "ua": _device(np.zeros(xface, dtype=np.float32)),
        "v3d": _device(np.arange(np.prod(yface)).reshape(yface) + 2),
        "va": _device(np.zeros(yface, dtype=np.float32)),
        "w3d": _device(np.arange(np.prod(zface)).reshape(zface) + 3),
        "wa": _device(np.zeros(zface, dtype=np.float32)),
        "rmh": _device(np.full(mass, 1.0, dtype=np.float32)),
    }
    for name in EQTSET2_HELD_FIELDS:
        fields[name] = _device(np.zeros(mass, dtype=np.float32))
    arrays = {
        "ppx": _device(np.zeros((nk + 2, nj + 2, ni + 2), dtype=np.float32)),
        "pp3d": _device(np.zeros((nk + 2, nj + 2, ni + 2), dtype=np.float32)),
        "ppi": _device(np.zeros((nk + 2, nj + 2, ni + 2), dtype=np.float32)),
        "rho": _device(np.zeros((nk + 2, nj + 2, ni + 2), dtype=np.float32)),
        "u3d": _device(np.ones((nk + 2, nj + 6, ni + 7), dtype=np.float32)),
        "ua": _device(np.zeros((nk + 2, nj + 6, ni + 7), dtype=np.float32)),
        "v3d": _device(np.full((nk + 2, nj + 7, ni + 6), 2.0, dtype=np.float32)),
        "va": _device(np.zeros((nk + 2, nj + 7, ni + 6), dtype=np.float32)),
        "w3d": _device(np.full((nk + 3, nj + 6, ni + 6), 3.0, dtype=np.float32)),
        "wa": _device(np.zeros((nk + 3, nj + 6, ni + 6), dtype=np.float32)),
        "uh": _device(np.ones((ni + 6,), dtype=np.float32)),
        "vh": _device(np.ones((nj + 6,), dtype=np.float32)),
    }
    pressure = (slice(1, nk + 1), slice(1, nj + 1), slice(1, ni + 1))
    arrays["ppx"][pressure] = np.float32(0.002)
    product = _Product(fields)

    component = _FakeComponent(nx=ni, ny=nj, nz=nk, dt_s=0.5)
    component.sequence = fake_cupy.sequence
    bridge = Eqtset2Wsm6Bridge(
        component=component,
        nx=ni,
        ny=nj,
        nz=nk,
        dt_s=0.5,
        idiss=False,
        efall=False,
        product_admission=False,
    )
    q = fields["q3d"]
    carriers = Eqtset2Wsm6Carriers(
        pi0=fields["pi0"],
        th0=fields["th0"],
        rho=fields["rho"],
        pp3d=fields["pp3d"],
        th3d=fields["th3d"],
        p=fields["prs"],
        alt=fields["rr"],
        phb=fields["total_geopotential_zf"],
        php=_device(np.zeros(zface, dtype=np.float32)),
        qv=q[0],
        qc=q[1],
        qr=q[2],
        qi=q[3],
        qs=q[4],
        qg=q[5],
        qpten=fields["qpten"],
        qtten=fields["qtten"],
        qvten=fields["qvten"],
        qcten=fields["qcten"],
    )
    transition = _Transition(bridge, carriers)

    def density(args):
        if hasattr(fake_cupy, "sequence"):
            fake_cupy.sequence.append("density")
        rho, c1, c2, rr, rf, *_dims = args
        np.reciprocal(rho, out=rr)
        rf[0] = rr[0]
        rf[-1] = rr[-1]
        for k in range(1, nk):
            np.reciprocal(c1[k] * rho[k - 1] + c2[k] * rho[k], out=rf[k])

    no_op = _FakeKernel()
    prepared = SimpleNamespace(
        arrays=arrays,
        metrics={"rmh": fields["rmh"]},
        open_compiled=SimpleNamespace(kernels=(no_op, no_op, no_op, no_op, no_op)),
    )
    compiled = CompiledWsm6Pdefq(
        None, (), b"ptx", b"cubin", "86", (("device_id", "0"),)
    )
    workspace = Wsm6PdefqWorkspace(
        ni, nj, nk, _device(np.zeros((6, nj)), dtype=np.float64)
    )
    state = create_cuda_gm1_child_post_final_rk_state(
        product=product,
        prepared=prepared,
        transition=transition,
        pdefq_asq=_device(np.zeros(6), dtype=np.float64),
        stream=fake_cupy.stream,
        forecast_compiled=SimpleNamespace(kernels=(no_op, _FakeKernel(density))),
        pdefq_compiled=compiled,
        pdefq_workspace=workspace,
        ni=ni,
        nj=nj,
        nk=nk,
        dx=50.0,
        dy=50.0,
        dz=100.0,
        dt=0.5,
    )
    return state


@pytest.fixture
def fake_runtime(monkeypatch):
    cupy = _FakeCupy()
    cupy.sequence = []
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(
        bridge_module.engine_pin, "inspect", lambda: SimpleNamespace(admitted=True)
    )

    def fake_pdefq(arrays, **kwargs):
        cupy.sequence.append("pdefq")
        np.maximum(arrays["q3d"], np.float32(0.0), out=arrays["q3d"])
        arrays["asq"] -= np.float64(0.25)  # signed ledgers are source-valid
        event = _FakeEvent(disable_timing=True)
        event.record(kwargs["stream"])
        return CudaWsm6PdefqLaunch(
            event,
            kwargs["compiled"],
            kwargs["workspace"],
            {"schema": "fake-pdefq/v1", "cadence": "exact"},
        )

    monkeypatch.setattr(post, "launch_cuda_wsm6_pdefq_pdscheme1", fake_pdefq)
    return cupy


def _final_sync(state: CudaGm1ChildPostFinalRkState, *, step_index: int = 0):
    slices = {
        "u3d": [[1, state.nk + 1, 1], [3, state.nj + 3, 1], [3, state.ni + 4, 1]],
        "v3d": [[1, state.nk + 1, 1], [3, state.nj + 4, 1], [3, state.ni + 3, 1]],
        "w3d": [[1, state.nk + 2, 1], [3, state.nj + 3, 1], [3, state.ni + 3, 1]],
        "pp3d": [[1, state.nk + 1, 1], [1, state.nj + 1, 1], [1, state.ni + 1, 1]],
    }
    copies = {}
    for name, source_slices in slices.items():
        source = state.prepared.arrays[name]
        destination = state.product.field(name)
        copies[name] = {
            "source_pointer": post._pointer(source),
            "destination_pointer": post._pointer(destination),
            "source_shape": list(source.shape),
            "source_slices": source_slices,
            "destination_shape": list(destination.shape),
        }
    receipt = {
        "schema": CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_SCHEMA,
        "abi": CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_ABI,
        "step_index": step_index,
        "completed_nrk": 3,
        "stream_pointer": state.stream_pointer,
        "source_order": "after completed solve2 RK3; before remove/EOS/apmasscon",
        "token_scope": "same-stream enqueue and exact device bindings; not a device-content hash",
        "producer_source_sha256": post.CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE_SHA256,
        "copies": copies,
    }
    token = post._canonical_receipt_sha256(receipt)
    receipt["operation_token_sha256"] = token
    event = _FakeEvent(disable_timing=True)
    event.record(state.stream)
    return SimpleNamespace(
        ready_event=event,
        step_index=step_index,
        stream_pointer=state.stream_pointer,
        operation_token_sha256=token,
        receipt=MappingProxyType(receipt),
    )


def _post_final_hooks(state: CudaGm1ChildPostFinalRkState, fake_runtime: _FakeCupy):
    held_before_recapture: dict[str, np.ndarray] = {}

    def apmasscon(step_index: int, stream: object):
        assert step_index == state.next_step_index and stream is state.stream
        fake_runtime.sequence.append("apmasscon")
        event = _FakeEvent(disable_timing=True)
        event.record(stream)
        receipt = {
            "schema": CUDA_APMASSCON_SCHEMA,
            "mass1_f64": 123.0,
            "launch": {"stream_pointer": state.stream_pointer},
        }
        scratch = _device(np.zeros(1), dtype=np.float64)
        return CudaApMassConLaunch(
            event,
            SimpleNamespace(),
            scratch,
            scratch.copy(),
            scratch.copy(),
            _device(np.zeros(1)),
            receipt,
        )

    def rho_boundary(step_index: int, stream: object):
        assert step_index == state.next_step_index and stream is state.stream
        held_before_recapture.update(
            {
                name: np.asarray(state.product.field(name)).copy()
                for name in EQTSET2_HELD_FIELDS
            }
        )
        fake_runtime.sequence.append("rho-boundary")
        physical = state.product.field("rho")
        padded = state.prepared.arrays["rho"]
        padded[1 : state.nk + 1, 1 : state.nj + 1, 1 : state.ni + 1] = physical
        event = _FakeEvent(disable_timing=True)
        event.record(stream)
        receipt = MappingProxyType(
            {
                "abi": post.CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_ABI,
                "physical_source_pointer": post._pointer(physical),
                "physical_source_shape": list(physical.shape),
                "producer_source_sha256": (
                    post.CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE_SHA256
                ),
                "padded_destination_pointer": post._pointer(padded),
                "padded_destination_shape": list(padded.shape),
                "physical_destination_slices": [
                    [1, state.nk + 1, 1],
                    [1, state.nj + 1, 1],
                    [1, state.ni + 1, 1],
                ],
                "schema": post.CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_SCHEMA,
                "source_order": (
                    "solve3.F:309-312 after ppx bcp and before held recapture"
                ),
                "step_index": step_index,
                "stream_pointer": state.stream_pointer,
                "boundary_order": ["west/east", "south/north"],
                "same_stream_no_host_sync": True,
            }
        )
        return SimpleNamespace(
            ready_event=event,
            step_index=step_index,
            stream_pointer=state.stream_pointer,
            receipt=receipt,
        )

    def tke_commit(step_index: int, stream: object):
        assert step_index == state.next_step_index and stream is state.stream
        assert np.array_equal(state.product.field("ua"), state.product.field("u3d"))
        assert np.array_equal(state.product.field("va"), state.product.field("v3d"))
        assert np.array_equal(state.product.field("wa"), state.product.field("w3d"))
        assert any(
            not np.array_equal(state.product.field(name), before)
            for name, before in held_before_recapture.items()
        )
        assert not np.array_equal(state.product.field("qa"), state.product.field("q3d"))
        fake_runtime.sequence.append("tke")
        event = _FakeEvent(disable_timing=True)
        event.record(stream)
        return CudaChildStretchedLaunch(
            event,
            MappingProxyType(
                {
                    "schema": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA,
                    "abi": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI,
                    "operator": "tke-solve3-commit",
                    "completed_nrk": 3,
                    "source_anchor": SGS_TKE_SOURCE_ANCHORS["tke_commit"],
                    "same_stream_no_host_sync": True,
                    "claim_limit": GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT,
                }
            ),
            SimpleNamespace(),
        )

    def closure_refresh(
        step_index: int, stream: object, tke_commit_receipt: MappingProxyType
    ):
        assert step_index == state.next_step_index and stream is state.stream
        fake_runtime.sequence.append("closure-refresh")
        event = _FakeEvent(disable_timing=True)
        event.record(stream)
        names = (
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
        arrays = MappingProxyType(
            {
                name: _device(
                    np.zeros(
                        (
                            (state.nk, state.nj, state.ni)
                            if name == "grdscl"
                            else (state.nk + 1, state.nj, state.ni)
                        ),
                        dtype=np.float32,
                    )
                )
                for name in names
            }
        )
        refresh_index = step_index + 1
        nm_def_receipt = {
            "schema": post.CUDA_GM1_CHILD_STRETCHED_NM_DEF_SCHEMA,
            "status": "ENQUEUED_LIVE_SOLVE1_CALCNM_CALCDEF_REFRESH",
            "refresh_index": refresh_index,
            "stream_pointer": state.stream_pointer,
            "same_stream_no_host_sync": True,
            "domain": {"ni": state.ni, "nj": state.nj, "nk": state.nk},
            "metrics": {
                "grid_sha256": (
                    "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"
                ),
                "proof_sha256": (
                    "277f2b8831f316b2e99a81a903f5ac20ee5d781e2c2b058963539f648f9ec02e"
                ),
                "ztop_f32": 15000.0,
                "reference_dz_f32": 100.0,
            },
        }
        sgs_receipt = {
            "schema": post.CUDA_SGS_TKE_SCHEMA,
            "kernel": {"source_sha256": "3" * 64},
            "launch": {"stream_pointer": state.stream_pointer},
        }
        descriptor = {
            "abi": post.CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_ABI,
            "completed_step_index": step_index + 1,
            "nm_def_receipt": nm_def_receipt,
            "operation_source_sha256": (
                post.CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE_SHA256
            ),
            "persistent_ready": [
                {
                    "name": name,
                    "pointer": post._pointer(value),
                    "shape": list(value.shape),
                }
                for name, value in arrays.items()
            ],
            "refresh_index": refresh_index,
            "same_stream_no_host_sync": True,
            "schema": post.CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SCHEMA,
            "sgs_receipt": sgs_receipt,
            "source_order": [
                "solve3 density publication",
                "cm1.F:3718-3788 live calcnm/calcdef",
                "turb.F:1326-1413 persistent-tkea tkekm coefficients",
                "checkpoint/history publication",
            ],
            "step_index": step_index,
            "stream_pointer": state.stream_pointer,
            "tke_commit_receipt_sha256": post._canonical_receipt_sha256(
                tke_commit_receipt
            ),
            "token_scope": (
                "same-stream completed-step cadence and exact device bindings; "
                "not a device-content hash"
            ),
        }
        token = post._canonical_receipt_sha256(descriptor)
        receipt = MappingProxyType({**descriptor, "operation_token_sha256": token})
        return SimpleNamespace(
            ready_event=event,
            step_index=step_index,
            refresh_index=refresh_index,
            stream_pointer=state.stream_pointer,
            operation_token_sha256=token,
            arrays=arrays,
            receipt=receipt,
        )

    return apmasscon, rho_boundary, tke_commit, closure_refresh


def _launch(
    state: CudaGm1ChildPostFinalRkState,
    fake_runtime: _FakeCupy,
    *,
    step_index: int = 0,
):
    apmasscon, rho_boundary, tke_commit, closure_refresh = _post_final_hooks(
        state, fake_runtime
    )
    return launch_cuda_gm1_child_post_final_rk(
        state,
        cadence=Gm1ChildPostFinalRkCadence(step_index, step_index),
        operator_sources=_operators(),
        final_dynamics_sync=_final_sync(state, step_index=step_index),
        mass1_f64=123.0,
        apmasscon_hook=apmasscon,
        rho_boundary_hook=rho_boundary,
        tke_commit_hook=tke_commit,
        closure_refresh_hook=closure_refresh,
    )


def test_import_and_metadata_are_cuda_free() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    script = (
        "import sys; import cm1core.cuda_gm1_child_post_final_rk as m; "
        "assert 'cupy' not in sys.modules; "
        "assert m.child_post_final_rk_metadata()['model']=='GM1'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    metadata = child_post_final_rk_metadata()
    assert metadata["postconditions"]["asq_nonnegative"] is False
    assert len(child_post_final_rk_metadata_sha256()) == 64


def test_operator_inventory_requires_baseline_and_exact_child_pins() -> None:
    operators = _operators()
    assert operators["closure.tke-evolution"] == "0" * 64
    # The live bytes of cuda_gm1_child_stretched_sgs_tke.py.  This literal was
    # last written at de35847; the module has since moved through reviewed
    # commits 8c7c06c, 7d93095, e4024cc, d4e29a2 and adae362 ("Port CM1
    # r21.1's two-part near-wall SGS models"), and once more when its pin of
    # the merged cuda_gm1_child_stretched_metrics.py bytes was corrected
    # (2525a928... before), and again when that module began reading a
    # binding's sigmaf(nk+1) back once (fc3db3bf... before), and again when its
    # stretched TKE source pin followed weno5's CM1 binary32 weights
    # (27503783... before), and again when it followed VF's CM1 zero-wbar
    # branches without WENO (09b732ec... before), and again when its two
    # compiles moved from cupy.RawModule (which CuPy builds with -ftz=true
    # appended, flushing CM1's binary32 sub-normals) to an NVRTC --ftz=false
    # image (db4f6258... before, cm1core.nvrtc_module, R-020), and again when
    # its metrics-module pin followed the flat column's mh/mf
    # (fixlg/flat-stretched-metric-order, 876f57b3... before); these are the
    # live bytes.  No sealed receipt records this manifest entry.
    assert operators[post.CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY] == (
        "aadcfd5424a554b88c52ceea746f64c48147e2cb1b53468d03b725d473e6b613"
    )
    assert (
        operators["closure.tke-evolution"]
        != operators[post.CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY]
    )
    assert len(require_child_post_final_rk_operator_sources(operators)) == 64
    missing = dict(operators)
    del missing["physics.wsm6-pdefq"]
    with pytest.raises(ValueError, match="omits required"):
        require_child_post_final_rk_operator_sources(missing)
    drift = dict(operators)
    drift[post.CUDA_GM1_CHILD_EQTSET2_WSM6_OPERATOR_KEY] = "1" * 64
    with pytest.raises(ValueError, match="differs"):
        require_child_post_final_rk_operator_sources(drift)
    collision = dict(operators)
    collision[post.CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY] = collision[
        "closure.tke-evolution"
    ]
    with pytest.raises(ValueError, match="differs"):
        require_child_post_final_rk_operator_sources(collision)

    normalized = {name: value for name, value in sorted(operators.items())}
    manifest = {
        "model": "GM1",
        "operators": normalized,
        "schema": "gm1.operator-source-manifest/v1",
    }
    compact = json.dumps(
        manifest, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    seal = require_child_post_final_rk_operator_sources(operators)
    assert seal == hashlib.sha256(compact).hexdigest()
    assert seal != hashlib.sha256(compact + b"\n").hexdigest()


def test_cadence_refuses_early_repeated_and_unsynchronized_calls() -> None:
    good = Gm1ChildPostFinalRkCadence(4, 4)
    require_child_post_final_rk_cadence(good, expected_step_index=4)
    with pytest.raises(CM1Refusal, match="nrk=3"):
        require_child_post_final_rk_cadence(
            Gm1ChildPostFinalRkCadence(4, 4, scalar_completed_nrk=2),
            expected_step_index=4,
        )
    with pytest.raises(CM1Refusal, match="apmasscon"):
        require_child_post_final_rk_cadence(
            Gm1ChildPostFinalRkCadence(4, 3), expected_step_index=4
        )
    with pytest.raises(CM1Refusal, match="stale"):
        require_child_post_final_rk_cadence(good, expected_step_index=5)


def test_compact_same_stream_transition_publishes_solve3_and_allows_signed_asq(
    fake_runtime,
) -> None:
    state = _state(fake_runtime)
    product, arrays = state.product, state.prepared.arrays
    current = {name: product.field(name).copy() for name in ("u3d", "v3d", "w3d")}
    result = _launch(state, fake_runtime)
    assert result.completed_step_index == 1
    assert result.stream_pointer == fake_runtime.stream.ptr
    assert state.next_step_index == 1
    assert state.completed_transitions == 1
    assert not state.poisoned and not state.in_flight
    assert np.all(state.pdefq_asq == np.float64(-0.25))
    assert np.min(product.field("q3d")) >= 0.0
    assert np.array_equal(
        product.field("qa").view(np.uint32), product.field("q3d").view(np.uint32)
    )
    assert np.array_equal(product.field("ua"), current["u3d"])
    assert np.array_equal(product.field("va"), current["v3d"])
    assert np.array_equal(product.field("wa"), current["w3d"])
    assert np.array_equal(arrays["ua"], arrays["u3d"])
    assert np.array_equal(arrays["va"], arrays["v3d"])
    assert np.array_equal(arrays["wa"], arrays["w3d"])
    assert np.array_equal(product.field("ppi"), product.field("pp3d"))
    assert np.array_equal(product.field("tha"), product.field("th3d"))
    assert result.receipt["postconditions"]["asq_finite_signed"] is True
    assert result.receipt["synchronization"]["same_stream"] is True
    assert fake_runtime.sequence == [
        "apmasscon",
        "pdefq",
        "wsm6",
        "rho-boundary",
        "tke",
        "density",
        "closure-refresh",
    ]
    with pytest.raises(CM1Refusal, match="stale"):
        _launch(state, fake_runtime)


def test_negative_post_wsm6_moisture_refuses_and_poisons(fake_runtime) -> None:
    state = _state(fake_runtime)
    original = state.transition.bridge.component.apply

    def negative(carriers, *, refl_10cm_due):
        result = original(carriers, refl_10cm_due=refl_10cm_due)
        carriers.qg.flat[0] = np.float32(-1.0e-7)
        return result

    state.transition.bridge.component.apply = negative
    with pytest.raises(CM1Refusal, match="moisture"):
        _launch(state, fake_runtime)
    assert state.poisoned
    with pytest.raises(CM1Refusal, match="partial transition"):
        _launch(state, fake_runtime)


def test_exact_sync_rho_and_closure_receipts_refuse_drift(fake_runtime) -> None:
    state = _state(fake_runtime)
    sync = _final_sync(state)
    sync_receipt = dict(sync.receipt)
    sync_receipt["operation_token_sha256"] = "f" * 64
    bad_sync = SimpleNamespace(
        ready_event=sync.ready_event,
        step_index=sync.step_index,
        stream_pointer=sync.stream_pointer,
        operation_token_sha256="f" * 64,
        receipt=MappingProxyType(sync_receipt),
    )
    apmasscon, rho_boundary, tke_commit, closure_refresh = _post_final_hooks(
        state, fake_runtime
    )
    with pytest.raises(ValueError, match="not canonical"):
        launch_cuda_gm1_child_post_final_rk(
            state,
            cadence=Gm1ChildPostFinalRkCadence(0, 0),
            operator_sources=_operators(),
            final_dynamics_sync=bad_sync,
            mass1_f64=123.0,
            apmasscon_hook=apmasscon,
            rho_boundary_hook=rho_boundary,
            tke_commit_hook=tke_commit,
            closure_refresh_hook=closure_refresh,
        )
    assert not state.poisoned

    state = _state(fake_runtime)
    apmasscon, rho_boundary, tke_commit, closure_refresh = _post_final_hooks(
        state, fake_runtime
    )

    def bad_rho(step_index: int, stream: object):
        result = rho_boundary(step_index, stream)
        receipt = dict(result.receipt)
        receipt["producer_source_sha256"] = "f" * 64
        return SimpleNamespace(
            ready_event=result.ready_event,
            step_index=result.step_index,
            stream_pointer=result.stream_pointer,
            receipt=MappingProxyType(receipt),
        )

    with pytest.raises(ValueError, match="producer differs"):
        launch_cuda_gm1_child_post_final_rk(
            state,
            cadence=Gm1ChildPostFinalRkCadence(0, 0),
            operator_sources=_operators(),
            final_dynamics_sync=_final_sync(state),
            mass1_f64=123.0,
            apmasscon_hook=apmasscon,
            rho_boundary_hook=bad_rho,
            tke_commit_hook=tke_commit,
            closure_refresh_hook=closure_refresh,
        )
    assert state.poisoned

    fake_runtime.sequence.clear()
    state = _state(fake_runtime)
    apmasscon, rho_boundary, tke_commit, closure_refresh = _post_final_hooks(
        state, fake_runtime
    )

    def legacy_metric_closure(
        step_index: int, stream: object, tke_receipt: MappingProxyType
    ):
        result = closure_refresh(step_index, stream, tke_receipt)
        receipt = dict(result.receipt)
        nm_def = dict(receipt["nm_def_receipt"])
        metrics = dict(nm_def["metrics"])
        metrics["dz_reference_f32"] = metrics.pop("reference_dz_f32")
        nm_def["metrics"] = metrics
        receipt["nm_def_receipt"] = nm_def
        receipt.pop("operation_token_sha256")
        token = post._canonical_receipt_sha256(receipt)
        receipt["operation_token_sha256"] = token
        return SimpleNamespace(
            ready_event=result.ready_event,
            step_index=result.step_index,
            refresh_index=result.refresh_index,
            stream_pointer=result.stream_pointer,
            operation_token_sha256=token,
            arrays=result.arrays,
            receipt=MappingProxyType(receipt),
        )

    with pytest.raises(ValueError, match="vertical identity"):
        launch_cuda_gm1_child_post_final_rk(
            state,
            cadence=Gm1ChildPostFinalRkCadence(0, 0),
            operator_sources=_operators(),
            final_dynamics_sync=_final_sync(state),
            mass1_f64=123.0,
            apmasscon_hook=apmasscon,
            rho_boundary_hook=rho_boundary,
            tke_commit_hook=tke_commit,
            closure_refresh_hook=legacy_metric_closure,
        )
    assert state.poisoned

    fake_runtime.sequence.clear()
    state = _state(fake_runtime)
    apmasscon, rho_boundary, tke_commit, closure_refresh = _post_final_hooks(
        state, fake_runtime
    )

    def bad_closure(step_index: int, stream: object, tke_receipt: MappingProxyType):
        result = closure_refresh(step_index, stream, tke_receipt)
        receipt = dict(result.receipt)
        receipt["operation_token_sha256"] = "f" * 64
        return SimpleNamespace(
            ready_event=result.ready_event,
            step_index=result.step_index,
            refresh_index=result.refresh_index,
            stream_pointer=result.stream_pointer,
            operation_token_sha256="f" * 64,
            arrays=result.arrays,
            receipt=MappingProxyType(receipt),
        )

    with pytest.raises(ValueError, match="not canonical"):
        launch_cuda_gm1_child_post_final_rk(
            state,
            cadence=Gm1ChildPostFinalRkCadence(0, 0),
            operator_sources=_operators(),
            final_dynamics_sync=_final_sync(state),
            mass1_f64=123.0,
            apmasscon_hook=apmasscon,
            rho_boundary_hook=rho_boundary,
            tke_commit_hook=tke_commit,
            closure_refresh_hook=bad_closure,
        )
    assert state.poisoned


def test_creation_rejects_geopotential_rebind(fake_runtime) -> None:
    state = _state(fake_runtime)
    state.transition.carriers = Eqtset2Wsm6Carriers(
        **{
            name: (_device(np.zeros_like(value)) if name == "phb" else value)
            for name, value in (
                (name, getattr(state.transition.carriers, name))
                for name in state.transition.carriers.__dataclass_fields__
            )
        }
    )
    with pytest.raises(ValueError, match="phb"):
        post._validate_state_bindings(state, fake_runtime)


def test_state_type_is_required() -> None:
    with pytest.raises(TypeError, match="CudaGm1ChildPostFinalRkState"):
        launch_cuda_gm1_child_post_final_rk(
            object(),
            cadence=Gm1ChildPostFinalRkCadence(0, 0),
            operator_sources=_operators(),
            final_dynamics_sync=object(),
            mass1_f64=1.0,
            apmasscon_hook=lambda *_args: None,
            rho_boundary_hook=lambda *_args: None,
            tke_commit_hook=lambda *_args: None,
            closure_refresh_hook=lambda *_args: None,
        )
    assert CudaGm1ChildPostFinalRkState.__name__.endswith("State")


def test_real_child_derived_compact_cuda_receipt_is_sealed() -> None:
    raw = REAL_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "70365bb73b4679223d6489c906e5e1a16e49537bf482b35e41c3569eb813c161"
    )
    receipt = json.loads(raw)
    assert receipt["status"] == "PASS_REAL_CHILD_DERIVED_COMPACT_POST_FINAL_RK"
    assert receipt["source"]["real_atmospheric_data"] is True
    assert receipt["source"]["synthetic_translation"] is False
    assert receipt["qa_raw_copy_of_q3d"] is True
    assert receipt["q3d_minimum"] >= 0.0
    assert receipt["qa_minimum"] >= 0.0
    assert receipt["transition"]["postconditions"] == {
        "asq_finite_signed": True,
        "asq_sha256": receipt["asq_f64_sha256"],
        "q3d_nonnegative": True,
        "qa_nonnegative": True,
    }
    historical = receipt["transition"]["operators"]["required_dependency_pins"]
    assert historical[post.CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY] == (
        "48c8f2e56dbf903f1f8a64cfb2b1b02ff4ec804c84f3826b986077e24dfdb613"
    )
    assert historical != dict(child_post_final_rk_operator_source_pins())


def test_historical_v1_receipt_is_preserved_and_cannot_satisfy_v2() -> None:
    raw = HISTORICAL_V1_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "14d2457e5a9369e4275a76a69533d5423c3a013f3c7a98aae7167f5b1162b2f3"
    )
    receipt = json.loads(raw)
    assert receipt["transition"]["schema"] == "gm1.cuda-child-post-final-rk/v1"
    historical = receipt["transition"]["operators"]["required_dependency_pins"]
    assert historical != dict(child_post_final_rk_operator_source_pins())
    assert post.CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_OPERATOR_KEY not in historical
    assert post.CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_OPERATOR_KEY not in historical


def test_operator_source_pins_measure_the_installed_package_not_a_frozen_list() -> None:
    """The composed operator digests come from the installed modules.

    A frozen digest list had gone stale twice against sources committed on this
    branch, and a stale entry did not weaken anything: it refused to build the
    child operator manifest at all, which sealed the whole child runner shut
    behind a bookkeeping mismatch rather than a scientific one.  What must
    survive is the property the guard was written for - every operator source is
    one nonempty regular file inside the installed package.
    """

    import hashlib
    from pathlib import Path as _Path

    import cm1core.cuda_gm1_child_post_final_rk as module

    root = _Path(module.__file__).resolve().parent
    pins = child_post_final_rk_operator_source_pins()
    for name, relative in module._DEPENDENCY_SOURCE_MODULES.items():
        path = root / relative
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size > 0
        assert pins[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    # The producer module measures itself too, so the manifest names the exact
    # transition that ran.
    assert pins[module.CUDA_GM1_CHILD_POST_FINAL_RK_OPERATOR_KEY] == (
        hashlib.sha256(_Path(module.__file__).resolve().read_bytes()).hexdigest()
    )


def test_an_operator_source_outside_the_installed_package_is_refused(
    tmp_path, monkeypatch
) -> None:
    import cm1core.cuda_gm1_child_post_final_rk as module

    foreign = tmp_path / "candidate.py"
    foreign.write_text("# uncommitted shared-worktree candidate\n", encoding="utf-8")
    monkeypatch.setitem(
        module._DEPENDENCY_SOURCE_MODULES,
        "dynamics.apmasscon",
        str(foreign),
    )
    with pytest.raises(RuntimeError) as error:
        child_post_final_rk_operator_source_pins()
    assert "never become production science" in str(error.value)
