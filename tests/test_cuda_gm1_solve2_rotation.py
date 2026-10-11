from __future__ import annotations

import inspect
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cm1core.cuda_gm1_child_stretched_momentum as momentum
import cm1core.cuda_gm1_solve2_rotation as live
import cm1core.cuda_gm1_tilestream_pre_acoustic_momentum_rk as rk1
import cm1core.cuda_gm1_tilestream_rk23_pre_acoustic_momentum as rk23
from cm1core.cuda_gm1_tilestream_rk1_pre_acoustic_lineage import (
    audit_gm1_rk1_pre_acoustic_source_contract,
)
from cm1core.rotation_forcing import (
    RotationForcingSelection,
    cpu_rotation_forcing_inplace,
)


ROOT = Path(__file__).resolve().parents[1]


class _Flags:
    c_contiguous = True


class _DeviceArray:
    def __init__(self, value: object = (), *, shape: tuple[int, ...] | None = None):
        if shape is None:
            self.value = np.ascontiguousarray(value, dtype=np.float32)
            self.shape = self.value.shape
        else:
            self.value = None
            self.shape = shape
        self.dtype = np.dtype(np.float32)
        self.flags = _Flags()


class _Event:
    def __init__(self, *, disable_timing: bool):
        self.disable_timing = disable_timing
        self.recorded_on: _Stream | None = None

    def record(self, stream: _Stream) -> None:
        self.recorded_on = stream


class _Stream:
    def __init__(self, ptr: int = 0xCAFE):
        self.ptr = ptr
        self.waited: list[_Event] = []

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *_: object) -> None:
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
        asarray=lambda value: _DeviceArray(value),
        cuda=SimpleNamespace(Event=_Event),
    )
    monkeypatch.setitem(sys.modules, "cupy", module)
    return module


def _selection(branch: int) -> RotationForcingSelection:
    beta, axisymmetric = {1: (0, 0), 2: (1, 0), 3: (0, 1)}[branch]
    return RotationForcingSelection(
        icor=1,
        beta_plane=beta,
        axisymmetric=axisymmetric,
        fcor_s_inv=8.1e-5,
        u_move_m_s=4.25,
        v_move_m_s=-2.75,
        center_y_m=7_500.0,
    )


def _binding(
    branch: int,
    *,
    nrk: int,
    stream: _Stream,
    kernel: _Kernel,
    ni: int = 300,
    nj: int = 300,
    nk: int = 75,
) -> live.CudaGm1Solve2RotationBinding:
    compiled = live.CompiledGm1Solve2Rotation(
        object(), kernel, live.CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256, "fake"
    )
    yh = np.arange(nj + 2, dtype=np.float32) * np.float32(50.0)
    return live.prepare_cuda_gm1_solve2_rotation(
        yh,
        ni=ni,
        nj=nj,
        nk=nk,
        selection=_selection(branch),
        nrk=nrk,
        stream=stream,
        compiled=compiled,
        allow_bounded_axisymmetric_u_validation=branch == 3,
    )


@pytest.mark.parametrize("branch", [1, 2, 3])
@pytest.mark.parametrize("nrk", [1, 2, 3])
def test_typed_binding_retains_all_source_branches_and_rk_streams(
    fake_cupy: SimpleNamespace, branch: int, nrk: int
) -> None:
    del fake_cupy
    stream = _Stream()
    binding = _binding(branch, nrk=nrk, stream=stream, kernel=_Kernel())
    receipt = binding.receipt
    assert receipt["selection"]["branch"] == {
        1: "cartesian-f-plane",
        2: "cartesian-beta-plane",
        3: "axisymmetric-u",
    }[branch]
    assert receipt["nrk"] == nrk
    assert receipt["stream_pointer"] == stream.ptr
    assert receipt["outer_axisymmetric_coordinate_admitted"] is False
    assert receipt["active_cuda_raw_bit_oracle"] is False
    assert binding.ready_event.recorded_on is stream
    if branch == 2:
        assert np.unique(binding.f2d.value.view(np.uint32)).size > 1
    else:
        assert np.unique(binding.f2d.value.view(np.uint32)).size == 1
    assert live.require_cuda_gm1_solve2_rotation_binding(
        binding, nrk=nrk, stream=stream
    ) is binding


def test_axisymmetric_and_adaptive_combinations_fail_closed(
    fake_cupy: SimpleNamespace,
) -> None:
    del fake_cupy
    stream = _Stream()
    compiled = live.CompiledGm1Solve2Rotation(
        object(), _Kernel(), live.CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256, "fake"
    )
    yh = np.arange(302, dtype=np.float32) * np.float32(50.0)
    with pytest.raises(ValueError, match="bounded u-inner-term"):
        live.prepare_cuda_gm1_solve2_rotation(
            yh, selection=_selection(3), nrk=1, stream=stream, compiled=compiled
        )
    with pytest.raises(ValueError, match="adaptive moving frame"):
        live.prepare_cuda_gm1_solve2_rotation(
            yh,
            selection=_selection(1),
            nrk=1,
            stream=stream,
            compiled=compiled,
            adaptive_move=True,
        )
    # adapt_dt is admitted: solve2.F:633-673 reads no timestep, so the term is
    # identical whatever large step the stage that integrates it was handed.
    admitted = live.prepare_cuda_gm1_solve2_rotation(
        yh,
        selection=_selection(1),
        nrk=1,
        stream=stream,
        compiled=compiled,
        adaptive_dt=True,
    )
    assert admitted.receipt["adaptive_dt"] is True
    with pytest.raises(ValueError, match="budget diagnostics"):
        live.prepare_cuda_gm1_solve2_rotation(
            yh,
            selection=_selection(1),
            nrk=1,
            stream=stream,
            compiled=compiled,
            budget_diagnostics=True,
        )


def test_enqueue_requires_exact_geometry_and_records_typed_source_order(
    fake_cupy: SimpleNamespace,
) -> None:
    del fake_cupy
    stream = _Stream()
    kernel = _Kernel()
    binding = _binding(2, nrk=2, stream=stream, kernel=kernel)
    arrays = {
        "u3d": _DeviceArray(shape=(77, 306, 307)),
        "uten": _DeviceArray(shape=(77, 306, 307)),
        "v3d": _DeviceArray(shape=(77, 307, 306)),
        "vten": _DeviceArray(shape=(77, 307, 306)),
    }
    with pytest.raises(ValueError, match="this binding built f2d for"):
        live.enqueue_cuda_gm1_solve2_rotation_after_divergence(
            binding,
            arrays,
            stream=stream,
            ni=299,
            nj=300,
            nk=75,
            nrk=2,
            ng=3,
        )
    receipt = live.enqueue_cuda_gm1_solve2_rotation_after_divergence(
        binding,
        arrays,
        stream=stream,
        ni=300,
        nj=300,
        nk=75,
        nrk=2,
        ng=3,
    )
    assert receipt["binding_operation_token_sha256"] == binding.receipt[
        "operation_token_sha256"
    ]
    assert receipt["same_stream_after_divx_before_advection"] is True
    assert receipt["active_cuda_raw_bit_oracle"] is False
    assert stream.waited == [binding.ready_event]
    assert len(kernel.calls) == 1
    kernel_args = kernel.calls[0][2]
    assert int(kernel_args[9]) == 1
    assert int(kernel_args[10]) == 0


def test_binding_and_enqueue_accept_any_domain_not_only_the_sealed_child(
    fake_cupy: SimpleNamespace,
) -> None:
    """The rotation kernel is generic; its binding used to refuse every grid.

    Before this, ``prepare`` demanded ``yh_window`` of shape ``(302,)`` and
    ``enqueue`` demanded exactly ``300x300x75`` with ``ng=3``, so no lane on
    any other grid -- every Cartesian idealized run there is -- could reach
    CM1's ``icor=1`` term at all.
    """

    del fake_cupy
    ni, nj, nk, ng = 60, 48, 40, 3
    stream = _Stream()
    kernel = _Kernel()
    binding = _binding(1, nrk=1, stream=stream, kernel=kernel, ni=ni, nj=nj, nk=nk)
    assert tuple(binding.f2d.shape) == (nj + 2, ni + 2)
    assert binding.receipt["domain"] == {"nx": ni, "ny": nj, "nk": nk}
    arrays = {
        "u3d": _DeviceArray(shape=(nk + 2, nj + 2 * ng, ni + 1 + 2 * ng)),
        "uten": _DeviceArray(shape=(nk + 2, nj + 2 * ng, ni + 1 + 2 * ng)),
        "v3d": _DeviceArray(shape=(nk + 2, nj + 1 + 2 * ng, ni + 2 * ng)),
        "vten": _DeviceArray(shape=(nk + 2, nj + 1 + 2 * ng, ni + 2 * ng)),
    }
    receipt = live.enqueue_cuda_gm1_solve2_rotation_after_divergence(
        binding, arrays, stream=stream, ni=ni, nj=nj, nk=nk, nrk=1, ng=ng
    )
    assert receipt["same_stream_after_divx_before_advection"] is True
    assert len(kernel.calls) == 1
    arguments = kernel.calls[0][2]
    assert (int(arguments[5]), int(arguments[6]), int(arguments[7])) == (ni, nj, nk)
    assert int(arguments[8]) == ng


def _fadd(left: np.float32, right: np.float32) -> np.float32:
    return np.float32(np.float32(left) + np.float32(right))


def _fsub(left: np.float32, right: np.float32) -> np.float32:
    return np.float32(np.float32(left) - np.float32(right))


def _fmul(left: np.float32, right: np.float32) -> np.float32:
    return np.float32(np.float32(left) * np.float32(right))


@pytest.mark.parametrize("branch", [1, 2, 3])
def test_padded_kernel_index_map_matches_compact_source_operator_raw_words(
    branch: int,
) -> None:
    ni, nj, nk, ng = 4, 3, 2, 3
    rng = np.random.default_rng(4190 + branch)
    u_window = rng.standard_normal((nk, nj + 2, ni + 2), dtype=np.float32)
    v_window = rng.standard_normal((nk, nj + 2, ni + 2), dtype=np.float32)
    initial_u = rng.standard_normal((nk, nj + 1, ni + 1), dtype=np.float32)
    initial_v = rng.standard_normal((nk, nj + 1, ni + 1), dtype=np.float32)
    yh = np.arange(nj + 2, dtype=np.float32) * np.float32(50.0)
    compact = {
        "u3d_window": u_window.copy(),
        "v3d_window": v_window.copy(),
        "uten": initial_u.copy(),
        "vten": initial_v.copy(),
        "yh_window": yh,
        "f2d": np.empty((nj + 2, ni + 2), dtype=np.float32),
    }
    cpu_rotation_forcing_inplace(
        compact, ni=ni, nj=nj, nk=nk, selection=_selection(branch)
    )
    u = np.zeros((nk + 2, nj + 2 * ng, ni + 1 + 2 * ng), dtype=np.float32)
    v = np.zeros((nk + 2, nj + 1 + 2 * ng, ni + 2 * ng), dtype=np.float32)
    uten = np.zeros_like(u)
    vten = np.zeros_like(v)
    u[1 : nk + 1, ng - 1 : ng + nj + 1, ng : ng + ni + 2] = u_window
    v[1 : nk + 1, ng : ng + nj + 2, ng - 1 : ng + ni + 1] = v_window
    uten[1 : nk + 1, ng : ng + nj + 1, ng : ng + ni + 1] = initial_u
    vten[1 : nk + 1, ng : ng + nj + 1, ng : ng + ni + 1] = initial_v
    fcor = np.float32(_selection(branch).fcor_s_inv)
    umove = np.float32(_selection(branch).u_move_m_s)
    vmove = np.float32(_selection(branch).v_move_m_s)
    f2d = compact["f2d"]
    for k in range(nk):
        z = k + 1
        for cj in range(nj + 1):
            y = cj + ng
            for ci in range(ni + 1):
                x = ci + ng
                if branch == 3:
                    if cj < nj:
                        pair = _fadd(v[z, y, x], v[z, y, x - 1])
                        uten[z, y, x] = _fadd(
                            uten[z, y, x], _fmul(pair, _fmul(fcor, np.float32(0.5)))
                        )
                    continue
                first = _fadd(v[z, y, x], v[z, y + 1, x])
                second = _fadd(v[z, y, x - 1], v[z, y + 1, x - 1])
                first = _fadd(first, second)
                if branch == 1:
                    first = _fmul(_fadd(_fmul(first, np.float32(0.25)), vmove), fcor)
                else:
                    fs = _fadd(f2d[cj + 1, ci + 1], f2d[cj + 1, ci])
                    first = _fmul(first, _fmul(fs, np.float32(0.125)))
                uten[z, y, x] = _fadd(uten[z, y, x], first)
                first = _fadd(u[z, y, x], u[z, y, x + 1])
                second = _fadd(u[z, y - 1, x], u[z, y - 1, x + 1])
                first = _fadd(first, second)
                if branch == 1:
                    first = _fmul(_fadd(_fmul(first, np.float32(0.25)), umove), fcor)
                else:
                    fs = _fadd(f2d[cj + 1, ci + 1], f2d[cj, ci + 1])
                    first = _fmul(first, _fmul(fs, np.float32(0.125)))
                vten[z, y, x] = _fsub(vten[z, y, x], first)
    actual_u = uten[1 : nk + 1, ng : ng + nj + 1, ng : ng + ni + 1]
    actual_v = vten[1 : nk + 1, ng : ng + nj + 1, ng : ng + ni + 1]
    assert np.array_equal(actual_u.view(np.uint32), compact["uten"].view(np.uint32))
    assert np.array_equal(actual_v.view(np.uint32), compact["vten"].view(np.uint32))


def test_generic_momentum_places_hook_strictly_between_divergence_and_advu() -> None:
    source = inspect.getsource(momentum.launch_cuda_gm1_child_stretched_momentum5)
    divergence = source.index("divergence(")
    hook = source.index("post_divergence_hook(", divergence)
    advu = source.index("advu(", hook)
    assert divergence < hook < advu
    assert "if post_divergence_receipt is not None:" in source


def test_icor0_receipts_and_real_idealized_restored_defaults_remain_unchanged() -> None:
    audit = audit_gm1_rk1_pre_acoustic_source_contract()
    assert audit["selected_profile"]["icor"] == 0
    assert "rotation_selection" not in audit
    assert "rotation_forcing" not in audit["operator_source_sha256"]
    assert inspect.signature(
        rk1.prepare_cuda_gm1_tilestream_pre_acoustic_momentum_rk
    ).parameters["rotation_binding"].default is None
    assert inspect.signature(
        rk23.launch_cuda_gm1_tilestream_rk23_pre_acoustic_momentum
    ).parameters["rotation_binding"].default is None
    assert inspect.signature(
        rk23.execute_cuda_gm1_tilestream_rk23_acoustic_stage
    ).parameters["rotation_binding"].default is None
    for path in (
        "tools/run_real_gm1_tilestream_solve1_full_held.py",
        "tools/run_real_gm1_idealized_tilestream_rk2_rk3.py",
        "tools/run_live_gm1_restored_tilestream_rk1.py",
    ):
        assert "rotation_binding=" not in (ROOT / path).read_text(encoding="utf-8")


def test_active_rotation_refuses_simultaneous_lspgrad_lineage() -> None:
    for module in (rk1, rk23):
        source = inspect.getsource(module)
        assert '"lspgrad_binding_operation_token_sha256"' in source
        assert "does not admit simultaneous live lspgrad forcing" in source


def test_active_rk1_source_audit_binds_rotation_without_cuda_claim() -> None:
    selection = {
        "icor": 1,
        "beta_plane": 1,
        "axisymmetric": 0,
        "branch": "cartesian-beta-plane",
    }
    audit = audit_gm1_rk1_pre_acoustic_source_contract(selection)
    assert audit["selected_profile"]["icor"] == 1
    assert audit["selected_profile"]["betaplane"] == 1
    assert audit["rotation_selection"] == selection
    assert audit["rotation_after_divx_before_momentum_advection"] is True
    assert audit["active_rotation_cuda_raw_bit_oracle"] is False
    assert audit["operator_source_sha256"]["rotation_forcing"] == (
        live.CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256
    )
    bounded = audit_gm1_rk1_pre_acoustic_source_contract(
        {**selection, "beta_plane": 0, "axisymmetric": 1}
    )
    assert bounded["selected_profile"]["axisymmetric"] == 0
    assert bounded["selected_profile"]["rotation_axisymmetric_u_inner_term"] == 1
    assert bounded["selected_profile"]["outer_axisymmetric_coordinate_admitted"] is False


def test_rk23_executor_forwards_exact_optional_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    seen: dict[str, object] = {}

    def launch(*args: object, **kwargs: object) -> object:
        del args
        seen.update(kwargs)
        return "pre-launch"

    monkeypatch.setattr(rk23, "launch_cuda_gm1_tilestream_rk23_pre_acoustic_momentum", launch)
    monkeypatch.setattr(
        rk23, "finish_cuda_gm1_tilestream_rk23_pre_acoustic_momentum", lambda value: "pre-result"
    )
    monkeypatch.setattr(
        rk23, "launch_cuda_gm1_tilestream_rk23_acoustic", lambda *args, **kwargs: "acoustic-launch"
    )
    monkeypatch.setattr(
        rk23, "finish_cuda_gm1_tilestream_rk23_acoustic", lambda *args, **kwargs: "done"
    )
    result = rk23.execute_cuda_gm1_tilestream_rk23_acoustic_stage(
        object(),
        object(),
        object(),
        {},
        {},
        {},
        object(),
        object(),
        object(),
        requested_nrk=2,
        th0=object(),
        qv0=object(),
        pi0=object(),
        th3d=object(),
        q3d=object(),
        stream=object(),
        rotation_binding=marker,  # type: ignore[arg-type]
    )
    assert result == "done"
    assert seen["rotation_binding"] is marker


def test_matrix_records_only_cpu_tested_typed_solve2_binding() -> None:
    matrix = json.loads(
        (ROOT / "src/cm1core/records/CM1-DYNAMICS-BRANCH-MATRIX.json").read_bytes()
    )
    profile = matrix["admitted_profiles"][
        "rotation-forcing-all-inner-branches-cpu-operator"
    ]
    typed = profile["typed_solve2_binding"]
    assert profile["admission"] == "operator-only"
    assert typed["admission"] == "cpu-tested-source-order-binding-only"
    assert typed["icor0_receipt_shape_unchanged"] is True
    assert typed["active_cuda_raw_bit_oracle"] is False
    assert typed["coupled_trajectory_admitted"] is False
    assert typed["requirements"]["nrk"] == [1, 2, 3]
    assert typed["requirements"]["axisymmetric_outer_coordinate"] is False
    assert typed["requirements"]["simultaneous_lspgrad"] is False


def test_validation_receipt_seals_cpu_only_source_order_binding() -> None:
    receipt = json.loads(
        (
            ROOT
            / "evidence/validation/cm1-rotation-solve2-source-order-binding-v1/validation-receipt.json"
        ).read_bytes()
    )
    assert receipt["status"] == "PASS_CPU_TYPED_SOURCE_ORDER_BINDING_ONLY"
    assert receipt["result"]["icor0_default_receipt_shape_unchanged"] is True
    assert receipt["result"]["active_cuda_raw_bit_oracle"] is False
    assert receipt["result"]["active_coupled_trajectory_admitted"] is False
    execution = receipt["execution"]
    assert execution["cpu_only"] is True and len(execution) == 7
    assert not any(value for key, value in execution.items() if key != "cpu_only")
    for artifact in receipt["artifacts"].values():
        path = ROOT / artifact["path"]
        raw = path.read_bytes()
        assert len(raw) == artifact["bytes"]
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
