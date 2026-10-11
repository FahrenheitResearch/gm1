from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_gm1_solve1_lspgrad as live
from cm1core import cuda_gm1_tilestream_solve1_full_held as full_held
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.large_scale_pressure_gradient import (
    LargeScalePressureGradientSelection,
)


ROOT = Path(__file__).resolve().parents[1]


def _selection(branch: int) -> LargeScalePressureGradientSelection:
    return LargeScalePressureGradientSelection(
        lspgrad=branch,
        fcor=1.0e-4,
        umove=3.0,
        vmove=-2.0,
        hurr_rad_m=30_000.0 if branch == 3 else None,
        hurr_rotate_degrees=25.0 if branch == 3 else None,
    )


def _sources(branch: int) -> dict[str, np.ndarray]:
    profile = np.linspace(-8.0, 12.0, 75, dtype=np.float32)
    if branch == 1:
        return {
            "u0_window": np.zeros((75, 302, 302), dtype=np.float32),
            "v0_window": np.zeros((75, 302, 302), dtype=np.float32),
        }
    if branch == 2:
        return {"ug": profile.copy(), "vg": profile[::-1].copy()}
    if branch == 3:
        return {"vg": profile.copy()}
    return {
        "ulspg": np.linspace(-0.02, 0.01, 75, dtype=np.float32),
        "vlspg": np.linspace(0.03, -0.01, 75, dtype=np.float32),
    }


@pytest.mark.parametrize("branch", [1, 2, 3, 4])
def test_all_active_branches_build_exact_physical_staggered_increments(
    branch: int,
) -> None:
    sources = _sources(branch)
    result = live.build_gm1_solve1_lspgrad_cpu_increments(
        sources, selection=_selection(branch)
    )
    assert result.uten1.shape == (75, 300, 301)
    assert result.vten1.shape == (75, 301, 300)
    assert result.uten1.dtype == result.vten1.dtype == np.dtype("<f4")
    assert result.uten1.flags.c_contiguous
    assert result.vten1.flags.c_contiguous
    receipt = result.receipt
    assert receipt["selection"]["lspgrad"] == branch
    assert receipt["companion_selection"] == {
        "icor": 0,
        "axisymmetric": 0,
        "adaptive_lspgrad": False,
    }
    assert receipt["completed_nrk"] == 0
    assert receipt["adaptive_lspgrad"] is False
    assert receipt["coupled_trajectory_admitted"] is False
    assert receipt["operation_token_sha256"] == live._operation_token(receipt)
    if branch == 1:
        expected_u = np.float32(-2.0) * np.float32(-1.0e-4)
        expected_v = np.float32(3.0) * np.float32(1.0e-4)
        assert np.all(result.uten1 == expected_u)
        assert np.all(result.vten1 == expected_v)
    if branch == 4:
        assert np.array_equal(result.uten1[:, 0, 0], sources["ulspg"])
        assert np.array_equal(result.vten1[:, 0, 0], sources["vlspg"])


def test_active_binding_refuses_noncanonical_carriers_geometry_and_cadence() -> None:
    with pytest.raises(ValueError, match="inventory/order changed"):
        live.build_gm1_solve1_lspgrad_cpu_increments(
            {"vg": _sources(2)["vg"], "ug": _sources(2)["ug"]},
            selection=_selection(2),
        )
    with pytest.raises(ValueError, match="300x300x75"):
        live.build_gm1_solve1_lspgrad_cpu_increments(
            _sources(4), selection=_selection(4), nx=24, ny=24
        )
    with pytest.raises(ValueError, match="exact 50 m"):
        live.build_gm1_solve1_lspgrad_cpu_increments(
            _sources(4), selection=_selection(4), dx=5.0, dy=5.0
        )
    with pytest.raises(ValueError, match="before RK stage 1"):
        live.build_gm1_solve1_lspgrad_cpu_increments(
            _sources(4), selection=_selection(4), completed_nrk=1
        )
    with pytest.raises(ValueError, match="simultaneous Coriolis"):
        live.build_gm1_solve1_lspgrad_cpu_increments(
            _sources(4), selection=_selection(4), icor=1
        )
    with pytest.raises(ValueError, match="axisymmetric=0"):
        live.build_gm1_solve1_lspgrad_cpu_increments(
            _sources(4), selection=_selection(4), axisymmetric=1
        )
    with pytest.raises(ValueError, match="hard-disabled"):
        live.build_gm1_solve1_lspgrad_cpu_increments(
            _sources(4), selection=_selection(4), adaptive_lspgrad=True
        )
    with pytest.raises(ValueError, match="supports only active values"):
        live.build_gm1_solve1_lspgrad_cpu_increments(
            {}, selection=LargeScalePressureGradientSelection(lspgrad=0, fcor=0.0)
        )


class _FakeEvent:
    def __init__(self, *, disable_timing: bool) -> None:
        assert disable_timing is True
        self.recorded_on: _FakeStream | None = None

    def record(self, stream: "_FakeStream") -> None:
        self.recorded_on = stream


class _FakeStream:
    def __init__(self, pointer: int) -> None:
        self.ptr = pointer
        self.waited: list[_FakeEvent] = []

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def wait_event(self, event: _FakeEvent) -> None:
        self.waited.append(event)


class _FakeCupy:
    ndarray = np.ndarray
    float32 = np.float32
    cuda = SimpleNamespace(Event=_FakeEvent)
    add = staticmethod(np.add)

    @staticmethod
    def asarray(value: np.ndarray) -> np.ndarray:
        return np.array(value, dtype=np.float32, copy=True, order="C")


def test_prepared_binding_slices_exact_global_increments_into_one_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = live.importlib.import_module

    def fake_import(name: str):
        return _FakeCupy if name == "cupy" else original_import(name)

    monkeypatch.setattr(live.importlib, "import_module", fake_import)
    stream = _FakeStream(8123)
    binding = live.prepare_cuda_gm1_solve1_lspgrad(
        _sources(4), selection=_selection(4), stream=stream
    )
    plan = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96)
    spec = plan.specs[1]
    arrays = {
        "uten1": np.zeros((75, spec.cny, spec.cnx + 1), dtype=np.float32),
        "vten1": np.zeros((75, spec.cny + 1, spec.cnx), dtype=np.float32),
    }
    receipt = live.enqueue_cuda_gm1_solve1_lspgrad_after_rayleigh(
        binding, arrays, spec=spec, stream=stream
    )
    assert np.array_equal(
        arrays["uten1"],
        binding.uten1[
            :, spec.cj0 : spec.cj0 + spec.cny, spec.ci0 : spec.ci0 + spec.cnx + 1
        ],
    )
    assert np.array_equal(
        arrays["vten1"],
        binding.vten1[
            :, spec.cj0 : spec.cj0 + spec.cny + 1, spec.ci0 : spec.ci0 + spec.cnx
        ],
    )
    assert stream.waited == [binding.ready_event]
    assert receipt["binding_operation_token_sha256"] == binding.receipt[
        "operation_token_sha256"
    ]
    assert receipt["same_stream_after_rayleigh_before_idiss"] is True
    assert receipt["operation_token_sha256"] == live._operation_token(receipt)
    with pytest.raises(ValueError, match="source-owning stream"):
        live.enqueue_cuda_gm1_solve1_lspgrad_after_rayleigh(
            binding, arrays, spec=spec, stream=_FakeStream(8124)
        )


def test_full_held_default_abi_and_active_source_order_are_strict() -> None:
    parameter = inspect.signature(
        full_held.launch_cuda_gm1_tiled_solve1_full_held
    ).parameters["lspgrad_binding"]
    assert parameter.default is None
    assert "lspgrad" not in " ".join(full_held.SOURCE_ORDER).lower()
    assert full_held.SOURCE_ORDER_WITH_ACTIVE_LSPGRAD[3] == full_held.SOURCE_ORDER[3]
    assert "active lspgrad=1..4" in full_held.SOURCE_ORDER_WITH_ACTIVE_LSPGRAD[4]
    assert full_held.SOURCE_ORDER_WITH_ACTIVE_LSPGRAD[5] == full_held.SOURCE_ORDER[4]

    source = (
        ROOT / "src/cm1core/cuda_gm1_tilestream_solve1_full_held.py"
    ).read_text(encoding="utf-8")
    start = source.index("def launch_cuda_gm1_tiled_solve1_full_held")
    positions = [
        source.index("launch_cuda_gm1_child_terrain_rayleigh(", start),
        source.index("enqueue_cuda_gm1_solve1_lspgrad_after_rayleigh(", start),
        source.index("idiss_consumer.apply(", start),
    ]
    assert positions == sorted(positions)


def test_full_held_aggregate_lineage_preserves_disabled_shape_and_refuses_mixes() -> None:
    assert full_held._aggregate_lspgrad_lineage({False}, [], [], []) is None
    selection = {
        "active": {"lspgrad": 2, "fcor_f32_bits": 1},
        "companion": {
            "icor": 0,
            "axisymmetric": 0,
            "adaptive_lspgrad": False,
        },
    }
    active = full_held._aggregate_lspgrad_lineage(
        {True}, ["a" * 64, "a" * 64], ["b" * 64, "c" * 64], [selection, selection]
    )
    assert active is not None
    assert active["lspgrad_binding_operation_token_sha256"] == "a" * 64
    assert active["lspgrad_selection"] == selection
    assert active["lspgrad_applied_after_rayleigh_before_idiss"] is True
    with pytest.raises(ValueError, match="mix active and disabled"):
        full_held._aggregate_lspgrad_lineage(
            {False, True}, ["a" * 64], ["b" * 64], [selection]
        )
    with pytest.raises(ValueError, match="different lspgrad configs"):
        full_held._aggregate_lspgrad_lineage(
            {True}, ["a" * 64, "d" * 64], ["b" * 64, "c" * 64], [selection, selection]
        )


def test_validation_receipt_seals_cpu_only_integration_without_cuda_claim() -> None:
    receipt = json.loads(
        (
            ROOT
            / "evidence/validation/cm1-lspgrad-solve1-source-order-binding-v1/validation-receipt.json"
        ).read_bytes()
    )
    assert receipt["status"] == "PASS_CPU_TYPED_SOURCE_ORDER_BINDING_ONLY"
    assert receipt["result"]["lspgrad_0_default_receipt_shape_unchanged"] is True
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
