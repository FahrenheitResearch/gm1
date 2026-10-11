from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from measured_source import measured_identity

from cm1core import cuda_gm1_solve1_idealized_radiation as live
from cm1core import cuda_gm1_tilestream_solve1_full_held as full_held
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.idealized_radiation import (
    IdealizedRadiationSelection,
    cpu_idealized_radiation_inplace,
    idealized_radiation_shapes,
)


ROOT = Path(__file__).resolve().parents[1]


def _selection(**changes: object) -> IdealizedRadiationSelection:
    values: dict[str, object] = {
        "rterm": 1,
        "radopt": 0,
        "eqtset": 2,
        "imoist": 1,
        "iice": 1,
        "nqv": 1,
        "nql1": 2,
        "nql2": 3,
        "nqs1": 4,
        "nqs2": 6,
        "axisymmetric": 0,
        "diagnostic": False,
        "mpi": False,
        "dt_seconds": 0.25,
        "dx_m": 50.0,
        "dy_m": 50.0,
        "dz_m": 200.0,
    }
    values.update(changes)
    return IdealizedRadiationSelection(**values)  # type: ignore[arg-type]


def _sources(nx: int = 4, ny: int = 3, nk: int = 2) -> dict[str, np.ndarray]:
    shape = (nk, ny, nx)
    tha = np.linspace(-2.0, 2.0, np.prod(shape), dtype=np.float32).reshape(shape)
    qa = np.empty((6, nk, ny, nx), dtype=np.float32)
    for species in range(6):
        qa[species].fill(np.float32((species + 1) * 1.0e-4))
    return {
        "tha": np.ascontiguousarray(tha),
        "pi0": np.full(shape, np.float32(1.0), dtype=np.float32),
        "ppi": np.full(shape, np.float32(0.01), dtype=np.float32),
        "th0": np.full(shape, np.float32(300.0), dtype=np.float32),
        "rho": np.full(shape, np.float32(1.1), dtype=np.float32),
        "ruh": np.linspace(0.9, 1.1, nx, dtype=np.float32),
        "rvh": np.linspace(0.8, 1.2, ny, dtype=np.float32),
        "rmh": np.full(shape, np.float32(1.0), dtype=np.float32),
        "xf": np.arange(nx + 2, dtype=np.float32) * np.float32(50.0),
        "qa": qa,
    }


def _direct(sources: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    nx, ny, nk = 4, 3, 2
    shapes = idealized_radiation_shapes(nx, ny, nk, 6)
    work: dict[str, np.ndarray] = {name: value for name, value in sources.items()}
    for name in ("thten1", "cvm", "mass_weight", "thrad", "prad", "tdiag"):
        work[name] = np.zeros(shapes[name], dtype=np.float32)
    work["bud"] = np.zeros(shapes["bud"], dtype=np.float64)
    work["qbudget"] = np.zeros(shapes["qbudget"], dtype=np.float64)
    cpu_idealized_radiation_inplace(
        work, ni=nx, nj=ny, nk=nk, numq=6, selection=_selection()
    )
    return work


def test_compact_cpu_projection_is_raw_word_exact_to_frozen_operator() -> None:
    sources = _sources()
    expected = _direct(sources)
    result = live.build_gm1_solve1_idealized_radiation_cpu_increment(
        sources,
        selection=_selection(),
        nx=4,
        ny=3,
        nk=2,
        allow_compact_validation=True,
    )
    assert np.array_equal(result.thrad.view(np.uint32), expected["thrad"].view(np.uint32))
    assert np.array_equal(result.prad.view(np.uint32), expected["prad"].view(np.uint32))
    assert np.array_equal(result.qbudget_delta, expected["qbudget"])
    cap = np.float32(np.float32(1.0) / np.float32(12.0 * 3600.0))
    assert result.thrad.flat[0] == cap
    assert result.thrad.flat[-1] == -cap
    receipt = result.receipt
    assert receipt["selection"]["rterm"] == 1
    assert receipt["companion_selection"] == {
        "lspgrad": 0,
        "icor": 0,
        "adaptive_dt": False,
        "axisymmetric_outer_coordinate": False,
    }
    assert receipt["pressure_tendency_coupled"] is False
    assert receipt["energy_budget_coupled"] is False
    assert receipt["active_cuda_raw_bit_oracle"] is False
    assert receipt["coupled_trajectory_admitted"] is False
    assert receipt["operation_token_sha256"] == live._operation_token(receipt)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rterm": 0}, "exact Cartesian"),
        ({"radopt": 1}, "exact Cartesian"),
        ({"eqtset": 1}, "exact Cartesian"),
        ({"imoist": 0}, "exact Cartesian"),
        ({"iice": 0}, "exact Cartesian"),
        ({"axisymmetric": 1}, "exact Cartesian"),
        ({"diagnostic": True}, "exact Cartesian"),
        ({"mpi": True}, "exact Cartesian"),
    ],
)
def test_live_projection_refuses_operator_branches_not_yet_coupled(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        live.build_gm1_solve1_idealized_radiation_cpu_increment(
            _sources(),
            selection=_selection(**changes),
            nx=4,
            ny=3,
            nk=2,
            allow_compact_validation=True,
        )


def test_live_projection_refuses_geometry_cadence_inventory_and_combinations() -> None:
    sources = _sources()
    with pytest.raises(ValueError, match="300x300x75"):
        live.build_gm1_solve1_idealized_radiation_cpu_increment(
            sources, selection=_selection(), nx=4, ny=3, nk=2
        )
    with pytest.raises(ValueError, match="before RK stage 1"):
        live.build_gm1_solve1_idealized_radiation_cpu_increment(
            sources,
            selection=_selection(),
            nx=4,
            ny=3,
            nk=2,
            completed_nrk=1,
            allow_compact_validation=True,
        )
    with pytest.raises(ValueError, match="simultaneous lspgrad"):
        live.build_gm1_solve1_idealized_radiation_cpu_increment(
            sources,
            selection=_selection(),
            nx=4,
            ny=3,
            nk=2,
            lspgrad=1,
            allow_compact_validation=True,
        )
    with pytest.raises(ValueError, match="simultaneous Coriolis"):
        live.build_gm1_solve1_idealized_radiation_cpu_increment(
            sources,
            selection=_selection(),
            nx=4,
            ny=3,
            nk=2,
            icor=1,
            allow_compact_validation=True,
        )
    with pytest.raises(ValueError, match="adaptive dt"):
        live.build_gm1_solve1_idealized_radiation_cpu_increment(
            sources,
            selection=_selection(),
            nx=4,
            ny=3,
            nk=2,
            adaptive_dt=True,
            allow_compact_validation=True,
        )
    with pytest.raises(ValueError, match="outer coordinate"):
        live.build_gm1_solve1_idealized_radiation_cpu_increment(
            sources,
            selection=_selection(),
            nx=4,
            ny=3,
            nk=2,
            axisymmetric_outer_coordinate=True,
            allow_compact_validation=True,
        )
    wrong_order = {name: sources[name] for name in reversed(sources)}
    with pytest.raises(ValueError, match="inventory/order changed"):
        live.build_gm1_solve1_idealized_radiation_cpu_increment(
            wrong_order,
            selection=_selection(),
            nx=4,
            ny=3,
            nk=2,
            allow_compact_validation=True,
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


def _fake_cpu_increment() -> live.Gm1Solve1IdealizedRadiationCpuIncrement:
    thrad = np.empty((75, 300, 300), dtype=np.float32)
    for k in range(75):
        thrad[k].fill(np.float32((k - 37) * 1.0e-8))
    receipt: dict[str, object] = {
        "schema": live.CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_BINDING_SCHEMA,
        "status": "PREPARED_CPU_EXACT_SOLVE1_IDEALIZED_RADIATION_INCREMENT",
        "completed_nrk": 0,
        "domain": {"nx": 300, "ny": 300, "nk": 75, "numq": 6},
        "selection": {
            "rterm": 1,
            "radopt": 0,
            "eqtset": 2,
            "imoist": 1,
            "iice": 1,
            "categories": {"nqv": 1, "nql1": 2, "nql2": 3, "nqs1": 4, "nqs2": 6},
            "axisymmetric": 0,
            "diagnostic": False,
            "mpi": False,
            "dt_f32_bits": int(np.asarray(np.float32(0.25)).view(np.uint32)),
            "dx_f32_bits": int(np.asarray(np.float32(50.0)).view(np.uint32)),
            "dy_f32_bits": int(np.asarray(np.float32(50.0)).view(np.uint32)),
            "dz_f32_bits": int(np.asarray(np.float32(200.0)).view(np.uint32)),
        },
        "companion_selection": {
            "lspgrad": 0,
            "icor": 0,
            "adaptive_dt": False,
            "axisymmetric_outer_coordinate": False,
        },
        "cpu_operator_operation_token_sha256": "a" * 64,
    }
    receipt["operation_token_sha256"] = live._operation_token(receipt)
    return live.Gm1Solve1IdealizedRadiationCpuIncrement(
        thrad=thrad,
        prad=np.zeros_like(thrad),
        qbudget_delta=np.zeros(10, dtype=np.float64),
        receipt=MappingProxyType(receipt),
    )


def test_prepared_binding_slices_exact_thrad_into_one_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = live.importlib.import_module

    def fake_import(name: str):
        return _FakeCupy if name == "cupy" else original_import(name)

    monkeypatch.setattr(live.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        live,
        "build_gm1_solve1_idealized_radiation_cpu_increment",
        lambda *_args, **_kwargs: _fake_cpu_increment(),
    )
    stream = _FakeStream(9123)
    binding = live.prepare_cuda_gm1_solve1_idealized_radiation(
        {}, selection=_selection(), stream=stream
    )
    plan = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96)
    spec = plan.specs[5]
    arrays = {
        "thten1": np.full(
            (75, spec.cny, spec.cnx), np.float32(2.0e-4), dtype=np.float32
        )
    }
    before = arrays["thten1"].copy()
    receipt = live.enqueue_cuda_gm1_solve1_idealized_radiation_after_idiss(
        binding,
        arrays,
        spec=spec,
        stream=stream,
        dt=0.25,
        dx=50.0,
        dy=50.0,
        dz=200.0,
    )
    expected = before + binding.thrad[
        :, spec.cj0 : spec.cj0 + spec.cny, spec.ci0 : spec.ci0 + spec.cnx
    ]
    assert np.array_equal(arrays["thten1"].view(np.uint32), expected.view(np.uint32))
    assert stream.waited == [binding.ready_event]
    assert receipt["same_stream_after_idiss_before_sgs"] is True
    assert receipt["pressure_tendency_coupled"] is False
    assert receipt["energy_budget_coupled"] is False
    assert receipt["operation_token_sha256"] == live._operation_token(receipt)
    with pytest.raises(ValueError, match="source-owning stream"):
        live.enqueue_cuda_gm1_solve1_idealized_radiation_after_idiss(
            binding,
            arrays,
            spec=spec,
            stream=_FakeStream(9124),
            dt=0.25,
            dx=50.0,
            dy=50.0,
            dz=200.0,
        )
    with pytest.raises(ValueError, match="geometry or timestep changed"):
        live.enqueue_cuda_gm1_solve1_idealized_radiation_after_idiss(
            binding,
            arrays,
            spec=spec,
            stream=stream,
            dt=0.5,
            dx=50.0,
            dy=50.0,
            dz=200.0,
        )


def test_full_held_default_abi_and_active_source_order_are_strict() -> None:
    parameter = inspect.signature(
        full_held.launch_cuda_gm1_tiled_solve1_full_held
    ).parameters["idealized_radiation_binding"]
    assert parameter.default is None
    assert "rterm" not in " ".join(full_held.SOURCE_ORDER).lower()
    order = full_held.SOURCE_ORDER_WITH_ACTIVE_IDEALIZED_RADIATION
    assert order[4] == full_held.SOURCE_ORDER[4]
    assert "active rterm=1" in order[5]
    assert order[6] == full_held.SOURCE_ORDER[5]

    source = (
        ROOT / "src/cm1core/cuda_gm1_tilestream_solve1_full_held.py"
    ).read_text(encoding="utf-8")
    start = source.index("def launch_cuda_gm1_tiled_solve1_full_held")
    positions = [
        source.index("idiss_consumer.apply(", start),
        source.index(
            "enqueue_cuda_gm1_solve1_idealized_radiation_after_idiss(", start
        ),
        source.index("launch_cuda_gm1_child_stretched_sgs_stress(", start),
        source.index("launch_cuda_gm1_child_stretched_scalar_turbulence(", start),
    ]
    assert positions == sorted(positions)


def test_full_held_radiation_lineage_preserves_default_and_refuses_mixes() -> None:
    assert (
        full_held._aggregate_idealized_radiation_lineage(
            {False}, [], [], [], []
        )
        is None
    )
    selection = {
        "active": {
            "rterm": 1,
            "radopt": 0,
            "eqtset": 2,
            "imoist": 1,
            "iice": 1,
            "axisymmetric": 0,
            "diagnostic": False,
            "mpi": False,
        },
        "companion": {
            "lspgrad": 0,
            "icor": 0,
            "adaptive_dt": False,
            "axisymmetric_outer_coordinate": False,
        },
    }
    active = full_held._aggregate_idealized_radiation_lineage(
        {True},
        ["a" * 64, "a" * 64],
        ["b" * 64, "c" * 64],
        ["d" * 64, "d" * 64],
        [selection, selection],
    )
    assert active is not None
    assert active["idealized_radiation_binding_operation_token_sha256"] == "a" * 64
    assert active["idealized_radiation_selection"] == selection
    assert active["idealized_radiation_applied_after_idiss_before_sgs"] is True
    assert active["idealized_radiation_active_cuda_raw_bit_oracle"] is False
    with pytest.raises(ValueError, match="mix active and disabled"):
        full_held._aggregate_idealized_radiation_lineage(
            {False, True}, ["a" * 64], ["b" * 64], ["d" * 64], [selection]
        )
    with pytest.raises(ValueError, match="different idealized-radiation configs"):
        full_held._aggregate_idealized_radiation_lineage(
            {True},
            ["a" * 64, "e" * 64],
            ["b" * 64, "c" * 64],
            ["d" * 64, "d" * 64],
            [selection, selection],
        )


def test_validation_receipt_seals_cpu_only_integration_without_cuda_claim() -> None:
    receipt = json.loads(
        (
            ROOT
            / "evidence/validation/cm1-idealized-radiation-solve1-source-order-binding-v1/validation-receipt.json"
        ).read_bytes()
    )
    assert receipt["status"] == "PASS_CPU_TYPED_SOURCE_ORDER_BINDING_ONLY"
    assert receipt["result"]["rterm_0_default_receipt_shape_unchanged"] is True
    assert receipt["result"]["active_cuda_raw_bit_oracle"] is False
    assert receipt["result"]["active_coupled_trajectory_admitted"] is False
    execution = receipt["execution"]
    assert execution["cpu_only"] is True and len(execution) == 7
    assert not any(value for key, value in execution.items() if key != "cpu_only")
    for artifact in receipt["artifacts"].values():
        path = ROOT / artifact["path"]
        # measured_identity: this file is one of the artifacts, reworded since it was sealed.
        size, digest = measured_identity(path)
        assert size == artifact["bytes"]
        assert digest == artifact["sha256"]
