from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_gm1_solve1_rayleigh_variants as live
from cm1core import cuda_gm1_tilestream_solve1_full_held as full_held
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.rayleigh_damping import (
    RayleighDampingSelection,
    cpu_rayleigh_damping_inplace,
    rayleigh_damping_shapes,
)


ROOT = Path(__file__).resolve().parents[1]


def _selection(irdamp: int, hrdamp: int, **changes: object) -> RayleighDampingSelection:
    values: dict[str, object] = {
        "upper_damping": irdamp,
        "horizontal_damping": hrdamp,
        "rdalpha_s_inv": 0.003,
        "zd_m": 12000.0,
        "maxz_m": 15000.0,
        "xhd_m": 50.0,
        "minx_m": 0.0,
        "maxx_m": 200.0,
        "miny_m": 0.0,
        "maxy_m": 150.0,
        "dx_m": 50.0,
        "dy_m": 50.0,
        "terrain_flag": False,
        "axisymmetric": 0,
    }
    values.update(changes)
    return RayleighDampingSelection(**values)  # type: ignore[arg-type]


def _sources(nx: int = 4, ny: int = 3, nk: int = 2) -> dict[str, np.ndarray]:
    shapes = rayleigh_damping_shapes(nx, ny, nk)
    state = shapes["ua"]
    halo = shapes["zh"]
    zh = np.empty(halo, dtype=np.float32)
    zh[0].fill(np.float32(5000.0))
    zh[1].fill(np.float32(13500.0))
    zf = np.empty(shapes["zf"], dtype=np.float32)
    zf[0].fill(np.float32(0.0))
    zf[1].fill(np.float32(10000.0))
    zf[2].fill(np.float32(15000.0))
    pattern = np.linspace(-2.0, 3.0, np.prod(state), dtype=np.float32).reshape(state)
    return {
        "zh": zh,
        "zf": zf,
        "xh": np.arange(-25.0, 275.0, 50.0, dtype=np.float32),
        "yh": np.arange(-25.0, 225.0, 50.0, dtype=np.float32),
        "ua": np.ascontiguousarray(pattern),
        "va": np.ascontiguousarray(pattern * np.float32(-0.5)),
        "wa": np.ascontiguousarray(pattern * np.float32(0.25)),
        "tha": np.ascontiguousarray(pattern * np.float32(0.1)),
        "u0": np.full(state, np.float32(0.5), dtype=np.float32),
        "v0": np.full(state, np.float32(-0.25), dtype=np.float32),
        "th0": np.full(state, np.float32(300.0), dtype=np.float32),
        "ruh": np.linspace(0.9, 1.1, nx, dtype=np.float32),
        "ruf": np.linspace(0.8, 1.2, nx, dtype=np.float32),
        "rvh": np.linspace(0.85, 1.15, ny, dtype=np.float32),
        "rvf": np.linspace(0.75, 1.25, ny, dtype=np.float32),
    }


def _direct(
    sources: dict[str, np.ndarray], selection: RayleighDampingSelection
) -> dict[str, np.ndarray]:
    nx, ny, nk = 4, 3, 2
    shapes = rayleigh_damping_shapes(nx, ny, nk)
    work = {name: value for name, value in sources.items()}
    for name in ("tauh", "taus", "tauf", "uten1", "vten1", "wten1", "thten1", "uavg", "vavg", "thavg"):
        work[name] = np.zeros(shapes[name], dtype=np.float32)
    work["cavg"] = np.zeros(shapes["cavg"], dtype=np.float64)
    cpu_rayleigh_damping_inplace(work, ni=nx, nj=ny, nk=nk, selection=selection)
    return work


@pytest.mark.parametrize("irdamp,hrdamp", [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1)])
def test_all_active_flat_variants_project_exact_physical_staggering(
    irdamp: int, hrdamp: int
) -> None:
    sources = _sources()
    selection = _selection(irdamp, hrdamp)
    expected = _direct(sources, selection)
    result = live.build_gm1_solve1_rayleigh_variant_cpu_held(
        sources,
        selection=selection,
        nx=4,
        ny=3,
        nk=2,
        allow_compact_validation=True,
    )
    projected = {
        "uten1": expected["uten1"][:, :3, :],
        "vten1": expected["vten1"][:, :, :4],
        "wten1": expected["wten1"][:, :3, :4],
        "thten1": expected["thten1"][:, :3, :4],
    }
    for name, value in projected.items():
        actual = getattr(result, name)
        assert actual.flags.c_contiguous
        assert np.array_equal(actual.view(np.uint32), value.view(np.uint32)), name
    assert result.receipt["selection"]["irdamp"] == irdamp
    assert result.receipt["selection"]["hrdamp_input"] == hrdamp
    assert result.receipt["active_cuda_raw_bit_oracle"] is False
    assert result.receipt["coupled_trajectory_admitted"] is False
    assert result.receipt["operation_token_sha256"] == live._operation_token(
        result.receipt
    )


def test_variant_refuses_disabled_foreign_geometry_and_simultaneous_branches() -> None:
    sources = _sources()
    common = {"nx": 4, "ny": 3, "nk": 2, "allow_compact_validation": True}
    with pytest.raises(ValueError, match="requires an active"):
        live.build_gm1_solve1_rayleigh_variant_cpu_held(
            sources, selection=_selection(0, 0), **common
        )
    with pytest.raises(ValueError, match="flat Cartesian"):
        live.build_gm1_solve1_rayleigh_variant_cpu_held(
            sources, selection=_selection(1, 0, terrain_flag=True), **common
        )
    with pytest.raises(ValueError, match="300x300x75"):
        live.build_gm1_solve1_rayleigh_variant_cpu_held(
            sources, selection=_selection(1, 0), nx=4, ny=3, nk=2
        )
    with pytest.raises(ValueError, match="before RK stage 1"):
        live.build_gm1_solve1_rayleigh_variant_cpu_held(
            sources, selection=_selection(1, 0), completed_nrk=1, **common
        )
    for name, kwargs in (
        ("lspgrad/rterm", {"lspgrad": 1}),
        ("lspgrad/rterm", {"rterm": 1}),
        ("serial fixed-dt", {"mpi": True}),
        ("serial fixed-dt", {"diagnostics": True}),
        ("serial fixed-dt", {"adaptive_dt": True}),
    ):
        with pytest.raises(ValueError, match=name):
            live.build_gm1_solve1_rayleigh_variant_cpu_held(
                sources, selection=_selection(1, 0), **common, **kwargs
            )
    wrong = {name: sources[name] for name in reversed(sources)}
    with pytest.raises(ValueError, match="inventory/order changed"):
        live.build_gm1_solve1_rayleigh_variant_cpu_held(
            wrong, selection=_selection(1, 0), **common
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
    copyto = staticmethod(np.copyto)

    @staticmethod
    def asarray(value: np.ndarray) -> np.ndarray:
        return np.array(value, dtype=np.float32, copy=True, order="C")


def _fake_cpu() -> live.Gm1Solve1RayleighVariantCpuHeld:
    shapes = {
        "uten1": (75, 300, 301),
        "vten1": (75, 301, 300),
        "wten1": (75, 300, 300),
        "thten1": (75, 300, 300),
    }
    arrays = {
        name: np.full(shape, np.float32(index * 1.0e-5), dtype=np.float32)
        for index, (name, shape) in enumerate(shapes.items(), start=1)
    }
    receipt: dict[str, object] = {
        "schema": live.CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_BINDING_SCHEMA,
        "status": "PREPARED_CPU_EXACT_FLAT_SOLVE1_RAYLEIGH_VARIANT",
        "completed_nrk": 0,
        "domain": {"nx": 300, "ny": 300, "nk": 75, "dx_m": 50.0, "dy_m": 50.0},
        "selection": {
            "irdamp": 2,
            "hrdamp_input": 1,
            "hrdamp_effective": 2,
            "tendency_reference": "horizontal-average",
            "terrain_flag": False,
            "axisymmetric": 0,
        },
        "companion_selection": {
            "lspgrad": 0,
            "rterm": 0,
            "mpi": False,
            "diagnostics": False,
            "adaptive_dt": False,
        },
        "cpu_operator_operation_token_sha256": "a" * 64,
    }
    receipt["operation_token_sha256"] = live._operation_token(receipt)
    return live.Gm1Solve1RayleighVariantCpuHeld(
        **arrays, receipt=MappingProxyType(receipt)
    )


def test_prepared_binding_overwrites_exact_global_values_on_one_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = live.importlib.import_module

    def fake_import(name: str):
        return _FakeCupy if name == "cupy" else original_import(name)

    monkeypatch.setattr(live.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        live,
        "build_gm1_solve1_rayleigh_variant_cpu_held",
        lambda *_args, **_kwargs: _fake_cpu(),
    )
    stream = _FakeStream(4412)
    binding = live.prepare_cuda_gm1_solve1_rayleigh_variant(
        {}, selection=_selection(2, 1), stream=stream
    )
    spec = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96).specs[6]
    arrays = {
        "uten1": np.zeros((75, spec.cny, spec.cnx + 1), dtype=np.float32),
        "vten1": np.zeros((75, spec.cny + 1, spec.cnx), dtype=np.float32),
        "wten1": np.zeros((75, spec.cny, spec.cnx), dtype=np.float32),
        "thten1": np.zeros((75, spec.cny, spec.cnx), dtype=np.float32),
    }
    receipt = live.enqueue_cuda_gm1_solve1_rayleigh_variant_override(
        binding, arrays, spec=spec, stream=stream
    )
    assert np.all(arrays["uten1"] == np.float32(1.0e-5))
    assert np.all(arrays["vten1"] == np.float32(2.0e-5))
    assert np.all(arrays["wten1"] == np.float32(3.0e-5))
    assert np.all(arrays["thten1"] == np.float32(4.0e-5))
    assert stream.waited == [binding.ready_event]
    assert receipt["default_rayleigh_values_overwritten_before_other_held_terms"] is True
    assert receipt["operation_token_sha256"] == live._operation_token(receipt)


def test_full_held_default_shape_order_and_lineage_are_strict() -> None:
    parameter = inspect.signature(
        full_held.launch_cuda_gm1_tiled_solve1_full_held
    ).parameters["rayleigh_variant_binding"]
    assert parameter.default is None
    order = full_held.SOURCE_ORDER_WITH_ACTIVE_RAYLEIGH_VARIANT
    assert order[2] == full_held.SOURCE_ORDER[2]
    assert "active flat irdamp/hrdamp" in order[3]
    assert order[4] == full_held.SOURCE_ORDER[4]
    source = (
        ROOT / "src/cm1core/cuda_gm1_tilestream_solve1_full_held.py"
    ).read_text(encoding="utf-8")
    start = source.index("def launch_cuda_gm1_tiled_solve1_full_held")
    positions = [
        source.index("launch_cuda_gm1_child_terrain_rayleigh(", start),
        source.index("enqueue_cuda_gm1_solve1_rayleigh_variant_override(", start),
        source.index("enqueue_cuda_gm1_solve1_lspgrad_after_rayleigh(", start),
        source.index("idiss_consumer.apply(", start),
    ]
    assert positions == sorted(positions)

    assert full_held._aggregate_rayleigh_variant_lineage({False}, [], [], [], []) is None
    selection = {
        "active": {
            "irdamp": 2,
            "hrdamp_input": 1,
            "hrdamp_effective": 2,
            "terrain_flag": False,
            "axisymmetric": 0,
        },
        "companion": {
            "lspgrad": 0,
            "rterm": 0,
            "mpi": False,
            "diagnostics": False,
            "adaptive_dt": False,
        },
    }
    active = full_held._aggregate_rayleigh_variant_lineage(
        {True},
        ["a" * 64, "a" * 64],
        ["b" * 64, "c" * 64],
        ["d" * 64, "d" * 64],
        [selection, selection],
    )
    assert active is not None
    assert active["rayleigh_variant_selection"] == selection
    assert active["rayleigh_variant_active_cuda_raw_bit_oracle"] is False
    with pytest.raises(ValueError, match="mix active/default"):
        full_held._aggregate_rayleigh_variant_lineage(
            {False, True}, ["a" * 64], ["b" * 64], ["d" * 64], [selection]
        )


def test_validation_receipt_seals_cpu_only_binding() -> None:
    receipt = json.loads(
        (
            ROOT
            / "evidence/validation/cm1-rayleigh-solve1-source-order-binding-v1/validation-receipt.json"
        ).read_bytes()
    )
    assert receipt["status"] == "PASS_CPU_TYPED_SOURCE_ORDER_BINDING_ONLY"
    assert receipt["result"]["default_rayleigh_receipt_shape_unchanged"] is True
    assert receipt["result"]["active_cuda_raw_bit_oracle"] is False
    assert receipt["result"]["active_coupled_trajectory_admitted"] is False
    for artifact in receipt["artifacts"].values():
        path = ROOT / artifact["path"]
        raw = path.read_bytes()
        assert len(raw) == artifact["bytes"]
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
