"""CPU geometry/producer and CUDA-enqueue-contract coverage; no GPU claim."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_gm1_tilestream_stretched_closure as closure
from cm1core import cuda_gm1_child_stretched_sgs_tke as sgs_tke
from cm1core.cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec, build_gm1_open_tile_plan
from cm1core.cuda_stretch_z1 import (
    CudaStretchZ1Launch, StretchZ1Parameters, StretchZ1Selection, build_stretch_z1_grid,
)
from cm1core.grid_geometry import CartesianGridGeometry


def _full_spec(grid: CartesianGridGeometry) -> Gm1TileSpec:
    return Gm1TileSpec(
        0, 0, 0, grid.nx, grid.ny, grid.nx, grid.ny, 3,
        0, grid.nx, 0, grid.ny, 0, 0, grid.nx, grid.ny,
    )


def _geometry_kwargs(grid: CartesianGridGeometry) -> dict[str, object]:
    return {
        "nk": grid.nz, "dx": grid.dx_m, "dy": grid.dy_m, "dz": grid.dz_m,
        "ztop": grid.ztop_m, "allow_compact_validation": False,
        "allow_uniform_idealized_validation": True, "geometry": grid,
    }


def _vertical(grid: CartesianGridGeometry) -> CudaStretchZ1Launch:
    parameters = StretchZ1Parameters(
        grid.nz, grid.dz_m, grid.ztop_m, 0, grid.ztop_m, grid.dz_m, grid.dz_m
    )
    vertical_grid = build_stretch_z1_grid(parameters)
    return CudaStretchZ1Launch(
        object(), object(), vertical_grid,
        {
            "parameters": asdict(parameters), "selection": asdict(StretchZ1Selection()),
            "derived": asdict(vertical_grid.derived),
            "kernel": {"source_sha256": closure.CUDA_STRETCH_Z1_SOURCE_SHA256},
        },
    )


def test_non_square_237_level_uniform_domain_owns_each_word_once() -> None:
    grid = CartesianGridGeometry(31, 23, 237, 75, 90, 75)
    plan = build_gm1_open_tile_plan(nx=31, ny=23, tile_nx=11, tile_ny=8)
    coverage = np.zeros(grid.surface_shape, dtype=np.uint8)
    for spec in plan.specs:
        assert closure._validate_geometry(spec, **_geometry_kwargs(grid)) == (
            spec.cnx, spec.cny, 237, 75.0, 90.0, 75.0, 17775.0
        )
        local = closure.tiled_stretched_closure_owned_slices(
            spec, allow_uniform_idealized_validation=True, geometry=grid
        )
        coverage[spec.j0:spec.j1, spec.i0:spec.i1] += np.ones((spec.cny, spec.cnx), dtype=np.uint8)[local]
    assert np.all(coverage == 1)
    assert closure._validate_geometry(_full_spec(grid), **_geometry_kwargs(grid))[:3] == (31, 23, 237)


@pytest.mark.parametrize("nz,dz", ((237, 75.0), (11, 125.0), (75, 200.0)))
def test_uniform_vertical_producer_matches_native_uniform_coordinates(nz: int, dz: float) -> None:
    grid = CartesianGridGeometry(17, 23, nz, 75, 90, dz)
    vertical = _vertical(grid)
    # param.F:7789-7812: zf(k)=dz*(k-1), then average adjacent faces.
    faces = np.arange(-1, nz + 2, dtype=np.float32) * np.float32(dz)
    centers = np.float32(0.5) * (faces[1:] + faces[:-1])
    np.testing.assert_array_equal(vertical.grid.sigmaf.view(np.uint32), faces.view(np.uint32))
    np.testing.assert_array_equal(vertical.grid.sigma.view(np.uint32), centers.view(np.uint32))
    assert vertical.grid.sigmaf[nz + 1] == grid.ztop_m
    np.testing.assert_array_equal(vertical.grid.mh, np.ones(nz + 2, dtype=np.float32))
    np.testing.assert_array_equal(vertical.grid.mf, np.ones(nz + 3, dtype=np.float32))
    assert closure._require_vertical(
        SimpleNamespace(ndarray=np.ndarray, float32=np.float32), vertical, nz,
        allow_uniform_idealized_validation=True, geometry=grid,
    ) is vertical


@pytest.mark.parametrize(
    "change", ({"nk": 75}, {"dx": 50.0}, {"dy": 75.0}, {"dz": 200.0},
               {"ztop": 15000.0}, {"allow_uniform_idealized_validation": False}),
)
def test_conflicting_geometry_refuses_before_cuda_import(change: dict[str, object]) -> None:
    grid = CartesianGridGeometry(17, 23, 237, 75, 90, 75)
    with pytest.raises(ValueError):
        closure.launch_cuda_gm1_tiled_stretched_closure(
            {}, spec=_full_spec(grid), vertical=None, reduction=None,
            **(_geometry_kwargs(grid) | change),
        )


def test_source_halo_topology_and_stencil_are_still_required() -> None:
    grid = CartesianGridGeometry(17, 23, 237, 75, 90, 75)
    spec = _full_spec(grid)
    for changed in (replace(spec, halo=2), replace(spec, ci0=1), replace(spec, nx=18), replace(spec, i1=0)):
        with pytest.raises(ValueError):
            closure._validate_geometry(changed, **_geometry_kwargs(grid))
    tiny = replace(grid, nx=6)
    with pytest.raises(ValueError, match="stencil"):
        closure._validate_geometry(_full_spec(tiny), **_geometry_kwargs(tiny))
    tiny = replace(grid, nz=3)
    with pytest.raises(ValueError, match="stencil"):
        closure._validate_geometry(_full_spec(tiny), **_geometry_kwargs(tiny))
    # No implicit promotion of an arbitrary grid through the legacy boolean.
    with pytest.raises(ValueError):
        closure._validate_geometry(spec, **(_geometry_kwargs(grid) | {"geometry": None}))


@pytest.mark.parametrize("spacing", (1e40, 1e-50))
def test_geometry_must_survive_the_actual_float32_kernel_abi(spacing: float) -> None:
    grid = CartesianGridGeometry(17, 23, 237, spacing, 75, 75)
    with pytest.raises(ValueError, match="float32"):
        closure._validate_geometry(_full_spec(grid), **_geometry_kwargs(grid))


@pytest.mark.parametrize("field", ("parameters", "selection", "derived", "kernel"))
def test_uniform_vertical_source_identity_cannot_be_relabelled(field: str) -> None:
    grid = CartesianGridGeometry(17, 23, 237, 75, 90, 75)
    vertical = _vertical(grid)
    changed = replace(vertical, receipt=dict(vertical.receipt) | {field: {}})
    with pytest.raises(ValueError):
        closure._require_vertical(
            SimpleNamespace(ndarray=np.ndarray, float32=np.float32), changed, 237,
            allow_uniform_idealized_validation=True, geometry=grid,
        )


class _DeviceArray(np.ndarray):
    @property
    def data(self) -> SimpleNamespace:
        return SimpleNamespace(ptr=self.__array_interface__["data"][0])

    def get(self) -> np.ndarray:
        return np.asarray(self)


class _Stream:
    ptr = 17

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def wait_event(self, _event: object) -> None:
        pass


class _Event:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def record(self, _stream: object) -> None:
        pass

    def synchronize(self) -> None:
        pass


def test_generic_geometry_reaches_each_enqueue_and_finish_binds_it(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = CartesianGridGeometry(17, 23, 237, 75, 90, 75)
    stream = _Stream()
    fake_cupy = SimpleNamespace(
        ndarray=_DeviceArray, float32=np.float32, uint32=np.uint32, uint8=np.uint8,
        empty=lambda shape, dtype: np.empty(shape, dtype=dtype).view(_DeviceArray),
        zeros=lambda shape, dtype: np.zeros(shape, dtype=dtype).view(_DeviceArray),
        cuda=SimpleNamespace(Event=_Event, get_current_stream=lambda: stream),
    )
    real_import = closure.importlib.import_module
    monkeypatch.setattr(closure.importlib, "import_module", lambda name: fake_cupy if name == "cupy" else real_import(name))
    calls: dict[str, tuple[object, ...]] = {}

    def kernel(name: str):
        def enqueue(_launch_grid, _block, args, *, stream):
            calls[name] = args
        return enqueue

    def calcnm(_arrays, **kwargs):
        calls["calcnm"] = (kwargs["ni"], kwargs["nj"], kwargs["nk"], kwargs["dz"])

    monkeypatch.setattr(closure, "launch_cuda_child_moist_nm", calcnm)
    compiled = closure.CompiledGm1TiledStretchedClosure(
        SimpleNamespace(kernels=tuple(kernel(name) for name in ("surface", "columns", "cross", "rds"))),
        object(),
        SimpleNamespace(kernels=tuple(kernel(name) for name in ("normal", "shear", "reduce"))),
        SimpleNamespace(kernels=tuple(kernel(name) for name in ("invalid", "fresh", "faces"))),
    )
    vertical = _vertical(grid)
    vertical = replace(vertical, grid=SimpleNamespace(
        sigma=vertical.grid.sigma.view(_DeviceArray), sigmaf=vertical.grid.sigmaf.view(_DeviceArray),
        derived=vertical.grid.derived,
    ))
    arrays = {
        name: fake_cupy.empty(shape, np.float32)
        for name, shape in closure.tiled_stretched_closure_input_shapes(grid.nx, grid.ny, grid.nz).items()
    }
    reduction = closure.begin_cuda_gm1_tiled_stretched_closure_reduction(stream=stream)
    launch = closure.launch_cuda_gm1_tiled_stretched_closure(
        arrays, spec=_full_spec(grid), vertical=vertical, reduction=reduction,
        compiled=compiled, stream=stream, **_geometry_kwargs(grid),
    )
    assert calls["calcnm"] == (17, 23, 237, 75.0)
    assert tuple(calls["columns"][-5:]) == (17, 23, 237, 17775.0, 75.0)
    assert tuple(calls["cross"][-6:-3]) == (17, 23, 237)
    assert calls["rds"][-1] == 237
    assert tuple(calls["reduce"][-3:]) == (17, 23, 237)
    for name in ("fresh", "invalid"):
        assert tuple(calls[name][-6:]) == (17, 23, 237, 75.0, 90.0, 75.0)
    assert tuple(calls["faces"][-3:]) == (17, 23, 237)
    assert launch.arrays["tkea"].shape == (238, 23, 17)
    assert launch.receipt["grid_geometry"] == grid.record()
    metric_view = sgs_tke.bind_uniform_cartesian_sgs_tke_metrics(vertical, launch, geometry=grid)
    assert metric_view.geometry == grid
    assert metric_view.arrays["sigma"] is vertical.grid.sigma
    assert metric_view.arrays["rds"] is launch.workspace["rds"]
    assert metric_view.ready_event is launch.ready_event
    assert sgs_tke.bind_gm1_child_stretched_sgs_tke_metrics(metric_view, geometry=grid) is metric_view
    with pytest.raises(ValueError, match="source-bound"):
        sgs_tke.bind_gm1_child_stretched_sgs_tke_metrics(metric_view, geometry=replace(grid, dz_m=100))
    plan = Gm1TilePlan(17, 23, 17, 23, 3, (_full_spec(grid),), "a" * 64)
    token = closure.finish_cuda_gm1_tiled_stretched_closure(
        reduction, [launch], plan=plan, stream=stream, geometry=grid
    )
    assert token.receipt["grid_geometry"] == grid.record()
    assert token.receipt["vertical_disposition"] == "source-audited-flat-uniform-cartesian"
    with pytest.raises(ValueError, match="required at finish"):
        closure.finish_cuda_gm1_tiled_stretched_closure(reduction, [launch], plan=plan, stream=stream)
    changed = dict(launch.receipt)
    changed["grid_geometry"] = replace(grid, dz_m=100).record()
    changed["operation_token_sha256"] = closure._operation_token(changed)
    with pytest.raises(ValueError, match="explicit uniform geometry"):
        closure.finish_cuda_gm1_tiled_stretched_closure(
            reduction, [replace(launch, receipt=MappingProxyType(changed))],
            plan=plan, stream=stream, geometry=grid,
        )
