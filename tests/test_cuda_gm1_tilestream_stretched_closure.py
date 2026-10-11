from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_gm1_tilestream_stretched_closure as closure
from cm1core.cuda_gm1_child_nm_def import CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256
from cm1core.cuda_gm1_child_stretched_nm_def import (
    CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256,
)
from cm1core.cuda_gm1_child_turbsmag_init_generic import (
    CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256,
)
from cm1core.cuda_gm1_tilestream import Gm1TileSpec, build_gm1_open_tile_plan
from cm1core.cuda_stretch_z1_terrain import CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256
from cm1core.cuda_sgs_tke import CUDA_SGS_TKE_SOURCE_SHA256
from cm1core.gm1_child_stretched_closure_v3 import CARRIER_ORDER


ROOT = Path(__file__).resolve().parents[1]


def test_operator_sources_and_source_order_are_exact() -> None:
    assert dict(closure.OPERATOR_SOURCE_SHA256) == {
        "static_metrics": CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256,
        "calcnm": CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256,
        "stretched_calcdef": CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256,
        "base_calcdef": "e642efc44273461ddfc1934012aaac87bca222a0b5ce5f713438f263d3265c62",
        "fresh_tke": CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256,
    }
    assert closure.SOURCE_ORDER == (
        "stretched terrain/static metrics on the compute window",
        "three-dimensional-base moist calcnm",
        "profile-aware terrain calcdef",
        "aggregate metric-domain validation",
        "fresh glibc-2.39-exact closure coefficients and TKE",
        "rigid vertical-face disposition and exact tke3d/tketen copy",
    )
    source = inspect.getsource(closure.launch_cuda_gm1_tiled_stretched_closure)
    assert source.index("surface(") < source.index("launch_cuda_child_moist_nm(")
    assert source.index("launch_cuda_child_moist_nm(") < source.index(
        "compiled.calcdef.kernels[0]("
    )
    assert source.index("compiled.calcdef.kernels[2](") < source.index(
        "compiled.fresh_tke.kernels[0]("
    )
    assert source.index("compiled.fresh_tke.kernels[0](") < source.index(
        "compiled.fresh_tke.kernels[1]("
    )
    assert ".synchronize(" not in source
    assert "counter.get(" not in source
    assert source.index("compiled.calcdef.kernels[2](") < source.index(
        "launch_cuda_sgs_tke("
    )
    assert closure.RETAINED_TKE_SOURCE_ORDER[-3:] == (
        "exact restored closure-refresh grdscl top-face disposition",
        "source-exact tkekm/tkebc coefficients around retained tkea",
        "exact tke3d copy with source-produced tketen",
    )
    assert CUDA_SGS_TKE_SOURCE_SHA256 not in closure.OPERATOR_SOURCE_SHA256.values()


def test_tile_shapes_preserve_staggering_and_exact_carrier_order() -> None:
    inputs = closure.tiled_stretched_closure_input_shapes(198, 198, 75)
    assert tuple(inputs) == closure.INPUT_NAMES
    assert inputs["terrain_msl"] == (198, 198)
    assert inputs["u3d"] == (75, 198, 199)
    assert inputs["v3d"] == (75, 199, 198)
    assert inputs["w3d"] == (76, 198, 198)
    assert inputs["q3d"] == (6, 75, 198, 198)
    outputs = closure.tiled_stretched_closure_output_shapes(198, 198, 75)
    assert tuple(outputs) == CARRIER_ORDER
    assert outputs["grdscl"] == (75, 198, 198)
    assert outputs["grdscl"] == (75, 198, 198)
    assert all(
        outputs[name] == (76, 198, 198) for name in CARRIER_ORDER if name != "grdscl"
    )

    workspace = closure.tiled_stretched_closure_workspace_shapes(198, 198, 75)
    assert tuple(workspace) == closure.WORKSPACE_NAMES
    assert workspace["gxu"] == (76, 198, 199)
    assert workspace["gyv"] == (76, 199, 198)
    assert workspace["s12"] == (75, 199, 199)
    assert workspace["s13"] == (76, 198, 199)
    assert workspace["s23"] == (76, 199, 198)
    assert workspace["saturation_mask"] == (75, 198, 198)


def test_buffer_inventory_is_exact_and_has_one_byte_mask() -> None:
    sizes = closure.tiled_stretched_closure_buffer_nbytes(
        tile_nx=192, tile_ny=192, nk=75
    )
    inputs = closure.tiled_stretched_closure_input_shapes(198, 198, 75)
    workspace = closure.tiled_stretched_closure_workspace_shapes(198, 198, 75)
    expected_inputs = 4 * sum(np.prod(shape) for shape in inputs.values())
    expected_workspace = sum(
        np.prod(shape) * (1 if name == "saturation_mask" else 4)
        for name, shape in workspace.items()
    )
    assert sizes["input_bytes"] == expected_inputs
    assert sizes["workspace_bytes"] == expected_workspace
    assert sizes["one_buffer_bytes"] == expected_inputs + expected_workspace
    assert sizes["double_buffer_bytes"] == 2 * sizes["one_buffer_bytes"]
    assert sizes["shared_vertical_profile_bytes"] == 2_480
    assert sizes["shared_invalid_counter_bytes"] == 4
    assert sizes["single_buffer_resident_bytes"] == 608_579_768
    assert sizes["double_buffer_resident_bytes"] == 1_217_157_052


@pytest.mark.parametrize(
    ("nx", "tile", "expected_tiles"), ((300, 96, 16), (3000, 192, 256))
)
def test_plan_owns_every_closure_word_once_and_hides_interior_tile_edges(
    nx: int, tile: int, expected_tiles: int
) -> None:
    plan = build_gm1_open_tile_plan(nx=nx, ny=nx, tile_nx=tile, tile_ny=tile)
    assert plan.ntiles == expected_tiles
    coverage = np.zeros((nx, nx), dtype=np.uint8)
    for spec in plan.specs:
        local_y, local_x = closure.tiled_stretched_closure_owned_slices(spec)
        assert local_y.stop - local_y.start == spec.j1 - spec.j0
        assert local_x.stop - local_x.start == spec.i1 - spec.i0
        if spec.j0:
            assert local_y.start >= 3
        if spec.j1 < nx:
            assert spec.cny - local_y.stop >= 3
        if spec.i0:
            assert local_x.start >= 3
        if spec.i1 < nx:
            assert spec.cnx - local_x.stop >= 3
        coverage[spec.j0 : spec.j1, spec.i0 : spec.i1] += 1
    assert np.all(coverage == 1)


def test_split_or_rotated_tile_geometry_refuses() -> None:
    plan = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96)
    first = plan.specs[0]
    with pytest.raises(ValueError, match="true global edge"):
        closure.tiled_stretched_closure_owned_slices(replace(first, ci0=1))
    interior = plan.specs[5]
    with pytest.raises(ValueError, match="source halo"):
        closure.tiled_stretched_closure_owned_slices(
            replace(interior, ci0=interior.i0 - 2)
        )
    with pytest.raises(ValueError, match="three-cell"):
        closure.tiled_stretched_closure_owned_slices(replace(first, halo=2))


def test_full_domain_window_is_exact_and_uniform_idealized_only() -> None:
    spec = Gm1TileSpec(
        index=0,
        ty=0,
        tx=0,
        nx=300,
        ny=300,
        tile_nx=300,
        tile_ny=300,
        halo=3,
        i0=0,
        i1=300,
        j0=0,
        j1=300,
        ci0=0,
        cj0=0,
        cnx=300,
        cny=300,
    )
    with pytest.raises(ValueError, match="compute-x extent"):
        closure.tiled_stretched_closure_owned_slices(spec)
    assert closure.tiled_stretched_closure_owned_slices(
        spec,
        allow_uniform_idealized_validation=True,
    ) == (slice(0, 300), slice(0, 300))
    assert closure._validate_geometry(
        spec,
        nk=75,
        dx=50.0,
        dy=50.0,
        dz=200.0,
        ztop=15000.0,
        allow_compact_validation=False,
        allow_uniform_idealized_validation=True,
    )[:3] == (300, 300, 75)
    with pytest.raises(ValueError, match="compute-x extent"):
        closure.tiled_stretched_closure_owned_slices(
            replace(spec, index=1),
            allow_uniform_idealized_validation=True,
        )


def test_only_admitted_real_geometries_pass_without_compact_override() -> None:
    p50 = build_gm1_open_tile_plan(nx=300, ny=300).specs[0]
    assert closure._validate_geometry(
        p50,
        nk=75,
        dx=50.0,
        dy=50.0,
        dz=100.0,
        ztop=15000.0,
        allow_compact_validation=False,
    )[:3] == (102, 102, 75)

    assert closure._validate_geometry(
        p50,
        nk=75,
        dx=50.0,
        dy=50.0,
        dz=200.0,
        ztop=15000.0,
        allow_compact_validation=False,
        allow_uniform_idealized_validation=True,
    )[:3] == (102, 102, 75)
    with pytest.raises(ValueError, match="uniform idealized closure requires"):
        closure._validate_geometry(
            p50,
            nk=75,
            dx=50.0,
            dy=50.0,
            dz=100.0,
            ztop=15000.0,
            allow_compact_validation=False,
            allow_uniform_idealized_validation=True,
        )
    p5 = build_gm1_open_tile_plan(nx=3000, ny=3000, tile_nx=192, tile_ny=192).specs[-1]
    assert closure._validate_geometry(
        p5,
        nk=75,
        dx=5.0,
        dy=5.0,
        dz=100.0,
        ztop=15000.0,
        allow_compact_validation=False,
    )[:3] == (198, 198, 75)
    compact = build_gm1_open_tile_plan(nx=20, ny=18, tile_nx=8, tile_ny=7).specs[0]
    with pytest.raises(Exception, match="only the admitted"):
        closure._validate_geometry(
            compact,
            nk=4,
            dx=10.0,
            dy=10.0,
            dz=100.0,
            ztop=15000.0,
            allow_compact_validation=False,
        )
    assert closure._validate_geometry(
        compact,
        nk=4,
        dx=10.0,
        dy=10.0,
        dz=100.0,
        ztop=15000.0,
        allow_compact_validation=True,
    )[:3] == (14, 13, 4)


def test_finish_is_the_only_host_sync_and_reduction_token_is_canonical() -> None:
    launch_source = inspect.getsource(closure.launch_cuda_gm1_tiled_stretched_closure)
    finish_source = inspect.getsource(closure.finish_cuda_gm1_tiled_stretched_closure)
    assert ".synchronize(" not in launch_source
    assert "counter.get(" not in launch_source
    assert finish_source.count(".synchronize(") == 1
    assert finish_source.count("counter.get(") == 1
    record = {
        "schema": closure.CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
        "status": "test",
    }
    first = closure._operation_token(record)
    record["operation_token_sha256"] = first
    assert closure._operation_token(record) == first


def test_finish_propagates_one_exact_retained_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96)
    reduction_receipt = {
        "schema": closure.CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_REDUCTION_SCHEMA,
        "stream_pointer": 17,
    }
    reduction_receipt["operation_token_sha256"] = closure._operation_token(
        reduction_receipt
    )

    class _Counter:
        @staticmethod
        def get() -> np.ndarray:
            return np.zeros((1,), dtype=np.uint32)

    class _Stream:
        ptr = 17

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class _Event:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def record(self, _stream: object) -> None:
            pass

        def synchronize(self) -> None:
            pass

    reduction = closure.Gm1TiledStretchedClosureReduction(
        _Counter(), object(), 17, MappingProxyType(reduction_receipt)
    )
    launches = []
    for spec in plan.specs:
        receipt = {
            "tile": spec.record(),
            "stream_pointer": 17,
            "reduction_operation_token_sha256": reduction_receipt[
                "operation_token_sha256"
            ],
                "closure_lifecycle": "restored-post-solve3-retained-tkea",
                "vertical_disposition": "sealed-stretched-child",
            "retained_tkea_source_sha256": "a" * 64,
            "retained_grdscl_source_sha256": "b" * 64,
            "retained_closure_operation_token_sha256": "c" * 64,
        }
        receipt["operation_token_sha256"] = closure._operation_token(receipt)
        launches.append(
            closure.CudaGm1TiledStretchedClosureLaunch(
                object(), {}, {}, object(), MappingProxyType(receipt)
            )
        )
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Event=_Event))
    monkeypatch.setattr(
        closure.importlib,
        "import_module",
        lambda name: fake_cupy if name == "cupy" else __import__(name),
    )
    token = closure.finish_cuda_gm1_tiled_stretched_closure(
        reduction, launches, plan=plan, stream=_Stream()
    )
    assert token.receipt["closure_lifecycle"] == ("restored-post-solve3-retained-tkea")
    assert token.receipt["retained_tkea_source_sha256"] == "a" * 64
    assert token.receipt["retained_grdscl_source_sha256"] == "b" * 64
    assert token.receipt["retained_closure_operation_token_sha256"] == "c" * 64
    assert token.receipt["operator_source_sha256"]["retained_tke"] == (
        CUDA_SGS_TKE_SOURCE_SHA256
    )

    mixed = list(launches)
    changed = dict(mixed[-1].receipt)
    changed["closure_lifecycle"] = "fresh-initial-tkea"
    changed.pop("retained_tkea_source_sha256")
    changed.pop("retained_grdscl_source_sha256")
    changed.pop("retained_closure_operation_token_sha256")
    changed["operation_token_sha256"] = closure._operation_token(changed)
    mixed[-1] = replace(mixed[-1], receipt=MappingProxyType(changed))
    with pytest.raises(ValueError, match="mix fresh and restored"):
        closure.finish_cuda_gm1_tiled_stretched_closure(
            reduction, mixed, plan=plan, stream=_Stream()
        )


def test_module_source_is_self_consistent_and_claim_stays_bounded() -> None:
    path = ROOT / "src/cm1core/cuda_gm1_tilestream_stretched_closure.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        "complete-child envelope"
        in closure.CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_CLAIM_LIMIT
    )
    assert (
        "no complete-child envelope"
        in closure.CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_CLAIM_LIMIT
    )
    assert closure.CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_STATUS.endswith(
        "PENDING_GLOBAL_REDUCTION"
    )
    assert "checkpoint SHA-256" in closure.RETAINED_TKE_CLAIM_LIMIT
    assert closure.RETAINED_TKE_SCRATCH_NAMES == (
        "tk",
        "grdscl_full",
        "rgrdscl",
        "cme",
        "ce1",
        "ce2",
        "csm",
    )
