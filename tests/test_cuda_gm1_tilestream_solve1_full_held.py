from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_gm1_tilestream_solve1_full_held as full_held
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_token_accepts_nested_immutable_receipts_only() -> None:
    receipt = MappingProxyType({"b": MappingProxyType({"value": 1}), "a": 2})
    assert json.loads(full_held._canonical(receipt)) == {
        "a": 2,
        "b": {"value": 1},
    }
    with pytest.raises(TypeError, match="is not JSON serializable"):
        full_held._canonical(object())


def test_tile_inventory_reuses_closure_strain_diffusivity_and_metrics() -> None:
    shapes = full_held.solve1_full_held_tile_shapes(ni=198, nj=198, nk=75)
    assert len(shapes) == 33
    assert shapes["tauh"] == (75, 200, 200)
    assert shapes["tauf"] == (76, 200, 200)
    assert shapes["u3d"] == (75, 198, 199)
    assert shapes["v3d"] == (75, 199, 198)
    assert shapes["w3d"] == (76, 198, 198)
    assert shapes["uten"] == (75, 198, 199)
    assert shapes["vten"] == (75, 199, 198)
    assert shapes["wten"] == (76, 198, 198)
    assert shapes["qten"] == (6, 75, 198, 198)
    assert shapes["pi0"] == (75, 198, 198)
    assert shapes["ppi"] == (75, 198, 198)
    for name in (
        "s11",
        "s12",
        "s13",
        "s22",
        "s23",
        "s33",
        "kmh",
        "kmv",
        "khh",
        "khv",
        "mh",
        "mf",
        "gz",
        "rgz",
        "gzu",
        "rgzu",
        "gzv",
        "rgzv",
        "gx",
        "gxu",
        "gy",
        "gyv",
    ):
        assert name not in shapes
    assert set(shapes) == set(full_held._ARRAY_NAMES)


def test_five_meter_memory_inventory_is_exact_and_fail_closed() -> None:
    inventory = full_held.five_meter_solve1_full_held_inventory()
    assert inventory["compute_tile"] == [198, 198]
    assert inventory["caller_array_bytes_each"] == 508_225_564
    assert inventory["scalar_workspace_bytes_each"] == 119_760_016
    assert inventory["idiss_workspace_bytes_each"] == 58_806_000
    assert inventory["one_tile_incremental_device_bytes"] == 686_791_580
    assert inventory["inventory_sha256"] == (
        "9e3f961827c1cf815bd18778b435682797e5a901044bcae25a20770eb4b5c8d6"
    )
    assert inventory["tile_streamable"] is True
    assert inventory["live_five_meter_full_held_token_available"] is False
    assert inventory["completed_rk_stage_available"] is False
    assert inventory["global_psolver_tile_streamable"] is False
    assert inventory["five_meter_meteorological_product"] is False


def test_invalid_memory_geometry_is_refused() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        full_held.five_meter_solve1_full_held_inventory(tile_nx=0)
    with pytest.raises(ValueError, match="positive integer"):
        full_held.five_meter_solve1_full_held_inventory(tile_ny=True)


def test_uniform_idealized_reference_dz_is_explicit_and_exact_before_cuda() -> None:
    spec = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96).specs[0]
    stream = SimpleNamespace(ptr=12)
    with pytest.raises(ValueError, match="reference dz differs"):
        full_held.launch_cuda_gm1_tiled_solve1_full_held(
            object(),
            object(),
            {},
            spec=spec,
            vertical=object(),
            dx=50.0,
            dy=50.0,
            dt=0.25,
            stream=stream,
            reference_dz=100.0,
            allow_uniform_idealized_validation=True,
        )
    with pytest.raises(ValueError, match="reference dz differs"):
        full_held.launch_cuda_gm1_tiled_solve1_full_held(
            object(),
            object(),
            {},
            spec=spec,
            vertical=object(),
            dx=50.0,
            dy=50.0,
            dt=0.25,
            stream=stream,
            reference_dz=200.0,
            allow_uniform_idealized_validation=False,
        )


def test_required_pressure_arrays_are_shape_checked_and_non_aliasing() -> None:
    class FakeArray:
        def __init__(self, shape: tuple[int, ...], pointer: int) -> None:
            self.dtype = "float32"
            self.shape = shape
            self.flags = SimpleNamespace(c_contiguous=True)
            self.data = SimpleNamespace(ptr=pointer)

    class FakeCupy:
        ndarray = FakeArray
        float32 = "float32"

    shapes = full_held.solve1_full_held_tile_shapes(ni=8, nj=7, nk=5)
    arrays = {
        name: FakeArray(shape, pointer)
        for pointer, (name, shape) in enumerate(shapes.items(), start=1)
    }
    checked = full_held._require_arrays(FakeCupy, arrays, ni=8, nj=7, nk=5)
    assert checked["pi0"].shape == (5, 7, 8)
    assert checked["ppi"].shape == (5, 7, 8)

    wrong_shape = dict(arrays)
    wrong_shape["ppi"] = FakeArray((5, 7, 7), 10_001)
    with pytest.raises(ValueError, match="ppi must be"):
        full_held._require_arrays(FakeCupy, wrong_shape, ni=8, nj=7, nk=5)

    aliased = dict(arrays)
    aliased["ppi"] = FakeArray((5, 7, 8), arrays["pi0"].data.ptr)
    with pytest.raises(ValueError, match="must not alias"):
        full_held._require_arrays(FakeCupy, aliased, ni=8, nj=7, nk=5)


def test_launch_source_order_is_literal_and_binds_tke_before_held_terms() -> None:
    source = (
        ROOT / "src/cm1core/cuda_gm1_tilestream_solve1_full_held.py"
    ).read_text(encoding="utf-8")
    start = source.index("def launch_cuda_gm1_tiled_solve1_full_held")
    ordered = (
        "tke_launch = _require_tke_launch(",
        "stream.wait_event(tke_launch.ready_event)",
        'checked["thten1"].fill',
        'checked["qten"].fill',
        "for name in _STRESS_TENDENCY_NAMES:",
        "launch_cuda_gm1_child_terrain_rayleigh(",
        "idiss_consumer.apply(",
        "launch_cuda_gm1_child_stretched_sgs_stress(",
        "launch_cuda_gm1_child_stretched_scalar_turbulence(",
        'cupy.add(checked["uten1"]',
        'cupy.add(checked["vten1"]',
        "checked[\"wten1\"]",
    )
    positions = [source.index(value, start) for value in ordered]
    assert positions == sorted(positions)
    assert '"momentum_or_scalar_advection_performed": False' in source
    assert '"rk_stage_performed": False' in source
    assert '"five_meter_meteorological_product": False' in source


def test_tke_lineage_is_strictly_typed_before_cuda() -> None:
    plan = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96)
    spec = plan.specs[0]
    with pytest.raises(TypeError, match="exact corrected closure result"):
        full_held._require_closure_launch(
            object(), spec=spec, stream_pointer=12
        )
    with pytest.raises(TypeError, match="exact same-tile solve1 TKE result"):
        full_held._require_tke_launch(
            object(), closure=object(), spec=spec, stream_pointer=12
        )
    with pytest.raises(TypeError, match="exact corrected closure token"):
        full_held._require_closure_token(object(), plan=plan)
    with pytest.raises(TypeError, match="exact completed solve1 TKE token"):
        full_held._require_tke_token(object(), closure={}, plan=plan)


def test_finish_refuses_non_oracle_domain_before_token_or_output_access() -> None:
    plan = build_gm1_open_tile_plan(nx=24, ny=24, tile_nx=12, tile_ny=12)
    with pytest.raises(ValueError, match="300 grid oracle"):
        full_held.finish_cuda_gm1_tiled_solve1_full_held(
            object(), object(), (), plan=plan, outputs={}
        )


def test_authority_and_all_operator_pins_are_frozen() -> None:
    assert full_held.AUTHORITY_SOLVE1_SHA256 == (
        "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
    )
    assert full_held.AUTHORITY_TURBTEND_SHA256 == (
        "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de"
    )
    assert full_held.GM1_CHILD_RAYLEIGH_SOURCE_SHA256 == (
        "6509e757e6f2a1f87b6ee50cb20d540ce9f618c262e4c2da40be04bda4bb138d"
    )
    assert full_held.CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE_SHA256 == (
        "9787d00ed8b3309fd33df148e186d85624cb96fa1871013cc54c03516aed0d83"
    )
    assert full_held.CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256 == (
        "7f57b86da8aaf2f3d80a61e1f7dfada465cc2878f6d5aeaf6fccb5a6e3950506"
    )
    assert full_held.SOURCE_ORDER[1] == (
        "same-tile TKE source plus stretched turbt diffusion"
    )
    assert full_held.SOURCE_ORDER.index(
        "stretched SGS stress adds momentum stress divergence"
    ) < full_held.SOURCE_ORDER.index(
        "stretched scalar SGS adds theta then qv/qc/qr/qi/qs/qg"
    )


def test_token_scope_stops_before_advection_rk_and_product() -> None:
    claim = full_held.GM1_TILESTREAM_SOLVE1_FULL_HELD_CLAIM_LIMIT
    assert "no advection" in claim
    assert "five-metre meteorological product" in claim
    assert full_held.CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS == (
        "ADMITTED_LIVE_TILESTREAM_SOLVE1_FULL_HELD"
    )


def test_scalar_projection_binds_exact_full_token_bytes_plan_and_stream() -> None:
    plan = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96)
    stream = SimpleNamespace(ptr=42)
    outputs = {
        "thten1": np.zeros((75, 300, 300), dtype="<f4"),
        "qten": np.zeros((6, 75, 300, 300), dtype="<f4"),
    }
    shapes = {
        "uten": [75, 300, 301],
        "vten": [75, 301, 300],
        "wten": [76, 300, 300],
        "thten1": [75, 300, 300],
        "qten": [6, 75, 300, 300],
    }
    hashes = {
        "uten": "1" * 64,
        "vten": "2" * 64,
        "wten": "3" * 64,
        **{
            name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for name, value in outputs.items()
        },
    }
    full_receipt = {
        "schema": full_held.CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
        "status": full_held.CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
        "completed_nrk": 0,
        "closure_operation_token_sha256": "4" * 64,
        "solve1_tke_operation_token_sha256": "5" * 64,
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_operation_tokens_sha256": "6" * 64,
        "stream_pointer": 42,
        "host_scatter_exact_once": True,
        "output_shapes": shapes,
        "output_sha256": hashes,
    }
    full_receipt["operation_token_sha256"] = full_held._operation_token(full_receipt)
    full_token = full_held.Gm1TiledSolve1FullHeldToken(
        MappingProxyType(full_receipt)
    )
    projected = full_held.project_gm1_tiled_solve1_scalar_held_from_full(
        full_token, outputs, plan=plan, stream=stream
    )
    receipt = projected.receipt
    assert receipt["projected_from_solve1_full_held_token_sha256"] == (
        full_receipt["operation_token_sha256"]
    )
    assert receipt["full_held_source_order_bound"] is True
    assert receipt["stream_pointer"] == 42
    assert receipt["tile_plan_sha256"] == plan.ownership_sha256
    assert receipt["output_sha256"] == {
        name: hashes[name] for name in outputs
    }
    assert receipt["operation_token_sha256"] == full_held._operation_token(receipt)

    with pytest.raises(TypeError, match="exact full-held token"):
        full_held.project_gm1_tiled_solve1_scalar_held_from_full(
            object(), outputs, plan=plan, stream=stream
        )
    with pytest.raises(ValueError, match="stream, plan"):
        full_held.project_gm1_tiled_solve1_scalar_held_from_full(
            full_token, outputs, plan=plan, stream=SimpleNamespace(ptr=43)
        )
    outputs["thten1"].flat[0] = np.float32(1.0)
    with pytest.raises(ValueError, match="thten1 bytes differ"):
        full_held.project_gm1_tiled_solve1_scalar_held_from_full(
            full_token, outputs, plan=plan, stream=stream
        )
