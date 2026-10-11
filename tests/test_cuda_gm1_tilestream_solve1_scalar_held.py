from __future__ import annotations

from pathlib import Path

import pytest

from cm1core import cuda_gm1_tilestream_solve1_scalar_held as scalar_held
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan


ROOT = Path(__file__).resolve().parents[1]


def test_tile_inventory_is_exact_and_nonaliasable_by_name() -> None:
    shapes = scalar_held.solve1_scalar_held_tile_shapes(ni=198, nj=198, nk=75)
    assert len(shapes) == 31
    assert shapes["tauh"] == (75, 200, 200)
    assert shapes["tauf"] == (76, 200, 200)
    assert shapes["u0"] == (75, 198, 199)
    assert shapes["v0"] == (75, 199, 198)
    assert shapes["qa"] == (6, 75, 198, 198)
    assert shapes["qten"] == (6, 75, 198, 198)
    assert shapes["thten1"] == (75, 198, 198)
    assert "khh" not in shapes
    assert "khv" not in shapes
    assert set(shapes) == set(scalar_held._ARRAY_NAMES)


def test_five_meter_memory_inventory_is_exact_and_fail_closed() -> None:
    inventory = scalar_held.five_meter_solve1_scalar_held_inventory()
    assert inventory["compute_tile"] == [198, 198]
    assert inventory["caller_array_bytes_each"] == 438_087_328
    assert inventory["closure_output_bytes_reused_each"] == 35_754_048
    assert inventory["scalar_workspace_bytes_each"] == 119_760_016
    assert inventory["idiss_workspace_bytes_each"] == 58_806_000
    assert inventory["one_tile_incremental_device_bytes"] == 616_653_344
    assert inventory["inventory_sha256"] == (
        "3acc3ac719b1db512f49062bba04dede7617132e6acd367c6ef41504c865f0c6"
    )
    assert inventory["tile_streamable"] is True
    assert inventory["live_five_meter_scalar_held_token_available"] is False
    assert inventory["scalar_rk1_available_at_five_meters"] is False
    assert inventory["global_psolver_tile_streamable"] is False
    assert inventory["five_meter_meteorological_product"] is False


def test_invalid_memory_geometry_is_refused() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        scalar_held.five_meter_solve1_scalar_held_inventory(tile_nx=0)
    with pytest.raises(ValueError, match="positive integer"):
        scalar_held.five_meter_solve1_scalar_held_inventory(tile_ny=True)


def test_launch_source_order_is_literal_and_narrow() -> None:
    source = (
        ROOT / "src/cm1core/cuda_gm1_tilestream_solve1_scalar_held.py"
    ).read_text(encoding="utf-8")
    start = source.index("def launch_cuda_gm1_tiled_solve1_scalar_held")
    ordered = (
        'checked["thten1"].fill',
        'checked["qten"].fill',
        "launch_cuda_gm1_child_terrain_rayleigh(",
        "idiss_consumer.apply(",
        "launch_cuda_gm1_child_stretched_scalar_turbulence(",
    )
    positions = [source.index(value, start) for value in ordered]
    assert positions == sorted(positions)
    assert '"scalar_advection_or_rk_performed": False' in source
    assert '"five_meter_meteorological_product": False' in source


def test_foreign_closure_objects_are_refused_before_cuda() -> None:
    plan = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=96, tile_ny=96)
    with pytest.raises(TypeError, match="exact corrected closure result"):
        scalar_held._require_closure_launch(
            object(), spec=plan.specs[0], stream_pointer=12
        )
    with pytest.raises(TypeError, match="exact corrected closure token"):
        scalar_held._require_closure_token(object(), plan=plan)


def test_finish_refuses_non_oracle_domain_before_output_access() -> None:
    plan = build_gm1_open_tile_plan(nx=24, ny=24, tile_nx=12, tile_ny=12)
    with pytest.raises(ValueError, match="300 grid oracle"):
        scalar_held.finish_cuda_gm1_tiled_solve1_scalar_held(
            object(), (), plan=plan, outputs={}
        )


def test_authority_and_stretched_operator_pins_are_frozen() -> None:
    assert scalar_held.AUTHORITY_SOLVE1_SHA256 == (
        "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
    )
    assert scalar_held.AUTHORITY_TURBTEND_SHA256 == (
        "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de"
    )
    assert scalar_held.CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256 == (
        "7f57b86da8aaf2f3d80a61e1f7dfada465cc2878f6d5aeaf6fccb5a6e3950506"
    )
    assert scalar_held.SOURCE_ORDER == (
        "positive-zero held thten1 and six-species qten",
        "terrain Rayleigh establishes thten1",
        "eqtset=2 idiss adds live closure dissten heating to thten1",
        "stretched scalar SGS adds theta then qv/qc/qr/qi/qs/qg",
    )


def test_token_scope_stops_before_rk_and_product() -> None:
    claim = scalar_held.GM1_TILESTREAM_SOLVE1_SCALAR_HELD_CLAIM_LIMIT
    assert "no scalar advection" in claim
    assert "five-metre meteorological product" in claim
    assert scalar_held.CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS == (
        "ADMITTED_LIVE_TILESTREAM_SOLVE1_SCALAR_HELD"
    )
