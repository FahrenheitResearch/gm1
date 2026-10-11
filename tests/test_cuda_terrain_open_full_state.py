from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cm1_marks

from cm1core.cuda_terrain_open_full_state import (
    COLD_START_HISTORY_SOURCE_ANCHOR,
    CUDA_TERRAIN_OPEN_FULL_STATE_CLAIM_LIMIT,
    CudaTerrainOpenFullState,
    DEFAULT_DEVICE_RESERVE_BYTES,
    FULL_PRODUCT_PAYLOAD_BYTES,
    _validate_cold_start_history_request,
    arm_cuda_terrain_open_full_state_step,
    full_state_memory_plan,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_EXPORT = ROOT / "work/initialized-product-v3/full"


def test_import_is_cuda_cold_and_history_source_anchor_is_live() -> None:
    command = (
        "import sys; import cm1core.cuda_terrain_open_full_state as m; "
        "assert not any(x == 'cupy' or x.startswith('cupy.') for x in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert COLD_START_HISTORY_SOURCE_ANCHOR == "src/cm1.F:1559-1566"
    source = (ROOT / "upstream/cm1-21.1/pristine/src/cm1.F").read_text()
    lines = source.splitlines()[1558:1566]
    # The west radiation array's allocation, then its zero fill, and the north one's zero fill.
    assert cm1_marks.is_line(lines[0], "0ff807ac9ca0b980/29")
    assert cm1_marks.is_line(lines[1], "f161aadec779b941/10")
    assert cm1_marks.is_line(lines[7], "04cac2637283f7f5/10")
    assert "no full-domain" not in CUDA_TERRAIN_OPEN_FULL_STATE_CLAIM_LIMIT


def test_full_state_memory_plan_is_exact_and_keeps_two_gibibytes_free() -> None:
    plan = full_state_memory_plan()
    assert plan.product_arena_bytes == FULL_PRODUCT_PAYLOAD_BYTES
    assert plan.prepared_array_bytes == 1_051_732_568
    assert plan.acoustic_scratch_bytes == 110_899_224
    assert plan.device_reserve_bytes == DEFAULT_DEVICE_RESERVE_BYTES
    assert plan.projected_peak_live_bytes == 2_332_461_092
    assert plan.required_free_after_product_bytes == 3_310_115_440
    assert plan.receipt_dict()["projected_peak_live_bytes"] == 2_332_461_092


def test_memory_plan_and_history_contract_refuse_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="ni must be a positive integer"):
        full_state_memory_plan(ni=0)
    with pytest.raises(ValueError, match="currently admits only cold-start-zero"):
        _validate_cold_start_history_request("resume", None)
    with pytest.raises(ValueError, match="refuses imported radiation histories"):
        _validate_cold_start_history_request("cold-start-zero", object())


def test_later_step_arm_requires_explicit_nonzero_history_provenance() -> None:
    state = CudaTerrainOpenFullState(
        prepared=object(),  # type: ignore[arg-type]
        memory_plan=full_state_memory_plan(),
        history_mode="cold-start-zero",
        memory_receipt={},
        armed=None,
        completed_steps=1,
    )
    with pytest.raises(ValueError, match="refuse zero/reset"):
        arm_cuda_terrain_open_full_state_step(
            state,
            epoch=1,
            a_current_provenance="coupler epoch one",
            tendency_provenance="coupler tendency epoch one",
            radiation_history_provenance="reset zero histories",
        )
    arm = arm_cuda_terrain_open_full_state_step(
        state,
        epoch=1,
        a_current_provenance="coupler epoch one",
        tendency_provenance="coupler tendency epoch one",
        radiation_history_provenance="preserved from completed epoch zero",
    )
    assert arm.epoch == 1
    assert state.armed is arm


def test_full_initialized_product_is_strictly_hash_bound_when_available() -> None:
    if not FULL_EXPORT.is_dir():
        pytest.skip("ignored full corrected-v3 initialized product is unavailable")
    from cm1core.cuda_initialized_product import load_initialized_product_export

    loaded = load_initialized_product_export(
        FULL_EXPORT,
        expected_manifest_sha256=(
            "7fd218a005b1c9f01d37b57452b281c2df6c797973268e8fe86b996eb6b61e2e"
        ),
        expected_payload_sha256=(
            "8f589cee64b044afcfef34dca6b638c3829946a999537885b1db3576d4ec7981"
        ),
        expected_source_artifact_sha256=(
            "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f"
        ),
        expected_source_contract_sha256=(
            "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12"
        ),
    )
    assert loaded.is_full_domain
    assert loaded.crop_origin_yx == (0, 0)
    assert loaded.crop_count_yx == (300, 300)
    assert loaded.payload_nbytes == FULL_PRODUCT_PAYLOAD_BYTES
