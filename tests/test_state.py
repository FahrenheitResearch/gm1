from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cm1core.config import load_config
from cm1core.layout import GridShape, LiveRegionPolicy
from cm1core.state import (
    CONFIRMED_INTERIOR_WORD,
    STRUCTURAL_CLAIM_LIMIT,
    UNCONFIRMED_HALO_WORD,
    build_structural_state_plan,
    initialize_structural_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _dry_config():
    return load_config(ROOT / "examples" / "dry-psolver2-bootstrap.toml")


def test_state_plan_binds_config_grid_fields_and_deferred_gates() -> None:
    config = _dry_config()
    plan = build_structural_state_plan(config)

    assert plan.config_sha256 == config.sha256
    assert plan.grid.as_dict() == {"nx": 16, "ny": 16, "nz": 16}
    assert plan.spacing_m == (1000.0, 1000.0, 1000.0)
    assert plan.halo.as_dict() == {"x": 3, "y": 3, "z": 1}
    assert tuple(field.layout.native_name for field in plan.fields) == (
        "ua",
        "u3d",
        "va",
        "v3d",
        "wa",
        "w3d",
        "ppi",
        "pp3d",
        "tha",
        "th3d",
    )
    assert plan.total_nbytes == 358_688
    assert {row.gate for row in plan.deferred} == {
        "O-INSTRUMENT",
        "O-GRID",
        "O-CLOSURES",
        "O-ARWEN-WSM6",
    }
    assert "not CM1 base-state" in STRUCTURAL_CLAIM_LIMIT


def test_structural_initialization_zeros_only_confirmed_interiors() -> None:
    state = initialize_structural_state(build_structural_state_plan(_dry_config()))

    ua = state.buffer("ua")
    assert ua.word_bits(i=1, j=1, k=1) == CONFIRMED_INTERIOR_WORD
    assert ua.word_bits(i=17, j=16, k=16) == CONFIRMED_INTERIOR_WORD
    assert ua.word_bits(i=-2, j=-2, k=0) == UNCONFIRMED_HALO_WORD
    assert ua.word_bits(i=18, j=16, k=16) == UNCONFIRMED_HALO_WORD

    wa = state.buffer("wa")
    assert wa.word_bits(i=1, j=1, k=17) == CONFIRMED_INTERIOR_WORD
    assert wa.word_bits(i=1, j=1, k=18) == UNCONFIRMED_HALO_WORD


def test_matching_allocations_do_not_merge_a_and_current_policies() -> None:
    state = initialize_structural_state(build_structural_state_plan(_dry_config()))
    for carrier, current, expected_policies in (
        (
            "ua",
            "u3d",
            (
                LiveRegionPolicy.X_FACE_A_TIME_LEVEL,
                LiveRegionPolicy.X_FACE_3D,
            ),
        ),
        (
            "ppi",
            "pp3d",
            (
                LiveRegionPolicy.SCALAR_A_TIME_LEVEL,
                LiveRegionPolicy.SCALAR_3D,
            ),
        ),
    ):
        first = state.buffer(carrier)
        second = state.buffer(current)
        assert first.field.geometry == second.field.geometry
        assert (first.field.layout.live_region, second.field.layout.live_region) == expected_policies
        assert first.data == second.data


def test_structural_allocation_obeys_explicit_byte_limit() -> None:
    plan = build_structural_state_plan(_dry_config())
    with pytest.raises(MemoryError, match="structural state needs"):
        initialize_structural_state(plan, max_bytes=plan.total_nbytes - 1)


def test_structural_plan_refuses_grid_drift_from_the_sealed_lane() -> None:
    plan = build_structural_state_plan(_dry_config())
    with pytest.raises(ValueError, match="dimensions differ from the fixed lane profile"):
        replace(plan, grid=GridShape(17, 16, 16))


def test_native_index_access_refuses_outside_allocation() -> None:
    state = initialize_structural_state(build_structural_state_plan(_dry_config()))
    with pytest.raises(IndexError):
        state.buffer("ppi").word_bits(i=-3, j=1, k=1)
    with pytest.raises(TypeError):
        state.buffer("ppi").word_bits(i=True, j=1, k=1)
    with pytest.raises(TypeError):
        state.buffer("ppi").word_bits(i=1.0, j=1, k=1)
