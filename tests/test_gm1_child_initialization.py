from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cm1core.errors import CM1Refusal
from cm1core.gm1_child_initialization import (
    GM1_CHILD_FLAT_GRID_RECEIPT_SHA256,
    GM1_CHILD_PARENT_CHECKPOINT_SHA256,
    GM1_CHILD_PLAN_SHA256,
    GM1_CHILD_PRODUCER_REQUIREMENTS_SHA256,
    GM1_CHILD_TERRAIN_OPERATOR_GATE_SHA256,
    load_gm1_child_initialization_plan,
    load_gm1_child_producer_requirements,
    load_gm1_child_terrain_operator_gate,
    require_gm1_child_producer_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "evidence/validation/gm1-t300-child-remap-preflight-v1/validated-parent-plan.json"
)
FLAT_GRID = ROOT / "evidence/validation/stretch-z1-child-15km-v1/receipt.json"
REQUIREMENTS = (
    ROOT
    / "evidence/validation/gm1-t300-child-remap-preflight-v1/producer-requirements.json"
)
TERRAIN_OPERATOR_GATE = (
    ROOT / "evidence/validation/stretch-z1-terrain-v1/validation-receipt.json"
)


def test_import_is_cuda_cold() -> None:
    command = (
        "import sys; import cm1core.gm1_child_initialization; "
        "assert not any(name == 'cupy' or name.startswith('cupy.') "
        "for name in sys.modules)"
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


def test_exact_parent_plan_and_official_flat_grid_are_admitted() -> None:
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == GM1_CHILD_PLAN_SHA256
    assert (
        hashlib.sha256(FLAT_GRID.read_bytes()).hexdigest()
        == GM1_CHILD_FLAT_GRID_RECEIPT_SHA256
    )
    plan = load_gm1_child_initialization_plan(PLAN, FLAT_GRID)
    assert plan.plan_sha256 == GM1_CHILD_PLAN_SHA256
    assert plan.receipt["parent"]["checkpoint"]["artifact_sha256"] == (
        GM1_CHILD_PARENT_CHECKPOINT_SHA256
    )
    assert len(plan.remap_3d_carriers) == 22
    assert len(plan.remap_surface_carriers) == 7
    assert plan.recompute_carriers == ("prs", "rf", "rho", "rr")
    assert plan.boundary_initialize_carriers == (
        "radbce",
        "radbcn",
        "radbcs",
        "radbcw",
    )
    assert plan.numerical_child_ready is False


def test_plan_hash_and_semantics_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="independent SHA-256"):
        load_gm1_child_initialization_plan(
            PLAN,
            FLAT_GRID,
            expected_plan_sha256="0" * 64,
        )
    value = json.loads(PLAN.read_bytes())
    value["geometry"]["selection"]["synthetic_translation"] = True
    tampered = tmp_path / "tampered-plan.json"
    raw = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    tampered.write_bytes(raw.encode() + b"\n")
    digest = hashlib.sha256(tampered.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source-coordinate 50 m child geometry"):
        load_gm1_child_initialization_plan(
            tampered,
            FLAT_GRID,
            expected_plan_sha256=digest,
        )


def test_numerical_child_remains_refused_until_all_producers_are_registered() -> None:
    plan = load_gm1_child_initialization_plan(PLAN, FLAT_GRID)
    with pytest.raises(CM1Refusal, match="GM1-50M-CHILD-PRODUCERS-PENDING"):
        require_gm1_child_producer_readiness(plan)
    with pytest.raises(CM1Refusal, match="GM1-50M-CHILD-PRODUCER-SCHEMA-UNREGISTERED"):
        require_gm1_child_producer_readiness(
            plan,
            {"schema": "unsealed.child-producer/v0"},
        )


def test_exact_readiness_only_requirements_are_admitted_but_keep_numerics_blocked() -> (
    None
):
    plan = load_gm1_child_initialization_plan(PLAN, FLAT_GRID)
    assert (
        hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
        == GM1_CHILD_PRODUCER_REQUIREMENTS_SHA256
    )
    requirements = load_gm1_child_producer_requirements(plan, REQUIREMENTS)
    assert requirements.parent_plan_sha256 == GM1_CHILD_PLAN_SHA256
    assert len(requirements.terrain_stretch_carriers) == 23
    assert len(requirements.hydrostatic_open_boundary_carriers) == 50
    assert requirements.numerical_child_output_allowed is False
    with pytest.raises(CM1Refusal, match="GM1-50M-CHILD-NUMERICAL-PRODUCERS-UNBOUND"):
        require_gm1_child_producer_readiness(plan, requirements)


def test_requirements_cannot_enable_output_without_sealed_producers(
    tmp_path: Path,
) -> None:
    plan = load_gm1_child_initialization_plan(PLAN, FLAT_GRID)
    value = json.loads(REQUIREMENTS.read_bytes())
    value["numerical_child_output_allowed"] = True
    tampered = tmp_path / "tampered-requirements.json"
    raw = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    tampered.write_bytes(raw.encode() + b"\n")
    digest = hashlib.sha256(tampered.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source/parent identity changed"):
        load_gm1_child_producer_requirements(
            plan,
            tampered,
            expected_sha256=digest,
        )


def test_analytic_terrain_operator_is_admitted_without_becoming_real_child() -> None:
    plan = load_gm1_child_initialization_plan(PLAN, FLAT_GRID)
    requirements = load_gm1_child_producer_requirements(plan, REQUIREMENTS)
    assert (
        hashlib.sha256(TERRAIN_OPERATOR_GATE.read_bytes()).hexdigest()
        == GM1_CHILD_TERRAIN_OPERATOR_GATE_SHA256
    )
    gate = load_gm1_child_terrain_operator_gate(
        requirements,
        TERRAIN_OPERATOR_GATE,
    )
    assert len(gate.carrier_names) == 23
    assert gate.real_child_terrain_ready is False
    assert gate.receipt["real_child_product_gate"]["status"] == (
        "BLOCKED_PENDING_SOURCE_COORDINATE_TERRAIN_PRODUCER"
    )
