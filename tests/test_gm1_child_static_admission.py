from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cm1core.gm1_child_initialization import (
    load_gm1_child_initialization_plan,
    load_gm1_child_producer_requirements,
)
from cm1core.gm1_child_producer_interfaces import (
    build_gm1_child_producer_interfaces,
)
from cm1core.gm1_child_static_admission import (
    GM1_CHILD_STATIC_ADMISSION_SCHEMA,
    GM1_CHILD_STATIC_COMBINED_SHA256,
    GM1_CHILD_STATIC_VALIDATION_SHA256,
    GM1_CHILD_STATIC_ZF_SHA256,
    GM1_CHILD_STATIC_ZH_SHA256,
    load_gm1_child_static_admission,
)
from host_portability import require_workspace_contained
from authority_trees import requires_native_reference, requires_work


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "evidence/validation/gm1-t300-child-remap-preflight-v1/validated-parent-plan.json"
REQUIREMENTS = ROOT / "evidence/validation/gm1-t300-child-remap-preflight-v1/producer-requirements.json"
FLAT_GRID = ROOT / "evidence/validation/stretch-z1-child-15km-v1/receipt.json"
AUTHORITY = ROOT / "work/native-reference/cm1r21.1"
TERRAIN = ROOT / "work/gm1-t300-child-terrain-v1"
VALIDATION = ROOT / "evidence/validation/gm1-child-terrain-metrics-v1/validation-receipt.json"


def _inputs():
    plan = load_gm1_child_initialization_plan(PLAN, FLAT_GRID)
    requirements = load_gm1_child_producer_requirements(plan, REQUIREMENTS)
    interfaces = build_gm1_child_producer_interfaces(
        plan, requirements, authority_root=AUTHORITY
    )
    return plan, requirements, interfaces


@requires_native_reference
@requires_work("gm1-t300-child-terrain-v1")
def test_import_and_real_static_admission_are_cuda_cold_and_exact() -> None:
    command = (
        "import sys; import cm1core.gm1_child_static_admission; "
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
    require_workspace_contained(ROOT, "work")
    plan, requirements, interfaces = _inputs()
    admitted = load_gm1_child_static_admission(
        plan,
        requirements,
        interfaces,
        workspace_root=ROOT,
        terrain_directory=TERRAIN,
        validation_receipt=VALIDATION,
    )
    assert admitted.schema == GM1_CHILD_STATIC_ADMISSION_SCHEMA
    assert admitted.validation_sha256 == GM1_CHILD_STATIC_VALIDATION_SHA256
    assert admitted.receipt["cuda_static_metrics"]["combined_output_sha256"] == GM1_CHILD_STATIC_COMBINED_SHA256
    assert hashlib.sha256(admitted.zh_path.read_bytes()).hexdigest() == GM1_CHILD_STATIC_ZH_SHA256
    assert hashlib.sha256(admitted.zf_path.read_bytes()).hexdigest() == GM1_CHILD_STATIC_ZF_SHA256
    assert len(admitted.carrier_names) == 23
    assert admitted.atmospheric_child_ready is False
    assert set(admitted.png_paths) == {
        "gm1-real-child-terrain-columns.png",
        "gm1-real-child-terrain-metrics.png",
    }


@requires_native_reference
@requires_work("gm1-t300-child-terrain-v1")
def test_tampered_validation_sha_is_refused(tmp_path: Path) -> None:
    plan, requirements, interfaces = _inputs()
    tampered = tmp_path / "validation.json"
    tampered.write_bytes(VALIDATION.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        load_gm1_child_static_admission(
            plan,
            requirements,
            interfaces,
            workspace_root=ROOT,
            terrain_directory=TERRAIN,
            validation_receipt=tampered,
        )


@requires_native_reference
@requires_work("gm1-t300-child-terrain-v1")
def test_semantically_false_pass_is_refused_even_when_resealed(tmp_path: Path) -> None:
    plan, requirements, interfaces = _inputs()
    value = json.loads(VALIDATION.read_bytes())
    value["atmospheric_child_ready"] = True
    raw = json.dumps(value, allow_nan=False, sort_keys=True).encode()
    altered = tmp_path / "altered.json"
    altered.write_bytes(raw)
    with pytest.raises(ValueError, match="classification changed"):
        load_gm1_child_static_admission(
            plan,
            requirements,
            interfaces,
            workspace_root=ROOT,
            terrain_directory=TERRAIN,
            validation_receipt=altered,
            expected_validation_sha256=hashlib.sha256(raw).hexdigest(),
        )
