from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from cm1core.gm1_child_initialization import (
    load_gm1_child_initialization_plan,
    load_gm1_child_producer_requirements,
)
from cm1core.gm1_child_producer_interfaces import (
    AUTHORITY_SOURCE_SHA256,
    GM1_CHILD_PRODUCER_INTERFACES_SCHEMA,
    GM1_CHILD_TERRAIN_INPUT_SHA256,
    GM1_CHILD_TERRAIN_METRIC_API,
    build_gm1_child_producer_interfaces,
    validate_gm1_child_producer_authority,
)
from authority_trees import requires_native_reference


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "evidence/validation/gm1-t300-child-remap-preflight-v1/validated-parent-plan.json"
)
REQUIREMENTS = (
    ROOT
    / "evidence/validation/gm1-t300-child-remap-preflight-v1/producer-requirements.json"
)
FLAT_GRID = ROOT / "evidence/validation/stretch-z1-child-15km-v1/receipt.json"
# CM1_AUTHORITY_ROOT arrives here as a symlink (authority_trees.link_supplied_assets),
# and the validator refuses a symlinked root by design; hand it the tree itself.
AUTHORITY = (ROOT / "work/native-reference/cm1r21.1").resolve()


def test_interface_import_is_cuda_cold() -> None:
    command = (
        "import sys; import cm1core.gm1_child_producer_interfaces; "
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


@requires_native_reference
def test_official_authority_and_all_three_interfaces_are_exact() -> None:
    assert validate_gm1_child_producer_authority(AUTHORITY) == AUTHORITY.resolve()
    assert len(AUTHORITY_SOURCE_SHA256) == 8
    plan = load_gm1_child_initialization_plan(PLAN, FLAT_GRID)
    requirements = load_gm1_child_producer_requirements(plan, REQUIREMENTS)
    interfaces = build_gm1_child_producer_interfaces(
        plan,
        requirements,
        authority_root=AUTHORITY,
    )
    receipt = interfaces.as_dict()
    assert receipt["schema"] == GM1_CHILD_PRODUCER_INTERFACES_SCHEMA
    assert receipt["numerical_child_output_allowed"] is False
    assert set(receipt["interfaces"]) == {
        "terrain_stretch_composition",
        "hydrostatic_and_open_boundary",
        "geopotential",
    }
    terrain = receipt["interfaces"]["terrain_stretch_composition"]
    assert terrain["status"] == "READY_FOR_STRICT_REAL_TERRAIN_LOADER_BINDING"
    assert terrain["consumer_api"] == GM1_CHILD_TERRAIN_METRIC_API
    assert terrain["input_shape"] == [300, 300]
    assert terrain["input_nbytes"] == 360000
    assert terrain["exact_input_identity"] == dict(GM1_CHILD_TERRAIN_INPUT_SHA256)
    assert len(terrain["required_carriers"]) == 23
    hydrostatic = receipt["interfaces"]["hydrostatic_and_open_boundary"]
    assert len(hydrostatic["required_carriers"]) == 50
    assert "preserve restart-grid-interpolated ppi/tha/qa" in hydrostatic[
        "source_order"
    ][1]
    assert all(
        "top-down ibalance=1 perturbation pressure" not in step
        for step in hydrostatic["source_order"]
    )
    assert hydrostatic["source_order"][4] == (
        "psolver=3 overwrites independently remapped ppx from ppi"
    )
    assert hydrostatic["open_boundary_invariants"] == {
        "all_four_boundaries_open": True,
        "cold_start_history_nonzero_words": 0,
        "parent_radiation_history_reuse_allowed": False,
        "source_valid_child_halos_required": True,
    }
    assert len(receipt["interfaces"]["geopotential"]["must_bind"]) == 5
