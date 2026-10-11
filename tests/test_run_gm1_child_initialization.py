from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_runner_import_is_cuda_cold() -> None:
    command = (
        "import sys; import tools.run_gm1_child_initialization; "
        "assert not any(name == 'cupy' or name.startswith('cupy.') "
        "for name in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "src")))
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_runner_reports_exact_blockers_without_creating_arrays() -> None:
    module = importlib.import_module("tools.run_gm1_child_initialization")
    args = type(
        "Args",
        (),
        {
            "plan": module.PLAN,
            "plan_sha256": module.GM1_CHILD_PLAN_SHA256,
            "flat_grid_receipt": module.FLAT_GRID,
            "flat_grid_receipt_sha256": module.GM1_CHILD_FLAT_GRID_RECEIPT_SHA256,
            "producer_requirements": module.REQUIREMENTS,
            "producer_requirements_sha256": (
                module.GM1_CHILD_PRODUCER_REQUIREMENTS_SHA256
            ),
            "terrain_operator_gate": module.TERRAIN_OPERATOR_GATE,
            "terrain_operator_gate_sha256": (
                module.GM1_CHILD_TERRAIN_OPERATOR_GATE_SHA256
            ),
        },
    )()
    receipt = module.run(args)
    assert receipt["status"] == "REFUSED_AWAITING_REAL_NUMERICAL_PRODUCERS"
    assert receipt["numerical_child_output_allowed"] is False
    assert set(receipt["producer_status"].values()) == {"REQUIRED_NOT_YET_BOUND"}
    assert receipt["carrier_contract"] == {
        "parent_carriers": 37,
        "remap_3d": 22,
        "remap_surface": 7,
        "recompute": 4,
        "boundary_initialize": 4,
        "terrain_stretch": 23,
    }
    assert receipt["no_numerical_weather_arrays_written"] is True
