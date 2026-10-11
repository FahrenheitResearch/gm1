from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from cm1core.gm1_child_atmosphere_admission import load_gm1_child_atmosphere
from cm1core.gm1_child_auxiliary_interfaces import (
    AUTHORITY_SOURCE_SHA256,
    BASE_SOURCE_PROFILES,
    FRESH_TKE_OPERATOR,
    GM1_CHILD_AUXILIARY_INTERFACES_SCHEMA,
    LIVE_DENSITY_SHAPES,
    TERRAIN_OPEN_CARRIER_SHAPES,
    build_gm1_child_auxiliary_interfaces,
    load_gm1_child_base_source,
    require_gm1_child_auxiliary_numerics,
)
from cm1core.gm1_child_initialization import (
    load_gm1_child_initialization_plan,
    load_gm1_child_producer_requirements,
)
from cm1core.gm1_child_producer_interfaces import (
    build_gm1_child_producer_interfaces,
)
from cm1core.gm1_child_static_admission import load_gm1_child_static_admission
from host_portability import require_workspace_contained
from authority_trees import requires_native_reference, requires_work


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "evidence/validation/gm1-t300-child-remap-preflight-v1/"
    "validated-parent-plan.json"
)
REQUIREMENTS = (
    ROOT
    / "evidence/validation/gm1-t300-child-remap-preflight-v1/"
    "producer-requirements.json"
)
FLAT_GRID = ROOT / "evidence/validation/stretch-z1-child-15km-v1/receipt.json"
AUTHORITY = ROOT / "work/native-reference/cm1r21.1"
TERRAIN = ROOT / "work/gm1-t300-child-terrain-v1"
STATIC_VALIDATION = (
    ROOT
    / "evidence/validation/gm1-child-terrain-metrics-v1/validation-receipt.json"
)
ATMOSPHERE = ROOT / "work/gm1-t300-child-atmosphere-v1"
BASE_SOURCE = ROOT / "work/initialized-product-hrrr-f02-roi-v1/full"


@pytest.fixture(scope="module")
def auxiliary_interfaces():
    require_workspace_contained(ROOT, "work")
    plan = load_gm1_child_initialization_plan(PLAN, FLAT_GRID)
    requirements = load_gm1_child_producer_requirements(plan, REQUIREMENTS)
    producer_interfaces = build_gm1_child_producer_interfaces(
        plan, requirements, authority_root=AUTHORITY
    )
    static = load_gm1_child_static_admission(
        plan,
        requirements,
        producer_interfaces,
        workspace_root=ROOT,
        terrain_directory=TERRAIN,
        validation_receipt=STATIC_VALIDATION,
    )
    atmosphere = load_gm1_child_atmosphere(ATMOSPHERE)
    base = load_gm1_child_base_source(BASE_SOURCE)
    return build_gm1_child_auxiliary_interfaces(
        static,
        atmosphere,
        base,
        workspace_root=ROOT,
        authority_root=AUTHORITY,
    )


def test_import_is_cuda_cold() -> None:
    command = (
        "import sys; import cm1core.gm1_child_auxiliary_interfaces; "
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
@requires_work(
    "gm1-t300-child-terrain-v1",
    "gm1-t300-child-atmosphere-v1",
    "initialized-product-hrrr-f02-roi-v1/full",
)
def test_exact_real_inputs_source_order_and_carrier_contracts(
    auxiliary_interfaces,
) -> None:
    receipt = auxiliary_interfaces.as_dict()
    assert receipt["schema"] == GM1_CHILD_AUXILIARY_INTERFACES_SCHEMA
    assert receipt["numerical_child_output_allowed"] is False
    assert receipt["real_inputs"]["synthetic_data"] is False
    assert len(AUTHORITY_SOURCE_SHA256) == 7
    assert receipt["stage_order"] == [
        "base_terrain_hydrostatic",
        "restart_state_and_family_boundaries",
        "a_to_current_copies",
        "live_moist_density",
        "psolver3_pressure_helper",
        "child_nm_deformation",
        "fresh_turbsmag_tke",
        "four_open_boundary_history",
        "zf_total_geopotential",
    ]
    restart = receipt["stages"]["restart_state_and_family_boundaries"]
    assert restart["preserve_without_rebalance"] == ["ppi", "tha", "qa"]
    copies = receipt["stages"]["a_to_current_copies"]
    assert "forbidden" in copies
    pressure = receipt["stages"]["psolver3_pressure_helper"]
    assert pressure["raw_bit_copy"] == "ppi->ppx"
    assert pressure["overwrites_independently_remapped_ppx"] is True
    assert len(TERRAIN_OPEN_CARRIER_SHAPES) == 50
    assert TERRAIN_OPEN_CARRIER_SHAPES["pi0"] == (77, 306, 306)
    assert TERRAIN_OPEN_CARRIER_SHAPES["ppi"] == (77, 302, 302)
    assert TERRAIN_OPEN_CARRIER_SHAPES["ua"] == (77, 306, 307)
    assert TERRAIN_OPEN_CARRIER_SHAPES["va"] == (77, 307, 306)
    assert TERRAIN_OPEN_CARRIER_SHAPES["wa"] == (78, 306, 306)
    assert TERRAIN_OPEN_CARRIER_SHAPES["radbcw"] == (77, 306)
    assert set(LIVE_DENSITY_SHAPES) == {"prs", "rho", "rr", "rf"}
    assert receipt["stages"]["live_moist_density"][
        "separate_payload_required"
    ] is True


@requires_native_reference
@requires_work(
    "gm1-t300-child-terrain-v1",
    "gm1-t300-child-atmosphere-v1",
    "initialized-product-hrrr-f02-roi-v1/full",
)
def test_base_profiles_and_fresh_tke_proof_are_exact_but_not_runnable(
    auxiliary_interfaces,
) -> None:
    receipt = auxiliary_interfaces.as_dict()
    assert receipt["real_inputs"]["base_source_profile_sha256"] == {
        name: sha for name, (_, sha) in BASE_SOURCE_PROFILES.items()
    }
    tke = receipt["stages"]["fresh_turbsmag_tke"]
    assert tke["status"] == "OPERATOR_PROVEN_INPUTS_BLOCKED"
    assert tke["operator"] == dict(FRESH_TKE_OPERATOR)
    assert receipt["stages"]["child_nm_deformation"]["status"] == (
        "BLOCKED_CHILD_CALCNM_AND_CALCDEF_NOT_ADMITTED"
    )
    with pytest.raises(
        ValueError, match="GM1-50M-CHILD-AUXILIARIES-UNBOUND"
    ):
        require_gm1_child_auxiliary_numerics(auxiliary_interfaces)


def test_wrong_base_source_inventory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_bytes(b"{}")
    (tmp_path / "receipt.json").write_bytes(b"{}")
    (tmp_path / "initialized-carriers.f32le").write_bytes(b"")
    (tmp_path / "extra").write_bytes(b"")
    with pytest.raises(ValueError, match="inventory changed"):
        load_gm1_child_base_source(tmp_path)
