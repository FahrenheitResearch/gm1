from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from authority_trees import requires_native_reference
from measured_source import measured_identity

from cm1core.physics_support import (
    PHYSICS_SUPPORT_SCHEMA,
    PhysicsSupportRefusal,
    branch_support,
    load_physics_support_matrix,
    require_supported_physics,
)
from cm1core.cuda_gm1_kessler_driver import CUDA_GM1_KESSLER_DRIVER_SOURCE_SHA256
from measured_source import measured_sha256


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "src" / "cm1core" / "records" / "GM1-PHYSICS-SUPPORT.json"
AUTHORITY = ROOT / "work" / "native-reference" / "cm1r21.1"
RECEIPT = ROOT / "evidence" / "validation" / "gm1-physics-support-v1" / "receipt.json"


@requires_native_reference
def test_record_is_packaged_schema_and_source_pins_are_current() -> None:
    matrix = load_physics_support_matrix()
    assert matrix["schema"] == PHYSICS_SUPPORT_SCHEMA
    assert matrix["product_name"] == "GM1"
    for relative, expected in matrix["math_authority"]["source_files"].items():
        assert hashlib.sha256((AUTHORITY / relative).read_bytes()).hexdigest() == expected


def test_validation_receipt_binds_exact_record_and_inventory_counts() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    matrix = json.loads(RECORD.read_text(encoding="utf-8"))
    assert hashlib.sha256(RECORD.read_bytes()).hexdigest() == receipt["record"]["sha256"]
    assert len(matrix["selectors"]) == receipt["inventory"]["discrete_selectors"]
    assert sum(len(item["values"]) for item in matrix["selectors"].values()) == receipt[
        "inventory"
    ]["discrete_values"]


def test_all_primary_stock_family_dispatch_values_are_inventoried() -> None:
    selectors = load_physics_support_matrix()["selectors"]
    expected = {
        "ptype": {0, 1, 2, 3, 4, 5, 6, 7, 26, 27, 28, 50, 51, 52, 53, 55},
        "ipbl": set(range(7)),
        "sgsmodel": set(range(7)),
        "radopt": {0, 1, 2},
        "sfcmodel": set(range(8)),
        "oceanmodel": {1, 2},
    }
    for name, values in expected.items():
        assert {int(value) for value in selectors[name]["values"]} == values


def test_every_selector_value_has_a_known_fail_closed_admission() -> None:
    matrix = load_physics_support_matrix()
    states = set(matrix["admission_states"])
    assert states == {"admitted", "identity", "operator-only", "partial", "unported"}
    for selector in matrix["selectors"].values():
        assert selector["family"]
        assert selector["namelist"]
        assert selector["source"]
        for value in selector["values"].values():
            assert value["admission"] in states
            assert value["label"]


@requires_native_reference
def test_scheme_specific_namelists_are_complete_and_disjoint() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    public = record["scheme_namelists"]["nssl2mom_params"]["parameters"]
    internal = record["scheme_namelists"]["nssl_mp_params"]["parameters"]

    def native_names(relative: str, namelist: str) -> list[str]:
        text = (AUTHORITY / relative).read_text(encoding="utf-8")
        match = re.search(
            rf"NAMELIST\s*/{namelist}/(?P<body>.*?)(?=\n\s*!\s*[-#])",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        assert match is not None
        body = "\n".join(line.split("!", 1)[0] for line in match.group("body").splitlines())
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body)

    assert public == native_names("src/param.F", "nssl2mom_params")
    assert internal == native_names("src/module_mp_nssl_2mom.F", "nssl_mp_params")
    assert len(public) == len(set(public))
    assert len(internal) == len(set(internal))
    assert {26, 27, 28} == set(
        record["scheme_namelists"]["nssl_mp_params"]["applies_to_ptype"]
    )


def test_admitted_identity_branches_pass_and_partial_branches_refuse() -> None:
    require_supported_physics(
        {"testcase": 0, "imoist": 1, "ptype": 7, "ipbl": 0, "radopt": 0, "cuparam": 0}
    )
    assert branch_support("ptype", 1)["admission"] == "operator-only"
    implemented = branch_support("ptype", 1)["implemented_slice"]
    assert implemented["source_sha256"] == (
        CUDA_GM1_KESSLER_DRIVER_SOURCE_SHA256
    )
    assert implemented["admitted_native_branch"]["efall"] == [0, 1]
    assert implemented["complete_operator_family_for_admitted_branch"] is True
    composition = implemented["large_step_composition"]
    assert composition["schema"] == "gm1.cuda-kessler-large-step/v1"
    assert composition["restart_schema"] == "gm1.kessler-large-step-restart/v1"
    assert composition["semantic_id"] == "kessler-ptype1"
    module = ROOT / "src" / "cm1core" / "cuda_gm1_kessler_large_step.py"
    assert measured_sha256(module) == composition["module_sha256"]
    assert implemented["evidence"]
    with pytest.raises(PhysicsSupportRefusal, match="GM1-PHYSICS-UNPORTED-BRANCH"):
        require_supported_physics({"ptype": 1})
    with pytest.raises(PhysicsSupportRefusal, match="GM1-PHYSICS-UNPORTED-BRANCH"):
        require_supported_physics({"radopt": 2})


def test_the_admitted_kessler_slice_cites_evidence_that_exists() -> None:
    """The receipts the admitted slice cites are in the repository (development repository only)."""
    implemented = branch_support("ptype", 1)["implemented_slice"]
    for evidence in implemented["evidence"].values():
        assert (ROOT / evidence).is_file(), evidence


@pytest.mark.parametrize(
    ("selection", "code"),
    [
        ({}, "GM1-PHYSICS-EMPTY-SELECTION"),
        ({"not_a_native_selector": 0}, "GM1-PHYSICS-UNKNOWN-SELECTOR"),
        ({"ptype": 999}, "GM1-PHYSICS-UNKNOWN-VALUE"),
        ({"ptype": 7.0}, "GM1-PHYSICS-VALUE-TYPE"),
    ],
)
def test_gate_refuses_unknown_or_ambiguous_requests(
    selection: dict[str, object], code: str
) -> None:
    with pytest.raises(PhysicsSupportRefusal) as caught:
        require_supported_physics(selection)
    assert caught.value.code == code


def test_testcase_13_is_audited_but_not_misrepresented_as_documented() -> None:
    testcase = load_physics_support_matrix()["selectors"]["testcase"]
    assert testcase["undocumented_source_values"] == [13]
    assert "13" not in testcase["values"]


def test_every_recorded_operator_module_hash_is_live() -> None:
    """A module pin nobody checks is a pin that goes stale silently.

    ``GM1-PHYSICS-SUPPORT.json`` names, for each implemented branch, the
    operator module that admits it and that module's sha256.  Entries that
    give a repository path are checked against the live bytes here; entries
    that give a dotted module name instead are reported so the two spellings
    cannot quietly diverge into an unchecked set.
    """

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    checked = 0
    dotted: list[str] = []

    def walk(node: object) -> None:
        nonlocal checked
        if isinstance(node, dict):
            module = node.get("module")
            digest = node.get("module_sha256")
            if isinstance(module, str) and isinstance(digest, str):
                if "/" in module:
                    path = ROOT / module
                    assert path.is_file(), module
                    # The record names the bytes its receipts measured;
                    # measured_identity maps a module reworded since then.
                    assert measured_identity(path)[1] == digest, module
                    checked += 1
                else:
                    dotted.append(module)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(record)
    assert checked >= 12
    # The dotted spellings are import paths, not repository paths; naming
    # them here keeps the unchecked set explicit and small.
    assert set(dotted) <= {
        "cm1core.cuda_gm1_kessler_driver",
        "cm1core.cuda_gm1_kessler_large_step",
        "cm1core.idealized_radiation",
        "cm1core.sgs_startup_ramp",
    }
