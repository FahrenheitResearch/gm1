from __future__ import annotations

import hashlib
import json
from pathlib import Path
from measured_source import measured_identity, unlabel


ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "evidence" / "validation" / "cuda-eqtset2-idiss-v1"
RECEIPT = DECK / "sm86.json"
INVENTORY = DECK / "artifact-inventory.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_native_oracle_and_sm86_consumer_are_raw_word_exact() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "gpuwm-cm1.cuda-eqtset2-idiss-sm86-evidence/v1"
    assert receipt["result"] == "passed"
    assert receipt["execution"]["device"] == "NVIDIA GeForce RTX 3080"
    assert receipt["execution"]["compute_capability"] == "86"
    assert receipt["execution"]["shared_gpu_mutex_released"] is True
    assert unlabel(receipt["oracle"]["fixture_id"]) == "O-EQTSET2-IDISS-V1"
    assert receipt["oracle"]["comparison_contract"] == {
        "max_ulp": 0,
        "mode": "raw-bit",
    }
    comparison = receipt["comparison"]
    assert comparison["exact"] is True
    assert comparison["mismatch_count"] == 0
    assert comparison["word_count"] == 24
    assert comparison["expected_raw_sha256"] == comparison["actual_raw_sha256"]
    assert comparison["actual_raw_sha256"] == (
        "f60291ba81d0085f7a33a8214ee2a31c259109bc458c5b870ff6c97b100a1650"
    )


def test_idiss_consumer_binding_and_pre_rk_cadence_are_sealed() -> None:
    receipt = _receipt()
    consumer = receipt["consumer"]["receipt"]
    assert consumer["consumer_abi"] == "gpuwm-cm1.eqtset2-idiss-consumer/v1"
    assert consumer["cadence"] == "pre-RK before prior microphysics tendency consume"
    assert consumer["configuration"] == {
        "bottom_boundary": 1,
        "equation_set": 2,
        "iice": 1,
        "imoist": 1,
        "nx": 3,
        "ny": 2,
        "nz": 4,
        "species_order": ["qv", "qc", "qr", "qi", "qs", "qg"],
    }
    assert consumer["binding"]["dissten_shape"] == [5, 2, 3]
    assert consumer["binding"]["supplied_live_dissten"] is True
    assert consumer["binding"]["host_copy"] is False
    assert consumer["binding"]["thten1_mutated_in_place"] is True
    assert consumer["synchronization"]["explicit_synchronize"] is False
    assert consumer["synchronization"]["ready_event_recorded"] is True


def test_idiss_negative_control_and_remaining_red_gates_are_explicit() -> None:
    receipt = _receipt()
    assert receipt["negative_control"] == {
        "changed_output_words": 24,
        "detected": True,
        "kind": "double every supplied dissten z-face word",
        "raw_sha256": (
            "93fd032440a980b9eb13b4d8362c72324fc61ed6f477c12acc7a4c5659e1bcb1"
        ),
    }
    assert receipt["red_gates"] == {
        "coupled_step": True,
        "efall_1": True,
        "live_sgs_dissten_producer_missing": True,
        "trajectory": True,
    }
    claim = receipt["claim_limit"].lower()
    assert "supplied-dissten" in claim
    assert "no sgs dissten producer" in claim
    assert "efall=1" in claim


def test_idiss_receipt_and_inventory_artifacts_are_byte_pinned() -> None:
    """Every artifact is still the executed revision, or says why it is not.

    ``artifacts`` is the manifest of what actually ran on the recorded
    sm_86 device and is never rewritten.  A source file that legitimately
    changes afterwards is listed once in ``source_evolution``, which carries
    the executed hash, the current hash and the live test that re-proves the
    arithmetic; anything else must still match byte for byte.
    """

    assert _sha256(RECEIPT) == (
        "1873306118eac27d80ff5c5c09208dc726838181ffd62aa85e37daf9ca0001a7"
    )
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert unlabel(inventory["fixture_id"]) == "O-EQTSET2-IDISS-V1"
    evolved = {
        entry["path"]: entry for entry in inventory.get("source_evolution", ())
    }
    for artifact in inventory["artifacts"]:
        prefix, path_text = artifact["path"].split(":", 1)
        assert prefix == "source-distribution"
        path = ROOT / path_text
        assert path.is_file()
        entry = evolved.pop(artifact["path"], None)
        if entry is None:
            assert measured_identity(path) == (artifact["bytes"], artifact["sha256"])
            continue
        # The executed manifest and the evolution record must agree on what
        # ran, or the deck would no longer name one revision.
        assert entry["executed_sha256"] == artifact["sha256"]
        assert entry["executed_bytes"] == artifact["bytes"]
        assert entry["current_sha256"] != artifact["sha256"]
        # A module reworded since (tests/measured_source.py) is compared as the record measured it.
        assert measured_identity(path) == (entry["current_bytes"], entry["current_sha256"])
        assert entry["arithmetic_unchanged"] is True
        assert entry["change"] and entry["changed_regions"]
        assert "test_cpu_scalar_is_raw_word_exact" in entry["arithmetic_proof"]
        assert entry["revalidation_required_before_claiming"]
    # Nothing may claim to have evolved that the deck never shipped.
    assert evolved == {}
