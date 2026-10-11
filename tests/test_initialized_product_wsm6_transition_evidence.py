from __future__ import annotations

import hashlib
import json
import struct

from sealed_evidence import assert_recaptured_or_owed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "evidence" / "validation" / "initialized-product-wsm6-transition-v1"
RECEIPT = DECK / "corrected-real-crop-sm86.json"
AUDIT_PNG = DECK / "corrected-real-crop-sm86.png"
MICROPHYSICS_PNG = DECK / "corrected-real-wsm6-microphysics.png"
FULL_DECK = (
    ROOT / "evidence" / "validation" / "initialized-product-wsm6-full-v1"
)
FULL_RECEIPT = FULL_DECK / "full-domain-sm86.json"
FULL_INVENTORY = FULL_DECK / "artifact-inventory.json"
FULL_TRANSITION_PNG = FULL_DECK / "full-real-wsm6-transition.png"
FULL_REFLECTIVITY_PNG = FULL_DECK / "full-real-wsm6-reflectivity.png"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_corrected_real_public_wsm6_and_durable_restart_gates_pass() -> None:
    receipt = _receipt()
    assert receipt["schema"] == (
        "gpuwm-cm1.initialized-product-eqtset2-wsm6-sm86-evidence/v1"
    )
    assert receipt["result"] == "passed"
    assert receipt["execution"]["device"] == "NVIDIA GeForce RTX 3080"
    assert receipt["execution"]["compute_capability"] == "86"
    assert receipt["execution"]["shared_gpu_mutex_owned_for_all_cuda"] is True
    assert receipt["execution"]["shared_gpu_mutex_released"] is True
    assert all(receipt["gates"].values())
    assert receipt["orchestration"]["python_scientific_input_transforms"] == 0
    assert receipt["orchestration"]["private_wsm6_module_imported_by_runner"] is False
    assert receipt["engine"]["admitted_audited_surface"] is True
    assert receipt["source"]["crop_origin_yx"] == [142, 142]
    assert receipt["source"]["crop_count_yx"] == [16, 16]
    assert receipt["source"]["full_domain"] is False
    assert "not a forecast" not in receipt["claim_limit"].lower()
    assert "no halos" in receipt["claim_limit"].lower()


def test_all_21_arrays_survive_rust_and_cuda_resume_raw_exact() -> None:
    receipt = _receipt()
    restart = receipt["restart"]
    assert restart["array_count"] == 21
    assert restart["durable_rust_serialization"] is True
    assert restart["pre_apply_staging_raw_exact"] is True
    assert restart["uninterrupted_resumed_second_raw_exact"] is True
    assert (
        restart["checkpoint_combined_sha256"]
        == restart["staged_combined_sha256"]
    )
    assert (
        restart["uninterrupted_second_combined_sha256"]
        == restart["resumed_second_combined_sha256"]
    )
    assert restart["checkpoint_artifact_sha256"] == (
        "d0bb7f1efde3048c3d381d243f672e17f5e0561f373942b69dbbc6c0f2ad9a25"
    )
    assert restart["payload_sha256"] == (
        "00a20ac989286262a87f37917deb759b0c49998da42dd2ef304fc7e69fee1511"
    )
    assert restart["restore_manifest_sha256"] == (
        "8190b1451bdb83485bb52670d80df23141bb8a1a9baf409531b9a311dfa1ba69"
    )
    assert restart["rust"]["inspect_equal_write"] is True
    assert restart["rust"]["write_receipt"]["carrier_count"] == 21
    assert restart["rust"]["restore_receipt"]["carrier_count"] == 21
    assert restart["rust"]["write_receipt"]["payload_nbytes"] == 1_466_368
    assert restart["rust"]["restore_receipt"]["manifest_nbytes"] == 9_300


def test_real_transition_mutates_state_and_preserves_measured_geopotential_bound() -> None:
    receipt = _receipt()
    changed = receipt["first_transition"]["state_mutated_word_counts"]
    assert set(changed) == {
        "state.pp3d",
        "state.th3d",
        "state.prs",
        "state.rho",
        "state.rr",
        "state.q3d",
    }
    assert all(count > 0 for count in changed.values())
    recovery = receipt["geopotential_recovery"]
    assert recovery["zf_raw_identity"] is False
    assert recovery["word_count"] == 19_456
    assert recovery["mismatch_count"] == 2_399
    assert recovery["recovered_count"] == 17_057
    assert recovery["maximum_ulp_distance"] == 1
    assert all(
        array["finite"]
        for array in receipt["first_transition"]["arrays"].values()
    )


def test_historical_crop_audit_png_is_byte_pinned() -> None:
    receipt = _receipt()
    for name in ("png",):
        artifact = receipt["artifacts"][name]
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["nbytes"]
        assert _sha256(path) == artifact["sha256"]
    payload = AUDIT_PNG.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", payload[16:24]) == (2880, 1800)


def test_nonzero_microphysics_visualization_is_byte_pinned() -> None:
    payload = MICROPHYSICS_PNG.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", payload[16:24]) == (2880, 1800)
    assert hashlib.sha256(payload).hexdigest() == (
        "d6328cb8a2ece4da53d3c5d33c1a16e41eba722fbb05e486544ebd6b53f3048e"
    )
    renderer = ROOT / "tools" / "render_initialized_product_wsm6_transition.py"
    assert _sha256(renderer) == (
        "7c0c8d31bb8d63f6f0506307a2387a58bca396914a4db1364c8976e3201c0f2b"
    )


def test_remaining_red_gates_are_explicit_without_durable_greenwashing() -> None:
    receipt = _receipt()
    assert receipt["red_gates"] == {
        "durable_transition_restart_serialization": False,
        "efall": True,
        "forecast": True,
        "halos_and_open_boundary_history": True,
        "idiss": True,
        "model_validation": True,
        "product_admission": True,
        "reflectivity_consumer": True,
        "tornado": True,
    }
    resumed = receipt["resumed_second_receipt"]
    assert resumed["restart"]["durable_serialization"] is True
    assert resumed["red_gates"]["durable_transition_restart_serialization"] is False


def test_full_real_wsm6_transition_and_rust_resume_gates_pass() -> None:
    receipt = json.loads(FULL_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == (
        "gpuwm-cm1.initialized-product-eqtset2-wsm6-sm86-evidence/v1"
    )
    assert receipt["result"] == "passed"
    assert receipt["source"]["full_domain"] is True
    assert receipt["source"]["crop_origin_yx"] == [0, 0]
    assert receipt["source"]["crop_count_yx"] == [300, 300]
    assert receipt["execution"]["device"] == "NVIDIA GeForce RTX 3080"
    assert receipt["execution"]["compute_capability"] == "86"
    assert receipt["execution"]["shared_gpu_mutex_owned_for_all_cuda"] is True
    assert receipt["execution"]["shared_gpu_mutex_released"] is True
    assert receipt["engine"]["admitted_audited_surface"] is True
    assert all(receipt["gates"].values())
    assert receipt["orchestration"]["python_scientific_input_transforms"] == 0
    assert receipt["orchestration"]["private_wsm6_module_imported_by_runner"] is False
    claim = receipt["claim_limit"].lower()
    assert "300x300x75" in claim
    assert "not a coupled dycore trajectory" in claim


def test_full_real_reflectivity_is_consumed_once_and_transient() -> None:
    receipt = json.loads(FULL_RECEIPT.read_text(encoding="utf-8"))
    reflectivity = receipt["reflectivity"]
    assert reflectivity == {
        "consumer": "gpuwm.core.refl.consume_refl_10cm",
        "dtype": "<f4",
        "finite": True,
        "maximum_dbz": 10.287071228027344,
        "minimum_dbz": -35.0,
        "nbytes": 27_000_000,
        "nonzero_count": 6_750_000,
        "raw_sha256": (
            "a90667a8c9b43d6e4a739a41894773be8222c1725999b6fc90ba43f380696bc0"
        ),
        "restart_state": False,
        "shape": [75, 300, 300],
        "transient_consumed_once": True,
        "units": "dBZ",
    }
    assert receipt["red_gates"]["reflectivity_consumer"] is False
    assert (
        receipt["resumed_second_receipt"]["red_gates"]["reflectivity_consumer"]
        is False
    )


def test_full_real_21_array_rust_restart_and_resume_are_raw_exact() -> None:
    receipt = json.loads(FULL_RECEIPT.read_text(encoding="utf-8"))
    restart = receipt["restart"]
    assert restart["array_count"] == 21
    assert restart["durable_rust_serialization"] is True
    assert restart["pre_apply_staging_raw_exact"] is True
    assert restart["uninterrupted_resumed_second_raw_exact"] is True
    assert restart["checkpoint_combined_sha256"] == (
        restart["staged_combined_sha256"]
    )
    assert restart["uninterrupted_second_combined_sha256"] == (
        restart["resumed_second_combined_sha256"]
    )
    assert restart["checkpoint_artifact_sha256"] == (
        "8f0d1e30defa8112e87512bfda7d344cfbc0cb56c5562f2b742cb203c6f7126e"
    )
    assert restart["payload_sha256"] == (
        "849c2e93d500387312e97a502f43ce555d96d6868b52b94fc46ac9f599893766"
    )
    assert restart["restore_manifest_sha256"] == (
        "2cadc50fc92bb8a472733b8235b5bcc84bc69dbf37e801d4968bb8b442c29010"
    )
    assert restart["rust"]["inspect_equal_write"] is True
    assert restart["rust"]["write_receipt"]["payload_nbytes"] == 515_520_000
    assert restart["rust"]["restore_receipt"]["carrier_count"] == 21


def test_full_real_sources_and_both_pngs_are_byte_pinned() -> None:
    """full-domain-sm86.json is a measured sm86 receipt.

    Its source-provenance entries were briefly rewritten to the post-September
    bytes.  That asserts the measured run was produced by source that did not
    exist when it ran, so they are back at their captured values and the three
    reviewed sources are named as re-capture debt instead.  The PNG, runner
    and every other entry is still compared exactly.
    """

    receipt = json.loads(FULL_RECEIPT.read_text(encoding="utf-8"))
    owed = {
        "bridge_source": "4e1909d Prove full real WSM6 restart transition",
        "component_source": "775c8b7 Add the hash-checked ArWen WSM6 column-capacity profile seam",
        "transition_source": "4e1909d Prove full real WSM6 restart transition",
    }
    for name, artifact in receipt["artifacts"].items():
        path = ROOT / artifact["path"]
        if name in owed:
            assert_recaptured_or_owed(path, artifact["sha256"], reviewed_by=owed[name])
            continue
        assert path.stat().st_size == artifact["nbytes"]
        assert _sha256(path) == artifact["sha256"]
    runner = (ROOT / receipt["artifacts"]["runner"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "gpuwm.core.wsm6" not in runner
    for path, dimensions in (
        (FULL_TRANSITION_PNG, (2880, 1800)),
        (FULL_REFLECTIVITY_PNG, (2415, 1995)),
    ):
        payload = path.read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", payload[16:24]) == dimensions

    inventory = json.loads(FULL_INVENTORY.read_text(encoding="utf-8"))
    assert inventory["fixture_id"] == (
        "CORRECTED-REAL-INITIALIZED-PRODUCT-WSM6-FULL-V1"
    )
    # The inventory carries the same re-capture debt as the receipt above, for
    # the same reason and by the same rule: the measured sm86 run executed the
    # bytes it names, so rewriting its pins would claim the run was produced by
    # source that did not exist when it ran.  The WSM6 binding is owed because
    # the component adapter was generalized over CM1's ptype (src/param.F builds
    # one q3d table per ptype), which moved most of its body into
    # src/cm1core/arwen_column_component.py; nothing about the executed WSM6
    # configuration changed, which tests/test_engine_capacity.py pins field by
    # field.
    inventory_owed = {
        "src/cm1core/arwen_wsm6_component.py":
            "the ptype-generic column adapter split, which moved this binding's "
            "body into src/cm1core/arwen_column_component.py",
        "src/cm1core/eqtset2_wsm6_bridge.py":
            "4e1909d Prove full real WSM6 restart transition",
        "src/cm1core/initialized_product_wsm6_transition.py":
            "4e1909d Prove full real WSM6 restart transition",
    }
    for artifact in inventory["artifacts"]:
        prefix, path_text = artifact["path"].split(":", 1)
        assert prefix == "source-distribution"
        path = ROOT / path_text
        if path_text in inventory_owed:
            assert_recaptured_or_owed(
                path, artifact["sha256"], reviewed_by=inventory_owed[path_text]
            )
            continue
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]
