from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from pathlib import Path

from sealed_evidence import assert_recaptured_or_owed


ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "evidence" / "validation" / "eqtset2-wsm6-bridge-v1"
RECEIPT = DECK / "eqtset2-wsm6-bridge-sm86-v1.json"
CURRENT_PRODUCT_RECEIPT = (
    ROOT
    / "evidence"
    / "validation"
    / "initialized-product-wsm6-full-v1"
    / "full-domain-sm86.json"
)
BRIDGE_SOURCE = "src/cm1core/eqtset2_wsm6_bridge.py"

# Both receipts were measured on an RTX 3080 (sm_86).  No sm_86 card is left
# to measure on, so neither can be captured again, and the bridge has been
# edited since both runs.  Each receipt's bridge pin is therefore historical:
# it names the bridge as it was in the commit that sealed the receipt, and
# that is what git can still check.
BRIDGE_RECEIPT_COMMIT = "b3f10a5b721de33276ce0a24af6b66d93ffb9628"
PRODUCT_RECEIPT_COMMIT = "4e1909d1f02e10ee405d2fac89accbd057670d92"
# The reviewed commits that changed the bridge after PRODUCT_RECEIPT_COMMIT.
BRIDGE_CHANGED_SINCE_PRODUCT_RECEIPT = (
    "1e5e9cd Open source-ordered pre-WSM6 mass correction seam; "
    "bc5859d Seal child post-final RK source order"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_bounded_cuda_bridge_receipt_passes_without_greenwashing_red_gates() -> None:
    receipt = _receipt()
    assert receipt["status"] == "passed-bounded-component-integration"
    assert receipt["execution"]["device"] == "NVIDIA GeForce RTX 3080"
    assert receipt["execution"]["compute_capability"] == "86"
    assert receipt["execution"]["repeat_count"] == 2
    assert receipt["execution"]["shared_gpu_mutex_owned_for_all_cuda"] is True
    assert receipt["execution"]["shared_gpu_mutex_released"] is True
    assert receipt["repeat_raw_word_exact"] is True
    assert all(receipt["runs"]["first"]["gates"].values())
    assert (
        receipt["runs"]["first"]["metrics"]["ptype5_pressure_relative_residual"]
        <= receipt["gates"]["ptype5_pressure_relative_residual"]
    )
    assert (
        receipt["runs"]["first"]["metrics"]["minimum_species_kg_kg-1"]
        >= (receipt["gates"]["minimum_species_kg_kg-1"])
    )
    assert receipt["red_gates"] == {
        "coupled_trajectory": True,
        "efall": True,
        "idiss": True,
        "product_admission": True,
        "reflectivity_consumer": True,
        "restart_import": True,
    }
    assert "not a coupled dycore trajectory" in receipt["claim_limit"]
    assert (
        "no idiss, efall"
        in receipt["runs"]["first"]["energy_metrics"]["coverage_limit"]
    )


def test_historical_runner_and_bridge_bytes_are_pinned_at_their_capture_commits() -> None:
    """Both sm_86 bridge pins are checked in git, where they can still be checked.

    The bridge moved after the full-domain run (see
    BRIDGE_CHANGED_SINCE_PRODUCT_RECEIPT), and there is no sm_86 device to
    measure the new bytes on.  Comparing the pin with the working tree would
    fail for good, and rewriting the pin would claim that a run measured
    bytes it never executed.  So this test checks what is still true:

    * each receipt is byte-identical to the one sealed in its commit, which
      makes that commit the source state the run was measured against;
    * the bridge in that commit has exactly the bytes and digest the receipt
      names;
    * the bridge changed between the two captures.

    The difference between the live bridge and the full-domain pin is
    reported as re-capture debt (``RecaptureOwed``), not as a failure.  Once
    the full-domain receipt is captured again, the receipt-identity check
    below fails until PRODUCT_RECEIPT_COMMIT names the new capture commit.
    """

    receipt = _receipt()
    runner_artifact = receipt["artifacts"]["runner"]
    runner_path = ROOT / runner_artifact["path"]
    assert runner_path.stat().st_size == runner_artifact["bytes"]
    assert _sha256(runner_path) == runner_artifact["sha256"]

    assert _git_blob(
        BRIDGE_RECEIPT_COMMIT, RECEIPT.relative_to(ROOT).as_posix()
    ) == RECEIPT.read_bytes()
    component_bridge = receipt["bridge"]["source"]
    assert component_bridge["path"] == BRIDGE_SOURCE
    blob = _git_blob(BRIDGE_RECEIPT_COMMIT, BRIDGE_SOURCE)
    assert len(blob) == component_bridge["bytes"]
    assert hashlib.sha256(blob).hexdigest() == component_bridge["sha256"]

    assert _git_blob(
        PRODUCT_RECEIPT_COMMIT, CURRENT_PRODUCT_RECEIPT.relative_to(ROOT).as_posix()
    ) == CURRENT_PRODUCT_RECEIPT.read_bytes()
    current = json.loads(CURRENT_PRODUCT_RECEIPT.read_text(encoding="utf-8"))
    bridge_artifact = current["artifacts"]["bridge_source"]
    assert bridge_artifact["path"] == BRIDGE_SOURCE
    blob = _git_blob(PRODUCT_RECEIPT_COMMIT, BRIDGE_SOURCE)
    assert len(blob) == bridge_artifact["nbytes"]
    assert hashlib.sha256(blob).hexdigest() == bridge_artifact["sha256"]
    assert current["execution"]["compute_capability"] == "86"

    assert component_bridge["sha256"] != bridge_artifact["sha256"]
    assert_recaptured_or_owed(
        ROOT / BRIDGE_SOURCE,
        bridge_artifact["sha256"],
        reviewed_by=BRIDGE_CHANGED_SINCE_PRODUCT_RECEIPT,
    )
    assert current["result"] == "passed"
    assert all(current["gates"].values())
    runner_source = (ROOT / receipt["artifacts"]["runner"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "gpuwm.core.wsm6" not in runner_source
    assert receipt["bridge"]["public_surface"] == "gpuwm.core.microphysics.apply"
    assert receipt["bridge"]["private_wsm6_imported_by_runner"] is False
    assert receipt["engine"]["admitted_audited_surface"] is True


def test_restart_inventory_and_raw_repeat_receipts_are_exact() -> None:
    receipt = _receipt()
    expected = [
        "qpten",
        "qtten",
        "qvten",
        "qcten",
        "component.h_diabatic",
        "component.effc",
        "component.effi",
        "component.effs",
        "component.mp_rainnc",
        "component.mp_rainncv",
        "component.mp_snownc",
        "component.mp_snowncv",
        "component.mp_graupelnc",
        "component.mp_graupelncv",
        "component.mp_sr",
    ]
    assert receipt["bridge"]["restart_inventory"] == expected
    first = receipt["runs"]["first"]
    repeat = receipt["runs"]["repeat"]
    assert (
        first["state_raw_little_endian_sha256"]
        == repeat["state_raw_little_endian_sha256"]
    )
    assert (
        first["restart_raw_little_endian_sha256"]
        == repeat["restart_raw_little_endian_sha256"]
    )
    assert all(item["recapture_raw_exact"] for item in first["held_exact"].values())
    assert all(
        item["remove_matches_declared_snapshot_raw"]
        for item in first["held_exact"].values()
    )
    assert receipt["bridge"]["restart_export_zero_copy_held_arrays"] is True
    assert receipt["bridge"]["restart_import_status"] == "red"


def test_cuda_bridge_png_is_byte_pinned_and_has_expected_dimensions() -> None:
    receipt = _receipt()
    artifact = receipt["artifacts"]["png"]
    path = ROOT / artifact["path"]
    payload = path.read_bytes()
    assert len(payload) == artifact["bytes"]
    assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", payload[16:24])
    assert (width, height) == (2790, 1800)
