from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from cm1core.gm1_child_stretched_closure_v3 import (
    GRID_50M,
    inspect_closure_artifact,
)
from host_portability import recorded_relative_to
from authority_trees import requires_work


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "work/gm1-t300-child-stretched-closure-v3"
EVIDENCE = ROOT / "evidence/validation/gm1-t300-child-stretched-closure-v3"
MANIFEST_SHA256 = "6ff7d5a61d8e7597c46b33887311ada5e1c763f18e2a822f5d5329392eb23ee5"
PAYLOAD_SHA256 = "cf8a3731006959a23445ea9af94405d1812235de18144329d0b830dd4c5a8281"
RECEIPT_SHA256 = "8b471481978dd3a8cecbe7f324a0f7ad55c93ff6c982e4e5dd3694ccbd6945ed"
VALIDATION_SHA256 = "4beb1e294ef094e1037fb9b6f634b38c3abc125ac9717b1ac7cbfc9df3658c7a"
IMPACT_SHA256 = "4523b1f12f56ab5639c5bde82694a335b2db2327478ff376706085d153981307"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@requires_work("gm1-t300-child-stretched-closure-v3")
def test_corrected_resident_closure_is_independently_byte_exact() -> None:
    assert inspect_closure_artifact(
        ARTIFACT,
        grid=GRID_50M,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_payload_sha256=PAYLOAD_SHA256,
        expected_receipt_sha256=RECEIPT_SHA256,
    )["action"] == "inspected"


def test_validation_receipt_binds_corrected_outputs_and_repeat() -> None:
    receipt_path = EVIDENCE / "receipt.json"
    assert _sha(receipt_path) == VALIDATION_SHA256
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["status"] == (
        "PASS_REAL_300X300X75_STRETCHED_FRESH_CLOSURE_ORACLE"
    )
    artifact = dict(receipt["artifact"])
    assert recorded_relative_to(artifact.pop("directory"), "work") == PurePosixPath(
        "work/gm1-t300-child-stretched-closure-v3"
    )
    assert artifact == {
        "manifest_sha256": MANIFEST_SHA256,
        "payload_nbytes": 355_320_000,
        "payload_sha256": PAYLOAD_SHA256,
        "receipt_sha256": RECEIPT_SHA256,
    }
    assert all(item["raw_bit_exact"] for item in receipt["repeat"].values())
    assert all(item["mismatch_count"] == 0 for item in receipt["repeat"].values())
    assert receipt["output_sha256"]["defv"] == (
        "35c0ce245ef7b647331cf6796b492d81bd1d6da6d63bfe19f21c3e1bce5c4167"
    )
    assert receipt["output_sha256"]["defh"] == (
        "930f8c270bdc59e8ba2d6d37301bbfaab79de9d81f3cbe490fc74b14c22f0184"
    )
    assert receipt["output_sha256"]["tkea"] == (
        "3bb187c73f153935d4312d56d6225b1bc81665f5419dbddeba69c16e678ff19b"
    )
    assert receipt["runtime"]["mutex_clean_release"] is True


def test_downstream_impact_map_refuses_evolved_lineage() -> None:
    impact_path = EVIDENCE / "downstream-impact-map.json"
    assert _sha(impact_path) == IMPACT_SHA256
    impact = json.loads(impact_path.read_bytes())
    assert impact["status"] == "REFUSED_REVALIDATION_REQUIRED"
    assert impact["summary"]["record_count"] == 59
    assert impact["summary"]["evolution_affected_record_count"] == 23
    assert impact["summary"]["all_non_control_products_refused_revalidation"] is True
    records = {item["path"]: item for item in impact["records"]}
    assert records[
        "work/gm1-t300-child-full-state-v1/manifest.json"
    ]["evolution_affected"] is False
    assert records[
        "work/gm1-real-50m-child-live-v2/step-00000600/manifest.json"
    ]["evolution_affected"] is True
    assert records[
        "work/gm1-real-50m-child-view-packs-v1/step-00001200/manifest.json"
    ]["status"] == "REFUSED_REVALIDATION_REQUIRED"
