from __future__ import annotations

import hashlib
import json
from pathlib import Path
from measured_source import development_evidence, measured_identity, unlabel, unlabel_path


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "src" / "cm1core" / "records"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authority_evidence_hashes_are_live() -> None:
    authority = json.loads((RECORDS / "CM1-AUTHORITY.json").read_text(encoding="utf-8"))
    for item in authority["authority_evidence_in_development_repository"]:
        assert item["path"].startswith("development-repository:")
        target = ROOT / item["path"].removeprefix("development-repository:")
        assert target.is_file()
        assert sha256(target) == item["sha256"]
    shipped = authority["authority_content_identity_in_source_distribution"]
    assert sha256(ROOT / shipped["path"]) == shipped["sha256"]
    assert "authority_evidence_in_source_distribution" not in authority


def test_native_repeat_receipt_hashes_are_live() -> None:
    authority = json.loads((RECORDS / "CM1-AUTHORITY.json").read_text(encoding="utf-8"))
    native = authority["native_reference"]
    assert (
        sha256(ROOT / native["rebuild_comparison"])
        == native["rebuild_comparison_sha256"]
    )
    assert (
        sha256(ROOT / native["one_step_reproduction"])
        == native["one_step_reproduction_sha256"]
    )


def test_dry_semantic_fixture_hash_and_line_endings_are_stable() -> None:
    profile = json.loads(
        (RECORDS / "CM1-PROFILE-MANIFEST.json").read_text(encoding="utf-8")
    )["dry_bootstrap"]["semantic_config_fixture"]
    target = ROOT / profile["source_distribution_path"]
    payload = target.read_bytes()
    assert len(payload) == profile["bytes"]
    assert hashlib.sha256(payload).hexdigest() == profile["sha256"]
    assert b"\r\n" not in payload


def test_real_product_semantic_fixture_hash_and_line_endings_are_stable() -> None:
    profile = json.loads(
        (RECORDS / "CM1-PROFILE-MANIFEST.json").read_text(encoding="utf-8")
    )["real_atmosphere_100m"]["semantic_config_fixture"]
    target = ROOT / profile["source_distribution_path"]
    payload = target.read_bytes()
    assert len(payload) == profile["bytes"]
    assert hashlib.sha256(payload).hexdigest() == profile["sha256"]
    assert b"\r\n" not in payload


def test_repository_declares_lf_for_hashed_text() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes


def test_post_getset_inventory_hashes_are_live_and_lf_only() -> None:
    evidence = development_evidence("oracles/post-getset-v1")
    inventory_path = evidence / "artifact-inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert unlabel(inventory["fixture_id"]) == "O-POST-GETSET-V1"
    payloads = dict(inventory["ignored_runtime_artifacts"]["complete_field_payloads"])
    # The sealed inventory names the per-file hashes' record by its path in the development repository.
    assert unlabel_path(payloads.pop("per_file_hashes")).endswith(
        "evidence/oracles/post-getset-v1/full-field-comparison.json"
    )
    assert payloads == {
        "bytes_per_run": 123_070_752,
        "count_per_run": 46,
        "hashes_require_ignored_runtime_bytes_to_recompute": True,
        "tracked": False,
    }
    items = [
        *inventory["tracked_artifacts"],
        *inventory["external_source_distribution_artifacts"],
    ]
    for item in items:
        label = item["path"]
        assert label.startswith("source-distribution:")
        target = ROOT / label.removeprefix("source-distribution:")
        assert target.is_file()
        assert measured_identity(target) == (item["bytes"], item["sha256"])
        assert item["hash_directly_recomputable_from_source_distribution"] is True
    assert b"\r" not in inventory_path.read_bytes()

    oracle = json.loads((RECORDS / "CM1-ORACLE-MANIFEST.json").read_text())
    post = next(
        item for item in oracle["fixtures"] if item["id"] == "O-POST-GETSET-V1"
    )
    assert sha256(inventory_path) == post["artifact_inventory_sha256"]
    assert sha256(evidence / "deck.json") == post["deck_manifest_sha256"]
    assert sha256(development_evidence("oracles/post-getset-v1/post-getset-receipt.json")) == post["receipt_sha256"]
