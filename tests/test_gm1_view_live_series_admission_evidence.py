from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "evidence"
    / "validation"
    / "gm1-view-live-series-admission-v1"
    / "receipt.json"
)
PACK_SOURCE = ROOT / "rust" / "gm1-view" / "src" / "pack.rs"
WATCH_SOURCE = ROOT / "rust" / "gm1-view" / "src" / "watch.rs"
PUBLISHER_SOURCE = ROOT / "rust" / "cm1-state" / "src" / "child_view_pack.rs"
EXPECTED_RECEIPT_SHA256 = (
    "5ff5b8c1e796c39d8eb8686294258509aa2eb02e9e7e233aad13aa883b7d8d82"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_file_sha256(commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def test_live_series_receipt_is_canonical_and_source_pinned() -> None:
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_RECEIPT_SHA256
    receipt = json.loads(raw)
    assert raw.rstrip(b"\n") == json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode()
    assert receipt["schema"] == "gm1.view-live-series-admission-evidence/v1"
    assert receipt["status"] == (
        "PASS_PROVENANCE_STABLE_MONOTONIC_LIVE_VIEW_SERIES"
    )
    assert _git_file_sha256(
        receipt["publisher"]["commit"],
        "rust/cm1-state/src/child_view_pack.rs",
    ) == receipt["publisher"]["source_sha256"]
    assert _git_file_sha256(
        receipt["viewer"]["commit"], "rust/gm1-view/src/pack.rs"
    ) == receipt["viewer"]["pack_source_sha256"]
    assert _git_file_sha256(
        receipt["viewer"]["commit"], "rust/gm1-view/src/watch.rs"
    ) == receipt["viewer"]["watch_source_sha256"]


def test_live_series_policy_varies_data_hashes_but_not_authority() -> None:
    receipt = json.loads(RECEIPT.read_bytes())
    authority = receipt["live_series_authority"]
    assert authority["coupled_authority"] is True
    assert authority["historical_exact_fixtures_retained"] is True
    assert authority["fixed_history_contract_sha256"] == (
        "a5d3f1c784544cf0398a0499d51fc77792b74fcab8129755b9e43b872c9dbb90"
    )
    assert authority["fixed_operator_manifest_sha256"] == (
        "6112fb9302f57f82b09118b3f38c6ae8dd3d82c59eb14fd01940f2dc43428c09"
    )
    assert authority["accepted_data_identity"].startswith("variable lowercase SHA-256")
    assert set(authority["refusals_tested"]) == {
        "mixed producer/source lineage",
        "foreign operator manifest",
        "foreign history contract",
        "corrupt q3d layout",
        "corrupt payload word",
        "noncanonical JSON or extra file",
        "step/time mismatch",
        "visible timeline rollback below cursor",
    }
    pack_source = PACK_SOURCE.read_text(encoding="utf-8")
    live_policy = pack_source.split("fn live_series_source_is_admitted", 1)[1].split(
        "fn validate_coordinates", 1
    )[0]
    assert authority["fixed_history_contract_sha256"] not in live_policy
    assert "FINAL_STEP_HISTORY_CONTRACT_SHA256" in live_policy
    assert authority["fixed_operator_manifest_sha256"] in live_policy
    assert receipt["pack"]["payload_sha256"] not in live_policy
    assert "authority_is_admitted(&manifest.producer, &manifest.source)" in pack_source
    assert "maximum_visible_step" in WATCH_SOURCE.read_text(encoding="utf-8")


def test_real_pair_pack_and_watch_gate_is_exact_and_non_synthetic() -> None:
    receipt = json.loads(RECEIPT.read_bytes())
    assert receipt["publisher"] == {
        "commit": "eff2bec1ab48f31067adea7d543c7d46ce460a1b",
        "executable_nbytes": 492_544,
        "executable_sha256": (
            "9c90b643bb311fdd38e56d0101173034bdc5b8d1fc5c33418633f5ad28f24dd8"
        ),
        "source_sha256": (
            "bad7907154a7867086ce2505b3454d144bfab4a2c371af34a45237c93e8db25d"
        ),
    }
    assert receipt["pack"]["payload_nbytes"] == 27_000_000
    assert receipt["pack"]["payload_sha256"] == (
        "88c3110ae8e2295a0b5644879b428c058e886ee6e6a40103289c5e3d7119daa2"
    )
    assert receipt["viewer"]["status"] == "PASS_GM1_VIEW_WATCH_ADMISSION"
    assert receipt["timings"]["validated_source_nbytes"] == 1_438_075_139
    assert receipt["timings"]["internal_per_pair_stage_timers"] is False
    assert receipt["tests"] == {
        "cm1_state_child_view_pack": 4,
        "gm1_view_clippy_deny_warnings": True,
        "gm1_view_lib": 62,
        "status": "PASS",
    }
