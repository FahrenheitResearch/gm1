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
    / "gm1-view-live-series-admission-v2"
    / "receipt.json"
)
EXPECTED_RECEIPT_SHA256 = (
    "2eca0f7c3e7bf13f8dc142e47685d640a01fa65e961255a9ebdd0d29e939f7f6"
)


def _git_file_sha256(commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def test_parallel_live_series_receipt_is_canonical_and_source_pinned() -> None:
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_RECEIPT_SHA256
    receipt = json.loads(raw)
    assert raw.rstrip(b"\n") == json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode()
    assert receipt["schema"] == "gm1.view-live-series-admission-evidence/v1"
    assert receipt["status"] == (
        "PASS_PARALLEL_PROVENANCE_STABLE_MONOTONIC_LIVE_VIEW_SERIES"
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


def test_parallel_pair_gate_preserves_full_verification_and_science_bytes() -> None:
    receipt = json.loads(RECEIPT.read_bytes())
    timings = receipt["timings"]
    assert timings["publisher"]["checkpoint_worker_seconds"] == 11.124451
    assert timings["publisher"]["history_worker_seconds"] == 5.523911
    assert timings["publisher"]["pair_parallel_wall_seconds"] == 11.147818
    assert timings["independent_inspect"]["pair_parallel_wall_seconds"] == 11.449782
    assert timings["validated_source_nbytes"] == 1_438_075_139
    assert receipt["pack"]["payload_nbytes"] == 27_000_000
    assert receipt["pack"]["payload_sha256"] == (
        "88c3110ae8e2295a0b5644879b428c058e886ee6e6a40103289c5e3d7119daa2"
    )
    assert receipt["pack"]["source_q3d_sha256"] == (
        "0055b6ecfe010ef205d68bbe3d0bc4a071220af679cde4af402c3e75518d7661"
    )
    assert receipt["viewer"]["status"] == "PASS_GM1_VIEW_WATCH_ADMISSION"


def test_parallel_authority_accepts_new_data_but_refuses_mixed_lineage() -> None:
    receipt = json.loads(RECEIPT.read_bytes())
    authority = receipt["live_series_authority"]
    assert authority["coupled_authority"] is True
    assert authority["historical_exact_fixtures_retained"] is True
    assert authority["accepted_data_identity"].startswith("variable lowercase SHA-256")
    assert "mixed producer/source lineage" in authority["refusals_tested"]
    assert "stale watcher binary with unadmitted producer tuple" in authority[
        "refusals_tested"
    ]

    source = _git_file_sha256(
        receipt["viewer"]["commit"], "rust/gm1-view/src/pack.rs"
    )
    assert source == receipt["viewer"]["pack_source_sha256"]
    pack_source = subprocess.run(
        [
            "git",
            "show",
            f'{receipt["viewer"]["commit"]}:rust/gm1-view/src/pack.rs',
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert receipt["publisher"]["commit"] in pack_source
    assert receipt["publisher"]["source_sha256"] in pack_source
    assert receipt["publisher"]["executable_sha256"] in pack_source
    assert "eff2bec1ab48f31067adea7d543c7d46ce460a1b" in pack_source
    assert "authority_is_admitted(&manifest.producer, &manifest.source)" in pack_source
