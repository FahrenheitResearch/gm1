from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sealed_evidence import assert_recaptured_or_owed


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/gm1-real-5m-child-atmosphere-v2"
VALIDATION_SHA256 = (
    "c51a34ffc1625bb60222a96c60d13b3fe60a04bb82cab396d6ad7acb95b69b48"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_5m_child_atmosphere_is_exact_and_source_bound() -> None:
    path = EVIDENCE / "validation-receipt.json"
    assert _sha256(path) == VALIDATION_SHA256
    receipt = json.loads(path.read_bytes())
    assert receipt["status"] == (
        "PASS_REAL_SOURCE_COORDINATE_3000X3000_5M_CHILD_ATMOSPHERE_RUST_ROUNDTRIP"
    )
    assert receipt["artifact"] == {
        "arena_nbytes": 84_027_600_000,
        "arena_sha256": (
            "36c945e488307900b30cf809ffd8c777d84f6cd5e4ecaa318dcecdc95fa23ce5"
        ),
        "carrier_count": 28,
        "directory": "work/gm1-real-5m-child-atmosphere-v2",
        "manifest_sha256": (
            "ba1391a3d9f4914f4b9a23f0d5021994d7587786829f4b85cb6eee0db5047447"
        ),
        "receipt_sha256": (
            "24790ddca9403c5066c69517e51442ff24d19d7f6dc38f999cabe96116983483"
        ),
    }
    assert receipt["grid"]["nx"] == 3_000
    assert receipt["grid"]["ny"] == 3_000
    assert receipt["grid"]["nz"] == 75
    assert receipt["contract"]["stream_chunk_bytes"] == 1 << 20
    assert receipt["memory_contract"]["full_generated_carrier_cache_bytes"] == 0
    assert receipt["synthetic_atmospheric_data"] is False
    assert receipt["synthetic_detail"] is False
    assert receipt["synthetic_translation"] is False


def test_real_5m_child_atmosphere_invariants_and_regression_are_frozen() -> None:
    receipt = json.loads((EVIDENCE / "validation-receipt.json").read_bytes())
    invariants = receipt["invariants"]
    assert invariants["all_carrier_bytes_independently_recomputed"] is True
    assert invariants["arena_offsets_contiguous_and_exhaustive"] is True
    assert invariants["current_a_pairs_raw_bit_equal"] is True
    assert len(invariants["current_a_pair_sha256"]) == 6
    assert invariants["qa_negative_words"] == 0
    assert invariants["qa_nonnegative"] is True
    assert invariants["tkea_present"] is False
    regression = receipt["historical_300x300_50m_regression"]
    assert regression["raw_bit_equal"] is True
    assert regression["matched_word_count"] == 210_150_000
    assert regression["arena_sha256"] == (
        "59aa935cee8f7a69187af97ead3e2b29fd50e6ef665ca6b22ccb07bc9a34c3b7"
    )


def test_real_5m_child_atmosphere_producer_sources_are_frozen() -> None:
    receipt = json.loads((EVIDENCE / "validation-receipt.json").read_bytes())
    assert_recaptured_or_owed(
        ROOT / "rust/cm1-state/src/child_atmosphere.rs",
        receipt["producer"]["module_source_sha256"],
        reviewed_by="78e1b2e Add bounded five-metre auxiliary writers",
    )
    assert_recaptured_or_owed(
        ROOT / "rust/cm1-state/src/bin/gm1_child_atmosphere.rs",
        receipt["producer"]["cli_source_sha256"],
        reviewed_by=(
            "gap/childseal: publish-grid accepts from-static-heights so any "
            "admitted child grid is reachable, not only the pinned 5 m preset"
        ),
    )
    assert receipt["static_heights"]["validation_receipt_sha256"] == (
        "cd3032b2a515cf40200532b84780e8b1a1d6a1b92b645d67c8df3333176e3766"
    )
