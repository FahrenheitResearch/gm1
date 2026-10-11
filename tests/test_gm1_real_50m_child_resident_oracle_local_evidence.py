from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "evidence"
    / "validation"
    / "gm1-real-50m-child-resident-oracle-local-v1"
)
SUMMARY_SHA256 = (
    "1ce79873f1d54b2dd06d83fc0200ab3120489064186b07fca5ccd6616b41f190"
)
RECEIPT_SHA256 = (
    "8f0dc85e719d280f7d780a1ddb953a6de939ae5a9423cc8ddf4ec59a202c1910"
)


def _load_exact(name: str, sha256: str) -> dict[str, object]:
    payload = (EVIDENCE / name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == sha256
    return json.loads(payload)


def test_local_resident_oracle_evidence_is_raw_word_exact_and_fail_closed() -> None:
    summary = _load_exact("oracle-summary.json", SUMMARY_SHA256)
    receipt = _load_exact("validation-receipt.json", RECEIPT_SHA256)
    status = "PASS_EXACT_UNINTERRUPTED_VS_PUBLISH_RESTORE_RAW_WORDS"
    assert summary["status"] == receipt["status"] == status
    assert summary["checkpoint_arena_raw_words_equal"] is True
    assert summary["history_arena_raw_words_equal"] is True
    assert summary["contract_identity_equal"] is True
    assert summary["model_time_identity_equal"] is True
    assert summary["completed_step"] == 3
    assert receipt["oracle_summary_sha256"] == SUMMARY_SHA256
    assert receipt["oracle"]["checkpoint_arena_sha256"] == (
        summary["checkpoint_arena_sha256"]
    )
    assert receipt["oracle"]["history_arena_sha256"] == (
        summary["history_arena_sha256"]
    )
    assert receipt["lifecycle_gate"]["completed_rk3_terminal_flags"] == [
        True,
        True,
        True,
    ]
    assert receipt["lifecycle_gate"]["next_rk1_fresh_flags"] == [
        False,
        False,
        False,
    ]
    assert receipt["final_device_state"]["mutex"] == "ABSENT"
    assert receipt["final_device_state"]["process"] == "EXITED"
    assert "no production trajectory" in receipt["claim_limit"]
