from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sealed_evidence import assert_recaptured_or_owed


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "evidence/validation/gm1-child-atmosphere-stream-replay-v1/receipt.json"
)
RECEIPT_SHA256 = "426a566a1f0e174451465e38620b4a213a85d4966435ba6f1f2ee9d1322ba032"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stream_replay_receipt_is_exact_real_and_bounded() -> None:
    assert _sha256(EVIDENCE) == RECEIPT_SHA256
    receipt = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert receipt["status"] == (
        "PASS_REAL_300X300_50M_CHILD_ATMOSPHERE_STREAM_RAW_BITS"
    )
    assert receipt["artifact"]["arena_nbytes"] == 840_600_000
    assert receipt["artifact"]["arena_sha256"] == (
        "59aa935cee8f7a69187af97ead3e2b29fd50e6ef665ca6b22ccb07bc9a34c3b7"
    )
    replay = receipt["replay"]
    assert replay["all_arena_bytes_compared"] is True
    assert replay["arena_raw_bit_equal"] is True
    assert replay["carrier_count"] == 28
    assert replay["generated_carrier_cache_bytes"] == 0
    assert replay["output_stream_chunk_bytes"] == 1_048_576
    assert receipt["synthetic_atmospheric_data"] is False
    assert receipt["synthetic_translation"] is False


def test_stream_replay_producer_and_5m_layout_are_pinned() -> None:
    receipt = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert_recaptured_or_owed(
        ROOT / "rust/cm1-state/src/child_atmosphere.rs",
        receipt["producer"]["module_source_sha256"],
        reviewed_by="78e1b2e Add bounded five-metre auxiliary writers",
    )
    # The same receipt owes a re-capture on its CLI source too.  The
    # whole-function skip this replaced hid that second debt entirely, so the
    # receipt looked like it owed one file when it owes two.
    assert_recaptured_or_owed(
        ROOT / "rust/cm1-state/src/bin/gm1_child_atmosphere.rs",
        receipt["producer"]["cli_source_sha256"],
        reviewed_by="6bdbaa4 Publish streamed real 5m child atmosphere",
    )
    assert receipt["grid_layout"] == {
        "accepted_300x300_50m_arena_nbytes": 840_600_000,
        "hard_3000x3000_5m_arena_nbytes": 84_027_600_000,
        "hard_5m_layout_only": True,
        "stagger_families_source_contained": [
            "mass",
            "packed",
            "x-face",
            "y-face",
            "z-face",
            "surface",
        ],
    }
