from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from sealed_evidence import assert_recaptured_or_owed


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/gm1-real-5m-child-terrain-v2"
VALIDATION_SHA256 = (
    "d99f1ae105773cc808a2fd11e330ddeed2ca5cf6a7f59f6db96d09f13ad239cd"
)
PNG_SHA256 = "b1e355d42b5792f1f2b426fe7e2d84c00d83b75e349e32177cce6f8a20d100eb"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_5m_terrain_receipt_is_source_bound_and_not_synthetic() -> None:
    path = EVIDENCE / "validation-receipt.json"
    assert _sha256(path) == VALIDATION_SHA256
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["status"] == (
        "PASS_REAL_SOURCE_COORDINATE_3000X3000_5M_TERRAIN_RUST_ROUNDTRIP"
    )
    assert receipt["artifact"] == {
        "directory": "work/gm1-real-5m-child-terrain-v2",
        "manifest_sha256": (
            "7a6b5139168af3384c58998bff8df1dd61f102c2f611ac163251425e607ca45e"
        ),
        "payload_nbytes": 36_000_000,
        "payload_sha256": (
            "2a444d93e482cc159a2b2b83392d41105a5587261cef801f623a2d4d9d51d72d"
        ),
        "receipt_sha256": (
            "553b27baa2f0fda542703a09ba924284f78e9c251e14cd56309958d39c151152"
        ),
    }
    assert receipt["grid"]["nx"] == 3_000
    assert receipt["grid"]["ny"] == 3_000
    assert receipt["grid"]["nz"] == 75
    assert receipt["contract"]["source_backed_interpolation"] is True
    assert receipt["contract"]["synthetic_detail"] is False
    assert receipt["contract"]["synthetic_translation"] is False
    regression = receipt["historical_300x300_50m_regression"]
    assert regression["raw_bit_equal"] is True
    assert regression["matched_word_count"] == 90_000
    assert regression["accepted_payload_sha256"] == (
        regression["parameterized_replay_payload_sha256"]
    )
    assert_recaptured_or_owed(
        ROOT / "rust/cm1-state/src/child_terrain.rs",
        receipt["producer"]["source_sha256"],
        reviewed_by=(
            "gap/childseal: the terrain producer records the parent checkpoint "
            "and validated plan it was built from instead of a frozen pair"
        ),
    )


def test_real_5m_terrain_png_is_exact_and_labels_source_backing() -> None:
    receipt = json.loads(
        (EVIDENCE / "validation-receipt.json").read_text(encoding="utf-8")
    )
    png_path = ROOT / receipt["png"]["path"]
    raw = png_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PNG_SHA256
    assert receipt["png"]["sha256"] == PNG_SHA256
    assert receipt["png"]["nbytes"] == len(raw)
    assert receipt["png"]["display_interpolation_only"].startswith(
        "nearest display rasterization"
    )
    assert receipt["png"]["producer_source_sha256"] == _sha256(
        ROOT / "tools/render_gm1_real_5m_child_terrain.py"
    )
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert raw[12:16] == b"IHDR"
    assert struct.unpack(">II", raw[16:24]) == (3072, 1368)
