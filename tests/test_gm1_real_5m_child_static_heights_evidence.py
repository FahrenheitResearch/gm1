from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sealed_evidence import assert_recaptured_or_owed


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/gm1-real-5m-child-static-heights-v2"
VALIDATION_SHA256 = (
    "cd3032b2a515cf40200532b84780e8b1a1d6a1b92b645d67c8df3333176e3766"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_5m_static_height_receipt_is_source_bound_and_exact() -> None:
    path = EVIDENCE / "validation-receipt.json"
    assert _sha256(path) == VALIDATION_SHA256
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["status"] == (
        "PASS_REAL_SOURCE_COORDINATE_3000X3000_5M_STATIC_HEIGHTS_RUST_ROUNDTRIP"
    )
    assert receipt["contract"] == {
        "abi": "gm1.source-coordinate-child-static-heights-f32/v2",
        "dtype": "little-endian-f32",
        "manifest_schema": (
            "gm1.source-coordinate-child-static-heights-manifest/v2"
        ),
        "order": "C[k,j,i]",
        "schema": "gm1.source-coordinate-child-static-heights/v2",
        "synthetic_detail": False,
        "synthetic_translation": False,
    }
    assert receipt["grid"]["nx"] == 3_000
    assert receipt["grid"]["ny"] == 3_000
    assert receipt["grid"]["nz"] == 75
    assert receipt["artifact"]["zh"] == {
        "nbytes": 2_700_000_000,
        "sha256": (
            "7aa08247f64d6c05fce30b12f7bb8ef8cd050637b4c3a69feea873f20b7d6864"
        ),
        "shape_kji": [75, 3_000, 3_000],
    }
    assert receipt["artifact"]["zf"] == {
        "nbytes": 2_736_000_000,
        "sha256": (
            "8ff218d8cd8f1f37b797d373b127b4f5b98097385f3cccdab4450e6a4bec449d"
        ),
        "shape_kji": [76, 3_000, 3_000],
    }
    regression = receipt["historical_300x300_50m_regression"]
    assert regression["raw_bit_equal"] is True
    assert regression["total_matched_words"] == 13_590_000
    assert regression["accepted_cuda"] == {
        "zf_sha256": regression["parameterized_rust_replay"]["zf_sha256"],
        "zh_sha256": regression["parameterized_rust_replay"]["zh_sha256"],
    }
    assert receipt["source_terrain"] == {
        "abi": "gm1.source-coordinate-child-terrain-f32/v2",
        "manifest_sha256": (
            "7a6b5139168af3384c58998bff8df1dd61f102c2f611ac163251425e607ca45e"
        ),
        "payload_sha256": (
            "2a444d93e482cc159a2b2b83392d41105a5587261cef801f623a2d4d9d51d72d"
        ),
        "receipt_sha256": (
            "553b27baa2f0fda542703a09ba924284f78e9c251e14cd56309958d39c151152"
        ),
    }


def test_real_5m_static_height_producer_sources_are_frozen() -> None:
    receipt = json.loads((EVIDENCE / "validation-receipt.json").read_bytes())
    assert_recaptured_or_owed(
        ROOT / "rust/cm1-state/src/child_static_heights.rs",
        receipt["producer"]["source_sha256"],
        reviewed_by=(
            "gap/childseal: read_child_static_height_identity admits any "
            "published static-height artifact from its own receipt"
        ),
    )
    assert receipt["producer"]["cli_source_sha256"] == _sha256(
        ROOT / "rust/cm1-state/src/bin/gm1_child_static_heights.rs"
    )
    assert receipt["tests"]["independent_full_byte_inspect"] is True
    assert receipt["transform"]["stream_chunk_bytes"] == 1 << 20
