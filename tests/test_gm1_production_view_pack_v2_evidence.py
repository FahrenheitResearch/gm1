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
    / "gm1-production-view-pack-v2-local-oracle"
    / "receipt.json"
)
EXPECTED_RECEIPT_SHA256 = (
    "f264506eb96e2d6d733b19f1f051d61280b225d819905744dada0541498bf420"
)
SOURCE_PATHS = [
    "rust/cm1-state/src/child_view_pack.rs",
    "rust/cm1-state/src/child_view_pack/production.rs",
    "rust/cm1-state/src/bin/gm1_child_view_pack.rs",
]


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_bytes())


def test_receipt_is_canonical_hash_pinned_and_source_reproducible() -> None:
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_RECEIPT_SHA256
    receipt = json.loads(raw)
    assert raw.rstrip(b"\n") == json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode()
    producer = receipt["producer"]
    archive = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            producer["commit"],
            "--",
            *SOURCE_PATHS,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert len(archive) == producer["source_archive_nbytes"]
    assert hashlib.sha256(archive).hexdigest() == producer["source_archive_sha256"]


def test_exact_channel_inventory_is_additive_and_raw_word_pinned() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "gm1.production-view-pack-v2-local-oracle/v1"
    assert receipt["status"] == "PASS_EXACT_REAL_STEP1_PRODUCTION_VIEW_PACK_V2"
    channels = receipt["channels"]
    assert set(channels) == {
        "cloud_condensate_kgkg",
        "pressure_pa",
        "static_zf_m",
        "terrain_msl_m",
        "total_geopotential_zf_m",
        "u_staggered_ms",
        "v_staggered_ms",
        "w_staggered_ms",
    }
    assert sum(channel["nbytes"] for channel in channels.values()) == 190_620_000
    assert channels["u_staggered_ms"]["shape"] == [75, 300, 301]
    assert channels["v_staggered_ms"]["shape"] == [75, 301, 300]
    assert channels["w_staggered_ms"]["shape"] == [76, 300, 300]
    assert channels["terrain_msl_m"]["sha256"] == (
        "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91"
    )
    assert channels["static_zf_m"]["sha256"] == (
        "df5aef17965915406fe334600afe2f8f2e1ebb5d07665f71820eaac9bde4128f"
    )


def test_source_pair_and_independent_every_byte_inspection_are_exact() -> None:
    receipt = _receipt()
    source = receipt["source"]
    assert source["step"] == 1
    assert source["model_time_f64_bits"] == "3fe0000000000000"
    assert source["pair_manifest_sha256"] == (
        "687b4de7ee61de115dea88536e1f9e7ff1f58bf155285f08215d1e7800f862d4"
    )
    assert source["pair_receipt_sha256"] == (
        "849e153bbc995f72f75a4410cefa5c8a4a58ccdce3a8648a997512427b222e2b"
    )
    assert receipt["independent_inspection"] == {
        "every_payload_byte_compared": True,
        "exit_code": 0,
        "manifest_and_receipt_bytes_compared": True,
        "wall_seconds": 4.724896,
    }
    assert receipt["output"]["payload_set_sha256"] == (
        "326bf931480d1496ad7661c699c0d54abee2f7d55e08ea2f2eecb95606d88cdc"
    )


def test_absent_science_is_fail_closed_not_synthesized() -> None:
    receipt = _receipt()
    assert receipt["optional_channels"] == {
        "reflectivity_dbz": "MISSING_SOURCE_CARRIER",
        "updraft_mass_center_ms": "DERIVABLE_FROM_EXACT_W3D_FORMULA_DEFERRED",
        "vertical_vorticity_s": (
            "DERIVABLE_FROM_EXACT_U3D_V3D_FORMULA_DEFERRED"
        ),
    }
    assert "no synthetic detail" in receipt["claim_limit"]
