from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/gm1-view-cloud-pack-real-child-v1"

EXPECTED_FILE_HASHES = {
    43: {
        "manifest.json": "25aad1d9b284f3d8bc9336d745554969778f0f7967db573b2bc781b25b5e3411",
        "receipt.json": "1828c7498b487bf09fc7f79a03bf72ec716854e4688fdcc2833406637e0f1906",
    },
    600: {
        "manifest.json": "6df3e89cebb195f64631c270eb493b6ccd8dc202cb2c5772e0064cbdee2a3ed9",
        "receipt.json": "862a6c369c0016b3ddff0078674f1d89d75ee1bea92d13b1f6506ca079223397",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_child_cloud_pack_evidence_is_exactly_cross_bound() -> None:
    validation = json.loads((EVIDENCE / "validation-receipt.json").read_bytes())
    assert validation["schema"] == "gm1.view-derived-cloud-volume-validation/v1"
    assert validation["status"] == "PASS_TWO_REAL_EVOLVED_CHILD_PACKS"
    assert validation["formula"] == (
        "out[word]=f32(q3d[1*words+word]+q3d[3*words+word])"
    )
    assert validation["source_species_order"] == ["qv", "qc", "qr", "qi", "qs", "qg"]
    assert validation["synthetic_detail"] is False
    assert validation["atomic_publication"] == {
        "final_directory_is_ready_signal": True,
        "inventory": ["cloud-condensate.f32le", "manifest.json", "receipt.json"],
        "partial_or_extra_files_refused": True,
        "payload_tracked_in_git": False,
    }

    common_coordinates = None
    common_static = None
    for step, expected_hashes in EXPECTED_FILE_HASHES.items():
        step_dir = EVIDENCE / f"step-{step:08d}"
        assert sorted(path.name for path in step_dir.iterdir()) == [
            "manifest.json",
            "receipt.json",
        ]
        for name, expected_hash in expected_hashes.items():
            assert _sha256(step_dir / name) == expected_hash

        manifest = json.loads((step_dir / "manifest.json").read_bytes())
        receipt = json.loads((step_dir / "receipt.json").read_bytes())
        sealed = validation["steps"][str(step)]
        assert manifest["schema"] == "gm1.view-derived-cloud-volume-manifest/v1"
        assert receipt["schema"] == "gm1.view-derived-cloud-volume-receipt/v1"
        assert manifest["abi"] == receipt["abi"] == validation["abi"]
        assert manifest["status"] == receipt["status"] == (
            "PASS_EXACT_QC_PLUS_QI_RENDER_TRANSPORT"
        )
        assert manifest["formula"] == receipt["formula"] == validation["formula"]
        assert manifest["quantized"] is receipt["quantized"] is False
        assert manifest["synthetic_detail"] is receipt["synthetic_detail"] is False
        assert manifest["payload"] == {
            "file": "cloud-condensate.f32le",
            "nbytes": 27_000_000,
            "sha256": sealed["payload_sha256"],
            "shape_kji": [75, 300, 300],
        }
        assert receipt["payload_nbytes"] == 27_000_000
        assert receipt["payload_sha256"] == sealed["payload_sha256"]
        assert receipt["manifest_sha256"] == expected_hashes["manifest.json"]
        assert manifest["source"] == receipt["source"]
        assert manifest["source"]["step"] == step
        assert manifest["source"]["model_time_f64_bits"] == sealed["model_time_f64_bits"]
        assert manifest["source"]["history"]["artifact_sha256"] == (
            sealed["source_history_sha256"]
        )
        assert manifest["source"]["pair_manifest_sha256"] == (
            sealed["source_pair_manifest_sha256"]
        )
        assert manifest["source"]["pair_receipt_sha256"] == (
            sealed["source_pair_receipt_sha256"]
        )
        assert manifest["source"]["q3d"]["sha256"] == sealed["q3d_sha256"]
        assert manifest["source"]["q3d"]["qc_sha256"] == sealed["qc_sha256"]
        assert manifest["source"]["q3d"]["qi_sha256"] == sealed["qi_sha256"]
        assert manifest["source"]["q3d"]["species_order"] == (
            validation["source_species_order"]
        )
        assert manifest["statistics"] == receipt["statistics"] == sealed["statistics"]
        assert sealed["independent_rust_recompute"] is True

        if common_coordinates is None:
            common_coordinates = manifest["coordinates"]
            common_static = manifest["static_identity"]
        else:
            assert manifest["coordinates"] == common_coordinates
            assert manifest["static_identity"] == common_static
