from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/validation/gm1-view-preview-v3-ui-live-profile-v1"
FILE_HASHES = {
    "interactive-live-60s.json": (
        "2ceea3ed80e76692c25653ec8c23ea5f50d3918b17a63397b448929fc37babf0"
    ),
    "interactive-quality-switch-pre-stable-gate.json": (
        "1e7255c2c8d76f0c83191cf756e1ffd9187f25bd76f21b1e6466f0d2ce1b94a7"
    ),
    "interactive-quality-switch-pre-stable-gate-refusal.json": (
        "80a116c842f151b55bdbd2c50606f44cd98bb41f4bcd8fae8503b3a9d37b5604"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_preview_v3_live_profile_is_stable_real_dx12_and_responsive() -> None:
    for relative, expected in FILE_HASHES.items():
        assert _sha256(BASE / relative) == expected

    receipt = json.loads((BASE / "interactive-live-60s.json").read_text("utf-8"))
    assert receipt["schema"] == "gm1.view.interactive-frame-profile/v1"
    assert receipt["status"] == "PASS_RESPONSIVE_LIVE_PROFILE"
    assert receipt["quality"] == "Scientific Live"
    assert receipt["quality_stable"] is True
    assert receipt["backend_actual"] == "Dx12"
    assert receipt["adapter_name"] == "NVIDIA GeForce RTX 3080"
    assert receipt["adapter_vendor"] == 0x10DE
    assert receipt["soak_seconds"] == 60.0
    assert receipt["samples"] >= 900
    assert receipt["mean_frame_ms"] <= 33.334
    assert receipt["p95_frame_ms"] <= 50.0
    assert receipt["worst_frame_ms"] <= 250.0
    assert receipt["exact_step"] == 1200
    assert receipt["exact_payload_sha256"] == (
        "62e072798e6c5f7c988a46af5617611588caca413cbf360797daaa1fecf3eacd"
    )
    assert receipt["scientific_interpolation"] is False
    assert receipt["per_frame_readback"] is False


def test_mixed_quality_interaction_profile_is_explicitly_refused() -> None:
    refusal = json.loads(
        (BASE / "interactive-quality-switch-pre-stable-gate-refusal.json").read_text(
            "utf-8"
        )
    )
    assert refusal["status"] == "REFUSED_MIXED_QUALITY_PROFILE"
    assert refusal["source_receipt_sha256"] == FILE_HASHES[
        "interactive-quality-switch-pre-stable-gate.json"
    ]
    assert refusal["scientific_data_changed"] is False
    assert "stable Scientific Live 60-second DX12 run" in refusal["replacement_gate"]
