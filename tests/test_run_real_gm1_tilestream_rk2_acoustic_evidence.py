from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/gm1-t300-rk2-acoustic-local-v1"
FULL_HELD = ROOT / ("evidence/validation/gm1-t300-rk2-acoustic-local-v1-full-held")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_rk2_acoustic_evidence_is_exact_and_incomplete_stage() -> None:
    receipt_path = EVIDENCE / "receipt.json"
    assert _sha(receipt_path) == (
        "3b1e099237e4d99b3cb784ab6604afd464ebf940e6b79cb16729106324d88ed8"
    )
    receipt = json.loads(receipt_path.read_text("ascii"))
    assert receipt["status"] == "PASS_REAL_300X300X75_RK2_GLOBAL_ACOUSTIC"
    assert receipt["producer"]["git_commit"] == (
        "57e8e7b214d49bc520e308449c7b7dca6d48fe1f"
    )
    assert receipt["completed_rk1_token"]["operation_token_sha256"] == (
        "35a4f575766440b56ba9820c4ad2552312636d0064106d9477d85c6ca9d085e7"
    )
    assert receipt["rk2_pre_acoustic_token"]["operation_token_sha256"] == (
        "08aefb60f60d6ca8be5a41571f1b3c35e9400bfdb0b1147daf5c3fd1db6fb385"
    )
    assert receipt["rk2_post_sound_velocity_token"]["operation_token_sha256"] == (
        "2c4145e59d1e6dda777d99eff1bba944c50987855fe6f3007dce8743306f1aad"
    )
    assert (
        receipt["rk2_acoustic_continuation_token"]["operation_token_sha256"]
        == "f713a24bf177772b2b6034be91d50a48cce13156767c241c519ad1c1c0933cb9"
    )
    assert receipt["execution"]["schedule"] == {
        "rk_stage_dt_seconds_f32": 0.25,
        "sound_dt_seconds_f32": 0.0833333358168602,
        "sound_substeps": 3,
    }
    assert receipt["post_boundary_pressure_restored_to_live_device"] is True
    assert receipt["current_scalar_device_carriers_reused"] is True
    assert receipt["completed_rk2_stage"] is False
    assert receipt["full_forecast"] is False


def test_nested_full_held_evidence_is_pinned() -> None:
    assert _sha(FULL_HELD / "receipt.json") == (
        "98e094fe3dd0053a86e51425774d34d3e4557d7855f712fda5236adf272711bf"
    )
    assert _sha(FULL_HELD / "real-solve1-full-held.png") == (
        "cb287571a9571361c3bfffd72afd816da443b9268a79e05e5b9bf1663161e998"
    )
