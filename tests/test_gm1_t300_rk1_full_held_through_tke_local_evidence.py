from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / (
    "evidence/validation/"
    "gm1-t300-rk1-full-held-through-tke-local-v1-attempt-0003"
)
FULL_HELD = ROOT / (
    "evidence/validation/"
    "gm1-t300-rk1-full-held-through-tke-local-v1-attempt-0003-full-held"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_integrated_real_rk1_evidence_is_exact_and_completed() -> None:
    receipt_path = EVIDENCE / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    assert _sha(receipt_path) == (
        "4d36eb57e4e66268a1d7ef2105062114dfb633edfd399ee29923db3003b5f1af"
    )
    assert receipt["status"] == (
        "PASS_REAL_300X300X75_RK1_POST_BOUNDARY_TKE_RAW_BIT_EXACT"
    )
    assert receipt["producer"]["git_commit"] == (
        "ee43cc601f2f01973460b4dc22ceff6f036b4dda"
    )
    assert receipt["comparison"]["mismatch_words"] == 0
    assert receipt["comparison"]["max_ulp"] == 0
    assert receipt["completed_broader_rk_stage"] is True
    assert receipt["missing_components"] == []
    assert receipt["upstream_scalar_lineage"][
        "same_stream_before_mutex_release"
    ] is True

    completed = receipt["tokens"]["completed_rk_stage"]
    assert completed["schema"] == "gm1.tilestream-completed-rk-stage-token/v1"
    assert completed["status"] == "ADMITTED_LIVE_TILESTREAM_COMPLETED_RK_STAGE"
    assert completed["operation_token_sha256"] == (
        "85efec6be4914d2c58f8c9cdde32c1c22ed728ee7eb50db6ca3ec9c5decf5d10"
    )
    assert tuple(completed["component_operation_token_sha256"]) == (
        "acoustic_continuation",
        "momentum_boundary",
        "post_acoustic_flux",
        "pre_acoustic_momentum_rk",
        "scalar_moist_rk",
        "scalar_pressure_boundary",
        "theta_divergence",
        "tke_rk",
    )
    assert _sha(EVIDENCE / "real-rk1-post-boundary-tke.png") == (
        "3ba56bf13f7eaf529df49875902f4c8b7c281221afd6249328746c425c8cd457"
    )


def test_nested_full_held_evidence_is_the_same_live_run() -> None:
    receipt_path = FULL_HELD / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    assert _sha(receipt_path) == (
        "caa6a092f0ea72f65fd86a11f0df3ed97afa5ea6860538752483f11d93ed2cf6"
    )
    assert receipt["status"] == (
        "PASS_REAL_300X300X75_SOLVE1_FULL_HELD_RAW_BIT_EXACT"
    )
    assert receipt["mismatch_words"] == 0
    assert receipt["producer"]["git_commit"] == (
        "ee43cc601f2f01973460b4dc22ceff6f036b4dda"
    )
    assert receipt["full_held_token"]["operation_token_sha256"] == (
        "34dd1fafa3bff8d0d12cecdba085c1d788aa7434cf5fcdb7e5047721a6730a69"
    )
    assert receipt["live_pre_acoustic_continuation"][
        "same_stream_before_mutex_release"
    ] is True
    assert receipt["runtime"]["mutex_clean_release"] is True
    assert _sha(FULL_HELD / "real-solve1-full-held.png") == (
        "cb287571a9571361c3bfffd72afd816da443b9268a79e05e5b9bf1663161e998"
    )
