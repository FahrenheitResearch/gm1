from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / (
    "evidence/validation/"
    "gm1-t300-completed-rk2-rk3-acoustic-local-v1-attempt-0004/receipt.json"
)
RECEIPT_SHA256 = (
    "178f26a0a29c8e983ff4a0eeb5134fca34f735ef2e35dc1d5d41f3acf8018095"
)


def test_real_completed_rk2_rk3_acoustic_evidence_is_sealed() -> None:
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_SHA256
    receipt = json.loads(raw)
    assert receipt["schema"] == (
        "gm1.cuda-completed-rk2-rk3-global-acoustic-real-run/v1"
    )
    assert receipt["status"] == (
        "PASS_REAL_300X300X75_COMPLETED_RK2_RK3_GLOBAL_ACOUSTIC"
    )
    assert receipt["fixture_class"].endswith("no synthetic meteorology")
    assert receipt["domain"] == {
        "dt_s": 0.5,
        "dx_m": 50.0,
        "dy_m": 50.0,
        "nx": 300,
        "ny": 300,
        "nz": 75,
        "requested_nrk": 3,
    }
    assert receipt["producer"]["git_commit"] == (
        "f49fe29a1c2f381afb1ee38e921a4fdf511e12ac"
    )
    assert receipt["producer"]["runner_sha256"] == (
        "d25c79cf7d26015dbcdfba5cd6edf6eb5f9145873088ec9cd65c55fe8ea840b1"
    )
    assert receipt["producer"]["rk2_finish_runner_sha256"] == (
        "15b1f56920230f6b7e8dec7f41bcd53ab398f881578b84c33982d8dc0f09dcd7"
    )
    assert receipt["same_resident_device_state"] is True
    assert receipt["same_stream_before_upstream_teardown"] is True
    assert receipt["completed_rk2_stage"] is True
    assert receipt["completed_rk3_stage"] is False
    assert receipt["full_forecast"] is False

    completed = receipt["completed_rk2_token"]
    assert completed["schema"] == "gm1.tilestream-completed-rk-stage-token/v1"
    assert completed["status"] == "ADMITTED_LIVE_TILESTREAM_COMPLETED_RK_STAGE"
    assert completed["completed_nrk"] == 2
    assert completed["next_nrk"] == 3
    assert completed["source_order_complete"] is True
    assert completed["operation_token_sha256"] == (
        "b2c3e1f2bcb064f6e7ff7f7afcf0a697c23f91b361f08bb08e12f9880b67d683"
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

    pre = receipt["rk3_pre_acoustic_token"]
    velocity = receipt["rk3_post_sound_velocity_token"]
    acoustic = receipt["rk3_acoustic_continuation_token"]
    assert pre["requested_nrk"] == velocity["requested_nrk"] == 3
    assert acoustic["requested_nrk"] == 3
    assert pre["prior_completed_rk_stage_token_sha256"] == (
        completed["operation_token_sha256"]
    )
    assert pre["operation_token_sha256"] == (
        "333d7589cf7349755136d84735f0138d9410667021a07a36fec3201093c7d410"
    )
    assert velocity["operation_token_sha256"] == (
        "ee51c67f9e7325c4163962116cfdc181c37879fe26592c5f9e631d0977a4c301"
    )
    assert acoustic["operation_token_sha256"] == (
        "667adfcc79f47ece1715b97e60aa183809ce9786afb19bf83f2d3ef284a4560f"
    )
    assert acoustic["pre_acoustic_momentum_rk_token_sha256"] == (
        pre["operation_token_sha256"]
    )
    assert acoustic["post_sound_velocity_token_sha256"] == (
        velocity["operation_token_sha256"]
    )
    assert acoustic["schedule"] == {
        "rk_stage_dt_seconds_f32": 0.5,
        "sound_dt_seconds_f32": 0.0833333358168602,
        "sound_substeps": 6,
    }
