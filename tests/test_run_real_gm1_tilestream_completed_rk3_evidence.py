from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / (
    "evidence/validation/"
    "gm1-t300-completed-rk3-final-tail-local-v1-attempt-0001/receipt.json"
)
RECEIPT_SHA256 = (
    "29459000ebce580589f6c36d83837d18f32b64daea21db8a1b768ab49ac0509d"
)


def test_real_completed_rk3_split_mass_evidence_is_sealed() -> None:
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_SHA256
    outer = json.loads(raw)
    assert outer["status"] == (
        "PASS_REAL_300X300X75_COMPLETED_RK2_RK3_GLOBAL_ACOUSTIC"
    )
    assert outer["producer"]["git_commit"] == (
        "152ba00f882dc5730b45d98040cbda7e0e978eba"
    )
    assert outer["producer"]["runner_sha256"] == (
        "1ac1b30ff3a6ff8b9d223d7235a6312c8c37349c4d3d95aa041c318735e35f25"
    )
    assert outer["same_resident_device_state"] is True
    assert outer["same_stream_before_upstream_teardown"] is True

    continuation = outer["downstream_continuation"]
    assert continuation["invoked"] is True
    assert continuation["metadata_exposed"] is True
    assert continuation["admitted_by_outer_runner"] is False
    assert continuation["result_type"] == (
        "tools.run_live_gm1_tilestream_rk3_stage_finish."
        "FinishedGm1LiveRk3Stage"
    )
    final = continuation["reported_metadata"]
    assert final["status"] == (
        "PASS_LIVE_300X300X75_COMPLETED_RK3_SPLIT_MASS_SAME_RESIDENCY"
    )
    assert final["completed_rk3_token_sha256"] == (
        "198ca6fd3454791d16153d07da7bb16a62237c7dfa8974f1ef3d400850e91ba8"
    )
    assert final["mass_finish_token_sha256"] == (
        "22ac1bc809374190fccea35505eb5aa6f04e7226b97a60b8661ccc986524cf76"
    )
    assert final["mass_output_sha256"] == {
        "pp3d": "770205963794427fdc29ba7c850f5a0f8617f1977d10a62642b5222f6e8131ac",
        "prs": "3d650d52944c08ba77fd430cd439a9b5e48be8846eac06c9a1cd7149e3cf3ece",
        "rho": "d4775656c0c2286435fe00d7b6534bfb7ab17e93000689adb9a78be9cb2b92a5",
    }
    assert final["tke"]["tile_count"] == 16
    assert final["tke"]["tile_plan_sha256"] == (
        "131d27cf17e6e25b130f324cb81ed99ccb40cd9499aa2792d27014f64085fbfc"
    )
    assert final["tke"]["output_tke3d_sha256"] == (
        "19b939e5f5e2fad4c822a09832f976251458533c178c9ac4c612878381600d4b"
    )
    assert tuple(final["component_operation_token_sha256"]) == (
        "acoustic_continuation",
        "apmasscon_finish",
        "apmasscon_reduction",
        "eqtset2_remove_eos",
        "mass_finish",
        "momentum_boundary",
        "post_acoustic_flux",
        "pre_acoustic_momentum_rk",
        "scalar_moist_rk",
        "theta_divergence",
        "tke_rk",
    )
    assert final["same_stream_before_upstream_teardown"] is True
    assert final["solve3_entered"] is False
    assert final["full_forecast"] is False
