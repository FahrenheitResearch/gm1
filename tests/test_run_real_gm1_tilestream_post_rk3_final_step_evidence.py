from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / (
    "evidence/validation/"
    "gm1-t300-post-rk3-physics-solve3-local-v1-attempt-0001/receipt.json"
)
RECEIPT_SHA256 = (
    "be1c5ed82832e68e99d9c18915b06f726ad7afb2d2b39deb974004afae4ad8c9"
)


def test_real_post_rk3_final_step_evidence_is_sealed() -> None:
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_SHA256
    outer = json.loads(raw)
    assert outer["status"] == (
        "PASS_REAL_300X300X75_COMPLETED_RK2_RK3_GLOBAL_ACOUSTIC"
    )
    assert outer["producer"]["git_commit"] == (
        "9a4830f7d85eab17ab3c5803dba6a23698eb7b8c"
    )
    assert outer["same_resident_device_state"] is True
    assert outer["same_stream_before_upstream_teardown"] is True

    rk3 = outer["downstream_continuation"]
    assert rk3["invoked"] is True
    assert rk3["reported_metadata"]["continuation_invoked"] is True
    final_result = rk3["continuation"]
    assert final_result["invoked"] is True
    assert final_result["metadata_exposed"] is True
    assert final_result["result_type"] == (
        "tools.run_live_gm1_tilestream_post_rk3_final_step."
        "FinishedGm1LivePostRk3FinalStep"
    )

    final = final_result["reported_metadata"]
    assert final["schema"] == "gm1.cuda-child-post-final-rk/v2"
    assert final["status"] == (
        "PASS_LIVE_300X300X75_POST_RK3_PDEFQ_WSM6_SOLVE3"
    )
    assert final["operation_token_sha256"] == (
        "78669b19c23aadbc9dd52c2e258231c698d6306c1ea455bdcba86c591a0cdfb6"
    )
    assert final["completed_rk3_operation_token_sha256"] == (
        "25a8b5f7b8c8f0f88d07a413b2914e72eb08aab17a26808ed85e4c6d6e0f3778"
    )
    assert final["mass_finish_operation_token_sha256"] == (
        "5f7b283b2e63a21a2a465d6a0f492bd46d611279c137ee097675de0db79dc368"
    )
    assert final["pdefq_receipt_sha256"] == (
        "3ae3da6d62fe8cfb0391ad8f588cd390887a90236f07185de4df0990d349f287"
    )
    assert final["wsm6_operation_token_sha256"] == (
        "58a0d284425d7e8b0e6363af4da822e4616c40a1389c3efc40cf76459b540b2d"
    )
    assert final["post_wsm6_operation_token_sha256"] == (
        "cd1aa491c8928ecd85861e68abfcb83c98af83e6c14eb05b4b69db9b49e0b528"
    )
    assert final["completed_step"] == 1
    assert final["model_time_seconds"] == 0.5
    assert final["eqtset2_replayed"] is False
    assert final["apmasscon_replayed"] is False
    assert final["ready_event_recorded"] is True
    assert final["same_stream_before_upstream_teardown"] is True
    assert final["checkpoint_history_overlap_same_objects"] is True
    assert len(final["checkpoint_carrier_names"]) == 37
    assert len(final["history_carrier_names"]) == 13
    assert tuple(final["sequence"]) == (
        "consume completed RK3 after split apmasscon finish",
        "pdefq",
        "public ArWen WSM6",
        "post-WSM6 CM1 EOS and qa",
        "solve3 ppx and rho boundaries",
        "solve3 held-rate recapture",
        "solve3 momentum/TKE/scalar current-to-A",
        "solve3 rr/rf density publication",
    )
