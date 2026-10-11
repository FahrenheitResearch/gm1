from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / (
    "evidence/validation/"
    "gm1-t300-tilestream-post-rk3-restart-v1-attempt-0004/receipt.json"
)
RECEIPT_SHA256 = (
    "4f78ebe42e4fa4066d93fcdd440e79b4b9c37f05688b5bbc18340f8213ae2acd"
)


def test_real_post_rk3_publish_restore_continuation_evidence_is_sealed() -> None:
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_SHA256
    receipt = json.loads(raw)
    assert receipt["schema"] == (
        "gm1.real-50m-child-tilestream-post-rk3-restart-run/v1"
    )
    assert receipt["status"] == (
        "PASS_REAL_300X300X75_TILESTREAM_POST_RK3_PUBLISH_RESTART_ONE_STEP"
    )
    assert receipt["runner"]["sha256"] == (
        "dbaae03c17fee5031ebd46d2f42c595f56fee3a5adcf370107e3a8f54012f308"
    )
    assert receipt["operator_manifest_sha256"] == (
        "6112fb9302f57f82b09118b3f38c6ae8dd3d82c59eb14fd01940f2dc43428c09"
    )
    assert receipt["source"]["publisher_executable_sha256"] == (
        "40d09265e96e157cfc1ff55d6892a119ce1fa9e7e8edddb234648b018bc6a4da"
    )
    assert receipt["completed_rk3"]["completed_nrk"] == 3
    assert receipt["completed_rk3"]["status"] == (
        "ADMITTED_LIVE_TILESTREAM_COMPLETED_RK_STAGE"
    )
    assert receipt["completed_rk3"]["operation_token_sha256"] == (
        "ade61887674bc7407d06927bdbf7551957afa227f6400c0dc3f471af31e319df"
    )
    assert receipt["final_step_operation_token_sha256"] == (
        "13ee11c956a047c2e81ad0fd38be261cefae6424e44537c8ef55a303c1ab3acf"
    )

    checkpoint = receipt["raw_arenas"]["checkpoint"]
    history = receipt["raw_arenas"]["history"]
    assert checkpoint["sha256"] == (
        "d6ccdf8270953ab24412c1fc1900c9aca17b36063608ff3e9794bfc4db3f0e85"
    )
    assert history["sha256"] == (
        "4999f36bbe8c0e14a7abb13ec641512c9a58f5d860ab598b05958a9a4464e5dd"
    )
    assert checkpoint["nbytes"] == 976_696_992
    assert history["nbytes"] == 461_340_000
    assert receipt["raw_arenas"]["raw_staging_removed_after_restore"] is True

    publication = receipt["publication"]
    published = publication["publish"]
    assert publication["independent_inspect"] == published
    assert published["checkpoint_artifact_sha256"] == (
        "7e86477a726eef633ab5531b56c5b4badafa2eb8249a8a4509ce1282ea85e6d3"
    )
    assert published["history_artifact_sha256"] == (
        "dbd57a7f5f00d19ccd4929c87219294dce6c84567fc49b41f0c49a26f00c7ee0"
    )
    assert published["manifest_sha256"] == (
        "687b4de7ee61de115dea88536e1f9e7ff1f58bf155285f08215d1e7800f862d4"
    )
    assert published["receipt_sha256"] == (
        "849e153bbc995f72f75a4410cefa5c8a4a58ccdce3a8648a997512427b222e2b"
    )
    assert published["checkpoint_raw_sha256"] == checkpoint["sha256"]
    assert published["history_raw_sha256"] == history["sha256"]
    assert publication["restore"]["durable_receipt"]["status"] == (
        "PASS_RAW_BIT_RESTORE"
    )
    assert publication["restore"]["durable_receipt"][
        "checkpoint_arena_sha256"
    ] == checkpoint["sha256"]
    assert publication["restore"]["durable_receipt"][
        "history_arena_sha256"
    ] == history["sha256"]
    assert publication["ready_signal"] == "atomic final step-directory appearance"

    continuation = receipt["continuation"]
    assert continuation["completed_step"] == 2
    assert continuation["model_time_f64_bits"] == "3ff0000000000000"
    assert continuation["device_name"] == "NVIDIA GeForce RTX 3080"
    assert continuation["gpu_mutex_clean_release"] is True
    assert continuation["restore"]["status"] == (
        "READY_BEFORE_REAL_CHILD_LARGE_STEP_2"
    )
    assert continuation["restore"]["source_recovery_commit"] is None
    assert tuple(stage["nrk"] for stage in continuation["operators"]["rk_stages"]) == (
        1,
        2,
        3,
    )
    assert continuation["gates"]["all_finite"] is True
    assert continuation["gates"]["current_a_raw_bit_equal"] is True
    assert continuation["gates"]["qa_q3d_raw_bit_equal"] is True
    memory = continuation["restart_device_memory_before_state"]
    assert memory["pool_used_bytes"] == 0
    assert memory["pool_total_bytes"] == 0
    assert memory["device_free_bytes"] == 8_693_743_616
