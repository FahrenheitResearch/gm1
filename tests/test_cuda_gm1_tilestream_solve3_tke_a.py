from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import pytest

from cm1core.cuda_gm1_tilestream_solve3_tke_a import (
    AUTHORITY_ANCHORS,
    CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_KERNEL_NAME,
    CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE,
    CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE_SHA256,
    GM1_ACCEPTED_EVOLVED_TKEA_SHA256,
    GM1_SOURCE_IDENTITY_SHA256,
    Gm1TiledSolve3EvolvedTkeReplayToken,
    audit_gm1_tiled_solve3_tke_a_admission,
    cpu_gm1_tiled_solve3_tke_a_inplace,
    five_meter_tiled_solve3_tke_a_inventory,
    reconstruct_gm1_four_open_padded_tke,
    reconstruct_gm1_padded_tkea_baseline,
    require_live_gm1_tiled_solve3_tke_a,
    tiled_solve3_tke_a_buffer_nbytes,
    tiled_solve3_tke_a_buffer_shapes,
)


np = pytest.importorskip("numpy")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _replay_token() -> Gm1TiledSolve3EvolvedTkeReplayToken:
    receipt = {
        "schema": "gm1.tilestream-evolved-tke-checkpoint-replay-token/v1",
        "status": "ADMITTED_DETACHED_REAL_EVOLVED_TKE_REPLAY_ONLY",
        "source_identity_sha256": GM1_SOURCE_IDENTITY_SHA256,
        "physical_tkea_sha256": GM1_ACCEPTED_EVOLVED_TKEA_SHA256,
        "completed_rk3_live_token": False,
        "stream_pointer": None,
    }
    receipt["operation_token_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    return Gm1TiledSolve3EvolvedTkeReplayToken(MappingProxyType(receipt))


def test_source_is_one_literal_tke_copy_and_pinned() -> None:
    assert hashlib.sha256(CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE.encode()).hexdigest() == (
        CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE_SHA256
    )
    assert CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_KERNEL_NAME in (
        CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE
    )
    assert "tkea[z]=tke3d[z]" in CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE
    for forbidden in ("float*rr", "float*rf", "ppi[", "tha[", "qa[", "psolver"):
        assert forbidden not in CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE


def test_four_open_reconstruction_preserves_bcw_order_and_rigid_faces() -> None:
    physical = np.arange(4 * 3 * 5, dtype="<f4").reshape(4, 3, 5)
    physical[0] = np.float32(0.0)
    physical[-1] = np.float32(0.0)
    padded = reconstruct_gm1_four_open_padded_tke(physical)
    assert padded.shape == (4, 5, 7)
    assert np.array_equal(padded[:, 1:-1, 1:-1], physical)
    assert np.array_equal(padded[1:-1, 1:-1, 0], physical[1:-1, :, 0])
    assert np.array_equal(padded[1:-1, 1:-1, -1], physical[1:-1, :, -1])
    assert np.array_equal(padded[1:-1, 0, :], padded[1:-1, 1, :])
    assert np.array_equal(padded[1:-1, -1, :], padded[1:-1, -2, :])
    assert np.count_nonzero(padded[[0, -1]].view(np.uint32)) == 0


def test_baseline_padding_preserves_diagnosed_bottom_until_commit() -> None:
    physical = np.arange(4 * 3 * 5, dtype="<f4").reshape(4, 3, 5)
    physical[-1] = np.float32(0.0)
    padded = reconstruct_gm1_padded_tkea_baseline(physical)
    assert np.array_equal(padded[:, 1:-1, 1:-1], physical)
    assert np.array_equal(padded[0, 1:-1, 1:-1], physical[0])
    assert np.count_nonzero(padded[-1].view(np.uint32)) == 0


def test_tile_cpu_copy_leaves_solve3_top_level_untouched() -> None:
    shapes = tiled_solve3_tke_a_buffer_shapes(tile_nx=5, tile_ny=4, nk=3)
    arrays = {
        "tke3d": np.arange(np.prod(shapes["tke3d"]), dtype="<f4").reshape(
            shapes["tke3d"]
        ),
        "tkea": np.full(shapes["tkea"], np.float32(-9.0)),
    }
    cpu_gm1_tiled_solve3_tke_a_inplace(
        arrays, active_nx=3, active_ny=2, nk=3
    )
    assert np.array_equal(arrays["tkea"][:, :2, :3], arrays["tke3d"][:, :2, :3])
    assert np.all(arrays["tkea"][:, 2:, :] == np.float32(-9.0))
    assert np.all(arrays["tkea"][:, :2, 3:] == np.float32(-9.0))


def test_replay_token_is_structurally_refused_as_live() -> None:
    with pytest.raises(
        RuntimeError, match="persisted tkea is detached replay evidence"
    ):
        require_live_gm1_tiled_solve3_tke_a(_replay_token(), stream_pointer=123)


def test_admission_audit_does_not_relabel_fresh_or_persisted_tke() -> None:
    audit = audit_gm1_tiled_solve3_tke_a_admission()
    assert audit["status"] == "BLOCKED_LIVE_ADMITTED_DETACHED_REPLAY_ONLY"
    assert audit["fresh_closure_is_completed_rk3"] is False
    assert audit["persisted_post_commit_tkea_is_precommit_tke3d_snapshot"] is False
    assert tuple(audit["authority"]["anchors"]) == AUTHORITY_ANCHORS
    assert "rr/rf" in audit["structural_refusals"]
    assert "five-metre meteorological product" in audit["structural_refusals"]


def test_five_meter_inventory_is_tiled_capacity_only() -> None:
    memory = tiled_solve3_tke_a_buffer_nbytes(tile_nx=192, tile_ny=192, nk=75)
    assert memory == {
        "host_per_buffer_bytes": 22_118_400,
        "device_per_buffer_bytes": 22_118_400,
        "double_host_bytes": 44_236_800,
        "double_device_bytes": 44_236_800,
    }
    inventory = five_meter_tiled_solve3_tke_a_inventory()
    assert inventory["padded_plane"] == [3002, 3002]
    assert inventory["tile_count"] == 256
    assert inventory["live_completed_rk3_token_available"] is False
    assert inventory["five_meter_meteorological_product"] is False
