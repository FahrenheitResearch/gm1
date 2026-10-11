from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from cm1core.cuda_gm1_child_large_step import (
    GM1_CHILD_RAYLEIGH_SOURCE,
    GM1_CHILD_RAYLEIGH_SOURCE_SHA256,
    GM1_CHILD_THETA_PACK_SOURCE,
    GM1_CHILD_THETA_PACK_SOURCE_SHA256,
    Gm1ChildFullStatePins,
    REAL_GM1_CHILD_FULL_STATE_PINS,
    child_terrain_rayleigh_shapes,
    compile_cuda_gm1_child_theta_pack,
    cpu_child_terrain_rayleigh_held_inplace,
    launch_cuda_gm1_child_terrain_rayleigh,
    load_gm1_child_full_state_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_FULL_STATE = ROOT / "work/gm1-t300-child-full-state-v1"


def _arrays(ni: int = 4, nj: int = 3, nk: int = 2) -> dict[str, np.ndarray]:
    shapes = child_terrain_rayleigh_shapes(ni, nj, nk)
    arrays = {
        name: np.zeros(shape, dtype=np.float32, order="C")
        for name, shape in shapes.items()
    }
    arrays["tauh"].fill(np.float32(0.75))
    arrays["taus"].fill(np.float32(0.5))
    arrays["tauf"].fill(np.float32(0.25))
    arrays["ua"][:] = np.arange(arrays["ua"].size, dtype=np.float32).reshape(
        arrays["ua"].shape
    ) * np.float32(0.125)
    arrays["va"][:] = np.arange(arrays["va"].size, dtype=np.float32).reshape(
        arrays["va"].shape
    ) * np.float32(-0.0625)
    arrays["wa"][:] = np.float32(2.0)
    arrays["tha"][:] = np.float32(-3.0)
    # Deliberately horizontally varying terrain base winds.  A former
    # u0[k]/v0[k] broadcast cannot reproduce these departures.
    arrays["u0"][:] = (
        np.arange(arrays["u0"].size, dtype=np.float32).reshape(arrays["u0"].shape)
        * np.float32(0.03125)
    )
    arrays["v0"][:] = (
        np.arange(arrays["v0"].size, dtype=np.float32).reshape(arrays["v0"].shape)
        * np.float32(-0.015625)
    )
    return arrays


def test_child_rayleigh_source_pin_and_staggering() -> None:
    assert hashlib.sha256(GM1_CHILD_RAYLEIGH_SOURCE.encode()).hexdigest() == (
        GM1_CHILD_RAYLEIGH_SOURCE_SHA256
    )
    shapes = child_terrain_rayleigh_shapes(300, 300, 75)
    assert shapes["u0"] == shapes["ua"] == (75, 300, 301)
    assert shapes["v0"] == shapes["va"] == (75, 301, 300)


def test_child_theta_pack_source_pin() -> None:
    assert hashlib.sha256(GM1_CHILD_THETA_PACK_SOURCE.encode()).hexdigest() == (
        GM1_CHILD_THETA_PACK_SOURCE_SHA256
    )


def test_child_rayleigh_cpu_source_order_and_broadcast_negative_control() -> None:
    arrays = _arrays()
    before_u = arrays["ua"].copy()
    before_v = arrays["va"].copy()
    cpu_child_terrain_rayleigh_held_inplace(arrays, ni=4, nj=3, nk=2)

    rate = np.float32(0.0033333333333)
    negh = np.multiply(np.float32(-rate), np.float32(0.5), dtype=np.float32)
    u_pair = np.add(np.float32(0.75), np.float32(0.75), dtype=np.float32)
    v_pair = np.add(np.float32(0.75), np.float32(0.75), dtype=np.float32)
    expected_u = np.multiply(
        np.multiply(negh, u_pair, dtype=np.float32),
        np.subtract(before_u, arrays["u0"], dtype=np.float32),
        dtype=np.float32,
    )
    expected_v = np.multiply(
        np.multiply(negh, v_pair, dtype=np.float32),
        np.subtract(before_v, arrays["v0"], dtype=np.float32),
        dtype=np.float32,
    )
    assert np.array_equal(arrays["uten1"].view(np.uint32), expected_u.view(np.uint32))
    assert np.array_equal(arrays["vten1"].view(np.uint32), expected_v.view(np.uint32))

    legacy_u = np.multiply(
        np.multiply(negh, u_pair, dtype=np.float32),
        np.subtract(before_u, arrays["u0"][:, :1, :1], dtype=np.float32),
        dtype=np.float32,
    )
    legacy_v = np.multiply(
        np.multiply(negh, v_pair, dtype=np.float32),
        np.subtract(before_v, arrays["v0"][:, :1, :1], dtype=np.float32),
        dtype=np.float32,
    )
    assert np.count_nonzero(legacy_u.view(np.uint32) != expected_u.view(np.uint32)) > 0
    assert np.count_nonzero(legacy_v.view(np.uint32) != expected_v.view(np.uint32)) > 0


def test_child_rayleigh_refuses_legacy_1d_base_wind() -> None:
    arrays = _arrays()
    arrays["u0"] = np.zeros((2,), dtype=np.float32)
    with pytest.raises(ValueError, match=r"u0.*\(2, 3, 5\)"):
        cpu_child_terrain_rayleigh_held_inplace(arrays, ni=4, nj=3, nk=2)


def test_child_full_state_pin_type_is_fail_closed() -> None:
    pins = Gm1ChildFullStatePins(
        manifest_sha256="0" * 64,
        first_step_sha256="1" * 64,
        checkpoint_admission_sha256="2" * 64,
        receipt_sha256="3" * 64,
        base_wind_sha256="4" * 63,
        producer_source_sha256="5" * 64,
        producer_executable_sha256="6" * 64,
    )
    with pytest.raises(ValueError, match="base_wind_sha256"):
        pins.validate()


@pytest.mark.skipif(not REAL_FULL_STATE.is_dir(), reason="real child artifact unavailable")
def test_final_real_child_envelope_and_base_wind_admit() -> None:
    loaded = load_gm1_child_full_state_envelope(
        REAL_FULL_STATE, pins=REAL_GM1_CHILD_FULL_STATE_PINS
    )
    assert loaded.u0.shape == (75, 300, 301)
    assert loaded.v0.shape == (75, 301, 300)
    assert loaded.first_step["lifecycle"]["completed_large_step"] is False
    assert loaded.checkpoint_admission["carrier_count"] == 37


@pytest.mark.skipif(not REAL_FULL_STATE.is_dir(), reason="real child artifact unavailable")
def test_superseded_pre_clippy_tuple_is_refused() -> None:
    stale = Gm1ChildFullStatePins(
        manifest_sha256="148e40cfbc2105d31534d20e6c7b2937eef79a1128c425cd2fd06d44b71a3c3f",
        first_step_sha256="2e78871c66e74a4eb3f3e5d05d2aadf768191a161d0941566536185d8ab61b74",
        checkpoint_admission_sha256=(
            "9d9b1ba71cb995233248ca7d1849724e2e42b0f5a3554ad3ddb4caef89c18688"
        ),
        receipt_sha256="dc1b3c5ca16d9b9b26bad3171685101a577770703fdc7f9d0f61f4f2520caea4",
        base_wind_sha256=REAL_GM1_CHILD_FULL_STATE_PINS.base_wind_sha256,
        producer_source_sha256=(
            "380f0ca132b276a1ba842b1449283409ea0f5fad0b6d5440669f870e6eb6d299"
        ),
        producer_executable_sha256=(
            "512505445f14f5fb8dedfed358e2933ce4339127f80e9c6b128774e6e9bedad0"
        ),
    )
    with pytest.raises(Exception, match="GM1-CHILD-FULL-STATE-DRIFT"):
        load_gm1_child_full_state_envelope(REAL_FULL_STATE, pins=stale)




@pytest.mark.skipif(
    os.environ.get("GM1_TEST_CUDA") != "1", reason="requires explicit CUDA gate"
)
def test_child_rayleigh_compact_cuda_raw_bit_exact() -> None:
    cupy = pytest.importorskip("cupy")
    host = _arrays(ni=5, nj=4, nk=3)
    expected = {name: value.copy() for name, value in host.items()}
    cpu_child_terrain_rayleigh_held_inplace(expected, ni=5, nj=4, nk=3)
    device = {name: cupy.asarray(value) for name, value in host.items()}
    launch = launch_cuda_gm1_child_terrain_rayleigh(
        device, ni=5, nj=4, nk=3
    )
    launch.ready_event.synchronize()
    for name in ("uten1", "vten1", "wten1", "thten1"):
        actual = cupy.asnumpy(device[name])
        assert np.array_equal(actual.view(np.uint32), expected[name].view(np.uint32))
    assert launch.receipt["base_wind"]["legacy_1d_broadcast_allowed"] is False


@pytest.mark.skipif(
    os.environ.get("GM1_TEST_CUDA") != "1", reason="requires explicit CUDA gate"
)
def test_child_theta_pack_compact_cuda_raw_bit_and_broadcast_negative_control() -> None:
    cupy = pytest.importorskip("cupy")
    ni, nj, nk, ng = 5, 4, 3, 3
    words = ni * nj * nk
    th0 = (
        np.arange(words, dtype=np.float32).reshape(nk, nj, ni)
        * np.float32(0.0625)
        + np.float32(300.0)
    )
    theta = (
        np.arange(words, dtype=np.float32).reshape(nk, nj, ni)
        * np.float32(-0.03125)
    )
    expected = np.add(
        np.subtract(th0, np.float32(300.0), dtype=np.float32),
        theta,
        dtype=np.float32,
    )
    legacy = np.add(
        np.subtract(th0[:, :1, :1], np.float32(300.0), dtype=np.float32),
        theta,
        dtype=np.float32,
    )
    assert np.count_nonzero(expected.view(np.uint32) != legacy.view(np.uint32)) > 0
    d_th0 = cupy.asarray(th0)
    d_theta = cupy.asarray(theta)
    padded = cupy.zeros((nk + 2, nj + 2 * ng, ni + 2 * ng), dtype=cupy.float32)
    compiled = compile_cuda_gm1_child_theta_pack()
    compiled.kernel(
        ((words + 255) // 256, 1, 1),
        (256, 1, 1),
        (
            d_th0,
            d_theta,
            padded,
            cupy.uint32(ni),
            cupy.uint32(nj),
            cupy.uint32(nk),
            cupy.uint32(ng),
        ),
    )
    actual = cupy.asnumpy(padded[1 : nk + 1, ng : ng + nj, ng : ng + ni])
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
