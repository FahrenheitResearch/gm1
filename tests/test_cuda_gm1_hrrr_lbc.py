from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.cuda_gm1_hrrr_lbc import (
    GM1_HRRR_LBC_CUDA_SOURCE_BYTES,
    GM1_HRRR_LBC_CUDA_SOURCE_SHA256,
    GM1_HRRR_LBC_KERNEL_NAMES,
    GM1_HRRR_LBC_REQUIRED_ARRAYS,
    HrrrLbcFrame,
    compile_cuda_gm1_hrrr_lbc,
    hrrr_lbc_davies_c2_spatial_weight,
    hrrr_lbc_davies_c2_weight,
    hrrr_lbc_edge_weight,
    resolve_hrrr_lbc_stage_clock,
)
from cm1core.errors import CM1Refusal


HASH = "a" * 64


def arrays(nx: int = 4, ny: int = 6, nz: int = 3) -> dict[str, np.ndarray]:
    return {
        "th0": np.zeros((nz,), dtype="<f4"),
        "u3d": np.zeros((nz, ny, nx + 1), dtype="<f4"),
        "v3d": np.zeros((nz, ny + 1, nx), dtype="<f4"),
        "w3d": np.zeros((nz + 1, ny, nx), dtype="<f4"),
        "th3d": np.zeros((nz, ny, nx), dtype="<f4"),
        "qv": np.zeros((nz, ny, nx), dtype="<f4"),
        "qc": np.zeros((nz, ny, nx), dtype="<f4"),
        "qr": np.zeros((nz, ny, nx), dtype="<f4"),
        "qi": np.zeros((nz, ny, nx), dtype="<f4"),
        "qs": np.zeros((nz, ny, nx), dtype="<f4"),
        "qg": np.zeros((nz, ny, nx), dtype="<f4"),
    }


def frame(values: dict[str, np.ndarray] | None = None) -> HrrrLbcFrame:
    return HrrrLbcFrame(values or arrays(), HASH, HASH, HASH, 3600.0)


def test_cuda_source_digest_and_entry_inventory_are_pinned() -> None:
    assert (
        hashlib.sha256(GM1_HRRR_LBC_CUDA_SOURCE_BYTES).hexdigest()
        == GM1_HRRR_LBC_CUDA_SOURCE_SHA256
    )
    source = GM1_HRRR_LBC_CUDA_SOURCE_BYTES.decode()
    assert len(GM1_HRRR_LBC_KERNEL_NAMES) == 5
    for name in GM1_HRRR_LBC_KERNEL_NAMES:
        assert f"void {name}(" in source


def test_quadratic_edge_weight_clamps_edge_and_leaves_interior() -> None:
    assert hrrr_lbc_edge_weight(0, 20) == 1.0
    assert hrrr_lbc_edge_weight(10, 20) == 0.25
    assert hrrr_lbc_edge_weight(19, 20) == pytest.approx(0.0025)
    assert hrrr_lbc_edge_weight(20, 20) == 0.0
    assert hrrr_lbc_edge_weight(200, 20) == 0.0


def test_davies_c2_taper_is_exact_at_boundaries_and_uses_stage_dt_over_tau() -> None:
    weights = [hrrr_lbc_davies_c2_spatial_weight(d, 20) for d in range(21)]
    assert weights[0] == 1.0
    assert weights[20] == 0.0
    assert all(left > right for left, right in zip(weights[:20], weights[1:]))
    x = 1.0 / 20.0
    assert weights[19] == pytest.approx(x**3 * (x * (6.0 * x - 15.0) + 10.0))
    assert hrrr_lbc_davies_c2_weight(
        0, 20, stage_timestep_s=0.5, relaxation_timescale_s=300.0
    ) == 1.0
    assert hrrr_lbc_davies_c2_weight(
        10, 20, stage_timestep_s=0.5, relaxation_timescale_s=300.0
    ) == pytest.approx(weights[10] * 0.5 / 300.0)
    assert hrrr_lbc_davies_c2_weight(
        20, 20, stage_timestep_s=0.5, relaxation_timescale_s=300.0
    ) == 0.0


@pytest.mark.parametrize(
    "distance,width,stage_dt,tau",
    [(-1, 20, 0.5, 300.0), (0, 1, 0.5, 300.0), (0, 20, 0.0, 300.0), (0, 20, 2.0, 1.0)],
)
def test_davies_c2_taper_fails_closed_on_invalid_controls(
    distance: int, width: int, stage_dt: float, tau: float
) -> None:
    with pytest.raises(ValueError):
        hrrr_lbc_davies_c2_weight(
            distance,
            width,
            stage_timestep_s=stage_dt,
            relaxation_timescale_s=tau,
        )


@pytest.mark.parametrize("distance,width", [(-1, 2), (True, 2), (0, 0), (0, True)])
def test_edge_weight_rejects_invalid_geometry(distance: object, width: object) -> None:
    with pytest.raises(ValueError):
        hrrr_lbc_edge_weight(distance, width)  # type: ignore[arg-type]


def test_frame_admits_exact_real_driver_inventory_and_shapes() -> None:
    candidate = frame()
    candidate.validate(nx=4, ny=6, nz=3)
    assert tuple(candidate.arrays) == GM1_HRRR_LBC_REQUIRED_ARRAYS


def test_frame_fails_closed_on_inventory_shape_dtype_hash_and_time() -> None:
    missing = arrays()
    del missing["qg"]
    with pytest.raises(ValueError, match="inventory"):
        frame(missing).validate(nx=4, ny=6, nz=3)

    wrong_shape = arrays()
    wrong_shape["u3d"] = np.zeros((3, 6, 4), dtype="<f4")
    with pytest.raises(ValueError, match="u3d shape"):
        frame(wrong_shape).validate(nx=4, ny=6, nz=3)

    wrong_dtype = arrays()
    wrong_dtype["qv"] = np.zeros((3, 6, 4), dtype="<f8")
    with pytest.raises(ValueError, match="qv must"):
        frame(wrong_dtype).validate(nx=4, ny=6, nz=3)

    with pytest.raises(ValueError, match="artifact hash"):
        HrrrLbcFrame(arrays(), "bad", HASH, HASH, 3600.0).validate(nx=4, ny=6, nz=3)
    with pytest.raises(ValueError, match="model time"):
        HrrrLbcFrame(arrays(), HASH, HASH, HASH, 0.0).validate(nx=4, ny=6, nz=3)


def test_legacy_clock_is_unchanged_and_explicit_clock_uses_cm1_stage_times() -> None:
    forecast = SimpleNamespace(
        time_seconds=10.0, selection=SimpleNamespace(dt_s=0.5)
    )
    legacy = SimpleNamespace(
        initialization_source_offset_s=0.0,
        source_start_offset_s=0.0,
        source_end_offset_s=3600.0,
        valid_offset_s=3600.0,
    )
    for nrk in (1, 2, 3):
        result = resolve_hrrr_lbc_stage_clock(legacy, forecast, nrk=nrk)
        assert result["relative_model_time_s"] == 10.5
        assert result["absolute_source_time_s"] == 10.5
        assert result["temporal_interpolation_fraction"] == 10.5 / 3600.0
        assert result["clock_mode"] == "legacy-f00-f01-end-step"

    forecast.time_seconds = 0.0
    f01_f02 = SimpleNamespace(
        initialization_source_offset_s=3600.0,
        source_start_offset_s=3600.0,
        source_end_offset_s=7200.0,
        valid_offset_s=7200.0,
    )
    stage_times = [
        float(np.float32(np.float32(0.5) / np.float32(value)))
        for value in (3, 2, 1)
    ]
    for nrk, relative in zip((1, 2, 3), stage_times):
        point = {
            "relative_model_time_s": relative,
            "absolute_source_time_s": 3600.0 + relative,
            "temporal_interpolation_fraction": relative / 3600.0,
        }
        result = resolve_hrrr_lbc_stage_clock(
            f01_f02, forecast, nrk=nrk, clock_point=point
        )
        assert result == {**point, "clock_mode": "explicit-cm1-rk-stage"}


def test_explicit_clock_refuses_legacy_rebase_mismatch_and_past_end() -> None:
    forecast = SimpleNamespace(
        time_seconds=0.0, selection=SimpleNamespace(dt_s=0.5)
    )
    f01_f02 = SimpleNamespace(
        initialization_source_offset_s=3600.0,
        source_start_offset_s=3600.0,
        source_end_offset_s=7200.0,
        valid_offset_s=7200.0,
    )
    with pytest.raises(CM1Refusal, match="EXPLICIT-CLOCK-REQUIRED"):
        resolve_hrrr_lbc_stage_clock(f01_f02, forecast, nrk=1)
    stage = float(np.float32(np.float32(0.5) / np.float32(3.0)))
    wrong = {
        "relative_model_time_s": stage,
        "absolute_source_time_s": 3600.0 + stage,
        "temporal_interpolation_fraction": 0.25,
    }
    with pytest.raises(CM1Refusal, match="EXPLICIT-CLOCK-MISMATCH"):
        resolve_hrrr_lbc_stage_clock(
            f01_f02, forecast, nrk=1, clock_point=wrong
        )
    forecast.time_seconds = 3600.0
    past = {
        "relative_model_time_s": 3600.0 + stage,
        "absolute_source_time_s": 7200.0 + stage,
        "temporal_interpolation_fraction": (3600.0 + stage) / 3600.0,
    }
    with pytest.raises(CM1Refusal, match="EXPLICIT-CLOCK-RANGE"):
        resolve_hrrr_lbc_stage_clock(
            f01_f02, forecast, nrk=1, clock_point=past
        )


def test_explicit_clock_supports_contiguous_f02_f03_interval() -> None:
    forecast = SimpleNamespace(
        time_seconds=3600.0, selection=SimpleNamespace(dt_s=0.5)
    )
    f02_f03 = SimpleNamespace(
        initialization_source_offset_s=3600.0,
        source_start_offset_s=7200.0,
        source_end_offset_s=10800.0,
        valid_offset_s=10800.0,
    )
    stage = float(np.float32(np.float32(0.5) / np.float32(2.0)))
    point = {
        "relative_model_time_s": 3600.0 + stage,
        "absolute_source_time_s": 7200.0 + stage,
        "temporal_interpolation_fraction": stage / 3600.0,
    }
    assert resolve_hrrr_lbc_stage_clock(
        f02_f03, forecast, nrk=2, clock_point=point
    )["clock_mode"] == "explicit-cm1-rk-stage"


def test_cuda_davies_mass_kernel_matches_independent_cpu_oracle() -> None:
    if os.environ.get("GM1_RUN_CUDA_HRRR_LBC_TEST") != "1":
        pytest.skip("set only while holding the GPU mutex")
    import cupy as cp

    ni, nj, nk, width = 8, 10, 3, 4
    shape = (nk, nj, ni)
    current = np.linspace(-1.0, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)
    f00 = current + np.float32(0.125)
    f01 = current + np.float32(0.625)
    alpha = np.float32(0.375)
    stage_factor = np.float32(np.float32(0.5) / np.float32(300.0))
    expected = current.copy()
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                distance = min(i, ni - 1 - i, j, nj - 1 - j)
                if distance >= width:
                    continue
                target = np.float32(
                    f00[k, j, i]
                    + np.float32(alpha * np.float32(f01[k, j, i] - f00[k, j, i]))
                )
                if distance == 0:
                    weight = np.float32(1.0)
                else:
                    x = np.float32(np.float32(width - distance) / np.float32(width))
                    x2 = np.float32(x * x)
                    x3 = np.float32(x2 * x)
                    polynomial = np.float32(
                        x * np.float32(np.float32(6.0) * x - np.float32(15.0))
                        + np.float32(10.0)
                    )
                    weight = np.float32(stage_factor * np.float32(x3 * polynomial))
                expected[k, j, i] = np.float32(
                    current[k, j, i]
                    + np.float32(weight * np.float32(target - current[k, j, i]))
                )

    compiled = compile_cuda_gm1_hrrr_lbc()
    device = cp.asarray(current)
    threads = (128, 1, 1)
    words = int(np.prod(shape))
    compiled.kernels[3](
        ((words + 127) // 128, 1, 1),
        threads,
        (
            device,
            cp.asarray(f00),
            cp.asarray(f01),
            cp.uint32(ni),
            cp.uint32(nj),
            cp.uint32(nk),
            cp.uint32(width),
            cp.float32(alpha),
            cp.float32(stage_factor),
            cp.uint32(1),
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    actual = cp.asnumpy(device)
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
