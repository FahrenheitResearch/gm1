from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cm1_marks

from authority_trees import requires_native_reference
import cm1core.cuda_gm1_forecast as forecast_module
from cm1core.cuda_gm1_forecast import (
    AUTHORITY_SOURCE_SHA256,
    DEFAULT_GM1_SELECTION,
    GM1_FORECAST_ABI,
    GM1_FORECAST_CUDA_SOURCE,
    GM1_FORECAST_CUDA_SOURCE_SHA256,
    GM1_FORECAST_KERNEL_NAMES,
    GM1_FORECAST_SCHEMA,
    GM1_HRRR_DAVIES_LATERAL_FORCING_SEMANTIC_ID,
    GM1_HRRR_F00_F01_SOURCE_CLOCK,
    GM1_HRRR_F01_F02_SOURCE_CLOCK,
    GM1_KESSLER_LARGE_STEP_SEMANTIC_ID,
    Gm1ForecastLateralForcing,
    Gm1ForecastSelection,
    Gm1ForecastSourceClock,
    _canonical_arch,
    _launch_gm1_lateral_forcing_stage,
    compile_cuda_gm1_forecast,
    gm1_source_clock_point,
    launch_cuda_gm1_large_step,
    validate_gm1_initialized_product_microphysics,
    validate_gm1_lateral_forcing,
    validate_gm1_selection,
    validate_gm1_terrain_open_backend,
)
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]


def test_import_is_cuda_cold_source_pinned_and_gm1_branded() -> None:
    command = (
        "import sys; import cm1core.cuda_gm1_forecast as module; "
        "assert module.GM1_FORECAST_SCHEMA.startswith('gm1.'); "
        "assert not any(name == 'cupy' or name.startswith('cupy.') "
        "for name in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert hashlib.sha256(GM1_FORECAST_CUDA_SOURCE.encode()).hexdigest() == (
        GM1_FORECAST_CUDA_SOURCE_SHA256
    )
    assert all(name in GM1_FORECAST_CUDA_SOURCE for name in GM1_FORECAST_KERNEL_NAMES)
    assert GM1_FORECAST_SCHEMA == "gm1.cuda-real-forecast/v1"
    assert GM1_FORECAST_ABI == "gm1.cuda-real-forecast-kernels/v1"


@requires_native_reference
def test_authority_sources_are_exact_cm1_r21_1_bytes() -> None:
    authority = ROOT / "work/native-reference/cm1r21.1"
    for relative, expected in AUTHORITY_SOURCE_SHA256.items():
        assert (
            hashlib.sha256((authority / relative).read_bytes()).hexdigest() == expected
        )


def test_real_100_m_configuration_is_a_preset_not_the_model_boundary() -> None:
    validate_gm1_selection(DEFAULT_GM1_SELECTION)
    assert DEFAULT_GM1_SELECTION == Gm1ForecastSelection(
        nx=300,
        ny=300,
        nz=75,
        dx_m=100.0,
        dy_m=100.0,
        dz_m=200.0,
        dt_s=0.5,
        acoustic_substeps=6,
        equation_set=2,
        psolver=3,
        terrain=True,
        lateral_boundaries=(2, 2, 2, 2),
        microphysics="wsm6-mp6",
        dry_mass_correction=True,
        coriolis=0,
        moving_frame=False,
    )
    validate_gm1_selection(Gm1ForecastSelection(dx_m=200.0))
    validate_gm1_selection(Gm1ForecastSelection(coriolis=1))
    with pytest.raises(ValueError, match="not routed"):
        validate_gm1_terrain_open_backend(Gm1ForecastSelection(dx_m=200.0))


def test_cropped_validation_changes_only_horizontal_counts() -> None:
    validate_gm1_terrain_open_backend(
        Gm1ForecastSelection(nx=24, ny=19), allow_cropped_validation=True
    )
    with pytest.raises(ValueError, match="not routed"):
        validate_gm1_terrain_open_backend(
            Gm1ForecastSelection(nx=24, ny=19, nz=74),
            allow_cropped_validation=True,
        )


def test_kessler_semantic_is_routed_but_wsm6_product_relabel_is_refused() -> None:
    kessler = Gm1ForecastSelection(microphysics=GM1_KESSLER_LARGE_STEP_SEMANTIC_ID)
    validate_gm1_terrain_open_backend(kessler)
    with pytest.raises(CM1Refusal, match="GM1-KESSLER-REAL-INITIALIZATION-UNAVAILABLE"):
        validate_gm1_initialized_product_microphysics(kessler)
    validate_gm1_initialized_product_microphysics(DEFAULT_GM1_SELECTION)
    with pytest.raises(ValueError, match="microphysics semantic"):
        validate_gm1_terrain_open_backend(
            Gm1ForecastSelection(microphysics="unported-scheme")
        )


def test_source_clock_maps_f00_f01_and_f01_f02_without_rebasing() -> None:
    legacy = gm1_source_clock_point(GM1_HRRR_F00_F01_SOURCE_CLOCK, 1800.0)
    assert legacy.absolute_source_time_s == 1800.0
    assert legacy.temporal_interpolation_fraction == 0.5
    future = gm1_source_clock_point(GM1_HRRR_F01_F02_SOURCE_CLOCK, 1800.0)
    assert future.relative_model_time_s == 1800.0
    assert future.absolute_source_time_s == 5400.0
    assert future.temporal_interpolation_fraction == 0.5
    end = gm1_source_clock_point(GM1_HRRR_F01_F02_SOURCE_CLOCK, 3600.0)
    assert end.absolute_source_time_s == 7200.0
    assert end.temporal_interpolation_fraction == 1.0
    with pytest.raises(CM1Refusal, match="GM1-HRRR-LBC-SOURCE-CLOCK-RANGE"):
        gm1_source_clock_point(GM1_HRRR_F01_F02_SOURCE_CLOCK, 3600.5)


def test_source_clock_refuses_wrong_absolute_offsets_and_discontinuous_intervals() -> (
    None
):
    wrong_hour = Gm1ForecastSourceClock(3600.0, 3600.0, 7100.0, 0.0, 3500.0, 1, 1, 2)
    with pytest.raises(CM1Refusal, match="GM1-HRRR-LBC-SOURCE-HOUR-MISMATCH"):
        gm1_source_clock_point(wrong_hour, 0.0)
    gap = Gm1ForecastSourceClock(3600.0, 7200.0, 10800.0, 0.0, 3600.0, 1, 2, 3)
    with pytest.raises(CM1Refusal, match="GM1-HRRR-LBC-SOURCE-OFFSET-MISMATCH"):
        gm1_source_clock_point(gap, 0.0)


@requires_native_reference
def test_lateral_forcing_audits_three_stage_clock_and_quadratic_blend_receipt() -> None:
    state = SimpleNamespace(
        time_seconds=0.0,
        selection=SimpleNamespace(dt_s=0.5),
        product=SimpleNamespace(
            export=SimpleNamespace(
                source_artifact_sha256=(
                    "28e89cae538cf349c713ea42630dd8cc664a084c0c88f71eac1b3a491f14d189"
                )
            )
        ),
    )

    def apply_stage(_state: object, nrk: int, point: object) -> dict[str, object]:
        return {
            "nrk": nrk,
            "target_model_time_s": point.relative_model_time_s,
            "target_absolute_source_time_s": point.absolute_source_time_s,
            "temporal_interpolation_fraction": point.temporal_interpolation_fraction,
            "relaxation_width_points": 20,
            "edge_weight": 1.0,
            "inner_zone_weight": 0.0025,
        }

    forcing = Gm1ForecastLateralForcing(GM1_HRRR_F01_F02_SOURCE_CLOCK, apply_stage)
    validate_gm1_lateral_forcing(forcing)
    source = (ROOT / "work/native-reference/cm1r21.1/src/solve2.F").read_text()
    # The RK stage step: dt over (nrkmax + 1 - nrk).
    assert cm1_marks.has(source, "e28a3da8130b4836/28")
    expected_stage_dt = [
        float(np.float32(np.float32(0.5) / np.float32(denominator)))
        for denominator in (3, 2, 1)
    ]
    results = [
        _launch_gm1_lateral_forcing_stage(state, forcing, nrk) for nrk in (1, 2, 3)
    ]
    assert [item["rk_stage_timestep_s"] for item in results] == expected_stage_dt
    for nrk, stage_dt, result in zip((1, 2, 3), expected_stage_dt, results):
        assert result["clock_point"]["relative_model_time_s"] == stage_dt
        assert result["clock_point"]["absolute_source_time_s"] == 3600.0 + stage_dt
        assert result["clock_point"]["temporal_interpolation_fraction"] == (
            stage_dt / 3600.0
        )
        assert result["operator"]["nrk"] == nrk
        assert result["operator"]["target_absolute_source_time_s"] == (
            3600.0 + stage_dt
        )

    def rectangular(_state: object, nrk: int, point: object) -> dict[str, object]:
        result = apply_stage(_state, nrk, point)
        result["inner_zone_weight"] = 0.0
        return result

    with pytest.raises(CM1Refusal, match="GM1-HRRR-LBC-BLEND-CONTRACT"):
        _launch_gm1_lateral_forcing_stage(
            state,
            Gm1ForecastLateralForcing(GM1_HRRR_F01_F02_SOURCE_CLOCK, rectangular),
            2,
        )
    with pytest.raises(CM1Refusal, match="GM1-HRRR-LBC-BLEND-WIDTH"):
        validate_gm1_lateral_forcing(
            Gm1ForecastLateralForcing(
                GM1_HRRR_F01_F02_SOURCE_CLOCK, apply_stage, relaxation_width_points=1
            )
        )
    wrong_initialization = SimpleNamespace(
        time_seconds=0.0,
        selection=SimpleNamespace(dt_s=0.5),
        product=SimpleNamespace(
            export=SimpleNamespace(
                source_artifact_sha256=(
                    "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f"
                )
            )
        ),
    )
    with pytest.raises(CM1Refusal, match="GM1-HRRR-LBC-INITIALIZATION-CLOCK"):
        _launch_gm1_lateral_forcing_stage(wrong_initialization, forcing, 1)
    with pytest.raises(ValueError, match="cropped validation"):
        validate_gm1_terrain_open_backend(
            Gm1ForecastSelection(nx=9, ny=19), allow_cropped_validation=True
        )


def test_lateral_forcing_audits_davies_c2_dttmp_over_tau_receipt() -> None:
    state = SimpleNamespace(
        time_seconds=0.0,
        selection=SimpleNamespace(dt_s=0.5),
        product=SimpleNamespace(
            export=SimpleNamespace(
                source_artifact_sha256=(
                    "28e89cae538cf349c713ea42630dd8cc664a084c0c88f71eac1b3a491f14d189"
                )
            )
        ),
    )
    width = 40
    tau = 300.0

    def apply_stage(_state: object, nrk: int, point: object) -> dict[str, object]:
        stage_dt = float(np.float32(np.float32(0.5) / np.float32(4 - nrk)))
        stage_factor = float(np.float32(np.float32(stage_dt) / np.float32(tau)))
        x = 1.0 / width
        spatial = x**3 * (x * (6.0 * x - 15.0) + 10.0)
        effective = spatial * stage_factor
        return {
            "nrk": nrk,
            "target_model_time_s": point.relative_model_time_s,
            "target_absolute_source_time_s": point.absolute_source_time_s,
            "temporal_interpolation_fraction": point.temporal_interpolation_fraction,
            "relaxation_width_points": width,
            "relaxation_mode": "davies-c2-stage-tendency",
            "relaxation_timescale_s": tau,
            "rk_stage_timestep_s": stage_dt,
            "stage_relaxation_factor": stage_factor,
            "edge_weight": 1.0,
            "inner_zone_spatial_weight": spatial,
            "inner_zone_effective_weight": effective,
            "inner_zone_weight": effective,
        }

    forcing = Gm1ForecastLateralForcing(
        GM1_HRRR_F01_F02_SOURCE_CLOCK,
        apply_stage,
        relaxation_width_points=width,
        semantic_id=GM1_HRRR_DAVIES_LATERAL_FORCING_SEMANTIC_ID,
        relaxation_mode="davies-c2-stage-tendency",
        relaxation_timescale_s=tau,
    )
    receipts = [
        _launch_gm1_lateral_forcing_stage(state, forcing, nrk) for nrk in (1, 2, 3)
    ]
    assert [item["operator"]["edge_weight"] for item in receipts] == [1.0] * 3
    assert all(item["operator"]["inner_zone_weight"] > 0.0 for item in receipts)
    assert all(
        item["operator"]["inner_zone_weight"]
        < item["operator"]["inner_zone_spatial_weight"]
        for item in receipts
    )
    assert all("dttmp/tau" in item["blend"] for item in receipts)

    def wrong_factor(_state: object, nrk: int, point: object) -> dict[str, object]:
        receipt = apply_stage(_state, nrk, point)
        receipt["stage_relaxation_factor"] = 1.0
        return receipt

    with pytest.raises(CM1Refusal, match="GM1-HRRR-LBC-DAVIES-CONTRACT"):
        _launch_gm1_lateral_forcing_stage(
            state,
            Gm1ForecastLateralForcing(
                GM1_HRRR_F01_F02_SOURCE_CLOCK,
                wrong_factor,
                relaxation_width_points=width,
                semantic_id=GM1_HRRR_DAVIES_LATERAL_FORCING_SEMANTIC_ID,
                relaxation_mode="davies-c2-stage-tendency",
                relaxation_timescale_s=tau,
            ),
            2,
        )


@pytest.mark.parametrize(
    "value,expected", [("compute_86", "86"), ("sm_86", "86"), ("8.6", "86")]
)
def test_cuda_architecture_is_canonical(value: str, expected: str) -> None:
    assert _canonical_arch(value) == expected
    with pytest.raises(ValueError, match="canonical digits"):
        _canonical_arch("native")


def test_product_launch_requires_a_real_forecast_state() -> None:
    with pytest.raises(TypeError, match="CudaGm1ForecastState"):
        launch_cuda_gm1_large_step(object(), solve1_hook=None)  # type: ignore[arg-type]


def test_pre_wsm6_hook_orders_apmasscon_then_pdefq_without_touching_qa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    fields = {
        "rho": np.ones((2, 3, 4), dtype=np.float32),
        "th3d": np.ones((2, 3, 4), dtype=np.float32),
        "q3d": np.ones((6, 2, 3, 4), dtype=np.float32),
        "qa": np.full((6, 2, 3, 4), 7.0, dtype=np.float32),
        "pp3d": np.ones((2, 3, 4), dtype=np.float32),
        "prs": np.ones((2, 3, 4), dtype=np.float32),
    }

    def launch_apmasscon(arrays: dict[str, object], **kwargs: object) -> object:
        calls.append(("apmasscon", arrays))
        return SimpleNamespace(receipt={"schema": "apmasscon-test"})

    def launch_pdefq(arrays: dict[str, object], **kwargs: object) -> object:
        calls.append(("pdefq", arrays))
        return SimpleNamespace(
            receipt={
                "schema": "gpuwm-cm1.cuda-wsm6-pdefq/v1",
                "cadence": (
                    "once after final-RK dynamics/apmasscon and before WSM6; "
                    "solve3 later copies q3d to qa"
                ),
            }
        )

    monkeypatch.setattr(forecast_module, "launch_cuda_apmasscon", launch_apmasscon)
    monkeypatch.setattr(
        forecast_module,
        "launch_cuda_wsm6_pdefq_pdscheme1",
        launch_pdefq,
    )
    state = SimpleNamespace(
        product=SimpleNamespace(field=lambda name: fields[name]),
        prepared=SimpleNamespace(
            arrays={
                "uh": np.ones(10, dtype=np.float32),
                "vh": np.ones(9, dtype=np.float32),
            },
            metrics={"rmh": np.ones((2, 3, 4), dtype=np.float32)},
        ),
        selection=SimpleNamespace(
            nx=4,
            ny=3,
            nz=2,
            dx_m=100.0,
            dy_m=100.0,
            dz_m=200.0,
            microphysics=forecast_module.GM1_WSM6_MICROPHYSICS_SEMANTIC_ID,
        ),
        pi0_3d=np.ones((2, 3, 4), dtype=np.float32),
        th0_3d=np.ones((2, 3, 4), dtype=np.float32),
        mass1_f64=1.0,
        stream=object(),
        apmasscon_compiled=object(),
        pdefq_compiled=object(),
        pdefq_workspace=object(),
        pdefq_asq=np.zeros(6, dtype=np.float64),
        retained=[],
    )
    qa_before = fields["qa"].copy()

    receipt = forecast_module._apmasscon_hook(state)(object())

    assert [name for name, _arrays in calls] == ["apmasscon", "pdefq"]
    assert calls[1][1]["q3d"] is fields["q3d"]
    assert "qa" not in calls[1][1]
    np.testing.assert_array_equal(fields["qa"], qa_before)
    assert receipt["pdefq"]["schema"] == "gpuwm-cm1.cuda-wsm6-pdefq/v1"
    assert len(state.retained) == 2


def test_cuda_forecast_kernels_compile_only_under_external_gpu_mutex() -> None:
    if os.environ.get("GM1_RUN_CUDA_FORECAST_COMPILE_TEST") != "1":
        pytest.skip("set only while holding the GPU mutex")
    compiled = compile_cuda_gm1_forecast()
    assert compiled.target_arch.isdigit()
    assert len(compiled.kernels) == len(GM1_FORECAST_KERNEL_NAMES)
    assert compiled.ptx.startswith(b"//")
    assert compiled.cubin.startswith(b"\x7fELF\x02\x01\x01")
