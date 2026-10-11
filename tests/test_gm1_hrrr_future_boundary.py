from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cm1core.cuda_gm1_hrrr_lbc import HrrrLbcFrame
from cm1core.gm1_hrrr_future_boundary import (
    GM1_FORCING_HISTORY_EXPORT_ABI,
    GM1_FORCING_HISTORY_EXPORT_RECEIPT_SCHEMA,
    bind_gm1_f01_f02_lateral_forcing,
    load_gm1_f02_forcing_export,
    load_gm1_hrrr_future_boundary_series,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_SERIES = (
    ROOT
    / "work"
    / "real-hrrr-20240521-19z-f02-f03"
    / "cm1-mapped-100m-f02-f03-v2-orientation-audited"
)
F01 = "28e89cae538cf349c713ea42630dd8cc664a084c0c88f71eac1b3a491f14d189"
F02 = "7994d10e1e910697e00ec91f3c3ad4b06197ce757d403ae5b2bbaa14ecbf3336"
F03 = "4a4cdd2aca1052b0f3ef4d170614e4cfdef1d410bf43777ce93332be497b8034"
CONTRACT = "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12"
RECEIPT = "483c0992613cd786baf058ea73ca36b127f25418eff65914e94cc696f141ec79"


def load_real_series():
    return load_gm1_hrrr_future_boundary_series(
        REAL_SERIES,
        expected_receipt_sha256=RECEIPT,
        expected_f01_artifact_sha256=F01,
        expected_f02_artifact_sha256=F02,
        expected_f03_artifact_sha256=F03,
        expected_contract_sha256=CONTRACT,
    )


@pytest.mark.skipif(not REAL_SERIES.is_dir(), reason="sealed real HRRR series not present")
def test_real_series_is_exact_f01_relative_f02_f03_clock() -> None:
    series = load_real_series()
    assert series.baseline_f01_artifact_sha256 == F01
    assert series.f02.artifact_sha256 == F02
    assert series.f03.artifact_sha256 == F03
    assert (
        series.f02.absolute_source_start_s,
        series.f02.absolute_source_end_s,
        series.f02.relative_model_start_s,
        series.f02.relative_model_end_s,
    ) == (3600.0, 7200.0, 0.0, 3600.0)
    assert (
        series.f03.absolute_source_start_s,
        series.f03.absolute_source_end_s,
        series.f03.relative_model_start_s,
        series.f03.relative_model_end_s,
    ) == (7200.0, 10800.0, 3600.0, 7200.0)


@pytest.mark.skipif(not REAL_SERIES.is_dir(), reason="sealed real HRRR series not present")
def test_real_series_refuses_wrong_independent_receipt_pin() -> None:
    with pytest.raises(ValueError, match="independent pin"):
        load_gm1_hrrr_future_boundary_series(
            REAL_SERIES,
            expected_receipt_sha256="0" * 64,
            expected_f01_artifact_sha256=F01,
            expected_f02_artifact_sha256=F02,
            expected_f03_artifact_sha256=F03,
            expected_contract_sha256=CONTRACT,
        )


def test_export_receipt_refuses_wrong_absolute_f02_clock_before_array_load() -> None:
    series = SimpleNamespace(
        contract_sha256=CONTRACT,
        f02=SimpleNamespace(artifact_sha256=F02, payload_sha256="b" * 64),
    )
    receipt = {
        "abi": GM1_FORCING_HISTORY_EXPORT_ABI,
        "array_count": 22,
        "contract_sha256": CONTRACT,
        "dt_f64_bits": "3fe0000000000000",
        "manifest_nbytes": 1,
        "manifest_sha256": "a" * 64,
        "model": "GM1",
        "model_time_f64_bits": "40ac200000000000",
        "publication_contract": "transport fixture; no atmospheric fields",
        "raw_nbytes": 1,
        "raw_sha256": "b" * 64,
        "schema": GM1_FORCING_HISTORY_EXPORT_RECEIPT_SCHEMA,
        "source_artifact_sha256": F02,
        "step": 14400,
        "table_nbytes": 1,
        "table_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="sealed f02 boundary"):
        load_gm1_f02_forcing_export(series, ROOT, receipt)


def test_runtime_binding_refuses_non_f02_frame_before_cuda() -> None:
    series = SimpleNamespace(
        contract_sha256=CONTRACT,
        f02=SimpleNamespace(
            absolute_source_start_s=3600.0,
            absolute_source_end_s=7200.0,
            relative_model_start_s=0.0,
            relative_model_end_s=3600.0,
            artifact_sha256=F02,
        ),
    )
    wrong = HrrrLbcFrame({}, "0" * 64, CONTRACT, "b" * 64, 7200.0)
    with pytest.raises(ValueError, match="sealed interval"):
        bind_gm1_f01_f02_lateral_forcing(object(), series, wrong)
