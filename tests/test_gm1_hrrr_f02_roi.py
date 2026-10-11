from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from cm1core.gm1_hrrr_f02_roi import (
    load_gm1_hrrr_f02_roi_series,
    validate_gm1_hrrr_f02_roi_series,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_ROI = (
    ROOT
    / "work/real-hrrr-20240521-19z-f02-f03"
    / "cm1-mapped-100m-f02-f03-v3-storm-centered"
)
RECEIPT = "7c2c5d04f6d3a25a7cb6e9508ba95cba18718fb998cece0c49a1d068de16956e"
F01 = "7ac37586d4a2361af384026d48c7505199b41040ff7c2f4860f857690abfd61d"
F02 = "356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d"
F03 = "a2b3d2d3009a432826e602ad5befd415ddd30667d58baac9679ccded87f5fe98"
CONTRACT = "a9ccd6f0c6650fd89cd42f16977209f57695464181bcb62beff14e9a9f61d896"


def load_real_roi():
    return load_gm1_hrrr_f02_roi_series(
        REAL_ROI,
        expected_receipt_sha256=RECEIPT,
        expected_f01_artifact_sha256=F01,
        expected_f02_artifact_sha256=F02,
        expected_f03_artifact_sha256=F03,
        expected_contract_sha256=CONTRACT,
    )


@pytest.mark.skipif(not REAL_ROI.is_dir(), reason="sealed real f02 ROI not present")
def test_real_f02_roi_preserves_source_coordinates_and_places_storm_interior() -> None:
    binding = load_real_roi()
    assert binding.center_latitude == pytest.approx(41.39464041135)
    assert binding.center_longitude == pytest.approx(-94.28268293687)
    assert (binding.source_full_i, binding.source_full_j) == (989, 638)
    assert (binding.mapped_i, binding.mapped_j) == (157, 149)
    assert binding.mapped_edge_distance_points == 142
    assert binding.mapped_reflectivity_dbz > 60.0


@pytest.mark.skipif(not REAL_ROI.is_dir(), reason="sealed real f02 ROI not present")
def test_f02_roi_refuses_an_edge_relabel_without_remapping() -> None:
    binding = load_real_roi()
    receipt = deepcopy(binding.series.receipt)
    receipt["frames"][0]["diagnostics"]["refd_1000m_agl_dbz"]["maximum"][
        "distance_to_nearest_edge_cells"
    ] = 0
    tampered = replace(binding.series, receipt=receipt)
    with pytest.raises(ValueError, match="not admitted in the mapped parent interior"):
        validate_gm1_hrrr_f02_roi_series(tampered)


@pytest.mark.skipif(not REAL_ROI.is_dir(), reason="sealed real f02 ROI not present")
def test_f02_roi_refuses_wrong_independent_series_hash() -> None:
    with pytest.raises(ValueError, match="independent pin"):
        load_gm1_hrrr_f02_roi_series(
            REAL_ROI,
            expected_receipt_sha256="0" * 64,
            expected_f01_artifact_sha256=F01,
            expected_f02_artifact_sha256=F02,
            expected_f03_artifact_sha256=F03,
            expected_contract_sha256=CONTRACT,
        )
