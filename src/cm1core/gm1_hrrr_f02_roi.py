"""Strict source-coordinate admission for the real interior-storm f02 ROI.

Rust owns HRRR decoding and remapping.  This module accepts only the sealed
300x300x75, 100 m series whose target center is the real f02 low-level
reflectivity maximum.  No array is shifted, translated, or recomputed here.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cm1core.gm1_hrrr_future_boundary import (
    Gm1HrrrFutureBoundarySeries,
    load_gm1_hrrr_future_boundary_series,
)


GM1_F02_ROI_CENTER_LAT_F64_BITS = "4044b28393b636d0"
GM1_F02_ROI_CENTER_LON_F64_BITS = "c05792177a2c3f9c"
GM1_F02_ROI_SOURCE_FULL_I = 989
GM1_F02_ROI_SOURCE_FULL_J = 638
GM1_F02_ROI_TARGET_I = 157
GM1_F02_ROI_TARGET_J = 149
GM1_F02_ROI_MINIMUM_EDGE_DISTANCE = 100


@dataclass(frozen=True, slots=True)
class Gm1HrrrF02RoiBinding:
    series: Gm1HrrrFutureBoundarySeries
    center_latitude: float
    center_longitude: float
    source_full_i: int
    source_full_j: int
    mapped_i: int
    mapped_j: int
    mapped_edge_distance_points: int
    mapped_reflectivity_dbz: float


def _f64_bits(value: object, label: str) -> float:
    if not isinstance(value, str) or len(value) != 16:
        raise ValueError(f"{label} must be an exact float64 bit string")
    try:
        decoded = struct.unpack(">d", bytes.fromhex(value))[0]
    except (ValueError, struct.error) as error:
        raise ValueError(f"{label} must be hexadecimal float64 bits") from error
    if not math.isfinite(decoded):
        raise ValueError(f"{label} must be finite")
    return decoded


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _f32_bits(value: object, label: str) -> float:
    if not isinstance(value, str) or len(value) != 8:
        raise ValueError(f"{label} must be an exact float32 bit string")
    try:
        decoded = struct.unpack(">f", bytes.fromhex(value))[0]
    except (ValueError, struct.error) as error:
        raise ValueError(f"{label} must be hexadecimal float32 bits") from error
    if not math.isfinite(decoded):
        raise ValueError(f"{label} must be finite")
    return float(decoded)


def validate_gm1_hrrr_f02_roi_series(
    series: Gm1HrrrFutureBoundarySeries,
) -> Gm1HrrrF02RoiBinding:
    """Require the real f02 storm anchor to be well inside the mapped domain."""

    if not isinstance(series, Gm1HrrrFutureBoundarySeries):
        raise TypeError("series must be a strict Gm1HrrrFutureBoundarySeries")
    receipt = _object(series.receipt, "future-boundary receipt")
    target = _object(receipt.get("target"), "future-boundary target")
    if (
        target.get("center_lat_f64_bits") != GM1_F02_ROI_CENTER_LAT_F64_BITS
        or target.get("center_lon_f64_bits") != GM1_F02_ROI_CENTER_LON_F64_BITS
        or target.get("nx") != 300
        or target.get("ny") != 300
        or target.get("nz") != 75
        or target.get("dx_m_f64_bits") != "4059000000000000"
        or target.get("dy_m_f64_bits") != "4059000000000000"
        or target.get("dz_m_f64_bits") != "4069000000000000"
    ):
        raise ValueError("f02 ROI target differs from the sealed storm-centered grid")
    decoded = _object(receipt.get("decoded_source"), "decoded source")
    orientation = _object(
        decoded.get("orientation_scan_audit"), "orientation scan audit"
    )
    if orientation.get("result") != "PASS_NO_TRANSPOSE_OR_SCAN_DISCONTINUITY":
        raise ValueError("f02 ROI lacks the Rust orientation/scan proof")
    extrema = orientation.get("source_reflectivity_extrema")
    if not isinstance(extrema, list) or len(extrema) != 2:
        raise ValueError("f02 ROI source-extrema inventory differs from v1")
    source_f02 = _object(extrema[0], "f02 source extrema")
    source_low = _object(
        _object(source_f02.get("refd_1000m_agl_dbz"), "source low reflectivity").get(
            "maximum"
        ),
        "source low reflectivity maximum",
    )
    if (
        source_f02.get("forecast_hour") != 2
        or source_low.get("source_full_i") != GM1_F02_ROI_SOURCE_FULL_I
        or source_low.get("source_full_j") != GM1_F02_ROI_SOURCE_FULL_J
    ):
        raise ValueError("f02 ROI source storm anchor differs from the audited HRRR cell")
    frames = receipt.get("frames")
    if not isinstance(frames, list) or len(frames) != 2:
        raise ValueError("f02 ROI frame inventory differs from v1")
    f02 = _object(frames[0], "f02 frame")
    mapped_low = _object(
        _object(
            _object(f02.get("diagnostics"), "f02 diagnostics").get(
                "refd_1000m_agl_dbz"
            ),
            "mapped low reflectivity",
        ).get("maximum"),
        "mapped low reflectivity maximum",
    )
    edge_distance = mapped_low.get("distance_to_nearest_edge_cells")
    if (
        f02.get("forecast_hour") != 2
        or mapped_low.get("i") != GM1_F02_ROI_TARGET_I
        or mapped_low.get("j") != GM1_F02_ROI_TARGET_J
        or type(edge_distance) is not int
        or edge_distance < GM1_F02_ROI_MINIMUM_EDGE_DISTANCE
    ):
        raise ValueError("f02 storm is not admitted in the mapped parent interior")
    return Gm1HrrrF02RoiBinding(
        series=series,
        center_latitude=_f64_bits(
            target["center_lat_f64_bits"], "f02 ROI center latitude"
        ),
        center_longitude=_f64_bits(
            target["center_lon_f64_bits"], "f02 ROI center longitude"
        ),
        source_full_i=GM1_F02_ROI_SOURCE_FULL_I,
        source_full_j=GM1_F02_ROI_SOURCE_FULL_J,
        mapped_i=GM1_F02_ROI_TARGET_I,
        mapped_j=GM1_F02_ROI_TARGET_J,
        mapped_edge_distance_points=int(edge_distance),
        mapped_reflectivity_dbz=_f32_bits(
            mapped_low.get("dbz_f32_bits"), "mapped low reflectivity maximum"
        ),
    )


def load_gm1_hrrr_f02_roi_series(
    directory: str | Path,
    *,
    expected_receipt_sha256: str,
    expected_f01_artifact_sha256: str,
    expected_f02_artifact_sha256: str,
    expected_f03_artifact_sha256: str,
    expected_contract_sha256: str,
) -> Gm1HrrrF02RoiBinding:
    series = load_gm1_hrrr_future_boundary_series(
        directory,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_f01_artifact_sha256=expected_f01_artifact_sha256,
        expected_f02_artifact_sha256=expected_f02_artifact_sha256,
        expected_f03_artifact_sha256=expected_f03_artifact_sha256,
        expected_contract_sha256=expected_contract_sha256,
    )
    return validate_gm1_hrrr_f02_roi_series(series)


__all__ = [
    "GM1_F02_ROI_CENTER_LAT_F64_BITS",
    "GM1_F02_ROI_CENTER_LON_F64_BITS",
    "GM1_F02_ROI_MINIMUM_EDGE_DISTANCE",
    "GM1_F02_ROI_SOURCE_FULL_I",
    "GM1_F02_ROI_SOURCE_FULL_J",
    "GM1_F02_ROI_TARGET_I",
    "GM1_F02_ROI_TARGET_J",
    "Gm1HrrrF02RoiBinding",
    "load_gm1_hrrr_f02_roi_series",
    "validate_gm1_hrrr_f02_roi_series",
]
