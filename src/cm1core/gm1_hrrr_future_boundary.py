"""Strict real-HRRR future-boundary admission for a GM1 f01 forecast.

Rust owns source decoding, CM1-grid mapping, and CM1FHS1 publication.  This
module independently pins the published series receipt, preserves its
absolute source clock, validates the deterministic carrier export, and binds
the f01-to-f02 interval to the GM1 CUDA stage callback.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat
import struct
from dataclasses import dataclass
from typing import Any, Mapping

from cm1core.cuda_gm1_forecast import (
    GM1_HRRR_DAVIES_LATERAL_FORCING_SEMANTIC_ID,
    GM1_HRRR_F01_F02_SOURCE_CLOCK,
    Gm1ForecastLateralForcing,
)
from cm1core.cuda_gm1_hrrr_lbc import (
    HrrrLbcFrame,
    create_cuda_gm1_hrrr_lbc_state,
    launch_cuda_gm1_hrrr_lbc_stage,
    load_hrrr_lbc_frame_export,
)


GM1_HRRR_FUTURE_BOUNDARY_SERIES_SCHEMA = (
    "gm1.hrrr-future-boundary-series/v1"
)
GM1_HRRR_FUTURE_BOUNDARY_STATUS = "PASS_REAL_HRRR_F02_F03_RUST_MAPPED"
GM1_FORCING_HISTORY_EXPORT_RECEIPT_SCHEMA = (
    "gm1.forcing-history-export-receipt/v1"
)
GM1_FORCING_HISTORY_EXPORT_ABI = "gm1.forcing-history-export/v1"
GM1_F01_SOURCE_OFFSET_S = 3600.0
GM1_F02_SOURCE_OFFSET_S = 7200.0
GM1_F03_SOURCE_OFFSET_S = 10800.0
GM1_MODEL_DT_S = 0.5
GM1_F01_STEP = 7200
GM1_F02_STEP = 14400
GM1_F03_STEP = 21600
GM1_FUTURE_BOUNDARY_FILES = frozenset(
    (
        "boundary-f02.cm1fhs",
        "boundary-f03.cm1fhs",
        "future-boundary-series-receipt.json",
    )
)


@dataclass(frozen=True, slots=True)
class Gm1HrrrFutureBoundaryFrame:
    forecast_hour: int
    path: Path
    artifact_sha256: str
    contract_sha256: str
    payload_sha256: str
    step: int
    absolute_source_start_s: float
    absolute_source_end_s: float
    relative_model_start_s: float
    relative_model_end_s: float


@dataclass(frozen=True, slots=True)
class Gm1HrrrFutureBoundarySeries:
    directory: Path
    receipt_path: Path
    receipt_sha256: str
    contract_sha256: str
    baseline_f01_artifact_sha256: str
    execution_identity_sha256: str
    f02: Gm1HrrrFutureBoundaryFrame
    f03: Gm1HrrrFutureBoundaryFrame
    receipt: Mapping[str, Any]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _require_exact_keys(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} key inventory differs from its sealed schema")
    return value


def _decode_f64_bits(value: object, label: str) -> float:
    if (
        not isinstance(value, str)
        or len(value) != 16
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 16 lowercase hexadecimal digits")
    decoded = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(decoded):
        raise ValueError(f"{label} must encode a finite float64")
    return decoded


def _require_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")


def _validate_target(value: object) -> Mapping[str, Any]:
    target = _require_exact_keys(
        value,
        {
            "center_lat_f64_bits",
            "center_lon_f64_bits",
            "dx_m_f64_bits",
            "dy_m_f64_bits",
            "dz_m_f64_bits",
            "nx",
            "ny",
            "nz",
        },
        "future-boundary target",
    )
    if (target["nx"], target["ny"], target["nz"]) != (300, 300, 75):
        raise ValueError("future-boundary target is not the admitted 300x300x75 grid")
    for name, expected in (
        ("dx_m_f64_bits", 100.0),
        ("dy_m_f64_bits", 100.0),
        ("dz_m_f64_bits", 200.0),
    ):
        if _decode_f64_bits(target[name], name) != expected:
            raise ValueError("future-boundary target spacing differs from 100x100x200 m")
    _decode_f64_bits(target["center_lat_f64_bits"], "target center latitude")
    _decode_f64_bits(target["center_lon_f64_bits"], "target center longitude")
    return target


def _parse_frame(
    root: Path,
    value: object,
    *,
    hour: int,
    expected_artifact_sha256: str,
    expected_contract_sha256: str,
    expected_target: Mapping[str, Any],
) -> Gm1HrrrFutureBoundaryFrame:
    frame = _require_exact_keys(
        value,
        {
            "artifact",
            "clock_interval",
            "diagnostics",
            "forecast_hour",
            "output_name",
            "source_fingerprint_sha256",
            "stats",
            "target",
        },
        f"f{hour:02d} series frame",
    )
    expected_name = f"boundary-f{hour:02d}.cm1fhs"
    if frame["forecast_hour"] != hour or frame["output_name"] != expected_name:
        raise ValueError(f"future-boundary frame is not exact f{hour:02d}")
    if frame["target"] != expected_target:
        raise ValueError("future-boundary frame target differs from the series target")
    _require_sha256(frame["source_fingerprint_sha256"], "source fingerprint")
    artifact = _require_exact_keys(
        frame["artifact"],
        {
            "artifact_nbytes",
            "artifact_sha256",
            "contract_sha256",
            "dt_f64_bits",
            "header_sha256",
            "model_time_f64_bits",
            "payload_nbytes",
            "payload_sha256",
            "step",
        },
        f"f{hour:02d} artifact receipt",
    )
    expected_end = float(hour * 3600)
    expected_start = expected_end - 3600.0
    expected_relative_start = expected_start - GM1_F01_SOURCE_OFFSET_S
    expected_relative_end = expected_end - GM1_F01_SOURCE_OFFSET_S
    expected_step = hour * 7200
    if (
        artifact["artifact_sha256"] != expected_artifact_sha256
        or artifact["contract_sha256"] != expected_contract_sha256
        or artifact["step"] != expected_step
        or _decode_f64_bits(artifact["dt_f64_bits"], "artifact dt")
        != GM1_MODEL_DT_S
        or _decode_f64_bits(artifact["model_time_f64_bits"], "artifact time")
        != expected_end
    ):
        raise ValueError(f"f{hour:02d} artifact identity or absolute clock differs")
    payload_sha256 = _require_sha256(
        artifact["payload_sha256"], f"f{hour:02d} payload hash"
    )
    _require_sha256(artifact["header_sha256"], f"f{hour:02d} header hash")
    interval = _require_exact_keys(
        frame["clock_interval"],
        {
            "absolute_source_end_s_f64_bits",
            "absolute_source_start_s_f64_bits",
            "relative_model_end_s_f64_bits",
            "relative_model_start_s_f64_bits",
        },
        f"f{hour:02d} clock interval",
    )
    decoded_interval = (
        _decode_f64_bits(
            interval["absolute_source_start_s_f64_bits"], "absolute source start"
        ),
        _decode_f64_bits(
            interval["absolute_source_end_s_f64_bits"], "absolute source end"
        ),
        _decode_f64_bits(
            interval["relative_model_start_s_f64_bits"], "relative model start"
        ),
        _decode_f64_bits(
            interval["relative_model_end_s_f64_bits"], "relative model end"
        ),
    )
    expected_interval = (
        expected_start,
        expected_end,
        expected_relative_start,
        expected_relative_end,
    )
    if decoded_interval != expected_interval:
        raise ValueError(f"f{hour:02d} absolute/relative source interval differs")
    path = root / expected_name
    _require_regular_file(path, f"f{hour:02d} boundary artifact")
    artifact_nbytes = artifact["artifact_nbytes"]
    if (
        isinstance(artifact_nbytes, bool)
        or not isinstance(artifact_nbytes, int)
        or artifact_nbytes != path.stat().st_size
        or _sha256_file(path) != expected_artifact_sha256
    ):
        raise ValueError(f"f{hour:02d} boundary artifact differs from its receipt")
    return Gm1HrrrFutureBoundaryFrame(
        hour,
        path,
        expected_artifact_sha256,
        expected_contract_sha256,
        payload_sha256,
        expected_step,
        expected_start,
        expected_end,
        expected_relative_start,
        expected_relative_end,
    )


def load_gm1_hrrr_future_boundary_series(
    directory: str | Path,
    *,
    expected_receipt_sha256: str,
    expected_f01_artifact_sha256: str,
    expected_f02_artifact_sha256: str,
    expected_f03_artifact_sha256: str,
    expected_contract_sha256: str,
) -> Gm1HrrrFutureBoundarySeries:
    """Load one exact Rust-published f02/f03 series without parsing CM1FHS1."""

    pins = tuple(
        _require_sha256(value, label)
        for value, label in (
            (expected_receipt_sha256, "future-boundary receipt hash"),
            (expected_f01_artifact_sha256, "f01 artifact hash"),
            (expected_f02_artifact_sha256, "f02 artifact hash"),
            (expected_f03_artifact_sha256, "f03 artifact hash"),
            (expected_contract_sha256, "future-boundary contract hash"),
        )
    )
    receipt_pin, f01_pin, f02_pin, f03_pin, contract_pin = pins
    root = Path(directory).absolute()
    try:
        metadata = root.lstat()
    except OSError as error:
        raise ValueError(f"cannot inspect future-boundary directory: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("future-boundary root must be a real directory")
    if {entry.name for entry in root.iterdir()} != GM1_FUTURE_BOUNDARY_FILES:
        raise ValueError("future-boundary file inventory differs from the sealed series")
    receipt_path = root / "future-boundary-series-receipt.json"
    _require_regular_file(receipt_path, "future-boundary receipt")
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt_pin:
        raise ValueError("future-boundary receipt differs from its independent pin")
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("future-boundary receipt is not strict UTF-8 JSON") from error
    if raw != _canonical_json(receipt):
        raise ValueError("future-boundary receipt is not canonical JSON")
    top = _require_exact_keys(
        receipt,
        {
            "authority",
            "baseline_f01",
            "claim_limit",
            "config_sha256",
            "contract_sha256",
            "decoded_source",
            "execution_identity_sha256",
            "frames",
            "model_clock",
            "model",
            "operator_contracts",
            "presentation_role",
            "raw_source",
            "schema",
            "status",
            "storm_source",
            "target",
        },
        "future-boundary receipt",
    )
    if (
        top["schema"] != GM1_HRRR_FUTURE_BOUNDARY_SERIES_SCHEMA
        or top["status"] != GM1_HRRR_FUTURE_BOUNDARY_STATUS
        or top["model"] != "GM1"
        or top["contract_sha256"] != contract_pin
        or top["presentation_role"]
        != (
            "source-admission evidence; native HRRR resolution/block geometry; "
            "not the user-facing evolved 100 m GM1 storm field"
        )
    ):
        raise ValueError("future-boundary series identity differs from v1")
    execution = _require_sha256(
        top["execution_identity_sha256"], "future-boundary execution identity"
    )
    _require_sha256(top["config_sha256"], "future-boundary configuration hash")
    baseline = _require_exact_keys(
        top["baseline_f01"],
        {"artifact", "receipt", "source_forecast_hour", "source_offset_s_f64_bits"},
        "baseline f01",
    )
    baseline_file = _require_exact_keys(
        baseline["artifact"], {"name", "nbytes", "sha256"}, "baseline f01 artifact"
    )
    baseline_receipt = baseline["receipt"]
    if (
        baseline_file["name"] != "baseline-f01.cm1fhs"
        or baseline_file["sha256"] != f01_pin
        or baseline["source_forecast_hour"] != 1
        or _decode_f64_bits(baseline["source_offset_s_f64_bits"], "f01 source offset")
        != GM1_F01_SOURCE_OFFSET_S
        or not isinstance(baseline_receipt, dict)
        or baseline_receipt.get("artifact_sha256") != f01_pin
        or baseline_receipt.get("contract_sha256") != contract_pin
        or baseline_receipt.get("step") != GM1_F01_STEP
        or _decode_f64_bits(baseline_receipt.get("model_time_f64_bits"), "f01 time")
        != GM1_F01_SOURCE_OFFSET_S
    ):
        raise ValueError("baseline f01 identity or absolute source clock differs")
    model_clock = _require_exact_keys(
        top["model_clock"],
        {
            "dt_s_f64_bits",
            "initialization_source_offset_s_f64_bits",
            "relative_model_origin_s_f64_bits",
            "semantics",
        },
        "future-boundary model clock",
    )
    if (
        _decode_f64_bits(model_clock["dt_s_f64_bits"], "model dt") != GM1_MODEL_DT_S
        or _decode_f64_bits(
            model_clock["initialization_source_offset_s_f64_bits"],
            "initialization source offset",
        )
        != GM1_F01_SOURCE_OFFSET_S
        or _decode_f64_bits(
            model_clock["relative_model_origin_s_f64_bits"], "relative model origin"
        )
        != 0.0
        or "explicit clock point" not in str(model_clock["semantics"])
    ):
        raise ValueError("future-boundary model clock is not f01-relative CM1 staging")
    target = _validate_target(top["target"])
    frames = top["frames"]
    if not isinstance(frames, list) or len(frames) != 2:
        raise ValueError("future-boundary series must contain exact f02/f03 frames")
    f02 = _parse_frame(
        root,
        frames[0],
        hour=2,
        expected_artifact_sha256=f02_pin,
        expected_contract_sha256=contract_pin,
        expected_target=target,
    )
    f03 = _parse_frame(
        root,
        frames[1],
        hour=3,
        expected_artifact_sha256=f03_pin,
        expected_contract_sha256=contract_pin,
        expected_target=target,
    )
    return Gm1HrrrFutureBoundarySeries(
        root,
        receipt_path,
        receipt_pin,
        contract_pin,
        f01_pin,
        execution,
        f02,
        f03,
        receipt,
    )


def load_gm1_f02_forcing_export(
    series: Gm1HrrrFutureBoundarySeries,
    directory: str | Path,
    export_receipt: Mapping[str, Any],
) -> HrrrLbcFrame:
    """Bind one Rust carrier export to the sealed f02 series artifact."""

    receipt = _require_exact_keys(
        export_receipt,
        {
            "abi",
            "array_count",
            "contract_sha256",
            "dt_f64_bits",
            "manifest_nbytes",
            "manifest_sha256",
            "model",
            "model_time_f64_bits",
            "publication_contract",
            "raw_nbytes",
            "raw_sha256",
            "schema",
            "source_artifact_sha256",
            "step",
            "table_nbytes",
            "table_sha256",
        },
        "forcing-history export receipt",
    )
    if (
        receipt["abi"] != GM1_FORCING_HISTORY_EXPORT_ABI
        or receipt["schema"] != GM1_FORCING_HISTORY_EXPORT_RECEIPT_SCHEMA
        or receipt["model"] != "GM1"
        or receipt["array_count"] != 22
        or receipt["contract_sha256"] != series.contract_sha256
        or receipt["source_artifact_sha256"] != series.f02.artifact_sha256
        or receipt["step"] != GM1_F02_STEP
        or _decode_f64_bits(receipt["dt_f64_bits"], "forcing export dt")
        != GM1_MODEL_DT_S
        or _decode_f64_bits(receipt["model_time_f64_bits"], "forcing export time")
        != GM1_F02_SOURCE_OFFSET_S
    ):
        raise ValueError("forcing-history export is not the sealed f02 boundary")
    frame = load_hrrr_lbc_frame_export(
        directory,
        expected_manifest_sha256=_require_sha256(
            receipt["manifest_sha256"], "forcing manifest hash"
        ),
        expected_raw_sha256=_require_sha256(
            receipt["raw_sha256"], "forcing raw hash"
        ),
        expected_artifact_sha256=series.f02.artifact_sha256,
        expected_contract_sha256=series.contract_sha256,
        expected_model_time_s=GM1_F02_SOURCE_OFFSET_S,
        expected_step=GM1_F02_STEP,
    )
    if frame.payload_sha256 != series.f02.payload_sha256:
        raise ValueError("forcing export payload differs from the sealed f02 frame")
    return frame


def bind_gm1_f01_f02_lateral_forcing(
    forecast_state: Any,
    series: Gm1HrrrFutureBoundarySeries,
    frame: HrrrLbcFrame,
    *,
    width: int = 20,
    relaxation_timescale_s: float = 300.0,
) -> tuple[Any, Gm1ForecastLateralForcing]:
    """Create the device-resident f01-to-f02 Davies boundary adapter."""

    if (
        series.f02.absolute_source_start_s != GM1_F01_SOURCE_OFFSET_S
        or series.f02.absolute_source_end_s != GM1_F02_SOURCE_OFFSET_S
        or series.f02.relative_model_start_s != 0.0
        or series.f02.relative_model_end_s != 3600.0
        or frame.artifact_sha256 != series.f02.artifact_sha256
        or frame.contract_sha256 != series.contract_sha256
        or frame.model_time_s != GM1_F02_SOURCE_OFFSET_S
    ):
        raise ValueError("f01-to-f02 runtime binding differs from its sealed interval")
    lbc = create_cuda_gm1_hrrr_lbc_state(
        forecast_state,
        frame,
        width=width,
        initialization_source_offset_s=GM1_F01_SOURCE_OFFSET_S,
        source_start_offset_s=GM1_F01_SOURCE_OFFSET_S,
        source_end_offset_s=GM1_F02_SOURCE_OFFSET_S,
        relaxation_timescale_s=relaxation_timescale_s,
    )

    def apply_stage(state: Any, nrk: int, clock_point: Any) -> Mapping[str, Any]:
        return launch_cuda_gm1_hrrr_lbc_stage(
            lbc, state, nrk=nrk, clock_point=clock_point
        )

    forcing = Gm1ForecastLateralForcing(
        GM1_HRRR_F01_F02_SOURCE_CLOCK,
        apply_stage,
        relaxation_width_points=width,
        semantic_id=GM1_HRRR_DAVIES_LATERAL_FORCING_SEMANTIC_ID,
        relaxation_mode="davies-c2-stage-tendency",
        relaxation_timescale_s=relaxation_timescale_s,
    )
    return lbc, forcing


__all__ = [
    "GM1_F01_SOURCE_OFFSET_S",
    "GM1_F02_SOURCE_OFFSET_S",
    "GM1_F03_SOURCE_OFFSET_S",
    "GM1_FORCING_HISTORY_EXPORT_ABI",
    "GM1_FORCING_HISTORY_EXPORT_RECEIPT_SCHEMA",
    "GM1_HRRR_FUTURE_BOUNDARY_SERIES_SCHEMA",
    "Gm1HrrrFutureBoundaryFrame",
    "Gm1HrrrFutureBoundarySeries",
    "bind_gm1_f01_f02_lateral_forcing",
    "load_gm1_f02_forcing_export",
    "load_gm1_hrrr_future_boundary_series",
]
