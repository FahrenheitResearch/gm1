"""CUDA-cold admission of the real fixed-50 m GM1 child static geometry."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from cm1core.cuda_gm1_child_terrain import (
    LoadedGm1ChildTerrain,
    load_gm1_child_terrain_export,
)
from cm1core.gm1_child_initialization import (
    Gm1ChildInitializationPlan,
    Gm1ChildProducerRequirements,
)
from cm1core.gm1_child_producer_interfaces import (
    GM1_CHILD_TERRAIN_INPUT_SHA256,
    Gm1ChildProducerInterfaces,
)


GM1_CHILD_STATIC_ADMISSION_SCHEMA = "gm1.fixed-50m-child-static-admission/v1"
GM1_CHILD_STATIC_VALIDATION_SCHEMA = (
    "gm1.cuda-source-coordinate-50m-child-terrain-metrics-validation/v1"
)
GM1_CHILD_STATIC_VALIDATION_SHA256 = (
    "f90fcadbe9f75eacc66a96d56dc9584166b2e50641a80e311f358c4bb49a27a8"
)
GM1_CHILD_STATIC_COMBINED_SHA256 = (
    "2ec8b6a5a48a81017527e66c4fa6b264f1476b737fa2b0a02f7795012877c0a9"
)
GM1_CHILD_STATIC_HEIGHT_MANIFEST_SHA256 = (
    "5459be4acf6ba3f8f83e6ff59bdfec11946d2c4846f60556bf02a1892289b9d6"
)
GM1_CHILD_STATIC_ZH_SHA256 = (
    "7fe66bd1639fb0cb4ad171be87b77484703a4d92ce868524e796e888a52a5159"
)
GM1_CHILD_STATIC_ZF_SHA256 = (
    "df5aef17965915406fe334600afe2f8f2e1ebb5d07665f71820eaac9bde4128f"
)
GM1_CHILD_STATIC_METRICS_PNG_SHA256 = (
    "8923f1d4e3b8d801fd0b9bb7b035b4bf10e692ea52302ca755878995d271f6e2"
)
GM1_CHILD_STATIC_COLUMNS_PNG_SHA256 = (
    "c0121c7a70a21479b46933690b4fce1fa3e2e8b4c0786323d6198dd150df7273"
)
GM1_CHILD_STATIC_KERNEL_SHA256 = (
    "905bec56020e2bf1a73c0c854a9d5f462ccc372334ab3ceee2ad50f041be2f18"
)
GM1_CHILD_STATIC_ADMISSION_CLAIM_LIMIT = (
    "strict installed-byte admission of one real source-coordinate terrain "
    "carrier, the passed 23-carrier 300x300x75 CM1-r21.1-derived static CUDA "
    "metric result, and its zh/zf export only; no atmospheric child state, "
    "hydrostatic rebalance, boundary initialization, forecast, storm, or "
    "tornado claim"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FIELD_SHAPES = MappingProxyType(
    {
        "terrain_msl": (300, 300),
        "zh": (75, 300, 300),
        "zf": (76, 300, 300),
        "mh": (75, 300, 300),
        "rmh": (75, 300, 300),
        "mf": (76, 300, 300),
        "rmf": (76, 300, 300),
        "c1": (76, 300, 300),
        "c2": (76, 300, 300),
        "gz": (300, 300),
        "rgz": (300, 300),
        "gzu": (300, 301),
        "rgzu": (300, 301),
        "gzv": (301, 300),
        "rgzv": (301, 300),
        "dzdx": (300, 300),
        "dzdy": (300, 300),
        "gx": (76, 300, 300),
        "gxu": (76, 300, 301),
        "gy": (76, 300, 300),
        "gyv": (76, 301, 300),
        "rds": (77,),
        "rdsf": (78,),
    }
)


@dataclass(frozen=True, slots=True)
class Gm1ChildStaticAdmission:
    schema: str
    validation_path: Path
    validation_sha256: str
    terrain: LoadedGm1ChildTerrain
    static_height_directory: Path
    static_height_manifest_path: Path
    zh_path: Path
    zf_path: Path
    png_paths: Mapping[str, Path]
    carrier_names: tuple[str, ...]
    receipt: Mapping[str, Any]
    atmospheric_child_ready: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{label} keys changed: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"static validation duplicates JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON token {token}")


def _read_json(path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    expected_sha256 = _sha(expected_sha256, "static validation expected SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ValueError("static validation must be a regular non-symlink file")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"static validation SHA-256 is {actual}; expected {expected_sha256}"
        )
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("static validation is not UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("static validation root must be an object")
    return value


def _rooted_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path must be a nonempty string")
    requested = root / relative
    if requested.is_symlink():
        raise ValueError(f"{label} must be a regular file within the workspace")
    candidate = requested.resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"{label} must be a regular file within the workspace")
    return candidate


def _validate_field_inventory(value: object) -> tuple[str, ...]:
    fields = _exact(value, set(_FIELD_SHAPES), "static metric field inventory")
    for name, shape in _FIELD_SHAPES.items():
        field = _exact(
            fields[name],
            {"bytes", "finite", "max", "min", "sha256", "shape", "words"},
            f"static metric field {name}",
        )
        words = math.prod(shape)
        if (
            field["shape"] != list(shape)
            or field["words"] != words
            or field["bytes"] != words * 4
            or field["finite"] is not True
        ):
            raise ValueError(f"static metric field {name} shape/finiteness changed")
        _sha(field["sha256"], f"static metric field {name} SHA-256")
        if not isinstance(field["min"], (int, float)) or not isinstance(
            field["max"], (int, float)
        ):
            raise ValueError(f"static metric field {name} extrema changed type")
        if not math.isfinite(float(field["min"])) or not math.isfinite(
            float(field["max"])
        ):
            raise ValueError(f"static metric field {name} extrema are nonfinite")
    return tuple(_FIELD_SHAPES)


def load_gm1_child_static_admission(
    plan: Gm1ChildInitializationPlan,
    requirements: Gm1ChildProducerRequirements,
    interfaces: Gm1ChildProducerInterfaces,
    *,
    workspace_root: str | Path,
    terrain_directory: str | Path,
    validation_receipt: str | Path,
    expected_validation_sha256: str = GM1_CHILD_STATIC_VALIDATION_SHA256,
) -> Gm1ChildStaticAdmission:
    """Bind the exact real terrain and passed CUDA static metrics, CUDA-cold."""

    if not isinstance(plan, Gm1ChildInitializationPlan):
        raise TypeError("plan must be a Gm1ChildInitializationPlan")
    if not isinstance(requirements, Gm1ChildProducerRequirements):
        raise TypeError("requirements must be a Gm1ChildProducerRequirements")
    if not isinstance(interfaces, Gm1ChildProducerInterfaces):
        raise TypeError("interfaces must be a Gm1ChildProducerInterfaces")
    if (
        requirements.parent_plan_sha256 != plan.plan_sha256
        or interfaces.parent_plan_sha256 != plan.plan_sha256
        or interfaces.requirements_sha256 != requirements.sha256
    ):
        raise ValueError("static admission inputs do not bind the same child plan")

    root_requested = Path(workspace_root)
    if root_requested.is_symlink():
        raise ValueError("workspace root must not be a symlink")
    root = root_requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace root must be a directory")
    pins = dict(GM1_CHILD_TERRAIN_INPUT_SHA256)
    terrain = load_gm1_child_terrain_export(
        terrain_directory,
        expected_manifest_sha256=pins["manifest_sha256"],
        expected_payload_sha256=pins["payload_sha256"],
        expected_receipt_sha256=pins["receipt_sha256"],
        expected_producer_source_sha256=pins["producer_source_sha256"],
        expected_producer_executable_sha256=pins["producer_executable_sha256"],
    )
    validation_requested = Path(validation_receipt)
    if validation_requested.is_symlink():
        raise ValueError("static validation must not be a symlink")
    validation_path = validation_requested.resolve(strict=True)
    receipt = _read_json(
        validation_path, expected_sha256=expected_validation_sha256
    )
    receipt = _exact(
        receipt,
        {
            "atmospheric_child_ready",
            "claim_boundary",
            "claim_limit",
            "cuda_static_metrics",
            "fixture_class",
            "gpu_mutex",
            "next_required_producers",
            "pngs",
            "schema",
            "source_terrain",
            "static_height_export",
            "status",
            "time_utc",
        },
        "static validation",
    )
    if (
        receipt["schema"] != GM1_CHILD_STATIC_VALIDATION_SCHEMA
        or receipt["status"]
        != "PASS_REAL_SOURCE_TERRAIN_AND_CUDA_STATIC_METRICS_300X300X75"
        or receipt["atmospheric_child_ready"] is not False
        or receipt["fixture_class"]
        != "real source-coordinate f02 terrain; no synthetic data"
    ):
        raise ValueError("static validation status/source classification changed")

    source = _exact(
        receipt["source_terrain"],
        {
            "abi",
            "directory",
            "manifest_sha256",
            "oracle",
            "payload_sha256",
            "producer_executable_sha256",
            "producer_source_sha256",
            "receipt_sha256",
            "schema",
        },
        "static validation source terrain",
    )
    if (
        source["schema"] != "gm1.source-coordinate-50m-child-terrain/v1"
        or source["abi"] != "gm1.source-coordinate-50m-child-terrain-f32/v1"
        or source["manifest_sha256"] != pins["manifest_sha256"]
        or source["payload_sha256"] != pins["payload_sha256"]
        or source["receipt_sha256"] != pins["receipt_sha256"]
        or source["producer_source_sha256"] != pins["producer_source_sha256"]
        or source["producer_executable_sha256"]
        != pins["producer_executable_sha256"]
        or source["directory"] != "work/gm1-t300-child-terrain-v1"
        or (root / source["directory"]).resolve(strict=True) != terrain.directory
    ):
        raise ValueError("static validation binds another terrain publication")
    oracle = _exact(
        source["oracle"],
        {
            "actual_sha256",
            "expected_sha256",
            "fixture_class",
            "max_ulp",
            "mismatch_count",
            "mode",
            "null_or_missing_is_pass",
            "passed",
            "record_count",
            "schema",
            "shape_yx",
            "transform_contract",
        },
        "real terrain oracle",
    )
    if (
        oracle["passed"] is not True
        or oracle["null_or_missing_is_pass"] is not False
        or oracle["mode"] != "raw-bit"
        or oracle["mismatch_count"] != 0
        or oracle["max_ulp"] != 0
        or oracle["record_count"] != 90000
        or oracle["shape_yx"] != [300, 300]
        or oracle["actual_sha256"] != pins["payload_sha256"]
        or oracle["expected_sha256"] != pins["payload_sha256"]
    ):
        raise ValueError("real terrain raw-bit oracle did not pass exactly")

    cuda = _exact(
        receipt["cuda_static_metrics"],
        {
            "combined_output_sha256",
            "elapsed_ms_event_timing",
            "fields",
            "invariants",
            "kernel",
            "launch_receipt",
            "shape_zyx",
            "source_order_top_oracle",
            "timing_claim",
        },
        "static CUDA metrics",
    )
    if (
        cuda["shape_zyx"] != [75, 300, 300]
        or cuda["combined_output_sha256"] != GM1_CHILD_STATIC_COMBINED_SHA256
    ):
        raise ValueError("static CUDA shape or combined output identity changed")
    carrier_names = _validate_field_inventory(cuda["fields"])
    invariants = _exact(
        cuda["invariants"],
        {
            "all_fields_finite",
            "gx_top_all_signed_or_positive_zero",
            "gy_top_all_signed_or_positive_zero",
            "minimum_physical_layer_m",
            "positive_vertical_and_terrain_metrics",
            "strictly_increasing_physical_faces",
            "surface_raw_bit_equal",
            "top_max_abs_error_m",
            "top_source_order_negative_control_detected",
            "top_source_order_raw_bit_equal",
            "top_within_derived_real_binary32_bound",
        },
        "static CUDA invariants",
    )
    required_true = set(invariants) - {
        "minimum_physical_layer_m",
        "top_max_abs_error_m",
    }
    if any(invariants[name] is not True for name in required_true):
        raise ValueError("one or more static CUDA invariants did not pass")
    if not float(invariants["minimum_physical_layer_m"]) > 0.0:
        raise ValueError("static CUDA physical layer is nonpositive")
    kernel = _exact(
        cuda["kernel"],
        {
            "cubin_sha256",
            "ptx_sha256",
            "runtime_identity",
            "source_sha256",
            "target_arch",
        },
        "static CUDA kernel",
    )
    if (
        kernel["source_sha256"] != GM1_CHILD_STATIC_KERNEL_SHA256
        or kernel["target_arch"] != "86"
    ):
        raise ValueError("static CUDA kernel source/target identity changed")

    launch = _exact(
        cuda["launch_receipt"],
        {
            "claim_limit",
            "fixture_class",
            "metrics",
            "next_required_producers",
            "schema",
            "source",
            "status",
        },
        "static CUDA launch receipt",
    )
    launch_source = _exact(
        launch["source"],
        {
            "abi",
            "manifest_sha256",
            "parent_checkpoint_sha256",
            "parent_plan_sha256",
            "payload_sha256",
            "receipt_sha256",
            "schema",
            "synthetic_translation",
        },
        "static CUDA launch source",
    )
    if (
        launch["schema"]
        != "gm1.cuda-source-coordinate-50m-child-terrain-metrics/v1"
        or launch["status"]
        != "PASS_REAL_SOURCE_TERRAIN_LAUNCHED_TO_EXACT_STATIC_METRICS"
        or launch_source["parent_plan_sha256"] != plan.plan_sha256
        or launch_source["payload_sha256"] != pins["payload_sha256"]
        or launch_source["manifest_sha256"] != pins["manifest_sha256"]
        or launch_source["receipt_sha256"] != pins["receipt_sha256"]
        or launch_source["synthetic_translation"] is not False
    ):
        raise ValueError("static CUDA launch receipt binds another source/plan")

    mutex = _exact(
        receipt["gpu_mutex"],
        {"clean_release", "owner_token_sha256", "path"},
        "static CUDA mutex",
    )
    if mutex["clean_release"] is not True:
        raise ValueError("static CUDA validation did not cleanly release the mutex")
    _sha(mutex["owner_token_sha256"], "static CUDA mutex owner token")

    heights = _exact(
        receipt["static_height_export"],
        {"abi", "directory", "manifest_sha256", "payloads", "schema"},
        "static height export",
    )
    if (
        heights["schema"]
        != "gm1.cuda-source-coordinate-50m-child-static-heights-manifest/v1"
        or heights["abi"]
        != "gm1.cuda-source-coordinate-50m-child-static-heights-f32/v1"
        or heights["manifest_sha256"]
        != GM1_CHILD_STATIC_HEIGHT_MANIFEST_SHA256
    ):
        raise ValueError("static height export identity changed")
    height_requested = root / heights["directory"]
    if height_requested.is_symlink():
        raise ValueError("static height export directory must not be a symlink")
    height_directory = height_requested.resolve(strict=True)
    if not height_directory.is_relative_to(root) or not height_directory.is_dir():
        raise ValueError("static height export directory escapes the workspace")
    height_manifest = _rooted_file(
        root, f"{heights['directory']}/manifest.json", "static height manifest"
    )
    if _sha256_file(height_manifest) != GM1_CHILD_STATIC_HEIGHT_MANIFEST_SHA256:
        raise ValueError("static height manifest bytes changed")
    payloads = _exact(heights["payloads"], {"zf", "zh"}, "static height payloads")
    height_paths: dict[str, Path] = {}
    for name, expected_shape, expected_sha in (
        ("zh", [75, 300, 300], GM1_CHILD_STATIC_ZH_SHA256),
        ("zf", [76, 300, 300], GM1_CHILD_STATIC_ZF_SHA256),
    ):
        payload = _exact(
            payloads[name],
            {"dtype", "file", "nbytes", "order", "sha256", "shape_kji", "words"},
            f"static height {name}",
        )
        words = math.prod(expected_shape)
        if payload != {
            "dtype": "little-endian-f32",
            "file": f"{name}.f32le",
            "nbytes": words * 4,
            "order": "C[k,j,i]",
            "sha256": expected_sha,
            "shape_kji": expected_shape,
            "words": words,
        }:
            raise ValueError(f"static height {name} metadata changed")
        path = _rooted_file(
            root, f"{heights['directory']}/{payload['file']}", f"static height {name}"
        )
        if _sha256_file(path) != expected_sha:
            raise ValueError(f"static height {name} bytes changed")
        height_paths[name] = path

    pngs = _exact(
        receipt["pngs"],
        {"gm1-real-child-terrain-columns.png", "gm1-real-child-terrain-metrics.png"},
        "static validation PNGs",
    )
    png_paths: dict[str, Path] = {}
    for name, expected_sha in (
        ("gm1-real-child-terrain-columns.png", GM1_CHILD_STATIC_COLUMNS_PNG_SHA256),
        ("gm1-real-child-terrain-metrics.png", GM1_CHILD_STATIC_METRICS_PNG_SHA256),
    ):
        metadata = _exact(
            pngs[name], {"bytes", "classification", "path", "sha256"}, f"PNG {name}"
        )
        if metadata["sha256"] != expected_sha or "not an atmospheric forecast" not in metadata[
            "classification"
        ]:
            raise ValueError(f"PNG {name} identity/classification changed")
        path = _rooted_file(root, metadata["path"], f"PNG {name}")
        if path.stat().st_size != metadata["bytes"] or _sha256_file(path) != expected_sha:
            raise ValueError(f"PNG {name} bytes changed")
        png_paths[name] = path

    return Gm1ChildStaticAdmission(
        schema=GM1_CHILD_STATIC_ADMISSION_SCHEMA,
        validation_path=validation_path,
        validation_sha256=expected_validation_sha256,
        terrain=terrain,
        static_height_directory=height_directory,
        static_height_manifest_path=height_manifest,
        zh_path=height_paths["zh"],
        zf_path=height_paths["zf"],
        png_paths=MappingProxyType(png_paths),
        carrier_names=carrier_names,
        receipt=MappingProxyType(dict(receipt)),
    )


__all__ = (
    "GM1_CHILD_STATIC_ADMISSION_CLAIM_LIMIT",
    "GM1_CHILD_STATIC_ADMISSION_SCHEMA",
    "GM1_CHILD_STATIC_COMBINED_SHA256",
    "GM1_CHILD_STATIC_HEIGHT_MANIFEST_SHA256",
    "GM1_CHILD_STATIC_VALIDATION_SCHEMA",
    "GM1_CHILD_STATIC_VALIDATION_SHA256",
    "GM1_CHILD_STATIC_ZF_SHA256",
    "GM1_CHILD_STATIC_ZH_SHA256",
    "Gm1ChildStaticAdmission",
    "load_gm1_child_static_admission",
)
