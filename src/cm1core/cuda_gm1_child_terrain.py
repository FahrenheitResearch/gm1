"""Strict real-source terrain loader and CUDA metric launch for the GM1 child."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .cuda_stretch_z1_terrain import (
    CUDA_STRETCH_Z1_TERRAIN_ABI,
    CUDA_STRETCH_Z1_TERRAIN_SCHEMA,
    CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256,
    CudaStretchZ1TerrainLaunch,
    CompiledStretchZ1Terrain,
    launch_cuda_stretch_z1_terrain,
)


GM1_CHILD_TERRAIN_SCHEMA = "gm1.source-coordinate-50m-child-terrain/v1"
GM1_CHILD_TERRAIN_MANIFEST_SCHEMA = (
    "gm1.source-coordinate-50m-child-terrain-manifest/v1"
)
GM1_CHILD_TERRAIN_ABI = "gm1.source-coordinate-50m-child-terrain-f32/v1"
GM1_CHILD_TERRAIN_METRICS_SCHEMA = (
    "gm1.cuda-source-coordinate-50m-child-terrain-metrics/v1"
)
GM1_CHILD_TERRAIN_PAYLOAD = "terrain-msl.f32le"
GM1_CHILD_TERRAIN_STATUS = "PASS_REAL_SOURCE_COORDINATE_TERRAIN"
# The terrain, plan and parent checkpoint of the first 50 m child this project
# admitted.  They are lineage evidence, not the gate: a terrain remapped from a
# parent at another completed step is a different surface with different bytes
# and a different plan, and refusing it here would refuse the child this chain
# exists to build.  What is required instead is that the artifact is internally
# consistent - the receipt anchors the manifest that is present, the manifest
# describes the payload that is present - and that the parent it names is
# recordable, so the composing gate can require every producer of one child to
# name the same parent.
GM1_CHILD_TERRAIN_EXPECTED_SHA256 = (
    "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91"
)
GM1_CHILD_PLAN_SHA256 = (
    "14609494f7c87937fcb5ec0db3f214809ed514380560c43beb18183e6eebce17"
)
GM1_CHILD_PARENT_CHECKPOINT_SHA256 = (
    "4f3a00ae66e0b68f0f05f4eff54a3625e846c0efee31b3ecbdee01c3c87ac1b4"
)
GM1_CHILD_TERRAIN_CLAIM_LIMIT = (
    "one exact real source-coordinate f02 terrain_msl carrier and its selected "
    "CM1 r21.1 fixed-terrain stretch_z=1 static metrics for the 300x300x75, "
    "50 m, 15 km-top child; no atmospheric remap, hydrostatic/base-state "
    "repair, child open-boundary initialization, pressure integration, time "
    "evolution, forecast skill, storm, or tornado claim"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX8 = re.compile(r"[0-9a-f]{8}\Z")
_SOURCE = {
    "initialized_product_manifest_sha256": "b25de4008d17ddad46b8057f022b50a76e6bff61b537fa9952f68e638adf2567",
    "initialized_product_payload_sha256": "ddafeaa8ca523e81925cb4b53e0577963a3e57e06114b6b7df8c6d436e0d68b1",
    "initialized_product_receipt_sha256": "379b989884e773a942ade62069e99ce5e032374c2ae2ef13679f32a524fbbdcd",
    "initialized_product_restart_sha256": "9efe09771763a3f727af715f7dd41a2c6256cd5613aa2f31637cc08c9721ed6e",
    "source_artifact_sha256": "356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d",
    "source_contract_sha256": "a9ccd6f0c6650fd89cd42f16977209f57695464181bcb62beff14e9a9f61d896",
    "source_kind": "boundary-analysis-history-f02-roi",
    "terrain_carrier": {
        "name": "terrain_msl",
        "sha256": "80a2ce41cc02e7f247a1852ebbeba50a8058e71d9b2426e0b3312a416bc59a19",
        "shape_yx": [300, 300],
    },
}
_TRANSFORM = {
    "formula": "row0=f64(v00)*(1-fx)+f64(v10)*fx; row1=f64(v01)*(1-fx)+f64(v11)*fx; child=f32(row0*(1-fy)+row1*fy)",
    "fractional_coordinates": "parent_i=149.25+0.5*child_i; parent_j=149.25+0.5*child_j; fx and fy are exactly 0.25 or 0.75",
    "interpolation": "coordinate-aware-bilinear",
    "required_parent_i": "149..299 inclusive bilinear stencil",
    "required_parent_j": "149..299 inclusive bilinear stencil",
    "rounding": "binary64 weighted sums in the written statement order, then one round-to-nearest-ties-to-even binary32 store",
}


@dataclass(frozen=True, slots=True)
class LoadedGm1ChildTerrain:
    directory: Path
    manifest_path: Path
    payload_path: Path
    receipt_path: Path
    manifest_sha256: str
    payload_sha256: str
    receipt_sha256: str
    producer_source_sha256: str
    producer_executable_sha256: str
    terrain_msl: Any
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CudaGm1ChildTerrainMetricsLaunch:
    ready_event: Any
    terrain: LoadedGm1ChildTerrain
    metrics: CudaStretchZ1TerrainLaunch
    receipt: Mapping[str, Any]


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hex digits")
    return value


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(value) != keys:
        raise ValueError(
            f"{label} keys changed: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )
    return value


def _read_canonical(path: Path, label: str) -> tuple[bytes, Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    if not raw or b"\r" in raw:
        raise ValueError(f"{label} bytes are empty or non-canonical")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    canonical = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    if raw != canonical:
        raise ValueError(f"{label} is not canonical compact JSON")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return raw, value


def _validate_source(value: object) -> None:
    source = _exact(value, set(_SOURCE), "child terrain source")
    if source != _SOURCE:
        raise ValueError("child terrain source identities changed")


def _validate_transform(value: object) -> None:
    transform = _exact(value, set(_TRANSFORM), "child terrain transform")
    if transform != _TRANSFORM:
        raise ValueError("child terrain source-coordinate transform changed")


def load_gm1_child_terrain_export(
    directory: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
    expected_receipt_sha256: str,
    expected_producer_source_sha256: str | None = None,
    expected_producer_executable_sha256: str | None = None,
) -> LoadedGm1ChildTerrain:
    """Load one exact Rust child-terrain publication while CUDA stays cold.

    The manifest, payload and receipt digests are always named by the caller -
    the composer takes them from the full-state envelope it is building, so the
    terrain opened here is the terrain that envelope was composed over.

    The producer identity defaults to the one the artifact's own receipt
    records, and the manifest must agree with it.  Pinning it to one build
    would refuse a terrain republished by a corrected producer even though the
    envelope, the parent and every digest above already say it is the right
    surface.
    """

    expected_manifest_sha256 = _sha(
        expected_manifest_sha256, "expected child terrain manifest SHA-256"
    )
    expected_payload_sha256 = _sha(
        expected_payload_sha256, "expected child terrain payload SHA-256"
    )
    expected_receipt_sha256 = _sha(
        expected_receipt_sha256, "expected child terrain receipt SHA-256"
    )
    if expected_producer_source_sha256 is not None:
        expected_producer_source_sha256 = _sha(
            expected_producer_source_sha256,
            "expected child terrain producer source SHA-256",
        )
    if expected_producer_executable_sha256 is not None:
        expected_producer_executable_sha256 = _sha(
            expected_producer_executable_sha256,
            "expected child terrain producer executable SHA-256",
        )

    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("child terrain export directory must not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("child terrain export must be a real directory")
    inventory = {entry.name for entry in os.scandir(root)}
    required = {GM1_CHILD_TERRAIN_PAYLOAD, "manifest.json", "receipt.json"}
    if inventory != required:
        raise ValueError(
            "child terrain directory inventory mismatch; "
            f"missing={sorted(required - inventory)}, extra={sorted(inventory - required)}"
        )
    payload_path = root / GM1_CHILD_TERRAIN_PAYLOAD
    manifest_path = root / "manifest.json"
    receipt_path = root / "receipt.json"
    if payload_path.is_symlink() or not payload_path.is_file():
        raise ValueError("child terrain payload must be a regular non-symlink file")
    manifest_raw, manifest = _read_canonical(manifest_path, "child terrain manifest")
    receipt_raw, receipt = _read_canonical(receipt_path, "child terrain receipt")
    identities = {
        "manifest": (hashlib.sha256(manifest_raw).hexdigest(), expected_manifest_sha256),
        "receipt": (hashlib.sha256(receipt_raw).hexdigest(), expected_receipt_sha256),
        "payload": (_sha_path(payload_path), expected_payload_sha256),
    }
    for label, (actual, expected) in identities.items():
        if actual != expected:
            raise ValueError(
                f"child terrain {label} SHA-256 is {actual}; independently expected {expected}"
            )
    # The receipt is what anchors this publication; an unnamed producer
    # expectation therefore comes from it, and the manifest must agree below.
    if expected_producer_source_sha256 is None:
        expected_producer_source_sha256 = _sha(
            receipt.get("producer_source_sha256"),
            "child terrain receipt producer_source_sha256",
        )
    if expected_producer_executable_sha256 is None:
        expected_producer_executable_sha256 = _sha(
            receipt.get("producer_executable_sha256"),
            "child terrain receipt producer_executable_sha256",
        )

    manifest = _exact(
        manifest,
        {
            "abi",
            "artifact_schema",
            "authority",
            "claim_limit",
            "dtype",
            "itemsize_bytes",
            "model",
            "order",
            "parent",
            "payload",
            "producer",
            "schema",
            "source",
            "statistics",
            "status",
            "synthetic_translation",
            "transform",
        },
        "child terrain manifest",
    )
    parent = _exact(
        manifest["parent"], {"checkpoint_sha256", "plan_schema", "plan_sha256"}, "child terrain parent"
    )
    payload = _exact(
        manifest["payload"], {"file", "nbytes", "sha256", "shape_yx"}, "child terrain payload"
    )
    producer = _exact(
        manifest["producer"], {"executable_sha256", "source_sha256"}, "child terrain producer"
    )
    statistics = _exact(
        manifest["statistics"],
        {"all_finite", "maximum_f32_bits", "minimum_f32_bits", "nonzero_count"},
        "child terrain statistics",
    )
    manifest_authority = _exact(
        manifest["authority"],
        {"model", "model_source", "tarball_sha256"},
        "child terrain manifest authority",
    )
    if (
        manifest["schema"] != GM1_CHILD_TERRAIN_MANIFEST_SCHEMA
        or manifest["artifact_schema"] != GM1_CHILD_TERRAIN_SCHEMA
        or manifest["abi"] != GM1_CHILD_TERRAIN_ABI
        or manifest["status"] != GM1_CHILD_TERRAIN_STATUS
        or manifest["model"] != "GM1"
        or manifest["dtype"] != "little-endian-f32"
        or manifest["itemsize_bytes"] != 4
        or manifest["order"] != "C"
        or manifest["synthetic_translation"] is not False
        or manifest_authority
        != {
            "model": "GM1",
            "model_source": "CM1 r21.1",
            "tarball_sha256": "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7",
        }
        or parent["plan_schema"] != "gm1.source-coordinate-50m-child-remap-plan/v1"
        or _sha(parent["checkpoint_sha256"], "child terrain parent checkpoint") is None
        or _sha(parent["plan_sha256"], "child terrain parent plan") is None
        or payload
        != {
            "file": GM1_CHILD_TERRAIN_PAYLOAD,
            "nbytes": 360000,
            "sha256": expected_payload_sha256,
            "shape_yx": [300, 300],
        }
        or producer
        != {
            "executable_sha256": expected_producer_executable_sha256,
            "source_sha256": expected_producer_source_sha256,
        }
        or statistics.get("all_finite") is not True
        or statistics.get("nonzero_count") != 90000
        or not isinstance(statistics.get("minimum_f32_bits"), str)
        or _HEX8.fullmatch(statistics["minimum_f32_bits"]) is None
        or not isinstance(statistics.get("maximum_f32_bits"), str)
        or _HEX8.fullmatch(statistics["maximum_f32_bits"]) is None
    ):
        raise ValueError("child terrain manifest selection, layout, or identity changed")
    _validate_source(manifest["source"])
    _validate_transform(manifest["transform"])

    receipt = _exact(
        receipt,
        {
            "abi",
            "all_finite",
            "authority",
            "claim_limit",
            "dtype",
            "manifest_sha256",
            "maximum_f32_bits",
            "minimum_f32_bits",
            "model",
            "nonzero_count",
            "order",
            "parent_checkpoint_sha256",
            "parent_plan_sha256",
            "payload_nbytes",
            "payload_sha256",
            "producer_executable_sha256",
            "producer_source_sha256",
            "schema",
            "shape_yx",
            "source",
            "source_coordinate_child_terrain_sha256",
            "status",
            "synthetic_translation",
            "transform",
        },
        "child terrain receipt",
    )
    authority = _exact(receipt["authority"], {"model", "model_source", "tarball_sha256"}, "child terrain authority")
    if (
        receipt["schema"] != GM1_CHILD_TERRAIN_SCHEMA
        or receipt["abi"] != GM1_CHILD_TERRAIN_ABI
        or receipt["status"] != GM1_CHILD_TERRAIN_STATUS
        or receipt["model"] != "GM1"
        or receipt["all_finite"] is not True
        or receipt["dtype"] != "little-endian-f32"
        or receipt["order"] != "C"
        or receipt["shape_yx"] != [300, 300]
        or receipt["payload_nbytes"] != 360000
        or receipt["payload_sha256"] != expected_payload_sha256
        or receipt["source_coordinate_child_terrain_sha256"] != expected_payload_sha256
        or receipt["manifest_sha256"] != expected_manifest_sha256
        or receipt["producer_source_sha256"] != expected_producer_source_sha256
        or receipt["producer_executable_sha256"] != expected_producer_executable_sha256
        or receipt["parent_plan_sha256"] != parent["plan_sha256"]
        or receipt["parent_checkpoint_sha256"] != parent["checkpoint_sha256"]
        or receipt["synthetic_translation"] is not False
        or receipt["nonzero_count"] != 90000
        or receipt["minimum_f32_bits"] != statistics["minimum_f32_bits"]
        or receipt["maximum_f32_bits"] != statistics["maximum_f32_bits"]
        or authority
        != {
            "model": "GM1",
            "model_source": "CM1 r21.1",
            "tarball_sha256": "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7",
        }
    ):
        raise ValueError("child terrain receipt identity or source binding changed")
    _validate_source(receipt["source"])
    _validate_transform(receipt["transform"])

    np = importlib.import_module("numpy")
    terrain = np.memmap(
        payload_path, dtype="<f4", mode="r", shape=(300, 300), order="C"
    )
    if (
        not terrain.flags.c_contiguous
        or terrain.dtype != np.dtype("float32")
        or not bool(np.all(np.isfinite(terrain)))
        or hashlib.sha256(memoryview(terrain)).hexdigest() != expected_payload_sha256
        or f"{int(np.min(terrain).view(np.uint32)):08x}" != statistics["minimum_f32_bits"]
        or f"{int(np.max(terrain).view(np.uint32)):08x}" != statistics["maximum_f32_bits"]
        or int(np.count_nonzero(terrain)) != 90000
    ):
        raise ValueError("child terrain payload words or statistics changed")
    terrain.flags.writeable = False
    return LoadedGm1ChildTerrain(
        directory=root,
        manifest_path=manifest_path,
        payload_path=payload_path,
        receipt_path=receipt_path,
        manifest_sha256=expected_manifest_sha256,
        payload_sha256=expected_payload_sha256,
        receipt_sha256=expected_receipt_sha256,
        producer_source_sha256=expected_producer_source_sha256,
        producer_executable_sha256=expected_producer_executable_sha256,
        terrain_msl=terrain,
        manifest=MappingProxyType(dict(manifest)),
        receipt=MappingProxyType(dict(receipt)),
    )


def launch_cuda_gm1_child_terrain_metrics(
    export: LoadedGm1ChildTerrain,
    *,
    stream: Any | None = None,
    compiled: CompiledStretchZ1Terrain | None = None,
    vertical: Any | None = None,
) -> CudaGm1ChildTerrainMetricsLaunch:
    """Upload the exact real terrain and launch its selected static metrics."""

    if not isinstance(export, LoadedGm1ChildTerrain):
        raise TypeError("export must be LoadedGm1ChildTerrain")
    # Whichever admitted terrain is handed in is the surface these metrics
    # describe, and the receipt below records exactly which one it was.  The
    # loader has already required it to be internally consistent and the
    # composing envelope has already required it to be the terrain the rest of
    # the child was built over; refusing anything but the first 50 m child's
    # surface here would refuse every later child for nothing that is wrong
    # with it.
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    with stream:
        device_terrain = cupy.asarray(export.terrain_msl, dtype=cupy.float32, order="C")
        metrics = launch_cuda_stretch_z1_terrain(
            device_terrain,
            stream=stream,
            compiled=compiled,
            vertical=vertical,
        )
    receipt = MappingProxyType(
        {
            "schema": GM1_CHILD_TERRAIN_METRICS_SCHEMA,
            "status": "PASS_REAL_SOURCE_TERRAIN_LAUNCHED_TO_EXACT_STATIC_METRICS",
            "fixture_class": "real source-coordinate f02 terrain; no synthetic data",
            "source": {
                "schema": GM1_CHILD_TERRAIN_SCHEMA,
                "abi": GM1_CHILD_TERRAIN_ABI,
                "manifest_sha256": export.manifest_sha256,
                "payload_sha256": export.payload_sha256,
                "receipt_sha256": export.receipt_sha256,
                "parent_plan_sha256": export.receipt["parent_plan_sha256"],
                "parent_checkpoint_sha256": export.receipt["parent_checkpoint_sha256"],
                "synthetic_translation": False,
            },
            "metrics": {
                "schema": CUDA_STRETCH_Z1_TERRAIN_SCHEMA,
                "abi": CUDA_STRETCH_Z1_TERRAIN_ABI,
                "kernel_source_sha256": CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256,
                "shape_zyx": [75, 300, 300],
                "carrier_names": list(metrics.state.arrays),
                "launch_receipt": metrics.receipt,
            },
            "claim_limit": GM1_CHILD_TERRAIN_CLAIM_LIMIT,
            "next_required_producers": [
                "terrain-varying hydrostatic/base-state reconstruction",
                "fresh four-open child halos and radiation histories",
                "terrain-following WSM6 total geopotential",
                "atmospheric parent-to-child remap",
            ],
        }
    )
    return CudaGm1ChildTerrainMetricsLaunch(
        metrics.ready_event, export, metrics, receipt
    )


__all__ = (
    "CudaGm1ChildTerrainMetricsLaunch",
    "GM1_CHILD_PARENT_CHECKPOINT_SHA256",
    "GM1_CHILD_PLAN_SHA256",
    "GM1_CHILD_TERRAIN_ABI",
    "GM1_CHILD_TERRAIN_CLAIM_LIMIT",
    "GM1_CHILD_TERRAIN_EXPECTED_SHA256",
    "GM1_CHILD_TERRAIN_MANIFEST_SCHEMA",
    "GM1_CHILD_TERRAIN_METRICS_SCHEMA",
    "GM1_CHILD_TERRAIN_SCHEMA",
    "LoadedGm1ChildTerrain",
    "launch_cuda_gm1_child_terrain_metrics",
    "load_gm1_child_terrain_export",
)
