"""Opaque CM1FCP-to-CUDA carrier transport for the real-product profile.

Rust verifies and transforms the forecast checkpoint into one independently
hashed byte arena.  This module is intentionally not a scientific decoder:
it validates the canonical transport manifest, copies the arena byte-for-byte
to one CUDA allocation, and exposes the Rust-described ``<f4`` C-order views.
It performs no regridding, thermodynamics, staggering, halo fill, terrain
metric construction, pressure rebalance, or numerical initialization.

Importing this module is CUDA-cold.  CuPy and NumPy are imported only after
the complete on-disk export and every independently supplied hash pass.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


CUDA_REAL_EXPORT_SCHEMA = "gpuwm-cm1.cuda-live-export/v1"
CUDA_REAL_EXPORT_RECEIPT_SCHEMA = "gpuwm-cm1.cuda-live-export-receipt/v1"
CUDA_REAL_CARRIER_ABI = "gpuwm-cm1.cuda-live-carriers/v1"
CUDA_REAL_INITIALIZATION_SCHEMA = "gpuwm-cm1.cuda-real-initialization/v1"
CUDA_REAL_TEST_INITIALIZATION_SCHEMA = (
    "gpuwm-cm1.cuda-real-initialization-test-double/v1"
)
CUDA_REAL_INITIALIZATION_CLAIM_LIMIT = (
    "byte-exact physical-domain CM1 carrier allocation/copy only; terrain metrics, "
    "halos, derived pressure/density, pressure rebalance, closure state, boundary "
    "history, dynamics, physics, and forecast stepping remain deferred"
)
CUDA_REAL_TEST_CLAIM_LIMIT = (
    "dependency-injected byte-transport orchestration test only; no CUDA residency, "
    "stream ordering, hardware, dynamics, or forecast claim"
)
CUDA_REAL_EXPORT_CLAIM_LIMIT = (
    "byte-exact physical-domain CM1 carrier transport only; terrain metrics, "
    "halos, derived pressure/density, pressure rebalance, closure state, boundary "
    "history, and forecast stepping remain deferred"
)
PAYLOAD_NAME = "carrier-payload.f32le"
MANIFEST_NAME = "manifest.json"
RECEIPT_NAME = "receipt.json"
ITEMSIZE = 4
# The loader reads the source extent and spacing out of the export's own
# manifest and validates the crop against them, so a parent published at
# another extent or spacing is transported unchanged rather than refused.  The
# frozen 75x300x300 / 100 m x 100 m x 200 m constants this module used to
# compare against are gone rather than left beside their replacement: a dead
# constant next to the check that superseded it is the thing a later reader
# mistakes for the live gate.  The domain of one export is now
# `LoadedLiveExport.source_shape_zyx` / `.source_spacing_m`.
SPACING_KEYS = ("target_dx_m_f64_bits", "target_dy_m_f64_bits", "target_dz_m_f64_bits")
SPECIES_ORDER = ("qv", "qc", "qr", "qi", "qs", "qg")
CARRIER_NAMES = (
    "pi0",
    "prs0",
    "qv0",
    "rho0",
    "th0",
    "u0",
    "v0",
    "ua",
    "u3d",
    "va",
    "v3d",
    "wa",
    "w3d",
    "ppi",
    "pp3d",
    "ppx",
    "tha",
    "th3d",
    "qa",
    "q3d",
    "terrain_msl",
)
_DUPLICATES = {
    "u3d": "ua",
    "v3d": "va",
    "w3d": "wa",
    "pp3d": "ppi",
    "ppx": "ppi",
    "th3d": "tha",
    "q3d": "qa",
}
_SOURCE_ARRAYS = {
    **{name: (name,) for name in CARRIER_NAMES[:7]},
    "ua": ("u3d",),
    "u3d": ("u3d",),
    "va": ("v3d",),
    "v3d": ("v3d",),
    "wa": ("w3d",),
    "w3d": ("w3d",),
    "ppi": ("pp3d",),
    "pp3d": ("pp3d",),
    "ppx": ("pp3d",),
    "tha": ("th3d",),
    "th3d": ("th3d",),
    "qa": SPECIES_ORDER,
    "q3d": SPECIES_ORDER,
    "terrain_msl": ("terrain_msl",),
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")
_HEX8 = re.compile(r"[0-9a-f]{8}\Z")
_DEFAULT_DRIVER_ATTESTATION = object()


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lower-case hexadecimal characters")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_canonical_json(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    if raw != _canonical_json_bytes(value):
        raise ValueError(f"{path.name} is not canonical compact sorted-key JSON")
    return raw, value


def _exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} key inventory mismatch; "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )


def _regular_file(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as error:
        raise ValueError(f"required live-export file is missing: {path.name}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"live-export entry must be a regular non-symlink file: {path.name}")


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _positive_spacing(value: object, label: str) -> float:
    """Decode one f64-bits grid spacing and require it to be a real length."""

    if not isinstance(value, str) or _HEX16.fullmatch(value) is None:
        raise ValueError(f"{label} must be sixteen lower-case hexadecimal characters")
    spacing = struct.unpack("<d", int(value, 16).to_bytes(8, "little"))[0]
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(
            f"{label} decodes to {spacing!r}; a non-finite or non-positive grid spacing "
            "would make every derived metric, height and crop bound meaningless"
        )
    return spacing


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class LiveCarrierSpec:
    name: str
    shape: tuple[int, ...]
    offset_nbytes: int
    nbytes: int
    sha256: str
    role: str
    source_arrays: tuple[str, ...]
    duplicate_of: str | None
    species_order: tuple[str, ...]
    minimum_f32_bits: str
    maximum_f32_bits: str
    finite_count: int
    nonzero_count: int

    @property
    def words(self) -> int:
        count = 1
        for extent in self.shape:
            count *= extent
        return count


@dataclass(frozen=True, slots=True)
class LoadedLiveExport:
    directory: Path
    payload_path: Path
    manifest_sha256: str
    manifest_nbytes: int
    payload_sha256: str
    payload_nbytes: int
    source_artifact_sha256: str
    source_contract_sha256: str
    crop_origin_zyx: tuple[int, int, int]
    crop_count_zyx: tuple[int, int, int]
    source_shape_zyx: tuple[int, int, int]
    source_spacing_m: tuple[float, float, float]
    carriers: tuple[LiveCarrierSpec, ...]

    def carrier(self, name: str) -> LiveCarrierSpec:
        for carrier in self.carriers:
            if carrier.name == name:
                return carrier
        raise KeyError(name)

    @property
    def is_full_domain(self) -> bool:
        return (
            self.crop_origin_zyx == (0, 0, 0)
            and self.crop_count_zyx == self.source_shape_zyx
        )

    def source_grid_record(self) -> dict[str, object]:
        """The parent domain this export declares, for a published receipt.

        A consumer that stamps a grid into its receipt must stamp this one.
        The spacing here is what every vertical and horizontal finite
        difference over these carriers has to be weighted by; a receipt that
        named a different spacing would date the whole product to a grid the
        parent never ran on.
        """

        return {
            "shape_zyx": list(self.source_shape_zyx),
            "dx_m": self.source_spacing_m[0],
            "dy_m": self.source_spacing_m[1],
            "dz_m": self.source_spacing_m[2],
        }


def load_cuda_live_export(
    directory: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
    expected_source_artifact_sha256: str,
    expected_source_contract_sha256: str,
) -> LoadedLiveExport:
    """Verify a Rust live-export without importing CUDA or numerical libraries."""

    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, "expected manifest SHA-256"
    )
    expected_payload_sha256 = _require_sha256(
        expected_payload_sha256, "expected payload SHA-256"
    )
    expected_source_artifact_sha256 = _require_sha256(
        expected_source_artifact_sha256, "expected source artifact SHA-256"
    )
    expected_source_contract_sha256 = _require_sha256(
        expected_source_contract_sha256, "expected source contract SHA-256"
    )
    requested_root = Path(directory)
    if requested_root.is_symlink():
        raise ValueError("live-export path must be a real directory, not a symlink")
    root = requested_root.resolve()
    if not root.is_dir():
        raise ValueError("live-export path must be a real directory, not a symlink")
    inventory = {entry.name for entry in os.scandir(root)}
    expected_inventory = {PAYLOAD_NAME, MANIFEST_NAME, RECEIPT_NAME}
    if inventory != expected_inventory:
        raise ValueError(
            "live-export directory inventory mismatch; "
            f"missing={sorted(expected_inventory - inventory)!r}, "
            f"extra={sorted(inventory - expected_inventory)!r}"
        )
    payload_path = root / PAYLOAD_NAME
    manifest_path = root / MANIFEST_NAME
    receipt_path = root / RECEIPT_NAME
    for path in (payload_path, manifest_path, receipt_path):
        _regular_file(path)

    manifest_raw, manifest = _read_canonical_json(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"live manifest SHA-256 is {manifest_sha256}; "
            f"independently expected {expected_manifest_sha256}"
        )
    receipt_raw, receipt = _read_canonical_json(receipt_path)
    del receipt_raw
    _exact_keys(
        receipt,
        {
            "admission",
            "carrier_abi",
            "carrier_count",
            "claim_limit",
            "manifest_nbytes",
            "manifest_sha256",
            "payload_nbytes",
            "payload_sha256",
            "publication_contract",
            "schema",
            "source_artifact_sha256",
            "source_contract_sha256",
        },
        "live-export receipt",
    )
    if receipt["schema"] != CUDA_REAL_EXPORT_RECEIPT_SCHEMA:
        raise ValueError("live-export receipt schema is unsupported")
    if receipt["admission"] != "real-product-300x300x75":
        raise ValueError("live-export receipt admission is unsupported")
    if receipt["carrier_abi"] != CUDA_REAL_CARRIER_ABI:
        raise ValueError("live-export receipt carrier ABI is unsupported")
    if receipt["claim_limit"] != CUDA_REAL_EXPORT_CLAIM_LIMIT:
        raise ValueError("live-export receipt claim boundary changed")
    if receipt["manifest_sha256"] != manifest_sha256:
        raise ValueError("live-export receipt does not anchor the manifest bytes")
    if receipt["manifest_nbytes"] != len(manifest_raw):
        raise ValueError("live-export receipt manifest byte count is wrong")

    _exact_keys(
        manifest,
        {
            "admission",
            "carrier_abi",
            "carriers",
            "claim_limit",
            "crop",
            "dtype",
            "itemsize",
            "order",
            "payload",
            "schema",
            "source",
            "source_grid",
        },
        "live-export manifest",
    )
    if manifest["schema"] != CUDA_REAL_EXPORT_SCHEMA:
        raise ValueError("live-export manifest schema is unsupported")
    if manifest["carrier_abi"] != CUDA_REAL_CARRIER_ABI:
        raise ValueError("live-export carrier ABI is unsupported")
    if manifest["admission"] != "real-product-300x300x75":
        raise ValueError("live-export admission is not the sealed real-product source")
    if manifest["dtype"] != "<f4" or manifest["itemsize"] != ITEMSIZE:
        raise ValueError("live-export top-level dtype must be little-endian float32")
    if manifest["order"] != "C":
        raise ValueError("live-export top-level order must be C")
    if manifest["claim_limit"] != CUDA_REAL_EXPORT_CLAIM_LIMIT:
        raise ValueError("live-export manifest claim boundary changed")

    source = manifest["source"]
    if not isinstance(source, dict):
        raise ValueError("live-export source must be an object")
    _exact_keys(
        source,
        {
            "artifact_nbytes",
            "artifact_sha256",
            "contract_sha256",
            "dt_f64_bits",
            "model_time_f64_bits",
            "profile_id",
            "step",
        },
        "live-export source",
    )
    if source["artifact_sha256"] != expected_source_artifact_sha256:
        raise ValueError("live-export source artifact differs from its independent hash")
    if source["contract_sha256"] != expected_source_contract_sha256:
        raise ValueError("live-export source contract differs from its independent hash")
    if source["profile_id"] != "hrrr-cm1-real-state-v1":
        raise ValueError("live-export source profile is unsupported")
    _positive_int(source["artifact_nbytes"], "source artifact byte count")
    if source["step"] != 0 or source["model_time_f64_bits"] != "0000000000000000":
        raise ValueError("live initialization requires a step-zero time-zero checkpoint")
    if source["dt_f64_bits"] != "3fe0000000000000":
        raise ValueError("live initialization requires exact dt=0.5 s")

    source_grid = manifest["source_grid"]
    if not isinstance(source_grid, dict):
        raise ValueError("live-export source_grid must be an object")
    _exact_keys(
        source_grid,
        {"shape_zyx", *SPACING_KEYS},
        "live-export source_grid",
    )
    # The transported domain is whatever the parent actually ran, read out of
    # the export's own manifest.  What is refused is a grid that cannot describe
    # the bytes: a non-positive extent, or a spacing that is not a finite
    # positive length, either of which would make the crop bounds below and
    # every downstream metric meaningless.
    raw_shape = source_grid["shape_zyx"]
    if not isinstance(raw_shape, list) or len(raw_shape) != 3:
        raise ValueError("live-export source grid shape must be a z/y/x triple")
    source_shape = tuple(
        _positive_int(extent, f"source grid axis {axis}")
        for axis, extent in enumerate(raw_shape)
    )
    source_spacing = tuple(
        _positive_spacing(source_grid[key], key) for key in SPACING_KEYS
    )

    crop = manifest["crop"]
    if not isinstance(crop, dict):
        raise ValueError("live-export crop must be an object")
    _exact_keys(crop, {"count_zyx", "origin_zyx", "stagger_rule"}, "live crop")
    origin = tuple(
        _positive_int(value, f"crop origin axis {axis}", allow_zero=True)
        for axis, value in enumerate(crop["origin_zyx"])
    )
    count = tuple(
        _positive_int(value, f"crop count axis {axis}")
        for axis, value in enumerate(crop["count_zyx"])
    )
    if len(origin) != 3 or len(count) != 3:
        raise ValueError("live-export crop origin/count must each have z/y/x rank three")
    if any(origin[axis] + count[axis] > source_shape[axis] for axis in range(3)):
        raise ValueError(
            "live-export crop exceeds the source grid its own manifest records; "
            "the transport would copy carrier words the parent domain does not contain"
        )

    payload = manifest["payload"]
    if not isinstance(payload, dict):
        raise ValueError("live-export payload must be an object")
    _exact_keys(payload, {"file", "nbytes", "sha256"}, "live payload")
    if payload["file"] != PAYLOAD_NAME:
        raise ValueError("live-export payload file name is unsupported")
    payload_nbytes = _positive_int(payload["nbytes"], "payload byte count")
    payload_sha256 = _require_sha256(payload["sha256"], "payload SHA-256")
    if payload_sha256 != expected_payload_sha256:
        raise ValueError("live-export payload differs from its independent SHA-256")
    if receipt["payload_sha256"] != payload_sha256 or receipt["payload_nbytes"] != (
        payload_nbytes
    ):
        raise ValueError("live-export receipt and manifest payload identities differ")
    if receipt["source_artifact_sha256"] != expected_source_artifact_sha256 or receipt[
        "source_contract_sha256"
    ] != expected_source_contract_sha256:
        raise ValueError("live-export receipt source identities differ from independent hashes")
    if os.lstat(payload_path).st_size != payload_nbytes:
        raise ValueError("live-export payload file size differs from its manifest")

    raw_carriers = manifest["carriers"]
    if not isinstance(raw_carriers, list):
        raise ValueError("live-export carriers must be an array")
    carriers = tuple(_parse_carrier(value, index) for index, value in enumerate(raw_carriers))
    if tuple(carrier.name for carrier in carriers) != CARRIER_NAMES:
        raise ValueError("live-export carrier order/inventory differs from the fixed ABI")
    if receipt["carrier_count"] != len(carriers):
        raise ValueError("live-export receipt carrier count is wrong")
    _validate_carrier_geometry(carriers, count)

    payload_hasher = hashlib.sha256()
    with payload_path.open("rb", buffering=0) as handle:
        offset = 0
        for carrier in carriers:
            if carrier.offset_nbytes != offset:
                raise ValueError(f"carrier {carrier.name} offset is not contiguous")
            field_hasher = hashlib.sha256()
            remaining = carrier.nbytes
            while remaining:
                block = handle.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"payload truncates carrier {carrier.name}")
                field_hasher.update(block)
                payload_hasher.update(block)
                remaining -= len(block)
            if field_hasher.hexdigest() != carrier.sha256:
                raise ValueError(f"carrier {carrier.name} payload SHA-256 mismatch")
            offset += carrier.nbytes
        if handle.read(1):
            raise ValueError("live payload has trailing bytes")
    if offset != payload_nbytes or payload_hasher.hexdigest() != payload_sha256:
        raise ValueError("live payload aggregate SHA-256 or byte count mismatch")

    return LoadedLiveExport(
        directory=root,
        payload_path=payload_path,
        manifest_sha256=manifest_sha256,
        manifest_nbytes=len(manifest_raw),
        payload_sha256=payload_sha256,
        payload_nbytes=payload_nbytes,
        source_artifact_sha256=expected_source_artifact_sha256,
        source_contract_sha256=expected_source_contract_sha256,
        crop_origin_zyx=origin,  # type: ignore[arg-type]
        crop_count_zyx=count,  # type: ignore[arg-type]
        source_shape_zyx=source_shape,  # type: ignore[arg-type]
        source_spacing_m=source_spacing,  # type: ignore[arg-type]
        carriers=carriers,
    )


def _parse_carrier(value: object, index: int) -> LiveCarrierSpec:
    if not isinstance(value, dict):
        raise ValueError(f"carrier {index} must be an object")
    _exact_keys(
        value,
        {
            "duplicate_of",
            "dtype",
            "itemsize",
            "name",
            "nbytes",
            "offset_nbytes",
            "order",
            "role",
            "shape",
            "sha256",
            "source_arrays",
            "species_order",
            "statistics",
        },
        f"carrier {index}",
    )
    name = _string(value["name"], f"carrier {index} name")
    if value["dtype"] != "<f4" or value["itemsize"] != ITEMSIZE or value["order"] != "C":
        raise ValueError(f"carrier {name} layout is not <f4 C-order")
    shape = tuple(
        _positive_int(extent, f"carrier {name} shape extent") for extent in value["shape"]
    )
    if not shape:
        raise ValueError(f"carrier {name} shape must have positive rank")
    words = 1
    for extent in shape:
        words *= extent
    nbytes = _positive_int(value["nbytes"], f"carrier {name} nbytes")
    if nbytes != words * ITEMSIZE:
        raise ValueError(f"carrier {name} byte count disagrees with shape")
    duplicate = value["duplicate_of"]
    if duplicate is not None and (not isinstance(duplicate, str) or not duplicate):
        raise ValueError(f"carrier {name} duplicate_of must be null or a name")
    sources = tuple(_string(item, f"carrier {name} source") for item in value["source_arrays"])
    species = tuple(_string(item, f"carrier {name} species") for item in value["species_order"])
    statistics = value["statistics"]
    if not isinstance(statistics, dict):
        raise ValueError(f"carrier {name} statistics must be an object")
    _exact_keys(
        statistics,
        {"finite_count", "maximum_f32_bits", "minimum_f32_bits", "nonzero_count"},
        f"carrier {name} statistics",
    )
    finite_count = _positive_int(statistics["finite_count"], f"carrier {name} finite count")
    nonzero_count = _positive_int(
        statistics["nonzero_count"],
        f"carrier {name} nonzero count",
        allow_zero=True,
    )
    if finite_count != words or nonzero_count > words:
        raise ValueError(f"carrier {name} finite/nonzero counts are inconsistent")
    minimum = _string(statistics["minimum_f32_bits"], f"carrier {name} minimum bits")
    maximum = _string(statistics["maximum_f32_bits"], f"carrier {name} maximum bits")
    if _HEX8.fullmatch(minimum) is None or _HEX8.fullmatch(maximum) is None:
        raise ValueError(f"carrier {name} range bits are not canonical f32 hex")
    return LiveCarrierSpec(
        name=name,
        shape=shape,
        offset_nbytes=_positive_int(
            value["offset_nbytes"], f"carrier {name} offset", allow_zero=True
        ),
        nbytes=nbytes,
        sha256=_require_sha256(value["sha256"], f"carrier {name} SHA-256"),
        role=_string(value["role"], f"carrier {name} role"),
        source_arrays=sources,
        duplicate_of=duplicate,
        species_order=species,
        minimum_f32_bits=minimum,
        maximum_f32_bits=maximum,
        finite_count=finite_count,
        nonzero_count=nonzero_count,
    )


def _validate_carrier_geometry(
    carriers: tuple[LiveCarrierSpec, ...], crop_count: tuple[int, int, int]
) -> None:
    nz, ny, nx = crop_count
    expected_shapes = {
        **{name: (nz,) for name in CARRIER_NAMES[:7]},
        "ua": (nz, ny, nx + 1),
        "u3d": (nz, ny, nx + 1),
        "va": (nz, ny + 1, nx),
        "v3d": (nz, ny + 1, nx),
        "wa": (nz + 1, ny, nx),
        "w3d": (nz + 1, ny, nx),
        "ppi": (nz, ny, nx),
        "pp3d": (nz, ny, nx),
        "ppx": (nz, ny, nx),
        "tha": (nz, ny, nx),
        "th3d": (nz, ny, nx),
        "qa": (len(SPECIES_ORDER), nz, ny, nx),
        "q3d": (len(SPECIES_ORDER), nz, ny, nx),
        "terrain_msl": (ny, nx),
    }
    by_name = {carrier.name: carrier for carrier in carriers}
    for name, shape in expected_shapes.items():
        if by_name[name].shape != shape:
            raise ValueError(f"carrier {name} shape {by_name[name].shape!r} != {shape!r}")
    for name, source in _DUPLICATES.items():
        carrier = by_name[name]
        original = by_name[source]
        if carrier.duplicate_of != source or carrier.sha256 != original.sha256:
            raise ValueError(f"carrier {name} is not an exact distinct duplicate of {source}")
        left = range(carrier.offset_nbytes, carrier.offset_nbytes + carrier.nbytes)
        right = range(original.offset_nbytes, original.offset_nbytes + original.nbytes)
        if max(left.start, right.start) < min(left.stop, right.stop):
            raise ValueError(f"carrier {name} aliases its cadence source {source}")
    for name in ("qa", "q3d"):
        if by_name[name].species_order != SPECIES_ORDER or by_name[name].source_arrays != (
            SPECIES_ORDER
        ):
            raise ValueError(f"carrier {name} has the wrong semantic species order")
    for carrier in carriers:
        if carrier.source_arrays != _SOURCE_ARRAYS[carrier.name]:
            raise ValueError(
                f"carrier {carrier.name} has the wrong Rust source-array mapping"
            )
        if carrier.name not in {"qa", "q3d"} and carrier.species_order:
            raise ValueError(f"non-moisture carrier {carrier.name} declares species")


@dataclass(frozen=True, slots=True)
class _DriverAllocation:
    arena: object
    fields: tuple[tuple[str, object], ...]
    ready_event: object
    stream: object
    pinned_host: object
    device_id: int
    stream_pointer: int
    event_pointer: int
    runtime_identity: tuple[tuple[str, str], ...]


class _LiveDriver(Protocol):
    def allocate_and_copy(self, export: LoadedLiveExport) -> _DriverAllocation: ...


class _CupyLiveDriver:
    def __init__(self, attestation: object) -> None:
        if attestation is not _DEFAULT_DRIVER_ATTESTATION:
            raise RuntimeError("production CuPy live driver lacks private attestation")
        self._cupy = importlib.import_module("cupy")
        self._numpy = importlib.import_module("numpy")

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="strict")
        return str(value)

    @classmethod
    def _arch(cls, value: object) -> str:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if all(isinstance(part, int) and not isinstance(part, bool) for part in value):
                return f"{value[0]}{value[1]}"
        text = cls._text(value).strip().lower()
        for prefix in ("compute_", "sm_"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
        if re.fullmatch(r"[0-9]+\.[0-9]+", text):
            major, minor = text.split(".", 1)
            text = major + minor
        if not text.isdigit():
            raise ValueError("CUDA architecture cannot be canonicalized")
        return text

    def allocate_and_copy(self, export: LoadedLiveExport) -> _DriverAllocation:
        cupy = self._cupy
        numpy = self._numpy
        device = cupy.cuda.Device()
        stream = cupy.cuda.get_current_stream()
        arena = cupy.empty(export.payload_nbytes, dtype=cupy.uint8)
        pinned = cupy.cuda.alloc_pinned_memory(export.payload_nbytes)
        host = numpy.frombuffer(pinned, dtype=numpy.uint8, count=export.payload_nbytes)
        view = memoryview(host).cast("B")
        total = 0
        with export.payload_path.open("rb", buffering=0) as handle:
            while total < export.payload_nbytes:
                copied = handle.readinto(view[total:])
                if copied is None or copied <= 0:
                    raise ValueError("live payload truncated during pinned-host copy")
                total += copied
            if handle.read(1):
                raise ValueError("live payload grew during pinned-host copy")
        if hashlib.sha256(view).hexdigest() != export.payload_sha256:
            raise ValueError("live payload changed after pre-CUDA validation")
        arena.set(host, stream=stream)
        fields = []
        for carrier in export.carriers:
            pointer = arena.data + carrier.offset_nbytes
            array = cupy.ndarray(
                carrier.shape,
                dtype=cupy.float32,
                memptr=pointer,
                order="C",
            )
            if int(array.nbytes) != carrier.nbytes or int(array.data.ptr) != (
                int(arena.data.ptr) + carrier.offset_nbytes
            ):
                raise ValueError(f"CUDA view construction drifted for {carrier.name}")
            fields.append((carrier.name, array))
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
        runtime = cupy.cuda.runtime
        properties = runtime.getDeviceProperties(device.id)
        device_name = properties.get("name", properties.get(b"name", "unknown"))
        identity = (
            ("compute_capability", self._arch(device.compute_capability)),
            ("cuda_driver_version", self._text(runtime.driverGetVersion())),
            ("cuda_runtime_version", self._text(runtime.runtimeGetVersion())),
            ("cupy_version", self._text(cupy.__version__)),
            ("device_id", self._text(device.id)),
            ("device_name", self._text(device_name)),
        )
        return _DriverAllocation(
            arena=arena,
            fields=tuple(fields),
            ready_event=event,
            stream=stream,
            pinned_host=pinned,
            device_id=int(device.id),
            stream_pointer=int(stream.ptr),
            event_pointer=int(event.ptr),
            runtime_identity=identity,
        )


@dataclass(frozen=True, slots=True)
class CudaRealInitialization:
    export: LoadedLiveExport
    arena: object
    fields: tuple[tuple[str, object], ...]
    ready_event: object
    stream: object
    device_id: int
    stream_pointer: int
    event_pointer: int
    runtime_identity: tuple[tuple[str, str], ...]
    audited_default_driver: bool
    schema: str
    claim_limit: str
    _pinned_host: object

    def field(self, name: str) -> object:
        for candidate, value in self.fields:
            if candidate == name:
                return value
        raise KeyError(name)

    def receipt_dict(self) -> dict[str, object]:
        return {
            "arena": {
                "dtype": "uint8 transport / <f4 carrier views",
                "nbytes": self.export.payload_nbytes,
                "payload_sha256": self.export.payload_sha256,
            },
            "audited_default_driver": self.audited_default_driver,
            "carrier_abi": CUDA_REAL_CARRIER_ABI,
            "carriers": [
                {
                    "duplicate_of": carrier.duplicate_of,
                    "name": carrier.name,
                    "nbytes": carrier.nbytes,
                    "offset_nbytes": carrier.offset_nbytes,
                    "sha256": carrier.sha256,
                    "shape": list(carrier.shape),
                }
                for carrier in self.export.carriers
            ],
            "claim_limit": self.claim_limit,
            "crop": {
                "count_zyx": list(self.export.crop_count_zyx),
                "full_domain": self.export.is_full_domain,
                "origin_zyx": list(self.export.crop_origin_zyx),
            },
            "device_id": self.device_id,
            "manifest_sha256": self.export.manifest_sha256,
            "ordering": [
                "all-disk-hashes-and-manifest-validated-before-CUDA-import",
                "one-uint8-device-arena-allocated",
                "one-opaque-pinned-host-to-device-copy-enqueued",
                "non-aliasing-f32-views-created-from-Rust-offsets",
                "ready-event-recorded-on-selected-stream",
            ],
            "ready_event": {
                "event_pointer": self.event_pointer,
                "stream_pointer": self.stream_pointer,
            },
            "runtime_identity": dict(self.runtime_identity),
            "schema": self.schema,
            "source": {
                "artifact_sha256": self.export.source_artifact_sha256,
                "contract_sha256": self.export.source_contract_sha256,
            },
        }


def initialize_cuda_real_carriers(
    export: LoadedLiveExport,
    *,
    allow_cropped_validation: bool = False,
    driver_factory: Callable[[], _LiveDriver] | None = None,
) -> CudaRealInitialization:
    """Allocate and enqueue one opaque real-state arena copy on CUDA.

    Full-domain export is required by default.  Cropped exports are admitted
    only through the explicit validation-only flag and retain the same bounded
    no-forecast claim.
    """

    if not isinstance(export, LoadedLiveExport):
        raise TypeError("export must be a LoadedLiveExport")
    if not isinstance(allow_cropped_validation, bool):
        raise TypeError("allow_cropped_validation must be a boolean")
    if not export.is_full_domain and not allow_cropped_validation:
        raise ValueError(
            "cropped live export is validation-only; set allow_cropped_validation=True"
        )
    if driver_factory is None:
        driver: _LiveDriver = _CupyLiveDriver(_DEFAULT_DRIVER_ATTESTATION)
        audited = True
    else:
        driver = driver_factory()
        audited = False
    allocation = driver.allocate_and_copy(export)
    names = tuple(name for name, _ in allocation.fields)
    if names != CARRIER_NAMES:
        raise ValueError("CUDA driver returned the wrong carrier view inventory/order")
    if allocation.arena is None or allocation.ready_event is None or allocation.stream is None:
        raise ValueError("CUDA driver returned null arena/event/stream handles")
    if allocation.device_id < 0 or allocation.stream_pointer < 0 or allocation.event_pointer < 1:
        raise ValueError("CUDA driver returned invalid device/stream/event metadata")
    if not allocation.runtime_identity:
        raise ValueError("CUDA driver returned no runtime identity")
    schema = (
        CUDA_REAL_INITIALIZATION_SCHEMA
        if audited
        else CUDA_REAL_TEST_INITIALIZATION_SCHEMA
    )
    claim = CUDA_REAL_INITIALIZATION_CLAIM_LIMIT if audited else CUDA_REAL_TEST_CLAIM_LIMIT
    return CudaRealInitialization(
        export=export,
        arena=allocation.arena,
        fields=allocation.fields,
        ready_event=allocation.ready_event,
        stream=allocation.stream,
        device_id=allocation.device_id,
        stream_pointer=allocation.stream_pointer,
        event_pointer=allocation.event_pointer,
        runtime_identity=allocation.runtime_identity,
        audited_default_driver=audited,
        schema=schema,
        claim_limit=claim,
        _pinned_host=allocation.pinned_host,
    )


__all__ = [
    "CARRIER_NAMES",
    "CUDA_REAL_CARRIER_ABI",
    "CUDA_REAL_INITIALIZATION_CLAIM_LIMIT",
    "CUDA_REAL_INITIALIZATION_SCHEMA",
    "CUDA_REAL_TEST_INITIALIZATION_SCHEMA",
    "CudaRealInitialization",
    "LiveCarrierSpec",
    "LoadedLiveExport",
    "initialize_cuda_real_carriers",
    "load_cuda_live_export",
]
