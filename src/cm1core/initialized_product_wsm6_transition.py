"""Live corrected-product binding for the CM1 eqtset=2 ArWen WSM6 seam.

This module performs no scientific transform.  It binds named views from the
strict Rust-owned 49-carrier initialized product, binds an independently
validated Rust-owned total-geopotential carrier, allocates only zero-valued
held/scratch storage, and dispatches the already source-audited
:mod:`cm1core.eqtset2_wsm6_bridge`.

The restart contract includes every mutable carrier owned by this transition,
all four CM1 held tendencies, and all eleven public-component persistence
arrays.  Restore is raw object/copy transport and never replays physics.
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
from types import MappingProxyType
from typing import Callable, Mapping, NoReturn

from .arwen_wsm6_component import (
    ARWEN_WSM6_COMPONENT_ID,
    WSM6_COMPONENT_PERSISTENCE,
    WSM6_SPECIES,
    WSM6_SURFACE_SCRATCH,
    create_arwen_wsm6_component,
)
from .cuda_initialized_product import (
    CARRIER_NAMES,
    CUDA_INITIALIZED_PRODUCT_SCHEMA,
    CudaInitializedProduct,
    INITIALIZED_PRODUCT_CARRIER_ABI,
    LoadedInitializedProduct,
    SPECIES_ORDER,
)
from .eqtset2_wsm6_bridge import (
    EQTSET2_HELD_FIELDS,
    Eqtset2Wsm6Bridge,
    Eqtset2Wsm6Carriers,
    Eqtset2Wsm6StepResult,
)
from .errors import refuse


INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI = (
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-transition/v1"
)
INITIALIZED_PRODUCT_WSM6_RECEIPT_SCHEMA = (
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-receipt/v1"
)
INITIALIZED_PRODUCT_WSM6_RESTART_SCHEMA = (
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-restart/v1"
)
PRODUCT_NZ = 75
PRODUCT_DX_M = struct.unpack(">d", bytes.fromhex("4059000000000000"))[0]
PRODUCT_DY_M = struct.unpack(">d", bytes.fromhex("4059000000000000"))[0]
PRODUCT_DZ_M = struct.unpack(">d", bytes.fromhex("4069000000000000"))[0]
PRODUCT_DT_S = struct.unpack(">d", bytes.fromhex("3fe0000000000000"))[0]
WSM6_GEOPOTENTIAL_SCHEMA = "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary/v1"
WSM6_GEOPOTENTIAL_RECEIPT_SCHEMA = (
    "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary-receipt/v1"
)
WSM6_GEOPOTENTIAL_CARRIER_ABI = "gpuwm-cm1.arwen-wsm6-total-geopotential/v1"
WSM6_GEOPOTENTIAL_PAYLOAD_NAME = "total-geopotential-zf.f32le"
WSM6_GEOPOTENTIAL_MANIFEST_NAME = "manifest.json"
WSM6_GEOPOTENTIAL_RECEIPT_NAME = "receipt.json"
WSM6_GEOPOTENTIAL_CARRIER_NAME = "total_geopotential_zf"
WSM6_GEOPOTENTIAL_CLAIM_LIMIT = (
    "exact Rust binary32 CM1 zf-to-ArWen total-geopotential transform and measured "
    "WSM6 recovery bounds; no raw zf recovery identity, php evolution, halos, "
    "open-boundary history or tendencies, microphysics execution, component "
    "persistence, coupled trajectory, forecast, tornado, or model-validation claim"
)
WSM6_GEOPOTENTIAL_PUBLICATION_CONTRACT = (
    "write the exact binary32 payload, canonical manifest, and canonical receipt "
    "into a new sibling staging directory; synchronize every file; rename only "
    "after all source, transform, and output hashes validate; refuse an existing "
    "destination"
)
WSM6_GEOPOTENTIAL_RECOVERY_STATEMENT_ORDER = (
    "geopotential = binary32(zf * G_binary32); recovered_z = "
    "binary32(geopotential / G_binary32), in that Rust statement order with "
    "IEEE-754 round-to-nearest ties-to-even"
)
WSM6_GEOPOTENTIAL_RECOVERY_METRIC = (
    "for finite nonnegative zf and recovered_z, ULP distance is the absolute "
    "difference of monotonically ordered IEEE-754 binary32 bit patterns; absolute "
    "and relative errors use exact binary32-to-binary64 conversion and relative "
    "error divides by abs(zf), with exact zero mapped to zero relative error"
)
WSM6_GEOPOTENTIAL_AUTHORITY = {
    "cm1_product_gravity_f32_bits": "411cf5c3",
    "consumer_commit": "abb5ff270f263f15b89a272e9b17648406ef0f1d",
    "consumer_gravity_f32_bits": "411cf5c3",
    "consumer_source_files": [
        {
            "anchors": "24",
            "path": "gpuwm/core/constants.py",
            "sha256": "76c8c11d1a75d17fd54d0a80646cec9dde04f649247ec2347d3d32e6c0500e9e",
        },
        {
            "anchors": "102-111",
            "path": "gpuwm/core/wsm6.py",
            "sha256": "3ba72f8e6f3e4453d7c864e0556804a89da2f161f238043ab7670191e5d6af3d",
        },
    ],
    "gravity_identity": (
        "PASS: CM1 initialized-product and ArWen WSM6 consumer gravity are the "
        "same binary32 word"
    ),
}
WSM6_GEOPOTENTIAL_CONSUMER_BINDING = {
    "phb": WSM6_GEOPOTENTIAL_CARRIER_NAME,
    "php": "consumer-owned exact binary32 zero",
    "wsm6_height_expression": "z8w = (phb + state.php) / DTYPE(c.G)",
    "zf_raw_identity": False,
}
WSM6_GEOPOTENTIAL_TRANSFORM = {
    "consumer_gravity_f32_bits": "411cf5c3",
    "operation": (
        "one binary32 multiplication per C-order zf word: "
        "total_geopotential_zf = zf * consumer_gravity"
    ),
    "rounding": "IEEE-754 binary32 round-to-nearest ties-to-even after multiplication",
    "source_carrier_name": "zf",
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")
_HEX8 = re.compile(r"[0-9a-f]{8}\Z")

PRODUCT_STATE_RESTART_FIELDS: tuple[str, ...] = (
    "state.pp3d",
    "state.th3d",
    "state.prs",
    "state.rho",
    "state.rr",
    "state.q3d",
)
PRODUCT_HELD_RESTART_FIELDS: tuple[str, ...] = tuple(
    f"held.{name}" for name in EQTSET2_HELD_FIELDS
)
PRODUCT_COMPONENT_RESTART_FIELDS: tuple[str, ...] = tuple(
    f"component.{name}" for name in WSM6_COMPONENT_PERSISTENCE
)
PRODUCT_WSM6_RESTART_FIELDS: tuple[str, ...] = (
    PRODUCT_STATE_RESTART_FIELDS
    + PRODUCT_HELD_RESTART_FIELDS
    + PRODUCT_COMPONENT_RESTART_FIELDS
)
WSM6_DURABLE_CHECKPOINT_SCHEMA = (
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-checkpoint/v1"
)
WSM6_DURABLE_CHECKPOINT_RECEIPT_SCHEMA = (
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-checkpoint-receipt/v1"
)
WSM6_DURABLE_RESTORE_MANIFEST_SCHEMA = (
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-restore-directory/v1"
)
WSM6_DURABLE_RESTORE_RECEIPT_SCHEMA = (
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-restore-receipt/v1"
)
WSM6_DURABLE_CARRIER_ABI = (
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-restart-carriers/v1"
)
WSM6_DURABLE_CAPTURE_PHASE = "post-transition-after-held-recapture"
WSM6_DURABLE_RESTORE_PHASE = "fresh-bridge-before-next-final-rk-remove"
WSM6_DURABLE_CLAIM_LIMIT = (
    "exact durable persistence and restore of these 21 transition-local arrays "
    "and cadence only; not all 49 product or A-time arrays, halos, boundary "
    "history, closure or dynamics state, scheduler or model clock, idiss, efall, "
    "reflectivity, coupled trajectory, forecast, tornado, or model-validation claim"
)
WSM6_DURABLE_ROLES: tuple[str, ...] = (
    "state-current-pressure-perturbation",
    "state-current-theta-perturbation",
    "diagnostic-pressure-pa",
    "diagnostic-moist-density",
    "reciprocal-diagnostic-moist-density",
    "state-current-packed-moisture",
    "CM1-held-next-step-Exner-tendency",
    "CM1-held-next-step-theta-tendency",
    "CM1-held-next-step-qv-tendency",
    "CM1-held-next-step-qc-tendency",
    "component-held-heating",
    "component-effective-radius-cloud",
    "component-effective-radius-ice",
    "component-effective-radius-snow",
    "component-surface-rain-total",
    "component-surface-rain-increment",
    "component-surface-snow-total",
    "component-surface-snow-increment",
    "component-surface-graupel-total",
    "component-surface-graupel-increment",
    "component-surface-frozen-precipitation-ratio",
)

_CLAIM_LIMIT = (
    "one source-ordered post-final-RK eqtset=2 transition over an exact "
    "corrected-real initialized-product arena using public ArWen WSM6, plus "
    "complete transition-local device and Rust-durable restart capture/restore; "
    "optional exactly-once public WSM6 reflectivity consumption; no halos, live "
    "pre-RK idiss wiring, efall=1, "
    "coupled dycore trajectory, forecast, tornado, or model-validation claim"
)
_GEOPOTENTIAL_ATTESTATION = object()
_CONSUMED_PRODUCT_CONTRACT = {
    "pi0": ("static-base-profile", ("pi0",)),
    "th0": ("static-base-profile", ("th0",)),
    "pp3d": (
        "stage-current-balanced-pressure-perturbation",
        ("pi0", "th0", "th3d", "qv", "qc", "qr", "qi", "qs", "qg", "zh"),
    ),
    "th3d": ("stage-current-theta-perturbation", ("th3d",)),
    "q3d": (
        "stage-current-packed-moisture",
        ("qv", "qc", "qr", "qi", "qs", "qg"),
    ),
    "prs": ("diagnostic-pressure-pa", ("pi0", "ppi")),
    "rho": (
        "diagnostic-moist-density",
        ("prs", "th0", "th3d", "pi0", "ppi", "qv"),
    ),
    "rr": ("reciprocal-diagnostic-moist-density", ("rho",)),
}


@dataclass(frozen=True, slots=True)
class LoadedWsm6GeopotentialAuxiliary:
    directory: Path
    payload_path: Path
    manifest_sha256: str
    manifest_nbytes: int
    payload_sha256: str
    payload_nbytes: int
    receipt_sha256: str
    source_product_manifest_sha256: str
    source_product_payload_sha256: str
    source_product_receipt_sha256: str
    source_restart_artifact_sha256: str
    source_restart_artifact_nbytes: int
    shape: tuple[int, int, int]
    source_zf_sha256: str
    source_zf_offset_nbytes: int
    recovery_word_count: int
    recovery_mismatch_count: int
    recovery_recovered_count: int
    recovery_maximum_ulp_distance: int
    recovery_maximum_absolute_error_m_f64_bits: str
    recovery_maximum_relative_error_f64_bits: str
    recovery_metric: str
    recovery_comparison_statement_order: str


@dataclass(frozen=True, slots=True)
class DurableWsm6RestartCarrier:
    name: str
    role: str
    path: Path
    shape: tuple[int, ...]
    offset_nbytes: int
    nbytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LoadedDurableWsm6Restart:
    directory: Path
    manifest_sha256: str
    manifest_nbytes: int
    checkpoint_artifact_sha256: str
    payload_sha256: str
    payload_nbytes: int
    completed_transitions: int
    source_product_manifest_sha256: str
    source_product_payload_sha256: str
    source_geopotential_manifest_sha256: str
    source_geopotential_payload_sha256: str
    source_artifact_sha256: str
    source_contract_sha256: str
    carriers: tuple[DurableWsm6RestartCarrier, ...]


def _validate_recovery(value: object, *, expected_words: int) -> dict[str, object]:
    recovery = _exact_keys(
        value,
        {
            "comparison_statement_order",
            "maximum_absolute_error_m_f64_bits",
            "maximum_relative_error_f64_bits",
            "maximum_ulp_distance",
            "metric",
            "mismatch_count",
            "recovered_count",
            "word_count",
        },
        "geopotential recovery",
    )
    if (
        recovery["comparison_statement_order"]
        != WSM6_GEOPOTENTIAL_RECOVERY_STATEMENT_ORDER
        or recovery["metric"] != WSM6_GEOPOTENTIAL_RECOVERY_METRIC
    ):
        raise ValueError("geopotential recovery definition changed")
    word_count = _strict_positive_int(recovery["word_count"], "recovery word_count")
    mismatch_count = _strict_nonnegative_int(
        recovery["mismatch_count"], "recovery mismatch_count"
    )
    recovered_count = _strict_nonnegative_int(
        recovery["recovered_count"], "recovery recovered_count"
    )
    maximum_ulp = _strict_nonnegative_int(
        recovery["maximum_ulp_distance"], "recovery maximum_ulp_distance"
    )
    if (
        word_count != expected_words
        or mismatch_count + recovered_count != word_count
        or maximum_ulp > 1
    ):
        raise ValueError("geopotential recovery counts or one-ULP bound changed")
    for key in (
        "maximum_absolute_error_m_f64_bits",
        "maximum_relative_error_f64_bits",
    ):
        bits = recovery[key]
        if not isinstance(bits, str) or _HEX16.fullmatch(bits) is None:
            raise ValueError(f"geopotential recovery {key} is not canonical f64 bits")
        numeric = struct.unpack(">d", bytes.fromhex(bits))[0]
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"geopotential recovery {key} is not finite/nonnegative")
    maximum_absolute = struct.unpack(
        ">d", bytes.fromhex(recovery["maximum_absolute_error_m_f64_bits"])
    )[0]
    if maximum_absolute > 0.0009765625:
        raise ValueError("geopotential recovery absolute error exceeds 2^-10 m")
    return recovery


def load_wsm6_geopotential_auxiliary(
    directory: Path | str,
    product: LoadedInitializedProduct,
    *,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
    expected_receipt_sha256: str,
    expected_source_restart_artifact_sha256: str,
) -> LoadedWsm6GeopotentialAuxiliary:
    """Strictly validate the Rust-owned one-carrier auxiliary while CUDA is cold."""

    if not isinstance(product, LoadedInitializedProduct):
        raise TypeError("product must be a LoadedInitializedProduct")
    expected_manifest_sha256 = _sha256(
        expected_manifest_sha256, "expected auxiliary manifest SHA-256"
    )
    expected_payload_sha256 = _sha256(
        expected_payload_sha256, "expected auxiliary payload SHA-256"
    )
    expected_receipt_sha256 = _sha256(
        expected_receipt_sha256, "expected auxiliary receipt SHA-256"
    )
    expected_source_restart_artifact_sha256 = _sha256(
        expected_source_restart_artifact_sha256,
        "expected source restart artifact SHA-256",
    )
    root = Path(directory).resolve(strict=True)
    if Path(directory).is_symlink() or not root.is_dir():
        raise ValueError("geopotential auxiliary path must be a real directory")
    expected_names = {
        WSM6_GEOPOTENTIAL_PAYLOAD_NAME,
        WSM6_GEOPOTENTIAL_MANIFEST_NAME,
        WSM6_GEOPOTENTIAL_RECEIPT_NAME,
    }
    if {entry.name for entry in root.iterdir()} != expected_names:
        raise ValueError("geopotential auxiliary directory inventory mismatch")
    payload_path = root / WSM6_GEOPOTENTIAL_PAYLOAD_NAME
    manifest_path = root / WSM6_GEOPOTENTIAL_MANIFEST_NAME
    receipt_path = root / WSM6_GEOPOTENTIAL_RECEIPT_NAME
    for path in (payload_path, manifest_path, receipt_path):
        _regular_file(path)

    manifest_raw, manifest = _read_canonical_json(manifest_path)
    receipt_raw, receipt = _read_canonical_json(receipt_path)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("geopotential manifest differs from independent SHA-256")
    if receipt_sha256 != expected_receipt_sha256:
        raise ValueError("geopotential receipt differs from independent SHA-256")

    receipt = _exact_keys(
        receipt,
        {
            "carrier_abi",
            "carrier_count",
            "claim_limit",
            "manifest_nbytes",
            "manifest_sha256",
            "payload_nbytes",
            "payload_sha256",
            "publication_contract",
            "recovery",
            "schema",
            "source_product_manifest_sha256",
            "source_product_payload_sha256",
            "source_restart_artifact_sha256",
        },
        "geopotential receipt",
    )
    if (
        receipt["schema"] != WSM6_GEOPOTENTIAL_RECEIPT_SCHEMA
        or receipt["carrier_abi"] != WSM6_GEOPOTENTIAL_CARRIER_ABI
        or receipt["carrier_count"] != 1
        or receipt["claim_limit"] != WSM6_GEOPOTENTIAL_CLAIM_LIMIT
        or receipt["publication_contract"] != WSM6_GEOPOTENTIAL_PUBLICATION_CONTRACT
    ):
        raise ValueError("geopotential receipt semantic contract changed")
    if (
        receipt["manifest_nbytes"] != len(manifest_raw)
        or receipt["manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("geopotential receipt does not anchor manifest bytes")

    manifest = _exact_keys(
        manifest,
        {
            "authority",
            "carrier",
            "carrier_abi",
            "claim_limit",
            "consumer_binding",
            "dtype",
            "itemsize",
            "order",
            "payload",
            "schema",
            "source_product",
            "source_restart",
            "transform",
        },
        "geopotential manifest",
    )
    if (
        manifest["schema"] != WSM6_GEOPOTENTIAL_SCHEMA
        or manifest["carrier_abi"] != WSM6_GEOPOTENTIAL_CARRIER_ABI
        or manifest["claim_limit"] != WSM6_GEOPOTENTIAL_CLAIM_LIMIT
        or manifest["dtype"] != "<f4"
        or manifest["itemsize"] != 4
        or manifest["order"] != "C"
    ):
        raise ValueError("geopotential manifest top-level contract changed")
    if manifest["authority"] != WSM6_GEOPOTENTIAL_AUTHORITY:
        raise ValueError("geopotential CM1/ArWen authority changed")
    if manifest["consumer_binding"] != WSM6_GEOPOTENTIAL_CONSUMER_BINDING:
        raise ValueError("geopotential consumer binding changed")
    if manifest["transform"] != WSM6_GEOPOTENTIAL_TRANSFORM:
        raise ValueError("geopotential binary32 transform changed")

    source_product = _exact_keys(
        manifest["source_product"],
        {
            "carrier_abi",
            "manifest_sha256",
            "payload_sha256",
            "receipt_sha256",
            "source_artifact_sha256",
            "source_contract_sha256",
        },
        "geopotential source_product",
    )
    product_receipt_path = product.directory / "receipt.json"
    _regular_file(product_receipt_path)
    product_receipt_sha256 = hashlib.sha256(
        product_receipt_path.read_bytes()
    ).hexdigest()
    expected_source_product = {
        "carrier_abi": INITIALIZED_PRODUCT_CARRIER_ABI,
        "manifest_sha256": product.manifest_sha256,
        "payload_sha256": product.payload_sha256,
        "receipt_sha256": product_receipt_sha256,
        "source_artifact_sha256": product.source_artifact_sha256,
        "source_contract_sha256": product.source_contract_sha256,
    }
    if source_product != expected_source_product:
        raise ValueError("geopotential auxiliary belongs to a different product")
    if (
        receipt["source_product_manifest_sha256"] != product.manifest_sha256
        or receipt["source_product_payload_sha256"] != product.payload_sha256
    ):
        raise ValueError("geopotential receipt/product identities differ")

    source_restart = _exact_keys(
        manifest["source_restart"],
        {"artifact_nbytes", "artifact_sha256", "schema"},
        "geopotential source_restart",
    )
    if (
        source_restart["schema"] != "gpuwm-cm1.initialized-product-restart/v1"
        or source_restart["artifact_sha256"] != expected_source_restart_artifact_sha256
        or receipt["source_restart_artifact_sha256"]
        != expected_source_restart_artifact_sha256
    ):
        raise ValueError("geopotential source restart identity changed")
    source_restart_nbytes = _strict_positive_int(
        source_restart["artifact_nbytes"], "source restart artifact_nbytes"
    )

    zf = product.carrier("zf")
    carrier = _exact_keys(
        manifest["carrier"],
        {
            "dtype",
            "itemsize",
            "name",
            "nbytes",
            "offset_nbytes",
            "order",
            "recovery",
            "role",
            "sha256",
            "shape",
            "source_carrier",
            "statistics",
            "units",
        },
        "geopotential carrier",
    )
    if (
        carrier["dtype"] != "<f4"
        or carrier["itemsize"] != 4
        or carrier["name"] != WSM6_GEOPOTENTIAL_CARRIER_NAME
        or carrier["offset_nbytes"] != 0
        or carrier["order"] != "C"
        or carrier["role"] != "ArWen-WSM6-base-total-geopotential"
        or carrier["units"] != "m2 s-2"
        or carrier["shape"] != list(zf.shape)
    ):
        raise ValueError("geopotential carrier ABI changed")
    carrier_nbytes = _strict_positive_int(carrier["nbytes"], "carrier nbytes")
    if carrier_nbytes != zf.nbytes:
        raise ValueError("geopotential carrier byte count differs from source zf")
    source_carrier = _exact_keys(
        carrier["source_carrier"],
        {"name", "nbytes", "offset_nbytes", "sha256"},
        "geopotential source_carrier",
    )
    if source_carrier != {
        "name": "zf",
        "nbytes": zf.nbytes,
        "offset_nbytes": zf.offset_nbytes,
        "sha256": zf.sha256,
    }:
        raise ValueError("geopotential source zf carrier changed")
    statistics = _exact_keys(
        carrier["statistics"],
        {"finite_count", "maximum_f32_bits", "minimum_f32_bits", "nonzero_count"},
        "geopotential statistics",
    )
    finite_count = _strict_positive_int(statistics["finite_count"], "finite_count")
    nonzero_count = _strict_nonnegative_int(
        statistics["nonzero_count"], "nonzero_count"
    )
    if finite_count != zf.words or nonzero_count > finite_count:
        raise ValueError("geopotential statistics counts changed")
    for key in ("minimum_f32_bits", "maximum_f32_bits"):
        if (
            not isinstance(statistics[key], str)
            or _HEX8.fullmatch(statistics[key]) is None
        ):
            raise ValueError(f"geopotential {key} is not canonical f32 bits")

    recovery = _validate_recovery(carrier["recovery"], expected_words=zf.words)
    receipt_recovery = _validate_recovery(receipt["recovery"], expected_words=zf.words)
    if recovery != receipt_recovery:
        raise ValueError("geopotential manifest/receipt recovery contracts differ")

    payload = _exact_keys(
        manifest["payload"], {"file", "nbytes", "sha256"}, "geopotential payload"
    )
    if payload["file"] != WSM6_GEOPOTENTIAL_PAYLOAD_NAME:
        raise ValueError("geopotential payload filename changed")
    payload_nbytes = _strict_positive_int(payload["nbytes"], "payload nbytes")
    payload_sha256 = _sha256(payload["sha256"], "payload sha256")
    carrier_sha256 = _sha256(carrier["sha256"], "carrier sha256")
    if (
        payload_nbytes != carrier_nbytes
        or payload_sha256 != carrier_sha256
        or payload_sha256 != expected_payload_sha256
        or receipt["payload_nbytes"] != payload_nbytes
        or receipt["payload_sha256"] != payload_sha256
        or os.lstat(payload_path).st_size != payload_nbytes
    ):
        raise ValueError("geopotential payload/receipt/carrier identities differ")
    digest = hashlib.sha256()
    with payload_path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    if digest.hexdigest() != payload_sha256:
        raise ValueError("geopotential payload bytes differ from SHA-256")

    return LoadedWsm6GeopotentialAuxiliary(
        directory=root,
        payload_path=payload_path,
        manifest_sha256=manifest_sha256,
        manifest_nbytes=len(manifest_raw),
        payload_sha256=payload_sha256,
        payload_nbytes=payload_nbytes,
        receipt_sha256=receipt_sha256,
        source_product_manifest_sha256=product.manifest_sha256,
        source_product_payload_sha256=product.payload_sha256,
        source_product_receipt_sha256=product_receipt_sha256,
        source_restart_artifact_sha256=expected_source_restart_artifact_sha256,
        source_restart_artifact_nbytes=source_restart_nbytes,
        shape=zf.shape,  # type: ignore[arg-type]
        source_zf_sha256=zf.sha256,
        source_zf_offset_nbytes=zf.offset_nbytes,
        recovery_word_count=int(recovery["word_count"]),
        recovery_mismatch_count=int(recovery["mismatch_count"]),
        recovery_recovered_count=int(recovery["recovered_count"]),
        recovery_maximum_ulp_distance=int(recovery["maximum_ulp_distance"]),
        recovery_maximum_absolute_error_m_f64_bits=str(
            recovery["maximum_absolute_error_m_f64_bits"]
        ),
        recovery_maximum_relative_error_f64_bits=str(
            recovery["maximum_relative_error_f64_bits"]
        ),
        recovery_metric=str(recovery["metric"]),
        recovery_comparison_statement_order=str(recovery["comparison_statement_order"]),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _canonical_compact_json_bytes(value: object) -> bytes:
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
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    if raw != _canonical_compact_json_bytes(value):
        raise ValueError(f"{path.name} is not canonical compact sorted-key JSON")
    return raw, value


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} key inventory mismatch; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lower-case hexadecimal characters")
    return value


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a strict nonnegative integer")
    return value


def _strict_positive_int(value: object, label: str) -> int:
    result = _strict_nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _regular_file(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"required auxiliary file is missing: {path.name}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(
            f"auxiliary entry must be a regular non-symlink file: {path.name}"
        )


def _refuse(
    *, code: str, reason: str, remedy: str, option: str, value: object
) -> NoReturn:
    refuse(
        code=code,
        gate="corrected-product-WSM6-transition",
        reason=reason,
        remedy=remedy,
        option=option,
        value=value,
    )


def _shape(value: object) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError, OverflowError):
        return None


def _is_f32(value: object) -> bool:
    dtype = getattr(value, "dtype", None)
    return bool(
        dtype is not None
        and getattr(dtype, "name", None) == "float32"
        and getattr(dtype, "kind", None) == "f"
        and getattr(dtype, "itemsize", None) == 4
    )


def _is_c_contiguous(value: object) -> bool:
    flags = getattr(value, "flags", None)
    if flags is None:
        return False
    result = getattr(flags, "c_contiguous", None)
    if result is None:
        try:
            result = flags["C_CONTIGUOUS"]
        except (KeyError, TypeError, IndexError):
            return False
    return bool(result)


@dataclass(frozen=True, slots=True)
class CudaWsm6Geopotential:
    """Opaque CUDA carrier loaded from a strict Rust auxiliary envelope.

    Construction is intentionally attested.  The future strict envelope
    loader is the only production caller allowed to mint this object; tests
    use :func:`_test_geopotential` without weakening the public constructor.
    """

    value: object
    source_product_manifest_sha256: str
    source_product_payload_sha256: str
    payload_sha256: str
    manifest_sha256: str
    receipt_sha256: str
    device_id: int
    stream_pointer: int
    ready_event: object
    schema: str
    carrier_name: str
    consumer_gravity_f32_bits: str
    zf_raw_identity: bool
    recovery_word_count: int
    recovery_mismatch_count: int
    recovery_recovered_count: int
    recovery_maximum_ulp_distance: int
    recovery_maximum_absolute_error_m_f64_bits: str
    recovery_maximum_relative_error_f64_bits: str
    recovery_metric: str
    recovery_comparison_statement_order: str
    _arena: object | None
    _pinned_host: object | None
    _attestation: object

    def __post_init__(self) -> None:
        if self._attestation is not _GEOPOTENTIAL_ATTESTATION:
            raise TypeError(
                "CudaWsm6Geopotential must come from the strict Rust envelope loader"
            )


def _make_geopotential(
    *,
    value: object,
    source_product_manifest_sha256: str,
    source_product_payload_sha256: str,
    payload_sha256: str,
    manifest_sha256: str,
    receipt_sha256: str,
    device_id: int,
    stream_pointer: int,
    ready_event: object,
    schema: str,
    carrier_name: str,
    consumer_gravity_f32_bits: str,
    zf_raw_identity: bool,
    recovery_word_count: int,
    recovery_mismatch_count: int,
    recovery_recovered_count: int,
    recovery_maximum_ulp_distance: int,
    recovery_maximum_absolute_error_m_f64_bits: str,
    recovery_maximum_relative_error_f64_bits: str,
    recovery_metric: str,
    recovery_comparison_statement_order: str,
    arena: object | None = None,
    pinned_host: object | None = None,
) -> CudaWsm6Geopotential:
    """Private construction edge used by the strict Rust auxiliary loader."""

    return CudaWsm6Geopotential(
        value=value,
        source_product_manifest_sha256=source_product_manifest_sha256,
        source_product_payload_sha256=source_product_payload_sha256,
        payload_sha256=payload_sha256,
        manifest_sha256=manifest_sha256,
        receipt_sha256=receipt_sha256,
        device_id=device_id,
        stream_pointer=stream_pointer,
        ready_event=ready_event,
        schema=schema,
        carrier_name=carrier_name,
        consumer_gravity_f32_bits=consumer_gravity_f32_bits,
        zf_raw_identity=zf_raw_identity,
        recovery_word_count=recovery_word_count,
        recovery_mismatch_count=recovery_mismatch_count,
        recovery_recovered_count=recovery_recovered_count,
        recovery_maximum_ulp_distance=recovery_maximum_ulp_distance,
        recovery_maximum_absolute_error_m_f64_bits=(
            recovery_maximum_absolute_error_m_f64_bits
        ),
        recovery_maximum_relative_error_f64_bits=(
            recovery_maximum_relative_error_f64_bits
        ),
        recovery_metric=recovery_metric,
        recovery_comparison_statement_order=recovery_comparison_statement_order,
        _arena=arena,
        _pinned_host=pinned_host,
        _attestation=_GEOPOTENTIAL_ATTESTATION,
    )


def initialize_cuda_wsm6_geopotential(
    auxiliary: LoadedWsm6GeopotentialAuxiliary,
    product: CudaInitializedProduct,
) -> CudaWsm6Geopotential:
    """Copy one validated Rust geopotential payload to the product CUDA stream."""

    if not isinstance(auxiliary, LoadedWsm6GeopotentialAuxiliary):
        raise TypeError("auxiliary must be a LoadedWsm6GeopotentialAuxiliary")
    if not isinstance(product, CudaInitializedProduct):
        raise TypeError("product must be a CudaInitializedProduct")
    if (
        product.schema != CUDA_INITIALIZED_PRODUCT_SCHEMA
        or not product.audited_default_driver
    ):
        raise ValueError(
            "geopotential CUDA transport requires the audited production product driver"
        )
    if (
        auxiliary.source_product_manifest_sha256 != product.export.manifest_sha256
        or auxiliary.source_product_payload_sha256 != product.export.payload_sha256
        or auxiliary.shape != product.export.carrier("zf").shape
    ):
        raise ValueError("geopotential auxiliary belongs to a different CUDA product")

    cupy = importlib.import_module("cupy")
    numpy = importlib.import_module("numpy")
    current_device = int(cupy.cuda.runtime.getDevice())
    stream = cupy.cuda.get_current_stream()
    if current_device != product.device_id or int(stream.ptr) != product.stream_pointer:
        raise ValueError(
            "current CUDA device/stream differs from initialized-product ownership"
        )

    arena = cupy.empty(auxiliary.payload_nbytes, dtype=cupy.uint8)
    pinned = cupy.cuda.alloc_pinned_memory(auxiliary.payload_nbytes)
    host = numpy.frombuffer(pinned, dtype=numpy.uint8, count=auxiliary.payload_nbytes)
    host_bytes = memoryview(host).cast("B")
    copied = 0
    digest = hashlib.sha256()
    with auxiliary.payload_path.open("rb", buffering=0) as handle:
        while copied < auxiliary.payload_nbytes:
            count = handle.readinto(host_bytes[copied:])
            if count is None or count <= 0:
                raise ValueError(
                    "geopotential payload truncated during opaque CUDA staging"
                )
            digest.update(host_bytes[copied : copied + count])
            copied += count
        if handle.read(1):
            raise ValueError("geopotential payload grew during opaque CUDA staging")
    if digest.hexdigest() != auxiliary.payload_sha256:
        raise ValueError("geopotential payload changed after disk validation")

    arena.set(host, stream=stream)
    value = cupy.ndarray(
        auxiliary.shape,
        dtype=cupy.float32,
        memptr=arena.data,
        order="C",
    )
    if int(value.nbytes) != auxiliary.payload_nbytes or int(value.data.ptr) != int(
        arena.data.ptr
    ):
        raise ValueError("geopotential CUDA view construction drifted")
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    return _make_geopotential(
        value=value,
        source_product_manifest_sha256=auxiliary.source_product_manifest_sha256,
        source_product_payload_sha256=auxiliary.source_product_payload_sha256,
        payload_sha256=auxiliary.payload_sha256,
        manifest_sha256=auxiliary.manifest_sha256,
        receipt_sha256=auxiliary.receipt_sha256,
        device_id=product.device_id,
        stream_pointer=product.stream_pointer,
        ready_event=event,
        schema=WSM6_GEOPOTENTIAL_SCHEMA,
        carrier_name=WSM6_GEOPOTENTIAL_CARRIER_NAME,
        consumer_gravity_f32_bits=WSM6_GEOPOTENTIAL_TRANSFORM[
            "consumer_gravity_f32_bits"
        ],
        zf_raw_identity=False,
        recovery_word_count=auxiliary.recovery_word_count,
        recovery_mismatch_count=auxiliary.recovery_mismatch_count,
        recovery_recovered_count=auxiliary.recovery_recovered_count,
        recovery_maximum_ulp_distance=(auxiliary.recovery_maximum_ulp_distance),
        recovery_maximum_absolute_error_m_f64_bits=(
            auxiliary.recovery_maximum_absolute_error_m_f64_bits
        ),
        recovery_maximum_relative_error_f64_bits=(
            auxiliary.recovery_maximum_relative_error_f64_bits
        ),
        recovery_metric=auxiliary.recovery_metric,
        recovery_comparison_statement_order=(
            auxiliary.recovery_comparison_statement_order
        ),
        arena=arena,
        pinned_host=pinned,
    )


@dataclass(frozen=True, slots=True)
class ProductWsm6Restart:
    schema: str
    transition_abi: str
    source_product_manifest_sha256: str
    source_product_payload_sha256: str
    completed_transitions: int
    arrays: Mapping[str, object]
    durable_checkpoint_artifact_sha256: str | None = None
    durable_restore_manifest_sha256: str | None = None
    durable_payload_sha256: str | None = None
    _arena: object | None = None
    _pinned_host: object | None = None
    _ready_event: object | None = None


def initialize_cuda_durable_product_wsm6_restart(
    restored: LoadedDurableWsm6Restart,
    product: CudaInitializedProduct,
    geopotential: CudaWsm6Geopotential,
) -> ProductWsm6Restart:
    """Copy one Rust-restored 21-array inventory into one opaque CUDA arena."""

    if not isinstance(restored, LoadedDurableWsm6Restart):
        raise TypeError("restored must be a LoadedDurableWsm6Restart")
    if not isinstance(product, CudaInitializedProduct):
        raise TypeError("product must be a CudaInitializedProduct")
    if not isinstance(geopotential, CudaWsm6Geopotential):
        raise TypeError("geopotential must be a CudaWsm6Geopotential")
    if (
        product.schema != CUDA_INITIALIZED_PRODUCT_SCHEMA
        or not product.audited_default_driver
        or restored.source_product_manifest_sha256 != product.export.manifest_sha256
        or restored.source_product_payload_sha256 != product.export.payload_sha256
        or restored.source_geopotential_manifest_sha256 != geopotential.manifest_sha256
        or restored.source_geopotential_payload_sha256 != geopotential.payload_sha256
    ):
        raise ValueError("durable WSM6 restart does not belong to the CUDA arenas")
    cupy = importlib.import_module("cupy")
    numpy = importlib.import_module("numpy")
    current_device = int(cupy.cuda.runtime.getDevice())
    stream = cupy.cuda.get_current_stream()
    if (
        current_device != product.device_id
        or current_device != geopotential.device_id
        or int(stream.ptr) != product.stream_pointer
        or int(stream.ptr) != geopotential.stream_pointer
    ):
        raise ValueError("durable WSM6 restart requires the product-owning stream")
    arena = cupy.empty(restored.payload_nbytes, dtype=cupy.uint8)
    pinned = cupy.cuda.alloc_pinned_memory(restored.payload_nbytes)
    host = numpy.frombuffer(pinned, dtype=numpy.uint8, count=restored.payload_nbytes)
    host_bytes = memoryview(host).cast("B")
    aggregate = hashlib.sha256()
    for carrier in restored.carriers:
        digest = hashlib.sha256()
        copied = 0
        target = host_bytes[
            carrier.offset_nbytes : carrier.offset_nbytes + carrier.nbytes
        ]
        with carrier.path.open("rb", buffering=0) as handle:
            while copied < carrier.nbytes:
                count = handle.readinto(target[copied:])
                if count is None or count <= 0:
                    raise ValueError(
                        f"durable WSM6 array {carrier.name} truncated during staging"
                    )
                block = target[copied : copied + count]
                digest.update(block)
                aggregate.update(block)
                copied += count
            if handle.read(1):
                raise ValueError(
                    f"durable WSM6 array {carrier.name} grew during staging"
                )
        if digest.hexdigest() != carrier.sha256:
            raise ValueError(
                f"durable WSM6 array {carrier.name} changed after validation"
            )
    if aggregate.hexdigest() != restored.payload_sha256:
        raise ValueError("durable WSM6 aggregate changed after validation")
    arena.set(host, stream=stream)
    arrays: dict[str, object] = {}
    for carrier in restored.carriers:
        pointer = arena.data + carrier.offset_nbytes
        value = cupy.ndarray(
            carrier.shape,
            dtype=cupy.float32,
            memptr=pointer,
            order="C",
        )
        if (
            int(value.nbytes) != carrier.nbytes
            or int(value.data.ptr) != int(arena.data.ptr) + carrier.offset_nbytes
        ):
            raise ValueError(
                f"durable WSM6 CUDA view construction drifted for {carrier.name}"
            )
        arrays[carrier.name] = value
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    return ProductWsm6Restart(
        schema=INITIALIZED_PRODUCT_WSM6_RESTART_SCHEMA,
        transition_abi=INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI,
        source_product_manifest_sha256=product.export.manifest_sha256,
        source_product_payload_sha256=product.export.payload_sha256,
        completed_transitions=restored.completed_transitions,
        arrays=MappingProxyType(arrays),
        durable_checkpoint_artifact_sha256=(restored.checkpoint_artifact_sha256),
        durable_restore_manifest_sha256=restored.manifest_sha256,
        durable_payload_sha256=restored.payload_sha256,
        _arena=arena,
        _pinned_host=pinned,
        _ready_event=event,
    )


@dataclass(frozen=True, slots=True)
class ProductWsm6StepResult:
    bridge_result: Eqtset2Wsm6StepResult
    reflectivity_10cm: object | None
    ready_event: object
    completed_transitions: int
    receipt: Mapping[str, object]
    restart: ProductWsm6Restart


def _restart_shapes(ny: int, nx: int) -> dict[str, tuple[int, ...]]:
    volume = (PRODUCT_NZ, ny, nx)
    surface = (ny, nx)
    shapes = {
        "state.pp3d": volume,
        "state.th3d": volume,
        "state.prs": volume,
        "state.rho": volume,
        "state.rr": volume,
        "state.q3d": (len(SPECIES_ORDER),) + volume,
    }
    shapes.update({name: volume for name in PRODUCT_HELD_RESTART_FIELDS})
    for component_name in WSM6_COMPONENT_PERSISTENCE:
        shapes[f"component.{component_name}"] = (
            surface if component_name in WSM6_SURFACE_SCRATCH else volume
        )
    return shapes


def load_durable_product_wsm6_restart(
    directory: Path | str,
    product: LoadedInitializedProduct,
    geopotential: LoadedWsm6GeopotentialAuxiliary,
    *,
    expected_manifest_sha256: str,
    expected_checkpoint_artifact_sha256: str,
    expected_payload_sha256: str,
    expected_completed_transitions: int,
) -> LoadedDurableWsm6Restart:
    """Validate one Rust CM1W6R1 restore directory without NumPy or CUDA."""

    if not isinstance(product, LoadedInitializedProduct):
        raise TypeError("product must be a LoadedInitializedProduct")
    if not isinstance(geopotential, LoadedWsm6GeopotentialAuxiliary):
        raise TypeError("geopotential must be a LoadedWsm6GeopotentialAuxiliary")
    expected_manifest_sha256 = _sha256(
        expected_manifest_sha256, "expected durable manifest SHA-256"
    )
    expected_checkpoint_artifact_sha256 = _sha256(
        expected_checkpoint_artifact_sha256,
        "expected CM1W6R1 checkpoint artifact SHA-256",
    )
    expected_payload_sha256 = _sha256(
        expected_payload_sha256, "expected durable payload SHA-256"
    )
    if (
        isinstance(expected_completed_transitions, bool)
        or not isinstance(expected_completed_transitions, int)
        or expected_completed_transitions < 1
    ):
        raise ValueError(
            "expected completed transition count must be a positive integer"
        )
    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("durable WSM6 restore directory must not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("durable WSM6 restore path must be a real directory")
    expected_names = {"manifest.json"} | {
        f"{name}.f32le" for name in PRODUCT_WSM6_RESTART_FIELDS
    }
    if {entry.name for entry in root.iterdir()} != expected_names:
        raise ValueError("durable WSM6 restore directory inventory mismatch")
    manifest_path = root / "manifest.json"
    _regular_file(manifest_path)
    manifest_raw, manifest = _read_canonical_json(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("durable WSM6 manifest differs from independent SHA-256")
    manifest = _exact_keys(
        manifest,
        {
            "arrays",
            "capture_phase",
            "carrier_abi",
            "claim_limit",
            "completed_transitions",
            "restore_phase",
            "schema",
            "source",
            "source_checkpoint_artifact_sha256",
        },
        "durable WSM6 restore manifest",
    )
    if (
        manifest["schema"] != WSM6_DURABLE_RESTORE_MANIFEST_SCHEMA
        or manifest["carrier_abi"] != WSM6_DURABLE_CARRIER_ABI
        or manifest["claim_limit"] != WSM6_DURABLE_CLAIM_LIMIT
        or manifest["capture_phase"] != WSM6_DURABLE_CAPTURE_PHASE
        or manifest["restore_phase"] != WSM6_DURABLE_RESTORE_PHASE
        or manifest["completed_transitions"] != expected_completed_transitions
        or manifest["source_checkpoint_artifact_sha256"]
        != expected_checkpoint_artifact_sha256
    ):
        raise ValueError("durable WSM6 restore semantic contract changed")
    source = _exact_keys(
        manifest["source"],
        {
            "geopotential_manifest_sha256",
            "geopotential_payload_sha256",
            "product_manifest_sha256",
            "product_payload_sha256",
            "source_artifact_sha256",
            "source_contract_sha256",
        },
        "durable WSM6 restore source",
    )
    expected_source = {
        "geopotential_manifest_sha256": geopotential.manifest_sha256,
        "geopotential_payload_sha256": geopotential.payload_sha256,
        "product_manifest_sha256": product.manifest_sha256,
        "product_payload_sha256": product.payload_sha256,
        "source_artifact_sha256": product.source_artifact_sha256,
        "source_contract_sha256": product.source_contract_sha256,
    }
    if source != expected_source:
        raise ValueError("durable WSM6 restore belongs to different source arenas")
    values = manifest["arrays"]
    if not isinstance(values, list) or len(values) != len(PRODUCT_WSM6_RESTART_FIELDS):
        raise ValueError("durable WSM6 restore must describe exactly 21 arrays")
    ny, nx = product.crop_count_yx
    shapes = _restart_shapes(ny, nx)
    carriers: list[DurableWsm6RestartCarrier] = []
    offset = 0
    aggregate = hashlib.sha256()
    for index, (value, name, role) in enumerate(
        zip(values, PRODUCT_WSM6_RESTART_FIELDS, WSM6_DURABLE_ROLES, strict=True)
    ):
        carrier = _exact_keys(
            value,
            {
                "dtype",
                "file",
                "itemsize",
                "name",
                "nbytes",
                "order",
                "role",
                "shape",
                "sha256",
                "species_order",
                "statistics",
            },
            f"durable WSM6 array {index}",
        )
        expected_shape = shapes[name]
        expected_nbytes = math.prod(expected_shape) * 4
        expected_species = list(SPECIES_ORDER) if name == "state.q3d" else []
        expected_file = f"{name}.f32le"
        if (
            carrier["dtype"] != "<f4"
            or carrier["itemsize"] != 4
            or carrier["file"] != expected_file
            or carrier["name"] != name
            or carrier["nbytes"] != expected_nbytes
            or carrier["order"] != "C"
            or carrier["role"] != role
            or carrier["shape"] != list(expected_shape)
            or carrier["species_order"] != expected_species
        ):
            raise ValueError(f"durable WSM6 array {name} ABI changed")
        carrier_sha256 = _sha256(
            carrier["sha256"], f"durable WSM6 array {name} SHA-256"
        )
        statistics = _exact_keys(
            carrier["statistics"],
            {
                "finite_count",
                "maximum_f32_bits",
                "minimum_f32_bits",
                "nonzero_count",
            },
            f"durable WSM6 array {name} statistics",
        )
        words = expected_nbytes // 4
        nonzero = _strict_nonnegative_int(
            statistics["nonzero_count"], f"{name} nonzero_count"
        )
        if statistics["finite_count"] != words or nonzero > words:
            raise ValueError(f"durable WSM6 array {name} statistics changed")
        for key in ("minimum_f32_bits", "maximum_f32_bits"):
            if (
                not isinstance(statistics[key], str)
                or _HEX8.fullmatch(statistics[key]) is None
            ):
                raise ValueError(f"durable WSM6 array {name} has invalid {key}")
        path = root / expected_file
        _regular_file(path)
        if os.lstat(path).st_size != expected_nbytes:
            raise ValueError(f"durable WSM6 array {name} byte count changed")
        digest = hashlib.sha256()
        with path.open("rb", buffering=0) as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                aggregate.update(block)
        if digest.hexdigest() != carrier_sha256:
            raise ValueError(f"durable WSM6 array {name} bytes changed")
        carriers.append(
            DurableWsm6RestartCarrier(
                name=name,
                role=role,
                path=path,
                shape=expected_shape,
                offset_nbytes=offset,
                nbytes=expected_nbytes,
                sha256=carrier_sha256,
            )
        )
        offset += expected_nbytes
    if aggregate.hexdigest() != expected_payload_sha256:
        raise ValueError("durable WSM6 restored aggregate payload SHA-256 changed")
    return LoadedDurableWsm6Restart(
        directory=root,
        manifest_sha256=manifest_sha256,
        manifest_nbytes=len(manifest_raw),
        checkpoint_artifact_sha256=expected_checkpoint_artifact_sha256,
        payload_sha256=expected_payload_sha256,
        payload_nbytes=offset,
        completed_transitions=expected_completed_transitions,
        source_product_manifest_sha256=product.manifest_sha256,
        source_product_payload_sha256=product.payload_sha256,
        source_geopotential_manifest_sha256=geopotential.manifest_sha256,
        source_geopotential_payload_sha256=geopotential.payload_sha256,
        source_artifact_sha256=product.source_artifact_sha256,
        source_contract_sha256=product.source_contract_sha256,
        carriers=tuple(carriers),
    )


def _validate_restart_without_cuda(
    restart: ProductWsm6Restart, *, ny: int, nx: int, product: CudaInitializedProduct
) -> None:
    if not isinstance(restart, ProductWsm6Restart):
        _refuse(
            code="CM1-PRODUCT-WSM6-RESTART-TYPE",
            reason="restart input is not the versioned transition restart object",
            remedy="load or capture a ProductWsm6Restart",
            option="restart",
            value=type(restart).__name__,
        )
    if (
        restart.schema != INITIALIZED_PRODUCT_WSM6_RESTART_SCHEMA
        or restart.transition_abi != INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI
    ):
        _refuse(
            code="CM1-PRODUCT-WSM6-RESTART-SCHEMA",
            reason="restart schema or transition ABI differs",
            remedy="migrate the restart with a measured versioned reader",
            option="restart.schema",
            value=[restart.schema, restart.transition_abi],
        )
    if (
        isinstance(restart.completed_transitions, bool)
        or not isinstance(restart.completed_transitions, int)
        or restart.completed_transitions < 0
    ):
        _refuse(
            code="CM1-PRODUCT-WSM6-RESTART-CADENCE",
            reason="completed transition count is not a strict nonnegative integer",
            remedy="restore the exact completed transition ordinal",
            option="restart.completed_transitions",
            value=restart.completed_transitions,
        )
    if (
        restart.source_product_manifest_sha256 != product.export.manifest_sha256
        or restart.source_product_payload_sha256 != product.export.payload_sha256
    ):
        _refuse(
            code="CM1-PRODUCT-WSM6-RESTART-SOURCE",
            reason="restart state belongs to a different initialized-product arena",
            remedy="resume against the exact hash-bound initialized product",
            option="restart.source_product",
            value={
                "manifest": restart.source_product_manifest_sha256,
                "payload": restart.source_product_payload_sha256,
            },
        )
    names = tuple(restart.arrays)
    if names != PRODUCT_WSM6_RESTART_FIELDS:
        _refuse(
            code="CM1-PRODUCT-WSM6-RESTART-INVENTORY",
            reason="transition restart names or order differ from the v1 ABI",
            remedy="restore all state, held, and component arrays in ABI order",
            option="restart.arrays",
            value=list(names),
        )
    shapes = _restart_shapes(ny, nx)
    for name in names:
        value = restart.arrays[name]
        if _shape(value) != shapes[name]:
            _refuse(
                code="CM1-PRODUCT-WSM6-RESTART-SHAPE",
                reason=f"restart field {name} has the wrong product-domain shape",
                remedy="restore an all-level, exact horizontal-domain array",
                option=f"restart.arrays.{name}.shape",
                value=_shape(value),
            )
        if not _is_f32(value) or not _is_c_contiguous(value):
            _refuse(
                code="CM1-PRODUCT-WSM6-RESTART-LAYOUT",
                reason=f"restart field {name} is not C-contiguous binary32",
                remedy="load exact raw binary32 words without conversion",
                option=f"restart.arrays.{name}.layout",
                value=str(getattr(value, "dtype", None)),
            )
    if len({id(restart.arrays[name]) for name in names}) != len(names):
        _refuse(
            code="CM1-PRODUCT-WSM6-RESTART-ALIAS",
            reason="two restart inventory entries share one array object",
            remedy="restore distinct state and persistence storage",
            option="restart.arrays.identity",
            value="aliased",
        )


class InitializedProductWsm6Transition:
    """Persistent post-final-RK transition bound to one corrected CUDA arena."""

    def __init__(
        self,
        *,
        product: CudaInitializedProduct,
        geopotential: CudaWsm6Geopotential,
        bridge: Eqtset2Wsm6Bridge,
        carriers: Eqtset2Wsm6Carriers,
        php_zero: object,
        completed_transitions: int,
        allow_cropped_validation: bool,
        pending_restart: ProductWsm6Restart | None,
    ) -> None:
        self.product = product
        self.geopotential = geopotential
        self.bridge = bridge
        self.carriers = carriers
        self.php_zero = php_zero
        self.completed_transitions = completed_transitions
        self.allow_cropped_validation = allow_cropped_validation
        self._pending_restart = pending_restart
        self._durable_restore = (
            None
            if pending_restart is None
            or pending_restart.durable_checkpoint_artifact_sha256 is None
            else {
                "checkpoint_artifact_sha256": (
                    pending_restart.durable_checkpoint_artifact_sha256
                ),
                "restore_manifest_sha256": (
                    pending_restart.durable_restore_manifest_sha256
                ),
                "payload_sha256": pending_restart.durable_payload_sha256,
                "completed_transitions": pending_restart.completed_transitions,
            }
        )
        self._last_receipt: dict[str, object] | None = None

    def _restart_arrays(self) -> dict[str, object]:
        result = {
            "state.pp3d": self.product.field("pp3d"),
            "state.th3d": self.product.field("th3d"),
            "state.prs": self.product.field("prs"),
            "state.rho": self.product.field("rho"),
            "state.rr": self.product.field("rr"),
            "state.q3d": self.product.field("q3d"),
        }
        result.update(
            {
                f"held.{name}": getattr(self.carriers, name)
                for name in EQTSET2_HELD_FIELDS
            }
        )
        component = self.bridge.component.persistent_arrays()
        if not component and self._pending_restart is not None:
            component = MappingProxyType(
                {
                    name: self._pending_restart.arrays[f"component.{name}"]
                    for name in WSM6_COMPONENT_PERSISTENCE
                }
            )
        result.update(
            {
                f"component.{name}": component[name]
                for name in WSM6_COMPONENT_PERSISTENCE
            }
        )
        if tuple(result) != PRODUCT_WSM6_RESTART_FIELDS:
            raise RuntimeError("product WSM6 restart inventory drifted")
        return result

    def capture_restart(self) -> ProductWsm6Restart:
        """Expose every transition-owned restart array without copying."""

        return ProductWsm6Restart(
            schema=INITIALIZED_PRODUCT_WSM6_RESTART_SCHEMA,
            transition_abi=INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI,
            source_product_manifest_sha256=self.product.export.manifest_sha256,
            source_product_payload_sha256=self.product.export.payload_sha256,
            completed_transitions=self.completed_transitions,
            arrays=MappingProxyType(self._restart_arrays()),
        )

    def clone_restart(self) -> ProductWsm6Restart:
        """Clone exact device words into independently owned restart storage."""

        cupy = importlib.import_module("cupy")
        arrays = {
            name: cupy.array(value, dtype=cupy.float32, order="C", copy=True)
            for name, value in self._restart_arrays().items()
        }
        return ProductWsm6Restart(
            schema=INITIALIZED_PRODUCT_WSM6_RESTART_SCHEMA,
            transition_abi=INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI,
            source_product_manifest_sha256=self.product.export.manifest_sha256,
            source_product_payload_sha256=self.product.export.payload_sha256,
            completed_transitions=self.completed_transitions,
            arrays=MappingProxyType(arrays),
        )

    def receipt_dict(self) -> dict[str, object]:
        if self._last_receipt is None:
            raise RuntimeError("product WSM6 transition has not completed an apply")
        return json.loads(_canonical_json_bytes(self._last_receipt))

    def receipt_bytes(self) -> bytes:
        return _canonical_json_bytes(self.receipt_dict())

    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.receipt_bytes()).hexdigest()

    def apply(
        self,
        *,
        final_rk: bool = True,
        refl_10cm_due: bool = False,
        pre_component_hook: Callable[[object], object] | None = None,
    ) -> ProductWsm6StepResult:
        bridge_kwargs: dict[str, object] = {
            "final_rk": final_rk,
            "refl_10cm_due": refl_10cm_due,
        }
        if pre_component_hook is not None:
            bridge_kwargs["pre_component_hook"] = pre_component_hook
        result = self.bridge.apply(self.carriers, **bridge_kwargs)
        self._pending_restart = None
        self.completed_transitions += 1
        restart = self.capture_restart()
        if result.component_result.call_index != self.completed_transitions:
            raise RuntimeError("restored WSM6 call cadence drifted")
        if result.call_index != self.completed_transitions:
            raise RuntimeError("restored eqtset2 bridge cadence drifted")
        ready = result.ready_event
        receipt: dict[str, object] = {
            "schema": INITIALIZED_PRODUCT_WSM6_RECEIPT_SCHEMA,
            "transition_abi": INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI,
            "claim_limit": _CLAIM_LIMIT,
            "completed_transitions": self.completed_transitions,
            "configuration": {
                "nx": self.product.export.crop_count_yx[1],
                "ny": self.product.export.crop_count_yx[0],
                "nz": PRODUCT_NZ,
                "dx_m": PRODUCT_DX_M,
                "dy_m": PRODUCT_DY_M,
                "dz_m": PRODUCT_DZ_M,
                "dt_s": PRODUCT_DT_S,
                "equation_set": 2,
                "native_ptype_context": 5,
                "semantic_component_id": ARWEN_WSM6_COMPONENT_ID,
                "species_order": list(SPECIES_ORDER),
                "refl_10cm_due": refl_10cm_due,
            },
            "source_product": {
                "manifest_sha256": self.product.export.manifest_sha256,
                "payload_sha256": self.product.export.payload_sha256,
                "carrier_abi_count": len(CARRIER_NAMES),
                "crop_origin_yx": list(self.product.export.crop_origin_yx),
                "crop_count_yx": list(self.product.export.crop_count_yx),
                "full_domain": self.product.export.is_full_domain,
            },
            "geopotential": {
                "schema": self.geopotential.schema,
                "carrier_name": self.geopotential.carrier_name,
                "manifest_sha256": self.geopotential.manifest_sha256,
                "payload_sha256": self.geopotential.payload_sha256,
                "receipt_sha256": self.geopotential.receipt_sha256,
                "consumer_gravity_f32_bits": (
                    self.geopotential.consumer_gravity_f32_bits
                ),
                "zf_raw_identity": self.geopotential.zf_raw_identity,
                "recovery": {
                    "word_count": self.geopotential.recovery_word_count,
                    "mismatch_count": self.geopotential.recovery_mismatch_count,
                    "recovered_count": self.geopotential.recovery_recovered_count,
                    "maximum_ulp_distance": (
                        self.geopotential.recovery_maximum_ulp_distance
                    ),
                    "maximum_absolute_error_m_f64_bits": (
                        self.geopotential.recovery_maximum_absolute_error_m_f64_bits
                    ),
                    "maximum_relative_error_f64_bits": (
                        self.geopotential.recovery_maximum_relative_error_f64_bits
                    ),
                    "metric": self.geopotential.recovery_metric,
                    "comparison_statement_order": (
                        self.geopotential.recovery_comparison_statement_order
                    ),
                },
                "binding": "total geopotential as phb; exact zero php",
            },
            "binding": {
                "python_science_transforms": False,
                "pi0_th0": "native one-dimensional profile views",
                "pressure": "prs",
                "density": "rho",
                "reciprocal_density": "rr",
                "packed_species": "q3d[0:6] in frozen order",
                "mutated_in_place": ["pp3d", "th3d", "prs", "rho", "rr", "q3d"],
                "pre_component_hook": pre_component_hook is not None,
                "a_time_level_untouched": ["ppi", "tha", "qa"],
                "reflectivity": {
                    "consumer_species": ["qv", "qr", "qs", "qg"],
                    "excluded_species_by_wsm6_authority": ["qc", "qi"],
                    "six_species_order": list(SPECIES_ORDER),
                    "transient_not_restart_state": True,
                },
            },
            "restart": {
                "schema": restart.schema,
                "inventory": list(restart.arrays),
                "array_count": len(restart.arrays),
                "capture_zero_copy": True,
                "restore_before_physics_without_replay": True,
                "component_persistence_zero_copy": True,
                "durable_serialization": self._durable_restore is not None,
                "durable_restore": self._durable_restore,
            },
            "bridge": {
                "receipt_sha256": self.bridge.receipt_sha256(),
                "receipt": self.bridge.receipt_dict(),
            },
            "synchronization": {
                "device_id": self.product.device_id,
                "stream_pointer": self.product.stream_pointer,
                "product_ready_precedes_transition": True,
                "geopotential_ready_precedes_transition": True,
                "ready_event_recorded": True,
                "explicit_synchronize": False,
                "host_state_copy": False,
            },
            "red_gates": {
                "idiss": True,
                "efall": True,
                "reflectivity_consumer": False,
                "durable_transition_restart_serialization": (
                    self._durable_restore is None
                ),
                "halos": True,
                "coupled_dycore_trajectory": True,
                "forecast": True,
                "tornado": True,
                "model_validation": True,
            },
        }
        _canonical_json_bytes(receipt)
        self._last_receipt = receipt
        return ProductWsm6StepResult(
            bridge_result=result,
            reflectivity_10cm=result.component_result.reflectivity_10cm,
            ready_event=ready,
            completed_transitions=self.completed_transitions,
            receipt=MappingProxyType(self.receipt_dict()),
            restart=restart,
        )


def _validate_product_and_geopotential(
    product: CudaInitializedProduct,
    geopotential: CudaWsm6Geopotential,
    *,
    allow_cropped_validation: bool,
) -> tuple[int, int]:
    if not isinstance(product, CudaInitializedProduct):
        raise TypeError("product must be a CudaInitializedProduct")
    if type(allow_cropped_validation) is not bool:
        raise TypeError("allow_cropped_validation must be a strict boolean")
    if (
        product.schema != CUDA_INITIALIZED_PRODUCT_SCHEMA
        or not product.audited_default_driver
    ):
        _refuse(
            code="CM1-PRODUCT-WSM6-UNATTESTED-ARENA",
            reason="transition requires the audited production CUDA product loader",
            remedy="load the strict corrected-v3 export with the default driver",
            option="product.schema",
            value=product.schema,
        )
    if not product.export.is_full_domain and not allow_cropped_validation:
        _refuse(
            code="CM1-PRODUCT-WSM6-CROP-VALIDATION-ONLY",
            reason="a horizontal crop is controlled validation, not product execution",
            remedy="set allow_cropped_validation=true only for bounded evidence",
            option="product.crop_count_yx",
            value=list(product.export.crop_count_yx),
        )
    if tuple(name for name, _ in product.fields) != CARRIER_NAMES:
        raise RuntimeError("initialized-product CUDA carrier inventory drifted")
    ny, nx = product.export.crop_count_yx
    volume = (PRODUCT_NZ, ny, nx)
    expected_shapes = {
        "pi0": (PRODUCT_NZ,),
        "th0": (PRODUCT_NZ,),
        "pp3d": volume,
        "th3d": volume,
        "q3d": (len(SPECIES_ORDER),) + volume,
        "prs": volume,
        "rho": volume,
        "rr": volume,
    }
    actual_shapes = {name: _shape(product.field(name)) for name in expected_shapes}
    if actual_shapes != expected_shapes:
        _refuse(
            code="CM1-PRODUCT-WSM6-GEOMETRY",
            reason=(
                "initialized product does not expose the sealed nz=75 "
                "pressure-density/species ABI"
            ),
            remedy="load the corrected-v3 75-level initialized product",
            option="product.geometry",
            value=actual_shapes,
        )
    for name, (role, sources) in _CONSUMED_PRODUCT_CONTRACT.items():
        spec = product.export.carrier(name)
        if spec.role != role or spec.source_arrays != sources:
            _refuse(
                code="CM1-PRODUCT-WSM6-CARRIER-SEMANTICS",
                reason=f"initialized-product carrier {name} changed semantic provenance",
                remedy="load the frozen corrected-v3 carrier ABI",
                option=f"product.carriers.{name}",
                value={"role": spec.role, "source_arrays": list(spec.source_arrays)},
            )
        value = product.field(name)
        if not _is_f32(value) or not _is_c_contiguous(value):
            _refuse(
                code="CM1-PRODUCT-WSM6-CARRIER-LAYOUT",
                reason=f"initialized-product carrier {name} is not C-contiguous binary32",
                remedy="retain the exact CUDA view created by the audited product loader",
                option=f"product.fields.{name}.layout",
                value=str(getattr(value, "dtype", None)),
            )
        owner = getattr(getattr(value, "device", None), "id", None)
        if owner is None or int(owner) != product.device_id:
            _refuse(
                code="CM1-PRODUCT-WSM6-CARRIER-DEVICE",
                reason=f"initialized-product carrier {name} is not on the arena device",
                remedy="bind the unmodified CUDA arena view",
                option=f"product.fields.{name}.device",
                value=owner,
            )
    mutable = tuple(
        product.field(name) for name in ("pp3d", "th3d", "q3d", "prs", "rho", "rr")
    )
    if len({id(value) for value in mutable}) != len(mutable):
        _refuse(
            code="CM1-PRODUCT-WSM6-CARRIER-ALIAS",
            reason="two mutable product transition carriers share one array object",
            remedy="bind the distinct contiguous offsets from the Rust arena",
            option="product.mutable_identity",
            value="aliased",
        )
    if tuple(SPECIES_ORDER) != tuple(WSM6_SPECIES):
        raise RuntimeError("initialized-product and public WSM6 species orders differ")
    if not isinstance(geopotential, CudaWsm6Geopotential):
        raise TypeError("geopotential must come from the strict Rust auxiliary loader")
    if (
        geopotential.source_product_manifest_sha256 != product.export.manifest_sha256
        or geopotential.source_product_payload_sha256 != product.export.payload_sha256
    ):
        _refuse(
            code="CM1-PRODUCT-WSM6-GEOPOTENTIAL-SOURCE",
            reason="geopotential was derived from a different initialized product",
            remedy="load the hash-bound Rust auxiliary for this exact arena",
            option="geopotential.source_product",
            value={
                "manifest": geopotential.source_product_manifest_sha256,
                "payload": geopotential.source_product_payload_sha256,
            },
        )
    word_count = (PRODUCT_NZ + 1) * ny * nx
    maximum_absolute_error = struct.unpack(
        ">d", bytes.fromhex(geopotential.recovery_maximum_absolute_error_m_f64_bits)
    )[0]
    maximum_relative_error = struct.unpack(
        ">d", bytes.fromhex(geopotential.recovery_maximum_relative_error_f64_bits)
    )[0]
    if (
        geopotential.zf_raw_identity
        or geopotential.consumer_gravity_f32_bits != "411cf5c3"
        or geopotential.recovery_word_count != word_count
        or not 0 <= geopotential.recovery_mismatch_count <= word_count
        or geopotential.recovery_recovered_count + geopotential.recovery_mismatch_count
        != word_count
        or not 0 <= geopotential.recovery_maximum_ulp_distance <= 1
        or not math.isfinite(maximum_absolute_error)
        or maximum_absolute_error < 0.0
        or not math.isfinite(maximum_relative_error)
        or maximum_relative_error < 0.0
    ):
        _refuse(
            code="CM1-PRODUCT-WSM6-GEOPOTENTIAL-EQUATION",
            reason=(
                "Rust auxiliary does not carry a valid bounded zf recovery "
                "contract under consumer gravity"
            ),
            remedy="regenerate the versioned source-ordered geopotential auxiliary",
            option="geopotential.contract",
            value={
                "consumer_gravity_f32_bits": geopotential.consumer_gravity_f32_bits,
                "zf_raw_identity": geopotential.zf_raw_identity,
                "word_count": geopotential.recovery_word_count,
                "mismatch_count": geopotential.recovery_mismatch_count,
                "recovered_count": geopotential.recovery_recovered_count,
                "maximum_ulp_distance": geopotential.recovery_maximum_ulp_distance,
            },
        )
    if _shape(geopotential.value) != (PRODUCT_NZ + 1, ny, nx):
        _refuse(
            code="CM1-PRODUCT-WSM6-GEOPOTENTIAL-SHAPE",
            reason="total geopotential does not cover every product z face",
            remedy="load an all-level auxiliary for the identical horizontal domain",
            option="geopotential.shape",
            value=_shape(geopotential.value),
        )
    if not _is_f32(geopotential.value) or not _is_c_contiguous(geopotential.value):
        _refuse(
            code="CM1-PRODUCT-WSM6-GEOPOTENTIAL-LAYOUT",
            reason="total geopotential is not C-contiguous binary32",
            remedy="use the strict Rust auxiliary CUDA view without conversion",
            option="geopotential.layout",
            value=str(getattr(geopotential.value, "dtype", None)),
        )
    geopotential_owner = getattr(
        getattr(geopotential.value, "device", None), "id", None
    )
    if geopotential_owner is None or int(geopotential_owner) != product.device_id:
        _refuse(
            code="CM1-PRODUCT-WSM6-GEOPOTENTIAL-DEVICE",
            reason="total geopotential CUDA view is not on the initialized arena device",
            remedy="load the auxiliary on the product-owning device",
            option="geopotential.value.device",
            value=geopotential_owner,
        )
    if (
        geopotential.device_id != product.device_id
        or geopotential.stream_pointer != product.stream_pointer
    ):
        _refuse(
            code="CM1-PRODUCT-WSM6-GEOPOTENTIAL-ORDERING",
            reason="product and geopotential are not resident on one CUDA device/stream",
            remedy="load both arenas on the selected transition stream",
            option="geopotential.device_stream",
            value=[geopotential.device_id, geopotential.stream_pointer],
        )
    return ny, nx


def initialize_product_wsm6_transition(
    product: CudaInitializedProduct,
    geopotential: CudaWsm6Geopotential,
    *,
    allow_cropped_validation: bool = False,
    restart: ProductWsm6Restart | None = None,
) -> InitializedProductWsm6Transition:
    """Bind a corrected product to the source-ordered public WSM6 transition."""

    ny, nx = _validate_product_and_geopotential(
        product,
        geopotential,
        allow_cropped_validation=allow_cropped_validation,
    )
    if restart is not None:
        _validate_restart_without_cuda(restart, ny=ny, nx=nx, product=product)

    cupy = importlib.import_module("cupy")
    current_device = int(cupy.cuda.runtime.getDevice())
    current_stream = cupy.cuda.get_current_stream()
    if (
        current_device != product.device_id
        or int(current_stream.ptr) != product.stream_pointer
    ):
        _refuse(
            code="CM1-PRODUCT-WSM6-CURRENT-STREAM",
            reason="current CUDA device/stream differs from the initialized arena owner",
            remedy="make the product-owning device and stream current",
            option="current_device_stream",
            value=[current_device, int(current_stream.ptr)],
        )

    q3d = product.field("q3d")
    species = {name: q3d[index] for index, name in enumerate(SPECIES_ORDER)}
    if any(not _is_c_contiguous(value) for value in species.values()):
        raise RuntimeError("packed-species views are not C-contiguous")
    php_zero = cupy.zeros_like(geopotential.value)

    if restart is None:
        held = {
            "qpten": cupy.zeros_like(product.field("pp3d")),
            "qtten": cupy.zeros_like(product.field("th3d")),
            "qvten": cupy.zeros_like(species["qv"]),
            "qcten": cupy.zeros_like(species["qc"]),
        }
        completed = 0
    else:
        state_destinations = {
            "state.pp3d": product.field("pp3d"),
            "state.th3d": product.field("th3d"),
            "state.prs": product.field("prs"),
            "state.rho": product.field("rho"),
            "state.rr": product.field("rr"),
            "state.q3d": product.field("q3d"),
        }
        for name, destination in state_destinations.items():
            cupy.copyto(destination, restart.arrays[name])
        held = {name: restart.arrays[f"held.{name}"] for name in EQTSET2_HELD_FIELDS}
        completed = restart.completed_transitions

    carriers = Eqtset2Wsm6Carriers(
        pi0=product.field("pi0"),
        th0=product.field("th0"),
        rho=product.field("rho"),
        pp3d=product.field("pp3d"),
        th3d=product.field("th3d"),
        p=product.field("prs"),
        alt=product.field("rr"),
        phb=geopotential.value,
        php=php_zero,
        qv=species["qv"],
        qc=species["qc"],
        qr=species["qr"],
        qi=species["qi"],
        qs=species["qs"],
        qg=species["qg"],
        qpten=held["qpten"],
        qtten=held["qtten"],
        qvten=held["qvten"],
        qcten=held["qcten"],
    )
    component = create_arwen_wsm6_component(
        component_id=ARWEN_WSM6_COMPONENT_ID,
        nx=nx,
        ny=ny,
        nz=PRODUCT_NZ,
        dx_m=PRODUCT_DX_M,
        dy_m=PRODUCT_DY_M,
        dz_m=PRODUCT_DZ_M,
        dt_s=PRODUCT_DT_S,
    )
    bridge = Eqtset2Wsm6Bridge(
        component=component,
        nx=nx,
        ny=ny,
        nz=PRODUCT_NZ,
        dt_s=PRODUCT_DT_S,
        idiss=False,
        efall=False,
        product_admission=False,
    )
    if restart is not None:
        bridge_restart = {
            **{name: restart.arrays[f"held.{name}"] for name in EQTSET2_HELD_FIELDS},
            **{
                f"component.{name}": restart.arrays[f"component.{name}"]
                for name in WSM6_COMPONENT_PERSISTENCE
            },
        }
        bridge.restore_restart_arrays(bridge_restart, completed_calls=completed)
    return InitializedProductWsm6Transition(
        product=product,
        geopotential=geopotential,
        bridge=bridge,
        carriers=carriers,
        php_zero=php_zero,
        completed_transitions=completed,
        allow_cropped_validation=allow_cropped_validation,
        pending_restart=restart,
    )


__all__ = [
    "CudaWsm6Geopotential",
    "DurableWsm6RestartCarrier",
    "INITIALIZED_PRODUCT_WSM6_RECEIPT_SCHEMA",
    "INITIALIZED_PRODUCT_WSM6_RESTART_SCHEMA",
    "INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI",
    "InitializedProductWsm6Transition",
    "LoadedDurableWsm6Restart",
    "LoadedWsm6GeopotentialAuxiliary",
    "PRODUCT_COMPONENT_RESTART_FIELDS",
    "PRODUCT_HELD_RESTART_FIELDS",
    "PRODUCT_STATE_RESTART_FIELDS",
    "PRODUCT_WSM6_RESTART_FIELDS",
    "ProductWsm6Restart",
    "ProductWsm6StepResult",
    "WSM6_DURABLE_CARRIER_ABI",
    "WSM6_DURABLE_CHECKPOINT_RECEIPT_SCHEMA",
    "WSM6_DURABLE_CHECKPOINT_SCHEMA",
    "WSM6_DURABLE_RESTORE_MANIFEST_SCHEMA",
    "WSM6_DURABLE_RESTORE_RECEIPT_SCHEMA",
    "initialize_cuda_durable_product_wsm6_restart",
    "initialize_cuda_wsm6_geopotential",
    "initialize_product_wsm6_transition",
    "load_durable_product_wsm6_restart",
    "load_wsm6_geopotential_auxiliary",
]
