"""Strict opaque transport for the Rust-owned initialized CM1 product arena.

Rust owns terrain construction and every initialization equation.  This module
does not decode or recompute any scientific field: it validates the canonical
hash-bound export, copies its one byte arena to CUDA, and exposes the already
described C-order float32 views.  NumPy and CuPy remain unimported until the
optional CUDA allocation entry point is called.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


INITIALIZED_PRODUCT_EXPORT_SCHEMA = "gpuwm-cm1.cuda-initialized-product-export/v1"
INITIALIZED_PRODUCT_RECEIPT_SCHEMA = (
    "gpuwm-cm1.cuda-initialized-product-export-receipt/v1"
)
INITIALIZED_PRODUCT_CARRIER_ABI = "gpuwm-cm1.cuda-initialized-product-carriers/v1"
CUDA_INITIALIZED_PRODUCT_SCHEMA = "gpuwm-cm1.cuda-initialized-product/v1"
CUDA_INITIALIZED_PRODUCT_TEST_SCHEMA = (
    "gpuwm-cm1.cuda-initialized-product-test-double/v1"
)
PAYLOAD_NAME = "initialized-carriers.f32le"
MANIFEST_NAME = "manifest.json"
RECEIPT_NAME = "receipt.json"
ITEMSIZE = 4
SOURCE_PROFILE_ID = "hrrr-cm1-real-state-v1"
REAL_SOURCE_SHAPE_ZYX = (75, 300, 300)
REAL_SPACING_BITS = {
    "target_dx_m_f64_bits": "4059000000000000",
    "target_dy_m_f64_bits": "4059000000000000",
    "target_dz_m_f64_bits": "4069000000000000",
}
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_FILES = (
    {
        "anchors": "5990-6037,9255-9322",
        "path": "src/adv_routines.F",
        "sha256": "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00",
    },
    {
        "anchors": "2160-2190",
        "path": "src/base.F",
        "sha256": "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
    },
    {
        "anchors": "12-172",
        "path": "src/bc.F",
        "sha256": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
    },
    {
        "anchors": "7-18,69-140",
        "path": "src/constants.F",
        "sha256": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
    },
    {
        "anchors": "1733-1785,2365-2467",
        "path": "src/init3d.F",
        "sha256": "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
    },
    {
        "anchors": "175-297",
        "path": "src/init_terrain.F",
        "sha256": "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c",
    },
    {
        "anchors": "6325-6327,7789-7812,8162-8207,8464-8492,8603-8627,8988-9062",
        "path": "src/param.F",
        "sha256": "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
    },
)
PUBLICATION_CONTRACT = (
    "write the binary32 payload, canonical manifest, and canonical receipt into "
    "a new sibling staging directory; synchronize every file; rename the "
    "directory only after all hashes validate; refuse an existing destination"
)
BASE_POLICY = (
    "exact horizontal broadcast of the checkpoint rho0 profile followed by "
    "source-ordered rr0, c1/c2 face interpolation, and bottom/top Fornberg "
    "extrapolation; replaceable by a future source-backed terrain-varying "
    "hydrostatic base producer"
)
EXPORT_CLAIM_LIMIT = (
    "byte-exact Rust-owned CM1 r21.1 physical-domain initialization carriers "
    "from an independently hashed real checkpoint; includes terrain coordinates/"
    "static metrics and audited ibalance=1 moist pressure/density completion; "
    "rho0_3d/rr0/rf0/rrf0 retain the explicit replaceable horizontal-broadcast "
    "base policy; no halos, open-boundary history or tendencies, closure state, "
    "pressure operator, dynamics, physics, trajectory, forecast, tornado, or "
    "model-validation claim"
)
EXPORT_FORECAST_STATE_CLAIM_LIMIT = (
    "byte-exact Rust-owned CM1 r21.1 physical-domain initialization carriers "
    "from an independently hashed real forecast state; includes terrain coordinates/"
    "static metrics and audited ibalance=1 moist pressure/density completion; "
    "rho0_3d/rr0/rf0/rrf0 retain the explicit replaceable horizontal-broadcast "
    "base policy; no halos, tendencies, closure state, pressure operator, dynamics, "
    "physics, trajectory, forecast, tornado, or model-validation claim"
)
EXPORT_CLAIM_LIMITS = frozenset((EXPORT_CLAIM_LIMIT, EXPORT_FORECAST_STATE_CLAIM_LIMIT))
CUDA_CLAIM_LIMIT = (
    "byte-exact allocation/copy/readiness of the independently validated Rust-owned "
    "initialized carrier arena; no Python scientific transform, halos, dynamics, "
    "physics, trajectory, forecast, tornado, or model-validation claim"
)
CUDA_TEST_CLAIM_LIMIT = (
    "dependency-injected opaque byte-transport orchestration test only; no CUDA "
    "residency, stream ordering, hardware, dynamics, or forecast claim"
)
CROP_STAGGER_RULE = (
    "all vertical levels retained; horizontal mass crop retains one terminal face "
    "on each u/v axis and all rigid-boundary z faces"
)
FORNBERG_SOURCE_CONTRACT = (
    "param.F:8988-9052 DOUBLE PRECISION recurrence; default-REAL normalized "
    "coordinates and default-REAL stored weights"
)
FORMULA_CONTRACTS = {
    "base_reciprocals": "base.F:2160-2190",
    "horizontal_gradients": (
        "adv_routines.F:9255-9322 with flx6 at 6026-6037 and bc2d open-edge halos"
    ),
    "moist_initialization": (
        "init3d.F:1733-1785,2365-2467 with param.F:8988-9052 Fornberg coefficients"
    ),
    "terrain_coordinates": "init_terrain.F:175-297",
    "vertical_metrics": "param.F:8464-8492,8603-8627",
}
CONFIGURATION = {
    "hadvordrs": 5,
    "hadvordrv": 5,
    "ibalance": 1,
    "iice": 1,
    "lateral_bc_west_east_south_north": [2, 2, 2, 2],
    "psolver": 3,
    "setppi": True,
}
SPECIES_ORDER = ("qv", "qc", "qr", "qi", "qs", "qg")
CONSTANT_BITS = {
    "cp": "447b6ccd",
    "g": "411cf5c3",
    "p00": "47c35000",
    "rd": "438f851f",
    "rv": "43e6c000",
}
CORRECTED_CARRIERS = ("ppi", "pp3d", "ppx", "prs", "rho", "rr", "rf")
DUPLICATE_GROUPS = (
    ("ua", "u3d"),
    ("va", "v3d"),
    ("wa", "w3d"),
    ("ppi", "pp3d", "ppx"),
    ("tha", "th3d"),
    ("qa", "q3d"),
)
SOURCE_PRESERVED_CARRIERS = (
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
    "tha",
    "th3d",
    "qa",
    "q3d",
    "terrain_msl",
)
PPI_SMALL_VALUE_RULE = (
    "if abs(ppi) < 1.0e-6 then assign positive binary32 zero before duplicating "
    "ppi into pp3d and ppx"
)
QV_DENSITY_RULE = (
    "thv uses source qv unchanged; only the density gas-constant factor uses "
    "max(0.0,qv)"
)
STATEMENT_ORDER = (
    "init3d.F:1733-1785 then 2365-2467 in source loop and assignment order"
)
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
    "zh",
    "zf",
    "mh",
    "rmh",
    "mf",
    "rmf",
    "c1",
    "c2",
    "rho0_3d",
    "rr0",
    "rf0",
    "rrf0",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
    "gx",
    "gxu",
    "gy",
    "gyv",
    "prs",
    "rho",
    "rr",
    "rf",
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")
_HEX8 = re.compile(r"[0-9a-f]{8}\Z")
_DEFAULT_DRIVER_ATTESTATION = object()


def _require_sha256(value: object, label: str) -> str:
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
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    if raw != _canonical_json_bytes(value):
        raise ValueError(f"{path.name} is not canonical compact sorted-key JSON")
    return raw, value


def _exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} key inventory mismatch; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _regular_file(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"required export file is missing: {path.name}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(
            f"export entry must be a regular non-symlink file: {path.name}"
        )


@dataclass(frozen=True, slots=True)
class InitializedCarrierSpec:
    name: str
    shape: tuple[int, ...]
    offset_nbytes: int
    nbytes: int
    sha256: str
    role: str
    source_arrays: tuple[str, ...]
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
class LoadedInitializedProduct:
    directory: Path
    payload_path: Path
    manifest_sha256: str
    manifest_nbytes: int
    payload_sha256: str
    payload_nbytes: int
    source_artifact_sha256: str
    source_contract_sha256: str
    crop_origin_yx: tuple[int, int]
    crop_count_yx: tuple[int, int]
    carriers: tuple[InitializedCarrierSpec, ...]

    def carrier(self, name: str) -> InitializedCarrierSpec:
        for carrier in self.carriers:
            if carrier.name == name:
                return carrier
        raise KeyError(name)

    @property
    def is_full_domain(self) -> bool:
        return self.crop_origin_yx == (0, 0) and self.crop_count_yx == (
            REAL_SOURCE_SHAPE_ZYX[1],
            REAL_SOURCE_SHAPE_ZYX[2],
        )


def _parse_carrier(value: object, index: int) -> InitializedCarrierSpec:
    carrier = _object(value, f"carrier {index}")
    _exact_keys(
        carrier,
        {
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
            "statistics",
        },
        f"carrier {index}",
    )
    name = _string(carrier["name"], f"carrier {index} name")
    if carrier["dtype"] != "<f4" or carrier["itemsize"] != 4 or carrier["order"] != "C":
        raise ValueError(f"carrier {name} layout is not <f4 C-order")
    raw_shape = _sequence(carrier["shape"], f"carrier {name} shape")
    shape = tuple(
        _positive_int(extent, f"carrier {name} shape extent") for extent in raw_shape
    )
    if not shape:
        raise ValueError(f"carrier {name} shape has zero rank")
    words = 1
    for extent in shape:
        words *= extent
    nbytes = _positive_int(carrier["nbytes"], f"carrier {name} byte count")
    if nbytes != words * ITEMSIZE:
        raise ValueError(f"carrier {name} byte count disagrees with shape")
    sources = tuple(
        _string(item, f"carrier {name} source array")
        for item in _sequence(carrier["source_arrays"], f"carrier {name} source arrays")
    )
    if not sources:
        raise ValueError(f"carrier {name} has no Rust source-array provenance")
    statistics = _object(carrier["statistics"], f"carrier {name} statistics")
    _exact_keys(
        statistics,
        {"finite_count", "maximum_f32_bits", "minimum_f32_bits", "nonzero_count"},
        f"carrier {name} statistics",
    )
    finite_count = _positive_int(
        statistics["finite_count"], f"carrier {name} finite count"
    )
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
        raise ValueError(f"carrier {name} statistics are not canonical f32 bit words")
    if any((int(bits, 16) & 0x7F80_0000) == 0x7F80_0000 for bits in (minimum, maximum)):
        raise ValueError(f"carrier {name} statistics contain a non-finite endpoint")
    return InitializedCarrierSpec(
        name=name,
        shape=shape,
        offset_nbytes=_positive_int(
            carrier["offset_nbytes"], f"carrier {name} offset", allow_zero=True
        ),
        nbytes=nbytes,
        sha256=_require_sha256(carrier["sha256"], f"carrier {name} SHA-256"),
        role=_string(carrier["role"], f"carrier {name} role"),
        source_arrays=sources,
        minimum_f32_bits=minimum,
        maximum_f32_bits=maximum,
        finite_count=finite_count,
        nonzero_count=nonzero_count,
    )


def _expected_shapes(ny: int, nx: int) -> dict[str, tuple[int, ...]]:
    nz = REAL_SOURCE_SHAPE_ZYX[0]
    mass = (nz, ny, nx)
    zface = (nz + 1, ny, nx)
    return {
        **{name: (nz,) for name in CARRIER_NAMES[:7]},
        "ua": (nz, ny, nx + 1),
        "u3d": (nz, ny, nx + 1),
        "va": (nz, ny + 1, nx),
        "v3d": (nz, ny + 1, nx),
        "wa": zface,
        "w3d": zface,
        "ppi": mass,
        "pp3d": mass,
        "ppx": mass,
        "tha": mass,
        "th3d": mass,
        "qa": (6, nz, ny, nx),
        "q3d": (6, nz, ny, nx),
        "terrain_msl": (ny, nx),
        "zh": mass,
        "zf": zface,
        "mh": mass,
        "rmh": mass,
        "mf": zface,
        "rmf": zface,
        "c1": zface,
        "c2": zface,
        "rho0_3d": mass,
        "rr0": mass,
        "rf0": zface,
        "rrf0": zface,
        "gz": (ny, nx),
        "rgz": (ny, nx),
        "gzu": (ny, nx + 1),
        "rgzu": (ny, nx + 1),
        "gzv": (ny + 1, nx),
        "rgzv": (ny + 1, nx),
        "dzdx": (ny, nx),
        "dzdy": (ny, nx),
        "gx": zface,
        "gxu": (nz + 1, ny, nx + 1),
        "gy": zface,
        "gyv": (nz + 1, ny + 1, nx),
        "prs": mass,
        "rho": mass,
        "rr": mass,
        "rf": zface,
    }


def _validate_carrier_geometry(
    carriers: tuple[InitializedCarrierSpec, ...], ny: int, nx: int
) -> None:
    if tuple(carrier.name for carrier in carriers) != CARRIER_NAMES:
        raise ValueError("initialized-product carrier order/inventory differs from ABI")
    expected = _expected_shapes(ny, nx)
    by_name = {carrier.name: carrier for carrier in carriers}
    for name in CARRIER_NAMES:
        if by_name[name].shape != expected[name]:
            raise ValueError(
                f"carrier {name} shape {by_name[name].shape!r} != {expected[name]!r}"
            )
    for group in DUPLICATE_GROUPS:
        hashes = {by_name[name].sha256 for name in group}
        if len(hashes) != 1:
            raise ValueError(f"duplicate group {group!r} is not byte-identical")
        intervals = sorted(
            (
                by_name[name].offset_nbytes,
                by_name[name].offset_nbytes + by_name[name].nbytes,
                name,
            )
            for name in group
        )
        for left, right in zip(intervals, intervals[1:]):
            if left[1] > right[0]:
                raise ValueError(
                    f"duplicate carriers {left[2]} and {right[2]} alias storage"
                )


def _validate_authority(value: object) -> None:
    authority = _object(value, "initialized-product authority")
    _exact_keys(authority, {"release", "source_files", "tarball_sha256"}, "authority")
    if authority["release"] != "CM1 21.1":
        raise ValueError("initialized-product authority release changed")
    if authority["tarball_sha256"] != AUTHORITY_TARBALL_SHA256:
        raise ValueError("initialized-product authority tarball changed")
    files = _sequence(authority["source_files"], "authority source_files")
    if len(files) != len(AUTHORITY_SOURCE_FILES):
        raise ValueError("initialized-product authority source-file count changed")
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(files):
        item = _object(raw, f"authority source file {index}")
        _exact_keys(
            item, {"anchors", "path", "sha256"}, f"authority source file {index}"
        )
        path = _string(item["path"], f"authority source file {index} path")
        _string(item["anchors"], f"authority source file {index} anchors")
        _require_sha256(item["sha256"], f"authority source file {path} SHA-256")
        normalized.append(item)
    if normalized != list(AUTHORITY_SOURCE_FILES):
        raise ValueError(
            "initialized-product authority file order/anchors/hashes changed"
        )


def _validate_configuration(value: object) -> None:
    configuration = _object(value, "initialized-product configuration")
    _exact_keys(configuration, set(CONFIGURATION), "initialized-product configuration")
    if configuration != CONFIGURATION:
        raise ValueError("initialized-product configuration is unsupported")


def _validate_initialization(value: object) -> None:
    initialization = _object(value, "initialized-product initialization")
    expected_keys = {
        "constants_f32_bits",
        "corrected_carriers",
        "duplicate_groups",
        "moist_species_order",
        "ppi_small_value_rule",
        "qv_density_rule",
        "source_preserved_carriers",
        "statement_order",
    }
    _exact_keys(initialization, expected_keys, "initialized-product initialization")
    constants = _object(
        initialization["constants_f32_bits"], "initialization constants"
    )
    _exact_keys(constants, set(CONSTANT_BITS), "initialization constants")
    if constants != CONSTANT_BITS:
        raise ValueError("initialized-product default-real constants changed")
    exact_values = {
        "corrected_carriers": list(CORRECTED_CARRIERS),
        "duplicate_groups": [list(group) for group in DUPLICATE_GROUPS],
        "moist_species_order": list(SPECIES_ORDER),
        "ppi_small_value_rule": PPI_SMALL_VALUE_RULE,
        "qv_density_rule": QV_DENSITY_RULE,
        "source_preserved_carriers": list(SOURCE_PRESERVED_CARRIERS),
        "statement_order": STATEMENT_ORDER,
    }
    for key, expected in exact_values.items():
        if initialization[key] != expected:
            raise ValueError(f"initialized-product {key} contract changed")


def load_initialized_product_export(
    directory: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
    expected_source_artifact_sha256: str,
    expected_source_contract_sha256: str,
) -> LoadedInitializedProduct:
    """Validate one Rust export completely without importing CUDA or NumPy."""

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
    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("initialized-product export path must not be a symlink")
    root = requested.resolve()
    if not root.is_dir():
        raise ValueError("initialized-product export path must be a real directory")
    inventory = {entry.name for entry in os.scandir(root)}
    expected_inventory = {PAYLOAD_NAME, MANIFEST_NAME, RECEIPT_NAME}
    if inventory != expected_inventory:
        raise ValueError(
            "initialized-product directory inventory mismatch; "
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
            f"initialized-product manifest SHA-256 is {manifest_sha256}; "
            f"independently expected {expected_manifest_sha256}"
        )
    _, receipt = _read_canonical_json(receipt_path)
    receipt_keys = {
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
    }
    _exact_keys(receipt, receipt_keys, "initialized-product receipt")
    if receipt["schema"] != INITIALIZED_PRODUCT_RECEIPT_SCHEMA:
        raise ValueError("initialized-product receipt schema is unsupported")
    if receipt["admission"] != "real-product-300x300x75":
        raise ValueError(
            "production Python refuses non-real initialized-product admission"
        )
    if receipt["carrier_abi"] != INITIALIZED_PRODUCT_CARRIER_ABI:
        raise ValueError("initialized-product receipt carrier ABI changed")
    if receipt["claim_limit"] not in EXPORT_CLAIM_LIMITS:
        raise ValueError("initialized-product receipt claim boundary changed")
    if receipt["publication_contract"] != PUBLICATION_CONTRACT:
        raise ValueError("initialized-product atomic publication contract changed")
    if receipt["manifest_sha256"] != manifest_sha256 or receipt[
        "manifest_nbytes"
    ] != len(manifest_raw):
        raise ValueError("initialized-product receipt does not anchor manifest bytes")

    manifest_keys = {
        "admission",
        "authority",
        "base_policy",
        "carrier_abi",
        "carriers",
        "claim_limit",
        "configuration",
        "crop",
        "dtype",
        "fornberg",
        "formula_contracts",
        "initialization",
        "itemsize",
        "order",
        "payload",
        "schema",
        "source",
        "source_grid",
    }
    _exact_keys(manifest, manifest_keys, "initialized-product manifest")
    if manifest["schema"] != INITIALIZED_PRODUCT_EXPORT_SCHEMA:
        raise ValueError("initialized-product manifest schema is unsupported")
    if manifest["admission"] != "real-product-300x300x75":
        raise ValueError(
            "production Python refuses generic initialized-product exports"
        )
    if manifest["carrier_abi"] != INITIALIZED_PRODUCT_CARRIER_ABI:
        raise ValueError("initialized-product manifest carrier ABI changed")
    if manifest["claim_limit"] not in EXPORT_CLAIM_LIMITS:
        raise ValueError("initialized-product manifest claim boundary changed")
    if manifest["claim_limit"] != receipt["claim_limit"]:
        raise ValueError("initialized-product receipt/manifest claim boundaries differ")
    if manifest["base_policy"] != BASE_POLICY:
        raise ValueError("initialized-product explicit base policy changed")
    if (
        manifest["dtype"] != "<f4"
        or manifest["itemsize"] != 4
        or manifest["order"] != "C"
    ):
        raise ValueError("initialized-product top-level layout must be C-order <f4")
    _validate_authority(manifest["authority"])
    _validate_configuration(manifest["configuration"])
    _validate_initialization(manifest["initialization"])
    formulas = _object(manifest["formula_contracts"], "formula_contracts")
    _exact_keys(formulas, set(FORMULA_CONTRACTS), "formula_contracts")
    if formulas != FORMULA_CONTRACTS:
        raise ValueError("initialized-product formula contracts changed")
    fornberg = _object(manifest["fornberg"], "fornberg")
    _exact_keys(
        fornberg,
        {"bottom_cgs_f32_bits", "source_contract", "top_cgt_f32_bits"},
        "fornberg",
    )
    if fornberg["source_contract"] != FORNBERG_SOURCE_CONTRACT:
        raise ValueError("initialized-product Fornberg source contract changed")
    for name in ("bottom_cgs_f32_bits", "top_cgt_f32_bits"):
        words = _sequence(fornberg[name], f"fornberg {name}")
        if len(words) != 3 or any(
            not isinstance(word, str) or _HEX8.fullmatch(word) is None for word in words
        ):
            raise ValueError(f"fornberg {name} must contain three canonical f32 words")

    source = _object(manifest["source"], "initialized-product source")
    source_keys = {
        "artifact_nbytes",
        "artifact_sha256",
        "contract_sha256",
        "dt_f64_bits",
        "model_time_f64_bits",
        "profile_id",
        "step",
    }
    source_kind = source.get("kind", "initial-checkpoint")
    if "kind" in source:
        source_keys.add("kind")
    _exact_keys(source, source_keys, "initialized-product source")
    if source["artifact_sha256"] != expected_source_artifact_sha256:
        raise ValueError(
            "initialized-product source artifact differs from independent hash"
        )
    if source["contract_sha256"] != expected_source_contract_sha256:
        raise ValueError(
            "initialized-product source contract differs from independent hash"
        )
    if source["profile_id"] != SOURCE_PROFILE_ID:
        raise ValueError("initialized-product source profile is unsupported")
    _positive_int(source["artifact_nbytes"], "source artifact byte count")
    if source_kind == "initial-checkpoint":
        if source["step"] != 0 or source["model_time_f64_bits"] != "0000000000000000":
            raise ValueError(
                "initialized-product checkpoint source must be step-zero/time-zero"
            )
    elif source_kind == "boundary-analysis-history-f01":
        if (
            source["step"] != 7200
            or source["model_time_f64_bits"] != "40ac200000000000"
        ):
            raise ValueError(
                "initialized-product f01 analysis source must be step 7200 at 3600 s"
            )
    elif source_kind == "boundary-analysis-history-f02-roi":
        if (
            source["step"] != 14400
            or source["model_time_f64_bits"] != "40bc200000000000"
        ):
            raise ValueError(
                "initialized-product f02 ROI source must be step 14400 at 7200 s"
            )
    else:
        raise ValueError("initialized-product source kind is unsupported")
    if source["dt_f64_bits"] != "3fe0000000000000":
        raise ValueError("initialized-product source requires exact dt=0.5 s")

    source_grid = _object(manifest["source_grid"], "initialized-product source_grid")
    _exact_keys(source_grid, {"shape_zyx", *REAL_SPACING_BITS}, "source_grid")
    if source_grid["shape_zyx"] != list(REAL_SOURCE_SHAPE_ZYX):
        raise ValueError("initialized-product source grid is not 75x300x300")
    for key, expected in REAL_SPACING_BITS.items():
        if source_grid[key] != expected:
            raise ValueError(f"initialized-product {key} changed")

    crop = _object(manifest["crop"], "initialized-product crop")
    _exact_keys(crop, {"count_yx", "origin_yx", "stagger_rule"}, "crop")
    if crop["stagger_rule"] != CROP_STAGGER_RULE:
        raise ValueError("initialized-product crop staggering contract changed")
    origins = _sequence(crop["origin_yx"], "crop origin_yx")
    counts = _sequence(crop["count_yx"], "crop count_yx")
    if len(origins) != 2 or len(counts) != 2:
        raise ValueError("initialized-product horizontal crop must have y/x rank two")
    origin = tuple(
        _positive_int(value, f"crop origin axis {axis}", allow_zero=True)
        for axis, value in enumerate(origins)
    )
    count = tuple(
        _positive_int(value, f"crop count axis {axis}")
        for axis, value in enumerate(counts)
    )
    if any(
        origin[axis] + count[axis] > REAL_SOURCE_SHAPE_ZYX[axis + 1]
        for axis in range(2)
    ):
        raise ValueError("initialized-product horizontal crop exceeds source grid")

    payload = _object(manifest["payload"], "initialized-product payload")
    _exact_keys(payload, {"file", "nbytes", "sha256"}, "payload")
    if payload["file"] != PAYLOAD_NAME:
        raise ValueError("initialized-product payload filename changed")
    payload_nbytes = _positive_int(payload["nbytes"], "payload byte count")
    payload_sha256 = _require_sha256(payload["sha256"], "payload SHA-256")
    if payload_sha256 != expected_payload_sha256:
        raise ValueError("initialized-product payload differs from independent SHA-256")
    if (
        receipt["payload_sha256"] != payload_sha256
        or receipt["payload_nbytes"] != payload_nbytes
    ):
        raise ValueError(
            "initialized-product receipt/manifest payload identities differ"
        )
    if (
        receipt["source_artifact_sha256"] != expected_source_artifact_sha256
        or receipt["source_contract_sha256"] != expected_source_contract_sha256
    ):
        raise ValueError("initialized-product receipt source identities changed")
    if os.lstat(payload_path).st_size != payload_nbytes:
        raise ValueError("initialized-product payload size differs from manifest")

    raw_carriers = _sequence(manifest["carriers"], "initialized-product carriers")
    carriers = tuple(
        _parse_carrier(value, index) for index, value in enumerate(raw_carriers)
    )
    if receipt["carrier_count"] != len(carriers) or len(carriers) != len(CARRIER_NAMES):
        raise ValueError("initialized-product receipt carrier count is wrong")
    _validate_carrier_geometry(carriers, count[0], count[1])

    aggregate = hashlib.sha256()
    offset = 0
    with payload_path.open("rb", buffering=0) as handle:
        for carrier in carriers:
            if carrier.offset_nbytes != offset:
                raise ValueError(f"carrier {carrier.name} offset is not contiguous")
            field_hash = hashlib.sha256()
            remaining = carrier.nbytes
            while remaining:
                block = handle.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"payload truncates carrier {carrier.name}")
                field_hash.update(block)
                aggregate.update(block)
                remaining -= len(block)
            if field_hash.hexdigest() != carrier.sha256:
                raise ValueError(f"carrier {carrier.name} payload SHA-256 mismatch")
            offset += carrier.nbytes
        if handle.read(1):
            raise ValueError("initialized-product payload has trailing bytes")
    if offset != payload_nbytes or aggregate.hexdigest() != payload_sha256:
        raise ValueError("initialized-product aggregate payload identity changed")

    return LoadedInitializedProduct(
        directory=root,
        payload_path=payload_path,
        manifest_sha256=manifest_sha256,
        manifest_nbytes=len(manifest_raw),
        payload_sha256=payload_sha256,
        payload_nbytes=payload_nbytes,
        source_artifact_sha256=expected_source_artifact_sha256,
        source_contract_sha256=expected_source_contract_sha256,
        crop_origin_yx=origin,  # type: ignore[arg-type]
        crop_count_yx=count,  # type: ignore[arg-type]
        carriers=carriers,
    )


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


class _InitializedDriver(Protocol):
    def allocate_and_copy(
        self, export: LoadedInitializedProduct
    ) -> _DriverAllocation: ...


class _CupyInitializedDriver:
    def __init__(self, attestation: object) -> None:
        if attestation is not _DEFAULT_DRIVER_ATTESTATION:
            raise RuntimeError(
                "production initialized-product driver lacks attestation"
            )
        self._cupy = importlib.import_module("cupy")
        self._numpy = importlib.import_module("numpy")

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="strict")
        return str(value)

    @classmethod
    def _arch(cls, value: object) -> str:
        if (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and all(
                isinstance(part, int) and not isinstance(part, bool) for part in value
            )
        ):
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

    def allocate_and_copy(self, export: LoadedInitializedProduct) -> _DriverAllocation:
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
                    raise ValueError(
                        "initialized-product payload truncated during H2D staging"
                    )
                total += copied
            if handle.read(1):
                raise ValueError("initialized-product payload grew during H2D staging")
        if hashlib.sha256(view).hexdigest() != export.payload_sha256:
            raise ValueError(
                "initialized-product payload changed after disk validation"
            )
        arena.set(host, stream=stream)
        fields: list[tuple[str, object]] = []
        for carrier in export.carriers:
            pointer = arena.data + carrier.offset_nbytes
            array = cupy.ndarray(
                carrier.shape, dtype=cupy.float32, memptr=pointer, order="C"
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
class CudaInitializedProduct:
    export: LoadedInitializedProduct
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
            "carrier_abi": INITIALIZED_PRODUCT_CARRIER_ABI,
            "carrier_count": len(self.export.carriers),
            "carriers": [
                {
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
                "count_yx": list(self.export.crop_count_yx),
                "full_domain": self.export.is_full_domain,
                "origin_yx": list(self.export.crop_origin_yx),
            },
            "device_id": self.device_id,
            "manifest_sha256": self.export.manifest_sha256,
            "ordering": [
                "all-disk-hashes-and-ABI-validated-before-CUDA-import",
                "one-uint8-device-arena-allocated",
                "one-opaque-pinned-host-to-device-copy-enqueued",
                "49-non-aliasing-f32-views-created-from-Rust-offsets",
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


def initialize_cuda_initialized_product(
    export: LoadedInitializedProduct,
    *,
    allow_cropped_validation: bool = False,
    driver_factory: Callable[[], _InitializedDriver] | None = None,
) -> CudaInitializedProduct:
    """Allocate one opaque initialized-product arena and record readiness."""

    if not isinstance(export, LoadedInitializedProduct):
        raise TypeError("export must be a LoadedInitializedProduct")
    if not isinstance(allow_cropped_validation, bool):
        raise TypeError("allow_cropped_validation must be a boolean")
    if not export.is_full_domain and not allow_cropped_validation:
        raise ValueError(
            "cropped initialized-product export is validation-only; set "
            "allow_cropped_validation=True"
        )
    if driver_factory is None:
        driver: _InitializedDriver = _CupyInitializedDriver(_DEFAULT_DRIVER_ATTESTATION)
        audited = True
    else:
        driver = driver_factory()
        audited = False
    allocation = driver.allocate_and_copy(export)
    if tuple(name for name, _ in allocation.fields) != CARRIER_NAMES:
        raise ValueError("CUDA driver returned wrong carrier view inventory/order")
    if (
        allocation.arena is None
        or allocation.ready_event is None
        or allocation.stream is None
    ):
        raise ValueError("CUDA driver returned null arena/event/stream handles")
    if (
        allocation.device_id < 0
        or allocation.stream_pointer < 0
        or allocation.event_pointer < 1
    ):
        raise ValueError("CUDA driver returned invalid device/stream/event metadata")
    if not allocation.runtime_identity:
        raise ValueError("CUDA driver returned no runtime identity")
    return CudaInitializedProduct(
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
        schema=(
            CUDA_INITIALIZED_PRODUCT_SCHEMA
            if audited
            else CUDA_INITIALIZED_PRODUCT_TEST_SCHEMA
        ),
        claim_limit=CUDA_CLAIM_LIMIT if audited else CUDA_TEST_CLAIM_LIMIT,
        _pinned_host=allocation.pinned_host,
    )


__all__ = [
    "CARRIER_NAMES",
    "CUDA_CLAIM_LIMIT",
    "CUDA_INITIALIZED_PRODUCT_SCHEMA",
    "CUDA_INITIALIZED_PRODUCT_TEST_SCHEMA",
    "CudaInitializedProduct",
    "EXPORT_CLAIM_LIMIT",
    "INITIALIZED_PRODUCT_CARRIER_ABI",
    "INITIALIZED_PRODUCT_EXPORT_SCHEMA",
    "INITIALIZED_PRODUCT_RECEIPT_SCHEMA",
    "InitializedCarrierSpec",
    "LoadedInitializedProduct",
    "initialize_cuda_initialized_product",
    "load_initialized_product_export",
]
