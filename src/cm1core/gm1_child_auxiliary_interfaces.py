"""Fail-closed interfaces for the fixed-50 m GM1 child auxiliaries.

This module does not construct a numerical child.  It binds the exact real
terrain/static publication, the exact Rust restart-grid atmosphere, the real
f02 base-profile bytes, and the selected fresh-TKE operator proof.  It then
records the CM1 r21.1 source order and every remaining producer seam.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

from cm1core.gm1_child_atmosphere_admission import (
    GM1_CHILD_ATMOSPHERE_ARENA_SHA256,
    GM1_CHILD_ATMOSPHERE_MANIFEST_SHA256,
    GM1_CHILD_ATMOSPHERE_RECEIPT_SHA256,
    LoadedGm1ChildAtmosphere,
)
from cm1core.gm1_child_initialization import GM1_CHILD_PLAN_SHA256
from cm1core.gm1_child_static_admission import (
    GM1_CHILD_STATIC_HEIGHT_MANIFEST_SHA256,
    GM1_CHILD_STATIC_VALIDATION_SHA256,
    GM1_CHILD_STATIC_ZF_SHA256,
    GM1_CHILD_STATIC_ZH_SHA256,
    Gm1ChildStaticAdmission,
)


GM1_CHILD_AUXILIARY_INTERFACES_SCHEMA = (
    "gm1.fixed-50m-child-auxiliary-sequence/v1"
)
GM1_CHILD_AUXILIARY_CLAIM_LIMIT = (
    "exact CUDA-cold source-sequence and real-input bindings for the fixed "
    "300x300x75, 50 m GM1 child base/hydrostatic, restart-state halo, live "
    "density, fresh closure, four-open-boundary, and geopotential producers; "
    "no numerical auxiliary publication, runnable child, evolution, forecast, "
    "storm, or tornado claim"
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = MappingProxyType(
    {
        "src/base.F": "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
        "src/bc.F": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
        "src/cm1.F": "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
        "src/constants.F": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
        "src/init3d.F": "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
        "src/restart_read.F": "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a",
        "src/turb.F": "9984ad47b7b5c8f92b8a2e98933c11474b63c7e315773117f4613620a660f04d",
    }
)
SOURCE_ANCHORS = MappingProxyType(
    {
        "driver_order": "src/cm1.F:1462-1471,2390-2403,2660-2705,2884-2904",
        "base_sounding_hydrostatic": "src/base.F:625-749",
        "base_halos_diagnostics_surface": "src/base.F:2044-2245",
        "restart_tke_exclusion": "src/restart_read.F:1206-1211",
        "restart_held_zero": "src/restart_read.F:2657-2664",
        "restart_grid_interpolation": "src/restart_read.F:3302-3465",
        "getset_family_boundaries": "src/init3d.F:2149-2161,2248-2268",
        "getset_a_to_current": "src/init3d.F:2280-2333",
        "getset_live_density": "src/init3d.F:2365-2467",
        "getset_psolver3_ppx": "src/init3d.F:2535-2575",
        "getset_fresh_turbulence": "src/init3d.F:2688-2756",
        "open_boundary_families": "src/bc.F:12-172,180-1206,1948-2172",
        "fresh_radiation_history": "src/cm1.F:1559-1566",
    }
)

BASE_SOURCE_MANIFEST_SHA256 = (
    "b25de4008d17ddad46b8057f022b50a76e6bff61b537fa9952f68e638adf2567"
)
BASE_SOURCE_PAYLOAD_SHA256 = (
    "ddafeaa8ca523e81925cb4b53e0577963a3e57e06114b6b7df8c6d436e0d68b1"
)
BASE_SOURCE_RECEIPT_SHA256 = (
    "379b989884e773a942ade62069e99ce5e032374c2ae2ef13679f32a524fbbdcd"
)
BASE_SOURCE_PAYLOAD_NBYTES = 1_169_829_300
BASE_SOURCE_Z_PROFILE_SHA256 = (
    "458b7ecc933a04b929e2d59e2a1b0cde856e583189d9690d5c481fed4e4f9419"
)
BASE_SOURCE_PROFILES = MappingProxyType(
    {
        "pi0": (0, "9b7108b591daf3b06d7a8f08f4a3e2c631c808175b03642da1356ad4c7e17d8c"),
        "prs0": (
            300,
            "b24cc3213f1997627083a21e75552f874264e2be4866c732165f91c1b728f733",
        ),
        "qv0": (
            600,
            "1298f4d0a36bc5795f3ca4ed34eb47e9a2ce52d269de4057b8c709b9c0062e3a",
        ),
        "rho0": (
            900,
            "fdff2d1dfb2e88c3cde8a7d0773a19bf50c94b20b00bbfdad81428cd64728a6b",
        ),
        "th0": (
            1200,
            "09d1c20d88e1a6b4ee9af5999eb7086f63472839c14e94c97ff33daec554f492",
        ),
    }
)

FRESH_TKE_OPERATOR = MappingProxyType(
    {
        "schema": "gm1.cuda-child-turbsmag-init/v1",
        "abi": "gm1.cuda-child-turbsmag-init-f32/v1",
        "api": "cm1core.cuda_gm1_child_turbsmag_init.launch_cuda_child_turbsmag_init",
        "commit": "d32af2cb8fd343e551a7911aedd2e057c827961c",
        "module_sha256": "a25c00213b70974f1b2faf52050859a055b83eea61d935f0deb67d99eb0b42f7",
        "kernel_source_sha256": "8937435b68d61aa6c602b5f5c37796216e15a61e725b83cc0e4aab3af7eb5ac5",
        "native_receipt_sha256": "173094127c34a8cd06689628e89b98444df8ff7a63419231db6a3e3dedc04d23",
        "pow13_receipt_sha256": "d0eba61ee44091b30dbf4e0ab1fd28e366c8f984d7840a68e1dc8ea3a52aacfe",
    }
)
GEOPOTENTIAL_INTERFACE = MappingProxyType(
    {
        "schema": "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary/v1",
        "receipt_schema": (
            "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary-receipt/v1"
        ),
        "abi": "gpuwm-cm1.arwen-wsm6-total-geopotential/v1",
        "carrier": "total_geopotential_zf",
        "shape": [76, 300, 300],
        "dtype": "<f4",
        "arithmetic": "one binary32 store of zf * binary32(g) per C-order word",
        "rust_source_sha256": (
            "07cc7d7e03b89e9d7d7e06fcfb02b92f5c90ccf87e60c60099e8cfd806ffa0fc"
        ),
    }
)

_BASE_NAMES = (
    "pi0",
    "prs0",
    "th0",
    "thv0",
    "rth0",
    "qv0",
    "rho0",
    "rr0",
    "rf0",
    "rrf0",
)
_SURFACE_NAMES = (
    "rho0s",
    "pi0s",
    "prs0s",
    "rth0s",
    "terrain_msl_halo",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
)
_PRESSURE_NAMES = ("ppi", "pp3d", "ppx", "piadv", "ppterm", "ppten", "thv")
_SCALAR_NAMES = ("tha", "th3d", "thten", "thten1")
_U_NAMES = ("ua", "u3d", "uten", "uten1")
_V_NAMES = ("va", "v3d", "vten", "vten1")
_W_NAMES = ("wa", "w3d", "wten", "wten1")
_RAD_WE_NAMES = ("radbcw", "radbce")
_RAD_SN_NAMES = ("radbcs", "radbcn")
TERRAIN_OPEN_CARRIER_SHAPES = MappingProxyType(
    {
        **{name: (77, 306, 306) for name in _BASE_NAMES},
        **{name: (306, 306) for name in _SURFACE_NAMES},
        **{name: (77, 302, 302) for name in _PRESSURE_NAMES},
        **{name: (77, 306, 306) for name in _SCALAR_NAMES},
        **{name: (77, 306, 307) for name in _U_NAMES},
        **{name: (77, 307, 306) for name in _V_NAMES},
        **{name: (78, 306, 306) for name in _W_NAMES},
        **{name: (77, 306) for name in _RAD_WE_NAMES},
        **{name: (77, 306) for name in _RAD_SN_NAMES},
    }
)
LIVE_DENSITY_SHAPES = MappingProxyType(
    {
        "prs": (75, 300, 300),
        "rho": (75, 300, 300),
        "rr": (75, 300, 300),
        "rf": (76, 300, 300),
    }
)
FRESH_TKE_INPUT_SHAPES = MappingProxyType(
    {
        "ruh": (300,),
        "rvh": (300,),
        "rmf": (76, 300, 300),
        "nm": (76, 300, 300),
        "defv": (76, 300, 300),
        "defh": (76, 300, 300),
    }
)
FRESH_TKE_OUTPUT_SHAPES = MappingProxyType(
    {
        "grdscl": (75, 300, 300),
        **{
            name: (76, 300, 300)
            for name in (
                "lenscl",
                "dissten",
                "kmh",
                "kmv",
                "khh",
                "khv",
                "tkea",
                "tke3d",
                "tketen",
            )
        },
    }
)


@dataclass(frozen=True, slots=True)
class Gm1ChildBaseSource:
    directory: Path
    manifest_path: Path
    payload_path: Path
    receipt_path: Path
    profile_sha256: Mapping[str, str]
    source_z_profile_sha256: str


@dataclass(frozen=True, slots=True)
class Gm1ChildAuxiliaryInterfaces:
    schema: str
    static: Gm1ChildStaticAdmission
    atmosphere: LoadedGm1ChildAtmosphere
    base_source: Gm1ChildBaseSource
    authority_root: Path
    stages: Mapping[str, Mapping[str, Any]]
    blockers: tuple[str, ...]
    numerical_child_output_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": "PASS_EXACT_INPUT_AND_SOURCE_SEQUENCE_NUMERICS_BLOCKED",
            "model": "GM1",
            "configuration": {
                "nx": 300,
                "ny": 300,
                "nz": 75,
                "dx_m": 50.0,
                "dy_m": 50.0,
                "ztop_m": 15000.0,
                "psolver": 3,
                "imoist": 1,
                "eqtset": 2,
                "terrain": True,
                "imove": 0,
                "lateral_boundaries": [2, 2, 2, 2],
                "interp_on_restart": True,
            },
            "authority": {
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
                "anchors": dict(SOURCE_ANCHORS),
            },
            "real_inputs": {
                "parent_plan_sha256": GM1_CHILD_PLAN_SHA256,
                "static_validation_sha256": self.static.validation_sha256,
                "static_height_manifest_sha256": (
                    GM1_CHILD_STATIC_HEIGHT_MANIFEST_SHA256
                ),
                "zh_sha256": GM1_CHILD_STATIC_ZH_SHA256,
                "zf_sha256": GM1_CHILD_STATIC_ZF_SHA256,
                "atmosphere_manifest_sha256": self.atmosphere.manifest_sha256,
                "atmosphere_arena_sha256": self.atmosphere.arena_sha256,
                "atmosphere_receipt_sha256": self.atmosphere.receipt_sha256,
                "base_source_manifest_sha256": BASE_SOURCE_MANIFEST_SHA256,
                "base_source_payload_sha256": BASE_SOURCE_PAYLOAD_SHA256,
                "base_source_receipt_sha256": BASE_SOURCE_RECEIPT_SHA256,
                "base_source_z_profile_sha256": BASE_SOURCE_Z_PROFILE_SHA256,
                "base_source_profile_sha256": dict(self.base_source.profile_sha256),
                "synthetic_data": False,
                "synthetic_translation": False,
            },
            "stage_order": list(self.stages),
            "stages": {name: dict(stage) for name, stage in self.stages.items()},
            "terrain_open_carrier_shapes": {
                name: list(shape) for name, shape in TERRAIN_OPEN_CARRIER_SHAPES.items()
            },
            "live_density_shapes": {
                name: list(shape) for name, shape in LIVE_DENSITY_SHAPES.items()
            },
            "blockers": list(self.blockers),
            "numerical_child_output_allowed": False,
            "claim_limit": GM1_CHILD_AUXILIARY_CLAIM_LIMIT,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 is {actual}; expected {expected_sha256}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error
    canonical = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    if raw != canonical:
        raise ValueError(f"{label} is not canonical JSON")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return value


def _validate_authority(authority_root: str | Path) -> Path:
    requested = Path(authority_root)
    if requested.is_symlink():
        raise ValueError("CM1 authority root must not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("CM1 authority root must be a directory")
    for relative, expected in AUTHORITY_SOURCE_SHA256.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing regular CM1 authority source {relative}")
        if _sha256_file(path) != expected:
            raise ValueError(f"CM1 authority source {relative} differs")
    return root


def load_gm1_child_base_source(directory: str | Path) -> Gm1ChildBaseSource:
    """Admit the exact f02 source profiles used as the child base sounding."""

    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("base-source directory must not be a symlink")
    root = requested.resolve(strict=True)
    expected_inventory = {
        "initialized-carriers.f32le",
        "manifest.json",
        "receipt.json",
    }
    entries = tuple(root.iterdir())
    if (
        not root.is_dir()
        or {entry.name for entry in entries} != expected_inventory
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("base-source directory inventory changed")
    manifest_path = root / "manifest.json"
    payload_path = root / "initialized-carriers.f32le"
    receipt_path = root / "receipt.json"
    manifest = _canonical_json(
        manifest_path, BASE_SOURCE_MANIFEST_SHA256, "base-source manifest"
    )
    receipt = _canonical_json(
        receipt_path, BASE_SOURCE_RECEIPT_SHA256, "base-source receipt"
    )
    payload = manifest.get("payload")
    source = manifest.get("source")
    grid = manifest.get("source_grid")
    if (
        manifest.get("schema") != "gpuwm-cm1.cuda-initialized-product-export/v1"
        or manifest.get("admission") != "real-product-300x300x75"
        or manifest.get("carrier_abi")
        != "gpuwm-cm1.cuda-initialized-product-carriers/v1"
        or not isinstance(payload, Mapping)
        or payload.get("file") != "initialized-carriers.f32le"
        or payload.get("nbytes") != BASE_SOURCE_PAYLOAD_NBYTES
        or payload.get("sha256") != BASE_SOURCE_PAYLOAD_SHA256
        or not isinstance(source, Mapping)
        or source.get("kind") != "boundary-analysis-history-f02-roi"
        or source.get("artifact_sha256")
        != "356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d"
        or source.get("contract_sha256")
        != "a9ccd6f0c6650fd89cd42f16977209f57695464181bcb62beff14e9a9f61d896"
        or not isinstance(grid, Mapping)
        or grid.get("shape_zyx") != [75, 300, 300]
        or grid.get("target_dz_m_f64_bits") != "4069000000000000"
    ):
        raise ValueError("base-source manifest identity/configuration changed")
    if (
        receipt.get("manifest_sha256") != BASE_SOURCE_MANIFEST_SHA256
        or receipt.get("payload_sha256") != BASE_SOURCE_PAYLOAD_SHA256
        or receipt.get("payload_nbytes") != BASE_SOURCE_PAYLOAD_NBYTES
        or receipt.get("source_artifact_sha256") != source["artifact_sha256"]
        or receipt.get("source_contract_sha256") != source["contract_sha256"]
    ):
        raise ValueError("base-source receipt does not bind the manifest/payload")
    if payload_path.stat().st_size != BASE_SOURCE_PAYLOAD_NBYTES:
        raise ValueError("base-source payload byte count changed")
    if _sha256_file(payload_path) != BASE_SOURCE_PAYLOAD_SHA256:
        raise ValueError("base-source payload SHA-256 changed")

    raw_carriers = manifest.get("carriers")
    if not isinstance(raw_carriers, list):
        raise ValueError("base-source carrier inventory is not a list")
    by_name = {
        item.get("name"): item for item in raw_carriers if isinstance(item, Mapping)
    }
    if len(by_name) != len(raw_carriers):
        raise ValueError("base-source carrier inventory has duplicates/non-objects")
    verified: dict[str, str] = {}
    with payload_path.open("rb") as stream:
        for name, (offset, expected_sha) in BASE_SOURCE_PROFILES.items():
            item = by_name.get(name)
            if (
                not isinstance(item, Mapping)
                or item.get("shape") != [75]
                or item.get("dtype") != "<f4"
                or item.get("order") != "C"
                or item.get("offset_nbytes") != offset
                or item.get("nbytes") != 300
                or item.get("sha256") != expected_sha
            ):
                raise ValueError(f"base-source profile {name} contract changed")
            stream.seek(offset)
            actual = hashlib.sha256(stream.read(300)).hexdigest()
            if actual != expected_sha:
                raise ValueError(f"base-source profile {name} bytes changed")
            verified[name] = actual
    z_bytes = b"".join(
        struct.pack("<f", (k + 0.5) * 200.0) for k in range(75)
    )
    if hashlib.sha256(z_bytes).hexdigest() != BASE_SOURCE_Z_PROFILE_SHA256:
        raise AssertionError("programmed source-height profile identity changed")
    return Gm1ChildBaseSource(
        directory=root,
        manifest_path=manifest_path,
        payload_path=payload_path,
        receipt_path=receipt_path,
        profile_sha256=MappingProxyType(verified),
        source_z_profile_sha256=BASE_SOURCE_Z_PROFILE_SHA256,
    )


def _validate_operator_proofs(workspace_root: Path) -> None:
    paths = (
        (
            workspace_root / "src/cm1core/cuda_gm1_child_turbsmag_init.py",
            FRESH_TKE_OPERATOR["module_sha256"],
            "fresh-TKE module",
        ),
        (
            workspace_root
            / "evidence/oracles/gm1-child-turbsmag-init-v1/"
            "gm1-child-turbsmag-init-receipt.json",
            FRESH_TKE_OPERATOR["native_receipt_sha256"],
            "fresh-TKE native receipt",
        ),
        (
            workspace_root
            / "evidence/validation/gm1-child-turbsmag-pow13-v1/"
            "validation-receipt.json",
            FRESH_TKE_OPERATOR["pow13_receipt_sha256"],
            "fresh-TKE selected-path receipt",
        ),
        (
            workspace_root / "rust/cm1-state/src/geopotential.rs",
            GEOPOTENTIAL_INTERFACE["rust_source_sha256"],
            "geopotential source",
        ),
    )
    for path, expected, label in paths:
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"{label} identity changed")
    pow13 = json.loads(paths[2][0].read_bytes())
    comparison = pow13.get("cuda", {}).get("comparisons", {}).get(
        "corrected_selected_path", {}
    )
    if (
        pow13.get("schema") != "gm1.cuda-child-turbsmag-pow13-validation/v1"
        or pow13.get("status") != "PASS_SELECTED_REAL_CHILD_POW13_RAW_BIT_EXACT"
        or pow13.get("cuda", {}).get("kernel_source_sha256")
        != FRESH_TKE_OPERATOR["kernel_source_sha256"]
        or comparison.get("passed") is not True
        or comparison.get("mismatch_count") != 0
        or comparison.get("record_count") != 6_660_000
    ):
        raise ValueError("fresh-TKE selected real-metric operator proof did not pass")


def build_gm1_child_auxiliary_interfaces(
    static: Gm1ChildStaticAdmission,
    atmosphere: LoadedGm1ChildAtmosphere,
    base_source: Gm1ChildBaseSource,
    *,
    workspace_root: str | Path,
    authority_root: str | Path,
) -> Gm1ChildAuxiliaryInterfaces:
    """Bind exact inputs and return the still-refusing child auxiliary plan."""

    if not isinstance(static, Gm1ChildStaticAdmission):
        raise TypeError("static must be a Gm1ChildStaticAdmission")
    if not isinstance(atmosphere, LoadedGm1ChildAtmosphere):
        raise TypeError("atmosphere must be a LoadedGm1ChildAtmosphere")
    if not isinstance(base_source, Gm1ChildBaseSource):
        raise TypeError("base_source must be a Gm1ChildBaseSource")
    if (
        static.validation_sha256 != GM1_CHILD_STATIC_VALIDATION_SHA256
        or atmosphere.manifest_sha256 != GM1_CHILD_ATMOSPHERE_MANIFEST_SHA256
        or atmosphere.arena_sha256 != GM1_CHILD_ATMOSPHERE_ARENA_SHA256
        or atmosphere.receipt_sha256 != GM1_CHILD_ATMOSPHERE_RECEIPT_SHA256
        or atmosphere.manifest["parent"]["plan_sha256"] != GM1_CHILD_PLAN_SHA256
        or static.receipt["cuda_static_metrics"]["launch_receipt"]["source"]
        ["parent_plan_sha256"]
        != GM1_CHILD_PLAN_SHA256
    ):
        raise ValueError("child auxiliary inputs do not bind the admitted real plan")
    requested_workspace = Path(workspace_root)
    if requested_workspace.is_symlink():
        raise ValueError("workspace root must not be a symlink")
    workspace = requested_workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace root must be a directory")
    authority = _validate_authority(authority_root)
    _validate_operator_proofs(workspace)

    stages: dict[str, Mapping[str, Any]] = {
        "base_terrain_hydrostatic": MappingProxyType(
            {
                "status": "SOURCE_BOUND_AWAITING_NUMERICAL_PUBLICATION",
                "inputs": ["static.zh", "static.c1", "static.c2", "real f02 sounding"],
                "sounding_profiles_used": ["pi0", "th0", "qv0"],
                "source_profiles_audited_not_reused": ["prs0", "rho0"],
                "source_height_contract": (
                    "75 little-endian f32 mass heights (k+0.5)*200 m, then "
                    "source Fornberg surface/top endpoints"
                ),
                "outputs": list(_BASE_NAMES + _SURFACE_NAMES[:4]),
                "rule": (
                    "base.F sounding interpolation to each terrain-following zh "
                    "column, hydrostatic pi0, then base halos/diagnostics/surface"
                ),
            }
        ),
        "restart_state_and_family_boundaries": MappingProxyType(
            {
                "status": "REAL_INTERIOR_ADMITTED_AWAITING_HALO_PUBLICATION",
                "inputs": ["ppi", "tha", "qa", "ua", "va", "wa"],
                "preserve_without_rebalance": ["ppi", "tha", "qa"],
                "rule": (
                    "restart_read overwrites init3d scratch; preserve the remapped "
                    "interior and apply exact bcu/bcv/bcw/bcs plus terrain bcwsfc"
                ),
            }
        ),
        "a_to_current_copies": MappingProxyType(
            {
                "status": "INTERFACE_READY_AWAITING_NUMERICAL_PUBLICATION",
                "raw_bit_copies": [
                    "ua->u3d",
                    "va->v3d",
                    "wa->w3d",
                    "ppi->pp3d",
                    "tha->th3d",
                    "qa->q3d",
                ],
                "forbidden": "rerun init3d.F:1733-1785 after restart interpolation",
            }
        ),
        "live_moist_density": MappingProxyType(
            {
                "status": "INTERFACE_READY_AWAITING_NUMERICAL_PUBLICATION",
                "inputs": ["pi0", "th0", "pp3d", "th3d", "q3d[qv]", "c1", "c2"],
                "outputs": {name: list(shape) for name, shape in LIVE_DENSITY_SHAPES.items()},
                "source_order": [
                    "prs=p00*(pi0+pp3d)**cpdrd",
                    "rho=prs/((th0+th3d)*(pi0+pp3d)*(rd+max(0,qv)*rv))",
                    "bcs(rho); rr=1/rho; rf=c1*rho[k-1]+c2*rho[k]",
                    "source Fornberg bottom/top rf",
                ],
                "separate_payload_required": True,
            }
        ),
        "psolver3_pressure_helper": MappingProxyType(
            {
                "status": "INTERFACE_READY_AWAITING_NUMERICAL_PUBLICATION",
                "overwrites_independently_remapped_ppx": True,
                "raw_bit_copy": "ppi->ppx",
                "rule": (
                    "after live density, interp_on_restart plus psolver=3 executes "
                    "ppx(i,j,k)=ppi(i,j,k), then bcs(ppx)"
                ),
                "authority": "src/init3d.F:2535-2575",
            }
        ),
        "child_nm_deformation": MappingProxyType(
            {
                "status": "BLOCKED_CHILD_CALCNM_AND_CALCDEF_NOT_ADMITTED",
                "required_outputs": ["nm", "defv", "defh"],
                "forbidden_substitute": "zero, parent-grid, or synthetic fields",
            }
        ),
        "fresh_turbsmag_tke": MappingProxyType(
            {
                "status": "OPERATOR_PROVEN_INPUTS_BLOCKED",
                "operator": dict(FRESH_TKE_OPERATOR),
                "inputs": {
                    name: list(shape) for name, shape in FRESH_TKE_INPUT_SHAPES.items()
                },
                "outputs": {
                    name: list(shape) for name, shape in FRESH_TKE_OUTPUT_SHAPES.items()
                },
                "disposition": [
                    "tkea bottom copies k=1 interior",
                    "tkea top is exact positive zero",
                    "tke3d is raw-bit tkea copy",
                    "tketen is exact positive zero",
                ],
            }
        ),
        "four_open_boundary_history": MappingProxyType(
            {
                "status": "POLICY_READY_AWAITING_ATOMIC_AUXILIARY_PUBLICATION",
                "lateral_tuple": [2, 2, 2, 2],
                "radbc_shapes": {
                    name: list(TERRAIN_OPEN_CARRIER_SHAPES[name])
                    for name in _RAD_WE_NAMES + _RAD_SN_NAMES
                },
                "radbc_initialization": "exact positive zero",
                "nonzero_words": 0,
                "parent_history_reuse_allowed": False,
            }
        ),
        "zf_total_geopotential": MappingProxyType(
            {
                "status": "INTERFACE_READY_AWAITING_ATOMIC_PAYLOAD_PUBLICATION",
                "input_zf_sha256": GM1_CHILD_STATIC_ZF_SHA256,
                **dict(GEOPOTENTIAL_INTERFACE),
            }
        ),
    }
    blockers = (
        "source-bound terrain/base/open auxiliary numerical artifact is not published",
        "child calcnm and calcdef outputs nm/defv/defh are not admitted",
        "fresh-TKE operator has no admitted real child nm/defv/defh inputs or output",
        "live prs/rho/rr/rf payload is not published",
        "zf-derived child geopotential payload is not published",
        "atomic all-producer receipt is not registered",
    )
    return Gm1ChildAuxiliaryInterfaces(
        schema=GM1_CHILD_AUXILIARY_INTERFACES_SCHEMA,
        static=static,
        atmosphere=atmosphere,
        base_source=base_source,
        authority_root=authority,
        stages=MappingProxyType(stages),
        blockers=blockers,
    )


def require_gm1_child_auxiliary_numerics(
    interfaces: Gm1ChildAuxiliaryInterfaces,
) -> NoReturn:
    """Always refuse until a later exact all-producer receipt is registered."""

    if not isinstance(interfaces, Gm1ChildAuxiliaryInterfaces):
        raise TypeError("interfaces must be a Gm1ChildAuxiliaryInterfaces")
    raise ValueError(
        "GM1-50M-CHILD-AUXILIARIES-UNBOUND: " + "; ".join(interfaces.blockers)
    )


__all__ = (
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "BASE_SOURCE_MANIFEST_SHA256",
    "BASE_SOURCE_PAYLOAD_SHA256",
    "BASE_SOURCE_PROFILES",
    "BASE_SOURCE_RECEIPT_SHA256",
    "BASE_SOURCE_Z_PROFILE_SHA256",
    "FRESH_TKE_INPUT_SHAPES",
    "FRESH_TKE_OPERATOR",
    "FRESH_TKE_OUTPUT_SHAPES",
    "GEOPOTENTIAL_INTERFACE",
    "GM1_CHILD_AUXILIARY_CLAIM_LIMIT",
    "GM1_CHILD_AUXILIARY_INTERFACES_SCHEMA",
    "Gm1ChildAuxiliaryInterfaces",
    "Gm1ChildBaseSource",
    "LIVE_DENSITY_SHAPES",
    "SOURCE_ANCHORS",
    "TERRAIN_OPEN_CARRIER_SHAPES",
    "build_gm1_child_auxiliary_interfaces",
    "load_gm1_child_base_source",
    "require_gm1_child_auxiliary_numerics",
)
