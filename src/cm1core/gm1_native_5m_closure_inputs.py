"""Strict, duplicate-free native 5 m input binding for fresh closure tiles.

The four native artifacts were independently produced and sealed before this
module existed.  This loader does not reinterpret or resize their fields.  It
binds their exact manifests, independently hashes every carrier consumed by
the closure operator, memory-maps those carriers read-only, and packs only one
physical compute window at a time.

The static-height artifact is provenance-bound but is not read by the closure
operator: tile metrics are rebuilt from the exact terrain and sealed stretched
vertical profiles by the admitted CUDA source.  No full-domain input array is
copied or materialized here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any

from cm1core.cuda_gm1_tilestream import Gm1TileSpec
from cm1core.cuda_gm1_tilestream_stretched_closure import INPUT_NAMES


SCHEMA = "gm1.native-5m-stretched-closure-input-binding/v1"
STATUS = "PASS_EXACT_NATIVE_5M_CLOSURE_INPUT_BINDING"
CLAIM_LIMIT = (
    "read-only native 3000x3000x75/5 m terrain, atmosphere, base, and live "
    "density carriers consumed by fresh closure only; no closure output, "
    "complete child, RK stage, checkpoint, forecast, or synthetic detail claim"
)
NX = 3000
NY = 3000
NZ = 75
STREAM_CHUNK_BYTES = 8 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}")

ARTIFACT_PINS = MappingProxyType(
    {
        "terrain": MappingProxyType(
            {
                "manifest_sha256": "7a6b5139168af3384c58998bff8df1dd61f102c2f611ac163251425e607ca45e",
                "receipt_sha256": "553b27baa2f0fda542703a09ba924284f78e9c251e14cd56309958d39c151152",
                "validation_sha256": "d99f1ae105773cc808a2fd11e330ddeed2ca5cf6a7f59f6db96d09f13ad239cd",
            }
        ),
        "atmosphere": MappingProxyType(
            {
                "manifest_sha256": "ba1391a3d9f4914f4b9a23f0d5021994d7587786829f4b85cb6eee0db5047447",
                "receipt_sha256": "24790ddca9403c5066c69517e51442ff24d19d7f6dc38f999cabe96116983483",
                "validation_sha256": "c51a34ffc1625bb60222a96c60d13b3fe60a04bb82cab396d6ad7acb95b69b48",
            }
        ),
        "static_heights": MappingProxyType(
            {
                "manifest_sha256": "e991bf9b85ca71d835b5d34c16e4c877a97af3e36dd12e69a44a930bc96e8104",
                "receipt_sha256": "1501521cbf0dd4a025d842916bf2cd0a5939f7c7150484751b52dd64ce920155",
                "validation_sha256": "cd3032b2a515cf40200532b84780e8b1a1d6a1b92b645d67c8df3333176e3766",
            }
        ),
        "auxiliary": MappingProxyType(
            {
                "manifest_sha256": "282b1ddbc09b3fe70b45c38ba60ce9ec96dc30c5cee2420a645247515a723002",
                "receipt_sha256": "f0762b758a997f4443ad311c66ade1e5b43f3d51353b1a45d9fa4553e24009e9",
                "validation_sha256": "6c7852db9d55bb96e844b577312b269f2e84cdbb206669051349afb43f7442d7",
            }
        ),
    }
)

PAYLOAD_PINS = MappingProxyType(
    {
        "terrain": MappingProxyType(
            {
                "file": "terrain-msl.f32le",
                "nbytes": 36_000_000,
                "sha256": "2a444d93e482cc159a2b2b83392d41105a5587261cef801f623a2d4d9d51d72d",
            }
        ),
        "atmosphere": MappingProxyType(
            {
                "file": "arena.f32le",
                "nbytes": 84_027_600_000,
                "sha256": "36c945e488307900b30cf809ffd8c777d84f6cd5e4ecaa318dcecdc95fa23ce5",
            }
        ),
        "auxiliary_terrain_open": MappingProxyType(
            {
                "file": "terrain-open.f32le",
                "nbytes": 92_416_017_536,
                "sha256": "66f8d27901e2468e3a998f247c9bb5608f61f3d002435b713440b7773177701e",
            }
        ),
        "auxiliary_live_density": MappingProxyType(
            {
                "file": "live-density.f32le",
                "nbytes": 10_836_000_000,
                "sha256": "e3f0e32677828420bbf401d3653eeffe7a77ad313f258ebf7cb1038cf86305a5",
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Native5mCarrier:
    group: str
    file: str
    name: str
    shape: tuple[int, ...]
    offset_nbytes: int
    nbytes: int
    sha256: str

    def record(self) -> dict[str, object]:
        return {
            "group": self.group,
            "file": self.file,
            "name": self.name,
            "shape": list(self.shape),
            "offset_nbytes": self.offset_nbytes,
            "nbytes": self.nbytes,
            "sha256": self.sha256,
            "dtype": "little-endian-f32",
            "order": "C",
        }


CONSUMED_CARRIERS = MappingProxyType(
    {
        "terrain_msl": Native5mCarrier(
            "terrain",
            "terrain-msl.f32le",
            "terrain_msl",
            (NY, NX),
            0,
            36_000_000,
            "2a444d93e482cc159a2b2b83392d41105a5587261cef801f623a2d4d9d51d72d",
        ),
        "pp3d": Native5mCarrier(
            "atmosphere",
            "arena.f32le",
            "pp3d",
            (NZ, NY, NX),
            21_852_000_000,
            2_700_000_000,
            "71d70044e9324fbdb5d4bce665afd37771f668bce46c3aaffb76edc860a86e46",
        ),
        "q3d": Native5mCarrier(
            "atmosphere",
            "arena.f32le",
            "q3d",
            (6, NZ, NY, NX),
            29_952_000_000,
            16_200_000_000,
            "eab3e8476d681634b13aa38d295fe59f414e205175344cc7f2c610d90f8a37f3",
        ),
        "th3d": Native5mCarrier(
            "atmosphere",
            "arena.f32le",
            "th3d",
            (NZ, NY, NX),
            62_352_000_000,
            2_700_000_000,
            "635c7f9ff45f55265e49275e53959f2e6515509fbe043e7d373db750291ca6a3",
        ),
        "u3d": Native5mCarrier(
            "atmosphere",
            "arena.f32le",
            "u3d",
            (NZ, NY, NX + 1),
            67_752_000_000,
            2_700_900_000,
            "f7b8a9e5e42a82b8e7a968906c303ce98a261b21a62897616e0c545fa32f9c85",
        ),
        "v3d": Native5mCarrier(
            "atmosphere",
            "arena.f32le",
            "v3d",
            (NZ, NY + 1, NX),
            73_153_800_000,
            2_700_900_000,
            "0e7cedd1d33dfe42b96c4efc1279229c95d03fc9caa234e47cc801529c4c93df",
        ),
        "w3d": Native5mCarrier(
            "atmosphere",
            "arena.f32le",
            "w3d",
            (NZ + 1, NY, NX),
            78_555_600_000,
            2_736_000_000,
            "b5fc9a72ddc68d469982930b7dd4b5226d4958b51f8ce7feaf7a7bf4cc99c3b8",
        ),
        "pi0": Native5mCarrier(
            "auxiliary_terrain_open",
            "terrain-open.f32le",
            "pi0",
            (NZ + 2, NY + 6, NX + 6),
            0,
            2_783_099_088,
            "bab72a50917e9552c1720e8ccc5262dc70c15b9d270bd00ca85f8b63441ac22b",
        ),
        "th0": Native5mCarrier(
            "auxiliary_terrain_open",
            "terrain-open.f32le",
            "th0",
            (NZ + 2, NY + 6, NX + 6),
            5_566_198_176,
            2_783_099_088,
            "69af17d687e863cf8741f2fafa2eb76d4e5b4e56b9a4f2a89304bd1fddc6eecf",
        ),
        "prs": Native5mCarrier(
            "auxiliary_live_density",
            "live-density.f32le",
            "prs",
            (NZ, NY, NX),
            0,
            2_700_000_000,
            "a6b51a27eddd0dd6ac8ed540d05556ba02a60549b841350d4e6512bc70ec0111",
        ),
        "rf": Native5mCarrier(
            "auxiliary_live_density",
            "live-density.f32le",
            "rf",
            (NZ + 1, NY, NX),
            2_700_000_000,
            2_736_000_000,
            "f4d8c69579425c39cb4b01b3d443cfc24f17d5fad3a996574ddc4cb3e57f912f",
        ),
        "rho": Native5mCarrier(
            "auxiliary_live_density",
            "live-density.f32le",
            "rho",
            (NZ, NY, NX),
            5_436_000_000,
            2_700_000_000,
            "89b614ad1c1a9aaa6ffb32e2435d76fe5a9f6c589c7ca2c65e248bd2c4be8d22",
        ),
    }
)


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(STREAM_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _sha_region(path: Path, carrier: Native5mCarrier) -> str:
    digest = hashlib.sha256()
    remaining = carrier.nbytes
    with path.open("rb") as handle:
        handle.seek(carrier.offset_nbytes)
        while remaining:
            block = handle.read(min(STREAM_CHUNK_BYTES, remaining))
            if not block:
                raise ValueError(f"native carrier is truncated: {carrier.name}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _canonical_json(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA-256 changed")
    value = json.loads(raw)
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if raw != canonical or not isinstance(value, Mapping):
        raise ValueError(f"{label} is not canonical compact JSON")
    return MappingProxyType(dict(value))


def _require_directory(root: Path, inventory: set[str], label: str) -> Path:
    if root.is_symlink():
        raise ValueError(f"{label} directory must not be a symlink")
    resolved = root.resolve(strict=True)
    entries = tuple(resolved.iterdir())
    if (
        not resolved.is_dir()
        or {entry.name for entry in entries} != inventory
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError(f"{label} directory inventory changed")
    return resolved


def _manifest_carriers(manifest: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    value = manifest.get("carriers")
    if not isinstance(value, list):
        raise ValueError(f"{label} carriers must be a list")
    result = {str(item.get("name")): item for item in value if isinstance(item, Mapping)}
    if len(result) != len(value):
        raise ValueError(f"{label} carrier names are not unique")
    return result


def _auxiliary_carriers(manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("native auxiliary outputs must be a list")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for group in outputs:
        if not isinstance(group, Mapping) or not isinstance(group.get("carriers"), list):
            raise ValueError("native auxiliary output group changed")
        file = str(group.get("file"))
        for item in group["carriers"]:
            if not isinstance(item, Mapping):
                raise ValueError("native auxiliary carrier changed")
            key = (file, str(item.get("name")))
            if key in result:
                raise ValueError("native auxiliary carrier names are not unique")
            result[key] = item
    return result


def _require_descriptor(record: Mapping[str, Any], carrier: Native5mCarrier) -> None:
    offset_key = "offset_bytes" if carrier.group == "atmosphere" else "offset_nbytes"
    expected = {
        "name": carrier.name,
        "shape": list(carrier.shape),
        offset_key: carrier.offset_nbytes,
        "nbytes": carrier.nbytes,
        "sha256": carrier.sha256,
        "dtype": "little-endian-f32",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"native carrier descriptor changed: {carrier.name}")
    if not str(record.get("order", "")).startswith("C"):
        raise ValueError(f"native carrier order changed: {carrier.name}")


def _validation_paths(repo: Path) -> Mapping[str, Path]:
    base = repo / "evidence/validation"
    return MappingProxyType(
        {
            "terrain": base / "gm1-real-5m-child-terrain-v2/validation-receipt.json",
            "atmosphere": base
            / "gm1-real-5m-child-atmosphere-v2/validation-receipt.json",
            "static_heights": base
            / "gm1-real-5m-child-static-heights-v2/validation-receipt.json",
            "auxiliary": base
            / "gm1-real-5m-child-auxiliary-stream-v2/validation-receipt.json",
        }
    )


@dataclass(frozen=True, slots=True)
class Native5mClosureInputs:
    roots: Mapping[str, Path]
    manifests: Mapping[str, Mapping[str, Any]]
    receipts: Mapping[str, Mapping[str, Any]]
    arrays: Mapping[str, Any]
    identity: Mapping[str, object]

    def tile(self, spec: Gm1TileSpec) -> Mapping[str, Any]:
        """Pack one exact physical compute window into C-order host arrays."""

        if not isinstance(spec, Gm1TileSpec):
            raise TypeError("spec must be Gm1TileSpec")
        if (spec.nx, spec.ny) != (NX, NY) or spec.halo != 3:
            raise ValueError("native closure tile differs from the admitted 5 m plan")
        np = importlib.import_module("numpy")
        y = slice(spec.cj0, spec.cj0 + spec.cny)
        x = slice(spec.ci0, spec.ci0 + spec.cnx)
        xu = slice(spec.ci0, spec.ci0 + spec.cnx + 1)
        yv = slice(spec.cj0, spec.cj0 + spec.cny + 1)
        base_y = slice(3 + spec.cj0, 3 + spec.cj0 + spec.cny)
        base_x = slice(3 + spec.ci0, 3 + spec.ci0 + spec.cnx)
        result = {
            "terrain_msl": np.ascontiguousarray(self.arrays["terrain_msl"][y, x]),
            "u3d": np.ascontiguousarray(self.arrays["u3d"][:, y, xu]),
            "v3d": np.ascontiguousarray(self.arrays["v3d"][:, yv, x]),
            "w3d": np.ascontiguousarray(self.arrays["w3d"][:, y, x]),
            "rho": np.ascontiguousarray(self.arrays["rho"][:, y, x]),
            "rf": np.ascontiguousarray(self.arrays["rf"][:, y, x]),
            "pi0": np.ascontiguousarray(
                self.arrays["pi0"][1 : NZ + 1, base_y, base_x]
            ),
            "th0": np.ascontiguousarray(
                self.arrays["th0"][1 : NZ + 1, base_y, base_x]
            ),
            "prs": np.ascontiguousarray(self.arrays["prs"][:, y, x]),
            "pp3d": np.ascontiguousarray(self.arrays["pp3d"][:, y, x]),
            "th3d": np.ascontiguousarray(self.arrays["th3d"][:, y, x]),
            "q3d": np.ascontiguousarray(self.arrays["q3d"][:, :, y, x]),
        }
        if tuple(result) != INPUT_NAMES:
            raise RuntimeError("native 5 m closure tile input order changed")
        if any(
            value.dtype != np.dtype("float32")
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
            for value in result.values()
        ):
            raise ValueError("native 5 m closure tile contains an invalid input")
        if bool(np.any(result["q3d"] < np.float32(0.0))):
            raise ValueError("native 5 m closure tile moisture is negative")
        return MappingProxyType(result)


def load_native_5m_closure_inputs(
    *,
    repo: Path,
    terrain: Path,
    atmosphere: Path,
    static_heights: Path,
    auxiliary: Path,
    verify_consumed_bytes: bool = True,
) -> Native5mClosureInputs:
    """Strictly bind and optionally re-hash every consumed source byte."""

    if not isinstance(verify_consumed_bytes, bool):
        raise TypeError("verify_consumed_bytes must be boolean")
    repo = Path(repo).resolve(strict=True)
    roots = {
        "terrain": _require_directory(
            Path(terrain), {"terrain-msl.f32le", "manifest.json", "receipt.json"}, "terrain"
        ),
        "atmosphere": _require_directory(
            Path(atmosphere), {"arena.f32le", "manifest.json", "receipt.json"}, "atmosphere"
        ),
        "static_heights": _require_directory(
            Path(static_heights), {"zh.f32le", "zf.f32le", "manifest.json", "receipt.json"}, "static heights"
        ),
        "auxiliary": _require_directory(
            Path(auxiliary),
            {
                "terrain-open.f32le",
                "live-density.f32le",
                "total-geopotential-zf.f32le",
                "base-wind.f32le",
                "manifest.json",
                "receipt.json",
            },
            "auxiliary",
        ),
    }
    manifests: dict[str, Mapping[str, Any]] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    validation: dict[str, str] = {}
    validation_paths = _validation_paths(repo)
    for name, root in roots.items():
        pins = ARTIFACT_PINS[name]
        manifests[name] = _canonical_json(
            root / "manifest.json", pins["manifest_sha256"], f"{name} manifest"
        )
        receipts[name] = _canonical_json(
            root / "receipt.json", pins["receipt_sha256"], f"{name} receipt"
        )
        validation[name] = _sha_path(validation_paths[name])
        if validation[name] != pins["validation_sha256"]:
            raise ValueError(f"{name} validation receipt identity changed")
    grid = manifests["atmosphere"].get("grid", {})
    if (
        grid.get("nx") != NX
        or grid.get("ny") != NY
        or grid.get("nz") != NZ
        or grid.get("dx_m_f64_bits") != "4014000000000000"
        or grid.get("dy_m_f64_bits") != "4014000000000000"
        or grid.get("ztop_m_f64_bits") != "40cd4c0000000000"
        or grid.get("stretch_grid_sha256")
        != "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"
    ):
        raise ValueError("native 5 m atmosphere grid identity changed")
    atmosphere_records = _manifest_carriers(manifests["atmosphere"], "atmosphere")
    auxiliary_records = _auxiliary_carriers(manifests["auxiliary"])
    for name, carrier in CONSUMED_CARRIERS.items():
        if carrier.group == "terrain":
            record = manifests["terrain"].get("payload", {})
            translated = {
                "name": "terrain_msl",
                "shape": record.get("shape_yx"),
                "offset_nbytes": 0,
                "nbytes": record.get("nbytes"),
                "sha256": record.get("sha256"),
                "dtype": "little-endian-f32",
                "order": "C",
            }
            _require_descriptor(translated, carrier)
        elif carrier.group == "atmosphere":
            _require_descriptor(atmosphere_records[name], carrier)
        else:
            _require_descriptor(auxiliary_records[(carrier.file, name)], carrier)
        path = roots[
            "auxiliary" if carrier.group.startswith("auxiliary") else carrier.group
        ] / carrier.file
        expected_file = PAYLOAD_PINS[carrier.group]
        if (
            path.stat().st_size != expected_file["nbytes"]
            or path.name != expected_file["file"]
        ):
            raise ValueError(f"native payload file identity changed: {carrier.group}")

    verified: dict[str, str] = {}
    verification_started = time.perf_counter()
    if verify_consumed_bytes:
        for name, carrier in CONSUMED_CARRIERS.items():
            path = roots[
                "auxiliary" if carrier.group.startswith("auxiliary") else carrier.group
            ] / carrier.file
            actual = _sha_region(path, carrier)
            if actual != carrier.sha256:
                raise ValueError(f"native consumed carrier bytes changed: {name}")
            verified[name] = actual
    verification_seconds = time.perf_counter() - verification_started

    np = importlib.import_module("numpy")
    arrays: dict[str, Any] = {}
    for name, carrier in CONSUMED_CARRIERS.items():
        path = roots[
            "auxiliary" if carrier.group.startswith("auxiliary") else carrier.group
        ] / carrier.file
        arrays[name] = np.memmap(
            path,
            dtype=np.dtype("<f4"),
            mode="r",
            offset=carrier.offset_nbytes,
            shape=carrier.shape,
            order="C",
        )
        arrays[name].flags.writeable = False
    consumed_nbytes = sum(carrier.nbytes for carrier in CONSUMED_CARRIERS.values())
    identity: dict[str, object] = {
        "schema": SCHEMA,
        "status": STATUS,
        "grid": {"nx": NX, "ny": NY, "nz": NZ, "dx_m": 5.0, "dy_m": 5.0},
        "artifacts": {
            name: {
                "directory": str(root),
                **dict(ARTIFACT_PINS[name]),
            }
            for name, root in roots.items()
        },
        "consumed_carriers": {
            name: carrier.record() for name, carrier in CONSUMED_CARRIERS.items()
        },
        "consumed_nbytes": consumed_nbytes,
        "full_domain_duplicate_bytes": 0,
        "independent_consumed_byte_verification": {
            "performed": verify_consumed_bytes,
            "verified_carrier_count": len(verified),
            "verified_nbytes": consumed_nbytes if verify_consumed_bytes else 0,
            "verification_seconds": verification_seconds,
            "sha256": verified,
        },
        "static_height_disposition": (
            "provenance-bound prior oracle; not read because closure rebuilds "
            "tile metrics from exact terrain and sealed vertical profiles"
        ),
        "synthetic_detail": False,
        "synthetic_translation": False,
        "claim_limit": CLAIM_LIMIT,
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return Native5mClosureInputs(
        MappingProxyType(roots),
        MappingProxyType(manifests),
        MappingProxyType(receipts),
        MappingProxyType(arrays),
        MappingProxyType(identity),
    )


__all__ = (
    "ARTIFACT_PINS",
    "CLAIM_LIMIT",
    "CONSUMED_CARRIERS",
    "NX",
    "NY",
    "NZ",
    "Native5mCarrier",
    "Native5mClosureInputs",
    "PAYLOAD_PINS",
    "SCHEMA",
    "STATUS",
    "load_native_5m_closure_inputs",
)
