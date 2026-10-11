"""Strict CUDA-cold admission of the real fixed-50 m child atmosphere.

Rust owns the CM1 restart-grid interpolation and the 840.6 MB arena.  This
module does not reinterpret or rewrite those words.  It verifies the exact
publication, exposes read-only memory-mapped carrier views, and keeps closure,
hydrostatic, boundary, and forecast readiness explicitly false.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


GM1_CHILD_ATMOSPHERE_SCHEMA = "gm1.source-coordinate-50m-child-atmosphere/v1"
GM1_CHILD_ATMOSPHERE_MANIFEST_SCHEMA = (
    "gm1.source-coordinate-50m-child-atmosphere-manifest/v1"
)
GM1_CHILD_ATMOSPHERE_ABI = (
    "gm1.source-coordinate-50m-child-atmosphere-arena-f32/v1"
)
GM1_CHILD_ATMOSPHERE_STATUS = "PASS_REAL_CM1_RESTART_GRID_CHILD_ATMOSPHERE"
GM1_CHILD_ATMOSPHERE_MANIFEST_SHA256 = (
    "88798478087b2c3d8d5a66a956e891ad5936741c3c47285bb243f709a90e30b8"
)
GM1_CHILD_ATMOSPHERE_ARENA_SHA256 = (
    "59aa935cee8f7a69187af97ead3e2b29fd50e6ef665ca6b22ccb07bc9a34c3b7"
)
GM1_CHILD_ATMOSPHERE_RECEIPT_SHA256 = (
    "17ca4f48a353d13ec69b6e45ca2e6180e7c8c125c60cf69a4b9d7fa702580c6c"
)
GM1_CHILD_ATMOSPHERE_PRODUCER_SOURCE_SHA256 = (
    "5999371aba40bf1b3aaf1af00230f0a883bf7d9d51ff681455db5ae666a3e55b"
)
GM1_CHILD_ATMOSPHERE_PRODUCER_EXECUTABLE_SHA256 = (
    "79c7305db869443e51eeaa4c2240224346dc4a4d39308ce4c5ac2e7a692f16b5"
)
GM1_CHILD_ATMOSPHERE_ARENA_NBYTES = 840_600_000
GM1_CHILD_ATMOSPHERE_CLAIM_LIMIT = (
    "strict installed-byte admission and read-only memory mapping of the exact "
    "Rust-produced 28-carrier real pdefq-clean T+300 to fixed-50 m child "
    "atmospheric arena only; closure TKE, terrain-varying hydrostatic/base "
    "state, live density, open-boundary/radiation history, geopotential, child "
    "integration, forecast, storm, and tornado claims are excluded"
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_RESTART_READ_SHA256 = (
    "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a"
)
CM1_INIT3D_SHA256 = (
    "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ARENA_FILES = frozenset({"arena.f32le", "manifest.json", "receipt.json"})
_ZERO_MASS_SHA256 = (
    "5a4a3c8e7475e11153c3bbeab9f407aa70157d38b8dc03f1b9bf113276a7e939"
)


@dataclass(frozen=True, slots=True)
class _ExpectedCarrier:
    shape: tuple[int, ...]
    disposition: str
    source_carrier: str | None
    stagger: str
    below_source_support_words: int


_EXPECTED_CARRIERS = MappingProxyType(
    {
        **{
            name: _ExpectedCarrier(
                (75, 300, 300),
                "ARWEN_PERSISTENCE_POLICY_CM1_ARITHMETIC",
                name,
                "scalar",
                450_000,
            )
            for name in (
                "component.effc",
                "component.effi",
                "component.effs",
                "component.h_diabatic",
            )
        },
        **{
            name: _ExpectedCarrier(
                (300, 300),
                "ARWEN_PERSISTENCE_POLICY_CM1_ARITHMETIC",
                name,
                "surface",
                0,
            )
            for name in (
                "component.mp_graupelnc",
                "component.mp_graupelncv",
                "component.mp_rainnc",
                "component.mp_rainncv",
                "component.mp_snownc",
                "component.mp_snowncv",
                "component.mp_sr",
            )
        },
        **{
            name: _ExpectedCarrier(
                (75, 300, 300),
                "CM1_INTERP_ON_RESTART_POSITIVE_ZERO",
                None,
                "scalar",
                0,
            )
            for name in ("held.qcten", "held.qpten", "held.qtten", "held.qvten")
        },
        "pp3d": _ExpectedCarrier(
            (75, 300, 300), "CM1_A_TO_CURRENT_RAW_COPY", "ppi", "scalar", 450_000
        ),
        "ppi": _ExpectedCarrier(
            (75, 300, 300), "CM1_RESTART_GRID_INTERPOLATE", "ppi", "scalar", 450_000
        ),
        "ppx": _ExpectedCarrier(
            (75, 300, 300), "CM1_RESTART_GRID_INTERPOLATE", "ppx", "scalar", 450_000
        ),
        "q3d": _ExpectedCarrier(
            (6, 75, 300, 300),
            "CM1_A_TO_CURRENT_RAW_COPY",
            "qa",
            "scalar",
            2_700_000,
        ),
        "qa": _ExpectedCarrier(
            (6, 75, 300, 300),
            "CM1_RESTART_GRID_INTERPOLATE",
            "qa",
            "scalar",
            2_700_000,
        ),
        "th3d": _ExpectedCarrier(
            (75, 300, 300), "CM1_A_TO_CURRENT_RAW_COPY", "tha", "scalar", 450_000
        ),
        "tha": _ExpectedCarrier(
            (75, 300, 300), "CM1_RESTART_GRID_INTERPOLATE", "tha", "scalar", 450_000
        ),
        "u3d": _ExpectedCarrier(
            (75, 300, 301), "CM1_A_TO_CURRENT_RAW_COPY", "ua", "x-face", 451_500
        ),
        "ua": _ExpectedCarrier(
            (75, 300, 301), "CM1_RESTART_GRID_INTERPOLATE", "ua", "x-face", 451_500
        ),
        "v3d": _ExpectedCarrier(
            (75, 301, 300), "CM1_A_TO_CURRENT_RAW_COPY", "va", "y-face", 451_500
        ),
        "va": _ExpectedCarrier(
            (75, 301, 300), "CM1_RESTART_GRID_INTERPOLATE", "va", "y-face", 451_500
        ),
        "w3d": _ExpectedCarrier(
            (76, 300, 300), "CM1_A_TO_CURRENT_RAW_COPY", "wa", "z-face", 0
        ),
        "wa": _ExpectedCarrier(
            (76, 300, 300), "CM1_RESTART_GRID_INTERPOLATE", "wa", "z-face", 0
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Gm1ChildAtmosphereCarrier:
    name: str
    shape: tuple[int, ...]
    offset_bytes: int
    nbytes: int
    sha256: str
    disposition: str
    source_carrier: str | None
    source_first_level_sha256: str | None
    below_source_support_words: int
    stagger: str


@dataclass(frozen=True, slots=True)
class LoadedGm1ChildAtmosphere:
    directory: Path
    manifest_path: Path
    arena_path: Path
    receipt_path: Path
    carriers: Mapping[str, Gm1ChildAtmosphereCarrier]
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]
    manifest_sha256: str
    arena_sha256: str
    receipt_sha256: str
    closure_tke_ready: bool = False
    hydrostatic_base_ready: bool = False
    open_boundary_ready: bool = False

    def field(self, name: str) -> Any:
        """Return one exact read-only ``numpy.memmap`` carrier view."""

        try:
            carrier = self.carriers[name]
        except KeyError as error:
            raise KeyError(f"unknown child-atmosphere carrier {name!r}") from error
        np = importlib.import_module("numpy")
        return np.memmap(
            self.arena_path,
            dtype=np.dtype("<f4"),
            mode="r",
            offset=carrier.offset_bytes,
            shape=carrier.shape,
            order="C",
        )


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
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"child atmosphere duplicates JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"child atmosphere contains non-finite JSON token {token}")


def _read_canonical_json(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _sha(expected_sha256, f"expected {label} SHA-256"):
        raise ValueError(f"{label} SHA-256 is {actual}; expected {expected_sha256}")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    canonical = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    if raw != canonical:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _hash_arena_and_carriers(
    path: Path, carriers: tuple[Gm1ChildAtmosphereCarrier, ...]
) -> tuple[str, Mapping[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("child atmosphere arena must be a regular non-symlink file")
    expected_nbytes = sum(carrier.nbytes for carrier in carriers)
    if expected_nbytes != GM1_CHILD_ATMOSPHERE_ARENA_NBYTES:
        raise ValueError(
            "child atmosphere carrier layout no longer exhausts the admitted "
            f"{GM1_CHILD_ATMOSPHERE_ARENA_NBYTES}-byte arena of this child ABI"
        )
    if path.stat().st_size != expected_nbytes:
        raise ValueError("child atmosphere arena byte count changed")
    arena_digest = hashlib.sha256()
    carrier_hashes: dict[str, str] = {}
    with path.open("rb") as stream:
        cursor = 0
        for carrier in carriers:
            if carrier.offset_bytes != cursor:
                raise ValueError("child atmosphere carrier offsets are not contiguous")
            remaining = carrier.nbytes
            digest = hashlib.sha256()
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("child atmosphere arena ended inside a carrier")
                digest.update(block)
                arena_digest.update(block)
                remaining -= len(block)
            carrier_hashes[carrier.name] = digest.hexdigest()
            cursor += carrier.nbytes
        if stream.read(1):
            raise ValueError("child atmosphere arena has trailing bytes")
    if cursor != expected_nbytes:
        raise ValueError("child atmosphere carrier bytes do not cover the arena")
    return arena_digest.hexdigest(), MappingProxyType(carrier_hashes)


def _validate_authority(manifest: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    for value, label in ((manifest, "manifest"), (receipt, "receipt")):
        authority = _exact(
            value["authority"],
            {"init3d", "model_source", "restart_read", "tarball_sha256"},
            f"child atmosphere {label} authority",
        )
        init3d = _exact(
            authority["init3d"],
            {"copy_lines", "sha256"},
            f"child atmosphere {label} init3d authority",
        )
        restart = _exact(
            authority["restart_read"],
            {
                "bracket_lines",
                "held_zero_lines",
                "interpolation_lines",
                "sha256",
                "tke_exclusion_lines",
            },
            f"child atmosphere {label} restart authority",
        )
        if (
            authority["model_source"] != "CM1 r21.1"
            or authority["tarball_sha256"] != AUTHORITY_TARBALL_SHA256
            or init3d
            != {"copy_lines": "2280-2333", "sha256": CM1_INIT3D_SHA256}
            or restart
            != {
                "bracket_lines": "539-609",
                "held_zero_lines": "2657-2664",
                "interpolation_lines": "3302-3465",
                "sha256": CM1_RESTART_READ_SHA256,
                "tke_exclusion_lines": "1206-1211",
            }
        ):
            raise ValueError(f"child atmosphere {label} authority changed")


def _parse_carriers(value: object) -> tuple[Gm1ChildAtmosphereCarrier, ...]:
    if not isinstance(value, list) or len(value) != len(_EXPECTED_CARRIERS):
        raise ValueError("child atmosphere must contain exactly 28 carriers")
    result: list[Gm1ChildAtmosphereCarrier] = []
    for raw, expected_name in zip(value, sorted(_EXPECTED_CARRIERS), strict=True):
        item = _exact(
            raw,
            {
                "authority",
                "below_source_support_words",
                "disposition",
                "dtype",
                "name",
                "nbytes",
                "offset_bytes",
                "order",
                "sha256",
                "shape",
                "source_carrier",
                "source_first_level_sha256",
                "stagger",
            },
            f"child atmosphere carrier {expected_name}",
        )
        expected = _EXPECTED_CARRIERS[expected_name]
        shape = tuple(item["shape"]) if isinstance(item["shape"], list) else ()
        nbytes = math.prod(expected.shape) * 4
        if (
            item["name"] != expected_name
            or shape != expected.shape
            or item["nbytes"] != nbytes
            or item["dtype"] != "little-endian-f32"
            or item["order"] != "C"
            or item["disposition"] != expected.disposition
            or item["source_carrier"] != expected.source_carrier
            or item["stagger"] != expected.stagger
            or item["below_source_support_words"]
            != expected.below_source_support_words
        ):
            raise ValueError(f"child atmosphere carrier {expected_name} contract changed")
        source_first = item["source_first_level_sha256"]
        if expected.source_carrier is None:
            if source_first is not None or item["sha256"] != _ZERO_MASS_SHA256:
                raise ValueError(f"held carrier {expected_name} is not exact positive zero")
        else:
            _sha(source_first, f"carrier {expected_name} source-first-level SHA-256")
        result.append(
            Gm1ChildAtmosphereCarrier(
                name=expected_name,
                shape=expected.shape,
                offset_bytes=item["offset_bytes"],
                nbytes=nbytes,
                sha256=_sha(item["sha256"], f"carrier {expected_name} SHA-256"),
                disposition=expected.disposition,
                source_carrier=expected.source_carrier,
                source_first_level_sha256=source_first,
                below_source_support_words=expected.below_source_support_words,
                stagger=expected.stagger,
            )
        )
    return tuple(result)


def _validate_manifest(
    manifest: Mapping[str, Any], carriers: tuple[Gm1ChildAtmosphereCarrier, ...]
) -> None:
    manifest = _exact(
        manifest,
        {
            "abi",
            "arena",
            "artifact_schema",
            "authority",
            "carriers",
            "claim_limit",
            "dispositions",
            "grid",
            "model",
            "parent",
            "producer",
            "schema",
            "static_heights",
            "status",
            "synthetic_translation",
            "transform",
        },
        "child atmosphere manifest",
    )
    if (
        manifest["schema"] != GM1_CHILD_ATMOSPHERE_MANIFEST_SCHEMA
        or manifest["artifact_schema"] != GM1_CHILD_ATMOSPHERE_SCHEMA
        or manifest["abi"] != GM1_CHILD_ATMOSPHERE_ABI
        or manifest["status"] != GM1_CHILD_ATMOSPHERE_STATUS
        or manifest["model"] != "GM1"
        or manifest["synthetic_translation"] is not False
    ):
        raise ValueError("child atmosphere model/schema/status changed")
    arena = _exact(
        manifest["arena"],
        {"carrier_count", "dtype", "file", "nbytes", "order", "sha256"},
        "child atmosphere arena metadata",
    )
    # The arena descriptor is checked against the carrier layout this reader
    # indexes, not against one historical digest: a wrong byte count or carrier
    # count would have every downstream memmap read past the end of one carrier
    # and into the next.  Which arena it is, is the caller's expectation.
    if arena != {
        "carrier_count": len(carriers),
        "dtype": "little-endian-f32",
        "file": "arena.f32le",
        "nbytes": sum(carrier.nbytes for carrier in carriers),
        "order": "C; carrier-major in manifest order",
        "sha256": _sha(arena.get("sha256"), "child atmosphere arena SHA-256"),
    }:
        raise ValueError("child atmosphere arena metadata changed")
    parent = _exact(
        manifest["parent"],
        {
            "checkpoint_artifact_sha256",
            "checkpoint_contract_sha256",
            "checkpoint_raw_arena_sha256",
            "operator_manifest_sha256",
            "plan_schema",
            "plan_sha256",
            "source_product_manifest_sha256",
        },
        "child atmosphere parent",
    )
    # The parent is recorded, not required to be the first one admitted.  What
    # must hold is that it is recordable and recorded under the plan schema
    # this chain reads, so the composing gate can require every producer of one
    # child to name the same parent; a producer that named no parent, or named
    # it under a schema nothing can re-read, could not be challenged at all.
    for key in (
        "checkpoint_artifact_sha256",
        "checkpoint_contract_sha256",
        "checkpoint_raw_arena_sha256",
        "operator_manifest_sha256",
        "plan_sha256",
        "source_product_manifest_sha256",
    ):
        _sha(parent[key], f"child atmosphere parent {key}")
    if parent["plan_schema"] != "gm1.source-coordinate-50m-child-remap-plan/v1":
        raise ValueError("child atmosphere parent plan schema changed")
    producer = _exact(
        manifest["producer"],
        {"executable_sha256", "source_sha256"},
        "child atmosphere producer",
    )
    for key in ("executable_sha256", "source_sha256"):
        _sha(producer[key], f"child atmosphere producer {key}")
    grid = _exact(
        manifest["grid"],
        {
            "dx_m_f64_bits",
            "dy_m_f64_bits",
            "nx",
            "ny",
            "nz",
            "stretch_abi",
            "stretch_grid_sha256",
            "stretch_schema",
            "terrain_fixed_domain",
            "ztop_m_f64_bits",
        },
        "child atmosphere grid",
    )
    if (
        grid["nx"],
        grid["ny"],
        grid["nz"],
        grid["dx_m_f64_bits"],
        grid["dy_m_f64_bits"],
        grid["ztop_m_f64_bits"],
        grid["terrain_fixed_domain"],
    ) != (300, 300, 75, "4049000000000000", "4049000000000000", "40cd4c0000000000", True):
        raise ValueError("child atmosphere grid changed")
    heights = _exact(
        manifest["static_heights"],
        {
            "abi",
            "manifest_sha256",
            "operator_commit",
            "source_terrain_manifest_sha256",
            "source_terrain_payload_sha256",
            "source_terrain_receipt_sha256",
            "validation_receipt_sha256",
            "zf_sha256",
            "zh_sha256",
        },
        "child atmosphere static heights",
    )
    # Recorded, not pinned: the static heights and terrain this atmosphere was
    # interpolated onto are named here so the composing gate can require them
    # to be the same artifacts it admits.  Requiring them to be one historical
    # set here would refuse every child built from another parent while
    # checking nothing the composing gate does not already check better.
    for key in (
        "manifest_sha256",
        "validation_receipt_sha256",
        "zh_sha256",
        "zf_sha256",
        "source_terrain_manifest_sha256",
        "source_terrain_payload_sha256",
        "source_terrain_receipt_sha256",
    ):
        _sha(heights[key], f"child atmosphere static-height {key}")
    dispositions = _exact(
        manifest["dispositions"],
        {
            "child_boundary_initialize",
            "child_closure_initialize",
            "child_hydrostatic_reconstruct",
            "closure_tkea_reason",
            "held_tendency_reason",
        },
        "child atmosphere dispositions",
    )
    if (
        dispositions["child_boundary_initialize"]
        != ["radbce", "radbcn", "radbcs", "radbcw"]
        or dispositions["child_closure_initialize"]
        != ["closure.tkea", "closure.tke3d", "closure.tketen"]
        or dispositions["child_hydrostatic_reconstruct"]
        != ["prs", "rho", "rr", "rf"]
        or "1206-1211" not in dispositions["closure_tkea_reason"]
        or "2657-2664" not in dispositions["held_tendency_reason"]
    ):
        raise ValueError("child atmosphere downstream dispositions changed")
    transform = _exact(
        manifest["transform"],
        {
            "below_support",
            "face_array_coordinates",
            "horizontal_mass_coordinates",
            "interpolation",
            "rounding",
            "strict_search",
            "vertical_mass_coordinate",
            "vertical_w_coordinate",
        },
        "child atmosphere transform",
    )
    if (
        transform["interpolation"]
        != "CM1 interp_on_restart normalized sigma/sigmaf trilinear"
        or transform["vertical_mass_coordinate"] != "sigma"
        or transform["vertical_w_coordinate"] != "sigmaf"
        or "duplicates old level 1" not in transform["below_support"]
        or "binary32" not in transform["rounding"]
    ):
        raise ValueError("child atmosphere interpolation contract changed")
    copies = {
        "pp3d": "ppi",
        "q3d": "qa",
        "th3d": "tha",
        "u3d": "ua",
        "v3d": "va",
        "w3d": "wa",
    }
    by_name = {carrier.name: carrier for carrier in carriers}
    for current, prognostic in copies.items():
        if by_name[current].sha256 != by_name[prognostic].sha256:
            raise ValueError(f"A-to-current pair {prognostic}/{current} differs")


def _validate_receipt(
    receipt: Mapping[str, Any],
    carriers: tuple[Gm1ChildAtmosphereCarrier, ...],
    manifest: Mapping[str, Any],
) -> None:
    receipt = _exact(
        receipt,
        {
            "abi",
            "all_finite",
            "arena_nbytes",
            "arena_sha256",
            "authority",
            "carrier_count",
            "carrier_sha256",
            "claim_limit",
            "manifest_sha256",
            "model",
            "parent_checkpoint_sha256",
            "parent_plan_sha256",
            "producer_executable_sha256",
            "producer_source_sha256",
            "schema",
            "static_heights_manifest_sha256",
            "status",
            "synthetic_translation",
        },
        "child atmosphere receipt",
    )
    # The receipt anchors the manifest beside it, so the two must describe one
    # artifact: same arena, same producer, same parent, same static heights.
    # That is what makes this publication self-consistent and therefore
    # consumable whichever parent it was interpolated from; an artifact whose
    # receipt and manifest disagreed would hand a consumer one document's
    # carrier offsets over another document's bytes.
    if (
        receipt["schema"] != GM1_CHILD_ATMOSPHERE_SCHEMA
        or receipt["abi"] != GM1_CHILD_ATMOSPHERE_ABI
        or receipt["status"] != GM1_CHILD_ATMOSPHERE_STATUS
        or receipt["model"] != "GM1"
        or receipt["all_finite"] is not True
        or receipt["synthetic_translation"] is not False
        or receipt["carrier_count"] != len(carriers)
        or receipt["arena_nbytes"] != sum(carrier.nbytes for carrier in carriers)
        or receipt["arena_sha256"] != manifest["arena"]["sha256"]
        or receipt["arena_nbytes"] != manifest["arena"]["nbytes"]
        or receipt["parent_checkpoint_sha256"]
        != manifest["parent"]["checkpoint_artifact_sha256"]
        or receipt["parent_plan_sha256"] != manifest["parent"]["plan_sha256"]
        or receipt["producer_source_sha256"] != manifest["producer"]["source_sha256"]
        or receipt["producer_executable_sha256"]
        != manifest["producer"]["executable_sha256"]
        or receipt["static_heights_manifest_sha256"]
        != manifest["static_heights"]["manifest_sha256"]
    ):
        raise ValueError("child atmosphere receipt identity/status changed")
    _sha(receipt["manifest_sha256"], "child atmosphere receipt manifest_sha256")
    summaries = receipt["carrier_sha256"]
    if not isinstance(summaries, list) or len(summaries) != len(carriers):
        raise ValueError("child atmosphere receipt carrier summary changed")
    for summary, carrier in zip(summaries, carriers, strict=True):
        summary = _exact(
            summary,
            {
                "below_source_support_words",
                "name",
                "sha256",
                "source_first_level_sha256",
            },
            f"child atmosphere receipt carrier {carrier.name}",
        )
        if summary != {
            "below_source_support_words": carrier.below_source_support_words,
            "name": carrier.name,
            "sha256": carrier.sha256,
            "source_first_level_sha256": carrier.source_first_level_sha256,
        }:
            raise ValueError(
                f"child atmosphere receipt carrier {carrier.name} differs from manifest"
            )


def load_gm1_child_atmosphere(
    directory: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_arena_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
) -> LoadedGm1ChildAtmosphere:
    """Admit and memory-map one qa-sealed Rust child-atmosphere publication.

    With no expected digests the artifact is admitted by the identity its own
    receipt records: the receipt names the manifest and the arena that are
    present, and every carrier digest, extent and disposition in the manifest
    must be the bytes on disk.  A child atmosphere interpolated from a parent
    at another completed step is then loadable instead of refused for not
    being the first one this project published.

    A caller that knows which publication it wants - the child-step composer,
    which takes them from the full-state envelope it is building - passes the
    digests and the artifact must be exactly those bytes.
    """

    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("child atmosphere directory must not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("child atmosphere must be a directory")
    inventory = {entry.name for entry in root.iterdir()}
    if inventory != _ARENA_FILES or any(entry.is_symlink() for entry in root.iterdir()):
        raise ValueError(
            "child atmosphere directory must contain exactly arena.f32le, "
            "manifest.json, and receipt.json as regular non-symlink files"
        )
    manifest_path = root / "manifest.json"
    arena_path = root / "arena.f32le"
    receipt_path = root / "receipt.json"
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    if expected_receipt_sha256 is None:
        expected_receipt_sha256 = receipt_sha256
    receipt = _read_canonical_json(
        receipt_path, expected_receipt_sha256, "child atmosphere receipt"
    )
    # The receipt is what anchors this publication, so when the caller names no
    # digests the manifest and arena expectations come from it.  Reading them
    # here rather than from a module constant is the difference between "this
    # artifact is internally consistent" and "this artifact is the first child
    # this project ever published".
    if expected_manifest_sha256 is None:
        expected_manifest_sha256 = _sha(
            receipt.get("manifest_sha256"), "child atmosphere receipt manifest_sha256"
        )
    if expected_arena_sha256 is None:
        expected_arena_sha256 = _sha(
            receipt.get("arena_sha256"), "child atmosphere receipt arena_sha256"
        )
    manifest = _read_canonical_json(
        manifest_path, expected_manifest_sha256, "child atmosphere manifest"
    )
    _validate_authority(manifest, receipt)
    carriers = _parse_carriers(manifest.get("carriers"))
    _validate_manifest(manifest, carriers)
    if receipt["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError(
            "child atmosphere manifest bytes are not the manifest its own receipt "
            "anchors; the published carrier layout and the published receipt "
            "describe different artifacts"
        )
    _validate_receipt(receipt, carriers, manifest)
    arena_sha256, carrier_hashes = _hash_arena_and_carriers(arena_path, carriers)
    if arena_sha256 != _sha(expected_arena_sha256, "expected child arena SHA-256"):
        raise ValueError(
            f"child atmosphere arena SHA-256 is {arena_sha256}; "
            f"expected {expected_arena_sha256}"
        )
    for carrier in carriers:
        if carrier_hashes[carrier.name] != carrier.sha256:
            raise ValueError(f"child atmosphere carrier {carrier.name} bytes changed")
    by_name = MappingProxyType({carrier.name: carrier for carrier in carriers})
    loaded = LoadedGm1ChildAtmosphere(
        directory=root,
        manifest_path=manifest_path,
        arena_path=arena_path,
        receipt_path=receipt_path,
        carriers=by_name,
        manifest=MappingProxyType(dict(manifest)),
        receipt=MappingProxyType(dict(receipt)),
        manifest_sha256=expected_manifest_sha256,
        arena_sha256=expected_arena_sha256,
        receipt_sha256=expected_receipt_sha256,
    )
    np = importlib.import_module("numpy")
    if bool(np.any(loaded.field("qa") < np.float32(0.0))):
        raise ValueError("qa contains a negative word after the Rust qa-seal gate")
    return loaded


__all__ = (
    "GM1_CHILD_ATMOSPHERE_ABI",
    "GM1_CHILD_ATMOSPHERE_ARENA_NBYTES",
    "GM1_CHILD_ATMOSPHERE_ARENA_SHA256",
    "GM1_CHILD_ATMOSPHERE_CLAIM_LIMIT",
    "GM1_CHILD_ATMOSPHERE_MANIFEST_SCHEMA",
    "GM1_CHILD_ATMOSPHERE_MANIFEST_SHA256",
    "GM1_CHILD_ATMOSPHERE_PRODUCER_EXECUTABLE_SHA256",
    "GM1_CHILD_ATMOSPHERE_PRODUCER_SOURCE_SHA256",
    "GM1_CHILD_ATMOSPHERE_RECEIPT_SHA256",
    "GM1_CHILD_ATMOSPHERE_SCHEMA",
    "GM1_CHILD_ATMOSPHERE_STATUS",
    "Gm1ChildAtmosphereCarrier",
    "LoadedGm1ChildAtmosphere",
    "load_gm1_child_atmosphere",
)
