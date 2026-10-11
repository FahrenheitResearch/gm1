"""CM1 r21.1 terrain-specialized all-open boundary operators.

This module seals the smallest source-adjacent terrain/open seam needed by
``psolver=3``.  It implements the physical-domain part of ``bcwsfc`` and the
four terrain branches of ``ssopenbc*`` with CM1's native float32 operation
order.  The shipped Rust terrain-static arena is loaded as an opaque,
hash-bound collection of non-aliasing C-order carriers.

The current Rust arena intentionally contains physical-domain metrics only
and a horizontally broadcast base-density policy.  Consequently this module
admits operator validation, but it refuses product admission: source-backed
terrain halos and a terrain-consistent hydrostatic base state are separate
required seams.  Nothing here is a forecast or a full terrain trajectory.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.errors import refuse


CUDA_TERRAIN_OPEN_BOUNDARY_SCHEMA = "gpuwm-cm1.cuda-terrain-open-boundary/v1"
CUDA_TERRAIN_OPEN_BOUNDARY_ABI = "gpuwm-cm1.cuda-terrain-open-boundary-kernels/v1"
CUDA_TERRAIN_OPEN_BOUNDARY_THREADS = 128

TERRAIN_STATIC_SCHEMA = "gpuwm-cm1.terrain-static-export/v1"
TERRAIN_STATIC_ABI = "gpuwm-cm1.terrain-static-carriers/v1"
TERRAIN_STATIC_REQUIRED = (
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
)
TERRAIN_STATIC_SOURCE_ARTIFACT_SHA256 = (
    "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f"
)
TERRAIN_STATIC_SOURCE_CONTRACT_SHA256 = (
    "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12"
)
TERRAIN_STATIC_CROP_MANIFEST_SHA256 = (
    "3969edef2849d350d817a21012bbf504f47bcd18bae44bd1467cc338f4cf14a5"
)
TERRAIN_STATIC_CROP_PAYLOAD_SHA256 = (
    "f2991ed6c805e8e172a7999a480ad7b825df9dc637476bb540c81b2e2d08b7a3"
)
TERRAIN_STATIC_FULL_MANIFEST_SHA256 = (
    "1ba7401dcb4274eb2263e90dbc4fdf95ccc91a71fe027e84f805c3f504ab95e7"
)
TERRAIN_STATIC_FULL_PAYLOAD_SHA256 = (
    "9befe5e9cfacd8e12647994f35ea852dbe00e17a70c70a09b5cd7220b81f3cbc"
)

SOURCE_ANCHORS = {
    "bcwsfc": "src/bc.F:1151-1178",
    "ssopenbcw": "src/bc.F:2392-2459",
    "ssopenbce": "src/bc.F:2467-2534",
    "ssopenbcs": "src/bc.F:2542-2609",
    "ssopenbcn": "src/bc.F:2617-2684",
    "sound_open_cadence": "src/sound.F:253-302",
    "terrain_coordinates": "src/init_terrain.F:175-297",
    "terrain_vertical_metrics": "src/param.F:8464-8492;8603-8627",
    "terrain_fornberg": "src/param.F:8988-9052",
    "terrain_base_density": "src/base.F:2160-2190",
}

CUDA_TERRAIN_OPEN_BOUNDARY_CLAIM_LIMIT = (
    "CM1 r21.1 serial terrain physical-domain bcwsfc and all four terrain "
    "ssopenbc normal-velocity operators for psolver=3, boundaries "
    "[2,2,2,2], uniform dx/dy/dz and source Fornberg/interior vertical "
    "interpolation; no source halo packer, terrain-consistent hydrostatic "
    "base admission, pressure rebalance, complete sound/solve2/advection, "
    "forecast, product, or performance claim"
)


@dataclass(frozen=True, slots=True)
class TerrainOpenSelection:
    psolver: int = 3
    west_boundary: int = 2
    east_boundary: int = 2
    south_boundary: int = 2
    north_boundary: int = 2
    bottom_boundary: int = 1
    top_boundary: int = 1
    hadvordrs: int = 5
    hadvordrv: int = 5
    terrain: bool = True
    mpi: bool = False
    axisymmetric: int = 0
    stretch_x: bool = False
    stretch_y: bool = False
    restricted_outflow: int = 0
    ngxy: int = CM1_OPEN_NGXY
    base_state_policy: str = "broadcast-validation-only"
    source_backed_halos: bool = False
    product_admission: bool = False


DEFAULT_TERRAIN_OPEN_SELECTION = TerrainOpenSelection()
_EXPECTED_SELECTION = {
    "dynamics.psolver": ("psolver", 3),
    "boundaries.west": ("west_boundary", 2),
    "boundaries.east": ("east_boundary", 2),
    "boundaries.south": ("south_boundary", 2),
    "boundaries.north": ("north_boundary", 2),
    "boundaries.bottom": ("bottom_boundary", 1),
    "boundaries.top": ("top_boundary", 1),
    "advection.hadvordrs": ("hadvordrs", 5),
    "advection.hadvordrv": ("hadvordrv", 5),
    "grid.terrain": ("terrain", True),
    "runtime.mpi": ("mpi", False),
    "grid.axisymmetric": ("axisymmetric", 0),
    "grid.stretch_x": ("stretch_x", False),
    "grid.stretch_y": ("stretch_y", False),
    "boundaries.restricted_outflow": ("restricted_outflow", 0),
    "grid.ngxy": ("ngxy", CM1_OPEN_NGXY),
    "base_state.policy": ("base_state_policy", "broadcast-validation-only"),
    "base_state.source_backed_halos": ("source_backed_halos", False),
    "admission.product": ("product_admission", False),
}


def validate_terrain_open_selection(selection: TerrainOpenSelection) -> None:
    """Refuse every branch not covered by the sealed validation operator."""

    if not isinstance(selection, TerrainOpenSelection):
        raise TypeError("selection must be a TerrainOpenSelection")
    for option, (attribute, expected) in _EXPECTED_SELECTION.items():
        value = getattr(selection, attribute)
        if type(value) is not type(expected) or value != expected:
            if attribute in {
                "base_state_policy",
                "source_backed_halos",
                "product_admission",
            }:
                reason = (
                    "the supplied Rust terrain arena has a horizontally broadcast "
                    "base-density policy and physical-domain metrics only; product "
                    "admission requires a source-backed terrain hydrostatic base and "
                    "authority-valid halo packer"
                )
                remedy = (
                    "supply and separately validate the combined hydrostatic-base and "
                    "terrain/open halo ABI before requesting product admission"
                )
                code = "CM1-TERRAIN-PRODUCT-SEAM-MISSING"
            else:
                reason = (
                    f"this source oracle covers only {option}={expected!r}; changing "
                    "it selects unmeasured CM1 boundary, metric, or advection code"
                )
                remedy = "use the sealed terrain all-open 5/5 validation profile"
                code = "CM1-TERRAIN-OPEN-UNSUPPORTED"
            refuse(
                code=code,
                gate="dry-dynamics",
                option=option,
                value=value,
                reason=reason,
                remedy=remedy,
            )


@dataclass(frozen=True, slots=True)
class TerrainStaticCarrier:
    name: str
    shape: tuple[int, ...]
    offset_nbytes: int
    nbytes: int
    sha256: str
    role: str


@dataclass(frozen=True, slots=True)
class LoadedTerrainStaticExport:
    directory: Path
    manifest_path: Path
    payload_path: Path
    manifest_sha256: str
    payload_sha256: str
    payload_nbytes: int
    source_artifact_sha256: str
    source_contract_sha256: str
    crop_origin_yx: tuple[int, int]
    crop_count_yx: tuple[int, int]
    source_shape_zyx: tuple[int, int, int]
    target_dx_m: float
    target_dy_m: float
    target_dz_m: float
    carriers: tuple[TerrainStaticCarrier, ...]
    arrays: Mapping[str, Any]
    base_policy: str
    is_full_domain: bool

    def carrier(self, name: str) -> TerrainStaticCarrier:
        for carrier in self.carriers:
            if carrier.name == name:
                return carrier
        raise KeyError(name)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _f64_from_bits(text: str) -> float:
    import struct

    if not isinstance(text, str) or not re.fullmatch(r"[0-9a-f]{16}", text):
        raise ValueError("manifest float64 bits must be 16 lowercase hex digits")
    return struct.unpack(">d", bytes.fromhex(text))[0]


def load_terrain_static_export(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
    expected_source_artifact_sha256: str = TERRAIN_STATIC_SOURCE_ARTIFACT_SHA256,
    expected_source_contract_sha256: str = TERRAIN_STATIC_SOURCE_CONTRACT_SHA256,
) -> LoadedTerrainStaticExport:
    """Load and cryptographically verify one Rust terrain-static arena.

    Every carrier is independently hashed and all byte ranges must be ordered,
    contiguous, distinct, and cover the payload exactly.  Arrays are read-only
    NumPy memmap views and remain physical-domain-only.
    """

    for name, value in (
        ("expected_manifest_sha256", expected_manifest_sha256),
        ("expected_payload_sha256", expected_payload_sha256),
        ("expected_source_artifact_sha256", expected_source_artifact_sha256),
        ("expected_source_contract_sha256", expected_source_contract_sha256),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{name} must be 64 lowercase hex digits")
    root = Path(directory).resolve(strict=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("terrain-static manifest.json is missing")
    manifest_bytes = manifest_path.read_bytes()
    if b"\r" in manifest_bytes:
        raise ValueError("terrain-static manifest must use deterministic LF bytes")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("terrain-static manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("terrain-static manifest is not canonical JSON") from error
    if manifest.get("schema") != TERRAIN_STATIC_SCHEMA:
        raise ValueError("terrain-static schema mismatch")
    if manifest.get("carrier_abi") != TERRAIN_STATIC_ABI:
        raise ValueError("terrain-static carrier ABI mismatch")
    configuration = manifest.get("configuration")
    if configuration != {
        "hadvordrs": 5,
        "hadvordrv": 5,
        "lateral_bc_west_east_south_north": [2, 2, 2, 2],
        "zsgrad_effective_order": 5,
    }:
        raise ValueError("terrain-static branch configuration mismatch")
    source = manifest.get("source", {})
    if source.get("artifact_sha256") != expected_source_artifact_sha256:
        raise ValueError("terrain-static source artifact SHA-256 mismatch")
    if source.get("contract_sha256") != expected_source_contract_sha256:
        raise ValueError("terrain-static source contract SHA-256 mismatch")
    payload_spec = manifest.get("payload", {})
    payload_file = payload_spec.get("file")
    if payload_file != "terrain-static.f32le":
        raise ValueError("terrain-static payload filename mismatch")
    payload_path = (root / payload_file).resolve(strict=True)
    if payload_path.parent != root or not payload_path.is_file():
        raise ValueError("terrain-static payload escapes its export directory")
    payload_nbytes = payload_path.stat().st_size
    if payload_nbytes != payload_spec.get("nbytes"):
        raise ValueError("terrain-static payload byte count mismatch")
    if payload_spec.get("sha256") != expected_payload_sha256:
        raise ValueError("terrain-static manifest payload SHA-256 mismatch")
    payload_sha = _sha256_path(payload_path)
    if payload_sha != expected_payload_sha256:
        raise ValueError("terrain-static payload SHA-256 mismatch")

    np = importlib.import_module("numpy")
    arena = np.memmap(payload_path, dtype=np.uint8, mode="r")
    carrier_specs = manifest.get("carriers")
    if not isinstance(carrier_specs, list):
        raise ValueError("terrain-static carriers must be a list")
    carriers: list[TerrainStaticCarrier] = []
    arrays: dict[str, Any] = {}
    cursor = 0
    for raw in carrier_specs:
        if not isinstance(raw, dict):
            raise ValueError("terrain-static carrier entry must be an object")
        name = raw.get("name")
        shape_raw = raw.get("shape")
        offset = raw.get("offset_nbytes")
        nbytes = raw.get("nbytes")
        digest = raw.get("sha256")
        role = raw.get("role")
        if (
            not isinstance(name, str)
            or not isinstance(shape_raw, list)
            or not shape_raw
            or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in shape_raw)
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or not isinstance(nbytes, int)
            or isinstance(nbytes, bool)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(role, str)
        ):
            raise ValueError("terrain-static carrier metadata is malformed")
        shape = tuple(shape_raw)
        expected_nbytes = math.prod(shape) * 4
        if raw.get("dtype") != "<f4" or raw.get("order") != "C":
            raise ValueError(f"terrain-static carrier {name} is not C-order <f4")
        if offset != cursor or nbytes != expected_nbytes:
            raise ValueError(f"terrain-static carrier {name} has a gap, alias, or wrong size")
        payload_slice = memoryview(arena)[offset : offset + nbytes]
        if hashlib.sha256(payload_slice).hexdigest() != digest:
            raise ValueError(f"terrain-static carrier {name} SHA-256 mismatch")
        array = np.ndarray(shape, dtype="<f4", buffer=arena, offset=offset, order="C")
        array.flags.writeable = False
        carriers.append(TerrainStaticCarrier(name, shape, offset, nbytes, digest, role))
        if name in arrays:
            raise ValueError(f"duplicate terrain-static carrier {name}")
        arrays[name] = array
        cursor += nbytes
    if cursor != payload_nbytes:
        raise ValueError("terrain-static carrier ranges do not cover the payload")
    names = tuple(carrier.name for carrier in carriers)
    if names != TERRAIN_STATIC_REQUIRED:
        raise ValueError("terrain-static carrier order or inventory mismatch")

    source_grid = manifest.get("source_grid", {})
    source_shape = source_grid.get("shape_zyx")
    if source_shape != [75, 300, 300]:
        raise ValueError("terrain-static source grid mismatch")
    dx = _f64_from_bits(source_grid.get("target_dx_m_f64_bits"))
    dy = _f64_from_bits(source_grid.get("target_dy_m_f64_bits"))
    dz = _f64_from_bits(source_grid.get("target_dz_m_f64_bits"))
    crop = manifest.get("crop")
    if crop is None:
        origin = (0, 0)
        count = (300, 300)
    else:
        origin_raw = crop.get("origin_yx")
        count_raw = crop.get("count_yx")
        if (
            not isinstance(origin_raw, list)
            or not isinstance(count_raw, list)
            or len(origin_raw) != 2
            or len(count_raw) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in origin_raw + count_raw)
        ):
            raise ValueError("terrain-static crop metadata is malformed")
        origin = tuple(origin_raw)
        count = tuple(count_raw)
    is_full = origin == (0, 0) and count == (300, 300)
    nk, nj, ni = arrays["zh"].shape
    if nk != 75 or (nj, ni) != count:
        raise ValueError("terrain-static carrier shapes disagree with crop metadata")
    expected_shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"terrain-static carrier {name} shape mismatch")
    base_policy = manifest.get("base_policy")
    if not isinstance(base_policy, str) or "horizontal broadcast" not in base_policy:
        raise ValueError("terrain-static interim base-density policy is not explicit")
    return LoadedTerrainStaticExport(
        root,
        manifest_path,
        payload_path,
        manifest_sha,
        payload_sha,
        payload_nbytes,
        source["artifact_sha256"],
        source["contract_sha256"],
        origin,
        count,
        tuple(source_shape),
        dx,
        dy,
        dz,
        tuple(carriers),
        arrays,
        base_policy,
        is_full,
    )


def terrain_static_shapes(*, ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    for name, value, minimum in (("ni", ni, 3), ("nj", nj, 3), ("nk", nk, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    return {
        "terrain_msl": (nj, ni),
        "zh": (nk, nj, ni),
        "zf": (nk + 1, nj, ni),
        "mh": (nk, nj, ni),
        "rmh": (nk, nj, ni),
        "mf": (nk + 1, nj, ni),
        "rmf": (nk + 1, nj, ni),
        "c1": (nk + 1, nj, ni),
        "c2": (nk + 1, nj, ni),
        "rho0_3d": (nk, nj, ni),
        "rr0": (nk, nj, ni),
        "rf0": (nk + 1, nj, ni),
        "rrf0": (nk + 1, nj, ni),
        "gz": (nj, ni),
        "rgz": (nj, ni),
        "gzu": (nj, ni + 1),
        "rgzu": (nj, ni + 1),
        "gzv": (nj + 1, ni),
        "rgzv": (nj + 1, ni),
        "dzdx": (nj, ni),
        "dzdy": (nj, ni),
        "gx": (nk + 1, nj, ni),
        "gxu": (nk + 1, nj, ni + 1),
        "gy": (nk + 1, nj, ni),
        "gyv": (nk + 1, nj + 1, ni),
    }


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _x(index: int) -> int:
    return index + CM1_OPEN_NGXY - 1


def _y(index: int) -> int:
    return index + CM1_OPEN_NGXY - 1


def terrain_coordinate_profiles(
    nk: int, dz: float, *, levels: Any | None = None
) -> dict[str, Any]:
    """Return source-indexed ``sigma``/``sigmaf``/``rds``/``rdsf`` and the
    ``param.F:8985-9060`` Fornberg weights of *this* coordinate.

    With ``levels`` unset the coordinate is ``param.F:7789-7811``'s uniform one
    and ``dz`` is its layer thickness.  ``levels`` is a
    :class:`cm1core.cm1_vertical_grid.VerticalLevels`; on that path ``dz``
    stays CM1's metric reference but no level is reconstructed from it.

    The weights are computed, never pinned.  ``(1.875, -1.25, 0.375)`` is the
    Fornberg quadratic through three *equally spaced* nodes; on a stretched
    column the lowest three scalar levels are not equally spaced, and reusing
    those three numbers would extrapolate ``rf0`` at the ground, the w-level
    ``u``/``v`` averages and the surface stress from a parabola fitted to the
    wrong points.
    """

    if isinstance(nk, bool) or not isinstance(nk, int) or nk < 3:
        raise ValueError("nk must be an integer >= 3")
    if isinstance(dz, bool) or not isinstance(dz, (int, float)):
        raise TypeError("dz must be numeric")
    dz = float(dz)
    if not math.isfinite(dz) or dz <= 0.0:
        raise ValueError("dz must be finite and > 0")
    np = importlib.import_module("numpy")
    dz32 = np.float32(dz)
    sigma = np.zeros(nk + 2, dtype=np.float32)
    sigmaf = np.zeros(nk + 2, dtype=np.float32)
    rds = np.zeros(nk + 2, dtype=np.float32)
    rdsf = np.zeros(nk + 2, dtype=np.float32)
    if levels is None:
        for k in range(1, nk + 1):
            sigma[k] = _mul(np, _sub(np, np.float32(k), np.float32(0.5)), dz32)
        for k in range(1, nk + 2):
            sigmaf[k] = _mul(np, np.float32(k - 1), dz32)
    else:
        from .cm1_vertical_grid import VerticalLevels

        if not isinstance(levels, VerticalLevels):
            raise TypeError("levels must be cm1_vertical_grid.VerticalLevels")
        if levels.nk != nk:
            raise ValueError("levels must carry exactly nk scalar levels")
        if float(np.float32(levels.dz_reference_m)) != float(dz32):
            raise ValueError(
                "levels dz reference must equal dz; param.F:8170 divides every "
                "mh word by exactly one number"
            )
        sigma[1 : nk + 1] = np.asarray(levels.sigma, dtype=np.float32)[1 : nk + 1]
        sigmaf[1 : nk + 2] = np.asarray(levels.sigmaf, dtype=np.float32)[1 : nk + 2]
    for k in range(2, nk + 1):
        rds[k] = _div(np, np.float32(1.0), _sub(np, sigma[k], sigma[k - 1]))
    for k in range(1, nk + 1):
        rdsf[k] = _div(np, np.float32(1.0), _sub(np, sigmaf[k + 1], sigmaf[k]))
    from .cm1_vertical_grid import fornberg_extrapolation_weights

    cgs = np.array(
        fornberg_extrapolation_weights(sigma, sigmaf, nk=nk, top=False), dtype=np.float32
    )
    cgt = np.array(
        fornberg_extrapolation_weights(sigma, sigmaf, nk=nk, top=True), dtype=np.float32
    )
    return {"sigma": sigma, "sigmaf": sigmaf, "rds": rds, "rdsf": rdsf, "cgs": cgs, "cgt": cgt}


def _require_boundary_arrays(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = {
        "u3d": (nk + 2, nj + 2 * CM1_OPEN_NGXY, ni + 1 + 2 * CM1_OPEN_NGXY),
        "uten": (nk + 2, nj + 2 * CM1_OPEN_NGXY, ni + 1 + 2 * CM1_OPEN_NGXY),
        "v3d": (nk + 2, nj + 1 + 2 * CM1_OPEN_NGXY, ni + 2 * CM1_OPEN_NGXY),
        "vten": (nk + 2, nj + 1 + 2 * CM1_OPEN_NGXY, ni + 2 * CM1_OPEN_NGXY),
        "w3d": (nk + 3, nj + 2 * CM1_OPEN_NGXY, ni + 2 * CM1_OPEN_NGXY),
        "radbcw": (nk + 2, nj + 2 * CM1_OPEN_NGXY),
        "radbce": (nk + 2, nj + 2 * CM1_OPEN_NGXY),
        "radbcs": (nk + 2, ni + 2 * CM1_OPEN_NGXY),
        "radbcn": (nk + 2, ni + 2 * CM1_OPEN_NGXY),
        "uh": (ni + 2 * CM1_OPEN_NGXY,),
        "vh": (nj + 2 * CM1_OPEN_NGXY,),
    }
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError(f"missing terrain-open arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name, shape in shapes.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must be a float32 numpy.ndarray")
        if value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"{name} must have C-order shape {shape}")
        if name in {"u3d", "v3d", "w3d"} and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("terrain-open arrays must not alias")
    one = np.float32(1.0).view(np.uint32)
    if not bool(np.all(result["uh"].view(np.uint32) == one)):
        raise ValueError("uh must be exactly one for the uniform-x product grid")
    if not bool(np.all(result["vh"].view(np.uint32) == one)):
        raise ValueError("vh must be exactly one for the uniform-y product grid")
    return result


def _require_static_arrays(
    metrics: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    missing = [name for name in TERRAIN_STATIC_REQUIRED if name not in metrics]
    if missing:
        raise ValueError(f"missing terrain-static metrics: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in TERRAIN_STATIC_REQUIRED:
        value = metrics[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"terrain-static {name} must be a float32 numpy.ndarray")
        if value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"terrain-static {name} must have C-order shape {shapes[name]}")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"terrain-static {name} contains non-finite values")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("terrain-static arrays must not alias")
    for name in ("mh", "rmh", "mf", "rmf", "rho0_3d", "rr0", "rf0", "rrf0", "gz", "rgz", "gzu", "rgzu", "gzv", "rgzv"):
        if not bool(np.all(result[name] > np.float32(0.0))):
            raise ValueError(f"terrain-static {name} must be strictly positive")
    return result


def _interp_u_at_w(
    np: Any,
    u: Any,
    *,
    kface: int,
    y: int,
    x: int,
    nk: int,
    profiles: Mapping[str, Any],
) -> Any:
    if kface == 1:
        c = profiles["cgs"]
        return _add(np, _add(np, _mul(np, c[0], u[1, y, x]), _mul(np, c[1], u[2, y, x])), _mul(np, c[2], u[3, y, x]))
    if kface == nk + 1:
        c = profiles["cgt"]
        return _add(np, _add(np, _mul(np, c[0], u[nk, y, x]), _mul(np, c[1], u[nk - 1, y, x])), _mul(np, c[2], u[nk - 2, y, x]))
    r2 = _mul(np, _sub(np, profiles["sigmaf"][kface], profiles["sigma"][kface - 1]), profiles["rds"][kface])
    r1 = _sub(np, np.float32(1.0), r2)
    return _add(np, _mul(np, r1, u[kface - 1, y, x]), _mul(np, r2, u[kface, y, x]))


def cpu_bcwsfc_terrain_physical_inplace(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    """Apply bc.F:1151-1178 on physical bottom faces only.

    CM1 also fills horizontal halo bottom faces.  Those are deliberately not
    synthesized because the Rust metric ABI contains no source-valid halos.
    """

    np = importlib.import_module("numpy")
    a = _require_boundary_arrays(arrays, ni=ni, nj=nj, nk=nk)
    m = _require_static_arrays(metrics, ni=ni, nj=nj, nk=nk)
    half = np.float32(0.5)
    for j0 in range(nj):
        y = _y(j0 + 1)
        for i0 in range(ni):
            x = _x(i0 + 1)
            uterm = _mul(np, _add(np, a["u3d"][1, y, x], a["u3d"][1, y, x + 1]), m["dzdx"][j0, i0])
            vterm = _mul(np, _add(np, a["v3d"][1, y, x], a["v3d"][1, y + 1, x]), m["dzdy"][j0, i0])
            a["w3d"][1, y, x] = _mul(np, _mul(np, half, _add(np, uterm, vterm)), m["gz"][j0, i0])


@dataclass(frozen=True, slots=True)
class TerrainOpenBoundaryDiagnostics:
    west_updates: int
    east_updates: int
    south_updates: int
    north_updates: int
    bottom_physical_updates: int
    source_halo_updates: int
    claim_limit: str = CUDA_TERRAIN_OPEN_BOUNDARY_CLAIM_LIMIT


def cpu_ssopen_terrain_all_inplace(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dts: float,
    dx: float,
    dy: float,
    dz: float,
    apply_bcwsfc: bool = True,
    selection: TerrainOpenSelection = DEFAULT_TERRAIN_OPEN_SELECTION,
) -> TerrainOpenBoundaryDiagnostics:
    """Apply all four terrain ``ssopenbc*`` operators in source order."""

    validate_terrain_open_selection(selection)
    for name, value in (("dts", dts), ("dx", dx), ("dy", dy), ("dz", dz)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    np = importlib.import_module("numpy")
    a = _require_boundary_arrays(arrays, ni=ni, nj=nj, nk=nk)
    m = _require_static_arrays(metrics, ni=ni, nj=nj, nk=nk)
    p = terrain_coordinate_profiles(nk, dz)
    dts32 = np.float32(dts)
    rdx = _div(np, np.float32(1.0), np.float32(dx))
    rdy = _div(np, np.float32(1.0), np.float32(dy))
    half = np.float32(0.5)

    # bc.F:2392-2459, west.  CM1 fills every vertical ``dum1`` word
    # before the separate update loop; preserving that snapshot is necessary
    # once a nonzero radiation history couples adjacent levels.
    west = np.empty((nk + 1, nj, 2), dtype=np.float32)
    for kface in range(1, nk + 2):
        for j0 in range(nj):
            y = _y(j0 + 1)
            west[kface - 1, j0, 0] = _interp_u_at_w(
                np, a["u3d"], kface=kface, y=y, x=_x(1), nk=nk, profiles=p
            )
            west[kface - 1, j0, 1] = _interp_u_at_w(
                np, a["u3d"], kface=kface, y=y, x=_x(2), nk=nk, profiles=p
            )
    for k in range(1, nk + 1):
        for j0 in range(nj):
            y = _y(j0 + 1)
            lo1, lo2 = west[k - 1, j0]
            hi1, hi2 = west[k, j0]
            horizontal = _mul(np, m["gz"][j0, 0], _sub(np, _mul(np, a["u3d"][k, y, _x(2)], m["rgzu"][j0, 1]), _mul(np, a["u3d"][k, y, _x(1)], m["rgzu"][j0, 0])))
            horizontal = _mul(np, _mul(np, horizontal, rdx), a["uh"][_x(1)])
            vertical = _sub(np, _mul(np, m["gx"][k, j0, 0], _add(np, hi2, hi1)), _mul(np, m["gx"][k - 1, j0, 0], _add(np, lo2, lo1)))
            vertical = _mul(np, _mul(np, half, vertical), p["rdsf"][k])
            radiation = _mul(np, -a["radbcw"][k, y], _add(np, horizontal, vertical))
            tendency = _add(np, radiation, a["uten"][k, y, _x(1)])
            a["u3d"][k, y, _x(1)] = _add(np, a["u3d"][k, y, _x(1)], _mul(np, dts32, tendency))

    # bc.F:2467-2534, east.
    east = np.empty((nk + 1, nj, 2), dtype=np.float32)
    for kface in range(1, nk + 2):
        for j0 in range(nj):
            y = _y(j0 + 1)
            east[kface - 1, j0, 0] = _interp_u_at_w(
                np, a["u3d"], kface=kface, y=y, x=_x(ni), nk=nk, profiles=p
            )
            east[kface - 1, j0, 1] = _interp_u_at_w(
                np,
                a["u3d"],
                kface=kface,
                y=y,
                x=_x(ni + 1),
                nk=nk,
                profiles=p,
            )
    for k in range(1, nk + 1):
        for j0 in range(nj):
            y = _y(j0 + 1)
            lo0, lo1 = east[k - 1, j0]
            hi0, hi1 = east[k, j0]
            horizontal = _mul(np, m["gz"][j0, ni - 1], _sub(np, _mul(np, a["u3d"][k, y, _x(ni + 1)], m["rgzu"][j0, ni]), _mul(np, a["u3d"][k, y, _x(ni)], m["rgzu"][j0, ni - 1])))
            horizontal = _mul(np, _mul(np, horizontal, rdx), a["uh"][_x(ni)])
            vertical = _sub(np, _mul(np, m["gx"][k, j0, ni - 1], _add(np, hi1, hi0)), _mul(np, m["gx"][k - 1, j0, ni - 1], _add(np, lo1, lo0)))
            vertical = _mul(np, _mul(np, half, vertical), p["rdsf"][k])
            radiation = _mul(np, -a["radbce"][k, y], _add(np, horizontal, vertical))
            tendency = _add(np, radiation, a["uten"][k, y, _x(ni + 1)])
            a["u3d"][k, y, _x(ni + 1)] = _add(np, a["u3d"][k, y, _x(ni + 1)], _mul(np, dts32, tendency))

    # bc.F:2542-2609, south.  The interpolation helper is layout-agnostic.
    south = np.empty((nk + 1, ni, 2), dtype=np.float32)
    for kface in range(1, nk + 2):
        for i0 in range(ni):
            x = _x(i0 + 1)
            south[kface - 1, i0, 0] = _interp_u_at_w(
                np, a["v3d"], kface=kface, y=_y(1), x=x, nk=nk, profiles=p
            )
            south[kface - 1, i0, 1] = _interp_u_at_w(
                np, a["v3d"], kface=kface, y=_y(2), x=x, nk=nk, profiles=p
            )
    for k in range(1, nk + 1):
        for i0 in range(ni):
            x = _x(i0 + 1)
            lo1, lo2 = south[k - 1, i0]
            hi1, hi2 = south[k, i0]
            horizontal = _mul(np, m["gz"][0, i0], _sub(np, _mul(np, a["v3d"][k, _y(2), x], m["rgzv"][1, i0]), _mul(np, a["v3d"][k, _y(1), x], m["rgzv"][0, i0])))
            horizontal = _mul(np, _mul(np, horizontal, rdy), a["vh"][_y(1)])
            vertical = _sub(np, _mul(np, m["gy"][k, 0, i0], _add(np, hi2, hi1)), _mul(np, m["gy"][k - 1, 0, i0], _add(np, lo2, lo1)))
            vertical = _mul(np, _mul(np, half, vertical), p["rdsf"][k])
            radiation = _mul(np, -a["radbcs"][k, x], _add(np, horizontal, vertical))
            tendency = _add(np, radiation, a["vten"][k, _y(1), x])
            a["v3d"][k, _y(1), x] = _add(np, a["v3d"][k, _y(1), x], _mul(np, dts32, tendency))

    # bc.F:2617-2684, north.
    north = np.empty((nk + 1, ni, 2), dtype=np.float32)
    for kface in range(1, nk + 2):
        for i0 in range(ni):
            x = _x(i0 + 1)
            north[kface - 1, i0, 0] = _interp_u_at_w(
                np, a["v3d"], kface=kface, y=_y(nj), x=x, nk=nk, profiles=p
            )
            north[kface - 1, i0, 1] = _interp_u_at_w(
                np,
                a["v3d"],
                kface=kface,
                y=_y(nj + 1),
                x=x,
                nk=nk,
                profiles=p,
            )
    for k in range(1, nk + 1):
        for i0 in range(ni):
            x = _x(i0 + 1)
            lo0, lo1 = north[k - 1, i0]
            hi0, hi1 = north[k, i0]
            horizontal = _mul(np, m["gz"][nj - 1, i0], _sub(np, _mul(np, a["v3d"][k, _y(nj + 1), x], m["rgzv"][nj, i0]), _mul(np, a["v3d"][k, _y(nj), x], m["rgzv"][nj - 1, i0])))
            horizontal = _mul(np, _mul(np, horizontal, rdy), a["vh"][_y(nj)])
            vertical = _sub(np, _mul(np, m["gy"][k, nj - 1, i0], _add(np, hi1, hi0)), _mul(np, m["gy"][k - 1, nj - 1, i0], _add(np, lo1, lo0)))
            vertical = _mul(np, _mul(np, half, vertical), p["rdsf"][k])
            radiation = _mul(np, -a["radbcn"][k, x], _add(np, horizontal, vertical))
            tendency = _add(np, radiation, a["vten"][k, _y(nj + 1), x])
            a["v3d"][k, _y(nj + 1), x] = _add(np, a["v3d"][k, _y(nj + 1), x], _mul(np, dts32, tendency))

    if apply_bcwsfc:
        cpu_bcwsfc_terrain_physical_inplace(a, m, ni=ni, nj=nj, nk=nk)
    return TerrainOpenBoundaryDiagnostics(
        nk * nj,
        nk * nj,
        nk * ni,
        nk * ni,
        ni * nj if apply_bcwsfc else 0,
        0,
    )


CUDA_TERRAIN_OPEN_BOUNDARY_KERNEL_NAMES = (
    "cm1_terrain_ssopen_snapshot_v2",
    "cm1_terrain_ssopen_update_v2",
    "cm1_terrain_bcwsfc_physical_v1",
)


CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE = r'''// gpuwm-cm1.cuda-terrain-open-boundary-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ float atw(const float* q,unsigned int kf,unsigned int y,unsigned int x,unsigned int nk,unsigned int ny,unsigned int nx,float dz){
  if(kf==1u)return a32(a32(m32(1.875f,q[i3(1u,y,x,ny,nx)]),m32(-1.25f,q[i3(2u,y,x,ny,nx)])),m32(0.375f,q[i3(3u,y,x,ny,nx)]));
  if(kf==nk+1u)return a32(a32(m32(1.875f,q[i3(nk,y,x,ny,nx)]),m32(-1.25f,q[i3(nk-1u,y,x,ny,nx)])),m32(0.375f,q[i3(nk-2u,y,x,ny,nx)]));
  const float sigmaf=m32((float)(kf-1u),dz),sigma=m32(s32((float)(kf-1u),0.5f),dz),rds=1.0f/dz;
  const float r2=m32(s32(sigmaf,sigma),rds),r1=s32(1.0f,r2);
  return a32(m32(r1,q[i3(kf-1u,y,x,ny,nx)]),m32(r2,q[i3(kf,y,x,ny,nx)]));
}
extern "C" __global__ void cm1_terrain_ssopen_snapshot_v2(
 const float* u,const float* v,float* us,float* vs,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;
 const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
 const unsigned long long uwe=(unsigned long long)(nk+1u)*nj,vsn=(unsigned long long)(nk+1u)*ni;
 for(unsigned long long q=first;q<uwe;q+=step){const unsigned int j=(unsigned int)(q%nj),kf=(unsigned int)(q/nj)+1u,y=j+ng;const unsigned long long b=q*4u;us[b]=atw(u,kf,y,ng,nk,nyu,nxu,dz);us[b+1u]=atw(u,kf,y,ng+1u,nk,nyu,nxu,dz);us[b+2u]=atw(u,kf,y,ni+ng-1u,nk,nyu,nxu,dz);us[b+3u]=atw(u,kf,y,ni+ng,nk,nyu,nxu,dz);}
 for(unsigned long long q=first;q<vsn;q+=step){const unsigned int i=(unsigned int)(q%ni),kf=(unsigned int)(q/ni)+1u,x=i+ng;const unsigned long long b=q*4u;vs[b]=atw(v,kf,ng,x,nk,nyv,nxv,dz);vs[b+1u]=atw(v,kf,ng+1u,x,nk,nyv,nxv,dz);vs[b+2u]=atw(v,kf,nj+ng-1u,x,nk,nyv,nxv,dz);vs[b+3u]=atw(v,kf,nj+ng,x,nk,nyv,nxv,dz);}
}
extern "C" __global__ void cm1_terrain_ssopen_update_v2(
 const float* uten,const float* vten,const float* uh,const float* vh,
 const float* rw,const float* re,const float* rs,const float* rn,
 const float* gz,const float* rgzu,const float* rgzv,const float* gx,const float* gy,
 const float* us,const float* vs,float* u,float* v,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
 float dts,float rdx,float rdy,float dz){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;
 const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
 const unsigned long long we=(unsigned long long)nk*nj,sn=(unsigned long long)nk*ni;
 for(unsigned long long q=first;q<we;q+=step){
   const unsigned int j0=(unsigned int)(q%nj),k=(unsigned int)(q/nj)+1u,y=j0+ng,xw=ng,xi=ng+1u,xe=ni+ng,xin=ni+ng-1u;
   const unsigned long long lo=((unsigned long long)(k-1u)*nj+j0)*4u,hi=((unsigned long long)k*nj+j0)*4u;float l1=us[lo],l2=us[lo+1u],h1=us[hi],h2=us[hi+1u];
   float hor=m32(gz[(unsigned long long)j0*ni],s32(m32(u[i3(k,y,xi,nyu,nxu)],rgzu[(unsigned long long)j0*(ni+1u)+1u]),m32(u[i3(k,y,xw,nyu,nxu)],rgzu[(unsigned long long)j0*(ni+1u)])));
   hor=m32(m32(hor,rdx),uh[ng]); float ver=s32(m32(gx[((unsigned long long)k*nj+j0)*ni],a32(h2,h1)),m32(gx[((unsigned long long)(k-1u)*nj+j0)*ni],a32(l2,l1))); ver=m32(m32(0.5f,ver),1.0f/dz);
   unsigned long long b=i3(k,y,xw,nyu,nxu); float t=a32(m32(-rw[(unsigned long long)k*(nj+2u*ng)+y],a32(hor,ver)),uten[b]);u[b]=a32(u[b],m32(dts,t));
   l1=us[lo+2u];l2=us[lo+3u];h1=us[hi+2u];h2=us[hi+3u];
   hor=m32(gz[(unsigned long long)j0*ni+ni-1u],s32(m32(u[i3(k,y,xe,nyu,nxu)],rgzu[(unsigned long long)j0*(ni+1u)+ni]),m32(u[i3(k,y,xin,nyu,nxu)],rgzu[(unsigned long long)j0*(ni+1u)+ni-1u])));hor=m32(m32(hor,rdx),uh[ni+ng-1u]);
   ver=s32(m32(gx[((unsigned long long)k*nj+j0)*ni+ni-1u],a32(h2,h1)),m32(gx[((unsigned long long)(k-1u)*nj+j0)*ni+ni-1u],a32(l2,l1)));ver=m32(m32(0.5f,ver),1.0f/dz);b=i3(k,y,xe,nyu,nxu);t=a32(m32(-re[(unsigned long long)k*(nj+2u*ng)+y],a32(hor,ver)),uten[b]);u[b]=a32(u[b],m32(dts,t));
 }
 for(unsigned long long q=first;q<sn;q+=step){
   const unsigned int i0=(unsigned int)(q%ni),k=(unsigned int)(q/ni)+1u,x=i0+ng,ys=ng,yi=ng+1u,yn=nj+ng,yin=nj+ng-1u;
   const unsigned long long lo=((unsigned long long)(k-1u)*ni+i0)*4u,hi=((unsigned long long)k*ni+i0)*4u;float l1=vs[lo],l2=vs[lo+1u],h1=vs[hi],h2=vs[hi+1u];
   float hor=m32(gz[i0],s32(m32(v[i3(k,yi,x,nyv,nxv)],rgzv[(unsigned long long)ni+i0]),m32(v[i3(k,ys,x,nyv,nxv)],rgzv[i0])));hor=m32(m32(hor,rdy),vh[ng]);float ver=s32(m32(gy[((unsigned long long)k*nj)*ni+i0],a32(h2,h1)),m32(gy[((unsigned long long)(k-1u)*nj)*ni+i0],a32(l2,l1)));ver=m32(m32(0.5f,ver),1.0f/dz);
   unsigned long long b=i3(k,ys,x,nyv,nxv);float t=a32(m32(-rs[(unsigned long long)k*(ni+2u*ng)+x],a32(hor,ver)),vten[b]);v[b]=a32(v[b],m32(dts,t));
   l1=vs[lo+2u];l2=vs[lo+3u];h1=vs[hi+2u];h2=vs[hi+3u];
   hor=m32(gz[(unsigned long long)(nj-1u)*ni+i0],s32(m32(v[i3(k,yn,x,nyv,nxv)],rgzv[(unsigned long long)nj*ni+i0]),m32(v[i3(k,yin,x,nyv,nxv)],rgzv[(unsigned long long)(nj-1u)*ni+i0])));hor=m32(m32(hor,rdy),vh[nj+ng-1u]);ver=s32(m32(gy[((unsigned long long)k*nj+nj-1u)*ni+i0],a32(h2,h1)),m32(gy[((unsigned long long)(k-1u)*nj+nj-1u)*ni+i0],a32(l2,l1)));ver=m32(m32(0.5f,ver),1.0f/dz);b=i3(k,yn,x,nyv,nxv);t=a32(m32(-rn[(unsigned long long)k*(ni+2u*ng)+x],a32(hor,ver)),vten[b]);v[b]=a32(v[b],m32(dts,t));
 }
}
extern "C" __global__ void cm1_terrain_bcwsfc_physical_v1(const float* u,const float* v,const float* gz,const float* dzdx,const float* dzdy,float* w,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj;const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxw=ni+2u*ng,nyw=nj+2u*ng;
 for(unsigned long long q=first;q<n;q+=step){unsigned int i0=(unsigned int)(q%ni),j0=(unsigned int)(q/ni),x=i0+ng,y=j0+ng;float ut=m32(a32(u[i3(1u,y,x,nyu,nxu)],u[i3(1u,y,x+1u,nyu,nxu)]),dzdx[q]),vt=m32(a32(v[i3(1u,y,x,nyv,nxv)],v[i3(1u,y+1u,x,nyv,nxv)]),dzdy[q]);w[i3(1u,y,x,nyw,nxw)]=m32(m32(0.5f,a32(ut,vt)),gz[q]);}(void)nk;
}
'''

CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_BYTES = CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE.encode("utf-8")
CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_SHA256 = (
    "cb62d3a9b96a79c50d214344882ba8e4cbd0879d7f0ca515f0399f4379071a62"
)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledTerrainOpenBoundaryKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_terrain_open_boundary() -> CompiledTerrainOpenBoundaryKernels:
    """Compile the pinned boundary operator for the active CUDA device."""

    digest = hashlib.sha256(CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_BYTES).hexdigest()
    if digest != CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_SHA256:
        raise RuntimeError("packaged terrain-open CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(
        CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE,
        "cm1_terrain_open_boundary_v1_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE,
        "cm1_terrain_open_boundary_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text) for name in CUDA_TERRAIN_OPEN_BOUNDARY_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks terrain-open entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled terrain-open image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_TERRAIN_OPEN_BOUNDARY_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledTerrainOpenBoundaryKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity)


@dataclass(frozen=True, slots=True)
class CudaTerrainOpenBoundaryLaunch:
    ready_event: Any
    compiled: CompiledTerrainOpenBoundaryKernels
    retained_arrays: tuple[Any, ...]
    receipt: dict[str, Any]


def launch_cuda_terrain_open_boundary(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dts: float,
    dx: float,
    dy: float,
    dz: float,
    selection: TerrainOpenSelection = DEFAULT_TERRAIN_OPEN_SELECTION,
    stream: Any | None = None,
    compiled: CompiledTerrainOpenBoundaryKernels | None = None,
) -> CudaTerrainOpenBoundaryLaunch:
    """Enqueue one device-resident terrain boundary transition."""

    validate_terrain_open_selection(selection)
    cupy = importlib.import_module("cupy")
    compiled = compiled or compile_cuda_terrain_open_boundary()
    stream = stream or cupy.cuda.Stream(non_blocking=True)
    host_shapes = {
        "u3d": (nk + 2, nj + 2 * CM1_OPEN_NGXY, ni + 1 + 2 * CM1_OPEN_NGXY),
        "uten": (nk + 2, nj + 2 * CM1_OPEN_NGXY, ni + 1 + 2 * CM1_OPEN_NGXY),
        "v3d": (nk + 2, nj + 1 + 2 * CM1_OPEN_NGXY, ni + 2 * CM1_OPEN_NGXY),
        "vten": (nk + 2, nj + 1 + 2 * CM1_OPEN_NGXY, ni + 2 * CM1_OPEN_NGXY),
        "w3d": (nk + 3, nj + 2 * CM1_OPEN_NGXY, ni + 2 * CM1_OPEN_NGXY),
        "radbcw": (nk + 2, nj + 2 * CM1_OPEN_NGXY),
        "radbce": (nk + 2, nj + 2 * CM1_OPEN_NGXY),
        "radbcs": (nk + 2, ni + 2 * CM1_OPEN_NGXY),
        "radbcn": (nk + 2, ni + 2 * CM1_OPEN_NGXY),
        "uh": (ni + 2 * CM1_OPEN_NGXY,),
        "vh": (nj + 2 * CM1_OPEN_NGXY,),
    }
    static_shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    for name, shape in host_shapes.items():
        value = arrays.get(name)
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32") or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"CUDA terrain-open {name} has invalid dtype, shape, or layout")
    for name, shape in static_shapes.items():
        value = metrics.get(name)
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32") or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"CUDA terrain-static {name} has invalid dtype, shape, or layout")
    pointers = [int(value.data.ptr) for value in arrays.values()] + [int(value.data.ptr) for value in metrics.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("CUDA terrain-open arrays and metrics must not alias")
    snapshot, ssopen, bcwsfc = compiled.kernels
    u_snapshot = cupy.empty(((nk + 1) * nj, 4), dtype=cupy.float32)
    v_snapshot = cupy.empty(((nk + 1) * ni, 4), dtype=cupy.float32)
    count = max((nk + 1) * nj, (nk + 1) * ni, ni * nj)
    grid = ((count + CUDA_TERRAIN_OPEN_BOUNDARY_THREADS - 1) // CUDA_TERRAIN_OPEN_BOUNDARY_THREADS, 1, 1)
    block = (CUDA_TERRAIN_OPEN_BOUNDARY_THREADS, 1, 1)
    f32, u32 = cupy.float32, cupy.uint32
    with stream:
        snapshot(
            grid,
            block,
            (
                arrays["u3d"], arrays["v3d"], u_snapshot, v_snapshot,
                u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), f32(dz),
            ),
            stream=stream,
        )
        ssopen(
            grid,
            block,
            (
                arrays["uten"], arrays["vten"], arrays["uh"], arrays["vh"],
                arrays["radbcw"], arrays["radbce"], arrays["radbcs"], arrays["radbcn"],
                metrics["gz"], metrics["rgzu"], metrics["rgzv"], metrics["gx"], metrics["gy"],
                u_snapshot, v_snapshot, arrays["u3d"], arrays["v3d"],
                u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY),
                f32(dts), f32(1.0 / dx), f32(1.0 / dy), f32(dz),
            ),
            stream=stream,
        )
        bcwsfc(
            grid,
            block,
            (
                arrays["u3d"], arrays["v3d"], metrics["gz"], metrics["dzdx"], metrics["dzdy"], arrays["w3d"],
                u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event()
        event.record(stream)
    return CudaTerrainOpenBoundaryLaunch(
        event,
        compiled,
        (u_snapshot, v_snapshot),
        {
            "schema": CUDA_TERRAIN_OPEN_BOUNDARY_SCHEMA,
            "source_sha256": CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
            "runtime": dict(compiled.runtime_identity),
            "updates": {
                "west": nk * nj,
                "east": nk * nj,
                "south": nk * ni,
                "north": nk * ni,
                "bcwsfc_physical": ni * nj,
                "bcwsfc_source_halos": 0,
            },
            "claim_limit": CUDA_TERRAIN_OPEN_BOUNDARY_CLAIM_LIMIT,
        },
    )


def compare_terrain_open_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare boundary outputs with raw-bit/no-null semantics."""

    np = importlib.import_module("numpy")
    fields: dict[str, Any] = {}
    for name in ("u3d", "v3d", "w3d"):
        if name not in expected or name not in candidate:
            raise ValueError(f"missing terrain-open comparison field {name}")
        left, right = expected[name], candidate[name]
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise TypeError(f"{name} comparison fields must be numpy.ndarray")
        if left.dtype != np.dtype("float32") or right.dtype != np.dtype("float32") or left.shape != right.shape:
            raise ValueError(f"{name} comparison dtype/shape mismatch")
        exact = bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))
        fields[name] = {
            "exact": exact,
            "expected_sha256": hashlib.sha256(left.tobytes(order="C")).hexdigest(),
            "candidate_sha256": hashlib.sha256(right.tobytes(order="C")).hexdigest(),
        }
    return {
        "mode": "raw-bit",
        "max_ulp": 0,
        "null_or_missing_is_pass": False,
        "passed": all(item["exact"] for item in fields.values()),
        "fields": fields,
    }
