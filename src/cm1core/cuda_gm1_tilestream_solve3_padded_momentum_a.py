"""Tiled solve3 four-open bc2d and padded momentum current-to-A completion.

This seam consumes the exact physical momentum-A token.  The admitted padded
terrain/open carriers already contain that token's finalized physical words.
It applies ``bc2d(w3d(:,:,1))`` in west, east, south, north order and then the
literal ``solve3.F:548-558`` padded copy range.  TKE and every later solve3
operation remain unavailable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import (
    Gm1TilePlan,
    Gm1TileSpec,
    build_gm1_open_tile_plan,
)
from .cuda_gm1_tilestream_solve3_momentum_a import (
    Gm1TiledSolve3PhysicalMomentumAToken,
)
from .cuda_sound3_periodic import _canonical_arch, _image_options


CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SCHEMA = (
    "gm1.cuda-tilestream-solve3-padded-momentum-a/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_ABI = (
    "gm1.cuda-tilestream-solve3-four-open-padded-momentum-a-f32/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_KERNEL_NAMES = (
    "gm1_tiled_solve3_bc2d_we_f32_v1",
    "gm1_tiled_solve3_bc2d_sn_f32_v1",
    "gm1_tiled_solve3_padded_momentum_a_f32_v1",
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE3_SHA256 = (
    "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"
)
AUTHORITY_BC_SHA256 = "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
AUTHORITY_ANCHORS = (
    "src/solve3.F:430-433",
    "src/solve3.F:548-559",
    "src/bc.F:12-172",
)
GM1_SOURCE_IDENTITY_SHA256 = (
    "d6d7588b9a2a2ecd979d4d5e609a602f52d488fd918c590bc3998a5319aebfe3"
)
GM1_ACCEPTED_PHYSICAL_RECEIPT_FILE_SHA256 = (
    "83f1c70d05c6ee4e3b6c50acb01dbcc9dc721d72aae4cdcb6509892b57d4d899"
)
GM1_ACCEPTED_PHYSICAL_RECEIPT_SEMANTIC_SHA256 = (
    "7366e1c4eb179b580f83d9dcb563335a2504bc71273068fd868314e0bda06cd7"
)
GM1_ACCEPTED_PHYSICAL_OUTPUT_SHA256 = (
    "fdad6aaa80e2325d99560dd56e54dc7e47cbf2ca359fb130a3431079625e2158"
)
GM1_ACCEPTED_PHYSICAL_CARRIER_SHA256 = MappingProxyType(
    {
        "u3d": "b55dcf1792d1fdd8dafb5eabda2b70f1a95c8aa4c7b67070d3421e6a05147402",
        "v3d": "4c2b5fecd445fcae0564f936bd12e90caa52f643c78d471ff5f3fe6b1c23ee89",
        "w3d": "67177ebbc40e2a02f56967c0371e5497277ba9c2c536d82ef061f4e1eee75824",
        "ua": "b55dcf1792d1fdd8dafb5eabda2b70f1a95c8aa4c7b67070d3421e6a05147402",
        "va": "4c2b5fecd445fcae0564f936bd12e90caa52f643c78d471ff5f3fe6b1c23ee89",
        "wa": "67177ebbc40e2a02f56967c0371e5497277ba9c2c536d82ef061f4e1eee75824",
    }
)
GM1_ACCEPTED_PADDED_INPUT_SHA256 = MappingProxyType(
    {
        "u3d": "a422c8757051537fea559fd339153120908042adf3408d4093c79877e7ca2e45",
        "v3d": "324e1ef2a67f433e120ee9b8505906c8825d621416dc2b23ff081d197f7587e2",
        "w3d": "10e58efd55079b9bac4ce35dfd023d700019059ceebb93a87ac7b3642a59b77f",
        "ua": "a422c8757051537fea559fd339153120908042adf3408d4093c79877e7ca2e45",
        "va": "324e1ef2a67f433e120ee9b8505906c8825d621416dc2b23ff081d197f7587e2",
        "wa": "10e58efd55079b9bac4ce35dfd023d700019059ceebb93a87ac7b3642a59b77f",
    }
)
GM1_ACCEPTED_PADDED_OUTPUT_SHA256 = MappingProxyType(
    {
        "u3d": "a422c8757051537fea559fd339153120908042adf3408d4093c79877e7ca2e45",
        "v3d": "324e1ef2a67f433e120ee9b8505906c8825d621416dc2b23ff081d197f7587e2",
        "w3d": "c6a2030467642f63f1fb9be6cc3be44b3525ed1ee48ba5376e526c9688b28957",
        "ua": "a422c8757051537fea559fd339153120908042adf3408d4093c79877e7ca2e45",
        "va": "324e1ef2a67f433e120ee9b8505906c8825d621416dc2b23ff081d197f7587e2",
        "wa": "9ebcada415f14d9ce63e716fa0a4adb85d70ae81997046f30b24667ea07cf3e8",
    }
)
GM1_ACCEPTED_PADDED_W_BOTTOM_SHA256 = (
    "ec883afdff363677c7ebb371e132e419e1f16d9b9dfd044ae1cde533ab29b01e"
)
GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_CLAIM_LIMIT = (
    "CM1 r21.1 selected serial four-open bc2d west/east then south/north on "
    "the terrain-finalized w bottom, followed by the exact padded "
    "u3d/v3d/w3d to ua/va/wa range through solve3.F:558; TKE, scalar A, "
    "rr/rf, global psolver, MPI, radiation/open-history mutation, full "
    "solve3/RK, and five-metre meteorological product remain refused"
)

_NG = 3
_PADDED_ORDER = ("u3d", "v3d", "w3d", "ua", "va", "wa")
_BUFFER_ORDER = (
    *_PADDED_ORDER,
    "w_bottom_in",
    "w_bottom_we",
    "w_bottom_out",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha_array(value: Any) -> str:
    return hashlib.sha256(
        value.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()


def _sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def padded_momentum_a_shapes(
    ni: int, nj: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    ni, nj, nk = (
        _positive_int(name, value)
        for name, value in (("ni", ni), ("nj", nj), ("nk", nk))
    )
    return MappingProxyType(
        {
            "u3d": (nk + 2, nj + 2 * _NG, ni + 1 + 2 * _NG),
            "v3d": (nk + 2, nj + 1 + 2 * _NG, ni + 2 * _NG),
            "w3d": (nk + 3, nj + 2 * _NG, ni + 2 * _NG),
            "ua": (nk + 2, nj + 2 * _NG, ni + 1 + 2 * _NG),
            "va": (nk + 2, nj + 1 + 2 * _NG, ni + 2 * _NG),
            "wa": (nk + 3, nj + 2 * _NG, ni + 2 * _NG),
        }
    )


def tiled_solve3_padded_momentum_a_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    return MappingProxyType(
        {
            **{name: (nk, tile_ny, tile_nx) for name in _PADDED_ORDER},
            "w_bottom_in": (tile_ny, tile_nx),
            "w_bottom_we": (tile_ny, tile_nx),
            "w_bottom_out": (tile_ny, tile_nx),
        }
    )


def _boundary_aligned_tile_extent(global_extent: int, preferred: int) -> int:
    """Keep each open boundary band and its source line in one owned tile."""

    global_extent = _positive_int("global_extent", global_extent)
    preferred = min(_positive_int("preferred", preferred), global_extent)
    minimum = min(global_extent, _NG + 1)
    extent = max(preferred, minimum)
    while extent < global_extent:
        remainder = global_extent % extent
        if remainder == 0 or remainder >= _NG + 1:
            break
        extent += 1
    return extent


def build_gm1_solve3_padded_momentum_a_plan(
    *, ni: int, nj: int, tile_nx: int, tile_ny: int
) -> Gm1TilePlan:
    """Build the exact padded plan with unsplit bc2d source/boundary rings."""

    ni = _positive_int("ni", ni)
    nj = _positive_int("nj", nj)
    padded_nx, padded_ny = ni + 2 * _NG, nj + 2 * _NG
    return build_gm1_open_tile_plan(
        nx=padded_nx,
        ny=padded_ny,
        tile_nx=_boundary_aligned_tile_extent(padded_nx, tile_nx),
        tile_ny=_boundary_aligned_tile_extent(padded_ny, tile_ny),
    )


def _require_local_bc2d_support(
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    global_nx: int,
    global_ny: int,
) -> None:
    """Refuse a tile plan that separates a halo band from its source line."""

    bounds = (("west", x0, x1), ("south", y0, y1))
    for name, lower, upper in bounds:
        if lower < _NG and not (lower == 0 and upper > _NG):
            raise ValueError(f"{name} bc2d boundary-support ownership is split")
    bounds = (
        ("east", x0, x1, global_nx),
        ("north", y0, y1, global_ny),
    )
    for name, lower, upper, global_extent in bounds:
        if upper > global_extent - _NG and not (
            upper == global_extent and lower <= global_extent - _NG - 1
        ):
            raise ValueError(f"{name} bc2d boundary-support ownership is split")


def _require_physical_token(
    token: object, *, source_identity_sha256: str, real: bool
) -> Mapping[str, Any]:
    if not isinstance(token, Gm1TiledSolve3PhysicalMomentumAToken):
        raise TypeError("physical token must be the exact momentum-A token type")
    receipt = token.receipt
    if (
        receipt.get("schema") != "gm1.tilestream-solve3-physical-momentum-a-token/v1"
        or receipt.get("status") != "ADMITTED_PHYSICAL_MOMENTUM_A_ONLY"
        or receipt.get("ready_for")
        != "horizontal bc2d and padded momentum-A completion only"
        or receipt.get("full_momentum_current_to_a_available") is not False
        or receipt.get("tke_commit_available") is not False
        or receipt.get("source_identity_sha256") != source_identity_sha256
    ):
        raise ValueError("physical momentum-A token scope or source identity changed")
    detached = dict(receipt)
    operation_token = detached.pop("operation_token_sha256", None)
    if operation_token != hashlib.sha256(_canonical(detached)).hexdigest():
        raise ValueError("physical momentum-A token identity changed")
    if real and receipt.get("outputs") != dict(GM1_ACCEPTED_PHYSICAL_CARRIER_SHA256):
        raise ValueError(
            "physical momentum-A output hashes differ from accepted evidence"
        )
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3PaddedMomentumABinding:
    u3d: Any
    v3d: Any
    w3d: Any
    ua: Any
    va: Any
    wa: Any
    physical_token: Gm1TiledSolve3PhysicalMomentumAToken
    ni: int
    nj: int
    nk: int
    source_identity_sha256: str
    physical_receipt_file_sha256: str
    physical_receipt_semantic_sha256: str
    compact_validation: bool

    @classmethod
    def create(
        cls,
        *,
        u3d: Any,
        v3d: Any,
        w3d: Any,
        ua: Any,
        va: Any,
        wa: Any,
        physical_token: Gm1TiledSolve3PhysicalMomentumAToken,
        ni: int,
        nj: int,
        nk: int,
        source_identity_sha256: str,
        physical_receipt_file_sha256: str,
        physical_receipt_semantic_sha256: str,
        allow_compact_validation: bool = False,
    ) -> "Gm1TiledSolve3PaddedMomentumABinding":
        np = importlib.import_module("numpy")
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        ni, nj, nk = (
            _positive_int(name, value)
            for name, value in (("ni", ni), ("nj", nj), ("nk", nk))
        )
        shapes = padded_momentum_a_shapes(ni, nj, nk)
        values = {
            name: value for name, value in locals().items() if name in _PADDED_ORDER
        }
        for name, shape in shapes.items():
            value = values[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f4")
                or value.shape != shape
                or not value.flags.c_contiguous
                or not value.flags.writeable
            ):
                raise ValueError(f"{name} must be writable host C-<f4 {shape}")
        value_list = tuple(values.values())
        for index, left in enumerate(value_list):
            if any(np.shares_memory(left, right) for right in value_list[index + 1 :]):
                raise ValueError("padded momentum current/A arrays must not alias")
        if not all(bool(np.all(np.isfinite(value))) for value in value_list):
            raise ValueError("padded momentum-A input contains non-finite words")
        source_identity_sha256 = _sha256(
            "source_identity_sha256", source_identity_sha256
        )
        physical_receipt_file_sha256 = _sha256(
            "physical_receipt_file_sha256", physical_receipt_file_sha256
        )
        physical_receipt_semantic_sha256 = _sha256(
            "physical_receipt_semantic_sha256", physical_receipt_semantic_sha256
        )
        _require_physical_token(
            physical_token,
            source_identity_sha256=source_identity_sha256,
            real=not allow_compact_validation,
        )
        if allow_compact_validation:
            if min(ni, nj, nk) < 2:
                raise ValueError("compact padded momentum-A domain is too small")
        else:
            if (ni, nj, nk) != (300, 300, 75):
                raise ValueError("real padded momentum-A geometry changed")
            if source_identity_sha256 != GM1_SOURCE_IDENTITY_SHA256:
                raise ValueError("real padded momentum-A source identity changed")
            if (
                physical_receipt_file_sha256
                != GM1_ACCEPTED_PHYSICAL_RECEIPT_FILE_SHA256
                or physical_receipt_semantic_sha256
                != GM1_ACCEPTED_PHYSICAL_RECEIPT_SEMANTIC_SHA256
            ):
                raise ValueError("physical momentum-A evidence pins changed")
            hashes = {name: _sha_array(value) for name, value in values.items()}
            if hashes != dict(GM1_ACCEPTED_PADDED_INPUT_SHA256):
                raise ValueError("accepted padded momentum input bytes changed")
            physical = {
                "u3d": u3d[1:76, 3:303, 3:304],
                "v3d": v3d[1:76, 3:304, 3:303],
                "w3d": w3d[1:77, 3:303, 3:303],
                "ua": ua[1:76, 3:303, 3:304],
                "va": va[1:76, 3:304, 3:303],
                "wa": wa[1:77, 3:303, 3:303],
            }
            if {name: _sha_array(value) for name, value in physical.items()} != dict(
                GM1_ACCEPTED_PHYSICAL_CARRIER_SHA256
            ):
                raise ValueError(
                    "padded carriers do not contain physical token outputs"
                )
        return cls(
            u3d,
            v3d,
            w3d,
            ua,
            va,
            wa,
            physical_token,
            ni,
            nj,
            nk,
            source_identity_sha256,
            physical_receipt_file_sha256,
            physical_receipt_semantic_sha256,
            allow_compact_validation,
        )


def fill_gm1_tiled_solve3_padded_momentum_a_source(
    binding: Gm1TiledSolve3PaddedMomentumABinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> Mapping[str, object]:
    if not isinstance(binding, Gm1TiledSolve3PaddedMomentumABinding):
        raise TypeError("binding must be Gm1TiledSolve3PaddedMomentumABinding")
    padded_nx, padded_ny = binding.ni + 2 * _NG, binding.nj + 2 * _NG
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        padded_nx,
        padded_ny,
    ):
        raise ValueError("tile spec does not cover padded momentum source")
    np = importlib.import_module("numpy")
    shapes = tiled_solve3_padded_momentum_a_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(target, Mapping) or tuple(target) != tuple(shapes):
        raise ValueError("padded momentum-A tile inventory/order changed")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    _require_local_bc2d_support(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        global_nx=padded_nx,
        global_ny=padded_ny,
    )
    active_ny, active_nx = y1 - y0, x1 - x0
    for name, shape in shapes.items():
        value = target[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"{name} tile must be writable C-<f4 {shape}")
        value.fill(np.float32(0.0))
    for name in _PADDED_ORDER:
        target[name][:, :active_ny, :active_nx] = getattr(binding, name)[
            1 : binding.nk + 1, y0:y1, x0:x1
        ]
    target["w_bottom_in"][:active_ny, :active_nx] = binding.w3d[1, y0:y1, x0:x1]
    target["w_bottom_we"][:active_ny, :active_nx] = target["w_bottom_in"][
        :active_ny, :active_nx
    ]
    target["w_bottom_out"][:active_ny, :active_nx] = target["w_bottom_in"][
        :active_ny, :active_nx
    ]
    values = tuple(target.values())
    for index, left in enumerate(values):
        if any(np.shares_memory(left, right) for right in values[index + 1 :]):
            raise ValueError("padded momentum-A tile arrays must not alias")
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "active_shape": [binding.nk, active_ny, active_nx],
            "global_padded_shape": [padded_ny, padded_nx],
            "copy_global_range_yx": [2, binding.nj + 5, 2, binding.ni + 5],
            "source_order": ["bc2d west/east", "bc2d south/north", "padded A copy"],
        }
    )


def cpu_gm1_solve3_padded_momentum_a_resident_inplace(
    binding: Gm1TiledSolve3PaddedMomentumABinding,
) -> None:
    """Literal serial all-open bc2d followed by padded current-to-A."""

    if not isinstance(binding, Gm1TiledSolve3PaddedMomentumABinding):
        raise TypeError("binding must be Gm1TiledSolve3PaddedMomentumABinding")
    ng, ni, nj, nk = _NG, binding.ni, binding.nj, binding.nk
    plane = binding.w3d[1]
    plane[:, :ng] = plane[:, ng : ng + 1]
    plane[:, ng + ni : ng + ni + ng] = plane[:, ng + ni - 1 : ng + ni]
    plane[:ng, :] = plane[ng : ng + 1, :]
    plane[ng + nj : ng + nj + ng, :] = plane[ng + nj - 1 : ng + nj, :]
    region = (slice(1, nk + 1), slice(2, nj + 5), slice(2, ni + 5))
    binding.ua[region] = binding.u3d[region]
    binding.va[region] = binding.v3d[region]
    binding.wa[region] = binding.w3d[region]


def cpu_gm1_tiled_solve3_padded_momentum_a_inplace(
    arrays: Mapping[str, Any],
    *,
    active_nx: int,
    active_ny: int,
    global_x0: int,
    global_y0: int,
    global_nx: int,
    global_ny: int,
) -> None:
    """Apply the two ordered bc2d phases and padded A copy to one tile."""

    np = importlib.import_module("numpy")
    if not isinstance(arrays, Mapping) or tuple(arrays) != _BUFFER_ORDER:
        raise ValueError("CPU padded momentum-A inventory/order changed")
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    global_nx = _positive_int("global_nx", global_nx)
    global_ny = _positive_int("global_ny", global_ny)
    if min(global_x0, global_y0) < 0:
        raise ValueError("global tile origins must be nonnegative")
    _require_local_bc2d_support(
        x0=global_x0,
        x1=global_x0 + active_nx,
        y0=global_y0,
        y1=global_y0 + active_ny,
        global_nx=global_nx,
        global_ny=global_ny,
    )
    source = arrays["w_bottom_in"]
    west_east = arrays["w_bottom_we"]
    output = arrays["w_bottom_out"]
    west_east[:active_ny, :active_nx] = source[:active_ny, :active_nx]
    if global_x0 == 0:
        west_east[:active_ny, :_NG] = source[:active_ny, _NG : _NG + 1]
    if global_x0 + active_nx == global_nx:
        local_edge = global_nx - _NG - 1 - global_x0
        west_east[:active_ny, active_nx - _NG : active_nx] = source[
            :active_ny, local_edge : local_edge + 1
        ]
    output[:active_ny, :active_nx] = west_east[:active_ny, :active_nx]
    if global_y0 == 0:
        output[:_NG, :active_nx] = west_east[_NG : _NG + 1, :active_nx]
    if global_y0 + active_ny == global_ny:
        local_edge = global_ny - _NG - 1 - global_y0
        output[active_ny - _NG : active_ny, :active_nx] = west_east[
            local_edge : local_edge + 1, :active_nx
        ]
    arrays["w3d"][0, :active_ny, :active_nx] = output[:active_ny, :active_nx]
    gx = np.arange(global_x0, global_x0 + active_nx)
    gy = np.arange(global_y0, global_y0 + active_ny)
    mask_x = (gx >= 2) & (gx < global_nx - 1)
    mask_y = (gy >= 2) & (gy < global_ny - 1)
    ys = np.flatnonzero(mask_y)
    xs = np.flatnonzero(mask_x)
    if ys.size and xs.size:
        region = np.ix_(np.arange(arrays["u3d"].shape[0]), ys, xs)
        arrays["ua"][region] = arrays["u3d"][region]
        arrays["va"][region] = arrays["v3d"][region]
        arrays["wa"][region] = arrays["w3d"][region]


def scatter_gm1_tiled_solve3_padded_momentum_a_output(
    binding: Gm1TiledSolve3PaddedMomentumABinding,
    spec: Gm1TileSpec,
    source: Mapping[str, Any],
) -> Mapping[str, object]:
    if not isinstance(binding, Gm1TiledSolve3PaddedMomentumABinding):
        raise TypeError("binding must be Gm1TiledSolve3PaddedMomentumABinding")
    shapes = tiled_solve3_padded_momentum_a_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(source, Mapping) or tuple(source) != tuple(shapes):
        raise ValueError("padded momentum-A scatter inventory/order changed")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    _require_local_bc2d_support(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        global_nx=binding.ni + 2 * _NG,
        global_ny=binding.nj + 2 * _NG,
    )
    active_ny, active_nx = y1 - y0, x1 - x0
    binding.w3d[1, y0:y1, x0:x1] = source["w3d"][0, :active_ny, :active_nx]
    for name in ("ua", "va", "wa"):
        getattr(binding, name)[1 : binding.nk + 1, y0:y1, x0:x1] = source[name][
            :, :active_ny, :active_nx
        ]
    return MappingProxyType(
        {
            "tile": spec.index,
            "padded_owned_yx": [y0, y1, x0, x1],
            "bc2d_words": active_ny * active_nx,
            "copy_words": (
                binding.nk
                * max(0, min(y1, binding.nj + 5) - max(y0, 2))
                * max(0, min(x1, binding.ni + 5) - max(x0, 2))
            ),
            "tke_mutation": False,
            "open_history_mutation": False,
        }
    )


CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE = r"""// gm1.cuda-tilestream-solve3-four-open-padded-momentum-a-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
extern "C" __global__ void gm1_tiled_solve3_bc2d_we_f32_v1(
 const float*src,float*dst,unsigned active_nx,unsigned active_ny,unsigned x0,unsigned global_nx,unsigned tile_nx){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)active_nx*active_ny;
 if(q<n){unsigned i=(unsigned)(q%active_nx),j=(unsigned)(q/active_nx),gx=x0+i,si=i;
  if(gx<3u)si=3u-x0;else if(gx>=global_nx-3u)si=global_nx-4u-x0;
  dst[(unsigned long long)j*tile_nx+i]=src[(unsigned long long)j*tile_nx+si];}
}
extern "C" __global__ void gm1_tiled_solve3_bc2d_sn_f32_v1(
 const float*src,float*dst,unsigned active_nx,unsigned active_ny,unsigned y0,unsigned global_ny,unsigned tile_nx){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)active_nx*active_ny;
 if(q<n){unsigned i=(unsigned)(q%active_nx),j=(unsigned)(q/active_nx),gy=y0+j,sj=j;
  if(gy<3u)sj=3u-y0;else if(gy>=global_ny-3u)sj=global_ny-4u-y0;
  dst[(unsigned long long)j*tile_nx+i]=src[(unsigned long long)sj*tile_nx+i];}
}
extern "C" __global__ void gm1_tiled_solve3_padded_momentum_a_f32_v1(
 const float*u3d,const float*v3d,float*w3d,float*ua,float*va,float*wa,const float*wbottom,
 unsigned active_nx,unsigned active_ny,unsigned x0,unsigned y0,unsigned global_nx,unsigned global_ny,unsigned nk,unsigned tile_nx,unsigned tile_ny){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nk*active_nx*active_ny;
 if(q<n){unsigned i=(unsigned)(q%active_nx);unsigned long long t=q/active_nx;unsigned j=(unsigned)(t%active_ny),k=(unsigned)(t/active_ny);
  unsigned gx=x0+i,gy=y0+j;unsigned long long z=((unsigned long long)k*tile_ny+j)*tile_nx+i;
  float wv=(k==0u)?wbottom[(unsigned long long)j*tile_nx+i]:w3d[z];if(k==0u)w3d[z]=wv;
  if(gx>=2u&&gx<global_nx-1u&&gy>=2u&&gy<global_ny-1u){ua[z]=u3d[z];va[z]=v3d[z];wa[z]=wv;}}
}
"""
CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE_BYTES = (
    CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE.encode("ascii")
)
CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE_BYTES
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledSolve3PaddedMomentumA:
    module_owner: Any
    kernels: tuple[Any, Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_tiled_solve3_padded_momentum_a() -> (
    CompiledGm1TiledSolve3PaddedMomentumA
):
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    common = ("--std=c++14", "--fmad=false")
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE,
        "gm1_tiled_solve3_padded_momentum_a_ptx.cu",
        method="ptx",
    ).compile(tuple(dict.fromkeys((*ptx_options, *common))))
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE,
        "gm1_tiled_solve3_padded_momentum_a_cubin.cu",
        method="cubin",
    ).compile(tuple(dict.fromkeys((*cubin_options, *common))))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode("utf-8")
    for name in CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_KERNEL_NAMES:
        if not re.search(rf"\.entry\s+{name}(?:\s|\()", decoded):
            raise ValueError(f"compiled PTX lacks padded momentum-A entry {name}")
    module = function.Module()
    module.load(cubin)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    runtime = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1TiledSolve3PaddedMomentumA(
        module,
        tuple(
            module.get_function(name)
            for name in CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_KERNEL_NAMES
        ),
        ptx,
        cubin,
        arch,
        runtime,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSolve3PaddedMomentumALaunch:
    ready_event: Any
    compiled: CompiledGm1TiledSolve3PaddedMomentumA
    receipt: Mapping[str, Any]


def launch_cuda_gm1_tiled_solve3_padded_momentum_a(
    arrays: Mapping[str, Any],
    binding: Gm1TiledSolve3PaddedMomentumABinding,
    spec: Gm1TileSpec,
    *,
    stream: Any | None = None,
    compiled: CompiledGm1TiledSolve3PaddedMomentumA | None = None,
) -> CudaGm1TiledSolve3PaddedMomentumALaunch:
    if not isinstance(binding, Gm1TiledSolve3PaddedMomentumABinding):
        raise TypeError("binding must be Gm1TiledSolve3PaddedMomentumABinding")
    cupy = importlib.import_module("cupy")
    shapes = tiled_solve3_padded_momentum_a_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("CUDA padded momentum-A inventory/order changed")
    ranges = []
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 {shape}")
        start = int(value.data.ptr)
        ranges.append((start, start + int(value.nbytes)))
    for index, (left_start, left_end) in enumerate(ranges):
        if any(
            left_start < right_end and right_start < left_end
            for right_start, right_end in ranges[index + 1 :]
        ):
            raise ValueError("CUDA padded momentum-A arrays must not alias")
    physical = _require_physical_token(
        binding.physical_token,
        source_identity_sha256=binding.source_identity_sha256,
        real=not binding.compact_validation,
    )
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    if int(physical["stream_pointer"]) != int(stream.ptr):
        raise ValueError("padded momentum-A must remain on the physical-token stream")
    compiled = (
        compile_cuda_gm1_tiled_solve3_padded_momentum_a()
        if compiled is None
        else compiled
    )
    y0, y1, x0, x1 = spec.owned_rect("mass")
    _require_local_bc2d_support(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        global_nx=binding.ni + 2 * _NG,
        global_ny=binding.nj + 2 * _NG,
    )
    active_ny, active_nx = y1 - y0, x1 - x0
    plane_words = active_ny * active_nx
    copy_words = binding.nk * plane_words
    threads = 256
    we, sn, copy = compiled.kernels
    u32 = cupy.uint32
    with stream:
        we(
            ((plane_words + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                arrays["w_bottom_in"],
                arrays["w_bottom_we"],
                u32(active_nx),
                u32(active_ny),
                u32(x0),
                u32(binding.ni + 2 * _NG),
                u32(spec.tile_nx),
            ),
            stream=stream,
        )
        sn(
            ((plane_words + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                arrays["w_bottom_we"],
                arrays["w_bottom_out"],
                u32(active_nx),
                u32(active_ny),
                u32(y0),
                u32(binding.nj + 2 * _NG),
                u32(spec.tile_nx),
            ),
            stream=stream,
        )
        copy(
            ((copy_words + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                *(arrays[name] for name in _PADDED_ORDER),
                arrays["w_bottom_out"],
                u32(active_nx),
                u32(active_ny),
                u32(x0),
                u32(y0),
                u32(binding.ni + 2 * _NG),
                u32(binding.nj + 2 * _NG),
                u32(binding.nk),
                u32(spec.tile_nx),
                u32(spec.tile_ny),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    record = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_ABI,
        "status": "ENQUEUED_FOUR_OPEN_BC2D_AND_PADDED_MOMENTUM_A_TILE",
        "tile": spec.index,
        "padded_owned_yx": [y0, y1, x0, x1],
        "source_identity_sha256": binding.source_identity_sha256,
        "physical_operation_token_sha256": physical["operation_token_sha256"],
        "physical_receipt_file_sha256": binding.physical_receipt_file_sha256,
        "source_order": ["bc2d west/east", "bc2d south/north", "padded momentum A"],
        "same_stream_no_host_sync": True,
        "stream_pointer": int(stream.ptr),
        "kernel_source_sha256": CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve3_sha256": AUTHORITY_SOLVE3_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "refusals": {
            "tke_commit": True,
            "scalar_current_to_a": True,
            "reciprocal_density": True,
            "global_psolver": True,
            "mpi": True,
            "open_history_mutation": True,
            "full_solve3_or_rk": True,
            "five_meter_meteorological_product": True,
        },
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return CudaGm1TiledSolve3PaddedMomentumALaunch(
        event, compiled, MappingProxyType(record)
    )


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3PaddedMomentumAToken:
    receipt: Mapping[str, Any]


def finish_gm1_tiled_solve3_padded_momentum_a(
    binding: Gm1TiledSolve3PaddedMomentumABinding,
    plan: Gm1TilePlan,
    launches: Sequence[CudaGm1TiledSolve3PaddedMomentumALaunch],
) -> Gm1TiledSolve3PaddedMomentumAToken:
    if not isinstance(binding, Gm1TiledSolve3PaddedMomentumABinding):
        raise TypeError("binding must be Gm1TiledSolve3PaddedMomentumABinding")
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (
        binding.ni + 2 * _NG,
        binding.nj + 2 * _NG,
    ):
        raise ValueError("padded momentum-A completion plan differs from domain")
    if len(launches) != plan.ntiles:
        raise ValueError("padded momentum-A completion requires every exact tile")
    for spec in plan.specs:
        y0, y1, x0, x1 = spec.owned_rect("mass")
        _require_local_bc2d_support(
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            global_nx=plan.nx,
            global_ny=plan.ny,
        )
    physical = _require_physical_token(
        binding.physical_token,
        source_identity_sha256=binding.source_identity_sha256,
        real=not binding.compact_validation,
    )
    tokens = []
    streams = set()
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledSolve3PaddedMomentumALaunch):
            raise TypeError("padded momentum-A completion requires exact launches")
        receipt = launch.receipt
        if (
            launch.ready_event is None
            or receipt.get("schema")
            != CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SCHEMA
            or receipt.get("status")
            != "ENQUEUED_FOUR_OPEN_BC2D_AND_PADDED_MOMENTUM_A_TILE"
            or receipt.get("tile") != spec.index
            or receipt.get("source_identity_sha256") != binding.source_identity_sha256
            or receipt.get("physical_operation_token_sha256")
            != physical["operation_token_sha256"]
        ):
            raise ValueError("padded momentum-A tile receipt identity/order differs")
        detached = dict(receipt)
        token = detached.pop("operation_token_sha256", None)
        if token != hashlib.sha256(_canonical(detached)).hexdigest():
            raise ValueError("padded momentum-A tile token changed")
        tokens.append(token)
        streams.add(int(receipt["stream_pointer"]))
    if streams != {int(physical["stream_pointer"])}:
        raise ValueError("padded momentum-A tiles changed streams")
    np = importlib.import_module("numpy")
    outputs = {name: getattr(binding, name) for name in _PADDED_ORDER}
    if not all(bool(np.all(np.isfinite(value))) for value in outputs.values()):
        raise ValueError("completed padded momentum-A output is non-finite")
    hashes = {name: _sha_array(value) for name, value in outputs.items()}
    if not binding.compact_validation and hashes != dict(
        GM1_ACCEPTED_PADDED_OUTPUT_SHA256
    ):
        raise ValueError("real padded momentum-A output differs from accepted oracle")
    record = {
        "schema": "gm1.tilestream-solve3-padded-momentum-a-token/v1",
        "status": "ADMITTED_PADDED_MOMENTUM_A_ONLY",
        "source_identity_sha256": binding.source_identity_sha256,
        "physical_operation_token_sha256": physical["operation_token_sha256"],
        "physical_receipt": {
            "file_sha256": binding.physical_receipt_file_sha256,
            "semantic_sha256": binding.physical_receipt_semantic_sha256,
        },
        "tile_count": plan.ntiles,
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_operation_tokens": tokens,
        "stream_pointer": next(iter(streams)),
        "outputs": hashes,
        "w_bottom_sha256": _sha_array(binding.w3d[1]),
        "source_order": [
            "accepted physical momentum-A token",
            "bc2d west/east then south/north",
            "padded u3d/v3d/w3d to ua/va/wa",
        ],
        "ready_for": "independent TKE-commit admission audit only",
        "tke_commit_available": False,
        "claim_limit": GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return Gm1TiledSolve3PaddedMomentumAToken(MappingProxyType(record))


def five_meter_tiled_solve3_padded_momentum_a_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192
) -> Mapping[str, object]:
    shapes = tiled_solve3_padded_momentum_a_buffer_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=75
    )
    per_buffer = sum(math.prod(shape) * 4 for shape in shapes.values())
    record = {
        "schema": "gm1.hybrid-tilestream-solve3-padded-momentum-a-memory/v1",
        "domain": [75, 3000, 3000],
        "padded_plane": [3006, 3006],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3006 / tile_nx) * math.ceil(3006 / tile_ny),
        "bc2d_boundary_support_local": (
            _boundary_aligned_tile_extent(3006, tile_nx) == tile_nx
            and _boundary_aligned_tile_extent(3006, tile_ny) == tile_ny
        ),
        "carriers": list(shapes),
        "host_per_buffer_bytes": per_buffer,
        "device_per_buffer_bytes": per_buffer,
        "double_host_bytes": 2 * per_buffer,
        "double_device_bytes": 2 * per_buffer,
        "full_domain_duplicate_bytes": 0,
        "canonical_store_mutations": ["w3d(k=1) padded plane", "ua", "va", "wa"],
        "global_pressure_workspace_executed": False,
        "tke_commit_executed": False,
        "open_history_mutation": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_BC_SHA256",
    "AUTHORITY_SOLVE3_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_ABI",
    "CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_KERNEL_NAMES",
    "CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE",
    "CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE_BYTES",
    "CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE_SHA256",
    "CompiledGm1TiledSolve3PaddedMomentumA",
    "CudaGm1TiledSolve3PaddedMomentumALaunch",
    "GM1_ACCEPTED_PADDED_INPUT_SHA256",
    "GM1_ACCEPTED_PADDED_OUTPUT_SHA256",
    "GM1_ACCEPTED_PADDED_W_BOTTOM_SHA256",
    "GM1_ACCEPTED_PHYSICAL_CARRIER_SHA256",
    "GM1_ACCEPTED_PHYSICAL_OUTPUT_SHA256",
    "GM1_ACCEPTED_PHYSICAL_RECEIPT_FILE_SHA256",
    "GM1_ACCEPTED_PHYSICAL_RECEIPT_SEMANTIC_SHA256",
    "GM1_SOURCE_IDENTITY_SHA256",
    "GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_CLAIM_LIMIT",
    "Gm1TiledSolve3PaddedMomentumABinding",
    "Gm1TiledSolve3PaddedMomentumAToken",
    "build_gm1_solve3_padded_momentum_a_plan",
    "compile_cuda_gm1_tiled_solve3_padded_momentum_a",
    "cpu_gm1_solve3_padded_momentum_a_resident_inplace",
    "cpu_gm1_tiled_solve3_padded_momentum_a_inplace",
    "fill_gm1_tiled_solve3_padded_momentum_a_source",
    "finish_gm1_tiled_solve3_padded_momentum_a",
    "five_meter_tiled_solve3_padded_momentum_a_inventory",
    "launch_cuda_gm1_tiled_solve3_padded_momentum_a",
    "padded_momentum_a_shapes",
    "scatter_gm1_tiled_solve3_padded_momentum_a_output",
    "tiled_solve3_padded_momentum_a_buffer_shapes",
)
