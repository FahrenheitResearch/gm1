"""Tile-local public-ArWen WSM6 boundary for the stretched GM1 child.

This module begins after the admitted, source-ordered ``pdefq`` finish.  It
reconstructs only the WSM6-ready thermodynamic carriers, invokes the existing
CM1-owned public ArWen component, and publishes the six moisture species plus
the eleven component-persistence fields.  ``thp`` is an ephemeral output that
must be drained into the subsequent CM1 post-physics rebuild before a tile
buffer is reused.

WSM6 owns complete vertical columns and has no horizontal stencil.  Fixed-size
edge tiles therefore use an exact nearest-physical-column transport pad; pad
words are executed to keep one CUDA shape but are never scattered.  The
selected source order is tile rows south-to-north and tiles west-to-east.

This is not the post-WSM6 CM1 EOS rebuild, ``qa`` publication, solve3,
pressure, boundary-history update, MPI reduction, a five-metre meteorological
product, or a coupled RK step.  Those boundaries remain fail-closed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .engine_capacity import capacity_level_bounds
from .arwen_wsm6_component import (
    WSM6_COMPONENT_PERSISTENCE,
    WSM6_SPEC,
    WSM6_SPECIES,
    WSM6_SURFACE_SCRATCH,
)
from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_gm1_tilestream_column import (
    ARWEN_WSM6_ESTATE_AUDIT_SHA256,
    MORRISON_DENSE_ICE_ORACLE_RECEIPT_SHA256,
    TILESTREAM_NATIVE_ORACLE_RECEIPT_SHA256,
    TILESTREAM_RECORD_IDENTITY,
    TILESTREAM_REFLECTIVITY_AUTHORITY,
    CudaGm1TiledColumnLaunch,
    launch_cuda_gm1_tiled_column_block,
    require_tiled_column_device_arrays,
    tiled_column_buffer_order,
    tiled_column_device_buffer_shapes,
    tiled_column_host_buffer_shapes,
)
from .dense_ice import DEFAULT_IHAIL


CUDA_GM1_TILESTREAM_WSM6_SCHEMA = TILESTREAM_RECORD_IDENTITY["wsm6-mp6"]["schema"]
CUDA_GM1_TILESTREAM_WSM6_ABI = TILESTREAM_RECORD_IDENTITY["wsm6-mp6"]["abi"]
GM1_ACCEPTED_REAL_PDEFQ_Q3D_SHA256 = (
    "aa94a2818c0a0f53f736e70c2690f3654da9e74678bd515f4d4dad5f2af07f9a"
)
GM1_ACCEPTED_REAL_PDEFQ_RECEIPT_SHA256 = (
    "7575dcf0e39ad6970cb83fc39b70840d41d2f06d0d11c6391d75f4e3cc059379"
)
ARWEN_WSM6_NATIVE_ORACLE_RECEIPT_SHA256 = (
    TILESTREAM_NATIVE_ORACLE_RECEIPT_SHA256["wsm6-mp6"]
)
#: The reflectivity formula this seam publishes is the one WRF's WSM6 owns
#: (see TILESTREAM_REFLECTIVITY_AUTHORITY, which the column launcher records).
WSM6_REFLECTIVITY_AUTHORITY = TILESTREAM_REFLECTIVITY_AUTHORITY["wsm6-mp6"]
GM1_TILESTREAM_WSM6_CLAIM_LIMIT = (
    "tile-local CM1-to-ArWen staging, one public WSM6 call per physical column, "
    "and publication of q3d/thp plus eleven WSM6 persistence carriers after the "
    "accepted pdefq boundary; no qa mutation, CM1 post-physics EOS, solve3, "
    "pressure, halo/open-history mutation, MPI reduction, five-metre "
    "meteorological product, or coupled-step claim"
)

_ORDER = tiled_column_buffer_order(WSM6_SPEC)
_MASS_INPUTS = ("pi0", "th0", "pp3d", "th3d", "p", "rho")
_VOLUME_PERSISTENCE = _ORDER["volume_persistence"]
_HOST_INPUT_ORDER = _ORDER["host_inputs"]
_HOST_BUFFER_ORDER = _ORDER["host_buffers"]
_DEVICE_BUFFER_ORDER = _ORDER["device_buffers"]
GM1_TILESTREAM_WSM6_H2D_NAMES = _HOST_INPUT_ORDER
GM1_TILESTREAM_WSM6_D2H_NAMES = _ORDER["d2h"]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_sha256(name: str, value: object) -> str:
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


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_f32(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive finite scalar") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def tiled_wsm6_host_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact pinned-host union of WSM6 inputs and drainable outputs."""

    return tiled_column_host_buffer_shapes(
        WSM6_SPEC, tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
    )


def tiled_wsm6_device_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact exposed device arrays; public-component scratch is separate."""

    return tiled_column_device_buffer_shapes(
        WSM6_SPEC, tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
    )


@dataclass(frozen=True, slots=True)
class Gm1TiledWsm6SourceBinding:
    """Writable physical q/persistence and immutable WSM6 staging sources."""

    q3d: Any
    pi0: Any
    th0: Any
    pp3d: Any
    th3d: Any
    p: Any
    rho: Any
    phb: Any
    persistence: Mapping[str, Any]
    nx: int
    ny: int
    nk: int
    source_identity_sha256: str
    pdefq_receipt_sha256: str
    completed_calls: int
    compact_validation: bool
    capacity_profile: str | None = None

    @classmethod
    def create(
        cls,
        *,
        q3d: Any,
        pi0: Any,
        th0: Any,
        pp3d: Any,
        th3d: Any,
        p: Any,
        rho: Any,
        phb: Any,
        persistence: Mapping[str, Any],
        source_identity_sha256: str,
        pdefq_receipt_sha256: str,
        completed_calls: int,
        allow_compact_validation: bool = False,
        capacity_profile: str | None = None,
    ) -> "Gm1TiledWsm6SourceBinding":
        np = importlib.import_module("numpy")
        if (
            not isinstance(q3d, np.ndarray)
            or q3d.dtype != np.dtype("<f4")
            or q3d.ndim != 4
            or q3d.shape[0] != len(WSM6_SPECIES)
            or not q3d.flags.c_contiguous
            or not q3d.flags.writeable
        ):
            raise ValueError("q3d must be writable host C-<f4 [6,nk,ny,nx]")
        _, nk, ny, nx = map(int, q3d.shape)
        mass = (nk, ny, nx)
        expected = {
            "pi0": mass,
            "th0": mass,
            "pp3d": mass,
            "th3d": mass,
            "p": mass,
            "rho": mass,
            "phb": (nk + 1, ny, nx),
        }
        sources = {
            "pi0": pi0,
            "th0": th0,
            "pp3d": pp3d,
            "th3d": th3d,
            "p": p,
            "rho": rho,
            "phb": phb,
        }
        for name, shape in expected.items():
            value = sources[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f4")
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"{name} must be host C-<f4 {shape}")
        if not isinstance(persistence, Mapping) or tuple(persistence) != (
            WSM6_COMPONENT_PERSISTENCE
        ):
            raise ValueError("persistence inventory/order differs from public WSM6 ABI")
        copied_persistence: dict[str, Any] = {}
        for name in WSM6_COMPONENT_PERSISTENCE:
            value = persistence[name]
            shape = (ny, nx) if name in WSM6_SURFACE_SCRATCH else mass
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f4")
                or value.shape != shape
                or not value.flags.c_contiguous
                or not value.flags.writeable
            ):
                raise ValueError(f"persistence {name} must be writable host C-<f4 {shape}")
            copied_persistence[name] = value
        values = [q3d, *sources.values(), *copied_persistence.values()]
        if not all(bool(np.all(np.isfinite(value))) for value in values):
            raise ValueError("tiled WSM6 source contains non-finite words")
        if not bool(np.all(q3d >= np.float32(0.0))):
            raise ValueError("post-pdefq q3d must be nonnegative")
        if not bool(np.all(p > np.float32(0.0))) or not bool(
            np.all(rho > np.float32(0.0))
        ):
            raise ValueError("WSM6 pressure and density must be strictly positive")
        if not bool(np.all(np.diff(phb, axis=0) > np.float32(0.0))):
            raise ValueError("total geopotential interfaces must increase vertically")
        if not bool(np.all(pi0 + pp3d > np.float32(0.0))) or not bool(
            np.all(th0 + th3d > np.float32(0.0))
        ):
            raise ValueError("CM1 total Exner and theta must be strictly positive")
        mutable = [q3d, *copied_persistence.values()]
        for index, left in enumerate(mutable):
            if any(np.shares_memory(left, right) for right in mutable[index + 1 :]):
                raise ValueError("q3d and WSM6 persistence outputs must not alias")
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        pdefq_receipt_sha256 = _require_sha256(
            "pdefq_receipt_sha256", pdefq_receipt_sha256
        )
        minimum, maximum = capacity_level_bounds(capacity_profile)
        if allow_compact_validation:
            if nk < minimum or nk > maximum:
                raise ValueError(
                    f"compact WSM6 validation requires {minimum} <= nk <= {maximum}"
                )
        else:
            if (nx, ny, nk) != (300, 300, 75):
                raise ValueError("real tiled WSM6 is sealed to the accepted 300x300x75 child")
            q_sha = hashlib.sha256(q3d.tobytes(order="C")).hexdigest()
            if q_sha != GM1_ACCEPTED_REAL_PDEFQ_Q3D_SHA256:
                raise ValueError("q3d differs from the accepted sealed pdefq publication")
            if pdefq_receipt_sha256 != GM1_ACCEPTED_REAL_PDEFQ_RECEIPT_SHA256:
                raise ValueError("pdefq receipt differs from the accepted sealed gate")
        return cls(
            q3d,
            pi0,
            th0,
            pp3d,
            th3d,
            p,
            rho,
            phb,
            MappingProxyType(copied_persistence),
            nx,
            ny,
            nk,
            _require_sha256("source_identity_sha256", source_identity_sha256),
            pdefq_receipt_sha256,
            _nonnegative_int("completed_calls", completed_calls),
            allow_compact_validation,
            capacity_profile,
        )


def _edge_indices(np: Any, start: int, count: int, limit: int) -> Any:
    return np.minimum(np.arange(start, start + count, dtype=np.int64), limit - 1)


def _copy_edge_clamped(
    np: Any, destination: Any, source: Any, *, y0: int, x0: int, ny: int, nx: int
) -> None:
    ys = _edge_indices(np, y0, ny, source.shape[-2])
    xs = _edge_indices(np, x0, nx, source.shape[-1])
    selected = np.take(np.take(source, ys, axis=-2), xs, axis=-1)
    np.copyto(destination, selected)


def fill_gm1_tiled_wsm6_source(
    binding: Gm1TiledWsm6SourceBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> Mapping[str, object]:
    """Fill one fixed tile; pad only beyond the physical north/east edges."""

    if not isinstance(binding, Gm1TiledWsm6SourceBinding):
        raise TypeError("binding must be Gm1TiledWsm6SourceBinding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("tile spec does not cover the WSM6 source domain")
    np = importlib.import_module("numpy")
    shapes = tiled_wsm6_host_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(target, Mapping) or tuple(target) != tuple(shapes):
        raise ValueError("tiled WSM6 host target inventory/order changed")
    for name, shape in shapes.items():
        value = target[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"host target {name} must be writable C-<f4 {shape}")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    for name in _MASS_INPUTS:
        _copy_edge_clamped(
            np,
            target[name],
            getattr(binding, name),
            y0=y0,
            x0=x0,
            ny=spec.tile_ny,
            nx=spec.tile_nx,
        )
    _copy_edge_clamped(
        np,
        target["phb"],
        binding.phb,
        y0=y0,
        x0=x0,
        ny=spec.tile_ny,
        nx=spec.tile_nx,
    )
    _copy_edge_clamped(
        np,
        target["q3d"],
        binding.q3d,
        y0=y0,
        x0=x0,
        ny=spec.tile_ny,
        nx=spec.tile_nx,
    )
    for name in WSM6_COMPONENT_PERSISTENCE:
        _copy_edge_clamped(
            np,
            target[name],
            binding.persistence[name],
            y0=y0,
            x0=x0,
            ny=spec.tile_ny,
            nx=spec.tile_nx,
        )
    target["thp"].fill(np.float32(0.0))
    active_ny, active_nx = y1 - y0, x1 - x0
    record = {
        "schema": "gm1.hybrid-tilestream-wsm6-source/v1",
        "tile": spec.index,
        "owned_yx": [y0, y1, x0, x1],
        "active_ny": active_ny,
        "active_nx": active_nx,
        "source_identity_sha256": binding.source_identity_sha256,
        "pdefq_receipt_sha256": binding.pdefq_receipt_sha256,
        "padding": {
            "policy": "nearest physical north/east column; transport only",
            "columns_executed_not_published": spec.tile_nx * spec.tile_ny
            - active_nx * active_ny,
        },
        "claim_limit": GM1_TILESTREAM_WSM6_CLAIM_LIMIT,
    }
    return MappingProxyType(record)


def scatter_gm1_tiled_wsm6_output(
    binding: Gm1TiledWsm6SourceBinding,
    spec: Gm1TileSpec,
    source: Mapping[str, Any],
) -> Mapping[str, object]:
    """Publish q/persistence physical words; leave ``thp`` for the next drain."""

    if not isinstance(binding, Gm1TiledWsm6SourceBinding):
        raise TypeError("binding must be Gm1TiledWsm6SourceBinding")
    shapes = tiled_wsm6_host_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(source, Mapping) or tuple(source) != tuple(shapes):
        raise ValueError("tiled WSM6 output inventory/order changed")
    np = importlib.import_module("numpy")
    for name, shape in shapes.items():
        value = source[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"host output {name} must be C-<f4 {shape}")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    np.copyto(
        binding.q3d[:, :, y0:y1, x0:x1],
        source["q3d"][:, :, :active_ny, :active_nx],
    )
    for name in WSM6_COMPONENT_PERSISTENCE:
        if name in WSM6_SURFACE_SCRATCH:
            np.copyto(
                binding.persistence[name][y0:y1, x0:x1],
                source[name][:active_ny, :active_nx],
            )
        else:
            np.copyto(
                binding.persistence[name][:, y0:y1, x0:x1],
                source[name][:, :active_ny, :active_nx],
            )
    record = {
        "schema": "gm1.hybrid-tilestream-wsm6-scatter/v1",
        "tile": spec.index,
        "owned_yx": [y0, y1, x0, x1],
        "published_q_words": len(WSM6_SPECIES)
        * binding.nk
        * active_ny
        * active_nx,
        "published_persistence_words": (
            len(_VOLUME_PERSISTENCE) * binding.nk
            + len(WSM6_SURFACE_SCRATCH)
        )
        * active_ny
        * active_nx,
        "thp_disposition": "must be drained before tile-buffer reuse; not scattered here",
        "padding_published": False,
        "qa_mutated": False,
        "post_wsm6_eos_performed": False,
    }
    return MappingProxyType(record)


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], *, nx: int, ny: int, nk: int
) -> dict[str, Any]:
    return require_tiled_column_device_arrays(
        cupy, WSM6_SPEC, arrays, nx=nx, ny=ny, nk=nk
    )


#: The launch token is component-independent; the WSM6 name is retained
#: because the post-WSM6 EOS boundary annotates its argument with it.
CudaGm1TiledWsm6Launch = CudaGm1TiledColumnLaunch


def launch_cuda_gm1_tiled_wsm6_block(
    arrays: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    source_identity_sha256: str,
    pdefq_receipt_sha256: str,
    completed_calls: int,
    block_index: int,
    active_nx: int,
    active_ny: int,
    ztop: float | None = None,
    stream: Any | None = None,
    capacity_profile: str | None = None,
    ihail: int = DEFAULT_IHAIL,
    refl_10cm_due: bool = False,
) -> CudaGm1TiledColumnLaunch:
    """Stage and enqueue one independent fixed-shape WSM6 column block."""

    return launch_cuda_gm1_tiled_column_block(
        WSM6_SPEC,
        arrays,
        nx=nx,
        ny=ny,
        nk=nk,
        dx=dx,
        dy=dy,
        dz=dz,
        dt=dt,
        source_identity_sha256=source_identity_sha256,
        pdefq_receipt_sha256=pdefq_receipt_sha256,
        completed_calls=completed_calls,
        block_index=block_index,
        active_nx=active_nx,
        active_ny=active_ny,
        ihail=ihail,
        claim_limit=GM1_TILESTREAM_WSM6_CLAIM_LIMIT,
        stream=stream,
        capacity_profile=capacity_profile,
        refl_10cm_due=refl_10cm_due,
    )


def launch_cuda_gm1_tiled_wsm6_tile(
    arrays: Mapping[str, Any],
    plan: Gm1TilePlan,
    spec: Gm1TileSpec,
    *,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    source_identity_sha256: str,
    pdefq_receipt_sha256: str,
    completed_calls: int,
    stream: Any | None = None,
    refl_10cm_due: bool = False,
) -> CudaGm1TiledWsm6Launch:
    """Plan-bound wrapper around :func:`launch_cuda_gm1_tiled_wsm6_block`."""

    if not isinstance(plan, Gm1TilePlan) or not isinstance(spec, Gm1TileSpec):
        raise TypeError("plan/spec must be Gm1TilePlan/Gm1TileSpec")
    if spec not in plan.specs:
        raise ValueError("tile spec is not a member of the supplied plan")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    return launch_cuda_gm1_tiled_wsm6_block(
        arrays,
        nx=spec.tile_nx,
        ny=spec.tile_ny,
        nk=nk,
        dx=dx,
        dy=dy,
        dz=dz,
        dt=dt,
        source_identity_sha256=source_identity_sha256,
        pdefq_receipt_sha256=pdefq_receipt_sha256,
        completed_calls=completed_calls,
        block_index=spec.index,
        active_nx=x1 - x0,
        active_ny=y1 - y0,
        stream=stream,
        refl_10cm_due=refl_10cm_due,
    )


def five_meter_tiled_wsm6_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, Any]:
    """Incremental 3000-square memory and lifecycle inventory only."""

    host = tiled_wsm6_host_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nk=nk)
    device = tiled_wsm6_device_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nk=nk)
    host_bytes = sum(math.prod(shape) * 4 for shape in host.values())
    device_bytes = sum(math.prod(shape) * 4 for shape in device.values())
    mass_bytes = nk * tile_ny * tile_nx * 4
    zface_bytes = (nk + 1) * tile_ny * tile_nx * 4
    component_scratch = 5 * mass_bytes + zface_bytes
    plan = {
        "schema": "gm1.hybrid-tilestream-wsm6-five-meter-inventory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [nk, tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "host_buffer_bytes_each": host_bytes,
        "double_host_buffers_bytes": 2 * host_bytes,
        "exposed_device_buffer_bytes_each": device_bytes,
        "public_component_scratch_bytes_each": component_scratch,
        "device_working_set_bytes_each": device_bytes + component_scratch,
        "double_device_working_sets_bytes": 2 * (device_bytes + component_scratch),
        "full_domain_derived_thp_alt_php_duplicate_bytes": 0,
        "full_domain_wsm6_scratch_duplicate_bytes": 0,
        "pdefq_budget_f64_bytes_before_drain": 6 * 3000 * 8,
        "pdefq_budget_lifetime": "finish event precedes WSM6; drain/reuse before tiles",
        "pressure_workspace_executed": False,
        "pressure_workspace_incremental_bytes": 0,
        "thp_lifetime": "tile-local; downstream CM1 EOS must drain before reuse",
        "canonical_store_updates": ["q3d", *WSM6_COMPONENT_PERSISTENCE],
        "claim_limit": GM1_TILESTREAM_WSM6_CLAIM_LIMIT,
    }
    plan["inventory_sha256"] = hashlib.sha256(_canonical(plan)).hexdigest()
    return MappingProxyType(plan)


__all__ = [
    "ARWEN_WSM6_ESTATE_AUDIT_SHA256",
    "MORRISON_DENSE_ICE_ORACLE_RECEIPT_SHA256",
    "ARWEN_WSM6_NATIVE_ORACLE_RECEIPT_SHA256",
    "CUDA_GM1_TILESTREAM_WSM6_ABI",
    "CUDA_GM1_TILESTREAM_WSM6_SCHEMA",
    "CudaGm1TiledWsm6Launch",
    "GM1_ACCEPTED_REAL_PDEFQ_Q3D_SHA256",
    "GM1_ACCEPTED_REAL_PDEFQ_RECEIPT_SHA256",
    "GM1_TILESTREAM_WSM6_CLAIM_LIMIT",
    "GM1_TILESTREAM_WSM6_D2H_NAMES",
    "GM1_TILESTREAM_WSM6_H2D_NAMES",
    "Gm1TiledWsm6SourceBinding",
    "fill_gm1_tiled_wsm6_source",
    "five_meter_tiled_wsm6_inventory",
    "launch_cuda_gm1_tiled_wsm6_block",
    "launch_cuda_gm1_tiled_wsm6_tile",
    "scatter_gm1_tiled_wsm6_output",
    "tiled_wsm6_device_buffer_shapes",
    "tiled_wsm6_host_buffer_shapes",
]
