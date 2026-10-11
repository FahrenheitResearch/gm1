"""Tile-local source-exact stretched closure for the GM1 child.

This module composes four already sealed operators on one supplied CUDA
stream: stretched terrain/static metrics, moist ``calcnm``, stretched
``calcdef``, and fresh glibc-2.39-exact Smagorinsky/TKE initialization.  It
does not publish a child, checkpoint, or forecast.  The caller owns host
gather/scatter and may publish only each :class:`Gm1TileSpec` mass rectangle.

An explicit checkpoint-bound mode replaces only fresh TKE initialization with
the later-step CM1 lifecycle: retained post-solve3 ``tkea`` followed by the
already sealed source-exact SGS/TKE operator.  It cannot be selected without
the exact retained carrier, its full checkpoint SHA-256, and private scratch.

The generic TKE domain gate is deliberately split across the tile sequence.
One device counter is zeroed before the first tile, atomically accumulated by
all tiles, and copied to the host once in ``finish``.  A per-tile host read or
the synchronizing public generic-TKE launcher is not used here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cm1core.cuda_gm1_child_nm_def import (
    CHILD_MOIST_ALL_NAMES,
    CHILD_MOIST_OUTPUT_NAMES,
    CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256,
    CompiledChildMoistNmKernel,
    compile_cuda_child_moist_nm,
    launch_cuda_child_moist_nm,
)
from cm1core.cuda_gm1_child_stretched_nm_def import (
    CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256,
    CompiledChildStretchedCalcdef,
    compile_cuda_gm1_child_stretched_calcdef,
)
from cm1core.cuda_gm1_child_turbsmag_init import (
    INPUT_ARRAY_NAMES as TURBSMAG_INPUT_NAMES,
    OUTPUT_ARRAY_NAMES as TURBSMAG_OUTPUT_NAMES,
)
from cm1core.cuda_gm1_child_turbsmag_init_generic import (
    CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256,
    CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS,
    CompiledGenericChildTurbsmagInit,
    compile_cuda_child_turbsmag_init_generic,
)
from cm1core.cuda_gm1_tilestream import (
    GM1_SOURCE_STENCIL_RADIUS,
    Gm1TilePlan,
    Gm1TileSpec,
)
from cm1core.cuda_gm1_tilestream_sgs_startup_ramp import (
    CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_MODULE_SHA256,
    CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_SCHEMA,
    CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_STATUS,
    CudaGm1TiledSgsStartupRampBinding,
    enqueue_cuda_gm1_tilestream_sgs_startup_ramp,
    require_cuda_gm1_tilestream_sgs_startup_ramp_binding,
)
from cm1core.cuda_stretch_z1 import (
    CUDA_STRETCH_Z1_SOURCE_SHA256,
    CudaStretchZ1Launch,
    StretchZ1Parameters,
    derive_stretch_z1,
)
from cm1core.cuda_stretch_z1_terrain import (
    CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256,
    STRETCH_Z1_CHILD_15KM_PARAMETERS,
    CompiledStretchZ1Terrain,
    compile_cuda_stretch_z1_terrain,
    stretch_z1_terrain_shapes,
)
from cm1core.cuda_sgs_tke import (
    CUDA_SGS_TKE_SOURCE_SHA256,
    CompiledSgsTkeKernels,
    launch_cuda_sgs_tke,
)
from cm1core.cuda_terrain_deformation import (
    CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256,
    CUDA_TERRAIN_DEFORMATION_THREADS,
    terrain_deformation_scratch_shapes,
)
from cm1core.errors import refuse
from cm1core.gm1_child_stretched_closure_v3 import CARRIER_ORDER
from cm1core.grid_geometry import CartesianGridGeometry


CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_SCHEMA = (
    "gm1.cuda-tilestream-stretched-fresh-closure/v1"
)
CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_ABI = (
    "gm1.cuda-tilestream-stretched-fresh-closure-f32/v1"
)
CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_REDUCTION_SCHEMA = (
    "gm1.cuda-tilestream-stretched-fresh-closure-reduction/v1"
)
CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA = (
    "gm1.cuda-tilestream-stretched-fresh-closure-token/v1"
)
CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_STATUS = (
    "ENQUEUED_TILE_LOCAL_CALCNM_CALCDEF_FRESH_TKE_PENDING_GLOBAL_REDUCTION"
)
CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_CLAIM_LIMIT = (
    "tile-local source-exact live Nm, stretched calcdef, and fresh closure "
    "coefficients/TKE only; caller-owned physical gather and exact mass-owned "
    "scatter remain required, the shared invalid counter must be admitted by "
    "the final reduction, and no complete-child envelope, RK stage, pressure "
    "solve, checkpoint, forecast, storm, or tornado claim is made"
)

SOURCE_ORDER = (
    "stretched terrain/static metrics on the compute window",
    "three-dimensional-base moist calcnm",
    "profile-aware terrain calcdef",
    "aggregate metric-domain validation",
    "fresh glibc-2.39-exact closure coefficients and TKE",
    "rigid vertical-face disposition and exact tke3d/tketen copy",
)
SOURCE_ORDER_WITH_ACTIVE_SGS_STARTUP_RAMP = (
    *SOURCE_ORDER[:4],
    "active ramp_sgs=1 establishes exact cme/ce1/ce2/csm before turbulence",
    "fresh closure km/kh/dissten/tkea refreshed with ramped coefficients",
    SOURCE_ORDER[-1],
)
OPERATOR_SOURCE_SHA256 = MappingProxyType(
    {
        "static_metrics": CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256,
        "calcnm": CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256,
        "stretched_calcdef": CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256,
        "base_calcdef": CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256,
        "fresh_tke": CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256,
    }
)
RETAINED_TKE_SOURCE_ORDER = (
    "stretched terrain/static metrics on the compute window",
    "three-dimensional-base moist calcnm",
    "profile-aware terrain calcdef",
    "aggregate metric-domain validation",
    "exact restored post-solve3 tkea tile bound by checkpoint SHA-256",
    "exact restored closure-refresh grdscl top-face disposition",
    "source-exact tkekm/tkebc coefficients around retained tkea",
    "exact tke3d copy with source-produced tketen",
)
RETAINED_TKE_CLAIM_LIMIT = (
    "tile-local source-exact live Nm, stretched calcdef, and SGS/TKE "
    "coefficients around one exact restored post-solve3 tkea checkpoint; "
    "the caller binds the full checkpoint SHA-256, the exact restored closure "
    "operation and grdscl carrier SHA-256, and supplies each physical tile "
    "without substitution, while gather/scatter, RK, pressure solve, checkpoint "
    "publication, forecast, storm, tornado, and five-metre claims remain excluded"
)
RETAINED_TKE_SCRATCH_NAMES = (
    "tk",
    "grdscl_full",
    "rgrdscl",
    "cme",
    "ce1",
    "ce2",
    "csm",
)

INPUT_NAMES = (
    "terrain_msl",
    "u3d",
    "v3d",
    "w3d",
    "rho",
    "rf",
    "pi0",
    "th0",
    "prs",
    "pp3d",
    "th3d",
    "q3d",
)
MOIST_SCRATCH_NAMES = tuple(name for name in CHILD_MOIST_OUTPUT_NAMES if name != "nm")
STATIC_NAMES = tuple(
    name
    for name in stretch_z1_terrain_shapes(1, 1)
    if name not in {"terrain_msl", "rds", "rdsf"}
)
WORKSPACE_NAMES = (
    *STATIC_NAMES,
    "rds",
    "rdsf",
    *MOIST_SCRATCH_NAMES,
    *terrain_deformation_scratch_shapes(3, 3, 3),
    "ruh",
    "rvh",
    *CARRIER_ORDER,
)
if len(WORKSPACE_NAMES) != len(set(WORKSPACE_NAMES)):
    raise RuntimeError("tile-local stretched closure workspace names alias")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _positive_int(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _spacing(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def tiled_stretched_closure_input_shapes(
    ni: int, nj: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact physical compute-window inputs, including horizontal staggering."""

    ni = _positive_int("ni", ni)
    nj = _positive_int("nj", nj)
    nk = _positive_int("nk", nk, 3)
    cell, face = (nk, nj, ni), (nk + 1, nj, ni)
    return MappingProxyType(
        {
            "terrain_msl": (nj, ni),
            "u3d": (nk, nj, ni + 1),
            "v3d": (nk, nj + 1, ni),
            "w3d": face,
            "rho": cell,
            "rf": face,
            "pi0": cell,
            "th0": cell,
            "prs": cell,
            "pp3d": cell,
            "th3d": cell,
            "q3d": (6, nk, nj, ni),
        }
    )


def tiled_stretched_closure_output_shapes(
    ni: int, nj: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    ni = _positive_int("ni", ni)
    nj = _positive_int("nj", nj)
    nk = _positive_int("nk", nk, 3)
    cell, face = (nk, nj, ni), (nk + 1, nj, ni)
    return MappingProxyType(
        {name: cell if name == "grdscl" else face for name in CARRIER_ORDER}
    )


def tiled_stretched_closure_workspace_shapes(
    ni: int, nj: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Persistent per-buffer allocation inventory for one compute window."""

    ni = _positive_int("ni", ni)
    nj = _positive_int("nj", nj)
    nk = _positive_int("nk", nk, 3)
    static = {
        name: shape
        for name, shape in stretch_z1_terrain_shapes(ni, nj, nk).items()
        if name != "terrain_msl"
    }
    cell = (nk, nj, ni)
    moist = {name: cell for name in MOIST_SCRATCH_NAMES if name != "saturation_mask"}
    moist["saturation_mask"] = cell
    result = {
        **static,
        **moist,
        **terrain_deformation_scratch_shapes(ni, nj, nk),
        "ruh": (ni,),
        "rvh": (nj,),
        **tiled_stretched_closure_output_shapes(ni, nj, nk),
    }
    if tuple(result) != WORKSPACE_NAMES:
        raise RuntimeError("tile-local stretched closure workspace order changed")
    return MappingProxyType(result)


def tiled_stretched_closure_buffer_nbytes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, int]:
    """Return one input/output-workspace and double-buffer byte inventory."""

    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    ni = tile_nx + 2 * GM1_SOURCE_STENCIL_RADIUS
    nj = tile_ny + 2 * GM1_SOURCE_STENCIL_RADIUS

    def words(shapes: Mapping[str, tuple[int, ...]]) -> int:
        return sum(math.prod(shape) for shape in shapes.values())

    input_bytes = 4 * words(tiled_stretched_closure_input_shapes(ni, nj, nk))
    workspace_shapes = tiled_stretched_closure_workspace_shapes(ni, nj, nk)
    workspace_words = sum(
        math.prod(shape) * (1 if name == "saturation_mask" else 4)
        for name, shape in workspace_shapes.items()
    )
    # CudaStretchZ1Launch retains four nk+2 and four nk+3 float32 profiles.
    shared_vertical_profile_bytes = 4 * (4 * (nk + 2) + 4 * (nk + 3))
    shared_invalid_counter_bytes = 4
    one_buffer_bytes = input_bytes + workspace_words
    return MappingProxyType(
        {
            "compute_nx": ni,
            "compute_ny": nj,
            "input_bytes": input_bytes,
            "workspace_bytes": workspace_words,
            "one_buffer_bytes": one_buffer_bytes,
            "double_buffer_bytes": 2 * one_buffer_bytes,
            "shared_vertical_profile_bytes": shared_vertical_profile_bytes,
            "shared_invalid_counter_bytes": shared_invalid_counter_bytes,
            "single_buffer_resident_bytes": (
                one_buffer_bytes
                + shared_vertical_profile_bytes
                + shared_invalid_counter_bytes
            ),
            "double_buffer_resident_bytes": (
                2 * one_buffer_bytes
                + shared_vertical_profile_bytes
                + shared_invalid_counter_bytes
            ),
        }
    )


def tiled_stretched_closure_owned_slices(
    spec: Gm1TileSpec,
    *,
    allow_uniform_idealized_validation: bool = False,
    geometry: CartesianGridGeometry | None = None,
) -> tuple[slice, slice]:
    """Return the exact local mass rectangle permitted for publication."""

    _validate_spec(
        spec,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
        geometry=geometry,
    )
    return (
        slice(spec.j0 - spec.cj0, spec.j1 - spec.cj0),
        slice(spec.i0 - spec.ci0, spec.i1 - spec.ci0),
    )


def _is_uniform_idealized_full_domain_window(
    spec: Gm1TileSpec, geometry: CartesianGridGeometry | None = None
) -> bool:
    nx, ny = (300, 300) if geometry is None else (geometry.nx, geometry.ny)
    return (
        spec.index == 0
        and spec.ty == 0
        and spec.tx == 0
        and spec.nx == nx
        and spec.ny == ny
        and spec.tile_nx == nx
        and spec.tile_ny == ny
        and spec.halo == GM1_SOURCE_STENCIL_RADIUS
        and spec.i0 == 0
        and spec.i1 == nx
        and spec.j0 == 0
        and spec.j1 == ny
        and spec.ci0 == 0
        and spec.cj0 == 0
        and spec.cnx == nx
        and spec.cny == ny
    )


def _validate_spec(
    spec: Gm1TileSpec,
    *,
    allow_uniform_idealized_validation: bool = False,
    geometry: CartesianGridGeometry | None = None,
) -> None:
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if not isinstance(allow_uniform_idealized_validation, bool):
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if geometry is not None:
        if not isinstance(geometry, CartesianGridGeometry):
            raise TypeError("geometry must be CartesianGridGeometry")
        if not allow_uniform_idealized_validation:
            raise ValueError("explicit Cartesian geometry requires uniform idealized mode")
        geometry.require_stencil(radius_xy=GM1_SOURCE_STENCIL_RADIUS, minimum_nz=4)
        if (spec.nx, spec.ny) != (geometry.nx, geometry.ny):
            raise ValueError("closure tile global dimensions differ from explicit geometry")
        for name in ("index", "tx", "ty", "i0", "j0", "ci0", "cj0"):
            _positive_int(name, getattr(spec, name), 0)
        for name in ("tile_nx", "tile_ny", "cnx", "cny", "i1", "j1"):
            _positive_int(name, getattr(spec, name))
        if not (
            0 <= spec.ci0 <= spec.i0 < spec.i1 <= spec.ci0 + spec.cnx <= spec.nx
            and 0 <= spec.cj0 <= spec.j0 < spec.j1 <= spec.cj0 + spec.cny <= spec.ny
        ):
            raise ValueError("closure compute window or ownership leaves the physical domain")
    if spec.halo != GM1_SOURCE_STENCIL_RADIUS:
        raise ValueError("closure tiles require the exact three-cell source halo")
    full_domain_idealized = (
        allow_uniform_idealized_validation
        and _is_uniform_idealized_full_domain_window(spec, geometry)
    )
    if not full_domain_idealized and spec.cnx != spec.tile_nx + 2 * spec.halo:
        raise ValueError("closure tile compute-x extent changed")
    if not full_domain_idealized and spec.cny != spec.tile_ny + 2 * spec.halo:
        raise ValueError("closure tile compute-y extent changed")
    left, right = spec.i0 - spec.ci0, spec.ci0 + spec.cnx - spec.i1
    south, north = spec.j0 - spec.cj0, spec.cj0 + spec.cny - spec.j1
    if spec.i0 and left < spec.halo:
        raise ValueError("interior west ownership lacks the source halo")
    if spec.i1 < spec.nx and right < spec.halo:
        raise ValueError("interior east ownership lacks the source halo")
    if spec.j0 and south < spec.halo:
        raise ValueError("interior south ownership lacks the source halo")
    if spec.j1 < spec.ny and north < spec.halo:
        raise ValueError("interior north ownership lacks the source halo")
    if spec.i0 == 0 and spec.ci0 != 0:
        raise ValueError("west domain tile does not expose the true global edge")
    if spec.i1 == spec.nx and spec.ci0 + spec.cnx != spec.nx:
        raise ValueError("east domain tile does not expose the true global edge")
    if spec.j0 == 0 and spec.cj0 != 0:
        raise ValueError("south domain tile does not expose the true global edge")
    if spec.j1 == spec.ny and spec.cj0 + spec.cny != spec.ny:
        raise ValueError("north domain tile does not expose the true global edge")


def _validate_geometry(
    spec: Gm1TileSpec,
    *,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    ztop: float,
    allow_compact_validation: bool,
    allow_uniform_idealized_validation: bool = False,
    geometry: CartesianGridGeometry | None = None,
) -> tuple[int, int, int, float, float, float, float]:
    _validate_spec(
        spec,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
        geometry=geometry,
    )
    nk = _positive_int("nk", nk, 3)
    dx, dy = _spacing("dx", dx), _spacing("dy", dy)
    dz, ztop = _spacing("dz", dz), _spacing("ztop", ztop)
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    if not isinstance(allow_uniform_idealized_validation, bool):
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if geometry is not None:
        if (nk, dx, dy, dz, ztop) != (
            geometry.nz, geometry.dx_m, geometry.dy_m, geometry.dz_m, geometry.ztop_m
        ):
            raise ValueError("closure dimensions/spacing/top differ from explicit uniform geometry")
        # The kernels receive binary32 geometry and reciprocals.  A finite
        # Python float can overflow or round to zero at that actual ABI.
        for name, value in (
            ("dx", dx), ("dy", dy), ("dz", dz), ("ztop", ztop),
            ("1/dx", 1.0 / dx), ("1/dy", 1.0 / dy),
        ):
            try:
                stored = struct.unpack("<f", struct.pack("<f", value))[0]
            except (OverflowError, struct.error):
                stored = math.inf
            if not math.isfinite(stored) or stored <= 0.0:
                raise ValueError(f"uniform closure {name} is not finite and positive in float32")
        return spec.cnx, spec.cny, nk, dx, dy, dz, ztop
    production = {
        (300, 300, 75): (50.0, 50.0),
        (3000, 3000, 75): (5.0, 5.0),
    }
    key = (spec.nx, spec.ny, nk)
    idealized = (
        key == (300, 300, 75)
        and (dx, dy, dz, ztop) == (50.0, 50.0, 200.0, 15000.0)
    )
    if allow_uniform_idealized_validation and not idealized:
        raise ValueError(
            "uniform idealized closure requires exact 300x300x75, "
            "dx=dy=50 m, dz=200 m, ztop=15 km"
        )
    if not allow_compact_validation and not allow_uniform_idealized_validation and key not in production:
        refuse(
            code="GM1-TILESTREAM-STRETCHED-CLOSURE-GRID",
            gate="gm1-tilestream-stretched-fresh-closure",
            option="global_shape_zyx",
            value=[nk, spec.ny, spec.nx],
            reason="only the admitted 300/50 oracle and native 3000/5 grid exist",
            remedy="use an admitted grid or add a source-backed producer/oracle",
        )
    expected = production.get(key, (dx, dy))
    expected_dz = 200.0 if allow_uniform_idealized_validation else 100.0
    if (dx, dy) != expected or dz != expected_dz or ztop != 15000.0:
        raise ValueError("closure tile geometry differs from the admitted grid")
    return spec.cnx, spec.cny, nk, dx, dy, dz, ztop


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledStretchedClosure:
    static: CompiledStretchZ1Terrain
    calcnm: CompiledChildMoistNmKernel
    calcdef: CompiledChildStretchedCalcdef
    fresh_tke: CompiledGenericChildTurbsmagInit


@dataclass(frozen=True, slots=True)
class Gm1TiledStretchedClosureReduction:
    counter: Any
    ready_event: Any
    stream_pointer: int
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1TiledStretchedClosureLaunch:
    ready_event: Any
    arrays: Mapping[str, Any]
    workspace: Mapping[str, Any]
    compiled: CompiledGm1TiledStretchedClosure
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledStretchedClosureToken:
    receipt: Mapping[str, object]


def compile_cuda_gm1_tiled_stretched_closure() -> CompiledGm1TiledStretchedClosure:
    return CompiledGm1TiledStretchedClosure(
        compile_cuda_stretch_z1_terrain(),
        compile_cuda_child_moist_nm(),
        compile_cuda_gm1_child_stretched_calcdef(),
        compile_cuda_child_turbsmag_init_generic(),
    )


def allocate_cuda_gm1_tiled_stretched_closure_workspace(
    *, ni: int, nj: int, nk: int
) -> Mapping[str, Any]:
    cupy = importlib.import_module("cupy")
    result: dict[str, Any] = {}
    for name, shape in tiled_stretched_closure_workspace_shapes(ni, nj, nk).items():
        dtype = cupy.uint8 if name == "saturation_mask" else cupy.float32
        result[name] = cupy.empty(shape, dtype=dtype)
    return MappingProxyType(result)


def allocate_cuda_gm1_tiled_retained_tke_workspace(
    *, ni: int, nj: int, nk: int
) -> Mapping[str, Any]:
    """Allocate only the private tkekm scratch omitted from closure output."""

    ni = _positive_int("ni", ni)
    nj = _positive_int("nj", nj)
    nk = _positive_int("nk", nk, 3)
    cupy = importlib.import_module("cupy")
    face = (nk + 1, nj, ni)
    return MappingProxyType(
        {
            name: cupy.empty(face, dtype=cupy.float32)
            for name in RETAINED_TKE_SCRATCH_NAMES
        }
    )


def begin_cuda_gm1_tiled_stretched_closure_reduction(
    *, stream: Any | None = None
) -> Gm1TiledStretchedClosureReduction:
    """Zero the sole device invalid counter without a host synchronization."""

    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    with stream:
        counter = cupy.zeros((1,), dtype=cupy.uint32)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_REDUCTION_SCHEMA,
        "status": "ENQUEUED_ZERO_SHARED_INVALID_COUNTER",
        "stream_pointer": int(stream.ptr),
        "counter_pointer": int(counter.data.ptr),
        "counter_shape": [1],
        "counter_dtype": "<u4",
        "host_reads_before_finish": 0,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledStretchedClosureReduction(
        counter, event, int(stream.ptr), MappingProxyType(receipt)
    )


def _require_vertical(
    cupy: Any,
    vertical: Any,
    nk: int,
    *,
    allow_uniform_idealized_validation: bool,
    geometry: CartesianGridGeometry | None = None,
) -> CudaStretchZ1Launch:
    if not isinstance(vertical, CudaStretchZ1Launch):
        raise TypeError("vertical must be the exact CudaStretchZ1Launch")
    expected = (
        {
            "nk": geometry.nz,
            "dz": geometry.dz_m,
            "ztop": geometry.ztop_m,
            "str_bot": 0.0,
            "str_top": geometry.ztop_m,
            "dz_bot": geometry.dz_m,
            "dz_top": geometry.dz_m,
        }
        if geometry is not None
        else
        {
            "nk": 75,
            "dz": 200.0,
            "ztop": 15000.0,
            "str_bot": 0.0,
            "str_top": 15000.0,
            "dz_bot": 200.0,
            "dz_top": 200.0,
        }
        if allow_uniform_idealized_validation
        else {
            name: getattr(STRETCH_Z1_CHILD_15KM_PARAMETERS, name)
            for name in STRETCH_Z1_CHILD_15KM_PARAMETERS.__dataclass_fields__
        }
    )
    if vertical.receipt.get("parameters") != expected:
        raise ValueError(
            "vertical launch differs from explicit uniform geometry"
            if geometry is not None
            else "vertical launch is not the selected exact 15 km stretched/uniform grid"
        )
    if geometry is not None:
        if not allow_uniform_idealized_validation or nk != geometry.nz:
            raise ValueError("explicit vertical geometry requires matching uniform mode and levels")
        if vertical.receipt.get("selection") != {
            "stretch_z": 1, "terrain_flag": False,
            "stretch_x": 0, "stretch_y": 0, "axisymm": 0,
        }:
            raise ValueError("uniform closure vertical producer selection differs")
        expected_derived = derive_stretch_z1(StretchZ1Parameters(**expected))
        if vertical.grid.derived != expected_derived or vertical.receipt.get("derived") != {
            name: getattr(expected_derived, name)
            for name in expected_derived.__dataclass_fields__
        }:
            raise ValueError("uniform closure vertical producer derivation differs")
        if vertical.receipt.get("kernel", {}).get("source_sha256") != CUDA_STRETCH_Z1_SOURCE_SHA256:
            raise ValueError("uniform closure vertical producer source identity differs")
    for name, shape in (("sigma", (nk + 2,)), ("sigmaf", (nk + 3,))):
        value = getattr(vertical.grid, name)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"vertical {name} must be CUDA float32 {shape}")
    return vertical


def _require_arrays(
    cupy: Any,
    arrays: Mapping[str, Any],
    shapes: Mapping[str, tuple[int, ...]],
    *,
    workspace: bool,
) -> dict[str, Any]:
    if set(arrays) != set(shapes):
        label = "workspace" if workspace else "input"
        raise ValueError(f"closure tile {label} inventory changed")
    result: dict[str, Any] = {}
    for name, shape in shapes.items():
        value = arrays[name]
        expected_dtype = (
            cupy.uint8 if workspace and name == "saturation_mask" else cupy.float32
        )
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != expected_dtype
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"closure tile {name} must be C-order CUDA "
                f"{expected_dtype.name} {shape}"
            )
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("closure tile arrays must not alias")
    return result


def launch_cuda_gm1_tiled_stretched_closure(
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    ztop: float,
    vertical: CudaStretchZ1Launch,
    reduction: Gm1TiledStretchedClosureReduction,
    stream: Any | None = None,
    compiled: CompiledGm1TiledStretchedClosure | None = None,
    workspace: Mapping[str, Any] | None = None,
    retained_tkea: Any | None = None,
    retained_tkea_source_sha256: str | None = None,
    retained_grdscl: Any | None = None,
    retained_grdscl_source_sha256: str | None = None,
    retained_closure_operation_token_sha256: str | None = None,
    retained_tke_compiled: CompiledSgsTkeKernels | None = None,
    retained_tke_workspace: Mapping[str, Any] | None = None,
    sgs_startup_ramp_binding: CudaGm1TiledSgsStartupRampBinding | None = None,
    allow_compact_validation: bool = False,
    allow_uniform_idealized_validation: bool = False,
    geometry: CartesianGridGeometry | None = None,
) -> CudaGm1TiledStretchedClosureLaunch:
    """Enqueue one compute window; no device-to-host read or host sync occurs."""

    ni, nj, nk, dx, dy, dz, ztop = _validate_geometry(
        spec,
        nk=nk,
        dx=dx,
        dy=dy,
        dz=dz,
        ztop=ztop,
        allow_compact_validation=allow_compact_validation,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
        geometry=geometry,
    )
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    if not isinstance(reduction, Gm1TiledStretchedClosureReduction):
        raise TypeError("reduction must be the exact shared reduction object")
    if int(stream.ptr) != reduction.stream_pointer:
        raise ValueError("closure tiles and invalid reduction must use one stream")
    if reduction.receipt.get("operation_token_sha256") != _operation_token(
        reduction.receipt
    ):
        raise ValueError("closure invalid-reduction token differs from its receipt")
    if (
        not isinstance(reduction.counter, cupy.ndarray)
        or reduction.counter.dtype != cupy.uint32
        or reduction.counter.shape != (1,)
        or not reduction.counter.flags.c_contiguous
    ):
        raise ValueError("shared invalid counter must be C-order CUDA uint32[1]")
    vertical = _require_vertical(
        cupy,
        vertical,
        nk,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
        geometry=geometry,
    )
    live = _require_arrays(
        cupy,
        arrays,
        tiled_stretched_closure_input_shapes(ni, nj, nk),
        workspace=False,
    )
    workspace = (
        allocate_cuda_gm1_tiled_stretched_closure_workspace(ni=ni, nj=nj, nk=nk)
        if workspace is None
        else workspace
    )
    work = _require_arrays(
        cupy,
        workspace,
        tiled_stretched_closure_workspace_shapes(ni, nj, nk),
        workspace=True,
    )
    all_pointers = [int(value.data.ptr) for value in (*live.values(), *work.values())]
    if len(all_pointers) != len(set(all_pointers)):
        raise ValueError("closure tile input and workspace arrays must not alias")
    if int(reduction.counter.data.ptr) in all_pointers:
        raise ValueError("shared invalid counter aliases a closure tile array")
    compiled = compiled or compile_cuda_gm1_tiled_stretched_closure()
    retained_mode = retained_tkea is not None
    startup_ramp: Mapping[str, object] | None = None
    if sgs_startup_ramp_binding is not None:
        if retained_mode:
            raise ValueError(
                "active SGS startup ramp does not admit restored retained TKE"
            )
        sgs_startup_ramp_binding = (
            require_cuda_gm1_tilestream_sgs_startup_ramp_binding(
                sgs_startup_ramp_binding,
                spec=spec,
                nk=nk,
                dx=dx,
                dy=dy,
                stream=stream,
            )
        )
    if retained_mode:
        face = (nk + 1, nj, ni)
        if (
            not isinstance(retained_tkea, cupy.ndarray)
            or retained_tkea.dtype != cupy.float32
            or retained_tkea.shape != face
            or not retained_tkea.flags.c_contiguous
        ):
            raise ValueError(
                f"retained post-solve3 tkea must be C-order CUDA float32 {face}"
            )
        if (
            not isinstance(retained_tkea_source_sha256, str)
            or len(retained_tkea_source_sha256) != 64
            or any(
                char not in "0123456789abcdef" for char in retained_tkea_source_sha256
            )
        ):
            raise ValueError("retained post-solve3 tkea source SHA-256 is malformed")
        mass = (nk, nj, ni)
        if (
            not isinstance(retained_grdscl, cupy.ndarray)
            or retained_grdscl.dtype != cupy.float32
            or retained_grdscl.shape != mass
            or not retained_grdscl.flags.c_contiguous
        ):
            raise ValueError(
                f"retained closure grdscl must be C-order CUDA float32 {mass}"
            )
        for label, value in (
            ("retained closure grdscl source", retained_grdscl_source_sha256),
            (
                "retained closure operation token",
                retained_closure_operation_token_sha256,
            ),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{label} SHA-256 is malformed")
        if not isinstance(retained_tke_compiled, CompiledSgsTkeKernels):
            raise TypeError("retained mode requires the exact compiled SGS/TKE image")
        if (
            not isinstance(retained_tke_workspace, Mapping)
            or tuple(retained_tke_workspace) != RETAINED_TKE_SCRATCH_NAMES
        ):
            raise ValueError("retained mode scratch inventory changed")
        for name in RETAINED_TKE_SCRATCH_NAMES:
            value = retained_tke_workspace[name]
            if (
                not isinstance(value, cupy.ndarray)
                or value.dtype != cupy.float32
                or value.shape != face
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"retained mode {name} scratch changed")
        retained_pointers = [
            int(retained_tkea.data.ptr),
            int(retained_grdscl.data.ptr),
            *(int(value.data.ptr) for value in retained_tke_workspace.values()),
        ]
        if len(set((*all_pointers, *retained_pointers))) != len(
            (*all_pointers, *retained_pointers)
        ):
            raise ValueError("retained TKE source or scratch aliases closure arrays")
    elif any(
        value is not None
        for value in (
            retained_tkea_source_sha256,
            retained_grdscl,
            retained_grdscl_source_sha256,
            retained_closure_operation_token_sha256,
            retained_tke_compiled,
            retained_tke_workspace,
        )
    ):
        raise ValueError("retained TKE arguments require one exact tkea tile")

    static = work
    threads = (128, 1, 1)
    largest = max(
        math.prod(shape)
        for name, shape in stretch_z1_terrain_shapes(ni, nj, nk).items()
        if name != "terrain_msl"
    )
    static_grid = ((largest + 127) // 128, 1, 1)
    column_grid = (((ni * nj) + 127) // 128, 1, 1)
    mass_words = nk * (nj + 1) * (ni + 1)
    face_words = (nk - 1) * (nj + 1) * (ni + 1)
    reduce_words = (nk - 1) * nj * ni
    calcdef_grids = tuple(
        (
            (words + CUDA_TERRAIN_DEFORMATION_THREADS - 1)
            // CUDA_TERRAIN_DEFORMATION_THREADS,
        )
        for words in (mass_words, face_words, reduce_words)
    )
    calcdef_block = (CUDA_TERRAIN_DEFORMATION_THREADS,)
    tke_words = max(ni * nj * (nk - 1), ni * nj * (nk + 1))
    tke_blocks = min(
        65535,
        (tke_words + CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS - 1)
        // CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS,
    )
    tke_grid = (tke_blocks, 1, 1)
    tke_block = (CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS, 1, 1)
    u32, f32 = cupy.uint32, cupy.float32
    rdx, rdy = f32(1.0 / dx), f32(1.0 / dy)
    top = f32(ztop)

    with stream:
        stream.wait_event(reduction.ready_event)
        stream.wait_event(vertical.ready_event)
        surface, columns, cross, rds = compiled.static.kernels
        surface(
            static_grid,
            threads,
            (
                live["terrain_msl"],
                *(
                    static[name]
                    for name in (
                        "gz",
                        "rgz",
                        "gzu",
                        "rgzu",
                        "gzv",
                        "rgzv",
                        "dzdx",
                        "dzdy",
                    )
                ),
                u32(ni),
                u32(nj),
                top,
                rdx,
                rdy,
            ),
            stream=stream,
        )
        columns(
            column_grid,
            threads,
            (
                live["terrain_msl"],
                vertical.grid.sigma,
                vertical.grid.sigmaf,
                *(
                    static[name]
                    for name in ("zh", "zf", "mh", "rmh", "mf", "rmf", "c1", "c2")
                ),
                u32(ni),
                u32(nj),
                u32(nk),
                top,
                f32(dz),
            ),
            stream=stream,
        )
        cross(
            static_grid,
            threads,
            (
                live["terrain_msl"],
                vertical.grid.sigmaf,
                *(
                    static[name]
                    for name in (
                        "gz",
                        "rgz",
                        "gzu",
                        "rgzu",
                        "gzv",
                        "rgzv",
                        "gx",
                        "gxu",
                        "gy",
                        "gyv",
                    )
                ),
                u32(ni),
                u32(nj),
                u32(nk),
                top,
                rdx,
                rdy,
            ),
            stream=stream,
        )
        rds(
            (1, 1, 1),
            (1, 1, 1),
            (
                vertical.grid.sigma,
                vertical.grid.sigmaf,
                static["rds"],
                static["rdsf"],
                u32(nk),
            ),
            stream=stream,
        )

        moist_arrays = {
            "c1": static["c1"],
            "c2": static["c2"],
            "mf": static["mf"],
            **{
                name: live[name]
                for name in ("pi0", "th0", "prs", "pp3d", "th3d", "q3d")
            },
            "nm": work["nm"],
            **{name: work[name] for name in MOIST_SCRATCH_NAMES},
        }
        if tuple(moist_arrays) != CHILD_MOIST_ALL_NAMES:
            raise RuntimeError("tile-local calcnm argument order changed")
        launch_cuda_child_moist_nm(
            moist_arrays,
            ni=ni,
            nj=nj,
            nk=nk,
            dz=dz,
            stream=stream,
            compiled=compiled.calcnm,
        )

        work["defv"].fill(f32(0.0))
        work["defh"].fill(f32(0.0))
        compiled.calcdef.kernels[0](
            calcdef_grids[0],
            calcdef_block,
            (
                live["u3d"],
                live["v3d"],
                live["w3d"],
                live["rho"],
                static["gz"],
                static["rgzu"],
                static["rgzv"],
                static["gx"],
                static["gy"],
                *(work[name] for name in ("s11", "s22", "s33", "s12")),
                u32(ni),
                u32(nj),
                u32(nk),
                rdx,
                rdy,
                vertical.grid.sigma,
                vertical.grid.sigmaf,
                static["rds"],
                static["rdsf"],
                top,
            ),
            stream=stream,
        )
        compiled.calcdef.kernels[1](
            calcdef_grids[1],
            calcdef_block,
            (
                live["u3d"],
                live["v3d"],
                live["w3d"],
                live["rf"],
                static["gz"],
                static["rgz"],
                work["s13"],
                work["s23"],
                u32(ni),
                u32(nj),
                u32(nk),
                rdx,
                rdy,
                vertical.grid.sigma,
                vertical.grid.sigmaf,
                static["rds"],
                static["rdsf"],
                top,
            ),
            stream=stream,
        )
        compiled.calcdef.kernels[2](
            calcdef_grids[2],
            calcdef_block,
            (
                live["rf"],
                static["c1"],
                static["c2"],
                *(work[name] for name in ("s11", "s22", "s33", "s12", "s13", "s23")),
                work["defv"],
                work["defh"],
                u32(ni),
                u32(nj),
                u32(nk),
            ),
            stream=stream,
        )

        work["ruh"].fill(f32(1.0))
        work["rvh"].fill(f32(1.0))
        for name in TURBSMAG_OUTPUT_NAMES:
            work[name].fill(f32(0.0))
        tke_inputs = {
            "ruh": work["ruh"],
            "rvh": work["rvh"],
            "rmf": static["rmf"],
            "nm": work["nm"],
            "defv": work["defv"],
            "defh": work["defh"],
        }
        if tuple(tke_inputs) != TURBSMAG_INPUT_NAMES:
            raise RuntimeError("tile-local fresh-TKE argument order changed")
        compiled.fresh_tke.kernels[0](
            tke_grid,
            tke_block,
            tuple(tke_inputs[name] for name in TURBSMAG_INPUT_NAMES)
            + (reduction.counter, u32(ni), u32(nj), u32(nk), f32(dx), f32(dy), f32(dz)),
            stream=stream,
        )
        if retained_mode:
            for name in (
                "grdscl",
                "lenscl",
                "dissten",
                "kmh",
                "kmv",
                "khh",
                "khv",
                "tke3d",
                "tketen",
            ):
                work[name].fill(f32(0.0))
            for value in retained_tke_workspace.values():
                value.fill(f32(0.0))
            cupy.copyto(work["tkea"], retained_tkea)
            cupy.copyto(retained_tke_workspace["grdscl_full"][1:], retained_grdscl)
            retained_launch = launch_cuda_sgs_tke(
                {
                    **tke_inputs,
                    "tkea": work["tkea"],
                    "tk": retained_tke_workspace["tk"],
                    "grdscl": retained_tke_workspace["grdscl_full"],
                    "rgrdscl": retained_tke_workspace["rgrdscl"],
                    "lenscl": work["lenscl"],
                    "dissten": work["dissten"],
                    "cme": retained_tke_workspace["cme"],
                    "ce1": retained_tke_workspace["ce1"],
                    "ce2": retained_tke_workspace["ce2"],
                    "csm": retained_tke_workspace["csm"],
                    "kmh": work["kmh"],
                    "kmv": work["kmv"],
                    "khh": work["khh"],
                    "khv": work["khv"],
                    "tketen": work["tketen"],
                },
                ni=ni,
                nj=nj,
                nk=nk,
                dx=dx,
                dy=dy,
                dz=dz,
                stream=stream,
                compiled=retained_tke_compiled,
            )
            cupy.copyto(work["grdscl"], retained_tke_workspace["grdscl_full"][1:])
            cupy.copyto(work["tke3d"], work["tkea"])
        else:
            compiled.fresh_tke.kernels[1](
                tke_grid,
                tke_block,
                tuple(tke_inputs[name] for name in TURBSMAG_INPUT_NAMES)
                + tuple(
                    work[name]
                    for name in (
                        "grdscl",
                        "lenscl",
                        "dissten",
                        "kmh",
                        "kmv",
                        "khh",
                        "khv",
                        "tkea",
                    )
                )
                + (u32(ni), u32(nj), u32(nk), f32(dx), f32(dy), f32(dz)),
                stream=stream,
            )
            startup_ramp = (
                enqueue_cuda_gm1_tilestream_sgs_startup_ramp(
                    sgs_startup_ramp_binding,
                    work,
                    spec=spec,
                    nk=nk,
                    dx=dx,
                    dy=dy,
                    stream=stream,
                )
                if sgs_startup_ramp_binding is not None
                else None
            )
            compiled.fresh_tke.kernels[2](
                tke_grid,
                tke_block,
                (
                    work["tkea"],
                    work["tke3d"],
                    work["tketen"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                ),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)

    owned_y, owned_x = tiled_stretched_closure_owned_slices(
        spec,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
        geometry=geometry,
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_ABI,
        "status": CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_STATUS,
        "tile": spec.record(),
        "compute_shape_zyx": [nk, nj, ni],
        "owned_local_yx": [owned_y.start, owned_y.stop, owned_x.start, owned_x.stop],
        "spacing_m": {"dx": dx, "dy": dy, "reference_dz": dz, "ztop": ztop},
        "stream_pointer": int(stream.ptr),
        "reduction_operation_token_sha256": reduction.receipt["operation_token_sha256"],
        "source_order": list(
            RETAINED_TKE_SOURCE_ORDER
            if retained_mode
            else SOURCE_ORDER_WITH_ACTIVE_SGS_STARTUP_RAMP
            if startup_ramp is not None
            else SOURCE_ORDER
        ),
        "operator_source_sha256": {
            **dict(OPERATOR_SOURCE_SHA256),
            **({"retained_tke": CUDA_SGS_TKE_SOURCE_SHA256} if retained_mode else {}),
            **(
                {
                    "sgs_startup_ramp_live_binding": (
                        CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_MODULE_SHA256
                    )
                }
                if startup_ramp is not None
                else {}
            ),
        },
        "closure_lifecycle": (
            "restored-post-solve3-retained-tkea"
            if retained_mode
            else "fresh-initial-tkea"
        ),
        "vertical_disposition": (
            "source-audited-flat-uniform-cartesian"
            if geometry is not None
            else "source-audited-flat-idealized-uniform-200m"
            if allow_uniform_idealized_validation
            else "sealed-stretched-child"
        ),
        **({"grid_geometry": geometry.record()} if geometry is not None else {}),
        **(
            {
                "retained_tkea_source_sha256": retained_tkea_source_sha256,
                "retained_grdscl_source_sha256": retained_grdscl_source_sha256,
                "retained_closure_operation_token_sha256": (
                    retained_closure_operation_token_sha256
                ),
                "retained_tke_receipt": retained_launch.receipt,
            }
            if retained_mode
            else {}
        ),
        **(
            {
                "sgs_startup_ramp": dict(startup_ramp),
                "sgs_startup_ramp_active": True,
                "sgs_startup_ramp_retained_tke_admitted": False,
            }
            if startup_ramp is not None
            else {}
        ),
        "same_stream_no_host_sync": True,
        "invalid_counter_host_read": False,
        "interior_tile_edges_published_as_boundaries": False,
        "claim_limit": (
            RETAINED_TKE_CLAIM_LIMIT
            if retained_mode
            else CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_CLAIM_LIMIT
        ),
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    outputs = MappingProxyType({name: work[name] for name in CARRIER_ORDER})
    return CudaGm1TiledStretchedClosureLaunch(
        event,
        outputs,
        MappingProxyType(work),
        compiled,
        MappingProxyType(receipt),
    )


def finish_cuda_gm1_tiled_stretched_closure(
    reduction: Gm1TiledStretchedClosureReduction,
    launches: Sequence[CudaGm1TiledStretchedClosureLaunch],
    *,
    plan: Gm1TilePlan,
    stream: Any | None = None,
    geometry: CartesianGridGeometry | None = None,
) -> Gm1TiledStretchedClosureToken:
    """Perform the sole host read and admit the complete tiled operator pass."""

    if not isinstance(plan, Gm1TilePlan):
        raise TypeError("plan must be Gm1TilePlan")
    if not isinstance(reduction, Gm1TiledStretchedClosureReduction):
        raise TypeError("reduction must be Gm1TiledStretchedClosureReduction")
    if len(launches) != plan.ntiles:
        raise ValueError("closure tile launch count differs from the sealed plan")
    if geometry is not None:
        if not isinstance(geometry, CartesianGridGeometry):
            raise TypeError("geometry must be CartesianGridGeometry")
        if (plan.nx, plan.ny) != (geometry.nx, geometry.ny):
            raise ValueError("closure plan dimensions differ from explicit geometry")
    tokens: list[str] = []
    indices: list[int] = []
    lifecycles: list[str] = []
    vertical_dispositions: list[str] = []
    retained_tkea_sources: list[str] = []
    retained_grdscl_sources: list[str] = []
    retained_closure_tokens: list[str] = []
    startup_ramp_modes: list[bool] = []
    startup_ramp_binding_tokens: list[str] = []
    startup_ramp_tile_tokens: list[str] = []
    startup_ramp_cpu_tokens: list[str] = []
    startup_ramp_selections: list[Mapping[str, object]] = []
    for launch in launches:
        if not isinstance(launch, CudaGm1TiledStretchedClosureLaunch):
            raise TypeError("closure launches contain a foreign result")
        receipt = launch.receipt
        if receipt.get("operation_token_sha256") != _operation_token(receipt):
            raise ValueError("closure tile operation token differs from its receipt")
        if receipt.get("reduction_operation_token_sha256") != reduction.receipt.get(
            "operation_token_sha256"
        ):
            raise ValueError("closure tiles do not share one invalid reduction")
        if receipt.get("stream_pointer") != reduction.stream_pointer:
            raise ValueError("closure tiles did not use the reduction stream")
        indices.append(int(receipt["tile"]["index"]))
        tokens.append(str(receipt["operation_token_sha256"]))
        lifecycle = receipt.get("closure_lifecycle")
        if lifecycle not in {
            "fresh-initial-tkea",
            "restored-post-solve3-retained-tkea",
        }:
            raise ValueError("closure tile lifecycle is not admitted")
        lifecycles.append(str(lifecycle))
        vertical_disposition = receipt.get("vertical_disposition")
        if vertical_disposition not in {
            "sealed-stretched-child",
            "source-audited-flat-idealized-uniform-200m",
            "source-audited-flat-uniform-cartesian",
        }:
            raise ValueError("closure tile vertical disposition is not admitted")
        vertical_dispositions.append(str(vertical_disposition))
        if geometry is not None:
            if (
                vertical_disposition != "source-audited-flat-uniform-cartesian"
                or receipt.get("grid_geometry") != geometry.record()
            ):
                raise ValueError("closure tiles do not bind the explicit uniform geometry")
            index = receipt["tile"]["index"]
            if not isinstance(index, int) or not 0 <= index < plan.ntiles:
                raise ValueError("closure tile index is outside the explicit geometry plan")
            spec = plan.specs[index]
            _validate_spec(spec, allow_uniform_idealized_validation=True, geometry=geometry)
            if (
                receipt.get("tile") != spec.record()
                or receipt.get("compute_shape_zyx") != [geometry.nz, spec.cny, spec.cnx]
                or receipt.get("spacing_m") != {
                    "dx": geometry.dx_m, "dy": geometry.dy_m,
                    "reference_dz": geometry.dz_m, "ztop": geometry.ztop_m,
                }
            ):
                raise ValueError("closure tile shape/spacing differs from explicit geometry")
        elif (
            vertical_disposition == "source-audited-flat-uniform-cartesian"
            or receipt.get("grid_geometry") is not None
        ):
            raise ValueError("explicit closure geometry is required at finish")
        startup_ramp_active = receipt.get("sgs_startup_ramp_active") is True
        startup_ramp_modes.append(startup_ramp_active)
        startup_ramp = receipt.get("sgs_startup_ramp")
        if startup_ramp_active:
            if (
                not isinstance(startup_ramp, Mapping)
                or startup_ramp.get("schema")
                != CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_SCHEMA
                or startup_ramp.get("status")
                != CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_STATUS
                or startup_ramp.get("tile") != receipt.get("tile")
                or startup_ramp.get("stream_pointer") != reduction.stream_pointer
                or startup_ramp.get("retained_tke_admitted") is not False
                or startup_ramp.get("operation_token_sha256")
                != _operation_token(startup_ramp)
            ):
                raise ValueError("active startup-ramp tile lineage changed")
            for key, output in (
                (
                    "binding_operation_token_sha256",
                    startup_ramp_binding_tokens,
                ),
                ("operation_token_sha256", startup_ramp_tile_tokens),
                (
                    "cpu_operator_operation_token_sha256",
                    startup_ramp_cpu_tokens,
                ),
            ):
                value = startup_ramp.get(key)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                ):
                    raise ValueError(f"startup-ramp tile {key} is malformed")
                output.append(value)
            selection = startup_ramp.get("selection")
            if not isinstance(selection, Mapping):
                raise ValueError("startup-ramp tile selection is malformed")
            startup_ramp_selections.append(selection)
        elif startup_ramp is not None:
            raise ValueError("disabled startup-ramp tile carries active lineage")
        if lifecycle == "restored-post-solve3-retained-tkea":
            source_sha256 = receipt.get("retained_tkea_source_sha256")
            if (
                not isinstance(source_sha256, str)
                or len(source_sha256) != 64
                or any(char not in "0123456789abcdef" for char in source_sha256)
            ):
                raise ValueError("retained closure tile source SHA-256 is malformed")
            retained_tkea_sources.append(source_sha256)
            for key, output in (
                ("retained_grdscl_source_sha256", retained_grdscl_sources),
                (
                    "retained_closure_operation_token_sha256",
                    retained_closure_tokens,
                ),
            ):
                value = receipt.get(key)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                ):
                    raise ValueError(f"retained closure tile {key} is malformed")
                output.append(value)
    if sorted(indices) != list(range(plan.ntiles)) or len(set(indices)) != plan.ntiles:
        raise ValueError("closure tiles do not cover every plan index exactly once")
    if len(set(lifecycles)) != 1:
        raise ValueError("closure tiles mix fresh and restored TKE lifecycles")
    if len(set(vertical_dispositions)) != 1:
        raise ValueError("closure tiles mix stretched and uniform vertical grids")
    lifecycle = lifecycles[0]
    retained_mode = lifecycle == "restored-post-solve3-retained-tkea"
    startup_ramp_active = set(startup_ramp_modes) == {True}
    if set(startup_ramp_modes) not in ({False}, {True}):
        raise ValueError("closure tiles mix active and disabled startup ramp")
    if startup_ramp_active:
        if retained_mode:
            raise ValueError("restored closure cannot be relabelled as startup-ramped")
        if (
            len(startup_ramp_binding_tokens) != plan.ntiles
            or len(set(startup_ramp_binding_tokens)) != 1
            or len(startup_ramp_tile_tokens) != plan.ntiles
            or len(startup_ramp_cpu_tokens) != plan.ntiles
            or len(set(startup_ramp_cpu_tokens)) != 1
            or len(startup_ramp_selections) != plan.ntiles
            or len({_canonical(value) for value in startup_ramp_selections}) != 1
        ):
            raise ValueError("closure tiles do not bind one exact startup ramp")
    if retained_mode:
        if (
            len(retained_tkea_sources) != plan.ntiles
            or len(set(retained_tkea_sources)) != 1
            or len(retained_grdscl_sources) != plan.ntiles
            or len(set(retained_grdscl_sources)) != 1
            or len(retained_closure_tokens) != plan.ntiles
            or len(set(retained_closure_tokens)) != 1
        ):
            raise ValueError("closure tiles do not bind one restored closure source")
        retained_tkea_source_sha256 = retained_tkea_sources[0]
        retained_grdscl_source_sha256 = retained_grdscl_sources[0]
        retained_closure_operation_token_sha256 = retained_closure_tokens[0]

    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    if int(stream.ptr) != reduction.stream_pointer:
        raise ValueError("closure finish must use the reduction stream")
    with stream:
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    event.synchronize()
    invalid_count = int(reduction.counter.get()[0])
    if invalid_count:
        refuse(
            code="GM1-TILESTREAM-STRETCHED-CLOSURE-DOMAIN",
            gate="gm1-tilestream-stretched-fresh-closure",
            option="metrics_or_closure_inputs",
            value={"invalid_interior_cell_count": invalid_count},
            reason="one or more tile metric/TKE operands left the admitted domain",
            remedy="repair the source-backed tile inputs or static metric producer",
        )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
        "status": "ADMITTED_TILE_LOCAL_STRETCHED_FRESH_CLOSURE_OPERATOR_ONLY",
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_indices": sorted(indices),
        "tile_operation_tokens_sha256": hashlib.sha256(
            _canonical(sorted(tokens))
        ).hexdigest(),
        "reduction_operation_token_sha256": reduction.receipt["operation_token_sha256"],
        "invalid_interior_cell_count": 0,
        "device_invalid_counter_reads": 1,
        "stream_pointer": reduction.stream_pointer,
        "vertical_disposition": vertical_dispositions[0],
        **({"grid_geometry": geometry.record()} if geometry is not None else {}),
        "source_order": list(
            RETAINED_TKE_SOURCE_ORDER
            if retained_mode
            else SOURCE_ORDER_WITH_ACTIVE_SGS_STARTUP_RAMP
            if startup_ramp_active
            else SOURCE_ORDER
        ),
        "operator_source_sha256": {
            **dict(OPERATOR_SOURCE_SHA256),
            **({"retained_tke": CUDA_SGS_TKE_SOURCE_SHA256} if retained_mode else {}),
            **(
                {
                    "sgs_startup_ramp_live_binding": (
                        CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_MODULE_SHA256
                    )
                }
                if startup_ramp_active
                else {}
            ),
        },
        "closure_lifecycle": lifecycle,
        **(
            {
                "retained_tkea_source_sha256": retained_tkea_source_sha256,
                "retained_grdscl_source_sha256": retained_grdscl_source_sha256,
                "retained_closure_operation_token_sha256": (
                    retained_closure_operation_token_sha256
                ),
            }
            if retained_mode
            else {}
        ),
        **(
            {
                "sgs_startup_ramp_active": True,
                "sgs_startup_ramp_binding_operation_token_sha256": (
                    startup_ramp_binding_tokens[0]
                ),
                "sgs_startup_ramp_tile_operation_tokens_sha256": hashlib.sha256(
                    _canonical(startup_ramp_tile_tokens)
                ).hexdigest(),
                "sgs_startup_ramp_cpu_operator_operation_token_sha256": (
                    startup_ramp_cpu_tokens[0]
                ),
                "sgs_startup_ramp_selection": dict(startup_ramp_selections[0]),
                "sgs_startup_ramp_active_cuda_raw_bit_oracle": False,
                "sgs_startup_ramp_coupled_trajectory_admitted": False,
            }
            if startup_ramp_active
            else {}
        ),
        "complete_child_available": False,
        "rk_stage_available": False,
        "claim_limit": (
            RETAINED_TKE_CLAIM_LIMIT
            if retained_mode
            else CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_CLAIM_LIMIT
        ),
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledStretchedClosureToken(MappingProxyType(receipt))


__all__ = (
    "CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_ABI",
    "CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_CLAIM_LIMIT",
    "CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_REDUCTION_SCHEMA",
    "CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_SCHEMA",
    "CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_STATUS",
    "CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA",
    "CudaGm1TiledStretchedClosureLaunch",
    "CompiledGm1TiledStretchedClosure",
    "Gm1TiledStretchedClosureReduction",
    "Gm1TiledStretchedClosureToken",
    "INPUT_NAMES",
    "OPERATOR_SOURCE_SHA256",
    "RETAINED_TKE_CLAIM_LIMIT",
    "RETAINED_TKE_SCRATCH_NAMES",
    "RETAINED_TKE_SOURCE_ORDER",
    "SOURCE_ORDER",
    "SOURCE_ORDER_WITH_ACTIVE_SGS_STARTUP_RAMP",
    "WORKSPACE_NAMES",
    "allocate_cuda_gm1_tiled_stretched_closure_workspace",
    "allocate_cuda_gm1_tiled_retained_tke_workspace",
    "begin_cuda_gm1_tiled_stretched_closure_reduction",
    "compile_cuda_gm1_tiled_stretched_closure",
    "finish_cuda_gm1_tiled_stretched_closure",
    "launch_cuda_gm1_tiled_stretched_closure",
    "tiled_stretched_closure_buffer_nbytes",
    "tiled_stretched_closure_input_shapes",
    "tiled_stretched_closure_output_shapes",
    "tiled_stretched_closure_owned_slices",
    "tiled_stretched_closure_workspace_shapes",
)
