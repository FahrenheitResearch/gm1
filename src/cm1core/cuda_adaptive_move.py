"""Exact bounded CM1 r21.1 adaptive moving-domain controller.

The admitted slice is the source-ordered ``do_adapt_move``/``imove=1``
controller for a flat, serial, four-periodic Cartesian domain.  The pressure
center is diagnosed from the lowest scalar level with the official 100-pass
nine-point smoother and type-2 pressure centroid.

Neither ``getpsmth`` nor ``getpcent`` reads ``dx``, ``dy`` or the vertical
coordinate (``azimavg.F:4012-4330``): the horizontal grid enters only through
the values of ``xh``/``yh``/``xfref``/``yfref``, and ``zref = 0`` selects
``kval = 1`` whatever height that level has.  The admission is therefore any
uniform spacing whose ``param.F:6649-6687,7393`` reference faces and mass
points binary32 holds exactly, and any vertical coordinate whose lowest scalar
level can be named; the receipt states that level's height rather than
assuming one.

This module does not imply moving terrain, open/LBC motion, nesting, MPI
decomposition, a complete time step, or a forecast.  Callers must supply the
completed-large-step cadence flags and the exact wind carrier views to shift.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .cuda_sound3_periodic import _canonical_arch, _image_options
from .cuda_stretch_z1 import (
    StretchZ1MetricProof,
    build_stretch_z1_grid,
    build_stretch_z1_metric_proof,
)
from .errors import refuse


CUDA_ADAPTIVE_MOVE_SCHEMA = "gpuwm-cm1.cuda-adaptive-move/v1"
CUDA_ADAPTIVE_MOVE_ABI = "gpuwm-cm1.cuda-adaptive-move-kernels/v1"
CUDA_ADAPTIVE_MOVE_STATE_SCHEMA = "gpuwm-cm1.cuda-adaptive-move-state/v1"
CM1_ADAPTIVE_MOVE_CLAIM_LIMIT = (
    "CM1 r21.1 flat serial four-periodic do_adapt_move/imove=1 completed-"
    "large-step controller: lowest-scalar-level pressure diagnosis, 100-pass "
    "nine-point smoothing, type-2 pressure centroid, one cadence update, "
    "moving-frame wind rebasing, and persistent controller state, at any "
    "uniform horizontal spacing whose reference faces and mass points are "
    "exact binary32 words and on any stretch_z=0/1/3/4 column whose lowest "
    "scalar level is named in the receipt. No terrain, open/LBC boundaries, "
    "MPI, nesting, moving-grid remap, coupled trajectory, forecast, storm, "
    "tornado, or performance claim"
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/azimavg.F": "169a9f82d0929ab88c4add87441b6f9ace70f5c83fd82ecbb9f43d700af75c2a",
    "src/bc.F": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
    "src/cm1.F": "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
    "src/input.F": "e8b3bb25e0b624d79da7d361a2027431b55f98434abfe19b4385d7c7e1692663",
    "src/param.F": "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
    "src/base.F": "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
    "src/restart_write.F": "d9c21d7071a4548cb9d3ffee1658550ce0e2a4ee7db9c67758ac8c61c07cddba",
    "src/restart_read.F": "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a",
}
SOURCE_ANCHORS = {
    "selection_initialization": "src/param.F:701-717",
    "selection_refusals": "src/param.F:1874-1967; terrain+imove refusal 1890-1903",
    "persistent_initialization": "src/cm1.F:348-349,388-390",
    "initial_frame_subtraction": "src/base.F:2661-2674",
    "completed_step_domain_location": "src/cm1.F:3507-3513",
    "diagnosis_cadence": "src/cm1.F:3641-3651",
    "controller_cadence": "src/cm1.F:3669-3679",
    "periodic_scalar_boundary": "src/bc.F:12-172",
    "pressure_smoothing": "src/azimavg.F:4012-4165",
    "pressure_centroid": "src/azimavg.F:4173-4330",
    "wind_frame_rebase": "src/azimavg.F:4338-4368",
    "adaptive_controller": "src/azimavg.F:4376-4528",
    "restart_payload_binary": (
        "src/restart_write.F:377-401; src/restart_read.F:397-421"
    ),
    "restart_payload_netcdf": (
        "src/restart_write.F:476-549; src/restart_read.F:662-735"
    ),
}

F32_STATE_FIELDS = (
    "umove",
    "umovem1",
    "umovem2",
    "vmove",
    "vmovem1",
    "vmovem2",
    "xcenter",
    "ycenter",
)
F64_STATE_FIELDS = ("domainlocx", "domainlocy", "adaptmovetim")
I32_STATE_FIELDS = ("icenter", "jcenter", "nwritemv", "mvrec")
WIND_FIELDS = ("u0", "ua", "u3d", "v0", "va", "v3d", "ug", "vg")
ADAPTIVE_MOVE_RESTART_SOURCE_ORDER = (
    "mvrec",
    "nwritemv",
    "icenter",
    "jcenter",
    "xcenter",
    "ycenter",
    "umove",
    "umovem1",
    "umovem2",
    "vmove",
    "vmovem1",
    "vmovem2",
    "domainlocx",
    "domainlocy",
    "adaptmovetim",
)


@dataclass(frozen=True, slots=True)
class AdaptiveMoveSelection:
    do_adapt_move: bool = True
    imove: int = 1
    axisymm: int = 0
    terrain_flag: bool = False
    mpi: bool = False
    initsfc: int = 1
    wbc: int = 1
    ebc: int = 1
    sbc: int = 1
    nbc: int = 1
    stretch_x: int = 0
    stretch_y: int = 0
    stretch_z: int = 1
    output_format: int = 2


DEFAULT_ADAPTIVE_MOVE_SELECTION = AdaptiveMoveSelection()

#: ``input.F:283`` vertical branches whose lowest scalar level this port can
#: name: the uniform column (``param.F:7789-7811``), the analytic stretch
#: (``param.F:7819-7953``) and the two explicit level lists
#: (``param.F:8005-8137``).  ``getpsmth`` never reads the coordinate --
#: ``zref = 0`` selects ``kval = 1`` (``azimavg.F:4046-4058``) -- so the branch
#: decides only which height the receipt says was diagnosed.
ADAPTIVE_MOVE_STRETCH_Z = (0, 1, 3, 4)


def _refuse_selection(name: str, value: object, wanted: object) -> None:
    refuse(
        code="ARWEN-ADAPTIVE-MOVE-SELECTION",
        gate="adaptive-moving-domain",
        option=name,
        value=value,
        reason=(
            "the bounded controller requires the exact flat serial "
            "four-periodic selection on a vertical coordinate whose lowest "
            "scalar level can be named"
        ),
        remedy=f"select {name}={wanted!r} or add a separately evidenced branch",
    )


def require_adaptive_move_selection(
    selection: AdaptiveMoveSelection = DEFAULT_ADAPTIVE_MOVE_SELECTION,
) -> None:
    if not isinstance(selection, AdaptiveMoveSelection):
        raise TypeError("selection must be an AdaptiveMoveSelection")
    expected = DEFAULT_ADAPTIVE_MOVE_SELECTION
    for name in selection.__dataclass_fields__:
        value = getattr(selection, name)
        wanted = getattr(expected, name)
        if name == "stretch_z":
            if type(value) is not int or value not in ADAPTIVE_MOVE_STRETCH_Z:
                _refuse_selection(name, value, ADAPTIVE_MOVE_STRETCH_Z)
            continue
        if type(value) is not type(wanted) or value != wanted:  # noqa: E721
            refuse(
                code="ARWEN-ADAPTIVE-MOVE-SELECTION",
                gate="adaptive-moving-domain",
                option=name,
                value=value,
                reason=(
                    "the bounded controller requires the exact flat serial "
                    "four-periodic stretch_z=1 selection"
                ),
                remedy=f"select {name}={wanted!r} or add a separately evidenced branch",
            )


@dataclass(slots=True)
class AdaptiveMoveState:
    umove: float
    umovem1: float
    umovem2: float
    vmove: float
    vmovem1: float
    vmovem2: float
    xcenter: float
    ycenter: float
    domainlocx: float
    domainlocy: float
    adaptmovetim: float
    icenter: int
    jcenter: int
    nwritemv: int
    mvrec: int


@dataclass(frozen=True, slots=True)
class AdaptivePressureCenter:
    psmth: Any
    minimum_icenter: int
    minimum_jcenter: int
    minimum_xcenter: float
    minimum_ycenter: float
    icenter: int
    jcenter: int
    xcenter: float
    ycenter: float
    centroid_weight: float


@dataclass(frozen=True, slots=True)
class AdaptiveMoveTransition:
    due: bool
    diagnosed: bool
    delta_u: float
    delta_v: float
    pressure_center: AdaptivePressureCenter | None


def initialize_adaptive_move_state(
    *, ni: int, nj: int, centerx: float, centery: float
) -> AdaptiveMoveState:
    if isinstance(ni, bool) or not isinstance(ni, int) or ni < 1:
        raise ValueError("ni must be a positive integer")
    if isinstance(nj, bool) or not isinstance(nj, int) or nj < 1:
        raise ValueError("nj must be a positive integer")
    np = importlib.import_module("numpy")
    centerx32, centery32 = np.float32(centerx), np.float32(centery)
    if not bool(np.isfinite(centerx32)) or not bool(np.isfinite(centery32)):
        raise ValueError("domain center coordinates must be finite float32 values")
    zero = float(np.float32(0.0))
    return AdaptiveMoveState(
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        float(centerx32),
        float(centery32),
        0.0,
        0.0,
        0.0,
        ni // 2 + 1,
        nj // 2 + 1,
        1,
        1,
    )


def _validate_state(state: AdaptiveMoveState) -> None:
    if not isinstance(state, AdaptiveMoveState):
        raise TypeError("state must be an AdaptiveMoveState")
    np = importlib.import_module("numpy")
    for name in F32_STATE_FIELDS:
        value = getattr(state, name)
        if not math.isfinite(float(value)) or float(np.float32(value)) != float(value):
            raise ValueError(
                f"state.{name} must be finite and exactly representable as float32"
            )
    for name in F64_STATE_FIELDS:
        if not math.isfinite(float(getattr(state, name))):
            raise ValueError(f"state.{name} must be finite")
    for name in I32_STATE_FIELDS:
        value = getattr(state, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not -(2**31) <= value < 2**31
        ):
            raise ValueError(f"state.{name} must be an int32 scalar")


def pack_adaptive_move_state(state: AdaptiveMoveState) -> tuple[Any, Any, Any]:
    """Return the sealed device/checkpoint carrier order."""

    _validate_state(state)
    np = importlib.import_module("numpy")
    f32 = np.ascontiguousarray(
        [getattr(state, name) for name in F32_STATE_FIELDS], dtype=np.float32
    )
    f64 = np.ascontiguousarray(
        [getattr(state, name) for name in F64_STATE_FIELDS], dtype=np.float64
    )
    i32 = np.ascontiguousarray(
        [getattr(state, name) for name in I32_STATE_FIELDS], dtype=np.int32
    )
    return f32, f64, i32


def adaptive_move_restart_contract() -> dict[str, Any]:
    """Describe the exact persistent state and its dtype-grouped carrier ABI.

    CM1's binary and NetCDF restart streams serialize the moving-domain
    fields among other model counters in ``ADAPTIVE_MOVE_RESTART_SOURCE_ORDER``.
    The CUDA state ABI deliberately groups those same values by dtype; it is
    not represented as a byte-identical copy of a Fortran restart record.
    """

    return {
        "schema": CUDA_ADAPTIVE_MOVE_STATE_SCHEMA,
        "source_serialization_order": list(ADAPTIVE_MOVE_RESTART_SOURCE_ORDER),
        "device_checkpoint_carriers": [
            {
                "name": "f32",
                "dtype": "little-endian-f32",
                "shape": [8],
                "fields": list(F32_STATE_FIELDS),
            },
            {
                "name": "f64",
                "dtype": "little-endian-f64",
                "shape": [3],
                "fields": list(F64_STATE_FIELDS),
            },
            {
                "name": "i32",
                "dtype": "little-endian-i32",
                "shape": [4],
                "fields": list(I32_STATE_FIELDS),
            },
        ],
        "source_anchors": {
            "binary": SOURCE_ANCHORS["restart_payload_binary"],
            "netcdf": SOURCE_ANCHORS["restart_payload_netcdf"],
            "initialization": SOURCE_ANCHORS["persistent_initialization"],
            "completed_step_cadence": (
                f"{SOURCE_ANCHORS['completed_step_domain_location']}; "
                f"{SOURCE_ANCHORS['controller_cadence']}"
            ),
        },
        "source_sha256": {
            name: AUTHORITY_SOURCE_SHA256[name]
            for name in (
                "src/cm1.F",
                "src/azimavg.F",
                "src/restart_write.F",
                "src/restart_read.F",
            )
        },
        "cadence": {
            "domain_location_uses_pre_update_speed": True,
            "controller_due_comparison": "mtime_after >= adaptmovetim",
            "controller_requires_nstep_for_speed_update": 1,
            "nwritemv_increments_when_due": 1,
            "adaptmovetim_advances_once_when_due": True,
            "catch_up_loop": False,
            "output_format_2_preserves_mvrec": True,
        },
    }


def unpack_adaptive_move_state(f32: Any, f64: Any, i32: Any) -> AdaptiveMoveState:
    np = importlib.import_module("numpy")
    for name, value, dtype, shape in (
        ("f32", f32, np.float32, (8,)),
        ("f64", f64, np.float64, (3,)),
        ("i32", i32, np.int32, (4,)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(dtype)
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} state carrier must be C-order {np.dtype(dtype)} shape {shape}"
            )
    values = {name: float(f32[index]) for index, name in enumerate(F32_STATE_FIELDS)}
    values.update(
        {name: float(f64[index]) for index, name in enumerate(F64_STATE_FIELDS)}
    )
    values.update(
        {name: int(i32[index]) for index, name in enumerate(I32_STATE_FIELDS)}
    )
    state = AdaptiveMoveState(**values)
    _validate_state(state)
    return state


def _lowest_scalar_level_m(np: Any, sigma: Any, sigmaf: Any, *, label: str) -> float:
    """``azimavg.F:4046-4058`` with ``zref = 0``: ``kval = 1``, named in metres.

    ``sigma``/``sigmaf`` carry CM1 indices, so ``sigma[1]`` is scalar level 1
    and ``sigmaf[1]`` the ground.  With ``terrain_flag`` false that level is
    ``param.F:8189-8191``'s midpoint of the first layer, and it is the height
    the receipt reports; nothing in the diagnosis arithmetic reads it.
    """

    sigma = np.asarray(sigma)
    sigmaf = np.asarray(sigmaf)
    if sigma.ndim != 1 or sigmaf.ndim != 1 or sigma.size < 3 or sigmaf.size < 3:
        raise ValueError(f"{label} carries no first model layer to diagnose")
    lowest = np.float32(sigma[1])
    ground = np.float32(sigmaf[1])
    first_top = np.float32(sigmaf[2])
    midpoint = np.multiply(
        np.float32(0.5), np.add(ground, first_top, dtype=np.float32), dtype=np.float32
    )
    if not (
        bool(np.isfinite(lowest))
        and bool(np.isfinite(first_top))
        and ground == np.float32(0.0)
        and ground < lowest < first_top
        and midpoint.view(np.uint32) == lowest.view(np.uint32)
    ):
        raise ValueError(
            f"{label}: sigma(1) = {float(lowest)!r} m is not param.F:8189-8191's "
            f"midpoint of a first layer from a flat ground at "
            f"{float(ground)!r} m to {float(first_top)!r} m, so kval = 1 would "
            "not name the lowest scalar level getpsmth diagnoses"
        )
    return float(lowest)


def validate_adaptive_move_stretch_proof(
    proof: StretchZ1MetricProof,
    *,
    sigma: Any | None = None,
) -> float:
    """Re-derive a ``stretch_z=1`` proof and name its lowest scalar level."""

    if not isinstance(proof, StretchZ1MetricProof):
        raise TypeError("vertical_grid_proof must be a StretchZ1MetricProof")
    expected = build_stretch_z1_metric_proof(proof.parameters)
    if proof != expected:
        raise ValueError("adaptive-move stretch_z=1 proof identity mismatch")
    np = importlib.import_module("numpy")
    grid = build_stretch_z1_grid(proof.parameters)
    if sigma is not None:
        shape = (proof.parameters.nk + 2,)
        if (
            not isinstance(sigma, np.ndarray)
            or sigma.dtype != np.dtype("float32")
            or sigma.shape != shape
            or not sigma.flags.c_contiguous
        ):
            raise ValueError(f"sigma must be C-order float32 shape {shape}")
        if not np.array_equal(sigma.view(np.uint32), grid.sigma.view(np.uint32)):
            raise ValueError("sigma differs from the proved stretch_z=1 grid")
    return _lowest_scalar_level_m(
        np, grid.sigma, grid.sigmaf, label="stretch_z=1 proof"
    )


def validate_adaptive_move_vertical_grid(vertical: Any) -> float:
    """Name the lowest scalar level of a proved or runtime vertical coordinate.

    Accepts a :class:`StretchZ1MetricProof` (re-derived from its parameters)
    or the runtime's own ``cm1_vertical_grid.VerticalLevels`` for any branch in
    :data:`ADAPTIVE_MOVE_STRETCH_Z`.
    """

    if isinstance(vertical, StretchZ1MetricProof):
        return validate_adaptive_move_stretch_proof(vertical)
    from .cm1_vertical_grid import VerticalLevels

    if not isinstance(vertical, VerticalLevels):
        raise TypeError(
            "vertical grid must be a StretchZ1MetricProof or "
            "cm1_vertical_grid.VerticalLevels"
        )
    if vertical.stretch_z not in ADAPTIVE_MOVE_STRETCH_Z:
        _refuse_selection("stretch_z", vertical.stretch_z, ADAPTIVE_MOVE_STRETCH_Z)
    np = importlib.import_module("numpy")
    return _lowest_scalar_level_m(
        np,
        vertical.sigma,
        vertical.sigmaf,
        label=f"stretch_z={vertical.stretch_z} coordinate",
    )


def adaptive_move_vertical_grid_identity(vertical: Any) -> str:
    """The sealed identity a receipt names for the diagnosed column."""

    if isinstance(vertical, StretchZ1MetricProof):
        return vertical.proof_sha256
    return str(vertical.sha256)


def _f32_add(np: Any, left: Any, right: Any) -> Any:
    return np.add(left, right, dtype=np.float32)


def _f32_sub(np: Any, left: Any, right: Any) -> Any:
    return np.subtract(left, right, dtype=np.float32)


def _f32_mul(np: Any, left: Any, right: Any) -> Any:
    return np.multiply(left, right, dtype=np.float32)


def _f32_div(np: Any, left: Any, right: Any) -> Any:
    return np.divide(left, right, dtype=np.float32)


#: ``xfref`` extents at or above this many faces are refused: the exactness
#: comparison below forms ``xfref(1) + d*(i-1)`` in binary64, which is itself
#: exact only while the face count stays far inside 2**29 (the CUDA dimension
#: arguments are uint32 and no admitted grid approaches it).
_MAX_REFERENCE_FACES = 1 << 24


def _exact_uniform_axis_error(np: Any, axis: str, faces: Any, mass: Any) -> str | None:
    """``param.F:6649-6687,7393``: one axis of CM1's reference coordinates.

    CM1 forms ``xfdp(i) = dble(dx)*(i-1)`` (less ``0.5d0*dble(dx)*nx`` under
    ``iorigin = 2``) and ``xh(i) = 0.5d0*(xfdp(i)+xfdp(i+1))`` in double
    precision and narrows both to ``real``.  The controller is word-exact on
    any spacing where that narrowing rounds nothing, so that is the admission:
    uniform faces whose spacing is a CM1 ``real``, every face equal in binary64
    to the double CM1 computes, and every mass point that double midpoint.
    """

    if faces.size >= _MAX_REFERENCE_FACES:
        return f"{axis}fref has {faces.size} faces, beyond the exactness bound"
    wide = faces.astype(np.float64)
    spacing = np.float64(wide[1] - wide[0])
    if not bool(spacing > 0.0) or np.float64(np.float32(spacing)) != spacing:
        return (
            f"{axis}fref spacing {float(spacing)!r} m is not a positive CM1 real"
        )
    expected = wide[0] + spacing * np.arange(wide.size, dtype=np.float64)
    if not bool(np.array_equal(wide, expected)):
        return (
            f"{axis}fref is not param.F:6649-6663's uniform d{axis}*(i-1) faces "
            "formed exactly in binary32"
        )
    midpoints = np.float64(0.5) * (wide[:-1] + wide[1:])
    if not bool(np.array_equal(mass.astype(np.float64), midpoints)):
        return (
            f"{axis}h is not param.F:6687's exact face midpoints "
            "0.5d0*(xfdp(i)+xfdp(i+1))"
        )
    return None


def _require_center_arrays(
    pressure_lowest: Any, xh: Any, yh: Any, xfref: Any, yfref: Any
) -> tuple[Any, Any, Any, Any, Any, int, int]:
    np = importlib.import_module("numpy")
    if (
        not isinstance(pressure_lowest, np.ndarray)
        or pressure_lowest.dtype != np.dtype("float32")
        or pressure_lowest.ndim != 2
        or not pressure_lowest.flags.c_contiguous
    ):
        raise ValueError("pressure_lowest must be a C-order float32 2-D array")
    nj, ni = pressure_lowest.shape
    if ni < 3 or nj < 3:
        raise ValueError("adaptive pressure diagnosis requires ni,nj >= 3")
    for name, value, shape in (
        ("xh", xh, (ni,)),
        ("yh", yh, (nj,)),
        ("xfref", xfref, (ni + 1,)),
        ("yfref", yfref, (nj + 1,)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order float32 shape {shape}")
    for name, value in (
        ("pressure_lowest", pressure_lowest),
        ("xh", xh),
        ("yh", yh),
        ("xfref", xfref),
        ("yfref", yfref),
    ):
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
    if not bool(np.all(np.diff(xh) > np.float32(0.0))) or not bool(
        np.all(np.diff(yh) > np.float32(0.0))
    ):
        raise ValueError("mass coordinates must be strictly increasing")
    for axis, faces, mass in (("x", xfref, xh), ("y", yfref, yh)):
        error = _exact_uniform_axis_error(np, axis, faces, mass)
        if error is not None:
            raise ValueError(
                "adaptive-move coordinates must be exact uniform reference "
                f"faces: {error}"
            )
    expected_xh = _f32_mul(np, np.float32(0.5), _f32_add(np, xfref[:-1], xfref[1:]))
    expected_yh = _f32_mul(np, np.float32(0.5), _f32_add(np, yfref[:-1], yfref[1:]))
    if not np.array_equal(
        xh.view(np.uint32), expected_xh.view(np.uint32)
    ) or not np.array_equal(yh.view(np.uint32), expected_yh.view(np.uint32)):
        raise ValueError("mass coordinates must be raw-bit face midpoints")
    values = (pressure_lowest, xh, yh, xfref, yfref)
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    ):
        raise ValueError("adaptive pressure/coordinate arrays must not alias")
    return pressure_lowest, xh, yh, xfref, yfref, ni, nj


def cpu_periodic_pressure_center(
    pressure_lowest: Any,
    xh: Any,
    yh: Any,
    xfref: Any,
    yfref: Any,
    *,
    centerx: float,
    centery: float,
) -> AdaptivePressureCenter:
    """Transcribe getpsmth/getpcent type=2 for serial periodic boundaries."""

    np = importlib.import_module("numpy")
    p, xh, yh, xfref, yfref, ni, nj = _require_center_arrays(
        pressure_lowest, xh, yh, xfref, yfref
    )
    centerx32, centery32 = np.float32(centerx), np.float32(centery)
    if not bool(np.isfinite(centerx32)) or not bool(np.isfinite(centery32)):
        raise ValueError("domain center must be finite float32")
    work = np.ascontiguousarray(p, dtype=np.float32).copy()
    one_ninth = np.float32(0.11111111)
    for _ in range(100):
        east = np.roll(work, -1, axis=1)
        west = np.roll(work, 1, axis=1)
        north = np.roll(work, -1, axis=0)
        south = np.roll(work, 1, axis=0)
        northeast = np.roll(east, -1, axis=0)
        northwest = np.roll(west, -1, axis=0)
        southeast = np.roll(east, 1, axis=0)
        southwest = np.roll(west, 1, axis=0)
        cardinal = _f32_add(np, _f32_add(np, east, west), _f32_add(np, north, south))
        diagonal = _f32_add(
            np,
            _f32_add(np, northeast, northwest),
            _f32_add(np, southeast, southwest),
        )
        total = _f32_add(np, work, _f32_add(np, cardinal, diagonal))
        work = np.ascontiguousarray(_f32_mul(np, one_ninth, total))

    west = np.roll(work, 1, axis=1)
    south = np.roll(work, 1, axis=0)
    southwest = np.roll(west, 1, axis=0)
    zeta = _f32_mul(
        np,
        np.float32(0.25),
        _f32_add(np, _f32_add(np, work, southwest), _f32_add(np, west, south)),
    )
    minimum_flat = int(np.argmin(zeta))
    minimum_j, minimum_i = divmod(minimum_flat, ni)

    ca = np.float64(0.0)
    cx = np.float64(0.0)
    cy = np.float64(0.0)
    radmax = np.float32(300000.0)
    for j in range(nj):
        dy = _f32_sub(np, yh[j], centery32)
        dy2 = _f32_mul(np, dy, dy)
        for i in range(ni):
            dx = _f32_sub(np, xh[i], centerx32)
            radius2 = _f32_add(np, _f32_mul(np, dx, dx), dy2)
            if np.sqrt(radius2, dtype=np.float32) < radmax:
                aa = work[j, i]
                ca = np.float64(ca + np.float64(aa))
                cx = np.float64(cx + np.float64(_f32_mul(np, aa, xh[i])))
                cy = np.float64(cy + np.float64(_f32_mul(np, aa, yh[j])))
    if not abs(float(ca)) > float(np.float32(1.0e-30)):
        raise ValueError("CM1 adaptive pressure centroid has zero weight")
    cx = np.float64(cx / ca)
    cy = np.float64(cy / ca)

    icenter = 0
    minval = np.float64(1.0e30)
    for i in range(ni + 1):
        distance = abs(np.float64(xfref[i]) - cx)
        if distance < minval:
            minval = distance
            icenter = i
    jcenter = 0
    minval = np.float64(1.0e30)
    for j in range(nj + 1):
        distance = abs(np.float64(yfref[j]) - cy)
        if distance < minval:
            minval = distance
            jcenter = j

    return AdaptivePressureCenter(
        work,
        minimum_i + 1,
        minimum_j + 1,
        float(xfref[minimum_i]),
        float(yfref[minimum_j]),
        icenter + 1,
        jcenter + 1,
        float(np.float32(cx)),
        float(np.float32(cy)),
        float(ca),
    )


def _require_host_winds(winds: Mapping[str, Any]) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    if not isinstance(winds, Mapping):
        raise TypeError("winds must be a mapping")
    missing = [name for name in WIND_FIELDS if name not in winds]
    if missing:
        raise ValueError(f"missing adaptive-move wind arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in WIND_FIELDS:
        value = winds[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"{name} must be a writeable C-order float32 array")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    if not (result["u0"].shape == result["ua"].shape == result["u3d"].shape):
        raise ValueError("u0, ua, and u3d must have identical supplied carrier shapes")
    if not (result["v0"].shape == result["va"].shape == result["v3d"].shape):
        raise ValueError("v0, va, and v3d must have identical supplied carrier shapes")
    if result["ug"].ndim != 1 or result["vg"].shape != result["ug"].shape:
        raise ValueError("ug and vg must be same-shape one-dimensional profiles")
    u_shape = result["u0"].shape
    v_shape = result["v0"].shape
    if len(u_shape) != 3 or len(v_shape) != 3 or 0 in u_shape or 0 in v_shape:
        raise ValueError(
            "u and v adaptive-move carriers must be nonempty [k,j,i] arrays"
        )
    if (
        u_shape[0] != result["ug"].size
        or v_shape[0] != result["vg"].size
        or u_shape[1] + 1 != v_shape[1]
        or u_shape[2] != v_shape[2] + 1
    ):
        raise ValueError("adaptive-move wind carriers do not share the CM1 u/v stagger")
    values = tuple(result.values())
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    ):
        raise ValueError("adaptive-move wind arrays must not alias")
    return result


def cpu_advance_adaptive_move_after_large_step(
    state: AdaptiveMoveState,
    winds: Mapping[str, Any],
    *,
    nstep: int,
    mtime_after: float,
    dbldt: float,
    adapt_move_frq: float,
    centerx: float,
    centery: float,
    pressure_lowest: Any | None = None,
    xh: Any | None = None,
    yh: Any | None = None,
    xfref: Any | None = None,
    yfref: Any | None = None,
    dowriteout: bool = False,
    dohifrqwrite: bool = False,
    doazimwrite: bool = False,
    restarted: bool = False,
    vertical_grid_proof: Any,
    selection: AdaptiveMoveSelection = DEFAULT_ADAPTIVE_MOVE_SELECTION,
) -> AdaptiveMoveTransition:
    """Apply CM1's completed-large-step domain-location/controller cadence."""

    require_adaptive_move_selection(selection)
    validate_adaptive_move_vertical_grid(vertical_grid_proof)
    _validate_state(state)
    a = _require_host_winds(winds)
    if isinstance(nstep, bool) or not isinstance(nstep, int) or nstep < 0:
        raise ValueError("nstep must be a nonnegative integer")
    if not all(
        type(value) is bool
        for value in (dowriteout, dohifrqwrite, doazimwrite, restarted)
    ):
        raise TypeError("cadence and restart flags must be bool")
    np = importlib.import_module("numpy")
    frequency = np.float32(adapt_move_frq)
    centerx32, centery32 = np.float32(centerx), np.float32(centery)
    if not bool(np.isfinite(frequency)) or frequency <= np.float32(0.0):
        raise ValueError("adapt_move_frq must be a positive finite float32")
    if (
        not math.isfinite(float(mtime_after))
        or not math.isfinite(float(dbldt))
        or float(dbldt) <= 0.0
    ):
        raise ValueError("mtime_after and positive dbldt must be finite float64 values")

    old_u = np.float32(state.umove)
    old_v = np.float32(state.vmove)
    state.domainlocx = float(
        np.float64(state.domainlocx) + np.float64(dbldt) * np.float64(old_u)
    )
    state.domainlocy = float(
        np.float64(state.domainlocy) + np.float64(dbldt) * np.float64(old_v)
    )
    due = float(mtime_after) >= float(state.adaptmovetim)
    diagnosis: AdaptivePressureCenter | None = None
    diagnosed = (
        due
        and nstep >= 1
        and (dowriteout or dohifrqwrite or doazimwrite)
        and not restarted
    )
    if diagnosed:
        if any(value is None for value in (pressure_lowest, xh, yh, xfref, yfref)):
            raise ValueError(
                "due center diagnosis requires pressure and coordinate arrays"
            )
        center_values = (pressure_lowest, xh, yh, xfref, yfref)
        if any(
            np.shares_memory(center_value, wind_value)
            for center_value in center_values
            for wind_value in a.values()
        ):
            raise ValueError(
                "adaptive pressure/coordinate inputs must not alias wind carriers"
            )
        diagnosis = cpu_periodic_pressure_center(
            pressure_lowest,
            xh,
            yh,
            xfref,
            yfref,
            centerx=float(centerx32),
            centery=float(centery32),
        )
        state.icenter = diagnosis.icenter
        state.jcenter = diagnosis.jcenter
        state.xcenter = diagnosis.xcenter
        state.ycenter = diagnosis.ycenter

    delta_u = np.float32(0.0)
    delta_v = np.float32(0.0)
    if due and nstep >= 1:
        term2 = np.float32(0.5)
        center_departure_u = _f32_div(
            np,
            _f32_mul(np, term2, _f32_sub(np, centerx32, np.float32(state.xcenter))),
            frequency,
        )
        lag_u = _f32_mul(
            np,
            _f32_sub(np, np.float32(1.0), term2),
            _f32_sub(np, np.float32(state.umovem2), np.float32(state.umovem1)),
        )
        new_u = _f32_add(np, _f32_sub(np, old_u, center_departure_u), lag_u)
        center_departure_v = _f32_div(
            np,
            _f32_mul(np, term2, _f32_sub(np, centery32, np.float32(state.ycenter))),
            frequency,
        )
        lag_v = _f32_mul(
            np,
            _f32_sub(np, np.float32(1.0), term2),
            _f32_sub(np, np.float32(state.vmovem2), np.float32(state.vmovem1)),
        )
        new_v = _f32_add(np, _f32_sub(np, old_v, center_departure_v), lag_v)
        state.umovem2 = float(np.float32(state.umovem1))
        state.vmovem2 = float(np.float32(state.vmovem1))
        state.umovem1 = float(old_u)
        state.vmovem1 = float(old_v)
        state.umove = float(new_u)
        state.vmove = float(new_v)
        delta_u = _f32_sub(np, old_u, new_u)
        delta_v = _f32_sub(np, old_v, new_v)
        for name in ("u0", "ua", "u3d", "ug"):
            np.add(a[name], delta_u, out=a[name], dtype=np.float32)
        for name in ("v0", "va", "v3d", "vg"):
            np.add(a[name], delta_v, out=a[name], dtype=np.float32)

    if due:
        state.nwritemv += 1
        state.adaptmovetim = float(
            np.float64(state.adaptmovetim) + np.float64(frequency)
        )
    _validate_state(state)
    return AdaptiveMoveTransition(
        due, diagnosed, float(delta_u), float(delta_v), diagnosis
    )


CUDA_ADAPTIVE_MOVE_SOURCE = r"""// gpuwm-cm1.cuda-adaptive-move-kernels/v1
static_assert(sizeof(float)==4,"float32 required");
static_assert(sizeof(double)==8,"float64 required");
static_assert(sizeof(int)==4,"int32 required");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
extern "C" __global__ void cm1_adapt_smooth9_periodic_f32_v1(const float* src,float* dst,unsigned ni,unsigned nj){unsigned n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=ni*nj)return;unsigned i=n%ni,j=n/ni,im=i?i-1:ni-1,ip=i+1==ni?0:i+1,jm=j?j-1:nj-1,jp=j+1==nj?0:j+1;float ew=a32(src[j*ni+ip],src[j*ni+im]);float ns=a32(src[jp*ni+i],src[jm*ni+i]);float card=a32(ew,ns);float dn=a32(src[jp*ni+ip],src[jp*ni+im]);float ds=a32(src[jm*ni+ip],src[jm*ni+im]);float diag=a32(dn,ds);dst[n]=m32(0.11111111f,a32(src[n],a32(card,diag)));}
extern "C" __global__ void cm1_adapt_center_serial_f32_v1(const float* p,const float* xh,const float* yh,const float* xf,const float* yf,float* sf,double* sd,int* si,int* status,unsigned ni,unsigned nj,float centerx,float centery,double mtime,int enabled){if(blockIdx.x||threadIdx.x||!enabled||mtime<sd[2])return;float rmin=1.0e30f;int ipmin=1,jpmin=1;for(unsigned j=0;j<nj;++j){unsigned jm=j?j-1:nj-1;for(unsigned i=0;i<ni;++i){unsigned im=i?i-1:ni-1;float z=m32(0.25f,a32(a32(p[j*ni+i],p[jm*ni+im]),a32(p[j*ni+im],p[jm*ni+i])));if(z<rmin){rmin=z;ipmin=(int)i+1;jpmin=(int)j+1;}}}double ca=0.0,cx=0.0,cy=0.0;for(unsigned j=0;j<nj;++j){float dy=s32(yh[j],centery);float dy2=m32(dy,dy);for(unsigned i=0;i<ni;++i){float dx=s32(xh[i],centerx);float rr=sqrtf(a32(m32(dx,dx),dy2));if(rr<300000.0f){float aa=p[j*ni+i];ca=__dadd_rn(ca,double(aa));cx=__dadd_rn(cx,double(m32(aa,xh[i])));cy=__dadd_rn(cy,double(m32(aa,yh[j])));}}}if(!(fabs(ca)>double(1.0e-30f))){status[0]=1;return;}cx=__ddiv_rn(cx,ca);cy=__ddiv_rn(cy,ca);int ic=ipmin,jc=jpmin;double minval=1.0e30;for(unsigned i=0;i<=ni;++i){double d=fabs(double(xf[i])-cx);if(d<minval){minval=d;ic=(int)i+1;}}minval=1.0e30;for(unsigned j=0;j<=nj;++j){double d=fabs(double(yf[j])-cy);if(d<minval){minval=d;jc=(int)j+1;}}si[0]=ic;si[1]=jc;sf[6]=(float)cx;sf[7]=(float)cy;}
extern "C" __global__ void cm1_adapt_commit_f32_f64_v1(float* sf,double* sd,int* si,float* delta,int* active,const int* status,int nstep,double mtime,double dbldt,float frq,float centerx,float centery){if(blockIdx.x||threadIdx.x)return;float oldu=sf[0],oldv=sf[3];sd[0]=__dadd_rn(sd[0],__dmul_rn(dbldt,double(oldu)));sd[1]=__dadd_rn(sd[1],__dmul_rn(dbldt,double(oldv)));delta[0]=0.0f;delta[1]=0.0f;active[0]=0;if(status[0])return;if(mtime>=sd[2]){if(nstep>=1){float term2=0.5f;float newu=a32(s32(oldu,d32(m32(term2,s32(centerx,sf[6])),frq)),m32(s32(1.0f,term2),s32(sf[2],sf[1])));float newv=a32(s32(oldv,d32(m32(term2,s32(centery,sf[7])),frq)),m32(s32(1.0f,term2),s32(sf[5],sf[4])));sf[2]=sf[1];sf[5]=sf[4];sf[1]=oldu;sf[4]=oldv;sf[0]=newu;sf[3]=newv;delta[0]=s32(oldu,newu);delta[1]=s32(oldv,newv);active[0]=1;}si[2]+=1;sd[2]=__dadd_rn(sd[2],double(frq));}}
extern "C" __global__ void cm1_adapt_add_delta_f32_v1(float* a,unsigned long long n,const float* delta,const int* active,unsigned which){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(active[0]&&q<n)a[q]=a32(a[q],delta[which]);}
"""
CUDA_ADAPTIVE_MOVE_SOURCE_BYTES = CUDA_ADAPTIVE_MOVE_SOURCE.encode("utf-8")
CUDA_ADAPTIVE_MOVE_SOURCE_SHA256 = (
    "c056d22a1b33239153ddb0152b0f6743db37cffa3c77e73795a145708eacb42f"
)


@dataclass(frozen=True, slots=True)
class CompiledAdaptiveMove:
    module_owner: Any
    smooth_kernel: Any
    center_kernel: Any
    commit_kernel: Any
    add_kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_adaptive_move() -> CompiledAdaptiveMove:
    actual = hashlib.sha256(CUDA_ADAPTIVE_MOVE_SOURCE_BYTES).hexdigest()
    if actual != CUDA_ADAPTIVE_MOVE_SOURCE_SHA256:
        raise RuntimeError(
            f"packaged adaptive-move CUDA source differs from pin: {actual}"
        )
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_ADAPTIVE_MOVE_SOURCE, "cm1_adaptive_move_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_ADAPTIVE_MOVE_SOURCE, "cm1_adaptive_move_cubin.cu", method="cubin"
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    names = (
        "cm1_adapt_smooth9_periodic_f32_v1",
        "cm1_adapt_center_serial_f32_v1",
        "cm1_adapt_commit_f32_f64_v1",
        "cm1_adapt_add_delta_f32_v1",
    )
    for name in names:
        if not re.search(rf"\.entry\s+{name}(?:\s|\()", text):
            raise ValueError(f"compiled PTX lacks {name}")
    module = function.Module()
    module.load(cubin)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledAdaptiveMove(
        module,
        module.get_function(names[0]),
        module.get_function(names[1]),
        module.get_function(names[2]),
        module.get_function(names[3]),
        ptx,
        cubin,
        arch,
        identity,
    )


@dataclass(frozen=True, slots=True)
class DeviceAdaptiveMoveState:
    f32: Any
    f64: Any
    i32: Any


def upload_adaptive_move_state(
    state: AdaptiveMoveState, *, stream: Any | None = None
) -> DeviceAdaptiveMoveState:
    cupy = importlib.import_module("cupy")
    f32, f64, i32 = pack_adaptive_move_state(state)
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    with stream:
        return DeviceAdaptiveMoveState(
            cupy.asarray(f32), cupy.asarray(f64), cupy.asarray(i32)
        )


def download_adaptive_move_state(state: DeviceAdaptiveMoveState) -> AdaptiveMoveState:
    cupy = importlib.import_module("cupy")
    if not isinstance(state, DeviceAdaptiveMoveState):
        raise TypeError("state must be a DeviceAdaptiveMoveState")
    return unpack_adaptive_move_state(
        cupy.asnumpy(state.f32), cupy.asnumpy(state.f64), cupy.asnumpy(state.i32)
    )


@dataclass(frozen=True, slots=True)
class CudaAdaptiveMoveLaunch:
    ready_event: Any
    compiled: CompiledAdaptiveMove
    state: DeviceAdaptiveMoveState
    psmth: Any
    delta: Any
    active: Any
    diagnostic_status: Any
    diagnosis_inputs_bound: bool
    wind_arrays: Mapping[str, Any]
    receipt: dict[str, Any]


def _device_span(value: Any) -> tuple[int, int]:
    start = int(value.data.ptr)
    return start, start + int(value.nbytes)


def _require_nonoverlapping_device_arrays(
    values: Mapping[str, Any], *, family: str
) -> None:
    spans = [(name, *_device_span(value)) for name, value in values.items()]
    for index, (left_name, left_start, left_end) in enumerate(spans):
        for right_name, right_start, right_end in spans[index + 1 :]:
            if left_start < right_end and right_start < left_end:
                raise ValueError(
                    f"{family} arrays {left_name} and {right_name} overlap"
                )


def _require_device_state(cupy: Any, state: DeviceAdaptiveMoveState) -> None:
    if not isinstance(state, DeviceAdaptiveMoveState):
        raise TypeError("state must be a DeviceAdaptiveMoveState")
    for name, value, dtype, shape in (
        ("f32", state.f32, cupy.float32, (8,)),
        ("f64", state.f64, cupy.float64, (3,)),
        ("i32", state.i32, cupy.int32, (4,)),
    ):
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != dtype
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"device {name} state carrier has wrong dtype/shape/layout"
            )
    _require_nonoverlapping_device_arrays(
        {"state.f32": state.f32, "state.f64": state.f64, "state.i32": state.i32},
        family="adaptive-move state",
    )


def _require_device_winds(cupy: Any, winds: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(winds, Mapping):
        raise TypeError("winds must be a mapping")
    missing = [name for name in WIND_FIELDS if name not in winds]
    if missing:
        raise ValueError(f"missing adaptive-move device winds: {', '.join(missing)}")
    result = {}
    for name in WIND_FIELDS:
        value = winds[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be a C-order CuPy float32 array")
        result[name] = value
    if not (result["u0"].shape == result["ua"].shape == result["u3d"].shape):
        raise ValueError("device u0, ua, and u3d shapes differ")
    if not (result["v0"].shape == result["va"].shape == result["v3d"].shape):
        raise ValueError("device v0, va, and v3d shapes differ")
    if result["ug"].ndim != 1 or result["vg"].shape != result["ug"].shape:
        raise ValueError("device ug and vg must be same-shape profiles")
    u_shape = result["u0"].shape
    v_shape = result["v0"].shape
    if len(u_shape) != 3 or len(v_shape) != 3 or 0 in u_shape or 0 in v_shape:
        raise ValueError("device u and v carriers must be nonempty [k,j,i] arrays")
    if (
        u_shape[0] != result["ug"].size
        or v_shape[0] != result["vg"].size
        or u_shape[1] + 1 != v_shape[1]
        or u_shape[2] != v_shape[2] + 1
    ):
        raise ValueError("device adaptive-move winds do not share the CM1 u/v stagger")
    _require_nonoverlapping_device_arrays(result, family="adaptive-move wind")
    return result


def _require_device_center_arrays(
    cupy: Any,
    pressure_lowest: Any,
    xh: Any,
    yh: Any,
    xfref: Any,
    yfref: Any,
) -> tuple[int, int]:
    if (
        not isinstance(pressure_lowest, cupy.ndarray)
        or pressure_lowest.dtype != cupy.float32
        or pressure_lowest.ndim != 2
        or not pressure_lowest.flags.c_contiguous
    ):
        raise ValueError("device pressure_lowest must be C-order CuPy float32 2-D")
    nj, ni = map(int, pressure_lowest.shape)
    if ni < 3 or nj < 3:
        raise ValueError("device adaptive pressure diagnosis requires ni,nj >= 3")
    values = {"pressure_lowest": pressure_lowest}
    for name, value, shape in (
        ("xh", xh, (ni,)),
        ("yh", yh, (nj,)),
        ("xfref", xfref, (ni + 1,)),
        ("yfref", yfref, (nj + 1,)),
    ):
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"device {name} must be C-order CuPy float32 shape {shape}"
            )
        values[name] = value
    _require_nonoverlapping_device_arrays(values, family="adaptive pressure/coordinate")
    if not all(
        bool(cupy.all(cupy.isfinite(value)).item()) for value in values.values()
    ):
        raise ValueError(
            "device adaptive pressure/coordinate input contains non-finite values"
        )
    for axis, faces, mass in (("x", xfref, xh), ("y", yfref, yh)):
        error = _exact_uniform_axis_error(cupy, axis, faces, mass)
        if error is not None:
            raise ValueError(
                "device adaptive-move coordinates must be exact uniform "
                f"reference faces: {error}"
            )
    expected_xh = cupy.float32(0.5) * (xfref[:-1] + xfref[1:])
    expected_yh = cupy.float32(0.5) * (yfref[:-1] + yfref[1:])
    if not bool(
        cupy.array_equal(xh.view(cupy.uint32), expected_xh.view(cupy.uint32))
    ) or not bool(
        cupy.array_equal(yh.view(cupy.uint32), expected_yh.view(cupy.uint32))
    ):
        raise ValueError("device mass coordinates must be raw-bit face midpoints")
    return ni, nj


def launch_cuda_adaptive_move_after_large_step(
    state: DeviceAdaptiveMoveState,
    winds: Mapping[str, Any],
    *,
    nstep: int,
    mtime_after: float,
    dbldt: float,
    adapt_move_frq: float,
    centerx: float,
    centery: float,
    pressure_lowest: Any | None = None,
    xh: Any | None = None,
    yh: Any | None = None,
    xfref: Any | None = None,
    yfref: Any | None = None,
    dowriteout: bool = False,
    dohifrqwrite: bool = False,
    doazimwrite: bool = False,
    restarted: bool = False,
    vertical_grid_proof: Any,
    selection: AdaptiveMoveSelection = DEFAULT_ADAPTIVE_MOVE_SELECTION,
    stream: Any | None = None,
    compiled: CompiledAdaptiveMove | None = None,
) -> CudaAdaptiveMoveLaunch:
    """Launch source-ordered diagnosis/controller math without a host state readback."""

    require_adaptive_move_selection(selection)
    lowest_level_m = validate_adaptive_move_vertical_grid(vertical_grid_proof)
    cupy = importlib.import_module("cupy")
    _require_device_state(cupy, state)
    a = _require_device_winds(cupy, winds)
    _require_nonoverlapping_device_arrays(
        {
            "state.f32": state.f32,
            "state.f64": state.f64,
            "state.i32": state.i32,
            **{f"wind.{name}": value for name, value in a.items()},
        },
        family="adaptive-move state/wind",
    )
    if isinstance(nstep, bool) or not isinstance(nstep, int) or nstep < 0:
        raise ValueError("nstep must be a nonnegative integer")
    if not all(
        type(value) is bool
        for value in (dowriteout, dohifrqwrite, doazimwrite, restarted)
    ):
        raise TypeError("cadence and restart flags must be bool")
    frequency = cupy.float32(adapt_move_frq)
    if not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
        raise ValueError("adapt_move_frq must be positive finite float32")
    centerx32 = cupy.float32(centerx)
    centery32 = cupy.float32(centery)
    if not math.isfinite(float(centerx32)) or not math.isfinite(float(centery32)):
        raise ValueError("domain center coordinates must be finite float32 values")
    if (
        not math.isfinite(float(mtime_after))
        or not math.isfinite(float(dbldt))
        or float(dbldt) <= 0.0
    ):
        raise ValueError("mtime_after and positive dbldt must be finite")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    diagnose = (
        nstep >= 1 and (dowriteout or dohifrqwrite or doazimwrite) and not restarted
    )
    if diagnose:
        if any(value is None for value in (pressure_lowest, xh, yh, xfref, yfref)):
            raise ValueError(
                "requested device center diagnosis lacks pressure/coordinate arrays"
            )
        with stream:
            ni, nj = _require_device_center_arrays(
                cupy, pressure_lowest, xh, yh, xfref, yfref
            )
        _require_nonoverlapping_device_arrays(
            {
                "state.f32": state.f32,
                "state.f64": state.f64,
                "state.i32": state.i32,
                **{f"wind.{name}": value for name, value in a.items()},
                "pressure_lowest": pressure_lowest,
                "xh": xh,
                "yh": yh,
                "xfref": xfref,
                "yfref": yfref,
            },
            family="adaptive-move launch",
        )
    else:
        ni = nj = 1
    compiled = compile_cuda_adaptive_move() if compiled is None else compiled
    if not isinstance(compiled, CompiledAdaptiveMove):
        raise TypeError("compiled must be a CompiledAdaptiveMove")
    threads = 256
    with stream:
        delta = cupy.zeros(2, dtype=cupy.float32)
        active = cupy.zeros(1, dtype=cupy.int32)
        diagnostic_status = cupy.zeros(1, dtype=cupy.int32)
        if diagnose:
            first = cupy.empty_like(pressure_lowest)
            second = cupy.empty_like(pressure_lowest)
            cupy.copyto(first, pressure_lowest)
            blocks = ((ni * nj + threads - 1) // threads, 1, 1)
            for iteration in range(100):
                compiled.smooth_kernel(
                    blocks,
                    (threads, 1, 1),
                    (first, second, cupy.uint32(ni), cupy.uint32(nj)),
                    stream=stream,
                )
                first, second = second, first
            psmth = first
            compiled.center_kernel(
                (1, 1, 1),
                (1, 1, 1),
                (
                    psmth,
                    xh,
                    yh,
                    xfref,
                    yfref,
                    state.f32,
                    state.f64,
                    state.i32,
                    diagnostic_status,
                    cupy.uint32(ni),
                    cupy.uint32(nj),
                    centerx32,
                    centery32,
                    cupy.float64(mtime_after),
                    cupy.int32(1),
                ),
                stream=stream,
            )
        else:
            psmth = cupy.empty((0,), dtype=cupy.float32)
        compiled.commit_kernel(
            (1, 1, 1),
            (1, 1, 1),
            (
                state.f32,
                state.f64,
                state.i32,
                delta,
                active,
                diagnostic_status,
                cupy.int32(nstep),
                cupy.float64(mtime_after),
                cupy.float64(dbldt),
                frequency,
                centerx32,
                centery32,
            ),
            stream=stream,
        )
        for name in ("u0", "ua", "u3d", "ug"):
            value = a[name]
            blocks = ((int(value.size) + threads - 1) // threads, 1, 1)
            compiled.add_kernel(
                blocks,
                (threads, 1, 1),
                (value, cupy.uint64(value.size), delta, active, cupy.uint32(0)),
                stream=stream,
            )
        for name in ("v0", "va", "v3d", "vg"):
            value = a[name]
            blocks = ((int(value.size) + threads - 1) // threads, 1, 1)
            compiled.add_kernel(
                blocks,
                (threads, 1, 1),
                (value, cupy.uint64(value.size), delta, active, cupy.uint32(1)),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_ADAPTIVE_MOVE_SCHEMA,
        "abi": CUDA_ADAPTIVE_MOVE_ABI,
        "state_schema": CUDA_ADAPTIVE_MOVE_STATE_SCHEMA,
        "selection": {
            name: getattr(selection, name) for name in selection.__dataclass_fields__
        },
        "cadence": {
            "nstep": nstep,
            "mtime_after": float(mtime_after),
            "dbldt": float(dbldt),
            "adapt_move_frq_f32": float(frequency),
            "diagnosis_inputs_bound": diagnose,
            "actual_due_and_diagnosis_device_gated": True,
            "restarted": restarted,
        },
        "vertical_grid": {
            "proof_sha256": adaptive_move_vertical_grid_identity(vertical_grid_proof),
            "lowest_scalar_level_m": lowest_level_m,
            "source_k_index": 1,
        },
        "kernel": {
            "source_sha256": CUDA_ADAPTIVE_MOVE_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
        },
        "runtime": dict(compiled.runtime_identity),
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
        },
        "restart_contract": adaptive_move_restart_contract(),
        "source_anchors": dict(SOURCE_ANCHORS),
        "claim_limit": CM1_ADAPTIVE_MOVE_CLAIM_LIMIT,
    }
    return CudaAdaptiveMoveLaunch(
        event,
        compiled,
        state,
        psmth,
        delta,
        active,
        diagnostic_status,
        diagnose,
        a,
        receipt,
    )


def validate_cuda_adaptive_move_completion(launch: CudaAdaptiveMoveLaunch) -> None:
    """Synchronize the launch and fail closed on a zero-weight centroid."""

    if not isinstance(launch, CudaAdaptiveMoveLaunch):
        raise TypeError("launch must be a CudaAdaptiveMoveLaunch")
    cupy = importlib.import_module("cupy")
    launch.ready_event.synchronize()
    status = int(cupy.asnumpy(launch.diagnostic_status)[0])
    if status != 0:
        raise ValueError("CM1 adaptive pressure centroid has zero weight")
    _validate_state(download_adaptive_move_state(launch.state))
    for name, value in launch.wind_arrays.items():
        if not bool(cupy.all(cupy.isfinite(value)).item()):
            raise ValueError(
                f"completed adaptive-move wind {name} contains non-finite values"
            )


__all__ = (
    "ADAPTIVE_MOVE_RESTART_SOURCE_ORDER",
    "ADAPTIVE_MOVE_STRETCH_Z",
    "adaptive_move_vertical_grid_identity",
    "validate_adaptive_move_vertical_grid",
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "AdaptiveMoveSelection",
    "AdaptiveMoveState",
    "AdaptiveMoveTransition",
    "AdaptivePressureCenter",
    "CM1_ADAPTIVE_MOVE_CLAIM_LIMIT",
    "CUDA_ADAPTIVE_MOVE_ABI",
    "CUDA_ADAPTIVE_MOVE_SCHEMA",
    "CUDA_ADAPTIVE_MOVE_SOURCE_BYTES",
    "CUDA_ADAPTIVE_MOVE_SOURCE_SHA256",
    "CUDA_ADAPTIVE_MOVE_STATE_SCHEMA",
    "CompiledAdaptiveMove",
    "CudaAdaptiveMoveLaunch",
    "DEFAULT_ADAPTIVE_MOVE_SELECTION",
    "DeviceAdaptiveMoveState",
    "F32_STATE_FIELDS",
    "F64_STATE_FIELDS",
    "I32_STATE_FIELDS",
    "SOURCE_ANCHORS",
    "WIND_FIELDS",
    "adaptive_move_restart_contract",
    "compile_cuda_adaptive_move",
    "cpu_advance_adaptive_move_after_large_step",
    "cpu_periodic_pressure_center",
    "download_adaptive_move_state",
    "initialize_adaptive_move_state",
    "launch_cuda_adaptive_move_after_large_step",
    "pack_adaptive_move_state",
    "require_adaptive_move_selection",
    "unpack_adaptive_move_state",
    "upload_adaptive_move_state",
    "validate_adaptive_move_stretch_proof",
    "validate_cuda_adaptive_move_completion",
)
