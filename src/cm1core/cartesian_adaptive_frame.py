"""Bind CM1's ``do_adapt_move`` controller to the Cartesian runtime.

``cuda_adaptive_move.py`` and ``cuda_adaptive_move_open.py`` already carry
CM1 r21.1's ``update_adapt_move``/``imove=1`` controller exactly -- the
lowest-scalar-level pressure diagnosis, the official hundred-pass nine-point
smoother, the type-2 pressure centroid, the one cadence update, the
moving-frame wind rebasing and the persistent controller state.  Nothing in
the configuration grammar or the Cartesian door reached them, so every run
this port has ever driven used a frame velocity declared once and never
changed.  This module is the seam that was missing, and only the seam: it
adds no numerics.

Two things have to be adapted, and both are stated rather than hidden.

*Boundaries.*  ``cuda_adaptive_move.py``'s smoother wraps (CM1's ``bc2d`` for
four-periodic lateral boundaries) and ``cuda_adaptive_move_open.py``'s
replicates the edge (CM1's ``bc2d`` for the four code-2 radiative/open
boundaries this lane actually runs).  The two reach different minima on the
same field, so the branch is selected from ``[boundaries]`` by value and every
other combination is refused rather than approximated.

*Carriers.*  ``azimavg.F:4338-4368``'s ``change_uvmove`` shifts ``u0``,
``ua``, ``u3d``, ``v0``, ``va``, ``v3d`` over ``ib:ie+1,jb:je,kb:ke`` and
``update_adapt_move`` shifts ``ug``/``vg`` beside them.  This port's Cartesian
dynamics holds ``ua``/``u3d``/``va``/``v3d`` as exactly that padded rectangle,
holds the horizontally uniform base wind as CM1's ``u0``/``v0`` compressed to
one word per level, and carries no ``ug``/``vg`` at all because
``solve1.F:590-631``'s large-scale pressure gradient is not in this lane.  The
sealed operators' own launch entry points require CM1's full eight-array
inventory and are left exactly as they are; the launch below drives the same
compiled kernels -- the same PTX, the same ``__fadd_rn`` -- over the carriers
this lane really owns, and the receipt names every one of them.

This module does not imply moving terrain, MPI decomposition, nesting, or a
storm-following nest.  Each of those is refused by name.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import cuda_adaptive_move as periodic
from . import cuda_adaptive_move_open as radiative_open
from .cuda_adaptive_move_restart_lifecycle import (
    ADAPTIVE_MOVE_RESTART_LIFECYCLE_SCHEMA,
    AdaptiveMoveBranch,
    adaptive_move_restart_lifecycle_contract,
)
from .errors import refuse


CARTESIAN_ADAPTIVE_FRAME_SCHEMA = "gpuwm-cm1.cartesian-adaptive-frame/v1"
CARTESIAN_GATE = "cartesian-runtime"
#: The restart scalar table's namespace for CM1's controller words.  A
#: fixed-frame run writes none of these rows, so its restart is byte-identical
#: to the one it wrote before the moving domain became adaptive.
ADAPTIVE_MOVE_RESTART_PREFIX = "adapt_move."

#: Neither ``getpsmth`` nor ``getpcent`` reads ``dx``, ``dy`` or the vertical
#: coordinate (``azimavg.F:4012-4330``).  The horizontal grid reaches the
#: controller only as the values of ``xh``/``yh``/``xfref``/``yfref``, which
#: ``param.F:6649-6663`` forms in double precision as ``dble(dx)*(i-1)`` (less
#: ``0.5d0*dble(dx)*nx`` under ``iorigin = 2``) and ``param.F:6687,7393``
#: narrows to ``real``; and ``zref = 0`` selects ``kval = 1``
#: (``azimavg.F:4046-4058``) whatever height that level has.  So the
#: controller is admitted at every spacing whose faces and mass points survive
#: that narrowing unrounded (:func:`cm1_reference_axis`), on every column
#: whose lowest scalar level can be named, and that height is reported rather
#: than assumed.


def cm1_reference_axis(origin_m: float, spacing_m: float, n: int) -> tuple[Any, Any]:
    """``param.F:6649-6687,7393`` for one axis: ``(xfref(1..n+1), xh(1..n))``.

    Returns C-order float32 arrays.  Raises ``ValueError`` naming the first
    word binary32 cannot hold exactly, or a spacing that is not a CM1 ``real``
    (``input.F`` reads ``dx`` into one, so ``dble(dx)`` is that binary32 word
    and not the decimal a document wrote).
    """

    np = importlib.import_module("numpy")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("axis cell count must be a positive integer")
    spacing32 = np.float32(spacing_m)
    spacing = float(spacing32)
    if not math.isfinite(spacing) or spacing <= 0.0 or spacing != float(spacing_m):
        raise ValueError(
            f"the spacing {spacing_m!r} m is not a CM1 real: input.F reads it "
            f"into binary32 as {spacing!r} m, so CM1's faces are "
            f"{spacing!r}*(i-1) m and not the grid this document declares"
        )
    offset = 0.0 if float(origin_m) == 0.0 else 0.5 * spacing * n
    if float(origin_m) != -offset:
        raise ValueError(
            f"origin {origin_m!r} m is neither iorigin=1's 0 nor iorigin=2's "
            f"-0.5*dx*nx = {-offset!r} m (param.F:6649-6663)"
        )
    index = np.arange(n + 1, dtype=np.float64)
    faces = np.float64(spacing) * index - np.float64(offset)
    mass = np.float64(0.5) * (faces[1:] + faces[:-1])
    for label, wide in (("xfref", faces), ("xh", mass)):
        narrow = wide.astype(np.float32)
        rounded = np.flatnonzero(narrow.astype(np.float64) != wide)
        if rounded.size:
            k = int(rounded[0])
            raise ValueError(
                f"{label}({k + 1}) is {float(wide[k])!r} m in CM1's double "
                f"precision but {float(narrow[k])!r} m once narrowed to real, "
                f"a {abs(float(narrow[k]) - float(wide[k])):.3e} m rounding "
                f"that recurs at {rounded.size} of {wide.size} words"
            )
    return (
        np.ascontiguousarray(faces, dtype=np.float32),
        np.ascontiguousarray(mass, dtype=np.float32),
    )

#: ``src/bc.F:12-172``.  CM1's lateral boundary codes, as ``[boundaries]``
#: spells them, mapped to the sealed operator that implements that halo.
ADAPTIVE_MOVE_BRANCH_BY_BOUNDARY: Mapping[int, AdaptiveMoveBranch] = {
    1: "periodic",
    2: "radiative-open",
}

#: ``azimavg.F:4351-4366`` shifts these; this lane's names for them.
CARTESIAN_U_CARRIERS = ("u0", "ua", "u3d")
CARTESIAN_V_CARRIERS = ("v0", "va", "v3d")

SOURCE_ANCHORS: Mapping[str, str] = {
    "namelist_words": "README.namelist:1789-1797; src/input.F:200,337",
    "selection_initialization": "src/param.F:703-717",
    "controller_state_initialization": "src/cm1.F:348-349,388-390,2527-2529",
    "completed_step_domain_location": "src/cm1.F:3507-3513",
    "diagnosis_cadence": "src/cm1.F:3641-3651",
    "controller_cadence": "src/cm1.F:3669-3679",
    "pressure_smoothing": "src/azimavg.F:4012-4165",
    "pressure_centroid": "src/azimavg.F:4173-4330",
    "wind_frame_rebase": "src/azimavg.F:4338-4368",
    "adaptive_controller": "src/azimavg.F:4376-4528",
    "restart_payload": (
        "src/restart_write.F:377-401; src/restart_read.F:397-421"
    ),
}


# --------------------------------------------------------------- selection


@dataclass(frozen=True, slots=True)
class AdaptiveFrameBinding:
    """One admitted adaptive moving domain, bound before a device is opened."""

    branch: AdaptiveMoveBranch
    adapt_move_frq_s: float
    centerx_m: float
    centery_m: float
    origin_x_m: float
    origin_y_m: float
    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    stretch_z: int
    vertical_grid_sha256: str
    lowest_scalar_level_m: float

    @property
    def module(self) -> Any:
        """The sealed operator whose ``bc2d`` this boundary family selects."""

        return periodic if self.branch == "periodic" else radiative_open

    @property
    def selection(self) -> Any:
        """The sealed operator's selection object, on this run's column."""

        if self.branch == "periodic":
            default = periodic.DEFAULT_ADAPTIVE_MOVE_SELECTION
        else:
            default = radiative_open.DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION
        return dataclasses.replace(default, stretch_z=self.stretch_z)

    def reference_coordinates(self) -> tuple[Any, Any, Any, Any]:
        """``(xfref, xh, yfref, yh)`` exactly as ``param.F`` forms them."""

        xfref, xh = cm1_reference_axis(self.origin_x_m, self.dx_m, self.nx)
        yfref, yh = cm1_reference_axis(self.origin_y_m, self.dy_m, self.ny)
        return xfref, xh, yfref, yh


def _lateral_codes(config: Any) -> tuple[int, int, int, int]:
    b = config.boundaries
    return (b.west, b.east, b.south, b.north)


def require_adaptive_frame(config: Any) -> AdaptiveFrameBinding | None:
    """Bind ``closures.do_adapt_move`` to a sealed controller, or refuse.

    ``config.py`` has already refused the pair by key, the cadence by value,
    and the ``param.F:703-717`` couplings that make ``umove``/``vmove``
    meaningless beside it.  What is left is whether *this* grid and *these*
    boundaries are ones the sealed controller was validated on.  Each refusal
    below names the wrong result running anyway would produce; none of them
    is a taste question.
    """

    if not config.closures.do_adapt_move:
        return None

    from .config import Lane

    if config.run.lane is not Lane.CARTESIAN:
        refuse(
            code="CM1-CARTESIAN-ADAPT-MOVE-LANE",
            gate=CARTESIAN_GATE,
            reason=(
                f"lane {config.run.lane.value!r} has no completed-large-step "
                "driver to hang cm1.F:3669-3679 on, so closures.do_adapt_move "
                "would be recorded in the document, admitted by the grammar, "
                "and then never called -- the run would integrate a fixed "
                "frame at umove = vmove = 0 while its receipt named an "
                "adaptive one"
            ),
            remedy="declare run.lane = \"cartesian\", the lane that drives the controller",
            option="run.lane",
            value=config.run.lane.value,
        )

    codes = _lateral_codes(config)
    if len(set(codes)) != 1 or codes[0] not in ADAPTIVE_MOVE_BRANCH_BY_BOUNDARY:
        refuse(
            code="CM1-CARTESIAN-ADAPT-MOVE-BOUNDARIES",
            gate=CARTESIAN_GATE,
            reason=(
                f"west/east/south/north = {codes} is neither four-periodic (1) "
                "nor four radiative/open (2). getpsmth calls bc2d "
                "(bc.F:12-172) before each of its hundred smoothing passes and "
                "again before the zeta interpolation, and the wrapping halo and "
                "the edge-replicating halo reach different minima on the same "
                "pressure field -- most of all on the two nearly equal minima a "
                "splitting supercell presents. A mixed or rigid-wall set has "
                "neither halo, so the storm centre this run steered by would be "
                "an artefact of a boundary condition the domain does not have, "
                "and param.F:1906-1950 stops CM1 outright for a moving domain "
                "against a rigid wall"
            ),
            remedy=(
                "declare boundaries.west/east/south/north all 2 (the open "
                "branch this lane runs) or all 1 (the periodic branch)"
            ),
            option="boundaries",
            value=list(codes),
        )
    branch = ADAPTIVE_MOVE_BRANCH_BY_BOUNDARY[codes[0]]

    if config.grid.terrain:
        refuse(
            code="CM1-CARTESIAN-ADAPT-MOVE-TERRAIN",
            gate=CARTESIAN_GATE,
            reason=(
                "param.F:1890-1903 is CM1's own refusal: terrain with imove!=0 "
                "stops the model. The terrain field is fixed to the ground "
                "while an adaptive frame translates the grid over it, so after "
                "the first cadence every column would stand on the height it "
                "had before the domain moved and the run would integrate a "
                "mountain sliding under a storm at the frame velocity"
            ),
            remedy="set grid.terrain = false, or declare a fixed frame",
            option="grid.terrain",
            value=config.grid.terrain,
        )

    grid = config.grid
    origin_x, origin_y = _horizontal_origin(config)
    for name, origin, spacing, count in (
        ("grid.dx_m", origin_x, grid.dx_m, grid.nx),
        ("grid.dy_m", origin_y, grid.dy_m, grid.ny),
    ):
        try:
            cm1_reference_axis(origin, spacing, count)
        except ValueError as error:
            refuse(
                code="CM1-CARTESIAN-ADAPT-MOVE-GRID",
                gate=CARTESIAN_GATE,
                reason=(
                    f"{name} = {spacing!r} m over {count} cells does not give "
                    f"reference coordinates binary32 holds exactly: {error}. "
                    "getpcent (azimavg.F:4270-4320) weights xh/yh by the "
                    "smoothed pressure and snaps the centroid to the nearest "
                    "xfref/yfref face, and param.F:6687,7393 narrows those "
                    "coordinates from double precision to real. On this "
                    "spacing that narrowing rounds, so CM1's xh is no longer "
                    "the midpoint of its own faces and its faces are no longer "
                    "uniform; the storm centre, and every frame velocity "
                    "azimavg.F:4431-4435 derives from it, would be computed on "
                    "coordinates the controller's word-exact proof does not "
                    "cover, and the operator refuses them after the device is "
                    "open"
                ),
                remedy=(
                    "declare a spacing that is a CM1 real with a short binary "
                    "significand -- 25.0, 50.0, 100.0, 125.0, 250.0 or 500.0 m "
                    "all qualify at any practical grid size -- or declare a "
                    "fixed frame"
                ),
                option=name,
                value=spacing,
            )

    from .cartesian_door import require_grid_extent

    vertical = require_grid_extent(config).levels
    try:
        if vertical.stretch_z not in periodic.ADAPTIVE_MOVE_STRETCH_Z:
            raise ValueError(
                f"stretch_z={vertical.stretch_z} has no admitted lowest-level "
                "identification"
            )
        lowest_scalar_level_m = periodic.validate_adaptive_move_vertical_grid(vertical)
    except ValueError as error:
        refuse(
            code="CM1-CARTESIAN-ADAPT-MOVE-VERTICAL",
            gate=CARTESIAN_GATE,
            reason=(
                "the storm centre is diagnosed from the lowest scalar level "
                "(azimavg.F:4046-4075, zref = 0 selects kval = 1) and the "
                "receipt names that level's height. This column's lowest "
                f"scalar level cannot be named: {error}. The run would steer "
                "the whole domain by the pressure of a level its own receipt "
                "misstates"
            ),
            remedy=(
                "declare a flat grid.stretch_z = 0, 1, 3 or 4 column, or a "
                "fixed frame"
            ),
            option="grid.stretch_z",
            value=vertical.stretch_z,
        )

    closures = config.closures
    if closures.bbc != 1 or closures.sfcmodel != 0 or closures.isfcflx != 0:
        refuse(
            code="CM1-CARTESIAN-ADAPT-MOVE-SURFACE",
            gate=CARTESIAN_GATE,
            reason=(
                f"closures.bbc = {closures.bbc}, sfcmodel = "
                f"{closures.sfcmodel}, isfcflx = {closures.isfcflx} put "
                "turb.F:296-307's ground-relative wind ugr = ua + umove into "
                "the surface stress, and this runtime reads umove from the "
                "frozen RuntimeConfig it was built with. An adaptive frame "
                "changes umove every cadence, so from the first update onward "
                "the surface would exert drag computed against a ground-"
                "relative wind that is wrong by the whole accumulated frame "
                "velocity -- and near-ground vorticity is exactly what that "
                "drag exists to generate"
            ),
            remedy=(
                "declare the free-slip lower boundary (closures.bbc = 1, "
                "sfcmodel = 0, isfcflx = 0) beside closures.do_adapt_move"
            ),
            option="closures.bbc",
            value=closures.bbc,
        )

    return AdaptiveFrameBinding(
        branch=branch,
        adapt_move_frq_s=closures.adapt_move_frq,
        centerx_m=_cm1_center(origin_x, grid.nx, grid.dx_m),
        centery_m=_cm1_center(origin_y, grid.ny, grid.dy_m),
        origin_x_m=origin_x,
        origin_y_m=origin_y,
        nx=grid.nx,
        ny=grid.ny,
        nz=grid.nz,
        dx_m=float(grid.dx_m),
        dy_m=float(grid.dy_m),
        stretch_z=int(vertical.stretch_z),
        vertical_grid_sha256=periodic.adaptive_move_vertical_grid_identity(vertical),
        lowest_scalar_level_m=lowest_scalar_level_m,
    )


def _horizontal_origin(config: Any) -> tuple[float, float]:
    from .cartesian_door import horizontal_origin_m

    return horizontal_origin_m(config)


def _cm1_center(origin_m: float, n: int, spacing_m: float) -> float:
    """``param.F:7406-7408``: ``minx + 0.5*(maxx-minx)`` in CM1 ``real``."""

    np = importlib.import_module("numpy")
    faces, _ = cm1_reference_axis(origin_m, spacing_m, n)
    minimum = faces[0]
    maximum = faces[n]
    return float(
        np.add(
            minimum,
            np.multiply(np.float32(0.5), np.subtract(maximum, minimum, dtype=np.float32), dtype=np.float32),
            dtype=np.float32,
        )
    )


# ------------------------------------------------------------------ record


def cartesian_frame_mode_record(
    frame: Any, binding: AdaptiveFrameBinding | None
) -> dict[str, Any]:
    """Which frame this run integrated in, and by what rule it changed.

    The receipt carries this for the same reason it carries
    ``cartesian_vertical_operator_record``: so a reader can tell, from the
    document alone, whether the coordinates a frame's winds are relative to
    were a constant declared before the run or a model variable the storm
    moved -- and, if the latter, how often it moved and which of CM1's two
    lateral-halo branches found the storm.
    """

    if binding is None:
        return {
            "family": "fixed-frame",
            "do_adapt_move": False,
            "imove": frame.imove,
            "u_move_m_s": frame.u_move_m_s,
            "v_move_m_s": frame.v_move_m_s,
            "authority": "CM1 r21.1 base.F:2663-2669",
            "selected": (
                "the frame velocity is the constant this document declares and "
                "no operator changes it"
            ),
        }
    return {
        "family": "adaptive-moving-domain",
        "do_adapt_move": True,
        # param.F:711 raises imove for every do_adapt_move run even though
        # param.F:706-707 starts the velocity at zero, so the published
        # selector is CM1's and not this port's derived one.
        "imove": 1,
        "adapt_move_frq_s": binding.adapt_move_frq_s,
        "branch": binding.branch,
        "operator": (
            "cm1core.cuda_adaptive_move"
            if binding.branch == "periodic"
            else "cm1core.cuda_adaptive_move_open"
        ),
        "lateral_boundary_code": 1 if binding.branch == "periodic" else 2,
        "initial_u_move_m_s": 0.0,
        "initial_v_move_m_s": 0.0,
        "domain_center_m": [binding.centerx_m, binding.centery_m],
        "horizontal_spacing_m": [binding.dx_m, binding.dy_m],
        "stretch_z": binding.stretch_z,
        "vertical_grid_sha256": binding.vertical_grid_sha256,
        "lowest_scalar_level_m": binding.lowest_scalar_level_m,
        "diagnosis_level": (
            "lowest scalar level, azimavg.F:4046-4058 zref = 0 -> kval = 1, "
            f"which on this column is sigma(1) = {binding.lowest_scalar_level_m!r} m"
        ),
        "authority": "CM1 r21.1 azimavg.F:4376-4528; cm1.F:3641-3679",
        "source_anchors": dict(SOURCE_ANCHORS),
        "restart_schema": ADAPTIVE_MOVE_RESTART_LIFECYCLE_SCHEMA,
        "restart_source_order": list(periodic.ADAPTIVE_MOVE_RESTART_SOURCE_ORDER),
        "shifted_carriers": {
            "u": list(CARTESIAN_U_CARRIERS),
            "v": list(CARTESIAN_V_CARRIERS),
            "authority": SOURCE_ANCHORS["wind_frame_rebase"],
            "note": (
                "u0/v0 are CM1's horizontally uniform base wind, which this "
                "lane stores as one word per level; ug/vg are absent because "
                "solve1.F:590-631's large-scale pressure gradient is not in "
                "this lane and there is no geostrophic profile to rebase"
            ),
        },
        "selected": (
            "the frame velocity is a model variable: azimavg.F:4431-4435 "
            f"updates it every {binding.adapt_move_frq_s} s from the smoothed "
            "lowest-level pressure centroid, and every wind this run publishes "
            "is relative to the frame as it stood at that frame's model time"
        ),
    }


# ------------------------------------------------------------- device seam


@dataclass(frozen=True, slots=True)
class CartesianAdaptiveMoveLaunch:
    ready_event: Any
    state: periodic.DeviceAdaptiveMoveState
    delta: Any
    active: Any
    diagnostic_status: Any
    diagnosis_inputs_bound: bool
    carriers: Mapping[str, Any]


def _require_cartesian_carriers(
    cupy: Any, carriers: Mapping[str, Any], *, nz: int
) -> dict[str, Any]:
    """Admit this lane's carriers as CM1's ``change_uvmove`` rectangle.

    ``azimavg.F:4345-4346`` declares ``u0``/``ua``/``u3d`` over
    ``ib:ie+1,jb:je,kb:ke`` and ``v0``/``va``/``v3d`` over
    ``ib:ie,jb:je+1,kb:ke``.  This lane's padded owners are exactly those two
    rectangles, and its base wind is the same field compressed to one word per
    level.  Anything else is refused here rather than shifted: a carrier that
    is not the whole rectangle would leave the halo in the old frame while the
    interior moved to the new one, and the first lateral boundary call after
    the cadence would advect that discontinuity into the domain.
    """

    names = (*CARTESIAN_U_CARRIERS, *CARTESIAN_V_CARRIERS)
    if not isinstance(carriers, Mapping):
        raise TypeError("carriers must be a mapping")
    missing = [name for name in names if name not in carriers]
    if missing:
        raise ValueError(
            f"missing adaptive-frame carriers: {', '.join(missing)}"
        )
    if set(carriers) != set(names):
        raise ValueError(
            "adaptive-frame carrier inventory must be exactly "
            + ", ".join(names)
        )
    result: dict[str, Any] = {}
    for name in names:
        value = carriers[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be a C-order CuPy float32 array")
        result[name] = value
    if result["ua"].shape != result["u3d"].shape:
        raise ValueError("ua and u3d must share one padded rectangle")
    if result["va"].shape != result["v3d"].shape:
        raise ValueError("va and v3d must share one padded rectangle")
    u_shape, v_shape = result["ua"].shape, result["va"].shape
    if len(u_shape) != 3 or len(v_shape) != 3 or 0 in u_shape or 0 in v_shape:
        raise ValueError("ua and va carriers must be nonempty [k,j,i] arrays")
    if u_shape[0] != v_shape[0]:
        raise ValueError("ua and va must share the padded vertical extent")
    if u_shape[1] + 1 != v_shape[1] or u_shape[2] != v_shape[2] + 1:
        raise ValueError(
            "adaptive-frame carriers do not share the CM1 u/v stagger"
        )
    for name in ("u0", "v0"):
        value = result[name]
        if value.ndim != 1 or int(value.size) != nz:
            raise ValueError(
                f"{name} must be this lane's one-word-per-level base wind of "
                f"size {nz}"
            )
    periodic._require_nonoverlapping_device_arrays(
        result, family="adaptive-frame carrier"
    )
    return result


def launch_cartesian_adaptive_move_after_large_step(
    binding: AdaptiveFrameBinding,
    compiled: Any,
    state: periodic.DeviceAdaptiveMoveState,
    carriers: Mapping[str, Any],
    *,
    nstep: int,
    mtime_after: float,
    dbldt: float,
    centerx: float,
    centery: float,
    pressure_lowest: Any | None = None,
    xh: Any | None = None,
    yh: Any | None = None,
    xfref: Any | None = None,
    yfref: Any | None = None,
    dowriteout: bool = False,
    restarted: bool = False,
    vertical_grid_proof: Any,
    stream: Any | None = None,
) -> CartesianAdaptiveMoveLaunch:
    """Run one completed-large-step controller pass over this lane's carriers.

    Every kernel launched here is one the sealed operator compiled: the same
    source, the same PTX, the same ``__fadd_rn``/``__fmul_rn`` associations.
    What differs from ``launch_cuda_adaptive_move*_after_large_step`` is the
    inventory the ``add_delta`` kernel is swept over, which is this lane's and
    is admitted above.
    """

    if not isinstance(binding, AdaptiveFrameBinding):
        raise TypeError("binding must be an AdaptiveFrameBinding")
    if binding.branch == "periodic":
        periodic.require_adaptive_move_selection(binding.selection)
    else:
        radiative_open.require_adaptive_move_open_selection(binding.selection)
    if (
        periodic.validate_adaptive_move_vertical_grid(vertical_grid_proof)
        != binding.lowest_scalar_level_m
        or periodic.adaptive_move_vertical_grid_identity(vertical_grid_proof)
        != binding.vertical_grid_sha256
    ):
        raise ValueError(
            "the vertical coordinate handed to the controller is not the one "
            "this adaptive frame was bound to"
        )
    cupy = importlib.import_module("cupy")
    periodic._require_device_state(cupy, state)
    checked = _require_cartesian_carriers(cupy, carriers, nz=binding.nz)
    if isinstance(nstep, bool) or not isinstance(nstep, int) or nstep < 0:
        raise ValueError("nstep must be a nonnegative integer")
    if not all(type(value) is bool for value in (dowriteout, restarted)):
        raise TypeError("cadence and restart flags must be bool")
    frequency = cupy.float32(binding.adapt_move_frq_s)
    if not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
        raise ValueError("adapt_move_frq must be positive finite float32")
    centerx32, centery32 = cupy.float32(centerx), cupy.float32(centery)
    if not math.isfinite(float(centerx32)) or not math.isfinite(float(centery32)):
        raise ValueError("domain center coordinates must be finite float32")
    if (
        not math.isfinite(float(mtime_after))
        or not math.isfinite(float(dbldt))
        or float(dbldt) <= 0.0
    ):
        raise ValueError("mtime_after and positive dbldt must be finite")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    # cm1.F:3642-3651.  CM1 diagnoses the centre only on a step that is
    # already writing output; config.py has already refused a cadence that
    # never lands on one.
    diagnose = nstep >= 1 and dowriteout and not restarted
    if diagnose:
        if any(value is None for value in (pressure_lowest, xh, yh, xfref, yfref)):
            raise ValueError(
                "requested centre diagnosis lacks pressure/coordinate arrays"
            )
        with stream:
            ni, nj = periodic._require_device_center_arrays(
                cupy, pressure_lowest, xh, yh, xfref, yfref
            )
        if (ni, nj) != (binding.nx, binding.ny):
            raise ValueError(
                "the diagnosed pressure rectangle is not this run's domain"
            )
        periodic._require_nonoverlapping_device_arrays(
            {
                "state.f32": state.f32,
                "state.f64": state.f64,
                "state.i32": state.i32,
                **{f"carrier.{name}": value for name, value in checked.items()},
                "pressure_lowest": pressure_lowest,
                "xh": xh,
                "yh": yh,
                "xfref": xfref,
                "yfref": yfref,
            },
            family="cartesian adaptive-frame launch",
        )
    else:
        ni = nj = 1
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
            for _ in range(100):
                compiled.smooth_kernel(
                    blocks,
                    (threads, 1, 1),
                    (first, second, cupy.uint32(ni), cupy.uint32(nj)),
                    stream=stream,
                )
                first, second = second, first
            compiled.center_kernel(
                (1, 1, 1),
                (1, 1, 1),
                (
                    first,
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
        for which, names in ((0, CARTESIAN_U_CARRIERS), (1, CARTESIAN_V_CARRIERS)):
            for name in names:
                value = checked[name]
                blocks = ((int(value.size) + threads - 1) // threads, 1, 1)
                compiled.add_kernel(
                    blocks,
                    (threads, 1, 1),
                    (
                        value,
                        cupy.uint64(value.size),
                        delta,
                        active,
                        cupy.uint32(which),
                    ),
                    stream=stream,
                )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return CartesianAdaptiveMoveLaunch(
        event, state, delta, active, diagnostic_status, diagnose, checked
    )


def validate_cartesian_adaptive_move_completion(
    launch: CartesianAdaptiveMoveLaunch,
) -> None:
    """Synchronize and fail closed on a zero-weight centroid."""

    if not isinstance(launch, CartesianAdaptiveMoveLaunch):
        raise TypeError("launch must be a CartesianAdaptiveMoveLaunch")
    cupy = importlib.import_module("cupy")
    launch.ready_event.synchronize()
    if int(cupy.asnumpy(launch.diagnostic_status)[0]) != 0:
        raise ValueError(
            "CM1's adaptive pressure centroid has zero weight "
            "(azimavg.F:4283): the domain would be steered by a division no "
            "value supports"
        )


# ------------------------------------------------------------------ driver


class CartesianAdaptiveFrame:
    """The completed-large-step controller, attached to a live runtime.

    ``cm1.F:3507-3513`` advances ``domainlocx``/``domainlocy`` with the frame
    velocity as it stands *before* the controller runs, and ``cm1.F:3669-3679``
    updates the velocity afterwards.  The sealed commit kernel does both in
    that order, so this driver runs it on every completed large step and the
    persistent state -- not the frozen configuration -- is where the domain's
    ground position comes from.
    """

    def __init__(
        self, runtime: Any, binding: AdaptiveFrameBinding, *, restarted: bool = False
    ) -> None:
        cupy = runtime.cp
        self.runtime = runtime
        self.binding = binding
        self.cp = cupy
        self.stream = runtime.stream
        grid = runtime.config.grid
        # config.py refuses the pair (CM1-CFG-LSPGRAD-ADAPTIVE-MOVE); this is
        # the floor for a caller that builds the runtime another way.  The
        # lspgrad=1 profiles are formed once from the initial frame, and the
        # controller below rebases u0/v0 (azimavg.F:4440-4449) under them.
        if int(getattr(runtime.config, "lspgrad", 0)) != 0:
            raise ValueError(
                "an adaptive moving domain rebases u0/v0 mid-run, but the "
                "lspgrad=1 forcing profiles were formed once from the initial "
                "frame; the forcing and the carried winds would name different "
                "frames after the first update"
            )
        if (grid.nx, grid.ny, grid.nz) != (binding.nx, binding.ny, binding.nz):
            raise ValueError(
                "the adaptive frame was bound to a different grid than this runtime"
            )
        if (float(grid.dx_m), float(grid.dy_m)) != (binding.dx_m, binding.dy_m):
            raise ValueError(
                "the adaptive frame was bound to a different horizontal spacing "
                "than this runtime"
            )
        # The runtime's own column, not a sealed one: the receipt's
        # diagnosis height is this run's sigma(1).
        self.proof = grid.levels
        if (
            periodic.adaptive_move_vertical_grid_identity(self.proof)
            != binding.vertical_grid_sha256
            or periodic.validate_adaptive_move_vertical_grid(self.proof)
            != binding.lowest_scalar_level_m
        ):
            raise ValueError(
                "the adaptive frame was bound to a different vertical coordinate "
                "than this runtime"
            )
        self.compiled = (
            periodic.compile_cuda_adaptive_move()
            if binding.branch == "periodic"
            else radiative_open.compile_cuda_adaptive_move_open()
        )
        self.host_state = periodic.initialize_adaptive_move_state(
            ni=binding.nx,
            nj=binding.ny,
            centerx=binding.centerx_m,
            centery=binding.centery_m,
        )
        self.state = periodic.upload_adaptive_move_state(
            self.host_state, stream=self.stream
        )
        xfref, xh, yfref, yh = binding.reference_coordinates()
        with self.stream:
            self.xfref = cupy.asarray(xfref)
            self.yfref = cupy.asarray(yfref)
            self.xh = cupy.asarray(xh)
            self.yh = cupy.asarray(yh)
            self.pressure_lowest = cupy.empty(
                (grid.ny, grid.nx), dtype=cupy.float32
            )
        # cm1.F:3646.  The first large step a resumed run takes repeats no
        # centre diagnosis: the centre in the restored state is the one CM1
        # carries forward.  cm1.F:2663 raises ``restarted`` when the restart is
        # read and cm1.F:4404 clears it at the end of that first iteration, so
        # the flag belongs to the step *after* the restored one.
        self._restarted_step = (
            int(runtime.clock.completed_steps) + 1 if restarted else None
        )
        self._snapshot_step = -1

    # ---- carriers

    def carriers(self) -> dict[str, Any]:
        d = self.runtime.dynamics
        return {
            "u0": d.base_profiles["u0"],
            "ua": d.arrays["ua"],
            "u3d": d.arrays["u3d"],
            "v0": d.base_profiles["v0"],
            "va": d.arrays["va"],
            "v3d": d.arrays["v3d"],
        }

    def _copy_lowest_pressure(self) -> None:
        """``azimavg.F:4071-4075``: ``dum1(i,j,1) = pp3d(i,j,kval)``, kval=1."""

        d = self.runtime.dynamics
        pp3d = d.arrays["pp3d"]
        # k = 0 is the one ghost level below ground on every admitted column,
        # so padded index 1 is CM1's kval = 1 whatever height sigma(1) has.
        if pp3d.ndim != 3 or int(pp3d.shape[0]) != self.binding.nz + 2:
            raise ValueError(
                f"pp3d has {pp3d.shape[0] if pp3d.ndim else 0} levels, not "
                f"nz + 2 = {self.binding.nz + 2}; padded index 1 would not be "
                "the lowest scalar level getpsmth diagnoses"
            )
        halo = (pp3d.shape[1] - self.binding.ny) // 2
        level = pp3d[1, halo : halo + self.binding.ny, halo : halo + self.binding.nx]
        self.cp.copyto(self.pressure_lowest, level)

    # ---- the seam

    def after_large_step(self) -> None:
        runtime = self.runtime
        nstep = int(runtime.clock.completed_steps)
        dowriteout = nstep in getattr(runtime, "history_frame_steps", frozenset())
        restarted = self._restarted_step == nstep
        diagnose = nstep >= 1 and dowriteout and not restarted
        if diagnose:
            self._copy_lowest_pressure()
        launch = launch_cartesian_adaptive_move_after_large_step(
            self.binding,
            self.compiled,
            self.state,
            self.carriers(),
            nstep=nstep,
            mtime_after=float(runtime.clock.model_seconds),
            dbldt=float(runtime.config.dt_s),
            centerx=self.binding.centerx_m,
            centery=self.binding.centery_m,
            pressure_lowest=self.pressure_lowest if diagnose else None,
            xh=self.xh if diagnose else None,
            yh=self.yh if diagnose else None,
            xfref=self.xfref if diagnose else None,
            yfref=self.yfref if diagnose else None,
            dowriteout=dowriteout,
            restarted=restarted,
            vertical_grid_proof=self.proof,
            stream=self.stream,
        )
        if diagnose:
            validate_cartesian_adaptive_move_completion(launch)
        self._snapshot_step = -1

    # ---- readback

    def snapshot(self) -> periodic.AdaptiveMoveState:
        """The controller state at this completed step, read back once."""

        step = int(self.runtime.clock.completed_steps)
        if self._snapshot_step != step:
            self.stream.synchronize()
            self.host_state = periodic.download_adaptive_move_state(self.state)
            self._snapshot_step = step
        return self.host_state

    # ---- the RuntimeClock frame source

    @property
    def u_move_m_s(self) -> float:
        return self.snapshot().umove

    @property
    def v_move_m_s(self) -> float:
        return self.snapshot().vmove

    @property
    def domainlocx_m(self) -> float:
        return self.snapshot().domainlocx

    @property
    def domainlocy_m(self) -> float:
        return self.snapshot().domainlocy

    def restart_scalars(self) -> dict[str, tuple[str, str]]:
        """CM1's fifteen controller words, in ``restart_write.F``'s own order.

        ``restart_write.F:377-401`` writes exactly these fifteen and
        ``restart_read.F:397-421`` reads them back; the order here is
        ``ADAPTIVE_MOVE_RESTART_SOURCE_ORDER``, which is that source order.
        They are carried as raw f64 bits for the same reason the rest of this
        table is: a decimal round trip of ``umove`` would restart the run in a
        frame one ulp from the one its carriers were shifted into.
        """

        from .cartesian_restart import _f64_bits

        state = self.snapshot()
        return {
            f"{ADAPTIVE_MOVE_RESTART_PREFIX}{name}": (
                "f64-bits", _f64_bits(float(getattr(state, name)))
            )
            for name in periodic.ADAPTIVE_MOVE_RESTART_SOURCE_ORDER
        }

    def restore(self, restored: Any) -> None:
        """Continue a persisted controller: ``restart_read.F:397-421``."""

        values: dict[str, Any] = {}
        for name in periodic.ADAPTIVE_MOVE_RESTART_SOURCE_ORDER:
            value = float(restored.f64(f"{ADAPTIVE_MOVE_RESTART_PREFIX}{name}"))
            if name in periodic.I32_STATE_FIELDS:
                if not float(value).is_integer():
                    raise ValueError(
                        f"restored adapt_move.{name} is {value!r}, not an "
                        "int32 counter"
                    )
                values[name] = int(value)
            else:
                values[name] = value
        state = periodic.AdaptiveMoveState(**values)
        periodic._validate_state(state)
        self.host_state = state
        self.state = periodic.upload_adaptive_move_state(state, stream=self.stream)
        # cm1.F:2663/4404: the flag is true for the one iteration that follows
        # the restart read, which completes step restored.step + 1.
        self._restarted_step = int(restored.step) + 1
        self._snapshot_step = -1

    def record(self) -> dict[str, Any]:
        """What this controller did, for the receipt."""

        state = self.snapshot()
        return {
            "schema": CARTESIAN_ADAPTIVE_FRAME_SCHEMA,
            "branch": self.binding.branch,
            "adapt_move_frq_s": self.binding.adapt_move_frq_s,
            "kernel_source_sha256": (
                periodic.CUDA_ADAPTIVE_MOVE_SOURCE_SHA256
                if self.binding.branch == "periodic"
                else radiative_open.CUDA_ADAPTIVE_MOVE_OPEN_SOURCE_SHA256
            ),
            "target_arch": self.compiled.target_arch,
            "completed_updates": int(state.nwritemv) - 1,
            "umove": state.umove,
            "vmove": state.vmove,
            "xcenter": state.xcenter,
            "ycenter": state.ycenter,
            "icenter": state.icenter,
            "jcenter": state.jcenter,
            "domainlocx": state.domainlocx,
            "domainlocy": state.domainlocy,
            "adaptmovetim": state.adaptmovetim,
            "restart_contract": dict(adaptive_move_restart_lifecycle_contract()),
            "claim_limit": (
                periodic.CM1_ADAPTIVE_MOVE_CLAIM_LIMIT
                if self.binding.branch == "periodic"
                else radiative_open.CM1_ADAPTIVE_MOVE_OPEN_CLAIM_LIMIT
            ),
        }


__all__ = (
    "ADAPTIVE_MOVE_BRANCH_BY_BOUNDARY",
    "ADAPTIVE_MOVE_RESTART_PREFIX",
    "CARTESIAN_ADAPTIVE_FRAME_SCHEMA",
    "CARTESIAN_U_CARRIERS",
    "CARTESIAN_V_CARRIERS",
    "SOURCE_ANCHORS",
    "cm1_reference_axis",
    "AdaptiveFrameBinding",
    "CartesianAdaptiveFrame",
    "CartesianAdaptiveMoveLaunch",
    "cartesian_frame_mode_record",
    "launch_cartesian_adaptive_move_after_large_step",
    "require_adaptive_frame",
    "validate_cartesian_adaptive_move_completion",
)
