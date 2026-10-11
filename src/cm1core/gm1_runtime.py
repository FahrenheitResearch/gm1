"""Configuration and source-ordered schedule for a fresh Cartesian GM1 run.

This is the runtime contract, independent of historical validation fixtures.
Numerical implementations live in the CUDA backend; this module neither
imports a GPU runtime nor allocates an atmospheric array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import functools
import math
import struct
from typing import Any, Protocol

from .cuda_wsm6_pdefq import (
    DEFAULT_WSM6_PDEFQ_LAYOUT,
    require_wsm6_pdefq_layout,
    wsm6_pdefq_launch_plan,
    wsm6_pdefq_workspace_bytes,
)
from .component_validation import component_validation
from .dense_ice import (
    DEFAULT_IHAIL,
    DenseIceAbsent,
    DenseIceSelection,
    default_ihail,
    dense_ice_selection,
)
from .grid_geometry import CartesianGridGeometry
from .moisture_species import (
    DEFAULT_COMPONENT_ID,
    MoistureSpeciesProfile,
    moisture_species_profile,
)

# The one definition of CM1 base.F:2663-2669's admitted frame speed, in m/s.
# It is the Rust MAX_FRAME_SPEED_M_PER_S in
# rust/cm1-state/src/idealized_initialization.rs, and test_gm1_runtime holds
# the two in step by reading the Rust source, so the number cannot drift
# between the refusal the configuration raises and the one the library would.
MAX_FRAME_SPEED_M_S = 150.0

# CM1 r21.1's running domain-location integral, cm1.F:266,348-349,3512-3513.
# The names and units are CM1's own: writeout.F:9634-9648 writes them as
# "domainlocx"/"domainlocy" described as "x location of (center of) domain
# (m)", and domaindiag.F:464-483 gives the same pair varunit "m".  The Rust
# owner of the format and of the restart round trip is
# rust/cm1-state/src/domain_location.rs.
DOMAIN_LOCATION_X_NAME = "domainlocx"
DOMAIN_LOCATION_Y_NAME = "domainlocy"
MOVE_SPEED_X_NAME = "umove"
MOVE_SPEED_Y_NAME = "vmove"
DOMAIN_LOCATION_UNITS = "m"
MOVE_SPEED_UNITS = "m/s"
DOMAIN_LOCATION_DESCRIPTIONS = {
    DOMAIN_LOCATION_X_NAME: "x location of (center of) domain",
    DOMAIN_LOCATION_Y_NAME: "y location of (center of) domain",
}
# cm1.F:348-349.
DOMAIN_LOCATION_INITIAL_M = 0.0
# cm1.F:3510-3513: advanced once per completed large step, immediately after
# mtime is stepped forward and before that step's output or restart is
# written.
DOMAIN_LOCATION_CADENCE = "post-large-step-after-model-time"
DOMAIN_LOCATION_AUTHORITY = "CM1 r21.1 cm1.F:348-349,3512-3513 (domainlocx/domainlocy)"
# Past this many steps the drift-free argument in domain_location.rs stops
# holding even for a binary32 frame speed and a power-of-two dt.
DOMAIN_LOCATION_EXACT_STEP_LIMIT = 1 << 29


def cm1_real(value: float) -> float:
    """The binary32 word CM1 would carry for ``value``.

    ``umove``/``vmove`` are CM1 ``real`` (base.F:2661-2674), and GM1's Rust
    frame is binary32 for the same reason, so this is the value that was
    actually subtracted from the base-state winds.
    """

    return struct.unpack("<f", struct.pack("<f", value))[0]

# ---------------------------------------------------------------------------
# CM1 r21.1's own timestep admissibility, transcribed from ``misclibs.F``.
#
# The large step itself is not a constant anywhere in CM1: ``solve2.F:145``
# declares ``real, intent(inout) :: dt`` and every stage multiplies by the
# word it is handed.  What CM1 does fix is what a given ``dt`` implies.
#
# CM1 r19 and later never accept ``nsound`` from the namelist: ``dtsmall``
# (misclibs.F:818-915) derives it from the large step and the grid on every
# change of ``dt``, and ``cm1.F:3054`` calls it once more before the first
# step.  For ``psolver=3`` (misclibs.F:857-868) the acoustic solver is
# vertically implicit, so only the horizontal spacings bound the small step,
# and misclibs.F:867 bounds it at 0.60 of the time a 360 m/s sound wave needs
# to cross one cell.  Everything below is that single criterion, expressed so
# both the configuration door and the runtime contract refuse the same value.
CM1_PSOLVER3_ACOUSTIC_SPEED_M_S = 360.0
CM1_PSOLVER3_ACOUSTIC_COURANT = 0.60
# misclibs.F:883: ``nsound``, the nearest integer to ``dbldt/dtsm``, never
# drops below four.
CM1_MIN_ACOUSTIC_SUBSTEPS = 4
# misclibs.F:835 declares ``max_nsound = 192``; misclibs.F:887-911 prints
# "Time step (dtl) needs to be smaller" and calls ``stopcm1`` above it
# whenever ``adapt_dt`` is off, which is the only branch this port admits.
CM1_MAX_ACOUSTIC_SUBSTEPS = 192

# ---------------------------------------------------------------------------
# CM1 r21.1's advective Courant number, transcribed from ``misclibs.F``.
#
# ``calccflquick`` (misclibs.F:2726-2814) and the statistics-package
# ``calccfl`` (misclibs.F:2584-2718) form the same norm: misclibs.F:2747-2749
# builds ``dtdx=0.5*dt*rdx`` and its two partners, and misclibs.F:2763-2765
# takes ``sqrt`` of the sum of the three squared face-averaged components.
# Because the face average of the two faces is twice the cell value, that is
# exactly ``dt*sqrt((u/dx)^2+(v/dy)^2+(w/dz)^2)``.
#
# The stop is misclibs.F:2714, which sets ``stopit`` once ``fmax`` reaches 1.50, at the end of
# ``calccfl``.  In an ``adapt_dt=0`` run -- the only kind this port admits,
# because ``dynamics.adaptive_timestep=true`` is refused at the door -- that is
# the one advective-Courant stop a serial build reaches: statpack.F:469 calls
# ``calccfl`` at statistics cadence whenever ``stat_cfl=1``, which is CM1's own
# shipped default (run/namelist.input:320).  solve2.F:1254-1256 measures the
# same norm with ``calccflquick`` on the final RK substep of every large step,
# but the stop it feeds, solve2.F:2082, is inside ``#ifdef MPI`` and a serial
# build never compiles it.  Once ``stopit`` is set, cm1.F:3517-3520 forces
# ``dostat`` and ``dowriteout`` so the statistics and the history frame for the
# tripping state are written, and only then does cm1.F:4385-4392 print
# "Courant number has exceeded 1.5 / Stopping model" and stop.
#
# cm1.F:3020-3023 is NOT this authority and must not be cited as it.  It sits
# inside ``get_adapt_dt: if( adapt_dt.eq.1 )`` (cm1.F:2947) and
# ``if( .not. restarted )`` (cm1.F:3013); its ``stopit=.true.`` is inside
# ``#ifdef MPI`` (cm1.F:3021/3025); and cm1.F:3035 clears ``stopit`` twelve
# lines later, before the first reader at cm1.F:3517.  It stops nothing, in any
# build, in any configuration.
#
# misclibs.F:617 is the value ``getnewdt`` steers an adaptive step toward.
CM1_COURANT_STOP = 1.5
CM1_COURANT_TARGET = 1.0

# ---------------------------------------------------------------------------
# CM1 r21.1's rotation selectors, transcribed from ``solve2.F`` and ``param.F``.
#
# ``solve2.F:633-653`` is the Cartesian f-plane branch and ``solve2.F:655-673``
# the beta plane; ``param.F:8945-8969`` builds ``f2d`` for both.  Two facts
# about that source decide what this contract admits.
#
# First, ``param.F:8952`` takes ``phi = asin( fcor/(2.0*omega) )``.  Outside
# that domain gfortran returns NaN, ``beta0`` is NaN, and every ``f2d`` word is
# NaN, so the whole momentum field goes non-finite on the first beta-plane
# stage.  ``omega`` is ``constants.F:87``, ``2*pi/86400``.
#
# Second, ``solve2.F:645,647`` add ``vmove``/``umove`` inside the f-plane term
# and ``solve2.F:667-672`` do not add them in the beta-plane term.  On the f
# plane that add-back is what makes icor=1 correct in a translating frame: the
# Coriolis acceleration is a function of the ground-relative wind, and
# ``base.F:2667`` has already subtracted the frame velocity from every carried
# wind.  The beta branch omits it, so a translating beta-plane run integrates a
# Coriolis acceleration short by ``fcor*vmove`` and ``fcor*umove`` everywhere.
CM1_OMEGA_RAD_S = 2.0 * math.pi / 86400.0
CM1_MAX_ABS_FCOR_S_INV = 2.0 * CM1_OMEGA_RAD_S
CM1_ROTATION_AUTHORITY = (
    "CM1 r21.1 src/solve2.F:631-720 (icor=1 branches), "
    "src/param.F:8945-8969 (f2d), src/constants.F:87 (omega)"
)
# ``lspgrad=1`` (README.namelist:531-538) is the other half of an f-plane
# icor=1 run: ``solve1.F:590-612`` subtracts fcor x (V0 + Vmove) from the held
# u/v tendency, so solve2.F:639-653's fcor x (V + Vmove) acts on V - V0 only
# and the base-state wind feels no net Coriolis force.
CM1_LSPGRAD_AUTHORITY = (
    "CM1 r21.1 src/solve1.F:536-612 (lspgrad=1 base-state geostrophic "
    "pressure gradient), README.namelist:531-538, base.F:2661-2674 "
    "(frame-relative u0/v0)"
)

# ---------------------------------------------------------------------------
# ``alph`` and ``kdiv``, the two real words the psolver=3 acoustic step reads.
#
# ``README.namelist:756-759``: "Off-centering coefficient for vertically
# implicit acoustic solver.  A value of 0.5 is centered-in-time.  Slight
# forward-in-time bias is recommended.  Default value is 0.60."  ``sound.F:78-82``
# sets ``beta = 1.0 - alph`` and ``asq = alph*alph`` from it.  All 23 of CM1's
# shipped decks set ``alph = 0.60``; GM1 ran 0.5 -- exactly centred, no damping
# of the vertically propagating acoustic mode -- until this contract carried it.
#
# ``kdiv`` is the divergence damper of ``README.namelist:751-754``, "only used
# when psolver=2,3"; every shipped deck sets 0.10, which is what GM1 already ran.
#
# ``csound`` is deliberately NOT a word here: ``misclibs.F:873-879`` and
# ``sounde.F``/``soundcb.F`` are its only readers, and all of them are
# psolver=2/6 code.  A psolver=3 run that accepted a csound key would carry a
# number that changes nothing, which is worse than not offering it.
CM1_DEFAULT_ALPH = 0.60
CM1_DEFAULT_KDIV = 0.10
CM1_ACOUSTIC_OFF_CENTERING_AUTHORITY = (
    "CM1 r21.1 README.namelist:756-759 (alph default 0.60), src/sound.F:78-82 "
    "(beta=1-alph, asq=alph*alph), README.namelist:751-754 (kdiv)"
)

# Where GM1 measures it is GM1's own choice, and is stated rather than
# attributed: CM1 acts on the number at statistics cadence, GM1 acts on it once
# before the first large step of an integration and again at every history
# frame.  The norm and the 1.50 are CM1's; the cadence is strictly earlier and
# no less frequent than CM1's, so no state CM1 would have abandoned is
# integrated further here.
CM1_COURANT_STOP_AUTHORITY = (
    "CM1 r21.1 src/misclibs.F:2726-2814 (calccflquick) and "
    "src/misclibs.F:2584-2718 (calccfl) form the norm; src/misclibs.F:2714 is "
    "the stop; src/statpack.F:469 reaches it at statistics cadence with "
    "stat_cfl=1; src/cm1.F:3517-3520 then writes the tripping frame and "
    "src/cm1.F:4385-4392 stops the model"
)

# ---------------------------------------------------------------------------
# The default-REAL word a large step is actually applied as.
#
# ``solve2.F:145`` declares ``real, intent(inout) :: dt`` and every stage of
# CM1 multiplies by that single-precision word; the GM1 kernels do the same.
# A step whose binary32 rounding is not a positive *normal* number cannot be
# applied: it flushes to zero or to a subnormal inside a kernel compiled with
# ``--ftz=false`` but stored as one float32 register, while the host clock and
# every float64 derivation advance by the requested value.
CM1_DT_F32_MIN_NORMAL_BITS = 0x00800000
CM1_DT_F32_INFINITY_BITS = 0x7F800000
CM1_MIN_NORMAL_DT_S = struct.unpack("<f", struct.pack("<I", CM1_DT_F32_MIN_NORMAL_BITS))[0]


def positive_normal_f32_bits(value: object) -> int | None:
    """Return the binary32 word this step will be applied as, or ``None``.

    ``None`` means the value cannot be carried as a CM1 default-REAL step at
    all: it is not a number, is not finite, is not positive, or rounds to zero
    or to a subnormal.  Callers turn that into their own refusal, so the
    predicate is stated exactly once.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    bits = struct.unpack("<I", struct.pack("<f", number))[0]
    if not (CM1_DT_F32_MIN_NORMAL_BITS <= bits < CM1_DT_F32_INFINITY_BITS):
        return None
    return bits


def cm1_acoustic_small_step_limit_s(*, dx_m: float, dy_m: float) -> float:
    """``misclibs.F:867``: the largest admissible ``dtsm`` for psolver=3.

    This is the three-dimensional branch of ``misclibs.F:857-868``,
    ``dtsm``: 0.60 times the smaller of ``min_dx`` and ``min_dy``, over 360.0.  CM1 evaluates it in default
    REAL; this evaluates it in float64, which is a few parts in 1e8 away and
    is compared with ``>`` exactly as ``misclibs.F:885`` does.
    """

    for name, value in (("dx_m", dx_m), ("dy_m", dy_m)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return (
        CM1_PSOLVER3_ACOUSTIC_COURANT
        * min(float(dx_m), float(dy_m))
        / CM1_PSOLVER3_ACOUSTIC_SPEED_M_S
    )


def _nint(value: float) -> int:
    """Fortran ``nint``: round half away from zero, not to even."""

    return int(math.floor(value + 0.5)) if value >= 0 else -int(math.floor(-value + 0.5))


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def cm1_diagnostic_acoustic_substeps(*, dt_s: float, dx_m: float, dy_m: float) -> int:
    """``misclibs.F:883-885``: the ``nsound`` CM1 itself would choose.

    The nearest integer to ``dbldt/dtsm``, at least 4, rounded up to an even number,
    then incremented by two once more if the resulting small step still
    exceeds ``dtsm``.  The result is what a psolver=3 CM1 run with this ``dt``
    and this grid would use, and is the floor this port refuses below.
    """

    dt = _positive_finite("dt_s", dt_s)
    limit = cm1_acoustic_small_step_limit_s(dx_m=dx_m, dy_m=dy_m)
    nsound = max(_nint(dt / limit), CM1_MIN_ACOUSTIC_SUBSTEPS)
    if nsound % 2:
        nsound += 1
    if dt / nsound > limit:
        nsound += 2
    return nsound


# One ulp-walk bound and one search width, so neither loop below can spin.
_MAX_LARGE_STEP_ULP_WALK = 64
_CLOCK_ALIGNED_SEARCH_WIDTH = 4096


def cm1_max_large_step_s(
    *, dx_m: float, dy_m: float, acoustic_substeps: int
) -> float:
    """The largest ``dt`` a fixed substep count can carry on this grid.

    In exact arithmetic the answer is ``nsound*dtsm``, but the door's own test
    is ``dt/nsound > dtsm`` (misclibs.F:885) and float64 rounds
    ``(nsound*dtsm)/nsound`` strictly above ``dtsm`` for many (spacing, count)
    pairs.  A refusal that names such a product as the maximum admissible step
    is a fixed point the same door rejects: the operator sets the printed value
    and gets back an identical refusal printing the same number.  The product is
    therefore walked down by one ulp at a time until it satisfies the
    inequality the refusal is about to be checked against, so the value printed
    is always admitted.
    """

    if type(acoustic_substeps) is not int or acoustic_substeps < 1:
        raise ValueError("acoustic_substeps must be a positive integer")
    limit = cm1_acoustic_small_step_limit_s(dx_m=dx_m, dy_m=dy_m)
    candidate = acoustic_substeps * limit
    # Both the product and the quotient are correctly rounded, so the walk is
    # monotone and terminates within a couple of ulps; the bound only stops a
    # future change to the criterion from spinning here.
    for _ in range(_MAX_LARGE_STEP_ULP_WALK):
        if candidate / acoustic_substeps <= limit:
            return candidate
        candidate = math.nextafter(candidate, 0.0)
    raise AssertionError(
        "the acoustic small-step criterion and its inverse disagree by more "
        f"than {_MAX_LARGE_STEP_ULP_WALK} ulps"
    )


def cm1_largest_clock_aligned_large_step_s(
    *,
    dx_m: float,
    dy_m: float,
    acoustic_substeps: int,
    event_seconds: tuple[float, ...],
) -> float | None:
    """The largest admissible ``dt`` the declared event clock also admits.

    ``cm1_max_large_step_s`` answers only CM1's acoustic question.  A fixed
    clock additionally requires every declared event -- the run duration, the
    history interval, the restart interval -- to fall on an exact large-step
    boundary, so the bare acoustic maximum is usually not a step an operator
    can actually declare.  This searches ``base/n`` for the smallest ``n`` that
    is inside the acoustic bound and divides every event exactly, which is the
    step a remedy can name and have work on the first try.

    ``None`` means no such step exists for this grid and these events, and the
    caller should name the acoustic maximum and the clock requirement
    separately rather than print a value that would be refused.
    """

    maximum = cm1_max_large_step_s(
        dx_m=dx_m, dy_m=dy_m, acoustic_substeps=acoustic_substeps
    )
    events = tuple(
        float(value)
        for value in event_seconds
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )
    if not events:
        return maximum if positive_normal_f32_bits(maximum) is not None else None
    base = min(events)
    start = max(1, math.ceil(base / maximum))
    for n in range(start, start + _CLOCK_ALIGNED_SEARCH_WIDTH):
        dt = base / n
        if dt > maximum or positive_normal_f32_bits(dt) is None:
            continue
        if all(
            math.isfinite(event / dt)
            and (event / dt) >= 1.0
            and (event / dt).is_integer()
            for event in events
        ):
            return dt
    return None


def acoustic_small_step_defect(
    *, dt_s: float, dx_m: float, dy_m: float, acoustic_substeps: int
) -> str | None:
    """Name the instability this (dt, grid, substep count) would produce.

    Returns ``None`` when ``misclibs.F:818-915`` would admit the triple, and
    otherwise one sentence naming the concrete wrong result, for a caller to
    raise or to refuse by key and value.
    """

    if type(acoustic_substeps) is not int or acoustic_substeps < 1:
        raise ValueError("acoustic_substeps must be a positive integer")
    dt = _positive_finite("dt_s", dt_s)
    limit = cm1_acoustic_small_step_limit_s(dx_m=dx_m, dy_m=dy_m)
    if acoustic_substeps > CM1_MAX_ACOUSTIC_SUBSTEPS:
        return (
            f"acoustic_substeps={acoustic_substeps} exceeds CM1's own "
            f"max_nsound={CM1_MAX_ACOUSTIC_SUBSTEPS} (misclibs.F:835), where "
            "misclibs.F:887-911 stops the run rather than run that many small "
            "steps"
        )
    small_step = dt / acoustic_substeps
    if small_step > limit:
        needed = cm1_diagnostic_acoustic_substeps(dt_s=dt, dx_m=dx_m, dy_m=dy_m)
        maximum = cm1_max_large_step_s(
            dx_m=dx_m, dy_m=dy_m, acoustic_substeps=acoustic_substeps
        )
        if needed <= CM1_MAX_ACOUSTIC_SUBSTEPS:
            tail = (
                f"CM1's own dtsmall derives nsound={needed} for this step and "
                "this grid; use at least that many acoustic substeps, or a "
                f"large step of at most {maximum!r} s"
            )
        else:
            # misclibs.F:887-911 with adapt_dt=0: CM1 has no substep route
            # above max_nsound, so naming one here would send the operator to
            # a count this same door refuses.
            tail = (
                f"CM1's own dtsmall would need nsound={needed} for this step, "
                f"above max_nsound={CM1_MAX_ACOUSTIC_SUBSTEPS}, so no substep "
                "count carries it and CM1's own answer at misclibs.F:887-911 "
                "is 'Time step (dtl) needs to be smaller'; use a large step of "
                f"at most {maximum!r} s"
            )
        return (
            f"the acoustic small step dt/nsound = {small_step!r} s exceeds the "
            f"{limit!r} s CM1 derives for this grid "
            f"(misclibs.F:867, 0.60*min(dx,dy)/"
            f"{CM1_PSOLVER3_ACOUSTIC_SPEED_M_S} m/s); psolver=3 is vertically "
            "implicit but keeps the horizontal sound-wave terms explicit "
            "inside the small step, so this cadence runs them above the "
            f"{CM1_PSOLVER3_ACOUSTIC_COURANT} of a cell-crossing time that "
            "CM1 holds itself to against a "
            f"{CM1_PSOLVER3_ACOUSTIC_SPEED_M_S} m/s reference sound speed. "
            "That number is CM1's margin, not the instability: the explicit "
            "horizontal terms amplify grid-scale pressure and divergence every "
            "small step once the acoustic Courant number formed with the true "
            "sound speed and the advecting wind reaches 1, and the margin is "
            "what keeps a warm, moist, fast-flowing column below it. CM1 never "
            "integrates in that band at all -- misclibs.F:883-885 derives "
            "nsound from dtsm instead of accepting one -- so this cadence has "
            f"no CM1 branch to be compared against. {tail}"
        )
    return None


def cm1_advective_courant(
    *,
    dt_s: float,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    u_m_s: float,
    v_m_s: float,
    w_m_s: float,
) -> float:
    """``misclibs.F:2747-2749,2763-2765``: CM1's advective Courant number.

    CM1 forms ``0.5*dt*rdx`` and multiplies it by ``ua(i)+ua(i+1)``, the sum
    of the two faces of the cell, so the factor of two cancels and the cell
    contribution is ``u*dt/dx``.  The two horizontal squares are summed before
    the vertical square is added, which is CM1's own grouping at
    misclibs.F:2763-2765 and is kept here.  Winds are frame-relative
    (``base.F:2667`` has already subtracted ``umove``/``vmove``), because
    ``u3d``/``v3d`` are what CM1 hands ``calccflquick``.
    """

    dt = _positive_finite("dt_s", dt_s)
    dx = _positive_finite("dx_m", dx_m)
    dy = _positive_finite("dy_m", dy_m)
    dz = _positive_finite("dz_m", dz_m)
    for name, value in (("u_m_s", u_m_s), ("v_m_s", v_m_s), ("w_m_s", w_m_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    horizontal = (float(u_m_s) * dt / dx) ** 2 + (float(v_m_s) * dt / dy) ** 2
    return math.sqrt(horizontal + (float(w_m_s) * dt / dz) ** 2)


def cm1_courant_limited_speed_m_s(
    *,
    dt_s: float,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    courant: float = CM1_COURANT_STOP,
) -> float:
    """Invert ``cm1_advective_courant`` over every wind direction.

    The Courant number a wind of a given *magnitude* produces depends on which
    way it points: ``sqrt(sum (V_i/d_i)^2)`` is largest when the whole vector
    lies along the axis with the smallest spacing.  The value returned is that
    worst case, ``courant*min(dx,dy,dz)/dt``, so a wind slower than it cannot
    reach ``courant`` however it is oriented.  It is a published capacity, not
    a refusal: the wind is not known until the state exists.
    """

    dt = _positive_finite("dt_s", dt_s)
    dx = _positive_finite("dx_m", dx_m)
    dy = _positive_finite("dy_m", dy_m)
    dz = _positive_finite("dz_m", dz_m)
    limit = _positive_finite("courant", courant)
    return limit * min(dx, dy, dz) / dt


def advective_courant_defect(
    *,
    courant: float,
    dt_s: float,
    grid_note: str = "",
) -> str | None:
    """Name the instability a measured Courant number has already reached.

    ``misclibs.F:2714`` sets ``stopit`` at 1.50 from ``calccfl``, which
    ``statpack.F:469`` calls at statistics cadence whenever ``stat_cfl=1``;
    ``cm1.F:3517-3520`` then writes the tripping state and ``cm1.F:4385-4392``
    abandons the run.  This returns that sentence, or ``None`` when the state
    is still inside CM1's own admitted range.

    ``cm1.F:3023`` is deliberately not cited: it is ``adapt_dt=1``-only,
    ``#ifdef MPI``-only, and cleared at ``cm1.F:3035`` before any reader.
    """

    if isinstance(courant, bool) or not isinstance(courant, (int, float)):
        raise TypeError("courant must be numeric")
    dt = _positive_finite("dt_s", dt_s)
    if not math.isfinite(float(courant)):
        return (
            f"the advective Courant number is {courant!r}; the wind field "
            "already contains a non-finite word, so no timestep can recover it"
        )
    if float(courant) < CM1_COURANT_STOP:
        return None
    return (
        f"the advective Courant number {float(courant):.6f} has reached CM1's "
        f"own stop value {CM1_COURANT_STOP} (misclibs.F:2714, reached from "
        "statpack.F:469; cm1.F:4385-4392 then stops the model) at "
        f"dt={dt!r} s{grid_note}; the fifth-order advection stencil is "
        "explicit in the large step, so beyond one cell per step it amplifies "
        "the shortest resolved wavelength every step and the wind, pressure "
        "and moisture fields diverge instead of forecasting. Reduce run.dt_s "
        "in proportion, or coarsen the grid"
    )


#: The components this port has an admitted, scheme-matched reflectivity
#: formula for.  CM1 keeps the same list in ``param.F:1583-1603``, where
#: ``output_dbz`` is refused for every ``ptype`` whose microphysics carries no
#: reflectivity routine -- the formula depends on the scheme's own assumed
#: rain/snow/graupel size distributions, so it is not interchangeable.  This
#: port drives WSM6 through the ArWen seam, whose reflectivity is WRF's
#: ``refl10cm_wsm6`` (``mp_wsm6.F90:2275-2444``): fixed-intercept exponential
#: rain, temperature-dependent snow intercept, fixed-intercept graupel, and
#: the Blahak melting-particle integration.
REFLECTIVITY_COMPONENTS: frozenset[str] = frozenset({"wsm6-mp6"})

#: CM1 README.namelist:1366-1369 and writeout.F:1081-1091 / :4089-4107.
REFLECTIVITY_AUTHORITY = (
    "CM1 r21.1 README.namelist:1366-1369 output_dbz; cm1.F:3214-3237 getdbz; "
    "writeout.F:4089-4107 dbz; writeout.F:1081-1091 and misclibs.F:2259-2292 "
    "calccref; WSM6 formula WRF mp_wsm6.F90:2275-2444 through the ArWen seam"
)


def reflectivity_level_index(grid: Any, height_m: float) -> int | None:
    """The model level whose reflectivity plane answers ``height_m``.

    ``misclibs.F:2454-2466`` scans ``k=nk,1,-1`` and keeps the last level
    whose ``zh`` is at or above the requested height, which is the lowest
    such level; a height the column never reaches is left unset and CM1's
    ``kcheck`` guard then skips the report.  That unset case is returned as
    ``None`` here rather than substituted with the model top, because a plane
    silently taken from the top of the domain would be published under the
    height the configuration named.

    The return is a zero-based index into this runtime's ``nz`` physical
    levels, which is what every device array here is indexed by.
    """

    nz = int(grid.nz)
    column = grid.levels.zh
    found: int | None = None
    for k in range(nz, 0, -1):
        # ``zh`` is CM1-indexed with one ghost level below the ground, so the
        # physical level k lives at offset k.
        if float(column[k]) >= float(height_m):
            found = k
    return None if found is None else found - 1


# One WSM6 call stages five mass volumes plus an interface volume of its own
# block.  This is not a physical bound -- it is the point past which a single
# block is certainly larger than any device this port runs on -- and it exists
# so an implausible declaration is refused on the host rather than by a failed
# allocation inside an already-advanced large step.
MAX_PHYSICS_BLOCK_CELLS = 4096


class StepPhase(str, Enum):
    CLOSURE = "closure"
    HELD_TENDENCIES = "solve1-held-tendencies"
    PRE_ACOUSTIC = "pre-acoustic"
    ACOUSTIC = "acoustic"
    POST_ACOUSTIC = "post-acoustic"
    SCALAR_MOIST = "scalar-moist-rk"
    REMOVE_HELD_EOS = "remove-held-and-moist-eos"
    MASS_BEGIN = "dry-mass-reduction"
    MOMENTUM_BOUNDARIES = "momentum-boundaries"
    SCALAR_BOUNDARIES = "scalar-boundaries"
    TKE = "tke-rk"
    MASS_FINISH = "dry-mass-correction"
    PDEFQ = "positive-definite-moisture"
    MICROPHYSICS = "microphysics"
    POST_PHYSICS = "post-physics-eos"
    SOLVE3 = "solve3"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """The runtime's own contract, bound from one already sealed document.

    The vertical coordinate is not a separate word here.  ``grid`` carries it:
    :class:`~cm1core.grid_geometry.CartesianGridGeometry` holds CM1's
    ``stretch_z`` branch and, for any branch but ``param.F:7789-7811``'s
    uniform one, the generated ``zh``/``zf``/``mh``/``mf``/``rds``/``rdsf``
    words themselves.  ``grid.dz_m`` stays CM1's namelist ``dz`` -- the metric
    reference ``param.F:8170`` divides every ``mh`` by -- so a consumer that
    reads ``dz_m`` as a layer thickness is wrong on a stretched grid and must
    read ``grid.levels`` instead.
    """

    grid: CartesianGridGeometry
    dt_s: float = 0.5
    acoustic_substeps: int = 6
    tile_nx: int = 64
    tile_ny: int = 64
    memory_budget_bytes: int | None = None
    # How the closure tile above was chosen.  ``None`` is a declared tile,
    # which is every document written before the tile keys became optional,
    # and projects nothing into record(); a document that leaves both keys
    # out gets the door's resolution record here (one whole-domain tile that
    # fits the declared budget) so the receipt says the tile was not declared.
    tile_selection: Any = field(default=None, compare=False)
    microphysics: str = DEFAULT_COMPONENT_ID
    # CM1 param.F:1567-1580 ihail: which dense precipitating ice category the
    # bound scheme is initialized with.  param.F:3407 passes it to Morrison
    # and param.F:3697-3706 passes it to WSM6; both reach the scheme through
    # cm1core.dense_ice, so a run records constants rather than a flag.
    #
    # None means "undeclared": __post_init__ resolves it to the selected
    # component's own recorded default, which is None again for a component
    # that has no dense-ice branch to select.  A literal integer handed to a
    # component with no selector is refused by cm1core.dense_ice rather than
    # recorded and ignored.
    ihail: int | None = None
    damping_depth_m: float | None = None
    damping_rate_s_inv: float = 0.0033333333333
    # CM1 base.F:2661-2674 umove/vmove.  Every wind this runtime carries is
    # relative to a frame translating at this velocity; the ground-relative
    # wind is this plus the carried word (init3d.F:32-33).  It is part of the
    # configuration, not a side note, because a consumer that does not know it
    # reports every position wrong by u_move_m_s times the model time.
    u_move_m_s: float = 0.0
    v_move_m_s: float = 0.0
    # CM1 param2 iorigin (param.F:7212-7213), declared once in [initialization]:
    # the frame every published absolute coordinate -- a parcel position among
    # them -- is measured in.  1 is the corner, 2 the domain centre.
    iorigin: int = 1
    # CM1 r21.1 param2/param3 rotation selectors, by CM1's own namelist names.
    # icor=0 is CM1's own supercell and squall-line default and stays GM1's;
    # 17 of CM1's 23 shipped decks set icor=1.
    icor: int = 0
    betaplane: int = 0
    fcor_s_inv: float = 5.0e-5
    # CM1 r21.1 param2 lspgrad.  0 is run/namelist.input:61 and every run
    # before the key existed; 1 is src/solve1.F:590-612, which needs icor=1 on
    # the f plane to have a Coriolis term to balance.
    lspgrad: int = 0
    # CM1 r21.1 param3 acoustic-solver words.  alph is CM1's own 0.60.
    alph: float = CM1_DEFAULT_ALPH
    kdiv: float = CM1_DEFAULT_KDIV
    # CM1 param2 adapt_dt.  When true the controller of misclibs.F:593-810
    # chooses each large step from the Courant and diffusion numbers the
    # completed step measured; ``dtl_s`` stays the document's declared step,
    # which misclibs.F:654 uses as the ceiling (never more than twice it) and
    # which the receipt records as the step that was asked for.
    adaptive_timestep: bool = False
    dtl_s: float | None = None
    # CM1 r21.1 lower boundary and surface layer, by CM1's own namelist
    # names.  bbc=1 is CM1's own supercell deck (run/namelist.input:70) and
    # the free-slip rigid bottom this runtime has always integrated; bbc=2
    # and bbc=3 replace the t13/t23 lower face (src/turb.F:4548-4605), which
    # is where surface drag enters the momentum tendency.
    bbc: int = 1
    sfcmodel: int = 0
    isfcflx: int = 0
    cecd: int = 1
    cnstcd: float = 0.001
    # CM1 README.namelist:1366-1369.  ``output_dbz`` is CM1's own namelist
    # word for "write simulated radar reflectivity", and CM1's distributed
    # idealized deck has it on (run/namelist.input:247) while its LES deck has
    # it off (run/config_files/les_ib_windtunnel/namelist.input:246).  It is
    # off by default here because the reflectivity carrier costs a further
    # float32 volume in every published frame and CM1's own param.F:1583-1603
    # treats the diagnostic as optional per run.
    output_dbz: bool = False
    # The single height whose horizontal reflectivity plane is published
    # beside the column maximum.  CM1 has no namelist word for it -- a CM1
    # user slices the 3-D ``dbz`` field themselves -- so the level is named in
    # metres and resolved to CM1's own descending level scan
    # (misclibs.F:2454-2466) against this grid's ``zh`` column.
    dbz_level_m: float | None = None
    # How CM1's pre-WSM6 positive-definite repair (src/misclibs.F:1037-1117)
    # is spread over the device.  Not a CM1 selector: both admitted layouts
    # keep misclibs.F:1065-1081's per-column k order and misclibs.F:1082's
    # left-associated west-to-east budj carry serial, and produce raw-word
    # identical q3d and asq.  What they change is how many thread blocks the
    # repair occupies -- the pinned ``row-serial`` layout parallelises over
    # physical rows alone, so a 448-row domain resolves it to four blocks
    # whatever ni and nk are.
    pdefq_launch: str = DEFAULT_WSM6_PDEFQ_LAYOUT
    # The rectangle one WSM6 call covers.  WSM6 is column physics -- CM1 calls
    # it once per physical column (src/mp_driver.F:1469-1479 inside the
    # i/j sweep) and no column reads another -- so this decides only how many
    # times the per-call staging, launch and drain are paid, and how much
    # component scratch one call needs.  It used to be the closure tile, which
    # is a different operator with a different stencil and a different reason
    # for its size; ``None`` still means "follow the closure tile", so a
    # document that does not declare it runs exactly the decomposition it ran
    # before.
    physics_block_nx: int | None = None
    physics_block_ny: int | None = None
    # CM1 r21.1 lateral boundaries, by CM1's own namelist names
    # (README.namelist:561-568,589-610,826-829).  ``wbc``..``nbc`` are 1
    # periodic, 2 open-radiative, 3 rigid free-slip, 4 rigid no-slip; ``irbc``
    # selects which radiative scheme an open side runs; ``roflux``,
    # ``nudgeobc``/``alphobc`` and ``cstar`` are the open-boundary companions.
    # The defaults are the all-open family this runtime has always integrated.
    wbc: int = 2
    ebc: int = 2
    sbc: int = 2
    nbc: int = 2
    irbc: int = 4
    roflux: int = 0
    nudgeobc: int = 0
    alphobc: float = 60.0
    cstar: float = 30.0
    # CM1's upper and lateral Rayleigh selectors (README.namelist:391-403).
    # ``irdamp`` is 0 off, 1 damp toward the base state, 2 toward the evolving
    # horizontal average; ``hrdamp`` is the lateral sponge with width ``xhd_m``.
    irdamp: int = 1
    hrdamp: int = 0
    xhd_m: float = 0.0
    # run/namelist.input:139,148-151 -- the remaining sfcmodel=1 options and
    # CM1's param12 initial surface conditions.  Their defaults are CM1's own
    # shipped deck, so a caller that declares none of them gets the surface
    # CM1 itself runs by default (run/namelist.input:130-151).
    cnstce: float = 0.001
    set_znt: int = 0
    cnst_znt: float = 0.16
    set_ust: int = 0
    cnst_ust: float = 0.25
    initsfc: int = 1
    tsk0_k: float = 299.28
    tmn0_k: float = 297.28
    xland0: float = 2.0
    lu0: int = 16
    season: int = 1
    oceanmodel: int = 1
    #: Derived in __post_init__ from ``microphysics``/``ihail``; never passed.
    profile: MoistureSpeciesProfile = field(init=False, repr=False, compare=False,
                                            default=None)  # type: ignore[assignment]
    dense_ice: DenseIceSelection | DenseIceAbsent = field(
        init=False, repr=False, compare=False, default=None,  # type: ignore[assignment]
    )
    # CM1 r21.1 radiation, boundary layer and cumulus, by CM1's own names.
    # The defaults are CM1's own and are what GM1's Cartesian storm lane has
    # always integrated: no radiation, the LES closure, no PBL scheme, no
    # convection parameterization.  rterm=1 is the one of the four that is
    # live on the device today (src/solve1.F:1003-1054).
    rterm: int = 0
    # base.F's qc0/qi0 (isnd = 4, 10, 11): whether the dynamics holds the two
    # base-state condensate volumes solve2.F:404's buoyancy subtracts.  False
    # is every sounding without base-state cloud, which subtracts exact zero.
    base_condensate: bool = False
    cm1setup: int = 1
    ipbl: int = 0
    cuparam: int = 0
    # README.namelist:334-352 and :1089-1092.  sgsmodel=1 is the Deardorff
    # closure every earlier run integrated; 3 and 4 add CM1's two-part
    # near-wall models (cm1core.cuda_gm1_two_part_sgs).
    sgsmodel: int = 1
    t2p_avg: int = 1
    # README.namelist:1081-1087, src/turb.F:240-262.  0 is src/cm1.F:355's
    # compiled default and what every earlier run integrated; every deck CM1
    # ships declares 1 with ramp_time = 1800 s.
    ramp_sgs: int = 0
    ramp_time_s: float = 0.0
    #: CM1's param11 selection, or ``None`` for src/param.F:6347's radopt=0.
    radiation: Any = None
    # README.namelist:583-588.  The top boundary for winds, which calcvort's
    # yvort/xvort upper closure reads (misclibs.F:1666-1683).  This lane's
    # boundary family is sealed to tbc=1 (free slip) by config.py:1521-1525.
    tbc: int = 1
    # CM1 &param9.  What the history frame contains beyond the prognostic
    # state; see cm1_output_selection for the 64 words and their dispositions.
    output: Any = None
    # CM1 writeout.F:2895-3030 output_cape/output_cin/output_lcl/output_lfc and
    # the getcape.F:60,70,73 source parameters behind them, by CM1's own names.
    # All four off is run/namelist.input's own shipped state and the state every
    # document that declares no [diagnostics] table is in: writeout.F:2895 then
    # runs no column sweep, the history carrier table is the twelve carriers it
    # has always been, and the published frame is byte for byte what it was.
    output_cape: bool = False
    output_cin: bool = False
    output_lcl: bool = False
    output_lfc: bool = False
    parcel_source: int = 3
    adiabat: int = 1
    mixed_layer_depth_m: float = 500.0
    pressure_increment_pa: float = 200.0
    storm_motion: str = "bunkers-right"
    # CM1's output_* diagnostic switches (src/input.F:120, src/writeout.F:408-545).
    # All-off is the state every run integrated before they existed, and an
    # all-off selection allocates nothing, computes nothing and is not
    # projected into this contract's record, so a default run's bytes and
    # every identity sealed from them are exactly what they were.
    diagnostics: Any = None
    # CM1's parcel/budget/circuit/pdcomp/hifrq selectors.  None is CM1's own
    # shipped deck (iprcl=0, output_pdcomp=0, dohifrq=.false.), which is what
    # every run before this table existed integrated, so an absent table
    # changes no result and no receipt word.
    lagrangian: Any = None
    # init3d.F:1303-1396's forcing arms (iinit = 9, 10, 12): an
    # cm1core.initiation_forcing.InitiationForcing, or None for every other
    # iinit, which is every run before the forcings existed.
    initiation_forcing: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.grid, CartesianGridGeometry):
            raise TypeError("grid must be CartesianGridGeometry")
        self.grid.require_stencil(radius_xy=3, minimum_nz=8)
        if isinstance(self.dt_s, bool) or not isinstance(self.dt_s, (int, float)):
            raise TypeError("dt_s must be numeric")
        if not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        object.__setattr__(self, "dt_s", float(self.dt_s))
        if type(self.adaptive_timestep) is not bool:
            raise TypeError("adaptive_timestep must be boolean")
        if self.dtl_s is None:
            object.__setattr__(self, "dtl_s", float(self.dt_s))
        elif isinstance(self.dtl_s, bool) or not isinstance(self.dtl_s, (int, float)):
            raise TypeError("dtl_s must be numeric or None")
        else:
            object.__setattr__(self, "dtl_s", float(self.dtl_s))
        if not math.isfinite(self.dtl_s) or self.dtl_s <= 0.0:
            raise ValueError("dtl_s must be finite and positive")
        if type(self.acoustic_substeps) is not int or self.acoustic_substeps < 4:
            raise ValueError("the acoustic schedule needs at least four substeps")
        if self.acoustic_substeps % 2:
            raise ValueError("CM1 acoustic diagnostic cadence needs an even substep count")
        # misclibs.F:818-915.  The large step is a free parameter here, as it
        # is in CM1; what is not free is the acoustic small step it implies.
        defect = acoustic_small_step_defect(
            dt_s=self.dt_s,
            dx_m=self.grid.dx_m,
            dy_m=self.grid.dy_m,
            acoustic_substeps=self.acoustic_substeps,
        )
        if defect is not None:
            raise ValueError(f"dt_s={self.dt_s} is inadmissible: {defect}")
        for name in ("tile_nx", "tile_ny"):
            value = getattr(self, name)
            if type(value) is not int or value < 7:
                raise ValueError(f"{name} must be an integer >= 7")
        if self.memory_budget_bytes is not None and (
            type(self.memory_budget_bytes) is not int or self.memory_budget_bytes <= 0
        ):
            raise ValueError("memory_budget_bytes must be a positive integer")
        if type(self.output_dbz) is not bool:
            raise TypeError("output_dbz must be a strict boolean, as CM1's 0/1 namelist word is")
        # Ordered before the component binding on purpose: a caller who asked
        # for reflectivity under a component that has none needs to be told
        # about the reflectivity, which is the key whose result would be
        # wrong rather than absent.
        if self.output_dbz and self.microphysics not in REFLECTIVITY_COMPONENTS:
            # CM1 param.F:1583-1603 refuses output_dbz for every ptype whose
            # scheme carries no reflectivity routine.  The formula is a
            # function of the scheme's own assumed size distributions, so
            # there is no generic one to fall back on.
            raise ValueError(
                f"the {self.microphysics!r} component has no admitted reflectivity "
                "formula; reflectivity is a function of the scheme's own assumed "
                "particle size distributions, so publishing another scheme's "
                "formula under this run's hydrometeors would report a dBZ field "
                "the simulated storm never scattered"
            )
        # The component registry below is the authority for which columns bind:
        # moisture_species refuses an untranscribed component by name and
        # names the wrong result.  A second hard-coded list here would refuse
        # every transcribed component one layer under the door, which is the
        # landed-but-unselectable failure this port tracks -- so the only
        # special case kept is "off".
        #
        # "off" is CM1's imoist=0: no column component at all.  It reaches
        # this contract only through the iconly door, which publishes the
        # initial condition and compiles nothing, so the runtime carries the
        # name without a kernel binding rather than refusing an inspection.
        if self.dbz_level_m is not None:
            if isinstance(self.dbz_level_m, bool) or not isinstance(self.dbz_level_m, (int, float)):
                raise TypeError("dbz_level_m must be numeric")
            object.__setattr__(self, "dbz_level_m", float(self.dbz_level_m))
            if not math.isfinite(self.dbz_level_m) or self.dbz_level_m <= 0.0:
                raise ValueError("dbz_level_m must be a finite positive height in metres")
            if not self.output_dbz:
                raise ValueError(
                    "dbz_level_m is declared with output_dbz off, so the level plane "
                    "would be named in the configuration and never computed, and every "
                    "published frame would carry no reflectivity at all"
                )
            if reflectivity_level_index(self.grid, self.dbz_level_m) is None:
                raise ValueError(
                    f"dbz_level_m={self.dbz_level_m!r} m is above this grid's highest "
                    f"scalar level; misclibs.F:2454-2466 leaves such a request unset and "
                    "the published plane would silently be the model top instead of the "
                    "height the configuration names"
                )
        # moisture_species refuses an untranscribed component by name and
        # dense_ice refuses an ihail outside CM1's own 0..1 range, each naming
        # the wrong result it prevents.
        # "off" (imoist = 0, iconly only) binds no column: the host
        # initializer still publishes base.F's six-row moist carrier group,
        # exact zero for a dry base state (base.F:2088-2100), so the
        # publication is laid out on the default component's axis.  Nothing
        # transports, repairs or feeds a scheme with those rows.
        layout = DEFAULT_COMPONENT_ID if self.microphysics == "off" else self.microphysics
        object.__setattr__(self, "profile", moisture_species_profile(layout))
        if self.ihail is None:
            object.__setattr__(self, "ihail", default_ihail(layout))
        object.__setattr__(
            self, "dense_ice", dense_ice_selection(layout, self.ihail)
        )
        if self.damping_depth_m is None:
            object.__setattr__(self, "damping_depth_m", min(3000.0, self.grid.ztop_m * 0.2))
        for name in ("damping_depth_m", "damping_rate_s_inv"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.damping_depth_m > self.grid.ztop_m * 0.5:
            raise ValueError("the upper damping layer cannot extend below half the model top")
        for name in ("u_move_m_s", "v_move_m_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            object.__setattr__(self, name, float(value))
            if not math.isfinite(getattr(self, name)):
                raise ValueError(
                    f"{name} must be finite; base.F:2667 subtracts it from every "
                    "base-state wind word, so a non-finite frame velocity makes "
                    "u0/v0 and the ua/va copied from them non-finite at every level"
                )
            if abs(getattr(self, name)) > MAX_FRAME_SPEED_M_S:
                raise ValueError(
                    f"{name} exceeds {MAX_FRAME_SPEED_M_S} m/s; the frame velocity is "
                    "a storm motion, and base.F:2667 would replace the declared "
                    "hodograph with a uniform frame-relative jet through the whole depth"
                )
            if cm1_real(getattr(self, name)) != getattr(self, name):
                raise ValueError(
                    f"{name}={getattr(self, name)!r} m/s is not a CM1 `real`; umove and "
                    "vmove are binary32 (base.F:2661-2674), so base.F:2667 would "
                    f"subtract {cm1_real(getattr(self, name))!r} m/s from every "
                    "base-state wind word while cm1.F:3512 integrated the declared "
                    "value, and the published domain location would walk away from "
                    "the carriers it describes at the difference times the model time"
                )
        for name, admitted in (("icor", (0, 1)), ("betaplane", (0, 1))):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer CM1 selector")
            if value not in admitted:
                raise ValueError(
                    f"{name}={value} is not a CM1 r21.1 selector; "
                    f"solve2.F:633-673 dispatches on {sorted(admitted)}"
                )
        if isinstance(self.fcor_s_inv, bool) or not isinstance(
            self.fcor_s_inv, (int, float)
        ):
            raise TypeError("fcor_s_inv must be numeric")
        object.__setattr__(self, "fcor_s_inv", float(self.fcor_s_inv))
        if not math.isfinite(self.fcor_s_inv):
            raise ValueError("fcor_s_inv must be finite")
        if self.betaplane == 1:
            if self.icor != 1:
                raise ValueError(
                    "betaplane=1 with icor=0 declares a latitude-varying "
                    "Coriolis parameter for a term solve2.F:632 never "
                    "evaluates; the run would integrate no rotation at all"
                )
            if abs(self.fcor_s_inv) > CM1_MAX_ABS_FCOR_S_INV:
                raise ValueError(
                    f"|fcor_s_inv|={abs(self.fcor_s_inv)} exceeds 2*omega="
                    f"{CM1_MAX_ABS_FCOR_S_INV}; param.F:8952 takes "
                    "asin(fcor/(2*omega)), which is NaN outside that domain, "
                    "so every f2d word and then the whole momentum field "
                    "would go non-finite on the first beta-plane stage"
                )
            if self.u_move_m_s != 0.0 or self.v_move_m_s != 0.0:
                raise ValueError(
                    "betaplane=1 with a translating frame is refused: "
                    "solve2.F:667-672 omits the +umove/+vmove the f-plane "
                    "branch at solve2.F:645,647 adds, so a frame moving at "
                    f"({self.u_move_m_s}, {self.v_move_m_s}) m/s would "
                    "integrate a Coriolis acceleration short by "
                    f"{abs(self.fcor_s_inv * self.v_move_m_s):.3e} and "
                    f"{abs(self.fcor_s_inv * self.u_move_m_s):.3e} m/s^2 at "
                    "every point, for the whole run"
                )
        if isinstance(self.lspgrad, bool) or not isinstance(self.lspgrad, int):
            raise TypeError("lspgrad must be an integer CM1 selector")
        if self.lspgrad not in (0, 1):
            raise ValueError(
                f"lspgrad={self.lspgrad} is not admitted; only solve1.F:590-612's "
                "base-state branch (1) is ported, and 2/3/4 force from ug/vg, "
                "gradient-wind or ulspg/vlspg profiles this runtime does not carry"
            )
        if self.lspgrad == 1 and (self.icor != 1 or self.betaplane != 0):
            raise ValueError(
                "lspgrad=1 subtracts fcor x (V0 + Vmove) (solve1.F:606-609), "
                "which only the f-plane icor=1 term fcor x (V + Vmove) "
                "(solve2.F:639-653) balances; with icor="
                f"{self.icor} betaplane={self.betaplane} the base-state wind "
                "itself would be accelerated for the whole run"
            )
        for name, lower, upper in (("alph", 0.0, 1.0), ("kdiv", 0.0, 1.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            object.__setattr__(self, name, float(value))
            value = getattr(self, name)
            if not math.isfinite(value) or not (lower <= value <= upper):
                raise ValueError(
                    f"{name}={value} is outside CM1's admitted [{lower}, {upper}]; "
                    "sound.F:78-82 forms beta=1-alph and asq=alph*alph, and an "
                    "off-centering weight outside the unit interval makes the "
                    "vertically implicit acoustic step unconditionally unstable"
                )
        from .cm1_output_selection import DEFAULT_OUTPUT_SELECTION, OutputSelection

        if self.output is None:
            object.__setattr__(self, "output", DEFAULT_OUTPUT_SELECTION)
        if not isinstance(self.output, OutputSelection):
            raise TypeError("output must be a cm1_output_selection.OutputSelection")
        # One loop over every integer CM1 selector this contract carries: the
        # lower boundary, the surface layer, the initial surface and the top.
        for name in (
            "bbc", "sfcmodel", "isfcflx", "cecd", "tbc",
            "set_znt", "set_ust", "initsfc", "lu0", "season", "oceanmodel",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer CM1 selector")
        if isinstance(self.iorigin, bool) or self.iorigin not in (1, 2):
            raise ValueError(
                "iorigin must be 1 or 2; param.F:6666-6672 stops CM1 on anything else"
            )
        if self.tbc not in (1, 2):
            raise ValueError(
                "tbc must be 1 (free slip) or 2 (no slip); README.namelist:583-588 "
                "also lists 3 (semi slip), whose t13/t23 upper face this runtime "
                "does not build"
            )
        if isinstance(self.cnstcd, bool) or not isinstance(self.cnstcd, (int, float)):
            raise TypeError("cnstcd must be numeric")
        object.__setattr__(self, "cnstcd", float(self.cnstcd))
        for name, tile in (("physics_block_nx", self.tile_nx), ("physics_block_ny", self.tile_ny)):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, tile)
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
            if value > MAX_PHYSICS_BLOCK_CELLS:
                raise ValueError(
                    f"{name}={value} exceeds {MAX_PHYSICS_BLOCK_CELLS}; one WSM6 call "
                    "stages five mass volumes and an interface volume of its own "
                    "block, so a block this wide would fail to allocate part-way "
                    "through a large step that has already advanced the state past "
                    "the pre-WSM6 seam, where it cannot be re-run"
                )
        # Refuses an unadmitted pdefq launch layout here, before any device
        # memory is priced, and by the same name the TOML key carries.
        require_wsm6_pdefq_layout(self.pdefq_launch)
        wsm6_pdefq_launch_plan(
            ni=self.grid.nx, nj=self.grid.ny, nk=self.grid.nz, layout=self.pdefq_launch
        )
        for name in (
            "cnstcd", "cnstce", "cnst_znt", "cnst_ust",
            "tsk0_k", "tmn0_k", "xland0",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            object.__setattr__(self, name, float(value))
        # The full CM1 cross-checks live in the configuration door; this is
        # the runtime contract's own floor, so a programmatic caller that
        # bypasses the door still cannot build a lower boundary the surface
        # operator would silently reinterpret.
        from .cuda_getcape import require_getcape_selection
        from .cuda_surface_drag import SurfaceDragSelection, validate_surface_drag_selection

        # The full CM1 cross-checks for the environment planes live in the
        # configuration door too; this is the runtime contract's own floor, so
        # a programmatic caller that bypasses the door still cannot ask for a
        # parcel or an adiabat getcape.F would stop on.
        require_getcape_selection(self.environment_field_selection)
        # The storm motion is not a getcape selector -- CM1 computes none at
        # all -- so it is checked here beside them: a receipt that names a
        # method nobody implemented would publish a helicity that cannot be
        # reproduced, and the sign of the reported SRH flips between movers.
        from .errors import refuse
        from .sounding_report import STORM_MOTIONS

        if self.storm_motion not in STORM_MOTIONS:
            refuse(
                code="CM1-SOUNDING-REPORT-MOTION",
                gate="support-matrix",
                reason=(
                    "storm-relative helicity is a line integral taken against one "
                    "storm motion, and CM1 r21.1 computes none anywhere in its "
                    f"source tree; {self.storm_motion!r} names no transcribed "
                    "method, so the receipt would publish a helicity no reader "
                    "could reproduce and whose sign nobody could interpret"
                ),
                remedy=(
                    "declare diagnostics.storm_motion as "
                    + " or ".join(STORM_MOTIONS)
                ),
                option="diagnostics.storm_motion",
                value=self.storm_motion,
            )
        validate_surface_drag_selection(
            SurfaceDragSelection(
                bbc=self.bbc,
                sfcmodel=self.sfcmodel,
                isfcflx=self.isfcflx,
                cecd=self.cecd,
                cnstcd=self.cnstcd,
                cnstce=self.cnstce,
                set_znt=self.set_znt,
                cnst_znt=self.cnst_znt,
                set_ust=self.set_ust,
                cnst_ust=self.cnst_ust,
                u_move_m_s=self.u_move_m_s,
                v_move_m_s=self.v_move_m_s,
            ),
            option_prefix="closures",
        )
        # The same floor for the lateral family: the door applies every CM1
        # cross-check, but a programmatic caller that bypasses it still cannot
        # build a family the boundary operator would silently reinterpret.
        self.lateral_boundary_selection
        for name in ("irdamp", "hrdamp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2):
                raise ValueError(f"{name} must be CM1's 0, 1 or 2")
        if isinstance(self.xhd_m, bool) or not isinstance(self.xhd_m, (int, float)):
            raise TypeError("xhd_m must be numeric")
        object.__setattr__(self, "xhd_m", float(self.xhd_m))
        if not math.isfinite(self.xhd_m) or self.xhd_m < 0.0:
            raise ValueError("xhd_m must be finite and nonnegative")
        if self.hrdamp >= 1 and self.xhd_m <= 0.0:
            raise ValueError(
                "hrdamp>=1 needs a positive xhd_m: param.F:8736 divides by it"
            )
        # The same floor for the three physics families: a programmatic
        # caller that bypasses the configuration door still cannot build a
        # run whose column is mixed twice, whose sun is switched on without
        # a place, or whose convection is parameterized at a spacing that
        # resolves it.
        from .cm1_physics_families import (
            Cm1CumulusSelection,
            Cm1RadiationSelection,
            Cm1TurbulenceScale,
            NO_RADIATION,
            validate_cumulus_selection,
            validate_radiation_selection,
            validate_turbulence_scale,
        )

        for name in ("rterm", "cm1setup", "ipbl", "cuparam", "sgsmodel", "t2p_avg", "ramp_sgs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer CM1 selector")
        if self.ramp_sgs not in (0, 1):
            raise ValueError("ramp_sgs is README.namelist:1082's 0 or 1")
        if isinstance(self.ramp_time_s, bool) or not isinstance(self.ramp_time_s, (int, float)):
            raise TypeError("ramp_time_s must be numeric")
        object.__setattr__(self, "ramp_time_s", float(self.ramp_time_s))
        if not math.isfinite(self.ramp_time_s) or self.ramp_time_s < 0.0:
            raise ValueError("ramp_time_s must be finite and nonnegative")
        from .cuda_gm1_two_part_sgs import validate_two_part_selection

        validate_two_part_selection(
            sgsmodel=self.sgsmodel, t2p_avg=self.t2p_avg,
            t2p_avg_declared=self.t2p_avg != 1, terrain=False,
            bbc=self.bbc, sfcmodel=self.sfcmodel,
        )
        if self.rterm not in (0, 1):
            raise ValueError(
                "src/solve1.F:1006 tests rterm against 1 and nothing else, so "
                f"rterm={self.rterm} would be the rterm=0 identity wearing a label"
            )
        if self.radiation is None:
            object.__setattr__(self, "radiation", NO_RADIATION)
        if not isinstance(self.radiation, Cm1RadiationSelection):
            raise TypeError("radiation must be a Cm1RadiationSelection")
        validate_turbulence_scale(
            Cm1TurbulenceScale(
                cm1setup=self.cm1setup, sgsmodel=1, ipbl=self.ipbl
            )
        )
        validate_radiation_selection(self.radiation, rterm=self.rterm)
        validate_cumulus_selection(
            Cm1CumulusSelection(
                cuparam=self.cuparam, dx_m=self.grid.dx_m, dy_m=self.grid.dy_m
            )
        )
        # The same floor for the rotation diagnostics and surface swaths: a
        # programmatic caller that bypasses the configuration door still
        # cannot ask for a diagnostic this grid, this microphysics or this
        # lower boundary cannot support.
        from .tornado_diagnostics import (
            DEFAULT_TORNADO_DIAGNOSTICS,
            TornadoDiagnosticSelection,
            validate_tornado_diagnostic_selection,
        )

        if self.diagnostics is None:
            object.__setattr__(self, "diagnostics", DEFAULT_TORNADO_DIAGNOSTICS)
        if not isinstance(self.diagnostics, TornadoDiagnosticSelection):
            raise TypeError("diagnostics must be a TornadoDiagnosticSelection")
        validate_tornado_diagnostic_selection(
            self.diagnostics,
            grid=self.grid,
            microphysics=self.microphysics,
            bbc=self.bbc,
            option_prefix="diagnostics",
        )

    @property
    def lateral_boundary_selection(self) -> Any:
        """CM1's lateral family and open-boundary companions as one selection."""

        from .cuda_boundary_families import LateralBoundarySelection

        return LateralBoundarySelection(
            west=self.wbc,
            east=self.ebc,
            south=self.sbc,
            north=self.nbc,
            irbc=self.irbc,
            roflux=self.roflux,
            nudgeobc=self.nudgeobc,
            alphobc=self.alphobc,
            cstar=self.cstar,
        )

    @property
    def environment_field_selection(self) -> Any:
        """CM1's four ``output_*`` switches and getcape's controls as one selection."""

        from .cuda_getcape import GetCapeFieldSelection

        return GetCapeFieldSelection(
            output_cape=self.output_cape,
            output_cin=self.output_cin,
            output_lcl=self.output_lcl,
            output_lfc=self.output_lfc,
            parcel_source=self.parcel_source,
            adiabat=self.adiabat,
            mixed_layer_depth_m=self.mixed_layer_depth_m,
            pressure_increment_pa=self.pressure_increment_pa,
        )

    @property
    def surface_drag_selection(self) -> Any:
        """CM1's lower-boundary selectors as one validated selection."""

        from .cuda_surface_drag import SurfaceDragSelection

        return SurfaceDragSelection(
            bbc=self.bbc,
            sfcmodel=self.sfcmodel,
            isfcflx=self.isfcflx,
            cecd=self.cecd,
            cnstcd=self.cnstcd,
            cnstce=self.cnstce,
            set_znt=self.set_znt,
            cnst_znt=self.cnst_znt,
            set_ust=self.set_ust,
            cnst_ust=self.cnst_ust,
            u_move_m_s=self.u_move_m_s,
            v_move_m_s=self.v_move_m_s,
        )

    @property
    def surface_field_selection(self) -> Any:
        """CM1's ``param12`` initial surface conditions as one selection.

        ``src/init_surface.F:40`` runs only for ``sfcmodel>=1`` or ``bbc=3``,
        so this is meaningful exactly when the lower boundary is not free
        slip; the caller checks that before building the fields.
        """

        from .surface_fields import SurfaceFieldSelection

        return SurfaceFieldSelection(
            initsfc=self.initsfc,
            sfcmodel=self.sfcmodel,
            bbc=self.bbc,
            isfcflx=self.isfcflx,
            set_znt=self.set_znt,
            cnst_znt=self.cnst_znt,
            tsk0_k=self.tsk0_k,
            tmn0_k=self.tmn0_k,
            xland0=self.xland0,
            lu0=self.lu0,
            season=self.season,
            oceanmodel=self.oceanmodel,
        )

    def record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "grid": self.grid.record(),
            "dt_s": self.dt_s,
            # README.namelist:222-225.  With adapt_dt on, dt_s above is the
            # step in force right now and dtl_s is the one the document asked
            # for; misclibs.F:654 never lets the controller exceed twice it.
            "adapt_dt": self.adaptive_timestep,
            "dtl_s": self.dtl_s,
            "acoustic_substeps": self.acoustic_substeps,
            # misclibs.F:818-915.  Published so a consumer can see the small
            # step the large step actually implies, and the count CM1's own
            # dtsmall would have derived for this grid.
            "acoustic_small_step_s": self.dt_s / self.acoustic_substeps,
            "acoustic_small_step_limit_s": cm1_acoustic_small_step_limit_s(
                dx_m=self.grid.dx_m, dy_m=self.grid.dy_m
            ),
            "cm1_diagnostic_acoustic_substeps": cm1_diagnostic_acoustic_substeps(
                dt_s=self.dt_s, dx_m=self.grid.dx_m, dy_m=self.grid.dy_m
            ),
            "acoustic_authority": "CM1 r21.1 misclibs.F:818-915 (dtsmall)",
            # misclibs.F:2714 via statpack.F:469.  The advective Courant number
            # needs the wind field and is measured on the live state; what the
            # configuration alone fixes is the wind speed at which this step
            # and this spacing would reach CM1's stop value, whichever way the
            # wind points.  Published so a consumer can see how much headroom
            # the declared step leaves before the run would be abandoned.
            "advective_courant_stop": CM1_COURANT_STOP,
            "advective_courant_target": CM1_COURANT_TARGET,
            "advective_courant_stop_speed_m_s": cm1_courant_limited_speed_m_s(
                dt_s=self.dt_s,
                dx_m=self.grid.dx_m,
                dy_m=self.grid.dy_m,
                dz_m=self.grid.dz_m,
            ),
            "advective_courant_target_speed_m_s": cm1_courant_limited_speed_m_s(
                dt_s=self.dt_s,
                dx_m=self.grid.dx_m,
                dy_m=self.grid.dy_m,
                dz_m=self.grid.dz_m,
                courant=CM1_COURANT_TARGET,
            ),
            "advective_courant_authority": CM1_COURANT_STOP_AUTHORITY,
            "tile_shape_yx": [self.tile_ny, self.tile_nx],
            # Present only for a base state with cloud (base.F isnd = 4, 10, 11).
            **({"base_condensate": "base.F qc0/qi0 in solve2.F:404's buoyancy"} if self.base_condensate else {}),
            # Present only for an undeclared tile, so a declared-tile record
            # -- and every hash sealed from it -- is exactly what it was.
            **(
                {"tile_selection": dict(self.tile_selection)}
                if self.tile_selection is not None
                else {}
            ),
            # mp_driver.F:1469-1479.  How many WSM6 calls one large step makes
            # and how wide each one is; published because it decides the
            # per-call overhead the step pays, and nothing the model computes.
            "physics_block_shape_yx": [
                min(self.physics_block_ny, self.grid.ny),
                min(self.physics_block_nx, self.grid.nx),
            ],
            "physics_block_count": (
                -(-self.grid.ny // min(self.physics_block_ny, self.grid.ny))
                * -(-self.grid.nx // min(self.physics_block_nx, self.grid.nx))
            ),
            "physics_block_follows_closure_tile": (
                self.physics_block_nx == self.tile_nx
                and self.physics_block_ny == self.tile_ny
            ),
            # misclibs.F:1037-1117.  Published because it decides how much of
            # the device the pre-WSM6 repair occupies; it decides no word the
            # model computes, and the recorded plan says which grids ran.
            "pdefq_launch": self.pdefq_launch,
            "pdefq_launch_plan": wsm6_pdefq_launch_plan(
                ni=self.grid.nx,
                nj=self.grid.ny,
                nk=self.grid.nz,
                layout=self.pdefq_launch,
            ),
            "pdefq_workspace_bytes": wsm6_pdefq_workspace_bytes(
                ni=self.grid.nx,
                nj=self.grid.ny,
                nk=self.grid.nz,
                layout=self.pdefq_launch,
            ),
            "memory_budget_bytes": self.memory_budget_bytes,
            "microphysics": self.microphysics,
            "moisture_species": self.profile.record(),
            "dense_ice": self.dense_ice.record(),
            # Every capability is selectable; one that nothing has measured
            # says so in the plan a reader sees first,
            # in a token a consumer can grep for, naming what was not checked.
            "component_validation": component_validation(self.microphysics).record(),
            "damping_depth_m": self.damping_depth_m,
            "damping_rate_s_inv": self.damping_rate_s_inv,
            # README.namelist:391-403,561-568,589-610,826-829.  Published by
            # CM1's own names so a reader can see which lateral family, which
            # radiative scheme and which Rayleigh branch this run integrated
            # rather than inferring them from the lane.
            "lateral_boundaries": self.lateral_boundary_selection.record(),
            "irdamp": self.irdamp,
            "hrdamp": self.hrdamp,
            "xhd_m": self.xhd_m,
            # base.F:2661-2674.  Published so a frame's ground-relative wind
            # can be recovered: it is u_move_m_s + the carried word
            # (init3d.F:32-33).  Where the domain itself is at a given model
            # time is not derived from this by the consumer -- the runtime
            # integrates it, as CM1 does, and publishes domainlocx/domainlocy
            # with every frame.
            "u_move_m_s": self.u_move_m_s,
            "v_move_m_s": self.v_move_m_s,
            "moving_frame_authority": "CM1 r21.1 base.F:2663-2669 (imove/umove/vmove)",
            # CM1's lower boundary, by CM1's own names.  bbc=1 is the
            # free-slip rigid bottom of run/namelist.input:70 and the
            # surface exerts no stress; bbc=2 and bbc=3 replace the t13/t23
            # lower face, so a reader can tell a frictionless run from a
            # run whose near-ground vorticity came from surface drag.
            "bbc": self.bbc,
            "tbc": self.tbc,
            "sfcmodel": self.sfcmodel,
            "isfcflx": self.isfcflx,
            "cecd": self.cecd,
            "cnstcd": self.cnstcd,
            "cnstce": self.cnstce,
            "set_znt": self.set_znt,
            "cnst_znt": self.cnst_znt,
            "set_ust": self.set_ust,
            "cnst_ust": self.cnst_ust,
            "initsfc": self.initsfc,
            "tsk0": self.tsk0_k,
            "tmn0": self.tmn0_k,
            "xland0": self.xland0,
            "lu0": self.lu0,
            "season": self.season,
            "oceanmodel": self.oceanmodel,
            "surface_authority": (
                "CM1 r21.1 init_surface.F:40-343 (initsfc, getlanduse, "
                "set_znt, the reality checks), sfcphys.F:81-236 (getcecd), "
                "sfcphys.F:341-361 (sfcflux), turb.F:292 (psfc), "
                "turbtend.F:517-521,587-596 (the scalar lower flux)"
            ),
            # CM1 &param9 as this run resolved it, with every default this
            # lane moved away from CM1's distributed deck named and explained,
            # and a loud UNVALIDATED row for anything selected but unproven.
            "output_selection": self.output.record(),
            "lower_boundary_authority": (
                "CM1 r21.1 turb.F:4515-4702 sfcstress, turb.F:3244-3260 tkebc, "
                "sfcphys.F:120-136 getcecd, solve1.F:932-935,950-988 the "
                "bbc=3 lowest-level dissipative heating"
            ),
            **self.physics_families_record(),
            # CM1 writeout.F:2895-3030.  Published so a reader can tell a frame
            # that carries the inflow environment from one that does not, and
            # which parcel the published CAPE belongs to.
            "environment_fields": self.environment_field_selection.record(),
            "storm_motion": self.storm_motion,
            "environment_field_authority": (
                "CM1 r21.1 writeout.F:2895-3030 (output_cape/output_cin/"
                "output_lcl/output_lfc) over getcape.F:14-594; the storm motion "
                "is Bunkers et al. (2000) internal dynamics, which CM1 r21.1 "
                "does not compute"
            ),
            # CM1's output_* switches.  Projected only when at least one is
            # on: an all-off selection is what every run carried before they
            # existed, so its record -- and the config_sha256 a published
            # history frame seals from it -- stays byte-identical.
            **(
                {
                    "diagnostics": self.diagnostics.record(),
                    "diagnostics_authority": (
                        "CM1 r21.1 input.F:120 output_vort/output_pv/output_uh, "
                        "writeout.F:408-545 the surface swaths, "
                        "solve3.F:855-1136 their running extrema, "
                        "misclibs.F:1417-1532 calcuh, misclibs.F:1540-2174 calcvort"
                    ),
                }
                if self.diagnostics is not None and self.diagnostics.active
                else {}
            ),
        }
        # README.namelist:1366-1369.  A run with the diagnostic off publishes
        # exactly the words it has always published -- the same rule the
        # sealed configuration document follows for its own optional keys --
        # so every frame receipt and plan hash of an existing run is unchanged
        # by the reflectivity carrier existing.
        if self.output_dbz:
            record["output_dbz"] = True
            record["reflectivity_authority"] = REFLECTIVITY_AUTHORITY
            if self.dbz_level_m is not None:
                record["dbz_level_m"] = self.dbz_level_m
                record["dbz_level_index"] = reflectivity_level_index(
                    self.grid, self.dbz_level_m
                )
        # The same rule for lspgrad: an lspgrad=0 run publishes neither key,
        # so its record -- and the config_sha256 every history frame seals
        # from it -- is byte-identical to the one it had before the key existed.
        if self.lspgrad == 1:
            record["lspgrad"] = 1
            record["lspgrad_authority"] = CM1_LSPGRAD_AUTHORITY
        # The same rule for the forcing arms: a run without one publishes no key.
        if self.initiation_forcing is not None:
            record["initiation_forcing"] = self.initiation_forcing.record()
        return record

    @property
    def reflectivity_level_index(self) -> int | None:
        """The zero-based model level of the declared reflectivity plane."""

        if not self.output_dbz or self.dbz_level_m is None:
            return None
        return reflectivity_level_index(self.grid, self.dbz_level_m)

    @property
    def numq(self) -> int:
        """CM1's ``numq`` for the bound component (``src/param.F``)."""

        return self.profile.numq

    def physics_families_record(self) -> dict[str, object]:
        """CM1's radiation, PBL and cumulus families, and what was not checked.

        Every selected family that GM1 has not measured against an oracle
        appears in ``unvalidated_physics``.  An empty list is a claim -- it
        says every selected family in this run has been checked -- so the
        list is always published, even when it is empty.
        """

        from .arwen_physics_component import (
            resolve_component,
            unvalidated_markers,
        )
        from .cm1_physics_families import (
            Cm1CumulusSelection,
            Cm1TurbulenceScale,
            RadiationClock,
        )
        from .cuda_gm1_two_part_sgs import TwoPartSelection

        scale = Cm1TurbulenceScale(
            cm1setup=self.cm1setup, sgsmodel=1, ipbl=self.ipbl
        )
        cumulus = Cm1CumulusSelection(
            cuparam=self.cuparam, dx_m=self.grid.dx_m, dy_m=self.grid.dy_m
        )
        record: dict[str, object] = {
            "turbulence_scale": scale.record(),
            "radiation": self.radiation.record(),
            "cumulus": cumulus.record(),
            # sgsmodel=1 publishes no key, so every earlier record is unchanged.
            **(
                {"two_part_sgs": TwoPartSelection(sgsmodel=self.sgsmodel, t2p_avg=self.t2p_avg).record()}
                if self.sgsmodel != 1
                else {}
            ),
            # ramp_sgs=0 publishes no key either.
            **(
                {"sgs_startup_ramp": {
                    "ramp_sgs": self.ramp_sgs,
                    "ramp_time_s": self.ramp_time_s,
                    "authority": "CM1 r21.1 src/turb.F:240-262, src/param.F:750, README.namelist:1081-1087",
                }}
                if self.ramp_sgs != 0
                else {}
            ),
            "rterm": self.rterm,
            "rterm_authority": (
                "CM1 r21.1 src/solve1.F:1003-1054 (Rotunno-Emanuel relaxation, "
                "capped at 2 K/day), README.namelist:555-558"
            ),
            "physics_component_dispositions": {
                "radiation": resolve_component(
                    "radiation", self.radiation.radopt
                ).record(),
                "pbl": resolve_component("pbl", self.ipbl).record(),
                "cumulus": resolve_component("cumulus", self.cuparam).record(),
            },
            "unvalidated_physics": unvalidated_markers(
                radopt=self.radiation.radopt,
                ipbl=self.ipbl,
                cuparam=self.cuparam,
                rterm=self.rterm,
            ),
        }
        if self.radiation.active:
            clock = RadiationClock(
                dtrad_s=self.radiation.dtrad_s, dt_s=self.dt_s
            )
            record["radiation_cadence"] = {
                "dtrad_s": self.radiation.dtrad_s,
                "large_steps_between_calls": clock.steps_between_calls(),
                "first_call_step": 1,
                "authority": "CM1 r21.1 src/cm1.F:3196-3199,3691-3713",
            }
        return record

    @property
    def imove(self) -> int:
        """CM1's ``imove`` for this configuration."""

        return int(self.u_move_m_s != 0.0 or self.v_move_m_s != 0.0)


@dataclass(frozen=True, slots=True)
class HistoryCadence:
    """Exact large-step arithmetic for one bounded integration and its frames.

    Every quantity is a whole number of large steps.  A fixed-timestep clock
    cannot place an event between two steps, so a duration or interval that is
    not an exact multiple of ``dt_s`` is refused here, on the host, rather than
    silently rounded into a model time nobody asked for.
    """

    dt_s: float
    duration_s: float
    interval_s: float
    total_steps: int
    steps_per_frame: int
    frame_count: int

    def frame_steps(self) -> tuple[int, ...]:
        """The completed-step number each history frame is taken at."""

        return tuple(
            self.steps_per_frame * index for index in range(1, self.frame_count + 1)
        )

    def frame_model_seconds(self) -> tuple[float, ...]:
        return tuple(step * self.dt_s for step in self.frame_steps())

    def record(self) -> dict[str, object]:
        return {
            "dt_s": self.dt_s,
            "duration_s": self.duration_s,
            "history_interval_s": self.interval_s,
            "total_steps": self.total_steps,
            "steps_per_frame": self.steps_per_frame,
            "frame_count": self.frame_count,
            "first_frame_model_seconds": self.interval_s,
            "last_frame_model_seconds": self.duration_s,
        }


def _exact_step_count(name: str, interval_s: object, dt_s: float) -> int:
    """One event interval expressed as a whole number of large steps."""

    if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError(f"{name} must be finite and positive")
    steps = interval_s / dt_s
    if not math.isfinite(steps) or steps < 1.0 or not float(steps).is_integer():
        raise ValueError(
            f"{name}={interval_s} is not a whole number of dt_s={dt_s} large "
            "steps; the fixed-timestep clock would place this event between two "
            "completed steps, where no publishable model state exists"
        )
    return int(steps)


def history_cadence(
    *, dt_s: float, duration_s: float, interval_s: float, adaptive: bool = False
) -> HistoryCadence:
    """Admit a requested integration length and frame interval, or refuse.

    CM1's own idealized deck runs ``timax`` seconds and writes every
    ``tapfrq``.  This reproduces that arithmetic exactly: frames land at
    ``interval_s, 2*interval_s, ... duration_s``, and the final frame is the
    completed state at ``duration_s``.  An interval that does not divide the
    duration is refused, because the run's last completed state would then
    never be published under the cadence the caller declared.
    """

    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)):
        raise TypeError("dt_s must be numeric")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if adaptive:
        # misclibs.F:659-681 ramps the step down so it lands exactly on the
        # next output time, so under adapt_dt the frame times are the whole
        # cadence: nothing has to be a whole number of large steps, and the
        # step count is not known until the run has taken them.
        for name, value in (("duration_s", duration_s), ("interval_s", interval_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        frames = duration_s / interval_s
        if not frames.is_integer() or frames < 1:
            raise ValueError(
                f"duration_s={duration_s} is not a whole number of "
                f"history_interval_s={interval_s}; the completed state of this "
                "run would never be published"
            )
        return HistoryCadence(
            float(dt_s), float(duration_s), float(interval_s), 0, 0, int(frames)
        )
    total_steps = _exact_step_count("duration_s", duration_s, dt_s)
    steps_per_frame = _exact_step_count("history_interval_s", interval_s, dt_s)
    if steps_per_frame > total_steps:
        raise ValueError(
            f"history_interval_s={interval_s} is {steps_per_frame} large steps "
            f"but duration_s={duration_s} is only {total_steps}; the run would "
            "end before its first frame and publish nothing"
        )
    if total_steps % steps_per_frame:
        raise ValueError(
            f"duration_s={duration_s} is {total_steps} large steps and "
            f"history_interval_s={interval_s} is {steps_per_frame}; the frames "
            "would stop short of the requested duration, so the completed state "
            "of this run would never be published"
        )
    return HistoryCadence(
        float(dt_s),
        float(duration_s),
        float(interval_s),
        total_steps,
        steps_per_frame,
        total_steps // steps_per_frame,
    )


# CM1 r21.1 cm1.F:3186 fires a restart when the model clock reaches `rsttim`,
# and cm1.F:4247-4252 advances `rsttim` by `rstfrq` until it is ahead of the
# model time again.  `rstfrq` is README.namelist:103-104's restart frequency in
# seconds.  This lane's clock is a fixed number of whole large steps, so the
# same schedule is exact step arithmetic rather than a floating comparison.
RESTART_CADENCE_AUTHORITY = "CM1 r21.1 cm1.F:3186,4247-4252 (rstfrq/rsttim)"
# README.namelist:1488-1510.  `.true.` -- CM1's own default -- puts the next
# output/stat/restart time at the resume time plus the namelist interval;
# `.false.` keeps whatever time the restart file carried, which is the option
# README.namelist:1508-1510 names as the one that may be needed for
# bit-identical restarts.  cm1.F:2711-2715 is the branch itself.
RESTART_RESET_FRQTIM_AUTHORITY = "CM1 r21.1 cm1.F:2711-2715, README.namelist:1488-1510 (restart_reset_frqtim)"
# cm1.F:301 starts the restart-file counter at one and cm1.F:4310 increments
# it after each write; README.restart names the files cm1rst_000001_*.dat and
# says "it starts at 1, and counts upward".
FIRST_RESTART_NUMBER = 1
# cm1.F:2660 and README.namelist:227.  0 = start from the initial condition,
# 1 = read a restart file.
IRST_VALUES = (0, 1)
# README.namelist:233-235 and cm1.F:4341-4350: build the initial condition and
# stop without integrating.
ICONLY_VALUES = (0, 1)


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """One completed large step at which this run publishes something."""

    step: int
    model_time_s: float
    history: bool
    restart: bool

    def record(self) -> dict[str, object]:
        return {
            "step": self.step,
            "model_time_s": self.model_time_s,
            "history": self.history,
            "restart": self.restart,
        }


@dataclass(frozen=True, slots=True)
class RunSchedule:
    """Every history and restart event of one bounded integration.

    CM1 keeps three independent clocks -- `taptim`, `rsttim` and `stattim` --
    and fires whichever the model time has reached (cm1.F:3180-3190).  This is
    the same thing with the floating comparison removed: both cadences are a
    whole number of large steps, so the events are computed once, before the
    first step, and the run cannot drift off them.
    """

    cadence: HistoryCadence
    start_step: int
    steps_per_restart: int | None
    restart_interval_s: float | None
    reset_frqtim: bool
    first_history_index: int
    first_restart_number: int
    events: tuple[ScheduledEvent, ...]

    @property
    def final_step(self) -> int:
        return self.cadence.total_steps

    @property
    def history_count(self) -> int:
        return sum(1 for event in self.events if event.history)

    @property
    def restart_count(self) -> int:
        return sum(1 for event in self.events if event.restart)

    def record(self) -> dict[str, object]:
        return {
            "schema": "gm1.cartesian-run-schedule/v1",
            "cadence": self.cadence.record(),
            "start_step": self.start_step,
            "start_model_time_s": self.start_step * self.cadence.dt_s,
            "restart_interval_s": self.restart_interval_s,
            "steps_per_restart": self.steps_per_restart,
            "restart_reset_frqtim": self.reset_frqtim,
            "restart_cadence_authority": RESTART_CADENCE_AUTHORITY,
            "restart_reset_frqtim_authority": RESTART_RESET_FRQTIM_AUTHORITY,
            "first_history_index": self.first_history_index,
            "first_restart_number": self.first_restart_number,
            "history_count": self.history_count,
            "restart_count": self.restart_count,
            "event_count": len(self.events),
            "first_event": self.events[0].record() if self.events else None,
            "last_event": self.events[-1].record() if self.events else None,
        }


def _event_steps(
    name: str, steps_per_event: int, start_step: int, total_steps: int, reset: bool
) -> tuple[int, ...]:
    """The completed-step numbers one cadence fires at, after ``start_step``.

    ``reset`` is CM1's ``restart_reset_frqtim`` (cm1.F:2711-2715).  With it
    the first event is ``start_step`` plus the interval, which is the resume
    time plus the namelist value; without it the events stay on the absolute
    grid the original run declared, so a resumed run publishes at the same
    model times an uninterrupted one does.
    """

    if reset:
        first = start_step + steps_per_event
    else:
        first = (start_step // steps_per_event + 1) * steps_per_event
    steps = tuple(range(first, total_steps + 1, steps_per_event))
    if not steps or steps[-1] != total_steps:
        raise ValueError(
            f"{name} fires at {steps[-1] if steps else 'no'} completed steps before "
            f"the run ends at {total_steps}; the {total_steps - start_step} large "
            f"steps remaining after step {start_step} are not a whole number of "
            f"{steps_per_event}-step intervals"
            + (
                ", because restart_reset_frqtim moved the schedule onto the resume "
                "time (cm1.F:2711-2715)"
                if reset
                else ""
            )
            + ", so this run's own completed state would never be published"
        )
    return steps


def run_schedule(
    *,
    dt_s: float,
    duration_s: float,
    history_interval_s: float,
    restart_interval_s: float | None = None,
    start_step: int = 0,
    reset_frqtim: bool = False,
    first_history_index: int = 1,
    first_restart_number: int = FIRST_RESTART_NUMBER,
) -> RunSchedule:
    """Admit one bounded integration's complete publication schedule.

    ``duration_s`` is absolute model time from the start of the simulation,
    exactly as CM1's ``timax`` is, so a run resumed at ``start_step`` steps
    integrates the remainder rather than the whole thing again.
    """

    cadence = history_cadence(
        dt_s=dt_s, duration_s=duration_s, interval_s=history_interval_s
    )
    if type(start_step) is not int or start_step < 0:
        raise ValueError("start_step must be a nonnegative integer")
    if start_step >= cadence.total_steps:
        raise ValueError(
            f"the run resumes at {start_step} completed large steps and "
            f"duration_s={duration_s} is {cadence.total_steps}; there is nothing "
            "left to integrate, and publishing a schedule here would republish "
            "frames of a model time that has already been written"
        )
    for name, value in (
        ("first_history_index", first_history_index),
        ("first_restart_number", first_restart_number),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(reset_frqtim, bool):
        raise TypeError("reset_frqtim must be a bool")
    history_steps = _event_steps(
        "history_interval_s", cadence.steps_per_frame, start_step,
        cadence.total_steps, reset_frqtim,
    )
    if restart_interval_s is None:
        steps_per_restart, restart_steps = None, ()
    else:
        steps_per_restart = _exact_step_count(
            "restart_interval_s", restart_interval_s, dt_s
        )
        restart_steps = _event_steps(
            "restart_interval_s", steps_per_restart, start_step,
            cadence.total_steps, reset_frqtim,
        )
    ordered = sorted(set(history_steps) | set(restart_steps))
    events = tuple(
        ScheduledEvent(
            step, step * float(dt_s), step in set(history_steps), step in set(restart_steps)
        )
        for step in ordered
    )
    return RunSchedule(
        cadence, start_step, steps_per_restart,
        None if restart_interval_s is None else float(restart_interval_s),
        reset_frqtim, first_history_index, first_restart_number, events,
    )


@dataclass(frozen=True, slots=True)
class ScheduledOperation:
    phase: StepPhase
    nrk: int | None


@functools.lru_cache(maxsize=1)
def large_step_schedule() -> tuple[ScheduledOperation, ...]:
    """CM1 r21.1 solve1 -> solve2 -> mp_driver -> solve3 chronology.

    The last-stage mass reduction straddles momentum boundaries and TKE.
    Moving it wholesale to the microphysics boundary would change the
    density read by TKE, even if the final pressure correction looked right.
    """

    order = [
        ScheduledOperation(StepPhase.CLOSURE, None),
        ScheduledOperation(StepPhase.HELD_TENDENCIES, None),
    ]
    for nrk in (1, 2, 3):
        order.extend(ScheduledOperation(phase, nrk) for phase in (
            StepPhase.PRE_ACOUSTIC,
            StepPhase.ACOUSTIC,
            StepPhase.POST_ACOUSTIC,
            StepPhase.SCALAR_MOIST,
        ))
        if nrk == 3:
            order.extend((
                ScheduledOperation(StepPhase.REMOVE_HELD_EOS, nrk),
                ScheduledOperation(StepPhase.MASS_BEGIN, nrk),
            ))
        order.append(ScheduledOperation(StepPhase.MOMENTUM_BOUNDARIES, nrk))
        if nrk < 3:
            # solve2.F:1682-1706 fills rho/pressure/theta/moisture only
            # between RK stages.  The final scalar fill belongs to solve3.
            order.append(ScheduledOperation(StepPhase.SCALAR_BOUNDARIES, nrk))
        order.append(ScheduledOperation(StepPhase.TKE, nrk))
        if nrk == 3:
            order.append(ScheduledOperation(StepPhase.MASS_FINISH, nrk))
    order.extend(ScheduledOperation(phase, None) for phase in (
        StepPhase.PDEFQ,
        StepPhase.MICROPHYSICS,
        StepPhase.POST_PHYSICS,
        StepPhase.SOLVE3,
    ))
    return tuple(order)


class CartesianBackend(Protocol):
    def execute(self, operation: ScheduledOperation, *, step: int) -> None: ...

    def synchronize(self) -> None: ...


@dataclass(slots=True)
class RuntimeClock:
    """Publish a completed model clock only after the entire step succeeds.

    The clock owns CM1's two moving-frame integrals as well as its model time.
    ``cm1.F:3510-3513`` steps ``mtime`` forward and then advances
    ``domainlocx``/``domainlocy`` by ``dt*umove`` and ``dt*vmove`` in the same
    breath, and nothing else in CM1 touches either, so the same three lines
    live here in the same order.

    ``domainlocx``/``domainlocy`` are where the domain is, in ground metres,
    at the model time this clock reads.  A translating frame buys a small
    domain by moving the grid with the storm, and the moment the first large
    step completes the grid's own coordinates stop being ground coordinates.
    Without this integral the run publishes carriers with no way to say what
    patch of ground they cover.
    """

    config: RuntimeConfig
    completed_steps: int = 0
    failed: bool = False
    # cm1.F:266 declares both double precision; cm1.F:348-349 start them at
    # zero.  Python floats are binary64, which is that declaration exactly.
    domainlocx_m: float = DOMAIN_LOCATION_INITIAL_M
    domainlocy_m: float = DOMAIN_LOCATION_INITIAL_M
    # cm1.F:3510 ``mtime = mtime + dbldt``.  With a fixed step this is exactly
    # ``completed_steps*dt_s`` and the property below still returns that; with
    # adapt_dt it is the only correct answer, because no two steps are the
    # same length.  Kept as a running binary64 sum, as CM1 keeps ``mtime``.
    accumulated_model_s: float = 0.0
    # CM1's ``umove``/``vmove`` are namelist constants until param.F:704-717
    # makes them model variables.  With ``do_adapt_move`` selected the
    # controller owns the frame velocity *and* the cm1.F:3512-3513 integral --
    # it advances the integral with the velocity as it stood before its own
    # update, which is exactly cm1.F's order -- so this clock defers to it
    # rather than accumulating a constant that is no longer constant.  Left
    # unset, every word below is exactly what it was before the moving domain
    # could be adaptive.
    adaptive_frame: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, RuntimeConfig):
            raise TypeError("config must be RuntimeConfig")
        if type(self.completed_steps) is not int or self.completed_steps < 0:
            raise ValueError("completed_steps must be a nonnegative integer")
        for name in ("domainlocx_m", "domainlocy_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            object.__setattr__(self, name, float(value))
            if not math.isfinite(getattr(self, name)):
                raise ValueError(
                    f"{name} must be finite; cm1.F:3512 adds to it once per large "
                    "step, so a non-finite start makes every domain location this "
                    "run publishes non-finite and no consumer can place one frame "
                    "on the ground"
                )
        if self.adaptive_frame is None:
            self._require_exact_domain_location("restored")

    @classmethod
    def resume(
        cls, config: RuntimeConfig, *, completed_steps: int,
        domainlocx_m: float, domainlocy_m: float,
        adaptive_frame: Any = None,
    ) -> "RuntimeClock":
        """Continue a persisted integral: ``restart_read.F:419-420``.

        CM1 declares the pair ``intent(inout)`` (``restart_read.F:151``) and
        overwrites it from the restart file, so a resumed run continues the
        integral.  An accumulator that restarted at zero here would mislocate
        every frame after the restart by the whole distance the domain had
        already travelled, and nothing in the artifact would say so.
        """

        if config.adaptive_timestep:
            # misclibs.F:785 changes dt between steps, so the completed model
            # time is a running sum no step count can reconstruct: resuming
            # from one would put every frame after the restart at the wrong
            # model time, and the history would say the run reached a time it
            # never did.  CM1 carries mtime in its own restart record; this
            # port's checkpoint does not, so the resume is refused rather than
            # guessed.
            raise ValueError(
                "a run with dynamics.adaptive_timestep cannot be resumed from "
                f"a step count: {completed_steps} completed steps of a varying "
                "dt do not determine the model time, and cm1.F:3510's mtime is "
                "not in this port's checkpoint.  Re-run from the start, or set "
                "dynamics.adaptive_timestep = false"
            )
        return cls(
            config, completed_steps=completed_steps, failed=False,
            domainlocx_m=domainlocx_m, domainlocy_m=domainlocy_m,
            accumulated_model_s=completed_steps * config.dt_s,
            adaptive_frame=adaptive_frame,
        )

    @property
    def model_seconds(self) -> float:
        """cm1.F:3510's ``mtime``.

        For a fixed step this is ``completed_steps*dt_s`` exactly, which is
        what it has always returned and what the domain-location proof below
        relies on.  Under ``adapt_dt`` the steps differ in length, so the
        running sum is the model time and the product is meaningless.
        """

        if self.config.adaptive_timestep:
            return self.accumulated_model_s
        return self.completed_steps * self.config.dt_s

    @property
    def domainlocx_increment_m(self) -> float:
        """``dbldt*dble(umove)``: cm1.F:3512's per-step increment."""

        return self.config.dt_s * cm1_real(self.config.u_move_m_s)

    @property
    def domainlocy_increment_m(self) -> float:
        """``dbldt*dble(vmove)``: cm1.F:3513's per-step increment."""

        return self.config.dt_s * cm1_real(self.config.v_move_m_s)

    def _require_exact_domain_location(self, stage: str) -> None:
        """Refuse the moment the running sum stops being ``n*dt*umove``.

        ``domain_location.rs`` proves the sum is exact while ``dt`` is a power
        of two, the frame speed fits binary32 and the run is shorter than
        ``DOMAIN_LOCATION_EXACT_STEP_LIMIT`` steps -- every partial sum is
        then representable in binary64 and no addition rounds.  This is that
        proof as a check, so a widened timestep or a longer run stops the run
        instead of publishing a ground position that has drifted away from
        ``umove*t`` by an amount that grows for the rest of the integration.
        """

        if self.config.adaptive_timestep:
            # misclibs.F:785 changes dt between large steps, so cm1.F:3512's
            # running sum is no longer n*dt*umove and the exactness argument
            # of domain_location.rs does not apply.  CM1 keeps the same
            # running sum; so does this.  The guard is retired for this branch
            # rather than made to fire on a configuration it cannot judge.
            for name, value in (
                (DOMAIN_LOCATION_X_NAME, self.domainlocx_m),
                (DOMAIN_LOCATION_Y_NAME, self.domainlocy_m),
            ):
                if not math.isfinite(value):
                    raise ValueError(f"{stage} {name} is not finite")
            return
        for name, seen, increment in (
            (DOMAIN_LOCATION_X_NAME, self.domainlocx_m, self.domainlocx_increment_m),
            (DOMAIN_LOCATION_Y_NAME, self.domainlocy_m, self.domainlocy_increment_m),
        ):
            exact = self.completed_steps * increment
            if seen != exact:
                raise ValueError(
                    f"{stage} {name} is {seen!r} m after {self.completed_steps} "
                    f"completed large steps of {self.config.dt_s} s at {increment!r} m "
                    f"per step, but the exact integral is {exact!r} m; cm1.F:3512's "
                    "running sum is exact only while dt is a power of two, the frame "
                    "speed fits binary32 and the run is shorter than "
                    f"{DOMAIN_LOCATION_EXACT_STEP_LIMIT} steps, and outside that every "
                    "frame from here on is placed on the wrong patch of ground"
                )

    def domain_location_record(self) -> dict[str, object]:
        """CM1's moving-frame scalars for this model time, by CM1's names."""

        # param.F:711 raises imove for every do_adapt_move run, and the
        # velocity a reader needs to recover a ground-relative wind is the one
        # the controller holds at this model time, not the zero param.F:706-707
        # leaves in the namelist.
        adaptive = self.adaptive_frame
        return {
            "authority": DOMAIN_LOCATION_AUTHORITY,
            "cadence": DOMAIN_LOCATION_CADENCE,
            "completed_steps": self.completed_steps,
            "model_time_s": self.model_seconds,
            "imove": self.config.imove if adaptive is None else 1,
            MOVE_SPEED_X_NAME: cm1_real(
                self.config.u_move_m_s if adaptive is None else adaptive.u_move_m_s
            ),
            MOVE_SPEED_Y_NAME: cm1_real(
                self.config.v_move_m_s if adaptive is None else adaptive.v_move_m_s
            ),
            DOMAIN_LOCATION_X_NAME: self.domainlocx_m,
            DOMAIN_LOCATION_Y_NAME: self.domainlocy_m,
            "units": {
                MOVE_SPEED_X_NAME: MOVE_SPEED_UNITS,
                MOVE_SPEED_Y_NAME: MOVE_SPEED_UNITS,
                DOMAIN_LOCATION_X_NAME: DOMAIN_LOCATION_UNITS,
                DOMAIN_LOCATION_Y_NAME: DOMAIN_LOCATION_UNITS,
            },
            "descriptions": dict(DOMAIN_LOCATION_DESCRIPTIONS),
        }

    def step(self, backend: CartesianBackend) -> None:
        if self.failed:
            raise RuntimeError("a failed CUDA step requires restoring a completed state")
        next_step = self.completed_steps + 1
        try:
            # --cuda-graphs on/verify (cm1core.cuda_graphs): the same schedule,
            # captured once and replayed where the replay is proven the same.
            graphs = getattr(backend, "schedule_graphs", None)
            if graphs is not None:
                graphs.run_step(backend, next_step, large_step_schedule())
            else:
                for operation in large_step_schedule():
                    backend.execute(operation, step=next_step)
                backend.synchronize()
        except BaseException:
            self.failed = True
            raise
        # cm1.F:3510:  mtime = mtime + dbldt
        self.completed_steps = next_step
        self.accumulated_model_s += self.config.dt_s
        # cm1.F:3512-3513, in CM1's order: the integral advances with the
        # model time, before this step's history, statistics or restart is
        # written, so the value published at step n is the integral over n
        # completed steps and not n-1.
        if self.adaptive_frame is None:
            self.domainlocx_m += self.domainlocx_increment_m
            self.domainlocy_m += self.domainlocy_increment_m
            self._require_exact_domain_location("accumulated")
            return
        # cm1.F:3512-3513 and then cm1.F:3641-3679, in CM1's order: the
        # controller advances the same double-precision integral with the
        # velocity as it stood before its own update and only then updates the
        # velocity, so the pair read back here is the one CM1 publishes for
        # this completed step.  The exactness proof above is not claimed and
        # not applied: the frame velocity is no longer a constant, so the
        # running sum is not n*dt*umove and there is nothing to check it
        # against but the controller's own binary64 accumulator.
        try:
            self.adaptive_frame.after_large_step()
            self.domainlocx_m = float(self.adaptive_frame.domainlocx_m)
            self.domainlocy_m = float(self.adaptive_frame.domainlocy_m)
        except BaseException:
            # The carriers have been advanced and the controller has not; the
            # runtime holds half a completed step and stepping it again would
            # rebase winds that were never rebased.
            self.failed = True
            raise
        for name, value in (
            (DOMAIN_LOCATION_X_NAME, self.domainlocx_m),
            (DOMAIN_LOCATION_Y_NAME, self.domainlocy_m),
        ):
            if not math.isfinite(value):
                self.failed = True
                raise ValueError(
                    f"the adaptive moving-domain controller published a "
                    f"non-finite {name} at completed step {self.completed_steps}; "
                    "every frame from here on would be placed nowhere on the "
                    "ground"
                )
