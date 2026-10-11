"""CM1 r21.1's adaptive large step, transcribed from ``misclibs.F``.

This is the controller of ``getnewdt`` (``src/misclibs.F:593-810``) and the
first-step selection of ``get_adapt_dt`` (``src/cm1.F:3013-3048``), as pure
host arithmetic over two measured numbers: ``cflmax``, the advective Courant
number of ``calccflquick`` (``src/misclibs.F:2726-2814``), and ``ksmax``, the
diffusion number of ``calcksquick`` (``src/misclibs.F:2822-2901``).

Fifteen of CM1 r21.1's twenty-three shipped configuration decks set
``adapt_dt = 1``.  Without this the large step is whatever the document
declared for the whole run: a storm that develops a 90 m/s updraft is
integrated at the step a quiescent sounding needed, and a run that could have
taken a step ten times larger for its first hour pays for the step its worst
minute will need.

Nothing here touches a device or a clock.  The controller is arithmetic so it
can be checked against CM1's own source line by line, and so the one place
that decides the next large step is the one place a test has to read.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass


CM1_ADAPTIVE_TIMESTEP_AUTHORITY = (
    "CM1 r21.1 src/misclibs.F:593-810 (getnewdt), src/cm1.F:3013-3048 "
    "(get_adapt_dt first step), src/misclibs.F:2726-2814 (calccflquick), "
    "src/misclibs.F:2822-2901 (calcksquick), src/misclibs.F:818-915 (dtsmall)"
)

#: ``misclibs.F:616``: ``real, parameter :: cfl_limit = 1.00``.  The comment
#: beside it in CM1's own source calls it "actually a target value".
CM1_CFL_LIMIT = 1.00
#: ``misclibs.F:617``: ``real, parameter :: max_change = 0.10``.  A step may
#: grow by at most ten percent per large step; it may shrink without limit.
CM1_MAX_CHANGE = 0.10
#: ``misclibs.F:619-628``.  The diffusion-number ceiling, and it is halved in
#: three dimensions because two horizontal directions share it.  CM1 applies
#: this branch only for ``cm1setup >= 1``; GM1's Cartesian lane is
#: ``cm1setup = 1`` (LES with an explicit sgsmodel), exactly as CM1's own
#: supercell deck is (``run/config_files/supercell/namelist.input:27``).
CM1_KS_LIMIT_3D = 0.18 / 2.0
CM1_KS_LIMIT_2D = 0.18
#: ``misclibs.F:637`` floors the measured Courant number before dividing.
CM1_CFLMAX_FLOOR = 1.0e-10
#: ``misclibs.F:650`` floors the measured diffusion number the same way.
CM1_KSMAX_FLOOR = 1.0e-10
#: ``misclibs.F:654``: a step never exceeds twice the declared ``dtl``.
CM1_MAX_DT_MULTIPLE_OF_DTL = 2.0
#: ``misclibs.F:669,673`` and every ramp-down block: the ``2.00001`` and
#: ``1.00001`` guards that stop a floating-point tie from splitting one step
#: into two.
_RAMP_THREE = 2.00001
_RAMP_TWO = 1.00001


def _real(value: float) -> float:
    """``value`` rounded to CM1's default REAL (binary32), returned as a Python float.

    ``getnewdt`` and ``dtsmall`` keep only ``dbldt``, ``mtime`` and the event
    times in double precision; ``dt``, ``dtlast``, ``cflmax``, ``ksmax``,
    ``ks_limit``, ``dtsm``, ``cfl_limit`` and ``max_change`` are default REAL
    (misclibs.F:605-617, 829), so every product or quotient of two of them is
    rounded to binary32 before it meets ``dbldt``.  One IEEE operation on two
    binary32 operands done in binary64 and rounded once to binary32 is the
    binary32 result (binary64 carries more than 2*24+2 bits).
    """

    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _nint(value: float) -> int:
    """Fortran ``NINT`` of a positive double: nearest, halves away from zero."""

    whole = math.floor(value)
    return int(whole) + (1 if value - whole >= 0.5 else 0)


#: ``misclibs.F:617``'s ``1.0+max_change`` in default REAL: 1.10000002384185791015625.
_REAL_GROWTH_CAP = _real(_real(1.0) + _real(0.10))


#: ``misclibs.F:835``: CM1's own ceiling on the small-step count.
CM1_MAX_NSOUND = 192
#: ``misclibs.F:883``: and its floor.
CM1_MIN_NSOUND = 4


def cm1_dtsmall(
    *,
    dt_s: float,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    psolver: int = 3,
    csound_m_s: float = 300.0,
    adapt_dt: bool = False,
) -> tuple[float, int]:
    """``misclibs.F:818-915``: the small step and the count, for any psolver.

    Returns ``(dt_s, nsound)``.  ``dt_s`` is returned because the adaptive
    branch at ``misclibs.F:886-893`` shrinks the large step rather than
    stopping when the count would exceed ``max_nsound``; with ``adapt_dt``
    false that case raises, exactly as ``misclibs.F:894-910`` stops the run.

    CM1 r19 and later never read ``nsound`` from a namelist
    (README.namelist:453-455): it is derived here on every change of ``dt``
    and once more before the first step (``cm1.F:3054``).  A comparison that
    hands the two codes different small-step counts is comparing two
    different acoustic schedules, whatever else it holds fixed.
    """

    for name, value in (("dt_s", dt_s), ("dx_m", dx_m), ("dy_m", dy_m), ("dz_m", dz_m)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("nx", nx), ("ny", ny)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if psolver in (1, 4, 5):
        # misclibs.F:837-840: these solvers take one small step per large step.
        return float(dt_s), 1
    r = _real
    # misclibs.F:829: dtsm is default REAL, and so are min_dx/min_dy/min_dz
    # and csound, so each spelling below is CM1's left-to-right REAL
    # arithmetic: (0.60*min_dx) rounded, then /360.0 rounded.
    if psolver == 2:
        # misclibs.F:844-855.
        if ny == 1:
            dtsm = r(r(r(0.60) * min(r(dx_m), r(dz_m))) / r(360.0))
        elif nx == 1:
            dtsm = r(r(r(0.60) * min(r(dy_m), r(dz_m))) / r(360.0))
        else:
            dtsm = r(r(r(0.50) * min(r(dx_m), r(dy_m), r(dz_m))) / r(360.0))
    elif psolver == 3:
        # misclibs.F:857-868: vertically implicit, so only the horizontal
        # spacings bound the small step.
        if ny == 1:
            dtsm = r(r(r(0.60) * r(dx_m)) / r(360.0))
        elif nx == 1:
            dtsm = r(r(r(0.60) * r(dy_m)) / r(360.0))
        else:
            dtsm = r(r(r(0.60) * min(r(dx_m), r(dy_m))) / r(360.0))
    elif psolver in (6, 7):
        # misclibs.F:869-881: the same shape with csound in place of 360.
        if csound_m_s <= 0.0:
            raise ValueError("csound_m_s must be positive for psolver=6,7")
        if ny == 1:
            dtsm = r(r(r(0.60) * min(r(dx_m), r(dz_m))) / r(csound_m_s))
        elif nx == 1:
            dtsm = r(r(r(0.60) * min(r(dy_m), r(dz_m))) / r(csound_m_s))
        else:
            dtsm = r(r(r(0.50) * min(r(dx_m), r(dy_m), r(dz_m))) / r(csound_m_s))
    else:
        raise ValueError(f"psolver={psolver} is not a CM1 r21.1 pressure solver")
    # dtsmall's second argument is dbldt: under adapt_dt the controller's
    # double; at a fixed step cm1.F:1254's dbldt = dble(dt), the declared
    # step as a default REAL.
    dbldt = float(dt_s) if adapt_dt else r(dt_s)
    # misclibs.F:883-885: nint(dbldt/dtsm) and dbldt/float(nsound) > dtsm are
    # double arithmetic on the REAL dtsm; NINT rounds half away from zero.
    nsound = max(_nint(dbldt / dtsm), CM1_MIN_NSOUND)
    if nsound % 2:
        nsound += 1
    if dbldt / float(nsound) > dtsm:
        nsound += 2
    if nsound > CM1_MAX_NSOUND:
        if not adapt_dt:
            raise ValueError(
                f"dtsmall derives nsound={nsound} for dt={dt_s} s on this grid, "
                f"above CM1's max_nsound={CM1_MAX_NSOUND}; misclibs.F:894-910 "
                "prints 'Time step (dtl) needs to be smaller' and stops the run"
            )
        # misclibs.F:886-893: with adapt_dt the step shrinks instead;
        # nsound*dtsm is INTEGER times REAL, a REAL, before it reaches dbldt.
        nsound = CM1_MAX_NSOUND
        return r(nsound * dtsm), nsound
    return float(dt_s), nsound


def cm1_ks_limit(*, nx: int, ny: int) -> float:
    """``misclibs.F:619-628``: the diffusion-number ceiling for this grid."""

    for name, value in (("nx", nx), ("ny", ny)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return CM1_KS_LIMIT_3D if (nx > 3 and ny > 3) else CM1_KS_LIMIT_2D


def cm1_diffusion_number(
    *,
    dt_s: float,
    dx_m: float,
    dy_m: float,
    max_eddy_coefficient_m2_s: float,
) -> float:
    """``calcksquick`` (``misclibs.F:2841-2890``) on a uniform horizontal grid.

    CM1 forms, for every cell at ``k >= 2``,

        tem1 = sqrt( (kmh*dtdx*uh*uh)**2 + (kmh*dtdy*vh*vh)**2 )

    and the same with ``khh`` and with ``kmv``, then takes the largest of all
    three over the whole domain.  ``dtdx = dt*rdx*rdx`` and
    ``dtdy = dt*rdy*rdy`` (``misclibs.F:2841-2842``).

    On a uniform horizontal grid ``param.F`` sets ``uh = vh = 1`` everywhere,
    so the three expressions differ only in which coefficient multiplies the
    single common factor ``sqrt(dtdx**2 + dtdy**2)``.  The domain maximum is
    therefore exactly that factor times the largest of ``kmh``, ``khh`` and
    ``kmv`` over ``k >= 2`` -- one scalar reduction instead of three, with no
    approximation.  This identity holds only while ``uh = vh = 1``, which is
    the same uniform-horizontal condition every operator in this lane already
    requires.
    """

    for name, value in (
        ("dt_s", dt_s),
        ("dx_m", dx_m),
        ("dy_m", dy_m),
        ("max_eddy_coefficient_m2_s", max_eddy_coefficient_m2_s),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if dt_s <= 0.0 or dx_m <= 0.0 or dy_m <= 0.0:
        raise ValueError("dt_s, dx_m and dy_m must be positive")
    if max_eddy_coefficient_m2_s < 0.0:
        raise ValueError("an eddy diffusivity maximum cannot be negative")
    dtdx = float(dt_s) / (float(dx_m) * float(dx_m))
    dtdy = float(dt_s) / (float(dy_m) * float(dy_m))
    return float(max_eddy_coefficient_m2_s) * math.hypot(dtdx, dtdy)


@dataclass(frozen=True, slots=True)
class RampDownTarget:
    """One scheduled event ``getnewdt`` will land a step exactly on.

    ``name`` is CM1's own clock name.  ``next_time_s`` is CM1's ``taptim``/
    ``rsttim``: the model time of the next such event.  ``frequency_s`` is
    its ``tapfrq``/``rstfrq``, added when the event is due at this very step
    because CM1 has not advanced its own target yet (``misclibs.F:661-664``).
    """

    name: str
    next_time_s: float
    frequency_s: float
    due_now: bool


def _ramp_down(dbldt: float, model_time_s: float, target: RampDownTarget) -> float:
    """One ``getnewdt`` ramp-down block, ``misclibs.F:659-681``."""

    if target.next_time_s <= 0.0:
        return dbldt
    remaining = target.next_time_s - model_time_s
    if target.due_now:
        remaining += target.frequency_s
    if _RAMP_THREE * dbldt < remaining <= 3.0 * dbldt:
        return remaining / 3.0
    if _RAMP_TWO * dbldt < remaining <= 2.0 * dbldt:
        return remaining / 2.0
    if remaining <= dbldt:
        return remaining
    return dbldt


@dataclass(frozen=True, slots=True)
class AdaptiveTimestepDecision:
    """What ``getnewdt`` chose, and every number it chose it from."""

    dt_s: float
    previous_dt_s: float
    cflmax_in: float
    ksmax_in: float
    cflmax_scaled: float
    ksmax_scaled: float
    limited_by: str

    def record(self) -> dict[str, object]:
        return {
            "authority": CM1_ADAPTIVE_TIMESTEP_AUTHORITY,
            "dt_s": self.dt_s,
            "previous_dt_s": self.previous_dt_s,
            "cflmax": self.cflmax_in,
            "ksmax": self.ksmax_in,
            "cflmax_at_new_dt": self.cflmax_scaled,
            "ksmax_at_new_dt": self.ksmax_scaled,
            "limited_by": self.limited_by,
            "cfl_limit": CM1_CFL_LIMIT,
            "ks_limit_applied": self.ksmax_scaled >= 0.0,
        }


def cm1_getnewdt(
    *,
    dt_s: float,
    cflmax: float,
    ksmax: float,
    dtl_s: float,
    ks_limit: float,
    model_time_s: float = 0.0,
    ramp_down_targets: Sequence[RampDownTarget] = (),
    apply_ramp_down: bool = True,
) -> AdaptiveTimestepDecision:
    """``misclibs.F:593-810``: the next large step, from this one's numbers.

    ``dt_s`` is the step just completed (CM1's double ``dbldt``); ``cflmax``
    and ``ksmax`` are the two numbers measured on the state it produced, at
    that step.  CM1 carries the working step in a double precision ``dbldt``
    and assigns the default-REAL ``dt`` from it (``misclibs.F:785``), but the
    factors it multiplies ``dbldt`` by are default REAL (``cfl_limit``,
    ``max_change``, ``cflmax``, ``ksmax``, ``ks_limit``, ``dt``), so each is
    rounded to binary32 here before it meets the double, as gfortran does.
    A growth-capped step is ``dbldt*1.10000002384185791015625``, not
    ``dbldt*1.1``: in binary64 the two differ by 2e-8 of the step, which
    moves every later step's start and so which step a forcing window's edge
    falls in (CM1's own steps show it on a forcing
    window's edge).

    ``apply_ramp_down`` is CM1's ``doinit``: the first call, from
    ``cm1.F:3039`` with ``.true.``, skips every scheduled-event block because
    no event has a target yet (``misclibs.F:658``).
    """

    for name, value in (
        ("dt_s", dt_s),
        ("cflmax", cflmax),
        ("ksmax", ksmax),
        ("dtl_s", dtl_s),
        ("ks_limit", ks_limit),
        ("model_time_s", model_time_s),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if dt_s <= 0.0 or dtl_s <= 0.0:
        raise ValueError("dt_s and dtl_s must be positive")
    if ks_limit <= 0.0:
        raise ValueError("ks_limit must be positive")
    if ksmax < 0.0:
        raise ValueError("ksmax cannot be negative")

    r = _real
    # dbldt enters as the double of the step just taken; dt is its default
    # REAL (misclibs.F:785 ``dt = dbldt``), and cflmax, ksmax, ks_limit,
    # cfl_limit and max_change are default REAL too (misclibs.F:605-617).
    dbldt = float(dt_s)
    dt_real = r(dt_s)
    # misclibs.F:637.  A domain at rest would otherwise divide by zero.
    measured_cfl = max(r(cflmax), r(CM1_CFLMAX_FLOOR))
    ratio_real = r(r(CM1_CFL_LIMIT) / measured_cfl)
    limited_by = "cfl"
    if measured_cfl > r(CM1_CFL_LIMIT):
        # misclibs.F:639-641: decrease, in proportion: cfl_limit/cflmax is a
        # REAL quotient, then promoted to multiply the double dbldt.
        dbldt = dbldt * ratio_real
    else:
        # misclibs.F:642-644: increase, but never by more than max_change.
        # min(1.0+max_change, cfl_limit/cflmax) is REAL: the cap is
        # 1.10000002384185791015625, not binary64's 1.1, so a capped step is
        # CM1's word only when multiplied by that.
        dbldt = dbldt * min(_REAL_GROWTH_CAP, ratio_real)
        limited_by = "cfl-growth-cap" if _REAL_GROWTH_CAP <= ratio_real else "cfl"
    # misclibs.F:647-650.  ksmax was measured at dt, so ksmax*dbldt/dt is the
    # diffusion number the candidate step would have: REAL ksmax times the
    # double dbldt, over the REAL dt, against the REAL ks_limit, all in
    # double; the fallback dt*ks_limit/max(1.0e-10,ksmax) is REAL throughout.
    ks_real = r(ksmax)
    ks_limit_real = r(ks_limit)
    if (ks_real * dbldt / dt_real) > ks_limit_real:
        candidate = r(r(dt_real * ks_limit_real) / max(r(CM1_KSMAX_FLOOR), ks_real))
        if candidate < dbldt:
            dbldt = candidate
            limited_by = "diffusion"
    # misclibs.F:654: never more than twice the declared dtl, dble(2.0*dtl).
    ceiling = r(r(CM1_MAX_DT_MULTIPLE_OF_DTL) * r(dtl_s))
    if dbldt > ceiling:
        dbldt = ceiling
        limited_by = "twice-dtl"
    if apply_ramp_down:
        for target in ramp_down_targets:
            if not isinstance(target, RampDownTarget):
                raise TypeError("ramp_down_targets must be RampDownTarget values")
            ramped = _ramp_down(dbldt, float(model_time_s), target)
            if ramped != dbldt:
                dbldt = ramped
                limited_by = f"event:{target.name}"
    if not math.isfinite(dbldt) or dbldt <= 0.0:
        raise ValueError(
            "getnewdt produced a non-positive large step; cflmax="
            f"{cflmax!r}, ksmax={ksmax!r}, dt_s={dt_s!r}"
        )
    # misclibs.F:800-802.  Both measured numbers are linear in dt, so CM1
    # rescales them to the new step rather than measuring again, in REAL:
    # tem = dt/dtlast with dt = dbldt rounded (misclibs.F:785).
    ratio = r(r(dbldt) / dt_real)
    return AdaptiveTimestepDecision(
        dt_s=dbldt,
        previous_dt_s=float(dt_s),
        cflmax_in=float(cflmax),
        ksmax_in=float(ksmax),
        cflmax_scaled=r(r(cflmax) * ratio),
        ksmax_scaled=r(r(ksmax) * ratio),
        limited_by=limited_by,
    )


__all__ = [
    "CM1_ADAPTIVE_TIMESTEP_AUTHORITY",
    "CM1_MAX_NSOUND",
    "CM1_MIN_NSOUND",
    "cm1_dtsmall",
    "CM1_CFLMAX_FLOOR",
    "CM1_CFL_LIMIT",
    "CM1_KSMAX_FLOOR",
    "CM1_KS_LIMIT_2D",
    "CM1_KS_LIMIT_3D",
    "CM1_MAX_CHANGE",
    "CM1_MAX_DT_MULTIPLE_OF_DTL",
    "AdaptiveTimestepDecision",
    "RampDownTarget",
    "cm1_diffusion_number",
    "cm1_getnewdt",
    "cm1_ks_limit",
]
