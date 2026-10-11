"""How long a differential run must be to be decisive for one term.

A wrong term diverges on its own physical timescale.  Picking a fixed duration
either wastes a thousand steps proving something a hundred already proved, or
stops before the term under test has had time to act.  So: name the term, give
the grid, and this returns the timescale, the run length in seconds, and the
number of large steps -- with the formula it used, so the choice is auditable
rather than a number someone remembered.

The rule throughout is *one characteristic time of the mechanism under test,
resolved by enough steps to see the curve*:

    horizon = crossings * timescale
    steps   = ceil(horizon / dt)

with ``crossings`` stated per term.  Terms whose timescale is hours
(Coriolis, radiation) are marked ``differential_unsuitable``: a storm-length
run is the wrong instrument and the entry says what to use instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

#: CM1's own reference sound speed for the acoustic stability bound
#: (``src/misclibs.F:883-885`` divides by 360 m/s).
REFERENCE_SOUND_SPEED_M_S = 360.0
#: A dry-adiabatic troposphere's buoyancy frequency, for gravity-wave terms.
REFERENCE_BRUNT_VAISALA_S_INV = 0.01


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    dz_m: float
    dt_s: float
    acoustic_substeps: int = 6
    wind_scale_m_s: float = 10.0
    shear_m_s: float = 20.0
    shear_depth_m: float = 3000.0
    sound_speed_m_s: float = REFERENCE_SOUND_SPEED_M_S

    @property
    def dh_m(self) -> float:
        return min(self.dx_m, self.dy_m)

    @property
    def lx_m(self) -> float:
        return self.nx * self.dx_m

    @property
    def ly_m(self) -> float:
        return self.ny * self.dy_m

    @property
    def lz_m(self) -> float:
        return self.nz * self.dz_m

    @property
    def dtsmall_s(self) -> float:
        return self.dt_s / max(self.acoustic_substeps, 1)


@dataclass(frozen=True)
class Horizon:
    term: str
    timescale_s: float
    crossings: float
    horizon_s: float
    large_steps: int
    acoustic_steps: int
    formula: str
    rationale: str
    differential_unsuitable: bool = False
    instead: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "timescale_s": self.timescale_s,
            "crossings": self.crossings,
            "horizon_s": self.horizon_s,
            "large_steps": self.large_steps,
            "acoustic_steps": self.acoustic_steps,
            "formula": self.formula,
            "rationale": self.rationale,
            "differential_unsuitable": self.differential_unsuitable,
            "instead": self.instead,
        }


def _make(term: str, grid: Grid, timescale: float, crossings: float, formula: str,
          rationale: str, *, unsuitable: bool = False, instead: str | None = None) -> Horizon:
    horizon = crossings * timescale
    steps = max(1, math.ceil(horizon / grid.dt_s))
    return Horizon(
        term=term,
        timescale_s=timescale,
        crossings=crossings,
        horizon_s=steps * grid.dt_s,
        large_steps=steps,
        acoustic_steps=steps * max(grid.acoustic_substeps, 1),
        formula=formula,
        rationale=rationale,
        differential_unsuitable=unsuitable,
        instead=instead,
    )


TERMS: Mapping[str, Callable[[Grid], Horizon]] = {}


def _term(name: str):
    def register(fn: Callable[[Grid], Horizon]) -> Callable[[Grid], Horizon]:
        TERMS[name] = fn
        return fn
    return register


@_term("acoustic")
def _acoustic(g: Grid) -> Horizon:
    return _make(
        "acoustic", g, g.dh_m / g.sound_speed_m_s, 100.0,
        "tau = min(dx,dy)/c, c = 360 m/s (misclibs.F:883-885); horizon = 100*tau",
        "A sound wave crosses one cell in dx/c. An error in psolver, in the vertically "
        "implicit acoustic step or in nsound shows up within a handful of crossings and "
        "grows every substep after; a hundred crossings is several hundred acoustic "
        "substeps, which is decisive and still seconds of model time.",
    )


@_term("psolver")
def _psolver(g: Grid) -> Horizon:
    return _acoustic(g)


@_term("advection")
def _advection(g: Grid) -> Horizon:
    return _make(
        "advection", g, g.dh_m / max(g.wind_scale_m_s, 1e-6), 1.0,
        "tau = min(dx,dy)/U; horizon = 1*tau (one advective cell crossing)",
        "Fifth-order advection, the metric factors mh/mf and the diffusion operators all "
        "act once per cell crossing. One crossing is the shortest window in which an "
        "order or metric error is unambiguous; anything longer is measuring the "
        "downstream consequences rather than the term.",
    )


@_term("metric")
def _metric(g: Grid) -> Horizon:
    return _advection(g)


@_term("diffusion")
def _diffusion(g: Grid) -> Horizon:
    return _advection(g)


@_term("boundary")
def _boundary(g: Grid) -> Horizon:
    length = max(g.lx_m, g.ly_m)
    return _make(
        "boundary", g, length / g.sound_speed_m_s, 1.5,
        "tau = max(Lx,Ly)/c; horizon = 1.5*tau (transit plus the first reflection)",
        "A lateral boundary condition cannot be wrong until a wave reaches it. The "
        "acoustic transit is the fastest signal, so L/c is the earliest the boundary can "
        "speak; a further half transit lets a spurious reflection come back into the "
        "interior where the comparator can see it.",
    )


@_term("gravity-wave")
def _gravity(g: Grid) -> Horizon:
    return _make(
        "gravity-wave", g, 2.0 * math.pi / REFERENCE_BRUNT_VAISALA_S_INV, 1.0,
        "tau = 2*pi/N, N = 0.01 s^-1; horizon = 1*tau (one buoyancy period)",
        "Buoyancy, the base-state stratification and the upper Rayleigh damping act on "
        "the buoyancy period. One period is a full oscillation, which is what separates "
        "a wrong restoring force from a phase offset.",
    )


@_term("damping")
def _damping(g: Grid) -> Horizon:
    return _gravity(g)


@_term("surface-drag")
def _drag(g: Grid) -> Horizon:
    return _make(
        "surface-drag", g, g.shear_depth_m / max(g.shear_m_s, 1e-6), 1.0,
        "tau = shear_depth/shear_speed; horizon = 1*tau (one near-ground shear turnover)",
        "Surface drag changes the wind through the depth of the shear layer in one "
        "turnover of that layer. A minute or two, not a storm.",
    )


@_term("microphysics")
def _microphysics(g: Grid) -> Horizon:
    return _make(
        "microphysics", g, g.dt_s, 20.0,
        "tau = dt; horizon = 20*dt (the first condensation)",
        "Saturation adjustment and the warm-rain conversions fire on the first step in "
        "which a parcel is supersaturated. Initialise a saturated or near-saturated "
        "profile and the scheme acts within seconds; do NOT wait for a bubble to build "
        "a cloud, that measures the dycore's transport of the bubble, not the scheme.",
    )


@_term("coriolis")
def _coriolis(g: Grid) -> Horizon:
    horizon = _make(
        "coriolis", g, 2.0 * math.pi / 1.0e-4, 1.0,
        "tau = 2*pi/f, f = 1e-4 s^-1; ~17.5 h",
        "The inertial period is many hours. A storm-length differential cannot resolve "
        "a Coriolis error: the dycore's own divergence swamps it long before f acts.",
        unsuitable=True,
        instead="an analytic inertial oscillation: uniform f, no other forcing, initial "
                "(u,v) = (U,0), and compare against u = U cos(ft), v = -U sin(ft) in "
                "closed form. That measures the term itself and costs one column.",
    )
    return horizon


@_term("rterm")
def _rterm(g: Grid) -> Horizon:
    return _make(
        "rterm", g, 12.0 * 3600.0, 1.0 / 100.0,
        "tau = 12*3600 s, CM1's own relaxation time (solve1.F:1028); horizon = tau/100",
        "Unlike a real radiation scheme this term is not diurnal: it is a Newtonian "
        "relaxation of the potential-temperature perturbation towards zero with a 12 h "
        "e-folding, capped at +-1/43200 K/s (2 K/day). Its signal is therefore linear in "
        "time from the first step, and one hundredth of the e-folding already moves theta' "
        "by one per cent of its amplitude -- about ten times this harness's measured "
        "rounding floor for th3d on the worked example. Waiting a whole e-folding would "
        "cost a hundred times the steps to measure the same coefficient, and by then the "
        "storm's own nonlinearity, not the term, would dominate the difference.",
        instead="run the pair rterm=0 / rterm=1 on BOTH sides and compare the two "
                "differences, not the two states: the open-lateral scalar divergence the "
                "worked example found is common to both members and cancels exactly.",
    )


@_term("radiation")
def _radiation(g: Grid) -> Horizon:
    return _make(
        "radiation", g, 86400.0, 1.0,
        "tau = 86400 s (one diurnal cycle)",
        "Radiative heating rates are a diurnal signal. Running a storm proves nothing "
        "about the scheme.",
        unsuitable=True,
        instead="an offline comparison against the RRTMG reference tables in "
                "run/RRTMG_LW_DATA and run/RRTMG_SW_DATA for a fixed profile.",
    )


def compute(term: str, grid: Grid) -> Horizon:
    if term not in TERMS:
        raise KeyError(
            f"unknown term {term!r}; the harness knows {sorted(TERMS)}. Add a term with its "
            "own timescale rather than reusing one whose mechanism does not match, or the "
            "horizon is a guess wearing a formula."
        )
    return TERMS[term](grid)
