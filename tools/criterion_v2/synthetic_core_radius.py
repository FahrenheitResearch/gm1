"""Why criterion v1's core-radius rule depends on the grid: synthetic vortices through v1's own code.

    nice -n 19 python3 tools/criterion_v2/synthetic_core_radius.py --out DIR [--workers 3]

SYNTHETIC ONLY.  No GM1 run is read or scored here.  Every number comes from
analytic vortices sampled on CM1's C grid and handed to the v1 criterion's
own host reference (``cm1core.nest_tornado_criterion``): ``reference_sample``
for the nine kernel words (centre, zeta_max, the two wind maxima, ncore),
``core_radius_m`` for ``dx*sqrt(ncore/pi)``, ``interior_corner_box`` for the
1000 m search box, ``height_bracket`` for the 10 m levels and
``CriterionLedger`` for the Z/W/C/M conditions, the 60 s episode and the
verdict.  Thresholds are read from ``CRITERION``, never retyped.

VORTICES (tangential wind only; a uniform radial or vertical flow adds no
vertical vorticity, so the core-radius rule sees only this)
  rankine   solid body inside RMW, potential outside: zeta = 2V/R, flat top
  burgers   Burgers-Rott: v = (G/2 pi r)(1 - exp(-r^2/r0^2)), RMW = 1.1209 r0
  sullivan  Sullivan (1959) two-celled: v = (G/2 pi r) H(r^2/L^2)/H(inf),
            H(x) = int_0^x exp(-t + 3 int_0^t (1-e^-s)/s ds) dt; zeta peaks on
            a ring, the axis holds 1/7.96 of the peak
  ring      thin two-celled ring: Gaussian vorticity annulus, e-folding
            half-width 0.1 of the ring radius, calm eye (a Sullivan-type
            vortex whose ring is thinner than Sullivan's)
  +sat      the burgers/sullivan parent plus three Burgers-Rott satellites
            (RMW 8 m, +25 m/s) on the parent RMW, orbiting at 0.5 Vmax/RMW:
            fixed physical small-scale structure that only fine grids resolve
Each is scaled to a given Vmax (50, 100 m/s) and RMW (20, 40, 75, 150 m).

SAMPLING (CM1 staggering, the nest's padded owners)
  u at (i dx, (j+1/2) dy), v at ((i+1/2) dx, j dy), corners at (i dx, j dy),
  ghosts filled analytically; winds stored as float32 model words.  The
  domain is 2400 m square on every grid (nx = 24, 120, 300, 600, 1200 at
  100, 20, 8, 4, 2 m), so v1's 1000 m edge margin leaves its own 400 m
  search box.  Frame-relative wind = vortex + T - F with the storm
  translating at T = 20 m/s toward 40 deg from east and the nest frame moving
  at F = (16.5, 16.5) m/s (the E100 decks' umove/vmove); the criterion adds F
  back for the ground-relative wind.

  snapshot    16 sub-grid phases of the axis ({0, 1/4, 1/2, 3/4} dx squared)
              at the domain centre: zeta_max, core radius, ncore, winds, the
              Z/W/C pass fraction, the centre's offset from the true axis and
              the mean corner vorticity within 100 m of the criterion centre
              (a circulation, for contrast with the peak-relative area)
  timeseries  90 s at the E100 parent step (0.375 s, 241 samples), the axis
              drifting at T - F = 3.8 m/s through the search box (sweeping
              grid phases) and the satellites orbiting, through
              CriterionLedger: condition counts, the longest qualifying run,
              the largest centre hop, the verdict
  vertical    what "lowest model level" versus 10 m AGL does to the sampled
              wind for a neutral log profile (z0 = 0.01, 0.1, 0.5 m) and for
              free slip, on the level sets of the E100 decks and the 2 m deck
              (point value at zh(1), and the layer-1 mean for a finite-volume
              reading of the lowest level);
              and the v1 sample itself (Burgers-Rott, RMW 75 m, 10 m Vmax 25
              and 50 m/s, log profile z0 = 0.1 m) taken at 10 m (v1's
              height_bracket) and at the lowest level, on the 20, 4 and 2 m
              level sets

OUTPUT (``--out`` must not exist): analytic.json, snapshot.tsv,
timeseries.tsv, vertical.tsv, vertical_criterion.tsv, summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cm1core.nest_tornado_criterion import (  # noqa: E402
    CRITERION,
    CRITERION_SHA256,
    NG,
    CriterionLedger,
    SamplePlan,
    core_radius_m,
    height_bracket,
    interior_corner_box,
    reference_corner_vorticity,
    reference_sample,
    sample_from_words,
)

SCHEMA = "gm1.criterion-v2.synthetic-v1-diagnosis/v1"
GRIDS_M = (100.0, 20.0, 8.0, 4.0, 2.0)
DOMAIN_M = 2400.0
VMAX_M_S = (50.0, 100.0)
RMW_M = (20.0, 40.0, 75.0, 150.0)
TRANSLATION_M_S = 20.0
TRANSLATION_HEADING_DEG = 40.0  # toward, counter-clockwise from east
FRAME_M_S = (16.5, 16.5)
PARENT_DT_S = 0.375
SERIES_S = 90.0
PHASES = (0.0, 0.25, 0.5, 0.75)
SATELLITE = {"count": 3, "rmw_m": 8.0, "vmax_m_s": 25.0, "orbit_speed_fraction": 0.5}
MEAN_DISK_M = 100.0


# ---------------------------------------------------------------------------
# Vortex profiles: omega(r) = v(r)/r (finite on the axis) and zeta(r).
# ---------------------------------------------------------------------------
def _burgers_rmw_ratio() -> float:
    """x = RMW/r0 of Burgers-Rott: the root of exp(x^2) = 1 + 2 x^2 (dv/dr = 0)."""
    q = 1.25
    for _ in range(60):
        f = math.exp(q) - 1.0 - 2.0 * q
        q -= f / (math.exp(q) - 2.0)
    return math.sqrt(q)


BURGERS_RMW_OVER_R0 = _burgers_rmw_ratio()


class _SullivanTable:
    """H, H' and H(x)/x of the Sullivan vortex on a uniform x table (binary64)."""

    def __init__(self, x_max: float = 80.0, h: float = 1.0e-4) -> None:
        n = int(round(x_max / h))
        x = np.arange(n + 1, dtype=np.float64) * h
        integrand = np.ones_like(x)
        integrand[1:] = -np.expm1(-x[1:]) / x[1:]
        ein = np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * h)))
        hp = np.exp(-x + 3.0 * ein)
        H = np.concatenate(([0.0], np.cumsum(0.5 * (hp[1:] + hp[:-1]) * h)))
        g = np.ones_like(x)
        g[1:] = H[1:] / x[1:]
        self.h, self.x, self.hp, self.H, self.g = h, x, hp, H, g
        self.H_inf = float(H[-1])
        # RMW: v ~ H(x)/sqrt(x) is largest at x_m.
        k = int(np.argmax(H[1:] / np.sqrt(x[1:]))) + 1
        self.x_rmw = float(x[k])
        self.H_rmw = float(H[k])
        # The vorticity ring: H' is largest where 3(1 - e^-x) = x.
        kp = int(np.argmax(hp))
        self.x_ring = float(x[kp])
        self.hp_ring = float(hp[kp])

    def lookup(self, table: np.ndarray, xq: np.ndarray, beyond: np.ndarray) -> np.ndarray:
        pos = xq / self.h
        k = np.minimum(pos.astype(np.int64), len(self.x) - 2)
        f = pos - k
        val = table[k] * (1.0 - f) + table[k + 1] * f
        return np.where(xq >= self.x[-1], beyond, val)


_SULLIVAN: _SullivanTable | None = None


def sullivan_table() -> _SullivanTable:
    global _SULLIVAN
    if _SULLIVAN is None:
        _SULLIVAN = _SullivanTable()
    return _SULLIVAN


class _RingTable:
    """A tabulated vorticity shape zeta_s(s), s = r/rc: circulation, RMW, lookups."""

    def __init__(self, half_width: float = 0.1, s_max: float = 60.0, h: float = 1.0e-4) -> None:
        n = int(round(s_max / h))
        s = np.arange(n + 1, dtype=np.float64) * h
        zs = np.exp(-(((s - 1.0) / half_width) ** 2))
        m = np.concatenate(([0.0], np.cumsum(0.5 * (zs[1:] * s[1:] + zs[:-1] * s[:-1]) * h)))
        g = np.empty_like(s)
        g[0] = 0.5 * zs[0]
        g[1:] = m[1:] / (s[1:] * s[1:])
        self.h, self.s, self.zs, self.m, self.g = h, s, zs, m, g
        self.half_width = half_width
        k = int(np.argmax(m[1:] / s[1:])) + 1
        self.s_rmw, self.m_rmw = float(s[k]), float(m[k])
        self.m_inf = float(m[-1])

    def lookup(self, table: np.ndarray, sq: np.ndarray, beyond: np.ndarray) -> np.ndarray:
        pos = sq / self.h
        k = np.minimum(pos.astype(np.int64), len(self.s) - 2)
        f = pos - k
        val = table[k] * (1.0 - f) + table[k + 1] * f
        return np.where(sq >= self.s[-1], beyond, val)


_RING: _RingTable | None = None


def ring_table() -> _RingTable:
    global _RING
    if _RING is None:
        _RING = _RingTable()
    return _RING


class Vortex:
    """One axisymmetric vortex of a named profile, scaled to (vmax, rmw)."""

    def __init__(self, kind: str, vmax: float, rmw: float) -> None:
        self.kind, self.vmax, self.rmw = kind, float(vmax), float(rmw)
        if kind == "rankine":
            self.zeta_peak = 2.0 * vmax / rmw
        elif kind == "burgers":
            self.r0 = rmw / BURGERS_RMW_OVER_R0
            self.c = vmax * rmw / (-math.expm1(-(rmw / self.r0) ** 2))  # G/(2 pi)
            self.zeta_peak = 2.0 * self.c / self.r0 ** 2
        elif kind == "sullivan":
            t = sullivan_table()
            self.L2 = rmw * rmw / t.x_rmw
            self.c = vmax * rmw * t.H_inf / t.H_rmw  # G/(2 pi)
            self.zeta_peak = 2.0 * self.c / (self.L2 * t.H_inf) * t.hp_ring
        elif kind == "ring":
            t = ring_table()
            self.rc = rmw / t.s_rmw
            self.a = vmax * t.s_rmw / t.m_rmw  # v = a m(s)/s
            self.zeta_peak = self.a / self.rc * float(t.zs.max())
        else:
            raise ValueError(kind)

    def omega(self, r2: np.ndarray) -> np.ndarray:
        """v(r)/r as a function of r^2."""
        if self.kind == "rankine":
            R2 = self.rmw * self.rmw
            with np.errstate(divide="ignore"):
                return np.where(r2 < R2, self.vmax / self.rmw, self.vmax * self.rmw / np.maximum(r2, R2))
        if self.kind == "burgers":
            q = r2 / (self.r0 * self.r0)
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.where(q > 1.0e-10, -np.expm1(-q) / q, 1.0 - 0.5 * q)
            return self.c / (self.r0 * self.r0) * g
        if self.kind == "ring":
            t = ring_table()
            sq = np.sqrt(r2) / self.rc
            with np.errstate(divide="ignore"):
                beyond = t.m_inf / np.maximum(sq * sq, t.s[-1] ** 2)
            return self.a / self.rc * t.lookup(t.g, sq, beyond)
        t = sullivan_table()
        x = r2 / self.L2
        with np.errstate(divide="ignore"):
            beyond = t.H_inf / np.maximum(x, t.x[-1])
        return self.c / (self.L2 * t.H_inf) * t.lookup(t.g, x, beyond)

    def zeta(self, r: np.ndarray) -> np.ndarray:
        r2 = r * r
        if self.kind == "rankine":
            return np.where(r < self.rmw, 2.0 * self.vmax / self.rmw, 0.0)
        if self.kind == "burgers":
            return 2.0 * self.c / self.r0 ** 2 * np.exp(-r2 / self.r0 ** 2)
        if self.kind == "ring":
            t = ring_table()
            return self.a / self.rc * t.lookup(t.zs, r / self.rc, np.zeros_like(r))
        t = sullivan_table()
        x = r2 / self.L2
        return 2.0 * self.c / (self.L2 * t.H_inf) * t.lookup(t.hp, x, np.zeros_like(x))

    def vt(self, r: np.ndarray) -> np.ndarray:
        return r * self.omega(r * r)


class Flow:
    """A parent vortex plus optional orbiting satellites, the whole pattern translating."""

    def __init__(self, kind: str, vmax: float, rmw: float, satellites: bool) -> None:
        self.parent = Vortex(kind, vmax, rmw)
        self.satellites = satellites
        self.sat = Vortex("burgers", SATELLITE["vmax_m_s"], SATELLITE["rmw_m"]) if satellites else None
        self.orbit_rad_s = SATELLITE["orbit_speed_fraction"] * vmax / rmw
        self.label = kind + ("+sat" if satellites else "")

    def centres(self, cx: float, cy: float, t: float) -> list[tuple[Vortex, float, float]]:
        out = [(self.parent, cx, cy)]
        if self.sat is not None:
            n = SATELLITE["count"]
            for m in range(n):
                a = self.orbit_rad_s * t + 2.0 * math.pi * m / n
                out.append((self.sat, cx + self.parent.rmw * math.cos(a),
                            cy + self.parent.rmw * math.sin(a)))
        return out


def analytic_reference(flow: Flow) -> dict:
    """Continuous values: the vorticity peak and the half-peak area's equivalent radius."""
    v = flow.parent
    r = np.linspace(0.0, 6.0 * v.rmw, 600001)
    z = v.zeta(r)
    zmax = float(z.max())
    inside = z >= 0.5 * zmax
    dr = r[1] - r[0]
    area = float(np.sum(2.0 * np.pi * r[inside]) * dr)
    idx = np.nonzero(inside)[0]
    vt100 = float(v.vt(np.array([MEAN_DISK_M]))[0])
    return {
        "zeta_peak_s_inv": zmax,
        "zeta_axis_s_inv": float(z[0]),
        "half_peak_inner_m": float(r[idx[0]]),
        "half_peak_outer_m": float(r[idx[-1]]),
        "half_peak_equiv_radius_m": math.sqrt(area / math.pi),
        "half_peak_equiv_radius_over_rmw": math.sqrt(area / math.pi) / v.rmw,
        "passes_core_radius_when_resolved": math.sqrt(area / math.pi) >= CRITERION["core_radius_min_m"],
        "mean_zeta_within_100m_of_axis_s_inv": 2.0 * vt100 / MEAN_DISK_M,
    }


# ---------------------------------------------------------------------------
# Sampling on CM1's C grid and v1's host reference.
# ---------------------------------------------------------------------------
def background() -> tuple[float, float]:
    a = math.radians(TRANSLATION_HEADING_DEG)
    return (TRANSLATION_M_S * math.cos(a) - FRAME_M_S[0], TRANSLATION_M_S * math.sin(a) - FRAME_M_S[1])


def make_plan(dx: float) -> SamplePlan:
    nx = ny = int(round(DOMAIN_M / dx))
    lo_i, hi_i = interior_corner_box(nx, dx, CRITERION["edge_margin_m"])
    lo_j, hi_j = interior_corner_box(ny, dx, CRITERION["edge_margin_m"])
    return SamplePlan(nx=nx, ny=ny, nz=2, dx_m=float(np.float32(dx)), dy_m=float(np.float32(dx)),
                      ku0=1, ku1=2, w0=1.0, w1=0.0, radius_m=CRITERION["disk_radius_m"],
                      u_move_m_s=FRAME_M_S[0], v_move_m_s=FRAME_M_S[1],
                      lo_i=lo_i, hi_i=hi_i, lo_j=lo_j, hi_j=hi_j)


def owners(flow: Flow, plan: SamplePlan, cx: float, cy: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Padded float32 u/v owners (levels 1 and 2 hold the plane) of the flow at time t."""
    dx, nx, ny = plan.dx_m, plan.nx, plan.ny
    bu, bv = background()
    xu = (np.arange(nx + 1 + 2 * NG) - NG) * dx
    yu = (np.arange(ny + 2 * NG) - NG + 0.5) * dx
    xv = (np.arange(nx + 2 * NG) - NG + 0.5) * dx
    yv = (np.arange(ny + 1 + 2 * NG) - NG) * dx
    uplane = np.full((ny + 2 * NG, nx + 1 + 2 * NG), bu, dtype=np.float64)
    vplane = np.full((ny + 1 + 2 * NG, nx + 2 * NG), bv, dtype=np.float64)
    for vortex, x0, y0 in flow.centres(cx, cy, t):
        X, Y = xu[None, :] - x0, yu[:, None] - y0
        uplane -= vortex.omega(X * X + Y * Y) * Y
        X, Y = xv[None, :] - x0, yv[:, None] - y0
        vplane += vortex.omega(X * X + Y * Y) * X
    u = np.zeros((plan.nz + 2,) + uplane.shape, dtype=np.float32)
    v = np.zeros((plan.nz + 2,) + vplane.shape, dtype=np.float32)
    u[1] = u[2] = uplane.astype(np.float32)
    v[1] = v[2] = vplane.astype(np.float32)
    return u, v


def sample(flow: Flow, plan: SamplePlan, cx: float, cy: float, t: float) -> dict:
    u, v = owners(flow, plan, cx, cy, t)
    words = reference_sample(u, v, plan=plan)
    found = bool(words[0] == 1.0)
    im, jm = int(words[1]), int(words[2])
    zeta = reference_corner_vorticity(u, v, plan)
    xc, yc = im * plan.dx_m, jm * plan.dy_m
    ex = np.arange(plan.nx + 1) * plan.dx_m - xc
    ey = np.arange(plan.ny + 1) * plan.dy_m - yc
    near = (ey[:, None] ** 2 + ex[None, :] ** 2) <= MEAN_DISK_M ** 2
    return {
        "words": words, "found": found, "im": im, "jm": jm,
        "zeta_max": float(words[3]), "wg": float(words[4]), "wf": float(words[5]),
        "ncore": int(words[6]), "r_core": core_radius_m(plan.dx_m, int(words[6])),
        "axis_offset_m": math.hypot(xc - cx, yc - cy),
        "mean_zeta_100m": float(zeta[near].mean()),
    }


def _stats(values: list[float]) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.min()), float(np.median(a)), float(a.max())


def run_case(case: tuple[str, bool, float, float, float]) -> dict:
    try:
        os.nice(19)
    except OSError:
        pass
    kind, sat, vmax, rmw, dx = case
    flow = Flow(kind, vmax, rmw, sat)
    plan = make_plan(dx)
    c = CRITERION
    centre = 0.5 * DOMAIN_M
    snaps = []
    for fy in PHASES:
        for fx in PHASES:
            s = sample(flow, plan, centre + fx * dx, centre + fy * dx, 0.0)
            snaps.append(s)
    z_ok = [s["zeta_max"] >= c["zeta_min_s_inv"] for s in snaps]
    w_ok = [s["wg"] >= c["ground_relative_wind_min_m_s"] for s in snaps]
    c_ok = [s["r_core"] >= c["core_radius_min_m"] for s in snaps]
    snapshot = {
        "zeta_max": _stats([s["zeta_max"] for s in snaps]),
        "r_core": _stats([s["r_core"] for s in snaps]),
        "ncore": _stats([s["ncore"] for s in snaps]),
        "wg": _stats([s["wg"] for s in snaps]),
        "axis_offset_m": _stats([s["axis_offset_m"] for s in snaps]),
        "mean_zeta_100m": _stats([s["mean_zeta_100m"] for s in snaps]),
        "pass_Z": sum(z_ok) / len(snaps), "pass_W": sum(w_ok) / len(snaps),
        "pass_C": sum(c_ok) / len(snaps),
        "pass_ZWC": sum(a and b and d for a, b, d in zip(z_ok, w_ok, c_ok)) / len(snaps),
    }
    # Time series through the ledger.
    bu, bv = background()
    nsteps = int(round(SERIES_S / PARENT_DT_S))
    x0 = centre - 0.5 * bu * SERIES_S
    y0 = centre - 0.5 * bv * SERIES_S
    ledger = CriterionLedger()
    hops, rcs, zms = [], [], []
    prev = None
    for k in range(nsteps + 1):
        t = k * PARENT_DT_S
        cx, cy = x0 + bu * t, y0 + bv * t
        s = sample(flow, plan, cx, cy, t)
        cs = sample_from_words(s["words"], step=k, model_time_s=t, dx_m=plan.dx_m, dy_m=plan.dy_m,
                               origin_x_m=0.0, origin_y_m=0.0,
                               domainlocx_m=FRAME_M_S[0] * t, domainlocy_m=FRAME_M_S[1] * t)
        ledger.observe(cs)
        if prev is not None:
            hops.append(math.hypot(cs.ground_x_m - prev[0], cs.ground_y_m - prev[1]))
        prev = (cs.ground_x_m, cs.ground_y_m)
        rcs.append(s["r_core"])
        zms.append(s["zeta_max"])
    series = {
        "samples": ledger.samples,
        "counts": dict(ledger.counts),
        "longest_run_s": ledger.longest_run_s,
        "verdict": "TORNADO" if ledger.tornado else "NO_TORNADO",
        "genesis_s": ledger.genesis_s,
        "max_centre_hop_m": max(hops),
        "hop_allowance_m": c["centre_speed_max_m_s"] * PARENT_DT_S + c["centre_jump_allowance_m"],
        "r_core": _stats(rcs),
        "zeta_max": _stats(zms),
    }
    return {"case": {"kind": flow.label, "vmax_m_s": vmax, "rmw_m": rmw, "dx_m": dx},
            "grid": {"nx": plan.nx, "search_corners_i": [plan.lo_i, plan.hi_i]},
            "snapshot": snapshot, "timeseries": series}


# ---------------------------------------------------------------------------
# Lowest model level versus 10 m AGL.
# ---------------------------------------------------------------------------
LEVEL_SETS = {
    # name: interface heights (m) of the lowest layers; zh = mid-layer
    "100 m isotropic (hypothetical, dz = 100 m)": [0.0, 100.0, 200.0],
    "E100 parent and 20 m children (dz_bot 20 m)": [0.0, 20.0, 40.0, 60.0],
    "8 m isotropic (hypothetical)": [0.0, 8.0, 16.0, 24.0],
    "E100V/E100T 4 m child (child_levels_m 0,4,8,...)": [0.0, 4.0, 8.0, 12.0, 16.0],
    "2 m deck E50N2 (child_levels_m 0,2,4,...)": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
}
Z0_M = (0.01, 0.1, 0.5)


def vertical_table() -> list[dict]:
    rows = []
    h = CRITERION["height_agl_m"]
    for name, faces in LEVEL_SETS.items():
        zh = [0.5 * (a + b) for a, b in zip(faces[:-1], faces[1:])]
        levels = SimpleNamespace(nk=len(zh), zh=[math.nan] + zh)
        bracket = height_bracket(levels, h)
        row = {"level_set": name, "zh1_m": zh[0], "bracket": None}
        if bracket is not None:
            k0, k1, w0, w1 = bracket
            row["bracket"] = {"k": [k0, k1], "zh_m": [zh[k0 - 1], zh[k1 - 1]], "weights": [w0, w1]}
        for z0 in Z0_M:
            prof = lambda z, z0=z0: math.log(z / z0)  # noqa: E731
            ref = prof(h)
            row[f"log_z0_{z0}_lowest_over_10m"] = prof(zh[0]) / ref
            # A finite-volume reading of the lowest level: the mean of the log
            # profile over its layer [z0, face(1)] (zero below z0), over the 10 m point.
            d = faces[1]
            row[f"log_z0_{z0}_layer1_mean_over_10m"] = (d * math.log(d / z0) - d + z0) / d / ref
            if bracket is not None:
                k0, k1, w0, w1 = bracket
                row[f"log_z0_{z0}_sampled10m_over_true10m"] = (
                    w0 * prof(zh[k0 - 1]) + w1 * prof(zh[k1 - 1])) / ref
            else:
                row[f"log_z0_{z0}_sampled10m_over_true10m"] = None
        row["free_slip_lowest_over_10m"] = 1.0
        rows.append(row)
    return rows


VERTICAL_CRITERION_SETS = {
    # name: (dx_m, scalar-level heights zh(1..) in m)
    "20 m (E100W/E100N levels)": (20.0, [10.0, 30.0, 50.0]),
    "4 m (E100V levels)": (4.0, [2.0, 6.0, 10.0, 14.0]),
    "2 m (E50N2 levels)": (2.0, [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]),
}
VERTICAL_CRITERION_Z0_M = 0.1
VERTICAL_CRITERION_VMAX_10M = (25.0, 50.0)
VERTICAL_CRITERION_RMW_M = 75.0


def vertical_criterion_case(case: tuple[str, float]) -> list[dict]:
    """v1's own sample of a log-profile Burgers-Rott vortex at 10 m and at the lowest level."""
    try:
        os.nice(19)
    except OSError:
        pass
    name, vmax10 = case
    dx, zh = VERTICAL_CRITERION_SETS[name]
    z0 = VERTICAL_CRITERION_Z0_M
    h = CRITERION["height_agl_m"]
    flow = Flow("burgers", vmax10, VERTICAL_CRITERION_RMW_M, False)
    base = make_plan(dx)
    nx, ny = base.nx, base.ny
    centre = 0.5 * DOMAIN_M
    a = math.radians(TRANSLATION_HEADING_DEG)
    tx, ty = TRANSLATION_M_S * math.cos(a), TRANSLATION_M_S * math.sin(a)
    xu = (np.arange(nx + 1 + 2 * NG) - NG) * dx
    yu = (np.arange(ny + 2 * NG) - NG + 0.5) * dx
    xv = (np.arange(nx + 2 * NG) - NG + 0.5) * dx
    yv = (np.arange(ny + 1 + 2 * NG) - NG) * dx
    X, Y = xu[None, :] - centre, yu[:, None] - centre
    ug = -flow.parent.omega(X * X + Y * Y) * Y + tx
    X, Y = xv[None, :] - centre, yv[:, None] - centre
    vg = flow.parent.omega(X * X + Y * Y) * X + ty
    nz = len(zh)
    u = np.zeros((nz + 2,) + ug.shape, dtype=np.float32)
    v = np.zeros((nz + 2,) + vg.shape, dtype=np.float32)
    for k, z in enumerate(zh, start=1):
        f = math.log(z / z0) / math.log(h / z0)
        u[k] = (f * ug - FRAME_M_S[0]).astype(np.float32)
        v[k] = (f * vg - FRAME_M_S[1]).astype(np.float32)
    levels = SimpleNamespace(nk=nz, zh=[math.nan] + list(zh))
    k0, k1, w0, w1 = height_bracket(levels, h)
    rows = []
    for where, (ka, kb, wa, wb) in (("10 m AGL (v1 height_bracket)", (k0, k1, w0, w1)),
                                    ("lowest level", (1, 2, 1.0, 0.0))):
        plan = SamplePlan(nx=nx, ny=ny, nz=nz, dx_m=base.dx_m, dy_m=base.dy_m, ku0=ka, ku1=kb,
                          w0=wa, w1=wb, radius_m=base.radius_m, u_move_m_s=FRAME_M_S[0],
                          v_move_m_s=FRAME_M_S[1], lo_i=base.lo_i, hi_i=base.hi_i,
                          lo_j=base.lo_j, hi_j=base.hi_j)
        words = reference_sample(u, v, plan=plan)
        rc = core_radius_m(plan.dx_m, int(words[6]))
        c = CRITERION
        rows.append({
            "level_set": name, "vmax_10m_m_s": vmax10, "rmw_m": VERTICAL_CRITERION_RMW_M,
            "z0_m": z0, "sampled_at": where,
            "height_m": wa * zh[ka - 1] + wb * zh[kb - 1],
            "log_factor": (wa * math.log(zh[ka - 1] / z0) + wb * math.log(zh[kb - 1] / z0)) / math.log(h / z0),
            "zeta_max_s_inv": float(words[3]), "ground_relative_wind_max_m_s": float(words[4]),
            "core_radius_m": rc,
            "conditions": "".join(L if ok else "-" for L, ok in (
                ("Z", words[3] >= c["zeta_min_s_inv"]),
                ("W", words[4] >= c["ground_relative_wind_min_m_s"]),
                ("C", rc >= c["core_radius_min_m"]))),
        })
    return rows


# ---------------------------------------------------------------------------
def analytic_rules() -> dict:
    rmin = CRITERION["core_radius_min_m"]
    out = {
        "ncore_needed_for_40m": {str(dx): math.ceil(math.pi * (rmin / dx) ** 2) for dx in GRIDS_M},
        "r_core_floor_one_corner_m": {str(dx): dx / math.sqrt(math.pi) for dx in GRIDS_M},
        "dx_at_which_one_corner_passes_m": rmin * math.sqrt(math.pi),
        "burgers_rmw_over_r0": BURGERS_RMW_OVER_R0,
    }
    t = sullivan_table()
    out["sullivan"] = {"x_rmw": t.x_rmw, "x_ring": t.x_ring, "H_inf": t.H_inf,
                       "axis_over_ring_zeta": 1.0 / t.hp_ring,
                       "ring_radius_over_rmw": math.sqrt(t.x_ring / t.x_rmw)}
    rt = ring_table()
    out["ring"] = {"half_width_over_ring_radius": rt.half_width,
                   "ring_radius_over_rmw": 1.0 / rt.s_rmw}
    for kind in ("rankine", "burgers", "sullivan", "ring"):
        ref = analytic_reference(Flow(kind, 100.0, 100.0, False))
        ratio = ref["half_peak_equiv_radius_over_rmw"]
        out[kind] = {**out.get(kind, {}),
            "zeta_peak_over_vmax_per_rmw": ref["zeta_peak_s_inv"] / (100.0 / 100.0),
            "half_peak_equiv_radius_over_rmw": ratio,
            "half_peak_inner_over_rmw": ref["half_peak_inner_m"] / 100.0,
            "half_peak_outer_over_rmw": ref["half_peak_outer_m"] / 100.0,
            "smallest_rmw_passing_C_when_resolved_m": rmin / ratio,
            "rmw_in_grid_lengths_needed_to_pass_C": {str(dx): rmin / ratio / dx for dx in GRIDS_M},
        }
    return out


def cases() -> list[tuple[str, bool, float, float, float]]:
    out = []
    for kind in ("rankine", "burgers", "sullivan", "ring"):
        for vmax in VMAX_M_S:
            for rmw in RMW_M:
                for dx in GRIDS_M:
                    out.append((kind, False, vmax, rmw, dx))
    for kind in ("burgers", "sullivan"):
        for vmax in VMAX_M_S:
            for dx in GRIDS_M:
                out.append((kind, True, vmax, 75.0, dx))
    # Big grids first so the pool finishes together.
    return sorted(out, key=lambda c: c[4])


def fmt3(t: tuple[float, float, float], f: str = "%.2f") -> str:
    return "/".join(f % x for x in t)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="one Vmax, burgers only (smoke test)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    todo = cases()
    if a.quick:
        todo = [c for c in todo if c[0] in ("ring", "sullivan") and c[2] == 100.0 and c[4] >= 4.0]
    analytic = {"rules": analytic_rules(), "profiles": {}}
    for kind, sat, vmax, rmw, dx in todo:
        key = f"{kind}{'+sat' if sat else ''} V{vmax:g} R{rmw:g}"
        if key not in analytic["profiles"]:
            analytic["profiles"][key] = analytic_reference(Flow(kind, vmax, rmw, sat))
    vcases = [(name, vm) for name in VERTICAL_CRITERION_SETS for vm in VERTICAL_CRITERION_VMAX_10M]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        vcrit = [row for rows in pool.map(vertical_criterion_case, vcases) for row in rows]
        results = list(pool.map(run_case, todo))
    results.sort(key=lambda r: (r["case"]["kind"], r["case"]["vmax_m_s"], r["case"]["rmw_m"],
                                -r["case"]["dx_m"]))
    with (a.out / "snapshot.tsv").open("x") as f:
        f.write("kind\tvmax_m_s\trmw_m\tdx_m\tzeta_peak_true\tr_half_true_m\t"
                "zeta_max_min/med/max\tr_core_min/med/max_m\tncore_med\twg_med\t"
                "axis_offset_med/max_m\tmean_zeta_100m_of_centre_med\tmean_zeta_100m_of_axis_true\t"
                "pass_Z\tpass_W\tpass_C\tpass_ZWC\n")
        for r in results:
            c, s = r["case"], r["snapshot"]
            ref = analytic["profiles"][f"{c['kind']} V{c['vmax_m_s']:g} R{c['rmw_m']:g}"]
            f.write("\t".join([
                c["kind"], "%g" % c["vmax_m_s"], "%g" % c["rmw_m"], "%g" % c["dx_m"],
                "%.2f" % ref["zeta_peak_s_inv"], "%.1f" % ref["half_peak_equiv_radius_m"],
                fmt3(s["zeta_max"]), fmt3(s["r_core"], "%.1f"), "%d" % s["ncore"][1],
                "%.1f" % s["wg"][1], "%.1f/%.1f" % (s["axis_offset_m"][1], s["axis_offset_m"][2]),
                "%.3f" % s["mean_zeta_100m"][1], "%.3f" % ref["mean_zeta_within_100m_of_axis_s_inv"],
                "%.2f" % s["pass_Z"], "%.2f" % s["pass_W"], "%.2f" % s["pass_C"], "%.2f" % s["pass_ZWC"],
            ]) + "\n")
    with (a.out / "timeseries.tsv").open("x") as f:
        f.write("kind\tvmax_m_s\trmw_m\tdx_m\tsamples\tn_Z\tn_W\tn_C\tn_M\tn_qual\t"
                "longest_run_s\tmax_hop_m\thop_allowance_m\tr_core_min/med/max_m\tverdict\n")
        for r in results:
            c, s = r["case"], r["timeseries"]
            k = s["counts"]
            f.write("\t".join([
                c["kind"], "%g" % c["vmax_m_s"], "%g" % c["rmw_m"], "%g" % c["dx_m"], str(s["samples"]),
                str(k["zeta"]), str(k["ground_relative_wind"]), str(k["core_radius"]),
                str(k["centre_move"]), str(k["qualifying"]), "%.3f" % s["longest_run_s"],
                "%.1f" % s["max_centre_hop_m"], "%.1f" % s["hop_allowance_m"],
                fmt3(s["r_core"], "%.1f"), s["verdict"],
            ]) + "\n")
    vert = vertical_table()
    with (a.out / "vertical.tsv").open("x") as f:
        cols = ["level_set", "zh1_m", "bracket"] + [k for k in vert[0] if k.startswith(("log", "free"))]
        f.write("\t".join(cols) + "\n")
        for row in vert:
            f.write("\t".join(
                ("%.3f" % row[k]) if isinstance(row[k], float) else json.dumps(row[k]) if k == "bracket"
                else str(row[k]) for k in cols) + "\n")
    with (a.out / "vertical_criterion.tsv").open("x") as f:
        cols = list(vcrit[0])
        f.write("\t".join(cols) + "\n")
        for row in vcrit:
            f.write("\t".join(("%.3f" % row[k]) if isinstance(row[k], float) else str(row[k])
                              for k in cols) + "\n")
    with (a.out / "analytic.json").open("x") as f:
        json.dump(analytic, f, indent=2, sort_keys=True)
    summary = {
        "schema": SCHEMA, "criterion_v1_sha256": CRITERION_SHA256, "criterion_v1": dict(CRITERION),
        "setup": {"grids_m": GRIDS_M, "domain_m": DOMAIN_M, "translation_m_s": TRANSLATION_M_S,
                  "translation_heading_deg": TRANSLATION_HEADING_DEG, "frame_m_s": FRAME_M_S,
                  "parent_dt_s": PARENT_DT_S, "series_s": SERIES_S, "phases": PHASES,
                  "satellite": SATELLITE},
        "wall_s": time.time() - t0, "results": results, "vertical": vert,
        "vertical_criterion": vcrit,
    }
    with (a.out / "summary.json").open("x") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print((a.out / "snapshot.tsv").read_text())
    print((a.out / "timeseries.tsv").read_text())
    print((a.out / "vertical.tsv").read_text())
    print((a.out / "vertical_criterion.tsv").read_text())
    print(json.dumps(analytic["rules"], indent=1))
    print("wall_s %.1f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
