"""Synthetic vortices through criterion v2: the pre-freeze checks.

    nice -n 19 python3 tools/criterion_v2/synthetic_v2.py --out DIR [--workers 3]

SYNTHETIC ONLY.  No GM1 run is read or scored here.  The analytic profiles
(Rankine, Burgers-Rott, Sullivan two-celled, thin two-celled ring, and a
parent with three orbiting satellite vortices) are those of
``synthetic_core_radius.py`` (the v1 diagnosis), sampled on CM1's C grid
as :class:`v2rule.Plane` 10 m planes, plus tornadoes inside a
mesocyclone, vortices just above the thresholds, and the negative controls: a
single-cell grid spike, straight shear lines (infinite and a 1 km segment),
broad and weak mesocyclones (RMW 600, 750 and 1500 m), a vortex whose only
strong wind is a gust 900 m away, a strong vortex that lasts only 30 s, a
strong vortex against the domain edge, and a different short-lived vortex in
each sample (60 s and 30 s output).

Every case is a time series: the pattern translates at 20 m/s toward 40 deg
(the storm), the nest frame moves at (16.5, 16.5) m/s, so the axis drifts
3.8 m/s across the grid (sweeping sub-grid phases) while the ground offset
follows the frame.  ``decide`` gives the run outcome.

OUTPUT (``--out`` must not exist): sweep.tsv (one row per case and grid),
snapshot.tsv (Vmax and RMW against the analytic values over 16 sub-grid
phases), summary.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import v2rule  # noqa: E402
from v2rule import CRITERION_V2, CRITERION_V2_SHA256, GridFacts, Plane, decide, evaluate_plane  # noqa: E402


def _diagnosis():
    name = "criterion_v2_synthetic_core_radius"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "synthetic_core_radius.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _diagnosis()
Vortex = D.Vortex

TRANSLATION_M_S = 20.0
TRANSLATION_HEADING_DEG = 40.0
FRAME_M_S = (16.5, 16.5)
BOUNDARY_CELLS = 5  # spec_zone 1 + relax_zone 4, the E-deck child's


def background() -> tuple[float, float]:
    a = math.radians(TRANSLATION_HEADING_DEG)
    return (TRANSLATION_M_S * math.cos(a) - FRAME_M_S[0], TRANSLATION_M_S * math.sin(a) - FRAME_M_S[1])


# ---------------------------------------------------------------------------
# Flow components: each adds (du, dv) at points relative to the plane origin.
# ---------------------------------------------------------------------------
class Component:
    def add(self, x: np.ndarray, y: np.ndarray, t: float, u: np.ndarray | None, v: np.ndarray | None) -> None:
        raise NotImplementedError


class VortexComponent(Component):
    """An analytic vortex whose axis starts at (x0, y0) and drifts with the background."""

    def __init__(self, kind: str, vmax: float, rmw: float, x0: float, y0: float,
                 satellites: bool = False, alive: tuple[float, float] | None = None, drift: bool = True):
        self.flow = D.Flow(kind, vmax, rmw, satellites)
        self.x0, self.y0 = x0, y0
        self.alive = alive
        self.drift = drift

    def centre(self, t: float) -> tuple[float, float]:
        if not self.drift:
            return self.x0, self.y0
        bu, bv = background()
        return self.x0 + bu * t, self.y0 + bv * t

    def add(self, x, y, t, u, v):
        if self.alive is not None and not (self.alive[0] <= t <= self.alive[1]):
            return
        cx, cy = self.centre(t)
        for vortex, xc, yc in self.flow.centres(cx, cy, t):
            X, Y = x - xc, y - yc
            om = vortex.omega(X * X + Y * Y)
            if u is not None:
                u -= om * Y
            if v is not None:
                v += om * X


class ShearLine(Component):
    """u = -dU/2 tanh((y - y0)/delta): a straight vortex sheet (gust-front shear line)."""

    def __init__(self, du: float, delta: float, y0: float):
        self.du, self.delta, self.y0 = du, delta, y0

    def add(self, x, y, t, u, v):
        if u is not None:
            u -= 0.5 * self.du * np.tanh((y - self.y0) / self.delta)


class FiniteShearLine(Component):
    """A vortex-sheet segment of length ``length`` (tapered ends): net circulation, no rolled-up vortex."""

    def __init__(self, du: float, delta: float, x0: float, y0: float, length: float, taper: float = 60.0):
        self.du, self.delta, self.x0, self.y0, self.length, self.taper = du, delta, x0, y0, length, taper

    def add(self, x, y, t, u, v):
        if u is None:
            return
        f = -0.5 * self.du * np.tanh((y - self.y0) / self.delta)
        f = f * 0.5 * (np.tanh((x - self.x0 + 0.5 * self.length) / self.taper)
                       - np.tanh((x - self.x0 - 0.5 * self.length) / self.taper))
        u += f


class GustPatch(Component):
    """A straight, localised gust (an RFD surge): u += U exp(-r^2/a^2), no rotation of its own to speak of."""

    def __init__(self, speed: float, x0: float, y0: float, a: float):
        self.speed, self.x0, self.y0, self.a = speed, x0, y0, a

    def add(self, x, y, t, u, v):
        if u is not None:
            u += self.speed * np.exp(-((x - self.x0) ** 2 + (y - self.y0) ** 2) / self.a ** 2)


class Uniform(Component):
    """A uniform frame-relative wind added everywhere (e.g. to cancel the storm's ground-relative motion)."""

    def __init__(self, du: float, dv: float):
        self.du, self.dv = du, dv

    def add(self, x, y, t, u, v):
        if u is not None:
            u += self.du
        if v is not None:
            v += self.dv


class Spike:
    """A single-cell circulation around the corner nearest (xs, ys): its four faces +-a."""

    def __init__(self, a: float, xs: float, ys: float):
        self.a, self.xs, self.ys = a, xs, ys

    def apply(self, u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> None:
        i = int(round(self.xs / dx))
        j = int(round(self.ys / dy))
        # corner (i, j): u faces (i, j-1/2) below and (i, j+1/2) above; v faces (i-1/2, j), (i+1/2, j)
        u[j, i] -= self.a
        u[j - 1, i] += self.a
        v[j, i] += self.a
        v[j, i - 1] -= self.a


def make_plane(components: list[Component], spikes: list[Spike], domain_m: float, dx: float,
               t: float) -> Plane:
    n = int(round(domain_m / dx))
    bu, bv = background()
    xu = np.arange(n + 1) * dx
    yu = (np.arange(n) + 0.5) * dx
    xv = (np.arange(n) + 0.5) * dx
    yv = np.arange(n + 1) * dx
    u = np.full((n, n + 1), bu)
    v = np.full((n + 1, n), bv)
    Xu, Yu = np.meshgrid(xu, yu)
    Xv, Yv = np.meshgrid(xv, yv)
    for comp in components:
        comp.add(Xu, Yu, t, u, None)
        comp.add(Xv, Yv, t, None, v)
    for s in spikes:
        s.apply(u, v, dx, dx)
    # the model's words are float32
    u = u.astype(np.float32).astype(np.float64)
    v = v.astype(np.float32).astype(np.float64)
    b = BOUNDARY_CELLS * dx
    L = n * dx
    return Plane(u=u, v=v, dx=dx, dy=dx, umove=FRAME_M_S[0], vmove=FRAME_M_S[1],
                 interior=(b, L - b, b, L - b), data=(b, L - b, b, L - b),
                 ground_offset=(FRAME_M_S[0] * t, FRAME_M_S[1] * t))


def run_series(build: Callable[[float], tuple[list[Component], list[Spike]]], domain_m: float, dx: float,
               times: list[float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = []
    for t in times:
        comps, spikes = build(t)
        s = evaluate_plane(make_plane(comps, spikes, domain_m, dx, t))
        s["model_time_s"] = t
        samples.append(s)
    verdict = decide(samples, GridFacts(dx_m=dx, dy_m=dx, lowest_scalar_level_m=min(10.0, 0.5 * dx)))
    return verdict, samples


# ---------------------------------------------------------------------------
# Named cases.  Each returns (domain_m, times, builder, expected-outcome-by-grid rule).
# ---------------------------------------------------------------------------
def _series(dt: float, span: float) -> list[float]:
    return [k * dt for k in range(int(round(span / dt)) + 1)]


def case(name: str) -> tuple[float, list[float], Callable[[float], tuple[list, list]]]:
    bu, bv = background()
    L = 1600.0
    times = _series(30.0, 150.0)
    x0, y0 = 0.5 * L - 0.5 * bu * 150.0, 0.5 * L - 0.5 * bv * 150.0

    def vortex(kind, vmax, rmw, sat=False, alive=None, xy=None, drift=True):
        xx, yy = xy if xy is not None else (x0, y0)
        c = VortexComponent(kind, vmax, rmw, xx, yy, satellites=sat, alive=alive, drift=drift)
        return lambda t: ([c], [])

    def compose(*parts):
        cs = [c for c in parts]
        return lambda t: (cs, [])

    def hop(dt: float, n: int, rmw: float, spots: list[tuple[float, float]]):
        # a different short-lived vortex in each sample, never two samples running
        cs = [VortexComponent("burgers", 30.0, rmw, x, y, drift=False, alive=(dt * k - 0.25 * dt, dt * k + 0.25 * dt))
              for k, (x, y) in enumerate(spots[:n])]
        return lambda t: (cs, [])

    spots = [(500.0, 500.0), (1100.0, 1100.0), (500.0, 1100.0), (1100.0, 500.0), (800.0, 800.0)]
    table: dict[str, tuple[float, list[float], Callable]] = {
        # positives (RMW >= 4 x 20 m)
        "rankine-R100-V50": (L, times, vortex("rankine", 50.0, 100.0)),
        "burgers-R100-V60": (L, times, vortex("burgers", 60.0, 100.0)),
        "sullivan-R100-V60": (L, times, vortex("sullivan", 60.0, 100.0)),
        "ring-R120-V50": (L, times, vortex("ring", 50.0, 120.0)),
        "burgers-R100-V100+sat": (L, times, vortex("burgers", 100.0, 100.0, sat=True)),
        "sullivan-R100-V100+sat": (L, times, vortex("sullivan", 100.0, 100.0, sat=True)),
        # a tornado inside a low-level mesocyclone (amendment: closure within 3 RMW)
        "burgers-R150-V50-in-meso-R700-V30": (L, times, compose(
            VortexComponent("burgers", 50.0, 150.0, x0, y0), VortexComponent("burgers", 30.0, 700.0, x0, y0))),
        "burgers-R100-V35-in-meso-R600-V30": (L, times, compose(
            VortexComponent("burgers", 35.0, 100.0, x0, y0), VortexComponent("burgers", 30.0, 600.0, x0, y0))),
        # near the thresholds (amendment: marginal band): 20 m may only be UNRESOLVED or TORNADO
        "rankine-R90-V21.5": (L, times, vortex("rankine", 21.5, 90.0)),
        "ring-R100-V21.5": (L, times, vortex("ring", 21.5, 100.0)),
        # resolvedness guard: RMW 40 m is below 4 dx at 20 m only (10 s output: the core
        # drifts 38 m per sample in the storm-following frame, within its 80 m identity bound)
        "burgers-R40-V60": (L, _series(10.0, 150.0), vortex("burgers", 60.0, 40.0)),
        # the same vortex at 30 s output drifts 115 m per sample, more than two RMWs
        "burgers-R40-V60-30s": (L, times, vortex("burgers", 60.0, 40.0)),
        # a small tornado 600 m from a strong straight shear line whose box-mean vorticity is larger than the
        # tornado's and supplies a row of first-guess maxima (amendment: 16 first guesses)
        "burgers-R30-V35+shear-line-dU55": (2000.0, _series(10.0, 150.0), compose(
            VortexComponent("burgers", 35.0, 30.0, 1000.0 - 0.5 * bu * 150.0, 1350.0 - 0.5 * bv * 150.0),
            ShearLine(55.0, 20.0, 750.0))),
        # negative controls
        "spike-a25": (L, times, lambda t: ([], [Spike(25.0, 800.0, 800.0)])),
        "shear-line-dU80": (L, times, lambda t: ([ShearLine(80.0, 20.0, 800.0)], [])),
        "finite-sheet-L1000-dU80": (2000.0, times, compose(FiniteShearLine(80.0, 15.0, 1000.0, 1000.0, 1000.0))),
        "weak-burgers-R150-V15": (L, times, vortex("burgers", 15.0, 150.0)),
        "strong-30s-burgers-R100-V60": (L, _series(10.0, 150.0), vortex("burgers", 60.0, 100.0, alive=(60.0, 90.0))),
        "edge-burgers-R100-V60": (L, times, vortex("burgers", 60.0, 100.0, drift=False,
                                                    xy=(L - BOUNDARY_CELLS * 20.0 - 150.0, 0.5 * L))),
        "broad-meso-burgers-R1500-V25": (2800.0, _series(60.0, 120.0),
                                         vortex("burgers", 25.0, 1500.0, drift=False, xy=(1400.0, 1400.0))),
        "meso-rankine-R750-V21": (3000.0, times, vortex("rankine", 21.0, 750.0, drift=False, xy=(1500.0, 1500.0))),
        "meso-burgers-R600-V21": (3000.0, times, vortex("burgers", 21.0, 600.0, drift=False, xy=(1500.0, 1500.0))),
        # still air on the ground (the storm's motion cancelled): the only wind >= 35 m/s is a gust
        "rankine-R500-V20.5+gust-900m": (3000.0, times, compose(
            Uniform(-FRAME_M_S[0] - bu, -FRAME_M_S[1] - bv),
            VortexComponent("rankine", 20.5, 500.0, 1500.0, 1500.0, drift=False),
            GustPatch(36.0, 1500.0, 600.0, 60.0))),
        "rankine-R200-V22+gust-400m": (L, times, compose(
            Uniform(-FRAME_M_S[0] - bu, -FRAME_M_S[1] - bv),
            VortexComponent("rankine", 22.0, 200.0, 800.0, 800.0, drift=False),
            GustPatch(40.0, 800.0, 400.0, 40.0))),
        # a different short-lived vortex in each sample: persistence cannot be shown
        "hop-60s": (L, _series(60.0, 180.0), hop(60.0, 4, 60.0, spots)),
        "hop-30s": (L, _series(30.0, 120.0), hop(30.0, 5, 100.0, spots)),
    }
    return table[name]


EXPECTED = {
    "rankine-R100-V50": "TORNADO",
    "burgers-R100-V60": "TORNADO",
    "sullivan-R100-V60": "TORNADO",
    "ring-R120-V50": "TORNADO",
    "burgers-R100-V100+sat": "TORNADO",
    "sullivan-R100-V100+sat": "TORNADO",
    "burgers-R150-V50-in-meso-R700-V30": "TORNADO",
    "burgers-R100-V35-in-meso-R600-V30": "TORNADO",
    "rankine-R90-V21.5": {20.0: ("UNRESOLVED", "TORNADO"), 8.0: "TORNADO", 4.0: "TORNADO", 2.0: "TORNADO"},
    "ring-R100-V21.5": {20.0: ("UNRESOLVED", "TORNADO"), 8.0: "TORNADO", 4.0: "TORNADO", 2.0: "TORNADO"},
    "burgers-R40-V60": {20.0: "UNRESOLVED", 8.0: "TORNADO", 4.0: "TORNADO", 2.0: "TORNADO"},
    "burgers-R40-V60-30s": "INSUFFICIENT_CADENCE",
    "burgers-R30-V35+shear-line-dU55": {20.0: "UNRESOLVED", 8.0: "UNRESOLVED", 4.0: "TORNADO", 2.0: "TORNADO"},
    "spike-a25": "NO_TORNADO",
    "shear-line-dU80": "NO_TORNADO",
    "finite-sheet-L1000-dU80": "NO_TORNADO",
    "weak-burgers-R150-V15": "NO_TORNADO",
    "strong-30s-burgers-R100-V60": "NO_TORNADO",
    "edge-burgers-R100-V60": "NO_TORNADO",
    "broad-meso-burgers-R1500-V25": "NO_TORNADO",
    "meso-rankine-R750-V21": "NO_TORNADO",
    "meso-burgers-R600-V21": "NO_TORNADO",
    "rankine-R500-V20.5+gust-900m": "NO_TORNADO",
    "rankine-R200-V22+gust-400m": "NO_TORNADO",
    "hop-60s": "INSUFFICIENT_CADENCE",
    "hop-30s": "INSUFFICIENT_CADENCE",
}
GRIDS = (20.0, 8.0, 4.0, 2.0)


def expected(name: str, dx: float) -> tuple[str, ...]:
    """The acceptable outcomes of a case on a grid."""
    e = EXPECTED[name]
    e = e[dx] if isinstance(e, dict) else e
    return e if isinstance(e, tuple) else (e,)


def run_case(arg: tuple[str, float]) -> dict[str, Any]:
    try:
        os.nice(19)
    except OSError:
        pass
    name, dx = arg
    domain, times, build = case(name)
    t0 = time.time()
    verdict, samples = run_series(build, domain, dx, times)
    return {"case": name, "dx_m": dx, "outcome": verdict["outcome"], "expected": "|".join(expected(name, dx)),
            "longest_q_s": verdict["longest_qualifying_run_s"],
            "longest_qu_s": verdict["longest_qualifying_or_unresolved_run_s"],
            "classes": "".join(s["class"] for s in samples),
            "conditions": " ".join(s["conditions"] for s in samples),
            "wn2": [round(s.get("wavenumber2_fraction", math.nan), 3) for s in samples],
            "zeta_core": [round(s.get("mean_zeta_core_s_inv", math.nan), 3) for s in samples],
            "vmax": [round(s.get("vmax_m_s", math.nan), 2) for s in samples],
            "rmw": [s.get("rmw_m") for s in samples],
            "wind": [round(s.get("ground_wind_max_m_s", math.nan), 2) for s in samples],
            "closure": [round(s.get("closure_ratio", math.nan), 3) for s in samples],
            "flags": sorted({f for s in samples for f in s.get("flags", [])}),
            "wall_s": time.time() - t0}


def snapshot_case(arg: tuple[str, float, float, float]) -> dict[str, Any]:
    """Vmax and RMW of one static vortex over 16 sub-grid phases against the analytic values."""
    try:
        os.nice(19)
    except OSError:
        pass
    kind, vmax, rmw, dx = arg
    L = 1600.0
    out = []
    for fy in (0.0, 0.25, 0.5, 0.75):
        for fx in (0.0, 0.25, 0.5, 0.75):
            c = VortexComponent(kind, vmax, rmw, 800.0 + fx * dx, 800.0 + fy * dx, drift=False)
            s = evaluate_plane(make_plane([c], [], L, dx, 0.0))
            out.append((s["vmax_m_s"], s["rmw_m"], math.hypot(s["centre_x_m"] - c.x0, s["centre_y_m"] - c.y0),
                        s["ground_wind_max_m_s"]))
    a = np.array(out)
    return {"kind": kind, "vmax": vmax, "rmw": rmw, "dx": dx, "rmw_over_dx": rmw / dx,
            "vmax_err_pct": [float(100 * (a[:, 0].min() / vmax - 1)), float(100 * (a[:, 0].max() / vmax - 1))],
            "rmw_m": [float(a[:, 1].min()), float(a[:, 1].max())],
            "centre_err_m_max": float(a[:, 2].max()),
            "ground_wind": [float(a[:, 3].min()), float(a[:, 3].max())]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args(argv)
    if a.workers > 3:
        raise SystemExit("at most 3 local worker processes")
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    jobs = [(n, dx) for n in EXPECTED for dx in GRIDS] + [(n, 100.0) for n in ("burgers-R100-V60",)]
    snaps = [(k, 50.0, r, dx) for k in ("rankine", "burgers", "sullivan", "ring")
             for dx in (20.0, 8.0, 4.0, 2.0) for r in (2.0 * dx, 3.0 * dx, 4.0 * dx, 8.0 * dx)
             if r <= 400.0]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        sweep = list(pool.map(run_case, jobs))
        snap = list(pool.map(snapshot_case, snaps))
    for r in sweep:
        if r["dx_m"] == 100.0:
            r["expected"] = "UNRESOLVED"
    with (a.out / "sweep.tsv").open("w") as f:
        cols = ["case", "dx_m", "outcome", "expected", "longest_q_s", "longest_qu_s", "classes", "conditions",
                "vmax", "rmw", "wind", "closure", "wn2", "zeta_core", "flags", "wall_s"]
        f.write("\t".join(cols) + "\n")
        for r in sweep:
            f.write("\t".join(json.dumps(r[c]) if isinstance(r[c], list) else str(r[c]) for c in cols) + "\n")
    with (a.out / "snapshot.tsv").open("w") as f:
        cols = ["kind", "vmax", "rmw", "dx", "rmw_over_dx", "vmax_err_pct", "rmw_m", "centre_err_m_max",
                "ground_wind"]
        f.write("\t".join(cols) + "\n")
        for r in snap:
            f.write("\t".join(json.dumps(r[c]) if isinstance(r[c], list) else str(r[c]) for c in cols) + "\n")
    mismatches = [(r["case"], r["dx_m"], r["outcome"], r["expected"]) for r in sweep
                  if r["outcome"] not in r["expected"].split("|")]
    summary = {"schema": "gm1.criterion-v2.synthetic-prefreeze/v1", "criterion_v2_sha256": CRITERION_V2_SHA256,
               "criterion_v2": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CRITERION_V2.items()},
               "cases": len(sweep), "mismatches": mismatches, "wall_s": time.time() - t0,
               "note": "synthetic only; no GM1 run read"}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"mismatches": mismatches, "wall_s": summary["wall_s"]}))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
