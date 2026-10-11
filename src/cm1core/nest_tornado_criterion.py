"""The fixed tornado criterion, sampled on the live nest child every parent step.

GM1 EXTENSION.  CM1 r21.1 has no nesting (``records/CM1-NEST-SEAM.md``
section 1) and makes no tornado claim anywhere in ``src``.  What this module
takes from CM1 is every formula a sample reads; what it adds is where the
sample is taken, the reductions over which points, and the rules that turn
samples into a verdict.  Section 5.9 of the seam record carries the same
statement.

THE CRITERION IS FIXED (:data:`CRITERION`)
------------------------------------------
No threshold is a configuration key.  ``[nest.tornado_criterion]`` turns the
criterion on and may declare stop rules; ``config.py`` refuses a threshold
written there by name (``CM1-CFG-NEST-TORNADO-CRITERION-FIXED``).  A verdict is
therefore decided by the same rule in every receipt that carries the same
:data:`CRITERION_SHA256`, and nobody can tune it after seeing a run.

WHERE (``cm1core.nest_live.NestDriver.after_parent_step``)
----------------------------------------------------------
Every parent step, after the child has taken its ``time_refinement_ratio``
steps, synchronised and been handed the parent's ground position
(``cm1.F:3512-3513``), and **before** two-way feedback and any nest move: the
child's words are its own, on the placement they were integrated on.

ONE SAMPLE (``gm1_nest_tornado_sample``, one serial device thread)
------------------------------------------------------------------
1. **Winds at 10 m AGL**, linear in height between the two child scalar
   levels that bracket it (:func:`height_bracket`, float64 weights):
   ``u10 = w0*u(k) + w1*u(k+1)`` on u faces and ``v10`` likewise, from the
   completed ``u3d``/``v3d``.
2. **Corner vorticity**, CM1 ``src/parcel.F:1066-1067`` (the ``solve3.F:894-895``
   ``svs`` expression): ``(v10(i)-v10(i-1))/dx - (u10(j)-u10(j-1))/dy`` at the
   ``(ny+1, nx+1)`` corners, as
   :func:`cm1core.vorticity_budget.corner_vertical_vorticity` evaluates it.
3. **Centre** ``X``: the first strict maximum of that vorticity, in
   ``DO j / DO i`` order, over the corners at least 1000 m from every child
   edge (:func:`interior_corner_box`).  Non-finite corners are skipped.
4. **Disk** ``D``: the scalar points and the corners within 300 m of ``X``.
5. **Over D**: the ground-relative speed of CM1 ``src/solve3.F:864-865``
   (``sws``: face winds averaged to the scalar point, ``umove``/``vmove``
   added) and the frame-relative speed without them, maxima over the scalar
   points; ``ncore``, the corners of ``D`` with vorticity at least half the
   maximum.

The host reads nine float64 words and computes the ground position of ``X``
(domain location + child origin + corner index times ``dx``) and the core
radius ``dx*sqrt(ncore/pi)``.

QUALIFYING SAMPLES, EPISODES, VERDICT (host, :class:`CriterionLedger`)
----------------------------------------------------------------------
A sample qualifies when ``zeta_max >= 0.30 s-1`` **and** the ground-relative
wind maximum ``>= 35 m/s`` (both finite: a non-finite word fails its
condition, since the sidecar can only carry it as ``null``) **and** the core
radius ``>= 40 m`` **and** the
ground position of ``X`` moved no more than ``60 m/s * dt + 40 m`` since the
previous sample (``dt`` the model time between them; the first sample of an
integration has no previous centre and passes that condition; a sample after
one that found no centre does not).  Consecutive qualifying samples whose
first-to-last model time spans at least 60 s are an **episode**; the verdict
is **TORNADO** when one exists, genesis being its first sample.

STOP RULES (optional keys)
--------------------------
``stop_after_episode_s``: an episode has ended, no qualifying run is in
progress, and that long has passed since the last qualifying sample of the
last episode.  ``stop_if_no_tornado_by_s``: model time has reached it, no
episode exists and no qualifying run is in progress (``config.py`` refuses a
cutoff at or after ``run.duration_s``).  Either sets a request; the
integration publishes the next history frame and then ends with a
``criterion_stop`` block, in the order ``courant_stop`` follows.  "No
qualifying run in progress" holds at every sample, not only at the one that
raised the request: a qualifying sample while a request is pending withdraws
it (the ledger counts the withdrawals and keeps the last, with when and why),
so a vortex forming one sample after the request is not cut at the next
frame.  A run that becomes an episode restarts the after-episode interval and
retires the cutoff; a run that ends short of one re-raises the request at its
first non-qualifying sample.  A request
still pending at the final scheduled event ends nothing: that event is the end
of the requested duration, so the run finishes with the completed-integration
status and the block records the request as unused
(``stop_request_disposition``).  Only an integration samples: the timing
harness (``benchmark``) refuses a nest that declares the criterion, since it
has no sidecar to publish every sample to.

RESTARTS (``cm1core.nest_restart``)
-----------------------------------
The ledger is restart state of the nest child: every value a later sample's
decision reads (the previous centre, the qualifying run in progress with its
peaks, the last episode's end, the stop request pending) and every whole-run
count the verdict reports (the episodes, genesis, the condition counts) are
written bit-exactly into the child restart's scalar table
(:meth:`CriterionLedger.checkpoint`, ``nest.tornado.*``) and read
back by :meth:`CriterionLedger.restore` before the resumed run's first sample,
so a resumed run continues the uninterrupted run's ledger: a qualifying run
that straddles the restart still becomes an episode at 60 s, not 60 s after
the resume.  The resumed run's sidecar header carries the same rows as
``carried_from_restart``, so :func:`replay_time_series` seeds its ledger from
them and re-decides this leg's rows against the same history; the receipt
names what was carried (``carried_from_restart``).  A restart written by a
nest that declared no criterion carries none, and a run that declares it at
resume starts its ledger at the resume, which the receipt says.

OUTPUTS
-------
``<history>/nest-child/tornado-criterion.jsonl``: a header object (criterion,
plan, ``time_refinement_ratio``, column names), then one compact JSON array per
sample -- keyed by the parent step and carrying the placement
(``parent_i0``/``parent_j0``) it was taken on -- flushed as it is
written, so a run that dies keeps every sample it took and the last row's
``episodes`` column is its verdict so far.  :func:`replay_time_series`
re-decides the whole ledger from those rows.  The receipt's
``nest_tornado_criterion`` block carries the verdict, the episodes, the
condition counts, the sidecar's SHA-256 and row count, and
``sample_wall_seconds``: the host wall the parent step waited on the sample
(count, total, mean, maximum), so the per-step cost of the serial kernel is a
measured number in every nested receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

NEST_TORNADO_CRITERION_SCHEMA = "gm1.nest-tornado-criterion/v1"
NEST_TORNADO_KERNEL_SCHEMA = "gm1.nest-tornado-criterion-kernel/v1"
NEST_TORNADO_TIME_SERIES_SCHEMA = "gm1.nest-tornado-criterion-time-series/v1"
CRITERION_DIRECTORY = "nest-child"
CRITERION_JSONL_FILENAME = "tornado-criterion.jsonl"

#: The lateral ghost width of CM1's open-boundary velocity owners.
NG = 3
#: Words one sample writes.
SAMPLE_WORDS = 9

#: The criterion.  Fixed before any run and never tuned; not configuration.
CRITERION: Mapping[str, float] = MappingProxyType({
    "height_agl_m": 10.0,
    "edge_margin_m": 1000.0,
    "disk_radius_m": 300.0,
    "zeta_min_s_inv": 0.30,
    "ground_relative_wind_min_m_s": 35.0,
    "core_zeta_fraction": 0.5,
    "core_radius_min_m": 40.0,
    "centre_speed_max_m_s": 60.0,
    "centre_jump_allowance_m": 40.0,
    "sustain_s": 60.0,
})
#: Threshold names a document might write into ``[nest.tornado_criterion]``:
#: the constants' own and the keys an earlier draft of the table took.  Each
#: is refused as fixed rather than as merely unknown.
FIXED_THRESHOLD_NAMES: frozenset[str] = frozenset(CRITERION) | frozenset({
    "sample_interval_parent_steps", "height_m", "vorticity_s_inv",
    "ground_relative_wind_m_s", "colocation_radius_m", "min_core_radius_m",
    "edge_margin_m", "max_center_speed_m_s", "sustain_s",
})
#: The model clock is a binary64 running sum; a span or a stop time is
#: reached within the runtime's own clock tolerance (1e-9 s).
CLOCK_TOLERANCE_S = 1.0e-9
#: Bounded receipt list.
EPISODE_BUDGET = 256

#: The sidecar's per-sample columns, in order.
TIME_SERIES_COLUMNS: tuple[str, ...] = (
    "step",
    "model_time_s",
    "parent_i0",
    "parent_j0",
    "found",
    "centre_i",
    "centre_j",
    "centre_ground_x_m",
    "centre_ground_y_m",
    "zeta_max_s_inv",
    "ground_relative_wind_max_m_s",
    "frame_relative_wind_max_m_s",
    "core_corners",
    "core_radius_m",
    "centre_move_m",
    "conditions",
    "run_span_s",
    "state",
    "episodes",
)
CONDITIONS_LEGEND = {
    "Z": "zeta_max >= 0.30 s-1",
    "W": "ground-relative wind maximum over the 300 m disk >= 35 m/s",
    "C": "core radius dx*sqrt(ncore/pi) >= 40 m",
    "M": "centre moved <= 60 m/s * dt + 40 m since the previous sample",
    "-": "that condition failed (position order Z, W, C, M)",
}
STATE_LEGEND = {
    "-": "not qualifying",
    "run": "qualifying; the run of consecutive qualifying samples spans < 60 s",
    "episode": "qualifying; the run spans >= 60 s (TORNADO)",
}

WORDING = (
    "a verdict under GM1's fixed criterion on the child's grid-volume 10 m winds; "
    "not a damage rating and not an observed tornado"
)

CM1_AUTHORITY = {
    "ground_relative_speed": (
        "src/solve3.F:864-865 (sws): sqrt((umove+0.5*(ua(i)+ua(i+1)))**2 + "
        "(vmove+0.5*(va(j)+va(j+1)))**2)"
    ),
    "corner_vorticity": (
        "src/parcel.F:1066-1067 and src/solve3.F:894-895 (svs, flat Cartesian): "
        "(v(i)-v(i-1))*rdx - (u(j)-u(j-1))*rdy at the corners"
    ),
    "ground_position": "src/cm1.F:3512-3513 domainlocx/domainlocy",
    "tornado_claim": "none: CM1 r21.1 makes no tornado claim; every verdict rule is GM1's",
}

GM1_CHOICES = {
    "sample_point": (
        "NestDriver.after_parent_step, every parent step, after the child's steps, "
        "synchronize and the domain-location copy, before feedback and any nest move"
    ),
    "height": (
        "10 m AGL, linear in height between the child scalar levels zh(k) <= 10 m <= zh(k+1), "
        "float64 weights; a child whose zh(1) is above 10 m is refused"
    ),
    "precision": (
        "every product, sum, difference, quotient and square root in float64 from the "
        "float32 model words, compiled without contraction; the host reference makes "
        "the same operations in the same order"
    ),
    "centre": "first strict maximum in DO j / DO i order over corners >= 1000 m from every child edge",
    "disk": "scalar points and corners whose distance from the centre corner is <= 300 m",
    "divide": "dv/dx - du/dy with dx, dy the child's binary32 spacing (CM1 multiplies by rdx)",
    "core_radius": "dx*sqrt(ncore/pi), ncore = disk corners with zeta >= 0.5*zeta_max",
    "centre_move": (
        "ground distance from the previous sample's centre; the first sample of an "
        "integration passes; a sample after one with no centre fails"
    ),
    "resolution_note": (
        "zeta >= 0.30 s-1 is easily reached at 20 m grid spacing; the ground-relative "
        "wind, core-size and continuity conditions are what separate a tornado"
    ),
}


# ---------------------------------------------------------------------------
# Geometry.
# ---------------------------------------------------------------------------
def interior_corner_box(n: int, spacing_m: float, margin_m: float) -> tuple[int, int]:
    """Corners ``c`` on one axis with ``c*d >= margin`` and ``(n-c)*d >= margin``.

    Returns the inclusive ``(lo, hi)``; ``lo > hi`` means none.  Decided by
    exact comparison of the corner's own float64 distance, not by a rounded
    quotient.
    """

    spacing, margin = float(spacing_m), float(margin_m)
    lo = max(0, int(math.floor(margin / spacing)) - 1)
    while lo * spacing < margin:
        lo += 1
    hi = n - lo
    while hi >= 0 and (n - hi) * spacing < margin:
        hi -= 1
    return lo, hi


def height_bracket(levels: Any, height_m: float) -> tuple[int, int, float, float] | None:
    """``(k, k+1, w0, w1)``: the child scalar levels around ``height_m`` and their weights.

    ``levels.zh`` is CM1-indexed (``zh[1]`` the lowest scalar level), which is
    also the z index of the padded ``u3d``/``v3d`` owners.  ``None`` when the
    height is below ``zh(1)`` or above ``zh(nk)``: no pair of levels brackets it.
    """

    h = float(height_m)
    nk = int(levels.nk)
    for k in range(1, nk):
        lower, upper = float(levels.zh[k]), float(levels.zh[k + 1])
        if lower <= h <= upper:
            w1 = (h - lower) / (upper - lower)
            return k, k + 1, 1.0 - w1, w1
    return None


@dataclass(frozen=True, slots=True)
class SamplePlan:
    """Everything one launch of ``gm1_nest_tornado_sample`` reads besides the winds."""

    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    ku0: int
    ku1: int
    w0: float
    w1: float
    radius_m: float
    u_move_m_s: float
    v_move_m_s: float
    lo_i: int
    hi_i: int
    lo_j: int
    hi_j: int

    def owner_shapes(self) -> dict[str, tuple[int, int, int]]:
        return {
            "u": (self.nz + 2, self.ny + 2 * NG, self.nx + 1 + 2 * NG),
            "v": (self.nz + 2, self.ny + 1 + 2 * NG, self.nx + 2 * NG),
        }

    def record(self) -> dict[str, Any]:
        return {
            "nx": self.nx, "ny": self.ny, "nz": self.nz, "dx_m": self.dx_m, "dy_m": self.dy_m,
            "levels_cm1_indexed": [self.ku0, self.ku1], "height_weights": [self.w0, self.w1],
            "disk_radius_m": self.radius_m,
            "frame_velocity_m_s": [self.u_move_m_s, self.v_move_m_s],
            "search_corners": {"i": [self.lo_i, self.hi_i], "j": [self.lo_j, self.hi_j]},
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SamplePlan":
        return cls(
            nx=int(record["nx"]), ny=int(record["ny"]), nz=int(record["nz"]),
            dx_m=float(record["dx_m"]), dy_m=float(record["dy_m"]),
            ku0=int(record["levels_cm1_indexed"][0]), ku1=int(record["levels_cm1_indexed"][1]),
            w0=float(record["height_weights"][0]), w1=float(record["height_weights"][1]),
            radius_m=float(record["disk_radius_m"]),
            u_move_m_s=float(record["frame_velocity_m_s"][0]),
            v_move_m_s=float(record["frame_velocity_m_s"][1]),
            lo_i=int(record["search_corners"]["i"][0]), hi_i=int(record["search_corners"]["i"][1]),
            lo_j=int(record["search_corners"]["j"][0]), hi_j=int(record["search_corners"]["j"][1]),
        )


def child_sample_plan(*, geometry: Any, child_config: Any) -> SamplePlan:
    """The plan for this child, or ``ValueError`` naming why it cannot be sampled.

    ``config.py`` refuses both cases by name before a device is touched; this
    raises only for a caller that built a nest around that check.
    """

    import numpy

    from .nest_vertical import levels_of_grid

    levels = levels_of_grid(child_config.grid)
    bracket = height_bracket(levels, CRITERION["height_agl_m"])
    if bracket is None:
        raise ValueError(
            f"no two child scalar levels bracket {CRITERION['height_agl_m']} m "
            f"(zh(1) = {float(levels.zh[1])} m)"
        )
    margin = CRITERION["edge_margin_m"]
    lo_i, hi_i = interior_corner_box(geometry.nx, geometry.dx_m, margin)
    lo_j, hi_j = interior_corner_box(geometry.ny, geometry.dy_m, margin)
    if lo_i > hi_i or lo_j > hi_j:
        raise ValueError(f"the child has no corner {margin} m from every edge")
    ku0, ku1, w0, w1 = bracket
    return SamplePlan(
        nx=int(geometry.nx), ny=int(geometry.ny), nz=int(child_config.grid.nz),
        dx_m=float(numpy.float32(geometry.dx_m)), dy_m=float(numpy.float32(geometry.dy_m)),
        ku0=ku0, ku1=ku1, w0=w0, w1=w1, radius_m=CRITERION["disk_radius_m"],
        u_move_m_s=float(child_config.u_move_m_s), v_move_m_s=float(child_config.v_move_m_s),
        lo_i=lo_i, hi_i=hi_i, lo_j=lo_j, hi_j=hi_j,
    )


# ---------------------------------------------------------------------------
# The kernel.
# ---------------------------------------------------------------------------
NEST_TORNADO_KERNEL_SOURCE = r'''// gm1.nest-tornado-criterion-kernel/v1
// CM1 r21.1 has no nesting and makes no tornado claim; the reductions are GM1's.
// Formulas: src/solve3.F:864-865 (ground-relative speed), src/parcel.F:1066-1067
// (corner vorticity).
#define GM1_TC_IDX(k, j, i, nj, ni) \
    ((((unsigned long long)(k) * (unsigned long long)(nj)) + (unsigned long long)(j)) \
     * (unsigned long long)(ni) + (unsigned long long)(i))

__device__ __forceinline__ int gm1_tc_clamp(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// One padded velocity owner at physical (j, i), interpolated between two levels.
__device__ __forceinline__ double gm1_tc_level(const float* f, int nj, int ni, int k0, int k1,
                                               double w0, double w1, int ng, int j, int i)
{
    return w0 * (double)f[GM1_TC_IDX(k0, j + ng, i + ng, nj, ni)]
         + w1 * (double)f[GM1_TC_IDX(k1, j + ng, i + ng, nj, ni)];
}

// Corner (j, i) vorticity of the interpolated winds.
__device__ __forceinline__ double gm1_tc_zeta(const float* u, const float* v,
                                              int nju, int niu, int njv, int niv,
                                              int k0, int k1, double w0, double w1, int ng,
                                              double dx, double dy, int j, int i)
{
    double dv = gm1_tc_level(v, njv, niv, k0, k1, w0, w1, ng, j, i)
              - gm1_tc_level(v, njv, niv, k0, k1, w0, w1, ng, j, i - 1);
    double du = gm1_tc_level(u, nju, niu, k0, k1, w0, w1, ng, j, i)
              - gm1_tc_level(u, nju, niu, k0, k1, w0, w1, ng, j - 1, i);
    return dv / dx - du / dy;
}

// Serial: launch with exactly one thread.
//
// u (nz+2, ny+2ng, nx+1+2ng), v (nz+2, ny+1+2ng, nx+2ng): CM1 open-boundary
// padded owners.  The centre is the first strict maximum of the corner
// vorticity over corners [lo_i, hi_i] x [lo_j, hi_j] in DO j / DO i order.
// out: 0 found, 1 centre i, 2 centre j, 3 zeta_max, 4 ground-relative wind max,
//      5 frame-relative wind max, 6 ncore, 7 disk scalar points, 8 disk corners.
extern "C" __global__ void gm1_nest_tornado_sample(
    const float* __restrict__ u, const float* __restrict__ v, double* __restrict__ out,
    int nju, int niu, int njv, int niv, int ku0, int ku1, double w0, double w1,
    int nx, int ny, int ng, double dx, double dy, double umove, double vmove,
    int lo_i, int hi_i, int lo_j, int hi_j, double radius)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0ULL) return;
    for (int w = 0; w < 9; ++w) out[w] = 0.0;
    out[1] = -1.0; out[2] = -1.0;
    int found = 0, im = -1, jm = -1;
    double zmax = 0.0;
    for (int j = lo_j; j <= hi_j; ++j) {
        for (int i = lo_i; i <= hi_i; ++i) {
            double z = gm1_tc_zeta(u, v, nju, niu, njv, niv, ku0, ku1, w0, w1, ng, dx, dy, j, i);
            if ((z - z) == 0.0 && (!found || z > zmax)) {
                zmax = z; im = i; jm = j; found = 1;
            }
        }
    }
    if (!found) return;
    double xc = (double)im * dx;
    double yc = (double)jm * dy;
    double rr = radius * radius;
    int bi0 = (int)((xc - radius) / dx) - 1, bi1 = (int)((xc + radius) / dx) + 1;
    int bj0 = (int)((yc - radius) / dy) - 1, bj1 = (int)((yc + radius) / dy) + 1;
    int ci0 = gm1_tc_clamp(bi0, 0, nx), ci1 = gm1_tc_clamp(bi1, 0, nx);
    int cj0 = gm1_tc_clamp(bj0, 0, ny), cj1 = gm1_tc_clamp(bj1, 0, ny);
    double half = 0.5 * zmax;
    double ncore = 0.0, ncorner = 0.0;
    for (int j = cj0; j <= cj1; ++j) {
        for (int i = ci0; i <= ci1; ++i) {
            double ex = (double)i * dx - xc;
            double ey = (double)j * dy - yc;
            if (!(ex * ex + ey * ey <= rr)) continue;
            ncorner = ncorner + 1.0;
            double z = gm1_tc_zeta(u, v, nju, niu, njv, niv, ku0, ku1, w0, w1, ng, dx, dy, j, i);
            if (z >= half) ncore = ncore + 1.0;
        }
    }
    int si0 = gm1_tc_clamp(bi0, 0, nx - 1), si1 = gm1_tc_clamp(bi1, 0, nx - 1);
    int sj0 = gm1_tc_clamp(bj0, 0, ny - 1), sj1 = gm1_tc_clamp(bj1, 0, ny - 1);
    double wg = 0.0, wf = 0.0, nscalar = 0.0;
    for (int j = sj0; j <= sj1; ++j) {
        for (int i = si0; i <= si1; ++i) {
            double ex = ((double)i + 0.5) * dx - xc;
            double ey = ((double)j + 0.5) * dy - yc;
            if (!(ex * ex + ey * ey <= rr)) continue;
            nscalar = nscalar + 1.0;
            double uf = 0.5 * (gm1_tc_level(u, nju, niu, ku0, ku1, w0, w1, ng, j, i)
                             + gm1_tc_level(u, nju, niu, ku0, ku1, w0, w1, ng, j, i + 1));
            double vf = 0.5 * (gm1_tc_level(v, njv, niv, ku0, ku1, w0, w1, ng, j, i)
                             + gm1_tc_level(v, njv, niv, ku0, ku1, w0, w1, ng, j + 1, i));
            double ug = umove + uf;
            double vg = vmove + vf;
            double s = sqrt(ug * ug + vg * vg);
            double sf = sqrt(uf * uf + vf * vf);
            if (s > wg) wg = s;
            if (sf > wf) wf = sf;
        }
    }
    out[0] = 1.0;
    out[1] = (double)im;
    out[2] = (double)jm;
    out[3] = zmax;
    out[4] = wg;
    out[5] = wf;
    out[6] = ncore;
    out[7] = nscalar;
    out[8] = ncorner;
}
'''
NEST_TORNADO_KERNEL_SHA256 = hashlib.sha256(
    NEST_TORNADO_KERNEL_SOURCE.encode("utf-8")
).hexdigest()
#: One identity for the rule a verdict was decided by: the constants and the kernel.
CRITERION_SHA256 = hashlib.sha256(
    json.dumps(
        {"schema": NEST_TORNADO_CRITERION_SCHEMA, "criterion": dict(CRITERION),
         "kernel_sha256": NEST_TORNADO_KERNEL_SHA256},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def compile_nest_tornado_criterion_module(cupy: Any) -> Any:
    from .nest_live_kernels import NEST_LIVE_NVRTC_OPTIONS
    from .nvrtc_module import compile_nvrtc_module

    return compile_nvrtc_module(
        cupy, code=NEST_TORNADO_KERNEL_SOURCE, options=NEST_LIVE_NVRTC_OPTIONS,
        filename="gm1_nest_tornado_criterion.cu",
    )


def launch_tornado_sample(cupy: Any, module: Any, *, u: Any, v: Any, out: Any,
                          plan: SamplePlan, stream: Any = None, kernel: Any = None) -> None:
    """One serial launch of ``gm1_nest_tornado_sample`` in the kernel's argument order.

    ``kernel`` is the function already taken from ``module`` (the live
    criterion takes it once, at construction); absent, it is taken here.
    """

    shapes = plan.owner_shapes()
    for name, array in (("u", u), ("v", v)):
        if tuple(int(n) for n in array.shape) != shapes[name] or str(array.dtype) != "float32":
            raise ValueError(
                f"the {name} owner is {tuple(array.shape)} {array.dtype}, not {shapes[name]} "
                "float32; the kernel would index another layout"
            )
        if not array.flags.c_contiguous:
            raise ValueError(f"the {name} owner must be C-contiguous")
    if tuple(out.shape) != (SAMPLE_WORDS,) or str(out.dtype) != "float64":
        raise ValueError(f"the sample record is {SAMPLE_WORDS} float64 words")
    if not (1 <= plan.ku0 < plan.ku1 <= plan.nz):
        raise ValueError("the height bracket names levels the child does not have")
    i32, f64 = cupy.int32, cupy.float64
    function = module.get_function("gm1_nest_tornado_sample") if kernel is None else kernel
    function(
        (1,), (1,),
        (u, v, out,
         i32(shapes["u"][1]), i32(shapes["u"][2]), i32(shapes["v"][1]), i32(shapes["v"][2]),
         i32(plan.ku0), i32(plan.ku1), f64(plan.w0), f64(plan.w1),
         i32(plan.nx), i32(plan.ny), i32(NG),
         f64(plan.dx_m), f64(plan.dy_m), f64(plan.u_move_m_s), f64(plan.v_move_m_s),
         i32(plan.lo_i), i32(plan.hi_i), i32(plan.lo_j), i32(plan.hi_j),
         f64(plan.radius_m)),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# The host reference, in the kernel's own operations.
# ---------------------------------------------------------------------------
def reference_winds(u: Any, v: Any, plan: SamplePlan) -> tuple[Any, Any]:
    """The float64 10 m planes of the padded owners, ``w0*f(k) + w1*f(k+1)``."""

    import numpy

    u32 = numpy.asarray(u, dtype=numpy.float32)
    v32 = numpy.asarray(v, dtype=numpy.float32)
    with numpy.errstate(invalid="ignore", over="ignore"):
        return _reference_planes(numpy, u32, v32, plan)


def _reference_planes(numpy: Any, u32: Any, v32: Any, plan: SamplePlan) -> tuple[Any, Any]:
    u10 = numpy.add(numpy.multiply(plan.w0, u32[plan.ku0].astype(numpy.float64)),
                    numpy.multiply(plan.w1, u32[plan.ku1].astype(numpy.float64)))
    v10 = numpy.add(numpy.multiply(plan.w0, v32[plan.ku0].astype(numpy.float64)),
                    numpy.multiply(plan.w1, v32[plan.ku1].astype(numpy.float64)))
    return u10, v10


def reference_corner_vorticity(u: Any, v: Any, plan: SamplePlan) -> Any:
    """``(ny+1, nx+1)`` float64 corner vorticity of the 10 m winds."""

    import numpy

    from .vorticity_budget import corner_vertical_vorticity

    u10, v10 = reference_winds(u, v, plan)
    # corner_vertical_vorticity reads levels 1..nz of a padded owner; the
    # interpolated plane is placed at level 1 of a two-level stack.
    zeta = corner_vertical_vorticity(
        numpy.stack([u10, u10]), numpy.stack([v10, v10]), nx=plan.nx, ny=plan.ny, nz=1,
        dx=plan.dx_m, dy=plan.dy_m, xp=numpy,
    )[0]
    if zeta.dtype != numpy.float64:
        raise RuntimeError("the reference vorticity left float64; its words would not be the kernel's")
    return zeta


def reference_sample(u: Any, v: Any, *, plan: SamplePlan) -> Any:
    """The nine words ``gm1_nest_tornado_sample`` writes, computed on the host."""

    import numpy

    # Non-finite model words propagate exactly as the kernel's IEEE arithmetic
    # propagates them; the warnings NumPy raises for that are not errors here.
    with numpy.errstate(invalid="ignore", over="ignore"):
        return _reference_sample(numpy, u, v, plan)


def _reference_sample(numpy: Any, u: Any, v: Any, plan: SamplePlan) -> Any:
    out = numpy.zeros(SAMPLE_WORDS, dtype=numpy.float64)
    out[1] = out[2] = -1.0
    zeta = reference_corner_vorticity(u, v, plan)
    u10, v10 = reference_winds(u, v, plan)
    box = numpy.zeros(zeta.shape, dtype=bool)
    if plan.lo_j <= plan.hi_j and plan.lo_i <= plan.hi_i:
        box[plan.lo_j:plan.hi_j + 1, plan.lo_i:plan.hi_i + 1] = True
    usable = box & numpy.isfinite(zeta)
    if not bool(numpy.any(usable)):
        return out
    flat = int(numpy.argmax(numpy.where(usable, zeta, -numpy.inf)))
    jm, im = divmod(flat, plan.nx + 1)
    zmax = float(zeta[jm, im])
    dx, dy = float(plan.dx_m), float(plan.dy_m)
    xc, yc = float(im) * dx, float(jm) * dy
    rr = float(plan.radius_m) * float(plan.radius_m)
    exc = numpy.arange(plan.nx + 1, dtype=numpy.float64) * dx - xc
    eyc = numpy.arange(plan.ny + 1, dtype=numpy.float64) * dy - yc
    corners = numpy.add((eyc * eyc)[:, None], (exc * exc)[None, :]) <= rr
    exs = (numpy.arange(plan.nx, dtype=numpy.float64) + 0.5) * dx - xc
    eys = (numpy.arange(plan.ny, dtype=numpy.float64) + 0.5) * dy - yc
    disk = numpy.add((eys * eys)[:, None], (exs * exs)[None, :]) <= rr
    s0, s1 = NG, NG + plan.ny
    uf = numpy.multiply(0.5, numpy.add(u10[s0:s1, NG:NG + plan.nx], u10[s0:s1, NG + 1:NG + plan.nx + 1]))
    vf = numpy.multiply(0.5, numpy.add(v10[s0:s1, NG:NG + plan.nx], v10[s0 + 1:s1 + 1, NG:NG + plan.nx]))
    ug, vg = numpy.add(float(plan.u_move_m_s), uf), numpy.add(float(plan.v_move_m_s), vf)
    out[0] = 1.0
    out[1], out[2] = float(im), float(jm)
    out[3] = zmax
    for word, speed in ((4, numpy.sqrt(ug * ug + vg * vg)), (5, numpy.sqrt(uf * uf + vf * vf))):
        # ``s > max`` from 0.0 skips NaN exactly as the kernel's comparison does.
        keep = disk & (speed > 0.0)
        out[word] = float(numpy.max(speed[keep])) if bool(numpy.any(keep)) else 0.0
    out[6] = float(numpy.count_nonzero(corners & (zeta >= 0.5 * zmax)))
    out[7] = float(numpy.count_nonzero(disk))
    out[8] = float(numpy.count_nonzero(corners))
    return out


# ---------------------------------------------------------------------------
# Samples, episodes and the verdict.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CriterionSample:
    step: int
    model_time_s: float
    found: bool
    centre_i: int
    centre_j: int
    ground_x_m: float
    ground_y_m: float
    zeta_max_s_inv: float
    ground_relative_wind_m_s: float
    frame_relative_wind_m_s: float
    core_corners: int
    core_radius_m: float
    #: The child's placement in the parent when the sample was taken (a
    #: following nest moves after the sample); ``None`` outside a live nest.
    parent_i0: int | None = None
    parent_j0: int | None = None


def core_radius_m(dx_m: float, core_corners: int) -> float:
    """``dx*sqrt(n/pi)``: the radius of a disk holding ``n`` corners of ``dx`` spacing."""

    return float(dx_m) * math.sqrt(float(core_corners) / math.pi)


def sample_from_words(words: Any, *, step: int, model_time_s: float, dx_m: float, dy_m: float,
                      origin_x_m: float, origin_y_m: float, domainlocx_m: float,
                      domainlocy_m: float, parent_i0: int | None = None,
                      parent_j0: int | None = None) -> CriterionSample:
    """The host sample of nine kernel words at one placement."""

    found = bool(float(words[0]) == 1.0)
    i, j = int(words[1]), int(words[2])
    return CriterionSample(
        parent_i0=None if parent_i0 is None else int(parent_i0),
        parent_j0=None if parent_j0 is None else int(parent_j0),
        step=int(step), model_time_s=float(model_time_s), found=found,
        centre_i=i if found else -1, centre_j=j if found else -1,
        ground_x_m=(float(domainlocx_m) + float(origin_x_m) + float(i) * float(dx_m)) if found else math.nan,
        ground_y_m=(float(domainlocy_m) + float(origin_y_m) + float(j) * float(dy_m)) if found else math.nan,
        zeta_max_s_inv=float(words[3]) if found else math.nan,
        ground_relative_wind_m_s=float(words[4]),
        frame_relative_wind_m_s=float(words[5]),
        core_corners=int(words[6]),
        core_radius_m=core_radius_m(dx_m, int(words[6])),
    )


@dataclass
class _Run:
    number: int
    start_s: float
    last_s: float
    start_xy: tuple[float, float]
    last_xy: tuple[float, float]
    samples: int = 0
    episode: bool = False
    peak_zeta: tuple[float, float] = (-math.inf, 0.0)
    peak_wg: tuple[float, float] = (-math.inf, 0.0)
    peak_wf: tuple[float, float] = (-math.inf, 0.0)
    peak_core_radius: tuple[float, float] = (-math.inf, 0.0)

    @property
    def span_s(self) -> float:
        return self.last_s - self.start_s

    def extend(self, s: CriterionSample) -> None:
        t = s.model_time_s
        self.last_s, self.last_xy = t, (s.ground_x_m, s.ground_y_m)
        self.samples += 1
        for name, value in (("peak_zeta", s.zeta_max_s_inv), ("peak_wg", s.ground_relative_wind_m_s),
                            ("peak_wf", s.frame_relative_wind_m_s),
                            ("peak_core_radius", s.core_radius_m)):
            # Finite only: the sidecar carries a non-finite word as null, and
            # a replay must reach the same peaks from the rows.
            if math.isfinite(value) and value > getattr(self, name)[0]:
                setattr(self, name, (float(value), t))

    def record(self) -> dict[str, Any]:
        def peak(pair: tuple[float, float]) -> dict[str, Any]:
            # A receipt is strict JSON: a blown-up word is recorded as null.
            return {"value": _finite(pair[0]), "model_time_s": pair[1]}

        span = self.span_s
        return {
            "number": self.number,
            "start_model_time_s": self.start_s,
            "end_model_time_s": self.last_s,
            "duration_s": span,
            "samples": self.samples,
            "start_ground_m": [_finite(v) for v in self.start_xy],
            "end_ground_m": [_finite(v) for v in self.last_xy],
            "mean_centre_ground_velocity_m_s": (
                None if span <= 0.0 else [(self.last_xy[0] - self.start_xy[0]) / span,
                                          (self.last_xy[1] - self.start_xy[1]) / span]
            ),
            "peak_zeta_s_inv": peak(self.peak_zeta),
            "peak_ground_relative_wind_m_s": peak(self.peak_wg),
            "peak_frame_relative_wind_m_s": peak(self.peak_wf),
            "peak_core_radius_m": peak(self.peak_core_radius),
        }


class CriterionLedger:
    """Deterministic host rules: samples in; conditions, episodes, verdict and stop requests out."""

    def __init__(self, stop_rules: Any = None) -> None:
        self.stop_after_episode_s = (
            None if stop_rules is None else getattr(stop_rules, "stop_after_episode_s", None))
        self.stop_if_no_tornado_by_s = (
            None if stop_rules is None else getattr(stop_rules, "stop_if_no_tornado_by_s", None))
        self.samples = 0
        self.first_s: float | None = None
        self.last_s: float | None = None
        self.previous: tuple[float, bool, float, float] | None = None
        self.run: _Run | None = None
        self.runs_started = 0
        self.longest_run_s = 0.0
        self.counts = {"zeta": 0, "ground_relative_wind": 0, "core_radius": 0,
                       "centre_move": 0, "qualifying": 0, "no_centre": 0}
        #: The closed episodes, as the runs they were (recorded at report time).
        self.episodes: list[_Run] = []
        self.episode_count = 0
        self.episodes_dropped = 0
        self.last_episode_end_s: float | None = None
        self.genesis_s: float | None = None
        self.declared_s: float | None = None
        self.stop_request: dict[str, Any] | None = None
        #: Requests withdrawn because a qualifying run began while they were
        #: pending: a count, and the last one with when and why.
        self.stop_requests_withdrawn = 0
        self.last_withdrawn_stop_request: dict[str, Any] | None = None

    @property
    def tornado(self) -> bool:
        return self.genesis_s is not None

    def observe(self, s: CriterionSample) -> tuple[str, float | None, str, float | None]:
        """One sample: ``(conditions, centre_move_m, state, run_span_s)``."""

        c = CRITERION
        t = float(s.model_time_s)
        self.samples += 1
        if self.first_s is None:
            self.first_s = t
        self.last_s = t
        # A non-finite maximum fails its condition.  The sidecar row carries it
        # as null (strict JSON), so a replay could never see it pass; a blown-up
        # wind is also not evidence of a vortex.
        zeta = bool(s.found and math.isfinite(s.zeta_max_s_inv)
                    and s.zeta_max_s_inv >= c["zeta_min_s_inv"])
        wind = bool(s.found and math.isfinite(s.ground_relative_wind_m_s)
                    and s.ground_relative_wind_m_s >= c["ground_relative_wind_min_m_s"])
        core = bool(s.found and s.core_radius_m >= c["core_radius_min_m"])
        move: float | None = None
        if not s.found:
            moved = False
            self.counts["no_centre"] += 1
        elif self.previous is None:
            moved = True
        elif not self.previous[1]:
            moved = False
        else:
            tp, _, xp, yp = self.previous
            move = math.hypot(s.ground_x_m - xp, s.ground_y_m - yp)
            moved = move <= c["centre_speed_max_m_s"] * (t - tp) + c["centre_jump_allowance_m"]
        self.previous = (t, bool(s.found), float(s.ground_x_m), float(s.ground_y_m))
        qualifies = zeta and wind and core and moved
        for name, ok in (("zeta", zeta), ("ground_relative_wind", wind), ("core_radius", core),
                         ("centre_move", moved), ("qualifying", qualifies)):
            self.counts[name] += int(ok)
        conditions = "".join(letter if ok else "-" for letter, ok in
                             (("Z", zeta), ("W", wind), ("C", core), ("M", moved)))
        span: float | None = None
        if qualifies:
            run = self.run
            if run is None:
                self.runs_started += 1
                xy = (float(s.ground_x_m), float(s.ground_y_m))
                run = self.run = _Run(number=self.runs_started, start_s=t, last_s=t,
                                      start_xy=xy, last_xy=xy)
            run.extend(s)
            span = run.span_s
            self.longest_run_s = max(self.longest_run_s, span)
            if not run.episode and span >= c["sustain_s"] - CLOCK_TOLERANCE_S:
                run.episode = True
                self.episode_count += 1
                if self.genesis_s is None:
                    self.genesis_s, self.declared_s = run.start_s, t
            state = "episode" if run.episode else "run"
        else:
            self._close()
            state = "-"
        self._stop_rules(s, t)
        return conditions, move, state, span

    def _close(self) -> None:
        run, self.run = self.run, None
        if run is None or not run.episode:
            return
        self.last_episode_end_s = run.last_s
        if len(self.episodes) < EPISODE_BUDGET:
            self.episodes.append(run)
        else:
            self.episodes_dropped += 1

    def _stop_rules(self, s: CriterionSample, t: float) -> None:
        if self.run is not None:
            # A qualifying run is in progress: no rule fires, and a request
            # raised before this run began is withdrawn.  Both rules read "no
            # qualifying run in progress" at every sample, not only at the one
            # that raised the request: a request that outlived its condition
            # would end the run at the next history event while a vortex was
            # forming.  What follows is decided sample by sample, as before: a
            # run that becomes an episode moves last_episode_end_s (a new
            # interval for stop_after_episode_s; stop_if_no_tornado_by_s never
            # fires again), and a run that ends short of one re-raises the
            # request at its first non-qualifying sample.
            request = self.stop_request
            if request is not None:
                self.stop_request = None
                self.stop_requests_withdrawn += 1
                self.last_withdrawn_stop_request = {
                    **request,
                    "withdrawn_at_step": int(s.step), "withdrawn_at_model_time_s": t,
                    "withdrawn_because": STOP_WITHDRAWN_BY_RUN,
                }
            return
        if self.stop_request is not None:
            return
        after = self.stop_after_episode_s
        if (after is not None and self.last_episode_end_s is not None
                and t - self.last_episode_end_s >= after - CLOCK_TOLERANCE_S):
            self.stop_request = {
                "rule": "stop_after_episode_s", "threshold_s": after,
                "last_episode_end_model_time_s": self.last_episode_end_s,
                "requested_at_step": int(s.step), "requested_at_model_time_s": t,
            }
            return
        cutoff = self.stop_if_no_tornado_by_s
        if cutoff is not None and not self.tornado and t >= cutoff - CLOCK_TOLERANCE_S:
            self.stop_request = {
                "rule": "stop_if_no_tornado_by_s", "threshold_s": cutoff,
                "requested_at_step": int(s.step), "requested_at_model_time_s": t,
            }

    # -- restart state ------------------------------------------------------------
    def checkpoint(self) -> dict[str, tuple[str, str]]:
        """Every field a later decision or the report reads, as typed scalar rows.

        The rows are CM1FCP1 scalar-table rows (``u64``, ``bool``, ``f64-bits``:
        binary64 as sixteen hex digits, so an infinite peak or a signed zero
        survives), named ``nest.tornado.*`` (:data:`CHECKPOINT_PREFIX`; at most
        64 characters, the envelope's scalar-name limit).
        :meth:`restore` reads them back to the same ledger.
        """

        P = CHECKPOINT_PREFIX
        rows: dict[str, tuple[str, str]] = {
            P + "schema": _u64(CHECKPOINT_SCHEMA_VERSION),
            P + "samples": _u64(self.samples),
            P + "runs_started": _u64(self.runs_started),
            P + "longest_run_s": _f64(self.longest_run_s),
            P + "episode_count": _u64(self.episode_count),
            P + "episodes_dropped": _u64(self.episodes_dropped),
            P + "stop_requests_withdrawn": _u64(self.stop_requests_withdrawn),
            **_optional_f64(P + "first_sample_s", self.first_s),
            **_optional_f64(P + "last_sample_s", self.last_s),
            **_optional_f64(P + "last_episode_end_s", self.last_episode_end_s),
            **_optional_f64(P + "genesis_s", self.genesis_s),
            **_optional_f64(P + "declared_s", self.declared_s),
        }
        for name in CONDITION_COUNT_NAMES:
            rows[P + f"counts.{name}"] = _u64(self.counts[name])
        previous = self.previous
        rows[P + "previous"] = _bool(previous is not None)
        if previous is not None:
            t, found, x, y = previous
            rows[P + "previous.model_time_s"] = _f64(t)
            rows[P + "previous.found"] = _bool(found)
            rows[P + "previous.ground_x_m"] = _f64(x)
            rows[P + "previous.ground_y_m"] = _f64(y)
        rows[P + "run"] = _bool(self.run is not None)
        if self.run is not None:
            rows.update(_run_rows(P + "run.", self.run))
        rows[P + "episodes"] = _u64(len(self.episodes))
        for n, run in enumerate(self.episodes):
            rows.update(_run_rows(P + f"episode.{n}.", run))
        rows.update(_request_rows(P + "stop_request.", self.stop_request))
        rows.update(_request_rows(P + "withdrawn.", self.last_withdrawn_stop_request))
        return rows

    def restore(self, scalars: Mapping[str, tuple[str, str]]) -> dict[str, Any]:
        """Continue from :meth:`checkpoint` rows; a missing or mistyped row is refused by name.

        Returns what was carried, for the receipt.  The ledger must not have
        observed a sample yet.
        """

        if self.samples:
            raise ValueError("the tornado-criterion ledger has already observed samples; "
                             "restoring over them would decide the verdict twice")
        P = CHECKPOINT_PREFIX
        read = _ScalarReader(scalars)
        version = read.u64(P + "schema")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"the child restart's tornado-criterion ledger is schema {version}, not "
                f"{CHECKPOINT_SCHEMA_VERSION}; a resume would read its rows by another layout"
            )
        self.samples = read.u64(P + "samples")
        self.runs_started = read.u64(P + "runs_started")
        self.longest_run_s = read.f64(P + "longest_run_s")
        self.episode_count = read.u64(P + "episode_count")
        self.episodes_dropped = read.u64(P + "episodes_dropped")
        self.stop_requests_withdrawn = read.u64(P + "stop_requests_withdrawn")
        self.first_s = read.optional_f64(P + "first_sample_s")
        self.last_s = read.optional_f64(P + "last_sample_s")
        self.last_episode_end_s = read.optional_f64(P + "last_episode_end_s")
        self.genesis_s = read.optional_f64(P + "genesis_s")
        self.declared_s = read.optional_f64(P + "declared_s")
        self.counts = {name: read.u64(P + f"counts.{name}") for name in CONDITION_COUNT_NAMES}
        self.previous = None
        if read.boolean(P + "previous"):
            self.previous = (
                read.f64(P + "previous.model_time_s"), read.boolean(P + "previous.found"),
                read.f64(P + "previous.ground_x_m"), read.f64(P + "previous.ground_y_m"),
            )
        self.run = _run_from_rows(read, P + "run.") if read.boolean(P + "run") else None
        self.episodes = [
            _run_from_rows(read, P + f"episode.{n}.") for n in range(read.u64(P + "episodes"))
        ]
        self.stop_request = _request_from_rows(read, P + "stop_request.")
        self.last_withdrawn_stop_request = _request_from_rows(read, P + "withdrawn.")
        if self.last_withdrawn_stop_request is not None:
            self.last_withdrawn_stop_request["withdrawn_because"] = STOP_WITHDRAWN_BY_RUN
        return self.carried_record()

    def carried_record(self) -> dict[str, Any]:
        """What a resumed ledger starts from, for the receipt and the sidecar header."""

        return {
            "samples": self.samples,
            "first_sample_model_time_s": self.first_s,
            "last_sample_model_time_s": self.last_s,
            "verdict_at_restart": "TORNADO" if self.tornado else "NO_TORNADO",
            "genesis_model_time_s": self.genesis_s,
            "episode_count": self.episode_count,
            "qualifying_run_in_progress": None if self.run is None else self.run.record(),
            "stop_request": None if self.stop_request is None else dict(self.stop_request),
        }

    def record(self) -> dict[str, Any]:
        episodes = [run.record() for run in self.episodes]
        if self.run is not None and self.run.episode:
            episodes.append({**self.run.record(), "in_progress_at_last_sample": True})
        return {
            "verdict": "TORNADO" if self.tornado else "NO_TORNADO",
            "genesis_model_time_s": self.genesis_s,
            "tornado_declared_model_time_s": self.declared_s,
            "samples": self.samples,
            "first_sample_model_time_s": self.first_s,
            "last_sample_model_time_s": self.last_s,
            "condition_counts": dict(self.counts),
            "qualifying_runs_started": self.runs_started,
            "longest_qualifying_run_s": self.longest_run_s,
            "episode_count": self.episode_count,
            "episodes": episodes,
            "episodes_dropped": self.episodes_dropped,
            "stop_request": self.stop_request,
            "stop_requests_withdrawn": self.stop_requests_withdrawn,
            "last_withdrawn_stop_request": self.last_withdrawn_stop_request,
        }


#: :meth:`CriterionLedger.checkpoint` row names, in the child restart's scalar
#: table and the resumed sidecar's header.
CHECKPOINT_PREFIX = "nest.tornado."
CHECKPOINT_SCHEMA_VERSION = 1
CONDITION_COUNT_NAMES = ("zeta", "ground_relative_wind", "core_radius", "centre_move",
                         "qualifying", "no_centre")
#: ``stop_request["rule"]`` by restart code, in declaration order (append only).
STOP_RULE_CODES = ("stop_after_episode_s", "stop_if_no_tornado_by_s")
_PEAKS = ("peak_zeta", "peak_wg", "peak_wf", "peak_core_radius")


def _u64(value: int) -> tuple[str, str]:
    value = int(value)
    if value < 0:
        raise ValueError(f"a u64 restart scalar cannot hold {value}")
    return ("u64", str(value))


def _f64(value: float) -> tuple[str, str]:
    return ("f64-bits", f"{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}")


def _bool(value: bool) -> tuple[str, str]:
    return ("bool", "true" if value else "false")


def _optional_f64(name: str, value: float | None) -> dict[str, tuple[str, str]]:
    rows = {name + ".set": _bool(value is not None)}
    if value is not None:
        rows[name] = _f64(value)
    return rows


def _run_rows(prefix: str, run: "_Run") -> dict[str, tuple[str, str]]:
    rows = {
        prefix + "number": _u64(run.number),
        prefix + "start_s": _f64(run.start_s),
        prefix + "last_s": _f64(run.last_s),
        prefix + "start_x_m": _f64(run.start_xy[0]),
        prefix + "start_y_m": _f64(run.start_xy[1]),
        prefix + "last_x_m": _f64(run.last_xy[0]),
        prefix + "last_y_m": _f64(run.last_xy[1]),
        prefix + "samples": _u64(run.samples),
        prefix + "episode": _bool(run.episode),
    }
    for name in _PEAKS:
        value, at = getattr(run, name)
        rows[prefix + f"{name}.value"] = _f64(value)
        rows[prefix + f"{name}.at_s"] = _f64(at)
    return rows


def _request_rows(prefix: str, request: Mapping[str, Any] | None) -> dict[str, tuple[str, str]]:
    rows = {prefix.rstrip(".") : _bool(request is not None)}
    if request is None:
        return rows
    rule = request["rule"]
    if rule not in STOP_RULE_CODES:
        raise ValueError(f"no restart code is assigned to stop rule {rule!r}")
    rows[prefix + "rule_code"] = _u64(STOP_RULE_CODES.index(rule))
    rows[prefix + "threshold_s"] = _f64(request["threshold_s"])
    rows[prefix + "requested_at_step"] = _u64(request["requested_at_step"])
    rows[prefix + "requested_at_s"] = _f64(request["requested_at_model_time_s"])
    rows.update(_optional_f64(prefix + "last_episode_end_s",
                              request.get("last_episode_end_model_time_s")))
    rows[prefix + "withdrawn"] = _bool("withdrawn_at_step" in request)
    if "withdrawn_at_step" in request:
        rows[prefix + "withdrawn_at_step"] = _u64(request["withdrawn_at_step"])
        rows[prefix + "withdrawn_at_s"] = _f64(request["withdrawn_at_model_time_s"])
    return rows


class _ScalarReader:
    def __init__(self, scalars: Mapping[str, tuple[str, str]]) -> None:
        self.scalars = scalars

    def get(self, name: str, kind: str) -> str:
        try:
            found, value = self.scalars[name]
        except KeyError:
            raise ValueError(
                f"the nest child restart carries no {name}; a resume without it would "
                "continue a tornado-criterion ledger it does not know"
            ) from None
        if found != kind:
            raise ValueError(f"nest restart scalar {name} is {found}, not {kind}")
        return value

    def u64(self, name: str) -> int:
        return int(self.get(name, "u64"))

    def f64(self, name: str) -> float:
        return struct.unpack("<d", struct.pack("<Q", int(self.get(name, "f64-bits"), 16)))[0]

    def boolean(self, name: str) -> bool:
        value = self.get(name, "bool")
        if value not in ("true", "false"):
            raise ValueError(f"nest restart scalar {name} is {value!r}, not a bool")
        return value == "true"

    def optional_f64(self, name: str) -> float | None:
        return self.f64(name) if self.boolean(name + ".set") else None


def _run_from_rows(read: _ScalarReader, prefix: str) -> "_Run":
    run = _Run(
        number=read.u64(prefix + "number"),
        start_s=read.f64(prefix + "start_s"), last_s=read.f64(prefix + "last_s"),
        start_xy=(read.f64(prefix + "start_x_m"), read.f64(prefix + "start_y_m")),
        last_xy=(read.f64(prefix + "last_x_m"), read.f64(prefix + "last_y_m")),
        samples=read.u64(prefix + "samples"), episode=read.boolean(prefix + "episode"),
    )
    for name in _PEAKS:
        setattr(run, name, (read.f64(prefix + f"{name}.value"),
                            read.f64(prefix + f"{name}.at_s")))
    return run


def _request_from_rows(read: _ScalarReader, prefix: str) -> dict[str, Any] | None:
    if not read.boolean(prefix.rstrip(".")):
        return None
    code = read.u64(prefix + "rule_code")
    if code >= len(STOP_RULE_CODES):
        raise ValueError(f"nest restart stop-rule code {code} names no rule")
    request: dict[str, Any] = {
        "rule": STOP_RULE_CODES[code],
        "threshold_s": read.f64(prefix + "threshold_s"),
    }
    last_end = read.optional_f64(prefix + "last_episode_end_s")
    if last_end is not None:
        request["last_episode_end_model_time_s"] = last_end
    request["requested_at_step"] = read.u64(prefix + "requested_at_step")
    request["requested_at_model_time_s"] = read.f64(prefix + "requested_at_s")
    if read.boolean(prefix + "withdrawn"):
        request["withdrawn_at_step"] = read.u64(prefix + "withdrawn_at_step")
        request["withdrawn_at_model_time_s"] = read.f64(prefix + "withdrawn_at_s")
    return request


def time_series_row(s: CriterionSample, outcome: tuple[str, float | None, str, float | None],
                    episodes: int) -> list[Any]:
    """One compact sidecar row in :data:`TIME_SERIES_COLUMNS` order."""

    conditions, move, state, span = outcome
    found = s.found
    return [
        int(s.step), float(s.model_time_s), s.parent_i0, s.parent_j0, found,
        s.centre_i if found else None, s.centre_j if found else None,
        float(s.ground_x_m) if found else None, float(s.ground_y_m) if found else None,
        _finite(s.zeta_max_s_inv) if found else None,
        _finite(s.ground_relative_wind_m_s), _finite(s.frame_relative_wind_m_s),
        int(s.core_corners), float(s.core_radius_m), move, conditions, span, state, int(episodes),
    ]


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


class TimeSeriesWriter:
    """``tornado-criterion.jsonl``: a header object, then one flushed array per sample."""

    def __init__(self, path: Path, header: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("xb")
        self._digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0
        self.closed = False
        self._line(dict(header))

    def _line(self, value: Any) -> None:
        line = (json.dumps(value, allow_nan=False, sort_keys=isinstance(value, dict),
                           separators=(",", ":")) + "\n").encode("utf-8")
        self._handle.write(line)
        self._handle.flush()
        self._digest.update(line)
        self.bytes += len(line)

    def write(self, row: list[Any]) -> None:
        self._line(row)
        self.rows += 1

    def close(self) -> dict[str, Any]:
        if not self.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self.closed = True
        return {"path": os.fspath(self.path), "sha256": self._digest.hexdigest(),
                "rows": self.rows, "bytes": self.bytes}

    def discard(self) -> None:
        if not self.closed:
            self._handle.close()
            self.closed = True
        try:
            self.path.unlink()
        except OSError:
            pass


def criterion_stop_record(event: Mapping[str, Any], *, frame_directory: str,
                          frames_requested: int, frames_published: int) -> dict[str, Any]:
    """The ``criterion_stop`` receipt block, written once the event's frame is on disk.

    ``frame_directory`` is the published frame's directory: its row's when the
    frame rows are within budget, else the formatted directory name, the
    fallback ``courant_stop`` takes.
    """

    request = dict(event["request"])
    return {
        "rule": request["rule"],
        "request": request,
        "frame_index": event["frame_index"],
        "frame_directory": str(frame_directory),
        "restart_number": event.get("restart_number"),
        "step": int(event["step"]),
        "model_time_s": float(event["model_time_s"]),
        "frames_requested": int(frames_requested),
        "frames_published": int(frames_published),
        "note": (
            "the declared nest.tornado_criterion stop rule fired; the history frame of "
            "the next event was published and the run then ended, the order "
            "courant_stop follows"
        ),
    }


#: ``stop_request_disposition`` words.
STOP_HONOURED = "honoured-at-the-next-history-event"
STOP_UNUSED_AT_DURATION = (
    "unused-pending-at-the-final-scheduled-event-the-requested-duration-was-integrated"
)
STOP_UNUSED_AT_COURANT = "unused-the-run-ended-at-cm1-advective-courant-stop"
#: ``last_withdrawn_stop_request.withdrawn_because``: the one reason a
#: pending request is withdrawn (:meth:`CriterionLedger._stop_rules`).
STOP_WITHDRAWN_BY_RUN = "a-qualifying-run-began-while-the-request-was-pending"


class SampleWall:
    """Host wall seconds the parent step waits on the sample: count, total, maximum.

    Measured around ``sample`` (the launch and the blocking read of nine
    words).  ``NestDriver.after_parent_step`` synchronises the child just
    before, so this is the sample's own cost and the per-step tax the
    criterion adds to a nested run, reported in every receipt rather than
    inferred from a second run without the table.
    """

    __slots__ = ("count", "total", "maximum")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total += seconds
        if seconds > self.maximum:
            self.maximum = seconds

    def record(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total": self.total,
            "mean": None if self.count == 0 else self.total / self.count,
            "maximum": None if self.count == 0 else self.maximum,
            "measured": "host wall around the launch and the read of the nine words, "
                        "the child synchronised just before",
        }


# ---------------------------------------------------------------------------
# The live criterion.
# ---------------------------------------------------------------------------
class NestTornadoCriterion:
    """Sample, decide and record, once every parent step."""

    def __init__(self, *, settings: Any, driver: Any, xp: Any, stream: Any,
                 module: Any = None) -> None:
        self.settings, self.xp, self.stream = settings, xp, stream
        self.plan = child_sample_plan(geometry=driver.geometry, child_config=driver.child.config)
        from .nest_vertical import levels_of_grid

        levels = levels_of_grid(driver.child.config.grid)
        self.bracket_heights_m = [float(levels.zh[self.plan.ku0]), float(levels.zh[self.plan.ku1])]
        self.time_refinement_ratio = int(driver.geometry.time_refinement_ratio)
        with self._on_stream():
            self.words = xp.zeros(SAMPLE_WORDS, dtype=xp.float64)
        self.module = compile_nest_tornado_criterion_module(xp) if module is None else module
        # The module is compiled above and the kernel taken here, when the
        # nest is attached, so an NVRTC failure refuses the run at startup
        # rather than after the first parent step has been paid for.
        self.kernel = self.module.get_function("gm1_nest_tornado_sample")
        self.ledger = CriterionLedger(settings)
        #: The last sample taken, whether or not it found a centre.  A
        #: vortex-following consumer -- cm1core.nest_child_window -- reads it
        #: rather than re-running the search on words it does not hold.
        self.last_sample: CriterionSample | None = None
        self.writer: TimeSeriesWriter | None = None
        self.time_series: list[dict[str, Any]] = []
        self.launches = 0
        self.sample_wall = SampleWall()
        #: Set by the integration: what became of a stop request (None: none).
        self.stop_request_disposition: str | None = None
        #: A resumed run: the ledger rows carried from the child restart
        #: (:meth:`restore_from_restart`) and what they say, else None.
        self.carried_rows: dict[str, tuple[str, str]] | None = None
        self.carried: dict[str, Any] | None = None

    def restore_from_restart(self, scalars: Mapping[str, tuple[str, str]]) -> dict[str, Any]:
        """Continue the ledger from a child restart's ``nest.tornado.*`` rows, before any sample."""

        if self.writer is not None:
            raise RuntimeError("the tornado-criterion time series is already open; the ledger "
                               "must be restored before the header is written")
        self.carried_rows = {name: (kind, value) for name, (kind, value) in scalars.items()}
        self.carried = self.ledger.restore(self.carried_rows)
        return {"carried_from_restart": dict(self.carried)}

    def _on_stream(self) -> Any:
        return self.stream if self.stream is not None else nullcontext()

    # -- output ---------------------------------------------------------------
    def header(self) -> dict[str, Any]:
        return {
            "schema": NEST_TORNADO_TIME_SERIES_SCHEMA,
            "criterion_schema": NEST_TORNADO_CRITERION_SCHEMA,
            "criterion": dict(CRITERION),
            "criterion_sha256": CRITERION_SHA256,
            "kernel_sha256": NEST_TORNADO_KERNEL_SHA256,
            "plan": self.plan.record(),
            "bracket_heights_m": list(self.bracket_heights_m),
            "time_refinement_ratio": self.time_refinement_ratio,
            "stop_rules": self.settings.record() if self.settings is not None else None,
            "columns": list(TIME_SERIES_COLUMNS),
            "conditions_legend": dict(CONDITIONS_LEGEND),
            "state_legend": dict(STATE_LEGEND),
            "rows": (
                "one JSON array per parent step after this header, in model order; step is "
                "the parent's completed_steps (a child frame's placement step is "
                "time_refinement_ratio times it)"
            ),
            # A resumed run: the ledger the rows below continue, as the child
            # restart's typed scalar rows, so a replay seeds the same ledger.
            **({} if self.carried_rows is None else {
                "carried_from_restart": {name: list(row) for name, row in self.carried_rows.items()},
            }),
        }

    def open(self, history_root: Path | str) -> None:
        """Open ``<history>/nest-child/tornado-criterion.jsonl`` and write its header."""

        if self.writer is not None and not self.writer.closed:
            raise RuntimeError("the tornado-criterion time series is already open")
        path = Path(history_root) / CRITERION_DIRECTORY / CRITERION_JSONL_FILENAME
        self.writer = TimeSeriesWriter(path, self.header())

    def close(self) -> dict[str, Any]:
        """Close the open sidecar (fsync) and return the receipt block."""

        if self.writer is not None and not self.writer.closed:
            self.time_series.append(self.writer.close())
        return self.record()

    def abandon(self) -> None:
        """A failed integration: keep a sidecar that holds samples, remove one that holds none.

        A sidecar with rows is the only record of a run that died, so it is
        closed in place.  One with only its header would keep the history root
        from being empty, and ``require_new_history_directory`` would then
        refuse the corrected retry at the same path.
        """

        writer = self.writer
        if writer is None or writer.closed:
            return
        if writer.rows == 0:
            writer.discard()
            try:
                writer.path.parent.rmdir()
            except OSError:
                pass
            self.writer = None
            return
        self.time_series.append(writer.close())

    # -- one parent step --------------------------------------------------------
    def after_parent_step(self, driver: Any) -> list[Any]:
        clock = driver.parent.clock
        g = driver.geometry
        started = time.perf_counter()
        words = self.sample(driver)
        self.sample_wall.add(time.perf_counter() - started)
        sample = sample_from_words(
            words, step=int(clock.completed_steps), model_time_s=float(clock.model_seconds),
            dx_m=float(g.dx_m), dy_m=float(g.dy_m), origin_x_m=float(g.origin_x_m),
            origin_y_m=float(g.origin_y_m), domainlocx_m=float(clock.domainlocx_m),
            domainlocy_m=float(clock.domainlocy_m),
            parent_i0=int(g.parent_i0), parent_j0=int(g.parent_j0),
        )
        self.last_sample = sample
        outcome = self.ledger.observe(sample)
        row = time_series_row(sample, outcome, self.ledger.episode_count)
        if self.writer is not None and not self.writer.closed:
            self.writer.write(row)
        return row

    def sample(self, driver: Any) -> list[float]:
        """One launch, then one read of nine words."""

        child = driver.child.dynamics
        with self._on_stream():
            launch_tornado_sample(
                self.xp, self.module, u=child.arrays["u3d"], v=child.arrays["v3d"],
                out=self.words, plan=self.plan, stream=self.stream, kernel=self.kernel,
            )
        self.launches += 1
        words = self.words
        host = (words.get(stream=self.stream) if self.stream is not None else words.get()) \
            if hasattr(words, "get") else words
        return [float(w) for w in host]

    # -- receipt ----------------------------------------------------------------
    @property
    def stop_request(self) -> dict[str, Any] | None:
        return self.ledger.stop_request

    def record(self) -> dict[str, Any]:
        return {
            "schema": NEST_TORNADO_CRITERION_SCHEMA,
            "gm1_extension": (
                "CM1 r21.1 has no nesting and makes no tornado claim; the reductions, "
                "episode rule and verdict are GM1's (records/CM1-NEST-SEAM.md section 5.9)"
            ),
            "criterion": dict(CRITERION),
            "criterion_sha256": CRITERION_SHA256,
            "kernel_schema": NEST_TORNADO_KERNEL_SCHEMA,
            "kernel_sha256": NEST_TORNADO_KERNEL_SHA256,
            "cm1_authority": dict(CM1_AUTHORITY),
            "gm1_choices": dict(GM1_CHOICES),
            "plan": self.plan.record(),
            "bracket_heights_m": list(self.bracket_heights_m),
            "stop_rules": self.settings.record() if self.settings is not None else None,
            **self.ledger.record(),
            "stop_request_disposition": self.stop_request_disposition,
            "device_launches": self.launches,
            "sample_wall_seconds": self.sample_wall.record(),
            "time_series": list(self.time_series),
            "wording": WORDING,
            "carried_from_restart": None if self.carried is None else dict(self.carried),
            "verdict_scope": (
                "the samples this runtime took, first to last sample model time"
                if self.carried is None else
                "the samples this runtime took and those the restart it resumed from "
                "carried (carried_from_restart), first to last sample model time"
            ),
        }


# ---------------------------------------------------------------------------
# Offline: replay the sidecar, recompute from published frames.
# ---------------------------------------------------------------------------
def read_time_series(path: Path | str) -> tuple[dict[str, Any], list[list[Any]]]:
    """The header and rows of one ``tornado-criterion.jsonl``; a torn last line is dropped."""

    lines = Path(path).read_bytes().split(b"\n")
    if not lines or not lines[0]:
        raise ValueError(f"{path} has no header")
    header = json.loads(lines[0])
    if header.get("schema") != NEST_TORNADO_TIME_SERIES_SCHEMA:
        raise ValueError(f"{path} is not a {NEST_TORNADO_TIME_SERIES_SCHEMA} time series")
    if list(header.get("columns", ())) != list(TIME_SERIES_COLUMNS):
        raise ValueError(f"{path} names columns {header.get('columns')}, not {list(TIME_SERIES_COLUMNS)}")
    rows: list[list[Any]] = []
    body = lines[1:]
    for n, line in enumerate(body):
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if n == len(body) - 1:
                break  # a run killed mid-write
            raise
    return header, rows


def replay_time_series(path: Path | str) -> dict[str, Any]:
    """Re-decide every condition, state, episode and the verdict from the sidecar rows.

    The rows carry every value a decision reads, at full binary64 precision, so
    the replay is exact.  Works on the sidecar of a run that died, where no
    receipt exists.
    """

    header, rows = read_time_series(path)
    if header.get("criterion_sha256") != CRITERION_SHA256:
        raise ValueError(
            f"{path} was decided by criterion {header.get('criterion_sha256')}; this module's "
            f"criterion is {CRITERION_SHA256}, so a replay would apply another rule"
        )
    stop = header.get("stop_rules") or {}
    from types import SimpleNamespace

    ledger = CriterionLedger(SimpleNamespace(
        stop_after_episode_s=stop.get("stop_after_episode_s"),
        stop_if_no_tornado_by_s=stop.get("stop_if_no_tornado_by_s"),
    ))
    carried = header.get("carried_from_restart")
    if carried is not None:
        # A resumed run's rows continue the restart's ledger: seed the same one.
        ledger.restore({name: (str(row[0]), str(row[1])) for name, row in carried.items()})
    dx = float(header["plan"]["dx_m"])
    column = {name: n for n, name in enumerate(TIME_SERIES_COLUMNS)}
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        found = bool(row[column["found"]])
        corners = int(row[column["core_corners"]])
        sample = CriterionSample(
            step=int(row[column["step"]]), model_time_s=float(row[column["model_time_s"]]),
            parent_i0=_int_or_none(row[column["parent_i0"]]),
            parent_j0=_int_or_none(row[column["parent_j0"]]),
            found=found,
            centre_i=int(row[column["centre_i"]]) if found else -1,
            centre_j=int(row[column["centre_j"]]) if found else -1,
            ground_x_m=float(row[column["centre_ground_x_m"]]) if found else math.nan,
            ground_y_m=float(row[column["centre_ground_y_m"]]) if found else math.nan,
            zeta_max_s_inv=float(row[column["zeta_max_s_inv"]]) if found else math.nan,
            ground_relative_wind_m_s=_nan_if_none(row[column["ground_relative_wind_max_m_s"]]),
            frame_relative_wind_m_s=_nan_if_none(row[column["frame_relative_wind_max_m_s"]]),
            core_corners=corners, core_radius_m=core_radius_m(dx, corners),
        )
        again = time_series_row(sample, ledger.observe(sample), ledger.episode_count)
        if again != row and len(mismatches) < 16:
            mismatches.append({"step": sample.step, "row": row, "replayed": again})
    return {
        "schema": NEST_TORNADO_CRITERION_SCHEMA,
        "kind": "replay-of-time-series",
        "path": os.fspath(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "criterion_sha256": CRITERION_SHA256,
        **ledger.record(),
        "carried_from_restart": None if carried is None else ledger_carried_summary(carried),
        "rows_identical": not mismatches,
        "mismatches": mismatches,
    }


def ledger_carried_summary(carried: Mapping[str, Any]) -> dict[str, Any]:
    """What a sidecar header's ``carried_from_restart`` rows say, decoded."""

    ledger = CriterionLedger()
    return ledger.restore({name: (str(row[0]), str(row[1])) for name, row in carried.items()})


def _nan_if_none(value: Any) -> float:
    return math.nan if value is None else float(value)


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def recompute_from_frames(child_root: Path | str) -> dict[str, Any]:
    """Recompute each published child frame's sample with :func:`reference_sample`.

    A child frame's placement document names the CHILD's step, which is the
    parent's ``completed_steps`` times ``time_refinement_ratio``
    (``NestDriver`` holds that invariant every parent step); the sidecar rows
    are keyed by the parent's step.  So a frame is matched to the row at
    ``placement step / time_refinement_ratio`` (the ratio from the sidecar
    header), and that row's model time must equal the placement's, which the
    child clock takes from the parent's at every parent-step boundary.

    A child frame is published after its step's feedback and any nest move;
    feedback writes only the parent, so a frame whose placement
    (``parent_i0``/``parent_j0``, integers, compared exactly) is the one the
    row was sampled on holds exactly the sampled winds and must reproduce the
    row's words.  A frame taken after a move is reported and not compared; a
    frame with no row of its step is counted as such.
    """

    from .nest_parcel_backward import read_child_frame
    from .nest_parcels import NEST_CHILD_PLACEMENT_DIRECTORY

    root = Path(child_root)
    header, rows = read_time_series(root / CRITERION_JSONL_FILENAME)
    plan = SamplePlan.from_record(header["plan"])
    ratio = int(header["time_refinement_ratio"])
    if ratio < 1:
        raise ValueError(f"the sidecar header names time_refinement_ratio {ratio}")
    column = {name: n for n, name in enumerate(TIME_SERIES_COLUMNS)}
    by_step = {int(row[column["step"]]): row for row in rows}
    compared = identical = moved = without_row = 0
    frames: list[dict[str, Any]] = []
    for placement_path in sorted((root / NEST_CHILD_PLACEMENT_DIRECTORY).glob("frame-*.json")):
        index = int(placement_path.stem.split("-")[1])
        frame = read_child_frame(root, index)
        p = frame.placement
        child_step = int(p["step"])
        parent_step, remainder = divmod(child_step, ratio)
        entry: dict[str, Any] = {"index": index, "child_step": child_step,
                                 "parent_step": parent_step if remainder == 0 else None}
        row = by_step.get(parent_step) if remainder == 0 else None
        if row is not None and float(row[column["model_time_s"]]) != float(p["model_time_s"]):
            # The step matched but the clock did not: not the sampled state.
            entry["model_time_mismatch"] = [row[column["model_time_s"]], p["model_time_s"]]
            row = None
        entry["jsonl_row"] = row is not None
        if row is None:
            without_row += 1
            frames.append(entry)
            continue
        placement_sampled = (row[column["parent_i0"]], row[column["parent_j0"]])
        placement_frame = (int(p["parent_i0"]), int(p["parent_j0"]))
        same_placement = placement_sampled == placement_frame
        entry["same_placement_as_sample"] = same_placement
        if not same_placement:
            entry["placement_sampled"], entry["placement_published"] = (
                list(placement_sampled), list(placement_frame))
            moved += 1
            frames.append(entry)
            continue
        words = reference_sample(frame.u, frame.v, plan=plan)
        sample = sample_from_words(
            words, step=parent_step, model_time_s=float(p["model_time_s"]), dx_m=float(p["dx_m"]),
            dy_m=float(p["dy_m"]), origin_x_m=float(p["origin_x_m"]),
            origin_y_m=float(p["origin_y_m"]), domainlocx_m=float(p["domainlocx_m"]),
            domainlocy_m=float(p["domainlocy_m"]),
        )
        recomputed = [
            sample.found, sample.centre_i if sample.found else None,
            sample.centre_j if sample.found else None,
            _finite(sample.zeta_max_s_inv) if sample.found else None,
            _finite(sample.ground_relative_wind_m_s), _finite(sample.frame_relative_wind_m_s),
            sample.core_corners,
        ]
        recorded = [row[column[name]] for name in (
            "found", "centre_i", "centre_j", "zeta_max_s_inv", "ground_relative_wind_max_m_s",
            "frame_relative_wind_max_m_s", "core_corners")]
        compared += 1
        entry["identical_to_jsonl"] = recomputed == recorded
        identical += int(entry["identical_to_jsonl"])
        if not entry["identical_to_jsonl"]:
            entry["recomputed"], entry["recorded"] = recomputed, recorded
        frames.append(entry)
    return {
        "schema": NEST_TORNADO_CRITERION_SCHEMA,
        "kind": "offline-recompute-from-frames",
        "child_root": os.fspath(root),
        "criterion_sha256": header.get("criterion_sha256"),
        "time_refinement_ratio": ratio,
        "frames_read": len(frames),
        "frames_compared": compared,
        "frames_identical_to_jsonl": identical,
        "frames_after_a_move": moved,
        "frames_without_a_row": without_row,
        "frames": frames,
    }


__all__ = [
    "CM1_AUTHORITY",
    "CRITERION",
    "CRITERION_JSONL_FILENAME",
    "CRITERION_SHA256",
    "CriterionLedger",
    "CriterionSample",
    "FIXED_THRESHOLD_NAMES",
    "NEST_TORNADO_CRITERION_SCHEMA",
    "NEST_TORNADO_KERNEL_SHA256",
    "NEST_TORNADO_KERNEL_SOURCE",
    "NestTornadoCriterion",
    "STOP_HONOURED",
    "STOP_UNUSED_AT_COURANT",
    "STOP_UNUSED_AT_DURATION",
    "STOP_WITHDRAWN_BY_RUN",
    "SamplePlan",
    "SampleWall",
    "TIME_SERIES_COLUMNS",
    "TimeSeriesWriter",
    "WORDING",
    "child_sample_plan",
    "compile_nest_tornado_criterion_module",
    "core_radius_m",
    "criterion_stop_record",
    "height_bracket",
    "interior_corner_box",
    "launch_tornado_sample",
    "read_time_series",
    "recompute_from_frames",
    "reference_corner_vorticity",
    "reference_sample",
    "reference_winds",
    "replay_time_series",
    "sample_from_words",
    "time_series_row",
]
