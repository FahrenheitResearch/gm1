"""The storm over time: its updraft and its spin at every height, one column per saved time.

Two looks, each one picture of the whole run so far, drawn again as each saved time lands (:mod:`.render` draws
them with the looks drawn once per run, ``RUNDIR/quicklook/<look>/still.png``):

``updraft-time-height``
    The strongest rising air at each height in a circle around the storm's main updraft, and under it the speed
    the fastest tenth of that circle reaches (one fast cell and a wide strong updraft look different), with the
    cloud top as a line, and the strongest spin at the lowest level in the circle as a line under them.  Pulse
    tracks (maxima of the updraft climbing from one saved time to the next) are drawn only when the run saved
    every 60 s or closer; otherwise the picture says they are left off and why.
``spin-time-height``
    The strongest counter-clockwise spin (vertical vorticity) at each height in a wider circle around the same
    updraft (:data:`SPIN_RADIUS_M`), on a height axis that gives the lowest kilometres room, with the most 2-5 km
    updraft helicity in that circle as a line under it.
    Its colours follow the grid (:func:`spin_range_for`: the spin look's rule, from the greater of the grid spacing
    and the lowest level's height), on a logarithmic scale from a tenth of that range to five times it, because the
    spin near the ground and the spin of the updraft aloft differ tenfold.  The helicity line's top follows the
    grid the same way (:func:`uh_top_for`), raised to hold a run's peak that passes it (:func:`helicity_top`).

Following the storm (:func:`follow`): at each saved time the places where the air rises fastest on average between
2 and 6 km (smoothed over 3 km, at least 10 km apart) are the updraft candidates; the followed updraft is the
strongest candidate within reach of where the last one was heading, measured on the ground (a moving domain
changes nothing); the saved times with no storm yet (or any more: :data:`STORM_MIN`, stated on the picture) are
dimmed, and a saved time whose storm was at the domain's edge is marked (:func:`at_finder_edge`).  When the run has
no single storm to follow (at most of the saved times with a storm, three or more updrafts at least half as strong
as the followed one, as a line of storms or a cluster has; or a followed updraft that keeps vanishing), every
column is the whole domain's instead, and the picture says so in plain words.

A hurricane run (CM1's swirl start) follows the lowest pressure at the lowest level, takes a circle wide enough for
the eyewall and the inner rain bands, its own range for the rising air, the fastest hundredth of the circle in the
second panel, and the fastest wind at the lowest level under them.  The spin look is made for the rotating updraft
of a thunderstorm and is not drawn for a hurricane (:data:`THUNDERSTORM_ONLY`).

The numbers of every saved time are kept in ``.state/timeheight.npz`` (the columns of each candidate's circle and
of the whole domain), so a new saved time reads only its own frame; the following is worked out again over all the
saved times each time the pictures are drawn.  A damaged store is read again from the frames, and nothing here
stops the pass's other looks: an error is recorded with the look.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Collection, Sequence

import numpy as np

from .frame import FrameUnreadable, QuickFrame

UPDRAFT = "updraft-time-height"
SPIN = "spin-time-height"
#: (id, title): the looks this module draws, drawn once per run (:data:`.render.ONCE`).
ONCE: tuple[tuple[str, str], ...] = ((UPDRAFT, "The updraft over time"), (SPIN, "The spin over time"))
IDS = tuple(pid for pid, _ in ONCE)
TITLES = dict(ONCE)
#: The look made for the rotating updraft of a thunderstorm, which a hurricane run does not draw.
THUNDERSTORM_ONLY = (SPIN,)
THUNDERSTORM_ONLY_WHY = ("This look is made for the rotating updraft of a thunderstorm. For a hurricane, see the "
                         "updraft over time.")
#: Why the spin over time is not drawn for frames that carry no spin.
NO_SPIN_WHY = "This run's saved frames carry no spin (no zvort, and no u and v to work it out from)."
STATE_NAME = "timeheight.npz"
STATE_SCHEMA = "gm1.quicklook-timeheight/v1"
PICTURE = "still.png"

# ---- finding and following the storm
#: The layer whose mean rising air finds a thunderstorm's updraft (m above ground, w levels).
FIND_LAYER_M = (2000.0, 6000.0)
#: The width the finder smooths that layer mean over (m): a storm's updraft, not one fast cell.
FIND_SMOOTH_M = 3000.0
#: Candidates kept at each saved time, and how far apart (m) two candidates must be.
CANDIDATES = 4
CANDIDATE_APART_M = 10000.0
#: The circle around the followed updraft whose rising air is drawn (m).
RADIUS_M = 5000.0
#: The circle whose spin is drawn (m): wider, because the rotation near the ground sits on the updraft's flank,
#: 5 to 6 km from the middle of the 2-6 km updraft on a 100 m supercell (measured), where a 5 km circle misses it.
SPIN_RADIUS_M = 8000.0
#: A hurricane's circle (m): the eyewall and the inner rain bands of the 4 and 8 km hurricane presets.
HURRICANE_RADIUS_M = 150000.0
#: How far a followed updraft may be from where it was heading at the next saved time: this distance plus this
#: speed times the time between the two saves.
REACH_M = 5000.0
REACH_SPEED = 15.0
#: The weakest smoothed 2-6 km rising air (m/s) that counts as a storm (a quiet run's ripples and a starting
#: bubble's first minutes are below it).
STORM_MIN = 3.0
#: No single storm: at more than RIVAL_SHARE of the saved times with a storm, RIVALS or more other updrafts are at
#: least half as strong as the followed one; or the followed updraft is lost (no candidate within reach) at more
#: than LOST_SHARE of them.
RIVAL_SHARE = 0.5
RIVALS = 2
LOST_SHARE = 0.25
#: Cells at the domain's edges left out of the finder and of the whole-domain numbers (an open edge's own noise).
EDGE_CELLS = 3

# ---- what the columns hold and how they are drawn
#: The share of the circle whose speed the second panel shows (the fastest tenth; a hurricane's fastest hundredth,
#: its eyewall being a thin ring in a wide circle).
FASTEST_SHARE = 0.10
HURRICANE_FASTEST_SHARE = 0.01
#: Cloud: cloud water plus cloud ice at least this much (g/kg).
CLOUD_G_KG = 0.1
#: Pulse tracks need saved times this close (s) or closer.
PULSE_INTERVAL_S = 60.0
PULSE_FROM_M = 3000.0         # maxima above this height ...
PULSE_W_MIN = 25.0            # ... at least this fast (m/s) ...
PULSE_PROMINENCE = 2.0        # ... standing this far (m/s) above the dips on either side
PULSE_RISE_MAX_M = 2400.0     # one save to the next: 0 to this much higher (per 60 s)
PULSE_MIN_SAVES = 3
#: Fixed, stated ranges.
W_TOP = 100.0                 # m/s, a thunderstorm's rising air (a 1 km supercell reaches 60, a 100 m one 105)
HURRICANE_W_TOP = 40.0        # m/s, a hurricane's (its first bursts reach 38 m/s at 4 km, the grown eyewall 10-20)
HURRICANE_WIND_TOP = 80.0     # m/s ground-relative, the hurricane wind look's range
UH_TOP = 1000.0               # m2/s2, the updraft look's range, the helicity line's top at a 1 km grid
UH_TOP_RANGE = 0.01           # 1/s: the spin range at which the helicity line's top is UH_TOP (see uh_top_for)
SPIN_LOG_BELOW = 10.0         # the spin colours start at a tenth of the grid's range ...
SPIN_LOG_ABOVE = 5.0          # ... and end at five times it
HEIGHT_TOP_M = 18000.0        # the height axis ends here or at the model top, whichever is lower
MESO_LAYER_M = (1000.0, 6000.0)   # the spin look's readout: the strongest spin of the updraft's rotation aloft
HURRICANE_HEIGHT_TOP_M = 20000.0


class TimeHeightError(ValueError):
    """The run's frames lack what a time-height look needs; the message says what."""


# ----------------------------------------------------------------------------------------------- small helpers
def spin_range_for(dx_m: float, z0_m: float) -> float:
    """The fixed spin range for a grid: the spin look's rule (:func:`.products.spin_range`, 1/dx rounded to a
    1-2-5 step) applied to the greater of the grid spacing and the lowest level's height above the ground.  Near
    the ground a vortex is resolved no better than the coarser of the two: a 67 m grid on 500 m levels (lowest
    level 250 m) gets the range of 250 m, 0.05 1/s, where its grid spacing alone gives 0.2 and draws its 0.016
    peak almost black.  The 100 m preset (lowest level 125 m) and a 100 m grid with a 10 m lowest level both keep
    0.1, the 1 km preset 0.01."""

    from .products import spin_range

    return spin_range(max(float(dx_m), float(z0_m or 0.0), 1.0))


def spin_range_words(dx_m: float, z0_m: float) -> str:
    """What sets the spin range, in a picture's words: the grid spacing, or the lowest level when that is higher
    above the ground than the grid spacing and so gives another range than the spin look's at this dx."""

    from . import style as S
    from .products import spin_range

    if spin_range_for(dx_m, z0_m) == spin_range(float(dx_m)):
        return "the spin look's range at this grid"
    return f"the range for a {S.fmt_height(float(z0_m))} lowest level"


def uh_top_for(spin_range_per_s: float) -> float:
    """The updraft helicity line's fixed top for a grid: :data:`UH_TOP` at a 1 km grid's spin range and coarser,
    scaled as the spin range is on finer grids (2000 at 500 m, 5000 at 250 m, 10000 at 100 m), since both grow as a
    finer grid resolves smaller and stronger eddies."""

    return max(UH_TOP, float(f"{UH_TOP * spin_range_per_s / UH_TOP_RANGE:.6g}"))


def finder_margin(nx: int, ny: int) -> int:
    """The cells at each edge of the domain the storm finder leaves out (:data:`EDGE_CELLS`, fewer on a tiny
    domain)."""

    return min(EDGE_CELLS, max(0, min(nx, ny) // 4))


def at_finder_edge(x: float, y: float, grid: dict[str, Any]) -> bool:
    """Whether a centre (x, y in the domain's metres) sits as near the domain's edge as the storm finder looks: an
    updraft there may reach on past it, into cells the finder leaves out or out of the domain."""

    nx, ny = int(grid["nx"]), int(grid["ny"])
    margin = finder_margin(nx, ny)
    i, j = int(x // float(grid["dx"])), int(y // float(grid["dy"]))
    return i <= margin or j <= margin or i >= nx - 1 - margin or j >= ny - 1 - margin


def box_mean(a: np.ndarray, n: int) -> np.ndarray:
    """The mean over an n x n box (n made odd) at every cell of a 2-D array; near the edges, over the cells inside."""

    n = max(1, int(n) | 1)
    if n == 1:
        return np.asarray(a, np.float32)
    h = n // 2
    a64 = np.asarray(a, np.float64)
    ny, nx = a64.shape
    pad = np.zeros((ny + 2 * h + 1, nx + 2 * h + 1))
    pad[h + 1:h + 1 + ny, h + 1:h + 1 + nx] = a64
    ones = np.zeros_like(pad)
    ones[h + 1:h + 1 + ny, h + 1:h + 1 + nx] = 1.0
    c = pad.cumsum(0).cumsum(1)
    o = ones.cumsum(0).cumsum(1)

    def window(s: np.ndarray) -> np.ndarray:
        return s[n:, n:] - s[:-n, n:] - s[n:, :-n] + s[:-n, :-n]

    return (window(c) / np.maximum(window(o), 1.0)).astype(np.float32)


def strongest_places(field_: np.ndarray, count: int, apart_cells: float, margin: int = 0) -> list[tuple[int, int]]:
    """Up to ``count`` (j, i) of the largest values of a 2-D array, each at least ``apart_cells`` from the others,
    the outer ``margin`` cells left out."""

    f = np.array(field_, np.float64)
    if margin > 0 and min(f.shape) > 2 * margin + 1:
        f[:margin, :] = -np.inf
        f[-margin:, :] = -np.inf
        f[:, :margin] = -np.inf
        f[:, -margin:] = -np.inf
    ny, nx = f.shape
    yy = np.arange(ny)[:, None]
    xx = np.arange(nx)[None, :]
    out = []
    for _ in range(count):
        flat = int(np.argmax(f))
        j, i = divmod(flat, nx)
        if not np.isfinite(f[j, i]):
            break
        out.append((j, i))
        f[(yy - j) ** 2 + (xx - i) ** 2 <= apart_cells ** 2] = -np.inf
    return out


def kth_largest(values: np.ndarray, k: int) -> float:
    """The k-th largest of ``values`` (k counted from 1); NaN for an empty array."""

    v = np.asarray(values, np.float32).ravel()
    if v.size == 0:
        return float("nan")
    k = max(1, min(int(k), v.size))
    return float(np.partition(v, v.size - k)[v.size - k])


def circle_cells(radius_m: float, dx: float, dy: float) -> int:
    """How many cells a circle of ``radius_m`` covers."""

    r = int(math.ceil(radius_m / min(dx, dy))) + 1
    jj, ii = np.mgrid[-r:r + 1, -r:r + 1]
    return int(((ii * dx) ** 2 + (jj * dy) ** 2 <= radius_m ** 2).sum())


def zeta_box(qf: QuickFrame, k: int, j0: int, j1: int, i0: int, i1: int) -> np.ndarray:
    """Vertical vorticity (1/s) at the scalar points [j0:j1, i0:i1] of level k: CM1's zvort when the frame carries
    it, otherwise dv/dx - du/dy at the cell corners averaged to the centres, exactly as
    :meth:`.QuickFrame.zeta_level` does for the whole level."""

    if qf.has("zvort"):
        return np.asarray(qf.view("zvort")[k, j0:j1, i0:i1], np.float32)
    ny, nx = qf.ny, qf.nx
    ja, jb = max(j0 - 1, 0), min(j1 + 1, ny)
    ia, ib = max(i0 - 1, 0), min(i1 + 1, nx)
    u = np.asarray(qf.view("u3d")[k, ja:jb, ia:ib + 1], np.float32)        # (rows, cols+1): u faces
    v = np.asarray(qf.view("v3d")[k, ja:jb + 1, ia:ib], np.float32)        # (rows+1, cols): v faces
    dvdx = (v[:, 1:] - v[:, :-1]) / np.float32(qf.dx)
    dudy = (u[1:, :] - u[:-1, :]) / np.float32(qf.dy)
    zc = dvdx[1:-1, :] - dudy[:, 1:-1]                                      # the interior corners
    z = np.pad(zc, 1, mode="edge")
    cen = 0.25 * (z[1:, 1:] + z[1:, :-1] + z[:-1, 1:] + z[:-1, :-1])      # the cell centres ja:jb, ia:ib
    return cen[j0 - ja:j1 - ja, i0 - ia:i1 - ia].astype(np.float32)


def _cloud_rows(qf: QuickFrame) -> list[int]:
    """The rows of the species carrier that are cloud water and cloud ice (those the run has)."""

    try:
        names = qf.species_names()
    except (ValueError, KeyError):
        return []
    return [names.index(n) for n in ("qc", "qi") if n in names]


# ----------------------------------------------------------------------------------------------- one frame
def find_centres(qf: QuickFrame, kind: str) -> tuple[list[tuple[float, float, float]], str]:
    """The storm's candidate centres in one frame, strongest first: (x, y in the domain's metres, strength), and
    in words how they were found.  A thunderstorm's strength is its smoothed 2-6 km rising air (m/s); a
    hurricane's, how far its lowest pressure is below the pressure at the area's edge (hPa)."""

    nx, ny, dx, dy = qf.nx, qf.ny, qf.dx, qf.dy
    margin = finder_margin(nx, ny)
    if kind == "hurricane":
        p = qf.pressure_level(0)
        if p is None:
            raise TimeHeightError("This run's saved frames carry no pressure to find the hurricane's centre by.")
        smooth = box_mean(p, 3)
        j, i = strongest_places(-smooth, 1, 1.0, margin)[0]
        edge = float(np.median(np.concatenate([p[0], p[-1], p[:, 0], p[:, -1]])))
        return [((i + 0.5) * dx, (j + 0.5) * dy, (edge - float(smooth[j, i])) / 100.0)], \
            "the lowest pressure at the lowest level"
    if not qf.has("w3d"):
        raise TimeHeightError("This run's saved frames carry no vertical wind.")
    w = qf.view("w3d")
    zf = qf.zf
    ks = [k for k in range(len(zf)) if FIND_LAYER_M[0] <= zf[k] <= FIND_LAYER_M[1]]
    if not ks:
        ks = [int(np.argmin(np.abs(zf - 0.5 * sum(FIND_LAYER_M))))]
    acc = np.zeros((ny, nx), np.float64)
    for k in ks:
        acc += np.maximum(np.asarray(w[k], np.float32), 0.0)
    acc /= len(ks)
    smooth = box_mean(acc, int(round(FIND_SMOOTH_M / dx)))
    places = strongest_places(smooth, CANDIDATES, CANDIDATE_APART_M / dx, margin)
    return [((i + 0.5) * dx, (j + 0.5) * dy, float(smooth[j, i])) for j, i in places], \
        (f"the strongest rising air between {FIND_LAYER_M[0] / 1000:g} and {FIND_LAYER_M[1] / 1000:g} km, "
         f"smoothed over {FIND_SMOOTH_M / 1000:g} km")


@dataclass
class Columns:
    """One circle's (or the whole domain's) numbers at one saved time."""

    w_max: np.ndarray          # (nz+1,) m/s on the w levels
    w_fast: np.ndarray         # (nz+1,) the speed the fastest share of the circle reaches
    w_min: np.ndarray          # (nz+1,)
    zeta_max: np.ndarray       # (nz,) 1/s on the scalar levels
    cloud_top: float           # m, NaN without cloud
    uh_max: float              # m2/s2, 2-5 km updraft helicity (NaN without it)
    wind0_max: float           # m/s ground-relative, at the lowest level
    at_edge: bool              # the circle reaches past the domain's edge
    spin_at_edge: bool = False  # the spin's (wider) circle does


def _cloud_top(qf: QuickFrame, rows: list[int], sl: tuple[slice, slice], mask: np.ndarray | None) -> float:
    if not rows or not qf.has("q3d"):
        return float("nan")
    q = qf.view("q3d")
    zh = qf.zh
    for k in range(qf.nz - 1, -1, -1):
        c = sum(np.asarray(q[r, k][sl], np.float32) for r in rows) * np.float32(1000.0)
        if mask is not None:
            c = np.where(mask, c, np.float32(0.0))
        if float(np.max(c)) >= CLOUD_G_KG:
            return float(zh[k])
    return float("nan")


def _w_columns(qf: QuickFrame, sl: tuple[slice, slice], mask: np.ndarray | None, k_fast: int):
    w = qf.view("w3d")
    n = w.shape[0]
    hi, fast, lo = (np.full(n, np.nan, np.float32) for _ in range(3))
    for k in range(n):
        v = np.asarray(w[k][sl], np.float32)
        v = v[mask] if mask is not None else v.ravel()
        hi[k], lo[k] = float(v.max()), float(v.min())
        fast[k] = kth_largest(v, k_fast)
    return hi, fast, lo


def circle_columns(qf: QuickFrame, x: float, y: float, radius: float, k_fast: int, uh: np.ndarray | None,
                   rows: list[int], spin_radius: float | None = None) -> Columns:
    """The numbers of the circle of ``radius`` around (x, y) (the domain's metres) at every height: the rising air
    and the cloud top in it, and the spin, the updraft helicity and the wind at the lowest level in the circle of
    ``spin_radius`` (the same circle when not given)."""

    nx, ny, dx, dy = qf.nx, qf.ny, qf.dx, qf.dy
    spin_radius = radius if spin_radius is None else spin_radius
    reach = max(radius, spin_radius)
    i0 = max(int(math.floor((x - reach) / dx)), 0)
    i1 = min(int(math.ceil((x + reach) / dx)) + 1, nx)
    j0 = max(int(math.floor((y - reach) / dy)), 0)
    j1 = min(int(math.ceil((y + reach) / dy)) + 1, ny)
    xs = (np.arange(i0, i1) + 0.5) * dx - x
    ys = (np.arange(j0, j1) + 0.5) * dy - y
    r2 = ys[:, None] ** 2 + xs[None, :] ** 2
    inside = r2 <= radius ** 2
    if not inside.any():
        inside = r2 <= r2.min()
    spin_in = r2 <= spin_radius ** 2
    if not spin_in.any():
        spin_in = r2 <= r2.min()
    sl = (slice(j0, j1), slice(i0, i1))
    w_max, w_fast, w_min = _w_columns(qf, sl, inside, k_fast)
    zeta_max = np.full(qf.nz, np.nan, np.float32)
    if qf.has("zvort") or (qf.has("u3d") and qf.has("v3d")):
        for k in range(qf.nz):
            zeta_max[k] = float(zeta_box(qf, k, j0, j1, i0, i1)[spin_in].max())
    uh_max = float(np.max(uh[sl][spin_in])) if uh is not None else float("nan")
    wind0 = float("nan")
    if qf.has("u3d") and qf.has("v3d"):
        u = np.asarray(qf.view("u3d")[0, j0:j1, i0:i1 + 1], np.float32)
        v = np.asarray(qf.view("v3d")[0, j0:j1 + 1, i0:i1], np.float32)
        uc = 0.5 * (u[:, 1:] + u[:, :-1]) + np.float32(qf.umove)
        vc = 0.5 * (v[1:, :] + v[:-1, :]) + np.float32(qf.vmove)
        wind0 = float(np.hypot(uc, vc)[spin_in].max())

    def cut(r: float) -> bool:
        return x - r < 0.0 or y - r < 0.0 or x + r > nx * dx or y + r > ny * dy

    return Columns(w_max, w_fast, w_min, zeta_max, _cloud_top(qf, rows, sl, inside), uh_max, wind0, cut(radius),
                   cut(spin_radius))


def domain_columns(qf: QuickFrame, k_fast: int, uh: np.ndarray | None, rows: list[int]) -> Columns:
    """The whole domain's numbers at every height (its outer :data:`EDGE_CELLS` cells left out)."""

    nx, ny = qf.nx, qf.ny
    e = EDGE_CELLS if min(nx, ny) > 4 * EDGE_CELLS else 0
    sl = (slice(e, ny - e), slice(e, nx - e))
    w_max, w_fast, w_min = _w_columns(qf, sl, None, k_fast)
    zeta_max = np.full(qf.nz, np.nan, np.float32)
    if qf.has("zvort") or (qf.has("u3d") and qf.has("v3d")):
        for k in range(qf.nz):
            zeta_max[k] = float(zeta_box(qf, k, 0, ny, 0, nx)[sl].max())
    uh_max = float(np.max(uh[sl])) if uh is not None else float("nan")
    wind0 = float("nan")
    if qf.has("u3d") and qf.has("v3d"):
        u, v = qf.uv_level(0, ground=True)
        wind0 = float(np.hypot(u, v)[sl].max())
    return Columns(w_max, w_fast, w_min, zeta_max, _cloud_top(qf, rows, sl, None), uh_max, wind0, False, False)


def measure(qf: QuickFrame, kind: str = "") -> dict[str, Any]:
    """Everything the two looks keep of one saved time: the candidates, the columns of each one's circle and of
    the whole domain, and where the frame was on the ground."""

    hurricane = kind == "hurricane"
    radius = HURRICANE_RADIUS_M if hurricane else RADIUS_M
    spin_radius = HURRICANE_RADIUS_M if hurricane else SPIN_RADIUS_M
    share = HURRICANE_FASTEST_SHARE if hurricane else FASTEST_SHARE
    centres, how = find_centres(qf, kind)
    cells = circle_cells(radius, qf.dx, qf.dy)
    k_fast = max(1, int(round(share * cells)))
    uh, uh_source = None, ""
    if not hurricane:
        try:
            uh_arr, uh_source = qf.uh()
            uh = np.asarray(uh_arr, np.float32)
        except (ValueError, KeyError):
            uh, uh_source = None, ""
    rows = _cloud_rows(qf)
    circles = [circle_columns(qf, x, y, radius, k_fast, uh, rows, spin_radius) for x, y, _s in centres]
    return {"t": qf.t, "key": qf.key, "centres": centres, "circles": circles,
            "domain": domain_columns(qf, k_fast, uh, rows), "domainloc": (qf.domainlocx, qf.domainlocy),
            "move": (qf.umove, qf.vmove), "found_by": how, "radius_m": radius, "spin_radius_m": spin_radius,
            "k_fast": k_fast, "share": share, "cells": cells,
            "zeta_source": "CM1 zvort" if qf.has("zvort") else "dv/dx − du/dy", "uh_source": uh_source,
            "grid": {"nx": qf.nx, "ny": qf.ny, "nz": qf.nz, "dx": qf.dx, "dy": qf.dy,
                     "zf": [float(z) for z in qf.zf], "zh": [float(z) for z in qf.zh]}}


# ----------------------------------------------------------------------------------------------- the run's state
_PER_CIRCLE = ("w_max", "w_fast", "w_min", "zeta_max")
_PER_CIRCLE_SCALAR = ("cloud_top", "uh_max", "wind0_max", "at_edge", "spin_at_edge")


class Builder:
    """Every saved time's numbers, in time order, kept in ``state_dir`` (``timeheight.npz``)."""

    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir)
        self.reset()
        self._load()

    @property
    def path(self) -> Path:
        return self.state_dir / STATE_NAME

    def reset(self, kind: str = "") -> None:
        self.kind = kind
        self.keys: list[str] = []
        self.rows: list[dict[str, Any]] = []      # per saved time: its numbers, or {"skipped": why}
        self.meta: dict[str, Any] = {}

    # -- disk -----------------------------------------------------------------------
    def _load(self) -> None:
        """The kept numbers, or nothing (every saved time read again) when the file is missing, from another
        schema, or damaged in any way (a truncated copy, a disk that filled while it was written): it is only a
        store of numbers the frames still hold."""

        if not self.path.is_file():
            return
        try:
            self._parse()
        except Exception:      # noqa: BLE001 - any damage: start again from the frames
            self.reset()

    def _parse(self) -> None:
        with np.load(self.path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            arrays = {name: np.array(data[name]) for name in data.files if name != "meta"}
        if not isinstance(meta, dict) or meta.get("schema") != STATE_SCHEMA:
            return
        self.kind = str(meta.get("kind") or "")
        self.keys = list(meta["keys"])
        self.meta = dict(meta.get("run") or {})
        rows = []
        n_read = 0
        for info in meta["frames"]:
            if "skipped" in info:
                rows.append(dict(info))
                continue
            f = n_read
            n_read += 1
            row = dict(info)
            row["centres"] = [tuple(map(float, c)) for c in arrays["centres"][f] if np.isfinite(c[2])]
            nc = len(row["centres"])
            row["circles"] = [Columns(*(arrays[f"c_{name}"][f, c] for name in _PER_CIRCLE),
                                      *(float(arrays[f"c_{name}"][f, c]) for name in _PER_CIRCLE_SCALAR[:3]),
                                      bool(arrays["c_at_edge"][f, c]), bool(arrays["c_spin_at_edge"][f, c]))
                             for c in range(nc)]
            row["domain"] = Columns(*(arrays[f"d_{name}"][f] for name in _PER_CIRCLE),
                                    *(float(arrays[f"d_{name}"][f]) for name in _PER_CIRCLE_SCALAR[:3]), False, False)
            rows.append(row)
        if len(rows) != len(self.keys) or [r["key"] for r in rows] != self.keys:
            raise ValueError("the kept numbers do not match their saved times")
        self.rows = rows

    def save(self) -> None:
        read = [r for r in self.rows if "skipped" not in r]
        frames = []
        for r in self.rows:
            if "skipped" in r:
                frames.append({k: r[k] for k in ("t", "key", "skipped")})
            else:
                frames.append({k: r[k] for k in ("t", "key", "domainloc", "move")})
        meta = {"schema": STATE_SCHEMA, "kind": self.kind, "keys": self.keys, "frames": frames, "run": self.meta}
        arrays: dict[str, np.ndarray] = {"meta": np.array(json.dumps(meta))}
        if read:
            nzw = len(self.meta["grid"]["zf"])
            nz = len(self.meta["grid"]["zh"])
            nf = len(read)
            centres = np.full((nf, CANDIDATES, 3), np.nan, np.float64)
            per = {name: np.full((nf, CANDIDATES, nzw if name.startswith("w") else nz), np.nan, np.float32)
                   for name in _PER_CIRCLE}
            scal = {name: np.full((nf, CANDIDATES), np.nan, np.float32) for name in _PER_CIRCLE_SCALAR}
            dom = {name: np.full((nf, nzw if name.startswith("w") else nz), np.nan, np.float32)
                   for name in _PER_CIRCLE}
            dscal = {name: np.full(nf, np.nan, np.float32) for name in _PER_CIRCLE_SCALAR[:3]}
            for f, r in enumerate(read):
                for c, (cen, col) in enumerate(zip(r["centres"], r["circles"])):
                    centres[f, c] = cen
                    for name in _PER_CIRCLE:
                        per[name][f, c] = getattr(col, name)
                    for name in _PER_CIRCLE_SCALAR:
                        scal[name][f, c] = float(getattr(col, name))
                for name in _PER_CIRCLE:
                    dom[name][f] = getattr(r["domain"], name)
                for name in _PER_CIRCLE_SCALAR[:3]:
                    dscal[name][f] = float(getattr(r["domain"], name))
            arrays["centres"] = centres
            arrays.update({f"c_{k}": v for k, v in {**per, **scal}.items()})
            arrays.update({f"d_{k}": v for k, v in {**dom, **dscal}.items()})
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_dir / ".timeheight.partial.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, self.path)

    # -- frames ---------------------------------------------------------------------
    def add(self, qf: QuickFrame) -> None:
        m = measure(qf, self.kind)
        grid = m.pop("grid")
        if not self.meta:
            self.meta = {"grid": grid, **{k: m[k] for k in ("found_by", "radius_m", "spin_radius_m", "k_fast",
                                                             "share", "cells",
                                                             "zeta_source", "uh_source")}}
        elif self.meta["grid"]["zf"] != grid["zf"] or self.meta["grid"]["nx"] != grid["nx"] \
                or self.meta["grid"]["ny"] != grid["ny"]:
            raise TimeHeightError("The run's grid changed between saved times; the columns cannot share a picture.")
        self.keys.append(qf.key)
        self.rows.append({k: m[k] for k in ("t", "key", "centres", "circles", "domain")}
                         | {"domainloc": list(m["domainloc"]), "move": list(m["move"])})

    def skip(self, t: float, key: str, why: str) -> None:
        self.keys.append(key)
        self.rows.append({"t": float(t), "key": key, "skipped": why})

    def catch_up(self, lineage: Sequence[tuple[float, str, Path, int]],
                 open_frame: Callable[[Path, str, int], QuickFrame], kind: str) -> int:
        """Bring the numbers up to ``lineage`` (the run's frames, oldest first); returns how many frames it read."""

        keys = [key for _t, key, _p, _s in lineage]
        if kind != self.kind or self.keys != keys[:len(self.keys)] or len(self.keys) > len(keys):
            self.reset(kind)
        read = 0
        from ..cm1out.frames import FrameError

        for t, key, path, segment in lineage[len(self.keys):]:
            try:
                qf = open_frame(path, key, segment)
            except (FrameError, FrameUnreadable, OSError, ValueError) as error:
                self.skip(t, key, f"{type(error).__name__}: {error}")
                continue
            try:
                self.add(qf)
                read += 1
            except (FrameUnreadable, KeyError) as error:
                self.keys.append(key)
                self.rows.append({"t": float(t), "key": key, "skipped": f"{type(error).__name__}: {error}"})
            finally:
                qf.close()
        return read

    @property
    def read_rows(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if "skipped" not in r]


# ----------------------------------------------------------------------------------------------- following
@dataclass
class Followed:
    """Which circle each saved time's columns come from, and why."""

    mode: str                            # "storm", "domain" or "hurricane"
    chosen: list[int | None]             # per read saved time: the candidate followed (None: the whole domain)
    reason: str = ""                     # plain words, for "domain"
    storm_times: int = 0                 # saved times with a storm
    crowded: int = 0                     # ... of which with RIVALS or more rivals at least half as strong
    lost: list[bool] = field(default_factory=list)       # per read saved time: no candidate within reach
    path: list[tuple[float, float] | None] = field(default_factory=list)   # ground position of the circle
    storm: list[bool] = field(default_factory=list)      # per read saved time: a storm was there to follow


def follow(rows: Sequence[dict[str, Any]], kind: str = "") -> Followed:
    """The storm through the run (see the module's words): a hurricane's centre, one thunderstorm followed on the
    ground, or the whole domain when there is no single storm."""

    n = len(rows)
    if kind == "hurricane":
        path = [_ground(r, 0) if r["centres"] else None for r in rows]
        return Followed("hurricane", [0 if r["centres"] else None for r in rows], path=path, lost=[False] * n,
                        storm_times=n, storm=[True] * n)
    chosen: list[int | None] = []
    lost: list[bool] = []
    path: list[tuple[float, float] | None] = []
    storm: list[bool] = []
    crowded = storm_times = 0
    prev = None            # (ground x, ground y, t, whether a storm was there)
    velocity = None
    for r in rows:
        cands = r["centres"]
        if not cands:
            chosen.append(None)
            lost.append(False)
            path.append(None)
            storm.append(False)
            continue
        strengths = [s for _x, _y, s in cands]
        ground = [_ground(r, c) for c in range(len(cands))]
        pick = 0
        missed = False
        if prev is not None and prev[3] and strengths[0] >= STORM_MIN:
            dt = max(r["t"] - prev[2], 1.0)
            vx, vy = velocity if velocity is not None else tuple(r.get("move") or (0.0, 0.0))
            px, py = prev[0] + vx * dt, prev[1] + vy * dt
            reach = REACH_M + REACH_SPEED * dt
            near = [c for c in range(len(cands)) if strengths[c] >= STORM_MIN
                    and math.hypot(ground[c][0] - px, ground[c][1] - py) <= reach]
            if near:
                pick = max(near, key=lambda c: strengths[c])
            else:
                missed = True
        s = strengths[pick]
        if s >= STORM_MIN:
            storm_times += 1
            rivals = sum(1 for c, sc in enumerate(strengths) if c != pick and sc >= max(0.5 * s, STORM_MIN))
            crowded += rivals >= RIVALS
            if prev is not None and not missed and prev[3]:
                dt = max(r["t"] - prev[2], 1.0)
                velocity = ((ground[pick][0] - prev[0]) / dt, (ground[pick][1] - prev[1]) / dt)
            elif missed:
                velocity = None
            prev = (ground[pick][0], ground[pick][1], r["t"], True)
        elif prev is None or not prev[3]:
            prev = (ground[pick][0], ground[pick][1], r["t"], False)
        chosen.append(pick)
        lost.append(missed)
        path.append(ground[pick] if s >= STORM_MIN else None)
        storm.append(s >= STORM_MIN)
    lost_n = sum(lost)
    if storm_times and crowded > RIVAL_SHARE * storm_times:
        at = "at the one saved time with a storm" if storm_times == 1 else \
            f"at {crowded} of the {storm_times} saved times with a storm"
        return Followed("domain", [None] * n, (
            f"No single storm to follow: {at}, "
            f"{RIVALS + 1} or more updrafts were at least half as strong as the followed one, as in a line or a "
            "cluster of storms. Each column is the whole domain's."), storm_times, crowded, [False] * n, [None] * n,
            storm)
    if storm_times and lost_n > LOST_SHARE * storm_times:
        return Followed("domain", [None] * n, (
            f"No single storm to follow: the followed updraft vanished at {lost_n} of the {storm_times} saved "
            "times with a storm (new updrafts kept taking over). Each column is the whole domain's."),
            storm_times, crowded, [False] * n, [None] * n, storm)
    return Followed("storm", chosen, "", storm_times, crowded, lost, path, storm)


def _ground(row: dict[str, Any], c: int) -> tuple[float, float]:
    x, y, _s = row["centres"][c]
    lx, ly = row.get("domainloc") or (0.0, 0.0)
    return float(x) + float(lx), float(y) + float(ly)


def columns_of(rows: Sequence[dict[str, Any]], followed: Followed) -> list[Columns]:
    return [r["domain"] if c is None else r["circles"][c] for r, c in zip(rows, followed.chosen)]


# ----------------------------------------------------------------------------------------------- pulse tracks
def pulse_peaks(z: np.ndarray, w: np.ndarray) -> list[tuple[float, float]]:
    """(height, w) of the maxima of one save's strongest-w profile above :data:`PULSE_FROM_M`, at least
    :data:`PULSE_W_MIN` and standing :data:`PULSE_PROMINENCE` above the lowest dip to the next higher maximum on
    either side, the height refined by a parabola through the three levels."""

    w = np.nan_to_num(np.asarray(w, np.float64), nan=-np.inf)
    out = []
    for k in range(1, len(w) - 1):
        if not (w[k] > w[k - 1] and w[k] >= w[k + 1]) or w[k] < PULSE_W_MIN or z[k] < PULSE_FROM_M:
            continue
        left = w[:k][::-1]
        right = w[k + 1:]

        def dip(side: np.ndarray) -> float:
            low = w[k]
            for v in side:
                if v > w[k]:
                    return low
                low = min(low, v)
            return low

        if w[k] - max(dip(left), dip(right)) < PULSE_PROMINENCE:
            continue
        zz, ww = np.asarray(z[k - 1:k + 2], np.float64), w[k - 1:k + 2]
        c = np.polyfit(zz - zz[1], ww, 2)
        zp = float(np.clip(zz[1] - c[1] / (2 * c[0]), zz[0], zz[2])) if c[0] < 0 else float(zz[1])
        out.append((zp, float(np.polyval(c, zp - zz[1]))))
    return out


def pulse_tracks(times: Sequence[float], z: np.ndarray, profiles: Sequence[np.ndarray],
                 interval_s: float) -> list[list[tuple[float, float, float]]]:
    """Rising pulses: a maximum joins a track that ended at the previous save when it is 0 to
    :data:`PULSE_RISE_MAX_M` (per minute) higher, the pair nearest the track's last rise first; tracks of
    :data:`PULSE_MIN_SAVES` saves or more.  Each track is [(t, height, w), ...]."""

    done: list[list[tuple[float, float, float]]] = []
    active: list[list[tuple[float, float, float]]] = []
    prev_t = None
    for t, w in zip(times, profiles):
        peaks = pulse_peaks(z, w)
        gap_ok = prev_t is not None and t - prev_t <= 1.5 * interval_s
        rise_max = PULSE_RISE_MAX_M * max(interval_s, 1.0) / 60.0
        pairs = []
        if gap_ok:
            for ti, tr in enumerate(active):
                pred = (tr[-1][1] - tr[-2][1]) if len(tr) >= 2 else 0.5 * rise_max
                for pi, (zp, _wp) in enumerate(peaks):
                    dz = zp - tr[-1][1]
                    if 0.0 <= dz <= rise_max:
                        pairs.append((abs(dz - pred), ti, pi))
        pairs.sort()
        used_t, used_p, nxt = set(), set(), []
        for _d, ti, pi in pairs:
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            active[ti].append((t, *peaks[pi]))
            nxt.append(active[ti])
        done += [tr for ti, tr in enumerate(active) if ti not in used_t]
        nxt += [[(t, *p)] for pi, p in enumerate(peaks) if pi not in used_p]
        active = nxt
        prev_t = t
    done += active
    return [tr for tr in done if len(tr) >= PULSE_MIN_SAVES]


def rise_speed(track: Sequence[tuple[float, float, float]]) -> float:
    """The least-squares rise (m/s) of a track's height against time: the speed of the pattern, not of the air."""

    t = np.array([p[0] for p in track])
    z = np.array([p[1] for p in track])
    return float(np.polyfit(t - t[0], z, 1)[0]) if len(t) >= 2 else float("nan")


# ----------------------------------------------------------------------------------------------- drawing
def _ramp(name: str, stops: Sequence[tuple[float, str]], vmin: float, vmax: float, unit: str, use: str):
    from . import style as S

    return S.Ramp(name, S.value_stops(stops), vmin, vmax, unit, "sequential", use)


def updraft_ramp():
    """Rising air, black through deep red, orange and amber to cream (lightness rises all the way)."""

    return _ramp("updraft_over_time", ((0.0, "#07080c"), (5.0, "#2a0b10"), (15.0, "#5e0f1c"), (30.0, "#a3241f"),
                                       (45.0, "#e04f16"), (60.0, "#f88f27"), (75.0, "#fcc867"), (90.0, "#fee9b8"),
                                       (100.0, "#fffaf0")), 0.0, 100.0, "m/s", "rising air over time")


def spin_ramp():
    """Counter-clockwise spin: the cyclonic half of the spin look's colours, black to crimson, amber and cream."""

    return _ramp("spin_over_time", ((0.0, "#07080c"), (0.2, "#2a0f1d"), (0.5, "#a3304f"), (0.78, "#e39a2d"),
                                    (1.0, "#fff4dc")), 0.0, 1.0, "", "spin over time, logarithmic")


@dataclass
class Axes:
    """A time-height panel's pixels: (x0, y0, x1, y1), the times' column edges and the height mapping."""

    box: tuple[int, int, int, int]
    t_lo: float
    t_hi: float
    z_top: float
    sqrt: bool = False

    def X(self, t: float | np.ndarray) -> np.ndarray:
        x0, _y0, x1, _y1 = self.box
        return x0 + (np.asarray(t, np.float64) - self.t_lo) / (self.t_hi - self.t_lo) * (x1 - x0)

    def frac(self, z: float | np.ndarray) -> np.ndarray:
        f = np.clip(np.asarray(z, np.float64) / self.z_top, 0.0, 1.0)
        return np.sqrt(f) if self.sqrt else f

    def Y(self, z: float | np.ndarray) -> np.ndarray:
        _x0, y0, _x1, y1 = self.box
        return y1 - self.frac(z) * (y1 - y0)

    def z_at(self, y: np.ndarray) -> np.ndarray:
        _x0, y0, _x1, y1 = self.box
        f = (y1 - np.asarray(y, np.float64)) / (y1 - y0)
        return (f ** 2 if self.sqrt else f) * self.z_top


def column_edges(times: Sequence[float]) -> np.ndarray:
    """Each saved time's column spans half-way to its neighbours; the outer halves mirror the inner ones."""

    t = np.asarray(times, np.float64)
    if len(t) == 1:
        return np.array([t[0] - 150.0, t[0] + 150.0])
    mid = 0.5 * (t[1:] + t[:-1])
    return np.concatenate([[t[0] - (mid[0] - t[0])], mid, [t[-1] + (t[-1] - mid[-1])]])


def raster(ax: Axes, edges_t: np.ndarray, levels_z: np.ndarray, values: np.ndarray, colour: Callable,
           dim: Sequence[bool] | None = None) -> np.ndarray:
    """The panel's RGB pixels: each column's value at each level, nearest level, no interpolation in time or height;
    a NaN or a level above the data is drawn as the empty background, and a column whose ``dim`` is set at 40 %."""

    from . import style as S

    x0, y0, x1, y1 = ax.box
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    tt = ax.t_lo + (xs - x0) / (x1 - x0) * (ax.t_hi - ax.t_lo)
    n = np.searchsorted(edges_t, tt, side="right") - 1
    valid_x = (n >= 0) & (n < values.shape[0])
    n = np.clip(n, 0, values.shape[0] - 1)
    zz = ax.z_at(ys)
    z = np.asarray(levels_z, np.float64)
    # each level spans half-way to its neighbours; the lowest reaches down to the ground
    top_edge = z[-1] + (0.5 * (z[-1] - z[-2]) if len(z) > 1 else 0.0)
    zedges = np.concatenate([[0.0], 0.5 * (z[1:] + z[:-1]), [top_edge]])
    k = np.searchsorted(zedges, zz, side="right") - 1
    valid_y = (k >= 0) & (k < len(z))
    k = np.clip(k, 0, len(z) - 1)
    v = values[n[None, :], k[:, None]]
    rgb = colour(v)
    void = np.array(S.hex_rgb("void"), np.float32)
    if dim is not None:
        faded = np.asarray(dim, bool)[n] & valid_x
        rgb[:, faded] = void + (rgb[:, faded] - void) * np.float32(0.4)
    empty = ~(valid_y[:, None] & valid_x[None, :]) | ~np.isfinite(v)
    rgb[empty] = void
    return S.to8(rgb)


def _lines(size: tuple[int, int], polylines: Sequence[tuple[Sequence[tuple[float, float]], float]],
           ss: int = 3) -> np.ndarray:
    """An anti-aliased coverage mask (0..1) of polylines [(points in pixels, width px)]; one point is a dot."""

    from PIL import Image, ImageDraw

    W, H = size
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    for pts, width in polylines:
        p = [(x * ss, y * ss) for x, y in pts]
        if len(p) >= 2:
            d.line(p, fill=255, width=max(1, int(round(width * ss))), joint="curve")
        elif p:
            r = width * ss
            d.ellipse((p[0][0] - r, p[0][1] - r, p[0][0] + r, p[0][1] + r), fill=255)
    return np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255.0


def _paint(cv: np.ndarray, mask: np.ndarray, colour: str, alpha: float = 1.0) -> None:
    from . import style as S

    a = (mask * alpha)[..., None]
    col = np.array(S.hex_rgb(colour), np.float32) * 255.0
    cv[:] = (cv.astype(np.float32) * (1.0 - a) + col * a + 0.5).astype(np.uint8)


def _ramp_bar(ov: Any, ramp: Any, box: tuple[int, int, int, int], positions: Sequence[tuple[float, str]],
              title: str, *, vertical: bool = False) -> None:
    """A colour key under its title, ticks at the given positions (0..1 along the bar)."""

    from PIL import Image

    x0, y0, x1, y1 = (int(v) for v in box)
    vals = np.linspace(0.0, 1.0, max(2, x1 - x0))
    strip = np.repeat((ramp.colours(vals, 0.0, 1.0) * 255).astype(np.uint8)[None], max(1, y1 - y0), axis=0)
    ov.draw_rgba(Image.fromarray(strip).convert("RGBA"), (x0, y0))
    ms = ov.px("micro")
    for p, lab in positions:
        tx = x0 + p * (x1 - x0)
        anc = "ls" if tx - x0 < 4 else "rs" if x1 - tx < 4 else "ms"
        ov.rule(tx, y1, ov.U(4), thickness=1, colour="muted", vertical=True)
        ov.text((tx, y1 + ov.U(8) + ms), lab, "micro", anchor=anc, halo=False, vertical=vertical)
    ov.text((x0, y0 - ov.U(9)), title, "micro", colour="data", anchor="ls", halo=False, vertical=vertical)


def time_ticks(t_lo: float, t_hi: float, hours: bool) -> list[tuple[float, str]]:
    """Round times inside [t_lo, t_hi] for the time axis, labelled as the clock says them."""

    span = max(t_hi - t_lo, 60.0)
    steps = [60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 21600, 43200, 86400]
    step = next((s for s in steps if span / s <= 9), 86400 * max(1, int(span // (86400 * 9)) + 1))
    first = math.ceil(max(t_lo, 0.0) / step) * step
    out = []
    t = first
    while t <= t_hi + 1e-6:
        if hours:
            whole = int(round(t / 3600.0))
            out.append((t, f"{whole} h" if whole % 24 else (f"{whole // 24} d" if whole else "0")))
        else:
            s = int(round(t))
            out.append((t, f"{s // 3600}:{s % 3600 // 60:02d}"))
        t += step
    return out


def _height_ticks(ax: Axes) -> list[float]:
    if ax.sqrt:
        cand = [0.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 8000.0, 12000.0, 16000.0, 20000.0]
    else:
        every = 2000 if ax.box[3] - ax.box[1] >= 300 else 4000      # a short panel gets fewer labels
        cand = [float(z) for z in range(0, 26000, every)]
    return [z for z in cand if z <= ax.z_top + 1.0]


def _num(v: float) -> str:
    """A spin value with the digits it needs (0.001, 0.05, 0.2)."""

    from . import style as S

    if not v:
        return "0"
    text = S.fmt_num(v, max(0, 1 - int(math.floor(math.log10(abs(v)) + 1e-9))))
    return text.rstrip("0").rstrip(".") if "." in text else text


def _km(z: float) -> str:
    return "0" if z == 0 else (f"{z / 1000:g}")


@dataclass
class Picture:
    """What both looks share: the run's numbers, the following, and the frame they end at."""

    rows: list[dict[str, Any]]
    followed: Followed
    meta: dict[str, Any]
    skipped: list[dict[str, Any]]
    kind: str
    interval_s: float | None

    @property
    def times(self) -> list[float]:
        return [float(r["t"]) for r in self.rows]

    @property
    def cols(self) -> list[Columns]:
        return columns_of(self.rows, self.followed)


def has_spin(pic: Picture) -> bool:
    """Whether any column has a spin value (frames with no zvort, and no u and v, have none)."""

    return any(np.isfinite(c.zeta_max).any() for c in pic.cols)


def _times_word(n: int) -> str:
    return "saved time" if n == 1 else "saved times"


def _axes_text(ov: Any, ax: Axes) -> None:
    x0, y0, _x1, _y1 = ax.box
    for z in _height_ticks(ax):
        y = float(ax.Y(z))
        ov.rule(x0 - ov.U(6), y, ov.U(4), thickness=1, colour="faint")
        ov.text((x0 - ov.U(10), y + ov.U(5)), _km(z), "micro", anchor="rs", halo=False)
    ov.text((x0 - ov.U(10), y0 - ov.U(6)), "km", "micro", anchor="rs", halo=False)


def _time_axis(ov: Any, ax: Axes, y: float, hours: bool) -> None:
    x0, _y0, x1, _y1 = ax.box
    for t, lab in time_ticks(ax.t_lo, ax.t_hi, hours):
        x = float(ax.X(t))
        if x < x0 - 1 or x > x1 + 1:
            continue
        ov.rule(x, y + ov.U(2), ov.U(5), thickness=1, colour="faint", vertical=True)
        ov.text((x, y + ov.U(21)), lab, "micro", anchor="ms", halo=False)
    ov.text((x1, y + ov.U(42)), "storm time since the start (hours)" if hours else
            "storm time since the start (h:mm)", "micro", anchor="rs", halo=False)


def _grid_lines(cv: np.ndarray, ax: Axes, hours: bool) -> None:
    """Faint lines at the height ticks and the time ticks, over the data (thin, so the columns stay readable)."""

    from . import style as S

    x0, y0, x1, y1 = ax.box
    rule = np.array(S.hex_rgb("rule"), np.float32) * 255
    for z in _height_ticks(ax)[1:]:
        y = int(round(float(ax.Y(z))))
        if y0 < y < y1:
            row = cv[y, x0:x1].astype(np.float32)
            cv[y, x0:x1] = (0.55 * row + 0.45 * rule).astype(np.uint8)
    for t, _lab in time_ticks(ax.t_lo, ax.t_hi, hours):
        x = int(round(float(ax.X(t))))
        if x0 < x < x1:
            col = cv[y0:y1, x].astype(np.float32)
            cv[y0:y1, x] = (0.55 * col + 0.45 * rule).astype(np.uint8)


def _strip(cv: np.ndarray, box: tuple[int, int, int, int], ax: Axes, times: Sequence[float],
           values: Sequence[float], top: float, colour: str) -> None:
    """A line under the panels: one dot per saved time, joined; values above ``top`` are drawn at the top."""

    from . import style as S

    x0, y0, x1, y1 = box
    v = np.clip(np.asarray(values, np.float64), 0.0, top)
    cv[y1:y1 + 1, x0:x1] = (np.array(S.hex_rgb("rule"), np.float32) * 255).astype(np.uint8)
    segs: list[list[tuple[float, float]]] = []
    run: list[tuple[float, float]] = []
    for t, val in zip(times, v):
        if np.isfinite(val):
            run.append((float(ax.X(t)), y1 - float(val) / top * (y1 - y0)))
        elif run:
            segs.append(run)
            run = []
    if run:
        segs.append(run)
    dots = [([p], 2.2) for seg in segs for p in seg]
    _paint(cv, _lines((cv.shape[1], cv.shape[0]), [(seg, 1.6) for seg in segs] + dots), colour)


def _over_top(values: Sequence[float], times: Sequence[float], top: float, fmt: Callable[[float], str],
              clock: Callable[[float], str]) -> str:
    """'; above the range at 3 times, peak 1680 at T+0:40:00' for a line whose values pass its fixed top, else ''."""

    v = np.asarray(values, np.float64)
    over = np.isfinite(v) & (v > top)
    if not over.any():
        return ""
    n = int(np.nanargmax(np.where(np.isfinite(v), v, -np.inf)))
    count = int(over.sum())
    return (f"; above the range at {count} {_times_word(count)}, drawn at its top, peak {fmt(float(v[n]))} at "
            f"{clock(times[n])}")


def helicity_top(values: Sequence[float], grid_top: float) -> float:
    """The helicity line's top: the grid's fixed top (:func:`uh_top_for`), raised to the 1-2-5 step at or above the
    run's peak when the peak passes it, so the line is never drawn flat against its top."""

    v = np.asarray(values, np.float64)
    peak = float(np.max(v[np.isfinite(v)])) if np.isfinite(v).any() else 0.0
    return grid_top if peak <= grid_top else float(_nice(peak))


def _strip_ticks(ov: Any, box: tuple[int, int, int, int], top: float, ticks: Sequence[tuple[float, str]]) -> None:
    x0, y0, _x1, y1 = box
    for val, lab in ticks:
        y = y1 - val / top * (y1 - y0)
        ov.rule(x0 - ov.U(6), y, ov.U(4), thickness=1, colour="faint")
        ov.text((x0 - ov.U(10), y + ov.U(5)), lab, "micro", anchor="rs", halo=False)


def _marks(ov: Any, ax: Axes, edges_t: np.ndarray, flags: Sequence[bool], colour: str, y: float) -> None:
    """A short bar above the panel over every column whose flag is set."""

    for n, on in enumerate(flags):
        if on:
            xa, xb = float(ax.X(edges_t[n])), float(ax.X(edges_t[n + 1]))
            ov.rule(xa + 1, y, max(1.0, xb - xa - 2), thickness=4, colour=colour, alpha=0.9)


def _inset(cv: np.ndarray, ov: Any, pic: Picture, box: tuple[int, int, int, int], radius: float) -> None:
    """Where the circle was, on the ground: the followed centres (older dimmer), the last circle, and the domain
    when the storm was first (dim) and last (grey) followed."""

    from PIL import Image, ImageDraw

    from . import style as S

    x0, y0, x1, y1 = box
    path = [p for p in pic.followed.path if p is not None]
    grid = pic.meta["grid"]
    lx, ly = grid["nx"] * grid["dx"], grid["ny"] * grid["dy"]
    followed = [r for r, p in zip(pic.rows, pic.followed.path) if p is not None] or [pic.rows[-1]]
    first_loc = [float(v) for v in (followed[0].get("domainloc") or (0.0, 0.0))]
    gx0, gy0 = (float(v) for v in (followed[-1].get("domainloc") or (0.0, 0.0)))
    xs = [gx0, gx0 + lx, first_loc[0], first_loc[0] + lx] + [p[0] for p in path]
    ys = [gy0, gy0 + ly, first_loc[1], first_loc[1] + ly] + [p[1] for p in path]
    xa, xb, ya, yb = min(xs), max(xs), min(ys), max(ys)
    span = max(xb - xa, yb - ya) * 1.04
    cx, cy = 0.5 * (xa + xb), 0.5 * (ya + yb)
    w, h = x1 - x0, y1 - y0
    mpp = span / min(w, h)

    def px(gx: float, gy: float) -> tuple[float, float]:
        return x0 + w / 2 + (gx - cx) / mpp, y0 + h / 2 - (gy - cy) / mpp

    ss = 3
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def loc(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[0] - x0) * ss, (p[1] - y0) * ss)

    if first_loc != [gx0, gy0]:
        a, b = px(first_loc[0], first_loc[1] + ly), px(first_loc[0] + lx, first_loc[1])
        d.rectangle((*loc(a), *loc(b)), outline=S.rgb8("dim") + (255,), width=ss)
    a, b = px(gx0, gy0 + ly), px(gx0 + lx, gy0)
    d.rectangle((*loc(a), *loc(b)), outline=S.rgb8("muted") + (255,), width=ss)
    n = len(path)
    for m, p in enumerate(path):
        q = loc(px(*p))
        r = 2.2 * ss
        alpha = int(90 + 165 * (m + 1) / n)
        d.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), fill=S.rgb8("gold") + (alpha,))
    if path:
        q = loc(px(*path[-1]))
        r = max(radius / mpp * ss, 4 * ss)
        d.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), outline=S.rgb8("gold") + (255,), width=2 * ss)
    patch = im.resize((w, h), Image.LANCZOS)
    ov.draw_rgba(patch, (x0, y0))
    ov.text((x0, y0 - ov.U(10)), "where the circle was, on the ground", "micro", colour="data", anchor="ls",
            halo=False)
    bar = _nice(span / 4)
    ov.rule(x0, y1 + ov.U(12), bar / mpp, thickness=2, colour="data", alpha=0.85)
    ov.text((x0 + bar / mpp + ov.U(8), y1 + ov.U(13)), S.fmt_km(bar), "micro", colour="data", anchor="lm",
            halo=False)
    ov.text((x1 + ov.U(14), y0 + ov.U(12)), "gold: its centre", "micro", anchor="ls", halo=False)
    ov.text((x1 + ov.U(14), y0 + ov.U(30)), "at each saved", "micro", anchor="ls", halo=False)
    ov.text((x1 + ov.U(14), y0 + ov.U(48)), "time with a storm", "micro", anchor="ls", halo=False)
    ov.text((x1 + ov.U(14), y0 + ov.U(76)), "boxes: the domain", "micro", anchor="ls", halo=False)
    ov.text((x1 + ov.U(14), y0 + ov.U(94)), "when it was first", "micro", anchor="ls", halo=False)
    ov.text((x1 + ov.U(14), y0 + ov.U(112)), "and last followed", "micro", anchor="ls", halo=False)


#: The small map's place: under the side words, no higher than INSET_TOP, no lower than INSET_BOTTOM (the
#: footer's), and drawn only when at least INSET_MIN px square fits.
INSET_TOP, INSET_BOTTOM, INSET_SIZE, INSET_MIN = 690, 965, 220, 140


def _side_words(ov: Any, words: Sequence[tuple[str, str]], y: float, *, vertical: bool = False) -> float:
    """The side column's sentences, wrapped, from ``y`` down; returns the height under the last one."""

    yy = y
    for text, colour_name in words:
        for line in ov.wrap(text, "micro", 1840 - RIGHT_X):
            ov.text((RIGHT_X, yy), line, "micro", colour=colour_name, anchor="ls", halo=False, vertical=vertical)
            yy += ov.U(19)
        yy += ov.U(8)
    return yy


def _place_inset(cv: np.ndarray, ov: Any, pic: Picture, yy: float, radius: float) -> bool:
    """The small map under the side words, as large as fits above the footer; returns whether it was drawn."""

    top = int(max(yy + 30, INSET_TOP))
    size = min(INSET_SIZE, INSET_BOTTOM - top)
    if size < INSET_MIN:
        return False
    _inset(cv, ov, pic, (RIGHT_X, top, RIGHT_X + size, top + size), radius)
    return True


def _nice(m: float) -> float:
    mag = 10 ** math.floor(math.log10(max(m, 1e-9)))
    for f in (1, 2, 5, 10):
        if f * mag >= m:
            return f * mag
    return 10 * mag


def _ordinal(n: int) -> str:
    n = int(n)
    tail = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{tail}"


W, H = 1920, 1080
PANEL_X = (150, 1420)
RIGHT_X = 1480


def _duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:g} h"
    if seconds >= 60:
        return f"{seconds / 60:g} min"
    return f"{seconds:g} s"


def _interval_words(seconds: float | None) -> str:
    return f"every {_duration(seconds)}" if seconds else "for the one saved time"


def _gap_words(times: Sequence[float]) -> str:
    """How far apart the saved times are: 'every 5 min', or, when they are not evenly spaced (a run resumed, or
    saved times that could not be read), '5 to 10 min apart'; '' for a single saved time."""

    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    if not gaps:
        return ""
    lo, hi = gaps[0], gaps[-1]
    if hi <= lo * 1.01:
        return _interval_words(lo)
    shortest, longest = _duration(lo), _duration(hi)
    if shortest.split()[1] == longest.split()[1]:
        shortest = shortest.split()[0]            # "5 to 10 min", not "5 min to 10 min"
    return f"{shortest} to {longest} apart"


def _cadence_words(times: Sequence[float]) -> str:
    """How the columns are spaced: 'a column every 5 min', or 'a column per saved time, 5 to 10 min apart'."""

    gap = _gap_words(times)
    if not gap:
        return "one saved time"
    return f"a column {gap}" if gap.startswith("every") else f"a column per saved time, {gap}"


def _flags(pic: Picture, spin: bool = False) -> tuple[list[bool], list[bool]]:
    """Per column with a storm: the storm was at the domain's edge (its circle reached past the edge, or its
    centre was as near the edge as the storm finder looks, where an updraft leaving the domain is held); the
    followed updraft vanished."""

    f = pic.followed
    n = len(pic.rows)
    if f.mode != "storm":
        return [False] * n, [False] * n
    grid = pic.meta["grid"]
    edge = []
    for r, c, col in zip(pic.rows, f.chosen, pic.cols):
        cut = col.spin_at_edge if spin else col.at_edge
        held = c is not None and at_finder_edge(float(r["centres"][c][0]), float(r["centres"][c][1]), grid)
        edge.append(bool(cut or held))
    return ([bool(e and s) for e, s in zip(edge, f.storm)], [bool(x and s) for x, s in zip(f.lost, f.storm)])


def storm_words() -> str:
    """What counts as a storm, in a picture's words (the threshold behind the dimmed columns)."""

    return (f"A storm: rising air of {STORM_MIN:g} m/s or more on average from {FIND_LAYER_M[0] / 1000:g} to "
            f"{FIND_LAYER_M[1] / 1000:g} km, smoothed over {FIND_SMOOTH_M / 1000:g} km; the circle is centred where "
            "it is strongest.")


def storm_caption(pic: Picture) -> str:
    """The line under a thunderstorm picture's time axis: what the dimmed columns are, and what counts as a storm;
    '' for a hurricane."""

    f = pic.followed
    if f.mode == "hurricane":
        return ""
    if f.mode == "domain":
        return storm_words().split(";")[0] + "."
    return ("Dimmed: no storm. " if f.storm_times < len(pic.rows) else "") + storm_words()


def _caption(ov: Any, pic: Picture, ax: Axes, y: float, hours: bool) -> None:
    """:func:`storm_caption` under the time axis, left of its label."""

    text = storm_caption(pic)
    if not text:
        return
    label = "storm time since the start (hours)" if hours else "storm time since the start (h:mm)"
    width = int(ax.box[2] - ax.box[0] - ov.measure(label, "micro")[0] - ov.U(40))
    for n, line in enumerate(ov.wrap(text, "micro", width)):
        ov.text((ax.box[0], y + n * ov.U(19)), line, "micro", colour="muted", anchor="ls", halo=False,
                vertical=True)


def _followed_words(pic: Picture, edge_flags: Sequence[bool], lost_flags: Sequence[bool],
                    clock: Callable[[float], str]) -> list[tuple[str, str]]:
    f = pic.followed
    n = len(pic.rows)
    if f.mode == "domain":
        return [(f.reason, "text")]
    if f.mode == "hurricane":
        each = "the one saved time" if n == 1 else f"each of the {n} saved times"
        return [(f"The circle follows the hurricane's centre (the lowest pressure at the lowest level) at {each}.",
                 "text")]
    if f.storm_times == 0:
        first = ("No storm yet at the one saved time: its column is dimmed" if n == 1 else
                 f"No storm yet: none of the {n} saved times had one, and the columns are dimmed")
        first += "; the circle sits on the strongest rising air there was."
    elif n == 1:
        first = "One storm followed: the one saved time had one."
    else:
        first = (f"One storm followed: {f.storm_times} of the {n} saved times had one."
                 + (" The columns without one are dimmed; there the circle sits on the strongest rising air there "
                    "was." if f.storm_times < n else ""))
    words = [(first, "text")]
    if any(edge_flags):
        # (at the edge: the circle reached past it, or the updraft was as near it as the storm finder looks)
        words.append(("Grey bar above a column: the storm was at the domain's edge; part of it may be missing.",
                      "muted"))
        last = max((m for m, s in enumerate(f.storm) if s), default=None)
        if last is not None and last < n - 1 and edge_flags[last]:
            words.append((f"The last saved time with a storm ({clock(pic.times[last])}) was at the edge: the storm "
                          "may have left the domain rather than died out.", "muted"))
    if any(lost_flags):
        words.append(("White bar above a column: the followed updraft vanished; the strongest one then was taken.",
                      "muted"))
    return words


def _following_words(pic: Picture, spin: bool = False) -> str:
    from . import style as S

    f = pic.followed
    radius = pic.meta.get("spin_radius_m", pic.meta["radius_m"]) if spin else pic.meta["radius_m"]
    if f.mode == "hurricane":
        return f"within {S.fmt_km(radius)} of the lowest pressure"
    if f.mode == "domain":
        return "the whole domain"
    return f"within {S.fmt_km(radius)} of the main updraft"


def _base_overlay(look: str, facts: Any, pic: Picture, notes: str, spin: bool = False) -> Any:
    from . import style as S

    ov = S.Overlay((W, H))
    t_last = pic.times[-1]
    data = S.data_line(facts.name, f"{S.fmt_grid(pic.meta['grid']['dx'])} grid", _following_words(pic, spin),
                       _cadence_words(pic.times))
    ov.title(TITLES[look], data)
    ov.clock(facts.clock(t_last))
    z0 = pic.meta["grid"]["zh"][0]
    ov.footer(facts.footer(t=t_last, level=f"lowest level {S.fmt_height(z0)}", wind="", notes=notes),
              vertical=[notes])
    ov.mark()
    return ov


def _top_of(pic: Picture) -> float:
    zf = pic.meta["grid"]["zf"]
    cap = HURRICANE_HEIGHT_TOP_M if pic.kind == "hurricane" else HEIGHT_TOP_M
    return float(min(zf[-1], cap))


def draw_updraft(pic: Picture, facts: Any, out: Path) -> dict[str, Any]:
    """The updraft over time (see the module's words)."""

    from . import products as P
    from . import style as S

    hurricane = pic.kind == "hurricane"
    times = pic.times
    cols = pic.cols
    grid = pic.meta["grid"]
    zf = np.asarray(grid["zf"], np.float64)
    top = W_TOP if not hurricane else HURRICANE_W_TOP
    ramp = updraft_ramp()
    colour = lambda v: ramp.colours(np.asarray(v, np.float64) / top * 100.0, 0.0, 100.0)  # noqa: E731
    edges = column_edges(times)
    z_top = _top_of(pic)
    hours = hurricane or (times[-1] - times[0]) >= 12 * 3600
    a = Axes((PANEL_X[0], 205, PANEL_X[1], 575), float(edges[0]), float(edges[-1]), z_top)
    b = Axes((PANEL_X[0], 632, PANEL_X[1], 790), float(edges[0]), float(edges[-1]), z_top)
    strip = (PANEL_X[0], 842, PANEL_X[1], 902)
    w_max = np.stack([c.w_max for c in cols])
    w_fast = np.stack([c.w_fast for c in cols])
    cv = P.canvas()
    dim = [not s for s in pic.followed.storm] if pic.followed.mode == "storm" else None
    for ax, values in ((a, w_max), (b, w_fast)):
        x0, y0, x1, y1 = ax.box
        cv[y0:y1, x0:x1] = raster(ax, edges, zf, values, colour, dim)
        _grid_lines(cv, ax, hours)
    # the cloud top: a step line across each column
    ctop = [c.cloud_top for c in cols]
    steps: list[list[tuple[float, float]]] = []
    run: list[tuple[float, float]] = []
    for n, z in enumerate(ctop):
        if np.isfinite(z):
            y = float(a.Y(z))
            run += [(float(a.X(edges[n])), y), (float(a.X(edges[n + 1])), y)]
        elif run:
            steps.append(run)
            run = []
    if run:
        steps.append(run)
    _paint(cv, _lines((W, H), [(s, 1.4) for s in steps]), "cold", 0.95)
    # pulse tracks, only with saves 60 s apart or closer, and only for one followed storm (in the whole domain's
    # columns, one save's maximum and the next one's may belong to different storms)
    tracks: list[list[tuple[float, float, float]]] = []
    close = bool(pic.interval_s) and pic.interval_s <= PULSE_INTERVAL_S + 1e-6
    pulses_on = close and pic.followed.mode != "domain"
    if pulses_on:
        tracks = pulse_tracks(times, zf, list(w_max), float(pic.interval_s))
        polys = []
        for tr in tracks:
            pts = [(float(a.X(t)), float(a.Y(z))) for t, z, _w in tr]
            polys.append((pts, 1.1))
            polys += [([p], 1.8) for p in pts]
        _paint(cv, _lines((W, H), polys), "#ffffff", 0.85)
    # under them: the strongest spin at the lowest level (a thunderstorm), or the fastest wind there (a hurricane)
    z0 = float(grid["zh"][0])
    if hurricane:
        strip_values = [c.wind0_max for c in cols]
        strip_top = HURRICANE_WIND_TOP
        strip_ticks = [(0.0, "0"), (40.0, "40"), (80.0, "80")]
        strip_colour = "#8fd3ff"
        strip_over = _over_top(strip_values, times, strip_top, lambda v: f"{v:.0f}", facts.clock)
    else:
        # the spin look's logarithmic scale: 0 at a tenth of the grid's range, 1 at five times it
        rng = spin_range_for(grid["dx"], z0)
        lo, hi = rng / SPIN_LOG_BELOW, rng * SPIN_LOG_ABOVE
        span = math.log10(hi / lo)
        strip_values = [(math.log10(max(float(c.zeta_max[0]), lo)) - math.log10(lo)) / span
                        if np.isfinite(c.zeta_max[0]) else float("nan") for c in cols]
        strip_top = 1.0
        strip_ticks = [(0.0, _num(lo)), (math.log10(rng / lo) / span, _num(rng)), (1.0, _num(hi))]
        strip_colour = "#e39a2d"
        strip_over = _over_top([c.zeta_max[0] for c in cols], times, hi, lambda v: S.fmt_num(v, 2), facts.clock)
    spin_line = hurricane or np.isfinite(strip_values).any()
    _strip(cv, strip, a, times, strip_values, strip_top, strip_colour)
    notes = S.data_line(
        "w on the model's w levels, nearest level, nothing interpolated",
        (f"fastest {'hundredth' if hurricane else 'tenth'}: the {_ordinal(pic.meta['k_fast'])} fastest of the "
         f"{pic.meta.get('cells', '')} cells in the circle" if pic.followed.mode != "domain" else
         f"second panel: the {_ordinal(pic.meta['k_fast'])} fastest cell anywhere in the domain"),
        f"cloud top: cloud water + ice {S.fmt_num(CLOUD_G_KG, 1)} g/kg or more")
    ov = _base_overlay(UPDRAFT, facts, pic, notes)
    in_words = "in the circle" if pic.followed.mode != "domain" else "in the domain"
    ov.text((a.box[0], a.box[1] - ov.U(12)), f"strongest rising air at each height {in_words}", "label",
            colour="text", anchor="ls")
    ov.text((a.box[2], a.box[1] - ov.U(12)), "blue line: the cloud top", "micro", colour="cold", anchor="rs",
            halo=False)
    share_words = "hundredth" if hurricane else "tenth"
    if pic.followed.mode != "domain":
        fastest = f"the speed the fastest {share_words} of the circle reaches"
    else:
        area = pic.meta["k_fast"] * grid["dx"] * grid["dy"] / 1.0e6
        digits = 0 if area >= 10 or abs(area - round(area)) < 0.05 else 1
        fastest = (f"the speed the fastest {S.fmt_num(area, digits)} km² of the domain reach "
                   f"(as much as a {share_words} of a {S.fmt_km(pic.meta['radius_m'])} circle)")
    ov.text((b.box[0], b.box[1] - ov.U(12)), fastest, "label", colour="text", anchor="ls")
    for ax in (a, b):
        _axes_text(ov, ax)
    _time_axis(ov, a, strip[3], hours)
    _caption(ov, pic, a, strip[3] + ov.U(42), hours)
    if hurricane:
        ov.text((strip[0], strip[1] - ov.U(10)),
                f"fastest wind at the lowest level {in_words} (m/s, ground-relative){strip_over}", "micro",
                colour="#8fd3ff", anchor="ls", halo=False)
    elif spin_line:
        within = (f"within {S.fmt_km(pic.meta.get('spin_radius_m', pic.meta['radius_m']))} of the updraft"
                  if pic.followed.mode != "domain" else "in the domain")
        ov.text((strip[0], strip[1] - ov.U(10)),
                f"strongest spin at the lowest level ({S.fmt_height(z0)}) {within} ({S.PER_S}, logarithmic, "
                f"as the spin over time){strip_over}", "micro", colour="#e39a2d", anchor="ls", halo=False)
    else:
        ov.text((strip[0], strip[1] - ov.U(10)),
                "strongest spin at the lowest level: not in this run's frames (no zvort, and no u and v to work it "
                "out from)", "micro", colour="muted", anchor="ls", halo=False)
    if spin_line:
        _strip_ticks(ov, strip, strip_top, strip_ticks)
    for tr in tracks:
        t_end, z_end, _w = tr[-1]
        label = f"{rise_speed(tr):.0f} · {max(w for _t, _z, w in tr):.0f}"
        lx, ly = float(a.X(t_end)) + ov.U(6), float(a.Y(z_end)) - ov.U(6)
        width = ov.measure(label, "micro", px=ov.U(13))[0]
        from PIL import Image

        ov.draw_rgba(Image.new("RGBA", (width + ov.U(8), ov.U(18)), S.rgb8("void", 0.8)),
                     (lx - ov.U(4), ly - ov.U(14)))
        ov.text((lx, ly), label, "micro", colour="title", anchor="ls", px=ov.U(13), halo=False)
    # marks above the top panel: the circle at the domain's edge, and a followed updraft lost
    edge_flags, lost_flags = _flags(pic)
    _marks(ov, a, edges, edge_flags, "muted", a.box[1] - ov.U(5))
    _marks(ov, a, edges, lost_flags, "#ffffff", a.box[1] - ov.U(5))
    # the key and the numbers, on the right
    rx = RIGHT_X
    ticks = [(v / top, f"{v:g}") for v in ((0, 25, 50, 75, 100) if not hurricane else (0, 10, 20, 30, 40))]
    ticks[-1] = (1.0, f"{top:g} m/s")
    _ramp_bar(ov, ramp, (rx, 222, rx + 340, 232), ticks, f"rising air, 0 to {top:g} m/s, fixed", vertical=True)
    best = int(np.nanargmax(np.nanmax(w_max, axis=1))) if np.isfinite(w_max).any() else 0
    kbest = int(np.nanargmax(w_max[best])) if np.isfinite(w_max[best]).any() else 0
    wbest = float(w_max[best, kbest]) if np.isfinite(w_max).any() else 0.0
    rows_out = [("strongest rising air", f"{wbest:.0f} m/s")]
    vertical = [rows_out[0][1]]
    sub = [f"at {S.fmt_height(float(zf[kbest]))}, {facts.clock(times[best])}"]
    finite_top = [z for z in ctop if np.isfinite(z)]
    # a cloud top at the model's top scalar level is the model's lid, not the cloud's own top
    at_lid = bool(finite_top) and max(finite_top) >= float(grid["zh"][-1]) - 0.5
    if finite_top:
        rows_out.append(("highest cloud top", S.fmt_height(max(finite_top))))
        sub.append("the model's top level: the cloud may reach higher" if at_lid else "")
    y = 290
    for (cap, val), extra in zip(rows_out, sub):
        ov.readout((rx, y), [(cap, val)], vertical=vertical)
        if extra:
            ov.text((rx, y + ov.U(66)), extra, "micro", colour="data", anchor="ls", halo=False)
        y += 96
    f = pic.followed
    words = _followed_words(pic, edge_flags, lost_flags, facts.clock)
    if close and not pulses_on:
        words.append(("Pulse tracks are left off: each column is the whole domain's, so the maxima of one saved "
                      "time and the next may belong to different storms.", "muted"))
    elif pulses_on:
        if tracks:
            speeds = [rise_speed(tr) for tr in tracks]
            words.append((f"White lines: {len(tracks)} rising pulses (maxima of at least {PULSE_W_MIN:g} m/s above "
                          f"{PULSE_FROM_M / 1000:g} km, seen at {PULSE_MIN_SAVES} or more saved times), labelled with "
                          f"their rise and their peak (m/s). They rise at {np.median(speeds):.0f} m/s (median): the "
                          "speed of the pattern, not of the air.", "text"))
        else:
            words.append((f"No rising pulses found (maxima of at least {PULSE_W_MIN:g} m/s above "
                          f"{PULSE_FROM_M / 1000:g} km, seen at {PULSE_MIN_SAVES} or more saved times).", "muted"))
    else:
        gap = _gap_words(times)
        cadence = (f"this run's saved times are {gap}" if gap and not gap.startswith("every")
                   else f"this run saved {gap}" if gap else "this run has one saved time so far")
        words.append((f"Pulse tracks are left off: they need saved times {PULSE_INTERVAL_S:g} s apart or closer, and "
                      f"{cadence}.", "muted"))
    if pic.skipped:
        words.append((f"{len(pic.skipped)} saved time{'s' if len(pic.skipped) > 1 else ''} could not be read and "
                      "left a gap.", "muted"))
    yy = _side_words(ov, words, y + 4, vertical=True)
    inset = f.mode != "domain" and _place_inset(cv, ov, pic, yy, float(pic.meta["radius_m"]))
    P.save(ov, cv, out)
    shown: dict[str, Any] = {"w_max": wbest, "w_max_height_m": float(zf[kbest]), "w_max_time_s": times[best],
                             "following": f.mode, "saved_times": len(cols), "radius_m": pic.meta["radius_m"],
                             "range_m_s": [0.0, top], "pulse_tracks": len(tracks) if pulses_on else None,
                             "storm_times": f.storm_times, "edge_times": int(sum(edge_flags)), "map": inset}
    if finite_top:
        shown["cloud_top_max_m"] = float(max(finite_top))
        shown["cloud_top_at_model_top"] = at_lid
    if not spin_line:
        shown["lowest_spin"] = None
    if f.mode == "domain":
        shown["why_whole_domain"] = f.reason
    return shown


def draw_spin(pic: Picture, facts: Any, out: Path) -> dict[str, Any]:
    """The spin over time (see the module's words)."""

    from . import products as P
    from . import style as S

    times = pic.times
    cols = pic.cols
    grid = pic.meta["grid"]
    zh = np.asarray(grid["zh"], np.float64)
    z0 = float(zh[0])
    rng = spin_range_for(grid["dx"], z0)
    lo, hi = rng / SPIN_LOG_BELOW, rng * SPIN_LOG_ABOVE
    ramp = spin_ramp()

    def colour(v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, np.float64)
        pos = (np.log10(np.clip(x, lo, None)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        pos = np.where(x <= lo, 0.0, pos)
        return ramp.colours(np.where(np.isfinite(x), pos, np.nan), 0.0, 1.0)

    edges = column_edges(times)
    z_top = _top_of(pic)
    hours = (times[-1] - times[0]) >= 12 * 3600
    a = Axes((PANEL_X[0], 205, PANEL_X[1], 715), float(edges[0]), float(edges[-1]), z_top, sqrt=True)
    strip = (PANEL_X[0], 772, PANEL_X[1], 872)
    zeta = np.stack([c.zeta_max for c in cols])
    cv = P.canvas()
    x0, y0, x1, y1 = a.box
    dim = [not s for s in pic.followed.storm] if pic.followed.mode == "storm" else None
    cv[y0:y1, x0:x1] = raster(a, edges, zh, zeta, colour, dim)
    _grid_lines(cv, a, hours)
    uh = [c.uh_max for c in cols]
    # the helicity line's top: fixed for the grid as the colours are, raised to hold the run's peak when it passes
    uh_grid_top = uh_top_for(rng)
    uh_top = helicity_top(uh, uh_grid_top)
    _strip(cv, strip, a, times, uh, uh_top, "#e39a2d")
    notes = S.data_line(f"spin: vertical vorticity ({pic.meta['zeta_source']}), the strongest counter-clockwise",
                        "nearest level, nothing interpolated",
                        f"updraft helicity: {pic.meta['uh_source'] or 'not in this run'}")
    ov = _base_overlay(SPIN, facts, pic, notes, spin=True)
    in_words = "in the circle" if pic.followed.mode != "domain" else "in the domain"
    ov.text((x0, y0 - ov.U(12)), f"strongest counter-clockwise spin at each height {in_words}", "label",
            colour="text", anchor="ls")
    ov.text((x1, y0 - ov.U(12)), "height axis stretched near the ground", "micro", anchor="rs", halo=False)
    _axes_text(ov, a)
    finite_uh = [v for v in uh if np.isfinite(v)]
    if not finite_uh:
        uh_words = "not in this run's frames"
    elif uh_top == uh_grid_top:
        uh_words = f"0 to {uh_top:g}, fixed at this grid"
    else:
        n_peak = int(np.nanargmax(np.where(np.isfinite(uh), uh, -np.inf)))
        uh_words = (f"0 to {uh_top:g}, raised from this grid's fixed {uh_grid_top:g} to hold the peak, "
                    f"{max(finite_uh):.0f} at {facts.clock(times[n_peak])}")
    ov.text((strip[0], strip[1] - ov.U(10)), f"most 2–5 km updraft helicity {in_words} (m² s⁻², {uh_words})",
            "micro", colour="#e39a2d", anchor="ls", halo=False)
    if finite_uh:
        _strip_ticks(ov, strip, uh_top, ((0.0, "0"), (uh_top / 2, f"{uh_top / 2:g}"), (uh_top, f"{uh_top:g}")))
    _time_axis(ov, a, strip[3], hours)
    _caption(ov, pic, a, strip[3] + ov.U(42), hours)
    edge_flags, lost_flags = _flags(pic, spin=True)
    _marks(ov, a, edges, edge_flags, "muted", y0 - ov.U(5))
    _marks(ov, a, edges, lost_flags, "#ffffff", y0 - ov.U(5))
    rx = RIGHT_X
    span = math.log10(hi) - math.log10(lo)
    ticks = []
    for v in (lo, rng, hi):
        ticks.append(((math.log10(v) - math.log10(lo)) / span, _num(v)))
    ticks[-1] = (1.0, f"{ticks[-1][1]} {S.PER_S}")
    _ramp_bar(ov, ramp, (rx, 222, rx + 340, 232), ticks, "spin, logarithmic, fixed at this grid")
    ov.text((rx + (math.log10(rng) - math.log10(lo)) / span * 340, 268),
            spin_range_words(grid["dx"], z0), "micro", colour="muted", anchor="ms", halo=False)
    # the strongest spin from 1 to 6 km (the updraft's own rotation; the spin at the lowest level has its own row,
    # and the anvil above has eddies of its own)
    layer = (zh >= MESO_LAYER_M[0]) & (zh <= MESO_LAYER_M[1])
    finite = np.where(np.isfinite(zeta) & layer[None, :], zeta, -np.inf)
    n, k = np.unravel_index(int(np.argmax(finite)), finite.shape)
    meso = bool(np.isfinite(finite[n, k]))
    zbest = float(zeta[n, k]) if meso else None
    y = 300
    band = f"strongest from {S.fmt_height(MESO_LAYER_M[0])} to {S.fmt_height(MESO_LAYER_M[1])}"
    ov.readout((rx, y), [(band, f"{S.fmt_num(zbest, 3)} {S.PER_S}" if meso else "none")])
    if meso:
        ov.text((rx, y + ov.U(66)), f"at {S.fmt_height(float(zh[k]))}, {facts.clock(times[n])}", "micro",
                colour="data", anchor="ls", halo=False)
    y += 96
    low = [c.zeta_max[0] for c in cols]
    has_low = bool(np.isfinite(low).any())
    nl = int(np.nanargmax(low)) if has_low else 0
    ov.readout((rx, y), [(f"strongest at {S.fmt_height(z0)}",
                          f"{S.fmt_num(float(low[nl]), 3)} {S.PER_S}" if has_low else "none")])
    if has_low:
        ov.text((rx, y + ov.U(66)), facts.clock(times[nl]), "micro", colour="data", anchor="ls", halo=False)
    y += 96
    f = pic.followed
    words = _followed_words(pic, edge_flags, lost_flags, facts.clock)
    words.append((f"Spin narrower than about {S.fmt_km(6 * grid['dx'])} (six grid spacings) is not resolved; the "
                  f"colours stop at {_num(lo)} and at {_num(hi)} {S.PER_S}. Each value is the strongest single "
                  "cell, so it can be one small whirl rather than the storm's whole rotation.", "muted"))
    if pic.skipped:
        words.append((f"{len(pic.skipped)} saved time{'s' if len(pic.skipped) > 1 else ''} could not be read and "
                      "left a gap.", "muted"))
    # the side words' speeds are of rising air (what counts as a storm), not of the wind
    yy = _side_words(ov, words, y + 4, vertical=True)
    inset = f.mode != "domain" and _place_inset(cv, ov, pic, yy,
                                                 float(pic.meta.get("spin_radius_m", pic.meta["radius_m"])))
    P.save(ov, cv, out)
    shown = {"zeta_max_1_6km": zbest, "zeta_max_height_m": float(zh[k]) if meso else None,
             "zeta_max_time_s": times[n] if meso else None,
             "zeta_max_all": float(np.nanmax(zeta)) if np.isfinite(zeta).any() else None,
             "zeta_lowest_max": float(low[nl]) if has_low else None, "range_per_s": rng,
             "range_set_by": "dx" if rng == P.spin_range(float(grid["dx"])) else "lowest level",
             "colours_per_s": [lo, hi], "following": f.mode, "saved_times": len(cols),
             "radius_m": pic.meta.get("spin_radius_m", pic.meta["radius_m"]),
             "helicity_top": uh_top if finite_uh else None, "helicity_grid_top": uh_grid_top,
             "helicity_max": max(finite_uh) if finite_uh else None, "storm_times": f.storm_times,
             "edge_times": int(sum(edge_flags)), "map": inset}
    if f.mode == "domain":
        shown["why_whole_domain"] = f.reason
    return shown


# ----------------------------------------------------------------------------------------------- the render pass
def _signature(lineage: Sequence[tuple[float, str, Path, int]], kind: str, label: str | None) -> str:
    text = json.dumps({"schema": STATE_SCHEMA, "frames": [key for _t, key, _p, _s in lineage], "kind": kind,
                       "label": label})
    return hashlib.sha256(text.encode()).hexdigest()


def _why_not(renderer: Any, lineage: Sequence[Any], pid: str, reason: str) -> str:
    """The reason a look over time is not drawn: as it said it, or, from CM1 netCDF files, the field they lack with the
    CM1 switch that writes it (:func:`.render.why_not`)."""

    if renderer.layout.kind != "netcdf" or not lineage:
        return reason
    from .render import why_not

    _t, key, path, segment = lineage[-1]
    try:
        qf = renderer._open(path, key, segment)
    except (FrameUnreadable, OSError, ValueError):
        return reason
    try:
        return why_not(qf, pid, reason)
    finally:
        qf.close()


def draw_once(renderer: Any, lineage: list[tuple[float, str, Path, int]], *, keep: Collection[str] = (),
              frames_lost: bool = False) -> bool:
    """Draw the two looks for :class:`.render.Renderer` ``renderer`` when its frames changed since they were last
    drawn; returns whether anything was drawn.  The looks in ``keep`` are left as they are; with ``frames_lost``
    (some drawn saved times have lost their saved frames) the numbers kept of each saved time are not written over
    with those of the fewer frames that are left."""

    import time as _time

    from .facts import run_facts

    wanted = [pid for pid in IDS if pid in renderer.once and pid not in keep]
    if not wanted or not lineage:
        return False
    from ..cm1out.frames import FrameError

    once = renderer.state["once"]
    begun = _time.perf_counter()
    facts = None
    for _t, key, path, segment in reversed(lineage):      # the facts of the newest frame that can be read
        try:
            qf = renderer._open(path, key, segment)
        except (FrameError, FrameUnreadable, OSError, ValueError):
            continue
        try:
            facts = run_facts(renderer.layout, qf.config, qf.dx, qf.dy, label=renderer.label,
                              settings=renderer.settings)
        finally:
            qf.close()
        break
    if facts is None:
        return False
    kind = facts.storm_kind
    signature = _signature(lineage, kind, renderer.label)
    if all((once.get(pid) or {}).get("signature") == signature for pid in wanted):
        return False
    builder = Builder(renderer.state_dir)
    try:
        builder.catch_up(lineage, renderer._open, kind)
    except TimeHeightError as why:
        for pid in wanted:
            once[pid] = {"signature": signature, "not_drawn": _why_not(renderer, lineage, pid, str(why))}
        return True
    except Exception as error:      # noqa: BLE001 - these two looks never stop the pass's other looks
        for pid in wanted:
            renderer.summary.problems.append(f"{pid}: {type(error).__name__}: {error}")
            once[pid] = {"signature": signature, "error": f"{type(error).__name__}: {error}"}
        return True
    try:
        if not frames_lost:
            builder.save()
    except OSError as error:
        # The numbers are only a store: the pictures are drawn from the ones in hand, and the next pass reads the
        # saved times again.
        renderer.log(f"{', '.join(wanted)}: the numbers of each saved time could not be kept ({error}); the next "
                     "pass reads them again")
    rows = builder.read_rows
    skipped = [r for r in builder.rows if "skipped" in r]
    read_s = (_time.perf_counter() - begun) / len(wanted)
    times = [float(r["t"]) for r in rows]
    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    interval = gaps[len(gaps) // 2] if gaps else None
    followed = follow(rows, kind) if rows else None
    for pid in wanted:
        if not rows or followed is None:
            once[pid] = {"signature": signature, "not_drawn": "No saved time of this run could be read."}
            continue
        if kind == "hurricane" and pid in THUNDERSTORM_ONLY:
            once[pid] = {"signature": signature, "not_drawn": THUNDERSTORM_ONLY_WHY}
            continue
        drawn_at = _time.perf_counter()
        pic = Picture(rows, followed, builder.meta, skipped, kind, interval)
        if pid == SPIN and not has_spin(pic):
            once[pid] = {"signature": signature, "not_drawn": NO_SPIN_WHY}
            continue
        file = renderer.out / pid / PICTURE
        try:
            shown = (draw_updraft if pid == UPDRAFT else draw_spin)(pic, facts, file)
        except Exception as error:      # noqa: BLE001 - recorded, and the pass's other looks are still drawn
            renderer.summary.problems.append(f"{pid}: {error}")
            once[pid] = {"signature": signature, "error": f"{type(error).__name__}: {error}"}
            continue
        once[pid] = {"signature": signature, "time": times[-1], "file": f"{pid}/{PICTURE}",
                     "thumb": f"{pid}/thumb/{PICTURE}", "facts": shown,
                     "seconds": round(_time.perf_counter() - drawn_at + read_s, 3)}
    return True


__all__ = ["CANDIDATES", "Builder", "Columns", "Followed", "IDS", "NO_SPIN_WHY", "ONCE", "SPIN", "STATE_SCHEMA",
           "THUNDERSTORM_ONLY", "THUNDERSTORM_ONLY_WHY", "TimeHeightError", "UPDRAFT", "at_finder_edge", "box_mean",
           "circle_columns", "column_edges", "domain_columns", "draw_once", "draw_spin", "draw_updraft",
           "find_centres", "finder_margin", "follow", "has_spin", "helicity_top", "kth_largest", "measure",
           "pulse_peaks", "pulse_tracks", "rise_speed", "spin_range_for", "spin_range_words", "storm_words",
           "strongest_places", "time_ticks", "uh_top_for", "zeta_box"]
