"""The storm in 3-D: the whole storm drawn in three dimensions on the CPU, cut open through its core.

A still (``gpuwm-cm1 quicklook RUNDIR --look storm3d``) and a short orbit film (``--film storm3d``), drawn on
request and never by the worker as frames land: one picture takes tens of seconds on one core.

What is drawn, and the picture says each of these:

* the cloud (cloud water, cloud ice and snow) as a white, see-through volume.  Its extinction is a look, not an
  optical property: 0.02 per metre for each g/m³, ramped in from 0.002 g/m³ so the edges thin out, and drawn at
  :data:`CLOUD_SEE_THROUGH` of that so the inside shows.  It is lit by a dim key light and by the sky, each dimmed
  by the cloud between it and the point (optical depth swept toward the light, level by level, as the
  clouds-from-above look does);
* nested see-through shells where the model's reflectivity reaches 30, 45 and 60 dBZ, in the NWS colours, each
  at a stated opacity (15, 40 and 90%).  It is the reflectivity the reflectivity look draws (the scheme's own, or
  CM1's calc_refl10cm), smoothed 1-2-1 across three cells; a run with no reflectivity draws shells of its
  precipitation water instead (rain and graupel or hail, 0.1, 1 and 3 g/m³) and says so;
* the storm cut open along a vertical plane through its core, square to the camera: the half nearer the camera is
  not drawn, and the cut face shows the field on the plane in the shells' bands, nearly solid
  (:data:`CUT_FACE_OPACITY`).  A thunderstorm is cut through its reflectivity core (the strongest reflectivity 1 to
  5 km above ground, averaged over about 1 km); a hurricane through its centre (the lowest pressure at the lowest
  level), so the eye shows.  A time with no echo up to the lowest shell yet (early in a run) is not drawn: there
  is nothing to cut open, and the refusal says so;
* heights at true scale for a thunderstorm.  A hurricane is hundreds of km across and 16 km deep, so its heights
  are stretched (10 times for its inner 500 km); the data line and the height scale say by how much;
* the ground dark and flat, with lines fixed to the ground.

On a large grid (a 67 m run has 22 million cells) the scene is cut to at most :data:`VOXEL_BUDGET` cells by
averaging blocks of s x s columns (the reflectivity is averaged in Z, mm⁶ m⁻³); the picture says
"drawn from 3 × 3 cell averages (300 m), to keep it to 4 million cells".  Levels are kept.  A request whose
estimated time is longer than :data:`STILL_LIMIT_S` (a still) or :data:`ORBIT_LIMIT_S` (an orbit) is refused
with the estimate and the largest choice that fits, before the scene is built (only a coarse pass over the run,
to find the core, comes first).

The ray marcher is numpy only, one process: front to back, a step of 0.6 cells, empty space skipped through a
coarse occupancy grid, each ray stopped once it is nearly opaque.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time as _time
from typing import Any, Callable, Sequence

import numpy as np

from . import products as P
from . import reflectivity as R
from . import style as S
from .frame import FrameUnreadable, QuickFrame

LOOK = "storm3d"
TITLE = "The storm in 3-D"
STILL_SIZE = (1920, 1080)
ORBIT_SIZE = (1280, 720)
#: The smallest picture whose words fit (the type does not shrink below 10 pixels) and the largest.
SIZE_RANGE = ((960, 540), (3840, 2160))
ORBIT_FRAMES = 72
ORBIT_FPS = 12.0
#: The most cells a scene holds; a bigger box is averaged over s x s columns (s the smallest that fits).
VOXEL_BUDGET = 4.0e6
#: The box drawn around a thunderstorm's core (half its width) and around a hurricane's centre.
THUNDERSTORM_HALF_M = 24_000.0
HURRICANE_HALF_M = 250_000.0
#: Shells: (threshold, opacity).  Reflectivity in dBZ; without reflectivity, precipitation water in g/m³.
SHELLS_DBZ = ((30.0, 0.15), (45.0, 0.40), (60.0, 0.90))
SHELLS_WATER = ((0.1, 0.15), (1.0, 0.40), (3.0, 0.90))
WATER_RGB = ((0.35, 0.62, 0.95), (0.20, 0.42, 0.95), (0.62, 0.40, 1.00))
#: The cloud's look: extinction per metre for each g/m³, the ramp it comes in over, the share drawn.
CLOUD_EXTINCTION = 0.020
CLOUD_RAMP = (0.002, 0.1)
CLOUD_SEE_THROUGH = 0.06
#: Lighting: a dim, cool key light from the west-south-west, and the sky.
SUN_AZIMUTH = 250.0
SUN_ELEVATION = 20.0
SUN_RGB = np.array((0.78, 0.84, 1.00), np.float32) * 1.5
SKY_RGB = np.array((0.40, 0.46, 0.60), np.float32) * 0.22
CLOUD_FILL = np.array((0.035, 0.042, 0.060), np.float32)
GROUND_RGB = np.array((0.010, 0.011, 0.013), np.float32)
GRID_RGB = np.array((0.10, 0.14, 0.20), np.float32)
GRID_GLOW = 0.55
SKY_ZENITH = np.array((0.0, 0.0, 0.0), np.float32)
SKY_HORIZON = np.array((0.010, 0.012, 0.018), np.float32)
EXPOSURE = 0.9
#: The camera: compass bearing from the storm to the camera, height above the horizon, field of view.
BEARING = 150.0
ELEVATION = 8.0
FOV = 35.0
#: Where the drawn storm may reach in the picture (-1..1 across, bottom to top): clear of the words above and below.
FRAME_X = 0.90
FRAME_TOP = 0.60
FRAME_BOTTOM = -0.50
#: The cut face is a cross-section, not a lit surface: drawn at nearly its table colour, nearly solid (the same
#: opacity for every band, said on the picture), so its bands read as the radar's vertical section does.
CUT_FACE_SHADE = 0.92
CUT_FACE_OPACITY = 0.85
#: The highest visible top is marked to the nearest 250 m (no finer than the levels near the top).
TOP_ROUND_M = 250.0
#: How long a request may take (estimated beforehand from the measured costs below), seconds.
STILL_LIMIT_S = 900.0
ORBIT_LIMIT_S = 3600.0
#: Costs on one core (:func:`estimate_seconds`), set above the measured ones so the estimate errs long: finding
#: the core, per million cells of the domain (0.4 s measured at 250 m); reading and building the box, per million
#: cells read (0.5 s); drawing, per million rays at a scene of a million cells (4 to 11 s, a hurricane at 4 km to
#: a supercell at 250 m).
PLAN_S_PER_MCELL = 0.6
BUILD_S_PER_MCELL = 2.0
DRAW_S_PER_MRAY = 10.0
NEAR_EDGE_FADE_M = 5_000.0


class Refused(ValueError):
    """A 3-D picture refused before it starts (too long, or nothing to draw), in plain words."""


# ----------------------------------------------------------------------------------------------- small maths
def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smooth121(a: np.ndarray) -> np.ndarray:
    """1-2-1 in both horizontal directions (edges repeat)."""

    p = np.pad(a, 1, mode="edge")
    b = 0.25 * p[:, :-2] + 0.5 * p[:, 1:-1] + 0.25 * p[:, 2:]
    return (0.25 * b[:-2] + 0.5 * b[1:-1] + 0.25 * b[2:]).astype(np.float32)


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box mean over (2r+1)² cells (edges repeat)."""

    if r <= 0:
        return np.asarray(a, np.float32)
    out = np.asarray(a, np.float64)
    for axis in (0, 1):
        pad = [(r + 1, r) if ax == axis else (0, 0) for ax in range(2)]
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        n = c.shape[axis]
        out = (np.take(c, np.arange(2 * r + 1, n), axis=axis) - np.take(c, np.arange(0, n - 2 * r - 1), axis=axis))
        out = out / (2 * r + 1)
    return out.astype(np.float32)


def block_mean(a: np.ndarray, s: int) -> np.ndarray:
    """Mean over s x s blocks of the last two axes (the array is already a whole number of blocks)."""

    if s == 1:
        return np.asarray(a, np.float32)
    ny, nx = a.shape[-2] // s, a.shape[-1] // s
    b = np.asarray(a, np.float32)[..., :ny * s, :nx * s]
    return b.reshape(b.shape[:-2] + (ny, s, nx, s)).mean(axis=(-3, -1), dtype=np.float32)


def shift_bilinear(a: np.ndarray, dj: float, di: float) -> np.ndarray:
    """out[j, i] = a(j + dj, i + di), bilinear, zero outside."""

    ny, nx = a.shape
    rows = np.arange(ny, dtype=np.float64)[:, None] + dj
    cols = np.arange(nx, dtype=np.float64)[None, :] + di
    return P.bilinear(a, np.broadcast_to(rows, (ny, nx)), np.broadcast_to(cols, (ny, nx)), cval=0.0)


def sun_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector toward a light at a compass azimuth (0 north, 90 east) and an elevation."""

    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def compass_name(bearing_deg: float) -> str:
    names = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
    return names[int(round((bearing_deg % 360.0) / 22.5)) % 16]


def nice_scale(raw: float) -> float:
    """The 1-2-5 step nearest ``raw`` (at least 1)."""

    if raw <= 1.0:
        return 1.0
    exponent = math.floor(math.log10(raw))
    options = [f * 10.0 ** e for e in (exponent, exponent + 1) for f in (1.0, 2.0, 5.0)]
    return float(min(options, key=lambda c: abs(math.log(c / raw))))


def height_stretch(width_m: float, top_m: float) -> float:
    """How many times heights are stretched: 1 (true scale) for a box up to 150 km across; a wider one (a
    hurricane) by the 1-2-5 factor that draws the storm about 0.3 times as tall as it is wide."""

    return 1.0 if width_m <= 150_000.0 else nice_scale(0.3 * width_m / max(top_m, 1000.0))


def tonemap(radiance: np.ndarray, exposure: float = EXPOSURE) -> np.ndarray:
    """ACES filmic curve to sRGB, 0..1."""

    x = radiance * exposure
    lin = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055).astype(np.float32)


def inverse_tonemap(display: np.ndarray, exposure: float = EXPOSURE) -> np.ndarray:
    """The radiance :func:`tonemap` draws as this display colour (so a shell keeps its table colour)."""

    d = np.clip(np.asarray(display, np.float64), 0.0, 0.995)
    lin = np.where(d <= 0.04045, d / 12.92, ((d + 0.055) / 1.055) ** 2.4)
    a, b = 2.51 - 2.43 * lin, 0.03 - 0.59 * lin
    x = (-b + np.sqrt(b * b + 4.0 * a * 0.14 * lin)) / (2.0 * a)
    return (x / exposure).astype(np.float32)


def nws_rgb(value: float) -> tuple[float, float, float]:
    """The NWS table colour of a dBZ value (the quick-looks' :data:`.style.NWS_REFLECTIVITY`)."""

    colours = [S.hex_rgb(c) for _, c in S.NWS_REFLECTIVITY]
    return colours[int(np.clip(math.floor((value - 5.0) / 5.0), 0, len(colours) - 1))]


# ----------------------------------------------------------------------------------------------- the plan
@dataclass
class Plan:
    """What one 3-D picture shows: the box, how it is sampled, the cut, and the words for each."""

    kind: str                     # "thunderstorm" or "hurricane"
    i0: int
    i1: int
    j0: int
    j1: int
    stride: int
    nk: int
    top_m: float                  # highest cloud in the box, m above ground
    zscale: float
    cut_xy: tuple[float, float]   # m, domain coordinates (south-west corner at 0, 0)
    cut_words: str
    field: str                    # "reflectivity" or "water"
    operator: str
    shells: tuple[tuple[float, float], ...]
    cells_read: int
    voxels: int
    domain_edges: list[str] = field(default_factory=list)
    domain_cells: int = 0
    #: said on the picture when a stronger core lies in the strip along the domain's edges, where no cut is made
    cut_note: str = ""

    @property
    def voxel_m(self) -> float:
        return self.stride * self.dx_m

    dx_m: float = 0.0


def _rho_level(f: QuickFrame, k: int, js: slice, is_: slice) -> np.ndarray:
    """Air density (kg/m³) on level k over a crop: the frame's rho, else CM1's from p, θ and qv."""

    if f.has("rho"):
        return np.asarray(f.view("rho")[k, js, is_], np.float32)
    from .satellite import CP, P00, RD, RV

    if not f.has("prs"):
        raise FrameUnreadable(f"{f.directory.name} has no air density and no pressure to make it")
    p = np.asarray(f.view("prs")[k, js, is_], np.float32)
    if f.has("th"):
        theta = np.asarray(f.view("th")[k, js, is_], np.float32)
    else:
        th0 = f.base_column("th0")
        if th0 is None or not f.has("th3d"):
            raise FrameUnreadable(f"{f.directory.name} has no theta to make the air density")
        theta = np.asarray(f.view("th3d")[k, js, is_], np.float32) + np.float32(th0[k])
    t = theta * (p / np.float32(P00)) ** np.float32(RD / CP)
    qv = np.asarray(f.species("qv", (k, js, is_)), np.float32) if f.has_species("qv") else np.zeros_like(p)
    return (p / (np.float32(RD) * t * (1.0 + np.maximum(qv, 0.0) * np.float32(RV / RD)))).astype(np.float32)


def _mass_level(f: QuickFrame, names: Sequence[str], k: int, js: slice, is_: slice) -> np.ndarray | None:
    """Mixing ratios summed over ``names`` (those the frame has) on level k, kg/kg, or None."""

    have = [n for n in names if f.has_species(n)]
    if not have:
        return None
    rows = f.species_names("q3d")
    q = f.view("q3d")
    total = np.zeros((js.stop - js.start, is_.stop - is_.start), np.float32)
    for n in have:
        total += np.maximum(np.asarray(q[rows.index(n), k, js, is_], np.float32), 0.0)
    return total


CLOUD_SPECIES = ("qc", "qi", "qs")
WATER_SPECIES = ("qr", "qg", "qh")


def _reflectivity_kind(f: QuickFrame) -> tuple[str, str]:
    """("carrier", words), ("morrison", words) or ("", why not)."""

    if f.has("dbz") and tuple(f.frame.carriers["dbz"].shape) == (f.nz, f.ny, f.nx):
        component = str(f.config.get("microphysics") or "")
        scheme = R.PLAIN_SCHEME.get(component, component or "the scheme")
        return "carrier", f"{scheme}'s own reflectivity"
    if str(f.config.get("microphysics") or "") in R.MORRISON and R._calc_function()[0] is not None:  # noqa: SLF001
        return "morrison", "CM1's calc_refl10cm (Morrison, 10 cm)"
    return "", "no reflectivity for this microphysics"


def _dbz_band(f: QuickFrame, kind: str, nk: int, js: slice, is_: slice) -> np.ndarray:
    """Reflectivity (dBZ) on levels 0..nk-1 of a crop."""

    if kind == "carrier":
        return np.asarray(f.view("dbz")[:nk, js, is_], np.float32)
    calc = R._calc_function()[0]  # noqa: SLF001
    return R.morrison_volume(f, calc, crop=(js, is_))[:nk]


def _band_rows(f: QuickFrame, width: int, stride: int, cells: float = 2.0e6) -> int:
    """Rows per band (a whole number of blocks) so one band holds about ``cells`` cells of every level."""

    rows = max(stride, int(cells // max(1, width * f.nz)) // stride * stride)
    return rows


def _core_reflectivity(f: QuickFrame, kind: str, step: int) -> tuple[np.ndarray, float]:
    """The strongest reflectivity (linear Z) 1 to 5 km above ground in every ``step``-th column."""

    js, is_ = slice(0, f.ny, step), slice(0, f.nx, step)
    if kind == "carrier":
        levels = [k for k in range(f.nz) if 1000.0 <= f.zh[k] <= 5000.0] or [0]
        z = np.zeros(((f.ny + step - 1) // step, (f.nx + step - 1) // step), np.float32)
        for k in levels:
            z = np.maximum(z, np.power(10.0, np.asarray(f.view("dbz")[k, js, is_], np.float32) / 10.0))
        return z, float(10.0 * np.log10(max(float(z.max()), 1e-6)))
    calc = R._calc_function()[0]  # noqa: SLF001
    nyc, nxc = len(range(0, f.ny, step)), len(range(0, f.nx, step))
    band = max(1, int(2.0e6 // max(1, nxc * f.nz)))
    levels = np.array([1000.0 <= z <= 5000.0 for z in f.zh])
    if not levels.any():
        levels[0] = True
    out = np.zeros((nyc, nxc), np.float32)
    for b0 in range(0, nyc, band):
        b1 = min(nyc, b0 + band)
        dbz = R.morrison_volume(f, calc, crop=(slice(b0 * step, b1 * step, step), slice(0, f.nx, step)))
        out[b0:b1] = np.power(10.0, dbz[levels].max(axis=0) / 10.0)
    return out, float(10.0 * np.log10(max(float(out.max()), 1e-6)))


def no_echo(field_kind: str, peak: float, where: str) -> "P.NotDrawn":
    """The plain refusal for a time with nothing up to the lowest shell yet (early in a run, or a cloud with no
    rain): the picture would show no shell and no cut face, only cloud."""

    if field_kind == "reflectivity":
        return P.NotDrawn(f"No echo of {SHELLS_DBZ[0][0]:g} dBZ or more yet at this time (the strongest {where} is "
                          f"{S.fmt_num(peak)} dBZ): there is no rain or hail to cut open. Draw a later saved time.")
    return P.NotDrawn(f"No rain, graupel or hail of {SHELLS_WATER[0][0]:g} g/m³ or more yet at this time (the most "
                      f"{where} is {S.fmt_num(peak, 2)} g/m³): there is nothing to cut open. Draw a later saved time.")


def plan_for(L: "P.Looks") -> Plan:
    """The box, sampling, cut and field of the 3-D picture of this frame (reads only what it needs)."""

    f = L.f
    hurricane = L.hurricane
    kind_r, operator = _reflectivity_kind(f)
    field_kind = "reflectivity" if kind_r else "water"
    if not (f.has("q3d") and any(f.has_species(n) for n in ("qc", "qi"))):
        raise P.NotDrawn("This run saved no cloud water or ice, so there is no cloud to draw.")
    step = max(1, int(math.ceil(math.sqrt(f.nx * f.ny / 250_000.0))))
    xs = (np.arange(0, f.nx, step) + 0.5) * f.dx
    ys = (np.arange(0, f.ny, step) + 0.5) * f.dy
    cut_note = ""
    if hurricane:
        p0 = f.pressure_level(0)
        if p0 is None:
            raise P.NotDrawn("This hurricane run's frames carry no pressure, so its centre cannot be found.")
        sm = box_blur(p0, max(1, int(round(20_000.0 / f.dx))))
        n = max(1, min(P.EDGE_CELLS * 3, min(sm.shape) // 4))
        inner = sm[n:-n, n:-n]
        jj, ii = np.unravel_index(int(np.argmin(inner)), inner.shape)
        cut = ((ii + n + 0.5) * f.dx, (jj + n + 0.5) * f.dy)
        half = min(HURRICANE_HALF_M, 0.5 * min(f.extent))
        cut_words = "cut open through the centre (the lowest pressure at the lowest level)"
    else:
        trim = max(1000.0, 8 * f.dx)
        if kind_r:
            zcore, peak = _core_reflectivity(f, kind_r, step)
            words_field = "reflectivity"
            if peak < SHELLS_DBZ[0][0]:
                raise no_echo("reflectivity", peak, "1 to 5 km up")
        else:
            zcore = None
            words_field = "precipitation water"
        if zcore is None:
            zcore = np.zeros((ys.size, xs.size), np.float32)
            peak = 0.0
            whole = (slice(0, f.ny), slice(0, f.nx))
            for k in [k for k in range(f.nz) if 1000.0 <= f.zh[k] <= 5000.0] or [0]:
                w = _mass_level(f, WATER_SPECIES + ("qs",), k, *whole)
                if w is not None:
                    zcore = np.maximum(zcore, w[::step, ::step])
                shells_w = _mass_level(f, WATER_SPECIES, k, *whole)
                if shells_w is not None:
                    peak = max(peak, float(np.max(shells_w * _rho_level(f, k, *whole))) * 1000.0)
            if peak < SHELLS_WATER[0][0]:
                raise no_echo("water", peak, "1 to 5 km up")
        r = max(1, int(round(500.0 / (f.dx * step))))
        sm = box_blur(zcore, r)
        mask = ((xs[None, :] > trim) & (xs[None, :] < f.extent[0] - trim)
                & (ys[:, None] > trim) & (ys[:, None] < f.extent[1] - trim))
        if float(np.where(mask, sm, 0.0).max()) <= 0.0:
            raise P.NotDrawn("No storm yet at this time: there is no rain or ice to cut open.")
        jj, ii = np.unravel_index(int(np.argmax(np.where(mask, sm, -np.inf))), sm.shape)
        cut = (float(xs[ii]), float(ys[jj]))
        half = THUNDERSTORM_HALF_M
        cut_words = f"cut open through the {words_field} core"
        if float(sm.max()) > 1.26 * float(sm[jj, ii]):
            cut_note = (f"the cut is through the strongest core at least {S.fmt_km(trim)} inside the domain; a "
                        "stronger one lies nearer its edge")
    i0 = int(np.clip(math.floor((cut[0] - half) / f.dx), 0, f.nx - 2))
    i1 = int(np.clip(math.ceil((cut[0] + half) / f.dx), i0 + 2, f.nx))
    j0 = int(np.clip(math.floor((cut[1] - half) / f.dy), 0, f.ny - 2))
    j1 = int(np.clip(math.ceil((cut[1] + half) / f.dy), j0 + 2, f.ny))
    # the highest cloud in the box, scanning down from the model's top over a coarse grid of columns
    sj, si = slice(j0, j1, step), slice(i0, i1, step)
    top = 0.0
    names = f.species_names("q3d")
    q = f.view("q3d")
    have = [names.index(n) for n in CLOUD_SPECIES if n in names]
    for k in range(f.nz - 1, -1, -1):
        level = sum(np.asarray(q[idx, k, sj, si], np.float32) for idx in have)
        if float(np.max(level)) * 1.2 * 1000.0 > 0.02:
            top = float(f.zf[min(k + 1, f.nz)])
            break
    if top <= 0.0:
        raise P.NotDrawn("No cloud yet at this time: there is nothing to draw in 3-D.")
    ztop = min(float(f.zf[-1]), top + 1000.0)
    nk = int(min(f.nz, max(2, np.searchsorted(f.zf, ztop, side="left"))))
    zscale = height_stretch(max((i1 - i0) * f.dx, (j1 - j0) * f.dy), top)
    cells = (i1 - i0) * (j1 - j0) * nk
    stride = max(1, int(math.ceil(math.sqrt(cells / VOXEL_BUDGET))))
    nxs, nys = max(2, (i1 - i0) // stride), max(2, (j1 - j0) // stride)
    i1, j1 = i0 + nxs * stride, j0 + nys * stride
    edges = []
    near = max(1000.0, 8 * f.dx) + 2000.0
    if i0 * f.dx <= near:
        edges.append("west")
    if (f.nx - i1) * f.dx <= near:
        edges.append("east")
    if j0 * f.dy <= near:
        edges.append("south")
    if (f.ny - j1) * f.dy <= near:
        edges.append("north")
    return Plan(kind="hurricane" if hurricane else "thunderstorm", i0=i0, i1=i1, j0=j0, j1=j1, stride=stride, nk=nk,
                top_m=top, zscale=zscale, cut_xy=(float(cut[0]), float(cut[1])), cut_words=cut_words,
                field=field_kind, operator=operator if kind_r else "", shells=SHELLS_DBZ if kind_r else SHELLS_WATER,
                cells_read=(i1 - i0) * (j1 - j0) * nk, voxels=nxs * nys * nk, domain_edges=edges, dx_m=f.dx,
                domain_cells=f.nx * f.ny * f.nz, cut_note=cut_note)


def estimate_seconds(plan: Plan, size: tuple[int, int], pictures: int = 1) -> float:
    """A plain estimate of a request's time on one core, from the measured costs."""

    build = PLAN_S_PER_MCELL * plan.domain_cells / 1e6 + BUILD_S_PER_MCELL * plan.cells_read / 1e6
    rays = size[0] * size[1] / 1e6
    depth = (plan.voxels / 1e6) ** (1.0 / 3.0)
    return build + pictures * DRAW_S_PER_MRAY * rays * max(depth, 0.3)


# ----------------------------------------------------------------------------------------------- the scene
@dataclass
class Scene:
    x0: float
    y0: float
    dxr: float
    dyr: float
    nk: int
    ny: int
    nx: int
    zscale: float
    zh_w: np.ndarray
    zf_w: np.ndarray
    vol: np.ndarray        # (nk*ny*nx, 6): drawn cloud extinction, sun shade, sun spread, sky shade, field, fade
    grad: np.ndarray       # (nk*ny*nx, 3): gradient of the field per world metre
    ktable: np.ndarray
    kstep: float
    occ: np.ndarray        # (ncz, ncy, ncx) bool
    occ_cell: float
    ground_amb: np.ndarray
    sun: np.ndarray
    step: float
    shells: tuple[tuple[float, float, tuple[float, float, float]], ...]
    grid_m: float
    grid_offset: tuple[float, float]
    plan: Plan
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def lo(self) -> np.ndarray:
        return np.array([self.x0, self.y0, 0.0])

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.x0 + self.nx * self.dxr, self.y0 + self.ny * self.dyr, float(self.zf_w[self.nk])])


def sun_optical_depth(ext: np.ndarray, zh_w: np.ndarray, zf_w: np.ndarray, dxr: float, dyr: float,
                      sun: np.ndarray) -> np.ndarray:
    """Optical depth toward the light at every cell centre, swept from the top down.

    Between two level centres the sweep takes sub-steps no longer than one cell sideways: each displaces the
    running depth toward the light and adds the extinction there (linear between the levels) times the slant
    path."""

    nk = ext.shape[0]
    sin_e = max(float(sun[2]), 1e-3)
    horiz = math.hypot(float(sun[0]), float(sun[1]))
    cot = horiz / sin_e
    ux, uy = (float(sun[0]) / horiz, float(sun[1]) / horiz) if horiz > 0 else (0.0, 0.0)
    tau = np.empty_like(ext, dtype=np.float32)
    tau[nk - 1] = ext[nk - 1] * (zf_w[nk] - zh_w[nk - 1]) / sin_e
    run = tau[nk - 1].copy()
    for k in range(nk - 2, -1, -1):
        dz = float(zh_w[k + 1] - zh_w[k])
        n_sub = max(1, int(math.ceil(dz * cot / min(dxr, dyr))))
        sub = dz / n_sub
        di, dj = sub * cot * ux / dxr, sub * cot * uy / dyr
        for m in range(n_sub):
            f_lo = (m + 0.5) / n_sub
            run = shift_bilinear(run, dj, di) + ((1.0 - f_lo) * ext[k + 1] + f_lo * ext[k]) * (sub / sin_e)
        tau[k] = run
    return tau


def build_scene(L: "P.Looks", plan: Plan, *, log: Callable[[str], None] = lambda _m: None) -> Scene:
    """Read the box, average it to the plan's cells, light it, and lay out the occupancy grid."""

    begun = _time.perf_counter()
    f = L.f
    s, nk = plan.stride, plan.nk
    js, is_ = slice(plan.j0, plan.j1), slice(plan.i0, plan.i1)
    nys, nxs = (plan.j1 - plan.j0) // s, (plan.i1 - plan.i0) // s
    cloud = np.zeros((nk, nys, nxs), np.float32)
    water = np.zeros_like(cloud)
    for k in range(nk):
        rho = _rho_level(f, k, js, is_) * np.float32(1000.0)        # g/m³ per kg/kg
        c = _mass_level(f, CLOUD_SPECIES, k, js, is_)
        if c is not None:
            cloud[k] = block_mean(c * rho, s)
        w = _mass_level(f, WATER_SPECIES, k, js, is_)
        if w is not None:
            water[k] = block_mean(w * rho, s)
    peak = None
    if plan.field == "reflectivity":
        kind, _words = _reflectivity_kind(f)
        fld = np.full((nk, nys, nxs), -35.0, np.float32)
        rows = _band_rows(f, plan.i1 - plan.i0, s)
        top = -35.0
        for b0 in range(plan.j0, plan.j1, rows):
            b1 = min(plan.j1, b0 + rows)
            band = _dbz_band(f, kind, nk, slice(b0, b1), is_)
            top = max(top, float(band.max()))
            linear = np.power(10.0, np.maximum(band, -35.0) / 10.0, dtype=np.float32)
            fld[:, (b0 - plan.j0) // s:(b1 - plan.j0) // s] = 10.0 * np.log10(np.maximum(block_mean(linear, s),
                                                                                         1e-4))
        peak = top
    else:
        fld = water.copy()
        peak = float(water.max())
    # nothing up to the lowest shell in the box (a hurricane spinning up; a box whose echo is all at its edge)
    if peak < min(t for t, _ in plan.shells):
        raise no_echo(plan.field, peak, "in the drawn box")
    # 1-2-1 across three cells (said on the picture); vertically too where levels are coarser than a cell
    for arr in (cloud, water, fld):
        for k in range(nk):
            arr[k] = smooth121(arr[k])
    dxr, dyr = f.dx * s, f.dy * s
    dz = np.diff(f.zh[:nk])
    if nk > 2:
        coarse = np.maximum(dz[:-1], dz[1:]) > dxr
        if coarse.any():
            for arr in (cloud, water, fld):
                mixed = 0.25 * arr[:-2] + 0.5 * arr[1:-1] + 0.25 * arr[2:]
                arr[1:-1] = np.where(coarse[:, None, None], mixed, arr[1:-1])
    x0, y0 = plan.i0 * f.dx, plan.j0 * f.dy
    zf_w = (f.zf[:nk + 1] * plan.zscale).astype(np.float64)
    zh_w = (f.zh[:nk] * plan.zscale).astype(np.float64)
    xs = x0 + (np.arange(nxs) + 0.5) * dxr
    ys = y0 + (np.arange(nys) + 0.5) * dyr
    # Where the cloud leaves the model's domain, or the drawn box, the drawing fades out (the lighting only at
    # the domain's sides: beyond them there is nothing).
    fade_m = min(NEAR_EDGE_FADE_M if plan.kind == "thunderstorm" else 0.1 * nxs * dxr, 0.15 * min(nxs * dxr,
                                                                                                    nys * dyr))
    dist = {"west": xs[None, :] - x0, "east": x0 + nxs * dxr - xs[None, :],
            "south": ys[:, None] - y0, "north": y0 + nys * dyr - ys[:, None]}
    edge_cloud = {"west": cloud[:, :, :2].max(), "east": cloud[:, :, -2:].max(),
                  "south": cloud[:, :2, :].max(), "north": cloud[:, -2:, :].max()}
    domain_fade = np.ones((nys, nxs), np.float32)
    box_fade = np.ones((nys, nxs), np.float32)
    domain_sides, box_sides = [], []
    for side in ("west", "east", "south", "north"):
        if edge_cloud[side] <= 0.05:
            continue
        ramp = smoothstep(0.0, fade_m, dist[side]).astype(np.float32)
        if side in plan.domain_edges:
            domain_fade = domain_fade * ramp
            domain_sides.append(side)
        else:
            box_fade = box_fade * ramp
            box_sides.append(side)
    sun = sun_vector(SUN_AZIMUTH, SUN_ELEVATION)
    ext_light = (CLOUD_EXTINCTION * cloud * smoothstep(*CLOUD_RAMP, cloud)
                 + 0.0008 * water * smoothstep(0.05, 0.5, water)) * domain_fade[None]
    tau_sun = sun_optical_depth(ext_light, zh_w, zf_w, dxr, dyr, sun)
    tau_up = np.empty_like(ext_light)
    tau_up[nk - 1] = ext_light[nk - 1] * (zf_w[nk] - zh_w[nk - 1])
    for k in range(nk - 2, -1, -1):
        tau_up[k] = tau_up[k + 1] + 0.5 * (ext_light[k + 1] + ext_light[k]) * (zh_w[k + 1] - zh_w[k])
    blur_r = max(1, int(round(3000.0 / dxr)))
    ao_r = max(1, int(round(300.0 / dxr)))
    for k in range(nk):
        tau_up[k] = box_blur(tau_up[k], blur_r)
    near = [box_blur(ext_light[k], ao_r) for k in range(nk)]
    for k in range(nk):
        below = near[k - 1] if k > 0 else near[k]
        above = near[k + 1] if k + 1 < nk else near[k]
        tau_up[k] += 0.6 * 600.0 * (0.25 * below + 0.5 * near[k] + 0.25 * above)
    del near
    wxy = domain_fade * box_fade
    ext_draw = CLOUD_EXTINCTION * CLOUD_SEE_THROUGH * cloud * smoothstep(*CLOUD_RAMP, cloud)
    n = nk * nys * nxs
    vol = np.empty((n, 6), np.float32)
    vol[:, 0] = ext_draw.reshape(-1)
    vol[:, 1] = np.exp(-tau_sun).reshape(-1)
    vol[:, 2] = (0.25 * np.exp(-0.3 * tau_sun) + 0.10 * np.exp(-0.1 * tau_sun)).reshape(-1)
    vol[:, 3] = (0.07 + 0.93 * np.exp(-0.3 * tau_up)).reshape(-1)
    vol[:, 4] = fld.reshape(-1)
    vol[:, 5] = np.broadcast_to(wxy[None], (nk, nys, nxs)).reshape(-1)
    ground_amb = (0.10 + 0.90 * np.exp(-0.3 * tau_up[0])).astype(np.float32)
    gz = np.gradient(fld, zh_w, axis=0) if nk > 1 else np.zeros_like(fld)
    gy = np.gradient(fld, dyr, axis=1) if nys > 1 else np.zeros_like(fld)
    gx = np.gradient(fld, dxr, axis=2) if nxs > 1 else np.zeros_like(fld)
    grad = np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=1).astype(np.float32)
    del gx, gy, gz, tau_sun, tau_up
    # world height -> fractional level
    dz_min = float(np.min(np.diff(zh_w))) if nk > 1 else float(zf_w[-1])
    kstep = max(dz_min / 8.0, float(zf_w[nk]) / 200_000.0)
    ktable = np.interp(np.arange(0.0, float(zf_w[nk]) + 2 * kstep, kstep), zh_w,
                       np.arange(nk, dtype=np.float64)).astype(np.float32)
    # occupancy: blocks of 4 x 4 cells, world-uniform in height, grown by one block
    step = 0.6 * min(dxr, dyr)
    block = 4
    cell = block * min(dxr, dyr)
    ncx, ncy = math.ceil(nxs / block), math.ceil(nys / block)
    ncz = max(1, math.ceil(float(zf_w[nk]) / cell))
    occ = np.zeros((ncz, ncy, ncx), bool)
    lowest = min(t for t, _ in plan.shells)
    margin = 3.0 if plan.field == "reflectivity" else 0.5 * lowest
    visible = wxy > 1e-3
    for k in range(nk):
        layer = ((ext_draw[k] * step > 1e-5) | (fld[k] >= lowest - margin)) & visible
        if not layer.any():
            continue
        pad = np.zeros((ncy * block, ncx * block), bool)
        pad[:nys, :nxs] = layer
        coarse = pad.reshape(ncy, block, ncx, block).any(axis=(1, 3))
        c0 = int(zf_w[k] // cell)
        c1 = min(int(zf_w[k + 1] // cell), ncz - 1)
        occ[c0:c1 + 1] |= coarse[None]
    for axis in range(3):
        grown = occ.copy()
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        grown[tuple(a)] |= occ[tuple(b)]
        grown[tuple(b)] |= occ[tuple(a)]
        occ = grown
    # what is drawn: its bounding corners frame the camera; its tallest column labels the top
    seen = (ext_draw * step > 0.002) | (fld >= lowest)
    seen &= visible[None]
    if not seen.any():
        raise P.NotDrawn("Nothing in the box is thick enough to draw yet.")
    kk = np.nonzero(seen.any(axis=(1, 2)))[0]
    jj = np.nonzero(seen.any(axis=(0, 2)))[0]
    ii = np.nonzero(seen.any(axis=(0, 1)))[0]
    content = (np.array([x0 + ii[0] * dxr, y0 + jj[0] * dyr, 0.0]),
               np.array([x0 + (ii[-1] + 1) * dxr, y0 + (jj[-1] + 1) * dyr, float(zf_w[kk[-1] + 1])]))
    tops_k = np.where(seen.any(axis=0), nk - 1 - np.argmax(seen[::-1], axis=0), -1)
    shells = []
    for n_shell, (thr, opacity) in enumerate(plan.shells):
        rgb = nws_rgb(thr) if plan.field == "reflectivity" else WATER_RGB[n_shell]
        shells.append((float(thr), float(opacity), tuple(float(c) for c in rgb)))
    span = max(nxs * dxr, nys * dyr)
    grid_m = 10_000.0 if span > 60_000.0 else 5_000.0
    if span > 400_000.0:
        grid_m = 50_000.0
    meta = {"stride": s, "voxel_m": dxr, "cells": n, "domain_fade": domain_sides, "box_fade": box_sides,
            "fade_m": fade_m, "content": content, "tops_k": tops_k, "peak": peak, "model_nz": int(f.nz),
            "model_top_m": float(f.zf[-1]), "build_s": round(_time.perf_counter() - begun, 2)}
    log(f"3-D scene {nxs} x {nys} x {nk} cells of {dxr:g} m (stride {s}) in {meta['build_s']} s")
    return Scene(x0=x0, y0=y0, dxr=dxr, dyr=dyr, nk=nk, ny=nys, nx=nxs, zscale=plan.zscale, zh_w=zh_w, zf_w=zf_w,
                 vol=vol, grad=grad, ktable=ktable, kstep=kstep, occ=occ, occ_cell=cell, ground_amb=ground_amb,
                 sun=sun, step=step, shells=tuple(shells), grid_m=grid_m,
                 grid_offset=(f.domainlocx, f.domainlocy), plan=plan, meta=meta)


# ----------------------------------------------------------------------------------------------- the camera
@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    fov_deg: float
    width: int
    height: int
    clip: tuple[np.ndarray, np.ndarray] | None = None   # (point, unit normal): draw where (p - point).n >= 0

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fwd = self.target - self.eye
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        return fwd, right, np.cross(right, fwd)

    @property
    def tan_half(self) -> float:
        return math.tan(math.radians(self.fov_deg) / 2.0)

    @property
    def heading_deg(self) -> float:
        fwd, _, _ = self.basis()
        return math.degrees(math.atan2(fwd[0], fwd[1])) % 360.0

    def rays(self, pix: np.ndarray) -> np.ndarray:
        fwd, right, up = self.basis()
        px = (pix % self.width).astype(np.float64)
        py = (pix // self.width).astype(np.float64)
        xn = (2.0 * (px + 0.5) / self.width - 1.0) * self.tan_half * self.width / self.height
        yn = (1.0 - 2.0 * (py + 0.5) / self.height) * self.tan_half
        d = fwd[None, :] + xn[:, None] * right[None, :] + yn[:, None] * up[None, :]
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def ray_at(self, px: float, py: float) -> np.ndarray:
        fwd, right, up = self.basis()
        xn = (2.0 * (px + 0.5) / self.width - 1.0) * self.tan_half * self.width / self.height
        yn = (1.0 - 2.0 * (py + 0.5) / self.height) * self.tan_half
        d = fwd + xn * right + yn * up
        return d / np.linalg.norm(d)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(px, py, depth) of world points; pixel centres are whole numbers."""

        fwd, right, up = self.basis()
        v = np.atleast_2d(points) - self.eye[None, :]
        zc = v @ fwd
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = (v @ right) / zc / (self.tan_half * self.width / self.height)
            yn = (v @ up) / zc / self.tan_half
        return (xn + 1.0) * self.width / 2.0 - 0.5, (1.0 - yn) * self.height / 2.0 - 0.5, zc

    def scaled(self, width: int, height: int) -> "Camera":
        return Camera(self.eye, self.target, self.fov_deg, width, height, self.clip)


def orbit_camera(target: np.ndarray, bearing_deg: float, elevation_deg: float, distance: float, width: int,
                 height: int) -> Camera:
    b, e = math.radians(bearing_deg), math.radians(elevation_deg)
    eye = target + distance * np.array([math.sin(b) * math.cos(e), math.cos(b) * math.cos(e), math.sin(e)])
    return Camera(eye, np.asarray(target, np.float64), FOV, width, height)


def _corners(scene: Scene, bearing: float | None = None) -> np.ndarray:
    """Points on the box around what is drawn; with a bearing, only those beyond the cut (the drawn half)."""

    lo, hi = scene.meta["content"]
    g = np.linspace(0.0, 1.0, 7)
    pts = np.array([[lo[0] + a * (hi[0] - lo[0]), lo[1] + b * (hi[1] - lo[1]), lo[2] + c * (hi[2] - lo[2])]
                    for a in g for b in g for c in (0.0, 1.0)])
    if bearing is None:
        return pts
    rad = math.radians(bearing)
    normal = -np.array([math.sin(rad), math.cos(rad), 0.0])     # from the camera toward the storm, level
    cut = scene.plan.cut_xy
    beyond = (pts - np.array([cut[0], cut[1], 0.0])[None, :]) @ normal >= -scene.dxr
    if beyond.sum() < 4:
        return pts
    # The cut face is the drawn half's side nearest the camera, so its ends (where the cut meets the box) are
    # framed too, not only the box's points beyond it.
    along = np.array([-normal[1], normal[0]])
    reach = []
    for axis in (0, 1):
        if abs(along[axis]) > 1e-9:
            reach += [(lo[axis] - cut[axis]) / along[axis], (hi[axis] - cut[axis]) / along[axis]]
    reach.sort()
    face = []
    if len(reach) >= 2:
        for s in np.linspace(reach[len(reach) // 2 - 1], reach[len(reach) // 2], 9):
            x, y = cut[0] + s * along[0], cut[1] + s * along[1]
            if lo[0] - 1.0 <= x <= hi[0] + 1.0 and lo[1] - 1.0 <= y <= hi[1] + 1.0:
                face += [[x, y, lo[2]], [x, y, hi[2]]]
    return np.vstack([pts[beyond]] + ([np.array(face)] if face else []))


def target_of(scene: Scene) -> np.ndarray:
    lo, hi = scene.meta["content"]
    return np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.3 * hi[2]])


def fit_distance(scene: Scene, bearing: float, width: int, height: int) -> float:
    """The nearest orbit distance at which the drawn content fits the picture (with room for the words)."""

    pts = _corners(scene, bearing)
    target = target_of(scene)

    def fits(d: float) -> bool:
        cam = orbit_camera(target, bearing, ELEVATION, d, width, height)
        px, py, zc = cam.project(pts)
        xn = (px + 0.5) * 2.0 / width - 1.0
        yn = 1.0 - (py + 0.5) * 2.0 / height
        return bool(np.all(zc > 0) and np.all(np.abs(xn) <= FRAME_X) and np.all(yn <= FRAME_TOP)
                    and np.all(yn >= FRAME_BOTTOM))

    extent = float(np.max(np.linalg.norm(pts - target[None, :], axis=1))) + 1.0
    lo, hi = 1e-3 * extent, extent
    while not fits(hi) and hi < 1e4 * extent:
        hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return hi


def camera_for(scene: Scene, bearing: float, width: int, height: int, distance: float | None = None) -> Camera:
    """The camera on ``bearing`` (from the storm), with the cut square to it through the plan's cut point."""

    d = distance if distance is not None else fit_distance(scene, bearing, width, height)
    cam = orbit_camera(target_of(scene), bearing, ELEVATION, d, width, height)
    fwd, _, _ = cam.basis()
    normal = np.array([fwd[0], fwd[1], 0.0])
    cut = scene.plan.cut_xy
    cam.clip = (np.array([cut[0], cut[1], 0.0]), normal / np.linalg.norm(normal))
    return cam


# ----------------------------------------------------------------------------------------------- the marcher
def ray_box(origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
        t1 = (lo[None, :] - origin[None, :]) * inv
        t2 = (hi[None, :] - origin[None, :]) * inv
    return np.maximum(np.minimum(t1, t2).max(axis=1), 0.0), np.maximum(t1, t2).min(axis=1)


def _index(scene: Scene, px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> tuple[np.ndarray, ...]:
    fi = (px - scene.x0) / scene.dxr - 0.5
    fj = (py - scene.y0) / scene.dyr - 0.5
    kz = np.clip((pz / scene.kstep).astype(np.int64), 0, scene.ktable.size - 1)
    return scene.ktable[kz], fj, fi


def trilinear(flat: np.ndarray, nk: int, ny: int, nx: int, fk: np.ndarray, fj: np.ndarray,
              fi: np.ndarray) -> np.ndarray:
    """Trilinear samples of a (nk*ny*nx, C) volume at fractional (k, j, i)."""

    fk = np.clip(fk, 0.0, nk - 1.0001)
    fj = np.clip(fj, 0.0, ny - 1.0001)
    fi = np.clip(fi, 0.0, nx - 1.0001)
    k0, j0, i0 = fk.astype(np.int64), fj.astype(np.int64), fi.astype(np.int64)
    wk = (fk - k0).astype(np.float32)[:, None]
    wj = (fj - j0).astype(np.float32)[:, None]
    wi = (fi - i0).astype(np.float32)[:, None]
    sk, sj = ny * nx, nx
    base = (k0 * ny + j0) * nx + i0
    c00 = flat.take(base, axis=0)
    c00 += (flat.take(base + 1, axis=0) - c00) * wi
    c01 = flat.take(base + sj, axis=0)
    c01 += (flat.take(base + sj + 1, axis=0) - c01) * wi
    c10 = flat.take(base + sk, axis=0)
    c10 += (flat.take(base + sk + 1, axis=0) - c10) * wi
    c11 = flat.take(base + sk + sj, axis=0)
    c11 += (flat.take(base + sk + sj + 1, axis=0) - c11) * wi
    c00 += (c01 - c00) * wj
    c10 += (c11 - c10) * wj
    c00 += (c10 - c00) * wk
    return c00


def _phase(cos_theta: np.ndarray) -> np.ndarray:
    """Two Henyey-Greenstein lobes and an isotropic part (isotropic = 1): backlit edges glow a little."""

    def hg(g: float) -> np.ndarray:
        return (1 - g * g) / (1 + g * g - 2 * g * cos_theta) ** 1.5

    return (0.38 + 0.50 * hg(0.72) + 0.12 * hg(-0.25)).astype(np.float32)


def march(scene: Scene, cam: Camera, pix: np.ndarray, seed: int = 0) -> np.ndarray:
    """(N, 5) per pixel: radiance RGB, the storm's transmittance (1 where nothing of it is drawn, the ground and
    sky aside), and the distance where the drawing passes half opacity."""

    dirs = cam.rays(pix)
    n = dirs.shape[0]
    o = cam.eye.astype(np.float64)
    radiance = np.zeros((n, 3), np.float32)
    trans = np.ones(n, np.float32)
    depth = np.full(n, np.inf, np.float32)
    tnear, tfar = ray_box(o, dirs, scene.lo, scene.hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        tground = np.where(dirs[:, 2] < -1e-9, -o[2] / dirs[:, 2], np.inf) if o[2] > 0 else np.full(n, np.inf)
    tfar = np.minimum(tfar, tground)
    h = scene.step
    skip = 0.9 * scene.occ_cell
    nk, ny, nx = scene.nk, scene.ny, scene.nx
    ncz, ncy, ncx = scene.occ.shape
    occ_flat = scene.occ.reshape(-1)
    shells = scene.shells
    cut_rgb = [inverse_tonemap(np.array(rgb, np.float32) * CUT_FACE_SHADE) for _t, _a, rgb in shells]
    clip_p, clip_n = cam.clip if cam.clip is not None else (None, None)
    soft = 2.0 * scene.dxr
    sun = scene.sun.astype(np.float32)
    phase_all = _phase((dirs @ scene.sun).astype(np.float32))
    start = tnear.copy()
    face = None
    if clip_p is not None:
        # Nothing nearer the camera than the cut is drawn, so each ray starts on the cut plane, where the cut
        # face (the field on the plane itself, drawn as the innermost shell it reaches: a cross-section in bands,
        # at nearly the table colours and CUT_FACE_OPACITY) is the first thing it meets.
        with np.errstate(divide="ignore", invalid="ignore"):
            along = dirs @ clip_n
            tcut = np.where(along > 1e-9, ((clip_p - o) @ clip_n) / along, np.inf)
        start = np.maximum(tnear, tcut)
        on_face = np.nonzero((tcut >= tnear) & (tcut < tfar))[0]
        if on_face.size:
            pc = o[None, :] + dirs[on_face] * tcut[on_face][:, None]
            ck, cj, ci = _index(scene, pc[:, 0], pc[:, 1], np.maximum(pc[:, 2], 0.0))
            face = (on_face, trilinear(scene.vol, nk, ny, nx, ck, cj, ci))
    act = np.nonzero(start < tfar)[0]
    if act.size:
        rng = np.random.default_rng(seed)
        t = (start[act] + h * rng.random(act.size)).astype(np.float64)
        tf = tfar[act]
        d = dirs[act].astype(np.float64)
        T = np.ones(act.size, np.float32)
        C = np.zeros((act.size, 3), np.float32)
        D = np.full(act.size, np.inf, np.float32)
        ph = phase_all[act]
        # the field at the previous sample: the cut face's for a ray that starts on it, else unknown
        prev = np.full(act.size, np.nan, np.float32)
        if face is not None:
            where = np.full(n, -1, np.int64)
            where[act] = np.arange(act.size)
            rows = where[face[0]]
            ok = rows >= 0
            rows, fv = rows[ok], face[1][ok]
            prev[rows] = fv[:, 4]
            level = np.full(rows.size, -1)
            for n_shell, (thr, _o, _c) in enumerate(shells):
                level[fv[:, 4] >= thr] = n_shell
            for n_shell in range(len(shells)):
                m = np.nonzero(level == n_shell)[0]
                if m.size:
                    a = (CUT_FACE_OPACITY * fv[m, 5]).astype(np.float32)
                    C[rows[m]] += (T[rows[m]] * a)[:, None] * cut_rgb[n_shell][None, :]
                    T[rows[m]] *= 1.0 - a
        tprev = t.copy()
        ids = act
        while ids.size:
            p = o[None, :] + d * t[:, None]
            cx = np.clip(((p[:, 0] - scene.x0) / scene.occ_cell).astype(np.int64), 0, ncx - 1)
            cy = np.clip(((p[:, 1] - scene.y0) / scene.occ_cell).astype(np.int64), 0, ncy - 1)
            cz = np.clip((p[:, 2] / scene.occ_cell).astype(np.int64), 0, ncz - 1)
            occ = occ_flat[(cz * ncy + cy) * ncx + cx]
            full = np.nonzero(occ)[0]
            if full.size:
                pf = p[full]
                fk, fj, fi = _index(scene, pf[:, 0], pf[:, 1], pf[:, 2])
                vals = trilinear(scene.vol, nk, ny, nx, fk, fj, fi)
                if clip_p is not None:
                    side = ((pf - clip_p[None, :]) @ clip_n).astype(np.float32)
                    beyond = side >= 0.0
                    wcut = smoothstep(0.0, soft, side)
                else:
                    beyond = np.ones(full.size, bool)
                    wcut = np.ones(full.size, np.float32)
                wdraw = vals[:, 5] * wcut
                alpha = wdraw * (1.0 - np.exp(-vals[:, 0] * h))
                live = alpha > 1e-5
                if live.any():
                    sel = full[live]
                    v = vals[live]
                    light = (SUN_RGB[None, :] * (0.65 * ph[sel] * v[:, 1] + v[:, 2])[:, None]
                             + SKY_RGB[None, :] * v[:, 3][:, None] + CLOUD_FILL[None, :])
                    a = alpha[live]
                    C[sel] += (T[sel] * a)[:, None] * light
                    T[sel] *= 1.0 - a
                f = vals[:, 4]
                was = prev[full]
                for thr, opacity, rgb in shells:
                    hit = beyond & (was < thr) & (f >= thr)
                    if not hit.any():
                        continue
                    hl = np.nonzero(hit)[0]
                    hsel = full[hl]
                    frac = np.clip((thr - was[hl]) / np.maximum(f[hl] - was[hl], 1e-6), 0.0, 1.0)
                    th = tprev[hsel] + frac * (t[hsel] - tprev[hsel])
                    ph_hit = o[None, :] + d[hsel] * th[:, None]
                    hk, hj, hi_ = _index(scene, ph_hit[:, 0], ph_hit[:, 1], np.maximum(ph_hit[:, 2], 0.0))
                    g = trilinear(scene.grad, nk, ny, nx, hk, hj, hi_)
                    nrm = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
                    view = -d[hsel].astype(np.float32)
                    flip = np.sum(nrm * view, axis=1) < 0
                    nrm[flip] *= -1.0
                    ndl = np.clip(nrm @ sun, 0.0, 1.0)
                    rim = (1.0 - np.clip(np.sum(nrm * view, axis=1), 0.0, 1.0)) ** 3
                    shade = np.clip(0.6 + 0.25 * ndl + 0.15 * nrm[:, 2] + 0.2 * rim, 0.55, 1.0)
                    col = inverse_tonemap(np.array(rgb, np.float32)[None, :] * shade[:, None])
                    a = np.clip(opacity * (0.8 + 0.4 * rim), 0.0, 1.0) * vals[hl, 5]
                    C[hsel] += (T[hsel] * a)[:, None] * col
                    T[hsel] *= 1.0 - a
                prev[full] = np.where(beyond, f, np.nan)
                newly = (T < 0.5) & ~np.isfinite(D)
                D[newly] = t[newly]
            empty = np.nonzero(~occ)[0]
            if empty.size:
                prev[empty] = -1.0e9
            tprev = t
            t = t + np.where(occ, h, skip)
            done = (t >= tf) | (T < 0.004)
            if done.any():
                fin = np.nonzero(done)[0]
                radiance[ids[fin]] = C[fin]
                trans[ids[fin]] = T[fin]
                depth[ids[fin]] = D[fin]
                keep = np.nonzero(~done)[0]
                ids, t, tf, tprev, d, T, C, D, ph, prev = (ids[keep], t[keep], tf[keep], tprev[keep], d[keep],
                                                          T[keep], C[keep], D[keep], ph[keep], prev[keep])
    drawn = trans.copy()
    # the ground behind, and the sky
    g = np.nonzero(np.isfinite(tground) & (trans > 0.004))[0]
    if g.size:
        tg = tground[g]
        gx = o[0] + dirs[g, 0] * tg
        gy = o[1] + dirs[g, 1] * tg
        fi = (gx - scene.x0) / scene.dxr - 0.5
        fj = (gy - scene.y0) / scene.dyr - 0.5
        amb = P.bilinear(scene.ground_amb, fj, fi, cval=1.0)
        spacing = scene.grid_m
        foot = tg * (2.0 * cam.tan_half / cam.height) / np.maximum(np.abs(dirs[g, 2]), 0.02)
        width = np.maximum(0.004 * spacing, foot)
        ux = np.abs(((gx + scene.grid_offset[0]) / spacing + 0.5) % 1.0 - 0.5) * spacing
        uy = np.abs(((gy + scene.grid_offset[1]) / spacing + 0.5) % 1.0 - 0.5) * spacing
        line = np.maximum(np.clip(1.0 - ux / width, 0, 1), np.clip(1.0 - uy / width, 0, 1))
        line *= np.clip(0.004 * spacing / width, 0.0, 1.0) ** 0.5
        alb = GROUND_RGB[None, :] * (1 - line[:, None]) + GRID_RGB[None, :] * line[:, None]
        col = alb * (SKY_RGB[None, :] * (amb * 0.8)[:, None]) + GRID_RGB[None, :] * (line * GRID_GLOW)[:, None]
        fog_len = max(70_000.0, 3.0 * float(np.linalg.norm(cam.target - cam.eye)))
        fog = np.exp(-tg / fog_len)[:, None]
        col = col * fog + SKY_HORIZON[None, :] * (1 - fog)
        radiance[g] += trans[g, None] * col
        newly = (trans[g] > 0.5) & ~np.isfinite(depth[g])
        depth[g[newly]] = tg[newly]
        trans[g] = 0.0
    s = np.nonzero(trans > 0.004)[0]
    if s.size:
        up = np.clip(dirs[s, 2], 0.0, 1.0) ** 0.45
        sky = SKY_HORIZON[None, :] * (1 - up[:, None]) + SKY_ZENITH[None, :] * up[:, None]
        radiance[s] += trans[s, None] * sky
    out = np.empty((n, 5), np.float32)
    out[:, 0:3] = radiance
    out[:, 3] = drawn
    out[:, 4] = depth
    return out


def render(scene: Scene, cam: Camera, *, seed: int = 0, chunk: int = 200_000) -> dict[str, np.ndarray]:
    """The picture's light (display RGB 0..1), the drawing's transmittance and depth, on one core."""

    n = cam.width * cam.height
    buf = np.empty((n, 5), np.float32)
    pix = np.arange(n, dtype=np.int64)
    # Interleaved rows of the picture per chunk, so every chunk holds about the same share of storm.
    chunks = max(1, int(math.ceil(n / chunk)))
    for c in range(chunks):
        sel = pix[c::chunks]
        buf[sel] = march(scene, cam, sel, seed + c)
    img = buf.reshape(cam.height, cam.width, 5)
    return {"rgb": tonemap(img[:, :, 0:3]), "trans": img[:, :, 3].copy(), "depth": img[:, :, 4].copy()}


# ----------------------------------------------------------------------------------------------- the words
def _plain_cells(plan: Plan) -> str:
    if plan.stride == 1:
        return f"every model cell drawn ({S.fmt_grid(plan.dx_m)})"
    return (f"drawn from {plan.stride} × {plan.stride} cell averages ({S.fmt_grid(plan.voxel_m)}), "
            f"to keep it to {VOXEL_BUDGET / 1e6:g} million cells")


def _scale_words(plan: Plan) -> str:
    return "true vertical scale" if plan.zscale == 1.0 else f"heights stretched {plan.zscale:g} times"


def _field_words(plan: Plan) -> tuple[str, str]:
    """(the shells, in the legend's words; where the numbers come from)."""

    values = ", ".join(f"{t:g}" for t, _ in plan.shells)
    if plan.field == "reflectivity":
        return f"shells: reflectivity {values} dBZ", plan.operator
    return (f"shells: rain and graupel or hail {values} g/m³",
            "no reflectivity in this run: precipitation water instead")


def _line(ov: S.Overlay, points: Sequence[tuple[float, float]], colour: str, width: float = 1.6,
          alpha: float = 0.9) -> None:
    """A thin polyline on the overlay (drawn 4x larger, then reduced, for smooth edges)."""

    from PIL import Image, ImageDraw

    pts = [(float(x), float(y)) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    if len(pts) < 2:
        return
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    lw = max(1, ov.U(width))
    x0, y0 = int(math.floor(min(xs))) - 2 * lw, int(math.floor(min(ys))) - 2 * lw
    x1, y1 = int(math.ceil(max(xs))) + 2 * lw, int(math.ceil(max(ys))) + 2 * lw
    if x1 <= 0 or y1 <= 0 or x0 >= ov.W or y0 >= ov.H:
        return
    ss = 4
    img = Image.new("RGBA", ((x1 - x0) * ss, (y1 - y0) * ss), (0, 0, 0, 0))
    ImageDraw.Draw(img).line([((x - x0) * ss, (y - y0) * ss) for x, y in pts], fill=S.rgb8(S.token(colour), alpha),
                             width=lw * ss, joint="curve")
    patch = img.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    cx0, cy0 = max(0, x0), max(0, y0)
    patch = patch.crop((cx0 - x0, cy0 - y0, min(x1, ov.W) - x0, min(y1, ov.H) - y0))
    ov.draw_rgba(patch, (cx0, cy0))


def _height_scale(ov: S.Overlay, cam: Camera, scene: Scene, trans: np.ndarray, top_px: float,
                  bottom_px: float) -> float | None:
    """A height scale standing on the cut face, slid along it until it is clear of the drawing."""

    fwd, right, _ = cam.basis()
    right_h = np.array([right[0], right[1], 0.0])
    right_h /= max(np.linalg.norm(right_h), 1e-9)
    cut = scene.plan.cut_xy
    base = np.array([cut[0], cut[1], 0.0])
    top_m = scene.plan.top_m
    step_km = next(s for s in (1.0, 2.0, 4.0, 5.0, 10.0) if top_m / 1000.0 / s <= 6)
    zs = np.arange(0.0, top_m / 1000.0 + 1e-6, step_km)
    depth = float((base - cam.eye) @ fwd)
    if depth <= 0:
        return None
    per_px = depth * 2.0 * cam.tan_half / cam.height
    h, w = trans.shape
    best, chosen = None, None
    for k in range(0, 60):
        for sign in ((1.0,) if k == 0 else (1.0, -1.0)):
            cand = base + right_h * sign * k * 14.0 * per_px * (w / 1920.0 + 0.5)
            pts = np.array([[cand[0], cand[1], z * 1000.0 * scene.zscale] for z in zs])
            qx, qy, qz = cam.project(pts)
            if np.any(qz <= 0):
                continue
            keep = (qy >= top_px) & (qy <= bottom_px)
            if keep.sum() < 2 or not (0.18 * w < qx[0] < 0.88 * w):
                continue
            xs = np.interp(np.linspace(0, 1, 24), [0, 1], [qx[0], qx[keep][-1]])
            ys = np.interp(np.linspace(0, 1, 24), [0, 1], [qy[0], qy[keep][-1]])
            clear = 1.0
            for dxp in (-ov.U(10), 0.0, ov.U(46)):
                xi = np.clip(np.round(xs + dxp).astype(int), 0, w - 1)
                yi = np.clip(np.round(ys).astype(int), 0, h - 1)
                clear = min(clear, float(np.percentile(trans[yi, xi], 15)))
            if best is None or clear > best[0] + 0.05:
                best = (clear, (qx, qy, keep))
            if clear >= 0.85:
                chosen = (qx, qy, keep)
                break
        if chosen is not None:
            break
    if chosen is None and best is not None:
        chosen = best[1]
    if chosen is None:
        return None
    qx, qy, keep = chosen
    last = int(np.nonzero(keep)[0][-1])
    _line(ov, list(zip(qx[:last + 1], qy[:last + 1])), "text", 1.6)
    foot = ov.U(12)
    _line(ov, [(qx[0] - foot, qy[0]), (qx[0] + foot, qy[0])], "text", 1.6)
    for n in range(1, last + 1):
        _line(ov, [(qx[n] - ov.U(6), qy[n]), (qx[n] + ov.U(6), qy[n])], "text", 1.4)
        ov.text((qx[n] + ov.U(12), qy[n]), f"{zs[n]:g} km", "label", colour="text", anchor="lm")
    words = "height on the cut face" + ("" if scene.zscale == 1.0 else f", stretched {scene.zscale:g} times")
    ov.text((qx[0] + ov.U(12), qy[0] + ov.U(18)), words, "micro", colour="data", anchor="ls")
    return float(qx[0])


def _compass(ov: S.Overlay, cam: Camera, scene: Scene, band_top: float) -> None:
    """North and east arrows lying on the ground, and which way the camera looks."""

    w, h = cam.width, cam.height
    anchor = (0.90 * w, min(0.80 * h, band_top - ov.U(70)))
    d = cam.ray_at(*anchor)
    text_xy = (anchor[0], anchor[1] + ov.U(30))
    if d[2] < -1e-6 and cam.eye[2] > 0 and math.degrees(math.asin(min(1.0, abs(float(d[2]))))) >= 3.0:
        g = cam.eye + d * (-cam.eye[2] / d[2])
        length = scene.grid_m
        for _ in range(16):
            tips = np.array([g, g + [0.0, length, 0.0], g + [length, 0.0, 0.0]])
            qx, qy, qz = cam.project(tips)
            size = max(math.hypot(qx[1] - qx[0], qy[1] - qy[0]), math.hypot(qx[2] - qx[0], qy[2] - qy[0]))
            if not np.all(qz > 0):
                break
            if size < ov.U(45):
                length *= 1.6
            elif size > ov.U(110):
                length /= 1.6
            else:
                break
        tips = np.array([g, g + [0.0, length, 0.0], g + [length, 0.0, 0.0]])
        qx, qy, qz = cam.project(tips)
        if np.all(qz > 0) and np.all((qx > 0) & (qx < w) & (qy > 0) & (qy < h)):
            label_y = []
            for n_tip, label in ((1, "N"), (2, "E")):
                vx, vy = qx[n_tip] - qx[0], qy[n_tip] - qy[0]
                ln = max(math.hypot(vx, vy), 1e-6)
                ux, uy = vx / ln, vy / ln
                head = ov.U(10)
                _line(ov, [(qx[0], qy[0]), (qx[n_tip], qy[n_tip])], "text", 1.6)
                _line(ov, [(qx[n_tip] - head * (ux - 0.5 * uy), qy[n_tip] - head * (uy + 0.5 * ux)),
                           (qx[n_tip], qy[n_tip]),
                           (qx[n_tip] - head * (ux + 0.5 * uy), qy[n_tip] - head * (uy - 0.5 * ux))], "text", 1.6)
                ov.text((qx[n_tip] + ux * ov.U(16), qy[n_tip] + uy * ov.U(16)), label, "label", colour="text",
                        anchor="mm")
                label_y.append(qy[n_tip] + uy * ov.U(16))
            # below the arrows and their letters; above them when the words above the legend leave no room (a
            # camera looking south has north, and its letter, pointing down the picture)
            low = max(max(qy[0], qy[1], qy[2]) + ov.U(8), max(label_y) + ov.U(14))
            text_xy = (qx[0], low + ov.U(26))
            if text_xy[1] > band_top:
                text_xy = (qx[0], min(min(qy[0], qy[1], qy[2]) - ov.U(10), min(label_y) - ov.U(16)))
    ov.text((text_xy[0], min(text_xy[1], band_top)), f"looking {compass_name(cam.heading_deg)}", "micro",
            colour="data", anchor="ms")


def _shell_legend(ov: S.Overlay, scene: Scene, x: float, y: float, width: float) -> None:
    """The NWS strip with a mark at each shell (and its opacity), or the water shells' chips."""

    from PIL import Image, ImageDraw

    plan = scene.plan
    ms = ov.px("micro")
    hgt = max(3, ov.U(9))
    if plan.field == "reflectivity":
        n = len(S.NWS_REFLECTIVITY)
        cw = width / n
        strip = Image.new("RGBA", (int(width), int(hgt)), (0, 0, 0, 0))
        d = ImageDraw.Draw(strip)
        for i, (_, c) in enumerate(S.NWS_REFLECTIVITY):
            d.rectangle((int(i * cw), 0, int((i + 1) * cw) - 1, hgt), fill=S.rgb8(c, 1.0))
        ov.draw_rgba(strip, (int(x), int(y)))
        for v in (10, 20, 30, 40, 50, 60, 70):
            ov.text((x + (v - 5) / 5.0 * cw, y + hgt + ov.U(8) + ms), f"{v}", "micro", anchor="ms", halo=False)
        ov.text((x + width + ov.U(10), y + hgt), "dBZ", "micro", anchor="ls", halo=False)
        for thr, opacity in plan.shells:
            xt = x + (thr - 5.0) / 5.0 * cw
            ov.text((xt, y - ov.U(4)), "▼", "micro", colour="text", anchor="ms")
            ov.text((xt, y - ov.U(4) - ms), f"{opacity * 100:.0f}%", "micro", colour="data", anchor="ms")
        ov.text((x, y - ov.U(10) - 2 * ms), "shells at each ▼, opacity as marked", "micro", colour="data",
                anchor="ls")
        return
    xs = x
    for (thr, opacity), rgb in zip(plan.shells, WATER_RGB):
        chip = Image.new("RGBA", (ov.U(18), hgt), tuple(int(round(255 * c)) for c in rgb) + (255,))
        ov.draw_rgba(chip, (int(xs), int(y)))
        label = f"{thr:g} g/m³ ({opacity * 100:.0f}%)"
        ov.text((xs + ov.U(24), y + hgt), label, "micro", anchor="ls", halo=False)
        xs += ov.U(24) + ov.measure(label, "micro")[0] + ov.U(28)
    ov.text((x, y - ov.U(10)), "shells: rain and graupel or hail, g/m³ (opacity as marked)", "micro",
            colour="data", anchor="ls")


def _bands(base: np.ndarray, u: float) -> np.ndarray:
    """Dark gradients behind the words at the top and the bottom (the drawing's own pixels, dimmed)."""

    h = base.shape[0]
    shade = np.ones(h, np.float32)
    top = int(230 * u)
    bottom = int(300 * u)
    shade[:top] = 1.0 - 0.55 * (1.0 - np.arange(top) / max(top, 1))
    shade[h - bottom:] = 1.0 - 0.80 * (np.arange(bottom) / max(bottom, 1))
    return (base.astype(np.float32) * shade[:, None, None] + 0.5).astype(np.uint8)


def compose(L: "P.Looks", scene: Scene, cam: Camera, shot: dict[str, np.ndarray]) -> tuple[S.Overlay, np.ndarray]:
    """The picture's words and marks over the drawing, in the quick-look grammar; returns (overlay, base)."""

    f = L.f
    plan = scene.plan
    h, w = shot["trans"].shape
    ov = S.Overlay((w, h))
    base = _bands(S.to8(shot["rgb"]), ov.u)
    shells_words, source = _field_words(plan)
    ov.title(TITLE, S.data_line(L.facts.name, f"{S.fmt_grid(f.dx)} grid", plan.cut_words, _scale_words(plan)))
    ov.clock(L.facts.clock(f.t))
    notes = S.data_line("CPU volume render, not a radar or a photograph", shells_words, _scale_words(plan))
    ov.footer(L.facts.footer(t=f.t, level=f"ground to {S.fmt_height(float(f.zf[plan.nk]))}", wind="", notes=notes))
    ov.mark()
    origin = L.facts.origin_m(f.nx, f.ny)
    cut_km = ((origin[0] + plan.cut_xy[0]) / 1000.0, (origin[1] + plan.cut_xy[1]) / 1000.0)
    peak = scene.meta.get("peak")
    box = f"{S.fmt_num(scene.nx * scene.dxr / 1000.0)} × {S.fmt_num(scene.ny * scene.dyr / 1000.0)} km box drawn"
    if plan.field == "reflectivity":
        peak_words = f"strongest reflectivity in the {box}: {S.fmt_num(peak)} dBZ"
    else:
        peak_words = f"most rain and graupel or hail in the {box}: {S.fmt_num(peak, 1)} g/m³"
    y = ov._head_bottom + ov.U(30)
    ov.text((ov.x0, y), peak_words, "label", colour="text", anchor="ls")
    ov.text((ov.x0, y + ov.U(26)), f"cut through x {S.fmt_num(cut_km[0], 1)} km, y {S.fmt_num(cut_km[1], 1)} km: "
            "the half nearer the camera is not drawn", "label", colour="data", anchor="ls")
    # how it is drawn, in the band above the footer (the film's pictures say all of it too)
    ms = ov.px("micro")
    values = ", ".join(f"{t:g}" for t, _ in plan.shells[:-1]) + f" and {plan.shells[-1][0]:g}"
    face_words = (f"the reflectivity on the cut, in bands from {values} dBZ" if plan.field == "reflectivity" else
                  f"the rain and graupel or hail on the cut, in bands from {values} g/m³")
    lines = [
        f"white: cloud water, ice and snow, see-through ({CLOUD_SEE_THROUGH * 100:.0f}% of its depth drawn)",
        f"{shells_words}, {source}",
        f"cut face: {face_words}, {CUT_FACE_OPACITY * 100:.0f}% opaque",
        f"{_plain_cells(plan)}; smoothed 1-2-1 over 3 cells",
    ]
    if plan.cut_note:
        lines.insert(3, plan.cut_note)
    if scene.meta["domain_fade"]:
        lines.append(f"faded over the last {S.fmt_km(scene.meta['fade_m'])} where the storm leaves the domain "
                     f"({', '.join(scene.meta['domain_fade'])})")
    if scene.meta["box_fade"]:
        lines.append(f"faded over the last {S.fmt_km(scene.meta['fade_m'])} of the drawn box "
                     f"({', '.join(scene.meta['box_fade'])})")
    lines.append("lit by a dim light from the west-south-west and by the sky: a look, not a measurement")
    foot_top = ov.y1 - 2 * int(ov.px("footer") * 1.38) - ov.U(12)
    for n, text in enumerate(reversed(lines)):
        ov.text((ov.x0, foot_top - n * int(ms * 1.55)), text, "micro", colour="muted", anchor="ls")
    notes_top = foot_top - len(lines) * int(ms * 1.55)
    legend_w = ov.U(330)
    legend_x = ov.x1 - ov.mark_width() - legend_w - ov.U(120)
    _shell_legend(ov, scene, legend_x, foot_top - ov.U(28), legend_w)
    bar_x = _height_scale(ov, cam, scene, shot["trans"], ov._head_bottom + ov.U(70), notes_top - ov.U(30))
    # the compass stays above the legend's words (at 1280 x 720 they would meet)
    _compass(ov, cam, scene, foot_top - ov.U(28) - ov.U(10) - 3 * ms - ov.U(16))
    _top_label(ov, cam, scene, shot["trans"], y + ov.U(50), bar_x)
    return ov, base


def _top_label(ov: S.Overlay, cam: Camera, scene: Scene, trans: np.ndarray, clear_y: float,
               bar_x: float | None = None) -> None:
    """The highest visible cloud top, marked with its height (to the nearest :data:`TOP_ROUND_M`): only where the
    drawing is seen at the mark, and said plainly when the cloud reaches the model's top."""

    from PIL import Image, ImageDraw

    tops = scene.meta["tops_k"]
    jj, ii = np.nonzero(tops >= 0)
    if not jj.size:
        return
    xs = scene.x0 + (ii + 0.5) * scene.dxr
    ys = scene.y0 + (jj + 0.5) * scene.dyr
    kt = tops[jj, ii] + 1
    zt = scene.zf_w[np.minimum(kt, scene.nk)]
    if cam.clip is not None:
        side = (np.stack([xs, ys, np.zeros_like(xs)], 1) - cam.clip[0][None, :]) @ cam.clip[1]
        keep = side >= 0
        xs, ys, zt, kt = xs[keep], ys[keep], zt[keep], kt[keep]
    if not xs.size:
        return
    qx, qy, qz = cam.project(np.stack([xs, ys, zt], 1))
    h, w = trans.shape
    ok = (qz > 0) & (qx > 0.25 * w) & (qx < 0.85 * w) & (qy > ov._head_bottom + ov.U(60)) & (qy < h)
    # the drawing must be seen there: the least transmittance just under the mark (a top too thin to see is not
    # marked)
    xi = np.clip(np.round(qx).astype(np.int64), 0, w - 1)
    seen = np.ones(qx.size, np.float32)
    for dy in range(0, max(2, ov.U(8)) + 1):
        yi = np.clip(np.round(qy).astype(np.int64) + dy, 0, h - 1)
        seen = np.minimum(seen, trans[yi, xi])
    ok &= seen < 0.8
    if not ok.any():
        return
    best = int(np.argmax(np.where(ok, zt, -1.0)))
    lx, ly = float(qx[best]), float(qy[best])
    ty = ly - ov.U(40)
    if ty - ov.px("label") < clear_y:
        ty = ly + ov.U(40)     # below the top, clear of the words above
    model_nz = int(scene.meta.get("model_nz") or 0)
    if model_nz and int(kt[best]) >= model_nz:
        words = f"cloud reaches the model top ({S.fmt_height(float(scene.meta['model_top_m']))})"
    else:
        top_m = round(float(zt[best]) / scene.zscale / TOP_ROUND_M) * TOP_ROUND_M
        words = f"{S.fmt_height(top_m)} top"
    # on the side away from the height scale, on a dark backing so it reads over white cloud too
    sign = 1.0 if bar_x is not None and bar_x < lx else -1.0
    _line(ov, [(lx, ly), (lx + sign * ov.U(36), ty)], "text", 1.2, 0.8)
    tx = lx + sign * ov.U(42)
    tw, asc, desc = ov.measure(words, "label")
    pad = ov.U(7)
    left = tx if sign > 0 else tx - tw
    pw, ph = int(tw + 2 * pad), int(asc + desc + 2 * pad)
    pill = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    ImageDraw.Draw(pill).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=ph // 2, fill=S.rgb8("void", 0.6))
    px0, py0 = int(round(left - pad)), int(round(ty - ph / 2))
    if 0 <= px0 and px0 + pw <= ov.W and 0 <= py0 and py0 + ph <= ov.H:
        ov.draw_rgba(pill, (px0, py0))
    ov.text((tx, ty), words, "label", colour="text", anchor="lm" if sign > 0 else "rm")


# ----------------------------------------------------------------------------------------------- the looks
def still(L: "P.Looks", out: Path, *, bearing: float = BEARING, size: tuple[int, int] = STILL_SIZE,
          log: Callable[[str], None] = lambda _m: None, plan: Plan | None = None) -> dict[str, Any]:
    """Draw the still into ``out`` (and its thumbnail); returns the numbers it shows."""

    begun = _time.perf_counter()
    _check_size(size)
    bearing = check_bearing(bearing)
    plan = plan or plan_for(L)
    guess = estimate_seconds(plan, size)
    if guess > STILL_LIMIT_S:
        raise Refused(too_long(plan, size, 1))
    scene = build_scene(L, plan, log=log)
    cam = camera_for(scene, bearing, size[0], size[1])
    shot = render(scene, cam)
    drawn = _time.perf_counter()
    ov, base = compose(L, scene, cam, shot)
    P.save(ov, base, out)
    log(f"3-D still drawn in {drawn - begun:.1f} s")
    return facts_of(plan, scene, cam, seconds=_time.perf_counter() - begun, estimate=guess)


def _check_size(size: tuple[int, int]) -> None:
    (w0, h0), (w1, h1) = SIZE_RANGE
    if not (w0 <= size[0] <= w1 and h0 <= size[1] <= h1):
        raise ValueError(f"a 3-D picture is from {w0}x{h0} to {w1}x{h1} pixels, not {size[0]}x{size[1]}")


def check_bearing(bearing: float) -> float:
    """A compass bearing in degrees (any finite number; 360 and more go round again)."""

    try:
        value = float(bearing)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        raise ValueError("--bearing takes a compass bearing in degrees, such as 240")
    return value


#: The 16:9 sizes a refusal may offer, largest first.
SIZES_16_9 = ((3840, 2160), (2560, 1440), (1920, 1080), (1600, 900), (1280, 720), (960, 540))


def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90.0 else f"{seconds / 60.0:.0f} minutes"


def too_long(plan: Plan, size: tuple[int, int], pictures: int) -> str:
    """Why a request is refused (its estimate against the limit), and the largest choice that fits the limit, or
    why none does: a smaller size for a still; fewer pictures, a smaller size, or both for an orbit."""

    limit = STILL_LIMIT_S if pictures == 1 else ORBIT_LIMIT_S
    guess = estimate_seconds(plan, size, pictures)
    what = "picture" if pictures == 1 else "orbit"
    head = (f"This {what} would take about {_duration(guess)} on one core, more than the "
            f"{f'{limit / 60:g}-minute' if limit >= 60 else f'{limit:g}-second'} limit.")
    smaller = [s for s in SIZES_16_9 if s[0] * s[1] < size[0] * size[1] and s[0] >= SIZE_RANGE[0][0]]
    sizes = [s for s in smaller if estimate_seconds(plan, s, pictures) <= limit]
    least = SIZE_RANGE[0]
    if pictures == 1:
        if sizes:
            return f"{head} Ask for a smaller one, such as --size {sizes[0][0]}x{sizes[0][1]}."
        return (f"{head} Even the smallest picture ({least[0]}x{least[1]}) would take about "
                f"{_duration(estimate_seconds(plan, least))}: this grid is too large to draw in 3-D on one core "
                "within the limit.")
    choices = []
    build = estimate_seconds(plan, size, 0)
    each = estimate_seconds(plan, size, 1) - build
    fit = int((limit - build) // each) if each > 0 else 0
    if fit >= 12:
        choices.append(f"fewer pictures, such as --orbit-frames {min(fit, pictures - 1)}")
    if sizes:
        choices.append(f"a smaller size, such as --size {sizes[0][0]}x{sizes[0][1]}")
    if choices:
        return f"{head} Ask for {' or '.join(choices)}."
    each = estimate_seconds(plan, least, 1) - estimate_seconds(plan, least, 0)
    fit = int((limit - estimate_seconds(plan, least, 0)) // each) if each > 0 else 0
    if fit >= 12:
        return (f"{head} Ask for fewer pictures at a smaller size, such as --orbit-frames {min(fit, pictures)} "
                f"--size {least[0]}x{least[1]}.")
    return (f"{head} Even 12 pictures at {least[0]}x{least[1]} would take about "
            f"{_duration(estimate_seconds(plan, least, 12))}: this grid is too large for an orbit on one core "
            "within the limit.")


def facts_of(plan: Plan, scene: Scene, cam: Camera, **more: Any) -> dict[str, Any]:
    return {"kind": plan.kind, "field": plan.field, "operator": plan.operator,
            "shells": [list(s) for s in plan.shells], "cut_xy_m": list(plan.cut_xy), "cut": plan.cut_words,
            "top_m": plan.top_m, "zscale": plan.zscale, "stride": plan.stride, "voxel_m": plan.voxel_m,
            "cells": scene.meta["cells"], "box_cells": [plan.i0, plan.i1, plan.j0, plan.j1, plan.nk],
            "peak": scene.meta.get("peak"), "bearing_deg": round(cam.heading_deg, 1),
            "looking": compass_name(cam.heading_deg), "build_s": scene.meta.get("build_s"),
            **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in more.items()}}


def orbit(L: "P.Looks", folder: Path, *, pictures: int = ORBIT_FRAMES, size: tuple[int, int] = ORBIT_SIZE,
          bearing: float = BEARING, should_stop: Callable[[], bool] = lambda: False,
          progress: Callable[[int, int], None] = lambda _d, _t: None,
          log: Callable[[str], None] = lambda _m: None) -> tuple[list[Path], dict[str, Any]]:
    """The orbit's pictures, one turn around the storm, with the cut always square to the camera."""

    from .film import save_picture

    begun = _time.perf_counter()
    _check_size(size)
    bearing = check_bearing(bearing)
    plan = plan_for(L)
    guess = estimate_seconds(plan, size, pictures)
    if guess > ORBIT_LIMIT_S:
        raise Refused(too_long(plan, size, pictures))
    scene = build_scene(L, plan, log=log)
    bearings = [bearing + 360.0 * n / pictures for n in range(pictures)]
    distance = max(fit_distance(scene, b, size[0], size[1]) for b in bearings[::max(1, pictures // 12)])
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cam = camera_for(scene, bearings[0], size[0], size[1], distance)
    for n, b in enumerate(bearings):
        if should_stop():
            break
        cam = camera_for(scene, b, size[0], size[1], distance)
        shot = render(scene, cam, seed=n)
        ov, base = compose(L, scene, cam, shot)
        out = folder / f"{n:05d}.png"
        save_picture(ov, base, out)
        paths.append(out)
        progress(n + 1, pictures)
    facts = facts_of(plan, scene, cam, seconds=_time.perf_counter() - begun, estimate=guess)
    facts.update({"pictures": len(paths), "degrees_per_picture": 360.0 / pictures, "time_s": L.f.t})
    return paths, facts


def gif_palette(pictures: Sequence[Path], sample: int = 6) -> Any:
    """One palette for every picture of the orbit's GIF: the reflectivity table's colours and the words' colours
    exactly, and the rest cut from a few of the pictures (median cut), so the legend reads as it does in the
    pictures and the MP4."""

    from PIL import Image

    from .film import GIF_SIZE

    keep: list[tuple[int, int, int]] = []
    for colour in [c for _, c in S.NWS_REFLECTIVITY] + ["title", "text", "data", "muted", "dim", "gold", "void"]:
        rgb = S.rgb8(colour)
        if rgb not in keep:
            keep.append(rgb)
    chosen = [pictures[int(round(n))] for n in np.linspace(0, len(pictures) - 1, min(sample, len(pictures)))]
    montage = Image.new("RGB", (GIF_SIZE[0], GIF_SIZE[1] * len(chosen)))
    for n, path in enumerate(chosen):
        with Image.open(path) as im:
            montage.paste(im.convert("RGB").resize(GIF_SIZE, Image.LANCZOS), (0, n * GIF_SIZE[1]))
    rest = 255 - len(keep)
    cut = montage.quantize(colors=rest, method=Image.MEDIANCUT, dither=Image.NONE)
    flat = [v for rgb in keep for v in rgb] + list(cut.getpalette()[:3 * rest])
    palette = Image.new("P", (1, 1))
    palette.putpalette(flat + flat[-3:] * (256 - len(flat) // 3))
    return palette


# ----------------------------------------------------------------------------------------------- on request
DRAWN = "drawn.json"
DRAWN_SCHEMA = "gm1.quicklook-storm3d/v1"
#: Why the last still asked for was not drawn (the Results screen says it), removed when one is drawn.
FAILED = "failed.json"


def read_drawn(out: Path) -> dict[str, Any]:
    """The stills drawn so far, by time (``quicklook/storm3d/drawn.json``); empty when there are none."""

    try:
        doc = json.loads((Path(out) / LOOK / DRAWN).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    stills = doc.get("stills") if isinstance(doc, dict) else None
    return stills if isinstance(stills, dict) else {}


def _write_drawn(out: Path, stills: dict[str, Any]) -> None:
    path = Path(out) / LOOK / DRAWN
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps({"schema": DRAWN_SCHEMA, "stills": stills}, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def pick(lineage: Sequence[tuple[float, str, Path, int]], time_s: float | None) -> tuple[float, str, Path, int]:
    """The saved time asked for (the nearest one), or the latest."""

    if not lineage:
        raise ValueError("the run has no published frames yet")
    if time_s is None:
        return lineage[-1]
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError("--time is a storm time in seconds, 0 or more.")
    return min(lineage, key=lambda row: abs(row[0] - time_s))


def draw_still(target: Path | str, *, out: Path | str | None = None, time_s: float | None = None,
               bearing: float = BEARING, size: tuple[int, int] = STILL_SIZE, label: str | None = None,
               namelist: Path | str | None = None, deck: Path | str | None = None,
               log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """``gpuwm-cm1 quicklook TARGET --look storm3d``: one still, kept beside the run's other pictures and listed in
    its ``index.json``; returns its record."""

    from .facts import run_facts
    from .frame import frames_of
    from .render import Renderer, _tag, left_behind

    bearing = check_bearing(bearing)
    renderer = Renderer(target, out=out, only=(LOOK,), label=label, namelist=namelist, deck=deck)
    lineage = frames_of(renderer.layout)
    t, key, path, segment = pick(lineage, time_s)
    begun = _time.perf_counter()
    failed = renderer.out / LOOK / FAILED
    qf = QuickFrame.open(path, key=key, segment=segment)
    try:
        facts = run_facts(renderer.layout, qf.config, qf.dx, qf.dy, label=label, settings=renderer.settings)
        tag = _tag(t)
        file = renderer.out / LOOK / f"{tag}.png"
        shown = still(P.Looks(qf, facts), file, bearing=bearing, size=size, log=log)
    except (P.NotDrawn, FrameUnreadable, ValueError) as why:
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(json.dumps({"schema": DRAWN_SCHEMA, "time": t, "message": str(why)}) + "\n",
                          encoding="utf-8")
        raise
    finally:
        qf.close()
    failed.unlink(missing_ok=True)
    stills = read_drawn(renderer.out)
    # a still of a saved time the run has since left (a resume from an earlier restart) is removed; one whose saved
    # frame was only deleted is kept (render.left_behind)
    for when in [w for w, row in stills.items() if not isinstance(row, dict)]:
        stills.pop(when)
    drawn_now = {f"{LOOK}/{tag}.png", f"{LOOK}/thumb/{tag}.png"}
    for when in left_behind(renderer.layout, lineage, stills):
        row = stills.pop(when)
        for name in (row.get("file"), row.get("thumb")):
            # (a still of the same saved time, from the branch the run left, had this very file: it was just drawn
            # over from the run's own frame, and is kept)
            if isinstance(name, str) and ".." not in Path(name).parts and name not in drawn_now:
                (renderer.out / name).unlink(missing_ok=True)
    record = {"frame": key, "segment": segment, "time": t, "file": f"{LOOK}/{tag}.png",
              "thumb": f"{LOOK}/thumb/{tag}.png", "facts": shown,
              "seconds": round(_time.perf_counter() - begun, 2)}
    stills[repr(float(t))] = record
    _write_drawn(renderer.out, stills)
    renderer.write_index(lineage)
    return {**record, "path": os.fspath(renderer.out / record["file"]), "index": os.fspath(renderer.out / "index.json")}


def film_pictures(job: Any, frames_dir: Path) -> tuple[list[Path], tuple[int, int], dict[str, Any]]:
    """The orbit film's pictures for ``quicklook.film`` (``--film storm3d``): one turn at the saved time asked for."""

    from .facts import read_settings, run_facts
    from .frame import frames_of
    from .ncread import discover

    # a GM1 run's frames, or CM1 netCDF output read as every other look reads it
    layout = discover(job.target)
    lineage = frames_of(layout)
    if not lineage:
        raise ValueError(f"{job.target} has no published frames yet")
    t, key, path, segment = pick(lineage, getattr(job, "time_s", None))
    # the film's own pass has said already when the settings cannot be read (quicklook.film)
    settings = layout.settings if layout.kind == "netcdf" else read_settings(layout)[0]
    size = tuple(getattr(job, "size", None) or ORBIT_SIZE)
    pictures = int(getattr(job, "pictures", None) or ORBIT_FRAMES)
    bearing = float(getattr(job, "bearing", None) if getattr(job, "bearing", None) is not None else BEARING)
    from .film import FilmError

    qf = QuickFrame.open(path, key=key, segment=segment)
    try:
        facts = run_facts(layout, qf.config, qf.dx, qf.dy, label=job.label, settings=settings)
        job.progress(0, pictures, message=f"building the 3-D scene at {facts.clock(t)}")
        paths, shown = orbit(P.Looks(qf, facts), frames_dir, pictures=pictures, size=size, bearing=bearing,
                             should_stop=job.should_stop, progress=lambda done, total: job.progress(done, total),
                             log=job.log)
    except Refused as why:
        raise FilmError(str(why)) from why
    finally:
        qf.close()
    shown["frame"] = key
    return paths, size, shown


__all__ = ["BEARING", "LOOK", "ORBIT_FRAMES", "ORBIT_SIZE", "Plan", "Refused", "STILL_SIZE", "Scene", "TITLE",
           "build_scene", "camera_for", "draw_still", "height_stretch", "estimate_seconds", "film_pictures", "orbit",
           "pick", "plan_for", "read_drawn", "render", "still"]
