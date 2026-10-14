"""Rotation axis, 0 to 10 km: where the storm turns at each height, drawn in three dimensions, from above, and as
its circulation by height.

The axis is a picture-tool method, not a CM1 diagnostic, and the picture says so.  At every scalar level from the
lowest to 10 km it is the centre of a circle of radius ``r``: :data:`RADIUS_M` (2 km), or three grid spacings when
that is wider (3 km on a 1 km grid, 12 km on a hurricane's 4 km grid).

* A thunderstorm's centre is the circle with the most circulation: the area sum of the vertical vorticity inside
  the circle, which equals the wind's line integral around it, so the circle that holds the most of the storm's
  turning is centred on its rotation.  The axis starts at the column of most 2-5 km updraft helicity (CM1's
  ``uh``, or CM1's calcuh sum on the frame's winds), averaged over the circle, at the level nearest
  :data:`REF_HEIGHT_M`.
* A hurricane's centre is the circle with the lowest mean pressure.  Its eye can be wider than the circle, and
  the circle with the most circulation then sits in the eyewall, not at the centre; the pressure is lowest in the
  middle of the eye at every height.  The axis starts at the lowest pressure at the lowest level.
* Each centre is a peak of its field (no point next to it is higher, or for the pressure lower): the highest one
  within reach of the level before that keeps the rule below, placed between grid points by a parabola through the
  peak and its neighbours.
* The axis is followed level by level from where it starts, down to the lowest level and up to 10 km, and obeys
  one rule, stated on the picture: no two of its centres lie further apart sideways than :data:`TILT_MAX` km per km
  of the height between them, plus :data:`SLACK_CELLS` grid spacings and never less than :data:`SLACK_MIN_M` (a
  centre is found to about a grid spacing, and on a fine grid the most circulation moves about inside the circle
  as the small vortices in it change from level to level).
  Where the next level's peak would break the rule, or the circle's circulation is below the least the axis is
  drawn from, the axis stops, and the picture says where and why.  The rule holds between every pair of heights,
  so thin stretched levels near the ground cannot add up small steps into a large drift.
* It is drawn only for a storm that rotates: a thunderstorm whose circulation at the starting level averages less
  than :data:`MIN_SPIN` around the circle (``circulation / 2 pi r``), or whose air in the circle is not rising
  there, is not drawn, and the picture's record says why (a hurricane: one with too little circulation around its
  centre at the lowest level).  An axis that covers less than :data:`MIN_DEPTH_M` of height is not drawn either.

The vorticity is CM1's own ``zvort`` when the frame carries it, else ``dv/dx - du/dy`` from the frame's winds
(:meth:`.frame.QuickFrame.zeta_level`); the pressure is the frame's ``prs``, else CM1's own from ``pi0`` and
``pp3d`` (:meth:`.frame.QuickFrame.pressure_level`).  The slices under the axis (the lowest level, the level
nearest 1 km when it is at least 600 m higher, and the level nearest 5 km) show that vorticity on one fixed range
for the grid spacing (:func:`axis_spin_range`), sampled bilinearly.
The circle sums use the whole grid cells whose centres are inside the circle; nothing else is smoothed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import products as P
from . import style as S

AXIS_TOP_M = 10000.0
REF_HEIGHT_M = 3000.0
RADIUS_M = 2000.0
RADIUS_CELLS = 3.0
TILT_MAX = 1.5             # km sideways per km of height, between any two heights of the axis
SLACK_CELLS = 1.5          # plus this many grid spacings (a centre is found to about a grid spacing) ...
SLACK_MIN_M = 500.0        # ... and never less than this (on a fine grid a centre moves about inside the circle)
MIN_SPIN = 3.0             # m/s: the circulation around the circle over its circumference
MIN_RISE = 1.0             # m/s: the mean vertical velocity in the circle there (a thunderstorm's rotating updraft)
MIN_DEPTH_M = 1000.0       # an axis shorter than this is not drawn
BOX_RADII = 4.0            # the 3-D box and the view from above reach this many circle radii from the axis
HURRICANE_SPIN_RANGE = 0.01  # 1/s on a hurricane's slices
#: The axis's colours by height: amber near the ground, through rose and violet, to pale blue at 10 km.
HEIGHT_COLOURS = S.Ramp("axis_height", ((0.0, "#ffb347"), (0.3, "#ff6f7d"), (0.6, "#d77cf0"), (1.0, "#9fd0ff")),
                        0.0, 1.0, "", "sequential", "the rotation axis's height (not in the shared ramps)")
SLICE_HEIGHTS = (None, 1000.0, 5000.0)   # None: the lowest level
#: Why the axis stops short of the lowest level or of 10 km, in the picture's words.
STOPS = {"weak": "the turning in the circle is weaker than the axis needs",
         "fast": "the centre moves sideways faster than the rule allows"}


def axis_spin_range(dx_m: float, hurricane: bool = False) -> float:
    """The slices' fixed vorticity range: 0.01 1/s at 1 km, scaled as 1/dx (a finer grid resolves stronger, smaller
    eddies) to the nearest 1-2-5 step, so 0.02 at 500 m, 0.05 at 250 m, 0.1 at 100 m; a hurricane's, 0.01 1/s on
    every grid.  The mesocyclone aloft is often stronger and is then drawn in the range's last colour; the
    circulation numbers beside the picture carry its strength."""

    if hurricane:
        return HURRICANE_SPIN_RANGE
    raw = 0.01 * 1000.0 / max(float(dx_m), 1.0)
    exponent = math.floor(math.log10(raw))
    candidates = [f * 10.0 ** e for e in (exponent - 1, exponent, exponent + 1) for f in (1.0, 2.0, 5.0)]
    return float(min(candidates, key=lambda c: abs(math.log(c / raw))))


def circle_radius(dx_m: float) -> float:
    return max(RADIUS_M, RADIUS_CELLS * float(dx_m))


def radius_words(r_m: float) -> str:
    """'a circle of 3 km radius': the radius, said as one (a "3 km circle" reads as 3 km across)."""

    return f"a circle of {S.fmt_km(r_m)} radius"


def _disk(rc: float) -> np.ndarray:
    n = int(math.ceil(rc))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    return (x * x + y * y <= rc * rc + 1e-9).astype(np.float32)


def disk_sum(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """The sum of ``a`` over the disk around every point (zero outside the array), by FFT convolution."""

    ny, nx = a.shape
    ky, kx = kernel.shape
    shape = (ny + ky, nx + kx)
    full = np.fft.irfft2(np.fft.rfft2(a, shape) * np.fft.rfft2(kernel, shape), shape)
    oy, ox = ky // 2, kx // 2
    return full[oy:oy + ny, ox:ox + nx].astype(np.float32)


def _peak(g: np.ndarray, j: int, i: int) -> tuple[float, float]:
    """(i, j) of the peak placed between grid points by a parabola through it and its neighbours."""

    def offset(a: float, b: float, c: float) -> float:
        d = a - 2.0 * b + c
        return 0.0 if d >= 0.0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))

    ny, nx = g.shape
    di = offset(g[j, i - 1], g[j, i], g[j, i + 1]) if 0 < i < nx - 1 else 0.0
    dj = offset(g[j - 1, i], g[j, i], g[j + 1, i]) if 0 < j < ny - 1 else 0.0
    return i + di, j + dj


def at_point(g: np.ndarray, i: float, j: float) -> float:
    """``g`` at a point between grid points (crop indices), bilinearly."""

    ny, nx = g.shape
    i0, j0 = int(min(max(math.floor(i), 0), nx - 2)), int(min(max(math.floor(j), 0), ny - 2))
    a, b = min(max(i - i0, 0.0), 1.0), min(max(j - j0, 0.0), 1.0)
    return float((1 - a) * (1 - b) * g[j0, i0] + a * (1 - b) * g[j0, i0 + 1] + (1 - a) * b * g[j0 + 1, i0]
                 + a * b * g[j0 + 1, i0 + 1])


def local_peaks(g: np.ndarray) -> np.ndarray:
    """Where ``g`` is at least as high as each of its eight neighbours (the array's edge counts as lower)."""

    pad = np.pad(g, 1, mode="constant", constant_values=-np.inf)
    ny, nx = g.shape
    out = np.ones(g.shape, bool)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dj or di:
                out &= g >= pad[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
    return out


class Axis:
    """The axis of one frame: its centre (x, y in m, domain frame), circulation (m2/s) and the circle's mean w
    (m/s) at each scalar level from the lowest to 10 km it reaches, or the reason there is none."""

    def __init__(self, L: "P.Looks") -> None:
        f = L.f
        self.L = L
        self.dx, self.dy = f.dx, f.dy
        self.r = circle_radius(max(f.dx, f.dy))
        self.kernel = _disk(self.r / f.dx)
        self.cell = f.dx * f.dy
        self.slack = max(SLACK_CELLS * max(f.dx, f.dy), SLACK_MIN_M)
        self.hurricane = L.hurricane
        top = min(AXIS_TOP_M, float(f.zh[-1]))
        self.levels = [k for k, z in enumerate(f.zh) if z <= top + 1e-6]
        self.top = float(f.zh[self.levels[-1]])
        self.min_circulation = 2.0 * math.pi * self.r * MIN_SPIN
        self.why: str | None = None
        self.start_words = ""
        self.method_words = (f"centre: lowest mean pressure in {radius_words(self.r)}" if self.hurricane
                             else f"centre: most circulation around {radius_words(self.r)}")
        #: why the axis stops below its lowest and above its highest level (a key of :data:`STOPS`), or None
        self.stops: dict[str, str | None] = {"below": None, "above": None}
        self.zeta_source = ""
        self._find()

    # -- the crop the search runs in -------------------------------------------------------
    def _crop(self, ci: int, cj: int) -> None:
        f = self.L.f
        half = int(math.ceil((2.0 * BOX_RADII + 2.0) * self.r / f.dx))
        self.i0, self.i1 = max(0, ci - half), min(f.nx, ci + half + 1)
        half_y = int(math.ceil((2.0 * BOX_RADII + 2.0) * self.r / f.dy))
        self.j0, self.j1 = max(0, cj - half_y), min(f.ny, cj + half_y + 1)

    def zeta(self, k: int) -> np.ndarray:
        z, self.zeta_source = self.L.f.zeta_level(k)
        return z[self.j0:self.j1, self.i0:self.i1]

    def circulation(self, k: int) -> np.ndarray:
        """The circulation (m2/s) around the circle centred on every point of the crop, at level k."""

        return disk_sum(self.zeta(k), self.kernel) * np.float32(self.cell)

    def low_pressure(self, k: int) -> np.ndarray:
        """Minus the mean pressure (Pa) in the circle centred on every point of the crop, at level k: highest
        where the circle's pressure is lowest.  A circle cut by the domain's edge averages the cells it holds."""

        p = self.L.f.pressure_level(k)
        crop = np.asarray(p[self.j0:self.j1, self.i0:self.i1], np.float64)
        if getattr(self, "_count", None) is None or self._count.shape != crop.shape:
            self._count = np.maximum(disk_sum(np.ones(crop.shape, np.float32), self.kernel), np.float32(1.0))
        mean = disk_sum((crop - float(crop.mean())).astype(np.float32), self.kernel) / self._count
        return -mean

    def w_mean(self, k: int, i: float, j: float) -> float:
        w = self.L.f.view("w3d")
        plane = 0.5 * (np.asarray(w[k][self.j0:self.j1, self.i0:self.i1], np.float32)
                       + np.asarray(w[k + 1][self.j0:self.j1, self.i0:self.i1], np.float32))
        s = disk_sum(plane, self.kernel) / np.float32(self.kernel.sum())
        return float(s[int(round(j)), int(round(i))])

    def _field(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(the field whose peak is the centre, the circulation) at level k, both on the crop."""

        g = self.circulation(k)
        return (self.low_pressure(k) if self.hurricane else g), g

    def _find(self) -> None:
        L, f = self.L, self.L.f
        if self.hurricane:
            if L.pressure0_hpa is None:
                self.why = "This run saved no pressure, which a hurricane's axis is found from."
                return
            p = L.pressure0_hpa
            cj, ci = np.unravel_index(int(np.argmin(p)), p.shape)
            kref = self.levels[0]
            self.start_words = "the storm's centre (its lowest pressure at the lowest level)"
        else:
            uh, self.uh_source = L.uh
            if float(uh.max()) <= 0.0:
                self.why = "No rotating updraft in this frame: the updraft helicity is zero everywhere."
                return
            smooth = disk_sum(np.maximum(uh, 0.0), self.kernel)
            cj, ci = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
            kref = int(min(self.levels, key=lambda k: abs(float(f.zh[k]) - REF_HEIGHT_M)))
            self.start_words = "the most 2–5 km updraft helicity"
        self._crop(int(ci), int(cj))
        self.kref = kref
        where = "at the lowest level" if self.hurricane else f"at {S.fmt_height(float(f.zh[kref]))}"
        storm = "storm" if self.hurricane else "updraft"
        # the starting level: the highest peak within one circle radius of the starting column
        field, g = self._field(kref)
        reach = self.r + 0.75 * max(f.dx, f.dy)
        ny, nx = field.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        near = ((xx - (ci - self.i0)) * self.dx) ** 2 + ((yy - (cj - self.j0)) * self.dy) ** 2 <= reach ** 2
        jb, ib = np.unravel_index(int(np.argmax(np.where(near, field, -np.inf))), field.shape)
        self.circulation_ref = float(g[jb, ib])
        start = self._best(field, g, (ci - self.i0, cj - self.j0), reach, {}, kref)
        if isinstance(start, tuple):
            self.circulation_ref = start[2]
        self.centres: dict[int, tuple[float, float, float]] = {}
        self.ks: list[int] = []
        if self.circulation_ref < self.min_circulation:
            # one wording for every such frame, so the run's record lists it once with its times
            self.why = (f"No rotating {storm} in this frame: the circulation around {radius_words(self.r)} "
                        f"{where} stays below {self.min_circulation / 1e4:.1f} × 10⁴ m²/s, the least the axis is "
                        "drawn from.")
            return
        if not isinstance(start, tuple):
            field_words = "the mean pressure in the circle has no low" if self.hurricane else "the turning has no centre"
            self.why = (f"No rotating {storm} in this frame: {where}, {field_words} within {S.fmt_km(self.r)} of "
                        f"{self.start_words}.")
            return
        centres = self.centres
        centres[kref] = start
        self.ks = [kref]
        self.w_ref = self.w_mean(kref, start[0], start[1])
        if not self.hurricane and self.w_ref < MIN_RISE:
            self.why = (f"No rotating updraft in this frame: the air in the most rotating circle at "
                        f"{S.fmt_height(float(f.zh[kref]))} is not rising.")
            return
        for direction, side in ((-1, "below"), (1, "above")):
            prev = kref
            k = kref + direction
            while k in self.levels:
                step = abs(float(f.zh[k]) - float(f.zh[prev]))
                field, g = self._field(k)
                reach = TILT_MAX * step + self.slack + 0.75 * max(f.dx, f.dy)
                found = self._best(field, g, centres[prev][:2], reach, centres, k)
                if not isinstance(found, tuple):
                    self.stops[side] = "fast"
                    break
                if found[2] < self.min_circulation:
                    self.stops[side] = "weak"
                    break
                centres[k] = found
                prev = k
                k += direction
        self.ks = sorted(centres)
        if float(f.zh[self.ks[-1]]) - float(f.zh[self.ks[0]]) < MIN_DEPTH_M - 1e-6:
            lo, hi = S.fmt_height(float(f.zh[self.ks[0]])), S.fmt_height(float(f.zh[self.ks[-1]]))
            self.why = (f"No rotation axis in this frame: the turning could be followed only from {lo} to {hi}, "
                        f"less than the {S.fmt_km(MIN_DEPTH_M)} the axis is drawn from.")

    def _best(self, field: np.ndarray, g: np.ndarray, at: tuple[float, float], reach_m: float,
              accepted: dict[int, tuple[float, float, float]], k: int) -> tuple[float, float, float] | str | None:
        """(i, j, circulation) of the highest peak of ``field`` within ``reach_m`` of ``at`` (crop indices) that
        keeps the rule against every ``accepted`` centre, placed between grid points, with the circulation around
        the circle there; "fast" when no peak in reach keeps it (the turning moves off faster); None when the
        reach holds no peak at all."""

        ny, nx = field.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        inside = ((xx - at[0]) * self.dx) ** 2 + ((yy - at[1]) * self.dy) ** 2 <= reach_m ** 2
        jj, ii = np.nonzero(inside & local_peaks(field))
        if not jj.size:
            return "fast" if accepted else None
        for n in np.argsort(-field[jj, ii], kind="stable"):
            j, i = int(jj[n]), int(ii[n])
            fi, fj = _peak(field, j, i)
            if not self._keeps_rule(fi, fj, k, accepted):
                if not self._keeps_rule(float(i), float(j), k, accepted):
                    continue
                fi, fj = float(i), float(j)        # the parabola alone would break the rule: the grid point
            # the circulation around the circle where the centre is drawn
            return fi, fj, at_point(g, fi, fj) if min(g.shape) > 1 else float(g[j, i])
        return "fast"

    def _keeps_rule(self, i: float, j: float, k: int, accepted: dict[int, tuple[float, float, float]]) -> bool:
        z = float(self.L.f.zh[k])
        for kk, (ia, ja, _g) in accepted.items():
            apart = math.hypot((i - ia) * self.dx, (j - ja) * self.dy)
            if apart > TILT_MAX * abs(z - float(self.L.f.zh[kk])) + self.slack + 1e-6:
                return False
        return True

    # -- what the picture shows ---------------------------------------------------------------
    def xy_m(self, k: int) -> tuple[float, float]:
        i, j, _g = self.centres[k]
        return (self.i0 + i + 0.5) * self.dx, (self.j0 + j + 0.5) * self.dy

    def points(self) -> np.ndarray:
        """(n, 4): x (m), y (m), z (m), circulation (m2/s) of each level of the axis, bottom up."""

        f = self.L.f
        return np.array([(*self.xy_m(k), float(f.zh[k]), self.centres[k][2]) for k in self.ks])

    def rule_words(self) -> str:
        why = (f"{SLACK_CELLS:g} grid spacings" if SLACK_CELLS * max(self.dx, self.dy) >= SLACK_MIN_M
               else f"the least the rule allows, more than {SLACK_CELLS:g} grid spacings on this fine a grid")
        return (f"No two of its centres lie further apart sideways than {TILT_MAX:g} km per km of the height "
                f"between them, plus {S.fmt_km(self.slack)} ({why}).")


# ----------------------------------------------------------------------------------------------- the 3-D view
class Camera:
    """An orthographic view from the south-south-west, a little above: no perspective, so heights and lengths
    compare anywhere in the picture.  World units are km, with heights multiplied by ``stretch``."""

    def __init__(self, azimuth_deg: float = 200.0, elevation_deg: float = 24.0) -> None:
        az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
        # the direction the camera looks (from the camera toward the scene)
        self.d = -np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
        self.right = np.cross(self.d, [0.0, 0.0, 1.0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.d)
        self.scale = 1.0
        self.ox = self.oy = 0.0

    def fit(self, corners: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        sx, sy = corners @ self.right, corners @ self.up
        x0, y0, x1, y1 = rect
        self.scale = min((x1 - x0) / (sx.max() - sx.min()), (y1 - y0) / (sy.max() - sy.min()))
        self.ox = x0 + 0.5 * (x1 - x0) - self.scale * 0.5 * (sx.max() + sx.min())
        self.oy = y0 + 0.5 * (y1 - y0) + self.scale * 0.5 * (sy.max() + sy.min())

    def project(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        return np.stack([self.ox + self.scale * (p @ self.right), self.oy - self.scale * (p @ self.up)], -1)

    def plane(self, h: float, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World x, y where the rays through pixels (xs, ys) meet the horizontal plane at height h."""

        a = (xs - self.ox) / self.scale
        b = -(ys - self.oy) / self.scale
        # p = a right + b up + t d, with p_z = h
        t = (h - a * self.right[2] - b * self.up[2]) / self.d[2]
        return a * self.right[0] + b * self.up[0] + t * self.d[0], a * self.right[1] + b * self.up[1] + t * self.d[1]


def _composite(base: np.ndarray, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return base * (1.0 - a) + rgb * a


def _line_layer(size: tuple[int, int], pts: np.ndarray, colours: list[str], widths: np.ndarray, ss: int = 3):
    """An RGBA layer with the polyline through ``pts`` (pixels), each segment its colour and width, anti-aliased by
    drawing ``ss`` times larger."""

    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for n in range(len(pts) - 1):
        (xa, ya), (xb, yb) = pts[n], pts[n + 1]
        width = max(1, int(round(0.5 * (widths[n] + widths[n + 1]) * ss)))
        col = S.rgb8(colours[n], 1.0)
        d.line((xa * ss, ya * ss, xb * ss, yb * ss), fill=col, width=width)
        r = width / 2.0
        d.ellipse((xb * ss - r, yb * ss - r, xb * ss + r, yb * ss + r), fill=col)
    if len(pts):
        r = max(1.0, widths[0] * ss / 2.0)
        d.ellipse((pts[0][0] * ss - r, pts[0][1] * ss - r, pts[0][0] * ss + r, pts[0][1] * ss + r),
                  fill=S.rgb8(colours[0], 1.0))
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def _slice_levels(f: Any, top: float) -> list[int]:
    """The slices under the axis: the lowest level, the level nearest 1 km when it is at least 600 m above the
    lowest, and the level nearest 5 km."""

    out = [0]
    k1 = f.level_near(1000.0)
    if float(f.zh[k1]) - float(f.zh[0]) >= 600.0:
        out.append(k1)
    k5 = f.level_near(5000.0)
    if float(f.zh[k5]) <= top and k5 not in out:
        out.append(k5)
    return out


def product_rotation_axis(L: "P.Looks", out: Path) -> dict[str, Any]:
    f = L.f
    A = Axis(L)
    if A.why:
        raise P.NotDrawn(A.why)
    pts = A.points()
    rng = axis_spin_range(f.dx, L.hurricane)
    half = BOX_RADII * A.r                      # m: the box reaches this far from its middle
    lx, ly = f.extent
    mx = float(np.clip(pts[:, 0].mean(), half, max(half, lx - half)))
    my = float(np.clip(pts[:, 1].mean(), half, max(half, ly - half)))
    hk = half / 1000.0
    stretch = max(1.0, 0.75 * 2.0 * hk / (AXIS_TOP_M / 1000.0))
    ztop_d = A.top / 1000.0 * stretch
    cam = Camera()
    corners = np.array([(x, y, z) for x in (-hk, hk) for y in (-hk, hk) for z in (0.0, ztop_d)])
    scene = (420, 172, 1320, 930)
    sx0, sy0, sx1, sy1 = scene
    cam.fit(corners, (sx0 + 30, sy0 + 20, sx1 - 30, sy1 - 60))
    ys, xs = np.mgrid[sy0:sy1, sx0:sx1].astype(np.float32) + 0.5
    hdr = np.zeros((sy1 - sy0, sx1 - sx0, 3), np.float32) + np.array(S.hex_rgb("void"), np.float32)
    size = (sx1 - sx0, sy1 - sy0)
    ax_x, ax_y = (pts[:, 0] - mx) / 1000.0, (pts[:, 1] - my) / 1000.0
    ax_z = pts[:, 2] / 1000.0 * stretch
    widths = 2.5 + 7.5 * np.clip(pts[:, 3] / max(float(pts[:, 3].max()), 1.0), 0.0, 1.0)
    colours = [HEIGHT_COLOURS.at(float(np.clip(z / AXIS_TOP_M, 0.0, 1.0))) for z in pts[:, 2]]
    shift = np.array([sx0, sy0], np.float64)
    proj = cam.project(np.stack([ax_x, ax_y, ax_z], -1)) - shift
    floor_d = float(f.zh[0]) / 1000.0 * stretch
    shadow = cam.project(np.stack([ax_x, ax_y, np.full_like(ax_z, floor_d)], -1)) - shift
    slices = _slice_levels(f, A.top)
    heights = {}
    drawn_to = -1
    for idx, k in enumerate(slices):
        zd = float(f.zh[k]) / 1000.0 * stretch
        heights[k] = zd
        if idx > 0:
            # the axis below this slice first, so the slice lies over it
            sel = np.nonzero(ax_z <= zd + 1e-9)[0]
            sel = sel[sel >= max(drawn_to, 0)]
            if sel.size > 1:
                layer = _line_layer(size, proj[sel], [colours[s] for s in sel], widths[sel])
                hdr = _composite(hdr, layer[..., :3], layer[..., 3])
                drawn_to = int(sel.max())
        wx, wy = cam.plane(zd, xs, ys)
        inside = (np.abs(wx) <= hk) & (np.abs(wy) <= hk)
        zeta, _source = f.zeta_level(k)
        vals = P.bilinear(zeta, (my + wy * 1000.0) / f.dy - 0.5, (mx + wx * 1000.0) / f.dx - 0.5)
        rgb = S.RAMPS["zeta"].colours(vals, -rng, rng)
        mag = np.clip(np.abs(vals) / rng, 0.0, 1.0)
        if idx == 0:
            alpha = np.where(inside, 1.0, 0.0)
        else:
            # a slice aloft is a glass sheet: faint where the spin is weak, solid where it is strong
            rgb = np.maximum(rgb, np.array(S.hex_rgb("#10141f"), np.float32))
            alpha = np.where(inside, 0.30 + 0.62 * mag ** 0.7, 0.0)
        hdr = _composite(hdr, rgb, alpha.astype(np.float32))
        if idx == 0:
            # the axis's shadow on the ground: where each height's centre lies over the floor
            layer = _line_layer(size, shadow, ["#7d879c"] * len(shadow), np.full(len(shadow), 2.0))
            hdr = _composite(hdr, layer[..., :3], layer[..., 3])
    sel = np.arange(max(drawn_to, 0), len(proj))
    if sel.size > 1:
        layer = _line_layer(size, proj[sel], [colours[s] for s in sel], widths[sel])
        hdr = _composite(hdr, layer[..., :3], layer[..., 3])
    cv = P.canvas()
    cv[sy0:sy1, sx0:sx1] = S.to8(hdr)
    # the slices' edges and the plumb line from the axis's foot (a vertical to read the lean against)
    from PIL import Image, ImageDraw

    im = Image.fromarray(cv)
    d = ImageDraw.Draw(im)
    for k, zd in heights.items():
        sq = cam.project(np.array([(-hk, -hk, zd), (hk, -hk, zd), (hk, hk, zd), (-hk, hk, zd), (-hk, -hk, zd)]))
        d.line([tuple(p) for p in sq], fill=S.rgb8("faint" if k == slices[0] else "dim"), width=1)
    foot = np.array([ax_x[0], ax_y[0], 0.0])
    pb, pt = cam.project(foot)[0], cam.project(foot + [0.0, 0.0, ztop_d])[0]
    for t0 in np.arange(0.0, 1.0, 0.025):
        a, b = pb + (pt - pb) * t0, pb + (pt - pb) * min(1.0, t0 + 0.0125)
        d.line([tuple(a), tuple(b)], fill=S.rgb8("muted"), width=1)
    ticks_km = np.arange(0.0, A.top / 1000.0 + 1e-6, 1.0)
    for zk in ticks_km:
        p = cam.project(foot + [0.0, 0.0, zk * stretch])[0]
        d.line([(p[0] - 4, p[1]), (p[0] + 4, p[1])], fill=S.rgb8("muted"), width=1)
    cv = np.asarray(im).copy()
    # the axis seen from above, and the circulation by height
    # the view from above holds the whole axis and the circle at its foot, with a margin
    xlo, xhi = min(pts[:, 0].min(), pts[0, 0] - A.r), max(pts[:, 0].max(), pts[0, 0] + A.r)
    ylo, yhi = min(pts[:, 1].min(), pts[0, 1] - A.r), max(pts[:, 1].max(), pts[0, 1] + A.r)
    span = 1.15 * max(xhi - xlo, yhi - ylo)
    plan = P.Box(1380, 214, 440, 0.5 * (xlo + xhi) - span / 2, 0.5 * (ylo + yhi) - span / 2, span, h=440)
    cv = _plan(cv, plan, pts, colours, A, f)
    chart = (1380, 742, 1840, 912)
    cv = _chart(cv, chart, A, f)
    ox, oy = L.facts.origin_m(f.nx, f.ny)
    k1 = min(A.ks, key=lambda k: abs(float(f.zh[k]) - 1000.0))
    k5 = min(A.ks, key=lambda k: abs(float(f.zh[k]) - 5000.0))
    x0m, y0m = A.xy_m(A.ks[0])
    x5m, y5m = A.xy_m(k5)
    lean = math.hypot(x5m - x0m, y5m - y0m)
    toward = P.COMPASS[int(round(math.degrees(math.atan2(y5m - y0m, x5m - x0m)) / 45.0)) % 8]
    zlow, z5 = float(f.zh[A.ks[0]]), float(f.zh[k5])
    vertical_words = (f"heights drawn {stretch:.1f} × the horizontal scale" if stretch > 1.05
                      else "true vertical scale")
    ov = P.overlay(L, "rotation-axis",
                   S.data_line(L.facts.name, f"{S.fmt_grid(f.dx)} grid", A.method_words,
                               f"foot at x {S.fmt_num((ox + x0m) / 1000.0, 1)}, "
                               f"y {S.fmt_num((oy + y0m) / 1000.0, 1)} km"),
                   level=f"{S.fmt_height(zlow)} to {S.fmt_height(float(f.zh[A.ks[-1]]))}", wind="",
                   notes=S.data_line("a picture tool's method, not CM1's", f"spin: {A.zeta_source}",
                                     f"slices ±{rng:g} {S.PER_S} fixed", vertical_words, L.frame_words()))
    x, y = ov.x0, scene[1] + ov.U(30)
    # the measured lean, however small: the words beside it say each centre is found to about a grid spacing
    lean_text = f"{S.fmt_km(lean)} {toward}" if lean >= 1.0 else "0 m"
    ov.readout((x, y), [(f"lean, {S.fmt_height(zlow)} to {S.fmt_height(z5)}", lean_text),
                        (f"circulation at {S.fmt_height(float(f.zh[k1]))}",
                         f"{A.centres[k1][2] / 1e4:.1f} × 10⁴ m²/s"),
                        (f"circulation at {S.fmt_height(z5)}", f"{A.centres[k5][2] / 1e4:.1f} × 10⁴ m²/s")])
    ms = ov.px("micro")
    follows = "the lowest pressure up" if A.hurricane else "the turning up and down"
    paragraphs = [f"The axis starts at {A.start_words}, at {S.fmt_height(float(f.zh[A.kref]))}, and follows "
                  f"{follows}. {A.rule_words()}",
                  f"each centre found to about a grid spacing, {S.fmt_km(max(f.dx, f.dy))}",
                  "width: circulation; colour: height",
                  (f"heights drawn {stretch:.1f} × the horizontal scale" if stretch > 1.05
                   else "true vertical scale"),
                  "dashed: a vertical from the foot, a tick every km",
                  "grey line on the floor: the axis's shadow"]
    if A.hurricane:
        paragraphs.insert(1, "the lowest pressure, not the most turning: a hurricane's eye can be wider than the "
                             "circle, and the most turning then sits in the eyewall around it")
    for side, k in (("below", A.ks[0]), ("above", A.ks[-1])):
        if A.stops[side]:
            paragraphs.append(f"{side} {S.fmt_height(float(f.zh[k]))} {STOPS[A.stops[side]]}, so the axis stops "
                              "there")
    yy = y + ov.U(272)
    for n, para in enumerate(paragraphs):
        for line in ov.wrap(para, "micro", ov.U(330)):
            ov.text((x, yy), line, "micro", colour="muted", anchor="ls")
            yy += int(ms * 1.5)
        if n == 0 or (A.hurricane and n == 1):
            yy += int(ms * 0.8)
    # the ruler's labels go on the side of the vertical away from the axis, each on a dark plate so it reads over
    # a bright slice
    right = float(np.mean(proj[:, 0] + sx0)) < float(pb[0])
    for zk in ticks_km[::2]:
        p = cam.project(foot + [0.0, 0.0, zk * stretch])[0]
        box = ov.text((p[0] + (ov.U(9) if right else -ov.U(9)), p[1]), f"{zk:g} km", "micro", colour="muted",
                      anchor="lm" if right else "rm", halo=False)
        _plate(cv, box, ov.U(4))
    for k, zd in heights.items():
        p = cam.project(np.array([hk, -hk, zd]))[0]
        ov.text((p[0] + ov.U(10), p[1]), f"spin at {S.fmt_height(float(f.zh[k]))}", "micro", colour="data",
                anchor="lm")
    # the top's height beside the axis's top, on the side away from the ruler's labels
    top_px = proj[-1] + shift
    ov.text((top_px[0] + (-ov.U(12) if right else ov.U(12)), top_px[1]), S.fmt_height(float(f.zh[A.ks[-1]])),
            "micro", colour="gold", anchor="rm" if right else "lm")
    ov.text((plan.x, plan.y - ov.U(10)), "the axis seen from above", "label", colour="text", anchor="ls")
    ov.text((plan.x + plan.w, plan.y - ov.U(10)), "north up", "micro", colour="data", anchor="rs")
    ov.text((plan.x, plan.y + plan.h + ov.U(20)),
            f"ring: the circle at the foot, {S.fmt_km(A.r)} in radius; a dot every km of height", "micro",
            colour="muted", anchor="ls", halo=False)
    P.scale_bar(ov, plan.x + ov.U(10), plan.y + plan.h - ov.U(14), plan)
    _end_labels(ov, plan, pts, (S.fmt_height(zlow), S.fmt_height(float(f.zh[A.ks[-1]]))))
    _chart_words(ov, chart, A, f)
    by = sy1 - ov.U(34)
    ov.ramp_bar("zeta", (sx1 - ov.U(330), by, sx1, by + max(3, ov.U(7))), vmin=-rng, vmax=rng,
                ticks=[(-rng, S.fmt_num(-rng, 3 if rng < 0.1 else 2)), (0.0, "0"), (rng, f"{rng:g} {S.PER_S}")],
                title="spin on the slices: clockwise | counter-clockwise")
    P.save(ov, cv, out)
    return {"radius_m": A.r, "start": A.start_words, "start_level_m": float(f.zh[A.kref]),
            "levels_m": [float(f.zh[k]) for k in A.ks],
            "x_km": [(ox + v) / 1000.0 for v in pts[:, 0]], "y_km": [(oy + v) / 1000.0 for v in pts[:, 1]],
            "circulation_m2_s": [float(v) for v in pts[:, 3]], "lean_km": lean / 1000.0, "lean_toward": toward,
            "lean_levels_m": [zlow, z5], "w_mean_start_m_s": A.w_ref, "spin_range_per_s": rng,
            "zeta_source": A.zeta_source, "vertical_stretch": round(stretch, 3),
            "slices_m": [float(f.zh[k]) for k in slices], "method": A.method_words,
            "rule": {"km_per_km": TILT_MAX, "plus_m": A.slack, "between": "every two heights"},
            "stops": {side: (STOPS[why] if why else None) for side, why in A.stops.items()}}


def _plan_px(box: "P.Box", p: np.ndarray) -> tuple[float, float]:
    return box.to_px(float(p[0]), float(p[1]))


def _plate(cv: np.ndarray, box: tuple[int, int, int, int], pad: int) -> None:
    """Darken the picture under a label (its box, padded), so it reads over a bright slice."""

    x0, y0, x1, y1 = box
    ys, ye = max(0, y0 - pad), min(cv.shape[0], y1 + pad)
    xs, xe = max(0, x0 - pad), min(cv.shape[1], x1 + pad)
    if ye > ys and xe > xs:
        void = np.array(S.rgb8("void"), np.float32)
        cv[ys:ye, xs:xe] = (cv[ys:ye, xs:xe].astype(np.float32) * 0.25 + void * 0.75 + 0.5).astype(np.uint8)


#: The places tried for a label around a point, as directions (right, left, up, down and the diagonals).
_AROUND = ((1, 0), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 1), (-1, -1), (-1, 1))


def _end_labels(ov: S.Overlay, box: "P.Box", pts: np.ndarray, words: tuple[str, str]) -> None:
    """The foot's and the top's heights beside their ends of the axis seen from above: each where it covers
    neither the line, nor its dots, nor the other label, nor the scale bar, preferring the side away from the
    line, inside the panel."""

    line = np.array([_plan_px(box, p) for p in pts], np.float64)
    samples = [line[:1]]
    for a, b in zip(line[:-1], line[1:]):
        n = max(2, int(math.hypot(*(b - a)) / 2.0) + 1)
        samples.append(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None])
    samples = np.concatenate(samples)
    taken = [(box.x, box.y + box.h - ov.U(34), box.x + ov.U(220), box.y + box.h)]     # the scale bar
    gap, clear = ov.U(9), ov.U(5)
    for end, text in ((0, words[0]), (len(line) - 1, words[1])):
        px, py = line[end]
        w, asc, desc = ov.measure(text, "micro")
        h = asc + desc
        # the direction away from the rest of the line, from this end
        back = line[min(3, len(line) - 1)] if end == 0 else line[max(0, len(line) - 4)]
        away = np.array([px, py]) - back
        away = away / (np.hypot(*away) or 1.0)
        order = sorted(_AROUND, key=lambda d: -(d[0] * away[0] + d[1] * away[1]) / math.hypot(*d))
        best, best_cost = None, None
        for d in order:
            x0 = px + gap if d[0] > 0 else px - gap - w if d[0] < 0 else px - w / 2.0
            y0 = py + gap if d[1] > 0 else py - gap - h if d[1] < 0 else py - h / 2.0
            x1, y1 = x0 + w, y0 + h
            cost = 0.0
            if x0 < box.x + 2 or x1 > box.x + box.w - 2 or y0 < box.y + 2 or y1 > box.y + box.h - 2:
                cost += 1e6
            hit = ((samples[:, 0] >= x0 - clear) & (samples[:, 0] <= x1 + clear)
                   & (samples[:, 1] >= y0 - clear) & (samples[:, 1] <= y1 + clear))
            cost += float(hit.sum())
            for tx0, ty0, tx1, ty1 in taken:
                if x0 - clear < tx1 and tx0 < x1 + clear and y0 - clear < ty1 and ty0 < y1 + clear:
                    cost += 1e4
            if best_cost is None or cost < best_cost:
                best, best_cost = (x0, y0, x1, y1), cost
            if cost == 0.0:
                break
        x0, y0, x1, y1 = best
        ov.text((x0, y0 + asc), text, "micro", colour="gold", anchor="ls")
        taken.append((x0, y0, x1, y1))


def _plan(cv: np.ndarray, box: "P.Box", pts: np.ndarray, colours: list[str], A: Axis, f: Any) -> np.ndarray:
    """The axis from above: its centres coloured by height over a faint grid, with the circle at its foot."""

    from PIL import Image, ImageDraw

    ss = 3
    im = Image.new("RGBA", (box.w * ss, box.h * ss), S.rgb8("void") + (255,))
    d = ImageDraw.Draw(im)
    every = P.nice_step(box.span, 6)
    for v in np.arange(math.ceil(box.mx0 / every) * every, box.mx0 + box.span, every):
        x = (v - box.mx0) / box.mpp * ss
        d.line((x, 0, x, box.h * ss), fill=S.rgb8("rule"), width=ss)
    for v in np.arange(math.ceil(box.my0 / every) * every, box.my0 + box.span_y, every):
        y = (box.my0 + box.span_y - v) / box.mpp * ss
        d.line((0, y, box.w * ss, y), fill=S.rgb8("rule"), width=ss)

    def px(xm: float, ym: float) -> tuple[float, float]:
        return (xm - box.mx0) / box.mpp * ss, (box.my0 + box.span_y - ym) / box.mpp * ss

    x0, y0 = px(pts[0, 0], pts[0, 1])
    rr = A.r / box.mpp * ss
    d.ellipse((x0 - rr, y0 - rr, x0 + rr, y0 + rr), outline=S.rgb8("faint"), width=ss)
    line = [px(p[0], p[1]) for p in pts]
    for n in range(len(line) - 1):
        d.line((*line[n], *line[n + 1]), fill=S.rgb8(colours[n], 1.0), width=3 * ss)
    # a dot at the level nearest each whole km
    for km in range(0, int(AXIS_TOP_M / 1000.0) + 1):
        n = int(np.argmin(np.abs(pts[:, 2] - km * 1000.0)))
        if abs(pts[n, 2] - km * 1000.0) <= 0.5 * _spacing(f, pts[n, 2]) + 1e-6:
            x, y = line[n]
            d.ellipse((x - 4 * ss, y - 4 * ss, x + 4 * ss, y + 4 * ss), fill=S.rgb8(colours[n], 1.0))
    small = np.asarray(im.resize((box.w, box.h), Image.LANCZOS).convert("RGB"))
    cv[box.y:box.y + box.h, box.x:box.x + box.w] = small
    return P.outline(cv, box)


def _spacing(f: Any, z: float) -> float:
    k = int(np.argmin(np.abs(f.zh - z)))
    return float(f.zf[k + 1] - f.zf[k])


def _chart_box(chart: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = chart
    return x0 + 56, y0 + 6, x1 - 8, y1 - 30


def _chart_scale(A: Axis) -> float:
    """The chart's right end, in units of 10^4 m2/s: a round number above the largest circulation."""

    top = max(float(max(v[2] for v in A.centres.values())), A.min_circulation) / 1e4 * 1.05
    step = P.nice_step(top, 2)
    return math.ceil(top / step) * step


def _chart(cv: np.ndarray, chart: tuple[int, int, int, int], A: Axis, f: Any) -> np.ndarray:
    """Circulation (x) against height (y), with the least the axis needs as a dashed line."""

    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = _chart_box(chart)
    gmax = _chart_scale(A) * 1e4
    im = Image.fromarray(cv)
    d = ImageDraw.Draw(im)
    d.line((x0, y1, x1, y1), fill=S.rgb8("faint"), width=1)
    d.line((x0, y0, x0, y1), fill=S.rgb8("faint"), width=1)
    xt = x0 + A.min_circulation / gmax * (x1 - x0)
    for y in range(y0, y1, 8):
        d.line((xt, y, xt, min(y + 4, y1)), fill=S.rgb8("dim"), width=1)
    pts = [(x0 + max(A.centres[k][2], 0.0) / gmax * (x1 - x0), y1 - float(f.zh[k]) / AXIS_TOP_M * (y1 - y0))
           for k in A.ks]
    for n in range(len(pts) - 1):
        z = float(f.zh[A.ks[n]])
        d.line((*pts[n], *pts[n + 1]), fill=S.rgb8(HEIGHT_COLOURS.at(min(1.0, z / AXIS_TOP_M)), 1.0), width=3)
    return np.asarray(im).copy()


def _chart_words(ov: S.Overlay, chart: tuple[int, int, int, int], A: Axis, f: Any) -> None:
    x0, y0, x1, y1 = _chart_box(chart)
    top = _chart_scale(A)
    ov.text((chart[0], chart[1] - ov.U(14)), "circulation by height", "label", colour="text", anchor="ls")
    for zk in (0.0, 5.0, 10.0):
        ov.text((x0 - ov.U(8), y1 - zk / 10.0 * (y1 - y0)), f"{zk:g} km", "micro", colour="muted", anchor="rm",
                halo=False)
    for v in (0.0, top / 2.0, top):
        ov.text((x0 + v / top * (x1 - x0), y1 + ov.U(18)), f"{v:g}", "micro", colour="muted", anchor="ms",
                halo=False)
    ov.text((x1, y1 + ov.U(36)), f"× 10⁴ m²/s around the circle of {S.fmt_km(A.r)} radius; dashed: the least it "
            "needs",
            "micro", colour="muted", anchor="rs", halo=False)


__all__ = ["AXIS_TOP_M", "Axis", "Camera", "axis_spin_range", "circle_radius", "disk_sum", "product_rotation_axis"]
