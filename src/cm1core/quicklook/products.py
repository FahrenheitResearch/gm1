"""The quick-look pictures: 1920 x 1080 PNGs in the viz 2.0 quick-look style, one CPU core, no GPU.

Generalised from the viz 2.0 quick-look code for any run CM1 users bring: every word comes from the run's records
(:mod:`.facts`), the domain's extent from the frame's own grid (a 64 x 76.8 km domain is drawn 64 x 76.8, not
squeezed square), and these picture rules hold on every product:

* the footer carries SIMULATED, the GM1 version, the microphysics and its checked state, dx, the real height of the
  level shown, the wind frame and T+;
* reflectivity only from the scheme's own operator or CM1's calc_refl10cm (:mod:`.reflectivity`);
* no smoothing beyond bilinear filtering: fields are sampled bilinearly, with no glow, bloom or blur;
* NWS reflectivity colours, and fixed, stated, symmetric ranges for signed fields;
* no tornado wording; :func:`.style.lint` runs on every label, and a picture that fails it is not written.

Products (id, title):

``reflectivity``      Reflectivity at 1 km: at 1 km above ground and the column maximum
``spin-near-ground``  Spin near the ground: vertical vorticity at the lowest model level (a range fixed for each
                      grid spacing, +-0.05 1/s at 250 m: :func:`spin_range`), the ground-relative wind there drawn
                      as streaks
``updraft-5km``       Updraft at 5 km: w on the w level nearest 5 km, and 2-5 km updraft helicity (CM1's uh)
``swath``             Wind swath since the start: the strongest lowest-level wind each place on the ground has had,
                      and the updraft-helicity track (:mod:`.swath`)
``surface``           Lowest level θ′ and wind: the cold pool, ground-relative streaks, gust fronts (convergence)
``rain``              Accumulated rain

A hurricane run (:attr:`.facts.RunFacts.storm_kind`, CM1's swirl start) is drawn with its own looks and ranges:

``wind-lowest``       Wind at the lowest level: the ground-relative wind speed, 0 to 80 m/s, with streaks and a
                      line at hurricane force (33 m/s)
``pressure-lowest``   Pressure at the lowest level: how far it is below the pressure at the area's edge, 0 to
                      70 hPa on a square-root scale, with a line every 4 hPa (the lowest model level, not sea
                      level, and it says so)

These two, the reflectivity and the rain (1 mm to 400 cm on a logarithmic scale) each show the whole area and,
beside it, the inner 500 km around the area's middle (:func:`hurricane_frame`); the slice through the strongest
updraft is drawn on a +-10 m/s range.  The looks made for a single
thunderstorm (:data:`THUNDERSTORM_ONLY`: the spin near the ground, the updraft at 5 km with its updraft helicity,
the wind swath, the cold pool, the hail and spin tracks) are not drawn for a hurricane, with one plain reason;
the two hurricane looks are not drawn for any other run.
"""

from __future__ import annotations

from functools import cached_property
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import reflectivity as R
from . import style as S
from .facts import RunFacts
from .frame import QuickFrame

W, H = 1920, 1080
THUMB = (480, 270)
PRODUCTS: tuple[tuple[str, str], ...] = (
    ("reflectivity", "Reflectivity at 1 km"),
    ("spin-near-ground", "Spin near the ground"),
    ("updraft-5km", "Updraft at 5 km"),
    ("swath", "Wind swath since the start"),
    ("surface", f"Lowest level θ{S.PRIME} and wind"),
    ("rain", "Accumulated rain"),
    # More looks, drawn by the same worker for each frame (their modules draw them with this style)
    ("clouds-from-above", "Clouds seen from above"),
    ("updraft-slice", "Slice through the strongest updraft"),
    ("hail-track", "Hail near the ground so far"),
    ("spin-track", "Strongest near-ground spin so far"),
    # A hurricane run's own looks (not drawn for any other run)
    ("wind-lowest", "Wind at the lowest level"),
    ("pressure-lowest", "Pressure at the lowest level"),
)
TITLES = dict(PRODUCTS)
#: Looks drawn only for a hurricane run, and the looks made for one thunderstorm that a hurricane run leaves out.
HURRICANE_ONLY = ("wind-lowest", "pressure-lowest")
THUNDERSTORM_ONLY = ("spin-near-ground", "updraft-5km", "swath", "surface", "hail-track", "spin-track")
THUNDERSTORM_ONLY_WHY = ("This look is made for thunderstorms. For a hurricane, see the wind and the pressure at "
                         "the lowest level.")
#: A hurricane's fixed, stated ranges.  They hold GM1's and CM1's runs of the hurricane presets
#: (records/GM1-HURRICANE-RUNS.json; measured): at most 70 m/s (GM1's 4 km run), 61 hPa below the area's edge
#: and 366 cm of rain in one cell (GM1's 8 km run over its 4 days).  A value above a range is drawn in the
#: range's last colour and the readout beside the picture says the number.
HURRICANE_WIND_RANGE = 80.0      # m/s, ground-relative, at the lowest level
HURRICANE_PRESSURE_RANGE = 70.0  # hPa below the pressure at the area's edge
HURRICANE_RAIN_RANGE = 400.0     # cm
HURRICANE_RAIN_FLOOR = 0.1       # cm (1 mm): where a hurricane's rain colours begin, on a logarithmic scale
HURRICANE_PRESSURE_LINES = 4.0   # hPa between the lines of the pressure picture
#: A pressure line is drawn where it lies less than this far (m) from the next one; farther out the pressure is
#: nearly flat and a line would wander with every ripple.
HURRICANE_LINES_APART = 110_000.0
#: The wind speed from which a storm is called a hurricane (64 knots, 33 m/s), marked on the wind picture.
HURRICANE_FORCE = 33.0
#: The width of a hurricane picture's inner panel, m (the eyewall and the inner rain bands).
HURRICANE_INNER = 500_000.0
#: The slice through the strongest updraft, m/s.  A grown hurricane's eyewall rises at 4 to 10 m/s at 5 km (GM1's
#: runs from hour 54 on, measured), which a thunderstorm's range of 50 or one of 20 draws nearly black; the
#: stronger bursts of the first two days (up to 41 m/s at 4 km) are drawn in the last colour, and the readout
#: beside the slice says their number.
HURRICANE_W_RANGE = 10.0
#: How many cells in from each side count as the area's edge (the far field the pressure is compared with).
EDGE_CELLS = 4

# Fixed, stated ranges (the same on every run of a grid spacing, so pictures compare).
SPIN_RANGE = 0.05       # 1/s, vertical vorticity at the lowest level at 250 m (spin_range scales it with dx)
W_RANGE = 50.0          # m/s
UH_RANGE = 1000.0       # m2/s2
THETA_RANGE = 8.0       # K
RAIN_RANGE = 8.0        # cm
SWATH_WIND_RANGE = 40.0  # m/s
REFLECTIVITY_HEIGHT = 1000.0
UPDRAFT_HEIGHT = 5000.0
SURFACE_LOOK = dict(conv_floor=0.0025, conv_ref=0.006, conv_gamma=0.9, conv_gain=0.42, conv_clip=4.0,
                    lic_px=12, lic_gain=0.26, speed_ref=25.0, theta_gain=0.62)


def spin_range(dx_m: float) -> float:
    """The fixed spin range for a grid spacing: ±0.01 1/s at 1 km, scaled as 1/dx (resolved vorticity grows as
    the grid resolves smaller eddies) and rounded to the nearest 1-2-5 step: ±0.02 at 500 m, ±0.05 at 250 m (the
    plan's range), ±0.1 at 100 m, ±0.2 at 50 m.  CM1's stock 1 km supercell peaks near 0.009 1/s at the lowest
    level, so one range for every grid would draw the coarse runs black or saturate the fine ones."""

    raw = 0.01 * 1000.0 / max(float(dx_m), 1.0)
    exponent = math.floor(math.log10(raw))
    candidates = [f * 10.0 ** e for e in (exponent - 1, exponent, exponent + 1) for f in (1.0, 2.0, 5.0)]
    return float(min(candidates, key=lambda c: abs(math.log(c / raw))))


class LintError(ValueError):
    """A picture's words break the honesty rules; it is not written."""


# ----------------------------------------------------------------------------------------------- layout
class Box:
    """A panel of w x h pixels at (x, y) showing [mx0, mx0 + span] x [my0, my0 + span_y] metres, north up."""

    def __init__(self, x, y, w, mx0=0.0, my0=0.0, span=1.0, *, h=None):
        self.x, self.y = int(x), int(y)
        self.w = int(w)
        self.h = int(h if h is not None else w)
        self.mx0, self.my0, self.span = float(mx0), float(my0), float(span)
        self.mpp = self.span / self.w
        self.span_y = self.h * self.mpp

    def to_px(self, xm: float, ym: float) -> tuple[float, float]:
        return self.x + (xm - self.mx0) / self.mpp, self.y + (self.my0 + self.span_y - ym) / self.mpp

    def axes_m(self) -> tuple[np.ndarray, np.ndarray]:
        c = (np.arange(self.w) + 0.5) * self.mpp
        r = (np.arange(self.h) + 0.5) * self.mpp
        return self.mx0 + c, self.my0 + self.span_y - r


def fit_box(lx: float, ly: float, cx: int, y: int, w_max: int, h_max: int) -> Box:
    """A box showing [0, lx] x [0, ly], as large as fits in w_max x h_max pixels, centred on column cx."""

    mpp = max(lx / w_max, ly / h_max)
    w, h = int(round(lx / mpp)), int(round(ly / mpp))
    return Box(cx - w // 2, y, w, 0.0, 0.0, w * mpp, h=h)


def single_box(extent: tuple[float, float]) -> Box:
    return fit_box(extent[0], extent[1], W // 2, 165, 800, 800)


def dual_boxes(extent: tuple[float, float]) -> tuple[Box, Box]:
    b = fit_box(extent[0], extent[1], 0, 182, 740, 740)
    gap = 44
    x0 = (W - 2 * b.w - gap) // 2
    return (Box(x0, b.y, b.w, 0.0, 0.0, b.span, h=b.h), Box(x0 + b.w + gap, b.y, b.w, 0.0, 0.0, b.span, h=b.h))


def bilinear(field: np.ndarray, rows: np.ndarray, cols: np.ndarray, *, cval: float | None = None) -> np.ndarray:
    """Bilinear interpolation of a 2-D array at fractional (row, col) indices (numpy only).

    Outside the array the edge value is used, or ``cval`` when given."""

    f = np.asarray(field)
    ny, nx = f.shape
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)
    outside = None
    if cval is not None:
        outside = (r < 0.0) | (r > ny - 1) | (c < 0.0) | (c > nx - 1)
    r = np.clip(r, 0.0, ny - 1)
    c = np.clip(c, 0.0, nx - 1)
    r0 = np.minimum(np.floor(r).astype(np.intp), max(ny - 2, 0))
    c0 = np.minimum(np.floor(c).astype(np.intp), max(nx - 2, 0))
    fr = (r - r0).astype(np.float32)
    fc = (c - c0).astype(np.float32)
    r1 = np.minimum(r0 + 1, ny - 1)
    c1 = np.minimum(c0 + 1, nx - 1)
    out = (f[r0, c0] * (1 - fr) * (1 - fc) + f[r1, c0] * fr * (1 - fc)
           + f[r0, c1] * (1 - fr) * fc + f[r1, c1] * fr * fc).astype(np.float32)
    if outside is not None:
        out[outside] = cval
    return out


def sample(field: np.ndarray, box: Box, dx: float, dy: float, origin=(0.0, 0.0), cval: float | None = None) -> np.ndarray:
    """Bilinear samples of a scalar-point field under every pixel of ``box`` (the only filtering a quick-look does)."""

    xm, ym = box.axes_m()
    ci = (xm - origin[0]) / dx - 0.5
    cj = (ym - origin[1]) / dy - 0.5
    rr = np.broadcast_to(cj[:, None], (box.h, box.w))
    cc = np.broadcast_to(ci[None, :], (box.h, box.w))
    return bilinear(field, rr, cc, cval=cval)


def canvas() -> np.ndarray:
    return np.tile(np.array(S.rgb8("void"), np.uint8), (H, W, 1))


def paste(cv: np.ndarray, img: np.ndarray, box: Box) -> None:
    cv[box.y:box.y + box.h, box.x:box.x + box.w] = img


def outline(cv: np.ndarray, box: Box, colour: str = "faint") -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(cv)
    ImageDraw.Draw(im).rectangle((box.x - 1, box.y - 1, box.x + box.w, box.y + box.h),
                                 outline=S.rgb8(S.token(colour)), width=1)
    return np.asarray(im).copy()


def nice_step(span_m: float, target: int = 10) -> float:
    """A 1-2-5 step (m) giving about ``target`` intervals over ``span_m``."""

    raw = span_m / max(1, target)
    mag = 10 ** math.floor(math.log10(raw))
    for f in (1, 2, 5, 10):
        if f * mag >= raw:
            return f * mag
    return 10 * mag


def ground_ticks(img: np.ndarray, box: Box, domainlocx: float, domainlocy: float, *, alpha: float = 0.22) -> tuple[np.ndarray, float]:
    """Faint '+' marks fixed to the ground (ground x = domain x + domainlocx); returns the image and the spacing."""

    from PIL import Image, ImageDraw

    every = nice_step(max(box.span, box.span_y), 12)
    ss = 4
    mask = Image.new("L", (box.w * ss, box.h * ss), 0)
    d = ImageDraw.Draw(mask)
    arm, hw = 5 * ss, 0.6 * ss
    gx0, gy0 = box.mx0 + domainlocx, box.my0 + domainlocy
    for X in np.arange(math.ceil(gx0 / every) * every, gx0 + box.span, every):
        for Y in np.arange(math.ceil(gy0 / every) * every, gy0 + box.span_y, every):
            px = (X - gx0) / box.mpp * ss
            py = (box.span_y - (Y - gy0)) / box.mpp * ss
            d.rectangle((px - arm, py - hw, px + arm, py + hw), fill=255)
            d.rectangle((px - hw, py - arm, px + hw, py + arm), fill=255)
    a = np.asarray(mask.resize((box.w, box.h), Image.BOX), np.float32) / 255 * alpha
    col = np.array(S.hex_rgb("data"), np.float32) * 255
    out = img.astype(np.float32) * (1 - a[..., None]) + col * a[..., None]
    return (out + 0.5).astype(np.uint8), every


def scale_bar(ov: S.Overlay, x: float, y: float, box: Box) -> None:
    length_m = nice_step(box.span, 6)
    L = length_m / box.mpp
    ov.rule(x, y, L, thickness=2, colour="data", alpha=0.85)
    ov.text((x + L + ov.U(10), y + ov.U(1)), S.fmt_km(length_m), "micro", colour="data", anchor="lm")


def panel_label(ov: S.Overlay, box: Box, text: str, right: str | None = None, *, vertical: bool = False) -> None:
    ov.text((box.x, box.y - ov.U(10)), text, "label", colour="text", anchor="ls")
    if right:
        ov.text((box.x + box.w, box.y - ov.U(10)), right, "label", colour="data", anchor="rs", vertical=vertical)


def dbz_legend(ov: S.Overlay, x: float, y: float, w: float) -> None:
    from PIL import Image, ImageDraw

    h = max(3, ov.U(8))
    n = len(S.NWS_REFLECTIVITY)
    cw = w / n
    strip = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(strip)
    for i, (_, c) in enumerate(S.NWS_REFLECTIVITY):
        d.rectangle((int(i * cw), 0, int((i + 1) * cw) - 1, h), fill=S.rgb8(c, 1.0))
    ov.draw_rgba(strip, (int(x), int(y)))
    ms = ov.px("micro")
    for v in (5, 20, 35, 50, 65):
        i = [b for b, _ in S.NWS_REFLECTIVITY].index(v)
        ov.text((x + i * cw, y + h + ov.U(8) + ms), f"{v}", "micro", anchor="ls", halo=False)
    ov.text((x + w, y + h + ov.U(8) + ms), "dBZ", "micro", anchor="rs", halo=False)


def ramp_legend(ov: S.Overlay, name: str, box: Box, vmin: float, vmax: float, ticks, *, title: str | None = None,
                y: float | None = None, width: int = 300, vertical: bool = False) -> None:
    by = box.y + box.h + ov.U(12) if y is None else y
    ov.ramp_bar(name, (box.x + box.w - ov.U(width), by, box.x + box.w, by + max(3, ov.U(7))), vmin=vmin, vmax=vmax,
                ticks=ticks, title=title, vertical=vertical)


def lic(noise_img: np.ndarray, du: np.ndarray, dv: np.ndarray, L: int) -> np.ndarray:
    """Line-integral convolution of ``noise_img`` along the unit field (du right, dv down, in pixels), Hann kernel.

    A texture that shows where the wind blows; it draws no data values."""

    Hh, Ww = noise_img.shape
    acc = noise_img.astype(np.float32).copy()
    wsum = np.ones_like(acc)
    for sgn in (1.0, -1.0):
        x = np.tile(np.arange(Ww, dtype=np.float32), (Hh, 1))
        y = np.repeat(np.arange(Hh, dtype=np.float32)[:, None], Ww, axis=1)
        for s in range(1, L + 1):
            ix = np.clip(np.rint(x).astype(np.int32), 0, Ww - 1)
            iy = np.clip(np.rint(y).astype(np.int32), 0, Hh - 1)
            x += sgn * du[iy, ix]
            y += sgn * dv[iy, ix]
            w = 0.5 + 0.5 * math.cos(math.pi * s / (L + 1))
            acc += w * bilinear(noise_img, y, x)
            wsum += w
    return acc / wsum


def noise(h: int, w: int, seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).random((h, w)).astype(np.float32)


def streaks(u: np.ndarray, v: np.ndarray, box: Box, *, px: int = 12, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """(texture around 0, speed) of a wind already sampled onto ``box``'s pixels."""

    sp = np.hypot(u, v) + 1e-6
    L = max(4, int(round(px * box.h / 800)))
    tx = lic(noise(box.h, box.w, seed), u / sp, -v / sp, L)
    return (tx - 0.5) / 0.12, sp


# ----------------------------------------------------------------------------------------------- one frame
class Looks:
    """The fields of one frame the products share, each computed once."""

    def __init__(self, qf: QuickFrame, facts: RunFacts) -> None:
        self.f = qf
        self.facts = facts

    # -- words -------------------------------------------------------------------
    def data_line(self, *more: str) -> str:
        lx, ly = self.f.extent
        return S.data_line(self.facts.name, f"{S.fmt_grid(self.f.dx)} grid",
                           f"whole {lx / 1000:g} × {ly / 1000:g} km domain", *more)

    def frame_words(self) -> str:
        if not self.f.domain_moves:
            return "domain fixed to the ground"
        return f"domain moving at {S.fmt_num(self.f.umove, 1)}/{S.fmt_num(self.f.vmove, 1)} m/s ground-relative"

    @property
    def z0(self) -> float:
        return float(self.f.zh[0])

    @property
    def hurricane(self) -> bool:
        """Whether this is a hurricane run (CM1's swirl start), which has its own looks and ranges."""

        return self.facts.storm_kind == "hurricane"

    @cached_property
    def pressure0_hpa(self) -> np.ndarray | None:
        """Pressure at the lowest model level, hPa, or None when the frame carries no pressure."""

        prs = self.f.pressure_level(0)
        return None if prs is None else (prs / np.float32(100.0)).astype(np.float32)

    @cached_property
    def pressure_edge_hpa(self) -> float | None:
        """The mean pressure at the lowest level over the area's outer :data:`EDGE_CELLS` cells (the far field)."""

        p = self.pressure0_hpa
        if p is None:
            return None
        n = max(1, min(EDGE_CELLS, min(p.shape) // 4))
        edge = np.ones(p.shape, dtype=bool)
        edge[n:-n, n:-n] = False
        return float(p[edge].mean(dtype=np.float64))

    # -- shared fields --------------------------------------------------------------
    @cached_property
    def reflectivity(self) -> R.Reflectivity:
        return R.volume(self.f)

    @cached_property
    def dbz_1km(self) -> tuple[np.ndarray, tuple[float, float]]:
        return R.at_height(self.reflectivity.dbz, self.f.zh, REFLECTIVITY_HEIGHT)

    @cached_property
    def dbz_column(self) -> np.ndarray:
        return self.reflectivity.dbz.max(axis=0)

    @cached_property
    def uv0_ground(self) -> tuple[np.ndarray, np.ndarray]:
        return self.f.uv_level(0, ground=True)

    @cached_property
    def wind0(self) -> np.ndarray:
        u, v = self.uv0_ground
        return np.hypot(u, v).astype(np.float32)

    @cached_property
    def zeta0(self) -> tuple[np.ndarray, str]:
        return self.f.zeta_level(0)

    @cached_property
    def theta0(self) -> np.ndarray:
        return self.f.theta_prime(0)

    @cached_property
    def conv0(self) -> np.ndarray:
        return self.f.conv_level(0)

    @cached_property
    def k5(self) -> int:
        return self.f.level_near(UPDRAFT_HEIGHT, faces=True)

    @cached_property
    def w5(self) -> np.ndarray:
        return self.f.get("w3d", self.k5)

    @cached_property
    def uh(self) -> tuple[np.ndarray, str]:
        return self.f.uh()

    @cached_property
    def w_extremes(self) -> tuple[float, float]:
        w = self.f.view("w3d")
        hi, lo = -np.inf, np.inf
        for k in range(w.shape[0]):
            plane = w[k]
            hi = max(hi, float(plane.max()))
            lo = min(lo, float(plane.min()))
        return hi, lo

    def cell_km2(self) -> float:
        return self.f.dx * self.f.dy / 1.0e6

    def statistics(self) -> dict[str, float]:
        """Domain extremes of this frame, with CM1's statistic names where CM1 has them (the time series)."""

        wmax, wmin = self.w_extremes
        th = self.theta0
        row: dict[str, float] = {
            "wmax": wmax, "wmin": wmin,
            "uh_max": float(self.uh[0].max()),
            "thpert_sfc_min": float(th.min()),
            "coldpool_area_km2_lt_minus1K": float((th < -1.0).sum() * self.cell_km2()),
            "zeta_sfc_max": float(self.zeta0[0].max()),
            "wind_sfc_max": float(self.wind0.max()),
        }
        if self.f.has("rain"):
            row["rain_max_cm"] = float(self.f.get("rain").max())
        if self.hurricane and self.pressure0_hpa is not None:
            # A hurricane's time series leads with its lowest pressure (at the lowest model level, not sea level).
            row["prs_sfc_min_hpa"] = float(self.pressure0_hpa.min())
        if "reflectivity" in self.__dict__ and self.reflectivity.available:
            row["dbz_max"] = float(self.dbz_column.max())
        return row


def save(ov: S.Overlay, base: np.ndarray, path: Path) -> tuple[Path, Path]:
    """Lint, composite and write the PNG and its thumbnail, each under a temporary name then renamed."""

    from PIL import Image

    problems = ov.check()
    if problems:
        raise LintError(f"{path.name}: " + "; ".join(problems))
    if ov.footer_cut:
        import warnings

        warnings.warn(f"viz20 footer of {path.name} cut to two lines, dropping: {ov.footer_cut}")
    img = Image.fromarray(ov.composite(base))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    img.save(tmp, format="PNG", compress_level=6)
    os.replace(tmp, path)
    thumb = path.parent / "thumb" / path.name
    thumb.parent.mkdir(exist_ok=True)
    ttmp = thumb.with_name(f".{thumb.name}.partial")
    img.resize(THUMB, Image.LANCZOS).save(ttmp, format="PNG", compress_level=6)
    os.replace(ttmp, thumb)
    return path, thumb


#: A hurricane picture's clock (hours and days, as Watching and Results say it).
long_clock = S.fmt_long_clock


def overlay(L: Looks, product: str, data: str, *, level: str, wind: str, notes: str,
            notes_vertical: bool = False) -> S.Overlay:
    """The frame grammar of one look.  ``notes_vertical``: the notes' speeds are vertical velocities (the lint's
    wind-frame rule is about horizontal wind; a domain fixed to the ground names no frame in the notes)."""

    ov = S.Overlay((W, H))
    ov.title(TITLES[product], data)
    # A hurricane picture's clock reads as the page around it does (hours and days); every other run's is H:MM:SS.
    ov.clock(long_clock(L.f.t) if L.hurricane else L.f.t)
    ov.footer(L.facts.footer(t=L.f.t, level=level, wind=wind, notes=notes), vertical=[notes] if notes_vertical else ())
    ov.mark()
    return ov


def dbz_image(d: np.ndarray) -> np.ndarray:
    c = S.dbz_colours(d)
    void = np.array(S.hex_rgb("void"), np.float32)
    return S.to8(c[..., :3] * c[..., 3:4] + void * (1 - c[..., 3:4]))


def dbz_max_text(d: np.ndarray) -> str:
    m = float(np.max(d))
    return f"max {m:.0f} dBZ" if m >= 5 else "nothing above 5 dBZ"


# ----------------------------------------------------------------------------------------------- products
def product_reflectivity(L: Looks, out: Path) -> dict[str, Any]:
    f = L.f
    refl = L.reflectivity
    if not refl.available:
        raise NotDrawn(refl.reason or "no reflectivity for this microphysics")
    if L.hurricane:
        return _reflectivity_hurricane(L, out)
    one, (lo, hi) = L.dbz_1km
    col = L.dbz_column
    b1, b2 = dual_boxes(f.extent)
    cv = canvas()
    spacing = 0.0
    for b, d in ((b1, one), (b2, col)):
        img, spacing = ground_ticks(dbz_image(sample(d, b, f.dx, f.dy)), b, f.domainlocx, f.domainlocy)
        paste(cv, img, b)
    cv = outline(outline(cv, b1), b2)
    between = f"1 km: linear in Z between the {S.fmt_height(lo)} and {S.fmt_height(hi)} levels"
    dense = (f.config.get("dense_ice") or {}).get("category")
    dense = dense if dense in ("hail", "graupel") else "graupel"
    ov = overlay(L, "reflectivity", L.data_line(refl.operator),
                 level="1 km and column maximum", wind="",
                 notes=S.data_line(between, f"from the model's rain, snow and {dense}, not a radar", L.frame_words(),
                                   f"+ fixed to the ground every {S.fmt_km(spacing)}"))
    panel_label(ov, b1, "1 km above ground", dbz_max_text(one))
    panel_label(ov, b2, "column maximum", dbz_max_text(col))
    dbz_legend(ov, b2.x + b2.w - ov.U(300), b2.y + b2.h + ov.U(12), ov.U(300))
    scale_bar(ov, b1.x, b1.y + b1.h + ov.U(16), b1)
    save(ov, cv, out)
    return {"dbz_max_1km": float(one.max()), "dbz_max_column": float(col.max()), "operator": refl.operator,
            "levels_m": [lo, hi]}


def product_spin(L: Looks, out: Path) -> dict[str, Any]:
    f = L.f
    box = single_box(f.extent)
    zeta, zsource = L.zeta0
    u, v = L.uv0_ground
    zs = sample(zeta, box, f.dx, f.dy)
    us, vs = sample(u, box, f.dx, f.dy), sample(v, box, f.dx, f.dy)
    rng = spin_range(f.dx)
    base = S.RAMPS["zeta"].colours(zs, -rng, rng)
    tx, sp = streaks(us, vs, box, seed=5)
    a = (np.clip(0.55 + 0.45 * tx, 0.0, 2.0) * np.clip(sp / 25.0, 0.25, 1.4) * 0.22)[..., None]
    rgb = base * (1.0 - np.clip(a, 0, 0.6)) + np.array(S.hex_rgb("silk"), np.float32) * 1.6 * a
    img, spacing = ground_ticks(S.to8(rgb), box, f.domainlocx, f.domainlocy)
    cv = canvas()
    paste(cv, img, box)
    cv = outline(cv, box)
    z0 = L.z0
    ov = overlay(L, "spin-near-ground", L.data_line(f"lowest model level, {S.fmt_height(z0)} above ground"),
                 level=f"lowest level {S.fmt_height(z0)}", wind="ground-relative",
                 notes=S.data_line(f"spin: vertical vorticity ({zsource}), ±{rng:g} {S.PER_S} fixed at this dx",
                                   f"streaks: the {S.fmt_height(z0)} wind, ground-relative", L.frame_words(),
                                   f"+ fixed to the ground every {S.fmt_km(spacing)}"))
    j, i = np.unravel_index(int(np.argmax(zeta)), zeta.shape)
    zmax, zmin = float(zeta.max()), float(zeta.min())
    wind = float(L.wind0.max())
    x, y = ov.x0, box.y + ov.U(10)
    ov.readout((x, y), [(f"strongest spin at {S.fmt_height(z0)}", f"{S.fmt_num(zmax, 3)} {S.PER_S}"),
                        ("strongest opposite spin", f"{S.fmt_num(zmin, 3)} {S.PER_S}"),
                        (f"fastest wind at {S.fmt_height(z0)}", f"{wind:.0f} m/s ground-relative")])
    ov.text((x, y + ov.U(300)), "gold ring: the strongest spin", "micro", colour="gold", anchor="ls")
    ov.text((x, y + ov.U(324)), f"features smaller than about {S.fmt_km(6 * f.dx)} are not resolved", "micro",
            colour="muted", anchor="ls")
    px, py = box.to_px((i + 0.5) * f.dx, (j + 0.5) * f.dy)
    ov.ring((px, py), max(10.0, 3.0 * f.dx / box.mpp) + 6)
    scale_bar(ov, x, box.y + box.h - ov.U(8), box)
    wr = ov.U(400)
    by = box.y + box.h - ov.U(40)
    ov.ramp_bar("zeta", (ov.x1 - wr, by, ov.x1, by + max(3, ov.U(7))), vmin=-rng, vmax=rng,
                ticks=[(-rng, S.fmt_num(-rng, 2)), (0.0, "0"), (rng, f"{rng:.2f} {S.PER_S}")],
                title=f"spin at {S.fmt_height(z0)}: clockwise | counter-clockwise")
    save(ov, cv, out)
    return {"height_m": z0, "dx_m": f.dx, "resolved_m": 6.0 * f.dx, "zeta_max": zmax, "zeta_min": zmin,
            "zeta_source": zsource, "range_per_s": rng, "wind_max_ground_relative": wind,
            "zeta_max_xy_m": [float((i + 0.5) * f.dx), float((j + 0.5) * f.dy)]}


def product_updraft(L: Looks, out: Path) -> dict[str, Any]:
    f = L.f
    w5, k5 = L.w5, L.k5
    z5 = float(f.zf[k5])
    uh, uh_source = L.uh
    b1, b2 = dual_boxes(f.extent)
    cv = canvas()
    img1 = S.to8(S.colours("w_sym", sample(w5, b1, f.dx, f.dy), -W_RANGE, W_RANGE))
    img2 = S.to8(S.RAMPS["zeta"].colours(sample(uh, b2, f.dx, f.dy), -UH_RANGE, UH_RANGE))
    img1, spacing = ground_ticks(img1, b1, f.domainlocx, f.domainlocy)
    img2, _ = ground_ticks(img2, b2, f.domainlocx, f.domainlocy)
    paste(cv, img1, b1)
    paste(cv, img2, b2)
    cv = outline(outline(cv, b1), b2)
    ov = overlay(L, "updraft-5km", L.data_line("gold ring: the strongest updraft at 5 km"),
                 level=f"w level {S.fmt_height(z5)}", wind="",
                 notes=S.data_line(f"w on the {S.fmt_height(z5)} w level, not interpolated",
                                   f"ranges ±{W_RANGE:g} m/s and ±{UH_RANGE:g} m² s⁻² fixed",
                                   f"updraft helicity: {uh_source}", L.frame_words(),
                                   f"+ fixed to the ground every {S.fmt_km(spacing)}"),
                 notes_vertical=True)
    j, i = np.unravel_index(int(np.argmax(w5)), w5.shape)
    w_label = f"max {S.fmt_num(float(w5.max()))} m/s, min {S.fmt_num(float(w5.min()))} m/s"
    panel_label(ov, b1, f"vertical velocity w at {S.fmt_height(z5)}", w_label, vertical=True)
    panel_label(ov, b2, "2–5 km updraft helicity", f"max {S.fmt_num(float(uh.max()))} m² s⁻²")
    for b in (b1, b2):
        ov.ring(b.to_px((i + 0.5) * f.dx, (j + 0.5) * f.dy), max(12.0, 4.0 * f.dx / b.mpp) + 6)
    ramp_legend(ov, "w_sym", b1, -W_RANGE, W_RANGE,
                [(-W_RANGE, S.fmt_num(-W_RANGE)), (0, "0"), (W_RANGE, f"{W_RANGE:.0f} m/s")], vertical=True)
    ramp_legend(ov, "zeta", b2, -UH_RANGE, UH_RANGE,
                [(-UH_RANGE, S.fmt_num(-UH_RANGE)), (0, "0"), (UH_RANGE, f"{UH_RANGE:.0f} m² s⁻²")])
    scale_bar(ov, b1.x, b1.y + b1.h + ov.U(16), b1)
    save(ov, cv, out)
    return {"level_m": z5, "w_max": float(w5.max()), "w_min": float(w5.min()), "uh_max": float(uh.max()),
            "w_max_xy_m": [float((i + 0.5) * f.dx), float((j + 0.5) * f.dy)]}


def product_swath(L: Looks, out: Path, swath: Any) -> dict[str, Any]:
    """``swath`` is this frame's :class:`.swath.Swath` (the running maxima on the ground)."""

    f = L.f
    if swath is None:
        raise NotDrawn("the frame has no lowest-level wind to build a swath from")
    lx, ly = swath.extent_m
    gap = 44
    wmax = (W - 2 * 80 - gap) // 2
    mpp = max(lx / wmax, ly / 700.0)
    bw, bh = int(round(lx / mpp)), int(round(ly / mpp))
    x0 = (W - 2 * bw - gap) // 2
    b1 = Box(x0, 200, bw, swath.origin_m[0], swath.origin_m[1], bw * mpp, h=bh)
    b2 = Box(x0 + bw + gap, 200, bw, swath.origin_m[0], swath.origin_m[1], bw * mpp, h=bh)
    cv = canvas()
    s1 = sample(swath.wind, b1, swath.dx, swath.dy, origin=swath.origin_m)
    img1 = S.to8(S.colours("ice", s1, 0.0, SWATH_WIND_RANGE))
    paste(cv, img1, b1)
    facts: dict[str, Any] = {"wind_max_ground_relative": float(np.nanmax(swath.wind)), "source": swath.source,
                             "frames_used": swath.frames_used, "height_m": L.z0}
    if swath.uh is not None:
        s2 = sample(swath.uh, b2, swath.dx, swath.dy, origin=swath.origin_m)
        paste(cv, S.to8(S.colours("ember", s2, 0.0, UH_RANGE)), b2)
        facts["uh_max"] = float(np.nanmax(swath.uh))
    cv = outline(outline(cv, b1), b2)
    from PIL import Image, ImageDraw

    im = Image.fromarray(cv)
    d = ImageDraw.Draw(im)
    dom_x, dom_y = f.extent
    for b in (b1, b2):
        for (gx, gy), colour in (((0.0, 0.0), "dim"), ((f.domainlocx, f.domainlocy), "muted")):
            xa, ya = b.to_px(gx, gy + dom_y)
            xb, yb = b.to_px(gx + dom_x, gy)
            d.rectangle((xa, ya, xb, yb), outline=S.rgb8(colour), width=1)
    cv = np.asarray(im).copy()
    z0 = L.z0
    since = "since the start" if swath.source == "model" or swath.t0 <= 0.0 else f"since {S.fmt_clock(swath.t0)}"
    ov = overlay(L, "swath", S.data_line(L.facts.name, f"{S.fmt_grid(f.dx)} grid",
                                         f"on the ground, {lx / 1000:g} × {ly / 1000:g} km", f"strongest {since}"),
                 level=f"lowest level {S.fmt_height(z0)}", wind="ground-relative",
                 notes=S.data_line(swath.method, "dim box: the domain at the start, grey box: now",
                                   f"ranges 0–{SWATH_WIND_RANGE:g} m/s ground-relative and 0–{UH_RANGE:g} m² s⁻² "
                                   "fixed"))
    panel_label(ov, b1, f"strongest wind at {S.fmt_height(z0)} {since}",
                f"max {facts['wind_max_ground_relative']:.0f} m/s ground-relative")
    ramp_legend(ov, "ice", b1, 0.0, SWATH_WIND_RANGE,
                [(0, "0"), (SWATH_WIND_RANGE / 2, f"{SWATH_WIND_RANGE / 2:.0f}"),
                 (SWATH_WIND_RANGE, f"{SWATH_WIND_RANGE:.0f} m/s")], title="wind, ground-relative")
    facts["since"] = since
    if swath.uh is not None:
        panel_label(ov, b2, f"2–5 km updraft helicity {since}", f"max {S.fmt_num(facts['uh_max'])} m² s⁻²")
        ramp_legend(ov, "ember", b2, 0.0, UH_RANGE, [(0, "0"), (UH_RANGE / 2, f"{UH_RANGE / 2:.0f}"),
                                                     (UH_RANGE, f"{UH_RANGE:.0f} m² s⁻²")])
    else:
        ov.text((b2.x + b2.w / 2, b2.y + b2.h / 2), "no updraft-helicity track in this run's frames", "label",
                colour="muted", anchor="ms")
    scale_bar(ov, b1.x, b1.y + b1.h + ov.U(16), b1)
    save(ov, cv, out)
    return facts


def product_surface(L: Looks, out: Path) -> dict[str, Any]:
    f = L.f
    look = SURFACE_LOOK
    box = single_box(f.extent)
    th = L.theta0
    u, v = L.uv0_ground
    ths = sample(th, box, f.dx, f.dy)
    cs = sample(L.conv0, box, f.dx, f.dy)
    us, vs = sample(u, box, f.dx, f.dy), sample(v, box, f.dx, f.dy)
    hdr = S.colours("theta_sym", ths, -THETA_RANGE, THETA_RANGE) * look["theta_gain"]
    tx, sp = streaks(us, vs, box, px=look["lic_px"])
    sm = np.clip(sp / look["speed_ref"], 0.25, 1.6)
    hdr += (np.clip(0.55 + 0.45 * tx, 0.0, 2.5) * sm * look["lic_gain"])[..., None] * np.array(S.hex_rgb("silk"), np.float32)
    e = np.clip((cs - look["conv_floor"]) / look["conv_ref"], 0.0, look["conv_clip"]) ** look["conv_gamma"]
    hdr += (e * look["conv_gain"])[..., None] * np.array(S.hex_rgb("#ffe6bf"), np.float32)
    img, spacing = ground_ticks(S.finish(hdr), box, f.domainlocx, f.domainlocy)
    cv = canvas()
    paste(cv, img, box)
    cv = outline(cv, box)
    z0 = L.z0
    ov = overlay(L, "surface", L.data_line(f"lowest model level, {S.fmt_height(z0)} above ground"),
                 level=f"lowest level {S.fmt_height(z0)}", wind="ground-relative",
                 notes=S.data_line(f"θ{S.PRIME} = θ minus the base state, ±{THETA_RANGE:g} K fixed",
                                   f"streaks: the {S.fmt_height(z0)} wind, ground-relative, brighter where faster",
                                   "bright lines: convergence (gust fronts)", L.frame_words()))
    area = float((th < -1.0).sum() * L.cell_km2())
    wind = float(L.wind0.max())
    x, y = ov.x0, box.y + ov.U(10)
    ov.readout((x, y), [(f"coldest θ{S.PRIME} at {S.fmt_height(z0)}", f"{S.fmt_num(float(th.min()), 1)} K"),
                        (f"fastest wind at {S.fmt_height(z0)}", f"{wind:.0f} m/s ground-relative"),
                        (f"θ{S.PRIME} below {S.fmt_num(-1)} K", f"{area:,.0f} km²")])
    scale_bar(ov, x, box.y + box.h - ov.U(8), box)
    wr = ov.U(400)
    by = box.y + box.h - ov.U(40)
    ov.ramp_bar("theta_sym", (ov.x1 - wr, by, ov.x1, by + max(3, ov.U(7))), vmin=-THETA_RANGE, vmax=THETA_RANGE,
                ticks=[(-8, S.fmt_num(-8)), (-4, S.fmt_num(-4)), (0, "0"), (4, "+4"), (8, "+8 K")],
                title=f"θ{S.PRIME} at {S.fmt_height(z0)}")
    save(ov, cv, out)
    return {"height_m": z0, "theta_min": float(th.min()), "wind_max_ground_relative": wind, "coldpool_km2": area}


def product_rain(L: Looks, out: Path) -> dict[str, Any]:
    f = L.f
    if not f.has("rain"):
        raise NotDrawn("This run did not save the rain that reached the ground.")   # CM1 output_rain = 0
    rain = f.get("rain")
    if L.hurricane:
        return _rain_hurricane(L, out, rain)
    box = single_box(f.extent)
    rs = sample(rain, box, f.dx, f.dy)
    rgb = S.colours("rain_cm", rs, 0.0, RAIN_RANGE)
    img = S.to8(rgb)
    edge = rs >= 0.1
    ring = np.zeros_like(edge)
    ring[:, 1:] |= edge[:, 1:] != edge[:, :-1]
    ring[1:, :] |= edge[1:, :] != edge[:-1, :]
    img[ring] = S.rgb8("muted")
    img, spacing = ground_ticks(img, box, f.domainlocx, f.domainlocy)
    cv = canvas()
    paste(cv, img, box)
    cv = outline(cv, box)
    moving = ("CM1's rain accumulates on the moving domain, not on the ground (the swath is on the ground)"
              if f.domain_moves else "")
    ov = overlay(L, "rain", L.data_line("since the start"), level="surface", wind="",
                 notes=S.data_line(moving, "grey line: 1 mm", f"0–{RAIN_RANGE:g} cm fixed", L.frame_words()))
    area = float((rain >= 0.1).sum() * L.cell_km2())
    x, y = ov.x0, box.y + ov.U(10)
    ov.readout((x, y), [("most rain in one cell", f"{float(rain.max()):.1f} cm"),
                        ("area above 1 mm", f"{area:,.0f} km²")])
    scale_bar(ov, x, box.y + box.h - ov.U(8), box)
    wr = ov.U(400)
    by = box.y + box.h - ov.U(40)
    ov.ramp_bar("rain_cm", (ov.x1 - wr, by, ov.x1, by + max(3, ov.U(7))), vmin=0.0, vmax=RAIN_RANGE,
                ticks=[(0, "0"), (2, "2"), (4, "4"), (6, "6"), (8, "8 cm")], title="rain since the start")
    save(ov, cv, out)
    return {"rain_max_cm": float(rain.max()), "rain_area_km2": area, "domain_moves": bool(f.domain_moves)}


# ----------------------------------------------------------------------------------------------- hurricane looks
def hurricane_boxes(extent: tuple[float, float]) -> tuple[Box, Box, float]:
    """A hurricane look's two panels: the whole area, and beside it the inner part around the area's middle.

    The inner panel is :data:`HURRICANE_INNER` across (500 km), or half the area's shorter side on a smaller area;
    it is fixed to the middle of the area, not to the storm, so pictures of different times compare.  Returns the
    two boxes and the inner panel's width in metres."""

    whole, second = dual_boxes(extent)
    inner = min(HURRICANE_INNER, 0.5 * min(extent))
    zoom = Box(second.x, second.y, second.w, 0.5 * (extent[0] - inner),
               0.5 * (extent[1] - inner * second.h / second.w), inner, h=second.h)
    return whole, zoom, inner


def line_alpha(field: np.ndarray, level: float | None = None, *, every: float | None = None,
               width: float = 1.4, far: float = 240.0) -> np.ndarray:
    """How strongly a contour line covers each pixel (0..1) of a field already sampled onto a panel's pixels.

    ``level``: one line where the field crosses that value.  ``every``: a line at each multiple of it.  The
    distance to the nearest line is the difference divided by the field's slope, so the line is ``width`` pixels
    wide with soft edges.  Lines every ``every`` fade out where they would be closer than a few pixels (a steep
    eyewall), so they never merge into a block, and where they would be more than ``far`` pixels apart (the
    nearly flat far field, where a line wanders with every small ripple)."""

    v = np.asarray(field, dtype=np.float64)
    gy, gx = np.gradient(v)
    slope = np.hypot(gx, gy)
    safe = np.maximum(slope, 1e-12)
    if every is None:
        distance = np.abs(v - float(level)) / safe
        alpha = np.clip(0.5 + 0.5 * width - distance, 0.0, 1.0)
    else:
        distance = np.abs(v / every - np.rint(v / every)) * every / safe
        apart = every / safe
        alpha = (np.clip(0.5 + 0.5 * width - distance, 0.0, 1.0) * np.clip((apart - 3.0) / 4.0, 0.0, 1.0)
                 * np.clip((far - apart) / (0.4 * far), 0.0, 1.0))
    alpha[slope * max(v.shape) < 1e-3 * (every or 1.0)] = 0.0
    return alpha.astype(np.float32)


def blend(img: np.ndarray, alpha: np.ndarray, colour: str, strength: float = 1.0) -> np.ndarray:
    """``img`` (uint8) with ``colour`` laid over it where ``alpha`` says."""

    a = (np.clip(alpha, 0.0, 1.0) * strength)[..., None]
    return (img.astype(np.float32) * (1.0 - a) + np.array(S.rgb8(colour), np.float32) * a + 0.5).astype(np.uint8)


def frame_mark(cv: np.ndarray, whole: Box, zoom: Box) -> np.ndarray:
    """The inner panel's area marked on the whole-area panel with a thin square."""

    from PIL import Image, ImageDraw

    x0, y1 = whole.to_px(zoom.mx0, zoom.my0)
    x1, y0 = whole.to_px(zoom.mx0 + zoom.span, zoom.my0 + zoom.span_y)
    im = Image.fromarray(cv)
    ImageDraw.Draw(im).rectangle((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
                                 outline=S.rgb8("muted"), width=1)
    return np.asarray(im).copy()


def hurricane_bar(ov: S.Overlay, ramp: str, zoom: Box, ticks: Sequence[tuple[float, str]], what: str) -> None:
    """The colour bar under the inner panel, with what it measures to its left; ``ticks`` are (position 0..1
    along the bar, label)."""

    by = zoom.y + zoom.h + ov.U(12)
    x0 = zoom.x + zoom.w - ov.U(330)
    ov.ramp_bar(ramp, (x0, by, zoom.x + zoom.w, by + max(3, ov.U(8))), vmin=0.0, vmax=1.0, ticks=ticks)
    ov.text((x0 - ov.U(12), by + ov.U(9)), what, "micro", colour="data", anchor="rs", halo=False)


def hurricane_frame(L: Looks, product: str, draw: Callable[[Box], np.ndarray], *, level: str, wind: str,
                    notes: str, left: str, right: str | None, lines: Sequence[tuple[str, str]],
                    data: str | None = None) -> tuple[S.Overlay, np.ndarray, Box, Box, float]:
    """The shared grammar of a hurricane look: the whole area and the inner part side by side, a label and one
    number over each, a scale bar under each, and up to two small lines of explanation under the whole area.

    ``draw(box)`` returns the uint8 picture of one panel.  ``lines`` are (text, colour token)."""

    f = L.f
    whole, zoom, inner = hurricane_boxes(f.extent)
    cv = canvas()
    spacings = []
    for box in (whole, zoom):
        img, spacing = ground_ticks(draw(box), box, f.domainlocx, f.domainlocy)
        spacings.append(spacing)
        paste(cv, img, box)
    cv = frame_mark(outline(outline(cv, whole), zoom), whole, zoom)
    plus = f"+ every {S.fmt_km(spacings[0])} (whole area) and {S.fmt_km(spacings[1])} (inner)"
    ov = overlay(L, product, data or L.data_line(), level=level, wind=wind,
                 notes=S.data_line(notes, L.frame_words(), plus))
    panel_label(ov, whole, "whole area", left)
    panel_label(ov, zoom, f"inner {S.fmt_km(inner)} (the square on the left)", right)
    scale_bar(ov, whole.x, whole.y + whole.h + ov.U(16), whole)
    scale_bar(ov, zoom.x, zoom.y + zoom.h + ov.U(16), zoom)
    ms = ov.px("micro")
    for n, (text, colour) in enumerate(lines[:2]):
        ov.text((whole.x + whole.w, whole.y + whole.h + ov.U(12) + ms + n * int(ms * 1.55)), text, "micro",
                colour=colour, anchor="rs", halo=False)
    return ov, cv, whole, zoom, inner


def rain_position(cm: np.ndarray | float) -> np.ndarray:
    """Where an amount of rain sits on a hurricane's rain colours, 0..1: logarithmic from 1 mm to the range's
    top, so a tenfold step in rain is the same step in colour."""

    lo, hi = HURRICANE_RAIN_FLOOR, HURRICANE_RAIN_RANGE
    v = np.maximum(np.asarray(cm, dtype=np.float64), 1e-9)
    return np.clip(np.log(v / lo) / math.log(hi / lo), 0.0, 1.0)


def pressure_position(hpa: np.ndarray | float) -> np.ndarray:
    """Where a pressure drop sits on a hurricane's pressure colours, 0..1: the square root of its share of the
    range, so the broad shallow low around the storm shows as well as the deep centre."""

    return np.sqrt(np.clip(np.asarray(hpa, dtype=np.float64) / HURRICANE_PRESSURE_RANGE, 0.0, 1.0))


def _rain_hurricane(L: Looks, out: Path, rain: np.ndarray) -> dict[str, Any]:
    """The rain of a hurricane run: days of rain, from 1 mm to 400 cm on a logarithmic colour scale, over the
    whole area and over its inner part."""

    f = L.f
    top, floor = HURRICANE_RAIN_RANGE, HURRICANE_RAIN_FLOOR
    void = np.array(S.hex_rgb("void"), np.float32)

    def draw(box: Box) -> np.ndarray:
        rs = sample(rain, box, f.dx, f.dy)
        rgb = S.colours("hurricane_rain", rain_position(rs), 0.0, 1.0)
        rgb[rs < floor] = void
        return S.to8(rgb)

    area = float((rain >= floor).sum() * L.cell_km2())
    most = float(rain.max())
    j, i = np.unravel_index(int(np.argmax(rain)), rain.shape)
    east, north = _centre_km(L, int(i), int(j))
    ov, cv, whole, zoom, inner = hurricane_frame(
        L, "rain", draw, level="surface", wind="",
        notes=S.data_line(f"rain since the start, {floor * 10:g} mm to {top:g} cm fixed for a hurricane run",
                          "logarithmic colours"),
        data=L.data_line("since the start"),
        left=f"above 1 mm: {area:,.0f} km²", right=f"most in one cell {most:.0f} cm",
        lines=[(f"most rain: {_from_centre(east, north)}", "data"),
               ("no colour: less than 1 mm", "muted")])
    hurricane_bar(ov, "hurricane_rain", zoom,
                  [(float(rain_position(v)), label) for v, label in
                   ((floor, "1 mm"), (1.0, "1 cm"), (10.0, "10 cm"), (100.0, "1 m"), (top, f"{top / 100.0:g} m"))],
                  "rain since the start")
    save(ov, cv, out)
    return {"rain_max_cm": most, "rain_area_km2": area, "domain_moves": bool(f.domain_moves), "range_cm": top,
            "floor_cm": floor, "scale": "logarithmic", "inner_km": inner / 1000.0}


def _reflectivity_hurricane(L: Looks, out: Path) -> dict[str, Any]:
    """A hurricane run's reflectivity at 1 km: the whole area, and its inner part where the eyewall and the rain
    bands are."""

    f = L.f
    refl = L.reflectivity
    one, (lo, hi) = L.dbz_1km
    col = L.dbz_column
    dense = (f.config.get("dense_ice") or {}).get("category")
    dense = dense if dense in ("hail", "graupel") else "graupel"
    ov, cv, whole, zoom, inner = hurricane_frame(
        L, "reflectivity", lambda box: dbz_image(sample(one, box, f.dx, f.dy)),
        level="1 km above the sea", wind="",
        notes=S.data_line(f"1 km: linear in Z between the {S.fmt_height(lo)} and {S.fmt_height(hi)} levels",
                          f"from the model's rain, snow and {dense}, not a radar"),
        data=L.data_line(refl.operator),
        left=dbz_max_text(one), right=f"anywhere in the column: {dbz_max_text(col)}",
        lines=[("both panels: 1 km above the sea", "data"),
               (f"features smaller than about {S.fmt_km(6 * f.dx)} are not resolved", "muted")])
    dbz_legend(ov, zoom.x + zoom.w - ov.U(300), zoom.y + zoom.h + ov.U(12), ov.U(300))
    save(ov, cv, out)
    return {"dbz_max_1km": float(one.max()), "dbz_max_column": float(col.max()), "operator": refl.operator,
            "levels_m": [lo, hi], "inner_km": inner / 1000.0}


def _centre_km(L: Looks, i: int, j: int) -> tuple[float, float]:
    """Where cell (j, i) is from the middle of the area, km east and km north."""

    f = L.f
    return ((i + 0.5) * f.dx - 0.5 * f.extent[0]) / 1000.0, ((j + 0.5) * f.dy - 0.5 * f.extent[1]) / 1000.0


#: The eight compass words, counter-clockwise from east (every 45 degrees).
COMPASS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


def _from_centre(east_km: float, north_km: float, of: str = "the middle of the area") -> str:
    """"30 km south-east of the middle of the area" (or of ``of``, such as the pressure centre) for a place in a
    hurricane picture, short enough for one small line."""

    distance = math.hypot(east_km, north_km)
    if distance < 0.5:
        return f"at {of}"
    octant = int(round(math.degrees(math.atan2(north_km, east_km)) / 45.0)) % 8
    return f"{distance:.0f} km {COMPASS[octant]} of {of}"


def _ring(ov: S.Overlay, box: Box, x_m: float, y_m: float, dx: float) -> None:
    """The gold ring around one place, when it lies inside the panel."""

    px, py = box.to_px(x_m, y_m)
    if box.x <= px <= box.x + box.w and box.y <= py <= box.y + box.h:
        ov.ring((px, py), max(8.0, 0.75 * dx / box.mpp) + 3)


def product_wind_lowest(L: Looks, out: Path) -> dict[str, Any]:
    """A hurricane run's wind at the lowest model level: the ground-relative speed on its fixed range, with
    streaks and a line at hurricane force, over the whole area and over its inner part."""

    if not L.hurricane:
        raise NotDrawn("This look is drawn for a hurricane run.")
    f = L.f
    u, v = L.uv0_ground
    top = HURRICANE_WIND_RANGE
    force = HURRICANE_FORCE

    def draw(box: Box) -> np.ndarray:
        us, vs = sample(u, box, f.dx, f.dy), sample(v, box, f.dx, f.dy)
        speed = np.hypot(us, vs)
        base = S.colours("hurricane_wind", speed, 0.0, top)
        tx, _ = streaks(us, vs, box, seed=5)
        # The streaks lighten and darken the colour a little along the wind; they carry no value of their own.
        shade = 1.0 + 0.22 * np.clip(tx, -1.5, 1.5)[..., None] * np.clip(speed / 8.0, 0.25, 1.0)[..., None]
        img = S.to8(np.clip(base * shade + 0.02 * np.clip(tx, 0.0, 1.5)[..., None], 0.0, 1.0))
        return blend(img, line_alpha(speed, force, width=1.6), "title", 0.9)

    z0 = L.z0
    wind0 = L.wind0
    j, i = np.unravel_index(int(np.argmax(wind0)), wind0.shape)
    fastest = float(wind0.max())
    east, north = _centre_km(L, int(i), int(j))
    lowest = float(L.pressure0_hpa.min()) if L.pressure0_hpa is not None else None
    # The fastest wind is placed from the storm's centre (the lowest pressure), not from the middle of the area,
    # which the storm leaves as it drifts; without pressure in the frames it is placed from the middle.
    where = _from_centre(east, north)
    from_low = None
    if L.pressure0_hpa is not None:
        pj, pi = np.unravel_index(int(np.argmin(L.pressure0_hpa)), L.pressure0_hpa.shape)
        low_east, low_north = _centre_km(L, int(pi), int(pj))
        from_low = [east - low_east, north - low_north]
        where = _from_centre(from_low[0], from_low[1], "the pressure centre")
    area = float((wind0 >= force).sum() * L.cell_km2())
    ov, cv, whole, zoom, inner = hurricane_frame(
        L, "wind-lowest", draw, level=f"lowest level {S.fmt_height(z0)}", wind="ground-relative",
        notes=S.data_line(f"wind speed at {S.fmt_height(z0)}, ground-relative, 0–{top:g} m/s fixed",
                          "streaks: the direction the wind blows"),
        data=L.data_line(f"lowest model level, {S.fmt_height(z0)} above the sea"),
        left=f"fastest {fastest:.0f} m/s ground-relative",
        right=f"lowest pressure {lowest:.0f} hPa" if lowest is not None else None,
        lines=[(f"gold ring: the fastest wind, {where}", "gold"),
               (f"white line: hurricane force, {force:g} m/s ground-relative", "data")])
    _ring(ov, zoom, (i + 0.5) * f.dx, (j + 0.5) * f.dy, f.dx)
    hurricane_bar(ov, "hurricane_wind", zoom,
                  [(value / top, label) for value, label in
                   ((0.0, "0"), (10.0, "10"), (20.0, "20"), (force, f"{force:g}"), (50.0, "50"), (65.0, "65"),
                    (top, f"{top:g}"))],
                  "wind, ground-relative, m/s")
    save(ov, cv, out)
    facts: dict[str, Any] = {"height_m": z0, "dx_m": f.dx, "resolved_m": 6.0 * f.dx,
                             "wind_max_ground_relative": fastest, "range_m_s": top,
                             "hurricane_force_m_s": force, "hurricane_force_area_km2": area,
                             "wind_max_xy_m": [float((i + 0.5) * f.dx), float((j + 0.5) * f.dy)],
                             "wind_max_from_centre_km": [east, north], "inner_km": inner / 1000.0}
    if lowest is not None:
        facts["p_min_hpa"] = lowest
    if from_low is not None:
        facts["wind_max_from_pressure_centre_km"] = from_low
    return facts


def product_pressure_lowest(L: Looks, out: Path) -> dict[str, Any]:
    """A hurricane run's pressure at the lowest model level, as hPa below the pressure at the area's edge, with
    a line every :data:`HURRICANE_PRESSURE_LINES` hPa, over the whole area and over its inner part.

    GM1 saves no sea-level pressure, so this is the pressure at the lowest model level and the picture says so;
    in CM1's run of the hurricane preset the two differ by about 3 hPa (25 m of air).
    """

    if not L.hurricane:
        raise NotDrawn("This look is drawn for a hurricane run.")
    p = L.pressure0_hpa
    far = L.pressure_edge_hpa
    if p is None or far is None:
        raise NotDrawn("This run's saved frames carry no pressure.")
    f = L.f
    top = HURRICANE_PRESSURE_RANGE
    every = HURRICANE_PRESSURE_LINES
    drop = (np.float32(far) - p).astype(np.float32)

    def draw(box: Box) -> np.ndarray:
        ds = sample(drop, box, f.dx, f.dy)
        img = S.to8(S.colours("hurricane_pressure", pressure_position(ds), 0.0, 1.0))
        # The same lines in both panels: one is drawn where the pressure falls steeply enough on the ground
        # (lines less than HURRICANE_LINES_APART apart), whatever the panel's scale.
        return blend(img, line_alpha(sample(p, box, f.dx, f.dy), every=every, width=1.3,
                                     far=HURRICANE_LINES_APART / box.mpp), "title", 0.55)

    z0 = L.z0
    j, i = np.unravel_index(int(np.argmin(p)), p.shape)
    lowest = float(p.min())
    deepest = float(drop.max())
    east, north = _centre_km(L, int(i), int(j))
    ov, cv, whole, zoom, inner = hurricane_frame(
        L, "pressure-lowest", draw, level=f"lowest level {S.fmt_height(z0)}", wind="",
        notes=S.data_line(f"pressure at the lowest model level ({S.fmt_height(z0)}), not sea level",
                          f"colours: hPa below the mean at the area's edge ({far:.0f} hPa), 0–{top:g} hPa fixed"),
        data=L.data_line(f"lowest model level, {S.fmt_height(z0)} above the sea"),
        left=f"lowest {lowest:.0f} hPa", right=f"{deepest:.0f} hPa below the area's edge",
        lines=[(f"gold ring: the lowest pressure, {_from_centre(east, north)}", "gold"),
               (f"lines: every {every:g} hPa, left out where they would touch or lie more than "
                f"{HURRICANE_LINES_APART / 1000.0:g} km apart", "data")])
    _ring(ov, zoom, (i + 0.5) * f.dx, (j + 0.5) * f.dy, f.dx)
    hurricane_bar(ov, "hurricane_pressure", zoom,
                  [(float(pressure_position(value)), label) for value, label in
                   ((0.0, "0"), (5.0, "5"), (15.0, "15"), (35.0, "35"), (top, f"{top:g} hPa"))],
                  "below the area's edge")
    save(ov, cv, out)
    return {"height_m": z0, "p_min_hpa": lowest, "p_edge_hpa": far, "p_drop_max_hpa": deepest, "range_hpa": top,
            "lines_every_hpa": every, "scale": "square root",
            "p_min_xy_m": [float((i + 0.5) * f.dx), float((j + 0.5) * f.dy)],
            "p_min_from_centre_km": [east, north], "inner_km": inner / 1000.0}


def w_range(L: "Looks") -> float:
    """The updraft slice's fixed range, m/s: a hurricane's drafts are a few m/s, a thunderstorm's tens."""

    return HURRICANE_W_RANGE if L.hurricane else W_RANGE


class NotDrawn(Exception):
    """This product cannot be drawn for this run; the message says why, in plain words."""


def _clouds_from_above(L: Looks, out: Path) -> dict[str, Any]:
    from .satellite import product_clouds

    return product_clouds(L, out)


def _updraft_slice(L: Looks, out: Path) -> dict[str, Any]:
    from .section import product_updraft_slice

    return product_updraft_slice(L, out)


RENDERERS: dict[str, Callable[..., dict[str, Any]]] = {
    "reflectivity": product_reflectivity,
    "spin-near-ground": product_spin,
    "updraft-5km": product_updraft,
    "surface": product_surface,
    "rain": product_rain,
    "clouds-from-above": _clouds_from_above,
    "updraft-slice": _updraft_slice,
    "wind-lowest": product_wind_lowest,
    "pressure-lowest": product_pressure_lowest,
}


__all__ = ["Box", "HURRICANE_ONLY", "LintError", "bilinear", "Looks", "NotDrawn", "PRODUCTS", "RENDERERS",
           "THUMB", "THUNDERSTORM_ONLY", "THUNDERSTORM_ONLY_WHY", "TITLES", "product_pressure_lowest",
           "product_swath", "product_wind_lowest", "sample", "save", "w_range"]
