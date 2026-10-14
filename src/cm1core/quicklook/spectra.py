"""Wind energy by size: the kinetic energy spectrum of the horizontal wind at a few heights, by wavelength.

How much of the wind's energy sits in eddies of each size, at one saved time.  The line falls toward small sizes;
where the grid stops resolving the flow it falls much faster, because the model's numerical diffusion (and the
advection scheme's own damping) removes eddies a few grid spacings across.  That happens near a wavelength of about
6 to 7 grid spacings (the effective resolution usually measured for models of this kind from such spectra): the picture
marks that band, so a user can see what the grid holds and what it does not.  Wavelengths are full wavelengths: an
eddy D across (one up-and-down of the wind) is about 2 D long, so a 1 km vortex sits near 2 km.

Method, at each height (the scalar level nearest it; its real height is said):

* u and v at the scalar points (the frame's own C-grid winds averaged to the cell centres);
* the area's best-fitting plane (mean and slope) is removed from each, so the frame's motion and the
  large-scale wind do not count; a Tukey window (cosine edges over :data:`TUKEY` of each side) tapers the edges,
  since an open-sided domain is not periodic, and the energy is divided by the window's mean square;
* the 2-D FFT's energy ``(|U|^2 + |V|^2) / 2`` is summed in rings of width ``dk = 2 pi / L`` (``L`` the longer
  side) and divided by ``dk``: E(k) in m3 s-2, so the sum of E dk is the mean kinetic energy per unit mass;
* only rings out to the grid's shortest wave (2 grid spacings) are drawn.

A dashed line (slope -5/3) and a dotted one (slope -3) are drawn for reference above the curve nearest 1 km, where
the area is at least :data:`GUIDES_MIN_CELLS` columns across.  They are guides, not fits.  The energy axis spans
whole decades from the curves' own smallest to largest values.  An area fewer than :data:`MIN_CELLS` columns across
has too few waves for a spectrum, and the picture says so.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import products as P
from . import style as S

#: The heights (m above ground) the look draws, where the model reaches them; the lowest model level always.
HEIGHTS = (1000.0, 5000.0, 10000.0)
#: The colours of the heights: the hodograph's (environment look), lowest to highest.
COLOURS = ("#ff5f57", "#ffb347", "#34d399", "#4cc3ff")
#: The window's cosine-tapered share of each side.
TUKEY = 0.25
#: The grid's limit, in grid spacings (the band the picture marks).
RESOLVED_DX = (6.0, 7.0)
#: The most decades the energy axis spans (below that, a curve's tail is cut at the axis).
DECADES = 11
#: The fewest columns across (each way) the spectrum is drawn for, and the fewest a side is tapered on.
MIN_CELLS = 8
#: The fewest columns across (the longer side) the reference slopes are drawn for: they run from a quarter of the
#: area down to 3 grid spacings, so a smaller area leaves them too short to read.
GUIDES_MIN_CELLS = 24
#: The least wind energy (m2 s-2, the plane-removed wind's, summed over every size) one height must have for a
#: spectrum: below it at every height there are no eddies yet (a storm's first saved time, before it has grown),
#: and what is left is floating-point noise, about 1e-25 m2 s-2.
EDDY_FLOOR_M2_S2 = 1.0e-6


SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def decade(value: float, _position: Any = None) -> str:
    """'10³', '10⁻⁴' for a whole power of ten, else nothing (a log axis's labels, in the picture's own font)."""

    if value <= 0:
        return ""
    e = math.log10(value)
    return "10" + str(int(round(e))).translate(SUPERSCRIPT) if abs(e - round(e)) < 1e-6 else ""


def tukey(n: int, alpha: float = TUKEY) -> np.ndarray:
    """A Tukey window of ``n`` points (cosine edges over ``alpha`` of the length, flat in between); a side shorter
    than :data:`MIN_CELLS` is not tapered (its edges would be all of it)."""

    if n < MIN_CELLS or alpha <= 0.0:
        return np.ones(n)
    x = np.arange(n) / (n - 1)
    w = np.ones(n)
    edge = alpha / 2.0
    lo = x < edge
    hi = x > 1.0 - edge
    w[lo] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[lo] / alpha - 1.0)))
    w[hi] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[hi] / alpha - 2.0 / alpha + 1.0)))
    return w


def detrend(field: np.ndarray) -> np.ndarray:
    """``field`` less its least-squares plane a + b x + c y."""

    f = np.asarray(field, np.float64)
    ny, nx = f.shape
    y, x = np.mgrid[0:ny, 0:nx]
    a = np.column_stack([np.ones(f.size), x.ravel(), y.ravel()])
    coef, *_ = np.linalg.lstsq(a, f.ravel(), rcond=None)
    return f - (a @ coef).reshape(ny, nx)


def spectrum(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """(k in rad/m, E(k) in m3 s-2) of the horizontal wind (u, v at the scalar points), as the module says."""

    ny, nx = u.shape
    window = np.outer(tukey(ny), tukey(nx))
    norm = float(np.mean(window ** 2))
    energy = np.zeros((ny, nx))
    for comp in (u, v):
        spec = np.fft.fft2(detrend(comp) * window) / (nx * ny)
        energy += 0.5 * np.abs(spec) ** 2
    energy /= norm
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kk = np.hypot(kx[None, :], ky[:, None])
    dk = 2.0 * np.pi / max(nx * dx, ny * dy)
    kmax = np.pi / max(dx, dy)                         # the shortest wave the grid holds: 2 grid spacings
    rings = np.rint(kk / dk).astype(np.int64)
    n = int(kmax // dk)
    total = np.bincount(rings.ravel(), weights=energy.ravel(), minlength=n + 1)[:n + 1]
    k = np.arange(n + 1) * dk
    return k[1:], total[1:] / dk


def fmt_wavelength(m: float) -> str:
    return S.fmt_km(m) if m >= 1000.0 else f"{m:g} m"


def heights_of(qf: Any) -> list[tuple[int, float]]:
    """(scalar level, its height) for the lowest level and each of :data:`HEIGHTS` the model reaches, without
    repeats."""

    zh = qf.zh
    rows: list[tuple[int, float]] = [(0, float(zh[0]))]
    for z in HEIGHTS:
        if z > float(zh[-1]) or z <= float(zh[0]):
            continue
        k = qf.level_near(z)
        if k not in [r[0] for r in rows]:
            rows.append((k, float(zh[k])))
    return rows


def product_spectra(L: "P.Looks", out: Path) -> dict[str, Any]:
    f = L.f
    if not (f.has("u3d") and f.has("v3d")):
        raise P.NotDrawn("This run's saved frames carry no horizontal wind.")
    if min(f.nx, f.ny) < MIN_CELLS:
        raise P.NotDrawn(f"This run's area is {f.nx} × {f.ny} columns, too narrow for a spectrum (it needs at least "
                         f"{MIN_CELLS} each way).")
    levels = heights_of(f)
    curves = []
    for k, z in levels:
        u, v = f.uv_level(k)
        kk, ee = spectrum(u, v, f.dx, f.dy)
        curves.append((k, z, kk, ee))
    positive = [ee[ee > 0] for *_r, ee in curves if np.any(ee > 0)]
    dk_m = 2.0 * np.pi / max(f.nx * f.dx, f.ny * f.dy)
    if not positive or not any(float(np.sum(ee)) * dk_m >= EDDY_FLOOR_M2_S2 for *_r, ee in curves):
        raise P.NotDrawn("No eddies yet: the wind is the same everywhere at these heights, so it has no spectrum.")
    hi = 10.0 ** math.ceil(math.log10(max(float(e.max()) for e in positive)) + 0.3)
    lo = max(10.0 ** math.floor(math.log10(min(float(e.min()) for e in positive))), hi / 10.0 ** DECADES)
    W, H = P.W, P.H
    fig = S.figure((W, H))
    left, right, top, bottom = 130 / W, 1 - 560 / W, 1 - 215 / H, 175 / H
    ax = fig.add_axes((left, bottom, right - left, top - bottom))
    ax.set_facecolor(S.PALETTE["void"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(S.PALETTE["rule"])
    fm = S.mpl_fontprops("micro", fig)
    fl = S.mpl_fontprops("label", fig)
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0))
        axis.set_major_formatter(FuncFormatter(decade))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(colors=S.PALETTE["muted"], labelsize=fm.get_size(), which="both")
    ax.grid(True, which="major", color=S.PALETTE["rule"], linewidth=0.8)
    dk = 2.0 * np.pi / max(f.extent)
    kmax = np.pi / max(f.dx, f.dy)
    ax.set_xlim(dk * 0.9, kmax * 1.15)
    ax.set_ylim(lo, hi)
    # the grid's limit: 7 to 6 grid spacings (gold band), and shorter waves to 2 grid spacings (grey)
    k7, k6 = 2.0 * np.pi / (RESOLVED_DX[1] * f.dx), 2.0 * np.pi / (RESOLVED_DX[0] * f.dx)
    ax.axvspan(k7, k6, color=S.PALETTE["gold"], alpha=0.22, lw=0, zorder=1)
    ax.axvspan(k6, kmax * 1.15, color=S.PALETTE["faint"], alpha=0.16, lw=0, zorder=1)
    ax.axvline(kmax, color=S.PALETTE["dim"], lw=1.0, ls=(0, (2, 3)), zorder=1)
    shown = []
    for n, (k, z, kk, ee) in enumerate(curves):
        colour = COLOURS[min(n, len(COLOURS) - 1)]
        good = ee > 0
        ax.plot(kk[good], ee[good], color=colour, lw=2.0, zorder=4, solid_capstyle="round")
        shown.append((z, colour, kk[good], ee[good]))
    # reference slopes through the curve nearest 1 km (3 times above it at 20 grid spacings), from a quarter of the
    # area down to 3 grid spacings; each label on a dark patch so a curve crossing it does not hide it
    ref = min(curves, key=lambda c: abs(c[1] - 1000.0))
    k_anchor = 2.0 * np.pi / (20.0 * f.dx)
    guides = bool(ref[2].size) and max(f.nx, f.ny) >= GUIDES_MIN_CELLS
    if guides:
        e_anchor = float(np.interp(math.log(k_anchor), np.log(ref[2]), np.log(np.maximum(ref[3], 1e-30))))
        e_anchor = math.exp(e_anchor) * 3.0
        for slope, label, style in ((-5.0 / 3.0, "slope −5/3", (0, (5, 4))), (-3.0, "slope −3", (0, (1.5, 3)))):
            ks = np.array([2.0 * np.pi / (max(f.extent) / 4.0), 2.0 * np.pi / (3.0 * f.dx)])
            es = e_anchor * (ks / k_anchor) ** slope
            ax.plot(ks, es, color=S.PALETTE["muted"], lw=1.1, ls=style, zorder=3)
            ax.text(ks[-1] * 1.08, es[-1], label, color=S.PALETTE["muted"], fontproperties=fm, va="center",
                    ha="left", zorder=5, clip_on=True,
                    bbox=dict(facecolor=S.PALETTE["void"], edgecolor="none", pad=1.5))
    ax.set_xlabel("wavenumber k (rad/m)", fontproperties=fm, color=S.PALETTE["muted"])
    ax.set_ylabel("E(k), m³ s⁻²", fontproperties=fm, color=S.PALETTE["muted"])
    # the wavelength axis on top
    top_ax = ax.secondary_xaxis("top", functions=(lambda k: 2 * np.pi / np.maximum(k, 1e-12),
                                                  lambda lam: 2 * np.pi / np.maximum(lam, 1e-12)))
    ticks = [w for w in (1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6)
             if dk * 0.9 <= 2 * np.pi / w <= kmax * 1.15]
    top_ax.set_xticks(ticks)
    top_ax.set_xticklabels([fmt_wavelength(w) for w in ticks])
    top_ax.xaxis.set_minor_formatter(NullFormatter())   # a small area's few ticks would get matplotlib's own labels
    top_ax.tick_params(colors=S.PALETTE["muted"], labelsize=fm.get_size())
    for t in top_ax.get_xticklabels():
        t.set_fontproperties(fm)
    top_ax.spines["top"].set_color(S.PALETTE["rule"])
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(fm)
    base = S.fig_to_rgb(fig)
    # words
    lam6, lam7 = RESOLVED_DX[0] * f.dx, RESOLVED_DX[1] * f.dx
    ov = P.overlay(L, "spectra",
                   L.data_line("the horizontal wind's energy by wavelength"),
                   level=", ".join(S.fmt_height(z) for _k, z, *_ in curves), wind="",
                   notes=S.data_line("plane removed, Tukey window, ring sums of the 2-D FFT",
                                     f"gold band: {RESOLVED_DX[0]:g} to {RESOLVED_DX[1]:g} grid spacings "
                                     f"({fmt_wavelength(lam6)} to {fmt_wavelength(lam7)})",
                                     "dashed −5/3, dotted −3: for reference, not fits" if guides else ""))
    # the axes' position in pixels
    x_axis = int(left * W)
    y_axis_top = int((1 - top) * H)
    ov.text((x_axis, y_axis_top - ov.U(44)), "wavelength", "micro", colour="muted", anchor="ls")
    # the key, direct-labelled at each line's start, and a legend on the right
    lx = int(right * W) + ov.U(60)
    ly = y_axis_top + ov.U(20)
    ov.text((lx, ly), "heights above the sea" if L.hurricane else "heights above ground", "label", colour="text",
            anchor="ls")
    ly += ov.U(34)
    for z, colour, kk, ee in shown:
        ov.rule(lx, ly - ov.U(6), ov.U(34), thickness=3, colour=colour)
        label = (f"{S.fmt_height(z)} (lowest level)" if z == curves[0][1] else S.fmt_height(z))
        ov.text((lx + ov.U(46), ly), label, "label", colour="data", anchor="ls")
        ly += ov.U(30)
    ly += ov.U(18)
    ov.rule(lx, ly - ov.U(6), ov.U(34), thickness=8, colour=S.PALETTE["gold"], alpha=0.35)
    ov.text((lx + ov.U(46), ly), f"the grid's limit, about {RESOLVED_DX[0]:g} to {RESOLVED_DX[1]:g} dx", "label",
            colour="data", anchor="ls")
    ly += ov.U(26)
    ov.text((lx + ov.U(46), ly), f"wavelengths {fmt_wavelength(lam6)} to {fmt_wavelength(lam7)} at dx "
            f"{S.fmt_grid(f.dx)}", "micro", colour="muted", anchor="ls")
    ly += ov.U(30)
    ov.rule(lx, ly - ov.U(6), ov.U(34), thickness=8, colour=S.PALETTE["faint"], alpha=0.5)
    ov.text((lx + ov.U(46), ly), "shorter: not resolved", "label", colour="data", anchor="ls")
    ly += ov.U(26)
    ov.text((lx + ov.U(46), ly), f"down to 2 dx ({fmt_wavelength(2 * f.dx)}), the shortest wave", "micro",
            colour="muted", anchor="ls")
    ly += ov.U(46)
    share = {}
    for z, colour, kk, ee in shown:
        total = float(np.sum(ee))
        if total > 0:
            share[z] = 100.0 * float(np.sum(ee[kk >= k7])) / total
    ov.text((lx, ly), f"energy in waves shorter than {fmt_wavelength(lam7)}", "micro", colour="muted", anchor="ls",
            tracking=0.04)
    ly += ov.U(34)
    for z, colour, kk, ee in shown:
        if z in share:
            ov.rule(lx, ly - ov.U(8), ov.U(14), thickness=3, colour=colour)
            pct = f"{share[z]:.1f}" if share[z] >= 1.0 else f"{share[z]:.2f}"
            ov.text((lx + ov.U(24), ly), f"{S.fmt_height(z)}:  {pct} % of its energy", "label",
                    colour="data", anchor="ls")
            ly += ov.U(30)
    ly += ov.U(16)
    ov.text((lx, ly), "an eddy D across sits near a wavelength of 2 D", "micro", colour="muted", anchor="ls")
    P.save(ov, base, out)
    facts: dict[str, Any] = {
        "heights_m": [z for _k, z, *_ in curves], "levels": [k + 1 for k, *_ in curves],
        "resolved_band_m": [lam6, lam7], "shortest_wave_m": 2.0 * f.dx, "dk_rad_m": dk,
        "window": f"Tukey {TUKEY:g}", "detrend": "least-squares plane", "reference_slopes": guides,
        "share_shorter_than_7dx_percent": {f"{z:.0f}": v for z, v in share.items()},
        "energy_m2_s2": {f"{z:.0f}": float(np.sum(ee) * dk) for z, _c, kk, ee in shown},
    }
    return facts


__all__ = ["COLOURS", "DECADES", "GUIDES_MIN_CELLS", "HEIGHTS", "MIN_CELLS", "RESOLVED_DX", "TUKEY", "detrend",
           "heights_of", "product_spectra", "spectrum", "tukey"]
