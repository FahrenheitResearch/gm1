"""Clouds in visible and infrared: the run's clouds seen from above, lit by a low sun, beside a simulated infrared
picture of how cold their tops are.  Simulated from the model's own fields, not a satellite retrieval.

The visible panel is the clouds-from-above look itself (:mod:`.satellite`: its light, its extinction formula and its
fixed particle sizes), drawn beside the infrared one.  The infrared panel sizes the particles by the run's own
microphysics scheme (:mod:`.particles`), and says which: the sizes change how deep into a cloud the infrared sees.

The infrared panel is a window channel near 10.3 micrometres, computed column by column:

* each level's absorption optical depth is its visible one over :data:`VISIBLE_OVER_INFRARED` (2.13): the visible
  extinction is twice the particles' cross-section, and at 10 to 11 micrometres a large particle absorbs about once
  its cross-section; 2.13 is the ratio used for ice cloud at 11 micrometres;
* each level emits at the model's air temperature (theta and pressure, CM1's ``T = theta (p / p00)^(Rd/cp)``) by
  Planck's law at 10.3 micrometres, dimmed by the cloud above it, summed from the top down;
* below the cloud the picture sees the air at the lowest model level: no water vapour or other gas absorbs, so clear
  air shows the air near the ground (a real satellite sees the clear air a few degrees colder);
* the brightness temperature is Planck's law turned round at 10.3 micrometres.

The model's top caps how cold an overshooting top can get (it cannot rise above the model top, and the damping layer
under it slows it); the picture says how high both are.  Nothing is smoothed: every plane is sampled bilinearly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import particles as PS
from . import products as P
from . import satellite as SA
from . import style as S
from .frame import FrameUnreadable

#: The visible optical depth over the infrared absorption optical depth (ice cloud at 11 micrometres).
VISIBLE_OVER_INFRARED = 2.13
WAVELENGTH = 10.3e-6                     # m
C1 = 1.191042972e-16                     # 2 h c^2, W m2 sr-1
C2 = 1.4387769e-2                        # h c / k, m K
#: The infrared colours' fixed range, degrees C.
TB_LOW, TB_HIGH = -90.0, 40.0
#: Where the colours start, degrees C: grey above, then a colour for every 10 degrees colder.
TB_COLOURED = -30.0
#: The cloud tops whose area the picture states (degrees C).
COLD_TOPS = -50.0
#: A column is a cloud top for the readout when its infrared optical depth is at least this (it hides what is
#: under it); the coldest such column is the one ringed.
CLOUD_TAU = 1.0
#: (degrees C, colour): grey from warm (dark) to -30 C (light), then a new colour every 10 degrees colder.  Only
#: the tops warmer than -30 C are grey: the coldest (overshooting) tops are purple from -70 C and pink from -80 C,
#: so none of them reads as a warm grey low cloud (the classic tables' grey band from -70 to -80 C would).
IR_STOPS: tuple[tuple[float, str], ...] = (
    (40.0, "#0b0c10"), (0.0, "#5b606b"), (-30.0, "#d7dbe2"),
    (-30.01, "#8fd3ff"), (-40.0, "#1d5fd8"),
    (-40.01, "#7ee08a"), (-50.0, "#11823f"),
    (-50.01, "#ffe066"), (-60.0, "#ff9326"),
    (-60.01, "#ff4a3d"), (-70.0, "#8f1220"),
    (-70.01, "#3d0a4f"), (-80.0, "#c45ce0"),
    (-80.01, "#ff4fa3"), (-90.0, "#ffb8dc"),
)


def planck(t_k: np.ndarray | float) -> np.ndarray:
    """Spectral radiance at :data:`WAVELENGTH` (W m-2 sr-1 m-1) of a black body at ``t_k``."""

    t = np.asarray(t_k, np.float64)
    return C1 / WAVELENGTH ** 5 / np.expm1(C2 / (WAVELENGTH * t))


def brightness(radiance: np.ndarray) -> np.ndarray:
    """The temperature (K) whose black-body radiance at :data:`WAVELENGTH` is ``radiance``."""

    b = np.maximum(np.asarray(radiance, np.float64), 1.0e-30)
    return C2 / (WAVELENGTH * np.log1p(C1 / (WAVELENGTH ** 5 * b)))


def temperature(qf: Any) -> np.ndarray:
    """The air temperature (K) on the scalar levels: CM1's T = theta (p / p00)^(Rd/cp)."""

    if not qf.has("prs"):
        raise FrameUnreadable(f"{qf.directory.name} has no pressure, so no air temperature")
    p = qf.get("prs")
    if qf.has("th"):
        theta = qf.get("th")
    else:
        th0 = qf.base_column("th0")
        if th0 is None or not qf.has("th3d"):
            raise FrameUnreadable(f"{qf.directory.name} has neither theta nor theta' with the base state")
        theta = qf.get("th3d") + th0.astype(np.float32)[:, None, None]
    return (theta * (p / np.float32(SA.P00)) ** np.float32(SA.RD / SA.CP)).astype(np.float32)


def infrared(beta: np.ndarray, t: np.ndarray, zf: np.ndarray, zh: np.ndarray) -> dict[str, np.ndarray]:
    """The column's brightness temperature (K), where its light comes from (the emission-weighted height, m), and
    its infrared optical depth, from the visible extinction ``beta`` (1/m) and the air temperature ``t`` (K)."""

    nz = beta.shape[0]
    dz = np.diff(np.asarray(zf, np.float64)[:nz + 1])
    radiance = np.zeros(beta.shape[1:], np.float64)
    seen = np.ones(beta.shape[1:], np.float64)          # how much of each level's light gets out of the top
    height = np.zeros(beta.shape[1:], np.float64)
    for k in range(nz - 1, -1, -1):
        tau = beta[k].astype(np.float64) * (dz[k] / VISIBLE_OVER_INFRARED)
        through = np.exp(-tau)
        weight = seen * (1.0 - through)
        radiance += weight * planck(t[k])
        height += weight * float(zh[k])
        seen *= through
    radiance += seen * planck(t[0])                     # the air at the lowest model level, through all the cloud
    return {"tb": brightness(radiance).astype(np.float32), "emit_m": height.astype(np.float32),
            "tau": (-np.log(np.maximum(seen, 1.0e-300))).astype(np.float32)}


def ir_rgb(tb_c: np.ndarray) -> np.ndarray:
    """float32 RGB 0..1 of brightness temperatures (degrees C) on :data:`IR_STOPS`, clipped to the range."""

    x = np.clip(np.asarray(tb_c, np.float64), TB_LOW, TB_HIGH)
    pos = np.array([v for v, _ in IR_STOPS])[::-1]                   # ascending for np.interp
    cols = np.array([S.hex_rgb(c) for _, c in IR_STOPS])[::-1]
    return np.stack([np.interp(x, pos, cols[:, ch]) for ch in range(3)], axis=-1).astype(np.float32)


def gif_keep() -> list[tuple[int, int, int]]:
    """The colours of the tops colder than -30 C, a degree apart, which this look's GIF film keeps exactly
    (:data:`.film.GIF_KEEP`): a few pixels of the coldest tops would otherwise turn a grey in the GIF's colours."""

    rgb = ir_rgb(np.arange(-31.0, TB_LOW - 0.5, -1.0))
    return [tuple(int(round(255.0 * c)) for c in row) for row in rgb]


def ir_bar(ov: S.Overlay, x0: int, y0: int, x1: int, y1: int) -> None:
    """The infrared colours as a bar from :data:`TB_HIGH` (left) to :data:`TB_LOW` (right), a tick at each step."""

    from PIL import Image

    values = np.linspace(TB_HIGH, TB_LOW, max(2, x1 - x0))
    strip = np.repeat((ir_rgb(values) * 255 + 0.5).astype(np.uint8)[None], max(1, y1 - y0), axis=0)
    ov.draw_rgba(Image.fromarray(strip).convert("RGBA"), (x0, y0))
    ms = ov.px("micro")
    for value in (40, 20, 0, -20, -30, -40, -50, -60, -70, -90):
        tx = x0 + (TB_HIGH - value) / (TB_HIGH - TB_LOW) * (x1 - x0)
        label = S.fmt_num(value, sign=True)
        anchor = "ls" if tx - x0 < 4 else "rs" if x1 - tx < 4 else "ms"
        ov.text((tx, y1 + ov.U(8) + ms), label, "micro", anchor=anchor, halo=False)


def top_words(qf: Any) -> tuple[str, dict[str, float]]:
    """'model top 20 km, damping layer from 15 km' (as the frame's record says), and the numbers."""

    top = float(qf.zf[-1])
    config = qf.config
    depth = config.get("damping_depth_m")
    rate = config.get("damping_rate_s_inv")
    damped = config.get("irdamp", 1) not in (0, False) and isinstance(depth, (int, float)) \
        and isinstance(rate, (int, float)) and depth > 0 and rate > 0
    facts = {"model_top_m": top}
    if damped:
        facts["damping_from_m"] = top - float(depth)
        return f"model top {S.fmt_height(top)}, damping layer from {S.fmt_height(top - float(depth))}", facts
    return f"model top {S.fmt_height(top)}", facts


def boxes(extent: tuple[float, float], inner: float | None = None) -> tuple["P.Box", "P.Box"]:
    """Two panels side by side, a little smaller than the two-panel looks' so three lines of words fit under them:
    the whole area, or with ``inner`` (m) a square that wide around the area's middle (a hurricane's inner part, as
    its own looks show it)."""

    lx, ly = (extent if inner is None else (inner, inner))
    b = P.fit_box(lx, ly, 0, 182, 700, 700)
    x0m = 0.0 if inner is None else 0.5 * (extent[0] - inner)
    y0m = 0.0 if inner is None else 0.5 * (extent[1] - inner)
    gap = 44
    x0 = (P.W - 2 * b.w - gap) // 2
    return (P.Box(x0, b.y, b.w, x0m, y0m, b.span, h=b.h), P.Box(x0 + b.w + gap, b.y, b.w, x0m, y0m, b.span, h=b.h))


def product_infrared(L: "P.Looks", out: Path) -> dict[str, Any]:
    f = L.f
    species = SA.species_present(f)
    if not any(name in species for name in ("qc", "qi")):
        raise P.NotDrawn("This run saved no cloud water or ice.")
    inner = min(P.HURRICANE_INNER, 0.5 * min(f.extent)) if L.hurricane else None
    b1, b2 = boxes(f.extent, inner)
    cv = P.canvas()
    # the visible panel: the clouds-from-above look, as it ships
    light = SA.light(SA.extinction(f, species), f.zf, f.dx, f.dy)
    img1, spacing = P.ground_ticks(SA.image(light, b1, f.dx, f.dy), b1, f.domainlocx, f.domainlocy, alpha=0.16)
    del light
    # the infrared panel: the scheme's own particle sizes
    rho = SA.density(f)
    t = temperature(f)
    beta, sizes = PS.optics(f, species, rho, t)
    ir = infrared(beta, t, f.zf, f.zh)
    del beta, rho, t
    tb_c = ir["tb"] - np.float32(273.15)
    img2 = S.to8(ir_rgb(P.sample(tb_c, b2, f.dx, f.dy)))
    img2, _ = P.ground_ticks(img2, b2, f.domainlocx, f.domainlocy, alpha=0.16)
    P.paste(cv, img1, b1)
    P.paste(cv, img2, b2)
    cv = P.outline(P.outline(cv, b1), b2)
    # the numbers are of what the panels show (a hurricane's inner part, else the whole area)
    i0, j0 = int(b2.mx0 // f.dx), int(b2.my0 // f.dy)
    i1, j1 = int(math.ceil((b2.mx0 + b2.span) / f.dx)), int(math.ceil((b2.my0 + b2.span_y) / f.dy))
    shown = tb_c[j0:j1, i0:i1]
    cloudy = ir["tau"][j0:j1, i0:i1] >= CLOUD_TAU
    coldest = emit = None
    if cloudy.any():
        jj, ii = np.unravel_index(int(np.argmin(np.where(cloudy, shown, np.inf))), shown.shape)
        j, i = jj + j0, ii + i0
        coldest = float(tb_c[j, i])
        emit = float(ir["emit_m"][j, i])
    cold_area = float((shown <= COLD_TOPS).sum() * L.cell_km2())
    top, top_facts = top_words(f)
    if inner is None:
        data = L.data_line("simulated, not a satellite retrieval")
    else:
        lx, ly = f.extent
        data = S.data_line(L.facts.name, f"{S.fmt_grid(f.dx)} grid",
                           f"the inner {S.fmt_km(inner)} of the {lx / 1000:g} × {ly / 1000:g} km area",
                           "simulated, not a satellite retrieval")
    ov = P.overlay(L, "infrared", data,
                   level="the whole column", wind="",
                   notes=S.data_line(f"infrared: emission at the model's air temperature, absorption = visible depth "
                                     f"/ {VISIBLE_OVER_INFRARED:g}, no water vapour", top, *sizes.notes))
    P.panel_label(ov, b1, "visible, lit by a low sun", f"sun {SA.SUN_ELEVATION:g}° up in the west-south-west")
    P.panel_label(ov, b2, f"infrared {WAVELENGTH * 1e6:g} µm, cloud-top temperature",
                  f"coldest {S.fmt_num(coldest)} °C" if coldest is not None else "no cloud top yet")
    if coldest is not None:
        for b in (b1, b2):
            P._ring(ov, b, (i + 0.5) * f.dx, (j + 0.5) * f.dy, f.dx)
    P.scale_bar(ov, b1.x, b1.y + b1.h + ov.U(16), b1)
    ms = ov.px("micro")
    row1 = b1.y + b1.h + ov.U(12) + ms
    row2 = row1 + int(ms * 1.55)
    if coldest is not None:
        ov.text((b1.x + b1.w, row1), f"gold ring: the coldest cloud top, {S.fmt_num(coldest)} °C, its light from "
                f"about {S.fmt_height(round(emit, -2))}", "micro", colour="gold", anchor="rs", halo=False)
    else:
        ov.text((b1.x + b1.w, row1), f"no cloud yet with an infrared optical depth of {CLOUD_TAU:g} or more", "micro",
                colour="muted", anchor="rs", halo=False)
    ov.text((b1.x + b1.w, row2), f"visible: the clouds-from-above look and its fixed particle sizes  ·  features "
            f"under about {S.fmt_km(6 * f.dx)} are not resolved", "micro", colour="muted", anchor="rs", halo=False)
    bar_w = ov.U(480)
    by = b2.y + b2.h + ov.U(14)
    ir_bar(ov, b2.x + b2.w - bar_w, by, b2.x + b2.w, by + max(3, ov.U(8)))
    row3 = row2 + int(ms * 1.55)
    ov.text((b1.x + b1.w, row3), S.data_line(L.frame_words(), f"+ fixed to the ground every {S.fmt_km(spacing)}"),
            "micro", colour="muted", anchor="rs", halo=False)
    ov.text((b2.x, row1), "cloud-top temperature, °C", "micro", colour="data", anchor="ls", halo=False)
    ov.text((b2.x, row2), f"{cold_area:,.0f} km² below {S.fmt_num(COLD_TOPS)} °C", "micro", colour="data",
            anchor="ls", halo=False)
    for words in ("infrared " + sizes.words, "infrared " + sizes.words.split(",")[0]):
        if ov.measure(words, "micro")[0] <= b2.w:
            break
    ov.text((b2.x, row3), words, "micro", colour="data", anchor="ls", halo=False)
    P.save(ov, cv, out)
    facts: dict[str, Any] = {
        "tb_min_c": coldest, "tb_min_emit_height_m": emit, "cloud_top_tau": CLOUD_TAU,
        "cold_tops_c": COLD_TOPS, "cold_tops_area_km2": cold_area,
        "wavelength_um": WAVELENGTH * 1e6, "visible_over_infrared": VISIBLE_OVER_INFRARED,
        "particle_sizes": sizes.scheme, "particle_sizes_source": sizes.source,
        "mean_radius_um": {k: round(v, 2) for k, v in sizes.mean_um.items()}, "species": species,
        "visible_particle_sizes": "fixed (the clouds-from-above look)",
        "range_c": [TB_LOW, TB_HIGH], "ground": "the air at the lowest model level", **top_facts,
    }
    if coldest is not None:
        facts["tb_min_xy_m"] = [float((i + 0.5) * f.dx), float((j + 0.5) * f.dy)]
    if inner is not None:
        facts["inner_km"] = inner / 1000.0
    return facts


__all__ = ["COLD_TOPS", "IR_STOPS", "TB_HIGH", "TB_LOW", "VISIBLE_OVER_INFRARED", "WAVELENGTH", "brightness",
           "gif_keep", "infrared", "ir_rgb", "planck", "product_infrared", "temperature"]
