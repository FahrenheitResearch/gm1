"""Clouds seen from above: a simulated visible picture of the model's cloud water and ice, lit by a low sun.

A CPU simplification of the viz 2.1 simulated-visible tool (whose GPU ray march stays a viz tool), in the same
physics and the quick-look's own style:

* visible extinction per cell ``beta = 1.5 rho q / (rho_w r_eff)`` summed over cloud water, cloud ice, snow,
  graupel or hail and rain, with a fixed effective radius per species (10, 30, 100, 500 and 500 micrometres);
  ``rho`` is the frame's air density, else CM1's own ``p / (Rd T (1 + qv Rv/Rd))`` from its pressure, theta and
  qv (misclibs.F:1292-1294: the density of the dry air, which the mixing ratios are per kilogram of);
* the brightness of a column seen straight down is the two-stream reflectance of its optical depth,
  ``R = tau (1 - g) / (2 + tau (1 - g))`` with ``g = 0.85``, dimmed where the sunlight reaching the cloud has
  already passed through other cloud: the optical depth toward a fixed low sun (elevation 12 degrees, from the
  west-south-west, azimuth 250 degrees) is summed along slant rays, level by level, each level shifted by its
  height over the tangent of the sun's elevation;
* the ground is dark, flat and uniform, darkened in the clouds' shadows and seen through the cloud above it.

Nothing is smoothed: every plane is sampled bilinearly onto the picture, as every quick-look is.  At 1 km the
small clouds a satellite would see are missing, and the cloud edges are blocky; the picture says so.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import products as P
from . import style as S
from .frame import FrameUnreadable

RHO_W = 1000.0
#: m: effective radius per species (the viz 2.1 tool's values).
R_EFF = {"qc": 10.0e-6, "qi": 30.0e-6, "qs": 100.0e-6, "qg": 500.0e-6, "qr": 500.0e-6}
G_HG = 0.85
SUN_ELEVATION = 12.0     # degrees
SUN_AZIMUTH = 250.0      # degrees clockwise from north: the sun is in the west-south-west
RD, RV, CP, P00 = 287.04, 461.5, 1005.7, 1.0e5     # CM1 constants.F:113, 125
SKY_LIGHT = 0.28         # the part of a cloud's light that is skylight, which a cloud's shadow does not remove
GROUND_ALBEDO = 0.075
SUN_RGB = np.array([1.0, 0.96, 0.90], np.float32)      # a low sun, slightly warm
SKY_RGB = np.array([0.82, 0.88, 1.0], np.float32)      # skylight, slightly cool
GROUND_RGB = np.array([0.46, 0.55, 0.44], np.float32)   # flat dark grey-green
EXPOSURE = 1.25


def species_present(qf: Any) -> list[str]:
    try:
        names = qf.species_names("q3d") if qf.has("q3d") else ()
    except (FrameUnreadable, ValueError, KeyError):
        names = ()
    return [name for name in R_EFF if name in names]


def density(qf: Any) -> np.ndarray:
    """Air density (kg/m3): the frame's rho, else CM1's rho = p / (Rd T (1 + max(qv, 0) Rv/Rd)) (misclibs.F:1292).

    CM1's rho is the dry air's density (the mixing ratios are per kilogram of dry air), so the extinction
    1.5 rho q / (rho_w r_eff) is the water's mass per cubic metre.  A frame's own rho is this same number: the
    recompute from cm1out's p, theta and qv with this formula matches the look's optical depth to 2e-7."""

    if qf.has("rho"):
        return qf.get("rho")
    if not (qf.has("prs") and (qf.has("th") or qf.has("th3d"))):
        raise FrameUnreadable(f"{qf.directory.name} has no air density, and no pressure and theta to make it")
    p = qf.get("prs")
    if qf.has("th"):
        theta = qf.get("th")
    else:
        th0 = qf.base_column("th0")
        if th0 is None:
            raise FrameUnreadable(f"{qf.directory.name} has theta' but not the base state, so no air density")
        theta = qf.get("th3d") + th0.astype(np.float32)[:, None, None]
    t = theta * (p / np.float32(P00)) ** np.float32(RD / CP)
    qv = qf.species("qv") if qf.has_species("qv") else np.zeros_like(p)
    return (p / (np.float32(RD) * t * (1.0 + np.maximum(qv, 0.0) * np.float32(RV / RD)))).astype(np.float32)


def extinction(qf: Any, species: list[str]) -> np.ndarray:
    """beta (1/m) on the scalar levels, (nz, ny, nx)."""

    rho = density(qf)
    beta = np.zeros(rho.shape, np.float32)
    for name in species:
        beta += np.maximum(qf.species(name), 0.0) * np.float32(1.0 / R_EFF[name])
    beta *= rho * np.float32(1.5 / RHO_W)
    return beta


def _shift(plane: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """``plane`` sampled at (i + sx, j + sy) cells, bilinearly; zero outside (clear air beyond the domain)."""

    ny, nx = plane.shape
    rows = np.arange(ny, dtype=np.float64)[:, None] + sy
    cols = np.arange(nx, dtype=np.float64)[None, :] + sx
    return P.bilinear(plane, np.broadcast_to(rows, (ny, nx)), np.broadcast_to(cols, (ny, nx)), cval=0.0)


def light(beta: np.ndarray, zf: np.ndarray, dx: float, dy: float) -> dict[str, np.ndarray]:
    """The column's optical depth, reflectance, sunlit fraction of its cloud, and the ground's sunlight.

    The slant optical depth toward the sun from the bottom of level k is built from the top down: the level above
    is shifted toward the sun by the level's depth over tan(elevation), so each level is sampled where the slant
    ray through it passes."""

    nz = beta.shape[0]
    dz = np.diff(zf[:nz + 1]).astype(np.float32)
    el = math.radians(SUN_ELEVATION)
    az = math.radians(SUN_AZIMUTH)
    # a ray toward the sun runs west-south-west (sin az, cos az) and up
    hx, hy = math.sin(az), math.cos(az)
    cot = 1.0 / math.tan(el)
    inv_sin = 1.0 / math.sin(el)
    ny, nx = beta.shape[1:]
    slant = np.zeros((ny, nx), np.float32)       # sun-ward optical depth from the top of the level being done
    tau_above = np.zeros((ny, nx), np.float32)   # vertical optical depth above that level
    lit_sum = np.zeros((ny, nx), np.float32)
    weight_sum = np.zeros((ny, nx), np.float32)
    for k in range(nz - 1, -1, -1):
        layer = beta[k] * dz[k]
        # the light scattered in this level and seen from above: (1 - exp(-layer)) exp(-tau_above)
        w = (1.0 - np.exp(-layer)) * np.exp(-tau_above)
        # sunlight reaching the middle of this level: through the cloud above it along the slant, and half this level
        lit_sum += w * np.exp(-(slant + 0.5 * layer * inv_sin))
        weight_sum += w
        tau_above += layer
        # the slant depth at the bottom of this level: the depth at its top, one level's run toward the sun away
        run = float(dz[k]) * cot
        slant = _shift(slant, hx * run / dx, hy * run / dy) + layer * inv_sin
    tau = tau_above
    x = tau * (1.0 - G_HG)
    reflect = x / (2.0 + x)
    lit = np.where(weight_sum > 1.0e-6, lit_sum / np.maximum(weight_sum, 1.0e-6), 1.0).astype(np.float32)
    return {"tau": tau, "reflect": reflect.astype(np.float32), "lit": lit, "ground_sun": np.exp(-slant)}


def image(fields: dict[str, np.ndarray], box: P.Box, dx: float, dy: float) -> np.ndarray:
    """The picture's pixels (uint8 RGB) from :func:`light`'s planes."""

    r = P.sample(fields["reflect"], box, dx, dy)
    lit = P.sample(fields["lit"], box, dx, dy)
    gsun = P.sample(fields["ground_sun"], box, dx, dy)
    sun = r * (1.0 - SKY_LIGHT) * lit
    sky = r * SKY_LIGHT
    ground = GROUND_ALBEDO * (1.0 - r) * (0.35 + 0.65 * gsun)
    hdr = EXPOSURE * (sun[..., None] * SUN_RGB + sky[..., None] * SKY_RGB) + ground[..., None] * GROUND_RGB
    return S.finish(hdr, k=1.6, gamma=1.2)


def product_clouds(L: "P.Looks", out: Path) -> dict[str, Any]:
    f = L.f
    species = species_present(f)
    if not any(name in species for name in ("qc", "qi")):
        raise P.NotDrawn("This run saved no cloud water or ice.")
    beta = extinction(f, species)
    fields = light(beta, f.zf, f.dx, f.dy)
    box = P.single_box(f.extent)
    img = image(fields, box, f.dx, f.dy)
    img, spacing = P.ground_ticks(img, box, f.domainlocx, f.domainlocy, alpha=0.16)
    cv = P.canvas()
    P.paste(cv, img, box)
    cv = P.outline(cv, box)
    tau = fields["tau"]
    cloudy = tau >= 1.0
    area = float(cloudy.sum() * L.cell_km2())
    top = np.where(beta * np.diff(f.zf[:f.nz + 1]).astype(np.float32)[:, None, None] > 0.05,
                   f.zh[:, None, None], 0.0).max(axis=0)
    words = {"qc": "cloud water", "qi": "cloud ice", "qs": "snow", "qg": "graupel or hail", "qr": "rain"}
    ov = P.overlay(L, "clouds-from-above", L.data_line("seen from straight above, lit by a low sun"),
                   level="the whole column", wind="",
                   notes=S.data_line("from the model's " + ", ".join(words[n] for n in species),
                                     f"sun {SUN_ELEVATION:g}° up in the west-south-west",
                                     "not satellite data", L.frame_words()))
    x, y = ov.x0, box.y + ov.U(10)
    ov.readout((x, y), [("cloud cover (optical depth above 1)", f"{area:,.0f} km²"),
                        ("highest cloud top", S.fmt_height(float(top.max())) if top.max() > 0 else "no cloud"),
                        ("thickest column, optical depth", f"{float(tau.max()):,.0f}")])
    ov.text((x, y + ov.U(300)), f"cloud edges are blocky at {S.fmt_grid(f.dx)}", "micro", colour="muted",
            anchor="ls")
    ov.text((x, y + ov.U(324)), f"small clouds under about {S.fmt_km(6 * f.dx)} are missing", "micro",
            colour="muted", anchor="ls")
    P.scale_bar(ov, x, box.y + box.h - ov.U(8), box)
    P.save(ov, cv, out)
    return {"cloud_area_km2": area, "cloud_top_max_m": float(top.max()), "tau_max": float(tau.max()),
            "species": species, "sun_elevation_deg": SUN_ELEVATION, "sun_azimuth_deg": SUN_AZIMUTH}


__all__ = ["R_EFF", "density", "extinction", "image", "light", "product_clouds", "species_present"]
