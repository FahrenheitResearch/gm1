"""Slice through the strongest updraft: two vertical sections, west-east and south-north, through the column of
strongest w on the w level nearest 5 km.

The sections are drawn with :mod:`.slices`' vertical sampler (bilinear along the line at every level, then
linear in height between levels), from the ground to 16 km or the model top, with nothing else done to the
numbers.  On them:

* w on the fixed, stated ±50 m/s range (the updraft look's colours);
* a silver line where cloud water plus cloud ice is 0.1 g/kg (the cloud's edge);
* a yellow line at 40 dBZ from the model's reflectivity, when the run has one (the same operator as the
  reflectivity look; the volume is shared with it in one pass);
* a gold ring at the strongest w on the section.

The sections move with the strongest updraft at each saved time, and the picture says where they are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import products as P
from . import style as S
from .slices import P_levels, _section

TOP_M = 16000.0
DBZ_LINE = 40.0
DBZ_COLOUR = "#fdf802"


def _edges(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask)
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    return edge


def _panel(L: "P.Looks", along: str, index: int, x0: int, y0: int, pw: int, ph: int, top: float,
           cv: np.ndarray) -> dict[str, Any]:
    """One section (``along`` = "x": west-east at row ``index``; "y": south-north at column ``index``)."""

    f = L.f
    if along == "x":
        length = f.nx * f.dx
        s = (np.arange(pw) + 0.5) / pw * f.nx - 0.5
        ci, cj = s, np.full(pw, float(index))
    else:
        length = f.ny * f.dy
        s = (np.arange(pw) + 0.5) / pw * f.ny - 0.5
        ci, cj = np.full(pw, float(index)), s
    zs = (1.0 - (np.arange(ph) + 0.5) / ph) * top
    w_sec = _section(f.view("w3d"), f.zf, zs, cj, ci)
    span = P.w_range(L)
    img = S.to8(S.colours("w_sym", w_sec, -span, span))
    notes = []
    if f.has_species("qc") and f.has_species("qi"):
        names = f.species_names("q3d")
        q = f.view("q3d")
        cloud = sum(_section(q[names.index(n)], f.zh, zs, cj, ci) for n in ("qc", "qi"))
        edge = _edges(cloud >= 1.0e-4)
        img[edge] = S.rgb8("cloud")
        notes.append("cloud")
    refl = L.reflectivity
    if refl.available:
        edge = _edges(_section(refl.dbz, f.zh, zs, cj, ci) >= DBZ_LINE)
        img[edge] = S.rgb8(DBZ_COLOUR)
        notes.append("dbz")
    cv[y0:y0 + ph, x0:x0 + pw] = img
    k, p = np.unravel_index(int(np.argmax(w_sec)), w_sec.shape)
    return {"length_m": length, "w_max": float(w_sec.max()), "w_min": float(w_sec.min()),
            "peak_px": (x0 + p + 0.5, y0 + k + 0.5), "peak_z_m": float(zs[k]), "notes": notes}


def product_updraft_slice(L: "P.Looks", out: Path) -> dict[str, Any]:
    f = L.f
    w5, k5 = L.w5, L.k5
    z5 = float(f.zf[k5])
    j, i = np.unravel_index(int(np.argmax(w5)), w5.shape)
    top = min(TOP_M, float(f.zf[-1]))
    x0, pw, ph, gap = 220, 1560, 280, 108
    y1 = 206
    y2 = y1 + ph + gap
    cv = P.canvas()
    we = _panel(L, "x", j, x0, y1, pw, ph, top, cv)
    sn = _panel(L, "y", i, x0, y2, pw, ph, top, cv)
    for y in (y1, y2):
        cv = P.outline(cv, P.Box(x0, y, pw, 0.0, 0.0, 1.0, h=ph))
    ox, oy = L.facts.origin_m(f.nx, f.ny)
    xk = (ox + (i + 0.5) * f.dx) / 1000.0
    yk = (oy + (j + 0.5) * f.dy) / 1000.0
    lines = []
    if "cloud" in we["notes"]:
        lines.append("silver line: the cloud's edge, cloud water plus ice 0.1 g/kg")
    if "dbz" in we["notes"]:
        lines.append(f"yellow line: {DBZ_LINE:.0f} dBZ, {L.reflectivity.operator}")
    ov = P.overlay(L, "updraft-slice",
                   L.data_line(f"through the strongest w at {S.fmt_height(z5)}",
                               f"x {S.fmt_num(xk, 1)} km, y {S.fmt_num(yk, 1)} km"),
                   level=f"ground to {S.fmt_height(top)}", wind="",
                   notes=S.data_line(f"w ±{P.w_range(L):g} m/s fixed", "bilinear between levels and columns",
                                     L.frame_words()),
                   notes_vertical=True)
    for n, line in enumerate(lines):
        ov.text((x0 + pw, y2 + ph + ov.U(76 + 22 * n)), line, "micro", colour="data", anchor="rs", halo=False)
    stretch_we = (ph / top) / (pw / we["length_m"])
    for (y, sec, label, a, b) in ((y1, we, f"west–east at y {S.fmt_num(yk, 1)} km", "W", "E"),
                                  (y2, sn, f"south–north at x {S.fmt_num(xk, 1)} km", "S", "N")):
        vtext = f"max {S.fmt_num(sec['w_max'])} m/s at {S.fmt_height(sec['peak_z_m'])}"
        ov.text((x0, y - ov.U(12)), label, "label", colour="text", anchor="ls")
        ov.text((x0 + pw, y - ov.U(12)), vtext, "label", colour="data", anchor="rs", vertical=True)
        ov.text((x0 - ov.U(14), y + ph / 2), a, "label", colour="muted", anchor="rm")
        ov.text((x0 + pw + ov.U(14), y + ph / 2), b, "label", colour="muted", anchor="lm")
        for zk in P_levels(top):
            yy = y + ph - zk / top * ph
            if zk > 0:
                ov.text((x0 - ov.U(40), yy), S.fmt_height(zk), "micro", colour="muted", anchor="rm", halo=False)
        ov.ring(sec["peak_px"], 12.0)
    step = P.nice_step(we["length_m"], 8)
    for d in np.arange(0.0, we["length_m"] + 1e-6, step):
        ov.text((x0 + d / we["length_m"] * pw, y1 + ph + ov.U(24)), S.fmt_num(d / 1000.0), "micro", colour="muted",
                anchor="ms", halo=False)
    step = P.nice_step(sn["length_m"], 8)
    for d in np.arange(0.0, sn["length_m"] + 1e-6, step):
        ov.text((x0 + d / sn["length_m"] * pw, y2 + ph + ov.U(24)), S.fmt_num(d / 1000.0), "micro",
                colour="muted", anchor="ms", halo=False)
    ov.text((x0 + pw, y2 + ph + ov.U(50)), f"km across the domain; vertical scale {stretch_we:.1f} × horizontal",
            "micro", colour="muted", anchor="rs", halo=False)
    ov.ramp_bar("w_sym", (x0, y2 + ph + ov.U(66), x0 + ov.U(400), y2 + ph + ov.U(66) + max(3, ov.U(7))),
                vmin=-P.w_range(L), vmax=P.w_range(L),
                ticks=[(-P.w_range(L), S.fmt_num(-P.w_range(L))), (0.0, "0"),
                       (P.w_range(L), f"{P.w_range(L):.0f} m/s")],
                title="vertical velocity w", vertical=True)
    P.save(ov, cv, out)
    return {"w5_max": float(w5.max()), "level_m": z5, "x_km": xk, "y_km": yk, "i": int(i) + 1, "j": int(j) + 1,
            "w_max_west_east": we["w_max"], "w_max_south_north": sn["w_max"], "top_m": top}


__all__ = ["product_updraft_slice"]
