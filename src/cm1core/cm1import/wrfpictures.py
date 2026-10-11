"""Two pictures for the column route, in the quick-look style (``quicklook/style.py`` as it is).

* :func:`map_png`: the WRF or ArWen domain, 960 x 540, coloured by 0-6 km bulk shear (or CAPE on request), with the
  coastline from LANDMASK, terrain contours, latitude and longitude lines, and the chosen point.  Its reply says
  where the grid sits in the picture, so a page can turn a click into a grid point.
* :func:`preview_png`: the column's skew-T and hodograph, 1920 x 1080, with CM1's getcape numbers, the shear and the
  helicity beside them.

Both carry the frame grammar (title and data line, the clock, the honesty footer, the GM1 mark) and pass its lint.
The numbers they draw are the column's own (:mod:`.wrfcolumn`) and the Rust report's (CM1 ``getcape.F``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from ..wrf_history import WrfHistory
from . import wrfcolumn as wc

MAP_SIZE = (960, 540)
#: The map's fixed, stated ranges (the same on every file, so two maps compare).
MAP_FIELDS = {
    "shear": {"title": "0–6 km shear", "range": (0.0, 40.0), "ticks": (0, 10, 20, 30, 40), "unit": "m/s",
              "ramp": "ember", "caption": "0–6 KM BULK SHEAR (M/S)"},
    "cape": {"title": "CAPE", "range": (0.0, 4000.0), "ticks": (0, 1000, 2000, 3000, 4000), "unit": "J/kg",
             "ramp": "ember", "caption": "SURFACE-BASED CAPE (J/KG)"},
}


def _clock_text(stamp: str) -> str:
    """``2025-12-25_01:00:00`` -> ``25 Dec 01:00 UTC`` (the valid time of the record, the map's clock)."""

    import datetime as dt

    when = dt.datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S")
    return when.strftime("%d %b %H:%M UTC").lstrip("0")


def _source_line(history: WrfHistory, record: int) -> str:
    from ..quicklook import style as S

    dims = history.dimensions
    dx = history.attribute("DX") or 0.0
    return S.data_line(history.path.name, f"{history.producer}, domain {history.domain_id}",
                       history.times[record].replace("_", " ") + " UTC",
                       f"{dims.get('west_east')} × {dims.get('south_north')} × {dims.get('bottom_top')} "
                       f"at {S.fmt_grid(float(dx))}")


def map_png(history: WrfHistory, record: int, out: Path | str, *, field: str = "shear",
            point: Mapping[str, Any] | None = None, library: Path | str | None = None) -> dict[str, Any]:
    """Draw the domain map; returns where the grid is in the picture and what was drawn."""

    import numpy as np

    from ..quicklook import products as P
    from ..quicklook import style as S

    if field not in MAP_FIELDS:
        raise ValueError(f"the map shows shear or cape, not {field!r}")
    spec = MAP_FIELDS[field]
    ny, nx = int(history.dimensions["south_north"]), int(history.dimensions["west_east"])
    sampled: dict[str, Any] | None = None
    if field == "shear":
        values = wc.shear_map(history, record)
    else:
        if library is None:
            from ..cartesian_door import resolve_initializer_library

            library = resolve_initializer_library(None)
        values, sampled = wc.cape_map(history, record, library)
    W, H = MAP_SIZE
    fig = S.figure((W, H))
    # The grid keeps its shape: the largest box of nx:ny that fits the plot area.
    area = (0.07, 0.17, 0.62, 0.66)            # left, bottom, width, height (figure fraction)
    aw, ah = area[2] * W, area[3] * H
    scale = min(aw / nx, ah / ny)
    pw, ph = nx * scale, ny * scale
    left = area[0] * W + (aw - pw) / 2
    bottom = area[1] * H + (ah - ph) / 2
    ax = fig.add_axes([left / W, bottom / H, pw / W, ph / H])
    ax.set_facecolor(S.PALETTE["void"])
    lo, hi = spec["range"]
    ramp = S.RAMPS[spec["ramp"]]
    rgb = ramp.colours(np.where(np.isfinite(values), values, np.nan), lo, hi)
    if field == "cape":
        # Unsampled columns are drawn as the sampled column at the centre of their block (nearest neighbour).
        stride = int(sampled["stride"]) if sampled else 1
        jj = np.clip((np.arange(ny) // stride) * stride + stride // 2, 0, ny - 1)
        ii = np.clip((np.arange(nx) // stride) * stride + stride // 2, 0, nx - 1)
        filled = values[jj][:, ii]
        rgb = ramp.colours(np.where(np.isfinite(filled), filled, 0.0), lo, hi)
    ax.imshow(rgb, origin="lower", extent=(0, nx, 0, ny), interpolation="nearest", aspect="auto", zorder=1)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(S.PALETTE["rule"])
    xc, yc = np.arange(nx) + 0.5, np.arange(ny) + 0.5
    if history.has("HGT"):
        hgt = history.array("HGT", record).astype(np.float64)
        top = float(np.nanmax(hgt))
        if top > 600.0:
            levels = np.arange(500.0, top, 1000.0 if top < 5000 else 1500.0)
            ax.contour(xc, yc, hgt, levels=levels, colors=[S.PALETTE["faint"]], linewidths=0.5, zorder=2)
    if history.has("LANDMASK"):
        land = history.array("LANDMASK", record).astype(np.float64)
        if 0.0 < float(land.mean()) < 1.0:
            ax.contour(xc, yc, land, levels=[0.5], colors=[S.PALETTE["cloud"]], linewidths=0.9, zorder=3)
    graticule: dict[str, list[float]] = {"lat": [], "lon": []}
    has_coords = history.has("XLAT") and history.has("XLONG")
    if has_coords:
        lat = history.array("XLAT", record).astype(np.float64)
        lon = history.array("XLONG", record).astype(np.float64)
        if float(np.ptp(lat)) > 0 and float(np.ptp(lon)) > 0:
            span = max(float(np.ptp(lat)), float(np.ptp(lon)))
            step = next(s for s in (0.5, 1, 2, 5, 10, 20) if span / s <= 8)
            lat_levels = np.arange(math.ceil(lat.min() / step) * step, lat.max(), step)
            lon_levels = np.arange(math.ceil(lon.min() / step) * step, lon.max(), step)
            ax.contour(xc, yc, lat, levels=lat_levels, colors=[S.PALETTE["dim"]], linewidths=0.5,
                       linestyles="dashed", zorder=2)
            ax.contour(xc, yc, lon, levels=lon_levels, colors=[S.PALETTE["dim"]], linewidths=0.5,
                       linestyles="dashed", zorder=2)
            graticule = {"lat": [float(v) for v in lat_levels], "lon": [float(v) for v in lon_levels]}
            fs = 8.0
            # Each line labelled where it meets the grid's west edge (latitude) or south edge (longitude).
            for value in lat_levels:
                edge = lat[:, 0]
                if edge.min() <= value <= edge.max() and np.all(np.diff(edge) > 0):
                    y = float(np.interp(value, edge, yc))
                    ax.text(-0.012 * nx, y, f"{abs(value):g}°{'N' if value >= 0 else 'S'}", color=S.PALETTE["muted"],
                            fontsize=fs, ha="right", va="center", clip_on=False)
            for value in lon_levels:
                edge = lon[0, :]
                if edge.min() <= value <= edge.max() and np.all(np.diff(edge) > 0):
                    x = float(np.interp(value, edge, xc))
                    ax.text(x, -0.02 * ny, f"{abs(value):g}°{'E' if value >= 0 else 'W'}", color=S.PALETTE["muted"],
                            fontsize=fs, ha="center", va="top", clip_on=False)
    marker = None
    if point is not None and point.get("i") and point.get("j"):
        pi, pj = float(point["i"]) - 0.5, float(point["j"]) - 0.5
        ax.plot([pi], [pj], marker="o", ms=11, mfc="none", mec=S.PALETTE["gold"], mew=2.0, zorder=5)
        ax.plot([pi], [pj], marker="o", ms=2.5, color=S.PALETTE["gold"], zorder=5)
        marker = {"i": int(point["i"]), "j": int(point["j"])}
    base = S.fig_to_rgb(fig)

    ov = S.Overlay((W, H))
    ov.title(spec["title"], _source_line(history, record))
    ov.clock(_clock_text(history.times[record]))
    notes = (f"one column in every {sampled['stride']} × {sampled['stride']} sampled; CM1 getcape.F surface parcel"
             if sampled else "winds ground-relative, rotated to the earth; lowest level to 6 km above the ground")
    ov.footer([S.data_line(S.SIMULATED + " by your " + history.producer + " run, not by GM1",
                           "coast from LANDMASK, grey lines terrain every 1 km" if history.has("LANDMASK")
                           else "grey lines: terrain", "dashed: latitude and longitude"), notes])
    ov.mark()
    bx0 = int(0.74 * W)
    ov.ramp_bar(spec["ramp"], (bx0, int(0.30 * H), int(0.95 * W), int(0.30 * H) + ov.U(18)), vmin=lo, vmax=hi,
                ticks=[(float(t), f"{t:g}") for t in spec["ticks"]], title=spec["caption"])
    finite = values[np.isfinite(values)]
    rows = []
    if finite.size:
        rows.append(("largest in the domain", f"{float(finite.max()):.0f} {spec['unit']}"))
    if marker is not None:
        v = values[marker["j"] - 1, marker["i"] - 1]
        if field == "cape" and not np.isfinite(v):
            v = filled[marker["j"] - 1, marker["i"] - 1]
        rows.append(("at the chosen point", f"{float(v):.0f} {spec['unit']}" if np.isfinite(v) else "not sampled"))
    ov.readout((bx0, int(0.50 * H)), rows)
    out = Path(out)
    P.save(ov, base, out)
    return {
        "file": str(out), "field": field, "size": [W, H], "nx": nx, "ny": ny,
        # the grid's box in picture pixels (x from the left, y from the top): i = 1 at x0, j = 1 at y1
        "grid_box_px": [round(left, 2), round(H - bottom - ph, 2), round(left + pw, 2), round(H - bottom, 2)],
        "range": list(spec["range"]), "unit": spec["unit"], "max": float(finite.max()) if finite.size else None,
        "graticule": graticule, "point": marker, "sampled": sampled,
    }


def preview_png(column: wc.WrfColumn, report: Mapping[str, Any] | None, out: Path | str) -> dict[str, Any]:
    """The column's skew-T, wind barbs and hodograph, with the getcape and hodograph numbers."""

    import numpy as np

    from ..quicklook import products as P
    from ..quicklook import style as S
    from ..quicklook.skewt import MS_TO_KT, SkewT, T_COLOUR, TD_COLOUR, dewpoint_k

    W, H = P.W, P.H
    p = column.p_pa
    z = np.asarray(column.z_m, dtype=np.float64)
    th = np.asarray(column.theta_k, dtype=np.float64)
    q = np.asarray(column.qv_kgkg, dtype=np.float64)
    if p is None:
        # No P+PB in the file: CM1's own reconstruction of the written file (base.F:627-635).
        thv = th * (1.0 + wc.CM1_REPS * q) / (1.0 + q)
        s = column.surface
        thv_s = s["theta_k"] * (1.0 + wc.CM1_REPS * s["qv_gkg"] / 1000.0) / (1.0 + s["qv_gkg"] / 1000.0)
        pi = [(s["pressure_hpa"] * 100.0 / wc.CM1_P00) ** (wc.CM1_RD / wc.CM1_CP)]
        zz = [0.0, *z]
        tv = [thv_s, *thv]
        for n in range(1, len(zz)):
            pi.append(pi[-1] - wc.CM1_G * (zz[n] - zz[n - 1]) / (wc.CM1_CP * 0.5 * (tv[n] + tv[n - 1])))
        p = wc.CM1_P00 * np.asarray(pi[1:]) ** (wc.CM1_CP / wc.CM1_RD)
    p = np.asarray(p, dtype=np.float64)
    t = th * (p / wc.CM1_P00) ** (wc.CM1_RD / wc.CM1_CP)
    td = dewpoint_k(p, q)
    u, v = np.asarray(column.u_ms, dtype=np.float64), np.asarray(column.v_ms, dtype=np.float64)

    fig = S.figure((W, H))
    fs = 10.5
    ax = fig.add_axes([0.042, 0.20, 0.36, 0.625])
    ax.set_facecolor("#060a12")
    sk = SkewT(ax)
    sk.background(fs)
    keep = p / 100.0 >= 100.0
    for series, colour in ((t - 273.15, T_COLOUR), (td - 273.15, TD_COLOUR)):
        ax.plot(sk.x(series[keep], p[keep] / 100.0), sk.y(p[keep] / 100.0), color=colour, lw=3.0, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")
    axb = fig.add_axes([0.405, 0.20, 0.045, 0.625])
    axb.set_xlim(0, 1)
    axb.set_ylim(*ax.get_ylim())
    axb.axis("off")
    bz = np.arange(0.0, 17000.0, 1000.0)
    bz = bz[(bz >= z[0]) & (bz <= z[-1])]
    bp = np.interp(bz, z, p / 100.0)
    ok = bp >= 102.0
    bz, bp = bz[ok], bp[ok]
    if bz.size:
        axb.plot([0.30, 0.30], [sk.y(bp[0]), sk.y(bp[-1])], color="#1d2740", lw=1.0, zorder=1)
        axb.barbs(np.full(bz.size, 0.30), sk.y(bp), np.interp(bz, z, u) * MS_TO_KT, np.interp(bz, z, v) * MS_TO_KT,
                  length=6.0, color="#cfd8e6", lw=1.1, zorder=3, sizes=dict(emptybarb=0.15, spacing=0.17, height=0.42),
                  pivot="middle")
    for hk in (1, 3, 6, 9, 12, 15):
        ph = float(np.interp(hk * 1000.0, z, p / 100.0))
        if ph >= 103.0 and hk * 1000.0 <= z[-1]:
            axb.text(1.0, sk.y(ph), f"{hk} km", color=S.PALETTE["muted"], fontsize=fs * 0.9, ha="right", va="center")

    # The hodograph: 0-10 km, coloured by height (0-1, 1-3, 3-6, 6-10 km), rings every 10 m/s.
    hz = np.concatenate([[0.0], np.arange(250.0, 10001.0, 250.0)])
    hz = hz[hz <= z[-1]]
    hu = np.interp(hz, z, u)
    hv = np.interp(hz, z, v)
    reach = max(10.0, math.ceil(float(np.max(np.hypot(hu, hv))) / 10.0) * 10.0)
    hx0, hw = 0.475, 0.2
    hh = hw * W / H
    hax = fig.add_axes([hx0, 0.825 - hh, hw, hh])
    hax.set_facecolor("#060a12")
    hax.set_xlim(-reach, reach)
    hax.set_ylim(-reach, reach)
    hax.set_aspect("equal")
    hax.set_xticks([])
    hax.set_yticks([])
    for spine in hax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1f2a40")
    for r in range(10, int(reach) + 1, 10):
        a = np.linspace(0, 2 * np.pi, 181)
        hax.plot(r * np.cos(a), r * np.sin(a), color="#1d2740", lw=0.8, zorder=1)
        hax.text(r * math.cos(math.radians(45)), r * math.sin(math.radians(45)), f"{r}", color=S.PALETTE["dim"],
                 fontsize=fs * 0.85, ha="left", va="bottom", zorder=2)
    hax.axhline(0, color="#1d2740", lw=0.8, zorder=1)
    hax.axvline(0, color="#1d2740", lw=0.8, zorder=1)
    bands = ((0.0, 1000.0, 0.95), (1000.0, 3000.0, 0.75), (3000.0, 6000.0, 0.55), (6000.0, 10000.0, 0.35))
    ramp = S.RAMPS["ember"]
    for lo, hi, shade in bands:
        m = (hz >= lo) & (hz <= hi)
        if m.sum() >= 2:
            hax.plot(hu[m], hv[m], color=ramp.at(shade), lw=3.0, zorder=4, solid_capstyle="round")
    for hk in (1, 3, 6):
        if hk * 1000.0 <= z[-1]:
            hax.plot([np.interp(hk * 1000.0, z, u)], [np.interp(hk * 1000.0, z, v)], "o", ms=4.5,
                     color=S.PALETTE["text"], zorder=5)
    motion = None
    if report:
        br = (report.get("bunkers_right_u_m_per_s"), report.get("bunkers_right_v_m_per_s"))
        if None not in br:
            motion = [float(br[0]), float(br[1])]
            hax.plot([motion[0]], [motion[1]], marker="o", ms=10, mfc="none", mec=S.PALETTE["gold"], mew=2.0,
                     zorder=6)
    base = S.fig_to_rgb(fig)

    ov = S.Overlay((W, H))
    src = column.source
    where = (f"{column.point['lat']:.3f}, {column.point['lon']:.3f}" if "lat" in column.point
             else f"i {column.point['i']}, j {column.point['j']}")
    ov.title("Your run's column", S.data_line(src["name"], f"{src['producer']}, domain {src['domain_id']}", where,
                                               f"grid point i {column.point['i']}, j {column.point['j']}"))
    ov.clock(_clock_text(src["times"]))
    s = column.surface
    ov.footer([S.data_line(S.SIMULATED + " by your " + src["producer"] + " run, not by GM1", "one column",
                           "winds ground-relative, rotated to the earth"),
               S.data_line("T from θ and P+PB; dewpoint from qv (CM1 getcape.F gettd)",
                           "numbers: CM1 getcape.F via GM1's report", "hodograph 0–10 km, rings every 10 (m/s)")])
    ov.mark()
    sy = int(0.175 * H)
    ov.text((int(0.042 * W), sy - ov.U(6)), "skew-T log-p", "label", colour="muted", anchor="ls")
    ov.text((int(0.4275 * W), sy - ov.U(6)), "wind (kt)", "micro", colour="muted", anchor="ms")
    ov.text((int(hx0 * W), sy - ov.U(6)), "hodograph (m/s), 0–1, 1–3, 3–6, 6–10 km", "label", colour="muted",
            anchor="ls")
    key_y = int(0.86 * H)
    kx = int(0.044 * W)
    for label_text, colour in (("temperature", T_COLOUR), ("dewpoint", TD_COLOUR)):
        ov.rule(kx, key_y - ov.U(6), ov.U(34), thickness=3, colour=colour)
        ov.text((kx + ov.U(44), key_y), label_text, "micro", colour="data", anchor="ls")
        kx += ov.U(44) + ov.measure(label_text, "micro")[0] + ov.U(36)
    if motion is not None:
        ov.text((int(hx0 * W), int((1.0 - 0.825 + hh) * H) + ov.U(30)), "gold ring: storm motion, Bunkers right mover",
                "micro", colour="data", anchor="ls")
    n = wc.summary_numbers(report)
    rows: list[tuple[str, str]] = []

    def fmt(value: Any, unit: str, nd: int = 0) -> str:
        return "not computed" if value is None else f"{S.fmt_num(float(value), nd)} {unit}"

    if n:
        rows += [("CAPE, surface parcel", fmt(n.get("cape_sb"), "J/kg")),
                 ("CIN, surface parcel", fmt(n.get("cin_sb"), "J/kg")),
                 ("0–1 km shear", fmt(n.get("shear_0_1km"), "m/s")),
                 ("0–6 km shear", fmt(n.get("shear_0_6km"), "m/s")),
                 ("0–1 km helicity (right mover)", fmt(n.get("srh_0_1km"), "m²/s²")),
                 ("0–3 km helicity (right mover)", fmt(n.get("srh_0_3km"), "m²/s²"))]
    rows.append(("surface pressure", f"{s['pressure_hpa']:.1f} hPa"))
    rows.append(("surface θ and vapour", f"{s['theta_k']:.1f} K, {s['qv_gkg']:.1f} g/kg"))
    ov.readout((int(0.715 * W), int(0.20 * H)), rows)
    out = Path(out)
    P.save(ov, base, out)
    return {"file": str(out), "size": [W, H], "numbers": n, "storm_motion": motion}


__all__ = ["MAP_FIELDS", "MAP_SIZE", "map_png", "preview_png"]
