"""Slices of any cm1out field: horizontal at a level, or vertical along a line.

The fields are the ones ``gpuwm-cm1 cm1out`` writes for the run (:func:`cm1core.cm1out.fields.plan`), under CM1's
names, units and grids, read from the same frame words, so a slice shows exactly what the cm1out file holds.  CM1
writes u and v relative to the moving domain (writeout.F writes ``ua``; umove and vmove are separate variables),
and the pictures say so.

Positions are in km in the run's own coordinates (the namelist's ``iorigin``: with ``iorigin = 2`` the domain
centre is x = y = 0), as ``xh``/``yh`` in the cm1out file.  Heights are the real heights of the field's own levels
(``zh`` for scalars, u and v; ``zf`` for w).  A horizontal slice is never interpolated in the vertical: it takes
the level nearest the height asked for and says which.  Pictures are sampled bilinearly, nothing more.

Each call writes one PNG under ``quicklook/slices/`` and returns a JSON record with the picture's geometry (the
pixel box of the map and its extent in km), so a screen can turn a click into a position.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..cm1out import fields as _fields
from ..cm1out import settings as _settings
from . import products as P
from . import reflectivity as R
from . import style as S
from .facts import RunFacts
from .frame import QuickFrame

VERTICAL_SPEEDS = {"w", "winterp", "sus", "sus2", "wmax", "wmin"}
GROUND_SPEEDS = {"sws", "sws2"}
FIXED = {  # name -> (ramp, vmin, vmax): stated ranges shared with the quick-looks
    "w": ("w_sym", -P.W_RANGE, P.W_RANGE), "winterp": ("w_sym", -P.W_RANGE, P.W_RANGE),
    "thpert": ("theta_sym", -P.THETA_RANGE, P.THETA_RANGE),
}
TOP_M = 18000.0


class SliceError(ValueError):
    """The slice asked for cannot be drawn; the message says why and what to ask instead."""


@dataclass
class Field:
    name: str
    description: str
    units: str
    grid: str                 # "2", "s", "u", "v" or "w"
    read: Any

    def levels(self, qf: QuickFrame) -> np.ndarray | None:
        if self.grid == "2":
            return None
        return qf.zf if self.grid == "w" else qf.zh


def _reflectivity_reader(qf: QuickFrame):
    """cm1out's reflectivity hook, filled by the quick-look operator when the build has no cm1out one."""

    cache: dict[str, Any] = {}

    def volume(_frame: Any) -> Any:
        if "dbz" not in cache:
            cache["dbz"] = R.volume(qf).dbz
        return cache["dbz"]

    def reader(frame: Any, name: str):
        # Offered when the frame can give it; computed only when a slice of it is drawn.
        if name not in ("dbz", "cref") or not R.possible(qf):
            return None
        return volume if name == "dbz" else (lambda f: volume(f).max(axis=0))

    return reader


def fields(qf: QuickFrame, settings: Any) -> dict[str, Field]:
    """Every cm1out field of this run's namelist that this frame can give, by CM1 name."""

    if settings is None:
        raise SliceError("The run has no CM1 namelist or GM1 deck beside its frames to say which cm1out fields it "
                         "writes; pass --namelist FILE or --deck FILE.")
    species = _settings.species(qf.frame)
    plan = _fields.plan(qf.frame, settings, qf.geometry, species, reflectivity=_reflectivity_reader(qf))
    out = {}
    for row in plan.variables:
        v = row.variable
        if v.grid in ("2", "s", "u", "v", "w"):
            out[v.name] = Field(v.name, v.long_name, v.units, v.grid, row.read)
    return out


def field_list(qf: QuickFrame, settings: Any) -> list[dict[str, Any]]:
    rows = []
    for f in fields(qf, settings).values():
        z = f.levels(qf)
        rows.append({"name": f.name, "description": f.description, "units": f.units, "grid": f.grid,
                     "levels_m": None if z is None else [round(float(v), 2) for v in z]})
    return rows


def _speed_frame(name: str, units: str, qf: QuickFrame) -> tuple[str, bool]:
    """(the words that say which frame a speed is in, whether it is a vertical velocity)."""

    if units.replace(" ", "") not in ("ms-1", "m/s"):
        return "", False
    if name in VERTICAL_SPEEDS:
        return "", True
    if name in GROUND_SPEEDS or not qf.domain_moves:
        return "ground-relative", False
    return "relative to the moving domain (CM1's u and v; add umove, vmove for ground-relative)", False


def _look(name: str, units: str, data: np.ndarray) -> tuple[str, float, float, str]:
    """(ramp, vmin, vmax, how the range was chosen)."""

    if name in ("dbz", "cref"):
        return "nws", 5.0, 75.0, "NWS reflectivity colours"
    if name in FIXED:
        ramp, lo, hi = FIXED[name]
        return ramp, lo, hi, f"range {S.fmt_num(lo)} to {S.fmt_num(hi)} fixed"
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return "ember", 0.0, 1.0, "no finite values"
    lo, hi = float(finite.min()), float(finite.max())
    if lo < 0.0 < hi:
        m = max(abs(lo), abs(hi))
        m = nice_ceiling(m) if m > 0 else 1.0
        return "signed", -m, m, f"range ±{m:g} from this slice's largest value"
    if hi == lo:
        hi = lo + (abs(lo) * 0.01 or 1.0)
    return "ember", lo, hi, f"range {S.fmt_num(lo, 3 if abs(hi) < 1 else 1)} to {S.fmt_num(hi, 3 if abs(hi) < 1 else 1)} from this slice"


def _colour(ramp: str, values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if ramp == "nws":
        c = S.dbz_colours(values)
        void = np.array(S.hex_rgb("void"), np.float32)
        return S.to8(c[..., :3] * c[..., 3:4] + void * (1 - c[..., 3:4]))
    return S.to8(S.RAMPS[ramp].colours(values, lo, hi))


def _to_mass(name: str, grid: str, data: np.ndarray) -> np.ndarray:
    """A u- or v-grid plane averaged to scalar points (for drawing only; the values are CM1's)."""

    if grid == "u":
        return 0.5 * (data[..., :, 1:] + data[..., :, :-1])
    if grid == "v":
        return 0.5 * (data[..., 1:, :] + data[..., :-1, :])
    return data


def _legend(ov: S.Overlay, ramp: str, box: P.Box, lo: float, hi: float, title: str, vertical: bool) -> None:
    if ramp == "nws":
        P.dbz_legend(ov, box.x + box.w - ov.U(300), box.y + box.h + ov.U(12), ov.U(300))
        return
    nd = 0 if max(abs(lo), abs(hi)) >= 10 else 2 if max(abs(lo), abs(hi)) >= 0.1 else 4
    ticks = [(lo, S.fmt_num(lo, nd)), ((lo + hi) / 2, S.fmt_num((lo + hi) / 2, nd)), (hi, S.fmt_num(hi, nd))]
    P.ramp_legend(ov, ramp, box, lo, hi, ticks, title=title, vertical=vertical)


def _parse_units(units: str) -> str:
    return {"m s-1": "m/s", "K": "K", "kg kg-1": "kg/kg", "Pa": "Pa", "dBZ": "dBZ", "s-1": S.PER_S,
            "m2 s-2": "m² s⁻²", "cm": "cm"}.get(units, units)


def nice_ceiling(v: float) -> float:
    """The smallest of 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 (times a power of ten) not below ``v``."""

    mag = 10 ** math.floor(math.log10(v))
    for f in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if f * mag >= v * (1 - 1e-9):
            return f * mag
    return 10 * mag


def fmt_value(v: float) -> str:
    """Three significant figures, with a true minus sign."""

    if v == 0 or not math.isfinite(v):
        return S.fmt_num(v)
    digits = max(0, 2 - int(math.floor(math.log10(abs(v)))))
    return S.fmt_num(v, min(digits, 6))


def slice_box(extent: tuple[float, float]) -> P.Box:
    """The map of a slice: left of centre, with room below for the km axis and to the right for the key."""

    return P.fit_box(extent[0], extent[1], P.W // 2 - 80, 175, 740, 740)


def horizontal(qf: QuickFrame, facts: RunFacts, settings: Any, name: str, out_dir: Path, *,
               height_m: float | None = None, level: int | None = None,
               line_km: tuple[float, float, float, float] | None = None, tag: str = "") -> dict[str, Any]:
    """A map of ``name`` at the level nearest ``height_m`` (or CM1 level ``level``, counted from 1)."""

    table = fields(qf, settings)
    if name not in table:
        raise SliceError(f"{name} is not a cm1out field of this run. Fields: {', '.join(sorted(table))}.")
    fld = table[name]
    data = np.asarray(fld.read(qf.frame), dtype=np.float32)
    z = fld.levels(qf)
    if z is None:
        plane, k, z_here = data, None, None
        out = Path(out_dir) / f"slice-{name}-{tag}.png"
    else:
        if level is not None:
            if not 1 <= level <= len(z):
                raise SliceError(f"{name} has levels 1 to {len(z)}; there is no level {level}.")
            k = level - 1
        else:
            k = int(np.argmin(np.abs(z - (height_m if height_m is not None else 1000.0))))
        plane = data[k]
        z_here = float(z[k])
        out = Path(out_dir) / f"slice-{name}-k{k + 1}-{tag}.png"
    plane = _to_mass(name, fld.grid, plane)
    box = slice_box(qf.extent)
    ramp, lo, hi, how = _look(name, fld.units, plane)
    img = _colour(ramp, P.sample(plane, box, qf.dx, qf.dy), lo, hi)
    img, spacing = P.ground_ticks(img, box, qf.domainlocx, qf.domainlocy)
    cv = P.canvas()
    P.paste(cv, img, box)
    cv = P.outline(cv, box)
    frame_words, vertical = _speed_frame(name, fld.units, qf)
    short_frame = frame_words.split(" (")[0] if frame_words else ""
    units = _parse_units(fld.units)
    where = ("at the surface" if z_here is None else
             f"level {k + 1} of {len(z)} ({'w faces' if fld.grid == 'w' else 'scalar levels'}), "
             f"{S.fmt_height(z_here)} above ground")
    ov = S.Overlay((P.W, P.H))
    title = f"{name} at the surface" if z_here is None else f"{name} at {S.fmt_height(z_here)}"
    ov.title(title + (", section line" if line_km is not None else ""),
             S.data_line(facts.name, f"{fld.description} ({units})", where))
    ov.clock(qf.t)
    ox, oy = facts.origin_m(qf.nx, qf.ny)
    ov.footer(facts.footer(t=qf.t, level=(f"level {S.fmt_height(z_here)}" if z_here is not None else "surface"),
                           wind=short_frame,
                           notes=S.data_line(f"cm1out's {name}", how,
                                             "no vertical interpolation" if z_here is not None else "",
                                             "x, y in km as the cm1out file's xh, yh",
                                             f"+ fixed to the ground every {S.fmt_km(spacing)}")))
    ov.mark()
    lo_v, hi_v = float(np.nanmin(plane)), float(np.nanmax(plane))
    rows = [("smallest", f"{fmt_value(lo_v)} {units}"), ("largest", f"{fmt_value(hi_v)} {units}")]
    rx = box.x + box.w + ov.U(48)
    ov.readout((rx, box.y), rows, vertical=[r[1] for r in rows] if vertical else ())
    if short_frame and not vertical:
        ov.text((rx, box.y + ov.U(150)), f"speeds {short_frame}" + (", as CM1 writes u and v"
                                                                     if "moving domain" in short_frame else ""),
                "micro", colour="muted", anchor="ls")
    legend_box = P.Box(rx, box.y + box.h - ov.U(60), ov.x1 - rx, h=1)
    _legend(ov, ramp, legend_box, lo, hi, f"{name} ({units})" + (f", {short_frame}" if short_frame and not vertical else ""),
            vertical)
    _axes_km(ov, box, ox, oy)
    record: dict[str, Any] = {}
    if line_km is not None:
        (ax, ay, bx, by) = line_km
        pa = box.to_px(ax * 1000.0 - ox, ay * 1000.0 - oy)
        pb = box.to_px(bx * 1000.0 - ox, by * 1000.0 - oy)
        _draw_line(ov, pa, pb)
        record["line_km"] = list(line_km)
    P.save(ov, cv, out)
    record.update({
        "kind": "horizontal", "field": name, "units": fld.units, "description": fld.description,
        "time_s": qf.t, "frame": qf.key, "level": None if k is None else k + 1, "height_m": z_here,
        "min": lo_v, "max": hi_v, "range": [lo, hi], "range_note": how,
        "file": out.name, "map": _geometry(box, qf, ox, oy),
    })
    return record


def _axes_km(ov: S.Overlay, box: P.Box, ox: float, oy: float) -> None:
    """km labels along the bottom and left edges of a map, in the run's coordinates."""

    step = P.nice_step(box.span, 6)
    x0 = ox + box.mx0
    for xm in np.arange(math.ceil(x0 / step) * step, x0 + box.span + 1e-6, step):
        px, _ = box.to_px(xm - ox, box.my0)
        ov.text((px, box.y + box.h + ov.U(22)), S.fmt_num(xm / 1000.0, 0 if step >= 1000 else 1), "micro",
                colour="muted", anchor="ms", halo=False)
    y0 = oy + box.my0
    for ym in np.arange(math.ceil(y0 / step) * step, y0 + box.span_y + 1e-6, step):
        _, py = box.to_px(box.mx0, ym - oy)
        ov.text((box.x - ov.U(8), py), S.fmt_num(ym / 1000.0, 0 if step >= 1000 else 1), "micro", colour="muted",
                anchor="rm", halo=False)
    ov.text((box.x + box.w / 2, box.y + box.h + ov.U(46)), "x (km)", "micro", colour="muted", anchor="ms", halo=False)
    ov.text((box.x - ov.U(44), box.y + box.h / 2), "y (km)", "micro", colour="muted", anchor="rm", halo=False)


def _geometry(box: P.Box, qf: QuickFrame, ox: float, oy: float) -> dict[str, Any]:
    """Where the map is in the picture and which km it shows: a click at pixel (px, py) is at
    x = x_km[0] + (px - left) / width * (x_km[1] - x_km[0]), y = y_km[1] - (py - top) / height * (...)."""

    return {"picture_px": [P.W, P.H], "left": box.x, "top": box.y, "width": box.w, "height": box.h,
            "x_km": [(ox + box.mx0) / 1000.0, (ox + box.mx0 + box.span) / 1000.0],
            "y_km": [(oy + box.my0) / 1000.0, (oy + box.my0 + box.span_y) / 1000.0],
            "dx_km": qf.dx / 1000.0, "dy_km": qf.dy / 1000.0}


def _draw_line(ov: S.Overlay, pa: tuple[float, float], pb: tuple[float, float]) -> None:
    from PIL import Image, ImageDraw

    ss = 3
    img = Image.new("RGBA", (ov.W * ss, ov.H * ss), (0, 0, 0, 0))
    ImageDraw.Draw(img).line((pa[0] * ss, pa[1] * ss, pb[0] * ss, pb[1] * ss), fill=S.rgb8("gold", 0.95),
                             width=max(1, ov.U(2.5)) * ss)
    ov.draw_rgba(img.resize((ov.W, ov.H), Image.LANCZOS), (0, 0))
    ov.text((pa[0] - ov.U(10), pa[1]), "A", "label", colour="gold", anchor="rm")
    ov.text((pb[0] + ov.U(10), pb[1]), "B", "label", colour="gold", anchor="lm")


def vertical(qf: QuickFrame, facts: RunFacts, settings: Any, name: str, out: Path, *,
             a_km: tuple[float, float], b_km: tuple[float, float], top_m: float | None = None) -> dict[str, Any]:
    """A section of ``name`` along the line A-B (km, run coordinates), from the ground to ``top_m``."""

    table = fields(qf, settings)
    if name not in table:
        raise SliceError(f"{name} is not a cm1out field of this run. Fields: {', '.join(sorted(table))}.")
    fld = table[name]
    z = fld.levels(qf)
    if z is None:
        raise SliceError(f"{name} is a surface field; a vertical section needs a 3-D field.")
    ox, oy = facts.origin_m(qf.nx, qf.ny)
    ax, ay = a_km[0] * 1000.0 - ox, a_km[1] * 1000.0 - oy
    bx, by = b_km[0] * 1000.0 - ox, b_km[1] * 1000.0 - oy
    length = math.hypot(bx - ax, by - ay)
    if length < 2 * min(qf.dx, qf.dy):
        raise SliceError("A and B are closer than two grid cells; draw a longer line.")
    lx, ly = qf.extent
    for x, y in ((ax, ay), (bx, by)):
        if not (0.0 <= x <= lx and 0.0 <= y <= ly):
            raise SliceError("A and B must both be inside the domain.")
    data = _to_mass(name, fld.grid, np.asarray(fld.read(qf.frame), dtype=np.float32))
    top = min(float(top_m or TOP_M), float(qf.zf[-1]))
    x0, y0, pw, ph = 150, 230, 1620, 600
    s = (np.arange(pw) + 0.5) / pw
    xs, ys = ax + s * (bx - ax), ay + s * (by - ay)
    ci, cj = xs / qf.dx - 0.5, ys / qf.dy - 0.5
    zs = (1.0 - (np.arange(ph) + 0.5) / ph) * top
    sec = _section(data, z, zs, cj, ci)
    ramp, lo, hi, how = _look(name, fld.units, sec)
    img = _colour(ramp, sec, lo, hi)
    cloud_note = ""
    if qf.has_species("qc") and qf.has_species("qi"):
        cloud = qf.species("qc") + qf.species("qi")
        m = _section(cloud, qf.zh, zs, cj, ci) >= 1e-4
        edge = np.zeros_like(m)
        edge[:, 1:] |= m[:, 1:] != m[:, :-1]
        edge[1:, :] |= m[1:, :] != m[:-1, :]
        img[edge] = S.rgb8("cloud")
        cloud_note = "silver line: cloud water plus cloud ice = 0.1 g/kg"
    cv = P.canvas()
    cv[y0:y0 + ph, x0:x0 + pw] = img
    box = P.Box(x0, y0, pw, 0.0, 0.0, length, h=ph)
    cv = P.outline(cv, box)
    stretch = (ph / top) / (pw / length)
    frame_words, vert = _speed_frame(name, fld.units, qf)
    units = _parse_units(fld.units)
    ov = S.Overlay((P.W, P.H))
    ov.title("Vertical section A–B",
             S.data_line(facts.name, f"{fld.description} ({units})",
                         f"A ({S.fmt_num(a_km[0], 1)}, {S.fmt_num(a_km[1], 1)}) km to B ({S.fmt_num(b_km[0], 1)}, "
                         f"{S.fmt_num(b_km[1], 1)}) km", f"vertical scale {stretch:.1f} × horizontal"))
    ov.clock(qf.t)
    ov.footer(facts.footer(t=qf.t, level=f"ground to {S.fmt_height(top)}",
                           wind=frame_words.split(" (")[0] if frame_words else "",
                           notes=S.data_line(f"cm1out's {name}", how, "bilinear between levels and columns",
                                             cloud_note)))
    ov.mark()
    for zk in P_levels(top):
        yy = y0 + ph - zk / top * ph
        ov.text((x0 - ov.U(12), yy), S.fmt_height(zk), "micro", colour="muted", anchor="rm", halo=False)
    step = P.nice_step(length, 8)
    for d in np.arange(0.0, length + 1e-6, step):
        xx = x0 + d / length * pw
        ov.text((xx, y0 + ph + ov.U(24)), S.fmt_num(d / 1000.0, 0 if step >= 1000 else 1), "micro", colour="muted",
                anchor="ms", halo=False)
    ov.text((x0 + pw / 2, y0 + ph + ov.U(48)), "distance from A (km)", "micro", colour="muted", anchor="ms", halo=False)
    ov.text((x0, y0 - ov.U(12)), "A", "label", colour="gold", anchor="ls")
    ov.text((x0 + pw, y0 - ov.U(12)), "B", "label", colour="gold", anchor="rs")
    legend_box = P.Box(x0, y0 + ph + ov.U(58), pw, h=1)
    _legend(ov, ramp, legend_box, lo, hi, f"{name} ({units})" + (f", {frame_words.split(' (')[0]}" if frame_words and not vert else ""), vert)
    P.save(ov, cv, out)
    return {"kind": "vertical", "field": name, "units": fld.units, "description": fld.description, "time_s": qf.t,
            "frame": qf.key, "a_km": list(a_km), "b_km": list(b_km), "length_km": length / 1000.0,
            "top_m": top, "range": [lo, hi], "range_note": how, "min": float(sec.min()), "max": float(sec.max()),
            "file": out.name}


def _section(data: np.ndarray, z: np.ndarray, zs: np.ndarray, cj: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """Bilinear along the line at every level, then linear in height between levels: (len(zs), len(ci))."""

    nz = data.shape[0]
    along = np.stack([P.bilinear(data[k], cj, ci) for k in range(nz)])      # (nz, npts)
    kk = np.interp(zs, z, np.arange(nz, dtype=np.float64))
    k0 = np.minimum(np.floor(kk).astype(np.intp), max(nz - 2, 0))
    w = (kk - k0).astype(np.float32)[:, None]
    k1 = np.minimum(k0 + 1, nz - 1)
    return (along[k0] * (1 - w) + along[k1] * w).astype(np.float32)


def P_levels(top: float) -> list[float]:
    step = P.nice_step(top, 6)
    return [float(v) for v in np.arange(0.0, top + 1e-6, step)]


__all__ = ["Field", "SliceError", "field_list", "fields", "horizontal", "vertical"]
