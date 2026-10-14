"""A hurricane run's two looks by radius and height, drawn from the rings around its pressure centre
(:mod:`.azimuthal`):

``swirl-by-radius``  Swirling wind by radius and height: the wind around the centre, averaged over each ring, from
                     the sea to 20 km and out to 300 km, on a fixed range of -20 to 80 m/s ground-relative (the
                     wind-lowest look's colours above zero), with a line every 10 m/s and a gold ring at the
                     strongest;
``warm-core``        Warm core by radius and height: the ring-mean potential temperature minus its mean over the
                     rings 300 to 450 km out, at each height, on a fixed range of -10 to +25 K (its colours change
                     faster near zero, and the bar under the picture says where) with a line every 2 K; a gold bar
                     marks the warm core's number (the mean within 40 km, at its warmest height).

Both are drawn linearly between ring middles and between levels, and nothing else is done to the numbers.  The
first ring is the centre cell alone, so it is drawn at the centre itself (:func:`drawn_radii`): the swirl there is
nil by definition, and drawn at that ring's middle it would be held flat out to half a grid spacing, where a line
at 0 m/s would stand on nothing.  The numbers (the radius of maximum wind, the strongest swirl, the warm core) keep
CM1's ring middles.

A run with the 8 km hurricane preset's settings (:func:`.facts.hurricane_checked` says it is compared) gets a second
panel beside GM1's: CM1 r21.1's own run with those settings and its own adaptive time step, through the same rings
of its saved 3-D output (``records/CM1-HURRICANE-8KM-RINGS.json``, every 3 hours to hour 96, made by
``tools/generate_hurricane_rings.py``), at the same storm time or its nearest saved time within 1.5 hours, which
the panel then names.  Any other run has GM1's panel alone, and the picture says CM1 has not been run with its
settings.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import products as P
from . import style as S
from .azimuthal import CORE_RADIUS, Rings

#: The rings drawn: out to this radius (m), from the sea to this height (m) or the model's top.
RADIUS_M = 300_000.0
TOP_M = 20_000.0
#: Fixed, stated ranges: the swirling wind, m/s ground-relative (above zero the wind-lowest look's colours, so a
#: speed is the same colour in both), and the warm core, K.  They hold GM1's and CM1's hurricane runs (measured on
#: every saved frame of four GM1 runs at 8 and 4 km and of CM1's run, over what the looks draw: the swirl from -6.9
#: to 64.8 m/s, the warm core's rings from -8.6 to +22.7 K, its number at most 16.5 K).
SWIRL_RANGE = (-20.0, 80.0)
WARM_RANGE = (-10.0, 25.0)
SWIRL_LINES = 10.0
WARM_LINES = 2.0
#: CM1's own rings of the compared run, and how far (s) a GM1 time may be from CM1's nearest saved time.
CM1_RECORD = "CM1-HURRICANE-8KM-RINGS.json"
CM1_RECORD_SCHEMA = "gm1.cm1-hurricane-rings/v1"
CM1_NEAREST_S = 5400.0

# Two panels (GM1 and CM1) or one, in pixels of the 1920 x 1080 picture.
PANEL_Y, PANEL_H = 236, 530
TWO = ((170, 770), (1070, 770))
ONE = ((610, 1230),)


@lru_cache(maxsize=1)
def cm1_record() -> Mapping[str, Any]:
    """CM1's own rings of its run with the 8 km hurricane preset's settings (scaled integers, as written)."""

    text = resources.files("cm1core.records").joinpath(CM1_RECORD).read_text(encoding="utf-8")
    record = json.loads(text)
    if record.get("schema") != CM1_RECORD_SCHEMA:
        raise ValueError(f"{CM1_RECORD} has an unknown schema")
    return record


def cm1_at(t: float) -> tuple[Mapping[str, Any], dict[str, np.ndarray]] | None:
    """CM1's saved time nearest ``t`` (within :data:`CM1_NEAREST_S`), with its fields as arrays, or None."""

    record = cm1_record()
    rows = record["times"]
    best = min(rows, key=lambda row: abs(float(row["t_s"]) - t))
    if abs(float(best["t_s"]) - t) > CM1_NEAREST_S:
        return None
    scale = record["scale"]
    fields = {name: np.asarray(best[name], dtype=np.float64) * float(scale[name]) for name in ("swirl", "warm")}
    fields["radius_m"] = np.asarray(record["radius_m"], dtype=np.float64)
    fields["z_m"] = np.asarray(record["z_m"], dtype=np.float64)
    return best, fields


def compact(rings: Rings, *, scale: float = 0.01) -> dict[str, Any]:
    """The rings a picture draws, for a record: those out to one ring past :data:`RADIUS_M`, the levels to the first
    above :data:`TOP_M`, in integers of ``scale``."""

    n = int(np.searchsorted(rings.radius_m, RADIUS_M, side="right")) + 1
    k = int(np.searchsorted(rings.z_m, TOP_M, side="right")) + 1
    as_int = lambda a: np.rint(np.asarray(a)[:k, :n] / scale).astype(int).tolist()  # noqa: E731
    return {"radius_m": rings.radius_m[:n].tolist(), "z_m": rings.z_m[:k].tolist(),
            "swirl": as_int(rings.swirl), "warm": as_int(rings.warm)}


def sample_rz(field: np.ndarray, radius_m: np.ndarray, z_m: np.ndarray, rs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """``field`` (level, ring) at radii ``rs`` and heights ``zs``: linear between ring middles and between levels,
    held beyond the first and the last."""

    ri = np.interp(rs, radius_m, np.arange(radius_m.size, dtype=np.float64))
    zi = np.interp(zs, z_m, np.arange(z_m.size, dtype=np.float64))
    rows = np.broadcast_to(zi[:, None], (zs.size, rs.size))
    cols = np.broadcast_to(ri[None, :], (zs.size, rs.size))
    return P.bilinear(np.asarray(field, dtype=np.float32), rows, cols)


def drawn_radii(radius_m: np.ndarray) -> np.ndarray:
    """The radii (m) the rings are drawn at: their middles, but the first ring, the centre cell alone, at the centre
    (0 m), where that cell is."""

    radii = np.asarray(radius_m, dtype=np.float64).copy()
    radii[0] = 0.0
    return radii


def _top(L: "P.Looks") -> float:
    return min(TOP_M, float(L.f.zf[-1]))


def _reach(rings_radius: np.ndarray, dr: float) -> float:
    """How far out the panel goes: :data:`RADIUS_M`, or the rings' own reach on a smaller area."""

    return min(RADIUS_M, float(rings_radius[-1] + 0.5 * dr))


def _draw_panel(cv: np.ndarray, x: int, w: int, field: np.ndarray, radius_m: np.ndarray, z_m: np.ndarray,
                reach: float, top: float, ramp: str, lo: float, hi: float, every: float) -> None:
    rs = (np.arange(w) + 0.5) / w * reach
    zs = (1.0 - (np.arange(PANEL_H) + 0.5) / PANEL_H) * top
    values = sample_rz(field, drawn_radii(radius_m), z_m, rs, zs)
    img = S.to8(S.colours(ramp, values, lo, hi))
    img = P.blend(img, P.line_alpha(values, every=every, width=1.3, far=320.0), "title", 0.5)
    cv[PANEL_Y:PANEL_Y + PANEL_H, x:x + w] = img


def _edged_bar(ov: S.Overlay, x: float, y: float, length: float, *, thickness: float = 3.0) -> None:
    """A gold bar (as :meth:`.style.Overlay.rule` draws one) on a dark edge, so it shows on a bright core too."""

    from PIL import Image

    t = max(1, ov.U(thickness))
    e = max(1, ov.U(1.5))
    n = int(length)
    if n > 0:
        ov.draw_rgba(Image.new("RGBA", (n + 2 * e, t + 2 * e), S.rgb8(S.token("void"), 0.95)), (x - e, y - e))
        ov.draw_rgba(Image.new("RGBA", (n, t), S.rgb8(S.token("gold"), 1.0)), (x, y))


def _to_px(x: int, w: int, reach: float, top: float, r: float, z: float) -> tuple[float, float]:
    return x + r / reach * w, PANEL_Y + PANEL_H - z / top * PANEL_H


def _axes(ov: S.Overlay, x: int, w: int, reach: float, top: float, *, heights: bool) -> None:
    from .slices import P_levels

    ms = ov.px("micro")
    for z in P_levels(top):
        if z > 0 and heights:
            y = PANEL_Y + PANEL_H - z / top * PANEL_H
            ov.text((x - ov.U(10), y), S.fmt_height(z), "micro", colour="muted", anchor="rm", halo=False)
    step = P.nice_step(reach, 6)
    for r in np.arange(0.0, reach + 1e-6, step):
        ov.text((x + r / reach * w, PANEL_Y + PANEL_H + ov.U(10) + ms), S.fmt_num(r / 1000.0), "micro",
                colour="muted", anchor="ms", halo=False)


def _cm1_words(row: Mapping[str, Any], t: float) -> str:
    when = S.fmt_long_clock(float(row["t_s"]))
    gap = float(row["t_s"]) - t
    if abs(gap) < 1.0:
        return f"CM1 r21.1 at {when}"
    side = "later" if gap > 0 else "earlier"
    return f"CM1 r21.1 at {when}, its nearest saved time ({S.fmt_num(abs(gap) / 3600.0, 1)} h {side})"


def _layout(L: "P.Looks") -> tuple[tuple[tuple[int, int], ...], bool]:
    from .facts import HURRICANE_CHECKED

    # CM1's own output (Open CM1 output) is never drawn beside CM1's record: it would be compared with itself
    compared = L.facts.checked == HURRICANE_CHECKED and _model(L) == "GM1"
    return (TWO if compared else ONE), compared


def _model(L: "P.Looks") -> str:
    """The model that ran the storm: "GM1", or another's own output read from its netCDF files ("CM1")."""

    return str(getattr(L.facts, "model", "GM1") or "GM1")


def _this_run(L: "P.Looks") -> str:
    """The words over the run's own panel: GM1's run, or another model's own output and which model it is."""

    if _model(L) == "GM1":
        return "GM1, this run"
    return f"{getattr(L.facts, 'made_by', '') or _model(L)}, this output"


def _rings_or_why(L: "P.Looks") -> Rings:
    if not L.hurricane:
        raise P.NotDrawn("This look is drawn for a hurricane run.")
    rings = L.rings
    if rings is None:
        raise P.NotDrawn("This run's saved frames carry no pressure, so there is no centre to draw rings around.")
    return rings


def _look(L: "P.Looks", out: Path, product: str, *, field: str, ramp: str, span: tuple[float, float],
          every: float, notes: str, readout: list[tuple[str, str]], mark: Any, label: Any,
          bar_ticks: list[tuple[float, str]], bar_title: str, lines: list[tuple[str, str]]) -> dict[str, Any]:
    """The shared grammar of the two looks.  ``mark(ov, x, w, reach, top, values)`` draws the gold mark on one
    panel; ``label(values)`` is the number over a panel; ``values`` is (rings or CM1's fields)."""

    rings = _rings_or_why(L)
    f = L.f
    top = _top(L)
    reach = _reach(rings.radius_m, rings.dr)
    panels, compared = _layout(L)
    cv = P.canvas()
    gm1_field = getattr(rings, field)
    (gx, gw) = panels[0]
    _draw_panel(cv, gx, gw, gm1_field, rings.radius_m, rings.z_m, reach, top, ramp, span[0], span[1], every)
    cm1 = cm1_at(f.t) if compared else None
    if compared and cm1 is not None:
        row, fields = cm1
        (cx, cw) = panels[1]
        _draw_panel(cv, cx, cw, fields[field], fields["radius_m"], fields["z_m"], reach, top, ramp, span[0],
                    span[1], every)
    for x, w in panels:
        cv = P.outline(cv, P.Box(x, PANEL_Y, w, 0.0, 0.0, 1.0, h=PANEL_H))
    z0 = L.z0
    ov = P.overlay(L, product,
                   L.data_line(f"rings {S.fmt_km(rings.dr)} wide around the pressure centre"),
                   level=f"sea to {S.fmt_height(top)}", wind="ground-relative" if field == "swirl" else "",
                   notes=notes)
    facts: dict[str, Any] = {"centre_xy_m": [float(rings.centre_m[0]), float(rings.centre_m[1])],
                             "ring_m": rings.dr, "radius_m": reach, "top_m": top, "range": list(span),
                             "lines_every": every, "height_m": z0, "compared_with_cm1": compared}
    ov.text((gx, PANEL_Y - ov.U(12)), _this_run(L), "label", colour="text", anchor="ls")
    ov.text((gx + gw, PANEL_Y - ov.U(12)), label(rings), "label", colour="data", anchor="rs")
    mark(ov, gx, gw, reach, top, rings)
    _axes(ov, gx, gw, reach, top, heights=True)
    ms = ov.px("micro")
    if compared:
        (cx, cw) = panels[1]
        _axes(ov, cx, cw, reach, top, heights=True)
        if cm1 is not None:
            row, fields = cm1
            view = _Cm1View(row, fields)
            ov.text((cx, PANEL_Y - ov.U(12)), _cm1_words(row, f.t), "label", colour="text", anchor="ls")
            ov.text((cx + cw, PANEL_Y - ov.U(12)), label(view), "label", colour="data", anchor="rs")
            mark(ov, cx, cw, reach, top, view)
            facts["cm1_time_s"] = float(row["t_s"])
        else:
            ov.text((cx, PANEL_Y - ov.U(12)), "CM1 r21.1", "label", colour="text", anchor="ls")
            ov.text((cx + cw / 2, PANEL_Y + PANEL_H / 2), "CM1's run ended at hour 96: nothing to compare here",
                    "label", colour="muted", anchor="ms")
            facts["cm1_time_s"] = None
        caption_x = cx + cw
    else:
        ov.readout((ov.x0, PANEL_Y + ov.U(4)), readout)
        if _model(L) == "GM1":
            ov.text((ov.x0, PANEL_Y + PANEL_H - ov.U(4)), "CM1 has not been run with these settings", "micro",
                    colour="muted", anchor="ls")
        caption_x = gx + gw
    ov.text((caption_x, PANEL_Y + PANEL_H + ov.U(18) + 2 * ms), "radius from the pressure centre, km", "micro",
            colour="muted", anchor="rs", halo=False)
    by = PANEL_Y + PANEL_H + ov.U(30) + 3 * ms
    ov.ramp_bar(ramp, (caption_x - ov.U(420), by, caption_x, by + max(3, ov.U(8))), vmin=span[0], vmax=span[1],
                ticks=bar_ticks, title=bar_title)
    if compared:
        lines = [*lines, ("right: CM1 r21.1's own run with these settings and its own adaptive time step, the same "
                          "rings of its saved 3-D output", "data")]
    for n, (text, colour) in enumerate(lines[:3]):
        ov.text((gx, by + n * int(ms * 1.6) + ms // 2), text, "micro", colour=colour, anchor="ls", halo=False)
    P.save(ov, cv, out)
    return facts


def _centre_words(L: "P.Looks", rings: Rings) -> str:
    """Where the pressure centre is from the middle of the area: "12 km north-west", "at the middle"."""

    east = (rings.centre_m[0] - 0.5 * L.f.extent[0]) / 1000.0
    north = (rings.centre_m[1] - 0.5 * L.f.extent[1]) / 1000.0
    words = P._from_centre(east, north, "")
    return "at the middle" if words.startswith("at ") else words.removesuffix(" of ").strip()


class _Cm1View:
    """CM1's rings from the record, with the few numbers the looks say (as :class:`.azimuthal.Rings` has them)."""

    def __init__(self, row: Mapping[str, Any], fields: Mapping[str, np.ndarray]) -> None:
        self.row = row
        self.swirl = fields["swirl"]
        self.warm = fields["warm"]
        self.radius_m = fields["radius_m"]
        self.z_m = fields["z_m"]

    def strongest_swirl(self) -> tuple[float, float, float]:
        return (float(self.row["swirl_max"]), float(self.row["swirl_max_radius_m"]),
                float(self.row["swirl_max_height_m"]))

    @property
    def rmw_m(self) -> float:
        return float(self.row["rmw_m"])

    def warm_core(self) -> tuple[float, float, int]:
        z = float(self.row["warm_core_height_m"])
        return float(self.row["warm_core_k"]), z, int(np.argmin(np.abs(self.z_m - z)))


def product_swirl_by_radius(L: "P.Looks", out: Path) -> dict[str, Any]:
    rings = _rings_or_why(L)
    lo, hi = SWIRL_RANGE
    z0 = L.z0
    speed, at_r, at_z = rings.strongest_swirl()

    def label(view: Any) -> str:
        s, _r, _z = view.strongest_swirl()
        return f"strongest {s:.0f} m/s ground-relative"

    def mark(ov: S.Overlay, x: int, w: int, reach: float, top: float, view: Any) -> None:
        _s, r, z = view.strongest_swirl()
        if r <= reach and z <= top:
            # dark-edged, so the ring shows on the brightest swirl (60 m/s and more is pale yellow) too
            ov.ring(_to_px(x, w, reach, top, r, z), 9.0, edge="void")
        rmw = view.rmw_m
        if rmw <= reach:
            # a tick under the panel's floor at the radius of maximum wind of the lowest level
            px = x + rmw / reach * w
            ov.rule(px - ov.U(1.5), PANEL_Y + PANEL_H + ov.U(1), ov.U(9), thickness=3, colour="gold", vertical=True)

    facts = _look(
        L, out, "swirl-by-radius", field="swirl", ramp="hurricane_swirl", span=SWIRL_RANGE, every=SWIRL_LINES,
        notes=S.data_line("the wind around the centre, averaged over each ring, ground-relative",
                          f"{S.fmt_num(lo)} to {hi:g} m/s fixed", f"lines every {SWIRL_LINES:g} m/s",
                          "linear between rings and levels"),
        readout=[("strongest swirl, ring mean", f"{speed:.0f} m/s ground-relative"),
                 ("where", f"{at_r / 1000:.0f} km out, {S.fmt_height(at_z)} up"),
                 (f"radius of maximum wind at {S.fmt_height(z0)}", f"{rings.rmw_m / 1000:.0f} km")],
        mark=mark, label=label,
        bar_ticks=[(lo, S.fmt_num(lo)), (0.0, "0"), (20.0, "20"), (40.0, "40"), (60.0, "60"), (hi, f"{hi:g} m/s")],
        bar_title="swirling wind, ground-relative (below 0: turning the other way)",
        lines=[("gold ring: the strongest swirl; gold tick: the radius of maximum wind at "
                f"{S.fmt_height(z0)}", "gold"),
               ("below 0 (green): air turning the other way, as the outflow does high up", "data")])
    facts.update({"swirl_max_ground_relative": speed, "swirl_max_radius_m": at_r, "swirl_max_height_m": at_z,
                  "rmw_m": rings.rmw_m, "swirl_max_lowest": rings.swirl_max_lowest})
    return facts


def product_warm_core(L: "P.Looks", out: Path) -> dict[str, Any]:
    rings = _rings_or_why(L)
    lo, hi = WARM_RANGE
    warm, warm_z, _k = rings.warm_core()
    far = rings.far_m
    far_words = (f"{far[0] / 1000:.0f} to {far[1] / 1000:.0f} km out" if rings.far_is_standard
                 else f"{far[0] / 1000:.0f} to {far[1] / 1000:.0f} km out, the outer third of this smaller area")

    def label(view: Any) -> str:
        k, z, _ = view.warm_core()
        return f"warm core {S.fmt_num(k, 1, sign=True)} K at {S.fmt_height(z)}"

    def mark(ov: S.Overlay, x: int, w: int, reach: float, top: float, view: Any) -> None:
        _k, z, _ = view.warm_core()
        if z <= top:
            x0, y = _to_px(x, w, reach, top, 0.0, z)
            x1, _ = _to_px(x, w, reach, top, min(CORE_RADIUS, reach), z)
            _edged_bar(ov, x0, y - ov.U(1.5), max(2.0, x1 - x0))
            # and a tick outside the panel's left side at that height, where the bar meets a bright core
            ov.rule(x0 - ov.U(9), y - ov.U(1.5), ov.U(8), thickness=3, colour="gold")

    facts = _look(
        L, out, "warm-core", field="warm", ramp="hurricane_warm", span=WARM_RANGE, every=WARM_LINES,
        notes=S.data_line("ring-mean potential temperature minus its mean " + far_words + " at the same height",
                          f"{S.fmt_num(lo)} to +{hi:g} K fixed", f"lines every {WARM_LINES:g} K",
                          "linear between rings and levels"),
        readout=[("warm core", f"{S.fmt_num(warm, 1, sign=True)} K"),
                 ("at", S.fmt_height(warm_z)),
                 ("the rings' centre, from the area's middle", _centre_words(L, rings))],
        mark=mark, label=label,
        bar_ticks=[(lo, S.fmt_num(lo)), (-5.0, S.fmt_num(-5)), (0.0, "0"), (5.0, "+5"), (10.0, "+10"),
                   (15.0, "+15"), (20.0, "+20"), (hi, f"+{hi:g} K")],
        bar_title=f"warmer (+) or colder ({S.MINUS}) than {far[0] / 1000:.0f} to {far[1] / 1000:.0f} km out",
        lines=[(f"gold bar: the warm core's number, the mean within {CORE_RADIUS / 1000:.0f} km of the centre at "
                "its warmest height", "gold"),
               ("potential temperature: the temperature air would have if brought down to 1000 hPa", "data")])
    facts.update({"warm_core_k": warm, "warm_core_height_m": warm_z, "far_m": list(far),
                  "far_is_standard": rings.far_is_standard, "warm_max_k": float(rings.warm.max())})
    return facts


__all__ = ["CM1_RECORD", "RADIUS_M", "SWIRL_RANGE", "TOP_M", "WARM_RANGE", "cm1_at", "cm1_record", "compact",
           "drawn_radii", "product_swirl_by_radius", "product_warm_core", "sample_rz"]
