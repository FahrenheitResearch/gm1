"""The compare pictures (:mod:`.compare`): runs side by side, runs over time, and the soundings card.

Every picture keeps the quick-looks' grammar and rules (:mod:`.products`, :mod:`.style`): the title and data line,
the gold clock, the two-line honesty footer, the GM1 mark, the lint, fixed and stated ranges, no smoothing beyond
bilinear sampling, and the lint's word rules.  Each run keeps one letter and one colour on every picture (the dataviz
reference palette's categorical slots, :data:`.compare.COLOURS`); its colour marks it, and its letter and name say
which it is, so no run is told apart by colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import products as P
from . import style as S

NotDrawn = P.NotDrawn
W, H = P.W, P.H
X0, X1 = 80, W - 80
PANEL_TOP = 228
PANEL_MAX = 530
#: The bottom of the area the pictures fill; the footer is below it.
CONTENT_BOTTOM = 950
TITLES = {"reflectivity": "Side by side: 1 km reflectivity", "clouds": "Side by side: clouds from above",
          "surface": f"Side by side: θ{S.PRIME} and wind", "wind": "Side by side: lowest-level wind"}
HATCH = (91, 100, 119)        # style token "dim"


@dataclass(eq=False)
class Context:
    runs: list
    t: float
    span: float
    differ: list[str]
    unread: list[str]
    hurricane: bool
    states: list | None = None
    series: list | None = None          # [(points, source)] per run
    profiles: list | None = None        # [(profile or None, words)] per run
    environments: list | None = None    # [(numbers or None, source)] per run
    first_look: str = "reflectivity"    # the storm look every run is drawn in: reflectivity, else clouds

    def clock(self, t: float) -> str:
        return S.fmt_long_clock(t) if self.hurricane else S.fmt_clock(t)


# ----------------------------------------------------------------------------------------------- shared words
def _grouped(runs: Sequence[Any], value: Any) -> str:
    """'0.1.2' when every run has one value, else '0.1.0 (A, B), 0.1.2 (C)'."""

    groups: dict[str, list[str]] = {}
    for r in runs:
        groups.setdefault(str(value(r)), []).append(r.letter)
    if len(groups) == 1:
        return next(iter(groups))
    return ", ".join(f"{v} ({', '.join(letters)})" for v, letters in groups.items())


def checks_differ(ctx: Context) -> bool:
    return len({r.facts.checked for r in ctx.runs}) > 1


def check_words(ctx: Context, r: Any) -> str:
    """A run's own check against CM1, said beside it when the runs' checks differ."""

    return f"CM1 check: {r.facts.checked}" if checks_differ(ctx) and r.facts.checked else ""


def footer_facts(ctx: Context, *, level: str, wind: str, t: float | None = None, when: str | None = None) -> str:
    runs = ctx.runs
    lines = {r.facts.scheme_line for r in runs}
    if len(lines) == 1:
        scheme = next(iter(lines))
    else:
        scheme = (runs[0].facts.scheme if len({r.facts.scheme for r in runs}) == 1
                  else "microphysics: " + _grouped(runs, lambda r: r.facts.scheme))
        scheme += ", each run's CM1 check shown with it"
    return S.data_line(S.SIMULATED, "GM1 " + _grouped(runs, lambda r: r.facts.version.split(" (")[0]), scheme,
                       "dx " + _grouped(runs, lambda r: S.fmt_grid(r.dx)), level, f"wind {wind}" if wind else "",
                       when if when is not None else ctx.clock(ctx.t if t is None else t))


def differ_words(ctx: Context) -> str:
    if ctx.differ:
        text = "these runs differ in: " + ", ".join(ctx.differ)
    else:
        text = "no setting that changes the storm differs between these runs"
    if ctx.unread:
        text += f" (settings not read for {', '.join(ctx.unread)})"
    return text


def fit_text(ov: S.Overlay, text: str, role: str, width: float, *, keep_end: int = 0) -> str:
    """``text`` cut with an ellipsis to fit ``width`` pixels; ``keep_end`` characters of its end are kept (a run's
    name often differs from another's only at its end)."""

    if ov.measure(text, role)[0] <= width:
        return text
    end = text[-keep_end:] if keep_end and len(text) > 2 * keep_end else ""
    head = text[:len(text) - len(end)]
    while len(head) > 1 and ov.measure(head + "…" + end, role)[0] > width:
        head = head[:-1]
    return head.rstrip() + "…" + end


def chip(ov: S.Overlay, x: float, y: float, colour: str, size: int = 12) -> None:
    """A small square of a run's colour, its top-left at (x, y)."""

    from PIL import Image

    ov.draw_rgba(Image.new("RGBA", (size, size), S.rgb8(colour, 1.0)), (int(x), int(y)))


def run_label(ov: S.Overlay, r: Any, x: float, y: float, width: float, role: str = "label") -> None:
    """'A  name' with the run's colour chip, its baseline at ``y``."""

    size = max(8, ov.px(role) * 2 // 3)
    chip(ov, x, y - size - 1, r.colour, size)
    ov.text((x + size + 8, y), fit_text(ov, f"{r.letter}  {r.name}", role, width - size - 8, keep_end=6), role,
            colour="text", anchor="ls")


def wrapped(ov: S.Overlay, x: float, y: float, text: str, width: float, *, role: str = "micro",
            colour: str | None = None, vertical: bool = False, max_lines: int = 3) -> float:
    """Draw ``text`` wrapped to ``width``; returns the next baseline."""

    step = int(ov.px(role) * 1.45)
    lines = ov.wrap(text, role, int(width))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = fit_text(ov, lines[-1] + " …", role, width)
    if vertical:
        ov.vertical.append(text)
    for line in lines:
        ov.text((x, y), line, role, colour=colour, anchor="ls", halo=False, vertical=vertical, record=False)
        y += step
    ov.texts.append(text)
    return y


# ----------------------------------------------------------------------------------------------- one panel
def panel_box(x: int, y: int, size: int, cx: float, cy: float, span: float) -> P.Box:
    return P.Box(x, y, size, cx - 0.5 * span, cy - 0.5 * span, span, h=size)


def outside(box: P.Box, extent: tuple[float, float]) -> np.ndarray:
    xm, ym = box.axes_m()
    out_x = (xm < 0.0) | (xm > extent[0])
    out_y = (ym < 0.0) | (ym > extent[1])
    return out_y[:, None] | out_x[None, :]


def hatch(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """What lies outside a run's area: black with thin diagonal lines."""

    if not mask.any():
        return img
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    lines = ((xx + yy) % 11) < 1.2
    out = img.copy()
    out[mask] = S.rgb8("void")
    out[mask & lines] = HATCH
    return out


def in_box(plane: np.ndarray, box: P.Box, dx: float, dy: float) -> np.ndarray:
    """The run's own grid cells whose centres lie inside ``box`` (the numbers a panel states are the model's
    values, not the picture's samples)."""

    ny, nx = plane.shape
    i0 = max(0, int(math.ceil(box.mx0 / dx - 0.5)))
    i1 = min(nx, int(math.floor((box.mx0 + box.span) / dx - 0.5)) + 1)
    j0 = max(0, int(math.ceil(box.my0 / dy - 0.5)))
    j1 = min(ny, int(math.floor((box.my0 + box.span_y) / dy - 0.5)) + 1)
    if i1 <= i0 or j1 <= j0:
        return np.zeros(0, np.float32)
    return np.asarray(plane[j0:j1, i0:i1]).ravel()


def draw_panel(state: Any, look: str, box: P.Box) -> tuple[np.ndarray, str]:
    """One run's panel (uint8 RGB of ``box``'s size) and the value it shows in words."""

    r = state.run
    mask = outside(box, r.extent)
    if look in state.missing:
        img = np.tile(np.array(S.rgb8("void"), np.uint8), (box.h, box.w, 1))
        return img, "not drawn"
    if state.source == "pictures":
        return _crop(state, look, box, mask)
    f = state.planes
    if look == "reflectivity":
        d = P.sample(f["dbz"], box, r.dx, r.dy, cval=-99.0)
        img = P.dbz_image(d)
        inside = in_box(f["dbz"], box, r.dx, r.dy)
        top = float(inside.max()) if inside.size else -99.0
        words = f"max {top:.0f} dBZ in the box" if top >= 5.0 else "nothing above 5 dBZ in the box"
    elif look == "clouds":
        from . import satellite as SA

        fields = {name: f[f"cloud_{name}"] for name in ("reflect", "lit", "ground_sun")}
        img = SA.image(fields, box, r.dx, r.dy)
        words = "cloud seen from straight above"
    elif look == "surface":
        look_ = P.SURFACE_LOOK
        ths = P.sample(f["theta0"], box, r.dx, r.dy, cval=0.0)
        cs = P.sample(f["conv0"], box, r.dx, r.dy, cval=0.0)
        us = P.sample(f["u0"], box, r.dx, r.dy, cval=0.0)
        vs = P.sample(f["v0"], box, r.dx, r.dy, cval=0.0)
        hdr = S.colours("theta_sym", ths, -P.THETA_RANGE, P.THETA_RANGE) * look_["theta_gain"]
        tx, sp = P.streaks(us, vs, box, px=look_["lic_px"])
        sm = np.clip(sp / look_["speed_ref"], 0.25, 1.6)
        hdr += (np.clip(0.55 + 0.45 * tx, 0.0, 2.5) * sm * look_["lic_gain"])[..., None] * \
            np.array(S.hex_rgb("silk"), np.float32)
        e = np.clip((cs - look_["conv_floor"]) / look_["conv_ref"], 0.0, look_["conv_clip"]) ** look_["conv_gamma"]
        hdr += (e * look_["conv_gain"])[..., None] * np.array(S.hex_rgb("#ffe6bf"), np.float32)
        img = S.finish(hdr)
        th_in = in_box(f["theta0"], box, r.dx, r.dy)
        sp_in = in_box(np.hypot(f["u0"], f["v0"]), box, r.dx, r.dy)
        words = (f"coldest θ{S.PRIME} {S.fmt_num(float(th_in.min()), 1)} K, fastest wind "
                 f"{float(sp_in.max()):.0f} m/s ground-relative, in the box") if th_in.size else ""
    elif look == "wind":
        us = P.sample(f["u0"], box, r.dx, r.dy, cval=0.0)
        vs = P.sample(f["v0"], box, r.dx, r.dy, cval=0.0)
        speed = np.hypot(us, vs)
        base = S.colours("hurricane_wind", speed, 0.0, P.HURRICANE_WIND_RANGE)
        tx, _ = P.streaks(us, vs, box, seed=5)
        shade = 1.0 + 0.22 * np.clip(tx, -1.5, 1.5)[..., None] * np.clip(speed / 8.0, 0.25, 1.0)[..., None]
        img = S.to8(np.clip(base * shade + 0.02 * np.clip(tx, 0.0, 1.5)[..., None], 0.0, 1.0))
        img = P.blend(img, P.line_alpha(speed, P.HURRICANE_FORCE, width=1.6), "title", 0.9)
        sp_in = in_box(np.hypot(f["u0"], f["v0"]), box, r.dx, r.dy)
        words = f"fastest {float(sp_in.max()):.0f} m/s ground-relative in the box" if sp_in.size else ""
    else:  # pragma: no cover - the looks are fixed
        raise ValueError(look)
    return hatch(img, mask), words


def _crop(state: Any, look: str, box: P.Box, mask: np.ndarray) -> tuple[np.ndarray, str]:
    """The panel cut from the run's own picture of this look (its frames were deleted), sampled at the nearest
    picture pixel so its colours stay the picture's own."""

    rgb, src = state.pictures[look]
    xm, ym = box.axes_m()
    cols = np.clip(np.floor((xm - src.mx0) / src.mpp).astype(int) + src.x, src.x, src.x + src.w - 1)
    rows = np.clip(src.y + np.floor((src.my0 + src.span_y - ym) / src.mpp).astype(int), src.y, src.y + src.h - 1)
    img = rgb[rows[:, None], cols[None, :], :3].astype(np.uint8)
    v = state.values
    if look == "reflectivity" and "dbz_max_1km" in v:
        words = (f"max {v['dbz_max_1km']:.0f} dBZ in its area" if v["dbz_max_1km"] >= 5.0
                 else "nothing above 5 dBZ in its area")
    elif look == "surface" and "theta_min" in v:
        words = (f"coldest θ{S.PRIME} {S.fmt_num(v['theta_min'], 1)} K, fastest wind "
                 f"{v.get('wind_max_ground_relative', 0.0):.0f} m/s ground-relative, in its area")
    elif look == "wind" and "wind_max_ground_relative" in v:
        words = f"fastest {v['wind_max_ground_relative']:.0f} m/s ground-relative in its area"
    else:
        words = "from its own picture"
    return hatch(img, mask), words


def ring_radius(size: int) -> float:
    return max(8.0, 0.03 * size)


def centre_ring(ov: S.Overlay, xy: tuple[float, float], radius: float) -> None:
    """The gold ring on a box's centre, over a dark ring so it shows on bright reflectivity and a hurricane's
    eyewall alike."""

    ov.ring(xy, radius, width=4.0, colour="void", alpha=0.85)
    ov.ring(xy, radius)


def backed_scale_bar(ov: S.Overlay, box: P.Box) -> None:
    """The quick-looks' scale bar in a box's top-left corner, on a dark backing so its label reads over the storm."""

    from PIL import Image

    length = P.nice_step(box.span, 6) / box.mpp
    label = S.fmt_km(P.nice_step(box.span, 6))
    x, y = box.x + ov.U(8), box.y + ov.U(14)
    w = int(length + ov.U(10) + ov.measure(label, "micro")[0] + ov.U(10))
    h = ov.U(22)
    patch = Image.new("RGBA", (max(1, min(w, box.w - ov.U(4))), h), S.rgb8("void", 0.72))
    ov.draw_rgba(patch, (box.x + ov.U(2), y - h // 2))
    P.scale_bar(ov, x, y, box)


# ----------------------------------------------------------------------------------------------- side by side
def storm_picture(ctx: Context, look: str, out: Path) -> dict[str, Any]:
    """The storm in every run at one storm time, side by side, each box centred on its own storm."""

    states = ctx.states or []
    if all(look in s.missing for s in states):
        raise NotDrawn("; ".join(f"{s.run.letter}: {s.missing[look]}" for s in states))
    n = len(states)
    gap = 36 if n <= 3 else 24
    ov = S.Overlay((W, H))
    times_differ = any(abs(s.t - ctx.t) > 0.5 for s in states)
    step = int(ov.px("micro") * 1.45)
    legend_h = ov.U(14) + ov.U(52) + (ov.U(20) if times_differ else 0) + ov.U(8)
    size = int(min((X1 - X0 - (n - 1) * gap) / n, PANEL_MAX))
    caption_h = 0
    for _ in range(3):
        # the captions under each panel wrap to its width; the panels shrink until panels, captions and the legend
        # fit above the footer
        lines = max(2 + sum(len(ov.wrap(text, "micro", size)) for text, _c, _v in _captions(ctx, s, look))
                    for s in states)
        caption_h = ov.U(20) + lines * step
        size = int(min(size, CONTENT_BOTTOM - PANEL_TOP - caption_h - legend_h))
    x0 = (W - (n * size + (n - 1) * gap)) // 2
    block = ov.U(32) + size + caption_h + legend_h
    top = int(max(PANEL_TOP, (PANEL_TOP - ov.U(40) + CONTENT_BOTTOM - block) / 2 + ov.U(32)))
    cv = P.canvas()
    boxes = []
    shown: list[dict[str, Any]] = []
    for k, state in enumerate(states):
        box = panel_box(x0 + k * (size + gap), top, size, state.centre.x, state.centre.y, ctx.span)
        img, words = draw_panel(state, look, box)
        P.paste(cv, img, box)
        cv = P.outline(cv, box)
        boxes.append(box)
        shown.append({"letter": state.run.letter, "time_s": state.t, "value": words, "source": state.source,
                      "centre_m": [state.centre.x, state.centre.y], "centre": state.centre.kind,
                      "not_drawn": state.missing.get(look)})
    when = (f"each at its save nearest {ctx.clock(ctx.t)}" if times_differ else f"all at {ctx.clock(ctx.t)}")
    ov.title(TITLES[look], S.data_line(f"{n} runs", f"boxes {S.fmt_km(ctx.span)} across, each on its own storm",
                                       when))
    ov.clock(ctx.clock(ctx.t))
    notes = _storm_notes(ctx, look, states)
    level = ("1 km" if look == "reflectivity" else "the whole column" if look == "clouds" else "lowest level of each run")
    wind = "ground-relative" if look in ("surface", "wind") else ""
    ov.footer([footer_facts(ctx, level=level, wind=wind), notes])
    ov.mark()
    bottom = top + size
    for box, state, row in zip(boxes, states, shown):
        r = state.run
        run_label(ov, r, box.x, box.y - ov.U(10), box.w)
        y = box.y + box.h + ov.U(20)
        if look in state.missing:
            ov.text((box.x + box.w / 2, box.y + box.h / 2), "not drawn", "label", colour="muted", anchor="ms")
            y = wrapped(ov, box.x, y, f"{r.letter}: {state.missing[look]}", box.w, colour="data", max_lines=4)
        else:
            y = wrapped(ov, box.x, y, S.data_line(row["value"], ctx.clock(state.t)), box.w, colour="data",
                        max_lines=2)
            centre_ring(ov, box.to_px(state.centre.x, state.centre.y), ring_radius(box.w))
        for text, colour, vertical in _captions(ctx, state, look):
            y = wrapped(ov, box.x, y, text, box.w, colour=colour, vertical=vertical, max_lines=4)
        bottom = max(bottom, y)
    y = bottom + ov.U(14)
    left, right = boxes[0].x, boxes[-1].x + boxes[-1].w
    P.scale_bar(ov, left, y + ov.U(6), boxes[0])
    if look == "reflectivity":
        P.dbz_legend(ov, right - ov.U(300), y, ov.U(300))
    elif look == "surface":
        ov.ramp_bar("theta_sym", (right - ov.U(400), y + ov.U(4), right, y + ov.U(4) + max(3, ov.U(7))),
                    vmin=-P.THETA_RANGE, vmax=P.THETA_RANGE,
                    ticks=[(-8, S.fmt_num(-8)), (-4, S.fmt_num(-4)), (0, "0"), (4, "+4"), (8, "+8 K")],
                    title=f"θ{S.PRIME} at each run's lowest level")
    elif look == "wind":
        fastest = P.HURRICANE_WIND_RANGE
        ov.ramp_bar("hurricane_wind", (right - ov.U(400), y + ov.U(4), right, y + ov.U(4) + max(3, ov.U(8))),
                    vmin=0.0, vmax=fastest, ticks=[(0.0, "0"), (20.0, "20"), (P.HURRICANE_FORCE, "33"),
                                                   (50.0, "50"), (fastest, f"{fastest:g} m/s")],
                    title="wind at the lowest level, ground-relative")
    ov.text((left, y + ov.U(46)), differ_words(ctx), "micro", colour="text", anchor="ls", halo=False)
    if times_differ:
        ov.text((left, y + ov.U(66)), "saved times drawn: " + ", ".join(f"{s.run.letter} {ctx.clock(s.t)}" for s in states)
                + f" (each run's save nearest {ctx.clock(ctx.t)})", "micro", colour="data", anchor="ls", halo=False)
    P.save(ov, cv, out)
    return {"look": look, "panel_px": size, "box_m": ctx.span, "runs": shown}


def _captions(ctx: Context, state: Any, look: str) -> list[tuple[str, str | None, bool]]:
    """The lines under a run's panel after its value: (text, colour, whether its speeds are vertical)."""

    r = state.run
    rows: list[tuple[str, str | None, bool]] = []
    if len({o.case_words() for o in ctx.runs}) > 1 and r.case_words():
        rows.append((r.case_words(), None, False))
    if len({o.start_words() for o in ctx.runs}) > 1 and r.start_words():
        rows.append((f"start: {r.start_words()}", None, False))
    rows += [
        (r.grid_words(), None, False),
        (S.data_line(f"lowest level {S.fmt_height(r.z0)}", r.interval_words()), None, False),
        (state.centre.words, None, True)]
    if check_words(ctx, r):
        rows.append((check_words(ctx, r), None, False))
    if state.source == "pictures" and look not in state.missing:
        rows.append(("cut from its own picture: its saved frames were deleted", "warm", False))
    return rows


def _storm_notes(ctx: Context, look: str, states: Sequence[Any]) -> str:
    tail = ["hatched: outside the run's area", "gold ring: the box centre"]
    if any(s.source == "pictures" for s in states):
        tail.append("+ in a cut panel: fixed to the ground")
    if look == "reflectivity":
        dense = sorted({w for s in states for w in [_dense(s)] if w})
        return S.data_line("1 km: linear in Z between levels",
                           f"from the model's rain, snow and {' or '.join(dense) or 'graupel'}, not a radar", *tail)
    if look == "clouds":
        return S.data_line("seen from straight above, lit by a low sun", "not every run has reflectivity",
                           "not satellite data", *tail)
    if look == "surface":
        return S.data_line(f"θ{S.PRIME} = θ minus the base state, ±{P.THETA_RANGE:g} K fixed",
                           "streaks: the lowest level's wind, ground-relative", "bright lines: convergence", *tail)
    return S.data_line(f"0–{P.HURRICANE_WIND_RANGE:g} m/s fixed", "streaks: where the wind blows",
                       f"white line: {P.HURRICANE_FORCE:g} m/s ground-relative", *tail)


def _dense(state: Any) -> str:
    scheme = state.run.facts.scheme
    return "hail" if "hail" in scheme else "graupel" if "graupel" in scheme or "Morrison" in scheme else ""


# ----------------------------------------------------------------------------------------------- over time
def chart_rows(ctx: Context) -> tuple[tuple[str, str, str], ...]:
    from .compare import HURRICANE_CHARTS, STORM_CHARTS

    return HURRICANE_CHARTS if ctx.hurricane else STORM_CHARTS


def series_picture(ctx: Context, out: Path) -> dict[str, Any]:
    """Every run's time series on one set of charts (the one-run time series' charts, :mod:`.series`)."""

    import os as _os

    series = ctx.series or []
    charts = chart_rows(ctx)
    if not any(pts for pts, _src in series):
        raise NotDrawn("No run has a time series: none has saved frames or a series.json.")
    # the time axis runs to the last saved time of any run (the storm time is never later than one save interval
    # past it: compare.storm_time refuses a later one), so no time asked for squashes the lines
    latest = max([p[0] for pts, _ in series for v in pts.values() for p in v] + [0.0])
    span_s = max(latest, 60.0)
    span_s = max(span_s, ctx.t if ctx.t <= 1.5 * span_s else 0.0)
    hours = span_s >= 172800.0
    per = 3600.0 if hours else 60.0
    tmax = span_s / per
    fig = S.figure((W, H))
    left, right, top, bottom = 110 / W, 1 - 80 / W, 1 - 300 / H, 175 / H
    cols, rows_n = 3, 2
    gw, gh = 90 / W, 110 / H
    pw = (right - left - (cols - 1) * gw) / cols
    ph = (top - bottom - (rows_n - 1) * gh) / rows_n
    fm = S.mpl_fontprops("micro", fig)
    fp = S.mpl_fontprops("label", fig)
    drawn: dict[str, list[str]] = {}
    for n, (key, label, unit) in enumerate(charts):
        r, c = divmod(n, cols)
        ax = fig.add_axes((left + c * (pw + gw), top - (r + 1) * ph - r * gh, pw, ph))
        ax.set_facecolor(S.PALETTE["void"])
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(S.PALETTE["rule"])
        ax.grid(True, axis="y", color=S.PALETTE["rule"], linewidth=1.0)
        ax.tick_params(colors=S.PALETTE["muted"], labelsize=fm.get_size())
        ends = []
        have = []
        for run, (pts, _src) in zip(ctx.runs, series):
            rows = pts.get(key) or []
            if not rows:
                continue
            have.append(run.letter)
            x = [p[0] / per for p in rows]
            y = [p[1] for p in rows]
            ax.plot(x, y, color=run.colour, linewidth=2.0, zorder=3, solid_capstyle="round")
            ax.plot([x[-1]], [y[-1]], "o", color=run.colour, markersize=5, markeredgecolor=S.PALETTE["void"],
                    markeredgewidth=1.5, zorder=4)
            ends.append((x[-1], y[-1], run.letter))
        drawn[key] = have
        ax.axvline(ctx.t / per, color=S.PALETTE["gold"], linewidth=1.0, alpha=0.6)
        ax.set_xlim(0, tmax * 1.06)
        if key != "prs_sfc_min_hpa":
            lo, hi = ax.get_ylim()
            if key in ("thpert_sfc_min", "wmin"):
                ax.set_ylim(lo, max(hi, 0.0))
            elif key == "dbz_max":
                ax.set_ylim(0.0, max(hi, 10.0))     # below 0 dBZ is the floor of a column with no echo
            else:
                ax.set_ylim(min(lo, 0.0), hi)
        _end_labels(ax, ends, fm)
        ax.set_title(f"{label} ({unit})", loc="left", fontproperties=fp, color=S.PALETTE["text"], pad=8)
        ax.set_xlabel("hours since the start" if hours else "minutes since the start", fontproperties=fm,
                      color=S.PALETTE["muted"])
        missing = [run for run in ctx.runs if run.letter not in have]
        reasons: dict[str, list[str]] = {}
        for run in missing:
            why = ("saved frames were deleted" if not run.has_frames else "frames hold no pressure"
                   if key == "dp_sfc_max" else "records do not have it")
            reasons.setdefault(why, []).append(run.letter)
        for m, (why, letters) in enumerate(reasons.items()):
            whose = "their" if len(letters) > 1 else "its"
            ax.text(0.99, 0.02 + 0.08 * m, f"{', '.join(letters)}: {whose} {why}", transform=ax.transAxes,
                    ha="right", va="bottom", fontproperties=fm, color=S.PALETTE["muted"])
    base = S.fig_to_rgb(fig)
    ov = S.Overlay((W, H))
    ov.title("Runs over time", S.data_line(f"{len(ctx.runs)} runs", "domain extremes at every saved frame",
                                           f"gold line: {ctx.clock(ctx.t)}, the side-by-side pictures' time"))
    # the clock is the comparison's storm time, as on its other pictures; the footer says how far the charts run
    ov.clock(ctx.clock(ctx.t))
    _legend(ov, ctx, ov.U(176))
    z0 = _grouped(ctx.runs, lambda r: S.fmt_height(r.z0))
    notes = ["lowest level " + z0]
    if not ctx.hurricane:
        notes.append("pressure drop: below the base state at the lowest level, from the saved frames")
        if any(p[1] < 0.0 for pts, _ in series for p in pts.get("dp_sfc_max") or []):
            notes.append("a drop under 0: above the base state everywhere")
    notes.append("wind ground-relative")
    ov.footer([footer_facts(ctx, level="", wind="", when=f"charts to {ctx.clock(latest)}"), S.data_line(*notes)])
    ov.mark()
    problems = ov.check()
    if problems:
        raise P.LintError("; ".join(problems))
    from PIL import Image

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.partial")
    img = Image.fromarray(ov.composite(np.asarray(base)))
    img.save(tmp, format="PNG", compress_level=6)
    _os.replace(tmp, out)
    thumb = out.parent / "thumb" / out.name
    thumb.parent.mkdir(exist_ok=True)
    img.resize(P.THUMB, Image.LANCZOS).save(thumb, format="PNG", compress_level=6)
    return {"charts": [key for key, _l, _u in charts], "runs_per_chart": drawn,
            "sources": {run.letter: src for run, (_pts, src) in zip(ctx.runs, series)},
            "time_unit": "h" if hours else "min", "time_s": ctx.t, "charts_to_s": latest}


def _end_labels(ax: Any, ends: list[tuple[float, float, str]], fm: Any) -> None:
    """Each line's letter just right of its last point, pushed apart so no two overlap."""

    if not ends:
        return
    trans = ax.transData
    points = sorted(((trans.transform((x, y))[1], x, y, letter) for x, y, letter in ends), key=lambda p: p[0])
    placed: list[float] = []
    gap = fm.get_size() * ax.figure.dpi / 72.0 * 1.15
    for py, x, y, letter in points:
        target = py if not placed or py - placed[-1] >= gap else placed[-1] + gap
        placed.append(target)
        ax.annotate(letter, xy=(x, y), xytext=(7, target - py), textcoords="offset pixels", va="center",
                    ha="left", fontproperties=fm, color=S.PALETTE["text"], annotation_clip=False)


def _legend(ov: S.Overlay, ctx: Context, y: int) -> None:
    runs = ctx.runs
    cols = 2 if len(runs) <= 4 else 3
    width = (X1 - X0) / cols
    for n, r in enumerate(runs):
        row, col = divmod(n, cols)
        x = X0 + col * width
        yy = y + row * ov.U(24)
        ov.rule(x, yy - ov.U(6), ov.U(30), thickness=3, colour=r.colour)
        # the run's name, and only what tells it from the others (the footer says what they share)
        parts = [f"{r.letter}  {r.name}"]
        parts.append(f"{S.fmt_grid(r.dx)} grid")
        if len({round(o.z0, 1) for o in runs}) > 1:
            parts.append(f"lowest level {S.fmt_height(r.z0)}")
        if len({o.interval for o in runs}) > 1:
            parts.append(r.interval_words())
        parts.append(check_words(ctx, r))
        text = S.data_line(*parts)
        ov.text((x + ov.U(40), yy), fit_text(ov, text, "micro", width - ov.U(56)), "micro", colour="data",
                anchor="ls")


# ----------------------------------------------------------------------------------------------- soundings
def card_groups(n: int) -> list[list[int]]:
    """Up to three runs to a card: 2, 3, 2+2, 3+2, 3+3."""

    if n <= 3:
        return [list(range(n))]
    first = 2 if n == 4 else 3
    return [list(range(first)), list(range(first, n))]


HODO_COLOURS = ((0.0, 1000.0, "#ff5f57"), (1000.0, 3000.0, "#ffb347"), (3000.0, 6000.0, "#34d399"),
                (6000.0, 9000.0, "#4cc3ff"), (9000.0, 12000.0, "#b9a8ff"))


def _num(env: Mapping[str, Any] | None, *path: str) -> float | None:
    value: Any = env
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


#: The card's bars: (label, the record's path, unit, format).
BARS = (("CAPE, mixed layer", ("mixed_layer", "cape_j_per_kg"), "J/kg", "{:,.0f}"),
        ("helicity 0–1 km, storm-relative", ("srh_0_1km_m2_per_s2",), "m² s⁻²", "{:,.0f}"),
        ("helicity 0–3 km, storm-relative", ("srh_0_3km_m2_per_s2",), "m² s⁻²", "{:,.0f}"),
        ("wind shear 0–6 km", ("shear_0_6km_m_per_s",), "m/s", "{:.0f}"))


def hodograph_limit(ctx: Context) -> float:
    speed = 10.0
    for (prof, _w), (env, _s) in zip(ctx.profiles or [], ctx.environments or []):
        if prof is not None:
            keep = prof["z"] <= 12000.0
            if keep.any():
                speed = max(speed, float(np.nanmax(np.hypot(prof["u"][keep], prof["v"][keep]))))
        ur, vr = _num(env, "bunkers_right_u_m_per_s"), _num(env, "bunkers_right_v_m_per_s")
        if ur is not None and vr is not None:
            speed = max(speed, math.hypot(ur, vr))
    return 10.0 * math.ceil((speed + 5.0) / 10.0)


def _hodograph(fig: Any, rect: tuple[float, float, float, float], prof: Mapping[str, np.ndarray] | None,
               env: Mapping[str, Any] | None, lim: float, fs: float, *, mover: bool = True) -> None:
    """The environment look's hodograph (:mod:`.environment`), on one scale for every run of a card.  ``mover``:
    mark the Bunkers right mover (a thunderstorm's; a hurricane's column has none worth marking)."""

    from matplotlib.patches import Circle

    ax = fig.add_axes(rect)
    ax.set_facecolor("#060a12")
    for r in range(10, int(lim) + 1, 10):
        ax.add_patch(Circle((0, 0), r, fill=False, color="#1d2740", lw=0.9, zorder=1))
        ax.text(r * 0.707 + 0.6, -r * 0.707 - 0.6, f"{r}", color=S.PALETTE["dim"], fontsize=fs * 0.85,
                ha="left", va="top", zorder=2)
    ax.axhline(0, color="#1d2740", lw=0.9, zorder=1)
    ax.axvline(0, color="#1d2740", lw=0.9, zorder=1)
    if prof is not None:
        z, u, v = prof["z"], prof["u"], prof["v"]
        for z0, z1, colour in HODO_COLOURS:
            sel = (z >= z0) & (z <= z1)
            if sel.sum() >= 1:
                idx = np.where(sel)[0]
                idx = np.append(idx, min(idx[-1] + 1, len(z) - 1))
                ax.plot(u[idx], v[idx], color=colour, lw=2.4, solid_capstyle="round", zorder=3)
        marked: list[tuple[float, float]] = []
        for hk in (1, 3, 6):
            if z[0] <= hk * 1000.0 <= z[-1]:
                uu, vv = float(np.interp(hk * 1000.0, z, u)), float(np.interp(hk * 1000.0, z, v))
                if any(math.hypot(uu - a, vv - b) < 0.22 * lim for a, b in marked):
                    continue        # a calm column: the heights would sit on one another
                marked.append((uu, vv))
                ax.plot([uu], [vv], "o", color="#cfd8e6", ms=3.5, zorder=4)
                ax.text(uu + 1.0, vv + 1.0, f"{hk} km", color=S.PALETTE["muted"], fontsize=fs * 0.85, zorder=5)
    ur, vr = _num(env, "bunkers_right_u_m_per_s"), _num(env, "bunkers_right_v_m_per_s")
    if mover and ur is not None and vr is not None:
        ax.plot([ur], [vr], marker="+", color=S.PALETTE["gold"], ms=12, mew=2.2, zorder=6)
        ax.text(ur + 1.2, vr - 1.2, "right mover", color=S.PALETTE["gold"], fontsize=fs * 0.9, va="top", zorder=6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#1f2a40")


#: The least room one of the card's bars needs: its label and value line, then the bar under it.
BAR_STEP_MIN = 24


def _readout_width(ov: S.Overlay, cap: str, val: str) -> int:
    """How wide :meth:`.style.Overlay.readout` draws one caption and its value."""

    return max(ov.measure(S.upper(cap), "micro", tracking=0.08)[0],
               ov.measure(val, "hero_number", px=int(ov.px("data") * 1.5))[0])


def left_lines(ov: S.Overlay, ctx: Context, r: Any, row_h: int, width: float) -> list[tuple[str, str | None, int]]:
    """The words under a run's name on the soundings card, as (text, colour, lines): its case (or grid), its start
    and, when the runs' checks differ, its CM1 check.  They get the room the four bars leave, never more: a case
    wraps to two lines and a check to two only while the bars still fit, and otherwise each is cut with an
    ellipsis to one line (the start goes last)."""

    line = int(ov.px("micro") * 1.45)
    room = max(1, int((row_h - ov.U(44) - len(BARS) * ov.U(BAR_STEP_MIN)) // line))
    rows: list[list[Any]] = [[r.case_words() or r.grid_words(), "data", 2]]
    if r.start_words():
        rows.append([f"start: {r.start_words()}", None, 1])
    if check_words(ctx, r):
        rows.append([check_words(ctx, r), None, 2])
    for row in rows:
        row[2] = max(1, min(row[2], len(ov.wrap(row[0], "micro", int(width)))))
    while sum(row[2] for row in rows) > room:
        two = [row for row in rows if row[2] > 1]
        if two:
            two[0][2] -= 1          # the case first, then the check
        elif len(rows) > 1:
            rows.pop(1 if len(rows) == 3 else -1)    # the start, then the check
        else:
            break
    return [(text, colour, lines) for text, colour, lines in rows]


def _value_at(points: Mapping[str, list], key: str, t: float) -> float | None:
    rows = points.get(key) or []
    if not rows:
        return None
    row = min(rows, key=lambda p: (abs(p[0] - t), p[0]))
    return float(row[1]) if abs(row[0] - t) <= 0.5 else None


def _small_skewt_labels(ax: Any, sk: Any) -> None:
    """The environment look's skew-T labels, placed for a panel this small: the 1000 hPa label is dropped (it would
    sit on the temperatures), and the temperatures move from inside the panel's foot, where the profile starts,
    to just under it."""

    from matplotlib.transforms import offset_copy

    for text in list(ax.texts):
        words = text.get_text()
        if words == "1000":
            text.remove()
        elif words.endswith("°"):
            text.set_y(sk.y_bot)
            text.set_verticalalignment("top")
            text.set_transform(offset_copy(ax.transData, fig=ax.figure, x=0.0, y=-2.0, units="dots"))
            text.set_clip_on(False)


def soundings_card(ctx: Context, group: Sequence[int], out: Path, *, number: int = 1, of: int = 1) -> dict[str, Any]:
    """Each run's sounding beside what it grew, up to three runs to a card, on one set of scales."""

    from .skewt import BASE_T, EDGE, SkewT, TD_COLOUR, T_COLOUR

    runs = [ctx.runs[n] for n in group]
    states = [ctx.states[n] for n in group] if ctx.states else [None] * len(group)
    profiles = [ctx.profiles[n] for n in group]
    envs = [ctx.environments[n] for n in group]
    series = [ctx.series[n] for n in group]
    k = len(runs)
    # the bottom row's skew-T temperatures sit just under its panel, clear of the keys at the foot
    top_y, bottom_y, row_gap = 196, 916, 18
    row_h = int((bottom_y - top_y - (k - 1) * row_gap) / max(k, 2 if k == 1 else k))
    sk_x = 420
    sk_w = min(row_h, 330)
    ho_x = sk_x + sk_w + 22
    ho_w = min(row_h, 300)
    divider = ho_x + ho_w + 34
    st_x = divider + 34
    st_w = min(row_h, 330)
    nu_x = st_x + st_w + 30
    lim = hodograph_limit(ctx)
    # bars on one scale for every run of the comparison (every card), so cards compare
    bar_top = []
    for _label, path, _unit, _fmt in BARS:
        values = [_num(env, *path) for env, _src in (ctx.environments or [])]
        bar_top.append(max([abs(v) for v in values if v is not None] + [1.0]))
    fig = S.figure((W, H))
    fs = 8.5
    rows_y = [top_y + n * (row_h + row_gap) for n in range(k)]
    for n, (prof_row, env_row) in enumerate(zip(profiles, envs)):
        prof, _words = prof_row
        env, _src = env_row
        y = rows_y[n]
        ax = fig.add_axes((sk_x / W, 1 - (y + row_h) / H, sk_w / W, row_h / H))
        ax.set_facecolor("#060a12")
        if prof is None:
            # no profile: an empty panel, so the words that say why sit on no chart lines
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(EDGE)
        else:
            sk = SkewT(ax)
            sk.background(fs)
            _small_skewt_labels(ax, sk)
            p_hpa = prof["p"] / 100.0
            keep = p_hpa >= 100.0
            for values, colour in ((prof["t"] - 273.15, T_COLOUR), (prof["td"] - 273.15, TD_COLOUR)):
                ax.plot(sk.x(values[keep], p_hpa[keep]), sk.y(p_hpa[keep]), color=colour, lw=2.4, zorder=4,
                        solid_capstyle="round", solid_joinstyle="round")
            lcl = _num((env or {}).get("mixed_layer") if env else None, "lcl_m_agl")
            if lcl is not None and prof["z"][0] <= lcl <= prof["z"][-1]:
                pl = float(np.interp(lcl, prof["z"], p_hpa))
                ax.plot([sk.x(-40, pl), sk.x(50, pl)], [sk.y(pl), sk.y(pl)], color=BASE_T, lw=1.0, ls=(0, (2, 3)),
                        zorder=3)
        _hodograph(fig, (ho_x / W, 1 - (y + ho_w) / H, ho_w / W, ho_w / H), prof, env, lim, fs,
                   mover=not ctx.hurricane)
    base = S.fig_to_rgb(fig)
    look = ctx.first_look
    boxes: list = []
    for n, state in enumerate(states):
        y = rows_y[n]
        if state is None:
            boxes.append(None)
            continue
        box = panel_box(st_x, y, st_w, state.centre.x, state.centre.y, ctx.span)
        img, words = draw_panel(state, look, box)
        base[box.y:box.y + box.h, box.x:box.x + box.w] = img
        base = P.outline(base, box)
        boxes.append((box, words, look))
    ov = S.Overlay((W, H))
    title = "Soundings and what they grew"
    part = f"card {number} of {of}" if of > 1 else ""
    ov.title(title, S.data_line(f"{len(ctx.runs)} runs", "each row: the air the run started in, then its storm",
                                part))
    ov.clock(ctx.clock(ctx.t))
    head_y = ov.U(184)
    ov.text((X0, head_y), "THE SOUNDING", "label", colour="text", anchor="ls")
    ov.text((X0 + ov.measure("THE SOUNDING", "label")[0] + ov.U(14), head_y), "input: what the run started in",
            "micro", colour="muted", anchor="ls")
    ov.text((st_x, head_y), "WHAT IT GREW", "label", colour="text", anchor="ls")
    head_x = st_x + ov.measure("WHAT IT GREW", "label")[0] + ov.U(14)
    head = f"output: its save nearest {ctx.clock(ctx.t)}, a {S.fmt_km(ctx.span)} box on its storm"
    if head_x + ov.measure(head + "; numbers: its whole area", "micro")[0] <= X1:
        head += "; numbers: its whole area"     # every row says it too, beside its numbers
    ov.text((head_x, head_y), head, "micro", colour="muted", anchor="ls")
    ov.rule(divider, top_y, bottom_y - top_y, thickness=1, colour="rule", vertical=True)
    facts_out = []
    left_w = sk_x - X0 - ov.U(24)
    # the numbers beside each storm, in two columns at one place on every row: the second as far right as its
    # widest number lets it stay inside the picture's margin (a wind with its frame is wide), never on the first
    grown = [(_grew(ctx, series[n][0] if series else {}, state, boxes[n][1], boxes[n][2])
              if state is not None and boxes[n] is not None else []) for n, state in enumerate(states)]
    wcol = W - X0 - nu_x
    cols = 2 if wcol >= 480 else 1
    cw = wcol / cols
    firsts: list[int] = []
    seconds: list[int] = []
    for numbers in grown:
        widths = [_readout_width(ov, cap, val) for cap, val, _v in numbers]
        per = math.ceil(len(numbers) / cols) if numbers else 0
        firsts += widths[:per]
        seconds += widths[per:]
    if cols == 1 and firsts:
        # one column (two runs to the card): nearer the storm box when its widest number needs the room
        nu_x = int(max(st_x + st_w + ov.U(16), min(nu_x, X1 - max(firsts))))
        wcol = W - X0 - nu_x
        cw = wcol
    col_x = [nu_x + c * cw for c in range(cols)]
    if cols == 2 and firsts and seconds:
        col_x[1] = max(nu_x + max(firsts) + ov.U(28), min(nu_x + cw, X1 - max(seconds)))
    for n, r in enumerate(runs):
        y = rows_y[n]
        prof, prof_words = profiles[n]
        env, env_source = envs[n]
        run_label(ov, r, X0, y + ov.U(18), left_w)
        yy = y + ov.U(40)
        for text, colour, lines in left_lines(ov, ctx, r, row_h, left_w):
            yy = wrapped(ov, X0, yy, text, left_w, colour=colour, max_lines=lines)
        yy += ov.U(4)
        bar_w = sk_x - X0 - ov.U(30)
        # every bar is drawn: they share the room left under the words (left_lines keeps enough for them)
        step = min(ov.U(40), (y + row_h - yy) / len(BARS))
        if env is None:
            ov.text((X0, yy + ov.U(12)), "the run recorded no CAPE or shear", "micro", colour="muted", anchor="ls")
        else:
            for (label, path, unit, fmt), topv in zip(BARS, bar_top):
                value = _num(env, *path)
                ov.text((X0, yy + ov.U(12)), label, "micro", colour="muted", anchor="ls")
                text = "not recorded" if value is None else f"{fmt.format(value)} {unit}"
                ov.text((X0 + bar_w, yy + ov.U(12)), text, "micro", colour="text", anchor="rs")
                ov.rule(X0, yy + ov.U(17), bar_w, thickness=4, colour="rule")
                if value is not None:
                    ov.rule(X0, yy + ov.U(17), max(1.0, bar_w * min(1.0, abs(value) / topv)), thickness=4,
                            colour=r.colour)
                yy += step
        if prof is None:
            ov.text((sk_x + sk_w / 2, y + row_h / 2), "no profile", "label", colour="muted", anchor="ms")
            wrapped(ov, sk_x + ov.U(12), y + row_h / 2 + ov.U(24), prof_words, sk_w - ov.U(24), max_lines=2)
        # what it grew
        state = states[n]
        entry: dict[str, Any] = {"letter": r.letter, "profile": prof_words, "sounding_numbers": env_source}
        if state is not None and boxes[n] is not None:
            box, words, use = boxes[n]
            if use not in state.missing:
                centre_ring(ov, box.to_px(state.centre.x, state.centre.y), ring_radius(box.w))
            else:
                ov.text((box.x + box.w / 2, box.y + box.h / 2), "not drawn", "label", colour="muted", anchor="ms")
            backed_scale_bar(ov, box)
            numbers = grown[n]
            per_col = math.ceil(len(numbers) / cols)
            for m, (cap, val, vertical) in enumerate(numbers):
                c, rr = divmod(m, per_col)
                ov.readout((col_x[c], y + ov.U(2) + rr * ov.U(60)), [(cap, val)],
                           vertical=[val] if vertical else ())
            yy2 = y + ov.U(6) + per_col * ov.U(60) + ov.U(14)
            # the numbers above are the whole area's (its time series); the words under the box are the box's
            yy2 = wrapped(ov, nu_x, yy2, S.data_line(words, "numbers above: its whole area",
                                                     f"{r.letter} at {ctx.clock(state.t)}"), wcol,
                          colour="data", max_lines=2)
            yy2 = wrapped(ov, nu_x, yy2, state.centre.words, wcol, vertical=True, max_lines=2)
            if state.source == "pictures" and use not in state.missing:
                wrapped(ov, nu_x, yy2, "storm cut from its own picture: its saved frames were deleted", wcol,
                        colour="warm", max_lines=1)
            entry.update({"time_s": state.t, "look": use, "numbers": {c: v for c, v, _ in numbers},
                          "centre": state.centre.kind})
        facts_out.append(entry)
    # keys
    key_y = ov.U(946)
    kx = X0
    for label_text, colour in (("temperature", T_COLOUR), ("dewpoint", TD_COLOUR), ("cloud base (LCL)", BASE_T)):
        ov.rule(kx, key_y - ov.U(5), ov.U(22), thickness=2, colour=colour)
        ov.text((kx + ov.U(28), key_y), label_text, "micro", colour="data", anchor="ls", halo=False)
        kx += ov.U(28) + ov.measure(label_text, "micro")[0] + ov.U(18)
    kx += ov.U(10)
    ov.text((kx, key_y), "hodograph, ground-relative (m/s):", "micro", colour="muted", anchor="ls", halo=False)
    kx += ov.measure("hodograph, ground-relative (m/s):", "micro")[0] + ov.U(12)
    for (_z0, _z1, colour), label_text in zip(HODO_COLOURS, ("0–1 km", "1–3", "3–6", "6–9", "9–12")):
        ov.rule(kx, key_y - ov.U(5), ov.U(22), thickness=3, colour=colour)
        ov.text((kx + ov.U(28), key_y), label_text, "micro", colour="data", anchor="ls", halo=False)
        kx += ov.U(28) + ov.measure(label_text, "micro")[0] + ov.U(16)
    if look == "reflectivity":
        P.dbz_legend(ov, X1 - ov.U(260), key_y - ov.U(10), ov.U(260))
    sources = _grouped(runs, lambda r: profiles[runs.index(r)][1])
    notes = S.data_line(f"profiles: {sources}", "numbers: each run's own before it started (CM1 getcape mixed-layer "
                        "parcel, Bunkers right mover)", "one scale for every run")
    level = "the whole column; storm: 1 km" if look == "reflectivity" else "the whole column"
    ov.footer([footer_facts(ctx, level=level, wind="ground-relative"), notes])
    ov.mark()
    P.save(ov, base, out)
    return {"runs": facts_out, "look": look, "row_px": row_h, "hodograph_limit_m_s": lim}


def _grew(ctx: Context, pts: Mapping[str, list], state: Any, words: str, look: str) -> list[tuple[str, str, bool]]:
    """The numbers beside a run's storm: from its time series at its drawn time."""

    t = state.t
    rows: list[tuple[str, str, bool]] = []
    if ctx.hurricane:
        p = _value_at(pts, "prs_sfc_min_hpa", t)
        if p is not None:
            rows.append(("lowest pressure, lowest level", f"{p:.0f} hPa", False))
    w = _value_at(pts, "wmax", t)
    if w is not None:
        rows.append(("strongest updraft", f"{w:.0f} m/s", True))
    if not ctx.hurricane:
        uh = _value_at(pts, "uh_max", t)
        if uh is not None:
            rows.append(("max 2–5 km updraft helicity", f"{uh:,.0f} m² s⁻²", False))
    wind = _value_at(pts, "wind_sfc_max", t)
    if wind is not None:
        rows.append(("fastest wind, lowest level", f"{wind:.0f} m/s ground-relative", False))
    if not ctx.hurricane:
        th = _value_at(pts, "thpert_sfc_min", t)
        if th is not None and len(rows) < 4:
            rows.append((f"coldest θ{S.PRIME}, lowest level", f"{S.fmt_num(th, 1)} K", False))
    return rows[:4]


__all__ = ["Context", "NotDrawn", "card_groups", "series_picture", "soundings_card", "storm_picture"]
