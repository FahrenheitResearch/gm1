"""The time series: domain extremes at every saved frame, with CM1's own curve when the run is CM1's stock storm.

``series.json`` (``gm1.quicklook-series/v1``) is what the Run and Results screens chart; ``timeseries.png`` is the
same in the quick-look style, for a terminal or a cluster.  The statistics use the definitions the release gate
used: max and min w over the whole domain, the largest 2-5 km updraft
helicity, the coldest θ′ and the area colder than -1 K at the lowest model level, the most rain in one cell.

The CM1 reference (``records/CM1-STOCK-SUPERCELL-SERIES.json``) is CM1 r21.1 itself on its stock supercell namelist,
measured on a Linux workstation.  It is drawn only when the run's namelist is that namelist in every setting that changes the
storm: output choices and the output, restart and statistics cadence may differ, and a shorter run shows the
reference up to its own end.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import style as S

SERIES_SCHEMA = "gm1.quicklook-series/v1"
RECORD = "CM1-STOCK-SUPERCELL-SERIES.json"
#: (key, label, unit): the first six are the charts the Run screen draws, with CM1's curve where it has one.
CHARTS: tuple[tuple[str, str, str], ...] = (
    ("wmax", "strongest updraft", "m/s"),
    ("wmin", "strongest downdraft", "m/s"),
    ("uh_max", "max 2–5 km updraft helicity", "m²/s²"),
    ("thpert_sfc_min", f"coldest θ{S.PRIME} at {{z0}}", "K"),
    ("coldpool_area_km2_lt_minus1K", f"cold pool, θ{S.PRIME} < {S.MINUS}1 K", "km²"),
    ("rain_max_cm", "most rain in one cell", "cm"),
    ("dbz_max", "highest reflectivity", "dBZ"),
    ("wind_sfc_max", "fastest wind at {z0}, ground-relative", "m/s"),
    ("zeta_sfc_max", "strongest spin at {z0}", S.PER_S),
)
#: A hurricane run's charts, in order: its lowest pressure and its fastest wind first, and none of the charts made
#: for one thunderstorm (updraft helicity, the cold pool, the near-ground spin).  The pressure is the lowest model
#: level's, not sea level; its chart's value axis does not start at zero (``"zero": false``).
HURRICANE_CHARTS: tuple[tuple[str, str, str], ...] = (
    ("prs_sfc_min_hpa", "lowest pressure at {z0} (not sea level)", "hPa"),
    ("wind_sfc_max", "fastest wind at {z0}, ground-relative", "m/s"),
    ("wmax", "strongest updraft", "m/s"),
    ("rain_max_cm", "most rain in one cell", "cm"),
    ("dbz_max", "highest reflectivity", "dBZ"),
    ("wmin", "strongest downdraft", "m/s"),
)
#: Charts whose value axis should not be stretched down to zero (a pressure near 1000 hPa).
NOT_FROM_ZERO = frozenset({"prs_sfc_min_hpa"})
#: A run this long (its end time, or its saved times) is charted in hours, not minutes (``"time_unit": "h"``).
HOURS_FROM_S = 172800.0
#: Settings that do not change the storm: output selection, statistics, logging, and the cadence of writes.
EXCLUDED_GROUPS = ("param9", "param10", "param13", "param14", "param15", "param16", "logging",
                   "nssl2mom_params", "nssl_mp_params")
EXCLUDED_KEYS = ("timax", "run_time", "tapfrq", "rstfrq", "statfrq", "prclfrq", "irst", "rstnum",
                 "output_format", "output_filetype", "output_interp", "restart_format", "restart_filetype",
                 "restart_reset_frqtim", "restart_file_theta", "restart_use_theta", "restart_file_dbz",
                 "restart_file_th0", "restart_file_prs0", "restart_file_pi0", "restart_file_rho0",
                 "restart_file_qv0", "restart_file_u0", "restart_file_v0", "restart_file_zs",
                 "restart_file_zh", "restart_file_zf", "restart_file_diags", "restart_file_azimavg",
                 "timestats", "terrain_flag", "procfiles", "outunits", "iorigin", "nodex", "nodey", "ppnode",
                 "timeformat", "timestats", "dowriteout", "stopit")


def storm_settings(values: Mapping[str, Any]) -> dict[str, Any]:
    """The settings that change the storm, from a namelist's effective values."""

    from ..cm1import import defaults

    table = defaults.load()
    out = {}
    for key, value in values.items():
        if key in EXCLUDED_KEYS or key.startswith("output_") or key.startswith("restart_"):
            continue
        groups = table.groups_of(key)
        if groups and all(group in EXCLUDED_GROUPS for group in groups):
            continue
        out[key] = value
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-12)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@lru_cache(maxsize=1)
def record() -> Mapping[str, Any]:
    text = resources.files("cm1core.records").joinpath(RECORD).read_text(encoding="utf-8")
    payload = json.loads(text)
    if payload.get("schema") != "gm1.cm1-stock-supercell-series/v1":
        raise ValueError(f"{RECORD} has an unknown schema")
    return payload


def reference_for(settings: Any) -> tuple[Mapping[str, Any] | None, str]:
    """(the CM1 reference for this run, why there is none): only for a namelist equal to the reference's."""

    if settings is None or getattr(settings, "source", "") != "namelist":
        return None, "no CM1 namelist to compare with CM1's stock storm"
    mine = storm_settings(settings.values)
    for ref in record()["references"]:
        theirs = ref["storm_settings"]
        differ = sorted(k for k in set(mine) | set(theirs) if not _same(mine.get(k), theirs.get(k)))
        if not differ:
            return ref, ""
    return None, "the namelist is not CM1's stock supercell (settings that change the storm differ)"


def series_document(rows: Sequence[tuple[float, Mapping[str, float]]], *, z0: float, name: str,
                    reference: Mapping[str, Any] | None, t_end: float | None, kind: str = "") -> dict[str, Any]:
    """``kind``: ``"hurricane"`` charts :data:`HURRICANE_CHARTS`; any other run charts :data:`CHARTS`."""

    charts = []
    for key, label, unit in (HURRICANE_CHARTS if kind == "hurricane" else CHARTS):
        points = [[t, round(float(r[key]), 6)] for t, r in rows if key in r]
        if not points:
            continue
        chart: dict[str, Any] = {"key": key, "label": label.format(z0=S.fmt_height(z0)), "unit": unit,
                                 "gm1": points}
        if key in NOT_FROM_ZERO:
            chart["zero"] = False
        if reference is not None and key in reference["series"]:
            chart["cm1"] = [p for p in reference["series"][key] if t_end is None or p[0] <= t_end + 1e-6]
        charts.append(chart)
    doc: dict[str, Any] = {
        "schema": SERIES_SCHEMA,
        "gm1_label": f"GM1, {name}",
        "note": "domain extremes at every saved frame; minutes since the start on the horizontal axis",
        "charts": charts,
    }
    if (t_end is not None and t_end >= HOURS_FROM_S) or (rows and max(t for t, _ in rows) >= HOURS_FROM_S):
        # A long run (two days of storm time or more, by its own end time or its saved times) is charted in hours.
        doc["time_unit"] = "h"
        doc["note"] = "domain extremes at every saved frame; hours since the start on the horizontal axis"
    if kind:
        doc["storm_kind"] = kind
    if reference is not None:
        doc["cm1_label"] = reference["label"]
        doc["provenance"] = reference["provenance"]
    return doc


def clock_words(doc: Mapping[str, Any], t: float) -> str:
    """The sheet's time, in its header and in its footer: a hurricane's charts run for days, so its time reads
    as the page around it does (hours and days); any other run's as ``T+H:MM:SS``."""

    return S.fmt_long_clock(t) if doc.get("storm_kind") == "hurricane" else S.fmt_clock(t)


def render_png(doc: Mapping[str, Any], path: Path, *, t_now: float, footer: Sequence[str]) -> Path:
    """The first six charts in the quick-look style (GM1 orange, CM1 dashed blue: dataviz slots 1 and 2)."""

    import os

    import numpy as np

    W, H = 1920, 1080
    fig = S.figure((W, H))
    charts = [c for c in doc["charts"]][:6]
    left, right, top, bottom = 110 / W, 1 - 80 / W, 1 - 250 / H, 185 / H
    cols, rows_n = 3, 2
    gw, gh = 90 / W, 110 / H
    pw = (right - left - (cols - 1) * gw) / cols
    ph = (top - bottom - (rows_n - 1) * gh) / rows_n
    hours = doc.get("time_unit") == "h"
    per = 3600.0 if hours else 60.0
    tmax = max([p[0] for c in charts for p in c["gm1"] + c.get("cm1", [])] + [t_now, 60.0]) / per
    fp = S.mpl_fontprops("label", fig)
    fm = S.mpl_fontprops("micro", fig)
    gm1_colour, cm1_colour = "#d95926", "#3987e5"
    for n, chart in enumerate(charts):
        r, c = divmod(n, cols)
        ax = fig.add_axes((left + c * (pw + gw), top - (r + 1) * ph - r * gh, pw, ph))
        ax.set_facecolor(S.PALETTE["void"])
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(S.PALETTE["rule"])
        ax.grid(True, axis="y", color=S.PALETTE["rule"], linewidth=1.0)
        ax.tick_params(colors=S.PALETTE["muted"], labelsize=fm.get_size())
        for key, colour, style in (("cm1", cm1_colour, (0, (4, 3))), ("gm1", gm1_colour, "-")):
            pts = chart.get(key) or []
            if pts:
                x = [p[0] / per for p in pts]
                y = [p[1] for p in pts]
                ax.plot(x, y, color=colour, linewidth=2.0, linestyle=style, zorder=3 if key == "gm1" else 2)
                ax.plot([x[-1]], [y[-1]], "o", color=colour, markersize=5, markeredgecolor=S.PALETTE["void"],
                        markeredgewidth=1.5, zorder=4)
        ax.axvline(t_now / per, color=S.PALETTE["gold"], linewidth=1.0, alpha=0.6)
        ax.set_xlim(0, tmax * 1.02)
        ax.set_title(f"{chart['label']} ({chart['unit']})", loc="left", fontproperties=fp,
                     color=S.PALETTE["text"], pad=8)
        ax.set_xlabel("hours since the start" if hours else "minutes since the start", fontproperties=fm,
                      color=S.PALETTE["muted"])
    base = S.fig_to_rgb(fig)
    ov = S.Overlay((W, H))
    has_cm1 = any(c.get("cm1") for c in charts)
    ov.title("Storm over time", S.data_line(doc["gm1_label"], "domain extremes at every saved frame",
                                            "gold line: the latest frame"))
    ov.clock(clock_words(doc, t_now))
    key_y = ov.U(178)
    ov.rule(ov.x0, key_y, ov.U(40), thickness=3, colour=gm1_colour)
    ov.text((ov.x0 + ov.U(52), key_y + ov.U(6)), doc["gm1_label"], "label", colour="data", anchor="ls")
    if has_cm1:
        x2 = ov.x0 + ov.U(420)
        for k in range(3):
            ov.rule(x2 + k * ov.U(16), key_y, ov.U(10), thickness=3, colour=cm1_colour)
        ov.text((x2 + ov.U(58), key_y + ov.U(6)), doc["cm1_label"], "label", colour="data", anchor="ls")
    lines = list(footer)
    if has_cm1:
        lines = [lines[0], doc["provenance"]]
    ov.footer(lines[:2])
    ov.mark()
    problems = ov.check()
    if problems:
        from .products import LintError

        raise LintError("; ".join(problems))
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    Image.fromarray(ov.composite(np.asarray(base))).save(tmp, format="PNG", compress_level=6)
    os.replace(tmp, path)
    return path


__all__ = ["CHARTS", "HURRICANE_CHARTS", "SERIES_SCHEMA", "reference_for", "record", "render_png", "series_document",
           "storm_settings"]
