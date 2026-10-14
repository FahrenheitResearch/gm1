"""How this storm was simulated: one card per run, drawn from the run's own records.

Drawn once per run (and again when a part of the run ends and writes its receipt), from:

* every segment's ``receipt.json``, the record of what ran: the GPU (``runtime_benchmark.gpu_name``), the time
  (``process_wall_seconds``; ``runtime_benchmark.wall_seconds`` and ``step_wall_seconds.total``), the storm time
  (``simulated_seconds``), the steps each segment ran (``executed_steps``; ``completed_steps`` is the clock's count
  since the start), the frames and restarts each published (``frame_rows_retained`` and ``restart_rows_retained``
  with their dropped rows; the cadence is only the plan) and the interval between saves (``integration_cadence``),
  the GPU memory sampled at its peak (``sampled_peak_device_used_bytes``, the whole card, and
  ``sampled_peak_registered_model_bytes``, the model's own arrays, of ``driver_component_warmup.device_total_bytes``),
  the physics as it ran (``runtime_benchmark.config``), the levels (``vertical_coordinate``), the start
  (``initialization``) and the starting air's numbers (``initial_sounding_parameters``);
* without a receipt yet (the first part still running), the newest frame's record of the same physics
  (``history-publication.json`` ``config``), and the card says the time and memory arrive when the part ends;
* the run's namelist or deck (``cm1out.settings``) for the words a receipt does not carry: the advection order and
  the Coriolis force;
* ``run.json`` for the case's name;
* one saved frame for the map and the levels (the frame's own grid): the saved time with the strongest echo, or a
  hurricane's newest, whose map shows the inner 500 km around the area's middle (as the hurricane looks do).

The computing time is split only where the host clock can split it: stepping the model, the checks and saves at each
output, and starting up.  The GPU is waited for before each output (``cuda_cartesian_runtime`` integrate), so the
stepping and the output can only trade the steps still queued then; no split by physics is drawn, since the GPU's
work runs asynchronously and per-phase host timers do not measure it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import products as P
from . import style as S

TITLE = "How this storm was simulated"
#: The three parts of the computing time (colour, words).
PARTS = (("#2f86cc", "stepping the model"), ("#8b98ad", "checks and saves"), ("#c77716", "starting up"))
SIDES = {1: "repeat (periodic)", 2: "open (waves leave)", 3: "walls (free slip)", 4: "walls (no slip)"}
GROUND = {1: "free slip (no drag)", 2: "no slip", 3: "surface drag"}


def _read(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dig(doc: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(doc, Mapping):
            return None
        doc = doc.get(key)
    return doc


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def duration(seconds: float) -> str:
    """'2 h', '1 h 15 min', '43 min 05 s', '37 s', '3 d 9 h': rounded to the last unit shown (a time with hours to
    the nearest minute, one with days to the nearest hour), so 5,614 s is '1 h 34 min'."""

    def nearest(value: float) -> int:
        return int(math.floor(value + 0.5))

    s = nearest(max(0.0, seconds))
    if s >= 3600:
        whole_minutes = nearest(s / 60.0)
        if whole_minutes < 24 * 60:
            hours, minutes = divmod(whole_minutes, 60)
            return f"{hours} h {minutes:02d} min" if minutes else f"{hours} h"
        days, hours = divmod(nearest(s / 3600.0), 24)
        return f"{days} d {hours} h" if hours else f"{days} d"
    minutes, secs = divmod(s, 60)
    if minutes:
        return f"{minutes} min {secs:02d} s" if secs else f"{minutes} min"
    return f"{secs} s"


def _published(receipts: Sequence[Mapping[str, Any]], kind: str) -> float | None:
    """How many frames (``kind`` "frame") or restarts a run's segments published: each receipt's rows kept and rows
    dropped (``runtime_benchmark.frame_rows_retained`` ...), else the length of its list of them."""

    count = 0.0
    for doc in receipts:
        rb = doc.get("runtime_benchmark") or {}
        kept, dropped = _num(rb.get(f"{kind}_rows_retained")), _num(rb.get(f"{kind}_rows_dropped"))
        if kept is not None:
            count += kept + (dropped or 0.0)
        elif isinstance(rb.get(f"{kind}s"), list):
            count += len(rb[f"{kind}s"])
        else:
            return None
    return count


def gpu_words(name: str) -> str:
    return name.replace("NVIDIA ", "").replace("GeForce ", "").strip()


# ---------------------------------------------------------------- the run's records
def gather(root: Path, segments: Sequence[Path], *, settings: Any = None, frame_config: Mapping[str, Any] | None = None,
           frames_saved: int | None = None) -> dict[str, Any]:
    """Everything the card says, each with where it came from, as plain numbers and words."""

    root = Path(root)
    run = _read(root / "run.json")
    run = run if isinstance(run, dict) else {}
    receipts = []
    for segment in segments:
        doc = _read(Path(segment) / "receipt.json")
        if isinstance(doc, dict) and isinstance(doc.get("runtime_benchmark"), dict):
            receipts.append(doc)
    last = receipts[-1] if receipts else {}
    rb = last.get("runtime_benchmark") or {}
    config = rb.get("config") if isinstance(rb.get("config"), dict) else (dict(frame_config or {}))
    facts: dict[str, Any] = {"receipts": len(receipts), "segments": len(list(segments)),
                             "name": str(run.get("name") or root.name), "case": run.get("case"),
                             "config_source": "receipt" if receipts else "frame"}
    # versions and the engine
    versions = []
    for doc in receipts:
        if doc.get("version") and str(doc["version"]) not in versions:
            versions.append(str(doc["version"]))
    if not versions and _dig(run, "gm1", "version"):
        versions = [str(run["gm1"]["version"])]
    facts["versions"] = versions
    facts["arwen"] = _dig(rb, "driver_component_warmup", "component_receipt", "arwen", "installed_version")
    # the GPU and the time
    gpus = []
    for doc in receipts:
        name = _dig(doc, "runtime_benchmark", "gpu_name")
        if isinstance(name, str) and gpu_words(name) not in gpus:
            gpus.append(gpu_words(name))
    if not gpus and isinstance(_dig(run, "gpu", "name"), str):
        gpus = [gpu_words(run["gpu"]["name"])]
    facts["gpus"] = gpus
    if receipts:
        def total(*keys: str) -> float | None:
            values = [_num(_dig(doc, *keys)) for doc in receipts]
            return None if any(v is None for v in values) else float(sum(values))

        facts["process_s"] = total("process_wall_seconds")
        facts["integrate_s"] = total("runtime_benchmark", "wall_seconds")
        facts["stepping_s"] = total("runtime_benchmark", "step_wall_seconds", "total")
        facts["storm_s"] = total("runtime_benchmark", "simulated_seconds")
        # the steps each segment ran (completed_steps is the clock's count since the start, which a resumed
        # segment carries on), and the frames and restarts each published (the cadence is only the plan)
        facts["steps"] = total("runtime_benchmark", "executed_steps")
        facts["frames"] = _published(receipts, "frame")
        facts["restarts"] = _published(receipts, "restart")
        rows = [(_num(_dig(doc, "runtime_benchmark", "step_wall_seconds", "mean")),
                 _num(_dig(doc, "runtime_benchmark", "step_wall_seconds", "count"))) for doc in receipts]
        rows = [(m, c) for m, c in rows if m is not None and c]
        if rows:
            facts["step_mean_s"] = sum(m * c for m, c in rows) / sum(c for _m, c in rows)
        peaks = [_num(_dig(doc, "runtime_benchmark", "sampled_peak_device_used_bytes")) for doc in receipts]
        model = [_num(_dig(doc, "runtime_benchmark", "sampled_peak_registered_model_bytes")) for doc in receipts]
        card = [_num(_dig(doc, "runtime_benchmark", "driver_component_warmup", "device_total_bytes"))
                for doc in receipts]
        facts["memory_peak_bytes"] = max([p for p in peaks if p is not None], default=None)
        facts["memory_model_bytes"] = max([p for p in model if p is not None], default=None)
        facts["memory_card_bytes"] = max([p for p in card if p is not None], default=None)
        facts["storm_end_s"] = _num(_dig(last, "runtime_benchmark", "model_seconds_at_end"))
        facts["completed_utc"] = last.get("completed_utc")
        facts["status"] = last.get("status")
    cadence = last.get("integration_cadence") or {}
    facts["dt_s"] = _num(cadence.get("dt_s")) or _num(config.get("dt_s"))
    facts["interval_s"] = _num(cadence.get("history_interval_s"))
    facts["adaptive"] = bool(config.get("adapt_dt")) or rb.get("adaptive_timestep") not in (None, False, {})
    facts["substeps"] = config.get("acoustic_substeps")
    if frames_saved is not None and facts.get("frames") is None:
        facts["frames_on_disk"] = frames_saved
    # the grid and the levels
    grid = config.get("grid") if isinstance(config.get("grid"), dict) else {}
    vertical = last.get("vertical_coordinate") or grid.get("vertical") or {}
    facts["grid"] = {k: grid.get(k) for k in ("nx", "ny", "nz", "dx_m", "dy_m", "dz_m")}
    facts["vertical"] = {k: vertical.get(k) for k in ("stretch_z", "nk", "dz_bot_m", "dz_top_m", "str_bot_m",
                                                      "str_top_m", "ztop_m", "lowest_mass_level_m",
                                                      "levels_below_1000m", "dz_reference_m")}
    facts["moving"] = {"u": _num(config.get("u_move_m_s")), "v": _num(config.get("v_move_m_s"))}
    # the physics as it ran
    facts["microphysics"] = config.get("microphysics")
    facts["dense_ice"] = _dig(config, "dense_ice", "category")
    facts["numq"] = _dig(config, "moisture_species", "numq")
    facts["turbulence"] = {k: _dig(config, "turbulence_scale", k) for k in ("cm1setup", "sgsmodel", "ipbl")}
    facts["radopt"] = _dig(config, "radiation", "radopt")
    facts["cuparam"] = _dig(config, "cumulus", "cuparam")
    lateral = config.get("lateral_boundaries") if isinstance(config.get("lateral_boundaries"), dict) else {}
    facts["sides"] = {k: lateral.get(k) for k in ("wbc", "ebc", "sbc", "nbc")}
    for key in ("bbc", "tbc", "isfcflx", "sfcmodel", "cecd", "cnstcd", "tsk0", "oceanmodel", "damping_depth_m",
                "irdamp", "rterm"):
        facts[key] = config.get(key)
    # the start
    init = last.get("initialization") or {}
    facts["init"] = {k: init.get(k) for k in ("iinit", "isnd", "iwnd", "bptpert_k", "bhrad_m", "bvrad_m", "zc_m",
                                              "hurr_vg_m_per_s", "fcor_s_inv")}
    environment = last.get("initial_sounding_parameters") or {}
    facts["environment"] = {
        "cape_ml": _num(_dig(environment, "mixed_layer", "cape_j_per_kg")),
        "cin_ml": _num(_dig(environment, "mixed_layer", "cin_j_per_kg")),
        "shear_0_6": _num(environment.get("shear_0_6km_m_per_s")),
        "srh_0_3": _num(environment.get("srh_0_3km_m2_per_s2")),
    }
    facts["own_sounding"] = (root / "input_sounding").is_file()
    facts["own_sounding_converted"] = (root / "input_sounding.original").is_file()
    # what only the namelist or deck says
    source = getattr(settings, "source", None) if settings is not None else None
    facts["settings_source"] = source if source in ("namelist", "deck") else None
    if settings is not None:
        for key in ("hadvordrs", "vadvordrs", "advwenos", "icor", "iinit", "isnd", "imove", "fcor"):
            try:
                facts.setdefault("namelist", {})[key] = settings.get(key)
            except (AttributeError, TypeError):
                pass
        if facts["init"].get("iinit") is None:
            facts["init"]["iinit"] = settings.get("iinit")
            facts["init"]["isnd"] = settings.get("isnd")
            facts["init"]["bptpert_k"] = settings.get("bptpert")
    return facts


# ---------------------------------------------------------------- the words
def _seconds_words(s: float) -> str:
    return f"{s:.3g} s"


#: CM1's overall setup (``cm1setup``, README.namelist:122-147), named, not described as resolving eddies: a 4 km
#: hurricane runs CM1's large-eddy setup too, and no turbulent eddy is on its grid.
SETUPS = {0: "no subgrid turbulence model (cm1setup 0)", 1: "CM1's large-eddy setup (cm1setup 1)",
          2: "CM1's mesoscale setup (cm1setup 2)", 3: "CM1's direct numerical setup (cm1setup 3)",
          4: "large-eddy setup inside a mesoscale one (cm1setup 4)"}
#: The subgrid model of the large-eddy setup (``sgsmodel``, README.namelist:334-346).
SUBGRID = {1: "1.5-order TKE mixing in all directions", 2: "Smagorinsky mixing in all directions",
           3: "1.5-order TKE mixing in all directions, two-part", 4: "1.5-order TKE mixing in all directions"}
#: Tile lines whose speeds are the same in every frame (a wind difference, the bulk shear: moving the frame changes
#: both winds alike), so the picture's rule that a wind speed names its frame (``style.lint``) does not apply.
FRAME_FREE = ("wind change ",)
#: The boundary-layer schemes of CM1's mesoscale setup (``ipbl``, README.namelist:310-332).
PBL = {1: "YSU boundary-layer scheme", 2: "CM1's simple boundary-layer scheme", 3: "GFS-EDMF boundary-layer scheme",
       4: "MYNN 2.5 boundary-layer scheme", 5: "MYNN 3 boundary-layer scheme", 6: "MYJ boundary-layer scheme"}
#: CM1's stand-in for radiation (``rterm`` = 1, solve1.F:1003-1054): the potential temperature's departure from the
#: base state relaxed over 12 hours, capped at 1 K per 12 hours (Rotunno and Emanuel 1987).
RTERM_WORDS = "no radiation scheme; CM1's simple cooling instead, at most 2 K a day (rterm)"


def physics_tiles(facts: Mapping[str, Any], scheme_words: str, checked: str) -> list[tuple[str, list[str]]]:
    """The five tiles under the card, in plain words, every one from the records."""

    names = facts.get("namelist") or {}
    dt = facts.get("dt_s")
    substeps = facts.get("substeps")
    equations = ["CM1 r21.1's equations on the GPU,", "compressible, split into sound steps:"]
    if facts.get("adaptive"):
        equations.append("the step chosen as it ran" + (f", {substeps} sound steps each" if substeps else ""))
    elif dt:
        equations.append(f"{substeps} sound steps per {_seconds_words(dt)} step" if substeps else
                         f"a {_seconds_words(dt)} step")
    order = names.get("hadvordrs")
    if isinstance(order, int) and order > 0:
        weno = ", WENO for heat and water" if names.get("advwenos") not in (None, 0) else ""
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(order, "th")
        equations.append(f"{order}{suffix}-order advection{weno}")
    # the Coriolis term is one of the equations' terms (and leaves room under Edges for the heating)
    icor = names.get("icor")
    fcor = _num(names.get("fcor")) or _num((facts.get("init") or {}).get("fcor_s_inv"))
    if icor == 1 and fcor:
        exponent = math.floor(math.log10(abs(fcor)))
        mantissa = fcor / 10.0 ** exponent
        power = str(exponent).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
        equations.append(f"Earth's spin on: f = {mantissa:g} × 10{power} s⁻¹")
    elif icor == 0:
        equations.append("no Earth's spin (Coriolis off)")
    rain = [scheme_words + (f", {facts['numq']} moisture fields" if facts.get("numq") else "")]
    if facts.get("arwen"):
        rain.append(f"run by ArWen {facts['arwen']} (gpuwm)")
    rain.append(checked)
    turb = facts.get("turbulence") or {}
    turbulence = []
    setup = turb.get("cm1setup")
    if setup is not None:
        turbulence.append(SETUPS.get(setup, f"CM1 setup {setup} (cm1setup)"))
    model = turb.get("sgsmodel")
    if model is not None and setup in (1, 4, None):          # CM1 reads sgsmodel only in its large-eddy setups
        turbulence.append(SUBGRID.get(model, f"CM1's subgrid model {model} (sgsmodel)"))
    extra = []
    if turb.get("ipbl") in (0, None):
        extra.append("no boundary-layer scheme")
    else:
        turbulence.append(PBL.get(turb["ipbl"], f"CM1's boundary-layer scheme {turb['ipbl']} (ipbl)"))
    if facts.get("cuparam") in (0, None):
        extra.append("no cumulus scheme")
    if extra:
        turbulence.append(", ".join(extra))
    sides = facts.get("sides") or {}
    values = {v for v in sides.values() if v is not None}
    edges = []
    if len(values) == 1:
        edges.append("sides: " + SIDES.get(values.pop(), "as the run's settings say"))
    elif values:
        edges.append("sides: west and east " + SIDES.get(sides.get("wbc"), "?") + ", south and north "
                     + SIDES.get(sides.get("sbc"), "?"))
    ground = GROUND.get(facts.get("bbc"), "")
    if facts.get("bbc") == 3 and facts.get("cecd") == 1 and _num(facts.get("cnstcd")) is not None:
        ground = f"surface drag, drag coefficient {facts['cnstcd']:g}"
    if ground:
        edges.append("ground: " + ground)
    heat = []
    if facts.get("isfcflx") == 1:
        if facts.get("oceanmodel") is not None and _num(facts.get("tsk0")) is not None and facts.get("sfcmodel"):
            heat.append(f"heat and moisture from a {float(facts['tsk0']) - 273.15:.1f} °C sea")
        else:
            heat.append("heat and moisture from the surface")
    else:
        heat.append("no surface heating")
    if facts.get("radopt") not in (0, None):
        heat.append(f"radiation (radopt {facts['radopt']})")
    elif facts.get("rterm") == 1:
        edges.append(", ".join(heat))
        heat = [RTERM_WORDS]
    else:
        heat.append("no radiation")
    edges.append(", ".join(heat))
    init = facts.get("init") or {}
    start = []
    isnd = init.get("isnd")
    if isnd == 7:
        start.append("your own sounding (input_sounding)" if facts.get("own_sounding_converted")
                     else "a sounding file (input_sounding)")
    elif isnd == 5:
        start.append("CM1's built-in supercell sounding")
    elif isnd is not None:
        start.append(f"CM1 sounding number {isnd} (isnd)")
    iinit = init.get("iinit")
    if iinit == 1:
        bubble = f"a {init.get('bptpert_k'):g} K warm bubble" if _num(init.get("bptpert_k")) else "a warm bubble"
        if _num(init.get("bhrad_m")) and _num(init.get("zc_m")):
            bubble += f", {init['bhrad_m'] / 1000:g} km radius, at {S.fmt_height(init['zc_m'])}"
        start.append(bubble)
    elif iinit == 7:
        start.append("CM1's hurricane swirl")
    elif iinit == 12:
        start.append("a push of rising air (updraft nudging)")
    elif iinit is not None:
        start.append(f"CM1 start number {iinit} (iinit)")
    env = facts.get("environment") or {}
    if env.get("cape_ml") is not None:
        start.append(f"CAPE {env['cape_ml']:,.0f} J/kg (mixed layer)")
    if env.get("shear_0_6") is not None and env["shear_0_6"] > 0.5:
        start.append(f"wind change 0–6 km (bulk shear): {env['shear_0_6']:.0f} m/s")
    return [("The equations", equations), ("Rain and hail", rain), ("Turbulence", [t for t in turbulence if t]),
            ("Edges and ground", edges), ("The start", start)]


def level_words(facts: Mapping[str, Any], zf: np.ndarray) -> tuple[str, str]:
    """(the big line, the small line) under the levels drawing."""

    zh = 0.5 * (zf[1:] + zf[:-1])
    dz = np.diff(zf)
    nz = len(dz)
    big = f"{nz} levels  ·  lowest at {S.fmt_height(float(zh[0]))}"
    if float(dz.max() - dz.min()) < 0.5:
        small = f"even layers of {S.fmt_height(float(dz[0]))} up to {S.fmt_height(float(zf[-1]))}"
    else:
        below = int((zh < 1000.0).sum())
        small = (f"layers {S.fmt_height(float(dz.min()))} deep near the ground growing to "
                 f"{S.fmt_height(float(dz.max()))}; {below} levels below 1 km")
    return big, small


# ---------------------------------------------------------------- drawing
def _box_map(qf: Any, facts: Mapping[str, Any], box: P.Box) -> tuple[np.ndarray | None, str]:
    """The map's pixels (column-maximum reflectivity) and what it is, or None and why there is none."""

    from . import reflectivity as R

    try:
        refl = R.volume(qf)
    except Exception as error:  # a frame that cannot be read gives a card without a map, not no card
        return None, f"no map: {type(error).__name__}"
    if not refl.available:
        reason = refl.reason or "no reflectivity for this scheme"
        if len(reason) > 1 and reason[1].islower():               # 'This run ...' reads on after the colon
            reason = reason[0].lower() + reason[1:]
        return None, "no map: " + reason
    column = refl.dbz.max(axis=0)
    return P.dbz_image(P.sample(column, box, qf.dx, qf.dy)), refl.operator


def map_box(lx: float, ly: float, x: int, y: int, w_max: int, h_max: int, inner: float | None = None) -> P.Box:
    """The map's box: the whole area, or with ``inner`` (m) a square that wide around the area's middle (a
    hurricane's inner part, as its own looks show it)."""

    if not inner:
        return P.fit_box(lx, ly, x, y, w_max, h_max)
    sx, sy = min(inner, lx), min(inner, ly)
    fit = P.fit_box(sx, sy, x, y, w_max, h_max)
    return P.Box(fit.x, fit.y, fit.w, 0.5 * (lx - sx), 0.5 * (ly - sy), fit.span, h=fit.h)


def draw(facts: Mapping[str, Any], run_facts: Any, out: Path, *, qf: Any = None, zf: np.ndarray | None = None,
         map_reason: str = "", map_time_words: str = "", inner: float | None = None) -> dict[str, Any]:
    """Draw the card to ``out``.  ``qf`` is the frame of the map and the levels (or None); ``zf`` the w-face heights
    when there is no frame; ``inner`` (m) draws the map over that wide a square around the area's middle."""

    from PIL import Image, ImageDraw

    W, H = P.W, P.H
    cv = P.canvas()
    ov = S.Overlay((W, H))
    versions = facts.get("versions") or []
    gm1 = ("GM1 " + " and ".join(versions)) if versions else ""
    room = ov.x1 - ov.x0 - ov.U(260)
    records = "receipt" if facts.get("receipts") else "saved frames"
    records = f"{records} and {facts['settings_source']}" if facts.get("settings_source") else records
    for parts in ((facts.get("name", ""), facts.get("case") or "", gm1, f"from the run's {records}"),
                  (facts.get("name", ""), facts.get("case") or "", gm1), (facts.get("case") or "", gm1),
                  (facts.get("name", ""), gm1)):
        data = S.data_line(*parts)
        if ov.measure(data, "data")[0] <= room:
            break
    ov.title(TITLE, data)
    storm_s = facts.get("storm_end_s")
    if storm_s is None and qf is not None:
        storm_s = float(qf.t)
    if storm_s is not None:
        ov.clock(run_facts.clock(storm_s) if hasattr(run_facts, "clock") else S.fmt_clock(storm_s))
    shown_time = float(storm_s or 0.0)
    head_y = ov.U(176)
    sub_y = head_y + ov.U(26)
    top, bottom = ov.U(222), ov.U(722)
    col_a, col_b, col_c = ov.x0, ov.U(712), ov.U(1236)
    # ---- 1 the area
    grid = facts.get("grid") or {}
    nx, ny = (qf.nx, qf.ny) if qf is not None else (grid.get("nx"), grid.get("ny"))
    dx, dy = (qf.dx, qf.dy) if qf is not None else (grid.get("dx_m"), grid.get("dy_m"))
    nz = qf.nz if qf is not None else (grid.get("nz") or (len(zf) - 1 if zf is not None else None))
    lx = (nx or 0) * (dx or 0.0)
    ly = (ny or 0) * (dy or 0.0)
    ov.text((col_a, head_y), "1  ·  the area", "label", colour="text", caps=True, style="bold")
    inner = inner if (inner and lx and ly and qf is not None and inner < max(lx, ly)) else None
    area = f"{round(lx / 1000, 1):g} × {round(ly / 1000, 1):g} km, seen from above" if lx else "the area"
    if inner:
        area += f"; the map shows its inner {S.fmt_km(inner)}"
    ov.text((col_a, sub_y), area,
            "micro", colour="muted")
    shown: dict[str, Any] = {}
    if qf is not None:
        shown["map_time_s"] = float(qf.t)
    if lx and ly:
        box = map_box(lx, ly, col_a + ov.U(280), top, ov.U(560), bottom - top - ov.U(16), inner)
        img, what = (None, map_reason or "no saved frame to draw the storm from")
        if qf is not None:
            img, what = _box_map(qf, facts, box)
        if img is not None:
            P.paste(cv, img, box)
            ov.text((box.x + ov.U(12), box.y + ov.U(22)), "column-maximum reflectivity, not a radar", "micro",
                    colour="data")
            if map_time_words:
                ov.text((box.x + ov.U(12), box.y + ov.U(42)), map_time_words, "micro", colour="data")
            P.dbz_legend(ov, box.x + box.w - ov.U(236), box.y + box.h + ov.U(12), ov.U(236))
            shown["map"] = what
            if inner:
                shown["map_inner_km"] = inner / 1000.0
        else:
            # the reason, wrapped to the box and centred in it (a scheme's own reason can be a long sentence)
            lines = ov.wrap(what, "micro", box.w - ov.U(24)) or [what]
            step = int(round(ov.px("micro") * 1.45))
            y0 = box.y + box.h / 2 - (len(lines) - 1) * step / 2
            for n, line in enumerate(lines):
                ov.text((box.x + box.w / 2, y0 + n * step), line, "micro", colour="muted", anchor="ms")
            shown["map"] = None
        cv = P.outline(cv, box)
        P.scale_bar(ov, box.x, box.y + box.h + ov.U(18), box)
    cells = (nx or 0) * (ny or 0) * (nz or 0)
    if nx and dx:
        ov.text((col_a, ov.U(784)), f"{nx} × {ny} columns, {S.fmt_grid(dx)} apart", "data", colour="title",
                style="bold")
    moving = facts.get("moving") or {}
    um, vm = moving.get("u"), moving.get("v")
    if um or vm:
        follow = (f"the domain moves {S.fmt_num(um or 0.0, 1)} m/s east and {S.fmt_num(vm or 0.0, 1)} m/s north, "
                  "ground-relative, with the storm")
    else:
        follow = "domain fixed to the ground"
    if cells:
        count = f"{cells / 1e6:,.1f} million" if cells >= 1e5 else f"{int(cells):,}"
        ov.text((col_a, ov.U(810)), f"{count} grid cells ({nx} × {ny} × {nz})", "micro", colour="data")
    ov.text((col_a, ov.U(830)), follow, "micro", colour="data")
    # ---- 2 the levels
    faces = np.asarray(qf.zf if qf is not None else (zf if zf is not None else []), np.float64)
    ov.text((col_b, head_y), "2  ·  the levels", "label", colour="text", caps=True, style="bold")
    if faces.size >= 2:
        nlev = faces.size - 1
        ov.text((col_b, sub_y), f"{nlev} layers from the ground to {S.fmt_height(float(faces[-1]))}", "micro",
                colour="muted")
        im = Image.fromarray(cv)
        d = ImageDraw.Draw(im)
        cx, cw = col_b + ov.U(250), ov.U(40)
        zx = col_b + ov.U(370)
        span = bottom - top
        zmax = float(faces[-1])
        zoom = 1000.0 if float(faces[1]) < 600.0 else 2000.0
        dz = np.diff(faces)
        lo, hi = float(dz.min()), float(dz.max())

        def tone(k: int) -> tuple[int, int, int]:
            t = 1.0 if hi - lo < 1e-6 else 1.0 - (dz[min(k, dz.size - 1)] - lo) / (hi - lo)
            a = np.array(S.hex_rgb("#5b6477")) * (1 - t) + np.array(S.hex_rgb("#ffd166")) * t
            return tuple(int(round(255 * c)) for c in a)

        d.rectangle((cx, top, cx + cw, bottom), outline=S.rgb8("faint"))
        for k, z in enumerate(faces):
            y = bottom - span * z / zmax
            d.line((cx, y, cx + cw, y), fill=tone(k), width=1)
        d.rectangle((zx, top, zx + cw, bottom), outline=S.rgb8("faint"))
        for k, z in enumerate(faces):
            if z <= zoom + 1e-6:
                y = bottom - span * z / zoom
                d.line((zx, y, zx + cw, y), fill=tone(k), width=2)
        zh = 0.5 * (faces[1:] + faces[:-1])
        for z in zh:
            if z < zoom:
                y = bottom - span * z / zoom
                d.ellipse((zx + cw / 2 - 2, y - 2, zx + cw / 2 + 2, y + 2), fill=S.rgb8("dim"))
        y0 = bottom - span * float(zh[0]) / zoom
        d.ellipse((zx + cw / 2 - 5, y0 - 5, zx + cw / 2 + 5, y0 + 5), fill=S.rgb8("gold"))
        yz = bottom - span * zoom / zmax
        d.rectangle((cx - 2, yz, cx + cw + 2, bottom), outline=S.rgb8("gold"), width=1)
        for (xa, ya), (xb, yb) in (((cx + cw + 2, yz), (zx, top)), ((cx + cw + 2, bottom), (zx, bottom))):
            n = 18
            for s in range(0, n, 2):
                d.line((xa + (xb - xa) * s / n, ya + (yb - ya) * s / n, xa + (xb - xa) * (s + 1) / n,
                        ya + (yb - ya) * (s + 1) / n), fill=S.rgb8("gold"), width=1)
        cv = np.asarray(im).copy()
        for value in np.arange(0.0, zoom + 1.0, zoom / 5.0):
            y = bottom - span * value / zoom
            ov.text((zx + cw + ov.U(8), y + ov.U(5)), S.fmt_height(value) if value else "0 m", "micro",
                    colour="muted", halo=False)
        notes = [(zmax, f"model top {S.fmt_height(zmax)}")]
        depth = _num(facts.get("damping_depth_m"))
        if depth and facts.get("irdamp") not in (0, False):
            notes.append((zmax - depth, f"damping from {S.fmt_height(zmax - depth)}"))
        vertical = facts.get("vertical") or {}
        if vertical.get("stretch_z") == 1 and _num(vertical.get("str_top_m")):
            notes.append((float(vertical["str_top_m"]), f"{S.fmt_height(float(vertical['dz_top_m']))} layers above "
                                                        f"{S.fmt_height(float(vertical['str_top_m']))}"))
            if _num(vertical.get("str_bot_m")) is not None:
                notes.append((0.5 * (float(vertical["str_bot_m"]) + float(vertical["str_top_m"])),
                              f"stretching {S.fmt_height(float(vertical['dz_bot_m']))} → "
                              f"{S.fmt_height(float(vertical['dz_top_m']))}"))
        notes.append((zoom, f"lowest {S.fmt_height(zoom)}, enlarged →"))
        placed: list[float] = []
        # from the ground up, each note at its height or moved up clear of the one below it
        for z, text in sorted(notes, key=lambda row: row[0]):
            y = bottom - span * z / zmax
            while any(abs(y - p) < ov.U(22) for p in placed):
                y -= ov.U(22)
            placed.append(y)
            ov.text((cx - ov.U(10), y + ov.U(5)), text, "micro", colour="data", anchor="rs")
        ov.text((zx + cw / 2, bottom + ov.U(26)), f"gold dot: the lowest level, {S.fmt_height(float(zh[0]))}",
                "micro", colour="gold", anchor="ms")
        big, small = level_words(facts, faces)
        ov.text((col_b, ov.U(784)), big, "data", colour="title", style="bold")
        ov.text((col_b, ov.U(810)), small, "micro", colour="data")
        shown.update({"levels": nlev, "lowest_level_m": float(zh[0]), "model_top_m": zmax})
    else:
        ov.text((col_b, sub_y), "the levels are not recorded", "micro", colour="muted")
    # ---- 3 the run
    gpus = facts.get("gpus") or []
    ov.text((col_c, head_y), "3  ·  the run", "label", colour="text", caps=True, style="bold")
    process = facts.get("process_s")
    if process is not None and gpus:
        ov.text((col_c, sub_y), f"on one {' and one '.join(gpus)}, measured", "micro", colour="muted")
    elif gpus:
        ov.text((col_c, sub_y), f"on one {' and one '.join(gpus)}", "micro", colour="muted")
    right = ov.x1
    width = right - col_c
    storm = facts.get("storm_s")
    if storm is None and storm_s is not None:
        storm = storm_s
    y = top + ov.U(52)
    if storm is not None:
        box = ov.text((col_c, y), duration(storm), "hero_number", colour="title")
        ov.text((box[2] + ov.U(14), y), "of storm", "data", colour="muted")
    if process is not None and storm:
        y += ov.U(78)
        box = ov.text((col_c, y), duration(process), "hero_number", colour="gold")
        ov.text((box[2] + ov.U(14), y), "of computing", "data", colour="muted")
        biggest = max(storm, process)
        im = Image.fromarray(cv)
        d = ImageDraw.Draw(im)
        for n, (value, colour) in enumerate(((storm, "#2b3650"), (process, "#ffd166"))):
            by = y + ov.U(30) + n * ov.U(34)
            d.rectangle((col_c, by, right, by + ov.U(24)), fill=S.rgb8("panel"))
            d.rectangle((col_c, by, col_c + max(2, int(width * value / biggest)), by + ov.U(24)), fill=S.rgb8(colour))
        cv = np.asarray(im).copy()
        for n, (value, label) in enumerate(((storm, "storm time"), (process, "computing time"))):
            by = y + ov.U(30) + n * ov.U(34)
            inside = width * value / biggest > ov.U(170)
            ov.text((col_c + ov.U(10) if inside else col_c + int(width * value / biggest) + ov.U(10),
                     by + ov.U(17)), label, "micro", colour="void" if (inside and n == 1) else "text", halo=False)
        ratio = storm / process
        y += ov.U(130)
        if 0.95 <= ratio < 1.05:
            speed = "about as fast as real time"
        elif ratio >= 1.0:
            speed = f"{ratio:,.0f}× faster than real time" if ratio >= 10 else f"{ratio:.1f}× faster than real time"
        else:
            speed = f"{1 / ratio:.1f}× slower than real time"
        ov.text((right, y), speed, "data", colour="gold", anchor="rs", style="bold")
        shown["speed"] = ratio
        stepping, integrate = facts.get("stepping_s"), facts.get("integrate_s")
        if stepping is not None and integrate is not None:
            parts = [stepping, max(0.0, integrate - stepping), max(0.0, process - integrate)]
            y += ov.U(46)
            ov.text((col_c, y), "where the computing time went", "micro", colour="muted", caps=True, tracking=0.06)
            ov.text((right, y), "host clock", "micro", colour="dim", anchor="rs")
            im = Image.fromarray(cv)
            d = ImageDraw.Draw(im)
            x = float(col_c)
            for (colour, _w), value in zip(PARTS, parts):
                w = width * value / max(process, 1e-9)
                if w >= 1.0:
                    d.rectangle((int(x), y + ov.U(12), int(x + max(w - 2, 1)), y + ov.U(34)), fill=S.rgb8(colour))
                x += w
            cv = np.asarray(im).copy()
            lx2, ly2 = col_c, y + ov.U(62)
            for (colour, words), value in zip(PARTS, parts):
                amount = f"{value / 60:.1f} min" if value >= 60 else f"{value:.1f} s"
                label = f"{words}  {amount}"
                wlen = ov.measure(label, "micro")[0] + ov.U(40)
                if lx2 + wlen > right:
                    lx2, ly2 = col_c, ly2 + ov.U(24)
                ov.rule(lx2, ly2 - ov.U(10), ov.U(12), thickness=10, colour=colour)
                ov.text((lx2 + ov.U(20), ly2), label, "micro", colour="data")
                lx2 += wlen
            shown["time_split_s"] = parts
            y = ly2
        stats = []
        steps = facts.get("steps")
        if steps:
            how = (f"of {_seconds_words(facts['dt_s'])}" if facts.get("dt_s") and not facts.get("adaptive")
                   else "the step chosen as it ran")
            stats.append((f"{steps:,.0f} steps", how))
        if facts.get("step_mean_s"):
            stats.append((f"{facts['step_mean_s']:.3f} s", "computing per step, the mean"))
        frames = facts.get("frames")
        if frames:
            every = f", one every {duration(facts['interval_s'])}" if facts.get("interval_s") else ""
            stats.append((f"{frames:,.0f} saved times", f"of the whole 3-D state{every}"))
        restarts = facts.get("restarts")
        if restarts is not None:
            stats.append((f"{restarts:,.0f} restart{'s' if restarts != 1 else ''}", "saved to go on from"))
        peak, card, model = (facts.get("memory_peak_bytes"), facts.get("memory_card_bytes"),
                             facts.get("memory_model_bytes"))
        if peak and card:
            stats.append((f"{peak / 2 ** 30:.1f} of {card / 2 ** 30:.1f} GiB", "GPU memory at its peak, whole card"))
        if model:
            stats.append((f"{model / 2 ** 30:.2f} GiB", "of it the model's own arrays"))
        sy = y + ov.U(52)
        for n, (big, small) in enumerate(stats[:6]):
            c, r = n % 2, n // 2
            xx, yy = col_c + c * (width // 2), sy + r * ov.U(62)
            ov.text((xx, yy), big, "data", colour="title", style="bold")
            ov.text((xx, yy + ov.U(24)), small, "micro", colour="muted")
    else:
        y += ov.U(70)
        ov.text((col_c, y), "the computing time and the GPU's memory", "label", colour="data")
        ov.text((col_c, y + ov.U(28)), "arrive in the receipt when this part of the run ends", "label",
                colour="data")
    # ---- the physics
    im = Image.fromarray(cv)
    ImageDraw.Draw(im).line((ov.x0, ov.U(842), ov.x1, ov.U(842)), fill=S.rgb8("rule"), width=1)
    cv = np.asarray(im).copy()
    tiles = physics_tiles(facts, run_facts.scheme, run_facts.checked)
    dropped: list[str] = []
    tw = (ov.x1 - ov.x0) // len(tiles)
    for n, (head, lines) in enumerate(tiles):
        xx = ov.x0 + n * tw
        ov.text((xx, ov.U(864)), head, "micro", colour="gold", caps=True, tracking=0.08, style="condensed_bold")
        rows = [(piece, line.startswith(FRAME_FREE)) for line in lines for piece in _fit(ov, line, tw - ov.U(18))]
        for m, (line, free) in enumerate(rows[:5]):
            ov.text((xx, ov.U(886) + m * ov.U(19)), line, "micro", colour="data", vertical=free)
        dropped += [line for line, _free in rows[5:]]
    # ---- the footer
    parts_words = (f"{facts['receipts']} receipt{'s' if facts['receipts'] != 1 else ''}" if facts.get("receipts")
                   else "the newest frame's record (no receipt yet)")
    settings_words = f", the {facts['settings_source']}" if facts.get("settings_source") else ""
    notes = S.data_line(f"numbers from {parts_words}{settings_words} and run.json",
                        "time split by the host clock, which waits for the GPU before each save",
                        "no split by physics: the GPU's work runs asynchronously")
    ov.footer(run_facts.footer(t=storm_s or 0.0, level="the whole run", wind="", notes=notes))
    ov.mark()
    if dropped:
        import warnings

        warnings.warn(f"run card: tile lines left out: {dropped}")
    P.save(ov, cv, out)
    shown["time_s"] = shown_time
    return shown


def _fit(ov: S.Overlay, text: str, width: int) -> list[str]:
    """``text`` wrapped to lines of at most ``width`` pixels of micro type."""

    return ov.wrap(text, "micro", width) or [text]


# ---------------------------------------------------------------- the frame of the map
def map_frame(rows: Mapping[str, Mapping[str, Any]], hurricane: bool) -> str | None:
    """The time key of the frame the map shows: the strongest echo a frame's statistics recorded (``dbz_max``), or a
    hurricane's newest saved time (its storm grows to the end)."""

    times = sorted(((float(t), row) for t, row in rows.items()), key=lambda kv: kv[0])
    if not times:
        return None
    if not hurricane:
        best = [(row.get("stats") or {}).get("dbz_max") for _t, row in times]
        if any(isinstance(b, (int, float)) for b in best):
            n = max(range(len(times)), key=lambda i: best[i] if isinstance(best[i], (int, float)) else -1e9)
            return repr(times[n][0])
    return repr(times[-1][0])


def signature(segments: Sequence[Path], extra: str = "") -> str:
    """What the card is drawn from changes when a segment writes its receipt or the run's records change."""

    import hashlib

    rows = []
    for segment in segments:
        path = Path(segment) / "receipt.json"
        try:
            st = path.stat()
            rows.append(f"{path}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            rows.append(f"{path}:none")
    rows.append(extra)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def summary(facts: Mapping[str, Any]) -> dict[str, Any]:
    """The numbers the card showed, for the index."""

    keep = ("receipts", "versions", "gpus", "process_s", "integrate_s", "stepping_s", "storm_s", "steps", "frames",
            "restarts", "step_mean_s", "memory_peak_bytes", "memory_model_bytes", "memory_card_bytes", "dt_s",
            "adaptive", "substeps", "arwen", "microphysics", "config_source")
    return {k: facts.get(k) for k in keep if facts.get(k) is not None}


# ---------------------------------------------------------------- the worker's once-per-run look
def levels_without_frame(facts: Mapping[str, Any]) -> np.ndarray | None:
    """The w-face heights from the receipt's level words (``vertical_coordinate``), through CM1's own level
    builder, for a run whose frames cannot be read; None when the words do not say."""

    vertical = facts.get("vertical") or {}
    nk = vertical.get("nk")
    if not isinstance(nk, int) or nk < 1 or vertical.get("stretch_z") not in (0, 1):
        return None
    try:
        from ..cm1_vertical_grid import build_vertical_levels

        levels = build_vertical_levels(
            stretch_z=int(vertical["stretch_z"]), nk=nk,
            dz_m=float(vertical.get("dz_reference_m") or (facts.get("grid") or {}).get("dz_m") or 0.0),
            ztop_m=float(vertical.get("ztop_m") or 0.0), str_bot_m=float(vertical.get("str_bot_m") or 0.0),
            str_top_m=float(vertical.get("str_top_m") or 0.0), dz_bot_m=float(vertical.get("dz_bot_m") or 0.0),
            dz_top_m=float(vertical.get("dz_top_m") or 0.0))
        return np.asarray(levels.face_heights_m, np.float64)
    except (ValueError, TypeError, KeyError, ImportError):
        return None


#: Why netCDF files get no card: it is drawn from a GM1 run's own records (its receipts and run.json say the card,
#: the GPU, the computing time and the steps), which CM1's own output never has and a GM1 run's cm1out files leave
#: behind in its folder.
NETCDF_WHY = ("How this storm was simulated is drawn from a GM1 run's own records (its receipts and run.json), "
              "which netCDF files do not carry: for a GM1 run, draw its run folder.")


def draw_once(renderer: Any, lineage: Sequence[tuple[float, str, Path, int]]) -> bool:
    """The worker's pass: draw the card when what it is drawn from has changed (a segment wrote its receipt, or the
    run ended).  Returns whether the card was drawn."""

    import time as _time

    from .facts import run_facts
    from .frame import FrameUnreadable, QuickFrame, same_frame

    layout = renderer.layout
    once = renderer.state["once"]
    if layout.kind == "netcdf":
        # CM1's own output, or a GM1 run's cm1out files: no receipt to say how it was run (a card would say a GPU
        # ran CM1's own storm)
        if (once.get("run-card") or {}).get("not_drawn") == NETCDF_WHY:
            return False
        once["run-card"] = {"signature": "netcdf", "not_drawn": NETCDF_WHY}
        return True
    segments = [Path(segment.frames_root).parent for segment in layout.segments]
    root = Path(layout.root if layout.kind != "frames" else layout.root.parent)
    ended, _record = layout.ended()
    sig = signature(segments, extra=f"ended={bool(ended)}|frames={'yes' if lineage else 'no'}")
    if (once.get("run-card") or {}).get("signature") == sig:
        return False
    begun = _time.perf_counter()
    rows = renderer.state.get("frames") or {}
    hurricane = any((row.get("facts") or {}).get("storm_kind") == "hurricane" for row in rows.values())
    chosen = map_frame(rows, hurricane)
    chosen_row = (rows.get(chosen) or {}) if chosen else {}
    key = chosen_row.get("frame")
    # (its frame named by its key, or by its folder alone in an older record)
    row = next((r for r in lineage if same_frame(key, r[1], chosen_row.get("segment"), r[3])), None)
    if row is not None:
        key = row[1]
    elif lineage:
        row = lineage[-1]
    file = renderer.out / "run-card" / "card.png"
    qf = None
    reason = ""
    try:
        if row is not None:
            try:
                qf = QuickFrame.open(row[2], key=row[1], segment=row[3])
            except (FrameUnreadable, OSError, ValueError, KeyError) as error:
                qf, reason = None, f"no map: the saved frame cannot be read ({type(error).__name__})"
        config = qf.config if qf is not None else {}
        facts = gather(root, segments, settings=renderer.settings, frame_config=config, frames_saved=len(lineage))
        if facts.get("storm_end_s") is None and lineage:
            facts["storm_end_s"] = float(lineage[-1][0])        # no receipt yet: the newest saved time
        grid = facts.get("grid") or {}
        dx = qf.dx if qf is not None else float(grid.get("dx_m") or 0.0)
        dy = qf.dy if qf is not None else float(grid.get("dy_m") or 0.0)
        if not config:
            config = {"microphysics": facts.get("microphysics"), "dense_ice": {"category": facts.get("dense_ice")}}
        facts_run = run_facts(layout, config, dx, dy, label=renderer.label, settings=renderer.settings)
        when = ""
        if qf is not None:
            clock = facts_run.clock(qf.t)
            when = (f"at {clock}, the newest saved time" if hurricane or not chosen or key != row[1]
                    or chosen == repr(lineage[-1][0]) else f"at {clock}, the saved time with the strongest echo")
        inner = None
        if hurricane and qf is not None:                    # a hurricane's map: the inner part, as its looks show it
            inner = min(P.HURRICANE_INNER, 0.5 * min(qf.nx * qf.dx, qf.ny * qf.dy))
        shown = draw(facts, facts_run, file, qf=qf, zf=None if qf is not None else levels_without_frame(facts),
                     map_reason=reason, map_time_words=when, inner=inner)
        record = {"signature": sig, "time": shown.pop("time_s"),
                  "file": "run-card/card.png", "thumb": "run-card/thumb/card.png",
                  "facts": {**summary(facts), **shown, "map_frame": row[1] if (row and qf is not None) else None},
                  "seconds": round(_time.perf_counter() - begun, 3)}
    except (P.LintError, ValueError, KeyError, TypeError) as error:
        renderer.summary.problems.append(f"run-card: {error}")
        record = {"signature": sig, "time": 0.0, "error": f"{type(error).__name__}: {error}"}
    finally:
        if qf is not None:
            qf.close()
    once["run-card"] = record
    return True


__all__ = ["NETCDF_WHY", "PARTS", "TITLE", "draw", "draw_once", "duration", "gather", "levels_without_frame", "map_frame",
           "physics_tiles", "signature", "summary"]
