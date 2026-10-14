"""Use my own sounding: one file looked at from end to end before a storm is run on it.

:func:`examine` is what ``gpuwm-cm1 sounding convert`` and guided mode's "Use my own sounding" card show: the
file read in whichever layout it has (:mod:`.sounding_formats`), what is wrong with it in plain words with the line
of the user's own file, the CM1 ``input_sounding`` it becomes, CM1 getcape's numbers for it, one hedged sentence
on the storm such air usually makes, and how to start the storm.

**The numbers** are the release's own: CM1's ``getcape.F`` parcels and GM1's bulk shear, Bunkers motion and
storm-relative helicity, from the initializer library's sounding report (``sounding_report``) on the model's own
levels.  Without the library the numbers are left out and the record says why; nothing else waits on them.

**The sentence** (:func:`verdict`) is Weisman and Klemp's regimes as the Make-your-own-environment card states
them (``analytic_sounding.regime``: CAPE below 500 J/kg, 0-6 km shear below 10 and below 20 m/s), with one more
case for a strong lid: CIN of 100 J/kg or more.  It is a rule of thumb, never a forecast, and says so.

**The start** (:func:`start_choice`).  A sounding file can be started with CM1's warm bubble (``iinit = 1``) or
with CM1's updraft nudging (``iinit = 12``, a push of rising air for the first 20 minutes).  What GM1 has measured:
none of the nine balloon soundings it ships made a storm from the warm bubble on a grid a GPU fits, and CM1 r21.1
made none from it on the 20 May 2013 file either; all nine made one from the push; the made environments
(Weisman and Klemp's air) make theirs from the bubble.  CM1's bubble is centred 1.4 km above the ground and warms
nothing at the ground itself, so it lifts the air that is at 1.4 km.  In the nine balloons the water vapour there
is 43 to 77 % of the surface value; in the made environments it is 81 to 100 %.  The default follows that one
measured difference: below :data:`BUBBLE_MOIST_RATIO` the push, at or above it the bubble.  It is a line between
two groups of runs, not a theory, and the record gives the number so a close call can be seen.

A file written by :func:`write` has its record beside it (``NAME.sounding.json``, or ``sounding.json`` beside a
file named ``input_sounding``), which is how a later ``preset fit --sounding FILE`` knows the name the user's file
had and names the run from it (:func:`kept`).

Standard library only, plus the installed initializer library for the numbers and matplotlib for the picture.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from . import sounding_formats as F

SCHEMA = "gpuwm-cm1.own-sounding/v1"
RECORD_NAME = "sounding.json"
#: Water vapour at the bubble's centre over the surface's: below this the push of rising air is the default.
BUBBLE_MOIST_RATIO = 0.8
#: Within this much of the line the choice is a close call, and the record says so.
CLOSE_CALL = 0.05
#: CIN (J/kg) from which the sentence says the air is capped, and from which it mentions a lid.
CAPPED_J_KG = 100.0
LID_J_KG = 50.0
#: The longest run-name stem made from a file name.
LABEL_CHARS = 32

STARTS = ("push", "bubble", "strong-bubble")
START_TITLES = {
    "push": "A push of rising air",
    "bubble": "A warm bubble",
    "strong-bubble": "A stronger warm bubble",
}
START_LINES = {
    "push": ("Air is pushed gently upward in one spot for the first 20 minutes (CM1's updraft nudging), which lifts "
             "the air from the ground up. The famous-day storms start this way."),
    "bubble": ("A bubble of air 1 K warmer than the air around it, 10 km in radius and centred 1.4 km above the "
               "ground (CM1's classic start). It works when the air at 1.4 km is nearly as moist as at the ground."),
    "strong-bubble": ("The same bubble made 3 K warmer. It lifts harder, but it still starts with the air at "
                      "1.4 km, so in dry air aloft it can still make no storm."),
}
TRAP = ("The known trap: on a real balloon sounding a warm bubble often makes no storm. CM1's bubble warms nothing at "
        "the ground and most at 1.4 km up, and in real air the moisture usually falls off quickly above the ground, "
        "so the bubble lifts drier air than the surface air the CAPE number is counted from. None of GM1's nine "
        "shipped balloon soundings made a storm from the bubble on a grid a GPU fits; all nine did from the push.")
HEDGE = ("A rule of thumb from studies of idealized storms (Weisman and Klemp), not a forecast: the same air can "
         "make another storm, or none.")
VERDICTS = {
    "stable": "Stable air: a storm is unlikely.",
    "weak": "Little instability: at most weak, short-lived showers are likely.",
    "capped": ("Capped: there is energy for a storm, but a stable layer near the ground holds it down, so it needs "
               "a strong push."),
    "single": "Unstable air with little wind shear: short-lived single storms that rain themselves out are likely.",
    "multicell": "Enough instability and some wind shear: clusters of storms (multicells) are likely.",
    "supercell": "Plenty of instability and shear for a rotating storm (a supercell).",
}


# ------------------------------------------------------------------ names


def label_from(name: str | None) -> str:
    """A run-name stem from a file name: letters, digits, '.', '_' and '-', starting with a letter or digit."""

    stem = Path(str(name or "").replace("\\", "/")).name
    for suffix in (".input_sounding", ".txt", ".csv", ".snd", ".dat", ".tsv", ".html", ".htm", ".sounding", ".text"):
        if stem.lower().endswith(suffix) and len(stem) > len(suffix):
            stem = stem[:-len(suffix)]
            break
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem)
    stem = re.sub(r"-{2,}", "-", stem).strip("-._")[:LABEL_CHARS].strip("-._")
    return stem or "sounding"


def record_path(out: Path | str) -> Path:
    """Where the record of a sounding written at ``out`` goes."""

    out = Path(out)
    return out.parent / RECORD_NAME if out.name == "input_sounding" else out.with_name(out.name + ".sounding.json")


def kept(sounding: Path | str) -> dict[str, Any] | None:
    """The record beside a sounding :func:`write` wrote, if the file is still the one it describes."""

    sounding = Path(sounding)
    try:
        record = json.loads(record_path(sounding).read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA:
            return None
        if hashlib.sha256(sounding.read_bytes()).hexdigest() != record["sha256"]:
            return None  # the file was changed after the record was written
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return record


def source_words(record: Mapping[str, Any]) -> str:
    """Where the air came from, for the budget screen and fit.json: the file's name and what was done to it."""

    name = str(record.get("name") or "a sounding file")
    words = f"your sounding {name}"
    if record.get("shipped"):
        words = f"the sounding GM1 ships, {record['shipped']}"
    if record.get("converted"):
        words += f", converted from {record.get('format_words')}"
    elif record.get("tidied"):
        words += ", with what the engine does not read (blank lines, commas, extra words) taken out"
    extension = record.get("extension") or {}
    if extension.get("extended"):
        words += (f", with made-up levels added from {extension['from_m'] / 1000.0:.1f} to "
                  f"{extension['to_m'] / 1000.0:.1f} km")
    return words


def results_words(sounding: Mapping[str, Any] | None) -> str | None:
    """What Results says of a run on the user's own sounding (None for any other run): whose air, and the files."""

    if not isinstance(sounding, Mapping) or not sounding.get("label"):
        return None
    own = sounding.get("own") or {}
    words = f"The air is {source_words(own) if own else 'your sounding ' + str(sounding['label'])}."
    words += " The file the model read is input_sounding in this run's folder"
    if own.get("converted") or own.get("tidied"):
        words += ", and your own file is beside it as input_sounding.original"
    return words + "."


# ------------------------------------------------------------------ numbers and words


def numbers(text: str, *, nz: int | None = None, dz_m: float | None = None, top_m: float | None = None,
            library: str | Path | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """(the numbers a page shows, the full report, why not): CM1 getcape and GM1's shear and helicity.

    On the model's levels when ``nz`` and ``dz_m`` are given (the storm the file will run in), else on 250 m levels
    up to the file's top."""

    from ..errors import CM1Refusal, fail_fast

    if nz is None or dz_m is None:
        dz_m = 250.0
        nz = max(2, min(160, int((top_m or 20000.0) / dz_m - 0.5)))
    try:
        from ..cartesian_door import resolve_initializer_library
        from ..sounding_report import initial_sounding_report

        with fail_fast():
            path = resolve_initializer_library(None if library is None else Path(library))
            report = initial_sounding_report(path, text.encode("ascii"), isnd=7, iwnd=0, nz=int(nz),
                                             dz_m=float(dz_m))
    except CM1Refusal as refusal:
        return None, None, plain_reason(refusal.detail.reason)
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        return None, None, plain_reason(str(error))
    sb = report.get("surface_based") or {}
    mu = report.get("most_unstable") or {}
    ml = report.get("mixed_layer") or {}
    shown = {
        "cape_j_kg": sb.get("cape_j_per_kg"), "cin_j_kg": sb.get("cin_j_per_kg"),
        "lcl_m": _height(sb.get("lcl_m_agl")), "lfc_m": _height(sb.get("lfc_m_agl")),
        "el_m": _height(sb.get("el_m_agl")),
        "cape_most_unstable_j_kg": mu.get("cape_j_per_kg"), "cape_mixed_layer_j_kg": ml.get("cape_j_per_kg"),
        "cin_mixed_layer_j_kg": ml.get("cin_j_per_kg"),
        "shear_0_1km_m_s": report.get("shear_0_1km_m_per_s"), "shear_0_6km_m_s": report.get("shear_0_6km_m_per_s"),
        "srh_0_1km_m2_s2": report.get("srh_0_1km_m2_per_s2"), "srh_0_3km_m2_s2": report.get("srh_0_3km_m2_per_s2"),
        "storm_motion_m_s": [report.get("bunkers_right_u_m_per_s"), report.get("bunkers_right_v_m_per_s")],
        "levels": {"nz": int(nz), "dz_m": float(dz_m)},
        "basis": ("CM1 getcape (surface parcel) and GM1's bulk shear, Bunkers right-mover motion and "
                  f"storm-relative helicity, on {int(nz)} model levels of {float(dz_m):g} m"),
    }
    cape = float(shown.get("cape_j_kg") or 0.0)
    if shown["lfc_m"] is not None and shown["el_m"] is None and cape > 0.0:
        # getcape found where rising air starts but not where it stops: it was still rising at the top.
        shown["el_note"] = "Surface air is still rising at the top of the levels these numbers are worked out on."
    elif shown["el_m"] is not None and shown["el_m"] < LOW_EL_M and cape > LOW_EL_CAPE:
        # getcape's EL is where surface air first stops rising; with this much CAPE it rises again higher up.
        shown["el_note"] = (f"Surface air first stops rising near {float(shown['el_m']) / 1000.0:.1f} km, then "
                            "rises again higher up, so no single height is shown for where it stops.")
        shown["el_m"] = None
    full = dict(report)
    full["grid"] = {"nz": int(nz), "dz_m": float(dz_m)}
    return shown, full, None


#: An EL below this height with more CAPE than this is where surface air first stops, not its top.
LOW_EL_M = 3000.0
LOW_EL_CAPE = 500.0


def _height(value: Any) -> float | None:
    """A height from getcape, or None for its -1 (no such level)."""

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0.0 else None


def plain_reason(reason: str) -> str:
    """Why the numbers were not worked out, in plain words (the full reason stays in the record's report)."""

    levels = re.search(r"own (\d+) mass levels", reason or "")
    if levels:
        return (f"too few levels: on the model's levels your file covers only {levels.group(1)}, and GM1 needs "
                "at least four to work out the numbers.")
    if "initializer library" in (reason or "") or "cannot be loaded" in (reason or ""):
        return "this GM1 has no initializer library to work them out with (gpuwm-cm1 doctor says why)."
    return "GM1 could not build the model's levels from this file, so it could not work them out."


def _partial(shown: dict[str, Any], top_m: float) -> dict[str, Any]:
    """The numbers of a file that stops short: those that need air above its top are left out."""

    out = dict(shown)
    for key, needs_m in (("shear_0_6km_m_s", 6000.0), ("srh_0_3km_m2_s2", 3000.0), ("srh_0_1km_m2_s2", 1000.0),
                         ("shear_0_1km_m_s", 1000.0)):
        if top_m < needs_m:
            out[key] = None
    if top_m < 6000.0:
        out["storm_motion_m_s"] = [None, None]
    if out.get("el_m") is None and str(out.get("el_note") or "").startswith("Surface air is still rising"):
        out["el_note"] = f"Surface air is still rising where your file stops, at {top_m / 1000.0:.1f} km."
    out["partial_top_m"] = top_m
    return out


def verdict(shown: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """One plain, hedged sentence on the storm such air usually makes, from the numbers; None without them."""

    if not shown or shown.get("cape_j_kg") is None or shown.get("shear_0_6km_m_s") is None:
        return None
    from .analytic_sounding import regime

    surface = float(shown["cape_j_kg"])
    best = max(surface, float(shown.get("cape_most_unstable_j_kg") or 0.0))
    shear = float(shown["shear_0_6km_m_s"])
    cin = abs(float(shown.get("cin_j_kg") or 0.0))
    if best < 100.0:
        kind = "stable"
    elif regime(best, shear) == "stable":
        kind = "weak"
    elif cin >= CAPPED_J_KG:
        kind = "capped"
    else:
        kind = regime(best, shear)
    sentence = VERDICTS[kind]
    extra = []
    if kind in ("single", "multicell", "supercell") and cin >= LID_J_KG:
        extra.append(f"A stable layer near the ground (CIN {cin:.0f} J/kg) may hold the storm back at first.")
    if best > surface + 100.0:
        extra.append(f"The most unstable air is above the ground ({best:.0f} J/kg against {surface:.0f} J/kg for "
                     "surface air).")
    return {"id": kind, "sentence": sentence, "more": extra, "note": HEDGE,
            "basis": {"cape_j_kg": best, "shear_0_6km_m_s": shear, "cin_j_kg": cin}}


def _g(value: float) -> str:
    """A mixing ratio for a sentence: one decimal, or two significant digits when it is small."""

    return f"{value:.1f}" if abs(value) >= 0.95 else f"{value:.2g}"


def start_choice(profile: F.Profile) -> dict[str, Any]:
    """How to start a storm on this sounding: the choices, and the default chosen from its moisture."""

    moisture = F.bubble_moisture(profile)
    if moisture is None:
        recommended = "push"
        why = ("The file does not reach 1.4 km with moisture GM1 can compare, so the push of rising air, which "
               "lifts the air from the ground up, is the safer start.")
        close = False
    else:
        percent = round(100.0 * moisture["ratio"])
        close = abs(moisture["ratio"] - BUBBLE_MOIST_RATIO) < CLOSE_CALL
        if moisture["ratio"] < BUBBLE_MOIST_RATIO:
            recommended = "push"
            why = (f"At 1.4 km above the ground, where a warm bubble is centred, your air holds {percent} % of the "
                   f"water vapour it holds at the ground ({_g(moisture['aloft_g_kg'])} against "
                   f"{_g(moisture['surface_g_kg'])} g/kg). That is like the balloon soundings GM1 ships (43 to "
                   "77 %), which made no storm from a warm bubble and did from the push. So the push of rising "
                   "air is chosen.")
        else:
            recommended = "bubble"
            why = (f"At 1.4 km above the ground, where a warm bubble is centred, your air holds {percent} % of the "
                   f"water vapour it holds at the ground ({_g(moisture['aloft_g_kg'])} against "
                   f"{_g(moisture['surface_g_kg'])} g/kg): a moist layer that deep is like the made environments "
                   "(81 to 100 %), whose storms start from a warm bubble. So the warm bubble is chosen.")
        if close:
            why += " It is a close call (the line is at 80 %): if no storm forms, try the other start."
    return {
        "recommended": recommended, "why": why, "close_call": close, "bubble_moisture": moisture,
        "line_ratio": BUBBLE_MOIST_RATIO, "trap": TRAP,
        "options": [{"id": start, "title": START_TITLES[start], "line": START_LINES[start],
                     "recommended": start == recommended} for start in STARTS],
        "basis": ("measured on GM1's own runs: nine shipped balloon soundings (no storm from CM1's warm bubble, a "
                  "storm from CM1's updraft nudging) and the made environments (a storm from the bubble)"),
    }


# ------------------------------------------------------------------ the whole look


def examine(text: str, *, name: str | None = None, speed_unit: str | None = None, extend: bool = False,
            nz: int | None = None, dz_m: float | None = None, top_needed_m: float | None = None,
            library: str | Path | None = None, shipped: str | None = None) -> dict[str, Any]:
    """Everything about one sounding file, as a record (``gpuwm-cm1.own-sounding/v1``).

    ``ok`` is whether a storm can be run on it as it stands (no problem left).  ``text`` is the CM1
    ``input_sounding`` whenever the file could be read at all, so it can be shown and saved even with a problem
    still to fix.  ``top_needed_m`` is the model's highest level (with ``nz`` and ``dz_m``, the storm's own levels,
    which the numbers are then computed on); ``extend`` adds the made-up levels a short sounding needs.
    """

    record: dict[str, Any] = {
        "schema": SCHEMA, "ok": False, "name": Path(str(name or "sounding")).name, "label": label_from(name),
        "shipped": shipped, "format": None, "format_words": None, "converted": False, "conversions": [],
        "dropped": [], "notes": [], "problems": [], "warnings": [], "extension": {"extended": False},
        "accepted": F.ACCEPTED,
    }
    try:
        reading = F.read(text, speed_unit=speed_unit)
    except F.SoundingRefused as refused:
        record["problems"] = [refused.as_dict()]
        return record
    profile = reading.profile
    record.update({
        "format": reading.format, "format_words": reading.format_words, "converted": reading.converted,
        "conversions": list(reading.conversions), "dropped": [dict(row) for row in reading.dropped],
        "notes": list(reading.notes), "station_m": reading.station_m,
    })
    problems, warnings = F.check(profile, top_needed_m=top_needed_m)
    out_text = reading.text
    record["warnings_from_reading"] = [{"code": "ground", "text": text} for text in reading.warnings]
    if reading.format == "cm1" and not F.strict_cm1(reading.text):
        # CM1's Fortran would read it, but the engine reads only the strict form: the same numbers, written again.
        out_text, taken = F.tidy_cm1(reading.text)
        record["tidied"] = True
        if taken:
            record["notes"].append("Your file was written again in the strict form the engine reads, with the "
                                   "same numbers: " + "; ".join(taken) + " were taken out.")
    top = next((finding for finding in problems if finding.code == "top"), None)
    if extend and top is not None and top.offer.get("extend_to_m") and len(problems) == 1:
        profile, extension = F.extend(profile, float(top.offer["extend_to_m"]))
        record["extension"] = extension
        out_text = F.render(profile)
        problems, warnings = F.check(profile, top_needed_m=top_needed_m)
    try:
        out_text.encode("ascii")
    except UnicodeError:
        out_text = F.render(profile)
    record.update({
        "levels": profile.levels, "top_m": profile.top_m,
        "first_level_m": profile.z_m[0] if profile.z_m else None,
        "surface": {"pressure_hpa": profile.surface_hpa, "theta_k": profile.surface_theta_k,
                    "qv_g_kg": profile.surface_qv_gkg},
        "model": {"top_needed_m": top_needed_m, "nz": nz, "dz_m": dz_m},
        "text": out_text, "sha256": hashlib.sha256(out_text.encode("ascii")).hexdigest(),
        "bytes": len(out_text.encode("ascii")),
    })
    blocking = [finding for finding in problems if finding.code != "top"]
    if not blocking:
        # The numbers of a file that only stops short are its own, on the levels it reaches: those that need air
        # above its top are left out, and no sentence is given on air the file does not have.
        short = top is not None and not record["extension"]["extended"]
        shown, full, why = numbers(out_text, nz=None if short else nz, dz_m=None if short else dz_m,
                                   top_m=profile.top_m, library=library)
        if shown and short:
            shown = _partial(shown, profile.top_m)
        record["numbers"] = shown
        record["report"] = full
        if why:
            record["numbers_reason"] = why
        record["verdict"] = None if short else verdict(shown)
        record["start"] = start_choice(profile)
    record["problems"] = [finding.as_dict() for finding in problems]
    record["notes"] += [finding.text for finding in warnings if finding.level == "note"]
    record["warnings"] = [*record.pop("warnings_from_reading"),
                          *(finding.as_dict() for finding in warnings if finding.level != "note")]
    record["ok"] = not problems
    return record


def write(record: Mapping[str, Any], out: Path | str, *, command: str | None = None) -> Path:
    """The record's CM1 text at ``out`` and the record beside it (never over a file that differs)."""

    out = Path(out)
    data = str(record["text"]).encode("ascii")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.partial")
    tmp.write_bytes(data)
    os.replace(tmp, out)
    kept_record = {key: value for key, value in record.items() if key not in ("text", "report")}
    kept_record["file"] = out.name
    kept_record["command"] = command
    target = record_path(out)
    tmp = target.with_name(f".{target.name}.partial")
    tmp.write_text(json.dumps(kept_record, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def preview_png(record: Mapping[str, Any], out: Path | str) -> dict[str, Any]:
    """The sounding's skew-T, wind barbs and hodograph with its numbers (the column route's picture code)."""

    from . import wrfcolumn as wc
    from . import wrfpictures

    profile = F.parse_cm1(str(record["text"]))
    surface = {"pressure_hpa": profile.surface_hpa, "theta_k": profile.surface_theta_k,
               "qv_gkg": profile.surface_qv_gkg}
    column = wc.WrfColumn(
        source={"name": record.get("name"), "producer": "sounding file", "domain_id": 0, "times": ""},
        point={"i": 1, "j": 1}, z_m=list(profile.z_m), theta_k=list(profile.theta_k),
        qv_kgkg=[value / 1000.0 for value in profile.qv_gkg], u_ms=list(profile.u_ms), v_ms=list(profile.v_ms),
        p_pa=None, surface=surface, rotation={}, theta_m={}, model_top_face_m=profile.top_m)
    extension = record.get("extension") or {}
    line = str(record.get("name") or "")
    if record.get("converted"):
        line += f", converted from {record.get('format_words')}"
    from ..quicklook import style as S

    first = ["a sounding file, not a GM1 run", "one column, the same air everywhere", "winds ground-relative"]
    if extension.get("extended"):
        first.append(f"above {extension['from_m'] / 1000.0:.1f} km the levels are made up")
    footer = [S.data_line(*first),
              S.data_line("T from θ and CM1's hydrostatic pressure; dewpoint from qv (CM1 getcape.F gettd)",
                          "numbers: CM1 getcape.F via GM1's report", "hodograph 0–10 km, rings every 10 (m/s)")]
    return wrfpictures.preview_png(column, record.get("report"), out,
                                   words={"title": "Your sounding", "line": line, "footer": footer})


__all__ = [
    "BUBBLE_MOIST_RATIO", "HEDGE", "RECORD_NAME", "SCHEMA", "STARTS", "START_LINES", "START_TITLES", "TRAP",
    "VERDICTS", "examine", "kept", "label_from", "numbers", "preview_png", "record_path", "results_words",
    "source_words",
    "start_choice", "verdict", "write",
]
