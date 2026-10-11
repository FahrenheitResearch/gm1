"""Customize this storm: the settings a guided preset may change without leaving guided mode.

The budget screen's "Customize this storm" panel, ``gpuwm-cm1 preset fit|new ID --custom KNOB=VALUE`` and
``gpuwm-cm1 preset options ID`` all go through here.  A setting is a *knob* with plain choices; :func:`apply` turns
the chosen values into the preset's own CM1 namelist words (and the GM1 settings CM1 keeps in its Fortran, such as
the warm bubble), written to a scratch namelist, so everything after it is the ordinary guided path: the guided
check (``presets.examine``, which offers only what GM1 runs as CM1 does), the budget fit and the import.  Nothing
here decides whether a choice runs as CM1 does; a choice that guided mode would turn down is shown greyed with the
check's own reason (:data:`GREYED`, which a test holds equal to ``examine``'s).

The knobs (``KNOB=VALUE``):

``grid=250|500|1000|2000``  grid spacing in metres (dx = dy).  The area is kept (nx, ny follow) and the time step
    follows the grid and the levels (:func:`time_step`), so it is never a choice of its own.
``levels=250|500``          height of each level in metres (dz); the model top stays the preset's (nz follows).
``area=KM``                 the longer side of the area in km, in steps of 8 cells, from the preset's smallest
    (120 km with parcels) to twice its own or 240 km; a non-square preset keeps its shape.
``time=MIN``                storm time in minutes (timax).
``pictures=MIN``            a saved picture every MIN minutes (tapfrq); it must split the storm time.
``push=N``                  the starting push: the warm bubble's strength in K (iinit = 1) or the push of rising
    air in m/s (iinit = 12).
``bubble=KM``               the warm bubble's radius in km (iinit = 1).
``bumps=off|on|seed:N``     random bumps at the start (irandp): on draws CM1's own random numbers, ``seed:N``
    another draw from GM1's.
``rotation=off|on``         Earth's rotation (icor; fcor stays the preset's).
``parcels=off|on``          follow air parcels (iprcl, nparcels = CM1's 60 x 60 x 10 parcel grid).
``scheme=morrison|wsm6|thompson`` and ``friction=off|on``  shown greyed with the guided check's reason.

A setting the user changes is *pinned*: the fit never changes it (``fit.PresetSpec.pinned``).  The air is not a
knob here: it is the Make-your-own-environment flags (``--instability --shear --hodograph``), which
``presets.with_environment`` takes on a Weisman-Klemp preset too, through :func:`made_air`.

Standard library only (the importer and the fit are read lazily, by the command that uses them).
"""

from __future__ import annotations

from dataclasses import replace
import math
import tempfile
import atexit
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .presets import Preset

OPTIONS_SCHEMA = "gpuwm-cm1.preset-options/v1"

#: Every knob, in the panel's order, with its group.
KNOBS: tuple[tuple[str, str], ...] = (
    ("grid", "size"), ("levels", "size"), ("area", "size"), ("time", "size"), ("pictures", "size"),
    ("push", "storm"), ("bubble", "storm"), ("bumps", "storm"),
    ("scheme", "physics"), ("friction", "physics"), ("rotation", "physics"),
    ("parcels", "extras"),
)
KNOB_IDS: tuple[str, ...] = tuple(knob for knob, _ in KNOBS)
GROUP_ORDER = ("size", "storm", "physics", "extras")
#: The knobs that are pinned once chosen (the fit's own four levers).
PINNABLE = ("grid", "area", "time", "pictures")
PIN_WORDS = {"grid": "grid spacing", "area": "area", "time": "storm time", "pictures": "picture interval"}

GRIDS_M = (250.0, 500.0, 1000.0, 2000.0)
LEVELS_M = (250.0, 500.0)
PICTURES_MIN = (1, 2, 5, 10, 15, 30)
STRENGTHS_K = (1.0, 1.5, 2.0, 3.0)
BUBBLES_KM = (5.0, 10.0, 15.0)
PUSHES_M_S = (5.0, 10.0, 15.0, 20.0)
#: The area step, cells per side: the fit's own shrink step.
AREA_STEP_CELLS = 8
AREA_MAX_KM = 240.0
#: CM1's own parcel grid (60 x 60 x 10, every 2 km across and 1 km up) covers 120 km, so the area is at least that.
PARCELS = 36000
PARCELS_FLOOR_KM = 120.0
#: CM1's own step: 6 s per 1 km of grid spacing and 6 s per 500 m of level height (the fit's own cap,
#: fit.CM1_DT_PER_DZ_S_PER_M).
DT_PER_DX_S_PER_M = 6.0 / 1000.0
DT_PER_DZ_S_PER_M = 6.0 / 500.0
#: CM1's own values of the warm bubble and the push of rising air (init3d.F; config.py's defaults).
CM1_BPTPERT_K = 1.0
CM1_BHRAD_M = 10000.0
CM1_WMAX_WNUDGE = 10.0
SCHEMES = {"morrison": 5, "wsm6": 7, "thompson": 3}
SCHEME_WORDS = {"morrison": "Morrison", "wsm6": "WSM6", "thompson": "Thompson"}
FRICTION = {"bbc": 3, "sfcmodel": 1, "cecd": 1}

#: Choices guided mode turns down today, with the guided check's own reason (tests hold these equal to
#: ``presets.examine``'s reasons for the customized namelist, so the words cannot drift from the check).
GREYED: Mapping[tuple[str, str], str] = {
    ("scheme", "wsm6"): ("Rain and hail: WSM6. GM1 solves WSM6 with slightly different equations from CM1's, and on "
                         "CM1's own storm GM1's run comes out stronger than CM1's (measured)."),
    ("scheme", "thompson"): "Rain and hail: Thompson. It has been checked only in a single column, not in a storm.",
    ("friction", "on"): "The ground-friction setup has not been checked against CM1 in a storm.",
}
#: Why the air cannot be changed on a preset, in words.
AIR_FAMOUS = "A famous day runs on that day's real weather balloon; that air is the point of it."
AIR_WIND = "This storm's wind is not one the two air settings can make."
AIR_FIXED = "This storm runs on its own sounding file."
#: The CM1 words Customize and the fit own: ``--set`` does not take them (the knob to use instead, in words).
FIT_OWNED: Mapping[str, tuple[str, str]] = {
    "dx": ("grid", "grid spacing"), "dy": ("grid", "grid spacing"),
    "dz": ("levels", "height of the levels"), "nz": ("levels", "number of levels"),
    "nx": ("area", "area"), "ny": ("area", "area"),
    "timax": ("time", "storm time"), "tapfrq": ("pictures", "picture interval"),
}


class CustomError(ValueError):
    """A knob that does not exist, or a value it cannot take here; the message says which, in plain words."""


# ------------------------------------------------------------------ reading a preset


def _effective(preset: "Preset") -> dict[str, Any]:
    from ..cm1import import coverage

    return dict(coverage.import_for_report(preset.namelist_path).effective)


def _int(effective: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = effective.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(effective: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = effective.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def time_step(dx_m: float, dz_m: float) -> float:
    """The time step a grid and its levels take: CM1's 6 s per 1 km across and per 500 m up, whichever is less."""

    return min(DT_PER_DX_S_PER_M * dx_m, DT_PER_DZ_S_PER_M * dz_m)


def _whole(value: float) -> bool:
    return abs(value - round(value)) < 1e-6


def _floor_km(preset: "Preset", dx_m: float, parcels: bool) -> float:
    from .fit import PresetSpec

    spec = PresetSpec.from_mapping({"id": preset.id, **dict(preset.record)}, "")
    floor = spec.min_width_m(dx_m) / 1000.0
    return max(floor, PARCELS_FLOOR_KM) if parcels else floor


def _largest_cells() -> int:
    from . import rates

    return int(rates.load_record()["largest_domain_cells"]["value"])


def _sides(effective: Mapping[str, Any]) -> tuple[int, int]:
    return _int(effective, "nx"), _int(effective, "ny")


def _air(preset: "Preset", effective: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the air may be changed here, and the sliders' starting place."""

    isnd, iwnd = _int(effective, "isnd"), _int(effective, "iwnd")
    made = dict((preset.record.get("sounding") or {}).get("environment") or {})
    given = dict((preset.sounding_given or {}).get("environment") or {})
    if given:
        made = given
    if made:
        return {"available": True, "why": None, "own": not given, "weisman_klemp": False,
                "instability": made.get("instability"), "shear": made.get("shear"),
                "hodograph": made.get("hodograph") or "curved"}
    if isnd == 5 and iwnd in (2, 4):
        from ..cm1import import analytic_sounding as A

        # Where CM1's own air sits on the sliders (getcape on the 1 km grid: 1872 J/kg, 31.1 m/s; measured).
        low, high = A.CAPE_RANGE_J_KG
        slow, shigh = A.SHEAR_RANGE_M_S
        return {"available": True, "why": None, "own": True, "weisman_klemp": True,
                "instability": round((1872.0 - low) / (high - low), 2),
                "shear": round((31.1 - slow) / (shigh - slow), 2), "hodograph": "curved"}
    if preset.group == "famous":
        return {"available": False, "why": AIR_FAMOUS}
    if isnd == 5:
        return {"available": False, "why": AIR_WIND}
    return {"available": False, "why": AIR_FIXED}


# ------------------------------------------------------------------ the choices


def _choice(value: Any, available: bool = True, why: str | None = None) -> dict[str, Any]:
    return {"value": value, "available": available, "why": why}


def storm_times_min(own_s: float) -> list[float]:
    """The storm times offered: the preset's own, and 15-minute steps from 30 min to 4 h (a short preset: 10-40 min)."""

    own = own_s / 60.0
    if own_s < 1800.0:
        values = {10.0, 15.0, 20.0, 30.0, 40.0}
    else:
        values = {float(minutes) for minutes in range(30, 241, 15)}
    values.add(round(own, 6))
    return sorted(values)


def picture_choices(time_s: float, dt_s: float, own_s: float) -> list[float]:
    """Picture intervals (minutes) that split ``time_s`` into whole pictures and land on whole time steps."""

    values = set(float(value) for value in PICTURES_MIN)
    values.add(round(own_s / 60.0, 6))
    return sorted(value for value in values
                  if value * 60.0 <= time_s + 1e-6 and _whole(time_s / (value * 60.0))
                  and _whole(value * 60.0 / dt_s))


def options(preset: "Preset", effective: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Each knob for ``preset``: its default (the preset's own value), its choices, which are available and why not.

    The page draws the panel from this (``preset list`` carries it per preset as ``customize``) and
    :func:`apply` checks a value against the same choices.
    """

    effective = dict(effective) if effective is not None else _effective(preset)
    nx, ny = _sides(effective)
    nz = _int(effective, "nz")
    dx, dz = _float(effective, "dx"), _float(effective, "dz")
    dt = _float(effective, "dtl")
    timax, tapfrq = _float(effective, "timax"), _float(effective, "tapfrq")
    iinit = _int(effective, "iinit")
    parcels = _int(effective, "iprcl") == 1
    ptype = _int(effective, "ptype")
    stretched = _int(effective, "stretch_z") != 0
    long_cells, short_cells = max(nx, ny), min(nx, ny)
    long_km = long_cells * dx / 1000.0
    floor = _floor_km(preset, dx, parcels)
    aspect = short_cells / long_cells if long_cells else 1.0
    largest = _largest_cells()
    settings = dict(preset.record.get("gm1_settings") or {})
    knobs: list[dict[str, Any]] = []

    def knob(knob_id: str, default: Any, choices: list[dict[str, Any]] | None = None, **more: Any) -> None:
        group = dict(KNOBS)[knob_id]
        knobs.append({"id": knob_id, "group": group, "default": default, "choices": choices or [], **more})

    grids = sorted(set(GRIDS_M) | {dx})
    knob("grid", dx, [_choice(value) for value in grids], unit="m")
    if not stretched:
        knob("levels", dz, [_choice(value) for value in sorted(set(LEVELS_M) | {dz})], unit="m",
             top_m=nz * dz)
    knob("area", round(long_km, 3), unit="km", min_km=floor / aspect if aspect else floor,
         max_km=max(long_km, min(2.0 * long_km, AREA_MAX_KM)), step_cells=AREA_STEP_CELLS, aspect=aspect,
         floor_km=floor, parcels_floor_km=PARCELS_FLOOR_KM, largest_cells=largest, nz=nz)
    times = storm_times_min(timax)
    knob("time", round(timax / 60.0, 6), [_choice(value) for value in times], unit="min")
    knob("pictures", round(tapfrq / 60.0, 6),
         [_choice(value) for value in picture_choices(timax, dt, tapfrq)], unit="min",
         all=list(sorted(set(float(v) for v in PICTURES_MIN) | {round(tapfrq / 60.0, 6)})))
    if iinit == 1:
        strength = float(settings.get("initialization.bptpert_k", CM1_BPTPERT_K))
        radius = float(settings.get("initialization.bhrad_m", CM1_BHRAD_M)) / 1000.0
        sizes = BUBBLES_KM if radius >= min(BUBBLES_KM) else (radius / 2.0, radius, radius * 1.5)
        knob("push", strength, [_choice(value) for value in sorted(set(STRENGTHS_K) | {strength})], unit="K",
             kind="bubble")
        knob("bubble", radius, [_choice(value) for value in sorted(set(sizes) | {radius})], unit="km")
    elif iinit == 12:
        target = float(settings.get("initialization.wmax_wnudge", CM1_WMAX_WNUDGE))
        knob("push", target, [_choice(value) for value in sorted(set(PUSHES_M_S) | {target})], unit="m/s",
             kind="updraft")
    bumps = "on" if _int(effective, "irandp") == 1 else "off"
    knob("bumps", bumps, [_choice("off"), _choice("on"), _choice("seed")])
    scheme = next((name for name, value in SCHEMES.items() if value == ptype), None)
    if scheme is not None:
        knob("scheme", scheme, [
            _choice(name, available=(name == scheme or (("scheme", name) not in GREYED)),
                    why=None if name == scheme else GREYED.get(("scheme", name)))
            for name in SCHEMES])
    friction = "on" if all(_int(effective, key) == value for key, value in FRICTION.items()) else "off"
    knob("friction", friction, [_choice("off"), _choice("on", available=friction == "on",
                                                        why=None if friction == "on" else GREYED[("friction", "on")])])
    knob("rotation", "on" if _int(effective, "icor") == 1 else "off", [_choice("off"), _choice("on")])
    # The parcel grid needs 120 km: a storm whose area cannot be that large shows On greyed, with why.
    area_cap = max(long_km, min(2.0 * long_km, AREA_MAX_KM)) * aspect
    room = parcels or area_cap >= PARCELS_FLOOR_KM - 1e-6
    knob("parcels", "on" if parcels else "off", [
        _choice("off"),
        _choice("on", available=room, why=None if room else (
            f"Follow air parcels needs an area at least {PARCELS_FLOOR_KM:g} km across, and this storm's area goes "
            f"up to {area_cap:g} km."))],
         floor_km=PARCELS_FLOOR_KM)
    return {
        "schema": OPTIONS_SCHEMA,
        "preset": preset.id,
        "groups": list(GROUP_ORDER),
        "knobs": knobs,
        "air": _air(preset, effective),
        "grid": {"dx_m": dx, "dz_m": dz, "nz": nz, "dt_s": dt, "top_m": nz * dz, "nx": nx, "ny": ny},
        "command": f"gpuwm-cm1 preset options {preset.id} --json",
    }


# ------------------------------------------------------------------ reading choices


def _number(knob: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise CustomError(f"{knob} must be a number, not {text!r}") from None
    if not math.isfinite(value):
        raise CustomError(f"{knob} must be a number, not {text!r}")
    return value


def _switch(knob: str, text: str) -> str:
    low = text.strip().lower()
    if low in ("on", "true", "yes", "1"):
        return "on"
    if low in ("off", "false", "no", "0"):
        return "off"
    raise CustomError(f"{knob} is on or off, not {text!r}")


def _bumps(text: str) -> str | int:
    low = text.strip().lower()
    if low.startswith("seed:"):
        try:
            seed = int(low.split(":", 1)[1])
        except ValueError:
            raise CustomError(f"bumps=seed:N takes a whole number, not {text!r}") from None
        if not 1 <= seed <= 2**64 - 1:
            raise CustomError(f"bumps=seed:N takes a whole number from 1 to {2**64 - 1}, not {seed}")
        return seed
    return _switch("bumps", text)


def parse(items: Iterable[str]) -> dict[str, Any]:
    """``["grid=500", "parcels=on"]`` (``--custom``) as ``{"grid": 500.0, "parcels": "on"}``; unknown names refused.

    A value is only read here; whether this preset offers it is :func:`apply`'s check against :func:`options`.
    """

    out: dict[str, Any] = {}
    for item in items:
        name, equals, text = str(item).partition("=")
        name = name.strip().lower()
        if not equals or not name:
            raise CustomError(f"--custom takes KNOB=VALUE, not {item!r}; the knobs are {', '.join(KNOB_IDS)}")
        if name not in KNOB_IDS:
            raise CustomError(f"There is no setting called {name}; the settings are {', '.join(KNOB_IDS)}")
        text = text.strip()
        if name in ("grid", "levels", "area", "time", "pictures", "push", "bubble"):
            out[name] = _number(name, text)
        elif name == "bumps":
            out[name] = _bumps(text)
        elif name == "scheme":
            low = text.lower()
            if low not in SCHEMES:
                raise CustomError(f"scheme is one of {', '.join(SCHEMES)}, not {text!r}")
            out[name] = low
        else:
            out[name] = _switch(name, text)
    return out


def custom_args(custom: Mapping[str, Any]) -> list[str]:
    """The ``--custom`` flags that give ``custom`` again, in the knobs' order (the recorded command)."""

    args: list[str] = []
    for knob in KNOB_IDS:
        if knob not in custom:
            continue
        value = custom[knob]
        if knob == "bumps" and isinstance(value, int) and not isinstance(value, bool):
            text = f"seed:{value}"
        elif isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        args += ["--custom", f"{knob}={text}"]
    return args


def parse_sets(items: Iterable[str]) -> dict[str, Any]:
    """``--set KEY=VALUE``: CM1 namelist words for the command line, as ``gpuwm-cm1 edit --set`` reads them."""

    out: dict[str, Any] = {}
    for item in items:
        key, equals, text = str(item).partition("=")
        key = key.strip().lower()
        if not equals or not key:
            raise CustomError(f"--set takes KEY=VALUE, not {item!r}")
        text = text.strip()
        low = text.lower()
        value: Any
        if low in (".true.", "true", "t", ".t."):
            value = True
        elif low in (".false.", "false", "f", ".f."):
            value = False
        else:
            try:
                value = int(text)
            except ValueError:
                try:
                    value = float(text)
                except ValueError:
                    value = text
        out[key] = value
    return out


# ------------------------------------------------------------------ applying choices


def _words_m(value_m: float) -> str:
    return f"{value_m / 1000.0:g} km" if value_m >= 1000.0 else f"{value_m:g} m"


def _words_min(minutes: float) -> str:
    whole = int(round(minutes))
    if abs(minutes - whole) > 1e-6:
        return f"{minutes:g} min"
    if whole >= 60 and whole % 60 == 0:
        return f"{whole // 60} h"
    if whole > 60:
        return f"{whole // 60} h {whole % 60} min"
    return f"{whole} min"


def _scratch() -> Path:
    folder = Path(tempfile.mkdtemp(prefix="gm1-custom-"))
    atexit.register(shutil.rmtree, folder, True)
    return folder


def _offered_value(knob: Mapping[str, Any], value: Any, words: str, greyed_too: bool = False) -> None:
    for choice in knob["choices"]:
        same = (math.isclose(float(choice["value"]), float(value), abs_tol=1e-6)
                if isinstance(value, float) and isinstance(choice["value"], (int, float)) else choice["value"] == value)
        if same:
            if not choice.get("available", True) and not greyed_too:
                raise CustomError(f"{choice.get('why')} You can still choose it in Advanced.")
            return
    raise CustomError(f"{words} is not one of the choices here")


def apply(preset: "Preset", custom: Mapping[str, Any], sets: Mapping[str, Any] | None = None, *,
          greyed_too: bool = False) -> "Preset":
    """``preset`` with the chosen knobs: its namelist rewritten in scratch, its GM1 settings, floors and pins.

    The result is an ordinary preset (``presets.Preset``) whose namelist is read from the scratch file and whose
    original stays the shipped one, so ``examine``, the fit and ``preset new`` treat it as they treat any preset,
    and the run folder's ``namelist.diff`` shows each change.  Each change is a ``customized`` entry
    (``{knob, keys {key: [from, to]}, sentence}``) the record carries into ``fit.json``.  A greyed choice is
    refused with its reason unless ``greyed_too`` (the test that holds :data:`GREYED` to the guided check).
    """

    sets = dict(sets or {})
    if not custom and not sets:
        return preset
    unknown = sorted(set(custom) - set(KNOB_IDS))
    if unknown:
        raise CustomError(f"There is no setting called {unknown[0]}; the settings are {', '.join(KNOB_IDS)}")
    effective = _effective(preset)
    offer = options(preset, effective)
    by_id = {knob["id"]: knob for knob in offer["knobs"]}
    for knob in custom:
        if knob not in by_id:
            raise CustomError(f"{preset.name} has no {knob} setting to change")
    nx0, ny0 = _sides(effective)
    nz0 = _int(effective, "nz")
    dx0, dz0, dt0 = _float(effective, "dx"), _float(effective, "dz"), _float(effective, "dtl")
    timax0, tapfrq0 = _float(effective, "timax"), _float(effective, "tapfrq")
    iinit = _int(effective, "iinit")
    values: dict[str, Any] = {}
    gm1: dict[str, Any] = {}
    record = dict(preset.record)
    settings0 = dict(record.get("gm1_settings") or {})
    customized: list[dict[str, Any]] = list(record.get("customized") or [])
    pinned = set(record.get("pinned") or [])
    seed: int | None = record.get("random_seed")

    def note(knob: str, keys: Mapping[str, Sequence[Any]], sentence: str) -> None:
        customized.append({"knob": knob, "keys": {key: list(pair) for key, pair in keys.items()}, "sentence": sentence})

    # The grid, the levels and the time step that follows them.
    dx = float(custom.get("grid", dx0))
    dz = float(custom.get("levels", dz0))
    if "grid" in custom:
        _offered_value(by_id["grid"], dx, f"A {_words_m(dx)} grid")
    if "levels" in custom:
        _offered_value(by_id["levels"], dz, f"Levels every {_words_m(dz)}")
    nz = nz0
    dt = dt0
    if not math.isclose(dx, dx0) or not math.isclose(dz, dz0):
        dt = time_step(dx, dz)
    if not math.isclose(dz, dz0):
        nz = int(round(nz0 * dz0 / dz))
        values.update(dz=dz, nz=nz)
        note("levels", {"dz": [dz0, dz], "nz": [nz0, nz]},
             f"Height levels every {_words_m(dz)} instead of every {_words_m(dz0)} ({nz} levels up to "
             f"{_words_m(nz * dz)}).")
    # The area: the longer side, the shape kept.
    parcels_on = custom.get("parcels", by_id["parcels"]["default"]) == "on"
    if parcels_on:
        on = next(choice for choice in by_id["parcels"]["choices"] if choice["value"] == "on")
        if not on.get("available", True):
            raise CustomError(str(on.get("why")))
    long0 = max(nx0, ny0) * dx0
    long_m = float(custom["area"]) * 1000.0 if "area" in custom else long0
    floor_km = _floor_km(preset, dx, parcels_on)
    aspect = min(nx0, ny0) / max(nx0, ny0) if max(nx0, ny0) else 1.0
    if "area" in custom:
        area = by_id["area"]
        if long_m > area["max_km"] * 1000.0 + 1e-6:
            raise CustomError(f"An area {custom['area']:g} km wide is larger than this storm offers "
                              f"({area['max_km']:g} km at most).")
        if long_m * aspect < floor_km * 1000.0 - 1e-6:
            raise CustomError(f"The area must be at least {floor_km / aspect:g} km across"
                              f"{_parcels_why(preset, dx, parcels_on, floor_km)}; {custom['area']:g} km is too small."
                              f"{_parcels_fix(by_id, parcels_on, floor_km)}")
    long_cells = int(round(long_m / dx))
    if "area" in custom:
        # The area moves in steps of 8 cells from the preset's own size (the slider's steps, the fit's own step).
        base = int(round(long0 / dx))
        long_cells = base + AREA_STEP_CELLS * int(round((long_cells - base) / AREA_STEP_CELLS))
        while long_cells * dx > by_id["area"]["max_km"] * 1000.0 + 1e-6:
            long_cells -= AREA_STEP_CELLS
        while long_cells * dx * aspect < floor_km * 1000.0 - 1e-6:
            long_cells += AREA_STEP_CELLS
    if nx0 >= ny0:
        nx, ny = long_cells, int(round(long_cells * ny0 / nx0))
    else:
        nx, ny = int(round(long_cells * nx0 / ny0)), long_cells
    if min(nx, ny) * dx < floor_km * 1000.0 - 1e-6:
        raise CustomError(f"The area must be at least {floor_km:g} km across"
                          f"{_parcels_why(preset, dx, parcels_on, floor_km)}; "
                          f"{min(nx, ny) * dx / 1000.0:g} km is too small.{_parcels_fix(by_id, parcels_on, floor_km)}")
    if "grid" in custom and not math.isclose(dx, dx0):
        values.update(dx=dx, dy=dx, dtl=dt)
        keys: dict[str, list[Any]] = {"dx": [dx0, dx], "dy": [dx0, dx], "dtl": [dt0, dt]}
        if "area" not in custom:
            keys.update(nx=[nx0, nx], ny=[ny0, ny])
        step = "" if math.isclose(dt, dt0) else f" (and a {dt:g} s time step instead of {dt0:g} s)"
        note("grid", keys, f"Grid spacing {_words_m(dx)} instead of {_words_m(dx0)}{step}.")
    elif not math.isclose(dt, dt0):
        values.update(dtl=dt)
        customized[-1]["keys"]["dtl"] = [dt0, dt]
        customized[-1]["sentence"] = customized[-1]["sentence"][:-1] + f", with a {dt:g} s time step instead of {dt0:g} s."
    if (nx, ny) != (nx0, ny0):
        values.update(nx=nx, ny=ny)
    if "area" in custom and abs(max(nx, ny) * dx - long0) > 1.0:
        before = f"{nx0 * dx0 / 1000.0:g} km" if nx0 == ny0 else f"{nx0 * dx0 / 1000.0:g} × {ny0 * dx0 / 1000.0:g} km"
        after = f"{nx * dx / 1000.0:g} km" if nx == ny else f"{nx * dx / 1000.0:g} × {ny * dx / 1000.0:g} km"
        note("area", {"nx": [nx0, nx], "ny": [ny0, ny]},
             f"An area {after} wide instead of {before}." if nx == ny else f"An area of {after} instead of {before}.")
    for knob in ("grid", "area"):
        if knob in custom:
            pinned.add(knob)
    # Storm time and pictures.
    timax = timax0
    if "time" in custom:
        _offered_value(by_id["time"], float(custom["time"]), f"{_words_min(float(custom['time']))} of storm time")
        timax = float(custom["time"]) * 60.0
        pinned.add("time")
        if not math.isclose(timax, timax0):
            values["timax"] = timax
            note("time", {"timax": [timax0, timax]},
                 f"{_words_min(timax / 60.0)} of storm time instead of {_words_min(timax0 / 60.0)}.")
        if timax < float(record.get("min_duration_s", 3600.0)):
            record["min_duration_s"] = timax
    tapfrq = tapfrq0
    if "pictures" in custom:
        tapfrq = float(custom["pictures"]) * 60.0
        if not float(custom["pictures"]) in [float(v) for v in by_id["pictures"]["all"]]:
            raise CustomError(f"A picture every {_words_min(float(custom['pictures']))} is not one of the choices")
        pinned.add("pictures")
        if not math.isclose(tapfrq, tapfrq0):
            values["tapfrq"] = tapfrq
            note("pictures", {"tapfrq": [tapfrq0, tapfrq]},
                 f"A saved picture every {_words_min(tapfrq / 60.0)} instead of every {_words_min(tapfrq0 / 60.0)}.")
    if not _whole(timax / tapfrq):
        raise CustomError(f"A picture every {_words_min(tapfrq / 60.0)} does not split {_words_min(timax / 60.0)} of "
                          "storm time into whole pictures; pick another picture interval")
    if not _whole(tapfrq / dt):
        raise CustomError(f"A picture every {_words_min(tapfrq / 60.0)} does not land on a whole number of "
                          f"{dt:g} s time steps; pick another picture interval")
    # The starting push.
    if "push" in custom:
        value = float(custom["push"])
        knob = by_id["push"]
        unit = "K" if knob.get("kind") == "bubble" else "m/s"
        _offered_value(knob, value, f"A push of {value:g} {unit}")
        key = "initialization.bptpert_k" if knob.get("kind") == "bubble" else "initialization.wmax_wnudge"
        was = float(settings0.get(key, knob["default"]))
        if not math.isclose(value, was):
            gm1[key] = value
            if knob.get("kind") == "bubble":
                note("push", {key: [was, value]},
                     f"A warm bubble {value:g} K warmer than the air around it instead of {was:g} K.")
            else:
                note("push", {key: [was, value]},
                     f"The push of rising air aims for {value:g} m/s upward (ground-relative) instead of {was:g} m/s.")
    if "bubble" in custom:
        value = float(custom["bubble"])
        _offered_value(by_id["bubble"], value, f"A bubble {value:g} km wide")
        key = "initialization.bhrad_m"
        was = float(settings0.get(key, CM1_BHRAD_M))
        if not math.isclose(value * 1000.0, was):
            gm1[key] = value * 1000.0
            note("bubble", {key: [was, value * 1000.0]},
                 f"A warm bubble {value:g} km in radius instead of {was / 1000.0:g} km.")
    # Random bumps.
    if "bumps" in custom:
        bumps = custom["bumps"]
        irandp0 = _int(effective, "irandp")
        seed0 = seed
        if bumps == "off":
            if irandp0 != 0:
                values["irandp"] = 0
                note("bumps", {"irandp": [irandp0, 0]}, "No random bumps at the start.")
            seed = None
        else:
            if irandp0 != 1:
                values["irandp"] = 1
            seed = bumps if isinstance(bumps, int) else None
            # Noted only when it changes something: the bumps turned on, or another draw than the preset's own.
            if irandp0 != 1 or seed != seed0:
                note("bumps", {"irandp": [irandp0, 1]},
                     f"Small random bumps in the starting air, from GM1's own random numbers (draw {bumps})."
                     if seed is not None else "Small random bumps in the starting air, from CM1's own random numbers.")
    # Physics.
    if "scheme" in custom:
        _offered_value(by_id["scheme"], custom["scheme"], f"Rain and hail: {SCHEME_WORDS[custom['scheme']]}",
                       greyed_too)
        ptype0, ptype = _int(effective, "ptype"), SCHEMES[custom["scheme"]]
        if ptype != ptype0:
            values["ptype"] = ptype
            note("scheme", {"ptype": [ptype0, ptype]}, f"Rain and hail by {SCHEME_WORDS[custom['scheme']]}.")
    if "friction" in custom:
        _offered_value(by_id["friction"], custom["friction"], "Ground friction", greyed_too)
        if custom["friction"] != by_id["friction"]["default"]:
            on = custom["friction"] == "on"
            keys = {key: [_int(effective, key), value if on else 0 if key != "bbc" else 1]
                    for key, value in FRICTION.items()}
            values.update({key: pair[1] for key, pair in keys.items()})
            note("friction", keys, "Ground friction on." if on else "No ground friction.")
    if "rotation" in custom and custom["rotation"] != by_id["rotation"]["default"]:
        on = custom["rotation"] == "on"
        values["icor"] = 1 if on else 0
        note("rotation", {"icor": [0 if on else 1, 1 if on else 0]},
             rotation_sentence(timax) if on else "Earth's rotation off.")
    if "parcels" in custom and custom["parcels"] != by_id["parcels"]["default"]:
        on = custom["parcels"] == "on"
        iprcl0, nparcels0 = _int(effective, "iprcl"), _int(effective, "nparcels", 1)
        values.update(iprcl=1 if on else 0, nparcels=PARCELS if on else 1)
        note("parcels", {"iprcl": [iprcl0, 1 if on else 0], "nparcels": [nparcels0, PARCELS if on else 1]},
             "Follows 36,000 air parcels on CM1's own parcel grid." if on else "Follows no air parcels.")
    if parcels_on:
        widths = record.get("min_width_km", 64.0)
        if isinstance(widths, Mapping):
            record["min_width_km"] = {key: max(float(value), PARCELS_FLOOR_KM) for key, value in widths.items()}
        else:
            record["min_width_km"] = max(float(widths), PARCELS_FLOOR_KM)
    # --set KEY=VALUE (the command line's own CM1 words), checked by the same guided check as the knobs.  The words
    # the knobs and the fit own are not taken: the fit would change them again, and the time step must follow them.
    for key in sets:
        refuse_set(key, preset.id)
    for key, value in sets.items():
        before = effective.get(key)
        values[key] = value
        note("set", {key: [before, value]}, f"Set {key} = {value} by hand.")
    if gm1:
        record["gm1_settings"] = {**settings0, **gm1}
    from .fit import set_namelist_values

    try:
        text = set_namelist_values(preset.namelist_text, values) if values else preset.namelist_text
    except ValueError as error:
        raise CustomError(str(error)) from error
    folder = _scratch()
    (folder / "namelist.input").write_text(text, encoding="ascii")
    record["customized"] = customized
    record["pinned"] = sorted(pinned)
    record["custom"] = {**dict(record.get("custom") or {}), **dict(custom)}
    record["custom_sets"] = {**dict(record.get("custom_sets") or {}), **sets}
    if seed is not None:
        record["random_seed"] = seed
    else:
        record.pop("random_seed", None)
    changed = replace(preset, record=record, namelist_text=text, namelist_file=folder / "namelist.input",
                      namelist_original=preset.namelist_original_text)
    if values.keys() & {"dx", "nx", "ny", "timax", "dz"}:
        # A longer storm or a smaller area can take the storm out of its area: the view then follows the storm with
        # the storm-track check's own frame and bubble (one more "customized" sentence), as a new sounding does.
        from .presets import follow_storm_if_it_leaves

        changed = follow_storm_if_it_leaves(changed)
    return changed


def rotation_sentence(timax_s: float) -> str:
    """What turning on Earth's rotation does over a storm this long, in words."""

    if timax_s < 3600.0 - 1e-6:
        return "Earth's rotation on. Over a storm this short it turns the wind very little."
    return ("Earth's rotation on. Nothing balances it, so the wind slowly turns as the hours pass, about 20 degrees "
            "in 2 hours, as it does in CM1.")


def refuse_set(key: str, preset_id: str) -> None:
    """``--set KEY=VALUE`` on a word the knobs or the fit own is refused, naming the ``--custom`` knob to use."""

    key = key.strip().lower()
    if key == "dtl":
        raise CustomError("--set dtl is not taken: the time step follows the grid spacing and the levels (CM1's own "
                          "6 s per 1 km across and per 500 m up), so use --custom grid=M or --custom levels=M.")
    if key in FIT_OWNED:
        knob, words = FIT_OWNED[key]
        raise CustomError(f"--set {key} is not taken: the {words} is one of the settings the fit and Customize own; "
                          f"use --custom {knob}=VALUE ('gpuwm-cm1 preset options {preset_id}' lists its choices).")


def _parcels_why(preset: "Preset", dx: float, parcels_on: bool, floor_km: float) -> str:
    if parcels_on and floor_km <= PARCELS_FLOOR_KM + 1e-6 and _floor_km(preset, dx, False) < floor_km:
        return " because followed air parcels need CM1's whole parcel grid"
    return ""


def _parcels_fix(by_id: Mapping[str, Any], parcels_on: bool, floor_km: float) -> str:
    """What to do when the parcel grid is what makes the area too small."""

    if not parcels_on or floor_km > PARCELS_FLOOR_KM + 1e-6:
        return ""
    area = by_id.get("area") or {}
    if float(area.get("max_km", 0.0)) * float(area.get("aspect", 1.0)) < PARCELS_FLOOR_KM - 1e-6:
        return " This storm's area cannot be that large, so turn off Follow air parcels."
    return " Turn off Follow air parcels, or pick a larger area."


def made_air(preset: "Preset") -> "Preset":
    """A Weisman-Klemp preset (isnd = 5, iwnd = 2 or 4) made ready for air from the two settings: isnd = 7, iwnd = 0.

    CM1 forces iwnd = 0 whenever isnd = 7 (param.F:836-852), so both change; the sounding file is the generator's
    (``presets.with_environment``).  Any other preset is returned as it is.
    """

    effective = _effective(preset)
    isnd, iwnd = _int(effective, "isnd"), _int(effective, "iwnd")
    if isnd != 5 or iwnd not in (2, 4):
        return preset
    from .fit import set_namelist_values

    text = set_namelist_values(preset.namelist_text, {"isnd": 7, "iwnd": 0})
    folder = _scratch()
    (folder / "namelist.input").write_text(text, encoding="ascii")
    record = dict(preset.record)
    record["customized"] = [*list(record.get("customized") or []), {
        "knob": "air", "keys": {"isnd": [isnd, 7], "iwnd": [iwnd, 0]},
        "sentence": "The air is made from your two settings instead of CM1's own built-in air.",
    }]
    record["custom_air"] = True
    return replace(preset, record=record, namelist_text=text, namelist_file=folder / "namelist.input",
                   namelist_original=preset.namelist_original_text)


def is_customized(preset: "Preset") -> bool:
    return bool(preset.record.get("customized"))


def pinned_words(pinned: Iterable[str]) -> str:
    words = [PIN_WORDS[knob] for knob in PINNABLE if knob in set(pinned)]
    if not words:
        return ""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


__all__ = ["CustomError", "FIT_OWNED", "GREYED", "KNOBS", "KNOB_IDS", "OPTIONS_SCHEMA", "PINNABLE", "apply",
           "custom_args", "is_customized", "made_air", "options", "parse", "parse_sets", "pinned_words",
           "refuse_set", "rotation_sentence", "time_step"]
