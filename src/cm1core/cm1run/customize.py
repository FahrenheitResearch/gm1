"""Customize this storm: the settings a guided preset may change without leaving guided mode.

The budget screen's "Customize this storm" panel, ``gpuwm-cm1 preset fit|new ID --custom KNOB=VALUE`` and
``gpuwm-cm1 preset options ID`` all go through here.  A setting is a *knob* with plain choices; :func:`apply` turns
the chosen values into the preset's own CM1 namelist words (and the GM1 settings CM1 keeps in its Fortran, such as
the warm bubble), written to a scratch namelist, so everything after it is the ordinary guided path: the guided
check (``presets.examine``, which offers only what GM1 runs as CM1 does), the budget fit and the import.  Nothing
here decides whether a choice runs as CM1 does; a choice that guided mode would turn down is shown greyed with the
check's own reason (:data:`GREYED`, which a test holds equal to ``examine``'s).

The knobs (``KNOB=VALUE``):

``grid=M``                  grid spacing in metres (dx = dy): any value, 67 as well as 250.  The area is kept as
    nearly as whole cells allow (nx, ny follow: the count rounded up to a whole cell) and the time step follows the grid and the levels
    (:func:`snapped_time_step`: CM1's 6 s per km, shortened to land on the output times, and to a short binary
    fraction when the domain moves), so it is never a choice of its own.  The only spacings turned down are those
    that cannot be a grid: fewer than 8 cells across the area.
``levels=M``                height of each level in metres (dz): any value; the model top stays the preset's and nz
    follows (the nearest whole number of levels, so the top moves by less than half a level when dz does not divide
    it), within the 8 to 256 levels the model is built for.
``area=KM``                 the longer side of the area in km, in steps of 8 cells, from the smallest area a storm
    can start in (:func:`hard_floor_km`; 120 km with parcels) to twice the preset's own or 240 km; a non-square
    preset keeps its shape.  Below the area the preset was set up for, the fit says so.
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
import struct
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

#: The quick picks beside the grid slider and the level box (any other value is typed or slid to).
GRIDS_M = (100.0, 250.0, 500.0, 1000.0, 2000.0)
LEVELS_M = (100.0, 125.0, 250.0, 500.0)
#: The grid slider's ends (a log scale); a typed value or ``--custom grid=M`` may be outside them.
GRID_SLIDER_M = (20.0, 4000.0)
#: The model needs 7 cells across (cartesian_door.require_grid_extent: twice the stencil radius and one); the area
#: moves in steps of 8, so the smallest side is 8 cells.
MIN_CELLS = 8
#: The levels the model is built for: at least 8 (the vertical closure and the implicit pressure columns,
#: CM1-CARTESIAN-GRID-EXTENT "require 8 levels") and at most 256 (the compiled column capacity of the rain and hail
#: schemes, CM1-CARTESIAN-LEVEL-CAPACITY).
LEVELS_RANGE = (8, 256)
#: The smallest area a storm is started in: 16 km, or the warm bubble's own width when that is more.
HARD_FLOOR_KM = 16.0
#: The most cells a custom grid's area starts from, before the fit sizes it to the card.  One restart file holds
#: 16 GiB, about 61 million cells of a storm's carried state (280 bytes a cell; the estimate's restart-size check),
#: and one saved frame about 150 million (CM1-CARTESIAN-HISTORY-PAYLOAD), so a fine grid's area starts no larger
#: than a run can keep instead of being turned down for its first size.
START_CELLS_MAX = 60_000_000
#: The most cells one saved frame holds (16 GiB of a storm's fields, about 110 bytes a cell,
#: CM1-CARTESIAN-HISTORY-PAYLOAD): past it the guided check refuses the grid whatever the card.
FRAME_CELLS_MAX = 150_000_000
#: Output times are whole minutes, so the time step is made to divide a minute (or the shortest output time).
CADENCE_S = 60.0
#: The spacings GM1 ships presets at; outside them a custom grid carries a one-line note.
PRESET_SPACINGS_M = (100.0, 2000.0)
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
    """A knob that does not exist, or a value it cannot take here; the message says which, in plain words.

    ``suggest`` is a value that is taken, for a one-tap button (``{"grid": 23.0}``, ``{"levels": 78.0}``)."""

    def __init__(self, message: str, *, suggest: Mapping[str, float] | None = None) -> None:
        super().__init__(message)
        self.suggest = dict(suggest or {})


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
    """The longest time step a grid and its levels take: CM1's 6 s per 1 km across and per 500 m up, whichever is
    less.  The step a run uses is this one made to land on its output times (:func:`snapped_time_step`)."""

    return min(DT_PER_DX_S_PER_M * dx_m, DT_PER_DZ_S_PER_M * dz_m)


def lands(interval_s: float, dt_s: float) -> bool:
    """Whether ``interval_s`` is a whole number of ``dt_s`` steps, by the runtime's own arithmetic
    (``config.py``'s CM1-CFG-CLOCK-ALIGNMENT: the quotient, in binary64, is a whole number of at least one)."""

    if dt_s <= 0.0 or interval_s <= 0.0:
        return False
    steps = interval_s / dt_s
    return math.isfinite(steps) and steps >= 1.0 and float(steps).is_integer()


def _common_interval_s(intervals_s: Iterable[float], *, with_minute: bool = True) -> float:
    """The longest time every one of ``intervals_s`` (and, ``with_minute``, a minute) is a whole multiple of, in
    whole milliseconds."""

    common = int(round(CADENCE_S * 1000.0)) if with_minute else 0
    for value in intervals_s:
        common = math.gcd(common, max(1, int(round(float(value) * 1000.0))))
    return common / 1000.0


#: The times a fit may move a run's end or its pictures to (15-minute storm times, the picture intervals): a step
#: that lands on more of them leaves the fit and the panel more to choose from.
_USUAL_TIMES_S = tuple(sorted({float(minutes) * 60.0 for minutes in (1, 2, 5, 10, 15, 20, 30, 40)}
                              | {float(minutes) * 60.0 for minutes in range(45, 241, 15)}))


def binary_steps(common_s: float) -> list[float]:
    """The steps one octave holds (1 s up to, not including, 2 s) that are short binary fractions and divide
    ``common_s`` when halved or doubled far enough: an odd divisor of ``common_s`` under 256 over a power of two.

    For 300 s these are 1, 75/64, 1.25, 1.5, 25/16 and 1.875 s; every step ``c * 2**i`` of them (0.375 s,
    0.390625 s, 0.5 s, 0.75 s, 1.5 s, 3 s, 6 s) divides 300 s exactly, and so every multiple of it."""

    scale = 1
    while scale <= 1024 and abs(common_s * scale - round(common_s * scale)) > 1e-9:
        scale *= 2
    whole = int(round(common_s * scale))
    if scale > 1024 or whole <= 0:
        return []
    while whole % 2 == 0:
        whole //= 2
    out = []
    for odd in range(1, min(whole, 255) + 1, 2):
        if whole % odd == 0:
            out.append(odd / float(1 << (odd.bit_length() - 1)))
    return sorted(out)


def snap_time_step(raw_s: float, intervals_s: Iterable[float], *, binary: bool = False) -> float:
    """The longest step no longer than ``raw_s`` that every one of ``intervals_s`` is a whole number of.

    The runtime places every saved picture, the restart and the end of the run on a whole step and refuses a step
    that does not (CM1-CFG-CLOCK-ALIGNMENT), so a step such as 0.402 s (67 m) is shortened to one that does.

    ``binary`` (a domain that moves: imove = 1) asks for a short binary fraction as well.  CM1 keeps the moving
    domain's position as a running sum, one ``dt * umove`` a step, and GM1's run checks that sum against
    ``n * dt * umove`` exactly; the two agree at every step only while nothing rounds, which is so when the step is
    a small odd number over a power of two (6, 3, 1.5, 0.75, 0.5, 0.375 s) and not for 0.6 s or 0.4 s, where the
    sum rounds in the last place after a few steps and the run stops.  The step is then the longest of
    :func:`binary_steps` no longer than ``raw_s`` (0.402 s becomes 0.390625 s, 25/64, with pictures every
    5 minutes; 0.6 s becomes 0.5859375 s, 75/128), at most a fifth shorter than ``raw_s`` and usually a few
    percent, and exact for every multiple of the picture interval.

    Otherwise (a domain that stays where it is) the step is a whole fraction ``common/n`` of the longest time all
    the intervals share with a minute, with ``n`` the smallest that keeps it at or under ``raw_s`` (0.402 s becomes
    0.4 s, 150 steps a minute; 6 s, 1.5 s and 0.6 s stay as they are).  Such a step can miss a whole count by one
    unit in the last place, so each ``n`` is tested with the runtime's own arithmetic (:func:`lands`) and, among
    the next ``n`` that shorten the step by no more than 2 %, the one that lands on the most of the usual storm
    times and picture intervals is taken.
    """

    intervals = sorted({float(value) for value in intervals_s if float(value) > 0.0})
    if not (raw_s > 0.0 and math.isfinite(raw_s)) or not intervals:
        raise CustomError("the time step cannot be worked out for this grid")
    if binary:
        common = _common_interval_s(intervals, with_minute=False)
        octave = binary_steps(common)
        if octave:
            # The octave raw_s lies in, and the one below it for a raw_s under the octave's first step.
            power = math.floor(math.log2(raw_s))
            for shift in (power, power - 1):
                fits = [step * 2.0 ** shift for step in octave if step * 2.0 ** shift <= raw_s * (1.0 + 1e-12)]
                fits = [step for step in fits if all(lands(value, step) for value in intervals)]
                if fits:
                    return max(fits)
    common = _common_interval_s(intervals)
    first = max(1, int(math.ceil(common / raw_s - 1e-9)))
    needed = (common, *intervals)

    def steps_of(count: int) -> Iterable[float]:
        """``common/count`` as binary64 gives it, and its two neighbours: the same step to sixteen figures, one of
        which is the one the whole counts come out exact for."""

        step = common / count
        return (step, math.nextafter(step, 0.0), math.nextafter(step, math.inf))

    best: tuple[int, float] | None = None
    for count in range(first, first + first // 50 + 1):
        for step in steps_of(count):
            if step > raw_s * (1.0 + 1e-12) or not all(lands(value, step) for value in needed):
                continue
            score = sum(1 for value in _USUAL_TIMES_S if lands(value, step))
            if best is None or score > best[0]:
                best = (score, step)
            break  # the first of the three that lands: a neighbour only where the quotient itself does not
        if best is not None and best[0] == len(_USUAL_TIMES_S):
            break
    if best is not None:
        return best[1]
    for count in range(first + first // 50 + 1, 4 * first + 64):
        for step in steps_of(count):
            if all(lands(value, step) for value in needed):
                return step
    raise CustomError("no time step lands on this run's output times")  # pragma: no cover - never met in a sweep


def snapped_time_step(dx_m: float, dz_m: float, intervals_s: Iterable[float], *, moving: bool = False) -> float:
    """The time step a grid, its levels and its output times take: :func:`time_step`, made to land on every one of
    ``intervals_s`` (the storm time, the picture interval, the restart interval) by :func:`snap_time_step`, as a
    short binary fraction when the domain moves (``moving``: imove = 1)."""

    return snap_time_step(time_step(dx_m, dz_m), intervals_s, binary=moving)


def model_length_m(value_m: float) -> float:
    """A grid spacing or a level height as the model holds it.  CM1 reads dx, dy and dz into a 4-byte REAL, and
    GM1's start takes only a length that word holds exactly; a typed 82.9 m is therefore the nearest such length,
    82.9000015258789 m, which is what CM1 itself makes of 82.9.  67, 125 and 62.5 m are held exactly and stay as
    they are.  A value that cannot be a spacing at all (under a metre, or not a finite number) is returned as it
    is, for the refusal that names it."""

    value = float(value_m)
    if not (math.isfinite(value) and 1.0 <= value < 1.0e7):
        return value
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def cells_across(length_m: float, dx_m: float) -> int:
    """The cells a side of ``length_m`` takes at ``dx_m``: the exact count when that is whole (120 km at 250 m is
    480), otherwise the next whole number up (at 67 m it is 1792, at 3350 m it is 36), never fewer than 8.  The area
    is then never smaller than the one asked for, and larger by less than one cell: a storm set up to just stay
    inside its area still does.  The model itself takes any count from 7; only the area slider and the fit's
    shrinking move in steps of 8 from here."""

    exact = length_m / dx_m
    whole = round(exact)
    return max(MIN_CELLS, int(whole if abs(exact - whole) < 1e-6 else math.ceil(exact)))


def levels_for(top_m: float, dz_m: float) -> int:
    """The levels a top of ``top_m`` takes at ``dz_m``: the nearest whole number (the top is then ``nz * dz``, less
    than half a level from ``top_m``)."""

    return int(round(top_m / dz_m))


def levels_bounds(top_m: float) -> tuple[float, float]:
    """The finest level height (to 0.1 m) and the thickest (to 1 m) whose count under ``top_m`` is within
    :data:`LEVELS_RANGE`, as :func:`levels_for` counts (the nearest whole number): 78 m and 2666 m under 20 km."""

    low, high = LEVELS_RANGE
    finest = math.ceil(top_m / (high + 0.5) * 10.0) / 10.0
    while levels_for(top_m, finest) > high:
        finest = round(finest + 0.1, 1)
    thickest = float(math.floor(top_m / (low - 0.5)))
    while thickest > 1.0 and levels_for(top_m, thickest) < low:
        thickest -= 1.0
    return finest, thickest


def _whole(value: float) -> bool:
    return abs(value - round(value)) < 1e-6


def hard_floor_km(effective: Mapping[str, Any], settings: Mapping[str, Any], dx_m: float) -> float:
    """The smallest area side a storm can be started in at ``dx_m``: 16 km, the warm bubble's own width when that
    is more (iinit = 1: twice its radius), and never fewer than 8 cells.  Below it nothing is offered; between it
    and the area the preset was set up for, the run is offered and the fit says what the edges do."""

    return max(HARD_FLOOR_KM, bubble_width_km(effective, settings), MIN_CELLS * dx_m / 1000.0)


def bubble_width_km(effective: Mapping[str, Any], settings: Mapping[str, Any]) -> float:
    """The area side the warm bubble itself needs (iinit = 1; 0 otherwise): the whole bubble has to lie inside the
    area (init3d.F builds a truncated one otherwise, and the run's own start refuses it), so its radius on both
    sides of its centre, which a preset may place off the area's centre (iorigin = 2: an offset from the centre,
    kept when the area shrinks)."""

    if _int(effective, "iinit") != 1:
        return 0.0
    reach = float(settings.get("initialization.bhrad_m", CM1_BHRAD_M))
    if _int(effective, "iorigin", 1) == 2:
        reach += max(abs(float(settings.get("initialization.bubble_center_x_m", 0.0) or 0.0)),
                     abs(float(settings.get("initialization.bubble_center_y_m", 0.0) or 0.0)))
    return 2.0 * reach / 1000.0


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
                  and lands(value * 60.0, dt_s))


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
    settings = dict(preset.record.get("gm1_settings") or {})
    comfortable = _floor_km(preset, dx, parcels)
    floor = max(hard_floor_km(effective, settings, dx), PARCELS_FLOOR_KM if parcels else 0.0)
    # Never above the area the preset was set up for, unless the bubble itself needs more.
    floor = max(bubble_width_km(effective, settings), min(floor, comfortable))
    aspect = short_cells / long_cells if long_cells else 1.0
    largest = _largest_cells()
    top_m = nz * dz
    knobs: list[dict[str, Any]] = []

    def knob(knob_id: str, default: Any, choices: list[dict[str, Any]] | None = None, **more: Any) -> None:
        group = dict(KNOBS)[knob_id]
        knobs.append({"id": knob_id, "group": group, "default": default, "choices": choices or [], **more})

    # The grid: any spacing.  "choices" are the quick picks; the slider runs between "slider_m" on a log scale and
    # a typed value may be anything that leaves at least 8 cells across the smallest area ("max_m").
    grids = sorted(set(GRIDS_M) | {dx})
    knob("grid", dx, [_choice(value) for value in grids], unit="m", free=True,
         slider_m=[min(GRID_SLIDER_M[0], dx), max(GRID_SLIDER_M[1], dx)],
         max_m=long_km * (aspect or 1.0) * 1000.0 / MIN_CELLS, own_m=dx)
    if not stretched:
        # The levels: any height that gives 8 to 256 levels under the preset's top.
        fits = [value for value in sorted(set(LEVELS_M) | {dz})
                if LEVELS_RANGE[0] <= levels_for(top_m, value) <= LEVELS_RANGE[1]]
        finest_dz, thickest_dz = levels_bounds(top_m)
        knob("levels", dz, [_choice(value) for value in fits], unit="m", top_m=top_m, free=True,
             min_m=finest_dz, max_m=thickest_dz, levels_range=list(LEVELS_RANGE))
    knob("area", round(long_km, 3), unit="km", min_km=floor / aspect if aspect else floor,
         max_km=max(long_km, min(2.0 * long_km, AREA_MAX_KM)), step_cells=AREA_STEP_CELLS, aspect=aspect,
         floor_km=floor, comfortable_km=comfortable, hard_floor_km=HARD_FLOOR_KM,
         bubble_floor_km=hard_floor_km(effective, settings, 0.0),
         parcels_floor_km=PARCELS_FLOOR_KM, largest_cells=largest, nz=nz)
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
        "grid": {"dx_m": dx, "dz_m": dz, "nz": nz, "dt_s": dt, "top_m": nz * dz, "nx": nx, "ny": ny,
                 "duration_s": timax, "output_interval_s": tapfrq,
                 "dt_per_dx_s_per_m": DT_PER_DX_S_PER_M, "dt_per_dz_s_per_m": DT_PER_DZ_S_PER_M,
                 "cadence_s": CADENCE_S, "usual_times_s": list(_USUAL_TIMES_S)},
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


def _words_exact_m(value_m: float, other_m: float) -> str:
    """``value_m`` in words that differ from ``other_m``'s: 1000.0001 m beside 1 km is "1.0000001 km", not a second
    "1 km"."""

    if _words_m(value_m) != _words_m(other_m):
        return _words_m(value_m)
    return f"{value_m / 1000.0:.8g} km" if value_m >= 1000.0 else f"{value_m:.8g} m"


def _words_s(seconds: float) -> str:
    """A time step as it is: 0.5 s, 0.390625 s and 0.5859375 s in full (short binary fractions are exact in a few
    digits), anything longer to six figures."""

    text = repr(float(seconds))
    if text.endswith(".0"):
        text = text[:-2]
    return f"{text} s" if len(text) <= 10 else f"{seconds:g} s"


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


def _grid_value(knob: Mapping[str, Any], dx_m: float) -> None:
    """Any spacing that can be a grid here: positive, and at least 8 cells across the preset's area."""

    if not dx_m > 0.0:
        raise CustomError(f"The grid spacing is a length in metres above zero, not {dx_m:g}")
    limit = float(knob.get("max_m") or 0.0)
    if limit and dx_m > limit + 1e-6:
        raise CustomError(f"A {_words_m(dx_m)} grid leaves fewer than {MIN_CELLS} cells across this storm's area, too "
                          f"few to be a grid; the coarsest grid here is {_words_m(limit)}.",
                          suggest={"grid": math.floor(limit)})


def _levels_value(knob: Mapping[str, Any], dz_m: float) -> int:
    """The levels ``dz_m`` gives under the preset's top, or why the model cannot take them and what it can."""

    if not dz_m > 0.0:
        raise CustomError(f"The height of a level is a length in metres above zero, not {dz_m:g}")
    top = float(knob["top_m"])
    count = levels_for(top, dz_m)
    low, high = LEVELS_RANGE
    finest, thickest = levels_bounds(top)
    if count > high:
        raise CustomError(f"Levels every {_words_m(dz_m)} up to {_words_m(top)} would be {count} levels, and the model "
                          f"is built for at most {high}; the finest levels here are every {_words_m(finest)}.",
                          suggest={"levels": finest})
    if count < low:
        raise CustomError(f"Levels every {_words_m(dz_m)} up to {_words_m(top)} would be {count} levels, and the model "
                          f"needs at least {low}; the thickest levels here are every {_words_m(thickest)}.",
                          suggest={"levels": thickest})
    return count


def spacing_notes(dx_m: float, dx0_m: float) -> list[str]:
    """The one-line notes a spacing outside what GM1 ships carries (it still runs)."""

    if math.isclose(dx_m, dx0_m):
        return []
    low, high = PRESET_SPACINGS_M
    if dx_m < low - 1e-6:
        return [f"A {_words_m(dx_m)} grid is finer than any storm GM1 ships ({_words_m(low)}), and finer than anything "
                "compared with CM1 (that comparison is at 1 km)."]
    if dx_m > high + 1e-6:
        return [f"A {_words_m(dx_m)} grid is coarser than any storm GM1 ships ({_words_m(high)}): a storm is only a "
                "few cells wide, so expect a weaker, smoother one."]
    return []


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

    # The storm time and the picture interval first (their sentences are added below, in the panel's order): the
    # time step has to land on both.
    time_notes: list[tuple[str, Mapping[str, Sequence[Any]], str]] = []
    timax = timax0
    if "time" in custom:
        _offered_value(by_id["time"], float(custom["time"]), f"{_words_min(float(custom['time']))} of storm time")
        timax = float(custom["time"]) * 60.0
        pinned.add("time")
        if not math.isclose(timax, timax0):
            values["timax"] = timax
            time_notes.append(("time", {"timax": [timax0, timax]},
                               f"{_words_min(timax / 60.0)} of storm time instead of {_words_min(timax0 / 60.0)}."))
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
            time_notes.append(("pictures", {"tapfrq": [tapfrq0, tapfrq]},
                               f"A saved picture every {_words_min(tapfrq / 60.0)} instead of every "
                               f"{_words_min(tapfrq0 / 60.0)}."))
    if not _whole(timax / tapfrq):
        raise CustomError(f"A picture every {_words_min(tapfrq / 60.0)} does not split {_words_min(timax / 60.0)} of "
                          "storm time into whole pictures; pick another picture interval")
    # The grid, the levels and the time step that follows them: any spacing, the step made to land on the output
    # times (snapped_time_step), the levels a whole number under the preset's top.
    dx = model_length_m(custom["grid"]) if "grid" in custom else dx0
    dz = model_length_m(custom["levels"]) if "levels" in custom else dz0
    if "grid" in custom:
        _grid_value(by_id["grid"], dx)
    nz = nz0
    if "levels" in custom and not math.isclose(dz, dz0):
        nz = _levels_value(by_id["levels"], dz)
    if dx < 1.0:
        # Under a metre the smallest area a storm starts in is past what one saved file holds whatever the levels
        # (said here, before a time step or a cell count is formed: at 1e-300 m the count is past what a number
        # holds).
        least_km = float(by_id["area"]["floor_km"])
        finest = math.ceil(least_km * 1000.0 * math.sqrt(nz / START_CELLS_MAX))
        raise CustomError(
            f"A {_words_m(dx)} grid over the smallest area a storm starts in ({least_km:g} km) is far more cells "
            f"than one restart file holds (about {START_CELLS_MAX // 1_000_000} million); the finest grid here is "
            f"about {finest:g} m.", suggest={"grid": float(finest)})
    dt = dt0
    if not math.isclose(dx, dx0) or not math.isclose(dz, dz0):
        rstfrq = _float(effective, "rstfrq")
        # A domain that moves (imove = 1) takes a short binary fraction, so the run's sum of its position is exact.
        dt = snapped_time_step(dx, dz, [timax, tapfrq, *([rstfrq] if rstfrq > 0.0 else [])],
                               moving=_int(effective, "imove") == 1)
    if not math.isclose(dz, dz0):
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
    # The smallest area a storm can start in (16 km, or the bubble's own width), not the area the preset was set
    # up for: between the two the run is offered, and the fit says what the edges do to it.
    bubble_m = (float(custom["bubble"]) * 1000.0 if "bubble" in custom
                else float(settings0.get("initialization.bhrad_m", CM1_BHRAD_M)))
    hard_km = max(hard_floor_km(effective, {**settings0, "initialization.bhrad_m": bubble_m}, dx),
                  PARCELS_FLOOR_KM if parcels_on else 0.0)
    comfortable_km = _floor_km(preset, dx, parcels_on)
    floor_km = max(bubble_width_km(effective, {**settings0, "initialization.bhrad_m": bubble_m}),
                   min(hard_km, comfortable_km))
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
    # Whole cells: the count across the preset's own area, rounded up (cells_across).
    long_cells = cells_across(long0, dx)
    if "area" in custom:
        # The area moves in steps of 8 cells from the preset's own size (the slider's steps, the fit's own step).
        # An area that is a whole number of cells (to the metre: the area a fit came to, handed back as the user's
        # own) is kept cell for cell, so "keep this area" keeps it.
        base, asked = long_cells, long_m / dx
        if abs(asked - round(asked)) <= min(0.25, 0.5 / dx + 1e-6):
            long_cells = max(MIN_CELLS, int(round(asked)))
        else:
            long_cells = base + AREA_STEP_CELLS * int(round((asked - base) / AREA_STEP_CELLS))
        while long_cells * dx > by_id["area"]["max_km"] * 1000.0 + 1e-6 and long_cells > AREA_STEP_CELLS:
            long_cells -= AREA_STEP_CELLS
        while long_cells * dx * aspect < floor_km * 1000.0 - 1e-6:
            long_cells += AREA_STEP_CELLS
    if nx0 == ny0:
        nx = ny = long_cells
    elif "area" not in custom:
        nx, ny = cells_across(nx0 * dx0, dx), cells_across(ny0 * dx0, dx)
    elif nx0 > ny0:
        nx, ny = long_cells, max(MIN_CELLS, int(round(long_cells * ny0 / nx0)))
    else:
        nx, ny = max(MIN_CELLS, int(round(long_cells * nx0 / ny0))), long_cells
    if "area" in custom and float(nx) * float(ny) * nz > FRAME_CELLS_MAX:
        # An area of the user's own that one saved frame cannot hold at this grid and these levels runs on no
        # card: say so, with the largest area sure to be saved (in the slider's own steps of 8 cells; what one
        # restart file holds, which is less than a frame).  Between the two the fit prices the restart itself.
        cells = float(nx) * float(ny) * nz
        most = math.sqrt(START_CELLS_MAX / (nz * aspect)) if aspect else 0.0
        fits = long_cells - AREA_STEP_CELLS * int(math.ceil((long_cells - most) / AREA_STEP_CELLS))
        if fits * dx * aspect >= floor_km * 1000.0 - 1e-6:
            raise CustomError(
                f"An area {max(nx, ny) * dx / 1000.0:.4g} km wide is {cells / 1e6:,.0f} million cells at a "
                f"{_words_m(dx)} grid with {nz} levels, more than one saved picture of the storm holds (about "
                f"{FRAME_CELLS_MAX // 1_000_000} million). An area up to {fits * dx / 1000.0:.4g} km wide can be "
                "saved.",
                suggest={"area": round(fits * dx / 1000.0, 3)})
        finest = math.ceil(floor_km * 1000.0 * math.sqrt(nz * aspect / START_CELLS_MAX) / (aspect or 1.0))
        raise CustomError(
            f"A {_words_m(dx)} grid over the smallest area a storm starts in ({floor_km:g} km) is more cells with "
            f"these levels than one restart file holds (about {START_CELLS_MAX // 1_000_000} million); the finest "
            f"grid here is about {finest:g} m.", suggest={"grid": float(finest)})
    # A fine grid over the preset's whole area can be more cells than one saved frame holds: the area then starts
    # from the most that does, and the fit sizes it to the card from there.
    start_from: tuple[int, int] | None = None
    if "area" not in custom and not math.isclose(dx, dx0) and nx * ny * nz > START_CELLS_MAX:
        scale = math.sqrt(START_CELLS_MAX / (nx * ny * nz))
        start_from = (nx, ny)
        nx = max(MIN_CELLS, AREA_STEP_CELLS * int(nx * scale / AREA_STEP_CELLS))
        ny = max(MIN_CELLS, AREA_STEP_CELLS * int(ny * scale / AREA_STEP_CELLS))
        # Never below the smallest area a storm starts in: whether that area can be saved and fits the card is
        # then the guided check's and the fit's to say, with their own reasons.
        least = AREA_STEP_CELLS * int(math.ceil(floor_km * 1000.0 / dx / AREA_STEP_CELLS - 1e-9))
        nx, ny = max(nx, min(start_from[0], least)), max(ny, min(start_from[1], least))
        if float(nx) * float(ny) * nz > FRAME_CELLS_MAX:
            # Not even the smallest area can be saved at this spacing, on any card: say so, and what can.  (In
            # floating point from the area itself, so even a spacing of 1e-300 m is a sentence, not an overflow.)
            side_x, side_y = min(start_from[0], least) * dx, min(start_from[1], least) * dx
            finest = math.ceil(math.sqrt(side_x * side_y * nz / START_CELLS_MAX))
            cells = side_x / dx * (side_y / dx) * nz
            many = f"{cells / 1e6:,.0f} million" if math.isfinite(cells) and cells < 1e15 else "far more"
            raise CustomError(
                f"A {_words_m(dx)} grid over the smallest area a storm starts in ({floor_km:g} km) is "
                f"{many} cells with these levels, and one restart file holds about "
                f"{START_CELLS_MAX // 1_000_000} million; the finest grid here is about {finest:g} m.",
                suggest={"grid": float(finest)})
    if min(nx, ny) * dx < floor_km * 1000.0 - 1e-6:
        raise CustomError(f"The area must be at least {floor_km:g} km across"
                          f"{_parcels_why(preset, dx, parcels_on, floor_km)}; "
                          f"{min(nx, ny) * dx / 1000.0:g} km is too small.{_parcels_fix(by_id, parcels_on, floor_km)}")
    if "grid" in custom and not math.isclose(dx, dx0):
        values.update(dx=dx, dy=dx, dtl=dt)
        keys: dict[str, list[Any]] = {"dx": [dx0, dx], "dy": [dx0, dx], "dtl": [dt0, dt]}
        if "area" not in custom:
            keys.update(nx=[nx0, nx], ny=[ny0, ny])
        step = "" if math.isclose(dt, dt0) else f" (and a {_words_s(dt)} time step instead of {_words_s(dt0)})"
        start = "" if start_from is None else (
            f" At this spacing the preset's whole area would be {start_from[0] * start_from[1] * nz / 1e6:,.0f} "
            f"million cells, more than one restart file holds (about 60 million), so the area starts at "
            f"{nx * dx / 1000.0:.0f} × {ny * dx / 1000.0:.0f} km and the fit sizes it to your card from there.")
        # Whole cells make the area up to a cell larger, which is a lot at a coarse spacing: say the area it comes to.
        long_now = max(nx, ny) * dx
        moved = "" if start_from is not None or "area" in custom or abs(long_now - long0) <= 0.01 * long0 else (
            f" Whole cells make the area {nx * dx / 1000.0:g} km wide instead of {nx0 * dx0 / 1000.0:g} km."
            if nx == ny else
            f" Whole cells make the area {nx * dx / 1000.0:g} × {ny * dx / 1000.0:g} km instead of "
            f"{nx0 * dx0 / 1000.0:g} × {ny0 * dx0 / 1000.0:g} km.")
        note("grid", keys, f"Grid spacing {_words_exact_m(dx, dx0)} instead of {_words_m(dx0)}{step}.{moved}{start}")
    elif not math.isclose(dt, dt0):
        values.update(dtl=dt)
        customized[-1]["keys"]["dtl"] = [dt0, dt]
        customized[-1]["sentence"] = customized[-1]["sentence"][:-1] + f", with a {_words_s(dt)} time step instead of {_words_s(dt0)}."
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
    # Storm time and pictures (read above; their sentences come here, after the size's).
    for knob, keys, sentence in time_notes:
        note(knob, keys, sentence)
    if not lands(tapfrq, dt) or not lands(timax, dt):
        raise CustomError(f"A picture every {_words_min(tapfrq / 60.0)} does not land on a whole number of "
                          f"{_words_s(dt)} time steps; pick another picture interval")
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
    # What the fit may go down to with the user's own choices (the smallest area a storm can start in), and the
    # one-line notes an unusual spacing carries.
    record["hard_min_width_km"] = floor_km
    if start_from is not None:
        # The area starts at the most one restart file holds, not at the preset's own: the fit's sentence and a
        # refusal for a storm that leaves it say so.
        record["start_capped"] = True
    else:
        record.pop("start_capped", None)
    notes = [*list(record.get("custom_notes") or []), *spacing_notes(dx, dx0)]
    if notes:
        record["custom_notes"] = list(dict.fromkeys(notes))
    record["custom"] = {**dict(record.get("custom") or {}), **dict(custom)}
    record["custom_sets"] = {**dict(record.get("custom_sets") or {}), **sets}
    if seed is not None:
        record["random_seed"] = seed
    else:
        record.pop("random_seed", None)
    changed = replace(preset, record=record, namelist_text=text, namelist_file=folder / "namelist.input",
                      namelist_original=preset.namelist_original_text)
    if values.keys() & {"dx", "nx", "ny", "timax", "dz"} and start_from is None:
        # A longer storm or a smaller area can take the storm out of its area: the view then follows the storm with
        # the storm-track check's own frame and bubble (one more "customized" sentence), as a new sounding does.
        # (Not for a fine grid whose area only starts here: the fit sizes it to the card, and a frame worked out
        # for the starting area would place the bubble where a smaller area cannot hold it.)
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
                          "6 s per 1 km across and per 500 m up, made to land on the output times), so use "
                          "--custom grid=M or --custom levels=M.")
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
           "refuse_set", "rotation_sentence", "time_step", "snap_time_step", "snapped_time_step", "binary_steps",
           "cells_across",
           "levels_for", "lands", "hard_floor_km", "spacing_notes"]
