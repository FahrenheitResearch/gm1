"""The run settings a user may change that a CM1 namelist does not hold.

CM1 keeps the warm bubble, the cold pool, the line thermal and the parcel
seed box in Fortran (``init3d.F``, ``parcel.F``), so a CM1 user edits source
to move them.  GM1 keeps them in the run directory's ``settings.toml``, beside
the output choices the importer records, and every import of the run's
namelist carries them into ``deck.toml``.

:data:`EDITABLE` is the whole list.  Each entry names its GM1 key (a dotted
``[table] key`` of the gpuwm-cm1 deck, or ``output.also_netcdf`` for the
writer), CM1's own word for it, its unit, and the ``iinit`` (or ``iprcl``)
arm that reads it.  Nothing outside the list is accepted from
``settings.toml`` or from ``gpuwm-cm1 edit --setting``.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import tomllib
from pathlib import Path
from typing import Any, Mapping


class SettingError(ValueError):
    """A setting that is not on the list, or a value it cannot take; the message says which."""


@dataclass(frozen=True)
class Editable:
    key: str             # "initialization.zc_m"
    label: str           # plain words
    cm1: str             # CM1's own word (init3d.F / parcel.F), shown as a hint
    kind: str            # "float" | "int" | "bool"
    unit: str = ""       # the unit the value is stored in (m, K, count)
    minimum: float | None = None
    maximum: float | None = None
    positive: bool = False
    #: Which group(s) of the Settings screen show it (bubble, coldpool, line, nudging, convergence, uforce,
    #: parcels, output, gpu).
    groups: tuple[str, ...] = ()
    #: The gpuwm-cm1 deck key it sets, when that is not ``key`` (settings.toml keeps a user's
    #: choice in its own table, apart from the values the importer chooses itself).
    deck_key: str | None = None

    @property
    def table(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.key.split(".", 1)[1]

    @property
    def deck(self) -> str | None:
        """The deck key it sets, or None for a value only the writer reads."""

        if self.deck_key is not None:
            return self.deck_key
        return self.key if self.table in ("initialization", "lagrangian") else None


def _e(key: str, label: str, cm1: str, kind: str, unit: str = "", *, groups: tuple[str, ...],
       minimum: float | None = None, maximum: float | None = None, positive: bool = False,
       deck_key: str | None = None) -> Editable:
    return Editable(key, label, cm1, kind, unit, minimum, maximum, positive, groups, deck_key)


#: Every setting a user may change outside the namelist, in Settings-screen order.
EDITABLE: tuple[Editable, ...] = (
    _e("initialization.bubble_center_x_m", "Centre, x", "ric", "float", "m", groups=("bubble", "benchmark")),
    _e("initialization.bubble_center_y_m", "Centre, y", "rjc", "float", "m", groups=("bubble",)),
    _e("initialization.zc_m", "Centre height", "zc", "float", "m", groups=("bubble", "line", "benchmark"),
       minimum=0.0, maximum=50000.0),
    _e("initialization.bhrad_m", "Horizontal radius", "bhrad", "float", "m", groups=("bubble", "line", "benchmark"),
       positive=True, maximum=1.0e6),
    _e("initialization.bvrad_m", "Vertical radius", "bvrad", "float", "m", groups=("bubble", "line", "benchmark"),
       positive=True, maximum=50000.0),
    _e("initialization.bptpert_k", "Amplitude", "bptpert", "float", "K",
       groups=("bubble", "coldpool", "line", "benchmark"), minimum=-30.0, maximum=30.0),
    # base.F:267-268, isnd = 4 (CM1 keeps these in its code; init3d.F:625-645 reads qt_mb again).
    _e("initialization.thec_mb_k", "Wet equivalent potential temperature", "thec_mb", "float", "K",
       groups=("benchmark_sounding",), minimum=200.0, maximum=400.0),
    _e("initialization.qt_mb", "Total water", "qt_mb", "float", "kg/kg", groups=("benchmark_sounding",),
       positive=True, maximum=0.1),
    _e("initialization.zdep_m", "Depth", "zdep", "float", "m", groups=("coldpool",), positive=True,
       maximum=50000.0),
    _e("initialization.random_line_amplitude_k", "Noise amplitude", "(random perturbation amplitude)", "float", "K",
       groups=("line",), minimum=0.0, maximum=10.0),
    _e("initialization.maintain_rh", "Keep relative humidity", "maintain_rh", "bool",
       groups=("bubble", "coldpool", "line")),
    # init3d.F:1370-1396, iinit = 12 (CM1 keeps these in its code).
    _e("initialization.xc_wnudge", "Centre, x", "xc_wnudge", "float", "m", groups=("nudging",)),
    _e("initialization.yc_wnudge", "Centre, y", "yc_wnudge", "float", "m", groups=("nudging",)),
    _e("initialization.zc_wnudge", "Centre height", "zc_wnudge", "float", "m", groups=("nudging",),
       minimum=0.0, maximum=50000.0),
    _e("initialization.xr_wnudge", "Radius, x", "xr_wnudge", "float", "m", groups=("nudging",), positive=True,
       maximum=1.0e6),
    _e("initialization.yr_wnudge", "Radius, y", "yr_wnudge", "float", "m", groups=("nudging",), positive=True,
       maximum=1.0e6),
    _e("initialization.zr_wnudge", "Radius, z", "zr_wnudge", "float", "m", groups=("nudging",), positive=True,
       maximum=50000.0),
    _e("initialization.wmax_wnudge", "Target updraft", "wmax_wnudge", "float", "m/s", groups=("nudging",),
       minimum=0.0, maximum=100.0),
    _e("initialization.alpha_wnudge", "Rate", "alpha_wnudge", "float", "1/s", groups=("nudging",),
       minimum=0.0, maximum=10.0),
    _e("initialization.t1_wnudge", "Starts to fade at", "t1_wnudge", "float", "s", groups=("nudging",),
       minimum=0.0),
    _e("initialization.t2_wnudge", "Stops at", "t2_wnudge", "float", "s", groups=("nudging",), positive=True),
    # init3d.F:1303-1333, iinit = 9.
    _e("initialization.dmax", "Divergence", "Dmax", "float", "1/s", groups=("convergence",),
       minimum=-1.0, maximum=1.0),
    _e("initialization.zdeep", "Depth", "zdeep", "float", "m", groups=("convergence",), positive=True,
       maximum=50000.0),
    _e("initialization.lamx", "Width, x", "lamx", "float", "m", groups=("convergence",), positive=True,
       maximum=1.0e6),
    _e("initialization.lamy", "Width, y", "lamy", "float", "m", groups=("convergence",), positive=True,
       maximum=1.0e6),
    _e("initialization.xcent", "Centre, x", "xcent", "float", "m", groups=("convergence",)),
    _e("initialization.ycent", "Centre, y", "ycent", "float", "m", groups=("convergence",)),
    _e("initialization.convtime", "Stops at", "convtime", "float", "s", groups=("convergence",), minimum=0.0),
    # init3d.F:1334-1347, iinit = 10.
    _e("initialization.xc_uforce", "Centre, x", "xc_uforce", "float", "m", groups=("uforce",)),
    _e("initialization.xr_uforce", "Radius, x", "xr_uforce", "float", "m", groups=("uforce",), positive=True,
       maximum=1.0e6),
    _e("initialization.zr_uforce", "Depth scale", "zr_uforce", "float", "m", groups=("uforce",), positive=True,
       maximum=50000.0),
    _e("initialization.alpha_uforce", "Strength", "alpha_uforce", "float", "m/s2", groups=("uforce",),
       minimum=-10.0, maximum=10.0),
    _e("initialization.t1_uforce", "Starts to fade at", "t1_uforce", "float", "s", groups=("uforce",), minimum=0.0),
    _e("initialization.t2_uforce", "Stops at", "t2_uforce", "float", "s", groups=("uforce",), positive=True),
    _e("lagrangian.seed_nx", "Parcels across (x)", "nparcels (x)", "int", "parcels", groups=("parcels",),
       minimum=1, maximum=4096),
    _e("lagrangian.seed_ny", "Parcels across (y)", "nparcels (y)", "int", "parcels", groups=("parcels",),
       minimum=1, maximum=4096),
    _e("lagrangian.seed_nz", "Parcels up (z)", "nparcels (z)", "int", "parcels", groups=("parcels",),
       minimum=1, maximum=1024),
    _e("lagrangian.seed_dx_m", "Spacing, x", "(parcel spacing)", "float", "m", groups=("parcels",), positive=True),
    _e("lagrangian.seed_dy_m", "Spacing, y", "(parcel spacing)", "float", "m", groups=("parcels",), positive=True),
    _e("lagrangian.seed_dz_m", "Spacing, z", "(parcel spacing)", "float", "m", groups=("parcels",), positive=True),
    _e("lagrangian.seed_x0_m", "First parcel, x (from the west edge)", "(first parcel)", "float", "m",
       groups=("parcels",), minimum=0.0),
    _e("lagrangian.seed_y0_m", "First parcel, y (from the south edge)", "(first parcel)", "float", "m",
       groups=("parcels",), minimum=0.0),
    _e("lagrangian.seed_z0_m", "First parcel, height above the lowest level", "(first parcel)", "float", "m",
       groups=("parcels",), minimum=0.0),
    _e("output.also_netcdf", "Also write netCDF", "(GM1: beside CM1's GrADS files)", "bool", groups=("output",)),
    # GM1's own run setting (CM1 has none): the most GPU memory the run's device registry may take.  The
    # importer writes 8192 MiB; the estimate recommends the priced peak with room.  Kept in
    # [gpu] so the importer's own [runtime] choices are never read back as the user's.
    _e("gpu.memory_budget_mib", "GPU memory budget", "GM1: runtime.memory_budget_mib", "int", "MiB",
       groups=("gpu",), minimum=256, maximum=1048576, deck_key="runtime.memory_budget_mib"),
)

BY_KEY: Mapping[str, Editable] = {item.key: item for item in EDITABLE}

#: Declared together or not at all (config.py's _BUBBLE_CENTER_KEYS).
PAIRS: tuple[tuple[str, str], ...] = (
    ("initialization.bubble_center_x_m", "initialization.bubble_center_y_m"),
)

#: Tables of settings.toml whose values the translator carries into deck.toml (the rest are the writer's).
DECK_TABLES = frozenset({"initialization", "lagrangian", "gpu"})


def parse_value(item: Editable, value: Any) -> Any:
    """``value`` (from TOML, JSON or the command line) as this setting's type, or SettingError."""

    if isinstance(value, str):
        text = value.strip()
        if item.kind == "bool":
            lowered = text.lower()
            if lowered in ("true", ".true.", "t", "1", "on", "yes"):
                return True
            if lowered in ("false", ".false.", "f", "0", "off", "no"):
                return False
            raise SettingError(f"{item.key} takes true or false, not {value!r}")
        try:
            value = int(text) if item.kind == "int" else float(text.replace("d", "e").replace("D", "e"))
        except ValueError as error:
            raise SettingError(f"{item.key} takes a {'whole ' if item.kind == 'int' else ''}number, not {value!r}") from error
    if item.kind == "bool":
        if isinstance(value, bool):
            return value
        raise SettingError(f"{item.key} takes true or false, not {value!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingError(f"{item.key} takes a number, not {value!r}")
    if item.kind == "int":
        if isinstance(value, float):
            if not value.is_integer():
                raise SettingError(f"{item.key} takes a whole number, not {value!r}")
            value = int(value)
    else:
        value = float(value)
        if not math.isfinite(value):
            raise SettingError(f"{item.key} takes a finite number, not {value!r}")
    unit = f" {item.unit}" if item.unit and item.unit != "parcels" else ""
    if item.positive and not value > 0:
        raise SettingError(f"{item.key} must be larger than 0{unit}, not {value!r}")
    if item.minimum is not None and value < item.minimum:
        raise SettingError(f"{item.key} must be at least {item.minimum:g}{unit}, not {value!r}")
    if item.maximum is not None and value > item.maximum:
        raise SettingError(f"{item.key} must be at most {item.maximum:g}{unit}, not {value!r}")
    return value


def check(values: Mapping[str, Any]) -> dict[str, Any]:
    """Typed, validated user settings; SettingError names the first bad one."""

    out: dict[str, Any] = {}
    for key, value in values.items():
        item = BY_KEY.get(key)
        if item is None:
            known = ", ".join(sorted(BY_KEY))
            raise SettingError(f"{key} is not a setting GM1 keeps outside the namelist; the settings are: {known}")
        out[key] = parse_value(item, value)
    for first, second in PAIRS:
        if (first in out) != (second in out):
            missing = second if first in out else first
            raise SettingError(
                f"{first} and {second} are set together (CM1's ric and rjc); give {missing} too, "
                "or reset both to put the bubble back at the domain centre"
            )
    return out


def read_file(path: Path) -> dict[str, Any]:
    """The user settings stored in a run directory's ``settings.toml`` (the importer's own choices left out).

    An unreadable file or a value off the list reads as nothing: a re-import
    then takes CM1's own values, and the Settings screen shows them.
    """

    try:
        document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return {}
    found: dict[str, Any] = {}
    for item in EDITABLE:
        table = document.get(item.table)
        if isinstance(table, Mapping) and item.name in table:
            found[item.key] = table[item.name]
    if "output.also_netcdf" in found and found["output.also_netcdf"] is False:
        del found["output.also_netcdf"]  # the importer writes false itself; it is not a user choice
    try:
        return check(found)
    except SettingError:
        kept: dict[str, Any] = {}
        for key, value in found.items():
            try:
                kept[key] = parse_value(BY_KEY[key], value)
            except SettingError:
                continue
        for first, second in PAIRS:
            if (first in kept) != (second in kept):
                kept.pop(first, None)
                kept.pop(second, None)
        return kept


def merge(stored: Mapping[str, Any], changes: Mapping[str, Any], resets: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """``stored`` with ``changes`` applied and ``resets`` removed, validated as a whole."""

    merged = dict(stored)
    for key in resets:
        if key not in BY_KEY:
            raise SettingError(f"{key} is not a setting GM1 keeps outside the namelist")
        merged.pop(key, None)
        for first, second in PAIRS:
            if key in (first, second):
                merged.pop(first, None)
                merged.pop(second, None)
    for key, value in changes.items():
        if key not in BY_KEY:
            raise SettingError(f"{key} is not a setting GM1 keeps outside the namelist")
        merged[key] = value
    return check(merged)


def deck_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    """The settings the translator writes into deck.toml, by their dotted gpuwm-cm1 deck keys."""

    return {BY_KEY[key].deck: value for key, value in values.items() if BY_KEY[key].deck is not None}


def as_tables(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for key, value in values.items():
        item = BY_KEY[key]
        tables.setdefault(item.table, {})[item.name] = value
    return tables


__all__ = [
    "BY_KEY", "DECK_TABLES", "EDITABLE", "Editable", "PAIRS", "SettingError", "as_tables", "check",
    "deck_overrides", "merge", "parse_value", "read_file",
]
