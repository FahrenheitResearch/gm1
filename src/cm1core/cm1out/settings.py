"""The CM1 settings a cm1out file is written for, and the grid it is written on.

Two sources, in this order:

1. the run's CM1 namelist (``RUNDIR/namelist.input``, or ``--namelist``),
   read by the importer exactly as CM1 r21.1 would read it: every r21.1 key,
   the file's value or r21.1's own default.  This is what decides the output
   format and file layout (``output_format``/``output_filetype``), which
   variables CM1 would write (``&param9``), the statistics (``&param10``),
   the units of distance (``outunits``) and the global attributes.
2. without a namelist, the GM1 deck the run was started from
   (``RUNDIR/deck.toml``, ``--deck``, or the configuration named by the
   event log): its ``[output]``/``[diagnostics]`` selection and the CM1 keys
   it carries under their own names, over r21.1's defaults.

The grid itself -- nx, ny, nz, dx, dy, the w-face heights, the moisture
species -- is always read from the frames, which are what is being written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# CM1 r21.1 param.F, the keys the writer and the statistics read.
FORMAT_NAMES = {1: "grads", 2: "nc"}


class SettingsError(ValueError):
    """The run's CM1 settings cannot be read."""


@dataclass(frozen=True)
class Cm1Settings:
    """Every r21.1 key the writer reads, after param.F's own adjustments."""

    values: Mapping[str, Any]
    #: Where the values came from, for the file's provenance attributes.
    source: str
    source_path: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def int(self, key: str, default: int = 0) -> int:
        value = self.values.get(key, default)
        return default if value is None else int(value)

    def float(self, key: str, default: float = 0.0) -> float:
        value = self.values.get(key, default)
        return default if value is None else float(value)

    @property
    def output_format(self) -> str:
        return FORMAT_NAMES.get(self.int("output_format", 1), "nc")

    @property
    def output_filetype(self) -> int:
        return self.int("output_filetype", 1)

    @property
    def distance_units(self) -> tuple[str, float]:
        """``aunit`` and ``outunitconv`` (param.F:4760-4768)."""

        if self.int("outunits", 1) == 2:
            return "meters", 1.0
        return "km", 0.001


def _param_f_adjustments(values: dict[str, Any]) -> dict[str, Any]:
    """The start-up rewrites param.F applies to keys the file reports.

    Each is CM1's own rule, quoted by line; a key CM1 rewrites is written as
    CM1 would have written it, not as the namelist spelled it.
    """

    v = values
    # param.F:684-687 clamp axisymm and imoist to 0/1.
    for key in ("axisymm", "imoist"):
        if key in v and v[key] is not None:
            v[key] = max(0, min(1, int(v[key])))
    # param.F:700-701: no domain motion without imove.
    if int(v.get("imove", 0) or 0) == 0:
        v["umove"] = 0.0
        v["vmove"] = 0.0
    # param.F:720: cm1setup=2 forces tconfig=2.
    if int(v.get("cm1setup", 1) or 1) == 2:
        v["tconfig"] = 2
    # param.F:855-870: a surface model needs an ocean model.
    if int(v.get("sfcmodel", 0) or 0) >= 1 and int(v.get("oceanmodel", 0) or 0) <= 0:
        v["oceanmodel"] = 1
    # param.F:873-889: a surface model, oceanmodel=2 or a PBL scheme needs bbc=3.
    if (int(v.get("sfcmodel", 0) or 0) >= 1 or int(v.get("oceanmodel", 0) or 0) == 2
            or int(v.get("ipbl", 0) or 0) >= 1):
        v["bbc"] = 3
    # param.F:1985: no Coriolis parameter without icor.
    if int(v.get("icor", 0) or 0) == 0:
        v["fcor"] = 0.0
    # param.F:2596-2601: passive tracers.
    iptra = max(0, min(1, int(v.get("iptra", 0) or 0)))
    v["iptra"] = iptra
    v["npt"] = max(1, int(v.get("npt", 1) or 1)) if iptra == 1 else 1
    _scheme_adjustments(v)
    _output_switch_adjustments(v)
    return v


def _scheme_adjustments(v: dict[str, Any]) -> None:
    """param.F's rewrites of the settings that decide which budget terms and fields exist.

    Each changes what CM1 writes (a budget slot, the tke field, a global
    attribute), so the file is written for the values CM1 ran with.
    """

    cm1setup = _flag(v, "cm1setup", 1)
    # param.F:682-683, :921-933, :936-949: eqtset is 1 or 2, and 1 for the
    # anelastic/incompressible solvers and for ptype 4, 7, 50-53 and 55.
    eqtset = max(1, min(2, _flag(v, "eqtset", 2)))
    if _flag(v, "psolver", 3) in (4, 5, 6, 7) or _flag(v, "ptype") in (4, 7, 50, 51, 52, 53, 55):
        eqtset = 1
    v["eqtset"] = eqtset
    # param.F:906-918: these surface models need surface fluxes.
    if _flag(v, "sfcmodel") in (2, 3, 4, 5, 6, 7) and _flag(v, "isfcflx") == 0:
        v["isfcflx"] = 1
    # param.F:1044-1073: the dry and the moist test cases.
    if _flag(v, "testcase") in (1, 2):
        v["imoist"] = 0
    if _flag(v, "testcase") in (3, 4, 5, 7, 8):
        v["imoist"] = 1
    # param.F:1076-1084: a subgrid LES closure only in an LES setup (idoles).
    if cm1setup not in (1, 4) and _flag(v, "sgsmodel") >= 1:
        v["sgsmodel"] = 0
    # param.F:1087-1101: no second-order artificial diffusion beside a closure.
    if cm1setup >= 1 and _flag(v, "idiff") >= 1 and _flag(v, "difforder") == 2:
        v["idiff"] = 0
    # param.F:1103-1117: the PBL schemes that heat by dissipation themselves.
    if _flag(v, "ipbl") in (3, 4, 5) and _flag(v, "idiss") >= 1 and cm1setup != 4:
        v["idiss"] = 0
    # param.F:2066-2071: axisymmetric runs drop sixth-order diffusion.
    if _flag(v, "axisymm") == 1 and _flag(v, "idiff") == 1 and _flag(v, "difforder") == 6:
        v["idiff"] = 0
        v["difforder"] = 0


def _flag(v: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = v.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _output_switch_adjustments(v: dict[str, Any]) -> None:
    """param.F's rewrites of the &param9 switches, so the file lists what CM1 writes.

    param.F:1582-1604 (output_dbz only for ptypes with a reflectivity),
    :4633-4690 (every switch clamped to 0/1; output_interp and output_zs only
    with terrain), :4711-4738 (the moisture, tke, km/kh, pblten, radten,
    surface and fall-speed switches), :4740-4743 (terrain forces zs and zh).
    """

    terrain = 1 if _flag(v, "terrain_flag") else 0
    for key in list(v):
        if key.startswith("output_") and key not in ("output_format", "output_filetype") and v[key] is not None:
            try:
                v[key] = max(0, min(1, int(v[key])))
            except (TypeError, ValueError):
                continue
    if "output_interp" in v:
        v["output_interp"] = _flag(v, "output_interp") * terrain
    if "output_zs" in v:
        v["output_zs"] = _flag(v, "output_zs") * terrain
    imoist = _flag(v, "imoist")
    ptype = _flag(v, "ptype")
    sgsmodel = _flag(v, "sgsmodel")
    ipbl = _flag(v, "ipbl")
    radopt = _flag(v, "radopt")
    if imoist == 1 and _flag(v, "output_dbz") == 1 and ptype not in (2, 3, 5, 7) and ptype < 26:
        v["output_dbz"] = 0
    if imoist == 0:
        for key in ("output_rain", "output_srs", "output_sgs", "output_qv", "output_qvpert", "output_q",
                    "output_dbz", "output_cape", "output_cin", "output_lcl", "output_lfc", "output_lwp",
                    "output_pwat", "output_qvbudget"):
            v[key] = 0
    if sgsmodel <= 0 or sgsmodel in (2, 6):
        v["output_tke"] = 0
    if sgsmodel <= 0 and ipbl == 0:
        v["output_km"] = 0
        v["output_kh"] = 0
    if ipbl < 1:
        v["output_pblten"] = 0
    v["output_radten"] = 1 if radopt >= 1 else 0
    if _flag(v, "bbc", 1) != 3 and _flag(v, "isfcflx") == 0 and radopt == 0:
        v["output_sfcparams"] = 0
        v["output_sfcdiags"] = 0
    if terrain:
        v["output_zs"] = 1
        v["output_zh"] = 1
    if imoist == 0 or ptype not in (3, 5):
        v["output_fallvel"] = 0


def _importer_values(path: Path) -> dict[str, Any]:
    from ..cm1import import import_namelist

    try:
        imported = import_namelist(path)
    except Exception as error:  # noqa: BLE001 - reported with the file named
        raise SettingsError(f"cannot read the CM1 namelist {path}: {error}") from error
    return dict(imported.effective)


def from_namelist(path: Path | str) -> Cm1Settings:
    path = Path(path)
    values = _importer_values(path)
    return Cm1Settings(_param_f_adjustments(values), "namelist", str(path))


# GM1 deck tables whose keys carry CM1's own names.
_DECK_TABLES = ("initialization", "boundaries", "closures", "physics", "output", "diagnostics",
                "radiation", "surface", "dynamics", "run")
# GM1 deck keys whose CM1 name differs.
_DECK_RENAMES = {
    ("boundaries", "west"): "wbc", ("boundaries", "east"): "ebc",
    ("boundaries", "south"): "sbc", ("boundaries", "north"): "nbc",
    ("boundaries", "bottom"): "bbc", ("boundaries", "top"): "tbc",
    ("closures", "sgs_model"): "sgsmodel",
    ("closures", "u_move_m_s"): "umove", ("closures", "v_move_m_s"): "vmove",
    ("closures", "coriolis"): "icor", ("closures", "fcor_s_inv"): "fcor",
    ("dynamics", "coriolis"): "icor", ("dynamics", "fcor_s_inv"): "fcor",
    ("run", "history_interval_s"): "tapfrq", ("run", "dt_s"): "dtl",
}


def from_deck(path: Path | str) -> Cm1Settings:
    """CM1 values from a GM1 deck, over CM1 r21.1's own defaults."""

    import tomllib

    path = Path(path)
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SettingsError(f"cannot read the GM1 deck {path}: {error}") from error
    from ..cm1import import defaults as _defaults

    # Keys a deck does not carry (the SGS constants' switches, the date, the
    # radiation centre) take r21.1's distributed run/namelist.input values,
    # the file every CM1 namelist starts from, over CM1's code defaults.
    defaults = _defaults.load()
    values, _filled = defaults.fill({})
    values.update({k: v for k, v in defaults.distributed_flat().items() if v is not None})
    # An &param9 word the deck leaves out ran at GM1's own value, which is
    # what the frames hold.
    from ..cm1_output_selection import SWITCH_BY_KEY

    values.update({key: switch.gm1_default for key, switch in SWITCH_BY_KEY.items()})
    for table in _DECK_TABLES:
        for key, value in (document.get(table) or {}).items():
            if isinstance(value, dict):
                continue
            name = _DECK_RENAMES.get((table, key), key)
            if isinstance(value, bool):
                value = int(value)
            values[name] = value
    physics = document.get("physics") or {}
    component = str(physics.get("component", ""))
    values["ptype"] = {"morrison-mp10": 5, "wsm6-mp6": 7, "thompson-mp8": 3}.get(component, values.get("ptype"))
    values["imoist"] = 1 if component else int(values.get("imoist", 0) or 0)
    closures = document.get("closures") or {}
    values["imove"] = int(bool(closures.get("moving_frame", False)))
    # The deck keeps GM1's own output_format (1); the namelist's choice is
    # not in it, so netCDF is the writer this path offers.  A deck without
    # output_filetype ran GM1's own, one frame per time (filetype 2).
    values["output_format"] = 2
    if "output_filetype" not in (document.get("output") or {}):
        values["output_filetype"] = 2
    return Cm1Settings(_param_f_adjustments(values), "deck", str(path))


def resolve(layout: Any, *, namelist: Path | str | None = None, deck: Path | str | None = None) -> Cm1Settings:
    """The run's CM1 settings: the namelist when there is one, else the deck."""

    if namelist is not None:
        return from_namelist(namelist)
    if deck is None:
        for candidate in layout.candidate_namelists():
            if candidate.is_file():
                return from_namelist(candidate)
        for candidate in layout.candidate_decks():
            if candidate.is_file():
                return from_deck(candidate)
        raise SettingsError(
            f"{layout.root} has no CM1 namelist (namelist.input) and no GM1 deck (deck.toml); "
            "pass --namelist FILE or --deck FILE"
        )
    return from_deck(deck)


@dataclass(frozen=True)
class Geometry:
    """The grid a frame was published on, CM1-indexed where CM1 is."""

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    #: w-face heights zf(1..nz+1) and scalar heights zh(1..nz), float32 metres.
    zf: Any
    zh: Any
    #: ``mh = dz/(zf(k+1)-zf(k))`` and ``rmh = 1/mh`` (param.F:8162-8209).
    mh: Any
    rmh: Any
    dt: float

    @property
    def maxz(self) -> float:
        return float(self.zf[-1])

    @property
    def uniform(self) -> bool:
        import numpy as np

        return bool(np.all(self.mh == np.float32(1.0)))


def geometry(frame: Any) -> Geometry:
    """The grid of one frame, from its own records and its zf carrier."""

    import numpy as np

    config = frame.publication.get("config") or {}
    grid = config.get("grid") or (frame.publication.get("profile") or {}).get("geometry")
    if not grid:
        raise SettingsError(f"{frame.directory}: the frame records no grid")
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    dz = np.float32(grid["dz_m"])
    if frame.has("zf"):
        zf = np.array(frame.array("zf")[:, 0, 0], dtype=np.float32)
    elif frame.has("metrics.zf_profile"):
        zf = np.array(frame.array("metrics.zf_profile"), dtype=np.float32)
    else:
        zf = (np.arange(nz + 1, dtype=np.float32) * dz).astype(np.float32)
    if zf.shape != (nz + 1,):
        raise SettingsError(f"{frame.directory}: zf has {zf.shape[0]} faces for nz = {nz}")
    # param.F:8168-8170: zh = 0.5*(zf(k+1)+zf(k)), mh = dz/(zf(k+1)-zf(k)), in REAL.
    zh = (np.float32(0.5) * (zf[1:] + zf[:-1])).astype(np.float32)
    mh = (dz / (zf[1:] - zf[:-1])).astype(np.float32)
    rmh = (np.float32(1.0) / mh).astype(np.float32)
    dt = float(frame.publication.get("dt_s") or config.get("dt_s") or 0.0)
    return Geometry(nx, ny, nz, float(grid["dx_m"]), float(grid["dy_m"]), float(dz), zf, zh, mh, rmh, dt)


@dataclass(frozen=True)
class Species:
    """The q3d axis CM1 would write: names, units, index ranges, cloudvar."""

    names: tuple[str, ...]
    units: tuple[str, ...]
    cloudvar: tuple[bool, ...]
    numq: int
    nqv: int
    ranges: Mapping[str, int]
    iice: int
    idm: int
    idmplus: int = 0

    @property
    def mass_names(self) -> tuple[str, ...]:
        return tuple(n for n, u in zip(self.names, self.units) if u == "kg/kg")


def species(frame: Any) -> Species:
    """The moisture axis from the frame's species record (param.F's table)."""

    record = ((frame.publication.get("config") or {}).get("moisture_species")) or {}
    scalars = record.get("scalars") or []
    if not scalars:
        raise SettingsError(f"{frame.directory}: the frame records no moisture species")
    ranges = {key: -2 for key in ("nql1", "nql2", "nqs1", "nqs2", "nnc1", "nnc2", "nzl1", "nzl2", "nvl1", "nvl2")}
    ranges.update({k: int(v) for k, v in (record.get("cm1_index_ranges") or {}).items()})
    return Species(
        names=tuple(row["name"] for row in scalars),
        units=tuple(row["unit"] for row in scalars),
        cloudvar=tuple(bool(row.get("cloudvar", False)) for row in scalars),
        numq=int(record.get("numq", len(scalars))),
        nqv=int(ranges.pop("nqv", 1)),
        ranges=ranges,
        iice=int(record.get("iice", 0)),
        idm=int(record.get("idm", 0)),
    )


__all__ = [
    "Cm1Settings",
    "Geometry",
    "SettingsError",
    "Species",
    "from_deck",
    "from_namelist",
    "geometry",
    "resolve",
    "species",
]
