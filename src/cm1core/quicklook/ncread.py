"""CM1 netCDF history files as quick-look frames: pictures of CM1's own output, and of GM1's ``cm1out`` files.

``gpuwm-cm1 quicklook TARGET`` takes, besides a GM1 run (:func:`discover`):

* one CM1 netCDF history file: ``cm1out.nc`` holding every saved time (CM1's ``output_filetype = 1``) or one
  ``cm1out_NNNNNN.nc``;
* a folder of them: one file per saved time (``output_filetype = 2``), a CM1 run folder, or a GM1 run's
  ``cm1out/`` folder.

A file is a history file when it has CM1's six grid dimensions (``xh yh zh xf yf zf``) and ``time``
(``writeout_nc.F``); the statistics, parcel and azimuthal-average files CM1 writes beside it have not, and are
left out.  Each saved time is one frame, and every quick-look reads it through :class:`NcFrame`, which gives the
looks the names they read from a GM1 frame:

==============  =====================================================================================
the looks read  from the file
==============  =====================================================================================
``u3d v3d w3d`` ``u``, ``v`` (on ``xf``/``yf``, relative to the domain as CM1 writes them) and ``w``
``th3d``        ``thpert``; else ``th`` minus the base state (:attr:`NcLayout.base_theta`)
``th0 pi0 ...`` the base state (``output_basestate``) as one column
``q3d nc3d``    every species CM1 writes under its own name (``qv``, ``qc``, ...; numbers ``nc*``)
``pp3d``        ``pipert``
anything else   the file's own variable under its own name (``prs``, ``rho``, ``dbz``, ``rain``, ``sws``,
                ``uh``, ``zvort``, ...)
==============  =====================================================================================

The grid is the file's own: ``xh/xf/yh/yf`` give nx, ny, dx, dy and where x = y = 0 is, and ``zh/zf`` every
level's real height (stretched levels included; CM1 writes them in km unless ``outunits = 2``).  A moving domain's
ground position is the time integral of the file's ``umove``/``vmove`` (CM1 moves the domain at umove, vmove every
step).  From beside the files comes the run's ``namelist.input`` (in the folder, or the folder above it for a GM1
run's ``cm1out/``): the kind of storm (``iinit = 7`` is a hurricane), ``ihail`` and ``timax``.  Without one the
run is drawn as a thunderstorm run and the summary says so.

What the file does not hold is named in plain words with the CM1 switch that writes it (:meth:`NcFrame.missing`),
and a look that needs it is not drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..cm1out import settings as _settings
from ..cm1out.frames import Carrier, FrameError
from .frame import FieldMissing

#: CM1's grid dimensions (writeout_nc.F netcdf_prelim): a history file has every one, and ``time``.
GRID_DIMS = ("xh", "yh", "zh", "xf", "yf", "zf")
#: The dimensions of each kind of history variable, after ``time`` (writeout_nc.F:305-339), and the look's grid.
DIMS_GRID = {("yh", "xh"): "2", ("zh", "yh", "xh"): "s", ("zh", "yh", "xf"): "u", ("zh", "yf", "xh"): "v",
             ("zf", "yh", "xh"): "w"}
#: CM1's ``ptype`` (README.namelist) as GM1's microphysics names, where GM1 has the scheme.
PTYPE_COMPONENT = {1: "kessler", 3: "thompson-mp8", 5: "morrison-mp10", 7: "wsm6-mp6"}
#: CM1's ``ptype`` in plain words, for the schemes GM1 does not have (README.namelist).
PTYPE_WORDS = {0: "no microphysics", 1: "Kessler", 2: "Goddard LFO", 3: "Thompson", 4: "GSR LFO", 5: "Morrison",
               6: "Rotunno-Emanuel water only", 7: "WSM6", 26: "NSSL 2-moment", 27: "NSSL 2-moment with hail",
               28: "NSSL 1-moment", 50: "P3 1-ice", 51: "P3 1-ice", 52: "P3 2-ice", 53: "P3 3-moment ice",
               55: "ISHMAEL"}
#: The GM1 frame names the looks read, and the CM1 variable (and namelist switch) each comes from.
RENAMED = {"u3d": "u", "v3d": "v", "w3d": "w", "pp3d": "pipert"}
SWITCH = {
    "u": "output_u", "v": "output_v", "w": "output_w", "th": "output_th", "thpert": "output_thpert",
    "prs": "output_prs", "pi": "output_pi", "pipert": "output_pipert", "rho": "output_rho", "dbz": "output_dbz",
    "rain": "output_rain", "uh": "output_uh", "zvort": "output_vort", "qv": "output_qv",
    "sws": "output_sws", "svs": "output_svs", "sgs": "output_sgs", "shs": "output_shs", "sus": "output_sus",
    "th0": "output_basestate", "pi0": "output_basestate", "prs0": "output_basestate", "qv0": "output_basestate",
}
#: Plain words for what a GM1 frame name is, when the file lacks it.
WHAT = {"u3d": "west-east wind (u)", "v3d": "south-north wind (v)", "w3d": "vertical velocity (w)",
        "th": "potential temperature (th)", "prs": "pressure (prs)", "pp3d": "pressure perturbation (pipert)",
        "rho": "air density (rho)", "rain": "rain at the ground (rain)", "uh": "updraft helicity (uh)",
        "dbz": "reflectivity (dbz)", "q3d": "moisture (qv, qc, ...)", "nc3d": "number concentrations (nci, ...)",
        "qv": "water vapor (qv)", "qc": "cloud water (qc)", "qi": "cloud ice (qi)", "qr": "rain water (qr)",
        "qs": "snow (qs)", "qg": "graupel or hail (qg)"}
#: The mixing-ratio unit CM1 writes for a species (``qunit``), and the base-state columns the looks read.
MASS_UNIT = "kg/kg"
BASE_COLUMNS = ("th0", "pi0", "prs0", "qv0", "u0", "v0")
#: What each look draws from, for the reason it gives when a CM1 file lacks it (:meth:`NcFrame.why_not`): each need
#: is met by any one of its choices (a choice is fields that are all needed), and its words name the fields and the
#: CM1 switches that write them.
_WIND = ((("u", "v"),), "wind (u, v) (CM1 writes it with output_u = 1 and output_v = 1)")
_W = ((("w",),), "vertical velocity (w) (CM1 writes it with output_w = 1)")
_THETA = ((("th",), ("thpert",)), "potential temperature (th or thpert) (CM1 writes them with output_th = 1 and "
                                  "output_thpert = 1)")
_PRS = ((("prs",),), "pressure (prs) (CM1 writes it with output_prs = 1)")
_SPECIES = ((("qc",), ("qr",), ("qi",), ("qs",), ("qg",)),
            "cloud water, rain or ice (qc, qr, qi, qs, qg) (CM1 writes them with output_q = 1)")
LOOK_NEEDS: dict[str, tuple[tuple[tuple[tuple[str, ...], ...], str], ...]] = {
    "spin-near-ground": (_WIND,),
    "updraft-5km": (_W,),
    "updraft-slice": (_W,),
    "surface": (_THETA, _WIND),
    "rain": (((("rain",),), "rain at the ground (rain) (CM1 writes it with output_rain = 1)"),),
    "clouds-from-above": (((("qc",), ("qi",)), "cloud water or cloud ice (qc, qi) (CM1 writes them with output_q = 1)"),),
    "swath": (((("sws2",), ("sws",), ("u", "v")), "wind swath (sws2 or sws) (CM1 writes it with output_sws = 1), nor "
                                                  "the wind (u, v) to build one from"),),
    "hail-track": (((("sgs2",), ("sgs",), ("qg",)), "graupel or hail track (sgs2 or sgs) (CM1 writes it with "
                                                    "output_sgs = 1), nor the graupel or hail itself (qg) to build one "
                                                    "from"),),
    "spin-track": (((("svs2",), ("svs",), ("u", "v")), "spin track (svs2 or svs) (CM1 writes it with output_svs = 1), "
                                                       "nor the wind (u, v) to build one from"),),
    "wind-lowest": (_WIND,),
    "pressure-lowest": (_PRS,),
    "environment": (_PRS, _THETA, _WIND),
    # the looks added beside these: the same words for the fields they draw from
    "infrared": (((("qc",), ("qi",)), "cloud water or cloud ice (qc, qi) (CM1 writes them with output_q = 1)"),),
    "spectra": (_WIND,),
    "side-view": (_SPECIES,),
    "water-paths": (_SPECIES,),
    "updraft-time-height": (_W,),
    "spin-time-height": (_W,),
}
#: Why CM1 output has no parcels look or film (CM1's own parcel file is not read).
PARCELS_NOT_READ = "GM1 does not read CM1's parcel file (cm1out_pdata.nc) yet, so CM1 output has no parcels look."
#: How many cells in from each corner make the corner columns of :attr:`NcLayout.base_theta` (4 x 4 at each).
CORNER = 4
#: A GM1 run's own records, one of which is in the folder above its cm1out/.
RUN_RECORDS = ("run.json", "namelist.input", "deck.toml", "import-report.json", "events.jsonl")
#: The other netCDF files CM1 (and GM1's cm1out) write beside the history: statistics, parcels, azimuthal averages.
COMPANION = re.compile(r"^cm1out_(stats|pdata|azimavg_stats|azimavg_\d+)\.nc$")


class NcError(FrameError):
    """A netCDF file or folder that cannot be drawn; the message says why in plain words, and what to do."""


def _netcdf4() -> Any:
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - the wheel's dependencies bring it
        raise NcError("Reading netCDF needs the netCDF4 package, which this Python does not have. "
                      "Install it with: pip install netCDF4") from error
    return netCDF4


def _open(path: Path) -> Any:
    netCDF4 = _netcdf4()
    try:
        ds = netCDF4.Dataset(path, "r")
    except OSError as error:
        raise NcError(f"{path} cannot be read as netCDF ({error}).") from error
    ds.set_auto_maskandscale(False)
    return ds


def is_netcdf(path: Path) -> bool:
    """Whether ``path`` is a netCDF file by its first bytes (classic ``CDF``, or HDF5 for netCDF-4)."""

    try:
        with open(path, "rb") as stream:
            head = stream.read(8)
    except OSError:
        return False
    return head[:3] == b"CDF" or head == b"\x89HDF\r\n\x1a\n"


def history_times(ds: Any) -> list[float] | None:
    """The saved times of a CM1 history file, or None when ``ds`` is not one."""

    dims = ds.dimensions
    if not all(name in dims for name in GRID_DIMS) or "time" not in dims or "time" not in ds.variables:
        return None
    return [float(t) for t in np.asarray(ds.variables["time"][:], dtype=np.float64).reshape(-1)]


def _units_factor(var: Any, what: str, path: Path) -> float:
    units = str(getattr(var, "units", "")).strip().lower()
    if units in ("km", "kilometers", "kilometres"):
        return 1000.0
    if units in ("m", "meters", "metres"):
        return 1.0
    raise NcError(f"{path.name}: {what} is in {units or 'no stated unit'!r}; CM1 writes km or m (outunits).")


def _metres(values: Any, factor: float) -> np.ndarray:
    """A coordinate in metres: CM1's km words times 1000, rounded to the millimetre (below a float32 km's
    resolution) so a level written as 0.75000006 km is the 750 m it was."""

    out = np.asarray(values, dtype=np.float64).reshape(-1) * factor
    return np.round(out, 3) if factor != 1.0 else out


@dataclass(frozen=True)
class NcGrid:
    """The grid of a CM1 history file, from its own coordinates."""

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    x0: float              # m: the west edge (xf(1)) in the run's own coordinates
    y0: float
    zf: np.ndarray         # m, nz + 1 faces
    zh: np.ndarray         # m, nz levels
    uniform: bool

    @property
    def iorigin(self) -> int:
        """CM1's iorigin as the coordinates show it: 2 when x = y = 0 is the domain's middle, else 1."""

        centred = abs(self.x0 + 0.5 * self.nx * self.dx) <= 0.01 * self.dx and \
            abs(self.y0 + 0.5 * self.ny * self.dy) <= 0.01 * self.dy
        return 2 if centred else 1

    def geometry(self) -> _settings.Geometry:
        f32 = np.float32
        zf = self.zf.astype(f32)
        zh = self.zh.astype(f32)
        if self.uniform:
            dz = float(zf[1] - zf[0])
            mh = np.ones(self.nz, f32)
        else:
            # CM1 (param.F) sets dz = ztop / nz on stretched levels; mh = dz / (zf(k+1) - zf(k)).
            dz = float(zf[-1] - zf[0]) / self.nz
            mh = (f32(dz) / (zf[1:] - zf[:-1])).astype(f32)
        return _settings.Geometry(self.nx, self.ny, self.nz, self.dx, self.dy, dz, zf, zh, mh,
                                  (f32(1.0) / mh).astype(f32), 0.0)

    def same_as(self, other: "NcGrid") -> bool:
        return ((self.nx, self.ny, self.nz) == (other.nx, other.ny, other.nz)
                and abs(self.dx - other.dx) <= 1e-6 * self.dx and abs(self.dy - other.dy) <= 1e-6 * self.dy
                and np.allclose(self.zf, other.zf, rtol=0, atol=1e-2))


def _rounding(faces: np.ndarray, factor: float) -> float:
    """How uneven (m) an even spacing can come back from the file by rounding alone.

    CM1 works the faces out in double precision, keeps them as REAL and writes them times 0.001 in km (param.F,
    writeout_nc.F); each face is then off by up to about one float32 step of the domain's largest coordinate, so a
    step between two faces by up to about two, and the spread of the steps by about four.  An 81 km domain at 30 m
    comes back with steps from 29.998 to 30.003 m.  CM1's stretch_x and stretch_y change the spacing by far more."""

    largest = float(np.max(np.abs(faces))) / factor
    return 4.0 * float(np.spacing(np.float32(largest))) * factor


def _uneven(steps: np.ndarray, faces: np.ndarray, factor: float) -> bool:
    """Whether the spacing ``steps`` of ``faces`` (m) changes by more than the file's rounding."""

    allowed = 1e-4 * float(np.median(steps)) + _rounding(faces, factor) + 1e-3
    return float(steps.max() - steps.min()) > allowed


def _spacing(faces: np.ndarray, axis: str, path: Path, factor: float = 1.0) -> float:
    steps = np.diff(faces)
    if steps.size == 0 or float(steps.min()) <= 0.0:
        raise NcError(f"{path.name}: {axis}f does not increase, so it is not a CM1 grid.")
    if _uneven(steps, faces, factor):
        raise NcError(
            f"{path.name}: the {axis} spacing changes across the domain (from {steps.min():g} m to {steps.max():g} m; "
            f"CM1's stretch_{axis}). GM1's pictures draw an even horizontal grid only, so this file is not drawn.")
    # The spacing CM1 was given (a REAL, often a round number): the mean of the faces, to the millimetre.
    return float(round((faces[-1] - faces[0]) / steps.size, 3))


def read_grid(ds: Any, path: Path) -> NcGrid:
    v = ds.variables
    for name in GRID_DIMS:
        if name not in v:
            raise NcError(f"{path.name} has no {name} coordinate, so its grid is not known.")
    hx = _units_factor(v["xf"], "xf", path)
    hy = _units_factor(v["yf"], "yf", path)
    hz = _units_factor(v["zf"], "zf", path)
    xf = _metres(v["xf"][:], hx)
    yf = _metres(v["yf"][:], hy)
    zf = _metres(v["zf"][:], hz)
    zh = _metres(v["zh"][:], _units_factor(v["zh"], "zh", path))
    nx, ny, nz = len(ds.dimensions["xh"]), len(ds.dimensions["yh"]), len(ds.dimensions["zh"])
    if (len(xf), len(yf), len(zf), len(zh)) != (nx + 1, ny + 1, nz + 1, nz):
        raise NcError(f"{path.name}: its coordinates do not have the sizes of its dimensions.")
    refuse_two_dimensional(ds, path, nx, ny)
    dx = _spacing(xf, "x", path, hx)
    dy = _spacing(yf, "y", path, hy)
    if nz < 2 or float(np.min(np.diff(zf))) <= 0.0:
        raise NcError(f"{path.name}: zf does not increase, so it is not a CM1 grid.")
    if abs(float(zf[0])) > 0.5:
        raise NcError(f"{path.name}: the lowest w level is at {zf[0]:g} m, not the ground. GM1's pictures draw "
                      "flat-ground runs (heights above the ground), so this file is not drawn.")
    uniform = not _uneven(np.diff(zf), zf, hz)
    return NcGrid(nx, ny, nz, dx, dy, float(xf[0]), float(yf[0]), zf, zh, uniform)


def refuse_two_dimensional(ds: Any, path: Path, nx: int, ny: int) -> None:
    """CM1's 2-D runs (one column across, ny = 1: a squall line or density current in x-z) and axisymmetric runs
    (``axisymm = 1``, where x is the radius) are not maps: GM1's looks draw a 3-D domain seen from above."""

    try:
        axisymm = int(np.asarray(ds.getncattr("axisymm")).reshape(-1)[0]) if "axisymm" in ds.ncattrs() else 0
    except (TypeError, ValueError, IndexError):
        axisymm = 0
    if axisymm == 1:
        raise NcError(f"{path.name} is an axisymmetric CM1 run (axisymm = 1): its x is the distance from the centre, "
                      "and GM1's looks draw a 3-D domain seen from above, so it is not drawn.")
    if nx < 2 or ny < 2:
        raise NcError(f"{path.name} is a 2-D CM1 run ({nx} × {ny} columns): GM1's looks draw a 3-D domain seen from "
                      "above, so it is not drawn. Its fields can be read with any netCDF tool.")


def _other_grid(path: Path, grid: NcGrid, first: Path | None, first_grid: NcGrid) -> str:
    return (f"{path.name} is on another grid than {first.name if first is not None else 'the first file'} "
            f"({grid.nx} x {grid.ny} x {grid.nz}, dx {grid.dx:g} m, against {first_grid.nx} x {first_grid.ny} x "
            f"{first_grid.nz}, dx {first_grid.dx:g} m). Draw each run's files from their own folder.")


def refuse_terrain(ds: Any, path: Path) -> None:
    """CM1 with terrain writes ``zs`` (output_zs, on whenever there is terrain): its levels follow the ground, and
    ``zh``/``zf`` are nominal heights, which GM1's pictures would call heights above the ground."""

    if "zs" not in ds.variables:
        return
    zs = np.asarray(ds.variables["zs"][0] if ds.variables["zs"].ndim == 3 else ds.variables["zs"][:], dtype=np.float64)
    if zs.size and float(np.nanmax(np.abs(zs))) > 0.0:
        raise NcError(f"{path.name} is a run with terrain (zs up to {float(np.nanmax(zs)):g} m): its levels follow the "
                      "ground. GM1's pictures draw flat-ground runs, so it is not drawn.")


# ----------------------------------------------------------------------------------------------- lazy arrays
def _simple(index: Any, ndim: int) -> tuple | None:
    """``index`` as a tuple of ints and slices (what netCDF reads directly), or None."""

    if not isinstance(index, tuple):
        index = (index,)
    out = []
    for n, part in enumerate(index):
        if part is Ellipsis:
            out.extend([slice(None)] * (ndim - (len(index) - 1)))
            continue
        if isinstance(part, (int, np.integer)):
            out.append(int(part))
        elif isinstance(part, slice) and (part.step in (None, 1)):
            out.append(part)
        else:
            return None
    if len(out) > ndim:
        return None
    return tuple(out)


class _Lazy:
    """One variable at one saved time, indexed like a numpy array and read from the file on use."""

    dtype = np.dtype(np.float32)

    def __init__(self, var: Any, time_index: int, shape: tuple[int, ...]) -> None:
        self.var = var
        self.t = time_index
        self.shape = shape
        self.ndim = len(shape)
        chunks = None
        try:
            chunks = var.chunking()
        except (AttributeError, RuntimeError):
            chunks = None
        # A variable stored in chunks deeper than one level is read whole once (each level read would unpack it).
        self._whole = isinstance(chunks, list) and len(chunks) == len(shape) + 1 and len(shape) == 3 and chunks[1] > 1
        self._all: np.ndarray | None = None

    def __len__(self) -> int:
        return self.shape[0]

    def _read_all(self) -> np.ndarray:
        if self._all is None:
            data = np.asarray(self.var[self.t], dtype=np.float32)
            if self._whole:
                self._all = data
            return data
        return self._all

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        data = self._read_all()
        return data if dtype is None else data.astype(dtype)

    def __getitem__(self, index: Any) -> np.ndarray:
        if self._whole or self._all is not None:
            return self._read_all()[index]
        simple = _simple(index, self.ndim)
        if simple is None:
            return self._read_all()[index]
        if simple == ():
            return self._read_all()
        return np.asarray(self.var[(self.t, *simple)], dtype=np.float32)


class _Column:
    """A base-state column broadcast to a 3-D shape (as CM1's own arrays are)."""

    dtype = np.dtype(np.float32)

    def __init__(self, column: np.ndarray, shape: tuple[int, ...]) -> None:
        self.column = np.asarray(column, dtype=np.float32)
        self.shape = shape
        self.ndim = len(shape)

    def __len__(self) -> int:
        return self.shape[0]

    def full(self) -> np.ndarray:
        return np.broadcast_to(self.column.reshape(-1, *([1] * (self.ndim - 1))), self.shape)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.array(self.full(), dtype=dtype or np.float32)

    def __getitem__(self, index: Any) -> np.ndarray:
        return np.array(self.full()[index], dtype=np.float32)


class _Minus:
    """``a - b`` read lazily (θ′ as th minus the base state)."""

    dtype = np.dtype(np.float32)

    def __init__(self, a: Any, b: Any) -> None:
        self.a, self.b = a, b
        self.shape = a.shape
        self.ndim = len(a.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        out = (np.asarray(self.a, dtype=np.float32) - np.asarray(self.b, dtype=np.float32)).astype(np.float32)
        return out if dtype is None else out.astype(dtype)

    def __getitem__(self, index: Any) -> np.ndarray:
        return (np.asarray(self.a[index], dtype=np.float32) - np.asarray(self.b[index], dtype=np.float32)
                ).astype(np.float32)


class _Stack:
    """Several species stacked as one (n, nz, ny, nx) array, the way a GM1 frame keeps ``q3d``."""

    dtype = np.dtype(np.float32)

    def __init__(self, rows: Sequence[Any]) -> None:
        self.rows = list(rows)
        self.shape = (len(self.rows), *self.rows[0].shape)
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        out = np.stack([np.asarray(row, dtype=np.float32) for row in self.rows])
        return out if dtype is None else out.astype(dtype)

    def __getitem__(self, index: Any) -> np.ndarray:
        parts = index if isinstance(index, tuple) else (index,)
        first, rest = parts[0], parts[1:]
        if isinstance(first, (int, np.integer)):
            row = self.rows[int(first)]
            return np.asarray(row[rest] if rest else np.asarray(row), dtype=np.float32)
        picked = self.rows[first] if isinstance(first, slice) else [self.rows[int(i)] for i in first]
        return np.stack([np.asarray(row[rest] if rest else np.asarray(row), dtype=np.float32) for row in picked])


# ----------------------------------------------------------------------------------------------- one saved time
@dataclass(frozen=True)
class NcTime:
    """One saved time of a history file: what a lineage row carries in place of a frame directory."""

    path: Path
    index: int            # the time's index in its file
    time_s: float
    count: int            # how many times the file holds
    layout: "NcLayout" = field(compare=False, repr=False, hash=False)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def key(self) -> str:
        return self.path.name if self.count == 1 else f"{self.path.name}[{self.index + 1}]"

    def open_frame(self) -> "NcFrame":
        return NcFrame(self.layout, self)


class NcFrame:
    """One saved time of a CM1 history file, with the interface of :class:`cm1core.cm1out.frames.Frame`."""

    def __init__(self, layout: "NcLayout", ref: NcTime) -> None:
        self.layout = layout
        self.ref = ref
        self.directory = ref.path
        self.key = ref.key
        self._ds = _open(ref.path)
        grid = read_grid(self._ds, ref.path)
        if not grid.same_as(layout.grid):
            self._ds.close()
            raise NcError(_other_grid(ref.path, grid, layout.times[0].path, layout.grid))
        self.geometry = layout.geometry
        self.carriers: dict[str, Carrier] = {}
        self._readers: dict[str, Any] = {}
        #: How θ′ was formed when CM1 did not write it (thpert), in words for a picture.
        self.theta_words_source = ""
        #: Whether the base-state pressure was taken from the corner columns (the file holds no prs0 or pi0).
        self.base_from_corners = False
        self._map()
        locx, locy, umove, vmove = layout.domain_position(ref)
        self.location = {"model_time_s": ref.time_s, "step": 0,
                         "moving_frame": {"umove": umove, "vmove": vmove, "domainlocx": locx, "domainlocy": locy}}
        self.publication = {"config": layout.config, "model_time_s": ref.time_s}
        #: The reflectivity look's words for the file's own ``dbz`` (quicklook.reflectivity).
        self.dbz_words = layout.dbz_words
        #: CM1's translated swaths (sws2, ...) here are the ground under the domain at this time (quicklook.swath).
        self.swath_window = True
        self.arena_sha256 = None

    # -- the Frame interface ---------------------------------------------------------------------
    @property
    def model_time_s(self) -> float:
        return self.ref.time_s

    @property
    def moving_frame(self) -> Mapping[str, Any]:
        return self.location["moving_frame"]

    @property
    def index(self) -> int:
        return self.ref.index

    @property
    def step(self) -> int:
        return 0

    def has(self, name: str) -> bool:
        return name in self._readers

    def array(self, name: str) -> Any:
        if name not in self._readers:
            raise FieldMissing(self.missing(name))
        return self._readers[name]()

    def close(self) -> None:
        try:
            self._ds.close()
        except (RuntimeError, OSError):
            pass

    # -- what is not there -----------------------------------------------------------------------
    def missing(self, name: str) -> str:
        """Plain words for a field this file does not hold, with the CM1 switch that writes it."""

        cm1 = RENAMED.get(name, "thpert" if name == "th3d" else name)
        what = WHAT.get(name, f"{cm1}")
        switch = SWITCH.get(cm1)
        if name == "th3d":
            return (f"{self.directory.name} has no potential temperature (CM1 writes th with output_th = 1 and "
                    "thpert with output_thpert = 1).")
        if switch:
            return f"{self.directory.name} has no {what} (CM1 writes it with {switch} = 1)."
        if cm1 in ("qc", "qi", "qr", "qs", "qg"):
            return f"{self.directory.name} has no {what} (CM1 writes the moisture with output_q = 1)."
        return f"{self.directory.name} has no {what}."

    def why_not(self, look: str, reason: str) -> str:
        """Why ``look`` is not drawn from this file, in plain words: the first field it draws from that the file
        lacks, named with the CM1 switch that writes it (:data:`LOOK_NEEDS`); else ``reason``, as the look said it."""

        name = self.directory.name
        if look == "reflectivity" and "dbz" not in self._readers:
            words = f"{name} has no reflectivity (dbz) (CM1 writes it with output_dbz = 1)"
            if reason.startswith(f"{name} has no "):
                # what CM1's calc_refl10cm would read from the file is missing too
                return f"{words}, and GM1 cannot work it out from the file: it {reason[len(name) + 1:]}"
            if not reason or "output_dbz" in reason:
                return f"{words}."
            return f"{words}. {reason}"
        for choices, words in LOOK_NEEDS.get(look, ()):
            if not any(all(field in self._readers for field in choice) for choice in choices):
                return f"{name} has no {words}."
        return reason

    # -- the names the looks read ----------------------------------------------------------------
    def _carrier(self, name: str, meaning: str, units: str, shape: tuple[int, ...], reader: Any) -> None:
        self.carriers[name] = Carrier(name, meaning, units, "", shape, 0)
        self._readers[name] = reader

    def _map(self) -> None:
        ds = self._ds
        t = self.ref.index
        g = self.layout.grid
        shapes = {"2": (g.ny, g.nx), "s": (g.nz, g.ny, g.nx), "u": (g.nz, g.ny, g.nx + 1),
                  "v": (g.nz, g.ny + 1, g.nx), "w": (g.nz + 1, g.ny, g.nx)}
        own: dict[str, tuple[Any, str]] = {}
        for name, var in ds.variables.items():
            dims = tuple(var.dimensions)
            if not dims or dims[0] != "time" or dims[1:] not in DIMS_GRID:
                continue
            grid = DIMS_GRID[dims[1:]]
            own[name] = (var, grid)
        lazy = {name: (lambda v=var, s=shapes[grid]: _Lazy(v, t, s)) for name, (var, grid) in own.items()}

        def add(name: str, source: str) -> None:
            var, grid = own[source]
            self._carrier(name, str(getattr(var, "long_name", source)), str(getattr(var, "units", "")),
                          shapes[grid], lazy[source])

        # Every variable under its own name (what a slice or a section shows, and what the looks read by CM1 name).
        for name in own:
            add(name, name)
        for gm1, cm1 in RENAMED.items():
            if cm1 in own:
                add(gm1, cm1)
        # The base state, as the one column GM1's frames carry.
        for name in BASE_COLUMNS:
            if name in own and own[name][1] in ("s", "u", "v"):
                var = own[name][0]
                column = np.asarray(var[t, :, 0, 0], dtype=np.float32)
                self._carrier(name, str(getattr(var, "long_name", name)), str(getattr(var, "units", "")),
                              (g.nz,), (lambda c=column: c))
        # θ′: CM1's thpert, else th minus the base state.
        if "thpert" in own:
            add("th3d", "thpert")
            self.theta_words_source = "thpert"
        elif "th" in own:
            base, words = self.layout.base_theta(self)
            if base is not None:
                th = lazy["th"]
                self._carrier("th3d", "potential temperature perturbation", "K", shapes["s"],
                              (lambda b=base: _Minus(th(), _Column(b, shapes["s"]))))
                if "th0" not in self._readers:
                    self._carrier("th0", "base-state potential temperature", "K", (g.nz,), (lambda b=base: b))
                self.theta_words_source = words
        # The base-state pressure (a column skew-T's dashed lines): the file's prs0 or pi0, else the corner columns'.
        if not ({"prs0", "pi0"} & set(self._readers)) and "prs" in own:
            base = self.layout.corner_column("prs")
            if base is not None:
                self._carrier("prs0", "base-state pressure", "Pa", (g.nz,), (lambda b=base: b))
                self.base_from_corners = True
        # The moisture species, stacked as a GM1 frame keeps them (mass first, then numbers).
        mass, number = self.layout.species(own)
        if mass:
            self._carrier("q3d", "current " + ",".join(mass) + " mixing ratios", MASS_UNIT,
                          (len(mass), *shapes["s"]), (lambda m=mass: _Stack([lazy[n]() for n in m])))
        if number:
            self._carrier("nc3d", "current " + ",".join(number) + " number concentrations", "#/kg",
                          (len(number), *shapes["s"]), (lambda m=number: _Stack([lazy[n]() for n in m])))

    @property
    def base_note(self) -> str:
        """How the base state a column skew-T draws was taken, when the file does not hold it."""

        from . import style as S

        corner_theta = self.theta_note != ""
        if not (corner_theta or self.base_from_corners):
            return ""
        what = " and ".join(w for w, on in (("θ", corner_theta), ("pressure", self.base_from_corners)) if on)
        return (f"dashed base state: {what} of the domain's corner columns at "
                f"{S.fmt_clock(self.layout.times[0].time_s)}, as the file holds no base state")

    @property
    def theta_note(self) -> str:
        """How this frame's θ′ was formed, in words for a picture: empty when it is CM1's own (thpert, or th
        minus th0), else how the base state was taken."""

        return "" if self.theta_words_source in ("", "thpert", "th0") else self.theta_words_source

    def slice_fields(self) -> list[tuple[str, str, str, str, Any]]:
        """(name, description, units, grid, reader) of every field of the file a slice can show."""

        rows = []
        for name, var in self._ds.variables.items():
            dims = tuple(var.dimensions)
            if not dims or dims[0] != "time" or dims[1:] not in DIMS_GRID:
                continue
            grid = DIMS_GRID[dims[1:]]
            rows.append((name, str(getattr(var, "long_name", name)), str(getattr(var, "units", "")), grid,
                         (lambda _frame, n=name: np.asarray(self._readers[n](), dtype=np.float32))))
        return rows


# ----------------------------------------------------------------------------------------------- the files
@dataclass
class NcLayout:
    """A folder (or one file) of CM1 history files, in the shape of :class:`cm1core.cm1out.frames.RunLayout`."""

    target: Path
    root: Path                       # the folder the files are in
    times: list[NcTime]
    grid: NcGrid
    attributes: dict[str, Any]       # the first file's global attributes
    single_file: bool
    kind: str = "netcdf"
    segments: tuple = ()
    notes: list[str] = field(default_factory=list)
    _settings: Any = field(default=None, repr=False)

    # -- RunLayout's interface -------------------------------------------------------------------
    @property
    def default_out(self) -> Path:
        return default_out(self)

    def candidate_namelists(self) -> tuple[Path, ...]:
        return (self.root / "namelist.input", self.root.parent / "namelist.input")

    def candidate_decks(self) -> tuple[Path, ...]:
        return (self.root / "deck.toml", self.root.parent / "deck.toml")

    def run_directory(self) -> Any:
        return None

    def lineage_settled(self) -> bool:
        return True

    def frames(self) -> list:
        return []

    def ended(self) -> tuple[bool, Mapping[str, Any] | None]:
        # The files are what there is: nothing more arrives (a follower is refused, see the door).
        return True, {"kind": "end", "reason": "the files are all there"}

    def has_event_log(self) -> bool:
        return False

    def lineage(self) -> list[tuple[float, str, NcTime, int]]:
        return [(ref.time_s, ref.key, ref, 1) for ref in self.times]

    # -- what the files are ----------------------------------------------------------------------
    @property
    def gm1(self) -> bool:
        """GM1's own cm1out files (they carry the ``gm1_*`` provenance attributes)."""

        return "gm1_producer" in self.attributes or "gm1_note" in self.attributes

    @cached_property
    def records_root(self) -> Path:
        """Where the run's own records are (run.json, import-report.json): a GM1 run's folder for its
        ``cm1out/``, else the files' folder."""

        if self.gm1 and any((self.root.parent / name).is_file() for name in RUN_RECORDS):
            return self.root.parent
        return self.root

    @cached_property
    def geometry(self) -> _settings.Geometry:
        return self.grid.geometry()

    @property
    def settings(self) -> Any:
        """The run's CM1 settings from a namelist (or a GM1 deck) beside the files, or None."""

        if self._settings is None:
            try:
                self._settings = _settings.resolve(self)
            except (_settings.SettingsError, OSError, ValueError):
                self._settings = False
                self.notes.append("No CM1 namelist (namelist.input) beside the files: they are drawn as a "
                                  "thunderstorm run. For a hurricane run, pass its namelist with --namelist FILE.")
        return self._settings or None

    def use_settings(self, settings: Any) -> None:
        self._settings = settings if settings is not None else False

    @cached_property
    def component(self) -> str:
        if self.gm1:
            words = str(self.attributes.get("gm1_microphysics", "")).split()
            return words[0] if words else ""
        try:
            return PTYPE_COMPONENT.get(int(self.attributes.get("ptype")), "")
        except (TypeError, ValueError):
            return ""

    @cached_property
    def scheme(self) -> str:
        """The microphysics in plain words."""

        from .facts import scheme_words

        if self.component:
            return scheme_words(self.config)
        try:
            ptype = int(self.attributes.get("ptype"))
        except (TypeError, ValueError):
            return "unknown microphysics"
        return f"{PTYPE_WORDS.get(ptype, 'microphysics')} (CM1 ptype {ptype})"

    @cached_property
    def config(self) -> dict[str, Any]:
        """The publication record a GM1 frame would carry, as far as the file says it."""

        g = self.grid
        config: dict[str, Any] = {"grid": {"nx": g.nx, "ny": g.ny, "nz": g.nz, "dx_m": g.dx, "dy_m": g.dy,
                                           "dz_m": float(self.geometry.dz)},
                                  "microphysics": self.component}
        settings = self.settings
        if self.component in ("morrison-mp10", "wsm6-mp6"):
            ihail = -1
            if settings is not None:
                try:
                    ihail = int(settings.int("ihail", -1))
                except (TypeError, ValueError):
                    ihail = -1
            if ihail in (0, 1):
                config["dense_ice"] = {"ihail": ihail, "category": "hail" if ihail == 1 else "graupel"}
            else:
                # CM1's ihail is in the namelist, not the file: without it, qg is graupel or hail
                from .tracks import EITHER_DENSE

                config["dense_ice"] = {"category": EITHER_DENSE}
        ds = _open(self.times[0].path)
        try:
            mass, number = self.species({name: (var, "s") for name, var in ds.variables.items()
                                         if tuple(var.dimensions)[1:] == ("zh", "yh", "xh")
                                         and tuple(var.dimensions)[:1] == ("time",)})
            units = {name: str(getattr(ds.variables[name], "units", "")) for name in number}
        finally:
            ds.close()
        scalars = [{"name": n, "unit": MASS_UNIT} for n in mass] + [{"name": n, "unit": units[n]} for n in number]
        if scalars:
            config["moisture_species"] = {"scalars": scalars, "numq": len(scalars)}
        return config

    @staticmethod
    def species(own: Mapping[str, tuple[Any, str]]) -> tuple[list[str], list[str]]:
        """The species a file holds: CM1 writes each under its own name, described by that name
        (writeout.F's output_q loop), qv as the water vapor mixing ratio."""

        mass, number = [], []
        for name, (var, grid) in own.items():
            if grid != "s":
                continue
            units = str(getattr(var, "units", ""))
            if name == "qv" and units == MASS_UNIT:
                mass.insert(0, name)
            elif str(getattr(var, "long_name", "")) == name and name != "qv":
                if units == MASS_UNIT:
                    mass.append(name)
                elif units.startswith("#"):
                    number.append(name)
        return mass, number

    @cached_property
    def dbz_words(self) -> str:
        """Where the file's ``dbz`` comes from, in the words of the reflectivity look's data line: a GM1 run's
        cm1out files say what its frames' pictures say (cm1out's CPU operator is the one GM1's pictures use, and the
        name already says "from netCDF"); CM1's own files say it is CM1's.  The scheme is in the footer."""

        if self.gm1:
            formed = {}
            try:
                formed = json.loads(str(self.attributes.get("gm1_formed", "{}")))
            except ValueError:
                formed = {}
            if formed.get("dbz") == "derived":
                return "CM1's calc_refl10cm (Morrison, 10 cm) on the CPU"
            return ""     # the frames' own words: the scheme's own reflectivity, computed in the model
        return "CM1's own reflectivity (dbz)"

    @cached_property
    def version(self) -> str:
        """The model that wrote the files, in words for the footer."""

        if self.gm1:
            return ""
        raw = str(self.attributes.get("CM1 version", "")).strip()
        found = re.match(r"^cm1r(\d+)\.(\d+)$", raw)
        if found:
            return f"CM1 r{found.group(1)}.{found.group(2)}"
        return f"CM1 ({raw})" if raw else "CM1"

    # -- the moving domain -------------------------------------------------------------------------
    @cached_property
    def _motion(self) -> list[tuple[float, float, float]]:
        """(time, umove, vmove) of every saved time, oldest first."""

        rows = []
        for ref in self.times:
            ds = _open(ref.path)
            try:
                u = float(np.asarray(ds.variables["umove"][ref.index])) if "umove" in ds.variables else 0.0
                v = float(np.asarray(ds.variables["vmove"][ref.index])) if "vmove" in ds.variables else 0.0
            finally:
                ds.close()
            rows.append((ref.time_s, u, v))
        return rows

    def domain_position(self, ref: NcTime) -> tuple[float, float, float, float]:
        """(domainlocx, domainlocy, umove, vmove) at ``ref``: the domain's ground position is the integral of
        umove and vmove over time, each held from one saved time to the next (CM1 moves the domain by umove*dt
        every step; with one speed throughout this is umove times the time)."""

        locx = locy = 0.0
        previous_t = 0.0
        speed = (0.0, 0.0)
        for n, (t, u, v) in enumerate(self._motion):
            if n == 0:
                speed = (u, v)
            locx += speed[0] * (t - previous_t)
            locy += speed[1] * (t - previous_t)
            previous_t, speed = t, (u, v)
            if t == ref.time_s:
                return locx, locy, u, v
        return locx, locy, speed[0], speed[1]

    def ground_cells(self) -> tuple[int, int, int, int]:
        """The ground cells (x0, y0, x1, y1) the domain covers over every saved time, placed as the swath places
        each saved time (its position rounded to whole cells)."""

        from .swath import _round_half_away

        g = self.grid
        # the domain at T+0 too (it is where the pictures' dim box "the domain at the start" is), as GM1's own
        # ground canvas starts at min(0, ...): files that begin later (after a restart, or a few files) need it
        xs, ys = [0], [0]
        for ref in self.times:
            locx, locy, _u, _v = self.domain_position(ref)
            xs.append(_round_half_away(locx / g.dx))
            ys.append(_round_half_away(locy / g.dy))
        return min(xs), min(ys), max(xs) + g.nx, max(ys) + g.ny

    # -- the base state ----------------------------------------------------------------------------
    def base_theta(self, frame: NcFrame) -> tuple[np.ndarray | None, str]:
        """θ's base state for files with ``th`` and no ``thpert``: the file's ``th0`` (output_basestate), else
        the median of θ in the domain's four corner columns (4 x 4 cells each) at the earliest saved time.  For
        CM1's idealized starts (a warm bubble, or a vortex, in the middle of an even base state) those columns
        hold the base state itself."""

        ds = frame._ds
        if "th0" in ds.variables:
            # CM1's own base state: θ′ is what CM1's thpert would have been, and needs no words
            return np.asarray(ds.variables["th0"][frame.ref.index, :, 0, 0], dtype=np.float32), "th0"
        column = self.corner_column("th")
        if column is None:
            return None, ""
        from . import style as S

        return column, f"θ minus θ of the corner columns at {S.fmt_clock(self.times[0].time_s)}"

    def corner_column(self, name: str) -> np.ndarray | None:
        """The median of ``name`` in the domain's four corner columns (4 x 4 cells each) at the earliest saved time,
        or None when the file has no such scalar field: the base state of a file that does not hold it."""

        cache = self.__dict__.setdefault("_corner_columns", {})
        if name not in cache:
            first = self.times[0]
            source = _open(first.path)
            try:
                var = source.variables.get(name)
                if var is None or tuple(var.dimensions) != ("time", "zh", "yh", "xh"):
                    cache[name] = None
                else:
                    n = max(1, min(CORNER, self.grid.nx // 2, self.grid.ny // 2))
                    nx, ny = self.grid.nx, self.grid.ny
                    blocks = [np.asarray(var[first.index, :, j0:j0 + n, i0:i0 + n], dtype=np.float64)
                              for j0 in (0, ny - n) for i0 in (0, nx - n)]
                    stack = np.concatenate([b.reshape(b.shape[0], -1) for b in blocks], axis=1)
                    cache[name] = np.median(stack, axis=1).astype(np.float32)
            finally:
                source.close()
        return cache[name]

    # -- the run's facts -----------------------------------------------------------------------------
    def run_facts(self, config: Mapping[str, Any], dx_m: float, dy_m: float, *, label: str | None = None,
                  settings: Any = None) -> Any:
        from . import facts as F

        settings = settings if settings is not None else self.settings
        iorigin = self.grid.iorigin
        storm_kind = ""
        if settings is not None:
            try:
                storm_kind = "hurricane" if int(settings.int("iinit", 1)) == F.HURRICANE_IINIT else ""
            except (TypeError, ValueError):
                storm_kind = ""
        root = self.records_root
        if self.gm1:
            # A GM1 run's cm1out files say what its frames say (its version, checked state and name), from the
            # run's own records when its folder is above them, and where the picture was read from.
            facts = None
            if root != self.root:
                from ..cm1out import frames as _frames

                try:
                    facts = F.run_facts(_frames.discover(root), config, dx_m, dy_m, label=label, settings=settings)
                except _frames.FrameError:
                    facts = None
            if facts is None:
                from .._version import __version__

                checked, checked_source = F.checked_state(root, None, self.component)
                if storm_kind == "hurricane":
                    checked = F.hurricane_checked(settings)
                    checked_source = ("GM1's and CM1's runs of the 8 km hurricane preset"
                                      if checked == F.HURRICANE_CHECKED else "no CM1 run with these hurricane settings")
                facts = F.RunFacts(name=label or root.name,
                                   version=f"{__version__} (the run's version is not recorded)",
                                   version_source="this package", scheme=F.scheme_words(config),
                                   component=self.component, checked=checked, checked_source=checked_source,
                                   dx_m=float(dx_m), dy_m=float(dy_m), iorigin=iorigin, storm_kind=storm_kind)
            facts.name = f"{facts.name}, from netCDF"
            facts.source = "cm1out netCDF"
            return facts
        from .._version import __version__

        name = label or (self.root.name if not self.single_file or self.target.stem == "cm1out" else self.target.stem)
        return F.RunFacts(name=f"{name}, CM1 output", version=self.version,
                          version_source="the file's CM1 version", scheme=self.scheme, component=self.component,
                          checked="", checked_source="CM1's own output", dx_m=float(dx_m), dy_m=float(dy_m),
                          iorigin=iorigin, storm_kind=storm_kind, made_by=self.version, model="CM1",
                          mark_word=F.CM1_MARK_WORD, source=f"{self.version} netCDF, drawn by GM1 {__version__}")


def default_out(layout: NcLayout) -> Path:
    """Where the pictures go: ``quicklook/`` in the files' folder, or ``NAME-quicklook/`` beside one file."""

    if layout.single_file and layout.target.is_file():
        return layout.target.parent / f"{layout.target.stem}-quicklook"
    return layout.root / "quicklook"


def _history_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".nc" and not p.name.startswith("."))


def looks_like_netcdf_target(target: Path) -> bool:
    """Whether ``target`` is a netCDF file, or a folder with netCDF files and no GM1 frames."""

    target = Path(target)
    if target.is_file():
        return target.suffix.lower() == ".nc" or is_netcdf(target)
    if not target.is_dir():
        return False
    if (target / "segments").is_dir() or (target / "frames").is_dir():
        return False
    from ..cm1out.frames import published_frames

    if published_frames(target):
        return False
    return any(True for _ in _history_files(target))


def open_layout(target: Path | str) -> NcLayout:
    """The CM1 history files at ``target`` (one file, or a folder of them), oldest time first."""

    target = Path(target)
    if target.is_file():
        files, root, single = [target], target.parent, True
    elif target.is_dir():
        files, root, single = _history_files(target), target, False
    else:
        raise NcError(f"{target} is not a file or a folder.")
    rows: list[tuple[float, Path, int, int]] = []
    skipped: list[str] = []
    grid = attributes = None
    first_path = None
    for path in files:
        if not is_netcdf(path):
            if single:
                raise NcError(f"{path} is not a netCDF file.")
            skipped.append(path.name)
            continue
        ds = _open(path)
        try:
            times = history_times(ds)
            if times is None:
                if single:
                    raise NcError(f"{path.name} is not a CM1 history file: it lacks CM1's grid dimensions "
                                  "(xh, yh, zh, xf, yf, zf) or time. CM1's statistics, parcel and azimuthal-average "
                                  "files are not drawn.")
                skipped.append(path.name)
                continue
            if grid is None:
                grid = read_grid(ds, path)
                refuse_terrain(ds, path)
                attributes = {key: ds.getncattr(key) for key in ds.ncattrs()}
                first_path = path
            else:
                # every file on the first one's grid, before anything is drawn
                other = read_grid(ds, path)
                if not other.same_as(grid):
                    raise NcError(_other_grid(path, other, first_path, grid))
        finally:
            ds.close()
        for index, t in enumerate(times):
            rows.append((t, path, index, len(times)))
    if not rows or grid is None:
        raise NcError(f"{target} holds no CM1 netCDF history file (cm1out.nc or cm1out_NNNNNN.nc, with CM1's grid "
                      "dimensions and time).")
    by_time: dict[float, list[tuple[float, Path, int, int]]] = {}
    for row in rows:
        by_time.setdefault(row[0], []).append(row)
    doubled = {t: group for t, group in by_time.items() if len({p for _t, p, _i, _n in group}) > 1}
    if doubled:
        t, group = sorted(doubled.items())[0]
        names = ", ".join(sorted({p.name for _t, p, _i, _n in group})[:3])
        raise NcError(f"More than one file holds the time {t:g} s ({names}). CM1 writes one file per process with "
                      "output_filetype = 3; GM1 draws a whole domain, so join the pieces first, or write the run with "
                      "output_filetype = 1 or 2. Two copies of one run in the folder: keep one.")
    layout = NcLayout(target=target, root=root, times=[], grid=grid, attributes=attributes or {},
                      single_file=single)
    layout.times = [NcTime(path, index, t, count, layout) for t, path, index, count in sorted(rows, key=lambda r: r[0])]
    # CM1's own companions (statistics, parcels, azimuthal averages) are left out without a word.
    skipped = [name for name in skipped if not COMPANION.match(name)]
    if skipped:
        layout.notes.append(f"Not CM1 history files, left out: {', '.join(skipped[:6])}"
                            + (f" and {len(skipped) - 6} more" if len(skipped) > 6 else "") + ".")
    if first_path is not None and layout.times[0].path != first_path:
        # the grid and attributes are the earliest time's file's
        ds = _open(layout.times[0].path)
        try:
            layout.grid = read_grid(ds, layout.times[0].path)
            layout.attributes = {key: ds.getncattr(key) for key in ds.ncattrs()}
        finally:
            ds.close()
    return layout


def discover(target: Path | str) -> Any:
    """A GM1 run's layout (:func:`cm1core.cm1out.frames.discover`), or the CM1 netCDF files at ``target``."""

    from ..cm1out import frames as _frames

    target = Path(target)
    if looks_like_netcdf_target(target):
        return open_layout(target)
    try:
        return _frames.discover(target)
    except _frames.FrameError as error:
        if target.is_dir() and (target / "cm1out").is_dir() and any(True for _ in _history_files(target / "cm1out")):
            raise _frames.FrameError(f"{error}. Its CM1-format files can be drawn: gpuwm-cm1 quicklook "
                                     f"{target / 'cm1out'}") from error
        if target.is_dir() and any(target.glob("cm1out*.ctl")):
            raise _frames.FrameError(
                f"{target} holds CM1's GrADS output (output_format = 1), which GM1's pictures do not read. Write "
                "CM1's history as netCDF (output_format = 2) to draw it.") from error
        if target.is_dir():
            # quicklook draws CM1's netCDF output too: say that the folder holds none of that either
            raise _frames.FrameError(f"{error}. Nor does it hold CM1 netCDF history files (cm1out.nc, or "
                                     "cm1out_NNNNNN.nc), which quicklook also draws.") from error
        raise


__all__ = ["NcError", "NcFrame", "NcGrid", "NcLayout", "NcTime", "default_out", "discover",
           "history_times", "is_netcdf", "looks_like_netcdf_target", "open_layout", "read_grid"]
