"""WRF and ArWen history files (``wrfout``), read record by record: the one wrfout reader GM1 has.

Two routes read a history file, and both read it here:

* the **column route** (:mod:`cm1core.cm1import.wrfcolumn`): one column of a WRF or ArWen run becomes CM1's
  ``input_sounding`` for an idealized GM1 storm;
* the **3D route** (:mod:`cm1core.arwen_chain`, WRF's ``ndown`` in GM1 terms): a finer GM1 domain takes its
  initial state and lateral boundaries from a sequence of history records.

WHAT IT READS
-------------
Every container WRF and ArWen write: netCDF classic CDF-1 (``NETCDF3_CLASSIC``), CDF-2
(``NETCDF3_64BIT_OFFSET``, ArWen's default Rust writer, ``gpuwm/io/wrfout.py``), CDF-5
(``NETCDF3_64BIT_DATA``) and netCDF-4/HDF5 (stock WRF's default with ``io_form_history = 2`` on a
netCDF-4 build).  It is read with the netCDF-C library through ``netCDF4`` (stock WRF often writes
netCDF-4, which the dependency-free Rust extractor cannot read, and the wheel does not ship the
extractor).  Values are read raw: no masking, no scaling, no fill
substitution; a binary32 field comes back as the same binary32 words the file holds.

The Rust extractor ``gm1-arwen-history-extract`` (``rust/cm1-state/src/arwen_history.rs``) stays the oracle on
classic files: :meth:`WrfHistory.write_frame_dir` writes its frame format (``frame.json`` + ``fields.f32le``,
schema ``gm1.arwen-history-frame/v1``) word for word, and ``tests/test_wrf_history.py`` compares the two.

WHO WROTE IT
------------
``producer`` is ``"ArWen"`` when the file carries gpuwm's provenance seal (the ``GPUWM_VERSION`` global) or its
``TITLE`` names gpuwm (``gpuwm/io/wrfout.py`` writes the caller's title, ``"gpuwm"`` by default, and the direct
exporter writes "... GPUWM NATIVE WRF ..."); otherwise ``"WRF"`` (stock WRF writes
``" OUTPUT FROM WRF V<version> MODEL"``).  Nothing a route does depends on it except the words it shows.

WHAT A FRAME IS
---------------
:meth:`WrfHistory.frame` returns a :class:`HistoryFrame`: one record, with :class:`cm1core.arwen_chain.ArwenFrame`'s
interface (``times``, ``valid_s``, ``attributes``, ``dimensions``, ``has``, ``entry``, ``array``, ``directory``
and ``header``), read lazily from the open file.  The record dimension is removed; a static (non-record)
variable is the variable itself, as the extractor writes it.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FRAME_SCHEMA = "gm1.arwen-history-frame/v1"
GATE = "wrf-history"
#: The variables the chain extracts, in the extractor's order (``arwen_history.rs`` ``CHAIN_VARIABLES``).
CHAIN_VARIABLES: tuple[str, ...] = (
    "U", "V", "W", "T", "THM", "PH", "PHB", "P", "PB", "HGT", "MAPFAC_M", "XLAT", "XLONG",
    "PSFC", "TH2", "T2", "Q2", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE",
    "QNSNOW", "QNRAIN", "QNGRAUPEL", "QNCLOUD", "QNWFA", "QNIFA", "QNCCN",
)
#: netCDF4's ``file_format`` -> the container's name as the extractor and the pages spell it.
FORMATS = {
    "NETCDF3_CLASSIC": "CDF-1",
    "NETCDF3_64BIT_OFFSET": "CDF-2",
    "NETCDF3_64BIT": "CDF-2",
    "NETCDF3_64BIT_DATA": "CDF-5",
    "NETCDF4": "netCDF-4",
    "NETCDF4_CLASSIC": "netCDF-4 classic",
}
#: WRF ``MAP_PROJ`` -> its name (``share/module_llxy.F``; 0 is an idealized run's Cartesian grid).
MAP_PROJECTIONS = {0: "idealized (no map)", 1: "Lambert conformal", 2: "polar stereographic", 3: "Mercator",
                   6: "latitude-longitude"}

REFUSAL_CODES = {
    "GM1-WRF-FILE": "the file is not a netCDF file netCDF-C can open, or not a WRF/ArWen history",
    "GM1-WRF-RECORD": "the record asked for is not in the file",
    "GM1-WRF-TIME": "the time asked for is not one of the file's Times",
    "GM1-WRF-TYPE": "a field is not binary32 and would be narrowed or reinterpreted",
    "GM1-WRF-OUTPUT": "a frame directory would be written into an existing directory",
}

_DOMAIN_IN_NAME = re.compile(r"_d(\d{2})(?:_|$|\.)")


def _refuse(code: str, reason: str, remedy: str, *, option: str | None = None, value: object = None) -> None:
    from .errors import refuse

    if code not in REFUSAL_CODES:
        raise KeyError(code)
    refuse(code=code, gate=GATE, reason=reason, remedy=remedy, option=option, value=value)


def wrf_spelling(text: str) -> str:
    """A time as WRF spells it, ``YYYY-MM-DD_HH:MM:SS``.

    WRF writes ``Times`` with an underscore; some post-processors rewrite it as ``YYYY-MM-DD HH:MM:SS`` or
    ``YYYY-MM-DDTHH:MM:SS`` (ISO 8601, sometimes with a trailing ``Z``).  All name the same time, so every
    time the reader keeps or compares is put in WRF's spelling first; a WRF-spelled time is returned as it is.
    """

    raw = str(text).strip().rstrip("\0").strip()
    if raw.endswith("Z"):
        raw = raw[:-1]
    if len(raw) == 19 and raw[10] in (" ", "T"):
        raw = raw[:10] + "_" + raw[11:]
    return raw


def parse_wrf_time(text: str) -> float:
    """A WRF ``Times`` string (WRF's or the ISO spelling, :func:`wrf_spelling`) as UTC seconds since the epoch."""

    stamp = _dt.datetime.strptime(wrf_spelling(text), "%Y-%m-%d_%H:%M:%S")
    return stamp.replace(tzinfo=_dt.timezone.utc).timestamp()


def _np() -> Any:
    import numpy

    return numpy


def _json_number(value: Any) -> Any:
    """One attribute number as the extractor spells it: finite as f64, integral without a fraction, else null."""

    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) < 2.0**53:
        return int(number)
    return number


def json_attribute(value: Any) -> Any:
    """A netCDF attribute value as the extractor transcribes it (text trimmed of NULs; one number or a list)."""

    np = _np()
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value.rstrip("\0")
    array = np.atleast_1d(np.asarray(value))
    if array.dtype.kind in "SU":
        return "".join(item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item)
                       for item in array.ravel()).rstrip("\0")
    numbers = [_json_number(item) for item in array.ravel()]
    return numbers[0] if len(numbers) == 1 else numbers


def _text(values: Any) -> str:
    """A CHAR record (``Times``) as text."""

    np = _np()
    array = np.asarray(values)
    if array.dtype.kind == "S":
        raw = b"".join(bytes(item) for item in array.ravel())
        return raw.decode("ascii", "replace").rstrip("\0").strip()
    if array.dtype.kind == "U":
        return "".join(str(item) for item in array.ravel()).rstrip("\0").strip()
    return str(values).rstrip("\0").strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path | str) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


class WrfHistory:
    """An open WRF or ArWen history file.  Use as a context manager, or call :meth:`close`."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._sha: tuple[str, int] | None = None
        try:
            import netCDF4
        except ImportError as error:  # pragma: no cover - netCDF4 ships with gpuwm
            _refuse("GM1-WRF-FILE", f"netCDF4 is not installed ({error}), so no history file can be read",
                    "pip install netCDF4", value=str(path))
        if not self.path.is_file():
            _refuse("GM1-WRF-FILE", f"there is no file at {self.path}",
                    "give the path of a wrfout or ArWen history file on this computer", value=str(path))
        try:
            self.dataset = netCDF4.Dataset(os.fspath(self.path), "r")
        except OSError as error:
            _refuse("GM1-WRF-FILE",
                    f"{self.path} could not be opened as netCDF ({error}); a WRF or ArWen history is a netCDF "
                    "file (classic CDF-1/2/5 or netCDF-4)",
                    "give the path of a wrfout_d0N_... file, not a namelist or a GRIB file", value=str(path))
        self.dataset.set_auto_maskandscale(False)
        try:
            self.dataset.set_always_mask(False)
        except AttributeError:  # pragma: no cover - very old netCDF4
            pass
        if "Times" not in self.dataset.variables:
            self.close()
            _refuse("GM1-WRF-FILE",
                    f"{self.path} has no Times variable, so it is not a WRF or ArWen history (every wrfout has "
                    "one CHAR record Times per output time)",
                    "give the path of a wrfout_d0N_... history file", value=str(path))
        self.format = FORMATS.get(self.dataset.file_format, self.dataset.file_format)
        times_var = self.dataset.variables["Times"]
        # In WRF's spelling (a file whose Times a post-processor rewrote in the ISO spelling reads the same).
        self.times: tuple[str, ...] = tuple(wrf_spelling(_text(times_var[r])) for r in range(times_var.shape[0]))
        self.records = len(self.times)

    # -- context ----------------------------------------------------------------
    def __enter__(self) -> "WrfHistory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        dataset = getattr(self, "dataset", None)
        if dataset is not None:
            try:
                dataset.close()
            except RuntimeError:
                pass
            self.dataset = None

    # -- identity ---------------------------------------------------------------
    @property
    def sha256(self) -> str:
        if self._sha is None:
            self._sha = file_sha256(self.path)
        return self._sha[0]

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    @property
    def attributes(self) -> dict[str, Any]:
        return {name: json_attribute(self.dataset.getncattr(name)) for name in self.dataset.ncattrs()}

    def attribute(self, name: str, default: Any = None) -> Any:
        if name not in self.dataset.ncattrs():
            return default
        return json_attribute(self.dataset.getncattr(name))

    @property
    def title(self) -> str:
        return str(self.attribute("TITLE", "") or "")

    @property
    def producer(self) -> str:
        """``"ArWen"`` (gpuwm's seal or title) or ``"WRF"``."""

        if "GPUWM_VERSION" in self.dataset.ncattrs() or "gpuwm" in self.title.lower():
            return "ArWen"
        return "WRF"

    @property
    def domain_id(self) -> int:
        """``GRID_ID``, else the ``_dNN`` of the file name, else 1."""

        grid = self.attribute("GRID_ID")
        if isinstance(grid, (int, float)) and grid >= 1:
            return int(grid)
        found = _DOMAIN_IN_NAME.search(self.path.name)
        return int(found.group(1)) if found else 1

    @property
    def dimensions(self) -> dict[str, int]:
        return {name: len(dim) for name, dim in self.dataset.dimensions.items()}

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.dataset.variables)

    def has(self, name: str) -> bool:
        return name in self.dataset.variables

    def is_record_variable(self, name: str) -> bool:
        var = self.dataset.variables[name]
        if not var.dimensions:
            return False
        dim = self.dataset.dimensions[var.dimensions[0]]
        return dim.isunlimited() or var.dimensions[0] == "Time"

    # -- time -------------------------------------------------------------------
    def valid_s(self, record: int) -> float:
        return parse_wrf_time(self.times[self.check_record(record)])

    def check_record(self, record: int) -> int:
        if isinstance(record, bool) or not isinstance(record, int) or not 0 <= record < self.records:
            _refuse("GM1-WRF-RECORD",
                    f"record {record!r} was asked for and {self.path.name} holds {self.records} "
                    f"({', '.join(self.times[:6])}{' ...' if self.records > 6 else ''})",
                    f"give a record from 0 to {self.records - 1}, or a time the file holds",
                    option="record", value=record)
        return record

    def record_of(self, when: str) -> int:
        """The record whose ``Times`` is ``when`` (``YYYY-MM-DD_HH:MM:SS``; a space or 'T' for '_' is accepted)."""

        text = str(when).strip().rstrip("Z")
        if len(text) == 16:
            text += ":00"
        text = wrf_spelling(text)
        for record, stamp in enumerate(self.times):
            if stamp == text:
                return record
        _refuse("GM1-WRF-TIME",
                f"{when!r} is not one of {self.path.name}'s times ({', '.join(self.times[:8])}"
                f"{' ...' if self.records > 8 else ''})",
                "give one of the file's times, as gpuwm-cm1 sounding times FILE lists them",
                option="time", value=when)
        raise AssertionError  # pragma: no cover

    # -- fields -----------------------------------------------------------------
    def array(self, name: str, record: int = 0) -> Any:
        """One field of one record as float32 words, the record dimension removed."""

        np = _np()
        var = self.dataset.variables[name]
        if var.dtype != np.float32:
            _refuse("GM1-WRF-TYPE",
                    f"{name} in {self.path.name} is {var.dtype}, not binary32; reading it as binary32 would narrow "
                    "or reinterpret words the model never held",
                    f"write {name} as a float field", option=name, value=str(var.dtype))
        if self.is_record_variable(name):
            data = var[self.check_record(record)]
        else:
            data = var[:]
        return np.ascontiguousarray(np.asarray(data), dtype="<f4")

    def field_row(self, name: str) -> dict[str, Any]:
        var = self.dataset.variables[name]
        dims = list(var.dimensions[1:] if self.is_record_variable(name) else var.dimensions)
        shape = [len(self.dataset.dimensions[d]) for d in dims]
        attrs = set(var.ncattrs())
        return {
            "dims": dims,
            "name": name,
            "shape": shape,
            "stagger": str(json_attribute(var.getncattr("stagger"))) if "stagger" in attrs else "",
            "units": str(json_attribute(var.getncattr("units"))) if "units" in attrs else "",
        }

    def source(self, record: int) -> dict[str, Any]:
        sha = self.sha256
        return {"bytes": self._sha[1] if self._sha else self.size_bytes, "format": self.format,
                "numrecs": self.records, "path": os.fspath(self.path), "record": int(record), "sha256": sha}

    def frame(self, record: int, names: Sequence[str] | None = None) -> "HistoryFrame":
        """Record ``record`` as a frame.  ``names`` limits it to those variables (the extractor's list)."""

        return HistoryFrame(self, self.check_record(record), None if names is None else tuple(names))

    def write_frame_dir(self, record: int, out: Path | str,
                        names: Sequence[str] = CHAIN_VARIABLES) -> Any:
        """Write record ``record`` in ``gm1-arwen-history-extract``'s format into the new directory ``out``.

        Word for word what the Rust extractor writes for the same file and record: the named binary32 variables
        in ``names`` order (absent ones listed under ``absent``), little-endian, C order, the record dimension
        removed, each hashed; the global attributes; every dimension with the record one at the record count.
        """

        from .arwen_chain import ArwenFrame

        self.check_record(record)
        wanted = [name for name in names if self.has(name)]
        absent = [name for name in names if not self.has(name)]
        for name in wanted:
            if self.dataset.variables[name].dtype != _np().float32:
                self.array(name, record)  # refuses with the variable named
        root = Path(out)
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            _refuse("GM1-WRF-OUTPUT",
                    f"{root} already exists; a frame is only ever written into a new directory, so two sources "
                    "can never be mixed in one frame",
                    "give a new directory", value=str(root))
        rows = []
        offset = 0
        with open(root / "fields.f32le", "wb") as handle:
            for name in wanted:
                raw = self.array(name, record).tobytes()
                handle.write(raw)
                row = self.field_row(name)
                row.update({"nbytes": len(raw), "offset": offset, "sha256": _sha256(raw)})
                rows.append(dict(sorted(row.items())))
                offset += len(raw)
        header = {
            "absent": absent,
            "attributes": self.attributes,
            "dimensions": self.dimensions,
            "fields": rows,
            "schema": FRAME_SCHEMA,
            "source": self.source(record),
            "times": self.times[record],
        }
        (root / "frame.json").write_text(json.dumps(header, sort_keys=True), encoding="utf-8")
        return ArwenFrame.open(root)

    # -- facts for a page -------------------------------------------------------
    def facts(self) -> dict[str, Any]:
        """What the file is, in the words and numbers a page shows (no field is read but XLAT/XLONG's corners)."""

        dims = self.dimensions
        attrs = self.attributes
        map_proj = attrs.get("MAP_PROJ")
        facts: dict[str, Any] = {
            "path": os.fspath(self.path),
            "name": self.path.name,
            "bytes": self.size_bytes,
            "format": self.format,
            "producer": self.producer,
            "title": self.title.strip(),
            "domain_id": self.domain_id,
            "records": self.records,
            "times": list(self.times),
            "nx": dims.get("west_east"),
            "ny": dims.get("south_north"),
            "nz": dims.get("bottom_top"),
            "dx_m": attrs.get("DX"),
            "dy_m": attrs.get("DY"),
            "map_proj": map_proj,
            "map_proj_name": MAP_PROJECTIONS.get(map_proj, f"MAP_PROJ = {map_proj}") if map_proj is not None else None,
            "mp_physics": attrs.get("MP_PHYSICS"),
            "use_theta_m": attrs.get("USE_THETA_M"),
            "start_date": attrs.get("SIMULATION_START_DATE") or attrs.get("START_DATE"),
        }
        if self.has("XLAT") and self.has("XLONG"):
            lat = self.array("XLAT", 0)
            lon = self.array("XLONG", 0)
            facts["lat_range"] = [float(lat.min()), float(lat.max())]
            facts["lon_range"] = [float(lon.min()), float(lon.max())]
        return facts


@dataclass
class HistoryFrame:
    """One record of an open history, with :class:`cm1core.arwen_chain.ArwenFrame`'s interface."""

    history: WrfHistory
    record: int
    names: tuple[str, ...] | None = None
    _entries: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _header: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def directory(self) -> Path:
        """The source file (an extracted frame's is its directory); refusals and receipts name it."""

        return self.history.path

    @property
    def times(self) -> str:
        return self.history.times[self.record]

    @property
    def valid_s(self) -> float:
        return parse_wrf_time(self.times)

    @property
    def attributes(self) -> Mapping[str, Any]:
        return self.history.attributes

    @property
    def dimensions(self) -> Mapping[str, int]:
        return self.history.dimensions

    def _listed(self, name: str) -> bool:
        return self.names is None or name in self.names

    def has(self, name: str) -> bool:
        return self._listed(name) and self.history.has(name)

    def entry(self, name: str) -> Mapping[str, Any]:
        if not self.has(name):
            from .arwen_chain import _refuse as chain_refuse

            chain_refuse(
                "GM1-CHAIN-VARIABLE",
                f"the history record at {self.times} in {self.history.path.name} carries no {name}; the reader "
                "would have to invent the field",
                f"write {name} to the history (for P and PB: include_diagnostic_pressure)",
                option=name, value=os.fspath(self.history.path),
            )
        if name not in self._entries:
            raw = self.history.array(name, self.record).tobytes()
            row = self.history.field_row(name)
            row.update({"nbytes": len(raw), "sha256": _sha256(raw)})
            self._entries[name] = row
        return self._entries[name]

    def array(self, name: str) -> Any:
        self.entry(name)
        return self.history.array(name, self.record)

    @property
    def header(self) -> Mapping[str, Any]:
        """``frame.json`` as the extractor would write it for this record (every listed field is read once)."""

        if self._header is None:
            names = self.names if self.names is not None else CHAIN_VARIABLES
            rows = []
            offset = 0
            for name in names:
                if not self.history.has(name):
                    continue
                row = dict(self.entry(name))
                row["offset"] = offset
                offset += int(row["nbytes"])
                rows.append(dict(sorted(row.items())))
            self._header = {
                "absent": [name for name in names if not self.history.has(name)],
                "attributes": self.attributes, "dimensions": self.dimensions, "fields": rows,
                "schema": FRAME_SCHEMA, "source": self.history.source(self.record), "times": self.times,
            }
        return self._header


def open_history(path: Path | str) -> WrfHistory:
    """Open a WRF or ArWen history file (CDF-1, CDF-2, CDF-5 or netCDF-4)."""

    return WrfHistory(path)


def open_frames(paths: Iterable[Path | str], names: Sequence[str] | None = CHAIN_VARIABLES) -> list[HistoryFrame]:
    """Every record of every file, in time order: the chain's frame sequence straight from history files.

    The files stay open for as long as the frames are used.
    """

    frames = []
    for path in paths:
        history = open_history(path)
        frames.extend(history.frame(record, names) for record in range(history.records))
    frames.sort(key=lambda frame: frame.valid_s)
    return frames


__all__ = [
    "CHAIN_VARIABLES", "FORMATS", "FRAME_SCHEMA", "HistoryFrame", "MAP_PROJECTIONS", "REFUSAL_CODES", "WrfHistory",
    "file_sha256", "json_attribute", "open_frames", "open_history", "parse_wrf_time", "wrf_spelling",
]
