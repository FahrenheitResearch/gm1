"""CM1 r21.1's GrADS output (``output_format = 1``) from GM1 frames.

CM1 writes GrADS as raw direct-access binary32 (native byte order, no record
markers) plus a ``.ctl`` descriptor per file, in its own descriptor grammar:

* history (``writeout.F:9265-9528`` ``write_outputctl``, ``:9760-10000``
  ``open_file``): one file per grid -- ``s`` (scalars, 2-D fields included,
  written as one level), ``u``, ``v`` and ``w`` -- holding, for each time,
  every variable of that grid in ``writeout.F``'s order, each variable its
  levels bottom to top, each level ``nx*ny`` words with x fastest.  The
  names follow ``output_filetype``: 1 is ``cm1out_s.dat`` for every time, 2
  is ``cm1out_NNNNNN_s.dat`` per time (``nwrite``, 1 for t = 0) with ``dset
  ^cm1out_00%y4_s.dat`` and ``options template``, 3 is the per-process
  ``cm1out_000000_NNNNNN_s.dat`` (GM1 is one process);
* ``cm1out_metadata.dat``/``.ctl`` (``writeout.F:9536-9757``): per write,
  ``mtime``, ``nstep``, ``dt``, ``adt`` (adapt_dt), ``nwrite`` and, for a
  moving domain, ``umove``, ``vmove``, ``domainlocx``, ``domainlocy``;
* statistics (``statpack.F:615-640``, ``:1466-1533``): ``cm1out_stats.dat``,
  one record of every statistic per time, and ``cm1out_stats.ctl``;
* parcels (``parcel.F:1626-1690``, ``:2144-2211``): ``cm1out_pdata.dat`` and
  ``cm1out_parcel.ctl`` (:mod:`.pdata`).

Every descriptor line is written with the Fortran edit descriptors CM1 uses
(``es14.6e2`` coordinates, ``a12,1x,i6,' 99 ',a60,1x,a12`` variable lines,
the 120-character ``dset`` string), so a GM1 descriptor differs from CM1's
own for the same run only in:

* the version in the title (``cm1r21.1 (GM1 GPU port)``);
* ``tdef``, which counts the times GM1 wrote;
* the variables GM1 does not produce, which are left out of the list (and
  so out of the records); :attr:`GradsHistory.notes` names them with the
  reason, and the converter writes that list to ``cm1out_gm1.json``;
* for ``output_filetype = 3``, ``dset``: CM1's descriptor names the joined
  files (``cm1out_00%y4_s.dat``) that its per-process files must first be
  combined into; GM1's one process writes the whole domain, so its
  descriptor names the files it wrote (``cm1out_000000_00%y4_s.dat``).

``tests/data/cm1-ref-24x24/grads/`` holds CM1's own descriptors for the
reference runs, and the tests hold these to them line for line.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: constants.F:63 grads_undef.
UNDEF = -999999.9
#: input.F:13 maxstring: the dset string is written with its trailing blanks.
MAXSTRING = 120
#: The version GM1's descriptors name (CM1 writes its own cm1version).
GRADS_VERSION = "cm1r21.1 (GM1 GPU port)"
#: writeout.F:9384 ctlloop order; the ``i`` file (output_interp) needs terrain.
FILE_GRIDS = ("s", "u", "v", "w")
#: Which history file each CM1 grid letter is written to.
FILE_OF_GRID = {"2": "s", "s": "s", "u": "u", "v": "v", "w": "w"}
METADATA_DAT = "cm1out_metadata.dat"
METADATA_CTL = "cm1out_metadata.ctl"
STATS_DAT = "cm1out_stats.dat"
STATS_CTL = "cm1out_stats.ctl"


# ---------------------------------------------------------------- Fortran edit descriptors

def _es14(value: float) -> str:
    """``es14.6e2`` of a REAL."""

    import numpy as np

    return f"{float(np.float32(value)):14.6E}"


def _a(text: str, width: int) -> str:
    """``aW`` output of a character variable: its first W characters, blank padded."""

    return str(text)[:width].ljust(width)


def _unit(units: str, width: int, limit: int) -> str:
    """CM1's parenthesised unit field: ``(`` + the unit's first ``limit`` characters + ``)`` in ``width``.

    writeout.F:9502-9507 (a12, at most 10 characters of the unit) and
    statpack.F:1506-1510 / parcel.F:2186-2190 (a16, 14).
    """

    field = [" "] * width
    body = _a(units, limit)
    field[1:1 + limit] = list(body)
    field[0] = "("
    nn = min(len(str(units).rstrip()), limit)
    field[nn + 1] = ")"
    return "".join(field)


def _undef_line() -> str:
    import numpy as np

    return f"undef {float(np.float32(UNDEF)):10.1f}"


def _tdef(count: int) -> str:
    return f"tdef {int(count):10d} linear 00:00Z01JAN0001 1YR"


# ---------------------------------------------------------------- names

def data_filename(filetype: int, grid: str, nwrite: int) -> str:
    """writeout.F open_file: the history data file of one grid at one write."""

    if filetype == 1:
        return f"cm1out_{grid}.dat"
    if filetype == 2:
        return f"cm1out_{nwrite:06d}_{grid}.dat"
    if filetype == 3:
        return f"cm1out_000000_{nwrite:06d}_{grid}.dat"
    raise ValueError(f"output_filetype must be 1, 2 or 3, not {filetype}")


def ctl_filename(grid: str) -> str:
    return f"cm1out_{grid}.ctl"


def dset(filetype: int, grid: str) -> str:
    """The descriptor's data-set name (writeout.F:9317-9365; see the module note for filetype 3)."""

    if filetype == 1:
        return f"cm1out_{grid}.dat"
    if filetype == 2:
        return f"cm1out_00%y4_{grid}.dat"
    return f"cm1out_000000_00%y4_{grid}.dat"


# ---------------------------------------------------------------- descriptors

@dataclass(frozen=True)
class GradsVariable:
    """One history variable of a GrADS file: CM1's four strings."""

    name: str
    long_name: str
    units: str
    #: CM1's grid letter: ``2``, ``s``, ``u``, ``v`` or ``w``.
    grid: str

    def levels(self, nz: int) -> int:
        """levs_output: 0 for a 2-D field, maxk for s/u/v, maxk+1 for w (writeout.F:237, :7755)."""

        return {"2": 0, "s": nz, "u": nz, "v": nz, "w": nz + 1}[self.grid]

    def records(self, nz: int) -> int:
        return max(1, self.levels(nz))

    def row(self) -> list[str]:
        return [self.name, self.long_name, self.units, self.grid]


def _axis(n: int, first: float, step: float) -> str:
    return f"{n:6d} linear {_es14(first)} {_es14(step)}"


def _levels(axis: str, values: Iterable[float]) -> list[str]:
    values = list(values)
    return [f"{axis}def {len(values):6d} levels "] + [f"  {_es14(value)}" for value in values]


def horizontal_origins(settings: Any, geometry: Any) -> dict[str, float]:
    """xf(1), xh(1), yf(1), yh(1) in metres, as CM1's REAL words (param.F:6646-6690)."""

    import numpy as np

    f32 = np.float32

    def first(n: int, spacing: float) -> tuple[float, float]:
        spacing = float(f32(spacing))
        if settings.int("iorigin", 1) == 2:
            face1 = spacing * 0 - 0.5 * spacing * n
            face2 = spacing * 1 - 0.5 * spacing * n
        else:
            face1, face2 = 0.0, spacing
        return float(f32(face1)), float(f32(0.5 * (face1 + face2)))

    xf1, xh1 = first(geometry.nx, geometry.dx)
    yf1, yh1 = first(geometry.ny, geometry.dy)
    return {"xf1": xf1, "xh1": xh1, "yf1": yf1, "yh1": yh1}


def vertical_is_listed(settings: Any, geometry: Any) -> bool:
    """CM1 lists the levels when stretch_z is on (writeout.F:9428-9447).

    GM1 also lists them when the frame's own column is not uniform, whatever
    the settings say: a ``linear`` line would put every level at the wrong
    height.
    """

    return settings.int("stretch_z", 0) != 0 or not geometry.uniform


def history_ctl(grid: str, variables: Sequence[GradsVariable], settings: Any, geometry: Any,
                *, filetype: int, times: int, version: str = GRADS_VERSION) -> str:
    """One history descriptor, as ``write_outputctl`` writes it."""

    import numpy as np

    f32 = np.float32
    _aunit, conv = settings.distance_units
    conv = f32(conv)
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    origin = horizontal_origins(settings, geometry)
    dx, dy = f32(geometry.dx), f32(geometry.dy)
    lines = [f"dset ^{_a(dset(filetype, grid), MAXSTRING)}"]
    if filetype >= 2:
        lines.append("options template")
    units_words = "meters" if settings.int("outunits", 1) == 2 else "km"
    lines.append(f"title CM1 output, using version {version.strip()}; units of x,y,z are {units_words}; "
                 "time is generic, see cm1out_metadata for actual times")
    lines.append(_undef_line())
    if grid == "u":
        lines.append("xdef " + _axis(nx + 1, conv * f32(origin["xf1"]), conv * dx))
    else:
        lines.append("xdef " + _axis(nx, conv * f32(origin["xh1"]), conv * dx))
    if grid == "v":
        lines.append("ydef " + _axis(ny + 1, conv * f32(origin["yf1"]), conv * dy))
    else:
        lines.append("ydef " + _axis(ny, conv * f32(origin["yh1"]), conv * dy))
    listed = vertical_is_listed(settings, geometry)
    dz = f32(geometry.dz)
    if grid == "w":
        if listed:
            lines += _levels("z", (conv * geometry.zf).astype(f32))
        else:
            lines.append("zdef " + _axis(nz + 1, 0.0, conv * dz))
    else:
        if listed:
            lines += _levels("z", (conv * geometry.zh).astype(f32))
        else:
            lines.append("zdef " + _axis(nz, conv * f32(geometry.zh[0]), conv * dz))
    lines.append(_tdef(times))
    lines.append(f"vars {len(variables):6d}")
    for variable in variables:
        lines.append(f"{_a(variable.name, 12)} {variable.levels(nz):6d} 99 {_a(variable.long_name, 60)} "
                     f"{_unit(variable.units, 12, 10)}")
    lines.append("endvars")
    return "\n".join(lines) + "\n"


def metadata_names(settings: Any) -> list[tuple[str, str]]:
    """writeout.F:9557-9716: the metadata variables CM1 writes for these settings, in order."""

    rows = [("mtime", "model time (seconds since beginning of simulation)"),
            ("nstep", "time steps (since beginning of simulation)"),
            ("dt", "current time step (seconds)")]
    if settings.int("adapt_dt", 0) == 1:
        rows.append(("adt", "average time step (seconds) since last call to statpack"))
    rows.append(("nwrite", "writeout number"))
    if settings.int("imove", 0) >= 1:
        rows += [("umove", "umove (m/s)"), ("vmove", "vmove (m/s)"),
                 ("domainlocx", "x location of (center of) domain (m)"),
                 ("domainlocy", "y location of (center of) domain (m)")]
    return rows


def metadata_ctl(names: Sequence[tuple[str, str]], times: int, version: str = GRADS_VERSION) -> str:
    """writeout.F:9718-9751."""

    lines = [f"dset ^{METADATA_DAT}",
             f"title CM1 metadata for 3d output files; using version {version.strip()}",
             _undef_line(), "xdef 1 linear 0 1", "ydef 1 linear 0 1", "zdef 1 linear 0 1",
             _tdef(times), f"vars {len(names):6d}"]
    # format 209 with three items: output stops at a60 (the trailing 1x
    # writes nothing, as gfortran's own file shows).
    lines += [f"{_a(name, 12)} {0:6d} 99 {_a(desc, 60)}" for name, desc in names]
    lines.append("endvars")
    return "\n".join(lines) + "\n"


def metadata_record(settings: Any, frame: Any, geometry: Any, nwrite: int) -> list[float]:
    """One metadata record's words for a frame, in :func:`metadata_names` order."""

    import numpy as np

    f32 = np.float32
    moving = frame.moving_frame or {}
    row = [float(f32(frame.model_time_s)), float(f32(frame.step)), float(f32(geometry.dt))]
    if settings.int("adapt_dt", 0) == 1:
        # GM1 publishes the step it took; CM1's adt averages the steps since statpack.
        row.append(float(f32(geometry.dt)))
    row.append(float(f32(nwrite)))
    if settings.int("imove", 0) >= 1:
        row += [float(f32(moving.get("umove", settings.float("umove")) or 0.0)),
                float(f32(moving.get("vmove", settings.float("vmove")) or 0.0)),
                float(f32(moving.get("domainlocx", 0.0) or 0.0)),
                float(f32(moving.get("domainlocy", 0.0) or 0.0))]
    return row


def _statistic_line(name: str, long_name: str, units: str) -> str:
    return f"{_a(name, 12)} 1 99 {_a(long_name, 40)} {_unit(units, 16, 14)}"


def stats_ctl(specs: Sequence[Any], records: int, version: str = GRADS_VERSION) -> str:
    """statpack.F:1466-1533 write_statsctl."""

    lines = [f"dset ^{_a(STATS_DAT, MAXSTRING)}",
             f"title CM1 stats output, using version {version.strip()}; time is generic, "
             "see variable mtime for actual times",
             _undef_line(), "xdef 1 linear 0 1", "ydef 1 linear 0 1", "zdef 1 linear 0 1",
             _tdef(records), f"vars {len(specs):6d}"]
    lines += [_statistic_line(spec.name, spec.long_name, spec.units) for spec in specs]
    lines.append("endvars")
    return "\n".join(lines) + "\n"


def parcel_ctl(names: Sequence[str], descriptions: Sequence[str], units: Sequence[str], *,
               nparcels: int, records: int, version: str = GRADS_VERSION) -> str:
    """parcel.F:2144-2211 write_prclctl."""

    lines = [f"dset ^{_a('cm1out_pdata.dat', MAXSTRING)}",
             f"title CM1 parcel data output, using version {version.strip()}; time is generic, "
             "see variable mtime for actual times",
             _undef_line(), f"xdef {int(nparcels):10d} linear 1 1", "ydef          1 linear 0 1",
             "zdef          1 linear 0 1", _tdef(records), f"vars {len(names):6d}"]
    lines += [_statistic_line(name, desc, unit) for name, desc, unit in zip(names, descriptions, units)]
    lines.append("endvars")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- writing

def write_text_atomic(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="ascii")
    os.replace(partial, path)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, path)


def words(array: Any) -> bytes:
    import numpy as np

    return np.ascontiguousarray(array, dtype="<f4").tobytes()


def variables_by_file(variables: Iterable[GradsVariable]) -> dict[str, list[GradsVariable]]:
    """The variables of each history file, in the order given (writeout.F's)."""

    files: dict[str, list[GradsVariable]] = {grid: [] for grid in FILE_GRIDS}
    for variable in variables:
        files[FILE_OF_GRID[variable.grid]].append(variable)
    return {grid: rows for grid, rows in files.items() if rows}


class GradsHistory:
    """The history files of one conversion: a fixed variable list, written time by time.

    A GrADS file has one variable list for every time (and with
    ``output_filetype`` 2 or 3 one descriptor serves every time's file), so
    the list is fixed by the first frame converted.  A later frame that
    cannot supply one of them gets CM1's missing value (``undef``) in its
    records; a variable only a later frame has is not written, and is named
    in :attr:`notes`.
    """

    def __init__(self, out: Path, settings: Any, geometry: Any, *, filetype: int,
                 variables: Sequence[GradsVariable]) -> None:
        self.out = Path(out)
        self.settings = settings
        self.geometry = geometry
        self.filetype = int(filetype)
        self.variables = list(variables)
        self.files = variables_by_file(self.variables)
        self.notes: dict[str, str] = {}

    def frame_bytes(self, grid: str) -> int:
        g = self.geometry
        plane = {"s": g.nx * g.ny, "u": (g.nx + 1) * g.ny, "v": g.nx * (g.ny + 1), "w": g.nx * g.ny}[grid]
        return 4 * plane * sum(v.records(g.nz) for v in self.files[grid])

    def write_time(self, frame: Any, plan: Any, *, index: int, nwrite: int) -> list[str]:
        """Write one frame: at time ``index`` of each ``cm1out_<g>.dat`` (filetype 1), else its own files.

        Returns the data files written.
        """

        import numpy as np

        readers = {row.variable.name: row.read for row in plan.variables}
        for name in readers:
            if name not in {v.name for v in self.variables}:
                self.notes.setdefault(
                    name, "Not in the GrADS files: their variable list was fixed by the first frame converted, "
                          "which did not have it.")
        written = []
        g = self.geometry
        for grid, variables in self.files.items():
            name = data_filename(self.filetype, grid, nwrite)
            path = self.out / name
            chunks = []
            for variable in variables:
                read = readers.get(variable.name)
                if read is None:
                    shape = {"2": (g.ny, g.nx), "s": (g.nz, g.ny, g.nx), "u": (g.nz, g.ny, g.nx + 1),
                             "v": (g.nz, g.ny + 1, g.nx), "w": (g.nz + 1, g.ny, g.nx)}[variable.grid]
                    chunks.append(np.full(shape, np.float32(UNDEF), dtype="<f4").tobytes())
                    self.notes.setdefault(
                        variable.name, "Written as undef at the times whose frame could not supply it.")
                    continue
                data = words(read(frame))
                expected = 4 * variable.records(g.nz) * {"s": g.nx * g.ny, "u": (g.nx + 1) * g.ny,
                                                          "v": g.nx * (g.ny + 1), "w": g.nx * g.ny}[grid]
                if len(data) != expected:
                    raise ValueError(f"{variable.name}: {len(data)} bytes where the {grid} file takes {expected}")
                chunks.append(data)
            block = b"".join(chunks)
            if self.filetype == 1:
                offset = index * self.frame_bytes(grid)
                mode = "r+b" if path.exists() else "wb"
                with path.open(mode) as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    if size < offset:
                        # A time CM1 never wrote before this one: undef, not zeros.
                        gap = (offset - size) // 4
                        handle.write(np.full(gap, np.float32(UNDEF), dtype="<f4").tobytes())
                    handle.seek(offset)
                    handle.write(block)
            else:
                write_bytes_atomic(path, block)
            written.append(name)
        return written

    def write_descriptors(self, times: int) -> list[str]:
        names = []
        for grid, variables in self.files.items():
            name = ctl_filename(grid)
            write_text_atomic(self.out / name,
                              history_ctl(grid, variables, self.settings, self.geometry,
                                          filetype=self.filetype, times=times))
            names.append(name)
        return names


def write_metadata(out: Path, names: Sequence[tuple[str, str]], rows: Sequence[Sequence[float]]) -> list[str]:
    import numpy as np

    data = np.asarray(rows, dtype="<f4").reshape(len(rows), len(names))
    write_bytes_atomic(Path(out) / METADATA_DAT, data.tobytes())
    write_text_atomic(Path(out) / METADATA_CTL, metadata_ctl(names, len(rows)))
    return [METADATA_CTL, METADATA_DAT]


def write_stats(out: Path, rows: Sequence[tuple[float, Mapping[str, float]]], specs: Sequence[Any]) -> list[str]:
    """``cm1out_stats.dat`` (one record of every statistic per time, undef where missing) and its descriptor."""

    import numpy as np

    missing = np.float32(UNDEF)
    table = np.asarray([[values.get(spec.name, missing) for spec in specs] for _time, values in rows],
                       dtype="<f4").reshape(len(rows), len(specs))
    write_bytes_atomic(Path(out) / STATS_DAT, table.tobytes())
    write_text_atomic(Path(out) / STATS_CTL, stats_ctl(specs, len(rows)))
    return [STATS_CTL, STATS_DAT]


def variables_from_rows(rows: Iterable[Sequence[str]]) -> list[GradsVariable]:
    return [GradsVariable(*row) for row in rows]


__all__ = [
    "FILE_GRIDS",
    "GRADS_VERSION",
    "GradsHistory",
    "GradsVariable",
    "METADATA_CTL",
    "METADATA_DAT",
    "STATS_CTL",
    "STATS_DAT",
    "UNDEF",
    "ctl_filename",
    "data_filename",
    "dset",
    "history_ctl",
    "metadata_ctl",
    "metadata_names",
    "metadata_record",
    "parcel_ctl",
    "stats_ctl",
    "variables_by_file",
    "variables_from_rows",
    "write_metadata",
    "write_stats",
]
