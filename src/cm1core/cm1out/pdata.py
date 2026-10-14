"""CM1's parcel output (``cm1out_pdata``) from a GM1 run's parcel track.

A Cartesian run with ``iprcl = 1`` writes its whole track once, at the end,
beside its history frames (``cuda_cartesian_runtime._publish_lagrangian_track``):

``<frames>/lagrangian-track/parcel-attributes.f32``
    ``(frames, npvals, nparcels)`` little-endian float32;
``parcel-attribute-table.tsv``
    ``index, name, units, filled`` per attribute, in ``parcel.F``'s order
    (``setup_parcel_vars``, :1898-2130), GM1's extension columns after;
``parcel-track.json``
    the receipt, with ``model_times_s`` and the arena's sha256.

CM1 writes the same variables in the same order (``parcel.F:1626-1690``):

* ``output_format = 2``: ``cm1out_pdata.nc`` as ``writepdata_nc``
  (``writeout_nc.F:1810-1961``) lays it out -- dimensions ``xh`` (the parcel
  number, "west-east location ... actually, really parcel ID number"),
  ``yh = zh = 1`` and ``time``; each attribute a ``(time, xh)`` float with
  CM1's 40-character description as ``long_name``;
* ``output_format = 1``: ``cm1out_pdata.dat``, one direct-access record of
  ``(npvals, nparcels)`` words per time -- the arena's own layout, frame by
  frame -- and ``cm1out_parcel.ctl``.

The words are copied, not re-formed: reading ``cm1out_pdata.nc`` back and
transposing gives ``parcel-attributes.f32`` word for word (:func:`read_back`).
What GM1's track holds that CM1's does not is said in the file (``gm1_*``
attributes, and :attr:`PdataResult.notes`):

* GM1 records its first parcel frame one ``prclfrq`` after the start; CM1
  also writes the seeded positions at t = 0;
* an attribute GM1 declares but cannot fill (``parcel-attribute-table.tsv``
  ``filled = no``), and a parcel the integrator stopped, hold -1e30;
* each segment of a resumed run seeds its parcels again (GM1's restarts do
  not carry parcel positions, CM1's do), so a parcel number names the same
  air only within one segment.

Nothing here imports CuPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import grads as _grads
from . import netcdf as _netcdf

TRACK_DIRECTORY = "lagrangian-track"
ARENA = "parcel-attributes.f32"
TABLE = "parcel-attribute-table.tsv"
RECEIPT = "parcel-track.json"
NETCDF_FILENAME = "cm1out_pdata.nc"
GRADS_DAT = "cm1out_pdata.dat"
GRADS_CTL = "cm1out_parcel.ctl"
#: GM1's sentinel for an unplaced parcel or an unfilled attribute (lagrangian_stage.parcel_frame).
SENTINEL = -1.0e30

#: parcel.F:1898-2130 setup_parcel_vars: desc_prcl (character(len=40)).
CM1_DESCRIPTIONS: Mapping[str, str] = {
    "x": "x position", "y": "y position", "z": "z position (above sea level)",
    "u": "u velocity", "v": "v velocity", "w": "w velocity",
    "mtime": "model time (seconds since beginning of simulation)",
    "th": "potential temperature", "t": "temperature", "prs": "pressure",
    "qv": "water vapor mixing ratio",
    "kmh": "horiz eddy viscosity for momentum", "kmv": "vert eddy viscosity for momentum",
    "khh": "horiz eddy diffusivity for scalars", "khv": "vert eddy diffusivity for scalars",
    "tke": "subgrid tke", "dbz": "reflectivity", "b": "buoyancy",
    "vpg": "vertical perturbation pressure gradient", "zvort": "vertical vorticity",
    "rho": "dry-air density", "qsl": "saturation mixing ratio wrt liquid",
    "qsi": "saturation mixing ratio wrt ice", "znt": "surface roughness length",
    "ust": "surface friction velocity", "zs": "terrain height",
    "sigma": "sigma (nondimensional height)",
}


def description(name: str) -> str:
    """CM1's desc_prcl for an attribute, cut to its 40 characters.

    Species columns are described by their own name (parcel.F:1995-2024);
    GM1's extension columns say they are GM1's.
    """

    if name in CM1_DESCRIPTIONS:
        text = CM1_DESCRIPTIONS[name]
    elif name[:6] in ("xvortb", "yvortb", "zvortb"):
        text = f"GM1 vorticity budget {name}"
    else:
        text = name
    return text[:40]


class PdataError(ValueError):
    """A parcel track that cannot be written as CM1's pdata, in plain words."""


@dataclass(frozen=True)
class Track:
    """One segment's published parcel track."""

    directory: Path
    names: tuple[str, ...]
    units: tuple[str, ...]
    filled: tuple[bool, ...]
    times: tuple[float, ...]
    frames: int
    nparcels: int
    sha256: str

    @property
    def npvals(self) -> int:
        return len(self.names)

    def signature(self) -> list[Any]:
        return [os.fspath(self.directory), self.sha256, self.frames, self.nparcels]

    def array(self) -> Any:
        """The arena as a read-only ``(frames, npvals, nparcels)`` float32 view."""

        import numpy as np

        path = self.directory / ARENA
        if self.frames == 0:
            return np.zeros((0, self.npvals, self.nparcels), dtype="<f4")
        words = np.memmap(path, dtype="<f4", mode="r")
        expected = self.frames * self.npvals * self.nparcels
        if words.size != expected:
            raise PdataError(f"{path} holds {words.size} words; its receipt says {expected}")
        return words.reshape(self.frames, self.npvals, self.nparcels)


def read_track(directory: Path | str) -> Track | None:
    """The track in ``directory`` (a ``lagrangian-track``), or None when it has none."""

    directory = Path(directory)
    arena, table, receipt = directory / ARENA, directory / TABLE, directory / RECEIPT
    if not (arena.is_file() and table.is_file() and receipt.is_file()):
        return None
    names, units, filled = [], [], []
    for number, line in enumerate(table.read_text(encoding="utf-8").splitlines()):
        if number == 0 or not line.strip():
            continue  # the header row
        parts = line.split("\t")
        if len(parts) != 4 or int(parts[0]) != len(names):
            raise PdataError(f"{table}:{number + 1}: expected 'index, name, units, filled' in order")
        names.append(parts[1])
        units.append(parts[2])
        filled.append(parts[3].strip() == "yes")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    shape = record.get("arena_shape_frames_npvals_nparcels") or []
    if len(shape) != 3 or int(shape[1]) != len(names):
        raise PdataError(f"{receipt} does not describe a ({len(names)}-attribute) parcel arena")
    times = tuple(float(t) for t in record.get("model_times_s") or ())
    if len(times) != int(shape[0]):
        raise PdataError(f"{receipt} lists {len(times)} times for {shape[0]} frames")
    size = arena.stat().st_size
    if size != 4 * int(shape[0]) * int(shape[1]) * int(shape[2]):
        raise PdataError(f"{arena} is {size} bytes; its receipt says {list(shape)} float32 words")
    return Track(directory, tuple(names), tuple(units), tuple(filled), times, int(shape[0]), int(shape[2]),
                 str(record.get("arena_sha256") or ""))


def lineage_tracks(layout: Any) -> list[tuple[Track, float | None]]:
    """The run's tracks along its lineage, oldest first, each with the time it counts to.

    In a ``gpuwm-cm1 run`` directory these are the tracks of the run's line
    as its frames follow it (``RunDirectory.frame_line``: an ancestor counts
    up to the time its descendant resumed from); elsewhere every segment's
    track, oldest first.
    """

    found: list[tuple[Track, float | None]] = []
    run = layout.run_directory()
    if run is not None:
        for segment, until in run.frame_line():
            track = read_track(segment.frames_dir / TRACK_DIRECTORY)
            if track is not None:
                found.append((track, until))
        return found
    for segment in layout.segments:
        track = read_track(Path(segment.frames_root) / TRACK_DIRECTORY)
        if track is not None:
            found.append((track, None))
    return found


@dataclass
class Merged:
    """The parcel frames that form the run's track: each time from its newest track."""

    track: Track
    rows: list[tuple[float, Track, int]]
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def times(self) -> list[float]:
        return [time for time, _track, _index in self.rows]

    def signature(self) -> list[Any]:
        return [[time, track.sha256, index] for time, track, index in self.rows]

    def record(self, position: int) -> Any:
        _time, track, index = self.rows[position]
        return track.array()[index]


def merge(tracks: Sequence[tuple[Track, float | None]]) -> Merged | None:
    """One track from the lineage's; None when the run published none."""

    if not tracks:
        return None
    newest = tracks[-1][0]
    notes: dict[str, str] = {}
    by_time: dict[float, tuple[Track, int]] = {}
    contributing: list[Track] = []
    for track, until in tracks:
        if (track.names, track.nparcels) != (newest.names, newest.nparcels):
            notes[f"skipped {track.directory}"] = (
                f"This segment tracked {track.nparcels} parcels with {track.npvals} attributes, the newest "
                f"{newest.nparcels} with {newest.npvals}; only tracks like the newest are written."
            )
            continue
        used = False
        for index, time in enumerate(track.times):
            if until is not None and time > until:
                continue
            by_time[time] = (track, index)
            used = True
        if used:
            contributing.append(track)
    if len(contributing) > 1:
        starts = sorted(min(t for t in track.times) for track in contributing[1:] if track.times)
        notes["reseeded"] = (
            "Each resumed segment seeds its parcels again (GM1's restarts do not carry parcel positions); "
            "a parcel number names the same air only between reseeds. The track reseeds before "
            + ", ".join(f"t = {t:g} s" for t in starts) + "."
        )
    rows = [(time, *by_time[time]) for time in sorted(by_time)]
    unfilled = [name for name, filled in zip(newest.names, newest.filled) if not filled]
    if unfilled:
        notes["unfilled"] = (
            "Declared but not filled by GM1 (every value is -1e30): " + ", ".join(unfilled) + "."
        )
    if rows and rows[0][0] > 0.0:
        notes["first_time"] = (
            f"GM1 records its first parcel frame at t = {rows[0][0]:g} s; CM1 also writes the seeded "
            "positions at t = 0."
        )
    return Merged(newest, rows, notes)


def _provenance(producer: str, merged: Merged) -> dict[str, Any]:
    unfilled = [name for name, filled in zip(merged.track.names, merged.track.filled) if not filled]
    return {
        "gm1_producer": producer,
        "gm1_note": (
            "Parcel trajectories simulated by GM1 (a GPU port of CM1 r21.1), not by CM1. Names, units "
            "and layout follow CM1 r21.1 writepdata_nc; the words are GM1's parcel-attributes.f32. "
            "-1e30 marks a parcel the integrator stopped or an attribute GM1 does not fill."
        ),
        "gm1_tracks": " ".join(sorted({os.fspath(track.directory) for _t, track, _i in merged.rows})),
        "gm1_arena_sha256": " ".join(sorted({track.sha256 for _t, track, _i in merged.rows})),
        "gm1_unfilled": " ".join(unfilled) or "none",
        "gm1_notes": json.dumps(merged.notes, sort_keys=True),
    }


def write_netcdf(path: Path, merged: Merged, *, producer: str, deflate: int = 0) -> None:
    """``cm1out_pdata.nc`` as writepdata_nc (writeout_nc.F:1810-1961) lays it out."""

    import netCDF4
    import numpy as np

    track = merged.track
    partial = path.with_name(path.name + ".partial")
    ds = netCDF4.Dataset(partial, "w", format="NETCDF4_CLASSIC")
    try:
        ds.set_fill_off()
        ds.createDimension("xh", track.nparcels)
        ds.createDimension("yh", 1)
        ds.createDimension("zh", 1)
        ds.createDimension("time", None)
        for name, long_name, units in (
            ("xh", "west-east location ... actually, really parcel ID number", "degree_east"),
            ("yh", "south-north location", "degree_north"),
            ("zh", "height", "m"),
        ):
            var = ds.createVariable(name, "f4", (name,))
            var.long_name, var.units = long_name, units
        var = ds.createVariable("time", "f4", ("time",))
        var.long_name, var.units, var.axis = "time", "seconds", "T"
        variables = []
        for name, units in zip(track.names, track.units):
            var = ds.createVariable(name, "f4", ("time", "xh"), zlib=deflate > 0,
                                    complevel=deflate or 1, shuffle=deflate > 0)
            var.long_name = description(name)
            var.units = units
            variables.append(var)
        ds.setncattr("CM1 version", _netcdf.CM1_VERSION)
        ds.setncattr("Conventions", "CF-1.7")
        ds.setncattr("missing_value", np.float32(_grads.UNDEF))
        for key, value in _provenance(producer, merged).items():
            ds.setncattr(key, value)
        ds["xh"][:] = np.arange(1, track.nparcels + 1, dtype=np.float32)
        ds["yh"][:] = np.float32(0.0)
        ds["zh"][:] = np.float32(0.0)
        ds["time"][:] = np.asarray(merged.times, dtype=np.float32)
        for position in range(len(merged.rows)):
            record = merged.record(position)
            for column, var in enumerate(variables):
                var[position, :] = record[column]
    finally:
        ds.close()
    os.replace(partial, path)


def write_grads(out: Path, merged: Merged) -> list[str]:
    """``cm1out_pdata.dat`` (the arena's own records, time by time) and ``cm1out_parcel.ctl``."""

    track = merged.track
    partial = out / (GRADS_DAT + ".partial")
    with partial.open("wb") as handle:
        for position in range(len(merged.rows)):
            handle.write(_grads.words(merged.record(position)))
    os.replace(partial, out / GRADS_DAT)
    _grads.write_text_atomic(
        out / GRADS_CTL,
        _grads.parcel_ctl(track.names, [description(n) for n in track.names], track.units,
                          nparcels=track.nparcels, records=len(merged.rows)),
    )
    return [GRADS_CTL, GRADS_DAT]


def read_back(path: Path | str) -> tuple[Any, list[str], list[float]]:
    """(``(times, npvals, nparcels)`` float32, attribute names, times) from a written cm1out_pdata.

    ``path`` is ``cm1out_pdata.nc`` or ``cm1out_pdata.dat`` (whose descriptor
    sits beside it).  The array is the arena's layout, so a round trip is a
    word-for-word comparison.
    """

    import numpy as np

    path = Path(path)
    if path.suffix == ".nc":
        import netCDF4

        ds = netCDF4.Dataset(path)
        try:
            names = [n for n in ds.variables if n not in ("xh", "yh", "zh", "time")]
            times = [float(t) for t in ds["time"][:]]
            data = np.stack([np.asarray(ds[n][:], dtype="<f4") for n in names], axis=1) if names else \
                np.zeros((len(times), 0, len(ds.dimensions["xh"])), dtype="<f4")
            return np.ascontiguousarray(data), names, times
        finally:
            ds.close()
    ctl = path.with_name(GRADS_CTL)
    lines = ctl.read_text(encoding="ascii").splitlines()
    nparcels = int(next(line for line in lines if line.startswith("xdef")).split()[1])
    start = next(i for i, line in enumerate(lines) if line.startswith("vars"))
    names = [line.split()[0] for line in lines[start + 1:] if line.split() and line.split()[0] != "endvars"]
    words = np.fromfile(path, dtype="<f4")
    data = words.reshape(-1, len(names), nparcels)
    column = names.index("mtime") if "mtime" in names else None
    times = [float(data[t, column, 0]) if column is not None else float(t) for t in range(data.shape[0])]
    return data, names, times


def sha256_words(array: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


@dataclass
class PdataResult:
    files: list[str]
    parcels: int
    times: int
    notes: dict[str, str]


__all__ = [
    "ARENA",
    "CM1_DESCRIPTIONS",
    "GRADS_CTL",
    "GRADS_DAT",
    "Merged",
    "NETCDF_FILENAME",
    "PdataError",
    "PdataResult",
    "RECEIPT",
    "TABLE",
    "TRACK_DIRECTORY",
    "Track",
    "description",
    "lineage_tracks",
    "merge",
    "read_back",
    "read_track",
    "sha256_words",
    "write_grads",
    "write_netcdf",
]
