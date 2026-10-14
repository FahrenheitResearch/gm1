"""``gpuwm-cm1 cm1out``: turn a run's published frames into CM1's output files.

One pass converts every published frame that has not been converted yet,
rewrites the statistics (``cm1out_stats.nc``, or ``cm1out_stats.dat``/``.ctl``
for GrADS) and, when the run tracked parcels, writes CM1's parcel file
(``cm1out_pdata.nc``, or ``cm1out_pdata.dat`` with ``cm1out_parcel.ctl``;
:mod:`.pdata`).  ``follow`` repeats the pass until the run's event log holds
its ``end`` record and no frame is left, so a sidecar can run beside the
model and exit with it.

The format is the namelist's ``output_format``: netCDF (:mod:`.netcdf`) or
GrADS (:mod:`.grads`), each in the ``output_filetype`` layout CM1 uses.  A
Morrison run's ``dbz`` and ``cref`` are computed from each frame with CM1's
``calc_refl10cm`` (:mod:`.reflectivity`).

What has been converted is kept in ``<out>/.cm1out-state.json`` (GrADS:
``.cm1out-state-grads.json``, so both formats can share the folder): a frame is
converted once (its arena digest is recorded), a later segment's frame at a
time already written replaces it (a resumed run supersedes what it resumed
from), and ``force`` starts again from nothing.  Each per-time file is
written under a ``.partial`` name and renamed into place, so a reader never
opens a half-written file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time as _time
from typing import Any, Callable, Mapping

from . import fields as _fields
from . import frames as _frames
from . import grads as _grads
from . import netcdf as _netcdf
from . import pdata as _pdata
from . import settings as _settings
from . import stats as _stats

STATE_FILENAME = ".cm1out-state.json"
#: Each format keeps its own record, so GrADS and netCDF can sit in one folder
#: (settings.toml ``[output] also_netcdf``).
STATE_FILENAMES = {"nc": STATE_FILENAME, "grads": ".cm1out-state-grads.json"}
STATE_SCHEMA = "gm1.cm1out-state/v1"
SUMMARY_SCHEMA = "gm1.cm1out-summary/v1"
FORMATS = ("nc", "grads")
#: What a GrADS conversion says about itself (netCDF files carry it as gm1_* attributes).
NOTES_FILENAME = "cm1out_gm1.json"
NOTES_SCHEMA = "gm1.cm1out-grads-notes/v1"


class Cm1outError(ValueError):
    """A conversion that cannot proceed, said in plain words."""


@dataclass
class Summary:
    out: Path
    format: str
    filetype: int
    settings_source: str
    files: list[str] = field(default_factory=list)
    stats_file: str | None = None
    frames_total: int = 0
    frames_converted: int = 0
    variables: list[str] = field(default_factory=list)
    not_produced: dict[str, str] = field(default_factory=dict)
    switches_not_produced: dict[str, str] = field(default_factory=dict)
    statistics: list[str] = field(default_factory=list)
    statistics_not_produced: dict[str, str] = field(default_factory=dict)
    statistics_derived: list[str] = field(default_factory=list)
    seconds: list[dict[str, float]] = field(default_factory=list)
    ended: bool = False
    end_reason: str | None = None
    #: CM1's parcel file(s), when the run tracked parcels.
    pdata_files: list[str] = field(default_factory=list)
    pdata: dict[str, Any] = field(default_factory=dict)
    #: What a GrADS conversion changed from CM1's own layout, and why.
    notes: dict[str, str] = field(default_factory=dict)
    #: The netCDF written beside GrADS output (settings.toml ``[output] also_netcdf``).
    also: "Summary | None" = None

    def record(self) -> dict[str, Any]:
        return {
            "schema": SUMMARY_SCHEMA,
            "out": os.fspath(self.out),
            "format": self.format,
            "output_filetype": self.filetype,
            "settings": self.settings_source,
            "files": list(self.files),
            "stats_file": self.stats_file,
            "frames_total": self.frames_total,
            "frames_converted_now": self.frames_converted,
            "variables": list(self.variables),
            "not_produced": dict(self.not_produced),
            "switches_not_produced": dict(self.switches_not_produced),
            "statistics": list(self.statistics),
            "statistics_not_produced": dict(self.statistics_not_produced),
            "statistics_derived": list(self.statistics_derived),
            "seconds_per_frame": list(self.seconds),
            "run_ended": self.ended,
            "end_reason": self.end_reason,
            "pdata_files": list(self.pdata_files),
            "pdata": dict(self.pdata),
            "notes": dict(self.notes),
            "also": None if self.also is None else self.also.record(),
            "open_with": self.open_with(),
        }

    def open_with(self) -> dict[str, str]:
        """Ready-to-copy lines that open these files the way CM1 users open CM1's."""

        out = os.fspath(self.out)
        lines: dict[str, str] = {}
        if self.pdata_files:
            lines["parcels"] = (f"open {out}/{_pdata.GRADS_CTL}" if self.format == "grads"
                                else f"xr.open_dataset('{out}/{_pdata.NETCDF_FILENAME}')")
        if not self.files:
            return lines
        if self.format == "grads":
            lines["grads"] = f"open {out}/{_grads.ctl_filename('s')}"
            if self.stats_file:
                lines["statistics"] = f"open {out}/{_grads.STATS_CTL}"
            return lines
        if self.filetype == 1:
            lines["ncview"] = f"ncview {out}/cm1out.nc"
            lines["xarray"] = f"xr.open_dataset('{out}/cm1out.nc')"
        else:
            lines["ncview"] = f"ncview {out}/{self.files[-1]}"
            pattern = "cm1out_0*.nc" if self.filetype == 2 else "cm1out_000000_*.nc"
            lines["xarray"] = f"xr.open_mfdataset('{out}/{pattern}', combine='nested', concat_dim='time')"
        if self.stats_file:
            lines["statistics"] = f"xr.open_dataset('{out}/{self.stats_file}')"
        return lines


def converted_lost(lost: int, converted: int, why: str) -> str:
    """Convert again (``--force``) on a run some of whose converted times have lost their saved frames."""

    return (f"{lost} of this run's {converted} converted times have lost their saved frames, so their files cannot "
            f"be written again: {why} would delete them. Nothing was deleted; its files are kept. Without --force, "
            "a pass converts only the frames not converted yet.")


def record_lost(lost: int, held: int, unread: list[str]) -> str:
    """A pass over a folder whose record of what was converted is missing or unreadable, while its files hold times
    whose saved frames are gone (or files that cannot be read)."""

    if lost:
        which = "has lost its saved frame" if lost == 1 else "have lost their saved frames"
        what = f"the files there hold {held} converted times, {lost} of which {which} and could not be written again"
    else:
        what = (f"{', '.join(unread[:3])}{' and more' if len(unread) > 3 else ''} cannot be read, so what the files "
                "there hold is not known")
    return (f"This folder's record of what was converted ({STATE_FILENAME}, or {STATE_FILENAMES['grads']} for "
            f"GrADS) is missing or cannot be read, and {what}: converting without the record could write over "
            "them. Nothing was changed; the files are kept. Convert into a new folder with --out.")


def _producer() -> str:
    from .._version import __version__

    return f"gpuwm-cm1 {__version__} cm1out"


def _nwrite(time_s: float, interval: float, ordinal: int) -> int:
    """CM1's history number: 1 at t = 0, one more per ``tapfrq``."""

    if interval > 0.0:
        count = time_s / interval
        if abs(count - round(count)) <= 1e-6 * max(1.0, abs(count)):
            return int(round(count)) + 1
    return ordinal + 1


def assign_nwrite(time_s: float, interval: float, ordinal: int, taken: Mapping[int, float]) -> int:
    """The history number of a new time, never one another time already holds.

    CM1 counts its writes; a frame on the ``tapfrq`` grid gets the number CM1
    would give it (so a run that did not publish t = 0 still starts at 2).  A
    frame off that grid (the namelist's tapfrq is not the interval the frames
    were written at) takes the next number in time order, and a later on-grid
    frame whose number is already taken does the same, so no time's file is
    ever written over by another's.
    """

    for candidate in (_nwrite(time_s, interval, ordinal), ordinal + 1):
        if taken.get(candidate, time_s) == time_s:
            return candidate
    return max(taken) + 1


def _reflectivity_reader() -> Callable[..., Any] | None:
    """The CPU reflectivity step, when this build has one (``cm1out/reflectivity.py``).

    It exports ``reader(frame, name)``, returning a callable that gives the
    ``dbz`` or ``cref`` array for the frame, or None when it cannot.
    """

    try:
        from . import reflectivity  # type: ignore[attr-defined]
    except ImportError:
        return None
    return getattr(reflectivity, "reader", None)


def _load_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if state.get("schema") == STATE_SCHEMA else {}


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


class Converter:
    """One run's conversion, pass by pass."""

    def __init__(self, target: Path | str, *, out: Path | str | None = None, fmt: str | None = None,
                 filetype: int | None = None, namelist: Path | str | None = None,
                 deck: Path | str | None = None, stats: bool = True, deflate: int = 0,
                 force: bool = False, log: Callable[[str], None] | None = None,
                 wait_for_history: bool = False, pdata: bool = True) -> None:
        self.target = Path(target)
        try:
            self.layout = _frames.discover(target)
        except _frames.FrameError:
            # A sidecar started beside (or just before) its run: the run
            # directory is there, its first segment or frame is not yet.
            if not (wait_for_history and self.target.is_dir()):
                raise
            self.layout = _frames.RunLayout("rundir", self.target, ())
        try:
            self.settings = _settings.resolve(self.layout, namelist=namelist, deck=deck)
        except _settings.SettingsError as error:
            raise Cm1outError(str(error)) from error
        chosen = fmt or self.settings.output_format
        if chosen not in FORMATS:
            raise Cm1outError(f"--format must be nc or grads, not {chosen!r}")
        self.format = chosen
        self.filetype = int(filetype if filetype is not None else self.settings.output_filetype)
        if self.filetype not in (1, 2, 3):
            raise Cm1outError(f"output_filetype must be 1, 2 or 3, not {self.filetype}")
        self.out = Path(out) if out is not None else self.layout.default_out
        self.stats_enabled = stats
        self.pdata_enabled = pdata
        self.deflate = int(deflate)
        if not 0 <= self.deflate <= 9:
            raise Cm1outError("--deflate takes a level from 0 (off) to 9")
        self.log = log or (lambda _message: None)
        if self.format == "grads" and self.deflate:
            # GrADS records are raw words; CM1 compresses only its netCDF.
            self.log("GrADS files are not compressed; --deflate applies to netCDF only")
            self.deflate = 0
        self.reflectivity = _reflectivity_reader()
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / STATE_FILENAMES[self.format]
        previous = _load_state(self.state_path)
        if not previous:
            # No record of what was converted (deleted, or torn), but the folder may hold this format's files: what
            # they hold is read from them.  A time whose frame is gone could not be written again, and converting
            # without the record would write over it, or write the statistics from fewer times.
            held, unread = self._held()
            lost = self._held_lost(held) if held else []
            if lost or unread:
                raise Cm1outError(record_lost(len(lost), len(held), unread))
        state = {} if force else previous
        why = "converting every frame again"
        if state and (state.get("format") != self.format or state.get("output_filetype") != self.filetype
                      or state.get("deflate", 0) != self.deflate):
            # Another layout or compression: the files the old state wrote
            # would sit beside the new ones (or be appended to), so they go.
            state = {}
            force = True
            why = "writing them in another layout or compression"
        if force:
            # A converted time whose frame is gone cannot be written again: its file is all that is left of it.
            lost = self._lost(previous.get("frames") or {})
            if lost:
                raise Cm1outError(converted_lost(len(lost), len(previous.get("frames") or {}), why))
            self._remove_previous()
        self.state: dict[str, Any] = state or self._fresh_state()
        self.geometry: _settings.Geometry | None = None
        self.species: _settings.Species | None = None
        self.summary = Summary(self.out, self.format, self.filetype,
                               f"{self.settings.source} {self.settings.source_path or ''}".strip())

    def _fresh_state(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA, "format": self.format, "output_filetype": self.filetype,
            "deflate": self.deflate, "frames": {}, "stats": {}, "stat_specs": [],
            "stats_not_produced": {}, "stats_derived": [], "not_produced": {}, "variables": [],
            "switches_not_produced": {}, "formed": {},
            # GrADS: the history variable list (fixed by the first frame), the
            # grid the descriptors describe, and what differs from CM1's layout.
            "grads_variables": [], "grads_geometry": None, "grads_notes": {},
            "pdata": None,
        }

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Look for the run's layout again: a new segment, or the first frames of a run that just started."""

        try:
            self.layout = _frames.discover(self.target)
        except _frames.FrameError:
            pass

    def _unlink_named(self, name: str) -> None:
        # Only a file this converter names, never a path the state file could point elsewhere.
        if name.startswith("cm1out") and Path(name).name == name and (self.out / name).is_file():
            (self.out / name).unlink()

    def _remove_previous(self) -> None:
        previous = _load_state(self.state_path)
        for row in (previous.get("frames") or {}).values():
            for name in _row_files(row):
                self._unlink_named(name)
        if previous:
            # Only this format's files: the other format's may share the folder.
            if self.format == "grads":
                fixed = [_grads.METADATA_CTL, _grads.METADATA_DAT, _grads.STATS_CTL, _grads.STATS_DAT,
                         _pdata.GRADS_CTL, _pdata.GRADS_DAT, NOTES_FILENAME]
                fixed += [_grads.ctl_filename(grid) for grid in _grads.FILE_GRIDS]
                fixed += [_grads.data_filename(1, grid, 0) for grid in _grads.FILE_GRIDS]
            else:
                fixed = ["cm1out.nc", _stats.STATS_FILENAME, _pdata.NETCDF_FILENAME]
            for name in fixed:
                self._unlink_named(name)
        suffixes = (".dat.partial", ".ctl.partial", ".json.partial") if self.format == "grads" else (".nc.partial",)
        for partial in self.out.glob("cm1out*.partial"):
            if partial.name.endswith(suffixes):
                partial.unlink()
        if self.state_path.is_file():
            self.state_path.unlink()

    def _latest(self) -> dict[float, tuple[str, Path]]:
        """The run's published frames by model time (a later segment wins): (key, path)."""

        latest: dict[float, tuple[str, Path]] = {}
        for segment, path in self.layout.frames():
            location = path / _frames.LOCATION
            try:
                time_s = float(json.loads(location.read_text(encoding="utf-8"))["model_time_s"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                time_s = _frames.open_frame(path).model_time_s
            key = _netcdf._relative(path, self.layout.root)
            latest[time_s] = (key, path)
        return latest

    def _left(self, rows: Mapping[str, Any], latest: Mapping[float, tuple[str, Path]]) -> list[str]:
        """The converted times of ``rows`` that belong to a branch the run has left, as its run.json says (a resume
        from an earlier restart, or a start over), in a run that still has saved frames.  A time whose frame was
        only deleted (by hand, or with its segment's folder) is not one of them."""

        run = self.layout.run_directory()
        if run is None or not self.layout.lineage_settled() or not self._has_frames():
            return []
        from ..cm1run.rundir import off_the_line

        line = run.recorded_line()
        keys = {key for key, _path in latest.values()}
        return [time_key for time_key, row in rows.items()
                if isinstance(row, Mapping) and row.get("frame") not in keys and off_the_line(line, row, time_key)]

    def _has_frames(self) -> bool:
        return any(_frames.published_frames(segment.frames_root) for segment in self.layout.segments)

    def _lost(self, rows: Mapping[str, Any], latest: Mapping[float, tuple[str, Path]] | None = None) -> list[str]:
        """The converted times of ``rows`` whose saved frame is gone though the run never left them: no frame of
        the run is at their time, and its records do not put them on a branch it has left.  Their files are all
        there is of them."""

        latest = self._latest() if latest is None else latest
        keys = {key for key, _path in latest.values()}
        times = {round(float(t), 6) for t in latest}
        left = set(self._left(rows, latest))
        lost = []
        for time_key, row in rows.items():
            if not isinstance(row, Mapping) or not row.get("frame") or row.get("frame") in keys or time_key in left:
                continue
            try:
                t = float(row["time"] if row.get("time") is not None else time_key)
            except (TypeError, ValueError):
                continue
            if round(t, 6) not in times:
                lost.append(time_key)
        return lost

    def _held(self) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """What this format's converted files in the folder hold, read from the files themselves (for a record that
        is missing or cannot be read): ``({key: {"frame", "time", "file"}}, the files that cannot be read)``.  A
        netCDF file says its times, and a file of one time the frame it came from; GrADS files are named in
        ``cm1out_gm1.json`` by their frames alone."""

        import math

        held: dict[str, dict[str, Any]] = {}
        unread: list[str] = []
        if self.format == "nc":
            names = sorted(p.name for p in self.out.glob("cm1out*.nc")
                           if p.is_file() and p.name != _pdata.NETCDF_FILENAME)
            if not names:
                return held, unread
            try:
                import netCDF4
                import numpy as np
            except ImportError:
                return held, names
            for name in names:
                try:
                    with netCDF4.Dataset(self.out / name, "r") as ds:
                        values = ds.variables["time"][:] if "time" in ds.variables else []
                        times = [float(t) for t in np.ma.filled(np.ma.asarray(values, dtype=np.float64),
                                                                np.nan).ravel()]
                        keys = str(ds.getncattr("gm1_frames")).split() if "gm1_frames" in ds.ncattrs() else []
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    unread.append(name)
                    continue
                for t in times:
                    if not math.isfinite(t):
                        continue
                    row = held.setdefault(repr(t), {"frame": None, "time": t, "file": name})
                    if len(times) == 1 and len(keys) == 1 and not row["frame"]:
                        row["frame"] = keys[0]
            return held, unread
        fixed = {_grads.STATS_DAT, _grads.METADATA_DAT, _pdata.GRADS_DAT}
        data = sorted(p.name for p in self.out.glob("cm1out*.dat") if p.is_file() and p.name not in fixed)
        if not data:
            return held, unread
        try:
            notes = json.loads((self.out / NOTES_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            notes = None
        frames = notes.get("frames") if isinstance(notes, dict) else None
        if not isinstance(frames, list):
            return held, data
        for key in frames:
            if isinstance(key, str) and key:
                held[f"frame:{key}"] = {"frame": key, "time": None, "file": ""}
        return held, unread

    def _held_lost(self, held: Mapping[str, Any]) -> list[str]:
        """The times of ``held`` (:meth:`_held`) whose saved frame is gone though the run's records do not put them
        on a branch it has left.  A time no file names a frame for is lost when no frame of the run is at it; a
        frame named without its time (GrADS), when its folder is gone."""

        latest = self._latest()
        keys = {key for key, _path in latest.values()}
        times = {round(float(t), 6) for t in latest}
        left = set(self._left({name: row for name, row in held.items() if row.get("frame")}, latest))
        lost = []
        for name, row in held.items():
            if name in left or (row.get("frame") and row["frame"] in keys):
                continue
            if row.get("time") is None:
                if row.get("frame") and not (Path(self.layout.root) / str(row["frame"])).is_dir():
                    lost.append(name)
            elif round(float(row["time"]), 6) not in times:
                lost.append(name)
        return lost

    def _pending(self) -> list[tuple[float, str, Path]]:
        """Published frames not yet converted, by model time (a later segment wins)."""

        latest = self._latest()
        self._drop_left_branches(latest)
        self.summary.frames_total = len(latest)
        pending = []
        done = self.state["frames"]
        for time_s, (key, path) in sorted(latest.items()):
            row = done.get(repr(time_s))
            if row is not None and row.get("frame") == key:
                continue
            pending.append((time_s, key, path))
        return pending

    def _drop_left_branches(self, latest: Mapping[float, tuple[str, Path]]) -> None:
        """Remove what was converted from frames of a branch the run has left.

        In a ``gpuwm-cm1 run`` directory a resume from an earlier restart, or
        a start over, leaves the old branch's frames behind; its converted
        times would otherwise sit beside the new branch's.  Only the run's
        records say so (:meth:`_left`): a converted time whose frame was only
        deleted keeps its file.  One file per time (output_filetype 2 or 3)
        loses just the left branch's files; a single cm1out.nc
        (output_filetype 1) cannot drop a time, so it is written again, unless
        it also holds times whose frames are gone, which could not be written
        again: then nothing is changed and the pass stops and says why.
        """

        stale = self._left(self.state["frames"], latest)
        if not stale:
            return
        if self.filetype == 1:
            lost = self._lost(self.state["frames"], latest)
            if lost:
                raise Cm1outError(
                    f"The run has left the branch {len(stale)} converted time(s) came from, and cm1out.nc, which "
                    f"holds every time in one file, would be written again without them; but {len(lost)} of its "
                    "times have lost their saved frames and could not be written again. Nothing was changed; "
                    "cm1out.nc is kept. Convert into a new folder with --out, or with output_filetype 2.")
            self.log(f"the run has left the branch {len(stale)} converted time(s) came from; "
                     "writing cm1out.nc again")
            self._remove_previous()
            self.state = self._fresh_state()
            _save_state(self.state_path, self.state)
            return
        for time_key in stale:
            row = self.state["frames"].pop(time_key)
            self.state["stats"].pop(time_key, None)
            names = _row_files(row)
            for name in names:
                self._unlink_named(name)
            self.log(f"t = {float(row.get('time', 0.0)):g} s: removed {' '.join(names) or 'its file'}, "
                     "from a branch the run has left")
        if self.stats_enabled:
            if self.state["stats"]:
                self._write_stats()
            else:
                names = (_grads.STATS_CTL, _grads.STATS_DAT) if self.format == "grads" else (_stats.STATS_FILENAME,)
                for name in names:
                    (self.out / name).unlink(missing_ok=True)
        if self.format == "grads":
            self._write_grads_descriptors()
        _save_state(self.state_path, self.state)

    def _interval(self) -> float:
        return self.settings.float("tapfrq", 0.0)

    def run_pass(self, should_stop: Callable[[], bool] | None = None) -> int:
        """Convert every pending frame; returns how many were converted.

        ``should_stop`` is asked between frames, so a stop request (a sidecar
        told to end) never leaves a frame half written.
        """

        pending = self._pending()
        if not pending:
            if self.pdata_enabled:
                self._update_pdata()
            return 0
        interval = self._interval()
        converted = 0
        # A time a resumed segment repeats is already known: it keeps its place.
        known_times = {float(t) for t in self.state["frames"]}
        for time_s, key, path in pending:
            if should_stop is not None and should_stop():
                break
            started = _time.perf_counter()
            frame = _frames.open_frame(path)
            try:
                geometry = _settings.geometry(frame)
                species = _settings.species(frame)
                if self.geometry is None:
                    self.geometry, self.species = geometry, species
                elif (geometry.nx, geometry.ny, geometry.nz) != (self.geometry.nx, self.geometry.ny, self.geometry.nz):
                    raise Cm1outError(f"{path} is on a {geometry.nx}x{geometry.ny}x{geometry.nz} grid, "
                                      f"not the run's {self.geometry.nx}x{self.geometry.ny}x{self.geometry.nz}")
                ordinal = len([t for t in known_times if t < time_s])
                previous = self.state["frames"].get(repr(time_s))
                if previous is not None and "nwrite" in previous:
                    nwrite = int(previous["nwrite"])
                else:
                    taken = {int(r["nwrite"]): float(t) for t, r in self.state["frames"].items() if "nwrite" in r}
                    nwrite = assign_nwrite(time_s, interval, ordinal, taken)
                plan = _fields.plan(frame, self.settings, geometry, species, reflectivity=self.reflectivity)
                if self.format == "grads":
                    written = self._write_grads(frame, plan, nwrite, geometry, ordinal=ordinal)
                    name = written[0] if written else ""
                else:
                    name = self._write(frame, plan, nwrite, geometry, species, ordinal=ordinal)
                    written = [name]
                wrote = _time.perf_counter()
                row: dict[str, Any] = {"frame": key, "time": time_s, "nwrite": nwrite, "file": name,
                                       "arena_sha256": frame.arena_sha256}
                if self.format == "grads":
                    row["files"] = written
                    row["meta"] = _grads.metadata_record(self.settings, frame, geometry, nwrite)
                if self.stats_enabled:
                    result = _stats.compute(_stats.inputs_from_frame(frame, species), self.settings, geometry, species)
                    self.state["stats"][repr(time_s)] = result.values
                    self._merge_specs(result, geometry, species)
                finished = _time.perf_counter()
                row["seconds"] = {"write": round(wrote - started, 4), "stats": round(finished - wrote, 4)}
                self.state["frames"][repr(time_s)] = row
                self.state["variables"] = sorted(set(self.state["variables"]) | set(plan.names))
                self.state["not_produced"].update(plan.not_produced)
                self.state.setdefault("formed", {}).update(
                    {r.variable.name: r.how for r in plan.variables if r.how not in ("copied", "renamed")})
                self.state.setdefault("switches_not_produced", {}).update(plan.switches_not_produced)
                self.summary.seconds.append({"time": time_s, **row["seconds"], "cells": geometry.nx * geometry.ny * geometry.nz})
                self.log(f"t = {time_s:g} s: {name} ({finished - started:.2f} s)")
            finally:
                frame.close()
            known_times.add(time_s)
            converted += 1
            if self.stats_enabled:
                self._write_stats()
            if self.format == "grads":
                self._write_grads_descriptors()
            _save_state(self.state_path, self.state)
        self.summary.frames_converted += converted
        if self.pdata_enabled:
            self._update_pdata()
        return converted

    # ---------------------------------------------------------------- GrADS
    def _write_grads(self, frame: Any, plan: Any, nwrite: int, geometry: Any, *, ordinal: int) -> list[str]:
        """One frame's GrADS records (:class:`.grads.GradsHistory`); returns the data files written."""

        if not self.state.get("grads_variables"):
            self.state["grads_variables"] = [
                _grads.GradsVariable(r.variable.name, r.variable.long_name, r.variable.units,
                                     r.variable.grid).row()
                for r in plan.variables
            ]
            self.state["grads_geometry"] = _geometry_record(geometry)
        history = _grads.GradsHistory(
            self.out, self.settings, geometry, filetype=self.filetype,
            variables=_grads.variables_from_rows(self.state["grads_variables"]),
        )
        written = history.write_time(frame, plan, index=ordinal, nwrite=nwrite)
        self.state.setdefault("grads_notes", {}).update(history.notes)
        for name in written:
            if name not in self.summary.files:
                self.summary.files.append(name)
        return written

    def _grads_geometry(self) -> Any:
        record = self.state.get("grads_geometry")
        return None if not record else _geometry_from_record(record)

    def _write_grads_descriptors(self) -> None:
        """The history descriptors (tdef counts the times), the metadata file and the notes."""

        geometry = self._grads_geometry()
        rows = sorted(self.state["frames"].values(), key=lambda row: float(row["time"]))
        if geometry is None or not rows:
            return
        variables = _grads.variables_from_rows(self.state["grads_variables"])
        history = _grads.GradsHistory(self.out, self.settings, geometry, filetype=self.filetype,
                                      variables=variables)
        times = len(rows) if self.filetype == 1 else max(int(row["nwrite"]) for row in rows)
        history.write_descriptors(times)
        names = _grads.metadata_names(self.settings)
        _grads.write_metadata(self.out, names, [row["meta"] for row in rows])
        self._write_notes()

    def _write_notes(self) -> None:
        """``cm1out_gm1.json``: what the netCDF files would say in their gm1_* attributes."""

        record = {
            "schema": NOTES_SCHEMA,
            "producer": _producer(),
            "note": ("Simulated by GM1 (a GPU port of CM1 r21.1), not by CM1. Names, units, descriptor "
                     "grammar and record layout follow CM1 r21.1 writeout.F; the values are GM1's."),
            "settings": f"{self.settings.source} {self.settings.source_path or ''}".strip(),
            "format": "grads", "output_filetype": self.filetype,
            "variables": [row[0] for row in self.state["grads_variables"]],
            "not_produced": dict(sorted(self.state["not_produced"].items())),
            "switches_not_produced": dict(sorted((self.state.get("switches_not_produced") or {}).items())),
            "formed": dict(sorted((self.state.get("formed") or {}).items())),
            "grads_notes": dict(sorted((self.state.get("grads_notes") or {}).items())),
            "differs_from_cm1_descriptors": (
                "The title names the GM1 port; tdef counts the times GM1 wrote; variables GM1 does not "
                "produce are left out of the variable list and the records"
                + ("; with output_filetype = 3 dset names the files GM1's one process wrote "
                   "(cm1out_000000_00%y4_<g>.dat)." if self.filetype == 3 else ".")
            ),
            "frames": [row["frame"] for row in sorted(self.state["frames"].values(),
                                                      key=lambda row: float(row["time"]))],
        }
        pdata = self.state.get("pdata")
        if pdata:
            record["pdata"] = {"files": pdata.get("files"), "notes": pdata.get("notes")}
        _grads.write_text_atomic(self.out / NOTES_FILENAME, json.dumps(record, indent=1, sort_keys=True) + "\n")

    # ---------------------------------------------------------------- parcels
    def _update_pdata(self) -> None:
        """Write CM1's parcel file when the run's parcel track is new or has changed."""

        try:
            merged = _pdata.merge(_pdata.lineage_tracks(self.layout))
        except _pdata.PdataError as error:
            self.log(f"parcels: {error}")
            return
        previous = self.state.get("pdata") or None
        if merged is None or not merged.rows:
            if previous and not self._track_left():
                # The track was deleted (it is kept beside the saved frames): the parcel file is all that is
                # left of it, and is kept.
                return
            if previous:
                for name in previous.get("files") or []:
                    self._unlink_named(name)
                self.state["pdata"] = None
                _save_state(self.state_path, self.state)
            return
        signature = merged.signature()
        files = previous.get("files") if previous else None
        if (previous and previous.get("signature") == signature and previous.get("format") == self.format
                and files and all((self.out / name).is_file() for name in files)):
            return
        if previous and files and previous.get("signature") != signature and self._lost(self.state["frames"]):
            # Some saved frames, and the track parts beside them, are gone: written again, the parcel file could
            # hold less than it does.
            self.log("parcels: some of the run's saved frames were deleted, so its parcel file is kept as it is")
            return
        if self.format == "grads":
            files = _pdata.write_grads(self.out, merged)
        else:
            _pdata.write_netcdf(self.out / _pdata.NETCDF_FILENAME, merged, producer=_producer(),
                                deflate=self.deflate)
            files = [_pdata.NETCDF_FILENAME]
        if previous:
            for name in previous.get("files") or []:
                if name not in files:
                    self._unlink_named(name)
        self.state["pdata"] = {
            "signature": signature, "format": self.format, "files": files,
            "parcels": merged.track.nparcels, "attributes": list(merged.track.names),
            "times": merged.times, "notes": merged.notes,
        }
        self.log(f"parcels: {' '.join(files)} ({merged.track.nparcels} parcels, {len(merged.rows)} times)")
        if self.format == "grads":
            self._write_notes()
        _save_state(self.state_path, self.state)

    def _track_left(self) -> bool:
        """Whether a parcel track is still on disk in any of the run's segments (one on a branch it has left)."""

        return any((Path(segment.frames_root) / _pdata.TRACK_DIRECTORY).is_dir() for segment in self.layout.segments)

    def _merge_specs(self, result: Any, geometry: Any, species: Any) -> None:
        # Every statistic any frame produced, in setup_stat_vars' order.
        produced = {row[0] for row in self.state["stat_specs"]} | {s.name for s in result.specs}
        self.state["stat_specs"] = [
            [spec.name, spec.long_name, spec.units]
            for spec, _group in _stats._names(self.settings, geometry, species) if spec.name in produced
        ]
        self.state["stats_not_produced"].update(result.not_produced)
        for name in result.values:
            self.state["stats_not_produced"].pop(name, None)
        self.state["stats_derived"] = sorted(set(self.state["stats_derived"]) | set(result.derived))

    def _write(self, frame: Any, plan: Any, nwrite: int, geometry: Any, species: Any, *, ordinal: int) -> str:
        name = _netcdf.filename(self.filetype, nwrite)
        fornberg = _stats.fornberg_ends(geometry)
        moving = frame.moving_frame
        umove = float(moving.get("umove", self.settings.float("umove")) or 0.0)
        vmove = float(moving.get("vmove", self.settings.float("vmove")) or 0.0)
        root = self.layout.root
        if self.filetype == 1:
            path = self.out / name
            existing = path.exists()
            frames_written = [frame]
            handle = _netcdf.HistoryFile(
                path, self.settings, geometry, species, plan, deflate=self.deflate, append=existing,
                fornberg=fornberg,
                provenance=_netcdf.provenance(producer=_producer(), settings=self.settings, frame=frame,
                                              plan=plan, run_root=root, frames=frames_written),
            )
            try:
                # CM1's time_index counts the writes into cm1out.nc; a run that
                # did not publish t = 0 starts at the first time it did publish.
                handle.write_time(ordinal, frame, plan, umove=umove, vmove=vmove)
                if existing:
                    listed = handle.get_attribute("gm1_frames", "")
                    key = _netcdf._relative(frame.directory, root)
                    if key not in listed.split():
                        handle.set_attribute("gm1_frames", (listed + " " + key).strip())
                    not_produced = dict(json.loads(handle.get_attribute("gm1_not_produced_reasons", "{}")))
                    not_produced.update(plan.not_produced)
                    handle.set_attribute("gm1_not_produced", " ".join(sorted(not_produced)) or "none")
                    handle.set_attribute("gm1_not_produced_reasons", json.dumps(not_produced, sort_keys=True))
            finally:
                handle.close()
        else:
            path = self.out / name
            partial = self.out / (name + ".partial")
            handle = _netcdf.HistoryFile(
                partial, self.settings, geometry, species, plan, deflate=self.deflate, fornberg=fornberg,
                provenance=_netcdf.provenance(producer=_producer(), settings=self.settings, frame=frame,
                                              plan=plan, run_root=root, frames=[frame]),
            )
            try:
                handle.write_time(0, frame, plan, umove=umove, vmove=vmove)
            finally:
                handle.close()
            os.replace(partial, path)
        if name not in self.summary.files:
            self.summary.files.append(name)
        return name

    def _write_stats(self) -> None:
        rows = sorted(((float(t), values) for t, values in self.state["stats"].items()), key=lambda r: r[0])
        specs = [_stats.StatSpec(*row) for row in self.state["stat_specs"]]
        if not rows or not specs:
            return
        if self.format == "grads":
            # statpack.F:615-640: one record of every statistic per time, and its descriptor.
            _grads.write_stats(self.out, rows, specs)
            self.summary.stats_file = _grads.STATS_CTL
            return
        path = self.out / _stats.STATS_FILENAME
        partial = self.out / (_stats.STATS_FILENAME + ".partial")
        not_produced = dict(sorted(self.state["stats_not_produced"].items()))
        attributes = {
            "CM1 version": _netcdf.CM1_VERSION,
            "Conventions": "CF-1.7",
            "missing_value": __import__("numpy").float32(_netcdf.MISSING_VALUE),
            "gm1_producer": _producer(),
            "gm1_note": (
                "Statistics computed by GM1 from its history frames (one record per frame, not per "
                "statfrq), with CM1 r21.1 statpack.F's formulas, names and units."
            ),
            "gm1_not_produced": " ".join(not_produced) or "none",
            "gm1_not_produced_reasons": json.dumps(not_produced, sort_keys=True),
            "gm1_derived": " ".join(self.state["stats_derived"]) or "none",
            "gm1_derived_note": (
                "These statistics use the total pressure recovered from prs as (prs/p00)**(rd/cp), "
                "because the frames carry no base state; they agree with CM1's own to a few parts "
                "in a million, except ppmax/ppmin, which subtract two pressures of about 1e5 Pa "
                "and agree to about 0.1 Pa. Set output_basestate = 1 for CM1's own words."
            ),
        }
        _stats.write_stats(partial, rows, specs, global_attributes=attributes, deflate=self.deflate)
        os.replace(partial, path)
        self.summary.stats_file = _stats.STATS_FILENAME

    def finish(self) -> Summary:
        s = self.summary
        s.variables = list(self.state["variables"])
        s.not_produced = dict(sorted(self.state["not_produced"].items()))
        s.switches_not_produced = dict(sorted((self.state.get("switches_not_produced") or {}).items()))
        s.statistics = [row[0] for row in self.state["stat_specs"]]
        s.statistics_not_produced = dict(sorted(self.state["stats_not_produced"].items()))
        s.statistics_derived = list(self.state["stats_derived"])
        stats_name = _grads.STATS_CTL if self.format == "grads" else _stats.STATS_FILENAME
        if self.stats_enabled and (self.out / stats_name).is_file():
            s.stats_file = stats_name
        s.files = sorted({name for row in self.state["frames"].values() for name in _row_files(row)})
        if self.format == "grads" and s.files:
            s.files = sorted(set(s.files) | {_grads.ctl_filename(g) for g in _grads.variables_by_file(
                _grads.variables_from_rows(self.state["grads_variables"]))} | {_grads.METADATA_CTL, _grads.METADATA_DAT})
            s.notes = dict(sorted((self.state.get("grads_notes") or {}).items()))
        pdata = self.state.get("pdata") or {}
        s.pdata_files = list(pdata.get("files") or [])
        if pdata:
            s.pdata = {"parcels": pdata.get("parcels"), "times": len(pdata.get("times") or []),
                       "notes": pdata.get("notes") or {}}
        s.ended, end = self.layout.ended()
        s.end_reason = None if end is None else end.get("reason")
        return s


def _row_files(row: Mapping[str, Any]) -> list[str]:
    """Every file a converted time wrote: its netCDF file, or its GrADS data files."""

    names = [str(name) for name in (row.get("files") or []) if name]
    if not names and row.get("file"):
        names = [str(row["file"])]
    return names


def _geometry_record(geometry: Any) -> dict[str, Any]:
    import numpy as np

    return {"nx": geometry.nx, "ny": geometry.ny, "nz": geometry.nz, "dx": geometry.dx, "dy": geometry.dy,
            "dz": geometry.dz, "dt": geometry.dt,
            "zf_words": [f"{int(w):08x}" for w in np.asarray(geometry.zf, dtype=np.float32).view(np.uint32)]}


def _geometry_from_record(record: Mapping[str, Any]) -> Any:
    import numpy as np

    zf = np.array([int(w, 16) for w in record["zf_words"]], dtype=np.uint32).view(np.float32)
    dz = np.float32(record["dz"])
    zh = (np.float32(0.5) * (zf[1:] + zf[:-1])).astype(np.float32)
    mh = (dz / (zf[1:] - zf[:-1])).astype(np.float32)
    return _settings.Geometry(int(record["nx"]), int(record["ny"]), int(record["nz"]), float(record["dx"]),
                              float(record["dy"]), float(record["dz"]), zf, zh, mh,
                              (np.float32(1.0) / mh).astype(np.float32), float(record.get("dt") or 0.0))


def run_output_choices(target: Path | str) -> dict[str, Any]:
    """The ``[output]`` table of a run directory's ``settings.toml`` (empty when there is none).

    The importer writes it (the namelist's output_format and output_filetype,
    and ``also_netcdf = false``); the Settings screen changes it.
    """

    import tomllib

    path = Path(target) / "settings.toml"
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    table = document.get("output")
    return dict(table) if isinstance(table, dict) else {}


def convert(target: Path | str, *, follow: bool = False, poll_s: float = 2.0,
            should_stop: Callable[[], bool] | None = None, also_netcdf: bool | None = None,
            **options: Any) -> Summary:
    """Convert a run's frames; with ``follow``, keep going until the run ends.

    When the output is GrADS and ``also_netcdf`` is on (default: the run's
    settings.toml ``[output] also_netcdf``), netCDF is written beside it, in
    the same folder and the same pass.  A follower holds the run folder's
    cm1out lock (:mod:`.follow_lock`): a second one waits for the first.
    """

    if follow:
        from .follow_lock import FollowerStopped, follow_lock

        try:
            with follow_lock(target, "cm1out", should_stop=should_stop, log=options.get("log")):
                return _convert(target, follow=True, poll_s=poll_s, should_stop=should_stop,
                                also_netcdf=also_netcdf, **options)
        except FollowerStopped as error:
            raise Cm1outError(str(error)) from error
    return _convert(target, follow=False, poll_s=poll_s, should_stop=should_stop, also_netcdf=also_netcdf,
                    **options)


def _convert(target: Path | str, *, follow: bool, poll_s: float, should_stop: Callable[[], bool] | None,
             also_netcdf: bool | None, **options: Any) -> Summary:
    converter = Converter(target, wait_for_history=follow, **options)
    converters = [converter]
    if also_netcdf is None:
        also_netcdf = run_output_choices(target).get("also_netcdf") is True
    if also_netcdf and converter.format == "grads":
        extra = {key: value for key, value in options.items() if key != "fmt"}
        extra["out"] = converter.out
        extra["filetype"] = converter.filetype
        converters.append(Converter(target, wait_for_history=follow, fmt="nc", **extra))
    should_stop = should_stop or (lambda: False)
    while True:
        converted = 0
        for each in converters:
            if follow:
                # Segments appear as a run is resumed, and a sidecar may start
                # before its run has published anything.
                each.refresh()
            converted += each.run_pass(should_stop)
        if not follow or should_stop():
            break
        ended, _record = converter.layout.ended()
        if ended and converted == 0 and not any(each._pending() for each in converters):
            break
        if converted == 0:
            deadline = _time.monotonic() + max(0.1, poll_s)
            while _time.monotonic() < deadline and not should_stop():
                _time.sleep(min(0.2, poll_s))
    summary = converter.finish()
    if len(converters) > 1:
        summary.also = converters[1].finish()
    return summary


__all__ = ["Cm1outError", "Converter", "FORMATS", "NOTES_FILENAME", "STATE_FILENAME", "STATE_FILENAMES",
           "Summary", "convert", "converted_lost", "run_output_choices"]
