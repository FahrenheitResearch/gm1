"""``gpuwm-cm1 quicklook``: draw a run's quick-looks, frame by frame, as the frames land.

One pass draws every published frame of the run that has not been drawn yet, oldest first, and rewrites
``index.json`` and ``series.json``; ``follow`` repeats the pass until the run's event log holds its ``end`` record
and nothing is left, exactly as ``gpuwm-cm1 cm1out --follow`` does, so the two sidecars start and end together.

Output, under ``RUNDIR/quicklook/``:

``<product>/tSSSSSS.png``        1920 x 1080 pictures, one per product and frame (seconds since the start)
``<product>/thumb/tSSSSSS.png``  480 x 270 copies for the screens' strips
``index.json``                   ``gm1.quicklook-index/v1``: the run's facts, every product's frames with the
                                 numbers each showed, what could not be drawn and why
``series.json``, ``timeseries.png``  the time series (:mod:`.series`)
``slices/``                      pictures asked for one at a time (:mod:`.slices`, :mod:`.skewt`)
``environment/``, ``parcels/``   the looks drawn once per run: the environment's skew-T and hodograph (from the
                                 earliest frame), and the air parcels (when the run's parcel track is written)
``films/``                       films made on request (:mod:`.film`)
``.state/``                      what has been drawn from which frame, and the running swath and track maxima

A frame is drawn once (its key and arena digest are recorded).  A later segment's frame at a time already drawn
replaces it, and when a run leaves a branch (a resume from an earlier restart) the pictures of the times it left
are removed, as cm1out removes their files.  Only the run's own records say that a time was left: a picture whose
saved frame is merely gone (deleted by hand, or by the GUI's Keep pictures only) is kept, and a run with no saved
frames left keeps every picture it has (:func:`has_saved_frames`).  While any drawn time has lost its frame
(:func:`lost_times`), ``--force`` and the radar refuse the run, and a look drawn once per run from its saved times
that it has is kept as it is rather than drawn again from fewer of them.

The pass's record (``.state/state.json``) is not the only word on what was drawn.  When it is missing or cannot be
read, it is rebuilt from ``index.json`` (and the time series' numbers from ``series.json``), and a picture of a
saved time that neither holds and whose frame is gone counts as a drawn time too (:func:`pictures_on_disk`), so a
pass never takes such a run for one with nothing to keep.

``TARGET`` may also be CM1 netCDF history files (:mod:`.ncread`): one file, or a folder of them.  Their pictures go
to ``quicklook/`` in that folder (``NAME-quicklook/`` beside a single file), each saved time is a frame, and a look
whose fields the files do not hold is not drawn and says which field and which CM1 switch writes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import shutil
import time as _time
from typing import Any, Callable, Mapping

from ..cm1out import frames as _frames
from . import products as P
from . import series as SE
from . import style as S
from . import timeheight as TH
from .facts import read_settings, run_facts
from .frame import FieldMissing, FrameUnreadable, QuickFrame, frames_of, same_frame
from .ncread import discover
from .swath import SwathBuilder
from .tracks import TRACK_IDS, TrackBuilder, product_track
from .worker import worker_environment, worker_facts

INDEX_SCHEMA = "gm1.quicklook-index/v1"
STATE_SCHEMA = "gm1.quicklook-state/v1"
SUMMARY_SCHEMA = "gm1.quicklook-summary/v1"
STATE_DIR = ".state"
STATE_FILE = "state.json"
PRODUCT_IDS = tuple(pid for pid, _ in P.PRODUCTS)
#: The radar looks (:mod:`.radar`), drawn again for every frame when the run's radar settings change.
RADAR_IDS = ("radar-storm-relative", "radar-velocity")
#: Looks drawn once per run, not once per frame: (id, title).
ONCE: tuple[tuple[str, str], ...] = (("environment", "The storm's environment"),
                                     ("parcels", "Air rising into the storm"),
                                     ("run-card", "How this storm was simulated")) + TH.ONCE
ONCE_IDS = tuple(pid for pid, _ in ONCE)
#: Looks drawn only when asked for (``gpuwm-cm1 quicklook RUNDIR --look storm3d``), never by the worker as frames
#: land: (id, title).  Each keeps its pictures' list in ``<id>/drawn.json`` (:mod:`.storm3d`), not in the state.
ON_REQUEST: tuple[tuple[str, str], ...] = (("storm3d", "The storm in 3-D"),)
ON_REQUEST_IDS = tuple(pid for pid, _ in ON_REQUEST)
LOOK_IDS = PRODUCT_IDS + ONCE_IDS + ON_REQUEST_IDS


class QuicklookError(ValueError):
    """A quick-look pass that cannot proceed, in plain words."""


#: Said by a pass over a run whose saved frames were all deleted: its pictures are all that is left of it.
FRAMES_GONE = ("This run's saved frames were deleted, so no new look can be drawn for it: the looks are drawn from "
               "the saved frames. Its pictures are kept as they are.")
#: ``--force`` on such a run: it would only delete them.
FRAMES_GONE_FORCE = ("This run's saved frames were deleted, so its pictures cannot be drawn again: --force would "
                     "only delete them. Nothing was deleted; its pictures are kept.")


#: Said by a pass whose record of what was drawn was missing or unreadable, or lacked pictures that are there.
REBUILT = ("This run's quick-look record was missing or could not be read, or did not list some of its pictures: "
           "it was rebuilt from the picture index and the pictures themselves, and every picture is kept.")


def _drawn_looks(row: Mapping[str, Any]) -> set[str]:
    """The looks a drawn time has a record of, but for a picture marked to be drawn again (new radar settings)."""

    return {pid for pid, pic in (row.get("products") or {}).items()
            if not (isinstance(pic, Mapping) and pic.get("stale"))}


def frames_lost_force(lost: int, drawn: int, left: int) -> str:
    """``--force`` on a run some of whose drawn saved times have lost their saved frames: it would delete their
    pictures, which cannot be drawn again."""

    return (f"{lost} of this run's {drawn} drawn saved times have lost their saved frames, so their pictures cannot "
            "be drawn again: --force would delete them. Nothing was deleted; its pictures are kept. Without --force, "
            f"a pass draws only the looks its pictures lack, from the {left} saved frame"
            f"{'' if left == 1 else 's'} left.")


def frames_lost_note(lost: int, drawn: int) -> str:
    """Said by a pass over such a run: what it keeps."""

    return (f"{lost} of this run's {drawn} drawn saved times have lost their saved frames: their pictures are kept, "
            "and so are the looks drawn once per run from every saved time that it has.")


def radar_lost(lost: int, drawn: int) -> str:
    """The radar of every saved time, asked for a run some of whose saved frames are gone."""

    return (f"{lost} of this run's {drawn} drawn saved times have lost their saved frames, so the radar cannot scan "
            "those times again and their radar pictures would no longer match the others. Nothing was changed; its "
            "pictures are kept.")


def has_saved_frames(layout: Any) -> bool:
    """Whether any saved frame of the run is still on disk, on its lineage or not (CM1 netCDF files: any saved
    time).  A run whose frames were all deleted, by hand or by the GUI's Keep pictures only, has none."""

    own = getattr(layout, "lineage", None)
    if own is not None:
        return bool(own())
    return any(_frames.published_frames(segment.frames_root) for segment in layout.segments)


def _line_keys(lineage: list[tuple[float, str, Path, int]]) -> tuple[set[str], set[tuple[int, str]], set[str]]:
    keys = {key for _t, key, _p, _s in lineage}
    return keys, {(segment, Path(key).name) for _t, key, _p, segment in lineage}, {Path(key).name for key in keys}


def _on_line(row: Mapping[str, Any], keys: set[str], named: set[tuple[int, str]], bare: set[str]) -> bool:
    """Whether a drawn time's frame is one of the run's frames now: by its key, or by the frame's folder alone, as a
    run named by a relative path recorded it (:func:`.frame.same_frame`)."""

    frame = row.get("frame")
    if not isinstance(frame, str) or not frame:
        return False
    if frame in keys:
        return True
    if "/" in frame:
        return False
    segment = row.get("segment")
    if isinstance(segment, int) and not isinstance(segment, bool):
        return (segment, frame) in named
    return frame in bare


def _time_of(name: str, row: Mapping[str, Any]) -> float | None:
    try:
        t = float(row["time"] if row.get("time") is not None else name)
    except (TypeError, ValueError):
        return None
    return t if math.isfinite(t) else None


def left_behind(layout: Any, lineage: list[tuple[float, str, Path, int]],
                rows: Mapping[str, Any]) -> list[str]:
    """The keys of ``rows`` (each a drawn time: its frame's key, segment and time) that belong to a branch the run
    has left, a resume from an earlier restart or a start over, as the run's own records say it: a ``gpuwm-cm1 run``
    directory whose run.json records its newest segment (:meth:`.RunDirectory.recorded_line`), and that still has
    saved frames.  A time whose frame is merely gone (deleted by hand, or by the GUI's Keep pictures only) is not
    left behind, nor is one of a segment whose whole folder was deleted: its pictures are all that is left of it."""

    run = layout.run_directory()
    if run is None or not layout.lineage_settled() or not has_saved_frames(layout):
        return []
    from ..cm1run.rundir import off_the_line

    line = run.recorded_line()
    keys, named, bare = _line_keys(lineage)
    return [name for name, row in rows.items()
            if isinstance(row, Mapping) and not _on_line(row, keys, named, bare) and off_the_line(line, row, name)]


def lost_times(layout: Any, lineage: list[tuple[float, str, Path, int]], rows: Mapping[str, Any]) -> list[str]:
    """The keys of ``rows`` (drawn times, as for :func:`left_behind`) whose saved frame is gone though the run never
    left them: deleted by hand, or by the GUI's Keep pictures only.  No frame of the run is at their time (a later
    segment's frame at that time would draw them again), and the run's records do not put them on a branch it has
    left.  Their pictures are all there is of them: nothing may delete them, and a look drawn once per run from
    every saved time is not drawn again over one drawn from more.  A time known only from its pictures (its record
    was lost) names no frame or segment, so it is never taken to be on a branch the run left."""

    keys, named, bare = _line_keys(lineage)
    times = {round(float(t), 6) for t, *_ in lineage}
    left = set(left_behind(layout, lineage, rows))
    lost = []
    for name, row in rows.items():
        if not isinstance(row, Mapping) or name in left or (not row.get("frame") and not row.get("recovered")):
            continue
        if _on_line(row, keys, named, bare):
            continue
        t = _time_of(name, row)
        if t is not None and round(t, 6) not in times:
            lost.append(name)
    return lost


#: A per-time picture's file name (:func:`_tag`): ``tSSSSSS.png``, or ``tSSSSSSSS.SSS`` with p for the point.
_PICTURE = re.compile(r"^t(\d+)(?:p(\d+))?\.png$")


def tag_time(name: str) -> float | None:
    """The saved time a picture's file name says (:func:`_tag`), or None."""

    found = _PICTURE.match(name)
    if found is None:
        return None
    whole, part = found.groups()
    return float(f"{int(whole)}.{part}") if part else float(int(whole))


def pictures_on_disk(out: Path, looks: tuple[str, ...] = PRODUCT_IDS) -> dict[float, dict[str, dict[str, Any]]]:
    """The pictures drawn for each saved time that are in ``out``, by the time each file's name says:
    ``{time: {look: {"file", "thumb"}}}``.  They are there whatever the pass's records say of them."""

    found: dict[float, dict[str, dict[str, Any]]] = {}
    for pid in looks:
        try:
            entries = list(os.scandir(Path(out) / pid))
        except OSError:
            continue
        for entry in entries:
            t = tag_time(entry.name)
            if t is not None and entry.is_file():
                found.setdefault(t, {})[pid] = {"file": f"{pid}/{entry.name}", "thumb": f"{pid}/thumb/{entry.name}"}
    return found


def _once_on_disk(out: Path) -> dict[str, dict[str, Any]]:
    """The looks drawn once per run whose picture is in ``out``: ``{look: {"file", "thumb", "time"}}`` (the time
    when the file's name says it)."""

    found: dict[str, dict[str, Any]] = {}
    for pid in ONCE_IDS:
        try:
            names = sorted(entry.name for entry in os.scandir(Path(out) / pid)
                           if entry.is_file() and entry.name.endswith(".png"))
        except OSError:
            continue
        if names:
            found[pid] = {"file": f"{pid}/{names[0]}", "thumb": f"{pid}/thumb/{names[0]}", "time": tag_time(names[0])}
    return found


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def record_from_index(out: Path) -> dict[str, Any] | None:
    """The pass's record rebuilt from ``out``'s ``index.json`` (and the time series' numbers from ``series.json``),
    for a record that is missing or cannot be read: each drawn time's frame, segment and pictures with the numbers
    they showed, the looks drawn once per run, and the run's facts as the index gave them.  None when there is no
    index to read."""

    try:
        index = json.loads((Path(out) / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        return None
    run = index.get("run") if isinstance(index.get("run"), dict) else {}
    facts = {key: value for key, value in run.items() if key not in ("grid", "lowest_level_m")}
    rows: dict[str, dict[str, Any]] = {}

    def row_at(t: float, frame: Any = None, segment: Any = None) -> dict[str, Any]:
        row = rows.setdefault(repr(float(t)), {"frame": None, "segment": None, "time": float(t), "products": {},
                                               "facts": dict(facts), "recovered": "index"})
        if isinstance(frame, str) and frame and not row["frame"]:
            row["frame"] = frame
        if isinstance(segment, int) and not isinstance(segment, bool) and row["segment"] is None:
            row["segment"] = segment
        return row

    for entry in index.get("frames") or []:
        if isinstance(entry, dict) and _number(entry.get("time_s")) is not None:
            row_at(float(entry["time_s"]), entry.get("frame"), entry.get("segment"))
    once: dict[str, dict[str, Any]] = {}
    for product in index.get("products") or []:
        if not isinstance(product, dict) or not isinstance(product.get("id"), str) or product.get("on_request"):
            continue
        pid = product["id"]
        drawn = [f for f in product.get("frames") or []
                 if isinstance(f, dict) and isinstance(f.get("file"), str) and _number(f.get("time_s")) is not None]
        reasons = [r for r in product.get("not_drawn") or [] if isinstance(r, dict) and r.get("reason")]
        if product.get("once"):
            if pid not in ONCE_IDS:
                continue
            if drawn:
                first = drawn[0]
                once[pid] = {"time": float(first["time_s"]), "file": first["file"],
                             "thumb": first.get("thumb") or f"{pid}/thumb/{Path(first['file']).name}",
                             "facts": first.get("facts") or {}, "recovered": "index"}
            elif reasons:
                once[pid] = {"not_drawn": str(reasons[0]["reason"]), "recovered": "index"}
            continue
        if pid not in PRODUCT_IDS:
            continue
        for entry in drawn:
            row_at(float(entry["time_s"]), entry.get("frame"), entry.get("segment"))["products"][pid] = {
                "file": entry["file"], "thumb": entry.get("thumb") or f"{pid}/thumb/{Path(entry['file']).name}",
                "facts": entry.get("facts") or {}}
        for reason in reasons:
            for t in reason.get("times_s") or []:
                if _number(t) is not None:
                    row_at(float(t))["products"].setdefault(pid, {"not_drawn": str(reason["reason"])})
    try:
        series = json.loads((Path(out) / "series.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        series = None
    if isinstance(series, dict) and isinstance(series.get("charts"), list):
        for row in rows.values():
            row["stats"] = {}       # the numbers each time showed, from the charts (none for a time they lack)
    for chart in (series.get("charts") or []) if isinstance(series, dict) else []:
        key = chart.get("key") if isinstance(chart, dict) else None
        for point in (chart.get("gm1") or []) if isinstance(key, str) else []:
            if isinstance(point, list) and len(point) == 2 and _number(point[0]) is not None \
                    and _number(point[1]) is not None and repr(float(point[0])) in rows:
                rows[repr(float(point[0]))].setdefault("stats", {})[key] = float(point[1])
    return {"frames": rows, "once": once, "run": run}


def drawn_rows(out: Path) -> dict[str, Any]:
    """The saved times the pictures in ``out`` were drawn from, keyed as the state keys them (its frame's key,
    segment and time): the pass's state, else its index (a state that was deleted), and any time a picture there
    shows that neither holds (named by its time alone)."""

    try:
        state = json.loads((Path(out) / STATE_DIR / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = None
    if isinstance(state, dict) and state.get("schema") == STATE_SCHEMA and isinstance(state.get("frames"), dict):
        rows = dict(state["frames"])
    else:
        rebuilt = record_from_index(Path(out))
        rows = dict(rebuilt["frames"]) if rebuilt else {}
    held = {round(t, 6) for name, row in rows.items() if isinstance(row, Mapping)
            for t in [_time_of(name, row)] if t is not None}
    for t, pics in sorted(pictures_on_disk(Path(out)).items()):
        if round(t, 6) not in held:
            rows[repr(t)] = {"frame": None, "segment": None, "time": t, "products": pics, "recovered": "pictures"}
    return rows


def _tag(t: float) -> str:
    if abs(t - round(t)) < 1e-6:
        return f"t{int(round(t)):06d}"
    return f"t{t:012.3f}".replace(".", "p")


def _utc() -> str:
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _finite(doc: Any) -> Any:
    """``doc`` with every number that is not finite (NaN, inf) as null: the index and the series are read by the
    GUI as strict JSON, which has no NaN."""

    if isinstance(doc, float):
        return doc if math.isfinite(doc) else None
    if isinstance(doc, dict):
        return {key: _finite(value) for key, value in doc.items()}
    if isinstance(doc, (list, tuple)):
        return [_finite(value) for value in doc]
    return doc


def _requested(out: Path, pid: str) -> list[dict[str, Any]]:
    """The pictures of a look drawn on request, oldest first, from ``<pid>/drawn.json``."""

    try:
        doc = json.loads((out / pid / "drawn.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = (doc.get("stills") or {}).values() if isinstance(doc, dict) and isinstance(doc.get("stills"), dict) else ()
    rows = [r for r in rows if isinstance(r, dict) and isinstance(r.get("time"), (int, float))
            and isinstance(r.get("file"), str)]
    return sorted(rows, key=lambda r: float(r["time"]))


def default_out(layout: _frames.RunLayout) -> Path:
    if layout.kind == "netcdf":
        return layout.default_out
    return (layout.root.parent if layout.kind == "frames" else layout.root) / "quicklook"


def why_not(qf: QuickFrame, look: str, reason: str) -> str:
    """The reason ``look`` gives for not being drawn: as the look said it, or, from a CM1 netCDF file, the field it
    lacks in plain words (:meth:`.ncread.NcFrame.why_not`)."""

    own = getattr(qf.frame, "why_not", None)
    return reason if own is None else own(look, reason)


def records_root(layout: _frames.RunLayout) -> Path:
    """The folder holding the run's own records (import-report.json, receipt.json, deck.toml)."""

    own = getattr(layout, "records_root", None)
    if own is not None:
        return Path(own)
    return layout.root if layout.kind != "frames" else layout.root.parent


@dataclass
class Summary:
    out: Path
    frames_total: int = 0
    frames_drawn: int = 0
    seconds: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: Plain lines about the whole pass (the run's settings could not be read).
    notes: list[str] = field(default_factory=list)
    ended: bool = False
    end_reason: str | None = None
    worker: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {"schema": SUMMARY_SCHEMA, "out": os.fspath(self.out), "frames_total": self.frames_total,
                "frames_drawn_now": self.frames_drawn, "seconds_per_frame": self.seconds,
                "problems": self.problems, "notes": self.notes, "run_ended": self.ended,
                "end_reason": self.end_reason,
                "index": os.fspath(self.out / "index.json"), "worker": self.worker}


class Renderer:
    """One run's quick-looks, pass by pass."""

    def __init__(self, target: Path | str, *, out: Path | str | None = None, only: tuple[str, ...] | None = None,
                 force: bool = False, label: str | None = None, log: Callable[[str], None] | None = None,
                 wait_for_history: bool = False, namelist: Path | str | None = None,
                 deck: Path | str | None = None) -> None:
        self.target = Path(target)
        self.namelist = None if namelist is None else Path(namelist)
        self.deck = None if deck is None else Path(deck)
        try:
            self.layout = discover(self.target)
        except _frames.FrameError:
            if not (wait_for_history and self.target.is_dir()):
                raise
            self.layout = _frames.RunLayout("rundir", self.target, ())
        if self.layout.kind == "netcdf" and wait_for_history:
            raise QuicklookError("CM1 netCDF files are drawn as they are: --follow is for a GM1 run that is still "
                                 "going. Leave out --follow.")
        unknown = [p for p in (only or ()) if p not in LOOK_IDS]
        if unknown:
            raise QuicklookError(f"no such quick-look: {', '.join(unknown)} (they are {', '.join(LOOK_IDS)})")
        self.products = tuple(p for p in only if p in PRODUCT_IDS) if only else PRODUCT_IDS
        self.once = tuple(p for p in only if p in ONCE_IDS) if only else ONCE_IDS
        self.out = Path(out) if out is not None else default_out(self.layout)
        self.label = label
        self.log = log or (lambda _message: None)
        self.state_dir = self.out / STATE_DIR
        #: Set when the pass's record was missing or unreadable and was rebuilt (:meth:`_load_state`).
        self.rebuilt: str | None = None
        self.state = self._load_state()
        if force:
            drawn = self.state["frames"]
            if drawn and not has_saved_frames(self.layout):
                # nothing could be drawn again: --force would only empty the run of its pictures
                raise QuicklookError(FRAMES_GONE_FORCE)
            lineage = frames_of(self.layout)
            lost = lost_times(self.layout, lineage, drawn)
            if lost:
                # the pictures of those times could not be drawn again: --force would only delete them
                raise QuicklookError(frames_lost_force(len(lost), len(drawn), len(lineage)))
            self._remove_previous()
            self.state = self._load_state()
        self.swath = SwathBuilder(self.state_dir)
        self.tracks = {pid: TrackBuilder(self.state_dir, pid) for pid in TRACK_IDS}
        if self.layout.kind == "netcdf":
            # every saved time is there: the swath and the tracks cover the ground the domain passes over from
            # their first picture on, so a film of them holds still
            cells = self.layout.ground_cells()
            for builder in (self.swath, *self.tracks.values()):
                builder.reserved = cells
        self.summary = Summary(self.out)
        self._settings = None
        #: Set when the run's settings could not be read: the one line said about it (``facts.read_settings``).
        self.settings_note: str | None = None
        #: The looks the last frame drawn had drawn in this pass, and those left out as hurricane runs' own.
        self.drawn_now: list[str] = []
        self.skipped_now: list[str] = []
        self._motion: tuple[tuple[float, float], str] | bool | None = None
        self._storm_track: Any = None

    # -- state ------------------------------------------------------------------
    def _load_state(self) -> dict[str, Any]:
        """The pass's record.  When it is missing or cannot be read, it is rebuilt from the index (and the series'
        numbers), so the times drawn before are still known, and from the pictures themselves
        (:meth:`_take_in_pictures`).  A picture of a saved time the record does not hold, whose frame is gone, is
        taken in as a drawn time whatever the record."""

        try:
            state = json.loads((self.state_dir / STATE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        self.rebuilt = None
        whole = not (isinstance(state, dict) and state.get("schema") == STATE_SCHEMA
                     and isinstance(state.get("frames"), dict))
        if whole:
            state = {"schema": STATE_SCHEMA, "frames": {}}
            rebuilt = record_from_index(self.out)
            if rebuilt is not None and (rebuilt["frames"] or rebuilt["once"]):
                state["frames"], state["once"] = rebuilt["frames"], rebuilt["once"]
                state["recovered_run"] = rebuilt["run"]
                self.rebuilt = "index"
        state.setdefault("once", {})
        if self._take_in_pictures(state, whole=whole) and self.rebuilt is None:
            self.rebuilt = "pictures"
        return state

    def _take_in_pictures(self, state: dict[str, Any], *, whole: bool) -> bool:
        """Pictures in the out folder that the record does not hold, taken into it.  Those of saved times whose
        frame is gone are all that is left of those times: they count as drawn (:func:`lost_times`) and are listed.
        With ``whole`` (the record was missing or unreadable), those of times whose frame is there are taken in as
        drawn from it too (a picture is drawn over at its time's file name, so the one there is the newest), and so
        are the looks drawn once per run: none is drawn again over from less (a look that follows the storm, or adds
        up the times before it, would miss the times whose frames are gone).  Returns whether any was."""

        on_disk = pictures_on_disk(self.out)
        once = _once_on_disk(self.out) if whole else {}
        if not on_disk and not once:
            return False
        lineage = {round(float(t), 6): (key, segment) for t, key, _path, segment in frames_of(self.layout)}
        rows = state["frames"]
        held = {round(t, 6): name for name, row in rows.items() if isinstance(row, Mapping)
                for t in [_time_of(name, row)] if t is not None}
        newest = sorted(((_time_of(n, r) or 0.0, r) for n, r in rows.items() if isinstance(r, Mapping)
                         and r.get("facts")), key=lambda pair: pair[0])
        facts = dict(newest[-1][1]["facts"]) if newest else dict(state.get("recovered_run") or {})
        facts = {key: value for key, value in facts.items() if key not in ("grid", "lowest_level_m")}
        taken = False
        for t, pics in sorted(on_disk.items()):
            frame = lineage.get(round(t, 6))
            if frame is not None and not whole:
                continue          # its frame is there: a pass draws what it lacks from it
            name = held.get(round(t, 6))
            if name is None:
                rows[repr(t)] = {"frame": frame[0] if frame else None, "segment": frame[1] if frame else None,
                                 "time": t, "facts": dict(facts),
                                 "products": {pid: {**pic, "facts": {}} for pid, pic in pics.items()},
                                 "recovered": "pictures"}
                taken = True
                continue
            products = rows[name].setdefault("products", {})
            for pid, pic in pics.items():
                if pid not in products:
                    products[pid] = {**pic, "facts": {}}
                    taken = True
        latest = max((_time_of(n, r) or 0.0 for n, r in rows.items() if isinstance(r, Mapping)), default=0.0)
        for pid, pic in once.items():
            if pid not in state["once"]:
                state["once"][pid] = {"file": pic["file"], "thumb": pic["thumb"], "facts": {},
                                      "time": pic["time"] if pic["time"] is not None else latest,
                                      "recovered": "pictures"}
                taken = True
        return taken

    def _save_state(self) -> None:
        _write_json(self.state_dir / STATE_FILE, self.state)

    def _remove_previous(self) -> None:
        # The looks drawn on request (the storm in 3-D) are kept: the pass never draws them again.
        for pid in PRODUCT_IDS + ONCE_IDS:
            shutil.rmtree(self.out / pid, ignore_errors=True)
        shutil.rmtree(self.state_dir, ignore_errors=True)
        for name in ("index.json", "series.json", "timeseries.png"):
            (self.out / name).unlink(missing_ok=True)

    def _remove_time(self, time_key: str) -> None:
        row = self.state["frames"].pop(time_key, None) or {}
        for pid, pic in (row.get("products") or {}).items():
            for name in (pic.get("file"), pic.get("thumb")):
                if name and ".." not in Path(name).parts:
                    (self.out / name).unlink(missing_ok=True)

    # -- the run ----------------------------------------------------------------
    def refresh(self) -> None:
        if self.layout.kind == "netcdf":
            return
        try:
            self.layout = _frames.discover(self.target)
        except _frames.FrameError:
            pass

    @property
    def settings(self) -> Any:
        if self._settings is None:
            from ..cm1out import settings as _settings

            if self.namelist is not None or self.deck is not None:
                # a file the user named must be read, or the pass stops and says why
                self._settings = _settings.resolve(self.layout, namelist=self.namelist, deck=self.deck)
                if self.layout.kind == "netcdf":
                    self.layout.use_settings(self._settings)
                return self._settings
            if self.layout.kind == "netcdf":
                found = self.layout.settings
                for note in self.layout.notes:
                    self.log(note)
                self._settings = found if found is not None else False
                return self._settings or None
            settings, note = read_settings(self.layout)
            self._settings = settings if settings is not None else False
            if note is not None:
                # One plain line: without the settings a hurricane is drawn as a thunderstorm run.
                self.settings_note = note
                self.summary.notes.append(note)
                self.log(note)
        return self._settings or None

    def _open(self, path: Path, key: str, segment: int) -> QuickFrame:
        return QuickFrame.open(path, key=key, segment=segment)

    def radar_extras(self) -> dict[str, Any]:
        """What the radar looks need from the run: its radar settings (``radar.json`` beside the pictures, read for
        each frame, so a change reaches the next picture) and the storm motion of its starting air
        (``run_motion``: the Bunkers right mover the run recorded, or its 0-6 km mean wind where the shear is too
        weak for a right mover; else the same worked out from the first saved time's corner air)."""

        from . import radar as RA

        extras: dict[str, Any] = {"radar_settings": RA.read_settings(self.out)}
        if self._motion is None:
            self._motion = False
            from . import environment as EN

            root = self.layout.root if self.layout.kind != "frames" else self.layout.root.parent
            env, _source = EN.parameters(Path(root), [Path(s.frames_root).parent for s in self.layout.segments])
            found = RA.recorded_motion(env)
            if found is None:
                # A run with no record of its own: the same on the first saved time's undisturbed air (the
                # environment look's corner columns, winds ground-relative).
                lineage = frames_of(self.layout)
                if lineage:
                    t, key, path, segment = lineage[0]
                    try:
                        qf = self._open(path, key, segment)
                    except (FrameUnreadable, OSError, ValueError, KeyError):
                        qf = None
                    if qf is not None:
                        try:
                            prof = EN.profile(qf)
                            found = RA.air_motion(prof["z"], prof["u"], prof["v"])
                        except (FrameUnreadable, ValueError, KeyError):
                            found = None
                        finally:
                            qf.close()
            if found is not None:
                self._motion = found
        if self._motion:
            extras["run_motion"] = self._motion
        return extras

    @property
    def storm_track(self) -> Any:
        """Where the radar looks put the storm at each saved time (:class:`.radar.StormTrack`)."""

        if self._storm_track is None:
            from . import radar as RA

            self._storm_track = RA.StormTrack(self.state_dir)
        return self._storm_track

    def _track_point(self, row: tuple[float, str, Path, int], track: Any) -> Any:
        """The storm's centre at a saved time the track has not kept yet (the radar looks follow the storm)."""

        from . import radar as RA

        t, key, path, segment = row
        try:
            qf = self._open(path, key, segment)
        except (FrameUnreadable, OSError, ValueError, KeyError):
            return RA.TrackPoint(float(t), 0.0, 0.0, False)
        try:
            facts = run_facts(self.layout, qf.config, qf.dx, qf.dy, label=self.label, settings=self.settings)
            looks = P.Looks(qf, facts)
            return RA.track_point(looks, RA.find_storm(looks, track))
        except (FrameUnreadable, ValueError, KeyError):
            return RA.TrackPoint(float(t), 0.0, 0.0, False)
        finally:
            qf.close()

    def _radar_storm(self, looks: P.Looks, before: list[tuple[float, str, Path, int]], key: str) -> None:
        """Find the storm of this saved time from where the times before put it, for the radar looks, and keep it."""

        from . import radar as RA

        track = self.storm_track.before(before, self._track_point)
        storm = RA.find_storm(looks, track)
        looks.extras["radar_track"] = track
        looks.extras["radar_storm"] = storm
        self.storm_track.put(len(before), key, RA.track_point(looks, storm))

    def forget(self, products: tuple[str, ...]) -> int:
        """Mark ``products`` as to be drawn again, for every frame, so the next pass draws them again (new radar
        settings); returns how many pictures will be drawn again.  Each drawn picture's record is kept, marked stale,
        until its time is drawn again: a pass stopped part way, or a time whose frame is gone, keeps the picture it
        has listed."""

        count = 0
        for row in self.state["frames"].values():
            products_now = row.get("products") or {}
            for pid in products:
                pic = products_now.get(pid)
                if pic is None:
                    continue
                if isinstance(pic, dict) and "file" in pic:
                    pic["stale"] = True
                else:
                    products_now.pop(pid)
                count += 1
        self._save_state()
        return count

    def pending(self) -> tuple[list[tuple[float, str, Path, int]], list[tuple[float, str, Path, int]]]:
        """(the run's frames oldest first, the ones not drawn yet); drops the times of a branch the run left
        (:func:`left_behind`: never a time whose saved frame was only deleted)."""

        lineage = frames_of(self.layout)
        self.summary.frames_total = len(lineage)
        if self.rebuilt is not None and REBUILT not in self.summary.notes:
            self.summary.notes.append(REBUILT)
            self.log(REBUILT)
        for time_key in left_behind(self.layout, lineage, self.state["frames"]):
            self.log(f"t = {float(time_key):g} s: removed its pictures, from a branch the run has left")
            self._remove_time(time_key)
        if not lineage and self.state["frames"] and FRAMES_GONE not in self.summary.notes \
                and not has_saved_frames(self.layout):
            self.summary.notes.append(FRAMES_GONE)
            self.log(FRAMES_GONE)
        elif lineage and self.state["frames"]:
            lost = lost_times(self.layout, lineage, self.state["frames"])
            note = frames_lost_note(len(lost), len(self.state["frames"])) if lost else None
            if note is not None and note not in self.summary.notes:
                self.summary.notes.append(note)
                self.log(note)
        done = self.state["frames"]
        todo = []
        for row in lineage:
            t, key, segment = row[0], row[1], row[3]
            prev = done.get(repr(t))
            # drawn from this frame (named by its key, or by its folder alone in an older record), with every look
            # this pass draws and the numbers of the time series
            if prev is not None and same_frame(prev.get("frame"), key, prev.get("segment"), segment) \
                    and set(self.products) <= _drawn_looks(prev) and "stats" in prev:
                continue
            todo.append(row)
        return lineage, todo

    def _interval(self, lineage: list[tuple[float, str, Path, int]]) -> float | None:
        times = [t for t, *_ in lineage]
        gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
        return gaps[len(gaps) // 2] if gaps else None

    def draw(self, row: tuple[float, str, Path, int], lineage: list[tuple[float, str, Path, int]]) -> dict[str, Any]:
        t, key, path, segment = row
        started = _time.perf_counter()
        qf = self._open(path, key, segment)
        try:
            facts = run_facts(self.layout, qf.config, qf.dx, qf.dy, label=self.label, settings=self.settings)
            looks = P.Looks(qf, facts)
            before = [r for r in lineage if r[0] < t]
            # the earlier saved times, for the looks that follow the storm (quicklook.nearground's storm track)
            looks.history = {"before": before, "open": self._open, "state_dir": self.state_dir, "layout": self.layout}
            record: dict[str, Any] = {"frame": key, "segment": segment, "time": t,
                                      "arena_sha256": qf.frame.arena_sha256, "products": {},
                                      "cells": qf.cells, "grid": [qf.nx, qf.ny, qf.nz], "dx_m": qf.dx,
                                      "z0_m": float(qf.zh[0])}
            previous = (self.state["frames"].get(repr(t)) or {})
            same = same_frame(previous.get("frame"), key, previous.get("segment"), segment)
            if same:
                record["products"] = dict(previous.get("products") or {})

            def drawn(pid: str) -> bool:
                pic = record["products"].get(pid)
                return isinstance(pic, dict) and "file" in pic and not pic.get("stale") \
                    and (self.out / pic["file"]).is_file()

            if any(pid in RADAR_IDS and not drawn(pid) for pid in self.products):
                looks.extras.update(self.radar_extras())
                try:
                    self._radar_storm(looks, before, key)
                except (FrameUnreadable, ValueError, KeyError):
                    pass      # the radar looks find the storm themselves and say what they could not read
            tag = _tag(t)
            self.drawn_now, self.skipped_now = [], []
            for pid in self.products:
                if drawn(pid):
                    continue      # drawn before from this very frame (a pass that adds looks draws only those)
                if pid in P.HURRICANE_ONLY and not looks.hurricane:
                    # A hurricane run's own look: left out of every other run (and of its index), not "not drawn".
                    record["products"][pid] = {"skipped": "a look for hurricane runs"}
                    self.skipped_now.append(pid)
                    continue
                file = self.out / pid / f"{tag}.png"
                begun = _time.perf_counter()
                try:
                    if looks.hurricane and pid in P.THUNDERSTORM_ONLY:
                        raise P.NotDrawn(P.THUNDERSTORM_ONLY_WHY)
                    if pid == "swath":
                        self.swath.catch_up(before, self._open)
                        self.swath.add(qf)
                        shown = P.product_swath(looks, file, self.swath.swath(self._interval(lineage)))
                    elif pid in self.tracks:
                        builder = self.tracks[pid]
                        builder.catch_up(before, self._open)
                        builder.add(qf)
                        shown = product_track(looks, file, builder.track(self._interval(lineage)), pid)
                    else:
                        shown = P.RENDERERS[pid](looks, file)
                    record["products"][pid] = {
                        "file": f"{pid}/{tag}.png", "thumb": f"{pid}/thumb/{tag}.png", "facts": shown,
                        "seconds": round(_time.perf_counter() - begun, 3)}
                    self.drawn_now.append(pid)
                except P.NotDrawn as why:
                    record["products"][pid] = {"not_drawn": why_not(qf, pid, str(why))}
                except FieldMissing as why:
                    # a CM1 file without a field this look needs: said in the index, not a fault
                    record["products"][pid] = {"not_drawn": why_not(qf, pid, str(why))}
                except (FrameUnreadable, P.LintError, ValueError, KeyError) as error:
                    plain = why_not(qf, pid, "")
                    if plain:
                        # a CM1 file without a field this look draws from: said in plain words, not a fault
                        record["products"][pid] = {"not_drawn": plain}
                    else:
                        record["products"][pid] = {"error": f"{type(error).__name__}: {error}"}
                        self.summary.problems.append(f"t = {t:g} s {pid}: {error}")
                        self.log(f"t = {t:g} s: {pid} was not drawn: {error}")
            # A pass that adds looks to a frame drawn before keeps that frame's numbers this pass did not work out
            # (the highest reflectivity comes only with a look that draws it), and puts this pass's on top.
            kept = dict(previous.get("stats") or {}) if same else {}
            try:
                record["stats"] = {**kept, **looks.statistics()}
            except (FrameUnreadable, ValueError, KeyError) as error:
                record["stats"] = kept
                self.summary.problems.append(f"t = {t:g} s statistics: {error}")
            record["facts"] = facts.as_dict()
        finally:
            qf.close()
        record["seconds"] = round(_time.perf_counter() - started, 3)
        return record

    def run_pass(self, should_stop: Callable[[], bool] | None = None) -> int:
        lineage, todo = self.pending()
        if self.rebuilt is not None and lineage:
            # the rebuilt record is kept at once, so what it holds is known to the next pass however this one ends
            self._save_state()
        drawn = 0
        for row in todo:
            if should_stop is not None and should_stop():
                break
            record = self.draw(row, lineage)
            self.state["frames"][repr(row[0])] = record
            self.summary.seconds.append({"time": row[0], "seconds": record["seconds"], "cells": record["cells"],
                                         "products": {pid: p.get("seconds") for pid, p in record["products"].items()}})
            self.swath.save()
            for builder in self.tracks.values():
                builder.save()
            if self._storm_track is not None:
                self._storm_track.save()
            self._save_state()
            self.write_index(lineage)
            if self.drawn_now or not self.skipped_now:
                self.log(f"t = {row[0]:g} s: {len(self.drawn_now)} quick-look(s) in {record['seconds']:.1f} s")
            else:
                self.log(f"t = {row[0]:g} s: nothing drawn: {', '.join(self.skipped_now)} "
                         f"{'is a look' if len(self.skipped_now) == 1 else 'are looks'} for hurricane runs")
            drawn += 1
        if lineage and self.draw_once(lineage):
            self._save_state()
            self.write_index(lineage)
        if todo == [] and not (self.out / "index.json").is_file() and lineage:
            self.write_index(lineage)
        self.summary.frames_drawn += drawn
        return drawn

    # -- the looks drawn once per run ----------------------------------------------
    def draw_once(self, lineage: list[tuple[float, str, Path, int]]) -> bool:
        """The environment (from the earliest frame) and the parcels (once the run's track is written), each
        drawn again only when what it is drawn from changes.  Returns whether anything was drawn."""

        changed = False
        once = self.state["once"]
        # A run some of whose drawn saved times have lost their saved frames keeps each of these looks that it has
        # as it is: drawn again, they would show fewer saved times (the storm over time, the run card), a later
        # time than the first (the environment), or a parcel track that went with the frames.  The ones it lacks
        # are drawn from the frames that are left.
        lost = bool(lineage) and bool(lost_times(self.layout, lineage, self.state["frames"]))
        keep = frozenset(pid for pid in ONCE_IDS if lost and once.get(pid) is not None)
        if "environment" in self.once and "environment" not in keep:
            t, key, path, segment = lineage[0]
            if (once.get("environment") or {}).get("frame") != key:
                once["environment"] = self._once_environment(t, key, path, segment)
                changed = True
        if "parcels" in self.once and "parcels" not in keep:
            signature = self._parcel_signature()
            if signature is not None and (once.get("parcels") or {}).get("signature") != signature:
                once["parcels"] = self._once_parcels(lineage, signature)
                changed = True
            elif signature is None and once.get("parcels") is not None and not (
                    "file" in once["parcels"] and not self._parcel_track_left()):
                # the run left the branch whose parcels these were; a picture of parcels whose track was only
                # deleted is all that is left of them, and is kept
                once.pop("parcels")
                changed = True
        if "run-card" in self.once and "run-card" not in keep:
            from . import runcard as RC

            changed = RC.draw_once(self, lineage) or changed
        # the storm over time (updraft and spin at every height), drawn again when the run's frames change
        changed = TH.draw_once(self, lineage, keep=keep, frames_lost=lost) or changed
        return changed

    def _parcel_track_left(self) -> bool:
        """Whether a parcel track is still on disk in any of the run's segments (one on a branch the run has left),
        rather than deleted."""

        from ..cm1out import pdata

        return any((Path(segment.frames_root) / pdata.TRACK_DIRECTORY).is_dir()
                   for segment in getattr(self.layout, "segments", ()))

    def redraw_refused(self, look: str, lineage: list[tuple[float, str, Path, int]]) -> str | None:
        """Why a look drawn once per run that has its picture is not drawn again now (``--look LOOK``), in plain
        words, or None: drawn again from what is left of the run, it would show less than it does."""

        row = (self.state.get("once") or {}).get(look) or {}
        if "file" not in row:
            return None          # nothing drawn: nothing to lose
        title = dict(ONCE).get(look, look)
        lost = lost_times(self.layout, lineage, self.state["frames"])
        if look == "parcels":
            signature = self._parcel_signature()
            if signature is None:
                if self._parcel_track_left():
                    return None      # the run left the branch whose parcels these were
                return ("The run's parcel track was deleted (it is kept beside the saved frames), so the air parcels "
                        "cannot be drawn again. Nothing was drawn; the picture it has is kept.")
            if signature == row.get("signature") or not lost:
                return None
            return (f"The run's parcel track has changed since its air parcels were drawn, and {len(lost)} of its "
                    f"{len(self.state['frames'])} drawn saved times have lost their saved frames (the track is kept "
                    "beside them): drawn again, the parcels could show less. Nothing was drawn; the picture it has "
                    "is kept.")
        if look == "environment":
            key = row.get("frame")
            if key:
                gone = not any(same_frame(key, k, None, s) for _t, k, _p, s in lineage) and not left_behind(
                    self.layout, lineage, {"environment": {"frame": key, "time": row.get("time")}})
            else:
                # known from the index or its picture alone (its record was lost): by its time
                t = _number(row.get("time"))
                gone = t is not None and round(t, 6) not in {round(float(s), 6) for s, *_ in lineage}
            if gone:
                return (f"{title} is drawn from the run's first saved frame, which was deleted: drawn again, it "
                        "would show a later time. Nothing was drawn; the picture it has is kept.")
            return None
        if not lost:
            return None
        drawn = len(self.state["frames"])
        return (f"{title} is drawn from the run's saved frames, and {len(lost)} of this run's {drawn} drawn saved "
                "times have lost their saved frames: drawn again, it would show fewer of them. Nothing was drawn; the "
                "picture it has is kept.")

    def _parcel_signature(self) -> str | None:
        from ..cm1out import pdata

        try:
            merged = pdata.merge(pdata.lineage_tracks(self.layout))
        except (pdata.PdataError, OSError, ValueError, KeyError):
            return None
        if merged is None:
            return None
        import hashlib

        return hashlib.sha256(json.dumps(merged.signature()).encode()).hexdigest()

    def _once_environment(self, t: float, key: str, path: Path, segment: int) -> dict[str, Any]:
        from . import environment as EN

        begun = _time.perf_counter()
        qf = self._open(path, key, segment)
        try:
            facts = run_facts(self.layout, qf.config, qf.dx, qf.dy, label=self.label, settings=self.settings)
            root = records_root(self.layout)
            segments = [Path(s.frames_root).parent for s in self.layout.segments]
            env, source = EN.parameters(Path(root), segments)
            tag = _tag(t)
            file = self.out / "environment" / f"{tag}.png"
            shown = EN.draw(qf, facts, file, env, source, varies=EN.air_varies(Path(root)))
            return {"frame": key, "time": t, "file": f"environment/{tag}.png", "thumb": f"environment/thumb/{tag}.png",
                    "facts": shown, "seconds": round(_time.perf_counter() - begun, 3)}
        except FieldMissing as why:
            return {"frame": key, "time": t, "not_drawn": why_not(qf, "environment", str(why))}
        except (FrameUnreadable, P.LintError, ValueError, KeyError) as error:
            plain = why_not(qf, "environment", "")
            if plain:
                return {"frame": key, "time": t, "not_drawn": plain}
            self.summary.problems.append(f"environment: {error}")
            return {"frame": key, "time": t, "error": f"{type(error).__name__}: {error}"}
        finally:
            qf.close()

    def _once_parcels(self, lineage: list[tuple[float, str, Path, int]], signature: str) -> dict[str, Any]:
        from . import parcels as PA

        begun = _time.perf_counter()
        t, key, path, segment = lineage[-1]
        qf = self._open(path, key, segment)
        try:
            facts = run_facts(self.layout, qf.config, qf.dx, qf.dy, label=self.label, settings=self.settings)
            extent, nxy = qf.extent, (qf.nx, qf.ny)
        finally:
            qf.close()
        file = self.out / "parcels" / "still.png"
        try:
            shown = PA.still(self.layout, facts, extent, nxy, file)
        except P.NotDrawn as why:
            return {"signature": signature, "not_drawn": str(why)}
        except (P.LintError, ValueError, KeyError) as error:
            self.summary.problems.append(f"parcels: {error}")
            return {"signature": signature, "error": f"{type(error).__name__}: {error}"}
        return {"signature": signature, "time": shown["time_s"], "file": "parcels/still.png",
                "thumb": "parcels/thumb/still.png", "facts": shown, "seconds": round(_time.perf_counter() - begun, 3)}

    # -- index and series ---------------------------------------------------------
    def write_index(self, lineage: list[tuple[float, str, Path, int]] | None = None) -> dict[str, Any]:
        rows = sorted(((float(k), v) for k, v in self.state["frames"].items()), key=lambda kv: kv[0])
        # The run's facts come from the newest time drawn from its frame; a time known only from the index or its
        # pictures (the record was lost) does not carry the grid, so the index's own words are kept for them.
        full = [row for _t, row in rows if row.get("grid")]
        last = full[-1] if full else None
        # A hurricane's index says its times as its pictures and pages do (hours and days); any other run's H:MM:SS.
        hurricane = bool(rows) and ((last or rows[-1][1]).get("facts") or {}).get("storm_kind") == "hurricane"
        clock = S.fmt_long_clock if hurricane else S.fmt_clock
        products = []
        for pid, title in P.PRODUCTS:
            frames, not_drawn = [], {}
            skipped = False
            for ordinal, (t, row) in enumerate(rows):
                pic = (row.get("products") or {}).get(pid)
                if not pic:
                    continue
                if "skipped" in pic:
                    skipped = True      # a look this kind of run does not have: it is not listed at all
                    continue
                if "file" in pic:
                    frames.append({"index": ordinal, "time_s": t, "clock": clock(t), "file": pic["file"],
                                   "thumb": pic["thumb"], "facts": pic.get("facts") or {},
                                   "segment": row.get("segment"), "frame": row.get("frame")})
                else:
                    not_drawn.setdefault(pic.get("not_drawn") or pic.get("error") or "not drawn", []).append(t)
            entry: dict[str, Any] = {"id": pid, "title": title, "frames": frames}
            if not_drawn:
                entry["not_drawn"] = [{"reason": reason, "times_s": times} for reason, times in not_drawn.items()]
            if frames or not_drawn or (pid in self.products and not skipped and pid not in P.HURRICANE_ONLY):
                products.append(entry)
        for pid, title in ONCE:
            row = (self.state.get("once") or {}).get(pid)
            if not row:
                continue
            entry = {"id": pid, "title": title, "once": True, "frames": []}
            if "file" in row:
                entry["frames"].append({"index": 0, "time_s": row["time"], "clock": clock(row["time"]),
                                        "file": row["file"], "thumb": row["thumb"], "facts": row.get("facts") or {}})
            else:
                entry["not_drawn"] = [{"reason": row.get("not_drawn") or row.get("error") or "not drawn",
                                       "times_s": []}]
            products.append(entry)
        for pid, title in ON_REQUEST:
            drawn = _requested(self.out, pid)
            if drawn:
                products.append({"id": pid, "title": title, "on_request": True, "frames": [
                    {"index": n, "time_s": float(row["time"]), "clock": clock(float(row["time"])), "file": row["file"],
                     "thumb": row.get("thumb"), "facts": row.get("facts") or {}, "frame": row.get("frame")}
                    for n, row in enumerate(drawn)]})
        run: dict[str, Any] = {}
        if last is not None:
            run = dict(last.get("facts") or {})
            nx, ny, nz = last.get("grid") or (0, 0, 0)
            run["grid"] = f"{nx} × {ny} × {nz}, dx {S.fmt_grid(float(last.get('dx_m') or 0))}"
            run["lowest_level_m"] = last.get("z0_m")
        elif rows:
            run = dict(self.state.get("recovered_run") or rows[-1][1].get("facts") or {})
        ended, end = self.layout.ended()
        self.summary.ended = ended
        self.summary.end_reason = None if end is None else end.get("reason")
        series = self.write_series(rows, run)
        index = {
            "schema": INDEX_SCHEMA,
            "updated_utc": _utc(),
            "run": run,
            "products": products,
            "series": series,
            # (a time known only from its pictures names no frame: it says so, for the screens' count of times lost)
            "frames": [{"index": i, "time_s": t, "clock": clock(t), "segment": row.get("segment"),
                        "frame": row.get("frame"),
                        **({"recovered": "pictures"} if not row.get("frame") and row.get("recovered") else {})}
                       for i, (t, row) in enumerate(rows)],
            "run_ended": ended,
            "end_reason": self.summary.end_reason,
            "worker": worker_facts(),
            "slices": "slices/",
        }
        if self.settings_note:
            index["notes"] = [self.settings_note]
        _write_json(self.out / "index.json", _finite(index))
        return index

    def _series_kept(self) -> dict[str, Any]:
        """The time series as it is on disk, for the index: its entry in the index there, else its files."""

        try:
            entry = json.loads((self.out / "index.json").read_text(encoding="utf-8")).get("series")
        except (OSError, ValueError, AttributeError):
            entry = None
        if isinstance(entry, dict) and entry.get("file"):
            return entry
        kept: dict[str, Any] = {}
        if (self.out / "series.json").is_file():
            kept["file"] = "series.json"
        if (self.out / "timeseries.png").is_file():
            kept["picture"] = "timeseries.png"
        return kept

    def write_series(self, rows: list[tuple[float, Mapping[str, Any]]], run: Mapping[str, Any]) -> dict[str, Any]:
        if any("stats" not in row for _t, row in rows) and (
                (self.out / "series.json").is_file() or (self.out / "timeseries.png").is_file()):
            # Some drawn times' numbers went with a lost record (they are drawn again with their frames, when they
            # are there): the series and its picture, which hold them, are kept as they are until then.
            return self._series_kept()
        stats = [(t, row.get("stats") or {}) for t, row in rows if row.get("stats")]
        if not stats:
            return {}
        last_facts = rows[-1][1].get("facts") or {}
        if last_facts.get("model", "GM1") != "GM1":
            # CM1's own output is not drawn beside CM1's reference curve (it would be compared with itself)
            reference, why = None, "this is CM1's own output"
        else:
            reference, why = SE.reference_for(self.settings, kind=str(run.get("storm_kind") or ""))
        timax = None
        if self.settings is not None:
            try:
                timax = float(self.settings.float("timax", 0.0)) or None
            except (TypeError, ValueError):
                timax = None
        z0 = float(run.get("lowest_level_m") or 0.0)
        name = str(run.get("name") or "this run")
        doc = SE.series_document(stats, z0=z0, name=name, reference=reference, t_end=timax,
                                 kind=str(run.get("storm_kind") or ""), model=str(last_facts.get("model") or "GM1"))
        if reference is None:
            doc["cm1_reference"] = why
        _write_json(self.out / "series.json", _finite(doc))
        out = {"file": "series.json", "cm1_reference": reference is not None}
        t_now = stats[-1][0]
        facts = rows[-1][1].get("facts") or {}
        footer = [S.data_line(S.SIMULATED, facts.get("made_by") or f"GM1 {facts.get('gm1_version', '')}",
                              ", ".join(str(w) for w in (facts.get("microphysics", ""), facts.get("checked", "")) if w),
                              f"dx {S.fmt_grid(float(facts.get('dx_m') or 0))}",
                              f"lowest level {S.fmt_height(z0)}", "wind ground-relative", SE.clock_words(doc, t_now))]
        if facts.get("mark_word"):
            footer = S.Footer(footer, str(facts["mark_word"]))
        try:
            SE.render_png(doc, self.out / "timeseries.png", t_now=t_now, footer=footer)
            out.update({"picture": "timeseries.png", "time_s": t_now})
        except (P.LintError, ValueError) as error:
            self.summary.problems.append(f"time series: {error}")
        return out


def render(target: Path | str, *, follow: bool = False, poll_s: float = 2.0,
           should_stop: Callable[[], bool] | None = None, **options: Any) -> Summary:
    """Draw a run's quick-looks; with ``follow``, keep going until the run ends.

    A follower holds the run folder's quicklook lock (``cm1out.follow_lock``):
    a second one, started with the next segment, waits for the first.
    """

    if follow:
        from ..cm1out.follow_lock import FollowerStopped, follow_lock

        try:
            with follow_lock(target, "quicklook", should_stop=should_stop, log=options.get("log")):
                return _render(target, follow=True, poll_s=poll_s, should_stop=should_stop, **options)
        except FollowerStopped as error:
            raise QuicklookError(str(error)) from error
    return _render(target, follow=False, poll_s=poll_s, should_stop=should_stop, **options)


def _render(target: Path | str, *, follow: bool, poll_s: float, should_stop: Callable[[], bool] | None,
            **options: Any) -> Summary:
    renderer = Renderer(target, wait_for_history=follow, **options)
    should_stop = should_stop or (lambda: False)
    while True:
        if follow:
            renderer.refresh()
        drawn = renderer.run_pass(should_stop)
        if not follow or should_stop():
            break
        ended, _record = renderer.layout.ended()
        if ended and drawn == 0 and not renderer.pending()[1]:
            break
        if drawn == 0:
            deadline = _time.monotonic() + max(0.1, poll_s)
            while _time.monotonic() < deadline and not should_stop():
                _time.sleep(min(0.2, poll_s))
    renderer.summary.worker = worker_facts()
    lineage, _todo = renderer.pending()
    if lineage and renderer.draw_once(lineage):
        renderer._save_state()
    if renderer.state["frames"] and (has_saved_frames(renderer.layout) or record_from_index(renderer.out) is None):
        # (a run whose saved frames were deleted keeps its index and pictures as they are; one whose index was lost
        # too gets it again, listing the pictures it has)
        renderer.write_index()
    return renderer.summary


__all__ = ["FRAMES_GONE", "FRAMES_GONE_FORCE", "INDEX_SCHEMA", "LOOK_IDS", "ONCE", "ONCE_IDS", "ON_REQUEST",
           "ON_REQUEST_IDS", "PRODUCT_IDS", "QuicklookError", "RADAR_IDS", "Renderer", "Summary", "default_out",
           "drawn_rows", "frames_lost_force", "frames_lost_note", "has_saved_frames", "left_behind", "lost_times",
           "radar_lost", "render", "worker_environment", "worker_facts"]
