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
are removed, as cm1out removes their files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import time as _time
from typing import Any, Callable, Mapping

from ..cm1out import frames as _frames
from . import products as P
from . import series as SE
from . import style as S
from .facts import run_facts
from .frame import FrameUnreadable, QuickFrame, frames_of
from .swath import SwathBuilder
from .tracks import TRACK_IDS, TrackBuilder, product_track
from .worker import worker_environment, worker_facts

INDEX_SCHEMA = "gm1.quicklook-index/v1"
STATE_SCHEMA = "gm1.quicklook-state/v1"
SUMMARY_SCHEMA = "gm1.quicklook-summary/v1"
STATE_DIR = ".state"
STATE_FILE = "state.json"
PRODUCT_IDS = tuple(pid for pid, _ in P.PRODUCTS)
#: Looks drawn once per run, not once per frame: (id, title).
ONCE: tuple[tuple[str, str], ...] = (("environment", "The storm's environment"),
                                     ("parcels", "Air rising into the storm"))
ONCE_IDS = tuple(pid for pid, _ in ONCE)
LOOK_IDS = PRODUCT_IDS + ONCE_IDS


class QuicklookError(ValueError):
    """A quick-look pass that cannot proceed, in plain words."""


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


def default_out(layout: _frames.RunLayout) -> Path:
    return (layout.root.parent if layout.kind == "frames" else layout.root) / "quicklook"


@dataclass
class Summary:
    out: Path
    frames_total: int = 0
    frames_drawn: int = 0
    seconds: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    ended: bool = False
    end_reason: str | None = None
    worker: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {"schema": SUMMARY_SCHEMA, "out": os.fspath(self.out), "frames_total": self.frames_total,
                "frames_drawn_now": self.frames_drawn, "seconds_per_frame": self.seconds,
                "problems": self.problems, "run_ended": self.ended, "end_reason": self.end_reason,
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
            self.layout = _frames.discover(self.target)
        except _frames.FrameError:
            if not (wait_for_history and self.target.is_dir()):
                raise
            self.layout = _frames.RunLayout("rundir", self.target, ())
        unknown = [p for p in (only or ()) if p not in LOOK_IDS]
        if unknown:
            raise QuicklookError(f"no such quick-look: {', '.join(unknown)} (they are {', '.join(LOOK_IDS)})")
        self.products = tuple(p for p in only if p in PRODUCT_IDS) if only else PRODUCT_IDS
        self.once = tuple(p for p in only if p in ONCE_IDS) if only else ONCE_IDS
        self.out = Path(out) if out is not None else default_out(self.layout)
        self.label = label
        self.log = log or (lambda _message: None)
        self.state_dir = self.out / STATE_DIR
        if force:
            self._remove_previous()
        self.state = self._load_state()
        self.swath = SwathBuilder(self.state_dir)
        self.tracks = {pid: TrackBuilder(self.state_dir, pid) for pid in TRACK_IDS}
        self.summary = Summary(self.out)
        self._settings = None

    # -- state ------------------------------------------------------------------
    def _load_state(self) -> dict[str, Any]:
        try:
            state = json.loads((self.state_dir / STATE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if state.get("schema") != STATE_SCHEMA:
            state = {"schema": STATE_SCHEMA, "frames": {}}
        state.setdefault("once", {})
        return state

    def _save_state(self) -> None:
        _write_json(self.state_dir / STATE_FILE, self.state)

    def _remove_previous(self) -> None:
        for pid in LOOK_IDS:
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
                return self._settings
            try:
                self._settings = _settings.resolve(self.layout)
            except (_settings.SettingsError, OSError, ValueError):
                self._settings = False
        return self._settings or None

    def _open(self, path: Path, key: str, segment: int) -> QuickFrame:
        return QuickFrame.open(path, key=key, segment=segment)

    def pending(self) -> tuple[list[tuple[float, str, Path, int]], list[tuple[float, str, Path, int]]]:
        """(the run's frames oldest first, the ones not drawn yet); drops the times of a branch the run left."""

        lineage = frames_of(self.layout)
        self.summary.frames_total = len(lineage)
        if self.layout.run_directory() is None or self.layout.lineage_settled():
            keys = {key for _t, key, _p, _s in lineage}
            for time_key in [k for k, row in self.state["frames"].items() if row.get("frame") not in keys]:
                self.log(f"t = {float(time_key):g} s: removed its pictures, from a branch the run has left")
                self._remove_time(time_key)
        done = self.state["frames"]
        todo = []
        for row in lineage:
            t, key = row[0], row[1]
            prev = done.get(repr(t))
            if prev is not None and prev.get("frame") == key and set(self.products) <= set(prev.get("products") or {}):
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
            record: dict[str, Any] = {"frame": key, "segment": segment, "time": t,
                                      "arena_sha256": qf.frame.arena_sha256, "products": {},
                                      "cells": qf.cells, "grid": [qf.nx, qf.ny, qf.nz], "dx_m": qf.dx,
                                      "z0_m": float(qf.zh[0])}
            previous = (self.state["frames"].get(repr(t)) or {})
            if previous.get("frame") == key:
                record["products"] = dict(previous.get("products") or {})
            tag = _tag(t)
            for pid in self.products:
                if pid in record["products"] and "file" in record["products"][pid] \
                        and (self.out / record["products"][pid]["file"]).is_file():
                    continue      # drawn before from this very frame (a pass that adds looks draws only those)
                file = self.out / pid / f"{tag}.png"
                begun = _time.perf_counter()
                try:
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
                except P.NotDrawn as why:
                    record["products"][pid] = {"not_drawn": str(why)}
                except (FrameUnreadable, P.LintError, ValueError, KeyError) as error:
                    record["products"][pid] = {"error": f"{type(error).__name__}: {error}"}
                    self.summary.problems.append(f"t = {t:g} s {pid}: {error}")
                    self.log(f"t = {t:g} s: {pid} was not drawn: {error}")
            try:
                record["stats"] = looks.statistics()
            except (FrameUnreadable, ValueError, KeyError) as error:
                record["stats"] = {}
                self.summary.problems.append(f"t = {t:g} s statistics: {error}")
            record["facts"] = facts.as_dict()
        finally:
            qf.close()
        record["seconds"] = round(_time.perf_counter() - started, 3)
        return record

    def run_pass(self, should_stop: Callable[[], bool] | None = None) -> int:
        lineage, todo = self.pending()
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
            self._save_state()
            self.write_index(lineage)
            self.log(f"t = {row[0]:g} s: {len(record['products'])} quick-look(s) in {record['seconds']:.1f} s")
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
        if "environment" in self.once:
            t, key, path, segment = lineage[0]
            if (once.get("environment") or {}).get("frame") != key:
                once["environment"] = self._once_environment(t, key, path, segment)
                changed = True
        if "parcels" in self.once:
            signature = self._parcel_signature()
            if signature is not None and (once.get("parcels") or {}).get("signature") != signature:
                once["parcels"] = self._once_parcels(lineage, signature)
                changed = True
            elif signature is None and once.get("parcels") is not None:
                once.pop("parcels")
                changed = True
        return changed

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
            root = self.layout.root if self.layout.kind != "frames" else self.layout.root.parent
            segments = [Path(s.frames_root).parent for s in self.layout.segments]
            env, source = EN.parameters(Path(root), segments)
            tag = _tag(t)
            file = self.out / "environment" / f"{tag}.png"
            shown = EN.draw(qf, facts, file, env, source, varies=EN.air_varies(Path(root)))
            return {"frame": key, "time": t, "file": f"environment/{tag}.png", "thumb": f"environment/thumb/{tag}.png",
                    "facts": shown, "seconds": round(_time.perf_counter() - begun, 3)}
        except (FrameUnreadable, P.LintError, ValueError, KeyError) as error:
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
        products = []
        for pid, title in P.PRODUCTS:
            frames, not_drawn = [], {}
            for ordinal, (t, row) in enumerate(rows):
                pic = (row.get("products") or {}).get(pid)
                if not pic:
                    continue
                if "file" in pic:
                    frames.append({"index": ordinal, "time_s": t, "clock": S.fmt_clock(t), "file": pic["file"],
                                   "thumb": pic["thumb"], "facts": pic.get("facts") or {},
                                   "segment": row.get("segment"), "frame": row.get("frame")})
                else:
                    not_drawn.setdefault(pic.get("not_drawn") or pic.get("error") or "not drawn", []).append(t)
            entry: dict[str, Any] = {"id": pid, "title": title, "frames": frames}
            if not_drawn:
                entry["not_drawn"] = [{"reason": reason, "times_s": times} for reason, times in not_drawn.items()]
            if pid in self.products or frames or not_drawn:
                products.append(entry)
        for pid, title in ONCE:
            row = (self.state.get("once") or {}).get(pid)
            if not row:
                continue
            entry = {"id": pid, "title": title, "once": True, "frames": []}
            if "file" in row:
                entry["frames"].append({"index": 0, "time_s": row["time"], "clock": S.fmt_clock(row["time"]),
                                        "file": row["file"], "thumb": row["thumb"], "facts": row.get("facts") or {}})
            else:
                entry["not_drawn"] = [{"reason": row.get("not_drawn") or row.get("error") or "not drawn",
                                       "times_s": []}]
            products.append(entry)
        run: dict[str, Any] = {}
        if rows:
            last = rows[-1][1]
            run = dict(last.get("facts") or {})
            nx, ny, nz = last.get("grid") or (0, 0, 0)
            run["grid"] = f"{nx} × {ny} × {nz}, dx {S.fmt_grid(float(last.get('dx_m') or 0))}"
            run["lowest_level_m"] = last.get("z0_m")
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
            "frames": [{"index": i, "time_s": t, "clock": S.fmt_clock(t), "segment": row.get("segment"),
                        "frame": row.get("frame")} for i, (t, row) in enumerate(rows)],
            "run_ended": ended,
            "end_reason": self.summary.end_reason,
            "worker": worker_facts(),
            "slices": "slices/",
        }
        _write_json(self.out / "index.json", index)
        return index

    def write_series(self, rows: list[tuple[float, Mapping[str, Any]]], run: Mapping[str, Any]) -> dict[str, Any]:
        stats = [(t, row.get("stats") or {}) for t, row in rows if row.get("stats")]
        if not stats:
            return {}
        reference, why = SE.reference_for(self.settings)
        timax = None
        if self.settings is not None:
            try:
                timax = float(self.settings.float("timax", 0.0)) or None
            except (TypeError, ValueError):
                timax = None
        z0 = float(run.get("lowest_level_m") or 0.0)
        name = str(run.get("name") or "this run")
        doc = SE.series_document(stats, z0=z0, name=name, reference=reference, t_end=timax)
        if reference is None:
            doc["cm1_reference"] = why
        _write_json(self.out / "series.json", doc)
        out = {"file": "series.json", "cm1_reference": reference is not None}
        t_now = stats[-1][0]
        facts = rows[-1][1].get("facts") or {}
        footer = [S.data_line(S.SIMULATED, f"GM1 {facts.get('gm1_version', '')}",
                              f"{facts.get('microphysics', '')}, {facts.get('checked', '')}",
                              f"dx {S.fmt_grid(float(facts.get('dx_m') or 0))}",
                              f"lowest level {S.fmt_height(z0)}", "wind ground-relative", S.fmt_clock(t_now))]
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
    if renderer.state["frames"]:
        renderer.write_index()
    return renderer.summary


__all__ = ["INDEX_SCHEMA", "LOOK_IDS", "ONCE", "ONCE_IDS", "PRODUCT_IDS", "QuicklookError", "Renderer", "Summary", "default_out", "render",
           "worker_environment", "worker_facts"]
