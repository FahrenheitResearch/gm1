"""Compare with other runs, on the Results screen.

``GET /api/runs/RUN/compare``
    The comparisons drawn into this run's ``quicklook/compare/`` (newest first): their state while being drawn,
    their runs, the time they show and every picture with its address.  With ``?choices=1`` it also says which
    runs of the Runs list cannot be compared with this one, and why (``cannot``: a hurricane run beside a
    thunderstorm run, or a run with no saved frames and no quick-look pictures), so the list can grey them out
    before Draw is pressed (the command refuses them as well, if they are named).
``POST /api/runs/RUN/compare {with: [RUN, ...], time_s?, box_km?}``
    ``gpuwm-cm1 quicklook RUNDIR --compare OTHER [OTHER ...] [--time S] [--box KM] --json``, launched in the
    background at nice 19 with no GPU; the runs are picked from the Runs list (this server's runs folder), one to
    five besides this one.  ``{"dry_run": true}`` returns the command alone, as every POST does.

The pictures themselves are :mod:`cm1core.quicklook.compare`'s; this module never imports the drawing code.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from . import commands as C
from .files import cm1_output, read_json

if TYPE_CHECKING:
    from .api import Api, Reply

#: quicklook.compare.MAX_RUNS, written out so the server never imports the drawing code (a test holds them equal).
MAX_RUNS = 6
#: What each picture of a comparison is, by its key in compare.json (quicklook.compare and compare_draw).
TITLES = {
    "storm-reflectivity": "Side by side: reflectivity at 1 km",
    "storm-clouds": "Side by side: clouds from above",
    "storm-surface": "Side by side: cold air and wind near the ground",
    "storm-wind": "Side by side: wind at the lowest level",
    "series": "Runs over time",
    "soundings": "Soundings and what they grew",
}
ORDER = ("storm-reflectivity", "storm-clouds", "storm-surface", "storm-wind", "series", "soundings")
#: The measured cost of a comparison on one core (a Linux workstation): 4 to 11 s for two to six 1 km runs with
#: their series.json, 5 s for two hurricane runs; a run without series.json adds about 0.3 s per saved frame at
#: 1 km.  The "about" figure scales the larger of these with the cells of the biggest run, as the More looks do.
SECONDS_BASE = 6.0
SECONDS_PER_RUN = 2.0
#: quicklook.facts.HURRICANE_IINIT, written out as MAX_RUNS is: CM1's iinit for the hurricane swirl, which makes a
#: run a hurricane run.
HURRICANE_IINIT = 7


def is_hurricane(rundir: Path) -> bool | None:
    """Whether a run is a hurricane run, as the quick-looks decide it (its namelist's iinit), else as its pictures
    recorded it; None when neither says."""

    try:
        text = (rundir / "namelist.input").read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = None
    if text is not None:
        found = re.search(r"(?im)^\s*iinit\s*=\s*(-?\d+)", text)
        return int(found.group(1)) == HURRICANE_IINIT if found else False
    index = read_json(rundir / "quicklook" / "index.json", default=None)
    shown = index.get("run") if isinstance(index, dict) and isinstance(index.get("run"), dict) else None
    if shown is not None and shown.get("storm_kind"):
        return shown.get("storm_kind") == "hurricane"
    return None


def has_storm(rundir: Path) -> bool:
    """Whether a run has saved frames (``frames/`` or ``segments/NNN/frames/``) or its quick-look pictures, or is
    CM1 netCDF output whose files are there to read."""

    if (rundir / "quicklook" / "index.json").is_file():
        return True
    found = cm1_output(rundir)
    if found is not None and found["source"].exists():
        return True
    places = [rundir / "frames"]
    if (rundir / "segments").is_dir():
        places += sorted(p / "frames" for p in (rundir / "segments").iterdir() if p.name.isdigit())
    for frames in places:
        try:
            with os.scandir(frames) as entries:
                if any(True for _ in entries):
                    return True
        except OSError:
            continue
    return False


def cannot(rundir: Path, others: list[tuple[str, Path]]) -> dict[str, str]:
    """{run id: why it cannot be compared with the run in ``rundir``} for the runs that cannot."""

    here = is_hurricane(rundir)
    out: dict[str, str] = {}
    for run_id, path in others:
        if not has_storm(path):
            out[run_id] = "no saved frames and no quick-look pictures, so there is nothing of its storm to compare"
            continue
        kind = is_hurricane(path)
        if here is not None and kind is not None and kind != here:
            out[run_id] = ("a hurricane run, compared only with other hurricane runs" if kind else
                           "a thunderstorm run: a hurricane run is compared only with other hurricane runs")
    return out


def folder(rundir: Path) -> Path:
    return rundir / "quicklook" / "compare"


def _url(run_id: str, relative: str) -> str:
    return f"/api/runs/{quote(run_id, safe='')}/files/quicklook/compare/" + quote(relative)


def comparisons(run_id: str, rundir: Path) -> list[dict[str, Any]]:
    """Every comparison in this run's ``quicklook/compare/``, newest first."""

    root = folder(rundir)
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        record = read_json(path / "compare.json", default=None)
        progress = read_json(path / "progress.json", default=None)
        if not isinstance(record, dict) and not isinstance(progress, dict):
            continue
        row: dict[str, Any] = {"folder": path.name, "state": "done" if isinstance(record, dict) else "none"}
        if isinstance(progress, dict):
            row["progress"] = {k: progress.get(k) for k in ("state", "done", "total", "step", "message",
                                                            "updated_utc")}
            if progress.get("state") in ("running", "failed"):
                row["state"] = progress["state"]
        stamp = (progress or {}).get("updated_utc") if isinstance(progress, dict) else None
        if isinstance(record, dict):
            row.update({"made_utc": record.get("made_utc"), "time_s": record.get("time_s"),
                        "kind": record.get("kind"), "differ": record.get("differ") or [],
                        "not_drawn": record.get("not_drawn") or {},
                        "runs": [{k: r.get(k) for k in ("letter", "colour", "name", "grid", "time_s",
                                                       "saved_frames")}
                                 for r in record.get("runs") or [] if isinstance(r, dict)]})
            stamp = record.get("made_utc") or stamp
            pictures = []
            shown = record.get("pictures") if isinstance(record.get("pictures"), dict) else {}
            for key in sorted(shown, key=lambda k: ORDER.index(k) if k in ORDER else len(ORDER)):
                for item in shown[key] if isinstance(shown[key], list) else [shown[key]]:
                    if not isinstance(item, dict) or not isinstance(item.get("file"), str):
                        continue
                    name = item["file"]
                    title = TITLES.get(key, key)
                    if isinstance(shown[key], list) and len(shown[key]) > 1:
                        title += f" ({', '.join(item.get('runs') or [])})"
                    pictures.append({"key": key, "title": title, "url": _url(run_id, f"{path.name}/{name}"),
                                     "thumb_url": _url(run_id, f"{path.name}/thumb/{name}")})
            row["pictures"] = pictures
        row["stamp"] = stamp or ""
        rows.append(row)
    running = [r for r in rows if r["state"] == "running"]
    finished = sorted((r for r in rows if r["state"] != "running"), key=lambda r: r["stamp"], reverse=True)
    return running + finished


def about_seconds(rundirs: list[Path]) -> int:
    """A plain "about N s" for drawing a comparison, from the measured costs (estimated)."""

    cells = 0.576e6
    for rundir in rundirs:
        index = read_json(rundir / "quicklook" / "index.json", default=None)
        grid = str(((index or {}).get("run") or {}).get("grid") or "") if isinstance(index, dict) else ""
        parts = grid.split(" × ")
        try:
            cells = max(cells, float(parts[0]) * float(parts[1]) * float(parts[2].split(",")[0]))
        except (IndexError, ValueError):
            pass
    scale = max(1.0, cells / 0.576e6) ** 0.5
    return int(round((SECONDS_BASE + SECONDS_PER_RUN * len(rundirs)) * scale))


class CompareApi:
    """The compare endpoints of one server (an :class:`.api.Api`)."""

    def __init__(self, api: "Api") -> None:
        self.api = api

    def get(self, parts: list[str], query: dict[str, list[str]]) -> "Reply | None":
        from .api import Reply

        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "compare":
            run_id = parts[1]
            rundir = self.api.root.existing_run(run_id)
            busy = bool([row for row in self.api.jobs.launched(run_id) if row.role == "compare"])
            body: dict[str, Any] = {"comparisons": comparisons(run_id, rundir), "drawing": busy,
                                    "max_runs": MAX_RUNS}
            if (query.get("choices") or [""])[0] in ("1", "true"):
                others = [(self.api.root.run_id(path), path) for path in self.api.root.iter_runs()
                          if path.resolve() != rundir.resolve()]
                body["cannot"] = cannot(rundir, others)
            return Reply(body=body)
        return None

    def post(self, run_id: str, rundir: Path, action: str, payload: dict[str, Any]) -> "Reply | None":
        if action != "compare":
            return None
        from .api import ApiError, Reply

        others = payload.get("with")
        if not isinstance(others, list) or not others or not all(isinstance(o, str) and o for o in others):
            raise ApiError(400, "with must list the runs to compare with, by their names in the Runs list.",
                           "Pick one to five other runs.")
        if len(others) > MAX_RUNS - 1:
            raise ApiError(400, f"At most {MAX_RUNS} runs are compared at once: this one and {MAX_RUNS - 1} others.",
                           "Pick fewer runs.")
        if len(set(others)) != len(others) or run_id in others:
            raise ApiError(400, "Each run can be picked once, and this run is always the first.",
                           "Pick other runs.")
        dirs = [self.api.root.existing_run(other) for other in others]
        time_s = self.api._number(payload, "time_s", low=0.0, high=1.0e9)
        box_km = self.api._number(payload, "box_km", low=2.0, high=2000.0)
        args = C.compare_args(rundir, dirs, time_s=time_s, box_km=box_km)
        line = C.display(args)
        with self.api.jobs.run_lock(run_id):
            if [row for row in self.api.jobs.launched(run_id) if row.role == "compare"]:
                raise ApiError(409, "A comparison of this run is being drawn now.", "Wait for it.", command=line)
            self.api.jobs.require("quicklook")
            launched = self.api.jobs.launch(run_id, rundir, args, role="compare")
            self.api.jobs.log_command(rundir, "compare", C.sidecar_line(args).removesuffix(" &"),
                                      "a comparison with other runs, on the CPU")
        about = about_seconds([rundir, *dirs])
        return Reply(200, {"ok": True, "action": "compare", "run": run_id, "command": line,
                           "pid": launched.process.pid, "about_s": about,
                           "message": f"Drawing the comparison on the computer's processor (nice 19, no GPU); "
                                      f"about {about} s.", "fix": ""})


__all__ = ["CompareApi", "HURRICANE_IINIT", "MAX_RUNS", "TITLES", "about_seconds", "cannot", "comparisons",
           "has_storm", "is_hurricane"]
