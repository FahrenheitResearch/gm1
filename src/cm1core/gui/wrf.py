"""The column route in the GUI: an environment from a WRF or ArWen run.

GET  /api/wrf/copy                   the words the card and the tab show (``gui/copy/wrf.json``)
GET  /api/wrf/files                  wrfout-looking files under the runs folder (a starting list; any path works)
GET  /api/wrf/pictures/KEY/NAME      a map or preview picture a command below drew
POST /api/wrf/open    {path, record?|time?, map?}         ``gpuwm-cm1 sounding times FILE ... --map-png P --json``
POST /api/wrf/column  {path, record?|time?, at|ij, map?}  ``gpuwm-cm1 sounding from-wrf FILE ... --out P --json``
POST /api/wrf/use     {path, record?|time?, at|ij, target: "import", name, folder?, namelist?, frame?}
                                     ``gpuwm-cm1 import NAMELIST --wrf FILE ... --out RUNDIR --json``
                      {..., target: "guided", budget?}   the column kept for the guided fit (``sounding from-wrf``
                                     into ``ROOT/.gm1-gui/wrf/soundings/``), and the route to "How long can it run?"

Every POST takes ``dry_run`` and returns its one command, as every GUI action does.  The pictures and soundings a
command writes go under ``ROOT/.gm1-gui/wrf/``, named by the request, so the line "Show command" prints before a
press is the line the press runs.  Standard library only: the commands read the file, never this process.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from . import commands as C
from .files import NAME_RE, PathRefused, require_name

if TYPE_CHECKING:  # pragma: no cover
    from .api import Api, Reply

COPY_FILE = Path(__file__).resolve().parent / "copy" / "wrf.json"
WRF_DIR = "wrf"
#: The guided card's storm: the isnd = 7 preset, fitted with the column as its sounding.
GUIDED_PRESET = "your-environment"
#: The advanced tab's default namelist: CM1's own supercell example, which the import turns to isnd = 7.
DEFAULT_NAMELIST = "cm1r21.1-supercell-namelist"
HISTORY_NAME = re.compile(r"^(wrfout|wrfhist|auxhist|history|arwen)[^/]*$|\.nc4?$|\.nc$", re.I)
#: How deep GET /api/wrf/files looks under the runs folder, and how many files it lists.
LIST_DEPTH = 3
LIST_LIMIT = 200
TIMES_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[_ T]\d{2}:\d{2}(:\d{2})?Z?$")


def load_copy() -> dict[str, Any]:
    return json.loads(COPY_FILE.read_text(encoding="utf-8"))


def _key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]


class WrfApi:
    def __init__(self, api: "Api") -> None:
        self.api = api

    # ------------------------------------------------------------ helpers

    @property
    def base(self) -> Path:
        return self.api.root.server_dir / WRF_DIR

    def _error(self, status: int, message: str, fix: str = "", **extra: Any):
        from .api import ApiError

        return ApiError(status, message, fix, **extra)

    def _history(self, payload: dict[str, Any]) -> Path:
        raw = payload.get("path")
        if not isinstance(raw, str) or not raw.strip():
            raise self._error(400, "No file was given.", "Type the full path of a wrfout or ArWen history file.")
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            path = self.api.root.path / path
        path = path.resolve()
        if not path.is_file():
            raise self._error(400, f"There is no file at {path}.",
                              "Check the path; it must be a file on the computer running GM1.")
        return path

    @staticmethod
    def _when(payload: dict[str, Any]) -> dict[str, Any]:
        time_text = payload.get("time")
        record = payload.get("record")
        if time_text not in (None, ""):
            if not isinstance(time_text, str) or not TIMES_RE.match(time_text.strip()):
                raise ValueError("time")
            return {"time": time_text.strip().replace(" ", "_").replace("T", "_").rstrip("Z")}
        if record not in (None, ""):
            if isinstance(record, bool) or not isinstance(record, int) or record < 0:
                raise ValueError("record")
            return {"record": record}
        return {}

    def when(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._when(payload)
        except ValueError:
            raise self._error(400, "The time must be one of the file's times, as YYYY-MM-DD_HH:MM:SS.",
                              "Pick a time from the list.") from None

    def point(self, payload: dict[str, Any]) -> dict[str, Any]:
        at, ij = payload.get("at"), payload.get("ij")
        if (at is None) == (ij is None):
            raise self._error(400, "Pick one point: a latitude and longitude, or a grid column.",
                              "Click the map, or type a latitude and longitude.")
        try:
            if at is not None:
                lat, lon = (float(v) for v in at)
                if not (-90.0 <= lat <= 90.0 and -360.0 <= lon <= 360.0):
                    raise ValueError
                return {"at": [lat, lon]}
            i, j = (int(v) for v in ij)
            if i < 1 or j < 1:
                raise ValueError
            return {"ij": [i, j]}
        except (TypeError, ValueError):
            raise self._error(400, "The point is not a latitude and longitude, or a grid column (i, j from 1).",
                              "Click the map, or type a latitude and longitude such as 35.2, -97.4.") from None

    def _map_field(self, payload: dict[str, Any]) -> str:
        field = payload.get("map") or "shear"
        if field not in ("shear", "cape"):
            raise self._error(400, f"The map shows shear or CAPE, not {field!r}.", "Pick shear or CAPE.")
        return str(field)

    def picture_url(self, path: Path) -> str:
        rel = path.relative_to(self.base)
        return "/api/wrf/pictures/" + "/".join(quote(part) for part in rel.parts)

    @staticmethod
    def _refusal(result: Any) -> tuple[str, str]:
        body = result.json if isinstance(result.json, dict) else {}
        reason = body.get("reason") or (result.stderr.strip().splitlines() or ["The command did not finish."])[-1]
        return str(reason), str(body.get("remedy") or "")

    # ------------------------------------------------------------ reads

    def get(self, parts: list[str], query: dict[str, list[str]]) -> "Reply | None":
        from .api import Reply

        if parts[:1] != ["wrf"]:
            return None
        if parts == ["wrf", "copy"]:
            return Reply(body=load_copy())
        if parts == ["wrf", "files"]:
            return Reply(body=self.files())
        if len(parts) >= 4 and parts[1] == "pictures":
            name = parts[-1]
            if not NAME_RE.match(name) or not name.endswith(".png"):
                raise PathRefused(f"{name!r} is not a picture name.")
            path = self.base.joinpath(*parts[2:])
            if not path.resolve().is_relative_to(self.base.resolve()) or not path.is_file():
                raise FileNotFoundError("No such picture.")
            return Reply(file=path)
        return None

    def files(self) -> dict[str, Any]:
        root = self.api.root.path
        found: list[dict[str, Any]] = []

        def walk(directory: Path, depth: int) -> None:
            if depth > LIST_DEPTH or len(found) >= LIST_LIMIT:
                return
            try:
                entries = sorted(directory.iterdir(), key=lambda p: p.name)
            except OSError:
                return
            for entry in entries:
                if entry.name.startswith(".") or entry.is_symlink():
                    continue
                if entry.is_dir():
                    if (entry / "run.json").is_file():
                        continue           # a GM1 run folder holds GM1's own files, not WRF histories
                    walk(entry, depth + 1)
                elif entry.is_file() and HISTORY_NAME.search(entry.name):
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        continue
                    found.append({"path": str(entry), "name": entry.name, "bytes": size,
                                  "folder": str(entry.parent.relative_to(root)) if entry.parent != root else ""})
                    if len(found) >= LIST_LIMIT:
                        return

        walk(root, 0)
        return {"root": str(root), "files": found, "depth": LIST_DEPTH, "limit": LIST_LIMIT}

    # ------------------------------------------------------------ actions

    def post(self, parts: list[str], payload: dict[str, Any]) -> "Reply | None":
        if parts[:1] != ["wrf"] or len(parts) != 2:
            return None
        handler = {"open": self.open, "column": self.column, "use": self.use}.get(parts[1])
        if handler is None:
            return None
        return handler(payload)

    def open(self, payload: dict[str, Any]) -> "Reply":
        from .api import Reply

        path = self._history(payload)
        when = self.when(payload)
        field = self._map_field(payload)
        stat = path.stat()
        key = _key("open", str(path), stat.st_size, stat.st_mtime_ns, when, field)
        png = self.base / "maps" / key / f"map-{field}.png"
        args = C.wrf_times_args(path, when=when, map_field=field, map_png=png, native=self.api.native)
        self.api.jobs.stop_if_dry(args)
        png.parent.mkdir(parents=True, exist_ok=True)
        result = self.api.jobs.host(args)
        body = result.json if isinstance(result.json, dict) else {}
        if result.rc != 0 or not body.get("ok"):
            reason, remedy = self._refusal(result)
            return Reply(200, {"ok": False, "command": result.command, "message": reason, "fix": remedy,
                               "code": body.get("code"), "output": result.as_dict()})
        if isinstance(body.get("map"), dict) and png.is_file():
            body["map"]["url"] = self.picture_url(png)
        return Reply(200, {**body, "ok": True, "command": result.command,
                           "message": f"{body.get('producer')} history: {body.get('records')} time(s)."})

    def _column_args(self, path: Path, when: dict[str, Any], point: dict[str, Any], field: str,
                     out_dir: Path) -> list[str]:
        return C.wrf_column_args(path, when=when, point=point, out=out_dir / "input_sounding",
                                 map_field=field, map_png=out_dir / f"map-{field}.png",
                                 preview_png=out_dir / "preview.png", native=self.api.native)

    def column(self, payload: dict[str, Any]) -> "Reply":
        from .api import Reply

        path = self._history(payload)
        when = self.when(payload)
        point = self.point(payload)
        field = self._map_field(payload)
        stat = path.stat()
        key = _key("column", str(path), stat.st_size, stat.st_mtime_ns, when, point, field)
        out_dir = self.base / "columns" / key
        args = self._column_args(path, when, point, field, out_dir)
        self.api.jobs.stop_if_dry(args)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = self.api.jobs.host(args)
        body = result.json if isinstance(result.json, dict) else {}
        if result.rc != 0 or not body.get("ok"):
            reason, remedy = self._refusal(result)
            return Reply(200, {"ok": False, "command": result.command, "message": reason, "fix": remedy,
                               "code": body.get("code"), "output": result.as_dict()})
        for part in ("preview", "map"):
            picture = body.get(part)
            if isinstance(picture, dict) and picture.get("file") and Path(picture["file"]).is_file():
                picture["url"] = self.picture_url(Path(picture["file"]))
        body.pop("environment", None)             # the full getcape block stays in wrf-column.json
        return Reply(200, {**body, "ok": True, "command": result.command, "key": key,
                           "message": "The column is ready. Look at it, then use it."})

    def use(self, payload: dict[str, Any]) -> "Reply":
        target = payload.get("target")
        if target == "import":
            return self._use_import(payload)
        if target == "guided":
            return self._use_guided(payload)
        raise self._error(400, "Use the column for what: target must be guided or import.",
                          "Send target: guided (a storm from the gallery) or import (a run from a namelist).")

    def _namelist(self, spec: Any) -> Path:
        if spec in (None, "", {}) or (isinstance(spec, dict) and spec.get("example") == "cm1-supercell"):
            from ..data import PackagedInputError, packaged_input_path

            try:
                return packaged_input_path(DEFAULT_NAMELIST)
            except PackagedInputError as error:
                raise self._error(500, f"CM1's supercell namelist is not installed ({error}).",
                                  "Reinstall gpuwm-cm1, or give your own namelist.") from error
        return self.api._input_file(spec, "namelist.input", 1024 * 1024, "namelist", required=True)

    def _use_import(self, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, run_url
        from .files import is_run_dir

        path = self._history(payload)
        when = self.when(payload)
        point = self.point(payload)
        name = require_name(str(payload.get("name") or ""), "run name")
        folder = str(payload.get("folder") or "").strip("/")
        run_id = f"{folder}/{name}" if folder else name
        rundir = self.api.root.run_path(run_id)
        if rundir.exists():
            raise ApiError(409, f"A run called {run_id!r} already exists.", "Pick another name.")
        if folder and is_run_dir(rundir.parent):
            raise ApiError(409, f"{folder!r} is a run folder; a run cannot go inside another run.",
                           "Pick another folder, or leave the folder empty.")
        namelist = self._namelist(payload.get("namelist"))
        frame = payload.get("frame", "namelist")
        if frame not in ("namelist", "follow"):
            raise self._error(400, "frame must be namelist or follow.",
                              "follow moves the domain with the storm; namelist keeps your namelist's frame.")
        args = C.wrf_import_args(namelist, path, when=when, point=point, out=rundir, frame=frame,
                                 native=self.api.native)
        self.api.jobs.stop_if_dry(args)
        rundir.parent.mkdir(parents=True, exist_ok=True)
        result = self.api.jobs.host(args)
        self.api._log(rundir, "import", result.command, f"rc {result.rc}")
        verdicts = {
            0: ("runnable", "Imported. GM1 runs it.", ""),
            3: ("needs-ok", "Imported. It runs once you OK the settings GM1 does not read.",
                "Open Check, read the list, and tick the box."),
            2: ("refused", "Imported, but GM1 cannot run it as written.",
                "Open Check: each refused setting says what to change."),
        }
        created = (rundir / "run.json").is_file()
        if not created:
            reason = (result.stderr.strip().splitlines() or ["The import did not finish."])[-1]
            try:
                detail = json.loads(result.stderr[result.stderr.index("{"):])
                reason = detail.get("reason") or reason
                fix = detail.get("remedy") or ""
            except ValueError:
                fix = ""
            return self.api._answer("import", None, result, False, reason, fix, verdict="error", url=None,
                                    mode=self.api.mode()["mode"])
        verdict, message, fix = verdicts.get(result.rc, (
            "error", "The import did not finish.", "Read the output below; it names the problem."))
        return self.api._answer("import", run_id, result, result.rc in (0, 3), message, fix, verdict=verdict,
                                url=run_url(run_id), mode=self.api.mode()["mode"])

    def _use_guided(self, payload: dict[str, Any]) -> "Reply":
        from .api import Reply

        path = self._history(payload)
        when = self.when(payload)
        point = self.point(payload)
        budget = str(payload.get("budget") or "1h")
        if budget not in ("10m", "1h", "overnight", "any"):
            raise self._error(400, "The time must be 10m, 1h, overnight or any.", "Pick one of the four.")
        stat = path.stat()
        key = _key("use", str(path), stat.st_size, stat.st_mtime_ns, when, point)
        out = self.base / "soundings" / key / "input_sounding"
        # The column must reach the top of the storm's model, or the fit would refuse it one screen later.
        from .guided import presets_dir

        storm = presets_dir() / GUIDED_PRESET / "namelist.input"
        args = C.wrf_column_args(path, when=when, point=point, out=out, native=self.api.native,
                                 top_of=storm if storm.is_file() else None)
        self.api.jobs.stop_if_dry(args)
        out.parent.mkdir(parents=True, exist_ok=True)
        result = self.api.jobs.host(args)
        body = result.json if isinstance(result.json, dict) else {}
        if result.rc != 0 or not body.get("ok"):
            reason, remedy = self._refusal(result)
            if body.get("code") == "GM1-WRF-TOP":
                words = load_copy()["strings"]
                reason = words["use.too_low"]["text"].format(top=f"{float(body.get('value') or 0) / 1000.0:.1f}")
                remedy = words["use.too_low_fix"]["text"]
            return Reply(200, {"ok": False, "command": result.command, "message": reason, "fix": remedy,
                               "code": body.get("code"), "output": result.as_dict()})
        where = body.get("point") or {}
        src = body.get("source") or {}
        summary = load_copy()["strings"]["guided.summary"]["text"].format(
            lat=f"{where.get('lat', 0):.2f}", lon=f"{where.get('lon', 0):.2f}",
            time=str(src.get("times", "")).replace("_", " ") + " UTC")
        return Reply(200, {
            "ok": True, "command": result.command, "sounding": str(out), "record_file": body.get("record_file"),
            "preset": GUIDED_PRESET, "budget": budget, "summary": summary,
            "route": f"guided/fit/{GUIDED_PRESET}/{budget}?sounding={quote(str(out), safe='')}",
            "fit_command": C.display(C.wrf_fit_args(GUIDED_PRESET, budget, out, self.api.root.path)),
            "message": "The column is kept for your storm. Next: how long can it run?",
        })


__all__ = ["DEFAULT_NAMELIST", "GUIDED_PRESET", "WrfApi", "load_copy"]
