"""Downscale a WRF or ArWen run (3D) in the GUI: the chain's Advanced screen.

GET  /api/chain/copy                 the words the screen shows (``gui/copy/chain.json``) and the verification label
GET  /api/chain/files                wrfout-looking files and folders under the runs folder (a starting list)
GET  /api/chain/pictures/KEY/NAME    a domain map a command below drew
POST /api/chain/open   {path, map_time?}                 ``gpuwm-cm1 chain inspect SRC --map-png P --json``
POST /api/chain/check  {path, window, ratio, start?, end?, every?, moisture?, scheme?}
                                     ``gpuwm-cm1 chain check --history SRC --window ... --json``
POST /api/chain/new    {path, window, ratio, start?, end?, every?, moisture?, scheme?, name, folder?, namelist?}
                                     ``gpuwm-cm1 chain new ... --out RUNDIR --json``: a run folder Start runs

``moisture`` is ``all-species`` (the default) or ``vapour-only`` (WRF's ``have_bcs_moist = .false.``);
``scheme`` is the GM1 microphysics a vapour-only run uses (``wsm6-mp6`` or ``morrison-mp10``).
``path`` is one history file or a folder of them (one domain's).  ``window`` is ``[i0, j0, ni, nj]`` in the
history's mass cells (first cell from 0, then the size), as ``--window`` takes it.  Every POST takes
``dry_run`` and returns its one command, as every GUI action does.  Maps go under ``ROOT/.gm1-gui/chain/``,
named by the request, so the line "Show command" prints before a press is the line the press runs.  Standard
library only: the commands read the history, never this process.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from . import commands as C
from .files import NAME_RE, PathRefused, is_run_dir, require_name

if TYPE_CHECKING:  # pragma: no cover
    from .api import Api, Reply

COPY_FILE = Path(__file__).resolve().parent / "copy" / "chain.json"
CHAIN_DIR = "chain"
HISTORY_NAME = re.compile(r"^(wrfout|wrfhist|auxhist|history|arwen)[^/]*$|\.nc4?$", re.I)
TIMES_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[_ T]\d{2}:\d{2}(:\d{2})?$")
LIST_DEPTH = 3
LIST_LIMIT = 200
#: The GM1 schemes the screen offers a vapour-only run (Thompson needs its acknowledgment: the command line).
SCHEMES = ("wsm6-mp6", "morrison-mp10")
#: The ratios the screen offers (records/GM1-ARWEN-CHAIN.md 2: odd, 3 or more).
RATIOS = (3, 5, 7, 9)
#: Preparing a big box reads every saved time; the check and inspect are seconds.
NEW_TIMEOUT_S = 1800.0


def load_copy() -> dict[str, Any]:
    return json.loads(COPY_FILE.read_text(encoding="utf-8"))


def _key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]


def _stamp(path: Path) -> list[Any]:
    """What makes a request's pictures stale: the file's (or the folder's files') sizes and times."""

    paths = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file())
    out: list[Any] = []
    for p in paths[:400]:
        try:
            st = p.stat()
        except OSError:
            continue
        out.append([p.name, st.st_size, st.st_mtime_ns])
    return out


class ChainApi:
    def __init__(self, api: "Api") -> None:
        self.api = api

    @property
    def base(self) -> Path:
        return self.api.root.server_dir / CHAIN_DIR

    def _error(self, status: int, message: str, fix: str = "", **extra: Any):
        from .api import ApiError

        return ApiError(status, message, fix, **extra)

    # ------------------------------------------------------------ the request

    def history(self, payload: dict[str, Any]) -> Path:
        raw = payload.get("path")
        if not isinstance(raw, str) or not raw.strip():
            raise self._error(400, "No history was given.",
                              "Type the full path of a wrfout file, or of a folder of one domain's wrfout files.")
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            path = self.api.root.path / path
        path = path.resolve()
        if not path.exists():
            raise self._error(400, f"There is nothing at {path}.",
                              "Check the path; it must be on the computer running GM1.")
        if is_run_dir(path):
            raise self._error(400, f"{path} is a GM1 run folder, not a WRF or ArWen history.",
                              "Give a wrfout file or a folder of them. A GM1 run is turned into frames with "
                              "gpuwm-cm1 chain frames-from-gm1.")
        return path

    @staticmethod
    def _time(value: Any, what: str) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str) or not TIMES_RE.match(value.strip()):
            raise ValueError(what)
        return value.strip().replace(" ", "_").replace("T", "_")

    def span(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            start = self._time(payload.get("start"), "start")
            end = self._time(payload.get("end"), "end")
        except ValueError as error:
            raise self._error(400, f"The {error} must be one of the saved times, as YYYY-MM-DD_HH:MM:SS.",
                              "Pick a time from the list.") from None
        every = payload.get("every")
        if every in (None, "", 0):
            every = None
        elif isinstance(every, bool) or not isinstance(every, (int, float)) or not every > 0:
            raise self._error(400, "The side update spacing must be a number of seconds.",
                              "Pick one of the spacings offered.")
        return {"start": start, "end": end, "every_s": float(every) if every else None}

    def window(self, payload: dict[str, Any]) -> list[int]:
        box = payload.get("window")
        try:
            if isinstance(box, str):
                box = [int(v) for v in box.split(",")]
            values = [int(v) for v in box]
            if len(values) != 4 or any(isinstance(v, bool) for v in box) or values[0] < 0 or values[1] < 0 \
                    or values[2] < 1 or values[3] < 1:
                raise ValueError
        except (TypeError, ValueError):
            raise self._error(400, "The box is not four whole numbers: first cell (i0, j0) and size (ni, nj).",
                              "Drag on the map to draw the box.") from None
        return values

    def ratio(self, payload: dict[str, Any]) -> int:
        ratio = payload.get("ratio")
        if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio not in RATIOS:
            raise self._error(400, f"The ratio must be one of {', '.join(map(str, RATIOS))}.",
                              "Pick how many GM1 cells go along each cell of your run.")
        return ratio

    def picture_url(self, path: Path) -> str:
        rel = path.relative_to(self.base)
        return "/api/chain/pictures/" + "/".join(quote(part) for part in rel.parts)

    @staticmethod
    def _failure(result: Any) -> tuple[str, str, str | None]:
        body = result.json if isinstance(result.json, dict) else {}
        refusal = body.get("refusal") if isinstance(body.get("refusal"), dict) else None
        if refusal:
            return str(refusal.get("reason") or ""), str(refusal.get("remedy") or ""), refusal.get("code")
        if body.get("error"):
            return str(body["error"]), str(body.get("fix") or ""), None
        lines = (result.stderr or "").strip().splitlines()
        return (lines[-1] if lines else "The command did not finish."), "Read the output below; it names the problem.", None

    # ------------------------------------------------------------ reads

    def get(self, parts: list[str], query: dict[str, list[str]]) -> "Reply | None":
        from .api import Reply

        if parts[:1] != ["chain"]:
            return None
        if parts == ["chain", "copy"]:
            from ..chain_door import VERIFIED

            return Reply(body={**load_copy(), "verified": VERIFIED, "ratios": list(RATIOS), "schemes": list(SCHEMES)})
        if parts == ["chain", "files"]:
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
        """History files under the runs folder, and the folders holding more than one (a starting list)."""

        root = self.api.root.path
        found: list[dict[str, Any]] = []
        folders: dict[str, int] = {}

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
                    folders[str(entry.parent)] = folders.get(str(entry.parent), 0) + 1
                    if len(found) >= LIST_LIMIT:
                        return

        walk(root, 0)
        many = [{"path": folder, "files": count} for folder, count in sorted(folders.items()) if count > 1]
        return {"root": str(root), "files": found, "folders": many, "depth": LIST_DEPTH, "limit": LIST_LIMIT}

    # ------------------------------------------------------------ actions

    def post(self, parts: list[str], payload: dict[str, Any]) -> "Reply | None":
        if parts[:1] != ["chain"] or len(parts) != 2:
            return None
        handler = {"open": self.open, "check": self.check, "new": self.new}.get(parts[1])
        if handler is None:
            return None
        return handler(payload)

    def open(self, payload: dict[str, Any]) -> "Reply":
        from .api import Reply

        path = self.history(payload)
        try:
            map_time = self._time(payload.get("map_time"), "map time")
        except ValueError:
            raise self._error(400, "The map's time must be one of the saved times.", "Pick a time from the list.") from None
        key = _key("open", str(path), _stamp(path), map_time)
        png = self.base / "maps" / key / "map.png"
        args = C.chain_inspect_args([path], map_png=png, map_time=map_time)
        self.api.jobs.stop_if_dry(args)
        png.parent.mkdir(parents=True, exist_ok=True)
        result = self.api.jobs.host(args)
        body = result.json if isinstance(result.json, dict) else {}
        if result.rc != 0 or body.get("action") != "inspect":
            reason, fix, code = self._failure(result)
            return Reply(200, {"ok": False, "command": result.command, "message": reason, "fix": fix, "code": code,
                               "output": result.as_dict()})
        if isinstance(body.get("map"), dict) and png.is_file():
            body["map"]["url"] = self.picture_url(png)
        count = body.get("count")
        return Reply(200, {**body, "ok": True, "command": result.command, "path": str(path),
                           "message": f"{body.get('producer')} history: {count} saved time{'s' if count != 1 else ''}."})

    def moisture(self, payload: dict[str, Any]) -> dict[str, Any]:
        """``moisture`` and ``scheme``: the lane, and the GM1 scheme a vapour-only run uses."""

        moisture = payload.get("moisture")
        scheme = payload.get("scheme")
        if moisture in (None, "", "all-species"):
            moisture = None
        elif moisture != "vapour-only":
            raise self._error(400, "Moisture must be every cloud and rain field or water vapour only.",
                              "Pick one of the two choices.")
        if scheme in (None, ""):
            scheme = None
        elif scheme not in SCHEMES or moisture is None:
            raise self._error(400, "GM1 microphysics can be picked only with water vapour only: WSM6 or Morrison.",
                              "Set Moisture to water vapour only, or leave the scheme as your run's.")
        return {"moisture": moisture, "scheme": scheme}

    def _request(self, payload: dict[str, Any]) -> tuple[Path, list[int], int, dict[str, Any]]:
        return self.history(payload), self.window(payload), self.ratio(payload), {**self.span(payload),
                                                                                  **self.moisture(payload)}

    def check(self, payload: dict[str, Any]) -> "Reply":
        from .api import Reply

        path, window, ratio, span = self._request(payload)
        args = C.chain_check_args([path], window, ratio, **span)
        self.api.jobs.stop_if_dry(args)
        result = self.api.jobs.host(args)
        body = result.json if isinstance(result.json, dict) else {}
        if result.rc not in (0, 3) or body.get("action") != "check":
            reason, fix, code = self._failure(result)
            return Reply(200, {"ok": False, "command": result.command, "message": reason, "fix": fix, "code": code,
                               "output": result.as_dict()})
        body.pop("facts", None)          # the admission's numbers stay in the command's own JSON
        return Reply(200, {**body, "ok": True, "command": result.command, "rc": result.rc,
                           "message": body.get("summary") or ""})

    def new(self, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, run_url

        path, window, ratio, span = self._request(payload)
        name = require_name(str(payload.get("name") or ""), "run name")
        folder = str(payload.get("folder") or "").strip("/")
        run_id = f"{folder}/{name}" if folder else name
        rundir = self.api.root.run_path(run_id)
        if rundir.exists():
            raise ApiError(409, f"A run called {run_id!r} already exists.", "Pick another name.")
        if folder and is_run_dir(rundir.parent):
            raise ApiError(409, f"{folder!r} is a run folder; a run cannot go inside another run.",
                           "Pick another folder, or leave the folder empty.")
        namelist = None
        if payload.get("namelist") not in (None, "", {}):
            namelist = self.api._input_file(payload.get("namelist"), "namelist.input", 1024 * 1024, "namelist",
                                            required=True)
        args = C.chain_new_args([path], window, ratio, rundir, namelist=namelist, **span)
        self.api.jobs.stop_if_dry(args)
        rundir.parent.mkdir(parents=True, exist_ok=True)
        result = self.api.jobs.host(args, timeout=NEW_TIMEOUT_S)
        created = (rundir / "run.json").is_file()
        if not created or result.rc != 0:
            reason, fix, code = self._failure(result)
            return self.api._answer("chain-new", None, result, False, reason, fix, verdict="error", url=None,
                                    code=code)
        self.api._log(rundir, "chain-new", result.command,
                      "a finer GM1 box driven by a WRF or ArWen history, prepared; nothing runs yet")
        body = result.json if isinstance(result.json, dict) else {}
        return self.api._answer("chain-new", run_id, result, True, load_copy()["screen"]["made"].format(name=run_id),
                                "", verdict="runnable", url=run_url(run_id), start=body.get("start"))


__all__ = ["ChainApi", "RATIOS", "SCHEMES", "load_copy"]
