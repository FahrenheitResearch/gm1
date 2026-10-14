"""Guided mode's "Use my own sounding" card: a sounding file of the user's own, checked and kept for a storm.

GET  /api/sounding/copy                  the words the card shows (``gui/copy/sounding.json``)
GET  /api/sounding/shipped               the balloon soundings GM1 ships (``data/soundings/index.json``), the
                                         storm the card runs, and how to name a Windows file under WSL
GET  /api/sounding/files/KEY/NAME        a kept ``input_sounding`` or the preview picture a check drew
POST /api/sounding/check   {text | path | shipped, name?, speed_unit?, extend?}
                                         ``gpuwm-cm1 sounding convert IN OUT --top-of NAMELIST ... --json``
POST /api/sounding/options {start}       ``gpuwm-cm1 preset options PRESET --start START --json``

``check`` takes the file as text (the browser reads the file the user chose, or what was pasted), as a path on
the computer running GM1, or as one of the shipped soundings' names.  Text is kept under
``ROOT/.gm1-gui/soundings/KEY/original/NAME`` (KEY names the text and the choices, so the line "Show command"
prints before a press is the line the press runs); the command reads it and, when a storm can run on it, writes
CM1's ``input_sounding`` and its record beside it under the same KEY.  The reply is the command's own record
(what layout the file has, each conversion, each problem with its line, the numbers, the sentence, the start),
with the picture's address and the route to "How long can it run?".

The storm is the ``your-environment`` preset run on the kept file (``preset fit your-environment --sounding FILE
--start START``), as the WRF card's is.  Standard library only: the command reads the file, never this process.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from . import commands as C
from .files import NAME_RE, PathRefused

if TYPE_CHECKING:  # pragma: no cover
    from .api import Api, Reply

COPY_FILE = Path(__file__).resolve().parent / "copy" / "sounding.json"
SOUNDINGS_DIR = "soundings"
#: The storm the card runs: the preset that reads a sounding file, as the WRF card's.
GUIDED_PRESET = "your-environment"
#: The largest text taken as a sounding (``cm1import.sounding_formats.MAX_BYTES``).
MAX_BYTES = 4 * 1024 * 1024
STARTS = ("push", "bubble", "strong-bubble")
WINDOWS_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def load_copy() -> dict[str, Any]:
    return json.loads(COPY_FILE.read_text(encoding="utf-8"))


def _key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]


def safe_name(name: Any) -> str:
    """A plain file name from what the browser called the file (its folder and odd characters dropped)."""

    text = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem, dot, suffix = text.rpartition(".")
    if not dot:
        stem, suffix = text, ""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_")[:48].strip("-_")
    suffix = re.sub(r"[^A-Za-z0-9_]+", "", suffix)[:16]
    text = f"{stem}.{suffix}" if stem and suffix else stem
    return text if text and NAME_RE.match(text) else "sounding.txt"


def shipped_soundings() -> list[dict[str, Any]]:
    """The balloon soundings in the package, as the card lists them (newest first)."""

    from .. import data

    folder = Path(data.__file__).resolve().parent / "soundings"
    try:
        index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = []
    for row in index.get("soundings") or []:
        name = str(row.get("filename") or "")
        if not (folder / name).is_file():
            continue
        launch = str(row.get("launch_utc") or "")
        rows.append({
            "id": name, "place": row.get("place"), "launch_utc": launch,
            "title": f"{row.get('place')}, {launch[:10]} {launch[11:13]}00 UTC",
            "observation": row.get("observation"), "levels": row.get("levels_used"),
            "top_m": row.get("top_m_agl"), "rights": row.get("rights"), "path": str(folder / name),
        })
    return sorted(rows, key=lambda row: row["launch_utc"], reverse=True)


class SoundingApi:
    def __init__(self, api: "Api") -> None:
        self.api = api

    @property
    def base(self) -> Path:
        return self.api.root.server_dir / SOUNDINGS_DIR

    def _error(self, status: int, message: str, fix: str = "", **extra: Any):
        from .api import ApiError

        return ApiError(status, message, fix, **extra)

    def _storm_namelist(self) -> Path | None:
        from .guided import presets_dir

        path = presets_dir() / GUIDED_PRESET / "namelist.input"
        return path if path.is_file() else None

    @staticmethod
    def _download_name(path: Path) -> str | None:
        """LABEL.input_sounding for a kept file (the name the user's file had), so two downloads do not clash."""

        if path.name != "input_sounding":
            return None
        try:
            label = json.loads((path.parent / "sounding.json").read_text(encoding="utf-8")).get("label")
        except (OSError, ValueError, AttributeError):
            return None
        return f"{label}.input_sounding" if isinstance(label, str) and NAME_RE.match(label) else None

    def file_url(self, path: Path) -> str:
        rel = path.relative_to(self.base)
        return "/api/sounding/files/" + "/".join(quote(part) for part in rel.parts)

    # ------------------------------------------------------------ reads

    def get(self, parts: list[str], query: dict[str, list[str]]) -> "Reply | None":
        from .api import Reply

        if parts[:1] != ["sounding"]:
            return None
        if parts == ["sounding", "copy"]:
            return Reply(body=load_copy())
        if parts == ["sounding", "shipped"]:
            return Reply(body=self.shipped())
        if len(parts) == 4 and parts[1] == "files":
            key, name = parts[2], parts[3]
            if not re.fullmatch(r"[0-9a-f]{20}", key) or name not in ("input_sounding", "preview.png"):
                raise PathRefused(f"{name!r} is not a kept sounding file.")
            path = self.base / key / name
            if not path.is_file():
                raise FileNotFoundError("No such file.")
            return Reply(file=path, download_name=self._download_name(path))
        return None

    def shipped(self) -> dict[str, Any]:
        from ..cm1run.gpuinfo import is_wsl2

        rows = [{key: value for key, value in row.items() if key != "path"} for row in shipped_soundings()]
        return {"soundings": rows, "preset": GUIDED_PRESET, "wsl": bool(is_wsl2()), "max_bytes": MAX_BYTES,
                "starts": list(STARTS)}

    # ------------------------------------------------------------ actions

    def post(self, parts: list[str], payload: dict[str, Any]) -> "Reply | None":
        if parts == ["sounding", "check"]:
            return self.check(payload)
        if parts == ["sounding", "options"]:
            return self.options(payload)
        return None

    def _source(self, payload: dict[str, Any], write: bool) -> tuple[Path, str, str | None, str]:
        """(the file the command reads, the name it is known by, the shipped sounding's words, what names it)."""

        text, path, shipped = payload.get("text"), payload.get("path"), payload.get("shipped")
        given = [value for value in (text, path, shipped) if value not in (None, "")]
        if not given and text == "" and payload.get("name"):
            raise self._error(400, f"The file {safe_name(payload.get('name'))} is empty.",
                              "Choose a file with a sounding in it.")
        if len(given) != 1:
            raise self._error(400, "Give one sounding: a file, pasted text, or one of the soundings GM1 ships.",
                              "Choose a file, paste the sounding's text, or pick one from the list.")
        if shipped not in (None, ""):
            row = next((row for row in shipped_soundings() if row["id"] == shipped), None)
            if row is None:
                raise self._error(400, f"GM1 ships no sounding called {shipped!r}.", "Pick one from the list.")
            return Path(row["path"]), row["id"], str(row["observation"] or row["title"]), f"shipped:{row['id']}"
        if path not in (None, ""):
            if not isinstance(path, str):
                raise self._error(400, "The path must be text.", "Type the full path of the sounding file.")
            raw = path.strip().strip('"').strip("'")
            windows = WINDOWS_PATH.match(raw)
            if windows and not Path(raw).exists():
                from ..cm1run.gpuinfo import is_wsl2

                if not is_wsl2():
                    raise self._error(400, "That is a Windows path, and GM1 is not running under WSL here, so it "
                                           "cannot open it.",
                                      "Use Choose a file, which reads the file through your browser.")
                # D:\soundings\file.txt typed under WSL: the same file as Linux names it.
                raw = f"/mnt/{windows.group(1).lower()}/" + windows.group(2).replace("\\", "/")
            found = Path(raw).expanduser()
            if not found.is_absolute():
                found = self.api.root.path / found
            found = found.resolve()
            if found.is_dir():
                raise self._error(400, f"{found} is a folder.", "Give the path of the sounding file inside it.")
            if not found.is_file():
                raise self._error(400, f"There is no file at {found}.",
                                  "Check the path; it must be a file on the computer running GM1. Or use Choose a "
                                  "file, which reads the file through your browser.")
            if found.stat().st_size > MAX_BYTES:
                raise self._error(413, "That file is too large to be a sounding (over 4 MB).",
                                  "Choose the sounding's own text file.")
            stat = found.stat()
            return (found, safe_name(payload.get("name") or found.name), None,
                    f"path:{found}:{stat.st_size}:{stat.st_mtime_ns}")
        if not isinstance(text, str):
            raise self._error(400, "The sounding is not text.", "Choose a plain text file, or paste its text.")
        data = text.encode("utf-8")
        if len(data) > MAX_BYTES:
            raise self._error(413, "That is too large to be a sounding (over 4 MB).",
                              "Choose the sounding's own text file.")
        name = safe_name(payload.get("name") or "pasted-sounding.txt")
        digest = hashlib.sha256(data).hexdigest()
        target = self.base / "uploads" / digest[:24] / name
        if write:
            try:
                same = target.read_bytes() == data
            except OSError:
                same = False
            if not same:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{name}.partial")
                temporary.write_bytes(data)
                temporary.replace(target)
        return target, name, None, f"text:{digest}:{name}"

    def check(self, payload: dict[str, Any]) -> "Reply":
        from .api import Reply

        speed_unit = payload.get("speed_unit") or None
        if speed_unit not in (None, "kt", "ms"):
            raise self._error(400, "The wind speed unit is kt or ms.", "Pick knots or metres per second.")
        extend = bool(payload.get("extend"))
        source, name, shipped, identity = self._source(payload, write=not self.api.jobs.dry_active())
        storm = self._storm_namelist()
        key = _key("check", identity, speed_unit, extend, str(storm))
        out_dir = self.base / key
        out = out_dir / "input_sounding"
        args = C.sounding_convert_args(source, out, speed_unit=speed_unit, extend=extend, top_of=storm, name=name,
                                       preview_png=out_dir / "preview.png", native=self.api.native)
        self.api.jobs.stop_if_dry(args)
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in (out, out_dir / "sounding.json", out_dir / "preview.png"):
            stale.unlink(missing_ok=True)      # a file that no longer passes leaves nothing to run on
        result = self.api.jobs.host(args)
        body = result.json if isinstance(result.json, dict) else None
        if body is None or "problems" not in body:
            reason = (body or {}).get("reason") or (result.stderr.strip().splitlines() or
                                                    ["The check did not finish."])[-1]
            return Reply(200, {"ok": False, "checked": False, "command": result.command, "message": str(reason),
                               "fix": str((body or {}).get("remedy") or ""), "output": result.as_dict()})
        record = dict(body)
        record["checked"] = True
        record["command"] = result.command
        record["key"] = key
        record["shipped"] = shipped
        record["preset"] = GUIDED_PRESET
        preview = record.get("preview")
        if isinstance(preview, dict) and preview.get("file") and Path(preview["file"]).is_file():
            preview["url"] = self.file_url(Path(preview["file"]))
        if record.get("ok") and out.is_file():
            if shipped:
                # The record kept beside the file says it is one GM1 ships, so the next screen and Results do too.
                side = out_dir / "sounding.json"
                try:
                    kept = json.loads(side.read_text(encoding="utf-8"))
                    kept["shipped"] = shipped
                    side.write_text(json.dumps(kept, indent=1, default=str) + "\n", encoding="utf-8")
                except (OSError, ValueError):
                    pass
            record["sounding"] = str(out)
            record["download"] = self.file_url(out)
            start = (record.get("start") or {}).get("recommended") or "push"
            record["route"] = self.route(out, start)
            record["fit_command"] = C.display(C.preset_fit_args(
                GUIDED_PRESET, "10m", self.api.root.path,
                given=C.preset_given_args(sounding=out) + C.preset_start_args(start)))
            record["message"] = "Your sounding is ready. Look at it, pick how the storm starts, then go on."
        else:
            first = (record.get("problems") or [{}])[0]
            record["message"] = str(first.get("text") or "This file cannot run as it is.")
            record["fix"] = str(first.get("fix") or "")
        return Reply(200, record)

    @staticmethod
    def route(sounding: Path, start: str, budget: str = "10m") -> str:
        """Where the card goes next: "How long can it run?" for the storm on this sounding, started this way."""

        query = f"sounding={quote(str(sounding), safe='')}"
        if start == "strong-bubble":
            query += "&start=bubble&push=3"
        elif start in STARTS:
            query += f"&start={start}"
        return f"guided/fit/{GUIDED_PRESET}/{budget}?{query}"

    def options(self, payload: dict[str, Any]) -> "Reply":
        """The Customize panel's choices for the storm started another way (the push has no bubble size)."""

        from .api import Reply

        start = payload.get("start")
        if start not in STARTS:
            raise self._error(400, "start is push, bubble or strong-bubble.", "Pick one of the three starts.")
        preset = payload.get("preset") or GUIDED_PRESET
        if not isinstance(preset, str) or not NAME_RE.match(preset):
            raise self._error(400, "No storm was picked.", "Pick a storm from the gallery.")
        args = C.preset_options_args(preset, start)
        result = self.api.jobs.host(args)
        if result.rc != 0 or not isinstance(result.json, dict):
            reason = (result.stderr.strip().splitlines() or ["The settings could not be read."])[-1]
            raise self._error(400, reason, "Pick another start.", command=result.command)
        return Reply(200, {"ok": True, "command": result.command, "customize": result.json})


__all__ = ["GUIDED_PRESET", "MAX_BYTES", "STARTS", "SoundingApi", "load_copy", "safe_name", "shipped_soundings"]
