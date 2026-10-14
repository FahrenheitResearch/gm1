"""The GUI's JSON endpoints.

Reads are GETs; every action is a POST whose response carries ``command``,
the ``gpuwm-cm1`` line it ran.  Run ids are ``NAME`` or
``FOLDER/NAME`` under ``--root``; in a URL the id is one path segment, its
slash written ``%2F``.

GET  /api/session                  version, token, mode, root, runner, the API table
GET  /api/mode                     the stored view
POST /api/mode                     {"mode": "guided" | "advanced"}
GET  /api/runs                     one row per run directory under the root
GET  /api/runs/RUN                 run.json, status, which setup files exist
GET  /api/runs/RUN/status
GET  /api/runs/RUN/report          import-report.json
GET  /api/runs/RUN/results[?mode=] cm1out files, quick-looks, receipt, disk, copy lines
GET  /api/runs/RUN/files/PATH      one file under the run directory
GET  /api/runs/RUN/events          Server-Sent Events (served by :mod:`.server`)
POST /api/import                   {"name", "folder"?, "namelist": {"path"}|{"text"}, "sounding"?, ...}
POST /api/runs/RUN/{check,geometry,estimate,start,resume,stop,pause,force-stop}
POST /api/runs/RUN/{fields,slice,section,column}   one picture of one frame (gpuwm-cm1 quicklook)
POST /api/runs/RUN/{film,look}     a film of a look, a look drawn on request (:mod:`.looks`)
GET  /api/copy, /api/presets, /api/presets/ID/picture; POST /api/fit, /api/guided/start
                                   guided mode (:mod:`.guided`)
GET  /api/gpus, /api/system, /api/runs/RUN/live; POST /api/runs/RUN/{convert,reproduce}
                                   the Runs, System, Run and Results screens (:mod:`.runview`)

The setup screens' endpoints (examples, setup, namelist, edit, recommended,
acknowledge, sounding, jobscript, measure) are :mod:`.setup`'s; advanced
mode's (restarts, commands, words, presets/new) are :mod:`.advanced`'s.

Two housekeeping POSTs run no ``gpuwm-cm1`` command, so the API table (which lists those) does not have them;
each still needs the token header, and its reply's ``command`` is the shell line that does the same:

POST /api/runs/RUN/delete          {"keep_pictures"?: bool} removes the run folder (``rm -r``), or with
                                   keep_pictures only its big files (cm1out, each segment's frames and restarts);
                                   refused while the run is going or GM1 is still working on it
POST /api/open-folder              {"run"?: RUN} shows the runs folder, or one run's, in the file manager
                                   (File Explorer under WSL2, xdg-open on a Linux desktop)

Under WSL2 ``/api/session``, ``/api/runs`` and ``/api/runs/RUN/results`` also name each folder as Windows
sees it (``root_windows``, ``folder_windows``, ``cm1out_folder_windows``).

Every POST takes ``"dry_run": true``: the request is checked and its exact
command returned (``"dry_run": true`` in the reply), with nothing run and no
file changed; that is what "Show command" shows before a button is pressed.
A POST that fails before it has a command of its own names the action's
``command_template`` from the API table instead.

Errors say what happened and what to do, one line each: ``message`` and
``fix``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import threading
import time
from typing import Any, Callable, Iterator
from urllib.parse import quote, unquote

from . import commands as C
from . import control
from . import storm
from .files import (
    PathRefused, Root, is_run_dir, is_view_only, listing, read_json, require_name, safe_file,
    stage_upload, tree_bytes, within, write_json_atomic,
)
from .guided import Guided
from .jobs import COMMANDS_LOG, DryRun, Jobs, Unavailable
from .prefs import MODES, Prefs
from .runstate import (
    EventTail, events_path, read_status, segment_dirs, summarize, this_host,
)

NAMELIST_LIMIT = 1024 * 1024
SOUNDING_LIMIT = 16 * 1024 * 1024
#: The guided Results screen opens on these looks, in this order.
GUIDED_LOOKS = ("reflectivity", "spin-near-ground", "updraft-5km", "swath")
#: A hurricane run's favourite looks (the thunderstorm looks are not drawn for it; ``quicklook.products``).
HURRICANE_LOOKS = ("reflectivity", "wind-lowest", "pressure-lowest")
ESCALATE_AFTER_S = 30.0
#: In a run folder whose big files were deleted to free the disk (the pictures kept): what was removed.
TRIMMED_FILE = "trimmed.json"
#: Said after a delete under Windows (WSL2), where the Windows drive does not show the space at once.
WSL_FREED_NOTE = ("The space is free for new storms now. The Windows drive shows it only after the Linux disk "
                  "file is made smaller (it does not shrink by itself).")
#: The locks a run folder's own processes hold while they work (cm1run.rundir, cm1out.follow_lock), and what
#: each one is doing.  They are held by whichever process started them: a run and its picture workers started
#: before this GUI was opened are not this server's processes, and only their locks say they are still going.
WORK_LOCKS = ((".run.lock", "the storm is still running"),
              (".cm1out-follow.lock", "its files are still being written"),
              (".quicklook-follow.lock", "its pictures are still being drawn"))


def work_in_progress(rundir: Path) -> list[str]:
    """What is still working in ``rundir``, by the locks held there (empty: nothing is)."""

    try:
        import fcntl
    except ImportError:  # no flock on this system: the status and this server's own processes are the guard
        return []
    busy = []
    for name, what in WORK_LOCKS:
        try:
            descriptor = os.open(rundir / name, os.O_RDONLY)
        except OSError:
            continue  # never made: nothing ever held it
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            busy.append(what)
        finally:
            os.close(descriptor)
    return busy
FORCE_OFFER_AFTER_S = 60.0
#: How often a resume's sidecars look for the engine's new segment before they start.
SIDECAR_WAIT_POLL_S = 0.2
POLL_S = 0.25
STATUS_EVERY_S = 1.0
KEEPALIVE_S = 15.0


@dataclass
class Reply:
    status: int = 200
    body: Any = None
    file: Path | None = None
    headers: dict[str, str] = field(default_factory=dict)
    #: The file comes from a run directory: data, served under the rules of files.run_file_type.
    run_file: bool = False
    #: The name a downloaded file is saved as (default: its own name).
    download_name: str | None = None


@dataclass
class _StopRequest:
    """One Stop pressed on one run: the segment (and process) it was aimed at.

    The escalation to SIGTERM acts only on that same run.  A later start or
    resume from this server cancels it, and a run started some other way
    writes another segment (segments are never reused), so it is never
    signalled by it.  A Stop pressed before the run made its segment is tied
    to the process this server launched instead.
    """

    at: float
    segment: str | None
    engine: Any = None                 # the Launched engine, when this server started the run
    timer: threading.Timer | None = None

    def aimed_at(self, status: dict[str, Any]) -> bool:
        if self.segment is not None and status.get("segment") != self.segment:
            return False
        return self.engine is None or self.engine.alive



#: GrADS descriptors cm1out writes beside the history (cm1out.grads): what each one opens.
GRADS_COMPANIONS = {"cm1out_stats.ctl": "GrADS, statistics", "cm1out_parcel.ctl": "GrADS, parcels"}


def results_copy_lines(cm1out: Path, names: list[str]) -> list[dict[str, str]]:
    """Ready-to-copy lines for the cm1out files: ncview and xarray for the history in netCDF (one file per time,
    or cm1out.nc with every time), GrADS for each history descriptor, the statistics and the parcels."""

    copy: list[dict[str, str]] = []
    history = [name for name in names if re.match(r"cm1out_\d+\.nc$", name)]
    if history:
        pattern = str(cm1out / "cm1out_0*.nc")
        copy.append({"label": "ncview", "text": f"ncview {pattern}"})
        copy.append({"label": "Python (xarray)",
                     "text": f"import xarray as xr; ds = xr.open_mfdataset('{pattern}', combine='by_coords')"})
    elif "cm1out.nc" in names:
        single = str(cm1out / "cm1out.nc")
        copy.append({"label": "ncview", "text": f"ncview {single}"})
        copy.append({"label": "Python (xarray)", "text": f"import xarray as xr; ds = xr.open_dataset('{single}')"})
    for name in sorted(name for name in names if name.endswith(".ctl")):
        if name == "cm1out_metadata.ctl":
            continue  # the grid's own descriptor, which the history descriptors do not need
        copy.append({"label": GRADS_COMPANIONS.get(name, "GrADS"), "text": f"open {cm1out / name}"})
    return copy

class ApiError(Exception):
    def __init__(self, status: int, message: str, fix: str = "", **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.fix = fix
        self.extra = extra

    def reply(self) -> Reply:
        return Reply(self.status, {"ok": False, "message": self.message, "fix": self.fix, **self.extra})


def run_url(run_id: str) -> str:
    return quote(run_id, safe="")


class Api:
    def __init__(self, root: Root, jobs: Jobs, prefs: Prefs, *, token: str, port: int,
                 version: str, bind: str, native: dict | None = None) -> None:
        #: A source checkout's native tools ({"library", "publisher"}), passed on every command that opens
        #: the GPU (``gpuwm-cm1 gui --native-dir``); None for an installed wheel, which has its own.
        self.native = native
        self.root = root
        self.jobs = jobs
        self.prefs = prefs
        self.token = token
        self.port = port
        self.version = version
        self.bind = bind
        self._stops: dict[str, _StopRequest] = {}
        self._lock = threading.Lock()
        self._place: dict[str, Any] | None = None
        self._closing = threading.Event()
        from .advanced import AdvancedApi
        from .setup import SetupApi

        self.setup = SetupApi(self)
        self.guided = Guided(self)
        from .runview import RunViews

        self.views = RunViews(self)
        self.advanced = AdvancedApi(self)
        from .looks import LooksApi

        self.looks = LooksApi(self)
        from .wrf import WrfApi

        self.wrf = WrfApi(self)
        from .sounding import SoundingApi

        self.sounding = SoundingApi(self)
        from .chain import ChainApi

        self.chain = ChainApi(self)

    # ------------------------------------------------------------ routing

    def handle(self, method: str, raw_path: str, query: dict[str, list[str]], body: bytes,
               content_type: str = "") -> Reply:
        segments = [unquote(part) for part in raw_path.split("/") if part]
        if method == "POST" and segments[:1] == ["api"]:
            return self._with_template(segments[1:], self._handle(method, segments, query, body, content_type))
        return self._handle(method, segments, query, body, content_type)

    def _handle(self, method: str, segments: list[str], query: dict[str, list[str]], body: bytes,
                content_type: str) -> Reply:
        try:
            if segments[:1] != ["api"]:
                raise ApiError(404, "No such page.")
            if method == "POST":
                payload = self._json_body(body, content_type)
                dry = payload.pop(C.DRY_RUN_KEY, False)
                if not isinstance(dry, bool):
                    raise ApiError(400, "dry_run must be true or false.", "Send dry_run: true to see the command only.")
                if not dry:
                    return self._post(segments[1:], payload)
                with self.jobs.dry_run():
                    try:
                        return self._post(segments[1:], payload)
                    except DryRun as shown:
                        return self._dry_reply(segments[1:], shown)
            if method in ("GET", "HEAD"):
                return self._get(segments[1:], query)
            raise ApiError(405, f"{method} is not used here.")
        except ApiError as error:
            return error.reply()
        except PathRefused as error:
            return Reply(400, {"ok": False, "message": str(error), "fix": "Use a plain name."})
        except FileNotFoundError as error:
            return Reply(404, {"ok": False, "message": str(error) or "Not found.", "fix": ""})
        except Unavailable as error:
            return Reply(501, {"ok": False, "message": str(error),
                               "fix": "Update gpuwm-cm1; this build does not have it."})

    @staticmethod
    def _dry_reply(parts: list[str], shown: DryRun) -> Reply:
        endpoint = C.endpoint_for(parts)
        run_id = parts[1] if len(parts) == 3 and parts[0] == "runs" else None
        return Reply(200, {
            "ok": True, "dry_run": True, "action": None if endpoint is None else endpoint.action, "run": run_id,
            "command": shown.command, **shown.extra,
            "message": "Nothing ran. This is the command the button runs.", "fix": "",
        })

    @staticmethod
    def _with_template(parts: list[str], reply: Reply) -> Reply:
        """Every POST reply names its action's command: the line it ran, or else the API table's template."""

        endpoint = C.endpoint_for(parts)
        if endpoint is not None and isinstance(reply.body, dict):
            reply.body.setdefault("action", endpoint.action)
            if not reply.body.get("command"):
                reply.body["command_template"] = endpoint.command
        return reply

    @staticmethod
    def _json_body(body: bytes, content_type: str) -> dict[str, Any]:
        if body and not content_type.split(";")[0].strip().lower() == "application/json":
            raise ApiError(415, "The request body must be JSON.", "Send Content-Type: application/json.")
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ApiError(400, f"The request body is not JSON ({error}).", "Send a JSON object.") from error
        if not isinstance(payload, dict):
            raise ApiError(400, "The request body must be a JSON object.", "Send a JSON object.")
        return payload

    def _get(self, parts: list[str], query: dict[str, list[str]]) -> Reply:
        if parts == ["session"]:
            return Reply(body=self.session())
        if parts == ["mode"]:
            return Reply(body=self.mode())
        if parts == ["api"]:
            return Reply(body={"endpoints": C.api_table()})
        if parts == ["copy"]:
            return Reply(body=self.guided.copy())
        if parts == ["presets"]:
            return Reply(body=self.guided.presets(refresh=(query.get("refresh") or [""])[0] == "1"))
        if len(parts) == 3 and parts[0] == "presets" and parts[2] == "picture":
            return Reply(file=self.guided.picture(parts[1]))
        if parts == ["runs"]:
            return Reply(body=self.runs())
        if len(parts) >= 2 and parts[0] == "runs":
            run_id, rest = parts[1], parts[2:]
            rundir = self.root.existing_run(run_id)
            if not rest:
                return Reply(body=self.run_detail(run_id, rundir))
            if rest == ["status"]:
                return Reply(body=self.status(run_id, rundir))
            if rest == ["report"]:
                return Reply(body=self.report(rundir))
            if rest == ["results"]:
                mode = (query.get("mode") or [""])[0]
                mode = mode if mode in MODES else self.mode()["mode"]
                return Reply(body=self.results(run_id, rundir, mode))
            if rest[0] == "files" and len(rest) > 1:
                path = safe_file(rundir, "/".join(rest[1:]))
                return Reply(file=path, run_file=True)
        found = self.setup.get(parts, query)
        if found is None:
            found = self.advanced.get(parts, query)
        if found is not None:
            return found
        found = self.views.get(parts, query)
        if found is not None:
            return found
        found = self.looks.get(parts, query)
        if found is not None:
            return found
        found = self.wrf.get(parts, query)
        if found is not None:
            return found
        found = self.sounding.get(parts, query)
        if found is not None:
            return found
        found = self.chain.get(parts, query)
        if found is not None:
            return found
        raise ApiError(404, "No such endpoint.")

    def _post(self, parts: list[str], payload: dict[str, Any]) -> Reply:
        if parts == ["mode"]:
            return Reply(body=self.set_mode(payload))
        if parts == ["open-folder"]:
            return self.open_folder(payload)
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "delete":
            return self.delete_run(parts[1], self.root.existing_run(parts[1]), payload)
        if parts == ["import"]:
            return self.import_namelist(payload)
        if parts == ["fit"]:
            return Reply(body=self.guided.fit(payload))
        if parts == ["guided", "start"]:
            return self.guided.start(payload)
        if parts == ["presets", "new"]:
            return self.advanced.preset_new(payload)
        found = self.wrf.post(parts, payload)
        if found is not None:
            return found
        found = self.sounding.post(parts, payload)
        if found is not None:
            return found
        found = self.chain.post(parts, payload)
        if found is not None:
            return found
        if len(parts) == 3 and parts[0] == "runs":
            run_id, action = parts[1], parts[2]
            rundir = self.root.existing_run(run_id)
            pictures = {"fields": self.fields, "slice": self.slice, "section": self.section, "column": self.column}
            if action in pictures:
                # Pictures of the frames change nothing in the run, so a view-only folder has them too.
                return pictures[action](run_id, rundir, payload)
            found = self.looks.post(run_id, rundir, action, payload)
            if found is not None:
                return found
            if is_view_only(rundir):
                raise ApiError(409, "This folder is a plain cartesian output folder; it opens view-only.",
                               "Import its namelist to make a run folder.")
            handler: Callable[[str, Path, dict[str, Any]], Reply] | None = {
                "check": self.check,
                "geometry": self.geometry,
                "estimate": self.estimate,
                "start": self.start,
                "resume": self.resume,
                "stop": self.stop,
                "pause": self.pause,
                "force-stop": self.force_stop,
            }.get(action)
            if handler is not None:
                return handler(run_id, rundir, payload)
            found = self.setup.post(run_id, rundir, action, payload)
            if found is not None:
                return found
            found = self.views.post(run_id, rundir, action, payload)
            if found is not None:
                return found
        raise ApiError(404, "No such action.")

    # ------------------------------------------------------------ reads

    def session(self) -> dict[str, Any]:
        mode = self.mode()
        return {
            "version": self.version,
            "token": self.token,
            "token_header": "X-GM1-Token",
            "mode": mode["mode"],
            "mode_source": mode["source"],
            "root": str(self.root.path),
            "runner": self.jobs.runner.kind,
            "host": this_host(),
            "bind": self.bind,
            "port": self.port,
            "commands_available": sorted(self.jobs.runner.available()),
            "endpoints": C.api_table(),
            **self.place(),
        }

    # ------------------------------------------------------------ where the runs are, as Windows sees it

    def place(self) -> dict[str, Any]:
        """``wsl``, ``root_windows`` (the runs folder's Windows name under WSL2, else None) and
        ``can_open_folder`` (this computer has a file manager to show a folder in).  Worked out once."""

        with self._lock:
            place = self._place
        if place is None:
            from ..cm1run.gpuinfo import is_wsl2
            from ..cm1run.wslpaths import windows_path
            from .browser import folder_opener

            wsl = is_wsl2()
            place = {"wsl": wsl, "root_windows": windows_path(self.root.path, wsl=wsl) if wsl else None,
                     "can_open_folder": folder_opener(wsl=wsl) is not None}
            with self._lock:
                self._place = place
        return dict(place)

    def windows_name(self, path: Path) -> str | None:
        """What Windows calls ``path`` (a folder under the runs folder), or None when not under WSL2."""

        base = self.place()["root_windows"]
        if not base:
            return None
        try:
            parts = Path(path).resolve().relative_to(self.root.path).parts
        except (ValueError, OSError):
            return None
        return "\\".join([base.rstrip("\\"), *parts])

    def mode(self) -> dict[str, Any]:
        mode, source = self.prefs.mode(self.root.has_runs())
        return {"mode": mode, "source": source, "modes": list(MODES)}

    def set_mode(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("mode")
        if mode not in MODES:
            raise ApiError(400, f"There is no {mode!r} mode.", "Use guided or advanced.")
        self.jobs.stop_if_dry(C.mode_args(mode))
        self.prefs.set_mode(mode)
        return {"ok": True, "mode": mode, "command": C.display(C.mode_args(mode)),
                "message": f"{mode.capitalize()} mode. Runs are not changed."}

    def runs(self) -> dict[str, Any]:
        rows = []
        host = this_host()
        for path in self.root.iter_runs():
            run_id = self.root.run_id(path)
            status = read_status(path, host=host, launched_alive=self.jobs.engine_alive(run_id))
            row = summarize(path, status, run_id)
            row["url"] = run_url(run_id)
            row.update(self.views.row_extra(path, row))
            row["trimmed"] = (path / TRIMMED_FILE).is_file()
            rows.append(row)
        place = self.place()
        return {"root": str(self.root.path), "runs": rows, "host": host,
                "total_bytes": sum(int(row.get("disk_bytes") or 0) for row in rows),
                "root_windows": place["root_windows"], "can_open_folder": place["can_open_folder"]}

    def status(self, run_id: str, rundir: Path) -> dict[str, Any]:
        status = read_status(rundir, launched_alive=self.jobs.engine_alive(run_id)).as_dict()
        if (rundir / TRIMMED_FILE).is_file():
            # Its restarts were deleted to free the disk: there is nothing to continue from.
            status["trimmed"] = True
            status["actions"] = [action for action in status.get("actions") or [] if action != "resume"]
        with self._lock:
            request = self._stops.get(run_id)
        if request is not None and status["state"] in ("running", "waiting") and request.aimed_at(status):
            waited = time.time() - request.at
            status["stop_requested_s_ago"] = round(waited, 1)
            status["force_stop_offered"] = bool(status.get("local")) and waited >= FORCE_OFFER_AFTER_S
        return status

    def run_detail(self, run_id: str, rundir: Path) -> dict[str, Any]:
        files = {name: (rundir / name).is_file() for name in (
            "namelist.input", "namelist.input.original", "input_sounding", "import-report.json",
            "import-report.txt", "settings.toml", "deck.toml", "estimate.json", "geometry.json",
            "fit.json", COMMANDS_LOG)}
        return {
            "id": run_id,
            "url": run_url(run_id),
            "path": str(rundir),
            "run": read_json(rundir / "run.json", default=None),
            "status": self.status(run_id, rundir),
            "files": files,
            "segments": [path.name for path in segment_dirs(rundir)],
        }

    def report(self, rundir: Path) -> dict[str, Any]:
        report = read_json(rundir / "import-report.json", default=None)
        if report is None:
            raise ApiError(404, "This run has no import report.",
                           "Runs made with gpuwm-cm1 import have one; this folder was made another way.")
        text_path = rundir / "import-report.txt"
        text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.is_file() else None
        return {"report": report, "text": text}

    def results(self, run_id: str, rundir: Path, mode: str) -> dict[str, Any]:
        base = f"/api/runs/{run_url(run_id)}/files/"
        cm1out = rundir / "cm1out"
        files = listing(cm1out)
        for row in files:
            row["url"] = base + "cm1out/" + quote(row["name"])
        looks = []
        index = read_json(rundir / "quicklook" / "index.json", default=None)
        # A picture with no storm in it yet says so beside it (:mod:`.storm`).
        notes = storm.run_notes(rundir, index)
        # Guided mode's words, to tell a line said beside a storm from one said in place of it.
        strings = (self.guided.copy()["guided"].get("strings") or {}) if notes else {}
        if isinstance(index, dict) and isinstance(index.get("products"), list):
            for product in index["products"]:
                if not isinstance(product, dict):
                    continue
                frames = []
                for frame in product.get("frames") or []:
                    if isinstance(frame, dict) and isinstance(frame.get("file"), str):
                        row = {**frame, "url": base + "quicklook/" + quote(frame["file"])}
                        if isinstance(frame.get("thumb"), str):
                            row["thumb_url"] = base + "quicklook/" + quote(frame["thumb"])
                        # A look drawn once per run (the environment, the parcels) is not a moment of the storm.
                        line = None if product.get("once") else storm.note_for(notes, frame.get("time_s"))
                        if line:
                            row["storm"] = line
                            if storm.shows_storm(line, strings):
                                row["storm_showing"] = True
                        frames.append(row)
                looks.append({**{k: v for k, v in product.items() if k != "frames"}, "frames": frames})
        series = None
        if isinstance(index, dict) and isinstance(index.get("series"), dict) and index["series"].get("file"):
            series = {**index["series"], "url": base + "quicklook/" + quote(str(index["series"]["file"]))}
            if isinstance(index["series"].get("picture"), str):
                series["picture_url"] = base + "quicklook/" + quote(index["series"]["picture"])
        if mode == "guided":
            shown = index.get("run") if isinstance(index, dict) and isinstance(index.get("run"), dict) else {}
            favourites = HURRICANE_LOOKS if shown.get("storm_kind") == "hurricane" else GUIDED_LOOKS
            order = {name: position for position, name in enumerate(favourites)}
            looks.sort(key=lambda look: order.get(str(look.get("id")), len(order)))
            for look in looks:
                look["favourite"] = str(look.get("id")) in order
        segments = segment_dirs(rundir)
        receipt = read_json(segments[-1] / "receipt.json", default=None) if segments else None
        copy = results_copy_lines(cm1out, [row["name"] for row in files])
        disk = {
            "frames_bytes": sum(tree_bytes(path / "frames") for path in segments),
            "restarts_bytes": sum(tree_bytes(path / "restarts") for path in segments),
            "cm1out_bytes": tree_bytes(cm1out),
            "quicklook_bytes": tree_bytes(rundir / "quicklook"),
        }
        disk["total_bytes"] = tree_bytes(rundir)
        status = self.status(run_id, rundir)
        body = {
            "id": run_id,
            "mode": mode,
            "status": status,
            "cm1out": files,
            "cm1out_folder": str(cm1out),
            "cm1out_folder_windows": self.windows_name(cm1out),
            "folder": str(rundir),
            "folder_windows": self.windows_name(rundir),
            "can_open_folder": self.place()["can_open_folder"],
            "trimmed": read_json(rundir / TRIMMED_FILE, default=None),
            "looks": looks,
            "series": series,
            "quicklook_run": index.get("run") if isinstance(index, dict) else None,
            "copy": copy,
            "receipt": receipt,
            "disk": disk,
            "commands": {
                "resume": C.display(C.resume_args(rundir, native=self.native)),
                "convert": C.display(["cm1out", str(rundir)]),
                "show_everything": C.display(C.mode_args("advanced")),
            },
        }
        if mode == "guided":
            body["guided"] = self.guided.results_extra(rundir, index if isinstance(index, dict) else None, status,
                                                       looks)
        return self.views.results_extra(run_id, rundir, body)

    # ------------------------------------------------------------ housekeeping: delete, open the folder

    def delete_run(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:  # noqa: C901
        """Delete a run's folder, or (``keep_pictures``) only its big files.  Never while it is going."""

        keep = payload.get("keep_pictures", False)
        if not isinstance(keep, bool):
            raise ApiError(400, "keep_pictures must be true or false.", "Leave it out to delete the whole run.")
        segments = segment_dirs(rundir)
        big = [rundir / "cm1out", *[segment / name for segment in segments for name in ("frames", "restarts")]]
        # A big folder that is a link, or sits behind one, keeps its files somewhere else (another disk):
        # GM1 deletes nothing through a link, so "pictures only" could not be true of this run.
        linked = [path for path in big if path.is_symlink() or (path.exists() and not within(rundir, path))]
        targets = [rundir] if not keep else [path for path in big if path.is_dir() and path not in linked]
        line = "rm -r -- " + " ".join(shlex.quote(str(path)) for path in targets) if targets else ""
        # A refusal carries no command: nothing was run, and the page lists the commands it ran.
        extra = {"action": "delete"}
        if is_view_only(rundir):
            raise ApiError(409, "This folder was not made by a GM1 run command, so GM1 does not delete it.",
                           f"Delete it yourself: {rundir}", **extra)
        if rundir.is_symlink() or rundir.resolve() == self.root.path:
            raise ApiError(409, "This run folder is a link, so GM1 does not delete it.",
                           f"Delete it yourself: {rundir}", **extra)
        state = self.status(run_id, rundir).get("state")
        if state in ("running", "waiting"):
            raise ApiError(409, "This storm is still running.", "Stop it first, then delete it.", **extra)
        if state == "stale":
            raise ApiError(409, "This run may still be going on another computer.",
                           "Delete it once it has ended.", **extra)
        if self.jobs.launched(run_id):
            raise ApiError(409, "GM1 is still working on this run (drawing its pictures or converting its files).",
                           "Wait a moment and try again.", **extra)
        busy = work_in_progress(rundir)
        if busy:
            # Deleting the files under a worker would lose the pictures it has not drawn yet.
            raise ApiError(409, f"GM1 is still working on this run: {', '.join(busy)}.",
                           "Wait until it is done (Watching says when the pictures are ready), then try again.",
                           **extra)
        if keep and linked:
            name = str(linked[0].relative_to(rundir))
            raise ApiError(409, f"{name} in this run is a link to another place, so GM1 does not delete it, and "
                                "nothing was deleted.",
                           f"Delete what it points to yourself, or delete the whole run. The link is {linked[0]}",
                           **extra)
        if keep and not targets:
            raise ApiError(409, "This run has no big files left to delete.", "", **extra)
        self.jobs.stop_if_dry(line, action="delete", run=run_id)
        size = sum(tree_bytes(path) for path in targets)
        with self.jobs.run_lock(run_id):
            if self.jobs.launched(run_id) or work_in_progress(rundir):
                raise ApiError(409, "GM1 has just started working on this run.", "Wait a moment and try again.",
                               **extra)
            failed = []
            for path in targets:
                if path.is_symlink() or (path != rundir and not within(rundir, path)):
                    continue  # became a link since it was looked at: left alone, and reported below
                shutil.rmtree(path, onerror=lambda _call, name, _error: failed.append(str(name)))
            left = rundir.exists() if not keep else any(path.exists() or path.is_symlink() for path in targets)
            if keep and not left:
                # The run is marked "pictures only" (no Continue, no new pictures) only when that is true of it.
                try:
                    write_json_atomic(rundir / TRIMMED_FILE, {
                        "schema": "gm1.gui-trimmed/v1", "freed_bytes": size,
                        "removed": [str(path.relative_to(rundir)) for path in targets]})
                except OSError:
                    pass
            if keep:
                self._log(rundir, "delete", line, "the big files; the pictures are kept" if not left
                          else "the big files; some could not be deleted")
        self.views.forget_disk(rundir)
        if left:
            raise ApiError(500, "Some of it could not be deleted.",
                           f"Delete it yourself: {failed[0] if failed else rundir}", **extra)
        name = rundir.name
        message = (f"Deleted the big files of {name}; its pictures are kept." if keep else f"Deleted {name}.")
        # Under Windows the Linux disk is one file on a Windows drive, and that file does not shrink by itself.
        note = WSL_FREED_NOTE if self.place()["wsl"] else ""
        return Reply(200, {"ok": True, "action": "delete", "run": run_id, "command": line, "freed_bytes": size,
                           "kept_pictures": keep, "message": message, "note": note, "fix": ""})

    def open_folder(self, payload: dict[str, Any]) -> Reply:
        """Show the runs folder, or one run's folder, in this computer's file manager."""

        from . import browser

        run_id = payload.get("run")
        if run_id is not None and not isinstance(run_id, str):
            raise ApiError(400, "run must be a run's name.", "Leave it out for the runs folder.")
        folder = self.root.existing_run(run_id) if run_id else self.root.path
        place = self.place()
        windows = self.windows_name(folder) if place["wsl"] else None
        found = browser.folder_opener(wsl=place["wsl"])
        shown = windows or str(folder)
        line = "" if found is None else shlex.join([Path(found[1]).name, shown])
        extra = {"action": "open-folder", "path": str(folder), "windows_path": windows}
        if found is None:
            raise ApiError(409, "This computer has no screen of its own to show a folder on.",
                           f"The folder is {shown}", **extra)
        self.jobs.stop_if_dry(line, run=run_id, **extra)
        started, how = browser.open_folder(folder, windows=windows, wsl=place["wsl"])
        if not started:
            raise ApiError(500, f"The folder was not opened ({how}).", f"Open it yourself: {shown}", **extra)
        return Reply(200, {"ok": True, "run": run_id, "command": line, "message": f"Opened {shown}", "fix": "",
                           **extra})

    # ------------------------------------------------------------ actions

    def _answer(self, action: str, run_id: str | None, result, ok: bool, message: str,
                fix: str = "", **extra: Any) -> Reply:
        body = {
            "ok": ok,
            "action": action,
            "run": run_id,
            "command": result.command,
            "rc": result.rc,
            "result": result.json,
            "message": message,
            "fix": fix,
            "output": result.as_dict(),
            **extra,
        }
        return Reply(200, body)

    def _log(self, rundir: Path, action: str, line: str, note: str = "") -> None:
        self.jobs.log_command(rundir, action, line, note)

    def import_namelist(self, payload: dict[str, Any]) -> Reply:
        name = require_name(str(payload.get("name") or ""), "run name")
        folder = str(payload.get("folder") or "").strip("/")
        run_id = f"{folder}/{name}" if folder else name
        rundir = self.root.run_path(run_id)
        if rundir.exists():
            raise ApiError(409, f"A run called {run_id!r} already exists.", "Pick another name.")
        if folder:
            parent = rundir.parent
            if is_run_dir(parent):
                raise ApiError(409, f"{folder!r} is a run folder; a run cannot go inside another run.",
                               "Pick another folder, or leave the folder empty.")
            if parent.exists() and not parent.is_dir():
                raise ApiError(409, f"{folder!r} is a file in the runs folder, not a folder.",
                               "Pick another folder name.")
        namelist = self._input_file(payload.get("namelist"), "namelist.input", NAMELIST_LIMIT,
                                    "namelist", required=True)
        sounding = self._input_file(payload.get("sounding"), "input_sounding", SOUNDING_LIMIT,
                                    "sounding", required=False)
        shown_mode = self.mode()
        shown = shown_mode["mode"]
        args = C.import_args(namelist, rundir, sounding, bool(payload.get("acknowledge_output_keys")))
        self.jobs.stop_if_dry(args)
        rundir.parent.mkdir(parents=True, exist_ok=True)
        result = self.jobs.host(args)
        self._log(rundir, "import", result.command, f"rc {result.rc}")
        verdicts = {
            0: ("runnable", "Imported. GM1 runs it.", ""),
            3: ("needs-ok", "Imported. It runs once you OK the settings GM1 does not read.",
                "Open Check, read the list, and tick the box."),
            2: ("refused", "Imported, but GM1 cannot run it as written.",
                "Open Check: each refused setting says what to change."),
        }
        verdict, message, fix = verdicts.get(result.rc, (
            "error", "The import did not finish.", "Read the output below; it names the problem."))
        notice = None
        # One of GM1's own examples picked from the guided gallery is not "importing a
        # namelist": the switch to advanced is for users who bring their own.
        if result.rc in (0, 2, 3) and not payload.get("example"):
            notice = self.prefs.switch_on_import(shown)
        elif payload.get("example") and shown_mode["source"] != "stored":
            # The first run made from the guided gallery keeps the view guided: with a
            # run present and nothing stored, the view would otherwise turn advanced.
            self.prefs.set_mode(shown)
        created = rundir.is_dir()
        return self._answer("import", run_id if created else None, result, result.rc in (0, 3),
                            message, fix, verdict=verdict, url=run_url(run_id) if created else None,
                            mode=self.mode()["mode"], mode_notice=notice)

    def _input_file(self, spec: Any, filename: str, limit: int, what: str,
                    required: bool) -> Path | None:
        """A namelist or sounding given by path, or uploaded as text (staged by its content's SHA-256)."""

        if spec in (None, "", {}):
            if required:
                raise ApiError(400, f"No {what} was given.", f"Give the {what}'s path or upload it.")
            return None
        if not isinstance(spec, dict):
            raise ApiError(400, f"The {what} must be {{'path': ...}} or {{'text': ...}}.", "")
        if "text" in spec:
            text = spec["text"]
            if not isinstance(text, str):
                raise ApiError(400, f"The uploaded {what} is not text.", "Upload the plain text file.")
            data = text.encode("utf-8")
            if len(data) > limit:
                raise ApiError(413, f"The uploaded {what} is larger than {limit // 1024} KB.",
                               f"Check that this is a CM1 {what}.")
            return stage_upload(self.root.server_dir, filename, data, write=not self.jobs.dry_active())
        raw = spec.get("path")
        if not isinstance(raw, str) or not raw.strip():
            raise ApiError(400, f"The {what} path is empty.", "Type the full path to the file.")
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            path = (self.root.path / path)
        path = path.resolve()
        if not path.is_file():
            raise ApiError(400, f"There is no file at {path}.", "Check the path; it must be a file on this computer.")
        if path.stat().st_size > limit:
            raise ApiError(413, f"{path} is larger than {limit // 1024} KB.", f"Check that this is a CM1 {what}.")
        return path

    def _deck(self, rundir: Path, action: str) -> Path:
        deck = rundir / "deck.toml"
        if not deck.is_file():
            args = C.check_args(deck) if action == "check" else C.plan_args(deck, rundir / "plan-receipt.json")
            raise ApiError(409, "This run has no deck.toml yet.",
                           "Import its namelist again; the import writes deck.toml when GM1 can run it.",
                           command=C.display(args))
        return deck

    def check(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        result = self.jobs.host(C.check_args(self._deck(rundir, "check")))
        self._log(rundir, "check", result.command, f"rc {result.rc}")
        ok = result.rc == 0
        return self._answer("check", run_id, result, ok,
                            "The settings pass every check." if ok else "The settings did not pass.",
                            "" if ok else "Read the reason below; it names the setting and the fix.")

    def geometry(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        sounding = rundir / "input_sounding"
        args = C.plan_args(self._deck(rundir, "geometry"), rundir / "plan-receipt.json",
                           sounding if sounding.is_file() else None)
        result = self.jobs.host(args)
        self._log(rundir, "geometry", result.command, f"rc {result.rc}")
        plan = result.json if isinstance(result.json, dict) else {}
        ok = result.rc == 0 and bool(plan)
        return self._answer("geometry", run_id, result, ok,
                            "Priced the run on this computer; no GPU was used." if ok else
                            "Could not price the run.",
                            "" if ok else "Read the reason below.",
                            geometry=plan.get("run_geometry"))

    def estimate(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        gpu = payload.get("gpu")
        if gpu is not None and not isinstance(gpu, int):
            raise ApiError(400, "gpu must be a whole number.", "Use 0 for the first GPU.")
        result = self.jobs.host(C.estimate_args(rundir, gpu))
        self._log(rundir, "estimate", result.command, f"rc {result.rc}")
        ok = result.rc == 0
        return self._answer("estimate", run_id, result, ok,
                            "Estimated memory, disk and time." if ok else "The estimate did not finish.",
                            "" if ok else "Read the reason below.")

    # ------------------------------------------------------------ pictures of one frame

    _FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")

    @staticmethod
    def _number(payload: dict[str, Any], key: str, *, low: float, high: float, required: bool = False) -> float | None:
        value = payload.get(key)
        if value is None:
            if required:
                raise ApiError(400, f"{key} is missing.", f"Send {key} as a number.")
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (low <= float(value) <= high):
            raise ApiError(400, f"{key} must be a number from {low:g} to {high:g}.", f"Send {key} as a number.")
        return float(value)

    def _point(self, payload: dict[str, Any], key: str, count: int, *, required: bool) -> list[float] | None:
        value = payload.get(key)
        if value is None:
            if required:
                raise ApiError(400, f"{key} is missing.", f"Send {key} as [{', '.join(['km'] * count)}].")
            return None
        if not isinstance(value, list) or len(value) != count:
            raise ApiError(400, f"{key} must be a list of {count} numbers (km).", "Positions are in km, as xh and yh.")
        return [self._number({"v": v}, "v", low=-1.0e5, high=1.0e5, required=True) for v in value]

    def _field(self, payload: dict[str, Any]) -> str:
        field = payload.get("field")
        if not isinstance(field, str) or not self._FIELD.match(field):
            raise ApiError(400, "field must be a cm1out field name such as w, thpert or dbz.",
                           "Ask /fields for the fields this run has.")
        return field

    def _picture(self, action: str, run_id: str, rundir: Path, args: list[str], message: str) -> Reply:
        result = self.jobs.host(args)
        self._log(rundir, action, result.command, f"rc {result.rc}")
        record = result.json if isinstance(result.json, dict) else {}
        ok = result.rc == 0 and bool(record)
        extra: dict[str, Any] = {}
        if ok and isinstance(record.get("file"), str):
            extra["picture_url"] = f"/api/runs/{run_url(run_id)}/files/quicklook/" + quote(record["file"])
        fix = ""
        if not ok:
            try:
                error = json.loads(result.stderr)
                fix = str(error.get("reason") or error.get("remedy") or "")
            except (ValueError, AttributeError):
                fix = (result.stderr.strip().splitlines() or [""])[-1]
            fix = fix or "Read the output below; it names the problem."
        return self._answer(action, run_id, result, ok, message if ok else "The picture was not drawn.", fix, **extra)

    def fields(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        time_s = self._number(payload, "time_s", low=0.0, high=1.0e9)
        return self._picture("fields", run_id, rundir, C.fields_args(rundir, time_s),
                             "The cm1out fields a slice can show.")

    def slice(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        field = self._field(payload)
        level = payload.get("level")
        if level is not None and (isinstance(level, bool) or not isinstance(level, int) or level < 1):
            raise ApiError(400, "level must be a whole number from 1 (the lowest level, as in CM1).", "")
        args = C.slice_args(rundir, field, height_m=self._number(payload, "height_m", low=0.0, high=1.0e5),
                            level=level, line_km=self._point(payload, "line_km", 4, required=False),
                            time_s=self._number(payload, "time_s", low=0.0, high=1.0e9))
        return self._picture("slice", run_id, rundir, args, f"{field} on one level is drawn.")

    def section(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        field = self._field(payload)
        args = C.section_args(rundir, field, self._point(payload, "from_km", 2, required=True),
                              self._point(payload, "to_km", 2, required=True),
                              top_m=self._number(payload, "top_m", low=100.0, high=1.0e5),
                              time_s=self._number(payload, "time_s", low=0.0, high=1.0e9))
        return self._picture("section", run_id, rundir, args, f"{field} along the line A-B is drawn.")

    def column(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        args = C.column_args(rundir, self._point(payload, "at_km", 2, required=False),
                             self._number(payload, "time_s", low=0.0, high=1.0e9))
        return self._picture("column", run_id, rundir, args, "The skew-T of one column is drawn.")

    def _launch_engine(self, action: str, run_id: str, rundir: Path, args: list[str]) -> Reply:
        with self.jobs.run_lock(run_id):
            status = read_status(rundir, launched_alive=self.jobs.engine_alive(run_id))
            line = C.display(args)
            if status.state in ("running", "waiting", "stale"):
                raise ApiError(409, f"This run is already {status.label.lower()}.",
                               "Stop it first, or wait for it to finish.", command=line)
            if action == "start" and status.detail.get("import_exit_code") == 2:
                raise ApiError(409, "GM1 cannot run this namelist as written.",
                               "Open Check: each refused setting says what to change.", command=line)
            self.jobs.require(args[0])
            names = [name for name in C.SIDECARS if name in self.jobs.runner.available()]
            self.jobs.stop_if_dry(line, sidecars=[C.sidecar_line(C.sidecar_args(name, rundir)) for name in names])
            # A Stop pressed on the previous run must not reach this one.
            self._forget_stop(run_id)
            leftover = [name for name in ("stop", "pause") if (rundir / control.CONTROL_DIR / name).exists()]
            for name in leftover:
                (rundir / control.CONTROL_DIR / name).unlink()
            before = len(segment_dirs(rundir))
            engine = self.jobs.launch(run_id, rundir, args, role=args[0])
            # Each sidecar's arguments.  "Also write netCDF" needs no sidecar of its own:
            # cm1out reads settings.toml and writes netCDF beside the GrADS files in the
            # same pass, so a second writer would race it for the same files.
            chosen = [(name, C.sidecar_args(name, rundir)) for name in names]
            sidecars = [C.sidecar_line(args) for _, args in chosen]
            if before == 0:
                for role, args in chosen:
                    self.jobs.launch(run_id, rundir, args, role=role, engine=engine, segments_before=before)
            elif chosen:
                # A resume (or a start over) of a run that already has segments: a
                # sidecar started now would find the previous segment's end record
                # and exit at once (`--follow` ends with the newest segment), so the
                # sidecars start once the engine's own segment exists, as the job
                # script's do (cm1run.jobscript).
                self._launch_sidecars_after_segment(run_id, rundir, engine, before, chosen)
            lines = [*sidecars, engine.command] + (["wait"] if sidecars else [])
            self._log(rundir, action, "\n".join(lines),
                      "the run, with its sidecars on this computer's CPUs" if sidecars else "the run")
        deadline = time.monotonic() + 0.5
        while engine.alive and time.monotonic() < deadline:
            time.sleep(0.05)
        rc = engine.process.returncode
        failed = rc not in (None, 0)
        tail = ""
        if failed:
            # A run gone before it began leaves its sidecars nothing to follow.
            self.jobs.end_orphans(run_id)
            try:
                tail = engine.log.read_text(encoding="utf-8", errors="replace")[-4000:]
            except OSError:
                tail = ""
        body = {
            "ok": not failed,
            "action": action,
            "run": run_id,
            "command": engine.command,
            "sidecars": sidecars,
            "pid": engine.process.pid,
            "rc": rc,
            "message": ("Started. Closing the GUI does not stop it." if not failed else
                        "The run stopped as soon as it started."),
            "fix": "" if not failed else "Read the output below; it names the problem.",
            "output": {"log_tail": tail},
            "status": self.status(run_id, rundir),
        }
        return Reply(200, body)

    def _launch_sidecars_after_segment(self, run_id: str, rundir: Path, engine: Any, before: int,
                                       chosen: list[tuple[str, list[str]]]) -> None:
        def watch() -> None:
            while not self._closing.is_set() and engine.alive and len(segment_dirs(rundir)) <= before:
                self._closing.wait(SIDECAR_WAIT_POLL_S)
            if self._closing.is_set() or len(segment_dirs(rundir)) <= before:
                return  # the engine ended without a segment: nothing to follow
            for role, args in chosen:
                try:
                    self.jobs.launch(run_id, rundir, args, role=role, engine=engine, segments_before=before)
                except Exception:  # noqa: BLE001 - a sidecar never takes the server down
                    pass

        threading.Thread(target=watch, name=f"gm1-sidecars-{run_id}", daemon=True).start()

    def start(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        return self._launch_engine("start", run_id, rundir, C.run_args(rundir, self.native))

    def resume(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        restart = payload.get("from_restart")
        segment = payload.get("from_segment")
        until = payload.get("until_s")
        if restart is not None and (isinstance(restart, bool) or not isinstance(restart, int) or restart < 1):
            raise ApiError(400, "from_restart must be a restart number (1, 2, ...).", "")
        if segment is not None and (isinstance(segment, bool) or not isinstance(segment, int) or segment < 1):
            raise ApiError(400, "from_segment must be a segment number (1, 2, ...).", "")
        if segment is not None and restart is None:
            raise ApiError(400, "from_segment picks the branch of a restart, so it needs from_restart.",
                           "Send from_restart with it.")
        if until is not None and (isinstance(until, bool) or not isinstance(until, (int, float)) or until <= 0):
            raise ApiError(400, "until_s must be a storm time in seconds.", "")
        return self._launch_engine("resume", run_id, rundir,
                                   C.resume_args(rundir, restart, until, self.native, from_segment=segment))

    def _control_reply(self, action: str, run_id: str, rundir: Path, args: list[str],
                       result: control.ControlResult, **extra: Any) -> Reply:
        line = C.display(args)
        if result.wrote or result.signalled:
            self._log(rundir, action, line)
        body = {"ok": result.ok, "action": action, "run": run_id, "command": line,
                "rc": result.rc, "message": result.message, "fix": result.fix,
                "status": self.status(run_id, rundir), **extra}
        return Reply(200, body)

    def _live_engine(self, run_id: str):
        engines = [row for row in self.jobs.launched(run_id) if row.role in ("run", "resume")]
        return engines[-1] if engines else None

    def _forget_stop(self, run_id: str) -> None:
        with self._lock:
            request = self._stops.pop(run_id, None)
        if request is not None and request.timer is not None:
            request.timer.cancel()

    def stop(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        # A run this server started may not have written its first record yet; its
        # control directory is read from the first large step on, so the file is
        # written for it all the same.
        self.jobs.stop_if_dry(C.stop_args(rundir))
        engine = self._live_engine(run_id)
        result = control.request(rundir, "stop", assume_live=engine is not None,
                                 launched_at=None if engine is None else engine.started)
        escalation = None
        if result.wrote:
            status = read_status(rundir, launched_alive=engine is not None).as_dict()
            with self._lock:
                request = self._stops.get(run_id)
                if request is not None and not request.aimed_at(status):
                    stale, request = request, None
                    self._stops.pop(run_id, None)
                    if stale.timer is not None:
                        stale.timer.cancel()
                if request is None:
                    # The first Stop starts the clock; pressing it again does not restart it.
                    request = _StopRequest(time.time(), status.get("segment"), engine)
                    self._stops[run_id] = request
                    if status.get("local"):
                        request.timer = threading.Timer(ESCALATE_AFTER_S, self._escalate,
                                                        args=(run_id, rundir, request))
                        request.timer.daemon = True
                        request.timer.start()
            if request.timer is not None:
                escalation = C.display(C.stop_args(rundir, "terminate"))
        return self._control_reply("stop", run_id, rundir, C.stop_args(rundir), result,
                                   escalates_to=escalation, escalate_after_s=ESCALATE_AFTER_S if escalation else None,
                                   force_stop_after_s=FORCE_OFFER_AFTER_S if escalation else None)

    def _escalate(self, run_id: str, rundir: Path, request: _StopRequest) -> None:
        try:
            with self._lock:
                if self._stops.get(run_id) is not request:
                    return
            status = read_status(rundir)
            if status.state not in ("running", "waiting") or not status.detail.get("local"):
                return
            if not request.aimed_at(status.as_dict()):
                return
            result = control.terminate(rundir)
            if result.signalled:
                self._log(rundir, "stop", C.display(C.stop_args(rundir, "terminate")),
                          f"the run had not stopped {ESCALATE_AFTER_S:.0f} s after Stop")
        except Exception:  # noqa: BLE001 - a timer never takes the server down
            pass

    def pause(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        self.jobs.stop_if_dry(C.pause_args(rundir))
        engine = self._live_engine(run_id)
        return self._control_reply("pause", run_id, rundir, C.pause_args(rundir),
                                   control.request(rundir, "pause", assume_live=engine is not None,
                                                   launched_at=None if engine is None else engine.started))

    def force_stop(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> Reply:
        self.jobs.stop_if_dry(C.stop_args(rundir, "force"))
        return self._control_reply("force-stop", run_id, rundir, C.stop_args(rundir, "force"),
                                   control.force_stop(rundir))

    # ------------------------------------------------------------ events

    def follow(self, run_id: str, rundir: Path, since: tuple[int, int] | None,
               stopping: threading.Event) -> Iterator[tuple[str, str | None, Any]]:
        """(event, id, data) for the SSE stream; ("keepalive", None, None) now and then.

        Every segment's records in order, then the newest segment's as they
        are written, then the next segment's when a resume starts one.  A
        ``status`` event is sent whenever the derived status changes.
        """

        tails: dict[str, EventTail] = {}
        finished: set[str] = set()
        last_status: tuple[Any, ...] | None = None
        next_status = 0.0
        last_sent = time.monotonic()
        view_only = is_view_only(rundir)

        while not stopping.is_set():
            logs: list[tuple[str, Path | None, Path]] = [
                (path.name, path, path / "events.jsonl") for path in segment_dirs(rundir)]
            if view_only and not logs:
                single = events_path(rundir)
                if single is not None:
                    logs = [("", None, single)]
            for position, (name, segment, log) in enumerate(logs):
                if name in finished:
                    continue
                if name not in tails:
                    tails[name] = EventTail(log)
                    if segment is not None:
                        last_sent = time.monotonic()
                        yield ("segment", None, {"segment": name})
                newer_exists = position < len(logs) - 1
                while True:
                    batch = list(self._records(tails[name], segment, since))
                    for item in batch:
                        last_sent = time.monotonic()
                        yield item
                    # A segment with a newer one beside it has ended: drain it fully.
                    if not (newer_exists and tails[name].pending()):
                        break
                if newer_exists:
                    finished.add(name)
                    tails.pop(name, None)
            now = time.monotonic()
            if now >= next_status:
                next_status = now + STATUS_EVERY_S
                status = self.status(run_id, rundir)
                key = (status.get("state"), status.get("label"), status.get("percent"),
                       status.get("segment"), status.get("force_stop_offered"))
                if key != last_status:
                    last_status = key
                    last_sent = now
                    yield ("status", None, status)
            if now - last_sent >= KEEPALIVE_S:
                last_sent = now
                yield ("keepalive", None, None)
            stopping.wait(POLL_S)

    @staticmethod
    def _records(tail: EventTail, segment: Path | None, since: tuple[int, int] | None
                 ) -> Iterator[tuple[str, str | None, Any]]:
        index = int(segment.name) if segment is not None and segment.name.isdigit() else 0
        for record in tail.read_new():
            seq = record.get("seq")
            if not isinstance(seq, int):
                continue
            if since is not None and (index, seq) <= since:
                continue
            kind = str(record.get("kind") or "record")
            yield (kind, f"{index}:{seq}", {**record, "segment": None if segment is None else segment.name})

    def close(self) -> None:
        self._closing.set()
        with self._lock:
            requests = list(self._stops.values())
            self._stops.clear()
        for request in requests:
            if request.timer is not None:
                request.timer.cancel()


def parse_event_id(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    left, sep, right = value.strip().partition(":")
    if not sep:
        return None
    try:
        return int(left), int(right)
    except ValueError:
        return None


__all__ = ["Api", "ApiError", "GUIDED_LOOKS", "HURRICANE_LOOKS", "Reply", "parse_event_id", "run_url"]
