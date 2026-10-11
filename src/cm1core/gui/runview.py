"""What the Run, Results, Runs and System screens read (screens 1, 9 and 10).

Every answer comes from the run directory and ``nvidia-smi``, so a
run started from a terminal, a job script on a cluster or this GUI looks the
same ("attach", 4.4): the server only reads what the run writes.

``GET /api/runs/RUN/live``
    The Run screen: status, the admitted run, start-up stages, the latest
    progress with its rates and GPU telemetry, the last write, warnings in
    plain words, the latest quick-look of each product, the time series, the
    log's tail, the segments, and the command of every button.
``GET /api/gpus``
    The Runs screen's GPU strip: each card, its driver, memory used of total,
    and which GM1 runs are on it.  Under WSL2 ``nvidia-smi`` cannot name the
    Windows processes, so the strip says the memory in use and no more.
``GET /api/system``
    The System screen: ``gpuwm-cm1 doctor --plain`` (:mod:`cm1core.cm1run.system`).
``POST /api/runs/RUN/convert``
    Convert again: ``gpuwm-cm1 cm1out RUNDIR --force [--format nc|grads]``,
    launched at nice 19 with no GPU, once the run has ended.
``POST /api/runs/RUN/reproduce``
    Reproduce bundle: ``gpuwm-cm1 reproduce RUNDIR``
    (:mod:`cm1core.cm1run.reproduce`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from ..cm1run.system import film_fix
from . import commands as C
from . import storm
from .files import is_view_only, listing, read_json, tree_bytes
from .runstate import (
    EVENTS_NAME, events_path, log_digest, segment_dirs, segment_rows, this_host, warning_words,
)

if TYPE_CHECKING:
    from .api import Api, Reply

#: How long a run folder's size, the GPU list and the system report are reused.
DISK_TTL_S = 20.0
GPU_TTL_S = 4.0
SYSTEM_TTL_S = 30.0
LOG_LINES = 160
LOG_TAIL_BYTES = 96 * 1024
#: The quick-looks the Run screen shows, in order (screen 8); the rest follow.
RUN_LOOKS = ("reflectivity", "updraft-5km", "surface", "rain", "swath", "spin-near-ground")
#: The favourite looks, in guided order.
GUIDED_LOOKS = ("reflectivity", "spin-near-ground", "updraft-5km", "swath")
#: The looks after them, in the More looks row's order (copy/looks.json "more_looks").
MORE_LOOKS = ("surface", "rain", "clouds-from-above", "updraft-slice", "hail-track", "spin-track", "environment",
              "parcels", "swirl")
#: Receipt facts the Results screen lists, in order: (label, path in the receipt).
RECEIPT_FACTS = (
    ("gpuwm-cm1", ("version",)),
    ("Status", ("status",)),
    ("Finished", ("completed_utc",)),
    ("GPU", ("runtime_benchmark", "gpu_name")),
    ("Steps", ("runtime_benchmark", "completed_steps")),
    ("Wall time of the steps", ("runtime_benchmark", "wall_seconds")),
    ("Whole process", ("process_wall_seconds",)),
    ("Model s per wall s", ("runtime_benchmark", "simulated_seconds_per_wall_second")),
    ("Peak GPU memory", ("runtime_benchmark", "sampled_peak_device_used_bytes")),
    ("Config sha256", ("config_sha256",)),
    ("CM1 authority", ("authority_sha256",)),
)


def _object(value: Any) -> dict[str, Any]:
    """A JSON object as a dict; anything else (a hand-edited or torn file) as an empty one."""

    return value if isinstance(value, dict) else {}


def _dig(document: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(document, dict):
            return None
        document = document.get(key)
    return document


def _utc_seconds(text: Any) -> float | None:
    from .runstate import _utc_seconds as parse

    return parse(text)


#: Seconds the processor takes to draw one saved time's pictures before a run has measured its own (a 1 km run on
#: a desktop: 25 saved times took about 95 s after a 35 s run on an RTX 4090).
DRAW_S_PER_FRAME = 4.0


def pictures_behind(saved: list[dict[str, Any]], newest_s: float | None, drawn: int, *,
                    now: float | None = None) -> dict[str, Any] | None:
    """How far the pictures (drawn on the processor) are behind the times the GPU has saved, or None.

    ``saved``: this segment's frame records (``model_time_s``, ``time_utc``); ``newest_s``: the newest picture's
    storm time; ``drawn``: the saved times drawn so far (every segment).  ``behind`` saved times have no picture
    yet, about ``about_s`` seconds of drawing at the rate this run has drawn them (or :data:`DRAW_S_PER_FRAME`).
    """

    times = [float(row["model_time_s"]) for row in saved
             if isinstance(row.get("model_time_s"), (int, float)) and not isinstance(row.get("model_time_s"), bool)]
    behind = [t for t in times if newest_s is None or t > float(newest_s) + 1e-6]
    if not behind:
        return None
    now = time.time() if now is None else now
    per = DRAW_S_PER_FRAME
    done_here = len(times) - len(behind)
    first = _utc_seconds(saved[0].get("time_utc")) if saved else None
    if first is not None and done_here >= 2:
        per = max(1.0, (now - first) / done_here)
    return {"behind": len(behind), "drawn": int(drawn), "saved": int(drawn) + len(behind),
            "about_s": int(max(5, round(len(behind) * per))),
            "lag_s": round(times[-1] - (float(newest_s) if newest_s is not None else times[0]), 3)}


def _round(value: Any, digits: int = 3) -> Any:
    return round(float(value), digits) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


def receipt_summary(receipt: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The receipt's headline facts as label/value rows (plain numbers, with units in the label)."""

    if not isinstance(receipt, dict):
        return []
    rows = []
    for label, path in RECEIPT_FACTS:
        value = _dig(receipt, path)
        if value is None:
            continue
        if label in ("Wall time of the steps", "Whole process") and isinstance(value, (int, float)):
            value = f"{float(value):.1f} s"
        elif label == "Peak GPU memory" and isinstance(value, (int, float)):
            value = f"{float(value) / 1024**3:.2f} GiB"
        elif label == "Model s per wall s" and isinstance(value, (int, float)):
            value = f"{float(value):.1f}"
        rows.append({"label": label, "value": str(value)})
    unvalidated = receipt.get("unvalidated")
    if isinstance(unvalidated, dict):
        names = []
        for capability in unvalidated.get("capabilities") or []:
            if isinstance(capability, dict):
                names.extend(str(value) for value in capability.get("values") or [])
        rows.append({"label": "Not checked against CM1", "value": ", ".join(names) if names else "nothing",
                     "note": "the receipt's unvalidated register"})
    return rows


def _log_tail(path: Path, lines: int = LOG_LINES) -> dict[str, Any] | None:
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            stream.seek(max(0, size - LOG_TAIL_BYTES))
            data = stream.read()
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace").splitlines()
    if size > LOG_TAIL_BYTES and text:
        text = text[1:]
    return {"lines": text[-lines:], "bytes": size}


class RunViews:
    """The data of screens 1, 8, 9 and 10, beside :class:`.api.Api` (which routes to it)."""

    def __init__(self, api: "Api") -> None:
        self.api = api
        self._lock = threading.Lock()
        self._disk: dict[str, tuple[float, int]] = {}
        self._gpus: tuple[float, dict[str, Any]] | None = None
        self._system: tuple[float, dict[str, Any]] | None = None

    # -------------------------------------------------------------- routing

    def get(self, parts: list[str], query: dict[str, list[str]]) -> "Reply | None":
        from .api import Reply

        refresh = (query.get("refresh") or [""])[0] == "1"
        if parts == ["gpus"]:
            return Reply(body=self.gpus(refresh=refresh))
        if parts == ["system"]:
            return Reply(body=self.system(refresh=refresh))
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "live":
            run_id = parts[1]
            return Reply(body=self.live(run_id, self.api.root.existing_run(run_id)))
        return None

    def post(self, run_id: str, rundir: Path, action: str, payload: dict[str, Any]) -> "Reply | None":
        handler = {"convert": self.convert, "reproduce": self.reproduce}.get(action)
        return None if handler is None else handler(run_id, rundir, payload)

    # -------------------------------------------------------------- Runs (screen 1)

    def disk_bytes(self, rundir: Path) -> int:
        key = os.fspath(rundir)
        now = time.monotonic()
        with self._lock:
            cached = self._disk.get(key)
        if cached is not None and now - cached[0] < DISK_TTL_S:
            return cached[1]
        size = tree_bytes(rundir)
        with self._lock:
            self._disk[key] = (now, size)
        return size

    def forget_disk(self, rundir: Path) -> None:
        """A folder just deleted or emptied is measured again the next time it is asked for."""

        with self._lock:
            self._disk.pop(os.fspath(rundir), None)

    def row_extra(self, rundir: Path, row: dict[str, Any]) -> dict[str, Any]:
        """The Runs table's Wall and Disk columns, and the GPU the run used."""

        segments = segment_rows(rundir) if not is_view_only(rundir) else []
        walls = [segment["wall_s"] for segment in segments if isinstance(segment.get("wall_s"), (int, float))]
        if is_view_only(rundir):
            digest = log_digest(events_path(rundir))
            if digest is not None and digest.last is not None and isinstance(digest.last.get("elapsed_s"),
                                                                              (int, float)):
                walls = [float(digest.last["elapsed_s"])]
        run = _object(read_json(rundir / "run.json", default={}))
        gpu = run.get("gpu") if isinstance(run.get("gpu"), dict) else {}
        device = gpu.get("device") if isinstance(gpu.get("device"), dict) else gpu
        return {
            "wall_s": round(sum(walls), 1) if walls else None,
            "disk_bytes": self.disk_bytes(rundir),
            "gpu": (device or {}).get("name"),
            "view_only": is_view_only(rundir),
            "segments": len(segments),
            "scheduler": run.get("scheduler") if isinstance(run.get("scheduler"), dict) else None,
        }

    def gpus(self, refresh: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            cached = self._gpus
        if cached is not None and not refresh and now - cached[0] < GPU_TTL_S:
            return cached[1]
        from ..cm1run import gpuinfo

        query = gpuinfo.query_gpus()
        runs = self._active_runs()
        out = []
        for gpu in query.gpus:
            # A run whose card is not recorded yet (a moment after Start) is shown on the first card.
            mine = [row for row in runs if row.get("gpu_uuid") == gpu.uuid or (
                not row.get("gpu_uuid") and row.get("gpu_index", 0 if gpu.index == 0 else None) == gpu.index)]
            apps = {app.pid: app for app in gpu.compute_apps if app.pid is not None}
            listed = []
            for row in mine:
                app = next((apps[pid] for pid in row["pids"] if pid in apps), None)
                listed.append({"id": row["id"], "name": row["name"], "state": row["state"],
                               "used_bytes": None if app is None else app.used_bytes})
            gm1_bytes = sum(item["used_bytes"] or 0 for item in listed)
            out.append({
                "index": gpu.index, "uuid": gpu.uuid, "name": gpu.name, "driver": gpu.driver_version,
                "sm": gpu.sm, "total_bytes": gpu.total_bytes, "used_bytes": gpu.used_bytes,
                "free_bytes": gpu.free_bytes, "in_use_bytes": gpu.in_use_bytes, "wsl2": gpu.wsl2,
                "processes_visible": gpu.processes_visible,
                "gm1_runs": listed, "gm1_bytes": gm1_bytes if any(item["used_bytes"] for item in listed) else None,
                "note": ("Under WSL2 nvidia-smi cannot name Windows programs; memory in use includes the Windows "
                         "desktop." if gpu.wsl2 and not gpu.compute_apps else ""),
            })
        answer = {"status": query.status, "message": query.message, "wsl2": query.wsl2, "gpus": out,
                  "host": this_host(), "command": "nvidia-smi --query-gpu=index,name,driver_version,memory.used,"
                                                  "memory.total --format=csv"}
        with self._lock:
            self._gpus = (now, answer)
        return answer

    def _active_runs(self) -> list[dict[str, Any]]:
        """Runs under the root on this host that hold (or wait for) a GPU, with their pids and GPU."""

        from .runstate import read_status

        found = []
        host = this_host()
        for path in self.api.root.iter_runs():
            if is_view_only(path):
                continue
            run_id = self.api.root.run_id(path)
            status = read_status(path, host=host, launched_alive=self.api.jobs.engine_alive(run_id))
            if status.state not in ("running", "waiting") or not status.detail.get("local"):
                continue
            segments = segment_dirs(path)
            state = read_json(segments[-1] / "segment.json", default={}) if segments else {}
            state = state if isinstance(state, dict) else {}
            pids = [ident.get("pid") for ident in (state.get("child"), state.get("supervisor"))
                    if isinstance(ident, dict) and ident.get("pid")]
            if status.detail.get("pid"):
                pids.append(status.detail["pid"])
            gpu = state.get("gpu") if isinstance(state.get("gpu"), dict) else {}
            device = gpu.get("device") if isinstance(gpu.get("device"), dict) else gpu
            run = _object(read_json(path / "run.json", default={}))
            found.append({"id": run_id, "name": run.get("name") or path.name, "state": status.state, "pids": pids,
                          "gpu_uuid": device.get("uuid"), "gpu_index": device.get("index")})
        return found

    # -------------------------------------------------------------- System (screen 10)

    def system(self, refresh: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            cached = self._system
        if cached is not None and not refresh and now - cached[0] < SYSTEM_TTL_S:
            return {**cached[1], "cached_s": round(now - cached[0], 1)}
        native_dir = None
        if self.api.native:
            native_dir = Path(self.api.native["library"]).parent
        args = C.doctor_args(self.api.root.path, native_dir)
        result = self.api.jobs.host(args, timeout=60.0)
        report = result.json if isinstance(result.json, dict) and "rows" in result.json else None
        answer: dict[str, Any] = {
            "ok": report is not None,
            "command": C.display(args),
            "rc": result.rc,
            "report": report,
            "message": "" if report is not None else "The check did not finish.",
            "fix": "" if report is not None else ((result.stderr.strip().splitlines() or ["Read the output."])[-1]),
            "runner": self.api.jobs.runner.kind,
        }
        with self._lock:
            self._system = (now, answer)
        return {**answer, "cached_s": 0.0}

    # -------------------------------------------------------------- Run (screen 8)

    def _base(self, run_id: str) -> str:
        return f"/api/runs/{quote(run_id, safe='')}/files/"

    def looks(self, run_id: str, rundir: Path, order: tuple[str, ...] = RUN_LOOKS) -> dict[str, Any]:
        """The latest picture of each quick-look, in ``order``, and the time series.  A picture with no storm in
        it yet carries the line that says so (``storm``, :mod:`.storm`)."""

        base = self._base(run_id)
        index = read_json(rundir / "quicklook" / "index.json", default=None)
        index = index if isinstance(index, dict) else {}
        notes = storm.run_notes(rundir, index)
        newest = None
        latest = []
        for product in index.get("products") or []:
            if not isinstance(product, dict):
                continue
            frames = [frame for frame in product.get("frames") or [] if isinstance(frame, dict)
                      and isinstance(frame.get("file"), str)]
            row: dict[str, Any] = {"id": product.get("id"), "title": product.get("title"), "count": len(frames)}
            if frames:
                last = frames[-1]
                row.update({"time_s": last.get("time_s"), "url": base + "quicklook/" + quote(last["file"]),
                            "thumb_url": base + "quicklook/" + quote(last["thumb"]) if isinstance(last.get("thumb"),
                                                                                                 str) else None})
                line = None if product.get("once") else storm.note_for(notes, last.get("time_s"))
                if line:
                    row["storm"] = line
                shown = last.get("time_s")
                # A look drawn once per run (the environment, the parcels) is not the newest saved time.
                if not product.get("once") and isinstance(shown, (int, float)) and not isinstance(shown, bool):
                    newest = float(shown) if newest is None else max(newest, float(shown))
            if product.get("not_drawn"):
                row["not_drawn"] = product["not_drawn"]
            latest.append(row)
        position = {name: number for number, name in enumerate(order)}
        latest.sort(key=lambda row: position.get(str(row.get("id")), len(position)))
        series = read_json(rundir / "quicklook" / "series.json", default=None)
        picture = index.get("series") if isinstance(index.get("series"), dict) else {}
        return {
            "looks": latest,
            "series": series if isinstance(series, dict) else None,
            "series_picture_url": (base + "quicklook/" + quote(str(picture["picture"])))
            if isinstance(picture.get("picture"), str) else None,
            "run": index.get("run"),
            "worker": index.get("worker"),
            "frames_drawn": len(index.get("frames") or []),
            "newest_picture_s": newest,
        }

    def live(self, run_id: str, rundir: Path) -> dict[str, Any]:
        api = self.api
        status = api.status(run_id, rundir)
        view_only = is_view_only(rundir)
        segments = segment_dirs(rundir)
        segment = segments[-1] if segments else None
        digest = log_digest(events_path(rundir, segment))
        admitted = (digest.admitted if digest else None) or {}
        startup = (digest.startup if digest else None) or {}
        progress = (digest.progress if digest else None) or {}
        now = time.time()
        writes = []
        if digest is not None:
            if digest.frames:
                frame = digest.frames[-1]
                writes.append({"kind": "frame", "number": frame.get("index"), **frame})
            if digest.restarts:
                restart = digest.restarts[-1]
                writes.append({"kind": "restart", **restart})
        last_write = max(writes, key=lambda row: _utc_seconds(row.get("time_utc")) or 0.0) if writes else None
        if last_write is not None:
            moment = _utc_seconds(last_write.get("time_utc"))
            last_write["age_s"] = None if moment is None else round(max(0.0, now - moment), 1)
        stages = startup.get("stages") if isinstance(startup.get("stages"), dict) else {}
        run = _object(read_json(rundir / "run.json", default={}))
        estimate = read_json(rundir / "estimate.json", default=None)
        planned_gpu = _dig(estimate, ("needs", "gpu_memory_bytes", "value")) if isinstance(estimate, dict) else None
        launched_here = any(row.role in ("run", "resume") for row in api.jobs.launched(run_id, alive_only=False))
        scheduler = status.get("scheduler") or (run.get("scheduler") if isinstance(run.get("scheduler"), dict) else None)
        if launched_here:
            started_by = "this GUI"
        elif isinstance(scheduler, dict) and scheduler.get("job_id"):
            started_by = f"{scheduler.get('kind') or 'batch'} job {scheduler.get('job_id')}"
        elif view_only:
            started_by = "gpuwm-cm1 cartesian (a plain output folder)"
        elif segments:
            started_by = "a terminal or another GUI"
        else:
            started_by = None
        log_path = segment / "run.log" if segment is not None else None
        log = _log_tail(log_path) if log_path is not None and log_path.is_file() else None
        if log is not None:
            log["url"] = self._base(run_id) + f"segments/{segment.name}/run.log"
        warnings = [{"code": warning.get("code"), "text": warning_words(warning), "time_utc": warning.get("time_utc")}
                    for warning in (digest.warnings if digest else [])]
        looks = self.looks(run_id, rundir)
        # When the first picture comes, and the next: the run's own saved-time interval (Watching says it).
        pictures = storm.schedule(admitted, status, looks.get("newest_picture_s"))
        # A fast GPU saves times faster than the processor draws them: how far behind the pictures are.
        behind = pictures_behind(digest.frames if digest else [], looks.get("newest_picture_s"),
                                 looks.get("frames_drawn") or 0)
        gpu = progress.get("gpu") if isinstance(progress.get("gpu"), dict) else (
            startup.get("gpu") if isinstance(startup.get("gpu"), dict) else None)
        body: dict[str, Any] = {
            "id": run_id,
            "url": quote(run_id, safe=""),
            "path": str(rundir),
            "name": run.get("name") or rundir.name,
            "case": run.get("case") or run.get("preset"),
            "view_only": view_only,
            "status": status,
            "attached": {"started_by": started_by, "launched_here": launched_here, "host": status.get("host"),
                         "local": status.get("local"), "this_host": this_host(), "scheduler": scheduler},
            "admitted": {key: admitted.get(key) for key in (
                "grid", "dt_s", "steps", "start_step", "nsound", "duration_s", "microphysics", "frames_expected",
                "restarts_expected", "history_interval_s", "restart_interval_s", "resumed_from", "config_sha256",
                "time_utc", "host")} if admitted else None,
            "startup": {"seconds": startup.get("seconds"), "nvrtc_compile_s": startup.get("nvrtc_compile_s"),
                        "nvrtc_programs": startup.get("nvrtc_programs"),
                        "stages": [[name, _round(value)] for name, value in stages.items()]} if startup else None,
            "progress": {key: progress.get(key) for key in (
                "step", "total_steps", "model_time_s", "duration_s", "eta_s", "s_per_step_recent", "s_per_step_mean",
                "model_s_per_wall_s", "phase", "heartbeat", "time_utc", "last_io")} if progress else None,
            "gpu": gpu,
            "planned_gpu_bytes": planned_gpu,
            "pictures": pictures,
            "pictures_behind": behind,
            "last_write": last_write,
            "frames": 0 if digest is None else len(digest.frames),
            "restarts": 0 if digest is None else len(digest.restarts),
            "warnings": warnings,
            "refused": digest.refused if digest else None,
            "end": digest.end if digest else None,
            **looks,
            "log": log,
            "segments": segment_rows(rundir) if not view_only else [],
            "commands": self.commands(rundir, view_only),
            "commands_log": self._base(run_id) + C_COMMANDS_LOG if (rundir / C_COMMANDS_LOG).is_file() else None,
        }
        return body

    def commands(self, rundir: Path, view_only: bool = False) -> dict[str, str]:
        if view_only:
            return {"convert": C.display(["cm1out", str(rundir)]),
                    "quicklook": C.display(["quicklook", str(rundir)])}
        native = self.api.native
        return {
            "start": C.display(C.run_args(rundir, native)),
            "stop": C.display(C.stop_args(rundir)),
            "pause": C.display(C.pause_args(rundir)),
            "force_stop": C.display(C.stop_args(rundir, "force")),
            "terminate": C.display(C.stop_args(rundir, "terminate")),
            "resume": C.display(C.resume_args(rundir, native=native)),
            "status": C.display(C.status_args(rundir)),
            "convert": C.display(C.convert_args(rundir)),
            "reproduce": C.display(C.reproduce_args(rundir)),
            "jobscript": C.display(C.jobscript_args(rundir, "slurm", native=native)),
        }

    # -------------------------------------------------------------- Results (screen 9)

    def results_extra(self, run_id: str, rundir: Path, body: dict[str, Any]) -> dict[str, Any]:
        """What the advanced Results screen adds to :meth:`.api.Api.results`."""

        view_only = is_view_only(rundir)
        index = read_json(rundir / "quicklook" / "index.json", default=None)
        frames = index.get("frames") if isinstance(index, dict) and isinstance(index.get("frames"), list) else []
        restarts = []
        if not view_only:
            try:
                from ..cm1run.rundir import RunDirectory

                restarts = [{"number": row.number, "model_time_s": row.model_time_s, "segment": row.segment}
                            for row in RunDirectory(rundir).lineage_restarts()]
            except Exception:  # noqa: BLE001 - a folder mid-write lists its restarts next time
                restarts = []
        bundle = rundir / "reproduce"
        reproduce = None
        if bundle.is_dir():
            files = listing(bundle)
            tarball = next((row["name"] for row in files if row["name"].endswith("-reproduce.tar.gz")), None)
            reproduce = {"folder": str(bundle), "files": files,
                         "tarball_url": self._base(run_id) + "reproduce/" + quote(tarball) if tarball else None,
                         "readme_url": self._base(run_id) + "reproduce/README.txt"
                         if (bundle / "README.txt").is_file() else None}
        converting = [row.command for row in self.api.jobs.launched(run_id) if row.role in ("convert", "cm1out")]
        report = read_json(rundir / "import-report.json", default=None)
        segments = segment_dirs(rundir)
        if view_only and body.get("receipt") is None:
            body["receipt"] = read_json(rundir / "receipt.json", default=None)
        receipt_path = (segments[-1] / "receipt.json") if segments else rundir / "receipt.json"
        body.update({
            "receipt_url": self._base(run_id) + quote(str(receipt_path.relative_to(rundir)))
            if receipt_path.is_file() else None,
            "view_only": view_only,
            "frames": frames,
            "restarts": restarts,
            "segments": segment_rows(rundir) if not view_only else [],
            "receipt_summary": receipt_summary(body.get("receipt")),
            "reproduce": reproduce,
            "converting": converting,
            "output": report.get("output") if isinstance(report, dict) else None,
            "raw_frames_bytes": body.get("disk", {}).get("frames_bytes"),
        })
        body.setdefault("commands", {}).update(self.commands(rundir, view_only))
        body.update(self.looks_extra(run_id, rundir, index if isinstance(index, dict) else None,
                                     body.get("status") or {}))
        body["pictures_behind"] = None
        if not view_only:
            segments = segment_dirs(rundir)
            digest = log_digest(events_path(rundir, segments[-1])) if segments else None
            if digest is not None:
                looks = self.looks(run_id, rundir)
                body["pictures_behind"] = pictures_behind(digest.frames, looks.get("newest_picture_s"),
                                                          looks.get("frames_drawn") or 0)
        return body

    def looks_extra(self, run_id: str, rundir: Path, index: dict[str, Any] | None,
                    status: dict[str, Any]) -> dict[str, Any]:
        """The More looks row, the films and their words (both Results views)."""

        from . import looks as LK

        films = LK.films(run_id, rundir)
        making = {row.command for row in self.api.jobs.launched(run_id) if row.role == "film"}
        for look, row in films.items():
            row["running_here"] = any(f" --film {look} " in f"{line} " for line in making)
        drawing = any(row.role == "looks" for row in self.api.jobs.launched(run_id))
        return {
            "more_looks": LK.more_looks(run_id, rundir, index, status, films),
            "films": films,
            "film_looks": list(LK.film_looks()),
            "film_fps": dict(LK.FILM_FPS),
            "mp4_available": LK.mp4_available(),
            "mp4_fix": film_fix(),
            "drawing_looks": drawing,
            "looks_ui": LK.ui_words(),
            "looks_about_s": {"all": LK.about_seconds(index, "all"), "swirl": LK.about_seconds(index, "swirl"),
                              "parcels": LK.about_seconds(index, "parcels")},
            "film_commands": {look: C.display(C.film_args(rundir, look)) for look in LK.film_looks()},
            "look_commands": {"all": C.display(C.quicklook_pass_args(rundir)),
                              **{look: C.display(C.look_args(rundir, look)) for look in LK.ONCE_LOOKS}},
        }

    def _ended_or_refuse(self, run_id: str, rundir: Path, what: str, command: str) -> None:
        from .api import ApiError

        status = self.api.status(run_id, rundir)
        if status.get("state") in ("running", "waiting", "stale"):
            raise ApiError(409, f"{what} once the run has ended.",
                           "While it runs, the cm1out sidecar converts each frame as it lands.", command=command)

    def convert(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, Reply

        fmt = payload.get("format")
        if fmt not in (None, "", "nc", "grads"):
            raise ApiError(400, "format must be nc or grads.", "Leave it out to follow the namelist's output_format.")
        args = C.convert_args(rundir, fmt or None)
        line = C.display(args)
        self._ended_or_refuse(run_id, rundir, "Convert again", line)
        if not segment_dirs(rundir):
            raise ApiError(409, "This run has no frames to convert yet.", "Start it first.", command=line)
        with self.api.jobs.run_lock(run_id):
            busy = [row for row in self.api.jobs.launched(run_id) if row.role in ("convert", "cm1out")]
            if busy:
                raise ApiError(409, "A conversion of this run is still going.", "Wait for it to finish.",
                               command=line)
            self.api.jobs.require("cm1out")
            launched = self.api.jobs.launch(run_id, rundir, args, role="convert")
            self.api.jobs.log_command(rundir, "convert", C.sidecar_line(args).removesuffix(" &"),
                                      "convert every frame again, on the CPU")
        what = {"nc": "netCDF", "grads": "GrADS"}.get(fmt or "", "the namelist's format")
        return Reply(200, {"ok": True, "action": "convert", "run": run_id, "command": line, "pid": launched.process.pid,
                           "message": f"Converting every frame again to {what} on the CPU (nice 19, no GPU). "
                                      "The file list updates when it is done.",
                           "fix": ""})

    def reproduce(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, Reply

        args = C.reproduce_args(rundir)
        result = self.api.jobs.host(args, timeout=60.0)
        self.api.jobs.log_command(rundir, "reproduce", result.command, f"rc {result.rc}")
        summary = result.json if isinstance(result.json, dict) else {}
        ok = result.rc == 0 and bool(summary.get("ok"))
        if not ok:
            detail = (result.stderr.strip().splitlines() or ["the bundle was not written"])[-1]
            raise ApiError(500, "The reproduce bundle was not written.", detail, command=result.command)
        tarball = summary.get("tarball_name")
        return Reply(200, {
            "ok": True, "action": "reproduce", "run": run_id, "command": result.command, "rc": result.rc,
            "message": f"Wrote {len(summary.get('files') or [])} files to {summary.get('folder')}.",
            "fix": "", "bundle": summary,
            "tarball_url": self._base(run_id) + "reproduce/" + quote(str(tarball)) if tarball else None,
        })


C_COMMANDS_LOG = "commands.log"

__all__ = ["GUIDED_LOOKS", "MORE_LOOKS", "RUN_LOOKS", "RunViews", "receipt_summary"]
