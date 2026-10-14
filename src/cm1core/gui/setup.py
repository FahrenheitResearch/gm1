"""The setup screens' endpoints: New run, Check, Sounding, Domain and time, Settings, Estimate.

Reads come from the run directory alone (``import-report.json`` and its
``setup`` block, ``estimate.json``, ``geometry.json``, ``settings.toml``,
``namelist.input`` and ``namelist.diff``, ``fit.json``); every change is one
``gpuwm-cm1`` command run as a host-only subprocess, returned in the reply
and appended to ``commands.log``:

GET  /api/examples[?wait=1][&refresh=1]    CM1's own examples and the presets, priced on this GPU in the
                                          background: each one's "price.state" says whether its numbers are
                                          here ("priced"), kept from an earlier pricing ("kept"), being checked
                                          again ("checking") or on their way ("pricing"); "pricing" says one
                                          is under way; wait=1 waits for it
GET  /api/runs/RUN/setup                  everything the six screens draw, for either mode
GET  /api/runs/RUN/namelist               the run's namelist, the user's original and the diff
GET  /api/runs/RUN/measure                how "Measure speed" is going
POST /api/runs/RUN/edit                   {"set": {KEY: VALUE}, "settings": {TABLE.KEY: VALUE}, "reset": [...]}
POST /api/runs/RUN/recommended            the storm-track check's frame and bubble
POST /api/runs/RUN/namelist               {"text": ...}: the editor's Save; {"revert": true}: back to your file
POST /api/runs/RUN/acknowledge            {"output_keys": true, "thompson": false}
POST /api/runs/RUN/sounding               {"sounding": {"path"} | {"text"}}
POST /api/runs/RUN/jobscript              {"scheduler": "slurm" | "pbs", "time", "partition", "account", "qos",
                                          "mem", "gpu_type", "cpus", "until_s", "setup": [LINE, ...], "python",
                                          "write"}
POST /api/runs/RUN/measure                Measure speed: a few steps on the GPU, detached

The mode (guided or advanced) is a view: the same replies serve both, and
guided mode draws the plain sentences (``plain``) where advanced draws CM1's
words.  Nothing here imports the model; standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import tomllib
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from . import commands as C
from .files import read_json, stage_upload
from .jobs import COMMANDS_LOG
from .runstate import segment_dirs

if TYPE_CHECKING:  # pragma: no cover
    from .api import Api, Reply

#: CM1's own examples, shipped byte for byte (``cm1core.data``'s packaged inputs).
CM1_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "id": "cm1-supercell",
        "input": "cm1r21.1-supercell-namelist",
        "name": "CM1 supercell",
        "plain_name": "Supercell, 1 km",
        "line": "One rotating thunderstorm on a coarse 1 km grid: CM1's own supercell example, unchanged.",
        "source": "CM1 r21.1 run/config_files/supercell/namelist.input, unchanged",
        "picture": "/static/img/example-supercell-reflectivity.png",
        "picture_caption": "Reflectivity at 1 km, T+1:45:00, from GM1 0.1.0's run of this namelist on an RTX 4090. Simulated.",
        "suggested_name": "supercell-1km",
        # The guided preset made from this namelist: guided mode's offer and its reason apply here too.
        "guided_preset": "supercell-1km",
    },
    {
        "id": "cm1-squall-line",
        "input": "cm1r21.1-squall-line-namelist",
        "name": "CM1 squall line",
        "plain_name": "Squall line, 1 km",
        "line": "A long line of storms moving east, with a pool of cold air behind it: CM1's own squall-line "
                "example, unchanged.",
        "source": "CM1 r21.1 run/config_files/squall_line/namelist.input, unchanged",
        "picture": None,
        "picture_caption": None,
        "suggested_name": "squall-line-1km",
        "guided_preset": "squall-line-1km",
    },
)


def _measured(log: Path) -> dict[str, Any] | None:
    """What ``gpuwm-cm1 estimate --measure --json`` measured: its step time, steps and GPU (from its log)."""

    try:
        text = log.read_text(encoding="utf-8", errors="replace")
        # The JSON document starts on a line of its own; warnings may come before or after it.
        start = 0 if text.startswith("{") else text.find("\n{") + 1
        payload = json.JSONDecoder().raw_decode(text[start:])[0] if text[start:].startswith("{") else None
    except (OSError, ValueError):
        return None
    measured = payload.get("measured") if isinstance(payload, dict) else None
    if not isinstance(measured, dict):
        return None
    return {key: measured.get(key) for key in ("seconds_per_step", "steps", "gpu_name", "cells")}


def _guided_withheld(preset_id: str | None) -> str | None:
    """Why guided mode does not offer ``preset_id`` (its gallery card's line), or None when it does or is unknown."""

    if not preset_id:
        return None
    try:
        from ..cm1run import presets as P

        examined = P.examine(P.load(preset_id))
    except Exception:  # noqa: BLE001 - a preset that cannot be read leaves the example as it is
        return None
    if examined.get("offered"):
        return None
    # The reason itself ("Repeating sides have not been ..."): the page shows it beside a "Not offered" mark.
    reasons = examined.get("reasons") or []
    return str(reasons[0]) if reasons else (examined.get("why_not") or "Not offered yet.")


#: How long a priced example stays fresh: the GPU's free memory changes as programs start and stop.
EXAMPLE_PRICE_TTL_S = 30.0
EXAMPLES_DIR = "examples"
#: The examples' estimates kept across server starts, under ``EXAMPLES_DIR``, keyed by the deck, the
#: GM1 install (version and estimator code), this machine's measured rates, the GPU and its free memory to
#: FREE_BUCKET_BYTES.
ESTIMATES_FILE = "estimates.json"
ESTIMATES_SCHEMA = "gm1.gui-example-estimates/v1"
FREE_BUCKET_BYTES = 256 * 1024 * 1024
#: The disk's free space to this, in the key: the estimate prices the run's output against it (``query_host``).
DISK_BUCKET_BYTES = 1024 * 1024 * 1024
#: Examples priced at once, each worker a host command at a time (jobs.HOST_CONCURRENCY slots).
PRICE_WORKERS = 2
#: ``GET /api/examples?wait=1`` waits at most this long for the pricing under way.
WAIT_LIMIT_S = 240.0
#: Standard card sizes for "the smallest card it fits" (GB).
CARD_SIZES_GB = (8, 10, 12, 16, 20, 24, 32, 40, 48, 80, 96)
#: The estimator's rule (records: FITS_FRACTION_OF_FREE).
FITS_FRACTION = 0.95
TEXT_LIMIT = 1024 * 1024
#: Shell lines a job script may run first on the node (gpuwm-cm1 jobscript --setup).
JOBSCRIPT_SETUP_LINES = 20


def _deck_sha(rundir: Path) -> str | None:
    """The sha256 of an example's deck.toml and input_sounding, as imported: what its estimate was made from."""

    digest = hashlib.sha256()
    try:
        digest.update((rundir / "deck.toml").read_bytes())
    except OSError:
        return None
    sounding = rundir / "input_sounding"
    if sounding.is_file():
        digest.update(b"\0input_sounding\0" + sounding.read_bytes())
    return digest.hexdigest()


def _machine_words(gpu: dict[str, Any]) -> list[str]:
    """What the estimate reads from ``nvidia-smi`` and ``query_host`` (estimate.py ``_add_checks``, 1036-1105), as
    words for the kept prices' key: the GPU itself, the programs on it, whether more than 1 GB of it is in use
    (the ``gpu-shared`` note), its free memory to FREE_BUCKET_BYTES; RAM, and the runs folder's disk free to
    DISK_BUCKET_BYTES.  The free memory and disk inside one bucket are the words a kept price may not share
    with a fresh one; :func:`_verdicts_hold` keeps every verdict the estimate draws from them the same."""

    host = gpu.get("host") if isinstance(gpu.get("host"), dict) else {}
    in_use = gpu.get("in_use_bytes")
    apps = sorted((str(app.get("pid")), str(app.get("name"))) for app in gpu.get("compute_apps") or ()
                  if isinstance(app, dict))
    disk_free = host.get("disk_free_bytes")
    return [str(gpu.get("uuid") or gpu.get("name")), str(gpu.get("name")), str(gpu.get("total_bytes")),
            str(gpu.get("sm")), str(gpu.get("wsl2")), json.dumps(apps),
            str(None if in_use is None else float(in_use) > 1.0e9),
            str(int(gpu["free_bytes"]) // FREE_BUCKET_BYTES),
            str(host.get("ram_total_bytes")), str(host.get("disk_root")), str(host.get("wsl2")),
            str(None if disk_free is None else int(disk_free) // DISK_BUCKET_BYTES)]


def _verdicts_hold(estimate: Any, gpu: dict[str, Any] | None) -> bool:
    """Whether a kept estimate, made with the GPU and disk free it records, draws every verdict the estimator
    draws from them the same way with the free memory and disk free in ``gpu`` now.

    The estimator compares them with fixed fractions: GPU need and the tile-128 need against
    ``FITS_FRACTION_OF_FREE`` of the free memory (``fits.gpu``, the ``gpu-memory`` check, the tile advice,
    ``largest-box``; estimate.py 537, 826-827, 1040-1041, 1165), the run's output against ``DISK_FITS_FRACTION``
    and ``DISK_WARN_FRACTION`` of the disk free (``fits.disk``, the ``disk`` check; 561-565, 1089-1097).  A kept
    price whose bucket still matches but whose verdict would flip is priced again.
    """

    from ..cm1run.estimate import DISK_FITS_FRACTION, DISK_WARN_FRACTION, FITS_FRACTION_OF_FREE

    if not isinstance(estimate, dict) or not isinstance(gpu, dict):
        return False
    try:
        was_gpu = (estimate.get("gpu") or {}).get("free_bytes")
        now_gpu = gpu.get("free_bytes")
        needs = estimate.get("needs") or {}
        recommendation = estimate.get("recommendation") or {}
        tiled = recommendation.get("gpu_memory_bytes", (recommendation.get("tile_128_option") or {}).get(
            "gpu_memory_bytes"))
        for need in ((needs.get("gpu_memory_bytes") or {}).get("value"), tiled):
            if need is None:
                continue
            if (was_gpu is None) != (now_gpu is None):
                return False
            if was_gpu is not None and (float(need) <= FITS_FRACTION_OF_FREE * float(was_gpu)) != (
                    float(need) <= FITS_FRACTION_OF_FREE * float(now_gpu)):
                return False
        disk = (needs.get("disk_bytes") or {}).get("value")
        was_disk = (estimate.get("host") or {}).get("disk_free_bytes")
        now_disk = (gpu.get("host") or {}).get("disk_free_bytes") if isinstance(gpu.get("host"), dict) else None
        if disk is not None and (bool(was_disk) or bool(now_disk)):
            if not was_disk or not now_disk:
                return False
            for fraction in (DISK_FITS_FRACTION, DISK_WARN_FRACTION):
                if (float(disk) / float(was_disk) > fraction) != (float(disk) / float(now_disk) > fraction):
                    return False
            if (float(disk) <= DISK_FITS_FRACTION * float(was_disk)) != (
                    float(disk) <= DISK_FITS_FRACTION * float(now_disk)):
                return False
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def _iso(epoch: float | None) -> str | None:
    return None if epoch is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _gb(value: float | None) -> str:
    if value is None:
        return ""
    gb = value / 1e9
    if gb >= 10:
        return f"{gb:.0f} GB"
    return f"{gb:.1f} GB"


def duration_words(seconds: float | None) -> str:
    """A wall or storm time in plain words: about 2 minutes, about 1.5 hours."""

    if seconds is None:
        return ""
    seconds = float(seconds)
    if seconds < 90:
        return f"about {max(1, round(seconds / 10) * 10):.0f} seconds"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"about {round(minutes):.0f} minute{'s' if round(minutes) != 1 else ''}"
    hours = minutes / 60.0
    text = f"{hours:.1f}".rstrip("0").rstrip(".")
    return f"about {text} hour{'s' if text != '1' else ''}"


def storm_time_words(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = float(seconds)
    if seconds % 3600 == 0:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''}"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}".rstrip("0").rstrip(".") + " hours"
    return f"{seconds / 60:.0f} minutes"


def spacing_words(metres: float | None) -> str:
    if metres is None:
        return ""
    metres = float(metres)
    if metres >= 1000 and metres % 1000 == 0:
        return f"{metres / 1000:.0f} km"
    return f"{metres:g} m"


def smallest_card_gb(need_bytes: float | None) -> int | None:
    if not need_bytes:
        return None
    for size in CARD_SIZES_GB:
        if need_bytes <= FITS_FRACTION * size * 1e9:
            return size
    return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _older_than_the_deck(rundir: Path, record: Any, record_time: float | None, latest_import: float | None) -> bool:
    """Whether an estimate or storm-track record was worked out for an earlier deck.

    By content when the record names the deck.toml it was worked out for (``deck_file_sha256``, written by
    ``gpuwm-cm1 estimate``), so a clock that steps (WSL2) cannot make a current record look stale; by file times for
    a record that does not.
    """

    recorded = record.get("deck_file_sha256") if isinstance(record, dict) else None
    if isinstance(recorded, str):
        try:
            return hashlib.sha256((rundir / "deck.toml").read_bytes()).hexdigest() != recorded
        except OSError:
            return True
    return latest_import is not None and record_time is not None and record_time + 1e-6 < latest_import


def _read_text(path: Path, limit: int = TEXT_LIMIT) -> str | None:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError:
        return None
    if len(data) > limit:
        return None
    return data.decode("utf-8", "replace")


def _toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return {}


# ------------------------------------------------------------------ the plain words


def plain_summary(values: dict[str, Any], estimate: dict[str, Any] | None) -> dict[str, Any]:
    """The run in one or two sentences, for guided mode."""

    nx, ny = values.get("nx"), values.get("ny")
    dx, dy = values.get("dx"), values.get("dy")
    sentences = []
    if isinstance(nx, (int, float)) and isinstance(dx, (int, float)):
        from ..cm1run.fit import article        # the fit's own rule for "A" or "An" (no model import)

        width = nx * dx / 1000.0
        depth = (ny or nx) * (dy or dx) / 1000.0
        size = f"{width:g} × {depth:g}"
        sentences.append(
            f"{article(size)} {size} km area at {spacing_words(dx)}, "
            f"{storm_time_words(values.get('timax'))} of storm time.")
    if estimate:
        wall = ((estimate.get("time") or {}).get("wall_s") or {}).get("value")
        gpu = estimate.get("gpu") or {}
        need = ((estimate.get("needs") or {}).get("gpu_memory_bytes") or {}).get("value")
        disk = ((estimate.get("needs") or {}).get("disk_bytes") or {}).get("value")
        parts = []
        if wall is not None:
            where = f"on your {gpu.get('name', 'GPU').replace('NVIDIA GeForce ', '')}" if gpu.get("name") else \
                "on a GPU like the ones GM1 has measured"
            parts.append(f"{duration_words(wall).capitalize()} {where} (estimated).")
        if need is not None:
            needs = f"Needs {_gb(need)} of GPU memory"
            if disk is not None:
                needs += f" and about {_gb(disk)} of disk"
            parts.append(needs + ".")
        sentences.extend(parts)
    return {"sentences": sentences, "text": " ".join(sentences)}


#: The coverage report's per-row fields a page draws.
REPORT_ROW_FIELDS = ("key", "keys", "value", "text", "plain", "fix", "plain_fix", "after", "defaulted", "evidence",
                     "acknowledge", "expand", "family")


def report_view(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """The six categories of an import report, compact, as the namelist editor redraws them after a Save."""

    if not isinstance(report, dict) or not isinstance(report.get("sections"), list):
        return None
    sections = []
    for section in report["sections"]:
        if not isinstance(section, dict):
            continue
        rows = [{key: row.get(key) for key in REPORT_ROW_FIELDS if row.get(key) is not None}
                for row in section.get("rows") or [] if isinstance(row, dict)]
        sections.append({
            "id": section.get("id"), "label": section.get("label"),
            "count": section.get("count", len(rows)), "collapsed": bool(section.get("collapsed")),
            "ack": section.get("ack"), "rows": rows,
            "families": [{key: family.get(key) for key in ("family", "id", "n", "keys", "evidence")}
                         for family in section.get("families") or [] if isinstance(family, dict)],
        })
    output = report.get("output") or {}
    return {
        "exit": report.get("exit"),
        "verdict": report.get("verdict"),
        "verdict_text": report.get("verdict_text"),
        "counts": {section["id"]: section["count"] for section in sections},
        "sections": sections,
        "first_refused": report.get("first_refused"),
        "acknowledgments": report.get("acknowledgments") or {},
        "aliases": report.get("aliases") or [],
        "not_r21": report.get("not_r21"),
        "output": {"text": output.get("text"), "format": output.get("format")} if isinstance(output, dict) else None,
    }


def estimate_view(estimate: dict[str, Any] | None) -> dict[str, Any] | None:
    """The Estimate screen's rows: needed against available, with the basis of each number."""

    if not isinstance(estimate, dict):
        return None
    needs = estimate.get("needs") or {}
    gpu = estimate.get("gpu") or {}
    host = estimate.get("host") or {}
    fits = estimate.get("fits") or {}
    rows = []
    gpu_need = (needs.get("gpu_memory_bytes") or {})
    if gpu_need:
        total = gpu.get("total_bytes")
        free = gpu.get("free_bytes")
        rows.append({
            "id": "gpu", "what": "GPU memory", "need_bytes": gpu_need.get("value"),
            "low_bytes": gpu_need.get("low"), "high_bytes": gpu_need.get("high"),
            "have_bytes": free, "total_bytes": total,
            "other_bytes": gpu.get("in_use_bytes"),
            "fits": fits.get("gpu"), "label": gpu_need.get("label"), "basis": gpu_need.get("basis"),
            "basis_label": gpu_need.get("basis_label"),
            "have_text": (f"{_gb(free)} free of {_gb(total)}" if free is not None and total else
                          "no GPU seen"),
            "note": ("Under WSL2 the Windows desktop's share of the card counts as in use; nvidia-smi cannot "
                     "name its programs." if gpu.get("wsl2") else None),
        })
    locked = needs.get("host_locked_bytes") or {}
    if locked:
        rows.append({
            "id": "host", "what": "Locked host memory", "need_bytes": locked.get("value"),
            "have_bytes": host.get("ram_total_bytes"), "total_bytes": host.get("ram_total_bytes"),
            "other_bytes": None, "fits": fits.get("host"), "label": locked.get("label"),
            "basis": locked.get("basis"), "basis_label": locked.get("basis_label"),
            "have_text": f"{_gb(host.get('ram_total_bytes'))} RAM" if host.get("ram_total_bytes") else "",
        })
    disk = needs.get("disk_bytes") or {}
    if disk:
        free = host.get("disk_free_bytes")
        total = host.get("disk_total_bytes")
        rows.append({
            "id": "disk", "what": "Disk", "need_bytes": disk.get("value"),
            "have_bytes": free, "total_bytes": total,
            "other_bytes": None if total is None or free is None else total - free,
            "fits": fits.get("disk"), "label": disk.get("label"), "basis": disk.get("basis"),
            "basis_label": disk.get("basis_label"),
            "have_text": f"{_gb(free)} free" if free is not None else "",
        })
    time_block = estimate.get("time") or {}
    wall = time_block.get("wall_s") or {}
    step = time_block.get("step") or {}
    run = estimate.get("run") or {}
    return {
        "rows": rows,
        "fits": fits.get("all"),
        "fits_rule": fits.get("rule"),
        "wall": {
            "value_s": wall.get("value"), "low_s": wall.get("low"), "high_s": wall.get("high"),
            "label": wall.get("label"), "basis": wall.get("basis"), "basis_label": wall.get("basis_label"),
            "words": duration_words(wall.get("value")),
        },
        "step": {"seconds": step.get("seconds_per_step"), "label": step.get("label_used") or step.get("label"),
                 "basis": step.get("basis_used") or step.get("basis"), "measured_here": step.get("measured_here")},
        "run": run,
        "checks": list(estimate.get("checks") or []),
        "gpu": {key: gpu.get(key) for key in ("name", "index", "uuid", "sm", "wsl2", "driver_version",
                                               "total_bytes", "free_bytes", "in_use_bytes")} if gpu else None,
        "gpu_query": estimate.get("gpu_query"),
        "assumptions": list(estimate.get("assumptions") or []),
        "recommendation": estimate.get("recommendation"),
        "command": estimate.get("command"),
        "fake_runner": bool(estimate.get("fake_runner")),
    }


def sounding_view(report: dict[str, Any], geometry: dict[str, Any] | None) -> dict[str, Any]:
    """The Sounding screen: the source, the getcape block, the hodograph numbers and the checks."""

    sounding = report.get("sounding") or {}
    values = (report.get("setup") or {}).get("values") or {}
    isnd = sounding.get("isnd", values.get("isnd"))
    iwnd = values.get("iwnd")
    from .refusal_text import ISND_NAMES

    sources = {
        5: "CM1's analytic Weisman–Klemp profile (isnd = 5); no input_sounding file is read.",
        7: "The input_sounding file: thermodynamics and winds (isnd = 7).",
        17: "The input_sounding file's thermodynamics, with CM1's analytic winds (isnd = 17).",
    }
    # base.F's other analytic profiles: built from CM1's own constants, no file read.  The profile's name comes
    # first and the isnd word last, as for 5, 7 and 17, so a name with its own brackets ("BOMEX") nests in none.
    # A run starts only from 5, 7, 17 and 9, whose runs matched CM1's; the others are shown, not run (config.py
    # CM1-CFG-BASE-CLOUD-INTEGRATION and CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION): 2 and 3 are dry (imoist = 0),
    # 4 and 10 carry cloud in the base state and 11 is moist without it, and GM1's run from each did not match
    # CM1's; the run from 1, 8, 12-14 and 18-23 has not been compared with CM1's.
    shown_only = {
        2: " Dry: shown only (imoist = 0, iconly = 1).",
        3: " Dry: shown only (imoist = 0, iconly = 1).",
        **{n: " Cloud in the base state: shown only (iconly = 1); GM1's run from it does not match CM1's yet."
           for n in (4, 10)},
        11: " Shown only (iconly = 1); GM1's run from it does not match CM1's yet.",
        **{n: " Shown only (iconly = 1); GM1's run from it has not been checked against CM1's yet."
           for n in (1, 8, 12, 13, 14, 18, 19, 20, 21, 22, 23)},
    }
    for number in (1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23):
        sources[number] = (f"CM1's analytic profile: {ISND_NAMES.get(number, 'analytic')} (isnd = {number}); "
                           "no input_sounding file is read." + shown_only.get(number, ""))
    winds = {
        1: "CM1's straight-line shear (iwnd = 1)",
        2: "CM1's quarter-circle hodograph (iwnd = 2)",
    }
    env = sounding.get("environment") or {}
    parcels = []
    for key, label in (("surface_based", "surface"), ("most_unstable", "most unstable"),
                       ("mixed_layer", "mixed layer (500 m)")):
        row = env.get(key)
        if isinstance(row, dict):
            parcels.append({
                "parcel": label,
                "source_hpa": None if row.get("source_pressure_pa") is None else row["source_pressure_pa"] / 100.0,
                "source_t_k": row.get("source_temperature_k"),
                "source_qv_gkg": None if row.get("source_qv_kg_per_kg") is None else row["source_qv_kg_per_kg"] * 1000.0,
                "lcl_m": row.get("lcl_m_agl"), "lfc_m": row.get("lfc_m_agl"), "el_m": row.get("el_m_agl"),
                "cape": row.get("cape_j_per_kg"), "cin": row.get("cin_j_per_kg"),
            })
    wind = []
    motion = (geometry or {}).get("storm_motion") or {}
    if env:
        wind = [
            {"label": "Storm motion, Bunkers right mover", "u": env.get("bunkers_right_u_m_per_s"),
             "v": env.get("bunkers_right_v_m_per_s"), "unit": "m/s"},
            {"label": "Left mover", "u": env.get("bunkers_left_u_m_per_s"), "v": env.get("bunkers_left_v_m_per_s"),
             "unit": "m/s"},
            {"label": "0–6 km mean wind", "u": env.get("bunkers_mean_wind_u_m_per_s"),
             "v": env.get("bunkers_mean_wind_v_m_per_s"), "unit": "m/s"},
            {"label": "0–1 km shear", "value": env.get("shear_0_1km_m_per_s"), "unit": "m/s"},
            {"label": "0–6 km shear", "value": env.get("shear_0_6km_m_per_s"), "unit": "m/s"},
            {"label": "0–1 km storm-relative helicity", "value": env.get("srh_0_1km_m2_per_s2"), "unit": "m²/s²"},
            {"label": "0–3 km storm-relative helicity", "value": env.get("srh_0_3km_m2_per_s2"), "unit": "m²/s²"},
        ]
    elif motion:
        wind = [
            {"label": "Storm motion, Bunkers right mover", "u": (motion.get("right_mover_m_s") or [None, None])[0],
             "v": (motion.get("right_mover_m_s") or [None, None])[1], "unit": "m/s"},
            {"label": "Left mover", "u": (motion.get("left_mover_m_s") or [None, None])[0],
             "v": (motion.get("left_mover_m_s") or [None, None])[1], "unit": "m/s"},
            {"label": "0–6 km mean wind", "u": (motion.get("mean_wind_m_s") or [None, None])[0],
             "v": (motion.get("mean_wind_m_s") or [None, None])[1], "unit": "m/s"},
        ]
    checks = []
    for problem in sounding.get("problems") or []:
        checks.append({"state": "fail", "text": problem.get("text", ""), "fix": problem.get("fix")})
    for warning in sounding.get("warnings") or []:
        checks.append({"state": "warn", "text": str(warning)})
    top = sounding.get("top_m")
    model_top = sounding.get("highest_model_scalar_level_m")
    if sounding.get("reads_file"):
        if top is not None and model_top is not None and not sounding.get("problems"):
            checks.append({"state": "ok", "text": f"The file reaches {top:,.0f} m, above the model's highest level "
                                                  f"({model_top:,.0f} m)."})
    elif isnd == 5 or (isnd is not None and isnd not in (7, 17)
                       and sounding.get("environment_status") == "computed"):
        # The other analytic arms refuse a grid they are not defined on (rust analytic_soundings.rs), so a
        # column the initializer built is defined at every level.
        checks.append({"state": "ok", "text": "The profile is defined at every model level up to the model top."})
    if sounding.get("environment_status") == "computed":
        checks.append({"state": "ok", "text": "The parcel numbers are CM1's getcape.F, computed by GM1's "
                                              "initializer on this run's own levels."})
    elif sounding:
        checks.append({"state": "note", "text": "The parcel numbers are not shown: "
                                                + str(sounding.get("environment_reason") or "not computed") + "."})
    if values.get("imove") == 1:
        checks.append({"state": "note", "text": (
            f"Winds are ground-relative. The run moves with umove = {values.get('umove'):g} and "
            f"vmove = {values.get('vmove'):g} m/s (imove = 1).")})
    surface = None
    if env:
        surface = {"p_hpa": (env.get("surface_pressure_pa") or 0) / 100.0,
                   "theta_k": env.get("surface_temperature_k"),
                   "qv_gkg": (env.get("surface_qv_kg_per_kg") or 0) * 1000.0,
                   "lowest_level_m": env.get("lowest_mass_level_m_agl")}
    elif sounding.get("surface"):
        raw = sounding["surface"]
        surface = {"p_hpa": raw.get("pressure_hpa") or raw.get("p_hpa"), "theta_k": raw.get("theta_k"),
                   "qv_gkg": raw.get("qv_g_per_kg") or raw.get("qv_gkg"), "lowest_level_m": None}
    return {
        "isnd": isnd,
        "iwnd": iwnd,
        "source": sources.get(isnd, f"isnd = {isnd}") if isnd is not None else None,
        "winds": winds.get(iwnd) if isnd is not None and isnd != 7 else None,
        "reads_file": bool(sounding.get("reads_file")),
        "file": {"path": sounding.get("path"), "sha256": sounding.get("sha256"), "levels": sounding.get("levels"),
                 "top_m": top} if sounding.get("reads_file") else None,
        "surface": surface,
        "parcels": parcels,
        "wind": wind,
        "checks": checks,
        "environment_status": sounding.get("environment_status"),
        "hodograph": (geometry or {}).get("hodograph_m_s"),
        "hodograph_frame": (geometry or {}).get("hodograph_frame"),
        "storm_motion": motion or None,
        "authority": env.get("cm1_parcel_authority"),
        "motion_authority": env.get("storm_motion_authority"),
        "claim_limit": env.get("claim_limit"),
    }


# ------------------------------------------------------------------ the class


class SetupApi:
    """The setup screens' endpoints, beside :class:`.api.Api` (which routes to them)."""

    def __init__(self, api: "Api") -> None:
        self.api = api
        self._examples_lock = threading.Lock()
        #: row key -> (monotonic time it was priced or confirmed, info, price key or None)
        self._priced: dict[str, tuple[float, dict[str, Any], str | None]] = {}
        self._priced_at: dict[str, float] = {}
        self._confirmed: set[str] = set()
        self._pricing: threading.Thread | None = None
        self._again: tuple[list[dict[str, Any]], bool] | None = None
        self._last_gpu: dict[str, Any] | None = None
        self._disk: dict[str, Any] | None = None
        self._disk_lock = threading.RLock()
        self._rates_mark: tuple[Any, str] | None = None
        self._code_digest: str | None = None

    # -------------------------------------------------------------- routing

    def get(self, parts: list[str], query: dict[str, list[str]]) -> "Reply | None":
        from .api import Reply

        if parts == ["examples"]:
            return Reply(body=self.examples(refresh=(query.get("refresh") or [""])[0] == "1",
                                            wait=(query.get("wait") or [""])[0] == "1"))
        if len(parts) == 3 and parts[0] == "runs" and parts[2] in ("setup", "namelist", "measure"):
            run_id = parts[1]
            rundir = self.api.root.existing_run(run_id)
            if parts[2] == "setup":
                return Reply(body=self.setup(run_id, rundir))
            if parts[2] == "namelist":
                return Reply(body=self.namelist(run_id, rundir))
            return Reply(body=self.measure_status(run_id, rundir))
        return None

    ACTIONS = ("edit", "recommended", "namelist", "acknowledge", "sounding", "jobscript", "measure")

    def post(self, run_id: str, rundir: Path, action: str, payload: dict[str, Any]) -> "Reply | None":
        handler = {
            "edit": self.edit,
            "recommended": self.recommended,
            "namelist": self.save_namelist,
            "acknowledge": self.acknowledge,
            "sounding": self.sounding,
            "jobscript": self.jobscript,
            "measure": self.measure,
        }.get(action)
        return None if handler is None else handler(run_id, rundir, payload)

    # -------------------------------------------------------------- examples

    def example_rows(self) -> list[dict[str, Any]]:
        """CM1's shipped examples, then the presets this package ships (``cm1core/presets/<id>/``)."""

        rows = []
        try:
            from ..data import PackagedInputError, packaged_input_path
        except Exception:  # noqa: BLE001 - a broken install still lists the presets
            packaged_input_path = None  # type: ignore[assignment]
            PackagedInputError = Exception  # type: ignore[assignment,misc]  # noqa: N806
        for spec in CM1_EXAMPLES:
            path = None
            reason = None
            if packaged_input_path is not None:
                try:
                    path = packaged_input_path(spec["input"])
                except PackagedInputError as error:  # type: ignore[misc]
                    reason = str(error)
            row = {**{k: v for k, v in spec.items() if k not in ("input", "guided_preset")}, "kind": "cm1",
                   "path": None if path is None else str(path), "sounding": None, "missing": reason}
            # Guided mode greys the preset made from this namelist; the example says the same, with the same reason.
            withheld = _guided_withheld(spec.get("guided_preset"))
            if withheld:
                row["offered"], row["not_offered_reason"] = False, withheld
            rows.append(row)
        presets = Path(__file__).resolve().parents[1] / "presets"
        if presets.is_dir():
            for directory in sorted(p for p in presets.iterdir() if p.is_dir()):
                spec = read_json(directory / "preset.json", default=None)
                namelist = directory / "namelist.input"
                if not isinstance(spec, dict) or not namelist.is_file():
                    continue
                sounding = directory / "input_sounding"
                picture = spec.get("picture")
                rows.append({
                    "id": f"preset-{directory.name}", "kind": "preset",
                    "name": spec.get("name") or directory.name, "plain_name": spec.get("name") or directory.name,
                    "line": spec.get("line") or "", "source": spec.get("source") or "GM1 preset",
                    "picture": picture if isinstance(picture, str) and picture.startswith("/static/") else None,
                    "picture_caption": spec.get("picture_caption"),
                    "suggested_name": directory.name,
                    "path": str(namelist), "sounding": str(sounding) if sounding.is_file() else None,
                    "offered": spec.get("offered", True), "not_offered_reason": spec.get("not_offered_reason"),
                    "missing": None,
                })
        # A preset that is CM1's own namelist unchanged is shown once, as the preset.
        digests: dict[str, str] = {}
        unique = []
        for row in rows:
            digest = None
            if row["path"]:
                try:
                    digest = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
                except OSError:
                    digest = None
            row["sha256"] = digest
            if digest is not None and digest in digests and row["kind"] == "preset":
                unique = [kept for kept in unique if kept.get("sha256") != digest]
            elif digest is not None and digest in digests:
                continue
            if digest is not None:
                digests[digest] = row["id"]
            unique.append(row)
        return unique

    # The examples are priced in the background: GET /api/examples answers at once with what is known and
    # says whether a pricing round is under way; the screen asks again until it is not.  A round
    #
    # 1. asks nvidia-smi once (gpuinfo, no CUDA) for the GPU and its free memory now;
    # 2. keeps every price made for the same deck and sounding, GM1 install (version and estimator code), measured
    #    rates, GPU and free memory to FREE_BUCKET_BYTES (the key, :meth:`_key`), from this server or from the
    #    copy kept on disk by an earlier one;
    # 3. prices the others on PRICE_WORKERS host slots at once, each worker pricing its share of the imported
    #    examples in one ``gpuwm-cm1 estimate --each`` call read a line at a time (importing first, one by one,
    #    the ones not yet imported), so each price lands as it is made.  Every line holds what the example's own
    #    command (``C.example_estimate_args``, still shown as its command) prints for it alone.

    def _example_dir(self, row: dict[str, Any]) -> Path:
        return self.api.root.server_dir / EXAMPLES_DIR / f"{row['id']}-{(row.get('sha256') or 'x')[:12]}"

    def _imported(self, row: dict[str, Any]) -> tuple[dict[str, Any] | None, Any]:
        """The example imported once into the server's folder: (import-report.json, the import's result or None)."""

        jobs = self.api.jobs
        cache = self._example_dir(row)
        report = read_json(cache / "import-report.json", default=None)
        imported = None
        if not isinstance(report, dict) or not (cache / "run.json").is_file():
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists():
                import shutil

                shutil.rmtree(cache, ignore_errors=True)
            imported = jobs.host(C.import_args(Path(row["path"]), cache,
                                               None if not row.get("sounding") else Path(row["sounding"])))
            report = read_json(cache / "import-report.json", default=None)
        return (report if isinstance(report, dict) else None), imported

    def _is_imported(self, row: dict[str, Any]) -> bool:
        cache = self._example_dir(row)
        return (cache / "run.json").is_file() and (cache / "deck.toml").is_file() and isinstance(
            read_json(cache / "import-report.json", default=None), dict)

    @staticmethod
    def _info(report: dict[str, Any] | None, imported: Any, estimate: Any, rc: int | None,
              command: str | None) -> dict[str, Any]:
        return {
            "report": report,
            "estimate": estimate if isinstance(estimate, dict) else None,
            "import_rc": None if imported is None else imported.rc,
            "estimate_rc": rc,
            "estimate_command": command,
            "error": (None if isinstance(report, dict) else
                      (imported.stderr[-600:] if imported is not None else "the example did not import")),
        }

    def _price(self, row: dict[str, Any]) -> dict[str, Any]:
        """Import the example once into the server's folder, and price it on this GPU, alone."""

        report, imported = self._imported(row)
        cache = self._example_dir(row)
        estimated = None
        if isinstance(report, dict) and (cache / "deck.toml").is_file():
            estimated = self.api.jobs.host(C.example_estimate_args(cache, self.api.root.path))
        return self._info(report, imported, None if estimated is None else estimated.json,
                          None if estimated is None else estimated.rc,
                          None if estimated is None else estimated.command)

    @staticmethod
    def _row_key(row: dict[str, Any]) -> str:
        return f"{row['id']}:{row.get('sha256')}"

    def _store(self, row: dict[str, Any], info: dict[str, Any], gpu: dict[str, Any] | None,
               from_disk: float | None = None) -> None:
        """Hold one example's price for this server (and for the next one, on disk, when it was priced here)."""

        key = self._row_key(row)
        deck_sha = _deck_sha(self._example_dir(row))
        price_key = self._key(deck_sha, gpu) if deck_sha is not None and info.get("estimate_rc") == 0 and \
            isinstance(info.get("estimate"), dict) else None
        with self._examples_lock:
            self._priced[key] = (time.monotonic(), info, price_key)
            self._priced_at[key] = time.time() if from_disk is None else from_disk
            self._confirmed.add(key)
        if price_key is not None and from_disk is None:
            self._disk_put(key, price_key, info)

    def _gpu_now(self) -> dict[str, Any] | None:
        """The GPU the examples are priced on and its free memory now, for the key: ``nvidia-smi`` through
        :mod:`..cm1run.gpuinfo` (no CUDA), as ``gpuwm-cm1 estimate`` asks it; the fake runner's made-up card."""

        if getattr(self.api.jobs.runner, "kind", "real") == "fake":
            free = os.environ.get("GM1_FAKE_GPU_FREE_BYTES") or "8000000000"
            return {"uuid": "fake-runner", "name": "Fake GPU", "free_bytes": int(float(free)),
                    "host": {"disk_free_bytes": int(500e9)}}
        try:
            from ..cm1run import gpuinfo

            gpu = gpuinfo.query_gpus().select(None)
            # The host as the estimate asks it (``--root``, estimate._host_for): RAM and the runs folder's disk.
            host = gpuinfo.query_host(self.api.root.path).record()
        except Exception:  # noqa: BLE001 - no answer: nothing is kept, everything is priced
            return None
        return None if gpu is None else {**gpu.record(), "host": host}

    def _price_all(self, rows: list[dict[str, Any]], refresh: bool = False) -> None:
        """One pricing round (see above)."""

        gpu = self._gpu_now()
        with self._examples_lock:
            self._last_gpu = gpu
        todo: list[dict[str, Any]] = []
        for row in rows:
            key = self._row_key(row)
            if not refresh and self._is_imported(row):
                deck_sha = _deck_sha(self._example_dir(row))
                want = None if deck_sha is None else self._key(deck_sha, gpu)
                with self._examples_lock:
                    held = self._priced.get(key)
                    if want is not None and held is not None and held[2] == want and \
                            _verdicts_hold(held[1].get("estimate"), gpu):
                        self._priced[key] = (time.monotonic(), held[1], held[2])
                        self._confirmed.add(key)
                        continue
                kept = None if want is None else self._disk_get(key, want)
                if kept is not None and _verdicts_hold(kept[1][0], gpu):
                    self._store(row, self._info(read_json(self._example_dir(row) / "import-report.json",
                                                          default=None), None, *kept[1]), gpu, from_disk=kept[0])
                    continue
            todo.append(row)
        if not todo:
            return
        shares = [todo[i::PRICE_WORKERS] for i in range(PRICE_WORKERS)]
        workers = [threading.Thread(target=self._price_share, args=(share, gpu), daemon=True,
                                    name=f"gm1-gui-examples-{i}") for i, share in enumerate(shares) if share]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

    def _price_share(self, rows: list[dict[str, Any]], gpu: dict[str, Any] | None) -> None:
        """One worker's share: every imported example in one ``--each`` call, importing the others first."""

        batch: list[tuple[dict[str, Any], dict[str, Any] | None, Any]] = []
        for row in rows:
            if self._is_imported(row):
                batch.append((row, read_json(self._example_dir(row) / "import-report.json", default=None), None))
                continue
            self._price_batch(batch, gpu)
            batch = []
            try:
                report, imported = self._imported(row)
            except Exception as error:  # noqa: BLE001 - one example never hides the others
                self._store(row, {"report": None, "estimate": None, "error": str(error)}, gpu)
                continue
            if report is None or not (self._example_dir(row) / "deck.toml").is_file():
                self._store(row, self._info(report, imported, None, None, None), gpu)
                continue
            self._price_batch([(row, report, imported)], gpu)
        self._price_batch(batch, gpu)

    def _price_batch(self, batch: list[tuple[dict[str, Any], dict[str, Any] | None, Any]],
                     gpu: dict[str, Any] | None) -> None:
        if not batch:
            return
        root = self.api.root
        by_target = {str(self._example_dir(row)): (row, report, imported) for row, report, imported in batch}
        done: set[str] = set()

        def on_line(text: str) -> None:
            line = json.loads(text)
            entry = by_target.get(str(line.get("target")))
            if entry is None:
                return
            row, report, imported = entry
            command = C.display(C.example_estimate_args(self._example_dir(row), root.path))
            self._store(row, self._info(report, imported, line.get("payload"), line.get("rc"), command), gpu)
            done.add(str(line.get("target")))

        try:
            self.api.jobs.host_lines(C.example_estimate_each_args([self._example_dir(r) for r, _, _ in batch],
                                                                  root.path), on_line)
        except Exception:  # noqa: BLE001 - an older runner without --each: priced one by one below
            pass
        for target, (row, report, imported) in by_target.items():
            if target in done:
                continue
            # A runner that does not know --each, or a line that never came: the example's own command.
            try:
                info = self._price(row)
            except Exception as error:  # noqa: BLE001
                info = {"report": report, "estimate": None, "error": str(error)}
            self._store(row, info, gpu)

    def _start_pricing(self, rows: list[dict[str, Any]], refresh: bool = False) -> threading.Thread:
        """Start a pricing round over ``rows`` unless one is under way (a refresh asked for meanwhile runs
        right after it, in the same thread); returns the thread."""

        with self._examples_lock:
            if self._pricing is not None:
                if refresh:
                    self._again = (rows, True)
                return self._pricing
            self._confirmed = set()
            thread = threading.Thread(target=self._rounds, args=(rows, refresh), name="gm1-gui-examples",
                                      daemon=True)
            self._pricing = thread
            thread.start()
            return thread

    def _rounds(self, rows: list[dict[str, Any]], refresh: bool) -> None:
        while True:
            try:
                self._price_all(rows, refresh)
            except Exception:  # noqa: BLE001 - the next GET starts another round
                pass
            with self._examples_lock:
                if self._again is None:
                    self._pricing = None
                    return
                (rows, refresh), self._again = self._again, None
                self._confirmed = set()

    # ---- the key, and the prices kept on disk for the first answer after the server starts

    def _key(self, deck_sha: str, gpu: dict[str, Any] | None) -> str | None:
        """(deck and sounding, GM1 install, measured rates, GPU, free memory to FREE_BUCKET_BYTES) as one digest."""

        if not isinstance(gpu, dict) or gpu.get("free_bytes") is None:
            return None
        words = [deck_sha, *self._install_words(), *_machine_words(gpu)]
        return hashlib.sha256("\n".join(words).encode()).hexdigest()

    def _install_words(self) -> tuple[str, str, str]:
        """(package version, sha256 of the installed code, sha256 of this machine's measured rates).

        The code is every module of the ``cm1core`` package and its records (the rates the estimator reads
        among them), hashed once (about 12 MB, 25 ms); the rates file is hashed again only when it changes (a
        finished run or Measure speed adds to it).
        """

        try:
            from .. import __version__
            from ..cm1run.rates import cache_path

            rates = cache_path()
            stat = rates.stat() if rates.is_file() else None
        except Exception:  # noqa: BLE001 - a fake install still keys by the rest
            return "unknown", "unknown", "unknown"
        if self._code_digest is None:
            package = Path(__file__).resolve().parents[1]
            digest = hashlib.sha256()
            files = sorted(package.rglob("*.py")) + sorted((package / "records").glob("*.json"))
            for path in files:
                try:
                    digest.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes())
                except OSError:
                    continue
            self._code_digest = digest.hexdigest()
        mark = None if stat is None else (str(rates), stat.st_mtime_ns, stat.st_size)
        with self._disk_lock:
            if self._rates_mark is None or self._rates_mark[0] != mark:
                try:
                    words = "none" if stat is None else hashlib.sha256(rates.read_bytes()).hexdigest()
                except OSError:
                    words = "unreadable"
                self._rates_mark = (mark, words)
            rates_words = self._rates_mark[1]
        return __version__, self._code_digest, rates_words

    def _disk_path(self) -> Path:
        return self.api.root.server_dir / EXAMPLES_DIR / ESTIMATES_FILE

    def _disk_load(self) -> dict[str, Any]:
        if self._disk is None:
            data = read_json(self._disk_path(), default=None)
            ok = isinstance(data, dict) and data.get("schema") == ESTIMATES_SCHEMA
            self._disk = data if ok else {"schema": ESTIMATES_SCHEMA, "entries": {}, "last_gpu": None}
        return self._disk

    def _disk_put(self, key: str, price_key: str, info: dict[str, Any]) -> None:
        """Keep the estimate (not the import report, which stays in the example's folder) under its key."""

        with self._disk_lock:
            disk = self._disk_load()
            disk.setdefault("entries", {})[key] = {
                "key": price_key, "priced_at": self._priced_at.get(key) or time.time(),
                "estimate": info.get("estimate"), "estimate_rc": info.get("estimate_rc"),
                "estimate_command": info.get("estimate_command")}
            disk["last_gpu"] = self._last_gpu
            try:
                path = self._disk_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(disk, sort_keys=True), encoding="utf-8")
                tmp.replace(path)
            except OSError:
                pass

    def _disk_get(self, key: str, price_key: str | None) -> tuple[float, tuple[Any, Any, Any]] | None:
        """(priced_at, (estimate, rc, command)) kept on disk for ``key`` under ``price_key``, or None."""

        if price_key is None:
            return None
        with self._disk_lock:
            entry = (self._disk_load().get("entries") or {}).get(key)
        if not isinstance(entry, dict) or entry.get("key") != price_key or not isinstance(entry.get("estimate"), dict):
            return None
        return float(entry.get("priced_at") or 0.0), (entry["estimate"], entry.get("estimate_rc"),
                                                      entry.get("estimate_command"))

    def examples(self, refresh: bool = False, wait: bool = False) -> dict[str, Any]:
        rows = self.example_rows()
        wanted = [row for row in rows if row["path"]]
        now = time.monotonic()
        with self._examples_lock:
            stale = refresh or any(
                (held := self._priced.get(self._row_key(row))) is None or now - held[0] > EXAMPLE_PRICE_TTL_S
                for row in wanted)
        thread = self._start_pricing(wanted, refresh) if stale else self._pricing
        if wait and thread is not None:
            thread.join(timeout=WAIT_LIMIT_S)
            with self._examples_lock:
                missing = [row for row in wanted if self._row_key(row) not in self._priced]
            if missing:
                # A round that was already under way may have started before this request's rows changed.
                self._start_pricing(wanted, refresh).join(timeout=WAIT_LIMIT_S)
        # One look at the round's state for every row: a price stored between reading "confirmed" and reading the
        # price itself would otherwise show a fresh price as "checking".
        with self._examples_lock:
            pricing = self._pricing is not None
            last_gpu = self._last_gpu
            confirmed = set(self._confirmed)
            held_now = dict(self._priced)
            at_now = dict(self._priced_at)
        priced: dict[str, dict[str, Any]] = {}
        states: dict[str, tuple[str, float | None]] = {}
        for row in wanted:
            key = self._row_key(row)
            held = held_now.get(key)
            at = at_now.get(key)
            if held is not None:
                priced[row["id"]] = held[1]
                states[row["id"]] = ("checking" if pricing and key not in confirmed else "priced", at)
                continue
            kept = None
            if pricing:
                # Before this round's first answer: the price kept on disk for the GPU as last seen.
                deck_sha = _deck_sha(self._example_dir(row))
                if deck_sha is not None:
                    if last_gpu is None:
                        with self._disk_lock:
                            last_gpu = self._disk_load().get("last_gpu")
                    kept = self._disk_get(key, self._key(deck_sha, last_gpu))
            if kept is not None:
                report = read_json(self._example_dir(row) / "import-report.json", default=None)
                priced[row["id"]] = self._info(report if isinstance(report, dict) else None, None, *kept[1])
                states[row["id"]] = ("kept", kept[0])
            else:
                states[row["id"]] = ("pricing" if pricing else "not-priced", None)
        gpu = None
        out = []
        for row in rows:
            info = priced.get(row["id"]) or {}
            report = info.get("report") or {}
            estimate = info.get("estimate")
            view = estimate_view(estimate)
            values = (report.get("setup") or {}).get("values") or {}
            if view and view.get("gpu") and gpu is None:
                gpu = view["gpu"]
            gpu_row = next((r for r in (view or {}).get("rows", []) if r["id"] == "gpu"), None)
            guided = report.get("guided") or {}
            offered = bool(guided.get("offered")) and row.get("offered", True) is not False and report.get("exit") == 0
            why_not = None
            if row.get("missing"):
                offered, why_not = False, "This example is missing from the installation."
            elif row.get("offered") is False:
                why_not = row.get("not_offered_reason") or "Not offered yet."
            elif not guided.get("offered", True):
                why_not = guided.get("why_not")
            elif report and report.get("exit") != 0:
                why_not = report.get("verdict_text")
            fits_now = None if view is None else view.get("fits")
            if offered and fits_now is False:
                need = gpu_row.get("need_bytes") if gpu_row else None
                have = gpu_row.get("have_bytes") if gpu_row else None
                why_not = (f"Needs {_gb(need)} of free GPU memory; yours has {_gb(have)}." if need and have is not None
                           else "It does not fit on this computer now: see Estimate.")
            out.append({
                "id": row["id"], "kind": row["kind"], "name": row["name"], "plain_name": row["plain_name"],
                "line": row["line"], "source": row["source"], "picture": row.get("picture"),
                "picture_caption": row.get("picture_caption"), "suggested_name": row["suggested_name"],
                "path": row["path"], "sounding": row.get("sounding"),
                "grid": {key: values.get(key) for key in ("nx", "ny", "nz", "dx", "dy", "dz")} if values else None,
                "duration_s": values.get("timax"),
                "microphysics": (report.get("component") or {}).get("scheme"),
                "cells": ((estimate or {}).get("run") or {}).get("cells"),
                "verdict": report.get("verdict"), "verdict_text": report.get("verdict_text"),
                "exit": report.get("exit"),
                # Withheld as guided mode withholds it (not a matter of this computer's GPU).
                "not_offered": (row.get("not_offered_reason") or "Not offered yet.") if row.get("offered") is False
                else None,
                "guided": {"offered": offered and fits_now is not False, "why_not": why_not,
                           "reasons": guided.get("reasons") or []},
                "estimate": None if view is None else {
                    "fits": fits_now, "gpu_need_bytes": gpu_row.get("need_bytes") if gpu_row else None,
                    "gpu_free_bytes": gpu_row.get("have_bytes") if gpu_row else None,
                    "smallest_card_gb": smallest_card_gb(gpu_row.get("need_bytes") if gpu_row else None),
                    "wall": view["wall"], "disk_bytes": next(
                        (r["need_bytes"] for r in view["rows"] if r["id"] == "disk"), None),
                },
                "error": info.get("error"),
                "price": {"state": states.get(row["id"], ("not-priced", None))[0],
                          "priced_at": _iso(states.get(row["id"], ("", None))[1])},
                "commands": {
                    "import": C.display(C.import_args(Path(row["path"]) if row["path"] else Path("NAMELIST"),
                                                      self.api.root.path / row["suggested_name"],
                                                      Path(row["sounding"]) if row.get("sounding") else None)),
                    "estimate": info.get("estimate_command"),
                },
            })
        return {"root": str(self.api.root.path), "gpu": gpu, "examples": out, "pricing": pricing,
                "note": ("CM1's two examples ship with GM1 unchanged, under UCAR's notice. The wall times are "
                         "estimated from GPU step times GM1 has measured; the basis is under each one.")}

    # -------------------------------------------------------------- reads

    def setup(self, run_id: str, rundir: Path) -> dict[str, Any]:
        api = self.api
        report = read_json(rundir / "import-report.json", default=None)
        report = report if isinstance(report, dict) else {}
        estimate = read_json(rundir / "estimate.json", default=None)
        geometry = read_json(rundir / "geometry.json", default=None)
        deck_time = _mtime(rundir / "deck.toml")
        report_time = _mtime(rundir / "import-report.json")
        latest_import = max((t for t in (deck_time, report_time) if t is not None), default=None)
        estimate_time = _mtime(rundir / "estimate.json")
        stale = estimate is None or _older_than_the_deck(rundir, estimate, estimate_time, latest_import)
        geometry_time = _mtime(rundir / "geometry.json")
        geometry_stale = geometry is None or _older_than_the_deck(rundir, geometry, geometry_time, latest_import)
        setup = report.get("setup") or {}
        values = setup.get("values") or {}
        sections = report.get("sections") or []
        counts = {section.get("id"): section.get("count", len(section.get("rows") or [])) for section in sections}
        status = api.status(run_id, rundir)
        exit_code = report.get("exit")
        has_run = bool(segment_dirs(rundir))
        steps = self._steps(exit_code, estimate is not None and not stale, has_run)
        settings_toml = _toml(rundir / "settings.toml")
        diff = _read_text(rundir / "namelist.diff")
        fit = read_json(rundir / "fit.json", default=None)
        acknowledgments = report.get("acknowledgments") or {}
        plain = plain_summary(values, estimate if not stale else None)
        sounding = sounding_view(report, geometry if isinstance(geometry, dict) else None)
        sounding["card"] = self._sounding_card(run_id, rundir)
        return {
            "id": run_id,
            "url": quote(run_id, safe=""),
            "path": str(rundir),
            "mode": api.mode()["mode"],
            "run": read_json(rundir / "run.json", default=None),
            "status": status,
            "has_run": has_run,
            "editable": not has_run and bool(report) and (rundir / "namelist.input.original").is_file(),
            "report": {
                "present": bool(report),
                "schema": report.get("schema"),
                "fake_runner": bool(report.get("fake_runner")),
                "verdict": report.get("verdict"),
                "verdict_text": report.get("verdict_text"),
                "exit": exit_code,
                "counts": counts,
                "acknowledgments": acknowledgments,
                "component": report.get("component"),
                "guided": report.get("guided"),
                "output": report.get("output"),
                "first_refused": report.get("first_refused"),
                "notes": report.get("notes") or [],
                "not_r21": report.get("not_r21"),
                "aliases": report.get("aliases") or [],
                "namelist": report.get("namelist"),
                "command": report.get("command"),
                "error": report.get("error"),
            },
            "setup": setup,
            "values": values,
            "settings_toml": settings_toml,
            "sounding": sounding,
            "estimate": estimate_view(estimate) if isinstance(estimate, dict) else None,
            "estimate_stale": stale,
            "geometry": geometry if isinstance(geometry, dict) else None,
            "geometry_stale": geometry_stale,
            "diff": diff,
            "fit": fit if isinstance(fit, dict) else None,
            "plain": plain,
            "steps": steps,
            "commands": {
                "start": C.display(C.run_args(rundir, api.native)),
                "estimate": C.display(C.estimate_args(rundir)),
                "check": C.display(C.check_args(rundir / "deck.toml")),
                "report_text": f"cat {rundir / 'import-report.txt'}",
                "jobscript": C.display(C.jobscript_args(rundir, "slurm", native=api.native)),
                "measure": C.display(C.measure_args(rundir, api.native)),
            },
            "commands_log": f"/api/runs/{quote(run_id, safe='')}/files/{COMMANDS_LOG}"
            if (rundir / COMMANDS_LOG).is_file() else None,
            "files": {
                "report_text": f"/api/runs/{quote(run_id, safe='')}/files/import-report.txt"
                if (rundir / "import-report.txt").is_file() else None,
                "report_json": f"/api/runs/{quote(run_id, safe='')}/files/import-report.json"
                if (rundir / "import-report.json").is_file() else None,
                "diff": f"/api/runs/{quote(run_id, safe='')}/files/namelist.diff"
                if (rundir / "namelist.diff").is_file() else None,
            },
        }

    #: Quick-look products that draw the run's sounding (the skew-T and hodograph card), by id.
    SOUNDING_CARDS = ("sounding", "sounding-t0", "environment-card", "skewt", "column-skewt")

    @classmethod
    def _sounding_card(cls, run_id: str, rundir: Path) -> dict[str, Any] | None:
        """The skew-T card the quick-look worker drew for this run, when it has drawn one."""

        index = read_json(rundir / "quicklook" / "index.json", default=None)
        products = index.get("products") if isinstance(index, dict) else None
        for product in products if isinstance(products, list) else []:
            if not isinstance(product, dict) or product.get("id") not in cls.SOUNDING_CARDS:
                continue
            frames = [f for f in product.get("frames") or [] if isinstance(f, dict) and isinstance(f.get("file"), str)]
            if frames:
                return {"title": product.get("title") or "Sounding",
                        "url": f"/api/runs/{quote(run_id, safe='')}/files/quicklook/{quote(frames[0]['file'])}"}
        # The quick-look draws the skew-T on request (POST column: quicklook/slices/column-WHERE-tNNNNNN.png), with the
        # run's starting sounding beside the column; the earliest one drawn stands for the sounding.
        drawn = []
        for path in (rundir / "quicklook" / "slices").glob("column-*.png"):
            match = re.search(r"-t(\d+)(?:p(\d+))?\.png$", path.name)
            if match:
                drawn.append((float(f"{match.group(1)}.{match.group(2) or 0}"), path.name))
        if drawn:
            time_s, name = min(drawn)
            return {"title": f"Column skew-T at T+{int(time_s) // 3600}:{int(time_s) % 3600 // 60:02d}:{int(time_s) % 60:02d}",
                    "url": f"/api/runs/{quote(run_id, safe='')}/files/quicklook/slices/{quote(name)}"}
        return None

    @staticmethod
    def _steps(exit_code: Any, estimated: bool, has_run: bool) -> list[dict[str, Any]]:
        """The five setup steps and whether each is done (the numbered row on screens 3-7)."""

        runs = exit_code in (0, 3)
        rows = []
        for index, (step, label) in enumerate((("check", "Check"), ("sounding", "Sounding"),
                                               ("domain", "Domain and time"), ("settings", "Settings"),
                                               ("estimate", "Estimate")), start=1):
            if step == "check":
                state = "done" if exit_code == 0 else "blocked" if exit_code == 2 else "todo"
            elif step == "estimate":
                state = "done" if estimated else "todo" if runs else "blocked"
            else:
                state = "open" if runs else "blocked"
            rows.append({"id": step, "number": index, "label": label, "state": state})
        return rows

    def namelist(self, run_id: str, rundir: Path) -> dict[str, Any]:
        text = _read_text(rundir / "namelist.input")
        original = _read_text(rundir / "namelist.input.original")
        if text is None:
            from .api import ApiError

            raise ApiError(404, "This run has no namelist.input.", "Import a namelist to make a run folder.")
        return {
            "id": run_id,
            "text": text,
            "original": original,
            "edited": original is not None and original != text,
            "diff": _read_text(rundir / "namelist.diff"),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "editable": not segment_dirs(rundir),
        }

    # -------------------------------------------------------------- actions

    def _require_editable(self, rundir: Path, command: str) -> None:
        from .api import ApiError

        if segment_dirs(rundir):
            raise ApiError(409, "This run has already run, so its namelist is fixed.",
                           "Import the namelist into a new run to change it.", command=command)
        if not (rundir / "namelist.input.original").is_file():
            raise ApiError(409, "This folder has no imported namelist.",
                           "Import a namelist to make a run folder.", command=command)

    def _edit_reply(self, action: str, run_id: str, rundir: Path, args: list[str], message_ok: str,
                    **extra: Any) -> "Reply":
        from .api import Reply

        line = C.display(args)
        self._require_editable(rundir, line)
        result = self.api.jobs.host(args)
        self.api._log(rundir, action, result.command, f"rc {result.rc}")
        payload = result.json if isinstance(result.json, dict) else None
        verdicts = {
            0: ("runnable", message_ok, ""),
            3: ("needs-ok", message_ok + " It runs once you OK the settings GM1 does not read.",
                "Open Check, read the list, and tick the box."),
            2: ("refused", "Saved, but GM1 cannot run the namelist as it is now.",
                "Open Check: each refused setting says what to change."),
        }
        if payload is None or payload.get("status") == "error":
            error = None
            try:
                error = json.loads(result.stderr[result.stderr.find("{"):]) if "{" in result.stderr else None
            except ValueError:
                error = None
            reason = (error or {}).get("reason") or (result.stderr.strip().splitlines() or ["The change was not made."])[-1]
            body = {
                "ok": False, "action": action, "run": run_id, "command": result.command, "rc": result.rc,
                "verdict": "error", "message": "The change was not made.", "fix": reason,
                "output": result.as_dict(), **extra,
            }
            return Reply(200, body)
        verdict, message, fix = verdicts.get(result.rc, ("error", "The change did not finish.", "Read the output."))
        edit = payload.get("edit") or {}
        diff_path = rundir / "namelist.diff"
        body = {
            "ok": result.rc in (0, 3),
            "action": action,
            "run": run_id,
            "command": result.command,
            "rc": result.rc,
            "verdict": verdict,
            "verdict_text": payload.get("verdict_text"),
            "exit": payload.get("exit"),
            "message": message,
            "fix": fix,
            "diff": edit.get("diff"),
            "settings_changed": edit.get("settings") or [],
            "first_refused": payload.get("first_refused"),
            # The six categories as the import just wrote them, for the page to draw in place.
            "report": report_view(payload),
            # namelist.diff: this run's namelist against your original file, as stored in the run folder.
            "stored_diff": _read_text(diff_path) if diff_path.is_file() else None,
            "output": result.as_dict(),
            **extra,
        }
        return Reply(200, body)

    def edit(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError

        sets = payload.get("set") or {}
        settings = payload.get("settings") or {}
        resets = payload.get("reset") or []
        seed = payload.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed < 2**64):
            raise ApiError(400, "The random seed must be a whole number from 1 to 2**64-1.", "")
        if not isinstance(sets, dict) or not isinstance(settings, dict) or not isinstance(resets, list):
            raise ApiError(400, "set and settings must be objects, reset a list.", "")
        for key in sets:
            if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", key):
                raise ApiError(400, f"{key!r} is not a namelist word.", "Use CM1's own lower-case name.")
        for key in list(settings) + list(resets):
            if not isinstance(key, str) or not re.fullmatch(r"[a-z]+\.[a-z0-9_]{1,40}", key):
                raise ApiError(400, f"{key!r} is not a setting name.", "Use TABLE.KEY, e.g. initialization.zc_m.")
        for value in list(sets.values()) + list(settings.values()):
            if not isinstance(value, (bool, int, float, str)) or (isinstance(value, str) and len(value) > 200):
                raise ApiError(400, "Each value must be a number, true/false, or a short text.", "")
        if not sets and not settings and not resets and seed is None:
            raise ApiError(400, "Nothing to change.", "Give set, settings, reset or seed.")
        args = C.edit_args(rundir, sets=sets, settings=settings, resets=[str(key) for key in resets], seed=seed)
        return self._edit_reply("edit", run_id, rundir, args, "Saved and checked again.")

    def recommended(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, Reply

        geometry = read_json(rundir / "geometry.json", default=None)
        if not isinstance(geometry, dict):
            estimate = read_json(rundir / "estimate.json", default=None)
            geometry = (estimate or {}).get("storm_track") if isinstance(estimate, dict) else None
        pair = (geometry or {}).get("recommended_pair") if isinstance(geometry, dict) else None
        checked = geometry.get("status") if isinstance(geometry, dict) else None
        if checked and checked != "evaluated":
            # The check ran and does not apply (a start other than CM1's warm bubble, for one): it says why.
            raise ApiError(409, "There is no recommended frame and bubble for this run.",
                           str(geometry.get("reason") or f"The storm-track check could not run ({checked})."),
                           command=C.display(C.estimate_args(rundir)))
        if not isinstance(pair, dict) or checked != "evaluated":
            raise ApiError(409, "There is no recommendation yet: the storm-track check has not run on this run.",
                           "Open Estimate (or press Estimate again) to run the check; it takes a second.",
                           command=C.display(C.estimate_args(rundir)))
        # The pair belongs to the namelist it was worked out for: once the run is imported again (an edit, the
        # editor's Save), a grid, time or wind change can move it, so an older one is not written.
        latest_import = max((t for t in (_mtime(rundir / "deck.toml"), _mtime(rundir / "import-report.json"))
                             if t is not None), default=None)
        worked_out = _mtime(rundir / "geometry.json") or _mtime(rundir / "estimate.json")
        record = read_json(rundir / "geometry.json", default=None)
        if not isinstance(record, dict) or "deck_file_sha256" not in record:
            record = read_json(rundir / "estimate.json", default=None)
        if _older_than_the_deck(rundir, record, worked_out, latest_import):
            raise ApiError(409, "The recommendation is older than the run's namelist, which has changed since.",
                           "Press Estimate again to work it out for the namelist as it is now.",
                           command=C.display(C.estimate_args(rundir)))
        report = read_json(rundir / "import-report.json", default=None) or {}
        values = ((report.get("setup") or {}).get("values")) or {}
        frame = pair.get("frame_m_s") or [None, None]
        bubble = pair.get("bubble_km") or [None, None]
        sets: dict[str, Any] = {"umove": float(frame[0]), "vmove": float(frame[1])}
        if values.get("imove") == 0:
            sets = {"imove": 1, **sets}
        settings: dict[str, Any] = {}
        if values.get("iinit", 1) == 1 and None not in bubble:
            settings = {"initialization.bubble_center_x_m": round(float(bubble[0]) * 1000.0, 1),
                        "initialization.bubble_center_y_m": round(float(bubble[1]) * 1000.0, 1)}
        elif values.get("iinit") == 12 and None not in bubble:
            # init3d.F:1372-1376: the updraft nudging's centre is where the storm starts.
            settings = {"initialization.xc_wnudge": round(float(bubble[0]) * 1000.0, 1),
                        "initialization.yc_wnudge": round(float(bubble[1]) * 1000.0, 1)}
        args = C.edit_args(rundir, sets=sets, settings=settings)
        if pair.get("uses_it"):
            return Reply(200, {"ok": True, "action": "recommended", "run": run_id, "command": C.display(args),
                               "rc": 0, "verdict": "unchanged", "diff": "",
                               "message": "The run already uses the recommended frame and bubble.", "fix": ""})
        return self._edit_reply("recommended", run_id, rundir, args,
                                "Moved the frame and the bubble as recommended, and checked the run again.",
                                recommended={"frame_m_s": frame, "bubble_km": bubble,
                                             "minimum_edge_clearance_km": pair.get("minimum_edge_clearance_km")})

    def save_namelist(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, NAMELIST_LIMIT

        if payload.get("revert") is True:
            # Back to the user's own file: the run folder keeps it, so the command names it there.
            original = rundir / "namelist.input.original"
            return self._edit_reply("namelist", run_id, rundir, C.edit_args(rundir, source=original),
                                    "Put your original namelist back and checked it again.")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(400, "The namelist is empty.", "Paste or type the namelist text.")
        data = text.encode("utf-8")
        if len(data) > NAMELIST_LIMIT:
            raise ApiError(413, "The namelist is larger than 1 MB.", "Check that this is a CM1 namelist.")
        stage = stage_upload(self.api.root.server_dir, "namelist.input", data, write=not self.api.jobs.dry_active())
        return self._edit_reply("namelist", run_id, rundir, C.edit_args(rundir, source=stage),
                                "Saved your namelist and checked it again.")

    def acknowledge(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError

        acknowledge = []
        if payload.get("output_keys"):
            acknowledge.append("output-keys")
        if payload.get("thompson"):
            acknowledge.append("thompson")
        if not acknowledge:
            raise ApiError(400, "Nothing to acknowledge.", "Send output_keys or thompson.")
        return self._edit_reply("acknowledge", run_id, rundir, C.edit_args(rundir, acknowledge=acknowledge),
                                "Noted. The run goes ahead without those settings.")

    def sounding(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import SOUNDING_LIMIT

        path = self.api._input_file(payload.get("sounding"), "input_sounding", SOUNDING_LIMIT,
                                    "sounding", required=True)
        return self._edit_reply("sounding", run_id, rundir, C.edit_args(rundir, sounding=path),
                                "Read the new sounding and checked the run again.")

    def jobscript(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, Reply

        scheduler = payload.get("scheduler") or "slurm"
        if scheduler not in ("slurm", "pbs"):
            raise ApiError(400, f"There is no {scheduler!r} scheduler here.", "Use slurm or pbs.")
        options: dict[str, Any] = {}
        for key in ("time", "partition", "account", "gpu_type", "qos", "mem"):
            value = payload.get(key)
            if value in (None, ""):
                continue
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,64}", value):
                raise ApiError(400, f"{key} {value!r} has characters a job script cannot take.",
                               "Use letters, digits and . _ : + -")
            options[key] = value
        if options.get("time") and not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", options["time"]):
            raise ApiError(400, "The time limit must be HH:MM:SS.", "For example 04:00:00.")
        if options.get("mem") and not re.fullmatch(r"\d{1,5}[GMgm]?", options["mem"]):
            raise ApiError(400, "Host memory must be a size such as 32G.", "For example 32G.")
        cpus = payload.get("cpus")
        if cpus is not None:
            if isinstance(cpus, bool) or not isinstance(cpus, int) or not 3 <= cpus <= 512:
                raise ApiError(400, "cpus must be a whole number from 3 (the run and its two helpers) to 512.", "")
            options["cpus"] = cpus
        until = payload.get("until_s")
        if until is not None:
            if isinstance(until, bool) or not isinstance(until, (int, float)) or until <= 0:
                raise ApiError(400, "until_s must be a storm time in seconds.", "")
            options["until_s"] = float(until)
        setup = payload.get("setup")
        if setup not in (None, [], ""):
            if isinstance(setup, str):
                setup = setup.splitlines()
            if not isinstance(setup, list) or len(setup) > JOBSCRIPT_SETUP_LINES or \
                    not all(isinstance(line, str) for line in setup):
                raise ApiError(400, f"setup must be up to {JOBSCRIPT_SETUP_LINES} lines of text.",
                               "One shell line each, for example module load cuda.")
            lines = [line.strip() for line in setup if line.strip()]
            for line in lines:
                # Each line goes into the script as written, one line each: no line breaks inside one, and
                # none that the command line would read as one of its own options.
                if len(line) > 300 or not line.isprintable() or line.startswith("-"):
                    raise ApiError(400, f"The line {line[:60]!r} cannot go into a job script.",
                                   "Give one shell command per line, up to 300 characters, "
                                   "for example module load cuda.")
            if lines:
                options["setup"] = lines
        python = payload.get("python")
        if python not in (None, ""):
            if not isinstance(python, str) or not re.fullmatch(r"[A-Za-z0-9_./+-]{1,256}", python) or \
                    python.startswith("-"):
                raise ApiError(400, f"python {python!r} is not a path a job script can use.",
                               "Give the full path of the node's Python, e.g. /opt/gm1/bin/python.")
            options["python"] = python
        write = payload.get("write", False)
        if not isinstance(write, bool):
            raise ApiError(400, "write must be true or false.", "")
        args = C.jobscript_args(rundir, scheduler, native=self.api.native, write=write, **options)
        result = self.api.jobs.host(args)
        self.api._log(rundir, "jobscript", result.command, f"rc {result.rc}")
        data = result.json if isinstance(result.json, dict) else {}
        script = data.get("script")
        ok = result.rc == 0 and isinstance(script, str)
        submit = "sbatch" if scheduler == "slurm" else "qsub"
        written = data.get("file") if ok and write else None
        return Reply(200, {
            "ok": ok, "action": "jobscript", "run": run_id, "command": result.command, "rc": result.rc,
            "script": script if ok else None, "scheduler": scheduler, "file": written,
            "submit": (f"{submit} {written}" if written else None),
            "message": ((f"Wrote {written}. Submit it on the cluster's login node with {submit}. " if written else
                         f"Save it on the cluster's login node and submit it with {submit}. ") +
                        "The two helpers that write CM1 files and pictures run on the GPU node's CPUs; this page "
                        "only reads the run folder." if ok else "The job script was not written."),
            "fix": "" if ok else ((result.stderr.strip().splitlines() or ["Read the output."])[-1]),
            "output": result.as_dict(),
        })

    def measure(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, Reply
        from .runstate import read_status

        args = C.measure_args(rundir, self.api.native)
        line = C.display(args)
        if not (rundir / "deck.toml").is_file():
            raise ApiError(409, "This run has no deck.toml, so there is nothing to measure.",
                           "Fix the refused settings on Check first.", command=line)
        with self.api.jobs.run_lock(run_id):
            if any(row.role == "measure" for row in self.api.jobs.launched(run_id)):
                raise ApiError(409, "Measure speed is already running for this run.", "Wait for it to finish.",
                               command=line)
            busy = []
            for path in self.api.root.iter_runs():
                other = self.api.root.run_id(path)
                state = read_status(path, launched_alive=self.api.jobs.engine_alive(other)).state
                if state in ("running", "waiting"):
                    busy.append(other)
            if busy or any(row.role == "measure" for row in self.api.jobs.launched()):
                raise ApiError(409, f"Your GPU is busy with {busy[0] if busy else 'another measurement'}.",
                               "Measure after it ends, so the numbers are this run's own.", command=line)
            self.api.jobs.require("estimate")
            row = self.api.jobs.launch(run_id, rundir, args, role="measure")
            self.api._log(rundir, "measure", row.command, "Measure speed: a few steps on the GPU")
        return Reply(200, {"ok": True, "action": "measure", "run": run_id, "command": row.command,
                           "pid": row.process.pid, "rc": None,
                           "message": "Measuring: a few steps on your GPU, about a minute with start-up.",
                           "fix": ""})

    def measure_status(self, run_id: str, rundir: Path) -> dict[str, Any]:
        rows = [row for row in self.api.jobs.launched(run_id, alive_only=False) if row.role == "measure"]
        if not rows:
            return {"id": run_id, "state": "none"}
        row = rows[-1]
        tail = ""
        try:
            tail = row.log.read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            pass
        rc = row.process.poll()
        state = "running" if rc is None else "done" if rc == 0 else "failed"
        return {"id": run_id, "state": state, "rc": rc, "command": row.command,
                "elapsed_s": round(time.time() - row.started, 1), "log_tail": tail,
                "measured": _measured(row.log) if state == "done" else None}

__all__ = ["CM1_EXAMPLES", "SetupApi", "duration_words", "estimate_view", "plain_summary", "report_view",
           "smallest_card_gb", "sounding_view"]
