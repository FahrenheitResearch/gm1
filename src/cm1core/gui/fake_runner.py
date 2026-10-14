"""A stand-in ``gpuwm-cm1`` for GUI tests and screen work: no GPU, no model.

``python -m cm1core.gui.fake_runner ARGS`` takes the arguments of the real
commands the GUI runs and writes the same run-directory contract
and the same ``gm1.run-events/v1`` log, fast, with every number made up:

- ``import NAMELIST [--sounding F] --out RUNDIR [--acknowledge-output-keys] [--json]``:
  the byte copy, run.json, import-report.json/.txt, settings.toml and deck.toml.
  Exit 0 runnable, 2 refused (stretch_x/stretch_y, or a ptype other than
  3, 5 and 7), 3 needs an OK (ptype 3).
- ``edit RUNDIR [--set K=V] [--setting T.K=V] [--from F] ...``: the namelist changed and imported again.
- ``check DECK --json``, ``cartesian DECK --plan --output X``, ``estimate RUNDIR [--measure] --json``,
  ``status RUNDIR --json``, ``jobscript RUNDIR --scheduler S --json``.
- ``run RUNDIR`` and ``resume RUNDIR [--from-restart N] [--until T]``: a segment
  under ``segments/NNN`` with frames, restarts, a receipt and the event log,
  honouring ``control/stop``, ``control/pause`` and SIGTERM as the runtime does.
- ``cm1out RUNDIR [--follow]`` and ``quicklook RUNDIR [--follow]``: the sidecars,
  writing placeholder files and small PNGs.  ``quicklook`` of CM1 netCDF output (a folder of
  ``cm1out_NNNNNN.nc`` or one ``cm1out.nc``, with ``--out``) gives each file a made-up saved time.
- ``stop`` and ``pause`` are the real commands (:mod:`.control`).

Every file it writes says ``fake_runner``.  Speed and outcome come from the
environment: ``GM1_FAKE_STEP_S`` (wall seconds per step, default 0.004),
``GM1_FAKE_STARTUP_S`` (0.3), ``GM1_FAKE_WAIT_GPU_S`` (0),
``GM1_FAKE_LAUNCH_DELAY_S`` (0: seconds before run or resume writes anything,
as a real run spends importing), ``GM1_FAKE_IGNORE_CONTROL`` (1: never read
control/, like a run still compiling, so only SIGTERM stops it), and
``GM1_FAKE_OUTCOME`` (empty, ``error``, ``courant-stop`` or ``vanish``: exit
halfway with no end record).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import sys
import time
import tomllib
from typing import Any, Mapping
import zlib

from . import control
from .files import write_json_atomic
from .runstate import proc_start_ticks, read_status, segment_dirs, this_host, iter_records

COMMANDS = ("import", "edit", "check", "cartesian", "estimate", "status", "jobscript", "run", "resume", "stop",
            "pause", "cm1out", "quicklook", "preset", "doctor", "reproduce",
            "render",  # the native renderer's picture of a restart: a placeholder here
            "sounding",  # the real sounding command (host-only; reads a WRF or ArWen history)
            "chain")  # the real chain command (host-only; reads a WRF or ArWen history, writes a run folder)
FAKE = "fake runner: nothing here ran a model; every number is made up"
#: The real quick-looks (cm1core.quicklook.products.PRODUCTS), ids and titles.
PRODUCTS = (
    ("reflectivity", "Reflectivity at 1 km"),
    ("spin-near-ground", "Spin near the ground"),
    ("updraft-5km", "Updraft at 5 km"),
    ("swath", "Wind swath since the start"),
    ("surface", "Lowest level \u03b8\u2032 and wind"),
    ("rain", "Accumulated rain"),
)
#: The quick-looks of a hurricane run (CM1's swirl start, ``iinit = 7``): its own wind and pressure looks in place
#: of the looks made for one thunderstorm (cm1core.quicklook.products.HURRICANE_ONLY and THUNDERSTORM_ONLY).
HURRICANE_PRODUCTS = (
    ("reflectivity", "Reflectivity at 1 km"),
    ("wind-lowest", "Wind at the lowest level"),
    ("pressure-lowest", "Pressure at the lowest level"),
    ("rain", "Accumulated rain"),
    ("swirl-by-radius", "Swirling wind by radius and height"),
    ("warm-core", "Warm core by radius and height"),
)
MICROPHYSICS = {5: "morrison-mp10", 7: "wsm6-mp6", 3: "thompson-mp8"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _refuse(code: str, reason: str, remedy: str, as_json: bool = True) -> int:
    document = {"status": "refused", "code": code, "gate": "fake-runner", "reason": reason,
                "remedy": remedy, "fake_runner": True}
    print(json.dumps(document, indent=2, sort_keys=True) if as_json else f"gpuwm-cm1: {reason} Remedy: {remedy}",
          file=sys.stderr)
    return 2


# ------------------------------------------------------------ namelist

_ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\n]+)")


def _value(text: str) -> Any:
    text = text.strip().rstrip(",").strip()
    lowered = text.lower()
    if lowered in (".true.", "t", ".t."):
        return True
    if lowered in (".false.", "f", ".f."):
        return False
    if text[:1] in ("'", '"'):
        return text.strip("'\"")
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text.replace("d", "e").replace("D", "e"))
    except ValueError:
        return text


def parse_namelist(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.split("!", 1)[0]
        if line.strip().startswith(("&", "/")):
            continue
        for key, value in _ASSIGN.findall(line):
            values.setdefault(key.lower(), _value(value))
    return values


def _deck_numbers(settings: dict[str, Any]) -> dict[str, Any]:
    def number(key: str, default: float) -> float:
        value = settings.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default

    rst = number("rstfrq", -3600.0)
    return {
        "nx": int(number("nx", 120)), "ny": int(number("ny", 120)), "nz": int(number("nz", 40)),
        "dx_m": number("dx", 1000.0), "dy_m": number("dy", 1000.0), "dz_m": number("dz", 500.0),
        "dt_s": number("dtl", 6.0), "duration_s": number("timax", 7200.0),
        "history_interval_s": number("tapfrq", 900.0),
        "restart_interval_s": rst if rst > 0 else None,
        "ptype": int(number("ptype", 5)),
        "isnd": int(number("isnd", 7)),
    }


# ------------------------------------------------------------ import and host calls

_SETUP_DEFAULTS = {
    "nx": 120, "ny": 120, "nz": 40, "dx": 1000.0, "dy": 1000.0, "dz": 500.0, "stretch_z": 0, "ztop": 18000.0,
    "dtl": 6.0, "timax": 7200.0, "adapt_dt": 0, "tapfrq": 900.0, "rstfrq": -3600.0, "statfrq": 60.0,
    "prclfrq": 60.0, "imove": 0, "umove": 0.0, "vmove": 0.0, "iorigin": 1, "wbc": 2, "ebc": 2, "sbc": 2,
    "nbc": 2, "bbc": 1, "tbc": 1, "isnd": 7, "iwnd": 0, "iinit": 1, "irandp": 0, "iprcl": 0, "imoist": 1,
    "ptype": 5, "ihail": 1, "sgsmodel": 1, "output_format": 1, "output_filetype": 1,
}
#: init3d.F's per-arm named parameters, as the fake shows them (the real importer reads config.py's table).
_ARMS = {
    1: {"zc_m": 1400.0, "bhrad_m": 10000.0, "bvrad_m": 1400.0, "bptpert_k": 1.0, "maintain_rh": False},
    2: {"zdep_m": 2500.0, "bptpert_k": -6.0, "maintain_rh": True},
    8: {"zc_m": 1500.0, "bhrad_m": 10000.0, "bvrad_m": 1500.0, "bptpert_k": 2.0, "maintain_rh": False,
        "random_line_amplitude_k": 0.2},
}
_PARCEL_DEFAULTS = {"seed_nx": 60, "seed_ny": 60, "seed_nz": 10, "seed_dx_m": 2000.0, "seed_dy_m": 2000.0,
                    "seed_dz_m": 1000.0, "seed_x0_m": 0.0, "seed_y0_m": 0.0, "seed_z0_m": 0.0}
_FAKE_WORDS = ("output_rain", "output_sws", "output_th", "output_thpert", "output_qv", "output_dbz",
               "output_u", "output_v", "output_w", "output_km", "output_uh", "output_cape")


def _user_settings(path: Path) -> dict[str, Any]:
    from ..cm1import import usersettings

    return usersettings.read_file(path) if path.is_file() else {}


def _setup_block(values: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    from ..cm1import import usersettings

    groups = []
    for gid, title, selector, arm in (("bubble", "Warm bubble", "iinit", 1), ("coldpool", "Cold pool", "iinit", 2),
                                      ("line", "Line thermal with noise", "iinit", 8),
                                      ("parcels", "Parcels", "iprcl", 1)):
        active = values.get(selector) == arm
        fields = []
        for item in usersettings.EDITABLE:
            if gid not in item.groups:
                continue
            if item.table == "lagrangian":
                default = _PARCEL_DEFAULTS[item.name]
            elif item.name.startswith("bubble_center"):
                iorigin2 = values.get("iorigin") == 2
                extent = values["nx"] * values["dx"] if item.name.endswith("x_m") else values["ny"] * values["dy"]
                default = 0.0 if iorigin2 else 0.5 * extent
            else:
                default = _ARMS.get(arm, _ARMS[1]).get(item.name, 0.0)
            value = user.get(item.key, default) if active else default
            factor = 0.001 if item.unit == "m" else 1.0
            fields.append({"key": item.key, "label": item.label, "cm1": item.cm1, "kind": item.kind,
                           "unit": "km" if item.unit == "m" else item.unit, "factor": factor,
                           "value": value * factor if item.kind == "float" else value,
                           "cm1_value": default * factor if item.kind == "float" else default,
                           "changed": item.key in user})
        groups.append({"id": gid, "title": title, "plain": title.lower(), "selector": f"{selector} = {arm}",
                       "source": "fake runner", "active": active, "fields": fields})
    words = []
    for word in _FAKE_WORDS:
        value = int(values.get(word, 1 if word in ("output_rain", "output_th", "output_u", "output_v", "output_w")
                               else 0) or 0)
        if word == "output_km":
            state, label, why = "refused", "not available", "not kept between steps (fake runner)"
        elif word in ("output_th", "output_u", "output_v", "output_w"):
            state, label, why = "always", "always written", "in every frame (fake runner)"
        else:
            state, label, why = ("written", "written", "") if value else ("off", "off", "")
        words.append({"word": word, "value": value, "state": state, "label": label, "why": why, "fields": [],
                      "editable": state in ("written", "off"), "unvalidated": None})
    return {"schema": "gm1.cm1-setup/v1", "values": values, "fortran": groups, "output_words": words,
            "user_settings": user, "random_seed": None, "deck_initialization": {}, "note": FAKE}


def _import_into(source: Path, data: bytes, out: Path, *, sounding: str | None, acknowledged: list[str],
                 user: dict[str, Any], as_json: bool | None, command: str, original_bytes: bytes | None = None,
                 overrides: Mapping[str, Any] | None = None) -> int:
    """The fake import's files in ``out``; ``original_bytes`` and ``overrides`` as the real importer takes them
    (a guided preset's namelist, and the fit's run settings); ``as_json`` None prints nothing."""

    settings = parse_namelist(data.decode("utf-8", errors="replace"))
    numbers = _deck_numbers(settings)
    refused, needs_ok = [], []
    for key in ("stretch_x", "stretch_y"):
        if settings.get(key, 0) not in (0, False):
            refused.append({"key": key, "keys": [key], "value": str(settings[key]),
                            "text": "horizontal stretching is not in v0.1",
                            "plain": "GM1 cannot run a stretched horizontal grid yet.",
                            "fix": f"set {key} = 0 (a uniform grid)", "plain_fix": "Use a uniform grid."})
    ptype = numbers["ptype"]
    if ptype not in MICROPHYSICS:
        refused.append({"key": "ptype", "keys": ["ptype"], "value": str(ptype),
                        "text": f"ptype = {ptype} is not offered", "plain": "This rain and hail scheme is not offered.",
                        "fix": "use ptype = 5 (Morrison) or 7 (WSM6)", "plain_fix": "Pick Morrison or WSM6."})
    elif ptype == 3 and "thompson" not in acknowledged and "output-keys" not in acknowledged:
        needs_ok.append({"key": "ptype", "keys": ["ptype"], "value": "3", "acknowledge": "thompson",
                         "text": "Thompson has run only a 120-s column", "plain": "Thompson is barely checked."})
    rc = 2 if refused else 3 if needs_ok else 0
    out.mkdir(parents=True, exist_ok=True)
    original = out / "namelist.input.original"
    if not original.exists():
        original.write_bytes(data if original_bytes is None else original_bytes)
    if (out / "namelist.input").resolve() != source.resolve():
        (out / "namelist.input").write_bytes(data)
    if data != original.read_bytes():
        import difflib

        diff = difflib.unified_diff(original.read_text(errors="replace").splitlines(keepends=True),
                                    data.decode("utf-8", "replace").splitlines(keepends=True),
                                    fromfile="namelist.input.original", tofile="namelist.input")
        (out / "namelist.diff").write_text("".join(diff), encoding="utf-8")
    else:
        (out / "namelist.diff").unlink(missing_ok=True)
    if sounding:
        shutil.copyfile(sounding, out / "input_sounding")
    verdict = {0: "runs", 2: "refused", 3: "runs-after-acknowledgment"}[rc]
    verdict_text = {0: "Runs, with the differences below (fake runner)",
                    2: f"Refused: {len(refused)} setting{'s' if len(refused) != 1 else ''} must change before this runs",
                    3: "Runs after you acknowledge: accept Thompson's limited checking"}[rc]
    values = {key: settings.get(key, default) for key, default in _SETUP_DEFAULTS.items()}
    values.update({key: value for key, value in settings.items() if key.startswith("output_")})
    sections = [
        {"id": "refused", "label": "Refused", "count": len(refused), "rows": refused},
        {"id": "not_read", "label": "Not read", "count": len(needs_ok), "rows": needs_ok},
        {"id": "differs", "label": "Runs differently from CM1", "count": 1, "rows": [
            {"key": "output_km", "keys": ["output_km"], "value": str(settings.get("output_km", 0)),
             "text": "Not saved (fake runner)", "plain": "A turbulence field is not saved."}]},
        {"id": "not_checked", "label": "Not yet checked against CM1", "count": 0, "rows": []},
        {"id": "not_needed", "label": "Not needed on one GPU", "count": 0, "collapsed": True, "rows": []},
        {"id": "same", "label": "Same as CM1", "count": len(settings), "collapsed": True, "rows": [],
         "families": [{"family": "Everything else (fake runner)", "n": len(settings), "evidence": "none"}]},
    ]
    report = {
        "schema": "gm1.cm1-import-report/v1", "fake_runner": True, "note": FAKE, "title": "Check",
        "exit": rc, "exit_code": rc, "verdict": verdict, "verdict_text": verdict_text,
        "namelist": {"path": str(source), "name": source.name, "sha256": _sha256(data),
                     "original_sha256": _sha256(original.read_bytes()), "edited": data != original.read_bytes()},
        "version": "r21.1", "not_r21": "none", "settings_not_in_r21": [], "aliases": [],
        "component": {"ptype": ptype, "scheme": {5: "Morrison double-moment", 7: "WSM6", 3: "Thompson"}.get(ptype, "?"),
                      "component": MICROPHYSICS.get(ptype)},
        "first_refused": refused[0] if refused else None,
        "acknowledgments": {"needed": [row["acknowledge"] for row in needs_ok],
                            "given": [a for a in acknowledged if a in ("thompson",) and ptype == 3],
                            "missing": [row["acknowledge"] for row in needs_ok]},
        "sections": sections,
        "guided": {"offered": rc == 0, "why_not": None if rc == 0 else (refused or needs_ok)[0]["plain"],
                   "reasons": []},
        "output": {"format": "GrADS" if values["output_format"] == 1 else "netCDF",
                   "output_format": values["output_format"], "output_filetype": values["output_filetype"],
                   "text": f"output_format = {values['output_format']} (fake runner)", "also_netcdf": False},
        "sounding": {"isnd": values["isnd"], "reads_file": values["isnd"] in (7, 17), "path": sounding,
                     "sha256": None, "levels": 0, "surface": None, "top_m": None, "problems": [], "warnings": [],
                     "environment_status": "not-computed", "environment_reason": "fake runner"},
        "comparison": None, "check": {"passes": 1, "status": "passed"}, "settings": {}, "notes": [FAKE],
        "command": command, "run_directory": str(out),
        "categories": {"refused": refused, "not_read": needs_ok},
        "setup": _setup_block(values, user),
    }
    write_json_atomic(out / "import-report.json", report)
    lines = [f"CM1 namelist: {source}", f"Result: {verdict_text}",
             f"Refused ({len(refused)})   Not read ({len(needs_ok)})"]
    lines += [f"  {row['key']} = {row['value']}: {row['text']}; {row.get('fix', '')}" for row in refused]
    (out / "import-report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    toml = ["# fake runner", "schema = \"gm1.cm1-run-settings/v1\""]
    tables: dict[str, dict[str, Any]] = {}
    for key, value in [*user.items(), *(overrides or {}).items()]:
        table, name = key.split(".", 1)
        tables.setdefault(table, {})[name] = value
    tables.setdefault("output", {}).setdefault("also_netcdf", False)
    tables["output"]["format"] = values["output_format"]
    tables["output"]["filetype"] = values["output_filetype"]
    for table in sorted(tables):
        toml.append(f"\n[{table}]")
        toml += [f"{name} = {json.dumps(value)}" for name, value in sorted(tables[table].items())]
    (out / "settings.toml").write_text("\n".join(toml) + "\n", encoding="utf-8")
    if rc != 2:
        deck = ["# gpuwm-cm1.config/v1 written by the fake runner: not a real deck", "[fake]"]
        deck += [f"{key} = {json.dumps(value)}" for key, value in numbers.items() if value is not None]
        deck += [f"imove = {int(values['imove'])}", f"umove = {float(values['umove'])}",
                 f"vmove = {float(values['vmove'])}", f"iorigin = {int(values['iorigin'])}"]
        for key in ("initialization.bubble_center_x_m", "initialization.bubble_center_y_m"):
            if key in user:
                deck.append(f"{key.split('.', 1)[1]} = {float(user[key])}")
        (out / "deck.toml").write_text("\n".join(deck) + "\n", encoding="utf-8")
    else:
        (out / "deck.toml").unlink(missing_ok=True)
    previous = {}
    try:
        previous = json.loads((out / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    run = {
        "schema": "gm1.cm1-run/v1",
        "fake_runner": True,
        "name": previous.get("name") or out.name,
        "created_utc": previous.get("created_utc") or _utc(),
        "host": this_host(),
        "namelist_sha256": _sha256(data),
        "acknowledgments": report["acknowledgments"]["given"],
        "grid": {"nx": numbers["nx"], "ny": numbers["ny"], "nz": numbers["nz"],
                 "dx_m": numbers["dx_m"], "dy_m": numbers["dy_m"]},
        "segments": previous.get("segments") or [],
    }
    # A re-import (an edit) keeps what the guided fit wrote, as the real record_import does, and an edited guided
    # run says so in its case.
    for key in ("preset", "case", "guided"):
        if key in previous:
            run[key] = previous[key]
    from ..cm1run.rundir import mark_edited

    probe = {**run, "namelist": {"sha256": run["namelist_sha256"]}}
    mark_edited(out, probe)
    if "case" in probe:
        run["case"] = probe["case"]
    write_json_atomic(out / "run.json", run)
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif as_json is not None:
        print((out / "import-report.txt").read_text(encoding="utf-8"), end="")
    return rc


def cmd_import(arguments: argparse.Namespace) -> int:
    source = Path(arguments.namelist)
    out = Path(arguments.out)
    try:
        data = source.read_bytes()
    except OSError as error:
        return _refuse("CM1-INPUT-ERROR", f"cannot read {source}: {error}", "give a readable namelist")
    reimport = out.is_dir() and (out / "namelist.input.original").is_file() and not (out / "segments").exists()
    if out.exists() and any(out.iterdir()) and not reimport:
        return _refuse("CM1-IMPORT-OUT-EXISTS", f"{out} already holds files",
                       "import into a new folder")
    acknowledged = ["output-keys"] if arguments.acknowledge_output_keys else []
    if getattr(arguments, "acknowledge_thompson", False):
        acknowledged.append("thompson")
    user = _user_settings(out / "settings.toml") if reimport else {}
    sounding = arguments.sounding
    original_bytes = None
    if getattr(arguments, "wrf", None):
        # The real column (host-only), then this runner's import of the namelist with isnd = 7.
        from ..cm1import.wrfcolumn import extract_column, parse_pair
        from ..cm1run.fit import set_namelist_values
        from ..errors import CM1Refusal
        from ..wrf_history import open_history

        try:
            with open_history(arguments.wrf) as history:
                record = (history.record_of(arguments.wrf_time) if arguments.wrf_time
                          else history.check_record(int(arguments.wrf_record or 0)))
                point = ({"at": parse_pair(arguments.wrf_at, "--wrf-at")} if arguments.wrf_at
                         else {"ij": tuple(int(v) for v in parse_pair(arguments.wrf_ij, "--wrf-ij"))})
                column = extract_column(history, record, **point)
        except (CM1Refusal, ValueError) as error:
            detail = error.detail.as_dict() if isinstance(error, CM1Refusal) else {"reason": str(error)}
            return _refuse(detail.get("code", "CM1-IMPORT-DOOR"), detail["reason"], detail.get("remedy", ""))
        out.mkdir(parents=True, exist_ok=True)
        staged = out / ".wrf-column-input_sounding"
        staged.write_bytes(column.sounding_bytes())
        (out / "wrf-column.json").write_text(json.dumps(column.record(), indent=1, default=str), encoding="utf-8")
        sounding = str(staged)
        original_bytes = data
        data = set_namelist_values(data.decode("ascii"), {"isnd": 7}).encode("ascii")
        rc = _import_into(source, data, out, sounding=sounding, acknowledged=acknowledged, user=user,
                          as_json=arguments.json, command="gpuwm-cm1 import (fake runner)",
                          original_bytes=original_bytes)
        staged.unlink(missing_ok=True)
        return rc
    return _import_into(source, data, out, sounding=sounding, acknowledged=acknowledged, user=user,
                        as_json=arguments.json, command="gpuwm-cm1 import (fake runner)")


def cmd_edit(arguments: argparse.Namespace) -> int:
    from ..cm1import import usersettings

    rundir = Path(arguments.rundir)
    if not (rundir / "namelist.input.original").is_file():
        return _refuse("CM1-EDIT", f"{rundir} holds no imported namelist", "import one first")
    if (rundir / "segments").exists():
        return _refuse("CM1-EDIT", f"{rundir} has run; its namelist is fixed", "import into a new folder")
    target = rundir / "namelist.input"
    data = Path(arguments.source).read_bytes() if arguments.source else target.read_bytes()
    text = data.decode("utf-8", "replace")
    for assignment in arguments.sets:
        key, _, value = assignment.partition("=")
        key = key.strip().lower()
        pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^,\n/]*)")
        if pattern.search(text):
            text = pattern.sub(lambda match: match.group(1) + value.strip(), text, count=1)
        else:
            text = re.sub(r"(?m)^\s*/\s*$", f" {key} = {value.strip()},\n /", text, count=1)
    try:
        stored = _user_settings(rundir / "settings.toml")
        changes = dict(item.partition("=")[::2] for item in arguments.settings)
        user = usersettings.merge(stored, {k.strip(): v for k, v in changes.items()}, tuple(arguments.resets))
    except usersettings.SettingError as error:
        return _refuse("CM1-EDIT", str(error), "see the reason")
    target.write_text(text, encoding="utf-8")
    previous = {}
    try:
        previous = json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    acknowledged = [] if arguments.drop_acknowledgments else list(
        (previous.get("acknowledgments") or {}).get("given") or [])
    if arguments.acknowledge_output_keys:
        acknowledged.append("output-keys")
    if arguments.acknowledge_thompson:
        acknowledged.append("thompson")
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = _import_into(target, text.encode("utf-8"), rundir, sounding=arguments.sounding,
                          acknowledged=acknowledged, user=user, as_json=True,
                          command="gpuwm-cm1 edit (fake runner)")
    report = json.loads(buffer.getvalue())
    before = data if not arguments.source else target.read_bytes()
    report["edit"] = {"namelist_changed": True, "set": dict(item.partition("=")[::2] for item in arguments.sets),
                      "from": arguments.source, "diff": "(fake runner) " + " ".join(arguments.sets),
                      "settings": [{"key": k, "from": stored.get(k), "to": user.get(k)}
                                   for k in sorted(set(stored) | set(user)) if stored.get(k) != user.get(k)],
                      "acknowledged": acknowledged}
    del before
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report["verdict_text"])
    return rc


def cmd_preset(arguments: argparse.Namespace) -> int:
    """``preset list|show|fit`` are the real commands (host-only, no GPU); ``new`` fits for real and
    imports with this runner's stand-in import, so ``run`` finds the fake deck it runs."""

    from ..cm1import import usersettings
    from ..cm1run import presets

    def importer(namelist: Path, rundir: Path, *, overrides: dict[str, Any], original: bytes,
                 command: str, sounding: Path | None = None) -> tuple[int, dict[str, Any]]:
        # As the real importer does: an override that is a user setting (by its deck key) is kept as one.
        extra = dict(overrides or {})
        user = {item.key: extra.pop(item.deck) for item in usersettings.EDITABLE
                if item.deck is not None and item.deck in extra}
        given = None if sounding is None else str(sounding)
        rc = _import_into(namelist, namelist.read_bytes(), rundir, sounding=given, acknowledged=[],
                          user=usersettings.check(user), as_json=None, command=command,
                          original_bytes=original, overrides=extra)
        return rc, json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))

    return presets.preset_door(arguments, importer=importer)


def cmd_jobscript(arguments: argparse.Namespace) -> int:
    rundir = Path(arguments.rundir).resolve()
    if arguments.scheduler not in ("slurm", "pbs"):
        return _refuse("CM1-JOBSCRIPT", f"no scheduler {arguments.scheduler!r}", "use slurm or pbs")
    head = (["#SBATCH --job-name=" + rundir.name, "#SBATCH --gres=gpu:1", f"#SBATCH --time={arguments.time}"]
            if arguments.scheduler == "slurm" else
            ["#PBS -N " + rundir.name, "#PBS -l select=1:ngpus=1", f"#PBS -l walltime={arguments.time}"])
    script = "\n".join(["#!/bin/bash", *head, "# fake runner: not a real job script", "set -euo pipefail",
                        f'RUNDIR="{rundir}"',
                        *(arguments.setup or []),
                        *([f"# the node's Python: {arguments.python}"] if arguments.python else []),
                        'CUDA_VISIBLE_DEVICES="" nice -n 19 gpuwm-cm1 cm1out "$RUNDIR" --follow &',
                        'CUDA_VISIBLE_DEVICES="" nice -n 19 gpuwm-cm1 quicklook "$RUNDIR" --follow &',
                        'gpuwm-cm1 run "$RUNDIR" --continue' + (f" --until {arguments.until}" if arguments.until else ""),
                        "wait", ""])
    target = None
    if arguments.write:
        target = rundir / f"job-{arguments.scheduler}.sh"
        target.write_text(script, encoding="utf-8")
    if arguments.json:
        print(json.dumps({"script": script, "file": None if target is None else str(target),
                          "command": "gpuwm-cm1 jobscript (fake runner)"}))
    else:
        print(script, end="")
    return 0


_COMPONENT_PTYPE = {"morrison-mp10": 5, "wsm6-mp6": 7, "thompson-mp8": 3}


def _read_deck(path: Path) -> dict[str, Any] | None:
    """The fake runner's own deck, or the numbers of a real gpuwm-cm1 deck (the mixed runner's runs)."""

    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return None
    if isinstance(document.get("fake"), dict):
        return document["fake"]
    grid, run = document.get("grid"), document.get("run")
    if not isinstance(grid, dict) or not isinstance(run, dict):
        return None
    init = document.get("initialization") or {}
    frame = next((table for table in document.values() if isinstance(table, dict) and "u_move_m_s" in table), {})
    for table in document.values():
        if isinstance(table, dict):
            for inner in table.values():
                if isinstance(inner, dict) and "u_move_m_s" in inner:
                    frame = inner
    deck = {
        "nx": int(grid.get("nx", 120)), "ny": int(grid.get("ny", 120)), "nz": int(grid.get("nz", 40)),
        "dx_m": float(grid.get("dx_m", 1000.0)), "dy_m": float(grid.get("dy_m", 1000.0)),
        "dz_m": float(grid.get("dz_m", 500.0)), "dt_s": float(run.get("dt_s", 6.0)),
        "duration_s": float(run.get("duration_s", 7200.0)),
        "history_interval_s": float(run.get("history_interval_s", 900.0)),
        "restart_interval_s": run.get("restart_interval_s"),
        "ptype": _COMPONENT_PTYPE.get(str((document.get("physics") or {}).get("component")), 5),
        "isnd": int(init.get("isnd", 7)), "iorigin": int(init.get("iorigin", 1)),
        "umove": float(frame.get("u_move_m_s", 0.0)), "vmove": float(frame.get("v_move_m_s", 0.0)),
    }
    for key in ("bubble_center_x_m", "bubble_center_y_m"):
        if key in init:
            deck[key] = float(init[key])
    return deck


def cmd_check(arguments: argparse.Namespace) -> int:
    deck = _read_deck(Path(arguments.config))
    if deck is None:
        return _refuse("CM1-CONFIG-UNREADABLE", f"{arguments.config} is not a deck the fake runner wrote",
                       "import the namelist with the fake runner")
    print(json.dumps({"status": "admitted", "fake_runner": True, "note": FAKE,
                      "config_sha256": _sha256(Path(arguments.config).read_bytes())}, indent=2, sort_keys=True))
    return 0


def _cells(deck: dict[str, Any]) -> int:
    return int(deck["nx"]) * int(deck["ny"]) * int(deck["nz"])


def _geometry(deck: dict[str, Any]) -> dict[str, Any]:
    width_km = deck["nx"] * deck["dx_m"] / 1000.0
    depth_km = deck["ny"] * deck["dy_m"] / 1000.0
    centred = int(deck.get("iorigin", 1)) == 2
    origin = [-width_km / 2, -depth_km / 2] if centred else [0.0, 0.0]
    bubble = [float(deck.get("bubble_center_x_m", (0.0 if centred else width_km * 500.0))) / 1000.0,
              float(deck.get("bubble_center_y_m", (0.0 if centred else depth_km * 500.0))) / 1000.0]
    frame = [float(deck.get("umove", 0.0)), float(deck.get("vmove", 0.0))]
    duration = float(deck["duration_s"])
    motion = [frame[0] + 2.0, frame[1] - 1.0]
    track = [[t, round(bubble[0] + (motion[0] - frame[0]) * t / 1000.0, 3),
              round(bubble[1] + (motion[1] - frame[1]) * t / 1000.0, 3)]
             for t in [duration * i / 8 for i in range(9)]]
    recommended = [round(motion[0] * 8) / 8, round(motion[1] * 8) / 8]
    uses = recommended == frame
    return {"schema": "gpuwm-cm1.storm-track/v1", "status": "evaluated", "fake_runner": True,
            "min_edge_clearance_km": round(width_km * 0.25, 1), "minimum_edge_clearance_km": round(width_km * 0.25, 1),
            "clearance_lost_at_s": None, "domain_km": [width_km, depth_km], "origin_km": origin,
            "bubble_km": bubble, "frame_m_s": frame, "duration_s": duration, "stays_inside": True,
            "open_sides": {"west": True, "east": True, "south": True, "north": True},
            "storm_motion": {"right_mover_m_s": motion, "left_mover_m_s": [motion[0] - 3, motion[1] + 8],
                             "mean_wind_m_s": [motion[0] - 1.5, motion[1] + 4]},
            "track_km": {"right": track, "left": track, "mean": track},
            "recommended_frame_m_s": recommended,
            "recommended_pair": {"frame_m_s": recommended, "bubble_km": [0.0, 0.0] if centred else bubble,
                                 "minimum_edge_clearance_km": round(width_km * 0.3, 1), "stays_inside": True,
                                 "uses_it": uses, "why": "made up by the fake runner"},
            "hodograph_m_s": [[z * 500.0, z * 1.5, z * 0.4] for z in range(0, 21)],
            "model": "made up by the fake runner"}


def cmd_cartesian(arguments: argparse.Namespace) -> int:
    if not arguments.plan:
        return _refuse("CM1-FAKE-NO-DEVICE", "the fake runner only prices plans",
                       "pass --plan, or use the real gpuwm-cm1")
    deck = _read_deck(Path(arguments.config))
    if deck is None:
        return _refuse("CM1-CONFIG-UNREADABLE", f"{arguments.config} is not a deck the fake runner wrote",
                       "import the namelist with the fake runner")
    cells = _cells(deck)
    print(json.dumps({
        "schema": "gpuwm-cm1.cartesian-plan/v1", "status": "priced-no-device-opened",
        "fake_runner": True, "note": FAKE,
        "run_geometry": _geometry(deck),
        "history_frame": {"payload_bytes": 84 * cells},
        "cells": cells,
    }, indent=2, sort_keys=True))
    return 0


def cmd_estimate(arguments: argparse.Namespace) -> int:
    if getattr(arguments, "each", None):
        # The real --each: one JSON line per target, each what the target alone prints.
        import contextlib
        import io

        for target in arguments.each:
            one = argparse.Namespace(**{**vars(arguments), "target": target, "each": None})
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                rc = cmd_estimate(one)
            try:
                payload = json.loads(buffer.getvalue())
            except ValueError:
                payload = {"status": "not-priced", "reason": buffer.getvalue()[-400:]}
            print(json.dumps({"schema": "gpuwm-cm1.estimate-each/v1", "target": str(target), "rc": rc,
                              "payload": payload}, sort_keys=True), flush=True)
            time.sleep(_env_float("GM1_FAKE_EACH_S", 0.0))
        return 0
    if arguments.target is None:
        return _refuse("CM1-INPUT-ERROR", "give a run folder", "name one")
    rundir = Path(arguments.target)
    deck = _read_deck(rundir / "deck.toml") if rundir.is_dir() else _read_deck(rundir)
    if deck is None:
        return _refuse("CM1-CONFIG-UNREADABLE", f"{rundir} has no fake deck", "import first")
    if getattr(arguments, "measure", False):
        time.sleep(_env_float("GM1_FAKE_MEASURE_S", 0.5))
    cells = _cells(deck)
    steps = int(round(deck["duration_s"] / deck["dt_s"]))
    frames = int(deck["duration_s"] // deck["history_interval_s"]) + 1
    step_s = max(0.068, cells * 43.5e-9)
    wall = 12.0 + steps * step_s
    need = int(1.2e9 + 461 * cells)
    # GM1_FAKE_GPU_FREE_BYTES: the fake card's free memory (the GUI keys the examples' kept prices by it).
    free, total, in_use = int(_env_float("GM1_FAKE_GPU_FREE_BYTES", 8e9)), int(10e9), int(2e9)
    disk_need, disk_free = frames * 193 * cells, int(500e9)
    geometry = _geometry(deck)
    estimate = {
        "schema": "gm1.estimate/v1", "fake_runner": True, "note": FAKE,
        "cells": cells, "steps": steps, "frames": frames,
        "gpu": {"index": arguments.gpu or 0, "name": "Fake GPU", "need_bytes": need,
                "free_bytes": free, "total_bytes": total, "in_use_bytes": in_use, "fits": need <= 0.95 * free,
                "wsl2": False, "sm": "sm_86"},
        "host_locked": {"need_bytes": 365 * cells, "ram_bytes": int(32e9)},
        "disk": {"need_bytes": disk_need, "free_bytes": disk_free},
        "wall": {"central_s": round(wall, 1), "low_s": round(wall * 0.75, 1), "high_s": round(wall * 1.25, 1),
                 "basis": "made up by the fake runner"},
        "track": geometry,
        "storm_track": geometry,
        "host": {"ram_total_bytes": int(32e9), "disk_free_bytes": disk_free, "disk_total_bytes": int(1e12),
                 "disk_root": str(getattr(arguments, "root", None) or rundir)},
        "needs": {
            "gpu_memory_bytes": {"value": need, "label": "estimated", "basis": "made up by the fake runner",
                                 "low": need * 0.9, "high": need * 1.1},
            "host_locked_bytes": {"value": 365 * cells, "label": "estimated", "basis": "made up by the fake runner"},
            "disk_bytes": {"value": disk_need, "label": "estimated", "basis": "made up by the fake runner"},
        },
        "fits": {"all": need <= 0.95 * free, "gpu": need <= 0.95 * free, "host": True, "disk": True,
                 "rule": "fake runner"},
        "time": {"wall_s": {"value": wall, "low": wall * 0.75, "high": wall * 1.25, "label": "estimated",
                            "basis": "made up by the fake runner", "basis_label": "rough"},
                 "step": {"seconds_per_step": step_s, "label": "rough", "basis": "made up by the fake runner",
                          "measured_here": bool(getattr(arguments, "measure", False))}},
        "run": {"cells": cells, "steps": steps, "frames": frames, "dt_s": deck["dt_s"],
                "duration_s": deck["duration_s"], "history_interval_s": deck["history_interval_s"],
                "acoustic_substeps": 4, "restarts": 1, "microphysics": MICROPHYSICS.get(deck["ptype"]),
                "grid": {"nx": deck["nx"], "ny": deck["ny"], "nz": deck["nz"], "dx_m": deck["dx_m"],
                         "dy_m": deck["dy_m"], "dz_m": deck["dz_m"]}},
        "recommendation": {"memory_budget_mib": max(1024, int(need * 1.1 / 2**20))},
        "checks": [{"id": "gpu-memory", "level": "ok", "text": "Fits in the fake GPU (fake runner)."},
                   {"id": "storm-track", "level": "ok", "text": "The storm stays inside the area (fake runner)."}],
        "command": f"gpuwm-cm1 estimate {rundir} --json",
    }
    if rundir.is_dir() and not getattr(arguments, "no_write", False):
        write_json_atomic(rundir / "estimate.json", estimate)
        write_json_atomic(rundir / "geometry.json", geometry)
    print(json.dumps(estimate, indent=2, sort_keys=True))
    return 0


def cmd_status(arguments: argparse.Namespace) -> int:
    status = read_status(Path(arguments.rundir)).as_dict()
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0


# ------------------------------------------------------------ run and resume

class _Terminated(Exception):
    pass


def _update_run(rundir: Path, row: dict[str, Any]) -> None:
    path = rundir / "run.json"
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run = {"schema": "gm1.cm1-run/v1", "fake_runner": True, "name": rundir.name}
    segments = [item for item in run.get("segments", []) if item.get("index") != row["index"]]
    segments.append(row)
    run["segments"] = segments
    write_json_atomic(path, run)


def _publish(parent: Path, name: str, files: dict[str, bytes]) -> int:
    staging = parent / ".incomplete" / name
    staging.mkdir(parents=True, exist_ok=True)
    total = 0
    for filename, data in files.items():
        (staging / filename).write_bytes(data)
        total += len(data)
    os.replace(staging, parent / name)
    return total


def _restarts(rundir: Path) -> list[dict[str, Any]]:
    rows = []
    for segment in segment_dirs(rundir):
        for record in iter_records(segment / "events.jsonl"):
            if record.get("kind") == "restart":
                rows.append({**record, "segment": segment.name})
    return rows


def cmd_run(arguments: argparse.Namespace, resume: bool = False) -> int:
    time.sleep(_env_float("GM1_FAKE_LAUNCH_DELAY_S", 0.0))
    rundir = Path(arguments.rundir).resolve()
    deck = _read_deck(rundir / "deck.toml")
    if deck is None:
        return _refuse("CM1-RUN-NO-DECK", f"{rundir} has no deck.toml", "import the namelist first", False)
    control_dir = rundir / control.CONTROL_DIR
    for name in ("stop", "pause"):
        if (control_dir / name).exists():
            return _refuse("CM1-RUN-CONTROL-LEFT", f"{control_dir / name} is left from an earlier request",
                           "remove it, then start again", False)
    control_dir.mkdir(exist_ok=True)
    dt = float(deck["dt_s"])
    duration = float(deck["duration_s"])
    history = float(deck["history_interval_s"])
    restart_interval = deck.get("restart_interval_s")
    start_step, next_frame, restart_number, resumed_from = 0, 0, 0, None
    if resume:
        restarts = _restarts(rundir)
        if arguments.from_restart is not None:
            restarts = [row for row in restarts if row.get("number") == arguments.from_restart]
        if getattr(arguments, "from_segment", None) is not None:
            restarts = [row for row in restarts if row.get("segment") == f"{arguments.from_segment:03d}"]
        if not restarts:
            return _refuse("CM1-RESUME-NO-RESTART", "there is no such restart to continue from",
                           "run until a restart is saved, or pick one that exists", False)
        chosen = restarts[-1]
        start_step = int(chosen["step"])
        restart_number = int(chosen["number"])
        next_frame = int(chosen.get("next_history_index", 0))
        resumed_from = chosen["directory"]
        resumed_segment = int(chosen["segment"])
    until = getattr(arguments, "until", None)
    if until is not None:
        # The real commands' storm-time words (seconds, 150min, 2.5h), for run as for resume.
        from ..cm1run.launch import parse_seconds

        duration = parse_seconds(until)
    total_steps = int(round(duration / dt))
    if total_steps <= start_step:
        return _refuse("CM1-RESUME-NOTHING-LEFT", "the run already reached that storm time",
                       "pass a later --until", False)
    index = len(segment_dirs(rundir)) + 1
    segment = rundir / "segments" / f"{index:03d}"
    segment.mkdir(parents=True)
    frames_dir, restarts_dir = segment / "frames", segment / "restarts"
    frames_dir.mkdir()
    restarts_dir.mkdir()
    _update_run(rundir, {
        "index": index, "number": index, "directory": f"segments/{index:03d}", "pid": os.getpid(),
        "pid_start_ticks": proc_start_ticks(os.getpid()), "host": this_host(),
        "started_utc": _utc(), "scheduler_job_id": None, "command": "resume" if resume else "run",
        # As cm1run records it: the segment and restart a resume went on from (the run's line).
        "resumed_from": ({"segment": resumed_segment, "restart": restart_number,
                          "model_time_s": start_step * dt} if resume else None),
    })
    from ..run_events import RunEventLog

    log = RunEventLog(segment / "events.jsonl")
    run_log = (segment / "run.log").open("a", encoding="utf-8")
    terminated: list[str] = []

    def on_term(signum: int, frame: Any) -> None:
        terminated.append(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, on_term)
    step_s = _env_float("GM1_FAKE_STEP_S", 0.004)
    outcome = os.environ.get("GM1_FAKE_OUTCOME", "")
    reads_control = os.environ.get("GM1_FAKE_IGNORE_CONTROL", "") != "1"
    frames_expected = int((duration - start_step * dt) // history) + (0 if resume else 1)
    restarts_expected = (int((duration - start_step * dt) // restart_interval)
                         if restart_interval else 1)
    grid = {"nx": int(deck["nx"]), "ny": int(deck["ny"]), "nz": int(deck["nz"]),
            "dx_m": float(deck["dx_m"]), "dy_m": float(deck["dy_m"]), "dz_m": float(deck["dz_m"]),
            "stretch_z": 0, "ztop_m": float(deck["dz_m"]) * int(deck["nz"])}
    log.emit("admitted", config=str(rundir / "deck.toml"),
             config_sha256=_sha256((rundir / "deck.toml").read_bytes()), grid=grid,
             microphysics=MICROPHYSICS.get(int(deck["ptype"]), "morrison-mp10"),
             steps=total_steps - start_step, start_step=start_step, dt_s=dt, nsound=4,
             adaptive_timestep=False, duration_s=duration, history_interval_s=history,
             frames_expected=frames_expected, restart_interval_s=restart_interval,
             restarts_expected=restarts_expected, history_directory=str(frames_dir),
             restart_directory=str(restarts_dir), resumed_from=resumed_from,
             receipt=str(segment / "receipt.json"), control_directory=str(control_dir),
             history_envelope="off", history_initial=not resume, host=this_host(),
             gpu_uuid_requested=None, fake_runner=True)
    run_log.write(f"{_utc()} fake runner: {FAKE}\n")
    wait_gpu = _env_float("GM1_FAKE_WAIT_GPU_S", 0.0)
    if wait_gpu > 0:
        log.set_phase("waiting-for-gpu")
        log.progress(_idle_progress(total_steps, duration))
        time.sleep(wait_gpu)
    log.set_phase("starting")
    startup = _env_float("GM1_FAKE_STARTUP_S", 0.3)
    time.sleep(startup)
    log.emit("startup", stages={"compile": startup * 0.6, "initializer": startup * 0.4},
             seconds=startup, nvrtc_compile_s=None, nvrtc_programs=None,
             gpu={"name": "Fake GPU", "uuid": None})
    log.set_phase("stepping")
    cells = grid["nx"] * grid["ny"] * grid["nz"]
    frames = restarts = 0
    last_io: dict[str, Any] | None = None

    def frame(step: int) -> None:
        nonlocal frames, next_frame, last_io
        payload = struct.pack("<4sdI", b"FAKE", step * dt, step) + bytes(2048)
        meta = json.dumps({"index": next_frame, "step": step, "model_time_s": step * dt,
                           "fake_runner": True, "note": FAKE}).encode()
        size = _publish(frames_dir, f"frame-{next_frame:06d}", {"frame.json": meta, "payload.bin": payload})
        log.emit("frame", index=next_frame, step=step, model_time_s=step * dt,
                 directory=str(frames_dir / f"frame-{next_frame:06d}"), bytes=size,
                 initial_condition=step == 0, restart_written=None)
        last_io = {"kind": "history", "label": f"frame-{next_frame:06d}", "seconds": 0.001, "bytes": size}
        frames += 1
        next_frame += 1

    def restart(step: int) -> None:
        nonlocal restarts, restart_number, last_io
        restart_number += 1
        meta = json.dumps({"number": restart_number, "step": step, "model_time_s": step * dt,
                           "fake_runner": True}).encode()
        # The publication receipt cm1run reads a restart by (restart-publication.json), with made-up bytes.
        receipt = json.dumps({"schema": "gm1.fake-restart-publication/v1", "fake_runner": True,
                              "number": restart_number, "step": step, "model_time_s": step * dt,
                              "next_history_index": next_frame}).encode()
        size = _publish(restarts_dir, f"restart-{restart_number:06d}",
                        {"restart.json": meta, "restart-publication.json": receipt})
        log.emit("restart", number=restart_number, step=step, model_time_s=step * dt,
                 directory=str(restarts_dir / f"restart-{restart_number:06d}"), bytes=size,
                 next_history_index=next_frame)
        last_io = {"kind": "restart", "label": f"restart-{restart_number:06d}", "seconds": 0.001, "bytes": size}
        restarts += 1

    def finish(reason: str, rc: int, step: int, *, signal_name: str | None = None,
               request: str | None = None, message: str | None = None) -> int:
        receipt = segment / "receipt.json"
        write_json_atomic(receipt, {
            "schema": "gm1.fake-receipt/v1", "fake_runner": True, "note": FAKE, "status": reason,
            "step": step, "model_time_s": step * dt, "frames_published": frames,
            "restarts_published": restarts, "cells": cells, "wall_s": round(time.monotonic() - began, 3),
        })
        log.set_phase("finishing")
        log.emit("end", reason=reason, rc=rc, receipt=str(receipt), step=step, model_time_s=step * dt,
                 frames_published=frames, restarts_published=restarts, signal=signal_name,
                 request=request, message=message)
        log.close()
        run_log.write(f"{_utc()} end: {reason}\n")
        run_log.close()
        # As the real supervisor does after the end record (cm1run.control.archive_requests): the stop or pause
        # this run honoured moves into its segment, so the next resume starts with a clean control/.
        for name in ("stop", "pause"):
            source = control_dir / name
            if source.exists():
                try:
                    os.replace(source, segment / f"control-{name}.ended")
                except OSError:
                    pass
        return rc

    began = time.monotonic()
    if not resume:
        frame(0)
    last_progress = 0.0
    pause_pending = False
    interval_steps = max(1, int(round(history / dt)))
    restart_steps = max(1, int(round(restart_interval / dt))) if restart_interval else None
    for step in range(start_step + 1, total_steps + 1):
        time.sleep(step_s)
        fraction = (step - start_step) / (total_steps - start_step)
        if outcome == "vanish" and fraction >= 0.5:
            os._exit(1)
        if outcome == "error" and fraction >= 0.4:
            return finish("error", 1, step, message="the fake runner was told to fail")
        if outcome == "courant-stop" and fraction >= 0.6:
            return finish("courant-stop", 1, step, message="the fake runner was told to stop on Courant")
        if terminated:
            return finish("stopped", 143, step - 1, signal_name=terminated[0])
        at_frame = step % interval_steps == 0
        at_restart = (restart_steps is not None and step % restart_steps == 0) or \
                     (restart_steps is None and step == total_steps)
        if at_frame:
            frame(step)
        if at_restart:
            restart(step)
            if pause_pending and step < total_steps:
                return finish("paused", 0, step, request="pause")
        if reads_control and (control_dir / "stop").exists():
            return finish("stopped", 0, step, request="stop")
        if reads_control and (control_dir / "pause").exists():
            pause_pending = True
        now = time.monotonic()
        if now - last_progress >= 0.2 or step == total_steps:
            last_progress = now
            elapsed = now - began
            done = step - start_step
            mean = elapsed / max(1, done)
            log.progress({
                "step": step, "total_steps": total_steps, "model_time_s": step * dt,
                "duration_s": duration, "s_per_step_recent": step_s, "s_per_step_mean": mean,
                "model_s_per_wall_s": dt / mean if mean > 0 else None,
                "eta_s": max(0.0, (total_steps - step) * mean), "last_io": last_io,
                "gpu": {"name": "Fake GPU", "memory_used_bytes": int(1.2e9 + 461 * cells),
                        "memory_total_bytes": int(10e9)},
            })
    return finish("completed", 0, total_steps)


def _idle_progress(total_steps: int, duration: float) -> dict[str, Any]:
    return {"step": None, "total_steps": total_steps, "model_time_s": None, "duration_s": duration,
            "s_per_step_recent": None, "s_per_step_mean": None, "model_s_per_wall_s": None,
            "eta_s": None, "last_io": None, "gpu": None}


# ------------------------------------------------------------ sidecars

def _png(width: int, height: int, seed: int, hue: tuple[int, int, int]) -> bytes:
    rows = []
    cx, cy = (seed * 7) % width, height // 2 + (seed % 5) - 2
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            d2 = (x - cx) ** 2 + 4 * (y - cy) ** 2
            glow = max(0.0, 1.0 - d2 / 260.0)
            row += bytes(min(255, int(base + tint * glow)) for base, tint in zip((5, 7, 13), hue))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def _frames(rundir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for segment in segment_dirs(rundir):
        base = segment / "frames"
        if not base.is_dir():
            continue
        for path in sorted(base.glob("frame-*")):
            try:
                rows.append((path, json.loads((path / "frame.json").read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
    return rows


def _netcdf_times(target: Path) -> list[float] | None:
    """CM1 netCDF output (one cm1out.nc, or cm1out_NNNNNN.nc files): made-up saved times, 300 s apart for one file
    per time, or 0 and 3600 s for one file holding every time; None for anything else."""

    from .files import CM1_HISTORY_NAME

    if target.is_file():
        names = [target.name] if target.suffix == ".nc" else []
    elif target.is_dir() and not (target / "segments").is_dir() and not (target / "frames").is_dir():
        names = sorted(p.name for p in target.iterdir() if CM1_HISTORY_NAME.match(p.name))
    else:
        names = []
    if not names:
        return None
    if len(names) == 1:
        return [0.0, 3600.0]
    return [300.0 * n for n in range(len(names))]


def _out(arguments: argparse.Namespace, rundir: Path) -> Path:
    """Where the pictures go: --out, as the real command takes it, else RUNDIR/quicklook."""

    return Path(arguments.out).resolve() if getattr(arguments, "out", None) else rundir / "quicklook"


def _ended(rundir: Path) -> bool:
    segments = segment_dirs(rundir)
    if not segments:
        return False
    return any(record.get("kind") == "end" for record in iter_records(segments[-1] / "events.jsonl"))


def _follow(rundir: Path, follow: bool, convert) -> int:
    """Convert each published frame once; with ``follow``, until the run is over.

    Over means an end record, or a run that is no longer running (killed, or
    lost with its node): a sidecar never outlives the run it follows.
    """

    done: set[int] = set()
    waited = time.monotonic()
    quiet_since: float | None = None
    while True:
        state = read_status(rundir).state if segment_dirs(rundir) else None
        if state in (None, "running", "waiting"):
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        # Not running for two seconds running: gone (a status read can race a new segment).
        over = _ended(rundir) or (quiet_since is not None and time.monotonic() - quiet_since > 2.0)
        for path, meta in _frames(rundir):
            index = int(meta["index"])
            if index not in done:
                convert(index, meta)
                done.add(index)
        if not follow or over:
            return 0
        if state is None and time.monotonic() - waited > 60.0:
            print("no run started in 60 s; stopping", file=sys.stderr)
            return 1
        time.sleep(0.1)


def cmd_cm1out(arguments: argparse.Namespace) -> int:
    rundir = Path(arguments.rundir).resolve()
    out = rundir / "cm1out"
    out.mkdir(exist_ok=True)

    def convert(index: int, meta: dict[str, Any]) -> None:
        data = b"FAKE netCDF placeholder: " + json.dumps(meta).encode() + b"\n"
        target = out / f"cm1out_{index + 1:06d}.nc"
        temporary = out / f".{target.name}.tmp"
        temporary.write_bytes(data)
        os.replace(temporary, target)

    rc = _follow(rundir, arguments.follow, convert)
    if _ended(rundir):
        (out / "cm1out_stats.nc").write_bytes(b"FAKE netCDF placeholder: statistics\n")
    return rc


def _fake_picture(arguments: argparse.Namespace) -> int:
    """quicklook --slice/--section/--column/--fields: one placeholder picture, and the real command's JSON shape."""

    rundir = Path(arguments.rundir).resolve()
    times = [float(meta["model_time_s"]) for _path, meta in _frames(rundir)] or _netcdf_times(rundir) or []
    if not times:
        # a folder of real published frames (attached, view-only): their clocks
        for path in sorted(rundir.glob("frames/frame-*")) + sorted(rundir.glob("segments/*/frames/frame-*")):
            location = path / "history-frame-location.json"
            if location.is_file():
                times.append(float(json.loads(location.read_text(encoding="utf-8"))["model_time_s"]))
    if not times:
        print(json.dumps({"status": "error", "reason": "no published frames yet", "fake_runner": True}), file=sys.stderr)
        return 2
    t = times[-1] if arguments.time is None else min(times, key=lambda v: abs(v - arguments.time))
    if arguments.fields:
        record = {"kind": "fields", "time_s": t, "fake_runner": True, "note": FAKE,
                  "fields": [{"name": n, "grid": g, "units": u, "description": d, "levels_m": None}
                             for n, g, u, d in (("w", "w", "m/s", "vertical velocity"),
                                                ("dbz", "s", "dBZ", "reflectivity"),
                                                ("rain", "2", "cm", "accumulated surface rainfall"))]}
        if _netcdf_times(rundir) is not None:
            # as the real command says of CM1 netCDF output: its saved times and its grid (made up here)
            record.update({"saved_times_s": [times[0], times[-1]], "saved_count": len(times),
                           "grid": {"nx": 24, "ny": 24, "nz": 40, "dx_m": 1000.0}})
        print(json.dumps(record))
        return 0
    kind, name = (("horizontal", f"slice-{arguments.slice}") if arguments.slice else
                  ("vertical", f"section-{arguments.section}") if arguments.section else ("column", "column"))
    slices = _out(arguments, rundir) / "slices"
    slices.mkdir(parents=True, exist_ok=True)
    file = f"{name}-t{int(round(t)):06d}.png"
    (slices / file).write_bytes(_png(96, 54, 7, (200, 160, 60)))
    print(json.dumps({"kind": kind, "time_s": t, "file": f"slices/{file}", "path": str(slices / file),
                      "fake_runner": True, "note": FAKE}))
    return 0


def cmd_render(arguments: argparse.Namespace) -> int:
    """render RUNDIR --product P: one placeholder picture in quicklook/native/ and the real command's JSON shape."""

    rundir = Path(arguments.rundir).resolve()
    restarts = sorted(rundir.glob("segments/*/restarts/restart-*"))
    if not restarts:
        print(json.dumps({"code": "CM1-RENDER-NO-RESTART", "reason": "no published restart checkpoint yet",
                          "fake_runner": True}), file=sys.stderr)
        return 2
    native = rundir / "quicklook" / "native"
    native.mkdir(parents=True, exist_ok=True)
    file = f"{arguments.product}-fake.png"
    (native / file).write_bytes(_png(96, 104, 11, (90, 170, 210)))
    print(json.dumps({"schema": "gpuwm-cm1.render/v2", "status": "rendered", "source": "restart",
                      "product": arguments.product, "file": f"native/{file}", "output": str(native / file),
                      "value_range": [[0.0, 1.0]], "units": "", "fake_runner": True, "note": FAKE,
                      "caveat": "Restart checkpoint, not a history frame"}))
    return 0


def _fake_film_or_look(arguments: argparse.Namespace) -> int:
    """quicklook --film LOOK / --look LOOK: placeholder pictures in the real command's layout and JSON shape."""

    rundir = Path(arguments.rundir).resolve()
    out = _out(arguments, rundir)
    nc_times = _netcdf_times(rundir)
    times = [float(meta["model_time_s"]) for _path, meta in _frames(rundir)] or nc_times or []
    if not times:
        print(json.dumps({"status": "error", "reason": "no published frames yet", "fake_runner": True}), file=sys.stderr)
        return 2
    if arguments.look:
        ended = _ended(rundir) or nc_times is not None
        if arguments.look == "parcels" and not (ended and _fake_setting(rundir, "iprcl") == "1"):
            reason = ("Parcels arrive when the run ends." if not ended else
                      "This run did not follow air parcels. To follow them, set CM1's iprcl = 1 for the next run: in Advanced, Namelist, or with gpuwm-cm1 edit RUNDIR --set iprcl=1 before it starts.")
            print(json.dumps({"status": "error", "reason": reason, "fake_runner": True}), file=sys.stderr)
            return 2
        if arguments.look == "storm3d":
            return _fake_storm3d(out, times)
        name = f"{arguments.look}/t{int(round(times[0])):06d}.png"
        (out / arguments.look).mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(_png(96, 54, 5, (120, 200, 160)))
        print(json.dumps({"kind": "look", "look": arguments.look, "file": name, "path": str(out / name),
                          "time_s": times[0], "fake_runner": True, "note": FAKE}))
        return 0
    look = arguments.film
    folder = out / "films" / look
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    total = max(1, len(times))
    progress = {"schema": "gm1.quicklook-film-progress/v1", "look": look, "state": "running", "done": 0,
                "total": total, "pid": os.getpid(), "message": ""}
    delay = _env_float("GM1_FAKE_FILM_S", 0.0) / total
    frames = []
    for number, _t in enumerate(times):
        (folder / "frames" / f"{number:05d}.png").write_bytes(_png(96, 54, number + 2, (160, 90, 220)))
        frames.append(f"films/{look}/frames/{number:05d}.png")
        progress["done"] = number + 1
        write_json_atomic(folder / "progress.json", progress)
        if delay:
            time.sleep(delay)
    (folder / "film.png").write_bytes(_png(96, 54, 3, (160, 90, 220)))
    files = {"apng": f"films/{look}/film.png"}
    if (arguments.format or "") == "gif":
        (folder / "film.gif").write_bytes(b"GIF89a fake runner placeholder")
        files["gif"] = f"films/{look}/film.gif"
    record = {"look": look, "title": look, "pictures": len(frames), "fps": arguments.fps or 6.0,
              "between": arguments.between or 0, "files": files, "frames": frames, "size": [96, 54],
              "made_utc": _utc(), "seconds": 0.0, "stopped": False, "note": FAKE, "mp4_available": False,
              "facts": {"times_s": times}, "fake_runner": True}
    index_path = out / "films" / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {"schema": "gm1.quicklook-films/v1", "films": {}}
    index.setdefault("films", {})[look] = record
    write_json_atomic(index_path, index)
    progress.update({"state": "done", "message": f"{len(frames)} pictures"})
    write_json_atomic(folder / "progress.json", progress)
    print(json.dumps(record))
    return 0


def _fake_storm3d(out: Path, times: list[float]) -> int:
    """quicklook --look storm3d: a placeholder still of the latest saved time, listed as the real one is
    (``storm3d/drawn.json`` and the index's ``storm3d`` entry, drawn on request)."""

    t = times[-1]
    name = f"storm3d/t{int(round(t)):06d}.png"
    (out / "storm3d" / "thumb").mkdir(parents=True, exist_ok=True)
    (out / name).write_bytes(_png(96, 54, 7, (230, 140, 60)))
    (out / "storm3d" / "thumb" / Path(name).name).write_bytes(_png(48, 27, 7, (230, 140, 60)))
    record = {"frame": "fake", "time": t, "file": name, "thumb": f"storm3d/thumb/{Path(name).name}",
              "facts": {"kind": "thunderstorm", "field": "reflectivity", "stride": 1, "fake_runner": True},
              "seconds": 0.0}
    try:
        drawn = json.loads((out / "storm3d" / "drawn.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        drawn = {"schema": "gm1.quicklook-storm3d/v1", "stills": {}}
    drawn.setdefault("stills", {})[repr(float(t))] = record
    write_json_atomic(out / "storm3d" / "drawn.json", drawn)
    try:
        index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {"schema": "gm1.quicklook-index/v1", "fake_runner": True, "products": [], "frames": []}
    products = [p for p in index.get("products") or [] if p.get("id") != "storm3d"]
    stills = sorted(drawn["stills"].values(), key=lambda r: r["time"])
    products.append({"id": "storm3d", "title": "The storm in 3-D", "on_request": True, "frames": [
        {"index": n, "time_s": r["time"], "clock": r["time"], "file": r["file"], "thumb": r["thumb"],
         "facts": r["facts"]} for n, r in enumerate(stills)]})
    index["products"] = products
    write_json_atomic(out / "index.json", index)
    print(json.dumps({**record, "kind": "look", "look": "storm3d", "path": str(out / name), "fake_runner": True,
                      "note": FAKE}))
    return 0


#: Made-up quick-look facts in the shape of a supercell's first half hour (quiet air, then a storm from about
#: T+0:20:00): (time s, strongest reflectivity at 1 km and in the column in dBZ, strongest updraft at 5 km in m/s).
FAKE_STORM = ((0.0, -35.0, -35.0, 0.0), (900.0, -34.0, -23.0, 0.3), (1200.0, -28.0, 7.0, 10.0),
              (1500.0, 2.0, 60.0, 19.0), (1800.0, 52.0, 65.0, 25.0))


#: Made-up facts in the shape of a hurricane's four days (a ragged first day, an eye by the second): (time s,
#: strongest reflectivity at 1 km and in the column in dBZ, fastest wind at the lowest level in m/s, lowest pressure
#: there in hPa, most rain in one cell in cm).
FAKE_HURRICANE = ((0.0, -35.0, -35.0, 15.0, 1006.0, 0.0), (21600.0, 41.0, 42.0, 14.0, 1006.5, 2.0),
                  (86400.0, 46.0, 52.0, 16.0, 1005.5, 20.0), (172800.0, 52.0, 58.0, 47.0, 979.0, 90.0),
                  (259200.0, 54.0, 60.0, 56.0, 963.0, 200.0), (345600.0, 54.0, 61.0, 59.0, 954.0, 360.0))


def _between(rows: tuple[tuple[float, ...], ...], seconds: float) -> tuple[float, ...]:
    """The made-up values at ``seconds``, straight lines between the table's rows."""

    if seconds >= rows[-1][0]:
        return tuple(rows[-1][1:])
    after = next(n for n, row in enumerate(rows) if row[0] > seconds)
    (t0, *a), (t1, *b) = rows[after - 1], rows[after]
    share = (seconds - t0) / (t1 - t0)
    return tuple(x + share * (y - x) for x, y in zip(a, b))


def _fake_hurricane_facts(product: str, seconds: float) -> dict[str, Any]:
    """The facts a hurricane run's quick-look records for ``product`` (cm1core.quicklook's keys), made up."""

    dbz_1km, dbz_column, wind, pressure, rain = (round(value, 2) for value in _between(FAKE_HURRICANE, seconds))
    if product == "reflectivity":
        return {"dbz_max_1km": dbz_1km, "dbz_max_column": dbz_column, "fake_runner": True}
    if product == "wind-lowest":
        return {"height_m": 25.0, "wind_max_ground_relative": wind, "range_m_s": 80.0, "p_min_hpa": pressure,
                "fake_runner": True}
    if product == "pressure-lowest":
        return {"height_m": 25.0, "p_min_hpa": pressure, "p_edge_hpa": 1012.0,
                "p_drop_max_hpa": round(1012.0 - pressure, 2), "range_hpa": 70.0, "fake_runner": True}
    if product == "rain":
        return {"rain_max_cm": rain, "range_cm": 400.0, "fake_runner": True}
    return {"fake_runner": True}


def _long_clock(seconds: int) -> str:
    """A hurricane's storm time as the pages say it: ``T+81 h (3 d 9 h)`` (cm1core.quicklook.style.fmt_long_clock)."""

    whole = seconds // 3600
    shown = str(whole) if seconds % 3600 == 0 else f"{seconds / 3600:.1f}"
    return f"T+{shown} h ({whole // 24} d {whole % 24} h)" if whole >= 24 else f"T+{shown} h"


def _fake_facts(product: str, seconds: float) -> dict[str, Any]:
    """The facts a quick-look records for ``product`` at ``seconds`` (cm1core.quicklook's keys), made up."""

    rows = FAKE_STORM
    if seconds >= rows[-1][0]:
        values = rows[-1][1:]
    else:
        after = next(n for n, row in enumerate(rows) if row[0] > seconds)
        (t0, *a), (t1, *b) = rows[after - 1], rows[after]
        share = (seconds - t0) / (t1 - t0)
        values = tuple(x + share * (y - x) for x, y in zip(a, b))
    dbz_1km, dbz_column, w = (round(value, 2) for value in values)
    if product == "reflectivity":
        return {"dbz_max_1km": dbz_1km, "dbz_max_column": dbz_column, "fake_runner": True}
    if product == "updraft-5km":
        return {"level_m": 5000.0, "w_max": w, "fake_runner": True}
    return {"fake_runner": True}


def _fake_setting(rundir: Path, key: str) -> str | None:
    try:
        text = (rundir / "namelist.input").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = re.search(rf"^\s*{key}\s*=\s*([^,\s]+)", text, re.M | re.I)
    return found.group(1) if found else None


def _fake_radar(arguments: argparse.Namespace) -> int:
    """quicklook --radar-site/--radar-tilt/--radar-motion: the settings file the real command keeps
    (cm1core.quicklook.radar, gm1.quicklook-radar/v1) and its JSON shape; no picture is drawn again."""

    rundir = Path(arguments.rundir).resolve()
    out = rundir / "quicklook"
    path = out / "radar.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    if not isinstance(settings, dict) or settings.get("schema") != "gm1.quicklook-radar/v1":
        settings = {"schema": "gm1.quicklook-radar/v1", "site_km": None, "tilt_deg": 0.5, "motion_m_s": None}
    for key, values, word in (("site_km", arguments.radar_site, "storm"), ("motion_m_s", arguments.radar_motion,
                                                                           "auto")):
        if values is None:
            continue
        if len(values) == 1 and values[0] == word:
            settings[key] = None
        else:
            try:
                settings[key] = [float(v) for v in values]
            except ValueError:
                settings[key] = []
            if len(settings[key]) != 2:
                return _refuse("CM1-INPUT-ERROR", f"takes two numbers or the word {word}",
                               "correct the named input and retry")
    if arguments.radar_tilt is not None:
        if not 0.1 <= arguments.radar_tilt <= 20.0:
            return _refuse("CM1-INPUT-ERROR", "The radar tilt must be from 0.1 to 20 degrees.",
                           "correct the named input and retry")
        settings["tilt_deg"] = float(arguments.radar_tilt)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, settings)
    print(json.dumps({"kind": "radar", "settings": settings, "pictures_redrawn": 0, "frames_drawn": 0,
                      "problems": [], "fake_runner": True, "note": FAKE}))
    return 0


def _fake_compare(arguments: argparse.Namespace) -> int:
    """quicklook --compare OTHER ...: placeholder pictures and compare.json in the real command's layout
    (cm1core.quicklook.compare) and JSON shape."""

    rundir = Path(arguments.rundir).resolve()
    runs = [rundir, *[Path(o).resolve() for o in arguments.compare]]
    digest = hashlib.sha256(" ".join(str(r) for r in runs).encode()).hexdigest()[:8]
    folder = Path(arguments.out) if arguments.out else rundir / "quicklook" / "compare" / f"with-fake-{digest}"
    (folder / "thumb").mkdir(parents=True, exist_ok=True)

    def refuse(reason: str) -> int:
        write_json_atomic(folder / "progress.json", {"schema": "gm1.quicklook-compare-progress/v1",
                                                     "state": "failed", "done": 0, "total": 5, "step": "",
                                                     "message": reason, "pid": os.getpid()})
        print(json.dumps({"status": "error", "reason": reason, "fake_runner": True}), file=sys.stderr)
        return 2

    if len(runs) > 6:
        return refuse("At most 6 runs can be compared at once.")
    last = []
    for run in runs:
        times = [float(meta["model_time_s"]) for _path, meta in _frames(run)]
        if not times:
            return refuse(f"{run.name}: no saved frames and no quick-look pictures, so there is nothing of its storm "
                          "to compare.")
        last.append(max(times))
    t = float(arguments.time) if arguments.time is not None else min(last)
    tag = f"t{int(round(t)):06d}"
    progress = {"schema": "gm1.quicklook-compare-progress/v1", "state": "running", "done": 0, "total": 5,
                "step": "reading the runs", "message": "", "pid": os.getpid()}
    write_json_atomic(folder / "progress.json", progress)
    time.sleep(_env_float("GM1_FAKE_COMPARE_S", 0.0))
    pictures: dict[str, Any] = {}
    for key, name in (("storm-reflectivity", f"storm-reflectivity-{tag}.png"),
                      ("storm-surface", f"storm-surface-{tag}.png"), ("series", "series.png")):
        for where in (folder / name, folder / "thumb" / name):
            where.write_bytes(_png(96, 54, 4, (90, 160, 230)))
        pictures[key] = {"file": name, "facts": {"fake_runner": True}}
    name = f"soundings-{tag}.png"
    for where in (folder / name, folder / "thumb" / name):
        where.write_bytes(_png(96, 54, 6, (230, 120, 80)))
    letters = "ABCDEF"
    pictures["soundings"] = [{"file": name, "runs": [letters[n] for n in range(len(runs))], "facts": {}}]
    colours = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300")
    record = {"schema": "gm1.quicklook-compare/v1", "made_utc": _utc(), "time_s": t, "box_m": 48000.0,
              "kind": "thunderstorm", "folder": str(folder),
              "runs": [{"letter": letters[n], "colour": colours[n], "name": run.name, "folder": str(run),
                        "grid": "fake runner", "saved_frames": True, "time_s": t} for n, run in enumerate(runs)],
              "differ": [], "settings_unread": [], "pictures": pictures, "not_drawn": {},
              "looks": ["storm", "series", "soundings"], "fake_runner": True, "note": FAKE}
    write_json_atomic(folder / "compare.json", record)
    progress.update({"state": "done", "done": 5, "step": "", "message": "4 pictures"})
    write_json_atomic(folder / "progress.json", progress)
    print(json.dumps(record))
    return 0


def cmd_quicklook(arguments: argparse.Namespace) -> int:
    if any(getattr(arguments, key, None) is not None for key in ("radar_site", "radar_tilt", "radar_motion")):
        return _fake_radar(arguments)
    if getattr(arguments, "compare", None):
        return _fake_compare(arguments)
    if getattr(arguments, "film", None) or getattr(arguments, "look", None):
        return _fake_film_or_look(arguments)
    if arguments.slice or arguments.section or arguments.column or arguments.fields:
        return _fake_picture(arguments)
    rundir = Path(arguments.rundir).resolve()
    out = _out(arguments, rundir)
    out.mkdir(parents=True, exist_ok=True)
    hues = ((250, 90, 60), (220, 60, 200), (240, 200, 90), (90, 200, 250), (150, 110, 240), (80, 220, 120))
    index_path = out / "index.json"
    # A hurricane run (CM1's swirl start) has its own looks, facts, clock and charts, as the real quick-looks do.
    hurricane = _fake_setting(rundir, "iinit") == "7"
    listed = HURRICANE_PRODUCTS if hurricane else PRODUCTS
    facts_of = _fake_hurricane_facts if hurricane else _fake_facts

    def convert(index: int, meta: dict[str, Any]) -> None:
        current = {}
        try:
            current = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        products = {row["id"]: row for row in current.get("products", [])}
        seconds = int(round(float(meta["model_time_s"])))
        for (product, title), hue in zip(listed, hues):
            (out / product).mkdir(exist_ok=True)
            name = f"{product}/t{seconds:06d}.png"
            (out / name).write_bytes(_png(96, 54, index + 3, hue))
            row = products.setdefault(product, {"id": product, "title": title, "frames": [],
                                                "footer": f"SIMULATED · {FAKE}", "fake_runner": True})
            if all(frame["index"] != index for frame in row["frames"]):
                row["frames"].append({"index": index, "time_s": seconds, "file": name,
                                      "facts": facts_of(product, seconds)})
        times = sorted({frame["time_s"] for row in products.values() for frame in row["frames"]})
        document = {"schema": "gm1.quicklook-index/v1", "fake_runner": True,
                    "products": [products[product] for product, _ in listed if product in products],
                    "frames": [{"index": i, "time_s": t,
                                "clock": _long_clock(t) if hurricane
                                else f"T+{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"}
                               for i, t in enumerate(times)],
                    "series": {"file": "series.json"}}
        if hurricane:
            # The run's own facts, as the real index states them: Results reads the kind of storm from here.
            document["run"] = {"name": rundir.name, "storm_kind": "hurricane", "lowest_level_m": 25.0,
                               "fake_runner": True}
        write_json_atomic(index_path, document)
        if hurricane:
            rows = [(t, _between(FAKE_HURRICANE, t)) for t in times]
            charts = [{"key": key, "label": label, "unit": unit, **({"zero": False} if key == "prs_sfc_min_hpa" else {}),
                       "gm1": [[t, round(values[column], 3)] for t, values in rows]}
                      for key, label, unit, column in (
                          ("prs_sfc_min_hpa", "lowest pressure at 25 m (not sea level)", "hPa", 3),
                          ("wind_sfc_max", "fastest wind at 25 m, ground-relative", "m/s", 2),
                          ("rain_max_cm", "most rain in one cell", "cm", 4),
                          ("dbz_max", "highest reflectivity", "dBZ", 1))]
            write_json_atomic(out / "series.json", {
                "schema": "gm1.quicklook-series/v1", "gm1_label": "GM1, fake runner", "note": FAKE,
                "time_unit": "h", "storm_kind": "hurricane", "charts": charts})
            return
        # A made-up time series in the shape of cm1core.quicklook.series (gm1.quicklook-series/v1).
        charts = []
        for key, label, unit, scale in (("wmax", "strongest updraft", "m/s", 50.0),
                                        ("wmin", "strongest downdraft", "m/s", -25.0),
                                        ("uh_max", "max 2–5 km updraft helicity", "m²/s²", 600.0),
                                        ("thpert_sfc_min", "coldest θ′ at 250 m", "K", -6.0),
                                        ("coldpool_area_km2_lt_minus1K", "cold pool, θ′ < −1 K", "km²", 1800.0),
                                        ("rain_max_cm", "most rain in one cell", "cm", 6.0)):
            charts.append({"key": key, "label": label, "unit": unit,
                           "gm1": [[t, round(scale * min(1.0, t / 3600.0), 3)] for t in times]})
        write_json_atomic(out / "series.json", {"schema": "gm1.quicklook-series/v1", "gm1_label": "GM1, fake runner",
                                                "note": FAKE, "charts": charts})

    nc_times = _netcdf_times(rundir)
    if nc_times is not None:
        # CM1 netCDF output: every saved time at once, as the real command draws files that are all there
        for index, t in enumerate(nc_times):
            convert(index, {"model_time_s": t})
        return 0
    return _follow(rundir, arguments.follow, convert)


# ------------------------------------------------------------ the command line

def cmd_doctor(arguments: argparse.Namespace) -> int:
    from ..cm1run.system import doctor_plain

    if not arguments.plain:
        print(json.dumps({"fake_runner": FAKE, "blocking": []}))
        return 0
    arguments.root = Path(arguments.root) if arguments.root else None
    arguments.native_dir = Path(arguments.native_dir) if arguments.native_dir else None
    return doctor_plain(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuwm-cm1 (fake runner)")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("import")
    command.add_argument("namelist")
    command.add_argument("--sounding")
    command.add_argument("--out", required=True)
    command.add_argument("--acknowledge-output-keys", action="store_true")
    command.add_argument("--acknowledge-thompson", action="store_true")
    command.add_argument("--seed", type=int)
    command.add_argument("--json", action="store_true")
    # The column of a WRF or ArWen run as the sounding, taken by the real column reader.
    for flag in ("--wrf", "--wrf-time", "--wrf-at", "--wrf-ij"):
        command.add_argument(flag)
    command.add_argument("--wrf-frame", default="namelist")  # this stand-in keeps the namelist's frame
    command.add_argument("--wrf-record", type=int)
    command.set_defaults(handler=cmd_import)
    command = commands.add_parser("edit")
    command.add_argument("rundir")
    command.add_argument("--set", dest="sets", action="append", default=[])
    command.add_argument("--from", dest="source")
    command.add_argument("--setting", dest="settings", action="append", default=[])
    command.add_argument("--reset-setting", dest="resets", action="append", default=[])
    command.add_argument("--sounding")
    command.add_argument("--acknowledge-output-keys", action="store_true")
    command.add_argument("--acknowledge-thompson", action="store_true")
    command.add_argument("--drop-acknowledgments", action="store_true")
    command.add_argument("--seed", type=int)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=cmd_edit)
    command = commands.add_parser("jobscript")
    command.add_argument("rundir")
    command.add_argument("--scheduler", required=True)
    command.add_argument("--time", default="04:00:00")
    command.add_argument("--partition")
    command.add_argument("--account")
    command.add_argument("--qos")
    command.add_argument("--mem")
    command.add_argument("--gpu-type")
    command.add_argument("--cpus", type=int)
    command.add_argument("--until")
    command.add_argument("--setup", action="append", default=[])
    command.add_argument("--python")
    command.add_argument("--write", action="store_true")
    command.add_argument("--library")
    command.add_argument("--history-publisher")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=cmd_jobscript)
    command = commands.add_parser("check")
    command.add_argument("config")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=cmd_check)
    command = commands.add_parser("cartesian")
    command.add_argument("config")
    command.add_argument("--output", required=True)
    command.add_argument("--plan", action="store_true")
    command.add_argument("--sounding")
    command.set_defaults(handler=cmd_cartesian)
    command = commands.add_parser("estimate")
    command.add_argument("target", nargs="?")
    command.add_argument("--each", action="append")
    command.add_argument("--gpu", type=int)
    command.add_argument("--root")
    command.add_argument("--no-write", action="store_true")
    command.add_argument("--measure", action="store_true")
    command.add_argument("--library")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=cmd_estimate)
    command = commands.add_parser("status")
    command.add_argument("rundir")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=cmd_status)
    command = commands.add_parser("run")
    command.add_argument("rundir")
    command.add_argument("--library")
    command.add_argument("--history-publisher")
    # The real run's words, so a guide's line runs here unchanged (tools/release/doc_commands.py).
    command.add_argument("--until")
    for flag in ("--no-wait", "--ignore-estimate", "--json", "--continue", "--start-over"):
        command.add_argument(flag, action="store_true")
    command.add_argument("--gpu")
    command.set_defaults(handler=cmd_run)
    command = commands.add_parser("resume")
    command.add_argument("rundir")
    command.add_argument("--from-restart", type=int)
    command.add_argument("--from-segment", type=int)
    command.add_argument("--until")
    for flag in ("--no-wait", "--ignore-estimate", "--json"):
        command.add_argument(flag, action="store_true")
    command.add_argument("--gpu")
    command.add_argument("--library")
    command.add_argument("--history-publisher")
    command.set_defaults(handler=lambda arguments: cmd_run(arguments, resume=True))
    for name, handler in (("cm1out", cmd_cm1out), ("quicklook", cmd_quicklook)):
        command = commands.add_parser(name)
        command.add_argument("rundir")
        command.add_argument("--follow", action="store_true")
        command.add_argument("--format", default="nc")
        command.add_argument("--force", action="store_true")
        # The real sidecars' other words, accepted so a guide's line runs here unchanged.
        command.add_argument("--out")
        command.add_argument("--namelist")
        command.add_argument("--deck")
        command.add_argument("--poll-s", type=float)
        if name == "cm1out":
            command.add_argument("--filetype")
            command.add_argument("--also-netcdf", action="store_true")
            command.add_argument("--no-stats", action="store_true")
            command.add_argument("--no-pdata", action="store_true")
            command.add_argument("--deflate", type=int)
            command.add_argument("--json", action="store_true")
        if name == "quicklook":
            command.add_argument("--only")
            command.add_argument("--label")
            command.add_argument("--slice")
            command.add_argument("--section")
            command.add_argument("--column", action="store_true")
            command.add_argument("--fields", action="store_true")
            command.add_argument("--height", type=float)
            command.add_argument("--level", type=int)
            command.add_argument("--line", nargs=4, type=float)
            command.add_argument("--from", dest="point_a", nargs=2, type=float)
            command.add_argument("--to", dest="point_b", nargs=2, type=float)
            command.add_argument("--top", type=float)
            command.add_argument("--at", nargs=2, type=float)
            command.add_argument("--time", type=float)
            command.add_argument("--json", action="store_true")
            # Films and the looks drawn once per run
            command.add_argument("--radar-site", nargs="+")
            command.add_argument("--radar-tilt", type=float)
            command.add_argument("--radar-motion", nargs="+")
            command.add_argument("--film")
            command.add_argument("--look", choices=("environment", "parcels", "run-card", "updraft-time-height",
                                                    "spin-time-height", "storm3d"))
            command.add_argument("--bearing", type=float)
            command.add_argument("--size")
            command.add_argument("--orbit-frames", type=int)
            command.add_argument("--fps", type=float)
            command.add_argument("--between", type=int)
            # Compare with other runs
            command.add_argument("--compare", nargs="+")
            command.add_argument("--box", type=float)
        command.set_defaults(handler=handler)
    command = commands.add_parser("render")
    command.add_argument("rundir")
    command.add_argument("--product", required=True)
    command.add_argument("--restart")
    command.add_argument("--field", action="append", default=[])
    command.add_argument("--height", type=float)
    command.add_argument("--level", type=int)
    command.add_argument("--output")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=cmd_render)
    control.add_commands(commands)
    # The System screen's check and the reproduce bundle read files and nvidia-smi only, so the fake runner runs
    # the real ones (a test points GPUWM_CM1_NVIDIA_SMI at a fixture).
    command = commands.add_parser("doctor")
    command.add_argument("--plain", action="store_true")
    command.add_argument("--root")
    command.add_argument("--native-dir")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=cmd_doctor)
    from ..cm1run.reproduce import add_command as add_reproduce_command

    add_reproduce_command(commands)
    from ..cm1import.sounding_door import add_parser as add_sounding_parser

    add_sounding_parser(commands)  # the real command; it opens no GPU
    from ..chain_door import add_command as add_chain_command

    add_chain_command(commands)  # the real command; it opens no GPU
    from ..cm1run.presets import add_preset_parser

    add_preset_parser(commands)
    commands.choices["preset"].set_defaults(handler=cmd_preset)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
