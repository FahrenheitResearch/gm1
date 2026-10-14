"""``gpuwm-cm1 import NAMELIST [--sounding F] --out RUNDIR``: a CM1 namelist into a run directory.

Reads the namelist as CM1 r21.1 would, writes the six-category coverage report
(``cm1import.coverage``) and, when the namelist runs, the gpuwm-cm1 deck.  The
run directory holds:

``namelist.input.original``  the user's file, byte for byte (kept on re-import)
``namelist.input``           the run's namelist
``namelist.diff``            its difference from the original, when there is one
``input_sounding``           the sounding, byte for byte (isnd = 7 or 17)
``import-report.json/.txt``  the coverage report
``settings.toml``            what the importer chose that the namelist does not say
``deck.toml``                the generated gpuwm-cm1 configuration; never hand-edited
``run.json``                 the run directory's record (``cm1run.rundir``), written last

Exit codes: 0 runs; 2 refused (every refusal is listed, in one pass); 3 runs
once acknowledged (output keys GM1 does not read: ``--acknowledge-output-keys``;
ptype = 3: ``--acknowledge-thompson``).

Importing into an existing run directory re-imports it: allowed while the
directory holds an import and no run (``segments/``), which is how an edited
namelist is checked again.  ``namelist.input.original`` stays the first file.

Nothing here opens a GPU or imports CuPy.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPORT_JSON = "import-report.json"
REPORT_TEXT = "import-report.txt"
ORIGINAL = "namelist.input.original"
NAMELIST = "namelist.input"
DIFF = "namelist.diff"
SOUNDING = "input_sounding"
SETTINGS = "settings.toml"
DECK = "deck.toml"
RUN_JSON = "run.json"
SETTINGS_SCHEMA = "gm1.cm1-run-settings/v1"


class ImportDoorError(ValueError):
    """The run directory cannot take this import; the message says why."""


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _atomic_write(path: Path, data: bytes) -> None:
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.chmod(temporary, 0o666 & ~_umask())
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def prepare_rundir(rundir: Path, *, staged_namelist: bool = False) -> bool:
    """Create ``rundir`` or check it can be re-imported; True for a re-import.

    ``staged_namelist``: a new run directory whose ``namelist.input`` a
    caller wrote itself (a guided preset's fitted namelist) counts as new.
    """

    if rundir.is_symlink():
        raise ImportDoorError(f"{rundir} is a symbolic link; import into a real directory")
    if not rundir.exists():
        rundir.mkdir(parents=True)
        return False
    if not rundir.is_dir():
        raise ImportDoorError(f"{rundir} exists and is not a directory")
    entries = {entry.name for entry in rundir.iterdir()}
    if not entries:
        return False
    if staged_namelist and {entry for entry in entries if not entry.startswith(".")} == {NAMELIST}:
        return False
    if (rundir / "segments").exists():
        raise ImportDoorError(
            f"{rundir} holds a run (segments/); a run directory's namelist is fixed once it has run. "
            "Import into a new directory"
        )
    if REPORT_JSON in entries and ORIGINAL in entries:
        return True  # an import; the GUI may have added its own files (commands.log) since
    # An import interrupted before its report holds only import files.
    import_files = {ORIGINAL, NAMELIST, DIFF, SOUNDING, SETTINGS, DECK, REPORT_JSON, REPORT_TEXT, RUN_JSON, "control"}
    hidden = {entry for entry in entries if entry.startswith(".")}
    if ORIGINAL in entries and entries <= import_files | hidden:
        return True
    raise ImportDoorError(
        f"{rundir} exists and is not an imported run directory; import into a new or empty directory"
    )


#: config.py's range for initialization.random_seed with irandp = 1: nonzero,
#: and one unsigned 64-bit word across the initializer ABI.
SEED_RANGE = (1, 2**64 - 1)


def check_seed(seed: int | None) -> int | None:
    """``seed`` if config.py admits it beside irandp = 1; ImportDoorError otherwise."""

    if seed is None:
        return None
    low, high = SEED_RANGE
    if isinstance(seed, bool) or not isinstance(seed, int) or not low <= seed <= high:
        raise ImportDoorError(f"--seed {seed!r}: the random seed must be a whole number from {low} to {high}")
    return seed


def _seed_argument(text: str) -> int:
    try:
        return check_seed(int(text))  # type: ignore[return-value]
    except (ValueError, ImportDoorError) as error:
        raise argparse.ArgumentTypeError(
            str(error) if isinstance(error, ImportDoorError) else f"{text!r} is not a whole number"
        ) from error


def previous_seed(rundir: Path) -> int | None:
    """The seed an earlier import of ``rundir`` chose (settings.toml), so a re-import keeps it."""

    import tomllib

    try:
        settings = tomllib.loads((rundir / SETTINGS).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return None
    seed = settings.get("initialization", {}).get("random_seed")
    try:
        return check_seed(seed)
    except ImportDoorError:
        return None


def import_command(
    namelist: Path | str,
    rundir: Path | str,
    *,
    sounding: Path | str | None = None,
    acknowledge: Sequence[str] = (),
    seed: int | None = None,
    library: Path | str | None = None,
    json_output: bool = False,
    wrf: Mapping[str, Any] | None = None,
) -> str:
    """The ``gpuwm-cm1 import`` line that repeats an import (the GUI's "Show command").

    ``wrf``: ``{"file", "time" | "record", "at" | "ij"}`` for a sounding from a WRF or ArWen run.
    """

    argv = ["gpuwm-cm1", "import", str(namelist)]
    argv += wrf_import_args(wrf)
    argv += ["--out", str(rundir)]
    if sounding is not None:
        argv += ["--sounding", str(sounding)]
    if "output-keys" in acknowledge:
        argv.append("--acknowledge-output-keys")
    if "thompson" in acknowledge:
        argv.append("--acknowledge-thompson")
    if seed is not None:
        argv += ["--seed", str(seed)]
    if library is not None:
        argv += ["--library", str(library)]
    if json_output:
        argv.append("--json")
    return shlex.join(argv)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value) if isinstance(value, float) else str(value)


def settings_toml(settings: Mapping[str, Any], output: Mapping[str, Any]) -> str:
    lines = [
        "# What the importer chose that the CM1 namelist does not say.",
        "# The GUI's Settings screen edits it; deck.toml is generated from the namelist and this file.",
        f'schema = "{SETTINGS_SCHEMA}"',
    ]
    tables = {name: dict(values) for name, values in settings.items()}
    tables.setdefault("output", {})
    tables["output"].setdefault("format", output.get("output_format"))
    tables["output"].setdefault("filetype", output.get("output_filetype"))
    tables["output"].setdefault("also_netcdf", False)
    for name in sorted(tables):
        lines.append("")
        lines.append(f"[{name}]")
        for key in sorted(tables[name]):
            value = tables[name][key]
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _parse_error_message(error: Exception, target: Path, source: Path, data: bytes) -> str:
    """The parser's message about the user's file: its name, and where a non-ASCII byte is.

    The parser reads the run directory's copy, so its message names that path;
    the user edits their own file.  It refuses any byte outside ASCII without
    saying where, and such a byte is usually a character pasted into a comment
    (a degree sign, a dash, a curly quote), so the line and column are added.
    """

    import re

    message = str(error).replace(str(target), str(source))
    offset = re.search(r"\bbyte (\d+)\b", message)
    if offset is not None and int(offset.group(1)) <= len(data):
        before = data[: int(offset.group(1))]
        line = before.count(b"\n") + 1
        column = len(before) - (before.rfind(b"\n") + 1) + 1
        message = message.replace(offset.group(0), f"{offset.group(0)} (line {line}, column {column})", 1)
    if "ASCII" in message:
        for number, line in enumerate(data.split(b"\n"), start=1):
            column = next((index for index, byte in enumerate(line) if byte >= 0x80), None)
            if column is not None:
                shown = line.decode("utf-8", "replace").strip()
                message += (f": line {number}, column {column + 1} holds a non-ASCII character "
                            f"({shown[:60]!r}); replace it with plain ASCII, even in a comment")
                break
    return message


def _parse_error_report(namelist: Path, error: Exception | str, command: str) -> dict[str, Any]:
    return {
        "schema": "gm1.cm1-import-report/v1",
        "title": "Check",
        "namelist": {"path": str(namelist), "name": namelist.name},
        "verdict": "refused",
        "verdict_text": "Refused: the namelist cannot be read",
        "exit": 2,
        "error": str(error),
        "sections": [
            {"id": "refused", "label": "Refused", "count": 1, "rows": [{
                "key": "namelist", "keys": ["namelist"], "value": namelist.name, "text": str(error),
                "fix": "correct the line the message names; CM1's own reader stops there too",
                "plain": "The namelist file cannot be read.", "plain_fix": "Fix the line the message names.",
                "code": "CM1-IMPORT-NAMELIST-SYNTAX", "stage": "namelist",
            }]},
        ],
        "command": command,
    }


def wrf_import_args(wrf: Mapping[str, Any] | None) -> list[str]:
    """``--wrf FILE [--wrf-time T | --wrf-record N] (--wrf-at LAT,LON | --wrf-ij I,J)``, as a shell line spells it."""

    if not wrf:
        return []
    args = ["--wrf", str(wrf["file"])]
    if wrf.get("time"):
        args += ["--wrf-time", str(wrf["time"])]
    elif wrf.get("record") is not None:
        args += ["--wrf-record", str(int(wrf["record"]))]
    if wrf.get("at") is not None:
        lat, lon = wrf["at"]
        args.append(f"--wrf-at={float(lat):g},{float(lon):g}")
    elif wrf.get("ij") is not None:
        i, j = wrf["ij"]
        args += ["--wrf-ij", f"{int(i)},{int(j)}"]
    if wrf.get("frame") and wrf["frame"] != "namelist":
        args += ["--wrf-frame", str(wrf["frame"])]
    return args


#: Where ``import --wrf`` keeps the column's record in the run directory.
WRF_COLUMN = "wrf-column.json"
#: ``--wrf-frame``: keep the namelist's moving frame and bubble, or follow the storm the column makes.
WRF_FRAMES = ("namelist", "follow")


def follow_the_storm(effective: Mapping[str, Any], sounding_bytes: bytes) -> dict[str, Any]:
    """The moving frame and bubble centre that keep this column's storm inside the namelist's domain.

    The storm-track check the importer runs (``cartesian_door.require_run_geometry`` with
    ``run_geometry.evaluate_run_geometry``) on the namelist's own grid, run length, frame, bubble and open sides,
    and, when the tracked storm would leave, its own remedy (``RunGeometry.recommendation``) with the frame rounded
    to 1/8 m/s (a frame must be a binary32 word, as ``cm1run.estimate.storm_track`` rounds it) and the bubble
    placed for that frame.  Nothing here decides admission: the import that follows runs the same check again.
    Returns ``{"action": "keep" | "follow" | "domain", ...}``.
    """

    from ..run_geometry import MEMBER_RIGHT, evaluate_run_geometry, hodograph_from_input_sounding

    nx, ny = int(effective["nx"]), int(effective["ny"])
    dx, dy = float(effective["dx"]), float(effective["dy"])
    duration = float(effective["timax"])
    moving = int(effective.get("imove", 0)) == 1
    frame = (float(effective.get("umove", 0.0)), float(effective.get("vmove", 0.0))) if moving else (0.0, 0.0)
    sides = tuple(int(effective.get(key, 2)) != 1 for key in ("wbc", "ebc", "sbc", "nbc"))
    if not any(sides):
        return {"action": "keep", "why": "every side repeats (periodic), so a storm cannot leave the domain"}
    origin = (0.0, 0.0) if int(effective.get("iorigin", 1)) == 1 else (-0.5 * nx * dx, -0.5 * ny * dy)
    corner_centre = (0.5 * nx * dx, 0.5 * ny * dy)
    hodograph = hodograph_from_input_sounding(sounding_bytes)

    def evaluate(frame_m_s: tuple[float, float], centre: tuple[float, float]) -> Any:
        return evaluate_run_geometry(hodograph=hodograph, nx=nx, ny=ny, dx_m=dx, dy_m=dy, duration_s=duration,
                                     frame_m_s=frame_m_s, bubble_center_xy_m=centre, open_sides=sides)

    geometry = evaluate(frame, corner_centre)
    if geometry.members[MEMBER_RIGHT].stays_inside:
        return {"action": "keep", "why": "the namelist's frame and bubble keep the predicted storm inside"}
    plan = geometry.recommendation()
    if plan["action"] not in ("move-the-bubble", "change-the-frame"):
        extent = plan.get("required_domain_extent_m") or [None, None]
        return {"action": "domain", "why": "no frame keeps the storm inside this domain for the whole run",
                "required_domain_km": [None if v is None else v / 1000.0 for v in extent]}
    rounded = tuple(round(v * 8.0) / 8.0 for v in plan["frame_m_s"])
    paired = evaluate(rounded, corner_centre)
    bubble = paired.recommended_bubble_center_xy_m
    check = evaluate(rounded, tuple(bubble))
    return {
        "action": "follow",
        "umove": rounded[0], "vmove": rounded[1], "imove": 1,
        "bubble_center_x_m": round(bubble[0] + origin[0], 1), "bubble_center_y_m": round(bubble[1] + origin[1], 1),
        "stays_inside": bool(check.members[MEMBER_RIGHT].stays_inside),
        "rule": "run_geometry.RunGeometry.recommendation, frame rounded to 1/8 m/s, bubble placed for that frame",
        "was": {"imove": int(moving), "umove": frame[0], "vmove": frame[1]},
    }


def run_wrf_import(namelist: Path, rundir: Path, wrf: Mapping[str, Any], *, command: str, frame: str = "namelist",
                   **options: Any) -> tuple[int, dict[str, Any], str]:
    """``import NAMELIST --wrf FILE ...``: the column as ``input_sounding``, then the import with ``isnd = 7``.

    The user's namelist is kept as ``namelist.input.original``; the run's ``namelist.input`` is it with
    ``isnd = 7`` (the change shows in ``namelist.diff``).  The column's record is ``RUNDIR/wrf-column.json``.
    A column that does not reach the namelist's highest scalar level is refused before anything is written.
    ``frame = "follow"`` also sets the moving frame and the bubble centre to the storm-track check's remedy when the
    namelist's own would lose the column's storm (:func:`follow_the_storm`); the change shows in the diff and in
    ``settings.toml``, and ``wrf-column.json`` records it.
    """

    from ..cm1run.fit import FitError, set_namelist_values
    from ..wrf_history import open_history
    from . import sounding as sounding_module, wrfcolumn

    source = Path(namelist).expanduser()
    if not source.is_file():
        raise ImportDoorError(f"{source} is not a file")
    rundir = Path(rundir).expanduser()
    if rundir.exists() and any(rundir.iterdir()):
        raise ImportDoorError(f"{rundir} is not empty; --wrf makes a new run directory. Import into a new one")
    original = source.read_bytes()
    try:
        text = set_namelist_values(original.decode("ascii"), {"isnd": 7})
    except (UnicodeDecodeError, FitError) as error:
        raise ImportDoorError(f"isnd = 7 could not be written into {source}: {error}") from error
    from . import coverage, namelist as namelist_module

    tool = options.get("tool") or namelist_module.locate_tool()
    # The column is read and checked before the run directory exists, so a refusal leaves nothing behind.
    effective: Mapping[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="gm1-wrf-import-") as scratch:
        probe = Path(scratch) / NAMELIST
        probe.write_bytes(text.encode("ascii"))
        try:
            effective = coverage.import_for_report(probe, tool=tool).effective
            top = sounding_module.highest_mass_level_m(effective)
        except Exception:  # noqa: BLE001 - the import below reports a namelist it cannot read
            top = None
    with open_history(wrf["file"]) as history:
        record = history.record_of(wrf["time"]) if wrf.get("time") else history.check_record(int(wrf.get("record") or 0))
        column = wrfcolumn.extract_column(history, record, at=wrf.get("at"), ij=wrf.get("ij"), need_top_m=top)
    if frame not in WRF_FRAMES:
        raise ImportDoorError(f"--wrf-frame is one of {', '.join(WRF_FRAMES)}, not {frame!r}")
    followed: dict[str, Any] = {"asked": frame}
    settings = dict(options.pop("settings", None) or {})
    if frame == "follow" and effective is not None:
        try:
            followed.update(follow_the_storm(effective, column.sounding_bytes()))
        except (TypeError, ValueError, KeyError) as error:
            followed.update({"action": "keep", "why": f"the storm track could not be predicted: {error}"})
        if followed.get("action") == "follow":
            try:
                text = set_namelist_values(text, {"imove": 1, "umove": followed["umove"], "vmove": followed["vmove"]})
            except FitError as error:
                raise ImportDoorError(f"the storm's frame could not be written into {source}: {error}") from error
            # init3d.F:1372-1376: with updraft nudging the storm starts at its centre.
            kx, ky = (("initialization.xc_wnudge", "initialization.yc_wnudge")
                      if int(effective.get("iinit", 1)) == 12
                      else ("initialization.bubble_center_x_m", "initialization.bubble_center_y_m"))
            settings.update({kx: followed["bubble_center_x_m"], ky: followed["bubble_center_y_m"]})
    staging = rundir / ".wrf-column"
    staging.mkdir(parents=True, exist_ok=True)
    staged_namelist = rundir / NAMELIST
    _atomic_write(staged_namelist, text.encode("ascii"))
    staged = staging / SOUNDING
    _atomic_write(staged, column.sounding_bytes())
    code, payload, text_report = run_import(staged_namelist, rundir, sounding=staged, command=command,
                                            original=original, tool=tool, settings=settings or None,
                                            **{k: v for k, v in options.items() if k != "tool"})
    column_record = column.record(sounding_file=SOUNDING, command=command)
    column_record["frame"] = followed
    _atomic_write(rundir / WRF_COLUMN, (json.dumps(column_record, indent=1, default=str) + "\n").encode("utf-8"))
    staged.unlink(missing_ok=True)
    try:
        staging.rmdir()
    except OSError:
        pass
    # run.json names where the sounding came from: the history file and its column, not the staged copy.
    run_json = rundir / RUN_JSON
    if run_json.is_file():
        run = json.loads(run_json.read_text(encoding="utf-8"))
        if isinstance(run.get("sounding"), dict):
            run["sounding"]["source_path"] = os.fspath(Path(wrf["file"]).expanduser().absolute())
            run["sounding"]["source_column"] = WRF_COLUMN
            from ..cm1run.rundir import write_json_atomic

            write_json_atomic(run_json, run)
    payload["wrf_column"] = {"file": WRF_COLUMN, "source": column_record["source"], "point": column_record["point"],
                             "coverage_note": column_record["coverage_note"],
                             "hydrostatic": column_record["hydrostatic"], "frame": followed}
    return code, payload, text_report + "\n" + frame_sentence(followed) + column_record["coverage_note"] + "\n"


def frame_sentence(followed: Mapping[str, Any]) -> str:
    """What ``--wrf-frame follow`` did, as one line of the text report (empty when it was not asked for)."""

    if followed.get("asked") != "follow":
        return ""
    action = followed.get("action")
    if action == "follow":
        was = followed.get("was") or {}
        before = (f"{float(was.get('umove', 0.0)):g}, {float(was.get('vmove', 0.0)):g} m/s" if was.get("imove")
                  else "no moving frame")
        line = (f"Frame: moved to follow this column's storm: umove = {followed['umove']:g}, vmove = "
                f"{followed['vmove']:g} m/s (was {before}), bubble at ({followed['bubble_center_x_m']:g}, "
                f"{followed['bubble_center_y_m']:g}) m; "
                + ("the storm-track check keeps it inside." if followed.get("stays_inside")
                   else "the storm-track check still says it leaves; see Refused."))
    elif action == "domain":
        need = followed.get("required_domain_km") or [None, None]
        size = f": it needs at least {need[0]:.0f} x {need[1]:.0f} km" if None not in need else ""
        line = (f"Frame: no moving frame keeps this column's storm inside this domain for the whole run{size} "
                "(widen nx and ny, or shorten timax).")
    else:
        line = f"Frame: kept the namelist's; {followed.get('why', '')}."
    return line + "\n"


def run_import(
    namelist: Path,
    rundir: Path,
    *,
    sounding: Path | None = None,
    acknowledge: Sequence[str] = (),
    seed: int | None = None,
    library: Path | None = None,
    environment: bool = True,
    command: str | None = None,
    tool: Any = None,
    settings: Mapping[str, Any] | None = None,
    reset_settings: Sequence[str] = (),
    overrides: Mapping[str, Any] | None = None,
    original: bytes | None = None,
) -> tuple[int, dict[str, Any], str]:
    """Import ``namelist`` into ``rundir``; returns (exit code, report JSON, report text).

    ``overrides`` are GM1 settings beyond the namelist (dotted gpuwm-cm1 keys;
    written to settings.toml and the deck).  One that is a setting the user may
    change (``usersettings.EDITABLE``, by its deck key: a guided preset's bubble,
    the fit's memory budget) is kept as the run's own setting, so the Settings
    screen shows it and a re-import keeps it.  ``original`` is the file a new
    run directory keeps as ``namelist.input.original`` when the namelist
    was written from another one (a guided preset and its fit): the diff
    then shows what changed.  A re-import always keeps the first original.

    A re-import keeps the run's own settings from ``settings.toml`` (the
    values CM1 keeps in Fortran, ``usersettings.EDITABLE``); ``settings``
    changes some and ``reset_settings`` puts some back to CM1's value.
    """

    from . import coverage, levels as levels_module, namelist as namelist_module, sounding as sounding_module
    from . import usersettings

    source = Path(namelist).expanduser()
    if not source.is_file():
        raise ImportDoorError(f"{source} is not a file")
    seed = explicit_seed = check_seed(seed)
    original_bytes_given = None if original is None else bytes(original)
    if tool is None:
        tool = namelist_module.locate_tool()  # before anything is written
    rundir = Path(rundir).expanduser()
    extra = dict(overrides or {})
    promoted: dict[str, Any] = {}
    for item in usersettings.EDITABLE:
        if item.deck is not None and item.deck in extra:
            promoted[item.key] = extra.pop(item.deck)
    changes = {**promoted, **dict(settings or {})}
    stored = usersettings.read_file(rundir / SETTINGS) if (rundir / SETTINGS).is_file() else {}
    try:
        user_settings = usersettings.merge(stored, changes, tuple(reset_settings))
    except usersettings.SettingError as error:
        raise ImportDoorError(str(error)) from error
    reimport = prepare_rundir(rundir, staged_namelist=original is not None)
    if not reimport:
        user_settings = usersettings.merge({}, changes, tuple(reset_settings))
    if reimport and seed is None:
        # Re-importing an edited namelist keeps the storm's random stream: the
        # seed the first import chose (or --seed gave) stays unless one is given.
        seed = previous_seed(rundir)
    source_bytes = source.read_bytes()
    original = rundir / ORIGINAL
    if not reimport or not original.exists():
        _atomic_write(original, source_bytes if original_bytes_given is None else original_bytes_given)
    target = rundir / NAMELIST
    if source.resolve() != target.resolve():
        _atomic_write(target, source_bytes)
    original_bytes = original.read_bytes()
    diff_path = rundir / DIFF
    if original_bytes != source_bytes:
        diff = difflib.unified_diff(
            original_bytes.decode("utf-8", "replace").splitlines(keepends=True),
            source_bytes.decode("utf-8", "replace").splitlines(keepends=True),
            fromfile=ORIGINAL, tofile=NAMELIST,
        )
        _atomic_write(diff_path, "".join(diff).encode("utf-8"))
    else:
        diff_path.unlink(missing_ok=True)

    command = command or import_command(source, rundir, sounding=sounding, acknowledge=acknowledge,
                                        seed=explicit_seed, library=library)
    try:
        imported = coverage.import_for_report(target, tool=tool)
    except namelist_module.NamelistError as error:
        message = _parse_error_message(error, target, source, source_bytes)
        payload = _parse_error_report(source, message, command)
        text = f"CM1 namelist: {source}\nResult: {payload['verdict_text']}\n\n{message}\n\nCommand: {command}\n"
        _atomic_write(rundir / REPORT_JSON, (json.dumps(payload, indent=1) + "\n").encode("utf-8"))
        _atomic_write(rundir / REPORT_TEXT, text.encode("utf-8"))
        (rundir / DECK).unlink(missing_ok=True)
        _record_run(rundir, payload, source=source, effective=None, sounding=None)
        return 2, payload, text

    effective = imported.effective
    sounding_path = None
    if int(effective.get("isnd", 0)) in sounding_module.FILE_SOUNDINGS:
        located = sounding_module.locate(source, sounding)
        if located is None and (rundir / SOUNDING).is_file():
            located = rundir / SOUNDING  # a re-import keeps the sounding it was given
        if located is not None and located.is_file():
            if located.resolve() != (rundir / SOUNDING).resolve():
                _atomic_write(rundir / SOUNDING, located.read_bytes())
            sounding_path = rundir / SOUNDING
        elif located is not None:
            sounding_path = located  # the report names the missing file
    # stretch_z = 3 and 4 read their heights from input_grid_z (param.F:8013, 8083): the
    # file beside the namelist, or the copy this run folder already keeps.
    levels_path = None
    try:
        level_branch = int(effective.get("stretch_z", 0))
    except (TypeError, ValueError):
        level_branch = 0
    if level_branch in (3, 4):
        located_levels = levels_module.locate(source, rundir)
        if located_levels is not None:
            try:
                heights = levels_module.parse(located_levels.read_bytes(), stretch_z=level_branch,
                                              nz=int(effective.get("nz", 0)))
            except levels_module.LevelsFileError as error:
                raise ImportDoorError(f"{located_levels}: {error}") from error
            if located_levels.resolve() != (rundir / levels_module.LEVELS_FILE).resolve():
                _atomic_write(rundir / levels_module.LEVELS_FILE, located_levels.read_bytes())
            levels_path = rundir / levels_module.LEVELS_FILE
            extra["grid.levels_m"] = list(heights)
    report = coverage.build_report(
        imported,
        sounding=sounding_path,
        acknowledge=acknowledge,
        seed=seed,
        library=library,
        environment=environment,
        command=command,
        overrides={**usersettings.deck_overrides(user_settings), **extra},
    )
    report.namelist = {**report.namelist, "path": str(source)}
    for table, values in usersettings.as_tables(user_settings).items():
        if table not in ("initialization", "lagrangian"):  # those two are recorded by the report builder
            report.settings.setdefault(table, {}).update(values)
    payload = report.as_dict()
    from .setup import setup_block

    payload["setup"] = setup_block(effective, report, user_settings, rundir=rundir)
    payload["namelist"]["original_sha256"] = hashlib.sha256(original_bytes).hexdigest()
    payload["namelist"]["edited"] = original_bytes != source_bytes
    payload["run_directory"] = str(rundir)
    payload["reimport"] = reimport
    payload["files"] = {
        "original": ORIGINAL, "namelist": NAMELIST, "diff": DIFF if original_bytes != source_bytes else None,
        "sounding": SOUNDING if sounding_path is not None and sounding_path.parent == rundir else None,
        "levels": levels_module.LEVELS_FILE if levels_path is not None else None,
        "settings": SETTINGS, "deck": DECK if report.toml is not None else None,
    }
    text = report.text()
    _atomic_write(rundir / SETTINGS, settings_toml(report.settings, report.output).encode("utf-8"))
    if report.toml is not None:
        _atomic_write(rundir / DECK, report.toml.encode("utf-8"))
    else:
        (rundir / DECK).unlink(missing_ok=True)
    _atomic_write(rundir / REPORT_JSON, (json.dumps(payload, indent=1, default=str) + "\n").encode("utf-8"))
    _atomic_write(rundir / REPORT_TEXT, text.encode("utf-8"))
    _record_run(rundir, payload, source=source, effective=effective,
                sounding=sounding_path if sounding_path is not None and sounding_path.parent == rundir else None,
                sounding_source=sounding)
    return report.exit_code, payload, text


def _record_run(rundir: Path, payload: Mapping[str, Any], *, source: Path, effective: Mapping[str, Any] | None,
                sounding: Path | None, sounding_source: Path | None = None) -> None:
    """``run.json`` (``gm1.cm1-run/v1``), last: what makes the folder a run directory.

    ``gpuwm-cm1 run``, ``status``, ``estimate`` and the GUI read it
    (``cm1run.rundir``); its ``import`` block is this report's verdict, its
    acknowledgments the ones this import was given.
    """

    from ..cm1run.rundir import record_import

    record_import(rundir, payload, source=source, effective=effective, sounding=sounding,
                  sounding_source=sounding_source)


def add_parser(commands: Any) -> None:
    """Register ``gpuwm-cm1 import`` on the CLI's subparsers."""

    command = commands.add_parser(
        "import",
        help="read a CM1 namelist, report what GM1 runs as CM1 does, and write a run directory",
        description=(
            "Read a CM1 r21.1 namelist.input (older and r22.0 keys are read by value), write the "
            "six-category coverage report and, when it runs, the gpuwm-cm1 deck into RUNDIR. "
            "Exit 0: runs; 2: refused (every refusal listed); 3: runs once acknowledged."
        ),
    )
    command.add_argument("namelist", type=Path, help="CM1 namelist.input")
    command.add_argument("--out", type=Path, required=True, metavar="RUNDIR",
                         help="new or empty run directory, or an imported one that has not run (re-import)")
    command.add_argument("--sounding", type=Path, metavar="FILE",
                         help="input_sounding for isnd = 7 or 17 (default: input_sounding beside the namelist)")
    command.add_argument("--acknowledge-output-keys", action="store_true",
                         help="run without the output settings GM1 does not read (the 'Not read' rows)")
    command.add_argument("--acknowledge-thompson", action="store_true",
                         help="run ptype = 3 (Thompson), which has been checked only in a 120-s column")
    command.add_argument("--seed", type=_seed_argument, default=None,
                         help="random seed for irandp = 1, 1 to 2**64-1 (default: 1, or the seed a re-imported "
                              "run directory already has); stored in settings.toml")
    command.add_argument("--library", type=Path, default=None,
                         help="Rust initializer library for the sounding's environment block (default: installed)")
    command.add_argument("--json", action="store_true", help="print import-report.json instead of the text report")
    wrf = command.add_argument_group(
        "sounding from a WRF or ArWen run",
        "Write input_sounding from one column of a wrfout or ArWen history and import with isnd = 7: one column's "
        "environment, the same air everywhere, flat ground, no boundary forcing (gpuwm-cm1 sounding from-wrf).")
    wrf.add_argument("--wrf", type=Path, metavar="FILE", help="a wrfout or ArWen history file")
    wrf.add_argument("--wrf-time", metavar="T", help="one of its times, YYYY-MM-DD_HH:MM:SS (default: the first)")
    wrf.add_argument("--wrf-record", type=int, metavar="N", help="or its record number")
    wrf.add_argument("--wrf-at", metavar="LAT,LON", help="the column nearest this point")
    wrf.add_argument("--wrf-ij", metavar="I,J", help="or the 1-based grid column")
    wrf.add_argument("--wrf-frame", choices=WRF_FRAMES, default="namelist",
                     help="namelist: keep the namelist's moving frame and bubble (default); follow: move them so the "
                          "storm-track check keeps this column's storm inside the domain (recorded in the diff)")
    command.set_defaults(handler=import_door)


def import_door(arguments: argparse.Namespace) -> int:
    acknowledge = []
    if arguments.acknowledge_output_keys:
        acknowledge.append("output-keys")
    if arguments.acknowledge_thompson:
        acknowledge.append("thompson")
    from .namelist import NamelistToolError

    if arguments.wrf is not None:
        return _wrf_door(arguments, acknowledge)
    if any(getattr(arguments, name) is not None for name in ("wrf_time", "wrf_record", "wrf_at", "wrf_ij")) or \
            arguments.wrf_frame != "namelist":
        print("gpuwm-cm1 import: --wrf-time, --wrf-record, --wrf-at, --wrf-ij and --wrf-frame go with --wrf FILE",
              file=sys.stderr)
        return 2
    try:
        code, payload, text = run_import(
            arguments.namelist,
            arguments.out,
            sounding=arguments.sounding,
            acknowledge=acknowledge,
            seed=arguments.seed,
            library=arguments.library,
            command=import_command(arguments.namelist, arguments.out, sounding=arguments.sounding,
                                   acknowledge=acknowledge, seed=arguments.seed, library=arguments.library,
                                   json_output=arguments.json),
        )
    except (ImportDoorError, NamelistToolError) as error:
        message = str(error)
        if arguments.json:
            print(json.dumps({"status": "error", "code": "CM1-IMPORT-DOOR", "reason": message,
                              "remedy": "see the reason"}, indent=2), file=sys.stderr)
        else:
            print(f"gpuwm-cm1 import: {message}", file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(payload, indent=1, default=str))
    else:
        sys.stdout.write(text)
    return code


def _wrf_door(arguments: argparse.Namespace, acknowledge: list[str]) -> int:
    from ..errors import CM1Refusal
    from .namelist import NamelistToolError
    from .wrfcolumn import parse_pair

    def fail(message: str, payload: dict[str, Any] | None = None) -> int:
        if arguments.json:
            print(json.dumps(payload or {"status": "error", "code": "CM1-IMPORT-DOOR", "reason": message,
                                         "remedy": "see the reason"}, indent=2, default=str), file=sys.stderr)
        else:
            print(f"gpuwm-cm1 import: {message}", file=sys.stderr)
        return 2

    if arguments.sounding is not None:
        return fail("--sounding and --wrf both give the sounding; give one")
    if bool(arguments.wrf_at) == bool(arguments.wrf_ij):
        return fail("with --wrf, give the column as --wrf-at LAT,LON or --wrf-ij I,J (one of them)")
    if arguments.wrf_time and arguments.wrf_record is not None:
        return fail("give --wrf-time or --wrf-record, not both")
    try:
        wrf: dict[str, Any] = {"file": arguments.wrf, "time": arguments.wrf_time, "record": arguments.wrf_record,
                               "frame": arguments.wrf_frame}
        if arguments.wrf_at:
            wrf["at"] = parse_pair(arguments.wrf_at, "--wrf-at")
        else:
            i, j = parse_pair(arguments.wrf_ij, "--wrf-ij")
            wrf["ij"] = (int(i), int(j))
    except ValueError as error:
        return fail(str(error))
    command = import_command(arguments.namelist, arguments.out, acknowledge=acknowledge, seed=arguments.seed,
                             library=arguments.library, json_output=arguments.json, wrf=wrf)
    try:
        code, payload, text = run_wrf_import(arguments.namelist, arguments.out, wrf, command=command,
                                             frame=arguments.wrf_frame, acknowledge=acknowledge, seed=arguments.seed,
                                             library=arguments.library)
    except CM1Refusal as refusal:
        return fail(refusal.detail.reason + " (" + refusal.detail.remedy + ")", refusal.detail.as_dict())
    except (ImportDoorError, NamelistToolError, OSError) as error:
        return fail(str(error))
    if arguments.json:
        print(json.dumps(payload, indent=1, default=str))
    else:
        sys.stdout.write(text)
    return code


__all__ = [
    "ImportDoorError", "WRF_COLUMN", "WRF_FRAMES", "follow_the_storm", "frame_sentence", "run_wrf_import", "wrf_import_args", "add_parser", "check_seed", "import_command", "import_door", "prepare_rundir",
    "previous_seed", "run_import", "settings_toml",
]
