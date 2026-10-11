"""``gpuwm-cm1 edit RUNDIR``: change an imported run's namelist or its own settings, then check it again.

Every change the GUI's setup screens make is this one command, so a session
in the GUI can be replayed from a terminal:

``--set KEY=VALUE``            a CM1 r21.1 namelist word, written into ``RUNDIR/namelist.input`` in place
                               (the layout, comments and the other lines are kept)
``--from FILE``                the run's whole namelist replaced by FILE (the namelist editor's Save)
``--setting TABLE.KEY=VALUE``  a value CM1 keeps in Fortran, or an output choice, in ``settings.toml``
                               (``usersettings.EDITABLE``: the bubble, cold pool, line thermal, parcel seed box)
``--reset-setting TABLE.KEY``  back to CM1's own value
``--sounding FILE``            a new ``input_sounding`` (isnd = 7 or 17)

Then the run directory is imported again from its own ``namelist.input``,
exactly as ``gpuwm-cm1 import RUNDIR/namelist.input --out RUNDIR`` does:
the coverage report, ``deck.toml`` and ``run.json`` are rewritten,
``namelist.input.original`` stays the user's first file and ``namelist.diff``
holds the difference from it.  The acknowledgments the last import was given
("Run without these settings", Thompson) are kept unless
``--drop-acknowledgments``.  A run directory that has run (``segments/``) is
not edited: its namelist is fixed once it has run.

Exit codes are the importer's: 0 runs, 2 refused, 3 runs once acknowledged.
Nothing here opens a GPU.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .door import (
    NAMELIST, ORIGINAL, REPORT_JSON, SETTINGS, ImportDoorError, _atomic_write, check_seed, run_import,
)

_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


class EditError(ValueError):
    """An edit that cannot be made; the message says why and what to do."""


def parse_fortran(text: str) -> Any:
    """One namelist value as the command line gives it: a number or .true./.false. (a quoted string is
    returned as text, which :func:`typed_value` refuses: no CM1 r21.1 namelist word is character)."""

    raw = text.strip()
    lowered = raw.lower()
    if lowered in (".true.", ".t.", "t", "true"):
        return True
    if lowered in (".false.", ".f.", "f", "false"):
        return False
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        value = float(raw.replace("d", "e").replace("D", "e"))
    except ValueError as error:
        raise EditError(
            f"{text!r} is not a namelist value: give a number, or .true. or .false."
        ) from error
    return value


def split_assignment(text: str, what: str) -> tuple[str, str]:
    key, sep, value = str(text).partition("=")
    key = key.strip().lower()
    if not sep or not key or not value.strip():
        raise EditError(f"{what} takes KEY=VALUE, not {text!r}")
    return key, value.strip()


#: What CM1 r21.1's namelist reader takes for each field type (records/CM1-R21-NAMELIST-DEFAULTS.json).
_TYPE_WORDS = {
    "integer": "a whole number",
    "real32": "a number",
    "real64": "a number",
    "logical": ".true. or .false.",
}


def typed_value(key: str, value: Any, kind: str | None) -> Any:
    """``value`` as CM1 r21.1 reads ``key`` (its field type), or EditError before anything is written.

    Every r21.1 namelist word is an integer, a real or a logical (none is
    character), so a quoted string, a fraction for an integer word or a 1 for a
    logical word is refused here, rather than written into namelist.input for
    CM1's reader to stop on.
    """

    wanted = _TYPE_WORDS.get(kind or "")
    if wanted is None:
        return value
    if kind == "logical":
        if isinstance(value, bool):
            return value
    elif isinstance(value, bool) or isinstance(value, str):
        pass
    elif kind == "integer":
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    raise EditError(f"{key} = {value!r}: CM1 r21.1 reads {key} as {wanted}, so the namelist was not changed")


def namelist_changes(assignments: Sequence[str]) -> dict[str, Any]:
    """``--set`` words as {key: value}; each must be a CM1 r21.1 namelist word, with a value of its type."""

    from . import defaults as _defaults

    inventory = _defaults.load()
    changes: dict[str, Any] = {}
    for text in assignments:
        key, value = split_assignment(text, "--set")
        if not _KEY.match(key) or not inventory.groups_of(key):
            raise EditError(
                f"{key} is not a CM1 r21.1 namelist setting; values CM1 keeps in Fortran "
                "(the bubble, the cold pool, parcels) are changed with --setting TABLE.KEY=VALUE"
            )
        changes[key] = typed_value(key, parse_fortran(value), inventory.type_of(key))
    return changes


def setting_changes(assignments: Sequence[str]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for text in assignments:
        key, value = split_assignment(text, "--setting")
        changes[key] = value
    return changes


def _previous_report(rundir: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads((rundir / REPORT_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def require_editable(rundir: Path) -> None:
    if not rundir.is_dir():
        raise EditError(f"{rundir} is not a folder; import a namelist into it first (gpuwm-cm1 import)")
    if not (rundir / ORIGINAL).is_file() or not (rundir / NAMELIST).is_file():
        raise EditError(f"{rundir} holds no imported namelist; import one first (gpuwm-cm1 import NAMELIST --out {rundir})")
    if (rundir / "segments").exists():
        raise EditError(
            f"{rundir} has run (segments/), and a run's namelist is fixed once it has run. "
            "Import the namelist into a new folder to change it"
        )


def edit_command(rundir: Path | str, *, sets: Sequence[str] = (), settings: Sequence[str] = (),
                 resets: Sequence[str] = (), source: Path | str | None = None, sounding: Path | str | None = None,
                 acknowledge: Sequence[str] = (), drop_acknowledgments: bool = False, seed: int | None = None,
                 library: Path | str | None = None, json_output: bool = False) -> str:
    """The ``gpuwm-cm1 edit`` line for these changes (the GUI's "Show command")."""

    argv = ["gpuwm-cm1", "edit", str(rundir)]
    if source is not None:
        argv += ["--from", str(source)]
    for text in sets:
        argv += ["--set", str(text)]
    for text in settings:
        argv += ["--setting", str(text)]
    for key in resets:
        argv += ["--reset-setting", str(key)]
    if sounding is not None:
        argv += ["--sounding", str(sounding)]
    if "output-keys" in acknowledge:
        argv.append("--acknowledge-output-keys")
    if "thompson" in acknowledge:
        argv.append("--acknowledge-thompson")
    if drop_acknowledgments:
        argv.append("--drop-acknowledgments")
    if seed is not None:
        argv += ["--seed", str(seed)]
    if library is not None:
        argv += ["--library", str(library)]
    if json_output:
        argv.append("--json")
    return shlex.join(argv)


def run_edit(
    rundir: Path,
    *,
    sets: Mapping[str, Any] | None = None,
    source: Path | None = None,
    settings: Mapping[str, Any] | None = None,
    resets: Sequence[str] = (),
    sounding: Path | None = None,
    acknowledge: Sequence[str] = (),
    drop_acknowledgments: bool = False,
    seed: int | None = None,
    library: Path | None = None,
    command: str | None = None,
) -> tuple[int, dict[str, Any], str]:
    """Edit ``rundir`` and import it again; (exit code, report JSON with an ``edit`` block, report text)."""

    from ..cm1run.fit import FitError, set_namelist_values
    from . import usersettings

    rundir = Path(rundir).expanduser()
    require_editable(rundir)
    seed = check_seed(seed)
    target = rundir / NAMELIST
    before_bytes = target.read_bytes()
    before_settings = usersettings.read_file(rundir / SETTINGS) if (rundir / SETTINGS).is_file() else {}
    try:
        usersettings.merge(before_settings, dict(settings or {}), tuple(resets))
    except usersettings.SettingError as error:
        raise EditError(str(error)) from error
    if source is not None:
        source = Path(source).expanduser()
        if not source.is_file():
            raise EditError(f"{source} is not a file")
        text_bytes = source.read_bytes()
    else:
        text_bytes = before_bytes
    if sets:
        try:
            text = text_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise EditError(
                f"{target} holds a character outside ASCII (byte {error.start}); CM1's reader refuses it. "
                "Fix it in the namelist editor first"
            ) from error
        try:
            text = set_namelist_values(text, dict(sets))
        except FitError as error:
            raise EditError(str(error)) from error
        text_bytes = text.encode("ascii")
    if text_bytes != before_bytes:
        _atomic_write(target, text_bytes)

    previous = _previous_report(rundir)
    given = [] if drop_acknowledgments else list((previous.get("acknowledgments") or {}).get("given") or [])
    acknowledged = list(dict.fromkeys([*given, *acknowledge]))
    try:
        code, payload, report_text = run_import(
            target, rundir, sounding=sounding, acknowledge=acknowledged, seed=seed, library=library,
            command=command, settings=settings, reset_settings=resets,
        )
    except ImportDoorError:
        if text_bytes != before_bytes:
            _atomic_write(target, before_bytes)  # nothing was imported: leave the run as it was
        raise
    after_settings = usersettings.read_file(rundir / SETTINGS) if (rundir / SETTINGS).is_file() else {}
    diff = "".join(difflib.unified_diff(
        before_bytes.decode("utf-8", "replace").splitlines(keepends=True),
        text_bytes.decode("utf-8", "replace").splitlines(keepends=True),
        fromfile="namelist.input (before)", tofile="namelist.input",
    ))
    setting_rows = []
    for key in sorted(set(before_settings) | set(after_settings)):
        if before_settings.get(key) != after_settings.get(key):
            setting_rows.append({"key": key, "from": before_settings.get(key), "to": after_settings.get(key)})
    payload["edit"] = {
        "namelist_changed": text_bytes != before_bytes,
        "set": {key: value for key, value in (sets or {}).items()},
        "from": None if source is None else os.fspath(source),
        "diff": diff,
        "settings": setting_rows,
        "acknowledged": acknowledged,
    }
    return code, payload, report_text


def add_parser(commands: Any) -> None:
    command = commands.add_parser(
        "edit",
        help="change an imported run's namelist words or its own settings, then check it again",
        description=(
            "Change RUNDIR's namelist.input in place (--set, or --from a whole file) and the settings CM1 keeps "
            "in Fortran (--setting: the bubble, cold pool, line thermal and parcel seed box), then import the "
            "run again: the coverage report, deck.toml and run.json are rewritten and namelist.diff shows the "
            "difference from your original file. Exit 0: runs; 2: refused; 3: runs once acknowledged."
        ),
    )
    command.add_argument("rundir", type=Path, help="an imported run directory that has not run")
    command.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                         help="a CM1 r21.1 namelist word, e.g. umove=15.0 or tapfrq=300 (repeatable)")
    command.add_argument("--from", dest="source", type=Path, metavar="FILE",
                         help="replace the run's namelist with this file first")
    command.add_argument("--setting", dest="settings", action="append", default=[], metavar="TABLE.KEY=VALUE",
                         help="a value CM1 keeps in Fortran, e.g. initialization.zc_m=1500 (repeatable)")
    command.add_argument("--reset-setting", dest="resets", action="append", default=[], metavar="TABLE.KEY",
                         help="put a --setting value back to CM1's own (repeatable)")
    command.add_argument("--sounding", type=Path, metavar="FILE", help="a new input_sounding (isnd = 7 or 17)")
    command.add_argument("--acknowledge-output-keys", action="store_true",
                         help="run without the output settings GM1 does not read (the 'Not read' rows)")
    command.add_argument("--acknowledge-thompson", action="store_true",
                         help="run ptype = 3 (Thompson), which has been checked only in a 120-s column")
    command.add_argument("--drop-acknowledgments", action="store_true",
                         help="do not keep the acknowledgments the last import was given")
    command.add_argument("--seed", type=int, default=None, help="random seed for irandp = 1, 1 to 2**64-1")
    command.add_argument("--library", type=Path, default=None,
                         help="Rust initializer library for the sounding's environment block (default: installed)")
    command.add_argument("--json", action="store_true", help="print import-report.json (with an edit block)")
    command.set_defaults(handler=edit_door)


def edit_door(arguments: argparse.Namespace) -> int:
    from .namelist import NamelistToolError
    from . import usersettings

    acknowledge = []
    if arguments.acknowledge_output_keys:
        acknowledge.append("output-keys")
    if arguments.acknowledge_thompson:
        acknowledge.append("thompson")
    command = edit_command(
        arguments.rundir, sets=arguments.sets, settings=arguments.settings, resets=arguments.resets,
        source=arguments.source, sounding=arguments.sounding, acknowledge=acknowledge,
        drop_acknowledgments=arguments.drop_acknowledgments, seed=arguments.seed, library=arguments.library,
        json_output=arguments.json,
    )
    try:
        code, payload, text = run_edit(
            arguments.rundir,
            sets=namelist_changes(arguments.sets),
            source=arguments.source,
            settings=setting_changes(arguments.settings),
            resets=tuple(arguments.resets),
            sounding=arguments.sounding,
            acknowledge=acknowledge,
            drop_acknowledgments=arguments.drop_acknowledgments,
            seed=arguments.seed,
            library=arguments.library,
            command=command,
        )
    except (EditError, ImportDoorError, NamelistToolError, usersettings.SettingError) as error:
        if arguments.json:
            print(json.dumps({"status": "error", "code": "CM1-EDIT", "reason": str(error),
                              "remedy": "see the reason", "command": command}, indent=2), file=sys.stderr)
        else:
            print(f"gpuwm-cm1 edit: {error}", file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(payload, indent=1, default=str))
    else:
        edit = payload.get("edit") or {}
        lines = []
        if edit.get("diff"):
            lines.append(edit["diff"].rstrip("\n"))
        for row in edit.get("settings") or []:
            lines.append(f"settings.toml: {row['key']} {row['from']!r} -> {row['to']!r}")
        if lines:
            sys.stdout.write("\n".join(lines) + "\n\n")
        sys.stdout.write(text)
    return code


__all__ = [
    "EditError", "add_parser", "edit_command", "edit_door", "namelist_changes", "parse_fortran", "run_edit",
    "setting_changes",
]
