"""Advanced mode's endpoints: every setting, Continue or extend, the run's commands, and a preset
opened as a namelist.

GET  /api/runs/RUN/restarts   the restart points Continue or extend can go on from, with each one's command
GET  /api/runs/RUN/commands   the run's commands.log, entry by entry, and as the shell script it is
GET  /api/runs/RUN/words      every CM1 r21.1 namelist word: the run's value, CM1's own value, and what the
                              coverage report says GM1 does with it
POST /api/presets/new         {"preset", "budget", "name", "gpu"?, "custom"?, "environment"? | "sounding"?,
                              "greyed"?}: ``gpuwm-cm1 preset new``, the preset fitted to this GPU and time (with
                              guided Customize's choices, when given) as a run folder to check and edit; a choice
                              Customize greys ({"scheme": "wsm6"}, {"friction": "on"}) is then set with one
                              ``gpuwm-cm1 edit --set``; nothing runs

The mode is a view: these read the same run directories guided
mode writes, and every change still goes through one ``gpuwm-cm1`` command
(``edit --set`` for a word changed in the every-setting table).  Standard
library only; the restarts are read with :mod:`cm1core.cm1run.rundir`, which
imports neither numpy nor CuPy.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import quote

from . import commands as C
from .files import NAME_RE, PathRefused, read_json, require_name
from .jobs import COMMANDS_LOG
from .runstate import segment_dirs

if TYPE_CHECKING:  # pragma: no cover
    from .api import Api, Reply

DEFAULTS_RECORD = Path(__file__).resolve().parents[1] / "records" / "CM1-R21-NAMELIST-DEFAULTS.json"
COMMANDS_LOG_LIMIT = 4 * 1024 * 1024
NAMELIST_LIMIT = 1024 * 1024
#: "Run it longer" steps the Continue or extend screen offers beside the namelist's own timax.
EXTEND_STEPS_S = (1800.0, 3600.0)
#: The coverage report's categories, in its order.
CATEGORIES = ("refused", "not_read", "differs", "not_checked", "not_needed", "same")


# ------------------------------------------------------------------ the namelist, read for display


_GROUP_START = re.compile(r"^\s*&([A-Za-z_]\w*)")
_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)(\s*\([^)]*\))?\s*=")


def _strip_comment(line: str) -> str:
    quote_char = None
    for index, char in enumerate(line):
        if quote_char:
            if char == quote_char:
                quote_char = None
        elif char in ("'", '"'):
            quote_char = char
        elif char == "!":
            return line[:index]
    return line


def _group_end(text: str) -> int | None:
    """The index of a namelist group's closing ``/`` in ``text`` (outside quotes), or None."""

    quote_char = None
    for index, char in enumerate(text):
        if quote_char:
            if char == quote_char:
                quote_char = None
        elif char in ("'", '"'):
            quote_char = char
        elif char == "/":
            return index
    return None


def read_words(text: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """The groups in file order, and each group's ``key -> value as written`` (for display only).

    What GM1 runs is read by the native parser when the run is imported; this
    only shows the file's own text beside each word, so a value is kept as
    written (``1000.0``, ``.false.``, ``1, 2, 3``).
    """

    order: list[str] = []
    groups: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if current is None:
            match = _GROUP_START.match(line)
            if not match:
                continue
            current = match.group(1).lower()
            if current not in groups:
                order.append(current)
                groups[current] = {}
            line = line[match.end():]
        end = _group_end(line)
        body = line if end is None else line[:end]
        matches = list(_ASSIGNMENT.finditer(body))
        for position, match in enumerate(matches):
            stop = matches[position + 1].start() if position + 1 < len(matches) else len(body)
            value = body[match.end():stop].strip().rstrip(",").strip()
            key = match.group(1).lower() + (re.sub(r"\s+", "", match.group(2)) if match.group(2) else "")
            groups[current][key] = value
        if end is not None:
            current = None
    return order, groups


def _load_defaults() -> dict[str, Any]:
    try:
        record = json.loads(DEFAULTS_RECORD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    groups = record.get("groups") if isinstance(record, dict) else None
    return groups if isinstance(groups, dict) else {}


def _group_rank(name: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"param(\d+)", name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def _fortran_default(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, float):
        return repr(value)
    return str(value)


def report_by_key(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each namelist key's category in the coverage report, with the row's words (families for "Same")."""

    found: dict[str, dict[str, Any]] = {}
    for section in report.get("sections") or []:
        if not isinstance(section, dict) or section.get("id") not in CATEGORIES:
            continue
        for row in section.get("rows") or []:
            if not isinstance(row, dict):
                continue
            for key in row.get("keys") or [row.get("key")]:
                if isinstance(key, str):
                    found.setdefault(key.lower(), {
                        "category": section["id"], "category_label": section.get("label"),
                        "text": row.get("text"), "plain": row.get("plain"), "fix": row.get("fix"),
                        "after": bool(row.get("after")),
                    })
        for family in section.get("families") or []:
            if not isinstance(family, dict):
                continue
            for key in family.get("keys") or []:
                if isinstance(key, str):
                    found.setdefault(key.lower(), {
                        "category": section["id"], "category_label": section.get("label"),
                        "text": family.get("family"), "plain": None, "fix": None, "after": False,
                        "evidence": family.get("evidence"),
                    })
    return found


def words_view(namelist_text: str | None, report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every CM1 r21.1 namelist word, group by group: the run's value, CM1's value, and the report's row."""

    defaults = _load_defaults()
    order, stated = read_words(namelist_text or "")
    by_key = report_by_key(report or {})
    aliases = {}
    for row in (report or {}).get("aliases") or []:
        if isinstance(row, dict) and isinstance(row.get("old"), str):
            aliases[row["old"].lower()] = row
    groups_out = []
    seen_groups = set()
    for name in [*order, *sorted((g for g in defaults if g not in order), key=_group_rank)]:
        if name in seen_groups:
            continue
        seen_groups.add(name)
        fields = defaults.get(name) or {}
        written = stated.get(name) or {}
        words = []
        for key in [*[k for k in written], *[k for k in fields if k not in written]]:
            spec = fields.get(key) if isinstance(fields.get(key), dict) else None
            row = by_key.get(key.split("(")[0])
            in_r21 = spec is not None
            word = {
                "key": key,
                "value": written.get(key),
                "in_file": key in written,
                "default": _fortran_default(spec.get("code_default")) if spec else None,
                # How r21.1 comes by that value: a line of its code (cm1-assignment, declaration-initializer), or
                # none at all (static-zero: gfortran and ifort read 0; the Fortran standard leaves it undefined).
                "basis": spec.get("basis") if spec else None,
                "type": spec.get("type") if spec else None,
                "source": spec.get("source") if spec else None,
                "in_r21": in_r21,
                "category": (row or {}).get("category") or (None if key in written else "default"),
                "category_label": (row or {}).get("category_label"),
                "text": (row or {}).get("text"),
                "plain": (row or {}).get("plain"),
                "fix": (row or {}).get("fix"),
            }
            if key in aliases:
                word["alias"] = aliases[key]
            words.append(word)
        groups_out.append({"group": name, "in_file": name in written or name in order, "words": words,
                           "stated": sum(1 for word in words if word["in_file"]), "total": len(words)})
    return {
        "groups": groups_out,
        "stated": sum(group["stated"] for group in groups_out),
        "total": sum(group["total"] for group in groups_out),
        "record": "records/CM1-R21-NAMELIST-DEFAULTS.json",
        "note": ("Values are the namelist's own text. A word not in the file takes CM1 r21.1's own value, as CM1 "
                 "does; the category is the coverage report's. Where r21.1's code sets no value, the word reads "
                 "as 0 with gfortran and ifort, which fill static storage with zeros."),
    }


# ------------------------------------------------------------------ commands.log


_ENTRY = re.compile(r"^#\s+(\S+)\s+([A-Za-z-]+)(?::\s*(.*))?$")


def read_commands_log(text: str) -> list[dict[str, Any]]:
    """``commands.log`` entry by entry: each ``# TIME ACTION[: NOTE]`` line and the command lines under it."""

    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _ENTRY.match(line)
        if match:
            entries.append({"time_utc": match.group(1), "action": match.group(2), "note": match.group(3) or "",
                            "lines": []})
        elif line.strip() and entries:
            entries[-1]["lines"].append(line)
        elif line.strip():
            entries.append({"time_utc": None, "action": None, "note": "", "lines": [line]})
    return entries


# ------------------------------------------------------------------ the class


class AdvancedApi:
    """Advanced mode's endpoints, beside :class:`.api.Api` (which routes to them)."""

    def __init__(self, api: "Api") -> None:
        self.api = api

    def get(self, parts: list[str], query: dict[str, list[str]]) -> "Reply | None":
        from .api import Reply

        if len(parts) == 3 and parts[0] == "runs" and parts[2] in ("restarts", "commands", "words"):
            run_id = parts[1]
            rundir = self.api.root.existing_run(run_id)
            handler = {"restarts": self.restarts, "commands": self.commands, "words": self.words}[parts[2]]
            return Reply(body=handler(run_id, rundir))
        return None

    # -------------------------------------------------------------- Continue or extend

    def restarts(self, run_id: str, rundir: Path) -> dict[str, Any]:
        """The restarts a resume can start from: the run's own line first, then any branch it has left."""

        api = self.api
        rows: list[dict[str, Any]] = []
        try:
            from ..cm1run.rundir import RunDirectory

            run = RunDirectory(rundir)
            lineage = run.lineage_restarts()
            everything = run.restarts()
        except Exception:  # noqa: BLE001 - a folder cm1run cannot read has no restarts to offer
            lineage, everything = [], []
        on_line = {(row.segment, row.number) for row in lineage}
        numbers: dict[int, int] = {}
        for row in everything:
            numbers[row.number] = numbers.get(row.number, 0) + 1
        newest = lineage[-1] if lineage else None
        for row in everything:
            shared = numbers.get(row.number, 0) > 1
            segment = row.segment if shared else None
            rows.append({
                "number": row.number, "segment": row.segment, "model_time_s": row.model_time_s, "step": row.step,
                "on_line": (row.segment, row.number) in on_line,
                "newest": newest is not None and (row.segment, row.number) == (newest.segment, newest.number),
                "needs_segment": shared,
                "command": C.display(C.resume_args(rundir, row.number, native=api.native, from_segment=segment)),
            })
        rows.sort(key=lambda row: (not row["on_line"], -(row["model_time_s"] or 0.0), row["segment"]))
        status = api.status(run_id, rundir)
        report = read_json(rundir / "import-report.json", default=None)
        values = ((report or {}).get("setup") or {}).get("values") or {} if isinstance(report, dict) else {}
        timax = values.get("timax") if isinstance(values.get("timax"), (int, float)) else status.get("duration_s")
        # How far the run's own line has got: "30 minutes longer" is counted from there, not from a branch it
        # has left (after going on from an early restart, the old line's later restarts are a branch).
        reached = max([row["model_time_s"] or 0.0 for row in rows if row["on_line"]] +
                      [status.get("model_time_s") or 0.0])
        options: list[dict[str, Any]] = []
        if isinstance(timax, (int, float)) and timax > (newest.model_time_s or 0.0 if newest else 0.0):
            options.append({"until_s": float(timax), "label": "to the namelist's timax", "default": True})
        for step in EXTEND_STEPS_S:
            until = float(reached) + step
            # A run paused an hour before timax: "60 minutes longer" is timax, offered once.
            if any(abs(option["until_s"] - until) < 0.5 for option in options):
                continue
            options.append({"until_s": until, "label": f"{step / 60:.0f} minutes longer", "default": not options})
        actions = status.get("actions") or []
        why_not = None
        if "resume" not in actions:
            state = status.get("state")
            if state in ("running", "waiting", "stale"):
                why_not = "The run is still going. Continue or extend is offered once it has ended."
            elif not rows:
                why_not = ("No restart has been saved yet. GM1 saves one at every rstfrq, and one at the end when "
                           "rstfrq is not positive, so a run stopped before then starts again from the beginning.")
            else:
                why_not = "This run cannot be continued now."
        return {
            "id": run_id,
            "state": status.get("state"),
            "label": status.get("label"),
            "actions": actions,
            "can_resume": "resume" in actions,
            "why_not": why_not,
            "restarts": rows,
            "timax_s": timax,
            "model_time_s": status.get("model_time_s"),
            "reached_s": reached,
            "until_options": options,
            "segments": [path.name for path in segment_dirs(rundir)],
            "commands": {
                "resume": C.display(C.resume_args(rundir, native=api.native)),
                "jobscript": C.display(C.jobscript_args(rundir, "slurm", native=api.native)),
                "status": C.display(C.status_args(rundir)),
            },
            "note": ("A resume writes a new segment and never changes the ones before it. Swaths and the ground "
                     "canvases carry across it, so a paused and resumed run matches an uninterrupted one."),
        }

    # -------------------------------------------------------------- commands.log

    def commands(self, run_id: str, rundir: Path) -> dict[str, Any]:
        path = rundir / COMMANDS_LOG
        text = ""
        truncated = False
        if path.is_file():
            with path.open("rb") as stream:
                data = stream.read(COMMANDS_LOG_LIMIT + 1)
            truncated = len(data) > COMMANDS_LOG_LIMIT
            text = data[:COMMANDS_LOG_LIMIT].decode("utf-8", "replace")
        entries = read_commands_log(text)
        return {
            "id": run_id,
            "path": str(path),
            "exists": path.is_file(),
            "truncated": truncated,
            "entries": entries,
            "count": len(entries),
            "text": text,
            "url": f"/api/runs/{quote(run_id, safe='')}/files/{COMMANDS_LOG}" if path.is_file() else None,
            "replay": f"bash {path}",
            "note": ("Every button in the GUI runs one gpuwm-cm1 command, and the server writes each one here. "
                     "Run this file with bash to do the same from a terminal; sidecar lines run in the background "
                     "at nice 19 with no GPU."),
        }

    # -------------------------------------------------------------- every setting

    def words(self, run_id: str, rundir: Path) -> dict[str, Any]:
        from .setup import _read_text

        text = _read_text(rundir / "namelist.input", NAMELIST_LIMIT)
        report = read_json(rundir / "import-report.json", default=None)
        view = words_view(text, report if isinstance(report, dict) else None)
        has_run = bool(segment_dirs(rundir))
        view.update({
            "id": run_id,
            "present": text is not None,
            "editable": text is not None and not has_run and (rundir / "namelist.input.original").is_file(),
            "why_fixed": "This run has already run, so its namelist is fixed." if has_run else None,
            "command_template": f"gpuwm-cm1 edit {rundir} --set KEY=VALUE --json",
        })
        return view

    # -------------------------------------------------------------- Start from a preset

    def preset_new(self, payload: Mapping[str, Any]) -> "Reply":
        from .api import ApiError, Reply, run_url
        from .guided import BUDGETS, Guided, _reason, unique_name

        preset = payload.get("preset")
        if not isinstance(preset, str) or not NAME_RE.match(preset):
            raise ApiError(400, "No preset was picked.", "Pick one of the presets.")
        budget = payload.get("budget")
        if budget not in BUDGETS:
            raise ApiError(400, f"There is no {budget!r} time.", "Use 10m, 1h, overnight or any.")
        gpu = payload.get("gpu")
        if gpu is not None and (isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0):
            raise ApiError(400, "gpu must be a whole number.", "Use 0 for the first GPU.")
        # Guided Customize's "All settings (Advanced)" sends the panel's choices and the made air with them.
        custom = Guided._custom(payload)
        given = Guided._given(payload) + custom
        sets = self._greyed_sets(payload.get("greyed"))
        root = self.api.root
        changed = self.api.guided.customized_list(payload) if custom and not payload.get("name") else []
        name = payload.get("name") or unique_name(preset, root.path, custom=bool(changed), customized=changed)
        try:
            name = require_name(str(name), "run name")
        except PathRefused as error:
            raise ApiError(400, str(error), "Use letters, digits, '.', '_' or '-'.") from error
        rundir = root.run_path(name)
        args = C.preset_new_args(preset, str(budget), root.path, name, gpu, given)
        if rundir.exists():
            raise ApiError(409, f"A run called {name!r} already exists.", "Pick another name.", command=C.display(args))
        self.api.jobs.stop_if_dry(args)
        result = self.api.jobs.host(args)
        created = rundir.is_dir() and (rundir / "run.json").is_file()
        if created:
            self.api.jobs.log_command(rundir, "preset-new", result.command,
                                      "the preset fitted to this GPU and time, as a run folder to edit")
        record = result.json if isinstance(result.json, dict) else {}
        if result.rc == 1:
            raise ApiError(409, str(record.get("message") or "This preset does not fit this computer for that time."),
                           "Pick a longer time, or import the preset's namelist as written from New run.",
                           command=result.command, fit=record.get("fit"))
        if result.rc != 0 or not created:
            raise ApiError(400, f"The run folder could not be made: {_reason(result)}",
                           "Read the output below; it names the problem.", command=result.command,
                           output=result.as_dict())
        message = "Made the run folder from the preset, fitted to this GPU and time. Nothing runs yet."
        if sets:
            # A choice guided mode greys (WSM6, Thompson, ground friction), carried here as one edit of the new
            # folder, so the user does not type the CM1 words; Check then shows what it means.
            edit = C.edit_args(rundir, sets=sets)
            edited = self.api.jobs.host(edit)
            self.api.jobs.log_command(rundir, "edit", edited.command,
                                      "the choice guided mode greys, set in Advanced")
            if edited.rc in (0, 3):
                message = ("Made the run folder from the preset, fitted to this GPU and time, with "
                           + ", ".join(words for words in self._greyed_words(payload.get("greyed")))
                           + ". Check shows what that choice means. Nothing runs yet.")
            else:
                message = (f"Made the run folder from the preset, but the change could not be made: "
                           f"{_reason(edited)} Set it in the namelist below.")
        return Reply(200, {
            "ok": True, "action": "preset-new", "run": name, "url": run_url(name), "command": result.command,
            "rc": result.rc, "fit": record.get("fit"), "result": record,
            "message": message,
            "fix": "", "output": result.as_dict(),
        })

    #: Guided's greyed choices that "All settings (Advanced)" carries into the new folder, as CM1 words.
    GREYED_SETS = {("scheme", "wsm6"): {"ptype": 7}, ("scheme", "thompson"): {"ptype": 3},
                   ("friction", "on"): {"bbc": 3, "sfcmodel": 1, "cecd": 1}}
    GREYED_WORDS = {("scheme", "wsm6"): "rain and hail by WSM6", ("scheme", "thompson"): "rain and hail by Thompson",
                    ("friction", "on"): "ground friction on"}

    @classmethod
    def _greyed_pairs(cls, greyed: Any) -> list[tuple[str, str]]:
        from .api import ApiError

        if greyed in (None, {}):
            return []
        if not isinstance(greyed, Mapping) or any((knob, value) not in cls.GREYED_SETS
                                                   for knob, value in greyed.items()):
            raise ApiError(400, "greyed must name a choice guided mode greys.",
                           'Send {"scheme": "wsm6"}, {"scheme": "thompson"} or {"friction": "on"}.')
        return list(greyed.items())

    @classmethod
    def _greyed_sets(cls, greyed: Any) -> dict[str, Any]:
        sets: dict[str, Any] = {}
        for pair in cls._greyed_pairs(greyed):
            sets.update(cls.GREYED_SETS[pair])
        return sets

    @classmethod
    def _greyed_words(cls, greyed: Any) -> list[str]:
        return [cls.GREYED_WORDS[pair] for pair in cls._greyed_pairs(greyed)]


__all__ = ["AdvancedApi", "read_commands_log", "read_words", "report_by_key", "words_view"]
