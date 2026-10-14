"""The coverage report: every setting of a CM1 namelist, in one of six categories.

``gpuwm-cm1 import`` writes it as ``import-report.json`` and ``.txt``; the
GUI's Check screen draws it.  Each setting the namelist states lands in
exactly one category:

``refused``
    GM1 will not run this value; the row says why and what to change.  Every
    refusal is collected in one pass: the importer's own (the microphysics
    scheme, keys outside CM1 r21.1, physics GM1 has no counterpart for), the
    value rules of ``records/CM1-VALUE-RULES.json``, then ``check`` and the
    host-side ``--plan`` run under :func:`cm1core.errors.collect_refusals`.
    When a check across settings stops a pass, the refused settings are set
    aside at CM1's own shipped value and the checks run again, so what they
    were hiding is found too; a refusal met only that way, or met while a
    refused value was still in place, is marked "(after fixing the above)".
``not_read``
    output and cosmetic keys GM1 does not read.  The run waits until the user
    acknowledges them ("Run without these N settings").
``differs``
    GM1 runs it, differently from CM1, and the row says how.  Only where the
    namelist's value matters.
``not_checked``
    GM1 runs it and nobody has compared it with CM1 at this value; each row
    names its evidence.
``not_needed``
    MPI layout, log format and CM1's own restart-file layout.
``same``
    everything else, counted by family with the evidence for each.

Nothing here opens a GPU.  The report is data (``CoverageReport.as_dict``,
schema ``gm1.cm1-import-report/v1``) and text (``CoverageReport.text``).
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from . import defaults as _defaults
from . import keymap as _keymap
from . import rules as _rules
from . import sounding as _sounding
from . import translate as _translate

SCHEMA = "gm1.cm1-import-report/v1"

#: The six categories, in report order: (id, label).
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("refused", "Refused"),
    ("not_read", "Not read"),
    ("differs", "Runs differently from CM1"),
    ("not_checked", "Not yet checked against CM1"),
    ("not_needed", "Not needed on one GPU"),
    ("same", "Same as CM1"),
)
CATEGORY_IDS = tuple(name for name, _ in CATEGORIES)

#: Acknowledgments a run can need before it starts.
ACK_OUTPUT_KEYS = "output-keys"
ACK_THOMPSON = "thompson"
ACKNOWLEDGMENTS = (ACK_OUTPUT_KEYS, ACK_THOMPSON)

#: Exit codes of ``gpuwm-cm1 import``.
EXIT_RUNS = 0
EXIT_REFUSED = 2
EXIT_ACKNOWLEDGE = 3

#: The component checked in place of a refused scheme, so one pass still
#: reaches every other setting.  The check's refusals of the stand-in's own
#: physics settings are not reported, and no microphysics row or comparison is
#: printed for it; the output rows that depend on the scheme (reflectivity,
#: the rain and graupel swaths) name it, and a note says it stood in.
STAND_IN_COMPONENT = "morrison-mp10"

#: The seed an imported ``irandp = 1`` deck starts from (``settings.toml``).
DEFAULT_RANDOM_SEED = 1

#: How many times the checks run with refused settings set aside.
MAX_PASSES = 8

#: Evidence for each "Same as CM1" family, by gpuwm-cm1 table.
FAMILIES: Mapping[str, tuple[str, str]] = {
    "grid": ("Grid", "records/CM1-SUPPORT-MATRIX.csv"),
    "run": ("Run length, time step, output and restart times", "records/CM1-CADENCE-MAP.md"),
    "dynamics": ("Advection, diffusion, sound waves and Coriolis", "records/CM1-DYNAMICS-BRANCH-MATRIX.json"),
    "boundaries": ("Lateral, top and bottom boundaries", "records/CM1-SUPPORT-MATRIX.csv"),
    "closures": ("Turbulence, damping, surface layer and moving frame", "records/CM1-SUPPORT-MATRIX.csv"),
    "surface": ("Surface", "records/CM1-SUPPORT-MATRIX.csv"),
    "physics": ("Microphysics", "records/GM1-STOCK-DECK-RATIOS.json"),
    "radiation": ("Radiation settings (radopt = 0: read by neither model)", "records/CM1-GM1-KEYMAP.json"),
    "initialization": ("Initial state", "records/CM1-INITIALIZATION-VALIDATION.json"),
    "output": ("Output words (&param9)", "records/CM1-SUPPORT-MATRIX.csv"),
    "diagnostics": ("Output words (&param9)", "records/CM1-SUPPORT-MATRIX.csv"),
    "lagrangian": ("Parcels and pressure decomposition (&param13)", "records/CM1-LAGRANGIAN-ANALYSIS.md"),
    "_inert": ("Settings neither model reads with these choices", "records/CM1-GM1-KEYMAP.json"),
    "_pinned": ("Settings GM1 always runs at the value the namelist has", "records/CM1-GM1-KEYMAP.json"),
    "_unread_group": ("Settings for other schemes and outputs, unread or off here", "records/CM1-GM1-KEYMAP.json"),
}
FAMILY_ORDER = (
    "grid", "run", "dynamics", "boundaries", "closures", "surface", "physics", "radiation",
    "initialization", "output", "lagrangian", "_pinned", "_inert", "_unread_group",
)

#: Settings a one-GPU run has no use for, and why.
NOT_NEEDED: Mapping[str, str] = {
    "ppnode": "MPI layout",
    "procfiles": "per-process output files (MPI)",
    "timeformat": "time format of CM1's log",
    "timestats": "CPU timing report",
    "run_time": "wall-clock limit for MPI jobs",
    "restart_format": "CM1's restart-file format; GM1 restarts have their own",
    "restart_filetype": "CM1's restart-file layout; GM1 restarts have their own",
    "restart_use_theta": "CM1's restart-file contents; a GM1 restart always carries the full state",
    "buffer_size": "CM1's log buffering",
    "compute_tasks_silent": "CM1's log (MPI)",
    "io_servers_silent": "CM1's log (MPI)",
    "stderr_logging": "CM1's log",
}
NOT_NEEDED_PREFIXES: Mapping[str, str] = {
    "restart_file_": "CM1's restart-file contents; a GM1 restart always carries the full state",
}

#: CM1 statistics words, covered by the statfrq row.
STATISTICS_GROUP = "param10"


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass
class Row:
    """One line of the report: one or more CM1 settings and what GM1 does with them."""

    category: str
    keys: tuple[str, ...]
    value: str
    text: str
    fix: str | None = None
    plain: str | None = None
    plain_fix: str | None = None
    code: str | None = None
    stage: str = "namelist"
    after: bool = False
    evidence: str | None = None
    acknowledge: str | None = None
    option: str | None = None
    detail: str | None = None
    expand: str | None = None
    defaulted: bool = False
    family: str | None = None
    #: How the keys are shown when there are many (``stat_* (29)``).
    display: str | None = None
    #: Settings this row accounts for that need no change of their own (CM1
    #: reads them only with the refused value this row names).
    covers: tuple[str, ...] = ()
    #: For a setting CM1 reads only under some selectors: those selectors.
    selectors: tuple[Mapping[str, Any], ...] = ()

    @property
    def key(self) -> str:
        if self.display:
            return self.display
        return ", ".join(self.keys)

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"key": self.key, "keys": list(self.keys), "value": self.value, "text": self.text}
        for name in ("fix", "plain", "plain_fix", "code", "evidence", "acknowledge", "option", "detail",
                     "expand", "family"):
            value = getattr(self, name)
            if value is not None:
                row[name] = value
        row["stage"] = self.stage
        if self.covers:
            row["covers"] = list(self.covers)
        if self.after:
            row["after"] = True
        if self.defaulted:
            row["defaulted"] = True
        return row


@dataclass
class CoverageReport:
    namelist: Mapping[str, Any]
    version: str
    component: Mapping[str, Any]
    rows: list[Row]
    same_families: list[Mapping[str, Any]]
    not_r21: list[Mapping[str, Any]]
    aliases: list[Mapping[str, Any]]
    output: Mapping[str, Any]
    sounding: Mapping[str, Any] | None
    comparison: Mapping[str, Any] | None
    acknowledged: tuple[str, ...]
    check: Mapping[str, Any]
    notes: list[str]
    settings: Mapping[str, Any]
    command: str | None = None
    toml: str | None = None
    plan: Mapping[str, Any] | None = None
    #: What the run will do that the user should know before it starts, though
    #: GM1 runs it as CM1 does (a forcing arm centred outside the domain).
    warnings: list[Mapping[str, Any]] = field(default_factory=list)

    # -- verdict ------------------------------------------------------------

    def category(self, name: str) -> list[Row]:
        return [row for row in self.rows if row.category == name]

    @property
    def needed_acknowledgments(self) -> tuple[str, ...]:
        needed = []
        for row in self.rows:
            if row.acknowledge and row.acknowledge not in needed:
                needed.append(row.acknowledge)
        return tuple(needed)

    @property
    def missing_acknowledgments(self) -> tuple[str, ...]:
        return tuple(name for name in self.needed_acknowledgments if name not in self.acknowledged)

    @property
    def exit_code(self) -> int:
        if self.category("refused"):
            return EXIT_REFUSED
        if self.missing_acknowledgments:
            return EXIT_ACKNOWLEDGE
        return EXIT_RUNS

    @property
    def verdict(self) -> str:
        return {EXIT_RUNS: "runs", EXIT_REFUSED: "refused", EXIT_ACKNOWLEDGE: "runs-after-acknowledgment"}[self.exit_code]

    @property
    def verdict_text(self) -> str:
        refused = self.category("refused")
        if refused:
            count = len({key for row in refused for key in row.keys})
            return f"Refused: {count} setting{'s' if count != 1 else ''} must change before this runs"
        missing = self.missing_acknowledgments
        if missing:
            parts = []
            if ACK_OUTPUT_KEYS in missing:
                count = sum(len(row.keys) for row in self.category("not_read"))
                parts.append(f"run without {count} output setting{'s' if count != 1 else ''} GM1 does not read")
            if ACK_THOMPSON in missing:
                parts.append("accept Thompson's limited checking")
            return "Runs after you acknowledge: " + "; ".join(parts)
        if self.category("differs") or self.category("not_checked"):
            return "Runs, with the differences below"
        return "Runs as CM1 does"

    # -- data ---------------------------------------------------------------

    def sections(self) -> list[dict[str, Any]]:
        sections = []
        for name, label in CATEGORIES:
            rows = self.category(name)
            section: dict[str, Any] = {"id": name, "label": label}
            if name == "same":
                section["collapsed"] = True
                section["count"] = sum(len(row.keys) for row in rows)
                section["families"] = [dict(item) for item in self.same_families]
                section["rows"] = [row.as_dict() for row in rows if row.text]
            else:
                section["count"] = sum(len(row.keys) for row in rows)
                section["rows"] = [row.as_dict() for row in rows]
                if name == "not_needed":
                    section["collapsed"] = True
                if name == "not_read" and rows:
                    count = section["count"]
                    section["ack"] = f"Run without these {count} setting{'s' if count != 1 else ''}"
            sections.append(section)
        return sections

    def as_dict(self) -> dict[str, Any]:
        refused = self.category("refused")
        return {
            "schema": SCHEMA,
            "title": "Check",
            "namelist": dict(self.namelist),
            "version": self.version,
            "not_r21": self.not_r21_summary,
            "settings_not_in_r21": list(self.not_r21),
            "aliases": list(self.aliases),
            "component": dict(self.component),
            "verdict": self.verdict,
            "verdict_text": self.verdict_text,
            "exit": self.exit_code,
            "first_refused": refused[0].as_dict() if refused else None,
            "acknowledgments": {
                "needed": list(self.needed_acknowledgments),
                "given": [name for name in self.acknowledged if name in self.needed_acknowledgments],
                "missing": list(self.missing_acknowledgments),
            },
            "sections": self.sections(),
            "guided": self.guided(),
            "output": dict(self.output),
            "sounding": None if self.sounding is None else dict(self.sounding),
            "comparison": None if self.comparison is None else dict(self.comparison),
            "check": dict(self.check),
            "settings": dict(self.settings),
            "notes": list(self.notes),
            # Only a report with a warning carries the key, so every other
            # report (and its golden file) is byte for byte what it was.
            **({"warnings": [dict(item) for item in self.warnings]} if self.warnings else {}),
            "command": self.command,
        }

    def guided(self) -> dict[str, Any]:
        """What guided mode needs: whether a beginner may be offered this run, and why not.

        Guided mode offers what GM1 runs: no refusal, nothing to
        acknowledge, and no microphysics that measurably runs differently
        from CM1's (the "Runs differently" row that carries the stock-deck
        ratios: WSM6, or Morrison should a ratio leave its release band; the 5 % target
        is printed, not a gate).
        The other "Runs differently" rows (output words, statistics cadence,
        the random stream) do not change the storm a beginner sees and do
        not block.  ``why_not`` holds the plain sentences, most important
        first; the gallery card shows the first.

        A setting that runs but has not yet been compared with CM1 (a "Not
        yet checked against CM1" row) does not block: the run is offered,
        and ``not_compared`` lists those rows' plain sentences so the card
        and the budget screen can say "Runs, not yet compared with CM1: ...".
        Such a row still blocks while it asks for an acknowledgment that has
        not been given (Thompson's limited checking).  A report with no such
        row has no ``not_compared`` key, so it is byte for byte what it was.
        """

        reasons: list[str] = []
        not_compared: list[str] = []
        missing = self.missing_acknowledgments
        for category in ("refused", "not_read", "not_checked", "differs"):
            for row in self.category(category):
                if category == "differs" and row.expand != "ratios":
                    continue
                text = row.plain or row.text
                if not text:
                    continue
                runs_unchecked = category == "not_checked" and row.acknowledge not in missing
                target = not_compared if runs_unchecked else reasons
                if text not in target:
                    target.append(text)
        guided: dict[str, Any] = {"offered": not reasons, "why_not": reasons[0] if reasons else None,
                                  "reasons": reasons}
        if not_compared:
            guided["not_compared"] = not_compared
        return guided

    @property
    def not_r21_summary(self) -> str:
        if not self.not_r21:
            return "none"
        names = ", ".join(row["key"] for row in self.not_r21[:6])
        more = f", and {len(self.not_r21) - 6} more" if len(self.not_r21) > 6 else ""
        return f"{len(self.not_r21)} ({names}{more})"

    # -- text ---------------------------------------------------------------

    def text(self) -> str:
        return render_text(self)


# ---------------------------------------------------------------------------
# Records the report reads
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def stock_deck_ratios() -> Mapping[str, Any]:
    text = resources.files("cm1core.records").joinpath("GM1-STOCK-DECK-RATIOS.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    if payload.get("schema") != "gm1.stock-deck-ratios/v1":
        raise ValueError("GM1-STOCK-DECK-RATIOS.json has an unknown schema")
    return payload


def _non_physical_groups() -> frozenset[str]:
    table = _keymap.load()
    return frozenset(row["cm1_group"] for row in table.non_physical_families) | frozenset(
        table.harness_owned_groups) | {"logging"}


def _option_keys() -> Mapping[str, tuple[str, ...]]:
    """gpuwm-cm1 option -> the CM1 keys it is written from."""

    mapping: dict[str, list[str]] = {}
    for entry in _keymap.load().entries:
        if entry.gm1_key:
            mapping.setdefault(entry.gm1_key, []).append(entry.cm1_key)
    result = {option: tuple(keys) for option, keys in mapping.items()}
    result["physics.component"] = ("ptype",)
    result["initialization.random_seed"] = ("irandp",)
    result["boundaries.bottom"] = ("bbc",)
    result["runtime.acoustic_substeps"] = ("dtl",)
    result["grid.nx*grid.dz_m"] = ("nz", "dz")
    result["grid.nz*grid.dz_m"] = ("nz", "dz")
    result["sounding"] = ("isnd",)
    # The storm-track check names the grid; the domain size is what the namelist sets.
    result["grid"] = ("nx", "ny")
    result["initialization"] = ("iinit",)
    return result


def _inverse(transform: str, value: Any) -> Any:
    """A gpuwm-cm1 value as the CM1 namelist would write it."""

    if transform in ("int-to-bool", "logical-to-bool"):
        if isinstance(value, bool):
            return int(value) if transform == "int-to-bool" else value
    return value


def fortran(value: Any) -> str:
    """A value as a namelist would write it."""

    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return f"{value:.1f}"
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(fortran(item) for item in value)
    return str(value)


# ---------------------------------------------------------------------------
# Building the report
# ---------------------------------------------------------------------------


class _Builder:
    def __init__(
        self,
        imported: Any,
        *,
        sounding_path: Path | None,
        acknowledged: Collection[str],
        seed: int | None,
        library: Path | None,
        environment: bool,
    ):
        self.imported = imported
        self.parsed = imported.parsed
        self.inventory = _defaults.load()
        self.table = _keymap.load()
        self.rules = _rules.load()
        self.option_keys = _option_keys()
        self.effective: dict[str, Any] = dict(imported.effective)
        self.stated = set(imported.stated)
        self.defaulted = {row.key for row in imported.defaulted}
        self.sounding_path = sounding_path
        self.acknowledged = tuple(name for name in ACKNOWLEDGMENTS if name in set(acknowledged))
        self.seed = seed
        self.library = library
        self.environment = environment
        self.rows: list[Row] = []
        self.notes: list[str] = []
        self.not_r21: list[dict[str, Any]] = []
        self.aliases: list[dict[str, Any]] = []
        self.set_aside: dict[str, Any] = {}
        self.stand_in = False
        self.translation: Any = None
        self.check_record: dict[str, Any] = {"passes": 0, "status": "not-run"}
        self.plan: dict[str, Any] | None = None
        self.config: Any = None
        self.sounding_check: _sounding.SoundingCheck | None = None
        self.order = self._order()
        self.settings: dict[str, Any] = {}
        self.runtime_overrides: dict[str, Any] = {}
        self.adjusted: list[dict[str, Any]] = []

    # -- helpers ------------------------------------------------------------

    def _order(self) -> dict[str, int]:
        order: dict[str, int] = {}
        for group_index, group in enumerate(self.parsed.group_order):
            for key_index, key in enumerate(self.parsed.groups.get(group, {})):
                order.setdefault(key, group_index * 1000 + key_index)
        return order

    def group_of(self, key: str) -> str | None:
        for group, rows in self.parsed.groups.items():
            if key in rows:
                return group
        found = self.inventory.groups_of(key)
        return found[0] if found else None

    def written(self, key: str) -> str:
        """The value as the namelist wrote it, else as CM1 would hold it."""

        for rows in self.parsed.groups.values():
            if key in rows:
                return str(rows[key].text).strip()
        value = self.effective.get(key)
        if key in self.stated:
            return fortran(value)  # read from an older key or an r22.0 one
        return f"{fortran(value)} (not set: CM1 r21.1's default)"

    def values(self, keys: Sequence[str]) -> str:
        texts = [self.written(key) for key in keys]
        if len(set(texts)) == 1:
            return texts[0]
        return ", ".join(texts)

    def cm1_keys(self, option: str | None) -> tuple[str, ...]:
        if not option:
            return ()
        if option in self.option_keys:
            keys = self.option_keys[option]
            if option == "physics.component" and int(self.effective.get("imoist", 1)) == 0:
                return ("imoist",)
            return keys
        if option.startswith("runtime."):
            # The importer's own run settings: what the namelist decides there is the grid.
            return ("nx", "ny", "nz")
        return ()

    def add(self, row: Row) -> Row:
        self.rows.append(row)
        return row

    def describe(self, row: Row, value: Any = None) -> Row:
        """Fill the row's plain words (and curated CM1 words) from ``gui.refusal_text``."""

        from ..gui import refusal_text

        found = refusal_text.describe(
            code=row.code, keys=row.keys, value=value, context=self.effective, category=row.category,
        )
        if found is None:
            return row
        if found.cm1:
            row.detail = row.detail or (row.text if row.text != found.cm1 else None)
            row.text = found.cm1
        if found.fix is not None:
            row.fix = found.fix
        row.plain = found.plain
        row.plain_fix = found.plain_fix
        return row

    def cm1_words(self, text: str) -> str:
        """Replace gpuwm-cm1 option names in a refusal's words by CM1's."""

        import re

        def swap(match: re.Match[str]) -> str:
            option = match.group(0)
            keys = self.option_keys.get(option)
            if keys and len(keys) == 1:
                return keys[0]
            return option

        return re.sub(r"\b(?:run|grid|dynamics|boundaries|closures|surface|physics|radiation|initialization|"
                      r"output|diagnostics|lagrangian|runtime)\.[a-z_0-9]+\b", swap, text)

    # -- the namelist itself -----------------------------------------------

    def namelist_rows(self) -> None:
        imported = self.imported
        for stop in imported.cm1_stops:
            groups = [part for part in ("param0", "param2", "param8") if f"&{part}" in stop]
            self.add(self.describe(Row(
                category="refused",
                keys=tuple(f"&{group}" for group in groups) or ("namelist",),
                value="missing" if "has no" in stop else "out of order",
                text=stop,
                fix="add the group, in the order &param0, &param2, &param8, as CM1's own namelist.input has them",
                code="CM1-IMPORT-STARTUP-GROUPS",
            )))
        if self.parsed.outside:
            lines = ", ".join(str(row.line) for row in self.parsed.outside[:8])
            self.notes.append(
                f"{len(self.parsed.outside)} line(s) outside the namelist groups (line {lines}); "
                "CM1 skips them, and so does GM1"
            )
        non_physical = _non_physical_groups()
        for reading in imported.aliases:
            entry = {
                "key": reading.key, "group": reading.group, "value": reading.text, "from": "an older CM1 release",
                "releases": list(reading.releases), "class": reading.klass, "reading": reading.message,
                "status": reading.status,
            }
            self.aliases.append({"key": reading.key, "status": reading.status, "text": reading.message,
                                 "assigns": dict(reading.assigns)})
            if reading.status in ("read", "same", "ignored"):
                entry["category"] = "same"
                self.not_r21.append(entry)
                continue
            if reading.status == "conflict" or reading.klass == "physics":
                category = "refused"
            elif reading.klass == "not-needed":
                category = "not_needed"
            else:
                category = "not_read"
            entry["category"] = category
            self.not_r21.append(entry)
            row = Row(
                category=category,
                keys=(reading.key,),
                value=reading.text,
                text=reading.message,
                code="CM1-IMPORT-OLD-KEY" if category == "refused" else None,
                fix=("remove it, or set the CM1 r21.1 setting it became" if category == "refused" else None),
                acknowledge=ACK_OUTPUT_KEYS if category == "not_read" else None,
            )
            self.add(self.describe(row, reading.value))
        for reading in imported.r22:
            entry = {
                "key": reading.key, "group": reading.group, "value": reading.text, "from": "CM1 r22.0",
                "class": reading.klass, "reading": reading.message,
            }
            if reading.status == "same":
                entry["category"] = "same"
                entry["text"] = "Same as CM1 r21.1"
                self.not_r21.append(entry)
                continue
            if reading.klass == "physics":
                category = "refused"
            else:
                category = "not_read"
            entry["category"] = category
            self.not_r21.append(entry)
            self.add(self.describe(Row(
                category=category,
                keys=(reading.key,),
                value=reading.text,
                text=reading.message,
                code="CM1-IMPORT-R22-PHYSICS" if category == "refused" else "CM1-IMPORT-R22-OUTPUT",
                fix=("set it to its CM1 r21.1 value, or remove it" if category == "refused" else None),
                acknowledge=ACK_OUTPUT_KEYS if category == "not_read" else None,
            ), reading.value))
        declared_groups = set(self.inventory.fields)
        grouped: dict[tuple[str, str], list[Any]] = {}
        for unknown in imported.unknown:
            group_is_physical = unknown.group in declared_groups and unknown.group not in non_physical
            category = "refused" if group_is_physical else "not_read"
            self.not_r21.append({"key": unknown.key, "group": unknown.group, "value": unknown.text,
                                 "from": "no CM1 release GM1 knows", "category": category,
                                 "class": "physics" if category == "refused" else "output"})
            grouped.setdefault((category, unknown.group), []).append(unknown)
        for (category, group), unknowns in grouped.items():
            keys = tuple(item.key for item in unknowns)
            names = ", ".join(keys)
            plural = len(keys) > 1
            if category == "refused":
                if plural:
                    text = (
                        f"{names} are not CM1 r21.1 settings, nor renamed older ones or CM1 r22.0 ones. "
                        f"CM1 r21.1 stops reading &{group} at them, and &{group} holds settings that change "
                        "the storm, so GM1 cannot tell what they would change (edited CM1 source reads them)"
                    )
                else:
                    text = (
                        f"{names} is not a CM1 r21.1 setting, nor a renamed older one or a CM1 r22.0 one. "
                        f"CM1 r21.1 stops reading &{group} at it, and &{group} holds settings that change "
                        "the storm, so GM1 cannot tell what it would change (edited CM1 source reads it)"
                    )
                fix = "remove them, or rename each to the CM1 r21.1 setting it stands for" if plural else \
                    "remove it, or rename it to the CM1 r21.1 setting it stands for"
            else:
                where = (f"&{group}, a group CM1 r21.1 does not read" if group not in declared_groups
                         else f"&{group}, which holds output settings only")
                text = f"Not {'CM1 r21.1 settings' if plural else 'a CM1 r21.1 setting'}; in {where}"
                fix = None
            self.add(self.describe(Row(
                category=category,
                keys=keys,
                value=", ".join(str(item.text) for item in unknowns) if len(keys) <= 4 else "",
                text=text,
                fix=fix,
                code="CM1-IMPORT-UNKNOWN-KEY" if category == "refused" else "CM1-IMPORT-UNKNOWN-OUTPUT-KEY",
                acknowledge=ACK_OUTPUT_KEYS if category == "not_read" else None,
                display=(f"&{group}: {keys[0]}, {keys[1]} and {len(keys) - 2} more" if len(keys) > 4 else None),
            ), unknowns[0].value))

    # -- microphysics ------------------------------------------------------

    def component(self) -> str:
        choice = self.imported.component
        if choice.status == "refused":
            self.stand_in = True
            self.add(self.describe(Row(
                category="refused",
                keys=("ptype",),
                value=self.written("ptype"),
                text=choice.reason,
                fix="use ptype = 5 (Morrison) or ptype = 7 (WSM6)",
                code="CM1-IMPORT-PTYPE",
                stage="namelist",
            ), choice.ptype))
            self.notes.append(
                f"ptype = {choice.ptype} is refused, so the other settings were checked with ptype = 5 "
                "(Morrison) standing in for it"
            )
            return STAND_IN_COMPONENT
        return str(choice.component)

    # -- translation -------------------------------------------------------

    def runtime_keys(self) -> dict[str, Any]:
        runtime = dict(_translate.PRODUCT_GM1_ONLY)
        # irandp = 1, iinit = 8's line noise and iinit = 7's warm-core noise
        # all draw from init3d.F:427-440's stream.
        random = int(self.effective.get("irandp", 0)) == 1 or int(self.effective.get("iinit", 1)) in (7, 8)
        if random and self.seed is None:
            # CM1's own random numbers: libgfortran's stream as init3d.F:427-440
            # seeds it, reproduced word for word (random_generator = "cm1").
            runtime["initialization.random_generator"] = "cm1"
            self.settings.setdefault("initialization", {})["random_generator"] = "cm1"
        elif random:
            # A seed the user chose: GM1's own stream, a different realization.
            seed = int(self.seed)
            runtime["initialization.random_seed"] = seed
            self.settings.setdefault("initialization", {})["random_seed"] = seed
        restart_at_end = self.restart_at_end()
        if restart_at_end is not None:
            runtime["run.restart_interval_s"] = restart_at_end
        runtime.update(self.runtime_overrides)
        return runtime

    def restart_at_end(self) -> float | None:
        """The run length, for a namelist whose negative rstfrq writes no restarts.

        CM1 writes no restart for rstfrq < 0 (README.namelist:103-104; cm1.F:3186
        tests rstfrq > 0).  A GM1 run keeps one restart at the end so it can be
        continued (``cm1run.launch`` schedules exactly that), so the
        deck carries the run length, not the translator's ``|rstfrq|``: a
        cadence the namelist never asked for is neither checked nor refused.
        """

        rstfrq = self.effective.get("rstfrq")
        timax = self.effective.get("timax")
        if isinstance(rstfrq, bool) or not isinstance(rstfrq, (int, float)) or not rstfrq < 0:
            return None
        if isinstance(timax, bool) or not isinstance(timax, (int, float)) or not timax > 0:
            return None
        return float(timax)

    def adjust(self) -> None:
        """Apply the changes CM1 makes to a namelist right after reading it.

        ``param.F`` clamps some selectors and rewrites others before anything
        runs (param.F:682-725, 836-853); the importer reads the namelist as
        CM1 runs it, so it applies the same changes and says so.  Two are left
        out because they only follow from a setting GM1 refuses anyway and would
        add a second row for it: ``tconfig = 2`` for ``cm1setup = 2``
        (param.F:720) and ``roflux = 1`` for ``psolver = 4, 5`` (param.F:721).
        """

        effective = self.effective

        def change(key: str, value: Any, where: str, why: str) -> None:
            if key not in effective or effective[key] == value:
                return
            self.adjusted.append({"key": key, "from": effective[key], "to": value, "where": where, "why": why})
            effective[key] = value

        def clamp(key: str, low: int, high: int, where: str) -> None:
            value = effective.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and not low <= value <= high:
                change(key, max(low, min(high, value)), where, f"CM1 clamps {key} to {low}..{high}")

        clamp("eqtset", 1, 2, "param.F:682-683")
        clamp("axisymm", 0, 1, "param.F:684-685")
        clamp("imoist", 0, 1, "param.F:686-687")
        if _as_int(effective.get("imoist")) != 1:
            change("efall", 0, "param.F:688", "CM1 turns the fallout energy term off without moisture")
        clamp("icor", 0, 1, "param.F:690-691")
        if _as_int(effective.get("icor")) == 0:
            change("betaplane", 0, "param.F:693", "CM1 turns the beta plane off without Coriolis")
        clamp("betaplane", 0, 1, "param.F:694-695")
        if _as_int(effective.get("axisymm")) == 1 or _as_int(effective.get("ny"), 99) <= 3:
            change("betaplane", 0, "param.F:699", "CM1 turns the beta plane off for 2-D and axisymmetric runs")
        if effective.get("do_adapt_move") is True and _as_int(effective.get("axisymm")) != 1:
            change("umove", 0.0, "param.F:703-716", "CM1 starts the adaptive frame from rest")
            change("vmove", 0.0, "param.F:703-716", "CM1 starts the adaptive frame from rest")
            change("imove", 1, "param.F:703-716", "CM1 turns the moving frame on for the adaptive frame")
        if _as_int(effective.get("imove")) == 0:
            change("umove", 0.0, "param.F:701", "CM1 ignores umove without imove = 1")
            change("vmove", 0.0, "param.F:702", "CM1 ignores vmove without imove = 1")
        clamp("irst", 0, 1, "param.F:717-718")
        cstar = effective.get("cstar")
        if isinstance(cstar, (int, float)) and cstar < 0.1:
            change("cstar", 30.0, "param.F:722", "CM1 replaces a phase speed below 0.1 m/s by 30 m/s")
        if _as_int(effective.get("irdamp")) == 0 and _as_int(effective.get("hrdamp")) == 0:
            change("rdalpha", 0.0, "param.F:724", "CM1 zeroes the damping rate with both dampers off")
        if _as_int(effective.get("irdamp")) >= 1 and _as_int(effective.get("hrdamp")) >= 1:
            change("hrdamp", effective.get("irdamp"), "param.F:725", "CM1 makes the lateral damper follow irdamp")
        clamp("output_pdcomp", 0, 1, "param.F:728")
        if _as_int(effective.get("isnd")) == 7 and _as_int(effective.get("iwnd")) > 0:
            change("iwnd", 0, "param.F:836-853", "CM1 uses the input_sounding winds with isnd = 7")
        testcase = _as_int(effective.get("testcase"))
        if testcase in (1, 2) and _as_int(effective.get("imoist"), 1) != 0:
            change("imoist", 0, "param.F:1044-1057", f"CM1 runs testcase = {testcase} dry")
        if testcase in (3, 4, 5, 7, 8) and _as_int(effective.get("imoist"), 1) != 1:
            change("imoist", 1, "param.F:1059-1072", f"CM1 runs testcase = {testcase} moist")
        # input.F declares these CM1 ``real``: CM1 runs the 32-bit value of what
        # the file wrote, and GM1 checks that it is handed exactly that value.
        for key in REAL32_EXACT:
            value = effective.get(key)
            if isinstance(value, float) and math.isfinite(value) and _binary32(value) != value:
                change(key, _binary32(value), "input.F", f"CM1 reads {key} as a 32-bit real")
        # A key the file does not set holds r21.1's code default, which for
        # these is 0; CM1 reads them only with their selector on, so with it
        # off the value is never used and GM1 takes CM1's shipped one.
        for key, (selector, on) in READ_ONLY_WITH.items():
            if key in self.stated or key not in effective or effective.get(selector) in on:
                continue
            group = self.group_of(key)
            row = self.inventory.field(group, key) if group else None
            if row is not None and row.has_distributed_value:
                change(key, row.distributed_value, "run/namelist.input",
                       f"not in the namelist, and CM1 reads it only with {selector} on")
        for item in self.adjusted:
            key = item["key"]
            if key in self.stated:
                before, after = fortran(item["from"]), fortran(item["to"])
                if before == after:
                    before, after = repr(item["from"]), repr(item["to"])
                self.notes.append(
                    f"{key} = {before} is read as {key} = {after}: "
                    f"{item['why']} ({item['where']}); GM1 runs CM1's value"
                )

    def neutral_value(self, key: str, option: str | None = None, *, allowed: Iterable[Any] | None = None) -> Any:
        """What a refused CM1 setting is set aside at while the checks run on."""

        current = self.effective.get(key)
        entry = self.table.by_cm1(key)
        candidates: list[Any] = []
        group = self.group_of(key)
        row = self.inventory.field(group, key) if group else None
        distributed = row.distributed_value if row is not None and row.has_distributed_value else None
        default = row.code_default if row is not None else None
        if allowed is not None and entry is not None:
            admitted = [_inverse(entry.transform, value) for value in allowed]
            for preferred in (distributed, default):
                if preferred is not None and any(preferred == value for value in admitted):
                    candidates.append(preferred)
            candidates.extend(admitted)
        candidates.extend(value for value in (distributed, default) if value is not None)
        for candidate in candidates:
            if candidate != current:
                return candidate
        return None

    def translate(self, component: str) -> Any:
        runtime = self.runtime_keys()
        for _ in range(MAX_PASSES):
            effective = dict(self.effective)
            param9 = self.inventory.fields["param9"]
            selection = {key: effective[key] for key in param9}
            try:
                return _translate.translate(
                    effective, component=component, gm1_only=runtime, output=selection,
                    declared=frozenset(self.stated),
                )
            except _translate.TranslationError as error:
                message = str(error)
                key = _translation_key(message, self.effective)
                row = Row(
                    category="refused",
                    keys=(key,) if key else ("namelist",),
                    value=self.written(key) if key else "",
                    text=self.cm1_words(message),
                    fix=(f"set {key} to a value CM1 r21.1 runs; the reason names what is wrong with this one"
                         if key else "correct the namelist as the reason says"),
                    code="CM1-IMPORT-TRANSLATION",
                    stage="translate",
                    defaulted=bool(key) and key not in self.stated,
                )
                if key and key in self.set_aside:
                    return None  # setting it aside did not help; the message says why
                self.add(self.describe(row, self.effective.get(key) if key else None))
                if not key:
                    return None
                if "does not build" in message and "stretch_z=1 column" in message:
                    # One row says what is wrong with the stretched levels.  The rest of the
                    # namelist is then read on even levels, so no later row is about a column
                    # the user did not ask for.
                    self.set_aside[key] = self.effective.get(key)
                    self.set_aside["stretch_z"] = 0
                    self.effective["stretch_z"] = 0
                    continue
                neutral = self.neutral_value(key)
                if neutral is None:
                    return None
                self.set_aside[key] = neutral
                self.effective[key] = neutral
        return None

    def translation_rows(self, translation: Any) -> None:
        for item in translation.unrepresentable:
            key = item["cm1_key"]
            if key == "eqtset":
                continue  # the WSM6 row says it (differs)
            gm1_runs = item.get("gm1_runs")
            if "cm1_reads_it_because" in item and gm1_runs is not None:
                because = _alternatives(item["cm1_reads_it_because"])
                text = (f"CM1 reads {key} here ({because}), and there GM1 always runs {key} = "
                        f"{fortran(gm1_runs)}: {item.get('note') or ''}").rstrip(": ")
                fix = f"set {key} = {fortran(gm1_runs)}"
            elif "cm1_reads_it_because" in item:
                because = _alternatives(item["cm1_reads_it_because"])
                names = sorted({name for alternative in item["cm1_reads_it_because"] for name in alternative})
                text = f"CM1 reads {key} here ({because}), and GM1 has no counterpart for it: {item.get('note') or ''}".rstrip(": ")
                fix = f"change {' or '.join(names)} so that CM1 does not read {key}"
            elif gm1_runs is not None:
                text = f"GM1 always runs {key} = {fortran(gm1_runs)}: {item.get('note') or ''}".rstrip(": ")
                fix = f"set {key} = {fortran(gm1_runs)}"
            else:
                text = f"{key} is not in CM1 r21.1's physical-key inventory, so what it selects is not run"
                fix = "remove it"
            self.add(self.describe(Row(
                category="refused",
                keys=(key,),
                value=self.written(key),
                text=text,
                fix=fix,
                code="CM1-IMPORT-UNREAD-PHYSICS",
                stage="translate",
                evidence="records/CM1-GM1-KEYMAP.json",
                detail=item.get("anchor"),
                defaulted=key not in self.stated,
                selectors=tuple(item.get("cm1_reads_it_because") or ()),
            ), item.get("effective")))
        for breach in _rules.judge(translation.gm1, self.rules):
            keys = self.cm1_keys(breach.option) or (breach.option,)
            entry = self.table.by_cm1(keys[0])
            allowed = [fortran(_inverse(entry.transform, value)) if entry else fortran(value) for value in breach.allowed]
            row = Row(
                category="refused",
                keys=keys,
                value=self.values(keys) if entry else fortran(breach.value),
                text=self.cm1_words(breach.rule.reason),
                fix=f"set {keys[0]} to {' or '.join(allowed)}" if len(allowed) <= 4 else f"set {keys[0]} to one of {', '.join(allowed)}",
                code=breach.code,
                stage="value-rule",
                option=breach.option,
                defaulted=all(key not in self.stated for key in keys),
            )
            self.add(self.describe(row, self.effective.get(keys[0])))
        self.drag_rows(translation)

    def drag_rows(self, translation: Any) -> None:
        """The surface layer's drag coefficient: an imported run runs CM1's constant one (cecd = 1) only.

        ``config.py`` admits CM1's three getcecd branches, but every imported run turns on the swath and
        vorticity diagnostics, whose ground-layer terms (``tornado_diagnostics.surface_layer_state``) are
        written for the constant drag coefficient alone, so a run with cecd = 2 or 3 would stop before its
        first step.  The import refuses it up front instead.
        """

        closures = translation.gm1.get("closures", {})
        cecd = closures.get("cecd")
        if closures.get("sfcmodel") != 1 or cecd not in (2, 3):
            return
        keys = ("cecd",)
        self.add(self.describe(Row(
            category="refused",
            keys=keys,
            value=self.values(keys),
            text=(f"cecd = {cecd} selects one of CM1's wind-dependent drag coefficients (sfcphys.F:138-194); "
                  "only cecd = 1 (constant drag coefficient) is implemented"),
            fix="set cecd = 1",
            code="CM1-IMPORT-CECD-UNSUPPORTED",
            stage="value-rule",
            option="closures.cecd",
            evidence="records/CM1-SUPPORT-MATRIX.csv",
            defaulted="cecd" not in self.stated,
        ), self.effective.get("cecd")))

    # -- check and plan ----------------------------------------------------

    def run_checks(self, component: str) -> None:
        translation = self.translate(component)
        self.translation = translation
        if translation is None:
            self.check_record = {"passes": 0, "status": "not-run: the namelist does not translate"}
            return
        self.translation_rows(translation)
        #: Rows met after an earlier refusal of the same pass, before any user
        #: setting was set aside: kept, and cleared, only if a later pass with
        #: the earlier refusals set aside meets them again; dropped if a later
        #: pass runs to the end without them.
        provisional: dict[tuple[str, str | None], Row] = {}
        status = "passed"
        passes = 0
        first: dict[str, Any] | None = None
        user_aside = False  # whether any namelist setting has been set aside yet
        for pass_index in range(MAX_PASSES):
            passes += 1
            if pass_index > 0:
                translation = self.translate(component)
                if translation is None:
                    status = "stopped: the namelist no longer translates with the settings set aside"
                    break
            document = tomllib.loads(_translate.render_toml(translation, ()))
            collected, config, plan, interrupted = _check_pass(document, self.sounding_path)
            if pass_index == 0:
                self.translation = translation
                if collected:
                    detail = collected[0].detail
                    first = {"code": detail.code, "option": detail.option,
                             "keys": list(self.cm1_keys(detail.option)), "value": _json_value(detail.value)}
            completed = interrupted is None and not any(item.ended_pass for item in collected)
            seen_here: set[tuple[str, str | None]] = set()
            new_aside = False
            retiled = False
            aside_this_pass = False
            for item in collected:
                detail = item.detail
                if self.stand_in and (detail.option or "").startswith("physics."):
                    continue
                if self._retile(detail) or self._deeper_columns(detail, component):
                    retiled = True
                    continue
                signature = (detail.code, detail.option)
                seen_here.add(signature)
                keys = self.cm1_keys(detail.option) or _keys_from_reason(detail.code, detail.reason)
                existing = self._find(detail.code, detail.option, keys)
                if existing is None:
                    row = self._refusal_row(item, keys, stage="check", after=False)
                    if not user_aside:
                        if item.after_earlier:
                            row.after = True
                            provisional[signature] = row
                        elif not item.independent and self._reads_set_aside(detail):
                            # A check across settings that read a value the
                            # translation set aside (a spacing, a stretch
                            # word), not the namelist's own.
                            row.after = True
                    else:
                        # Surfaced only once the refusals above were set aside.
                        row.after = not item.independent
                    self.add(row)
                elif signature in provisional and user_aside:
                    existing.after = False  # it holds with the earlier refusals set aside
                    provisional.pop(signature)
                if signature in provisional:
                    continue  # set aside only once a later pass confirms it
                if self._set_aside(keys, detail):
                    new_aside = aside_this_pass = True
            if not new_aside and not retiled:
                # Nothing left to set aside but refusals still waiting for
                # confirmation: set them aside (they stay "after").
                for signature, row in list(provisional.items()):
                    item_keys = tuple(key for key in row.keys if key in self.effective)
                    if self._set_aside(item_keys, None, option=row.option, code=row.code or ""):
                        new_aside = aside_this_pass = True
                    provisional.pop(signature)
            if completed and user_aside:
                for signature, row in list(provisional.items()):
                    if signature not in seen_here and row in self.rows:
                        self.rows.remove(row)  # a consequence of a value set aside since
                        provisional.pop(signature)
            if interrupted is not None:
                self.notes.append(
                    "a later check could not run on a value refused above "
                    f"({type(interrupted).__name__}: {interrupted}); it runs once that value is fixed"
                )
            user_aside = user_aside or aside_this_pass
            if not seen_here and not retiled and interrupted is None:
                status = "passed" if not self.set_aside else "passed with the refused settings set aside"
                self.config, self.plan = config, plan
                break
            if not new_aside and not retiled:
                status = (
                    "stopped: the checks after this point need the refused settings above fixed"
                    if not completed else "complete"
                )
                break
        else:
            status = f"stopped after {MAX_PASSES} passes"
        self.check_record = {
            "passes": passes,
            "status": status,
            "first_refusal": first,
            "set_aside": {key: fortran(value) for key, value in self.set_aside.items()},
        }
        if self.runtime_overrides:
            self.check_record["runtime"] = dict(self.runtime_overrides)

    def _reads_set_aside(self, detail: Any) -> bool:
        """Whether a refusal's words name a setting the importer has set aside."""

        import re

        if not self.set_aside:
            return False
        names: set[str] = set()
        for key in self.set_aside:
            names.add(key)
            entry = self.table.by_cm1(key)
            if entry is not None and entry.gm1_key:
                names.update((entry.gm1_key, entry.gm1_key.split(".", 1)[-1]))
        text = f"{detail.reason} {detail.remedy}"
        return any(re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", text) for name in names)

    def _deeper_columns(self, detail: Any, component: str) -> bool:
        """WSM6 deeper than its default 80 levels: take the 256-level column build GM1 ships."""

        if detail.code != "CM1-CARTESIAN-LEVEL-CAPACITY" or component != "wsm6-mp6":
            return False
        from ..engine_capacity import WSM6_256_CAPACITY_PROFILE

        if "runtime.capacity_profile" in self.runtime_overrides or _as_int(self.effective.get("nz")) > 256:
            return False
        self.runtime_overrides["runtime.capacity_profile"] = WSM6_256_CAPACITY_PROFILE
        self.settings.setdefault("runtime", {})["capacity_profile"] = WSM6_256_CAPACITY_PROFILE
        self.notes.append(
            f"WSM6 columns deeper than 80 levels run GM1's 256-level build ({WSM6_256_CAPACITY_PROFILE})"
        )
        return True

    def _retile(self, detail: Any) -> bool:
        """Take GM1's own remedy for a closure too large to run untiled.

        The memory budget and the closure tiles are GM1's run settings, not
        CM1's; the estimator sets the budget from the GPU.  Here the
        importer takes the measured tile the refusal names, or raises the
        budget to what the untiled step needs, and says so.
        """

        import re

        if detail.code not in ("CM1-CARTESIAN-UNTILED-BUDGET", "CM1-CARTESIAN-UNTILED-PHYSICS-BLOCK"):
            return False
        if "runtime.tile_nx" in self.runtime_overrides and "runtime.memory_budget_mib" in self.runtime_overrides:
            return False
        tile = re.search(r"runtime\.tile_nx = runtime\.tile_ny = (\d+)", detail.remedy)
        if detail.code == "CM1-CARTESIAN-UNTILED-PHYSICS-BLOCK":
            size = 128
        elif tile:
            size = int(tile.group(1))
        else:
            size = None
        if size is not None and "runtime.tile_nx" not in self.runtime_overrides:
            for name in ("tile_nx", "tile_ny", "physics_block_nx", "physics_block_ny"):
                self.runtime_overrides[f"runtime.{name}"] = size
            self.settings.setdefault("runtime", {}).update(
                {name: size for name in ("tile_nx", "tile_ny", "physics_block_nx", "physics_block_ny")})
            self.notes.append(
                f"the turbulence step is too large to run as one tile under GM1's default {self.runtime_keys()['runtime.memory_budget_mib']} MiB "
                f"budget, so it runs in {size} x {size} tiles (runtime.tile_nx, tile_ny); the estimate compares the "
                "whole run with your GPU"
            )
            return True
        need = re.search(r"memory_budget_mib to at least (\d+)", detail.remedy)
        if need and "runtime.memory_budget_mib" not in self.runtime_overrides:
            mib = int(need.group(1))
            self.runtime_overrides["runtime.memory_budget_mib"] = mib
            self.settings.setdefault("runtime", {})["memory_budget_mib"] = mib
            self.notes.append(
                f"the turbulence step needs at least {mib} MiB of GPU memory; the estimate compares it with your GPU"
            )
            return True
        return False

    def _set_aside(self, keys: Iterable[str], detail: Any, *, option: str | None = None, code: str = "") -> bool:
        """Set refused CM1 settings aside at a value the checks admit; True if any was."""

        if detail is not None:
            option, code = detail.option, detail.code
        changed = False
        for key in keys:
            if key in self.set_aside or key not in self.effective:
                continue
            allowed = self.rules.admitted(option) if option else None
            neutral = self._event_neutral(key, code)
            if neutral is None:
                neutral = self.neutral_value(key, option, allowed=allowed)
            if neutral is not None:
                self.set_aside[key] = neutral
                self.effective[key] = neutral
                changed = True
        return changed

    @staticmethod
    def _event_neutral_codes() -> frozenset[str]:
        return frozenset({
            "CM1-CFG-CLOCK-ALIGNMENT", "CM1-CFG-EVENT-OUTSIDE-RUN", "CM1-CFG-CARTESIAN-RESTART-CADENCE",
            "CM1-CFG-CARTESIAN-CADENCE", "CM1-CFG-RESTART-SIGN",
        })

    def _event_neutral(self, key: str, code: str) -> Any:
        """An output or restart interval set aside at the run length, as the corpus classifier does."""

        if code in self._event_neutral_codes() and key in ("tapfrq", "rstfrq", "statfrq", "prclfrq"):
            timax = self.effective.get("timax")
            if isinstance(timax, (int, float)) and timax > 0 and self.effective.get(key) != timax:
                return float(timax)
        return None

    def _find(self, code: str, option: str | None, keys: tuple[str, ...]) -> Row | None:
        for row in self.rows:
            if row.category != "refused":
                continue
            if option is not None and row.option == option:
                return row
            if row.code == code and row.keys == keys and keys:
                return row
            if keys and row.keys == keys and row.stage in ("value-rule", "translate", "namelist"):
                return row
        return None

    def _refusal_row(self, item: Any, keys: tuple[str, ...], *, stage: str, after: bool) -> Row:
        detail = item.detail
        shown = keys or ((detail.option,) if detail.option else (detail.code,))
        if keys:
            value = self.values(keys)
        elif isinstance(detail.value, (bool, int, float, str)):
            value = fortran(detail.value)
        else:
            value = ""
        row = Row(
            category="refused",
            keys=shown,
            value=value,
            text=self.cm1_words(detail.reason),
            fix=self.cm1_words(detail.remedy),
            code=detail.code,
            stage=stage,
            after=after,
            option=detail.option,
            defaulted=bool(keys) and all(key not in self.stated for key in keys),
        )
        return self.describe(row, self.effective.get(keys[0]) if keys else detail.value)

    # -- sounding ----------------------------------------------------------

    #: The words that choose the base-state sounding.  A run check may set a refused one aside
    #: (``set_aside``) so the rest of the namelist is still checked; the Sounding block describes the
    #: namelist's own choice regardless, so an analytic profile refused for a run is still the one shown.
    SOUNDING_WORDS = ("isnd", "iwnd")

    def sounding_rows(self) -> None:
        effective = dict(self.effective)
        for key in self.SOUNDING_WORDS:
            if key in self.set_aside and key in self.imported.effective:
                effective[key] = self.imported.effective[key]
        check = _sounding.check_sounding(self.sounding_path, effective)
        if self.environment:
            check = _sounding.environment(check, effective, library=self.library)
        self.sounding_check = check
        for problem in check.problems:
            self.add(self.describe(Row(
                category="refused",
                keys=("input_sounding",) if check.reads_file else ("isnd",),
                value=Path(check.path).name if check.path else f"isnd = {check.isnd}",
                text=problem.text,
                fix=problem.fix,
                code=problem.code,
                stage="sounding",
            )))

    def fold_dependents(self) -> None:
        """Fold a refused setting CM1 reads only because of another refused one into that row.

        ``dx_inner`` is read only with ``stretch_x = 1 or 2``: once stretch_x is
        fixed it is read by neither model and needs no change of its own, so it
        is listed with stretch_x (``covers``) rather than as a second refusal.
        """

        refused = [row for row in self.rows if row.category == "refused"]
        refused_keys = {key for row in refused for key in row.keys}
        for row in list(refused):
            if not row.selectors:
                continue
            blockers: list[str] = []
            for alternative in row.selectors:
                blocked = [name for name in alternative if name in refused_keys and name not in row.keys]
                if not blocked:
                    break
                blockers.append(blocked[0])
            else:
                target = next((other for other in refused if blockers[0] in other.keys and other is not row), None)
                if target is None:
                    continue
                target.covers = target.covers + tuple(key for key in row.keys if key not in target.covers)
                self.rows.remove(row)

    # -- the register of known differences -----------------------------------

    def register_rows(self, component: str) -> None:
        from .register import known_rows

        taken = {key for row in self.rows for key in row.keys + row.covers}
        for row in known_rows(self, component):
            if any(key in taken for key in row.keys) and row.category != "same":
                row = replace(row, keys=tuple(key for key in row.keys if key not in taken))
                if not row.keys:
                    continue
            self.add(self.describe(row))
            taken.update(row.keys)

    def same_rows(self) -> list[dict[str, Any]]:
        taken = {key for row in self.rows for key in row.keys + row.covers}
        non_physical = _non_physical_groups()
        translation = self.translation
        inert = set()
        if translation is not None:
            inert = {item["cm1_key"] for item in translation.inert_unmapped} | {
                item["cm1_key"] for item in translation.component_inert}
        families: dict[str, list[str]] = {}
        stated_r21 = [key for key in sorted(self.stated, key=lambda k: self.order.get(k, 10**9))
                      if self.inventory.groups_of(key)]
        for key in stated_r21:
            if key in taken:
                continue
            entry = self.table.by_cm1(key)
            group = self.group_of(key)
            if entry is not None and entry.gm1_key:
                family = entry.gm1_key.split(".", 1)[0]
                if family == "diagnostics":
                    family = "output"
                if family == "radiation" and int(self.effective.get("radopt", 0)) != 0:
                    family = "radiation"
            elif key in inert or group in non_physical:
                family = "_unread_group" if group in non_physical else "_inert"
            elif entry is not None and not entry.affects_state:
                family = "_inert"
            else:
                family = "_pinned"
            families.setdefault(family, []).append(key)
            self.add(Row(category="same", keys=(key,), value=self.written(key), text="", family=family,
                         stage="register"))
        result = []
        for family in FAMILY_ORDER:
            keys = families.get(family)
            if not keys:
                continue
            label, evidence = FAMILIES[family]
            if family == "radiation" and int(self.effective.get("radopt", 0)) != 0:
                label = "Radiation settings"
            result.append({"family": label, "id": family, "n": len(keys), "evidence": evidence, "keys": keys})
        return result


#: CM1 ``real`` settings GM1 requires as the exact 32-bit value CM1 holds.
REAL32_EXACT = ("umove", "vmove", "adapt_move_frq")

#: Settings CM1 reads only with a selector on: (selector, the values that read it).
READ_ONLY_WITH: Mapping[str, tuple[str, tuple[Any, ...]]] = {
    "adapt_move_frq": ("do_adapt_move", (True,)),
    "azimavgfrq": ("doazimavg", (True,)),
    "diagfrq": ("dodomaindiag", (True,)),
    "prclfrq": ("iprcl", (1,)),
}


def _binary32(value: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _translation_key(message: str, effective: Mapping[str, Any]) -> str | None:
    import re

    if "stretch_z=" in message and "reads its levels from a file" in message:
        return "stretch_z"
    # A stretched column that does not build is about the level words, not about zd,
    # the word whose translation happened to build it first (translate.model_top_m).
    if "does not build" in message and "stretch_z=1 column" in message:
        from ..cm1_vertical_grid import LEVEL_COUNT_HINT

        if LEVEL_COUNT_HINT in message:
            return "nz"
        for word in ("str_bot", "str_top", "dz_bot", "dz_top", "ztop"):
            if f"{word} =" in message or f"{word} -" in message:
                return word
        return "stretch_z"
    match = re.match(r"CM1 (\w+) = ", message)
    if match:
        return match.group(1)
    if "stretch_z" in message:
        return "stretch_z"
    if "acoustic step count" in message or "dtsmall" in message:
        # cm1_dtsmall's own reason follows the last ": " and names the input
        # it could not use (cm1_adaptive_timestep.py); dtl only when it is the
        # step itself, or the count it derives, that is wrong.
        inner = message.rsplit(": ", 1)[-1]
        for pattern, key in ((r"dx_m\b", "dx"), (r"dy_m\b", "dy"), (r"dz_m\b", "dz"), (r"nx\b", "nx"),
                             (r"ny\b", "ny"), (r"psolver=", "psolver"), (r"csound_m_s\b", "csound")):
            if re.match(pattern, inner):
                return key
        return "dtl"
    return None


def _json_value(value: Any) -> Any:
    from ..errors import _json_safe

    return _json_safe(value)


#: Refusals whose option names a whole table: the CM1 keys lead their reason.
_REASON_KEYS = {"CM1-CFG-DIAGNOSTIC-SELECTION"}


def _keys_from_reason(code: str, reason: str) -> tuple[str, ...]:
    """``diagnostics.output_uh/output_shs: ...`` -> (output_uh, output_shs), for refusals that name a table.

    ``tornado_diagnostics.validate_tornado_diagnostic_selection`` leads each
    reason with the table prefix the door passes (``diagnostics.``).
    """

    import re

    if code not in _REASON_KEYS:
        return ()
    match = re.match(r"^(?:diagnostics\.)?((?:output_\w+)(?:/output_\w+)*):", reason)
    if not match:
        return ()
    return tuple(match.group(1).split("/"))


def _alternatives(alternatives: Sequence[Mapping[str, Any]]) -> str:
    """``[{"cm1setup": [2, 4], "ipbl": [2]}]`` -> ``cm1setup = 2 or 4 and ipbl = 2``."""

    parts = []
    for alternative in alternatives:
        terms = []
        for name, values in alternative.items():
            spelled = " or ".join(fortran(value) for value in values)
            terms.append(f"{name} = {spelled}")
        parts.append(" and ".join(terms))
    return "; or ".join(parts)


def import_for_report(path: str | Path, *, tool: Any = None) -> Any:
    """Read ``path`` as CM1 r21.1 would; the report does the translation itself."""

    return _translate.import_namelist(path, tool=tool, translate_now=False)


def _check_pass(document: Mapping[str, Any], sounding: Path | None) -> tuple[list[Any], Any, Any, BaseException | None]:
    """One run of ``check`` and ``--plan`` under collect_refusals, plus the physics-family blockers."""

    from ..config import parse_mapping
    from ..errors import CM1Refusals, CollectedRefusal, RefusalDetail, collect_refusals

    config = None
    plan = None
    collected: list[Any] = []
    interrupted: BaseException | None = None
    try:
        with collect_refusals():
            config = parse_mapping(document)
            from ..cartesian_door import cartesian_plan_payload

            plan = cartesian_plan_payload(config, sounding=sounding)
    except CM1Refusals as aggregate:
        collected.extend(aggregate.refusals)
        interrupted = aggregate.interrupted
    if config is not None and not any(item.ended_pass for item in collected) and interrupted is None:
        from ..preflight import physics_component_blockers

        for blocker in physics_component_blockers(config):
            option = blocker.reason.split("=", 1)[0].strip() if "=" in blocker.reason else None
            collected.append(CollectedRefusal(
                detail=RefusalDetail(code=blocker.code, gate=blocker.gate, reason=blocker.reason,
                                     remedy=blocker.remedy, option=option),
                independent=True, after_earlier=False,
            ))
    return collected, config, plan, interrupted


def output_line(effective: Mapping[str, Any]) -> dict[str, Any]:
    output_format = int(effective.get("output_format", 1))
    filetype = int(effective.get("output_filetype", 1))
    formats = {1: "GrADS", 2: "netCDF"}
    layouts = {
        1: "every time in one file",
        2: "one file per time",
        3: "one file per time (one GPU writes what CM1's MPI ranks would split)",
    }
    name = formats.get(output_format, f"output_format = {output_format}")
    layout = layouts.get(filetype, f"output_filetype = {filetype}")
    text = f"{name} (output_format = {output_format}), {layout} (output_filetype = {filetype}), as the namelist asks."
    return {
        "format": name,
        "output_format": output_format,
        "output_filetype": filetype,
        "text": text,
        "also_netcdf": False if output_format == 1 else None,
        "writer": "gpuwm-cm1 cm1out",
    }


def comparison_block(component: str) -> dict[str, Any] | None:
    record = stock_deck_ratios()
    scheme = record["schemes"].get(component)
    if scheme is None:
        return None
    return {
        "title": "GM1 against CM1 r21.1 on CM1's stock supercell namelist",
        "scheme": component,
        "reference": scheme["reference"],
        "basis": "; ".join(part for part in (record["window"], scheme.get("measured"), record["measured"]) if part),
        "deck": record["deck"],
        "evidence": record["evidence"],
        "not_measured": record.get("not_measured"),
        "rows": scheme["rows"],
        "ratio_range": scheme["ratio_range"],
        "twin_range": scheme["twin_range"],
        "location": scheme["location"],
        "onset": scheme["onset"],
        "gates_release": scheme["gates_release"],
        "all_inside_band": scheme["all_inside_band"],
        "all_inside_target": scheme["all_inside_target"],
        "target": record["target"],
        "families": record["families"],
    }


def build_report(
    imported: Any,
    *,
    sounding: str | Path | None = None,
    acknowledge: Collection[str] = (),
    seed: int | None = None,
    library: str | Path | None = None,
    environment: bool = False,
    command: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> CoverageReport:
    """The six-category report for one imported namelist (``translate.NamelistImport``).

    ``overrides`` are GM1 settings the run takes beyond the namelist, as dotted
    gpuwm-cm1 keys (settings.toml's tables): the run's own settings that CM1
    keeps in Fortran (``usersettings``: the bubble, cold pool, line thermal and
    parcel seed box), the GPU memory budget, and the run settings a guided fit
    chose (the closure tile).  They go into the deck, are checked with it, and
    are written back to settings.toml.
    """

    builder = _Builder(
        imported,
        sounding_path=None if sounding is None else Path(sounding),
        acknowledged=acknowledge,
        seed=seed,
        library=None if library is None else Path(library),
        environment=environment,
    )
    for dotted, value in (overrides or {}).items():
        builder.runtime_overrides[dotted] = value
        table, name = dotted.split(".", 1)
        builder.settings.setdefault(table, {})[name] = value
    builder.adjust()
    builder.namelist_rows()
    component = builder.component()
    builder.run_checks(component)
    builder.sounding_rows()
    builder.fold_dependents()
    builder.register_rows(component)
    families = builder.same_rows()
    rows = _ordered(builder)
    choice = imported.component
    component_record = {
        "ptype": choice.ptype,
        "imoist": choice.imoist,
        "scheme": choice.scheme,
        "component": choice.component,
        "status": choice.status,
        "reason": choice.reason,
    }
    parsed = imported.parsed
    toml = None
    if builder.translation is not None and not any(row.category == "refused" for row in rows):
        # Translated again so the run settings the checks chose (tiles, the
        # 256-level WSM6 build) are in the deck; nothing was set aside.
        final = builder.translate(component)
        toml = _translate.render_toml(final, (
            f"# gpuwm-cm1 configuration imported from CM1 namelist {Path(parsed.path).name if parsed.path else 'namelist'}",
            f"# (sha256 {parsed.sha256}) by cm1core.cm1import through records/CM1-GM1-KEYMAP.json.",
            "# Do not hand-edit: edit the namelist and import it again.",
        ))
    report = CoverageReport(
        namelist={
            "path": parsed.path,
            "name": Path(parsed.path).name if parsed.path else None,
            "sha256": parsed.sha256,
            "groups": list(parsed.group_order),
            "settings_stated": len(imported.stated),
            "settings_defaulted": len(imported.defaulted),
        },
        version=imported.version,
        component=component_record,
        rows=rows,
        same_families=families,
        not_r21=builder.not_r21,
        aliases=builder.aliases,
        output=output_line(builder.imported.effective),
        sounding=None if builder.sounding_check is None else builder.sounding_check.as_dict(),
        comparison=comparison_block(str(choice.component)) if choice.component in ("morrison-mp10", "wsm6-mp6") else None,
        acknowledged=builder.acknowledged,
        check=builder.check_record,
        notes=builder.notes,
        settings=builder.settings,
        command=command,
        toml=toml,
        plan=builder.plan,
        warnings=_run_warnings(builder),
    )
    return report


def _run_warnings(builder: "_Builder") -> list[dict[str, Any]]:
    """Warnings about what the checked run will do, for a deck whose checks passed."""

    if builder.config is None:
        return []
    from ..cartesian_door import initiation_forcing_reach

    warnings: list[dict[str, Any]] = []
    reach = initiation_forcing_reach(builder.config)
    if reach is not None and reach.get("warning"):
        iinit = int(reach["iinit"])
        keys = {12: ["iinit", "xc_wnudge", "yc_wnudge"], 9: ["iinit", "xcent", "ycent"], 10: ["iinit", "xc_uforce"]}[iinit]
        warnings.append({
            "code": reach["code"],
            "keys": keys,
            "text": reach["warning"],
            "centre_inside_domain": reach["centre_inside_domain"],
            "forced_points": reach["forced_points"],
        })
    return warnings


def first_pass_document(imported: Any, *, seed: int | None = None) -> dict[str, Any] | None:
    """The gpuwm-cm1 document the report's first check pass runs, or None if it does not translate.

    The namelist as CM1 reads it (param.F's adjustments applied), translated
    with the importer's run settings and, for a refused scheme, the stand-in
    component.  Nothing is set aside yet.
    """

    builder = _Builder(imported, sounding_path=None, acknowledged=(), seed=seed, library=None, environment=False)
    builder.adjust()
    choice = imported.component
    component = STAND_IN_COMPONENT if choice.status == "refused" else str(choice.component)
    runtime = builder.runtime_keys()
    param9 = builder.inventory.fields["param9"]
    try:
        translation = _translate.translate(
            builder.effective, component=component, gm1_only=runtime,
            output={key: builder.effective[key] for key in param9}, declared=frozenset(builder.stated),
        )
    except _translate.TranslationError:
        return None
    return tomllib.loads(_translate.render_toml(translation, ()))


def _ordered(builder: _Builder) -> list[Row]:
    """Refused rows: namelist and translation first (in file order), then check order."""

    stage_rank = {"namelist": 0, "translate": 1, "value-rule": 1, "sounding": 3, "check": 2, "register": 4}
    order = builder.order

    def refused_key(item: tuple[int, Row]) -> tuple[int, int, int]:
        index, row = item
        rank = stage_rank.get(row.stage, 5)
        if rank <= 1:
            position = min((order.get(key, 10**6 + index) for key in row.keys), default=10**6 + index)
            return (rank, position, index)
        return (rank, 0, index)

    refused = [row for _, row in sorted(
        ((index, row) for index, row in enumerate(builder.rows) if row.category == "refused"), key=refused_key)]
    others = [row for row in builder.rows if row.category != "refused"]
    return refused + others


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int, indent: str) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width, initial_indent="", subsequent_indent=indent) or [""]


def render_text(report: CoverageReport) -> str:
    width = 100
    lines: list[str] = []
    name = report.namelist.get("path") or "namelist"
    lines.append(f"CM1 namelist: {name}")
    lines.append(f"Settings not in CM1 r21.1: {report.not_r21_summary}")
    if report.sounding and report.sounding.get("reads_file"):
        lines.append(f"Sounding: {report.sounding.get('path') or 'none given'} (isnd = {report.sounding['isnd']})")
    elif report.sounding:
        lines.append(f"Sounding: none needed (isnd = {report.sounding['isnd']})")
    lines.append(f"Result: {report.verdict_text}")
    for warning in report.warnings:
        lines.extend(_wrap(f"Warning: {warning['text']}", width, "  "))
    lines.append("")
    counts = {name: sum(len(row.keys) for row in report.category(name)) for name in CATEGORY_IDS}
    for name, label in CATEGORIES:
        rows = report.category(name)
        if name == "same":
            lines.append(f"{label} ({counts[name]})")
            for row in rows:
                if row.text:
                    lines.extend("  " + part for part in _wrap(f"{row.key} = {row.value}: {row.text}", width - 2, "    "))
            for family in report.same_families:
                lines.append(f"  {family['family']} ({family['n']})    {family['evidence']}")
            continue
        lines.append(f"{label} ({counts[name]})")
        for row in rows:
            head = f"{row.key} = {row.value}" if row.value != "" else row.key
            body = row.text
            if row.after:
                body = "(after fixing the above) " + body
            lines.extend("  " + part for part in _wrap(f"{head}: {body}", width - 2, "    "))
            if row.fix:
                lines.extend("    " + part for part in _wrap(f"Fix: {row.fix}", width - 4, "      "))
            if row.covers:
                lines.extend("    " + part for part in _wrap(
                    f"With it: {', '.join(row.covers)} (read only with this setting; no change of their own)",
                    width - 4, "      "))
            if row.evidence and name in ("not_checked", "differs"):
                lines.append(f"    Evidence: {row.evidence}")
        if name == "not_read" and rows:
            given = ACK_OUTPUT_KEYS in report.acknowledged
            lines.append(
                "  Acknowledged (--acknowledge-output-keys): the run goes ahead without them." if given else
                f"  The run waits until you acknowledge these {counts[name]} setting(s): "
                "import again with --acknowledge-output-keys, or tick 'Run without these settings'."
            )
    if report.not_r21:
        lines.append(f"Settings not in CM1 r21.1 ({len(report.not_r21)})")
        labels = {"same": "Same as CM1 r21.1", "refused": "Refused", "not_read": "Not read",
                  "not_needed": "Not needed on one GPU"}
        for item in report.not_r21:
            klass = f", {item['class']}" if item.get("class") else ""
            reading = labels.get(item.get("category", ""), item.get("category", ""))
            if item["from"] == "an older CM1 release" and item.get("reading"):
                if item.get("status") == "read":
                    reading = "read as its CM1 r21.1 setting"
                line = f"{item['reading']}: {reading}"  # "pertcor = 1 read as lspgrad = 1 (...)"
            else:
                line = f"{item['key']} = {item['value']} ({item['from']}{klass}): {reading}"
            lines.extend("  " + part for part in _wrap(line, width - 2, "    "))
    lines.append("")
    lines.append("Output: " + report.output["text"] + ("   [Also write netCDF]" if report.output.get("output_format") == 1 else ""))
    if report.sounding and report.sounding.get("warnings"):
        for warning in report.sounding["warnings"]:
            lines.extend(_wrap(f"Sounding check: {warning}", width, "  "))
    if report.sounding and report.sounding.get("environment"):
        from ..sounding_report import format_sounding_report

        lines.append("")
        lines.append("Sounding: the t=0 environment GM1 computes (CM1's base.F:3143-3189 block, plus shear and helicity)")
        lines.extend(format_sounding_report(dict(report.sounding["environment"])).rstrip("\n").splitlines())
    if report.comparison:
        comparison = report.comparison
        lines.append("")
        lines.append(comparison["title"] + f" ({comparison['reference']})")
        lines.extend(_wrap(comparison["basis"], width, ""))
        lines.append(f"  {'statistic':<42}{'CM1':>10}{'GM1':>10}{'GM1/CM1':>9}{'twin':>7}{'band':>7}")
        for row in comparison["rows"]:
            flag = "" if row["inside_band"] else "  outside the band"
            lines.append(
                f"  {row['label']:<42}{_short(row['cm1']):>10}{_short(row['gm1']):>10}"
                f"{row['ratio']:>9.3f}{row['twin']:>7.3f}{'±' + str(int(row['band'] * 100)) + '%':>7}{flag}"
            )
        location = comparison["location"]
        onset = comparison["onset"]
        lines.append(f"  {location['rule']}: worst {location['worst_cells']:g} cells")
        lines.append(f"  {onset['rule']}: {onset['gm1']:g} against {onset['cm1']:g} m/s (ratio {onset['ratio']:.3f})")
        lines.append(f"  Target: {comparison['target']}.")
        if comparison.get("not_measured"):
            lines.extend(_wrap(f"  Not measured: {comparison['not_measured']}.", width, "    "))
        lines.append(f"  Evidence: {comparison['evidence']}")
    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.extend(_wrap(f"Note: {note}", width, "  "))
    if report.command:
        lines.append("")
        lines.append(f"Command: {report.command}")
    return "\n".join(lines) + "\n"


def _short(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


__all__ = [
    "ACKNOWLEDGMENTS",
    "ACK_OUTPUT_KEYS",
    "ACK_THOMPSON",
    "CATEGORIES",
    "CoverageReport",
    "EXIT_ACKNOWLEDGE",
    "EXIT_REFUSED",
    "EXIT_RUNS",
    "Row",
    "SCHEMA",
    "build_report",
    "comparison_block",
    "first_pass_document",
    "fortran",
    "import_for_report",
    "output_line",
    "render_text",
    "stock_deck_ratios",
]
