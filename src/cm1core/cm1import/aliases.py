"""Keys from CM1 releases before r21.1, read as their r21.1 settings.

``records/CM1-KEY-ALIASES.json`` lists every key CM1 r16-r20.3 declared that
r21.1 does not declare (or declares in another group), found from each
release's source, with the value mapping a CM1 document states:

* ``renamed`` -- pertcor became lspgrad in r19; hadvorder became hadvordrs and
  hadvordrv in r17.  The value carries over.
* ``moved`` -- ipbl moved from param12 to param2 in r19.
* ``value-mapped`` -- iturb = 1 is cm1setup = 1 with sgsmodel = 1; dns = 1 is
  cm1setup = 3.  Values the documents do not map are ``unmapped``.
* ``hard-wired`` / ``derived`` -- r21.1 always runs one value, or computes the
  setting itself (nsound).
* ``removed`` -- no r21.1 setting.

:func:`resolve` turns a file's undeclared assignments into readings.  A
reading never overrides a key the file also sets under its r21.1 name; if the
two disagree it is a ``conflict``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Mapping

from .namelist import UnknownAssignment

RECORD_NAME = "CM1-KEY-ALIASES.json"
SCHEMA = "gm1.cm1-namelist-key-aliases/v1"


@dataclass(frozen=True)
class AliasRow:
    key: str
    kind: str
    klass: str
    old_group: str
    releases: tuple[str, ...]
    note: str
    r21_targets: tuple[str, ...]
    values: Mapping[str, Mapping[str, Any]]
    unmapped_values: Mapping[str, str]
    r21_value: Any = None
    successor: tuple[str, ...] = ()


@dataclass(frozen=True)
class AliasReading:
    """One old-release assignment and what it means in r21.1.

    ``status`` is ``read`` (``assigns`` holds the r21.1 settings), ``same``
    (r21.1 always does this), ``ignored`` (r21.1 computes it), ``unmapped``
    (this value has no r21.1 setting), ``removed`` or ``conflict``.
    """

    group: str
    key: str
    text: str
    value: Any
    kind: str
    klass: str
    status: str
    message: str
    releases: tuple[str, ...]
    assigns: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Aliases:
    rows: Mapping[tuple[str, str], AliasRow]

    def row(self, group: str, key: str) -> AliasRow | None:
        return self.rows.get((group, key))


@lru_cache(maxsize=1)
def load() -> Aliases:
    payload = json.loads(resources.files("cm1core.records").joinpath(RECORD_NAME).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{RECORD_NAME} has schema {payload.get('schema')!r}, not {SCHEMA!r}")
    rows: dict[tuple[str, str], AliasRow] = {}
    for key, row in payload["aliases"].items():
        targets = tuple(target.split(".", 1)[1] for target in row.get("r21_targets", ()))
        rows[(row["old_group"], key)] = AliasRow(
            key=key,
            kind=row["kind"],
            klass=row["class"],
            old_group=row["old_group"],
            releases=tuple(row["releases"]),
            note=row["note"],
            r21_targets=targets,
            values=row.get("values", {}),
            unmapped_values=row.get("unmapped_values", {}),
            r21_value=row.get("r21_value"),
            successor=tuple(row.get("successor", ())),
        )
    return Aliases(rows=rows)


def _value_key(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fortran(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    return str(value)


def _spelled(assigns: Mapping[str, Any]) -> str:
    return ", ".join(f"{key} = {_fortran(value)}" for key, value in assigns.items())


def _reading(unknown: UnknownAssignment, row: AliasRow) -> AliasReading:
    base = dict(
        group=unknown.group, key=unknown.key, text=unknown.text, value=unknown.value,
        kind=row.kind, klass=row.klass, releases=row.releases,
    )
    head = f"{unknown.key} = {unknown.text}"
    if row.kind in ("renamed", "moved"):
        if row.values:
            assigns = row.values.get(_value_key(unknown.value))
            if assigns is None:
                return AliasReading(**base, status="unmapped",
                                    message=f"{head} has no r21.1 setting ({row.note})")
        else:
            assigns = {target: unknown.value for target in row.r21_targets}
        return AliasReading(**base, status="read", assigns=dict(assigns),
                            message=f"{head} read as {_spelled(assigns)} ({row.note})")
    if row.kind == "value-mapped":
        assigns = row.values.get(_value_key(unknown.value))
        if assigns is None:
            why = row.unmapped_values.get(_value_key(unknown.value), "CM1's documents give no r21.1 setting for this value")
            return AliasReading(**base, status="unmapped", message=f"{head}: {why} ({row.note})")
        if not assigns:
            return AliasReading(**base, status="same", message=f"{head} changes nothing in r21.1 ({row.note})")
        return AliasReading(**base, status="read", assigns=dict(assigns),
                            message=f"{head} read as {_spelled(assigns)} ({row.note})")
    if row.kind == "hard-wired":
        if _value_key(unknown.value) == _value_key(row.r21_value):
            return AliasReading(**base, status="same", message=f"{head} is what r21.1 always does ({row.note})")
        return AliasReading(**base, status="unmapped", message=f"{head} has no r21.1 setting ({row.note})")
    if row.kind == "derived":
        return AliasReading(**base, status="ignored", message=f"{head} is not read ({row.note})")
    return AliasReading(**base, status="removed", message=f"{head} is not a CM1 r21.1 setting ({row.note})")


def resolve(
    unknown: Iterable[UnknownAssignment],
    stated: Mapping[str, Any],
) -> tuple[tuple[AliasReading, ...], tuple[UnknownAssignment, ...]]:
    """Readings for the undeclared assignments the alias record knows.

    ``stated`` is the file's own r21.1 assignments, flat.  Returns the
    readings and the assignments the record does not know.
    """

    table = load()
    readings: list[AliasReading] = []
    rest: list[UnknownAssignment] = []
    for row in unknown:
        alias = table.row(row.group, row.key)
        if alias is None:
            rest.append(row)
            continue
        readings.append(_reading(row, alias))
    # A setting the file states under its r21.1 name wins; two readings that
    # set one r21.1 key differently cannot both be what the user meant.
    claimed: dict[str, tuple[Any, AliasReading]] = {}
    resolved: list[AliasReading] = []
    for reading in readings:
        clash = [
            (target, value) for target, value in reading.assigns.items()
            if (target in stated and stated[target] != value)
            or (target in claimed and claimed[target][0] != value)
        ]
        if clash:
            target, value = clash[0]
            other = (
                f"the file also sets {target} = {stated[target]}" if target in stated
                else f"{claimed[target][1].key} = {claimed[target][1].text} reads as {target} = {claimed[target][0]}"
            )
            resolved.append(AliasReading(
                group=reading.group, key=reading.key, text=reading.text, value=reading.value,
                kind=reading.kind, klass=reading.klass, releases=reading.releases,
                status="conflict",
                message=f"{reading.key} = {reading.text} reads as {target} = {value}, but {other}",
            ))
            continue
        for target, value in reading.assigns.items():
            claimed[target] = (value, reading)
        resolved.append(reading)
    return tuple(resolved), tuple(rest)


def assignments(readings: Iterable[AliasReading], stated: Mapping[str, Any]) -> dict[str, Any]:
    """The r21.1 settings the readings add; the file's own keys are kept."""

    added: dict[str, Any] = {}
    for reading in readings:
        if reading.status != "read":
            continue
        for target, value in reading.assigns.items():
            if target not in stated:
                added[target] = value
    return added
