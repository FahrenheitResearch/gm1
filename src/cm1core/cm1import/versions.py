"""Which CM1 release wrote a namelist, and what its CM1 r22.0 keys mean in r21.1.

``records/CM1-R22-NEW-KEYS.json`` lists every key CM1 r22.0 declares that
r21.1 does not (computed from r22.0's source), each with:

* ``class`` -- ``physics`` (changes what r22.0 computes), ``output`` (only
  what it writes) or ``cosmetic``;
* ``applies_when`` -- the settings under which r22.0 reads the key at all
  (cfl_limit only with adapt_dt = 1, trn_* only with testcase = 17);
* ``r21_equivalent`` -- the value at which r22.0 does what r21.1 does
  (cfl_limit = 1.0; brkup_db = 300e-6 m, or below 1e-20, which r22.0 reads as
  300e-6 m).

:func:`classify` gives each r22.0 key in a file one reading.  A key r22.0
does not read under the file's settings, or reads at its r21.1 equivalent, is
``same``: the run is what r21.1 would do.  Otherwise it ``differs``, and its
class says whether that changes the physics or only the output.

Two r21.1 keys change in an r22.0 file.  ``output_vort`` means staggered
vorticity in r22.0; r21.1's meaning is r22.0's ``output_vortinterp``.  And
r22.0 split ``cnstce`` into ``cnstch`` and ``cnstcq``, which give r21.1's
``cnstce`` when they are equal.  :func:`r21_settings` applies both.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Mapping

from .namelist import UnknownAssignment

RECORD_NAME = "CM1-R22-NEW-KEYS.json"
SCHEMA = "gm1.cm1-r22.0-new-namelist-keys/v1"
R21 = "r21.1"
R22 = "r22.0"
OLDER = "before r21.1"


@dataclass(frozen=True)
class R22Key:
    group: str
    key: str
    type: str
    klass: str
    applies_when: Mapping[str, Any] | None
    r21_equivalent: Mapping[str, Any]
    note: str


@dataclass(frozen=True)
class R22Reading:
    """One r22.0 key in a file.

    ``status`` is ``same`` (r22.0 does what r21.1 does with it) or
    ``differs``.  ``read_by_r22`` says whether r22.0 reads it under the
    file's settings.  ``assigns`` holds r21.1 settings it stands for
    (``output_vortinterp`` -> ``output_vort``; ``cnstch`` = ``cnstcq`` ->
    ``cnstce``).
    """

    group: str
    key: str
    text: str
    value: Any
    klass: str
    status: str
    read_by_r22: bool
    message: str
    assigns: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class R22Keys:
    keys: Mapping[tuple[str, str], R22Key]
    changed_meaning: Mapping[str, Mapping[str, Any]]
    removed_r21_keys: Mapping[str, Mapping[str, Any]]

    def row(self, group: str, key: str) -> R22Key | None:
        return self.keys.get((group, key))


@lru_cache(maxsize=1)
def load() -> R22Keys:
    payload = json.loads(resources.files("cm1core.records").joinpath(RECORD_NAME).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{RECORD_NAME} has schema {payload.get('schema')!r}, not {SCHEMA!r}")
    keys = {
        (group, key): R22Key(
            group=group, key=key, type=row["type"], klass=row["class"],
            applies_when=row["applies_when"], r21_equivalent=row["r21_equivalent"], note=row["note"],
        )
        for group, rows in payload["keys"].items()
        for key, row in rows.items()
    }
    return R22Keys(
        keys=keys,
        changed_meaning=payload["changed_meaning"],
        removed_r21_keys=payload["removed_r21_keys"],
    )


def detect(unknown: Iterable[UnknownAssignment]) -> str:
    """``r22.0`` if the file sets any r22.0-only key, ``before r21.1`` if any
    key only an older release declared, else ``r21.1``.  A file with both is
    reported as r22.0, since r22.0 keys change meaning first."""

    from . import aliases

    rows = tuple(unknown)
    table = load()
    if any(table.row(row.group, row.key) is not None for row in rows):
        return R22
    old = aliases.load()
    if any(old.row(row.group, row.key) is not None for row in rows):
        return OLDER
    return R21


def _as_real32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _equal(kind: str, left: Any, right: Any) -> bool:
    if kind in ("real32", "real64") or isinstance(left, float) or isinstance(right, float):
        try:
            a, b = float(left), float(right)
        except (TypeError, ValueError):
            return False
        if kind == "real64":
            return a == b
        return _as_real32(a) == _as_real32(b)
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right) and type(left) is type(right)
    return left == right


def condition_holds(condition: Mapping[str, Any] | None, settings: Mapping[str, Any]) -> bool:
    """Evaluate an ``applies_when`` condition on the file's r21.1 settings."""

    if condition is None:
        return True
    if "all" in condition:
        return all(condition_holds(part, settings) for part in condition["all"])
    if "any" in condition:
        return any(condition_holds(part, settings) for part in condition["any"])
    value = settings.get(condition["key"])
    return any(_equal("", value, allowed) if not isinstance(allowed, bool) else value is allowed
               for allowed in condition["in"])


def matches_equivalent(row: R22Key, value: Any, settings: Mapping[str, Any]) -> bool:
    """Whether ``value`` is the r21.1-equivalent value of this r22.0 key."""

    def check(rule: Mapping[str, Any]) -> bool:
        kind = rule["kind"]
        if kind == "value":
            return _equal(row.type, value, rule["value"])
        if kind == "at-most":
            return isinstance(value, (int, float)) and not isinstance(value, bool) and value <= rule["value"]
        if kind == "below":
            return isinstance(value, (int, float)) and not isinstance(value, bool) and value < rule["value"]
        if kind == "any-of":
            return any(check(option) for option in rule["options"])
        if kind == "value-when":
            for case in rule["cases"]:
                if condition_holds(case["when"], settings):
                    return _equal(row.type, value, case["value"])
            return _equal(row.type, value, rule["otherwise"])
        if kind == "r21-key":
            required = rule.get("requires_equal", ())
            return all(_equal(row.type, settings.get(name), value) for name in required)
        if kind == "unread":
            return False
        raise ValueError(f"unknown r21_equivalent kind {kind!r}")

    if isinstance(value, float) and not math.isfinite(value):
        return False
    return check(row.r21_equivalent)


def r21_settings(
    unknown: Iterable[UnknownAssignment],
    stated: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """The r21.1 settings an r22.0 file means, and why.

    ``output_vort`` takes the file's ``output_vortinterp`` (0 if unset), and
    ``cnstce`` takes ``cnstch`` when ``cnstch`` equals ``cnstcq``.  Returns
    ``({r21_key: value}, {r21_key: reason})``.
    """

    r22 = {row.key: row.value for row in unknown if load().row(row.group, row.key) is not None}
    settings: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    settings["output_vort"] = r22.get("output_vortinterp", 0)
    reasons["output_vort"] = (
        "in CM1 r22.0, r21.1's output_vort is output_vortinterp "
        f"(= {settings['output_vort']}); r22.0's output_vort writes staggered vorticity"
    )
    if "cnstce" not in stated and "cnstch" in r22 and "cnstcq" in r22:
        if _equal("real32", r22["cnstch"], r22["cnstcq"]):
            settings["cnstce"] = r22["cnstch"]
            reasons["cnstce"] = f"cnstch = cnstcq = {r22['cnstch']} is r21.1's cnstce"
    return settings, reasons


def classify(
    unknown: Iterable[UnknownAssignment],
    settings: Mapping[str, Any],
    *,
    r22_output_vort: Any = None,
) -> tuple[tuple[R22Reading, ...], tuple[UnknownAssignment, ...]]:
    """Readings for the r22.0 keys; returns them and the keys r22.0 does not declare.

    ``settings`` is the file's effective r21.1 settings (after defaults), with
    the r22.0 keys' own values available for the conditions that name them.
    ``r22_output_vort`` is the file's ``output_vort`` read with r22.0's
    meaning, reported as an output row when it is on.
    """

    table = load()
    readings: list[R22Reading] = []
    rest: list[UnknownAssignment] = []
    rows = tuple(unknown)
    context = dict(settings)
    context.update({row.key: row.value for row in rows if table.row(row.group, row.key) is not None})
    for row in rows:
        key = table.row(row.group, row.key)
        if key is None:
            rest.append(row)
            continue
        read = condition_holds(key.applies_when, context)
        equivalent = matches_equivalent(key, row.value, context)
        head = f"{row.key} = {row.text}"
        assigns: dict[str, Any] = {}
        if key.r21_equivalent.get("kind") == "r21-key" and equivalent:
            assigns[key.r21_equivalent["r21_key"]] = row.value
        if not read:
            status, message = "same", f"{head}: CM1 r22.0 does not read it with these settings ({key.note})"
        elif equivalent:
            status, message = "same", f"{head} is CM1 r21.1's behaviour ({key.note})"
        else:
            status, message = "differs", f"{head} has no CM1 r21.1 equivalent ({key.note})"
        readings.append(R22Reading(
            group=row.group, key=row.key, text=row.text, value=row.value, klass=key.klass,
            status=status, read_by_r22=read, message=message, assigns=assigns,
        ))
    if r22_output_vort not in (None, 0, False):
        meaning = table.changed_meaning["output_vort"]
        readings.append(R22Reading(
            group="param9", key="output_vort", text=str(r22_output_vort), value=r22_output_vort,
            klass="output", status="differs", read_by_r22=True,
            message=(
                f"output_vort = {r22_output_vort} in CM1 r22.0 writes {meaning['r22_meaning']}; "
                f"CM1 r21.1 has no such output (its output_vort is {meaning['r21_meaning']})"
            ),
        ))
    return tuple(readings), tuple(rest)
