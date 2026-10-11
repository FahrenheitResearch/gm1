"""CM1 r21.1's own values for the keys a namelist does not set.

``records/CM1-R21-NAMELIST-DEFAULTS.json`` is generated from the pinned r21.1
source by ``tools/generate_cm1_r21_namelist_support.py``.  For each of the 583
fields it gives the value CM1 holds when the file is silent and how that value
arises (``basis``):

``cm1-assignment``
    set before the namelist is read (cm1.F's start-up block, or
    init_module_wrf_error for ``logging``);
``declaration-initializer``
    the module variable's initializer (NSSL internals);
``static-zero``
    r21.1 sets nothing, and the compilers zero-fill static storage.  An
    importer that fills such a key should say so: an unset ``iorigin`` reads
    0, which CM1 itself stops on (param.F:6668).

It also carries r21.1's shipped ``run/namelist.input`` values, which the
translator's "differs from CM1's distributed deck" test compares against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

RECORD_NAME = "CM1-R21-NAMELIST-DEFAULTS.json"
SCHEMA = "gm1.cm1-r21.1-namelist-defaults/v1"


@dataclass(frozen=True)
class FieldDefault:
    group: str
    key: str
    type: str
    code_default: Any
    basis: str
    source: str | None
    has_distributed_value: bool
    distributed_value: Any


@dataclass(frozen=True)
class Defaults:
    fields: Mapping[str, Mapping[str, FieldDefault]]
    required_groups: tuple[str, ...]
    bases: Mapping[str, str]
    distributed: Mapping[str, Any]

    def field(self, group: str, key: str) -> FieldDefault | None:
        return self.fields.get(group, {}).get(key)

    def groups_of(self, key: str) -> tuple[str, ...]:
        return tuple(group for group, rows in self.fields.items() if key in rows)

    def distributed_flat(self) -> Mapping[str, Any]:
        """r21.1's shipped run/namelist.input, flattened by key."""

        return self.distributed

    def type_of(self, key: str) -> str | None:
        for group in self.groups_of(key):
            return self.fields[group][key].type
        return None

    def fill(self, stated: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], tuple[FieldDefault, ...]]:
        """Every r21.1 key, flat: the stated value, else CM1's code default.

        ``stated`` is ``{group: {key: value}}``.  Returns the flat effective
        values and the defaults that were used, in group order.  A name two
        NSSL groups share holds one Fortran variable, so the first value wins.
        """

        effective: dict[str, Any] = {}
        for group, rows in stated.items():
            for key, value in rows.items():
                effective.setdefault(key, value)
        filled: list[FieldDefault] = []
        for group in sorted(self.fields):
            for key, row in self.fields[group].items():
                if key in effective:
                    continue
                effective[key] = row.code_default
                filled.append(row)
        return effective, tuple(filled)


def _payload() -> Mapping[str, Any]:
    text = resources.files("cm1core.records").joinpath(RECORD_NAME).read_text(encoding="utf-8")
    payload = json.loads(text)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{RECORD_NAME} has schema {payload.get('schema')!r}, not {SCHEMA!r}")
    return payload


@lru_cache(maxsize=1)
def load() -> Defaults:
    payload = _payload()
    fields: dict[str, dict[str, FieldDefault]] = {}
    for group, rows in payload["groups"].items():
        fields[group] = {
            key: FieldDefault(
                group=group,
                key=key,
                type=row["type"],
                code_default=row["code_default"],
                basis=row["basis"],
                source=row["source"],
                has_distributed_value="distributed_value" in row,
                distributed_value=row.get("distributed_value"),
            )
            for key, row in rows.items()
        }
    distributed: dict[str, Any] = {}
    for group in sorted(fields):
        for key, row in fields[group].items():
            if row.has_distributed_value:
                distributed.setdefault(key, row.distributed_value)
    return Defaults(
        fields=fields,
        required_groups=tuple(payload["required_groups"]["groups"]),
        bases=dict(payload["bases"]),
        distributed=distributed,
    )
