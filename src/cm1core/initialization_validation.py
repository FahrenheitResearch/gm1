"""What each CM1 initialization selector has actually been checked against.

Every capability is selectable now, and anything not
validated in the pass that exposed it carries a loud, machine-readable marker
in the run receipt and in ``gpuwm-cm1 doctor``, naming what was not checked.

This module is the marker's single source.  It reads
``records/CM1-INITIALIZATION-VALIDATION.json``, which names, per selector
value, whether it was validated, by what method, and against what evidence.
An entry that is not ``validated`` produces a receipt row stating in words
what a user would otherwise have to assume.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

VALIDATION_RECORD = Path(__file__).with_name("records") / "CM1-INITIALIZATION-VALIDATION.json"
VALIDATION_SCHEMA = "gpuwm-cm1.initialization-validation/v1"
UNVALIDATED_MARKER = "UNVALIDATED"


@lru_cache(maxsize=1)
def validation_record() -> Mapping[str, Any]:
    payload = json.loads(VALIDATION_RECORD.read_text(encoding="utf-8"))
    if payload.get("schema") != VALIDATION_SCHEMA:
        raise ValueError(
            f"{VALIDATION_RECORD} declares schema {payload.get('schema')!r}, "
            f"not {VALIDATION_SCHEMA!r}"
        )
    return payload


def selector_status(selector: str, value: object) -> Mapping[str, Any] | None:
    """The record row for one selector value, or ``None`` if it has none."""

    table = validation_record()["selectors"].get(selector)
    if table is None:
        return None
    return table.get(str(value))


def unvalidated_initialization(selection: Any, config: Any = None) -> dict[str, Any]:
    """Name every selected initialization capability that was not checked.

    The returned mapping always has the same shape, so a consumer can test one
    key rather than parse prose: ``status`` is ``"validated"`` only when every
    selected value carries a validated row, and ``items`` lists one entry per
    selected value that does not.
    """

    # iinit=8 and irandp=1 carry random numbers: CM1's own stream
    # (random_generator = 1) has rows of its own, "8:cm1" and "1:cm1".
    cm1_stream = int(getattr(selection, "random_generator", 0)) == 1
    iinit: object = f"{selection.iinit}:cm1" if cm1_stream and selection.iinit == 8 else selection.iinit
    irandp: object = f"{selection.irandp}:cm1" if cm1_stream and selection.irandp == 1 else selection.irandp
    selected: list[tuple[str, object]] = [
        ("isnd", selection.isnd),
        ("iwnd", selection.iwnd),
        ("iinit", iinit),
        ("ibalance", selection.ibalance),
        ("imoist", selection.imoist),
        ("irandp", irandp),
        ("maintain_rh", bool(selection.maintain_rh)),
    ]
    if config is not None and getattr(config, "initialization", None) is not None:
        selected.append(("iorigin", config.initialization.iorigin))
    items: list[dict[str, Any]] = []
    for selector, value in selected:
        row = selector_status(selector, value)
        if row is None:
            items.append(
                {
                    "selector": selector,
                    "value": value,
                    "marker": UNVALIDATED_MARKER,
                    "not_checked": (
                        f"{selector}={value} has no row in {VALIDATION_RECORD.name}; "
                        "nothing in this distribution has compared it against CM1"
                    ),
                    "method": None,
                    "evidence": None,
                }
            )
            continue
        if row.get("status") == "validated":
            continue
        items.append(
            {
                "selector": selector,
                "value": value,
                "marker": UNVALIDATED_MARKER,
                "not_checked": row.get("not_checked", "not stated"),
                "method": row.get("method"),
                "evidence": row.get("evidence"),
            }
        )
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "validated" if not items else UNVALIDATED_MARKER,
        "marker": None if not items else UNVALIDATED_MARKER,
        "items": items,
        "record": VALIDATION_RECORD.name,
    }


def validated_rows() -> list[dict[str, Any]]:
    """Every row in the record, flattened, for ``doctor``."""

    rows: list[dict[str, Any]] = []
    for selector, table in sorted(validation_record()["selectors"].items()):
        for value, row in sorted(table.items()):
            rows.append({"selector": selector, "value": value, **row})
    return rows
