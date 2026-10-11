"""Minimal Fortran namelist read/write for CM1 r21.1 decks.

Only the shapes CM1's own ``run/config_files`` use: one ``&group`` per block,
``key = value,`` lines, ``!`` comments, ``/`` terminator, scalar integer,
real, logical and character values.  Group order and key order are preserved
so an emitted deck diffs cleanly against the shipped one it came from.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, MutableMapping

_GROUP = re.compile(r"^\s*&(\w+)\s*$")
_END = re.compile(r"^\s*/\s*$")
_ASSIGN = re.compile(r"^\s*(\w+)\s*=\s*(.*?)\s*,?\s*$")

Deck = "OrderedDict[str, MutableMapping[str, Any]]"


def _strip_comment(line: str) -> str:
    out: list[str] = []
    quote: str | None = None
    for ch in line:
        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def parse_value(text: str) -> Any:
    text = text.strip().rstrip(",").strip()
    low = text.lower()
    if low in (".true.", "t", ".t."):
        return True
    if low in (".false.", "f", ".f."):
        return False
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(low.replace("d", "e"))
    except ValueError:
        return text


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return f"'{value}'"


def read(path: Path) -> Deck:
    deck: Deck = OrderedDict()
    group: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        match = _GROUP.match(line)
        if match:
            group = match.group(1)
            deck.setdefault(group, OrderedDict())
            continue
        if _END.match(line):
            group = None
            continue
        if group is None:
            continue
        match = _ASSIGN.match(line)
        if match:
            deck[group][match.group(1)] = parse_value(match.group(2))
    return deck


def write(deck: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for group, fields in deck.items():
        lines.append(f" &{group}")
        width = max((len(k) for k in fields), default=1)
        for key, value in fields.items():
            lines.append(f" {key.ljust(width)} = {format_value(value)},")
        lines.append(" /")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def locate(deck: Mapping[str, Mapping[str, Any]], key: str) -> str | None:
    """Return the group a namelist key lives in, or None."""

    for group, fields in deck.items():
        if key in fields:
            return group
    return None


def apply(deck: Deck, overrides: Mapping[str, Any]) -> list[str]:
    """Set every override in place.  Returns the keys the deck does not have."""

    unknown: list[str] = []
    for key, value in overrides.items():
        group = locate(deck, key)
        if group is None:
            unknown.append(key)
            continue
        deck[group][key] = value
    return unknown
