"""Read a CM1 namelist.input with the native ``cm1-r21-namelist`` parser.

The Rust tool types every assignment against the 583-field r21.1 inventory
generated from the pinned source, and with ``--unknown report`` returns every
assignment r21.1 does not declare instead of refusing the file, so a namelist
from CM1 r22.0 or from before r19 is read completely.  This module runs it and
turns its ``gm1.cm1-r21.1-namelist-inspection/v1`` JSON into Python values.

The grammar is gfortran's: several assignments per line, blank or comma
separators, no final comma needed, ``/`` or ``&end`` terminators, ``!``
comments outside strings, any letter case and every Fortran logical spelling.
Text outside the groups (a title line, notes between groups) is passed over as
gfortran passes over it, and returned in ``outside`` so a caller can say CM1
ignores it.  ``group_order`` keeps the order the file opens its groups, which
decides whether CM1 starts (cm1.F:480-482).
No cm1.exe, CM1 tree or ``CM1_AUTHORITY_ROOT`` is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

TOOL_NAME = "cm1-r21-namelist"
INSPECTION_SCHEMA = "gm1.cm1-r21.1-namelist-inspection/v1"
#: An explicit path to the tool, for a source checkout or a test.
TOOL_ENV = "GM1_CM1_R21_NAMELIST"
_REPO = Path(__file__).resolve().parents[3]
#: Where a source checkout's build puts the tool.
CHECKOUT_CANDIDATES = (
    _REPO / "rust" / "cm1-state" / "target" / "release" / TOOL_NAME,
    _REPO / "build" / "rust-native-target" / "release" / TOOL_NAME,
)
BUILD_COMMAND = "cargo build --release --locked --bin cm1-r21-namelist --manifest-path rust/cm1-state/Cargo.toml"


class NamelistError(ValueError):
    """The file is not a namelist the parser can read; the message says where."""


class NamelistToolError(RuntimeError):
    """The native parser is missing or is not the version this module reads."""


@dataclass(frozen=True)
class Assignment:
    group: str
    key: str
    type: str
    value: Any
    text: str


@dataclass(frozen=True)
class UnknownAssignment:
    """An assignment r21.1 does not declare, as the file wrote it."""

    group: str
    key: str
    group_declared: bool
    text: str
    value: Any
    value_type: str


@dataclass(frozen=True)
class OutsideText:
    """A line outside every group; gfortran's NAMELIST read skips it."""

    line: int
    text: str


@dataclass(frozen=True)
class ParsedNamelist:
    path: str
    sha256: str
    document_sha256: str
    groups: Mapping[str, Mapping[str, Assignment]]
    unknown: tuple[UnknownAssignment, ...]
    #: Every group, declared or not, in the order the file opens it.
    group_order: tuple[str, ...] = ()
    #: Text outside every group, which CM1 never reads.
    outside: tuple[OutsideText, ...] = ()

    def values(self) -> dict[str, dict[str, Any]]:
        """``{group: {key: value}}`` for the declared assignments."""

        return {group: {key: row.value for key, row in rows.items()} for group, rows in self.groups.items()}

    def flat(self) -> dict[str, Any]:
        """``{key: value}`` for the declared assignments."""

        result: dict[str, Any] = {}
        for rows in self.groups.values():
            for key, row in rows.items():
                result.setdefault(key, row.value)
        return result

    def stated_keys(self) -> frozenset[str]:
        return frozenset(key for rows in self.groups.values() for key in rows)


def _is_tool(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


@lru_cache(maxsize=8)
def _checked(path: str) -> str:
    try:
        completed = subprocess.run([path, "--abi"], capture_output=True, text=True, timeout=30, check=False)
    except OSError as error:
        raise NamelistToolError(f"{path} cannot run: {error}") from error
    from ..rust_tools import EXPECTED_ABIS

    abi = completed.stdout.strip()
    if completed.returncode != 0 or abi != EXPECTED_ABIS[TOOL_NAME]:
        raise NamelistToolError(
            f"{path} reports ABI {abi!r}; this importer reads {EXPECTED_ABIS[TOOL_NAME]!r}. "
            f"Rebuild it: {BUILD_COMMAND}"
        )
    return path


def locate_tool() -> Path:
    """The parser: ``$GM1_CM1_R21_NAMELIST``, the wheel's sealed copy, or a checkout build."""

    explicit = os.environ.get(TOOL_ENV)
    if explicit:
        path = Path(explicit)
        if not _is_tool(path):
            raise NamelistToolError(f"{TOOL_ENV}={explicit} is not an executable file")
        return Path(_checked(str(path)))
    from .. import rust_tools

    try:
        installed = rust_tools.verify_native_tools().get(TOOL_NAME)
    except (OSError, rust_tools.NativeToolError):
        installed = None
    if installed is not None:
        return Path(_checked(str(installed)))
    for candidate in CHECKOUT_CANDIDATES:
        if _is_tool(candidate):
            return Path(_checked(str(candidate)))
    raise NamelistToolError(
        f"the native CM1 namelist parser {TOOL_NAME} is not installed; in a source "
        f"checkout build it with: {BUILD_COMMAND}"
    )


def parse_inspection(payload: Mapping[str, Any], *, path: str = "", sha256: str = "") -> ParsedNamelist:
    """The parser's JSON as Python values."""

    if payload.get("schema") != INSPECTION_SCHEMA:
        raise NamelistToolError(f"parser output schema {payload.get('schema')!r} is not {INSPECTION_SCHEMA}")
    missing = [name for name in ("group_order", "outside_groups") if name not in payload]
    if missing:
        raise NamelistToolError(
            f"the parser's output has no {', '.join(missing)}: it was built before this importer. "
            f"Rebuild it: {BUILD_COMMAND}"
        )
    groups: dict[str, dict[str, Assignment]] = {}
    for group, rows in payload["values"].items():
        groups[group] = {
            key: Assignment(group=group, key=key, type=row["type"], value=row["value"], text=row["text"])
            for key, row in rows.items()
        }
    unknown = tuple(
        UnknownAssignment(
            group=row["group"],
            key=row["field"],
            group_declared=row["group_declared"],
            text=row["text"],
            value=row["value"],
            value_type=row["value_type"],
        )
        for row in payload["unknown"]
    )
    return ParsedNamelist(
        path=path,
        sha256=sha256,
        document_sha256=payload["document_sha256"],
        groups=groups,
        unknown=unknown,
        group_order=tuple(payload["group_order"]),
        outside=tuple(OutsideText(line=int(row["line"]), text=row["text"]) for row in payload["outside_groups"]),
    )


def inspect(path: str | os.PathLike[str], *, tool: Path | None = None) -> ParsedNamelist:
    """Parse ``path``; undeclared keys come back in ``unknown``, not as a refusal."""

    source = Path(path).resolve(strict=True)
    executable = tool if tool is not None else locate_tool()
    completed = subprocess.run(
        [str(executable), "inspect", "--json", "--unknown", "report", str(source)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip().removeprefix(f"{TOOL_NAME}: ")
        raise NamelistError(f"{source}: {message}")
    payload = json.loads(completed.stdout)
    return parse_inspection(
        payload,
        path=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
