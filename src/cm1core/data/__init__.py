"""Small package-owned registry/configuration assets."""

from __future__ import annotations

import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import re
from typing import Mapping


PACKAGED_INPUTS_SCHEMA = "gpuwm-cm1.required-inputs/v1"
_INDEX_NAME = "packaged-inputs.json"
_INDEX_KEYS = {"inputs", "redistribution", "schema"}
_INPUT_KEYS = {
    "bytes",
    "filename",
    "name",
    "notice_section",
    "sha256",
    "shipped",
    "source_path",
    "source_release",
}


class PackagedInputError(RuntimeError):
    """The installed pristine input asset is missing, changed, or unadmitted."""


def _root() -> Path:
    traversable = resources.files("cm1core.data")
    try:
        return Path(os.fspath(traversable)).resolve(strict=True)
    except TypeError as error:
        raise PackagedInputError(
            "cm1core.data is not installed as a filesystem package; the pristine "
            "input sounding cannot be opened by path"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_index(root: Path) -> list[dict[str, object]]:
    path = root / _INDEX_NAME
    if not path.is_file() or path.is_symlink():
        raise PackagedInputError(
            f"installed {_INDEX_NAME} is missing or not a regular file"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackagedInputError(
            f"installed {_INDEX_NAME} is invalid: {error}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != _INDEX_KEYS:
        raise PackagedInputError(
            "installed packaged-input index has unknown or missing keys"
        )
    if payload.get("schema") != PACKAGED_INPUTS_SCHEMA:
        raise PackagedInputError(
            "installed packaged-input index schema is not admitted"
        )
    rows = payload["inputs"]
    if not isinstance(rows, list) or not rows:
        raise PackagedInputError("installed packaged-input inventory is empty")
    admitted: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != _INPUT_KEYS:
            raise PackagedInputError(
                "packaged-input index row has unknown or missing keys"
            )
        admitted.append(row)
    return admitted


def packaged_inputs() -> Mapping[str, Mapping[str, object]]:
    """Return the provenance row of every pristine asset shipped in the wheel."""

    rows = _read_index(_root())
    index: dict[str, Mapping[str, object]] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise PackagedInputError("packaged-input name is not admitted")
        if name in index:
            raise PackagedInputError(f"duplicate packaged-input row for {name}")
        index[name] = row
    return index


def packaged_input_path(name: str) -> Path:
    """Return one hash-verified pristine asset installed beside this package."""

    root = _root()
    row = packaged_inputs().get(name)
    if row is None:
        raise PackagedInputError(
            f"packaged input {name!r} is not in the installed inventory; the "
            "Cartesian runtime would read an unprovenanced sounding"
        )
    filename = row.get("filename")
    size = row.get("bytes")
    digest = row.get("sha256")
    if not isinstance(filename, str) or "/" in filename or "\\" in filename:
        raise PackagedInputError(f"packaged-input filename is invalid for {name}")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise PackagedInputError(f"packaged-input byte count is invalid for {name}")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PackagedInputError(f"packaged-input SHA-256 is invalid for {name}")
    path = root / filename
    if not path.is_file() or path.is_symlink():
        raise PackagedInputError(
            f"packaged input {filename} is missing or not a regular file"
        )
    if path.stat().st_size != size or _sha256(path) != digest:
        raise PackagedInputError(
            f"packaged input {filename} differs from its installed provenance row"
        )
    return path


__all__ = [
    "PACKAGED_INPUTS_SCHEMA",
    "PackagedInputError",
    "packaged_input_path",
    "required_input_identity",
    "required_inputs",
    "packaged_inputs",
]


def required_inputs() -> Mapping[str, Mapping[str, object]]:
    """Identities of the CM1 assets this package needs but does not ship.

    CM1-AUTHORITY.json's redistribution record names the tar-only runtime
    assets as a blocker and its package policy is to ship identities and maps
    rather than CM1 runtime files.  Recording the identity lets a caller's own
    extraction be verified without redistributing anything.
    """

    root = _root()
    return {
        str(entry["name"]): dict(entry)
        for entry in _read_index(root)
        if not entry.get("shipped", False)
    }


def required_input_identity(name: str) -> Mapping[str, object]:
    """One recorded identity, or a refusal naming what is unknown."""

    recorded = required_inputs()
    if name not in recorded:
        raise PackagedInputError(
            f"required input {name!r} is not in the installed identity inventory; "
            "the runtime would read an unprovenanced asset"
        )
    return recorded[name]
