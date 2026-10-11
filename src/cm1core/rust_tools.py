"""Verified launchers for the mandatory Rust executables bundled in the wheel."""

from __future__ import annotations

import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
from typing import Mapping


NATIVE_MANIFEST_SCHEMA = "gpuwm-cm1.native-tools/v1"
EXPECTED_ABIS = {
    "cm1-state": "gpuwm-cm1-state/v1",
    "gm1-forecast-checkpoint": (
        "gm1.forecast-checkpoint-cli/v1;carrier_abi=gm1.forecast-checkpoint-carriers/v1"
    ),
    "gm1-child-post-step": (
        "gm1.real-50m-child-post-step-cli/v1"
        ";checkpoint_carrier_abi=gm1.forecast-checkpoint-carriers/v1"
        ";history_carrier_abi=gm1.real-50m-child-render-carriers/v1"
    ),
    "cm1-forecast-publish": "gpuwm-cm1-forecast-publish/v3",
    "gm1-cartesian-restart": (
        "gm1.cartesian-restart-cli/v1;carrier_abi=gm1.cartesian-restart-carriers/v1"
    ),
    "gm1-history-grads-export": "gpuwm-cm1-history-grads-export/v1",
    "checkpoint-wind-grads": "gpuwm-cm1-checkpoint-wind-grads/v1",
    "cm1-r21-namelist": (
        "gm1.cm1-r21.1-namelist-cli/v2"
        ";document_schema=gm1.cm1-r21.1-namelist-document/v1"
        ";support_schema=gm1.cm1-r21.1-configuration-support/v1"
        ";binding_schema=gm1.cm1-r21.1-configuration-binding/v1"
        ";inspection_schema=gm1.cm1-r21.1-namelist-inspection/v1"
    ),
    "cm1-render": "gpuwm-cm1-render/v1",
    "cm1-f64-window": "gpuwm-cm1-f64-window/v1",
}
# Logical name -> Cargo [lib] name.  A cdylib has no ``--abi`` door, so its
# identity is the sealed byte count and digest alone.
EXPECTED_LIBRARIES = {"cm1-state": "cm1_state"}
_MANIFEST_KEYS = {
    "libraries",
    "platform",
    "platform_tag",
    "python_minimum",
    "schema",
    "tools",
}
_TOOL_KEYS = {"abi", "bytes", "filename", "sha256"}
_LIBRARY_KEYS = {"bytes", "filename", "name", "sha256"}


class NativeToolError(RuntimeError):
    """The installed Rust payload is missing, changed, or for another platform."""


def _filename(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _library_filename(lib_name: str) -> str:
    if os.name == "nt":
        return f"{lib_name}.dll"
    if sys.platform == "darwin":
        return f"lib{lib_name}.dylib"
    return f"lib{lib_name}.so"


def platform_tag() -> str:
    """Return this interpreter's wheel platform tag.

    A wheel is selected by its platform tag, not by the raw
    ``sysconfig.get_platform()`` string, and the two differ only by the
    ``-``/``.``/space to ``_`` substitution below.  Gating the installed
    payload on the tag is the documented compatibility relation across the
    3.11-3.13 range this project claims: any interpreter that pip would let
    install this wheel reports the same tag, while a payload built for a
    different tag names a genuinely different ELF/PE target.
    """

    raw = sysconfig.get_platform()
    return raw.replace("-", "_").replace(".", "_").replace(" ", "_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, object]:
    path = root / "native-tools.json"
    if not path.is_file() or path.is_symlink():
        raise NativeToolError(
            "installed native-tools.json is missing or not a regular file"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeToolError(
            f"installed native-tools.json is invalid: {error}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
        raise NativeToolError(
            "installed native tool manifest has unknown or missing keys"
        )
    if payload.get("schema") != NATIVE_MANIFEST_SCHEMA:
        raise NativeToolError("installed native tool manifest schema is not admitted")
    if payload.get("platform_tag") != platform_tag():
        raise NativeToolError(
            f"native tools target wheel platform tag "
            f"{payload.get('platform_tag')!r}, not {platform_tag()!r}; executing "
            "them would run another target's machine code"
        )
    if payload.get("python_minimum") != "3.12":
        raise NativeToolError(
            "native tool manifest Python floor is not the admitted 3.12"
        )
    return payload


def _verified_row(
    row: object, keys: set[str], subject: str
) -> tuple[str, int, str]:
    if not isinstance(row, dict) or set(row) != keys:
        raise NativeToolError(f"{subject} manifest row has unknown or missing keys")
    filename = row.get("filename")
    size = row.get("bytes")
    digest = row.get("sha256")
    if not isinstance(filename, str) or not filename:
        raise NativeToolError(f"{subject} filename is not admitted: {filename!r}")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise NativeToolError(f"{subject} byte count is invalid for {filename}")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise NativeToolError(f"{subject} SHA-256 is invalid for {filename}")
    return filename, size, digest


def _verified_file(path: Path, size: int, digest: str, filename: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise NativeToolError(f"{filename} is missing or not a regular file")
    if path.stat().st_size != size or _sha256(path) != digest:
        raise NativeToolError(f"{filename} differs from its installed manifest")
    return path


def _package_root(root: Path | None) -> Path:
    if root is None:
        traversable = resources.files("cm1core")
        try:
            root = Path(os.fspath(traversable))
        except TypeError as error:
            raise NativeToolError(
                "cm1core is not installed as a filesystem package; native tools cannot execute"
            ) from error
    return root.resolve(strict=True)


def verify_native_tools(root: Path | None = None) -> Mapping[str, Path]:
    """Return exact verified executables from an installed package root."""

    root = _package_root(root)
    manifest = _read_manifest(root)
    rows = manifest["tools"]
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_ABIS):
        raise NativeToolError("native tool manifest inventory is incomplete")
    expected_filenames = {_filename(name): name for name in EXPECTED_ABIS}
    observed: dict[str, Path] = {}
    for row in rows:
        filename, size, digest = _verified_row(row, _TOOL_KEYS, "native tool")
        if filename not in expected_filenames:
            raise NativeToolError(f"native tool filename is not admitted: {filename!r}")
        logical_name = expected_filenames[filename]
        if logical_name in observed:
            raise NativeToolError(
                f"duplicate native tool manifest row for {logical_name}"
            )
        if row.get("abi") != EXPECTED_ABIS[logical_name]:
            raise NativeToolError(f"native tool ABI mismatch for {logical_name}")
        observed[logical_name] = _verified_file(
            root / "bin" / filename, size, digest, f"native tool {filename}"
        )
    if set(observed) != set(EXPECTED_ABIS):
        raise NativeToolError("native tool manifest names are incomplete")
    return observed


def verify_native_libraries(root: Path | None = None) -> Mapping[str, Path]:
    """Return exact verified shared libraries from an installed package root."""

    root = _package_root(root)
    manifest = _read_manifest(root)
    rows = manifest["libraries"]
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_LIBRARIES):
        raise NativeToolError("native library manifest inventory is incomplete")
    observed: dict[str, Path] = {}
    for row in rows:
        filename, size, digest = _verified_row(row, _LIBRARY_KEYS, "native library")
        logical_name = row.get("name")
        if not isinstance(logical_name, str) or logical_name not in EXPECTED_LIBRARIES:
            raise NativeToolError(
                f"native library name is not admitted: {logical_name!r}"
            )
        if filename != _library_filename(EXPECTED_LIBRARIES[logical_name]):
            raise NativeToolError(
                f"native library filename is not admitted: {filename!r}"
            )
        if logical_name in observed:
            raise NativeToolError(
                f"duplicate native library manifest row for {logical_name}"
            )
        observed[logical_name] = _verified_file(
            root / "lib" / filename, size, digest, f"native library {filename}"
        )
    if set(observed) != set(EXPECTED_LIBRARIES):
        raise NativeToolError("native library manifest names are incomplete")
    return observed


def native_library_path(name: str, root: Path | None = None) -> Path:
    """Return one hash-verified installed cdylib the Cartesian runtime opens."""

    if name not in EXPECTED_LIBRARIES:
        raise NativeToolError(
            f"native library {name!r} is not an admitted installed artifact; "
            "loading it would bind the runtime to an unsealed shared object"
        )
    return verify_native_libraries(root)[name]


def _launch(name: str) -> int:
    try:
        executable = verify_native_tools()[name]
    except (OSError, NativeToolError) as error:
        print(f"{name}: {error}", file=sys.stderr)
        return 2
    try:
        completed = subprocess.run([str(executable), *sys.argv[1:]], check=False)
    except OSError as error:
        print(f"{name}: cannot execute verified native tool: {error}", file=sys.stderr)
        return 2
    return completed.returncode


def state_main() -> int:
    return _launch("cm1-state")


def gm1_forecast_checkpoint_main() -> int:
    return _launch("gm1-forecast-checkpoint")


def gm1_cartesian_restart_main() -> int:
    return _launch("gm1-cartesian-restart")


def render_main() -> int:
    return _launch("cm1-render")


def f64_window_main() -> int:
    return _launch("cm1-f64-window")


def grads_export_main() -> int:
    return _launch("gm1-history-grads-export")


def checkpoint_wind_grads_main() -> int:
    return _launch("checkpoint-wind-grads")


def cm1_r21_namelist_main() -> int:
    return _launch("cm1-r21-namelist")


__all__ = [
    "EXPECTED_ABIS",
    "grads_export_main",
    "checkpoint_wind_grads_main",
    "cm1_r21_namelist_main",
    "EXPECTED_LIBRARIES",
    "NATIVE_MANIFEST_SCHEMA",
    "NativeToolError",
    "f64_window_main",
    "gm1_cartesian_restart_main",
    "gm1_forecast_checkpoint_main",
    "native_library_path",
    "platform_tag",
    "render_main",
    "state_main",
    "verify_native_libraries",
    "verify_native_tools",
]
