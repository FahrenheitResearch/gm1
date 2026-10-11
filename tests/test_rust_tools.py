from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig

import pytest

from cm1core.rust_tools import (
    EXPECTED_ABIS,
    EXPECTED_LIBRARIES,
    NativeToolError,
    native_library_path,
    platform_tag,
    verify_native_libraries,
    verify_native_tools,
)


def _filename(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _library_filename(lib_name: str) -> str:
    if os.name == "nt":
        return f"{lib_name}.dll"
    if sys.platform == "darwin":
        return f"lib{lib_name}.dylib"
    return f"lib{lib_name}.so"


def _fixture(root: Path) -> dict[str, object]:
    binary_root = root / "bin"
    binary_root.mkdir(parents=True)
    rows = []
    for index, (name, abi) in enumerate(EXPECTED_ABIS.items(), start=1):
        payload = b"native-test-payload\0" + bytes([index])
        filename = _filename(name)
        (binary_root / filename).write_bytes(payload)
        rows.append(
            {
                "abi": abi,
                "bytes": len(payload),
                "filename": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    library_root = root / "lib"
    library_root.mkdir(parents=True)
    libraries = []
    for index, (name, lib_name) in enumerate(EXPECTED_LIBRARIES.items(), start=1):
        payload = b"native-test-library\0" + bytes([index])
        filename = _library_filename(lib_name)
        (library_root / filename).write_bytes(payload)
        libraries.append(
            {
                "bytes": len(payload),
                "filename": filename,
                "name": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "libraries": libraries,
        "platform": sysconfig.get_platform(),
        "platform_tag": platform_tag(),
        "python_minimum": "3.12",
        "schema": "gpuwm-cm1.native-tools/v1",
        "tools": rows,
    }
    (root / "native-tools.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def test_verified_native_inventory_requires_exact_hashes_and_platform(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    tools = verify_native_tools(tmp_path)
    assert set(tools) == set(EXPECTED_ABIS)

    tools["cm1-state"].write_bytes(b"changed")
    with pytest.raises(NativeToolError, match="differs from its installed manifest"):
        verify_native_tools(tmp_path)

    _fixture(tmp_path / "other")
    other_manifest = tmp_path / "other" / "native-tools.json"
    payload = json.loads(other_manifest.read_text(encoding="utf-8"))
    payload["platform_tag"] = "definitely_another_platform"
    other_manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(NativeToolError, match="another target's machine code"):
        verify_native_tools(tmp_path / "other")

    assert manifest["schema"] == "gpuwm-cm1.native-tools/v1"


def test_native_inventory_refuses_unknown_rows_and_missing_files(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    manifest["tools"][0]["unexpected"] = True
    (tmp_path / "native-tools.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NativeToolError, match="unknown or missing keys"):
        verify_native_tools(tmp_path)

    manifest = _fixture(tmp_path / "missing")
    filename = manifest["tools"][0]["filename"]
    (tmp_path / "missing" / "bin" / filename).unlink()
    with pytest.raises(NativeToolError, match="missing"):
        verify_native_tools(tmp_path / "missing")


def test_native_library_path_resolves_and_verifies_the_installed_cdylib(
    tmp_path: Path,
) -> None:
    manifest = _fixture(tmp_path)
    libraries = verify_native_libraries(tmp_path)
    assert set(libraries) == set(EXPECTED_LIBRARIES)
    resolved = native_library_path("cm1-state", tmp_path)
    assert resolved == tmp_path / "lib" / _library_filename("cm1_state")
    assert resolved.read_bytes().startswith(b"native-test-library")
    assert manifest["libraries"][0]["name"] == "cm1-state"

    with pytest.raises(NativeToolError, match="not an admitted installed artifact"):
        native_library_path("cm1-render", tmp_path)

    resolved.write_bytes(b"changed")
    with pytest.raises(NativeToolError, match="differs from its installed manifest"):
        native_library_path("cm1-state", tmp_path)


def test_native_library_inventory_refuses_unknown_rows_and_missing_files(
    tmp_path: Path,
) -> None:
    manifest = _fixture(tmp_path)
    manifest["libraries"][0]["unexpected"] = True
    (tmp_path / "native-tools.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NativeToolError, match="unknown or missing keys"):
        verify_native_libraries(tmp_path)

    manifest = _fixture(tmp_path / "renamed")
    manifest["libraries"][0]["name"] = "cm1-render"
    (tmp_path / "renamed" / "native-tools.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(NativeToolError, match="name is not admitted"):
        verify_native_libraries(tmp_path / "renamed")

    manifest = _fixture(tmp_path / "missing")
    filename = manifest["libraries"][0]["filename"]
    (tmp_path / "missing" / "lib" / filename).unlink()
    with pytest.raises(NativeToolError, match="missing"):
        verify_native_libraries(tmp_path / "missing")


def test_platform_tag_is_the_wheel_spelling_of_the_interpreter_platform() -> None:
    assert platform_tag() == sysconfig.get_platform().replace("-", "_").replace(
        ".", "_"
    ).replace(" ", "_")
    assert "-" not in platform_tag()
    assert "." not in platform_tag()
