"""Structure gates for what the built wheel and sdist are obliged to carry."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import runpy
import sys
import tomllib

import pytest

from cm1core.data import PACKAGED_INPUTS_SCHEMA, packaged_input_path, packaged_inputs
from cm1core.rust_tools import EXPECTED_ABIS, EXPECTED_LIBRARIES, platform_tag


ROOT = Path(__file__).resolve().parents[1]
SOUNDING_NAME = "cm1r21.1-cpm-radconvequil-sounding"
SOUNDING_FILENAME = "cm1r21.1-cpm_RadConvEquil-input_sounding"


def _literal(module_path: Path, name: str) -> object:
    """Return one literal module constant without importing the module.

    ``build_rust`` imports setuptools build commands that only exist in the
    pinned build environment, so the inventory it declares is read from the
    file instead of from an import.
    """

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{module_path.name} does not define {name}")


def _tool_module(name: str) -> dict[str, object]:
    return runpy.run_path(str(ROOT / "tools" / f"{name}.py"), run_name=name)


def test_build_seals_exactly_the_tools_and_libraries_the_runtime_verifies() -> None:
    tools = _literal(ROOT / "build_rust.py", "NATIVE_TOOLS")
    libraries = _literal(ROOT / "build_rust.py", "NATIVE_LIBRARIES")
    assert {name: abi for name, _, abi, _ in tools} == EXPECTED_ABIS
    assert {name for name, _, _ in libraries} == set(EXPECTED_LIBRARIES)
    assert {name: lib for name, _, lib in libraries} == EXPECTED_LIBRARIES
    for name, _, _, stems in tools:
        assert name in stems, f"{name} must accept its own Cargo target name"


def test_every_sealed_tool_has_a_cargo_target_that_can_emit_it() -> None:
    tools = _literal(ROOT / "build_rust.py", "NATIVE_TOOLS")
    for _, relative_manifest, _, stems in tools:
        crate = (ROOT / relative_manifest).parent
        manifest = tomllib.loads(
            (ROOT / relative_manifest).read_text(encoding="utf-8")
        )
        declared = {entry["name"] for entry in manifest.get("bin", [])}
        # Cargo also auto-discovers src/bin/<stem>.rs and src/main.rs.
        discovered = {item.stem for item in (crate / "src" / "bin").glob("*.rs")}
        if (crate / "src" / "main.rs").is_file():
            discovered.add(manifest["package"]["name"])
        assert set(stems) & (declared | discovered), stems


def test_state_crate_publishes_the_cdylib_the_cartesian_runtime_opens() -> None:
    manifest = tomllib.loads(
        (ROOT / "rust/cm1-state/Cargo.toml").read_text(encoding="utf-8")
    )
    assert "cdylib" in manifest["lib"]["crate-type"]
    assert manifest["lib"]["name"] == EXPECTED_LIBRARIES["cm1-state"]


def test_platform_tag_matches_the_wheel_tag_this_interpreter_would_build() -> None:
    for module_name in (
        "setuptools.command.bdist_wheel",
        "setuptools._distutils.command.bdist_wheel",
        "wheel.bdist_wheel",
    ):
        try:
            __import__(module_name)
        except ImportError:
            continue
        get_platform = getattr(sys.modules[module_name], "get_platform", None)
        if get_platform is None:
            continue
        assert platform_tag() == get_platform(None)
        assert _literal(ROOT / "build_rust.py", "NATIVE_MANIFEST_SCHEMA")
        return
    pytest.skip("no wheel platform-tag implementation is importable")


def test_pyproject_ships_the_native_payload_and_packaged_inputs() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["cm1core"]
    assert {"bin/*", "data/*", "lib/*", "native-tools.json"} <= set(package_data)
    scripts = pyproject["project"]["scripts"]
    # Every advertised console script must reach a real launcher; the sealed
    # tools no door dispatches to stay reachable through the inventory only.
    import cm1core.rust_tools as rust_tools

    for target in scripts.values():
        module_name, _, attribute = target.partition(":")
        if module_name != "cm1core.rust_tools":
            continue
        assert hasattr(rust_tools, attribute)


def test_manifest_in_carries_the_smoke_tools_and_packaged_inputs() -> None:
    manifest_in = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert "include tools/smoke_installed_wheel.py" in manifest_in
    assert "include tools/smoke_installed_cuda_wheel.py" in manifest_in
    assert "recursive-include src/cm1core/data *" in manifest_in
    assert "recursive-include rust/cm1-state/schema *.json" in manifest_in


def test_the_cm1_sounding_identity_is_recorded_and_not_redistributed() -> None:
    """CM1 runtime assets are named, not shipped.

    CM1-AUTHORITY.json's redistribution record names the tar-only runtime
    assets as a blocker and its package policy is to ship identities and maps
    rather than CM1 runtime files. Bundling the sounding verbatim broke that
    policy; recording its identity lets the caller's own extraction be
    verified instead.

    What this gives up: the package no longer works without the caller
    fetching one 869-byte file. The door's refusal names the file, the
    retrieval tool, the size and the digest.
    """

    from cm1core.data import (
        PACKAGED_INPUTS_SCHEMA,
        PackagedInputError,
        packaged_input_path,
        required_input_identity,
    )

    assert PACKAGED_INPUTS_SCHEMA == "gpuwm-cm1.required-inputs/v1"
    identity = required_input_identity("cm1r21.1-cpm-radconvequil-sounding")
    assert identity["shipped"] is False
    assert identity["bytes"] == 869
    assert identity["sha256"] == (
        "295bf8d89fa4624676e8b0d90ce232cd5b053587c699f23700fd22df5feec6a5"
    )
    assert identity["source_release"] == "cm1r21.1"
    assert identity["source_path"] == "run/config_files/cpm_RadConvEquil/input_sounding"
    assert identity["notice_section"] == "CM1 attribution"

    # The bytes themselves must not be installed.
    with pytest.raises(PackagedInputError):
        packaged_input_path("cm1r21.1-cpm-radconvequil-sounding")

def test_notice_carries_the_attribution_the_packaged_input_cites() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for row in packaged_inputs().values():
        assert str(row["notice_section"]) in notice
    assert "CM1 Numerical Model, Release 21.1" in notice


def test_artifact_inspector_requires_the_new_wheel_and_sdist_members() -> None:
    inspector = _tool_module("inspect_artifacts")
    assert inspector["NATIVE_TOOL_ABIS"] == EXPECTED_ABIS
    assert set(inspector["NATIVE_LIBRARY_NAMES"]) == set(EXPECTED_LIBRARIES)
    assert inspector["EXPECTED_LIBRARY_MEMBERS"] == {
        f"cm1core/lib/{inspector['native_library_filename'](lib_name)}"
        for lib_name in EXPECTED_LIBRARIES.values()
    }
    # The identity index ships; the CM1 asset it names must not.
    assert SOUNDING_FILENAME in inspector["REQUIRED_PACKAGED_INPUTS"]
    assert inspector["REQUIRED_INPUT_INDEX"] == "packaged-inputs.json"
    assert "packaged-inputs.json" not in inspector["REQUIRED_PACKAGED_INPUTS"]
    for schema in inspector["REQUIRED_STATE_SCHEMAS"]:
        assert (ROOT / "rust/cm1-state/schema" / schema).is_file()
    assert inspector["platform_tag"]() == platform_tag()


def test_smoke_tools_expect_the_full_sealed_inventory() -> None:
    for name in ("smoke_installed_wheel", "smoke_installed_cuda_wheel"):
        module = _tool_module(name)
        assert module["SEALED_NATIVE_TOOLS"] == set(EXPECTED_ABIS)
        assert module["SEALED_NATIVE_LIBRARIES"] == set(EXPECTED_LIBRARIES)
        assert module["REQUIRED_PACKAGED_INPUTS"] == {SOUNDING_NAME}


def test_wheel_smoke_blocker_codes_exist_in_preflight() -> None:
    module = _tool_module("smoke_installed_wheel")
    source = (ROOT / "src/cm1core/preflight.py").read_text(encoding="utf-8")
    declared = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("CM1-")
    }
    expected = (
        module["REQUIRED_BLOCKER_CODES"]
        | module["ADMITTED_BLOCKER_CODES"]
        | module["FORBIDDEN_BLOCKER_CODES"]
    )
    assert expected <= declared
    assert module["REQUIRED_BLOCKER_CODES"] <= module["ADMITTED_BLOCKER_CODES"]
    assert not module["ADMITTED_BLOCKER_CODES"] & module["FORBIDDEN_BLOCKER_CODES"]


def test_artifact_inspector_declares_each_native_abi_exactly_once() -> None:
    # A repeated dict key silently keeps the last value: a stale
    # forecast-publish/v2 line after the v3 one made every correct wheel fail
    # inspection with "ABI mismatch for cm1-forecast-publish".
    tree = ast.parse((ROOT / "tools" / "inspect_artifacts.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "NATIVE_TOOL_ABIS"
            for target in node.targets
        ):
            keys = [ast.literal_eval(key) for key in node.value.keys]
            assert len(keys) == len(set(keys)), keys
            assert dict(zip(keys, map(ast.literal_eval, node.value.values))) == dict(EXPECTED_ABIS)
            return
    raise AssertionError("inspect_artifacts.py does not define NATIVE_TOOL_ABIS")


def test_the_wheel_build_ships_the_example_decks() -> None:
    source = (ROOT / "build_rust.py").read_text(encoding="utf-8")
    assert '(root / "examples").glob("*.toml")' in source
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include examples *.toml" in manifest


def test_every_shipped_cartesian_deck_loads_and_names_its_sounding_need() -> None:
    from cm1core.doors import _example_summary

    rows = {deck.stem: _example_summary(deck) for deck in sorted((ROOT / "examples").glob("cartesian-*.toml"))}
    assert rows
    for name, row in rows.items():
        assert row["loads"] is True, (name, row.get("load_error"))
        assert row["lane"] == "cartesian", name
        assert isinstance(row["needs_sounding"], bool), name
    # The two analysis decks are tornado workflows; a deck that needs a file
    # nobody installs cannot be a first run.
    for name in ("cartesian-tornado-budgets", "cartesian-tornado-lagrangian-analysis"):
        assert rows[name]["needs_sounding"] is False, name
