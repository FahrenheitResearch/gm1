"""Shared fixtures for the CM1 namelist importer tests.

The importer runs the native ``cm1-r21-namelist`` parser.  In an installed
wheel it is sealed in ``cm1core/bin``; in a checkout it is built from
``rust/cm1-state``.  :func:`namelist_tool` finds it, building it once with
Cargo when a checkout has not yet, so these tests exercise the real parser
rather than a stand-in.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "cm1import"

SUPERCELL = DATA / "r21.1-supercell.namelist.input"
SQUALL_LINE = DATA / "r21.1-squall_line.namelist.input"
RUN_DEFAULT = DATA / "r21.1-run.namelist.input"
R22_SUPERCELL = DATA / "r22.0-supercell.namelist.input"
R18_RUN = DATA / "r18-run.namelist.input"

_BUILT: list[Path] = []


def namelist_tool() -> Path:
    from cm1core.cm1import import namelist

    if _BUILT:
        return _BUILT[0]
    try:
        tool = namelist.locate_tool()
    except namelist.NamelistToolError:
        cargo = shutil.which("cargo")
        if cargo is None:
            pytest.skip(f"the native parser is not built and Cargo is absent: {namelist.BUILD_COMMAND}")
        subprocess.run(
            [cargo, "build", "--release", "--locked", "-j", "4", "--bin", "cm1-r21-namelist",
             "--manifest-path", str(ROOT / "rust" / "cm1-state" / "Cargo.toml")],
            check=True, capture_output=True, cwd=ROOT,
        )
        tool = namelist.locate_tool()
    _BUILT.append(tool)
    return tool


def write(tmp_path: Path, text: str, name: str = "namelist.input") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="ascii")
    return path


def edited(source: Path, tmp_path: Path, replacements: dict[str, str], name: str = "namelist.input") -> Path:
    """A copy of ``source`` with whole lines ``key = ...`` replaced."""

    lines = []
    seen = set()
    for line in source.read_text(encoding="ascii").splitlines():
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip().lower() if "=" in stripped else None
        if key in replacements:
            lines.append(f" {key} = {replacements[key]},")
            seen.add(key)
        else:
            lines.append(line)
    missing = set(replacements) - seen
    if missing:
        raise KeyError(f"{source.name} sets none of {sorted(missing)}")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def flat_toml(text: str) -> dict[str, object]:
    import tomllib

    def walk(node: dict[str, object], prefix: str) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in node.items():
            if isinstance(value, dict):
                out.update(walk(value, f"{prefix}{key}."))
            else:
                out[f"{prefix}{key}"] = value
        return out

    return walk(tomllib.loads(text), "")


# ---------------------------------------------------------------------------
# CM1's shipped decks and the real-namelist corpus, for the coverage report
# ---------------------------------------------------------------------------

def scoping_decks() -> Path:
    """The importer scoping's translated CM1 decks (development repository only)."""

    from dev_paths import dev_path

    return dev_path("scoping-decks")


def scoping_results() -> Path:
    """The scoping's ``check`` results for CM1's 23 shipped decks (development repository only)."""

    return scoping_decks() / "results.json"


def corpus() -> Path:
    """The real-namelist corpus (development repository only)."""

    from dev_paths import dev_path

    return dev_path("namelist-corpus")


def pristine_config_files() -> Path | None:
    """``run/config_files`` of the pinned CM1 r21.1 tree, or None (CM1_R21_1_PRISTINE)."""

    from authority_trees import CM1_PRISTINE

    path = CM1_PRISTINE / "run" / "config_files"
    return path if path.is_dir() else None


def shipped_decks() -> list[str]:
    """CM1 r21.1's 23 shipped config_files decks, by name (from the authority manifest)."""

    names = set()
    for line in (ROOT / "evidence/authority/cm1r21.1-content-sha256.tsv").read_text().splitlines():
        fields = line.split("\t")
        parts = fields[2].split("/") if len(fields) == 3 else []
        if len(parts) == 4 and parts[:2] == ["run", "config_files"] and parts[3] == "namelist.input":
            names.add(parts[2])
    return sorted(names)


def shipped_deck(name: str) -> Path:
    """One shipped deck from the pristine tree, checked against the manifest; skips without the tree."""

    import hashlib

    root = pristine_config_files()
    if root is None:
        pytest.skip("requires CM1 authority tree: set CM1_R21_1_PRISTINE to an extracted cm1r21.1")
    path = root / name / "namelist.input"
    relative = f"run/config_files/{name}/namelist.input"
    manifest = {}
    for line in (ROOT / "evidence/authority/cm1r21.1-content-sha256.tsv").read_text().splitlines():
        fields = line.split("\t")
        if len(fields) == 3:
            manifest[fields[2]] = fields[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[relative], relative
    return path


def _fortran_literal(value: object) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return "'" + value.replace("'", "") + "'"
    if isinstance(value, list):
        return ", ".join(_fortran_literal(item) for item in value)
    return repr(value)


def corpus_namelists(tmp_path: Path) -> list[tuple[str, Path]]:
    """Every corpus entry as a namelist file: the original, or one written from its key summary."""

    import json

    entries = []
    for entry in sorted((corpus() / "entries").iterdir()):
        namelist = entry / "namelist.input"
        if namelist.is_file():
            entries.append((entry.name, namelist))
            continue
        summary = json.loads((entry / "keys.json").read_text(encoding="utf-8"))
        lines = []
        for group, rows in summary["groups"].items():
            lines.append(f" &{group}")
            lines.extend(f"  {key} = {_fortran_literal(value)}," for key, value in rows.items())
            lines.append(" /")
        path = tmp_path / f"{entry.name}.namelist.input"
        path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")
        entries.append((entry.name, path))
    return entries
