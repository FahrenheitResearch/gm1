"""The importer's generated records, and their generator.

``tools/generate_cm1_r21_namelist_support.py --records-dir`` writes the three
records from pinned CM1 sources.  Where those trees are present
(``upstream/``, gitignored) the records are regenerated and must be
byte-identical; everywhere the facts they state are checked directly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cm1import_support import ROOT, RUN_DEFAULT

RECORDS = ROOT / "src" / "cm1core" / "records"
UPSTREAM = ROOT / "upstream"
TREES = ("cm1-21.1/pristine", "cm1-22.0", "cm1-20.3", "cm1-19.10", "cm1-18", "cm1-17", "cm1-16")


def _record(name: str) -> dict:
    return json.loads((RECORDS / name).read_text(encoding="utf-8"))


def test_defaults_cover_every_r21_field() -> None:
    record = _record("CM1-R21-NAMELIST-DEFAULTS.json")
    assert record["schema"] == "gm1.cm1-r21.1-namelist-defaults/v1"
    assert record["counts"]["fields"] == 583 and record["counts"]["groups"] == 25
    assert record["counts"]["cm1-assignment"] + record["counts"]["declaration-initializer"] + record["counts"]["static-zero"] == 583
    support = json.loads((ROOT / "rust/cm1-state/schema/cm1-r21.1-namelist-support.json").read_text())
    inventory = {group["name"]: {f["name"]: f["type"] for f in group["fields"]} for group in support["groups"]}
    assert {g: {k: row["type"] for k, row in rows.items()} for g, rows in record["groups"].items()} == inventory
    assert record["required_groups"]["groups"] == ["param0", "param2", "param8"]


def test_defaults_state_what_cm1_r21_does_with_an_unset_key() -> None:
    from cm1core.cm1import import defaults

    table = defaults.load()
    ramp = table.field("param12", "ramp_sgs")
    assert (ramp.code_default, ramp.basis, ramp.source) == (0, "cm1-assignment", "src/cm1.F:355")
    assert ramp.distributed_value == 1
    alph = table.field("param3", "alph")
    assert (alph.code_default, alph.basis, alph.distributed_value) == (0.0, "static-zero", 0.6)
    iorigin = table.field("param2", "iorigin")
    assert (iorigin.code_default, iorigin.basis, iorigin.distributed_value) == (0, "static-zero", 2)
    assert table.field("param3", "alphobc").code_default == 60.0
    assert table.field("logging", "stderr_logging").code_default == 0


def test_the_distributed_values_are_r21s_run_namelist() -> None:
    from cm1core.cm1import import defaults
    from cm1core.oracle import namelist as oracle_namelist

    shipped = {}
    for fields in oracle_namelist.read(RUN_DEFAULT).values():
        shipped.update(fields)
    distributed = defaults.load().distributed_flat()
    assert set(distributed) == set(shipped)
    for key, value in shipped.items():
        assert distributed[key] == value, key


def test_the_r22_and_alias_records_cite_their_sources() -> None:
    r22 = _record("CM1-R22-NEW-KEYS.json")
    assert r22["release"]["tarball_sha256"] == "05f990bb79055873c49499c92d0ff447524a2795016a474447a7e87dd5a86226"
    for rows in r22["keys"].values():
        for key, row in rows.items():
            assert row["sources"], key
            assert row["class"] in r22["classes"], key
            assert row["r21_equivalent"]["kind"] in r22["equivalent_kinds"], key
    aliases = _record("CM1-KEY-ALIASES.json")
    assert aliases["counts"]["keys"] == 44
    for key, row in aliases["aliases"].items():
        assert row["sources"] and row["releases"], key
        assert row["kind"] in aliases["kinds"], key


@pytest.mark.skipif(
    not all((UPSTREAM / tree).is_dir() for tree in TREES),
    reason="regenerating needs the pinned CM1 trees under upstream/ (see upstream/README.md)",
)
def test_the_generator_reproduces_the_committed_records(tmp_path: Path) -> None:
    generator = ROOT / "tools" / "generate_cm1_r21_namelist_support.py"
    subprocess.run(
        [sys.executable, str(generator), "--authority-root", str(UPSTREAM / "cm1-21.1/pristine"),
         "--output", str(tmp_path / "support.json"), "--records-dir", str(tmp_path)],
        check=True, capture_output=True,
    )
    assert (tmp_path / "support.json").read_bytes() == (
        ROOT / "rust/cm1-state/schema/cm1-r21.1-namelist-support.json"
    ).read_bytes()
    for name in ("CM1-R21-NAMELIST-DEFAULTS.json", "CM1-R22-NEW-KEYS.json", "CM1-KEY-ALIASES.json"):
        assert (tmp_path / name).read_bytes() == (RECORDS / name).read_bytes(), name
