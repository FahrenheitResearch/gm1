"""Keys from CM1 releases before r21.1, read as their r21.1 settings.

``pertcor`` was renamed ``lspgrad`` in CM1 r19 (CHANGES r19.10:120-127;
README.namelist:535 "lspgrad = 1 was called pertcor").  A user's r18 deck
with ``pertcor = 1`` asks for exactly what an r21.1 deck with ``lspgrad = 1``
asks for, so it must run that way and say so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cm1import_support import R18_RUN, SUPERCELL, flat_toml, namelist_tool


def _with_lines(source: Path, tmp_path: Path, group: str, lines: list[str]) -> Path:
    text = source.read_text(encoding="ascii")
    marker = f"&{group}"
    index = text.index(marker) + len(marker)
    out = tmp_path / "namelist.input"
    out.write_text(text[:index] + "\n" + "\n".join(f" {line}," for line in lines) + text[index:], encoding="ascii")
    return out


def _without(source: Path, tmp_path: Path, *keys: str) -> Path:
    lines = [line for line in source.read_text(encoding="ascii").splitlines() if line.split("=")[0].strip() not in keys]
    out = tmp_path / "base.input"
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out


def test_pertcor_1_reads_as_lspgrad_1_and_is_reported(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    base = _without(SUPERCELL, tmp_path, "lspgrad", "icor")
    deck = _with_lines(base, tmp_path, "param2", ["pertcor = 1", "icor = 1"])
    imported = import_namelist(deck, tool=namelist_tool())
    [reading] = imported.aliases
    assert (reading.key, reading.status, dict(reading.assigns)) == ("pertcor", "read", {"lspgrad": 1})
    assert reading.message == "pertcor = 1 read as lspgrad = 1 (renamed in CM1 r19)"
    assert imported.effective["lspgrad"] == 1
    assert "lspgrad" in imported.stated
    assert imported.version == "before r21.1"
    assert imported.unknown == ()
    document = flat_toml(imported.toml())
    assert document["dynamics.lspgrad"] == 1
    carried = {row["cm1_key"]: row for row in imported.translation.carried}
    assert carried["lspgrad"]["gm1_key"] == "dynamics.lspgrad"


def test_pertcor_against_an_explicit_lspgrad_is_a_conflict(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    deck = _with_lines(SUPERCELL, tmp_path, "param2", ["pertcor = 1"])
    imported = import_namelist(deck, tool=namelist_tool())
    [reading] = imported.aliases
    assert reading.status == "conflict"
    assert "the file also sets lspgrad = 0" in reading.message
    assert imported.effective["lspgrad"] == 0


def test_pertcor_equal_to_an_explicit_lspgrad_is_no_conflict(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    deck = _with_lines(SUPERCELL, tmp_path, "param2", ["pertcor = 0"])
    [reading] = import_namelist(deck, tool=namelist_tool()).aliases
    assert reading.status == "read"


def test_cm1_r18_run_namelist_reads_as_r21(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    imported = import_namelist(R18_RUN, tool=namelist_tool())
    assert imported.version == "before r21.1"
    assert imported.unknown == ()
    status = {reading.key: (reading.status, dict(reading.assigns)) for reading in imported.aliases}
    assert status["iturb"] == ("read", {"cm1setup": 1, "sgsmodel": 1})
    assert status["pertcor"] == ("read", {"lspgrad": 0})
    assert status["ipbl"] == ("read", {"ipbl": 0})
    assert status["restart_format"] == ("read", {"restart_format": 1})
    assert status["pdscheme"][0] == "same"
    assert status["nsound"][0] == "ignored"
    assert status["dns"][0] == "same"
    assert status["nodex"][0] == "removed" and status["output_path"][0] == "removed"
    assert imported.effective["cm1setup"] == 1 and imported.effective["sgsmodel"] == 1
    klass = {reading.key: reading.klass for reading in imported.aliases}
    assert klass["nodex"] == "not-needed" and klass["output_path"] == "output"


@pytest.mark.parametrize("line, status, fragment", [
    ("iturb = 3", "unmapped", "only partly documented"),
    ("iturb = 2", "read", "cm1setup = 1, sgsmodel = 2"),
    ("dns = 1", "read", "cm1setup = 3"),
    ("pdscheme = 0", "unmapped", "has no r21.1 setting"),
    ("neweqts = 0", "read", "eqtset = 1"),
    ("hadvorder = 5", "read", "hadvordrs = 5, hadvordrv = 5"),
    ("advweno = 2", "read", "advwenos = 2, advwenov = 0"),
])
def test_value_mapped_old_keys(tmp_path: Path, line: str, status: str, fragment: str) -> None:
    from cm1core.cm1import import aliases
    from cm1core.cm1import.namelist import UnknownAssignment

    key, value = (part.strip() for part in line.split("="))
    unknown = UnknownAssignment(group="param2", key=key, group_declared=True, text=value, value=int(value), value_type="integer")
    [reading], rest = aliases.resolve([unknown], {})
    assert rest == ()
    assert reading.status == status and fragment in reading.message


def test_two_old_keys_that_disagree_conflict() -> None:
    from cm1core.cm1import import aliases
    from cm1core.cm1import.namelist import UnknownAssignment

    rows = [
        UnknownAssignment(group="param2", key="iturb", group_declared=True, text="1", value=1, value_type="integer"),
        UnknownAssignment(group="param2", key="dns", group_declared=True, text="1", value=1, value_type="integer"),
    ]
    readings, _ = aliases.resolve(rows, {})
    assert [r.status for r in readings] == ["read", "conflict"]
    assert "iturb = 1 reads as cm1setup = 1" in readings[1].message


def test_every_alias_target_is_an_r21_field() -> None:
    from cm1core.cm1import import aliases, defaults

    table = defaults.load()
    for row in aliases.load().rows.values():
        for target in row.r21_targets:
            assert table.groups_of(target), (row.key, target)
        for mapping in row.values.values():
            for target in mapping:
                assert table.groups_of(target), (row.key, target)
