"""What CM1 r21.1 itself stops on before it reads a physics key.

cm1.F:480-482 reads ``&param0``, ``&param2`` and ``&param8`` one after
another, with no ``end=`` branch and no rewind between them, before param.F
reads the other groups (each after a rewind, each with ``end=``).  A file
without one of the three, or with them in another order, ends that read at
the end of the file and CM1 stops.  The importer says so and does not
translate the file, rather than running a configuration CM1 would not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cm1import_support import RUN_DEFAULT, SQUALL_LINE, SUPERCELL, namelist_tool


def _groups(source: Path) -> dict[str, str]:
    """The file's groups as text blocks, keyed by name, in file order."""

    text = source.read_text(encoding="ascii")
    blocks: dict[str, str] = {}
    for chunk in text.split("&")[1:]:
        name = chunk.split(None, 1)[0].lower()
        blocks[name] = "&" + chunk
    return blocks


def _write(tmp_path: Path, blocks: list[str]) -> Path:
    path = tmp_path / "namelist.input"
    path.write_text("".join(blocks), encoding="ascii")
    return path


@pytest.mark.parametrize("deck", [SUPERCELL, SQUALL_LINE, RUN_DEFAULT], ids=lambda path: path.name)
def test_cm1s_own_decks_start(deck: Path) -> None:
    from cm1core.cm1import import import_namelist

    imported = import_namelist(deck, tool=namelist_tool())
    assert imported.cm1_stops == ()
    assert imported.parsed.group_order[:3] == ("param0", "param1", "param2")
    assert imported.translation is not None


def test_a_file_without_param8_is_one_cm1_stops_on(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    blocks = _groups(SUPERCELL)
    deck = _write(tmp_path, [text for name, text in blocks.items() if name != "param8"])
    imported = import_namelist(deck, tool=namelist_tool())
    [stop] = imported.cm1_stops
    assert "cm1.F:480-482" in stop and "has no &param8" in stop
    assert imported.translation is None and imported.toml() is None


def test_param2_before_param0_is_one_cm1_stops_on(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    blocks = _groups(SUPERCELL)
    order = ["param2"] + [name for name in blocks if name != "param2"]
    deck = _write(tmp_path, [blocks[name] for name in order])
    imported = import_namelist(deck, tool=namelist_tool())
    [stop] = imported.cm1_stops
    assert "in that order without rewinding" in stop
    assert stop.endswith("opens them as &param2, &param0, &param8")
    assert imported.translation is None


def test_the_other_groups_may_come_in_any_order(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist
    from cm1import_support import flat_toml

    blocks = _groups(SUPERCELL)
    others = [name for name in blocks if name not in ("param0", "param2", "param8")]
    order = list(reversed(others[:3])) + ["param0", "param2"] + others[3:] + ["param8"]
    deck = _write(tmp_path, [blocks[name] for name in order])
    imported = import_namelist(deck, tool=namelist_tool())
    assert imported.cm1_stops == ()
    reference = import_namelist(SUPERCELL, tool=namelist_tool())
    assert flat_toml(imported.toml()) == flat_toml(reference.toml())


def test_an_unset_large_step_is_refused_by_key(tmp_path: Path) -> None:
    # dtl is static storage with no initializer in r21.1: a file that never
    # sets it runs CM1 with dtl = 0.  The translator names the keys.
    from cm1core.cm1import import import_namelist
    from cm1core.cm1import.translate import TranslationError

    deck = _write(tmp_path, [" &param0\n nx = 40, ny = 40, nz = 40\n /\n",
                             " &param2\n imoist = 1, ptype = 5\n /\n", " &param8\n /\n"])
    with pytest.raises(TranslationError, match=r"dtl = 0\.0"):
        import_namelist(deck, tool=namelist_tool())
