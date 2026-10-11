"""CM1 r22.0 keys: each with its class and its r21.1 equivalent.

CM1 r22.0 (21 Sep 2026) adds keys an r21.1 parser would drop silently.  Each
now reads as one of: the same as r21.1 (r22.0 does not read it with the
file's settings, or reads it at the value r21.1 hard-codes), or different,
with its class saying whether that changes the physics or only the output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cm1import_support import R22_SUPERCELL, edited, namelist_tool


@pytest.fixture(scope="module")
def r22_supercell():
    from cm1core.cm1import import import_namelist

    return import_namelist(R22_SUPERCELL, tool=namelist_tool())


def test_r22_supercell_is_detected_and_every_new_key_classified(r22_supercell) -> None:
    imported = r22_supercell
    assert imported.version == "r22.0"
    assert imported.unknown == ()
    new_keys = [reading for reading in imported.r22 if reading.key != "output_vort"]
    # r22.0's supercell deck sets 31 keys r21.1 does not declare.  (It also
    # sets outunits and betaplane, which are r21.1 keys r21.1's own supercell
    # deck omits.)
    assert len(new_keys) == 31
    assert {reading.klass for reading in new_keys} <= {"physics", "output", "cosmetic"}
    differs = {(reading.key, reading.klass) for reading in imported.r22 if reading.status == "differs"}
    # r22.0's supercell sets the Morrison breakup diameter to 400e-6 m; r21.1
    # hard-codes 300e-6 m (morrison.F:493).  And output_vort now means
    # staggered vorticity.  Everything else is r21.1's behaviour.
    assert differs == {("brkup_db", "physics"), ("output_vort", "output")}
    same = {reading.key for reading in imported.r22 if reading.status == "same"}
    assert {"cfl_limit", "ramp_down", "trn_swirl", "gamma_cp", "cnstch", "output_z_max", "output_hail"} <= same


def test_r22_changes_of_meaning_are_applied(r22_supercell) -> None:
    imported = r22_supercell
    # r21.1's output_vort is r22.0's output_vortinterp.
    assert imported.effective["output_vort"] == 1
    assert "output_vortinterp" in imported.r22_settings["output_vort"]
    # cnstch = cnstcq = 0.001 is r21.1's cnstce.
    assert imported.effective["cnstce"] == pytest.approx(0.001)


def test_r22_supercell_translates_and_the_door_admits_it(r22_supercell, tmp_path: Path) -> None:
    from cm1core.config import load_config

    path = tmp_path / "deck.toml"
    path.write_text(r22_supercell.toml(), encoding="utf-8")
    load_config(path)


@pytest.mark.parametrize("adapt_dt, status", [(1, "differs"), (0, "same")])
def test_a_physics_key_differs_only_where_r22_reads_it(tmp_path: Path, adapt_dt: int, status: str) -> None:
    from cm1core.cm1import import import_namelist

    deck = edited(R22_SUPERCELL, tmp_path, {"cfl_limit": "1.5", "adapt_dt": str(adapt_dt)})
    readings = {reading.key: reading for reading in import_namelist(deck, tool=namelist_tool()).r22}
    assert readings["cfl_limit"].status == status
    assert readings["cfl_limit"].klass == "physics"
    assert readings["cfl_limit"].read_by_r22 is bool(adapt_dt)


def test_an_output_key_changed_is_an_output_difference(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    deck = edited(R22_SUPERCELL, tmp_path, {"output_temperature": "1"})
    readings = {reading.key: reading for reading in import_namelist(deck, tool=namelist_tool()).r22}
    assert (readings["output_temperature"].status, readings["output_temperature"].klass) == ("differs", "output")


def test_brkup_db_at_r21s_value_is_the_same(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    for value in ("300.0e-6", "0.0"):
        deck = edited(R22_SUPERCELL, tmp_path, {"brkup_db": value})
        readings = {reading.key: reading for reading in import_namelist(deck, tool=namelist_tool()).r22}
        assert readings["brkup_db"].status == "same", value


def test_unequal_cnstch_and_cnstcq_have_no_r21_cnstce(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    deck = edited(R22_SUPERCELL, tmp_path, {"cnstch": "0.0012", "isfcflx": "1", "cecd": "1"})
    imported = import_namelist(deck, tool=namelist_tool())
    readings = {reading.key: reading for reading in imported.r22}
    assert readings["cnstch"].status == "differs"
    assert "cnstce" not in imported.r22_settings


def test_the_record_holds_every_r22_key() -> None:
    from cm1core.cm1import import versions

    table = versions.load()
    assert len(table.keys) == 64
    outside_nssl = [row for (group, _), row in table.keys.items() if group != "nssl_mp_params"]
    assert len(outside_nssl) == 31
    assert set(table.changed_meaning) == {"output_vort", "irandp"}
    assert set(table.removed_r21_keys) == {"cnstce"}
