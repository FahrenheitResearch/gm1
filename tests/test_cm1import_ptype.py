"""CM1's ptype as a gpuwm-cm1 physics component, pinned.

README.namelist:458-495 is CM1's table: 5 is Morrison, 7 is WSM6, 3 is
Thompson.  A mapping that drifted would run the wrong scheme under a receipt
naming the right one (the harness once mapped WSM6 to ptype 5), so the table,
the component specs, the live admission record and the key map are held
together here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cm1import_support import ROOT, SUPERCELL, edited, namelist_tool
from cm1core.cm1import import PTYPE_COMPONENTS, component_for


def test_the_pinned_ptype_table() -> None:
    assert dict(PTYPE_COMPONENTS) == {
        5: ("morrison-mp10", "runs"),
        7: ("wsm6-mp6", "runs"),
        3: ("thompson-mp8", "acknowledge"),
    }
    assert (component_for(1, 5).component, component_for(1, 5).status) == ("morrison-mp10", "runs")
    assert (component_for(1, 7).component, component_for(1, 7).status) == ("wsm6-mp6", "runs")
    thompson = component_for(1, 3)
    assert (thompson.component, thompson.status) == ("thompson-mp8", "acknowledge")
    assert "120-s column" in thompson.reason


@pytest.mark.parametrize("ptype, scheme", [
    (26, "NSSL 2-moment"), (27, "NSSL 2-moment"), (28, "NSSL single-moment"),
    (1, "Kessler"), (2, "Goddard"), (4, "Gilmore"), (6, "Rotunno-Emanuel"),
    (50, "P3"), (51, "P3"), (52, "P3"), (53, "P3"), (55, "ISHMAEL"), (0, "vapor only"),
])
def test_other_ptypes_are_refused_naming_the_scheme(ptype: int, scheme: str) -> None:
    choice = component_for(1, ptype)
    assert choice.component is None and choice.status == "refused"
    assert scheme in choice.scheme and scheme in choice.reason
    assert f"ptype = {ptype}" in choice.reason


def test_ptype_26_is_refused_naming_nssl() -> None:
    choice = component_for(1, 26)
    assert choice.status == "refused"
    assert "NSSL" in choice.reason


def test_a_ptype_cm1_does_not_have_is_refused_as_such() -> None:
    choice = component_for(1, 99)
    assert choice.status == "refused" and "not a CM1 r21.1 microphysics scheme" in choice.reason


def test_imoist_zero_is_the_off_component_whatever_ptype_says() -> None:
    assert component_for(0, 26).component == "off"


def test_the_table_is_the_component_specs_and_the_live_admission_record() -> None:
    from cm1core.column_components import COLUMN_COMPONENT_SPECS
    from cm1core.oracle.spec import COMPONENT_TO_CM1
    from cm1core.profiles import cartesian_live_admission

    admission = cartesian_live_admission()
    for ptype, (component, _) in PTYPE_COMPONENTS.items():
        assert COLUMN_COMPONENT_SPECS[component].profile.cm1_ptype == ptype
        assert admission["column_components"][component]["cm1_ptype"] == ptype
        assert COMPONENT_TO_CM1[component] == (1, ptype)
        statement = admission["product_differences"][f"native_ptype_{ptype}"]
        assert statement.startswith(f"ptype = {ptype} runs ArWen {component}")


def test_the_key_map_note_names_the_right_schemes() -> None:
    from cm1core.cm1import import keymap

    note = keymap.load().by_cm1("ptype").note
    assert "ptype=5 is morrison-mp10" in note and "7 is wsm6-mp6" in note
    assert "ptype=5 is wsm6" not in note


def test_check_prints_the_live_ptype_statements(capsys) -> None:
    from cm1core.cli import main

    assert main(["check", str(ROOT / "examples" / "cartesian-supercell-morrison-hail.toml")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["product_differences"]["native_ptype_5"].startswith("ptype = 5 runs ArWen morrison-mp10")
    # The sealed identity keeps its text: config_sha256 is taken over it.
    sealed = payload["config"]["profile_identity"]["product_differences"]["native_ptype_5"]
    assert sealed == "substituted by ArWen semantic component wsm6-mp6"
    assert "sealed_identity_text" in payload["product_differences"]


@pytest.mark.parametrize("ptype, component", [(5, "morrison-mp10"), (7, "wsm6-mp6"), (3, "thompson-mp8")])
def test_a_namelist_carries_its_ptype_into_the_document(tmp_path: Path, ptype: int, component: str) -> None:
    from cm1core.cm1import import import_namelist
    from cm1import_support import flat_toml

    deck = edited(SUPERCELL, tmp_path, {"ptype": str(ptype)})
    imported = import_namelist(deck, tool=namelist_tool())
    assert imported.component.component == component
    assert flat_toml(imported.toml())["physics.component"] == component


def test_a_namelist_with_ptype_26_is_not_translated(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    deck = edited(SUPERCELL, tmp_path, {"ptype": "26"})
    imported = import_namelist(deck, tool=namelist_tool())
    assert imported.component.status == "refused" and "NSSL" in imported.component.reason
    assert imported.translation is None and imported.toml() is None
