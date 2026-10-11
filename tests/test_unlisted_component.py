"""thompson-aerosol-mp28 is not offered by the distribution (mp28 hidden).

The Cartesian door refuses it by value unless the developer switch is on, doctor
does not advertise it, and the wheel ships no example deck naming it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cm1core import doctor
from cm1core.cartesian_door import require_listed_component
from cm1core.column_components import UNLISTED_COMPONENTS
from cm1core.config import load_config
from cm1core.errors import CM1Refusal

ROOT = Path(__file__).resolve().parents[1]
MP28_DECK = ROOT / "examples" / "cartesian-supercell-thompson-aerosol.toml"
WSM6_DECK = ROOT / "examples" / "cartesian-supercell-reflectivity.toml"


def test_the_door_refuses_mp28_without_the_developer_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GPUWM_CM1_UNLISTED_MP28", raising=False)
    config = load_config(MP28_DECK)
    assert config.physics.component == "thompson-aerosol-mp28"
    with pytest.raises(CM1Refusal) as refused:
        require_listed_component(config)
    detail = refused.value.detail
    assert detail.code == "CM1-CARTESIAN-UNLISTED-COMPONENT"
    assert detail.option == "physics.component"
    assert "Numerical Recipes" in detail.reason
    assert "thompson-mp8" in detail.remedy


def test_the_developer_switch_readmits_it_and_other_components_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPUWM_CM1_UNLISTED_MP28", "1")
    require_listed_component(load_config(MP28_DECK))
    monkeypatch.delenv("GPUWM_CM1_UNLISTED_MP28", raising=False)
    require_listed_component(load_config(WSM6_DECK))


def test_doctor_does_not_advertise_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GPUWM_CM1_UNLISTED_MP28", raising=False)
    rows = {finding.subject: finding for finding in doctor._column_component_findings()}
    row = rows["column component thompson-aerosol-mp28"]
    assert row.status == doctor.INFO
    assert "not offered in this release" in row.detail
    assert "ArWen mp_physics" not in row.detail
    assert set(UNLISTED_COMPONENTS) == {"thompson-aerosol-mp28"}


def test_the_wheel_build_leaves_its_deck_out() -> None:
    import ast

    # build_rust.py imports setuptools' wheel command, which a test environment
    # need not have; read the constant from its source instead.
    tree = ast.parse((ROOT / "build_rust.py").read_text(encoding="utf-8"))
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "UNSHIPPED_EXAMPLES" for target in node.targets)
    ]
    assert len(values) == 1
    shipped_out = {element.value for element in ast.walk(values[0]) if isinstance(element, ast.Constant)}
    assert MP28_DECK.name in shipped_out
