from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from cm1core.config import load_config, parse_mapping
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]


CASES = (
    pytest.param("run.lane", "unknown", "CM1-CFG-UNSUPPORTED-VALUE", id="SM-RUN-LANE"),
    pytest.param("run.duration_s", -1.0, "CM1-CFG-NUMERIC-RANGE", id="SM-RUN-DURATION"),
    pytest.param("run.dt_s", 0.0, "CM1-CFG-NUMERIC-RANGE", id="SM-RUN-DT"),
    pytest.param("run.history_interval_s", 0.0, "CM1-CFG-NUMERIC-RANGE", id="SM-RUN-HISTORY"),
    pytest.param("run.restart_interval_s", 0.0, "CM1-CFG-NUMERIC-RANGE", id="SM-RUN-RESTART"),
    pytest.param("grid.nx", 6, "CM1-CFG-INTEGER", id="SM-GRID-NX"),
    pytest.param("grid.ny", 6, "CM1-CFG-INTEGER", id="SM-GRID-NY"),
    pytest.param("grid.nz", 6, "CM1-CFG-INTEGER", id="SM-GRID-NZ"),
    pytest.param("grid.dx_m", 0.0, "CM1-CFG-NUMERIC-RANGE", id="SM-GRID-DX"),
    pytest.param("grid.dy_m", 0.0, "CM1-CFG-NUMERIC-RANGE", id="SM-GRID-DY"),
    pytest.param("grid.dz_m", 0.0, "CM1-CFG-NUMERIC-RANGE", id="SM-GRID-DZ"),
    pytest.param("grid.terrain", True, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-GRID-TERRAIN"),
    pytest.param("grid.stretch_x", True, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-GRID-STRETCH-X"),
    pytest.param("grid.stretch_y", True, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-GRID-STRETCH-Y"),
    pytest.param("grid.stretch_z", 5, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-GRID-STRETCH-Z"),
    pytest.param("grid.ztop_m", 15000.0, "CM1-CFG-STRETCH-CONTROL-UNUSED", id="SM-GRID-ZTOP"),
    pytest.param("grid.str_bot_m", 200.0, "CM1-CFG-STRETCH-CONTROL-UNUSED", id="SM-GRID-STR-BOT"),
    pytest.param("grid.str_top_m", 7000.0, "CM1-CFG-STRETCH-CONTROL-UNUSED", id="SM-GRID-STR-TOP"),
    pytest.param("grid.dz_bot_m", 20.0, "CM1-CFG-STRETCH-CONTROL-UNUSED", id="SM-GRID-DZ-BOT"),
    pytest.param("grid.dz_top_m", 320.0, "CM1-CFG-STRETCH-CONTROL-UNUSED", id="SM-GRID-DZ-TOP"),
    pytest.param("grid.levels_m", [10.0, 5.0], "CM1-CFG-LEVEL-ARRAY", id="SM-GRID-LEVELS"),
    pytest.param("dynamics.psolver", 4, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-DYN-PSOLVER"),
    # adapt_dt=1 is now admitted -- 15 of CM1's 23 shipped decks set it.  What
    # is still refused, and is the row this case now exercises, is adapt_dt
    # together with a translating frame: the published domain location's
    # exactness proof (rust/cm1-state/src/domain_location.rs) needs one dt for
    # the whole run, and the official product deck translates.
    pytest.param(
        "dynamics.adaptive_timestep", True, "CM1-CFG-ADAPTIVE-TRANSLATING-FRAME",
        id="SM-DYN-ADAPTIVE",
    ),
    pytest.param("dynamics.scalar_horizontal_order", 3, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-DYN-SCALAR-HORDER"),
    pytest.param("dynamics.scalar_vertical_order", 3, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-DYN-SCALAR-VORDER"),
    pytest.param("dynamics.velocity_horizontal_order", 3, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-DYN-VELOCITY-HORDER"),
    pytest.param("dynamics.velocity_vertical_order", 3, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-DYN-VELOCITY-VORDER"),
    pytest.param("dynamics.scalar_weno", 1, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-DYN-SCALAR-WENO"),
    pytest.param("dynamics.velocity_weno", 2, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-DYN-VELOCITY-WENO"),
    pytest.param("dynamics.weno_order", 3, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-DYN-WENO-ORDER"),
    pytest.param("dynamics.dry_mass_correction", False, "CM1-CFG-PROFILE-DRIFT", id="SM-DYN-DRY-MASS"),
    # CM1's four lateral values (README.namelist:561-568) are all admitted now,
    # in any combination including its own mixed squall-line family, so the
    # named stop is a value CM1 itself does not have.
    pytest.param("boundaries.west", 5, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-BC-WEST"),
    pytest.param("boundaries.east", 0, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-BC-EAST"),
    pytest.param("boundaries.south", 5, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-BC-SOUTH"),
    pytest.param("boundaries.north", 0, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-BC-NORTH"),
    # boundaries.bottom is CM1's bbc; declaring it alone now disagrees with
    # the lower boundary closures.bbc names, which is the more specific stop.
    pytest.param("boundaries.bottom", 2, "CM1-CFG-BBC-DISAGREEMENT", id="SM-BC-BOTTOM"),
    # tbc=2/3 remain refused: this port's stress kernel clamps the top face to
    # free slip with no branch, so a declared lid would produce no drag.
    pytest.param("boundaries.top", 2, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-BC-TOP"),
    # irbc 1, 2 and 4 are admitted; 3 is not a CM1 scheme.
    pytest.param("boundaries.radiative_order", 3, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-BC-RADIATIVE"),
    pytest.param("closures.sgs_model", 2, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-CLOSURE-SGS"),
    pytest.param("closures.vertical_implicit_diffusion", False, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-CLOSURE-IMPLICIT-DIFFUSION"),
    # irdamp 0, 1 and 2 are admitted (src/solve1.F:404-438); 3 is not a branch.
    pytest.param("closures.upper_damping", 3, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-CLOSURE-UPPER-DAMPING"),
    pytest.param("closures.upper_damping_depth_m", 20000.0, "CM1-CFG-DAMPING-DEPTH", id="SM-CLOSURE-DAMPING-DEPTH"),
    pytest.param("closures.upper_damping_rate_s_inv", 0.0, "CM1-CFG-NUMERIC-RANGE", id="SM-CLOSURE-DAMPING-RATE"),
    pytest.param("closures.moving_frame", False, "CM1-CFG-MOVING-FRAME-INACTIVE", id="SM-CLOSURE-MOVING-FRAME"),
    pytest.param("closures.u_move_m_s", math.nan, "CM1-CFG-NUMERIC-RANGE", id="SM-CLOSURE-U-MOVE"),
    pytest.param("closures.v_move_m_s", math.inf, "CM1-CFG-NUMERIC-RANGE", id="SM-CLOSURE-V-MOVE"),
    pytest.param("physics.component", "native-cm1-ptype5", "CM1-CFG-UNSUPPORTED-VALUE", id="SM-PHYSICS-COMPONENT"),
    pytest.param("physics.equation_set", 1, "CM1-CFG-UNSUPPORTED-VALUE", id="SM-PHYSICS-EQUATION-SET"),
)


def product_mapping() -> dict[str, object]:
    payload = load_config(ROOT / "examples" / "official-supercell-product.toml").canonical_dict()
    return {section: payload[section] for section in ("run", "grid", "dynamics", "boundaries", "closures", "physics")}


# CM1's five lower-boundary selectors are optional as a group, so each case
# declares a complete, admitted surface and then breaks exactly one key.
SEMI_SLIP = {"bbc": 3, "sfcmodel": 1, "isfcflx": 0, "cecd": 1, "cnstcd": 0.0035}

LOWER_BOUNDARY_CASES = (
    pytest.param("bbc", 4, "CM1-CFG-BBC-RANGE", id="SM-CLOSURE-BBC"),
    pytest.param("sfcmodel", 2, "CM1-CFG-SFCMODEL-UNSUPPORTED", id="SM-CLOSURE-SFCMODEL"),
    pytest.param("isfcflx", 2, "CM1-CFG-ISFCFLX-RANGE", id="SM-CLOSURE-ISFCFLX"),
    pytest.param("cecd", 4, "CM1-CFG-CECD-RANGE", id="SM-CLOSURE-CECD"),
    pytest.param("cnstcd", 0.0, "CM1-CFG-CNSTCD-RANGE", id="SM-CLOSURE-CNSTCD"),
)


@pytest.mark.parametrize(("key", "bad_value", "expected_code"), LOWER_BOUNDARY_CASES)
def test_named_lower_boundary_refusal(
    key: str, bad_value: object, expected_code: str
) -> None:
    raw = copy.deepcopy(product_mapping())
    raw["closures"].update(SEMI_SLIP)  # type: ignore[index]
    raw["boundaries"]["bottom"] = 3  # type: ignore[index]
    parse_mapping(copy.deepcopy(raw))  # the complete surface is admitted
    raw["closures"][key] = bad_value  # type: ignore[index]
    if key == "bbc":
        raw["boundaries"]["bottom"] = bad_value  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == expected_code
    assert caught.value.detail.option == f"closures.{key}"


def test_the_lower_boundary_documentation_names_the_keys_the_parser_requires() -> None:
    """A reader who follows the doc must not hit CM1-CFG-LOWER-BOUNDARY-PARTIAL.

    The parser requires all five keys or none, so the prose has to say five
    and name the same five.
    """

    import re

    from cm1core.config import _LOWER_BOUNDARY_KEYS

    text = (ROOT / "docs" / "GM1-CARTESIAN-RUNTIME.md").read_text(encoding="utf-8")
    section = text.split("## The lower boundary", 1)[1].split("\n## ", 1)[0]
    counts = {"Three": 0, "Four": 0, "Five": 0}
    for word in counts:
        counts[word] = len(re.findall(rf"\b{word} keys in `\[closures\]`", section))
    assert counts == {"Three": 0, "Four": 0, "Five": 1}
    declared = set(re.findall(r"`(bbc|sfcmodel|isfcflx|cecd|cnstcd)`", section))
    assert declared == set(_LOWER_BOUNDARY_KEYS)
    # and the deck the section points at declares exactly those keys
    deck = load_config(
        ROOT / "examples" / "cartesian-supercell-surface-drag.toml"
    ).canonical_dict()
    assert set(deck["closures"]) >= set(_LOWER_BOUNDARY_KEYS)


def test_a_half_declared_lower_boundary_is_refused_by_the_missing_key() -> None:
    raw = copy.deepcopy(product_mapping())
    raw["closures"]["bbc"] = 3  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-LOWER-BOUNDARY-PARTIAL"
    assert caught.value.detail.option.startswith("closures.")


@pytest.mark.parametrize(("option", "bad_value", "expected_code"), CASES)
def test_named_support_matrix_refusal(
    option: str, bad_value: object, expected_code: str
) -> None:
    raw = copy.deepcopy(product_mapping())
    section, key = option.split(".", 1)
    raw[section][key] = bad_value  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == expected_code
    assert caught.value.detail.option in {option, "boundaries"}
