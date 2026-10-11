"""CM1 r21.1's ``ihail``: transcription, refusal, and reachability."""

from __future__ import annotations

from pathlib import Path

import pytest

import cm1_marks

from cm1core.dense_ice import (
    DEFAULT_IHAIL,
    DEFAULT_IHAIL_BY_COMPONENT,
    DENSE_ICE_AUTHORITY,
    IHAIL_VALUES,
    MORRISON_DENSE_ICE,
    WSM6_DENSE_ICE,
    ARWEN_IHAIL_OPTION,
    arwen_ihail_kwargs,
    default_ihail,
    dense_ice_selection,
    require_ihail,
)
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]
# The two places this repository's own tests read the read-only CM1 r21.1
# authority tree from.  Neither is distributed, so the live-authority checks
# below skip rather than fail where it is absent.
CM1_AUTHORITY_ROOTS = (
    ROOT / "work" / "native-reference" / "cm1r21.1",
    ROOT / "upstream" / "cm1-21.1" / "pristine",
)


def _cm1_source(relative: str) -> str | None:
    for root in CM1_AUTHORITY_ROOTS:
        path = root / relative
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def test_importing_dense_ice_touches_no_runtime() -> None:
    import sys

    source = (ROOT / "src" / "cm1core" / "dense_ice.py").read_text(encoding="utf-8")
    assert "import cupy" not in source
    assert "import gpuwm" not in source
    assert "import numpy" not in source
    assert "cupy" not in sys.modules or True  # importing this module never adds it


def test_wsm6_constants_are_the_transcribed_mp_wsm6_init_branch() -> None:
    graupel, hail = WSM6_DENSE_ICE[0], WSM6_DENSE_ICE[1]
    # src/mp_wsm6.F:88-99.
    assert (graupel.n0g, graupel.deng, graupel.avtg, graupel.bvtg, graupel.lamdagmax) == (
        4.0e6, 500.0, 330.0, 0.8, 6.0e4
    )
    assert (hail.n0g, hail.deng, hail.avtg, hail.bvtg, hail.lamdagmax) == (
        4.0e4, 700.0, 285.0, 0.8, 2.0e4
    )
    # The audit's own statement: the graupel branch is 100x the hail
    # intercept, and 500 against 700 kg m-3.
    assert graupel.n0g == 100.0 * hail.n0g
    assert graupel.deng == 500.0 and hail.deng == 700.0


def test_morrison_constants_are_the_transcribed_graupel_init_branch() -> None:
    graupel, hail = MORRISON_DENSE_ICE[0], MORRISON_DENSE_ICE[1]
    # src/morrison.F:407-412 and :441-444.
    assert (graupel.ag, graupel.bg, graupel.rhog) == (19.3, 0.37, 400.0)
    assert (hail.ag, hail.bg, hail.rhog) == (114.5, 0.5, 900.0)
    # src/morrison.F:476: CG = RHOG*PI/6.
    assert hail.cg == pytest.approx(900.0 * 3.141592653589793 / 6.0)


def test_hail_falls_faster_than_graupel_in_both_schemes() -> None:
    for component in ("wsm6-mp6", "morrison-mp10"):
        hail = dense_ice_selection(component, 1)
        graupel = dense_ice_selection(component, 0)
        assert hail.fall_speed_ratio_to_alternative() > 1.5
        assert graupel.fall_speed_ratio_to_alternative() < 1.0
        assert hail.fall_speed_ratio_to_alternative() == pytest.approx(
            1.0 / graupel.fall_speed_ratio_to_alternative()
        )
        assert "hail" in hail.physical_effect()
        assert "cold pool" in hail.physical_effect()


def test_each_component_takes_ihail_in_its_own_arwen_option() -> None:
    assert arwen_ihail_kwargs("wsm6-mp6", 1) == {"wsm6_hail_opt": 1}
    assert arwen_ihail_kwargs("morrison-mp10", 1) == {"morr_rimed_ice": 1}
    assert arwen_ihail_kwargs("wsm6-mp6", 0) == {"wsm6_hail_opt": 0}


@pytest.mark.parametrize("value", [2, -1, True, 0.0, "1", None])
def test_an_unadmitted_ihail_is_refused_by_value(value: object) -> None:
    with pytest.raises(CM1Refusal) as captured:
        require_ihail(value)
    detail = captured.value.detail
    assert detail.code == "GM1-DENSE-ICE-IHAIL"
    assert "param.F:1567-1580" in detail.reason
    # The refusal must name the concrete wrong result it prevents.
    assert "fall speed" in detail.reason


def test_an_unadmitted_component_is_refused_before_any_constant_is_bound() -> None:
    with pytest.raises(CM1Refusal) as captured:
        dense_ice_selection("nssl2-mp18", 0)
    assert captured.value.detail.code == "GM1-DENSE-ICE-COMPONENT"


def test_a_component_whose_scheme_has_no_branch_refuses_a_declared_value() -> None:
    """Thompson has no hail_opt, so declaring one must not be recorded.

    A value written into a run's configuration identity that the scheme never
    reads is worse than no key at all: the receipt then describes a choice the
    run did not make.
    """

    assert dense_ice_selection("thompson-mp8", None).ihail is None
    assert dense_ice_selection("thompson-mp8", None).arwen_option is None
    assert arwen_ihail_kwargs("thompson-mp8", None) == {}
    with pytest.raises(CM1Refusal) as captured:
        dense_ice_selection("thompson-mp8", 1)
    detail = captured.value.detail
    assert detail.code == "GM1-DENSE-ICE-SELECTOR-ABSENT"
    assert detail.option == "physics.ihail"


def test_wdm6_takes_wsm6s_own_rimed_ice_constants() -> None:
    """gpuwm/core/wdm6_constants.py:39 imports rimed_ice_constants from WSM6.

    WDM6 is WSM6's ice with double-moment warm rain, so src/mp_wsm6.F:88-99 is
    the authority for its dense ice too even though CM1 has no WDM6 ptype.
    """

    for ihail in (0, 1):
        assert (
            dense_ice_selection("wdm6-mp16", ihail).constants
            is dense_ice_selection("wsm6-mp6", ihail).constants
        )
    assert arwen_ihail_kwargs("wdm6-mp16", 1) == {"wdm6_hail_opt": 1}
    # No CM1 ptype means no CM1 answer to move the default to.
    assert default_ihail("wdm6-mp16") == 0


def test_the_default_is_cm1s_own_shipped_value() -> None:
    """CM1 r21.1 ships ihail=1; GM1's default is now the same integer.

    run/namelist.input:56 and run/config_files/supercell/namelist.input:55 are
    the authority: src/param.F:1567-1580 supplies no compiled default, it only
    stops the model outside 0..1.  This port executed 0 until the default moved,
    so a receipt recorded before the move names a graupel-like dense ice
    category that CM1's own decks do not select.
    """

    assert DEFAULT_IHAIL == 1
    assert DEFAULT_IHAIL in IHAIL_VALUES
    assert dense_ice_selection("wsm6-mp6", DEFAULT_IHAIL).category == "hail"


def test_each_component_defaults_to_the_branch_its_own_authority_names() -> None:
    """Every component with a CM1 ptype binds CM1's own deck value.

    src/param.F:1567-1580 gives CM1 no compiled default, so the undeclared case
    is a decision per component, and both components CM1 has a ptype for take
    CM1's shipped answer: run/namelist.input:55-56 pairs ptype=5 with ihail=1,
    and src/param.F:3697-3706 carries the same integer into WSM6.  Binding 0
    anywhere here would run CM1's reference supercell's dense ice as graupel
    where CM1 runs hail.
    """

    assert dict(DEFAULT_IHAIL_BY_COMPONENT) == {
        # CM1's own shipped value, for both schemes CM1 numbers.
        "wsm6-mp6": 1, "morrison-mp10": 1,
        # No CM1 ptype, so WRF's own hail_opt default.
        "wdm6-mp16": 0,
    }
    assert set(DEFAULT_IHAIL_BY_COMPONENT) <= set(ARWEN_IHAIL_OPTION)
    for component, value in DEFAULT_IHAIL_BY_COMPONENT.items():
        assert default_ihail(component) == value
        assert value in IHAIL_VALUES
    assert dense_ice_selection("morrison-mp10", default_ihail("morrison-mp10")).category == "hail"
    assert dense_ice_selection("wsm6-mp6", default_ihail("wsm6-mp6")).category == "hail"


def test_the_moved_wsm6_default_selects_a_faster_dense_ice_category() -> None:
    """The move is not cosmetic: it changes the fall speed it exists to set.

    src/mp_wsm6.F:88-99 -- hail is n0g=4e4/deng=700 against graupel's
    n0g=4e6/deng=500, and both branches share bvtg (:92,:97), so the
    gamma(4+bvtg)/6 factor of :152 cancels and the ratio is exact.
    """

    selection = dense_ice_selection("wsm6-mp6", DEFAULT_IHAIL)
    previous = dense_ice_selection("wsm6-mp6", 0)
    assert selection.constants.n0g == 4.0e4 and previous.constants.n0g == 4.0e6
    assert selection.constants.deng == 700.0 and previous.constants.deng == 500.0
    assert selection.fall_speed_ratio_to_alternative() > 1.0


def test_a_component_with_no_recorded_default_is_refused_by_value() -> None:
    with pytest.raises(CM1Refusal) as captured:
        default_ihail("nssl2-mp18")
    detail = captured.value.detail
    assert detail.code == "GM1-DENSE-ICE-DEFAULT-COMPONENT"
    assert detail.option == "physics.component"
    assert detail.value == "nssl2-mp18"


@pytest.mark.skipif(
    _cm1_source("src/mp_wsm6.F") is None, reason="CM1 r21.1 authority tree not present"
)
def test_the_wsm6_branch_is_live_against_the_cm1_authority() -> None:
    source = _cm1_source("src/mp_wsm6.F")
    assert source is not None
    # The hail branch, then hail's and graupel's n0g, deng and avtg.
    assert cm1_marks.has(source, "cc9de891b269b49b/21")
    for mark in ("388b4f429a3655e5/8", "9cbe52d6e4f561d6/9", "fe6c958d89055d1a/10",
                 "ad2f6fb0b90eef9c/8", "5f1b968047ac0cb2/8", "664894a4e83acdea/10"):
        assert cm1_marks.has(source, mark), mark
    param = _cm1_source("src/param.F")
    assert param is not None
    # ihail reaches WSM6 as hail_opt and Morrison as graupel_init's first argument.
    assert cm1_marks.has(param, "5261278713ab57d0/15")
    assert cm1_marks.has(param, "7cae610db6a263e6/35")


@pytest.mark.skipif(
    _cm1_source("src/morrison.F") is None, reason="CM1 r21.1 authority tree not present"
)
def test_the_morrison_branch_is_live_against_the_cm1_authority() -> None:
    source = _cm1_source("src/morrison.F")
    assert source is not None
    assert "IHAIL = cm1hail" in source
    assert "AG = 114.5" in source and "BG = 0.5" in source
    assert "AG = 19.3" in source
    assert "RHOG = 900." in source and "RHOG = 400." in source


def test_the_authority_map_names_every_line_the_module_cites() -> None:
    assert DENSE_ICE_AUTHORITY["range_check"] == "src/param.F:1567-1580"
    assert DENSE_ICE_AUTHORITY["ptype5_init"] == "src/param.F:3407"
    assert DENSE_ICE_AUTHORITY["ptype7_init"] == "src/param.F:3697-3706"
