"""CM1's ``q3d`` scalar inventories and their reachability from a document."""

from __future__ import annotations

from pathlib import Path

import pytest

import cm1_marks

from cm1core.errors import CM1Refusal
from cm1core.moisture_species import (
    DEFAULT_COMPONENT_ID,
    MOISTURE_COMPONENT_IDS,
    MORRISON_PROFILE,
    WSM6_PROFILE,
    moisture_scalar_count,
    moisture_species_profile,
    require_equation_set,
)


ROOT = Path(__file__).resolve().parents[1]
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


def test_the_wsm6_axis_is_cm1_ptype_seven() -> None:
    assert WSM6_PROFILE.cm1_ptype == 7
    assert WSM6_PROFILE.numq == 6
    assert WSM6_PROFILE.names == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert WSM6_PROFILE.number_names == ()
    assert WSM6_PROFILE.qmag == (1.0e-2,) * 6
    assert (WSM6_PROFILE.nqv, WSM6_PROFILE.nql1, WSM6_PROFILE.nql2) == (1, 2, 3)
    assert (WSM6_PROFILE.nqs1, WSM6_PROFILE.nqs2) == (4, 6)
    assert (WSM6_PROFILE.nnc1, WSM6_PROFILE.nnc2) == (-2, -2)
    assert WSM6_PROFILE.repaired_scalars == 6
    assert WSM6_PROFILE.vapour_index == 0
    assert WSM6_PROFILE.condensate_indices == (1, 2, 3, 4, 5)


def test_the_morrison_axis_is_cm1_ptype_five_with_inum_one() -> None:
    assert MORRISON_PROFILE.cm1_ptype == 5
    assert MORRISON_PROFILE.numq == 10
    assert MORRISON_PROFILE.names == (
        "qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg"
    )
    assert MORRISON_PROFILE.mass_names == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert MORRISON_PROFILE.number_names == ("nci", "ncs", "ncr", "ncg")
    # src/param.F:3464-3473: the number rows carry a different qmag, which is
    # what scales the WENO positive-definite epsilon at src/solve2.F:1472.
    assert MORRISON_PROFILE.qmag == (1.0e-2,) * 6 + (1.0e6,) * 4
    assert (MORRISON_PROFILE.nnc1, MORRISON_PROFILE.nnc2) == (7, 10)
    assert (MORRISON_PROFILE.iice, MORRISON_PROFILE.idm) == (1, 1)
    # src/mp_driver.F:505-520: the asq(7) call is commented out, so the
    # positive-definite repair stops after the mass prefix.
    assert MORRISON_PROFILE.repaired_scalars == 6
    assert MORRISON_PROFILE.number_indices == (6, 7, 8, 9)
    assert MORRISON_PROFILE.units[6:] == ("#/kg",) * 4


def test_the_mass_prefix_is_shared_so_a_prefix_view_is_contiguous() -> None:
    # Every consumer that hands CM1's mass-only operators a leading-axis
    # window depends on this: calcnm, pdefq and the post-physics EOS.  The
    # post-physics EOS/qa kernel is compiled for exactly six mass rows, so a
    # bound component whose prefix is not CM1's six has no publication path
    # and column_component_blockers says so by name.
    from cm1core.moisture_species import _PROFILES

    for profile in _PROFILES.values():
        assert profile.names[:6] == WSM6_PROFILE.names, profile.component_id
        assert profile.mass_indices == tuple(range(6)), profile.component_id


def test_a_scheme_cm1_has_no_ptype_for_records_that_rather_than_guessing() -> None:
    """WDM6 is bound and CM1 numbers no such scheme; the axis says where it is from."""

    from cm1core.moisture_species import (
        TRANSCRIBED_UNBOUND_PROFILES,
        THOMPSON_PROFILE,
        WDM6_PROFILE,
    )

    assert THOMPSON_PROFILE.cm1_ptype == 3
    assert THOMPSON_PROFILE.record()["cm1_scheme_present"] is True
    assert THOMPSON_PROFILE.axis_authority == "src/param.F:3322-3398"
    assert THOMPSON_PROFILE.names == (
        "qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncr"
    )
    # WDM6's axis is transcribed and its scheme is not selectable: ArWen's
    # WDM6 refuses without WRF's XLAND mask, which this lane has no field for.
    assert WDM6_PROFILE.cm1_ptype is None
    assert WDM6_PROFILE.record()["cm1_scheme_present"] is False
    assert "CM1 r21.1 has no WDM6 ptype" in WDM6_PROFILE.axis_authority
    assert set(TRANSCRIBED_UNBOUND_PROFILES) == {"wdm6-mp16"}
    assert set(TRANSCRIBED_UNBOUND_PROFILES).isdisjoint(MOISTURE_COMPONENT_IDS)


def test_a_blocked_scheme_is_refused_by_naming_its_one_blocker() -> None:
    """"Unsupported" would be a lie: ArWen compiles these and GM1 measured them."""

    from cm1core.column_component_blockers import COLUMN_COMPONENT_BLOCKERS

    for component_id in COLUMN_COMPONENT_BLOCKERS:
        with pytest.raises(CM1Refusal) as captured:
            moisture_species_profile(component_id)
        detail = captured.value.detail
        assert detail.code == "GM1-COLUMN-COMPONENT-BLOCKED", component_id
        assert COLUMN_COMPONENT_BLOCKERS[component_id].blocker in detail.reason


def test_an_unadmitted_component_is_refused_by_key_and_value() -> None:
    with pytest.raises(CM1Refusal) as captured:
        moisture_species_profile("goddard-lfo")
    detail = captured.value.detail
    assert detail.code == "GM1-MOISTURE-COMPONENT-UNSUPPORTED"
    assert "how many scalars to transport" in detail.reason
    with pytest.raises(CM1Refusal) as captured:
        moisture_species_profile(5)
    assert captured.value.detail.code == "GM1-MOISTURE-COMPONENT-TYPE"


def test_only_the_equation_set_the_large_step_executes_is_admitted() -> None:
    assert require_equation_set("wsm6-mp6", 2) == 7
    assert require_equation_set("morrison-mp10", 2) == 5
    with pytest.raises(CM1Refusal) as captured:
        require_equation_set("morrison-mp10", 1)
    assert captured.value.detail.code == "GM1-MOISTURE-EQUATION-SET"


def test_the_inventory_is_stable_and_wsm6_is_the_default() -> None:
    assert MOISTURE_COMPONENT_IDS == (
        "wsm6-mp6", "morrison-mp10", "thompson-mp8", "thompson-aerosol-mp28"
    )
    assert DEFAULT_COMPONENT_ID == "wsm6-mp6"
    assert moisture_scalar_count(DEFAULT_COMPONENT_ID) == 6


@pytest.mark.skipif(
    _cm1_source("src/param.F") is None, reason="CM1 r21.1 authority tree not present"
)
def test_both_tables_are_live_against_the_cm1_authority() -> None:
    param = _cm1_source("src/param.F")
    assert param is not None
    # Ten scalars, the last four of them the number concentrations nci, ncs, ncr and ncg.
    for mark in ("975f8b7e2600b69d/28", "982d950210915280/14",
                 "7845ac29ee18974f/14", "61d56888b96f4268/14", "6fde2761a8134679/15"):
        assert cm1_marks.has(param, mark), mark
    assert "nnc1 = 7" in param and "nnc2 = 10" in param
    morrison = _cm1_source("src/morrison.F")
    assert morrison is not None
    # The inum=1 table is the only reachable one in an unmodified r21.1.
    assert "      INUM = 1" in morrison
