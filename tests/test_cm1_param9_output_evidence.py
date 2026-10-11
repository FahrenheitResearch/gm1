"""The recorded &param9 validation, read back and checked for honesty.

This test does not re-run anything.  It reads the receipt this campaign wrote
and asserts that it still says what it said: which product was compared
against CM1, at what level it agreed, and -- for the products that did NOT
stay inside their stated tolerance -- that the receipt still shows the
decaying ratio and the absolute difference that justify the reading, rather
than a growing one nobody looked at again.
"""

import json
from pathlib import Path

import pytest

_EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "evidence" / "validation" / "cm1-param9-output" / "receipt.json"
)

pytestmark = pytest.mark.skipif(
    not _EVIDENCE.is_file(), reason="the &param9 validation receipt is not in this tree"
)


def _receipt() -> dict:
    return json.loads(_EVIDENCE.read_text())


def test_the_receipt_names_its_authority_and_its_claim_limit():
    receipt = _receipt()
    for anchor in ("run/namelist.input:213-278", "src/misclibs.F:1417-1532",
                   "src/solve3.F:857-1130", "src/getcape.F:14-520"):
        assert anchor in receipt["cm1_authority"]
    assert "not forecast skill" in receipt["claim_limit"]


def test_every_compared_product_either_held_its_tolerance_or_decayed():
    """The discriminator is the SHAPE of the curve, not one number.

    A diagnostic operator that disagrees with CM1 produces a ratio that grows
    with the field; one that agrees on a field which starts at zero produces
    one that shrinks as the field becomes real.  Nothing in this receipt is
    allowed to be outside tolerance AND growing.
    """

    products = _receipt()["differential_vs_cm1"]["products"]
    assert len(products) >= 20
    for name, row in products.items():
        if row["within_tolerance_at_every_frame"]:
            continue
        assert row["ratio_decays"], (
            f"{name} is outside its stated tolerance and its ratio is not "
            "decaying; that is the signature of a wrong operator, not of a "
            "near-zero field"
        )


def test_the_products_with_a_real_signal_held_their_tolerance():
    """These are the fields this case actually has, so they are the test."""

    products = _receipt()["differential_vs_cm1"]["products"]
    for name in ("cape", "cin", "lcl", "lfc", "xvort", "yvort", "th", "pwat",
                 "sws", "sps", "rain", "prate"):
        assert products[name]["within_tolerance_at_every_frame"], name


def test_the_getcape_differential_is_against_cm1s_own_binary():
    columns = _receipt()["getcape_vs_cm1_binary"]["columns"]
    assert len(columns) >= 4
    for name, row in columns.items():
        assert row["status"] == "PASS", name
        assert row["comparisons"] == 24, name
        # At worst a libm difference, and mostly not even that.
        assert row["worst_relative_difference"] < 1e-4, name
    # And the observed soundings are severe-weather soundings, so the number
    # the operator returns has to be a severe-weather number.
    observed = [row for name, row in columns.items() if name != "synthetic"]
    assert len(observed) == 3
    for row in observed:
        assert row["surface_parcel_cape_j_kg"] > 2000.0
        assert 200.0 < row["surface_parcel_lcl_m"] < 1200.0


def test_every_unvalidated_marker_names_what_was_not_checked():
    rows = _receipt()["unvalidated"]
    assert rows
    for row in rows:
        assert row["marker"] == "UNVALIDATED"
        assert row["option"].startswith("output_")
        assert row["cm1_output_names"]
        assert len(row["not_checked"]) > 60
    options = {row["option"] for row in rows}
    # The three this pass could not settle, by name.
    assert {"output_dbz", "output_vort", "output_uh"} <= options


def test_the_receipt_and_the_selection_table_agree_on_what_is_unvalidated():
    """A product marked unvalidated in one place and not the other is a lie."""

    from cm1core.cm1_output_selection import SWITCH_BY_KEY

    recorded = {row["option"] for row in _receipt()["unvalidated"]}
    declared = {
        switch.key for switch in SWITCH_BY_KEY.values()
        if switch.unvalidated is not None
    }
    # Everything the selection table flags must be in the receipt; the receipt
    # may carry more, because a marker can depend on a run's own selectors
    # (output_vort under bbc=3) rather than on the switch alone.
    assert declared <= recorded
