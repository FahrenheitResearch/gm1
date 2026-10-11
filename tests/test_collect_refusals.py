"""``errors.collect_refusals``: every refusal of one pass, the first exactly the fail-fast one.

Inside the block a value check (one setting judged on its own value) records
its refusal and returns; a check across settings records its refusal and ends
the pass; at the end every refusal is raised as one ``CM1Refusals`` whose
``detail`` is the first.  The parity tests hold that first refusal to the one
the same checks raise outside the block, on every document the importer's
first pass builds from CM1's shipped decks, the real-namelist corpus and the
stock deck's variants.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cm1import_support import (
    R22_SUPERCELL,
    SQUALL_LINE,
    SUPERCELL,
    corpus_namelists,
    edited,
    namelist_tool,
    pristine_config_files,
    shipped_decks,
)


def _value(code: str, option: str, value: object = 0) -> None:
    from cm1core.errors import refuse

    refuse(code=code, gate="g", reason=f"{option} bad", remedy="fix it", option=option, value=value, independent=True)


def _across(code: str, option: str) -> None:
    from cm1core.errors import refuse

    refuse(code=code, gate="g", reason=f"{option} disagrees", remedy="fix it", option=option, value=1)


def test_value_checks_continue_and_a_check_across_settings_ends_the_pass() -> None:
    from cm1core.errors import CM1Refusal, CM1Refusals, collect_refusals

    reached = []

    def checks() -> None:
        _value("A", "x")
        reached.append("after A")
        _value("B", "y")
        reached.append("after B")
        _across("C", "z")
        reached.append("after C")

    with pytest.raises(CM1Refusal) as outside:
        checks()
    assert outside.value.detail.code == "A" and reached == []
    with pytest.raises(CM1Refusals) as inside:
        with collect_refusals():
            checks()
    rows = inside.value.refusals
    assert [(row.detail.code, row.independent, row.after_earlier, row.ended_pass) for row in rows] == [
        ("A", True, False, False), ("B", True, False, False), ("C", False, True, True),
    ]
    assert reached == ["after A", "after B"]
    # The aggregate is a CM1Refusal whose detail is the fail-fast one.
    assert isinstance(inside.value, CM1Refusal)
    assert inside.value.detail == outside.value.detail


def test_a_block_with_no_refusal_is_transparent() -> None:
    from cm1core.errors import collect_refusals

    with collect_refusals() as collector:
        value = 41 + 1
    assert value == 42 and collector.refusals == []


def test_an_error_after_a_refusal_ends_the_pass_and_is_kept() -> None:
    from cm1core.errors import CM1Refusals, collect_refusals

    with pytest.raises(CM1Refusals) as caught:
        with collect_refusals():
            _value("A", "x")
            _ = 1 / 0
    assert isinstance(caught.value.interrupted, ZeroDivisionError)
    assert [row.detail.code for row in caught.value.refusals] == ["A"]


def test_an_error_with_no_refusal_propagates_unchanged() -> None:
    from cm1core.errors import collect_refusals

    with pytest.raises(ZeroDivisionError):
        with collect_refusals():
            _ = 1 / 0


def test_fail_fast_raises_inside_a_collection() -> None:
    from cm1core.errors import CM1Refusal, CM1Refusals, collect_refusals, collecting, fail_fast

    translated = []
    with pytest.raises(CM1Refusals) as caught:
        with collect_refusals():
            assert collecting()
            try:
                with fail_fast():
                    assert not collecting()
                    _value("A", "grid.x")
            except CM1Refusal as refusal:
                translated.append(refusal.detail.code)
                _value("A-TRANSLATED", "nest.x")
    assert translated == ["A"]
    assert [row.detail.code for row in caught.value.refusals] == ["A-TRANSLATED"]
    assert not collecting()


def test_nested_blocks_merge_into_the_outer_one() -> None:
    from cm1core.errors import CM1Refusals, collect_refusals

    with pytest.raises(CM1Refusals) as caught:
        with collect_refusals():
            _value("A", "x")
            try:
                with collect_refusals():
                    _value("B", "y")
            except CM1Refusals as inner:
                assert [row.detail.code for row in inner.refusals] == ["B"]
                raise
    assert [row.detail.code for row in caught.value.refusals] == ["A", "B"]


def test_a_repeated_refusal_is_recorded_once() -> None:
    from cm1core.errors import CM1Refusals, collect_refusals

    with pytest.raises(CM1Refusals) as caught:
        with collect_refusals():
            _value("A", "x")
            _value("A", "x")
    assert len(caught.value.refusals) == 1


def test_config_value_helpers_return_the_value_they_were_given() -> None:
    from cm1core import config
    from cm1core.errors import CM1Refusals, collect_refusals

    with pytest.raises(CM1Refusals) as caught:
        with collect_refusals():
            assert config._integer_choice("dynamics.psolver", 7, {2, 3}, "reason") == 7
            assert config._finite_positive("run.dt_s", "six") == "six"
            assert config._finite_positive("run.dt_s", -1.0) == -1.0
            assert config._boolean("grid.terrain", 1) == 1
            assert config._level_array("grid.levels_m", [0.0, "x", 10.0]) == (0.0, 10.0)
    codes = [(row.detail.code, row.detail.option) for row in caught.value.refusals]
    assert codes == [
        ("CM1-CFG-UNSUPPORTED-VALUE", "dynamics.psolver"),
        ("CM1-CFG-NUMERIC-TYPE", "run.dt_s"),
        ("CM1-CFG-NUMERIC-RANGE", "run.dt_s"),
        ("CM1-CFG-BOOLEAN", "grid.terrain"),
        ("CM1-CFG-LEVEL-ARRAY", "grid.levels_m"),
    ]
    assert all(row.independent for row in caught.value.refusals)


# ---------------------------------------------------------------------------
# Parity with the fail-fast checks, on real documents
# ---------------------------------------------------------------------------


def _documents(tmp_path: Path) -> list[tuple[str, dict]]:
    from cm1core.cm1import.coverage import first_pass_document, import_for_report

    tool = namelist_tool()
    sources: list[tuple[str, Path]] = [
        ("supercell", SUPERCELL),
        ("squall_line", SQUALL_LINE),
        ("r22-supercell", R22_SUPERCELL),
        ("synthetic", edited(SUPERCELL, tmp_path, {"stretch_x": "1", "ptype": "26", "iinit": "4"}, "s1")),
        ("psolver-7", edited(SUPERCELL, tmp_path, {"psolver": "7"}, "s2")),
        ("dtl-bad", edited(SUPERCELL, tmp_path, {"dtl": "7.0", "tapfrq": "1000.0"}, "s3")),
        ("nx-1", edited(SUPERCELL, tmp_path, {"nx": "1"}, "s4")),
    ]
    sources += [(f"corpus:{name}", path) for name, path in corpus_namelists(tmp_path)]
    root = pristine_config_files()
    if root is not None:
        sources += [(f"shipped:{name}", root / name / "namelist.input") for name in shipped_decks()]
    documents = []
    for name, path in sources:
        document = first_pass_document(import_for_report(path, tool=tool))
        if document is not None:
            documents.append((name, document))
    return documents


def _fail_fast(document: dict):
    from cm1core.cartesian_door import cartesian_plan_payload
    from cm1core.config import parse_mapping
    from cm1core.errors import CM1Refusal

    try:
        config = parse_mapping(copy.deepcopy(document))
        cartesian_plan_payload(config)
    except CM1Refusal as refusal:
        return refusal.detail, None
    return None, config.sha256


def _collected(document: dict):
    from cm1core.cartesian_door import cartesian_plan_payload
    from cm1core.config import parse_mapping
    from cm1core.errors import CM1Refusals, collect_refusals

    try:
        with collect_refusals():
            config = parse_mapping(copy.deepcopy(document))
            cartesian_plan_payload(config)
    except CM1Refusals as aggregate:
        return aggregate, None
    return None, config.sha256


def test_the_first_collected_refusal_is_the_fail_fast_refusal(tmp_path: Path) -> None:
    documents = _documents(tmp_path)
    assert len(documents) >= 25
    refused = 0
    several = 0
    for name, document in documents:
        detail, accepted = _fail_fast(document)
        aggregate, collected_accepted = _collected(document)
        if detail is None:
            assert aggregate is None, name
            assert collected_accepted == accepted, name  # the same sealed configuration
            continue
        refused += 1
        assert aggregate is not None, name
        assert aggregate.refusals[0].detail == detail, name
        assert aggregate.detail == detail, name
        several += len(aggregate.refusals) > 1
    # The corpus and the shipped decks exercise collection, not just the first stop.
    assert refused >= 10 and several >= 5
