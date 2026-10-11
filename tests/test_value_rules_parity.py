"""``records/CM1-VALUE-RULES.json`` against ``config.py``: the record is the code's own rules.

The record lists every ``_choice``, ``_integer_choice`` and
``_optional_integer_choice`` site of ``config.py`` with the option it judges,
the values it admits and when it applies.  These tests hold it to the code
three ways: the record is exactly what the extractor reads from the current
``config.py``; the extractor misses no call site; and every key-level rule
refuses, through ``config.parse_mapping`` itself, the values the record says it
refuses and admits the ones it says it admits -- the importer's key-level
verdict (``rules.judge``) agrees with the check it stands in for.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from cm1import_support import ROOT, SUPERCELL, namelist_tool

RECORD = ROOT / "src" / "cm1core" / "records" / "CM1-VALUE-RULES.json"
CONFIG = ROOT / "src" / "cm1core" / "config.py"


def test_the_record_is_what_config_py_says() -> None:
    from cm1core.cm1import import rules

    regenerated = rules.record_payload(rules.extract_value_rules())
    recorded = json.loads(RECORD.read_text(encoding="utf-8"))
    assert recorded == json.loads(json.dumps(regenerated)), (
        "records/CM1-VALUE-RULES.json is stale: run tools/generate_cm1_value_rules.py"
    )
    assert rules.load().rules == tuple(rules.extract_value_rules())


def _call_sites() -> int:
    """Every call of the three helpers outside their own definitions, literal loops expanded."""

    from cm1core.cm1import.rules import CHOICE_HELPERS

    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    count = 0
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name in CHOICE_HELPERS:
            continue
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CHOICE_HELPERS):
                continue
            weight = 1
            for loop in ast.walk(function):
                if isinstance(loop, ast.For) and isinstance(loop.iter, (ast.Tuple, ast.List)) and any(
                    child is node for child in ast.walk(loop)
                ):
                    weight *= len(loop.iter.elts)
            count += weight
    return count


def test_no_call_site_is_missed() -> None:
    from cm1core.cm1import import rules

    extracted = rules.extract_value_rules()
    assert len(extracted) == _call_sites()
    applies = {rule.applies for rule in extracted}
    assert applies <= {"cartesian", "when-declared", "conditional", "dynamic"}
    # The rules the importer judges at key level are the ones the plan names.
    key_level = {rule.option for rule in extracted if rule.key_level}
    assert {"dynamics.psolver", "initialization.iinit", "grid.stretch_x", "grid.stretch_y",
            "dynamics.velocity_weno", "initialization.isnd", "grid.stretch_z"} <= key_level


@pytest.fixture(scope="module")
def stock_document() -> dict:
    from cm1core.cm1import.coverage import first_pass_document, import_for_report

    document = first_pass_document(import_for_report(SUPERCELL, tool=namelist_tool()))
    assert document is not None
    return document


def _refusals(document: dict) -> list:
    from cm1core.config import parse_mapping
    from cm1core.errors import CM1Refusals, collect_refusals

    try:
        with collect_refusals():
            parse_mapping(copy.deepcopy(document))
    except CM1Refusals as aggregate:
        return [row.detail for row in aggregate.refusals]
    return []


def _outside(allowed: tuple, integer: bool) -> list:
    if all(isinstance(value, bool) for value in allowed):
        return [not allowed[0]] if len(allowed) == 1 else []
    if all(isinstance(value, str) for value in allowed):
        return ["not-an-admitted-value"]
    numbers = [value for value in allowed if isinstance(value, (int, float)) and not isinstance(value, bool)]
    candidates = [max(numbers) + 1, min(numbers) - 1] if numbers else []
    if integer:
        candidates.append(1.5)
    return [value for value in candidates if value not in allowed]


def _key_level_rules() -> list:
    from cm1core.cm1import import rules

    seen = set()
    chosen = []
    for rule in rules.load().key_level:
        if rule.option in seen:
            continue
        seen.add(rule.option)
        chosen.append(rule)
    return chosen


@pytest.mark.parametrize("rule", _key_level_rules(), ids=lambda rule: rule.option)
def test_a_key_level_rule_is_what_the_check_does(rule, stock_document: dict) -> None:
    from cm1core.cm1import import rules

    record = rules.load()
    section, _, name = rule.option.partition(".")
    if section not in stock_document:
        pytest.skip(f"the stock deck declares no [{section}] table")
    document_codes = {"CM1-CFG-UNSUPPORTED-VALUE", "CM1-CFG-INTEGER"}
    admitted = record.admitted(rule.option)
    for bad in _outside(admitted, rule.integer):
        document = copy.deepcopy(stock_document)
        document[section][name] = bad
        refused = [detail for detail in _refusals(document) if detail.option == rule.option]
        assert refused, (rule.option, bad)
        breaches = [breach for breach in rules.judge(document) if breach.option == rule.option]
        assert breaches, (rule.option, bad)
        if refused[0].code in document_codes:
            assert breaches[0].code == refused[0].code, (rule.option, bad)
        else:
            # A more specific refusal of the same value comes first (lspgrad = 2,
            # 3 and 4 are CM1 branches this port has not ported); the rule's own
            # refusal is behind it.
            assert rule.option == "dynamics.lspgrad" and refused[0].code == "CM1-CFG-LSPGRAD-UNPORTED"
    for good in admitted:
        document = copy.deepcopy(stock_document)
        document[section][name] = good
        refused = [detail for detail in _refusals(document)
                   if detail.option == rule.option and detail.code in document_codes]
        assert not refused, (rule.option, good, refused)
        assert not [breach for breach in rules.judge(document) if breach.option == rule.option], (rule.option, good)


def test_the_stock_deck_breaks_no_rule(stock_document: dict) -> None:
    from cm1core.cm1import import rules

    assert rules.judge(stock_document) == []
