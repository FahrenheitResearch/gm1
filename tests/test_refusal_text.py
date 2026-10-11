"""``gui/refusal_text.py``: plain words for refusals, in the style guided mode uses.

Every refused row the importer produces for CM1's shipped decks and the
real-namelist corpus carries plain words written for its code or setting (not
the generic sentence), and every plain sentence keeps the copy rules: a full
sentence, short, no marketing words, no tornado wording, no names other than
GM1, CM1 and the schemes', and no CM1 setting names -- those are the grey
hints beside it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cm1import_support import SUPERCELL, corpus_namelists, edited, namelist_tool, pristine_config_files, shipped_decks

BANNED = (
    "powerful", "seamless", "seamlessly", "blazing", "revolutionary", "cutting-edge", "state-of-the-art",
    "amazing", "incredible", "effortless", "easy", "easily", "simply", "just", "world-class", "best",
    "magic", "stunning", "tornado", "tornadoes", "twister", "unleash", "supercharge",
)
NAMES = ("ArWen", "gpuwm", "CUDA", "CuPy", "NVRTC", "WRF", "cm1core")
#: CM1 setting names that are also plain English words.
ENGLISH = {"day", "hour", "minute", "second", "month", "year", "season", "time", "dns", "viscosity"}


def _cm1_keys() -> set[str]:
    from cm1core.cm1import import defaults

    return {key for rows in defaults.load().fields.values() for key in rows} - ENGLISH


def _lint(text: str, keys: set[str]) -> list[str]:
    problems = []
    if not text.endswith("."):
        problems.append("does not end with a full stop")
    if len(text) > 140:
        problems.append(f"{len(text)} characters")
    words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", text)
    lowered = {word.lower() for word in words}
    problems += [f"banned word {word!r}" for word in BANNED if word in lowered]
    problems += [f"names {name}" for name in NAMES if name in words]
    problems += [f"CM1 setting {word}" for word in words if word in keys]
    return problems


def _all_plain_texts() -> list[str]:
    from cm1core.gui import refusal_text

    texts = []
    for table in (refusal_text.BY_KEY, refusal_text.BY_CODE):
        for name, entry in table.items():
            for value in (0, 1, 2, 3, 7, 12, 26, 53, "x"):
                found = entry(value, {"dx": 100.0})
                if found is not None:
                    texts += [found.plain] + ([found.plain_fix] if found.plain_fix else [])
    for entry in refusal_text.BY_CODE_AND_KEY.values():
        found = entry(1, {})
        texts += [found.plain, found.plain_fix]
    for found in refusal_text.GENERIC.values():
        texts += [found.plain, found.plain_fix]
    return sorted(set(text for text in texts if text))


def test_every_plain_sentence_keeps_the_copy_rules() -> None:
    keys = _cm1_keys()
    texts = _all_plain_texts()
    assert len(texts) >= 60
    problems = {text: _lint(text, keys) for text in texts}
    assert not {text: found for text, found in problems.items() if found}


def test_the_table_covers_about_thirty_codes_and_settings() -> None:
    from cm1core.gui import refusal_text

    covered = refusal_text.covered()
    assert len(covered["keys"]) + len(covered["codes"]) + len(covered["code_and_key"]) >= 30


def _sources(tmp_path: Path) -> list[Path]:
    paths = [path for _, path in corpus_namelists(tmp_path)]
    root = pristine_config_files()
    if root is not None:
        paths += [root / name / "namelist.input" for name in shipped_decks()]
    paths.append(edited(SUPERCELL, tmp_path, {"stretch_x": "1", "ptype": "26", "isnd": "15"}))
    return paths


def test_every_refused_row_has_its_own_plain_words(tmp_path: Path) -> None:
    from cm1core.cm1import.coverage import build_report, import_for_report
    from cm1core.gui.refusal_text import GENERIC

    keys = _cm1_keys()
    generic = []
    rows = 0
    register_texts = set()
    for path in _sources(tmp_path):
        report = build_report(import_for_report(path, tool=namelist_tool()))
        for row in report.rows:
            if row.category == "refused":
                rows += 1
                assert row.plain and row.plain_fix and row.fix, row.as_dict()
                if row.plain == GENERIC["refused"].plain:
                    generic.append((row.code, row.keys))
            elif row.plain:
                register_texts.add(row.plain)
    assert rows >= 100
    # The corpus and the shipped decks: every refusal has words of its own.
    assert len(generic) <= rows // 20, generic
    problems = {text: _lint(text, keys) for text in register_texts}
    assert not {text: found for text, found in problems.items() if found}


def test_guided_texts_for_the_three_headline_refusals() -> None:
    from cm1core.gui.refusal_text import describe

    nssl = describe(code="CM1-IMPORT-PTYPE", keys=("ptype",), value=26, context={}, category="refused")
    assert nssl.plain == "NSSL microphysics is not available in GM1."
    dycoms = describe(code="CM1-CFG-UNSUPPORTED-VALUE", keys=("isnd",), value=15, context={}, category="refused")
    assert dycoms.plain == "This test profile needs different physical constants than GM1 is built with."
    # Every init3d.F arm exists; only a value with no arm is refused.
    unknown = describe(code="CM1-CFG-UNSUPPORTED-VALUE", keys=("iinit",), value=6, context={}, category="refused")
    assert unknown.plain == "This is not one of CM1's ways of starting the storm."
    assert "iinit = 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12 and 13" in unknown.cm1
    stretch = describe(code="CM1-CFG-UNSUPPORTED-VALUE", keys=("stretch_x",), value=True, context={"dx": 100.0},
                       category="refused")
    assert "60 x 60 km x 80 levels" in stretch.fix
    assert describe(code=None, keys=("kdiv",), value=0.2, context={}, category="not_checked") is None
