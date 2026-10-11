"""The differential harness and the importer share one translator.

``oracle/spec.py`` used to carry its own translator, which needed a cm1.exe to
read CM1's distributed deck.  It now calls ``cm1import.translate``, which
judges an unmapped key against what GM1 integrates (the key map's
``gm1_behaviour``) and needs no CM1 build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cm1import_support import ROOT, SUPERCELL

CASE = ROOT / "examples" / "oracle" / "acoustic-warm-bubble-32.differential.toml"


def test_the_harness_calls_the_shared_translator(monkeypatch: pytest.MonkeyPatch) -> None:
    from cm1core.cm1import import translate
    from cm1core.oracle.spec import HARNESS_OUTPUT, load_case

    calls = []
    original = translate.translate

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(translate, "translate", spy)
    load_case(CASE).translate(SUPERCELL)
    [call] = calls
    assert call["output"] is HARNESS_OUTPUT
    # The supercell fidelity work moved this worked example to Morrison, CM1's
    # eqtset=2 scheme (CM1 integrates WSM6 at eqtset=1).
    assert call["component"] == "morrison-mp10"


def test_the_harness_translates_without_a_cm1_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cm1core.oracle import cm1_run
    from cm1core.oracle.spec import DifferentialCase, load_case

    monkeypatch.delenv("CM1_AUTHORITY_ROOT", raising=False)
    monkeypatch.setattr(cm1_run, "AUTHORITY_SEARCH_PATH", (tmp_path / "no-cm1",))
    case = load_case(CASE)
    # horizturb and iptra have no gpuwm-cm1 key, so the translator compares
    # them with what GM1 integrates (the key map's gm1_behaviour).  It once
    # compared them with CM1's distributed deck, which needed a cm1.exe.
    probing = DifferentialCase(
        name=case.name, term=case.term, notes=case.notes,
        cm1={**case.cm1, "horizturb": 1, "iptra": 1}, gm1_only=case.gm1_only,
    )
    receipt = probing.translate(SUPERCELL).receipt()
    named = {row["cm1_key"]: row for row in receipt["unrepresentable_overrides"]}
    assert named["horizturb"]["gm1_runs"] == 0
    assert named["iptra"]["gm1_runs"] == 0
    assert case.translate(SUPERCELL).unrepresentable == ()


def test_the_harness_error_is_the_translator_error() -> None:
    from cm1core.cm1import.translate import TranslationError
    from cm1core.oracle.spec import CaseError, _model_top_m

    assert CaseError is TranslationError
    with pytest.raises(CaseError, match="reads its levels from a file"):
        _model_top_m({"stretch_z": 3, "nz": 40, "dz": 200.0})


def test_the_old_keymap_name_is_the_same_reader() -> None:
    from cm1core.cm1import import keymap as shared
    from cm1core.oracle import keymap as old

    assert old.load is shared.load and old.RECORD == shared.RECORD
