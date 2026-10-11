"""CM1 r21.1's three runnable shipped decks, imported with no CM1 build anywhere.

The scoping translated CM1's supercell and squall_line decks with the
differential harness (kept in the development repository), which
needed a cm1.exe on the authority search path.  The importer needs none: the
parser is native, unset keys take r21.1's code defaults from a record, and the
"differs from CM1's distributed deck" test reads the same record.

The importer's documents must equal the scoping's, key for key, except:

* the ``&param9`` selection -- ``[output]``, ``[diagnostics]`` and
  ``physics.output_dbz``.  The harness forces its own selection on both
  models; the importer keeps the user's;
* keys the namelist does not set, stated at CM1 r21.1's code default.  The scoping left them to gpuwm-cm1's own default;
* ``physics.component`` and ``initialization.iorigin``, which are intentional
  too (the importer derives the component from ptype and keeps
  iorigin as written).  On these three decks neither differs, and the test
  says so;
* ``closures.ramp_sgs`` and ``closures.ramp_time``, which the key map gained
  after the scoping (the supercell fidelity work bound CM1's SGS start-up ramp,
  turb.F:240-262).  All three decks state ``ramp_sgs = 1`` and
  ``ramp_time = 1800.0``; the scoping's documents could not carry them, and
  the importer carries them as stated.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from cm1import_support import (
    DATA,
    ROOT,
    RUN_DEFAULT,
    SQUALL_LINE,
    SUPERCELL,
    flat_toml,
    namelist_tool,
    scoping_decks,
)

#: Each deck's reference translation: the scoping's (in the development repository) or a shipped one.
REFERENCES = {
    SUPERCELL: "supercell/supercell.toml",
    SQUALL_LINE: "squall_line/squall_line.toml",
    RUN_DEFAULT: DATA / "r21.1-run.harness-translation.toml",
}
AUTHORITY_PATHS = {
    SUPERCELL: "run/config_files/supercell/namelist.input",
    SQUALL_LINE: "run/config_files/squall_line/namelist.input",
    RUN_DEFAULT: "run/namelist.input",
}
PARAM9_PROJECTION = ("output.", "diagnostics.", "physics.output_dbz")


def _reference(deck: Path) -> Path:
    reference = REFERENCES[deck]
    return reference if isinstance(reference, Path) else scoping_decks() / reference


PLAN_LISTED = ("physics.component", "initialization.iorigin")
#: Keys the key map gained after the scoping, carried as the namelist states them.
MAPPED_AFTER_SCOPING = {"closures.ramp_sgs": ("ramp_sgs", 1), "closures.ramp_time": ("ramp_time", 1800.0)}
#: The rain and hail swaths the scoping translation switched off beside Morrison; the importer keeps them (CM1's qr and qg
#: rows are Morrison's rows 3 and 6, solve3.F:993-1047), so the harness selection (1) now reaches the document.
KEPT_AFTER_SCOPING = {"diagnostics.output_srs", "diagnostics.output_sgs"}


@pytest.fixture()
def no_cm1_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No CM1_AUTHORITY_ROOT, no cm1.exe on the search path, and a trap on the lookup."""

    from cm1core.oracle import cm1_run

    monkeypatch.delenv("CM1_AUTHORITY_ROOT", raising=False)
    monkeypatch.setattr(cm1_run, "AUTHORITY_SEARCH_PATH", (tmp_path / "no-cm1",))
    with pytest.raises(cm1_run.CM1RunError):
        cm1_run.authority_run_dir()

    def trap(*_args, **_kwargs):
        raise AssertionError("the importer looked for a CM1 build")

    monkeypatch.setattr(cm1_run, "authority_run_dir", trap)
    yield


def test_the_fixtures_are_the_pinned_r21_decks() -> None:
    manifest = {}
    for line in (ROOT / "evidence/authority/cm1r21.1-content-sha256.tsv").read_text().splitlines():
        fields = line.split("\t")
        if len(fields) == 3:
            manifest[fields[2]] = fields[0]
    for deck, relative in AUTHORITY_PATHS.items():
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == manifest[relative], relative


@pytest.mark.parametrize("deck", list(REFERENCES), ids=lambda path: path.name)
def test_runnable_deck_translates_as_the_scoping_did(deck: Path, no_cm1_build) -> None:
    from cm1core.cm1import import import_namelist
    from cm1core.cm1import import defaults as cm1_defaults

    assert "CM1_AUTHORITY_ROOT" not in os.environ
    imported = import_namelist(deck, tool=namelist_tool())
    assert imported.version == "r21.1"
    assert imported.component.component == "morrison-mp10"
    assert imported.component.status == "runs"
    new = flat_toml(imported.toml())
    reference = flat_toml(_reference(deck).read_text(encoding="utf-8"))

    differing = sorted(
        key for key in set(new) | set(reference) if new.get(key, "<absent>") != reference.get(key, "<absent>")
    )
    param9 = [key for key in differing if key.startswith(PARAM9_PROJECTION)]
    listed = [key for key in differing if key in PLAN_LISTED]
    later = [key for key in differing if key in MAPPED_AFTER_SCOPING]
    rest = [key for key in differing if key not in param9 and key not in listed and key not in later]
    assert later == sorted(MAPPED_AFTER_SCOPING)
    for key in later:
        name, value = MAPPED_AFTER_SCOPING[key]
        assert key not in reference
        assert name in imported.parsed.stated_keys()
        assert new[key] == imported.effective[name] == value, key
    # Neither plan-listed field differs on these decks.
    assert listed == []
    assert new["physics.component"] == reference["physics.component"] == "morrison-mp10"
    assert new["initialization.iorigin"] == reference["initialization.iorigin"]
    # Every other difference is a key the namelist does not set, stated at
    # CM1 r21.1's own code default.
    defaults = cm1_defaults.load()
    stated = imported.parsed.stated_keys()
    filled = {row.key for row in imported.defaulted}
    for key in rest:
        assert key not in reference, f"{key} changed value: {reference[key]!r} -> {new.get(key)!r}"
        from cm1core.cm1import import keymap

        [entry] = [e for e in keymap.load().entries if e.gm1_key == key]
        assert entry.cm1_key not in stated and entry.cm1_key in filled, key
        [group] = defaults.groups_of(entry.cm1_key)
        assert new[key] == defaults.field(group, entry.cm1_key).code_default, key
    expected_rest = [] if deck == RUN_DEFAULT else ["dynamics.beta_plane"]
    assert rest == expected_rest
    # The &param9 projection is the namelist's own selection.
    parsed = imported.parsed.values()["param9"]
    for key in param9:
        name = key.split(".", 1)[1]
        if key.startswith("output.") and key in new:
            assert new[key] == parsed[name], key


@pytest.mark.parametrize("deck", list(REFERENCES), ids=lambda path: path.name)
def test_the_harness_selection_reproduces_the_scoping_translation(deck: Path, no_cm1_build) -> None:
    """Same parser, defaults and translator; the harness's &param9 selection.

    With HARNESS_OUTPUT in place of the namelist's selection the importer's
    effective configuration translates to the scoping document exactly, bar
    the keys the namelist does not set and the ramp pair the key map gained
    after the scoping.
    """

    from cm1core.cm1import import import_namelist, translate
    from cm1core.oracle.spec import HARNESS_OUTPUT

    imported = import_namelist(deck, tool=namelist_tool())
    translation = translate.translate(
        imported.effective,
        component="morrison-mp10",
        gm1_only=translate.PRODUCT_GM1_ONLY,
        output=HARNESS_OUTPUT,
    )
    new = flat_toml(translate.render_toml(translation, ()))
    reference = flat_toml(_reference(deck).read_text(encoding="utf-8"))
    differing = {key for key in set(new) | set(reference) if new.get(key) != reference.get(key)}
    expected = set(MAPPED_AFTER_SCOPING) | (set() if deck == RUN_DEFAULT else {"dynamics.beta_plane"})
    assert differing == expected | KEPT_AFTER_SCOPING
    for key in KEPT_AFTER_SCOPING:
        assert reference[key] == 0 and new[key] == 1, key


@pytest.mark.parametrize("deck", list(REFERENCES), ids=lambda path: path.name)
def test_the_imported_document_is_admitted_by_the_configuration_door(deck: Path, tmp_path: Path, no_cm1_build) -> None:
    from cm1core.cm1import import import_namelist
    from cm1core.config import load_config

    imported = import_namelist(deck, tool=namelist_tool())
    path = tmp_path / "deck.toml"
    path.write_text(imported.toml(), encoding="utf-8")
    config = load_config(path)
    assert config.run.lane.value == "cartesian"
    # CM1's &param9 words the door does not take are recorded, not dropped:
    # km and kh are on in all three decks and gpuwm-cm1 writes neither.
    refused = {row["cm1_key"] for row in imported.translation.output_not_admitted}
    assert {"output_km", "output_kh", "output_filetype", "output_dbz"} <= refused
    assert imported.output_writer == {"output_format": 1, "output_filetype": 1}
