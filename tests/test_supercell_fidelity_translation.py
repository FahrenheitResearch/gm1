"""CM1's stock supercell deck crosses to GM1 with nothing dropped in silence.

Two things the translator did on CM1's own supercell deck, both found by the
supercell-fidelity comparison (evidence/oracle/supercell-fidelity):

* it dropped ``ramp_sgs = 1`` without a word, because "unrepresentable" meant
  "differs from CM1's distributed run/namelist.input", and every shipped deck
  sets ``ramp_sgs = 1``;
* it skipped any key the r21.1 inventory did not hold (a newer CM1's, or a
  misspelling), so what that key selected simply was not run.

The ramp now crosses as ``closures.ramp_sgs``/``closures.ramp_time``, and an
unmapped key is judged against what GM1 actually integrates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from authority_trees import CM1_PRISTINE, requires_cm1_pristine
from cm1core.errors import CM1Refusal
from cm1core.oracle import keymap
from cm1core.oracle.spec import DifferentialCase

REPO = Path(__file__).resolve().parents[1]
SUPERCELL = CM1_PRISTINE / "run" / "config_files" / "supercell" / "namelist.input"
GM1_ONLY = {
    "run.lane": "cartesian",
    "runtime.warmup_steps": 0,
    "runtime.memory_budget_mib": 5000,
    "runtime.capacity_profile": "default",
}


def _case(component: str, **cm1: object) -> DifferentialCase:
    return DifferentialCase(
        name="cm1-supercell-stock", term="microphysics", notes="",
        cm1=cm1, gm1_only={**GM1_ONLY, "physics.component": component},
    )


def test_every_unmapped_state_key_says_what_gm1_runs() -> None:
    table = keymap.load()
    payload = json.loads(keymap.RECORD.read_text(encoding="utf-8"))
    assert payload["gm1_behaviour_unrecorded"] == []
    for entry in table.unmapped:
        if entry.affects_state:
            behaviour = entry.gm1_behaviour
            # gm1_runs, read_when, or both: CM1 reads it only under read_when,
            # and there GM1 runs gm1_runs (nt_c: Thompson only, at ArWen's 100).
            assert behaviour and (("gm1_runs" in behaviour) or ("read_when" in behaviour)), entry.cm1_key
            if "gm1_runs" in behaviour and "read_when" in behaviour:
                assert entry.cm1_key == "nt_c", entry.cm1_key
    ramp = table.by_cm1("ramp_sgs")
    assert ramp.gm1_key == "closures.ramp_sgs"
    assert table.by_cm1("ramp_time").gm1_key == "closures.ramp_time"
    assert "7 is wsm6-mp6" in table.by_cm1("ptype").note


@requires_cm1_pristine
def test_the_stock_supercell_deck_carries_cm1s_sgs_ramp(tmp_path: Path) -> None:
    from cm1core.config import load_config

    case = _case("morrison-mp10")
    translation = case.translate(SUPERCELL)
    assert translation.gm1["closures"]["ramp_sgs"] == 1
    assert translation.gm1["closures"]["ramp_time"] == 1800.0
    assert translation.unrepresentable == ()
    assert translation.notices == ()
    inert = {row["cm1_key"] for row in translation.inert_unmapped}
    # DNS, nudging and hurricane words are unread on both sides of this deck.
    assert {"viscosity", "lsnudge_tau", "hurr_vg", "var1"} <= inert
    deck = tmp_path / "stock.toml"
    deck.write_text(case.gm1_toml(SUPERCELL), encoding="utf-8")
    config = load_config(deck)
    assert (config.closures.ramp_sgs, config.closures.ramp_time) == (1, 1800.0)
    from cm1core.cartesian_door import build_runtime_config

    runtime = build_runtime_config(config)
    assert (runtime.ramp_sgs, runtime.ramp_time_s) == (1, 1800.0)


@requires_cm1_pristine
def test_a_value_gm1_does_not_run_is_named_even_at_cm1s_shipped_default() -> None:
    translation = _case("morrison-mp10", idiss=0, horizturb=1, iptra=1).translate(SUPERCELL)
    named = {row["cm1_key"]: row for row in translation.unrepresentable}
    # iptra=1 also makes CM1 read npt and pdtra (src/solve2.F:1906-2066).
    assert set(named) == {"idiss", "horizturb", "iptra", "npt", "pdtra"}
    assert named["idiss"]["gm1_runs"] == 1
    assert named["npt"]["cm1_reads_it_because"] == [{"iptra": [1]}]
    # nt_c is read by Thompson only (src/param.F:1711): inert beside Morrison,
    # named beside Thompson, whatever value the deck ships.
    assert "nt_c" in {row["cm1_key"] for row in _case("morrison-mp10").translate(SUPERCELL).inert_unmapped}
    thompson = _case("thompson-mp8").translate(SUPERCELL)
    reasons = {row["cm1_key"]: row for row in thompson.unrepresentable}
    assert reasons["nt_c"]["cm1_reads_it_because"] == [{"ptype": [3]}]
    assert thompson.notices and thompson.notices[0]["run"] == [1, 3]


@requires_cm1_pristine
def test_a_key_outside_the_r21_1_inventory_is_named(tmp_path: Path) -> None:
    text = SUPERCELL.read_text(encoding="utf-8")
    # CM1 r22.0's surface-exchange word, and a statistics word in &param10.
    text = text.replace(" &param12\n", " &param12\n cnstch     = 0.001,\n", 1)
    text = text.replace(" &param10\n", " &param10\n stat_newword = 1,\n", 1)
    assert "cnstch" in text and "stat_newword" in text
    deck = tmp_path / "namelist.input"
    deck.write_text(text, encoding="utf-8")
    translation = _case("morrison-mp10").translate(deck)
    named = {row["cm1_key"]: row for row in translation.unrepresentable}
    assert set(named) == {"cnstch"}
    assert "not in CM1 r21.1's physical-key inventory" in named["cnstch"]["note"]
    assert translation.receipt()["unvalidated"] is True


@requires_cm1_pristine
def test_the_wsm6_case_names_the_equation_set_cm1_runs() -> None:
    translation = _case("wsm6-mp6").translate(SUPERCELL)
    named = {row["cm1_key"]: row for row in translation.unrepresentable}
    assert set(named) == {"eqtset"}
    assert named["eqtset"]["cm1_runs"] == 1
    assert "param.F:936-948" in named["eqtset"]["note"]


def _document(**closures: object) -> str:
    text = (REPO / "examples" / "cartesian-supercell-morrison-hail.toml").read_text(encoding="utf-8")
    extra = "".join(f"{key} = {value!r}\n".replace("'", '"') for key, value in closures.items())
    return text.replace("[closures]\n", "[closures]\n" + extra, 1)


def _load(tmp_path: Path, text: str):
    from cm1core.config import load_config

    path = tmp_path / "deck.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_the_ramp_pair_is_optional_all_or_none_and_cm1s_values(tmp_path: Path) -> None:
    absent = _load(tmp_path, _document())
    assert (absent.closures.ramp_sgs, absent.closures.ramp_time) == (0, 0.0)
    assert "ramp_sgs" not in absent.canonical_dict()["closures"]
    declared = _load(tmp_path, _document(ramp_sgs=1, ramp_time=1800.0))
    assert declared.canonical_dict()["closures"]["ramp_sgs"] == 1
    for bad, code in (
        ({"ramp_sgs": 1}, "CM1-CFG-SGS-RAMP-PARTIAL"),
        ({"ramp_sgs": 2, "ramp_time": 1800.0}, "CM1-CFG-SGS-RAMP"),
        ({"ramp_sgs": 1, "ramp_time": -1.0}, "CM1-CFG-SGS-RAMP-TIME"),
    ):
        with pytest.raises(CM1Refusal) as refused:
            _load(tmp_path, _document(**bad))
        assert refused.value.detail.code == code


def test_nt_c_is_read_for_thompson_only_and_gm1_runs_arwens_100() -> None:
    from cm1core.cm1import import translate

    table = keymap.load()
    behaviour = table.by_cm1("nt_c").gm1_behaviour
    assert behaviour["read_when"] == [{"ptype": [3]}] and behaviour["gm1_runs"] == 100.0
    base = dict(translate.distributed_reference())

    def unrepresentable(component: str, nt_c: float) -> dict:
        effective = {**base, "nt_c": nt_c}
        translation = translate.translate(
            effective, component=component, gm1_only=translate.PRODUCT_GM1_ONLY,
            output={}, declared=("nt_c",),
        )
        return {row["cm1_key"]: row for row in translation.unrepresentable}

    # Morrison: CM1 does not read nt_c, whatever its value.
    assert "nt_c" not in unrepresentable("morrison-mp10", 250.0)
    # Thompson: CM1 reads it, and GM1 runs 100 cm-3.
    assert "nt_c" not in unrepresentable("thompson-mp8", 100.0)
    row = unrepresentable("thompson-mp8", 250.0)["nt_c"]
    assert row["gm1_runs"] == 100.0 and row["cm1_reads_it_because"] == [{"ptype": [3]}]
