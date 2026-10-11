"""The coverage report: six categories, every refusal in one pass.

One test group each:

1. CM1's refused shipped decks exit 2, each lists every key-level refusal, and
   the first refusal the check meets is the one ``xlate/decks/results.json``
   recorded (the scoping ran the same translator and ``check``);
2. a namelist with ``stretch_x = 1``, ``ptype = 26`` and ``isnd = 15`` exits 2
   with all three in one run;
3. CM1 r22.0's supercell lists all 32 r22-only keys with their class, physics
   keys at their r21.1 value read "Same as CM1 r21.1", a changed physics key
   exits 2 and a changed output key alone exits 3;
4. the stock deck's report matches its golden file, carries the stock-deck ratios, has
   no ramp_sgs row, a kdiv row only off 0.10, and squall_line's periodic
   "Not yet checked" row while the SGS closure's periodic caveat stands;

and the invariants the GUI relies on: every stated setting lands in exactly
one category, the JSON has the Check screen's shape, the exit code follows the
rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cm1import_support import (
    DATA,
    R18_RUN,
    R22_SUPERCELL,
    RUN_DEFAULT,
    scoping_results,
    SQUALL_LINE,
    SUPERCELL,
    corpus_namelists,
    edited,
    namelist_tool,
    shipped_deck,
    shipped_decks,
)

GOLDEN = DATA / "golden"
RUNNABLE_SHIPPED = ("squall_line", "supercell")


def _report(path: Path, **options):
    from cm1core.cm1import.coverage import build_report, import_for_report

    return build_report(import_for_report(path, tool=namelist_tool()), **options)


def _refused_keys(report) -> set[str]:
    return {key for row in report.category("refused") for key in row.keys + row.covers}


def _row(report, category: str, key: str):
    rows = [row for row in report.category(category) if key in row.keys]
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# 1. CM1's shipped decks
# ---------------------------------------------------------------------------


def test_cm1_ships_23_decks_and_21_are_refused() -> None:
    names = shipped_decks()
    assert len(names) == 23
    assert set(RUNNABLE_SHIPPED) <= set(names)
    results = json.loads(scoping_results().read_text(encoding="utf-8"))
    assert set(results) == set(names)
    # The scoping ran 23 decks: 19 refused by check, 2 its translator could
    # not translate, and the 2 that run.  (The plan's "20" counted one fewer.)
    refused = [name for name in names if results[name].get("check_rc") != 0]
    assert len(refused) == 21


@pytest.mark.parametrize("name", [name for name in shipped_decks() if name not in RUNNABLE_SHIPPED])
def test_a_refused_shipped_deck_lists_every_key_level_refusal(name: str) -> None:
    from cm1core.cm1import import rules
    from cm1core.cm1import.coverage import _option_keys, first_pass_document, import_for_report

    path = shipped_deck(name)
    report = _report(path)
    assert report.exit_code == 2
    refused = _refused_keys(report)
    imported = import_for_report(path, tool=namelist_tool())

    # Key level, derived independently of the report: the scheme, the value
    # rules on the first pass's document, and every physics setting the
    # translation names as run differently from the namelist.
    expected: set[str] = set()
    if imported.component.status == "refused":
        expected.add("ptype")
    document = first_pass_document(imported)
    options = _option_keys()
    if document is not None:
        for breach in rules.judge(document):
            expected.update(options.get(breach.option, (breach.option,)))
    from cm1core.cm1import import translate as translate_module
    from cm1core.cm1import.coverage import STAND_IN_COMPONENT, _Builder

    builder = _Builder(imported, sounding_path=None, acknowledged=(), seed=None, library=None, environment=False)
    builder.adjust()
    component = STAND_IN_COMPONENT if imported.component.status == "refused" else imported.component.component
    try:
        translation = translate_module.translate(
            builder.effective, component=component, gm1_only=builder.runtime_keys(),
            output={key: builder.effective[key] for key in builder.inventory.fields["param9"]},
            declared=frozenset(builder.stated),
        )
    except translate_module.TranslationError as error:
        match = re.match(r"CM1 (\w+) = ", str(error))
        expected.add("stretch_z" if "reads its levels from a file" in str(error) else match.group(1))
    else:
        expected.update(item["cm1_key"] for item in translation.unrepresentable if item["cm1_key"] != "eqtset")
    missing = expected - refused
    assert not missing, (name, sorted(missing))

    # The first refusal the check meets is the one the scoping recorded.
    results = json.loads(scoping_results().read_text(encoding="utf-8"))[name]
    first = report.check["first_refusal"]
    if results.get("check_err"):
        option = re.search(r"\(option ([^=)]*)=", results["check_err"]).group(1)
        if option == "initialization.random_seed":
            # irandp = 1: the importer gives the run CM1's own random numbers
            # (random_generator = "cm1", settings.toml), so the scoping's
            # refusal is gone -- the same as CM1, no row -- and the check goes
            # on to the next refusal.
            assert report.settings["initialization"]["random_generator"] == "cm1"
            assert "random_seed" not in report.settings["initialization"]
            assert _row(report, "differs", "irandp") is None
            assert first is None or first["option"] != "initialization.random_seed"
        elif option == "initialization.iinit" and int(builder.effective.get("iinit", 1)) == 7:
            # The scoping ran before iinit = 7 (CM1's tropical-cyclone vortex,
            # init3d.F:697-1256) was transcribed; it is admitted now, so the
            # check goes on to the deck's next refusal.
            assert _row(report, "refused", "iinit") is None
            assert first is None or first["option"] != "initialization.iinit"
        else:
            assert first is not None and first["option"] == option, (name, first, option)
            assert set(options.get(option, ())) <= refused
    else:
        # The scoping's translator stopped before check: stretch_x = 2 is not a
        # boolean, and stretch_z = 3 reads its levels from a file.
        key = {"hurricane_axisymmetric": "stretch_x", "les_StratoCuDrizzle": "stretch_z"}[name]
        row = _row(report, "refused", key)
        assert row is not None and row.stage == "translate"


@pytest.mark.parametrize("name", RUNNABLE_SHIPPED)
def test_the_runnable_shipped_decks_run(name: str) -> None:
    report = _report(shipped_deck(name))
    assert report.exit_code == 0 and report.check["first_refusal"] is None
    assert report.toml is not None


# ---------------------------------------------------------------------------
# 2. All refusals in one pass
# ---------------------------------------------------------------------------


def test_three_refusals_in_one_run(tmp_path: Path) -> None:
    deck = edited(SUPERCELL, tmp_path, {"stretch_x": "1", "ptype": "26", "isnd": "15"})
    report = _report(deck)
    assert report.exit_code == 2
    rows = {row.keys[0]: row for row in report.category("refused")}
    assert {"stretch_x", "ptype", "isnd"} <= set(rows)
    assert not any(rows[key].after for key in ("stretch_x", "ptype", "isnd"))
    # The stretch lengths CM1 reads only with stretch_x on travel with it.
    assert {"dx_inner", "dx_outer", "nos_x_len", "tot_x_len"} <= set(rows["stretch_x"].covers)
    assert rows["ptype"].text.startswith("ptype = 26 is NSSL")
    assert rows["isnd"].text.startswith(
        "isnd = 15 (the DYCOMS-II stratocumulus test) runs only with testcase = 4 or 5"
    )
    assert rows["isnd"].plain and rows["ptype"].plain and rows["stretch_x"].plain
    assert "ptype = 5 (Morrison) standing in" in " ".join(report.notes)


def test_a_check_across_settings_hidden_behind_a_value_refusal_is_found(tmp_path: Path) -> None:
    # psolver = 7 is a value refusal; an event interval that is not a whole
    # number of steps is a check across settings.  Both are reported.
    deck = edited(SUPERCELL, tmp_path, {"psolver": "7", "tapfrq": "1003.0"})
    report = _report(deck)
    keys = _refused_keys(report)
    assert {"psolver", "tapfrq"} <= keys
    assert report.exit_code == 2


# ---------------------------------------------------------------------------
# 3. CM1 r22.0
# ---------------------------------------------------------------------------


def test_r22_supercell_lists_every_r22_key_with_its_class() -> None:
    report = _report(R22_SUPERCELL)
    r22 = [row for row in report.not_r21 if row["from"] == "CM1 r22.0"]
    assert len(r22) == 32
    assert {row["class"] for row in r22} <= {"physics", "output", "cosmetic"}
    same_physics = [row for row in r22 if row["class"] == "physics" and row["category"] == "same"]
    assert same_physics and all(row["text"] == "Same as CM1 r21.1" for row in same_physics)
    assert {"cfl_limit", "ramp_down", "gamma_cp"} <= {row["key"] for row in same_physics}
    # r22.0's own supercell sets the Morrison breakup diameter to 400e-6 m;
    # r21.1 hard-codes 300e-6 m.  That changes the physics: refused.
    assert report.exit_code == 2
    assert [row.keys for row in report.category("refused")] == [("brkup_db",)]
    # output_vort means staggered vorticity in r22.0: an output difference.
    assert [row.keys for row in report.category("not_read")] == [("output_vort",)]


def test_r22_physics_and_output_changes(tmp_path: Path) -> None:
    base = {"brkup_db": "300.0e-6", "output_vort": "0"}
    assert _report(edited(R22_SUPERCELL, tmp_path, base, "a")).exit_code == 0
    # A physics key changed where r22.0 reads it: refused.
    report = _report(edited(R22_SUPERCELL, tmp_path, {**base, "cfl_limit": "1.5", "adapt_dt": "1"}, "b"))
    assert report.exit_code == 2 and _row(report, "refused", "cfl_limit") is not None
    # cfl_limit is read only with adapt_dt = 1; with adapt_dt = 0 r22.0 does
    # not read it and it is r21.1's behaviour.
    report = _report(edited(R22_SUPERCELL, tmp_path, {**base, "cfl_limit": "1.5"}, "c"))
    assert report.exit_code == 0
    assert [row["category"] for row in report.not_r21 if row["key"] == "cfl_limit"] == ["same"]
    # Only an output key changed: runs after an acknowledgment.
    report = _report(edited(R22_SUPERCELL, tmp_path, {**base, "output_temperature": "1"}, "d"))
    assert report.exit_code == 3 and report.missing_acknowledgments == ("output-keys",)
    acknowledged = _report(edited(R22_SUPERCELL, tmp_path, {**base, "output_temperature": "1"}, "e"),
                           acknowledge=("output-keys",))
    assert acknowledged.exit_code == 0 and acknowledged.toml is not None


def test_an_older_release_is_read_through_its_aliases() -> None:
    report = _report(R18_RUN)
    assert report.version == "before r21.1"
    readings = {item["key"]: item for item in report.aliases}
    # pertcor became lspgrad in CM1 r19; iturb became cm1setup/sgsmodel.
    assert readings["pertcor"]["status"] == "read" and "lspgrad" in readings["pertcor"]["assigns"]
    assert readings["iturb"]["status"] == "read"
    assert readings["nsound"]["status"] == "ignored"
    # r18 predates weno_order and doimpl: r21.1 holds 0 for both, CM1 r21.1
    # itself stops on weno_order = 0 with WENO on, and GM1 runs doimpl = 1 only.
    for key in ("weno_order", "doimpl"):
        row = _row(report, "refused", key)
        assert row is not None and row.defaulted and "(not set" in row.value
    assert report.category("not_read")  # r18's output keys that r21.1 dropped


# ---------------------------------------------------------------------------
# 4. The stock deck's report
# ---------------------------------------------------------------------------


def _golden_text(report, source: str) -> str:
    text = report.text()
    return text.replace(report.namelist["path"], source)


def _golden_json(report) -> dict:
    payload = json.loads(json.dumps(report.as_dict(), default=str))
    payload["namelist"]["path"] = "supercell/namelist.input"
    payload["command"] = None
    return payload


def test_the_stock_deck_report_matches_its_golden_file() -> None:
    report = _report(SUPERCELL)
    assert report.exit_code == 0
    text = _golden_text(report, "supercell/namelist.input")
    assert text == (GOLDEN / "supercell.import-report.txt").read_text(encoding="utf-8")
    assert _golden_json(report) == json.loads((GOLDEN / "supercell.import-report.json").read_text(encoding="utf-8"))


def test_the_stock_deck_carries_its_ratios_and_no_ramp_row() -> None:
    from cm1core.cm1import.coverage import stock_deck_ratios

    report = _report(SUPERCELL)
    ptype = _row(report, "same", "ptype")
    assert ptype is not None and "0.99-1.05" in ptype.text and ptype.expand == "ratios"
    record = stock_deck_ratios()["schemes"]["morrison-mp10"]
    assert report.comparison["rows"] == record["rows"]
    ratios = {row["key"]: row["ratio"] for row in report.comparison["rows"]}
    assert ratios["wmax"] == pytest.approx(1.005, abs=5e-4)
    assert ratios["zvort_max"] == pytest.approx(1.046, abs=5e-4)
    assert report.comparison["all_inside_band"] and report.comparison["all_inside_target"]
    # GM1 binds CM1's SGS ramp: ramp_sgs runs as CM1 does.
    for category in ("refused", "not_read", "differs", "not_checked"):
        assert _row(report, category, "ramp_sgs") is None
    assert _row(report, "same", "ramp_sgs") is not None


@pytest.mark.parametrize("kdiv, shown", [("0.10", False), ("0.20", True)])
def test_kdiv_is_not_yet_checked_only_off_cm1s_value(kdiv: str, shown: bool, tmp_path: Path) -> None:
    report = _report(edited(SUPERCELL, tmp_path, {"kdiv": kdiv}))
    assert (_row(report, "not_checked", "kdiv") is not None) is shown
    assert report.exit_code == 0


def test_squall_line_shows_the_periodic_row_while_the_closure_caveat_stands() -> None:
    from cm1core.boundary_validation_register import register

    caveat = register()["wbc/ebc/sbc/nbc=1 (periodic)"]["caveat"] or ""
    report = _report(SQUALL_LINE)
    row = _row(report, "not_checked", "sbc")
    if "SGS closure" in caveat:
        assert row is not None and row.keys == ("sbc", "nbc") and "turbtend.F" in row.text
    else:
        assert row is None
    assert report.exit_code == 0


def test_the_comparison_record_is_the_evidence() -> None:
    import subprocess
    import sys

    from cm1import_support import ROOT

    evidence = ROOT / "evidence" / "oracle" / "supercell-fidelity" / "storm-morrison.json"
    if not evidence.is_file():
        pytest.skip("the supercell-fidelity evidence is not in this checkout")
    completed = subprocess.run([sys.executable, str(ROOT / "tools" / "generate_stock_deck_ratios.py"), "--check"],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# The other rows
# ---------------------------------------------------------------------------


def test_thompson_runs_only_at_arwens_droplet_number_and_after_acknowledgment(tmp_path: Path) -> None:
    report = _report(edited(SUPERCELL, tmp_path, {"ptype": "3"}, "a"))
    row = _row(report, "refused", "nt_c")
    assert report.exit_code == 2 and row is not None and row.fix == "set nt_c = 100.0"
    report = _report(edited(SUPERCELL, tmp_path, {"ptype": "3", "nt_c": "100.0"}, "b"))
    assert report.exit_code == 3 and report.missing_acknowledgments == ("thompson",)
    assert _row(report, "not_checked", "ptype").acknowledge == "thompson"
    report = _report(edited(SUPERCELL, tmp_path, {"ptype": "3", "nt_c": "100.0"}, "c"), acknowledge=("thompson",))
    assert report.exit_code == 0


def test_wsm6_runs_differently_with_its_ratios(tmp_path: Path) -> None:
    report = _report(edited(SUPERCELL, tmp_path, {"ptype": "7"}))
    row = _row(report, "differs", "ptype")
    assert row is not None and "eqtset = 1" in row.text and row.expand == "ratios"
    assert report.comparison["scheme"] == "wsm6-mp6" and not report.comparison["gates_release"]
    assert report.exit_code == 0


def test_cm1s_own_post_read_changes_are_applied(tmp_path: Path) -> None:
    # imove = 0 with a frame speed: CM1 zeroes the speed (param.F:701-702).
    report = _report(edited(SUPERCELL, tmp_path, {"imove": "0"}, "a"))
    assert report.exit_code == 0
    assert any("umove = 12.5 is read as umove = 0.0" in note for note in report.notes)
    # umove = 0.4 is the 32-bit real CM1 holds.
    report = _report(edited(SUPERCELL, tmp_path, {"umove": "0.4"}, "b"))
    assert report.exit_code == 0
    assert any(note.startswith("umove = 0.4 is read as umove = 0.4000000059604645") for note in report.notes)


def test_an_unknown_physics_key_is_refused_and_an_unknown_group_is_not_read(tmp_path: Path) -> None:
    text = SUPERCELL.read_text(encoding="ascii")
    text = text.replace(" &param2\n", " &param2\n my_new_switch = 1,\n", 1)
    text += "\n &mygroup\n mine = 3,\n /\n"
    path = tmp_path / "namelist.input"
    path.write_text(text, encoding="ascii")
    report = _report(path)
    assert _row(report, "refused", "my_new_switch").code == "CM1-IMPORT-UNKNOWN-KEY"
    assert _row(report, "not_read", "mine").acknowledge == "output-keys"
    assert report.exit_code == 2


def test_a_missing_sounding_is_refused_and_a_given_one_is_checked(tmp_path: Path) -> None:
    deck = edited(SUPERCELL, tmp_path, {"isnd": "7", "iwnd": "0"})
    report = _report(deck)
    row = _row(report, "refused", "input_sounding")
    assert row is not None and row.code == "CM1-IMPORT-SOUNDING-MISSING"
    sounding = tmp_path / "input_sounding"
    # Winds equal to the namelist's frame speed (12.5, 3.0): the storm stays put in the domain.
    lines = ["1000.0 300.0 14.0"] + [f"{z:.1f} {300 + z / 250:.2f} {max(14 - z / 1000, 0.0):.3f} 12.5 3.0"
                                     for z in range(250, 25001, 250)]
    sounding.write_text("\n".join(lines) + "\n", encoding="ascii")
    report = _report(deck, sounding=sounding)
    assert report.exit_code == 0, [row.as_dict() for row in report.category("refused")]
    assert report.sounding["levels"] == 100 and report.sounding["problems"] == []


# ---------------------------------------------------------------------------
# Invariants the GUI relies on
# ---------------------------------------------------------------------------


def _sources(tmp_path: Path) -> list[tuple[str, Path]]:
    from cm1import_support import pristine_config_files

    sources = [("supercell", SUPERCELL), ("squall_line", SQUALL_LINE), ("run", RUN_DEFAULT),
               ("r22", R22_SUPERCELL), ("r18", R18_RUN)]
    sources += corpus_namelists(tmp_path)
    if pristine_config_files() is not None:
        sources += [(name, shipped_deck(name)) for name in shipped_decks()]
    return sources


def test_every_stated_setting_lands_in_exactly_one_category(tmp_path: Path) -> None:
    from cm1core.cm1import import defaults

    inventory = defaults.load()
    for name, path in _sources(tmp_path):
        report = _report(path)
        placed: dict[str, int] = {}
        for row in report.rows:
            for key in row.keys + row.covers:
                placed[key] = placed.get(key, 0) + 1
        imported_stated = set(report_stated(path))
        for key in imported_stated:
            if not inventory.groups_of(key):
                continue
            assert placed.get(key) == 1, (name, key, placed.get(key))
        twice = [key for key, count in placed.items() if count > 1]
        assert not twice, (name, twice)


def report_stated(path: Path) -> frozenset[str]:
    from cm1core.cm1import.coverage import import_for_report

    return import_for_report(path, tool=namelist_tool()).stated


def test_the_json_has_the_check_screens_shape(tmp_path: Path) -> None:
    for name, path in _sources(tmp_path)[:8]:
        payload = json.loads(json.dumps(_report(path).as_dict(), default=str))
        assert payload["schema"] == "gm1.cm1-import-report/v1"
        assert [section["id"] for section in payload["sections"]] == [
            "refused", "not_read", "differs", "not_checked", "not_needed", "same"]
        assert payload["exit"] in (0, 2, 3)
        assert (payload["exit"] == 2) == bool(payload["sections"][0]["rows"])
        for section in payload["sections"]:
            for row in section["rows"]:
                assert {"key", "keys", "value", "text", "stage"} <= set(row)
                if section["id"] == "refused":
                    assert row["fix"] and row["plain"], (name, row)
        same = payload["sections"][-1]
        assert same["count"] >= sum(family["n"] for family in same["families"])
        for family in same["families"]:
            assert family["evidence"]


def test_every_evidence_path_exists(tmp_path: Path) -> None:
    from cm1import_support import ROOT
    from cm1core.cm1import.coverage import FAMILIES

    paths = {evidence for _, evidence in FAMILIES.values()}
    for name, path in _sources(tmp_path)[:8]:
        paths.update(row.evidence for row in _report(path).rows if row.evidence)
    for evidence in sorted(paths):
        # "records/..." names a record inside the package.
        where = ROOT / "src" / "cm1core" / evidence if evidence.startswith("records/") else ROOT / evidence
        assert where.exists(), evidence


# ---------------------------------------------------------------------------
# Guided and advanced modes
# ---------------------------------------------------------------------------


def test_guided_mode_offers_only_what_runs_as_cm1_does(tmp_path: Path) -> None:
    stock = _report(SUPERCELL).as_dict()["guided"]
    assert stock == {"offered": True, "why_not": None, "reasons": []}
    # The squall line waits for the periodic-sides fix: not yet checked, so not offered, with the plain reason.
    squall = _report(SQUALL_LINE).as_dict()["guided"]
    if _row(_report(SQUALL_LINE), "not_checked", "sbc") is not None:
        assert not squall["offered"] and squall["why_not"].startswith("Repeating sides")
    refused = _report(edited(SUPERCELL, tmp_path, {"ptype": "26"})).as_dict()["guided"]
    assert not refused["offered"] and refused["why_not"] == "NSSL microphysics is not available in GM1."
    # WSM6 runs (exit 0) but measurably differently from CM1's ptype = 7: not a beginner's run.
    wsm6 = _report(edited(SUPERCELL, tmp_path, {"ptype": "7"}, "w")).as_dict()
    assert wsm6["exit"] == 0 and not wsm6["guided"]["offered"]
    assert wsm6["guided"]["why_not"].startswith("Rain and hail: WSM6.")
    # The random stream of irandp = 1 differs from CM1's, and does not block.
    assert _report(edited(SUPERCELL, tmp_path, {"irandp": "1"}, "r")).as_dict()["guided"]["offered"]


@pytest.mark.parametrize("rstfrq", ["-999.9", "-1.0", "-3600.0"])
def test_a_negative_rstfrq_keeps_one_restart_at_the_end(rstfrq: str, tmp_path: Path) -> None:
    # CM1 writes no restart for rstfrq < 0; GM1 keeps one at the end, so a
    # cadence the namelist never asked for (|rstfrq|) is not checked or refused.
    report = _report(edited(SUPERCELL, tmp_path, {"rstfrq": rstfrq}))
    assert report.exit_code == 0, [row.as_dict() for row in report.category("refused")]
    row = _row(report, "differs", "rstfrq")
    assert row is not None and "one restart at the end (timax = 7200.0 s)" in row.text
    assert "restart_interval_s = 7200.0" in report.toml


def test_a_diagnostic_refusal_names_the_cm1_output_words(tmp_path: Path) -> None:
    # A 4 km deep model cannot hold CM1's 2-5 km updraft-helicity layer.
    report = _report(edited(SUPERCELL, tmp_path, {"nz": "8"}))
    row = next(row for row in report.category("refused") if row.code == "CM1-CFG-DIAGNOSTIC-SELECTION")
    assert row.keys == ("output_uh", "output_shs") and row.value == "1"
    assert row.plain == "Updraft helicity needs a model at least 5 km deep."


def test_a_bad_grid_spacing_is_refused_by_its_own_name(tmp_path: Path) -> None:
    report = _report(edited(SUPERCELL, tmp_path, {"dx": "-1000.0"}))
    rows = {row.keys[0]: row for row in report.category("refused")}
    assert "dx" in rows and "dtl" not in rows and not rows["dx"].after
    assert rows["dx"].fix and rows["dx"].plain and rows["dx"].plain_fix
    # dx is set aside at CM1's shipped 2000 m to check the rest; a check that
    # reads that stand-in (dx against dy) is marked, not presented as a fault of dy.
    assert "dy" not in rows or rows["dy"].after


def test_the_gui_can_grey_out_the_values_gm1_refuses() -> None:
    from cm1core.cm1import.rules import admitted_cm1

    assert admitted_cm1("psolver") == (3,)
    assert admitted_cm1("iinit") == (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13)
    assert admitted_cm1("testcase") == (0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert admitted_cm1("stretch_x") == (0,)
    assert admitted_cm1("kdiv") is None  # a real GM1 admits in [0, 1], not a choice


def test_the_text_report_lists_every_setting_not_in_r21() -> None:
    text = _report(R22_SUPERCELL).text()
    block = text.split("Settings not in CM1 r21.1 (32)\n", 1)[1]
    assert "cfl_limit = 1.0 (CM1 r22.0, physics): Same as CM1 r21.1" in block
    assert "brkup_db = 400.0e-6 (CM1 r22.0, physics): Refused" in block


@pytest.mark.parametrize("cecd", ["2", "3"])
def test_a_wind_dependent_drag_coefficient_is_refused_up_front(cecd: str, tmp_path: Path) -> None:
    """With the surface layer on (bbc = 3, sfcmodel = 1), only CM1's constant drag coefficient runs in 0.1.1.

    The run itself would otherwise stop before its first step, after an import that reported it runnable."""

    report = _report(edited(SUPERCELL, tmp_path, {"bbc": "3", "sfcmodel": "1", "cecd": cecd}))
    assert report.exit_code != 0 and report.verdict == "refused"
    rows = [row for row in report.category("refused") if "cecd" in row.keys]
    assert len(rows) == 1, [row.as_dict() for row in report.category("refused")]
    row = rows[0]
    assert row.code == "CM1-IMPORT-CECD-UNSUPPORTED" and row.value == cecd
    assert row.text == f"Only cecd = 1 (constant drag coefficient) is implemented in 0.1.1; cecd = {cecd} is not."
    assert row.fix == "Set cecd = 1."
    assert row.plain and row.plain_fix
    assert report.guided()["offered"] is False


def test_the_constant_drag_coefficient_still_imports_as_a_run(tmp_path: Path) -> None:
    report = _report(edited(SUPERCELL, tmp_path, {"bbc": "3", "sfcmodel": "1", "cecd": "1"}))
    assert not report.category("refused"), [row.as_dict() for row in report.category("refused")]
    assert report.exit_code == 0
    # Without the surface layer CM1 never reads cecd, so the stock deck's cecd = 3 is not refused.
    assert not _report(SUPERCELL).category("refused")
