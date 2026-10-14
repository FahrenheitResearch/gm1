"""The hurricane presets, the rule that offers a storm whose settings are not all compared with CM1 yet, and a
hurricane run's own words and looks.

1. ``hurricane-8km`` is CM1 r21.1's ``hurricane_3d_cpm`` namelist with the listed words changed, on CM1's own
   sounding, and ``hurricane-4km`` is the same on a 4 km grid; the fit never makes either smaller than 1000 km
   or shorter than 3 days, and says why.  Both were run on a GPU and are offered, each with what GM1's own run
   showed and, at 8 km, how much weaker it is than CM1's run of the same settings.
2. A setting that runs but is not yet compared with CM1 no longer holds a preset back: the guided check lists it
   (``not_compared``) and the card says "Runs, not yet compared with CM1: ...".  A refusal still blocks, and so
   does an acknowledgment that was not given.  A preset nobody has run on a GPU stays withheld by its own
   ``preset.json`` ("not run yet").
3. Customize offers a hurricane its storm time in days, its picture interval and its sea surface temperature, and
   nothing made for a thunderstorm.
4. A hurricane run's pictures say what a hurricane is doing (no "No storm yet"), its quick-looks are its wind and
   its pressure at the lowest level, and its time series leads with its lowest pressure.

CPU only: the real preset code on a fake GPU, the fake runner, and frames of CM1's own 24 x 24 storm.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import warnings

import numpy as np
import pytest

from cm1core.cm1import import coverage
from cm1core.cm1run import customize as C
from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.cm1run.fit import set_namelist_values
from cm1core.gui import guided as G
from cm1core.gui import storm
from cm1core.gui.copy_lint import lint_text
from cm1import_support import namelist_tool
from gui_support import start_server, stop_server

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "gpuinfo"
HURRICANE = "hurricane-8km"
HURRICANE_4KM = "hurricane-4km"
#: The fixed ranges of a hurricane's wind and pressure pictures, and what its pictures' footer says of CM1.
WIND_RANGE, PRESSURE_RANGE = 80.0, 70.0
FOOTER_WORDS = "compared with CM1 on this 8 km case (measured)"
FOOTER_NOT_COMPARED = "not compared with CM1 at these settings"
STRINGS = json.loads((ROOT / "src/cm1core/gui/copy/guided.json").read_text(encoding="utf-8"))["strings"]
#: What GM1's and CM1's hurricane runs did (the packaged record the cards and guides say their numbers from).
RUNS = json.loads((ROOT / "src/cm1core/records/GM1-HURRICANE-RUNS.json").read_text(encoding="utf-8"))
WORDS = RUNS["words"]
NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="the quick-looks need matplotlib")
#: The three settings of the hurricane preset that run and are not yet compared with CM1, as the card says them.
#: Its levels (three equal 50 m layers at the ground, 500 m ones at the top) are not among them: CM1 ran the 8 km
#: hurricane on these levels with the same vortex, drag and Morrison (records/GM1-HURRICANE-RUNS.json), so the
#: stretched-levels row leaves the presets' own case out (unvalidated.HURRICANE_PRESET_COLUMN).
NOT_COMPARED = [
    "The large-scale pressure gradient has not been checked against CM1 in a storm.",
    "Repeating sides have not been fully checked against CM1 yet.",
    "Ground friction has been compared with CM1 in one storm on one card only: CM1's supercell on an RTX 4090 "
    "(measured).",
]
#: What the guided check says of ground friction: what was measured (the 0.1.2 release notes' comparison), and no more.
FRICTION = NOT_COMPARED[-1]
#: The words changed from CM1's hurricane_3d_cpm namelist, and what each is in the preset.
CHANGED = {
    "nx": 128, "ny": 128, "dx": 8000.0, "dy": 8000.0, "stretch_x": 0, "stretch_y": 0, "timax": 345600.0,
    "adapt_dt": 0, "dtl": 7.5, "tapfrq": 10800.0, "rstfrq": -345600.0, "cm1setup": 1, "ipbl": 0, "sgsmodel": 1, "tconfig": 1, "horizturb": 0,
    "ndcnst": 250.0, "cecd": 1, "cnstcd": 0.0024, "cnstce": 0.0012, "str_bot": 150.0, "str_top": 4000.0,
    "output_format": 2,
}


def _fake_gpu(monkeypatch, tmp_path, fixture: str, wsl2: bool = False) -> None:
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(FIXTURES / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(FIXTURES / f"{fixture}.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1" if wsl2 else "0")
    monkeypatch.setenv("GPUWM_CM1_HOST_FIXTURE", str(FIXTURES / "host-64gb-1tb.json"))
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))


@pytest.fixture(autouse=True)
def _machine(tmp_path, monkeypatch):
    namelist_tool()
    _fake_gpu(monkeypatch, tmp_path, "rtx5090-32gb")
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


def offered_copy(tmp_path: Path, *preset_ids: str) -> Path:
    """The shipped presets with ``preset_ids`` no longer withheld by their own preset.json (as after a GPU run)."""

    root = tmp_path / "offered-presets"
    shutil.copytree(P.presets_root(), root, ignore=shutil.ignore_patterns("__pycache__"))
    for preset_id in preset_ids:
        path = root / preset_id / "preset.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("offered", None)
        record.pop("not_offered_reason", None)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return root


def _guided(namelist_text: str, tmp_path: Path, name: str, sounding: Path | None = None) -> dict:
    """The import report's guided block for a namelist, as ``presets.examine`` reads it."""

    path = tmp_path / f"{name}.namelist.input"
    path.write_text(namelist_text, encoding="ascii")
    report = coverage.build_report(coverage.import_for_report(path), environment=False, sounding=sounding)
    return {**report.guided(), "exit": report.exit_code}


def _squeezed(namelist_text: str) -> str:
    """A namelist with its white space gone and CM1's ``dtl = 10.000`` written as the number it is."""

    return re.sub(r"dtl=10\.0+,", "dtl=10.0,", re.sub(r"\s", "", namelist_text))


# ------------------------------------------------------------------ the preset


def test_the_hurricane_preset_is_cm1s_hurricane_case_with_the_listed_words_changed():
    preset = P.load(HURRICANE)
    record = preset.record
    assert record["name"] == "Hurricane, quick look, 8 km" and record["group"] == "other"
    assert record["storm_kind"] == "hurricane" and record["budget"] == "1h"
    source = record["source"]
    assert source["file"] == "CM1 r21.1 run/config_files/hurricane_3d_cpm/namelist.input"
    stock = ROOT / "tests/data/cm1import/r21.1-hurricane_3d_cpm.namelist.input"
    assert source["sha256"] == hashlib.sha256(stock.read_bytes()).hexdigest()
    assert {change["key"]: change["to"] for change in source["changes"]} == CHANGED
    assert all(change["why"] for change in source["changes"])
    # Only those words differ from CM1's file: writing them back gives CM1's file again.
    back = set_namelist_values(preset.namelist_text, {change["key"]: change["from"] for change in source["changes"]})
    assert _squeezed(back) == _squeezed(stock.read_text(encoding="ascii"))
    # CM1's own sounding, byte for byte.
    sounding = ROOT / "tests/data/cm1import/r21.1-hurricane_3d_cpm.input_sounding"
    assert (preset.folder / "input_sounding").read_bytes() == sounding.read_bytes()
    assert record["sounding"]["sha256"] == hashlib.sha256(sounding.read_bytes()).hexdigest()
    assert record["sounding"]["source"] == "CM1 r21.1 run/config_files/hurricane_3d_cpm/input_sounding"
    # The floors the fit keeps, and the honest label on the card and the budget screen.
    assert record["min_width_km"] == {"default": 1000} and record["min_duration_s"] == 259200
    assert record["ladder_m"] == []
    assert "not CM1's stock hurricane" in record["line"] and "not CM1's stock hurricane" in record["note"]
    for field in ("name", "line", "note", "floors_why"):
        assert lint_text(record[field], where=field) == []
    # It was run on a GPU, so nothing withholds it, and the card says how it compares with CM1's run.
    assert "offered" not in record and "not_offered_reason" not in record
    assert "measured" in record["note"] and "measured" in record["eye_forms"]["basis"]
    # What GM1's own run of the preset showed (measured on an RTX 4090), beside CM1's run of the same changed
    # settings on its own adaptive time step: said exactly so, with the numbers formatted from the record.
    assert WORDS["card_8km_measured"] in record["note"] and WORDS["card_step"] in record["note"]
    assert ("CM1 r21.1 run with the same changed settings but its own adaptive time step (from 10 s, as CM1's "
            "file has it)") in record["note"] and "a fixed 7.5 s time step" in record["note"]
    assert "stops early or makes a much weaker storm, a known issue" in record["note"]
    gm1, cm1 = RUNS["runs"]["gm1-8km"], RUNS["runs"]["cm1-8km"]
    assert gm1["ran_to_h"] == 96.0 and gm1["ended"] == "completed" and gm1["engine"].startswith("ArWen 2.8.9")
    low = gm1["summary"]["lowest_pressure"]
    assert f"{low['hpa']:.1f} hPa" in record["note"] and f"hour {low['hour']}" in record["note"]
    assert f"{cm1['summary']['lowest_pressure']['hpa']:.1f} hPa" in record["note"]
    assert record["ran_to_s"] == 345600.0
    assert record["eye_forms"]["basis"].startswith("GM1's run of this preset on an RTX 4090")
    assert storm.eye_forms(HURRICANE, G.presets_dir()) == 172800.0
    # The card's picture is from that run, of the look the preset names, and says so.
    picture = json.loads((preset.folder / "picture.json").read_text(encoding="utf-8"))
    assert picture["run"]["preset"] == HURRICANE and picture["run"]["gpu"] == "NVIDIA GeForce RTX 4090"
    assert picture["run"]["look"] == record["picture_look"] == "wind-lowest"
    assert picture["run"]["grid"]["dx_m"] == 8000.0 and picture["run"]["model_time_s"] == 324000.0
    assert picture["caption"] == ("Wind at the lowest level, T+90 h (3 d 18 h), from a run at 8 km on an RTX 4090. "
                                  "Simulated.")
    # It is the run the record's numbers come from: the whole 4 days, started when that run started.
    assert picture["run"]["grid"]["duration_s"] == 345600.0
    assert picture["run"]["made_utc"][:16] == RUNS["runs"]["gm1-8km"]["started_utc"][:16]
    assert lint_text(picture["caption"], where="caption") == []
    facts = P.examine(preset)["facts"]
    assert (facts["nx"], facts["ny"], facts["nz"], facts["dx_m"]) == (128, 128, 59, 8000.0)
    assert facts["domain_km"] == [1024.0, 1024.0] and facts["repeating_sides"] == ["wbc", "ebc", "sbc", "nbc"]
    assert facts["words"] == "8 km grid · 1024 × 1024 km · 4 days of storm time"


def test_the_hurricanes_take_a_fixed_time_step_because_the_adaptive_one_goes_wrong():
    """With CM1's adaptive time step GM1's 8 km hurricane parts from CM1's once rain forms and two runs of three
    stopped early (hours 20.8 and 29.1); with a fixed 7.5 s step it stays close to CM1's until the last day
    (measured).  Both presets carry the fixed step and say why."""

    for preset_id in (HURRICANE, HURRICANE_4KM):
        preset = P.load(preset_id)
        changes = {change["key"]: change for change in preset.record["source"]["changes"]}
        assert (changes["adapt_dt"]["from"], changes["adapt_dt"]["to"]) == (1, 0), preset_id
        assert "stops early or makes a much weaker storm (measured)" in changes["adapt_dt"]["why"]
        assert (changes["dtl"]["from"], changes["dtl"]["to"]) == (10.0, 7.5), preset_id
        effective = C._effective(preset)
        assert float(effective["dtl"]) == 7.5 and int(effective["adapt_dt"]) == 0, preset_id
        # Every picture interval Customize offers lands on a whole number of 7.5 s steps.
        assert all(float(hours) * 3600.0 % 7.5 == 0.0 for hours in C.HURRICANE_PICTURES_H)
        assert "a fixed 7.5 s where CM1's file has one that changes with the storm" in preset.record["note"]
        assert WORDS["known_issue_adaptive_step"] in preset.record["note"]


def test_the_import_report_says_the_adaptive_step_runs_differently_on_stretched_levels(tmp_path):
    """A namelist that keeps CM1's adaptive step on stretched levels (CM1's hurricane case as an Advanced user
    would import it) is told so, with the fix; a uniform column with the adaptive step gets no such row."""

    def adaptive_row(text: str, name: str, sounding: Path | None):
        path = tmp_path / f"{name}.namelist.input"
        path.write_text(text, encoding="ascii")
        report = coverage.build_report(coverage.import_for_report(path), environment=False, sounding=sounding)
        rows = [row for section in report.as_dict()["sections"] for row in section["rows"]
                if "adapt_dt" in (row.get("keys") or [row.get("key")]) and section["id"] == "differs"]
        return report, rows

    hurricane = P.load(HURRICANE)
    report, rows = adaptive_row(set_namelist_values(hurricane.namelist_text, {"adapt_dt": 1}), "adaptive",
                                hurricane.sounding_path)
    assert report.exit_code == 0 and len(rows) == 1, rows
    assert "runs stopped early" in rows[0]["text"] and "adapt_dt = 0" in rows[0]["fix"]
    # Its evidence is a file that ships with GM1, not a development tool.
    assert rows[0]["evidence"] == "records/GM1-HURRICANE-RUNS.json"
    from importlib import resources

    assert resources.files("cm1core").joinpath(rows[0]["evidence"]).is_file()
    assert lint_text(rows[0]["plain"], where="adapt_dt") == []
    # The preset itself (a fixed step) has no such row, and neither has a uniform column with the adaptive step.
    assert adaptive_row(hurricane.namelist_text, "fixed", hurricane.sounding_path)[1] == []
    supercell = P.load("supercell-1km")
    assert adaptive_row(set_namelist_values(supercell.namelist_text, {"adapt_dt": 1}), "uniform",
                        supercell.sounding_path)[1] == []


def test_the_4_km_hurricane_is_the_same_case_on_a_finer_grid_and_says_it_is_not_compared():
    preset = P.load(HURRICANE_4KM)
    record = preset.record
    assert record["name"] == "Hurricane, 4 km" and record["group"] == "other" and record["budget"] == "overnight"
    assert record["storm_kind"] == "hurricane" and "offered" not in record
    quick = {change["key"]: change["to"] for change in P.load(HURRICANE).record["source"]["changes"]}
    finer = {change["key"]: change["to"] for change in record["source"]["changes"]}
    # CM1's own 4 km spacing is kept, so dx and dy are not changed words; everything else is the quick look's.
    assert finer == {**{key: value for key, value in quick.items() if key not in ("dx", "dy")}, "nx": 256, "ny": 256}
    stock = ROOT / "tests/data/cm1import/r21.1-hurricane_3d_cpm.namelist.input"
    back = set_namelist_values(preset.namelist_text, {change["key"]: change["from"] for change in record["source"]["changes"]})
    assert _squeezed(back) == _squeezed(stock.read_text(encoding="ascii"))
    assert (preset.folder / "input_sounding").read_bytes() == (P.load(HURRICANE).folder / "input_sounding").read_bytes()
    examined = P.examine(preset)
    assert examined["offered"] is True and examined["not_compared"] == NOT_COMPARED
    facts = examined["facts"]
    assert (facts["nx"], facts["ny"], facts["nz"], facts["dx_m"]) == (256, 256, 59, 4000.0)
    assert facts["words"] == "4 km grid · 1024 × 1024 km · 4 days of storm time"
    assert "Not compared with CM1 at 4 km." in record["line"]
    assert "CM1 has not been run with these settings at 4 km" in record["note"]
    assert record["min_width_km"] == {"default": 1000} and record["min_duration_s"] == 259200
    for field in ("name", "line", "note", "floors_why"):
        assert lint_text(record[field], where=field) == []
    # What GM1's own run of it showed (measured on an RTX 4090; CM1 has not been run at 4 km), and which engine
    # those numbers come from: its whole 4 days were run on the engine GM1 now runs on.
    assert WORDS["card_4km_measured"] in record["note"]
    assert RUNS["four_km_basis"] == "gm1-4km" and RUNS["runs"]["gm1-4km"]["summary"]["last_hour"] == 96
    assert "with ArWen 2.8.9 (gpuwm 2.8.9) ran its whole 4 days" in record["note"]
    assert "it was strongest at hour 90" in record["note"]
    assert record["ran_to_s"] == 345600.0
    assert record["eye_forms"]["basis"].startswith("GM1's run of this preset on an RTX 4090")
    picture = json.loads((preset.folder / "picture.json").read_text(encoding="utf-8"))
    assert picture["run"]["preset"] == HURRICANE_4KM and picture["run"]["grid"]["dx_m"] == 4000.0
    assert picture["run"]["look"] == record["picture_look"] == "wind-lowest"
    assert picture["caption"].startswith("Wind at the lowest level, T+90 h (3 d 18 h), from a run at 4 km")
    assert picture["run"]["made_utc"][:16] == RUNS["runs"]["gm1-4km"]["started_utc"][:16]
    # Its eye is clear by the end of the second day, as the quick look's is, and the picture notes say that hour.
    eye = storm.eye_forms(HURRICANE_4KM, G.presets_dir())
    assert eye == record["eye_forms"]["to_s"] and 108000.0 < eye <= 259200.0
    notes = storm.notes({"preset": HURRICANE_4KM}, _index([(0.0, -35.0, -35.0), (eye - 10800.0, 50.0, 55.0),
                                                           (eye, 50.0, 55.0)]), G.presets_dir(), STRINGS)
    assert f"the eye is clear by {storm.day_words(eye)}" in notes[eye - 10800.0] and eye not in notes
    assert C.options(preset)["knobs"][0]["choices"][0]["value"] == 4000.0


def test_an_adaptive_hurricanes_time_is_priced_with_the_steps_it_takes(tmp_path):
    """CM1's adaptive step grows to twice dtl; GM1's two adaptive hurricane runs took 23128 and 24621 steps
    where timax / dtl is 46080 (measured), so a hurricane on the adaptive step is priced with that many fewer
    steps and its upper bound keeps every declared step.  The presets take a fixed step and are priced step for
    step, as every other preset is."""

    from cm1core.cm1run import estimate as E

    assert E.ADAPTIVE_HURRICANE_STEP_OF_DTL == 1.85
    fixed = P.fit(P.load(HURRICANE), "1h")
    assert fixed.offered, fixed.reason
    assert fixed.estimate.steps == 46080 and fixed.estimate.dt_s == 7.5
    assert "adapts" not in fixed.estimate.wall.basis
    assert fixed.estimate.wall.high < 1.6 * fixed.estimate.wall.value
    preset = P.load(HURRICANE)
    folder = tmp_path / "adaptive"
    folder.mkdir()
    (folder / "namelist.input").write_text(set_namelist_values(preset.namelist_text, {"adapt_dt": 1}),
                                           encoding="ascii")
    shutil.copy(preset.folder / "input_sounding", folder / "input_sounding")
    priced = E.price(E.deck_from_namelist(folder / "namelist.input", sounding=folder / "input_sounding"))
    assert priced.steps == 46080
    assert "the time step adapts (adapt_dt = 1)" in priced.wall.basis and "24909 longer ones" in priced.wall.basis
    assert priced.wall.high > 1.6 * priced.wall.value > 1.6 * priced.wall.low


def test_settings_not_yet_compared_with_cm1_are_listed_and_do_not_hold_a_preset_back(tmp_path):
    shipped = P.examine(P.load(HURRICANE))
    assert shipped["import_exit"] == 0 and shipped["not_compared"] == NOT_COMPARED
    # It has been run on a GPU, so it is offered, with its unchecked settings said.
    assert shipped["offered"] is True and shipped["why_not"] is None and shipped["reasons"] == []
    # A preset nobody has run on a GPU yet is withheld by its own preset.json, with the plain reason.
    squall = P.examine(P.load("squall-line-1km"))
    assert squall["offered"] is False and squall["import_exit"] == 0
    assert squall["why_not"] == "Not offered yet: not run yet. Nobody has run this storm on a GPU as it is set up here."
    after_a_run = P.examine(P.load("squall-line-1km", offered_copy(tmp_path, "squall-line-1km")))
    assert after_a_run["offered"] is True and after_a_run["why_not"] is None and after_a_run["reasons"] == []
    assert after_a_run["not_compared"] == [NOT_COMPARED[1]]
    said = P.not_compared_words(NOT_COMPARED)
    assert said == "Runs, not yet compared with CM1: " + " ".join(NOT_COMPARED)
    assert P.not_compared_words([]) is None
    assert lint_text(said, where="not_compared") == []
    assert STRINGS["gallery.not_compared"]["text"] == P.NOT_COMPARED_LEAD
    row = next(row for row in P.list_record()["presets"] if row["id"] == HURRICANE)
    assert row["not_compared"] == NOT_COMPARED and row["storm_kind"] == "hurricane" and row["offered"] is True
    # A preset with nothing unchecked says nothing.
    plain = next(row for row in P.list_record()["presets"] if row["id"] == "supercell-1km")
    assert plain["not_compared"] == [] and plain["storm_kind"] is None and plain["offered"] is True


def test_a_refusal_and_a_missing_acknowledgment_still_block(tmp_path):
    hurricane = P.load(HURRICANE)
    # CM1's own wind-dependent ocean drag is a true refusal: it blocks, whatever else is only unchecked.
    refused = _guided(set_namelist_values(hurricane.namelist_text, {"cecd": 3}), tmp_path, "cecd3",
                      hurricane.sounding_path)
    assert refused["exit"] != 0 and refused["offered"] is False
    assert any("drag" in reason.lower() for reason in refused["reasons"]), refused["reasons"]
    assert not set(refused["reasons"]) & set(NOT_COMPARED)
    # CM1's file as it is is refused too (its sideways-stretched grid, its boundary layer scheme, its drag), though
    # its levels, which stretch from the ground, now run.
    stock = (ROOT / "tests/data/cm1import/r21.1-hurricane_3d_cpm.namelist.input").read_text(encoding="ascii")
    as_shipped = _guided(stock, tmp_path, "stock", hurricane.sounding_path)
    assert as_shipped["exit"] != 0 and as_shipped["offered"] is False and len(as_shipped["reasons"]) >= 5
    supercell = P.load("supercell-1km")
    # Thompson needs an acknowledgment guided mode never gives: its "not yet checked" row still blocks.
    thompson = _guided(set_namelist_values(supercell.namelist_text, {"ptype": 3}), tmp_path, "thompson",
                       supercell.sounding_path)
    assert thompson["exit"] != 0 and thompson["offered"] is False
    assert any(reason.startswith("Rain and hail: Thompson.") for reason in thompson["reasons"])
    # Ground friction runs and needs no acknowledgment: offered, and said.
    friction = _guided(set_namelist_values(supercell.namelist_text, {"bbc": 3, "sfcmodel": 1, "cecd": 1}), tmp_path,
                       "friction", supercell.sounding_path)
    assert friction["exit"] == 0 and friction["offered"] is True and friction["reasons"] == []
    assert friction["not_compared"] == [FRICTION]


def test_only_the_presets_never_run_on_a_gpu_are_withheld_for_it():
    """What the rule changed: the two squall lines and the two hurricanes pass it.  The hurricanes have been run
    on a GPU and are offered; each squall line is kept back by its own preset.json until it has been run; every
    other preset is offered or withheld as before."""

    unchecked = {}
    for preset in P.load_all():
        examined = P.examine(preset)
        if examined["not_compared"]:
            unchecked[preset.id] = examined
    assert sorted(unchecked) == ["hurricane-4km", "hurricane-8km", "squall-line-1km", "squall-line-250m"]
    for preset_id, examined in unchecked.items():
        assert examined["import_exit"] == 0
        if preset_id.startswith("hurricane"):
            assert examined["offered"] is True and examined["reasons"] == [], preset_id
            continue
        assert examined["offered"] is False
        assert examined["reasons"] == [P.load(preset_id).record["not_offered_reason"]]
        assert examined["why_not"].startswith("Not offered yet: not run yet.")
    offered = [preset.id for preset in P.load_all() if P.examine(preset)["offered"]]
    # The 100 m supercell is offered since 0.1.4.
    assert len(offered) == 21 and "supercell-100m" in offered


# ------------------------------------------------------------------ the fit


def test_the_fit_never_makes_a_hurricane_smaller_than_its_swirl_or_shorter_than_its_eye(tmp_path, monkeypatch):
    preset = P.load(HURRICANE, offered_copy(tmp_path, HURRICANE))
    hour = P.fit(preset, "1h")
    record = P.fit_record(preset, hour, command="test")
    assert hour.offered, record["reason"]
    assert record["facts"]["domain_km"] == [1024.0, 1024.0] and record["facts"]["dx_m"] == 8000.0
    assert record["facts"]["duration_s"] == 345600.0 and "4 days of storm time" in record["summary"]
    assert record["not_compared"] == NOT_COMPARED
    assert "Runs, not yet compared with CM1: " in P._fit_text(record)
    # Ten minutes is too short for four days of storm, and a hurricane is not cut below three: it says so.
    short = P.fit(preset, "10m")
    refused = P.fit_record(preset, short, command="test")
    assert not short.offered and refused["reason_kind"] == "time"
    assert "pick a longer time than about 10 minutes" in refused["reason"]
    assert refused["reason"].endswith(preset.record["floors_why"])
    assert "1000 km" in preset.record["floors_why"] and "3 days" in preset.record["floors_why"]
    # A slower card: the storm time is shortened, never below 3 days, and the area is never touched.
    _fake_gpu(monkeypatch, tmp_path, "rtx4060-8gb-desktop")
    slower = P.fit(preset, "1h")
    if slower.offered:
        facts = P.fit_record(preset, slower, command="test")["facts"]
        assert facts["domain_km"] == [1024.0, 1024.0] and 259200.0 <= facts["duration_s"] <= 345600.0
    else:
        assert P.fit_record(preset, slower, command="test")["reason"].endswith(preset.record["floors_why"])
        # The card does not strand it there: it fits the next time, which the gallery opens it on and names.
        overnight = P.fit(preset, "overnight")
        assert overnight.offered and P.fit_record(preset, overnight, command="test")["budget"] == "overnight"
    gallery = (ROOT / "src/cm1core/gui/static/guided/gallery.js").read_text(encoding="utf-8")
    assert 'const tries = [p.budget, "overnight"]' in gallery
    assert "fit.budget !== own.budget" in gallery and 't("gallery.needs_budget"' in gallery
    needs = STRINGS["gallery.needs_budget"]["text"]
    assert needs == "Too long for {own} on your GPU, so it opens on {budget}." and lint_text(needs, where="needs") == []


def test_a_fresh_install_prices_both_hurricanes_on_an_rtx_4090_at_their_measured_speed(tmp_path, monkeypatch):
    """No measurement of its own (an empty cache): the record's two points price the cards, about 15 minutes at
    8 km and about 50 at 4 km, where the two-term model said about 37 and about 70."""

    _fake_gpu(monkeypatch, tmp_path, "rtx4090-24gb")
    root = offered_copy(tmp_path, HURRICANE, HURRICANE_4KM)
    for preset_id, budget, seconds, minutes in ((HURRICANE, "1h", 0.0198, (15.0, 17.0)),
                                                (HURRICANE_4KM, "overnight", 0.0642, (49.0, 52.0))):
        preset = P.load(preset_id, root)
        result = P.fit(preset, budget)
        record = P.fit_record(preset, result, command="test")
        assert result.offered, record["reason"]
        assert record["facts"]["duration_s"] == 345600.0
        step = result.estimate.step
        assert record["estimate"]["wall_rate_label"] == "measured"
        assert step["regime"] == "this-grid" and step["seconds_per_step"] == pytest.approx(seconds)
        assert step["label_used"] == "measured" and "on one computer" in step["basis_used"]
        assert minutes[0] * 60.0 <= record["estimate"]["wall_s"] <= minutes[1] * 60.0, record["estimate"]["wall_s"]
    # A withheld preset is not fitted at all.
    withheld = P.fit(P.load("squall-line-1km"), "1h")
    assert not withheld.offered and withheld.reason.startswith("Not offered yet: not run yet.")


def test_the_storm_track_check_says_a_hurricane_stays_in_the_middle():
    track = P.storm_track_for(P.load(HURRICANE))
    if track is None:
        pytest.skip("the storm-track check needs the initializer library")
    assert track["status"] == "not-evaluated-doubly-periodic" and track["meets_edge_rule"] is True
    assert "swirl at the centre of the area" in track["reason"] and track["iinit"] == 7


# ------------------------------------------------------------------ Customize


def test_customize_offers_a_hurricane_only_what_makes_sense_for_it(tmp_path):
    preset = P.load(HURRICANE, offered_copy(tmp_path, HURRICANE))
    offer = C.options(preset)
    knobs = {knob["id"]: knob for knob in offer["knobs"]}
    # No area, levels, push, bubble, bumps, scheme, friction, rotation or parcels: a hurricane needs its ocean
    # surface and the rotation, and its area is its swirl's.
    assert list(knobs) == ["grid", "time", "pictures", "sea"]
    assert [choice["value"] for choice in knobs["grid"]["choices"]] == [8000.0]
    assert [choice["value"] / 1440.0 for choice in knobs["time"]["choices"]] == [3.0, 4.0, 5.0, 6.0]
    # GM1 has run this storm for 4 days: a longer storm time is offered and says "not run yet".
    assert [choice.get("note") for choice in knobs["time"]["choices"]] == [None, None, "not run yet", "not run yet"]
    assert knobs["time"]["ran_to_min"] == 5760.0 and "not run yet" in knobs["time"]["note"]
    assert lint_text(knobs["time"]["note"], where="time note") == []
    # The listing says a hurricane's times in days and hours, each with the minutes --custom takes.
    listing = P._options_text(offer)
    assert "time      now 4 days (5760 min); 3 days (4320), 4 days (5760), 5 days (7200, not run yet)" in listing
    assert "pictures  now 3 h (180 min); 1 h (60), 3 h (180), 6 h (360), 12 h (720)" in listing
    assert "the value in minutes, which --custom takes" in listing and " min; 4320, 5760" not in listing
    assert offer["storm_kind"] == "hurricane"
    # A thunderstorm preset's listing is what it was (minutes), and it carries no kind.
    plain = C.options(P.load("supercell-1km"))
    assert "storm_kind" not in plain and "now 120 min;" in P._options_text(plain)
    longer = C.apply(preset, {"time": 7200.0}).record["customized"][0]["sentence"]
    assert longer == ("5 days of storm time instead of 4 days. GM1 has run this storm for 4 days; the time after "
                      "that is not run yet.")
    assert knobs["time"]["default"] == 5760.0 and knobs["pictures"]["default"] == 180.0
    assert knobs["pictures"]["all"] == [60.0, 180.0, 360.0, 720.0]
    assert knobs["sea"]["default"] == 28.0 and knobs["sea"]["unit"] == "°C"
    assert [choice["value"] for choice in knobs["sea"]["choices"]] == [26.0, 27.0, 28.0, 29.0, 30.0, 31.0]
    assert offer["air"] == {"available": False, "why": C.AIR_FIXED}
    assert C.parse(["sea=30", "time=4320", "pictures=360"]) == {"sea": 30.0, "time": 4320.0, "pictures": 360.0}
    changed = C.apply(preset, {"sea": 30.0, "time": 4320.0, "pictures": 360.0})
    effective = C._effective(changed)
    assert float(effective["tsk0"]) == pytest.approx(303.15) and float(effective["timax"]) == 259200.0
    assert float(effective["tapfrq"]) == 21600.0 and float(effective["dx"]) == 8000.0
    sentences = {entry["knob"]: entry["sentence"] for entry in changed.record["customized"]}
    assert sentences["time"] == "3 days of storm time instead of 4 days."
    assert sentences["pictures"] == "A saved picture every 6 h instead of every 3 h."
    assert sentences["sea"].startswith("A sea surface at 30 °C instead of 28 °C.")
    assert changed.record["pinned"] == ["pictures", "time"]
    assert C.custom_args(changed.record["custom"]) == ["--custom", "time=4320", "--custom", "pictures=360",
                                                         "--custom", "sea=30"]
    examined = P.examine(changed)
    assert examined["offered"] is True and examined["not_compared"] == NOT_COMPARED
    assert P.fit(changed, "1h").offered
    # What a hurricane does not offer is refused in plain words.
    with pytest.raises(C.CustomError, match="is not one of the choices"):
        C.apply(preset, {"grid": 4000.0})
    with pytest.raises(C.CustomError, match="is not one of the choices"):
        C.apply(preset, {"sea": 35.0})
    with pytest.raises(C.CustomError, match="has no rotation setting"):
        C.apply(preset, {"rotation": "off"})
    with pytest.raises(C.CustomError, match="is not one of the choices"):
        C.apply(preset, {"pictures": 5.0})
    # A thunderstorm preset has no sea to set.
    with pytest.raises(C.CustomError, match="has no sea setting"):
        C.apply(P.load("supercell-1km"), {"sea": 30.0})


def test_a_hurricane_refuses_cms_adaptive_time_step_with_the_measured_reason_and_the_advanced_route(capsys, tmp_path):
    """``preset fit|new hurricane-* --set adapt_dt=1`` is the setting measured to stop early or make a much weaker
    storm: guided mode refuses it, says why with the numbers, and names Advanced as the way to try it."""

    from cm1core.cli import main

    for preset_id in (HURRICANE, HURRICANE_4KM):
        preset = P.load(preset_id)
        with pytest.raises(C.CustomError) as caught:
            C.apply(preset, {}, {"adapt_dt": 1})
        text = str(caught.value)
        assert text.startswith("--set adapt_dt is not taken for a hurricane.")
        assert WORDS["known_issue_adaptive_step"] in text and C.HURRICANE_ADVANCED in text
        assert "Advanced" in text and "gpuwm-cm1 import" in text
        # The fixed step the preset already has is taken; the time step itself is the run's own.
        assert C.apply(preset, {}, {"adapt_dt": 0}).record["customized"][0]["sentence"] == "Set adapt_dt = 0 by hand."
        with pytest.raises(C.CustomError, match="fixed 7.5 s time step is the one the storm was run with"):
            C.apply(preset, {}, {"dtl": 5})
        with pytest.raises(C.CustomError, match="a hurricane preset keeps its area"):
            C.apply(preset, {}, {"nx": 64})
        # The sea temperature by its CM1 word would go round the sea knob's 26 to 31 degrees: refused, naming
        # the knob, the range and the Advanced route.
        with pytest.raises(C.CustomError) as sea:
            C.apply(preset, {}, {"tsk0": 310})
        assert str(sea.value) == C.hurricane_sea_why()
        for words in ("--set tsk0 is not taken for a hurricane", "26 to 31 °C", "--custom sea=C", "301.15",
                      "Advanced", "gpuwm-cm1 import"):
            assert words in str(sea.value), words
        # Even a value inside the range goes through the knob, so the card's sentence and its pin are made.
        with pytest.raises(C.CustomError, match="--custom sea=C"):
            C.apply(preset, {}, {"tsk0": 301.15})
        # CM1's words for the levels: only this column has been run.
        for key, value in (("stretch_z", 0), ("ztop", 20000.0), ("dz_bot", 100.0), ("dz_top", 400.0),
                           ("str_bot", 0.0), ("str_top", 3000.0)):
            with pytest.raises(C.CustomError) as levels:
                C.apply(preset, {}, {key: value})
            assert str(levels.value) == C.hurricane_levels_why(key)
            assert f"--set {key} is not taken for a hurricane: only these levels have been run" in str(levels.value)
            assert "Advanced" in str(levels.value) and f"set {key} there" in str(levels.value)
        assert set(C.HURRICANE_LEVEL_WORDS) == {"stretch_z", "ztop", "dz_bot", "dz_top", "str_bot", "str_top"}
    for command in ("fit", "new"):
        rc = main(["preset", command, HURRICANE, "--budget", "1h", "--set", "adapt_dt=1"]
                  + (["--root", str(tmp_path / "runs")] if command == "new" else []))
        err = capsys.readouterr().err
        assert rc == 2 and "--set adapt_dt is not taken for a hurricane." in err and "Advanced" in err, (command, err)
    for flags in (["--set", "tsk0=310"], ["--set", "stretch_z=0"]):
        rc = main(["preset", "fit", HURRICANE, "--budget", "1h", *flags])
        err = capsys.readouterr().err
        assert rc == 2 and "is not taken for a hurricane" in err and "Advanced" in err, (flags, err)
    # A thunderstorm preset keeps its own rule for the time step, and takes tsk0 and the level words as before.
    with pytest.raises(C.CustomError, match="--custom grid=M"):
        C.apply(P.load("supercell-1km"), {}, {"dtl": 5})
    C.refuse_set("tsk0", "supercell-1km")
    C.refuse_set("stretch_z", "supercell-1km")


def test_a_hurricane_takes_no_other_air_and_says_so_in_plain_words(capsys, tmp_path):
    """``--instability``, ``--shear`` and ``--sounding`` on a hurricane card are refused before anything is made
    (a made sounding stops below a hurricane's 25 km top, which read as a message about a temporary file)."""

    from cm1core.cli import main

    sounding = tmp_path / "input_sounding"
    sounding.write_text(P.load(HURRICANE).sounding_text, encoding="utf-8")
    for preset_id in (HURRICANE, HURRICANE_4KM):
        for flags in (["--instability", "0.5", "--shear", "0.5"], ["--instability", "0.5"], ["--shear", "0.5"],
                      ["--sounding", str(sounding)], ["--sounding", str(tmp_path / "missing")]):
            for command in ("fit", "new"):
                rc = main(["preset", command, preset_id, *flags, "--budget", "overnight"]
                          + (["--root", str(tmp_path / "runs")] if command == "new" else []))
                err = capsys.readouterr().err
                assert rc == 2 and C.HURRICANE_AIR in err, (preset_id, command, flags, err)
                assert "24000 m" not in err and "gm1-environment" not in err
    assert not (tmp_path / "runs").exists()
    for words in ("--sounding, --instability and --shear are not taken", "--custom sea=C", "Advanced",
                  "gpuwm-cm1 import NAMELIST --sounding FILE"):
        assert words in C.HURRICANE_AIR, words


def test_the_budget_screen_says_a_stretched_column_as_it_is():
    """59 levels from 50 m at the bottom to 500 m aloft, not "59 x 500 m"; an even column says what it said."""

    facts = P.examine(P.load(HURRICANE))["facts"]
    assert (facts["stretch_z"], facts["dz_bottom_m"], facts["dz_top_m"]) == (1, 50.0, 500.0)
    fitted = P.fit_record(P.load(HURRICANE), P.fit(P.load(HURRICANE), "1h"), command="x")["facts"]
    assert (fitted["dz_bottom_m"], fitted["dz_top_m"]) == (50.0, 500.0)
    assert "stretch_z" not in P.examine(P.load("supercell-1km"))["facts"]
    assert STRINGS["budget.levels_stretched"]["text"] == "{count} levels, {bottom} at the bottom to {top} aloft"
    budget = (ROOT / "src/cm1core/gui/static/guided/budget.js").read_text(encoding="utf-8")
    assert 't("budget.levels_stretched"' in budget and "f.dz_bottom_m" in budget


def test_a_hurricane_pictures_footer_says_the_time_in_hours_and_days():
    from cm1core.quicklook import facts as QF

    common = dict(name="x", version="0", version_source="s", scheme="Morrison", component="morrison-mp10",
                  checked="c", checked_source="s", dx_m=8000.0, dy_m=8000.0)
    assert QF.RunFacts(**common, storm_kind="hurricane").footer(t=291600.0)[0].endswith("T+81 h (3 d 9 h)")
    # The time-series sheet says it the same way in its header and its footer.
    from cm1core.quicklook import series as SE

    assert SE.clock_words({"storm_kind": "hurricane"}, 345600.0) == "T+96 h (4 d 0 h)"
    assert SE.clock_words({}, 5400.0) == "T+1:30:00"
    render = (ROOT / "src/cm1core/quicklook/render.py").read_text(encoding="utf-8")
    assert "SE.clock_words(doc, t_now)" in render and "S.fmt_clock(t_now))]" not in render
    assert QF.RunFacts(**common).footer(t=5400.0)[0].endswith("T+1:30:00")


def test_the_hurricane_record_says_what_was_compared_and_is_current():
    """Every number the cards, the guides and the release notes say about the hurricane runs is formatted from the
    record's series (tools/generate_hurricane_runs.py); the guides' blocks are what the record says."""

    assert RUNS["compared"]["cm1"] == ("CM1 r21.1 run with the same changed settings but its own adaptive time step "
                                       "(from 10 s, as CM1's file has it)")
    assert RUNS["compared"]["cm1_words"] == "adapt_dt = 1, dtl = 10"
    assert RUNS["runs"]["cm1-8km"]["time_step"].startswith("adaptive (adapt_dt = 1) from dtl = 10 s")
    for run_id, run in RUNS["runs"].items():
        assert [len(row) for row in run["series"]] == [len(run["columns"])] * len(run["series"]), run_id
        assert all(b[0] - a[0] == 3 for a, b in zip(run["series"], run["series"][1:])), run_id
    assert RUNS["runs"]["gm1-8km"]["series"][-1][0] == 96
    for text in RUNS["words"].values():
        assert lint_text(text, where="hurricane words") == [], text
    tool = ROOT / "tools" / "generate_hurricane_runs.py"
    if not tool.is_file() or not (ROOT / "docs/gui-release/hurricane-2.8.9/runs.json").is_file():
        pytest.skip("the generator and its evidence stay in the development sources")
    import subprocess
    import sys

    done = subprocess.run([sys.executable, str(tool), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr


def test_ground_friction_is_a_customize_choice_that_says_it_is_not_yet_compared():
    preset = P.load("supercell-1km")
    knob = next(knob for knob in C.options(preset)["knobs"] if knob["id"] == "friction")
    assert knob["default"] == "off" and all(choice["available"] for choice in knob["choices"])
    assert ("friction", "on") not in C.GREYED
    examined = P.examine(C.apply(preset, {"friction": "on"}))
    assert examined["offered"] is True
    assert examined["not_compared"] == [FRICTION]
    # The note under the switch says what was measured, with the numbers the 0.1.2 release notes give, and that
    # no other storm and no other card has been compared.
    note = STRINGS["custom.friction_note"]["text"]
    notes = " ".join((ROOT / "docs" / "RELEASE-NOTES.md").read_text(encoding="utf-8").split())
    assert "nine statistics are 0.986 to 1.018 of CM1's and max vertical vorticity is 1.072" in notes
    for words in ("RTX 4090", "0.986 to 1.018", "1.072", "(measured)", "any other storm", "any other card"):
        assert words in note, words
    assert lint_text(note, where="friction note") == []
    # The import report's row gives the same numbers; the schemes guided mode does not offer stay greyed.
    import inspect

    from cm1core.cm1import import register

    row = re.sub(r'"\s*"', "", " ".join(inspect.getsource(register).split()))
    assert "CM1's supercell with cecd = 1 on an RTX 4090" in row
    assert "statistics are 0.986 to 1.018 of CM1's and max vertical vorticity is 1.072 (measured)" in row
    assert "has not been compared with CM1 in a storm" not in row
    scheme = next(knob for knob in C.options(preset)["knobs"] if knob["id"] == "scheme")
    greyed = {choice["value"]: choice for choice in scheme["choices"] if not choice["available"]}
    assert set(greyed) == {"wsm6", "thompson"} and all(choice["why"] for choice in greyed.values())
    for name in ("wsm6", "thompson"):
        with pytest.raises(C.CustomError, match="You can still choose it in Advanced"):
            C.apply(preset, {"scheme": name})
    # The guides say the same: friction is a choice with its measured comparison, the two schemes are greyed.
    quickstart = " ".join((ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8").split())
    assert "(WSM6 or Thompson rain and hail) is greyed" in quickstart
    assert "Ground friction is a choice you can turn on" in quickstart and "0.986 to 1.018" in quickstart
    assert "ground friction) is greyed" not in quickstart


def test_the_release_note_text_names_ground_friction_under_its_own_heading():
    text = (ROOT / "docs" / "RELEASE-NOTES.md").read_text(encoding="utf-8")
    release = text.split("## 0.1.5\n")[1].split("\n## ")[0]
    assert release.index("### What is new") < release.index("### Also in this release: ground friction in Customize")
    assert "Hurricane, quick look, 8 km" in release and "Hurricane, 4 km" in release
    section = " ".join(release.split("### Also in this release: ground friction in Customize")[1].split("###")[0].split())
    for words in ("every storm that has the panel", "RTX 4090 only", "0.986 to 1.018", "1.072",
                  "WSM6 and Thompson stay greyed"):
        assert words in section, words


# ------------------------------------------------------------------ the pictures' words


def _index(rows, *, ended=False):
    """A quick-look index with the reflectivity facts of ``rows``: (t, dbz at 1 km, dbz in the column)."""

    return {"schema": "gm1.quicklook-index/v1", "run_ended": ended, "end_reason": "completed" if ended else None,
            "products": [{"id": "reflectivity", "frames": [
                {"time_s": t, "file": f"reflectivity/t{int(t):06d}.png",
                 "facts": {"dbz_max_1km": one, "dbz_max_column": column}} for t, one, column in rows]}]}


def test_a_hurricane_never_says_no_storm_yet():
    run = {"preset": HURRICANE}
    rows = [(0.0, -35.0, -35.0), (10800.0, -20.0, 3.0), (43200.0, 38.0, 46.0), (129600.0, 48.0, 52.0),
            (172800.0, 52.0, 56.0), (291600.0, 55.0, 58.0)]
    notes = storm.notes(run, _index(rows), G.presets_dir(), STRINGS)
    none = "The starting swirl has no rain yet. Rain cells usually appear around it by about hour 6."
    assert notes[0.0] == none and notes[10800.0] == none
    assert notes[43200.0] == notes[129600.0]
    assert notes[43200.0] == ("Rain cells are forming around the swirl. It looks ragged at first, then tightens: "
                              "in GM1's own run of this storm the eye is clear by hour 48 (the end of day 2).")
    # From the eye's usual time on the pictures speak for themselves.
    assert 172800.0 not in notes and 291600.0 not in notes
    assert not any("No storm" in line for line in notes.values())
    # A run that ended with no rain at all says that, not "no storm formed".
    dry = storm.notes(run, _index([(0.0, -35.0, -35.0), (345600.0, -30.0, -30.0)], ended=True), G.presets_dir(),
                      STRINGS)
    assert set(dry.values()) == {"No rain had formed around the swirl when this run ended."}
    for key in ("storm.hurricane.none", "storm.hurricane.ragged", "storm.hurricane.none_ended"):
        assert lint_text(STRINGS[key]["text"], where=key) == []
    assert storm.storm_kind(HURRICANE, G.presets_dir()) == "hurricane"
    assert storm.storm_kind("supercell-1km", G.presets_dir()) is None
    assert storm.eye_forms(HURRICANE, G.presets_dir()) == 172800.0
    assert storm.day_words(172800.0) == "hour 48 (the end of day 2)" and storm.day_words(108000.0) == "hour 30"
    # A thunderstorm preset keeps its own words.
    usual = storm.notes({"preset": "supercell-1km"}, _index([(0.0, -35.0, -35.0)]), G.presets_dir(), STRINGS)
    assert usual[0.0].startswith("No storm yet: ")


# ------------------------------------------------------------------ the quick-looks


def _hurricane_run(tmp_path: Path) -> Path:
    """Frames of CM1's 24 x 24 storm under a namelist that starts from CM1's hurricane swirl (iinit = 7)."""

    from quicklook_support import NAMELIST_24, run_directory

    text = NAMELIST_24.read_text(encoding="ascii")
    swirl, count = re.subn(r"(?m)^(\s*iinit\s*=\s*)\d+", r"\g<1>7", text)
    assert count == 1
    namelist = tmp_path / "hurricane.namelist.input"
    namelist.write_text(swirl, encoding="ascii")
    return run_directory(tmp_path, namelist=namelist, name="swirl", move=(0.0, 0.0), swath="domain")


@NEEDS_MATPLOTLIB
def test_a_hurricane_run_draws_its_wind_and_pressure_and_leaves_out_the_thunderstorm_looks(tmp_path):
    from cm1core.quicklook import products as PR
    from cm1core.quicklook import render as RD

    run = _hurricane_run(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        summary = RD.render(run)
    assert summary.frames_drawn == 3 and summary.problems == []
    index = json.loads((run / "quicklook" / "index.json").read_text())
    assert index["run"]["storm_kind"] == "hurricane"
    # The index says the times as the pictures do: hours and days, never H:MM:SS.
    assert [frame["clock"] for frame in index["frames"]] == ["T+0.2 h", "T+0.3 h", "T+0.4 h"]
    products = {product["id"]: product for product in index["products"]}
    assert all(frame["clock"].endswith(" h") for product in index["products"] for frame in product["frames"])
    for look in ("reflectivity", "rain", "clouds-from-above", "updraft-slice", "wind-lowest", "pressure-lowest"):
        assert [frame["time_s"] for frame in products[look]["frames"]] == [900.0, 1200.0, 1500.0], look
        for frame in products[look]["frames"]:
            assert (run / "quicklook" / frame["file"]).is_file() and (run / "quicklook" / frame["thumb"]).is_file()
    for look in PR.THUNDERSTORM_ONLY:
        assert products[look]["frames"] == [], look
        assert [row["reason"] for row in products[look]["not_drawn"]] == [PR.THUNDERSTORM_ONLY_WHY], look
    wind = products["wind-lowest"]["frames"][-1]["facts"]
    pressure = products["pressure-lowest"]["frames"][-1]["facts"]
    # The ranges hold GM1's own hurricane runs (at most 70 m/s and 61 hPa below the edge, measured).
    assert wind["range_m_s"] == PR.HURRICANE_WIND_RANGE == WIND_RANGE and wind["height_m"] == 250.0
    assert 0.0 < wind["wind_max_ground_relative"] < WIND_RANGE
    assert max(row[2] for run in RUNS["runs"].values() for row in run["series"]) < WIND_RANGE
    assert max(RUNS["rain_max_cm"].values()) < PR.HURRICANE_RAIN_RANGE
    assert pressure["range_hpa"] == PR.HURRICANE_PRESSURE_RANGE == PRESSURE_RANGE
    assert 900.0 < pressure["p_min_hpa"] <= pressure["p_edge_hpa"] + 5.0
    # The fastest wind is placed from the pressure centre; the lowest pressure from the middle of the area.
    centre = [w - p for w, p in zip(wind["wind_max_from_centre_km"], pressure["p_min_from_centre_km"])]
    assert wind["wind_max_from_pressure_centre_km"] == pytest.approx(centre)
    assert PR._from_centre(20.0, 0.0) == "20 km east of the middle of the area"
    assert PR._from_centre(0.0, -7.0, "the pressure centre") == "7 km south of the pressure centre"
    assert PR._from_centre(0.2, 0.1, "the pressure centre") == "at the pressure centre"
    assert PR.HURRICANE_W_RANGE == 10.0 and PR.W_RANGE == 50.0
    # Each hurricane look shows the whole area and its inner part (half of this small test area: 12 of 24 km),
    # and records the marks and scales it was drawn with.
    reflectivity = products["reflectivity"]["frames"][-1]["facts"]
    rain = products["rain"]["frames"][-1]["facts"]
    assert wind["inner_km"] == pressure["inner_km"] == rain["inner_km"] == reflectivity["inner_km"] == 12.0
    assert wind["hurricane_force_m_s"] == PR.HURRICANE_FORCE == 33.0 and wind["hurricane_force_area_km2"] >= 0.0
    assert pressure["lines_every_hpa"] == PR.HURRICANE_PRESSURE_LINES == 4.0 and pressure["scale"] == "square root"
    assert rain["floor_cm"] == PR.HURRICANE_RAIN_FLOOR == 0.1 and rain["scale"] == "logarithmic"
    assert {"dbz_max_1km", "dbz_max_column", "levels_m"} <= set(reflectivity)
    from types import SimpleNamespace

    assert PR.w_range(SimpleNamespace(hurricane=True)) == 10.0 and PR.w_range(SimpleNamespace(hurricane=False)) == 50.0
    # A hurricane picture's clock reads as Watching and Results do: hours and days, not H:MM:SS.
    assert PR.long_clock(291600.0) == "T+81 h (3 d 9 h)" and PR.long_clock(129600.0) == "T+36 h (1 d 12 h)"
    assert PR.long_clock(21600.0) == "T+6 h" and PR.long_clock(1500.0) == "T+0.4 h" and PR.long_clock(0.0) == "T+0 h"
    # Every picture of the run says its time that way: the run's facts give the one clock.
    from cm1core.quicklook import environment as EN, film as FI, parcels as PA, skewt as SK, slices as SL, tracks as TR

    for module in (EN, FI, PA, SK, SL, TR):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "ov.clock(qf.t)" not in source and "ov.clock(t)" not in source and "ov.clock(f.t)" not in source
        assert '"clock", None) or S.fmt_clock)(' in source, module.__name__
    # Every picture's footer says how this case compares with CM1, not the scheme's thunderstorm comparison.
    from cm1core.quicklook import facts as QF

    # This test's run is a 24 x 24 grid, which no CM1 hurricane run has: its footer says it is not compared.
    assert index["run"]["checked"] == QF.HURRICANE_NOT_COMPARED == FOOTER_NOT_COMPARED
    assert QF.HURRICANE_CHECKED == FOOTER_WORDS and "supercell" not in QF.HURRICANE_CHECKED
    assert pressure["p_drop_max_hpa"] == pytest.approx(pressure["p_edge_hpa"] - pressure["p_min_hpa"], abs=1e-3)
    assert wind["p_min_hpa"] == pytest.approx(pressure["p_min_hpa"])
    assert products["rain"]["frames"][-1]["facts"]["range_cm"] == 400.0 == PR.HURRICANE_RAIN_RANGE
    # The rain range holds the most rain GM1's runs made in one cell (measured).
    assert max(RUNS["rain_max_cm"].values()) <= PR.HURRICANE_RAIN_RANGE
    # The time series leads with the lowest pressure and the fastest wind, and has no thunderstorm chart.
    series = json.loads((run / "quicklook" / "series.json").read_text())
    keys = [chart["key"] for chart in series["charts"]]
    assert keys[:2] == ["prs_sfc_min_hpa", "wind_sfc_max"] and series["storm_kind"] == "hurricane"
    assert not {"uh_max", "thpert_sfc_min", "coldpool_area_km2_lt_minus1K", "zeta_sfc_max"} & set(keys)
    assert series["charts"][0]["zero"] is False and series["charts"][0]["unit"] == "hPa"
    assert "not sea level" in series["charts"][0]["label"]
    assert (run / "quicklook" / "timeseries.png").is_file()
    # The Results screen's More looks row leaves those looks out for a hurricane run (no buttons that only say
    # they are not drawn), and keeps them for a thunderstorm run.
    from cm1core.gui import looks as LK

    assert LK.THUNDERSTORM_ONLY == PR.THUNDERSTORM_ONLY
    in_row = {look["id"] for look in LK.load_copy()["more_looks"]} & set(PR.THUNDERSTORM_ONLY)
    assert in_row == {"surface", "hail-track", "spin-track"}
    ended = {"state": "finished"}
    shown = [row["id"] for row in LK.more_looks(run.name, run, index, ended, films_now={})]
    assert "rain" in shown and not set(shown) & in_row, shown
    # A hurricane's Customize has no parcel choice, so a hurricane run that followed no parcels has no parcel
    # button telling the user to turn them on there.
    assert "parcels" not in shown
    assert "parcels" not in {knob["id"] for knob in C.options(P.load(HURRICANE))["knobs"]}
    thunderstorm = {**index, "run": {key: value for key, value in index["run"].items() if key != "storm_kind"}}
    kept = {row["id"] for row in LK.more_looks(run.name, run, thunderstorm, ended, films_now={})}
    assert in_row | {"parcels"} <= kept


def test_a_hurricane_pictures_colours_lines_and_inner_panel():
    """The pieces the hurricane looks are drawn with: honest fixed ranges on scales that show structure."""

    from cm1core.quicklook import products as PR
    from cm1core.quicklook import style as S

    # The preset's 1024 km area: the inner panel is the middle 500 km, the same size on screen as the whole area.
    whole, zoom, inner = PR.hurricane_boxes((1024000.0, 1024000.0))
    assert inner == PR.HURRICANE_INNER == 500000.0 and (zoom.w, zoom.h) == (whole.w, whole.h)
    assert (zoom.mx0, zoom.my0, zoom.span) == (262000.0, 262000.0, 500000.0) and whole.span == 1024000.0
    assert zoom.x > whole.x + whole.w
    # A smaller area shows its middle half, and says that width.
    assert PR.hurricane_boxes((300000.0, 300000.0))[2] == 150000.0
    # Wind: every ramp rises in lightness (the strongest part is the brightest), the colours part from about
    # 10 m/s, and hurricane force is where they turn from violet to red.
    for name in ("hurricane_wind", "hurricane_rain", "hurricane_pressure"):
        assert S.RAMPS[name].check() == [], name
    ramp = S.RAMPS["hurricane_wind"]
    assert (ramp.vmin, ramp.vmax) == (0.0, PR.HURRICANE_WIND_RANGE)
    light = lambda speed: float(S.lightness(ramp.colours(np.array([speed]))[0]))  # noqa: E731
    assert light(0.0) < 5.0 and light(10.0) > light(0.0) + 12.0 and light(20.0) > light(10.0) + 8.0
    assert light(33.0) > light(20.0) + 6.0 and light(62.0) > 80.0 > light(45.0)
    red, green, blue = ramp.colours(np.array([PR.HURRICANE_FORCE]))[0]
    assert red > 0.7 > blue > green, (red, green, blue)
    cool = ramp.colours(np.array([15.0]))[0]
    assert cool[2] > cool[0] and cool[2] > cool[1]
    # Rain: logarithmic from 1 mm to the range's top, so each tenfold step is the same step along the colours.
    position = PR.rain_position(np.array([0.01, 0.1, 1.0, 10.0, 100.0, 400.0, 900.0]))
    assert position[0] == position[1] == 0.0 and position[-2] == position[-1] == 1.0
    steps = np.diff(position[1:5])
    assert np.allclose(steps, steps[0]) and 0.25 < steps[0] < 0.30
    # Pressure: the square root of the drop's share of the range, so 5 hPa of 70 is already a quarter along.
    drop = PR.pressure_position(np.array([-3.0, 0.0, 5.0, 17.5, 70.0, 90.0]))
    assert list(drop[:2]) == [0.0, 0.0] and drop[3] == pytest.approx(0.5) and list(drop[-2:]) == [1.0, 1.0]
    assert 0.25 < drop[2] < 0.28
    # The lines: a cone of pressure falling half an hPa a pixel towards its middle, so one line every 8 pixels.
    y, x = np.mgrid[0:200, 0:200].astype(np.float64)
    radius = np.hypot(x - 100.0, y - 100.0)
    lines = PR.line_alpha(940.0 + 0.5 * radius, every=4.0)
    row = lines[100, 100:190]
    crossings = [n for n in range(1, len(row) - 1) if row[n] > 0.5 and row[n] >= row[n - 1] and row[n] > row[n + 1]]
    assert crossings and all(b - a == 8 for a, b in zip(crossings, crossings[1:])), crossings
    assert all((940.0 + 0.5 * n) % 4.0 == 0.0 for n in crossings)
    # ... left out where they would touch (a slope of 4 hPa a pixel) and where the field is nearly flat.
    assert float(PR.line_alpha(4.0 * radius, every=4.0).max()) == 0.0
    assert float(PR.line_alpha(1000.0 + 0.001 * radius, every=4.0).max()) == 0.0
    assert float(PR.line_alpha(np.full((50, 50), 1000.0), every=4.0).max()) == 0.0
    # One line at one value (hurricane force on the wind picture): on the 33 m/s circle and nowhere else.
    one = PR.line_alpha(radius, PR.HURRICANE_FORCE, width=1.6)
    assert one[100, 133] > 0.9 and one[100, 140] == 0.0 and one[100, 120] == 0.0
    # The words a hurricane's Results panels say match what is drawn.
    copy = json.loads((ROOT / "src/cm1core/gui/copy/looks.json").read_text(encoding="utf-8"))
    said = {look["id"]: look for look in copy["looks"] + copy["more_looks"]}
    assert "33 m/s ground-relative" in said["wind-lowest"]["text"] and "80 m/s ground-relative" in said["wind-lowest"]["text"]
    assert "one every 4 hPa" in said["pressure-lowest"]["text"] and "70 hPa" in said["pressure-lowest"]["text"]
    assert "start at 1 mm and run to 400 cm" in said["rain"]["text_hurricane"]
    for look in ("wind-lowest", "pressure-lowest"):
        assert lint_text(said[look]["text"], where=look) == []
    for look in ("rain", "reflectivity"):
        assert lint_text(said[look]["text_hurricane"], where=look) == []


def test_a_hurricane_footer_claims_the_cm1_comparison_only_for_the_settings_compared(tmp_path):
    """GM1 and CM1 have both run the 8 km preset as shipped (measured); a run with those settings and any storm
    time or picture interval says so in its footer, and a run with another grid, sea, time step or drag does not."""

    from cm1core.cm1out import settings as CS
    from cm1core.quicklook import facts as QF

    def words(name: str, **changes) -> str:
        path = tmp_path / f"{name}.namelist.input"
        path.write_text(set_namelist_values(P.load(HURRICANE).namelist_text, changes) if changes
                        else P.load(HURRICANE).namelist_text, encoding="ascii")
        return QF.hurricane_checked(CS.from_namelist(path))

    assert words("shipped") == QF.HURRICANE_CHECKED
    assert words("three-days-hourly", timax=259200.0, tapfrq=3600.0) == QF.HURRICANE_CHECKED
    for name, change in (("4km", {"nx": 256, "ny": 256, "dx": 4000.0, "dy": 4000.0}), ("sea", {"tsk0": 302.15}),
                         ("adaptive", {"adapt_dt": 1}), ("step", {"dtl": 5.0}), ("drag", {"cnstcd": 0.0015}),
                         ("drag-kind", {"cecd": 2}),
                         ("droplets", {"ndcnst": 100.0})):
        assert words(name, **change) == QF.HURRICANE_NOT_COMPARED, name
    # The levels are among the compared settings too (another column would not build as a namelist here, so the
    # words are changed on the shipped settings).
    shipped = CS.from_namelist(P.load(HURRICANE).namelist_path)

    class Changed:
        def __init__(self, **changes):
            self.changes = changes

        def float(self, key, default=float("nan")):
            return float(self.changes[key]) if key in self.changes else shipped.float(key, default)

    assert QF.hurricane_checked(Changed()) == QF.HURRICANE_CHECKED
    for key, value in (("stretch_z", 0), ("ztop", 20000.0), ("str_bot", 300.0), ("str_top", 5500.0),
                       ("dz_bot", 25.0), ("dz_top", 400.0)):
        assert QF.hurricane_checked(Changed(**{key: value})) == QF.HURRICANE_NOT_COMPARED, key
    assert P.load(HURRICANE_4KM).record["source"]["changes"]  # the 4 km preset is one of those other grids
    assert QF.hurricane_checked(None) == QF.HURRICANE_NOT_COMPARED
    for text in (QF.HURRICANE_CHECKED, QF.HURRICANE_NOT_COMPARED):
        assert lint_text(text, where="footer") == []


def test_a_long_run_is_charted_in_hours_and_a_short_one_in_minutes():
    from cm1core.quicklook import series as SE

    rows = [(t * 10800.0, {"prs_sfc_min_hpa": 1006.0 - t, "wind_sfc_max": 15.0 + t, "wmax": 2.0}) for t in range(33)]
    long_run = SE.series_document(rows, z0=25.0, name="h", reference=None, t_end=None, kind="hurricane")
    assert long_run["time_unit"] == "h" and "hours since the start" in long_run["note"]
    assert [chart["key"] for chart in long_run["charts"]] == ["prs_sfc_min_hpa", "wind_sfc_max", "wmax"]
    # A four-day run is charted in hours from its first picture on, not only once it has run two days.
    early = SE.series_document(rows[:3], z0=25.0, name="h", reference=None, t_end=345600.0, kind="hurricane")
    assert early["time_unit"] == "h"
    short = SE.series_document([(t * 300.0, {"wmax": 20.0, "wmin": -8.0}) for t in range(25)], z0=250.0, name="s",
                               reference=None, t_end=None)
    assert "time_unit" not in short and "storm_kind" not in short and "minutes since the start" in short["note"]
    assert all("zero" not in chart for chart in short["charts"])


def test_the_pages_know_a_hurricane_runs_looks_and_a_long_runs_clock():
    """The page's own rules (the static JavaScript and the server's lists agree with the quick-look worker)."""

    from cm1core.gui import api, looks
    from cm1core.quicklook import products as PR

    static = ROOT / "src/cm1core/gui/static"
    parts = (static / "js/run/parts.js").read_text(encoding="utf-8")
    assert 'export const HURRICANE_FAVOURITES = ["reflectivity", "wind-lowest", "pressure-lowest"];' in parts
    assert api.HURRICANE_LOOKS == ("reflectivity", "wind-lowest", "pressure-lowest")
    assert set(PR.HURRICANE_ONLY) < set(api.HURRICANE_LOOKS) and set(PR.HURRICANE_ONLY) < set(looks.FILM_LOOKS)
    assert not set(api.HURRICANE_LOOKS) & set(PR.THUNDERSTORM_ONLY)
    core = (static / "js/core.js").read_text(encoding="utf-8")
    assert "export function stormClock(t, total)" in core and "const LONG_RUN_S = 172800;" in core
    assert "stormClock(f.time_s || 0)" in parts
    # Continue's typed storm time keeps the one typed form.
    assert "return clock(Math.round(end) + (reached ? 3600 : 0));" in parts
    chart = (static / "js/chart.js").read_text(encoding="utf-8")
    assert 'series.time_unit === "h"' in chart and "Hours since the start on the horizontal axis." in chart
    gallery = (static / "guided/gallery.js").read_text(encoding="utf-8")
    assert 't("gallery.not_compared")' in gallery and "p.not_compared" in gallery
    budget = (static / "guided/budget.js").read_text(encoding="utf-8")
    assert "fit.not_compared" in budget
    customize = (static / "guided/customize.js").read_text(encoding="utf-8")
    assert '"sea"' in customize and "knobs.sea" in customize
    copy = json.loads((ROOT / "src/cm1core/gui/copy/guided.json").read_text(encoding="utf-8"))
    for key in ("custom.what.sea", "custom.sea_note", "custom.friction_note"):
        assert lint_text(copy["strings"][key]["text"], where=key) == []
    assert copy["terms"]["sea"] == {"text": "Sea surface temperature", "hint": "tsk0"}


def test_a_run_on_the_adaptive_time_step_is_not_offered_continue(tmp_path):
    """The runtime refuses to resume a run whose time step adapts (the restart does not carry its storm time),
    so an ended one has no Continue and no "Run it 1 hour longer": the pages say why instead."""

    from cm1core.gui import runstate

    adaptive, fixed = tmp_path / "adaptive", tmp_path / "fixed"
    for folder, value in ((adaptive, "true"), (fixed, "false")):
        folder.mkdir()
        (folder / "deck.toml").write_text(f"[dynamics]{chr(10)}adaptive_timestep = {value}{chr(10)}", encoding="utf-8")
    assert runstate.adaptive_timestep(adaptive) is True and runstate.adaptive_timestep(fixed) is False
    assert runstate.adaptive_timestep(tmp_path / "no-such-run") is False
    ended = {"last_restart": 1, "last_frame": 3}
    assert "resume" in runstate._actions("finished", dict(ended))
    for state in ("finished", "stopped", "paused"):
        actions = runstate._actions(state, {**ended, "no_resume_why": runstate.ADAPTIVE_NO_CONTINUE})
        assert "resume" not in actions and "results" in actions, state
    assert lint_text(runstate.ADAPTIVE_NO_CONTINUE, where="no_resume_why") == []
    assert "cannot be continued or run longer" in runstate.ADAPTIVE_NO_CONTINUE
    static = ROOT / "src/cm1core/gui/static"
    results = (static / "js/run/results.js").read_text(encoding="utf-8")
    assert "else if (st.no_resume_why) longerWhy = st.no_resume_why;" in results
    assert "st.no_resume_why ||" in results
    assert "} else if (st.no_resume_why) {" in (static / "js/run/run.js").read_text(encoding="utf-8")
    assert 'status.get("no_resume_why")' in (ROOT / "src/cm1core/gui/advanced.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------ through the GUI server (the fake runner)


@pytest.fixture()
def live(tmp_path, monkeypatch):
    _fake_gpu(monkeypatch, tmp_path, "rtx3080-wsl2-desktop-4p8", wsl2=True)
    monkeypatch.setenv(P.PRESETS_ENV, str(offered_copy(tmp_path, HURRICANE)))
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        yield server, client, Path(server.root.path)
    finally:
        stop_server(server)


def test_the_gallery_card_the_budget_screen_and_results_of_a_hurricane(live):
    server, client, root = live
    rows = {row["id"]: row for row in client.get("/api/presets").json["presets"]}
    card = rows[HURRICANE]
    assert card["offered"] is True and card["storm_kind"] == "hurricane" and card["group"] == "other"
    assert card["not_compared"] == NOT_COMPARED and "not CM1's stock hurricane" in card["line"]
    assert card["facts"]["words"] == "8 km grid · 1024 × 1024 km · 4 days of storm time"
    assert [knob["id"] for knob in card["customize"]["knobs"]] == ["grid", "time", "pictures", "sea"]
    # The squall lines stay greyed with the plain reason, and say what is not yet compared.
    assert rows["squall-line-1km"]["offered"] is False
    assert rows["squall-line-1km"]["why_not"].startswith("Not offered yet: not run yet.")
    assert rows["squall-line-1km"]["not_compared"] == [NOT_COMPARED[1]]
    fitted = client.post("/api/fit", {"preset": HURRICANE, "budget": "1h", "custom": {"sea": 29}}).json
    assert fitted["offered"], fitted["message"]
    fit = fitted["fit"]
    assert fit["not_compared"] == NOT_COMPARED and fit["facts"]["domain_km"] == [1024.0, 1024.0]
    assert [entry["knob"] for entry in fit["customized"]] == ["sea"]
    assert "--custom sea=29" in fitted["start"]["commands"][0]
    short = client.post("/api/fit", {"preset": HURRICANE, "budget": "10m"}).json
    assert not short["offered"] and short["fit"]["reason_kind"] == "time"
    assert "A hurricane is not made smaller than 1000 km across" in short["message"]
    # The run folder, without starting it: run.json says what kind of storm it is.
    made = client.post("/api/presets/new", {"preset": HURRICANE, "budget": "1h"})
    assert made.status == 200, made.raw
    name = made.json["run"]
    rundir = root / name
    run = json.loads((rundir / "run.json").read_text(encoding="utf-8"))
    assert run["preset"] == HURRICANE and run["guided"]["storm_kind"] == "hurricane"
    # Its pictures, as the quick-look worker indexes a hurricane run (written here by hand: nothing ran).
    times = [0.0, 43200.0, 291600.0]
    dbz = {0.0: (-35.0, -35.0), 43200.0: (38.0, 46.0), 291600.0: (55.0, 58.0)}

    def frames(look, facts):
        return [{"index": n, "time_s": t, "clock": "", "file": f"{look}/t{int(t):07d}.png",
                 "thumb": f"{look}/thumb/t{int(t):07d}.png", "facts": facts(t)} for n, t in enumerate(times)]

    index = {
        "schema": "gm1.quicklook-index/v1", "run_ended": False, "end_reason": None,
        "run": {"name": name, "storm_kind": "hurricane", "dx_m": 8000.0, "lowest_level_m": 25.0,
                "grid": "128 × 128 × 59, dx 8 km"},
        "frames": [{"index": n, "time_s": t} for n, t in enumerate(times)],
        "products": [
            {"id": "reflectivity", "title": "Reflectivity at 1 km",
             "frames": frames("reflectivity", lambda t: {"dbz_max_1km": dbz[t][0], "dbz_max_column": dbz[t][1]})},
            {"id": "spin-near-ground", "title": "Spin near the ground", "frames": [],
             "not_drawn": [{"reason": "This look is made for thunderstorms.", "times_s": times}]},
            {"id": "rain", "title": "Accumulated rain", "frames": frames("rain", lambda t: {"rain_max_cm": 1.0})},
            {"id": "wind-lowest", "title": "Wind at the lowest level",
             "frames": frames("wind-lowest", lambda t: {"wind_max_ground_relative": 15.0 + t / 6000.0})},
            {"id": "pressure-lowest", "title": "Pressure at the lowest level",
             "frames": frames("pressure-lowest", lambda t: {"p_min_hpa": 1006.0 - t / 5700.0})},
        ],
    }
    (rundir / "quicklook").mkdir()
    (rundir / "quicklook" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    results = client.get(f"/api/runs/{name}/results?mode=guided").json
    looks = results["looks"]
    assert [look["id"] for look in looks][:3] == ["reflectivity", "wind-lowest", "pressure-lowest"]
    assert [look["id"] for look in looks if look["favourite"]] == ["reflectivity", "wind-lowest", "pressure-lowest"]
    assert results["quicklook_run"]["storm_kind"] == "hurricane"
    by = {look["id"]: look for look in looks}
    first, second, last = by["reflectivity"]["frames"]
    assert first["storm"] == STRINGS["storm.hurricane.none"]["text"]
    assert second["storm"].startswith("Rain cells are forming around the swirl.") and "hour 48" in second["storm"]
    # That line is said beside a storm that is there, so the picture keeps its strongest value.
    assert second["storm_showing"] is True and "storm_showing" not in first
    assert by["wind-lowest"]["frames"][1]["strongest"] == "22 m/s ground-relative"
    assert "storm" not in last
    assert by["wind-lowest"]["frames"][-1]["strongest"] == "64 m/s ground-relative"
    assert by["pressure-lowest"]["frames"][-1]["strongest"] == "a lowest pressure of 955 hPa"
    guided = results["guided"]
    assert "25 m above the sea" in guided["looks"]["wind-lowest"]["text"]
    # The looks every run has say a hurricane's words: no hook at the back of the storm, an eyewall.
    assert "eyewall" in guided["looks"]["reflectivity"]["text"] and "hook" not in guided["looks"]["reflectivity"]["text"]
    more = {row["id"]: row for row in results["more_looks"]}
    assert "For a hurricane the colours start at 1 mm and run to 400 cm in tenfold steps" in more["rain"]["text"]
    assert "grey line" not in more["rain"]["text"]
    copy_more = {look["id"]: look for look in json.loads((ROOT / "src/cm1core/gui/copy/looks.json")
                                                     .read_text(encoding="utf-8"))["more_looks"]}
    for look in ("swirl", "environment", "rain", "clouds-from-above", "updraft-slice"):
        words = copy_more[look]["text_hurricane"]
        assert "gust front" not in words and "hodograph" not in words and lint_text(words, where=look) == []
    for look in ("wind-lowest", "pressure-lowest", "reflectivity"):
        assert "The left panel is the whole area; the right panel is the part inside the square" in \
            guided["looks"][look]["text"], look
    assert "The white line is hurricane force (33 m/s ground-relative)" in guided["looks"]["wind-lowest"]["text"]
    assert "one every 4 hPa" in guided["looks"]["pressure-lowest"]["text"]
    # A storm of days is run on by a day, and the panel says that day is past what GM1 has run of it.
    longer = guided["run_longer"]
    assert longer["span_s"] == 86400.0 and longer["until_s"] == 345600.0 + 86400.0
    assert longer["note"] == "GM1 has run this storm for 4 days; the time after that is not run yet."
    assert lint_text(longer["note"], where="run longer") == []
    assert guided["facts"]["storm_kind"] == "hurricane"
    assert "8 km grid spacing" in guided["looks"]["pressure-lowest"]["text"]
    assert "4 days of storm time, a saved time every 3 h" in guided["this_run"]
    # The live view carries the kind too, so Watching shows the same three looks.
    live_view = client.get(f"/api/runs/{name}/live").json
    assert live_view["run"]["storm_kind"] == "hurricane"
    assert next(look for look in live_view["looks"] if look["id"] == "reflectivity")["time_s"] == 291600.0


def test_the_fake_runner_gives_a_hurricane_run_its_own_looks_clock_and_charts(live, monkeypatch):
    """Screen work on a computer with no GPU sees a hurricane's real screens: the wind and pressure looks in
    place of the thunderstorm ones, storm time in hours and days, and the pressure and wind charts."""

    from cm1core.gui import fake_runner as F
    from cm1core.gui import looks as LK
    from cm1core.quicklook import style as S
    from gui_support import wait_for

    for seconds in (0, 3600, 5400, 21600, 86400, 129600, 291600, 345600):
        assert F._long_clock(seconds) == S.fmt_long_clock(seconds), seconds
    assert [look for look, _ in F.HURRICANE_PRODUCTS] == ["reflectivity", "wind-lowest", "pressure-lowest", "rain"]
    assert not {look for look, _ in F.HURRICANE_PRODUCTS} & set(LK.THUNDERSTORM_ONLY)
    assert F._fake_hurricane_facts("wind-lowest", 345600.0)["wind_max_ground_relative"] == 59.0
    assert F._fake_hurricane_facts("pressure-lowest", 0.0)["p_min_hpa"] == 1006.0
    server, client, root = live
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.00001")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.1")
    fit = client.post("/api/fit", {"preset": HURRICANE, "budget": "1h"}).json
    name = fit["start"]["name"]
    started = client.post("/api/guided/start", {"preset": HURRICANE, "budget": "1h", "name": name})
    assert started.status == 200 and started.json["ok"], started.raw
    wait_for(lambda: client.get(f"/api/runs/{name}/status").json["state"] == "finished", timeout=180)

    def drawn():
        looks = client.get(f"/api/runs/{name}/results?mode=guided").json.get("looks") or []
        return looks and all(len(look.get("frames") or []) == 33 for look in looks[:3])

    wait_for(drawn, timeout=120)
    index = json.loads((root / name / "quicklook" / "index.json").read_text(encoding="utf-8"))
    assert [product["id"] for product in index["products"]] == [look for look, _ in F.HURRICANE_PRODUCTS]
    assert index["run"]["storm_kind"] == "hurricane" and index["run"]["lowest_level_m"] == 25.0
    clocks = [frame["clock"] for frame in index["frames"]]
    assert clocks[:3] == ["T+0 h", "T+3 h", "T+6 h"] and clocks[-1] == "T+96 h (4 d 0 h)" and len(clocks) == 33
    results = client.get(f"/api/runs/{name}/results?mode=guided").json
    assert [look["id"] for look in results["looks"] if look["favourite"]] == ["reflectivity", "wind-lowest",
                                                                              "pressure-lowest"]
    assert results["quicklook_run"]["storm_kind"] == "hurricane"
    by = {look["id"]: look for look in results["looks"]}
    assert by["wind-lowest"]["frames"][-1]["strongest"] == "59 m/s ground-relative"
    assert by["pressure-lowest"]["frames"][-1]["strongest"] == "a lowest pressure of 954 hPa"
    assert by["reflectivity"]["frames"][0]["storm"] == STRINGS["storm.hurricane.none"]["text"]
    assert "storm" not in by["reflectivity"]["frames"][-1]
    assert not {row["id"] for row in results["more_looks"]} & set(LK.THUNDERSTORM_ONLY)
    assert "25 m above the sea" in results["guided"]["looks"]["wind-lowest"]["text"]
    series = json.loads((root / name / "quicklook" / "series.json").read_text(encoding="utf-8"))
    assert series["storm_kind"] == "hurricane" and series["time_unit"] == "h"
    assert [chart["key"] for chart in series["charts"]][:2] == ["prs_sfc_min_hpa", "wind_sfc_max"]
    assert series["charts"][0]["zero"] is False and series["charts"][0]["gm1"][-1] == [345600, 954.0]
    # The run ended at its 4 days: a day more is offered, and said to be past what GM1 has run.
    longer = results["guided"]["run_longer"]
    assert longer["span_s"] == 86400.0 and longer["until_s"] == 345600.0 + 86400.0 and "not run yet" in longer["note"]
    # A thunderstorm run on the fake runner is what it was: its six looks, H:MM:SS, no kind.
    assert [look for look, _ in F.PRODUCTS][:4] == ["reflectivity", "spin-near-ground", "updraft-5km", "swath"]
    assert F._fake_facts("updraft-5km", 1800.0) == {"level_m": 5000.0, "w_max": 25.0, "fake_runner": True}


def test_the_pages_say_a_hurricanes_own_words_for_its_changes_and_its_longer_run():
    static = ROOT / "src/cm1core/gui/static"
    budget = (static / "guided/budget.js").read_text(encoding="utf-8")
    assert 't(knobsById.sea ? "custom.caption_storm_sea" : "custom.caption_storm")' in budget
    sea = STRINGS["custom.caption_storm_sea"]["text"]
    assert "sea temperature" in sea and "air" not in sea and "push" not in sea and lint_text(sea, where="sea") == []
    assert "the air or the push" in STRINGS["custom.caption_storm"]["text"]
    results = (static / "js/run/results.js").read_text(encoding="utf-8")
    assert 'longer.span_s >= 86400 ? "results.run_longer_day" : "results.run_longer"' in results
    assert "longer.note" in results
    assert STRINGS["results.run_longer_day"]["text"] == "Run it 1 day longer"
    assert STRINGS["results.run_longer"]["text"] == "Run it 1 hour longer"
    assert G.RUN_LONGER_DAY_S == 86400.0 and G.RUN_LONGER_S == 3600.0
    # The group's line names the hurricane first, as its cards come first.
    assert STRINGS["gallery.group.other_line"]["text"].startswith("A hurricane,")
