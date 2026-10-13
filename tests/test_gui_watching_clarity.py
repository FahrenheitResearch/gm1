"""Watching a run, said plainly: the progress block, when the next picture comes, and "no storm yet".

1. The Run screen (guided Watching and advanced Run) opens with a bar a finger wide, the storm time done of the
   whole, the share done and the time left, and when the next picture comes (before the first: when the first
   comes, where the picture will be).  The live view carries the run's picture schedule (``pictures``).
2. A picture with no storm in it yet says so beside it and in "What am I looking at", from the quick-look index's
   own numbers (``gui/storm.py``); a preset whose storm forms at a measured time says when.
3. Every offered preset saves a picture at least every 5 minutes, and its card's picture is a saved time of the
   preset as it is now.

CPU only: the fake runner, the real preset code on a fake GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.gui import guided as G
from cm1core.gui import storm
from cm1core.gui.fake_runner import _fake_facts
from gui_support import start_server, stop_server, wait_for
from cm1import_support import namelist_tool

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "gpuinfo"
STATIC = ROOT / "src" / "cm1core" / "gui" / "static"
STRINGS = json.loads((ROOT / "src" / "cm1core" / "gui" / "copy" / "guided.json").read_text(encoding="utf-8"))["strings"]


def _index(rows, *, ended=False, reason=None):
    """A quick-look index with the reflectivity and updraft facts of ``rows``: (t, dbz 1 km, dbz column, w 5 km)."""

    return {
        "schema": "gm1.quicklook-index/v1",
        "run_ended": ended,
        "end_reason": reason,
        "products": [
            {"id": "reflectivity", "frames": [{"time_s": t, "file": f"reflectivity/t{int(t):06d}.png",
                                               "facts": {"dbz_max_1km": a, "dbz_max_column": b}} for t, a, b, _ in rows]},
            {"id": "updraft-5km", "frames": [{"time_s": t, "file": f"updraft-5km/t{int(t):06d}.png",
                                              "facts": {"level_m": 5000.0, "w_max": w}} for t, _, _, w in rows]},
        ],
    }


#: The 1 km supercell's first half hour, as its quick-looks recorded it (a real run's index).
SUPERCELL_1KM = [(0.0, -35.0, -35.0, 0.0), (300.0, -35.0, -35.0, 0.18), (600.0, -35.0, -35.0, 0.19),
                 (900.0, -34.0, -23.0, 0.3), (1200.0, -28.0, 7.0, 10.3), (1500.0, 2.0, 60.0, 19.3),
                 (1800.0, 52.0, 65.0, 25.1), (2100.0, 58.0, 68.0, 28.6)]


# ------------------------------------------------------------------ no storm yet: the words and when


def test_quiet_air_is_no_storm_a_storm_aloft_is_forming_and_nothing_after_the_first_echo_at_1_km():
    states = storm.states(_index(SUPERCELL_1KM + [(2400.0, -10.0, -5.0, 0.5)]))
    assert states == {0.0: "none", 300.0: "none", 600.0: "none", 900.0: "none", 1200.0: "forming",
                      1500.0: "forming"}
    # a storm that has rained itself out later is not "not yet"
    assert 2400.0 not in states
    # the thresholds: 5 dBZ anywhere in the column, or 2 m/s of updraft at 5 km
    assert storm.states(_index([(0.0, 4.9, 4.9, 1.9)])) == {0.0: "none"}
    assert storm.states(_index([(0.0, 4.9, 5.0, 0.0)])) == {0.0: "forming"}
    assert storm.states(_index([(0.0, -35.0, -35.0, 2.0)])) == {0.0: "forming"}
    assert storm.states(_index([(0.0, 5.0, 30.0, 9.0)])) == {}
    # no facts (an older index, a look that failed): no words
    assert storm.states({"products": [{"id": "reflectivity", "frames": [{"time_s": 0.0, "file": "x.png"}]}]}) == {}
    assert storm.states(None) == {}


def test_a_preset_with_a_measured_start_says_when_and_any_other_run_says_it_plainly(tmp_path):
    index = _index(SUPERCELL_1KM)
    usual = storm.notes({"preset": "supercell-1km"}, index, G.presets_dir(), STRINGS)
    assert usual[0.0] == ("No storm yet: in this setup the storm usually appears between T+0:20:00 and T+0:30:00 "
                          "of storm time.")
    assert usual[900.0] == usual[0.0] and set(usual) == {0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0}
    assert usual[1200.0].startswith("The storm is starting to form higher up.")
    assert storm.notes({"preset": "supercell-250m"}, index, G.presets_dir(), STRINGS)[0.0] == usual[0.0]
    for run in ({"preset": "oun-2013-05-20"}, {"preset": "no-such-preset"}, {"preset": "../x"}, {}):
        assert storm.notes(run, index, G.presets_dir(), STRINGS)[0.0] == "No storm yet: it has not formed at this time."
    # a run that ran to its end with no echo at 1 km at any saved time: no storm formed, not "not yet"
    never = storm.notes({"preset": "supercell-1km"}, _index(SUPERCELL_1KM[:4], ended=True, reason="completed"),
                        G.presets_dir(), STRINGS)
    assert set(never.values()) == {"No storm formed in this run: this air did not make one in the storm time it ran."}
    # ... as does one that ran past the time this storm usually shows
    late = [(t, -35.0, -35.0, 0.2) for t in range(0, 2101, 300)]
    assert set(storm.notes({"preset": "supercell-1km"}, _index(late, ended=True, reason="stopped"), G.presets_dir(),
                           STRINGS).values()) == set(never.values())
    # ... or reached the run's own end time (run.json), whatever its last record says
    assert set(storm.notes({"duration_s": 900.0}, _index(SUPERCELL_1KM[:4], ended=True), G.presets_dir(),
                           STRINGS).values()) == set(never.values())
    # ... while one still running says "not yet"
    assert storm.notes({}, _index(SUPERCELL_1KM[:4]), G.presets_dir(), STRINGS)[900.0].startswith("No storm yet")
    # the run folder's own run.json names the preset
    (tmp_path / "run.json").write_text(json.dumps({"preset": "supercell-1km"}))
    assert storm.run_notes(tmp_path, index) == usual
    assert storm.note_for(usual, 300) == usual[300.0] and storm.note_for(usual, None) is None


def test_a_run_stopped_before_its_storm_usually_shows_is_not_said_to_have_no_storm():
    """The 250 m supercell stopped at T+0:10:00: its storm usually shows at T+0:20:00 to T+0:30:00."""

    quiet = [(t, -35.0, -35.0, 0.2) for t in (0.0, 300.0, 600.0)]
    for reason in ("stopped", "error", None):
        notes = storm.notes({"preset": "supercell-250m"}, _index(quiet, ended=True, reason=reason), G.presets_dir(),
                            STRINGS)
        assert set(notes) == {0.0, 300.0, 600.0}
        assert set(notes.values()) == {"No storm yet: the run ended at T+0:10:00, before this storm usually appears "
                                       "(T+0:20:00 to T+0:30:00)."}, reason
    # A run with no known storm time that ended early says only "not yet".
    early = storm.notes({}, _index(quiet, ended=True, reason="stopped"), G.presets_dir(), STRINGS)
    assert set(early.values()) == {"No storm yet: it has not formed at this time."}


def test_a_look_drawn_once_per_run_carries_no_storm_line(tmp_path):
    """The environment (time 0) is the air the run starts from, not a moment of the storm: no "no storm yet" on it,
    in Results or in the live view, even when the run's first saved time had no storm."""

    from cm1core.gui.api import Api
    from cm1core.gui.files import Root
    from cm1core.gui.jobs import Jobs, fake_runner
    from cm1core.gui.prefs import Prefs

    root = tmp_path / "runs"
    rundir = root / "storm"
    (rundir / "quicklook").mkdir(parents=True)
    (rundir / "run.json").write_text(json.dumps({"preset": "supercell-1km"}), encoding="utf-8")
    index = _index(SUPERCELL_1KM, ended=True, reason="completed")
    index["products"].append({"id": "environment", "title": "The storm's environment", "once": True,
                              "frames": [{"index": 0, "time_s": 0.0, "file": "environment/t000000.png",
                                          "facts": {}}]})
    (rundir / "quicklook" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    runs = Root(root)
    api = Api(runs, Jobs(fake_runner(), runs.server_dir), Prefs(tmp_path / "config"), token="t", port=1,
              version="x", bind="127.0.0.1")
    try:
        body = api.results("storm", rundir, "guided")
        looks = {look["id"]: look for look in body["looks"]}
        assert looks["reflectivity"]["frames"][0]["storm"].startswith("No storm yet")
        assert "storm" not in looks["environment"]["frames"][0]
        live = {row["id"]: row for row in api.views.looks("storm", rundir)["looks"]}
        assert "storm" not in live["environment"]
    finally:
        api.close()


def test_pictures_behind_the_gpu_are_counted_and_timed():
    from cm1core.gui.runview import DRAW_S_PER_FRAME, pictures_behind

    saved = [{"model_time_s": float(t), "time_utc": "2026-10-10T12:00:00Z"} for t in range(0, 7201, 300)]
    assert pictures_behind(saved, 7200.0, 25) is None
    assert pictures_behind([], None, 0) is None
    # 7 of 25 drawn: 18 to go, at the rate this run has drawn them (70 s for 7, so 10 s each).
    behind = pictures_behind(saved, 1800.0, 7, now=1_791_000_000.0)
    assert behind["behind"] == 18 and behind["drawn"] == 7 and behind["saved"] == 25
    assert behind["lag_s"] == 5400.0
    from cm1core.gui.runstate import _utc_seconds

    start = _utc_seconds("2026-10-10T12:00:00Z")
    behind = pictures_behind(saved, 1800.0, 7, now=start + 70.0)
    assert behind["about_s"] == 18 * 10
    # Before any is drawn, the desktop figure.
    behind = pictures_behind(saved[:3], None, 0, now=start + 5.0)
    assert behind["behind"] == 3 and behind["about_s"] == int(max(5, round(3 * DRAW_S_PER_FRAME)))


def test_watching_and_results_say_when_pictures_are_still_being_drawn():
    run, results, parts = _js("js/run/run.js"), _js("js/run/results.js"), _js("js/run/parts.js")
    assert "live.pictures_behind" in run and 't(over ? "watch.drawing" : "watch.behind"' in run
    assert '"watch.finished_drawing"' in run
    assert "data.pictures_behind" in results and 't("watch.drawing"' in results
    for key in ("watch.drawing", "watch.behind", "watch.finished_drawing", "watch.first_picture_drawing",
                "watch.stop_confirm", "storm.none_stopped"):
        assert key in STRINGS, key
    # "Waiting for the GPU: another run is using it" only when one is.
    assert 'Number(queue.ahead || 0) > 0 || Boolean(queue.holder)' in run
    # The first picture's words follow the model once the GPU has started.
    assert "if (next <= 0 && Number(model) > next) return (words.firstDrawing || words.drawing)" in parts
    # Stop on a run with no saved point asks first.
    assert 'window.confirm(t("watch.stop_confirm"))' in run


def test_the_supercell_presets_carry_their_measured_start_inside_their_storm_time():
    for preset_id in ("supercell-1km", "supercell-250m"):
        record = P.load(preset_id).record
        window = record["storm_appears"]
        assert storm.appears(preset_id, G.presets_dir()) == (window["from_s"], window["to_s"]) == (1200.0, 1800.0)
        assert window["basis"].startswith("measured:")
        assert window["to_s"] <= record["min_duration_s"]
    assert storm.appears("oun-2013-05-20", G.presets_dir()) is None


def test_the_fake_runner_records_a_quiet_start_then_a_storm():
    states = storm.states(_index([(t, _fake_facts("reflectivity", t)["dbz_max_1km"],
                                   _fake_facts("reflectivity", t)["dbz_max_column"],
                                   _fake_facts("updraft-5km", t)["w_max"]) for t in range(0, 3601, 300)]))
    assert states[0.0] == states[900.0] == "none" and states[1200.0] == "forming" and 1800.0 not in states


def test_a_quiet_reflectivity_is_said_as_it_is():
    """-35 dBZ is no echo at all; the strongest value is never said as +35 dBZ (spin keeps either sign)."""

    assert G.strongest_words("reflectivity", {"dbz_max_1km": -35.0}) == "-35 dBZ"
    assert G.strongest_words("reflectivity", {"dbz_max_1km": 52.4}) == "52 dBZ"
    assert G.strongest_words("spin-near-ground", {"zeta_max": 0.01, "zeta_min": -0.02}) == "0.020 per second"


# ------------------------------------------------------------------ when the next picture comes


def test_the_picture_schedule_of_a_first_segment_and_of_a_continued_one():
    admitted = {"history_interval_s": 300.0, "duration_s": 7200.0, "start_step": 0, "dt_s": 6.0}
    first = storm.schedule(admitted, {"duration_s": 7200.0}, None)
    assert first == {"interval_s": 300.0, "first_s": 0.0, "shown_s": None, "next_s": 0.0, "duration_s": 7200.0}
    assert storm.schedule(admitted, {}, 900.0)["next_s"] == 1200.0
    assert storm.schedule(admitted, {}, 7200.0)["next_s"] is None          # the last picture is drawn
    # a segment continued from the restart at T+1:00:00: its first picture is the next saved time after it
    resumed = {"history_interval_s": 900.0, "duration_s": 10800.0, "start_step": 2400, "dt_s": 1.5}
    assert storm.schedule(resumed, {}, None)["first_s"] == 4500.0
    assert storm.schedule({}, {}, None) is None and storm.schedule(None, None, None) is None
    assert storm.clock(4500) == "T+1:15:00"


# ------------------------------------------------------------------ through the server, on the fake runner


@pytest.fixture()
def live(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(FIXTURES / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(FIXTURES / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setenv("GPUWM_CM1_HOST_FIXTURE", str(FIXTURES / "host-64gb-1tb.json"))
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.002")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.1")
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        yield server, client
    finally:
        stop_server(server)


def test_watching_and_results_say_no_storm_yet_and_when_pictures_come(live):
    server, client = live
    started = client.post("/api/guided/start", {"preset": "supercell-1km", "budget": "10m", "name": "w"})
    assert started.status == 200, started.raw
    wait_for(lambda: client.get("/api/runs/w/status").json["state"] == "finished", timeout=120)
    wait_for(lambda: client.get("/api/runs/w/live").json["frames_drawn"] == 25, timeout=60)
    view = client.get("/api/runs/w/live").json
    assert view["pictures"]["interval_s"] == 300.0 and view["pictures"]["first_s"] == 0.0
    assert view["pictures"]["shown_s"] == 7200.0 and view["pictures"]["next_s"] is None
    assert not any(look.get("storm") for look in view["looks"])        # the storm is there at T+2:00:00
    usual = ("No storm yet: in this setup the storm usually appears between T+0:20:00 and T+0:30:00 of storm "
             "time.")
    for mode in ("guided", "advanced"):
        body = client.get(f"/api/runs/w/results?mode={mode}").json
        looks = {look["id"]: look for look in body["looks"]}
        lines = {frame["time_s"]: frame.get("storm") for frame in looks["reflectivity"]["frames"]}
        assert lines[0] == lines[900] == usual
        assert lines[1200].startswith("The storm is starting to form higher up.")
        assert lines[1800] is None and lines[7200] is None
        # every look of a saved time says the same
        assert {frame["time_s"]: frame.get("storm") for frame in looks["swath"]["frames"]} == lines
    guided = client.get("/api/runs/w/results?mode=guided").json
    first = next(look for look in guided["looks"] if look["id"] == "reflectivity")["frames"][0]
    assert first["strongest"] == "-35 dBZ"


# ------------------------------------------------------------------ the pages


def _js(path: str) -> str:
    return (STATIC / path).read_text(encoding="utf-8")


def test_both_run_screens_open_with_the_progress_block_and_say_when_the_next_picture_comes():
    run = _js("js/run/run.js")
    parts = _js("js/run/parts.js")
    assert run.count("progressBlock(") == 2 and run.count("pictureWords(") == 2
    assert "progressBar()" not in run
    # before the first picture its time is said where the picture will be
    assert run.count("if (first && next) view.setEmpty(next);") == 2
    for name in ("progressBlock", "pictureWords", "stormLine", "spanWords"):
        assert f"export function {name}(" in parts
    css = _js("css/gm1.css")
    big = re.search(r"\.meter\.progress\.big \{ height: (\d+)px;", css)
    assert big and int(big.group(1)) >= 12
    assert ".nostorm {" in css


def test_a_picture_with_no_storm_yet_says_so_above_it_on_every_run_and_results_view():
    run, results, parts = _js("js/run/run.js"), _js("js/run/results.js"), _js("js/run/parts.js")
    assert run.count("note: (look, f) => stormLine(f)") == 2
    assert results.count("note: (look, f) => stormLine(f)") == 2
    # the live view's latest picture keeps its line; the viewer draws the note above the picture
    assert "storm: look.storm" in parts and "put(noteBox, opts.note ? opts.note(look, f) : null);" in parts
    assert parts.index("tabs, noteBox, empty, figure") > 0
    # "What am I looking at" says it too, in place of a strongest value from quiet air
    # (a hurricane's early pictures have a storm in them and a line beside it: those keep their value)
    assert "more ? null : noStorm," in results
    assert "frame.strongest && (!quiet || frame.storm_showing)" in results


def test_every_watching_string_the_pages_ask_for_is_in_the_copy():
    for path in ("js/run/run.js", "js/run/results.js"):
        for key in re.findall(r"""\bt\("([a-z_]+\.[a-z_.]+)\"""", _js(path)):
            assert key in STRINGS, (path, key)
    for key in ("storm.none_usual", "storm.none", "storm.forming", "storm.none_ended"):
        assert key in STRINGS
    assert "{from}" in STRINGS["storm.none_usual"]["text"] and "{to}" in STRINGS["storm.none_usual"]["text"]


# ------------------------------------------------------------------ pictures every 5 minutes


def _tapfrq(text: str) -> float:
    found = re.search(r"^\s*tapfrq\s*=\s*([0-9.]+)", text, re.M)
    assert found
    return float(found.group(1))


def test_every_offered_preset_saves_a_picture_at_least_every_5_minutes_and_its_card_picture_is_a_saved_time():
    namelist_tool()
    offered = 0
    for preset in P.load_all():
        if not P.examine(preset)["offered"]:
            continue
        offered += 1
        interval = _tapfrq(preset.namelist_text)
        if preset.record.get("storm_kind") == "hurricane":
            # A storm of days, not hours: a picture every 3 hours at most (33 pictures in 4 days).
            assert interval <= 10800.0, preset.id
            continue
        assert interval <= 300.0, preset.id
        picture = preset.folder / "picture.json"
        if picture.is_file():
            shown = json.loads(picture.read_text(encoding="utf-8"))["run"]["model_time_s"]
            assert shown % interval == 0, (preset.id, shown, interval)
    assert offered >= 15
    record = P.load("supercell-250m").record
    assert {"key": "tapfrq", "from": 900.0, "to": 300.0,
            "why": "a picture every 5 minutes, so the live view moves"} in record["source"]["changes"]
