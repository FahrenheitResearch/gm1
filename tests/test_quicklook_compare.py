"""``gpuwm-cm1 quicklook RUNDIR --compare OTHER ...``: runs side by side, runs over time, and the soundings card.

CPU only.  The frames carry CM1 r21.1's own 24 x 24 x 40 storm (``quicklook_support``), as the other quick-look
tests do.  What is checked is what the comparison must carry:

1. every picture is drawn, passes the honesty lint and keeps its footer whole; the record names each run, its time
   and its centre;
2. the storm time: the latest time every run reached, each run drawn at its save nearest it, and a picture whose
   runs were saved at different times says each run's own time;
3. each box is centred on its own storm: the rotating updraft, else the strongest updraft at 5 km, else the strongest
   reflectivity at 1 km, else the middle;
4. a run whose saved frames were deleted is cut from its own pictures (its series from its series.json), and says so;
5. the settings that differ between the runs are named;
6. what cannot be compared is refused in plain words (one run, seven runs, a run twice, a hurricane with a
   thunderstorm, a storm time after every run ended), and the command line draws and refuses the same;
7. the soundings card draws every bar for every run, however full its left column, and says that its big numbers
   are the whole area's.

The quick-looks themselves are not touched: this theme adds modules and draws no existing look differently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest

import importlib.util

NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")

from cm1import_support import namelist_tool  # noqa: E402
from cm1out_support import end_event  # noqa: E402
from quicklook_support import NAMELIST_24, publish_storm, run_directory  # noqa: E402

from cm1core.quicklook import compare as CO  # noqa: E402
from cm1core.quicklook import compare_draw as CD  # noqa: E402
from cm1core.quicklook import products as P  # noqa: E402
from cm1core.quicklook import render as RD  # noqa: E402
from cm1core.quicklook import style as S  # noqa: E402
from cm1core.quicklook.frame import QuickFrame, frames_of  # noqa: E402
from cm1core.cm1out import frames as _frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


@pytest.fixture()
def overlays(monkeypatch):
    """Every Overlay a compare picture saves through products.save, with its file name."""

    seen: list[tuple[str, S.Overlay]] = []
    original = P.save

    def spy(ov, base, path):
        seen.append((Path(path).name, ov))
        return original(ov, base, path)

    monkeypatch.setattr(P, "save", spy)
    return seen


def _make(targets, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)      # a cut footer or a long title is a warning: fail on it
        return CO.make(targets, **kwargs)


def _size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _changed_namelist(tmp_path: Path, name: str, key: str, value: str) -> Path:
    text = NAMELIST_24.read_text(encoding="utf-8")
    changed, count = re.subn(rf"(?im)^(\s*{key}\s*=\s*)[^,\n]+", rf"\g<1>{value}", text)
    assert count == 1, key
    path = tmp_path / f"{name}.namelist.input"
    path.write_text(changed, encoding="utf-8")
    return path


def _quiet_run(tmp_path: Path) -> Path:
    """A run whose air does not move: no updraft anywhere."""

    run = tmp_path / "quiet"
    (run / "frames").mkdir(parents=True)
    shutil.copy(NAMELIST_24, run / "namelist.input")
    publish_storm(run / "frames", 0, 1500.0, scale=0.0)
    end_event(run / "events.jsonl", frames=1)
    return run


# ---------------------------------------------------------------- every picture

@NEEDS_MATPLOTLIB
def test_two_runs_are_drawn_side_by_side_over_time_and_with_their_soundings(tmp_path, overlays):
    a = run_directory(tmp_path, name="a")
    b = run_directory(tmp_path, name="b", times=(900.0, 1500.0))
    out = tmp_path / "cmp"
    record = _make([a, b], out=out)
    assert record["schema"] == CO.SCHEMA and record["time_s"] == 1500.0 and record["kind"] == "thunderstorm"
    assert [r["letter"] for r in record["runs"]] == ["A", "B"]
    assert [r["colour"] for r in record["runs"]] == list(CO.COLOURS[:2])
    assert set(record["pictures"]) == {"storm-reflectivity", "storm-surface", "series", "soundings"}
    for key, value in record["pictures"].items():
        for row in value if isinstance(value, list) else [value]:
            assert _size(out / row["file"]) == (1920, 1080), key
            assert (out / "thumb" / row["file"]).is_file()
    assert json.loads((out / "compare.json").read_text(encoding="utf-8"))["pictures"] == record["pictures"]
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    assert progress["state"] == "done" and progress["done"] == progress["total"]
    # the honesty rules hold on every picture saved through the quick-looks' own save
    assert len(overlays) == 3
    for name, ov in overlays:
        assert ov.check() == [] and ov.footer_cut == [], name
        words = "\n".join(ov.texts).lower()
        assert "simulated" in words and "tornad" not in words
    storm = record["pictures"]["storm-reflectivity"]["facts"]
    assert [row["letter"] for row in storm["runs"]] == ["A", "B"] and storm["box_m"] == 24000.0
    assert all(row["source"] == "frames" and row["centre"] == "rotation" for row in storm["runs"])


@NEEDS_MATPLOTLIB
def test_the_storm_time_is_the_latest_every_run_reached_and_each_runs_own_save_is_said(tmp_path, overlays):
    a = run_directory(tmp_path, name="a", times=(900.0, 1200.0, 1500.0))
    b = run_directory(tmp_path, name="b", times=(600.0, 1000.0, 1400.0))
    record = _make([a, b], out=tmp_path / "cmp", only=("storm",))
    assert record["time_s"] == 1400.0
    assert {r["letter"]: r["time_s"] for r in record["runs"]} == {"A": 1500.0, "B": 1400.0}
    words = "\n".join(t for _n, ov in overlays for t in ov.texts)
    assert "each at its save nearest T+0:23:20" in words
    assert "saved times drawn: A T+0:25:00, B T+0:23:20" in words
    asked = _make([a, b], out=tmp_path / "cmp2", only=("storm",), time_s=1000.0)
    assert {r["letter"]: r["time_s"] for r in asked["runs"]} == {"A": 900.0, "B": 1000.0}


@NEEDS_MATPLOTLIB
def test_a_time_after_every_run_ended_is_refused_and_the_charts_run_to_the_last_save(tmp_path):
    a = run_directory(tmp_path, name="a", times=(900.0, 1200.0, 1500.0))
    b = run_directory(tmp_path, name="b", times=(900.0, 1500.0))
    # a typo such as T+20:00:00 for a run of 25 minutes: refused in words, and its progress.json says so (the
    # Results screen shows that line), rather than a clock of years and a squashed time axis
    late = re.escape("The storm time T+5:33:20 is after every run ended: the last save of any of them is A at "
                     "T+0:25:00.")
    with pytest.raises(CO.CompareError, match=late):
        CO.make([a, b], time_s=20000.0, out=tmp_path / "late")
    progress = json.loads((tmp_path / "late" / "progress.json").read_text(encoding="utf-8"))
    assert progress["state"] == "failed" and "after every run ended" in progress["message"]
    with pytest.raises(CO.CompareError, match="after every run ended"):
        CO.make([a, b], time_s=1.0e9, out=tmp_path / "later")
    # up to one save interval past the last save is still a storm time: each run is drawn at its last save
    record = _make([a, b], time_s=2000.0, out=tmp_path / "cmp", only=("storm", "series"))
    assert record["time_s"] == 2000.0
    assert {r["letter"]: r["time_s"] for r in record["runs"]} == {"A": 1500.0, "B": 1500.0}
    facts = record["pictures"]["series"]["facts"]
    assert facts["time_unit"] == "min" and facts["time_s"] == 2000.0 and facts["charts_to_s"] == 1500.0


# ---------------------------------------------------------------- the centres

def test_each_box_is_centred_on_its_own_storm(tmp_path):
    rotating = run_directory(tmp_path, name="rot", times=(1500.0,))
    plain = run_directory(tmp_path, name="plain", times=(1500.0,), uh=False)
    quiet = _quiet_run(tmp_path)
    for run, kind in ((rotating, "rotation"), (quiet, "middle")):
        _t, key, path, segment = frames_of(_frames.discover(run))[-1]
        qf = QuickFrame.open(path, key=key, segment=segment)
        try:
            centre = CO.storm_centre(qf, hurricane=False)
            if kind == "rotation":
                uh = qf.get("uh")
                j, i = np.unravel_index(int(np.argmax(uh)), uh.shape)
                # the helicity-weighted centre, within 6 km of the peak
                assert abs(centre.x - (i + 0.5) * qf.dx) <= CO.CENTRE_RADIUS
                assert abs(centre.y - (j + 0.5) * qf.dy) <= CO.CENTRE_RADIUS
            else:
                assert (centre.x, centre.y) == (0.5 * qf.extent[0], 0.5 * qf.extent[1])
                assert "no updraft of 5 m/s at 5 km" in centre.words
                # a storm with no updraft left but rain still falling is centred on its strongest reflectivity
                dbz = np.full((qf.ny, qf.nx), -35.0, np.float32)
                dbz[5:8, 15:18] = 45.0
                rain = CO.storm_centre(qf, hurricane=False, dbz_1km=dbz)
                assert rain.kind == "rain" and "strongest reflectivity at 1 km" in rain.words
                assert (rain.x, rain.y) == pytest.approx((16500.0, 6500.0))
                weak = CO.storm_centre(qf, hurricane=False, dbz_1km=dbz - 30.0)
                assert weak.kind == "middle"
            assert centre.kind == kind
        finally:
            qf.close()
    _t, key, path, segment = frames_of(_frames.discover(plain))[-1]
    qf = QuickFrame.open(path, key=key, segment=segment)
    try:
        centre = CO.storm_centre(qf, hurricane=False)
        uh, source = qf.uh()
        # no CM1 uh in the frame: CM1's calcuh sum on the frame's winds decides, as the updraft look does
        assert source.startswith("CM1's calcuh") and centre.kind == ("rotation" if uh.max() >= CO.UH_ROTATING
                                                                      else "updraft")
    finally:
        qf.close()


def test_a_box_centred_near_the_edge_hatches_what_lies_outside_the_run(tmp_path):
    box = CD.panel_box(0, 0, 100, 0.0, 12000.0, 24000.0)
    mask = CD.outside(box, (24000.0, 24000.0))
    assert mask[:, :45].all() and not mask[:, 55:].any()
    img = np.zeros((100, 100, 3), np.uint8) + 200
    hatched = CD.hatch(img, mask)
    assert (hatched[:, 60:] == 200).all()
    assert set(map(tuple, hatched[:, :45].reshape(-1, 3))) == {S.rgb8("void"), CD.HATCH}


# ---------------------------------------------------------------- a run whose frames were deleted

@NEEDS_MATPLOTLIB
def test_a_run_whose_frames_were_deleted_is_cut_from_its_own_pictures(tmp_path):
    a = run_directory(tmp_path, name="a")
    b = run_directory(tmp_path, name="b")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        RD.render(b)
    shutil.rmtree(b / "frames")
    record = _make([a, b], out=tmp_path / "cmp")
    storm = {row["letter"]: row for row in record["pictures"]["storm-reflectivity"]["facts"]["runs"]}
    assert storm["A"]["source"] == "frames" and storm["B"]["source"] == "pictures"
    assert storm["B"]["centre"] == "records" and "in its area" in storm["B"]["value"]
    assert record["runs"][1]["saved_frames"] is False
    assert record["pictures"]["series"]["facts"]["sources"] == {"A": "its saved frames", "B": "its series.json"}
    # the pressure drop needs the saved frames, and the chart says whose were deleted
    assert record["pictures"]["series"]["facts"]["runs_per_chart"]["dp_sfc_max"] == ["A"]


def test_cm1_output_opened_for_pictures_is_read_from_its_files(tmp_path):
    """CM1 netCDF output opened for pictures (its folder holds only cm1-output.json and the pictures) is compared
    from the files where they are, by its own name, and nothing is written beside the files."""

    import subprocess
    import sys

    a = run_directory(tmp_path, name="a")
    b = run_directory(tmp_path, name="b")
    done = subprocess.run([sys.executable, "-m", "cm1core", "cm1out", str(b), "--format", "nc", "--no-stats",
                           "--filetype", "2"], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(Path(CO.__file__).parents[2])})
    assert done.returncode == 0, done.stderr
    files = b / "cm1out"
    shutil.rmtree(b / "frames")
    before = sorted(p.name for p in files.iterdir())
    opened = tmp_path / "runs" / "cm1-files"
    opened.mkdir(parents=True)
    (opened / "cm1-output.json").write_text(json.dumps({"schema": "gm1.cm1-output/v1", "name": "cm1-files",
                                                        "source": str(files)}), encoding="utf-8")
    record = _make([a, opened], out=tmp_path / "cmp")
    assert record["runs"][1]["name"] == "cm1-files" and record["runs"][1]["saved_frames"] is True
    assert record["runs"][1]["folder"] == str(opened)
    storm = {row["letter"]: row for row in record["pictures"]["storm-reflectivity"]["facts"]["runs"]}
    assert storm["B"]["source"] == "frames" and storm["B"]["centre"] == storm["A"]["centre"]
    # its sounding is its first saved time's, as for a run with its frames
    assert CO.profile(CO.load_run(opened, "B", CO.COLOURS[1]))[1].startswith("first frame")
    assert sorted(p.name for p in files.iterdir()) == before
    # the Results screen offers it among the runs to compare with
    from cm1core.gui import compare as GC

    assert GC.has_storm(opened) and GC.cannot(a, [("cm1-files", opened)]) == {}
    # files no longer where the folder says: compared from its pictures, as a run whose frames were deleted
    (opened / "cm1-output.json").write_text(json.dumps({"schema": "gm1.cm1-output/v1", "name": "cm1-files",
                                                        "source": str(tmp_path / "gone")}), encoding="utf-8")
    assert CO.opened_cm1_output(opened) is None and not GC.has_storm(opened)


def test_a_picture_from_another_layout_is_not_cut(tmp_path):
    box = CO.picture_box("reflectivity", (24000.0, 24000.0), False)
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    assert not CO.outline_holds(rgb, box)
    from PIL import Image, ImageDraw

    im = Image.fromarray(rgb)
    ImageDraw.Draw(im).rectangle((box.x - 1, box.y - 1, box.x + box.w, box.y + box.h), outline=CO.FAINT, width=1)
    assert CO.outline_holds(np.asarray(im), box)


# ---------------------------------------------------------------- what differs, and refusals

def test_the_settings_that_differ_are_named(tmp_path):
    a = run_directory(tmp_path, name="a", times=(1500.0,))
    b = run_directory(tmp_path, name="b", times=(1500.0,), namelist=_changed_namelist(tmp_path, "b", "tsk0", "301.0"))
    runs = CO.load_runs([a, b])
    words, unread = CO.differences(runs)
    assert unread == [] and words == ["settings tsk0"]
    same, _ = CO.differences(CO.load_runs([a, run_directory(tmp_path, name="c", times=(1500.0,))]))
    assert same == []


def test_parcels_are_not_a_difference_and_the_time_step_is_said_in_words(tmp_path):
    text = NAMELIST_24.read_text(encoding="utf-8")
    for key, value in (("iprcl", "1"), ("nparcels", "100"), ("prcl_th", "0"), ("dtl", "3.0")):
        text, count = re.subn(rf"(?im)^(\s*{key}\s*=\s*)[^,\n]+", rf"\g<1>{value}", text)
        assert count == 1, key
    changed = tmp_path / "parcels.namelist.input"
    changed.write_text(text, encoding="utf-8")
    a = run_directory(tmp_path, name="a", times=(1500.0,))
    b = run_directory(tmp_path, name="b", times=(1500.0,), namelist=changed)
    words, unread = CO.differences(CO.load_runs([a, b]))
    # CM1's parcels follow air through the run and do not change the storm; dtl is named as what it is
    assert unread == [] and words == ["time step (dtl)"]


def test_lengths_are_said_in_three_figures_and_whole_km_as_they_are():
    assert [CO.km_words(m) for m in (49312.0, 20559.2, 120000.0, 1536000.0, 100500.0, 750.0, 48000.0)] == \
        ["49.3", "20.6", "120", "1536", "100.5", "0.75", "48"]


def test_what_cannot_be_compared_is_refused_in_plain_words(tmp_path, monkeypatch):
    a = run_directory(tmp_path, name="a", times=(1500.0,))
    with pytest.raises(CO.CompareError, match="at least one other run"):
        CO.load_runs([a])
    with pytest.raises(CO.CompareError, match="named twice"):
        CO.load_runs([a, Path(os.fspath(a))])
    with pytest.raises(CO.CompareError, match="At most 6 runs"):
        CO.load_runs([a] * 7)
    with pytest.raises(CO.CompareError, match="does not exist"):
        CO.load_runs([a, tmp_path / "nosuch"])
    b = run_directory(tmp_path, name="b", times=(1500.0,))
    original = CO.load_run

    def as_hurricane(target, letter, colour):
        run = original(target, letter, colour)
        if letter == "B":
            run.facts.storm_kind = "hurricane"
        return run

    monkeypatch.setattr(CO, "load_run", as_hurricane)
    with pytest.raises(CO.CompareError, match=r"hurricane run \(B\) is compared only with other hurricane runs"):
        CO.load_runs([a, b])
    monkeypatch.setattr(CO, "load_run", original)
    # a run with nothing to draw: refused, and its progress.json says why (the Results screen shows that line)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CO.CompareError, match="nothing of its storm to compare"):
        CO.make([a, empty], out=tmp_path / "failed")
    progress = json.loads((tmp_path / "failed" / "progress.json").read_text(encoding="utf-8"))
    assert progress["state"] == "failed" and "empty: no saved frames" in progress["message"]
    with pytest.raises(CO.CompareError, match="no compare look"):
        CO.make([a, b], only=("radar",), out=tmp_path / "x")
    with pytest.raises(CO.CompareError, match="--box"):
        CO.make([a, b], box_km=0.5, out=tmp_path / "x")


#: A run's own numbers before it started, as the importer's report records them (environment.parameters).
ENVIRONMENT = {"mixed_layer": {"cape_j_per_kg": 2100.0, "lcl_m_agl": 900.0}, "srh_0_1km_m2_per_s2": 150.0,
               "srh_0_3km_m2_per_s2": 320.0, "shear_0_6km_m_per_s": 31.0, "bunkers_right_u_m_per_s": 9.0,
               "bunkers_right_v_m_per_s": 3.0}


@NEEDS_MATPLOTLIB
def test_a_three_run_card_draws_every_bar_for_every_run_and_says_whose_numbers_are_whose(tmp_path, overlays,
                                                                                        monkeypatch):
    runs = [run_directory(tmp_path, name=n, times=(900.0, 1500.0)) for n in ("a", "b", "c")]
    original = CO.load_run
    # the runs' CM1 checks differ (each says its own, one wraps) and the cases are long: the left column is
    # at its fullest, as in a card of a 1 km run, an any-grid run and a stretched-level run
    checks = {"A": "within 5 % of CM1 on CM1's stock supercell (measured)",
              "B": "same as CM1 in the importer's check", "C": "not yet checked against CM1"}

    def described(target, letter, colour):
        run = original(target, letter, colour)
        run.facts.checked = checks[letter]
        run.record = {"case": f"Supercell, 67 m grid (customized from Supercell, quick look, 1 km) {letter}"}
        return run

    monkeypatch.setattr(CO, "load_run", described)
    monkeypatch.setattr(CO, "environment", lambda run: (dict(ENVIRONMENT), "the importer's report"))
    record = _make(runs, out=tmp_path / "cmp", only=("soundings",))
    assert [row["runs"] for row in record["pictures"]["soundings"]] == [["A", "B", "C"]]
    cards = [ov for name, ov in overlays if name.startswith("soundings")]
    assert len(cards) == 1
    ov = cards[0]
    assert ov.check() == [] and ov.footer_cut == []
    for label, _path, _unit, _fmt in CD.BARS:
        assert ov.texts.count(label) == 3, label            # never dropped for want of room
    assert sum(t.startswith("CM1 check: ") for t in ov.texts) == 3
    # the big numbers are the whole area's (its time series), the words under the box are the box's: both said
    assert any(t.endswith("numbers: its whole area") for t in ov.texts)
    assert sum("numbers above: its whole area" in t for t in ov.texts) == 3


def test_the_cards_hold_up_to_three_runs():
    assert [CD.card_groups(n) for n in (2, 3, 4, 5, 6)] == [
        [[0, 1]], [[0, 1, 2]], [[0, 1], [2, 3]], [[0, 1, 2], [3, 4]], [[0, 1, 2], [3, 4, 5]]]


def test_the_run_colours_are_the_validated_palette_in_its_fixed_order():
    assert CO.COLOURS == ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300")
    assert len(CO.LETTERS) == len(CO.COLOURS) == CO.MAX_RUNS


def test_a_soundings_file_is_read_as_cm1_reads_it(tmp_path):
    path = tmp_path / "input_sounding"
    path.write_text("1000.0 300.0 14.0\n500.0 301.0 13.0 2.0 1.0\n1500.0 304.0 10.0 6.0 3.0\n"
                    "6000.0 320.0 1.0 20.0 5.0\n", encoding="utf-8")
    prof = CO._profile_from_file(path)
    assert prof["z"].tolist() == [0.0, 500.0, 1500.0, 6000.0]
    # base.F:617-618: the surface wind extrapolated from the first two levels
    assert prof["u"][0] == pytest.approx(0.0) and prof["v"][0] == pytest.approx(0.0)
    assert prof["p"][0] == pytest.approx(100000.0) and np.all(np.diff(prof["p"]) < 0)
    assert prof["t"][0] == pytest.approx(300.0) and np.all(prof["td"] <= prof["t"] + 1e-9)


# ---------------------------------------------------------------- the command line and the server's copies

@NEEDS_MATPLOTLIB
def test_the_command_line_compares_runs_and_refuses_in_words(tmp_path):
    a = run_directory(tmp_path, name="a", times=(900.0, 1500.0))
    b = run_directory(tmp_path, name="b", times=(900.0, 1500.0))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "0"}
    base = [sys.executable, "-m", "cm1core", "quicklook", str(a)]
    done = subprocess.run(base + ["--compare", str(b), "--only", "series", "--json"], capture_output=True,
                          text=True, env=env)
    assert done.returncode == 0, done.stderr
    record = json.loads(done.stdout)
    folder = Path(record["folder"])
    assert folder.parent == a / "quicklook" / "compare" and (folder / "series.png").is_file()
    assert record["worker"]["nice"] == 19 and record["worker"]["cuda_visible_devices"] == ""
    refused = subprocess.run(base + ["--compare", str(a)], capture_output=True, text=True, env=env)
    assert refused.returncode == 2 and "named twice" in refused.stderr
    # a time that is not a storm time is refused in the same words as a negative one, never with a traceback
    for odd in ("nan", "inf", "-inf", "-5"):
        refused = subprocess.run(base + ["--compare", str(b), f"--time={odd}"], capture_output=True, text=True, env=env)
        assert refused.returncode == 2 and "--time is a storm time in seconds, 0 or more." in refused.stderr, odd
        assert "Traceback" not in refused.stderr, odd
    for odd in (float("nan"), float("inf")):
        with pytest.raises(CO.CompareError, match="--time is a storm time in seconds, 0 or more."):
            CO.make([a, b], time_s=odd, only=("series",))


def test_the_servers_copies_match_the_drawing_code():
    from cm1core.gui import compare as GC

    assert GC.MAX_RUNS == CO.MAX_RUNS
    keys = {f"storm-{look}" for look in CD.TITLES} | {"series", "soundings"}
    assert keys == set(GC.TITLES) == set(GC.ORDER)
