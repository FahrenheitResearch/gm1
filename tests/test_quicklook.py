"""``gpuwm-cm1 quicklook``: the quick-look pictures, their words, swaths, slices, the column skew-T and --follow.

The frames carry CM1 r21.1's own storm state (``quicklook_support``), so every product draws real rain, hail, an
updraft and a cold pool.  What is checked is what a picture must carry: the footer's
facts, the honesty lint, reflectivity only from CM1's formula or the scheme's own, bilinear sampling only, fixed
and stated ranges; and of the worker: CPU only at nice 19, the same run-directory contract as cm1out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import warnings

import numpy as np
import pytest

from cm1core.quicklook import facts as FA
from cm1core.quicklook import products as P
from cm1core.quicklook import reflectivity as R
from cm1core.quicklook import render as RD
from cm1core.quicklook import series as SE
from cm1core.quicklook import skewt as SK
from cm1core.quicklook import slices as SL
from cm1core.quicklook import style as S
from cm1core.quicklook.frame import QuickFrame, lineage
from cm1core.quicklook.swath import SwathBuilder, plane_of

import importlib.util  # noqa: E402

#: Drawing needs matplotlib, which gpuwm (GM1's one dependency) brings; a test tree without gpuwm skips these.
NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")
from cm1import_support import namelist_tool  # noqa: E402
from cm1out_support import end_event, round_half_away  # noqa: E402
from quicklook_support import (  # noqa: E402
    NAMELIST_24, STOCK_NAMELIST, publish_storm, run_directory, storm,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "cm1core" / "quicklook"


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


@pytest.fixture()
def pictures(monkeypatch):
    """Every Overlay a product saves, with its words."""

    seen: list[tuple[str, S.Overlay]] = []
    original = P.save

    def spy(ov, base, path):
        seen.append((Path(path).parent.name if Path(path).parent.name != "slices" else Path(path).name, ov))
        return original(ov, base, path)

    monkeypatch.setattr(P, "save", spy)
    return seen


def _png_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _render(run: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return RD.render(run, **kwargs)


# ---------------------------------------------------------------- the products

@NEEDS_MATPLOTLIB
def test_every_quick_look_draws_from_a_cm1_storm(tmp_path, pictures):
    run = run_directory(tmp_path)
    summary = _render(run)
    assert summary.frames_drawn == 3 and summary.problems == []
    index = json.loads((run / "quicklook" / "index.json").read_text())
    assert index["schema"] == "gm1.quicklook-index/v1"
    # The per-frame looks, then the environment (drawn once per run; this storm has no parcel track)
    assert [p["id"] for p in index["products"]] == [pid for pid, _ in P.PRODUCTS] + ["environment"]
    assert [p["id"] for p in index["products"]][:4] == ["reflectivity", "spin-near-ground", "updraft-5km", "swath"]
    for product in [p for p in index["products"] if not p.get("once")]:
        assert [f["time_s"] for f in product["frames"]] == [900.0, 1200.0, 1500.0], product["id"]
        for frame in product["frames"]:
            assert _png_size(run / "quicklook" / frame["file"]) == (1920, 1080)
            assert _png_size(run / "quicklook" / frame["thumb"]) == (480, 270)
            assert frame["clock"] == S.fmt_clock(frame["time_s"])
    assert index["run"]["microphysics"] == "Morrison with hail"
    assert index["run"]["grid"] == "24 × 24 × 40, dx 1 km"
    assert index["run_ended"] is True
    assert not list((run / "quicklook").rglob("*.partial"))
    # the numbers each picture showed are the frame's own
    state = np.load(ROOT / "tests" / "data" / "cm1-ref-24x24" / "stats-run-t1500.npz")
    last = {p["id"]: p["frames"][-1]["facts"] for p in index["products"]}
    assert last["updraft-5km"]["w_max"] == pytest.approx(float(state["w"][10].max()))
    assert last["updraft-5km"]["level_m"] == 5000.0
    assert last["surface"]["theta_min"] == pytest.approx(float(state["thpert"][0].min()))
    assert last["spin-near-ground"]["height_m"] == 250.0
    assert last["spin-near-ground"]["resolved_m"] == 6000.0
    assert last["reflectivity"]["levels_m"] == [750.0, 1250.0]


@NEEDS_MATPLOTLIB
def test_footers_carry_the_plan_facts_and_pass_the_lint(tmp_path, pictures):
    run = run_directory(tmp_path)
    _render(run, only=None)
    assert len(pictures) == 3 * len(P.PRODUCTS) + 1        # every per-frame look at 3 frames, the environment once
    for name, ov in pictures:
        assert ov.check() == [], name
        assert ov.footer_cut == [], (name, ov.footer_cut)
        text = "\n".join(ov.texts)
        first = next(t for t in ov.texts if t.startswith(S.SIMULATED))
        assert first.startswith(f"simulated{S.SEP}GM1 ")
        assert "Morrison with hail" in first
        assert "dx 1 km" in first
        assert re.search(r"T\+\d+:\d\d:\d\d$", first), first
        assert "tornad" not in text.lower()
    by = {name: ov for name, ov in pictures}
    assert "lowest level 250 m" in "\n".join(by["spin-near-ground"].texts)
    assert "wind ground-relative" in "\n".join(by["spin-near-ground"].texts)
    assert "w level 5 km" in "\n".join(by["updraft-5km"].texts)
    assert "1 km and column maximum" in "\n".join(by["reflectivity"].texts)
    assert "wind ground-relative" in "\n".join(by["swath"].texts)


@NEEDS_MATPLOTLIB
def test_a_domain_fixed_to_the_ground_draws_every_look(tmp_path, pictures):
    """imove = 0: no frame motion to name in the notes; the updraft look's speeds are vertical velocities."""

    run = run_directory(tmp_path, move=(0.0, 0.0), swath="domain")
    summary = _render(run)
    assert summary.problems == [], summary.problems
    index = json.loads((run / "quicklook" / "index.json").read_text())
    assert {p["id"]: len(p["frames"]) for p in index["products"]} == {**{pid: 3 for pid, _ in P.PRODUCTS},
                                                                       "environment": 1}
    assert index["products"][3]["frames"][-1]["facts"]["source"] == "model"          # CM1's sws, ground-fixed
    for name, ov in pictures:
        assert ov.check() == [] and ov.footer_cut == [], name
        # the ground tracks are on the ground and the environment is a column: they name no domain motion
        assert "domain fixed to the ground" in "\n".join(ov.texts) or name in (
            "swath", "timeseries.png", "hail-track", "spin-track", "environment"), name


def test_the_checked_state_comes_from_the_import_report_or_the_receipt(tmp_path):
    run = run_directory(tmp_path, times=(900.0,))
    layout, rows = lineage(run)
    qf = QuickFrame.open(rows[0][2])
    try:
        facts = FA.run_facts(layout, qf.config, qf.dx, qf.dy)
        assert facts.checked == "no CM1 comparison recorded for this run"
        (run / "receipt.json").write_text(json.dumps({"version": "9.9.9", "unvalidated": {"capabilities": [
            {"option": "physics.component", "values": ["morrison-mp10"]}]}}))
        facts = FA.run_facts(layout, qf.config, qf.dx, qf.dy)
        assert (facts.version, facts.checked) == ("9.9.9", "not checked against CM1 on this setup")
        report = {"sections": [{"id": "same", "rows": [{"key": "ptype", "keys": ["ptype"]}]}],
                  "comparison": {"scheme": "morrison-mp10", "rows": [{"inside_target": True}]}}
        (run / "import-report.json").write_text(json.dumps(report))
        assert FA.run_facts(layout, qf.config, qf.dx, qf.dy).checked == \
            "within 5 % of CM1 on CM1's stock supercell (measured)"
        report["sections"][0]["id"] = "differs"
        (run / "import-report.json").write_text(json.dumps(report))
        assert FA.run_facts(layout, qf.config, qf.dx, qf.dy).checked == "runs differently from CM1"
    finally:
        qf.close()


def test_the_lint_refuses_tornado_words_marketing_and_unframed_winds():
    assert S.lint(["simulated", "fastest wind 32 m/s ground-relative"]) == []
    assert S.lint(["fastest wind 32 m/s ground-relative"]) == ["'simulated' is missing"]
    assert any("tornado" in p for p in S.lint(["simulated", "a tornado forms"]))
    assert any("marketing" in p for p in S.lint(["simulated", "a stunning storm"]))
    assert any("wind speed" in p for p in S.lint(["simulated", "fastest wind 32 m/s here"]))
    assert S.lint(["simulated", "max 36 m/s, min 11 m/s"], vertical=["max 36 m/s, min 11 m/s"]) == []


def test_ramps_rise_in_lightness_away_from_zero():
    for ramp in S.RAMPS.values():
        assert ramp.check() == [], ramp.name


def test_titles_keep_greek_letters_lower_case():
    assert S.upper(f"Lowest level θ{S.PRIME} and wind") == f"LOWEST LEVEL θ{S.PRIME} AND WIND"


def test_the_spin_range_is_fixed_for_a_grid_spacing():
    ranges = {dx: P.spin_range(dx) for dx in (50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)}
    assert ranges == {50.0: 0.2, 100.0: 0.1, 250.0: 0.05, 500.0: 0.02, 1000.0: 0.01, 2000.0: 0.005}
    assert P.spin_range(250.0) == P.SPIN_RANGE


def test_sampling_is_bilinear_and_nothing_more():
    field = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert P.bilinear(field, np.array([0.5]), np.array([1.5]))[0] == pytest.approx(0.25 * (1 + 2 + 5 + 6))
    assert P.bilinear(field, np.array([2.0]), np.array([3.0]))[0] == 11.0
    assert P.bilinear(field, np.array([-3.0]), np.array([9.0]))[0] == 3.0          # the edge value outside
    assert P.bilinear(field, np.array([-3.0]), np.array([9.0]), cval=-1.0)[0] == -1.0
    box = P.Box(0, 0, 4, 0.0, 0.0, 4000.0, h=3)
    # a pixel centred on a cell centre is that cell's value
    assert np.array_equal(P.sample(field, box, 1000.0, 1000.0), field[::-1])


# ---------------------------------------------------------------- reflectivity

def test_morrison_reflectivity_is_cm1s_calc_refl10cm_on_the_columns_with_hydrometeors(tmp_path):
    import importlib.util

    run = run_directory(tmp_path, times=(1500.0,))
    frame = sorted((run / "frames").glob("frame-*"))[0]
    qf = QuickFrame.open(frame)
    try:
        refl = R.volume(qf)
        assert refl.available and "calc_refl10cm" in refl.operator
        spec = importlib.util.spec_from_file_location("md", ROOT / "tools" / "morrison_dbz.py")
        md = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(md)
        full, cref = md.frame_dbz(frame)
        hydro = refl.dbz > R.FLOOR_DBZ
        assert hydro.any()
        assert np.array_equal(refl.dbz[hydro], full[hydro])       # the same words where it is drawn
        assert (full[~hydro] < 5.0).all()                           # nothing drawn where it was skipped
    finally:
        qf.close()


def test_a_scheme_with_its_own_reflectivity_uses_it_and_one_without_none(tmp_path):
    own = np.full((40, 24, 24), 42.0, np.float32)
    root = tmp_path / "frames"
    root.mkdir()
    publish_storm(root, 0, 900.0, dbz=own, component="wsm6-mp6")
    publish_storm(root, 1, 1200.0, component="thompson-mp8")
    frames = sorted(root.glob("frame-*"))
    qf = QuickFrame.open(frames[0])
    try:
        refl = R.volume(qf)
        assert np.array_equal(refl.dbz, own) and refl.operator.startswith("WSM6's own reflectivity")
    finally:
        qf.close()
    qf = QuickFrame.open(frames[1])
    try:
        refl = R.volume(qf)
        assert not refl.available and "Thompson" in refl.reason
    finally:
        qf.close()


def test_a_scheme_that_writes_its_own_reflectivity_says_why_a_frame_has_none(tmp_path):
    from cm1core.gm1_runtime import REFLECTIVITY_COMPONENTS

    assert set(R.OWN_REFLECTIVITY) == set(REFLECTIVITY_COMPONENTS)
    root = tmp_path / "frames"
    root.mkdir()
    publish_storm(root, 0, 0.0, component="wsm6-mp6")        # the starting frame: WSM6 has not run yet
    publish_storm(root, 1, 900.0, component="wsm6-mp6")      # a run that did not ask for output_dbz
    reasons = []
    for frame in sorted(root.glob("frame-*")):
        qf = QuickFrame.open(frame)
        try:
            refl = R.volume(qf)
            assert not refl.available
            reasons.append(refl.reason)
        finally:
            qf.close()
    assert "starting frame" in reasons[0] and "first step" in reasons[0]
    assert "output_dbz = 0" in reasons[1]
    # CM1's ihail reaches WSM6 too (param.F:3697-3706): the footer names the category the frame ran
    assert FA.scheme_words({"microphysics": "wsm6-mp6", "dense_ice": {"category": "hail", "ihail": 1}}) == \
        "WSM6 with hail"
    assert FA.scheme_words({"microphysics": "wsm6-mp6"}) == "WSM6"


@NEEDS_MATPLOTLIB
def test_no_reflectivity_is_said_in_the_index_not_drawn(tmp_path):
    root = tmp_path / "run" / "frames"
    root.mkdir(parents=True)
    publish_storm(root, 0, 900.0, component="thompson-mp8")
    end_event(tmp_path / "run" / "events.jsonl")
    _render(tmp_path / "run", only=("reflectivity",))
    index = json.loads((tmp_path / "run" / "quicklook" / "index.json").read_text())
    entry = index["products"][0]
    assert entry["frames"] == [] and "Thompson" in entry["not_drawn"][0]["reason"]


# ---------------------------------------------------------------- swaths

@NEEDS_MATPLOTLIB
def test_the_swath_is_the_running_maximum_on_the_ground(tmp_path):
    run = run_directory(tmp_path, swath="none")
    _render(run, only=("swath",))
    index = json.loads((run / "quicklook" / "index.json").read_text())
    frames = index["products"][0]["frames"]
    peaks = [f["facts"]["wind_max_ground_relative"] for f in frames]
    assert peaks == sorted(peaks) and frames[-1]["facts"]["source"] == "frames"
    # rebuilt by hand: each frame's lowest-level ground-relative speed placed at round(domainloc / dx)
    builder = SwathBuilder(tmp_path / "state")
    layout, rows = lineage(run)
    expected = []
    for t, key, path, seg in rows:
        qf = QuickFrame.open(path, key=key)
        try:
            u, v = qf.uv_level(0, ground=True)
            expected.append((round_half_away(qf.domainlocx / qf.dx), round_half_away(qf.domainlocy / qf.dy),
                             np.hypot(u, v)))
            builder.add(qf)
        finally:
            qf.close()
    sw = builder.swath(300.0)
    for ox, oy, speed in expected:
        window = sw.wind[oy - sw.origin_cells[1]:oy - sw.origin_cells[1] + 24, ox - sw.origin_cells[0]:ox - sw.origin_cells[0] + 24]
        assert (window >= speed.astype(np.float32) - 1e-6).all()
    assert float(sw.wind.max()) == pytest.approx(max(float(s.max()) for *_, s in expected))
    assert sw.extent_m == (1000.0 * sw.wind.shape[1], 1000.0 * sw.wind.shape[0])


@NEEDS_MATPLOTLIB
def test_the_models_ground_canvas_is_used_when_the_frames_have_it(tmp_path):
    run = run_directory(tmp_path, swath="canvas")
    _render(run, only=("swath",))
    facts = json.loads((run / "quicklook" / "index.json").read_text())["products"][0]["frames"][-1]["facts"]
    assert facts["source"] == "model" and facts["since"] == "since the start"
    layout, rows = lineage(run)
    qf = QuickFrame.open(rows[-1][2])
    try:
        plane = plane_of(qf)
        assert plane.source == "model" and plane.origin == (0, 0)
        assert np.array_equal(plane.wind, qf.get("sws2"))
    finally:
        qf.close()


def test_a_branch_the_run_left_is_folded_again(tmp_path):
    builder = SwathBuilder(tmp_path / "state")
    run = run_directory(tmp_path, swath="none")
    layout, rows = lineage(run)
    opener = lambda path, key, seg: QuickFrame.open(path, key=key, segment=seg)  # noqa: E731
    builder.catch_up(rows, opener)
    builder.save()
    again = SwathBuilder(tmp_path / "state")
    assert again.keys == [key for _t, key, _p, _s in rows]
    other = [(t, key + "-other", p, s) for t, key, p, s in rows[:2]]
    again.catch_up(other, opener)
    assert again.keys == [key for _t, key, _p, _s in other]


# ---------------------------------------------------------------- the pass, --follow and replacement

@NEEDS_MATPLOTLIB
def test_a_frame_is_drawn_once_and_a_later_segment_replaces_it(tmp_path):
    run = run_directory(tmp_path, times=(900.0, 1200.0))
    _render(run)
    first = (run / "quicklook" / "reflectivity" / "t001200.png").stat().st_mtime_ns
    assert _render(run).frames_drawn == 0
    assert (run / "quicklook" / "reflectivity" / "t001200.png").stat().st_mtime_ns == first
    # a new frame at a time already drawn (another key) is drawn again
    later = run / "frames"
    publish_storm(later, 2, 1200.0, scale=0.5)
    assert _render(run).frames_drawn == 1


@NEEDS_MATPLOTLIB
def test_follow_draws_each_frame_as_it_lands_and_ends_with_the_run(tmp_path):
    run = tmp_path / "run"
    root = run / "frames"
    root.mkdir(parents=True)
    (run / "namelist.input").write_text(NAMELIST_24.read_text())
    (run / "events.jsonl").write_text("")

    def publisher():
        for index, t in enumerate((900.0, 1200.0)):
            time.sleep(0.4)
            publish_storm(root, index, t)
        # a frame still being written is never read
        (root / ".incomplete" / "frame-000002").mkdir(parents=True)
        time.sleep(0.4)
        end_event(run / "events.jsonl", frames=2)

    thread = threading.Thread(target=publisher)
    thread.start()
    summary = _render(run, follow=True, poll_s=0.1)
    thread.join()
    assert summary.frames_drawn == 2 and summary.ended and summary.end_reason == "completed"
    index = json.loads((run / "quicklook" / "index.json").read_text())
    assert [f["time_s"] for f in index["frames"]] == [900.0, 1200.0]


@NEEDS_MATPLOTLIB
def test_the_time_series_carries_cm1s_curve_only_for_cm1s_stock_namelist(tmp_path):
    from cm1core.cm1out.settings import from_namelist

    stock = from_namelist(STOCK_NAMELIST)
    ref, why = SE.reference_for(stock)
    assert ref is not None and ref["id"] == "cm1-ptype5" and why == ""
    text = STOCK_NAMELIST.read_text()
    for edit, expected in (((r"tapfrq\s*=\s*900.0", "tapfrq = 300.0"), "cm1-ptype5"),
                           ((r"timax\s*=\s*7200.0", "timax = 1800.0"), "cm1-ptype5"),
                           ((r"ptype\s*=\s*5", "ptype = 7"), "cm1-ptype7"),
                           ((r"dx\s*=\s*1000.0", "dx = 500.0"), None)):
        path = tmp_path / "namelist.input"
        path.write_text(re.sub(edit[0], edit[1], text, count=1))
        ref, why = SE.reference_for(from_namelist(path))
        assert (ref["id"] if ref else None) == expected, edit
    run = run_directory(tmp_path / "a")
    _render(run)
    doc = json.loads((run / "quicklook" / "series.json").read_text())
    assert doc["schema"] == "gm1.quicklook-series/v1" and "cm1_label" not in doc
    assert "not CM1's stock supercell" in doc["cm1_reference"]
    assert [c["key"] for c in doc["charts"]][:6] == ["wmax", "wmin", "uh_max", "thpert_sfc_min",
                                                     "coldpool_area_km2_lt_minus1K", "rain_max_cm"]
    assert _png_size(run / "quicklook" / "timeseries.png") == (1920, 1080)


def test_the_stock_series_record_is_current():
    completed = subprocess.run([sys.executable, str(ROOT / "tools" / "generate_stock_supercell_series.py"), "--check"],
                               capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------- slices and the column skew-T

def _one(tmp_path):
    run = run_directory(tmp_path, times=(1500.0,))
    layout, rows = lineage(run)
    from cm1core.cm1out.settings import resolve

    settings = resolve(layout)
    qf = QuickFrame.open(rows[0][2], key=rows[0][1])
    facts = FA.run_facts(layout, qf.config, qf.dx, qf.dy, settings=settings)
    return run, qf, facts, settings


def test_slices_show_cm1out_fields_by_their_cm1_names(tmp_path, pictures):
    run, qf, facts, settings = _one(tmp_path)
    try:
        names = {row["name"] for row in SL.field_list(qf, settings)}
        assert {"w", "u", "v", "th", "qr", "dbz", "rain", "uinterp"} <= names
        out = run / "quicklook" / "slices"
        record = SL.horizontal(qf, facts, settings, "w", out, height_m=5000.0, line_km=(-8.0, -2.0, 8.0, 3.0),
                               tag="t001500")
        assert record["level"] == 11 and record["height_m"] == 5000.0 and record["range"] == [-50.0, 50.0]
        assert record["max"] == pytest.approx(float(np.load(ROOT / "tests/data/cm1-ref-24x24/stats-run-t1500.npz")["w"][10].max()))
        geo = record["map"]
        assert geo["x_km"] == pytest.approx([-12.0, 12.0]) and geo["y_km"] == pytest.approx([-12.0, 12.0])
        assert _png_size(out / record["file"]) == (1920, 1080)
        u = SL.horizontal(qf, facts, settings, "u", out, level=1, tag="t001500")
        assert u["level"] == 1
        # a field of one sign is drawn over its own range; one of both signs over a symmetric one
        assert u["range"] == [u["min"], u["max"]] if u["max"] < 0 or u["min"] > 0 else u["range"][0] == -u["range"][1]
        rain = SL.horizontal(qf, facts, settings, "rain", out, tag="t001500")
        assert rain["level"] is None and rain["file"] == "slice-rain-t001500.png"
        section = SL.vertical(qf, facts, settings, "w", out / "section.png", a_km=(-10.0, 0.0), b_km=(10.0, 0.0))
        assert section["length_km"] == pytest.approx(20.0) and section["top_m"] == 18000.0
        with pytest.raises(SL.SliceError, match="not a cm1out field"):
            SL.horizontal(qf, facts, settings, "nosuch", out)
        with pytest.raises(SL.SliceError, match="inside the domain"):
            SL.vertical(qf, facts, settings, "w", out / "x.png", a_km=(-30.0, 0.0), b_km=(10.0, 0.0))
        with pytest.raises(SL.SliceError, match="surface field"):
            SL.vertical(qf, facts, settings, "rain", out / "x.png", a_km=(-10.0, 0.0), b_km=(10.0, 0.0))
    finally:
        qf.close()
    words = "\n".join("\n".join(ov.texts) for _n, ov in pictures)
    assert "relative to the moving domain" in words        # CM1's u is relative to the moving domain
    for _name, ov in pictures:
        assert ov.check() == [] and ov.footer_cut == []


def test_without_cm1s_uh_the_updraft_helicity_is_cm1s_calcuh_sum(tmp_path):
    """misclibs.F:1513-1528: only cyclonic updrafts count (max(0, wbar) * max(0, zbar)), each level's depth
    clipped to 2-5 km AGL.  Written out here straight from the Fortran, on CM1's own 1500 s storm."""

    root = tmp_path / "frames"
    root.mkdir()
    publish_storm(root, 0, 1500.0, uh=False)
    qf = QuickFrame.open(sorted(root.glob("frame-*"))[0])
    try:
        assert not qf.has("uh")
        uh, source = qf.uh()
        assert source == "CM1's calcuh sum on the frame's winds"
        carriers, _frame = storm(1500.0)
        u, v, w = (np.asarray(carriers[n], np.float64) for n in ("u3d", "v3d", "w3d"))
        nz, ny, nx = w.shape[0] - 1, w.shape[1], w.shape[2]
        zf = np.arange(nz + 1) * 500.0
        zh = 0.5 * (zf[1:] + zf[:-1])
        expected = np.zeros((ny, nx))
        for k in range(nz):
            if not 2000.0 <= zh[k] <= 5000.0:
                continue
            # zeta(i,j) at the corner (xf(i), yf(j)), CM1 indices i = 1..ni+1, j = 1..nj+1 (interior corners here)
            zeta = np.full((ny + 1, nx + 1), np.nan)
            for j in range(1, ny):
                for i in range(1, nx):
                    zeta[j, i] = (v[k, j, i] - v[k, j, i - 1]) / 1000.0 - (u[k, j, i] - u[k, j - 1, i]) / 1000.0
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    wbar = max(0.0, 0.5 * (w[k, j, i] + w[k + 1, j, i]))
                    zbar = max(0.0, 0.25 * (zeta[j, i] + zeta[j, i + 1] + zeta[j + 1, i] + zeta[j + 1, i + 1]))
                    expected[j, i] += (min(zf[k + 1], 5000.0) - max(zf[k], 2000.0)) * wbar * zbar
        inner = (slice(1, ny - 1), slice(1, nx - 1))
        assert float(expected[inner].max()) > 10.0                  # the storm has a rotating updraft here
        assert np.allclose(uh[inner], expected[inner], rtol=1e-4, atol=1e-3)
        assert float(uh.min()) >= 0.0                               # a downdraft never counts
    finally:
        qf.close()


def test_a_column_without_theta_is_refused_not_drawn_from_the_perturbation(tmp_path):
    from cm1core.quicklook.frame import FrameUnreadable
    from cm1out_support import publish
    from quicklook_support import mark_microphysics

    carriers, frame = storm(1500.0)
    for name in ("th", "th0", "pi0", "prs0", "qv0"):
        carriers.pop(name)
    root = tmp_path / "frames"
    root.mkdir()
    path = publish(root, 0, carriers, **frame)
    mark_microphysics(path)
    qf = QuickFrame.open(path)
    try:
        facts = FA.RunFacts(name="x", version="0", version_source="", scheme="Morrison", component="morrison-mp10",
                            checked="", checked_source="", dx_m=1000.0, dy_m=1000.0)
        with pytest.raises(FrameUnreadable, match="neither th nor th3d"):
            SK.column(qf, facts, tmp_path / "column.png")
    finally:
        qf.close()


@NEEDS_MATPLOTLIB
def test_the_column_skew_t_is_the_frames_own_column(tmp_path, pictures):
    run, qf, facts, settings = _one(tmp_path)
    try:
        out = run / "quicklook" / "slices"
        record = SK.column(qf, facts, out / "column.png")
        w5 = qf.get("w3d", 10)
        j, i = np.unravel_index(int(np.argmax(w5)), w5.shape)
        assert (record["i"], record["j"]) == (i + 1, j + 1)
        assert record["x_km"] == pytest.approx(-12.0 + i + 0.5) and record["qv0_source"] == "the frame's qv0"
        at = SK.column(qf, facts, out / "column-at.png", x_km=3.2, y_km=-4.9)
        assert (at["x_km"], at["y_km"]) == (3.5, -4.5)
        assert _png_size(out / "column.png") == (1920, 1080)
        # the dewpoint is CM1's gettd
        assert SK.dewpoint_k(np.array([100000.0]), np.array([0.014]))[0] == pytest.approx(292.2, abs=0.1)
    finally:
        qf.close()
    for _name, ov in pictures:
        assert ov.check() == [] and ov.footer_cut == []


@NEEDS_MATPLOTLIB
def test_the_command_line_draws_slices_and_lists_fields(tmp_path):
    run = run_directory(tmp_path, times=(1500.0,))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    base = [sys.executable, "-m", "cm1core", "quicklook", str(run)]
    listed = subprocess.run(base + ["--fields", "--json"], capture_output=True, text=True, env=env)
    assert listed.returncode == 0, listed.stderr
    assert "w" in {row["name"] for row in json.loads(listed.stdout)["fields"]}
    section = subprocess.run(base + ["--section", "w", "--from", "-10", "-2", "--to", "10", "3", "--json"],
                             capture_output=True, text=True, env=env)
    assert section.returncode == 0, section.stderr
    record = json.loads(section.stdout)
    assert record["file"].startswith("slices/section-w-") and Path(record["path"]).is_file()
    column = subprocess.run(base + ["--column", "--at", "-3", "4", "--json"], capture_output=True, text=True, env=env)
    assert column.returncode == 0, column.stderr
    bad = subprocess.run(base + ["--slice", "nosuch"], capture_output=True, text=True, env=env)
    assert bad.returncode == 2 and "not a cm1out field" in bad.stderr


@NEEDS_MATPLOTLIB
def test_a_folder_of_frames_alone_takes_its_namelist_from_the_command_line(tmp_path):
    run = run_directory(tmp_path, times=(1500.0,))
    namelist = tmp_path / "elsewhere.namelist.input"
    (run / "namelist.input").rename(namelist)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    base = [sys.executable, "-m", "cm1core", "quicklook", str(run)]
    refused = subprocess.run(base + ["--fields"], capture_output=True, text=True, env=env)
    assert refused.returncode == 2 and "--namelist FILE" in refused.stderr
    listed = subprocess.run(base + ["--fields", "--namelist", str(namelist), "--json"], capture_output=True, text=True,
                            env=env)
    assert listed.returncode == 0, listed.stderr
    assert "w" in {row["name"] for row in json.loads(listed.stdout)["fields"]}
    column = subprocess.run(base + ["--column", "--namelist", str(namelist), "--json"], capture_output=True,
                            text=True, env=env)
    assert column.returncode == 0, column.stderr
    assert json.loads(column.stdout)["x_km"] < 0.0            # iorigin = 2 from the namelist: x = 0 in the middle
    drawn = subprocess.run(base + ["--only", "rain", "--namelist", str(namelist), "--json"], capture_output=True,
                           text=True, env=env)
    assert drawn.returncode == 0, drawn.stderr
    missing = subprocess.run(base + ["--only", "rain", "--namelist", str(tmp_path / "nosuch"), "--json"],
                             capture_output=True, text=True, env=env)
    assert missing.returncode == 2 and "nosuch" in missing.stderr


# ---------------------------------------------------------------- the worker

@NEEDS_MATPLOTLIB
def test_the_worker_runs_on_the_cpu_at_nice_19(tmp_path):
    run = run_directory(tmp_path, times=(900.0,))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "0"}
    completed = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(run), "--only", "rain", "--json"],
                               capture_output=True, text=True, env=env)
    assert completed.returncode == 0, completed.stderr
    worker = json.loads(completed.stdout)["worker"]
    assert worker["nice"] == 19 and worker["cuda_visible_devices"] == "" and worker["cupy_imported"] is False
    assert json.loads((run / "quicklook" / "index.json").read_text())["worker"]["nice"] == 19


def test_the_worker_lowers_its_own_priority_from_any_start(monkeypatch):
    """The subprocess test above inherits pytest's niceness (often 19 already); this one starts the worker's
    logic at nice 0 and at nice 5 and checks it asks for exactly the rest of the way to 19, and hides the GPU."""

    from cm1core.quicklook import worker as WK

    for start in (0, 5, 19):
        level = {"nice": start}
        calls = []

        def fake_nice(increment, level=level, calls=calls):
            calls.append(increment)
            level["nice"] += increment
            return level["nice"]

        monkeypatch.setattr(WK.os, "nice", fake_nice)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        for name in (*WK.THREAD_VARIABLES, "MPLBACKEND"):
            monkeypatch.delenv(name, raising=False)
        facts = WK.worker_environment()
        assert level["nice"] == 19 and [c for c in calls if c] == ([19 - start] if start < 19 else [])
        assert facts["nice"] == 19 and facts["cuda_visible_devices"] == "" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert all(os.environ[name] == "1" for name in WK.THREAD_VARIABLES)


def test_the_package_names_no_home_paths_runs_storms_or_places():
    banned = re.compile(r"/home\b|El ?Reno|elreno|E100|Weisman|2011|Moore|Joplin|VORTEX2|tornad", re.I)
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [m.group(0) for m in banned.finditer(text)]
        # "tornado" appears only in the honesty rule that refuses it, and in the swath record's field name
        allowed = {"tornad"} if path.name in ("style.py", "swath.py", "products.py") else set()
        assert not [h for h in hits if h.lower() not in allowed], (path.name, hits)
    assert not re.search(r"/home\b", "\n".join(p.read_text(errors="replace") for p in PACKAGE.rglob("*.txt")))
