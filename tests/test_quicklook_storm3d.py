"""The storm in 3-D (``gpuwm-cm1 quicklook RUNDIR --look storm3d`` and ``--film storm3d``), on the CPU.

The frames carry CM1 r21.1's own 24 x 24 x 40 storm (``quicklook_support``).  Here:

1. a still: the storm cut open through its reflectivity core, at true vertical scale, every word linted, the
   footer whole, listed in ``index.json`` as a look drawn on request and drawn again for another time;
2. the bounds: a box larger than the cell budget is averaged over s x s columns and says so; a request whose
   estimate is too long is refused, with the estimate and a smaller choice, before the scene is built;
3. plain refusals: no cloud yet, no echo yet (a cloud with no rain), a bearing that is not a number; a run with
   no reflectivity draws its precipitation water, and refuses when it has none yet;
4. a hurricane is cut through its centre, and a wide box has its heights stretched by a stated factor;
5. the cut face is a cross-section in the table's colours, nearly solid, and says so;
6. the orbit film: one turn of N pictures, the cut always square to the camera, every note on every picture, and
   a GIF whose legend keeps the table's colours;
7. the worker never draws it as frames land, a pass with --force keeps it, and drawing it changes no other
   picture;
8. the Results screen's row, refusal, buttons and estimate, kept equal to the worker's without numpy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import warnings

import numpy as np
import pytest

from cm1import_support import namelist_tool
from cm1out_support import end_event, publish
from quicklook_support import NAMELIST_24, mark_microphysics, run_directory, storm

from cm1core.cm1out import frames as _frames
from cm1core.cm1out import settings as _settings
from cm1core.gui import looks as GL
from cm1core.quicklook import film as F
from cm1core.quicklook import products as P
from cm1core.quicklook import render as RD
from cm1core.quicklook import storm3d as D
from cm1core.quicklook import style as S
from cm1core.quicklook.facts import run_facts
from cm1core.quicklook.frame import QuickFrame, frames_of

SMALL = (960, 540)


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


@pytest.fixture
def words(monkeypatch):
    """Every overlay a still or film picture saves, so a test can read its words."""

    seen: list = []
    real_save, real_picture = P.save, F.save_picture

    def save(ov, base, path):
        seen.append(ov)
        return real_save(ov, base, path)

    def picture(ov, base, path):
        seen.append(ov)
        return real_picture(ov, base, path)

    monkeypatch.setattr(P, "save", save)
    monkeypatch.setattr(F, "save_picture", picture)
    return seen


def _looks(run: Path, when: float | None = None) -> P.Looks:
    layout = _frames.discover(run)
    t, key, path, segment = D.pick(frames_of(layout), when)
    qf = QuickFrame.open(path, key=key, segment=segment)
    facts = run_facts(layout, qf.config, qf.dx, qf.dy, settings=_settings.resolve(layout))
    return P.Looks(qf, facts)


def _text(ov) -> str:
    return "\n".join(ov.texts)


def _hashes(folder: Path) -> dict[str, str]:
    return {os.fspath(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*.png")) if "storm3d" not in p.parts}


# ---------------------------------------------------------------- the still

@pytest.fixture(autouse=True)
def whole_footers():
    """A footer cut to fit warns (quicklook.products.save); here that fails the test."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


def test_the_still_cuts_the_storm_open_and_says_how_it_was_drawn(tmp_path, words):
    run = run_directory(tmp_path)
    record = D.draw_still(run)
    path = Path(record["path"])
    assert path == run / "quicklook" / "storm3d" / "t001500.png" and path.is_file()
    assert (run / "quicklook" / "storm3d" / "thumb" / "t001500.png").is_file()
    from PIL import Image

    with Image.open(path) as im:
        assert im.size == D.STILL_SIZE
        pixels = np.asarray(im.convert("RGB"), np.float32)
    # something storm-coloured is drawn: the warm NWS colours of the 45 and 60 dBZ shells in the middle
    middle = pixels[250:820, 300:1600]
    assert ((middle[..., 0] > middle[..., 1] + 40) & (middle[..., 0] > middle[..., 2] + 60)).sum() > 500
    facts = record["facts"]
    assert facts["field"] == "reflectivity" and facts["zscale"] == 1.0 and facts["stride"] == 1
    assert "calc_refl10cm" in facts["operator"] and facts["peak"] > 45.0
    assert 0.0 < facts["cut_xy_m"][0] < 24_000.0 and 0.0 < facts["cut_xy_m"][1] < 24_000.0
    text = _text(words[-1])
    for said in ("simulated", "CPU volume render, not a radar or a photograph", "true vertical scale",
                 "cut open through the reflectivity core", "the half nearer the camera is not drawn",
                 "shells: reflectivity 30, 45, 60 dBZ", "see-through (6% of its depth drawn)",
                 "shells at each ▼, opacity as marked", "15%", "40%", "90%",
                 "cut face: the reflectivity on the cut, in bands from 30, 45 and 60 dBZ, 85% opaque",
                 "every model cell drawn (1 km)", "smoothed 1-2-1 over 3 cells", "a look, not a measurement",
                 "height on the cut face", "dBZ", "looking NNW", "T+0:25:00"):
        assert said in text, said
    # the shells' numbers are opacities, and the box drawn is said with its size (here the whole 24 km domain)
    assert "see-through as marked" not in text
    box = (facts["box_cells"][1] - facts["box_cells"][0], facts["box_cells"][3] - facts["box_cells"][2])
    assert f"strongest reflectivity in the {box[0]} × {box[1]} km box drawn: {facts['peak']:.0f} dBZ" in text
    # the picture lint (no storm word the GUI never says, simulated, wind frames) passes; the footer is whole
    assert words[-1].check() == [] and not words[-1].footer_cut
    # listed as a look drawn on request, and kept apart from the worker's state
    index = json.loads((run / "quicklook" / "index.json").read_text())
    entry = next(p for p in index["products"] if p["id"] == "storm3d")
    assert entry["on_request"] is True and [f["time_s"] for f in entry["frames"]] == [1500.0]
    assert entry["frames"][0]["file"] == "storm3d/t001500.png"
    # another time is added beside it, oldest first, and the same time again replaces its picture
    D.draw_still(run, time_s=900.0, size=SMALL)
    D.draw_still(run, time_s=1500.0, size=SMALL)
    entry = next(p for p in json.loads((run / "quicklook" / "index.json").read_text())["products"]
                 if p["id"] == "storm3d")
    assert [f["time_s"] for f in entry["frames"]] == [900.0, 1500.0]
    with Image.open(path) as im:
        assert im.size == SMALL


def test_the_camera_stands_where_asked_and_the_cut_faces_it(tmp_path, words):
    run = run_directory(tmp_path, times=(1500.0,))
    L = _looks(run)
    plan = D.plan_for(L)
    scene = D.build_scene(L, plan)
    for bearing, looking in ((150.0, "NNW"), (0.0, "S"), (270.0, "E")):
        cam = D.camera_for(scene, bearing, *SMALL)
        assert D.compass_name(cam.heading_deg) == looking
        point, normal = cam.clip
        fwd, _right, _up = cam.basis()
        # the cut is upright, through the core, and square to the camera's level view
        assert normal[2] == 0.0 and np.allclose(point[:2], plan.cut_xy)
        assert np.allclose(normal[:2] / np.linalg.norm(normal[:2]), fwd[:2] / np.linalg.norm(fwd[:2]))
        # the core, half way up the storm, is in the picture
        px, py, depth = cam.project(np.array([[plan.cut_xy[0], plan.cut_xy[1], 0.5 * plan.top_m]]))
        assert depth[0] > 0 and 0 < px[0] < SMALL[0] and 0 < py[0] < SMALL[1]
        # the cut face's ends (where the cut meets what is drawn) are framed too, clear of the picture's sides
        lo, hi = scene.meta["content"]
        right_h = np.array([_right[0], _right[1], 0.0]) / np.linalg.norm(_right[:2])
        ends = []
        for a in np.linspace(-80_000.0, 80_000.0, 1601):
            q = np.array([plan.cut_xy[0], plan.cut_xy[1], 0.0]) + a * right_h
            if lo[0] <= q[0] <= hi[0] and lo[1] <= q[1] <= hi[1]:
                ends.append(q)
        ends = np.array([ends[0], ends[-1], ends[0] + [0, 0, hi[2]], ends[-1] + [0, 0, hi[2]]])
        ex, _ey, ez = cam.project(ends)
        assert np.all(ez > 0) and np.all(np.abs((ex + 0.5) * 2.0 / SMALL[0] - 1.0) <= D.FRAME_X + 0.01)


def test_the_cut_face_is_a_cross_section_in_the_tables_colours(tmp_path):
    """On the cut face the 30 to 45 dBZ band is the table's green, nearly solid (CUT_FACE_OPACITY): not a dark tint
    that reads like the weak-echo hole beside it."""

    run = run_directory(tmp_path, times=(1500.0,))
    L = _looks(run)
    plan = D.plan_for(L)
    scene = D.build_scene(L, plan)
    cam = D.camera_for(scene, D.BEARING, *SMALL)
    shot = D.render(scene, cam)
    _fwd, right, _up = cam.basis()
    right_h = np.array([right[0], right[1], 0.0]) / np.linalg.norm(right[:2])
    base = np.array([plan.cut_xy[0], plan.cut_xy[1], 0.0])
    pts = np.array([base + a * right_h + [0.0, 0.0, z] for a in np.linspace(-9000.0, 9000.0, 73)
                    for z in np.linspace(500.0, 12_000.0, 47)])

    def field_at(q):
        fk, fj, fi = D._index(scene, q[:, 0], q[:, 1], q[:, 2])
        return D.trilinear(scene.vol, scene.nk, scene.ny, scene.nx, fk, fj, fi)

    here = field_at(pts)
    # inside the band: the field 32 to 43 dBZ here and half a cell around, and nothing faded
    steady = (here[:, 4] >= 32.0) & (here[:, 4] <= 43.0) & (here[:, 5] > 0.95)
    for shift in ([0.0, 0.0, -0.5 * scene.dxr], [0.0, 0.0, 0.5 * scene.dxr], -0.5 * scene.dxr * right_h,
                  0.5 * scene.dxr * right_h):
        around = field_at(pts + shift)[:, 4]
        steady &= (around >= 30.5) & (around <= 44.5)
    assert steady.sum() >= 5
    px, py, _ = cam.project(pts[steady])
    rgb = shot["rgb"][np.round(py).astype(int), np.round(px).astype(int)]
    green = np.array(S.hex_rgb(dict(S.NWS_REFLECTIVITY)[30]))
    assert green[1] > 0.5 and green[0] == 0.0
    # mostly the table's green at nearly its own brightness (the old 15% tint was about half of it)
    assert np.median(rgb[:, 1]) > 0.75 * D.CUT_FACE_SHADE * green[1]
    assert np.mean(rgb[:, 1] > rgb[:, 0] + 0.15) >= 0.8 and np.mean(rgb[:, 1] > rgb[:, 2] + 0.15) >= 0.8


# ---------------------------------------------------------------- bounds

def test_a_box_larger_than_the_budget_is_averaged_and_says_so(tmp_path, monkeypatch, words):
    run = run_directory(tmp_path, times=(1500.0,))
    monkeypatch.setattr(D, "VOXEL_BUDGET", 6000.0)
    record = D.draw_still(run, size=SMALL)
    facts = record["facts"]
    assert facts["stride"] >= 2 and facts["voxel_m"] == facts["stride"] * 1000.0
    assert facts["cells"] <= 6000 * 1.0001
    text = _text(words[-1])
    assert f"drawn from {facts['stride']} × {facts['stride']} cell averages" in text
    assert "to keep it to 0.006 million cells" in text


def test_a_request_that_would_take_too_long_is_refused_before_the_scene_is_built(tmp_path, monkeypatch):
    run = run_directory(tmp_path, times=(1500.0,))
    monkeypatch.setattr(D, "STILL_LIMIT_S", 0.001)
    built = []
    monkeypatch.setattr(D, "build_scene", lambda *a, **k: built.append(1))
    with pytest.raises(D.Refused, match=r"This picture would take about .* on one core, more than the "
                                         r"0.001-second limit. Even the smallest picture \(960x540\) would take"):
        D.draw_still(run)
    assert built == []
    # the Results screen says why, in the same words, where the picture would be
    RD.render(run, only=("reflectivity",))
    index = json.loads((run / "quicklook" / "index.json").read_text())
    row = {r["id"]: r for r in GL.more_looks("storm", run, index, {"state": "finished"}, films_now={})}["storm3d"]
    assert row["why_not"].startswith("This picture would take about") and row["action"] == "look"
    monkeypatch.setattr(D, "ORBIT_LIMIT_S", 0.001)
    job = F.FilmJob(run, "storm3d", ("apng",), 12.0, 0, run / "quicklook", pictures=12, size=SMALL)
    with pytest.raises(F.FilmError, match=r"This orbit would take about .* Even 12 pictures at 960x540"):
        F.make(job)
    assert built == []


def test_cm1_netcdf_output_is_cut_open_and_orbited_from_its_files(tmp_path):
    """CM1's own netCDF output (Runs, Open CM1 output) has the storm in 3-D too: the still and the orbit film read
    the files as every other look does.  CM1's 24 x 24 file has no echo yet, and both say so in the same words."""

    source = tmp_path / "cm1run"
    source.mkdir()
    shutil.copy(Path(__file__).resolve().parent / "data" / "cm1-ref-24x24" / "cm1out_000001.nc",
                source / "cm1out_000001.nc")
    out = tmp_path / "pictures"
    with pytest.raises(P.NotDrawn, match="No echo of 30 dBZ or more yet"):
        D.draw_still(source, out=out)
    job = F.FilmJob(source, "storm3d", ("apng",), 12.0, 0, out, pictures=12, size=SMALL)
    with pytest.raises(F.FilmError, match="No echo of 30 dBZ or more yet"):
        F.make(job)
    assert sorted(p.name for p in source.iterdir()) == ["cm1out_000001.nc"]


def test_a_refusal_offers_the_largest_choice_that_fits(monkeypatch):
    """The smaller size or picture count a refusal offers is itself within the limit, and the next larger is not."""

    import re

    plan = D.Plan(kind="thunderstorm", i0=0, i1=192, j0=0, j1=192, stride=1, nk=72, top_m=17000.0, zscale=1.0,
                  cut_xy=(0.0, 0.0), cut_words="", field="reflectivity", operator="", shells=D.SHELLS_DBZ,
                  cells_read=192 * 192 * 72, voxels=192 * 192 * 72, dx_m=250.0, domain_cells=480 * 480 * 80)
    # a still: with the limit between two sizes, the offer is the larger one that fits
    limit = 0.5 * (D.estimate_seconds(plan, (1280, 720)) + D.estimate_seconds(plan, (1600, 900)))
    monkeypatch.setattr(D, "STILL_LIMIT_S", limit)
    words = D.too_long(plan, (1920, 1080), 1)
    assert words.startswith("This picture would take about ") and words.endswith("such as --size 1280x720.")
    # an orbit: the picture count offered fits and one more does not; a smaller size offered fits too
    words = D.too_long(plan, (3840, 2160), 360)
    frames = int(re.search(r"--orbit-frames (\d+)", words).group(1))
    assert 12 <= frames < 360
    assert D.estimate_seconds(plan, (3840, 2160), frames) <= D.ORBIT_LIMIT_S
    assert D.estimate_seconds(plan, (3840, 2160), frames + 1) > D.ORBIT_LIMIT_S
    size = re.search(r"--size (\d+)x(\d+)", words)
    assert size is not None
    assert D.estimate_seconds(plan, (int(size.group(1)), int(size.group(2))), 360) <= D.ORBIT_LIMIT_S


def test_the_estimate_grows_with_the_box_and_the_pictures():
    plan = D.Plan(kind="thunderstorm", i0=0, i1=192, j0=0, j1=192, stride=1, nk=72, top_m=17000.0, zscale=1.0,
                  cut_xy=(0.0, 0.0), cut_words="", field="reflectivity", operator="", shells=D.SHELLS_DBZ,
                  cells_read=192 * 192 * 72, voxels=192 * 192 * 72, dx_m=250.0, domain_cells=480 * 480 * 80)
    one = D.estimate_seconds(plan, D.STILL_SIZE)
    # measured on one core for this box (250 m, 2.7 million cells): 9 s to find the core and build, 31 s to draw
    assert 40.0 <= one <= 120.0
    assert D.estimate_seconds(plan, D.ORBIT_SIZE, 72) > 15 * one
    # the server's estimate, made without numpy from the index's grid words, errs long
    index = {"run": {"grid": "480 × 480 × 80, dx 250 m"}}
    assert GL.storm3d_seconds(index) >= one
    assert GL.about_seconds(index, "storm3d") > GL.about_seconds(index, "storm3d-still")


# ---------------------------------------------------------------- plain refusals and other runs

def test_no_cloud_yet_is_said_plainly(tmp_path):
    run = tmp_path / "quiet"
    (run / "frames").mkdir(parents=True)
    (run / "namelist.input").write_bytes(NAMELIST_24.read_bytes())
    carriers, frame = storm(1500.0)
    carriers["q3d"] = carriers["q3d"].copy()
    carriers["q3d"][1:] = 0.0
    mark_microphysics(publish(run / "frames", 0, carriers, **frame))
    end_event(run / "events.jsonl", frames=1)
    with pytest.raises(P.NotDrawn, match="No echo of 30 dBZ or more yet at this time|No cloud yet at this time"):
        D.draw_still(run, size=SMALL)


def test_a_cloud_with_no_echo_yet_is_not_cut_open(tmp_path):
    """Early in a run (cloud, no rain or hail yet) there is nothing to cut open: no 'core' of -35 dBZ is drawn, and
    the Results screen says why where the picture would be."""

    run = tmp_path / "early"
    (run / "frames").mkdir(parents=True)
    (run / "namelist.input").write_bytes(NAMELIST_24.read_bytes())
    carriers, frame = storm(1500.0)
    carriers["q3d"] = carriers["q3d"].copy()
    for n in (2, 4, 5):          # qr, qs, qg: no precipitation; the cloud (qc, qi) stays
        carriers["q3d"][n] = 0.0
    assert carriers["q3d"][1].max() > 1e-4
    mark_microphysics(publish(run / "frames", 0, carriers, **frame))
    end_event(run / "events.jsonl", frames=1)
    with pytest.raises(P.NotDrawn, match=r"No echo of 30 dBZ or more yet at this time \(the strongest 1 to 5 km up "
                                         r"is −35 dBZ\): there is no rain or hail to cut open. Draw a later saved "
                                         r"time."):
        D.draw_still(run, size=SMALL)
    failed = json.loads((run / "quicklook" / "storm3d" / "failed.json").read_text(encoding="utf-8"))
    assert failed["message"].startswith("No echo of 30 dBZ")
    # the orbit is refused in the same words
    job = F.FilmJob(run, "storm3d", ("apng",), 12.0, 0, run / "quicklook", pictures=12, size=SMALL)
    with pytest.raises(F.FilmError, match="No echo of 30 dBZ or more yet"):
        F.make(job)


def test_a_stronger_core_in_the_strip_along_the_domain_edge_is_said(tmp_path, words):
    """No cut is made within 8 cells of the domain's edge; when the strongest core lies there, the picture says the
    cut is through the strongest core inside, and that a stronger one lies nearer the edge."""

    note = "the cut is through the strongest core at least 8 km inside the domain; a stronger one lies nearer its edge"
    run = run_directory(tmp_path, times=(1500.0,))      # CM1's storm sits 8.5 km from the domain's west edge
    D.draw_still(run, size=SMALL)
    assert note in _text(words[-1])
    # the same storm moved 8 cells east, away from the edge: cut through its core, and no note
    moved = tmp_path / "moved"
    (moved / "frames").mkdir(parents=True)
    (moved / "namelist.input").write_bytes(NAMELIST_24.read_bytes())
    carriers, frame = storm(1500.0)
    for name, value in list(carriers.items()):
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[-1] in (24, 25) and name not in (
                "sws2", "shs2"):
            carriers[name] = np.roll(value, 8, axis=-1).copy()
    mark_microphysics(publish(moved / "frames", 0, carriers, **frame))
    end_event(moved / "events.jsonl", frames=1)
    plan = D.plan_for(_looks(moved))
    assert plan.cut_note == "" and plan.cut_xy[0] == 13_500.0


def test_a_bearing_that_is_not_a_number_is_refused_before_anything_is_read(tmp_path):
    from cm1core.quicklook import door

    run = run_directory(tmp_path, times=(1500.0,))
    for bad in (float("nan"), float("inf"), "east"):
        with pytest.raises(ValueError, match="--bearing takes a compass bearing in degrees, such as 240"):
            D.draw_still(run, bearing=bad, size=SMALL)
    assert not (run / "quicklook" / "storm3d").exists()
    assert door._bearing(None) is None and door._bearing(400.0) == 400.0
    with pytest.raises(ValueError, match="--bearing takes a compass bearing"):
        door._bearing(float("nan"))


def test_a_time_that_is_not_a_storm_time_is_refused_before_anything_is_drawn(tmp_path):
    import subprocess
    import sys

    from cm1core.quicklook import door

    run = run_directory(tmp_path, times=(1500.0,))
    for odd in (float("nan"), float("inf"), -5.0):
        with pytest.raises(ValueError, match="--time is a storm time in seconds, 0 or more."):
            D.draw_still(run, time_s=odd, size=SMALL)
        with pytest.raises(ValueError, match="--time is a storm time in seconds, 0 or more."):
            door._storm_time(odd)
    assert door._storm_time(None) is None and door._storm_time(0.0) == 0.0
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "CUDA_VISIBLE_DEVICES": ""}
    base = [sys.executable, "-m", "cm1core", "quicklook", str(run)]
    for args in (["--look", "storm3d", "--time", "nan"], ["--look", "storm3d", "--time", "inf"],
                 ["--film", "storm3d", "--time", "nan"], ["--film", "storm3d", "--time=-inf"]):
        refused = subprocess.run(base + args, capture_output=True, text=True, env=env)
        assert refused.returncode == 2 and "--time is a storm time in seconds, 0 or more." in refused.stderr, args
        assert "Traceback" not in refused.stderr, args
    assert not (run / "quicklook" / "storm3d").exists() and not (run / "quicklook" / "films").exists()


def test_a_run_without_reflectivity_draws_its_precipitation_water(tmp_path, words):
    run = run_directory(tmp_path, times=(1500.0,), component="kessler")
    record = D.draw_still(run, size=SMALL)
    assert record["facts"]["field"] == "water" and record["facts"]["shells"] == [list(s) for s in D.SHELLS_WATER]
    text = _text(words[-1])
    assert "shells: rain and graupel or hail 0.1, 1, 3 g/m³" in text
    assert "no reflectivity in this run: precipitation water instead" in text
    assert "cut open through the precipitation water core" in text
    assert "cut face: the rain and graupel or hail on the cut, in bands from 0.1, 1 and 3 g/m³, 85% opaque" in text
    assert "shells: rain and graupel or hail, g/m³ (opacity as marked)" in text


def test_a_run_without_reflectivity_and_without_rain_yet_is_refused(tmp_path):
    run = tmp_path / "dry"
    (run / "frames").mkdir(parents=True)
    (run / "namelist.input").write_bytes(NAMELIST_24.read_bytes())
    carriers, frame = storm(1500.0)
    carriers["q3d"] = carriers["q3d"].copy()
    for n in (2, 5):
        carriers["q3d"][n] *= 1e-5
    mark_microphysics(publish(run / "frames", 0, carriers, **frame), "kessler")
    end_event(run / "events.jsonl", frames=1)
    with pytest.raises(P.NotDrawn, match=r"No rain, graupel or hail of 0.1 g/m³ or more yet at this time \(the most "
                                         r"1 to 5 km up is 0.0\d g/m³\): there is nothing to cut open"):
        D.draw_still(run, size=SMALL)


def test_a_schemes_own_reflectivity_is_drawn_as_it_is_saved(tmp_path, words):
    """A run whose scheme writes its own reflectivity (WSM6's dbz carrier) is cut and drawn from that carrier."""

    carriers, _frame = storm(1500.0)
    rain = carriers["q3d"][2] + carriers["q3d"][5]
    dbz = np.where(rain > 1e-6, 10.0 * np.log10(np.maximum(rain, 1e-12) * 1e9), -35.0).astype(np.float32)
    run = run_directory(tmp_path, times=(1500.0,), component="wsm6-mp6", dbz=dbz)
    record = D.draw_still(run, size=SMALL)
    facts = record["facts"]
    assert facts["field"] == "reflectivity" and facts["operator"] == "WSM6's own reflectivity"
    assert abs(facts["peak"] - float(dbz.max())) < 1e-3
    text = _text(words[-1])
    assert "shells: reflectivity 30, 45, 60 dBZ, WSM6's own reflectivity" in text


def test_a_hurricane_is_cut_through_its_centre_and_a_wide_box_is_stretched_by_a_stated_factor(tmp_path, words):
    run = run_directory(tmp_path, times=(1500.0,))
    L = _looks(run)
    L.facts = dataclasses.replace(L.facts, storm_kind="hurricane")
    plan = D.plan_for(L)
    assert plan.kind == "hurricane" and "centre (the lowest pressure at the lowest level)" in plan.cut_words
    p0 = L.f.pressure_level(0)
    assert p0 is not None
    # a 24 km box is drawn at true scale; a hurricane's 500 km box 10 times stretched, a 1000 km one 20 times
    assert plan.zscale == 1.0
    assert D.height_stretch(48_000.0, 16_000.0) == 1.0
    assert D.height_stretch(504_000.0, 16_500.0) == 10.0
    assert D.height_stretch(1_024_000.0, 16_000.0) == 20.0
    D.still(L, tmp_path / "h.png", size=SMALL, plan=dataclasses.replace(plan, zscale=10.0))
    text = _text(words[-1])
    assert "heights stretched 10 times" in text and "height on the cut face, stretched 10 times" in text


# ---------------------------------------------------------------- the orbit film

def test_the_orbit_film_turns_once_around_the_storm(tmp_path, monkeypatch, words):
    run = run_directory(tmp_path, times=(900.0, 1500.0))
    monkeypatch.setattr(F, "mp4_available", lambda: False)
    job = F.FilmJob(run, "storm3d", ("apng", "gif"), 12.0, 0, run / "quicklook", time_s=900.0, pictures=12,
                    size=SMALL)
    record = F.make(job)
    assert record["pictures"] == 12 and record["fps"] == 12.0 and record["size"] == list(SMALL)
    assert record["title"] == "The storm in 3-D, one turn around it"
    assert record["facts"]["degrees_per_picture"] == 30.0 and record["facts"]["time_s"] == 900.0
    with pytest.raises(ValueError, match="from 960x540 to 3840x2160 pixels, not 480x270"):
        D.orbit(_looks(run), tmp_path / "small", pictures=12, size=(480, 270))
    assert F.read_apng_frames(run / "quicklook" / "films" / "storm3d" / "film.png")[0] == 12
    headings = [D.compass_name(ov_heading) for ov_heading in
                [(D.BEARING + 180.0 + 30.0 * n) % 360.0 for n in range(12)]]
    looking = [next(t for t in ov.texts if t.startswith("looking ")) for ov in words[-12:]]
    assert looking == [f"looking {h}" for h in headings]
    for ov in words[-12:]:
        assert "T+0:15:00" in _text(ov) and "simulated" in _text(ov) and not ov.check()
        # the film's pictures say all the still says: what the white is, the cut face, and that the light is a look
        for said in ("white: cloud water, ice and snow, see-through", "shells at each ▼, opacity as marked",
                     "cut face: the reflectivity on the cut", "every model cell drawn", "a look, not a measurement",
                     "CPU volume render, not a radar or a photograph"):
            assert said in _text(ov), said
        assert not ov.footer_cut
    # the GIF: one palette for every picture, holding the reflectivity table's colours, so the legend keeps them
    from PIL import Image

    table = {S.rgb8(c) for _, c in S.NWS_REFLECTIVITY}
    with Image.open(run / "quicklook" / "films" / "storm3d" / "film.gif") as gif:
        assert gif.n_frames == 12 and gif.size == F.GIF_SIZE
        for n in range(gif.n_frames):
            gif.seek(n)
            seen = {tuple(c) for c in np.asarray(gif.convert("RGB")).reshape(-1, 3)}
            assert table <= seen, sorted(table - seen)
    progress = json.loads((run / "quicklook" / "films" / "storm3d" / "progress.json").read_text())
    assert progress["state"] == "done" and progress["done"] == 12


# ---------------------------------------------------------------- the worker and the other pictures

def test_the_worker_never_draws_it_and_drawing_it_changes_no_other_picture(tmp_path):
    run = run_directory(tmp_path)
    RD.render(run)
    out = run / "quicklook"
    assert not (out / "storm3d").exists()
    assert "storm3d" not in {p["id"] for p in json.loads((out / "index.json").read_text())["products"]}
    before = _hashes(out)
    D.draw_still(run, size=SMALL)
    assert _hashes(out) == before
    # the next pass keeps the still listed, and draws nothing again
    RD.render(run)
    assert _hashes(out) == before
    index = json.loads((out / "index.json").read_text())
    assert [p["id"] for p in index["products"]][-1] == "storm3d"
    # a pass drawn again from nothing (--force) keeps them: it never draws them again
    stills = sorted(p.name for p in (out / "storm3d").glob("*.png"))
    RD.render(run, force=True)
    assert sorted(p.name for p in (out / "storm3d").glob("*.png")) == stills and stills
    index = json.loads((out / "index.json").read_text())
    assert [p["id"] for p in index["products"]][-1] == "storm3d"


def test_the_top_label_marks_a_top_that_is_seen_to_the_nearest_250_m(tmp_path, words):
    run = run_directory(tmp_path, times=(1500.0,))
    L = _looks(run)
    plan = D.plan_for(L)
    scene = D.build_scene(L, plan)
    cam = D.camera_for(scene, D.BEARING, *SMALL)
    shot = D.render(scene, cam)
    ov = S.Overlay(SMALL)
    D._top_label(ov, cam, scene, shot["trans"], 0.0)
    label = [t for t in ov.texts if t.endswith(" top")]
    assert len(label) == 1
    km = float(label[0].split()[0])
    assert abs(km * 4.0 - round(km * 4.0)) < 1e-9 and 1.0 <= km <= plan.top_m / 1000.0 + 0.25
    # nothing seen anywhere (the drawing see-through everywhere): no top is marked
    ov = S.Overlay(SMALL)
    D._top_label(ov, cam, scene, np.ones_like(shot["trans"]), 0.0)
    assert not [t for t in ov.texts if "top" in t]
    # a cloud at the model's last level says so, not a "top" at the model's lid
    scene.meta["model_nz"] = scene.nk
    tops = scene.meta["tops_k"].copy()
    scene.meta["tops_k"] = np.where(tops >= 0, scene.nk - 1, -1)
    ov = S.Overlay(SMALL)
    D._top_label(ov, cam, scene, np.zeros_like(shot["trans"]), 0.0)
    assert [t for t in ov.texts if "top" in t] == [
        f"cloud reaches the model top ({S.fmt_height(float(L.f.zf[-1]))})"]


def test_long_estimates_are_said_in_minutes():
    assert GL.about_words(10) == "10 s" and GL.about_words(120) == "120 s"
    assert GL.about_words(408) == "7 minutes" and GL.about_words(3600) == "60 minutes"
    copy = GL.load_copy()["ui"]
    assert "{about}" in copy["draw_3d_about"] and "{about}" in copy["orbit_about"]
    text = next(r for r in GL.load_copy()["more_looks"] if r["id"] == "storm3d")["text"]
    assert "a minute or more" not in text and "each button says about how long" in text


# ---------------------------------------------------------------- the Results screen

def test_the_servers_lists_and_costs_are_the_workers():
    assert GL.REQUEST_LOOKS == RD.ON_REQUEST_IDS == (D.LOOK,)
    assert "storm3d" in GL.FILM_LOOKS and "storm3d" in GL.DATA_FILMS
    assert GL.FILM_FPS["storm3d"] == F.DEFAULT_FPS["storm3d"]
    assert GL.STORM3D_HALF_M == {"thunderstorm": D.THUNDERSTORM_HALF_M, "hurricane": D.HURRICANE_HALF_M}
    assert GL.STORM3D_VOXELS == D.VOXEL_BUDGET and GL.STORM3D_PICTURES == D.ORBIT_FRAMES
    assert GL.STORM3D_COSTS == {"plan_s_per_mcell": D.PLAN_S_PER_MCELL, "build_s_per_mcell": D.BUILD_S_PER_MCELL,
                                "draw_s_per_mray": D.DRAW_S_PER_MRAY}
    assert GL.STORM3D_SIZES == {"still": D.STILL_SIZE, "orbit": D.ORBIT_SIZE}


def test_the_more_looks_row_offers_the_still_and_the_orbit(tmp_path):
    run = run_directory(tmp_path)
    RD.render(run)
    index = json.loads((run / "quicklook" / "index.json").read_text())
    row = {r["id"]: r for r in GL.more_looks("storm", run, index, {"state": "finished"}, films_now={})}["storm3d"]
    assert row["kind"] == "request" and row["title"] == "The storm in 3-D"
    assert row["available"] is False and row["action"] == "look" and row["film_too"] is True
    assert row["why_not"] == "Not drawn yet. Draw it to make the picture of the latest saved time."
    assert "1 km" not in row["note"] and "{" not in row["note"] and "6 km" in row["note"]
    D.draw_still(run, size=SMALL)
    index = json.loads((run / "quicklook" / "index.json").read_text())
    row = {r["id"]: r for r in GL.more_looks("storm", run, index, {"state": "running"}, films_now={})}["storm3d"]
    assert row["available"] and row["count"] == 1 and row["why_not"] is None and row["action"] == "look"
    copy = GL.load_copy()
    from cm1core.gui.copy_lint import lint_text

    said = [copy["why_not"]["draw_3d"]["text"]] + [copy["ui"][k] for k in
                                                   ("draw_3d", "draw_3d_about", "drawing_3d", "make_orbit",
                                                    "orbit_about")]
    said += [v for k, v in next(r for r in copy["more_looks"] if r["id"] == "storm3d").items()
             if k in ("title", "short", "text", "text_hurricane", "note")]
    assert [p for t in said for p in lint_text(t)] == []
