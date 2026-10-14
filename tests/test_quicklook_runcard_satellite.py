"""The run card, the visible and infrared clouds, and the wind's energy by size.

CPU only.  The frames carry CM1 r21.1's own 24 x 24 x 40 storm (``quicklook_support``).  Checked here:

1. the three looks are drawn by the worker (the two per-frame looks at every saved time, the card once, and again only
   when a receipt changes), every picture's words pass the lint and nothing is cut from a footer;
2. the particle sizes are each scheme's own rules from CM1 r21.1's source (worked by hand here), and the picture says
   which rules it used;
3. the infrared: a clear column shows the air at the lowest level, an opaque high cloud shows its top's temperature,
   Planck's law turns round exactly, and only tops warmer than -30 C are grey;
4. the spectrum: a single wave peaks at its wavelength, the energy adds up (Parseval), a plane has none, the band the
   picture marks is 6 to 7 grid spacings, and a narrow area gives a plain refusal, not a NaN;
5. the card's numbers are the receipts' (summed over segments), its words follow the run's settings (CM1's setup by
   name, its simple cooling in place of radiation), a long reason for no map stays inside the map's box, a
   hurricane's map is its inner part, and a run whose frames are gone still gets a card;
6. the Results screen offers the three looks, and the server's lists match the worker's.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pytest

from cm1import_support import namelist_tool
from quicklook_support import run_directory

from cm1core.gui import looks as GL
from cm1core.quicklook import infrared as IR
from cm1core.quicklook import particles as PS
from cm1core.quicklook import products as P
from cm1core.quicklook import render as RD
from cm1core.quicklook import runcard as RC
from cm1core.quicklook import spectra as SP
from cm1core.quicklook import style as S

NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")
NEW_LOOKS = ("infrared", "spectra")


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


@pytest.fixture()
def pictures(monkeypatch):
    seen: list[tuple[str, S.Overlay]] = []
    original = P.save

    def spy(ov, base, path):
        seen.append((Path(path).parent.name, ov))
        return original(ov, base, path)

    monkeypatch.setattr(P, "save", spy)
    return seen


def _render(run: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return RD.render(run, **kwargs)


def _index(run: Path) -> dict:
    return json.loads((run / "quicklook" / "index.json").read_text())


def receipt(*, gpu: str = "NVIDIA GeForce RTX 4090", process: float = 100.0, wall: float = 90.0, steps: float = 80.0,
            simulated: float = 1500.0, start: float = 0.0, count: int = 250, frames: int = 5,
            restarts: int = 1, extra: dict | None = None) -> dict:
    """A receipt with the fields the card reads, as cuda_cartesian_runtime and cartesian_door write them."""

    config = {"acoustic_substeps": 4, "dt_s": 6.0, "adapt_dt": False, "microphysics": "morrison-mp10",
              "dense_ice": {"category": "hail"}, "moisture_species": {"numq": 10},
              "turbulence_scale": {"cm1setup": 1, "sgsmodel": 1, "ipbl": 0}, "radiation": {"radopt": 0},
              "cumulus": {"cuparam": 0}, "lateral_boundaries": {"wbc": 2, "ebc": 2, "sbc": 2, "nbc": 2},
              "bbc": 3, "tbc": 1, "isfcflx": 0, "sfcmodel": 0, "cecd": 1, "cnstcd": 0.0024, "tsk0": 300.0,
              "u_move_m_s": 12.5, "v_move_m_s": 3.0, "damping_depth_m": 5000.0, "irdamp": 1,
              "grid": {"nx": 24, "ny": 24, "nz": 40, "dx_m": 1000.0, "dy_m": 1000.0, "dz_m": 500.0}}
    doc = {
        "version": "0.1.5", "process_wall_seconds": process, "completed_utc": "2026-10-13T00:00:00+00:00",
        "status": "executed-cartesian-integration-to-requested-duration",
        "runtime_benchmark": {
            "gpu_name": gpu, "wall_seconds": wall, "simulated_seconds": simulated,
            "completed_steps": count + int(start / 6.0), "executed_steps": count,
            "frame_rows_retained": frames, "frame_rows_dropped": 0, "restart_rows_retained": restarts,
            "restart_rows_dropped": 0,
            "model_seconds_at_start": start, "model_seconds_at_end": start + simulated,
            "step_wall_seconds": {"total": steps, "mean": steps / count, "count": count},
            "sampled_peak_device_used_bytes": 3 * 2 ** 30, "sampled_peak_registered_model_bytes": 2 ** 28,
            "driver_component_warmup": {"device_total_bytes": 24 * 2 ** 30,
                                        "component_receipt": {"arwen": {"installed_version": "2.8.9"}}},
            "adaptive_timestep": None, "config": config},
        "integration_cadence": {"dt_s": 6.0, "frame_count": frames, "history_interval_s": 300.0, "total_steps": count},
        "run_schedule": {"restart_count": restarts},
        "vertical_coordinate": {"stretch_z": 0, "nk": 40, "dz_reference_m": 500.0, "ztop_m": 20000.0,
                                "lowest_mass_level_m": 250.0},
        "initialization": {"iinit": 1, "isnd": 5, "bptpert_k": 1.0, "bhrad_m": 10000.0, "zc_m": 1400.0},
        "initial_sounding_parameters": {"mixed_layer": {"cape_j_per_kg": 2000.0}, "shear_0_6km_m_per_s": 30.0},
    }
    if extra:
        doc.update(extra)
    return doc


# ---------------------------------------------------------------- the worker draws the three looks

@NEEDS_MATPLOTLIB
def test_the_new_looks_are_drawn_at_every_saved_time_and_the_card_once(tmp_path, pictures):
    run = run_directory(tmp_path)
    summary = _render(run)
    assert summary.problems == []
    index = _index(run)
    by = {p["id"]: p for p in index["products"]}
    for look in NEW_LOOKS:
        assert [f["time_s"] for f in by[look]["frames"]] == [900.0, 1200.0, 1500.0], look
        assert by[look]["title"] == P.TITLES[look]
    card = by["run-card"]
    assert card["once"] is True and [f["file"] for f in card["frames"]] == ["run-card/card.png"]
    from PIL import Image

    for name in ("run-card/card.png", "infrared/t001500.png", "spectra/t001500.png"):
        with Image.open(run / "quicklook" / name) as im:
            assert im.size == (1920, 1080), name
    # every picture's words pass the honesty lint, and no footer clause was cut
    for name, ov in pictures:
        assert ov.check() == [] and ov.footer_cut == [], (name, ov.footer_cut)
    words = {name: "\n".join(ov.texts) for name, ov in pictures}
    assert "simulated, not a satellite retrieval" in words["infrared"]
    assert "particle sizes: Morrison's own" in words["infrared"]
    assert "visible: the clouds-from-above look and its fixed particle sizes" in words["infrared"]
    assert "6 to 7 grid spacings" in words["spectra"] and "dashed −5/3, dotted −3: for reference, not fits" in words[
        "spectra"]
    assert "23,040 grid cells (24 × 24 × 40)" in words["run-card"]
    # a run with no receipt yet: the card says where its words came from and what is still to come
    assert "the newest frame's record (no receipt yet)" in words["run-card"]
    assert "arrive in the receipt when this part of the run ends" in words["run-card"]
    facts = by["infrared"]["frames"][-1]["facts"]
    assert facts["particle_sizes"] == "Morrison" and facts["visible_over_infrared"] == 2.13
    assert facts["tb_min_c"] < -20.0 and facts["model_top_m"] == 20000.0
    spectra = by["spectra"]["frames"][-1]["facts"]
    assert spectra["heights_m"][0] == 250.0 and spectra["resolved_band_m"] == [6000.0, 7000.0]


@NEEDS_MATPLOTLIB
def test_the_card_reads_the_receipts_and_is_drawn_again_only_when_one_changes(tmp_path, pictures):
    run = run_directory(tmp_path)
    (run / "receipt.json").write_text(json.dumps(receipt()), encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"name": "my storm", "case": "A test case"}), encoding="utf-8")
    _render(run, only=("reflectivity", "run-card"))
    card = next(p for p in _index(run)["products"] if p["id"] == "run-card")["frames"][0]
    facts = card["facts"]
    assert facts["gpus"] == ["RTX 4090"] and facts["process_s"] == 100.0 and facts["storm_s"] == 1500.0
    assert facts["time_split_s"] == pytest.approx([80.0, 10.0, 10.0])
    assert facts["speed"] == pytest.approx(15.0) and facts["arwen"] == "2.8.9"
    # the map is drawn from one of the saved times (the strongest echo the frames' statistics recorded)
    assert facts["map"] and card["time_s"] in (900.0, 1200.0, 1500.0)
    texts = "\n".join(next(ov for name, ov in pictures if name == "run-card").texts)
    assert "ground-relative wind change" not in texts and "large eddies on the grid" not in texts
    for words in ("my storm", "A test case", "GM1 0.1.5", "on one RTX 4090, measured", "15× faster than real time",
                  "CM1's large-eddy setup (cm1setup 1)", "1.5-order TKE mixing in all directions",
                  "23,040 grid cells (24 × 24 × 40)", "no surface heating, no radiation",
                  "surface drag, drag coefficient 0.0024", "sides: open (waves leave)", "a 1 K warm bubble",
                  "CAPE 2,000 J/kg (mixed layer)", "wind change 0–6 km (bulk shear): 30 m/s", "250 steps",
                  "3.0 of 24.0 GiB", "run by ArWen 2.8.9 (gpuwm)", "numbers from 1 receipt, the namelist and run.json",
                  "from the run's receipt and namelist"):
        assert words in texts, words
    mtime = (run / "quicklook" / "run-card" / "card.png").stat().st_mtime_ns
    _render(run, only=("run-card",))
    assert (run / "quicklook" / "run-card" / "card.png").stat().st_mtime_ns == mtime
    # a new receipt (the run went on) draws it again
    (run / "receipt.json").write_text(json.dumps(receipt(process=200.0)), encoding="utf-8")
    _render(run, only=("run-card",))
    facts = next(p for p in _index(run)["products"] if p["id"] == "run-card")["frames"][0]["facts"]
    assert facts["process_s"] == 200.0


@NEEDS_MATPLOTLIB
def test_netcdf_files_get_no_card_and_say_why(tmp_path):
    """CM1's own output opened for pictures (or a GM1 run's cm1out files) has no receipt: no card says a GPU ran it;
    the index gives the reason instead."""

    import shutil

    source = tmp_path / "cm1run"
    source.mkdir()
    shutil.copy(Path(__file__).resolve().parent / "data" / "cm1-ref-24x24" / "cm1out_000001.nc",
                source / "cm1out_000001.nc")
    out = tmp_path / "pictures"
    RD.render(source, out=out, only=("run-card",))
    card = {p["id"]: p for p in json.loads((out / "index.json").read_text())["products"]}["run-card"]
    assert card["frames"] == [] and card["not_drawn"][0]["reason"] == RC.NETCDF_WHY
    assert not (out / "run-card" / "card.png").exists()
    # a second pass leaves it as it is
    assert RC.draw_once(RD.Renderer(source, out=out, only=("run-card",)), []) is False


def test_the_card_sums_the_segments_and_says_the_settings_in_plain_words(tmp_path):
    root = tmp_path / "run"
    first, second = root / "segments" / "001", root / "segments" / "002"
    for seg, doc in ((first, receipt(process=100.0, wall=90.0, steps=80.0, count=250)),
                     (second, receipt(gpu="NVIDIA GeForce RTX 3080", process=50.0, wall=45.0, steps=40.0, count=125,
                                      simulated=750.0, start=1500.0, frames=3, restarts=0))):
        seg.mkdir(parents=True)
        (seg / "receipt.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "input_sounding").write_text("1000 300 14\n", encoding="utf-8")
    (root / "input_sounding.original").write_text("a balloon\n", encoding="utf-8")
    facts = RC.gather(root, [first, second])
    assert facts["receipts"] == 2 and facts["gpus"] == ["RTX 4090", "RTX 3080"]
    assert (facts["process_s"], facts["integrate_s"], facts["stepping_s"]) == (150.0, 135.0, 120.0)
    # the steps each segment ran, not the clock's count since the start (the second segment's completed_steps is 375)
    assert facts["storm_s"] == 2250.0 and facts["steps"] == 375 and facts["frames"] == 8 and facts["restarts"] == 1
    assert facts["step_mean_s"] == pytest.approx(120.0 / 375.0)
    assert facts["storm_end_s"] == 2250.0
    facts["init"].update({"isnd": 7, "iinit": 12})
    facts["namelist"] = {"hadvordrs": 5, "advwenos": 0, "icor": 1, "fcor": 5.0e-5}
    tiles = dict(RC.physics_tiles(facts, "Morrison with hail", "not yet checked against CM1"))
    assert "5th-order advection" in tiles["The equations"]
    assert "4 sound steps per 6 s step" in tiles["The equations"]
    assert "your own sounding (input_sounding)" in tiles["The start"]
    assert "a push of rising air (updraft nudging)" in tiles["The start"]
    assert "Earth's spin on: f = 5 × 10⁻⁵ s⁻¹" in tiles["The equations"]
    assert tiles["Rain and hail"][-1] == "not yet checked against CM1"
    # rounded to the last unit shown: 5,614 s (1 h 33.6 min) is 1 h 34 min, not cut to 1 h 33 min
    assert [RC.duration(s) for s in (37, 300, 2580, 3599, 3600, 5614, 7170, 7260, 86390, 291600, 293500)] == [
        "37 s", "5 min", "43 min", "59 min 59 s", "1 h", "1 h 34 min", "2 h", "2 h 01 min", "1 d", "3 d 9 h",
        "3 d 10 h"]
    # an odd time step is said to three figures
    facts["dt_s"] = 0.390625
    assert "4 sound steps per 0.391 s step" in dict(RC.physics_tiles(facts, "Morrison", "x"))["The equations"]


def test_the_card_names_cm1s_setup_and_its_stand_in_for_radiation():
    """A 4 km hurricane runs CM1's large-eddy setup with no eddy on its grid, and CM1's simple cooling (rterm) in
    place of a radiation scheme: the card names both as what they are."""

    doc = receipt(extra={})
    config = doc["runtime_benchmark"]["config"]
    config.update({"rterm": 1, "isfcflx": 1, "sfcmodel": 1, "oceanmodel": 1, "tsk0": 301.15,
                   "lateral_boundaries": {"wbc": 1, "ebc": 1, "sbc": 1, "nbc": 1}})
    with_rterm = dict(RC.physics_tiles(_gathered(doc), "Morrison", "x"))
    edges = with_rterm["Edges and ground"]
    assert "heat and moisture from a 28.0 °C sea" in edges
    assert RC.RTERM_WORDS in edges and "no radiation" not in [line for line in edges if line != RC.RTERM_WORDS]
    assert "at most 2 K a day" in RC.RTERM_WORDS and "no radiation scheme" in RC.RTERM_WORDS
    assert with_rterm["Turbulence"][:2] == ["CM1's large-eddy setup (cm1setup 1)",
                                             "1.5-order TKE mixing in all directions"]
    config["rterm"] = 0
    assert "heat and moisture from a 28.0 °C sea, no radiation" in dict(
        RC.physics_tiles(_gathered(doc), "Morrison", "x"))["Edges and ground"]
    # the mesoscale setup does not read the subgrid model, so the card does not name one
    config["turbulence_scale"] = {"cm1setup": 2, "sgsmodel": 1, "ipbl": 1}
    assert dict(RC.physics_tiles(_gathered(doc), "Morrison", "x"))["Turbulence"] == [
        "CM1's mesoscale setup (cm1setup 2)", "YSU boundary-layer scheme", "no cumulus scheme"]


def _gathered(doc: dict) -> dict:
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "receipt.json").write_text(json.dumps(doc), encoding="utf-8")
        return RC.gather(Path(folder), [Path(folder)])


@NEEDS_MATPLOTLIB
def test_a_run_whose_frames_are_gone_still_gets_its_card(tmp_path, monkeypatch):
    """The receipt alone: the levels come from CM1's own level builder, and the map says why it is missing."""

    from cm1core.quicklook import facts as FA

    doc = receipt()
    facts = RC.gather(tmp_path, [tmp_path], frame_config={})
    assert facts["receipts"] == 0
    (tmp_path / "receipt.json").write_text(json.dumps(doc), encoding="utf-8")
    facts = RC.gather(tmp_path, [tmp_path])
    zf = RC.levels_without_frame(facts)
    assert zf is not None and len(zf) == 41 and zf[1] == 500.0 and zf[-1] == 20000.0
    run_facts = FA.RunFacts(name="gone", version="0.1.5", version_source="receipt", scheme="Morrison with hail",
                            component="morrison-mp10", checked="not yet checked against CM1", checked_source="x",
                            dx_m=1000.0, dy_m=1000.0)
    seen = []
    original = P.save
    monkeypatch.setattr(P, "save", lambda ov, base, path: (seen.append(ov), original(ov, base, path))[1])
    shown = RC.draw(facts, run_facts, tmp_path / "card.png", qf=None, zf=zf)
    assert shown["map"] is None and shown["levels"] == 40 and shown["lowest_level_m"] == 250.0
    assert "no saved frame to draw the storm from" in "\n".join(seen[0].texts)
    assert seen[0].check() == [] and seen[0].footer_cut == []


@NEEDS_MATPLOTLIB
def test_a_long_no_map_reason_stays_inside_the_maps_box(tmp_path, monkeypatch):
    """A WSM6 run without reflectivity output gives a long reason; it is wrapped and centred inside the box."""

    from cm1core.quicklook import facts as FA

    (tmp_path / "receipt.json").write_text(json.dumps(receipt()), encoding="utf-8")
    facts = RC.gather(tmp_path, [tmp_path])
    run_facts = FA.RunFacts(name="wsm6", version="0.1.5", version_source="receipt", scheme="WSM6",
                            component="wsm6", checked="not yet checked against CM1", checked_source="x",
                            dx_m=1000.0, dy_m=1000.0)
    boxes, drawn = [], []
    outline, text = P.outline, S.Overlay.text
    monkeypatch.setattr(P, "outline", lambda cv, box, *a, **k: (boxes.append(box), outline(cv, box, *a, **k))[1])
    monkeypatch.setattr(S.Overlay, "text", lambda self, xy, words, *a, **k: drawn.append(
        (words, text(self, xy, words, *a, **k))) or drawn[-1][1])
    reason = ("no map: this run did not ask WSM6 for its reflectivity (output_dbz = 0), so no reflectivity picture "
              "is drawn.")
    shown = RC.draw(facts, run_facts, tmp_path / "card.png", qf=None, zf=RC.levels_without_frame(facts),
                    map_reason=reason)
    assert shown["map"] is None
    box = boxes[0]
    # the texts centred on the box's middle, inside it from top to bottom: the reason's lines and nothing else
    lines = [(words, bbox) for words, bbox in drawn if box.y < bbox[1] and bbox[3] < box.y + box.h
             and abs((bbox[0] + bbox[2]) / 2 - (box.x + box.w / 2)) <= 2]
    assert len(lines) >= 2 and " ".join(words for words, _b in lines) == reason
    for words, (x0, _y0, x1, _y1) in lines:
        assert box.x <= x0 and x1 <= box.x + box.w, (words, x0, x1, box.x, box.w)


@NEEDS_MATPLOTLIB
def test_a_hurricanes_map_shows_the_inner_part_and_says_so(tmp_path, monkeypatch):
    from cm1core.quicklook import facts as FA
    from cm1core.quicklook.frame import QuickFrame

    box = RC.map_box(1_024_000.0, 1_024_000.0, 360, 222, 560, 484, inner=P.HURRICANE_INNER)
    assert (box.mx0, box.my0, box.span) == (262_000.0, 262_000.0, pytest.approx(500_000.0, rel=0.01))
    run = run_directory(tmp_path, times=(1500.0,))
    qf = QuickFrame.open(run / "frames" / "frame-000000")
    seen = []
    original = P.save
    monkeypatch.setattr(P, "save", lambda ov, base, path: (seen.append(ov), original(ov, base, path))[1])
    try:
        facts = RC.gather(run, [run], frame_config=qf.config)
        run_facts = FA.RunFacts(name="swirl", version="0.1.5", version_source="receipt", scheme="Morrison with hail",
                                component="morrison-mp10", checked="not yet checked against CM1", checked_source="x",
                                dx_m=qf.dx, dy_m=qf.dy, storm_kind="hurricane")
        shown = RC.draw(facts, run_facts, tmp_path / "card.png", qf=qf, inner=12_000.0)
    finally:
        qf.close()
    assert shown["map"] and shown["map_inner_km"] == 12.0
    assert "24 × 24 km, seen from above; the map shows its inner 12 km" in seen[0].texts


def test_the_map_is_the_strongest_echo_or_a_hurricanes_newest_time():
    rows = {"900.0": {"stats": {"dbz_max": 40.0}}, "1200.0": {"stats": {"dbz_max": 62.0}},
            "1500.0": {"stats": {"dbz_max": 55.0}}}
    assert RC.map_frame(rows, hurricane=False) == "1200.0"
    assert RC.map_frame(rows, hurricane=True) == "1500.0"
    assert RC.map_frame({"0.0": {"stats": {}}, "300.0": {}}, hurricane=False) == "300.0"
    assert RC.map_frame({}, hurricane=False) is None


# ---------------------------------------------------------------- particle sizes, by hand from CM1's source

def test_morrison_sizes_are_its_radiation_radii_from_the_saved_numbers():
    rho = np.array([1.0])
    q = {"qi": np.array([1.0e-4]), "qs": np.array([1.0e-3]), "qr": np.array([1.0e-3]), "qg": np.array([1.0e-3]),
         "qc": np.array([1.0e-3])}
    n = {"nci": np.array([1.0e5]), "ncs": np.array([1.0e4]), "ncr": np.array([1.0e3]), "ncg": np.array([1.0e3])}
    r = PS.morrison(q, n, rho, ndcnst=250.0, ice="hail")
    # morrison.F:5001-5178: LAMI = (CONS12 NI / QI)^(1/3), CONS12 = Gamma(4) pi rhoi / 6; EFFI = 3 / (2 LAMI)
    lam_i = (math.gamma(4.0) * math.pi * 500.0 / 6.0 * 1.0e5 / 1.0e-4) ** (1.0 / 3.0)
    assert r["qi"][0] == pytest.approx(1.5 / lam_i, rel=1e-6)
    lam_s = (math.gamma(4.0) * math.pi * 100.0 / 6.0 * 1.0e4 / 1.0e-3) ** (1.0 / 3.0)
    assert r["qs"][0] == pytest.approx(1.5 / lam_s, rel=1e-6)
    lam_r = (math.pi * 997.0 * 1.0e3 / 1.0e-3) ** (1.0 / 3.0)               # rain: pi rhow NR / QR
    assert r["qr"][0] == pytest.approx(1.5 / lam_r, rel=1e-6)
    lam_g = (math.gamma(4.0) * math.pi * 900.0 / 6.0 * 1.0e3 / 1.0e-3) ** (1.0 / 3.0)   # hail: rhog 900
    assert r["qg"][0] == pytest.approx(1.5 / max(lam_g, 1.0 / 2000.0e-6), rel=1e-6)
    # cloud droplets: 250 per cm3, the shape PGAM, EFFC = Gamma(p+4) / Gamma(p+3) / LAMC / 2
    p = min(max(1.0 / (0.0005714 * 250.0 + 0.2714) ** 2 - 1.0, 2.0), 10.0)
    lam_c = (math.pi / 6.0 * 997.0 * 250.0e6 * math.gamma(p + 4.0) / (1.0e-3 * math.gamma(p + 1.0))) ** (1.0 / 3.0)
    assert r["qc"][0] == pytest.approx((p + 3.0) / lam_c / 2.0, rel=1e-6)
    assert 5.0e-6 < r["qc"][0] < 20.0e-6
    # the slope limits: too few crystals for their mass gives the largest size Morrison allows
    few = PS.morrison({"qi": np.array([1.0e-3])}, {"nci": np.array([1.0])}, rho, ndcnst=250.0)
    assert few["qi"][0] == pytest.approx(1.5 * (2.0 * 125.0e-6 + 100.0e-6), rel=1e-6)


def test_wsm6_and_thompson_sizes_follow_their_own_rules():
    rho = np.array([1.0, 1.0])
    t = np.array([263.15, 233.15])
    w = PS.wsm6({"qc": np.array([1.0e-3, 1.0e-3]), "qs": np.array([1.0e-3, 1.0e-3]), "qr": np.array([1.0e-3, 1.0e-3])},
                rho, t)
    lam_c = (math.pi * 1000.0 / 6.0 * 3.0e8 / 1.0e-3) ** (1.0 / 3.0)
    assert w["qc"][0] == pytest.approx(1.5 / lam_c, rel=1e-6)
    # snow's intercept grows as it gets colder (alpha = 0.12): more, smaller flakes at -40 C than at -10 C
    assert w["qs"][1] < w["qs"][0]
    assert w["qr"][0] == pytest.approx(1.5 / (math.pi * 1000.0 * 8.0e6 / 1.0e-3) ** 0.25, rel=1e-6)
    th = PS.thompson({"qc": np.array([1.0e-3, 1.0e-3]), "qs": np.array([1.0e-4, 1.0e-4])}, {}, rho, t)
    assert 2.51e-6 <= th["qc"][0] <= 50.0e-6 and 10.01e-6 <= th["qs"][1] <= 999.0e-6
    # calc_effectRad's cloud water with the scheme's 100 droplets per cm3 (inu_c = 12, g_ratio 2730)
    lam = (100.0e6 * math.pi * 1000.0 / 6.0 * 2730.0 / 1.0e-3) ** (1.0 / 3.0)
    assert th["qc"][0] == pytest.approx(0.5 * 15.0 / lam, rel=1e-6)


@NEEDS_MATPLOTLIB
def test_the_picture_says_which_sizes_it_used(tmp_path):
    from cm1core.quicklook.frame import QuickFrame
    from cm1core.quicklook import satellite as SA

    run = run_directory(tmp_path, times=(1500.0,))
    qf = QuickFrame.open(run / "frames" / "frame-000000")
    try:
        species = SA.species_present(qf)
        rho = SA.density(qf)
        t = IR.temperature(qf)
        beta, sizes = PS.optics(qf, species, rho, t)
        assert sizes.scheme == "Morrison" and "Morrison's own" in sizes.words
        assert sizes.notes == ["cloud droplets 250 per cm³ (the run's namelist)"]
        assert beta.shape == (qf.nz, qf.ny, qf.nx) and float(beta.max()) > 0.0
        # a scheme with no rules here: the fixed sizes, said as such

        class Kessler:
            config = {"microphysics": "kessler"}

            def __getattr__(self, name):
                return getattr(qf, name)

        _beta, fixed = PS.optics(Kessler(), species, rho, t)
        assert fixed.scheme == "fixed" and fixed.words.startswith("particle sizes fixed (")
    finally:
        qf.close()


# ---------------------------------------------------------------- the infrared

def test_planck_turns_round_and_a_clear_column_shows_the_lowest_air():
    t = np.array([180.0, 220.0, 288.0, 310.0])
    assert np.allclose(IR.brightness(IR.planck(t)), t, rtol=1e-12)
    nz, zf = 10, np.arange(11) * 1000.0
    zh = 0.5 * (zf[1:] + zf[:-1])
    temps = (290.0 - 6.5e-3 * zh)[:, None, None] * np.ones((nz, 2, 2))
    clear = IR.infrared(np.zeros((nz, 2, 2), np.float32), temps, zf, zh)
    assert np.allclose(clear["tb"], 290.0 - 6.5e-3 * 500.0, atol=1e-3)
    beta = np.zeros((nz, 2, 2), np.float32)
    beta[8, 0, 0] = 0.1                                         # an opaque cloud at 8-9 km over one column
    beta[8, 1, 1] = 0.0002                                      # a thin one over another
    cloudy = IR.infrared(beta, temps, zf, zh)
    assert cloudy["tb"][0, 0] == pytest.approx(290.0 - 6.5e-3 * 8500.0, abs=0.01)
    assert cloudy["emit_m"][0, 0] == pytest.approx(8500.0, abs=1.0)
    assert temps[8, 0, 0] < cloudy["tb"][1, 1] < clear["tb"][1, 1]
    assert IR.ir_rgb(np.array([IR.TB_LOW - 30.0]))[0] == pytest.approx(IR.ir_rgb(np.array([IR.TB_LOW]))[0])


def test_only_tops_warmer_than_minus_30_are_grey():
    """The words say grey is warmer than -30 C: so no colder top is grey, and none comes near a warm top's grey (the
    overshooting tops, -70 to -80 C, once had a grey ramp that matched +17 C)."""

    warm = IR.ir_rgb(np.arange(IR.TB_COLOURED + 0.5, IR.TB_HIGH, 0.25))
    cold = IR.ir_rgb(np.arange(IR.TB_LOW, IR.TB_COLOURED, 0.25))
    assert float((warm.max(axis=1) - warm.min(axis=1)).max()) < 0.07                       # warm: grey
    assert float((cold.max(axis=1) - cold.min(axis=1)).min()) > 0.2                        # cold: coloured
    nearest = np.sqrt(((cold[:, None, :] - warm[None, :, :]) ** 2).sum(axis=-1)).min()
    assert float(nearest) > 0.15
    # each 10-degree step starts a new colour: the two sides of every step differ clearly
    for edge in (-30.0, -40.0, -50.0, -60.0, -70.0, -80.0):
        a, b = IR.ir_rgb(np.array([edge + 0.005, edge - 0.02]))
        assert float(np.abs(a - b).max()) > 0.2, edge


def test_the_infrared_gif_keeps_the_coldest_tops_colours(tmp_path):
    """A few pixels of a -76 C top in a picture of many colours: the GIF's median cut alone gives them the colour of
    a large area; the infrared look's GIF keeps the cold colours, and every other look's GIF is made as before."""

    from PIL import Image

    from cm1core.quicklook import film as F

    y, x = np.mgrid[0:1080, 0:1920]
    base = np.stack([20 + 210 * x / 1919, 20 + 210 * y / 1079, 20 + 105 * (x / 1919 + y / 1079)], -1).astype(np.uint8)
    cold = tuple(int(round(255 * c)) for c in IR.ir_rgb(np.array([-76.0]))[0])
    base[500:510, 900:910] = cold                                                    # 5 x 5 pixels in the GIF
    picture = Image.fromarray(base)

    def spot(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))[252, 452].astype(int)

    plain = F.write_gif([picture], tmp_path / "plain.gif", 4.0)
    kept = F.write_gif([picture], tmp_path / "kept.gif", 4.0, keep=F.gif_keep("infrared"))
    assert np.abs(spot(kept) - np.array(cold)).max() <= 6
    assert np.abs(spot(plain) - np.array(cold)).max() > 30
    assert F.gif_keep("reflectivity") == [] and len(F.gif_keep("infrared")) == 60
    assert all(max(c) - min(c) > 40 for c in F.gif_keep("infrared"))                # none of them grey
    old = picture.convert("RGB").resize(F.GIF_SIZE, Image.LANCZOS).quantize(colors=255, method=Image.MEDIANCUT,
                                                                            dither=Image.NONE)
    new = F._gif_picture(picture, ())
    assert new.tobytes() == old.tobytes() and new.getpalette() == old.getpalette()


@NEEDS_MATPLOTLIB
def test_with_no_cloud_yet_no_top_is_ringed(tmp_path, pictures, monkeypatch):
    """Clear air shows the air near the ground; the coldest spot there is no cloud top, and the picture says so."""

    run = run_directory(tmp_path, times=(900.0,))
    clear = lambda qf, species, rho, t: (np.zeros(rho.shape, np.float32),                       # noqa: E731
                                         PS.Sizes(scheme="Morrison", words="particle sizes: Morrison's own",
                                                  source="test"))
    monkeypatch.setattr(PS, "optics", clear)
    _render(run, only=("infrared",))
    facts = next(p for p in _index(run)["products"] if p["id"] == "infrared")["frames"][0]["facts"]
    assert facts["tb_min_c"] is None and "tb_min_xy_m" not in facts and facts["cold_tops_area_km2"] == 0.0
    texts = "\n".join(next(ov for name, ov in pictures if name == "infrared").texts)
    assert "no cloud top yet" in texts and "gold ring" not in texts


# ---------------------------------------------------------------- the spectrum

def test_a_single_wave_peaks_at_its_wavelength_and_the_energy_adds_up():
    n, dx = 128, 250.0
    x = (np.arange(n) + 0.5) * dx
    wavelength = n * dx / 8.0
    u = np.sin(2.0 * np.pi * x / wavelength)[None, :] * np.ones((n, 1))
    v = np.zeros((n, n))
    k, e = SP.spectrum(u, v, dx, dx)
    assert 2.0 * np.pi / k[int(np.argmax(e))] == pytest.approx(wavelength)
    dk = k[1] - k[0]
    window = np.outer(SP.tukey(n), SP.tukey(n))
    expected = 0.5 * np.mean((SP.detrend(u) * window) ** 2) / np.mean(window ** 2)
    assert float(np.sum(e) * dk) == pytest.approx(expected, rel=0.02)
    # a plane (a uniform wind and a steady slope across the area) has no energy left
    y = (np.arange(n) + 0.5)[:, None] * dx
    plane = 12.5 + 1.0e-4 * x[None, :] - 3.0e-4 * y
    _k, flat = SP.spectrum(plane, plane, dx, dx)
    assert float(np.sum(flat)) < 1.0e-12
    # the shortest wave drawn is two grid spacings
    assert k[-1] <= np.pi / dx + 1e-12


def test_the_grids_limit_is_6_to_7_grid_spacings():
    """The marks sit at wavelengths of 6 and 7 dx, and the picture says an eddy D across sits near a wavelength of
    2 D (a feature marked at a wavelength equal to its width would sit twice too far out in wavenumber)."""

    assert SP.RESOLVED_DX == (6.0, 7.0)
    assert SP.decade(1.0e-4) == "10⁻⁴" and SP.decade(1000.0) == "10³" and SP.decade(3.0) == ""


def test_a_narrow_side_is_not_tapered_to_nothing():
    """A side of a few cells would be all edge: it is left untapered, so the energy stays finite."""

    assert np.array_equal(SP.tukey(2), np.ones(2)) and np.array_equal(SP.tukey(SP.MIN_CELLS - 1), np.ones(7))
    assert SP.tukey(SP.MIN_CELLS)[0] == 0.0 and SP.tukey(SP.MIN_CELLS).max() == 1.0
    rng = np.random.default_rng(3)
    u, v = rng.normal(size=(2, 40)), rng.normal(size=(2, 40))
    _k, e = SP.spectrum(u, v, 1000.0, 1000.0)
    assert np.all(np.isfinite(e)) and float(e.sum()) > 0.0


@NEEDS_MATPLOTLIB
def test_too_narrow_an_area_says_so_and_a_small_one_has_no_guides(tmp_path, pictures, monkeypatch):
    from cm1core.quicklook.frame import QuickFrame

    run = run_directory(tmp_path, times=(1500.0,))
    qf = QuickFrame.open(run / "frames" / "frame-000000")
    try:
        class Narrow:                       # the same storm on a strip 6 columns wide
            nx, ny = 24, 6

            def __getattr__(self, name):
                return getattr(qf, name)

        looks = type("L", (), {"f": Narrow()})()
        with pytest.raises(P.NotDrawn, match="too narrow for a spectrum"):
            SP.product_spectra(looks, tmp_path / "narrow.png")
    finally:
        qf.close()
    # an area narrower than the guides need (here: the need set above this area's 24 columns) leaves them and their
    # words off
    monkeypatch.setattr(SP, "GUIDES_MIN_CELLS", 25)
    _render(run, only=("spectra",))
    facts = next(p for p in _index(run)["products"] if p["id"] == "spectra")["frames"][0]["facts"]
    texts = "\n".join(next(ov for name, ov in pictures if name == "spectra").texts)
    assert facts["reference_slopes"] is False and "dotted" not in texts and "for reference" not in texts


def test_a_wind_with_no_eddies_yet_has_no_spectrum(tmp_path):
    """A storm's first saved time: the wind at every height is the base state's, the same everywhere, and what the
    plane's removal leaves is the float's last digit (about 1e-25 m2/s2 of energy). That is refused in words, not
    drawn as a spectrum of noise; a weak eddy field of 1 cm/s is well above the floor."""

    from cm1core.quicklook.frame import QuickFrame

    rng = np.random.default_rng(3)
    n, dx = 24, 1000.0
    y, x = np.mgrid[0:n, 0:n]
    plane = (12.0 + 0.004 * x - 0.003 * y).astype(np.float32)
    calm = (plane * (1.0 + rng.normal(0.0, 1e-7, plane.shape))).astype(np.float32)
    dk = 2.0 * np.pi / (n * dx)
    _k, e = SP.spectrum(calm, calm, dx, dx)
    assert float(e.sum()) * dk < 1e-9 < SP.EDDY_FLOOR_M2_S2
    ripple = calm + np.float32(0.01) * np.sin(2.0 * np.pi * x / 8.0).astype(np.float32)
    _k, e = SP.spectrum(ripple, ripple, dx, dx)
    assert float(e.sum()) * dk > 10.0 * SP.EDDY_FLOOR_M2_S2

    run = run_directory(tmp_path, times=(1500.0,))
    qf = QuickFrame.open(run / "frames" / "frame-000000")
    try:
        class Calm:                         # the same frame, its wind a plane at every height
            def __getattr__(self, name):
                return getattr(qf, name)

            def uv_level(self, _k):
                return calm[: qf.ny, : qf.nx], calm[: qf.ny, : qf.nx]

        looks = type("L", (), {"f": Calm()})()
        with pytest.raises(P.NotDrawn, match="^No eddies yet: the wind is the same everywhere"):
            SP.product_spectra(looks, run / "calm.png")
        assert not (run / "calm.png").exists()
    finally:
        qf.close()


# ---------------------------------------------------------------- the Results screen

def test_the_servers_lists_match_the_workers():
    from cm1core.quicklook import film as F

    assert GL.FILM_LOOKS == F.film_looks()
    assert GL.ONCE_LOOKS == RD.ONCE_IDS and "run-card" in GL.ONCE_LOOKS
    assert {"infrared", "spectra", "run-card"} <= set(GL.NEW_LOOKS)
    copy = {row["id"]: row for row in GL.load_copy()["more_looks"]}
    assert copy["run-card"]["kind"] == "once" and copy["infrared"]["kind"] == copy["spectra"]["kind"] == "frames"
    for key in ("text", "text_hurricane"):
        words = copy["infrared"][key]
        assert "not a satellite retrieval" in words and "Both pictures size" not in words
        # the infrared sizes particles by the run's scheme; the visible keeps the fixed sizes of Clouds seen from above
        assert "the visible one keeps the fixed sizes of Clouds seen from above" in words
        assert "Grey is warmer than −30 °C" in words and "purple" in words
    assert "dashed and dotted lines" in copy["spectra"]["text"]
    assert "drawn when the first saved time lands, and again when each part of the run ends" in copy["run-card"][
        "text"]
    from cm1core.gui.runview import MORE_LOOKS

    assert list(MORE_LOOKS) == [row["id"] for row in GL.load_copy()["more_looks"]]


@NEEDS_MATPLOTLIB
def test_a_run_drawn_before_these_looks_offers_to_draw_them(tmp_path):
    run = run_directory(tmp_path)
    _render(run, only=("reflectivity", "clouds-from-above", "environment"))
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    assert rows["infrared"]["action"] == "all" and rows["spectra"]["action"] == "all"
    assert rows["run-card"]["action"] == "look"
    assert rows["infrared"]["why_not"].startswith("This run's pictures were drawn before")
    _render(run)
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    for look in ("infrared", "spectra", "run-card"):
        assert rows[look]["available"] and rows[look]["why_not"] is None, look
        assert "{" not in rows[look]["text"] and "{" not in rows[look]["note"], look


@NEEDS_MATPLOTLIB
def test_the_command_line_draws_the_card_again(tmp_path):
    import subprocess
    import sys

    run = run_directory(tmp_path, times=(1500.0,))
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    drawn = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(run), "--look", "run-card", "--json"],
                           capture_output=True, text=True, env=env)
    assert drawn.returncode == 0, drawn.stderr
    record = json.loads(drawn.stdout)
    assert record["look"] == "run-card" and record["file"] == "run-card/card.png" and Path(record["path"]).is_file()
    assert record["worker"]["nice"] == 19 and record["worker"]["cuda_visible_devices"] == ""
