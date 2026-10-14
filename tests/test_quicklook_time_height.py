"""The storm over time: the updraft and the spin at every height, one column per saved time (quicklook.timeheight).

CPU only.  The frames carry CM1 r21.1's own 24 x 24 x 40 storm (``quicklook_support``), as the other quick-look
tests do.  Here:

1. both looks are drawn once per run, from every saved time, as one 1920 x 1080 picture each, with the footer and
   the honesty lint of every quick-look, and drawn again when a saved time lands (reading only that frame);
2. the storm is followed on the ground; a line or a cluster of storms falls back to the whole domain and says so;
   a hurricane follows its lowest pressure and has no spin look;
3. pulse tracks are traced only from saves 60 s apart or closer, and the picture says when they are left off;
4. the spin's colours follow the grid: the spin look's rule from the greater of the grid spacing and the lowest
   level's height; the helicity line's top follows the same range and is raised to hold a run's peak;
5. the other looks are byte-identical with and without these two, and the GUI's lists hold the worker's;
6. a damaged store of numbers is read again from the frames, and an error here never stops the other looks;
7. what the pictures cannot see is said: the threshold of a storm, a storm held at the domain's edge, a cloud top
   at the model's top, frames without spin.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pytest

import importlib.util

NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")

from cm1import_support import namelist_tool  # noqa: E402
from quicklook_support import NAMELIST_24, publish_storm, run_directory  # noqa: E402

from cm1core.gui import looks as GL  # noqa: E402
from cm1core.quicklook import products as P  # noqa: E402
from cm1core.quicklook import render as RD  # noqa: E402
from cm1core.quicklook import style as S  # noqa: E402
from cm1core.quicklook import timeheight as TH  # noqa: E402

LOOKS = (TH.UPDRAFT, TH.SPIN)


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


@pytest.fixture
def pictures(monkeypatch):
    """Every Overlay a look saves, with its words."""

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
    return json.loads((run / "quicklook" / "index.json").read_text(encoding="utf-8"))


def _png_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


# ---------------------------------------------------------------- the two looks in a pass

@NEEDS_MATPLOTLIB
def test_both_looks_are_one_picture_of_every_saved_time(tmp_path, pictures):
    run = run_directory(tmp_path)
    summary = _render(run)
    assert summary.problems == []
    index = _index(run)
    once = {p["id"]: p for p in index["products"] if p.get("once")}
    for look in LOOKS:
        assert once[look]["title"] == TH.TITLES[look]
        [frame] = once[look]["frames"]
        assert frame["time_s"] == 1500.0 and frame["clock"] == S.fmt_clock(1500.0)
        assert frame["file"] == f"{look}/still.png"
        assert _png_size(run / "quicklook" / frame["file"]) == (1920, 1080)
        assert _png_size(run / "quicklook" / frame["thumb"]) == (480, 270)
        assert frame["facts"]["saved_times"] == 3 and frame["facts"]["following"] == "storm"
    updraft = once[TH.UPDRAFT]["frames"][0]["facts"]
    spin = once[TH.SPIN]["frames"][0]["facts"]
    # the numbers are the frames' own: the strongest w of the circle is the domain's (one storm in 24 x 24 km)
    state = np.load(Path(__file__).resolve().parent / "data" / "cm1-ref-24x24" / "stats-run-t1500.npz")
    assert updraft["w_max"] == pytest.approx(float(state["w"].max()), rel=1e-6)
    assert updraft["range_m_s"] == [0.0, TH.W_TOP] and updraft["pulse_tracks"] is None
    assert spin["range_per_s"] == TH.spin_range_for(1000.0, 250.0) == 0.01
    assert spin["colours_per_s"] == pytest.approx([0.001, 0.05])
    assert spin["radius_m"] == TH.SPIN_RADIUS_M and 1000.0 <= spin["zeta_max_height_m"] <= 6000.0
    assert spin["zeta_max_1_6km"] <= spin["zeta_max_all"] and spin["range_set_by"] == "dx"
    assert spin["helicity_grid_top"] == TH.UH_TOP and spin["helicity_top"] >= spin["helicity_max"]
    by = {name: ov for name, ov in pictures if name in LOOKS}
    assert set(by) == set(LOOKS)
    for name, ov in by.items():
        assert ov.check() == [] and ov.footer_cut == [], name
        text = "\n".join(ov.texts)
        first = next(t for t in ov.texts if t.startswith(S.SIMULATED))
        assert first.startswith(f"simulated{S.SEP}GM1 ") and "dx 1 km" in first and "lowest level 250 m" in first
        assert re.search(r"T\+\d+:\d\d:\d\d$", first), first
        assert "tornad" not in text.lower() and "nan" not in text.split()
        # what counts as a storm (the dimmed columns) is stated on both
        assert TH.storm_words() in " ".join(ov.texts), name
    # with saves 300 s apart the pulses are left off, and the picture says why
    assert "Pulse tracks are left off" in "\n".join(by[TH.UPDRAFT].texts)
    assert "every 5 min" in "\n".join(by[TH.UPDRAFT].texts)


@NEEDS_MATPLOTLIB
def test_a_new_saved_time_reads_only_its_own_frame(tmp_path, monkeypatch):
    run = run_directory(tmp_path, times=(900.0, 1200.0), end=False)
    _render(run)
    first = _index(run)
    reads: list[str] = []
    original = TH.Builder.add

    def counting(self, qf):
        reads.append(qf.key)
        return original(self, qf)

    monkeypatch.setattr(TH.Builder, "add", counting)
    # nothing new: nothing is read or drawn again
    picture = run / "quicklook" / TH.UPDRAFT / "still.png"
    stamp = picture.stat().st_mtime_ns
    _render(run)
    assert reads == [] and picture.stat().st_mtime_ns == stamp
    # a third saved time lands: only it is read, and both pictures are drawn again with three columns
    publish_storm(run / "frames", 2, 1500.0)
    _render(run)
    assert [Path(key).name for key in reads] == ["frame-000002"]
    once = {p["id"]: p for p in _index(run)["products"] if p.get("once")}
    assert once[TH.UPDRAFT]["frames"][0]["facts"]["saved_times"] == 3
    assert once[TH.UPDRAFT]["frames"][0]["time_s"] == 1500.0
    assert {p["id"] for p in first["products"] if p.get("once")} >= set(LOOKS)


@NEEDS_MATPLOTLIB
def test_a_run_with_one_saved_time_draws_both_and_says_so(tmp_path, pictures):
    run = run_directory(tmp_path, times=(1500.0,), end=False)
    summary = _render(run)
    assert summary.problems == []
    once = {p["id"]: p for p in _index(run)["products"] if p.get("once")}
    assert all(once[look]["frames"][0]["facts"]["saved_times"] == 1 for look in LOOKS)
    texts = " | ".join(t for name, ov in pictures if name == TH.UPDRAFT for t in ov.texts)
    assert "this run has one saved time so far" in texts and "one saved time" in texts


@NEEDS_MATPLOTLIB
def test_the_look_is_drawn_again_on_request(tmp_path, capsys, monkeypatch):
    import argparse

    from cm1core.quicklook import door
    from cm1core.quicklook import worker as WK

    # the door makes its process a sidecar (nice 19, no GPU): keep this test process as it was
    monkeypatch.setattr(WK.os, "nice", lambda increment: 0)
    for name in (*WK.THREAD_VARIABLES, "MPLBACKEND", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    run = run_directory(tmp_path)
    _render(run, only=("reflectivity",))
    assert not (run / "quicklook" / TH.SPIN).exists()
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    door.add_command(commands)
    arguments = parser.parse_args(["quicklook", str(run), "--look", TH.SPIN, "--json"])
    assert arguments.handler(arguments) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["look"] == TH.SPIN and record["file"] == f"{TH.SPIN}/still.png"
    assert (run / "quicklook" / TH.SPIN / "still.png").is_file()


# ---------------------------------------------------------------- a damaged store, an error, frames without spin

def _frame_looks_drawn(run: Path) -> dict[str, int]:
    """How many saved times each look drawn for every saved time has a picture of."""

    return {p["id"]: len([f for f in p.get("frames") or [] if f.get("file")]) for p in _index(run)["products"]
            if not p.get("once")}


#: The looks this small storm draws at every saved time.  The others say why at the times they are not drawn: the
#: flow near the ground is drawn for grids of 250 m and finer, the storm from the side once there is cloud, the
#: rotation axis where the storm turns.
EVERY_TIME = ("reflectivity", "spin-near-ground", "updraft-5km", "swath", "surface", "rain", "clouds-from-above",
              "updraft-slice", "hail-track", "spin-track")


def _drawn_every_time(run: Path, times: int) -> None:
    """The looks drawn for every saved time are drawn: each has a picture of every saved time, or says why not."""

    drawn = _frame_looks_drawn(run)
    assert all(drawn[look] == times for look in EVERY_TIME), drawn
    for product in _index(run)["products"]:
        if product.get("once"):
            continue
        said = sum(len(row.get("times_s") or []) for row in product.get("not_drawn") or [])
        assert drawn[product["id"]] + said == times, (product["id"], drawn[product["id"]], said)


@NEEDS_MATPLOTLIB
@pytest.mark.parametrize("damage", ["truncated", "not a zip", "empty"])
def test_a_damaged_store_is_read_again_and_every_look_is_drawn(tmp_path, damage):
    run = run_directory(tmp_path, times=(900.0, 1200.0), end=False)
    _render(run)
    store = run / "quicklook" / RD.STATE_DIR / TH.STATE_NAME
    whole = store.read_bytes()
    store.write_bytes({"truncated": whole[:len(whole) // 3], "not a zip": b"not a zip file", "empty": b""}[damage])
    publish_storm(run / "frames", 2, 1500.0)
    summary = _render(run)
    assert summary.problems == []
    once = {p["id"]: p for p in _index(run)["products"] if p.get("once")}
    for look in LOOKS:
        assert once[look]["frames"][0]["facts"]["saved_times"] == 3, look
    _drawn_every_time(run, 3)
    # the store is whole again, with every saved time
    assert len(TH.Builder(store.parent).keys) == 3


@NEEDS_MATPLOTLIB
def test_an_error_in_these_looks_never_stops_the_others(tmp_path, monkeypatch):
    run = run_directory(tmp_path)

    def unexpected(qf, kind=""):
        raise RuntimeError("a frame this code did not expect")

    monkeypatch.setattr(TH, "measure", unexpected)
    summary = _render(run)
    assert sorted(problem.split(":")[0] for problem in summary.problems) == sorted(LOOKS)
    once = {p["id"]: p for p in _index(run)["products"] if p.get("once")}
    assert all(once[look]["frames"] == [] for look in LOOKS)
    assert once["environment"]["frames"]
    _drawn_every_time(run, 3)


@NEEDS_MATPLOTLIB
def test_a_store_that_cannot_be_written_still_draws_both(tmp_path, monkeypatch):
    run = run_directory(tmp_path)

    def full(self):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(TH.Builder, "save", full)
    logged: list[str] = []
    summary = _render(run, log=logged.append)
    assert summary.problems == []
    once = {p["id"]: p for p in _index(run)["products"] if p.get("once")}
    assert all(once[look]["frames"][0]["facts"]["saved_times"] == 3 for look in LOOKS)
    assert any("could not be kept" in line for line in logged)


@NEEDS_MATPLOTLIB
def test_a_cloud_top_at_the_model_top_says_so(tmp_path, pictures, monkeypatch):
    run = run_directory(tmp_path)
    monkeypatch.setattr(TH, "_cloud_top", lambda qf, rows, sl, mask: float(qf.zh[-1]))
    _render(run, only=LOOKS)
    once = {p["id"]: p for p in _index(run)["products"] if p.get("once")}
    facts = once[TH.UPDRAFT]["frames"][0]["facts"]
    assert facts["cloud_top_at_model_top"] is True and facts["cloud_top_max_m"] == 19750.0
    texts = [t for name, ov in pictures if name == TH.UPDRAFT for t in ov.texts]
    assert "the model's top level: the cloud may reach higher" in texts


@NEEDS_MATPLOTLIB
def test_frames_without_spin_leave_the_spin_look_off_and_say_so(tmp_path, pictures):
    import shutil

    from cm1out_support import end_event, publish
    from quicklook_support import mark_microphysics, storm

    run = tmp_path / "nowind"
    frames = run / "frames"
    frames.mkdir(parents=True)
    shutil.copy(NAMELIST_24, run / "namelist.input")
    for index, t in enumerate((900.0, 1200.0)):
        carriers, frame = storm(t, swath="none")
        for name in ("u3d", "v3d"):
            carriers.pop(name)
        mark_microphysics(publish(frames, index, carriers, **frame))
    end_event(run / "events.jsonl", frames=2)
    summary = _render(run, only=LOOKS)
    # (the per-frame statistics say they lack u3d; these two looks have no problem)
    assert not [problem for problem in summary.problems if problem.startswith(LOOKS)]
    once = {p["id"]: p for p in _index(run)["products"] if p.get("once")}
    assert once[TH.SPIN]["frames"] == []
    assert [row["reason"] for row in once[TH.SPIN]["not_drawn"]] == [TH.NO_SPIN_WHY]
    assert once[TH.UPDRAFT]["frames"][0]["facts"]["lowest_spin"] is None
    texts = " | ".join(t for name, ov in pictures if name == TH.UPDRAFT for t in ov.texts)
    assert "strongest spin at the lowest level: not in this run's frames" in texts and "nan" not in texts


@NEEDS_MATPLOTLIB
def test_pulse_tracks_are_left_off_for_the_whole_domain(tmp_path, pictures, monkeypatch):
    run = run_directory(tmp_path, times=(1380.0, 1440.0, 1500.0))

    def no_single_storm(rows, kind=""):
        n = len(rows)
        return TH.Followed("domain", [None] * n, "No single storm to follow: a test.", n, n, [False] * n,
                           [None] * n, [True] * n)

    monkeypatch.setattr(TH, "follow", no_single_storm)
    _render(run, only=(TH.UPDRAFT,))
    once = {p["id"]: p for p in _index(run)["products"] if p.get("once")}
    assert once[TH.UPDRAFT]["frames"][0]["facts"]["pulse_tracks"] is None
    texts = " ".join(t for name, ov in pictures if name == TH.UPDRAFT for t in ov.texts)
    assert "Pulse tracks are left off: each column is the whole domain's" in texts


# ---------------------------------------------------------------- following the storm

def _row(t: float, centres, loc=(0.0, 0.0), move=(0.0, 0.0)) -> dict:
    return {"t": t, "centres": [tuple(c) for c in centres], "domainloc": list(loc), "move": list(move)}


def test_one_storm_is_followed_on_the_ground():
    # a moving domain (10 m/s east) with the storm still in it; a second, weaker storm 30 km away
    rows = [_row(300.0 * n, [(40e3, 40e3, 10.0 + n), (70e3, 40e3, 4.0)], loc=(3000.0 * n, 0.0), move=(10.0, 0.0))
            for n in range(6)]
    f = TH.follow(rows)
    assert f.mode == "storm" and f.chosen == [0] * 6 and not any(f.lost) and f.storm_times == 6
    assert [p[0] for p in f.path] == [40e3 + 3000.0 * n for n in range(6)]
    # the stronger rival appears later but out of reach: the followed storm is kept
    rows[4]["centres"] = [(10e3, 70e3, 30.0), (40e3, 40e3, 14.0)]
    rows[5]["centres"] = [(10e3, 70e3, 31.0), (41e3, 40e3, 15.0)]
    f = TH.follow(rows)
    assert f.chosen[4:] == [1, 1] and f.mode == "storm"
    # before a storm forms (weaker than STORM_MIN) the strongest rising air is taken and the column dimmed
    calm = [_row(0.0, [(5e3, 5e3, 0.2)])] + rows
    calm[1:] = [{**r, "t": r["t"] + 300.0} for r in calm[1:]]
    f = TH.follow(calm)
    assert f.storm[0] is False and f.path[0] is None and all(f.storm[1:])


def test_a_line_of_storms_falls_back_to_the_whole_domain_and_says_why():
    rows = [_row(900.0 * n, [(100e3, y, 12.0 + 0.1 * k) for k, y in enumerate((50e3, 70e3, 90e3, 110e3))])
            for n in range(1, 8)]
    f = TH.follow(rows)
    assert f.mode == "domain" and f.chosen == [None] * 7
    assert f.reason.startswith("No single storm to follow") and "whole domain" in f.reason
    assert "line or a cluster" in f.reason


def test_a_followed_updraft_that_keeps_vanishing_falls_back_too():
    rows = [_row(300.0 * n, [((20e3 if n % 2 else 80e3), 40e3, 10.0)]) for n in range(1, 9)]
    f = TH.follow(rows)
    assert f.mode == "domain" and "vanished" in f.reason


def test_a_hurricane_follows_its_lowest_pressure():
    rows = [_row(10800.0 * n, [(500e3 + n * 1e3, 500e3, 5.0 + n)]) for n in range(5)]
    f = TH.follow(rows, "hurricane")
    assert f.mode == "hurricane" and f.chosen == [0] * 5 and all(f.storm)


@NEEDS_MATPLOTLIB
def test_a_hurricane_run_draws_the_updraft_and_not_the_spin(tmp_path):
    import re as _re

    text = NAMELIST_24.read_text(encoding="ascii")
    swirl, count = _re.subn(r"(?m)^(\s*iinit\s*=\s*)\d+", r"\g<1>7", text)
    assert count == 1
    namelist = tmp_path / "hurricane.namelist.input"
    namelist.write_text(swirl, encoding="ascii")
    run = run_directory(tmp_path, namelist=namelist, name="swirl", move=(0.0, 0.0), swath="domain")
    _render(run)
    index = _index(run)
    assert index["run"]["storm_kind"] == "hurricane"
    once = {p["id"]: p for p in index["products"] if p.get("once")}
    updraft = once[TH.UPDRAFT]["frames"][0]
    assert updraft["facts"]["following"] == "hurricane" and updraft["facts"]["radius_m"] == TH.HURRICANE_RADIUS_M
    assert updraft["facts"]["range_m_s"] == [0.0, TH.HURRICANE_W_TOP]
    assert updraft["clock"].endswith(" h")
    assert once[TH.SPIN]["frames"] == []
    assert [row["reason"] for row in once[TH.SPIN]["not_drawn"]] == [TH.THUNDERSTORM_ONLY_WHY]
    # the Results row has no spin button for a hurricane, and keeps the updraft
    shown = [row["id"] for row in GL.more_looks(run.name, run, index, {"state": "finished"}, films_now={})]
    assert TH.UPDRAFT in shown and TH.SPIN not in shown


# ---------------------------------------------------------------- pulses, ranges, the grid

def test_pulse_tracks_follow_maxima_climbing_from_save_to_save():
    z = np.arange(0.0, 16001.0, 250.0)
    times = [60.0 * n for n in range(6)]
    profiles = []
    for n in range(6):
        centre = 4000.0 + 1500.0 * n                       # rising 25 m/s
        profiles.append(20.0 + 20.0 * np.exp(-((z - centre) / 600.0) ** 2))
    [track] = TH.pulse_tracks(times, z, profiles, 60.0)
    assert len(track) == 6 and TH.rise_speed(track) == pytest.approx(25.0, rel=0.02)
    assert max(w for _t, _z, w in track) == pytest.approx(40.0, abs=0.5)
    # the same climb seen every 5 min rises 7.5 km between saves: it is not linked into a track
    assert TH.pulse_tracks(times, z, profiles[::5] * 3, 300.0) == []
    # a maximum below 3 km, or slower than 25 m/s, is not a pulse
    assert TH.pulse_peaks(z, 20.0 + 20.0 * np.exp(-((z - 2000.0) / 600.0) ** 2)) == []
    assert TH.pulse_peaks(z, 10.0 + 10.0 * np.exp(-((z - 8000.0) / 600.0) ** 2)) == []


def test_the_spin_colours_follow_the_grid_and_the_lowest_level():
    """The spin look's rule (1/dx on a 1-2-5 step) on the greater of dx and the lowest level's height."""

    import inspect

    assert TH.spin_range_for(1000.0, 250.0) == P.spin_range(1000.0) == 0.01
    assert TH.spin_range_for(250.0, 62.5) == P.spin_range(250.0) == 0.05
    assert TH.spin_range_for(100.0, 10.0) == P.spin_range(100.0) == 0.1
    # the shipped 100 m preset (250 m levels, so a 125 m lowest level) keeps the spin look's 0.1
    assert TH.spin_range_for(100.0, 125.0) == P.spin_range(100.0) == 0.1
    # a 67 m grid on 500 m levels (lowest level 250 m) resolves spin near the ground like a 250 m grid; on 50 m
    # levels like its own grid
    assert TH.spin_range_for(67.0, 250.0) == P.spin_range(250.0) == 0.05
    assert TH.spin_range_for(67.0, 25.0) == P.spin_range(67.0) == 0.2
    assert TH.spin_range_for(1000.0, 50.0) == 0.01
    # the colour key says what set the range
    assert TH.spin_range_words(1000.0, 250.0) == TH.spin_range_words(100.0, 125.0) == \
        "the spin look's range at this grid"
    assert TH.spin_range_words(67.0, 250.0) == "the range for a 250 m lowest level"
    # once the spin look's own range follows the lowest level too, the two rules are one
    if "lowest_m" in inspect.signature(P.spin_range).parameters:
        for dx, z0 in ((1000.0, 250.0), (100.0, 125.0), (67.0, 250.0), (67.0, 25.0), (250.0, 62.5)):
            assert TH.spin_range_for(dx, z0) == P.spin_range(dx, z0), (dx, z0)


def test_the_helicity_line_follows_the_grid_and_is_never_flat_at_its_top():
    assert [TH.uh_top_for(r) for r in (0.005, 0.01, 0.02, 0.05, 0.1)] == [1000.0, 1000.0, 2000.0, 5000.0, 10000.0]
    assert TH.helicity_top([300.0, 903.0, float("nan")], 1000.0) == 1000.0
    # a run whose peak passes the grid's top gets the next 1-2-5 step, and the picture says so (below)
    assert TH.helicity_top([11406.0, 19876.0], 10000.0) == 20000.0
    assert TH.helicity_top([float("nan")], 2000.0) == 2000.0


def _columns(nz: int = 4, at_edge: bool = False) -> TH.Columns:
    zero = np.zeros(nz + 1, np.float32)
    return TH.Columns(zero, zero, zero, np.zeros(nz, np.float32), float("nan"), float("nan"), float("nan"),
                      at_edge, at_edge)


def test_a_storm_held_at_the_domains_edge_is_marked_and_said():
    # a 2 km grid of 36 x 36 cells: the finder leaves out 3 cells at each edge, so it can come no nearer the
    # southern edge than y = 7 km, where a 5 km circle still does not reach past the edge
    grid = {"nx": 36, "ny": 36, "dx": 2000.0, "dy": 2000.0, "zf": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "zh": [250.0, 750.0, 1250.0, 1750.0]}
    assert TH.finder_margin(36, 36) == 3
    assert TH.at_finder_edge(33e3, 7e3, grid) and TH.at_finder_edge(65e3, 40e3, grid)
    assert not TH.at_finder_edge(35e3, 9e3, grid) and not TH.at_finder_edge(36e3, 36e3, grid)
    ys = (25e3, 17e3, 11e3, 7e3, 7e3, 7e3)                 # drifting south, then held at the finder's margin
    rows = [{**_row(300.0 * (n + 1), [(35e3, y, 12.0)]), "circles": [_columns()], "domain": _columns()}
            for n, y in enumerate(ys)]
    rows += [{**_row(300.0 * (len(ys) + m + 1), [(40e3, 30e3, 1.0)]), "circles": [_columns()],
              "domain": _columns()} for m in range(3)]
    f = TH.follow(rows)
    assert f.mode == "storm" and f.storm == [True] * 6 + [False] * 3
    pic = TH.Picture(rows, f, {"grid": grid, "radius_m": 5000.0}, [], "", 300.0)
    edge, lost = TH._flags(pic)
    assert edge == [False, False, False, True, True, True, False, False, False] and not any(lost)
    words = " ".join(text for text, _colour in TH._followed_words(pic, edge, lost, S.fmt_clock))
    assert "Grey bar above a column: the storm was at the domain's edge" in words
    assert f"The last saved time with a storm ({S.fmt_clock(1800.0)}) was at the edge" in words
    assert "may have left the domain rather than died out" in words
    # the threshold behind the dimmed columns is stated under the time axis
    caption = TH.storm_caption(pic)
    assert caption.startswith("Dimmed: no storm. ") and caption.endswith(TH.storm_words())
    assert "3 m/s or more on average from 2 to 6 km, smoothed over 3 km" in caption
    # a circle that reaches past the edge is marked as before
    rows[1]["circles"] = [_columns(at_edge=True)]
    assert TH._flags(TH.Picture(rows, f, {"grid": grid, "radius_m": 5000.0}, [], "", 300.0))[0][1]


def test_counts_of_one_read_as_one():
    assert "at 1 saved time," in TH._over_top([1500.0, 10.0], [600.0, 900.0], 1000.0, str, S.fmt_clock)
    assert "at 2 saved times," in TH._over_top([1500.0, 1200.0], [600.0, 900.0], 1000.0, str, S.fmt_clock)
    grid = {"nx": 36, "ny": 36, "dx": 2000.0, "dy": 2000.0, "zf": [0.0, 500.0], "zh": [250.0]}
    for strength, start in ((12.0, "One storm followed: the one saved time had one."),
                            (1.0, "No storm yet at the one saved time")):
        rows = [{**_row(600.0, [(35e3, 35e3, strength)]), "circles": [_columns(0)], "domain": _columns(0)}]
        pic = TH.Picture(rows, TH.follow(rows), {"grid": grid, "radius_m": 5000.0}, [], "", None)
        assert TH._followed_words(pic, [False], [False], S.fmt_clock)[0][0].startswith(start)


def test_the_fastest_share_and_the_circles():
    assert TH.circle_cells(5000.0, 1000.0, 1000.0) == 81
    assert TH.kth_largest(np.arange(100.0), 10) == 90.0
    assert TH.kth_largest(np.array([3.0]), 8) == 3.0
    places = TH.strongest_places(np.array([[0, 1, 0, 0, 5], [0, 9, 0, 0, 0], [0, 0, 0, 0, 7]], float), 3, 2.0)
    # the 9, then the 7 (the 5 is within 2 cells of it), then the first cell left (a weak candidate, kept)
    assert places == [(1, 1), (2, 4), (0, 3)]
    a = np.arange(25.0).reshape(5, 5)
    assert TH.box_mean(a, 3)[2, 2] == pytest.approx(12.0) and TH.box_mean(a, 3)[0, 0] == pytest.approx(3.0)
    edges = TH.column_edges([600.0, 1200.0, 1500.0, 1800.0])
    assert list(edges) == [300.0, 900.0, 1350.0, 1650.0, 1950.0]
    assert [lab for _t, lab in TH.time_ticks(0.0, 7200.0, False)][:3] == ["0:00", "0:15", "0:30"]
    assert [lab for _t, lab in TH.time_ticks(0.0, 345600.0, True)][:3] == ["0", "12 h", "1 d"]


@NEEDS_MATPLOTLIB
def test_the_zeta_of_a_box_is_the_spin_looks_own(tmp_path):
    from cm1core.quicklook.frame import QuickFrame, frames_of
    from cm1core.cm1out import frames as F

    run = run_directory(tmp_path)
    t, key, path, segment = frames_of(F.discover(run))[-1]
    qf = QuickFrame.open(path, key=key, segment=segment)
    try:
        for k in (0, 7):
            whole = qf.zeta_level(k)[0]
            assert np.array_equal(TH.zeta_box(qf, k, 5, 17, 3, 20), whole[5:17, 3:20])
            assert np.array_equal(TH.zeta_box(qf, k, 0, qf.ny, 0, qf.nx), whole)
    finally:
        qf.close()


# ---------------------------------------------------------------- the other looks, the lists

@NEEDS_MATPLOTLIB
def test_the_other_looks_are_byte_identical_with_and_without_these(tmp_path):
    a = run_directory(tmp_path / "a")
    b = run_directory(tmp_path / "b")
    _render(a, only=tuple(pid for pid in RD.LOOK_IDS if pid not in LOOKS))
    _render(b)
    files = sorted(p.relative_to(a / "quicklook") for p in (a / "quicklook").rglob("*.png")
                   if p.parent.name != "thumb" and p.name != "timeseries.png")
    assert files and not any(str(f).startswith(LOOKS) for f in files)
    for rel in files:
        assert (a / "quicklook" / rel).read_bytes() == (b / "quicklook" / rel).read_bytes(), rel


@NEEDS_MATPLOTLIB
def test_a_run_drawn_before_these_looks_offers_to_draw_them_in_a_pass(tmp_path):
    run = run_directory(tmp_path)
    _render(run, only=tuple(pid for pid in RD.LOOK_IDS if pid not in LOOKS))
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    for look in LOOKS:
        # every saved time is read: a pass in the background ("Draw the new looks"), not a call that waits
        assert rows[look]["action"] == "all" and not rows[look]["available"], look
        assert rows[look]["why_not"].startswith("This run's pictures were drawn before")
    assert rows["environment"]["available"]
    # the pass draws only these two, from the frames it reads once
    _render(run)
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    assert all(rows[look]["available"] and rows[look]["count"] == 1 for look in LOOKS)
    # while the run goes, the worker draws them as the saved times land
    live = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "running"}, films_now={})}
    assert all(live[look]["available"] for look in LOOKS)


def test_the_servers_lists_are_the_workers():
    from cm1core.gui import runview as RV
    from cm1core.quicklook import door

    assert GL.ONCE_LOOKS == RD.ONCE_IDS and set(LOOKS) <= set(GL.NEW_LOOKS) and set(GL.PASS_LOOKS) == set(LOOKS)
    assert GL.ONCE_THUNDERSTORM_ONLY == TH.THUNDERSTORM_ONLY
    assert set(GL.ONCE_DRAWN) == set(GL.ONCE_LOOKS)
    copy = {row["id"]: row for row in GL.load_copy()["more_looks"]}
    for look in LOOKS:
        assert copy[look]["kind"] == "once" and copy[look]["title"] == TH.TITLES[look]
        assert look in RV.MORE_LOOKS
    # the words give each look's own circle, and what counts as a storm (the dimmed columns)
    layer = (f"or more on average from {TH.FIND_LAYER_M[0] / 1000:g} to {TH.FIND_LAYER_M[1] / 1000:g} km, smoothed "
             f"over {TH.FIND_SMOOTH_M / 1000:g} km")
    assert f"{TH.STORM_MIN:g} m/s {layer}" in TH.storm_words()
    # (the pages write speeds of rising air out in words, which their copy lint keeps apart from wind speeds)
    words = f"{TH.STORM_MIN:g} metres per second {layer}"
    assert words in copy[TH.UPDRAFT]["text"] and words in copy[TH.SPIN]["note"]
    assert "at least half as strong" in copy[TH.UPDRAFT]["text"]
    assert f"{TH.RADIUS_M / 1000:g} km circle" in copy[TH.UPDRAFT]["text"]
    assert f"{TH.SPIN_RADIUS_M / 1000:g} km circle" in copy[TH.SPIN]["text"]
    assert f"{TH.HURRICANE_RADIUS_M / 1000:g} km" in copy[TH.UPDRAFT]["text_hurricane"]
    source = Path(door.__file__).read_text(encoding="utf-8")
    assert all(look in source for look in LOOKS)
