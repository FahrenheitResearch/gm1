"""Picture fixes: the helicity track without CM1's shs, the spin range on coarse levels, a column outside the domain,
a run whose settings cannot be read, a hurricane's film menu, and the GUI's slice extras and "draw again".

1. A run whose frames carry the model's wind maximum but no helicity maximum (``sws``/``sws2`` without
   ``shs``/``shs2``: GM1 writes none on stretched levels) keeps the helicity half of the swath: each saved frame's
   2-5 km updraft helicity (CM1's ``uh``, else summed from w with CM1's calcuh method), and says so.  A run with
   CM1's ``shs`` draws exactly what it drew.
2. The spin range follows the lowest level's height as well as dx; every preset keeps its range.
3. ``--column --at`` outside the domain is refused in plain words, as ``--section`` is.
4. A run whose namelist cannot be read is drawn with one plain line saying so.
5. A hurricane run's film menu leaves out the looks it never draws (and the parcels when it followed none), and
   every other run's the hurricane looks.
6. The GUI's slices send a level k and a section top, and Results offers "Draw every picture again".
7. Pictures that read wrongly: a swath canvas taller than wide (titles over each other, the helicity's words on its
   colour legend), a coarse grid's spin legend (0.00), a hurricane's swirl film (one white disk), the updraft
   footer of a run without CM1's uh, a 2-D field called "at the surface", and a section top that is not a height.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

from cm1core.quicklook import film as F
from cm1core.quicklook import products as P
from cm1core.quicklook import render as RD
from cm1core.quicklook import skewt as SK
from cm1core.quicklook import slices as SL
from cm1core.quicklook import style as S
from cm1core.quicklook import swath as SW
from cm1core.quicklook.frame import QuickFrame, lineage

import importlib.util  # noqa: E402

NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")
from cm1import_support import namelist_tool  # noqa: E402
from quicklook_support import NAMELIST_24, run_directory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "src" / "cm1core" / "presets"
STATIC = ROOT / "src" / "cm1core" / "gui" / "static"


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


@pytest.fixture()
def pictures(monkeypatch):
    """Every Overlay a product saves, with its words."""

    seen: list[tuple[str, S.Overlay]] = []
    original = P.save

    def spy(ov, base, path):
        seen.append((Path(path).parent.name, ov))
        return original(ov, base, path)

    monkeypatch.setattr(P, "save", spy)
    return seen


@pytest.fixture()
def boxes(monkeypatch):
    """The box of every text, colour bar and rule each picture draws, by picture (its Overlay, kept alive here so
    no later picture takes its place)."""

    drawn: dict[S.Overlay, list[tuple[str, tuple[float, float, float, float]]]] = {}
    text, bar, rule = S.Overlay.text, S.Overlay.ramp_bar, S.Overlay.rule

    def spy_text(self, xy, words, *args, **kwargs):
        box = text(self, xy, words, *args, **kwargs)
        if words and tuple(box) != (0, 0, 0, 0):
            drawn.setdefault(self, []).append((words, tuple(box)))
        return box

    def spy_bar(self, name, box, **kwargs):
        drawn.setdefault(self, []).append((f"{name} colour bar", tuple(box)))
        return bar(self, name, box, **kwargs)

    def spy_rule(self, x, y, length, *, thickness=2.0, vertical=False, **kwargs):
        t = max(1, self.U(thickness))
        w, h = (t, length) if vertical else (length, t)
        drawn.setdefault(self, []).append(("scale bar" if not vertical else "rule", (x, y, x + w, y + h)))
        return rule(self, x, y, length, thickness=thickness, vertical=vertical, **kwargs)

    monkeypatch.setattr(S.Overlay, "text", spy_text)
    monkeypatch.setattr(S.Overlay, "ramp_bar", spy_bar)
    monkeypatch.setattr(S.Overlay, "rule", spy_rule)
    return drawn


def _collisions(items) -> list[tuple[str, str]]:
    """Every pair of drawn boxes that overlap, and every box outside the picture."""

    out = []
    for n, (a, ra) in enumerate(items):
        if ra[0] < 0 or ra[1] < 0 or ra[2] > P.W or ra[3] > P.H:
            out.append((a, "outside the picture"))
        for b, rb in items[n + 1:]:
            if ra[0] < rb[2] and rb[0] < ra[2] and ra[1] < rb[3] and rb[1] < ra[3]:
                out.append((a, b))
    return out


def _render(run: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return RD.render(run, **kwargs)


def _index(run: Path) -> dict:
    return json.loads((run / "quicklook" / "index.json").read_text(encoding="utf-8"))


def _swath_facts(run: Path) -> dict:
    products = {p["id"]: p for p in _index(run)["products"]}
    return products["swath"]["frames"][-1]["facts"]


# ---------------------------------------------------------------- 1. the helicity track without CM1's shs

@NEEDS_MATPLOTLIB
@pytest.mark.parametrize("with_uh", [False, True])
def test_the_models_wind_without_its_helicity_keeps_the_helicity_track(tmp_path, pictures, with_uh):
    """GM1's stretched levels: sws2 on the ground canvas, no shs2.  The helicity half is each saved frame's own,
    placed where the domain was, and the picture says it was summed from w (or is CM1's uh of each frame)."""

    run = run_directory(tmp_path, swath="canvas-wind", uh=with_uh)
    _render(run, only=("swath",))
    facts = _swath_facts(run)
    assert facts["source"] == "model" and facts["since"] == "since the start"
    words = SW.UH_WORDS[SW.UH_CM1 if with_uh else SW.UH_CALCUH]
    assert words in facts["uh_method"] and "at each saved frame only (every 300 s)" in facts["uh_method"]
    assert "the frames carry no CM1 shs2" in facts["uh_method"]
    # rebuilt by hand: the running maximum of each frame's own helicity at round(domainloc / dx)
    layout, rows = lineage(run)
    expected = []
    for t, key, path, seg in rows:
        qf = QuickFrame.open(path, key=key)
        try:
            assert qf.has("sws2") and not qf.has("shs2") and qf.has("uh") is with_uh
            plane = SW.plane_of(qf)
            assert plane.source == "model" and plane.uh_source == (SW.UH_CM1 if with_uh else SW.UH_CALCUH)
            assert plane.uh_origin == SW.frame_origin(qf) != plane.origin
            uh, source = qf.uh()
            assert source == ("CM1 uh" if with_uh else "CM1's calcuh sum")
            expected.append((SW.frame_origin(qf), uh))
        finally:
            qf.close()
    builder = SW.SwathBuilder(run / "quicklook" / ".state")
    sw = builder.swath(300.0)
    assert sw.uh is not None and float(sw.uh.max()) == pytest.approx(facts["uh_max"])
    assert facts["uh_max"] == pytest.approx(max(float(uh.max()) for _o, uh in expected)) and facts["uh_max"] > 10.0
    for (ox, oy), uh in expected:
        x, y = ox - sw.origin_cells[0], oy - sw.origin_cells[1]
        assert (sw.uh[y:y + uh.shape[0], x:x + uh.shape[1]] >= uh - 1e-6).all()
    # The wind is still the model's own: the footer says sws2 alone, the panel says how the helicity was made.
    by = dict(pictures)
    text = "\n".join(by["swath"].texts)
    assert "wind: the model's every-step maximum on the ground (CM1's sws2)" in text
    assert "2–5 km updraft helicity since the start, saved frames" in text
    assert f"helicity {words}" in facts["uh_method"] and f"helicity {words}" in text     # a wide canvas: in full
    assert "no updraft-helicity track" not in text
    assert by["swath"].check() == [] and by["swath"].footer_cut == []


@NEEDS_MATPLOTLIB
def test_a_domain_fixed_to_the_ground_without_shs_keeps_the_helicity_track(tmp_path, pictures):
    run = run_directory(tmp_path, move=(0.0, 0.0), swath="domain", uh=False)
    _render(run, only=("swath",))
    facts = _swath_facts(run)
    assert facts["source"] == "model" and facts["uh_max"] > 10.0
    assert SW.UH_WORDS[SW.UH_CALCUH] in facts["uh_method"]
    assert facts["uh_method"].endswith("the frames carry no CM1 shs")       # sws, not the ground canvas sws2
    assert all(ov.check() == [] and ov.footer_cut == [] for _name, ov in pictures)


@NEEDS_MATPLOTLIB
def test_a_run_with_cm1s_shs_draws_its_swath_as_before(tmp_path, pictures):
    """The model's own helicity maximum: the words are 0.1.5's, and nothing is said under the panel."""

    run = run_directory(tmp_path, swath="canvas")
    _render(run, only=("swath",))
    facts = _swath_facts(run)
    assert facts["source"] == "model" and "uh_method" not in facts
    text = "\n".join(dict(pictures)["swath"].texts)
    assert "the model's every-step maximum on the ground (CM1's sws2 and shs2)" in text
    assert "saved frames" not in text and SW.UH_WORDS[SW.UH_CALCUH] not in text
    builder = SW.SwathBuilder(run / "quicklook" / ".state")
    assert builder.uh_sources == {SW.UH_MODEL} and builder.swath(300.0).uh_method == ""


def test_maxima_kept_by_an_earlier_version_are_folded_again(tmp_path):
    """0.1.5 kept no helicity for a run without shs; its kept maxima are folded again from the frames, so a
    run's helicity track is never blank for its early frames."""

    run = run_directory(tmp_path, swath="canvas-wind", uh=False)
    layout, rows = lineage(run)
    opener = lambda path, key, seg: QuickFrame.open(path, key=key, segment=seg)  # noqa: E731
    state = tmp_path / "state"
    builder = SW.SwathBuilder(state)
    builder.catch_up(rows, opener)
    builder.save()
    again = SW.SwathBuilder(state)
    assert again.keys == [key for _t, key, _p, _s in rows] and again.uh is not None
    # rewrite the kept state as 0.1.5 did: no version, no helicity sources, no helicity
    with np.load(state / SW.STATE, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        wind = np.array(data["wind"])
    for key in ("version", "uh_sources"):
        meta.pop(key)
    np.savez(state / SW.STATE, wind=wind, meta=np.array(json.dumps(meta)))
    old = SW.SwathBuilder(state)
    assert old.keys == [] and old.wind is None            # not trusted: folded again by catch_up
    old.catch_up(rows, opener)
    assert old.uh is not None and np.array_equal(old.uh, again.uh)


# ---------------------------------------------------------------- 2. the spin range

def test_the_spin_range_follows_the_lowest_levels_height_as_well_as_dx():
    # the grid spacing alone, as before
    assert {dx: P.spin_range(dx) for dx in (50.0, 100.0, 250.0, 500.0, 1000.0)} == \
        {50.0: 0.2, 100.0: 0.1, 250.0: 0.05, 500.0: 0.02, 1000.0: 0.01}
    # a fine grid on 500 m levels (lowest level 250 m): the level sets it (the 67 m run peaked near 0.016 1/s)
    assert P.spin_range(67.0, 250.0) == 0.05 and P.spin_range(33.0, 250.0) == 0.05
    assert P.spin_range(82.9, 250.0) == 0.05 and P.spin_scale_m(67.0, 250.0) == 250.0
    assert P.spin_range_words(67.0, 250.0) == "for a 250 m lowest level"
    # a lowest level at or below dx, or one that rounds to dx's own range, changes nothing
    for dx, z0 in ((1000.0, 250.0), (250.0, 125.0), (100.0, 125.0), (150.0, 50.0), (333.0, 250.0), (250.0, 10.0),
                   (8000.0, 25.0)):
        assert P.spin_range(dx, z0) == P.spin_range(dx), (dx, z0)
        assert P.spin_range_words(dx, z0) == "at this dx"


def test_every_preset_keeps_its_spin_range():
    """The presets' pictures are what they were: each lowest level is at or near the grid spacing."""

    checked = 0
    for namelist in sorted(PRESETS.glob("*/namelist.input")):
        text = namelist.read_text(encoding="ascii")

        def word(name: str) -> float:
            return float(re.search(rf"(?m)^\s*{name}\s*=\s*([0-9.]+)", text).group(1))

        dz = word("dz_bot") if int(word("stretch_z")) else word("dz")
        assert P.spin_range(word("dx"), dz / 2.0) == P.spin_range(word("dx")), namelist.parent.name
        checked += 1
    assert checked >= 20


@NEEDS_MATPLOTLIB
def test_the_spin_look_and_track_say_what_sets_their_range(tmp_path, pictures, monkeypatch):
    run = run_directory(tmp_path, times=(1500.0,))
    _render(run, only=("spin-near-ground", "spin-track"))
    by = dict(pictures)
    assert f"±0.01 {S.PER_S} fixed at this dx" in "\n".join(by["spin-near-ground"].texts)
    assert "fixed at this dx" in "\n".join(by["spin-track"].texts)
    facts = {p["id"]: p["frames"][-1]["facts"] for p in _index(run)["products"] if p["frames"]}
    assert facts["spin-near-ground"]["range_per_s"] == 0.01 and facts["spin-near-ground"]["range_set_by"] == "dx"
    # the same frame, as if its grid were 50 m under its 250 m lowest level: the level sets the range, and says so
    pictures.clear()
    monkeypatch.setattr(QuickFrame, "dx", property(lambda self: 50.0))
    monkeypatch.setattr(QuickFrame, "dy", property(lambda self: 50.0))
    _render(run, only=("spin-near-ground",), force=True)
    words = "\n".join(dict(pictures)["spin-near-ground"].texts)
    assert f"±0.05 {S.PER_S} fixed for a 250 m lowest level" in words
    assert dict(pictures)["spin-near-ground"].footer_cut == []
    spin = {p["id"]: p for p in _index(run)["products"]}["spin-near-ground"]["frames"][-1]["facts"]
    assert spin["range_per_s"] == 0.05 and spin["range_set_by"] == "lowest level"


# ---------------------------------------------------------------- 3. a column outside the domain

def _one(tmp_path: Path):
    from cm1core.cm1out import settings as _settings
    from cm1core.quicklook.facts import run_facts

    run = run_directory(tmp_path, times=(1500.0,))
    layout, rows = lineage(run)
    qf = QuickFrame.open(rows[-1][2], key=rows[-1][1])
    settings = _settings.resolve(layout)
    return run, qf, run_facts(layout, qf.config, qf.dx, qf.dy, settings=settings), settings


@NEEDS_MATPLOTLIB
def test_a_column_outside_the_domain_is_refused_in_plain_words(tmp_path):
    run, qf, facts, _settings = _one(tmp_path)
    out = run / "quicklook" / "slices"
    try:
        # iorigin = 2: the 24 km domain runs from -12 to 12 km
        for x, y in ((512000.0, 0.0), (-12.5, 0.0), (0.0, 12.01)):
            with pytest.raises(SL.SliceError, match=r"outside the domain, which runs from x = -12 to 12 km and "
                                                    r"y = -12 to 12 km"):
                SK.column(qf, facts, out / "column.png", x_km=x, y_km=y)
        # the edges themselves are inside: the far edge is the last column
        edge = SK.column(qf, facts, out / "column-edge.png", x_km=12.0, y_km=-12.0)
        assert (edge["i"], edge["j"]) == (24, 1)
        inside = SK.column(qf, facts, out / "column-in.png", x_km=3.2, y_km=-4.9)
        assert (inside["x_km"], inside["y_km"]) == (3.5, -4.5)
    finally:
        qf.close()


@NEEDS_MATPLOTLIB
def test_the_command_line_refuses_a_column_outside_the_domain(tmp_path):
    run = run_directory(tmp_path, times=(1500.0,))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    base = [sys.executable, "-m", "cm1core", "quicklook", str(run)]
    bad = subprocess.run(base + ["--column", "--at", "512000", "0"], capture_output=True, text=True, env=env)
    assert bad.returncode == 2 and "is outside the domain" in bad.stderr and "Traceback" not in bad.stderr
    section = subprocess.run(base + ["--section", "w", "--from", "-40", "0", "--to", "0", "0"], capture_output=True,
                             text=True, env=env)
    assert section.returncode == 2 and "inside the domain" in section.stderr


# ---------------------------------------------------------------- 4. settings that cannot be read

@NEEDS_MATPLOTLIB
def test_a_run_whose_namelist_cannot_be_read_says_so_in_one_line(tmp_path, monkeypatch):
    run = run_directory(tmp_path, times=(1500.0,))
    seen: list[str] = []
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(tmp_path / "no-such-namelist-tool"))
    summary = _render(run, only=("rain",), log=seen.append)
    assert len(summary.notes) == 1 and seen[0] == summary.notes[0]      # said once, before the frames
    line = summary.notes[0]
    assert line.startswith("Drawn without the run's settings (cannot read the CM1 namelist ")
    assert "a hurricane run is drawn as a thunderstorm run" in line and "--namelist FILE or --deck FILE" in line
    assert "\n" not in line
    assert _index(run)["notes"] == [line]
    assert summary.record()["notes"] == [line]


@NEEDS_MATPLOTLIB
def test_frames_with_no_namelist_beside_them_say_so_and_a_readable_one_says_nothing(tmp_path):
    run = run_directory(tmp_path, times=(1500.0,))
    summary = _render(run, only=("rain",))
    assert summary.notes == [] and "notes" not in _index(run)
    (run / "namelist.input").unlink()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    drawn = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(run), "--only", "rain", "--force"],
                           capture_output=True, text=True, env=env)
    assert drawn.returncode == 0, drawn.stderr
    assert "Drawn without the run's settings (there is no CM1 namelist or GM1 deck beside these frames)" in \
        drawn.stdout + drawn.stderr
    column = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(run), "--column", "--json"],
                            capture_output=True, text=True, env=env)
    assert column.returncode == 0, column.stderr
    assert "Drawn without the run's settings" in column.stderr
    assert json.loads(column.stdout)["note"].startswith("Drawn without the run's settings")


# ---------------------------------------------------------------- 5. a hurricane's film menu

def test_a_hurricane_runs_film_menu_leaves_out_the_thunderstorm_looks():
    from cm1core.gui import looks as LK

    assert LK.HURRICANE_ONLY == P.HURRICANE_ONLY and LK.THUNDERSTORM_ONLY == P.THUNDERSTORM_ONLY
    hurricane = LK.film_looks_for("hurricane")
    assert not set(hurricane) & set(P.THUNDERSTORM_ONLY) and set(P.HURRICANE_ONLY) <= set(hurricane)
    assert {"reflectivity", "rain", "clouds-from-above", "updraft-slice", "swirl", "parcels"} <= set(hurricane)
    storm = LK.film_looks_for(None)
    assert not set(storm) & set(P.HURRICANE_ONLY) and set(P.THUNDERSTORM_ONLY) <= set(storm)
    assert set(storm) | set(hurricane) == set(LK.FILM_LOOKS)
    # a hurricane that followed no air parcels has no parcels film; a thunderstorm run keeps it (its film says how
    # to follow them next time, which a hurricane's Customize cannot)
    assert "parcels" not in LK.film_looks_for("hurricane", parcels=False)
    assert set(LK.film_looks_for("hurricane", parcels=False)) == set(hurricane) - {"parcels"}
    assert "parcels" in LK.film_looks_for(None, parcels=False)


def test_whether_a_run_follows_parcels(tmp_path):
    from cm1core.gui import looks as LK

    run = tmp_path / "run"
    run.mkdir()
    assert LK.follows_parcels(run, None) is False                     # nothing to say it does
    (run / "namelist.input").write_text(" &param2\n iprcl     =  0,\n /\n", encoding="ascii")
    assert LK.follows_parcels(run, {}) is False
    (run / "namelist.input").write_text(" &param2\n iprcl     =  1,\n /\n", encoding="ascii")
    assert LK.follows_parcels(run, None) is True                      # a run still going: its track comes at the end
    (run / "namelist.input").write_text(" &param2\n iprcl     =  0,\n /\n", encoding="ascii")
    drawn = {"products": [{"id": "parcels", "frames": [{"file": "parcels/t000000.png"}]}]}
    assert LK.follows_parcels(run, drawn) is True
    assert LK.follows_parcels(run, {"products": [{"id": "parcels", "frames": []}]}) is False
    track = run / "segments" / "000" / "frames" / "lagrangian-track" / "parcel-attributes.f32"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"")
    assert LK.follows_parcels(run, None) is True


@NEEDS_MATPLOTLIB
def test_a_film_of_a_look_the_run_does_not_draw_says_why(tmp_path):
    from test_hurricane_preset import _hurricane_run

    run = _hurricane_run(tmp_path)
    _render(run)
    assert F.why_no_pictures(run / "quicklook", "swath") == P.THUNDERSTORM_ONLY_WHY
    job = F.FilmJob(run, "spin-near-ground", (), 6.0, 8, run / "quicklook")
    with pytest.raises(F.FilmError, match="made for thunderstorms"):
        F.make(job)
    storm = run_directory(tmp_path / "storm", times=(900.0, 1500.0))
    _render(storm)
    with pytest.raises(F.FilmError, match="drawn for hurricane runs only"):
        F.make(F.FilmJob(storm, "wind-lowest", (), 6.0, 8, storm / "quicklook"))


# ---------------------------------------------------------------- 6. the GUI

def test_draw_again_is_the_quicklook_pass_with_force():
    from cm1core.gui import commands as C
    from cm1core.gui import looks as LK

    run = Path("/runs/storm")
    assert C.display(C.quicklook_pass_args(run)) == "gpuwm-cm1 quicklook /runs/storm"
    assert C.display(C.quicklook_pass_args(run, again=True)) == "gpuwm-cm1 quicklook /runs/storm --force"
    index = {"frames": [{}] * 25, "run": {"grid": "120 × 120 × 40, dx 1 km"}}
    # 0.1.5's looks drawn again (measured as a whole), and each look added since at its own cost
    added = sum(LK.SECONDS_PER_FRAME_LOOK[look] for look in LK.NEW_LOOKS
                if look not in LK.LOOKS_015 and look not in LK.HURRICANE_ONLY)
    assert LK.about_seconds(index, "again") == round(25 * (LK.SECONDS_PER_FRAME_ALL + added))     # 0.576 M cells
    finer = {"frames": [{}] * 25, "run": {"grid": "120 × 120 × 53, dx 1 km"}}
    radar = sum(LK.SECONDS_PER_FRAME_LOOK[look] for look in LK.RADAR_LOOKS)
    assert LK.about_seconds(finer, "again") == round(25 * ((LK.SECONDS_PER_FRAME_ALL + added - radar)
                                                           * (763200 / 576000) ** 0.5
                                                           + radar * (763200 / 576000) ** LK.RADAR_CELLS_EXPONENT))
    assert LK.about_seconds(index, "again") > LK.about_seconds(index, "all")


def test_the_slices_send_a_level_and_a_section_top_and_results_offer_draw_again():
    results = (STATIC / "js/run/results.js").read_text(encoding="utf-8")
    assert 'h("option", { value: "level" }, "Level k")' in results
    assert "if (byLevel.value === \"level\") payload.level = Number(levelIn.value);" in results
    assert "else payload.height_m = Number(levelIn.value || 0);" in results
    assert "payload.top_m = top;" in results
    assert 'body: { look: "all", again: true }' in results and "(data.look_commands || {}).again" in results
    assert '"Draw every picture again"' in results


def test_the_server_takes_draw_again_and_refuses_a_hurricanes_thunderstorm_film(tmp_path, monkeypatch):
    from cm1core.gui import looks as LK
    from gui_support import start_server, stop_server
    from test_hurricane_preset import _hurricane_run

    root = tmp_path / "runs"
    root.mkdir()
    run_directory(root, times=(900.0, 1500.0), name="storm")
    _hurricane_run(root)
    server, client = start_server(root, tmp_path / "config")
    try:
        answer = client.post("/api/runs/storm/look", {"look": "all", "again": True})
        assert answer.status == 200 and answer.json["ok"], answer.json
        assert answer.json["command"].endswith("quicklook " + str(root / "storm") + " --force")
        assert answer.json["message"].startswith("Drawing every picture again")
        assert "--force" in (root / "storm" / "commands.log").read_text()
        # the hurricane run's pictures record its kind once drawn; before that its namelist's preset does not
        (root / "swirl" / "quicklook").mkdir(parents=True, exist_ok=True)
        (root / "swirl" / "quicklook" / "index.json").write_text(json.dumps(
            {"run": {"storm_kind": "hurricane"}, "frames": [], "products": []}), encoding="utf-8")
        refused = client.post("/api/runs/swirl/film", {"look": "swath"})
        assert refused.status == 409 and "made for thunderstorms" in refused.json["message"]
        assert "swath" not in refused.json["fix"] and "wind-lowest" in refused.json["fix"]
        results = client.get("/api/runs/swirl/results?mode=advanced").json
        assert not set(results["film_looks"]) & set(P.THUNDERSTORM_ONLY)
        assert set(results["film_commands"]) == set(results["film_looks"])
        # the hurricane followed no air parcels (iprcl = 0): no parcels film, and asking says so without the
        # thunderstorm runs' advice
        assert "parcels" not in results["film_looks"] and "parcels" not in results["film_commands"]
        refused = client.post("/api/runs/swirl/film", {"look": "parcels"})
        assert refused.status == 409 and refused.json["message"] == LK.FILM_NO_HURRICANE_PARCELS
        assert "iprcl = 1" not in refused.json["message"] + refused.json["fix"]
        assert "parcels" not in refused.json["fix"] and "wind-lowest" in refused.json["fix"]
        # every look in the menu has its words, even one the run's rows leave out (its parcels, before any)
        assert results["film_titles"]["parcels"] == "Air rising into the storm"
        assert "(data.film_titles || {})[lid]" in (STATIC / "js/run/results.js").read_text(encoding="utf-8")
        assert results["look_commands"]["again"].endswith("--force")
        plain = client.get("/api/runs/storm/results?mode=advanced").json
        assert not set(plain["film_looks"]) & set(P.HURRICANE_ONLY) and "swath" in plain["film_looks"]
        refused = client.post("/api/runs/storm/film", {"look": "wind-lowest"})
        assert refused.status == 409 and "hurricane runs only" in refused.json["message"]
    finally:
        stop_server(server)


# ---------------------------------------------------------------- 7. pictures that read wrongly

#: The swath's shapes: a domain fixed to the ground (a square canvas), one moving east (wide), and one moving north
#: at 12.5 and 25 m/s (a canvas twice and three times as tall as wide), with and without CM1's shs2.
SHAPES = {"square, no shs": dict(move=(0.0, 0.0), swath="domain", uh=False),
          "east, no shs": dict(swath="canvas-wind", uh=False),
          "east": dict(swath="canvas"),
          "north, no shs": dict(move=(0.0, 12.5), swath="canvas-wind", uh=False),
          "north": dict(move=(0.0, 12.5), swath="canvas"),
          "far north, no shs": dict(move=(0.0, 25.0), swath="canvas-wind", uh=False)}


@NEEDS_MATPLOTLIB
@pytest.mark.parametrize("shape", list(SHAPES))
def test_no_word_of_a_swath_runs_into_another_whatever_its_shape(tmp_path, pictures, boxes, shape):
    run = run_directory(tmp_path, times=(900.0, 1500.0), **SHAPES[shape])
    _render(run, only=("swath",))
    ov = dict(pictures)["swath"]
    assert ov.check() == [] and ov.footer_cut == []
    assert _collisions(boxes[ov]) == []
    facts = _swath_facts(run)
    if "no shs" in shape:
        # how the helicity was made: its two lines under a wide panel, two shorter ones under a square one, and
        # the whole sentence beside a narrow one
        said = {"square, no shs": "short", "east, no shs": "long"}.get(shape, "beside")
        sw = SW.SwathBuilder(run / "quicklook" / ".state").swath(600.0)
        assert facts["uh_method"] == sw.uh_method
        if said == "beside":
            assert sw.uh_method in " ".join(ov.texts)
        else:
            assert set(sw.uh_lines if said == "long" else sw.uh_short) <= set(ov.texts)


@NEEDS_MATPLOTLIB
def test_a_swath_where_everything_fits_is_drawn_as_before(tmp_path, pictures):
    """The wide canvas (the storm moving east): one row of words over each panel, colour legends 300 wide."""

    run = run_directory(tmp_path, times=(900.0, 1500.0), swath="canvas-wind", uh=False)
    _render(run, only=("swath",))
    ov = dict(pictures)["swath"]
    sw = SW.SwathBuilder(run / "quicklook" / ".state").swath(600.0)
    lx, ly = sw.extent_m
    assert lx > ly
    text = "\n".join(ov.texts)
    assert "strongest wind at 250 m since the start" in text
    assert sw.uh_lines[0] in text and sw.uh_lines[1] in text     # the long lines, under the panel


def test_a_narrow_panel_stacks_its_title_over_its_value():
    ov = S.Overlay((P.W, P.H))
    wide = P.Box(80, 200, 858, 0.0, 0.0, 120_000.0, h=700)
    narrow = P.Box(500, 200, 320, 0.0, 0.0, 25_000.0, h=700)
    title, value = "strongest wind at 250 m since the start", "max 28 m/s ground-relative"
    assert P.label_fits(ov, wide, title, value) and not P.label_fits(ov, narrow, title, value)
    assert P.label_fits(ov, narrow, title, None)


def test_a_title_that_does_not_fit_beside_its_value_takes_its_short_words(boxes):
    """A hurricane's first saved times: no rain yet, so the inner panel's value is long ("anywhere in the column:
    nothing above 5 dBZ") and its title ran into it; the title takes its short words on the same row."""

    ov = S.Overlay((P.W, P.H))
    ov.title("Reflectivity at 1 km", "hurricane · 8 km grid · whole 1024 × 1024 km domain")
    zoom = P.Box(982, 182, 740, 262_000.0, 262_000.0, 500_000.0, h=740)
    title, value = "inner 500 km (the square on the left)", "anywhere in the column: nothing above 5 dBZ"
    assert not P.label_fits(ov, zoom, title, value) and P.label_fits(ov, zoom, "inner 500 km", value)
    P.panel_label(ov, zoom, title, value, short="inner 500 km")
    assert "inner 500 km" in ov.texts and title not in ov.texts
    assert _collisions(boxes[ov]) == []
    # with room, the full words and one row, as always
    ov = S.Overlay((P.W, P.H))
    P.panel_label(ov, zoom, title, "anywhere in the column: max 52 dBZ", short="inner 500 km")
    assert title in ov.texts and _collisions(boxes[ov]) == []


def test_a_spin_legend_writes_small_ranges_in_full():
    assert P.spin_tick(0.001, "0.00") == "0.001" and P.spin_tick(-0.001, "−0.00") == "−0.001"
    assert P.spin_tick(0.005, "0.01") == "0.005" and P.spin_tick(0.0025, "0.003") == "0.0025"
    assert P.spin_tick(0.0005, "0.001") == "0.0005"
    # every preset's range is 0.01 1/s or more: its legend is written as it always was
    for dx in (100.0, 250.0, 500.0, 1000.0):
        rng = P.spin_range(dx)
        assert rng >= 0.01
        assert P.spin_tick(rng, f"{rng:.2f}") == f"{rng:.2f}" and P.spin_tick(-rng, "x") == "x"
        assert P.spin_tick(rng / 2, S.fmt_num(rng / 2, 3)) == S.fmt_num(rng / 2, 3)


def test_a_hurricanes_swirl_range_is_its_own():
    # GM1's 8 and 4 km hurricanes peak at 0.006 to 0.010 1/s at the lowest level: ±0.01, where the thunderstorm
    # rule gives ±0.001 and ±0.002
    assert P.spin_range(8000.0, 25.0) == 0.001 and P.spin_range(4000.0, 25.0) == 0.002
    assert P.swirl_range(8000.0, 25.0, hurricane=True) == P.swirl_range(4000.0, 25.0, hurricane=True) == 0.01
    assert P.spin_range_words(8000.0, 25.0, hurricane=True) == "for a hurricane"
    # a finer grid's own range is kept when it is wider, and a thunderstorm run's range is unchanged
    assert P.swirl_range(500.0, 25.0, hurricane=True) == 0.02
    assert P.spin_range_words(500.0, 25.0, hurricane=True) == "at this dx"
    assert P.swirl_range(8000.0, 25.0) == 0.001 and P.spin_range_words(8000.0, 25.0) == "at this dx"
    assert P.HURRICANE_SPIN_RANGE == 0.01


def _swirl_words(run: Path, monkeypatch) -> list[str]:
    seen: list[S.Overlay] = []
    original = F.save_picture

    def spy(ov, base, path):
        seen.append(ov)
        return original(ov, base, path)

    monkeypatch.setattr(F, "save_picture", spy)
    F.make(F.FilmJob(run, "swirl", (), 6.0, 0, run / "quicklook"))
    assert seen and all(ov.check() == [] and ov.footer_cut == [] for ov in seen)
    return seen[-1].texts


@NEEDS_MATPLOTLIB
def test_a_coarse_grids_swirl_legend_and_a_hurricanes_swirl_range(tmp_path, monkeypatch):
    from test_hurricane_preset import _hurricane_run

    # a thunderstorm run on a 2 km grid: ±0.005, written in full (two decimals wrote 0.01)
    storm = run_directory(tmp_path / "storm", times=(900.0, 1500.0))
    monkeypatch.setattr(QuickFrame, "dx", property(lambda self: 2000.0))
    monkeypatch.setattr(QuickFrame, "dy", property(lambda self: 2000.0))
    words = "\n".join(_swirl_words(storm, monkeypatch))
    assert f"0.005 {S.PER_S}" in words and "−0.005" in words and f"spin ±0.005 {S.PER_S} fixed at this dx" in words
    # the same frames as a hurricane's on an 8 km grid: its own ±0.01
    monkeypatch.setattr(QuickFrame, "dx", property(lambda self: 8000.0))
    monkeypatch.setattr(QuickFrame, "dy", property(lambda self: 8000.0))
    run = _hurricane_run(tmp_path)
    words = "\n".join(_swirl_words(run, monkeypatch))
    assert f"spin ±0.01 {S.PER_S} fixed for a hurricane" in words and f"0.01 {S.PER_S}" in words
    assert "0.00" not in words.replace("0.001", "")


@NEEDS_MATPLOTLIB
def test_the_updraft_footer_of_a_run_without_cm1s_uh_keeps_every_clause(tmp_path, pictures):
    run = run_directory(tmp_path, times=(1500.0,), swath="canvas-wind", uh=False)
    _render(run, only=("updraft-5km",))
    ov = dict(pictures)["updraft-5km"]
    text = "\n".join(ov.texts)
    assert "updraft helicity: CM1's calcuh sum" in text and "+ fixed to the ground every" in text
    assert ov.footer_cut == []


@NEEDS_MATPLOTLIB
def test_a_2d_field_is_called_one_and_a_section_top_must_be_a_height(tmp_path):
    run, qf, facts, settings = _one(tmp_path)
    out = run / "quicklook" / "slices"
    try:
        flat = [name for name, f in SL.fields(qf, settings).items() if f.levels(qf) is None]
        assert flat, "the fixture writes a 2-D field"
        record = SL.horizontal(qf, facts, settings, flat[0], out)
        assert record["height_m"] is None and record["level"] is None
        for bad in (float("nan"), float("inf"), -5.0, 0.0):
            with pytest.raises(SL.SliceError, match="The top of a section is a height above the ground in metres"):
                SL.vertical(qf, facts, settings, "w", out / "section.png", a_km=(-8.0, 0.0), b_km=(8.0, 0.0),
                            top_m=bad)
        good = SL.vertical(qf, facts, settings, "w", out / "section.png", a_km=(-8.0, 0.0), b_km=(8.0, 0.0),
                           top_m=6000.0)
        assert good["top_m"] == 6000.0
    finally:
        qf.close()
    results = (STATIC / "js/run/results.js").read_text(encoding="utf-8")
    assert '"the surface"' not in results and "rec.description ? ` · ${rec.description}`" in results


@NEEDS_MATPLOTLIB
def test_a_2d_fields_slice_says_what_it_is(tmp_path, monkeypatch):
    seen: list[S.Overlay] = []
    original = P.save

    def spy(ov, base, path):
        seen.append(ov)
        return original(ov, base, path)

    monkeypatch.setattr(P, "save", spy)
    run, qf, facts, settings = _one(tmp_path)
    try:
        flat = [name for name, f in SL.fields(qf, settings).items() if f.levels(qf) is None]
        SL.horizontal(qf, facts, settings, flat[0], run / "quicklook" / "slices")
    finally:
        qf.close()
    text = "\n".join(seen[-1].texts)
    assert f"{flat[0]}, a 2-D field".upper() in text and "a 2-D field: one value for each column" in text
    assert "at the surface" not in text


@NEEDS_MATPLOTLIB
def test_the_command_line_refuses_a_section_top_that_is_not_a_height(tmp_path):
    run = run_directory(tmp_path, times=(1500.0,))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    base = [sys.executable, "-m", "cm1core", "quicklook", str(run), "--section", "w", "--from", "-8", "0", "--to",
            "8", "0", "--top"]
    for bad in ("nan", "-5"):
        refused = subprocess.run(base + [bad], capture_output=True, text=True, env=env)
        assert refused.returncode == 2 and "Traceback" not in refused.stderr, refused.stderr
        assert "The top of a section is a height above the ground in metres, more than 0" in refused.stderr
