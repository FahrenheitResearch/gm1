"""The storm from the side, the water and ice above each place, and the rotation axis (quicklook.sideview, .axis).

CPU only.  The side view and the water paths are drawn from CM1 r21.1's own 24 x 24 x 40 storm
(``quicklook_support``), and their numbers are worked out again here from the stored state; the rotation axis is
checked on a vortex whose centre is known at every height (a leaning vortex with a 2 km core over a rising
column), on one leaning further than the axis's rule allows, on a hurricane whose eye is wider than the circle (a
hollow vortex), and on air that does not turn, which it must refuse in words.  The looks that were there before stay
byte-identical when these three are drawn beside them.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import warnings

import numpy as np
import pytest

NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")

from cm1import_support import namelist_tool  # noqa: E402
from cm1out_support import end_event, publish  # noqa: E402
from quicklook_support import MASS, NAMELIST_24, STATE, run_directory, storm  # noqa: E402

from cm1core.cm1out import settings as SE  # noqa: E402
from cm1core.gui import looks as GL  # noqa: E402
from cm1core.quicklook import axis as AX  # noqa: E402
from cm1core.quicklook import products as P  # noqa: E402
from cm1core.quicklook import render as RD  # noqa: E402
from cm1core.quicklook import sideview as SV  # noqa: E402
from cm1core.quicklook import style as S  # noqa: E402
from cm1core.quicklook.facts import run_facts  # noqa: E402
from cm1core.quicklook.frame import QuickFrame, lineage  # noqa: E402

NEW = ("side-view", "water-paths", "rotation-axis")
#: The looks 0.1.5 drew for every frame of a thunderstorm run.
BEFORE = ("reflectivity", "spin-near-ground", "updraft-5km", "swath", "surface", "rain", "clouds-from-above",
          "updraft-slice", "hail-track", "spin-track")


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
    return json.loads((run / "quicklook" / "index.json").read_text(encoding="utf-8"))


def _looks(run: Path, index: int = -1) -> P.Looks:
    layout, rows = lineage(run)
    t, key, path, segment = rows[index]
    qf = QuickFrame.open(path, key=key, segment=segment)
    return P.Looks(qf, run_facts(layout, qf.config, qf.dx, qf.dy, settings=SE.resolve(layout)))


def _png_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


# ---------------------------------------------------------------- the side view and the water paths

@NEEDS_MATPLOTLIB
def test_the_side_view_and_water_paths_show_cm1s_own_species_with_their_numbers(tmp_path):
    run = run_directory(tmp_path)
    summary = _render(run, only=NEW)
    assert summary.problems == []
    index = _index(run)
    products = {p["id"]: p for p in index["products"]}
    for look in NEW:
        assert [f["time_s"] for f in products[look]["frames"]] == [900.0, 1200.0, 1500.0], look
        for frame in products[look]["frames"]:
            assert _png_size(run / "quicklook" / frame["file"]) == (1920, 1080)
    d = np.load(STATE)
    rho, dz, dy = d["rho"].astype(np.float64), 500.0, 1000.0
    side = products["side-view"]["frames"][-1]["facts"]
    paths = products["water-paths"]["frames"][-1]["facts"]
    # The core is the strongest w on the w level nearest 5 km (face 10 of the 500 m levels), as the updraft slice.
    j, i = np.unravel_index(int(np.argmax(d["w"][10])), d["w"][10].shape)
    assert side["core_ij"] == [int(i) + 1, int(j) + 1] == paths["core_ij"]
    assert side["core"] == "the strongest w at 5 km"
    # A 20 km slab around the core's row, inside the 24 km domain.
    j0, j1 = side["slab_rows"][0] - 1, side["slab_rows"][1]
    assert j1 - j0 == 20 and 0 <= j0 <= j <= j1 <= 24
    # The slab's path: rho q dy summed over the slab's rows, each light its species together.
    groups = {"water": ("qc",), "rain": ("qr",), "ice": ("qi", "qs"), "dense": ("qg",)}
    for group, names in groups.items():
        slab = sum((rho[:, j0:j1, :] * np.maximum(d[n], 0.0)[:, j0:j1, :]).sum(axis=1) * dy for n in names)
        assert side["slab_path_max_kg_m2"][group] == pytest.approx(float(slab.max()), rel=1e-4), group
    # The cut: the mixing ratios on the core's row, g/kg.
    for name in ("qc", "qr", "qi", "qs", "qg"):
        assert side["cut_max_g_kg"][name] == pytest.approx(float(d[name][:, j, :].max()) * 1000.0, rel=1e-5)
    # The water paths: rho q dz over every level, kg/m2 (1 kg/m2 = 1 mm of water).
    for name in ("qc", "qr", "qi", "qs", "qg"):
        path = (rho * np.maximum(d[name], 0.0) * dz).sum(axis=0)
        assert paths["path_max_kg_m2"][name] == pytest.approx(float(path.max()), rel=1e-4), name
    total = sum((rho * np.maximum(d[n], 0.0) * dz).sum(axis=0) for n in ("qc", "qr", "qi", "qs", "qg"))
    assert paths["path_max_kg_m2"]["all"] == pytest.approx(float(total.max()), rel=1e-4)
    # This run's dense ice is hail (ihail = 1), and it says so; a thunderstorm is drawn at true vertical scale.
    assert side["lights"] == ["cloud water", "rain", "cloud ice and snow", "hail"] and paths["dense_ice"] == "hail"
    assert side["vertical_stretch"] == pytest.approx(1.0, abs=0.01)
    assert side["ranges"]["slab_kg_m2"] == {g: list(SV.SIDE_RANGES[g]) for g in groups}
    assert paths["range_kg_m2"] == [0.01, 100.0] and paths["scale"] == "logarithmic"


@NEEDS_MATPLOTLIB
def test_the_three_looks_say_what_they_are_and_pass_the_lint(tmp_path, pictures):
    run = run_directory(tmp_path, times=(1500.0,))
    _render(run, only=NEW)
    by = {name: ov for name, ov in pictures}
    assert set(by) == set(NEW)
    for name, ov in pictures:
        assert ov.check() == [] and ov.footer_cut == [], (name, ov.check(), ov.footer_cut)
        text = "\n".join(ov.texts)
        assert "tornad" not in text.lower()
        first = next(t for t in ov.texts if t.startswith(S.SIMULATED))
        assert "Morrison with hail" in first and "dx 1 km" in first
    side = "\n".join(by["side-view"].texts)
    for words in ("seen from the south", "summed through the 20 km slab", "mixing ratio", "true vertical scale",
                  "the lights add where species mix", "log scale", "the strongest w at 5 km"):
        assert words in side, words
    paths = "\n".join(by["water-paths"].texts)
    for words in ("ρ q dz summed over every level", "1 kg m⁻² is 1 mm of water", "0.01–100 kg m⁻² fixed",
                  "logarithmic", "not satellite data", "the side view's slab"):
        assert words in paths, words
    axis = " ".join(by["rotation-axis"].texts)          # its notes are wrapped: read them as one line
    for words in ("a picture tool's method, not CM1's", "around a circle of 3 km radius",
                  "the most 2–5 km updraft helicity", "heights drawn", "× 10⁴ m²/s", "spin: ",
                  "No two of its centres lie further apart sideways than 1.5 km per km of the height between them, "
                  "plus 1.5 km (1.5 grid spacings).", "3 km in radius"):
        assert words in axis, words
    for words in ("3 km circle", "2 km circle", "at most 1.5 km sideways"):
        assert words not in axis, words


def _kessler_run(tmp_path: Path, *, empty: bool = False) -> Path:
    """A run whose scheme carries cloud water and rain only (Kessler); ``empty``: no condensate anywhere."""

    run = tmp_path / "kessler"
    frames = run / "frames"
    frames.mkdir(parents=True)
    shutil.copy(NAMELIST_24, run / "namelist.input")
    spec = {"names": ["qv", "qc", "qr"], "cm1_index_ranges": {"nqv": 1, "nql1": 2, "nql2": 3},
            "iice": 0, "idm": 0, "numq": 3, "component_id": "kessler", "cm1_ptype": 1}
    carriers, frame = storm(1500.0, move=(0.0, 0.0), swath="domain")
    q = carriers["q3d"][:3].copy()
    if empty:
        q[1:] = 0.0
    carriers["q3d"] = q
    carriers.pop("nc3d")
    publish(frames, 0, carriers, species=spec, microphysics="kessler", ihail=None,
            meanings={"q3d": "current qv,qc,qr mixing ratios"}, **frame)
    end_event(run / "events.jsonl", frames=1)
    return run


@NEEDS_MATPLOTLIB
def test_a_scheme_with_cloud_water_and_rain_only_draws_two_lights_and_three_maps(tmp_path):
    run = _kessler_run(tmp_path)
    assert _render(run, only=NEW[:2]).problems == []
    products = {p["id"]: p for p in _index(run)["products"]}
    side = products["side-view"]["frames"][0]["facts"]
    paths = products["water-paths"]["frames"][0]["facts"]
    assert side["species"] == ["qc", "qr"] and side["lights"] == ["cloud water", "rain"]
    assert set(paths["path_max_kg_m2"]) == {"qc", "qr", "all"}


@NEEDS_MATPLOTLIB
def test_air_with_no_cloud_rain_or_ice_is_not_drawn_and_says_why(tmp_path):
    run = _kessler_run(tmp_path, empty=True)
    _render(run, only=NEW[:2])
    products = {p["id"]: p for p in _index(run)["products"]}
    for look in NEW[:2]:
        assert products[look]["frames"] == []
        assert [row["reason"] for row in products[look]["not_drawn"]] == ["No cloud, rain or ice in the air yet."]


def test_the_groups_and_ranges_are_stated_and_logarithmic():
    assert SV.log_position(np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1e4]), SV.PATH_RANGE).tolist() == \
        pytest.approx([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    for ranges in (SV.SIDE_RANGES, SV.CUT_RANGES):
        assert set(ranges) == {"water", "rain", "ice", "dense"}
        for lo, hi in ranges.values():
            assert hi / lo == pytest.approx(1000.0)        # three tenfold steps each
    assert SV.GROUP == {"qc": "water", "qr": "rain", "qi": "ice", "qs": "ice", "qg": "dense"}
    # the species colours rise in lightness, as every ramp the pictures use
    for key in SV.LIGHTS:
        assert S.RAMPS[f"species_{key}"].check() == [], key


def _hurricane_run(tmp_path: Path) -> Path:
    text = NAMELIST_24.read_text(encoding="ascii")
    swirl, count = re.subn(r"(?m)^(\s*iinit\s*=\s*)\d+", r"\g<1>7", text)
    assert count == 1
    namelist = tmp_path / "hurricane.namelist.input"
    namelist.write_text(swirl, encoding="ascii")
    return run_directory(tmp_path, namelist=namelist, name="swirl", move=(0.0, 0.0), swath="domain")


@NEEDS_MATPLOTLIB
def test_a_hurricane_is_cut_through_its_centre_and_its_axis_needs_a_swirl(tmp_path):
    run = _hurricane_run(tmp_path)
    assert _render(run, only=NEW).problems == []
    index = _index(run)
    assert index["run"]["storm_kind"] == "hurricane"
    products = {p["id"]: p for p in index["products"]}
    side = products["side-view"]["frames"][-1]["facts"]
    # the core is the lowest pressure at the lowest level
    d = np.load(STATE)
    j, i = np.unravel_index(int(np.argmin(d["prs"][0])), d["prs"][0].shape)
    assert side["core_ij"] == [int(i) + 1, int(j) + 1]
    assert side["core"] == "its lowest pressure at the lowest level"
    # this 24 km test area is narrower than the true-scale window, so it is drawn at true scale
    assert side["vertical_stretch"] == pytest.approx(1.0, abs=0.01)
    # CM1's supercell under a hurricane namelist has no swirl at the lowest level: the axis says so, plainly
    # (one wording for all three frames, so the record lists it once with their times)
    assert products["rotation-axis"]["frames"] == []
    assert products["rotation-axis"]["not_drawn"] == [{
        "reason": "No rotating storm in this frame: the circulation around a circle of 3 km radius at the lowest "
                  "level stays below 5.7 × 10⁴ m²/s, the least the axis is drawn from.",
        "times_s": [900.0, 1200.0, 1500.0]}]


# ---------------------------------------------------------------- the rotation axis on a vortex with a known centre

GRID = {"nx": 40, "ny": 40, "nz": 24, "dx_m": 1000.0, "dy_m": 1000.0, "dz_m": 500.0, "ztop_m": 12000.0}
LEAN = (0.6, -0.3)          # km east and north per km of height
FOOT = (16.0, 20.0)          # km: the centre at the ground


def _centre_km(z_m: float, lean: tuple[float, float] = LEAN, foot: tuple[float, float] = FOOT) -> tuple[float, float]:
    return foot[0] + lean[0] * z_m / 1000.0, foot[1] + lean[1] * z_m / 1000.0


def _vortex_run(tmp_path: Path, *, gamma: float = 2.0e5, rise: float = 15.0, uh: float | None = None,
                name: str = "vortex", lean: tuple[float, float] = LEAN, foot: tuple[float, float] = FOOT,
                weak_above: float | None = None) -> Path:
    """A vortex with a smooth 2 km core (circulation ``gamma`` m2/s, a tenth of it above ``weak_above``) whose centre
    leans with height, around a column rising at ``rise`` m/s; the winds on CM1's staggered grid, so the vorticity is
    the frame's own dv/dx - du/dy."""

    nx, ny, nz = GRID["nx"], GRID["ny"], GRID["nz"]
    zf = np.arange(nz + 1) * 500.0
    zh = 0.5 * (zf[1:] + zf[:-1])
    u = np.zeros((nz, ny, nx + 1), np.float32)
    v = np.zeros((nz, ny + 1, nx), np.float32)
    w = np.zeros((nz + 1, ny, nx), np.float32)
    rc = 2000.0

    def swirl(dx_m, dy_m, z):
        r2 = np.maximum(dx_m ** 2 + dy_m ** 2, 1.0)
        g = gamma * (0.1 if weak_above is not None and z > weak_above else 1.0)
        return g / (2.0 * math.pi) * (1.0 - np.exp(-r2 / rc ** 2)) / r2

    for k in range(nz):
        cx, cy = (c * 1000.0 for c in _centre_km(zh[k], lean, foot))
        xu, yu = np.meshgrid(np.arange(nx + 1) * 1000.0, (np.arange(ny) + 0.5) * 1000.0)
        u[k] = -swirl(xu - cx, yu - cy, zh[k]) * (yu - cy)
        xv, yv = np.meshgrid((np.arange(nx) + 0.5) * 1000.0, np.arange(ny + 1) * 1000.0)
        v[k] = swirl(xv - cx, yv - cy, zh[k]) * (xv - cx)
    for k in range(nz + 1):
        cx, cy = (c * 1000.0 for c in _centre_km(min(zf[k], zh[-1]), lean, foot))
        xs, ys = np.meshgrid((np.arange(nx) + 0.5) * 1000.0, (np.arange(ny) + 0.5) * 1000.0)
        w[k] = rise * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / 3000.0 ** 2) * math.sin(math.pi * zf[k] / zf[-1])
    rho = np.broadcast_to((1.2 * np.exp(-zh / 9000.0)).astype(np.float32)[:, None, None], (nz, ny, nx)).copy()
    q = np.zeros((len(MASS), nz, ny, nx), np.float32)
    q[1] = np.where(w[:-1] > 1.0, 1.0e-3, 0.0)
    carriers = {"rho": rho, "q3d": q, "th3d": np.zeros((nz, ny, nx), np.float32), "u3d": u, "v3d": v, "w3d": w,
                "zf": np.broadcast_to(zf.astype(np.float32)[:, None, None], (nz + 1, ny, nx)).copy(),
                "prs": np.broadcast_to((1.0e5 * np.exp(-zh / 8000.0)).astype(np.float32)[:, None, None],
                                       (nz, ny, nx)).copy()}
    if uh is not None:
        carriers["uh"] = np.full((ny, nx), uh, np.float32)
        cx, cy = (int(c) for c in FOOT)
        carriers["uh"][cy, cx] = uh * 10.0
    run = tmp_path / name
    frames = run / "frames"
    frames.mkdir(parents=True)
    shutil.copy(NAMELIST_24, run / "namelist.input")
    publish(frames, 0, carriers, time_s=1800.0, step=300, grid=GRID, dt_s=6.0,
            meanings={"q3d": "current " + ",".join(MASS) + " mixing ratios"},
            stagger={"u3d": "u", "v3d": "v", "w3d": "w", "zf": "w"})
    end_event(run / "events.jsonl", frames=1)
    return run


def test_the_axis_follows_a_leaning_vortex_to_within_a_grid_cell(tmp_path):
    L = _looks(_vortex_run(tmp_path))
    try:
        A = AX.Axis(L)
        assert A.why is None
        assert A.r == 3000.0 and A.start_words == "the most 2–5 km updraft helicity"
        assert A.zeta_source == "dv/dx − du/dy"
        zs = [float(L.f.zh[k]) for k in A.ks]
        assert zs[0] == 250.0 and zs[-1] == 9750.0          # every level from the lowest to 10 km
        for k in A.ks:
            x, y = A.xy_m(k)
            cx, cy = _centre_km(float(L.f.zh[k]))
            assert math.hypot(x / 1000.0 - cx, y / 1000.0 - cy) < 0.5, (float(L.f.zh[k]), x, y, cx, cy)
            # the circle's circulation is the vortex's inside 3 km, give or take the cells the circle holds
            assert A.centres[k][2] == pytest.approx(2.0e5 * (1.0 - math.exp(-(3.0 / 2.0) ** 2)), rel=0.12)
    finally:
        L.f.close()


@NEEDS_MATPLOTLIB
def test_the_axis_picture_reports_the_lean_and_circulation(tmp_path, pictures):
    run = _vortex_run(tmp_path)
    assert _render(run, only=("rotation-axis",)).problems == []
    facts = _index(run)["products"][0]["frames"][0]["facts"]
    # from 250 m to 4.75 km the centre moves 4.5 km x (0.6, -0.3) km per km: 3.0 km to the south-east
    assert facts["lean_levels_m"] == [250.0, 4750.0]
    assert facts["lean_km"] == pytest.approx(4.5 * math.hypot(*LEAN), abs=0.5)
    assert facts["lean_toward"] == "south-east"
    assert facts["spin_range_per_s"] == 0.01 and facts["radius_m"] == 3000.0
    # the level nearest 1 km is 750 m, only 500 m over the lowest, so the slices are the lowest and 5 km
    assert facts["slices_m"] == [250.0, 4750.0]
    ov = pictures[0][1]
    assert ov.check() == [] and ov.footer_cut == []
    assert "heights drawn 1.8 × the horizontal scale" in "\n".join(ov.texts)


def test_air_that_does_not_turn_or_does_not_rise_gets_no_axis_and_the_reason(tmp_path):
    still = _looks(_vortex_run(tmp_path, gamma=0.0, name="still"))
    weak = _looks(_vortex_run(tmp_path, gamma=2.0e4, name="weak"))
    sinking = _looks(_vortex_run(tmp_path, rise=-15.0, uh=50.0, name="sinking"))
    try:
        assert AX.Axis(still).why == "No rotating updraft in this frame: the updraft helicity is zero everywhere."
        # the weak vortex holds 1.8 x 10^4 m2/s inside 3 km, below the 2 pi r x 3 m/s the axis needs
        A = AX.Axis(weak)
        assert 1.5e4 <= A.circulation_ref <= 1.8e4 and A.min_circulation == pytest.approx(2 * math.pi * 3000 * 3)
        assert A.why == ("No rotating updraft in this frame: the circulation around a circle of 3 km radius at "
                         "2.75 km stays below 5.7 × 10⁴ m²/s, the least the axis is drawn from.")
        assert AX.Axis(sinking).why == ("No rotating updraft in this frame: the air in the most rotating circle at "
                                        "2.75 km is not rising.")
        with pytest.raises(P.NotDrawn):
            AX.product_rotation_axis(still, tmp_path / "never.png")
        assert not (tmp_path / "never.png").exists()
    finally:
        for L in (still, weak, sinking):
            L.f.close()


def _keeps_the_rule(A: AX.Axis) -> float:
    """The most any two of the axis's centres break its rule by (m; zero or less: they keep it)."""

    pts = A.points()
    worst = -np.inf
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            apart = math.hypot(pts[a, 0] - pts[b, 0], pts[a, 1] - pts[b, 1])
            worst = max(worst, apart - (AX.TILT_MAX * abs(pts[a, 2] - pts[b, 2]) + A.slack))
    return worst


@NEEDS_MATPLOTLIB
def test_the_axis_keeps_its_rule_on_a_vortex_that_leans_too_far_and_says_where_it_stops(tmp_path, pictures):
    """A vortex leaning 3 km per km of height, twice what the rule allows: the axis is drawn only as far as the rule
    holds between every two of its heights, and the picture says where it stops and why."""

    run = _vortex_run(tmp_path, lean=(3.0, 0.0), foot=(10.0, 20.0), name="steep")
    L = _looks(run)
    try:
        A = AX.Axis(L)
        assert A.why is None
        assert A.slack == 1500.0                     # 1.5 grid spacings of 1 km
        assert _keeps_the_rule(A) <= 1e-6
        zs = [float(L.f.zh[k]) for k in A.ks]
        assert zs[0] > 250.0 and zs[-1] < 9750.0     # it reaches neither the lowest level nor 10 km
        assert A.stops == {"below": "fast", "above": "fast"}
        for k in A.ks:                               # what it does draw is on the vortex
            x, y = A.xy_m(k)
            cx, cy = _centre_km(float(L.f.zh[k]), (3.0, 0.0), (10.0, 20.0))
            assert math.hypot(x / 1000.0 - cx, y / 1000.0 - cy) < 0.6
    finally:
        L.f.close()
    assert _render(run, only=("rotation-axis",)).problems == []
    facts = _index(run)["products"][0]["frames"][0]["facts"]
    assert facts["rule"] == {"km_per_km": 1.5, "plus_m": 1500.0, "between": "every two heights"}
    assert facts["stops"] == {"below": AX.STOPS["fast"], "above": AX.STOPS["fast"]}
    text = " ".join(pictures[0][1].texts)
    for side, z in (("below", zs[0]), ("above", zs[-1])):
        assert f"{side} {S.fmt_height(z)} the centre moves sideways faster than the rule allows, so the axis stops " \
               "there" in text


def test_the_axis_stops_where_the_turning_is_weaker_than_it_needs(tmp_path):
    """Above 6 km the vortex holds a tenth of its circulation, below 2 pi r x 3 m/s: the axis stops there and says
    so, instead of wandering on through weak turning."""

    L = _looks(_vortex_run(tmp_path, weak_above=6000.0, name="shallow"))
    try:
        A = AX.Axis(L)
        assert A.why is None
        assert float(L.f.zh[A.ks[0]]) == 250.0 and float(L.f.zh[A.ks[-1]]) == 5750.0
        assert A.stops == {"below": None, "above": "weak"}
        assert all(A.centres[k][2] >= A.min_circulation for k in A.ks)
    finally:
        L.f.close()


def _ring_hurricane_run(tmp_path: Path, *, lean_km_per_km: float = 0.4) -> tuple[Path, dict]:
    """A hurricane on a 4 km grid whose spin is a hollow ring: winds rising as r^2 to 50 m/s at 25 km (the radius
    of maximum wind) and falling as r^-1/2 beyond, so the vorticity is weakest at the centre and strongest at the
    eyewall; the pressure lowest at the centre (5 hPa deep over 60 km); the centre leaning east with height."""

    grid = {"nx": 48, "ny": 48, "nz": 24, "dx_m": 4000.0, "dy_m": 4000.0, "dz_m": 500.0, "ztop_m": 12000.0}
    nx, ny, nz, d = grid["nx"], grid["ny"], grid["nz"], grid["dx_m"]
    zf = np.arange(nz + 1) * 500.0
    zh = 0.5 * (zf[1:] + zf[:-1])
    rmw, vmax = 25000.0, 50.0

    def centre(z: float) -> tuple[float, float]:
        return 96000.0 + lean_km_per_km * z, 96000.0

    def speed(r):
        r = np.maximum(r, 1.0)
        return np.where(r <= rmw, vmax * (r / rmw) ** 2, vmax * (rmw / r) ** 0.5)

    u = np.zeros((nz, ny, nx + 1), np.float32)
    v = np.zeros((nz, ny + 1, nx), np.float32)
    prs = np.zeros((nz, ny, nx), np.float32)
    for k in range(nz):
        cx, cy = centre(zh[k])
        xu, yu = np.meshgrid(np.arange(nx + 1) * d, (np.arange(ny) + 0.5) * d)
        r = np.hypot(xu - cx, yu - cy)
        u[k] = -speed(r) * (yu - cy) / np.maximum(r, 1.0)
        xv, yv = np.meshgrid((np.arange(nx) + 0.5) * d, np.arange(ny + 1) * d)
        r = np.hypot(xv - cx, yv - cy)
        v[k] = speed(r) * (xv - cx) / np.maximum(r, 1.0)
        xs, ys = np.meshgrid((np.arange(nx) + 0.5) * d, (np.arange(ny) + 0.5) * d)
        r2 = (xs - cx) ** 2 + (ys - cy) ** 2
        prs[k] = 1.0e5 * np.exp(-zh[k] / 8000.0) - 500.0 * np.exp(-r2 / 60000.0 ** 2)
    rho = np.broadcast_to((1.2 * np.exp(-zh / 9000.0)).astype(np.float32)[:, None, None], (nz, ny, nx)).copy()
    carriers = {"rho": rho, "q3d": np.zeros((len(MASS), nz, ny, nx), np.float32),
                "th3d": np.zeros((nz, ny, nx), np.float32), "u3d": u, "v3d": v,
                "w3d": np.zeros((nz + 1, ny, nx), np.float32), "prs": prs,
                "zf": np.broadcast_to(zf.astype(np.float32)[:, None, None], (nz + 1, ny, nx)).copy()}
    text = NAMELIST_24.read_text(encoding="ascii")
    swirl, count = re.subn(r"(?m)^(\s*iinit\s*=\s*)\d+", r"\g<1>7", text)
    assert count == 1
    run = tmp_path / "ring"
    frames = run / "frames"
    frames.mkdir(parents=True)
    (run / "namelist.input").write_text(swirl, encoding="ascii")
    publish(frames, 0, carriers, time_s=3 * 86400.0, step=7200, grid=grid, dt_s=36.0,
            meanings={"q3d": "current " + ",".join(MASS) + " mixing ratios"},
            stagger={"u3d": "u", "v3d": "v", "w3d": "w", "zf": "w"})
    end_event(run / "events.jsonl", frames=1)
    return run, {"centre": centre, "rmw": rmw}


def test_a_hurricanes_axis_is_its_centre_not_its_eyewall(tmp_path):
    """The eye (radius of maximum wind 25 km) is wider than the 12 km circle, so the circle with the most
    circulation sits in the eyewall: a hurricane's axis is found from the lowest mean pressure in the circle, and it
    stays within one grid spacing of the true centre at every height from the lowest level to 10 km."""

    run, known = _ring_hurricane_run(tmp_path)
    L = _looks(run)
    try:
        assert L.hurricane
        A = AX.Axis(L)
        assert A.why is None and A.r == 12000.0 and A.slack == 6000.0
        assert A.method_words == "centre: lowest mean pressure in a circle of 12 km radius"
        assert A.start_words == "the storm's centre (its lowest pressure at the lowest level)"
        assert [float(L.f.zh[k]) for k in (A.ks[0], A.ks[-1])] == [250.0, 9750.0]
        assert A.stops == {"below": None, "above": None}
        for k in A.ks:
            x, y = A.xy_m(k)
            cx, cy = known["centre"](float(L.f.zh[k]))
            assert math.hypot(x - cx, y - cy) <= L.f.dx, (float(L.f.zh[k]), x, y, cx, cy)
        assert _keeps_the_rule(A) <= 1e-6
        # the test vortex is hollow for the circle: the most circulation is out in the eyewall, not at the centre
        g = AX.disk_sum(L.f.zeta_level(0)[0], A.kernel)
        j, i = np.unravel_index(int(np.argmax(g)), g.shape)
        cx, cy = known["centre"](float(L.f.zh[0]))
        assert math.hypot((i + 0.5) * L.f.dx - cx, (j + 0.5) * L.f.dy - cy) >= 0.5 * known["rmw"]
    finally:
        L.f.close()


def test_the_end_labels_seen_from_above_clear_the_line_and_each_other():
    """An upright axis puts its foot under its top: the two heights must not print over each other or the line."""

    ov = S.Overlay()
    box = P.Box(1380, 214, 440, 0.0, 0.0, 20000.0, h=440)
    boxes: list[tuple[int, int, int, int]] = []
    original = ov.text

    def spy(xy, text, role="label", **kw):
        got = original(xy, text, role, **kw)
        boxes.append(got)
        return got

    ov.text = spy
    upright = np.array([(10000.0, 10000.0, z, 1.0) for z in (250.0, 2000.0, 5000.0, 9750.0)])
    leaning = np.array([(10000.0 - 0.3 * z, 10000.0 + 0.2 * z, z, 1.0) for z in np.arange(250.0, 10000.0, 500.0)])
    for pts in (upright, leaning):
        boxes.clear()
        AX._end_labels(ov, box, pts, ("250 m", "9.75 km"))
        a, b = boxes
        assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]), (a, b)
        for x0, y0, x1, y1 in (a, b):
            assert box.x <= x0 and x1 <= box.x + box.w and box.y <= y0 and y1 <= box.y + box.h
            for p in pts:
                px, py = box.to_px(p[0], p[1])
                assert not (x0 <= px <= x1 and y0 <= py <= y1), (p, (x0, y0, x1, y1))


def test_a_trace_is_printed_as_a_trace():
    assert SV.fmt_amount(0.00000008) == "under 0.001" and SV.fmt_amount(0.000003) == "under 0.001"
    assert SV.fmt_amount(0.0) == "0" and SV.fmt_amount(0.05) == "0.05" and SV.fmt_amount(12.47) == "12"


@NEEDS_MATPLOTLIB
def test_a_hurricanes_water_paths_give_the_most_in_the_window_they_show(tmp_path, monkeypatch):
    run = _hurricane_run(tmp_path)
    monkeypatch.setattr(SV, "HURRICANE_WINDOW", 10000.0)        # a 10 km window in the 24 km test area
    assert _render(run, only=("water-paths",)).problems == []
    facts = _index(run)["products"][0]["frames"][-1]["facts"]
    x0, y0, w, h = facts["region_km"]
    assert (w, h) == (10.0, 10.0)
    L = _looks(run)
    try:
        ox, oy = (v / 1000.0 for v in L.facts.origin_m(L.f.nx, L.f.ny))
    finally:
        L.f.close()
    d = np.load(STATE)
    # every 1 km column the window covers, wholly or in part (it is centred on a column's middle)
    i0, i1 = math.floor(x0 - ox + 1e-6), math.ceil(x0 - ox + w - 1e-6)
    j0, j1 = math.floor(y0 - oy + 1e-6), math.ceil(y0 - oy + h - 1e-6)
    beyond = []
    for name in ("qc", "qr", "qi", "qs", "qg"):
        path = (d["rho"].astype(np.float64) * np.maximum(d[name], 0.0) * 500.0).sum(axis=0)
        assert facts["path_max_kg_m2"][name] == pytest.approx(float(path[j0:j1, i0:i1].max()), rel=1e-4), name
        beyond.append(float(path.max()) > 1.01 * facts["path_max_kg_m2"][name])
    assert any(beyond)            # so the test tells the window's most from the whole area's


def test_the_axis_ranges_and_circles_follow_the_grid():
    assert [AX.axis_spin_range(dx) for dx in (1000.0, 500.0, 250.0, 100.0, 67.0)] == [0.01, 0.02, 0.05, 0.1, 0.2]
    assert AX.axis_spin_range(4000.0, hurricane=True) == AX.axis_spin_range(8000.0, hurricane=True) == 0.01
    assert [AX.circle_radius(dx) for dx in (67.0, 100.0, 250.0, 1000.0, 8000.0)] == \
        [2000.0, 2000.0, 2000.0, 3000.0, 24000.0]
    kernel = AX._disk(3.0)
    a = np.zeros((9, 9), np.float32)
    a[4, 4] = 1.0
    assert AX.disk_sum(a, kernel).sum() == pytest.approx(kernel.sum())
    assert AX.disk_sum(np.ones((9, 9), np.float32), kernel)[4, 4] == pytest.approx(kernel.sum())


# ---------------------------------------------------------------- the looks drawn before stay as they were

@NEEDS_MATPLOTLIB
def test_the_looks_drawn_before_are_byte_identical_beside_the_new_ones(tmp_path):
    alone = run_directory(tmp_path / "a")
    beside = run_directory(tmp_path / "b")
    _render(alone, only=BEFORE)
    _render(beside)
    for look in BEFORE:
        for path in sorted((alone / "quicklook" / look).glob("*.png")):
            twin = beside / "quicklook" / look / path.name
            assert path.read_bytes() == twin.read_bytes(), (look, path.name)


# ---------------------------------------------------------------- the Results screen

def test_results_offers_the_new_looks_for_a_run_drawn_before_them(tmp_path):
    run = tmp_path / "storm"
    # a run drawn before them still has its saved frames (one whose frames were deleted is offered no pass)
    (run / "frames" / "frame-000000").mkdir(parents=True)
    frames = [{"index": n, "time_s": t} for n, t in enumerate((900.0, 1200.0))]
    index = {"run": {"dx_m": 1000.0}, "frames": frames,
             "products": [{"id": pid, "frames": [{"file": f"{pid}/t000900.png"}]} for pid in BEFORE]}
    rows = {row["id"]: row for row in GL.more_looks("storm", run, index, {"state": "finished"}, films_now={})}
    older = GL.load_copy()["why_not"]["older_run"]["text"]
    for look in NEW:
        assert rows[look]["available"] is False
        assert rows[look]["why_not"] == older and rows[look]["action"] == "all", look
        assert "1 km" in rows[look]["note"]           # the run's own grid spacing in its resolution line
    assert set(NEW) <= set(GL.NEW_LOOKS) and set(NEW) <= set(GL.FILM_LOOKS)
    # a hurricane run keeps all three (none is made for one thunderstorm), each with its own words
    hurricane = dict(index, run={"dx_m": 8000.0, "storm_kind": "hurricane"})
    rows = {row["id"]: row for row in GL.more_looks("h", run, hurricane, {"state": "finished"}, films_now={})}
    copy = {row["id"]: row for row in GL.load_copy()["more_looks"]}
    for look in NEW:
        assert look in rows and rows[look]["text"] == copy[look]["text_hurricane"]
