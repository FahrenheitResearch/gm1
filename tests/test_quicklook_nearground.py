"""Flow near the ground: the storm-relative flow, the streamwise spin and the cold-pool depth.

CPU only.  The frames are a small 100 m storm made here with known numbers (``fine_run``): levels 20 m apart near
the ground growing to 12 km, a base wind that turns and strengthens with height, one round updraft that moves across
the saved times at a known ground-relative speed, and a round cold pool whose depth is known.  So each check below
compares with a number worked out by hand, not with the code's own output:

1. the storm's motion is its own track when it has three saved times over ten minutes, else the Bunkers estimate
   (the run's record, or the base-state wind), and every picture says which; the track follows one updraft back,
   so a second storm 25 km away that is the strongest at some 15-minute saves is not mixed in;
2. the cold-pool depth is the height where theta rises through 1 K below the median of the area, linear between
   levels, with cold air above a warmer gap left out; the area shown holds all the cold air so far, in steps, and
   never shrinks; cold air at the domain's edge is said;
3. the horizontal spin of a sheared wind is its shear; the spin look needs levels 100 m apart near the ground;
4. the three pictures are drawn, linted and fit their footers on a fine grid, and the longest words stay clear of
   the colour bars; a coarser grid, coarse levels, a run one column wide and a low model top get their plain
   reasons, in the index, on the Results screen and when a film of the look is asked for;
5. the looks drawn before these existed stay byte-identical when these are drawn beside them.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pytest

NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")

from cm1out_support import MORRISON_SPECIES, end_event, publish  # noqa: E402
from quicklook_support import run_directory  # noqa: E402

from cm1core.gui import looks as GL  # noqa: E402
from cm1core.quicklook import nearground as NG  # noqa: E402
from cm1core.quicklook import products as P  # noqa: E402
from cm1core.quicklook import render as RD  # noqa: E402
from cm1core.quicklook import style as S  # noqa: E402
from cm1core.quicklook.frame import QuickFrame  # noqa: E402

NX = NY = 72
DX = 100.0
#: Levels: 20 m layers up to 200 m, then 19 layers growing by 30 % each to 12 km.
GROWTH = 1.3
_FIRST = (12000.0 - 200.0) * (GROWTH - 1.0) / (GROWTH ** 19 - 1.0)
ZF = np.concatenate([np.arange(0.0, 200.0 + 1e-6, 20.0),
                     200.0 + np.cumsum(_FIRST * GROWTH ** np.arange(19))]).astype(np.float32)
NZ = len(ZF) - 1
ZH = 0.5 * (ZF[1:] + ZF[:-1])
#: The storm's ground-relative motion, the domain's, and where the updraft starts (domain metres).
STORM = (12.0, 7.0)
MOVE = (11.0, 6.0)
START = (2050.0, 2050.0)
#: The cold pool: a disk this wide around this point; theta' = -3 K at the ground rising linearly to 0 at 600 m,
#: so 1 K below its surroundings at exactly 400 m; and a colder layer aloft (1000-1200 m) over a warmer gap.
POOL = (5050.0, 3550.0, 1500.0)
WARM = 0.3          # K added everywhere: the depth is measured from the area's median, not from the base state


def base_wind(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground-relative base wind: u from 5 to 20 m/s over 6 km, v from 0 to 10 m/s over 3 km."""

    z = np.asarray(z, dtype=np.float64)
    return 5.0 + 15.0 * np.clip(z / 6000.0, 0.0, 1.0), 10.0 * np.clip(z / 3000.0, 0.0, 1.0)


def fine_carriers(t: float, calm: bool = False, pool=POOL) -> dict[str, np.ndarray]:
    pool = pool(t) if callable(pool) else pool
    xc = START[0] + (STORM[0] - MOVE[0]) * t
    yc = START[1] + (STORM[1] - MOVE[1]) * t
    x = (np.arange(NX) + 0.5) * DX
    y = (np.arange(NY) + 0.5) * DX
    xx, yy = np.meshgrid(x, y)
    ug, vg = base_wind(ZH)
    u = np.empty((NZ, NY, NX + 1), np.float32)
    v = np.empty((NZ, NY + 1, NX), np.float32)
    u[:] = (ug - MOVE[0]).astype(np.float32)[:, None, None]
    v[:] = (vg - MOVE[1]).astype(np.float32)[:, None, None]
    bump = np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / 900.0 ** 2)
    w = (0.0 if calm else 40.0) * np.sin(np.pi * np.clip(ZF / 12000.0, 0.0, 1.0))[:, None, None] * bump[None]
    w = w.astype(np.float32)
    th = np.zeros((NZ, NY, NX), np.float32) + np.float32(WARM)
    disk = np.hypot(xx - pool[0], yy - pool[1]) <= pool[2]
    profile = np.where(ZH < 600.0, -3.0 + 3.0 * ZH / 600.0, 0.0) + np.where((ZH > 1000.0) & (ZH < 1200.0), -2.0, 0.0)
    th += (profile.astype(np.float32)[:, None, None] * disk[None]).astype(np.float32)
    return {
        "u3d": u, "v3d": v, "w3d": w, "th3d": th,
        "q3d": np.zeros((6, NZ, NY, NX), np.float32), "nc3d": np.zeros((4, NZ, NY, NX), np.float32),
        "zf": np.broadcast_to(ZF.reshape(-1, 1, 1), (NZ + 1, NY, NX)).copy(),
        "u0": (ug - MOVE[0]).astype(np.float32), "v0": (vg - MOVE[1]).astype(np.float32),
        "th0": (300.0 + 0.004 * ZH).astype(np.float32),
    }


def fine_run(tmp_path: Path, times=(0.0, 300.0, 600.0, 900.0), *, name: str = "fine", receipt: dict | None = None,
             zf: np.ndarray | None = None, drop: tuple[str, ...] = (), calm: bool = False, pool=POOL) -> Path:
    """A cartesian output directory of the small 100 m storm (frames/, events.jsonl).  ``pool``: the cold pool's
    (x, y, radius), or a function of the time giving it."""

    run = tmp_path / name
    frames = run / "frames"
    frames.mkdir(parents=True)
    grid = {"nx": NX, "ny": NY, "nz": NZ, "dx_m": DX, "dy_m": DX, "dz_m": 100.0, "ztop_m": float(ZF[-1])}
    for index, t in enumerate(times):
        carriers = {k: a for k, a in fine_carriers(t, calm, pool).items() if k not in drop}
        if zf is not None:
            carriers["zf"] = np.broadcast_to(zf.reshape(-1, 1, 1), (len(zf), NY, NX)).copy()
        path = publish(frames, index, carriers, time_s=t, step=int(t), grid=grid, dt_s=1.0, umove=MOVE[0],
                       vmove=MOVE[1], domainloc=(MOVE[0] * t, MOVE[1] * t), species=MORRISON_SPECIES)
        table = path / "history-field-table.tsv"
        rows = []
        for line in table.read_text(encoding="ascii").splitlines():
            cols = line.split("\t")
            if cols[0] == "q3d":
                cols[1] = "current qv,qc,qr,qi,qs,qg mixing ratios"
            elif cols[0] == "nc3d":
                cols[1] = "current nci,ncs,ncr,ncg number concentrations"
            rows.append("\t".join(cols))
        table.write_text("\n".join(rows) + "\n", encoding="ascii")
        doc = json.loads((path / "history-publication.json").read_text(encoding="utf-8"))
        doc["config"]["dense_ice"] = {"ihail": 1, "category": "hail"}
        (path / "history-publication.json").write_text(json.dumps(doc), encoding="utf-8")
    end_event(run / "events.jsonl", frames=len(times))
    if receipt is not None:
        (run / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return run


def _render(run: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return RD.render(run, **kwargs)


def _index(run: Path) -> dict:
    return json.loads((run / "quicklook" / "index.json").read_text(encoding="utf-8"))


def _product(run: Path, pid: str) -> dict:
    return next(p for p in _index(run)["products"] if p["id"] == pid)


# ---------------------------------------------------------------- the lists the server keeps without numpy

def test_the_servers_lists_are_the_pictures():
    assert GL.NEAR_GROUND == P.NEAR_GROUND == NG.IDS
    assert GL.NEAR_GROUND_DX_M == NG.FINEST_DX_M
    assert set(P.NEAR_GROUND) <= set(P.THUNDERSTORM_ONLY) and set(P.NEAR_GROUND) <= set(GL.THUNDERSTORM_ONLY)
    assert all(len(P.TITLES[pid].split()) <= 6 for pid in P.NEAR_GROUND)


# ---------------------------------------------------------------- the storm's motion

@NEEDS_MATPLOTLIB
def test_the_storm_motion_is_the_runs_own_track(tmp_path):
    run = fine_run(tmp_path)
    summary = _render(run, only=("storm-relative-flow",))
    assert summary.problems == []
    frames = _product(run, "storm-relative-flow")["frames"]
    assert [f["time_s"] for f in frames] == [0.0, 300.0, 600.0, 900.0]
    early, last = frames[1]["facts"]["storm_motion"], frames[-1]["facts"]["storm_motion"]
    # two saved times over 5 minutes are not a track: the sounding's estimate stands in, and says why
    assert early["source"] == "bunkers-base-state" and "2 saved times over 5 min" in early["why_not_track"]
    assert last["source"] == "track" and last["saved_times"] == 4 and last["span_s"] == 900.0
    assert last["u_m_s"] == pytest.approx(STORM[0], abs=0.05)
    assert last["v_m_s"] == pytest.approx(STORM[1], abs=0.05)
    track = json.loads((run / "quicklook" / ".state" / NG.TRACK_FILE).read_text(encoding="utf-8"))
    assert track["schema"] == NG.TRACK_SCHEMA and len(track["frames"]) == 4


def test_a_jump_to_another_cell_starts_the_track_again():
    rows = {f"k{n}": {"t": 300.0 * n, "xg": 12.0 * 300.0 * n, "yg": 0.0} for n in range(5)}
    rows["k1"]["xg"] += 30000.0          # another cell, 30 km away, was strongest at the second saved time
    track = NG.StormTrack(None)
    track.rows = rows
    fit, _why = track.fit(list(rows), 1200.0)
    assert fit is not None and fit[2] == 3 and fit[0] == pytest.approx(12.0)
    rows["k3"]["xg"] += 30000.0
    fit, why = track.fit(list(rows), 1200.0)
    assert fit is None and why.startswith("the storm's updraft could not be followed back to the saved time before")
    fit, why = track.fit(["k0"], 0.0)
    assert fit is None and why == ("the storm's updraft has been seen at only 1 saved time in the last 30 min; a "
                                   "track needs 3 saved times over 10 min or more")


def _two_cells(order: str, saves: int = 3, dt: float = 900.0) -> dict[str, dict]:
    """Rows of two updrafts 25 km apart (A moving 12, 7 m/s; B moving 8, -3 m/s), every ``dt`` s; ``order`` names
    the stronger one at each saved time, as the storm's place (the ring) would be."""

    rows = {}
    for n in range(saves):
        t = dt * n
        a = (10_000.0 + 12.0 * t, 10_000.0 + 7.0 * t, 30.0 if order[n] == "A" else 20.0)
        b = (10_000.0 + 8.0 * t, 35_000.0 - 3.0 * t, 30.0 if order[n] == "B" else 20.0)
        strongest, other = (a, b) if order[n] == "A" else (b, a)
        rows[f"k{n}"] = {"t": t, "xg": strongest[0], "yg": strongest[1], "w": strongest[2],
                         "peaks": [list(strongest), list(other)]}
    return rows


def test_the_track_follows_one_updraft_when_two_take_turns_being_strongest():
    """Saves every 15 min, two cells 25 km apart (as the two halves of a split) taking turns being the strongest:
    the step from one to the other is about 30 m/s, under the 40 m/s limit, so only following the one updraft back
    keeps the other out of the motion."""

    track = NG.StormTrack(None)
    for order, expect in (("ABA", (12.0, 7.0)), ("BAB", (8.0, -3.0)), ("BBA", (12.0, 7.0)), ("AAB", (8.0, -3.0))):
        track.rows = _two_cells(order)
        fit, why = track.fit(list(track.rows), 1800.0)
        assert fit is not None, (order, why)
        assert (fit[0], fit[1]) == pytest.approx(expect, abs=1e-6), order
        assert fit[2] == 3 and fit[3] == 1800.0
    # with only the strongest place kept at each saved time (no other updraft to choose), a switch is a miss of
    # far more than 8 km from the line through the later places: no track, and the sounding's estimate stands in
    rows = _two_cells("ABAB", saves=4)
    for row in rows.values():
        row.pop("peaks")
    track.rows = rows
    fit, why = track.fit(list(rows), 2700.0)
    assert fit is None and "followed at 2 saved times over 15 min" in why
    # saves every 5 min: a weak cell sits nearer the storm's place now than the storm itself was 5 min before (as
    # in a real 100 m run); the storm is still the one followed, over all seven saved times
    rows = {}
    for n in range(7):
        t = 300.0 * n
        storm = [60_000.0 + 16.0 * t, 50_000.0 + 9.0 * t, 45.0]
        weak = [storm[0] + 7_500.0, storm[1] + 4_000.0, 9.0]            # 3 km from where the storm is next
        rows[f"k{n}"] = {"t": t, "xg": storm[0], "yg": storm[1], "peaks": [storm, weak]}
    track.rows = rows
    fit, why = track.fit(list(rows), 1800.0)
    assert fit is not None and fit[2] == 7 and (fit[0], fit[1]) == pytest.approx((16.0, 9.0)), why
    # B alone before it was the stronger one: A's places at those times are not mixed in
    rows = _two_cells("BBA")
    rows["k2"].update({"xg": rows["k2"]["peaks"][1][0], "yg": rows["k2"]["peaks"][1][1]})   # B is the ring now
    for name in ("k0", "k1"):
        rows[name]["peaks"] = [p for p in rows[name]["peaks"] if p[2] == 30.0]
    track.rows = rows
    fit, why = track.fit(list(rows), 1800.0)
    assert fit is not None and (fit[0], fit[1]) == pytest.approx((8.0, -3.0))
    # no updraft within reach at the saved time before (another storm far away): the track stops there
    rows["k1"]["peaks"] = [[60_000.0, 60_000.0, 30.0]]
    fit, why = track.fit(list(rows), 1800.0)
    assert fit is None and why.startswith("the storm's updraft could not be followed back")


def test_each_saved_time_keeps_its_strongest_updrafts_apart_from_each_other():
    from types import SimpleNamespace

    nx = ny = 200
    xx, yy = np.meshgrid((np.arange(nx) + 0.5) * 250.0, (np.arange(ny) + 0.5) * 250.0)

    def cell(x, y, w):
        return w * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2000.0 ** 2)

    w = (cell(10_000.0, 10_000.0, 30.0) + cell(15_000.0, 10_000.0, 20.0)          # one storm with two cores
         + cell(10_000.0, 35_000.0, 25.0) + cell(40_000.0, 40_000.0, 4.0)).astype(np.float32)   # another; a weak one
    qf = SimpleNamespace(dx=250.0, dy=250.0, zf=np.array([0.0, 5000.0]), domainlocx=1000.0, domainlocy=2000.0,
                         level_near=lambda z, faces=False: 1, get=lambda name, k: w)
    peaks, level = NG.storm_peaks(qf)
    assert level == 5000.0 and len(peaks) == 2
    assert (peaks[0]["x"], peaks[0]["y"]) == pytest.approx((10_125.0, 10_125.0))
    assert (peaks[1]["x"], peaks[1]["y"]) == pytest.approx((10_125.0, 35_125.0))
    assert (peaks[0]["xg"], peaks[0]["yg"]) == pytest.approx((11_125.0, 12_125.0))
    assert peaks[0]["w"] > peaks[1]["w"] >= NG.TRACK_MIN_W
    assert NG.storm_place(qf)["x"] == peaks[0]["x"]


def test_without_a_track_the_motion_is_the_bunkers_estimate_the_run_recorded(tmp_path):
    receipt = {"initial_sounding_parameters": {"schema": "gm1.sounding-parameters/v1",
                                               "bunkers_right_u_m_per_s": 9.5, "bunkers_right_v_m_per_s": 4.25}}
    run = fine_run(tmp_path, times=(0.0, 300.0), receipt=receipt)
    from cm1core.cm1out import frames as F
    from cm1core.quicklook.facts import run_facts
    from cm1core.quicklook.frame import frames_of

    layout = F.discover(run)
    rows = frames_of(layout)
    t, key, path, segment = rows[-1]
    qf = QuickFrame.open(path, key=key, segment=segment)
    try:
        L = P.Looks(qf, run_facts(layout, qf.config, qf.dx, qf.dy))
        L.history = {"before": rows[:-1], "open": lambda p, k, s: QuickFrame.open(p, key=k, segment=s),
                     "state_dir": tmp_path / "state", "layout": layout}
        motion = NG.motion_for(L)
    finally:
        qf.close()
    assert motion.source == "bunkers-record" and (motion.u, motion.v) == (9.5, 4.25)
    assert "Bunkers right-mover estimate from the sounding (the run's receipt)" in motion.how


def test_the_base_state_bunkers_estimate_is_the_published_method_on_the_ground_relative_wind(tmp_path):
    from cm1core.run_geometry import Hodograph, bunkers_storm_motion

    run = fine_run(tmp_path, times=(0.0,))
    qf = QuickFrame.open(run / "frames" / "frame-000000")
    try:
        got = NG._bunkers_base_state(qf)
    finally:
        qf.close()
    ug, vg = base_wind(ZH)
    expect = bunkers_storm_motion(Hodograph((0.0, *map(float, ZH)), (float(ug[0]), *map(float, ug)),
                                            (float(vg[0]), *map(float, vg)))).right_mover_m_s
    assert got == pytest.approx(expect, abs=1e-4)


def test_without_the_base_state_wind_the_estimate_reads_the_first_saved_times_corner(tmp_path):
    from cm1core.cm1out import frames as F
    from cm1core.quicklook.facts import run_facts
    from cm1core.quicklook.frame import frames_of

    run = fine_run(tmp_path, times=(0.0, 300.0), drop=("u0", "v0"))
    layout = F.discover(run)
    rows = frames_of(layout)
    t, key, path, segment = rows[-1]
    qf = QuickFrame.open(path, key=key, segment=segment)
    try:
        L = P.Looks(qf, run_facts(layout, qf.config, qf.dx, qf.dy))
        L.history = {"before": rows[:-1], "open": lambda p, k, s: QuickFrame.open(p, key=k, segment=s),
                     "state_dir": None, "layout": layout}
        motion = NG.motion_for(L)
        base = NG._bunkers_of(qf, *[np.asarray(c, np.float64) - m for c, m in zip(base_wind(ZH), MOVE)])
    finally:
        qf.close()
    # the base wind is the same in every column, so the corner columns hold it
    assert motion.source == "bunkers-corner" and "the corner columns of the first saved time (T+0:00:00)" in motion.how
    assert (motion.u, motion.v) == pytest.approx(base, abs=1e-4)


# ---------------------------------------------------------------- the cold pool and the spin

def test_the_cold_pool_depth_is_where_theta_warms_through_one_kelvin_below_the_area(tmp_path):
    run = fine_run(tmp_path, times=(0.0,))
    qf = QuickFrame.open(run / "frames" / "frame-000000")
    try:
        depth, first, _used = NG.cold_depth(qf)
    finally:
        qf.close()
    i, j = int(POOL[0] // DX), int(POOL[1] // DX)
    # theta' rises linearly from -3 K at the ground to 0 at 600 m: 1 K below the area's median at exactly 400 m,
    # whatever the whole area's offset (WARM); the colder layer at 1-1.2 km sits over a warmer gap and is left out
    assert depth[j, i] == pytest.approx(400.0, abs=0.05)
    assert first[j, i] == pytest.approx(-3.0 + 3.0 * ZH[0] / 600.0, abs=1e-5)
    assert depth[0, 0] == 0.0 and float(depth.max()) == pytest.approx(400.0, abs=0.05)
    assert int((depth > 0).sum()) == int((np.hypot((np.arange(NX)[None] + 0.5) * DX - POOL[0],
                                                   (np.arange(NY)[:, None] + 0.5) * DX - POOL[1]) <= POOL[2]).sum())


def test_the_cold_pool_picture_shows_the_square_holding_all_the_cold_air():
    from types import SimpleNamespace

    qf = SimpleNamespace(dx=250.0, dy=250.0, extent=(100_000.0, 100_000.0))
    extent = qf.extent
    cold = np.zeros((400, 400), bool)
    assert NG._cold_box(qf, cold) is None and NG.cold_square(extent, []) is None   # no cold air: the whole area
    cold[100:140, 200:240] = True                                  # 10 km across: a 40 km square around it
    assert NG._cold_box(qf, cold) == [50_000.0, 25_000.0, 60_000.0, 35_000.0]
    assert NG.cold_square(extent, [NG._cold_box(qf, cold)]) == pytest.approx((35_000.0, 10_000.0, 40_000.0, 40_000.0))
    cold[100:140, 0:240] = True              # 60 km across, at the west edge: 70 km, up to 80 km, kept inside
    assert NG.cold_square(extent, [NG._cold_box(qf, cold)]) == pytest.approx((0.0, 0.0, 80_000.0, 80_000.0))
    assert NG.square_words((0.0, 0.0, 85_750.0, 41_520.0)) == "85.8 × 41.5 km"
    assert [NG.step_up(s) for s in (1.0, 40_000.0, 41_500.0, 60_000.0, 85_800.0, 130_000.0, 400_000.0)] == \
        [40_000.0, 40_000.0, 60_000.0, 60_000.0, 120_000.0, 160_000.0, 480_000.0]
    assert NG.edges_reached(cold) == ["west"]
    cold[0, 5] = cold[-1, 5] = cold[7, -1] = True
    assert NG.edges_reached(cold) == ["north", "south", "east", "west"]


def test_the_cold_pool_square_never_shrinks_from_one_saved_time_to_the_next():
    """A cold pool that grows, moves and then shrinks: the area shown holds the cold air of every saved time so
    far, its side changes only in the stated steps, and it never gets smaller."""

    extent = (200_000.0, 150_000.0)
    pools = [(60e3, 60e3, 3e3), (66e3, 62e3, 9e3), (75e3, 66e3, 18e3), (88e3, 70e3, 30e3), (95e3, 72e3, 6e3),
             (100e3, 75e3, 0.0), (140e3, 20e3, 2e3)]
    boxes, sides = [], []
    for x, y, r in pools:
        if r > 0:
            boxes.append([x - r, y - r, x + r, y + r])
        square = NG.cold_square(extent, boxes)
        x0, y0, w, h = square
        assert w == h or w == extent[0] or h == extent[1]
        assert NG.step_up(w) == w or w == extent[0]
        for b in boxes:
            assert x0 <= b[0] and y0 <= b[1] and b[2] <= x0 + w and b[3] <= y0 + h
        sides.append(w)
    assert sides == [40_000.0, 40_000.0, 60_000.0, 80_000.0, 80_000.0, 80_000.0, 120_000.0]


def test_the_horizontal_spin_of_a_sheared_wind_is_its_shear(tmp_path):
    run = fine_run(tmp_path, times=(0.0,))
    qf = QuickFrame.open(run / "frames" / "frame-000000")
    try:
        k = NG.spin_face(qf.zh, qf.zf)
        ox, oy, u, v, z = NG.horizontal_spin(qf, k)
    finally:
        qf.close()
    assert z == 100.0 and (qf.zh[k - 1], qf.zh[k]) == pytest.approx((90.0, 110.0))
    # far from the updraft w = 0: omega_x = -dv/dz, omega_y = du/dz, and the wind is the base wind at 100 m
    assert ox[-1, -1] == pytest.approx(-10.0 / 3000.0, rel=1e-4)
    assert oy[-1, -1] == pytest.approx(15.0 / 6000.0, rel=1e-4)
    ug, vg = base_wind(np.array([100.0]))
    assert (u[-1, -1], v[-1, -1]) == pytest.approx((float(ug[0]), float(vg[0])), abs=1e-4)


def test_the_spin_look_needs_levels_100_m_apart_near_the_ground():
    uniform = np.arange(0.0, 18001.0, 250.0)
    with pytest.raises(P.NotDrawn, match="250 m apart \\(the first at 125 m\\)"):
        NG.spin_face(0.5 * (uniform[1:] + uniform[:-1]), uniform)
    hundred = np.arange(0.0, 18001.0, 100.0)
    assert NG.spin_face(0.5 * (hundred[1:] + hundred[:-1]), hundred) == 1
    coarse = np.arange(0.0, 18001.0, 125.0)
    with pytest.raises(P.NotDrawn, match="125 m apart"):
        NG.spin_face(0.5 * (coarse[1:] + coarse[:-1]), coarse)
    assert NG.spin_face(ZH, ZF) == 5


# ---------------------------------------------------------------- the pictures

@NEEDS_MATPLOTLIB
def test_the_three_looks_draw_on_a_fine_grid_with_their_words(tmp_path, monkeypatch):
    seen = []
    original = P.save

    def spy(ov, base, path):
        seen.append((Path(path).parent.name, ov))
        return original(ov, base, path)

    monkeypatch.setattr(P, "save", spy)
    run = fine_run(tmp_path)
    summary = _render(run, only=P.NEAR_GROUND)
    assert summary.problems == []
    for pid in P.NEAR_GROUND:
        frames = _product(run, pid)["frames"]
        assert [f["time_s"] for f in frames] == [0.0, 300.0, 600.0, 900.0], pid
        from PIL import Image

        with Image.open(run / "quicklook" / frames[-1]["file"]) as im:
            assert im.size == (1920, 1080)
    assert len(seen) == 12
    for name, ov in seen:
        assert ov.check() == [] and ov.footer_cut == [], (name, ov.check(), ov.footer_cut)
        text = "\n".join(ov.texts)
        assert "simulated" in text and "dx 100 m" in text
    words = {name: "\n".join(ov.texts) for name, ov in seen}
    assert "wind storm-relative" in words["storm-relative-flow"]
    assert "from the run's own track" in words["storm-relative-flow"]
    assert "w level 100 m" in words["streamwise-spin"] and "by this picture" in words["streamwise-spin"]
    assert "at least 1 K below the median θ of the whole area" in words["cold-pool-depth"]
    assert "+ fixed to the ground every" in words["cold-pool-depth"]
    cold = _product(run, "cold-pool-depth")["frames"][-1]["facts"]
    assert cold["threshold_k"] == 1.0 and cold["depth_max_m"] == pytest.approx(400.0, abs=0.05)
    # the coldest air is said as how far below the air around it, not as a negative number "below"
    assert cold["coldest_minus_environment_k"] == pytest.approx(-3.0 + 3.0 * ZH[0] / 600.0, abs=1e-5)
    assert f"{-cold['coldest_minus_environment_k']:.1f} K" in words["cold-pool-depth"].split("\n")
    assert cold["reaches_edges"] == [] and "reaches the domain" not in words["cold-pool-depth"]
    spin = _product(run, "streamwise-spin")["frames"][-1]["facts"]
    assert spin["height_m"] == 100.0 and spin["storm_motion"]["source"] == "track"
    assert spin["spin_max"] <= spin["spin_max_whole_area"]


@NEEDS_MATPLOTLIB
def test_without_an_updraft_the_squares_are_the_middle_of_the_area_and_say_so(tmp_path, monkeypatch):
    seen = []
    original = P.save

    def spy(ov, base, path):
        seen.append((Path(path).parent.name, chr(10).join(ov.texts)))
        return original(ov, base, path)

    monkeypatch.setattr(P, "save", spy)
    run = fine_run(tmp_path, times=(0.0,), calm=True)
    _render(run, only=("storm-relative-flow", "streamwise-spin"))
    words = dict(seen)
    for pid in ("storm-relative-flow", "streamwise-spin"):
        assert "in the middle of the area" in words[pid] and "around the storm" not in words[pid], pid
        assert "no updraft of 5 m/s at 5 km yet" in words[pid], pid
    facts = _product(run, "storm-relative-flow")["frames"][0]["facts"]
    assert "storm_xy_m" not in facts and facts["storm_motion"]["source"] == "bunkers-base-state"


def _spy_texts(monkeypatch) -> list[tuple[str, S.Overlay, list[tuple[str, tuple[int, int, int, int]]]]]:
    """Each picture saved: (look, overlay, [(text, its box on the picture)])."""

    seen = []
    text, save = S.Overlay.text, P.save

    def text_spy(self, xy, words, *args, **kwargs):
        box = text(self, xy, words, *args, **kwargs)
        self.__dict__.setdefault("drawn", []).append((words, box))
        return box

    def save_spy(ov, base, path):
        seen.append((Path(path).parent.name, ov, list(ov.__dict__.get("drawn", []))))
        return save(ov, base, path)

    monkeypatch.setattr(S.Overlay, "text", text_spy)
    monkeypatch.setattr(P, "save", save_spy)
    return seen


@NEEDS_MATPLOTLIB
def test_the_longest_storm_motion_words_wrap_clear_of_the_colour_bars(tmp_path, monkeypatch):
    """No sounding recorded and no base-state wind: the motion is worked out from the first saved time's corner
    columns, the longest of the words, and they must not run under the colour bars on the right."""

    seen = _spy_texts(monkeypatch)
    run = fine_run(tmp_path, times=(0.0, 300.0), drop=("u0", "v0"))
    assert _render(run, only=("storm-relative-flow",)).problems == []
    how = ("the Bunkers right-mover estimate from the corner columns of the first saved time (T+0:00:00), worked out "
           "for this picture: the storm's updraft has been followed at 2 saved times over 5 min; a track needs 3 "
           "saved times over 10 min or more")
    (_look, ov, drawn) = seen[-1]
    texts = [words for words, _box in drawn]
    first = next(n for n, words in enumerate(texts) if words.startswith("the Bunkers"))
    lines = [texts[first]]
    for words in texts[first + 1:]:
        if not how.startswith(" ".join([*lines, words])):
            break
        lines.append(words)
    assert " ".join(lines) == how and len(lines) >= 2
    bars = NG.flow_bars_left(ov)
    # every line on the left under the panels, above the footer, ends before the first colour bar
    lower = [(words, box) for words, box in drawn if box[0] == ov.x0 and 700 < box[1] < 960]
    assert len(lower) >= 4 and all(words in [w for w, _ in lower] for words in lines)
    for words, box in lower:
        assert box[2] < bars, (words, box, bars)
    assert ov.check() == [] and ov.footer_cut == []


@NEEDS_MATPLOTLIB
def test_the_cold_pool_area_holds_the_cold_air_so_far_and_says_when_it_reaches_the_edge(tmp_path, monkeypatch):
    """A small run with the steps made small (the 7.2 km area is under the usual 40 km): a cold pool that grows
    and then shrinks is shown in an area that only grows, the same whether the saved times are drawn in order or
    the last one is drawn on its own; and cold air at the domain's edge is said."""

    monkeypatch.setattr(NG, "COLD_BOX_M", 1000.0)
    monkeypatch.setattr(NG, "COLD_MARGIN_M", 200.0)
    monkeypatch.setattr(NG, "COLD_STEPS_M", (1000.0, 1500.0))
    pools = {0.0: (2000.0, 2000.0, 400.0), 300.0: (3000.0, 2600.0, 900.0), 600.0: (3500.0, 3000.0, 300.0)}
    run = fine_run(tmp_path, times=tuple(pools), pool=lambda t: pools[t])
    assert _render(run, only=("cold-pool-depth",)).problems == []
    frames = _product(run, "cold-pool-depth")["frames"]
    views = [f["facts"]["box_m"] for f in frames]
    assert [v[2] for v in views] == [1500.0, 3000.0, 3000.0]
    for n, view in enumerate(views):
        for x, y, r in list(pools.values())[:n + 1]:
            assert view[0] <= x - r and view[1] <= y - r and x + r <= view[0] + view[2] and y + r <= view[1] + view[3]
    notes = json.loads((run / "quicklook" / ".state" / NG.COLD_FILE).read_text(encoding="utf-8"))
    assert notes["schema"] == NG.COLD_SCHEMA and len(notes["frames"]) == 3

    from cm1core.cm1out import frames as F
    from cm1core.quicklook.facts import run_facts
    from cm1core.quicklook.frame import frames_of

    layout = F.discover(run)
    rows = frames_of(layout)
    t, key, path, segment = rows[-1]
    qf = QuickFrame.open(path, key=key, segment=segment)
    try:
        L = P.Looks(qf, run_facts(layout, qf.config, qf.dx, qf.dy))
        L.history = {"before": rows[:-1], "open": lambda p, k, s: QuickFrame.open(p, key=k, segment=s),
                     "state_dir": tmp_path / "fresh", "layout": layout}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            alone = NG.product_cold_pool(L, tmp_path / "alone.png")
    finally:
        qf.close()
    assert alone["box_m"] == views[-1]

    seen = _spy_texts(monkeypatch)
    edge = fine_run(tmp_path, times=(0.0,), name="edge", pool=(0.0, 3550.0, 1500.0))
    assert _render(edge, only=("cold-pool-depth",)).problems == []
    facts = _product(edge, "cold-pool-depth")["frames"][0]["facts"]
    assert facts["reaches_edges"] == ["west"]
    (_look, ov, drawn) = seen[-1]
    said = " ".join(words for words, _box in drawn)
    assert "the cold air reaches the domain's west edge: the area and the depths count only what is inside" in said
    assert ov.check() == [] and ov.footer_cut == []
    room = P.W // 2 - 400
    for words, box in drawn:
        if box[0] == ov.x0 and 165 < box[1] < 965:
            assert box[2] < room, (words, box)


@NEEDS_MATPLOTLIB
def test_a_run_one_column_wide_or_with_a_low_top_gets_a_plain_reason(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "NY", 1)
    narrow = fine_run(tmp_path, times=(0.0,), name="narrow")
    summary = _render(narrow, only=P.NEAR_GROUND)
    # (the run's statistics, older than these looks, have their own problem with one row; not these looks)
    assert not [p for p in summary.problems if any(pid in p for pid in P.NEAR_GROUND)], summary.problems
    assert [len(_product(narrow, pid)["frames"]) for pid in P.NEAR_GROUND] == [1, 0, 1]
    reason = _product(narrow, "streamwise-spin")["not_drawn"][0]["reason"]
    assert reason == ("Not drawn: this run is 72 × 1 columns, too few to work out how w changes from column to "
                      "column. The spin needs 3 columns or more each way.")
    monkeypatch.setitem(globals(), "NY", 72)

    low = np.arange(0.0, 4000.0 + 1.0, 100.0).astype(np.float32)
    monkeypatch.setitem(globals(), "ZF", low)
    monkeypatch.setitem(globals(), "NZ", len(low) - 1)
    monkeypatch.setitem(globals(), "ZH", 0.5 * (low[1:] + low[:-1]))
    top = fine_run(tmp_path, times=(0.0,), name="top", drop=("u0", "v0"))
    assert _render(top, only=("storm-relative-flow",)).problems == []
    reason = _product(top, "storm-relative-flow")["not_drawn"][0]["reason"]
    assert reason == ("No storm motion to take away: the storm has no steady track yet (the storm's updraft has been "
                      "seen at only 1 saved time in the last 30 min; a track needs 3 saved times over 10 min or "
                      "more), and the run's wind does not reach 6 km, which the Bunkers estimate needs.")


@NEEDS_MATPLOTLIB
def test_a_coarse_grid_and_coarse_levels_get_their_plain_reasons(tmp_path):
    run = run_directory(tmp_path)                      # CM1's own 1 km storm
    summary = _render(run, only=P.NEAR_GROUND)
    assert summary.problems == []
    for pid in P.NEAR_GROUND:
        product = _product(run, pid)
        assert product["frames"] == []
        reason = product["not_drawn"][0]["reason"]
        assert reason.startswith("This look is drawn for grids of 250 m and finer; this run's grid is 1 km"), reason
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    for pid in P.NEAR_GROUND:
        assert rows[pid]["available"] is False and "250 m and finer" in rows[pid]["why_not"], rows[pid]
        assert rows[pid]["action"] is None
        # the reason says which grids the look is drawn for; the note under it does not say it again
        assert rows[pid]["note"] == "", rows[pid]

    uniform = np.arange(0.0, 250.0 * NZ + 1.0, 250.0).astype(np.float32)
    levels = fine_run(tmp_path, times=(0.0, 300.0), name="levels", zf=uniform)
    _render(levels, only=("streamwise-spin",))
    reason = _product(levels, "streamwise-spin")["not_drawn"][0]["reason"]
    assert "250 m apart (the first at 125 m)" in reason and "100 m apart or closer" in reason
    rows = {row["id"]: row for row in GL.more_looks("storm", levels, _index(levels), {"state": "finished"},
                                                   films_now={})}
    assert rows["streamwise-spin"]["why_not"] == reason and rows["streamwise-spin"]["hint"] == "dz = 100"
    assert rows["streamwise-spin"]["note"].startswith("Drawn for grids of 250 m and finer with model levels")

    # a film of a look that will not be drawn gives the look's own reason, not "its pictures are still to come"
    from cm1core.quicklook import film as F

    for target, look, why in ((levels, "streamwise-spin", reason),
                              (run, "cold-pool-depth", _product(run, "cold-pool-depth")["not_drawn"][0]["reason"])):
        with pytest.raises(F.FilmError) as caught:
            F.make(F.FilmJob(target, look, (), 2.0, 0, target / "quicklook"))
        assert str(caught.value) == f"The {look} look has no pictures to film. {why}"


@NEEDS_MATPLOTLIB
def test_a_run_drawn_before_these_looks_offers_to_draw_them_and_a_coarse_one_says_why(tmp_path):
    run = fine_run(tmp_path, times=(0.0, 300.0))
    _render(run, only=("reflectivity", "clouds-from-above", "environment"))
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    for pid in P.NEAR_GROUND:
        assert rows[pid]["action"] == "all" and rows[pid]["why_not"].startswith("This run's pictures were drawn")
    coarse = run_directory(tmp_path / "km")
    _render(coarse, only=("reflectivity", "clouds-from-above", "environment"))
    rows = {row["id"]: row for row in GL.more_looks("storm", coarse, _index(coarse), {"state": "finished"},
                                                   films_now={})}
    for pid in P.NEAR_GROUND:
        assert rows[pid]["action"] is None
        assert rows[pid]["why_not"] == "This look is drawn for grids of 250 m and finer; this run's grid is 1 km."


@NEEDS_MATPLOTLIB
def test_the_looks_drawn_before_stay_byte_identical_beside_these(tmp_path):
    """The thunderstorm looks of 0.1.5, drawn alone and drawn beside the three new ones, from the same frames."""

    older = tuple(pid for pid, _ in P.PRODUCTS if pid not in P.HURRICANE_ONLY and pid not in P.NEAR_GROUND)
    alone = fine_run(tmp_path / "a", times=(0.0, 300.0))
    beside = fine_run(tmp_path / "b", times=(0.0, 300.0))
    with warnings.catch_warnings():
        # this small made-up run is not what those looks' footers are sized for (one of them cuts its last
        # clause); here only the bytes matter
        warnings.simplefilter("ignore", UserWarning)
        RD.render(alone, only=older)
        RD.render(beside)
    assert len(_product(beside, "storm-relative-flow")["frames"]) == 2
    def older_pictures(run: Path) -> list[Path]:
        return sorted(p.relative_to(run / "quicklook") for p in (run / "quicklook").rglob("*.png")
                      if p.parent.name in older)

    pictures = older_pictures(alone)
    # this run saves no rain, no cloud and no air density, and its updraft does not turn: the looks drawn from
    # those (rain and the clouds seen from above, and some of the newer looks) say so instead of drawing; every
    # other look draws both saved times
    drawn = [pid for pid in older if _product(alone, pid)["frames"]]
    assert not {"rain", "clouds-from-above"} & set(drawn)
    assert {"reflectivity", "spin-near-ground", "updraft-5km", "swath", "surface", "updraft-slice", "hail-track",
            "spin-track"} <= set(drawn)
    assert all(_product(alone, pid)["not_drawn"] for pid in older if pid not in drawn)
    assert len(pictures) == 2 * len(drawn) and pictures == older_pictures(beside)
    for name in pictures:
        assert (alone / "quicklook" / name).read_bytes() == (beside / "quicklook" / name).read_bytes(), name


def test_the_estimate_of_drawing_the_new_looks_includes_these_on_a_fine_grid():
    fine = {"frames": [{}] * 10, "run": {"grid": "480 × 480 × 80, dx 250 m"}}
    coarser = {"frames": [{}] * 10, "run": {"grid": "480 × 480 × 80, dx 300 m"}}
    added = 10 * (GL.SECONDS_PER_FRAME_NEAR_GROUND + GL.SECONDS_PER_FRAME_NEAR_GROUND_PER_MCELL * 18.432)
    # the same cells on a grid these looks are not drawn for: the other new looks alone
    assert GL.about_seconds(fine, "all") == pytest.approx(GL.about_seconds(coarser, "all") + added, abs=1.0)
    # a run that has every other look lacks only these: their own measured cost, the pass's part included
    drawn = {**fine, "products": [{"id": look} for look in GL.NEW_LOOKS]}
    assert GL.about_seconds(drawn, "all") == round(added)
    # a hurricane on such a grid draws none of them
    hurricane = {**fine, "run": {**fine["run"], "storm_kind": "hurricane"}}
    assert GL.about_seconds({**hurricane, "products": drawn["products"]}, "all") > 0
    assert GL.about_seconds(hurricane, "all") < GL.about_seconds(fine, "all")
    assert GL.SECONDS_PER_FRAME_NEAR_GROUND > 0 and GL.SECONDS_PER_FRAME_NEAR_GROUND_PER_MCELL > 0
