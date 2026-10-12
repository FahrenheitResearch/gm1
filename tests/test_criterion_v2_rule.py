"""Tornado criterion v2 on synthetic vortices and synthetic GM1 output (no GM1 run is read).

* the same analytic vortex gets the same verdict at 20, 8, 4 and 2 m (one-
  and two-celled, flat-topped, thin ring, with satellite vortices);
* the resolvedness guard: RMW 40 m is UNRESOLVED at 20 m and TORNADO finer;
  a 100 m grid is UNRESOLVED;
* negative controls are NO_TORNADO on every grid: a single-cell grid spike,
  a straight shear line (vortex sheet), a weak vortex, a broad mesocyclone,
  a strong vortex lasting 30 s, a strong vortex against the domain edge;
* the scorer reads child frames (stretched/explicit levels, moving
  placement), parent frames and a window stream, interpolates to 10 m AGL
  and states the cadence; one frame or 90 s frames are INSUFFICIENT_CADENCE.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "criterion_v2"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = _load("synthetic_v2", TOOLS / "synthetic_v2.py")
R = _load("v2rule", TOOLS / "v2rule.py")
SC = _load("criterion_v2_score", TOOLS / "score.py")

GRIDS = (20.0, 8.0, 4.0, 2.0)
POSITIVE = ("rankine-R100-V50", "burgers-R100-V60", "sullivan-R100-V60", "ring-R120-V50",
            "burgers-R100-V100+sat", "sullivan-R100-V100+sat",
            "burgers-R150-V50-in-meso-R700-V30", "burgers-R100-V35-in-meso-R600-V30")
NEGATIVE = ("spike-a25", "shear-line-dU80", "finite-sheet-L1000-dU80", "weak-burgers-R150-V15",
            "strong-30s-burgers-R100-V60", "edge-burgers-R100-V60", "broad-meso-burgers-R1500-V25",
            "meso-rankine-R750-V21", "meso-burgers-R600-V21", "rankine-R500-V20.5+gust-900m",
            "rankine-R200-V22+gust-400m")
UNOBSERVABLE = ("hop-60s", "hop-30s", "burgers-R40-V60-30s")


def _outcome(name: str, dx: float) -> str:
    return S.run_case((name, dx))["outcome"]


@pytest.mark.parametrize("name", POSITIVE)
def test_same_vortex_same_verdict_on_every_grid(name: str) -> None:
    got = {dx: _outcome(name, dx) for dx in GRIDS}
    assert set(got.values()) == {"TORNADO"}, got


@pytest.mark.parametrize("name", NEGATIVE)
def test_negative_controls_rejected_on_every_grid(name: str) -> None:
    got = {dx: _outcome(name, dx) for dx in GRIDS}
    assert set(got.values()) == {"NO_TORNADO"}, got


@pytest.mark.parametrize("name", UNOBSERVABLE)
def test_persistence_not_observable_is_insufficient_cadence(name: str) -> None:
    # A different vortex in each sample, or one that moves more than its core
    # diameter between samples in the storm-following frame, cannot show that
    # one vortex persisted (amendment: identity links).
    got = {dx: _outcome(name, dx) for dx in GRIDS}
    assert set(got.values()) == {"INSUFFICIENT_CADENCE"}, got


@pytest.mark.parametrize("name", ("rankine-R90-V21.5", "ring-R100-V21.5"))
def test_near_threshold_resolved_vortex_is_never_no_tornado(name: str) -> None:
    # Measured Vmax reads a few per cent low at 4-5 dx; within 0.5 dx/RMW of a
    # threshold the sample is marginal (U), not N (amendment: marginal band).
    got = {dx: _outcome(name, dx) for dx in GRIDS}
    assert got[20.0] in ("UNRESOLVED", "TORNADO") and {got[8.0], got[4.0], got[2.0]} == {"TORNADO"}, got


def test_resolvedness_guard() -> None:
    got = {dx: _outcome("burgers-R40-V60", dx) for dx in GRIDS}
    assert got == {20.0: "UNRESOLVED", 8.0: "TORNADO", 4.0: "TORNADO", 2.0: "TORNADO"}
    assert _outcome("burgers-R100-V60", 100.0) == "UNRESOLVED"


def test_two_celled_centre_is_the_axis_not_the_ring() -> None:
    # v1 took the first strict maximum, which sits on the ring and hops round it.
    for kind in ("sullivan", "ring"):
        c = S.VortexComponent(kind, 60.0, 100.0, 803.0, 797.0, drift=False)
        s = R.evaluate_plane(S.make_plane([c], [], 1600.0, 4.0, 0.0))
        assert math.hypot(s["centre_x_m"] - 803.0, s["centre_y_m"] - 797.0) <= 2.0
        assert abs(s["rmw_m"] - 100.0) <= 6.0 and abs(s["vmax_m_s"] - 60.0) <= 3.0


def test_criterion_constants_are_hashed() -> None:
    assert R.CRITERION_V2_SHA256 == __import__("hashlib").sha256(
        R._canonical(R.CRITERION_V2).encode("ascii")).hexdigest()
    with pytest.raises(TypeError):
        R.CRITERION_V2["sustain_s"] = 60.0  # type: ignore[index]


Q = {"found": True, "class": "Q", "conditions": "ERCAZWS", "centre_ground_x_m": 0.0, "centre_ground_y_m": 0.0,
     "centre_x_m": 0.0, "centre_y_m": 0.0, "vmax_m_s": 30.0, "ground_wind_max_m_s": 40.0, "rmw_m": 100.0,
     "flags": []}
G4 = R.GridFacts(dx_m=4.0, dy_m=4.0, lowest_scalar_level_m=2.0)


def test_decide_cadence_rules() -> None:
    q, g = Q, G4
    one = R.decide([dict(q, model_time_s=0.0)], g)
    assert one["outcome"] == "INSUFFICIENT_CADENCE"
    sparse = R.decide([dict(q, model_time_s=90.0 * k) for k in range(4)], g)
    assert sparse["outcome"] == "INSUFFICIENT_CADENCE"
    f60 = R.decide([dict(q, model_time_s=60.0 * k) for k in range(3)], g)
    assert f60["outcome"] == "TORNADO" and "60 s output" in f60["caption"]
    # a break in the middle: 60 s + 60 s is not 120 s
    br = [dict(q, model_time_s=30.0 * k) for k in range(9)]
    br[4] = dict(br[4], **{"class": "N"})
    assert R.decide(br, g)["outcome"] == "NO_TORNADO"
    assert R.decide(br, g)["reported_60s_rule"]["would_be_tornado"]
    # a jump larger than 60 m/s * dt + 100 m breaks continuity
    jump = [dict(q, model_time_s=30.0 * k) for k in range(6)]
    jump[3] = dict(jump[3], centre_ground_x_m=2000.0)
    assert R.decide(jump, g)["outcome"] == "NO_TORNADO"
    coarse = R.decide([dict(q, model_time_s=30.0 * k) for k in range(6)],
                      R.GridFacts(dx_m=4.0, dy_m=4.0, lowest_scalar_level_m=25.0))
    assert coarse["outcome"] == "UNRESOLVED" and "lowest scalar level" in coarse["reason"]


def test_record_with_a_gap_is_insufficient_cadence() -> None:
    # Reviewer reproducer (code bug at the first freeze): samples at 0, 60, 150, 240, 330, 420 s, all Q, gave
    # NO_TORNADO because one interval <= 60 s existed somewhere.  No stretch at <= 60 s spacing lasts 120 s.
    v = R.decide([dict(Q, model_time_s=t) for t in (0.0, 60.0, 150.0, 240.0, 330.0, 420.0)], G4)
    assert v["outcome"] == "INSUFFICIENT_CADENCE", v
    # 75 s frames (E100W's child frames) can never be judged
    v = R.decide([dict(Q, model_time_s=4275.0 + 75.0 * k) for k in range(12)], R.GridFacts(20.0, 20.0, 10.0))
    assert v["outcome"] == "INSUFFICIENT_CADENCE"


def _cand(x: float, y: float, klass: str = "Q", vmax: float = 30.0, rmw: float = 100.0) -> dict:
    return {"class": klass, "conditions": "ERCAZWS", "centre_x_m": x, "centre_y_m": y, "centre_ground_x_m": x,
            "centre_ground_y_m": y, "vmax_m_s": vmax, "rmw_m": rmw, "ground_wind_max_m_s": 40.0, "flags": []}


def _sample(t: float, cands: list[dict]) -> dict:
    best = dict(max(cands, key=lambda c: c["vmax_m_s"]))
    best.update(found=True, model_time_s=t, candidates=cands, umove=0.0, vmove=0.0)
    return best


def test_second_qualifying_vortex_does_not_break_the_first() -> None:
    # Reviewer reproducer (code bug at the first freeze: only the per-sample strongest vortex was followed).
    # Tornado A at x=700 is Q in every 0.5 s record for 240 s; a stronger B at x=1400 flickers in and out.
    ss = []
    for k in range(481):
        t = 0.5 * k
        cands = [_cand(700.0, 0.0, vmax=40.0)]
        if k % 2:
            cands.append(_cand(1400.0, 0.0, vmax=55.0, rmw=80.0))
        ss.append(_sample(t, cands))
    v = R.decide(ss, G4)
    assert v["outcome"] == "TORNADO" and v["longest_qualifying_run_s"] == 240.0, v


def test_different_vortices_are_not_stitched_into_one() -> None:
    # Reviewer reproducer: three vortices 2 km apart, each Q in one 60 s frame, gave TORNADO.
    ss = [_sample(0.0, [_cand(0.0, 0.0)]), _sample(60.0, [_cand(2000.0, 0.0)]), _sample(120.0, [_cand(0.0, 100.0)])]
    v = R.decide(ss, G4)
    assert v["outcome"] == "INSUFFICIENT_CADENCE" and v["longest_qualifying_run_s"] == 0.0, v
    # A -> B -> A: B (700 m away) lives 60 s; A's own track has a 120 s hole, so no episode
    ss = [_sample(60.0 * k, [_cand(0.0 if k in (0, 3, 4) else 700.0, 0.0)]) for k in range(5)]
    assert R.decide(ss, G4)["longest_qualifying_run_s"] == 60.0


def test_identity_is_judged_in_the_storm_following_frame() -> None:
    # A vortex carried by the domain's motion moves 900 m on the ground per 60 s and is one vortex.
    ss = [dict(_sample(60.0 * k, [_cand(900.0 * k, 0.0)]), umove=15.0) for k in range(3)]
    assert R.decide(ss, G4)["outcome"] == "TORNADO"
    ss = [dict(_sample(60.0 * k, [_cand(900.0 * k, 0.0)]), umove=0.0) for k in range(3)]
    assert R.decide(ss, G4)["outcome"] == "INSUFFICIENT_CADENCE"


def test_brief_dips_in_a_window_stream_are_bridged() -> None:
    # Reviewer reproducer: 240 s of Q at 0.5 s with one N record every 100 s gave NO_TORNADO while the same flow
    # at 60 s frames was TORNADO.  Gaps up to bridge_gap_s (3 s, the EF-scale gust duration) are bridged.
    ss = [dict(Q, model_time_s=0.5 * k, **({"class": "N"} if k % 200 == 199 else {})) for k in range(481)]
    v = R.decide(ss, G4)
    assert v["outcome"] == "TORNADO" and v["longest_qualifying_run_s"] == 240.0
    gap = [dict(Q, model_time_s=0.5 * k, **({"class": "N"} if 200 <= k < 210 else {})) for k in range(481)]
    assert R.decide(gap, G4)["longest_qualifying_run_s"] == 135.0


def test_hazard_wind_is_ground_relative_with_the_frame_motion_added() -> None:
    # The W condition had no test at the first freeze; flipping the sign of umove passed every test.
    # Frame-relative background +5 m/s in x, vortex 25 m/s, frame moving +10 m/s: ground max 40 (W holds);
    # with the sign flipped the maximum would be 30 (W fails).
    c = S.VortexComponent("rankine", 25.0, 100.0, 800.0, 800.0, drift=False)
    pl = S.make_plane([c, S.Uniform(5.0 - S.background()[0], -S.background()[1])], [], 1600.0, 8.0, 0.0)
    pl.umove, pl.vmove = 10.0, 0.0
    s = R.evaluate_plane(pl)
    assert 38.5 <= s["ground_wind_max_m_s"] <= 40.5 and s["conditions"][5] == "W", s["conditions"]
    pl.umove = 0.0
    s0 = R.evaluate_plane(pl)
    assert s0["ground_wind_max_m_s"] < 31.0 and s0["conditions"][5] == "-"


def test_shape_and_intensity_conditions() -> None:
    # A 1 km vortex-sheet segment has |V2| = (2/3) Vbar on its RMW circle; a broad mesocyclone has a mean
    # core vorticity below 0.1 /s.  A vortex with small satellites is axisymmetric in wavenumber 2.
    sheet = S.make_plane([S.FiniteShearLine(80.0, 15.0, 1000.0, 1000.0, 1000.0)], [], 2000.0, 8.0, 0.0)
    s = R.evaluate_plane(sheet)
    assert s["wavenumber2_fraction"] > 0.6 and s["class"] == "N"
    meso = S.make_plane([S.VortexComponent("rankine", 21.0, 750.0, 1500.0, 1500.0, drift=False)], [], 3000.0,
                        8.0, 0.0)
    s = R.evaluate_plane(meso)
    assert s["mean_zeta_core_s_inv"] < 0.06 and s["conditions"][4] == "-"
    sat = S.make_plane([S.VortexComponent("burgers", 100.0, 100.0, 800.0, 800.0, satellites=True, drift=False)],
                       [], 1600.0, 4.0, 0.0)
    s = R.evaluate_plane(sat)
    assert s["class"] == "Q" and s["wavenumber2_fraction"] < 0.1


# ---------------------------------------------------------------------------
# Synthetic GM1 output through the scorer.
# ---------------------------------------------------------------------------
Z0 = 0.1


def _log_factor(z: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(z, 0.2) / Z0) / math.log(10.0 / Z0)


def _write_frame(fdir: Path, fields: dict[str, np.ndarray], extra_location: dict | None = None) -> None:
    fdir.mkdir(parents=True)
    rows, off, blobs = [], 0, []
    for name, arr in fields.items():
        a = np.ascontiguousarray(arr, dtype="<f4")
        shape = ",".join(str(s) for s in a.shape)
        rows.append("\t".join([name, "desc", "unit", "stagger", shape, "bounds", "path", str(off)]))
        blobs.append(a.tobytes())
        off += a.nbytes
    (fdir / "history-field-table.tsv").write_text("\n".join(rows) + "\n")
    (fdir / "history-carriers.f32").write_bytes(b"".join(blobs))
    loc = {"schema": "gm1.cartesian-history-frame-location/v1"}
    loc.update(extra_location or {})
    (fdir / "history-frame-location.json").write_text(json.dumps(loc))


def _volume(plane, zh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = _log_factor(zh)
    return (f[:, None, None] * plane.u[None]).astype("<f4"), (f[:, None, None] * plane.v[None]).astype("<f4")


def _child_run(tmp: Path, dx: float, zh: np.ndarray, times: list[float], moves: bool) -> Path:
    root = tmp / "nest-child"
    (root / "placements").mkdir(parents=True)
    L = 1600.0
    n = int(round(L / dx))
    bu, bv = S.background()
    for idx, t in enumerate(times):
        # the storm's axis in ground coordinates; the child origin follows it in steps when ``moves``
        gx = 5000.0 + (bu + S.FRAME_M_S[0]) * t
        gy = 5000.0 + (bv + S.FRAME_M_S[1]) * t
        dom = (S.FRAME_M_S[0] * t, S.FRAME_M_S[1] * t)
        origin = (gx - dom[0] - 800.0 + (37.0 if moves and idx % 2 else 0.0), gy - dom[1] - 800.0)
        origin = (dx * round(origin[0] / dx), dx * round(origin[1] / dx))
        cx, cy = gx - dom[0] - origin[0], gy - dom[1] - origin[1]
        comp = S.VortexComponent("burgers", 60.0, 100.0, cx, cy, drift=False)
        plane = S.make_plane([comp], [], L, dx, 0.0)
        u, v = _volume(plane, zh)
        nz = len(zh)
        fields = {"u3d": u, "v3d": v, "terrain_msl": np.zeros((n, n)),
                  "prs": np.zeros((nz, n, n)), "prs0": np.zeros(nz)}
        _write_frame(root / f"frame-{idx:06d}", fields)
        zf = np.concatenate(([0.0], 0.5 * (zh[1:] + zh[:-1]), [2 * zh[-1] - zh[-2]]))
        placement = {"schema": "gm1.cartesian-nest-child-placement/v1", "index": idx, "model_time_s": t,
                     "nx": n, "ny": n, "nz": nz, "dx_m": dx, "dy_m": dx, "origin_x_m": origin[0],
                     "origin_y_m": origin[1], "domainlocx_m": dom[0], "domainlocy_m": dom[1],
                     "u_move_m_s": S.FRAME_M_S[0], "v_move_m_s": S.FRAME_M_S[1], "spec_zone": 1,
                     "relax_zone": 4, "parent_i0": 0, "parent_j0": 0,
                     "zh_cm1_indexed": [-zh[0]] + zh.tolist() + [zh[-1] + 1.0],
                     "zf_cm1_indexed": [-zf[1]] + zf.tolist()}
        (root / "placements" / f"frame-{idx:06d}.json").write_text(json.dumps(placement))
    return root


NEST = ("--nest-start-s", "-1000", "--parent-dx", "100")


def _score(src: Path, out: Path, *extra: str) -> dict:
    assert SC.main([str(src), "--out", str(out), *extra]) == 0
    return json.loads((out / "verdict.json").read_text())


def test_frozen_manifest_matches_the_rule_and_scorer() -> None:
    check = SC.frozen_check()
    assert check["matches_frozen"], check
    assert check["manifest"]["criterion_v2_sha256"] == R.CRITERION_V2_SHA256


def test_scorer_refuses_a_nest_without_its_spawn_time(tmp_path: Path) -> None:
    zh = np.array([10.0, 30.0, 50.0])
    root = _child_run(tmp_path, 20.0, zh, [0.0, 30.0, 60.0, 90.0, 120.0], moves=False)
    assert SC.main([str(root), "--out", str(tmp_path / "o")]) == 2


def test_scorer_spinup_samples_are_judged_at_the_parent_spacing(tmp_path: Path) -> None:
    # Reviewer reproducer: a 100 m-grid vortex interpolated onto a child passed at the child's dx.  During the
    # child's first nest_spinup_s its samples are judged at the parent's spacing.
    zh = np.array([2.0, 6.0, 10.0, 14.0])
    root = _child_run(tmp_path, 8.0, zh, [0.0, 30.0, 60.0, 90.0, 120.0], moves=False)
    v = _score(root, tmp_path / "spin", "--nest-start-s", "-120", "--parent-dx", "100")
    assert v["outcome"] == "UNRESOLVED" and v["nest_spinup"]["samples_judged_at_parent_dx"] == 5, v
    v = _score(root, tmp_path / "after", *NEST)
    assert v["outcome"] == "TORNADO" and v["protocol_conforming"], v


def test_scorer_child_frames_stretched_levels_moving_placement(tmp_path: Path) -> None:
    zh = np.array([2.0, 6.0, 10.0, 14.0, 18.0, 23.0, 30.0, 40.0])  # 4 m layers then stretched
    # 30 s frames: the axis drifts 3.8 m/s in the storm-following frame, 115 m per frame (< 2 RMW)
    root = _child_run(tmp_path, 8.0, zh, [0.0, 30.0, 60.0, 90.0, 120.0], moves=True)
    v = _score(root, tmp_path / "out", *NEST)
    assert v["outcome"] == "TORNADO", v
    assert v["grid"]["lowest_scalar_level_m"] == 2.0
    # no live sidecar: v1 is recomputed from the frames with its own host sample
    assert v["v1_beside"]["verdict"] in ("TORNADO", "NO_TORNADO") and "recomputed" in v["v1_beside"]["source"]
    assert "30 s output" in v["caption"] and "8 m grid" in v["caption"]
    rows = (tmp_path / "out" / "samples.tsv").read_text().splitlines()
    assert len(rows) == 6
    # ground centre consistent across the moving placement
    col = rows[0].split("\t")
    gx = [float(r.split("\t")[col.index("centre_ground_x_m")]) for r in rows[1:]]
    bu, _ = S.background()
    for k, t in enumerate([0.0, 30.0, 60.0, 90.0, 120.0]):
        assert abs(gx[k] - (5000.0 + (bu + S.FRAME_M_S[0]) * t)) <= 2.0


def test_scorer_ten_metre_interpolation_not_lowest_level(tmp_path: Path) -> None:
    # With a log profile the 2 m wind is 0.65 of the 10 m wind (z0 = 0.1 m); v2 reads 10 m.
    zh = np.array([2.0, 6.0, 10.0, 14.0, 18.0])
    root = _child_run(tmp_path, 8.0, zh, [0.0], moves=False)
    job = SC.child_jobs(root)[0]
    s = SC.score_job(job)
    assert abs(s["vmax_m_s"] - 60.0) <= 2.0
    zh2 = np.array([1.0, 3.0, 7.0, 13.0, 21.0])  # 10 m between levels: linear interpolation of the log profile
    root2 = _child_run(tmp_path / "b", 8.0, zh2, [0.0], moves=False)
    s2 = SC.score_job(SC.child_jobs(root2)[0])
    assert 0.95 * 60.0 <= s2["vmax_m_s"] <= 1.03 * 60.0


def test_scorer_cadence_outcomes(tmp_path: Path) -> None:
    zh = np.array([10.0, 30.0, 50.0])
    one = _child_run(tmp_path / "one", 20.0, zh, [0.0], moves=False)
    assert _score(one, tmp_path / "o1", *NEST)["outcome"] == "INSUFFICIENT_CADENCE"
    sparse = _child_run(tmp_path / "sparse", 20.0, zh, [0.0, 90.0, 180.0], moves=False)
    assert _score(sparse, tmp_path / "o2", *NEST)["outcome"] == "INSUFFICIENT_CADENCE"
    ok = _child_run(tmp_path / "ok", 20.0, zh, [0.0, 30.0, 60.0, 90.0, 120.0], moves=False)
    v = _score(ok, tmp_path / "o3", *NEST)
    assert v["outcome"] == "TORNADO" and v["median_interval_s"] == 30.0


def test_scorer_parent_frames(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    zh = np.array([10.0, 30.0, 55.0])
    for idx, t in enumerate([0.0, 60.0, 120.0]):
        comp = S.VortexComponent("burgers", 60.0, 400.0, 6000.0, 6000.0, drift=False)
        plane = S.make_plane([comp], [], 12000.0, 100.0, 0.0)
        u, v = _volume(plane, zh)
        _write_frame(root / f"frame-{idx:06d}", {"u3d": u, "v3d": v, "zhval": zh},
                     {"index": idx, "model_time_s": t,
                      "moving_frame": {"umove": 16.5, "vmove": 16.5, "domainlocx": 16.5 * t, "domainlocy": 16.5 * t}})
    deck = tmp_path / "deck.toml"
    deck.write_text("[grid]\ndx_m = 100.0\ndy_m = 100.0\nstretch_x = false\nstretch_y = false\n")
    v = _score(root, tmp_path / "out", "--deck", str(deck))
    assert v["outcome"] == "UNRESOLVED" and "25 m" in v["reason"]


def test_scorer_window_stream(tmp_path: Path) -> None:
    root = tmp_path / "nest-child-window"
    root.mkdir()
    dx, nx, ny, nz = 4.0, 225, 225, 6
    zh_full = [-2.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0]
    NG = 3
    shapes = {"uadv": (nz + 2, ny + 2 * NG, nx + 1 + 2 * NG), "vadv": (nz + 2, ny + 1 + 2 * NG, nx + 2 * NG)}
    fields, off = [], 0
    for name, shp in shapes.items():
        fields.append({"name": name, "shape_zyx": list(shp), "offset_words": off})
        off += int(np.prod(shp))
    rec_words = off
    rows = ["\t".join(["index", "model_time_s", "offset_bytes", "i0", "j0", "nx", "ny", "nz", "origin_x_m",
                       "origin_y_m"])]
    blobs = []
    times = [0.5 * k for k in range(0, 241, 20)]  # 10 s apart over 120 s
    for q, t in enumerate(times):
        comp = S.VortexComponent("sullivan", 60.0, 100.0, 450.0 + 3.0 * q, 450.0, drift=False)
        pl = S.make_plane([comp], [], nx * dx, dx, 0.0)
        f = _log_factor(np.array(zh_full[1:nz + 1]))
        ua = np.zeros(shapes["uadv"], "<f4")
        va = np.zeros(shapes["vadv"], "<f4")
        for k in range(nz):
            ua[k + 1, NG:NG + ny, NG:NG + nx + 1] = f[k] * pl.u
            va[k + 1, NG:NG + ny + 1, NG:NG + nx] = f[k] * pl.v
        blobs.append(ua.tobytes() + va.tobytes())
        rows.append("\t".join(str(x) for x in [q, t, q * rec_words * 4, 137, 137, nx, ny, nz, 1000.0, 1000.0]))
    (root / "window-stream.f32").write_bytes(b"".join(blobs))
    (root / "window-records.tsv").write_text("\n".join(rows) + "\n")
    receipt = {"child": {"dx_m": dx, "dy_m": dx, "nx": 500, "ny": 500, "u_move_m_s": 16.5, "v_move_m_s": 16.5,
                         "spec_zone": 1, "relax_zone": 4},
               "zh_cm1_indexed": zh_full, "fields": fields}
    (root / "window-stream.json").write_text(json.dumps(receipt))
    v = _score(root, tmp_path / "out", *NEST)
    assert v["source"]["kind"] == "window"
    assert v["outcome"] == "TORNADO", v
    assert v["median_interval_s"] == 10.0
