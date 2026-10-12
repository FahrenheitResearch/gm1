"""Independent verifier: recomputation of criterion-v2 quantities on raw E100R2-PF-4m window records.

Numpy only.  Does NOT import v2rule.py, score.py or window_segments.py.  Thresholds are typed in from
docs/GM1-TORNADO-CRITERION-V2.md, not read from the scorer.  CPU only (run with nice 19, ionice idle).

  python3 vs_records.py index                      # trajectory, spans, L0-only, cadence
  python3 vs_records.py records T [T ...]          # full recomputation at these model times
  python3 vs_records.py courant T0 T1 STRIDE       # CM1 calccflquick measure (misclibs.F:2726-2814) on the window
  python3 vs_records.py layout T                   # window uadv/vadv vs the child history frame u3d/v3d at T
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

# Where the run's S3 stage folder and the scoring output folder are (the raw data are not published).
S3 = os.environ.get("GM1_RUN_S3", "RUN/S3")
SEG0 = S3 + "/frames/nest-child-window"
TAIL = S3 + "/L0-window-tail-3300.09375-3452.4375"
RUNS = os.environ.get("GM1_SCORING", "SCORING")
NG = 3
NX = NY = 227
NZ = 45
DX = 4.0
UM = VM = 16.5          # deck [closures] u_move_m_s / v_move_m_s
CHILD_N = 500
BOUNDARY = 1 + 4        # spec_zone + relax_zone
T_CLEAN = 3452.4375
NEST_START = 2160.0
# criterion v2 thresholds, typed from docs/GM1-TORNADO-CRITERION-V2.md
VT_MIN, CLOSE_F, CLOSE_R, WN2_MAX, ZETA_MIN, HAZ_R, GW_MIN, RES_DX = 20.0, 0.9, 3.0, 0.5, 0.1, 1.5, 35.0, 4.0
SPINUP, EDGE = 300.0, 300.0
UADV_SHAPE = (NZ + 2, NY + 2 * NG, NX + 1 + 2 * NG)
VADV_SHAPE = (NZ + 2, NY + 1 + 2 * NG, NX + 2 * NG)
WADV_SHAPE = (NZ + 3, NY + 2 * NG, NX + 2 * NG)
REC_WORDS = 34393884


def rows_of(stream):
    lines = open(stream + "/window-records.tsv").read().rstrip("\n").split("\n")
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def trajectory():
    out = []
    for r in rows_of(SEG0):
        t = float(r["model_time_s"])
        if t <= 3300.0 + 1e-9:
            out.append(dict(r, arena=SEG0 + "/window-stream.f32", l0_index=int(r["index"]), t=t))
    readme = json.load(open(TAIL + "/README.json"))
    for r in rows_of(TAIL):
        out.append(dict(r, arena=TAIL + "/window-stream.f32", l0_index=int(r["index"]) + readme["l0_index_first"],
                        t=float(r["model_time_s"])))
    return out


def cmd_index():
    tr = trajectory()
    t = np.array([r["t"] for r in tr])
    d = np.diff(t)
    li = np.array([r["l0_index"] for r in tr])
    rep = {"records": len(tr), "t_first": t[0], "t_last": t[-1], "strictly_increasing": bool(np.all(d > 0)),
           "cadence_unique": sorted(set(np.round(d, 9).tolist())), "l0_index_contiguous": bool(np.all(np.diff(li) == 1)),
           "l0_index_range": [int(li[0]), int(li[-1])], "after_clean": int(np.sum(t > T_CLEAN + 1e-9)),
           "t_equals_2160_plus_k_cadence": bool(np.allclose(t, 2160.0 + 0.09375 * (li + 1), atol=1e-9)),
           "spans": {"pre3312 (t<3312)": int(np.sum(t < 3312.0 - 1e-9)),
                     "3312on (3312<=t<=3452.4375)": int(np.sum((t >= 3312.0 - 1e-9) & (t <= T_CLEAN + 1e-9))),
                     "whole": int(np.sum(t <= T_CLEAN + 1e-9))},
           "last_pre3312": float(t[t < 3312.0 - 1e-9][-1]), "first_3312on": float(t[t >= 3312.0 - 1e-9][0])}
    # origin jumps: the window is fixed at (i0, j0) in the child; origin moves with the nest in whole parent cells
    ox = np.array([float(r["origin_x_m"]) for r in tr])
    oy = np.array([float(r["origin_y_m"]) for r in tr])
    i0 = {int(r["i0"]) for r in tr}
    j0 = {int(r["j0"]) for r in tr}
    jx = np.nonzero((np.diff(ox) != 0) | (np.diff(oy) != 0))[0]
    rep["i0_j0"] = [sorted(i0), sorted(j0)]
    rep["origin_jumps"] = len(jx)
    rep["origin_jump_sizes_m"] = sorted({(float(ox[k + 1] - ox[k]), float(oy[k + 1] - oy[k])) for k in jx})
    rep["centre_minus_origin"] = sorted({(float(r["centre_x_m"]) - float(r["origin_x_m"]),
                                         float(r["centre_y_m"]) - float(r["origin_y_m"])) for r in tr})
    rep["held_or_clamped"] = sum(1 for r in tr if r["centre_held"] != "0" or r["clamped"] != "0")
    # views the scorer read
    for lab in ("E100R2-PF-4m-window-pre3312", "E100R2-PF-4m-window-3312on", "E100R2-PF-4m-window"):
        import glob, os
        vt = []
        for v in sorted(glob.glob(f"{RUNS}/views/{lab}/seg-*")):
            arena = os.path.realpath(v + "/window-stream.f32")
            assert arena in (os.path.realpath(SEG0 + "/window-stream.f32"), os.path.realpath(TAIL + "/window-stream.f32")), arena
            for r in rows_of(v):
                vt.append((float(r["model_time_s"]), r["sha256"], arena, int(r["offset_bytes"])))
        by_t = {round(r["t"], 6): (r["sha256"], os.path.realpath(r["arena"]), int(r["offset_bytes"])) for r in tr}
        same = sum(1 for x in vt if by_t.get(round(x[0], 6)) == (x[1], x[2], x[3]))
        rep["view_" + lab] = {"records": len(vt), "t": [vt[0][0], vt[-1][0]], "identical_to_L0_trajectory_rows": same,
                              "max_t": max(x[0] for x in vt)}
    print(json.dumps(rep, indent=1, default=float))


def row_at(tr, t):
    for r in tr:
        if abs(r["t"] - t) < 1e-6:
            return r
    raise SystemExit(f"no record at {t}")


def read_record(r, verify=True):
    off = int(r["offset_bytes"])
    mm = np.memmap(r["arena"], dtype="<f4", mode="r", offset=off, shape=(REC_WORDS,))
    ok = None
    if verify:
        ok = hashlib.sha256(mm.tobytes()).hexdigest() == r["sha256"]
    n_u = math.prod(UADV_SHAPE)
    n_v = math.prod(VADV_SHAPE)
    ua = np.asarray(mm[:n_u]).reshape(UADV_SHAPE)
    va = np.asarray(mm[n_u:n_u + n_v]).reshape(VADV_SHAPE)
    wa = np.asarray(mm[n_u + n_v:n_u + n_v + math.prod(WADV_SHAPE)]).reshape(WADV_SHAPE)
    return ua, va, wa, ok


def level10(ua, va):
    # scalar heights from the deck's child_levels_m (w faces 0,4,8,12,...): zh(k) = (zf(k)+zf(k+1))/2 -> 2, 6, 10 m;
    # 10 m is CM1 k = 3 exactly; padded index = k (index 0 is the k = 0 ghost)
    k = 3
    u = ua[k, NG:NG + NY, NG:NG + NX + 1].astype(np.float64)       # u[j, i] at (i dx, (j+1/2) dy)
    v = va[k, NG:NG + NY + 1, NG:NG + NX].astype(np.float64)       # v[j, i] at ((i+1/2) dx, j dy)
    return u, v


def interp(f, x, y):
    """Bilinear on a regular lattice: f[j, i] at (i, j) in index units."""
    nj, ni = f.shape
    ii = np.clip(np.floor(x).astype(int), 0, ni - 2)
    jj = np.clip(np.floor(y).astype(int), 0, nj - 2)
    fx, fy = x - ii, y - jj
    return ((1 - fy) * ((1 - fx) * f[jj, ii] + fx * f[jj, ii + 1]) + fy * ((1 - fx) * f[jj + 1, ii] + fx * f[jj + 1, ii + 1]))


def uv_at(u, v, x, y):
    return interp(u, x / DX, y / DX - 0.5), interp(v, x / DX - 0.5, y / DX)


EDGE_LO, EDGE_HI = 0.5 * DX, (NX - 0.5) * DX      # both staggered lattices cover this square


def vbar(u, v, cx, cy, radii, naz):
    th = (np.arange(naz) + 0.5) * 2 * np.pi / naz
    X = cx + radii[:, None] * np.cos(th)[None, :]
    Y = cy + radii[:, None] * np.sin(th)[None, :]
    uu, vv = uv_at(u, v, X, Y)
    return (-np.sin(th)[None, :] * uu + np.cos(th)[None, :] * vv).mean(axis=1)


def ravail(cx, cy):
    return min(1000.0, cx - EDGE_LO, EDGE_HI - cx, cy - EDGE_LO, EDGE_HI - cy)


def peak(u, v, cx, cy):
    ra = ravail(cx, cy)
    radii = np.arange(4.0, ra + 1e-9, 4.0)
    radii = radii[radii >= 8.0]
    if len(radii) < 1:
        return -np.inf
    return float(vbar(u, v, cx, cy, radii, 96).max())


def my_centre(u, v, x0, y0):
    """Exhaustive search of the peak azimuthal-mean tangential wind: 2 m lattice +-24 m, then 0.5 m lattice +-2 m."""
    best = (peak(u, v, x0, y0), x0, y0)
    for half, step in ((24.0, 2.0), (2.0, 0.5)):
        cx, cy = best[1], best[2]
        for dx in np.arange(-half, half + 1e-9, step):
            for dy in np.arange(-half, half + 1e-9, step):
                f = peak(u, v, cx + dx, cy + dy)
                if f > best[0]:
                    best = (f, cx + dx, cy + dy)
    return best[1], best[2]


def describe(u, v, cx, cy, t):
    ra = ravail(cx, cy)
    radii = np.arange(2.0, ra + 1e-9, 2.0)
    radii = radii[radii >= 8.0 - 1e-9]
    vb = vbar(u, v, cx, cy, radii, 360)
    k = int(np.argmax(vb))
    vmax, rmw = float(vb[k]), float(radii[k])
    beyond = (radii > rmw) & (radii <= min(CLOSE_R * rmw, ra) + 1e-9)
    clos = float(vb[beyond].min() / vmax) if beyond.any() else float("nan")
    th = (np.arange(360) + 0.5) * 2 * np.pi / 360
    uu, vv = uv_at(u, v, cx + rmw * np.cos(th), cy + rmw * np.sin(th))
    vt = -np.sin(th) * uu + np.cos(th) * vv
    wn2 = 2 * abs(np.mean(vt * np.exp(-2j * th))) / vmax
    # ground-relative wind at scalar points: frame-relative C-grid averages + (umove, vmove) (CM1 base.F frame)
    uc = 0.5 * (u[:, :-1] + u[:, 1:]) + UM
    vc = 0.5 * (v[:-1, :] + v[1:, :]) + VM
    xs = (np.arange(NX) + 0.5) * DX
    r2 = (xs[None, :] - cx) ** 2 + (xs[:, None] - cy) ** 2
    gs = np.hypot(uc, vc)
    gmax = float(gs[r2 <= (HAZ_R * rmw) ** 2].max())
    judged = 100.0 if t < NEST_START + SPINUP - 1e-9 else DX
    zeta = 2 * vmax / rmw
    delta = 0.5 * judged / rmw

    def g(val, thr):
        return "ok" if val >= thr else ("marg" if val >= thr * (1 - delta) else "fail")
    st = {"R": g(vmax, VT_MIN), "C": "ok" if (beyond.any() and clos <= CLOSE_F) else "fail",
          "A": "ok" if wn2 <= WN2_MAX else "fail", "Z": g(zeta, ZETA_MIN), "W": g(gmax, GW_MIN),
          "S": "ok" if rmw >= RES_DX * judged - 1e-9 else "fail"}
    # E: child interior (boundary 5 cells) less 300 m, in window-local metres; the window is at i0 = j0 = 137
    lo, hi = BOUNDARY * DX + EDGE - 137 * DX, (CHILD_N - BOUNDARY) * DX - EDGE - 137 * DX
    st["E"] = "ok" if (lo <= cx <= hi and lo <= cy <= hi and EDGE_LO <= cx <= EDGE_HI and EDGE_LO <= cy <= EDGE_HI) else "fail"
    if any(st[c] == "fail" for c in "ERCAZW"):
        cls = "N"
    elif st["S"] == "ok" and all(st[c] == "ok" for c in "RZW"):
        cls = "Q"
    else:
        cls = "U"
    return {"cx": cx, "cy": cy, "vmax": vmax, "rmw": rmw, "closure": clos, "wn2": float(wn2), "zeta": zeta,
            "gwind": gmax, "class": cls, "judged_dx": judged, "states": st}


def samples_rows(label):
    import gzip
    p = f"{RUNS}/{label}/samples.tsv"
    lines = open(p).read().rstrip("\n").split("\n")
    head = lines[0].split("\t")
    return {round(float(ln.split("\t")[1]), 4): dict(zip(head, ln.split("\t"))) for ln in lines[1:]}


def cmd_records(times):
    tr = trajectory()
    labels = ("E100R2-PF-4m-window-pre3312", "E100R2-PF-4m-window-3312on", "E100R2-PF-4m-window")
    rows = {lab: samples_rows(lab) for lab in labels}
    for t in times:
        r = row_at(tr, t)
        ua, va, wa, ok = read_record(r)
        u, v = level10(ua, va)
        key = round(t, 4)
        srows = {lab: rows[lab][key] for lab in labels if key in rows[lab]}
        lab0 = next(iter(srows))
        s = srows[lab0]
        same_across = len({tuple(x.get(c) for c in ("class", "centre_x_m", "centre_y_m", "vmax_m_s", "rmw_m",
                                                       "ground_wind_max_m_s")) for x in srows.values()}) == 1
        sx, sy = float(s["centre_x_m"]), float(s["centre_y_m"])
        at_sc = describe(u, v, sx, sy, t)
        mx, my = my_centre(u, v, sx, sy)
        at_me = describe(u, v, mx, my, t)
        # ground position of the scorer's centre: local + origin + umove t
        gx = sx + float(r["origin_x_m"]) + UM * t
        gy = sy + float(r["origin_y_m"]) + VM * t
        out = {"t": t, "l0_index": r["l0_index"], "arena": r["arena"].split("/")[-2], "sha256_ok": ok,
               "origin": [float(r["origin_x_m"]), float(r["origin_y_m"])],
               "scorer": {k: s[k] for k in ("class", "conditions", "centre_x_m", "centre_y_m", "vmax_m_s", "rmw_m",
                                             "closure_ratio", "wavenumber2_fraction", "mean_zeta_core_s_inv",
                                             "ground_wind_max_m_s", "judged_dx_m", "centre_ground_x_m",
                                             "centre_ground_y_m", "q_run_s")},
               "rows_identical_across_spans": same_across, "spans_holding": list(srows),
               "mine_at_scorer_centre": at_sc, "my_centre_search": at_me,
               "ground_centre_mine": [gx, gy],
               "ground_centre_diff_m": [gx - float(s["centre_ground_x_m"]), gy - float(s["centre_ground_y_m"])],
               "u10_level_check": {"u_k2_k3_k4_at_centre_cell": [float(ua[k, NG + int(sy // DX), NG + int(sx // DX)])
                                                                 for k in (2, 3, 4)]}}
        print(json.dumps(out, default=float))
        sys.stdout.flush()


def cmd_courant(t0, t1, stride):
    tr = trajectory()
    # mh at scalar levels: dz_ref / (zf(k+1)-zf(k)); the lowest 25 layers are 4 m -> 25 (placement mf_cm1_indexed)
    faces = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 68.0,
             72.0, 76.0, 80.0, 84.0, 88.0, 92.0, 96.0, 100.0, 106.0, 114.0, 124.0, 136.0, 150.0, 166.0, 184.0, 200.0,
             220.0, 248.2051239013672, 284.6153869628906, 329.23077392578125, 382.05126953125, 443.0769348144531,
             512.3077392578125, 589.7435913085938, 675.3846435546875, 769.2307739257812, 871.2820434570312,
             981.53857421875]
    dz = np.diff(np.array(faces))[:NZ]
    dt = 0.03125
    sel = [r for r in tr if t0 - 1e-9 <= r["t"] <= t1 + 1e-9]
    sel = sel[::stride]
    for r in sel:
        ua, va, wa, _ = read_record(r, verify=False)
        a = ua[1:NZ + 1, NG:NG + NY, NG:NG + NX + 1].astype(np.float64)
        b = va[1:NZ + 1, NG:NG + NY + 1, NG:NG + NX].astype(np.float64)
        w = wa[1:NZ + 2, NG:NG + NY, NG:NG + NX].astype(np.float64)
        cx = (a[:, :, :-1] + a[:, :, 1:]) * 0.5 * dt / DX
        cy = (b[:, :-1, :] + b[:, 1:, :]) * 0.5 * dt / DX
        cz = (w[:-1] + w[1:]) * 0.5 * dt / dz[:, None, None]
        c = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        k, j, i = np.unravel_index(int(np.argmax(c)), c.shape)
        print(json.dumps({"t": r["t"], "cfl": float(c.max()), "at_kji": [int(k) + 1, int(j), int(i)],
                          "vh_max_frame": float(np.hypot(0.5 * (a[:, :, :-1] + a[:, :, 1:]),
                                                         0.5 * (b[:, :-1, :] + b[:, 1:, :])).max())}))
        sys.stdout.flush()


def cmd_layout(t):
    """Window uadv/vadv at 10 m vs the child's u3d/v3d history frame at the same model time."""
    import glob
    tr = trajectory()
    r = row_at(tr, t)
    ua, va, _, ok = read_record(r)
    frame = None
    for p in sorted(glob.glob(S3 + "/frames/nest-child/placements/frame-*.json")):
        pl = json.load(open(p))
        if abs(pl["model_time_s"] - t) < 1e-6:
            frame = (p, pl)
    p, pl = frame
    fdir = p.replace("/placements/", "/").replace(".json", "")
    tab = {}
    for ln in open(fdir + "/history-field-table.tsv"):
        c = ln.rstrip("\n").split("\t")
        tab[c[0]] = (tuple(int(x) for x in c[4].split(",")), int(c[7]))
    arena = np.memmap(fdir + "/history-carriers.f32", dtype="<f4", mode="r")

    def fld(n):
        shp, off = tab[n]
        return arena[off // 4: off // 4 + math.prod(shp)].reshape(shp)
    u3, v3 = fld("u3d"), fld("v3d")
    i0, j0 = int(r["i0"]), int(r["j0"])
    res = {"t": t, "frame": fdir.split("/")[-1], "sha_ok": ok,
           "window_origin_minus_child_origin_m": [float(r["origin_x_m"]) - pl["origin_x_m"] - i0 * DX,
                                                  float(r["origin_y_m"]) - pl["origin_y_m"] - j0 * DX],
           "domainloc_over_t": [pl["domainlocx_m"] / t, pl["domainlocy_m"] / t]}
    for k in (1, 3, 10):
        uw = ua[k, NG:NG + NY, NG:NG + NX + 1].astype(np.float64)
        vw = va[k, NG:NG + NY + 1, NG:NG + NX].astype(np.float64)
        best = None
        oi = int(round(res["window_origin_minus_child_origin_m"][0] / DX))
        oj = int(round(res["window_origin_minus_child_origin_m"][1] / DX))
        cand = sorted({(a, b) for a in (-1, 0, 1, oi - 1, oi, oi + 1) for b in (-1, 0, 1, oj - 1, oj, oj + 1)})
        for si, sj in cand:
            if True:
                if not (0 <= j0 + sj and j0 + sj + NY + 1 <= CHILD_N + 1 and 0 <= i0 + si and i0 + si + NX + 1 <= CHILD_N + 1):
                    continue
                uc = u3[k - 1, j0 + sj:j0 + sj + NY, i0 + si:i0 + si + NX + 1].astype(np.float64)
                vc = v3[k - 1, j0 + sj:j0 + sj + NY + 1, i0 + si:i0 + si + NX].astype(np.float64)
                rms = float(np.sqrt(np.mean((uw - uc) ** 2) + np.mean((vw - vc) ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, si, sj)
                if si == 0 and sj == 0:
                    rms0 = rms
        res[f"k{k}"] = {"rms_diff_at_i0j0": rms0, "best_shift": best[1:], "best_rms": best[0],
                        "max_abs_u_window": float(np.abs(uw).max()), "max_abs_u_frame": float(np.abs(u3[k - 1]).max())}
    # the ghost convention: uadv level index 0 is k=0
    res["uadv_k0_vs_k1_rms"] = float(np.sqrt(np.mean((ua[0].astype(np.float64) - ua[1]) ** 2)))
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "index":
        cmd_index()
    elif c == "records":
        cmd_records([float(x) for x in sys.argv[2:]])
    elif c == "courant":
        cmd_courant(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    elif c == "layout":
        for x in sys.argv[2:]:
            cmd_layout(float(x))
