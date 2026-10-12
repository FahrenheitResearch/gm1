"""Independent verifier: re-tracking of the criterion-v2 candidates (numpy/stdlib only; no scorer import).

  python3 vs_track.py RUNS_DIR
Reads each span's candidates.tsv and samples.tsv as data, rebuilds the identity-linked Q chains with its own code
(rule text: docs/GM1-TORNADO-CRITERION-V2.md, time-series section), and runs labelled sensitivity cases.
"""
import json
import math
import sys

import numpy as np

UM = VM = 16.5
SPEED, JUMP, BRIDGE, SUSTAIN, MAXDT = 60.0, 100.0, 3.0, 120.0, 60.0
SPANS = {"E100R2-PF-4m-window-pre3312": 2160.09375, "E100R2-PF-4m-window-3312on": 3312.0,
         "E100R2-PF-4m-window": 2160.09375}
CAD = 0.09375


def load(runs, label):
    t0 = SPANS[label]
    rows = open(f"{runs}/{label}/candidates.tsv").read().rstrip("\n").split("\n")
    head = rows[0].split("\t")
    cands = []
    for ln in rows[1:]:
        d = dict(zip(head, ln.split("\t")))
        k = int(d["sample"])
        t = t0 + CAD * k
        assert abs(t - float(d["model_time_s"])) < 6e-5, (k, t, d["model_time_s"])
        cands.append((k, t, d["class"], float(d["centre_ground_x_m"]), float(d["centre_ground_y_m"]), float(d["rmw_m"]),
                      float(d["vmax_m_s"]), float(d["ground_wind_max_m_s"]), float(d["centre_x_m"]), float(d["centre_y_m"])))
    srows = open(f"{runs}/{label}/samples.tsv").read().rstrip("\n").split("\n")
    sh = srows[0].split("\t")
    samples = [dict(zip(sh, ln.split("\t"))) for ln in srows[1:]]
    return cands, samples


def track(cands, nsamp, t0, keep):
    """start[c], pred[c] for every candidate c; only candidates with keep(c) take part (Q chains)."""
    by_k = [[] for _ in range(nsamp)]
    for i, c in enumerate(cands):
        by_k[c[0]].append(i)
    start = np.full(len(cands), np.nan)
    pred = np.full(len(cands), -1, dtype=np.int64)
    gx = np.array([c[3] for c in cands])
    gy = np.array([c[4] for c in cands])
    rmw = np.array([c[5] for c in cands])
    for k in range(nsamp):
        tk = t0 + CAD * k
        for iq in by_k[k]:
            if not keep(cands[iq]):
                continue
            st, pr = tk, -1
            m = k - 1
            while m >= 0:
                dt = tk - (t0 + CAD * m)
                if dt > MAXDT + 1e-6 or (m < k - 1 and dt > BRIDGE + 1e-6):
                    break
                for ip in by_k[m]:
                    if not math.isfinite(start[ip]) or start[ip] >= st:
                        continue
                    dxg, dyg = gx[iq] - gx[ip], gy[iq] - gy[ip]
                    if math.hypot(dxg, dyg) > SPEED * dt + JUMP:
                        continue
                    if math.hypot(dxg - UM * dt, dyg - VM * dt) <= rmw[ip] + rmw[iq]:
                        st, pr = start[ip], ip
                m -= 1
            start[iq], pred[iq] = st, pr
    return start, pred


def summary(cands, start, pred, t0):
    t = np.array([c[1] for c in cands])
    dur = np.where(np.isfinite(start), t - start, -1.0)
    ok = dur >= SUSTAIN - 1e-6
    out = {"longest_s": float(dur.max()), "tornado": bool(ok.any()),
           "genesis_s": float(start[ok].min()) if ok.any() else None}
    if ok.any():
        # the longest chain ending at the last time it reaches: follow predecessors back
        end = int(np.argmax(np.where(ok, t + 1e-9 * dur, -1)))
        mem = []
        n = end
        while n >= 0:
            mem.append(n)
            n = int(pred[n])
        mem.reverse()
        r = np.array([cands[m][5] for m in mem])
        tt = np.array([cands[m][1] for m in mem])
        g = np.array([cands[m][7] for m in mem])
        vm = np.array([cands[m][6] for m in mem])
        circ = 2 * np.pi * r * vm
        hops = []
        for a, b in zip(mem, mem[1:]):
            dt = cands[b][1] - cands[a][1]
            rel = math.hypot(cands[b][3] - cands[a][3] - UM * dt, cands[b][4] - cands[a][4] - VM * dt)
            hops.append((rel, cands[a][5], cands[b][5], dt))
        hops = np.array(hops)
        beyond_max = hops[:, 0] > np.maximum(hops[:, 1], hops[:, 2])
        out["chain"] = {"start_s": float(tt[0]), "end_s": float(tt[-1]), "members": len(mem),
                        "max_gap_s": float(np.diff(tt).max()), "rmw_median": float(np.median(r)),
                        "rmw_min": float(r.min()), "rmw_le24": int((r <= 24).sum()),
                        "rmw_le24_after_first_60s": int(((r <= 24) & (tt > tt[0] + 60)).sum()),
                        "rmw_p5_p95": [float(np.percentile(r, 5)), float(np.percentile(r, 95))],
                        "ground_wind_peak": float(g.max()), "ground_wind_peak_t": float(tt[int(np.argmax(g))]),
                        "vmax_peak": float(vm.max()), "vmax_peak_t": float(tt[int(np.argmax(vm))]),
                        "circulation_median": float(np.median(circ)), "circulation_peak": float(circ.max()),
                        "circulation_peak_t": float(tt[int(np.argmax(circ))]), "rmw_max": float(r.max()),
                        "hops_rel_move_max_m": float(hops[:, 0].max()),
                        "hops_beyond_larger_rmw": int(beyond_max.sum()),
                        "first_10_members": [[round(cands[m][1], 5), cands[m][5], round(cands[m][6], 2)] for m in mem[:10]]}
    return out


def main(runs):
    rep = {}
    for label, t0 in SPANS.items():
        cands, samples = load(runs, label)
        ns = len(samples)
        cls = {}
        for s in samples:
            cls[s["class"]] = cls.get(s["class"], 0) + 1
        base_start, base_pred = track(cands, ns, t0, lambda c: c[2] == "Q")
        r = {"samples": ns, "candidates": len(cands), "sample_classes": cls,
             "frozen_rule": summary(cands, base_start, base_pred, t0)}
        # sensitivity (NOT the rule): candidates with RMW <= 24 m removed
        s1, p1 = track(cands, ns, t0, lambda c: c[2] == "Q" and c[5] > 24.0)
        r["sens_no_rmw_le24"] = summary(cands, s1, p1, t0)
        # sensitivity (NOT the rule): identity only when the move is within the larger RMW (not the sum)
        r["sens_note"] = "sensitivity cases are not criterion v2; they test attack 2 only"
        if label.endswith("pre3312"):
            # sensitivity: span cut at 3300.0 s (Courant 0.70 there)
            keep = [i for i, c in enumerate(cands) if c[1] <= 3300.0 + 1e-9]
            sub = [cands[i] for i in keep]
            ns3 = int(round((3300.0 - t0) / CAD)) + 1
            s3, p3 = track(sub, ns3, t0, lambda c: c[2] == "Q")
            r["sens_cut_3300"] = summary(sub, s3, p3, t0)
        # origin jumps: row-centre ground move vs local move across the window's 100 m jumps
        lx = np.array([float(s["centre_x_m"]) if s["centre_x_m"] else np.nan for s in samples])
        ly = np.array([float(s["centre_y_m"]) if s["centre_y_m"] else np.nan for s in samples])
        gx = np.array([float(s["centre_ground_x_m"]) if s["centre_ground_x_m"] else np.nan for s in samples])
        gy = np.array([float(s["centre_ground_y_m"]) if s["centre_ground_y_m"] else np.nan for s in samples])
        q = np.array([s["class"] == "Q" for s in samples])
        # origin = ground - local - umove t
        tt = t0 + CAD * np.arange(ns)
        ox = np.round(gx - lx - UM * tt, 2)
        oy = np.round(gy - ly - VM * tt, 2)
        jump = (np.abs(np.diff(ox)) > 50) | (np.abs(np.diff(oy)) > 50)
        both = q[1:] & q[:-1]
        gm = np.hypot(np.diff(gx) - UM * CAD, np.diff(gy) - VM * CAD)
        lm = np.hypot(np.diff(lx), np.diff(ly))
        r["origin_jumps"] = {"n": int(jump.sum()), "n_with_Q_both_sides": int((jump & both).sum()),
                             "frame_rel_move_at_jumps_median_m": float(np.median(gm[jump & both])) if (jump & both).any() else None,
                             "frame_rel_move_at_jumps_max_m": float(np.max(gm[jump & both])) if (jump & both).any() else None,
                             "local_move_at_jumps_median_m": float(np.median(lm[jump & both])) if (jump & both).any() else None,
                             "frame_rel_move_elsewhere_median_m": float(np.median(gm[~jump & both])),
                             "frame_rel_move_elsewhere_p99_m": float(np.percentile(gm[~jump & both], 99))}
        rep[label] = r
        print(label, json.dumps(r, default=float), flush=True)
    json.dump(rep, open("track.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])
