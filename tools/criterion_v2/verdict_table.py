"""OFFLINE, CPU-ONLY: collect criterion v2 verdicts into the verdict table and the machine-readable file.

    python3 tools/criterion_v2/verdict_table.py --runs DIR --json OUT.json --markdown OUT.md
        [--root KEY=DIR ...] [--code DIR]

Reads what ``score_runs.py`` wrote in ``DIR`` (``<label>/verdict.json``,
``samples.tsv``, ``candidates.tsv``, ``<label>.not-scored.json``) for every
source in the scoring manifest (``scoring-sources-v2.json``) and writes, per run and source, the v2
verdict beside the v1 verdict, the key numbers, and the verdict per time
window.  Nothing here decides a verdict: outcomes, captions, episodes and
genesis are copied from ``verdict.json``.  The only computation is
bookkeeping: the members of each TORNADO episode are recovered with the
frozen rule's own tracking (``v2rule._chains`` on the candidates as written,
checked against the episode ``verdict.json`` reports), so that the episode's
circulation can be given; the frame velocity comes from the source metadata
(it must be readable where this runs).

``--code DIR`` is the checkout whose ``tools/criterion_v2`` holds the frozen
``v2rule.py``/``score.py`` the verdicts were made with (default: this one).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def fmt_t(t: float | None) -> str:
    if t is None or not math.isfinite(t):
        return "-"
    m = int(t // 60)
    s = t - 60 * m
    return f"T+{m}:{int(round(s)):02d}" if abs(s - round(s)) < 1e-6 else f"T+{m}:{s:04.1f}"


def read_tsv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln]


def num(x: str | None) -> float:
    try:
        return float(x) if x not in (None, "", "nan") else math.nan
    except ValueError:
        return math.nan


def frame_velocity(sc, kind: str, path: Path, deck: Path | None) -> tuple[float, float] | None:
    """The domain's own motion (umove, vmove) as the scorer read it; ``None`` when not readable here."""
    try:
        if kind == "child":
            jobs = sc.child_jobs(path)
        elif kind == "window":
            jobs = sc.window_jobs(path)
        else:
            dx, dy = sc.deck_spacing(deck)
            jobs = sc.parent_jobs(path, dx, dy)
    except Exception:  # noqa: BLE001 - the source is not on this machine
        return None
    vel = {(j["umove"], j["vmove"]) for j in jobs}
    return vel.pop() if len(vel) == 1 else None


def episode_members(rule, samples: list[dict[str, str]], cands: list[dict[str, str]], vel: tuple[float, float],
                    episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each reported episode's member candidates, from the frozen rule's own chain tracking."""
    per: dict[int, list[dict[str, Any]]] = {int(s["sample"]): [] for s in samples}
    for c in cands:
        per[int(c["sample"])].append({"class": c["class"], "centre_ground_x_m": num(c["centre_ground_x_m"]),
                                      "centre_ground_y_m": num(c["centre_ground_y_m"]), "rmw_m": num(c["rmw_m"]),
                                      "vmax_m_s": num(c["vmax_m_s"]), "ground_wind_max_m_s": num(c["ground_wind_max_m_s"]),
                                      "model_time_s": num(c["model_time_s"])})
    ks = sorted(per)
    ss = [{"model_time_s": num(samples[k]["model_time_s"]), "umove": vel[0], "vmove": vel[1], "candidates": per[k]}
          for k in ks]
    t = [s["model_time_s"] for s in ss]
    ch = rule._chains(ss, t, ("Q",), True)
    extended = {pred for row in ch for _, pred in row if pred is not None}
    out = []
    used: set[tuple[int, int]] = set()
    for ep in episodes:
        found = None
        for k, row in enumerate(ch):
            if abs(t[k] - ep["end_s"]) > 1e-3:
                continue
            for ic, (st, _) in enumerate(row):
                if (math.isfinite(st) and abs(st - ep["start_s"]) < 1e-3 and (k, ic) not in extended
                        and (k, ic) not in used):
                    keys, node = [], (k, ic)
                    while node is not None:
                        keys.append(node)
                        node = ch[node[0]][node[1]][1]
                    keys.reverse()
                    if len(keys) == ep["samples"]:
                        found = keys
                        used.add((k, ic))
                        break
            if found:
                break
        if not found:
            out.append({"recovered": False})
            continue
        keys = found
        found = [ss[m]["candidates"][i] for m, i in keys]
        circ = [2 * math.pi * m["rmw_m"] * m["vmax_m_s"] for m in found]
        kmax = max(range(len(found)), key=lambda i: circ[i])
        out.append({
            "recovered": True, "samples": len(found), "member_keys": keys,
            "circulation_median_m2_s": sorted(circ)[len(circ) // 2],
            "circulation_peak_m2_s": circ[kmax], "circulation_peak_time_s": found[kmax]["model_time_s"],
            "circulation_range_m2_s": [min(circ), max(circ)],
            "vmax_peak_m_s": max(m["vmax_m_s"] for m in found),
            "ground_wind_peak_m_s": max(m["ground_wind_max_m_s"] for m in found),
            "ground_wind_peak_time_s": max(found, key=lambda m: m["ground_wind_max_m_s"])["model_time_s"],
            "rmw_range_m": [min(m["rmw_m"] for m in found), max(m["rmw_m"] for m in found)],
            "rmw_median_m": sorted(m["rmw_m"] for m in found)[len(found) // 2],
            "matches_verdict": (abs(max(m["vmax_m_s"] for m in found) - ep["vmax_peak_m_s"]) < 1e-3
                                and abs(max(m["ground_wind_max_m_s"] for m in found) - ep["ground_wind_peak_m_s"]) < 1e-3),
        })
    return out


def group_events(episodes: list[dict[str, Any]], eps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Episodes that share a member candidate are one tracked vortex (branches of one chain); group them.

    ``verdict.json`` lists every maximal chain; when several candidates of
    one vortex are kept per sample, overlapping chains share members.  Each
    event reports its longest chain (``primary``) and how many chains it holds.
    """
    n = len(episodes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Two episodes are joined when they share a member candidate.  Joining every episode that holds a key
    # to the first episode seen with it gives the same groups as testing every pair (the connected
    # components of the shared-member graph) in time linear in the members, which matters when a
    # sub-second window stream lists thousands of overlapping chains.
    owner: dict[tuple[int, int], int] = {}
    for i, e in enumerate(eps[:n]):
        if not e.get("recovered"):
            continue
        for key in map(tuple, e.get("member_keys", [])):
            j = owner.setdefault(key, i)
            if j != i:
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    events = []
    for idx in groups.values():
        prim = max(idx, key=lambda i: (episodes[i]["span_s"], -episodes[i]["start_s"]))
        m = eps[prim] if prim < len(eps) else {}
        events.append({"start_s": min(episodes[i]["start_s"] for i in idx),
                       "end_s": max(episodes[i]["end_s"] for i in idx),
                       "chains": len(idx), "primary": {**episodes[prim], **{k: v for k, v in m.items() if k != "member_keys"}}})
    events.sort(key=lambda e: e["start_s"])
    return events


def class_segments(samples: list[dict[str, str]]) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    for s in samples:
        t, c = num(s["model_time_s"]), s["class"]
        if segs and segs[-1]["class"] == c:
            segs[-1]["end_s"] = t
            segs[-1]["samples"] += 1
        else:
            segs.append({"class": c, "start_s": t, "end_s": t, "samples": 1})
    return segs


def window_numbers(samples: list[dict[str, str]], t0: float, t1: float) -> dict[str, Any]:
    rows = [s for s in samples if t0 - 1e-6 <= num(s["model_time_s"]) <= t1 + 1e-6]
    if not rows:
        return {"samples": 0}
    best = max(rows, key=lambda s: num(s["ground_wind_max_m_s"]) if math.isfinite(num(s["ground_wind_max_m_s"])) else -1)
    counts: dict[str, int] = {}
    for s in rows:
        counts[s["class"]] = counts.get(s["class"], 0) + 1
    fin = [num(s["vmax_m_s"]) for s in rows if math.isfinite(num(s["vmax_m_s"]))]
    return {"samples": len(rows), "class_counts": counts,
            "ground_wind_peak_m_s": num(best["ground_wind_max_m_s"]), "ground_wind_peak_time_s": num(best["model_time_s"]),
            "rmw_at_peak_m": num(best["rmw_m"]), "circulation_at_peak_m2_s": num(best["circulation_rmw_m2_s"]),
            "vmax_peak_m_s": max(fin) if fin else None}


def time_windows(v: dict[str, Any], samples: list[dict[str, str]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The record cut at the edges of each tornado event (grouped episodes), each with its state and numbers."""
    if not samples:
        return []
    t_first, t_last = num(samples[0]["model_time_s"]), num(samples[-1]["model_time_s"])
    cuts: list[tuple[float, float, str, int | None]] = []
    spin = (v.get("nest_spinup") or {}).get("spinup_end_s")
    if v["outcome"] == "TORNADO":
        cursor = t_first
        for n, ev in enumerate(events, 1):
            if ev["start_s"] > cursor + 1e-6:
                cuts.append((cursor, ev["start_s"], "before genesis" if n == 1 else f"between tornado {n - 1} and {n}",
                             None))
            cuts.append((ev["start_s"], ev["end_s"], f"tornado {n}", n - 1))
            cursor = max(cursor, ev["end_s"])
        if t_last > cursor + 1e-6:
            cuts.append((cursor, t_last, "after the tornado", None))
    else:
        cuts.append((t_first, t_last, "whole record", None))
    out = []
    for a, b, state, k in cuts:
        rec: dict[str, Any] = {"start_s": a, "end_s": b, "start": fmt_t(a), "end": fmt_t(b), "state": state,
                               "caption": v["caption"] if (k is not None or v["outcome"] != "TORNADO") else None,
                               "numbers": window_numbers(samples, a, b)}
        if k is not None:
            rec["tornado"] = events[k]
            if abs(a - t_first) < 1e-6:
                rec["genesis_note"] = ("the record opens with this vortex already qualifying: genesis is at or "
                                       "before the first sample")
        if spin is not None and a < spin - 1e-6:
            rec["spinup_note"] = (f"samples before {fmt_t(spin)} are judged at the parent spacing "
                                  f"({v['nest_spinup']['parent_dx_m']:g} m): nest spin-up")
        out.append(rec)
    return out


def summarise(src: dict[str, Any], runs: Path, rule, sc, roots: dict[str, str], resolve) -> dict[str, Any]:
    label = src["label"]
    base = {"label": label, "run": src["run"], "kind": src["kind"], "grid": src["grid"], "note": src.get("note")}
    vp = runs / label / "verdict.json"
    if not vp.is_file():
        ns = runs / f"{label}.not-scored.json"
        why = json.loads(ns.read_text())["reason"] if ns.is_file() else "not run"
        return {**base, "status": "not scored", "reason": why}
    v = json.loads(vp.read_text())
    samples = read_tsv(runs / label / "samples.tsv")
    cands = read_tsv(runs / label / "candidates.tsv")
    path = resolve(src["path"], roots)
    deck = resolve(src["deck"], roots) if src.get("deck") else None
    vel = frame_velocity(sc, src["kind"], path, deck)
    eps = episode_members(rule, samples, cands, vel, v["episodes"]) if (v["episodes"] and vel) else []
    events = group_events(v["episodes"], eps) if eps else []
    intervals = sorted({round(num(b["model_time_s"]) - num(a["model_time_s"]), 4) for a, b in zip(samples, samples[1:])})
    whole = window_numbers(samples, -math.inf, math.inf)
    run_json = runs / f"{label}.run.json"
    return {
        **base, "status": "scored",
        "source_path": v["source"]["path"], "samples": v["samples"],
        "record": [v["record_start_s"], v["record_end_s"]], "record_T": [fmt_t(v["record_start_s"]), fmt_t(v["record_end_s"])],
        "cadence_s": v["median_interval_s"], "intervals_s": intervals[:6], "max_interval_s": v["max_interval_s"],
        "dx_m": v["grid"]["dx_m"], "lowest_scalar_level_m": v["grid"]["lowest_scalar_level_m"],
        "v2": {k: v.get(k) for k in ("outcome", "caption", "reason", "genesis_model_time_s", "episodes",
                                      "longest_qualifying_run_s", "longest_qualifying_or_unresolved_run_s",
                                      "longest_qualifying_run_plausible_links_s", "reported_60s_rule",
                                      "class_counts", "condition_counts", "marginal_counts", "flag_counts",
                                      "cadence_statement")},
        "genesis_T": fmt_t(v.get("genesis_model_time_s")),
        "v1": v.get("v1_beside"),
        "tornado_events": events,
        "episodes_recovered": (all(e.get("recovered") and e.get("matches_verdict") for e in eps)
                               if v["episodes"] else None),
        "frame_velocity_m_s": vel,
        "whole_record_numbers": whole,
        "class_segments": [{**s, "start": fmt_t(s["start_s"]), "end": fmt_t(s["end_s"])} for s in class_segments(samples)],
        "time_windows": time_windows(v, samples, events),
        "integrity": {"criterion_v2_sha256": v["criterion_v2_sha256"], "files_sha256": v["files_sha256"],
                      "matches_frozen": v["matches_frozen"], "protocol_conforming": v["protocol_conforming"],
                      "scorer_commit": v["scorer_commit"], "nest_spinup": v.get("nest_spinup"),
                      "source": v["source"],
                      "run": json.loads(run_json.read_text()) if run_json.is_file() else None},
    }


def v1_text(rec: dict[str, Any], by_label: dict[str, dict[str, Any]]) -> str:
    v1 = rec.get("v1")
    if rec["kind"] == "parent":
        return "- (v1 judges nest children only)"
    if v1 is None and rec["kind"] == "window":
        child = by_label.get(rec["label"].replace("-window", "-child"), {})
        v1 = child.get("v1")
        if v1 and "verdict" in v1:
            return f"{v1['verdict']} (the run's v1, {v1_basis(v1)})"
        return "-"
    if not v1:
        return "-"
    if "error" in v1:
        return f"error: {v1['error']}"
    g = f", genesis {fmt_t(v1['genesis_model_time_s'])}" if v1.get("genesis_model_time_s") else ""
    return f"{v1['verdict']}{g} ({v1_basis(v1)})"


def v1_basis(v1: dict[str, Any]) -> str:
    """v1's own basis: the live sidecar (every parent step) or v1's host sample recomputed on the child frames."""
    return ("live, every parent step" if (v1.get("source") or "").startswith("live v1 sidecar")
            else "recomputed on the child frames")


def f0(x: Any, unit: str = "") -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.0f}{unit}"


def row_numbers(rec: dict[str, Any]) -> tuple[str, str, str, str]:
    v2 = rec["v2"]
    if v2["outcome"] == "TORNADO" and rec["tornado_events"]:
        evs = rec["tornado_events"]
        e = max(evs, key=lambda ev: ev["primary"]["span_s"])["primary"]
        wind = f"{e['ground_wind_peak_m_s']:.1f} m/s at {fmt_t(e['ground_wind_peak_time_s'])}"
        rmw = f"median {e['rmw_median_m']:.0f} m (range {e['rmw_range_m'][0]:.0f}-{e['rmw_range_m'][1]:.0f} m)"
        circ = (f"median {e['circulation_median_m2_s'] / 1000:.1f}e3, peak {e['circulation_peak_m2_s'] / 1000:.1f}e3 "
                f"m2/s at {fmt_t(e['circulation_peak_time_s'])}")
        dur = (f"{e['span_s']:g} s ({e['samples']} samples, {fmt_t(e['start_s'])}-{fmt_t(e['end_s'])}); "
               f"{len(evs)} tracked tornado(es)")
        return wind, rmw, circ, dur
    w = rec["whole_record_numbers"]
    ok = bool(w.get("samples"))
    wind = (f"{w['ground_wind_peak_m_s']:.1f} m/s at {fmt_t(w['ground_wind_peak_time_s'])}"
            if ok and math.isfinite(w["ground_wind_peak_m_s"]) else "-")
    rmw = f"{w['rmw_at_peak_m']:.0f} m (at the peak-wind sample)" if ok and math.isfinite(w["rmw_at_peak_m"]) else "-"
    circ = (f"{w['circulation_at_peak_m2_s'] / 1000:.1f}e3 m2/s (at the peak-wind sample)"
            if ok and math.isfinite(w["circulation_at_peak_m2_s"]) else "-")
    dur = (f"longest tracked Q {v2['longest_qualifying_run_s']:g} s; Q or U "
           f"{v2['longest_qualifying_or_unresolved_run_s']:g} s; Q via plausible links "
           f"{v2['longest_qualifying_run_plausible_links_s']:g} s")
    return wind, rmw, circ, dur


def cadence_text(rec: dict[str, Any]) -> str:
    c = rec["cadence_s"]
    if c is None or (isinstance(c, float) and not math.isfinite(c)):
        return "one sample"
    what = "window stream" if rec["kind"] == "window" else "frames"
    extra = f", max gap {rec['max_interval_s']:g} s" if rec["max_interval_s"] and rec["max_interval_s"] > c + 1e-6 else ""
    return f"{c:g} s {what}{extra}; {rec['samples']} samples {rec['record_T'][0]}-{rec['record_T'][1]}"


KIND_TEXT = {"window": "window stream", "child": "child frames", "parent": "parent frames"}


def genesis_text(r: dict[str, Any]) -> str:
    if r["v2"]["outcome"] != "TORNADO":
        return "-"
    g = r["v2"]["genesis_model_time_s"]
    if g is not None and r["record"][0] is not None and abs(g - r["record"][0]) < 1e-6:
        return f"{fmt_t(g)} (first sample: at or before)"
    return fmt_t(g)


def run_verdicts(recs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per run: the verdict of its finest-cadence scored source that is not INSUFFICIENT_CADENCE.

    The pre-registration says a window stream decides when frames cannot show
    persistence (docs/GM1-TORNADO-CRITERION-V2.md); the span it covers is
    stated.  A run whose every scored source is INSUFFICIENT_CADENCE gets that.
    """
    out: dict[str, dict[str, Any]] = {}
    by_run: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        by_run.setdefault(r["run"], []).append(r)
    for run, rs in by_run.items():
        scored = [r for r in rs if r["status"] == "scored"]
        if not scored:
            out[run] = {"outcome": None, "basis": "no source scored",
                        "pending": [r["label"] for r in rs]}
            continue
        decisive = [r for r in scored if r["v2"]["outcome"] != "INSUFFICIENT_CADENCE"]
        pick = min(decisive or scored, key=lambda r: (r["cadence_s"] if r["cadence_s"] is not None and
                                                        math.isfinite(r["cadence_s"]) else math.inf))
        out[run] = {"outcome": pick["v2"]["outcome"], "caption": pick["v2"]["caption"], "source": pick["label"],
                    "span": pick["record_T"], "genesis": genesis_text(pick),
                    "basis": (f"{pick['label']} ({KIND_TEXT[pick['kind']]}, {pick['cadence_s']:g} s, "
                              f"{pick['record_T'][0]}-{pick['record_T'][1]})"
                              if pick["cadence_s"] is not None and math.isfinite(pick["cadence_s"])
                              else f"{pick['label']} (one sample)"),
                    "other_sources": {r["label"]: (r["v2"]["outcome"] if r["status"] == "scored" else "not scored")
                                      for r in rs if r is not pick},
                    "pending": [r["label"] for r in rs if r["status"] != "scored"]}
    return out


def markdown_table(recs: list[dict[str, Any]]) -> str:
    by_label = {r["label"]: r for r in recs}
    head = ("| Run (source) | Grid | Cadence | Old verdict (v1) | v2 verdict | Genesis (v2) | Peak 10 m AGL "
            "ground-relative wind | RMW | Circulation 2&pi;&middot;RMW&middot;Vmax | Duration | Notes |")
    lines = [head, "|" + "---|" * 11]
    for r in recs:
        name = f"{r['run']} ({KIND_TEXT[r['kind']]})"
        if r["status"] != "scored":
            why = "; ".join(x for x in (r.get("note"), r["reason"]) if x)
            lines.append(f"| {name} | {r['grid']} | - | - | **not scored** | - | - | - | - | - | {why} |")
            continue
        wind, rmw, circ, dur = row_numbers(r)
        v2 = r["v2"]
        verdict = f"**{v2['outcome']}**"
        notes = [r.get("note") or ""]
        if v2.get("reason"):
            notes.append(v2["reason"])
        cc = v2.get("class_counts") or {}
        notes.append("classes " + ", ".join(f"{k} {cc[k]}" for k in ("Q", "U", "N") if k in cc))
        if not r["integrity"]["protocol_conforming"] or not r["integrity"]["matches_frozen"]:
            notes.append("NOT protocol-conforming")
        grid = f"{r['dx_m']:g} m (lowest level {r['lowest_scalar_level_m']:g} m)"
        lines.append(f"| {name} | {grid} | {cadence_text(r)} | {v1_text(r, by_label)} | {verdict} | "
                     f"{genesis_text(r)} | {wind} | {rmw} | {circ} | {dur} | "
                     f"{'; '.join(n for n in notes if n)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--code", type=Path, default=HERE.parents[1])
    ap.add_argument("--manifest", type=Path, default=HERE / "scoring-sources-v2.json")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(HERE))
    import score_runs  # noqa: E402
    sys.path.insert(0, str(a.code / "tools" / "criterion_v2"))
    import importlib
    rule = importlib.import_module("v2rule")
    sc = importlib.import_module("score")
    roots, sources = score_runs.load_manifest(a.manifest)
    for r in a.root:
        k, _, val = r.partition("=")
        roots[k] = val
    recs = [summarise(s, a.runs, rule, sc, roots, score_runs.resolve) for s in sources]
    frozen = json.loads((a.code / "tools" / "criterion_v2" / "FROZEN.json").read_text())
    runs: dict[str, dict[str, Any]] = {}
    verdicts = run_verdicts(recs)
    for r in recs:
        runs.setdefault(r["run"], {"run_verdict": verdicts[r["run"]], "sources": []})["sources"].append(r)
    doc = {
        "schema": "gm1.tornado-verdicts/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "criterion": {"name": "GM1 tornado criterion v2 (pre-registered)", "doc": "docs/GM1-TORNADO-CRITERION-V2.md",
                      "freeze_commit": "198de287e3e304e32172003ad699c1d27e4211b3",
                      "frozen_utc": "2026-09-22T19:27:37Z", "frozen": frozen},
        "caption_wording": dict(rule.CAPTIONS),
        "time_convention": "T+m:ss is model time / 60 (T+70 = 4200 s)",
        "runs": runs,
    }
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(doc, indent=1, default=str, allow_nan=False) + "\n")
    a.markdown.write_text(markdown_table(recs))
    for r in recs:
        print(r["label"], r["status"], (r.get("v2") or {}).get("outcome"), r.get("reason", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
