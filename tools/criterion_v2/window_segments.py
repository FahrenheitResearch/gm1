"""OFFLINE, CPU-ONLY adapter: score a window stream kept in several segments, without a receipt, with the
frozen criterion v2 scorer.

    nice -n 19 ionice -c3 python3 tools/criterion_v2/window_segments.py score --spec SPEC.json --span LABEL \
        --runs DIR [--workers N] [--root KEY=DIR ...]
    python3 tools/criterion_v2/window_segments.py check --spec SPEC.json [--verify-sha N] [--root KEY=DIR ...]
    nice -n 19 python3 tools/criterion_v2/window_segments.py v1-window --spec SPEC.json --span LABEL --runs DIR
    python3 tools/criterion_v2/window_segments.py collect --spec SPEC.json --runs DIR --json OUT.json \
        --markdown OUT.md [--root KEY=DIR ...]

NOT PART OF THE FROZEN CRITERION.  ``v2rule.py`` and ``score.py`` are imported unchanged; ``score.py`` checks
both against ``FROZEN.json`` and refuses otherwise.  This file changes no threshold, window rule or definition.
It only does what the frozen window reader (``score.window_jobs``) cannot do by itself:

1. **No receipt.**  A stream whose run ended abnormally has no ``window-stream.json``.  The fields the frozen
   reader takes from it (child spacing, child nx/ny, specified and relaxation zones, the frame motion
   umove/vmove, the scalar level heights ``zh_cm1_indexed`` and the ``uadv``/``vadv`` block layout) are
   derived from the run's deck: child dx = parent dx / refinement_ratio, child nx = parent_span_x x ratio,
   levels from ``child_levels_m`` (the w faces, 0 at the ground) with CM1's mid-point scalar levels and mirrored
   ghosts (CM1 r21.1 param.F:8165-8172: ``zh(k) = 0.5*(zf(k+1)+zf(k))``, ``zf(0) = -zf(2)``,
   ``zh(0) = -zh(1)``), and the schema-v3 record layout (``uadv (nz+2, ny+2NG, nx+1+2NG)`` then
   ``vadv (nz+2, ny+1+2NG, nx+2NG)`` at word offsets 0 and the size of ``uadv``).  When a reference receipt
   written by the same code and deck is given, every one of those fields must equal it, and the reference
   stream's records must be byte-identical (index SHA-256) to this stream's records at the same model time.
2. **Several segments.**  A record trajectory may be split over several stream directories (e.g. the records
   after a restart point copied to their own arena with their own index from 0).  Each segment gets a view
   directory (the derived receipt, its index rows inside the segment's time range and the span, a symlink to
   its arena); the frozen ``score.window_jobs`` reads each view, and the jobs are concatenated in time order.
   Model times must be strictly increasing across segments.  ``index`` is rewritten to the trajectory's own
   record number (segment index + ``index_offset``); ``offset_bytes`` stays the segment's own.
3. **Spans.**  A span is a time range of the trajectory given in the spec (a data-quality split stated there),
   scored as its own source with every record in it read (no ``--every``/``--start``/``--end``, so the frozen
   scorer's ``protocol_conforming`` stays as it computes it).  The span, not the scorer, chooses the records;
   the spec says why.

The frame: window winds are relative to the translating frame (CM1 base.F:2663-2670).  The frozen scorer
already adds umove/vmove for the hazard wind (solve3.F:864-865 ``sws`` form, ``v2rule`` step 7) and puts the
window on the ground as ``origin + umove * t`` (step 10), so this adapter passes the deck's umove/vmove and the
index rows' own ``origin_x_m``/``origin_y_m`` (metres from the parent grid origin in the translating frame)
through unchanged.  The time cadence (0.09375 s here) needs nothing: the rule's links are in seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
NG = 3
RECEIPT_KEYS_CHILD = ("dx_m", "dy_m", "nx", "ny", "spec_zone", "relax_zone", "u_move_m_s", "v_move_m_s")


class AdapterError(Exception):
    pass


# ---------------------------------------------------------------------------
# Receipt from the deck.
# ---------------------------------------------------------------------------
def v3_shapes(nx: int, ny: int, nz: int) -> list[tuple[str, tuple[int, ...]]]:
    """The schema-v3 record's blocks in order (cm1core.nest_child_window v3; 15 float32 blocks)."""
    ring = (nz + 1, ny + 2, nx + 2)
    return [("uadv", (nz + 2, ny + 2 * NG, nx + 1 + 2 * NG)), ("vadv", (nz + 2, ny + 1 + 2 * NG, nx + 2 * NG)),
            ("wadv", (nz + 3, ny + 2 * NG, nx + 2 * NG))] + \
        [(n, ring) for n in ("rho", "prs", "th", "qv", "qc", "qr", "qis", "qg")] + \
        [("uten", (nz + 1, ny + 2, nx + 3)), ("vten", (nz + 1, ny + 3, nx + 2)), ("wten", (nz + 2, ny + 2, nx + 2)),
         ("znt", (ny + 2 * NG, nx + 2 * NG))]


def deck_receipt(deck: Path, nx: int, ny: int, nz: int) -> dict[str, Any]:
    """The receipt fields ``score.window_jobs`` reads, derived from a stage deck and the window's nx, ny, nz."""
    d = tomllib.loads(Path(deck).read_text())
    g, n, cl = d["grid"], d["nest"], d["closures"]
    if not cl.get("moving_frame"):
        raise AdapterError(f"{deck}: moving_frame is not set; umove/vmove would be undefined")
    ratio = int(n["refinement_ratio"])
    faces = [float(x) for x in n["child_levels_m"]]
    if faces[0] != 0.0 or any(b <= a for a, b in zip(faces, faces[1:])):
        raise AdapterError(f"{deck}: child_levels_m must start at 0 and increase")
    zh = [0.5 * (faces[k + 1] + faces[k]) for k in range(len(faces) - 1)]       # param.F:8169
    zh_cm1 = [-zh[0]] + zh                                                         # param.F:8172
    zf_cm1 = [-faces[1]] + faces                                                   # param.F:8165
    if nz + 1 > len(zh):
        raise AdapterError(f"window nz {nz} exceeds the child's {len(zh)} levels")
    fields, off = [], 0
    for name, shp in v3_shapes(nx, ny, nz):
        words = math.prod(shp)
        fields.append({"name": name, "shape_zyx": list(shp), "offset_words": off, "words": words})
        off += words
    return {
        "schema": "gm1.nest-child-window-stream/v3",
        "child": {"dx_m": float(g["dx_m"]) / ratio, "dy_m": float(g["dy_m"]) / ratio,
                  "nx": int(n["parent_span_x"]) * ratio, "ny": int(n["parent_span_y"]) * ratio,
                  "nz": int(n["child_nz"]), "spec_zone": int(n["spec_zone"]), "relax_zone": int(n["relax_zone"]),
                  "u_move_m_s": float(cl["u_move_m_s"]), "v_move_m_s": float(cl["v_move_m_s"])},
        "window": {"nx": nx, "ny": ny, "nz": nz},
        "zh_cm1_indexed": zh_cm1[: nz + 2],
        "zf_cm1_indexed": zf_cm1[: nz + 3],
        "fields": fields,
        "record_words": off,
        "record_bytes": 4 * off,
        "parent_dx_m": float(g["dx_m"]),
    }


def compare_receipt(derived: dict[str, Any], ref: dict[str, Any]) -> list[str]:
    """Every field the frozen reader uses, derived vs the reference receipt: [] when they agree."""
    bad = []
    for k in RECEIPT_KEYS_CHILD:
        if float(derived["child"][k]) != float(ref["child"][k]):
            bad.append(f"child.{k}: derived {derived['child'][k]} vs reference {ref['child'][k]}")
    for k in ("nx", "ny", "nz"):
        if int(derived["window"][k]) != int(ref["window"][k]):
            bad.append(f"window.{k}: derived {derived['window'][k]} vs reference {ref['window'][k]}")
    nzh = derived["window"]["nz"] + 1
    zr = [float(x) for x in ref["zh_cm1_indexed"][:nzh]]
    zd = [float(x) for x in derived["zh_cm1_indexed"][:nzh]]
    if max(abs(a - b) for a, b in zip(zr, zd)) > 1e-3 or len(zr) != len(zd):
        bad.append(f"zh_cm1_indexed: derived {zd[:5]}... vs reference {zr[:5]}...")
    rf = {f["name"]: f for f in ref["fields"]}
    for f in derived["fields"]:
        r = rf.get(f["name"])
        if r is None or list(r["shape_zyx"]) != f["shape_zyx"] or int(r["offset_words"]) != f["offset_words"]:
            bad.append(f"field {f['name']}: derived {f['shape_zyx']}@{f['offset_words']} vs reference "
                       f"{r and (r['shape_zyx'], r['offset_words'])}")
    if "record_bytes" in ref and int(ref["record_bytes"]) != derived["record_bytes"]:
        bad.append(f"record_bytes: derived {derived['record_bytes']} vs reference {ref['record_bytes']}")
    return bad


# ---------------------------------------------------------------------------
# Spec, segments and views.
# ---------------------------------------------------------------------------
def load_spec(path: Path, roots_override: list[str]) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    roots = dict(spec.get("roots", {}))
    for r in roots_override:
        k, _, v = r.partition("=")
        roots[k] = v
    spec["roots"] = roots
    return spec


def resolve(value: str, roots: dict[str, str]) -> Path:
    key, sep, rel = value.partition(":")
    if sep and key in roots:
        return Path(roots[key]) / rel
    return Path(value)


def index_rows(stream: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Header and whole rows of a window index (a trailing partial line is not a record)."""
    data = (stream / "window-records.tsv").read_bytes()
    lines = data[: data.rfind(b"\n")].decode().split("\n")
    head = lines[0].split("\t")
    return head, [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if len(ln.split("\t")) == len(head)]


def _inside(t: float, lo: float | None, hi: float | None, hi_exclusive: bool = False) -> bool:
    if lo is not None and t < lo - 1e-9:
        return False
    if hi is not None and (t >= hi - 1e-9 if hi_exclusive else t > hi + 1e-9):
        return False
    return True


def trajectory(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The derived receipt and every segment's rows (whole, inside the segment's range and the clean limit)."""
    ws = spec["window_segments"]
    roots = spec["roots"]
    segs = []
    geom = set()
    for k, s in enumerate(ws["segments"]):
        stream = resolve(s["stream"], roots)
        _, rows = index_rows(stream)
        arena = stream / "window-stream.f32"
        size = arena.stat().st_size
        kept = []
        for r in rows:
            t = float(r["model_time_s"])
            if not _inside(t, s.get("t_min"), s.get("t_max")) or not _inside(t, None, ws.get("clean_t_max")):
                continue
            if int(r["offset_bytes"]) + int(r["bytes"]) > size:
                raise AdapterError(f"{stream}: record {r['index']} runs past the arena ({size} B)")
            geom.add((int(r["nx"]), int(r["ny"]), int(r["nz"]), int(r["bytes"])))
            kept.append(r)
        off = int(s.get("index_offset", 0))
        if s.get("index_offset_from"):
            off = int(json.loads((stream / s["index_offset_from"]).read_text())[s["index_offset_key"]])
        segs.append({"k": k, "stream": stream, "arena": arena.resolve(), "rows": kept, "index_offset": off,
                     "t_min": s.get("t_min"), "t_max": s.get("t_max")})
    if len(geom) != 1:
        raise AdapterError(f"window geometry differs between records: {sorted(geom)}")
    nx, ny, nz, nbytes = geom.pop()
    receipt = deck_receipt(resolve(ws["deck"], roots), nx, ny, nz)
    if receipt["record_bytes"] != nbytes:
        raise AdapterError(f"records are {nbytes} B, the v3 layout of {nx}x{ny}x{nz} is {receipt['record_bytes']} B")
    times = [float(r["model_time_s"]) for s in segs for r in s["rows"]]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise AdapterError("model times are not strictly increasing across the segments")
    return receipt, segs


def reference_check(spec: dict[str, Any], receipt: dict[str, Any], segs: list[dict[str, Any]]) -> dict[str, Any]:
    """The derived receipt against a reference receipt from the same code and deck, and its records' identity."""
    ws = spec["window_segments"]
    ref_dir = ws.get("reference_stream")
    if not ref_dir:
        return {"reference": None}
    ref_dir = resolve(ref_dir, spec["roots"])
    ref = json.loads((ref_dir / "window-stream.json").read_text())
    bad = compare_receipt(receipt, ref)
    _, ref_rows = index_rows(ref_dir)
    by_t = {round(float(r["model_time_s"]), 6): r["sha256"] for s in segs for r in s["rows"]}
    same = sum(1 for r in ref_rows if by_t.get(round(float(r["model_time_s"]), 6)) == r["sha256"])
    matched = sum(1 for r in ref_rows if round(float(r["model_time_s"]), 6) in by_t)
    if bad:
        raise AdapterError(f"derived receipt disagrees with {ref_dir}: {bad}")
    if matched == 0 or same != matched:
        raise AdapterError(f"{ref_dir}: {same} of {matched} records at shared times are byte-identical")
    return {"reference": str(ref_dir), "fields_agree": True, "records_at_shared_times": matched,
            "records_byte_identical": same}


def verify_sha(segs: list[dict[str, Any]], per_segment: int) -> list[dict[str, Any]]:
    """SHA-256 of ``per_segment`` records spread over each segment (first, last, evenly between) vs the index."""
    out = []
    for s in segs:
        rows = s["rows"]
        if not rows or per_segment <= 0:
            continue
        picks = sorted({round(i * (len(rows) - 1) / max(1, per_segment - 1)) for i in range(per_segment)})
        with open(s["arena"], "rb") as fh:
            for i in picks:
                r = rows[i]
                fh.seek(int(r["offset_bytes"]))
                h = hashlib.sha256()
                left = int(r["bytes"])
                while left:
                    b = fh.read(min(left, 1 << 24))
                    h.update(b)
                    left -= len(b)
                ok = h.hexdigest() == r["sha256"]
                out.append({"segment": s["k"], "index": r["index"], "model_time_s": float(r["model_time_s"]), "ok": ok})
                if not ok:
                    raise AdapterError(f"segment {s['k']} record {r['index']}: SHA-256 differs from the index")
    return out


def span_of(spec: dict[str, Any], label: str) -> dict[str, Any]:
    for sp in spec["window_segments"]["spans"]:
        if sp["label"] == label:
            return sp
    raise AdapterError(f"no span {label!r} in the spec")


def build_views(receipt: dict[str, Any], segs: list[dict[str, Any]], span: dict[str, Any], views: Path,
                provenance: dict[str, Any]) -> list[Path]:
    """One view directory per segment holding records of the span: receipt, index, arena symlink."""
    views.mkdir(parents=True, exist_ok=False)
    out = []
    cols = ["index", "segment_index", "model_time_s", "offset_bytes", "bytes", "i0", "j0", "nx", "ny", "nz",
            "origin_x_m", "origin_y_m", "sha256"]
    for s in segs:
        rows = [r for r in s["rows"] if _inside(float(r["model_time_s"]), span.get("t_min"), span.get("t_max"),
                                                bool(span.get("t_max_exclusive")))]
        if not rows:
            continue
        v = views / f"seg-{s['k']}"
        v.mkdir()
        rec = dict(receipt)
        rec["adapter"] = {**provenance, "segment": s["k"], "stream": str(s["stream"]), "arena": str(s["arena"]),
                          "index_offset": s["index_offset"], "span": span}
        (v / "window-stream.json").write_text(json.dumps(rec, indent=1) + "\n")
        lines = ["\t".join(cols)]
        for r in rows:
            row = dict(r, segment_index=r["index"], index=str(int(r["index"]) + s["index_offset"]))
            lines.append("\t".join(row[c] for c in cols))
        (v / "window-records.tsv").write_text("\n".join(lines) + "\n")
        os.symlink(s["arena"], v / "window-stream.f32")
        out.append(v)
    if not out:
        raise AdapterError(f"span {span['label']} holds no record")
    return out


def load_score():
    sys.path.insert(0, str(HERE))
    import score  # noqa: E402  (the frozen scorer; it checks itself and v2rule.py against FROZEN.json)
    return score


def score_span(spec: dict[str, Any], label: str, runs: Path, workers: int) -> dict[str, Any]:
    ws = spec["window_segments"]
    span = span_of(spec, label)
    receipt, segs = trajectory(spec)
    ref = reference_check(spec, receipt, segs)
    if abs(float(ws["parent_dx_m"]) - receipt["parent_dx_m"]) > 1e-9:
        raise AdapterError(f"parent_dx_m {ws['parent_dx_m']} vs the deck's {receipt['parent_dx_m']}")
    views = build_views(receipt, segs, span, runs / "views" / label,
                        {"tool": "tools/criterion_v2/window_segments.py", "deck": str(resolve(ws["deck"], spec["roots"])),
                         "reference_check": ref})
    sc = load_score()
    frozen_window_jobs = sc.__dict__.setdefault("_frozen_window_jobs", sc.window_jobs)

    def combined(_root: Path) -> list[dict[str, Any]]:
        jobs = [j for v in views for j in frozen_window_jobs(v)]
        jobs.sort(key=lambda j: j["model_time_s"])
        t = [j["model_time_s"] for j in jobs]
        if any(b <= a for a, b in zip(t, t[1:])):
            raise sc.Refused("window segments overlap in model time")
        return jobs

    args = [str(views[0]), "--out", str(runs / label), "--kind", "window", "--label", label,
            "--workers", str(workers), "--nest-start-s", f"{float(ws['nest_start_s']):g}",
            "--parent-dx", f"{float(ws['parent_dx_m']):g}"]
    (runs / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sc.window_jobs = combined
    try:
        rc = sc.main(args)
    finally:
        sc.window_jobs = frozen_window_jobs
    rec = {"label": label, "status": "scored" if rc == 0 else f"scorer exit {rc}",
           "command": [sys.executable, str(Path(__file__).resolve()), "score", "--span", label],
           "scorer_argv": args, "wall_s": time.time() - t0, "workers": workers,
           "nest_start_s": float(ws["nest_start_s"]), "nest_start_source": ws.get("nest_start_source"),
           "parent_dx_m": float(ws["parent_dx_m"]), "parent_dx_source": "deck [grid] dx_m",
           "adapter": {"views": [str(v) for v in views], "span": span, "reference_check": ref,
                       "records": sum(len((v / "window-records.tsv").read_text().splitlines()) - 1 for v in views),
                       "segments": [{"k": s["k"], "stream": str(s["stream"]), "index_offset": s["index_offset"],
                                     "rows_in_segment": len(s["rows"])} for s in segs]}}
    (runs / f"{label}.run.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return rec


# ---------------------------------------------------------------------------
# v1 beside, at window cadence (reported only; v1 decides nothing under v2).
# ---------------------------------------------------------------------------
def v1_window(spec: dict[str, Any], label: str, runs: Path) -> dict[str, Any]:
    """v1's own host sample and ledger on every window record of a span (for a run with no live v1 sidecar).

    The live v1 samples every parent step with a 300 m disk about the strongest corner vorticity inside a
    1000 m edge margin, which on a 2 km child is the child's centre corner alone.  A window record holds that
    disk when the window covers it: the record's two 10 m bracketing levels are put into a child-sized padded
    array at the window's own place (i0, j0) and v1's ``reference_sample`` reads it; every corner and cell v1
    reads must lie inside the window's interior (checked), so the zero fill outside is never read.
    """
    import numpy as np
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from cm1core.nest_tornado_criterion import (CRITERION, CRITERION_SHA256, CriterionLedger, SamplePlan,
                                                interior_corner_box, reference_sample, sample_from_words)
    sc = load_score()
    span = span_of(spec, label)
    receipt, segs = trajectory(spec)
    ch = receipt["child"]
    cnx, cny, dx, dy = int(ch["nx"]), int(ch["ny"]), float(ch["dx_m"]), float(ch["dy_m"])
    um, vm = float(ch["u_move_m_s"]), float(ch["v_move_m_s"])
    nx, ny, nz = (int(receipt["window"][k]) for k in ("nx", "ny", "nz"))
    zh = np.asarray(receipt["zh_cm1_indexed"][1:nz + 1], dtype=np.float64)
    k0, k1, w0, w1, note = sc.bracket(zh, float(CRITERION["height_agl_m"]))
    if note:
        raise AdapterError(f"v1: {note}")
    lo_i, hi_i = interior_corner_box(cnx, dx, CRITERION["edge_margin_m"])
    lo_j, hi_j = interior_corner_box(cny, dy, CRITERION["edge_margin_m"])
    rad_i = int(math.ceil(CRITERION["disk_radius_m"] / dx)) + 1
    rad_j = int(math.ceil(CRITERION["disk_radius_m"] / dy)) + 1
    plan = SamplePlan(nx=cnx, ny=cny, nz=2, dx_m=float(np.float32(dx)), dy_m=float(np.float32(dy)),
                      ku0=0, ku1=1, w0=w0, w1=w1, radius_m=CRITERION["disk_radius_m"], u_move_m_s=um,
                      v_move_m_s=vm, lo_i=lo_i, hi_i=hi_i, lo_j=lo_j, hi_j=hi_j)
    lay = {f["name"]: (tuple(f["shape_zyx"]), int(f["offset_words"])) for f in receipt["fields"]}
    ledger = CriterionLedger()
    n = 0
    t_first = t_last = None
    for s in segs:
        mm = np.memmap(s["arena"], dtype="<f4", mode="r")
        for r in s["rows"]:
            t = float(r["model_time_s"])
            if not _inside(t, span.get("t_min"), span.get("t_max"), bool(span.get("t_max_exclusive"))):
                continue
            i0, j0 = int(r["i0"]), int(r["j0"])
            # v1 reads corners lo..hi plus the disk about them; the window's interior corners are i0+1..i0+nx-1
            if not (i0 + 1 <= lo_i - rad_i and hi_i + rad_i <= i0 + nx - 1
                    and j0 + 1 <= lo_j - rad_j and hi_j + rad_j <= j0 + ny - 1):
                raise AdapterError(f"record {r['index']}: v1's search box and disk are not inside the window")
            base = int(r["offset_bytes"]) // 4
            up = np.zeros((2, cny + 2 * NG, cnx + 1 + 2 * NG), np.float32)
            vp = np.zeros((2, cny + 1 + 2 * NG, cnx + 2 * NG), np.float32)
            for m, k in enumerate((k0, k1)):
                shp, off = lay["uadv"]
                a = mm[base + off: base + off + math.prod(shp)].reshape(shp)
                up[m, NG + j0: NG + j0 + ny, NG + i0: NG + i0 + nx + 1] = a[k + 1, NG:NG + ny, NG:NG + nx + 1]
                shp, off = lay["vadv"]
                a = mm[base + off: base + off + math.prod(shp)].reshape(shp)
                vp[m, NG + j0: NG + j0 + ny + 1, NG + i0: NG + i0 + nx] = a[k + 1, NG:NG + ny + 1, NG:NG + nx]
            words = reference_sample(up, vp, plan=plan)
            ox = float(r["origin_x_m"]) - i0 * dx + um * t
            oy = float(r["origin_y_m"]) - j0 * dy + vm * t
            ledger.observe(sample_from_words(words, step=int(r["index"]) + s["index_offset"], model_time_s=t,
                                             dx_m=plan.dx_m, dy_m=plan.dy_m, origin_x_m=ox, origin_y_m=oy,
                                             domainlocx_m=0.0, domainlocy_m=0.0))
            n += 1
            t_first = t if t_first is None else t_first
            t_last = t
    rec = ledger.record()
    out = {k: rec.get(k) for k in ("verdict", "genesis_model_time_s", "samples", "longest_qualifying_run_s",
                                   "condition_counts", "first_sample_model_time_s", "last_sample_model_time_s")}
    out.update(source=("no live v1 sidecar: v1's host sample (reference_sample, the child-centre corner and its "
                        "300 m disk) recomputed on every window record and decided by v1's ledger at window "
                        "cadence (the live criterion samples every parent step)"),
               criterion_v1_sha256=CRITERION_SHA256, span=span, records=n, record_span_s=[t_first, t_last])
    (runs / f"{label}.v1-window.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


# ---------------------------------------------------------------------------
# Collection (bookkeeping only; verdicts are copied from verdict.json).
# ---------------------------------------------------------------------------
def collect(spec: dict[str, Any], runs: Path, out_json: Path, out_md: Path) -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import verdict_table as vt  # noqa: E402
    import v2rule as rule  # noqa: E402
    sc = load_score()
    roots = dict(spec["roots"], runs=str(runs))

    def res(value: str, rts: dict[str, str]) -> Path:
        return resolve(value, rts)

    frozen_basis = vt.v1_basis

    def basis(v1: dict[str, Any]) -> str:
        return ("recomputed on the window records" if "window record" in (v1.get("source") or "")
                else frozen_basis(v1))

    vt.v1_basis = basis
    recs = [vt.summarise(s, runs, rule, sc, roots, res) for s in spec["sources"]]
    by_label = {r["label"]: r for r in recs}
    v1_from = spec.get("v1_from")
    for r in recs:
        if r["kind"] == "window" and r.get("status") == "scored":
            vw = runs / f"{r['label']}.v1-window.json"
            if vw.is_file():
                r["v1"] = json.loads(vw.read_text())
                r["v1_from"] = "window records"
            elif v1_from in by_label:
                r["v1"] = by_label[v1_from].get("v1")
                r["v1_from"] = v1_from
    head = spec["headline"]
    runs_doc: dict[str, dict[str, Any]] = {}
    for r in recs:
        runs_doc.setdefault(r["run"], {"sources": []})["sources"].append(r)
    for run, entry in runs_doc.items():
        pick = by_label.get(head["source"])
        if pick is None or pick["run"] != run or pick["status"] != "scored":
            entry["run_verdict"] = {"outcome": None, "basis": "headline source not scored"}
            continue
        entry["run_verdict"] = {
            "outcome": pick["v2"]["outcome"], "caption": pick["v2"]["caption"], "source": pick["label"],
            "span": pick["record_T"], "genesis": vt.genesis_text(pick), "basis": head["why"],
            "other_sources": {x["label"]: (x["v2"]["outcome"] if x["status"] == "scored" else "not scored")
                              for x in entry["sources"] if x is not pick}}
    frozen = json.loads((HERE / "FROZEN.json").read_text())
    doc = {
        "schema": "gm1.tornado-verdicts/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "about": spec.get("about"),
        "criterion": {"name": "GM1 tornado criterion v2 (pre-registered)", "doc": "docs/GM1-TORNADO-CRITERION-V2.md",
                      "freeze_commit": "198de287e3e304e32172003ad699c1d27e4211b3",
                      "frozen_utc": "2026-09-22T19:27:37Z", "frozen": frozen},
        "adapter": {"tool": "tools/criterion_v2/window_segments.py",
                    "what": "derived receipt, segment views and spans; no rule change (see the tool's docstring)"},
        "caption_wording": dict(rule.CAPTIONS),
        "time_convention": "T+m:ss is model time / 60 (T+55:12 = 3312 s)",
        "runs": runs_doc,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1, default=str, allow_nan=False) + "\n")
    out_md.write_text(vt.markdown_table(recs))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("score", "check", "collect", "v1-window"))
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--span")
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--verify-sha", type=int, default=0, help="records per segment to SHA-256 against the index")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec, a.root)
    try:
        if a.action == "check":
            receipt, segs = trajectory(spec)
            rep = {"receipt": {k: receipt[k] for k in ("child", "window", "record_bytes")},
                   "zh_first": receipt["zh_cm1_indexed"][:5],
                   "segments": [{"k": s["k"], "stream": str(s["stream"]), "rows": len(s["rows"]),
                                 "t": [float(s["rows"][0]["model_time_s"]), float(s["rows"][-1]["model_time_s"])]
                                 if s["rows"] else None, "index_offset": s["index_offset"]} for s in segs],
                   "reference_check": reference_check(spec, receipt, segs),
                   "sha_checks": verify_sha(segs, a.verify_sha)}
            for sp in spec["window_segments"]["spans"]:
                n = sum(1 for s in segs for r in s["rows"] if _inside(float(r["model_time_s"]), sp.get("t_min"),
                                                                       sp.get("t_max"), bool(sp.get("t_max_exclusive"))))
                rep.setdefault("spans", {})[sp["label"]] = n
            print(json.dumps(rep, indent=1, default=str))
            return 0
        if a.action == "score":
            if not a.span or not a.runs:
                raise AdapterError("score needs --span and --runs")
            rec = score_span(spec, a.span, a.runs, a.workers)
            print(json.dumps({k: rec[k] for k in ("label", "status", "wall_s")}))
            return 0 if rec["status"] == "scored" else 2
        if a.action == "v1-window":
            if not a.span or not a.runs:
                raise AdapterError("v1-window needs --span and --runs")
            rec = v1_window(spec, a.span, a.runs)
            print(json.dumps({k: rec[k] for k in ("verdict", "genesis_model_time_s", "samples",
                                                   "longest_qualifying_run_s")}))
            return 0
        if not (a.runs and a.json and a.markdown):
            raise AdapterError("collect needs --runs, --json and --markdown")
        collect(spec, a.runs, a.json, a.markdown)
        return 0
    except AdapterError as error:
        print(json.dumps({"status": "refused", "reason": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
