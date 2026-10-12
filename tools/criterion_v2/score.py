"""OFFLINE, CPU-ONLY: score GM1 output with tornado criterion v2.

    nice -n 19 ionice -c3 python3 tools/criterion_v2/score.py SOURCE --out DIR
        [--kind child|parent|window] [--deck DECK.toml] [--dx M]
        [--nest-start-s S --parent-dx M] [--start S] [--end S] [--every N]
        [--workers N] [--label TEXT]

SOURCE is one of
  child    a nest-child directory: ``frame-NNNNNN/`` history frames plus
           ``placements/frame-NNNNNN.json`` (grid, placement, moving frame,
           explicit or stretched child levels ``zh_cm1_indexed``);
  parent   a frames directory of whole-domain history frames
           (``frame-NNNNNN/history-field-table.tsv``,
           ``history-carriers.f32``, ``history-frame-location.json``); the
           horizontal spacing comes from ``--deck`` ([grid] dx_m, dy_m;
           stretched horizontal grids are refused) or ``--dx``; the level
           heights from the frame's ``zhval`` profile;
  window   a ``nest-child-window`` stream (``window-stream.json``,
           ``window-records.tsv``, ``window-stream.f32``); advecting winds
           ``uadv``/``vadv``, levels from the receipt.
``--kind`` is inferred when omitted.

A nest child or window stream needs ``--nest-start-s`` (the model time the
child was spawned) and ``--parent-dx`` (the parent's spacing): samples
earlier than ``nest_start + nest_spinup_s`` carry only the parent's
structure, so their resolvedness is judged at the parent spacing.

PROTOCOL: a pre-registered verdict reads every sample of the source
(``--every 1``, no ``--end``; ``--start`` only for a parent standing in after
the child's spawn).  Other choices are allowed for exploration but the verdict
is marked ``protocol_conforming: false``.  The scorer refuses to run when
``v2rule.py`` or ``score.py`` differ from the SHA-256 recorded in
``FROZEN.json`` (``--unfrozen`` runs anyway and marks the verdict as not v2).

Every sample: the frame-relative winds interpolated linearly in height to
10 m AGL between the two scalar levels that bracket it (the run's own level
heights; the lowest level is never used as a stand-in), handed to
``v2rule.evaluate_plane``; then ``v2rule.decide``.  Ground positions are
plane position + placement origin + the domain's own motion
(``domainlocx/y``; for a window stream, which does not record it,
``umove * model_time``, whose constant offset cancels in every centre move).

Writes (``--out`` must not exist): ``samples.tsv`` (one row per sample,
``v2rule.SAMPLE_COLUMNS``) and ``verdict.json`` (outcome, caption, reason,
episodes, counts, the criterion's SHA-256 and the scorer's git commit, the
source and every choice made reading it), and ``candidates.tsv`` (every
refined candidate of every sample).  For a nest child, the v1 verdict is put
beside the v2 verdict: replayed from the live v1 sidecar
``tornado-criterion.jsonl`` when there is one, else recomputed with v1's
own host sample from the child frames at frame cadence (labelled so).  Terrain is refused (a flat-ground
reader: ``terrain_msl`` must be zero).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import v2rule  # noqa: E402
from v2rule import (CRITERION_V2, CRITERION_V2_SHA256, SAMPLE_COLUMNS, GridFacts, Plane,  # noqa: E402
                    decide, evaluate_plane, sample_row)

TABLE = "history-field-table.tsv"
ARENA = "history-carriers.f32"
LOCATION = "history-frame-location.json"
NG = 3


class Refused(Exception):
    pass


# ---------------------------------------------------------------------------
# Vertical interpolation.
# ---------------------------------------------------------------------------
def bracket(zh: np.ndarray, height: float) -> tuple[int, int, float, float, str | None]:
    """0-based levels ``(k, k+1)`` of physical scalar heights ``zh`` around ``height`` and weights.

    When ``height`` is below the lowest level, the lowest level alone is used
    and a note says so (the grid gate then makes the verdict UNRESOLVED).
    """
    zh = np.asarray(zh, dtype=np.float64)
    if height < zh[0]:
        return 0, 0, 1.0, 0.0, f"10 m is below the lowest scalar level {zh[0]:g} m"
    for k in range(len(zh) - 1):
        if zh[k] <= height <= zh[k + 1]:
            w1 = (height - zh[k]) / (zh[k + 1] - zh[k])
            return k, k + 1, 1.0 - w1, w1, None
    raise Refused(f"no pair of scalar levels brackets {height} m (levels {zh[:4]} ...)")


# ---------------------------------------------------------------------------
# History frames.
# ---------------------------------------------------------------------------
def read_table(frame_dir: Path) -> dict[str, tuple[tuple[int, ...], int]]:
    out = {}
    for line in (frame_dir / TABLE).read_text().splitlines():
        f = line.split("\t")
        if len(f) < 8:
            continue
        out[f[0]] = (tuple(int(v) for v in f[4].split(",")), int(f[7]))
    return out


def carrier(frame_dir: Path, table: dict, name: str, levels: tuple[int, ...] | None = None) -> np.ndarray:
    shape, offset = table[name]
    mm = np.memmap(frame_dir / ARENA, dtype="<f4", mode="r")
    n = int(np.prod(shape))
    a = mm[offset // 4: offset // 4 + n].reshape(shape)
    if levels is None:
        return np.asarray(a, dtype=np.float64)
    return np.stack([np.asarray(a[k], dtype=np.float64) for k in levels])


def check_flat(frame_dir: Path, table: dict) -> None:
    if "terrain_msl" in table:
        t = carrier(frame_dir, table, "terrain_msl")
        if np.any(t != 0.0):
            raise Refused(f"{frame_dir}: terrain_msl is not zero; this reader assumes flat ground (10 m AGL "
                          "would need the terrain-following heights)")


def frame_plane(frame_dir: Path, *, nx: int, ny: int, nz: int, dx: float, dy: float, zh: np.ndarray,
                umove: float, vmove: float, boundary_cells: int, ground_offset: tuple[float, float]
                ) -> tuple[Plane, str | None, float]:
    table = read_table(frame_dir)
    for name, shape in (("u3d", (nz, ny, nx + 1)), ("v3d", (nz, ny + 1, nx))):
        if name not in table or table[name][0] != shape:
            raise Refused(f"{frame_dir}: {name} is {table.get(name, ((None,),))[0]}, not {shape}")
    check_flat(frame_dir, table)
    k0, k1, w0, w1, note = bracket(zh, float(CRITERION_V2["height_agl_m"]))
    u = carrier(frame_dir, table, "u3d", (k0, k1))
    v = carrier(frame_dir, table, "v3d", (k0, k1))
    u10 = w0 * u[0] + w1 * u[1]
    v10 = w0 * v[0] + w1 * v[1]
    pp = None
    if "prs" in table and "prs0" in table and table["prs"][0] == (nz, ny, nx):
        p = carrier(frame_dir, table, "prs", (k0, k1))
        p0 = carrier(frame_dir, table, "prs0")
        pp = w0 * (p[0] - p0[k0]) + w1 * (p[1] - p0[k1])
    b = boundary_cells
    box = (b * dx, (nx - b) * dx, b * dy, (ny - b) * dy)
    plane = Plane(u=u10, v=v10, dx=dx, dy=dy, umove=umove, vmove=vmove, interior=box, data=box,
                  ground_offset=ground_offset, p_prime_pa=pp)
    return plane, note, float(zh[0])


def child_jobs(root: Path) -> list[dict[str, Any]]:
    pdir = root / "placements"
    jobs = []
    for pp in sorted(pdir.glob("frame-*.json")):
        p = json.loads(pp.read_text())
        idx = int(p["index"])
        fdir = root / f"frame-{idx:06d}"
        if not (fdir / TABLE).is_file():
            continue
        zh = np.asarray(p["zh_cm1_indexed"], dtype=np.float64)[1:int(p["nz"]) + 1]
        jobs.append({
            "kind": "child", "frame_dir": str(fdir), "index": idx, "model_time_s": float(p["model_time_s"]),
            "nx": int(p["nx"]), "ny": int(p["ny"]), "nz": int(p["nz"]), "dx": float(p["dx_m"]),
            "dy": float(p["dy_m"]), "zh": zh.tolist(), "umove": float(p["u_move_m_s"]),
            "vmove": float(p["v_move_m_s"]),
            "boundary_cells": int(p.get("spec_zone", 0)) + int(p.get("relax_zone", 0)),
            "ground_offset": (float(p["domainlocx_m"]) + float(p["origin_x_m"]),
                              float(p["domainlocy_m"]) + float(p["origin_y_m"])),
            "placement": [int(p["parent_i0"]), int(p["parent_j0"])],
        })
    if not jobs:
        raise Refused(f"{root}: no frame with a placement document; a child frame without its placement "
                      "cannot be put on the ground")
    return jobs


def parent_jobs(root: Path, dx: float, dy: float) -> list[dict[str, Any]]:
    jobs = []
    for fdir in sorted(root.glob("frame-*")):
        if not (fdir / TABLE).is_file() or not (fdir / LOCATION).is_file():
            continue
        loc = json.loads((fdir / LOCATION).read_text())
        table = read_table(fdir)
        nz, ny, nx1 = table["u3d"][0]
        zh = carrier(fdir, table, "zhval")
        mf = loc.get("moving_frame") or {}
        jobs.append({
            "kind": "parent", "frame_dir": str(fdir), "index": int(loc["index"]),
            "model_time_s": float(loc["model_time_s"]), "nx": nx1 - 1, "ny": ny, "nz": nz, "dx": dx, "dy": dy,
            "zh": zh.tolist(), "umove": float(mf.get("umove", 0.0)), "vmove": float(mf.get("vmove", 0.0)),
            "boundary_cells": 0,
            "ground_offset": (float(mf.get("domainlocx", 0.0)), float(mf.get("domainlocy", 0.0))),
        })
    if not jobs:
        raise Refused(f"{root}: no history frame with a location document")
    return jobs


def deck_spacing(deck: Path) -> tuple[float, float]:
    d = tomllib.loads(deck.read_text())
    g = d["grid"]
    if g.get("stretch_x") or g.get("stretch_y"):
        raise Refused(f"{deck}: stretched horizontal grid; v2's profile assumes uniform dx, dy")
    return float(g["dx_m"]), float(g["dy_m"])


# ---------------------------------------------------------------------------
# Window streams.
# ---------------------------------------------------------------------------
def window_jobs(root: Path) -> list[dict[str, Any]]:
    receipt = json.loads((root / "window-stream.json").read_text())
    lines = (root / "window-records.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln]
    ch = receipt["child"]
    zh = np.asarray(receipt["zh_cm1_indexed"], dtype=np.float64)
    layout = {e["name"]: (tuple(int(v) for v in e["shape_zyx"]), int(e["offset_words"])) for e in receipt["fields"]}
    jobs = []
    for r in rows:
        nx, ny, nz = int(r["nx"]), int(r["ny"]), int(r["nz"])
        jobs.append({
            "kind": "window", "stream": str(root), "index": int(r["index"]),
            "model_time_s": float(r["model_time_s"]), "offset_bytes": int(r["offset_bytes"]),
            "nx": nx, "ny": ny, "nz": nz, "dx": float(ch["dx_m"]), "dy": float(ch["dy_m"]),
            "zh": zh[1:nz + 1].tolist(), "umove": float(ch["u_move_m_s"]), "vmove": float(ch["v_move_m_s"]),
            "i0": int(r["i0"]), "j0": int(r["j0"]), "child_nx": int(ch["nx"]), "child_ny": int(ch["ny"]),
            "boundary_cells": int(ch.get("spec_zone", 0)) + int(ch.get("relax_zone", 0)),
            "origin": (float(r["origin_x_m"]), float(r["origin_y_m"])),
            "layout": {k: [list(v[0]), v[1]] for k, v in layout.items() if k in ("uadv", "vadv")},
        })
    return jobs


def window_plane(job: dict[str, Any]) -> tuple[Plane, str | None, float]:
    nx, ny, nz = job["nx"], job["ny"], job["nz"]
    zh = np.asarray(job["zh"])
    k0, k1, w0, w1, note = bracket(zh, float(CRITERION_V2["height_agl_m"]))
    mm = np.memmap(Path(job["stream"]) / "window-stream.f32", dtype="<f4", mode="r")
    base = job["offset_bytes"] // 4

    def field(name: str) -> np.ndarray:
        shape, off = job["layout"][name]
        expect = {"uadv": [nz + 2, ny + 2 * NG, nx + 1 + 2 * NG], "vadv": [nz + 2, ny + 1 + 2 * NG, nx + 2 * NG]}
        if list(shape) != expect[name]:
            raise Refused(f"window {name} shape {shape} is not {expect[name]}")
        return mm[base + off: base + off + int(np.prod(shape))].reshape(shape)

    ua, va = field("uadv"), field("vadv")
    # CM1 level k (1-based) is padded index k; zh index 0 here is k = 1
    u = [np.asarray(ua[k + 1, NG:NG + ny, NG:NG + nx + 1], dtype=np.float64) for k in (k0, k1)]
    v = [np.asarray(va[k + 1, NG:NG + ny + 1, NG:NG + nx], dtype=np.float64) for k in (k0, k1)]
    u10 = w0 * u[0] + w1 * u[1]
    v10 = w0 * v[0] + w1 * v[1]
    dx, dy = job["dx"], job["dy"]
    b = job["boundary_cells"]
    # the child's free interior in window-local metres
    interior = ((b - job["i0"]) * dx, (job["child_nx"] - b - job["i0"]) * dx,
                (b - job["j0"]) * dy, (job["child_ny"] - b - job["j0"]) * dy)
    data = (0.0, nx * dx, 0.0, ny * dy)
    t = job["model_time_s"]
    plane = Plane(u=u10, v=v10, dx=dx, dy=dy, umove=job["umove"], vmove=job["vmove"], interior=interior,
                  data=data, ground_offset=(job["origin"][0] + job["umove"] * t, job["origin"][1] + job["vmove"] * t))
    return plane, note, float(zh[0])


# ---------------------------------------------------------------------------
def score_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        os.nice(19)
    except OSError:
        pass
    t0 = time.time()
    if job["kind"] == "window":
        plane, note, z1 = window_plane(job)
    else:
        plane, note, z1 = frame_plane(
            Path(job["frame_dir"]), nx=job["nx"], ny=job["ny"], nz=job["nz"], dx=job["dx"], dy=job["dy"],
            zh=np.asarray(job["zh"]), umove=job["umove"], vmove=job["vmove"],
            boundary_cells=job["boundary_cells"], ground_offset=tuple(job["ground_offset"]))
    if job.get("resolution_dx"):
        plane.resolution_dx = float(job["resolution_dx"])
    s = evaluate_plane(plane)
    s["model_time_s"] = job["model_time_s"]
    s["index"] = job["index"]
    s["lowest_level_m"] = z1
    if note:
        s.setdefault("flags", []).append("below-lowest-level")
    s["wall_s"] = time.time() - t0
    for k in ("first_guess",):
        if k in s:
            s[k] = list(s[k])
    return s


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "tools/criterion_v2"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def v1_beside(root: Path, jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """v1's verdict: the sidecar's replay, or v1's host sample recomputed on every child frame."""
    sys.path.insert(0, str(ROOT / "src"))
    side = root / "tornado-criterion.jsonl"
    try:
        if side.is_file():
            from cm1core.nest_tornado_criterion import replay_time_series
            r = replay_time_series(side)
            out = {k: r.get(k) for k in ("verdict", "genesis_model_time_s", "samples", "longest_qualifying_run_s",
                                         "condition_counts", "first_sample_model_time_s",
                                         "last_sample_model_time_s")}
            out["source"] = "live v1 sidecar replayed (v1 samples every parent step)"
            return out
        return v1_from_frames(jobs)
    except Exception as error:  # the v1 verdict is reported beside v2, never required
        return {"error": f"{type(error).__name__}: {error}"}


def v1_from_frames(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """v1 recomputed from child frames: its host sample (``reference_sample``) and ledger, at frame cadence."""
    from cm1core.nest_tornado_criterion import (CRITERION, CRITERION_SHA256, NG as V1NG, CriterionLedger,
                                                SamplePlan, interior_corner_box, reference_sample,
                                                sample_from_words)
    ledger = CriterionLedger()
    for job in sorted(jobs, key=lambda j: j["model_time_s"]):
        fdir = Path(job["frame_dir"])
        table = read_table(fdir)
        zh = np.asarray(job["zh"])
        k0, k1, w0, w1, note = bracket(zh, float(CRITERION["height_agl_m"]))
        if note:
            raise Refused(f"v1: {note}")
        nx, ny = job["nx"], job["ny"]
        u = carrier(fdir, table, "u3d", (k0, k1))
        v = carrier(fdir, table, "v3d", (k0, k1))
        up = np.zeros((2, ny + 2 * V1NG, nx + 1 + 2 * V1NG), np.float32)
        vp = np.zeros((2, ny + 1 + 2 * V1NG, nx + 2 * V1NG), np.float32)
        up[:, V1NG:V1NG + ny, V1NG:V1NG + nx + 1] = u
        vp[:, V1NG:V1NG + ny + 1, V1NG:V1NG + nx] = v
        lo_i, hi_i = interior_corner_box(nx, job["dx"], CRITERION["edge_margin_m"])
        lo_j, hi_j = interior_corner_box(ny, job["dy"], CRITERION["edge_margin_m"])
        plan = SamplePlan(nx=nx, ny=ny, nz=2, dx_m=float(np.float32(job["dx"])), dy_m=float(np.float32(job["dy"])),
                          ku0=0, ku1=1, w0=w0, w1=w1, radius_m=CRITERION["disk_radius_m"],
                          u_move_m_s=job["umove"], v_move_m_s=job["vmove"], lo_i=lo_i, hi_i=hi_i,
                          lo_j=lo_j, hi_j=hi_j)
        words = reference_sample(up, vp, plan=plan)
        go = job["ground_offset"]
        ledger.observe(sample_from_words(words, step=int(job["index"]), model_time_s=job["model_time_s"],
                                         dx_m=plan.dx_m, dy_m=plan.dy_m, origin_x_m=go[0], origin_y_m=go[1],
                                         domainlocx_m=0.0, domainlocy_m=0.0))
    r = ledger.record()
    out = {k: r.get(k) for k in ("verdict", "genesis_model_time_s", "samples", "longest_qualifying_run_s",
                                 "condition_counts", "first_sample_model_time_s", "last_sample_model_time_s")}
    out["source"] = ("no live v1 sidecar: v1's host sample recomputed on every child frame and decided by v1's "
                     "ledger at frame cadence (the live criterion samples every parent step)")
    out["criterion_v1_sha256"] = CRITERION_SHA256
    return out


def frozen_check() -> dict[str, Any]:
    """The SHA-256 of the rule and scorer files against ``FROZEN.json``."""
    got = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ("v2rule.py", "score.py")}
    manifest_path = HERE / "FROZEN.json"
    want = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    files = want.get("files", {})
    ok = (bool(files) and all(files.get(k) == v for k, v in got.items())
          and want.get("criterion_v2_sha256") == CRITERION_V2_SHA256)
    return {"files_sha256": got, "manifest": want, "matches_frozen": ok}


def finite_json(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    return x


def infer_kind(src: Path) -> str:
    if (src / "window-stream.json").is_file():
        return "window"
    if (src / "placements").is_dir():
        return "child"
    return "parent"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=("child", "parent", "window"))
    ap.add_argument("--deck", type=Path)
    ap.add_argument("--dx", type=float)
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--nest-start-s", type=float, help="model time the nest child was spawned (child/window)")
    ap.add_argument("--parent-dx", type=float, help="the parent's horizontal spacing, m (child/window)")
    ap.add_argument("--unfrozen", action="store_true",
                    help="run although v2rule.py/score.py differ from FROZEN.json (the result is not a v2 verdict)")
    a = ap.parse_args(argv)
    frozen = frozen_check()
    if not frozen["matches_frozen"] and not a.unfrozen:
        print(json.dumps({"status": "refused", "reason": "v2rule.py or score.py differ from FROZEN.json; "
                          "this would not be a criterion v2 verdict", "check": frozen}), file=sys.stderr)
        return 2
    if a.workers < 1 or a.workers > 6:
        raise SystemExit("--workers must be 1..6")
    src = a.source
    kind = a.kind or infer_kind(src)
    try:
        if kind == "child":
            jobs = child_jobs(src)
        elif kind == "window":
            jobs = window_jobs(src)
        else:
            if a.deck is not None:
                dx, dy = deck_spacing(a.deck)
            elif a.dx is not None:
                dx = dy = float(a.dx)
            else:
                raise Refused("a parent frames directory needs --deck or --dx: its frames carry no spacing")
            jobs = parent_jobs(src, dx, dy)
    except Refused as error:
        print(json.dumps({"status": "refused", "reason": str(error)}), file=sys.stderr)
        return 2
    jobs = [j for j in jobs if (a.start is None or j["model_time_s"] >= a.start - 1e-9)
            and (a.end is None or j["model_time_s"] <= a.end + 1e-9)]
    jobs.sort(key=lambda j: j["model_time_s"])
    jobs = jobs[::max(1, a.every)]
    if not jobs:
        print(json.dumps({"status": "refused", "reason": "no sample in the requested interval"}), file=sys.stderr)
        return 2
    spinup_end = None
    if kind in ("child", "window"):
        if a.nest_start_s is None or a.parent_dx is None:
            print(json.dumps({"status": "refused", "reason": "a nest child or window needs --nest-start-s and "
                              "--parent-dx (its spin-up samples are judged at the parent spacing)"}),
                  file=sys.stderr)
            return 2
        spinup_end = a.nest_start_s + float(CRITERION_V2["nest_spinup_s"])
        for j in jobs:
            if j["model_time_s"] < spinup_end - 1e-9:
                j["resolution_dx"] = float(a.parent_dx)
    spacing = {(j["dx"], j["dy"]) for j in jobs}
    if len(spacing) != 1:
        print(json.dumps({"status": "refused", "reason": f"mixed grid spacings {sorted(spacing)}"}), file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    try:
        if a.workers == 1:
            samples = [score_job(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                samples = list(pool.map(score_job, jobs))
    except Refused as error:
        print(json.dumps({"status": "refused", "reason": str(error)}), file=sys.stderr)
        return 2
    dx, dy = spacing.pop()
    z1 = max(s["lowest_level_m"] for s in samples)
    notes = []
    if kind == "window":
        notes.append("window stream: advecting (acoustic-average) winds uadv/vadv, not the completed u3d/v3d; "
                     "profiles are cut at the window's own edges; ground offset = origin + umove*t")
    grid = GridFacts(dx_m=dx, dy_m=dy, lowest_scalar_level_m=z1, notes=notes)
    verdict = decide(samples, grid)
    with (a.out / "samples.tsv").open("w") as f:
        f.write("\t".join(SAMPLE_COLUMNS + ("index",)) + "\n")
        for k, s in enumerate(samples):
            f.write("\t".join(sample_row(k, s) + [str(s["index"])]) + "\n")
    with (a.out / "candidates.tsv").open("w") as f:
        f.write("\t".join(("sample", "model_time_s") + v2rule.CANDIDATE_KEYS) + "\n")
        for k, s in enumerate(samples):
            for q in s.get("candidates", []):
                row = [str(k), "%.4f" % s["model_time_s"]]
                for key in v2rule.CANDIDATE_KEYS:
                    val = q.get(key)
                    row.append(",".join(val) if isinstance(val, list) else
                               ("%.4f" % val if isinstance(val, float) else str(val)))
                f.write("\t".join(row) + "\n")
    conforming = (a.every == 1 and a.end is None and (a.start is None or kind == "parent") and not a.unfrozen
                  and frozen["matches_frozen"])
    verdict.update({
        "label": a.label,
        "source": {"kind": kind, "path": os.fspath(src.resolve()), "samples_read": len(jobs),
                   "first_index": jobs[0]["index"], "last_index": jobs[-1]["index"], "every": a.every,
                   "start_s": a.start, "end_s": a.end,
                   "deck": os.fspath(a.deck) if a.deck else None, "dx_override": a.dx},
        "cadence_statement": (f"judged from {kind} output at a median {verdict['median_interval_s']:g} s spacing; "
                              "a TORNADO needs the same vortex (overlapping cores in the storm-following frame) "
                              "in every linked sample; what happens between samples is not observed"
                              if verdict["median_interval_s"] == verdict["median_interval_s"] else None),
        "criterion_v2": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CRITERION_V2.items()},
        "scorer_commit": git_commit(),
        "scorer_sha256": frozen["files_sha256"]["v2rule.py"],
        "files_sha256": frozen["files_sha256"],
        "matches_frozen": frozen["matches_frozen"],
        "protocol_conforming": conforming,
        "nest_spinup": ({"nest_start_s": a.nest_start_s, "parent_dx_m": a.parent_dx, "spinup_end_s": spinup_end,
                         "samples_judged_at_parent_dx": sum(1 for j in jobs if j.get("resolution_dx"))}
                        if spinup_end is not None else None),
        "wall_s": time.time() - t0,
    })
    if kind == "child":
        verdict["v1_beside"] = v1_beside(src, jobs)
    (a.out / "verdict.json").write_text(json.dumps(finite_json(verdict), indent=1, allow_nan=False, default=str) + "\n")
    print(json.dumps({k: verdict[k] for k in ("outcome", "caption", "reason", "samples", "median_interval_s",
                                              "longest_qualifying_run_s", "class_counts")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
