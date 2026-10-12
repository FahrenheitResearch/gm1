"""OFFLINE, CPU-ONLY: score every GM1 run with tornado criterion v2 (the scoring stage).

    nice -n 19 ionice -c3 python3 tools/criterion_v2/score_runs.py --out DIR
        [--manifest scoring-sources-v2.json] [--root KEY=DIR ...]
        [--only LABEL ...] [--jobs 1|2] [--scorer PATH/score.py] [--list]

This is not part of the frozen criterion.  It runs the frozen scorer
(``score.py``, which checks itself and ``v2rule.py`` against ``FROZEN.json``)
once per source in the manifest (``scoring-sources-v2.json``: run labels,
``ROOT:relative`` paths, spawn times and their sources), with the arguments
pre-registered in
``docs/GM1-TORNADO-CRITERION-V2.md`` ("Pre-registered sources and
arguments"): every sample, ``--workers 1``, the nest spawn time and parent
spacing for a child or window, ``--start`` only for a parent standing in after
a child's spawn.  Each source writes ``DIR/<label>/`` (``verdict.json``,
``samples.tsv``, ``candidates.tsv``) and ``DIR/logs/<label>.log``.

A source whose data is absent (not copied, not yet produced, an empty window
stream) is not scored; ``DIR/<label>.not-scored.json`` says why.  A source
whose ``DIR/<label>/verdict.json`` already exists is left alone, so the
command can be re-run when new output appears (the 2 m run).

Spawn times: a child is spawned when its run resumes from the parent restart,
so ``nest_start`` is ``receipt.json`` ``restart.resumed_model_time_s`` when
the run directory has a receipt (checked against the value written here when
both exist), otherwise the value written here and its stated source.  The
parent spacing is the parent deck's ``[grid] dx_m``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

MANIFEST = HERE / "scoring-sources-v2.json"


def load_manifest(path: Path = MANIFEST) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """``(roots, sources)`` from the scoring manifest (run names and paths live there, not in code)."""
    doc = json.loads(Path(path).read_text())
    return dict(doc["roots"]), list(doc["sources"])


def resolve(spec: str, roots: dict[str, str]) -> Path:
    key, _, rel = spec.partition(":")
    return Path(roots[key]) / rel


def availability(src: dict[str, Any], path: Path) -> str | None:
    """``None`` when the source has data to score, else why not."""
    kind = src["kind"]
    if not path.is_dir():
        return f"{path} does not exist on this machine"
    if kind == "window":
        recs = path / "window-records.tsv"
        stream = path / "window-stream.f32"
        if not (path / "window-stream.json").is_file():
            return f"{path}: no window-stream.json (the stream has no receipt: empty or unfinished)"
        if not recs.is_file() or len(recs.read_text().splitlines()) < 2 or stream.stat().st_size == 0:
            return f"{path}: the window stream holds no record"
        return None
    if kind == "child":
        if not (path / "placements").is_dir() or not any((path / "placements").glob("frame-*.json")):
            return f"{path}: no placement documents"
        if not any((path / p.stem / "history-field-table.tsv").is_file()
                   for p in (path / "placements").glob("frame-*.json")):
            return f"{path}: no child frame with a placement"
        return None
    if not any(path.glob("frame-*/history-frame-location.json")):
        return f"{path}: no history frame"
    return None


def nest_arguments(src: dict[str, Any], roots: dict[str, str]) -> tuple[list[str], dict[str, Any]]:
    """``--nest-start-s``/``--parent-dx`` and where each came from."""
    facts: dict[str, Any] = {}
    start = float(src["nest_start_s"])
    facts["nest_start_source"] = src.get("spawn_source", "manifest")
    if src.get("receipt"):
        rp = resolve(src["receipt"], roots)
        if rp.is_file():
            got = json.loads(rp.read_text()).get("restart", {}).get("resumed_model_time_s")
            if got is not None:
                if abs(float(got) - start) > 1e-6:
                    raise SystemExit(f"{src['label']}: receipt resumed at {got} s, manifest says {start} s")
                facts["nest_start_source"] = f"{rp} restart.resumed_model_time_s"
    if src.get("parent_deck"):
        dp = resolve(src["parent_deck"], roots)
        if dp.is_file():
            parent_dx = float(tomllib.loads(dp.read_text())["grid"]["dx_m"])
            facts["parent_dx_source"] = f"{dp} [grid] dx_m"
        elif "parent_dx_m" in src:
            parent_dx = float(src["parent_dx_m"])
            facts["parent_dx_source"] = "manifest (parent deck not on this machine)"
        else:
            raise SystemExit(f"{src['label']}: parent deck {dp} missing")
    else:
        parent_dx = float(src["parent_dx_m"])
        facts["parent_dx_source"] = "manifest parent_dx_m"
    facts.update(nest_start_s=start, parent_dx_m=parent_dx)
    return ["--nest-start-s", f"{start:g}", "--parent-dx", f"{parent_dx:g}"], facts


def command(src: dict[str, Any], roots: dict[str, str], scorer: Path, out: Path) -> tuple[list[str], dict[str, Any]]:
    path = resolve(src["path"], roots)
    args = [sys.executable, str(scorer), str(path), "--out", str(out / src["label"]), "--label", src["label"],
            "--workers", "1", "--kind", src["kind"]]
    facts: dict[str, Any] = {}
    if src["kind"] in ("child", "window"):
        extra, facts = nest_arguments(src, roots)
        args += extra
    else:
        args += ["--deck", str(resolve(src["deck"], roots))]
        if src.get("start_s") is not None:
            args += ["--start", f"{float(src['start_s']):g}"]
    return ["nice", "-n", "19", "ionice", "-c3"] + args, facts


def run_one(src: dict[str, Any], roots: dict[str, str], scorer: Path, out: Path) -> dict[str, Any]:
    label = src["label"]
    path = resolve(src["path"], roots)
    done = out / label / "verdict.json"
    if done.is_file():
        return {"label": label, "status": "exists"}
    why = availability(src, path)
    if why is not None:
        rec = {"label": label, "run": src["run"], "kind": src["kind"], "path": str(path), "status": "not scored",
               "reason": why, "note": src.get("note"), "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out / f"{label}.not-scored.json").write_text(json.dumps(rec, indent=1) + "\n")
        return rec
    stale = out / f"{label}.not-scored.json"
    if stale.is_file():
        stale.unlink()
    cmd, facts = command(src, roots, scorer, out)
    t0 = time.time()
    with (out / "logs" / f"{label}.log").open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}).returncode
    rec = {"label": label, "status": "scored" if rc == 0 else f"scorer exit {rc}", "command": cmd,
           "wall_s": time.time() - t0, **facts}
    (out / f"{label}.run.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--root", action="append", default=[], help="KEY=DIR, overriding the manifest's roots")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--scorer", type=Path, default=HERE / "score.py")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if not 1 <= a.jobs <= 2:
        raise SystemExit("--jobs must be 1 or 2 (each job is one niced process)")
    roots, sources = load_manifest(a.manifest)
    for r in a.root:
        k, _, v = r.partition("=")
        roots[k] = v
    todo = [s for s in sources if not a.only or s["label"] in a.only]
    if a.list:
        for s in todo:
            print(s["label"], resolve(s["path"], roots), availability(s, resolve(s["path"], roots)) or "ready")
        return 0
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    (a.out / "sources.json").write_text(json.dumps({"roots": roots, "scorer": str(a.scorer.resolve()),
                                                    "manifest": str(a.manifest.resolve()), "sources": sources},
                                                   indent=1) + "\n")
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for rec in pool.map(lambda s: run_one(s, roots, a.scorer.resolve(), a.out), todo):
            print(json.dumps({k: rec.get(k) for k in ("label", "status", "reason", "wall_s") if k in rec}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
