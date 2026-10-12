"""The criterion v2 scoring-stage tools on synthetic GM1 output (no GM1 run is read).

* ``score_runs.py`` runs the frozen scorer once per manifest source with the
  pre-registered arguments, records a missing source as not scored, leaves a
  scored source alone when re-run, and refuses a spawn time that disagrees
  with the run's receipt;
* ``verdict_table.py`` copies the verdicts, recovers the TORNADO episode's
  members with the frozen tracking, groups overlapping chains into one
  tracked tornado, and writes the table and the per-time-window file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "criterion_v2"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RULE_TESTS = _load("test_criterion_v2_rule", ROOT / "tests" / "test_criterion_v2_rule.py")
SR = _load("score_runs", TOOLS / "score_runs.py")
VT = _load("verdict_table", TOOLS / "verdict_table.py")


def _manifest(tmp: Path, sources: list[dict]) -> Path:
    p = tmp / "manifest.json"
    p.write_text(json.dumps({"roots": {"data": str(tmp / "data")}, "sources": sources}))
    return p


def _sources() -> list[dict]:
    common = {"run": "X", "grid": "8 m", "nest_start_s": -1000.0, "parent_dx_m": 100.0, "note": "synthetic"}
    return [{**common, "label": "X-child", "kind": "child", "path": "data:nest-child",
             "receipt": "data:receipt.json"},
            {**common, "label": "X-window", "kind": "window", "path": "data:nest-child-window"}]


@pytest.fixture(scope="module")
def scored(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("scoring")
    zh = np.array([2.0, 6.0, 10.0, 14.0, 18.0, 23.0, 30.0, 40.0])
    RULE_TESTS._child_run(tmp / "data", 8.0, zh, [0.0, 30.0, 60.0, 90.0, 120.0], moves=True)
    (tmp / "data" / "receipt.json").write_text(json.dumps({"restart": {"resumed_model_time_s": -1000.0}}))
    manifest = _manifest(tmp, _sources())
    out = tmp / "runs"
    assert SR.main(["--out", str(out), "--manifest", str(manifest)]) == 0
    return out, manifest


def test_score_runs_scores_present_sources_and_records_missing_ones(scored: tuple[Path, Path]) -> None:
    out, _ = scored
    v = json.loads((out / "X-child" / "verdict.json").read_text())
    assert v["outcome"] == "TORNADO" and v["protocol_conforming"] and v["matches_frozen"], v
    run = json.loads((out / "X-child.run.json").read_text())
    assert run["nest_start_s"] == -1000.0 and run["parent_dx_m"] == 100.0
    assert "--workers" in run["command"] and run["command"][run["command"].index("--workers") + 1] == "1"
    assert run["nest_start_source"].endswith("restart.resumed_model_time_s")
    miss = json.loads((out / "X-window.not-scored.json").read_text())
    assert miss["status"] == "not scored" and "does not exist" in miss["reason"]


def test_score_runs_leaves_a_scored_source_alone(scored: tuple[Path, Path], capsys: pytest.CaptureFixture) -> None:
    out, manifest = scored
    before = (out / "X-child" / "verdict.json").stat().st_mtime_ns
    assert SR.main(["--out", str(out), "--manifest", str(manifest), "--only", "X-child"]) == 0
    assert '"status": "exists"' in capsys.readouterr().out
    assert (out / "X-child" / "verdict.json").stat().st_mtime_ns == before


def test_score_runs_refuses_a_spawn_time_the_receipt_contradicts(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "receipt.json").write_text(json.dumps({"restart": {"resumed_model_time_s": 3600.0}}))
    src = {**_sources()[0], "nest_start_s": 4200.0}
    with pytest.raises(SystemExit, match="receipt resumed at 3600"):
        SR.nest_arguments(src, {"data": str(tmp_path / "data")})


def test_verdict_table_copies_verdicts_and_groups_the_tracked_tornado(scored: tuple[Path, Path], tmp_path: Path) -> None:
    out, manifest = scored
    j, md = tmp_path / "v.json", tmp_path / "t.md"
    assert VT.main(["--runs", str(out), "--json", str(j), "--markdown", str(md), "--manifest", str(manifest)]) == 0
    doc = json.loads(j.read_text())
    run = doc["runs"]["X"]
    assert run["run_verdict"]["outcome"] == "TORNADO" and run["run_verdict"]["source"] == "X-child"
    child, window = run["sources"]
    v = json.loads((out / "X-child" / "verdict.json").read_text())
    assert child["v2"]["outcome"] == v["outcome"] and child["v2"]["caption"] == v["caption"]
    assert child["episodes_recovered"] is True
    assert len(child["tornado_events"]) == 1
    ev = child["tornado_events"][0]
    assert ev["chains"] == len(v["episodes"]) and ev["primary"]["span_s"] == max(e["span_s"] for e in v["episodes"])
    assert ev["primary"]["circulation_peak_m2_s"] > 0
    states = [w["state"] for w in child["time_windows"]]
    assert "tornado 1" in states
    assert window["status"] == "not scored"
    text = md.read_text()
    assert "**TORNADO**" in text and "**not scored**" in text and "X (child frames)" in text


def _group_events_pairwise(episodes, eps):
    """``group_events`` as first written (every pair of episodes tested for a shared member)."""
    n = len(episodes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keysets = [set(map(tuple, e.get("member_keys", []))) if e.get("recovered") else set() for e in eps]
    for i in range(n):
        for j in range(i + 1, n):
            if i < len(keysets) and j < len(keysets) and keysets[i] & keysets[j]:
                parent[find(i)] = find(j)
    groups = {}
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


@pytest.mark.parametrize("seed", range(12))
def test_group_events_equals_the_pairwise_grouping(seed: int) -> None:
    """The linear-time grouping gives the pairwise grouping's events exactly, on random overlapping chains."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    episodes, eps = [], []
    for i in range(n):
        start = float(rng.integers(0, 50))
        span = float(rng.integers(120, 200))
        k0 = int(rng.integers(0, 300))
        keys = [[k, int(rng.integers(0, 3))] for k in range(k0, k0 + int(rng.integers(1, 25)))]
        episodes.append({"start_s": start, "end_s": start + span, "span_s": span, "samples": len(keys)})
        rec = bool(rng.random() > 0.1)
        eps.append({"recovered": rec, "member_keys": keys, "samples": len(keys)} if rec else {"recovered": False})
    assert VT.group_events(episodes, eps) == _group_events_pairwise(episodes, eps)
