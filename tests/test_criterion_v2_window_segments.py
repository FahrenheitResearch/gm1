"""The window-segments adapter (tools/criterion_v2/window_segments.py) on the synthetic vortices v2 was frozen with.

No GM1 run is read.  A synthetic schema-v3 window stream (frame-relative winds, the frame moving at 16.5 m/s,
the window re-centring in 100 m jumps, a 0.09375 s burst inside a 6 s cadence) is written twice: as one stream
with a real receipt, read by the frozen ``score.py`` directly, and as two receipt-less segments (the second with
its own index from 0 and a README giving its offset), read through the adapter with the receipt derived from a
deck.  The two must give the same samples, candidates and outcome; the rule and scorer files are the frozen ones.
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
W = _load("window_segments", TOOLS / "window_segments.py")
SC = W.load_score()

NG = 3
DX = 4.0
NX = NY = 113            # a 452 m window inside a 2 km child (i0 = j0 = 193: the child's centre)
NZ = 6
FACES = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0]
NEST_START = 2160.0
Z0 = 0.1

DECK = """
[run]
irst = 1
[grid]
dx_m = 100.0
dy_m = 100.0
[closures]
moving_frame = true
u_move_m_s = 16.5
v_move_m_s = 16.5
[nest]
refinement_ratio = 25
parent_span_x = 20
parent_span_y = 20
spec_zone = 1
relax_zone = 4
child_nz = 9
child_levels_m = [%s]
""" % ", ".join(f"{f:.1f}" for f in FACES)


def _times() -> list[float]:
    """After the spin-up (2160 + 300 s): every 64 child strides (6 s) for 126 s, plus a 0.09375 s burst."""
    base = [2466.0 + 6.0 * k for k in range(22)]
    burst = [2520.0 + 0.09375 * k for k in range(1, 9)]
    return sorted(base + burst)


def _record(t: float, kind: str, vmax: float, rmw: float, origin: tuple[float, float]) -> bytes:
    # the vortex sits still in the translating frame at model-frame (30700, 31300); the window's origin jumps
    comp = S.VortexComponent(kind, vmax, rmw, 30700.0 - origin[0], 31300.0 - origin[1], drift=False)
    pl = S.make_plane([comp], [], NX * DX, DX, t)
    zh = [0.5 * (a + b) for a, b in zip(FACES, FACES[1:])][:NZ]
    f = np.log(np.maximum(np.array(zh), 0.2) / Z0) / math.log(10.0 / Z0)
    shapes = dict(W.v3_shapes(NX, NY, NZ))
    blocks = []
    for name, shp in W.v3_shapes(NX, NY, NZ):
        a = np.zeros(shp, "<f4")
        if name == "uadv":
            for k in range(NZ):
                a[k + 1, NG:NG + NY, NG:NG + NX + 1] = f[k] * pl.u
        elif name == "vadv":
            for k in range(NZ):
                a[k + 1, NG:NG + NY + 1, NG:NG + NX] = f[k] * pl.v
        blocks.append(a.tobytes())
    assert shapes["uadv"][0] == NZ + 2
    return b"".join(blocks)


def _origin(q: int) -> tuple[float, float]:
    # 100 m re-centring jumps every 10 records, as the tracker-following window does
    return (30524.0 - 100.0 * ((q // 10) % 2), 31024.0 + 100.0 * ((q // 10) % 2))


HEAD = ["index", "child_step", "parent_step", "model_time_s", "offset_bytes", "bytes", "i0", "j0", "nx", "ny", "nz",
        "origin_x_m", "origin_y_m", "centre_x_m", "centre_y_m", "centre_held", "clamped", "sha256"]


def _write(root: Path, recs: list[tuple[float, bytes, tuple[float, float]]], receipt: dict | None,
           readme: dict | None = None) -> None:
    import hashlib
    root.mkdir(parents=True)
    rows, off = ["\t".join(HEAD)], 0
    with open(root / "window-stream.f32", "wb") as fh:
        for q, (t, blob, org) in enumerate(recs):
            fh.write(blob)
            rows.append("\t".join(str(x) for x in [q, 0, 0, repr(t), off, len(blob), 193, 193, NX, NY, NZ,
                                                   org[0], org[1], 0.0, 0.0, 0, 0,
                                                   hashlib.sha256(blob).hexdigest()]))
            off += len(blob)
    (root / "window-records.tsv").write_text("\n".join(rows) + "\n")
    if receipt is not None:
        (root / "window-stream.json").write_text(json.dumps(receipt))
    if readme is not None:
        (root / "README.json").write_text(json.dumps(readme))


def _stream(tmp: Path, kind: str, vmax: float, rmw: float, split_at: float) -> dict:
    deck = tmp / "deck.toml"
    deck.write_text(DECK)
    recs = [(t, _record(t, kind, vmax, rmw, _origin(q)), _origin(q)) for q, t in enumerate(_times())]
    receipt = W.deck_receipt(deck, NX, NY, NZ)
    # a hand-written receipt with the numbers a real v3 receipt carries, independent of deck_receipt
    zh = [0.5 * (a + b) for a, b in zip(FACES, FACES[1:])]
    real = {"child": {"dx_m": 4.0, "dy_m": 4.0, "nx": 500, "ny": 500, "spec_zone": 1, "relax_zone": 4,
                      "u_move_m_s": 16.5, "v_move_m_s": 16.5},
            "window": {"nx": NX, "ny": NY, "nz": NZ},
            "zh_cm1_indexed": [-zh[0]] + zh,
            "fields": [{"name": "uadv", "shape_zyx": [NZ + 2, NY + 6, NX + 7], "offset_words": 0},
                       {"name": "vadv", "shape_zyx": [NZ + 2, NY + 7, NX + 6],
                        "offset_words": (NZ + 2) * (NY + 6) * (NX + 7)}] + receipt["fields"][2:],
            "record_bytes": receipt["record_bytes"]}
    whole = tmp / "whole"
    _write(whole, recs, real)
    a = [r for r in recs if r[0] <= split_at]
    b = [r for r in recs if r[0] > split_at]
    _write(tmp / "segA", a, None)
    _write(tmp / "segB", b, None, {"l0_index_first": len(a)})
    ref = tmp / "ref"
    _write(ref, b[:3], real)
    spec = {"roots": {"t": str(tmp)},
            "window_segments": {
                "deck": "t:deck.toml", "reference_stream": "t:ref", "nest_start_s": NEST_START,
                "parent_dx_m": 100.0, "clean_t_max": 1e9,
                "segments": [{"stream": "t:segA", "t_max": split_at, "index_offset": 0},
                             {"stream": "t:segB", "t_min": split_at + 1e-3, "index_offset_from": "README.json",
                              "index_offset_key": "l0_index_first"}],
                "spans": [{"label": "all", "t_min": None, "t_max": None},
                          {"label": "early", "t_min": None, "t_max": 2520.0, "t_max_exclusive": True},
                          {"label": "late", "t_min": 2520.0, "t_max": None}]}}
    (tmp / "spec.json").write_text(json.dumps(spec))
    return {"spec": tmp / "spec.json", "whole": whole, "n": len(recs), "real": real, "receipt": receipt}


def _drop_wall(tsv: Path) -> list[list[str]]:
    lines = tsv.read_text().splitlines()
    head = lines[0].split("\t")
    keep = [i for i, h in enumerate(head) if h != "wall_s"]
    return [[ln.split("\t")[i] for i in keep] for ln in lines]


def test_frozen_files_unchanged() -> None:
    assert SC.frozen_check()["matches_frozen"]


def test_derived_receipt_matches_a_real_one_and_refuses_a_different_one(tmp_path: Path) -> None:
    st = _stream(tmp_path, "burgers", 60.0, 50.0, 2502.0)
    assert W.compare_receipt(st["receipt"], st["real"]) == []
    bad = json.loads(json.dumps(st["real"]))
    bad["child"]["relax_zone"] = 3
    assert any("relax_zone" in x for x in W.compare_receipt(st["receipt"], bad))
    bad = json.loads(json.dumps(st["real"]))
    bad["zh_cm1_indexed"][3] += 1.0
    assert any("zh_cm1_indexed" in x for x in W.compare_receipt(st["receipt"], bad))


def test_adapter_equals_the_frozen_scorer_on_one_stream(tmp_path: Path) -> None:
    st = _stream(tmp_path, "sullivan", 60.0, 50.0, 2502.0)
    direct = tmp_path / "direct"
    assert SC.main([str(st["whole"]), "--out", str(direct), "--nest-start-s", f"{NEST_START:g}",
                    "--parent-dx", "100"]) == 0
    spec = W.load_spec(st["spec"], [])
    runs = tmp_path / "runs"
    rec = W.score_span(spec, "all", runs, 1)
    assert rec["status"] == "scored" and rec["adapter"]["records"] == st["n"]
    vd = json.loads((direct / "verdict.json").read_text())
    va = json.loads((runs / "all" / "verdict.json").read_text())
    assert vd["outcome"] == va["outcome"] == "TORNADO", (vd, va)
    assert va["matches_frozen"] and va["protocol_conforming"]
    for k in ("episodes", "longest_qualifying_run_s", "longest_qualifying_or_unresolved_run_s", "class_counts",
              "condition_counts", "median_interval_s", "genesis_model_time_s"):
        assert vd[k] == va[k], k
    assert _drop_wall(direct / "samples.tsv") == _drop_wall(runs / "all" / "samples.tsv")
    assert (direct / "candidates.tsv").read_text() == (runs / "all" / "candidates.tsv").read_text()
    # ground-relative hazard wind: the frame's 16.5 m/s is added (Vmax 60 m/s frame-relative, 10 m level)
    lines = (runs / "all" / "samples.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    gw = [float(ln.split("\t")[head.index("ground_wind_max_m_s")]) for ln in lines[1:]]
    vm = [float(ln.split("\t")[head.index("vmax_m_s")]) for ln in lines[1:]]
    assert min(g - v for g, v in zip(gw, vm)) > 10.0
    # the ground centre is continuous across the 100 m window jumps (the vortex is still in the frame)
    gx = [float(ln.split("\t")[head.index("centre_ground_x_m")]) for ln in lines[1:]]
    t = [float(ln.split("\t")[head.index("model_time_s")]) for ln in lines[1:]]
    assert max(abs((x - 16.5 * tt) - (gx[0] - 16.5 * t[0])) for x, tt in zip(gx, t)) <= 2.0
    # the second segment's records carry the trajectory's own record numbers
    idx = [int(ln.split("\t")[-1]) for ln in lines[1:]]
    assert idx == list(range(st["n"]))


def test_spans_read_only_their_records(tmp_path: Path) -> None:
    st = _stream(tmp_path, "burgers", 60.0, 50.0, 2502.0)
    spec = W.load_spec(st["spec"], [])
    runs = tmp_path / "runs"
    early = W.score_span(spec, "early", runs, 1)
    late = W.score_span(spec, "late", runs, 1)
    ve = json.loads((runs / "early" / "verdict.json").read_text())
    vl = json.loads((runs / "late" / "verdict.json").read_text())
    assert ve["record_end_s"] < 2520.0 and vl["record_start_s"] == 2520.0
    assert early["adapter"]["records"] + late["adapter"]["records"] == st["n"]
    # each span is shorter than 120 s of one tracked vortex, so neither alone is a TORNADO
    assert ve["outcome"] != "TORNADO" and vl["outcome"] != "TORNADO"


def test_negative_control_through_the_adapter(tmp_path: Path) -> None:
    st = _stream(tmp_path, "burgers", 15.0, 150.0, 2502.0)   # weak Burgers R150 V15: NO_TORNADO on every grid
    spec = W.load_spec(st["spec"], [])
    W.score_span(spec, "all", tmp_path / "runs", 1)
    assert json.loads((tmp_path / "runs" / "all" / "verdict.json").read_text())["outcome"] == "NO_TORNADO"


def test_refuses_overlapping_segments_and_a_changed_reference(tmp_path: Path) -> None:
    st = _stream(tmp_path, "burgers", 60.0, 50.0, 2502.0)
    spec = W.load_spec(st["spec"], [])
    spec["window_segments"]["segments"][1].update(stream="t:segA", t_min=None, index_offset_from=None)
    with pytest.raises(W.AdapterError):
        W.trajectory(spec)
    spec = W.load_spec(st["spec"], [])
    rows = (tmp_path / "ref" / "window-records.tsv").read_text().splitlines()
    rows[1] = "\t".join(rows[1].split("\t")[:-1] + ["0" * 64])
    (tmp_path / "ref" / "window-records.tsv").write_text("\n".join(rows) + "\n")
    receipt, segs = W.trajectory(spec)
    with pytest.raises(W.AdapterError):
        W.reference_check(spec, receipt, segs)


def test_v1_beside_at_window_cadence(tmp_path: Path) -> None:
    # v1 (reported beside v2, deciding nothing) on window records: its child-centre corner and 300 m disk must lie
    # inside the window; a strong Burgers R100 V60 vortex on the child's centre for 72 s is a v1 TORNADO (60 s).
    nx = ny = 160
    i0 = 170                                          # corners 171..329 hold v1's 250 +- 76
    deck = tmp_path / "deck.toml"
    deck.write_text(DECK)
    receipt = W.deck_receipt(deck, nx, ny, NZ)
    zh = [0.5 * (a + b) for a, b in zip(FACES, FACES[1:])][:NZ]
    f = np.log(np.maximum(np.array(zh), 0.2) / Z0) / math.log(10.0 / Z0)
    origin = (30000.0, 31000.0)
    recs = []
    for q in range(13):
        t = 2466.0 + 6.0 * q
        comp = S.VortexComponent("burgers", 60.0, 100.0, 320.0, 320.0, drift=False)
        pl = S.make_plane([comp], [], nx * DX, DX, t)
        blocks = []
        for name, shp in W.v3_shapes(nx, ny, NZ):
            a = np.zeros(shp, "<f4")
            for k in range(NZ):
                if name == "uadv":
                    a[k + 1, NG:NG + ny, NG:NG + nx + 1] = f[k] * pl.u
                elif name == "vadv":
                    a[k + 1, NG:NG + ny + 1, NG:NG + nx] = f[k] * pl.v
            blocks.append(a.tobytes())
        recs.append((t, b"".join(blocks)))
    root = tmp_path / "seg"
    root.mkdir()
    rows, off = ["\t".join(HEAD)], 0
    with open(root / "window-stream.f32", "wb") as fh:
        for q, (t, blob) in enumerate(recs):
            fh.write(blob)
            rows.append("\t".join(str(x) for x in [q, 0, 0, repr(t), off, len(blob), i0, i0, nx, ny, NZ,
                                                   origin[0], origin[1], 0, 0, 0, 0, "-"]))
            off += len(blob)
    (root / "window-records.tsv").write_text("\n".join(rows) + "\n")
    assert receipt["record_bytes"] == len(recs[0][1])
    spec = {"roots": {"t": str(tmp_path)},
            "window_segments": {"deck": "t:deck.toml", "nest_start_s": NEST_START, "parent_dx_m": 100.0,
                                "segments": [{"stream": "t:seg"}],
                                "spans": [{"label": "all", "t_min": None, "t_max": None}]}}
    runs = tmp_path / "runs"
    runs.mkdir()
    v1 = W.v1_window(spec, "all", runs)
    assert v1["records"] == 13 and v1["verdict"] == "TORNADO", v1
    # a window too small to hold v1's disk is refused, not zero-filled into the sample
    rows2 = [rows[0]] + ["\t".join(r.split("\t")[:6] + ["200", "200"] + r.split("\t")[8:]) for r in rows[1:]]
    (root / "window-records.tsv").write_text("\n".join(rows2) + "\n")
    with pytest.raises(W.AdapterError):
        W.v1_window(spec, "all", runs)
