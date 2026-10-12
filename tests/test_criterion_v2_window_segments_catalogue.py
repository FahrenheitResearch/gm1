"""The window-segments adapter on every synthetic case v2 was frozen with, at the 4 m grid of E100R2-PF-4m.

No GM1 run is read.  Each case of ``synthetic_v2.case`` (the pre-freeze catalogue: 8 positives, the threshold and
resolvedness cases, 13 negative controls and the two hopping cases) is sampled at dx = 4 m exactly as
``synthetic_v2.run_series`` samples it, and each plane is written as a schema-v3 window record (frame-relative winds,
the frame moving at 16.5 m/s, the same plane on the two scalar levels bracketing 10 m).  The records are written
twice: as one stream with a receipt, read by the frozen ``score.py`` directly, and as two receipt-less segments (the
second with its own index from 0 and a README giving its offset), read through the adapter with the receipt derived
from a deck.  The adapter must give the frozen scorer's samples, candidates and outcome, and that outcome must be one
the catalogue expects on a 4 m grid (``synthetic_v2.expected``).

About 5 CPU-minutes; it runs when GM1_CRITERION_CATALOGUE=1 (the default suite keeps the fast adapter tests only).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_criterion_v2_window_segments import S, SC, W, _drop_wall  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("GM1_CRITERION_CATALOGUE") != "1",
                                reason="the synthetic catalogue through the adapter: set GM1_CRITERION_CATALOGUE=1")

NG = 3
DX = 4.0
FACES = [0.0, 8.0, 16.0, 24.0, 32.0]     # scalar levels 4, 12, 20, 28 m: 10 m lies between 4 and 12 m
NZ = 3
NEST_START = -10000.0                    # no record is inside the nest spin-up

HEAD = ["index", "child_step", "parent_step", "model_time_s", "offset_bytes", "bytes", "i0", "j0", "nx", "ny", "nz",
        "origin_x_m", "origin_y_m", "centre_x_m", "centre_y_m", "centre_held", "clamped", "sha256"]


def _deck(n: int) -> str:
    # child nx = parent_span x ratio = n, child dx = 100 / 25 = 4 m, boundary 1 + 4 = synthetic_v2.BOUNDARY_CELLS
    assert n % 25 == 0 and S.BOUNDARY_CELLS == 5 and S.FRAME_M_S == (16.5, 16.5)
    return ("[grid]\ndx_m = 100.0\ndy_m = 100.0\n[closures]\nmoving_frame = true\nu_move_m_s = 16.5\n"
            "v_move_m_s = 16.5\n[nest]\nrefinement_ratio = 25\nparent_span_x = %d\nparent_span_y = %d\n"
            "spec_zone = 1\nrelax_zone = 4\nchild_nz = 4\nchild_levels_m = [%s]\n"
            % (n // 25, n // 25, ", ".join(f"{f:.1f}" for f in FACES)))


def _write(root: Path, recs: list[tuple[float, np.ndarray, np.ndarray]], n: int, receipt: dict | None,
           readme: dict | None = None) -> None:
    """Window records of the whole child (i0 = j0 = 0): uadv and vadv hold the plane; the other blocks are zero."""
    root.mkdir(parents=True)
    shapes = W.v3_shapes(n, n, NZ)
    rec_bytes = 4 * sum(int(np.prod(s)) for _, s in shapes)
    rows, off = ["\t".join(HEAD)], 0
    with open(root / "window-stream.f32", "wb") as fh:
        for q, (t, u, v) in enumerate(recs):
            ua = np.zeros(shapes[0][1], "<f4")
            va = np.zeros(shapes[1][1], "<f4")
            for k in (1, 2):                                   # CM1 levels 1, 2 (4 and 12 m)
                ua[k, NG:NG + n, NG:NG + n + 1] = u
                va[k, NG:NG + n + 1, NG:NG + n] = v
            fh.seek(off)
            fh.write(ua.tobytes())
            fh.write(va.tobytes())
            rows.append("\t".join(str(x) for x in [q, 0, 0, repr(t), off, rec_bytes, 0, 0, n, n, NZ,
                                                   0.0, 0.0, 0.0, 0.0, 0, 0, "-"]))
            off += rec_bytes
        fh.truncate(off)                                       # the zero blocks stay sparse
    (root / "window-records.tsv").write_text("\n".join(rows) + "\n")
    if receipt is not None:
        (root / "window-stream.json").write_text(json.dumps(receipt))
    if readme is not None:
        (root / "README.json").write_text(json.dumps(readme))


@pytest.mark.parametrize("name", sorted(S.EXPECTED))
def test_catalogue_case_through_the_adapter(name: str, tmp_path: Path) -> None:
    domain, times, build = S.case(name)
    n = int(round(domain / DX))
    recs = []
    for t in times:
        comps, spikes = build(t)
        pl = S.make_plane(comps, spikes, domain, DX, t)
        recs.append((t, pl.u.astype("<f4"), pl.v.astype("<f4")))
    deck = tmp_path / "deck.toml"
    deck.write_text(_deck(n))
    receipt = W.deck_receipt(deck, n, n, NZ)
    _write(tmp_path / "whole", recs, n, receipt)
    half = len(recs) // 2
    _write(tmp_path / "segA", recs[:half], n, None)
    _write(tmp_path / "segB", recs[half:], n, None, {"l0_index_first": half})
    split = times[half - 1]
    spec = {"roots": {"t": str(tmp_path)},
            "window_segments": {
                "deck": "t:deck.toml", "nest_start_s": NEST_START, "parent_dx_m": 100.0, "clean_t_max": 1e9,
                "segments": [{"stream": "t:segA", "t_max": split, "index_offset": 0},
                             {"stream": "t:segB", "t_min": times[half], "index_offset_from": "README.json",
                              "index_offset_key": "l0_index_first"}],
                "spans": [{"label": "all", "t_min": None, "t_max": None}]}}
    direct = tmp_path / "direct"
    assert SC.main([str(tmp_path / "whole"), "--out", str(direct), "--nest-start-s", f"{NEST_START:g}",
                    "--parent-dx", "100"]) == 0
    runs = tmp_path / "runs"
    rec = W.score_span(spec, "all", runs, 1)
    assert rec["status"] == "scored" and rec["adapter"]["records"] == len(recs)
    vd = json.loads((direct / "verdict.json").read_text())
    va = json.loads((runs / "all" / "verdict.json").read_text())
    assert va["matches_frozen"] and va["protocol_conforming"]
    for k in ("outcome", "episodes", "longest_qualifying_run_s", "longest_qualifying_or_unresolved_run_s",
              "class_counts", "condition_counts", "median_interval_s", "genesis_model_time_s"):
        assert vd[k] == va[k], (name, k, vd[k], va[k])
    assert _drop_wall(direct / "samples.tsv") == _drop_wall(runs / "all" / "samples.tsv")
    assert (direct / "candidates.tsv").read_text() == (runs / "all" / "candidates.tsv").read_text()
    assert va["outcome"] in S.expected(name, DX), (name, va["outcome"], S.expected(name, DX))
