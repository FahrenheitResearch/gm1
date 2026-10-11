"""Frames of a real CM1 storm for the quick-look tests.

``tests/data/cm1-ref-24x24/stats-run-t1500.npz`` is CM1 r21.1's own state of its supercell namelist at 24 x 24 x 40
(1 km) after 1500 s: rain, hail, a 24 m/s updraft and a cold pool (its README says how it was made).  These helpers
publish it as GM1 frames, the way ``gpuwm-cm1 cartesian`` does (``cm1out_support.publish``), with the extras the
quick-looks read: the base-state columns, ``th3d``, accumulated rain, the swath words and the frame's microphysics
record.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from cm1out_support import REF, REF_GRID, canvas_anchor, end_event, publish, round_half_away

ROOT = Path(__file__).resolve().parents[1]
STATE = REF / "stats-run-t1500.npz"
NAMELIST_24 = REF / "stats-run.namelist.input"
STOCK_NAMELIST = ROOT / "tests" / "data" / "cm1import" / "r21.1-supercell.namelist.input"
MASS = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBER = ("nci", "ncs", "ncr", "ncg")
DT = 6.0
TOTAL_STEPS = 400


def storm(time_s: float, *, scale: float = 1.0, swath: str = "canvas", uh: bool = True,
          dbz: np.ndarray | None = None, move: tuple[float, float] = (12.5, 3.0)) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carriers of CM1's 1500 s state (winds scaled by ``scale``) and the frame's clock and motion.

    ``swath``: ``canvas`` (the ground-anchored sws2/shs2), ``domain`` (sws/shs only) or ``none``."""

    d = np.load(STATE)
    nz, ny, nx = 40, 24, 24
    zf = (np.arange(nz + 1, dtype=np.float32) * np.float32(500.0))
    u, v, w = d["u"] * scale, d["v"] * scale, d["w"] * scale
    carriers: dict[str, Any] = {
        "prs": d["prs"], "pp3d": d["pipert"], "rho": d["rho"],
        "q3d": np.stack([d[n] for n in MASS]), "nc3d": np.stack([d[n] for n in NUMBER]),
        "th3d": d["thpert"], "th": d["th"], "u3d": u, "v3d": v, "w3d": w,
        "zf": np.broadcast_to(zf.reshape(-1, 1, 1), (nz + 1, ny, nx)).copy(),
        "rain": (d["qr"][0] * 400.0 * time_s / 1500.0).astype(np.float32),
        "th0": d["th0"], "pi0": d["pi0"], "prs0": d["prs0"], "qv0": d["qv0"],
    }
    uc = 0.5 * (u[0, :, 1:] + u[0, :, :-1]) + np.float32(move[0])
    vc = 0.5 * (v[0, 1:, :] + v[0, :-1, :]) + np.float32(move[1])
    speed = np.hypot(uc, vc).astype(np.float32)
    if uh:
        carriers["uh"] = np.maximum(0.0, w[10] * 20.0).astype(np.float32)
    if dbz is not None:
        carriers["dbz"] = dbz.astype(np.float32)
    frame = dict(time_s=time_s, step=int(round(time_s / DT)), grid=REF_GRID, dt_s=DT, umove=move[0], vmove=move[1],
                 domainloc=(move[0] * time_s, move[1] * time_s))
    if swath == "canvas":
        anchor = canvas_anchor(dt_s=DT, total_steps=TOTAL_STEPS, umove=move[0], vmove=move[1])
        fx = round_half_away(move[0] * DT * TOTAL_STEPS / 1000.0)
        fy = round_half_away(move[1] * DT * TOTAL_STEPS / 1000.0)
        ox = round_half_away(move[0] * time_s / 1000.0) - min(0, fx)
        oy = round_half_away(move[1] * time_s / 1000.0) - min(0, fy)
        shape = (ny + abs(fy) + 1, nx + abs(fx) + 1)
        for name, plane in (("sws2", speed), ("shs2", carriers.get("uh", np.zeros((ny, nx), np.float32)))):
            canvas = np.zeros(shape, np.float32)
            canvas[oy:oy + ny, ox:ox + nx] = plane
            carriers[name] = canvas
        carriers["sws"] = speed
        frame["swath_anchor"] = anchor
    elif swath == "domain":
        carriers["sws"] = speed
    return carriers, frame


def mark_microphysics(frame_dir: Path, component: str = "morrison-mp10", ihail: int = 1) -> None:
    """Add what a published frame records about its microphysics (config.microphysics, config.dense_ice) and
    the species lists its field table's meaning column carries, as the runtime writes them."""

    table = frame_dir / "history-field-table.tsv"
    rows = []
    for line in table.read_text(encoding="ascii").splitlines():
        cols = line.split("\t")
        if cols[0] == "q3d":
            cols[1] = "current " + ",".join(MASS) + " mixing ratios"
        elif cols[0] == "nc3d":
            cols[1] = "current " + ",".join(NUMBER) + " number concentrations"
        rows.append("\t".join(cols))
    table.write_text("\n".join(rows) + "\n", encoding="ascii")
    path = frame_dir / "history-publication.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["config"]["microphysics"] = component
    doc["config"]["dense_ice"] = {"ihail": ihail, "category": "hail" if ihail else "graupel"}
    path.write_text(json.dumps(doc), encoding="utf-8")


def publish_storm(root: Path, index: int, time_s: float, **kwargs: Any) -> Path:
    component = kwargs.pop("component", "morrison-mp10")
    carriers, frame = storm(time_s, **kwargs)
    path = publish(root, index, carriers, **frame)
    mark_microphysics(path, component)
    return path


def run_directory(tmp_path: Path, times=(900.0, 1200.0, 1500.0), *, namelist: Path = NAMELIST_24,
                  name: str = "storm", end: bool = True, **kwargs: Any) -> Path:
    """A ``gpuwm-cm1 cartesian`` output directory (``frames/``, ``events.jsonl``) with a namelist beside it."""

    run = tmp_path / name
    frames = run / "frames"
    frames.mkdir(parents=True)
    shutil.copy(namelist, run / "namelist.input")
    for index, t in enumerate(times):
        publish_storm(frames, index, t, scale=0.6 + 0.4 * t / 1500.0, **kwargs)
    if end:
        end_event(run / "events.jsonl", frames=len(times))
    return run


__all__ = ["NAMELIST_24", "STOCK_NAMELIST", "mark_microphysics", "publish_storm", "run_directory", "storm"]
