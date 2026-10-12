#!/usr/bin/env python3
"""Where and when to start a nest: read a parent run's frames, propose parent_i0, parent_j0 and a start time.

    python3 tools/nest_place.py FRAMES_DIR [--restarts RESTART_DIR] [--span 20]
        [--near X_M,Y_M] [--search-radius-m 8000] [--json]

FRAMES_DIR is the folder a parent-only ``gpuwm-cm1 cartesian --integrate --history-dir`` run wrote
(``frame-NNNNNN`` folders).  RESTART_DIR is the same run's ``--restart-dir``.  Nothing here opens a GPU.

WHAT IT DOES, for every frame:

1. takes the vertical vorticity at the level nearest ``--height-m`` (10 m) and averages it over boxes
   100 m, 300 m and 500 m wide.  A box average is the circulation around the box divided by its area,
   so a single noisy cell counts for little in the wider boxes;
2. looks for the largest value of each, inside ``--search-radius-m`` of the storm's rotating updraft
   (the largest updraft helicity in the frame) or of ``--near X_M,Y_M`` when that is given, and never
   closer than ``--edge-m`` to the side of the domain;
3. calls the frame "tight" when the three maxima sit within ``--spread-max-m`` (150 m) of each other
   and the 300 m value is at least ``--zeta-min`` (0.05 per second): one compact vortex, not a line of
   eddies along a gust front.

Frames more than two minutes apart are only screened: the tool lists the times with tight rotation and
stops there, because a placement needs closely spaced frames.

A start is proposed at the first time whose frame, the frame before it and the frame after it are all
tight and whose 300 m maximum moved no faster than ``--speed-max-m-s`` (11.7 m/s) from one to the next:
a vortex that is compact and moving smoothly.  With ``--restarts`` only times that have a restart are
considered, and the restart's number is printed.  The child is centred on the 300 m maximum:
``parent_i0 = its cell - span / 2``, and the same for ``parent_j0``.

This is the method the 4 m tornado run's nest was placed with, with one difference: that run searched
around the vortex an earlier 20 m nest had found, and this tool searches around the updraft instead.

It proposes; it does not decide.  Look at the frames (``gpuwm-cm1 quicklook``) before spending GPU
hours on its answer, and check the answer with ``gpuwm-cm1 cartesian DECK --plan``.

Unsupported source tool: run it from a clone of the repository with gpuwm-cm1 installed
(pip install gpuwm-cm1). It uses the installed package when there is one and this
checkout's src/ otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The installed gpuwm-cm1 when there is one (a repository clone carries no built native
# tools); this checkout's src/ otherwise.
if importlib.util.find_spec("cm1core") is None:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

SCHEMA = "gpuwm-cm1.nest-placement/v1"
BOXES_M = (100.0, 300.0, 500.0)


def box_mean(field: np.ndarray, cells: int) -> np.ndarray:
    """The mean over a cells x cells box centred on each point (cells is odd); NaN where the box leaves the grid."""

    if cells <= 1:
        return field.astype(np.float64)
    half = cells // 2
    total = np.zeros((field.shape[0] + 1, field.shape[1] + 1), dtype=np.float64)
    total[1:, 1:] = np.cumsum(np.cumsum(field, axis=0, dtype=np.float64), axis=1)
    out = np.full(field.shape, np.nan, dtype=np.float64)
    ny, nx = field.shape
    if ny < cells or nx < cells:
        return out
    inner = (total[cells:, cells:] - total[:-cells, cells:] - total[cells:, :-cells] + total[:-cells, :-cells])
    out[half:ny - half, half:nx - half] = inner / float(cells * cells)
    return out


def box_cells(width_m: float, dx_m: float) -> int:
    """The odd number of cells nearest a box width_m wide."""

    cells = max(1, int(round(width_m / dx_m)))
    return cells if cells % 2 else cells + 1


def read_restart_times(directory: Path) -> list[tuple[int, float]]:
    rows = []
    for path in sorted(directory.glob("restart-*")):
        match = re.fullmatch(r"restart-(\d+)", path.name)
        record = path / "restart-publication.json"
        if match and record.is_file():
            rows.append((int(match.group(1)), float(json.loads(record.read_text(encoding="utf-8"))["model_time_s"])))
    return rows


def frame_row(directory: Path, *, height_m: float, near: tuple[float, float] | None, radius_m: float,
              edge_m: float, spread_max_m: float, zeta_min: float) -> dict | None:
    from cm1core.quicklook.frame import QuickFrame

    try:
        frame = QuickFrame.open(directory)
    except Exception as error:  # noqa: BLE001 - an unfinished or foreign folder is skipped, by name
        print(f"  {directory.name}: skipped ({error})", file=sys.stderr)
        return None
    try:
        dx, dy = float(frame.dx), float(frame.dy)
        level = frame.level_near(height_m)
        zeta, source = frame.zeta_level(level)
        zeta = np.asarray(zeta, dtype=np.float32)
        ny, nx = zeta.shape
        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) + 0.5) * dy
        anchor, anchor_from = near, "--near"
        if anchor is None:
            try:
                helicity, _ = frame.uh()
                helicity = np.asarray(helicity, dtype=np.float32)
                j, i = np.unravel_index(int(np.nanargmax(helicity)), helicity.shape)
                if float(helicity[j, i]) > 0.0:
                    anchor, anchor_from = (float(x[i]), float(y[j])), "the largest updraft helicity"
            except Exception:  # noqa: BLE001 - a frame without the winds for it has no updraft to search around
                anchor = None
        inside = ((x[None, :] >= edge_m) & (x[None, :] <= nx * dx - edge_m)
                  & (y[:, None] >= edge_m) & (y[:, None] <= ny * dy - edge_m))
        if anchor is None:
            anchor_from = "none (the whole domain was searched)"
        else:
            inside &= (x[None, :] - anchor[0]) ** 2 + (y[:, None] - anchor[1]) ** 2 <= radius_m ** 2
        if not inside.any():
            return {"frame": directory.name, "time_s": float(frame.t), "tight": False, "note": "nothing to search"}
        found = {}
        for width in BOXES_M:
            mean = box_mean(zeta, box_cells(width, dx))
            mean = np.where(inside & np.isfinite(mean), mean, -np.inf)
            j, i = np.unravel_index(int(np.argmax(mean)), mean.shape)
            found[width] = (float(x[i]), float(y[j]), float(mean[j, i]), int(i), int(j))
        x3, y3, value3, i3, j3 = found[300.0]
        spread = max(math.hypot(found[w][0] - x3, found[w][1] - y3) for w in (100.0, 500.0))
        return {
            "frame": directory.name,
            "time_s": float(frame.t),
            "level_m": float(frame.zh[level]),
            "vorticity_from": source,
            "search_centre_m": None if anchor is None else [round(anchor[0]), round(anchor[1])],
            "search_centre_from": anchor_from,
            "box300": {"x_m": round(x3), "y_m": round(y3), "zeta_s_inv": round(value3, 4), "i": i3, "j": j3},
            "box100_zeta_s_inv": round(found[100.0][2], 4),
            "box500_zeta_s_inv": round(found[500.0][2], 4),
            "spread_m": round(spread),
            "tight": bool(spread <= spread_max_m and value3 >= zeta_min),
            "grid": {"nx": int(nx), "ny": int(ny), "dx_m": dx, "dy_m": dy},
        }
    finally:
        frame.close()


def propose(rows: list[dict], restarts: list[tuple[int, float]] | None, *, span: int,
            speed_max_m_s: float) -> dict | None:
    """The first time whose own frame and its two neighbours are tight and moving smoothly."""

    by_time = {round(row["time_s"], 3): index for index, row in enumerate(rows)}
    if restarts is None:
        candidates = [(None, row["time_s"]) for row in rows]
    else:
        candidates = [(number, time_s) for number, time_s in restarts]
    for number, time_s in candidates:
        index = by_time.get(round(time_s, 3))
        if index is None:
            continue
        if 0 < index < len(rows) - 1:
            triple = rows[index - 1:index + 2]
        elif index == 0 and len(rows) >= 3:
            triple = rows[0:3]
        else:
            continue
        if not all(row.get("tight") for row in triple):
            continue
        smooth = True
        for first, second in zip(triple[:-1], triple[1:]):
            moved = math.hypot(second["box300"]["x_m"] - first["box300"]["x_m"],
                               second["box300"]["y_m"] - first["box300"]["y_m"])
            if moved > speed_max_m_s * abs(second["time_s"] - first["time_s"]):
                smooth = False
        if not smooth:
            continue
        here = rows[index]
        grid = here["grid"]
        i0 = here["box300"]["i"] - span // 2
        j0 = here["box300"]["j"] - span // 2
        proposal = {
            "rstnum": number,
            "start_s": here["time_s"],
            "centre_m": [here["box300"]["x_m"], here["box300"]["y_m"]],
            "zeta300_s_inv": here["box300"]["zeta_s_inv"],
            "parent_i0": i0,
            "parent_j0": j0,
            "parent_span": span,
        }
        margin = 20
        if not (margin <= i0 and i0 + span <= grid["nx"] - margin and margin <= j0 and j0 + span <= grid["ny"] - margin):
            proposal["warning"] = f"the child would start within {margin} parent cells of the side of the domain"
        return proposal
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", type=Path, help="the parent run's --history-dir")
    parser.add_argument("--restarts", type=Path, help="the parent run's --restart-dir")
    parser.add_argument("--span", type=int, default=20, help="[nest] parent_span_x and parent_span_y (default 20)")
    parser.add_argument("--near", help="search around X_M,Y_M (metres from the domain's south-west corner)")
    parser.add_argument("--search-radius-m", type=float, default=8000.0)
    parser.add_argument("--height-m", type=float, default=10.0)
    parser.add_argument("--edge-m", type=float, default=5000.0)
    parser.add_argument("--zeta-min", type=float, default=0.05)
    parser.add_argument("--spread-max-m", type=float, default=150.0)
    parser.add_argument("--speed-max-m-s", type=float, default=11.7)
    parser.add_argument("--json", action="store_true", help="print the whole record as JSON")
    args = parser.parse_args(argv)

    near = None
    if args.near:
        parts = args.near.split(",")
        if len(parts) != 2:
            parser.error("--near is X_M,Y_M")
        near = (float(parts[0]), float(parts[1]))
    folders = sorted(path for path in args.frames.glob("frame-*") if path.is_dir())
    if not folders:
        print(f"nest_place: no frame-NNNNNN folders in {args.frames}", file=sys.stderr)
        return 2
    rows = []
    for folder in folders:
        row = frame_row(folder, height_m=args.height_m, near=near, radius_m=args.search_radius_m,
                        edge_m=args.edge_m, spread_max_m=args.spread_max_m, zeta_min=args.zeta_min)
        if row is None:
            continue
        rows.append(row)
        if not args.json and "box300" in row:
            box = row["box300"]
            print(f"  {row['frame']}  t = {row['time_s']:7.1f} s   300 m box {box['zeta_s_inv']:.3f} /s at "
                  f"({box['x_m'] / 1000.0:6.2f}, {box['y_m'] / 1000.0:6.2f}) km, cell ({box['i']}, {box['j']});  "
                  f"100 m {row['box100_zeta_s_inv']:.3f}, 500 m {row['box500_zeta_s_inv']:.3f};  "
                  f"spread {row['spread_m']:4d} m   {'tight' if row['tight'] else 'not tight'}", flush=True)
    rows = [row for row in rows if "box300" in row]
    if not rows:
        print("nest_place: no readable frame", file=sys.stderr)
        return 2
    rows.sort(key=lambda row: row["time_s"])
    restarts = read_restart_times(args.restarts) if args.restarts else None
    if restarts and not args.json:
        print()
        print("Restarts: " + ", ".join(f"{number} at {time_s:g} s" for number, time_s in restarts) + ".")
    proposal = propose(rows, restarts, span=args.span, speed_max_m_s=args.speed_max_m_s)
    intervals = sorted({round(b["time_s"] - a["time_s"], 3) for a, b in zip(rows[:-1], rows[1:])})
    record = {
        "schema": SCHEMA,
        "frames_directory": str(args.frames),
        "restarts": None if restarts is None else [{"rstnum": n, "model_time_s": t} for n, t in restarts],
        "rule": {
            "field": f"vertical vorticity at the level nearest {args.height_m:g} m, averaged over 100, 300 and 500 m boxes",
            "search": f"within {args.search_radius_m:g} m of the search centre, at least {args.edge_m:g} m from the side",
            "tight": f"the three maxima within {args.spread_max_m:g} m of each other and the 300 m value at least "
                     f"{args.zeta_min:g} per second",
            "start": "the first time whose frame and its two neighbours are tight and whose 300 m maximum moved no "
                     f"faster than {args.speed_max_m_s:g} m/s between them",
            "placement": "the child is centred on the 300 m maximum: parent_i0 and parent_j0 are its cell less half the span",
        },
        "frames": rows,
        "proposal": proposal,
    }
    if args.json:
        print(json.dumps(record, indent=1))
        return 0 if proposal else 1
    if intervals and max(intervals) > 120.0:
        tight = [row["time_s"] for row in rows if row["tight"]]
        print()
        print(f"These frames are up to {max(intervals):g} s apart: enough to see when there is rotation near the "
              "ground, too coarse to place a nest.")
        if tight:
            print("Tight rotation at t = " + ", ".join(f"{t:g}" for t in tight) + " s.")
            print("Continue the parent from a restart a few minutes before the time you want, with a frame every "
                  "60 s, and run this tool again on those frames.")
        else:
            print("No frame shows tight rotation near the ground. Run the parent further, or pass --near X_M,Y_M.")
        return 0 if tight else 1
    if proposal is None:
        print("\nNo start is proposed: no time has three tight frames in a row with a smoothly moving maximum"
              + (" at a restart." if restarts is not None else "."))
        print("Run the parent further, or look at the frames and pass --near X_M,Y_M.")
        return 1
    print(f"\nProposed start: t = {proposal['start_s']:g} s, 300 m box vorticity {proposal['zeta300_s_inv']:.3f} /s at "
          f"({proposal['centre_m'][0]}, {proposal['centre_m'][1]}) m.")
    if "warning" in proposal:
        print(f"WARNING: {proposal['warning']}.")
    print("Put this in the nested deck:\n")
    print("  [run]")
    print(f"  rstnum = {proposal['rstnum']}" if proposal["rstnum"] is not None
          else "  rstnum = (the number of the restart written at that time; pass --restarts to have it printed)")
    print("  [nest]")
    print(f"  parent_i0 = {proposal['parent_i0']}")
    print(f"  parent_j0 = {proposal['parent_j0']}")
    print(f"  parent_span_x = {proposal['parent_span']}")
    print(f"  parent_span_y = {proposal['parent_span']}")
    print("  [lagrangian.nest_child_window]   (if the deck has one)")
    print(f"  start_s = {proposal['start_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
