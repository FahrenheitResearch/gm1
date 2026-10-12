"""OFFLINE DIAGNOSTIC: backward trajectories through a run's saved nest-child frames.

    python3 tools/nest_backward_trajectories.py \
        --child-root <history-dir>/nest-child --out <new-dir> \
        --seed-frame parent-grid-origin \
        --seed-box x0_m,y0_m,z0_m,dx_m,dy_m,dz_m,nx,ny,nz \
        --start-time-s 3600 --end-time-s 3000 --dt-s 1.0

Seed coordinates are metres from the parent grid origin (the parent's west and
south faces and the ground), the frame the child parcels are published in.
The method, its interpolation error estimate and its citations are in
:mod:`cm1core.nest_parcel_backward` and in the receipt it writes.

Unsupported source tool: run it from a clone of the repository with gpuwm-cm1 installed
(pip install gpuwm-cm1). It uses the installed package when there is one and this
checkout's src/ otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The installed gpuwm-cm1 when there is one (a repository clone carries no built native
# tools); this checkout's src/ otherwise.
if importlib.util.find_spec("cm1core") is None:
    sys.path.insert(0, str(ROOT / "src"))

from cm1core.lagrangian_config import NEST_CHILD_TABLE, parse_lagrangian_table  # noqa: E402
from cm1core.nest_parcel_backward import (  # noqa: E402
    BackwardRequest,
    BackwardTrajectoryError,
    integrate_backward,
    read_child_frames,
    write_backward_trajectories,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-frame", required=True)
    parser.add_argument("--seed-box", required=True, help="x0_m,y0_m,z0_m,dx_m,dy_m,dz_m,nx,ny,nz")
    parser.add_argument("--start-time-s", type=float, required=True)
    parser.add_argument("--end-time-s", type=float, required=True)
    parser.add_argument("--dt-s", type=float, required=True)
    args = parser.parse_args(argv)
    fields = args.seed_box.split(",")
    if len(fields) != 9:
        parser.error("--seed-box needs nine comma-separated values")
    x0, y0, z0, dx, dy, dz = (float(v) for v in fields[:6])
    nx, ny, nz = (int(v) for v in fields[6:])
    table = {NEST_CHILD_TABLE: {
        "seed_frame": args.seed_frame, "nparcels": nx * ny * nz, "prclfrq_s": 1.0,
        "seed_nx": nx, "seed_ny": ny, "seed_nz": nz, "seed_dx_m": dx, "seed_dy_m": dy,
        "seed_dz_m": dz, "seed_x0_m": x0, "seed_y0_m": y0, "seed_z0_m": z0,
    }}
    try:
        seeding = parse_lagrangian_table(table).nest_child
        request = BackwardRequest(seeding, args.start_time_s, args.end_time_s, args.dt_s)
        frames = read_child_frames(args.child_root)
        result = integrate_backward(frames, request)
        receipt = write_backward_trajectories(args.out, result, frames, request)
    except (BackwardTrajectoryError, ValueError) as error:
        print(json.dumps({"status": "refused", "reason": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "written", "kind": receipt["kind"], "out": str(args.out),
        "status_counts": receipt["status_counts"],
        "position_error_estimate_m": receipt["position_error_estimate_m"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
