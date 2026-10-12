"""OFFLINE DIAGNOSTIC: the trajectories that reach a vortex, and their vorticity budget.

    python3 tools/nest_window_trajectories.py \
        --stream <history-dir>/nest-child-window --out <new-dir> \
        --target-time-s 4080 --back-to-s 3960 \
        --centre auto --radius-m 60 --heights-m 10,60,120,200 \
        --rings 5 --per-ring 24 --circuit 36

It reads a :mod:`cm1core.nest_child_window` stream -- the model's own advecting
velocity words, one record per child step -- and

1. seeds parcels *where the vorticity already is*: an equal-area disk of
   ``--radius-m`` about the vortex centre, at each ``--heights-m``, rather than
   on a lattice that a 40-60 m core slips between;
2. integrates them **backward through the stream's own records** -- with no
   time interpolation at all on a stream written every child step
   (``wind_source = "model-step"``), or blended between records placed at the
   midpoints of the steps they average on a strided one
   (``wind_source = "history-blend"``, stepping at the child's own dt) --
   publishing beside every position the
   measured round-trip residual, the sensitivity amplification and the
   position error estimate built from them;
3. samples the vorticity budget at every position of every trajectory and
   checks that it closes, per component (:mod:`cm1core.nest_window_budget`),
   with friction as the curl of the step's held momentum tendency;
4. optionally carries one closed material circuit of ``--circuit`` vertices
   and publishes its circulation at every record.

``--target-time-s`` is the time of one of the stream's records (the times are
the ``model_time_s`` column of ``window-records.tsv``).  A stream needs its
``window-stream.json``, which the run writes when it ends: a run that stopped
on an error leaves the records and their index but nothing this tool can read.

``--centre auto`` takes the centre the stream itself followed at the target
time, which is the tornado criterion's own centre when the run declared
``follow = "tornado-criterion"``.  Every claim, every refusal and every
citation is in the receipts this writes.

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

from cm1core.nest_child_window import (  # noqa: E402
    NestChildWindowError,
    read_window_stream,
)
from cm1core.nest_parcel_backward import (  # noqa: E402
    BackwardRequest,
    BackwardTrajectoryError,
    integrate_backward,
    write_backward_trajectories,
)
from cm1core.nest_window_budget import (  # noqa: E402
    BUDGET_RECEIPT_FILENAME,
    TrajectoryBudgetError,
    budget_along,
    circuit_positions,
    circulation_along,
    core_disk_positions,
    write_trajectory_budget,
)


def _record_at(stream, time_s: float):
    for record in stream.records:
        if abs(record.model_time_s - time_s) <= 1.0e-6 * max(1.0, abs(time_s)):
            return record
    raise NestChildWindowError(
        f"the stream holds no record at model time {time_s} s; its records run "
        f"[{stream.records[0].model_time_s}, {stream.records[-1].model_time_s}] s every "
        f"{stream.dt_s} s, and a target between two of them would have to be interpolated -- "
        "which is the error this stream exists to remove"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", type=Path, required=True,
                        help="the published nest-child-window directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-time-s", type=float, required=True)
    parser.add_argument("--back-to-s", type=float, required=True)
    parser.add_argument("--centre", default="auto",
                        help='"auto" (the centre the stream followed) or "x_m,y_m"')
    parser.add_argument("--radius-m", type=float, default=60.0)
    parser.add_argument("--heights-m", default="10,60,120,200")
    parser.add_argument("--rings", type=int, default=5)
    parser.add_argument("--per-ring", type=int, default=24)
    parser.add_argument("--circuit", type=int, default=0,
                        help="vertices of one closed material circuit, or 0 for none")
    parser.add_argument("--circuit-radius-m", type=float, default=0.0,
                        help="default: --radius-m")
    parser.add_argument("--circuit-height-m", type=float, default=0.0,
                        help="default: the first --heights-m")
    parser.add_argument("--verify", action="store_true",
                        help="check every record against the SHA-256 its index row carries")
    parser.add_argument("--no-budget", action="store_true")
    args = parser.parse_args(argv)

    try:
        stream = read_window_stream(args.stream)
        strided = int(stream.receipt["selectors"]["stride_child_steps"]) > 1
        if strided and any(abs(record.model_time_s - args.target_time_s) <= 1.0e-6
                           for record in stream.records):
            # A record's own time was given.  On a strided stream the record holds the wind
            # averaged over the child step that ends then, so the trajectory starts at that
            # step's midpoint; --back-to-s moves with it, so the span stays what was asked.
            args.target_time_s -= stream.midpoint_shift_s
            args.back_to_s -= stream.midpoint_shift_s
        target = _record_at(stream, args.target_time_s
                            + (stream.midpoint_shift_s if strided else 0.0))
        if args.centre == "auto":
            centre = (target.centre_x_m, target.centre_y_m)
        else:
            parts = args.centre.split(",")
            if len(parts) != 2:
                parser.error('--centre is "auto" or "x_m,y_m"')
            centre = (float(parts[0]), float(parts[1]))
        heights = [float(v) for v in args.heights_m.split(",")]
        disk = core_disk_positions(centre_x_m=centre[0], centre_y_m=centre[1],
                                   radius_m=args.radius_m, heights_m=heights,
                                   rings=args.rings, per_ring=args.per_ring)
        circuit_count = int(args.circuit)
        first_circuit_parcel = disk.shape[1]
        if circuit_count:
            import numpy

            ring = circuit_positions(
                centre_x_m=centre[0], centre_y_m=centre[1],
                radius_m=args.circuit_radius_m or args.radius_m,
                height_m=args.circuit_height_m or heights[0], count=circuit_count)
            positions = numpy.concatenate([disk, ring], axis=1)
        else:
            positions = disk
        stride = int(stream.receipt["selectors"]["stride_child_steps"])
        child_dt = float(stream.receipt["child"]["dt_s"])
        if stride == 1:
            # One record per child step: reverse each with no interpolation.
            shift, source, dt = 0.0, "model-step", stream.dt_s
            margin = 0.5 * stream.dt_s
        else:
            # A strided stream: blend records placed at the MIDPOINT of the
            # step each averages, stepping at the child's own dt.
            shift, source, dt = stream.midpoint_shift_s, "history-blend", child_dt
            margin = 2.0 * stream.dt_s
        frames = [f for f in stream.frames(verify=args.verify, time_shift_s=shift)
                  if args.back_to_s - margin <= f.model_time_s
                  <= args.target_time_s + margin]
        request = BackwardRequest(None, args.target_time_s, args.back_to_s, dt,
                                  wind_source=source, positions=positions)
        result = integrate_backward(frames, request)
        trajectory_receipt = write_backward_trajectories(
            args.out / "trajectories", result, frames, request)
        summary = {
            "status": "written",
            "out": str(args.out),
            "stream": str(args.stream),
            "target_time_s": args.target_time_s,
            "back_to_s": args.back_to_s,
            "centre_m": list(centre),
            "parcels": int(positions.shape[1]),
            "core_disk_parcels": int(disk.shape[1]),
            "status_counts": trajectory_receipt["status_counts"],
            "position_error_estimate_m": trajectory_receipt["position_error_estimate_m"],
        }
        if circuit_count:
            report = circulation_along(result, first_parcel=first_circuit_parcel,
                                       count=circuit_count, name="target-circuit")
            path = args.out / "circulation.json"
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
            summary["circulation"] = {
                "file": str(path), "samples": len(report["samples"]),
                "first_m2_s": report["samples"][0]["circulation_m2_s"] if report["samples"] else None,
                "last_m2_s": report["samples"][-1]["circulation_m2_s"] if report["samples"] else None,
            }
        if not args.no_budget:
            budget = budget_along(stream, result, verify=args.verify, time_shift_s=shift)
            receipt = write_trajectory_budget(args.out / "budget", budget, stream=stream)
            summary["budget"] = {
                "file": str(args.out / "budget" / BUDGET_RECEIPT_FILENAME),
                "closure": receipt["closure"],
                "friction": "curl of the held momentum tendency; see friction_definition",
            }
            if "coriolis_definition" in receipt:
                summary["budget"]["coriolis"] = (
                    "curl of CM1's f-plane Coriolis force re-formed from the stream's winds "
                    "and header; see coriolis_definition")
    except (NestChildWindowError, BackwardTrajectoryError, TrajectoryBudgetError,
            ValueError) as error:
        print(json.dumps({"status": "refused", "reason": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
