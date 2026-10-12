#!/usr/bin/env python3
"""The few lines of a ``gpuwm-cm1 cartesian DECK --plan`` answer a person needs before a run.

    gpuwm-cm1 cartesian DECK --plan --integrate ... > plan.json
    python3 tools/plan_summary.py plan.json

Prints what the plan says about the clock (steps, frames, restarts), the device memory it lists for the
parent and for a nest child, the child's grid and step, and the storm-track check.  It reads the plan's
JSON and nothing else: no GPU, no model.

The memory figure is the plan's own list of model arrays.  A run also holds the CUDA context, work space
and a memory pool, so the card's peak is higher: about 4 GiB higher in the two runs it was measured on
(a 640 x 768 x 80 parent alone, and the same parent with a 500 x 500 x 103 child).

Unsupported source tool: it needs only Python, and a plan written by gpuwm-cm1 0.1.0 or later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = float(2 ** 30)
#: What the two measured runs held on the card above the plan's listed bytes.
MEASURED_OVERHEAD_GIB = 4.0


def summary(plan: dict) -> dict:
    cadence = plan["integration_cadence"]
    schedule = plan["run_schedule"]
    parent = plan["dynamics"]["maximum_listed_phase_payload_bytes"] / GIB
    grid = plan["dynamics"]["geometry"]
    out = {
        "status": plan.get("status"),
        "start_s": schedule["start_model_time_s"],
        "end_s": cadence["duration_s"],
        "parent_steps": int(cadence["total_steps"]) - int(schedule["start_step"]),
        "frames": schedule["history_count"],
        "restarts": schedule["restart_count"],
        "parent_grid": [grid.get("nx"), grid.get("ny"), grid.get("nz")],
        "parent_listed_gib": round(parent, 2),
        "listed_gib": round(parent, 2),
        "storm_track": (plan.get("run_geometry") or {}).get("status"),
        "nest": None,
    }
    nest = plan.get("nest")
    if nest:
        allocation = plan["nest_allocation"]
        child = allocation["dynamics"]["maximum_listed_phase_payload_bytes"] / GIB
        out["nest"] = {
            "cells": [nest["nx"], nest["ny"], allocation["closure"]["geometry"]["nz"]],
            "dx_m": nest["dx_m"],
            "child_dt_s": nest["child_dt_s"],
            "child_steps_per_parent_step": nest["child_steps_per_parent_step"],
            "south_west_corner_m": [nest["origin_x_m"], nest["origin_y_m"]],
            "listed_gib": round(child, 2),
            "courant_one_wind_m_s": allocation["config"].get("advective_courant_target_speed_m_s"),
            "follows": (nest.get("storm_following") or {}).get("follow_tracker"),
        }
        out["listed_gib"] = round(parent + child, 2)
    return out


def render(row: dict) -> str:
    grid = row["parent_grid"]
    lines = [
        f"status        {row['status']}",
        f"model time    {row['start_s']:g} to {row['end_s']:g} s: {row['parent_steps']} parent steps, "
        f"{row['frames']} frames, {row['restarts']} restarts",
        f"parent        {grid[0]} x {grid[1]} x {grid[2]} cells, {row['parent_listed_gib']:.2f} GiB listed",
    ]
    nest = row["nest"]
    if nest:
        cells = nest["cells"]
        lines.append(f"child         {cells[0]} x {cells[1]} x {cells[2]} cells at {nest['dx_m']:g} m, "
                     f"{nest['listed_gib']:.2f} GiB listed")
        lines.append(f"child step    {nest['child_dt_s']:g} s ({nest['child_steps_per_parent_step']} per parent step); "
                     f"a wind of {nest['courant_one_wind_m_s']:g} m/s crosses one child cell per step")
        lines.append(f"child corner  ({nest['south_west_corner_m'][0]:g}, {nest['south_west_corner_m'][1]:g}) m "
                     f"from the parent's south-west corner; follows: {nest['follows']}")
    lines.append(f"device memory {row['listed_gib']:.2f} GiB listed; allow about {MEASURED_OVERHEAD_GIB:g} GiB more "
                 f"on the card (about {row['listed_gib'] + MEASURED_OVERHEAD_GIB:.0f} GiB)")
    lines.append(f"storm track   {row['storm_track']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path, help="the JSON a cartesian --plan command printed")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"plan_summary: cannot read {args.plan}: {error}. A refused plan prints its reason to the "
              "terminal and leaves this file empty.", file=sys.stderr)
        return 2
    if plan.get("schema") != "gpuwm-cm1.cartesian-plan/v1":
        print(f"plan_summary: {args.plan} is not a cartesian plan (schema {plan.get('schema')!r})", file=sys.stderr)
        return 2
    row = summary(plan)
    print(json.dumps(row, indent=1) if args.json else render(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
