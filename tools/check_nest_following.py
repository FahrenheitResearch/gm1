#!/usr/bin/env python3
"""Did a storm-following nest follow what its receipt says it followed?

Reads the ``storm_following`` block (``gm1.cartesian-nest-move/v1``,
``cm1core.nest_move.MovingNest.record``) of a finished ``gpuwm-cm1 cartesian
--integrate`` receipt, on the host, and checks the tracker's own words against
the contract they were accepted under:

* every accepted surface-tracker centre moved, on each axis, no further than
  ``follow_max_speed_m_s * follow_interval_s`` times the follow intervals since
  the previous accepted centre (the window a held sample widens; a centre that
  broke this would be a window edge or another circulation, not the storm).
  Under ``stop``, which has no too-fast test, a centroid may also lie up to
  ``follow_smoothing_radius_m`` past the maximum it was taken around;
* every relocation moved the centre, on each axis, no further than
  ``follow_relocation_radius_m + follow_smoothing_radius_m`` (a maximum inside
  the relocation window, then a centroid over the box around it);
* at most ``--max-relocations`` relocations.

A run that stopped by name wrote no receipt (the refusal goes to stderr and the
exit status is 2), so a receipt with a storm-following block is itself the
"no refusal" evidence; this tool refuses a receipt without one.

``--compare OTHER`` lines the tracker centres of a second receipt up with the
first by parent step and reports their separation: two runs from the same
restart that differ only in ``follow_smoothing_radius_m`` should agree to a
few child cells when both follow the same circulation, and a persistent
separation of several box half-widths means one of them is on another one.

    tools/check_nest_following.py RECEIPT.json [--max-relocations N] [--compare OTHER.json]

Exit status 0 when every check passes, 1 otherwise; the JSON summary is printed
either way.

Unsupported source tool: run it from a clone of the repository with gpuwm-cm1 installed
(pip install gpuwm-cm1). It uses the installed package when there is one and this
checkout's src/ otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA = "gm1.cartesian-nest-move/v1"
SURFACE_TRACKER = "wrf-surface-circulation"
#: Centroid arithmetic is float64 on child-face indices; this absorbs the
#: metre conversion, never a whole child cell.
TOLERANCE_M = 1.0e-3


def storm_following_blocks(document: Any) -> list[dict[str, Any]]:
    """Every ``storm_following`` record in a receipt, wherever it is nested."""

    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("schema") == SCHEMA and "tracker_events" in node:
                found.append(node)
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    return found


def summarize(block: dict[str, Any]) -> dict[str, Any]:
    settings = block["settings"]
    interval_steps = int(settings["interval_parent_steps"])
    reach_m = float(settings["max_speed_m_s"]) * float(settings["interval_s"])
    # hold refuses a centroid beyond the reach (too-fast-hold); stop bounds
    # only the maximum, and the centroid box reaches R past it.
    slack_m = 0.0 if settings.get("on_edge") == "hold" else float(settings["smoothing_radius_m"] or 0.0)
    tracks = [row for row in block["tracker_events"] if row.get("centre_m") is not None]
    relocations = [h for h in block.get("holds", []) if h.get("event") == "relocation"]
    summary: dict[str, Any] = {
        "tracker": settings["tracker"],
        "on_edge": settings.get("on_edge", "stop"),
        "smoothing_radius_m": settings["smoothing_radius_m"],
        "reach_m_per_interval": reach_m,
        "tracker_samples": len(block["tracker_events"]),
        "held_samples": sum(1 for row in block["tracker_events"] if row.get("held") is True),
        "hold_counts": block.get("hold_counts"),
        "holds_not_listed": block.get("holds_not_listed"),
        "max_consecutive_holds": block.get("max_consecutive_holds"),
        "relocations": len(relocations),
        "moves": block.get("move_count"),
        "first_centre_m": tracks[0]["centre_m"] if tracks else None,
        "last_centre_m": tracks[-1]["centre_m"] if tracks else None,
        "centre_steps": [],
        "relocation_displacements_m": [h.get("displacement_m") for h in relocations],
    }
    if settings["tracker"] != SURFACE_TRACKER:
        return summary
    previous = None
    for row in tracks:
        if row.get("held") is True:
            continue
        if previous is not None and row.get("relocated") is not True:
            intervals = max(1, math.ceil((row["parent_step"] - previous["parent_step"]) / interval_steps))
            dx = row["centre_m"][0] - previous["centre_m"][0]
            dy = row["centre_m"][1] - previous["centre_m"][1]
            summary["centre_steps"].append({
                "from_parent_step": previous["parent_step"],
                "to_parent_step": row["parent_step"],
                "intervals": intervals,
                "axis_distance_m": max(abs(dx), abs(dy)),
                "allowed_m": intervals * reach_m + slack_m,
            })
        previous = row
    return summary


def failures(block: dict[str, Any], summary: dict[str, Any], *, max_relocations: int) -> list[str]:
    settings = block["settings"]
    problems = []
    if not block["tracker_events"]:
        problems.append("the tracker never sampled")
    for step in summary["centre_steps"]:
        if step["axis_distance_m"] > step["allowed_m"] + TOLERANCE_M:
            problems.append(
                f"the accepted centre moved {step['axis_distance_m']!r} m on an axis between parent "
                f"steps {step['from_parent_step']} and {step['to_parent_step']}, beyond the "
                f"{step['allowed_m']!r} m {step['intervals']} follow interval(s) reach"
            )
    if summary["relocations"] > max_relocations:
        problems.append(f"{summary['relocations']} relocations, more than --max-relocations {max_relocations}")
    radius = settings.get("relocation_radius_m")
    if radius is not None:
        bound = float(radius) + float(settings["smoothing_radius_m"])
        for displacement in summary["relocation_displacements_m"]:
            if displacement is None or max(abs(float(v)) for v in displacement) > bound + TOLERANCE_M:
                problems.append(
                    f"a relocation moved the centre {displacement!r} m, beyond the relocation "
                    f"radius plus the smoothing radius, {bound!r} m"
                )
    return problems


def separation(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Tracker-centre distance between two runs at the parent steps both sampled."""

    other = {row["parent_step"]: row["centre_m"] for row in second["tracker_events"]
             if row.get("centre_m") is not None}
    distances = [
        math.hypot(row["centre_m"][0] - other[row["parent_step"]][0],
                   row["centre_m"][1] - other[row["parent_step"]][1])
        for row in first["tracker_events"]
        if row.get("centre_m") is not None and row["parent_step"] in other
    ]
    if not distances:
        return {"common_samples": 0}
    return {
        "common_samples": len(distances),
        "min_m": min(distances),
        "median_m": statistics.median(distances),
        "max_m": max(distances),
        "last_m": distances[-1],
    }


def only_block(path: Path) -> dict[str, Any]:
    blocks = storm_following_blocks(json.loads(path.read_text(encoding="utf-8")))
    if len(blocks) != 1:
        raise SystemExit(
            f"{path}: {len(blocks)} storm_following records ({SCHEMA}); a finished following-nest "
            "receipt has exactly one (a run that stopped by name writes no receipt)"
        )
    return blocks[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--max-relocations", type=int, default=1)
    parser.add_argument("--compare", type=Path)
    arguments = parser.parse_args(argv)
    block = only_block(arguments.receipt)
    summary = summarize(block)
    problems = failures(block, summary, max_relocations=arguments.max_relocations)
    report: dict[str, Any] = {"receipt": str(arguments.receipt), **summary,
                              "max_axis_step_m": max((s["axis_distance_m"] for s in summary["centre_steps"]),
                                                     default=None),
                              "failures": problems}
    report.pop("centre_steps")
    if arguments.compare is not None:
        report["compare"] = {"receipt": str(arguments.compare),
                             **separation(block, only_block(arguments.compare))}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
