#!/usr/bin/env python3
"""Capture one published parent frame onto the child a ``[nest]`` table declares.

WHY THIS EXISTS
---------------
A ``[nest]`` table costs GPU hours to run and none to get wrong.  The window
can sit off the storm, the relaxation zone can cut through the updraft, and
the refinement ratio can price a child that does not fit on the card -- and
every one of those is visible from a parent frame that has already been
written, before anything is launched.  This tool reads a parent history frame
and the same document's ``[nest]`` table and writes the child's view of that
frame as raw ``f32`` planes, next to the parent's own view of the same region,
so the two can be rendered side by side.

WHAT IT IS NOT
--------------
It is not a forecast and it is not a nested run.  The child planes it writes
are the parent's state **interpolated** onto the child's mesh by
``nest_boundary.interpolate_parent_to_child``, which is the operator that
fills a real child's boundary rows.  Interpolation adds no information: a
child plane here has the parent's structure on a finer mesh, and nothing
finer.  What refinement buys is what the child's own dynamics do with that
state afterwards, and that is a run, not a capture.  The manifest this writes
says so in the ``claim_limit`` field, in those words, so a plane cannot be
lifted out of context and read as a refined forecast.

USAGE
-----
    python3 tools/nest_window_preview.py CONFIG.toml FRAME_DIR OUT_DIR
        [--carrier NAME[,NAME...]] [--level-index K]

``CONFIG.toml`` must declare a ``[nest]`` table; the geometry, the window and
the refinement ratio all come from it, so what is previewed is what would run.

Unsupported source tool: run it from a clone of the repository with gpuwm-cm1 installed
(pip install gpuwm-cm1). It uses the installed package when there is one and this
checkout's src/ otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The installed gpuwm-cm1 when there is one (a repository clone carries no built native
# tools); this checkout's src/ otherwise.
if importlib.util.find_spec("cm1core") is None:
    sys.path.insert(0, str(REPO / "src"))

import numpy  # noqa: E402

from cm1core.config import load_config  # noqa: E402
from cm1core.errors import CM1Refusal, refuse  # noqa: E402
from cm1core.nest_downscale import downscale_level  # noqa: E402
from cm1core.nest_geometry import NestGeometry  # noqa: E402

CLAIM_LIMIT = (
    "these child planes are one published parent state interpolated onto the "
    "child mesh, not a nested forecast. Interpolation adds no information: the "
    "child plane carries the parent's structure on a finer mesh and nothing "
    "finer. What refinement buys is what the child's own dynamics produce from "
    "this state, which requires running the child"
)

FIELD_TABLE = "history-field-table.tsv"
ARENA = "history-carriers.f32"


def read_field_table(frame: Path) -> dict[str, dict[str, object]]:
    """Name -> {shape, offset, stagger} for one published frame."""

    table: dict[str, dict[str, object]] = {}
    text = (frame / FIELD_TABLE).read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 8:
            raise ValueError(f"a field-table row has {len(parts)} columns, not 8")
        name, _meaning, units, stagger, shape_csv, _bounds, _path, offset = parts
        table[name] = {
            "units": units,
            "stagger": stagger,
            "shape": tuple(int(v) for v in shape_csv.split(",")),
            "offset": int(offset),
        }
    return table


def read_carrier(frame: Path, entry: dict[str, object]) -> numpy.ndarray:
    """One carrier out of the arena, by offset and shape, little-endian f32."""

    shape = tuple(entry["shape"])  # type: ignore[arg-type]
    count = 1
    for dim in shape:
        count *= dim
    array = numpy.fromfile(
        frame / ARENA, dtype="<f4", count=count, offset=int(entry["offset"])  # type: ignore[arg-type]
    )
    if array.size != count:
        raise ValueError(
            f"the arena holds {array.size} words where the field table says {count}"
        )
    return array.reshape(shape)


def mass_level(array: numpy.ndarray, stagger: str, level: int) -> numpy.ndarray:
    """One horizontal plane at mass-point extent, whatever the stagger.

    ``u`` and ``v`` carry one extra face along their own axis and ``w`` one
    extra level; a preview trims to the mass extent rather than inventing an
    interpolation, because the point here is where the window sits, not a
    staggered reconstruction.
    """

    if array.ndim == 4:  # mass-species
        array = array[0]
    if array.ndim == 2:
        plane = array
    else:
        plane = array[min(level, array.shape[0] - 1)]
    return plane


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("frame", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--carrier", default="w3d,th3d,q3d")
    parser.add_argument("--level-index", type=int, default=0)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if config.nest is None:
        refuse(
            code="CM1-NEST-PREVIEW-NO-NEST",
            gate="config-grammar",
            reason=(
                "this document declares no [nest] table, so there is no child "
                "window to capture; previewing a window that the document does "
                "not declare would show a geometry no run would ever use"
            ),
            remedy="add a [nest] table to the document, or preview a nested one",
            option="nest",
            value=None,
        )
    nest = config.nest
    geometry = NestGeometry(
        parent_nx=config.grid.nx,
        parent_ny=config.grid.ny,
        parent_dx_m=config.grid.dx_m,
        parent_dy_m=config.grid.dy_m,
        refinement_ratio=nest.refinement_ratio,
        time_refinement_ratio=nest.time_refinement_ratio,
        parent_i0=nest.parent_i0,
        parent_j0=nest.parent_j0,
        parent_span_x=nest.parent_span_x,
        parent_span_y=nest.parent_span_y,
        spec_zone=nest.spec_zone,
        relax_zone=nest.relax_zone,
        feedback=nest.feedback,
    )

    table = read_field_table(args.frame)
    args.out.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []
    for name in args.carrier.split(","):
        name = name.strip()
        if name not in table:
            raise SystemExit(
                f"{name!r} is not in this frame; it carries {sorted(table)}"
            )
        entry = table[name]
        plane = mass_level(read_carrier(args.frame, entry), str(entry["stagger"]), args.level_index)
        # The parent's own view of exactly the covered region, so the two
        # panels show the same ground rather than two different windows.
        parent_window = plane[
            geometry.parent_j0 : geometry.parent_j0 + geometry.parent_span_y,
            geometry.parent_i0 : geometry.parent_i0 + geometry.parent_span_x,
        ]
        # The capture needs the parent's halo around the window for its
        # five-point stencil, so it is taken from the full plane and not from
        # the trimmed one above.
        child = downscale_level(plane[:, : geometry.parent_nx], geometry)
        parent_path = args.out / f"parent-window.{name}.f32"
        child_path = args.out / f"child-capture.{name}.f32"
        numpy.ascontiguousarray(parent_window, dtype="<f4").tofile(parent_path)
        numpy.ascontiguousarray(child, dtype="<f4").tofile(child_path)
        written.append(
            {
                "carrier": name,
                "units": entry["units"],
                "stagger": entry["stagger"],
                "level_index": args.level_index,
                "parent_plane": os.fspath(parent_path),
                "parent_nx": int(parent_window.shape[1]),
                "parent_ny": int(parent_window.shape[0]),
                "parent_dx_m": geometry.parent_dx_m,
                "child_plane": os.fspath(child_path),
                "child_nx": geometry.nx,
                "child_ny": geometry.ny,
                "child_dx_m": geometry.dx_m,
                "parent_min": float(parent_window.min()),
                "parent_max": float(parent_window.max()),
                "child_min": float(child.min()),
                "child_max": float(child.max()),
            }
        )

    manifest = {
        "schema": "gm1.nest-window-preview/v1",
        "claim_limit": CLAIM_LIMIT,
        "config": os.fspath(args.config),
        "config_sha256": config.sha256,
        "frame": os.fspath(args.frame),
        "window_origin_m": [geometry.origin_x_m, geometry.origin_y_m],
        "window_extent_m": [geometry.extent_x_m, geometry.extent_y_m],
        "refinement_ratio": geometry.refinement_ratio,
        "spec_zone": geometry.spec_zone,
        "relax_zone": geometry.relax_zone,
        "planes": written,
    }
    (args.out / "preview-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CM1Refusal as refusal:
        print(f"nest-window-preview: {refusal}", file=sys.stderr)
        raise SystemExit(2)
