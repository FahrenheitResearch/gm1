"""OFFLINE, CPU-ONLY: turn published window records (CM1-style netCDF) into a stream the criterion v2 scorer reads.

    python3 tools/criterion_v2/netcdf_to_window.py NCDIR --out DIR [--before S] [--from S]
        [--child-nx 500 --child-ny 500 --spec-zone 1 --relax-zone 4]

NOT PART OF THE FROZEN CRITERION.  Nothing here scores anything.  Needs the ``netCDF4`` Python package.

The 4 m tornado run E100R2-PF-4m is published as netCDF files, one window record per file, in the Hugging
Face datasets ``deepguess/GM1-E100R2-PF-4m-last-5min-every-1s``, ``...-last-2min-every-record`` and
``...-top10-frames``.  ``score.py`` does not read netCDF: it reads a ``nest-child-window`` stream.  This
tool writes such a stream from a folder of those files, holding what the scorer reads and nothing else:
the winds ``u`` and ``v`` on the scalar levels from the ground up to the upper one of the two that bracket
10 m.  The result has the layout ``window_excerpt.py`` writes:

    DIR/window-stream.json    the receipt: grid spacing, frame velocity, level heights, field layout
    DIR/window-records.tsv    one row per file: the run's record number, model time, window placement and
                              origin, and the SHA-256 the file records for the raw record it was made from
    DIR/window-stream.f32     the winds, on the scorer's padded layout (the halo cells are zero and are
                              never read)

``score.py DIR --out OUT --nest-start-s 2160 --parent-dx 100`` then scores it as a source of its own, and
``check_recorded.py --rows OUT --against LABEL`` compares every per-sample measurement with the recorded
row of the same record number.

What each file gives: ``u`` and ``v`` (the window's advecting winds, relative to the moving grid), ``zh``
(scalar level heights, km), ``xf``/``yf`` (4 m spacing), ``time``, ``umove``/``vmove``, and the global
attributes ``gm1_record_index``, ``gm1_window_i0``, ``gm1_window_j0``, ``gm1_window_origin_x_m``,
``gm1_window_origin_y_m`` and ``gm1_record_sha256``.  The level heights are stored in km as 32-bit numbers
and are rounded here to the millimetre (2, 6, 10 m ...), which is what the run's levels are.

What a file does not give is the nest's own size and edge zones, which the scorer uses to keep candidates
out of the nest's boundary zone.  They are arguments, with E100R2-PF-4m's values as the defaults: a
500 x 500 cell nest with a 1-cell specified zone and a 4-cell relaxation zone.

``--before S`` keeps the files with model time below S, ``--from S`` those at or after S.  For this run
3312.0 s is where the Courant number first passed 1: the headline verdict uses only records before it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score  # noqa: E402  (the frozen scorer: its level bracket and halo width are used)
from window_excerpt import COLUMNS  # noqa: E402

SCHEMA = "gm1.nest-child-window-excerpt/v1"
NG = score.NG


def read_facts(path: Path, netcdf: Any) -> dict[str, Any]:
    with netcdf.Dataset(path) as data:
        zh = [round(float(z) * 1000.0, 3) for z in data.variables["zh"][:]]
        xf = np.asarray(data.variables["xf"][:], dtype=np.float64) * 1000.0
        yf = np.asarray(data.variables["yf"][:], dtype=np.float64) * 1000.0
        return {
            "path": path, "model_time_s": float(data.variables["time"][0]),
            "nx": len(data.dimensions["xh"]), "ny": len(data.dimensions["yh"]), "nz": len(data.dimensions["zh"]),
            "zh": zh, "dx": round(float(np.median(np.diff(xf))), 3), "dy": round(float(np.median(np.diff(yf))), 3),
            "umove": float(data.variables["umove"][0]), "vmove": float(data.variables["vmove"][0]),
            "index": int(data.getncattr("gm1_record_index")),
            "i0": int(data.getncattr("gm1_window_i0")), "j0": int(data.getncattr("gm1_window_j0")),
            "origin_x_m": float(data.getncattr("gm1_window_origin_x_m")),
            "origin_y_m": float(data.getncattr("gm1_window_origin_y_m")),
            "source_sha256": str(data.getncattr("gm1_record_sha256")),
            "run": str(data.getncattr("gm1_run")).split(":")[0] if "gm1_run" in data.ncattrs() else "",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ncdir", type=Path, help="a folder of the dataset's .nc files")
    parser.add_argument("--out", type=Path, required=True, help="the stream folder to write (must not exist)")
    parser.add_argument("--before", type=float, help="keep files with model time below this, s")
    parser.add_argument("--from", dest="start", type=float, help="keep files with model time at or after this, s")
    parser.add_argument("--child-nx", type=int, default=500)
    parser.add_argument("--child-ny", type=int, default=500)
    parser.add_argument("--spec-zone", type=int, default=1)
    parser.add_argument("--relax-zone", type=int, default=4)
    arguments = parser.parse_args(argv)
    try:
        import netCDF4
    except ImportError:
        print("this tool needs the netCDF4 package: python3 -m pip install netCDF4", file=sys.stderr)
        return 2
    paths = sorted(arguments.ncdir.glob("*.nc"))
    if not paths:
        print(f"{arguments.ncdir}: no .nc file", file=sys.stderr)
        return 2
    facts = [read_facts(path, netCDF4) for path in paths]
    facts = [f for f in facts if (arguments.before is None or f["model_time_s"] < arguments.before)
             and (arguments.start is None or f["model_time_s"] >= arguments.start)]
    facts.sort(key=lambda f: f["model_time_s"])
    if not facts:
        print("no file in the requested interval", file=sys.stderr)
        return 2
    first = facts[0]
    same = ("nx", "ny", "nz", "zh", "dx", "dy", "umove", "vmove", "run")
    for f in facts:
        if any(f[key] != first[key] for key in same):
            print(f"{f['path'].name}: its grid, levels, frame velocity or run differ from {first['path'].name}",
                  file=sys.stderr)
            return 2
    times = [f["model_time_s"] for f in facts]
    if any(b <= a for a, b in zip(times, times[1:])):
        print("two files hold the same model time", file=sys.stderr)
        return 2
    nx, ny = first["nx"], first["ny"]
    zh = np.asarray(first["zh"], dtype=np.float64)
    _, upper, _, _, note = score.bracket(zh, float(score.CRITERION_V2["height_agl_m"]))
    if note:
        print(f"{note}: the scorer could not judge these files", file=sys.stderr)
        return 2
    levels = upper + 1
    shapes = {"uadv": (levels + 2, ny + 2 * NG, nx + 1 + 2 * NG), "vadv": (levels + 2, ny + 1 + 2 * NG, nx + 2 * NG)}
    fields, offset = [], 0
    for name, shape in shapes.items():
        fields.append({"name": name, "shape_zyx": list(shape), "offset_words": offset, "words": int(np.prod(shape))})
        offset += int(np.prod(shape))
    arguments.out.mkdir(parents=True, exist_ok=False)
    lines = ["\t".join(COLUMNS)]
    written = 0
    with open(arguments.out / "window-stream.f32", "wb") as arena:
        for f in facts:
            with netCDF4.Dataset(f["path"]) as data:
                data.set_auto_mask(False)
                u = np.asarray(data.variables["u"][0, :levels], dtype="<f4")
                v = np.asarray(data.variables["v"][0, :levels], dtype="<f4")
            ua = np.zeros(shapes["uadv"], dtype="<f4")
            va = np.zeros(shapes["vadv"], dtype="<f4")
            ua[1:levels + 1, NG:NG + ny, NG:NG + nx + 1] = u
            va[1:levels + 1, NG:NG + ny + 1, NG:NG + nx] = v
            payload = ua.tobytes() + va.tobytes()
            arena.write(payload)
            row = {"index": f["index"], "model_time_s": repr(f["model_time_s"]), "offset_bytes": written,
                   "bytes": len(payload), "i0": f["i0"], "j0": f["j0"], "nx": nx, "ny": ny, "nz": levels,
                   "origin_x_m": repr(f["origin_x_m"]), "origin_y_m": repr(f["origin_y_m"]),
                   "sha256": hashlib.sha256(payload).hexdigest(), "source_sha256": f["source_sha256"]}
            lines.append("\t".join(str(row[column]) for column in COLUMNS))
            written += len(payload)
    (arguments.out / "window-records.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema": SCHEMA,
        "child": {"dx_m": first["dx"], "dy_m": first["dy"], "nx": arguments.child_nx, "ny": arguments.child_ny,
                  "spec_zone": arguments.spec_zone, "relax_zone": arguments.relax_zone,
                  "u_move_m_s": first["umove"], "v_move_m_s": first["vmove"]},
        "window": {"nx": nx, "ny": ny, "nz": levels},
        # index 0 is CM1's mirrored ghost level below the ground, as in a stream's own receipt
        "zh_cm1_indexed": [-float(zh[0])] + [float(z) for z in zh[:levels + 1]],
        "fields": fields,
        "record_words": offset,
        "record_bytes": 4 * offset,
        "excerpt": {"made_from": "netCDF window records", "run": first["run"], "files": len(facts),
                    "source_window_nz": first["nz"], "scalar_levels_kept": levels, "fields_kept": ["u", "v"],
                    "first_model_time_s": times[0], "last_model_time_s": times[-1],
                    "first_index": facts[0]["index"], "last_index": facts[-1]["index"],
                    "before_s": arguments.before, "from_s": arguments.start, "bytes": written,
                    "child_size_and_zones": "from the command line (the files do not hold them)"},
    }
    (arguments.out / "window-stream.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(receipt["excerpt"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
