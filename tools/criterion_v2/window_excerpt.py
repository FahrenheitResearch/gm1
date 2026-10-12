"""OFFLINE, CPU-ONLY: cut a small excerpt of a window stream that the criterion v2 scorer can still read.

    python3 tools/criterion_v2/window_excerpt.py STREAM [STREAM ...] --out DIR
        [--start S] [--end S] [--every N] [--last-at S]
    python3 tools/criterion_v2/window_excerpt.py --spec SPEC.json [--root KEY=DIR ...] --out DIR
        [--start S] [--end S] [--every N] [--last-at S]

NOT PART OF THE FROZEN CRITERION.  Nothing here scores anything.

A ``nest-child-window`` stream holds fifteen fields on every window level, which is far more than the
scorer reads: ``score.py`` takes the advecting winds ``uadv`` and ``vadv`` on the two scalar levels that
bracket 10 m above ground.  This tool copies, for the records chosen, only those two fields on the levels
from the ground up to the upper one of that pair (three levels when the lowest levels are 2, 6 and 10 m),
byte for byte, and writes a stream folder with its own receipt and index:

    DIR/window-stream.json    the source's receipt with the window's level count, level heights and field
                              layout cut down, and an ``excerpt`` block saying what was kept
    DIR/window-records.tsv    one row per kept record: the source's record number (``index``), model time,
                              window placement and origin, the SHA-256 of the kept bytes, and the SHA-256
                              the source index gives for the whole record (``source_sha256``)
    DIR/window-stream.f32     the kept bytes

``score.py DIR --out OUT --nest-start-s S --parent-dx M`` then scores the excerpt as a source of its own.
Because the scorer reads the same bytes at the same placement, each sample's measurements (centre, RMW,
winds, class) equal those of the full stream's record with the same ``index``; the time-series columns
and the verdict are those of the excerpt's own records (``check_recorded.py --rows`` compares the two).

The records: every record of the source streams (given in time order) whose model time lies in
``[--start, --end]``, thinned to every Nth.  With ``--last-at S`` the thinning is counted back from the
record at model time S, so that record is kept.

A stream folder given by position needs its receipt (``window-stream.json``).  ``--spec`` takes a
``window_segments.py`` spec instead (a stream without a receipt, or one kept in several segments); the
receipt is then the one that adapter derives from the deck.
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

import score  # noqa: E402  (the frozen scorer: its level bracket and window halo width are used)
import window_segments  # noqa: E402

SCHEMA = "gm1.nest-child-window-excerpt/v1"
COLUMNS = ("index", "model_time_s", "offset_bytes", "bytes", "i0", "j0", "nx", "ny", "nz", "origin_x_m",
           "origin_y_m", "sha256", "source_sha256")


def sources_from_streams(streams: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = None
    rows = []
    for stream in streams:
        this = json.loads((stream / "window-stream.json").read_text(encoding="utf-8"))
        if receipt is None:
            receipt = this
        elif (this["child"], this["window"], this["fields"]) != (receipt["child"], receipt["window"],
                                                              receipt["fields"]):
            raise SystemExit(f"{stream}: its receipt differs from the first stream's")
        _, index = window_segments.index_rows(stream)
        rows += [dict(row, arena=stream / "window-stream.f32") for row in index]
    return receipt, rows


def sources_from_spec(spec_path: Path, roots: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = window_segments.load_spec(spec_path, roots)
    receipt, segments = window_segments.trajectory(spec)
    rows = []
    for segment in segments:
        rows += [dict(row, index=str(int(row["index"]) + segment["index_offset"]), arena=segment["arena"])
                 for row in segment["rows"]]
    return receipt, rows


def choose(rows: list[dict[str, Any]], start: float | None, end: float | None, every: int,
           last_at: float | None) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: float(row["model_time_s"]))
    times = [float(row["model_time_s"]) for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise SystemExit("model times do not increase strictly across the streams")
    anchor = 0
    if last_at is not None:
        near = [k for k, t in enumerate(times) if abs(t - last_at) < 1.0e-6]
        if not near:
            raise SystemExit(f"no record at model time {last_at} s")
        anchor = near[0] % every
        end = last_at if end is None else min(end, last_at)
    kept = [row for k, row in enumerate(rows) if k % every == anchor
            and (start is None or times[k] >= start - 1.0e-9) and (end is None or times[k] <= end + 1.0e-9)]
    if not kept:
        raise SystemExit("no record in the requested interval")
    return kept


def write_excerpt(receipt: dict[str, Any], rows: list[dict[str, Any]], out: Path, about: dict[str, Any]
                  ) -> dict[str, Any]:
    nz = int(receipt["window"]["nz"])
    zh = np.asarray(receipt["zh_cm1_indexed"], dtype=np.float64)[1:nz + 1]
    _, upper, _, _, note = score.bracket(zh, float(score.CRITERION_V2["height_agl_m"]))
    if note:
        raise SystemExit(f"{note}: the scorer could not judge this stream, so no excerpt is cut")
    levels = upper + 1                      # scalar levels kept, from the ground up
    layout = {field["name"]: field for field in receipt["fields"]}
    fields, offset = [], 0
    for name in ("uadv", "vadv"):
        full = [int(v) for v in layout[name]["shape_zyx"]]
        shape = [levels + 2, full[1], full[2]]
        fields.append({"name": name, "shape_zyx": shape, "offset_words": offset, "words": int(np.prod(shape))})
        offset += int(np.prod(shape))
    out.mkdir(parents=True, exist_ok=False)
    lines = ["\t".join(COLUMNS)]
    written = 0
    with open(out / "window-stream.f32", "wb") as arena:
        for row in rows:
            if (int(row["nx"]), int(row["ny"]), int(row["nz"])) != tuple(int(receipt["window"][k])
                                                                         for k in ("nx", "ny", "nz")):
                raise SystemExit(f"record {row['index']}: its size differs from the receipt's window")
            source = np.memmap(row["arena"], dtype="<f4", mode="r")
            base = int(row["offset_bytes"]) // 4
            digest = hashlib.sha256()
            for field in fields:
                full = [int(v) for v in layout[field["name"]]["shape_zyx"]]
                first = base + int(layout[field["name"]]["offset_words"])
                block = np.ascontiguousarray(source[first:first + int(np.prod(full))].reshape(full)[:levels + 2])
                payload = block.tobytes()
                digest.update(payload)
                arena.write(payload)
            del source
            new = dict(row, offset_bytes=str(written), bytes=str(4 * offset), nz=str(levels),
                       sha256=digest.hexdigest(), source_sha256=row.get("sha256", ""))
            lines.append("\t".join(str(new[column]) for column in COLUMNS))
            written += 4 * offset
    (out / "window-records.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    times = [float(row["model_time_s"]) for row in rows]
    cut = {key: value for key, value in receipt.items() if key not in ("fields", "adapter", "zf_cm1_indexed")}
    cut.update({
        "schema": SCHEMA,
        "window": {**receipt["window"], "nz": levels},
        "zh_cm1_indexed": [float(v) for v in receipt["zh_cm1_indexed"][:levels + 2]],
        "fields": fields,
        "record_words": offset,
        "record_bytes": 4 * offset,
        "excerpt": {**about, "source_schema": receipt.get("schema"), "source_window_nz": nz,
                    "scalar_levels_kept": levels, "fields_kept": ["uadv", "vadv"], "records": len(rows),
                    "first_model_time_s": times[0], "last_model_time_s": times[-1],
                    "first_index": int(rows[0]["index"]), "last_index": int(rows[-1]["index"]),
                    "bytes": written},
    })
    (out / "window-stream.json").write_text(json.dumps(cut, indent=1) + "\n", encoding="utf-8")
    return cut["excerpt"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("streams", nargs="*", type=Path, help="window stream folders with a receipt, in time order")
    parser.add_argument("--spec", type=Path, help="a window_segments.py spec, in place of stream folders")
    parser.add_argument("--root", action="append", default=[], help="KEY=DIR, for the spec's roots")
    parser.add_argument("--out", type=Path, required=True, help="the excerpt's folder (must not exist)")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--last-at", type=float, help="count the thinning back from the record at this model time")
    arguments = parser.parse_args(argv)
    if bool(arguments.streams) == bool(arguments.spec):
        parser.error("give stream folders or --spec, not both and not neither")
    if arguments.every < 1:
        parser.error("--every must be 1 or more")
    if arguments.spec:
        receipt, rows = sources_from_spec(arguments.spec, arguments.root)
    else:
        receipt, rows = sources_from_streams(arguments.streams)
    kept = choose(rows, arguments.start, arguments.end, arguments.every, arguments.last_at)
    about = {"every": arguments.every, "start_s": arguments.start, "end_s": arguments.end,
             "last_at_s": arguments.last_at, "source_records": len(rows)}
    print(json.dumps(write_excerpt(receipt, kept, arguments.out, about), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
