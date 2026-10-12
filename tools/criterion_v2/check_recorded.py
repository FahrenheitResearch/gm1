"""OFFLINE, CPU-ONLY: decide the recorded verdicts again from their published per-sample files.

    python3 tools/criterion_v2/check_recorded.py [--only LABEL ...] [--results DIR] [--json OUT.json]
    python3 tools/criterion_v2/check_recorded.py --rows SCORED_DIR --against LABEL

NOT PART OF THE FROZEN CRITERION.  ``v2rule.py`` and ``score.py`` are imported unchanged, and this tool stops
when either differs from ``FROZEN.json``.

For every recorded source under ``tools/criterion_v2/results/`` (a folder holding ``verdict.json``,
``samples.tsv`` and ``candidates.tsv``, plain or gzipped) it

1. rebuilds each sample and its candidate vortices from the two tables,
2. runs the frozen time-series rule on them (``v2rule.decide``: tracking, bridging, duration, the grid and
   cadence gates, the outcome and its caption), and
3. compares the result with what ``verdict.json`` and ``samples.tsv`` record: the outcome, caption, reason,
   genesis time, every episode (start, end, members, peak winds, RMW range), the longest runs, the class,
   condition, marginal and flag counts, and each sample's run lengths, centre move and continuity.

What this checks: given the per-sample measurements, the frozen rule gives the recorded verdict and numbers.
What it cannot check: the per-sample measurements themselves (centre, RMW, winds, class).  Those come from
the model's wind fields, which ``score.py`` reads from a run's own output; the tables are that reading's
record.  The tables store four decimals, so times and distances are compared to 0.001.

The frame velocity of each source (the storm-following frame of the identity links) is read from the
collected verdict file beside the source folders (``tornado-verdicts-*.json``, ``frame_velocity_m_s``).
The exit status is 0 only when every source compared equal.

``--rows SCORED_DIR --against LABEL`` is the other half, for someone who holds some of a recorded source's
model output (an excerpt cut by ``window_excerpt.py``): ``SCORED_DIR`` is what ``score.py`` wrote for it, and
every per-sample measurement and candidate row in it is compared, as text, with the recorded source's row of
the same record number.  The columns that depend on the other samples of a record are left out.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score  # noqa: E402  (the frozen scorer: only its FROZEN.json self-check is used)
import v2rule  # noqa: E402

TOL = 1.0e-3
FLOAT_KEYS = ("genesis_model_time_s", "longest_qualifying_run_s", "longest_qualifying_or_unresolved_run_s",
              "longest_qualifying_run_plausible_links_s", "record_start_s", "record_end_s", "median_interval_s",
              "max_interval_s")
EXACT_KEYS = ("outcome", "reason", "samples", "class_counts", "condition_counts", "marginal_counts",
              "flag_counts")
EPISODE_FLOATS = ("start_s", "end_s", "span_s", "vmax_peak_m_s", "ground_wind_peak_m_s")
ROW_FLOATS = ("q_run_s", "qu_run_s", "centre_move_m")


def read_table(folder: Path, name: str) -> list[dict[str, str]]:
    plain, packed = folder / name, folder / (name + ".gz")
    if plain.is_file():
        text = plain.read_text(encoding="utf-8")
    elif packed.is_file():
        text = gzip.decompress(packed.read_bytes()).decode("utf-8")
    else:
        raise FileNotFoundError(f"{folder}: no {name} or {name}.gz")
    lines = text.splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:] if line]


def num(text: str | None) -> float:
    try:
        return float(text) if text not in (None, "", "nan") else math.nan
    except ValueError:
        return math.nan


def rebuild(samples: list[dict[str, str]], candidates: list[dict[str, str]], velocity: tuple[float, float]
            ) -> list[dict[str, Any]]:
    """The sample dictionaries ``v2rule.decide`` takes, from the two recorded tables."""

    per: dict[int, list[dict[str, Any]]] = {int(row["sample"]): [] for row in samples}
    for row in candidates:
        per[int(row["sample"])].append({
            "class": row["class"], "conditions": row["conditions"],
            "centre_x_m": num(row["centre_x_m"]), "centre_y_m": num(row["centre_y_m"]),
            "centre_ground_x_m": num(row["centre_ground_x_m"]), "centre_ground_y_m": num(row["centre_ground_y_m"]),
            "vmax_m_s": num(row["vmax_m_s"]), "rmw_m": num(row["rmw_m"]),
            "ground_wind_max_m_s": num(row["ground_wind_max_m_s"]),
        })
    out = []
    for row in samples:
        found = row["found"] == "1"
        sample: dict[str, Any] = {
            "model_time_s": num(row["model_time_s"]), "found": found, "class": row["class"],
            "conditions": row["conditions"], "flags": [flag for flag in row["flags"].split(",") if flag],
            "umove": velocity[0], "vmove": velocity[1], "candidates": per[int(row["sample"])],
        }
        if found:
            for key in ("centre_x_m", "centre_y_m", "centre_ground_x_m", "centre_ground_y_m"):
                sample[key] = num(row[key])
        out.append(sample)
    return out


def close(a: Any, b: Any) -> bool:
    # verdict.json writes a value that is not a number (one sample has no interval) as null
    a = math.nan if a is None else float(a)
    b = math.nan if b is None else float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= TOL


def compare(recorded: dict[str, Any], again: dict[str, Any], rows: list[dict[str, str]],
            samples: list[dict[str, Any]]) -> list[str]:
    """What differs between the recorded verdict and the one decided again (empty when nothing does)."""

    differ = [key for key in EXACT_KEYS if recorded.get(key) != again.get(key)]
    differ += [key for key in FLOAT_KEYS if not close(recorded.get(key), again.get(key))]
    # The caption names the output interval.  The tables' four-decimal times round it (0.09375 s reads as
    # 0.0938 s), so the caption is checked as the rule's wording for this outcome, grid and the recorded
    # interval, and the interval itself is compared above.
    interval = recorded.get("median_interval_s")
    wording = v2rule.CAPTIONS[again["outcome"]].format(
        grid=f"{max(float(recorded['grid']['dx_m']), float(recorded['grid']['dy_m'])):g} m",
        cadence=f"{math.nan if interval is None else float(interval):g} s")
    if recorded.get("caption") != wording:
        differ.append("caption")
    was, now = recorded.get("episodes", []), again.get("episodes", [])
    if len(was) != len(now):
        differ.append(f"episodes ({len(was)} recorded, {len(now)} decided again)")
    else:
        for index, (a, b) in enumerate(zip(was, now)):
            same = (a["samples"] == b["samples"] and all(close(a[key], b[key]) for key in EPISODE_FLOATS)
                    and all(close(x, y) for x, y in zip(a["rmw_range_m"], b["rmw_range_m"]))
                    and all(close(x, y) for x, y in zip(a["centre_ground_start_m"], b["centre_ground_start_m"])))
            if not same:
                differ.append(f"episode {index}")
                break
    for row, sample in zip(rows, samples):
        bad = [key for key in ROW_FLOATS if not close(num(row[key]), sample[key])]
        if (row["continuous"] == "1") != bool(sample["continuous"]):
            bad.append("continuous")
        if bad:
            differ.append(f"sample {row['sample']} at {row['model_time_s']} s: {', '.join(bad)}")
            break
    return differ


#: Runs whose nest passed its time step limit before the last record: label prefix -> the model time (s).
#: A peak wind from a source that reaches past that time is printed with a note.
TIME_STEP_LIMIT_PASSED_S = {"E100R2-PF-4m": 3312.0}


def past_the_limit(label: str, last_model_time_s: float) -> float | None:
    """The time step limit this source reaches past, or ``None``."""

    for prefix, limit_s in TIME_STEP_LIMIT_PASSED_S.items():
        if label.startswith(prefix) and last_model_time_s >= limit_s:
            return limit_s
    return None


def velocities(results: Path) -> dict[str, tuple[float, float]]:
    """label -> (umove, vmove) from every collected verdict file under ``results``."""

    out: dict[str, tuple[float, float]] = {}
    for path in sorted(results.glob("*/tornado-verdicts-*.json")):
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"].values():
            for source in run["sources"]:
                velocity = source.get("frame_velocity_m_s")
                if velocity:
                    out[source["label"]] = (float(velocity[0]), float(velocity[1]))
    return out


def check_source(folder: Path, velocity: tuple[float, float]) -> dict[str, Any]:
    started = time.time()
    recorded = json.loads((folder / "verdict.json").read_text(encoding="utf-8"))
    rows = read_table(folder, "samples.tsv")
    samples = rebuild(rows, read_table(folder, "candidates.tsv"), velocity)
    grid = v2rule.GridFacts(dx_m=float(recorded["grid"]["dx_m"]), dy_m=float(recorded["grid"]["dy_m"]),
                            lowest_scalar_level_m=float(recorded["grid"]["lowest_scalar_level_m"]),
                            notes=list(recorded["grid"].get("notes", [])))
    again = v2rule.decide(samples, grid)
    differ = compare(recorded, again, rows, samples)
    if recorded.get("criterion_v2_sha256") != v2rule.CRITERION_V2_SHA256:
        differ.append("criterion_v2_sha256")
    if recorded.get("files_sha256") != score.frozen_check()["files_sha256"]:
        differ.append("files_sha256 (the verdict was made with other rule or scorer files)")
    peak = max((episode["ground_wind_peak_m_s"] for episode in again["episodes"]), default=None)
    return {"label": folder.name, "samples": len(rows), "recorded": recorded["outcome"], "again": again["outcome"],
            "genesis_model_time_s": again["genesis_model_time_s"],
            "longest_tracked_qualifying_s": again["longest_qualifying_run_s"],
            "episode_peak_ground_wind_m_s": peak,
            "reaches_past_time_step_limit_s": past_the_limit(
                folder.name, max((float(row["model_time_s"]) for row in rows if row.get("model_time_s")), default=0.0)),
            "differs": differ, "same": not differ,
            "seconds": round(time.time() - started, 1)}


#: The columns of samples.tsv that depend on the other samples of the record (set by ``v2rule.decide``), and its
#: running number; every other column is measured from one record alone.
SERIES_COLUMNS = ("sample", "centre_move_m", "continuous", "q_run_s", "qu_run_s")


def compare_rows(scored: Path, recorded: Path) -> dict[str, Any]:
    """Per-sample measurements of a scored folder against the recorded source's rows with the same ``index``."""

    mine, theirs = read_table(scored, "samples.tsv"), read_table(recorded, "samples.tsv")
    by_index = {row["index"]: row for row in theirs}
    columns = [column for column in mine[0] if column not in SERIES_COLUMNS]
    absent = [row["index"] for row in mine if row["index"] not in by_index]
    unequal = [row["index"] for row in mine if row["index"] in by_index
               and any(row[column] != by_index[row["index"]][column] for column in columns)]

    def candidates(folder: Path, numbers: dict[str, str]) -> dict[str, list[tuple[str, ...]]]:
        out: dict[str, list[tuple[str, ...]]] = {}
        for row in read_table(folder, "candidates.tsv"):
            if row["sample"] in numbers:
                out.setdefault(numbers[row["sample"]], []).append(
                    tuple(value for key, value in row.items() if key != "sample"))
        return out

    wanted = {row["index"] for row in mine}
    mine_c = candidates(scored, {row["sample"]: row["index"] for row in mine})
    theirs_c = candidates(recorded, {row["sample"]: row["index"] for row in theirs if row["index"] in wanted})
    unequal_c = sorted(index for index in wanted if mine_c.get(index, []) != theirs_c.get(index, []))
    return {"samples": len(mine), "columns_compared": columns, "absent_from_recorded": absent,
            "samples_unequal": unequal, "candidate_rows": sum(len(rows) for rows in mine_c.values()),
            "samples_with_unequal_candidates": unequal_c,
            "same": not absent and not unequal and not unequal_c}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--only", nargs="*", help="source labels (folder names); default: every recorded source")
    parser.add_argument("--json", type=Path, help="write the full record here")
    parser.add_argument("--rows", type=Path, help="a folder score.py wrote for an excerpt of a recorded source: "
                        "compare its per-sample measurements with the recorded rows (needs --against)")
    parser.add_argument("--against", help="with --rows: the recorded source's label")
    arguments = parser.parse_args(argv)
    frozen = score.frozen_check()
    if not frozen["matches_frozen"]:
        print("refused: v2rule.py or score.py differ from FROZEN.json, so this would not be criterion v2",
              file=sys.stderr)
        return 2
    if arguments.rows or arguments.against:
        if not (arguments.rows and arguments.against):
            parser.error("--rows and --against go together")
        found = sorted(arguments.results.glob(f"*/{arguments.against}/verdict.json"))
        if len(found) != 1:
            print(f"refused: no recorded source named {arguments.against}", file=sys.stderr)
            return 2
        record = compare_rows(arguments.rows, found[0].parent)
        print(f"{record['samples']} samples and {record['candidate_rows']} candidate rows of {arguments.rows} "
              f"against {arguments.against}, by record number")
        print(f"samples absent from the recorded source: {len(record['absent_from_recorded'])}; with a different "
              f"measurement: {len(record['samples_unequal'])}; with different candidates: "
              f"{len(record['samples_with_unequal_candidates'])}")
        print("every measurement equals the recorded one" if record["same"] else "DIFFERENT")
        if arguments.json:
            arguments.json.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        return 0 if record["same"] else 1
    velocity = velocities(arguments.results)
    folders = sorted(path.parent for path in arguments.results.glob("*/*/verdict.json"))
    if arguments.only:
        missing = sorted(set(arguments.only) - {folder.name for folder in folders})
        if missing:
            print(f"refused: no recorded source named {', '.join(missing)}", file=sys.stderr)
            return 2
        folders = [folder for folder in folders if folder.name in arguments.only]
    print(f"criterion v2 {v2rule.CRITERION_V2_SHA256[:12]}..., rule and scorer files equal FROZEN.json")
    print(f"{'source':<32} {'samples':>7}  {'recorded':<21} {'decided again':<21} result")
    records = []
    for folder in folders:
        if folder.name not in velocity:
            print(f"{folder.name:<32} no frame velocity in the collected verdict file: not checked")
            records.append({"label": folder.name, "same": False, "differs": ["no frame velocity"]})
            continue
        record = check_source(folder, velocity[folder.name])
        records.append(record)
        result = "same" if record["same"] else "DIFFERENT: " + "; ".join(record["differs"])
        print(f"{record['label']:<32} {record['samples']:>7}  {record['recorded']:<21} {record['again']:<21} {result}")
        if record["again"] == "TORNADO":
            print(f"{'':<32} genesis {record['genesis_model_time_s']:.4f} s, tracked "
                  f"{record['longest_tracked_qualifying_s']:.3f} s, peak 10 m ground-relative wind "
                  f"{record['episode_peak_ground_wind_m_s']:.1f} m/s")
            if record["reaches_past_time_step_limit_s"] is not None:
                print(f"{'':<32} note: this source has records after "
                      f"{record['reaches_past_time_step_limit_s']:g} s, when the nest's time step was too long "
                      f"for its winds; do not quote this wind")
    passed = bool(records) and all(record["same"] for record in records)
    print(f"{sum(1 for record in records if record['same'])} of {len(records)} recorded sources decided again "
          f"with the same result")
    if arguments.json:
        arguments.json.write_text(json.dumps({"schema": "gm1.criterion-v2-check-recorded/v1",
                                              "criterion_v2_sha256": v2rule.CRITERION_V2_SHA256,
                                              "files_sha256": frozen["files_sha256"], "sources": records,
                                              "passed": passed}, indent=1) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
