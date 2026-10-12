"""OFFLINE: the nest tornado criterion's sidecar, replayed or recomputed from child frames.

    python3 tools/tornado_criterion.py replay <history>/nest-child/tornado-criterion.jsonl
    python3 tools/tornado_criterion.py frames <history>/nest-child [--out report.json]

``replay`` re-decides every condition, episode and the verdict from the
per-sample rows alone (:func:`cm1core.nest_tornado_criterion.replay_time_series`),
so a run that died without a receipt still has its verdict, and a receipt's
verdict can be checked against its own samples.  A resumed run's sidecar
header carries the ledger rows the child restart handed it
(``carried_from_restart``); the replay seeds its ledger from them, so the
verdict it re-decides is the whole run's, as the receipt's is.  ``frames`` recomputes the
sample of every published child frame with the host reference the device
kernel is held to (:func:`recompute_from_frames`) and compares it with the row
of the same parent step (the placement's child step divided by the header's
time_refinement_ratio); frames published after a nest move, and frames with no
row, are counted, not compared.

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

from cm1core.nest_tornado_criterion import recompute_from_frames, replay_time_series  # noqa: E402

SUMMARY_KEYS = {
    "replay": ("kind", "path", "sha256", "verdict", "genesis_model_time_s",
               "tornado_declared_model_time_s", "samples", "first_sample_model_time_s",
               "last_sample_model_time_s", "condition_counts", "longest_qualifying_run_s",
               "episode_count", "stop_request", "stop_requests_withdrawn",
               "last_withdrawn_stop_request", "carried_from_restart", "rows_identical"),
    "frames": ("kind", "child_root", "criterion_sha256", "frames_read", "frames_compared",
               "frames_identical_to_jsonl", "frames_after_a_move", "frames_without_a_row"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="re-decide the verdict from the sidecar rows")
    replay.add_argument("jsonl", type=Path)
    replay.add_argument("--out", type=Path, default=None, help="full report (must not exist)")
    frames = commands.add_parser("frames", help="recompute samples from published child frames")
    frames.add_argument("child_root", type=Path)
    frames.add_argument("--out", type=Path, default=None, help="full report (must not exist)")
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            report = replay_time_series(args.jsonl)
        else:
            report = recompute_from_frames(args.child_root)
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "refused", "reason": str(error)}), file=sys.stderr)
        return 2
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps({key: report[key] for key in SUMMARY_KEYS[args.command]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
