# Independent verification of the E100R2-PF-4m criterion v2 scoring

These files are the independent numpy check that is summarised in `docs/GM1-TORNADO-VERDICTS-V2.md`, section "Independent verification (2026-09-25)". Neither script imports `v2rule.py`, `score.py` or `window_segments.py`.

It ran on CPUs only (`nice -n 19 ionice -c3`, `CUDA_VISIBLE_DEVICES=""`). `vs_records.py` reads the run's raw window stream and child frames, which are not published; it finds them through `GM1_RUN_S3` (the run's S3 stage folder) and `GM1_SCORING` (the scoring output folder with its `views/`). `vs_track.py` needs only the `candidates.tsv` and `samples.tsv` files published here (gunzip them first).

| File | What it holds |
|---|---|
| `vs_records.py index` → `index.json` | The L0 trajectory, the span counts and edges, the origin jumps, and a check that the scorer's view rows are identical to L0's rows. |
| `vs_records.py layout T` → `layout.jsonl` | The window `uadv`/`vadv` compared with the child frame's `u3d`/`v3d` at 2400, 2880, 3000, 3240 and 3360 s, and the frame velocity from each frame's `domainloc`. |
| `vs_records.py records T...` → `records.jsonl` | The raw-byte SHA-256 check and a full recomputation of the rule quantities for 13 records: once at the scorer's centre, and once with the script's own exhaustive centre search. |
| `vs_records.py courant T0 T1 STRIDE` → `courant-*.jsonl(.gz)` | CM1's `calccflquick` measure (misclibs.F:2726-2814) computed over the window. |
| `vs_track.py RUNS` → `track.json` | Q chains rebuilt with the script's own code from `candidates.tsv`, plus sensitivity cases, which are labelled and are not criterion v2. |
| `tests.log`, `catalogue.log` | The criterion v2 tests at development commit `0a78d9d6`. |

To gunzip the Courant files: `gunzip -k courant-*.jsonl.gz`.
