# GM1 tornado verdicts, criterion v2 (with the v1 verdicts beside them)

| | |
|---|---|
| Scored | 2026-09-22, after the freeze. Every source in the pre-registered table that had data was scored, and the results are reported as they came out. |
| Criterion | v2 as frozen in `198de287e3e304e32172003ad699c1d27e4211b3` (2026-09-22T19:27:37Z). Constants SHA-256 `807b2cc83dcef09bf8067ab6854a0851a910217e9c7fae972c709af83282532e`. `v2rule.py` `87f49b8d…8e27`, `score.py` `d46e453e…7fc4fcc` (`tools/criterion_v2/FROZEN.json`). Rule text: `docs/GM1-TORNADO-CRITERION-V2.md`. |
| Scorer checkout | `022367000461d8822de994a819b48028a9a258e5` (the freeze commit plus a doc-only commit), cloned clean on the scoring machine. Every verdict has `matches_frozen: true` and `protocol_conforming: true`. |
| Where it ran | CPU only, under `nice -n 19 ionice -c3`, with at most 2 scorer processes of about 100 MB each. No GPU was used. The one source scored apart from the others is the E100W window (121 GB): it was scored later that day on the machine that holds it, by one scorer process, CPU only, `nice -n 19 ionice -c3`, from a clean checkout of the same `0223670`, 30.1 min wall. See "How it was run". |
| Outputs | `tools/criterion_v2/results/scoring-v2/` holds `verdict.json`, `samples.tsv` and `candidates.tsv` for each source, plus the logs and `tornado-verdicts-v2.json`, the machine-readable copy. |
| E100R2-PF-4m (added 2026-09-25) | A post-hoc run, not in the pre-registered table, scored with the same frozen files (`matches_frozen: true`, `protocol_conforming: true`, scorer commit `344e3899`). Its window stream has no receipt and lies in two arenas, so it is read through the adapter `tools/criterion_v2/window_segments.py`, which changes no rule. CPU only. Outputs in `tools/criterion_v2/results/scoring-v2-pf4m/`; machine-readable copy `tornado-verdicts-v2-pf4m.json` there (the older JSON is unchanged). See "E100R2-PF-4m (post-hoc run)" below. |
| Reading this outside the development repository | Commit hashes on this page name commits of GM1's private development repository, where the freeze and the scoring were done; they are not in the public history. Run labels (E100V, E100W, E100N and so on) name development runs. Only E100R2-PF-4m has public model output, and none of the others' output is published. The recorded result files had their machine paths replaced by the placeholders `RUNS`, `RUN`, `SCORING` and `CHECKOUT`; no number was changed (`tools/criterion_v2/results/README.md`). What a reader can and cannot check is in "How to check this yourself, and what cannot be checked yet". |

## Words used on this page

This page is the scoring record, kept close to how it was written. The rule itself, with every threshold, is
[GM1-TORNADO-CRITERION-V2.md](GM1-TORNADO-CRITERION-V2.md). These are the short forms used below.

| Word | Meaning |
|---|---|
| T+mm:ss | model time since the start of the parent run |
| parent, child | the coarse grid and the fine nested grid that runs inside it ([NESTING.md](NESTING.md)) |
| child frames | the whole child, saved every minute or two |
| window, window stream | a small box of the child around the vortex, written several times a second |
| record, sample | one saved time of a window or of the frames |
| dx | the grid spacing of the source being scored |
| RMW | radius of maximum wind: where the wind averaged round the centre is fastest |
| Q, U, N | the class of one sample: qualifies, cannot be judged on this grid, does not qualify (step 9 of the rule) |
| E R C A Z W S | the seven conditions on one sample (step 8 of the rule): the centre is inside the search region, the rotation reaches 20 m/s, the wind profile is closed, the flow is round and not a line, the core vorticity, a 35 m/s wind near the ground, and a core at least 4 cells in radius. A lower-case letter is a near miss |
| tracked | linked from one sample to the next as the same vortex |
| spin-up | a child's first 300 s, which hold only what the parent gave it and are judged at the parent's spacing |
| v1, live sidecar | the first version of the rule, which runs inside a nested run (`[nest.tornado_criterion]`) |
| leg L0, L1 | the nested stage of E100R2-PF-4m as first run (L0), and a later leg resumed from 3300 s (L1), whose records are not used here |
| arena | one file of window records. That run's window lies in two |
| clean | in the recorded file names and notes: a record that could be read, up to 3452.4 s. It does not mean the time step limit held. That ended at 3312 s |
| the viz | the project's own films of these runs, which take their captions from the JSON file |
| a code such as `198de287...` | a commit of the private development repository |

Two blocks further down, under "Independent verification" and "Not scored yet, and how to score them", need
model output that is not published. What can be run from public data is in the section after the next.

## Results in brief

* **E100V (4 m): TORNADO** in its 0.5 s window, captioned "simulated tornado (GM1 criterion v2, 4 m grid, 0.5 s output)".
  * One tracked vortex qualified from the window's first record (T+76:00) to its last (T+82:00): 360 s, 719 of 721 records Q.
  * Genesis is at or before T+76:00. The window opens with the vortex already qualifying.
  * Peak 10 m ground-relative wind 111.7 m/s at T+81:40. The azimuthal-mean tangential wind peaked at 82.0 m/s. Median RMW 58 m.
  * Its 60 s child frames give INSUFFICIENT_CADENCE. Every frame from T+75:00 to T+85:00 is Q (11 frames). Between frames, though, the centre moved 112–658 m in the storm-following frame, which is more than the two RMWs added together (84–224 m). The window shows the vortex drifting about 3.4 m/s relative to the domain.
  * The old verdict was NO_TORNADO. That was v1's live sidecar, whose search region on this 2 km child was the single centre corner (1000 m edge margin).
* **E100N (20 m, the first "tornado"): UNRESOLVED** ("grid too coarse to judge").
  * The old verdict was TORNADO, recomputed from the 30 s child frames with v1's own sample code: genesis T+74:30, 330 s.
  * Under v2, the RMW is 40–82 m (median 48 m = 2.4 dx) and reaches 4 dx = 80 m in only 6 of 60 frames. 16 frames sit at the profile's 40 m inner limit.
  * Classes: 5 Q, 45 U, 10 N. The longest tracked Q chain is 30 s. The longest tracked Q-or-U chain is 210 s (T+80:00–83:30).
  * Its strongest frames are T+75:00–79:30, with Vmax 67–91 m/s and RMW 48–82 m. Only T+76:00 and T+76:30 reach 80 m, and those two are Q.
  * The peak frame, T+78:00, has Vmax 90.9 m/s, 117.1 m/s ground-relative at 10 m and p′ −123 hPa. Its RMW is 56 m = 2.8 dx.
* **E100W (20 m): UNRESOLVED** ("grid too coarse to judge") in its 0.5 s window, T+72:00–82:00. The window decides for the span it covers.
  * No genesis: the longest tracked Q chain is 0 s, and there is no episode.
  * Window classes: 1 Q, 1200 U, 0 N. The longest tracked Q-or-U chain is 512 s (T+73:28–82:00), 420 s of it after the spin-up.
  * From T+75:00 on, every record holds E, R, C, A, Z and W. Every record but one fails S. The RMW is 40–204 m (median 44 m = 2.2 dx), and it reaches 4 dx = 80 m only in the one Q record.
  * Peak 10 m ground-relative wind 92.1 m/s at T+76:33.5, where the RMW is 44 m and the circulation 14.2e3 m²/s. The azimuthal-mean tangential wind peaked at 58.0 m/s (T+79:35).
  * Its 75 s child frames give INSUFFICIENT_CADENCE, as the pre-registration said they would. All 12 frames are U (RMW 40–74 m, below 80 m).
  * Old verdict: NO_TORNADO (live v1; the core radius held in 565 of 2400 steps; longest qualifying run 46 s).
* **E100L-500t and E100M (20 m): INSUFFICIENT_CADENCE.** Their frames are 150 s and 120 s apart, above the 60 s limit.
  * Old verdicts, recomputed from those frames: TORNADO (genesis T+80:00, 450 s) and TORNADO (genesis T+76:00, 240 s).
  * Their frames are mostly U, like E100N's: 2 Q, 7 U, 1 N and 1 Q, 8 U, 1 N.
* **E100T (4 m test): INSUFFICIENT_CADENCE** (one frame). That frame is at T+71:00, inside the nest spin-up, so it is judged at 100 m and is U.
  * Old verdict: NO_TORNADO (live v1; ζ and wind failed at every step).
  * Its window stream is empty (0 records) and was not scored.
* **100 m and 200 m runs: UNRESOLVED** by the grid gate. These are E100P (E100V's parent frames stand in for it after T+70), E100N's parent, E100, and the 200 m screens E and T. v1 does not judge parent grids.
* **2 m run (E50N2): not scored.** It has produced no frames yet. The command to score it is below.
* **E100R2-PF-4m (4 m, post-hoc, post latent-heating fix): TORNADO** in its 0.09375 s window before 3312.0 s (T+55:12), the headline span. Stated in one line: a simulated tornado tracked for 455 s with a peak 10 m ground-relative wind of about 128 m/s (129.0 m/s in the last record before the time step limit was passed). Caption "simulated tornado (GM1 criterion v2, 4 m grid, 0.09375 s output)".
  * Genesis T+47:36.9 (2856.9 s). One tracked vortex qualifies from there to the span's last record, T+55:11.9: 455.0 s, 4851 records.
  * Peak 10 m ground-relative wind 129.0 m/s at T+55:11.9, the span's last record (127.6 m/s at T+54:28.1 before 3300 s). The azimuthal-mean tangential wind peaks at 60.8 m/s. Median RMW 128 m = 32 dx (range 16-200 m).
  * After 3312.0 s, where CM1's Courant number was above 1 at times (peak 1.28), the same vortex stays Q in all 1499 records to the last readable one (T+57:32.4). The span alone is TORNADO (140.4 s, genesis at or before T+55:12, peak wind 163.3 m/s). The whole window to the last readable record is one tracked vortex for 595.5 s. These later numbers, and any figure for the whole window, come from numerics that were running away. They are not the run's headline and are not to be quoted.
  * The 120 s full-nest frames give INSUFFICIENT_CADENCE (every frame from T+48:00 to T+56:00 is Q).
  * v1 beside it (no live sidecar; recomputed at window cadence on v1's single centre corner): NO_TORNADO, longest qualifying run 4.6 s.

## How to check this yourself, and what cannot be checked yet

This section is about E100R2-PF-4m, the one run on this page with public model output. How the run was made,
and its decks, are in [NESTING.md](NESTING.md) and `examples/tornado-4m/`. In short:

* **You can check today, from a clone and public data:** that the rule and scorer are the frozen ones; that
  the frozen rule gives every recorded verdict from the recorded per-sample tables; and, by downloading
  12.7 GB, that the scorer reads the model's own winds and gives TORNADO for the last 161 s before the time
  step limit was passed (3150 to 3311 s), with every measurement equal to the recorded one.
* **You cannot check yet:** the genesis time (T+47:36.9), the 455 s duration, or the 129.0 m/s record. They
  rest on window records from 2160 to 3150 s and on every record at 0.09375 s, and those are not public.
  Nobody outside the project has re-derived the headline verdict from the model's winds.
* **Nothing here says the run is right.** These checks show that the verdict follows from the output by the
  frozen rule. They do not test the model. The run was made once, on one RTX 5090, with a build from before
  the 0.1.0 release, and it has not been repeated on a released build. It was scored after the fact, not
  from the pre-registered table. CM1 has no nesting, so there is no CM1 run to compare it with.

All commands run from the top of a clone of the repository. They need Python 3.11 or later with `numpy`,
no GPU, and no installed `gpuwm-cm1`. (Scoring a nest's child frames also works out the older v1 verdict
for comparison, and that part imports the clone's `src/cm1core`. Without GM1's Python packages it records
an error in the `v1_beside` entry and the v2 verdict is unaffected.)

### 1. The recorded verdicts follow from the recorded tables (seconds, nothing to download)

Every scored source has three files under `tools/criterion_v2/results/`: `samples.tsv` (one row per sample
with the strongest candidate's centre, radius of maximum wind, winds and class), `candidates.tsv` (every
candidate vortex of every sample) and `verdict.json`. For the three E100R2-PF-4m window spans the two
tables are gzipped. `check_recorded.py` first checks `v2rule.py` and `score.py` against the hashes in
`FROZEN.json`, then runs the frozen tracking and duration rule on the tables again and compares the result
with `verdict.json`: outcome, caption, genesis, every episode, the longest runs and every count.

```sh
python3 tools/criterion_v2/check_recorded.py
python3 tools/criterion_v2/check_recorded.py --only E100R2-PF-4m-window-pre3312
```

The first command ends with this line, and the second prints the headline:

```text
17 of 17 recorded sources decided again with the same result
```

```text
E100R2-PF-4m-window-pre3312        12287  TORNADO               TORNADO               same
                                 genesis 2856.9375 s, tracked 454.969 s, peak 10 m ground-relative wind 129.0 m/s
```

For the two sources that reach past 3312 s (`E100R2-PF-4m-window-3312on` and `E100R2-PF-4m-window`) the first
command prints a peak of 163.3 m/s with a note under it: that wind is from after the time step limit was passed
and is not to be quoted.

What this shows: given the 12287 rows of per-record measurements, the frozen rule gives TORNADO with that
genesis, duration and peak. What it does not show: that the rows are what the model's winds give. For that
you need the winds (check 3).

You can also read the tables directly. This lists the last five records before 3312 s (time, class,
radius of maximum wind, 10 m ground-relative wind):

```sh
gzip -dc tools/criterion_v2/results/scoring-v2-pf4m/E100R2-PF-4m-window-pre3312/samples.tsv.gz | cut -f2,4,12,20 | tail -n 5
```

### 2. The rule gives its expected answers on made-up vortices (minutes, nothing to download)

The rule was frozen with a set of analytic vortices and negative controls whose expected outcomes were
written first. The sweep makes them again and scores them. The recorded sweep is
`tools/criterion_v2/results/synthetic-v2-prefreeze/sweep.tsv` (105 rows), and the `diff` prints nothing when
every case and grid gives the recorded outcome. The tests need `pytest`. Allow a few minutes to twenty for each, depending on the
machine: the sweep took 18 minutes with 3 workers on a busy one, and gave the recorded table in every column
but the timing.

```sh example
python3 tools/criterion_v2/synthetic_v2.py --out ~/gm1-verdict-check/synthetic --workers 3
cut -f1-4 ~/gm1-verdict-check/synthetic/sweep.tsv | diff - <(cut -f1-4 tools/criterion_v2/results/synthetic-v2-prefreeze/sweep.tsv)
python3 -m pytest -q tests/test_criterion_v2_rule.py tests/test_criterion_v2_synthetic.py tests/test_criterion_v2_scoring_tools.py tests/test_criterion_v2_window_segments.py tests/test_criterion_v2_recorded.py
```

### 3. The scorer on the public winds: TORNADO for the last 161 s before the limit (12.7 GB to download)

The public data are four Hugging Face datasets, all CC BY 4.0, all netCDF in CM1's history layout:

| Dataset | What it holds | Before 3312 s | From 3312 s on (Courant number above 1 at times) |
|---|---|---|---|
| [deepguess/GM1-E100R2-PF-4m-last-5min-every-1s](https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-last-5min-every-1s) | 301 window records, one per second, 3150 to 3450 s, 23.82 GB | 162 records, 3150.0 to 3310.97 s | 139 records |
| [deepguess/GM1-E100R2-PF-4m-last-2min-every-record](https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-last-2min-every-record) | 1307 window records, every 0.09375 s, 3330.0 to 3452.4375 s, 104.46 GB | none | all of it |
| [deepguess/GM1-E100R2-PF-4m-top10-frames](https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-top10-frames) | 10 window records, 3314.16 to 3452.44 s, 0.8 GB | none | all of it |
| [deepguess/GM1-E100R2-PF-4m-full-nest](https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-full-nest) | the whole 4 m nest at 10 times, 2280 to 3360 s every 120 s, 6.77 GB | 9 frames | 1 frame (3360 s) |

Two things to know before using them:

* **The scorer does not read netCDF.** `score.py` reads GM1's own frame folders and window streams.
  `tools/criterion_v2/netcdf_to_window.py` turns a folder of the window-record files into a small stream
  the scorer reads. It needs the `netCDF4` package. It does not read the full-nest files.
* **Two of the four datasets lie wholly after 3312 s**, and so does almost half of a third. From 3312.0 s
  the Courant number was above 1 at times, the winds grew to 163 m/s, and the run failed at 3452.8 s. Winds
  from that span must not be quoted as the run's intensity. Only records before 3312 s back the headline.
  The dataset cards and the files' own attributes were written before this scoring and still say the
  vortex "has not been scored"; this page is the scoring.

The 162 records before 3312 s in the every-second dataset are enough for a verdict of their own: they are
1.03 s apart and span 161 s, and the rule needs one tracked vortex for 120 s. Download them (the manifest
lists every file's model time and SHA-256), convert, score, and compare with the recorded rows:

```sh example
mkdir -p ~/pf4m-5min && cd ~/pf4m-5min
B=https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-last-5min-every-1s/resolve/main
curl -sSLO $B/MANIFEST.tsv
awk -F'\t' 'NR>1 && $4 < 3312 {print $1}' MANIFEST.tsv | xargs -P 4 -I{} curl -sSLO $B/{}
awk -F'\t' 'NR>1 && $4 < 3312 {print $3"  "$1}' MANIFEST.tsv | sha256sum -c --quiet -
```

Then, back at the top of the clone:

```sh example
python3 -m pip install netCDF4
python3 tools/criterion_v2/netcdf_to_window.py ~/pf4m-5min --out ~/gm1-verdict-check/stream --before 3312
python3 tools/criterion_v2/score.py ~/gm1-verdict-check/stream --out ~/gm1-verdict-check/scored --nest-start-s 2160 --parent-dx 100 --workers 4
python3 tools/criterion_v2/check_recorded.py --rows ~/gm1-verdict-check/scored --against E100R2-PF-4m-window-pre3312
```

What we got doing exactly this from the public files (the conversion took half a minute and the scoring
about a minute on four processes):

```text
{
 "outcome": "TORNADO",
 "caption": "simulated tornado (GM1 criterion v2, 4 m grid, 1.03125 s output)",
 "reason": null,
 "samples": 162,
 "median_interval_s": 1.03125,
 "longest_qualifying_run_s": 160.96875,
 "class_counts": {
  "Q": 162
 }
}
```

```text
162 samples and 367 candidate rows of (your folder) against E100R2-PF-4m-window-pre3312, by record number
samples absent from the recorded source: 0; with a different measurement: 0; with different candidates: 0
every measurement equals the recorded one
```

So from public data: one vortex qualifies in all 162 records from 3150.0 to 3310.97 s (T+52:30 to
T+55:11), tracked for 161.0 s, with a peak 10 m ground-relative wind of 126.7 m/s at 3268.03 s, a peak
azimuthal-mean tangential wind of 58.0 m/s and a median radius of maximum wind of 130 m. Each of those 162
records has the same centre, radius, winds and class, to every printed digit, as the row with the same
record number in the recorded headline table. That verdict is the scorer's verdict on those 162 records as a
source of their own. It is not the headline verdict, and its "genesis" is only the first record you gave it.

The same conversion reads the records from 3312 s on (`--from 3312` in place of `--before 3312`, compared
`--against E100R2-PF-4m-window-3312on`). We checked five of them that way and they equal the recorded rows.
They are in the span that must not be quoted.

The conversion takes two things on trust that the files do not hold: the nest's size (500 x 500 cells) and
its edge zones (1 specified cell and 4 relaxation cells). They are the tool's defaults, they are in the
run's deck (`examples/tornado-4m/S3.toml`), and they only decide how close to the nest's edge a vortex may be counted.

### 4. Score your own run

The scorer reads what a GM1 nested run writes. Give it the model time at which the nest was started and
the parent's grid spacing:

```sh example
python3 tools/criterion_v2/score.py RUN/frames/nest-child-window --out ~/gm1-verdict-check/mine-window --nest-start-s 2160 --parent-dx 100
python3 tools/criterion_v2/score.py RUN/frames/nest-child --out ~/gm1-verdict-check/mine-frames --nest-start-s 2160 --parent-dx 100
```

* **Frames 120 s apart cannot give TORNADO.** The rule needs samples at most 60 s apart over 120 s. This
  run's ten child frames give INSUFFICIENT_CADENCE, although the vortex qualifies in every frame from
  T+48:00 to T+56:00. Write a window stream, or frames at most 60 s apart.
* **A window stream is large.** This run's was 137.6 MB per record and 1.6 TB in all. The scorer reads two
  levels of two fields from it. `tools/criterion_v2/window_excerpt.py` copies only that part (2.2 MB per
  record here), so a stretch of a stream can be kept or passed on and still be scored.
* **A stream without its receipt** (`window-stream.json` is written when a run ends normally) or kept in
  several pieces goes through `tools/criterion_v2/window_segments.py`, with a spec like
  `tools/criterion_v2/scoring-sources-v2-pf4m.json`.
* The tools under `tools/criterion_v2/` are source tools run from a clone. They are not part of the
  installed `gpuwm-cm1` command.

### 5. What cannot be checked yet

* **The headline itself.** Genesis at T+47:36.9, 455.0 s of tracking and the 129.0 m/s record at
  3311.90625 s come from all 12287 window records from 2160.09 to 3311.91 s. The public records start at
  3150 s and are one per second, and the record at 3311.90625 s is not among them. The raw stream is
  1.6 TB on one disk and is not published. Until records before 3150 s are public, the first 293 s of the
  tornado, and its genesis, can be read in the tables (check 1) and not re-derived from winds.
* **The independent verification's raw-record part.** Its files are in
  `tools/criterion_v2/results/scoring-v2-pf4m/verify/`. `vs_track.py` runs on the published tables.
  `vs_records.py` needs the raw stream and the child frames.
* **The Courant numbers** that put the boundary at 3312.0 s. The values are published
  (`verify/courant-*.jsonl`), but recomputing them needs the vertical wind on every level of each record,
  which only the netCDF files from 3150 s on hold; we have not written that check for the public files.
* **Every other run on this page** (E100V, E100W, E100N and the rest). Their tables are published and pass
  check 1. None of their model output is public.
* **The run on a released build.** The run has not been repeated with 0.1.x, and a repeat need not form
  its vortex at the same place or time.

## What changed from v1

* The 4 m two-celled vortex that v1 missed is a TORNADO in the 0.5 s stream.
* The 20 m vortices that v1 called TORNADO (E100N, and E100L-500t and E100M at frame cadence) are UNRESOLVED or INSUFFICIENT_CADENCE. At 20 m their radius of maximum wind is below 4 grid lengths.
* E100W, which v1 called NO_TORNADO, is UNRESOLVED in its 0.5 s window for the same reason. Its vortex is tracked through the whole window, but its RMW is mostly 2 dx.

## Verdict table

Column notes:
* **Peak wind:** the maximum ground-relative wind at 10 m AGL within 1.5 RMW of the centre. For a TORNADO it is taken over the tracked episode. Otherwise it is the maximum over the strongest candidate of each sample, whatever that candidate's class.
* **RMW and circulation:** over the episode for a TORNADO, otherwise at the peak-wind sample. Circulation is 2π·RMW·Vmax, where Vmax is the peak azimuthal-mean tangential wind.
* **Duration:** for other outcomes, the longest identity-linked (tracked) chains.
* **Class letters:** Q qualifies, U cannot be judged on this grid, N does not qualify.

| Run (source) | Grid | Cadence | Old verdict (v1) | v2 verdict | Genesis (v2) | Peak 10 m AGL ground-relative wind | RMW | Circulation 2&pi;&middot;RMW&middot;Vmax | Duration | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| E100V (window stream) | 4 m (lowest level 2 m) | 0.5 s window stream; 721 samples T+76:00-T+82:00 | NO_TORNADO (the run's v1, live, every parent step) | **TORNADO** | T+76:00 (first sample: at or before) | 111.7 m/s at T+81:40 | median 58 m (range 16-246 m) | median 13.7e3, peak 37.2e3 m2/s at T+80:30 | 360 s (715 samples, T+76:00-T+82:00); 1 tracked tornado(es) | 0.5 s window stream, T+76-82 (advecting winds uadv/vadv); classes Q 719, U 2 |
| E100V (child frames) | 4 m (lowest level 2 m) | 60 s frames; 15 samples T+71:00-T+85:00 | NO_TORNADO (live, every parent step) | **INSUFFICIENT_CADENCE** | - | 106.0 m/s at T+82:00 | 68 m (at the peak-wind sample) | 33.2e3 m2/s (at the peak-wind sample) | longest tracked Q 60 s; Q or U 60 s; Q via plausible links 600 s | 60 s child frames T+71-85; qualifying vortices were present in consecutive samples for 840 s, but between samples the centre moved more than the two core radii in the storm-following frame: the output interval cannot show that one vortex persisted; classes Q 11, U 4 |
| E100P (parent frames) | 100 m (lowest level 10 m) | 60 s frames; 15 samples T+71:00-T+85:00 | - (v1 judges nest children only) | **UNRESOLVED** | - | 72.7 m/s at T+83:00 | 200 m (at the peak-wind sample) | 31.8e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 240 s; Q via plausible links 0 s | E100V's 100 m parent frames stand in for E100P after T+70 (pre-registered); horizontal grid spacing 100 m exceeds 25 m; classes U 11, N 4 |
| E100W (child frames) | 20 m (lowest level 10 m) | 75 s frames; 12 samples T+71:15-T+85:00 | NO_TORNADO (live, every parent step) | **INSUFFICIENT_CADENCE** | - | 88.9 m/s at T+80:00 | 40 m (at the peak-wind sample) | 13.9e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 0 s; Q via plausible links 0 s | 75 s child frames: INSUFFICIENT_CADENCE by construction (pre-registered); 12 sample(s) spanning 825 s with intervals [75.0] s: no stretch of the record at <= 60 s spacing lasts 120 s; classes U 12 |
| E100W (window stream) | 20 m (lowest level 10 m) | 0.5 s window stream; 1201 samples T+72:00-T+82:00 | NO_TORNADO (the run's v1, live, every parent step) | **UNRESOLVED** | - | 92.1 m/s at T+76:33.5 | 44 m (at the peak-wind sample) | 14.2e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 512 s; Q via plausible links 0 s | 0.5 s window stream, T+72-82; 121 GB, so scored on the machine that holds it; one tracked vortex met every condition for >= 120 s except that its RMW was below 4 dx or a threshold was met only within the discretisation uncertainty (marginal); classes Q 1, U 1200 |
| E100N (child frames) | 20 m (lowest level 10 m) | 30 s frames; 60 samples T+60:30-T+90:00 | TORNADO, genesis T+74:30 (recomputed on the child frames) | **UNRESOLVED** | - | 117.1 m/s at T+78:00 | 56 m (at the peak-wind sample) | 32.0e3 m2/s (at the peak-wind sample) | longest tracked Q 30 s; Q or U 210 s; Q via plausible links 30 s | fixed 20 m nest, 30 s child frames T+60-90; no live v1 sidecar; one tracked vortex met every condition for >= 120 s except that its RMW was below 4 dx or a threshold was met only within the discretisation uncertainty (marginal); classes Q 5, U 45, N 10 |
| E100N parent (parent frames) | 100 m (lowest level 10 m) | 30 s frames; 60 samples T+60:30-T+90:00 | - (v1 judges nest children only) | **UNRESOLVED** | - | 86.1 m/s at T+80:00 | 200 m (at the peak-wind sample) | 30.5e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 300 s; Q via plausible links 0 s | E100N's 100 m parent frames, 30 s; horizontal grid spacing 100 m exceeds 25 m; classes U 20, N 40 |
| E100L-500t (child frames) | 20 m (lowest level 10 m) | 150 s frames; 10 samples T+67:30-T+90:00 | TORNADO, genesis T+80:00 (recomputed on the child frames) | **INSUFFICIENT_CADENCE** | - | 116.6 m/s at T+80:00 | 56 m (at the peak-wind sample) | 32.1e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 0 s; Q via plausible links 0 s | 20 m nest, 150 s child frames; 10 sample(s) spanning 1350 s with intervals [150.0] s: no stretch of the record at <= 60 s spacing lasts 120 s; classes Q 2, U 7, N 1 |
| E100M (child frames) | 20 m (lowest level 10 m) | 120 s frames; 10 samples T+72:00-T+90:00 | TORNADO, genesis T+76:00 (recomputed on the child frames) | **INSUFFICIENT_CADENCE** | - | 121.0 m/s at T+78:00 | 44 m (at the peak-wind sample) | 24.4e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 0 s; Q via plausible links 0 s | 20 m nest, 120 s child frames; 10 sample(s) spanning 1080 s with intervals [120.0] s: no stretch of the record at <= 60 s spacing lasts 120 s; classes Q 1, U 8, N 1 |
| E100T (child frames) | 4 m (lowest level 10 m) | one sample | NO_TORNADO (live, every parent step) | **INSUFFICIENT_CADENCE** | - | 34.2 m/s at T+71:00 | 16 m (at the peak-wind sample) | 1.1e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 0 s; Q via plausible links 0 s | 4 m test, one child frame: INSUFFICIENT_CADENCE by construction; 1 sample(s) spanning 0 s with intervals [] s: no stretch of the record at <= 60 s spacing lasts 120 s; classes U 1 |
| E100T (window stream) | 4 m | - | - | **not scored** | - | - | - | - | - | E100T's window stream; E100T/frames/nest-child-window: no window-stream.json (the stream has no receipt: empty or unfinished) |
| E100 (parent frames) | 100 m (lowest level 10 m) | 300 s frames; 20 samples T+5:00-T+100:00 | - (v1 judges nest children only) | **UNRESOLVED** | - | 70.9 m/s at T+80:00 | 200 m (at the peak-wind sample) | 43.8e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 0 s; Q via plausible links 0 s | 100 m storm run, 300 s frames; horizontal grid spacing 100 m exceeds 25 m; classes U 2, N 18 |
| E (parent frames) | 200 m (lowest level 10 m) | 300 s frames; 24 samples T+5:00-T+120:00 | - (v1 judges nest children only) | **UNRESOLVED** | - | 44.0 m/s at T+55:00 | 400 m (at the peak-wind sample) | 30.7e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 0 s; Q via plausible links 0 s | 200 m screen, 300 s frames; horizontal grid spacing 200 m exceeds 25 m; classes U 1, N 23 |
| T (parent frames) | 200 m (lowest level 10 m) | 300 s frames; 20 samples T+5:00-T+100:00 | - (v1 judges nest children only) | **UNRESOLVED** | - | 13.6 m/s at T+25:00 | 1000 m (at the peak-wind sample) | 3.1e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 0 s; Q via plausible links 0 s | 200 m screen, 300 s frames; horizontal grid spacing 200 m exceeds 25 m; classes N 20 |
| E50N2 (child frames) | 2 m | - | - | **not scored** | - | - | - | - | - | 2 m moving nest, 15 s child frames; E50N2/frames/nest-child did not exist when the scoring ran |
| E50N2 (window stream) | 2 m | - | - | **not scored** | - | - | - | - | - | 2 m window stream, 180-720 s; E50N2/frames/nest-child-window did not exist when the scoring ran |
| E100R2-PF-4m (window stream, before 3312 s: headline) | 4 m (lowest level 2 m) | 0.09375 s window stream; 12287 samples T+36:00.1-T+55:11.9 | NO_TORNADO (recomputed on the window records) | **TORNADO** | T+47:36.9 | 129.0 m/s at T+55:11.9 | median 128 m (range 16-200 m) | median 30.4e3, peak 58.0e3 m2/s at T+53:14.6 | 454.969 s (4851 samples, T+47:36.9-T+55:11.9); 1 tracked tornado(es) | 0.09375 s window stream before 3312.0 s (Courant number <= 1), via window_segments.py; post-hoc run; classes Q 4889, U 3531, N 3867 |
| E100R2-PF-4m (window stream, 3312-3452.4 s) | 4 m (lowest level 2 m) | 0.09375 s window stream; 1499 samples T+55:12-T+57:32.4 | NO_TORNADO (recomputed on the window records) | **TORNADO** | T+55:12 (first sample: at or before) | 163.3 m/s at T+57:26.8 | median 122 m (range 80-172 m) | median 53.2e3, peak 70.8e3 m2/s at T+57:20.1 | 140.438 s (1499 samples, T+55:12-T+57:32.4); 1 tracked tornado(es) | 0.09375 s window stream 3312.0-3452.4375 s (Courant number above 1 at times, peak 1.28), via window_segments.py; classes Q 1499 |
| E100R2-PF-4m (window stream, whole window to the last readable record; includes the span after 3312 s, not to be quoted) | 4 m (lowest level 2 m) | 0.09375 s window stream; 13786 samples T+36:00.1-T+57:32.4 | NO_TORNADO (recomputed on the window records) | **TORNADO** | T+47:36.9 | 163.3 m/s at T+57:26.8 | median 124 m (range 16-200 m) | median 36.5e3, peak 70.8e3 m2/s at T+57:20.1 | 595.5 s (6350 samples, T+47:36.9-T+57:32.4); 1 tracked tornado(es) | the whole clean 0.09375 s window stream 2160.09375-3452.4375 s, via window_segments.py; classes Q 6388, U 3531, N 3867 |
| E100R2-PF-4m (child frames) | 4 m (lowest level 2 m) | 120 s frames; 10 samples T+38:00-T+56:00 | NO_TORNADO (recomputed on the child frames) | **INSUFFICIENT_CADENCE** | - | 111.2 m/s at T+56:00 | 120 m (at the peak-wind sample) | 52.0e3 m2/s (at the peak-wind sample) | longest tracked Q 0 s; Q or U 0 s; Q via plausible links 0 s | 120 s full 4 m nest frames 2280-3360 s (the 3360 s frame is inside the Courant > 1 span); 10 sample(s) spanning 1080 s with intervals [120.0] s: no stretch of the record at <= 60 s spacing lasts 120 s; classes Q 5, U 2, N 3 |

The same table is written by `tools/criterion_v2/verdict_table.py`
(`tools/criterion_v2/results/scoring-v2/verdict-table.md`). The recorded tables are kept as they were written:
there the last E100R2-PF-4m window row is labelled "whole clean window" and has no note. Read it with the note
above.

## Run verdicts and captions

The rule's text, not a new rule, says which source decides when a run has more than one. When frames
cannot show persistence, the window stream decides for the span it covers. E100W's verdict must come
from its window.

| Run | v2 run verdict | From | Caption (exact wording) |
|---|---|---|---|
| E100V | TORNADO | 0.5 s window, T+76:00–82:00 (genesis at or before T+76:00) | simulated tornado (GM1 criterion v2, 4 m grid, 0.5 s output) |
| E100W | UNRESOLVED | 0.5 s window, T+72:00–82:00 (the 75 s child frames alone give INSUFFICIENT_CADENCE) | grid too coarse to judge (GM1 criterion v2) |
| E100N | UNRESOLVED | 30 s child frames | grid too coarse to judge (GM1 criterion v2) |
| E100L-500t | INSUFFICIENT_CADENCE | 150 s child frames | output too sparse or too short to judge duration (GM1 criterion v2) |
| E100M | INSUFFICIENT_CADENCE | 120 s child frames | output too sparse or too short to judge duration (GM1 criterion v2) |
| E100T | INSUFFICIENT_CADENCE | one child frame | output too sparse or too short to judge duration (GM1 criterion v2) |
| E100P, E100N parent, E100, E, T | UNRESOLVED | 100 m / 200 m frames | grid too coarse to judge (GM1 criterion v2) |
| E50N2 (2 m) | not scored yet | - | - |
| E100R2-PF-4m (post-hoc) | TORNADO | 0.09375 s window before 3312.0 s, T+36:00.1-55:11.9 (genesis T+47:36.9); the later span, 3312.0-3452.4 s, is reported, not the headline (Courant number above 1) | simulated tornado (GM1 criterion v2, 4 m grid, 0.09375 s output) |

Captions per time window for the viz are in `tornado-verdicts-v2.json`. Look under
`runs.<run>.sources[].time_windows`, which gives each window's state, caption and numbers, and
`class_segments`, which gives the Q/U/N class over time.
* E100V's window is one tornado window, T+76:00–82:00.
* In E100V's 60 s frames, T+71:00–74:00 is U: spin-up until T+75:00, judged at 100 m. T+75:00–85:00 is Q in every frame.
* Outside T+76–82, E100V has only frame-cadence evidence, so its caption there is the INSUFFICIENT_CADENCE one.
* E100W's window has one time window, the whole record T+72:00–82:00, with the UNRESOLVED caption. Its records before T+75:00 are spin-up, judged at 100 m. The class is U throughout except one Q record at T+76:12.5.
* Outside T+72–82 (the 75 s frames at T+71:15, T+82:30, T+83:45 and T+85:00), E100W has only frame-cadence evidence, so its caption there is the INSUFFICIENT_CADENCE one.

## Per-run notes

**E100V window (4 m, 0.5 s, `uadv`/`vadv` at 10 m).**
* 721 records. All 721 hold E, R, C, A, Z and W. S holds in 719: the two U records (T+78:04.5 and T+78:06.5) have RMW 14 m, below 4 dx = 16 m.
* The tracked chain has 715 records. The 3 s bridge carries it across those two records.
* The row candidate's RMW is 14–160 m (median 56 m). The tracked chain's is 16–246 m (median 58 m). The larger values come from records where Vbar peaks far out; 34 records have a row RMW over 100 m.
* The inner/core vorticity ratio, which is reported only and is below 1 for a two-celled vortex, is below 1 in 356 records.
* verdict.json lists 42 overlapping maximal chains. They all start at T+76:00 and share candidates, so they are one tracked vortex. The table groups them.
* The window has no pressure (no `prs`). The 60 s frames give p′ down to −75.5 hPa.

**E100V child frames (60 s).**
* The four frames before T+75:00 are judged at the parent's 100 m (spin-up) and are U. The later eleven are Q.
* Between consecutive frames, the strongest candidate moved 112–658 m in the storm-following frame, against RMW sums of 84–224 m. Only one such link (T+81:00→82:00, 112 m against 124 m) is an identity link, and the longest tracked Q chain is 60 s.
* So v2 cannot show from these frames that one vortex persisted. This is the pre-registered limit of frame cadence, and the window decides instead.

**E100N child frames (20 m, 30 s).**
* v2 finds a strong vortex: Vmax up to 90.9 m/s, 10 m ground-relative wind up to 117.1 m/s, p′ down to −123 hPa, all at T+78:00, where the RMW is 56 m. At 20 m its RMW is mostly 2–4 dx (median 2.4 dx), and S fails in 54 of 60 frames.
* R is marginal in 13 frames and W in 6. 16 frames are flagged `rmw-at-inner-radius`: Vbar peaks at the profile's 2 dx inner limit, so the true RMW may be smaller than this grid can show.
* The five Q frames are T+74:30, T+76:00, T+76:30, T+81:00 and T+81:30. No two of them are tracked as the same vortex for 120 s.
* The Q-or-U chain of 210 s (T+80:00–83:30) makes the outcome UNRESOLVED rather than NO_TORNADO.
* v1 beside it was recomputed from the 30 s frames (E100N has no live sidecar): TORNADO, genesis T+74:30, longest qualifying run 330 s.

**E100W window (20 m, 0.5 s, `uadv`/`vadv` at 10 m).**
* 1201 records, T+72:00–82:00, 301 × 301 cells (6 km). The lowest scalar level is 10 m, so 10 m is read directly.
* The first 360 records (to T+74:59.5) are spin-up, judged at the parent's 100 m. They are all U: RMW 40–126 m against 4 dx = 400 m. R is marginal in 62 records and W in 21, all of them in the spin-up.
* From T+75:00 on (841 records), 840 records hold E, R, C, A, Z and W and fail only S. Their RMW is 40–204 m (median 44 m = 2.2 dx). 219 records are flagged `rmw-at-inner-radius` (Vbar peaks at the 40 m = 2 dx inner limit, so the true RMW may be smaller). Vmax is 22.9–58.0 m/s (median 46.6 m/s), and the 10 m ground-relative wind is 48.4–92.1 m/s.
* The one Q record, T+76:12.5, is a broad candidate: RMW 204 m, Vmax 22.9 m/s, 62.6 m/s ground-relative. It is not tracked into any other record, so the longest tracked Q chain is 0 s, and 0 s through plausible links too.
* The longest tracked Q-or-U chain has 1024 records, all U, over 512 s (T+73:28–82:00): RMW median 44 m (40–178 m), circulation median 12.9e3 m²/s, peak 10 m wind 92.1 m/s at T+76:33.5. 840 of its records lie after the spin-up and span 420 s (T+75:00–82:00), so the UNRESOLVED does not rest on spin-up samples. This was located with the frozen `v2rule._chains` on `candidates.tsv`, the bookkeeping `verdict_table.py` does for episodes. It decides nothing.
* The inner/core vorticity ratio (reported only) is below 1 in 1 record. The window has no pressure (no `prs`); the 75 s frames give p′ down to −50.8 hPa.
* E100V and E100W both resume E100P's restart at T+70 (`restart-000002`), with a 4 m and a 20 m nest. At 20 m a vortex with an RMW of 40–60 m is below 4 dx = 80 m. So it is U, not a verdict either way, as the rule says ("On a 20 m grid, v2 cannot see a tornado with RMW < 80 m").
* The spawn time and parent spacing are the manifest's (4200 s, 100 m). They match E100W's `receipt.json` (`restart.resumed_model_time_s` 4200.0, resumed from `E100P/restarts/restart-000002`) and `E100W.toml` (`[grid] dx_m = 100.0`).

**E100W child frames (20 m, 75 s).**
* The first three frames (to T+73:45) are judged at 100 m (spin-up).
* All 12 frames are U: RMW 40–74 m, peak 10 m wind 88.9 m/s at T+80:00, p′ down to −50.8 hPa.
* The live v1 said NO_TORNADO. Its core radius held in 565 of 2400 parent steps.

**E100L-500t and E100M (20 m, 150 s and 120 s).**
* INSUFFICIENT_CADENCE by the cadence rule.
* The frames show the same 20 m picture as E100N: RMW mostly 40–72 m (U), a few Q frames with RMW 88–134 m, and winds up to 116.6 and 121.0 m/s.
* v1 recomputed on these frames says TORNADO for both. v1's 60 s run and 40 m jump allowance are loose at 120–150 s spacing.

**E100T (4 m test, one frame at T+71:00).**
* The frame is inside the spin-up, so it is judged at 100 m.
* δ = 0.5·100/16 is above 1, so R and W read as marginal (lower case) and the class is U, even though Vmax is only 11.4 m/s.
* The outcome is INSUFFICIENT_CADENCE either way.

**Parents and screens.** UNRESOLVED by the grid gate. Their samples are in `samples.tsv`. At 100 m the RMW sits at the 200 m profile inner limit in most samples.

## Observations recorded, rule not changed

Nothing below changed a threshold or the code. A change would be a v3 with both verdicts kept.

1. **Spin-up samples judged at 100 m are U whenever they are closed and round.** With RMW ≤ 50 m on a 4 m or 20 m child, δ = 0.5·100/RMW ≥ 1, so the marginal band covers any value (E100T's 11 m/s vortex reads U). This can only turn a NO_TORNADO into UNRESOLVED, and only if a Q-or-U chain of 120 s or more lies in the spin-up. No verdict here depends on that: E100V's longest tracked Q-or-U chain is 60 s, and E100N's 210 s chain is at T+80:00–83:30, after its spin-up. E100W's frames give 0 s. Its window's 512 s chain starts in the spin-up (T+73:28), but 420 s of it (T+75:00–82:00) lies after it, so the window's UNRESOLVED stands without the spin-up samples.
2. **Overlapping episodes.** When several candidates of one vortex are kept, verdict.json lists every maximal chain. For E100V that is 42 chains. The aggregation groups chains that share candidates. This is bookkeeping and does not affect the outcome.
3. **Frame cadence against drift.** E100V's vortex drifts about 3.4 m/s relative to the domain, which is about 200 m per minute, more than its core diameter. So 60 s frames cannot link it, even though it qualifies in every frame. This is the limit written in the rule ("What v2 does not claim").
4. **v1's E100V verdict came from a single point.** The live v1 on E100V's 2 km child searched one corner, the child centre, because of the 1000 m edge margin (`search_corners` i, j = 250..250). Its NO_TORNADO says nothing about the vortex away from that point.

## E100R2-PF-4m (post-hoc run)

Idealized supercell in the 24 May 2011 El Reno environment, after the latent-heating fix. A 4 m moving
nest (500 x 500 cells, 2 km, ratio 25) was spawned at 2160 s from a 100 m parent (S3 deck, `rstnum = 8`).
A 227 x 227 x 45 window follows the nest centre every 0.09375 s. The run is not in the pre-registered
table. It was scored after the fact with the unchanged frozen files. The three spans are in the
adapter's spec (`tools/criterion_v2/scoring-sources-v2-pf4m.json`, commit `9cd35503`, 2026-09-24T23:00:13Z; its spans, segments and times are unchanged since, and the published copy has only its machine paths replaced by placeholders)
and were taken from the Courant diagnosis in the run's status notes (2026-09-24T11:10Z, not published), which predates all
criterion work on this run. The scoring recorded here started at 2026-09-25T09:12Z. A stopped earlier
attempt's run script was reported as 23:02Z, two minutes after the spec commit; its files are deleted, so
that order cannot be checked now. It does not matter for the boundary: 3312.0 s is fixed by the Courant
number, and the independent verification recomputed it from the records (below).

**The data read.** Leg L0 only, and never `S3/frames-L1`.
* `S3/frames/nest-child-window`: 12160 records, 2160.09375-3300.0 s.
* `S3/L0-window-tail-3300.09375-3452.4375`: L0's own records after the L1 resume point, 1626 records with their own index from 0 (`README.json`: `l0_index_first` 12160).
* The last readable record is at 3452.4375 s (the recorded files call this the clean limit). L0 failed at 3452.8 s, and its runaway records 13786-13789 were deleted.
* 13786 records in all, every one read. A SHA-256 check of 16 records spread over both arenas matched the index.

**The spans.** CM1's Courant number (`calccflquick`, misclibs.F:2726-2814) was above 1 at times from T+55:12 (3312.0 s), and most of the time from T+56:45 (peak 1.28 at T+57:06). Leg L0 failed at T+57:33.
* Before 3312.0 s: 12287 records, 2160.09375-3311.90625 s. This is the headline.
* 3312.0-3452.4375 s: 1499 records.
* The whole window to the last readable record: 13786 records. The recorded files call it "the whole clean window". It includes the 1499 records after 3312.0 s.
* The 10 full 4 m nest frames, 2280-3360 s every 120 s, read by `score.py` directly, as E100V's frames were. The 3360 s frame lies inside the Courant span.

**The frame.** The window's winds are relative to the translating domain (`umove = vmove = 16.5` m/s; CM1 base.F:2663-2670). The frozen scorer handles this as it did for E100V and E100W.
* The hazard wind is the frame wind plus (umove, vmove) (v2rule step 7, CM1 solve3.F:864-865 `sws` form).
* The window is placed on the ground at `origin + umove * t` (step 10).
* Identity links subtract the domain's motion again (the storm-following frame).
* The adapter passes the deck's umove/vmove and each index row's `origin_x_m`/`origin_y_m` through unchanged. The winds are the advecting `uadv`/`vadv`, as for E100V. The first scalar level is 2 m, and 10 m is CM1 level 3 exactly (child faces every 4 m; param.F:8169).

### What the adapter did

The adapter is `tools/criterion_v2/window_segments.py`. It changes no threshold, window rule or definition. `v2rule.py` and `score.py` are the frozen bytes (`FROZEN.json`; `git diff 2b70e09f` touches neither). It does three things.
* **No receipt.** The run ended abnormally, so the window stream has no `window-stream.json`. The adapter derives the fields the frozen reader takes from a receipt from `S3/deck.toml`.
  * Child spacing 100/25 = 4 m; child nx = ny = 20 x 25 = 500; spec + relax zone 1 + 4 = 5.
  * umove and vmove 16.5 m/s.
  * Scalar levels from `child_levels_m`, using CM1's mid-points and mirrored ghost (param.F:8165-8172).
  * The schema-v3 record layout, 137575536 B per record.
  * Every derived field equals the real receipt that the same deck and code wrote in a short control rerun resumed at 3300 s (not published). Its 32 records (3300.09375-3303.0 s) are byte-identical, by index SHA-256, to L0's tail at the same times.
* **Two arenas.** Each segment gets a view directory: the derived receipt, the segment's index rows inside the span, and a symlink to its arena. The frozen `score.window_jobs` reads each view, and the jobs are joined in time order. Model times must increase strictly across segments.
* **Spans.** A span only chooses records. Inside it the scorer reads every record (no `--every`/`--start`/`--end`), so `protocol_conforming` stays as the frozen scorer computes it.
* **v1 beside.** PF-4m has no live v1 sidecar: its deck has no `[nest.tornado_criterion]`, because v1's 1000 m margin degenerates on a 2 km child. So v1's own `reference_sample` and ledger were run on every window record. The window's two 10 m bracketing levels are placed into a child-sized array at the window's own (i0, j0). The adapter checks that v1's centre corner and its 300 m disk lie inside the window. v1 is reported only; under v2 it decides nothing.
* **Tests.** The fast tests (`tests/test_criterion_v2_window_segments.py`, 7 tests) run on synthetic streams. The adapter's samples, candidates and outcome equal the frozen scorer's on the same stream written with a receipt. A span reads only its records. The adapter refuses overlapping segments and a reference that disagrees. The v1 at window cadence is checked too.
  * The catalogue test (`tests/test_criterion_v2_window_segments_catalogue.py`, `GM1_CRITERION_CATALOGUE=1`) runs all 26 synthetic cases v2 was frozen with, at 4 m, written as a receipt-less two-segment stream. The adapter gives the frozen scorer's samples, candidates and outcome on every case, and an outcome the catalogue expects at 4 m. It ran: 33 passed.
* **Bookkeeping.** `verdict_table.group_events` now groups overlapping chains in time linear in their members. PF-4m lists 2254 and 2741 overlapping maximal chains of the one vortex, which the pairwise test could not finish. A test checks that the grouping is the same as the pairwise grouping.

### Results per span

The T+ times are model time / 60. "Q", "U" and "N" are the rule's classes. "Tracked" means identity-linked (`v2rule._chains`). RMW/dx is over the tracked episode's members.

| Span | v2 | Genesis | Tracked Q | Peak 10 m ground-relative wind | Vmax (azimuthal mean) | RMW (RMW/dx) | Circulation 2&pi;&middot;RMW&middot;Vmax | Classes |
|---|---|---|---|---|---|---|---|---|
| Before 3312.0 s (headline), T+36:00.1-55:11.9 | **TORNADO** | T+47:36.9 (2856.9375 s) | 455.0 s, T+47:36.9-55:11.9 (4851 records; runs to the span's end) | 129.0 m/s at T+55:11.9 (127.6 m/s at T+54:28.1 before 3300 s) | 60.8 m/s at T+54:22.7 | median 128 m, range 16-200 m (median 32 dx, 5-95 % 19-43 dx, min 4 dx) | median 30.4e3, peak 58.0e3 m2/s at T+53:14.6 | Q 4889, U 3531 (3199 in the spin-up to 2460 s, judged at 100 m), N 3867 |
| 3312.0-3452.4375 s, T+55:12-57:32.4 (Courant number above 1 at times; not to be quoted) | **TORNADO** | at or before T+55:12 (the first record) | 140.4 s, the whole span (1499 records) | 163.3 m/s at T+57:26.8 | 81.4 m/s at T+57:15.5 | median 122 m, range 80-172 m (median 30.5 dx, min 20 dx) | median 53.2e3, peak 70.8e3 m2/s at T+57:20.1 | Q 1499 |
| Whole window to the last readable record, T+36:00.1-57:32.4 (includes the span after 3312 s; not to be quoted) | **TORNADO** | T+47:36.9 | 595.5 s, T+47:36.9-57:32.4 (6350 records; one vortex across 3312 s to the last readable record) | 163.3 m/s at T+57:26.8, after 3312 s | 81.4 m/s at T+57:15.5 | median 124 m, range 16-200 m (median 31 dx) | median 36.5e3, peak 70.8e3 m2/s at T+57:20.1 | Q 6388, U 3531, N 3867 |
| 120 s full-nest frames, T+38:00-56:00 | **INSUFFICIENT_CADENCE** | - | 0 s (frames 120 s apart; the limit is 60 s) | 111.2 m/s at T+56:00 (inside the Courant span) | 69.0 m/s at T+56:00 | 120 m at the peak-wind frame | 52.0e3 m2/s at the peak-wind frame | Q 5 (every frame T+48:00-56:00), U 2 (T+38, 40: spin-up), N 3 (T+42-46) |

* **Time ranges that pass.** The tracked Q episode runs from T+47:36.9 to the end of each span that holds it. Before genesis, 36 records are Q, in five short runs between T+41:00.0 and T+41:38.9 (2460.0-2498.9 s, just after the spin-up). None of them is tracked for 120 s, so they are not genesis. From T+41:40 to T+47:36, the records are N or U.
* **Longest tracked Q-or-U chain:** 455.6 s before 3312 s, and 596.2 s over the whole window.
* **Conditions before 3312 s:** E 12287, R 8100, C 12285, A 12287, Z 11965, W 11579 and S 8747 of 12287 records. The marginal counts are R 331, Z 4 and W 201. 149 records are flagged `rmw-at-inner-radius` and 2 `extent-limited`.
* **Inside the episode**, the RMW is mostly 100-170 m. 21 of its 4851 members have an RMW of 24 m or less (minimum 16 m = 4 dx at T+50:17.7). The frozen tracking carries the chain through a smaller core that lies inside the larger one. Identity needs the centre move to be at most the two RMWs added together.
* **The 10 m wind rises through the end of the headline span:** the per-minute maxima are 72 m/s at T+49, 93 at T+51, 114 at T+52, 128 at T+54 and 147 in T+55:00-56:00, most of that minute after 3312 s. The headline peak, 129.0 m/s, is the span's last record.
* **v1 beside, at window cadence:** NO_TORNADO in every span. The longest qualifying run is 4.6 s before 3312 s and 6.1 s after. v1's ζ condition held in 3943 of 12287 records before 3312 s. v1 reads only the child's centre corner and its 300 m disk; see observation 4.
* **v1 on the 120 s frames:** NO_TORNADO.
* **Run verdict:** TORNADO from the headline span, caption "simulated tornado (GM1 criterion v2, 4 m grid, 0.09375 s output)".
  * The 3312-3452.4 s span and the whole window are reported beside it. Their larger winds (to 163.3 m/s) come from records where the numerics were running away, and nothing may cite them as the run's intensity.
  * The window has no pressure in the scorer's read, which reads only `uadv`/`vadv`.

### Independent verification (2026-09-25)

An adversarial check with its own numpy code, which imports none of `v2rule.py`, `score.py` or
`window_segments.py` (`tools/criterion_v2/results/scoring-v2-pf4m/verify/`). It ran on CPUs only, under
`nice -n 19 ionice -c3`. No GPU was used. Outcome: **every verdict and headline number is confirmed.
Nothing was wrong, so nothing was re-scored.**

* **Frozen files.** `git diff 2b70e09f HEAD` is empty for `v2rule.py`, `score.py`, `FROZEN.json` and
  `docs/GM1-TORNADO-CRITERION-V2.md`. The SHA-256 of both code files equals `FROZEN.json`. Every verdict
  carries criterion hash `807b2cc8...`, `matches_frozen: true` and `protocol_conforming: true`.
* **Records read (L0 only).** The verifier rebuilt the trajectory from the two raw indexes: 13786 records,
  2160.09375-3452.4375 s, one cadence (0.09375 s), and L0 record numbers 0-13785 without gaps. No record
  is later than 3452.4375 s. Every row of the scorer's 3 views (12287, 1499 and 13786 rows) is the same
  row as the L0 trajectory's (time, SHA-256, arena, offset), and both arenas are L0's. `S3/frames-L1` is
  never referenced. `frames/nest-child-window` is exactly 12160 records long. The span edges fall at
  3311.90625 s (the last record before 3312.0 s) and 3312.0 s (the first record from it).
* **Layout, level and placement.** At 2400, 2880, 3000, 3240 and 3360 s, the window's `uadv`/`vadv` were
  compared with the child history frame's completed `u3d`/`v3d` at CM1 levels 1, 3 and 10. The RMS
  difference is 0.04-0.42 m/s, which is the gap between the acoustic-average and the completed wind. The
  best offset is the one the window's own `origin_x_m`/`origin_y_m` give. Where the frame and the window
  straddle a nest move (2400 and 2880 s), the fit is at the 25-cell (100 m) shift that the origins predict.
  Any other shift gives 7-12 m/s. The 10 m wind is CM1 level 3 exactly (scalar heights 2, 6, 10 m;
  param.F:8165-8172), so no vertical interpolation is involved.
* **Frame velocity.** Every child frame's own `domainlocx_m`/`domainlocy_m` divided by its model time is
  16.5 m/s. That is the model's accumulated domain motion (CM1 base.F:2663-2670 subtracts umove and vmove
  from the base state), and it agrees with the deck.
* **13 records recomputed from the raw bytes:** 2400.0, 2459.90625, 2460.0, 2856.84375, 2856.9375
  (genesis), 3212.8125, 3300.0, 3300.09375 (the arena seam and a 100 m origin jump), 3311.90625 (the
  headline peak), 3312.0, 3370.875, 3446.8125 (peak 163.3 m/s) and 3452.4375 s. Each record's SHA-256
  matches its index.
  * At the scorer's centre, the verifier's own profile gives the same results as the scorer to 4
    decimals on every record: Vmax, RMW, closure, wavenumber-2 share, the 10 m ground-relative maximum
    (cell-centred frame wind plus (16.5, 16.5), CM1 solve3.F:864-865 `sws` form), and the class.
  * The ground centre (local + origin + umove t) matches to 1e-4 m.
  * An exhaustive centre search (2 m lattice +-24 m, then 0.5 m) lands within 0.71 m (one diagonal lattice step) on 12 of 13 records,
    with the same class and Vmax within 0.08 m/s. At 2460.0 s it climbed to a second, stronger local
    maximum 35 m away (Vmax 23.3 m/s, RMW 10 m, class U), which the scorer keeps as a separate candidate.
    The row the scorer reports there is the Q candidate. This is not in the tornado chain.
* **Tracking rebuilt from `candidates.tsv`** with the verifier's own code (identity: ground move at most
  60 m/s x dt + 100 m, and the storm-following move at most RMW1 + RMW2; bridge 3 s; 120 s). It gives the
  scorer's result in each span:
  * before 3312 s: genesis 2856.9375 s, 454.97 s, 4851 members, largest gap 0.28 s, RMW median 128 m
    (5-95 % 76-172 m, min 16, max 200), Vmax peak 60.75 m/s at 3262.69 s, circulation median 30.4e3 and
    peak 58.0e3 m2/s at 3194.63 s, and 10 m peak 129.04 m/s at 3311.91 s;
  * 3312-3452.4 s: 140.44 s, 1499 members, 163.34 m/s at 3446.81 s, Vmax 81.37 m/s;
  * the whole window: 595.5 s, 6350 members.
  * The sample classes (Q 4889, U 3531, N 3867) match too.
* **Attack 2 (chain hopping).** No link in any chain needs the "sum of the RMWs" allowance beyond the
  larger core: the storm-following move of every link is within the larger of the two RMWs (largest move
  112 m). The chain's first member is a 22 m core (Vmax 20.33 m/s). The next member, 0.19 s later, is the
  184 m vortex. As a sensitivity case (not criterion v2), every candidate with an RMW of 24 m or less was
  removed. The verdict is unchanged: genesis moves one record, to 2857.03125 s, and the chain still runs
  to the span's end (454.9 s) with the same peak.
* **Attack 1 (is 129 m/s clean?).** CM1's Courant number was recomputed (`calccflquick`,
  misclibs.F:2726-2814, on `uadv`/`vadv`/`wadv` over the window, which is a lower bound on the whole child, child dt 0.03125 s, mh from the
  levels). It reproduces the status notes' diagnosis at 3300.0 s (0.70145).
  * It is at most 1 on every record from 3100 to 3311.90625 s and on every 4th record from 2160 to 3100 s
    (max 0.78). It first exceeds 1 at 3312.0 s (1.020).
  * The headline peak record, 3311.90625 s, has a Courant number of 0.985. Only the four records from
    3311.63 s on exceed 0.9.
  * So 129.0 m/s lies inside the span's stated limit, but at its edge. The span cut at 3300.0 s (Courant
    0.70) is still TORNADO: genesis 2856.9375 s, 443.1 s, with peak 127.6 m/s at 3268.125 s, where the
    Courant number is 0.86. Quote 127.6 m/s for a margin-safe peak.
* **Tests** at development commit `0a78d9d6`, on CPUs only: the criterion v2 rule, synthetic,
  scoring-tool and window-segment tests gave 69 passed and 26 skipped (the catalogue, which needs an
  opt-in). The catalogue with `GM1_CRITERION_CATALOGUE=1`: 26 passed (all 26 synthetic cases v2 was frozen with, through the adapter at 4 m).

### How it was run

* **Where and how.** CPU only, under `nice -n 19 ionice -c3` with `CUDA_VISIBLE_DEVICES=""`. The code was a clean clone of the development repository at `344e3899`. The driver is `tools/criterion_v2/results/scoring-v2-pf4m/run.sh`.
* **Processes.** One scorer at a time with 5 workers (the scorer, forkserver and workers made 7 processes), next to the 1-process catalogue test. The v1 passes ran as 3 single processes afterwards.
* **Wall time.** The headline span took 24.3 min, the later span 2.7 min, the whole window 26.0 min and the frames 18 s.
* **Reproduced.** A stopped earlier attempt had scored the same spans from `9cd35503` (the same adapter reader). Its `samples.tsv` and `candidates.tsv` are byte-identical to these for all four sources.
* **Collected** with `window_segments.py collect` at `d148c7ff`, which is bookkeeping only. `episodes_recovered: true` for all three window sources.
* **Files.** The window sources' `samples.tsv` and `candidates.tsv` are stored gzipped (`gzip -n`). Gunzip them before re-running `collect` on this directory.
* **To re-run** (this needs the run's raw window stream and child frames, which are not published; `--root run=DIR --root s3=DIR` say where they are):

```
nice -n 19 ionice -c3 python3 tools/criterion_v2/window_segments.py check --spec tools/criterion_v2/scoring-sources-v2-pf4m.json --verify-sha 8
nice -n 19 ionice -c3 python3 tools/criterion_v2/window_segments.py score --spec tools/criterion_v2/scoring-sources-v2-pf4m.json \
    --span E100R2-PF-4m-window-pre3312 --runs <dir> --workers 5      # and -3312on, -window
nice -n 19 ionice -c3 python3 tools/criterion_v2/score_runs.py --manifest tools/criterion_v2/scoring-sources-v2-pf4m.json \
    --out <dir> --only E100R2-PF-4m-child --jobs 1
nice -n 19 ionice -c3 python3 tools/criterion_v2/window_segments.py v1-window --spec ... --span <label> --runs <dir>
python3 tools/criterion_v2/window_segments.py collect --spec ... --runs <dir> --json <dir>/tornado-verdicts-v2-pf4m.json --markdown <dir>/verdict-table.md
```

## Not scored yet, and how to score them

The same command scores whatever has appeared since. It skips sources that already have a verdict and
writes a `.not-scored.json` for any that is still missing. It needs the development runs' output, which is
not published, so it is here as a record and cannot be run from a clone.

```
nice -n 19 ionice -c3 python3 tools/criterion_v2/score_runs.py --out <dir> --jobs 1 \
    [--root viz=<dir holding E100V/, E100W/ ...>] [--only E50N2-child E50N2-window]
python3 tools/criterion_v2/verdict_table.py --runs <dir> --json <dir>/tornado-verdicts-v2.json \
    --markdown <dir>/verdict-table.md
```

* **E50N2 (2 m)** child frames and window, once its frames exist. The manifest takes the spawn time from the run's `receipt.json` (`restart.resumed_model_time_s`, expected 120 s) and the parent spacing from `decks/E50N2.toml` (50 m).
* **E100T window**: the stream is empty (0 records).

## How it was run

* `tools/criterion_v2/score_runs.py` runs the frozen `score.py` once per source in `tools/criterion_v2/scoring-sources-v2.json`. The manifest holds the labels, paths, spawn times and where each spawn time came from.
  * Every sample is read (no `--every`, `--end`), with `--workers 1`.
  * Children and windows get `--nest-start-s` and `--parent-dx`. Where a receipt exists, the spawn time is checked against its `restart.resumed_model_time_s`: E100N 3600 s, E100L-500t 3900 s, E100M 4200 s.
  * E100V and E100W use 4200 s, from the pre-registered table. E100T uses 4200 s from its live sidecar's first sample (4200.375 s = parent step 11201); its deck was not on the scoring machine.
  * E100V's parent uses `--start 4200`, as pre-registered.
* The E100V window was launched directly, with the same arguments, before the driver existed (`E100V-window.run.json`).
* **The E100W window was scored on the machine that holds its 121 GB stream.** It was scored after the other sources, and the result is reported as it came out.
  * The scorer checkout was a clean shallow clone of `0223670`. `v2rule.py` and `score.py` were checked against `FROZEN.json` before the run, and the verdict has `matches_frozen: true`, `protocol_conforming: true` and `scorer_commit` `0223670…`.
  * The command was `score_runs.py --root viz=RUNS --only E100W-window --jobs 1`, which gives the same pre-registered arguments (`--nest-start-s 4200 --parent-dx 100`, `--workers 1`, every record).
  * It ran as one scorer process, CPU only, under `nice -n 19 ionice -c3`, using about 75 MB. It took 1807 s (30.1 min) for 1201 records. It read only two levels of `uadv`/`vadv` per record, a few GB of the 121 GB stream. The GPU was not used.
  * `verdict_table.py` collected E100W's two sources there (a copy of the manifest holding only them). Its window record and E100W's `run_verdict` replaced the "not scored" entries in `tornado-verdicts-v2.json` (see `updates` there), and its row replaced the one in the tables. No other run's record changed.
  * The scratch directory there was deleted afterwards. Only the results in `tools/criterion_v2/results/scoring-v2/` were kept.
* `tools/criterion_v2/verdict_table.py` collects the verdicts. It recovers each TORNADO episode's members with the frozen `v2rule._chains` on the written candidates, to give the circulation, and checks them against the episode `verdict.json` reports (`episodes_recovered: true`). It decides nothing.
