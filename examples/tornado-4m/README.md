# The 4 m tornado run, as decks

This folder holds the decks of the run that made GM1's 4 m tornado, and two variants
you can try on your own card. It is in the repository only, not in the pip package:
clone the repository to use it. The step by step guide is
[docs/NESTING.md](../../docs/NESTING.md). Read that first if you have not run a nest before.

| Files | What they are | Sounding | Card |
|---|---|---|---|
| `S1.toml`, `P.toml`, `S3.toml` | the original run: 100 m parent, 4 m nest | the original one, which is not distributed (see below) | 32 GB |
| `norman/` | the same grids fitted to a sounding GM1 ships | Norman, Oklahoma, 1800 UTC 24 May 2011 | 32 GB |
| `small/` | a smaller parent with a 20 m nest | the same Norman sounding | 16 GB or 24 GB |

## What the original run was

Run E100R2-PF-4m is an idealized supercell: one warm bubble in air that is the same
everywhere, taken from a sounding for the 24 May 2011 El Reno, Oklahoma storm. It is not
a reconstruction of that day's storms. It ran in three stages on one RTX 5090 (32 GB):

1. `S1.toml`: the 100 m parent alone, 640 x 768 x 80 cells (64 x 76.8 km, 20 m layers near
   the ground), from 0 to 6600 s, with a restart every 5 minutes.
2. `P.toml`: the parent alone again, continued from the restart at 2100 s to 2700 s with a
   frame every 60 s, to find a moment when the rotation near the ground was steady.
3. `S3.toml`: the parent continued from the restart at 2160 s with a 4 m child grid
   (500 x 500 x 103 cells, 2 x 2 km, lowest level 2 m above the ground) that follows the
   vortex and feeds back into the parent.

Before stage 2 a 20 m nest was run from the same restart to see where the vortex was.
That stage is not needed to repeat the recipe and its deck is not here; `small/S3.toml`
has the same kind of nest.

## What it found

The 4 m stage was scored TORNADO by GM1's frozen tornado criterion (version 2,
[docs/GM1-TORNADO-CRITERION-V2.md](../../docs/GM1-TORNADO-CRITERION-V2.md)), using the
records before 3312 s:

- the tornado formed at T+47:36.9 (2856.9 s);
- one vortex was tracked for 455 s;
- the peak 10 m ground-relative wind was about 128 m/s (129.0 m/s in the last record
  before the time step limit was passed);
- the median radius of maximum wind was 128 m.

What you must know before quoting any of that:

- **The run did not finish.** At 3312 s the 4 m child's Courant number passed 1 (the wind
  crossed more than one 4 m cell in one 0.03125 s step, which happens at 128 m/s), and the
  run stopped at 3452.8 s. Winds from after 3312 s, up to 163 m/s in the saved records,
  come from numerics that were failing. They must not be cited.
- **It was scored after the fact.** The run was not registered before it was made.
- **It ran on a build that predates GM1 0.1.0**, on one 32 GB card. The nest code changed
  before the release. A re-run on GM1 0.1.x will be a different trajectory, and the nest
  may have to be placed somewhere else at some other time.
- **CM1 has no nesting**, so the nested stage cannot be compared with CM1. It is GM1's own
  extension.
- **You cannot yet re-derive the headline from public data.** The verdict was scored from
  the run's own window stream (13,786 records, one every 0.09375 s, 1.6 TB), which is not
  public. The four public datasets hold CM1-style netCDF files, and most of their records
  lie after 3312 s. One of them, the window once a second, has 162 records from before
  3312 s: converted with `tools/criterion_v2/netcdf_to_window.py` and scored, they give
  TORNADO for the last 161 s before the limit, which is not the genesis time, the 455 s or
  the 129.0 m/s record.
  [docs/GM1-TORNADO-VERDICTS-V2.md](../../docs/GM1-TORNADO-VERDICTS-V2.md) has the
  commands, every scored sample, and what cannot be checked yet.

The public data, all CC-BY-4.0:

| Dataset | What it holds | Before or after 3312 s |
|---|---|---|
| <https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-full-nest> | the whole 4 m nest at its 10 saved times, 2280 to 3360 s | 9 before, 1 after |
| <https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-last-5min-every-1s> | the window once a second, 3150 to 3450 s | 162 records before, 139 after |
| <https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-last-2min-every-record> | every window record, 3330 to 3452.4 s | all after |
| <https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-top10-frames> | ten chosen window records, 3314 to 3452 s | all after |

Only records from before 3312 s back the numbers above.

## The sounding

The decks read their profile from the file you pass with `--sounding`.

**The original file is not in this repository.** Where it came from has not been
confirmed well enough to redistribute it. What can be said about it:

- sha256 `eabada6c37b805bc05fd3be39fe2eadca048d86bdc7a3882971d33634d78de05`, 5142 bytes,
  165 lines;
- it is in CM1's `input_sounding` format. Line 1 is the surface: pressure 947.62 hPa,
  potential temperature 304.86 K, vapour mixing ratio 19.22 g/kg. The other 164 lines
  are height above ground (m), potential temperature (K), vapour mixing ratio (g/kg),
  and the west-east and south-north wind (m/s), from 10 m to 20000 m. The lowest levels
  are 100 m apart (10, 110, 210 m and so on), so it is a model-style profile, not a raw
  balloon record;
- with the parent deck, `tools/report_sounding_parameters.py` reports mixed-layer CAPE
  4422 J/kg, CIN 11 J/kg, 0-1 km shear 17.4 m/s, 0-6 km shear 27.3 m/s, 0-1 km
  storm-relative helicity 390 m2/s2 and 0-3 km 589 m2/s2.

If you have that file, `S1.toml`, `P.toml` and `S3.toml` run as they are.

**What works without it.** GM1 ships the Norman, Oklahoma balloon of 1800 UTC 24 May 2011
(`src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding`, sha256
`d72f8ee9189ca0eaefcb3e3cfc332a7ddfb844a346adf9027baf3b9e5c5acd56`), the profile of the
`oun-2011-05-24` preset. The decks in `norman/` and `small/` use it. It is a different
environment: a midday balloon some distance from the storm, with mixed-layer CAPE
3136 J/kg, CIN 51 J/kg, 0-1 km shear 11.7 m/s, 0-6 km shear 23.9 m/s, 0-1 km
storm-relative helicity 167 m2/s2 and 0-3 km 195 m2/s2 on the same grid. That is less than
half the low-level helicity of the original. `norman/` has not been run on a GPU. The
first 25 minutes of `small/` have (see "Run on a GPU with the released package" below),
which is before that storm rotates near the ground. Whether either makes a tornado is
not known.

A moving frame and a bubble position belong to one sounding. If you pass a different
profile to a deck, the run-geometry check refuses it by name
(`CM1-CARTESIAN-RUN-GEOMETRY`) and prints the frame and the bubble to use.

## Checked with the released package

Every deck here was run through `gpuwm-cm1 check` and
`gpuwm-cm1 cartesian DECK --plan --integrate` with GM1 0.1.0 installed from PyPI into a new
environment, on the CPU, and again with the GM1 0.1.3 wheel. All of them pass both, and
the two versions give the same plans. What the plans say:

| Deck | Grid | Model time | Parent steps | Frames | Restarts | Device memory the plan lists |
|---|---|---|---:|---:|---:|---|
| `S1.toml` | 640 x 768 x 80 at 100 m | 0 to 6600 s | 17,600 | 22 | 22 | 14.12 GiB |
| `P.toml` | the same | 2100 to 2700 s | 1,600 | 10 | 4 | 14.12 GiB |
| `S3.toml` | the same, and a 500 x 500 x 103 child at 4 m | 2160 to 3300 s | 3,040 (12 child steps each) | 19 | 4 | 14.12 + 9.24 = 23.36 GiB |
| `norman/S1.toml` | 640 x 768 x 80 at 100 m | 0 to 6600 s | 17,600 | 22 | 22 | 14.12 GiB |
| `norman/P.toml` | the same | 2100 to 2700 s | 1,600 | 10 | 4 | 14.12 GiB |
| `norman/S3.toml` | the same, and a 500 x 500 x 103 child at 4 m | 2160 to 3300 s | 3,040 (12 child steps each) | 19 | 4 | 14.12 + 9.24 = 23.36 GiB |
| `small/S1.toml` | 448 x 448 x 80 at 100 m | 0 to 3600 s | 9,600 | 12 | 12 | 5.78 GiB |
| `small/P.toml` | the same | 2100 to 2700 s | 1,600 | 10 | 4 | 5.78 GiB |
| `small/S3.toml` | the same, and a 300 x 300 x 80 child at 20 m | 2160 to 3000 s | 2,240 (3 child steps each) | 14 | 3 | 5.78 + 2.60 = 8.38 GiB |

Read the table with these in mind:

- **The listed memory is not the peak.** The plan lists the model's own arrays. The card
  also holds the CUDA context, work space and a memory pool. In the original run the
  parent peaked at 18,482 MiB against 14.12 GiB listed, and the 4 m stage at about
  28,000 MiB against 23.36 GiB listed: about 4 GiB more both times. On an RTX 4090 the
  `small/` parent held 8,464 MiB against 5.78 GiB listed, and the nested stage with the
  wider child 16,800 MiB against 12.97 GiB listed: 2.5 and 3.4 GiB more. Allow 4 GiB. It
  is why the 4 m stage needs a 32 GB card, and why `small/S3.toml` (8.38 GiB listed) is
  sized for 16 GB.
- **On a 24 GB card** the small nest can be wider. With `parent_span_x` and
  `parent_span_y` set to 100 (a 10 x 10 km child of 500 x 500 cells), and `parent_i0` and
  `parent_j0` moved to keep it inside the parent, the plan lists 5.78 + 7.19 = 12.97 GiB.
  Nothing else in the deck has to change.
- **The decks that continue a run** (`P`, `S3`) need a restart to plan against. For this
  table all of them, the `norman/` and `small/` ones included, were planned against the
  original run's restarts, from which the plan takes only the model time. In `norman/`
  and `small/` the `rstnum`, the placement and the times are placeholders, marked
  `PLACEHOLDER` in the files: replace them with your own run's.
- **`gpuwm-cm1 estimate` prices the parent only.** It says so
  ("Nests are not priced by this estimator"). For `small/S1.toml` on an RTX 5090 it
  estimates about 8.7 GB of device memory and about 49 minutes. On an RTX 4090 it said
  about 1.7 hours. The measured 0.40 s a step there puts the hour at about 64 minutes.
- **No configuration smaller than a 32 GB card has produced a tornado.** `small/` is for
  learning the steps and looking at a storm at 20 m.

## Run on a GPU with the released package

One run so far: the first 25 minutes of `small/`, run on GM1 0.1.0 on an RTX 4090 (24 GB),
installed by pip, with two other small jobs on the card. It has not been repeated on a
later release; 0.1.2 moved the engine to ArWen 2.8.9.

| Stage | Storm time | Parent steps | Seconds a step | Wall time |
|---|---|---:|---:|---:|
| `small/S1.toml`, shortened to end at 1200 s | 0 to 1200 s | 3200 | 0.399 | 1307 s |
| `small/P.toml` with `rstnum = 4`, to 1440 s | 1200 to 1440 s | 640 | 0.401 | 291 s |
| `small/S3.toml` with `rstnum = 5` and a 100-cell span (a 500 x 500 x 80 child at 20 m) | 1260 to 1500 s | 640, 3 child steps each | 1.997 | 1335 s |

- All three ended with status 0. The child had no value that was not finite in its four
  frames or its 480 window records. Its tracker moved it at all 33 looks, with no hold
  and no relocation.
- It wrote 4 parent frames and 4 child frames with a placement file each, a window of
  480 records (6.82 GB), one nested restart, the live check's 640 samples and the
  receipt. `gpuwm-cm1 quicklook`, the film, `gpuwm-cm1 cm1out` and the tools of the
  nesting guide all ran on those files.
- **It did not follow the storm.** At 21 minutes the rotation near the ground was 0.003
  per second and `tools/nest_place.py` proposed no start, so the child was placed by
  hand. The tracker took a weak eddy on the outflow and moved the child 2.3 km in four
  minutes. At 1500 s the parent's updraft (27.7 m/s) was outside the child. The live
  tornado check said NO_TORNADO. A nest that follows a real vortex has not yet been
  shown on GM1 0.1.x.
- Not run: `small/` past 24 minutes of storm time, the 300 x 300 child, `norman/`, and
  the original decks.

## Disk

Sizes measured in the original run, with `--history-envelope off` (the default writes a
second copy of every frame, which doubles the frame sizes):

| File | Size |
|---|---:|
| one parent frame (640 x 768 x 80) | 3.4 GB |
| one parent-only restart | 11.0 GB |
| one 4 m child frame | 2.2 GB |
| one restart of the nested stage (parent and child) | 18.1 GB |
| one window record (227 x 227 x 45 cells) | 137.6 MB |

So `S1.toml` writes 74 GB of frames and 241 GB of restarts if you keep them all. Each
`restart-NNNNNN` folder stands alone, and you only need the one you continue from: delete
the others when the stage is done. `P.toml` writes 78 GB. `S3.toml` writes about 490 GB:
105 GB of frames, 73 GB of restarts and a 313.7 GB window (2280 records, one every 0.5 s).

For `small/`, measured on the RTX 4090 run: 1.3 GB per parent frame, 4.2 GB per parent
restart, 1.6 GB per frame of the 500 x 500 child, 9.4 GB per nested restart with that
child, and a window of 1.70 GB per model minute. The three stages as the decks are
written come to about 190 GB with every restart kept (66, 30 and 93 GB), and one child
netCDF file from `gpuwm-cm1 cm1out` is 2.09 GB.

## What was changed in `S3.toml`

`S3.toml` is not the deck exactly as it ran. As it ran, it asked for 4200 s and a window
record every 0.09375 s until 3960 s. That is the configuration that stopped at 3452.8 s
after writing 1.6 TB. The deck here differs in three places:

| Key | As run | Here | Why |
|---|---|---|---|
| `run.duration_s` | 4200 | 3300 | ends inside the span that stayed clean |
| `run.history_interval_s` | 120 | 60 | 3300 s is not a whole number of 120 s frames |
| window `stride_child_steps`, `end_s`, `budget_bytes_per_model_minute` | 3, 3960, 90e9 | 16, 3300, 17e9 | one record every 0.5 s, 313.7 GB in place of 1.6 TB |

The `[nest]` table, the placement (`parent_i0 = 295`, `parent_j0 = 304`) and the start at
2160 s are as run. `S1.toml` and `P.toml` are as run apart from their comments.

**Ending at 3300 s does not make your run safe.** Your trajectory will differ, and your
vortex may reach 128 m/s sooner or not at all. If the child's wind nears 128 m/s, the
remedy is a shorter child step: `time_refinement_ratio = 16` gives 0.0234375 s, which
moves the limit to 171 m/s. With `stride_child_steps = 21` (a record every 0.49 s, inside
the same budget) that deck plans at the same 23.36 GiB. It costs a third more child steps,
and it has never been run to the end. GM1 cannot change the child's step when it
resumes a nested restart (`CM1-CARTESIAN-NEST-RESTART-MISMATCH`), so the change means
starting the nested stage again from the parent restart.

## Running it

From the root of a clone, with GM1 installed. The `python3 tools/...` lines have to be
run by the Python GM1 is installed in: activate the pip environment first, or, if you
used the installer, type `"$(uv tool dir)/gpuwm-cm1/bin/python"` in place of `python3`.
The first block is the original run and
needs the original sounding (here `~/soundings/elreno.input_sounding`, a name for the
example). Check and plan before every stage: add `--plan` to the same command, which
prices it and opens no GPU.

```sh example
mkdir -p ~/gm1-runs/tornado
gpuwm-cm1 cartesian examples/tornado-4m/S1.toml --integrate \
    --sounding ~/soundings/elreno.input_sounding --history-envelope off \
    --output ~/gm1-runs/tornado/S1-receipt.json \
    --history-dir ~/gm1-runs/tornado/S1-frames --restart-dir ~/gm1-runs/tornado/S1-restarts
gpuwm-cm1 cartesian examples/tornado-4m/P.toml --integrate \
    --sounding ~/soundings/elreno.input_sounding --history-envelope off \
    --restart-from ~/gm1-runs/tornado/S1-restarts \
    --output ~/gm1-runs/tornado/P-receipt.json \
    --history-dir ~/gm1-runs/tornado/P-frames --restart-dir ~/gm1-runs/tornado/P-restarts
python3 tools/nest_place.py ~/gm1-runs/tornado/P-frames --restarts ~/gm1-runs/tornado/P-restarts
gpuwm-cm1 cartesian examples/tornado-4m/S3.toml --integrate \
    --sounding ~/soundings/elreno.input_sounding --history-envelope off \
    --restart-from ~/gm1-runs/tornado/P-restarts \
    --output ~/gm1-runs/tornado/S3-receipt.json \
    --history-dir ~/gm1-runs/tornado/S3-frames --restart-dir ~/gm1-runs/tornado/S3-restarts
```

`tools/nest_place.py` prints the `rstnum`, `parent_i0`, `parent_j0` and start time for your
run. Put them into a copy of `S3.toml` before the last command. Run on the original run's
stage 2 frames, it prints the values that run used (restart 8, 2160 s, 295 and 304).

With the shipped sounding, nothing else is needed. This checks and prices the first stage
of both variants on the CPU:

```sh
mkdir -p ~/gm1-runs/tornado
gpuwm-cm1 check examples/tornado-4m/norman/S1.toml
gpuwm-cm1 check examples/tornado-4m/small/S1.toml
gpuwm-cm1 cartesian examples/tornado-4m/small/S1.toml --plan --integrate \
    --sounding src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding \
    --output ~/gm1-runs/tornado/small-S1-receipt.json \
    --history-dir ~/gm1-runs/tornado/small-S1-frames \
    --restart-dir ~/gm1-runs/tornado/small-S1-restarts > ~/gm1-runs/tornado/small-S1-plan.json
```

[docs/NESTING.md](../../docs/NESTING.md) walks through the `small/` variant from there to a
picture of the nest.
