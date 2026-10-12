# Nesting: a finer grid that follows the storm

The storms you start from the GUI run on one grid. A tornado is about 100 m wide, and
to show one a grid needs cells a few metres across, which no card holds over a whole
storm. A **nest** gets you there without paying for fine cells everywhere: a second, much finer grid (the *child*) runs inside
the first (the *parent*) on the same card at the same time, only where the vortex is.

This page explains what the nest is, what card it needs, and how to run one from the
command line, with commands you can copy. It ends with every `[nest]` setting, how to
look at the result, and what to do when a run stops.

## What a nest is in GM1

- **Two grids, one run.** The parent is an ordinary storm run, for example 100 m. The
  child covers a small square of it, for example 2 km at 4 m. After every parent step the
  child takes several short steps of its own.
- **Two-way.** The child takes the values on its edges from the parent, and gives its
  interior back to the parent. The parent's storm feels what the child resolved.
- **It follows the storm.** The child finds the rotation near the ground every few
  seconds and moves, one parent cell at a time, to keep it in the middle.
- **Its own levels.** The child can have thinner layers near the ground than the parent.
- **Odd ratios only.** The child's cells are the parent's divided by an odd whole number,
  and the result has to be a length the model can store exactly. From a 100 m parent
  that leaves two choices: **20 m** (ratio 5) and **4 m** (ratio 25).

**This is not CM1.** CM1 has no nesting of any kind. The nest is GM1's own extension,
built from the moving nest of the WRF model and standard boundary relaxation. Everything
GM1 says about matching CM1 is about single-grid runs. A nested run has no CM1 run to be
compared with, and nothing here claims one.

## What has been done with it

One run with a 4 m nest inside a 100 m parent produced a tornado: a vortex tracked for
455 s with a peak 10 m wind of about 128 m/s. It ran on one 32 GB card, on a build that
predates GM1 0.1.0, and it stopped early when the 4 m grid's time step became too long
for its winds. [examples/tornado-4m/README.md](../examples/tornado-4m/README.md) has the
decks, the numbers and every caveat. The verdict, the rule that gave it and what a reader
can check are in [GM1-TORNADO-VERDICTS-V2.md](GM1-TORNADO-VERDICTS-V2.md).

GM1 0.1.x has not repeated that run. What has run on a release is the start of this
page's own example: the `small/` decks, run on GM1 0.1.0 on an RTX 4090 (24 GB), installed
by pip. It has not been repeated on a later release; 0.1.2 moved the engine to ArWen 2.8.9.

| Stage | Storm time | Parent steps | Seconds a step | Wall time | The card held (the plan listed) |
|---|---|---:|---:|---:|---|
| Parent alone (`S1`, shortened to end at 1200 s) | 0 to 1200 s | 3200 | 0.399 | 1307 s | 8.27 GiB (5.78) |
| Parent continued (`P`, from restart 4) | 1200 to 1440 s | 640 | 0.401 | 291 s | not sampled |
| Nest (`S3`, from restart 5, a 500 x 500 x 80 child at 20 m) | 1260 to 1500 s | 640, 3 child steps each | 1.997 | 1335 s | 16.41 GiB (12.97) |

Two other small jobs shared the card, so the times are upper figures. Every stage ended
with status 0. The child stayed finite in all four of its frames and all 480 window
records. Its tracker moved it at each of its 33 looks, with no hold. The run wrote the
child's frames, the window (a record every 0.5 s), a nested restart and the receipt.

What that run did not show is a nest following a storm. There was time for 25 minutes of
storm, which is before this storm rotates near the ground: the rotation was 0.003 per
second, `nest_place.py` proposed no start, and the child was placed by hand. It followed
a weak eddy on the outflow and moved 2.3 km in four minutes, away from the storm. The
live tornado check said NO_TORNADO, as it should with no vortex. Whether this example
ever makes tight rotation with this sounding is not known: its parent has not been run
past 24 minutes of storm time.

Every deck on this page was also checked and priced on the CPU with GM1 0.1.0, and again
with GM1 0.1.3, which gives the same plans; the tools were run on that run's files and on the original run's. No nested run on a card
smaller than 32 GB has made a tornado.

## What you need

- **An NVIDIA card.** The example for this page fits 16 GB. The 4 m run needs 32 GB.

  | Run | Parent | Child | Memory the plan lists | Card |
  |---|---|---|---|---|
  | `examples/tornado-4m/small/` | 448 x 448 x 80 at 100 m | 300 x 300 x 80 at 20 m | 8.38 GiB | 16 GB |
  | the same with a 100-cell span | 448 x 448 x 80 at 100 m | 500 x 500 x 80 at 20 m | 12.97 GiB | 24 GB |
  | `examples/tornado-4m/S3.toml` | 640 x 768 x 80 at 100 m | 500 x 500 x 103 at 4 m | 23.36 GiB | 32 GB |

  The card holds more than the plan lists: 2.5 to 4 GiB more in the runs where it was
  measured. The second row ran on an RTX 4090 and held 16.4 GiB. The first row's
  300 x 300 child has not been run.
- **Disk.** More than the word "small" suggests. Measured on the RTX 4090 run: a parent
  frame of the small example is 1.3 GB, a parent restart 4.2 GB, a frame of the
  500 x 500 child 1.6 GB, a restart of the nested stage 9.4 GB, and the window 1.70 GB
  for each minute of storm time.

  | Stage of `small/`, as the decks are written | Frames | Restarts | Window | In all |
  |---|---:|---:|---:|---:|
  | 1, the parent for an hour | 12 x 1.3 GB | 12 x 4.2 GB | | 66 GB |
  | 2, ten more minutes | 10 x 1.3 GB | 4 x 4.2 GB | | 30 GB |
  | 3, the nest for 14 minutes (500 x 500 child) | 14 x 1.3 GB and 14 x 1.6 GB | 3 x 9.4 GB | 24 GB | 93 GB |

  That is about 190 GB if you keep everything. Each `restart-NNNNNN` folder stands alone
  and you need only the one the next stage continues from: delete the others when a
  stage is done and stage 1 drops to 20 GB. The 4 m run is larger again: 315 GB for its
  stage 1, 78 GB for stage 2 and about 490 GB for the nested stage
  ([the example page](../examples/tornado-4m/README.md) has the sizes).
- **Time.** On the RTX 4090 a parent step of the small example took 0.40 s and a nested
  step with the 500 x 500 child 2.0 s, with two other small jobs on the card. At that
  pace stage 1 takes about 64 minutes, stage 2 about 11, and the 14 minutes of the nested
  stage about 75. (`gpuwm-cm1 estimate` said 1.7 hours for stage 1 on that card: it was
  pessimistic.)
- **GM1 installed, and a clone of the repository beside it.** The example decks and the
  small tools this page uses are in the repository, not in the pip package:

  ```sh example
  git clone https://github.com/FahrenheitResearch/gm1
  cd gm1
  ```

  Every command below is typed in that folder.
- **One Python for the tools.** A line that starts `python3 tools/...` has to be run by
  the Python that GM1 is installed in.

  - If you installed with pip into an environment, activate it first
    (`. ~/gm1/bin/activate`): `python3` is then the right one.
  - If you used the installer, GM1 has a Python of its own and your system's `python3`
    does not know it. Type `"$(uv tool dir)/gpuwm-cm1/bin/python"` wherever a line here
    says `python3`. (The installer put `uv` in `~/.local/bin`.)

  With another Python, `tools/report_sounding_parameters.py` stops with
  `CM1-CARTESIAN-INITIALIZER-MISSING`, because it uses a compiled part of the installed
  package. The other tools then fall back on the clone's own source and need numpy.
- **The command line.** The GUI, the presets and `gpuwm-cm1 run` do not know about
  nests. A nested run is a *deck* (a text file of settings) given to
  `gpuwm-cm1 cartesian`. It prints its progress in the terminal and has no stop button.
  To be able to stop it cleanly, add `--control-dir DIR` to the run command: a file named
  `stop` created in DIR ends the run at the next step and still writes the receipt, and
  one named `pause` ends it right after the next restart. `gpuwm-cm1 estimate` prices
  only the parent.

## The recipe

A nest is expensive, so you do not run it from the start. You run the cheap parent
alone, find the moment the storm starts to rotate near the ground, and start the nest
there:

1. run the parent with restarts;
2. find the rotation;
3. continue the parent for a few minutes with frequent frames;
4. place the nest;
5. continue once more, this time with a `[nest]` table.

The commands use the `small/` example and the sounding GM1 ships for it (the Norman,
Oklahoma balloon of 1800 UTC 24 May 2011). They write to `~/gm1-runs/nest`, one folder
for each stage.

### 1. Check and price the parent (CPU, a few seconds)

```sh
mkdir -p ~/gm1-runs/nest/S1 ~/gm1-runs/nest/P ~/gm1-runs/nest/S3
gpuwm-cm1 check examples/tornado-4m/small/S1.toml
gpuwm-cm1 cartesian examples/tornado-4m/small/S1.toml --plan --integrate \
    --sounding src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding \
    --output ~/gm1-runs/nest/S1/receipt.json \
    --history-dir ~/gm1-runs/nest/S1/frames \
    --restart-dir ~/gm1-runs/nest/S1/restarts > ~/gm1-runs/nest/S1/plan.json
python3 tools/plan_summary.py ~/gm1-runs/nest/S1/plan.json
```

`check` reads the deck and refuses anything it does not understand. When it accepts the
deck it prints it back as about 300 lines of JSON that end with
`"status": "accepted-grammar-and-first-slice"`. That is the answer "yes", not an error
dump. `--plan` prices the run and opens no GPU. It writes nothing but the answer it
prints, so the same folders can be named again for the real run. `plan_summary.py` prints the lines that matter:

```text
status        priced-no-device-opened
model time    0 to 3600 s: 9600 parent steps, 12 frames, 12 restarts
parent        448 x 448 x 80 cells, 5.78 GiB listed
device memory 5.78 GiB listed; allow about 4 GiB more on the card (about 10 GiB)
storm track   tracked-member-stays-inside
```

To see the air the storm will grow in (CAPE, shear, helicity), and what the parent will
cost in time and disk on your card:

```sh
python3 tools/report_sounding_parameters.py examples/tornado-4m/small/S1.toml \
    --sounding src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding
gpuwm-cm1 estimate examples/tornado-4m/small/S1.toml \
    --sounding src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding
```

The sounding report gives an equilibrium level ("EL") of about 1370 m beside a CAPE of
more than 3000 J/kg. Both are right. As in CM1, the EL it prints is the first height
above the level of free convection where the lifted air is no longer warmer than the air
around it, and this balloon has a warm, dry layer (a cap) from 1.3 to 1.4 km. The CAPE
is counted through the whole depth of the storm above it.

### 2. Run the parent (GPU)

The same command without `--plan`. `--history-envelope off` halves the size of every
frame.

```sh example
gpuwm-cm1 cartesian examples/tornado-4m/small/S1.toml --integrate --history-envelope off \
    --sounding src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding \
    --output ~/gm1-runs/nest/S1/receipt.json \
    --history-dir ~/gm1-runs/nest/S1/frames \
    --restart-dir ~/gm1-runs/nest/S1/restarts
```

It writes a frame and a restart every 5 minutes of storm time: `S1/frames/frame-000001`
and `S1/restarts/restart-000001` are at 300 s, number 7 is at 2100 s, and so on.

### 3. Find the rotation

Draw the pictures, then ask where the rotation near the ground is tight:

```sh example
gpuwm-cm1 quicklook ~/gm1-runs/nest/S1/frames --deck examples/tornado-4m/small/S1.toml
python3 tools/nest_place.py ~/gm1-runs/nest/S1/frames --restarts ~/gm1-runs/nest/S1/restarts
```

The pictures go into a `quicklook` folder *beside* the frames folder, not inside it:
here `~/gm1-runs/nest/S1/quicklook/`. "Spin near the ground" is the one to look at. That
is why every stage has a folder of its own in this recipe. Frames folders that sit side
by side in one folder share one `quicklook` folder, and drawing the second stage removes
the pictures of the first. `--out DIR` sends the pictures anywhere you like.

`nest_place.py` prints one line per frame and the times with tight rotation. When it
finds none it says so and ends with status 1, which stops a script that runs under
`set -e`. Frames 5 minutes apart are enough to see *when*, not enough to say *where*:
the next step fixes that.

If no frame shows tight rotation, this storm did not make a vortex near the ground on
this grid. That is a real result. Try another sounding (the run-geometry check will tell
you the frame and bubble to use with it), or a longer run.

### 4. Continue the parent with a frame every minute

Copy the second deck:

```sh example
cp examples/tornado-4m/small/P.toml ~/gm1-runs/nest/P.toml
```

**Edit the copy before you run it.** Two numbers in its `[run]` table are marked
`PLACEHOLDER`: `rstnum`, the restart to continue from (one or two restarts before the
first tight time), and `duration_s`, the time to stop at (ten minutes later, a multiple
of 180 s). As shipped they continue from restart 7 (2100 s) and stop at 2700 s. Then
check it, price it and run it:

```sh example
gpuwm-cm1 check ~/gm1-runs/nest/P.toml
gpuwm-cm1 cartesian ~/gm1-runs/nest/P.toml --plan --integrate \
    --sounding src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding \
    --restart-from ~/gm1-runs/nest/S1/restarts \
    --output ~/gm1-runs/nest/P/receipt.json \
    --history-dir ~/gm1-runs/nest/P/frames \
    --restart-dir ~/gm1-runs/nest/P/restarts > ~/gm1-runs/nest/P/plan.json
python3 tools/plan_summary.py ~/gm1-runs/nest/P/plan.json
gpuwm-cm1 cartesian ~/gm1-runs/nest/P.toml --integrate --history-envelope off \
    --sounding src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding \
    --restart-from ~/gm1-runs/nest/S1/restarts \
    --output ~/gm1-runs/nest/P/receipt.json \
    --history-dir ~/gm1-runs/nest/P/frames \
    --restart-dir ~/gm1-runs/nest/P/restarts
```

`--restart-from` names the folder the earlier stage wrote its restarts into. This stage
writes a frame at every multiple of 60 s of storm time and a restart at every multiple
of 180 s, numbered on from the one it started at. So the first restart can come sooner
than 180 s after the start: continued from restart 4 (1200 s), the RTX 4090 run wrote
restart 5 at 1260 s and restart 6 at 1440 s. `nest_place.py` prints the time of each
restart. Trust that, not the number.

### 5. Place the nest

```sh example
python3 tools/nest_place.py ~/gm1-runs/nest/P/frames --restarts ~/gm1-runs/nest/P/restarts --span 60
```

`--span` is the width of the child in parent cells: 60 for the 16 GB child, 100 if you
will run the wider child on a 24 GB card.

It ends with the lines to put in the nested deck, like these (from the original 4 m
run's own frames, with `--span 20`):

```text
Proposed start: t = 2160 s, 300 m box vorticity 0.139 /s at (30550, 31450) m.
Put this in the nested deck:

  [run]
  rstnum = 8
  [nest]
  parent_i0 = 295
  parent_j0 = 304
  parent_span_x = 20
  parent_span_y = 20
  [lagrangian.nest_child_window]   (if the deck has one)
  start_s = 2160.0
```

"Placing the nest" below says how it decides. It proposes; you decide. Look at the
frames it names before you spend hours on its answer.

**Do not start the nest until the tool proposes a start.** The tracker always follows
the strongest rotation it finds at the child's lowest level. Where there is no vortex
yet, that is some weak eddy on the storm's outflow, and the child goes with it.
In the RTX 4090 run the nest was started at 21 minutes, on rotation of only 0.003 per
second, where the tool had proposed nothing. The tracker's first look put the maximum
1.5 km from the point chosen by hand, the child moved 2.3 km in four minutes, and the
storm's updraft ended outside it. There was no hold and no error, and
`check_nest_following.py` found nothing wrong.

### 6. Run the nest

Copy the third deck:

```sh example
cp examples/tornado-4m/small/S3.toml ~/gm1-runs/nest/S3.toml
```

**Edit the copy before you run it.** Every value marked `PLACEHOLDER` in it has to
become your own:

- `rstnum`, `parent_i0`, `parent_j0` and the window's `start_s`: the numbers from
  step 5;
- `duration_s` and the window's `end_s`: when you want it to stop. `duration_s` is a
  multiple of 300 s in this deck, and `end_s` is no later than it;
- for the wider child on a 24 GB card, also set `parent_span_x` and `parent_span_y` to
  100. Nothing else changes: the deck's memory budget already allows it.

Before you spend the GPU time, you can see what the child will start from. This
interpolates one parent frame (here frame 8; use the frame at your start time) onto the
child grid the deck declares, and writes both as raw planes with a description beside
them. It is the parent's state on a finer mesh, not a forecast. Without
`--level-index` it takes the lowest level, where the vertical wind is zero everywhere,
so ask for a level that has an updraft. Levels count from 0 at the ground, and level 22
of this example's parent is about 1 km up:

```sh example
python3 tools/nest_window_preview.py ~/gm1-runs/nest/S3.toml \
    ~/gm1-runs/nest/P/frames/frame-000008 ~/gm1-runs/nest/S3/preview --level-index 22
```

Then check it, price it and run it. The only difference from step 4 is that the deck has
a `[nest]` table:

```sh example
gpuwm-cm1 check ~/gm1-runs/nest/S3.toml
gpuwm-cm1 cartesian ~/gm1-runs/nest/S3.toml --plan --integrate \
    --sounding src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding \
    --restart-from ~/gm1-runs/nest/P/restarts \
    --output ~/gm1-runs/nest/S3/receipt.json \
    --history-dir ~/gm1-runs/nest/S3/frames \
    --restart-dir ~/gm1-runs/nest/S3/restarts > ~/gm1-runs/nest/S3/plan.json
python3 tools/plan_summary.py ~/gm1-runs/nest/S3/plan.json
gpuwm-cm1 cartesian ~/gm1-runs/nest/S3.toml --integrate --history-envelope off \
    --sounding src/cm1core/data/soundings/koun-2011-05-24-1800z.input_sounding \
    --restart-from ~/gm1-runs/nest/P/restarts \
    --output ~/gm1-runs/nest/S3/receipt.json \
    --history-dir ~/gm1-runs/nest/S3/frames \
    --restart-dir ~/gm1-runs/nest/S3/restarts
```

A nested step costs far more than a parent step. On the RTX 4090 a parent step of this
example took 0.40 s alone and 2.0 s with the 500 x 500 x 80 child (3 child steps). In
the original 4 m run a parent step took 0.53 s alone and 5.3 s with the nest (12 child
steps of 500 x 500 x 103 cells), on an RTX 5090.

### 7. Look at it

```sh example
gpuwm-cm1 quicklook ~/gm1-runs/nest/S3/frames --deck ~/gm1-runs/nest/S3.toml
gpuwm-cm1 quicklook ~/gm1-runs/nest/S3/frames/nest-child --deck ~/gm1-runs/nest/S3.toml
gpuwm-cm1 quicklook ~/gm1-runs/nest/S3/frames/nest-child --deck ~/gm1-runs/nest/S3.toml --film spin-near-ground
```

The first line draws the parent of the nested stage into `S3/quicklook/`. The other two
draw the child, and by the same rule their pictures land beside the `nest-child` folder,
in `S3/frames/quicklook/`, with the film in
`S3/frames/quicklook/films/spin-near-ground/` (`film.gif` and `film.png`).

Look at the child's reflectivity first: it shows whether the storm is inside the child.
The child's "spin near the ground" picture is drawn on a fixed scale that ends at 0.5
per second, so a child with only weak rotation is almost black.

"Looking at the result" below lists everything a nested run leaves on disk.

## Every `[nest]` setting

The nested deck is the parent's deck with this table added. Without the table the run
is exactly the single-grid run it was. This is the table of
`examples/tornado-4m/small/S3.toml`:

```toml
[nest]
refinement_ratio = 5
time_refinement_ratio = 3
parent_i0 = 194
parent_j0 = 194
parent_span_x = 60
parent_span_y = 60
spec_zone = 1
relax_zone = 4
feedback = 1
acoustic_substeps = 6
follow_storm = true
follow_tracker = "wrf-surface-circulation"
follow_interval_s = 7.5
follow_max_speed_m_s = 20.0
follow_smoothing_radius_m = 300.0
follow_on_edge = "hold"
follow_hold_limit = 4
follow_relocation_radius_m = 2000.0
```

**The grid**

| Key | Meaning |
|---|---|
| `refinement_ratio` | The child's cell is the parent's divided by this. Odd, 3 or more, and the result must be a length the model stores exactly: from 100 m, 5 (20 m) or 25 (4 m). |
| `time_refinement_ratio` | The child's time step is the parent's `dt_s` divided by this, and the child takes that many steps for each parent step. Make it large enough that the fastest wind you expect crosses less than one child cell per step. |
| `parent_i0`, `parent_j0` | The parent cell at the child's south-west corner, counted from 0 at the parent's south-west corner. |
| `parent_span_x`, `parent_span_y` | How many parent cells the child covers each way. The child has that many times the ratio cells: 60 x 5 = 300. |
| `acoustic_substeps` | Sound-wave substeps in each child step. When the time ratio is close to the grid ratio, use the parent's value. The plan checks it against CM1's own rule for the child's cell and step. |

**The edges**

| Key | Meaning |
|---|---|
| `spec_zone` | Rows at the child's edge that the parent sets outright. 1 in every run so far. |
| `relax_zone` | Width, from the edge, of the band where the child is pulled toward the parent. More than `spec_zone`. 4 in every run so far. |
| `feedback` | 1: the child's interior is written back into the parent (two-way). 0: one-way. |

**Following the storm**

| Key | Meaning |
|---|---|
| `follow_storm` | `true` makes the child move. |
| `follow_tracker` | What it follows. `"wrf-surface-circulation"` follows the rotation at the child's lowest level and is the one to use for a tornado. `"cm1-pressure-held-offset"` (used when the key is absent) and `"cm1-pressure-centre"` follow the parent's low-level pressure centre. |
| `follow_interval_s` | Seconds between looks for the vortex. |
| `follow_max_speed_m_s` | The fastest vortex, relative to the moving grid, that the search will reach. |
| `follow_smoothing_radius_m` | The tracker averages vorticity over a box of this half-width before it looks for the maximum. Required with the surface tracker, and the setting that matters most: see below. |
| `follow_on_edge` | `"hold"` keeps the nest where it is when a look finds nothing usable, and carries on. Without the key the run stops at the first such look. |
| `follow_hold_limit` | With `"hold"`: after this many held looks in a row, search a wider area and jump to the strongest rotation in it. |
| `follow_relocation_radius_m` | With `"hold"`: the radius of that wider search. At least `(follow_hold_limit + 1) x follow_max_speed_m_s x follow_interval_s` plus one child cell. Twice that is safer. |
| `follow_corral_m` | Optional: how far the child must stay from the parent's edge. |

**Choosing `follow_smoothing_radius_m`.** Make it about the size of the circulation you
want to follow. Too small, and a short-lived eddy on the gust front captures the tracker
and the nest wanders off with it. Too large, and the box cannot centre on a compact
vortex. The original 4 m run was first started with 100 m: the nest followed gust-front
eddies to the south-south-west, away from the vortex. With 250 m it stayed on the
tornado for 21 minutes of storm time. The 20 m nest used 300 m.

**The child's own levels** (optional; without them the child has the parent's levels)

| Key | Meaning |
|---|---|
| `child_nz` | How many levels the child has. |
| `child_stretch_z` | How they are given: 1 (the same rule as `[grid]` `stretch_z = 1`, with the four keys below), 3 (a list of scalar level heights) or 4 (a list of the heights between levels). |
| `child_str_bot_m`, `child_str_top_m`, `child_dz_bot_m`, `child_dz_top_m` | With 1: constant thickness `child_dz_bot_m` below `child_str_bot_m`, constant `child_dz_top_m` above `child_str_top_m`, stretching between. |
| `child_levels_m` | With 3 or 4: the heights in metres, lowest first. With 4 there is one more height than `child_nz`. |

The child's top is the parent's top, and the child may not be coarser than the parent at
any height. `examples/tornado-4m/S3.toml` uses `child_stretch_z = 4` with 4 m layers up
to 100 m.

**The live tornado check** (optional)

An empty `[nest.tornado_criterion]` table makes the run test the child at every parent
step: rotation of at least 0.30 per second at 10 m with a ground-relative wind of at
least 35 m/s near it and a core at least 40 m in radius, sustained for 60 s. The
thresholds are fixed and cannot be set in the deck. The only keys are two stop rules,
`stop_after_episode_s` and `stop_if_no_tornado_by_s`. The verdict goes into the receipt
and every sample into `nest-child/tornado-criterion.jsonl`.

This live check is the first version of GM1's tornado rule. The verdicts GM1 publishes
come from the second version, which is not in the run: a scorer
(`tools/criterion_v2/score.py`) reads the child frames or the window stream afterwards.
The two can disagree. On the 4 m run the first version, recomputed from the window
records, did not call a tornado and the second did.
[GM1-TORNADO-CRITERION-V2.md](GM1-TORNADO-CRITERION-V2.md) is the rule, and
[GM1-TORNADO-VERDICTS-V2.md](GM1-TORNADO-VERDICTS-V2.md) has the command that scores a
run of your own.

The check keeps 1000 m clear of the child's edges. On a 2 km child, such as the 4 m one,
that leaves one point to search, so the 4 m deck leaves the table out.

**The window** (optional)

The `[lagrangian.nest_child_window]` table writes a small box of the child around the
vortex many times a second. That is what lets you follow air into a tornado afterwards,
and it is what a tornado verdict has to be scored from: child frames a minute or two
apart cannot show how long a vortex lasted.

| Key | Meaning |
|---|---|
| `frame` | `"parent-grid-origin"`: positions are metres from the parent's south-west corner. |
| `follow` | `"nest-centre"`: the box stays at the centre of the child. `"tornado-criterion"`: it follows the live check's vortex centre (needs that table). |
| `center_x_m`, `center_y_m` | Not read with those two choices. Write 0.0. |
| `half_width_x_m`, `half_width_y_m` | Half the box's width each way. |
| `top_m` | The box's top. It always reaches down to the ground. |
| `stride_child_steps` | Write one record every this many child steps. |
| `start_s`, `end_s` | The model times it writes between. |
| `budget_bytes_per_model_minute` | The most disk you allow it per minute of storm time. The plan works out the real cost and refuses the run (`CM1-CARTESIAN-NEST-WINDOW-DISK-BUDGET`) if it is more, naming the real figure. |

The box must stay inside the part of the child the parent does not control, with a few
cells to spare, and below the child's top.

## Placing the nest

`tools/nest_place.py` does for you what was done by hand for the 4 m run. For each
parent frame it:

1. takes the vertical vorticity at the level nearest 10 m and averages it over boxes
   100 m, 300 m and 500 m wide. A box average is the circulation around the box divided
   by its area, so one noisy cell counts for little in the wider boxes;
2. finds the largest value of each within 8 km of the storm's rotating updraft (the
   largest updraft helicity in the frame), and never within 5 km of the side of the
   domain;
3. calls the frame *tight* when the three maxima are within 150 m of each other and the
   300 m value is at least 0.05 per second. That is one compact vortex, not a line of
   eddies.

It proposes the first restart time whose frame, the one before and the one after are
all tight, and whose 300 m maximum moved slower than 11.7 m/s between them: a vortex
that is compact and moving smoothly. The first frame of the folder has none before it,
so there the two frames after it are used. The last frame is never proposed. Frames
more than two minutes apart are only screened: the tool lists the tight times and
proposes nothing. The child is centred there: `parent_i0` and
`parent_j0` are the cell of the 300 m maximum less half the span.

The 4 m run searched around the vortex that an earlier 20 m nest had found. The tool
searches around the updraft, which needs no earlier nest. Run on that run's frames it
gives the same answer: restart 8, 2160 s, cells 295 and 304. `--near X_M,Y_M` makes it
search around a point you choose, and every threshold is an option
(`python3 tools/nest_place.py --help`).

To do it by eye: draw the "spin near the ground" pictures of the one-minute frames, find
the time from which one tight maximum moves smoothly, read its position in kilometres,
divide by the parent's cell size to get its cell, and subtract half the span.

## Looking at the result

A nested run with `--history-dir FRAMES` and `--restart-dir RESTARTS` leaves:

| Path | What it is |
|---|---|
| `FRAMES/frame-NNNNNN/` | the parent, every `history_interval_s` |
| `FRAMES/nest-child/frame-NNNNNN/` | the child at the same times |
| `FRAMES/nest-child/placements/frame-NNNNNN.json` | where the child was at each of them |
| `FRAMES/nest-child/tornado-criterion.jsonl` | the live check's samples, if the deck has that table |
| `FRAMES/nest-child-window/window-stream.f32`, `window-records.tsv` | the window's records and their index, if the deck has a window |
| `FRAMES/nest-child-window/window-stream.json` | the window's description, written when the run ends |
| `RESTARTS/restart-NNNNNN/` with `nest-child/` inside | a restart of both grids |
| the `--output` file | the receipt, written when the run ends: what ran, where the nest went, the verdict |

**Pictures and films.** `gpuwm-cm1 quicklook` draws a child folder as it draws any
run, given the deck. On the original run's 10 saved 4 m frames, GM1 0.1.0 drew all ten
looks (reflectivity, spin near the ground, updraft, the swaths, the surface, rain,
clouds from above and the tracks) in 6 minutes 23 seconds on the CPU. The swath and
track pictures of a child are drawn on the child's own moving square, so read them as
"what the child saw", not as a map of the ground.

**CM1-style files.** `gpuwm-cm1 cm1out` writes netCDF from a child folder the same way,
into a `cm1out` folder beside the folder you name: here `S3/frames/cm1out/`. From the
original run's 10 saved 4 m frames, GM1 0.1.0 wrote 10 files of 2.7 GB each, and from
the RTX 4090 run's four 500 x 500 x 80 child frames, 4 files of 2.09 GB each:

```sh example
gpuwm-cm1 cm1out ~/gm1-runs/nest/S3/frames/nest-child --deck ~/gm1-runs/nest/S3.toml --format nc
```

**The live check, again.** This re-decides the verdict from the samples alone, which
also works for a run that stopped without a receipt:

```sh example
python3 tools/tornado_criterion.py replay ~/gm1-runs/nest/S3/frames/nest-child/tornado-criterion.jsonl
```

**Did the nest follow what it says it followed?**

```sh example
python3 tools/check_nest_following.py ~/gm1-runs/nest/S3/receipt.json
```

This checks that the child went where its tracker pointed, never faster than the deck
allows. It does not check that the tracker was on your storm. In the RTX 4090 run it
reported no failure for a child that had followed a weak eddy out of the storm. Look at
the child's reflectivity picture for that.

**Air into the vortex.** Seed points in the vortex and follow them backward through the
window:

```sh example
python3 tools/nest_window_trajectories.py \
    --stream ~/gm1-runs/nest/S3/frames/nest-child-window --out ~/gm1-runs/nest/S3/air \
    --target-time-s 3000 --back-to-s 2940 --centre auto \
    --radius-m 200 --heights-m 10,60 --rings 3 --per-ring 12
```

`--target-time-s` is the time of one of the window's records (the `model_time_s` column
of `window-records.tsv`). It writes these into the `--out` folder, here
`~/gm1-runs/nest/S3/air`:

- `trajectories/backward-trajectories.f32`: 32-bit numbers in the shape steps x 12 x
  points. The file `backward-trajectories.json` beside it gives that shape, names the 12
  rows (position, wind, time, a status code and the error estimates), lists the model
  time of every step and says what each status code means.
  `backward-trajectory-table.tsv` is the same as a text table;
- `budget/trajectory-budget.f32`, `trajectory-budget.json` and
  `trajectory-budget-table.tsv`, in the same form: the terms of the vorticity budget at
  every position, and how well they add up;
- `circulation.json`, if you asked for a closed circuit with `--circuit`.

To read the positions:

```python
import json
from pathlib import Path

import numpy as np

folder = Path.home() / "gm1-runs" / "nest" / "S3" / "air" / "trajectories"
record = json.loads((folder / "backward-trajectories.json").read_text())
rows = np.fromfile(folder / "backward-trajectories.f32", dtype="<f4")
rows = rows.reshape(record["arena_shape_steps_rows_nparcels"])
x, y, z = rows[:, 0, :], rows[:, 1, :], rows[:, 2, :]   # metres; a row per step back, a column per point
```

Positions are metres from the parent's south-west corner, in the grid that moves with
the storm. The window needs its `window-stream.json`, which the run writes when it
ends: a run that stopped on an error leaves records this tool cannot read.

`tools/nest_backward_trajectories.py` does the same through the child's frames. It
reports its own position error, and with frames a minute or more apart near a strong
vortex that error is larger than the storm: use the window.

These tools were each run with GM1 0.1.0 on saved files of the original chain of runs,
and all but `nest_backward_trajectories.py` on the files of the RTX 4090 run. They are
small source tools, not supported commands.

## When it stops

A run that cannot go on stops with a code and a sentence that says what to change. The
ones you are likely to meet:

| Code | What happened | What to do |
|---|---|---|
| `CM1-CARTESIAN-RUN-GEOMETRY` | The storm would leave the parent's domain before the end. The moving frame or the bubble does not suit this sounding. | Use the frame and bubble position the message prints. |
| `CM1-CFG-NEST-RATIO-PARITY` | An even `refinement_ratio`. | Use an odd one. |
| `CM1-CFG-NEST-CHILD-SPACING` | The child's cell size cannot be stored exactly (100 m / 3, for example). | From 100 m use 5 or 25. |
| `CM1-CFG-CARTESIAN-CADENCE` | `duration_s` is not a whole number of frame intervals. | Change one of them. |
| `CM1-CARTESIAN-RESTART-SELECTION` | The deck continues a run and no `--restart-from` was given. | Pass the earlier stage's restart folder. |
| `CM1-CARTESIAN-NEST-WINDOW-DISK-BUDGET` | The window costs more disk than the deck allows. | Raise the budget to the figure in the message, or write less often. |
| `CM1-CARTESIAN-NEST-FOLLOW-LOST` | The tracker found no usable wind near the vortex. | Usually the child has become unstable: see the last row. |
| `CM1-CARTESIAN-NEST-FOLLOW-RELOCATION` | After the held looks, no rotation was found inside the relocation radius. | A larger `follow_relocation_radius_m`. The vortex may also have died. |
| `CM1-CARTESIAN-NEST-FOLLOW-CORRAL` | The vortex kept trying to leave through the parent's edge. | The parent domain is too small for this storm's path. |
| `CM1-CARTESIAN-NEST-RESTART-MISMATCH` | A nested restart was continued with a different nest (another ratio or time ratio). | Continue with the same `[nest]` table, or start the nested stage again from the parent-only restart. |
| `CM1-CARTESIAN-RUN-FAILED` | The model produced values that are not physical. On a fine child this is almost always the time step: the wind began to cross more than one child cell per step. | A larger `time_refinement_ratio`, started again from the parent-only restart. |

About the last row: `plan_summary.py` prints the wind that crosses one child cell per
child step. For the 4 m deck it is 128 m/s. The original run passed it at 3312 s and
stopped at 3452.8 s. Nothing after the moment a run passes that speed should be
trusted, even if the run carries on for a while.

An out-of-memory stop is a card that is too small for what the plan listed plus about
4 GiB. The receipt's `sampled_peak_device_used_bytes` counts everything on the card,
other programs included, so on a shared card it is not this run's own use. Close other
programs that use the card, or make the child smaller with `parent_span_x` and
`parent_span_y`.
