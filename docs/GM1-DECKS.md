# Running GM1 decks: beyond the GUI

The GUI and `gpuwm-cm1 run` take a CM1 namelist and keep to what has been compared with CM1. GM1's own input is a
TOML file called a deck, and `gpuwm-cm1 cartesian` runs one directly. A deck reaches everything the engine has:
nests, stretched near-ground levels, the storm-following domain, parcels and budgets, restarts and ensembles, every
side type. This page is the short route through that door. The full reference, key by key, is
[`GM1-CARTESIAN-RUNTIME.md`](GM1-CARTESIAN-RUNTIME.md); nesting has its own page,
[`NESTING.md`](NESTING.md).

Much of this has not been compared with CM1, and some of it has no CM1 counterpart at all. Each recipe below says
what was checked. `gpuwm-cm1 doctor --science` prints the same per capability, and every run's receipt lists what was
not checked for that run. Push it, and when something breaks, paste `gpuwm-cm1 doctor --report` into an issue
together with the command and its last lines.

How this page was checked: every command in a block marked `sh` runs in the test suite on a computer with no GPU.
Blocks marked `sh gpu` need your graphics card; the suite checks their spelling and every option against the
installed program, and does not run them. The numbers in the tables come from `gpuwm-cm1 check`,
`gpuwm-cm1 cartesian --plan` and `gpuwm-cm1 estimate` on a CPU, with the 0.1.0 wheel and again with the
0.1.3 wheel, which gave the same answers.

## A first deck, start to finish

List the shipped decks and copy them into a folder:

```sh
gpuwm-cm1 examples
gpuwm-cm1 examples --copy decks
```

Pick one. `cartesian-upstream-bubble-seeded` is a 1 km supercell on a 120 km area (about 1.4 GB of GPU memory, so
any supported card). Its storm stays clear of the edge for about 1.8 of its two hours, and the `estimate` below says
so. Three checks cost nothing and open no GPU:

```sh
gpuwm-cm1 check decks/cartesian-upstream-bubble-seeded.toml
gpuwm-cm1 cartesian decks/cartesian-upstream-bubble-seeded.toml --plan --output plan.json
gpuwm-cm1 estimate decks/cartesian-upstream-bubble-seeded.toml
```

- `check` reads the deck strictly: an unknown key or a value GM1 does not run is refused by name, with a remedy.
- `cartesian --plan` admits the run as the run itself would (geometry, time step, memory layout, the storm-track
  check, the list of what is not checked against CM1) and prints the plan as JSON. `--output` is required, but with
  `--plan` nothing is written there; add `> plan.txt` to keep the plan.
- `estimate` prices it against this computer: GPU memory, host memory, disk, wall time, and a warning when the storm
  would reach the edge of the area. It does not price a nest: for a nested deck only the parent is counted, and it
  says so.

Then run it. Every output path must be new:

```sh gpu
mkdir run1
gpuwm-cm1 cartesian decks/cartesian-upstream-bubble-seeded.toml --integrate --output run1/receipt.json --history-dir run1/frames --restart-dir run1/restarts --control-dir run1/control --history-initial --history-envelope off
```

`--integrate` steps to the deck's `run.duration_s` and writes one frame folder every `run.history_interval_s` into
`run1/frames`, and a restart every `run.restart_interval_s` into `run1/restarts`. `--history-initial` also writes
the starting state as the first frame, as CM1 does. `--history-envelope off` writes one copy of each frame, not
two (the second copy is only for `gpuwm-cm1 render`); `gpuwm-cm1 run` uses the same setting. A progress line is
printed every 30 seconds (`--progress-interval SECONDS` changes that). A deck that reads a sounding file (the table below says which) also needs
`--sounding FILE`.

Look at it. The pictures and CM1's own output files are made on the CPU from the frames, while the run is going or
afterwards:

```sh gpu
gpuwm-cm1 quicklook run1/frames --deck decks/cartesian-upstream-bubble-seeded.toml
gpuwm-cm1 cm1out run1/frames --deck decks/cartesian-upstream-bubble-seeded.toml --format nc
gpuwm-cm1 perf run1/receipt.json
```

`quicklook` draws the same looks as the GUI (reflectivity, spin near the ground, updraft, the wind swath and the
rest) into `run1/quicklook`: a folder beside the frames folder, not inside it. `--slice FIELD --height M`,
`--section`, `--column` and `--film LOOK` work as in [`CM1-USERS.md`](CM1-USERS.md). `cm1out` writes CM1-format
netCDF (or GrADS with `--format grads`) into `run1/cm1out`, by the same rule. `--out DIR` sends either somewhere
else. Give every run a folder of its own, as here: two frames folders side by side in one folder share one
`quicklook` folder, and drawing the second run removes the pictures of the first. `perf` summarises where the time
went from the receipt.

For a nested run the parent's frames are in the history folder and the child's in its `nest-child` subfolder; point
`quicklook` at that subfolder with the same `--deck`, and the child's pictures land beside it, in the history
folder's `quicklook`. Its swath and track looks assume one moving domain and are
not right for a moving nest yet; the per-frame looks and `--slice` are.

### What a deck run does not get

`gpuwm-cm1 cartesian` is the engine's own door. The run folder that `gpuwm-cm1 run`, `status`, `stop`, `pause`,
`resume` and the GUI share does not exist for it:

- no GUI progress, no entry in the GUI's run list, no `gpuwm-cm1 status`;
- no waiting in line for a busy card, and no check of free memory before it starts: run `estimate` first. With
  `--gpu-uuid UUID --lock DIR` the run takes an exclusive lock on that card before it opens it (`nvidia-smi -L`
  prints the UUID);
- no automatic pictures or CM1 files: run `quicklook` and `cm1out` yourself, as above (both take `--follow`);
- `gpuwm-cm1 reproduce` and `jobscript` take run folders, not decks. The receipt at `--output` records the deck's
  hash, the sounding's hash, the versions and the step times, so keep it with the deck.

### Stopping, and going on later

Start the run with `--control-dir DIR` (as above). From another terminal:

```sh gpu
touch run1/control/stop
```

The run ends at the end of the step it is on, after any frame being written is safe on disk, writes its receipt and
exits 0. A file named `pause` instead ends it right after the next restart is written. Ctrl-C also stops a run, at
once and with exit code 130: the frames and restarts already written stay on disk. Use the stop file when you want
the receipt.

To continue from a restart, copy the deck, set `irst = 1` and `rstnum = N` in its `[run]` table (N is the number in
the folder name `restart-00000N`), and run it with new output paths and `--restart-from` naming the first run's
restart folder. `run.duration_s` is the model time to stop at, counted from the start of the first run.
`cartesian-supercell-checkpointed` and `cartesian-supercell-resume` show the keys as a pair (as shipped, the first
is refused at the plan: see the table), and "Checkpoint and restart" in the reference has the rules. A restart is read only by the GM1 version that wrote it.

## The shipped decks

`gpuwm-cm1 examples` lists 39 decks; the repository's `examples/` folder holds one more. The table gives what each
shows, its grid, and what the 0.1.0 and 0.1.3 wheels say about it on a CPU (the same).

Read the last column with care:

- "admitted" means `check` and `cartesian --plan` both pass. It does not mean the deck has been run. No deck of
  this table was run on a GPU for this page, and the project has no record of a full run of any of these files
  exactly as written. The decks that have been run as written from a pip install are the first stages of the small
  nesting example, run on GM1 0.1.0 on an RTX 4090 ([`NESTING.md`](NESTING.md), "What has been done with it"). The physics each
  deck switches on has its own device tests (`gpuwm-cm1 doctor --science`); the largest single domain GM1 has run
  is 39.3 million cells, so the decks above that size are designs.
- "GPU memory" is the estimate for the parent domain with the tiles the deck declares. "Smallest card" leaves one
  fifth of the card free for the desktop and start-up. Check your own card with `gpuwm-cm1 estimate DECK`.
- Many of the small 60 km decks exist to show one switch. The storm-track check says their storm reaches the edge
  after about 20 minutes; after that the largest values are edge effects.

| Deck | What it shows | Grid | Storm time | GPU memory (estimate) | Smallest card | On the 0.1.0 and 0.1.3 wheels, CPU only |
|---|---|---|---|---|---|---|
| `cartesian-adaptive-moving-domain` | A 50 m supercell domain that follows its own storm (CM1's `do_adapt_move`) | 960 x 960 x 75, 50 m | 2 h | 30 GB | 32 GB, with nothing else on the card | admitted; 69.1 M cells, 1.4 TB of output and about 30 h on an RTX 5090 by the estimate: not run as written |
| `cartesian-environment-fields` | CAPE, CIN, LCL and LFC written as planes in every frame; a 5 s harness run | 96 x 96 x 40, 500 m | 5 s | 1.4 GB | 8 GB | admitted; needs CM1's own `input_sounding` (see below) |
| `cartesian-fifty-metre-fine-step` | A 3 km box at 50 m with a 0.05 s step: the small-step path | 64 x 64 x 64, 50 m | 10 min | 1.3 GB | 8 GB | admitted |
| `cartesian-idealized-radiation-relaxation` | CM1's `rterm = 1` Newtonian cooling | 32 x 32 x 40, 1000 m | 7.3 min | 1.1 GB | 8 GB | admitted |
| `cartesian-initial-condition-only` | CM1's `iconly`: build the starting state, write it, stop | 120 x 120 x 40, 1000 m | 2 h | 1.4 GB | 8 GB | `check` passes; the plan needs `--restart-dir` or `--history-dir`; needs CM1's own `input_sounding` (see below) |
| `cartesian-nested-storm-following-mesh` | A 100 m storm-following two-way nest in a 900 m parent (ratio 9) | 120 x 120 x 40, 900 m | 1.5 h | 0.92 GB parent; child not priced (its arrays: 1.1 GB) | 8 GB | admitted |
| `cartesian-nested-tornado-window` | A 20 m nest in a 100 m parent with a vortex-following high-rate window | 800 x 800 x 80, 100 m | 1.5 h | 23 GB parent; child not priced (its arrays: 6.8 GB) | 32 GB | admitted; 51.2 M parent cells, more than any domain GM1 has run (39.3 M): not run as written |
| `cartesian-open-boundary-companions` | CM1's four open-boundary companion options, all on; a 60 s run | 96 x 96 x 40, 500 m | 60 s | 1.3 GB | 8 GB | admitted; needs CM1's own `input_sounding` (see below) |
| `cartesian-periodic-laterals` | Four repeating (periodic) sides; a 60 s run | 96 x 96 x 40, 500 m | 60 s | 1.3 GB | 8 GB | admitted; needs CM1's own `input_sounding` (see below) |
| `cartesian-restart-and-ensemble` | A second leg from a restart, and one ensemble member branched from it | 32 x 32 x 40, 1000 m | 20 min | 1.1 GB | 8 GB | `check` passes; the plan needs `--restart-from` (a first leg's restarts) |
| `cartesian-squall-line-rkw-shear` | CM1's squall-line initialization on a 120 km area | 120 x 120 x 40, 1000 m | 2 h | 1.4 GB | 8 GB | admitted |
| `cartesian-stretched-tornado-grade-grid` | 50 m with 20 m layers near the ground over 60 km | 1200 x 1200 x 75, 50 m | 2 h | 47 GB | none (more than 32 GB) | admitted; needs more than 32 GB and cannot write a restart (its state is over the 16 GiB restart limit): not run as written |
| `cartesian-stretched-tornado-two-part-sgs` | The same grid with CM1's two-part near-wall turbulence model | 1200 x 1200 x 75, 50 m | 2 h | 47 GB | none (more than 32 GB) | admitted; needs more than 32 GB and cannot write a restart (its state is over the 16 GiB restart limit): not run as written |
| `cartesian-supercell-attempt` | A 220 km supercell domain on the analytic Weisman-Klemp sounding | 220 x 220 x 40, 1000 m | 2 h | 2.0 GB | 8 GB | admitted |
| `cartesian-supercell-checkpointed` | CM1's 2 h supercell with restarts every interval | 120 x 120 x 40, 1000 m | 2 h | 1.4 GB | 8 GB | `check` passes; the plan refuses it with CM1's sounding (`CM1-CARTESIAN-RUN-GEOMETRY`: the storm leaves at 53 min) and prints the frame and bubble that fix it; needs CM1's own `input_sounding` (see below) |
| `cartesian-supercell-coriolis` | CM1's supercell with Coriolis on (a 6 s step) | 60 x 60 x 40, 1000 m | 2 h | 1.2 GB | 8 GB | admitted; the storm-track check says the storm reaches the edge at about 20 min |
| `cartesian-supercell-moore-20130520-18z-16gib` | Norman 20 May 2013 18 UTC sounding, 125 m, 56 km | 448 x 448 x 80, 125 m | 2 h | 7.9 GB | 12 GB | admitted; its sounding is copied beside it |
| `cartesian-supercell-moore-20130520-18z-32gib` | The same sounding at 100 m, 54 km | 544 x 544 x 80, 100 m | 2 h | 11 GB | 16 GB | admitted; its sounding is copied beside it |
| `cartesian-supercell-morrison-hail` | CM1's supercell on Morrison with hail (`ptype = 5`, `ihail = 1`) | 60 x 60 x 40, 1000 m | 2 h | 0.69 GB | 8 GB | admitted; the storm-track check says the storm reaches the edge at about 20 min |
| `cartesian-supercell-reflectivity` | CM1's supercell with simulated reflectivity in the frames (WSM6) | 60 x 60 x 40, 1000 m | 2 h | 1.2 GB | 8 GB | admitted; the storm-track check says the storm reaches the edge at about 20 min |
| `cartesian-supercell-resume` | The second half of the checkpointed run, from its restart | 120 x 120 x 40, 1000 m | 2 h | 1.4 GB | 8 GB | `check` passes; the plan needs `--restart-from` (a first leg's restarts); needs CM1's own `input_sounding` (see below) |
| `cartesian-supercell-storm-diagnostics` | The 220 km supercell with CM1's storm diagnostics on | 220 x 220 x 40, 1000 m | 2 h | 2.0 GB | 8 GB | admitted |
| `cartesian-supercell-surface-drag` | Semi-slip ground (`bbc = 3`, constant drag coefficient) | 60 x 60 x 40, 1000 m | 2 h | 1.2 GB | 8 GB | admitted; the storm-track check says the storm reaches the edge at about 20 min |
| `cartesian-supercell-swok-20190521-00z-16gib` | Norman 21 May 2019 00 UTC sounding, 125 m, 56 km | 448 x 448 x 80, 125 m | 2 h | 7.9 GB | 12 GB | admitted; its sounding is copied beside it |
| `cartesian-supercell-swok-20190521-00z-32gib` | The same sounding at 100 m, 54 km | 544 x 544 x 80, 100 m | 2 h | 11 GB | 16 GB | admitted; its sounding is copied beside it |
| `cartesian-supercell-thompson` | CM1's supercell on Thompson (`ptype = 3`); unverified scheme | 60 x 60 x 40, 1000 m | 2 h | 0.68 GB | 8 GB | admitted; the storm-track check says the storm reaches the edge at about 20 min |
| `cartesian-supercell-thompson-aerosol` | Aerosol-aware Thompson, which this release does not offer | 60 x 60 x 40, 1000 m | 2 h | 0.69 GB | 8 GB | in the repository only (not listed by `examples`): `check` passes, the plan refuses the scheme |
| `cartesian-supercell-translating-frame` | CM1's supercell in its translating frame (`imove = 1`) | 60 x 60 x 40, 1000 m | 2 h | 1.2 GB | 8 GB | admitted; the storm-track check says the storm reaches the edge at about 20 min |
| `cartesian-surface-enthalpy-flux-over-water` | A water surface exchanging heat and moisture (`cecd = 3`) | 60 x 60 x 40, 1000 m | 2 h | 1.2 GB | 8 GB | admitted; the storm-track check says the storm reaches the edge at about 29 min |
| `cartesian-surface-prescribed-roughness` | Semi-slip ground with a prescribed roughness length | 60 x 60 x 40, 1000 m | 2 h | 1.2 GB | 8 GB | admitted; the storm-track check says the storm reaches the edge at about 20 min |
| `cartesian-tornado-budgets` | CM1's per-process budgets and the vorticity budget; a 10 min run | 96 x 96 x 40, 1000 m | 10 min | 1.3 GB | 8 GB | admitted |
| `cartesian-tornado-diagnostics` | Every CM1 rotation diagnostic and surface swath on | 120 x 120 x 40, 1000 m | 2 h | 1.4 GB | 8 GB | admitted |
| `cartesian-tornado-lagrangian-analysis` | Parcels, backward paths, circulation and budgets inline; a 10 min run | 96 x 96 x 40, 1000 m | 10 min | 0.75 GB | 8 GB | admitted |
| `cartesian-uniform-grid` | The plain uniform grid; a 5 s harness run | 96 x 96 x 40, 500 m | 5 s | 1.4 GB | 8 GB | admitted; needs CM1's own `input_sounding` (see below) |
| `cartesian-upstream-bubble-seeded` | A supercell whose bubble starts upstream so the storm stays inside for most of the 2 h | 120 x 120 x 40, 1000 m | 2 h | 1.4 GB | 8 GB | admitted; `estimate` warns that the storm reaches the edge at about 1.8 h |
| `cartesian-wk-analytic-sounding-shear` | The analytic Weisman-Klemp sounding with its shear profile | 120 x 120 x 40, 1000 m | 2 h | 1.4 GB | 8 GB | admitted; the storm-track check says the storm reaches the edge at about 73 min |
| `dry-psolver2-bootstrap` | A dry-dynamics development profile | not priced |  | not priced | not applicable | another door's profile: `check` passes, `cartesian` refuses it (`CM1-CARTESIAN-LANE`) |
| `official-supercell-native-context` | A development profile of CM1's supercell | not priced |  | not priced | not applicable | another door's profile: `check` passes, `cartesian` refuses it (`CM1-CARTESIAN-LANE`) |
| `official-supercell-product` | A development profile of CM1's supercell | not priced |  | not priced | not applicable | another door's profile: `check` passes, `cartesian` refuses it (`CM1-CARTESIAN-LANE`) |
| `real-atmosphere-100m-product` | A real-atmosphere 100 m development profile | not priced |  | not priced | not applicable | another door's profile: `check` passes, `cartesian` refuses it (`CM1-CARTESIAN-LANE`) |

Decks that need CM1's own `input_sounding`: CM1's files are not redistributed. `gpuwm-cm1 examples` prints, for each
such deck, the file's name inside the CM1 r21.1 release, its size and SHA-256, and where to download the release;
pass the extracted file with `--sounding`. The four decks named for a day carry a NOAA balloon sounding, which
`examples --copy` writes beside the deck as `NAME.input_sounding`.

To change a deck, edit the copy: `[run]` holds the length, the time step and the output and restart intervals,
`[grid]` the cells and spacings, `[physics]` the microphysics scheme, `[runtime]` the memory budget and tile size.
Run `check` and `cartesian --plan` after every change; they name the key and the remedy when a change is refused.

## Recipes

Each recipe starts from CM1's own supercell namelist, which GM1 ships unchanged, and changes it with
`gpuwm-cm1 edit`. That gives a normal run folder, so `gpuwm-cm1 run`, the GUI's advanced mode, `status`, `stop` and
the automatic pictures all work. Each was imported, edited and priced on a CPU with the 0.1.0 wheel, and this release's test suite
runs the same commands; none was run on a GPU for this page.

```sh
gpuwm-cm1 examples --cm1 --copy cm1
```

Every `import` and `edit` prints the coverage report. Read the "Not yet checked against CM1" rows: they are the
honest label of the run you are about to make.

When `edit` refuses a value it exits with status 2, but the refused value stays written in the run folder's
namelist, and every later `edit` of that folder fails on it. Set the key back to a value that runs
(`gpuwm-cm1 edit RUNDIR --set KEY=VALUE`), or import again into a new folder.

### A 100 m supercell

In guided mode the finest grid offered is 250 m; the 100 m preset is greyed. The engine runs 100 m; these are the 100 m preset's own
changes to CM1's file, with a 0.5 s step:

```sh
gpuwm-cm1 import cm1/supercell/namelist.input --out runs/supercell-100m
gpuwm-cm1 edit runs/supercell-100m --set nx=600 --set ny=600 --set nz=80 --set dx=100 --set dy=100 --set dz=250 --set dtl=0.5 --setting initialization.bubble_center_x_m=11000 --setting initialization.bubble_center_y_m=0
gpuwm-cm1 estimate runs/supercell-100m
```

- 600 x 600 x 80 cells over 60 x 60 km, 2 h, 14,400 steps. The coverage report has no "Not yet checked" row: this
  is the stock supercell on a finer grid. CM1 agreement itself was measured at 1 km only.
- The bubble sits 11 km east of the centre so that the storm stays inside the 60 km area; give both bubble
  settings together.
- Memory: about 27 GB with the whole domain as one closure tile (a 32 GB card). On a smaller card the estimate
  prints the tile size that fits; a GPU budget makes GM1 choose it. With
  `gpuwm-cm1 edit runs/supercell-100m --setting gpu.memory_budget_mib=15500` the same run is priced at about 15 GB
  (25 tiles of 128), which fits a 16 GB card with nothing else on it, and runs more slowly.
- Disk: about 73 GB with a picture every 15 minutes of storm time.
- Status: priced, not run. Storms on 640 x 640 x 80 and 640 x 768 x 80 grids at 100 m have run for hours on an RTX
  5090 with a 0.375 s step; this exact namelist has not been run.
- A moving domain needs a time step that binary numbers hold exactly, such as 0.5, 0.375 or 0.25 s. With
  `dtl = 0.6` the run stops after six steps on a domain-position check. See "The moving domain" below.

Then `gpuwm-cm1 run runs/supercell-100m`.

### 20 m layers near the ground (a stretched column)

CM1's `stretch_z = 1` puts thin layers near the ground under a deep model top. These values give 20 m layers below
200 m (the lowest model level at 10 m), growing to 320 m by 7 km, in 75 levels under a 15 km top:

```sh
gpuwm-cm1 import cm1/supercell/namelist.input --out runs/stretched
gpuwm-cm1 edit runs/stretched --set nx=256 --set ny=256 --set dx=250 --set dy=250 --set dtl=1.5 --set stretch_z=1 --set nz=75 --set dz=100 --set ztop=15000 --set str_bot=200 --set str_top=7000 --set dz_bot=20 --set dz_top=320 --set zd=10000
gpuwm-cm1 estimate runs/stretched
```

- 256 x 256 x 75 at 250 m over 64 km: about 5.2 GB (an 8 GB card with little else on it).
- Set all of the stretch values together. `stretch_z = 1` alone on CM1's file is refused, and `zd` (where the upper
  damping layer starts) must be below `ztop`: CM1's file has `zd = 15000`, which with a 15 km top is refused.
- Label: "Not yet checked against CM1" for `stretch_z = 1`. Stretched levels match CM1 in dry comparisons; no
  stretched run has been compared with CM1 after cloud forms. The tornado runs all used this kind of column.
- On a stretched column the updraft-helicity and vorticity swaths (`output_uh`, `output_vort`, `output_svs`,
  `output_shs`) are not written; the report lists them under "Runs differently".
- The estimate warns that this storm nears the edge after about 28 minutes and prints a frame and bubble position
  that keep it inside; apply them with `--set umove=... --set vmove=...` and the two bubble settings.

### The near-ground physics of the tornado runs

Semi-slip ground with a constant drag coefficient, Coriolis acting on the departure from the base wind, Morrison
with hail, and moisture held at the sounding's relative humidity inside the bubble, on the stretched column above:

```sh
gpuwm-cm1 import cm1/supercell/namelist.input --out runs/tornado-physics
gpuwm-cm1 edit runs/tornado-physics --set nx=256 --set ny=256 --set dx=250 --set dy=250 --set dtl=1.5 --set stretch_z=1 --set nz=75 --set dz=100 --set ztop=15000 --set str_bot=200 --set str_top=7000 --set dz_bot=20 --set dz_top=320 --set zd=10000
gpuwm-cm1 edit runs/tornado-physics --set bbc=3 --set sfcmodel=1 --set cecd=1 --set cnstcd=0.005 --set icor=1 --set lspgrad=1 --setting initialization.maintain_rh=true
gpuwm-cm1 estimate runs/tornado-physics
```

- Label: three "Not yet checked against CM1" rows (the report's heading counts four, because it counts keys and
  one row has two): stretched levels with cloud; `lspgrad = 1`, compared only in
  a 15-minute bubble and a column; and the semi-slip ground with the surface layer, not compared in a storm on this
  kind of grid. On the uniform 1 km supercell the drag alone gave nine of ten statistics within 1.3 % of CM1's
  (release notes).
- With this semi-slip ground (`bbc = 3`, `sfcmodel = 1`) an imported namelist must have `cecd = 1`. The import
  refuses `cecd = 2` and `3` there, because every imported run also writes the 10 m wind swath, which is written
  for constant drag only: such a run would stop before its first step. (The message words it as "only
  cecd = 1 (constant drag coefficient) is implemented".) On the free-slip ground of CM1's own file the key does nothing and any value is
  accepted. A deck may declare `cecd = 2` or `3` and the plan admits it
  (`cartesian-surface-enthalpy-flux-over-water` does); no run with them has been made on a GPU.
- Turbulence is CM1's prognostic TKE scheme (`sgsmodel = 1`), as in every tornado run. The two-part near-wall
  models (`sgsmodel = 3` and `4`) are admitted and have no storm comparison with CM1.
- `maintain_rh` matters on a real sounding: with it off, a warm bubble can be too dry to reach free convection.
  CM1 users know it as the `maintain_rh` logical in `init3d.F`.
- Size: this is the 250 m grid above. For the grid under the 4 m nest (100 m, 640 x 768 x 80) see
  [`NESTING.md`](NESTING.md).

### A squall line with repeating sides

```sh
gpuwm-cm1 import cm1/squall_line/namelist.input --out runs/squall
gpuwm-cm1 estimate runs/squall
```

- CM1's squall-line namelist imports and runs: 300 x 60 x 40 at 1 km, 3 h, about 1.2 GB.
- Label: "Not yet checked against CM1" for the repeating north and south sides. The momentum and sound-wave faces
  match CM1's; the turbulence scheme still treats a repeating side's edge as an open side's, where CM1 does not.
  That one difference is why the squall-line cards are greyed in guided mode.

### The moving domain

CM1 moves the whole domain at a fixed speed (`imove = 1`, `umove`, `vmove`) so that a small area holds a moving
storm. GM1 does the same. The rules, each checked against the 0.1.0 wheel:

- `umove` and `vmove` are read as CM1 reads them, as 32-bit numbers: `umove = 11.4` runs as 11.399999618530273 and
  the report says so.
- Solid side walls with a moving domain are refused, exactly where CM1 stops.
- An adaptive time step (`adapt_dt = 1`) with a moving domain is refused. CM1's supercell file moves its domain, so
  use a fixed `dtl`.
- The time step: the ground position is checked against CM1's own running sum after every step, and the check is
  exact. A step that binary numbers cannot hold exactly, such as 0.6 s, makes that sum round in its last place
  (10.8 m against 10.799999999999999 m after six steps) and stops the run with a message about the domain's
  position. Steps of 6, 3, 1.5, 0.75, 0.5, 0.375 and 0.25 s pass. (The message suggests reducing the grid or the
  device budget; the remedy is the time step.)
- `estimate` and `cartesian --plan` run a storm-track check and print the frame and bubble position that keep the
  storm inside. On an analytic sounding it is a warning; with a sounding file, a deck whose storm leaves is refused
  at the plan with the remedy.

The storm-following domain (`do_adapt_move = .true.`, CM1's own controller) finds the storm from smoothed surface
pressure and re-aims the frame as it goes:

```sh
gpuwm-cm1 import cm1/supercell/namelist.input --out runs/following
gpuwm-cm1 edit runs/following --set do_adapt_move=.true. --set adapt_move_frq=900
gpuwm-cm1 cartesian runs/following/deck.toml --plan --output following-plan.json
```

- `adapt_move_frq` must be a whole number of output intervals (`tapfrq`, 900 s in CM1's file): the import refuses
  anything else.
- The ground must be free-slip (CM1's default), and the four sides must be all open or all repeating. The import
  does not check these two; the run does. That is why the last line above plans the imported deck: a refusal there
  (`CM1-CARTESIAN-ADAPT-MOVE-SURFACE`, `CM1-CARTESIAN-ADAPT-MOVE-BOUNDARIES`) costs nothing.
- Not with an adaptive time step.
- Status: the controller's arithmetic is compared with CM1's; no whole storm run with it has been compared with
  CM1.

## Going further

- Nests, including the 4 m tornado run and its decks: [`NESTING.md`](NESTING.md) and `examples/tornado-4m/`.
- A finer box inside a WRF or ArWen run: [`WRF-DOWNSCALING.md`](WRF-DOWNSCALING.md) (`gpuwm-cm1 chain`).
- Every deck key, the restart format, profiling and CUDA graphs: [`GM1-CARTESIAN-RUNTIME.md`](GM1-CARTESIAN-RUNTIME.md).
- What is and is not checked against CM1: `gpuwm-cm1 doctor --science`, and the release notes.
