# Downscaling a WRF or ArWen run with GM1

You have a WRF run, or an ArWen run (ArWen is gpuwm, a GPU WRF that writes the
same `wrfout` history files), and you want a storm-scale GM1 run from it. GM1
offers two routes. They answer different questions, and each has limits you
should know before you trust its output.

| | Route 1: one column as the environment | Route 2: a finer box driven by your run (3D) |
|---|---|---|
| What GM1 takes from your run | The air above one point at one time: temperature, moisture and wind by height | The full 3D state in a box at one time, and the box's four sides at every saved time after it |
| What GM1 runs | An idealized storm in that environment, the same air everywhere, as CM1 runs its own soundings | A finer, flat GM1 grid inside your run's grid, started from your run and steered by it at the sides |
| What it is good for | "What storm does this environment make?" | "What does this storm look like at 200 m?" |
| What it is not | Not your run's storm, fronts, drylines or terrain | Not two-way nesting (nothing goes back to your run), no terrain, no map inside the box (GM1 takes the box's own map scale, one value), not your run's land surface |
| Where | Guided: "Use a sounding from my WRF or ArWen run". Advanced: New run, From a WRF or ArWen run | Advanced: Downscale a WRF or ArWen run (3D). Command line: `gpuwm-cm1 chain` |
| Checked on a GPU | Runs as any idealized GM1 storm | With an idealized ArWen storm run, with GM1 driving itself, and with one real-weather ArWen run (an atmospheric river over the ocean, water vapour only); **not compared with observations** (see "What has been checked") |

Both routes read the history files the same way: classic netCDF (what ArWen
writes) and netCDF-4 (what stock WRF usually writes), one or many saved times
per file, WRF's or ArWen's own variable names. Nothing needs converting first.

## Route 1: one column as the environment

Pick a time and a point on a map of your run. GM1 takes the column there
(winds turned from your grid to east and north, heights above the ground, the
surface values from `PSFC`, `T2` and `Q2`), writes it as CM1's `input_sounding`,
shows its skew-T and hodograph with CAPE and shear, and runs it as an idealized
storm. From the command line it is one option of the importer:
`gpuwm-cm1 import NAMELIST --wrf FILE --wrf-time TIME --wrf-at=LAT,LON --out RUNDIR`.

This is one column's environment, horizontally the same everywhere. The storm
starts from GM1's own warm bubble, not from anything in your run; there is no
terrain and nothing drives the sides. If you want your run's own storm at finer
spacing, use route 2.

## Route 2: a finer box driven by your run (3D)

This is WRF's `ndown` in GM1 terms. You draw a box on your run's grid and pick
how much finer GM1 should be (3, 5, 7 or 9 GM1 cells along each cell of your
run). GM1 then

* starts the box from one saved time of your run: winds, temperature, pressure,
  moisture and every cloud, rain and ice field (or, if you choose water vapour
  only, the vapour alone), moved onto GM1's finer grid (heights above the
  ground column by column, as CM1 moves a restart onto a new grid);
* holds the box's outermost row to your run and relaxes the next four rows
  toward it, moving linearly in time from one saved time to the next, exactly
  as WRF does at a nest's sides (`spec_zone = 1`, `relax_zone = 4`);
* runs CM1's own numerics everywhere else.

### What your run needs

GM1 checks each of these before it makes the run and says in plain words which
one fails and why. Nothing is relaxed to make a box pass.

| check | what GM1 needs | why |
|---|---|---|
| Saved times | at least two, evenly spaced, from one domain (`d01` or `d02`, not both) | the sides move from one saved time to the next |
| Saved often enough | the fastest wind takes longer to cross the box than the time between saved times | otherwise air enters and leaves between two saved times and the sides never see it; 5 min histories suit storm-scale boxes |
| Microphysics | every cloud and rain field: WSM6 (6), Thompson (8) or Morrison (10), with every field it carries saved. Water vapour only (`--moisture vapour-only`): any scheme | GM1 runs these three the way CM1 does; another scheme's fields would be poured into the wrong categories, so WSM3, P3, NSSL and the rest are refused by name. With water vapour only, GM1 takes nothing but the vapour, as WRF itself does by default for real-data runs (`have_bcs_moist = .false.`): clouds and rain start at zero, form in GM1's own scheme (WSM6 or Morrison, `--scheme`), and do not come in through the sides |
| Grid | Lambert, polar stereographic, Mercator or idealized, with square cells | GM1's cells are square |
| Nesting | an odd ratio of 3 or more; GM1's cells are your cells' size on the ground at the box, divided by the ratio | the box's cells must line up with yours |
| The box | at least 3 of your cells from every side of your domain | the transfer reads one cell around the box |
| Flat ground | the ground inside the box varies by less than 50 m (`--terrain-limit`) | GM1 is flat: hills in the box would be left out |
| Map scale | changes by less than 1 % across the box (`--map-factor-limit`) | GM1 has no map inside the box: it takes your cells' true size at the box (your `DX` divided by the box's mean map scale), so a length is off only by how much the scale changes across the box |
| Rotation | the same Coriolis parameter as your run at the box's centre | GM1 uses one value for the box |
| Temperature | dry theta (`T`), or moist theta with `THM` consistent with it | a moist theta read as dry would start moist air too warm |
| Model top | GM1's top under your run's last level in every column of the box | above it, pressure would be held constant with height, out of balance; GM1 lowers its own top to fit |

Two declared limits (`--terrain-limit`, `--map-factor-limit`) can be changed;
the run records what you declared. Declaring a looser limit does not make the
hills or the map appear in GM1; it only means you accept a box GM1 describes
less well.

### Step by step on the command line

Say your WRF run wrote `~/wrf-run/wrfout_d02_2026-05-01_18:00:00` with three
saved times, 5 min apart. First, what is in it:

```sh
gpuwm-cm1 chain inspect ~/wrf-run/wrfout_d02_2026-05-01_18:00:00 --map-png ~/wrf-run/map.png
```

```text
WRF history, 3 saved time(s): 2026-05-01_18:00:00 .. 2026-05-01_18:10:00
  grid 16 x 16 x 30, DX 3000 m, Lambert conformal
  microphysics WSM6 (6) -> GM1 wsm6-mp6
  ground 0 .. 0 m
  map factor 1.0000 .. 1.0000
```

`map.png` shows the ground (grey by height, water dark blue), updrafts of
5 m/s or more (ground-relative) outlined in orange, and the three cells along
each side no box may use. A box is
given as its first cell (counted from 0, west to east and south to north) and
its size, in your run's cells. Check a 5 × 5 box at ratio 3; the folder of
history files works as well as one file:

```sh
gpuwm-cm1 chain check --history ~/wrf-run --window 5,5,5,5 --ratio 3
```

```text
  ok  At least two saved times: 3 saved times
  ok  Every saved time from one grid: One grid, one scheme, one ground
  ok  Evenly spaced saved times: Every 5 min
  ...
  ok  Map scale even over the box: GM1 takes your run's cell size on the ground here (map scale 1.0000); the scale changes by 0.00% across the box (limit 1%)
  ok  Microphysics GM1 has: WSM6 (6) in your run, the same scheme in GM1
  ok  Your run reaches GM1's top: Your run's top is above GM1's
  ok  Saved often enough: The fastest wind crosses the box in 11.4 min; saved every 5 min
Every check passes. GM1 can run this box.
```

`--start`, `--end` and `--every` choose the saved times; `--json` prints every
check with its code and the numbers behind it. Then make the run folder and run
it like any GM1 run:

```sh
gpuwm-cm1 chain new --history ~/wrf-run --window 5,5,5,5 --ratio 3 --out ~/gm1-runs/downscaled
gpuwm-cm1 run ~/gm1-runs/downscaled
```

`chain new` reads every saved time it needs and writes an ordinary run folder:
the physics and numerics come from a CM1 namelist (the 1 km supercell's unless
you give `--namelist`), imported the way any namelist is, with the grid, clock,
initial state and sides set by the chain. The prepared state and side tables
are in `RUNDIR/chain`, with a receipt of what was read and admitted. Check,
Estimate, the GUI's Run and Results screens and `gpuwm-cm1 cm1out` all work on
it. A downscaled run is not continued from a restart: to go longer, make a new
box that covers the longer time.

When a check fails, the line says why. For example, the largest flat box in
California's Central Valley in a real ArWen run of an atmospheric river, saved
hourly (the worked example below uses the same run):

```text
  NO  Flat ground: Ground varies by 96 m, over the 50 m limit. GM1 is flat and would leave the hills out. Move the box over flatter ground or water.  [GM1-CHAIN-TERRAIN]
  NO  Saved often enough: The fastest wind crosses the box in 25.6 min, sooner than the 60 min between saved times: air could enter and leave unseen. Save more often, or draw a bigger box.  [GM1-CHAIN-CROSSING]
```

### Step by step in the GUI

Switch to Advanced and open **Downscale (3D)**:

1. **Your run.** Type the path of a history file or a folder of one domain's
   files, and press Open. The screen says what the run is and draws its map.
2. **The box and the time.** Drag on the map to draw the box (or type its first
   cell and size), pick how much finer, the start and end, and how often the
   sides are updated. The map redraws at the start time, so you can put the box
   on the storm.
3. **Can GM1 run this box?** Choose Moisture (every cloud and rain field, or
   water vapour only; the screen starts at water vapour only when GM1 does not
   run your run's scheme) and, for water vapour only, GM1's microphysics. Check
   lists every check with Passes or Does not pass and the reason, then what the
   run would take on your GPU and disk.
4. **Make the run.** Once every check passes, Make the run prepares the box and
   opens the new run's Check screen. Start it from there as any run.

Every button has Show command: the exact `gpuwm-cm1 chain` line it runs.

### Making a WRF or ArWen run GM1 can downscale

* Use WSM6 (`mp_physics = 6`), Thompson (8) or Morrison (10).
* Save histories often: every 5 min for a storm-scale box
  (`history_interval = 5`), with `P`, `PB`, `PH`, `PHB`, `U`, `V`, `W`, `T`, every
  microphysics field, `HGT`, `MAPFAC_M`, `XLAT`, `XLONG`, `PSFC`, `T2` and `Q2`.
* Put the storm over flat ground or water. GM1 takes the map scale at the box,
  so any latitude works as long as the scale changes by less than 1 % across the
  box: a smaller box, or a projection whose true latitude is nearer the box,
  keeps it even.
* An idealized run (`ideal.exe`, or ArWen's own idealized cases) is flat and
  unmapped already and downscales as it is.

## A worked example: an atmospheric river, offshore

A real-weather run that GM1 can downscale as it is: the ArWen control run of the
atmospheric river that reached California on 25 December 2025 (3 km cells on a
Lambert map, 300 × 270 cells, P3 microphysics, saved every hour from 00:00 to
03:00 UTC). Most boxes in it are refused, each for a reason: over the Sierra the
ground is not flat; in the Central Valley any box flat enough is crossed by the
wind in under half an hour, sooner than the hourly saves. Over the ocean, 80 km
and more off the coast west of San Francisco, a large enough box passes every
check.

On the command line (`SIERRA` is the run's `wrfout` folder):

```sh example
gpuwm-cm1 chain inspect $SIERRA --map-png ~/sierra-map.png
gpuwm-cm1 chain check --history $SIERRA --window 6,40,36,66 --ratio 3 \
    --start 2025-12-25_01:00:00 --moisture vapour-only --scheme morrison-mp10
```

```text
  ok  GM1 cells nest exactly: 3 GM1 cells of 1014.79 m per cell of your run, on the ground
  ok  Flat ground: Ground varies by 0 m (limit 50 m)
  ok  Map scale even over the box: GM1 takes your run's cell size on the ground here (map scale 0.9854); the scale changes by 0.07% across the box (limit 1%)
  ok  Earth's rotation matches: Rotation of latitude 37.3
  ok  Microphysics GM1 has: Your run used P3 (50). GM1 runs Morrison (10) and takes only your run's water vapour: its clouds and rain start at zero and form in GM1
  ok  Saved often enough: The fastest wind crosses the box in 68.0 min; saved every 60 min
  ...
Every check passes. GM1 can run this box.
```

Without `--moisture vapour-only` the same box is refused on the microphysics
alone: GM1 does not run P3, and its ice has no counterpart among GM1's schemes.
Then:

```sh example
gpuwm-cm1 chain new --history $SIERRA --window 6,40,36,66 --ratio 3 \
    --start 2025-12-25_01:00:00 --moisture vapour-only --scheme morrison-mp10 \
    --output-every 300 --out ~/gm1-runs/sierra-offshore
gpuwm-cm1 run ~/gm1-runs/sierra-offshore
```

In the GUI (Advanced, **Downscale (3D)**): open the `wrfout` folder, type the
box as first cell east 6, first cell north 40, 36 cells east and 66 cells north
(or drag it on the ocean at the map's lower left), Finer by 3, Start 01:00:00,
End 03:00:00; Moisture starts at water vapour only because GM1 does not run P3;
keep Morrison; Check, then Make the run.

What you get: 108 × 198 × 40 GM1 cells of 1015 m on the ground (110 × 201 km,
36.4-38.2 N, 125.3-124.0 W), a 6 s step, 2 h from 01:00 to 03:00 UTC. On an
RTX 3080 it took 87 s and peaked at 1.95 GiB of GPU memory (GM1's estimate
beforehand was 1.4 GB); with a history every
5 min of the box it writes about 2.3 GB. What to expect, measured on that run
(`src/cm1core/records/GM1-ARWEN-CHAIN.md` section 9.2):

* The box starts with no cloud or rain (ArWen's were 2 % of the water in the
  box). GM1's Morrison scheme makes its own: half of ArWen's amount after
  15 min, all of it after 30 min; the first rain reaches the sea after about
  16 min. Over the two hours GM1 rains 2.2 mm on the box against ArWen's
  3.1 mm.
* The first minutes carry a pressure adjustment (the moved state is not
  rebalanced); it halves within minutes and settles within about an hour.
* GM1's updrafts, at 1 km, are stronger than ArWen's at 3 km: up to
  21 m/s ground-relative, against 9 m/s ground-relative.

This is idealized physics (CM1's 1 km supercell namelist: no radiation, and no
heat or moisture from the sea surface) in a real-weather box: not a reconstruction of the
day, and not compared with observations.

## What has been checked (route 2)

On one GPU, an RTX 3080 (`src/cm1core/records/GM1-ARWEN-CHAIN.md` section 8):

* **A steady hold.** Two identical saved times: the box's outer rows kept their
  values to the last bit, and the inside moved only by GM1's own settling.
* **A 200 m box in an ArWen supercell.** ArWen ran its Weisman-Klemp supercell
  at 1 km with WSM6; GM1 ran a 24 km box at 200 m on the storm for 30 min. The
  forced rows stayed 2-5 times closer to ArWen than the inside, where the 200 m
  storm goes its own way; in the part of the box inside the forced rows (where
  only GM1's own numerics act), the air-mass budget closed to 3e-5 of the mass
  and the water budget to 0.3 % of the water. GM1's pressure changed faster
  than ArWen's (3.2 times, against a target of 2): GM1's own 1 km run changes as
  fast. Like the next case, the first minute carries a short adjustment, gone
  within about 70 s. This run found two defects, both fixed: GM1's top above
  the run's last level gave a pressure burst 15 times ArWen's, and the
  outermost row drifted from your run (by up to 6 K in temperature after
  30 min) until it was reset to your run at the end of every step, as WRF
  does.
* **A 200 m box in GM1's own 1 km run.** The same picture, and the pressure
  change 1.6 times the driver's; the first minute carries a short adjustment
  (the moved state is not rebalanced), gone within about 70 s.

* **A 1 km box in a real atmospheric river** (the worked example below): a
  real-weather ArWen run, over the ocean, water vapour only. The relaxed rows
  stayed closer to ArWen than the inside for wind, temperature and vapour (the
  pressure differed from ArWen's by about 0.15 hPa everywhere in the box);
  inside the relaxed rows the air-mass budget closed to 1e-5 of the mass and
  the water budget to 0.02 % of the water over 2 h; GM1's pressure changed
  hour to hour at 0.9 times ArWen's rate. GM1 made its own clouds, reaching
  ArWen's amount after 30 min, and 62 % of ArWen's rain in the first hour and
  77 % in the second.

None of these has been compared with observations or with a finer WRF run.
Treat a real-weather box as a GM1 run in a real-weather setting, not a
reconstruction of the day.

## Roadmap

In the order they are planned (record section 7):

1. **Mapped grids.** GM1 already takes the box's mean map scale (so boxes away
   from the true latitude are admitted); next, map-factor terms inside the box,
   so a box whose scale changes by more than 1 % can run.
2. **Terrain.** Chain into GM1's terrain-following grid, so a box over hills
   keeps them.
3. **Land surface.** Take skin temperature, soil, land use and roughness from
   your run instead of the namelist's surface.
4. **A box that follows the storm.** Move the box along the storm's track.
5. **Longer runs.** Continue a downscaled run from a restart.
6. **Balance and cost.** Rebalance the moved state's pressure (removing the
   first-minute adjustment), damp the top as WRF does, move the transfer to
   GM1's Rust tools, and read only the side strips of each saved time.
