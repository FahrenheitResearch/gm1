# GM1 for CM1 users

GM1 (`gpuwm-cm1`) is a CUDA port of CM1 r21.1's dynamical core, with column microphysics from ArWen, for one NVIDIA
GPU. It reads your CM1 `namelist.input`, says setting by setting what it runs as CM1 does, runs it, and writes CM1's
own `cm1out` files. This guide covers the command line; every step is also a screen in the GUI's advanced mode
(`gpuwm-cm1 gui --mode advanced`), and every GUI action shows its command ([`COMMANDS.md`](COMMANDS.md)).
Installing: [`INSTALL.md`](INSTALL.md).

Every command block below is run by the release's doc-test, in order, on CM1's own supercell namelist: on an RTX 3080
before the release, and with a stand-in for the GPU in the CPU test suite.

## What maps, and what does not

| CM1 | GM1 v0.1 |
|---|---|
| Uniform grids, 100 m to 2 km; `stretch_z` | Runs. `stretch_z` with cloud is "Not yet checked against CM1". |
| `stretch_x`, `stretch_y` | Refused (every horizontal operator assumes uniform spacing); v0.2. |
| `ptype = 5` (Morrison) | ArWen's WRF-lineage Morrison, not CM1's `morrison.F`. Matches CM1 on its supercell namelist within 5.1 % (below); the report names the one statistic past GM1's 5 % target. |
| `ptype = 7` (WSM6) | Runs with `eqtset = 2`; CM1 forces `eqtset = 1` for WSM6, so the report says "Runs differently". |
| `ptype = 3` (Thompson) | Runs after `--acknowledge-thompson`, with `nt_c = 100.0` (GM1's Thompson always runs that droplet concentration; CM1's own file has 250, which is refused with that fix line): checked only in a 120-s column, on an earlier ArWen. ArWen 2.8.9 corrected a table read in Thompson, so its storm differs from GM1 0.1.0's; it has not been compared with CM1 on this engine. |
| Other `ptype` (NSSL, P3, Kessler, ...) | Refused by name. |
| `iinit` 0, 1, 2, 3, 5, 7, 8, 11, 13 (bubbles, cold pool, tropical-cyclone vortex, line thermal, the gravity-wave and near-surface waves) | Runs as CM1 does; the initial state is CM1's word for word. The vortex needs `iorigin = 2` and `ibalance = 0`, as CM1's own arm does. The bubble, cold pool, line thermal and parcel seed box CM1 sets in Fortran are settings (`gpuwm-cm1 edit --setting`). The random noise of `iinit = 8` and `irandp = 1` is CM1's own, number for number (the same random numbers gfortran's `random_number` gives CM1); import with `--seed N` for GM1's stream and another realization. |
| `iinit` 12 (updraft nudging), 9 (forced convergence), 10 (momentum forcing) | Runs as CM1 does: the push is applied where CM1 applies it (inside the acoustic steps for 12 and 9, in the held wind tendency for 10), with CM1's own strength, size and timing from `init3d.F`, which are settings here (`initialization.xc_wnudge`, `wmax_wnudge`, `t2_wnudge`, ...). As in CM1, a restart turns the push off. Checked against CM1 r21.1 on its supercell namelist ("The initiation options", below). |
| `iinit` 4 (the moist benchmark bubble) | Shown only (`iconly = 1`): the starting state is CM1's word for word, but GM1's run from it does not match CM1's yet (over 15 minutes, both on Morrison, w and theta' differed from the first minute), so a run is refused. |
| `isnd = 5` (analytic) and `isnd = 7`, `17` (`input_sounding`) | Runs; the sounding is read and checked. |
| `isnd = 9` (dry, constant stability) | Runs; the base state is CM1's word for word, and the first 15 minutes matched CM1's (both on Morrison). |
| `isnd` 1, 8, 12-14, 18-23 (CM1's other analytic soundings) | Shown only (`iconly = 1`): the base state is CM1's word for word, but GM1's run from it has not been compared with CM1's yet, so a run is refused. `isnd` 2 and 3 are dry: shown only (`imoist = 0`, `iconly = 1`). `isnd` 6 and 15 are refused, as CM1 has no base state for 6 and 15 needs other constants. |
| `isnd` 4, 10 (a base state with cloud in it) and 11 | Shown only (`iconly = 1`): the base state is CM1's word for word, but GM1's run from it does not match CM1's yet (over 15 minutes, both on Morrison, w and theta' differed from the first minute), so a run is refused. |
| `ibalance` 0, 1, 2 | Runs; with 2 (the anelastic balance) the first 15 minutes matched CM1's. |
| Open-radiative, periodic, `irbc = 2`, rigid walls | Runs; rigid walls and repeating sides are "Not yet checked". |
| Radiation, PBL, cumulus, `sfcmodel` 2-7 | Refused or not coupled. |
| `sfcmodel = 1`, `bbc = 3`, Coriolis, `lspgrad`, `imove` | Runs; some combinations are "Not yet checked". |
| `&param9` output words | Honoured; the words GM1 does not produce are listed (`kmh`, `kmv`, `khh`, `khv`, `rain2`). |
| `&param10` statistics | From the frames at each output time until in-run statistics land. |
| `output_format`, `output_filetype` | Honoured: netCDF or GrADS, one file or one per time. |
| CM1 restart files, MPI, `nodex`/`nodey` | GM1 has its own restarts; one GPU needs no decomposition. |
| Fortran edits (`init3d.F`, `base.F`), heat-flux initiation | Not available. |

Older CM1 keys are read under their r21.1 names (for example `pertcor` as `lspgrad`), and r22.0 keys are judged by
value: a physics key at its r21.1-equivalent value is "Same as CM1 r21.1", at any other value it is refused.

## Import your namelist

CM1's own supercell and squall-line namelists come with GM1; use yours instead of the first line if you have one.

```sh
gpuwm-cm1 examples --cm1 supercell --copy cm1
gpuwm-cm1 import cm1/supercell/namelist.input --out runs/supercell
```

`import` needs no CM1 build. It writes the run folder (`runs/supercell`: your namelist byte for byte, the run's
namelist, the report, the settings CM1 keeps in Fortran, and GM1's deck) and prints the coverage report:

```text
CM1 namelist: cm1/supercell/namelist.input
Settings not in CM1 r21.1: none
Sounding: none needed (isnd = 5)
Result: Runs, with the differences below

Refused (0)
Not read (0)
Runs differently from CM1 (37)
  output_km, output_kh = 1: Not saved: eddy viscosity (kmh, kmv) and eddy diffusivity (khh, khv).
      GM1's turbulence closure does not keep them between steps
  output_dbz = 1: GM1 has no reflectivity operator inside the Morrison step, so frames carry no dbz.
      It is computed after each frame with CM1's calc_refl10cm formula on the CPU by gpuwm-cm1
      cm1out
  statfrq, stat_* (29 words) = 60.0: CM1 writes the &param10 statistics every 60.0 s. GM1 takes them
      from the history frames, every 900.0 s, until in-run statistics land
  rstfrq = -3600.0: CM1 writes no restart files for a negative rstfrq. GM1 writes one restart at the
      end (timax = 7200.0 s) so the run can be continued
  ...
Not yet checked against CM1 (0)
Not needed on one GPU (21)
  ppnode = 128: MPI layout
  ...
Same as CM1 (286)
  ptype = 5: ArWen's WRF-lineage Morrison, not CM1's morrison.F. On CM1's stock supercell, GM1/CM1
      mature-phase ratios are 0.985-1.050 (CM1 against itself: 0.984-1.003); past the 5 % target:
      max vertical vorticity 1.0503
  Grid (15)    records/CM1-SUPPORT-MATRIX.csv
  ...
Output: GrADS (output_format = 1), every time in one file (output_filetype = 1), as the namelist asks.
```

Exit status: 0 runs, 2 refused, 3 runs once acknowledged. The same report is in `runs/supercell/import-report.txt`
and, for scripts, `import-report.json`.

### The six report categories

1. **Refused**: the key, the value, why, and what to change. Every refusal of the namelist is listed in one pass;
   one that depends on an earlier one says "(after fixing the above)".
2. **Not read**: output and cosmetic keys GM1 does not read. The run is blocked until you acknowledge them
   (`--acknowledge-output-keys`, or the tick box in the GUI); the acknowledgment is kept in `run.json` and in each
   receipt.
3. **Runs differently from CM1**: where the value matters and GM1 does something else, with the reason.
4. **Not yet checked against CM1**: settings GM1 runs that have not been compared with CM1 at this value, each with
   its evidence path.
5. **Not needed on one GPU**: MPI layout, CM1's restart-file format and the like.
6. **Same as CM1**: a count per family, each with its evidence.

A refused namelist looks like this (CM1's supercell with `ptype = 26` and `stretch_x = 1`):

```text
Result: Refused: 2 settings must change before this runs

Refused (2)
  ptype = 26: ptype = 26 is NSSL 2-moment (graupel only), which GM1 does not have.
    Fix: Use ptype = 5 (Morrison) or ptype = 7 (WSM6).
  stretch_x = 1: GM1 has no horizontal grid stretching in v0.1 (stretch_x = 1).
    Fix: Use a uniform grid (stretch_x = 0).
```

## Change settings

`edit` changes namelist words (`--set`) and the settings CM1 keeps in Fortran (`--setting`), then imports again. The
difference from your original file is kept in `namelist.diff`. Here: a restart every 30 minutes, netCDF beside the
GrADS files the namelist asks for, and at most 4000 MiB of GPU memory, for a card that also drives a desktop.

```sh
gpuwm-cm1 edit runs/supercell --set rstfrq=1800 --setting output.also_netcdf=true --setting gpu.memory_budget_mib=4000
```

The `--setting` names are `initialization.bubble_center_x_m` and `bubble_center_y_m` (in the namelist's `iorigin`
convention), `zc_m`, `bhrad_m`, `bvrad_m` and `bptpert_k` (bubble and line thermal), `zdep_m` (cold pool),
`random_line_amplitude_k` and `maintain_rh`; `lagrangian.seed_nx`, `seed_ny`, `seed_nz`, `seed_dx_m`, `seed_dy_m` and
`seed_dz_m` (the parcel seed box); `output.also_netcdf`; and `gpu.memory_budget_mib`. They start at CM1's own values
(`init3d.F`), and the GUI's Settings screen shows each beside CM1's name; `--reset-setting` puts one back.

Two of these decide whether a storm forms at all on a real sounding. With `initialization.maintain_rh=true` the
warm bubble keeps the sounding's relative humidity as it is warmed (CM1's `maintain_rh` in `init3d.F`); with CM1's
default, false, its relative humidity falls as it is warmed, and on some soundings it never reaches free convection. And
where a bubble fails either way, CM1's updraft nudging (`iinit = 12`) starts the storm with a sustained push, which
is how the famous-day presets start theirs. Finer grids, stretched levels and nests are in
[`GM1-DECKS.md`](GM1-DECKS.md).

## Check it fits

```sh
gpuwm-cm1 estimate runs/supercell
```

```text
Estimate for supercell on NVIDIA GeForce RTX 3080
  Grid 120 x 120 x 40 at 1000 m (120 x 120 km, 0.58 M cells); 2 h of storm time in 1200 steps of 6 s; morrison-mp10
  GPU memory  about 1.1 GB (0.99 GB-1.2 GB, estimated); free 2.4 GB: fits
  Host memory 0.23 GB locked of 67 GB RAM: fits
  Disk        about 2.0 GB (frames 0.57 GB, netCDF 0.74 GB, restarts 0.66 GB); free 338 GB: fits
  Wall time   about 69 s (52 s-87 s; estimated from a rate measured on this GPU)
  Storm track evaluated: closest approach to an open side 37 km
  - 8.3 GB of your GPU is in use by other programs (under WSL2 this includes the Windows desktop, ...)
```

The estimate compares with the memory actually free: a Windows desktop keeps 4-5 GB of the card. After each run the
measured speed is kept (`~/.cache/gpuwm-cm1/rates.json`), and `gpuwm-cm1 estimate RUNDIR --measure` runs a few steps
to measure it now. What fits, estimated from measured runs (Morrison, 80 levels):

| Grid spacing | Largest uniform area | Card |
|---|---|---|
| 1 km, 500 m | every shipped CM1 convection namelist (0.8 GiB measured) | 8 GB and up |
| 250 m | 64 km (3.5 GB measured); 100 km on a headless 8 GB card; 120 km on 12 GB | 8-12 GB |
| 100 m | 60 km on 16 GB; 78 km on 24 GB | 16 GB and up |

## Run, stop and continue

```sh
gpuwm-cm1 run runs/supercell --until 1h
gpuwm-cm1 resume runs/supercell
gpuwm-cm1 status runs/supercell
```

`run` waits for the GPU if another run holds it, integrates into `segments/001`, and prints its progress. `--until`
ends it early (seconds, `90min` or `1.5h`); `resume` continues from the newest restart to `timax` in a new segment,
or to a later `--until` to extend the run. A paused or stopped run continued this way is bit-identical to one that
never stopped, swaths included. `--detach` returns at once, and the run keeps going when the terminal closes.

To stop a run early and keep what it has done:

```sh
gpuwm-cm1 import cm1/supercell/namelist.input --out runs/paused
gpuwm-cm1 edit runs/paused --set rstfrq=1800 --setting gpu.memory_budget_mib=4000
gpuwm-cm1 run runs/paused --detach
gpuwm-cm1 pause runs/paused
gpuwm-cm1 status runs/paused --wait
gpuwm-cm1 resume runs/paused --until 1h
```

`pause` stops the run right after its next restart is written, so it can be continued from exactly there. `stop`
stops at the end of the current step; `resume` then continues from the last restart written before it. Both work
from any computer that sees the run folder (they write `control/pause` and `control/stop`). `status --wait` answers
once the run has finished, paused or stopped.

## CM1's output files

```sh
gpuwm-cm1 cm1out runs/supercell
```

`cm1out` converts the run's frames into CM1's own files in `runs/supercell/cm1out/`, following `output_format` and
`output_filetype`: here GrADS (`cm1out_s.dat` with its `.ctl` descriptors) and, because of the edit above,
`cm1out.nc`, laid out as `writeout_nc.F` writes it: `xh`, `yh`, `zh`, `xf`, `yf`, `zf` in km, `time` in seconds,
CM1's variable names and units, and CM1's global attributes, plus `gm1_*` provenance attributes and a
`gm1_not_produced` list. `cm1out_stats.nc` carries CM1's statistics names at each output time. For Morrison, `dbz`
and `cref` are computed from each frame with CM1's `calc_refl10cm` on the CPU. In the GUI, the writer runs beside the
model as frames appear.

Your xarray scripts read it as they read CM1's:

```python
import xarray as xr

ds = xr.open_dataset("runs/supercell/cm1out/cm1out.nc")
print(ds.sizes)
print("max w (m/s) at each time:", ds.w.max(dim=("zf", "yh", "xh")).values.round(2))
print("lowest-level theta' min (K):", float((ds.th - ds.th.isel(xh=0, yh=0)).isel(zh=0).min()))
```

`ncview runs/supercell/cm1out/cm1out.nc` and GrADS's `open runs/supercell/cm1out/cm1out_s.ctl` work as with CM1.

## Pictures

```sh
gpuwm-cm1 quicklook runs/supercell --only reflectivity
gpuwm-cm1 quicklook runs/supercell --slice w --height 5000
```

`quicklook` draws the pictures the GUI shows (reflectivity at 1 km, spin near the ground, updraft at 5 km, the wind
swath, the lowest level's θ′ and wind, rain, and the time series) into `runs/supercell/quicklook/`, on the CPU at
nice 19 with no GPU. `--slice`, `--section` and `--column` draw a horizontal slice, a vertical section or a column
skew-T of any `cm1out` field. Every picture says SIMULATED, the GM1 version, the microphysics, the grid spacing, the
real height of the level and the storm time.

## Clusters and remote GPUs

On a GPU computer you reach over ssh, run `gpuwm-cm1 gui` there and forward its port from your own computer:

```sh cluster
ssh -L 8765:localhost:8765 gpu-box
```

On a cluster, `jobscript` writes a Slurm or PBS script that runs the storm on a GPU node, with the CM1-format writer
and the pictures as two sidecars on the node's CPUs (nice 19, no GPU), so nothing heavy runs on the login node:

```sh
gpuwm-cm1 jobscript runs/supercell --scheduler slurm --time 01:00:00 --partition gpu --write
```

```sh cluster
sbatch runs/supercell/job-slurm.sh
```

The script runs `gpuwm-cm1 run RUNDIR --continue`, so submitting it again continues from the last restart.
`--setup 'module load ...'` adds lines before it, and `--python` names the environment's Python on the node. The GUI
on the login node attaches to the run folder, shows its progress and writes its stop and pause files. The job
script has not yet run on a real Slurm or PBS cluster; please report what you find.

## Reproduce a run

```sh
gpuwm-cm1 reproduce runs/supercell
```

writes `runs/supercell/reproduce/` and a `.tar.gz` with the namelist, sounding, report, settings, deck, receipts,
versions and the commands that run it again. The GUI also keeps `commands.log` in each run folder: every command it
ran, so a GUI session replays from a terminal.

## The guided presets

The GUI's guided mode starts from five presets, each a CM1 r21.1 namelist, fitted to your GPU and a time budget:

```sh
gpuwm-cm1 preset list
gpuwm-cm1 preset show supercell-250m
gpuwm-cm1 preset fit supercell-250m --budget 1h
```

`preset fit` shrinks the area, then coarsens one rung, then shortens the storm time until the run fits in 95% of the
free GPU memory, half the RAM and half the free disk within the budget; it never changes the physics, boundaries,
sounding or initiation, and it says each change in a sentence. `preset new ID --budget B --root FOLDER --name NAME`
writes the fitted run folder. In advanced mode, **New run > Start from a preset** opens it in the namelist editor.

## From a WRF or ArWen run

CM1 has no lateral boundary forcing; GM1 adds one route that uses it, and one that does not. `gpuwm-cm1 chain`
(advanced mode: **Downscale (3D)**) runs a finer, flat GM1 box inside a WRF or ArWen run, started from one saved time
and driven at its sides by the saved times after it, as WRF's `ndown` does. It admits a box only over flat ground where
the map scale changes by less than 1 % across the box (GM1 takes the box's own cell size on the ground), taking every
cloud and rain field from a WSM6, Thompson or Morrison run or water vapour alone from any run, and says in plain words
why it refuses any other. It has run on one GPU with an idealized ArWen storm, with GM1 driving itself, and with one
real-weather ArWen run (an atmospheric river over the ocean, water vapour only); none has been compared with
observations. The other route takes one column of the run as an `input_sounding` for an idealized storm. Both are described,
with their limits and what has been checked, in `docs/WRF-DOWNSCALING.md`.

## How close GM1 is to CM1

CM1 r21.1's supercell namelist (1 km, 2 h, Morrison), GM1 on ArWen 2.8.9 on an RTX 4090 and an RTX 3080 (the two
cards give the same output word for word) against CM1 r21.1 built from the release tarball (gfortran -O2, OpenMP),
mature-phase means over 3600-7200 s. "Twin" is CM1 against
itself with 0.0001 K added at the lowest level. The rows are `records/GM1-STOCK-DECK-RATIOS.json`'s.

| Statistic | GM1 / CM1 | Twin / CM1 |
|---|---|---|
| max w | 1.003 | 1.003 |
| max 2-5 km updraft helicity | 1.006 | 0.993 |
| updraft area, w > 10 m/s at 5 km | 1.014 | 1.001 |
| max vertical vorticity | 1.050 | 0.984 |
| min w | 1.011 | 1.001 |
| lowest-level θ′ min | 0.988 | 1.002 |
| cold-pool area, θ′ < -1 K | 0.985 | 0.998 |
| rain area > 1 mm | 1.011 | 1.000 |
| max rain | 1.011 | 0.999 |
| max graupel/hail mixing ratio | 1.011 | 0.998 |

The strongest updraft's column at 5 km is within one cell of CM1's at every 15-minute output from 1800 s, and max w
at 1800 s is 29.273 against CM1's 29.348 m/s. Every statistic is inside its band (10 % for the updraft statistics,
15 % for the rest), which is what the report's "Same as CM1" asks of Morrison; max vertical vorticity, at 1.0503,
is just past GM1's 5 % target, and the report says so. GM1 0.1.0, on ArWen 2.8.0, gave 0.986 to 1.046: ArWen 2.8.9 rebuilt its
Morrison kernel, and GM1's output now differs from 0.1.0's by one unit in the last place from about 25 minutes of
storm time, a difference the storm then grows. No RTX 50-series card has run this comparison on ArWen 2.8.9.
Pointwise, w departs from CM1 after the first rain beyond the twin
(nrmsd 0.029 at 1800 s against 0.0005): ArWen's Morrison is WRF's, not CM1's `morrison.F`. The same namelist with
WSM6 is 9-21% stronger than CM1's `ptype = 7`, which runs `eqtset = 1`. Other namelists have not been compared
with CM1 as whole storms; the coverage report says which settings have been checked and where.

### The initiation options

The same namelist with each other `iinit`, GM1 on an RTX 3080 against CM1 r21.1, measured before the move to ArWen
2.8.9 and not repeated on it (on the stock namelist above, the move shifted the ten ratios by at most 0.0135). The
three forcings (updraft nudging 12, forced convergence 9, momentum forcing 10) are CM1's statements word for word
(every word of `get_wnudge`, `convinitu`/`convinitv` and `solve1.F`'s u forcing compiled from CM1's source under
gfortran, host and GPU; for 10, `cosh` is the correctly rounded one of glibc 2.41 and later, which the reference CM1
build calls and a CM1 built against an older glibc does not, so that CM1's u forcing differs by an ulp in a few
words), and through each forcing's window, every minute, the whole w field sits on CM1's own twin
floor until rain forms (for 12, max w 24.406 against CM1's 24.404 m/s at 7 minutes). Every other arm's initial
state is CM1's word for word. Over 2 hours, mature-phase means:

| `iinit` | max w | min w | max rain | rain area | max UH | 5 km updraft column from 1800 s |
|---|---|---|---|---|---|---|
| 12, updraft nudging | 1.018 | 0.990 | 1.018 | 1.003 | 1.000 | within 1 cell to 4500 s; then a third cell grows weaker in GM1 |
| 9, forced convergence | 1.006 | 0.961 | 1.007 | 1.002 | 1.045 | within 1.4 cells at every frame |
| 8, line thermal (CM1's own random numbers) | 1.009 | 0.985 | 0.976 | 1.004 | 0.948 | within 1 cell at every frame |

CM1 against its twin stays within 1% and 0 cells for 12 and 9, so later departures beyond that are GM1's (the same
Morrison difference as above), not the forcing's. The momentum forcing (10) is a squall-line start, and the
tropical-cyclone vortex (7) on this supercell namelist is a multicell case: in both, the single strongest column hops
between near-equal cells in CM1's own twin too.

## Speed on the same case

| Case | GM1 | CM1 r21.1 | Ratio |
|---|---|---|---|
| CM1's supercell, 1 km, 2 h, 0.58 M cells | RTX 3080: 95.7 s | 8 OpenMP threads: 259.6 s | 2.7× |
| The same physics at 250 m, 256 × 256 × 80, 5.24 M cells | RTX 3080: 0.2283 s per step | 8 threads: 1.807 s per step | 7.9× |
| 100 m case, 128 × 128 × 80 stretched, 900 s | RTX 3080: 138.2 s | 12 threads: 1,558.8 s | 11.3× |

All measured. CM1 ran on a Core Ultra 7 270K Plus, gfortran -O2, OpenMP; a pinned, MPI or
`-O3 -march=native` build may be faster. The GM1 times are from before the
latent-heating fix, which added about 7% per step, and on an earlier ArWen (a step on ArWen 2.8.9 takes 1 to 4 %
less time than on GM1 0.1.0's ArWen 2.8.0 on an RTX 4090, and about 3 to 6 % less on an RTX 3080). No comparison with CM1 on a cluster is made.

NCAR's OpenACC work on CM1 (Dennis et al. 2024; NHUG 2025) reports its GPU speed for a dry boundary-layer case with
the fluid solver only; its physics are not yet on the GPU, and CM1 r22.0 contains no OpenACC directives. Official
CM1 with moist physics is CPU-only, and GM1 makes no speed comparison with that work.

## What to send when something is wrong

`gpuwm-cm1 reproduce RUNDIR` and the `.tar.gz` it writes, the output of `gpuwm-cm1 doctor --plain`, and for an
import, the namelist and `import-report.json`.

## An environment from a WRF or ArWen run

One column of a WRF or ArWen history can be the sounding of a CM1 run. GM1 reads the file (any netCDF format,
several times per file, `d01` or a nest), takes the grid column nearest a latitude and longitude, and writes CM1's
`input_sounding` for `isnd = 7`:

- heights from `(PH + PHB) / g − HGT`, each mass level halfway between its two faces;
- θ from `T + 300` (dry θ in WRF and ArWen histories); with `USE_THETA_M = 1` the file's `THM` must agree with it;
- qv from `QVAPOR`; U and V moved to the mass point and turned to earth coordinates with `SINALPHA`/`COSALPHA`;
- the surface line from `PSFC`, `TH2` (or `T2`) and `Q2`; `U10`/`V10` are reported only, since `input_sounding`
  has no surface wind.

The column must reach the namelist's top level, or the file is refused. `wrf-column.json`, next to the sounding,
records the file's SHA-256, the time, the point and how far CM1's own hydrostatic rebuild of the pressure lands from
the file's `P + PB`.

```sh
gpuwm-cm1 sounding times ~/wrf/wrfout_d01_2024-05-20_18:00:00
gpuwm-cm1 sounding from-wrf ~/wrf/wrfout_d01_2024-05-20_18:00:00 --time 2024-05-20_19:00:00 --at=35.2,-97.4 --out ~/wrf/input_sounding
gpuwm-cm1 import cm1/supercell/namelist.input --wrf ~/wrf/wrfout_d01_2024-05-20_18:00:00 --wrf-time 2024-05-20_19:00:00 --wrf-at=35.2,-97.4 --wrf-frame follow --out runs/wrf-storm
```

```text
wrfout_d01_2024-05-20_18:00:00: WRF, domain 1, netCDF-4, 10 x 9 x 40 at 3000 m, Lambert conformal
  record 0: 2024-05-20_18:00:00
  record 1: 2024-05-20_19:00:00
  record 2: 2024-05-20_20:00:00
```

`import --wrf` writes the sounding into the run folder, sets `isnd = 7` in the run's namelist (the change is in
`namelist.diff`) and then imports as usual. A storm in new air moves differently from the one the namelist was set
up for: `--wrf-frame follow` runs the storm-track check on the column's winds and, when the storm would leave the
domain, moves `umove`, `vmove` and the bubble so it stays inside. Without it the namelist's own frame is kept, and
Check says when the storm would leave. In the GUI it is **New run > From a WRF or ArWen run** (click the map of
your domain, look at the skew-T and hodograph, then Import and check).

This is one column as the environment: the same air everywhere in the domain, flat ground, and no weather coming
in from the sides. It is not a downscaling of your run. On a synthetic WRF file whose every column is CM1's own
Weisman-Klemp supercell environment, GM1 builds the same base state as `isnd = 5` within 0.6 K, 0.05 g/kg and
0.25 m/s (the largest gaps at the 12 km tropopause kink, from the file's coarser levels), and its storm's strongest
updraft stays within 7 % of the stock run's at every 15-minute output of the first hour (RTX 3080, measured). For
a GM1 domain driven by your whole run, see **Downscale a WRF or ArWen run (3D)** in advanced mode.
