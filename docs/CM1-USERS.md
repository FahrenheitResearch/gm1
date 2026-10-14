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
| Uniform grids, 100 m to 2 km | Runs. |
| `stretch_z = 1`, `3`, `4` | Runs, including levels that stretch from the ground (`str_bot = 0`); compared with CM1 in a supercell ("Stretched levels", below). Levels that stretch from the ground or to the top are listed under "Not yet checked" with what that comparison found. `stretch_z = 2` is refused. |
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

## Stretched levels

CM1's `stretch_z = 1` puts thin layers near the ground and thicker ones aloft, and GM1 runs it as CM1 does. These
commands take CM1's supercell from 40 even 500 m layers to 53 layers, 100 m deep at the ground and 500 m deep above
9.3 km, check the result, price it and run it:

```sh
gpuwm-cm1 import cm1/supercell/namelist.input --out runs/stretched
gpuwm-cm1 edit runs/stretched --set stretch_z=1 --set nz=53 --set ztop=19300.0 --set str_bot=300.0 --set str_top=9300.0 --set dz_bot=100.0 --set dz_top=500.0
gpuwm-cm1 check runs/stretched/deck.toml
gpuwm-cm1 estimate runs/stretched
gpuwm-cm1 run runs/stretched
```

In the GUI the same steps are the Namelist, Settings and Run screens of `gpuwm-cm1 gui --mode advanced`.

**CM1 fixes `nz`; you do not choose it.** The five words describe three parts, and CM1 counts the layers of each
(`param.F:7823-7877`):

| Part | Layers | In the example |
|---|---|---|
| below `str_bot`, layers of `dz_bot` | `str_bot / dz_bot` | 300 / 100 = 3 |
| between `str_bot` and `str_top`, layers whose mean depth is `(dz_bot + dz_top) / 2` | `(str_top - str_bot) / ((dz_bot + dz_top) / 2)` | 9000 / 300 = 30 |
| above `str_top`, layers of `dz_top` | `(ztop - str_top) / dz_top` | 10000 / 500 = 20 |

`nz` must be the sum, 3 + 30 + 20 = 53, and each part must come out a whole number (CM1 allows 1 cm). `zd`, the
bottom of the damping layer, must stay below the new `ztop`. When `nz` is not the sum, the report says which `nz`
fits, and `gpuwm-cm1 check` says the same for a deck:

```text
Refused (1)
  nz = 40: nz = 40 does not fit these stretched levels: CM1 needs nz = 53 here (3 layers of dz_bot
      below str_bot, 30 stretched layers between str_bot and str_top, 20 layers of dz_top above
      str_top; param.F:7823-7877).
    Fix: Set nz = 53.
```

When a part is not a whole number of layers the row names it and the nearest heights that are ("`str_bot = 250` m is
not a whole number of `dz_bot = 100` m layers (the nearest are 200 and 300 m)").

**Levels that stretch from the ground.** `str_bot = 0`, CM1's common setting, has no even layers at the ground: the
levels thicken from the first one. CM1's hurricane cases use 59 such levels, 50 m at the ground and 500 m above
5.5 km. On the supercell:

```sh
gpuwm-cm1 import cm1/supercell/namelist.input --out runs/stretched-ground
gpuwm-cm1 edit runs/stretched-ground --set stretch_z=1 --set nz=59 --set ztop=25000.0 --set str_bot=0.0 --set str_top=5500.0 --set dz_bot=50.0 --set dz_top=500.0 --set zd=20000.0
```

The report lists such a column under "Not yet checked against CM1", with what the comparison below found: it
runs; with ground friction all ten statistics are within their bands; with a free-slip ground max vertical
vorticity came out 16 % high.

The same holds at the top (`str_top = ztop`). Such columns do not run with parcels (`iprcl = 1`), the budget
outputs or a nest yet; keep three equal layers at each end for those (`str_bot` of at least 3 × `dz_bot`, and
`ztop - str_top` of at least 3 × `dz_top`).

**Levels from a file.** `stretch_z = 3` reads `nz` scalar-level heights and `stretch_z = 4` reads `nz + 1` w-level
heights (the first is 0) from CM1's `input_grid_z`, one height in metres per line. Put the file beside the namelist
and import; the run folder keeps a copy. `ztop`, `str_bot`, `str_top`, `dz_bot` and `dz_top` are not read on these
two, as in CM1.

**`stretch_z = 2` is refused.** CM1 builds those levels by trial and does not make them end at `ztop`, so it has
two model tops, and GM1 has one. To run on the same levels, write CM1's scalar-level heights (`zh` in its output)
into `input_grid_z` and use `stretch_z = 3`.

**How close stretched levels are to CM1.** CM1's supercell on the 53 levels above, with Morrison, two hours, GM1 on an
RTX 4090 against CM1 r21.1; mature-phase means over 3600-7200 s and CM1's twin as in "How close GM1 is to CM1" below.
"5 km" is the w level nearest 5000 m.

| Statistic, on the 53 stretched levels | GM1 / CM1 | Twin / CM1 |
|---|---|---|
| max w | 1.015 | 1.004 |
| max 2-5 km updraft helicity | 0.989 | 1.000 |
| updraft area, w > 10 m/s at 5 km | 1.040 | 1.010 |
| max vertical vorticity | 0.941 | 1.000 |
| min w | 0.997 | 0.999 |
| lowest-level θ′ min | 0.998 | 0.999 |
| cold-pool area, θ′ < -1 K | 0.990 | 1.000 |
| rain area > 1 mm | 1.011 | 1.000 |
| max rain | 1.012 | 1.000 |
| max graupel/hail mixing ratio | 0.996 | 0.996 |

Every statistic is inside its band (10 % for the updraft statistics, 15 % for the rest) and nine are within 5 %; max
vertical vorticity is 5.9 % low. The strongest updraft's column at 5 km is within one cell of CM1's at every 15-minute
output, and max w at 1800 s is 31.30 against CM1's 31.16 m/s. The first rain reaches the ground in the same 5-minute
output in both models. GM1's rain total over the domain is 11 % below CM1's at 30 minutes, when there is only a trace,
5 % below at 45 minutes, within 1.4 % from 55 minutes and 0.4 % below at the end; even levels show most of the early
difference too (7 % and 4 %), so it comes from the microphysics scheme and not from the levels.

On CM1's hurricane column (59 levels that stretch from the ground, 50 m deep there) the same supercell, with Morrison,
was run for two hours in both models. With free slip nine of the ten statistics are within 2.4 % of CM1's and max
vertical vorticity is 15.6 % high, outside its 10 % band (CM1 against itself: 2.0 %); that maximum is at 12 to 14 km,
where the levels are even, in the last half hour, and the two models agree within 1 % until 75 minutes. The same storm
on a column with three equal 50 m layers at the ground, which already ran before, gives the same picture (11.9 %
high), and so does the run with the old ground weights, so this is not the levels at the ground. With surface drag
(`bbc = 3`, `sfcmodel = 1`, `cecd = 1`) all ten are inside their bands, 0.987 to 1.086 of CM1's. In three 100-second
comparisons without cloud on such columns, pressure and density agree with CM1 to 2 parts in a million of their peaks
and the horizontal winds to 3 in 100,000, as closely as on a column with even layers at both ends.

One storm on each column was compared, with Morrison. WSM6 and Thompson were not run on stretched levels.

**Limits on stretched levels.**

- `output_uh`, `output_vort`, `output_shs` and `output_svs` (updraft helicity, vorticity and their swaths) are not
  saved from an imported namelist; the report lists them under "Runs differently from CM1".
- Levels that stretch from the ground or to the top do not run with parcels, the budget outputs or a nest.
- On CM1's hurricane column, which stretches from the ground, the supercell with a free-slip ground has max
  vertical vorticity 15.6 % above CM1's, outside its band. No storm was compared on levels that stretch to the top.
  The report says so for every column that stretches from the ground or to the top.
- In a run with cloud the report also lists every other stretched column under "Not yet checked against CM1", with
  its reason: levels that go higher than the 53 above, other levels, or WSM6 or Thompson. The 53 levels above with
  Morrison, from the stretch words or from `input_grid_z`, and the hurricane presets' own levels in their own storm,
  which CM1 ran at 8 km, carry no row.
- Guided mode's **Customize this storm** offers even levels only.
- `stretch_x` and `stretch_y` are refused.

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
gpuwm-cm1 quicklook runs/supercell --radar-site 40 -20 --radar-tilt 0.5
gpuwm-cm1 quicklook runs/supercell --look updraft-time-height
```

`quicklook` draws the pictures the GUI shows (reflectivity at 1 km, spin near the ground, updraft at 5 km, the wind
swath, the lowest level's θ′ and wind, rain, and the time series) into `runs/supercell/quicklook/`, on the CPU at
nice 19 with no GPU. `--slice`, `--section` and `--column` draw a horizontal slice, a vertical section or a column
skew-T of any `cm1out` field. `--compare OTHER [OTHER ...]` draws the run beside up to five other run folders:
side by side at one storm time (`--time S`, default the latest time all reached; `--box KM` for the box each storm
is centred in), over time, and their soundings beside what each grew, into `runs/supercell/quicklook/compare/`.
`--look updraft-time-height` and `--look spin-time-height` draw the whole run as one picture of height against time
(the strongest rising air, and the strongest spin, near the storm's main updraft, one column per saved time;
`quicklook/updraft-time-height/still.png`); every pass draws them too.
Every picture says SIMULATED, the GM1 version, the microphysics, the grid spacing, the
real height of the level and the storm time. The swath's updraft helicity is CM1's `shs` (or `shs2`) when the run
saves it; otherwise it is the strongest of each saved time's 2-5 km updraft helicity, CM1's `uh` or, without it,
summed from w and the winds by the picture tool with CM1's calcuh method, and the picture says which. The spin
colours are fixed by the grid spacing, or by the lowest level's height when that is greater (a hurricane's swirl
film by +/-0.01 per second). `--force` draws every picture again (after an update that changes how a look is
drawn).

```sh gpu
gpuwm-cm1 render runs/supercell --list
gpuwm-cm1 render runs/supercell --product supercell
gpuwm-cm1 render runs/supercell --product vorticity --height 1000
```

`render` draws the same run in a second style, with GM1's native renderer, from the run's newest restart checkpoint
(`--list` lists the restarts and `--restart N` picks another). Its products: `reflectivity` (the column maximum) and
`reflectivity-level` (at a height, 1 km unless `--height` says otherwise), `updraft` (the strongest in each column),
`vorticity`, `cold-pool` and `scalar --field NAME` (one level: `--level K`, 1 the lowest, or `--height M`), the
three-panel `supercell`, two blended pictures (`vortex-composite`, `storm-fusion`), `updraft-helicity` (2-5 km),
and the swaths CM1 keeps every step (`swath-wind`, `swath-vorticity`, `swath-pressure`, `swath-helicity`). A restart
is the state saved at the end of a segment and at each restart interval, not at every saved time, so every picture
says "restart checkpoint at T+..., not a history frame", with the lowest and highest value in it, its colour scale
and how each field was made: reflectivity is CM1's `calc_refl10cm` on the CPU (Morrison runs; a WSM6 run writes its
own to the history frames only, so `render` refuses it), vorticity is CM1's dv/dx - du/dy from the restart's winds
(with their boundary halo at the domain's edge, as CM1 forms it), and updraft helicity is CM1's 2-5 km sum. The pictures go to `runs/supercell/quicklook/native/`, on the CPU with no GPU.
In the GUI they are under Native pictures on the Results screen (advanced mode).

The two radar looks (`radar-storm-relative`, `radar-velocity`) are one low scan of a simulated Doppler radar: the
model's wind along the beam, storm-relative or ground-relative, beside the reflectivity the same beam sees (the
reflectivity look's own: the scheme's dBZ, or CM1's `calc_refl10cm` for Morrison), with the beam path of the 4/3
earth, a 0.95 degree beam, 250 m gates and 0.5 degree radials. The radar stands 40 km south-east of the storm in each
picture (the storm is followed from one saved time to the next) until `--radar-site X Y` fixes it on the ground (km,
in the run's own x and y at the start; `storm` puts it back) and `--radar-tilt` sets the scan's angle; both draw the
radar looks of every saved time again. `--radar-motion U V` (m/s east and north, ground-relative) sets the storm
motion the storm-relative look takes out; by default it is the domain's motion when the domain moves with the storm,
else the Bunkers right mover of the starting air, or its 0-6 km mean wind where the 0-6 km shear is under 10 m/s.
The pictures say what they are not: no attenuation, clutter or velocity folding, and no fall speed.

### Pictures of CM1's own output

`quicklook` draws CM1 netCDF output too (`output_format = 2`): CM1's own, or a GM1 run's `cm1out` folder. It only
reads the files.

```sh
gpuwm-cm1 quicklook path/to/cm1run                    # a folder of cm1out_NNNNNN.nc, or a CM1 run folder
gpuwm-cm1 quicklook path/to/cm1run/cm1out.nc          # one file that holds every saved time
gpuwm-cm1 quicklook path/to/cm1run --slice dbz --height 1000
gpuwm-cm1 quicklook path/to/cm1run --film reflectivity
```

Each look whose fields the files hold is drawn for each saved time into `quicklook/` beside the files
(`cm1out-quicklook/` beside a single `cm1out.nc`, or `--out DIR`). `--slice`, `--section`, `--column`, `--fields`,
`--look` and `--film` work as on a GM1 run, with the file's own variables under their own names.

- The grid is the file's: `xh`, `xf`, `yh`, `yf`, and the real height of every level from `zh` and `zf`, stretched
  levels included. Files that are not drawn say why: a grid stretched in x or y (`stretch_x`, `stretch_y`), a run with
  terrain, a 2-D or axisymmetric run (`ny = 1`, `axisymm = 1`), files on two different grids in one folder, and
  files cut into one piece per process (`output_filetype = 3`).
- A look that needs a field the files do not hold is not drawn and says which, with the CM1 switch that writes it:
  "cm1out_000001.nc has no vertical velocity (w) (CM1 writes it with output_w = 1)." The time series charts what
  the files hold.
- θ′ is CM1's `thpert`; without it, `th` minus the base state `th0` (`output_basestate`); with neither, `th` minus θ of
  the domain's four corner columns at the first saved time, and the lowest-level look and the column skew-T say so.
- Reflectivity is the file's own `dbz`. Without it, a Morrison run's is CM1's `calc_refl10cm` from the file's rain,
  snow and hail, when the file also has `rho` (`output_rho = 1`) and the namelist gives `ihail`; otherwise the look
  says what is missing.
- With the run's `namelist.input` beside the files (or `--namelist FILE`), a hurricane run (`iinit = 7`) gets the
  hurricane looks, and Morrison and WSM6 runs say hail or graupel as `ihail` does; without one the files are drawn
  as a thunderstorm run, the dense ice is called "graupel or hail", and the command says so.
- A moving domain is placed on the ground from the file's `umove` and `vmove`. The wind swath and the hail and spin
  tracks use CM1's translated swaths (`sws2`, `shs2`, `sgs2`, `svs2`), placed under the domain at each saved time,
  on ground that reaches back to where the domain was at the start. Each file holds them only under the domain at
  its time: what the run added to ground the domain left before the next file is in no file (before the first
  file, none of it is). Their strongest values are the run's, but edges can be stepped and areas smaller, and the
  pictures say so beside them.
- A picture of CM1's output says "CM1 r21.1" where a GM1 run's says the GM1 version, its name ends "CM1 output", and
  the mark reads "GM1 picture": CM1 simulated the storm and GM1 made the picture. A GM1 run drawn from its `cm1out`
  files keeps its own footer and words, and its name ends "from netCDF". Drawn from the frames or from the files, a
  look of a GM1 run shows the same numbers and, apart from the name and a few pixels one colour level apart or on a
  colour step, the same picture. The wind swath and the tracks are the exception, for the reason above: the same strongest values,
  and areas that can come out a few per cent smaller (1,168 against 1,232 km² of hail track on a 2 km run whose
  domain moved 5 km between saved times).

In the GUI, **Runs, Open CM1 output** takes the folder (or the file) and a name; a GM1 run's `cm1out` folder can be
opened the same way. The files stay where they are; the new entry opens with its looks, films and slices drawn on
the computer's processor, and no run controls. A folder the files cannot be drawn from is refused with the reason.

Three looks show the storm in depth, for every frame and every kind of run, hurricanes too:
`--only side-view,water-paths,rotation-axis` draws just these. `side-view` cuts the storm open from west to east
through its core (the strongest w at 5 km, or a hurricane's lowest pressure): the model's own cloud water, rain,
cloud ice and snow, and graupel or hail (whatever the scheme carries), summed through a 20 km slab and on the cut
itself, each on a fixed logarithmic scale. `water-paths` maps each species' column total (`rho q dz` summed over
every level, kg/m²). `rotation-axis` draws, from the lowest level to 10 km, the centre of the circle of radius 2 km
(or three grid spacings) with the most circulation at each height (a hurricane: with the lowest mean pressure, as
its eye is wider than the circle), with its lean. No two centres lie further apart sideways than 1.5 km per km of
the height between them plus 1.5 grid spacings (at least 500 m); where that rule or the least circulation the axis
needs would be broken, the axis stops and says why. It is a picture tool's method, not a CM1 diagnostic, and a
storm that does not rotate gets a plain reason instead of an axis. For one species on its own
along any line, `--section qs --from XA YA --to XB YB` draws CM1's own field.

On a grid of 250 m or finer it also draws the flow near the ground, with
`--only storm-relative-flow,streamwise-spin,cold-pool-depth` to draw just those: the storm-relative flow at 0.5, 3
and 10 km (storm motion from the run's own track, else the Bunkers estimate, and the picture says which), the
horizontal vorticity on the w level nearest 100 m coloured by its angle to the storm-relative wind (it needs levels
100 m apart or closer near the ground), and the depth of the air at least 1 K colder than the median θ at its height.

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
