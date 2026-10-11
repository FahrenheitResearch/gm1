# gpuwm-cm1 release notes

## 0.1.1

A release of fixes for Windows. The model, the engine pin (gpuwm 2.8.0 and gpuwm-data 2.8.0) and every number in
the tables are those of 0.1.0, and everything under "0.1.0" below describes 0.1.1 too. Verified on Windows 11 with an
RTX 3080.

### Fixed on Windows

- **GM1 opens your browser.** 0.1.0 opened a File Explorer window (showing Documents) instead of the browser, and
  the link had to be copied from the console. 0.1.1 opens the link in the Windows default browser. The console says
  "Opened" only when that worked, and otherwise prints the link for you to open.
- **Opening GM1 again goes to the running storm.** A page that was just opened goes straight to the storm that is
  running, and the first screen and Runs say "A storm is running" with a **Watch it** button. Starting GM1 while it
  is already open brings its page up in the browser again.
- **The link survives a restart.** A tab or a bookmark from an earlier start keeps working: GM1 keeps the link's key
  with your runs folder, readable by you only, so a link stays good until you ask for a new one with `gpuwm-cm1 gui
  --new-token`. A page opened with an old key, or with none, says "This link is out of date" and how to get the
  right one.
- **Folders are shown as Windows paths**, the way File Explorer names them
  (`\\wsl.localhost\Ubuntu-24.04\home\YOUR-NAME\gm1-runs`), in the console, the doctor, Runs and Results, with an
  **Open folder** button. The installer ends by naming the folder your storms will be in.
- **Storm sizes, Delete and Keep pictures only.** Runs shows the size of each storm and the total. **Delete**
  removes a storm. **Keep pictures only** removes its large model files and keeps the pictures; such a storm can no
  longer be continued. Both ask first, naming the storm and its size, and both are refused while the storm is
  running or while GM1 is still drawing its pictures.
- **The doctor's disk line names the Windows drive.** Under WSL2 it gives the free space of the Windows drive that
  holds Ubuntu's disk as well as the Linux figure, and judges by the smaller of the two.
- **Smaller wording fixes.** The "Starting" sentence goes once the storm is running, and "Stopping at the next step"
  goes once it has stopped. The per cent done agrees with the storm time shown beside it. The first card no longer
  says "about 2 minutes on most GPUs" above your own GPU's time. The installer says "Updating" when it updates.

The running-storm page, the kept link, the storm sizes, Delete and Keep pictures only work the same way on Linux.

### How to update

Run the install line again: on Windows the PowerShell line, on Linux the `curl` line (`docs/INSTALL.md`, "Update
and uninstall"). With pip: `pip install --upgrade gpuwm-cm1`. Close GM1 and let a running storm finish first; the
installer stops with a message while one runs. Your storms stay where they are. If a File Explorer window that 0.1.0
opened is still on the screen, close it.

### Known limitations

The known limitations of 0.1.0 all apply to 0.1.1 unchanged; this release fixes none of them. They are listed under
"0.1.0" below ("Known limitations", "Not in v0.1" and "Further known limits"). In short:

- Morrison (`ptype = 5`) and WSM6 (`ptype = 7`) run. Thompson (`ptype = 3`) runs only after an acknowledgment and its
  output is unverified on this release's engine. NSSL and every other scheme are refused by name.
- 21 of CM1's 23 shipped namelists are refused; the supercell and the squall line run.
- Some guided storms are greyed, each with its reason: the squall-line presets, the 100 m supercell, and any storm
  that needs more memory than the card has free.
- GM1 does not check whether another program is already using the GPU.
- Surface drag runs with `cecd = 1` only.
- Restarts do not cross GM1 versions: finish a storm with the version that started it.
- On Windows, closing the GM1 window closes the GUI, never a running storm. Shutting down, restarting or sleeping the
  computer, or `wsl --shutdown`, stops a storm.
- Reaching the GUI from Windows is measured in WSL2's default network mode only.

## 0.1.0

GM1 (the `gpuwm-cm1` package) runs CM1 r21.1's idealized storm cases on one NVIDIA GPU (Linux or WSL2), on uniform
grids from 100 m to 2 km, with CM1-format output and a local web GUI. Bring a CM1 `namelist.input`: GM1 says, setting
by setting, what it runs as CM1 does, runs it, and writes CM1's own output files. The column physics run through
ArWen, the Apache-2.0 GPU weather engine published as the `gpuwm` package. `Development Status :: 3 - Alpha`.

0.1.0 is the first release. The fidelity and speed tables below were measured on its code with gpuwm 2.7.4, before
the engine pin moved to 2.8.0; with gpuwm 2.8.0 the fidelity runs give the same words (see "Engine pin").

gpuwm-cm1 is an independent port. It is not affiliated with or endorsed by the University Corporation for
Atmospheric Research (UCAR) or the NSF National Center for Atmospheric Research (NCAR).

Start with `docs/QUICKSTART.md` (a first storm in the GUI) or `docs/CM1-USERS.md` (your own namelist). Install:
`docs/INSTALL.md`: one line in PowerShell on Windows (it sets up WSL2 and Ubuntu, and a GM1 shortcut), one line in
a Linux terminal, or `pip install gpuwm-cm1`, which installs CuPy for CUDA 13 and NVRTC with it. When something does
not work: `docs/TROUBLESHOOTING.md`, `gpuwm-cm1 doctor --plain` (one fix line per problem) and `gpuwm-cm1 doctor
--report` (one block to paste into a bug report).

### Supported microphysics

- **Morrison (`ptype = 5`)**: runs; on CM1's own supercell namelist it matches CM1 r21.1 within a few per cent
  ("How close to CM1", below).
- **WSM6 (`ptype = 7`)**: runs, through ArWen's WSM6. The coverage report says "Runs differently": CM1 forces
  `eqtset = 1` for WSM6, and GM1 runs `eqtset = 2` with WSM6's heating converted to constant volume.
- **Thompson (`ptype = 3`)**: runs only after an acknowledgment (`--acknowledge-thompson`). Its one check, a 120-s
  column against WRF v4.6.1's own Fortran, was made on an earlier ArWen release; gpuwm 2.8.0 changes Thompson's
  numerics, so that check has not yet been repeated on the engine this release is pinned to. Treat Thompson output
  as unverified.
- **WDM6 and aerosol-aware Thompson**: not offered. CM1 r21.1 has no `ptype` for either.

**Engine pin.** 0.1.0 requires gpuwm 2.8.0 and gpuwm-data 2.8.0 (`>=2.8.0,<2.8.1`). Before a run, GM1 checks that the
installed ArWen files for the selected scheme match the SHA-256 digests recorded for gpuwm 2.8.0, and refuses the
scheme if they differ. Status of each scheme on that pin:

| Scheme | Pinned to gpuwm 2.8.0 bytes | GPU revalidation on 2.8.0 |
|---|---|---|
| Morrison (`ptype = 5`) | yes | passed on an RTX 4090: CM1's supercell namelist (from the command line, and in the GUI with a stop and a continue), an `iinit = 12` namelist and a famous preset give the same words as with gpuwm 2.7.4 on the same card, and the ten statistics against CM1 r21.1 are unchanged (0.986-1.046). Not yet repeated on an RTX 5090. |
| WSM6 (`ptype = 7`) | yes, including the 256-level capacity tier | passed on an RTX 4090: CM1's supercell namelist with `ptype = 7` gives the same words as with gpuwm 2.7.4. A 237-level column with the 256-level capacity profile is admitted and completes a step; its numerical and resource measurements are still to be made. |
| Thompson (`ptype = 3`) | yes | needs a new measurement (2.8.0 changes its numerics). |
| WDM6, aerosol-aware Thompson | yes | none planned; not offered. |
| YSU, MYNN, RRTMG, New Tiedtke | yes | none planned; no GM1 run couples them. |

- **Every other scheme** (NSSL, P3, Kessler, Goddard, GSR, RE87, ISHMAEL) is refused by name.

### Known limitations

- **Some guided presets are greyed** ("not offered") with the measured reason: the squall-line presets until the SGS
  closure's repeating-side families match CM1, the 100 m supercell (in 0.1.0 it stops early on a domain-position
  check; being fixed), and any preset that needs more memory than the card has free.
- **`gpuwm-cm1 run` and the GUI do not check whether another program is already using the GPU.** Run on a card with
  enough free memory. (The low-level `gpuwm-cm1 cartesian --gpu-uuid` path does check.)
- **The first run on a machine compiles GPU kernels** (about 10 s) without printing a notice.
- **Surface drag** (`bbc = 3`, `sfcmodel = 1`) runs with `cecd = 1` (constant drag coefficient) only. The import
  refuses `cecd = 2` and `3`, which 0.1.0 does not implement; set `cecd = 1`.
- **NSSL microphysics is refused.** It is the first item for v0.2 (it blocks 17 of 21 real namelists collected for
  this release).
- **21 of CM1's 23 shipped namelists are refused** (LES, hurricane, 2-D, single-column, DNS and terrain cases); the
  supercell and squall line run.
- **Slurm and PBS job scripts** are written but have not yet run on a real cluster.
- **6 CPU tests fail on hosts without AVX-512.** In the source tree,
  `tests/test_cartesian_restart.py::test_published_frames_and_restarts_are_the_pre_profiling_bytes` (its six profiler
  and publication-mode variants) compares a CPU run's published files with digests recorded where numpy ran its
  AVX-512 kernels. numpy's AVX2 and SSE kernels round some float32 results differently, so on an AVX2-only CPU the
  second frame's words differ (and with SSE only, the first frame's too). The GPU path is unaffected.
- **The GPU path has been tested on RTX-class cards only** (RTX 3080, 4090, 5070 Ti and 5090); data-centre GPUs
  have not been tested.
- Further limits are listed under "Further known limits" below.

## Starting a storm, the famous presets and real-weather downscaling

- **CM1's other ways to start a storm.** Every CM1 r21.1 `iinit` arm was audited against GM1, and the 3D arms GM1
  lacked or ran differently were added, word for word with CM1's source:
  - `iinit = 12`, **updraft nudging** (Naylor and Gilmore 2012; `init3d.F:1370-1396`, `misclibs.F:539-585`,
    `sound.F:564-590`), which about half of the real CM1 namelists collected for this release use;
  - `iinit = 9`, forced convergence (Loftus et al. 2008), and `iinit = 10`, momentum forcing (Morrison et al. 2015);
  - `iinit = 7`, CM1's tropical-cyclone vortex, word for word at time zero;
  - `iinit = 8` and `irandp = 1` now use CM1's own random numbers (the gfortran stream `init3d.F:427-440` seeds), so
    they match CM1 at time zero instead of "running differently"; `iinit = 3`, 5 and 11 put their perturbations
    where CM1 does when a namelist does not say otherwise.
  The words CM1 keeps inside `init3d.F` (the nudging centre, strength, size and time window; the convergence and
  momentum-forcing words) are settings under CM1's own names with CM1's values as defaults, in the Settings
  screen's "Updraft nudging", "Forced convergence" and "Momentum forcing" groups. Against CM1 r21.1 on the stock
  1 km deck with `iinit = 12`, max w agrees to 0.002 m/s through 7 minutes. `iinit = 4` (the moist benchmark) and
  CM1's other analytic soundings (`isnd` 1-4, 8, 10-14, 18-23) are built as CM1 builds them but shown only
  (`iconly = 1`): GM1's run from `isnd` 4, 10 and 11 and from `iinit = 4` does not match CM1's yet, and its run from
  the others has not been compared with CM1's yet. A run starts from `isnd` 5, 7, 17 or 9 (whose first 15 minutes
  matched CM1's).
- **The nine famous-environment presets are offered.** None makes a storm from CM1's warm bubble on a grid a GPU
  fits, so they start with CM1's updraft nudging, with CM1's own words (a 10 m/s target in an ellipsoid 20 km across, fading
  from minute 15, gone at minute 20), centred where the storm-track check recommends. All nine made a storm on the
  RTX 3080 at 1 km (80-158 s each). Two were run in CM1 r21.1 as well (Norman, 20 May 2013 and 3 May 1999): CM1
  makes the same storms, max w within 1 %, in the same column, and just as short-lived. Each card
  says the storm is started by a gentle push of rising air, what its storm did in the run behind its picture, and
  that it is the day's sounding in an idealized storm, not a reconstruction of that day's storms.
- **Downscale (3D) on real weather.** The chain takes the history's cell length on the ground (its map scale over
  the box) instead of requiring a map scale within 1 % of 1, and can drive the box with water vapour
  only (WRF's `have_bcs_moist = .false.`) under GM1's own microphysics when GM1 does not run the history's scheme.
  With the same limits otherwise, a box over the ocean off California in the ArWen control run of the 25 December
  2025 atmospheric river passes every check, and ran for 2 h on the RTX 3080 (`docs/WRF-DOWNSCALING.md`, "A worked
  example"). Its outermost row equals WRF's target each step; the relaxation zone does not pull pressure toward
  the driving run as well as it does wind, temperature and moisture (recorded, not hidden). Boxes over the Sierra
  and the Central Valley are refused, each with its reason.
- **The live nest holds its outermost row** to the parent each step, as WRF's `spec_bdy_final` does.
- **The looks, independently checked.** 1,501 numbers from the looks were recomputed from the netCDF output alone
  and all agree. Clouds from above uses CM1's dry-air density; Morrison runs keep CM1's rain and hail swaths
  (`output_srs`, `output_sgs`), as CM1 does; the environment look of a downscaled run does not say the air is the
  same everywhere; the looks' "about N s" estimates come from measured costs.

## The GUI's looks, starter storms and WRF or ArWen input

- **The GUI's design.** Near-black glass panels, the ink and hairline scales, a cyan accent and focus ring, violet
  for "new". Every text on every screen measured at WCAG AA against what is actually painted behind it, at 1440 and
  400 px.
- **More looks** on Results, from any run, with no extra setup, each with its own "What am I looking at" and a line on
  what the grid spacing can show: cold air near the ground and gust fronts, rain so far, clouds seen from above (a
  simplified visible-light picture from the model's cloud water and ice, lit by a low sun; not satellite data), a
  slice through the strongest updraft, hail near the ground so far, the strongest near-ground spin so far, the
  storm's environment (skew-T, hodograph, CAPE, CIN, shear, helicity, storm motion), air rising into the storm (runs
  that follow air parcels) and the swirl near the ground (a film). Guided Results opens on the four favourites;
  the rest sit in one "More looks" row. Drawn on the CPU: all ten per-frame looks take 3.8 s per frame at 1 km on one
  core.
- **Save as film** for any look that changes with time: an animated PNG always, an MP4 with the optional extra
  (`pip install "gpuwm-cm1[film]"`, imageio-ffmpeg, whose separately licensed ffmpeg program is run, never bundled),
  else a GIF. The 1 km swirl film (217 pictures) takes about a minute.
- **More starter storms**, in four groups: Start here (the quick 1 km supercell stays first), Explore (make your own
  environment with two sliders, storms that split, multicells, a storm in almost calm air), Other storm types (a
  line of storms, a gust front, a warm bubble growing into a cloud at 250 m, the squall lines) and Famous
  environments (nine public-domain NWS balloon soundings of well-known days, each "this day's sounding in an
  idealized storm, not a reconstruction of that day's storms"). Every offered card ran on an RTX 3080 for its picture
  and time. The famous cards start their storms with CM1's updraft nudging (above): CM1's warm bubble makes none on a
  3080-sized grid, and CM1 r21.1 itself makes none from the bubble on the 20 May 2013 deck either (the same updraft to
  0.01 m/s).
- **An environment from a WRF or ArWen run** (the column route): open a `wrfout` or ArWen history, pick a time and a
  point on a map of 0-6 km shear or CAPE, and GM1 writes CM1's `input_sounding` from that one column (destaggered,
  winds turned to earth directions, theta from T or THM, heights from PH+PHB), previews its skew-T and hodograph, and
  runs it as an idealized storm. `gpuwm-cm1 sounding times|from-wrf` and `gpuwm-cm1 import NAMELIST --wrf FILE`; in
  guided mode the card "Use a sounding from my WRF or ArWen run". One column, the same everywhere: no terrain, no
  boundary forcing.
- **Downscale a WRF or ArWen run (3D)**, in advanced mode only: the ArWen chain (WRF's ndown in GM1's terms)
  reads WRF histories as well as ArWen's; `gpuwm-cm1 chain inspect|check|new`; a screen to choose the box on a map,
  the ratio and the time span, with every admission check in plain words. Its outermost row is held to the driving
  run each step as WRF's `spec_bdy_final` does. **Verification status:** host-tested, and four device runs on the
  RTX 3080 (a steady hold, a box in an ArWen idealized supercell history, a box in a GM1 1 km run, and the
  real-weather box above). `docs/WRF-DOWNSCALING.md` has both routes, the limits and the roadmap for terrain and
  mapped coordinates.

## Running CM1 namelists

- **CM1's own storm, within a few per cent.** Each scheme's constant-pressure latent heating reaches CM1's
  constant-volume equations as CM1's does, and CM1's supercell namelist gives the ten mature-phase statistics at
  0.986-1.046 of CM1 r21.1's, where CM1 against a copy of itself nudged by 0.0001 K gives 0.984-1.003 (measured).
- **CM1's SGS start-up**: `ramp_sgs`/`ramp_time` are honoured, and step 1's TKE starts as CM1's does.
- **Periodic sides**: the momentum and acoustic faces of repeating north-south or east-west sides match CM1's.
- **`gpuwm-cm1 import`**: any CM1 r21.1 namelist, with no CM1 build. Older CM1 keys are read under their r21.1
  names; r22.0 keys are judged by value. The six-category coverage report lists every refusal in one pass. `ptype`
  5 is Morrison, 7 is WSM6, 3 is Thompson after an acknowledgment; any other scheme is refused by name.
- **`gpuwm-cm1 edit`**: change namelist words and the settings CM1 keeps in Fortran (the bubble, cold pool, line
  thermal, parcel seed box), then check again; the difference from your file is kept.
- **Run control**: `run`, `stop`, `pause`, `resume`, `status`, a per-GPU queue, crash detection that knows which host
  a run is on, and `jobscript` for Slurm and PBS with the output writer and pictures as sidecars on the GPU node's
  CPUs. The wind swaths and ground canvases are carried in restarts, so a paused and resumed run is bit-identical to
  one that never stopped (measured on the RTX 3080 in all 37 carriers).
- **`gpuwm-cm1 cm1out`**: CM1's own output files from a run's frames. netCDF laid out as `writeout_nc.F` writes it
  (CM1's dimensions in km, names, units and 71 global attributes) or GrADS with CM1's descriptors, following
  `output_format` and `output_filetype`; `cm1out_stats`; `cm1out_pdata` for parcels; and for Morrison, `dbz` and
  `cref` from CM1's `calc_refl10cm` on the CPU. A t = 0 frame is written. A CM1 user's xarray script ran unchanged
  on real CM1 output and on GM1's.
- **`gpuwm-cm1 estimate`**: GPU memory, locked host memory, disk and wall time against what is free now (a desktop
  display can hold several GB of the card), a storm-track check, and "Measure speed".
- **`gpuwm-cm1 gui`**: a local web app on 127.0.0.1 with a token, in two modes over one engine.
  - *Guided* (the default for a first run): the starter presets (a 1 km supercell and the supercell at 250 m; the
    supercell at 100 m and the squall lines at 1 km and 250 m are greyed in 0.1.0), each fitted to your GPU's free
    memory and a time you pick (about 10 minutes,
    about 1 hour, overnight); one Start; a live picture; results on four looks (reflectivity, spin near the ground,
    updraft at 5 km, the wind swath), each with a "What am I looking at" panel. A preset that cannot run here is
    greyed with the reason.
  - *Advanced*: the namelist editor, the coverage report, every setting, the estimator's numbers, run control with
    Continue or extend, job scripts, output choices, slices and a column skew-T, the reproduce bundle, and "Show
    command" beside every action. The mode is remembered per user and never changes a run folder.
- **`gpuwm-cm1 quicklook`**: the viz 2.0 pictures from each frame, on the CPU.
- **`gpuwm-cm1 doctor --plain`**: is this computer ready, in plain words with one fix line per problem.
- **`gpuwm-cm1 examples --cm1`**: CM1's supercell and squall-line namelists, shipped unchanged under UCAR's notice.
- **Packaging**: a `manylinux_2_28` wheel for Python 3.12-3.14, built in a container; no CUDA toolkit or
  `CUDA_PATH` needed.

## How close to CM1

| Case | What was compared | Result |
|---|---|---|
| CM1's supercell namelist, Morrison (`ptype = 5`), 1 km, 2 h | ten mature-phase statistics (3600-7200 s) against CM1 r21.1 `ptype = 5` | 0.986-1.046; CM1's twin 0.984-1.003. Max w 4.100 / 29.273 m/s at 900 / 1800 s against 4.102 / 29.348. The strongest updraft's column at 5 km is within one cell of CM1's at every 15-min output. |
| The same, pointwise | w after the first rain | departs from CM1 beyond the twin (nrmsd 0.029 at 1800 s against 0.0005): ArWen's WRF-lineage Morrison is not CM1's `morrison.F`. |
| The same namelist with WSM6 (`ptype = 7`) | against CM1 `ptype = 7` | 9-21% stronger. CM1 forces `eqtset = 1` for WSM6; GM1 runs `eqtset = 2` with WSM6's heating converted to constant volume. The report says "Runs differently". |
| Thompson (`ptype = 3`) | one 120-s column | Runs only after an acknowledgment. |
| CM1's supercell namelist with surface drag (`bbc = 3`, `sfcmodel = 1`, `cecd = 1`) | against CM1 r21.1 with the same namelist | Drag is claimed only with `cecd = 1`, CM1's constant drag coefficient: nine of the ten mature-phase statistics within 1.3% of CM1's (0.987-1.013), max vertical vorticity 1.091; the strongest updraft's column at 5 km within one cell of CM1's at every 15-min output; the drag slows the lowest-level mean wind as CM1's does, to 0.0001 m/s for the first 30 minutes. Drag moves CM1's own ten statistics by 2% at most on this case. The report still says "Not yet checked against CM1". `cecd = 2` and `3` are refused by the import (see "Not in v0.1"). |
| Stretched levels with cloud, `lspgrad = 1`, rigid walls, `irbc = 1`, repeating sides | | "Not yet checked against CM1" in the report. |

## Speed on the same case (measured)

| Case | GM1 | CM1 r21.1 | Ratio |
|---|---|---|---|
| CM1's supercell, 1 km, 2 h, 0.58 M cells | RTX 3080: 95.7 s | 8 OpenMP threads: 259.6 s | 2.7× |
| The same physics at 250 m, 256 × 256 × 80, 5.24 M cells | RTX 3080: 0.2283 s per step | 8 threads: 1.807 s per step | 7.9× |
| 100 m case, 128 × 128 × 80 stretched, 900 s | RTX 3080: 138.2 s | 12 threads: 1,558.8 s | 11.3× |

CM1: a Core Ultra 7 270K Plus, gfortran -O2, OpenMP. GM1:
before the latent-heating fix, which added about 7% per step. After the fix, the imported stock namelist runs its 2 h
in 70-72 s on the RTX 3080 from launch to exit. NCAR's OpenACC work on CM1 covers the dry solver; official CM1 with
moist physics is still CPU-only, and GM1 makes no speed comparison with the GPU work.

## What fits (estimated from measured runs; Morrison; 80 levels)

| Grid spacing | Largest uniform area | Card |
|---|---|---|
| 1 km, 500 m | every shipped CM1 convection namelist (0.8 GiB measured) | 8 GB and up |
| 250 m | 64 km (3.5 GB measured); 100 km on a headless 8 GB card; 120 km on 12 GB | 8-12 GB |
| 100 m | 60 km on 16 GB; 78 km on 24 GB | 16 GB and up |

`gpuwm-cm1 estimate` uses the memory actually free on your card.

## Not in v0.1

| | Why |
|---|---|
| `stretch_x`, `stretch_y` | Refused: every horizontal operator assumes uniform spacing. Planned for v0.2. |
| NSSL, P3, Kessler, Goddard, GSR, RE87, ISHMAEL microphysics | Refused by name. NSSL is the first v0.2 item (it blocks 17 of 21 real namelists collected for this release). |
| `iinit` 4 (the moist benchmark), `isnd` 4, 10, 11 | Shown only (`iconly = 1`): the starting state is CM1's, but GM1's run from it does not match CM1's yet. `iinit` 7, 9, 10 and 12 (updraft nudging) run. |
| `isnd` 1, 8, 12-14, 18-23 (CM1's other analytic soundings) | Shown only (`iconly = 1`): the starting state is CM1's, but GM1's run from it has not been compared with CM1's yet. |
| Radiation, PBL, cumulus schemes, `sfcmodel` 2-7 | Refused or not coupled. |
| `cecd` 2 and 3 with `sfcmodel = 1` (wind-dependent drag) | Refused by the import: only `cecd = 1` (constant drag coefficient) is implemented in 0.1.0; set `cecd = 1`. |
| `eqtset = 1` dynamics | GM1 runs `eqtset = 2` only. |
| Reading or writing CM1 restart files | The formats differ. GM1's own restarts continue GM1 runs. |
| Fortran edits (`init3d.F`, `base.F`), heat-flux initiation | The bubble, cold pool, line thermal and parcel seed box are settings instead. |
| LES, hurricane, 2-D, single-column, DNS and terrain namelists | 21 of CM1's 23 shipped namelists are refused (the supercell and squall line run). |
| Native Windows, macOS, more than one GPU, MPI | Linux and WSL2 on one NVIDIA GPU. |
| GPUs older than compute capability 7.5 (V100), drivers older than R580 | CUDA 13 does not support them. |
| Submitting batch jobs from the GUI | The GUI writes the job script for you to submit. |
| 3-D views, WebGL | v0.2. Films of the picture looks are available now (Save as film). |
| `thompson-aerosol-mp28` | Not offered: it reaches routines adapted from Numerical Recipes, whose licence does not permit redistribution, pending a legal review; no CM1 `ptype` selects it. |

## Further known limits

- **The squall-line presets are greyed** ("not offered") until the SGS closure's repeating-side families
  match CM1; CM1's squall-line namelist imports in advanced mode with those rows under "Not yet checked".
- **Statistics come from the frames.** `cm1out_stats` has CM1's names at each output time, not at `statfrq`.
- **Not written yet:** `kmh`, `kmv`, `khh`, `khv` and `rain2`; the report lists them.
- **Parcels restart with each segment**: they are not carried in restarts, the parcel `dbz`, `qsl` and `qsi` are not
  filled, and there is no t = 0 parcel frame.
- **Restarts do not cross GM1 versions.** A restart written by an earlier GM1 is refused by Continue, so finish a
  run with the version that started it.
- **The job script has not run on a real Slurm or PBS cluster yet.**
- **Estimates:** wall times on cards GM1 has not run on are scaled by memory bandwidth ("rough"). The storm-track
  check places CM1's supercell about 10 km west of where GM1 runs it, so its recommended frame and bubble are
  optional. Host memory for the pictures is not priced (about 3.3 GB per 39 M-cell frame for Morrison reflectivity).
- **Gallery pictures** come from 10-minute fitted runs, and the 100 m card shows the 250 m picture.
- **WSL2**: reaching the GUI from Windows is measured in WSL2's default network mode only.
- **Presets:** the famous presets' cards were made from 1 km runs (their 250 m bases need about 9.7 GB of GPU
  memory). The fitted time estimates of two starter presets were off by 25-53 % and the gust front's 30 % high;
  the famous presets' wall-time estimate is 51-57 % low for Wilmington and Jackson (the other seven within 10 %). The
  Wilmington card starts one storm, not a line. CM1's calm-air option (`iwnd = 0`) is refused by the run-start check,
  so the calm presets use 2 m/s of shear.
- **Drawing the looks:** all ten looks cost about 8 s per frame at 5 M cells on one core, and up to 15 s for a mature
  storm's frame (reflectivity's melting-layer step, CM1's `calc_refl10cm`), against a 15 s aim; no 100 m frame was
  timed.
- **3D downscaling** has run on one real-weather box, offshore and flat; its relaxation zone leaves pressure about as
  far from the driving run as the interior.
- **Old glibc:** a CM1 built with glibc 2.40 or older rounds `coshf` differently, so its `iinit = 10` forcing differs
  from GM1's (and from a newer CM1 build) in the last bit of some words.
- **`nz > 80` with WSM6** needs a capacity overlay (`docs/ARWEN-WSM6-CAPACITY-256-AUTHORITY.md`).

## Capabilities of the `cartesian` door (for GM1 decks)

`gpuwm-cm1 cartesian DECK` runs GM1's own TOML decks, which reach capabilities the CM1 importer does not offer.
`gpuwm-cm1 doctor --science` prints the per-key version of this table.

| Capability | Status |
|---|---|
| Uniform-Cartesian dycore: RK3, `psolver = 3` acoustics, 5th-order advection, SGS/TKE, dry-mass correction, upper damping | Validated against CM1 at operator level, and on CM1's supercell namelist as a whole storm |
| ArWen WSM6 column physics | Validated at the column seam against WRF |
| Morrison | Matches CM1 `ptype = 5` on CM1's supercell namelist (above); other namelists not checked |
| Initialization selectors | 30 of 32 selector values checked against CM1 at time zero |
| Lateral boundaries: open-radiative, periodic, `irbc = 2`, `roflux`, `nudgeobc`, `irdamp = 2`, `hrdamp` | Validated against CM1; rigid walls and `irbc = 1` not checked |
| Coriolis, beta plane, adaptive time step | Validated against CM1 |
| Parcels (`parcel.F`) | Validated against CM1 |
| Budgets, pressure decomposition | Device-tested; not compared with a CM1 run end to end |
| Restarts | Bit-identical pause and resume on the GPU |
| Live nests, two-part SGS on stretched decks, Lagrangian extras | GM1-only; not part of the GUI release |
| Simulated reflectivity | WSM6: on the GPU, ArWen's 10 cm operator. Morrison: CM1's `calc_refl10cm` on the CPU in `cm1out` |

## Licensing notes

- **gpuwm-cm1** is Apache-2.0 (`LICENSE`).
- **CM1 r21.1** is the mathematical authority, used under UCAR's release grant, reproduced in `NOTICE`, which must be
  kept. CM1's supercell and squall-line namelists ship unchanged as examples under that grant. No other CM1 source,
  table, executable or runtime asset is bundled.
- **ArWen** (the `gpuwm` package, an installed dependency, not bundled) is Apache-2.0.
- **Shipped soundings** are NOAA/NWS rawinsonde observations (U.S. Government works), as recorded in
  `cm1core/data/soundings/PROVENANCE.md`.
- **NVIDIA CUDA**: the wheel bundles no NVIDIA component; installing it installs NVIDIA's own wheels (CuPy's CUDA 13
  toolkit and NVRTC) as dependencies, under the NVIDIA CUDA Toolkit EULA.
- **DejaVu fonts** for the pictures ship with their licence (`cm1core/quicklook/fonts/LICENSE-DejaVu.txt`).

## Authority and attribution

The mathematical authority is the official CM1 r21.1 release tarball, SHA-256
`dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7`. CM1 is the work of George H. Bryan and NCAR. The
port does not mix scientific source from another CM1 tree. ArWen supplies column physics through a measured,
versioned seam.

## Verification

The GPU checks below were made on this release's code with gpuwm 2.7.4, before the engine pin moved to 2.8.0. With
gpuwm 2.8.0 on an RTX 4090, these runs give the same words as with gpuwm 2.7.4 on that card: CM1's stock supercell
(also in the GUI with a stop and a continue), the famous preset, the `iinit = 12` namelist, WSM6, and the seven
fixed-step dynamics cases (74 of 74 frames); the two adaptive-step dynamics cases stay within their recorded bounds
of CM1 (see "Engine pin").

- **Dynamics:** the nine dynamics cases held against CM1 give the same words on the RTX 3080 every time they are run
  (102 of 102 frames).
- **CM1's stock supercell:** a fresh Python 3.12 venv gives the max w series 4.10 / 29.27 m/s at 900 / 1800 s, and so
  does the GUI with a stop and a continue.
- **A famous preset:** Norman, 20 May 2013 with updraft nudging gives the series its card was made from.
- **An `iinit = 12` namelist in the GUI:** CM1's supercell namelist with `iinit = 12`, run through the GUI's advanced
  mode, gives max w within 0.997-1.034 of CM1 r21.1's own run at every saved time.
- **The real-weather chain box:** the offshore box of the atmospheric-river run gives the same bytes as its first run,
  in every field both write.
- **CPU suite:** on a Linux computer without AVX-512, the only failures are the AVX-512 pins under "Known limitations".
- **Walkthrough:** 64 screens at 1440 and 400 px: 0 errors, 0 overflow.
- **Build.** `tools/release/build_manylinux.sh` in `quay.io/pypa/manylinux_2_28_x86_64` (AlmaLinux 8.10, glibc 2.28,
  Rust 1.95.0) on a Linux build host: the sdist, the wheel built from its extraction,
  `auditwheel repair` (the Rust tools and `libcm1_state.so` need nothing newer than glibc 2.28), and
  `tools/inspect_artifacts.py --require-manylinux`: 20 of 20 checks pass. The build is reproducible: two builds
  of one commit give byte-identical wheels and sdists.
- **Install, a CM1 user's first hour** (`tools/release/smoke_cm1_user.sh`, fresh environments, `CUDA_PATH` and
  `CUDA_HOME` unset): on an RTX 3080 under WSL2 with Python 3.12 (venv and pip) and 3.14 (uv): doctor ready, CM1's
  supercell namelist imported, run for its 2 h (about 70 s), converted by `cm1out`, and read by a CM1 user's own
  xarray script; on a Linux computer with no GPU, with Python 3.14 (venv and pip) and 3.12 (uv): doctor names the
  missing driver, import, estimate, the presets and the GUI server work (the GUI process maps no CUDA library), and the 3080 run's
  folder converts and draws there. Every one prints the same max w series: 0, 4.10, 29.27, 49.79, 56.05, 60.34,
  59.85, 58.37, 55.09 m/s at 0-7200 s.
- **Both guides, command by command, for real** on the RTX 3080 (`tools/release/doc_commands.py --mode real`):
  QUICKSTART's 7 commands and CM1-USERS' 22 commands and its xarray block, every one exit 0. The run stopped at 1 h
  and resumed, and the run paused at its 30-min restart and resumed, give the uninterrupted run's max w series. The
  CPU suite runs both guides against the fake runner (`tests/test_docs_guides.py`), with `preset fit` for every preset
  and budget on three GPU fixtures.
- **The GUI, both modes, headless** (Chromium driving the GUI of the installed wheel on the RTX 3080 computer, its
  first launch):
  - beginner: the guided gallery (five cards; the squall lines greyed until the SGS closure is matched, the 100 m card greyed with the free
    memory it needs), Supercell quick look, About 10 minutes, Start: the first picture 18 s after Start, finished in
    73 s, Results on the four looks with their panels;
  - advanced: CM1's supercell namelist imported (exit 0), the coverage report, the namelist and all 583 settings,
    Sounding, Domain and time, Settings (restarts every 30 min, netCDF beside GrADS), Estimate (the recommended GPU
    budget), Start, Stop after the 30-min restart, Continue from it to 2 h, Results (a slice and a column skew-T),
    Commands; the CM1 user script on its `cm1out.nc` prints the uninterrupted run's numbers;
  - 0 console or page errors, 0 failed or outside requests, 0 px of horizontal overflow, no action without its Show
    command, at 1440 and 400 px; 27 screens, 54 screenshots and a contact sheet.
- **CPU suite in the wheel environment** (the wheel in a fresh Python 3.12 venv, a source checkout, no GPU): on a
  Windows 11 computer under WSL2 whose CPU has AVX-512, every test passed but one, a file-time race in the GUI's "is
  this estimate stale" check, since fixed (it could make the Domain screen ask for Estimate again on WSL2). Skips are
  tests that need a GPU, a built CM1 or local evidence that is not distributed.
