# gpuwm-cm1 (GM1)

GM1 runs CM1 (Cloud Model 1, NCAR's research thunderstorm model) on one NVIDIA graphics card, with a point-and-click
web page for your first storm. CM1 users can bring their own `namelist.input`: GM1 says, setting by setting, what it
runs as CM1 r21.1 does, runs it, and writes CM1's own output files. The local web GUI has a guided mode for a first
storm and an advanced mode with every setting.

**Status: alpha** (`Development Status :: 3 - Alpha`). Read [`docs/RELEASE-NOTES.md`](docs/RELEASE-NOTES.md) before
drawing a scientific conclusion from anything it produces.

## What GM1 has done

Each line names the file in this repository that backs it.

- **A tornado on a 4 m grid, on one graphics card.** An idealized supercell on a 100 m grid, with a 4 m two-way
  nest that follows the storm (500 x 500 x 103 cells, lowest level 2 m), on one RTX 5090 using about 28 GB. A
  criterion fixed before the run was scored rates it TORNADO: one vortex tracked for 455 s from T+47:36.9, peak 10 m
  ground-relative wind about 128 m/s (129.0 m/s in the last record before the time step limit was passed), median
  radius of maximum wind 128 m. Read these with it: it is one run, made on a development build that predates the 0.1.0 release, and
  scored after the fact; after 3312 s the nest's time step was too long for its winds, and the run stopped at
  3452.8 s, so nothing after 3312 s is cited; CM1 has no nesting, so there is no CM1 run to compare it with; and
  the headline (its genesis time, the 455 s, the 129.0 m/s record) cannot yet be re-derived from the public datasets
  by someone outside the project. What can be: the public one-record-a-second dataset, scored, gives TORNADO for the
  last 161 s before the limit, and the verdicts page says how to check that.
  [`docs/NESTING.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/NESTING.md),
  [`examples/tornado-4m/`](https://github.com/FahrenheitResearch/gm1/blob/main/examples/tornado-4m),
  [`docs/GM1-TORNADO-VERDICTS-V2.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/GM1-TORNADO-VERDICTS-V2.md).
- **The same criterion says no when the grid cannot show it.** Earlier 20 m nests reached 117 to 121 m/s near the
  ground; the criterion rates them "grid too coarse to judge" or "saved too rarely to judge", not tornadoes. An
  earlier 4 m run, listed before any run was scored, rates TORNADO (360 s, 111.7 m/s). The same verdicts page.
- **CM1's own supercell, within a few per cent of CM1.** Ten mature-storm statistics at 0.985 to 1.050 of CM1
  r21.1's on an RTX 4090 and an RTX 3080, where CM1 against a copy of itself gives 0.984 to 1.003. "How close to CM1", below.
- **2.7 to 11.3 times faster than CM1 on a desktop CPU**, with an RTX 3080: the gap grows as the grid gets finer.
  "Speed on the same case", below.
- **A 39 million cell storm on one card.** 640 x 768 x 80 cells at 100 m (64 x 77 km) on an RTX 5090, the largest
  single domain run so far. `src/cm1core/records/GM1-GPU-RATES.json`.
- **Real weather, downscaled.** A 1 km box inside an ArWen run of the 25 December 2025 atmospheric river off
  California: 2 h in 87 s on an RTX 3080, with `gpuwm-cm1 chain`. WRF histories are read too. One flat box over
  the ocean; not compared with observations. [`docs/WRF-DOWNSCALING.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/WRF-DOWNSCALING.md).
- **Stop a run and continue it without changing a bit.** A paused and resumed run equals the uninterrupted one in
  all 37 fields it carries (RTX 3080). [`docs/RELEASE-NOTES.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/RELEASE-NOTES.md), "Running CM1
  namelists".
- **Nine famous days.** Nine NOAA balloon soundings each made a storm at 1 km on an RTX 3080 in 80 to 158 s. Two
  were also run in CM1, which makes the same storms. The release notes, "Starting a storm".

## Beyond the GUI

The GUI is the easy way in. The decks, `gpuwm-cm1 chain` and `gpuwm-cm1 reproduce` are in the package you already
installed, one command away. The nesting examples, a few small tools and the tornado scorer are in this repository:
clone it beside your install. Much of this has not been compared with CM1; each page says what was checked. Push it, and when something breaks, paste
`gpuwm-cm1 doctor --report` into an issue with the command you ran.

- **GM1's own decks.** A deck is a TOML file that reaches everything the engine has. `gpuwm-cm1 examples` lists the
  39 that ship and `gpuwm-cm1 examples --copy decks` copies them. `gpuwm-cm1 check DECK` reads one strictly,
  `gpuwm-cm1 cartesian DECK --plan --output plan.json` admits and prices it without opening the GPU, and
  `gpuwm-cm1 cartesian DECK --integrate --output receipt.json --history-dir frames --restart-dir restarts` runs
  it. [`docs/GM1-DECKS.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/GM1-DECKS.md) walks through a first deck, lists every example deck with its
  memory and the card it fits, and has recipes for a 100 m storm, 20 m layers near the ground, the near-ground
  physics of the tornado runs, a squall line and the moving domain.
- **Nesting.** A fine grid inside a coarser one, on the same card, following the storm.
  [`docs/NESTING.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/NESTING.md) and the decks of the 4 m run in
  [`examples/tornado-4m/`](https://github.com/FahrenheitResearch/gm1/blob/main/examples/tornado-4m). The start of
  the small example (a 100 m parent, then a 20 m child that moves) has run on GM1 0.1.0 on an RTX
  4090 (installed by pip), in 16.4 GiB. That run was too short to have a vortex for the child to follow.
- **A finer box inside a WRF or ArWen run.** `gpuwm-cm1 chain inspect|check|new`, then `gpuwm-cm1 run`.
  [`docs/WRF-DOWNSCALING.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/WRF-DOWNSCALING.md).
- **A run someone else can repeat.** `gpuwm-cm1 reproduce RUNDIR` writes the namelist, sounding, deck, receipts
  and versions of a run, with the commands that run it again, as one folder and a `.tar.gz`.
- **The 4 m run's data.** Four public datasets (CC-BY-4.0, CM1-format netCDF) of the run named E100R2-PF-4m:
  [the whole nest every 120 s, 2280 to 3360 s](https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-full-nest) (10 files, 6.8 GB),
  [ten chosen records](https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-top10-frames) (0.8 GB),
  [the last five minutes, one record a second](https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-last-5min-every-1s) (24 GB) and
  [the last two minutes, every record](https://huggingface.co/datasets/deepguess/GM1-E100R2-PF-4m-last-2min-every-record) (1307 files, 104 GB).
  Three of the four cover the last minutes, after the time step limit was passed at 3312 s: all ten chosen
  records, all of the last two minutes and 139 of the 301 records of the last five. Their largest winds carry that error and are not
  the numbers quoted above. Nine of the ten whole-nest files are from before it.

```python
# pip install huggingface_hub xarray netCDF4
from huggingface_hub import hf_hub_download
import xarray as xr

path = hf_hub_download("deepguess/GM1-E100R2-PF-4m-full-nest", "cm1out_000028.nc", repo_type="dataset")
ds = xr.open_dataset(path)  # the whole 4 m nest at 3240 s (T+54:00), 729 MB, before the limit was passed
print(ds["sws"].attrs["long_name"], float(ds["sws"].max()), ds["sws"].attrs["units"])
```

## Start here

| | |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | New to storm modelling: install, open the GUI, pick a storm, press Start |
| [`docs/CM1-USERS.md`](docs/CM1-USERS.md) | CM1 users: your own namelist, the coverage report, output, run control, clusters |
| [`docs/INSTALL.md`](docs/INSTALL.md) | Requirements, the installers, pip or uv, updating and uninstalling, building the wheel |
| [`docs/SOUNDINGS.md`](docs/SOUNDINGS.md) | Use your own sounding: the file layouts GM1 reads, how they are converted, how the storm is started, what the numbers mean |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Something does not work: the common problems, each with its fix, and how to ask for help |
| [`docs/COMMANDS.md`](docs/COMMANDS.md) | Every command the GUI's "Show command" can print |
| [`docs/GM1-DECKS.md`](docs/GM1-DECKS.md) | Beyond the GUI: GM1's own decks, the example decks and what each needs, recipes for finer grids |
| [`docs/NESTING.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/NESTING.md) | A fine grid inside a coarser one, and the 4 m tornado run |
| [`docs/GM1-TORNADO-VERDICTS-V2.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/GM1-TORNADO-VERDICTS-V2.md) | Which runs rate as tornadoes, by which rule, and what you can check yourself |
| [`docs/WRF-DOWNSCALING.md`](docs/WRF-DOWNSCALING.md) | A sounding or a finer 3D box from a WRF or ArWen run |

## Install

You need an NVIDIA graphics card (RTX 20 series or newer, 8 GB or more) with driver R580 or newer, on Windows 10 or
11 or on Linux (x86-64).

**Windows**: open PowerShell (Start menu, type PowerShell; not Command Prompt) and paste this line. It sets up WSL2
and Ubuntu if needed, installs GM1 there, and adds a GM1 shortcut to the Start menu and the desktop:

```powershell install
irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1 | iex
```

**Linux** (also Ubuntu in WSL2): open a terminal and paste this line. It installs GM1 with its own Python, no sudo:

```sh install
curl -LsSf https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.sh | sh
```

**With pip** (Linux only, or inside Ubuntu on Windows; pip in Windows PowerShell cannot install GM1). Only if you
already use Python 3.12 to 3.14; otherwise use the line above. On Ubuntu or Debian, first: `sudo apt install
python3-venv`. It installs CuPy and NVIDIA's CUDA libraries too:

```sh install
python3 -m venv ~/gm1
. ~/gm1/bin/activate
pip install gpuwm-cm1
```

Then start it with the GM1 shortcut (Windows) or type `gpuwm-cm1 gui` in a new terminal (Linux): the GUI opens in
your browser. If anything goes wrong,
[`docs/TROUBLESHOOTING.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/TROUBLESHOOTING.md) has the
common problems with their fixes, and `gpuwm-cm1 doctor --plain` checks the computer, one fix line per problem.
[`docs/INSTALL.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/INSTALL.md) covers updating,
uninstalling, the exact tested versions (`constraints-gpu-cu13.txt`), uv, clusters and building from source.

### Common questions

- **Mac?** No. Apple computers have no NVIDIA graphics card.
- **AMD or Intel graphics?** No. GM1 needs an NVIDIA card: RTX 20 series or newer, with 8 GB of memory or more.
- **A laptop?** Yes, if it has such an NVIDIA card.
- **How much disk?** About 7 GB free: 2.5 GB for GM1 and about 4 GB for the first storm (on Windows add about 2 GB for
  Ubuntu, all on drive C:). Finer storms need more; GM1 says how much before you press Start.
- **Is anything sent over the internet?** No. GM1 runs entirely on your computer, has no account and sends nothing
  anywhere; the internet is used only to install and update it. Its page is reachable only from your own computer,
  with the key in its link.
- **Where are my storms?** In the folder `gm1-runs` in your home folder (on Windows, inside Ubuntu:
  [`docs/TROUBLESHOOTING.md`](https://github.com/FahrenheitResearch/gm1/blob/main/docs/TROUBLESHOOTING.md) says
  where).

## What v0.1 does

- **Import** any CM1 r21.1 namelist (older and r22.0 keys are read by value) with no CM1 build:
  `gpuwm-cm1 import namelist.input --out RUNDIR` writes a six-category coverage report (refused, not read, runs
  differently from CM1, not yet checked against CM1, not needed on one GPU, same as CM1) with every refusal of the
  namelist in one pass.
- **Run** it on one GPU with `run`, `stop`, `pause`, `resume` and `status`; a paused and resumed run is bit-identical
  to one that never stopped. `jobscript` writes a Slurm or PBS script.
- **CM1's output files**: `cm1out` writes netCDF laid out as CM1's `writeout_nc.F` writes it, or GrADS with CM1's
  descriptors, following the namelist's `output_format` and `output_filetype`, plus `cm1out_stats` and
  `cm1out_pdata`.
- **Estimate** GPU memory, host memory, disk and wall time against what this computer has free, before a run.
- **Downscale** a WRF or ArWen run: `chain` makes a finer flat GM1 domain driven on its four sides by the saved
  history, and `sounding from-wrf` makes a CM1 `input_sounding` from one of its columns.
- **Reproduce**: `reproduce` writes a run's inputs, receipts, versions and commands as one bundle.
- **The GUI** (`gpuwm-cm1 gui`, on 127.0.0.1 with a token): guided presets fitted to your GPU and a time budget
  (thunderstorms, two hurricanes at 8 km and 4 km made from CM1's own hurricane case, and a storm in the air of a
  sounding file of your own),
  each one adjustable in guided mode ("Customize this storm": grid spacing, levels, area, storm time, picture
  interval, the air, the starting push, random bumps, Earth's rotation, air parcels; `preset ... --custom KNOB=VALUE`
  from a terminal) with the estimate updating as you change it, the live view and results on four quick-looks; or, in advanced mode, the namelist editor, the coverage report, every
  setting, the estimator's numbers, resume and job scripts, with "Show command" beside every action.

Uniform grids from 100 m to 2 km, and CM1's stretched levels (compared with CM1 in a storm), through an imported namelist
(the GUI's guided mode takes any grid spacing you pick or type
for a thunderstorm, and runs its hurricanes at 8 km and 4 km), and
through GM1's own decks also stretched near-ground levels and storm-following nests ("Beyond the GUI", above); Morrison (`ptype = 5`) and WSM6 (`ptype = 7`)
microphysics, Thompson (`ptype = 3`) after an acknowledgment (CM1's own file also needs `nt_c = 100.0` for it). What
is refused and why is in the release notes.

## How close to CM1 (measured)

CM1 r21.1's own supercell namelist (1 km, 2 h, Morrison), GM1 on ArWen 2.8.9 on an RTX 4090 and an RTX 3080 (the
two cards give the same output word for word) against CM1 r21.1 built from the release tarball:

| | GM1 | CM1 r21.1 |
|---|---|---|
| Ten mature-phase statistics (3600-7200 s means) | 0.985 to 1.050 of CM1's on an RTX 4090 and an RTX 3080 (re-measured on ArWen 2.8.9) | CM1 with a 0.0001 K nudge against CM1: 0.984-1.003 |
| Maximum updraft at 900 s / 1800 s | 4.100 / 29.273 m/s | 4.102 / 29.348 m/s |
| Column of the strongest updraft at 5 km | within one grid cell of CM1's at every 15-min output | |

That is one deck, on two cards that give the same numbers. Max vertical vorticity (1.0503) is just past GM1's 5 %
target; every statistic is inside its release band (10 % for the updraft statistics, 15 % for the rest). No RTX
50-series card has run this comparison on ArWen 2.8.9. Every other deck's coverage report says what has and has not been checked against CM1.

## Speed on the same case (measured)

| Case | GM1 | CM1 r21.1 | Ratio |
|---|---|---|---|
| CM1's supercell, 1 km, 2 h, 0.58 M cells | RTX 3080: 95.7 s | 8 OpenMP threads: 259.6 s | 2.7× |
| The same physics at 250 m, 256 × 256 × 80, 5.24 M cells | RTX 3080: 0.2283 s per step | 8 threads: 1.807 s per step | 7.9× |
| 100 m case, 128 × 128 × 80 stretched, 900 s | RTX 3080: 138.2 s | 12 threads: 1,558.8 s | 11.3× |

CM1 ran on a Core Ultra 7 270K Plus, gfortran -O2, OpenMP; the
GM1 times are before a later latent-heating fix, which added about 7% per step, and on an earlier ArWen (a step on ArWen 2.8.9 takes
1 to 4 % less time than on GM1 0.1.0's ArWen 2.8.0 on an RTX 4090, and about 3 to 6 % less on an RTX 3080). Speed is not the point at 1 km;
it is at 250 m and finer. No comparison with CM1 on a cluster is made.

## Design commitments

- The official CM1 r21.1 release tarball is the mathematical authority. Scientific source is never mixed from
  another CM1 tree.
- Production dynamics and physics stay on the GPU. Python orchestrates; ArWen (the `gpuwm` package) supplies the
  column physics through a measured, versioned seam, so whole-model parity with native CM1 is claimed only where it
  was measured.
- Unknown configuration is refused by key and value, and every refusal names the concrete wrong result it prevents.
- Claims are bounded by receipts. Every run records its grid, schedule, sounding, source hashes, step times, device
  memory and final field checks.

## Known limitations

The release notes list them in full ([`docs/RELEASE-NOTES.md`](docs/RELEASE-NOTES.md), "Known limitations"). In
short: some guided presets are greyed with their measured reason (the squall lines); `gpuwm-cm1 run` and the GUI do not check whether another
program is already using the GPU, so run on a card with enough free memory; the first run on a machine compiles GPU
kernels (about 10 s) without a notice; an imported namelist's surface drag runs with `cecd = 1` only; NSSL and other unported microphysics
schemes are refused by name; 21 of CM1's 23 shipped namelists are refused as CM1 ships them (the supercell and squall
line run, and the 3-D hurricane once ten of its settings are replaced with what GM1 has); job
scripts for Slurm and PBS have not yet run on a real cluster; and the GPU path has been tested on RTX-class cards
only. On ArWen 2.8.9, GM1 has run on an RTX 4090 and an RTX 3080 (no RTX 50-series card), and Thompson's
storm, which changed with the engine's corrected table read, has not been compared with CM1.

## Citing

If you use GM1 in published work, please cite CM1, the model it ports:

Bryan, G. H., and J. M. Fritsch, 2002: A benchmark simulation for moist nonhydrostatic numerical models. *Mon. Wea.
Rev.*, **130**, 2917-2928.

## Attribution

CM1 is the work of George H. Bryan and NCAR. `LICENSE` is GM1's licence; `NOTICE` carries CM1's licence terms and
lists the CM1 files and data this distribution bundles, and the notices of the bundled DejaVu fonts. ArWen (the
`gpuwm` package) is a separate dependency, installed from PyPI under its own licence; none of it is bundled here.
gpuwm-cm1 is an independent port, not affiliated with or endorsed by UCAR or NCAR.
