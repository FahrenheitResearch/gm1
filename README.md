# gpuwm-cm1 (GM1)

GM1 runs CM1 (Cloud Model 1, NCAR's research thunderstorm model) on one NVIDIA graphics card, with a point-and-click
web page for your first storm. CM1 users can bring their own `namelist.input`: GM1 says, setting by setting, what it
runs as CM1 r21.1 does, runs it, and writes CM1's own output files. The local web GUI has a guided mode for a first
storm and an advanced mode with every setting.

**Status: alpha** (`Development Status :: 3 - Alpha`). Read [`docs/RELEASE-NOTES.md`](docs/RELEASE-NOTES.md) before
drawing a scientific conclusion from anything it produces.

## Start here

| | |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | New to storm modelling: install, open the GUI, pick a storm, press Start |
| [`docs/CM1-USERS.md`](docs/CM1-USERS.md) | CM1 users: your own namelist, the coverage report, output, run control, clusters |
| [`docs/INSTALL.md`](docs/INSTALL.md) | Requirements, the installers, pip or uv, updating and uninstalling, building the wheel |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Something does not work: the common problems, each with its fix, and how to ask for help |
| [`docs/COMMANDS.md`](docs/COMMANDS.md) | Every command the GUI's "Show command" can print |

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
- **The GUI** (`gpuwm-cm1 gui`, on 127.0.0.1 with a token): guided presets fitted to your GPU and a time budget,
  each one adjustable in guided mode ("Customize this storm": grid spacing, levels, area, storm time, picture
  interval, the air, the starting push, random bumps, Earth's rotation, air parcels; `preset ... --custom KNOB=VALUE`
  from a terminal) with the estimate updating as you change it, the live view and results on four quick-looks; or, in advanced mode, the namelist editor, the coverage report, every
  setting, the estimator's numbers, resume and job scripts, with "Show command" beside every action.

Uniform grids from 100 m to 2 km; Morrison (`ptype = 5`) and WSM6 (`ptype = 7`) microphysics, Thompson (`ptype = 3`)
after an acknowledgment. What is refused and why is in the release notes.

## How close to CM1 (measured)

CM1 r21.1's own supercell namelist (1 km, 2 h, Morrison), GM1 against CM1 r21.1 built from the release tarball:

| | GM1 | CM1 r21.1 |
|---|---|---|
| Ten mature-phase statistics (3600-7200 s means) | 0.986-1.046 of CM1's | CM1 with a 0.0001 K nudge against CM1: 0.984-1.003 |
| Maximum updraft at 900 s / 1800 s | 4.100 / 29.273 m/s | 4.102 / 29.348 m/s |
| Column of the strongest updraft at 5 km | within one grid cell of CM1's at every 15-min output | |

That is one deck. Every other deck's coverage report says what has and has not been checked against CM1.

## Speed on the same case (measured)

| Case | GM1 | CM1 r21.1 | Ratio |
|---|---|---|---|
| CM1's supercell, 1 km, 2 h, 0.58 M cells | RTX 3080: 95.7 s | 8 OpenMP threads: 259.6 s | 2.7× |
| The same physics at 250 m, 256 × 256 × 80, 5.24 M cells | RTX 3080: 0.2283 s per step | 8 threads: 1.807 s per step | 7.9× |
| 100 m case, 128 × 128 × 80 stretched, 900 s | RTX 3080: 138.2 s | 12 threads: 1,558.8 s | 11.3× |

CM1 ran on a Core Ultra 7 270K Plus, gfortran -O2, OpenMP; the
GM1 times are before a later latent-heating fix, which added about 7% per step. Speed is not the point at 1 km;
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
short: some guided presets are greyed with their measured reason (the squall lines, and in 0.1.1 the 100 m
supercell, which stops early on a domain-position check); `gpuwm-cm1 run` and the GUI do not check whether another
program is already using the GPU, so run on a card with enough free memory; the first run on a machine compiles GPU
kernels (about 10 s) without a notice; surface drag runs with `cecd = 1` only; NSSL and other unported microphysics
schemes are refused by name; 21 of CM1's 23 shipped namelists are refused (the supercell and squall line run); job
scripts for Slurm and PBS have not yet run on a real cluster; and the GPU path has been tested on RTX-class cards
only.

## Citing

If you use GM1 in published work, please cite CM1, the model it ports:

Bryan, G. H., and J. M. Fritsch, 2002: A benchmark simulation for moist nonhydrostatic numerical models. *Mon. Wea.
Rev.*, **130**, 2917-2928.

## Attribution

CM1 is the work of George H. Bryan and NCAR. `LICENSE` is GM1's licence; `NOTICE` carries CM1's licence terms and
lists the CM1 files and data this distribution bundles, and the notices of the bundled DejaVu fonts. ArWen (the
`gpuwm` package) is a separate dependency, installed from PyPI under its own licence; none of it is bundled here.
gpuwm-cm1 is an independent port, not affiliated with or endorsed by UCAR or NCAR.
