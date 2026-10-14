# Installing gpuwm-cm1 0.1.8

This is an alpha release. Read `docs/RELEASE-NOTES.md` for what it runs, how closely it
matches CM1 and what it does not do yet. The first storm is in `docs/QUICKSTART.md`; CM1 users with their own
namelist should read `docs/CM1-USERS.md`. When something does not work, `docs/TROUBLESHOOTING.md` has the common
problems with their fixes.

## What you need

| | |
|---|---|
| Computer | Linux x86_64 with glibc 2.28 or newer (RHEL/Rocky/Alma 8, Ubuntu 20.04 and later), or Windows 10 (21H2 or newer) or 11 with WSL2, which the Windows installer sets up |
| GPU | One NVIDIA GPU of compute capability 7.5 or newer with at least 8 GB of memory. GM1 has run on the RTX 3080, 4090, 5070 Ti and 5090; other cards of 7.5 and newer (RTX 20 series, A100, H100) should work but have not been tried. Older cards (GTX 10 series and before) cannot run CUDA 13. A Windows desktop keeps 4-5 GB of the card for itself, so a card that also drives your screens has less free. |
| Driver | NVIDIA driver R580 or newer. Under WSL2 this is the Windows driver; install nothing GPU-related inside WSL2. |
| Python | None with the installers: they bring their own Python 3.12. With pip: 3.12, 3.13 or 3.14 (3.12 and 3.14 are tested) |
| Disk | About 2.5 GB for the install with the installers, about 4 GB with pip (1.4 GB of it is pip's download cache: `pip cache purge` frees it). Then the storms: about 4 GB for the first guided storm (the 1 km supercell), about 120 GB for the 250 m supercell and up to about 190 GB for the 100 m one (2 h, a picture every 5 min), and for the hurricanes (4 days, a picture every 3 h) about 7.5 GB at 8 km and 30 GB at 4 km, or about 22 GB and 85 GB with a picture every hour. A grid spacing of your own can need more; Ready to start says how much before you press Start. On Windows add about 2 GB for Ubuntu, all on drive C:. |

You do **not** need a compiler, MPI, a netCDF library, the CM1 source, a CUDA toolkit or `CUDA_PATH`. The kernels are
compiled on your GPU at the start of each run by NVIDIA's NVRTC, which the install brings.

## Install

### Windows

Open PowerShell (Start menu, type PowerShell; not Command Prompt, where `irm` is not known) and paste:

```powershell install
irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1 | iex
```

It checks the Windows NVIDIA driver and says if it must be updated, and installs WSL2 with Ubuntu when they are
missing (Windows asks for permission). When WSL itself was missing, restart, let Ubuntu ask for a user name and
password, and paste the same line again; when only Ubuntu was missing, Ubuntu asks for its user name and password in
the same window and the install goes on. It installs GM1 inside Ubuntu with the Linux installer below, and adds a
**GM1** shortcut to the Start menu and the desktop, and **GM1 check** to the Start menu. The GM1 shortcut starts GM1
in a window and opens the GUI in your Windows default browser. Closing that window closes the GUI, never a running
storm: open GM1 again and the page shows it. Shutting down or restarting Windows stops a storm. GM1 check checks the
computer and saves `gm1-report.txt` on your desktop for a help request. The installer ends by saying where your storms
will be in File Explorer (`\\wsl.localhost\Ubuntu-24.04\home\YOUR-NAME\gm1-runs`).

Settings are environment variables, set in the same PowerShell window before the line. `GM1_DESKTOP_SHORTCUT=0`
leaves the desktop alone; `GM1_DISTRO` picks another WSL distribution than Ubuntu (`wsl --list` shows yours):

```powershell install
$env:GM1_DISTRO = 'Ubuntu-24.04'; irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1 | iex
```

### Linux, and Ubuntu in WSL2

```sh install
curl -LsSf https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.sh | sh
```

It checks the computer (x86-64, glibc, the NVIDIA driver), installs [uv](https://docs.astral.sh/uv/) into
`~/.local/bin` when it is missing or older than the one GM1 is tested with, installs GM1 with uv's own Python 3.12
(`uv tool install`, never the system's Python), with the exact versions of the release's `constraints-gpu-cu13.txt`
when it can download them (it says so when it cannot), adds the film add-on for MP4 films, puts `~/.local/bin` on
`PATH` for new terminals, and runs `gpuwm-cm1 doctor --plain`. Nothing needs sudo. Open a new terminal afterwards, so
it finds `gpuwm-cm1`. Settings, as environment variables before `sh`: `GM1_VERSION=0.1.8` (a given release),
`GM1_WHEEL=FILE.whl` (a wheel file), `GM1_FIND_LINKS=FOLDER`, `GM1_CONSTRAINTS=FILE`, `GM1_REINSTALL=1` (repairs a
damaged install), `GM1_NO_MODIFY_PATH=1` and `GM1_SKIP_DOCTOR=1`.

### With pip

Only if you already use Python; otherwise use the installer line above. In a new environment with Python 3.12, 3.13
or 3.14, on Linux or inside Ubuntu on Windows (pip in Windows PowerShell cannot install GM1). On Ubuntu or Debian
first: `sudo apt install python3-venv`. Ubuntu 22.04 and Debian 12 have a Python older than 3.12: use the installer
line or the uv route below.

```sh install
python3 -m venv ~/gm1
. ~/gm1/bin/activate
pip install gpuwm-cm1
gpuwm-cm1 doctor --plain
```

`pip install gpuwm-cm1` installs everything a run needs: ArWen (the `gpuwm` physics package), CuPy for CUDA 13 and
NVIDIA's NVRTC 13.3 with the CUDA libraries (about 2.6 GB in the environment, and 1.4 GB in pip's download cache,
which `pip cache purge` frees). In a new terminal later, type `. ~/gm1/bin/activate` first. MP4 films need the film
add-on: `pip install "gpuwm-cm1[film]"`. The install lines of earlier guides, `"gpuwm-cm1[gpu-cu13]"` and
`"gpuwm-cm1[gpu]"`, still work and install the same.

Do not add `gpuwm[gpu]`: ArWen's own gpu extra means CUDA 12 (`cupy-cuda12x`), and it would put a second, CUDA 12
CuPy into the same `cupy` module.

### The exact tested versions

`pip install gpuwm-cm1` holds NVRTC and nvJitLink to 13.3.x and CuPy to 14.x, and lets pip pick the rest (numpy and
small CUDA helper packages may be newer patch releases than the tested ones). To install exactly the CUDA
user-space versions every GM1 check ran with (CuPy 14.2.0, NVRTC 13.3.33 and the rest), download
`constraints-gpu-cu13.txt` from the GitHub release and pass it:

```sh install
pip install "gpuwm-cm1==0.1.8" -c constraints-gpu-cu13.txt
```

The Linux installer does this by itself.

### With uv (a cluster login node with an old system Python)

[uv](https://docs.astral.sh/uv/) brings its own Python, so a node whose system Python is 3.6 or 3.9 still works:

```sh install
uv venv -p 3.12 ~/gm1
. ~/gm1/bin/activate
uv pip install "gpuwm-cm1==0.1.8" -c constraints-gpu-cu13.txt
gpuwm-cm1 doctor --plain
```

`uv tool install --python 3.12 gpuwm-cm1` installs it as a command of its own instead, as the Linux installer does.

### From the release's wheel file

The release folder holds the wheel, the sdist, `constraints-gpu-cu13.txt` and `SHA256SUMS`. Check the downloads,
then install the wheel file. Each file you downloaded must say OK:

```sh install
sha256sum -c --ignore-missing SHA256SUMS
pip install "./gpuwm_cm1-0.1.8-py3-none-manylinux_2_28_x86_64.whl" -c constraints-gpu-cu13.txt
```

## Check the computer and start

`gpuwm-cm1 doctor --plain` checks the driver, CUDA, the GPUs and their free memory, ArWen (the physics package),
GM1's own tools, Python, disk and RAM, and which guided storms fit this computer. It prints one fix line for anything
missing and exits 0 when the computer is ready. Then:

```sh install
gpuwm-cm1 gui
```

It opens the GUI in your browser (on Windows, the Windows browser) and prints the link too. Closing the browser never
stops a run, and neither does closing the terminal or, on Windows, the GM1 window: that closes the GUI, and the storm
keeps going until it finishes. Shutting down or restarting the computer (on Windows also `wsl --shutdown`) stops a
running storm; Run it longer or Resume continues it later from its last saved point. Opening GM1 again while it is
open brings that page up in the browser again. `gpuwm-cm1 gui --no-browser` only prints the
link. The link is kept with the runs folder (in `.gm1-gui/token`, readable only by you), so it is the same at every
start and an old tab or bookmark keeps working; `gpuwm-cm1 gui --new-token` makes a new one.

### Windows 10 or 11 with WSL2, by hand

Everything above for Linux also works inside your own WSL2 Ubuntu. `gpuwm-cm1 gui` opens the Windows default browser
on `http://127.0.0.1:8765/?token=...` (both `127.0.0.1` and `localhost` reach WSL2 in its default network mode) and
says "Opened the link in your browser" only when Windows took the link; otherwise it prints the link to open
yourself. It also prints the runs folder as File Explorer names it. The GPU panel shows the memory in use but cannot
name Windows programs holding it.

### A GPU computer you reach over ssh

Start the GUI there and forward its port:

```sh cluster
ssh -L 8765:localhost:8765 gpu-box
```

then run `gpuwm-cm1 gui` on `gpu-box` and open the printed link on your own computer (over ssh it does not try to
open a browser). The GUI listens only on `127.0.0.1` and needs the link's token.

### A cluster

A job script runs the storm on a GPU node with two helpers on the node's CPUs (the CM1-format writer and the
pictures); `docs/CM1-USERS.md` has the steps. The GUI on the login node only reads the run folder and writes its stop
and pause files.

## Update and uninstall

Run the installer line again to update to the newest release: on Windows the PowerShell line, on Linux the `curl`
line. Close GM1 and let running storms finish first: the installer stops with a message while one runs. With pip:
`pip install --upgrade gpuwm-cm1`.

To uninstall, on Windows (WSL and Ubuntu stay installed; to remove Ubuntu too, and everything in it including your
storms, type `wsl --unregister Ubuntu` in PowerShell afterwards):

```powershell install
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1))) -Uninstall
```

On Linux:

```sh install
curl -LsSf https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.sh | sh -s -- --uninstall
```

Uninstalling also removes the 2.5 GB of downloaded packages from uv's cache; uv itself stays in `~/.local/bin`. With
pip: `pip uninstall gpuwm-cm1`, then `pip cache purge` frees pip's 1.4 GB download cache. Your runs in `~/gm1-runs`
are always kept; delete that folder yourself when you no longer want them.

## Build from source

The release wheel is built in a `manylinux_2_28` container, never on a desktop:

```sh cluster
git archive --format=tar.gz --prefix=gpuwm-cm1/ HEAD > gm1-src.tar.gz
SOURCE_COMMIT=$(git rev-parse HEAD) tools/release/build_manylinux.sh gm1-src.tar.gz dist-rc
```

`tools/release/build_manylinux.sh` needs docker (or `ENGINE=podman`) and network access for rustup and PyPI. Inside
`quay.io/pypa/manylinux_2_28_x86_64` it builds the sdist, extracts it, builds the wheel from the extraction with
Rust 1.95.0 (the Rust tools and `libcm1_state.so` are sealed into the wheel with their hashes), runs
`auditwheel repair` (which proves they link only against manylinux_2_28 libraries), then
`tools/inspect_artifacts.py --require-manylinux`, and writes `build-receipt.json` and `SHA256SUMS`.

Build the wheel from the extracted sdist, not from a checkout, and clear `src/*.egg-info` first: a stale
`SOURCES.txt` makes setuptools put files `MANIFEST.in` excludes back into the sdist. `tools/inspect_artifacts.py`
catches that and private paths, missing records, missing GUI pages and presets, a wheel whose base requirements are
not exactly ArWen and the CUDA 13 stack, and a wheel whose tag does not carry its sealed payload.

`tools/release/smoke_cm1_user.sh` is the release's install check: a fresh environment with `CUDA_PATH` and
`CUDA_HOME` unset, then `doctor`, CM1's supercell namelist, `import`, `run`, `cm1out` and a CM1 user's own xarray
script on the netCDF output (`--cpu-only` on a computer without a GPU).

## If something refuses

A refusal names the setting and value that caused it, what would go wrong, and what to change. `gpuwm-cm1 import`
collects every refusal of a namelist in one pass; see `docs/CM1-USERS.md`. `gpuwm-cm1 records` lists the installed
scientific records and their hashes. For problems with the install or the computer, see `docs/TROUBLESHOOTING.md`.
