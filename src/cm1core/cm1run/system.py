"""``gpuwm-cm1 doctor --plain``: is this computer ready to run GM1, in plain words (screen 10).

The System screen and a terminal read the same report.  One row per thing a
run needs, each ``ok``, ``note`` (works, with something worth knowing) or
``fail`` (a run cannot start until it is fixed), and one fix line per
failure:

- the NVIDIA driver (R580 or newer: CUDA 13);
- CUDA: CuPy and NVRTC, found without opening a CUDA context;
- each GPU: its name, architecture, memory and the memory free now, and
  whether GM1 has run on that architecture (CUDA 13 needs compute
  capability 7.5 or newer);
- ArWen (gpuwm): the audited microphysics files against their recorded digests;
- the native tools: the installed, hash-checked executables of a wheel, or a
  source checkout's built tools passed with ``--native-dir``;
- Python, the disk under the runs folder, and RAM;
- the guided presets: how many fit this computer at their own time, and why
  any other is greyed here (too little free GPU memory, RAM or disk).

Every ``fail`` row carries one exact fix: the command to type, or the one
thing to change.  Under WSL2 the driver is the Windows one, so its fix is on
the Windows side, never a driver inside WSL.  ``gpuwm-cm1 doctor --report``
(:mod:`.report`) prints the same rows with the versions a bug report needs,
as one Markdown block.

``gpuwm-cm1 doctor`` without ``--plain`` keeps its developer report (every
finding, the unvalidated register).  Nothing here imports CuPy or numpy, opens
a device or holds the GPU; the GPU facts come from ``nvidia-smi``
(:mod:`.gpuinfo`), whose fixtures a test points ``GPUWM_CM1_NVIDIA_SMI`` at.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

from . import gpuinfo

SYSTEM_SCHEMA = "gpuwm-cm1.system/v1"
#: CUDA 13 needs R580.
MINIMUM_DRIVER = 580
#: CUDA 13 supports compute capability 7.5 (Turing) and newer.
MINIMUM_COMPUTE_CAPABILITY = (7, 5)
#: The Python versions the release supports (numpy 2.5.3 needs 3.12).
PYTHON_RANGE = ((3, 12), (3, 14))
#: The card size most runs need.
SMALLEST_CARD_BYTES = 8 * 1024**3
#: Below this much free disk under the runs folder a 250 m run does not fit (0.44 + 0.57 GB a frame).
DISK_NOTE_BYTES = 20 * 10**9
DISK_FAIL_BYTES = 1 * 10**9
RAM_NOTE_BYTES = 16 * 10**9
NATIVE_FILES = {"library": "libcm1_state.so", "publisher": "cm1_forecast_publish"}
STATES = ("ok", "note", "fail")

#: Where the NVIDIA driver comes from.
NVIDIA_DOWNLOAD = "https://www.nvidia.com/Download/index.aspx"
#: The one-line installer (Linux and WSL2).  Rerunning it updates or repairs an install.
INSTALLER_URL = "https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.sh"
INSTALLER_LINE = f"curl -LsSf {INSTALLER_URL} | sh"
#: The same, reinstalling every package (a damaged CuPy or ArWen).
INSTALLER_REINSTALL_LINE = f"curl -LsSf {INSTALLER_URL} | GM1_REINSTALL=1 sh"
#: The GPU requirements gpuwm-cm1 itself declares (pyproject.toml).
GPU_REQUIREMENTS = ("cupy-cuda13x[ctk]>=14.2,<15", "nvidia-cuda-nvrtc>=13.3,<13.4",
                    "nvidia-nvjitlink>=13.3,<13.4")
#: ArWen and its data, as gpuwm-cm1 requires them (engine_pin).
ARWEN_REQUIREMENTS = ("gpuwm>=2.8.0,<2.8.1", "gpuwm-data>=2.8.0,<2.8.1")
#: Under WSL2, after the Windows driver changes.
WSL_RESTART = "then in Windows PowerShell run wsl --update and then wsl --shutdown, and open Ubuntu again"
WSL_NO_LINUX_DRIVER = "Never install an NVIDIA driver inside WSL."


@dataclass(frozen=True)
class Row:
    id: str
    area: str
    state: str          # ok | note | fail
    text: str
    fix: str = ""

    def record(self) -> dict[str, str]:
        row = {"id": self.id, "area": self.area, "state": self.state, "text": self.text}
        if self.fix:
            row["fix"] = self.fix
        return row


def _gib(value: int | float | None) -> str:
    if value is None:
        return "?"
    return f"{value / 1024**3:.1f} GB"


def _gb(value: int | float | None) -> str:
    if value is None:
        return "?"
    return f"{value / 1e9:,.0f} GB" if value >= 10e9 else f"{value / 1e9:.1f} GB"


def _distribution(*names: str) -> tuple[str, str] | None:
    """(name, version) of the first installed distribution of ``names``."""

    from importlib import metadata

    for name in names:
        try:
            return name, metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _cupy_distribution() -> tuple[str, str] | None:
    """The installed CuPy wheel (cupy-cuda13x, cupy-cuda12x, or a source build), never importing it."""

    from importlib import metadata

    found = []
    try:
        for distribution in metadata.distributions():
            name = (distribution.metadata.get("Name") or "").lower()
            if name == "cupy" or name.startswith("cupy-cuda"):
                found.append((name, distribution.version))
    except Exception:  # noqa: BLE001 - a broken metadata folder must not stop the report
        return None
    return sorted(found)[0] if found else None


# ---------------------------------------------------------------- how GM1 was installed, and how to repair it

def install_method() -> dict[str, str]:
    """How this gpuwm-cm1 was installed, for the fix lines and the report.

    ``kind``: ``uv-tool`` (the installer's ``uv tool install``), ``source`` (a
    checkout on ``PYTHONPATH``, or an editable install), ``venv`` or ``system``;
    ``installer``: ``uv`` or ``pip``; ``words``: the same in a few words.
    """

    from importlib import metadata

    prefix = Path(sys.prefix)
    try:
        installer = (metadata.distribution("gpuwm-cm1").read_text("INSTALLER") or "").strip()
    except Exception:  # noqa: BLE001 - not installed (a checkout) or unreadable metadata
        installer = ""
    package = Path(__file__).resolve().parents[1]
    in_site = any(part in ("site-packages", "dist-packages") for part in package.parts)
    if in_site and (prefix / "uv-receipt.toml").is_file():
        return {"kind": "uv-tool", "installer": "uv", "words": "uv tool (the installer)"}
    if not in_site:
        return {"kind": "source", "installer": installer or "pip", "words": "a source checkout"}
    installer = installer if installer in ("uv", "pip") else "pip"
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda = bool(conda_prefix) and Path(conda_prefix).resolve() == prefix.resolve()
    where = "a conda environment" if conda else "a virtual environment" if in_venv else "no virtual environment"
    return {"kind": "venv" if (in_venv or conda) else "system", "installer": installer,
            "words": f"{'uv pip' if installer == 'uv' else 'pip'} in {where}"}


def _pip(arguments: str, method: Mapping[str, str]) -> str:
    import shlex

    python = shlex.quote(sys.executable)
    if method.get("installer") == "uv":
        return f"uv pip install --python {python} {arguments}"
    return f"{python} -m pip install {arguments}"


def film_fix(method: Mapping[str, str] | None = None) -> str:
    """How to add MP4 films (imageio-ffmpeg) for the way this copy was installed: the installer brings it, so
    an installer copy without it is repaired by the installer line; pip and uv pip add the film extra."""

    method = install_method() if method is None else method
    if method["kind"] == "uv-tool":
        return f"paste the installer line again, which adds it: {INSTALLER_LINE}"
    if method["kind"] == "source":
        return 'pip install "gpuwm-cm1[film]" (or pip install imageio-ffmpeg)'
    return _pip('"gpuwm-cm1[film]"', method)


def reinstall_fix(what: str, method: Mapping[str, str] | None = None) -> str:
    """The one command that puts back ``what``: ``package`` (GM1's own files, its native tools among them),
    ``gpu`` (CuPy and NVRTC) or ``arwen`` (gpuwm and gpuwm-data), for the way this copy was installed."""

    method = install_method() if method is None else method
    if method["kind"] == "uv-tool":
        return INSTALLER_LINE if what == "package" else INSTALLER_REINSTALL_LINE
    if what == "package":
        if method["kind"] == "source":
            return "cargo build --release --locked --manifest-path rust/cm1-state/Cargo.toml"
        return _pip("--force-reinstall --no-deps gpuwm-cm1", method)
    if what == "gpu":
        return _pip(" ".join(f'"{requirement}"' for requirement in GPU_REQUIREMENTS), method)
    return _pip("--force-reinstall --no-deps " + " ".join(f'"{requirement}"' for requirement in ARWEN_REQUIREMENTS),
                method)


# ---------------------------------------------------------------- the rows

def _driver_fix(wsl2: bool, *, update: bool = False) -> str:
    if wsl2:
        verb = "Update" if update else "Install or update"
        return (f"{verb} the NVIDIA driver on Windows to R{MINIMUM_DRIVER} or newer ({NVIDIA_DOWNLOAD}), "
                f"{WSL_RESTART}. {WSL_NO_LINUX_DRIVER}")
    verb = "Update the NVIDIA driver to" if update else "Install the NVIDIA driver"
    return (f"{verb} R{MINIMUM_DRIVER} or newer ({NVIDIA_DOWNLOAD}; on Ubuntu: sudo ubuntu-drivers install "
            f"nvidia:{MINIMUM_DRIVER}), then restart the computer and check that nvidia-smi says {MINIMUM_DRIVER} "
            "or newer.")


def driver_row(query: gpuinfo.GpuQuery) -> Row:
    wsl2 = bool(query.wsl2)
    if query.status == "no-nvidia-smi":
        if wsl2:
            return Row("driver", "Driver", "fail",
                       "No NVIDIA GPU is visible inside WSL2: the Windows NVIDIA driver is missing or too old, "
                       "or WSL needs an update.", _driver_fix(True))
        return Row("driver", "Driver", "fail", "No NVIDIA driver found.", _driver_fix(False))
    if query.status != "ok":
        message = query.message or "nvidia-smi did not answer."
        if wsl2:
            return Row("driver", "Driver", "fail", message + " Inside WSL2 this is the Windows driver.",
                       _driver_fix(True))
        return Row("driver", "Driver", "fail", message,
                   f"Restart the computer; if nvidia-smi still fails, reinstall the NVIDIA driver R{MINIMUM_DRIVER} "
                   f"or newer ({NVIDIA_DOWNLOAD}).")
    if not query.gpus:
        if wsl2:
            return Row("driver", "Driver", "fail", "The NVIDIA driver answers but no GPU is visible inside WSL2.",
                       _driver_fix(True))
        return Row("driver", "Driver", "fail", "The NVIDIA driver answers but lists no GPU.",
                   "Check that the card is seated, powered and enabled in the BIOS, then restart the computer.")
    version = query.gpus[0].driver_version or ""
    match = re.match(r"(\d+)", version)
    if match and int(match.group(1)) < MINIMUM_DRIVER:
        where = " (the Windows driver)" if wsl2 else ""
        return Row("driver", "Driver", "fail",
                   f"NVIDIA {version}{where}: older than R{MINIMUM_DRIVER}, which CUDA 13 needs.",
                   _driver_fix(wsl2, update=True))
    where = " (the Windows driver, seen through WSL2)" if wsl2 else ""
    return Row("driver", "Driver", "ok", f"NVIDIA {version or 'driver'}{where}; R{MINIMUM_DRIVER} or newer is needed.")


#: Where WSL2 keeps the Windows driver's CUDA library.
WSL_LIBCUDA = "/usr/lib/wsl/lib/libcuda.so.1"
#: Set to a library path to load instead of libcuda.so.1, or to "none" (the tests' computer without one).
LIBCUDA_ENV = "GPUWM_CM1_LIBCUDA"


def libcuda_loadable() -> bool | None:
    """Whether the NVIDIA driver's CUDA library loads (no context is made and no GPU is held), or None when a
    test's stand-in nvidia-smi answers and no stand-in library was named."""

    import ctypes

    forced = os.environ.get(LIBCUDA_ENV)
    if forced is None and os.environ.get(gpuinfo.NVIDIA_SMI_ENV):
        return None
    if forced == "none":
        return False
    names = [forced] if forced else ["libcuda.so.1", WSL_LIBCUDA]
    for name in names:
        try:
            ctypes.CDLL(name)
        except OSError:
            continue
        return True
    return False


def libcuda_row(query: gpuinfo.GpuQuery) -> Row | None:
    """A failing Driver row when nvidia-smi answers but the driver's CUDA library cannot be loaded (a broken or
    half-removed driver), else None."""

    if query.status != "ok" or not query.gpus or libcuda_loadable() is not False:
        return None
    wsl2 = bool(query.wsl2)
    return Row("driver", "Driver", "fail",
               "The NVIDIA driver answers, but its CUDA library (libcuda) cannot be loaded, so no storm can run.",
               _driver_fix(wsl2))


def cuda_row(method: Mapping[str, str] | None = None) -> Row:
    cupy = _cupy_distribution()
    try:
        found = cupy is not None or importlib.util.find_spec("cupy") is not None
    except (ImportError, ValueError):  # a process that refuses CuPy imports (the GUI server) refuses to look too
        found = False
    if not found:
        return Row("cuda", "CUDA", "fail", "CuPy is not installed, so no kernel can be compiled or run. gpuwm-cm1 "
                   "installs CuPy for CUDA 13 with NVRTC; something removed it.", reinstall_fix("gpu", method))
    name, version = cupy if cupy is not None else ("cupy", "a source build")
    nvrtc = _distribution("nvidia-cuda-nvrtc", "nvidia-cuda-nvrtc-cu13", "nvidia-cuda-nvrtc-cu12")
    nvrtc_words = (f"NVRTC from the {nvrtc[0]} {nvrtc[1]} wheel" if nvrtc is not None else
                   "NVRTC from this computer's CUDA install")
    text = (f"{name} {version}; {nvrtc_words}. Kernels are compiled at the start of each run; "
            "no CUDA toolkit or CUDA_PATH is needed.")
    if name.startswith("cupy-cuda") and not name.startswith("cupy-cuda13"):
        return Row("cuda", "CUDA", "note", text + " The release is built and tested with cupy-cuda13x, which "
                   "gpuwm-cm1 installs.", reinstall_fix("gpu", method))
    return Row("cuda", "CUDA", "ok", text)


def too_old(gpu: gpuinfo.GpuInfo) -> bool:
    """Compute capability below 7.5, which CUDA 13 does not support (an unknown one is not too old)."""

    major, _, minor = (gpu.compute_capability or "").partition(".")
    if not major.isdigit() or not (minor or "0").isdigit():
        return False
    return (int(major), int(minor or 0)) < MINIMUM_COMPUTE_CAPABILITY


def gpu_rows(query: gpuinfo.GpuQuery) -> list[Row]:
    rows: list[Row] = []
    try:
        from .rates import load_record

        architectures = dict(load_record().get("architectures_run") or {})
    except Exception:  # noqa: BLE001 - the record is part of the package; a broken one only drops the note
        architectures = {}
    usable = [gpu for gpu in query.gpus if not too_old(gpu)]
    for gpu in query.gpus:
        area = f"GPU {gpu.index}"
        sm = gpu.sm or "unknown architecture"
        if too_old(gpu):
            text = (f"{gpu.name}, {sm}: older than the RTX 20 series. CUDA 13, which GM1 needs, supports "
                    "compute capability 7.5 (Turing) and newer, so GM1 cannot use this card.")
            if usable:
                rows.append(Row(f"gpu-{gpu.index}", area, "note", text,
                                f"Run on GPU {usable[0].index} instead: gpuwm-cm1 run RUNDIR --gpu {usable[0].index}"))
            else:
                rows.append(Row(f"gpu-{gpu.index}", area, "fail", text,
                                "Use a computer with an NVIDIA GPU of compute capability 7.5 or newer "
                                "(RTX 20 series or newer)."))
            continue
        ran = gpu.sm in architectures if gpu.sm else False
        state = "ok" if ran and gpu.total_bytes >= SMALLEST_CARD_BYTES * 0.97 else "note"
        text = (f"{gpu.name}, {sm}, {_gib(gpu.total_bytes)}, {_gib(gpu.free_bytes)} free now. "
                + ("GM1 has run on this architecture." if ran else
                   "GM1 has not run on this architecture yet; it has run on "
                   + ", ".join(sorted(architectures)) + "." if architectures else ""))
        fix = ""
        if gpu.total_bytes < SMALLEST_CARD_BYTES * 0.97:
            text += " Smaller than the 8 GB most runs need; the estimate says what fits."
        rows.append(Row(f"gpu-{gpu.index}", area, state, text.strip(), fix))
        in_use = gpu.in_use_bytes
        if in_use > 512 * 1024**2:
            if gpu.wsl2 and not gpu.compute_apps:
                words = (f"{_gib(in_use)} is in use by other programs, most likely the Windows desktop. "
                         "Under WSL2 they cannot be named.")
            elif gpu.compute_apps:
                names = ", ".join(f"{app.name} (pid {app.pid})" for app in gpu.compute_apps[:4])
                words = f"{_gib(in_use)} is in use: {names}."
            else:
                words = f"{_gib(in_use)} is in use by the display or other programs."
            rows.append(Row(f"gpu-{gpu.index}-memory", f"{area} memory", "note", words,
                            "Nothing to do: the estimate plans with the memory free now. A card with no display "
                            "attached frees what the desktop holds."))
    return rows


def arwen_row(method: Mapping[str, str] | None = None) -> Row:
    try:
        from ..engine_pin import inspect

        estate = inspect()
    except Exception as error:  # noqa: BLE001 - the report names the failure instead of raising
        return Row("arwen", "ArWen", "fail", f"The microphysics engine could not be checked: {error}",
                   reinstall_fix("arwen", method))
    if estate.admitted:
        return Row("arwen", "ArWen", "ok",
                   f"gpuwm {estate.version}; its audited microphysics files match the recorded digests.")
    if estate.version is None:
        return Row("arwen", "ArWen", "fail", "gpuwm (ArWen, the microphysics engine) is not installed.",
                   reinstall_fix("arwen", method))
    problems = []
    if estate.moved:
        problems.append(f"{len(estate.moved)} audited file(s) differ from the recorded digests")
    if estate.absent:
        problems.append(f"{len(estate.absent)} audited file(s) are missing")
    return Row("arwen", "ArWen", "fail",
               f"gpuwm {estate.version}: " + ("; ".join(problems) or "not the audited version") + ".",
               reinstall_fix("arwen", method))


def native_row(native_dir: Path | None = None, method: Mapping[str, str] | None = None) -> Row:
    from .. import rust_tools

    try:
        tools = rust_tools.verify_native_tools()
        rust_tools.verify_native_libraries()
    except (OSError, rust_tools.NativeToolError) as error:
        installed_error = str(error)
    else:
        return Row("native", "Native tools", "ok",
                   f"{len(tools)} installed executables and the initializer library pass their hash checks.")
    if native_dir is not None:
        directory = Path(native_dir).expanduser()
        missing = [name for name in NATIVE_FILES.values() if not (directory / name).is_file()]
        parser_ok = True
        try:
            from ..cm1import.namelist import locate_tool

            locate_tool()
        except Exception:  # noqa: BLE001
            parser_ok = False
        if not missing and parser_ok:
            return Row("native", "Native tools", "ok",
                       f"A source checkout: the built initializer and publisher in {directory} "
                       "(gpuwm-cm1 gui --native-dir) and the namelist parser.")
        problems = []
        if missing:
            problems.append(f"{directory} has no {' or '.join(missing)}")
        if not parser_ok:
            problems.append("the namelist parser cm1-r21-namelist is not built (or GM1_CM1_R21_NAMELIST does not "
                            "name it)")
        return Row("native", "Native tools", "fail", f"A source checkout: {'; '.join(problems)}.",
                   "cargo build --release --locked --manifest-path rust/cm1-state/Cargo.toml")
    method = install_method() if method is None else method
    source = (" In a source checkout, build them and pass gpuwm-cm1 gui --native-dir DIR instead."
              if method["kind"] == "source" else "")
    return Row("native", "Native tools", "fail",
               f"The installed native tools did not pass their check: {installed_error}.{source}",
               reinstall_fix("package", method))


def python_row() -> Row:
    version = ".".join(str(part) for part in sys.version_info[:3])
    low, high = PYTHON_RANGE
    if sys.version_info[:2] < low:
        return Row("python", "Python", "fail", f"{version}: GM1 needs Python 3.12 or newer.",
                   f"Install GM1 with the installer, which brings its own Python 3.12: {INSTALLER_LINE}")
    if sys.version_info[:2] > high:
        return Row("python", "Python", "note", f"{version}: newer than the 3.12 to 3.14 GM1 is tested on.")
    return Row("python", "Python", "ok", f"{version} (3.12 to 3.14 are supported).")


#: Where the runs go when the disk under the default one is short.
ROOM_FIX = "Free some disk space, or keep the runs on a disk with room: gpuwm-cm1 gui --root FOLDER"


def disk_row(host: gpuinfo.HostInfo) -> Row:
    free, total = host.disk_free_bytes, host.disk_total_bytes
    where = host.disk_root
    if free is None:
        return Row("disk", "Disk", "note", f"{where}: free space could not be read.")
    text = f"{where}: {_gb(free)} free of {_gb(total)}."
    if free < DISK_FAIL_BYTES:
        return Row("disk", "Disk", "fail", text + " Not enough for even a small run's output.", ROOM_FIX)
    if free < DISK_NOTE_BYTES:
        return Row("disk", "Disk", "note", text + " Enough for 1 km runs; a 250 m run writes up to about 120 GB. "
                   "The estimate says what each run needs, and a storm that does not fit is greyed.", ROOM_FIX)
    return Row("disk", "Disk", "ok", text)


#: Giving WSL2 more memory, as a line to paste into Windows PowerShell (Notepad would save .wslconfig.txt).
WSL_MEMORY_FIX = ('Give Ubuntu more memory: in Windows PowerShell paste  '
                  'Add-Content "$env:USERPROFILE\\.wslconfig" "`n[wsl2]`nmemory=16GB"  then  wsl --shutdown  and '
                  'open GM1 again. If that file already has a [wsl2] line, do not add a second one: open it with  '
                  'notepad "$env:USERPROFILE\\.wslconfig"  and put memory=16GB on the line under [wsl2].')


def ram_row(host: gpuinfo.HostInfo) -> Row:
    total, available = host.ram_total_bytes, host.ram_available_bytes
    if total is None:
        return Row("ram", "RAM", "note", "The host's memory could not be read.")
    limit = " (the WSL2 limit)" if host.wsl2 else ""
    text = f"{_gb(total)}, {_gb(available)} free{limit}."
    if total < RAM_NOTE_BYTES:
        fix = (WSL_MEMORY_FIX if host.wsl2 else "")
        return Row("ram", "RAM", "note", text + " A run keeps about 365 bytes of RAM per grid cell locked for "
                   "its writes (a frame and a restart; about 6.7 GB for 18 million cells); the estimate warns "
                   "above half of RAM.", fix)
    return Row("ram", "RAM", "ok", text)


#: What a greyed preset's reason names (cm1run.fit), and the one thing that frees it.
SHORTAGE_FIXES = (
    ("GPU memory", "Close programs that use the graphics card (games, video, another GM1 run), then check again."),
    ("of RAM", "Close programs to free RAM, then check again."),
    ("free disk", ROOM_FIX),
)


def presets_row(query: gpuinfo.GpuQuery, host: gpuinfo.HostInfo) -> Row | None:
    """How many guided presets fit this computer at their own time, and why any other is greyed here.

    Fitted as the gallery fits them (``gpuwm-cm1 preset fit ID --budget B``), against the GPU memory free
    now, RAM and the disk under the runs folder.  None without a usable GPU: the driver and GPU rows
    already say why.
    """

    gpu = next((gpu for gpu in query.gpus if not too_old(gpu)), None)
    if query.status != "ok" or gpu is None:
        return None
    try:
        from .presets import examine, fit, load_all

        everywhere, fits, short = [], [], []
        for preset in load_all():
            if not examine(preset)["offered"]:
                everywhere.append(preset)
                continue
            result = fit(preset, preset.budget, gpu=gpu, host=host)
            (fits if result.offered else short).append((preset, result))
    except Exception as error:  # noqa: BLE001 - the report names the failure instead of raising
        return Row("presets", "Presets", "note", f"The storms could not be fitted to this computer: {error}")
    offered = len(fits) + len(short)
    text = f"{len(fits)} of the {offered} storms fit this computer at their own time."
    if everywhere:
        text += f" {len(everywhere)} more are greyed in this release on every computer."
    if not short:
        return Row("presets", "Presets", "ok", text)
    named = "; ".join(f"{preset.name}: {result.reason.rstrip('.')}" for preset, result in short[:4])
    named += f"; and {len(short) - 4} more." if len(short) > 4 else "."
    reasons = " ".join(result.reason for _preset, result in short)
    fixes = [fix for needle, fix in SHORTAGE_FIXES if needle in reasons]
    if fixes and gpu.wsl2 and fixes[0] == SHORTAGE_FIXES[0][1]:
        fixes[0] += " On Windows the desktop itself keeps 4-5 GB of the card."
    fix = fixes[0] if fixes else "Pick a longer time for these storms."
    return Row("presets", "Presets", "note" if fits else "fail", f"{text} Greyed here: {named}", fix)


# ---------------------------------------------------------------- the report

def system_report(root: str | os.PathLike[str] = ".", *, native_dir: str | os.PathLike[str] | None = None,
                  query: gpuinfo.GpuQuery | None = None, host: gpuinfo.HostInfo | None = None,
                  presets: bool = True) -> dict[str, Any]:
    """Every row, the exit code (0 ready, 1 something to fix) and the command that produced it."""

    query = gpuinfo.query_gpus() if query is None else query
    host = gpuinfo.query_host(root) if host is None else host
    method = install_method()
    rows = [libcuda_row(query) or driver_row(query), cuda_row(method), *gpu_rows(query), arwen_row(method),
            native_row(None if native_dir is None else Path(native_dir), method), python_row(), disk_row(host),
            ram_row(host)]
    fitted = presets_row(query, host) if presets else None
    if fitted is not None:
        rows.append(fitted)
    failures = [row for row in rows if row.state == "fail"]
    command = ["gpuwm-cm1", "doctor", "--plain", "--root", os.fspath(root)]
    if native_dir is not None:
        command += ["--native-dir", os.fspath(native_dir)]
    return {
        "schema": SYSTEM_SCHEMA,
        "ready": not failures,
        "exit": 1 if failures else 0,
        "summary": ("Ready to run." if not failures else
                    f"{len(failures)} thing{'s' if len(failures) != 1 else ''} to fix before a run can start."),
        "rows": [row.record() for row in rows],
        "wsl2": bool(query.wsl2 or host.wsl2),
        "gpus": [gpu.record() for gpu in query.gpus],
        "host": host.record(),
        "install": dict(method),
        "command": " ".join(_quote(word) for word in command),
    }


def _quote(word: str) -> str:
    import shlex

    return shlex.quote(word)


def render_text(report: Mapping[str, Any]) -> str:
    rows: Iterable[Mapping[str, Any]] = report["rows"]
    width = max((len(str(row["area"])) for row in report["rows"]), default=8)
    mark = {"ok": "ok  ", "note": "note", "fail": "FIX "}
    lines = [report["summary"], ""]
    for row in rows:
        lines.append(f"{mark.get(row['state'], row['state'])}  {str(row['area']).ljust(width)}  {row['text']}")
        if row.get("fix") and row["state"] in ("fail", "note"):
            lines.append(f"{'':4}  {'':{width}}  {'fix' if row['state'] == 'fail' else 'tip'}: {row['fix']}")
    lines.append("")
    lines.append("GM1 is ready when no line starts with FIX. This check never starts a run and never holds the GPU.")
    lines.append("Asking for help? gpuwm-cm1 doctor --report prints all of this as one block to paste.")
    return "\n".join(lines)


#: The GUI's runs folder (``gpuwm-cm1 gui`` without ``--root``).
DEFAULT_RUNS = Path("~/gm1-runs")


def default_root() -> Path:
    """The folder the disk check reads: the GUI's ``~/gm1-runs``, or the home folder before it exists."""

    runs = DEFAULT_RUNS.expanduser()
    return runs if runs.is_dir() else Path.home()


def doctor_plain(arguments: Any) -> int:
    """The ``doctor --plain`` door, and ``doctor --report``."""

    explicit = getattr(arguments, "root", None) is not None
    root = Path(arguments.root) if explicit else default_root()
    native_dir = getattr(arguments, "native_dir", None)
    if getattr(arguments, "report", False):
        from .report import doctor_report

        print(doctor_report(root, native_dir=native_dir,
                            look_for_runs=explicit or root.expanduser() != Path.home()))
        return 0
    report = system_report(root, native_dir=native_dir)
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return int(report["exit"])


__all__ = ["INSTALLER_LINE", "WSL_MEMORY_FIX", "film_fix", "libcuda_loadable", "libcuda_row", "MINIMUM_DRIVER", "NVIDIA_DOWNLOAD", "Row", "SYSTEM_SCHEMA", "arwen_row", "cuda_row",
           "default_root", "disk_row", "doctor_plain", "driver_row", "gpu_rows", "install_method", "native_row",
           "presets_row", "python_row", "ram_row", "reinstall_fix", "render_text", "system_report", "too_old"]
