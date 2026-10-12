"""Every GUI action and the one ``gpuwm-cm1`` command it is.

The server builds each command's arguments here, runs exactly those through
the runner, returns ``command`` (the line a user would type) in the POST
response, and appends it to ``RUNDIR/commands.log``.  No action exists in the
GUI without a command.

Kinds:

``host``
    Host-only calls, run to completion as a subprocess with no GPU
    (``CUDA_VISIBLE_DEVICES`` empty): import, check, cartesian --plan,
    estimate, status, the guided presets (preset list, fit, new), and the
    one-picture quicklook calls (a slice, a section, a column skew-T, the
    field list), which also run at nice 19.
``launch``
    Detached in their own session so the GUI can close under them: run and
    resume own the GPU; cm1out and quicklook are the sidecars (nice 19, no
    GPU).  These are the only commands the server launches (4.7).
``control``
    stop and pause write a control file; ``stop --terminate`` sends SIGTERM and
    ``stop --force`` SIGKILL to a run on this host.  The server does this in
    process through the same code as the command (:mod:`.control`).
``view``
    The mode toggle, which never touches a run directory.
``measure``
    "Measure speed" (``estimate RUNDIR --measure``): a few steps on the GPU,
    launched detached like a run, so it waits for nothing and the page polls
    it.  The only other command the server starts that opens the GPU.

:data:`API` is the table the advanced-mode check walks: every POST endpoint and the
command it returns.  Any POST also takes ``"dry_run": true`` (:data:`DRY_RUN_KEY`):
the request is checked and the exact command returned with nothing run, which
is what "Show command" shows before a button is pressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any, Mapping, Sequence

PROGRAM = "gpuwm-cm1"

HOST_COMMANDS = frozenset({"import", "edit", "check", "cartesian", "estimate", "status", "jobscript", "quicklook",
                           "preset", "doctor", "reproduce",
                           "sounding",  # sounding times|from-wrf (reads a WRF or ArWen history; no GPU)
                           "chain"})  # chain inspect|check|new (reads a WRF or ArWen history; no GPU)
LAUNCH_COMMANDS = frozenset({"run", "resume", "cm1out", "quicklook"})
CONTROL_COMMANDS = frozenset({"stop", "pause"})
#: Sidecars run on the host's CPUs beside a local run, never on the GPU.
SIDECARS = ("cm1out", "quicklook")


def display(args: Sequence[str]) -> str:
    """The command line a user types for ``args``."""

    return shlex.join([PROGRAM, *[str(arg) for arg in args]])


def import_args(namelist: Path, out: Path, sounding: Path | None = None,
                acknowledge_output_keys: bool = False) -> list[str]:
    args = ["import", str(namelist)]
    if sounding is not None:
        args += ["--sounding", str(sounding)]
    args += ["--out", str(out)]
    if acknowledge_output_keys:
        args.append("--acknowledge-output-keys")
    return args + ["--json"]


def check_args(deck: Path) -> list[str]:
    return ["check", str(deck), "--json"]


def plan_args(deck: Path, output: Path, sounding: Path | None = None) -> list[str]:
    args = ["cartesian", str(deck), "--plan", "--output", str(output)]
    if sounding is not None:
        args += ["--sounding", str(sounding)]
    return args


def estimate_args(rundir: Path, gpu: int | None = None) -> list[str]:
    args = ["estimate", str(rundir)]
    if gpu is not None:
        args += ["--gpu", str(int(gpu))]
    return args + ["--json"]


def status_args(rundir: Path) -> list[str]:
    return ["status", str(rundir), "--json"]


def fortran_literal(value: object) -> str:
    """A JSON value as the namelist word ``gpuwm-cm1 edit --set KEY=VALUE`` takes."""

    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text if ("." in text or "e" in text or "n" in text) else text + ".0"
    text = str(value).strip()
    try:
        float(text.replace("d", "e").replace("D", "e"))
        return text
    except ValueError:
        pass
    if text.lower() in (".true.", ".false.", "t", "f", ".t.", ".f."):
        return text
    return "'" + text.replace("'", "") + "'"


def edit_args(rundir: Path, *, sets: dict | None = None, settings: dict | None = None,
              resets: Sequence[str] = (), source: Path | None = None, sounding: Path | None = None,
              acknowledge: Sequence[str] = (), seed: int | None = None) -> list[str]:
    """``gpuwm-cm1 edit``: every change the setup screens make (namelist words, settings.toml values)."""

    args = ["edit", str(rundir)]
    if source is not None:
        args += ["--from", str(source)]
    for key, value in (sets or {}).items():
        args += ["--set", f"{key}={fortran_literal(value)}"]
    for key, value in (settings or {}).items():
        args += ["--setting", f"{key}={'true' if value is True else 'false' if value is False else _number(value) if isinstance(value, (int, float)) else value}"]
    for key in resets:
        args += ["--reset-setting", str(key)]
    if sounding is not None:
        args += ["--sounding", str(sounding)]
    if "output-keys" in acknowledge:
        args.append("--acknowledge-output-keys")
    if "thompson" in acknowledge:
        args.append("--acknowledge-thompson")
    if seed is not None:
        args += ["--seed", str(int(seed))]
    return args + ["--json"]


def jobscript_args(rundir: Path, scheduler: str, *, time: str | None = None, partition: str | None = None,
                   account: str | None = None, gpu_type: str | None = None, cpus: int | None = None,
                   qos: str | None = None, mem: str | None = None, until_s: float | None = None,
                   setup: Sequence[str] | None = None, python: str | None = None,
                   write: bool = False, native: dict | None = None) -> list[str]:
    """``gpuwm-cm1 jobscript``: a Slurm or PBS script that runs the run on a GPU node with its two sidecars.

    The script runs ``gpuwm-cm1 run RUNDIR --continue``, so submitting it again continues from the last
    restart; ``until_s`` extends the run to that storm time (Continue or extend, on a cluster).  ``setup``
    lines run first on the node (``module load ...``, a venv's ``activate``), and ``python`` is the node's
    Python when it is not this one.  ``write`` puts it in the run folder as ``job-SCHEDULER.sh`` instead of
    printing it.
    """

    if scheduler not in ("slurm", "pbs"):
        raise ValueError(f"no {scheduler!r} scheduler")
    args = ["jobscript", str(rundir), "--scheduler", scheduler]
    for flag, value in (("--time", time), ("--partition", partition), ("--account", account),
                        ("--qos", qos), ("--mem", mem), ("--gpu-type", gpu_type)):
        if value:
            args += [flag, str(value)]
    if cpus is not None:
        args += ["--cpus", str(int(cpus))]
    if until_s is not None:
        args += ["--until", _number(until_s)]
    for line in setup or ():
        args += ["--setup", str(line)]
    if python:
        args += ["--python", str(python)]
    if write:
        args.append("--write")
    return args + native_args(native) + ["--json"]


def measure_args(rundir: Path, native: dict | None = None) -> list[str]:
    return ["estimate", str(rundir), "--measure", *native_args(native, publisher=False), "--json"]


def example_estimate_args(rundir: Path, root: Path) -> list[str]:
    return ["estimate", str(rundir), "--root", str(root), "--no-write", "--json"]


def example_estimate_each_args(rundirs: list[Path], root: Path) -> list[str]:
    """Every example priced in one call against one nvidia-smi answer: a JSON line each, as
    ``example_estimate_args`` prints for that example alone (``gpuwm-cm1 estimate --each``)."""

    args = ["estimate", "--root", str(root), "--no-write", "--json"]
    for rundir in rundirs:
        args += ["--each", str(rundir)]
    return args


def native_args(native: dict | None, publisher: bool = True) -> list[str]:
    """``--library`` and ``--history-publisher`` for a source checkout (``gpuwm-cm1 gui --native-dir``)."""

    if not native:
        return []
    args = ["--library", str(native["library"])]
    if publisher:
        args += ["--history-publisher", str(native["publisher"])]
    return args


def run_args(rundir: Path, native: dict | None = None) -> list[str]:
    return ["run", str(rundir), *native_args(native)]


def resume_args(rundir: Path, from_restart: int | None = None, until_s: float | None = None,
                native: dict | None = None, from_segment: int | None = None) -> list[str]:
    """``gpuwm-cm1 resume``: continue from restart N (default: the newest on the run's line) to timax, or to
    ``until_s`` to extend the run; ``from_segment`` picks the branch that holds N when two do."""

    args = ["resume", str(rundir)]
    if from_restart is not None:
        args += ["--from-restart", str(int(from_restart))]
    if from_segment is not None:
        args += ["--from-segment", str(int(from_segment))]
    if until_s is not None:
        args += ["--until", _number(until_s)]
    return args + native_args(native)


def sidecar_args(name: str, rundir: Path) -> list[str]:
    if name not in SIDECARS:
        raise ValueError(f"{name!r} is not a sidecar")
    return [name, str(rundir), "--follow"]


def _num(value: float) -> str:
    """A position, height or time as the command line reads it back: never in exponent form, because argparse
    takes ``-1e-05`` for an option and refuses the command.  Six decimals (a millimetre in km) is finer than any
    grid a slice is drawn on."""

    value = float(value)
    if value.is_integer():
        return str(int(value))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def fields_args(rundir: Path, time_s: float | None = None) -> list[str]:
    args = ["quicklook", str(rundir), "--fields"]
    if time_s is not None:
        args += ["--time", _num(time_s)]
    return args + ["--json"]


def slice_args(rundir: Path, field: str, *, height_m: float | None = None, level: int | None = None,
               line_km: Sequence[float] | None = None, time_s: float | None = None) -> list[str]:
    args = ["quicklook", str(rundir), "--slice", field]
    if level is not None:
        args += ["--level", str(int(level))]
    elif height_m is not None:
        args += ["--height", _num(height_m)]
    if line_km is not None:
        args += ["--line", *[_num(v) for v in line_km]]
    if time_s is not None:
        args += ["--time", _num(time_s)]
    return args + ["--json"]


def section_args(rundir: Path, field: str, a_km: Sequence[float], b_km: Sequence[float], *,
                 top_m: float | None = None, time_s: float | None = None) -> list[str]:
    args = ["quicklook", str(rundir), "--section", field, "--from", *[_num(v) for v in a_km],
            "--to", *[_num(v) for v in b_km]]
    if top_m is not None:
        args += ["--top", _num(top_m)]
    if time_s is not None:
        args += ["--time", _num(time_s)]
    return args + ["--json"]


def column_args(rundir: Path, at_km: Sequence[float] | None = None, time_s: float | None = None) -> list[str]:
    args = ["quicklook", str(rundir), "--column"]
    if at_km is not None:
        args += ["--at", *[_num(v) for v in at_km]]
    if time_s is not None:
        args += ["--time", _num(time_s)]
    return args + ["--json"]


# Films and the looks drawn on request
def film_args(rundir: Path, look: str, fmt: str | None = None, fps: float | None = None,
              between: int | None = None) -> list[str]:
    """Results' "Save as film": ``gpuwm-cm1 quicklook RUNDIR --film LOOK``, launched on the CPU at nice 19."""

    args = ["quicklook", str(rundir), "--film", look]
    if fmt:
        args += ["--format", fmt]
    if fps is not None:
        args += ["--fps", _num(fps)]
    if between is not None:
        args += ["--between", str(int(between))]
    return args + ["--json"]


def look_args(rundir: Path, look: str) -> list[str]:
    """A look drawn once per run (the environment, the air parcels), drawn now."""

    return ["quicklook", str(rundir), "--look", look, "--json"]


def quicklook_pass_args(rundir: Path) -> list[str]:
    """One pass of the quick-look worker: draws every look the run's pictures do not have yet."""

    return ["quicklook", str(rundir)]


def stop_args(rundir: Path, how: str = "") -> list[str]:
    args = ["stop", str(rundir)]
    if how == "terminate":
        args.append("--terminate")
    elif how == "force":
        args.append("--force")
    elif how:
        raise ValueError(f"stop has no {how!r} form")
    return args


def pause_args(rundir: Path) -> list[str]:
    return ["pause", str(rundir)]


def convert_args(rundir: Path, fmt: str | None = None) -> list[str]:
    """Results' "Convert again": every frame converted again from nothing, once the run has ended."""

    args = ["cm1out", str(rundir), "--force"]
    if fmt:
        if fmt not in ("nc", "grads"):
            raise ValueError(f"cm1out has no {fmt!r} format")
        args += ["--format", fmt]
    return args


def reproduce_args(rundir: Path) -> list[str]:
    return ["reproduce", str(rundir), "--json"]


def doctor_args(root: Path, native_dir: Path | None = None) -> list[str]:
    """The System screen: is this computer ready, in plain words (``doctor --plain``)."""

    args = ["doctor", "--plain", "--root", str(root)]
    if native_dir is not None:
        args += ["--native-dir", str(native_dir)]
    return args + ["--json"]


def mode_args(mode: str) -> list[str]:
    return ["gui", "--mode", mode]


def preset_list_args() -> list[str]:
    return ["preset", "list", "--json"]


def preset_given_args(sounding: Path | str | None = None, environment: Mapping[str, Any] | None = None) -> list[str]:
    """The flags that give a preset its sounding: ``--sounding FILE`` or the made environment's."""

    if sounding is not None:
        return ["--sounding", str(sounding)]
    if environment:
        return ["--instability", f"{float(environment['instability']):g}", "--shear",
                f"{float(environment['shear']):g}", "--hodograph", str(environment.get("hodograph") or "curved")]
    return []


def preset_custom_args(custom: Mapping[str, Any] | None) -> list[str]:
    """The ``--custom KNOB=VALUE`` flags of guided Customize's choices, in the panel's order.

    Each name must be one of the knobs (``cm1run.customize.KNOB_IDS``); whether this preset offers the value is
    the ``preset`` command's own check, which refuses it with the reason."""

    from ..cm1run.customize import KNOB_IDS

    args: list[str] = []
    for knob in KNOB_IDS:
        if not custom or knob not in custom:
            continue
        value = custom[knob]
        if isinstance(value, bool):
            text = "on" if value else "off"
        elif isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        args += ["--custom", f"{knob}={text}"]
    unknown = sorted(set(custom or {}) - set(KNOB_IDS))
    if unknown:
        raise ValueError(f"There is no setting called {unknown[0]}.")
    return args


def preset_fit_args(preset: str, budget: str, root: Path | None = None, gpu: int | None = None,
                    given: Sequence[str] = ()) -> list[str]:
    args = ["preset", "fit", preset, "--budget", budget, *given]
    if gpu is not None:
        args += ["--gpu", str(int(gpu))]
    if root is not None:
        args += ["--root", str(root)]
    return args + ["--json"]


def preset_new_args(preset: str, budget: str, root: Path, name: str, gpu: int | None = None,
                    given: Sequence[str] = ()) -> list[str]:
    """Guided Start's first command: the fitted run folder ROOT/NAME (then ``run`` runs it)."""

    args = ["preset", "new", preset, "--budget", budget, "--root", str(root), "--name", name, *given]
    if gpu is not None:
        args += ["--gpu", str(int(gpu))]
    return args + ["--json"]


# A sounding from a WRF or ArWen run (the column route) ------------------------------------------------


def _wrf_when(when: dict | None) -> list[str]:
    when = when or {}
    if when.get("time"):
        return ["--time", str(when["time"])]
    if when.get("record") is not None:
        return ["--record", str(int(when["record"]))]
    return []


def _wrf_point(point: dict, prefix: str = "") -> list[str]:
    if point.get("at") is not None:
        lat, lon = point["at"]
        # '=' keeps a negative latitude from reading as an option (argparse), in a shell and here alike.
        return [f"--{prefix}at={_number(lat)},{_number(lon)}"]
    i, j = point["ij"]
    return [f"--{prefix}ij", f"{int(i)},{int(j)}"]


def _wrf_library(native: dict | None) -> list[str]:
    """A source checkout's initializer library (``gui --native-dir``), for CM1's getcape numbers."""

    return ["--library", str(native["library"])] if native and native.get("library") else []


def wrf_times_args(path: Path, *, when: dict | None = None, map_field: str | None = None,
                   map_png: Path | None = None, native: dict | None = None) -> list[str]:
    """``sounding times FILE``: what the file is, its times, and the domain map."""

    args = ["sounding", "times", str(path), *_wrf_when(when)]
    if map_png is not None:
        args += ["--map", map_field or "shear", "--map-png", str(map_png)]
        if map_field == "cape":
            args += _wrf_library(native)
    return args + ["--json"]


def wrf_column_args(path: Path, *, when: dict | None, point: dict, out: Path, map_field: str | None = None,
                    map_png: Path | None = None, preview_png: Path | None = None,
                    native: dict | None = None, top_of: Path | None = None) -> list[str]:
    """``sounding from-wrf FILE``: the column at a point as input_sounding, with its preview and map;
    ``top_of``: a namelist whose highest level the column must reach (the guided card's storm)."""

    args = ["sounding", "from-wrf", str(path), *_wrf_when(when), *_wrf_point(point), "--out", str(out)]
    if top_of is not None:
        args += ["--top-of", str(top_of)]
    if map_png is not None:
        args += ["--map", map_field or "shear", "--map-png", str(map_png)]
    if preview_png is not None:
        args += ["--preview-png", str(preview_png)]
    return args + _wrf_library(native) + ["--json"]


def wrf_import_args(namelist: Path, path: Path, *, when: dict | None, point: dict, out: Path,
                    frame: str = "namelist", native: dict | None = None) -> list[str]:
    """``import NAMELIST --wrf FILE ...``: the column as the run's input_sounding, imported with isnd = 7;
    ``frame="follow"`` moves the frame and bubble so the storm-track check keeps the column's storm inside."""

    when = when or {}
    args = ["import", str(namelist), "--wrf", str(path)]
    if when.get("time"):
        args += ["--wrf-time", str(when["time"])]
    elif when.get("record") is not None:
        args += ["--wrf-record", str(int(when["record"]))]
    args += _wrf_point(point, "wrf-")
    if frame != "namelist":
        args += ["--wrf-frame", frame]
    return args + ["--out", str(out), *_wrf_library(native), "--json"]


def wrf_fit_args(preset: str, budget: str, sounding: Path, root: Path | None = None) -> list[str]:
    """The guided card's next step: the preset fitted with the column as its sounding (``--sounding``),
    built by the guided screen's own :func:`preset_fit_args`, so it is the line the budget screen runs."""

    return preset_fit_args(preset, budget, root, given=preset_given_args(sounding=sounding))


# Downscaling a WRF or ArWen run into a finer GM1 domain (3D, the chain) ------------------------------------


def _chain_request(history: Sequence[Path], window: Sequence[int], ratio: int, *, start: str | None = None,
                   end: str | None = None, every_s: float | None = None, moisture: str | None = None,
                   scheme: str | None = None) -> list[str]:
    args: list[str] = []
    for path in history:
        args += ["--history", str(path)]
    args += ["--window", ",".join(str(int(v)) for v in window), "--ratio", str(int(ratio))]
    if start:
        args += ["--start", str(start)]
    if end:
        args += ["--end", str(end)]
    if every_s:
        args += ["--every", _number(every_s)]
    if scheme:
        args += ["--scheme", str(scheme)]
    if moisture:
        args += ["--moisture", str(moisture)]
    return args


def chain_inspect_args(history: Sequence[Path], map_png: Path | None = None, map_time: str | None = None) -> list[str]:
    """``chain inspect FILES|DIR``: what the history is (times, grid, scheme, ground) and its domain map."""

    args = ["chain", "inspect", *[str(path) for path in history]]
    if map_png is not None:
        args += ["--map-png", str(map_png)]
        if map_time:
            args += ["--map-time", str(map_time)]
    return args + ["--json"]


def chain_check_args(history: Sequence[Path], window: Sequence[int], ratio: int, *, start: str | None = None,
                     end: str | None = None, every_s: float | None = None, moisture: str | None = None,
                     scheme: str | None = None) -> list[str]:
    """``chain check``: every admission check in plain words, and what the run would take; nothing is written."""

    return ["chain", "check", *_chain_request(history, window, ratio, start=start, end=end, every_s=every_s,
                                              moisture=moisture, scheme=scheme), "--json"]


def chain_new_args(history: Sequence[Path], window: Sequence[int], ratio: int, out: Path, *,
                   start: str | None = None, end: str | None = None, every_s: float | None = None,
                   namelist: Path | None = None, moisture: str | None = None,
                   scheme: str | None = None) -> list[str]:
    """``chain new``: admit, prepare and write a run folder that ``gpuwm-cm1 run`` runs."""

    args = ["chain", "new", *_chain_request(history, window, ratio, start=start, end=end, every_s=every_s,
                                            moisture=moisture, scheme=scheme), "--out", str(out)]
    if namelist is not None:
        args += ["--namelist", str(namelist)]
    return args + ["--json"]


def sidecar_line(args: Sequence[str]) -> str:
    """How commands.log spells a sidecar: in the background, nice 19, no GPU."""

    return f"nice -n 19 env CUDA_VISIBLE_DEVICES= {display(args)} &"


def _number(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str
    action: str
    kind: str
    command: str          # the command template, as "Show command" prints it
    note: str = ""


API: tuple[Endpoint, ...] = (
    Endpoint("POST", "/api/mode", "mode", "view", "gpuwm-cm1 gui --mode MODE",
             "Stores the view per OS user; no run directory changes."),
    Endpoint("POST", "/api/import", "import", "host",
             "gpuwm-cm1 import NAMELIST [--sounding FILE] --out RUNDIR [--acknowledge-output-keys] --json"),
    Endpoint("POST", "/api/runs/RUN/edit", "edit", "host",
             "gpuwm-cm1 edit RUNDIR [--set KEY=VALUE ...] [--setting TABLE.KEY=VALUE ...] "
             "[--reset-setting TABLE.KEY ...] --json",
             "Domain and time, Settings: changes namelist words in place, or the values CM1 keeps in Fortran; "
             "then checks the run again."),
    Endpoint("POST", "/api/runs/RUN/recommended", "recommended", "host",
             "gpuwm-cm1 edit RUNDIR --set umove=U --set vmove=V --setting initialization.bubble_center_x_m=X "
             "--setting initialization.bubble_center_y_m=Y --json",
             "Domain and time: Use the recommended frame and bubble (from the estimate's storm-track check). With "
             "CM1's updraft nudging (iinit = 12) the push is moved instead: --setting initialization.xc_wnudge=X "
             "--setting initialization.yc_wnudge=Y."),
    Endpoint("POST", "/api/runs/RUN/namelist", "namelist", "host",
             "gpuwm-cm1 edit RUNDIR --from EDITED_NAMELIST --json",
             "Check: the namelist editor's Save; the diff against your original file is kept."),
    Endpoint("POST", "/api/runs/RUN/acknowledge", "acknowledge", "host",
             "gpuwm-cm1 edit RUNDIR --acknowledge-output-keys --json",
             "Check: Run without the settings GM1 does not read (and ptype = 3 with --acknowledge-thompson)."),
    Endpoint("POST", "/api/runs/RUN/sounding", "sounding", "host",
             "gpuwm-cm1 edit RUNDIR --sounding FILE --json", "Sounding: a new input_sounding (isnd = 7 or 17)."),
    Endpoint("POST", "/api/runs/RUN/check", "check", "host", "gpuwm-cm1 check RUNDIR/deck.toml --json"),
    Endpoint("POST", "/api/runs/RUN/geometry", "geometry", "host",
             "gpuwm-cm1 cartesian RUNDIR/deck.toml --plan --output RUNDIR/plan-receipt.json [--sounding RUNDIR/input_sounding]"),
    Endpoint("POST", "/api/runs/RUN/estimate", "estimate", "host", "gpuwm-cm1 estimate RUNDIR [--gpu N] --json"),
    Endpoint("POST", "/api/runs/RUN/measure", "measure", "measure", "gpuwm-cm1 estimate RUNDIR --measure --json",
             "Estimate: Measure speed runs a few steps on the GPU (about a minute) and keeps the measured rate."),
    Endpoint("POST", "/api/runs/RUN/jobscript", "jobscript", "host",
             "gpuwm-cm1 jobscript RUNDIR --scheduler slurm|pbs [--time HH:MM:SS] [--partition P] [--account A] "
             "[--qos Q] [--mem 32G] [--gpu-type T] [--cpus N] [--until SECONDS] [--setup LINE ...] [--python PY] "
             "[--write] --json",
             "Estimate, and Continue or extend: a job script for a cluster. It runs gpuwm-cm1 run RUNDIR --continue, "
             "so submitting it again continues from the last restart; the two sidecars run on the GPU node's CPUs."),
    Endpoint("POST", "/api/runs/RUN/start", "start", "launch", "gpuwm-cm1 run RUNDIR",
             "Local runs also get the cm1out and quicklook sidecars (nice 19, no GPU)."),
    Endpoint("POST", "/api/runs/RUN/resume", "resume", "launch",
             "gpuwm-cm1 resume RUNDIR [--from-restart N] [--from-segment K] [--until SECONDS]",
             "Continue or extend: from a restart to timax, or to a later storm time; a new segment."),
    Endpoint("POST", "/api/runs/RUN/stop", "stop", "control", "gpuwm-cm1 stop RUNDIR",
             "A local run that has not ended 30 s later gets gpuwm-cm1 stop RUNDIR --terminate."),
    Endpoint("POST", "/api/runs/RUN/pause", "pause", "control", "gpuwm-cm1 pause RUNDIR"),
    Endpoint("POST", "/api/runs/RUN/force-stop", "force-stop", "control", "gpuwm-cm1 stop RUNDIR --force",
             "Offered 60 s after Stop, for runs on this computer."),
    Endpoint("POST", "/api/runs/RUN/fields", "fields", "host", "gpuwm-cm1 quicklook RUNDIR --fields [--time S] --json",
             "The cm1out fields a slice can show, with their levels; reads the frame only."),
    Endpoint("POST", "/api/runs/RUN/slice", "slice", "host",
             "gpuwm-cm1 quicklook RUNDIR --slice FIELD [--height M | --level K] [--line XA YA XB YB] [--time S] --json",
             "Draws RUNDIR/quicklook/slices/slice-*.png on the CPU at nice 19."),
    Endpoint("POST", "/api/runs/RUN/section", "section", "host",
             "gpuwm-cm1 quicklook RUNDIR --section FIELD --from XA YA --to XB YB [--top M] [--time S] --json",
             "Draws RUNDIR/quicklook/slices/section-*.png on the CPU at nice 19."),
    Endpoint("POST", "/api/runs/RUN/column", "column", "host",
             "gpuwm-cm1 quicklook RUNDIR --column [--at X Y] [--time S] --json",
             "Draws RUNDIR/quicklook/slices/column-*.png, the skew-T of one column, on the CPU at nice 19."),
    Endpoint("POST", "/api/runs/RUN/convert", "convert", "launch", "gpuwm-cm1 cm1out RUNDIR --force [--format nc|grads]",
             "Results: Convert again, once the run has ended; on the CPU at nice 19 with no GPU."),
    Endpoint("POST", "/api/runs/RUN/reproduce", "reproduce", "host", "gpuwm-cm1 reproduce RUNDIR --json",
             "Results: Reproduce bundle, RUNDIR/reproduce/ and a .tar.gz of the namelist, sounding, report, settings, "
             "deck, receipts and versions."),
    Endpoint("POST", "/api/runs/RUN/film", "film", "launch",
             "gpuwm-cm1 quicklook RUNDIR --film LOOK [--format apng|mp4|gif] [--fps N] [--between K] --json",
             "Results: Save as film, on the CPU at nice 19 with no GPU, written to RUNDIR/quicklook/films/LOOK/; "
             "{look, stop: true} stops a film this GUI started."),
    Endpoint("POST", "/api/runs/RUN/look", "look", "host",
             "gpuwm-cm1 quicklook RUNDIR --look environment|parcels --json",
             "Results: draw a look that is drawn once per run now; {look: \"all\"} launches gpuwm-cm1 quicklook RUNDIR "
             "(the looks a run's pictures do not have yet)."),
    # A preset that runs a sounding file takes {"sounding": FILE} or {"environment": {"instability",
    # "shear", "hodograph"?}}; the reply's fit.sounding names the file, its SHA-256 and where it came from.
    # {"custom": {KNOB: VALUE}} is guided Customize's choices (--custom KNOB=VALUE; preset options ID lists them).
    Endpoint("POST", "/api/fit", "fit", "host",
             "gpuwm-cm1 preset fit PRESET --budget 10m|1h|overnight|any [--sounding FILE | --instability I --shear S "
             "--hodograph curved|straight] [--custom KNOB=VALUE ...] [--gpu N] --root ROOT --json",
             "Changes nothing: the storm Start would make on this GPU, for this time."),
    Endpoint("POST", "/api/guided/start", "guided-start", "host",
             "gpuwm-cm1 preset new PRESET --budget 10m|1h|overnight|any --root ROOT --name NAME [--sounding FILE | "
             "--instability I --shear S --hodograph curved|straight] [--custom KNOB=VALUE ...] [--gpu N] --json",
             "Then gpuwm-cm1 run ROOT/NAME, with the sidecars, as Start does; the reply's command has both lines."),
    Endpoint("POST", "/api/presets/new", "preset-new", "host",
             "gpuwm-cm1 preset new PRESET --budget 10m|1h|overnight|any --root ROOT --name NAME [--instability I "
             "--shear S --hodograph curved|straight] [--custom KNOB=VALUE ...] [--gpu N] --json",
             "New run, Start from a preset: the preset fitted to this GPU and time, as a run folder to edit; "
             "nothing runs. A choice guided Customize greys (WSM6, Thompson, ground friction) is then set with "
             "gpuwm-cm1 edit ROOT/NAME --set ptype=7 (or ptype=3; or bbc=3, sfcmodel=1, cecd=1) --json."),
    # An environment from a WRF or ArWen run (the column route)
    Endpoint("POST", "/api/wrf/open", "wrf-open", "host",
             "gpuwm-cm1 sounding times FILE [--time T | --record N] --map shear|cape --map-png PNG --json",
             "From a WRF or ArWen run: what the file is, its times, and the domain map (0-6 km shear or CAPE)."),
    Endpoint("POST", "/api/wrf/column", "wrf-column", "host",
             "gpuwm-cm1 sounding from-wrf FILE [--time T | --record N] (--at=LAT,LON | --ij I,J) --out FILE "
             "--map shear|cape --map-png PNG --preview-png PNG --json",
             "The column at the point as input_sounding, with its skew-T and hodograph; nothing runs."),
    Endpoint("POST", "/api/wrf/use", "wrf-use", "host",
             "gpuwm-cm1 import NAMELIST --wrf FILE [--wrf-time T] (--wrf-at=LAT,LON | --wrf-ij I,J) "
             "[--wrf-frame follow] --out RUNDIR --json | gpuwm-cm1 sounding from-wrf FILE ... --out FILE --json",
             "target import: a run folder from a namelist with the column as its sounding (isnd = 7); target guided: "
             "the column kept for the gallery's storm, then How long can it run?"),
    # Downscaling a WRF or ArWen run (3D, the chain)
    Endpoint("POST", "/api/chain/open", "chain-open", "host",
             "gpuwm-cm1 chain inspect FILES|DIR --map-png PNG [--map-time T] --json",
             "Downscale (3D): what the history is (saved times, grid, microphysics, ground) and its map."),
    Endpoint("POST", "/api/chain/check", "chain-check", "host",
             "gpuwm-cm1 chain check --history FILES|DIR --window I0,J0,NI,NJ --ratio R [--start T] [--end T] "
             "[--every S] [--scheme S] [--moisture vapour-only] --json",
             "Downscale (3D): every admission check in plain words, and what the run would take; nothing is written."),
    Endpoint("POST", "/api/chain/new", "chain-new", "host",
             "gpuwm-cm1 chain new --history FILES|DIR --window I0,J0,NI,NJ --ratio R [--start T] [--end T] "
             "[--every S] [--scheme S] [--moisture vapour-only] --out RUNDIR [--namelist FILE] --json",
             "Downscale (3D): prepares the box from the history and writes a run folder; Start runs it as any run."),
)

#: Any POST reply may be asked for its command alone: ``{"dry_run": true, ...}`` checks the request, builds the
#: exact command and returns it without running anything or changing any file ("Show command" before a press).
DRY_RUN_KEY = "dry_run"


def endpoint_for(parts: Sequence[str]) -> Endpoint | None:
    """The API table's row for a POST path's segments (after ``/api``), e.g. ``["runs", ID, "stop"]``."""

    parts = [str(part) for part in parts]
    for endpoint in API:
        template = endpoint.path.split("/")[2:]
        if len(template) != len(parts):
            continue
        if all(want == got or want == "RUN" for want, got in zip(template, parts)):
            return endpoint
    return None


def api_table() -> list[dict[str, str]]:
    return [endpoint.__dict__.copy() for endpoint in API]


__all__ = [
    "API",
    "CONTROL_COMMANDS",
    "DRY_RUN_KEY",
    "Endpoint",
    "HOST_COMMANDS",
    "LAUNCH_COMMANDS",
    "PROGRAM",
    "SIDECARS",
    "api_table",
    "chain_check_args",
    "chain_inspect_args",
    "chain_new_args",
    "check_args",
    "reproduce_args",
    "doctor_args",
    "convert_args",
    "column_args",
    "display",
    "edit_args",
    "endpoint_for",
    "estimate_args",
    "example_estimate_args",
    "fields_args",
    "film_args",
    "look_args",
    "quicklook_pass_args",
    "fortran_literal",
    "jobscript_args",
    "measure_args",
    "native_args",
    "import_args",
    "mode_args",
    "pause_args",
    "plan_args",
    "preset_fit_args",
    "preset_custom_args",
    "preset_given_args",
    "preset_list_args",
    "preset_new_args",
    "resume_args",
    "run_args",
    "section_args",
    "sidecar_args",
    "sidecar_line",
    "slice_args",
    "status_args",
    "stop_args",
    "wrf_column_args",
    "wrf_fit_args",
    "wrf_import_args",
    "wrf_times_args",
]
