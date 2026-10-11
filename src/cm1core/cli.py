"""The lazy, refusal-first ``gpuwm-cm1`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .cartesian_door import cartesian_door, perf_door
from .cartesian_history import (
    CARTESIAN_HISTORY_CARRIER_PROFILES,
    COMPLETE_CARRIER_PROFILE,
)
from .doors import (
    check_door,
    doctor_door,
    examples_door,
    preflight_door,
    records_door,
    render_door,
    restart_door,
    run_door,
    version_door,
)
from .errors import CM1Refusal


def _config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", type=Path, help="strict UTF-8 gpuwm-cm1 TOML")


def _output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)


#: What ``gpuwm-cm1`` with no command, and the top of ``--help``, say first.
START_WORDS = ("GM1: CM1 storm simulations on your NVIDIA graphics card.\n"
               "New here? Type:          gpuwm-cm1 gui\n"
               "Something wrong? Type:   gpuwm-cm1 doctor\n"
               "Asking for help? Paste:  gpuwm-cm1 doctor --report")


def _version_words() -> str:
    from ._version import __version__

    return f"gpuwm-cm1 {__version__}"


def _doctor(arguments: argparse.Namespace) -> int:
    """``gpuwm-cm1 doctor``: the plain check unless the developer report is asked for (--science, --explain)."""

    if not (getattr(arguments, "science", False) or getattr(arguments, "explain", False)):
        arguments.plain = True
    return doctor_door(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuwm-cm1",
        description=START_WORDS,
        epilog="gui, doctor, preset, run, status and stop are the everyday commands; the others are for CM1 "
               "users and developers (docs/COMMANDS.md and docs/CM1-USERS.md).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_version_words())
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    # The GUI first: the one command most people need (standard library only; it opens no device).
    from .gui.server import add_command as _gui_command

    _gui_command(commands)

    command = commands.add_parser(
        "doctor", help="check that this computer can run GM1, with one fix line per problem")
    command.add_argument("--json", action="store_true", help="the same as JSON")
    command.add_argument(
        "--plain", action="store_true",
        help="the default: is this computer ready to run (driver, CUDA, GPUs, ArWen, native tools, Python, disk "
             "and RAM), in plain words with one fix line per failure (the GUI's System screen); exit 0 when ready",
    )
    command.add_argument(
        "--science", "--full", dest="science", action="store_true",
        help="the developer report instead: every installed component and what has and has not been checked "
             "against CM1",
    )
    command.add_argument("--explain", action="store_true", help="with --science: say more about each finding")
    command.add_argument(
        "--report", action="store_true",
        help="everything a bug report needs as one Markdown block to paste: versions, OS, driver, GPUs, how GM1 "
             "was installed, the --plain rows and the newest run's status (no user or host names in it)",
    )
    command.add_argument("--root", type=Path,
                         help="with --plain or --report: the runs folder (default: ~/gm1-runs, the GUI's, "
                              "or your home folder before it exists)")
    command.add_argument(
        "--native-dir", type=Path, metavar="DIR",
        help="with --plain, in a source checkout: the folder of the built libcm1_state.so and cm1_forecast_publish",
    )
    command.set_defaults(handler=_doctor)

    command = commands.add_parser(
        "version", help="report package and authority identity"
    )
    command.set_defaults(handler=version_door)

    command = commands.add_parser(
        "examples",
        help="list the shipped example decks, or copy them with --copy DIR",
    )
    command.add_argument("names", nargs="*", help="example names; default: all")
    command.add_argument(
        "--copy", type=Path, help="copy the selected decks into this directory"
    )
    command.add_argument(
        "--cm1", action="store_true",
        help="CM1's own namelists instead (run/config_files supercell and squall_line, unchanged); "
             "--copy DIR writes DIR/<case>/namelist.input",
    )
    command.set_defaults(handler=examples_door)

    command = commands.add_parser(
        "records", help="list installed scientific records and hashes"
    )
    command.set_defaults(handler=records_door)

    command = commands.add_parser(
        "check", help="validate and seal the strict configuration"
    )
    _config(command)
    command.add_argument(
        "--json",
        action="store_true",
        help="emit the canonical JSON receipt (the default)",
    )
    command.set_defaults(handler=check_door)

    from .cm1import.door import add_parser as add_import_parser
    from .cm1import.edit import add_parser as add_edit_parser

    add_import_parser(commands)
    add_edit_parser(commands)
    __import__("cm1core.cm1import.sounding_door", fromlist=["add_parser"]).add_parser(commands)  # sounding

    command = commands.add_parser(
        "preflight", help="perform allocation-free scientific admission"
    )
    _config(command)
    command.add_argument(
        "--json",
        action="store_true",
        help="emit the blocker list as JSON (the default)",
    )
    command.set_defaults(handler=preflight_door)

    for name, help_text in (
        ("init", "initialize CM1 state after all gates pass"),
        ("forecast", "run the device-resident CM1/ArWen product after all gates pass"),
    ):
        command = commands.add_parser(name, help=help_text)
        _config(command)
        _output(command)
        command.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
        command.set_defaults(handler=run_door)

    command = commands.add_parser(
        "cartesian",
        help="run the configurable Cartesian CUDA runtime after all gates pass",
    )
    _config(command)
    _output(command)
    command.add_argument(
        "--history-publisher",
        type=Path,
        help=(
            "explicit cm1-forecast-publish executable; default: the installed "
            "native tool. A source checkout has no installed manifest."
        ),
    )
    command.add_argument(
        "--history-dir",
        type=Path,
        help=(
            "publish history frames of completed device state into this new "
            "directory, through the installed forecast publisher"
        ),
    )
    # No argparse ``choices=`` here on purpose.  Every other option in this
    # parser refuses through the project's refuse() machinery, which names the
    # concrete wrong result and its remedy; ``choices=`` would exit 2 with a
    # bare usage line before require_cartesian_history_carriers could produce
    # that refusal document.  The admitted spellings are in the help text.
    command.add_argument(
        "--history-carriers",
        default=COMPLETE_CARRIER_PROFILE,
        metavar="{" + ",".join(CARTESIAN_HISTORY_CARRIER_PROFILES) + "}",
        help=(
            "which carriers each frame publishes; one of "
            f"{', '.join(CARTESIAN_HISTORY_CARRIER_PROFILES)}. 'complete' is "
            "every carrier this runtime holds; 'reconstructible-omitted' drops "
            "prs and zf and publishes instead the base.pi0 and "
            "metrics.zf_profile columns cm1-render rebuilds them from"
        ),
    )
    command.add_argument(
        "--restart-dir",
        type=Path,
        help=(
            "write one complete restart into this new directory every "
            "run.restart_interval_s (CM1 rstfrq), each named restart-NNNNNN. "
            "With run.iconly = 1 it receives the initial condition and the run "
            "stops without integrating."
        ),
    )
    command.add_argument(
        "--restart-from",
        type=Path,
        help=(
            "the previous run's --restart-dir; with run.irst = 1 and "
            "run.rstnum = N the run continues from restart-NNNNNN inside it"
        ),
    )
    command.add_argument(
        "--integrate",
        action="store_true",
        help=(
            "step to run.duration_s and publish one complete frame directory "
            "every run.history_interval_s, instead of running the timing harness"
        ),
    )
    command.add_argument(
        "--library",
        type=Path,
        help="explicit Rust idealized initializer; default: the installed native library",
    )
    command.add_argument(
        "--sounding",
        type=Path,
        help=(
            "CM1 input_sounding file; required when the deck reads one (isnd = 7, "
            "the default).  No sounding is installed: CM1 runtime assets are not "
            "redistributed.  Decks with isnd = 5 (analytic Weisman-Klemp) need none"
        ),
    )
    command.add_argument(
        "--gpu-uuid",
        help="take the ecosystem exclusive-GPU mutex for this device UUID",
    )
    command.add_argument(
        "--lock", type=Path, help="directory mutex held with --gpu-uuid"
    )
    command.add_argument(
        "--plan",
        action="store_true",
        help="price the run on the host and exit without opening a device",
    )
    command.add_argument("--trace-phases", action="store_true")
    # As with --history-carriers, no argparse ``choices=``/``type=``: a bad
    # value is refused by the door with a reason and a remedy, not by a bare
    # usage line.
    command.add_argument(
        "--profile",
        default="summary",
        metavar="{off,summary,detailed}",
        help=(
            "performance recording: 'off' records nothing; 'summary' (default, "
            "negligible overhead) records per-phase device time from stream "
            "markers, host enqueue time, kernel launches, copies, I/O stages and "
            "startup into the receipt; 'detailed' adds per-carrier I/O rows and "
            "NVTX ranges for nsys. Changes no simulated word"
        ),
    )
    command.add_argument(
        "--progress-interval",
        default="30",
        metavar="SECONDS",
        help=(
            "wall seconds between live progress lines on stderr (step, model "
            "time, s/step, ETA, last I/O cost, GPU telemetry when NVML is "
            "importable); 0 disables them. Needs --profile summary or detailed"
        ),
    )
    # No ``choices=`` either: the door refuses a bad value with its remedy.
    command.add_argument(
        "--event-parts",
        default="beside",
        metavar="{beside,in-turn}",
        help=(
            "how --integrate writes a frame and a restart taken at the same "
            "step: 'beside' (default) has the background worker write the "
            "restart while it writes the frame, renamed frame first, so the "
            "event costs the slower write, not the sum; 'in-turn' writes them "
            "one after the other. Use 'in-turn' on storage whose two writes "
            "contend, a nearly full disk, or a host without memory for both "
            "publishers' payloads at once (about 1.4 GB plus 4.4 GB at "
            "512x512x75): beside each other, a frame that meets ENOSPC or the "
            "OOM killer first is lost with its restart. Same bytes either way"
        ),
    )
    # No ``choices=``: the door refuses a bad value with its remedy.
    command.add_argument(
        "--cuda-graphs",
        default="off",
        metavar="{off,on,verify}",
        help=(
            "capture each large step's device work as a CUDA graph once and "
            "replay it ('on'), or capture every step and compare it argument "
            "byte by byte with the graph 'on' would replay ('verify'). 'off' "
            "(default) runs every step in Python. Not part of the sealed "
            "configuration; changes no simulated word"
        ),
    )
    command.add_argument(
        "--events",
        type=Path,
        metavar="PATH",
        help=(
            "write the run's gm1.run-events/v1 log to this new file: one JSON "
            "object per line, flushed per line (admitted, startup, progress at "
            "least every 10 s, frame, restart, warning, refused, end). Needs "
            "--integrate"
        ),
    )
    command.add_argument(
        "--control-dir",
        type=Path,
        metavar="DIR",
        help=(
            "check DIR once per large step: a file named 'stop' ends the run at "
            "that step boundary after the publication in flight is durable; "
            "'pause' ends it right after the next restart is durable. Either "
            "writes the receipt and exits 0. Created if absent; refused if it "
            "already holds either file. Needs --integrate"
        ),
    )
    # No ``choices=``: the door refuses a bad value with its remedy.
    command.add_argument(
        "--history-envelope",
        default="on",
        metavar="{on,off}",
        help=(
            "'on' (default) also writes each frame as history.cm1fhs, a second "
            "copy of the raw arena that gpuwm-cm1 render reads; 'off' writes "
            "the raw arena and its tables only, half the frame disk. Restarts "
            "are unaffected"
        ),
    )
    command.add_argument(
        "--history-initial",
        action="store_true",
        help=(
            "publish the initialized state as frame-000000 before the first "
            "large step, as CM1 writes its t=0 output (without dbz and cref, "
            "which nothing has computed at t=0). Needs --integrate"
        ),
    )
    command.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    command.set_defaults(handler=cartesian_door)

    from .cm1run.estimate import add_estimate_parser

    add_estimate_parser(commands)

    # The guided presets: list, show, fit and new.
    from .cm1run.presets import add_preset_parser

    add_preset_parser(commands)

    command = commands.add_parser(
        "perf", help="summarise the performance section of a Cartesian run receipt"
    )
    command.add_argument("receipt", type=Path, help="receipt written by cartesian --output")
    command.add_argument(
        "--json", action="store_true", help="print the performance section as JSON"
    )
    command.set_defaults(handler=perf_door)

    command = commands.add_parser("restart", help="resume a semantic CM1 checkpoint")
    command.add_argument("checkpoint", type=Path)
    _output(command)
    command.add_argument("--contract-sha256", required=True)
    command.add_argument("--artifact-sha256", required=True)
    command.add_argument("--operator-manifest-sha256", required=True)
    command.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    command.set_defaults(handler=restart_door)

    command = commands.add_parser(
        "render", help="render GM1 state through the native Rust path"
    )
    command.add_argument("history", type=Path)
    _output(command)
    command.add_argument("--format", dest="history_format")
    command.add_argument(
        "--contract-sha256",
        help=(
            "forecast-contract SHA-256 from the run receipt; default: the "
            "frame's sibling history-publication.json, else the envelope "
            "header. Always checked against the header"
        ),
    )
    command.add_argument(
        "--product",
        help=(
            "reflectivity (CM1 calccref column maximum) | reflectivity-level "
            "(--level K of the same volume) | updraft | vorticity | cold-pool | "
            "scalar | supercell | vortex-composite | storm-fusion"
        ),
    )
    command.add_argument("--field", action="append", default=[])
    for axis in ("nx", "ny", "nz"):
        command.add_argument(
            f"--{axis}",
            type=int,
            help="default: from the requested field's shape in the envelope header; checked against it when given",
        )
    command.add_argument("--record", type=int)
    command.add_argument("--level", type=int, default=0)
    command.add_argument("--label", default="")
    command.add_argument("--valid-seconds", type=float)
    command.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    command.set_defaults(handler=render_door)

    command = commands.add_parser(
        "cm1out",
        help="write CM1-format output (netCDF or GrADS cm1out files, statistics, parcels) from a run's frames",
        description=(
            "Convert a run's published history frames into CM1 r21.1's own output files: "
            "netCDF laid out as writeout_nc.F writes it, with CM1's names, units and "
            "global attributes, or GrADS .dat files with CM1's .ctl descriptors; CM1's "
            "statistics (cm1out_stats); CM1's parcel file (cm1out_pdata) when the run "
            "tracked parcels; and, for Morrison, dbz and cref from CM1's calc_refl10cm. "
            "The format and file layout follow the run's namelist (output_format, "
            "output_filetype) unless given here."
        ),
    )
    command.add_argument(
        "target", type=Path,
        help="a run directory (segments/NNN/frames), a cartesian output directory (frames/), or a history directory",
    )
    command.add_argument(
        "--format", choices=("nc", "grads"),
        help="nc (output_format = 2) or grads (output_format = 1); default: the namelist's",
    )
    command.add_argument(
        "--filetype", type=int, choices=(1, 2, 3),
        help=("1: one file for every time (cm1out.nc, or cm1out_s.dat ...); 2: one per time "
              "(cm1out_NNNNNN.nc, or cm1out_NNNNNN_s.dat ...); 3: CM1's per-process names; "
              "default: the namelist's output_filetype"),
    )
    command.add_argument("--out", type=Path, help="output directory; default: TARGET/cm1out")
    command.add_argument("--namelist", type=Path, help="the run's CM1 namelist; default: TARGET/namelist.input")
    command.add_argument("--deck", type=Path, help="the run's GM1 deck, used when there is no namelist")
    command.add_argument(
        "--follow", action="store_true",
        help="keep converting new frames until the run's event log records its end",
    )
    command.add_argument("--poll-s", type=float, default=2.0, help="seconds between checks with --follow (default 2)")
    command.add_argument("--force", action="store_true", help="convert every frame again from nothing")
    command.add_argument("--no-stats", action="store_true", help="do not write the statistics (cm1out_stats)")
    command.add_argument("--no-pdata", action="store_true",
                         help="do not write the parcel file (cm1out_pdata) of a run that tracked parcels")
    command.add_argument(
        "--also-netcdf", action="store_true",
        help="with GrADS output, write netCDF beside it too; default: the run's settings.toml [output] also_netcdf",
    )
    command.add_argument(
        "--deflate", type=int, default=0,
        help="netCDF compression level, 0 (off, the default) to 9; CM1 itself uses 1 (GrADS is never compressed)",
    )
    command.add_argument("--json", action="store_true", help="print the summary as JSON")
    command.set_defaults(handler=_cm1out_door)

    # Quick-looks: CPU-only pictures of a run's
    # frames, the sidecar beside cm1out.  Imported on use: numpy, never CuPy.
    from .quicklook.door import add_command as _quicklook_command

    _quicklook_command(commands)

    # Run directories: run, resume, status,
    # stop, pause and jobscript.  Standard library only, so a login node or
    # the GUI server can use them without the GPU stack.
    from .cm1run.doors import add_commands as add_run_directory_commands

    add_run_directory_commands(commands)

    # The Results screen's "Reproduce bundle": the recipe and record of a run in one folder.
    from .cm1run.reproduce import add_command as add_reproduce_command

    add_reproduce_command(commands)

    # Downscaling a WRF or ArWen run into a finer GM1 domain (3D; records/GM1-ARWEN-CHAIN.md).
    from .chain_door import add_command as add_chain_command

    add_chain_command(commands)

    # The local web GUI and the run-control commands it maps its buttons to.
    # Standard library only: importing them opens no device.
    from .gui.control import add_commands as _run_control_commands

    _run_control_commands(commands)
    return parser


def _cm1out_door(arguments: argparse.Namespace) -> int:
    # Imported on use: the converter reads numpy and netCDF4, never CuPy.
    from .cm1out.door import cm1out_door

    return cm1out_door(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if getattr(arguments, "command", None) is None:
        print(f"{START_WORDS}\nEvery command: gpuwm-cm1 --help")
        return 0
    try:
        return int(arguments.handler(arguments))
    except CM1Refusal as error:
        if getattr(arguments, "json", False):
            print(
                json.dumps(
                    error.detail.as_dict(), indent=2, sort_keys=True, allow_nan=False
                ),
                file=sys.stderr,
            )
        else:
            print(f"gpuwm-cm1: {error}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError, ValueError) as error:
        if getattr(arguments, "json", False):
            print(
                json.dumps(
                    {
                        "status": "error",
                        "code": "CM1-INPUT-ERROR",
                        "gate": "input",
                        "reason": str(error),
                        "remedy": "correct the named input and retry",
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ),
                file=sys.stderr,
            )
        else:
            print(f"gpuwm-cm1: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
