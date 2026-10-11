"""The ``gpuwm-cm1 cm1out`` door: arguments in, files and a summary out."""

from __future__ import annotations

import json
import signal
import sys
from typing import Any


def _print_summary(summary: Any) -> None:
    record = summary.record()
    print(f"Wrote {record['format']} (output_filetype = {record['output_filetype']}) to {record['out']}")
    files = record["files"]
    if files:
        shown = files if len(files) <= 4 else files[:2] + ["..."] + files[-1:]
        print(f"  history: {len(files)} file(s): {' '.join(shown)}")
    else:
        print("  history: no published frames yet")
    if record["stats_file"]:
        print(f"  statistics: {record['stats_file']} ({len(record['statistics'])} statistics)")
    if record["pdata_files"]:
        pdata = record["pdata"]
        print(f"  parcels: {' '.join(record['pdata_files'])} ({pdata.get('parcels')} parcels, "
              f"{pdata.get('times')} times)")
        for reason in (pdata.get("notes") or {}).values():
            print(f"    {reason}")
    if record["not_produced"]:
        print(f"  not written ({len(record['not_produced'])}), with the reason in each file's gm1_not_produced_reasons:")
        for name, reason in record["not_produced"].items():
            print(f"    {name}: {reason}")
    if record["switches_not_produced"]:
        print(f"  settings whose fields are not written ({len(record['switches_not_produced'])}):")
        for name, reason in record["switches_not_produced"].items():
            print(f"    {name}: {reason}")
    if record["seconds_per_frame"]:
        last = record["seconds_per_frame"][-1]
        print(f"  last frame: {last['write'] + last['stats']:.2f} s ({last['cells'] / 1e6:.2f} M cells)")
    also = record.get("also")
    if also:
        print(f"  also netCDF: {len(also['files'])} file(s)"
              + (f", {also['stats_file']}" if also.get("stats_file") else "")
              + (f", {' '.join(also['pdata_files'])}" if also.get("pdata_files") else ""))
    for name, reason in record["notes"].items():
        print(f"  note on {name}: {reason}")
    for tool, line in record["open_with"].items():
        print(f"  open with {tool}: {line}")
    if record["run_ended"]:
        print(f"  the run has ended ({record['end_reason']})")


def cm1out_door(arguments: Any) -> int:
    from .convert import Cm1outError, convert
    from .frames import FrameError

    stop = {"requested": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop["requested"] = True

    if arguments.follow:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, request_stop)
    log = (lambda message: print(message, file=sys.stderr)) if not arguments.json else None
    try:
        summary = convert(
            arguments.target,
            out=arguments.out, fmt=arguments.format, filetype=arguments.filetype,
            namelist=arguments.namelist, deck=arguments.deck, stats=not arguments.no_stats,
            deflate=arguments.deflate, force=arguments.force, log=log,
            pdata=not getattr(arguments, "no_pdata", False),
            also_netcdf=True if getattr(arguments, "also_netcdf", False) else None,
            follow=arguments.follow, poll_s=arguments.poll_s,
            should_stop=lambda: stop["requested"],
        )
    except (Cm1outError, FrameError) as error:
        raise ValueError(str(error)) from error
    if arguments.json:
        print(json.dumps(summary.record(), indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    return 0


__all__ = ["cm1out_door"]
