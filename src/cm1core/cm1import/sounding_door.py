"""``gpuwm-cm1 sounding``: a sounding from a WRF or ArWen run, for an idealized GM1 storm (the column route).

::

    gpuwm-cm1 sounding times FILE [--time T | --record N] [--map shear|cape --map-png P] [--library L] [--json]
    gpuwm-cm1 sounding from-wrf FILE [--time T | --record N] (--at LAT,LON | --ij I,J) --out FILE
                                [--map shear|cape --map-png P] [--preview-png P] [--top-m M | --top-of NAMELIST]
                                [--library L] [--json]

``times`` says what the file is (who wrote it, its grid, projection and times) and, with ``--map-png``, draws the
domain coloured by 0-6 km bulk shear or CAPE.  ``from-wrf`` writes the column at a point as CM1's ``input_sounding``
(:mod:`.wrfcolumn` has every rule), its record ``wrf-column.json`` beside it (``OUT.wrf-column.json`` unless OUT is
named ``input_sounding``), and on request the skew-T and hodograph preview.  ``--top-m`` refuses a column whose
highest level is below a deck's highest scalar level (CM1 stops on such a file); ``--top-of NAMELIST`` takes that
level from a namelist, read by the importer's own parser (the guided card names its storm's namelist).

What it is: one column's environment, the same air everywhere in the domain, flat ground, no boundary forcing.
For a GM1 domain driven by the whole run, see ``docs/WRF-DOWNSCALING.md`` (the 3D route).

Exit codes: 0 written, 2 refused (the reason and what to do; with ``--json`` also as JSON on stdout).
Nothing here opens a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

TIMES_SCHEMA = "gm1.wrf-history-facts/v1"


def record_path(out: Path) -> Path:
    """Where ``wrf-column.json`` goes for a sounding written at ``out``."""

    out = Path(out)
    return out.parent / "wrf-column.json" if out.name == "input_sounding" else out.with_name(out.name + ".wrf-column.json")


def from_wrf_command(path: Path | str, *, out: Path | str, time: str | None = None, record: int | None = None,
                     at: Sequence[float] | None = None, ij: Sequence[int] | None = None,
                     map_field: str | None = None, map_png: Path | str | None = None,
                     preview_png: Path | str | None = None, top_m: float | None = None,
                     top_of: Path | str | None = None, json_output: bool = True) -> list[str]:
    """``sounding from-wrf`` arguments (after the program name)."""

    args = ["sounding", "from-wrf", str(path)]
    if time is not None:
        args += ["--time", str(time)]
    elif record is not None:
        args += ["--record", str(int(record))]
    if at is not None:
        args.append(f"--at={float(at[0]):g},{float(at[1]):g}")
    if ij is not None:
        args += ["--ij", f"{int(ij[0])},{int(ij[1])}"]
    args += ["--out", str(out)]
    if map_png is not None:
        args += ["--map", map_field or "shear", "--map-png", str(map_png)]
    if preview_png is not None:
        args += ["--preview-png", str(preview_png)]
    if top_m is not None:
        args += ["--top-m", f"{float(top_m):g}"]
    elif top_of is not None:
        args += ["--top-of", str(top_of)]
    if json_output:
        args.append("--json")
    return args


def column_words(sounding: Path | str) -> str | None:
    """Where a sounding written by ``from-wrf`` came from, in words, from the ``wrf-column.json`` beside it
    (None for any other file): what the guided fit screen says instead of the kept file's path."""

    sounding = Path(sounding)
    record = record_path(sounding)
    try:
        column = json.loads(record.read_text(encoding="utf-8"))
        source, point = column["source"], column["point"]
        digest = column["sounding"]["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    import hashlib

    try:
        if hashlib.sha256(sounding.read_bytes()).hexdigest() != digest:
            return None                   # the file was changed after the column was written
    except OSError:
        return None
    where = (f"{point['lat']:.2f}, {point['lon']:.2f} (grid column {point['i']}, {point['j']})"
             if isinstance(point.get("lat"), (int, float)) and "requested" in point and "lat" in point["requested"]
             else f"grid column {point.get('i')}, {point.get('j')}")
    return (f"one column of your {source.get('producer', 'WRF')} run {source.get('name', '')} at {where}, "
            f"{str(source.get('times', '')).replace('_', ' ')} UTC")


def namelist_top_m(path: Path | str) -> float:
    """The highest scalar level (m) of a namelist's own vertical grid, read by the importer's parser."""

    from . import coverage, sounding as sounding_module

    effective = coverage.import_for_report(Path(path)).effective
    top = sounding_module.highest_mass_level_m(effective)
    if top is None:
        raise ValueError(f"--top-of {path}: the namelist's vertical grid could not be built, so its top is unknown")
    return float(top)


def times_command(path: Path | str, *, record: int | None = None, time: str | None = None,
                  map_field: str | None = None, map_png: Path | str | None = None, json_output: bool = True) -> list[str]:
    args = ["sounding", "times", str(path)]
    if time is not None:
        args += ["--time", str(time)]
    elif record is not None:
        args += ["--record", str(int(record))]
    if map_png is not None:
        args += ["--map", map_field or "shear", "--map-png", str(map_png)]
    if json_output:
        args.append("--json")
    return args


def _record_of(history: Any, arguments: argparse.Namespace) -> int:
    if getattr(arguments, "time", None):
        return history.record_of(arguments.time)
    if getattr(arguments, "record", None) is not None:
        return history.check_record(int(arguments.record))
    return 0


def _point(arguments: argparse.Namespace) -> dict[str, Any]:
    from .wrfcolumn import parse_pair

    if bool(arguments.at) == bool(arguments.ij):
        raise ValueError("give the point as --at LAT,LON or as --ij I,J (1-based grid point), one of them")
    if arguments.at:
        return {"at": parse_pair(arguments.at, "--at")}
    i, j = parse_pair(arguments.ij, "--ij")
    if not (float(i).is_integer() and float(j).is_integer()):
        raise ValueError(f"--ij takes whole grid numbers, as in 150,120; got {arguments.ij!r}")
    return {"ij": (int(i), int(j))}


def _picture(draw: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """A picture from wrfpictures, or why it is not drawn: the numbers and the sounding never wait on a picture."""

    from . import wrfpictures

    try:
        return getattr(wrfpictures, draw)(*args, **kwargs)
    except ImportError as error:           # matplotlib (a gpuwm dependency) missing from this Python
        return {"file": None, "drawn": False,
                "why": f"pictures need matplotlib, which this Python cannot import ({error}); pip install matplotlib"}


def times(arguments: argparse.Namespace) -> dict[str, Any]:
    from ..wrf_history import open_history

    with open_history(arguments.file) as history:
        record = _record_of(history, arguments)
        facts = history.facts()
        payload: dict[str, Any] = {"schema": TIMES_SCHEMA, "ok": True, **facts, "record": record,
                                   "times_display": [t.replace("_", " ") + " UTC" for t in history.times]}
        missing = [name for name in ("U", "V", "T", "PH", "PHB", "HGT", "QVAPOR") if not history.has(name)]
        payload["column_ready"] = not missing
        payload["missing"] = missing
        payload["surface_fields"] = {name: history.has(name) for name in ("PSFC", "TH2", "T2", "Q2", "U10", "V10")}
        payload["rotation"] = ("SINALPHA/COSALPHA" if history.has("SINALPHA") and history.has("COSALPHA")
                               else "none (grid axes earth-aligned)" if facts.get("map_proj") in (0, 3)
                               else "missing")
        if arguments.map_png:
            payload["map"] = _picture("map_png", history, record, arguments.map_png, field=arguments.map or "shear",
                                      library=arguments.library)
    return payload


def from_wrf(arguments: argparse.Namespace, *, command: str) -> dict[str, Any]:
    from ..wrf_history import open_history
    from . import wrfcolumn as wc

    point = _point(arguments)
    out = Path(arguments.out)
    with open_history(arguments.file) as history:
        record = _record_of(history, arguments)
        need = arguments.top_m
        if getattr(arguments, "top_of", None) is not None:
            need = namelist_top_m(arguments.top_of)
        column = wc.extract_column(history, record, need_top_m=need, **point)
        data = column.sounding_bytes()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.partial")
        tmp.write_bytes(data)
        os.replace(tmp, out)
        payload = column.record(sounding_file=out.name, command=command)
        report, why = wc.environment_report(column, arguments.library)
        payload["environment"] = report
        payload["environment_status"] = "computed" if report else "not-computed"
        if why:
            payload["environment_reason"] = why
        payload["numbers"] = wc.summary_numbers(report)
        if arguments.preview_png:
            payload["preview"] = _picture("preview_png", column, report, arguments.preview_png)
        if arguments.map_png:
            payload["map"] = _picture("map_png", history, record, arguments.map_png, field=arguments.map or "shear",
                                      point=column.point, library=arguments.library)
    payload["ok"] = True
    payload["sounding"]["path"] = str(out)
    payload["levels_table"] = [
        {"z_m": round(float(z), 1), "theta_k": round(float(t), 2), "qv_gkg": round(max(float(q), 0.0) * 1000.0, 3),
         "u_ms": round(float(u), 2), "v_ms": round(float(v), 2)}
        for z, t, q, u, v in zip(column.z_m, column.theta_k, column.qv_kgkg, column.u_ms, column.v_ms)]
    target = record_path(out)
    target.write_text(json.dumps({k: v for k, v in payload.items() if k not in ("levels_table",)}, indent=1,
                                 default=str) + "\n", encoding="utf-8")
    payload["record_file"] = str(target)
    return payload


def _refusal_payload(error: Exception) -> dict[str, Any]:
    from ..errors import CM1Refusal

    if isinstance(error, CM1Refusal):
        return {"ok": False, **error.detail.as_dict()}
    return {"ok": False, "status": "error", "code": "GM1-WRF-INPUT", "reason": str(error),
            "remedy": "correct the named input and retry"}


def sounding_door(arguments: argparse.Namespace) -> int:
    from ..errors import CM1Refusal

    try:
        if arguments.sounding_command == "times":
            payload = times(arguments)
        else:
            point = _point(arguments)
            command = shlex.join(["gpuwm-cm1", *from_wrf_command(
                arguments.file, out=arguments.out, time=arguments.time, record=arguments.record,
                at=point.get("at"), ij=point.get("ij"), map_field=arguments.map, map_png=arguments.map_png,
                preview_png=arguments.preview_png, top_m=arguments.top_m, top_of=arguments.top_of,
                json_output=arguments.json)])
            payload = from_wrf(arguments, command=command)
    except (CM1Refusal, ValueError, OSError) as error:
        payload = _refusal_payload(error)
        if arguments.json:
            print(json.dumps(payload, indent=1, default=str))
        else:
            print(f"gpuwm-cm1 sounding: {payload.get('reason')} ({payload.get('remedy')})", file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(payload, indent=1, default=str))
    elif arguments.sounding_command == "times":
        print(f"{payload['name']}: {payload['producer']}, domain {payload['domain_id']}, {payload['format']}, "
              f"{payload['nx']} x {payload['ny']} x {payload['nz']} at {payload['dx_m']} m, "
              f"{payload['map_proj_name']}")
        for n, stamp in enumerate(payload["times"]):
            print(f"  record {n}: {stamp}")
    else:
        s = payload["surface"]
        print(f"wrote {payload['sounding']['path']} ({payload['levels']} levels to {payload['top_m']:,.0f} m; "
              f"surface {s['pressure_hpa']:.1f} hPa, theta {s['theta_k']:.2f} K, qv {s['qv_gkg']:.2f} g/kg)")
        print(f"record: {payload['record_file']}")
        n = payload.get("numbers") or {}
        if n:
            print(f"CAPE {n.get('cape_sb', 0):.0f} J/kg, CIN {n.get('cin_sb', 0):.0f} J/kg, 0-6 km shear "
                  f"{n.get('shear_0_6km', 0):.1f} m/s, 0-3 km SRH {n.get('srh_0_3km', 0):.0f} m2/s2")
        print(payload["coverage_note"])
    return 0


def add_parser(commands: Any) -> None:
    """Register ``gpuwm-cm1 sounding`` on the CLI's subparsers."""

    command = commands.add_parser(
        "sounding",
        help="a CM1 input_sounding from one column of a WRF or ArWen run",
        description=(
            "Read a WRF or ArWen history file (wrfout; CDF-1, CDF-2, CDF-5 or netCDF-4) and write one column's "
            "environment as CM1's input_sounding (isnd = 7), for an idealized GM1 storm: the same air everywhere, "
            "flat ground, no boundary forcing. For a GM1 domain driven by the whole run, use "
            "gpuwm-cm1 chain."
        ),
    )
    sub = command.add_subparsers(dest="sounding_command", required=True)
    for name, help_text in (("times", "what the file is and its times; optionally the domain map"),
                            ("from-wrf", "write the column at a point as input_sounding")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("file", type=Path, help="a wrfout or ArWen history file")
        when = p.add_mutually_exclusive_group()
        when.add_argument("--time", help="one of the file's times, YYYY-MM-DD_HH:MM:SS")
        when.add_argument("--record", type=int, help="the record number (0 is the first)")
        p.add_argument("--map", choices=("shear", "cape"), default=None,
                       help="what the domain map shows (default shear: 0-6 km bulk shear)")
        p.add_argument("--map-png", type=Path, default=None, help="draw the domain map to this PNG (960 x 540)")
        p.add_argument("--library", type=Path, default=None,
                       help="Rust initializer library for CM1's getcape numbers (default: installed)")
        p.add_argument("--json", action="store_true", help="print the result as JSON")
        if name == "from-wrf":
            p.add_argument("--at", help="the point as LAT,LON (the nearest grid column is used)")
            p.add_argument("--ij", help="the point as a 1-based grid column I,J (I west-east, J south-north)")
            p.add_argument("--out", type=Path, required=True, help="the input_sounding to write")
            p.add_argument("--preview-png", type=Path, default=None,
                           help="draw the column's skew-T and hodograph to this PNG")
            top = p.add_mutually_exclusive_group()
            top.add_argument("--top-m", type=float, default=None,
                             help="refuse a column whose highest level is below this height (a deck's highest level)")
            top.add_argument("--top-of", type=Path, default=None, metavar="NAMELIST",
                             help="refuse a column whose highest level is below this namelist's highest level")
        p.set_defaults(handler=sounding_door, sounding_command=name)


__all__ = ["add_parser", "column_words", "from_wrf", "from_wrf_command", "namelist_top_m", "record_path", "sounding_door", "times",
           "times_command"]
