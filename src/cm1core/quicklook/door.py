"""The ``gpuwm-cm1 quicklook`` door: arguments in, pictures and a JSON summary out.

    gpuwm-cm1 quicklook TARGET [--follow] [--only IDS] [--out DIR] [--force] [--label NAME] [--json]
                            [--namelist FILE | --deck FILE]   (every form: for frames with neither beside them)
    gpuwm-cm1 quicklook TARGET --slice FIELD [--height M | --level K] [--line XA YA XB YB] [--time S] [--json]
    gpuwm-cm1 quicklook TARGET --section FIELD --from XA YA --to XB YB [--top M] [--time S] [--json]
    gpuwm-cm1 quicklook TARGET --column [--at X Y] [--time S] [--json]
    gpuwm-cm1 quicklook TARGET --fields [--time S] [--json]
    gpuwm-cm1 quicklook TARGET --look environment|parcels [--json]     (a look drawn once per run, drawn again now)
    gpuwm-cm1 quicklook TARGET --film LOOK [--format apng|mp4|gif] [--fps N] [--between K] [--json]

The first process line makes this process a sidecar: no GPU visible, nice 19, single-threaded pools
(:func:`.render.worker_environment`), before numpy or matplotlib is imported.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any


def add_command(commands: Any) -> None:
    command = commands.add_parser(
        "quicklook",
        help="draw a run's quick-look pictures (CPU only), or a slice or column skew-T of one frame",
        description=(
            "Draw the quick-looks of a run's published frames: reflectivity at 1 km, spin near the ground, updraft "
            "at 5 km, the wind swath, the lowest level θ′ and wind, and rain, with the time series. With --follow, "
            "keep drawing new frames until the run ends. --slice, --section and --column draw one picture of one "
            "frame. Runs on the CPU at nice 19 with no GPU visible."
        ),
    )
    command.add_argument("target", type=Path,
                         help="a run directory (segments/NNN/frames), a cartesian output directory (frames/), "
                              "or a history directory")
    command.add_argument("--follow", action="store_true",
                         help="keep drawing new frames until the run's event log records its end")
    command.add_argument("--poll-s", type=float, default=2.0, help="seconds between checks with --follow (default 2)")
    command.add_argument("--only", default="", help="comma list of quick-looks to draw (default: all)")
    command.add_argument("--out", type=Path, help="output directory; default: TARGET/quicklook")
    command.add_argument("--force", action="store_true", help="draw every frame again from nothing")
    command.add_argument("--label", help="the name the pictures give the run (default: its run.json name or folder)")
    command.add_argument("--namelist", type=Path,
                         help="the run's CM1 namelist, when TARGET has none beside its frames (as for cm1out)")
    command.add_argument("--deck", type=Path, help="the run's GM1 deck, when TARGET has no namelist or deck")
    command.add_argument("--slice", metavar="FIELD", help="a horizontal slice of this cm1out field")
    command.add_argument("--section", metavar="FIELD", help="a vertical section of this cm1out field along --from/--to")
    command.add_argument("--column", action="store_true", help="the skew-T of one model column")
    command.add_argument("--fields", action="store_true", help="list the cm1out fields a slice can show")
    command.add_argument("--height", type=float, help="slice height above ground (m); the nearest level is used")
    command.add_argument("--level", type=int, help="slice level k, counted from 1 at the bottom as in CM1")
    command.add_argument("--line", nargs=4, type=float, metavar=("XA", "YA", "XB", "YB"),
                         help="km: draw the section line A-B on a horizontal slice")
    command.add_argument("--from", dest="point_a", nargs=2, type=float, metavar=("XA", "YA"),
                         help="km, in the run's own x, y (the cm1out file's xh, yh)")
    command.add_argument("--to", dest="point_b", nargs=2, type=float, metavar=("XB", "YB"), help="km")
    command.add_argument("--top", type=float, help="top of a vertical section (m); default 18 km or the model top")
    command.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"),
                         help="km: the column for --column (default: the strongest updraft at 5 km)")
    command.add_argument("--time", type=float, help="model time (s) of the frame; default the latest")
    command.add_argument("--look", choices=("environment", "parcels"),
                         help="draw a look that is drawn once per run again now: the environment's skew-T and "
                              "hodograph, or the air parcels (runs with parcels, once the run has ended)")
    command.add_argument("--film", metavar="LOOK",
                         help="make a film of a look: any per-frame look (swath, reflectivity, ...), swirl (the "
                              "spin look's streaks animated) or parcels; written to TARGET/quicklook/films/LOOK/")
    command.add_argument("--format", dest="formats", action="append", choices=("apng", "mp4", "gif"),
                         help="film formats besides the pictures (repeatable). Default: the animated PNG, and an "
                              "MP4 when imageio-ffmpeg is installed (the installer adds it; with pip: "
                              "pip install \"gpuwm-cm1[film]\"), else a GIF")
    command.add_argument("--fps", type=float, help="film pictures per second (default: 6 for a look's saved "
                                                   "pictures, 18 for swirl, 12 for parcels)")
    command.add_argument("--between", type=int, default=8,
                         help="swirl: blended pictures between two saved times (default 8)")
    command.add_argument("--json", action="store_true", help="print the result as JSON")
    command.set_defaults(handler=quicklook_door)


def _numbers(values: Any) -> tuple[float, ...] | None:
    return None if values is None else tuple(float(v) for v in values)


def quicklook_door(arguments: argparse.Namespace) -> int:
    from .worker import worker_environment

    worker = worker_environment()
    if arguments.film:
        return _film(arguments, worker)
    if arguments.look:
        return _look(arguments, worker)
    one_picture = arguments.slice or arguments.section or arguments.column or arguments.fields
    if one_picture:
        return _one_picture(arguments, worker)
    from ..cm1out.frames import FrameError
    from ..cm1out.settings import SettingsError
    from .render import QuicklookError, render

    stop = {"requested": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop["requested"] = True

    if arguments.follow:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, request_stop)
    log = (lambda message: print(message, file=sys.stderr, flush=True)) if not arguments.json else None
    only = tuple(p for p in arguments.only.split(",") if p) or None
    try:
        summary = render(arguments.target, follow=arguments.follow, poll_s=arguments.poll_s,
                         should_stop=lambda: stop["requested"], out=arguments.out, only=only,
                         force=arguments.force, label=arguments.label, log=log,
                         namelist=arguments.namelist, deck=arguments.deck)
    except (QuicklookError, FrameError, SettingsError) as error:
        raise ValueError(str(error)) from error
    record = summary.record()
    if arguments.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(f"Drawn: {record['frames_drawn_now']} frame(s) of {record['frames_total']} into {record['out']}")
        if record["seconds_per_frame"]:
            last = record["seconds_per_frame"][-1]
            print(f"  last frame: {last['seconds']:.1f} s ({last['cells'] / 1e6:.2f} M cells)")
        for problem in record["problems"]:
            print(f"  not drawn: {problem}")
        print(f"  index: {record['index']}")
        print(f"  worker: nice {worker['nice']}, CUDA_VISIBLE_DEVICES={worker['cuda_visible_devices']!r}")
        if record["run_ended"]:
            print(f"  the run has ended ({record['end_reason']})")
    return 0


def _film(arguments: argparse.Namespace, worker: dict[str, Any]) -> int:
    from ..cm1out.frames import FrameError, discover
    from . import film as F
    from .render import default_out

    stop = {"requested": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop["requested"] = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    try:
        layout = discover(arguments.target)
    except FrameError as error:
        raise ValueError(str(error)) from error
    out = arguments.out or default_out(layout)
    look = arguments.film
    fps = arguments.fps or F.DEFAULT_FPS.get(look, 6.0)
    if not 0.5 <= fps <= 60:
        raise ValueError("--fps must be from 0.5 to 60 pictures per second")
    if not 0 <= arguments.between <= 30:
        raise ValueError("--between must be from 0 to 30 pictures")
    log = (lambda message: print(message, file=sys.stderr, flush=True)) if not arguments.json else (lambda _m: None)
    job = F.FilmJob(Path(arguments.target), look, tuple(arguments.formats or ()), float(fps), int(arguments.between),
                    Path(out), label=arguments.label, log=log, should_stop=lambda: stop["requested"])
    try:
        record = F.make(job)
    except F.FilmError as error:
        try:
            job.progress(0, 0, state="failed", message=str(error))
        except OSError:
            pass
        raise ValueError(str(error)) from error
    record["worker"] = worker
    record["out"] = os.fspath(out)
    if arguments.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        state = "stopped after" if record["stopped"] else "made from"
        print(f"Film of {look} {state} {record['pictures']} pictures in {record['seconds']:.1f} s:")
        for kind, name in record["files"].items():
            print(f"  {kind}: {Path(out) / name}")
    return 0


def _look(arguments: argparse.Namespace, worker: dict[str, Any]) -> int:
    from ..cm1out.frames import FrameError
    from .frame import frames_of
    from .render import QuicklookError, Renderer

    try:
        renderer = Renderer(arguments.target, out=arguments.out, only=(arguments.look,), label=arguments.label,
                            namelist=arguments.namelist, deck=arguments.deck)
    except (QuicklookError, FrameError) as error:
        raise ValueError(str(error)) from error
    lineage = frames_of(renderer.layout)
    if not lineage:
        raise ValueError(f"{arguments.target} has no published frames yet")
    renderer.state["once"].pop(arguments.look, None)
    renderer.draw_once(lineage)
    renderer._save_state()
    index = renderer.write_index(lineage)
    row = renderer.state["once"].get(arguments.look) or {}
    record: dict[str, Any] = {"kind": "look", "look": arguments.look, "worker": worker}
    if "file" in row:
        record.update({"file": row["file"], "path": os.fspath(renderer.out / row["file"]), "time_s": row["time"],
                       "facts": row.get("facts") or {}})
    else:
        ended, _end = renderer.layout.ended()
        why = row.get("not_drawn") or row.get("error") or ""
        if arguments.look == "parcels" and not why:
            why = ("Parcels arrive when the run ends." if not ended
                   else "This run did not follow air parcels. To follow them, set CM1's iprcl = 1 for the next run: in Advanced, Namelist, or with gpuwm-cm1 edit RUNDIR --set iprcl=1 before it starts.")
        raise ValueError(why or "The look was not drawn.")
    record["index"] = os.fspath(renderer.out / "index.json")
    record["products"] = len(index.get("products") or [])
    if arguments.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(f"Wrote {record['path']}")
    return 0


def _one_picture(arguments: argparse.Namespace, worker: dict[str, Any]) -> int:
    from ..cm1out import frames as _frames
    from ..cm1out import settings as _settings
    from . import skewt, slices
    from .facts import run_facts
    from .frame import QuickFrame, frames_of
    from .render import default_out, _tag

    layout = _frames.discover(arguments.target)
    lineage = frames_of(layout)
    if not lineage:
        raise ValueError(f"{arguments.target} has no published frames yet")
    if arguments.time is None:
        t, key, path, segment = lineage[-1]
    else:
        t, key, path, segment = min(lineage, key=lambda row: abs(row[0] - arguments.time))
    if arguments.namelist is not None or arguments.deck is not None:
        settings = _settings.resolve(layout, namelist=arguments.namelist, deck=arguments.deck)
    else:
        try:
            settings = _settings.resolve(layout)
        except _settings.SettingsError:
            settings = None
    out_dir = (arguments.out or (default_out(layout))) / "slices"
    qf = QuickFrame.open(path, key=key, segment=segment)
    first = None
    try:
        facts = run_facts(layout, qf.config, qf.dx, qf.dy, label=arguments.label, settings=settings)
        tag = _tag(qf.t)
        if arguments.fields:
            record: dict[str, Any] = {"kind": "fields", "time_s": qf.t, "frame": key,
                                      "fields": slices.field_list(qf, settings)}
        elif arguments.slice:
            record = slices.horizontal(qf, facts, settings, arguments.slice, out_dir, height_m=arguments.height,
                                       level=arguments.level, line_km=_numbers(arguments.line), tag=tag)
        elif arguments.section:
            a = _numbers(arguments.point_a)
            b = _numbers(arguments.point_b)
            if a is None or b is None:
                raise ValueError("--section needs --from XA YA and --to XB YB (km)")
            name = "_".join(f"{v:g}" for v in (*a, *b)).replace("-", "m")
            out = out_dir / f"section-{arguments.section}-{name}-{tag}.png"
            record = slices.vertical(qf, facts, settings, arguments.section, out, a_km=a, b_km=b, top_m=arguments.top)
        else:
            at = _numbers(arguments.at)
            if lineage[0][1] != key:
                first = QuickFrame.open(lineage[0][2], key=lineage[0][1], segment=lineage[0][3])
            where = "max" if at is None else "_".join(f"{v:g}" for v in at).replace("-", "m")
            out = out_dir / f"column-{where}-{tag}.png"
            record = skewt.column(qf, facts, out, x_km=None if at is None else at[0],
                                  y_km=None if at is None else at[1], first=first or qf)
    except slices.SliceError as error:
        raise ValueError(str(error)) from error
    finally:
        qf.close()
        if first is not None:
            first.close()
    if "file" in record:
        record["file"] = f"slices/{record['file']}"
        record["path"] = os.fspath(out_dir.parent / record["file"])
    record["worker"] = worker
    if arguments.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    elif arguments.fields:
        for row in record["fields"]:
            print(f"{row['name']:12s} {row['grid']}  {row['units']:10s} {row['description']}")
    else:
        print(f"Wrote {record['path']}")
    return 0


__all__ = ["add_command", "quicklook_door"]
