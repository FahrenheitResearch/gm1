"""The ``gpuwm-cm1 quicklook`` door: arguments in, pictures and a JSON summary out.

    gpuwm-cm1 quicklook TARGET [--follow] [--only IDS] [--out DIR] [--force] [--label NAME] [--json]
                            [--namelist FILE | --deck FILE]   (every form: for frames with neither beside them)
    gpuwm-cm1 quicklook TARGET --slice FIELD [--height M | --level K] [--line XA YA XB YB] [--time S] [--json]
    gpuwm-cm1 quicklook TARGET --section FIELD --from XA YA --to XB YB [--top M] [--time S] [--json]
    gpuwm-cm1 quicklook TARGET --column [--at X Y] [--time S] [--json]
    gpuwm-cm1 quicklook TARGET --fields [--time S] [--json]
    gpuwm-cm1 quicklook TARGET --look environment|parcels|run-card|updraft-time-height|spin-time-height [--json]
                                                     (a look drawn once per run, drawn now)
    gpuwm-cm1 quicklook TARGET --look storm3d [--time S] [--bearing DEG] [--size WxH] [--json]   (the storm in 3-D)
    gpuwm-cm1 quicklook TARGET --film LOOK [--format apng|mp4|gif] [--fps N] [--between K] [--json]
    gpuwm-cm1 quicklook TARGET [--radar-site X Y | --radar-site storm] [--radar-tilt DEG]
                            [--radar-motion U V | --radar-motion auto] [--json]   (the radar looks, drawn again)
    gpuwm-cm1 quicklook TARGET --compare OTHER [OTHER ...] [--time S] [--box KM] [--only LOOKS] [--out DIR] [--json]
    gpuwm-cm1 quicklook TARGET --film storm3d [--time S] [--orbit-frames N] [--size WxH] [--json]   (one turn)

The first process line makes this process a sidecar: no GPU visible, nice 19, single-threaded pools
(:func:`.render.worker_environment`), before numpy or matplotlib is imported.
"""

from __future__ import annotations

import argparse
import json
import math
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
                              "a history directory, or CM1 netCDF output: a cm1out.nc or cm1out_NNNNNN.nc file, "
                              "or a folder of them (CM1's own, or a GM1 run's cm1out/)")
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
    command.add_argument("--look", choices=("environment", "parcels", "run-card", "updraft-time-height",
                                            "spin-time-height", "storm3d"),
                         help="draw a look that is drawn once per run again now: the environment's skew-T and "
                              "hodograph, the air parcels (runs with parcels, once the run has ended), the run card "
                              "(how this storm was simulated, from the run's own records), or the storm over time "
                              "(the updraft or the spin at every height, one column per saved time); or storm3d, "
                              "the storm in 3-D cut open through its core, at --time (default the latest saved "
                              "time), drawn on request on the CPU")
    command.add_argument("--bearing", type=float,
                         help="storm3d: where the camera stands, a compass bearing from the storm in degrees "
                              "(default 150, south-south-east, looking north-north-west)")
    command.add_argument("--size", help="storm3d: picture size WxH (default 1920x1080 for a still, 1280x720 for "
                                        "the orbit film)")
    command.add_argument("--orbit-frames", type=int,
                         help="--film storm3d: pictures in one turn around the storm (default 72, 5 degrees apart)")
    command.add_argument("--film", metavar="LOOK",
                         help="make a film of a look: any per-frame look (swath, reflectivity, ...), swirl (the "
                              "spin look's streaks animated), parcels, or storm3d (one turn around the storm in 3-D "
                              "at --time); written to TARGET/quicklook/films/LOOK/")
    command.add_argument("--format", dest="formats", action="append", choices=("apng", "mp4", "gif"),
                         help="film formats besides the pictures (repeatable). Default: the animated PNG, and an "
                              "MP4 when imageio-ffmpeg is installed (the installer adds it; with pip: "
                              "pip install \"gpuwm-cm1[film]\"), else a GIF")
    command.add_argument("--fps", type=float, help="film pictures per second (default: 6 for a look's saved "
                                                   "pictures, 18 for swirl, 12 for parcels and storm3d)")
    command.add_argument("--between", type=int, default=8,
                         help="swirl: blended pictures between two saved times (default 8)")
    command.add_argument("--radar-site", nargs="+", metavar="X_Y",
                         help="the radar looks: put the radar at X Y km on the ground (the run's own x, y at the "
                              "start; it stays there when the domain moves), or 'storm' for the default, 40 km "
                              "south-east of the storm (120 km from a hurricane's eye) in each picture; the radar "
                              "looks of every saved time are drawn again")
    command.add_argument("--radar-tilt", type=float, metavar="DEG",
                         help="the radar looks: the scan's elevation angle, 0.1 to 20 degrees (default 0.5)")
    command.add_argument("--radar-motion", nargs="+", metavar="U_V",
                         help="the storm-relative radar look: the storm motion to take out, U V m/s east and north "
                              "(ground-relative), or 'auto' for the run's own (the domain's motion, else the "
                              "Bunkers right mover of its starting air, or its 0-6 km mean wind where the 0-6 km "
                              "shear is under 10 m/s)")
    command.add_argument("--compare", nargs="+", type=Path, metavar="OTHER",
                         help="compare TARGET with up to five other runs: side by side at one storm time (--time, "
                              "default the latest time all reached), over time, and their soundings; written to "
                              "TARGET/quicklook/compare/ (with --only: storm, series, soundings)")
    command.add_argument("--box", type=float, metavar="KM",
                         help="--compare: the width of each run's storm box (default 48 km; 500 km for hurricanes)")
    command.add_argument("--json", action="store_true", help="print the result as JSON")
    command.set_defaults(handler=quicklook_door)


def _numbers(values: Any) -> tuple[float, ...] | None:
    return None if values is None else tuple(float(v) for v in values)


def _size(text: str | None) -> tuple[int, int] | None:
    """--size WxH, from 960x540 (smaller, the words would not fit) to 3840x2160."""

    if text is None:
        return None
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"--size takes WxH in pixels, such as 1280x720, not {text!r}") from None
    if not (960 <= width <= 3840 and 540 <= height <= 2160):
        raise ValueError("--size must be from 960x540 to 3840x2160 pixels")
    return width, height


def _bearing(value: float | None) -> float | None:
    """--bearing: a compass bearing in degrees (a finite number), checked before anything is read."""

    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError("--bearing takes a compass bearing in degrees, such as 240")
    return float(value)


def _storm_time(value: float | None) -> float | None:
    """--time of the comparison and the storm in 3-D: a storm time in seconds (finite, 0 or more), checked before
    anything is read."""

    if value is None:
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError("--time is a storm time in seconds, 0 or more.")
    return float(value)


def quicklook_door(arguments: argparse.Namespace) -> int:
    from .worker import worker_environment

    worker = worker_environment()
    if arguments.compare:
        return _compare(arguments, worker)
    if arguments.radar_site is not None or arguments.radar_tilt is not None or arguments.radar_motion is not None:
        code = _radar(arguments, worker)
        if not arguments.film:
            return code
    if arguments.film:
        return _film(arguments, worker)
    if arguments.look:
        return _look(arguments, worker)
    one_picture = arguments.slice or arguments.section or arguments.column or arguments.fields
    if one_picture:
        return _one_picture(arguments, worker)
    from ..cm1out.frames import FrameError
    from ..cm1out.settings import SettingsError
    from .ncread import looks_like_netcdf_target
    from .render import QuicklookError, render

    if arguments.follow and looks_like_netcdf_target(arguments.target):
        # refused before anything is written beside the files (a follower takes a lock in its folder)
        raise ValueError("CM1 netCDF files are drawn as they are: --follow is for a GM1 run that is still going. "
                         "Leave out --follow.")

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
    from ..cm1out.frames import FrameError
    from . import film as F
    from .ncread import discover
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
    extra: dict[str, Any] = {}
    if look == "storm3d":
        if arguments.orbit_frames is not None and not 12 <= arguments.orbit_frames <= 360:
            raise ValueError("--orbit-frames must be from 12 to 360 pictures")
        extra = {"time_s": _storm_time(arguments.time), "pictures": arguments.orbit_frames,
                 "size": _size(arguments.size), "bearing": _bearing(arguments.bearing)}
    job = F.FilmJob(Path(arguments.target), look, tuple(arguments.formats or ()), float(fps), int(arguments.between),
                    Path(out), label=arguments.label, log=log, should_stop=lambda: stop["requested"], **extra)
    try:
        record = F.make(job)
    except F.FilmError as error:
        if not isinstance(error, F.FilmKept):      # a film kept as it is keeps its progress too
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


def _pair(values: list[str] | None, word: str, what: str) -> tuple[float, float] | None | str:
    """Two numbers, or ``word`` (the default), from an option that takes either."""

    if values is None:
        return "keep"
    if len(values) == 1 and values[0].strip().lower() == word:
        return None
    try:
        numbers = [float(v) for v in values]
    except ValueError:
        numbers = []
    if len(numbers) != 2:
        raise ValueError(f"{what} takes two numbers or the word {word}, not {' '.join(values)!r}.")
    return numbers[0], numbers[1]


def _radar(arguments: argparse.Namespace, worker: dict[str, Any]) -> int:
    """Write the run's radar settings and draw the radar looks of every saved time again."""

    from contextlib import nullcontext
    from dataclasses import replace

    from ..cm1out.follow_lock import FollowerStopped, follow_lock
    from ..cm1out.frames import FrameError
    from . import radar as RA
    from .frame import frames_of
    from .ncread import looks_like_netcdf_target
    from .render import RADAR_IDS, QuicklookError, Renderer, has_saved_frames, lost_times, radar_lost

    site = _pair(arguments.radar_site, "storm", "--radar-site")
    motion = _pair(arguments.radar_motion, "auto", "--radar-motion")
    log = (lambda message: print(message, file=sys.stderr, flush=True)) if not arguments.json else None
    # CM1 netCDF files are drawn as they are and never followed, so no follower's lock is taken in their folder.
    lock = (nullcontext(False) if looks_like_netcdf_target(Path(arguments.target))
            else follow_lock(arguments.target, "quicklook", should_stop=lambda: True))
    try:
        with lock:
            renderer = Renderer(arguments.target, out=arguments.out, only=RADAR_IDS, label=arguments.label,
                                log=log, namelist=arguments.namelist, deck=arguments.deck)
            if renderer.state["frames"] and not has_saved_frames(renderer.layout):
                # its radar pictures would be forgotten and none drawn again
                raise QuicklookError("This run's saved frames were deleted, so the radar cannot scan it again. "
                                     "Nothing was changed; its pictures are kept.")
            lost = lost_times(renderer.layout, frames_of(renderer.layout), renderer.state["frames"])
            if lost:
                # the times whose frames are gone would keep pictures from the old radar beside the new ones
                raise QuicklookError(radar_lost(len(lost), len(renderer.state["frames"])))
            settings = RA.read_settings(renderer.out)
            if site != "keep":
                settings = replace(settings, site_km=site)
            if motion != "keep":
                settings = replace(settings, motion_m_s=motion)
            if arguments.radar_tilt is not None:
                settings = replace(settings, tilt_deg=float(arguments.radar_tilt))
            RA.write_settings(renderer.out, settings.check())
            again = renderer.forget(RADAR_IDS)
            drawn = renderer.run_pass()
            lineage = frames_of(renderer.layout)
            if lineage:
                renderer.write_index(lineage)
    except FollowerStopped as error:
        raise ValueError("The run's quick-look worker is drawing its pictures now. Set the radar once the run has "
                         "ended; the radar looks of every saved time are then drawn again.") from error
    except (RA.SettingsError, QuicklookError, FrameError) as error:
        raise ValueError(str(error)) from error
    record: dict[str, Any] = {"kind": "radar", "settings": settings.record(), "pictures_redrawn": again,
                              "frames_drawn": drawn, "problems": renderer.summary.problems,
                              "out": os.fspath(renderer.out), "index": os.fspath(renderer.out / "index.json"),
                              "worker": worker}
    if arguments.json and not arguments.film:
        print(json.dumps(record, indent=2, sort_keys=True))
    elif not arguments.json:
        where = ("beside the storm in each picture" if settings.site_km is None
                 else f"at x {settings.site_km[0]:g} km, y {settings.site_km[1]:g} km on the ground")
        print(f"Radar {where}, {settings.tilt_deg:g} degree scan: drew the radar looks of {drawn} saved time(s) "
              f"into {renderer.out}")
        for problem in renderer.summary.problems:
            print(f"  not drawn: {problem}")
    return 0


def _look(arguments: argparse.Namespace, worker: dict[str, Any]) -> int:
    from ..cm1out.frames import FrameError
    from .frame import frames_of
    from .render import FRAMES_GONE, QuicklookError, Renderer, has_saved_frames

    if arguments.look == "storm3d":
        return _storm3d(arguments, worker)

    try:
        renderer = Renderer(arguments.target, out=arguments.out, only=(arguments.look,), label=arguments.label,
                            namelist=arguments.namelist, deck=arguments.deck,
                            log=None if arguments.json else (lambda message: print(message, file=sys.stderr,
                                                                                   flush=True)))
    except (QuicklookError, FrameError) as error:
        raise ValueError(str(error)) from error
    lineage = frames_of(renderer.layout)
    if not lineage and arguments.look != "run-card":       # the card can be drawn from a run's receipts alone
        if renderer.state["frames"] and not has_saved_frames(renderer.layout):
            raise ValueError(FRAMES_GONE)
        raise ValueError(f"{arguments.target} has no published frames yet")
    # A look that would be drawn again from less than its picture shows (frames or a parcel track deleted since) is
    # refused, and a look that is not drawn this time keeps the picture it had.
    refused = renderer.redraw_refused(arguments.look, lineage)
    if refused:
        raise ValueError(refused)
    previous = renderer.state["once"].pop(arguments.look, None)
    renderer.draw_once(lineage)
    if "file" not in (renderer.state["once"].get(arguments.look) or {}) and "file" in (previous or {}):
        failed = renderer.state["once"].get(arguments.look) or {}
        renderer.state["once"][arguments.look] = previous
        renderer._save_state()
        renderer.write_index(lineage)
        why = failed.get("not_drawn") or failed.get("error") or "The look was not drawn."
        raise ValueError(f"{why} The picture it had is kept.")
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
        if arguments.look == "parcels" and not why and renderer.layout.kind == "netcdf":
            from .ncread import PARCELS_NOT_READ

            why = PARCELS_NOT_READ
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


def _compare(arguments: argparse.Namespace, worker: dict[str, Any]) -> int:
    from ..cm1out.frames import FrameError
    from . import compare as CO

    log = (lambda message: print(message, file=sys.stderr, flush=True)) if not arguments.json else (lambda _m: None)
    only = tuple(p for p in arguments.only.split(",") if p) or None
    when = _storm_time(arguments.time)
    try:
        record = CO.make([arguments.target, *arguments.compare], time_s=when, box_km=arguments.box,
                         only=only, out=arguments.out, log=log)
    except (CO.CompareError, FrameError) as error:
        raise ValueError(str(error)) from error
    record["worker"] = worker
    if arguments.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        folder = Path(record["folder"])
        for value in record["pictures"].values():
            for row in value if isinstance(value, list) else [value]:
                print(f"Wrote {folder / row['file']}")
        for key, why in record["not_drawn"].items():
            print(f"  not drawn: {key}: {why}")
    return 0


def _storm3d(arguments: argparse.Namespace, worker: dict[str, Any]) -> int:
    from ..cm1out.frames import FrameError
    from . import products as P
    from . import storm3d as D
    from .render import QuicklookError

    log = (lambda message: print(message, file=sys.stderr, flush=True)) if not arguments.json else (lambda _m: None)
    when = _storm_time(arguments.time)
    try:
        record = D.draw_still(arguments.target, out=arguments.out, time_s=when,
                              bearing=D.BEARING if arguments.bearing is None else _bearing(arguments.bearing),
                              size=_size(arguments.size) or D.STILL_SIZE, label=arguments.label,
                              namelist=arguments.namelist, deck=arguments.deck, log=log)
    except (QuicklookError, FrameError, P.NotDrawn) as error:
        raise ValueError(str(error)) from error
    record.update({"kind": "look", "look": "storm3d", "worker": worker})
    if arguments.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(f"Wrote {record['path']} ({record['seconds']:.0f} s)")
    return 0


def _one_picture(arguments: argparse.Namespace, worker: dict[str, Any]) -> int:
    from ..cm1out import settings as _settings
    from . import skewt, slices
    from .facts import read_settings, run_facts
    from .frame import FieldMissing, QuickFrame, frames_of
    from .ncread import discover
    from .render import default_out, _tag

    layout = discover(arguments.target)
    lineage = frames_of(layout)
    if not lineage:
        raise ValueError(f"{arguments.target} has no published frames yet")
    if arguments.time is None:
        t, key, path, segment = lineage[-1]
    else:
        t, key, path, segment = min(lineage, key=lambda row: abs(row[0] - arguments.time))
    note = None
    if arguments.namelist is not None or arguments.deck is not None:
        settings = _settings.resolve(layout, namelist=arguments.namelist, deck=arguments.deck)
        if layout.kind == "netcdf":
            layout.use_settings(settings)
    elif layout.kind == "netcdf":
        settings = layout.settings
    else:
        settings, note = read_settings(layout)
        if note is not None:
            # One plain line: without the settings a hurricane is drawn as a thunderstorm run.
            print(note, file=sys.stderr, flush=True)
    out_dir = (arguments.out or (default_out(layout))) / "slices"
    qf = QuickFrame.open(path, key=key, segment=segment)
    first = None
    try:
        facts = run_facts(layout, qf.config, qf.dx, qf.dy, label=arguments.label, settings=settings)
        tag = _tag(qf.t)
        if arguments.fields:
            record: dict[str, Any] = {"kind": "fields", "time_s": qf.t, "frame": key,
                                      "fields": slices.field_list(qf, settings)}
            if layout.kind == "netcdf":
                # what the files hold, for the GUI's Open CM1 output (its list row and header before the pictures)
                record["saved_times_s"] = [lineage[0][0], lineage[-1][0]]
                record["saved_count"] = len(lineage)
                record["grid"] = {"nx": qf.nx, "ny": qf.ny, "nz": qf.nz, "dx_m": qf.dx}
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
    except (slices.SliceError, FieldMissing) as error:
        raise ValueError(str(error)) from error
    finally:
        qf.close()
        if first is not None:
            first.close()
    if "file" in record:
        record["file"] = f"slices/{record['file']}"
        record["path"] = os.fspath(out_dir.parent / record["file"])
    if note is not None:
        record["note"] = note
    record["worker"] = worker
    notes = list(getattr(layout, "notes", []) or [])
    if notes:
        record["notes"] = notes
    if arguments.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    elif arguments.fields:
        for row in record["fields"]:
            print(f"{row['name']:12s} {row['grid']}  {row['units']:10s} {row['description']}")
    else:
        print(f"Wrote {record['path']}")
    if not arguments.json:
        for note in notes:
            print(f"  note: {note}", file=sys.stderr)
    return 0


__all__ = ["add_command", "quicklook_door"]
