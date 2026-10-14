"""More looks and films on the Results screen.

``POST /api/runs/RUN/film {look, format?, fps?, between?}``
    ``gpuwm-cm1 quicklook RUNDIR --film LOOK [--format F] [--fps N] [--between K] --json``, launched in the
    background at nice 19 with no GPU (the swirl film takes about a minute at 1 km); ``{look, stop: true}``
    stops a film this server started.  The film's progress is in the results payload.
``POST /api/runs/RUN/radar {site: [x, y] | "storm", tilt?}``
    ``gpuwm-cm1 quicklook RUNDIR --radar-site X Y|storm [--radar-tilt DEG] --json``, launched in the background at
    nice 19 with no GPU once the run has ended: the radar looks' settings (``quicklook/radar.json``), and both radar
    looks of every saved time drawn again.
``POST /api/runs/RUN/look {look}``
    ``gpuwm-cm1 quicklook RUNDIR --look environment|parcels|run-card|updraft-time-height|spin-time-height --json``:
    a look drawn once per run, drawn now (a host-only call).  ``{look: "all"}`` launches ``gpuwm-cm1 quicklook
    RUNDIR`` in the background, which draws the looks a run's pictures do not have yet (a run drawn before these
    looks existed); ``{look: "all", again: true}`` launches ``gpuwm-cm1 quicklook RUNDIR --force``, which draws
    every picture again with this version (after an update that changes how a look is drawn).  A run whose saved
    frames were deleted (by hand, or by Keep pictures only) is refused both, and every look drawn from the frames:
    it keeps its pictures, and More looks offers none of them (:func:`.files.frames_gone`).  ``{look:
    "storm3d"}`` launches ``gpuwm-cm1 quicklook RUNDIR --look storm3d --json`` in the background: the storm in 3-D
    at the latest saved time, which takes seconds to a few minutes on one core; its orbit film is the film of
    ``storm3d``.

A film is offered of the looks the run draws: a hurricane run's film menu leaves out the looks made for one
thunderstorm (and the parcels when it followed none), and every other run's the hurricane looks
(:func:`film_looks_for`); the endpoint refuses the others in plain words.

:func:`more_looks` is what both Results views read: every look after the four favourites, with its title and
"What am I looking at" words filled with the run's own numbers (``copy/looks.json``), whether the run has it and,
if not, why, in plain words; and each film's state.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import quote

from . import commands as C
from .files import _frame_there, cm1_output, drawn_times, frames_gone, frames_missing, read_json

if TYPE_CHECKING:
    from .api import Api, Reply

COPY = Path(__file__).resolve().parent / "copy" / "looks.json"
FILM_FORMATS = ("apng", "mp4", "gif")
ONCE_LOOKS = ("environment", "parcels", "run-card", "updraft-time-height", "spin-time-height")
#: The looks every run's pictures have had since the More looks row, and the radar looks, the clouds in visible and
#: infrared, the wind's energy by size, the run card, the storm over time and the storm from the side since they
#: came; a run drawn before lacks them (its "Draw the new looks").  A hurricane run drawn before its looks by radius
#: and height existed lacks those two, and is offered the same.
NEW_LOOKS = ("clouds-from-above", "updraft-slice", "hail-track", "spin-track", "environment", "swirl-by-radius",
             "warm-core", "radar-storm-relative", "radar-velocity", "infrared", "spectra", "run-card",
             "updraft-time-height", "spin-time-height", "side-view", "water-paths", "rotation-axis")
#: The new looks 0.1.5 already drew, which "draw every picture again" was measured with (SECONDS_PER_FRAME_ALL).
LOOKS_015 = ("clouds-from-above", "updraft-slice", "hail-track", "spin-track", "environment")
#: The radar looks (quicklook.radar), whose settings the Results screen can change.
RADAR_LOOKS = ("radar-storm-relative", "radar-velocity")
#: quicklook.radar's settings file and schema (kept here so the server does not load the picture code).
RADAR_FILE = "radar.json"
RADAR_SCHEMA = "gm1.quicklook-radar/v1"
#: Looks drawn once per run from every saved time: a run drawn before them gets "Draw the new looks" (a pass in the
#: background, which reads each saved time once), not "Draw it" (one call that waits for the picture).
PASS_LOOKS = ("updraft-time-height", "spin-time-height")
#: What "Draw it" says once a look drawn once per run is drawn.
ONCE_DRAWN = {"environment": "The environment is drawn.", "parcels": "The air parcels are drawn.",
              "run-card": "The run card is drawn.", "updraft-time-height": "The updraft over time is drawn.",
              "spin-time-height": "The spin over time is drawn."}
#: The flow-near-the-ground looks (quicklook.products.NEAR_GROUND), drawn on grids of 250 m and finer
#: (quicklook.nearground.FINEST_DX_M, kept here so the server does not load the picture code; a test holds them
#: equal).  A run drawn before they existed offers to draw them; a coarser run says why it has none.
NEAR_GROUND = ("storm-relative-flow", "streamwise-spin", "cold-pool-depth")
NEAR_GROUND_DX_M = 250.0
#: Their cost on one core per saved time, drawn alone (Draw the new looks): a part for the pictures, which are the
#: same size on every grid, and a part per million cells, mostly the flow look's 30 dBZ line on its square.
#: Measured on loaded Linux workstations: 9.0-14.2 s at 250 m (18.4 M cells), 4.8-24.3 s at 100 m (16.1 M cells),
#: 10.1-32.7 s at 100 m (39.3 M cells); drawn with the reflectivity look in the same pass, 4-7 s at 250 m.  A pass
#: that draws them alone costs this (the pass's own part included); beside other new looks it is added to theirs.
SECONDS_PER_FRAME_NEAR_GROUND = 4.0
SECONDS_PER_FRAME_NEAR_GROUND_PER_MCELL = 0.4
#: Measured costs on one core for "about N s" of Draw the new looks, which draws only the looks a run's pictures
#: lack.  Whole passes over the 1 km supercell's 25 saved frames (0.58 M cells) on one core of a Windows 11 machine
#: with WSL2 took 33.2 s for the first five new looks, 32.0 s for the side view, water paths and rotation axis, and
#: 47.2 s for all eight: a pass costs 0.72 s per frame whatever it draws (reading the frame, the fields its looks
#: share, the index), and each look adds its own cost below.  The looks grow at most with the square root of the
#: cell count (0.07-0.47 fitted for the first five, 0.21 for the last three at 250 m), so that is used.  The others,
#: on the same footing:
#: - the two radar looks: a pass drawing only them took 2.93 s a frame at 1 km (25 saved times, Morrison's
#:   reflectivity worked out for them), so 2.21 s beyond the pass, half each; their scan is the same on any grid, so
#:   they grow only slowly with the cells (RADAR_CELLS_EXPONENT);
#: - the clouds in visible and infrared and the wind's energy by size: 0.55 s a frame together beside four of the
#:   first five (0.61 of those four's time in the same pass), half each;
#: - a hurricane's swirling wind and warm core by radius and height: 1.10 s a frame together beside the reflectivity
#:   on the 8 km hurricane's 28 saved times (0.97 M cells), on a busier Linux machine whose pass of the first five
#:   took 1.8 times as long as the figures above, so 0.46 s at 1 km on this footing, half each;
#: - the storm over time reads each saved time once (0.03 s at 1 km) and draws its two pictures once per run; the run
#:   card and the environment are drawn once per run.
SECONDS_PER_FRAME_PASS = 0.72
SECONDS_PER_FRAME_LOOK = {"clouds-from-above": 0.2, "updraft-slice": 0.16, "hail-track": 0.12, "spin-track": 0.13,
                          "environment": 0.0, "side-view": 0.22, "water-paths": 0.1, "rotation-axis": 0.24,
                          "radar-storm-relative": 1.105, "radar-velocity": 1.105, "infrared": 0.275,
                          "spectra": 0.275, "swirl-by-radius": 0.23, "warm-core": 0.23, "run-card": 0.0,
                          "updraft-time-height": 0.03, "spin-time-height": 0.0}
#: A new look not measured yet counts as the dearest measured one (a test asks for every new look's own figure).
SECONDS_PER_FRAME_UNMEASURED = max(SECONDS_PER_FRAME_LOOK.values())
#: A run drawn before any of the new looks: all of them.
SECONDS_PER_FRAME_NEW = SECONDS_PER_FRAME_PASS + sum(SECONDS_PER_FRAME_LOOK.get(look, SECONDS_PER_FRAME_UNMEASURED)
                                                     for look in NEW_LOOKS)
#: Every look of one saved frame drawn again (``--force``): 6.9 s at 0.58 M cells (1 km) and 8.3 s at 0.76 M cells
#: (1 km on 53 stretched levels) with 0.1.5's looks, one core at nice 19 of a busy Linux workstation (measured);
#: scaled by cells as above, and each look added since (NEW_LOOKS but LOOKS_015) adds its own cost.
SECONDS_PER_FRAME_ALL = 7.0
CELLS_EXPONENT = 0.5
#: Both radar looks of one saved time, drawn again on one core of a Linux workstation (measured with
#: gpuwm-cm1 quicklook RUNDIR --radar-site storm): 2.93 s at 0.58 M cells (1 km, 25 saved times, Morrison's
#: reflectivity worked out for them; 2.47 s with WSM6's own) and about 4.5 s at 18.4 M cells (250 m).  The scan
#: costs the same on any grid, so the time grows only slowly with the cells.
SECONDS_PER_FRAME_RADAR = 2.93
RADAR_CELLS_EXPONENT = 0.13
#: A per-frame look's film: 2.9 s for 25 pictures at 1 km, 2.1 s for 13 at 250 m.
SECONDS_PER_FILM_PICTURE = 0.12
#: The swirl: 70.3 s for 217 pictures at 1 km and 35.3 s for 109 at 250 m (Linux, one core; 56 s when first measured).
SECONDS_PER_SWIRL_PICTURE = 0.33
#: The storm in 3-D (quicklook.storm3d, written out here so the server never imports numpy; a test holds them
#: equal): the box drawn around a thunderstorm and a hurricane (half its width, m), the most cells a scene holds,
#: and the one-core costs its estimate is made of.
STORM3D_HALF_M = {"thunderstorm": 24_000.0, "hurricane": 250_000.0}
STORM3D_VOXELS = 4.0e6
STORM3D_COSTS = {"plan_s_per_mcell": 0.6, "build_s_per_mcell": 2.0, "draw_s_per_mray": 10.0}
STORM3D_SIZES = {"still": (1920, 1080), "orbit": (1280, 720)}
STORM3D_PICTURES = 72
#: The parcels film: 22.7 s for 120 parcel times with 36,000 parcels (Linux).
SECONDS_PER_PARCEL_PICTURE = 0.2
#: What the server says when asked to draw for a run whose saved frames were deleted (files.frames_gone).
FRAMES_GONE_WHY = ("This run's saved frames were deleted, so no new picture can be drawn for it: the looks are drawn "
                   "from the saved frames.")


def _drawn(rundir: Path, index: Mapping[str, Any] | None) -> int:
    """How many saved times the run's pictures were drawn at (:func:`.files.drawn_times`)."""

    return len(drawn_times(rundir, index))


def again_refused(rundir: Path, index: Mapping[str, Any] | None) -> str | None:
    """Why "Draw every picture again" (``quicklook RUNDIR --force``) is not offered, or None: some of the saved
    times its pictures were drawn at have lost their saved frames (:func:`.files.frames_missing`), so it would
    delete pictures that cannot be drawn again (``quicklook.render.frames_lost_force``)."""

    from .api import TRIMMED_FILE

    if (rundir / TRIMMED_FILE).is_file() or frames_gone(rundir, index):
        return ("This run's saved frames were deleted, so its pictures cannot be drawn again; they are kept.")
    missing = frames_missing(rundir, index)
    if not missing:
        return None
    return (f"{missing} of this run's {_drawn(rundir, index)} drawn saved times have lost their saved frames, "
            "so their pictures cannot be drawn again: drawing every picture again would delete them. They are "
            "kept; Draw the new looks still draws what the pictures lack, from the frames that are left.")


def radar_refused(rundir: Path, index: Mapping[str, Any] | None) -> str | None:
    """Why the radar of every saved time is not drawn again, or None (``quicklook.render.radar_lost``)."""

    if frames_gone(rundir, index):
        return "This run's saved frames were deleted, so the radar cannot scan it again. The pictures it has are kept."
    missing = frames_missing(rundir, index)
    if not missing:
        return None
    return (f"{missing} of this run's {_drawn(rundir, index)} drawn saved times have lost their saved frames, "
            "so the radar cannot scan those times again, and their radar pictures would no longer match the "
            "others. The radar pictures it has are kept.")


def film_refused(rundir: Path, index: Mapping[str, Any] | None, look: str,
                 films_now: Mapping[str, Any] | None = None) -> str | None:
    """Why a film made from the run's data (the swirl, the parcels, the storm in 3-D) is not made again over the
    one the run has, or None (``quicklook.film.remake_refused``)."""

    if look not in DATA_FILMS:
        return None
    record = read_json(rundir / "quicklook" / "films" / "index.json", default=None)
    made = ((record.get("films") or {}) if isinstance(record, dict) else {}).get(look)
    if not isinstance(made, dict) and not (rundir / "quicklook" / "films" / look / "film.png").is_file():
        return None
    missing = frames_missing(rundir, index)
    if not missing:
        return None
    if look == "storm3d":
        frame = ((made or {}).get("facts") or {}).get("frame") if isinstance((made or {}).get("facts"), dict) else None
        if isinstance(frame, str) and frame and ".." not in Path(frame).parts \
                and _frame_there(rundir, {"frame": frame}):
            return None
        return ("The 3-D film this run has was made at a saved time whose frame was deleted, so it could not be made "
                "again, and a new one would replace it. The film it has is kept.")
    source = "the parcel track kept beside the saved frames" if look == "parcels" else "the saved frames"
    return (f"{missing} of this run's {_drawn(rundir, index)} drawn saved times have lost their saved frames, "
            f"and this film is made from {source}: made again, it would show less than the one it has. The film it "
            "has is kept.")


def load_copy() -> dict[str, Any]:
    return json.loads(COPY.read_text(encoding="utf-8"))


#: The looks a film can be made of: every per-frame look (quicklook.products.PRODUCTS), then the swirl and the
#: parcels (quicklook.film.DERIVED).  Written out here so the server never imports numpy; a test holds the two
#: lists equal.
FILM_LOOKS = ("reflectivity", "spin-near-ground", "updraft-5km", "swath", "surface", "rain", "clouds-from-above",
              "infrared", "spectra", "updraft-slice", "hail-track", "spin-track", "wind-lowest", "pressure-lowest",
              "swirl-by-radius", "warm-core", "radar-storm-relative", "radar-velocity", "side-view", "water-paths",
              "rotation-axis", *NEAR_GROUND, "swirl", "parcels", "storm3d")
#: quicklook.film.DEFAULT_FPS, for the same reason.
FILM_FPS = {"default": 6, "swirl": 18, "parcels": 12, "storm3d": 12}
#: Looks drawn only when asked for (quicklook.render.ON_REQUEST_IDS): the storm in 3-D.
REQUEST_LOOKS = ("storm3d",)


def film_looks() -> tuple[str, ...]:
    return FILM_LOOKS


#: What a film of a look a run does not draw says (quicklook.products.THUNDERSTORM_ONLY_WHY for a hurricane).
FILM_NOT_FOR_HURRICANE = ("This look is made for thunderstorms, and a hurricane run does not draw it. For a "
                          "hurricane, film the wind or the pressure at the lowest level.")
FILM_HURRICANE_ONLY = "This look is drawn for hurricane runs only."
#: What a film of a hurricane run's parcels says when the run followed none.  A hurricane's Customize has no parcel
#: choice, so the thunderstorm runs' advice (turn on Follow air parcels) is left out, as More looks leaves the look out.
FILM_NO_HURRICANE_PARCELS = ("This hurricane run did not follow air parcels (CM1's iprcl = 0), so there is no film "
                             "of them.")


def film_looks_for(kind: str | None, parcels: bool = True) -> tuple[str, ...]:
    """The looks a film can be made of for one kind of run: a hurricane run leaves out the looks made for one
    thunderstorm (it draws none of them), and every other run the hurricane looks.  A hurricane run that
    followed no air parcels (``parcels`` false, :func:`follows_parcels`) leaves out the parcels film too."""

    if kind == "hurricane":
        return tuple(look for look in FILM_LOOKS
                     if look not in THUNDERSTORM_ONLY and (parcels or look != "parcels"))
    return tuple(look for look in FILM_LOOKS if look not in HURRICANE_ONLY)


def follows_parcels(rundir: Path, index: Mapping[str, Any] | None) -> bool:
    """Whether the run follows air parcels: its pictures have the parcels look, it wrote a parcel track, or its
    namelist sets CM1's iprcl = 1 (a run still going writes its track when it ends)."""

    for product in (index or {}).get("products") or []:
        if isinstance(product, Mapping) and product.get("id") == "parcels":
            if any(isinstance(f, Mapping) and f.get("file") for f in product.get("frames") or []):
                return True
    if _has_parcel_track(rundir):
        return True
    try:
        text = (rundir / "namelist.input").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    found = re.search(r"^\s*iprcl\s*=\s*([0-9]+)", text, re.M | re.I)
    return bool(found) and int(found.group(1)) == 1


def film_titles() -> dict[str, str]:
    """The title of every look after the four favourites (copy ``more_looks``), for a film menu whose run's rows
    leave one out (a hurricane run's parcels when it followed none)."""

    return {str(row["id"]): str(row["title"]) for row in load_copy().get("more_looks") or [] if row.get("title")}


def storm_kind_of(rundir: Path, index: Mapping[str, Any] | None) -> str | None:
    """``"hurricane"`` for a hurricane run (the pictures' own record, else the preset it was made from)."""

    from .guided import run_facts

    try:
        return run_facts(rundir, index).get("storm_kind")
    except (AttributeError, TypeError, ValueError):
        return None


def mp4_available() -> bool:
    """Whether the optional film extra (imageio-ffmpeg) is installed, without importing it."""

    import importlib.util

    return importlib.util.find_spec("imageio_ffmpeg") is not None


def _fill(text: str, values: Mapping[str, str | None], clauses: Mapping[str, str] | None = None) -> str:
    from .guided import _fill as fill

    return fill(text, values, clauses)


def _values(rundir: Path, index: Mapping[str, Any] | None) -> dict[str, str | None]:
    from .guided import distance_words, duration_words, run_facts

    try:
        facts = run_facts(rundir, index)
    except (AttributeError, TypeError, ValueError):
        # a hand-edited or torn run.json (valid JSON, not an object): the words leave their numbers out
        facts = {}
    dx, height, interval = facts.get("dx_m"), facts.get("lowest_level_m"), facts.get("interval_s")
    # A folder with frames and no run.json, deck or fit (a view-only cartesian folder): the pictures' own index
    # states the grid spacing it drew and the saved times it drew them at.
    shown = (index or {}).get("run") if isinstance((index or {}).get("run"), Mapping) else {}
    if dx is None and isinstance(shown.get("dx_m"), (int, float)):
        dx = float(shown["dx_m"])
    if interval is None:
        times = sorted({float(f["time_s"]) for f in (index or {}).get("frames") or []
                        if isinstance(f, Mapping) and isinstance(f.get("time_s"), (int, float))})
        steps = {round(b - a, 3) for a, b in zip(times, times[1:])}
        if len(steps) == 1:
            interval = steps.pop()
    return {"dx": None if dx is None else distance_words(dx),
            "six_dx": None if dx is None else distance_words(6.0 * dx),
            "height": None if height is None else f"{height:.0f}",
            "interval": None if interval is None else duration_words(interval)}


def domain_moves(rundir: Path) -> bool:
    """Whether the run's domain moves with the storm (CM1's imove = 1 with umove or vmove not zero), from its deck.

    The rain look's own facts say it for each frame (``domain_moves``); this is for pictures drawn before they
    did."""

    import tomllib

    try:
        deck = tomllib.loads((rundir / "deck.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return False
    table = deck.get("closures") or {}
    if not table.get("moving_frame"):
        return False
    try:
        return float(table.get("u_move_m_s") or 0.0) != 0.0 or float(table.get("v_move_m_s") or 0.0) != 0.0
    except (TypeError, ValueError):
        return False


def air_varies(rundir: Path) -> bool:
    """Whether the run starts from a 3D state that differs from place to place (a deck with a ``[chain]`` table,
    or the chain's ``chain/`` folder), as quicklook.environment.air_varies says for the picture's footer (kept
    apart so the server does not load the drawing modules)."""

    import tomllib

    if (rundir / "chain" / "chain-receipt.json").is_file():
        return True
    try:
        deck = tomllib.loads((rundir / "deck.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return False
    return isinstance(deck.get("chain"), dict)


def _base(run_id: str) -> str:
    return f"/api/runs/{quote(run_id, safe='')}/files/quicklook/"


def films(run_id: str, rundir: Path) -> dict[str, Any]:
    """Each film's state: made (with its files), being made (with its progress), stopped or failed."""

    base = _base(run_id)
    out: dict[str, Any] = {}
    folder = rundir / "quicklook" / "films"
    index = read_json(folder / "index.json", default=None)
    made = index.get("films") if isinstance(index, dict) and isinstance(index.get("films"), dict) else {}
    looks = set(made) | {p.name for p in folder.iterdir() if p.is_dir()} if folder.is_dir() else set(made)
    for look in sorted(looks):
        row: dict[str, Any] = {"look": look}
        record = made.get(look) if isinstance(made.get(look), dict) else None
        progress = read_json(folder / look / "progress.json", default=None)
        if isinstance(progress, dict):
            row["progress"] = {k: progress.get(k) for k in ("state", "done", "total", "message", "updated_utc",
                                                            "seconds")}
        if record is not None:
            files = record.get("files") if isinstance(record.get("files"), dict) else {}
            row.update({
                "made_utc": record.get("made_utc"), "pictures": record.get("pictures"), "fps": record.get("fps"),
                "stopped": record.get("stopped"), "note": record.get("note"),
                "urls": {kind: base + quote(str(name)) for kind, name in files.items() if isinstance(name, str)},
                "frames": [{"url": base + quote(str(name)), "time_s": None}
                           for name in record.get("frames") or [] if isinstance(name, str)],
            })
            times = (record.get("facts") or {}).get("times_s") if isinstance(record.get("facts"), dict) else None
            if isinstance(times, list) and len(times) == len(row["frames"]):
                for frame, t in zip(row["frames"], times):
                    frame["time_s"] = t
        state = (row.get("progress") or {}).get("state")
        row["state"] = state if state in ("running", "failed", "stopped") else ("done" if record else "none")
        out[look] = row
    return out


#: The looks made for one thunderstorm that a hurricane run does not draw (quicklook.products.THUNDERSTORM_ONLY,
#: kept here so this module does not load the picture code; a test holds the two equal).
THUNDERSTORM_ONLY = ("spin-near-ground", "updraft-5km", "swath", "surface", "hail-track", "spin-track",
                     "radar-storm-relative") + NEAR_GROUND
#: The looks drawn for a hurricane run only (quicklook.products.HURRICANE_ONLY, held equal by a test): its two
#: favourites and the two More looks rows with ``hurricane_only``.
HURRICANE_ONLY = ("wind-lowest", "pressure-lowest", "swirl-by-radius", "warm-core")
#: The look over time made for one thunderstorm's rotating updraft (quicklook.timeheight.THUNDERSTORM_ONLY; a test
#: holds the two equal).
ONCE_THUNDERSTORM_ONLY = ("spin-time-height",)


def _coarse_for_near_ground(index: Mapping[str, Any] | None) -> bool:
    """Whether the run's grid, as its pictures' index states it, is coarser than the near-ground looks are drawn
    for (False when the index does not say)."""

    shown = (index or {}).get("run") if isinstance((index or {}).get("run"), Mapping) else {}
    dx = shown.get("dx_m")
    return isinstance(dx, (int, float)) and not isinstance(dx, bool) and float(dx) > NEAR_GROUND_DX_M + 1e-6


def more_looks(run_id: str, rundir: Path, index: Mapping[str, Any] | None, status: Mapping[str, Any],
               films_now: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every look after the four favourites, in the More looks row's order, as the Results screen shows it."""

    copy = load_copy()
    values = _values(rundir, index)
    why = {key: row.get("text", "") for key, row in (copy.get("why_not") or {}).items()}
    hints = {key: row.get("hint") for key, row in (copy.get("why_not") or {}).items() if row.get("hint")}
    products = {str(p.get("id")): p for p in (index or {}).get("products") or [] if isinstance(p, dict)}
    frames_seen = len((index or {}).get("frames") or [])
    live = status.get("state") in ("running", "waiting", "stale")
    ended = status.get("state") in ("finished", "stopped", "failed", "paused") or bool((index or {}).get("run_ended"))
    films_now = films_now if films_now is not None else films(run_id, rundir)
    moves = domain_moves(rundir)
    drawn_before = frames_seen > 0 and not any(look in products for look in NEW_LOOKS)
    shown = (index or {}).get("run") if isinstance((index or {}).get("run"), Mapping) else {}
    hurricane = shown.get("storm_kind") == "hurricane"
    cm1 = cm1_output(rundir) is not None
    # A run whose saved frames were deleted (by hand, or by Keep pictures only) keeps its pictures: nothing is offered
    # that draws from the frames, and a look it lacks says why.  Its run card is drawn from its receipt, unless Keep
    # pictures only took the run's data (the server refuses any new picture of such a run).
    gone, trimmed = frames_gone(rundir, index), False
    if gone:
        from .api import TRIMMED_FILE

        trimmed = (rundir / TRIMMED_FILE).is_file()
    rows = []
    for look in copy.get("more_looks") or []:
        lid = look["id"]
        if cm1 and lid == "parcels":
            # CM1's own output: its parcel file is not read (quicklook.ncread.PARCELS_NOT_READ), so no parcels row
            continue
        product = products.get(lid) or {}
        count = len([f for f in product.get("frames") or [] if isinstance(f, dict) and f.get("file")])
        if hurricane and (lid in THUNDERSTORM_ONLY or lid in ONCE_THUNDERSTORM_ONLY
                          or (lid == "parcels" and count == 0)):
            # Made for one thunderstorm and not drawn for a hurricane: the row leaves them out rather than
            # offering buttons that only say so.  So is a hurricane's parcel look when it followed none: its
            # Customize has no parcel choice, so the usual "turn on Follow air parcels" would not be true.
            continue
        if look.get("hurricane_only") and not hurricane:
            # A hurricane's own look (by radius and height around its centre): no other run draws it.
            continue
        # A hurricane run's own words for a look every run has.
        words = look.get("text_hurricane") if hurricane and look.get("text_hurricane") else look.get("text", "")
        text = _fill(words, values, look.get("clause_if"))
        notes = [_fill(look.get("note", ""), values)]
        last = (product.get("frames") or [{}])[-1] if product.get("frames") else {}
        if look.get("note_sampled") and (last.get("facts") or {}).get("source") == "frames":
            # a track kept from the saved frames only (the run saved no every-step maximum for it)
            notes.append(_fill(look["note_sampled"], values))
        varies = (last.get("facts") or {}).get("air_varies")
        if look.get("note_varies") and (varies if isinstance(varies, bool) else air_varies(rundir)):
            # the chain's 3D start: no one column is the storm's environment (CM1's base state is one column)
            notes[0] = _fill(look["note_varies"], values)
        moving = (last.get("facts") or {}).get("domain_moves")
        if look.get("note_moving") and (moving if isinstance(moving, bool) else moves):
            notes.append(_fill(look["note_moving"], values))
        row: dict[str, Any] = {"id": lid, "title": look["title"], "short": look.get("short", ""), "text": text,
                               "note": " ".join(n for n in notes if n), "kind": look.get("kind", "frames"),
                               # 0.1.0 is the first release: no look is newer than another, so none is marked new.
                               "is_new": False, "count": count, "available": count > 0,
                               "why_not": None, "hint": None, "action": None}
        not_drawn = [n.get("reason") for n in product.get("not_drawn") or [] if isinstance(n, dict)]
        if lid in DATA_FILMS:
            # a film made from the saved frames that would show less made again: kept, and said why
            row["film_why_not"] = film_refused(rundir, index, lid, films_now)
        if row["kind"] == "film":
            film = films_now.get(lid) or {}
            row["film"] = film
            row["available"] = bool(film.get("frames"))
            if not row["available"]:
                if frames_seen < 2:
                    row["why_not"] = why.get("film_needs_frames")
                else:
                    row["why_not"], row["action"] = why.get("film_first"), "film"
        elif row["kind"] == "request":
            # Drawn only when asked: a still of the latest saved time (Draw it, again for a newer time), and its
            # orbit film beside it whether or not a still is drawn yet.
            row["film_too"] = True
            row["film"] = films_now.get(lid) or {}
            # why the last still asked for was not drawn (quicklook.storm3d writes it), in its own plain words
            failed = read_json(rundir / "quicklook" / lid / "failed.json", default=None)
            failed = str(failed.get("message") or "") if isinstance(failed, dict) else ""
            if frames_seen == 0:
                row["why_not"] = None if count else why.get("not_yet")
            else:
                row["action"] = "look"
                if count == 0:
                    row["why_not"] = failed or why.get("draw_3d")
                elif failed:
                    said = f"The last picture asked for was not drawn: {failed}"
                    row["note"] = " ".join(n for n in (row["note"], said) if n)
        elif count == 0:
            if lid == "parcels":
                reason = " ".join(str(r) for r in not_drawn if r)
                if live:
                    row["why_not"] = why.get("parcels_later")
                elif reason and "did not follow" not in reason:
                    row["why_not"] = reason
                else:
                    row["why_not"], row["hint"] = why.get("no_parcels"), hints.get("no_parcels")
                    if ended and not reason and _has_parcel_track(rundir):
                        # a track on disk the pictures have not drawn yet (a run drawn before this look existed)
                        row["why_not"], row["hint"] = why.get("older_run"), None
                        row["action"] = "look"
            elif not_drawn:
                row["why_not"] = str(not_drawn[0])
                row["hint"] = {"rain": hints.get("no_rain"), "hail-track": hints.get("no_hail")}.get(lid)
                if lid == "streamwise-spin" and "levels" in row["why_not"]:
                    row["hint"] = hints.get("fine_levels")
            elif lid in NEAR_GROUND and _coarse_for_near_ground(index):
                # a run drawn before these looks existed, on a grid they are not drawn for: say so, offer nothing
                row["why_not"] = _fill(why.get("near_ground_grid", ""), values)
            elif drawn_before or (lid in NEW_LOOKS + NEAR_GROUND and frames_seen and not live):
                row["why_not"], row["action"] = why.get("older_run"), ("look" if lid in ONCE_LOOKS
                                                                       and lid not in PASS_LOOKS else "all")
            else:
                row["why_not"] = why.get("not_yet")
        if lid in NEAR_GROUND and count == 0 and _coarse_for_near_ground(index):
            row["note"] = ""          # the reason says which grids it is drawn for; the note would say it again
        if gone and row["action"] is not None and (lid != "run-card" or trimmed):
            row["action"] = None
            if not row["available"]:
                row["why_not"], row["hint"] = why.get("frames_gone"), None
            if row.get("film_too") and not (row.get("film") or {}).get("frames"):
                row["film_too"] = False          # the orbit film is drawn from the frames too
        rows.append(row)
    return rows


def radar_settings(rundir: Path) -> dict[str, Any]:
    """The run's radar settings (``quicklook/radar.json``, as quicklook.radar reads it): where the radar stands
    (None: beside the storm in each picture), the scan's tilt and the storm motion set, if any."""

    doc = read_json(rundir / "quicklook" / RADAR_FILE, default=None)
    out: dict[str, Any] = {"site_km": None, "tilt_deg": 0.5, "motion_m_s": None}
    if not isinstance(doc, dict) or doc.get("schema") != RADAR_SCHEMA:
        return out

    def pair(value: Any) -> list[float] | None:
        if (isinstance(value, list) and len(value) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
            return [float(v) for v in value]
        return None

    tilt = doc.get("tilt_deg")
    if isinstance(tilt, (int, float)) and not isinstance(tilt, bool) and 0.1 <= tilt <= 20.0:
        out["tilt_deg"] = float(tilt)
    out["site_km"] = pair(doc.get("site_km"))
    out["motion_m_s"] = pair(doc.get("motion_m_s"))
    return out


def radar_state(rundir: Path, index: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the Results screen's radar row shows: the settings, where the radar stands in the newest radar picture
    (the run's own x, y at the start, km), the command and its time."""

    now = None
    for product in (index or {}).get("products") or []:
        if isinstance(product, Mapping) and product.get("id") in RADAR_LOOKS:
            for frame in product.get("frames") or []:
                facts = (frame.get("facts") or {}) if isinstance(frame, Mapping) else {}
                site = facts.get("site_km") if isinstance(facts, Mapping) else None
                if isinstance(site, list) and len(site) == 2:
                    now = [round(float(site[0]), 1), round(float(site[1]), 1)]
    settings = radar_settings(rundir)
    site = settings["site_km"]
    shown = (index or {}).get("run") if isinstance((index or {}).get("run"), Mapping) else {}
    # A hurricane run has one radar look (ground-relative), and its radar is placed from the eye.
    return {"looks": list(RADAR_LOOKS), "settings": settings, "now_km": now,
            "hurricane": shown.get("storm_kind") == "hurricane",
            # some saved times lost their frames: the radar is not drawn again (quicklook.render.radar_lost)
            "why_not": radar_refused(rundir, index),
            "command": C.display(C.radar_args(rundir, site if site is not None else "storm", settings["tilt_deg"])),
            "about_s": about_seconds(index, "radar")}


def _has_parcel_track(rundir: Path) -> bool:
    """Whether the run wrote a parcel track (GM1's own, or cm1out's pdata file), without reading it."""

    return (any(rundir.glob("segments/*/frames/lagrangian-track/parcel-attributes.f32"))
            or (rundir / "cm1out" / "cm1out_pdata.nc").is_file())


def ui_words() -> dict[str, str]:
    return dict(load_copy().get("ui") or {})


def storm3d_seconds(index: Mapping[str, Any] | None, pictures: int = 1, size: tuple[int, int] = (1920, 1080)) -> float:
    """quicklook.storm3d.estimate_seconds from what the index says of the grid (every level counted, so it errs
    long): reading the run for its core, building the box's scene, and drawing ``pictures`` pictures."""

    run = (index or {}).get("run") or {}
    found = re.match(r"(\d+) × (\d+) × (\d+), dx ([\d.]+) (k?m)", str(run.get("grid") or ""))
    nx, ny, nz, dx = (120, 120, 40, 1000.0) if not found else (
        int(found.group(1)), int(found.group(2)), int(found.group(3)),
        float(found.group(4)) * (1000.0 if found.group(5) == "km" else 1.0))
    half = STORM3D_HALF_M["hurricane" if run.get("storm_kind") == "hurricane" else "thunderstorm"]
    side_x, side_y = min(nx, 2.0 * half / dx), min(ny, 2.0 * half / dx)
    cells = side_x * side_y * nz
    stride = max(1, math.ceil(math.sqrt(cells / STORM3D_VOXELS)))
    voxels = cells / stride ** 2
    rays = size[0] * size[1] / 1e6
    costs = STORM3D_COSTS
    return (costs["plan_s_per_mcell"] * nx * ny * nz / 1e6 + costs["build_s_per_mcell"] * cells / 1e6
            + pictures * costs["draw_s_per_mray"] * rays * max((voxels / 1e6) ** (1.0 / 3.0), 0.3))


def about_words(seconds: float) -> str:
    """A plain duration for "About ...": seconds up to two minutes, then whole minutes."""

    s = int(round(seconds))
    return f"{s} s" if s <= 120 else f"{int(round(s / 60.0))} minutes"


def about_seconds(index: Mapping[str, Any] | None, look: str, between: int = 8) -> int:
    """A plain "about N s" for drawing the new looks or making a film, from the measured 1 km costs scaled by
    the run's cells (estimated)."""

    if look == "storm3d":
        return int(round(storm3d_seconds(index, STORM3D_PICTURES, STORM3D_SIZES["orbit"])))
    if look == "storm3d-still":
        return int(round(storm3d_seconds(index, 1, STORM3D_SIZES["still"])))
    frames = len((index or {}).get("frames") or [])
    run = (index or {}).get("run") or {}
    cells = 0.58e6
    grid = str(run.get("grid") or "")
    found = re.match(r"(\d+) × (\d+) × (\d+)", grid)
    if found:
        cells = float(found.group(1)) * float(found.group(2)) * float(found.group(3))
    scale = max(1.0, cells / 0.576e6) ** CELLS_EXPONENT
    radar = SECONDS_PER_FRAME_RADAR * max(1.0, cells / 0.576e6) ** RADAR_CELLS_EXPONENT
    if look in ("all", "again"):
        radar_scale = max(1.0, cells / 0.576e6) ** RADAR_CELLS_EXPONENT
        # A look a run of its kind never draws is no part of a pass over it.
        hurricane = str(run.get("storm_kind") or "") == "hurricane"
        skip = set(THUNDERSTORM_ONLY) | set(ONCE_THUNDERSTORM_ONLY) if hurricane else set(HURRICANE_ONLY)
        kinds = [name for name in NEW_LOOKS if name not in skip]
        # a thunderstorm on a grid of 250 m or finer also draws the flow near the ground, at its own measured cost
        spacing = re.search(r"dx (\d+(?:\.\d+)?) (m|km)\b", grid)
        near = (not hurricane and bool(found and spacing) and float(spacing.group(1))
                * (1000.0 if spacing.group(2) == "km" else 1.0) <= NEAR_GROUND_DX_M + 1e-6)
        near_ground = SECONDS_PER_FRAME_NEAR_GROUND + SECONDS_PER_FRAME_NEAR_GROUND_PER_MCELL * cells / 1.0e6

        def seconds(names: list[str]) -> float:
            # each look at its measured 1 km cost, grown with the cells (the radar looks only slowly)
            return sum(SECONDS_PER_FRAME_LOOK.get(name, SECONDS_PER_FRAME_UNMEASURED)
                       * (radar_scale if name in RADAR_LOOKS else scale) for name in names)

        if look == "again":
            # every picture drawn again: 0.1.5's looks, measured as a whole, and each look added since
            added = [name for name in kinds if name not in LOOKS_015]
            return int(round(max(5.0, frames * (SECONDS_PER_FRAME_ALL * scale + seconds(added)
                                                + (near_ground if near else 0.0)))))
        # the new looks the pictures do not have yet (all of them when the index says nothing)
        have = {p.get("id") for p in (index or {}).get("products") or [] if isinstance(p, Mapping)}
        missing = [name for name in kinds if name not in have]
        near_missing = near and any(name not in have for name in NEAR_GROUND)
        if not missing and not near_missing:
            missing, near_missing = kinds, near
        per_frame = (SECONDS_PER_FRAME_PASS * scale + seconds(missing)) if missing else 0.0
        return int(round(max(5.0, frames * (per_frame + (near_ground if near_missing else 0.0)))))
    if look == "radar":
        return int(round(max(5.0, frames * radar)))
    if look == "swirl":
        return int(round(max(5.0, (max(frames, 1) - 1) * (between + 1) * SECONDS_PER_SWIRL_PICTURE)))
    if look == "parcels":
        times = None
        for product in (index or {}).get("products") or []:
            if isinstance(product, Mapping) and product.get("id") == "parcels":
                for frame in product.get("frames") or []:
                    value = (frame.get("facts") or {}).get("times") if isinstance(frame, Mapping) else None
                    times = value if isinstance(value, int) else times
        # the parcel times the look found; before it is drawn, 120 (a 2 h run saving parcels every minute)
        return int(round(max(5.0, (times or 120) * SECONDS_PER_PARCEL_PICTURE)))
    return max(2, int(round(frames * SECONDS_PER_FILM_PICTURE)))


#: The films drawn from the run's data files, not from its pictures (quicklook.film.DERIVED).
DATA_FILMS = ("swirl", "parcels", "storm3d")


class LooksApi:
    """The film and look endpoints of one server (an :class:`.api.Api`)."""

    def __init__(self, api: "Api") -> None:
        self.api = api

    def get(self, parts: list[str], query: dict[str, list[str]]) -> "Reply | None":
        from .api import Reply

        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "films":
            run_id = parts[1]
            rundir = self.api.root.existing_run(run_id)
            return Reply(body={"films": films(run_id, rundir)})
        return None

    def post(self, run_id: str, rundir: Path, action: str, payload: dict[str, Any]) -> "Reply | None":
        handler = {"film": self.film, "look": self.look, "radar": self.radar}.get(action)
        return None if handler is None else handler(run_id, rundir, payload)

    # ---- films

    def _film_rows(self, run_id: str, look: str | None = None) -> list[Any]:
        rows = [row for row in self.api.jobs.launched(run_id) if row.role == "film"]
        if look is not None:
            rows = [row for row in rows if f" --film {look} " in f"{row.command} "]
        return rows

    def film(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, Reply

        look = payload.get("look")
        if not isinstance(look, str) or look not in film_looks():
            raise ApiError(400, "look must be one of the looks a film can be made of.",
                           f"Send look as one of: {', '.join(film_looks())}.")
        if payload.get("stop") is not True:
            index = read_json(rundir / "quicklook" / "index.json", default=None)
            index = index if isinstance(index, dict) else None
            kind = storm_kind_of(rundir, index)
            allowed = film_looks_for(kind, kind != "hurricane" or follows_parcels(rundir, index))
            if look not in allowed:
                # A look this kind of run never draws, or a hurricane's parcels when it followed none: say so,
                # rather than wait for pictures that never come.
                why = (FILM_HURRICANE_ONLY if kind != "hurricane" else
                       FILM_NO_HURRICANE_PARCELS if look == "parcels" else FILM_NOT_FOR_HURRICANE)
                raise ApiError(409, why, f"Film one of: {', '.join(allowed)}.",
                               command=C.display(C.film_args(rundir, look)))
        fmt = payload.get("format")
        if fmt not in (None, "") and fmt not in FILM_FORMATS:
            raise ApiError(400, "format must be apng, mp4 or gif.", "Leave it out for the usual files.")
        fps = payload.get("fps")
        if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0.5 <= fps <= 60):
            raise ApiError(400, "fps must be a number from 0.5 to 60.", "Leave it out for the look's usual speed.")
        between = payload.get("between")
        if between is not None and (isinstance(between, bool) or not isinstance(between, int)
                                    or not 0 <= between <= 30):
            raise ApiError(400, "between must be a whole number from 0 to 30.", "Leave it out for 8.")
        args = C.film_args(rundir, look, fmt or None, None if fps is None else float(fps), between)
        line = C.display(args)
        if payload.get("stop") is True:
            rows = self._film_rows(run_id, look)
            self.api.jobs.stop_if_dry(args)
            if not rows:
                raise ApiError(409, "No film of this look is being made by this GUI.",
                               "A film started from a terminal stops with Ctrl+C there.", command=line)
            for row in rows:
                self.api.jobs.end(row, grace_s=5.0)
            self.api.jobs.log_command(rundir, "film", line, "stopped")
            return Reply(200, {"ok": True, "action": "film", "run": run_id, "command": line, "stopped": True,
                               "message": "Stopped. The pictures made so far are kept.", "fix": ""})
        if look in DATA_FILMS and payload.get("stop") is not True:
            from .api import TRIMMED_FILE

            if (rundir / TRIMMED_FILE).is_file():
                # These two are drawn from the run's data, which "Keep pictures only" deleted.
                raise ApiError(409, "This run's data files were deleted to free disk space, and this film is "
                                    "drawn from them.", "A film of any look that has pictures can still be made.",
                               command=line)
            index = read_json(rundir / "quicklook" / "index.json", default=None)
            if frames_gone(rundir, index):
                raise ApiError(409, "This run's saved frames were deleted, and this film is drawn from them.",
                               "A film of any look that has pictures can still be made.", command=line)
            refused = film_refused(rundir, index if isinstance(index, dict) else None, look)
            if refused:
                raise ApiError(409, refused, "A film of any look that has pictures can still be made.", command=line)
        if look == "parcels":
            state = self.api.status(run_id, rundir).get("state")
            if state in ("running", "waiting", "stale"):
                raise ApiError(409, "Parcels arrive when the run ends.", "Make this film once the run has ended.",
                               command=line)
        with self.api.jobs.run_lock(run_id):
            if self._film_rows(run_id, look):
                raise ApiError(409, "A film of this look is being made now.", "Wait for it, or press Stop.",
                               command=line)
            self.api.jobs.require("quicklook")
            launched = self.api.jobs.launch(run_id, rundir, args, role="film")
            self.api.jobs.log_command(rundir, "film", C.sidecar_line(args).removesuffix(" &"),
                                      "a film, on the CPU")
        index = read_json(rundir / "quicklook" / "index.json", default=None)
        about = about_seconds(index if isinstance(index, dict) else None, look, between if between is not None else 8)
        return Reply(200, {"ok": True, "action": "film", "run": run_id, "command": line,
                           "pid": launched.process.pid, "about_s": about,
                           "message": "Making the film on the computer's processor (nice 19, no GPU); about "
                                      f"{about_words(about) if look in REQUEST_LOOKS else f'{about} s'}.",
                           "fix": ""})

    # ---- the radar looks' settings

    def radar(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, Reply, TRIMMED_FILE

        site = payload.get("site", "storm")
        if site != "storm":
            if (not isinstance(site, list) or len(site) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in site)
                    or not all(abs(float(v)) <= 5000.0 for v in site)):
                raise ApiError(400, "The radar site must be two numbers, x and y in km, each within 5000 km.",
                               'Or send site as "storm" for the radar beside the storm in each picture.')
            site = [float(site[0]), float(site[1])]
        tilt = payload.get("tilt", radar_settings(rundir)["tilt_deg"])
        if isinstance(tilt, bool) or not isinstance(tilt, (int, float)) or not 0.1 <= float(tilt) <= 20.0:
            raise ApiError(400, "The radar tilt must be a number from 0.1 to 20 degrees.",
                           "Radars scan storms from 0.5 degrees; leave it out to keep the tilt.")
        args = C.radar_args(rundir, site, float(tilt))
        line = C.display(args)
        if (rundir / TRIMMED_FILE).is_file():
            raise ApiError(409, "This run's data files were deleted to free disk space, so the radar cannot scan it "
                                "again.", "The pictures it has are kept.", command=line)
        if frames_gone(rundir, read_json(rundir / "quicklook" / "index.json", default=None)):
            raise ApiError(409, "This run's saved frames were deleted, so the radar cannot scan it again.",
                           "The pictures it has are kept.", command=line)
        index = read_json(rundir / "quicklook" / "index.json", default=None)
        refused = radar_refused(rundir, index if isinstance(index, dict) else None)
        if refused:
            raise ApiError(409, refused, "Nothing was changed.", command=line)
        state = self.api.status(run_id, rundir).get("state")
        if state in ("running", "waiting", "stale"):
            raise ApiError(409, "While the run goes, its quick-look worker draws every look as frames land.",
                           "Move the radar once the run has ended.", command=line)
        with self.api.jobs.run_lock(run_id):
            if [row for row in self.api.jobs.launched(run_id) if row.role in ("quicklook", "looks")]:
                raise ApiError(409, "The looks of this run are being drawn now.", "Wait for them.", command=line)
            self.api.jobs.require("quicklook")
            launched = self.api.jobs.launch(run_id, rundir, args, role="looks")
            self.api.jobs.log_command(rundir, "radar", C.sidecar_line(args).removesuffix(" &"),
                                      "the radar looks drawn again with a new radar, on the CPU")
        index = read_json(rundir / "quicklook" / "index.json", default=None)
        about = about_seconds(index if isinstance(index, dict) else None, "radar")
        return Reply(200, {"ok": True, "action": "radar", "run": run_id, "command": line,
                           "pid": launched.process.pid, "about_s": about,
                           "message": f"Drawing the radar looks again on the computer's processor; about {about} s.",
                           "fix": ""})

    # ---- looks drawn on request

    def look(self, run_id: str, rundir: Path, payload: dict[str, Any]) -> "Reply":
        from .api import ApiError, Reply

        from .api import TRIMMED_FILE

        look = payload.get("look")
        if (rundir / TRIMMED_FILE).is_file():
            # "Keep pictures only" deleted what a look is drawn from; a redraw would have nothing to read.
            raise ApiError(409, "This run's data files were deleted to free disk space, so no new picture can be "
                                "drawn from it.", "The pictures it has are kept: open Results to see them.")
        if look != "run-card" and frames_gone(rundir, read_json(rundir / "quicklook" / "index.json", default=None)):
            # Its saved frames were deleted by hand: a pass would find nothing to draw from (the run card is drawn
            # from the run's receipt).
            raise ApiError(409, FRAMES_GONE_WHY, "The pictures it has are kept: open Results to see them.")
        if look == "all":
            # {look: "all", again: true}: every picture drawn again from nothing with this version (--force), as
            # after an update that changes how a look is drawn.
            again = payload.get("again") is True
            args = C.quicklook_pass_args(rundir, again=again)
            line = C.display(args)
            if again:
                index = read_json(rundir / "quicklook" / "index.json", default=None)
                refused = again_refused(rundir, index if isinstance(index, dict) else None)
                if refused:
                    # some drawn times lost their frames: --force would delete pictures that cannot be drawn again
                    raise ApiError(409, refused, "Nothing was deleted.", command=line)
            state = self.api.status(run_id, rundir).get("state")
            if state in ("running", "waiting", "stale"):
                raise ApiError(409, "While the run goes, its quick-look worker draws every look as frames land.",
                               "Wait for the run to end.", command=line)
            with self.api.jobs.run_lock(run_id):
                if [row for row in self.api.jobs.launched(run_id) if row.role in ("quicklook", "looks")]:
                    raise ApiError(409, "The looks of this run are being drawn now.", "Wait for them.", command=line)
                self.api.jobs.require("quicklook")
                launched = self.api.jobs.launch(run_id, rundir, args, role="looks")
                self.api.jobs.log_command(rundir, "look", C.sidecar_line(args).removesuffix(" &"),
                                          "draw every picture again, on the CPU" if again
                                          else "draw the looks the pictures do not have yet, on the CPU")
            index = read_json(rundir / "quicklook" / "index.json", default=None)
            about = about_seconds(index if isinstance(index, dict) else None, "again" if again else "all")
            what = "every picture again" if again else "the new looks"
            return Reply(200, {"ok": True, "action": "look", "run": run_id, "command": line,
                               "pid": launched.process.pid, "about_s": about,
                               "message": f"Drawing {what} on the computer's processor; about {about} s.",
                               "fix": ""})
        if look in REQUEST_LOOKS:
            return self.storm3d(run_id, rundir)
        if look not in ONCE_LOOKS:
            raise ApiError(400, f"look must be {', '.join(ONCE_LOOKS + REQUEST_LOOKS)} or all.",
                           "The other looks are drawn for every saved time by the quick-look worker.")
        args = C.look_args(rundir, str(look))
        return self.api._picture("look", run_id, rundir, args, ONCE_DRAWN[str(look)])

    def storm3d(self, run_id: str, rundir: Path) -> "Reply":
        """The storm in 3-D at the latest saved time, drawn in the background on the CPU (seconds to minutes)."""

        from .api import ApiError, Reply

        args = C.look_args(rundir, "storm3d")
        line = C.display(args)
        index = read_json(rundir / "quicklook" / "index.json", default=None)
        if not (isinstance(index, dict) and index.get("frames")):
            raise ApiError(409, "This run has no saved times to draw yet.", "Wait for its first saved time.",
                           command=line)
        with self.api.jobs.run_lock(run_id):
            if [row for row in self.api.jobs.launched(run_id) if row.role == "storm3d"]:
                raise ApiError(409, "The storm in 3-D is being drawn now.", "Wait for it.", command=line)
            self.api.jobs.require("quicklook")
            launched = self.api.jobs.launch(run_id, rundir, args, role="storm3d")
            self.api.jobs.log_command(rundir, "look", C.sidecar_line(args).removesuffix(" &"),
                                      "the storm in 3-D, on the CPU")
        about = about_seconds(index, "storm3d-still")
        return Reply(200, {"ok": True, "action": "look", "run": run_id, "command": line, "pid": launched.process.pid,
                           "about_s": about,
                           "message": f"Drawing the storm in 3-D on the computer's processor (nice 19, no GPU); "
                                      f"about {about_words(about)}.", "fix": ""})


__all__ = ["HURRICANE_ONLY", "LooksApi", "RADAR_LOOKS", "about_seconds", "about_words", "again_refused",
           "film_looks_for", "film_refused", "films", "load_copy", "more_looks", "radar_refused", "radar_settings",
           "radar_state", "storm3d_seconds", "ui_words"]
