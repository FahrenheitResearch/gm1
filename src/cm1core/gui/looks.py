"""More looks and films on the Results screen.

``POST /api/runs/RUN/film {look, format?, fps?, between?}``
    ``gpuwm-cm1 quicklook RUNDIR --film LOOK [--format F] [--fps N] [--between K] --json``, launched in the
    background at nice 19 with no GPU (the swirl film takes about a minute at 1 km); ``{look, stop: true}``
    stops a film this server started.  The film's progress is in the results payload.
``POST /api/runs/RUN/look {look}``
    ``gpuwm-cm1 quicklook RUNDIR --look environment|parcels --json``: a look drawn once per run, drawn now (a
    host-only call).  ``{look: "all"}`` launches ``gpuwm-cm1 quicklook RUNDIR`` in the background, which draws
    the looks a run's pictures do not have yet (a run drawn before these looks existed).

:func:`more_looks` is what both Results views read: every look after the four favourites, with its title and
"What am I looking at" words filled with the run's own numbers (``copy/looks.json``), whether the run has it and,
if not, why, in plain words; and each film's state.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import quote

from . import commands as C
from .files import read_json

if TYPE_CHECKING:
    from .api import Api, Reply

COPY = Path(__file__).resolve().parent / "copy" / "looks.json"
FILM_FORMATS = ("apng", "mp4", "gif")
ONCE_LOOKS = ("environment", "parcels")
#: The looks every run's pictures have had since the More looks row; a run drawn before lacks them (its "Draw the new looks").
NEW_LOOKS = ("clouds-from-above", "updraft-slice", "hail-track", "spin-track", "environment")
#: Measured costs on one core for "about N s": the More looks take
#: 0.87-0.92 s per saved frame at 0.58 M cells (1 km) and 0.97-1.28 s at 1.31 M cells (250 m) (a Linux
#: workstation and a Windows 11 machine with WSL2), so they grow at most with the square root of the cell count (0.07-0.47 fitted), not near-linearly
#: as the first estimate (0.9) assumed; the upper figures are used.
SECONDS_PER_FRAME_NEW = 0.9
CELLS_EXPONENT = 0.5
#: A per-frame look's film: 2.9 s for 25 pictures at 1 km, 2.1 s for 13 at 250 m.
SECONDS_PER_FILM_PICTURE = 0.12
#: The swirl: 70.3 s for 217 pictures at 1 km and 35.3 s for 109 at 250 m (Linux, one core; 56 s when first measured).
SECONDS_PER_SWIRL_PICTURE = 0.33
#: The parcels film: 22.7 s for 120 parcel times with 36,000 parcels (Linux).
SECONDS_PER_PARCEL_PICTURE = 0.2


def load_copy() -> dict[str, Any]:
    return json.loads(COPY.read_text(encoding="utf-8"))


#: The looks a film can be made of: every per-frame look (quicklook.products.PRODUCTS), then the swirl and the
#: parcels (quicklook.film.DERIVED).  Written out here so the server never imports numpy; a test holds the two
#: lists equal.
FILM_LOOKS = ("reflectivity", "spin-near-ground", "updraft-5km", "swath", "surface", "rain", "clouds-from-above",
              "updraft-slice", "hail-track", "spin-track", "wind-lowest", "pressure-lowest", "swirl", "parcels")
#: quicklook.film.DEFAULT_FPS, for the same reason.
FILM_FPS = {"default": 6, "swirl": 18, "parcels": 12}


def film_looks() -> tuple[str, ...]:
    return FILM_LOOKS


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
THUNDERSTORM_ONLY = ("spin-near-ground", "updraft-5km", "swath", "surface", "hail-track", "spin-track")


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
    rows = []
    for look in copy.get("more_looks") or []:
        lid = look["id"]
        product = products.get(lid) or {}
        count = len([f for f in product.get("frames") or [] if isinstance(f, dict) and f.get("file")])
        if hurricane and (lid in THUNDERSTORM_ONLY or (lid == "parcels" and count == 0)):
            # Made for one thunderstorm and not drawn for a hurricane: the row leaves them out rather than
            # offering buttons that only say so.  So is a hurricane's parcel look when it followed none: its
            # Customize has no parcel choice, so the usual "turn on Follow air parcels" would not be true.
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
        if row["kind"] == "film":
            film = films_now.get(lid) or {}
            row["film"] = film
            row["available"] = bool(film.get("frames"))
            if not row["available"]:
                if frames_seen < 2:
                    row["why_not"] = why.get("film_needs_frames")
                else:
                    row["why_not"], row["action"] = why.get("film_first"), "film"
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
            elif drawn_before or (lid in NEW_LOOKS and frames_seen and not live):
                row["why_not"], row["action"] = why.get("older_run"), ("look" if lid in ONCE_LOOKS else "all")
            else:
                row["why_not"] = why.get("not_yet")
        rows.append(row)
    return rows


def _has_parcel_track(rundir: Path) -> bool:
    """Whether the run wrote a parcel track (GM1's own, or cm1out's pdata file), without reading it."""

    return (any(rundir.glob("segments/*/frames/lagrangian-track/parcel-attributes.f32"))
            or (rundir / "cm1out" / "cm1out_pdata.nc").is_file())


def ui_words() -> dict[str, str]:
    return dict(load_copy().get("ui") or {})


def about_seconds(index: Mapping[str, Any] | None, look: str, between: int = 8) -> int:
    """A plain "about N s" for drawing the new looks or making a film, from the measured 1 km costs scaled by
    the run's cells (estimated)."""

    frames = len((index or {}).get("frames") or [])
    run = (index or {}).get("run") or {}
    cells = 0.58e6
    grid = str(run.get("grid") or "")
    found = re.match(r"(\d+) × (\d+) × (\d+)", grid)
    if found:
        cells = float(found.group(1)) * float(found.group(2)) * float(found.group(3))
    scale = max(1.0, cells / 0.576e6) ** CELLS_EXPONENT
    if look == "all":
        return int(round(max(5.0, frames * SECONDS_PER_FRAME_NEW * scale)))
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
DATA_FILMS = ("swirl", "parcels")


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
        handler = {"film": self.film, "look": self.look}.get(action)
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
                           "message": f"Making the film on the computer's processor (nice 19, no GPU); about {about} s.",
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
        if look == "all":
            args = C.quicklook_pass_args(rundir)
            line = C.display(args)
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
                                          "draw the looks the pictures do not have yet, on the CPU")
            index = read_json(rundir / "quicklook" / "index.json", default=None)
            about = about_seconds(index if isinstance(index, dict) else None, "all")
            return Reply(200, {"ok": True, "action": "look", "run": run_id, "command": line,
                               "pid": launched.process.pid, "about_s": about,
                               "message": f"Drawing the new looks on the computer's processor; about {about} s.",
                               "fix": ""})
        if look not in ONCE_LOOKS:
            raise ApiError(400, "look must be environment, parcels or all.",
                           "The other looks are drawn for every saved time by the quick-look worker.")
        args = C.look_args(rundir, str(look))
        return self.api._picture("look", run_id, rundir, args,
                                 "The environment is drawn." if look == "environment" else "The air parcels are drawn.")


__all__ = ["LooksApi", "about_seconds", "films", "load_copy", "more_looks", "ui_words"]
