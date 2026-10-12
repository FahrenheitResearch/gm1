"""Guided mode's endpoints and words.

GET  /api/copy                    every guided string (gui/copy/guided.json) and the looks' panels (looks.json)
GET  /api/presets                 the gallery: ``gpuwm-cm1 preset list --json``, with picture URLs
GET  /api/presets/ID/picture      the card's picture (a PNG from a real run; the list says whose)
POST /api/fit                     {"preset", "budget", "gpu"?, "sounding"? | "environment"?, "custom"?}:
                                  ``gpuwm-cm1 preset fit ... --json``
POST /api/guided/start            {"preset", "budget", "name"?, "gpu"?, "sounding"? | "environment"?, "custom"?}:
                                  ``gpuwm-cm1 preset new ...`` then ``gpuwm-cm1 run RUNDIR`` (with its sidecars),
                                  as Start does

A preset that runs a sounding file (``isnd = 7``) may be given another one: ``"sounding"`` is a file
(``--sounding FILE``, for example the column the WRF card keeps from a WRF or ArWen run) and ``"environment"`` is
``{"instability", "shear", "hodograph"?}``, the Make-your-own-environment card's two sliders
(``--instability I --shear S --hodograph H``).  The fit's reply then carries ``fit.sounding``: the file's SHA-256,
where it came from and, for a made environment, getcape's numbers and the sentence the card shows.
``GET /api/presets/ID.SETTING/picture`` is one measured setting's picture of a card that has several.

``"custom"`` is the budget screen's "Customize this storm" panel: ``{KNOB: VALUE}`` for the knobs
``gpuwm-cm1 preset options ID`` lists (grid, levels, area, time, pictures, push, bubble, bumps, rotation,
parcels, ...), passed as ``--custom KNOB=VALUE``.  An unknown name is refused here (400); a value the preset does
not offer is refused by the ``preset`` command with its reason.  A customized Start names its run
``PRESET-custom-YYYYMMDD-HHMM``.

The fit and the new run directory are made by the ``preset`` command in a
host-only subprocess, as every action is; the server keeps
no preset logic of its own and never imports the estimator.  A guided run is
an ordinary run directory, so Watching and Results read it as the advanced
screens do; Results in guided mode adds the four looks' "What am I looking
at" panels, filled with the run's own numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import time
import tomllib
from typing import TYPE_CHECKING, Any, Mapping

from . import commands as C
from .files import NAME_RE, PathRefused, read_json, require_name

if TYPE_CHECKING:  # pragma: no cover
    from .api import Api

COPY_DIR = Path(__file__).resolve().parent / "copy"
PRESETS_ENV = "GPUWM_CM1_PRESETS_DIR"
#: "any" is "As long as it takes": no limit on the time.
BUDGETS = ("10m", "1h", "overnight", "any")
#: How long the gallery list and a fit are reused before they are asked for again.
LIST_TTL_S = 300.0
FIT_TTL_S = 20.0
#: "Run it longer" adds this much storm time.
RUN_LONGER_S = 3600.0
#: The largest file a sounding may be (a balloon with 7000 levels is about 400 kB).
SOUNDING_MAX_BYTES = 4 * 1024 * 1024
#: The oldest NVIDIA driver branch GM1's CUDA 13 runtime runs on (R580+).
MINIMUM_DRIVER = 580


def presets_dir() -> Path:
    explicit = os.environ.get(PRESETS_ENV)
    return Path(explicit) if explicit else Path(__file__).resolve().parents[1] / "presets"


def load_copy() -> dict[str, Any]:
    """Both copy files, as one JSON document."""

    return {
        "guided": json.loads((COPY_DIR / "guided.json").read_text(encoding="utf-8")),
        "looks": json.loads((COPY_DIR / "looks.json").read_text(encoding="utf-8")),
    }


# ------------------------------------------------------------------ numbers in words


def distance_words(metres: float) -> str:
    if metres >= 1000.0:
        km = metres / 1000.0
        return f"{km:.0f} km" if abs(km - round(km)) < 0.05 or km >= 10 else f"{km:.1f} km"
    return f"{metres:.0f} m"


def duration_words(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = seconds / 3600.0
    if abs(hours - round(hours)) < 0.05:
        return f"{round(hours):.0f} h"
    whole = int(hours)
    return f"{whole} h {round((hours - whole) * 60):.0f} min"


def gpu_blocker(gpu: Mapping[str, Any] | None, strings: Mapping[str, Any]) -> dict[str, str] | None:
    """Why no storm can start on this computer's GPU at all, or None.

    The fit prices a run even with no GPU (for an RTX 3080, so a login node can
    size a job), but guided Start runs it here and now: with no GPU, or a
    driver older than GM1's CUDA runtime needs, it would stop as soon as it
    began.  Guided mode offers nothing that fails, so such a card is greyed with
    this reason.
    """

    def say(key: str, **values: Any) -> str:
        text = str((strings.get(key) or {}).get("text") or key)
        for name, value in values.items():
            text = text.replace("{" + name + "}", str(value))
        return text

    if not isinstance(gpu, Mapping):
        return {"blocked": "no-gpu", "message": say("gallery.no_gpu"), "fix": say("gallery.no_gpu_fix")}
    driver = str(gpu.get("driver_version") or "")
    match = re.match(r"\s*(\d+)", driver)
    if match and int(match.group(1)) < MINIMUM_DRIVER:
        return {"blocked": "old-driver", "message": say("gallery.old_driver", driver=driver, minimum=MINIMUM_DRIVER),
                "fix": say("gallery.old_driver_fix")}
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def run_facts(rundir: Path, index: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The run's grid and times from its own files: run.json, deck.toml and the quick-look index."""

    run = read_json(rundir / "run.json", default={}) or {}
    cm1 = run.get("cm1") or {}
    deck: dict[str, Any] = {}
    try:
        deck = tomllib.loads((rundir / "deck.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        deck = {}
    grid = deck.get("grid") or {}
    table = deck.get("run") or {}
    # A guided run's fit.json says what the fit chose; run.json's grid is what a run folder made
    # another way (or the fake runner) records.
    fitted = (read_json(rundir / "fit.json", default={}) or {}).get("facts") or {}
    listed = run.get("grid") if isinstance(run.get("grid"), dict) else {}

    def first(*values: Any) -> float | None:
        for value in values:
            number = _number(value)
            if number is not None:
                return number
        return None

    dx = first(grid.get("dx_m"), cm1.get("dx"), fitted.get("dx_m"), listed.get("dx_m"))
    dz = first(grid.get("dz_m"), cm1.get("dz"), fitted.get("dz_m"), listed.get("dz_m"))
    nx = first(grid.get("nx"), cm1.get("nx"), fitted.get("nx"), listed.get("nx"))
    ny = first(grid.get("ny"), cm1.get("ny"), fitted.get("ny"), listed.get("ny"))
    duration = first(table.get("duration_s"), cm1.get("timax"), fitted.get("duration_s"))
    interval = first(table.get("history_interval_s"), cm1.get("tapfrq"), fitted.get("output_interval_s"))
    facts: dict[str, Any] = {"dx_m": dx, "dz_m": dz, "nx": nx, "ny": ny, "duration_s": duration,
                             "interval_s": interval, "lowest_level_m": None, "lowest_level_basis": None}
    # The quick-looks' index states the real height of the level they drew (its "run" block).
    index = index if isinstance(index, Mapping) else {}
    shown = next((block for block in (index.get("run"), index.get("facts"))
                  if isinstance(block, Mapping) and _number(block.get("lowest_level_m"))), None)
    if shown is not None:
        facts["lowest_level_m"] = float(shown["lowest_level_m"])
        facts["lowest_level_basis"] = "the quick-look's frame"
    elif dz is not None and not grid.get("stretch_z"):
        # A uniform column: CM1's lowest scalar level is half a level up.
        facts["lowest_level_m"] = dz / 2.0
        facts["lowest_level_basis"] = "half of the uniform level spacing dz"
    return facts


def _fill(template: str, values: Mapping[str, str | None], clauses: Mapping[str, str] | None = None) -> str:
    text = template
    for key, clause in (clauses or {}).items():
        if values.get(key) is None:
            text = text.replace(clause, "")
    for key, value in values.items():
        if value is not None:
            text = text.replace("{" + key + "}", value)
    # A sentence whose number is not known is left out rather than shown with a gap.
    sentences = re.split(r"(?<=\.)\s+", text)
    return " ".join(sentence for sentence in sentences if not re.search(r"\{[a-z_]+\}", sentence))


def panels(facts: Mapping[str, Any], looks_copy: Mapping[str, Any]) -> dict[str, Any]:
    """Each look's "What am I looking at" text, and the "This run" line, filled with the run's numbers."""

    dx = facts.get("dx_m")
    height = facts.get("lowest_level_m")
    values = {
        "dx": None if dx is None else distance_words(dx),
        "six_dx": None if dx is None else distance_words(6.0 * dx),
        "height": None if height is None else f"{height:.0f}",
    }
    out: dict[str, Any] = {}
    for look in looks_copy.get("looks", []):
        out[look["id"]] = {
            "id": look["id"],
            "title": look["title"],
            "short": look.get("short", ""),
            "text": _fill(look["text"], values, look.get("clause_if")),
            "unit": look.get("unit", ""),
        }
    this_run = None
    if all(facts.get(key) is not None for key in ("dx_m", "nx", "ny", "duration_s", "interval_s")):
        this_run = _fill(looks_copy.get("this_run", ""), {
            "dx": distance_words(facts["dx_m"]),
            "width": f"{facts['nx'] * facts['dx_m'] / 1000.0:g}",
            "depth": f"{facts['ny'] * facts['dx_m'] / 1000.0:g}",
            "duration": duration_words(facts["duration_s"]),
            "interval": duration_words(facts["interval_s"]),
        })
    return {"looks": out, "order": [look["id"] for look in looks_copy.get("looks", [])], "this_run": this_run}


#: What each look's own numbers are called in a quick-look frame's facts, and how they are said.
STRONGEST = {
    "reflectivity": (("dbz_max_1km",), "{value:.0f} dBZ"),
    "spin-near-ground": (("zeta_max", "zeta_min"), "{value:.3f} per second"),
    "updraft-5km": (("w_max",), "{value:.0f} metres per second upward"),
    "swath": (("wind_max_ground_relative",), "{value:.0f} m/s ground-relative"),
}
#: Looks whose strongest value is the largest of either sign (spin one way or the other).  For the others the
#: largest value is said as it is: a reflectivity of -35 dBZ is no echo at all, not 35 dBZ.
EITHER_SIGN = frozenset({"spin-near-ground"})


def strongest_words(look_id: str, frame_facts: Mapping[str, Any] | None) -> str | None:
    """The strongest value a picture shows, in words, from the quick-look's own facts for that frame."""

    if not isinstance(frame_facts, Mapping) or look_id not in STRONGEST:
        return None
    keys, template = STRONGEST[look_id]
    values = [abs(value) if look_id in EITHER_SIGN else value
              for value in (_number(frame_facts.get(key)) for key in keys) if value is not None]
    if not values:
        return None
    return template.format(value=max(values))


# ------------------------------------------------------------------ the endpoints


class Guided:
    """The guided endpoints of one server (an :class:`.api.Api`)."""

    def __init__(self, api: "Api") -> None:
        self.api = api
        self._lock = threading.Lock()
        self._list: tuple[float, dict[str, Any]] | None = None
        self._fits: dict[tuple[str, str, int | None], tuple[float, dict[str, Any]]] = {}
        #: Run names a guided Start is making now: a second Start with the same name waits its turn (409).
        self._starting: set[str] = set()

    # ---- reads

    def copy(self) -> dict[str, Any]:
        return load_copy()

    def presets(self, refresh: bool = False) -> dict[str, Any]:
        from .api import ApiError

        with self._lock:
            cached = self._list
        if cached is not None and not refresh and time.monotonic() - cached[0] < LIST_TTL_S:
            return cached[1]
        result = self.api.jobs.host(C.preset_list_args())
        if result.rc != 0 or not isinstance(result.json, dict):
            said = (result.stderr or "").strip()
            if "cm1-r21-namelist" in said or "namelist parser" in said or "GM1_CM1_R21_NAMELIST" in said:
                # Every card is read through the native namelist parser: without it no storm can be shown.
                from ..cm1import.namelist import BUILD_COMMAND, TOOL_ENV

                raise ApiError(500, "The storms cannot be shown: GM1's CM1 namelist reader (cm1-r21-namelist) "
                                    "is not installed or not found.",
                               "Reinstall gpuwm-cm1 from its wheel, which carries it. In a source checkout build it "
                               f"({BUILD_COMMAND}) or set {TOOL_ENV} to its path, then reload this page. System "
                               "lists what is missing.",
                               command=result.command, output=result.as_dict())
            last = said.splitlines()[-1] if said else ""
            raise ApiError(500, "The presets could not be read" + (f": {last}" if last else "."),
                           "Run gpuwm-cm1 preset list in a terminal; it names the problem.",
                           command=result.command, output=result.as_dict())
        body = dict(result.json)
        for row in body.get("presets", []):
            picture = row.get("picture") or {}
            if picture.get("available") and picture.get("preset"):
                picture["url"] = f"/api/presets/{picture['preset']}/picture"
        body["command"] = result.command
        with self._lock:
            self._list = (time.monotonic(), body)
        return body

    def picture(self, preset_id: str) -> Path:
        if not NAME_RE.match(preset_id or ""):
            raise PathRefused(f"{preset_id!r} is not a preset name.")
        # "ID.SETTING": one measured setting of a card with several (pictures/SETTING.png).
        folder, _, setting = preset_id.partition(".")
        if setting:
            path = presets_dir() / folder / "pictures" / f"{setting}.png"
            if not path.is_file():
                raise FileNotFoundError(f"The preset {folder!r} has no picture for {setting!r}.")
            return path
        path = presets_dir() / preset_id / "picture.png"
        if not path.is_file():
            raise FileNotFoundError(f"The preset {preset_id!r} has no picture.")
        return path

    # ---- actions

    @staticmethod
    def _given(payload: Mapping[str, Any]) -> list[str]:
        """The sounding flags a request gives its preset: ``--sounding FILE`` or the made environment's."""

        from .api import ApiError

        sounding, environment = payload.get("sounding"), payload.get("environment")
        if sounding not in (None, "") and environment not in (None, {}):
            raise ApiError(400, "Give a sounding file or an environment, not both.", "Send one of them.")
        if sounding not in (None, ""):
            if not isinstance(sounding, str):
                raise ApiError(400, "sounding must be a file path.", "Send the path of an input_sounding file.")
            path = Path(sounding).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise ApiError(400, f"There is no sounding file at {sounding}.",
                               "Make the column again, or give the full path of an input_sounding file.")
            if path.stat().st_size > SOUNDING_MAX_BYTES:
                raise ApiError(400, f"{sounding} is too large to be a sounding file.",
                               "Give the path of an input_sounding file.")
            return C.preset_given_args(sounding=path)
        if environment not in (None, {}):
            if not isinstance(environment, Mapping):
                raise ApiError(400, "environment must be an object.", 'Send {"instability": 0.5, "shear": 0.5}.')
            values = {}
            for key in ("instability", "shear"):
                value = environment.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                    raise ApiError(400, f"environment.{key} must be a number from 0 to 1.",
                                   "Move the slider, or send a number from 0 to 1.")
                values[key] = round(float(value), 3)
            hodograph = environment.get("hodograph") or "curved"
            if hodograph not in ("curved", "straight"):
                raise ApiError(400, "environment.hodograph must be curved or straight.", "Send curved or straight.")
            return C.preset_given_args(environment={**values, "hodograph": hodograph})
        return []

    @staticmethod
    def _custom(payload: Mapping[str, Any]) -> list[str]:
        """The Customize panel's choices as ``--custom KNOB=VALUE`` flags (none when nothing was changed)."""

        from .api import ApiError

        custom = payload.get("custom")
        if custom in (None, {}):
            return []
        if not isinstance(custom, Mapping):
            raise ApiError(400, "custom must be an object.", 'Send {"grid": 500} or leave it out.')
        for knob, value in custom.items():
            if (value is None or isinstance(value, (list, dict)) or not isinstance(value, (str, int, float, bool))
                    or (isinstance(value, str) and (len(value) > 40 or any(ch.isspace() for ch in value)))):
                raise ApiError(400, f"The setting {knob} must be one number or word.",
                               "Pick one of its choices, or press Reset to the preset.")
        try:
            return C.preset_custom_args(custom)
        except ValueError as error:
            raise ApiError(400, str(error), "Press Reset to the preset, then pick the settings again.") from error

    @staticmethod
    def _choice(payload: Mapping[str, Any]) -> tuple[str, str, int | None]:
        from .api import ApiError

        preset = payload.get("preset")
        if not isinstance(preset, str) or not NAME_RE.match(preset):
            raise ApiError(400, "No storm was picked.", "Pick a storm from the gallery.")
        budget = payload.get("budget")
        if budget not in BUDGETS:
            raise ApiError(400, f"There is no {budget!r} time.", "Pick about 10 minutes, about 1 hour, overnight or as long as it takes.")
        gpu = payload.get("gpu")
        if gpu is not None and (isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0):
            raise ApiError(400, "gpu must be a whole number.", "Use 0 for the first GPU.")
        return preset, str(budget), gpu

    def fit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from .api import ApiError

        preset, budget, gpu = self._choice(payload)
        custom = self._custom(payload)
        given = self._given(payload) + custom
        key = (preset, budget, gpu, tuple(given))
        args = C.preset_fit_args(preset, budget, self.api.root.path, gpu, given)
        self.api.jobs.stop_if_dry(args)
        with self._lock:
            cached = self._fits.get(key)
        if cached is not None and time.monotonic() - cached[0] < FIT_TTL_S and not payload.get("refresh"):
            return {**cached[1], "cached": True}
        result = self.api.jobs.host(args)
        if result.rc not in (0, 1) or not isinstance(result.json, dict):
            reason = _reason(result).rstrip()
            if reason and not reason.endswith((".", "?", "!")):
                reason += "."
            # A Customize choice the preset cannot take says which in its reason: the fix is that choice, not
            # another storm.
            fix = ("Change the setting it names, or press Reset to the preset." if custom
                   else "Pick another storm, or switch to Advanced to see the details.")
            # A value that is taken instead of the one refused ({"grid": 23}), for the screen's one-tap button.
            raise ApiError(400, f"The storm could not be fitted: {reason}", fix,
                           command=result.command, output=result.as_dict(), suggest=_suggest(result))
        record = result.json
        offered = record.get("status") == "fits"
        body = {
            "ok": True, "action": "fit", "run": None, "command": result.command, "rc": result.rc,
            "offered": offered, "fit": record,
            "message": record.get("summary") if offered else record.get("reason"),
            "fix": "" if offered else "Pick a longer time or another storm.",
        }
        blocker = gpu_blocker(record.get("gpu"), load_copy()["guided"]["strings"]) if offered else None
        if blocker is not None:
            offered = False
            body.update({"offered": False, **blocker})
        if offered:
            # What Start would run, so "Show command" can show it before it is pressed.
            # Named from what the fit says changed, as run.json's case and fit.json say it (a choice at the
            # preset's own value is not a change).
            name = unique_name(preset, self.api.root.path, custom=bool(record.get("customized")),
                               customized=record.get("customized"))
            rundir = self.api.root.run_path(name)
            body["start"] = {"name": name, "commands": [
                C.display(C.preset_new_args(preset, budget, self.api.root.path, name, gpu, given)),
                C.display(C.run_args(rundir, self.api.native)),
            ]}
        with self._lock:
            self._fits[key] = (time.monotonic(), body)
        return body

    def customized(self, payload: Mapping[str, Any]) -> bool:
        """Whether the fit of ``payload`` changes the preset (its ``customized`` list); a refused fit counts as not."""

        return bool(self.customized_list(payload))

    def customized_list(self, payload: Mapping[str, Any]) -> list[Any]:
        """What the fit of ``payload`` says the user changed (its ``customized`` list); none for a refused fit."""

        from .api import ApiError

        if self.api.jobs.dry_active():
            # A dry run shows the commands only: named from the choices sent.
            custom = payload.get("custom") or {}
            grid = custom.get("grid") if isinstance(custom, Mapping) else None
            named = [{"knob": "grid", "keys": {"dx": [None, grid]}}] if isinstance(grid, (int, float)) else []
            return named or ([{"knob": "custom"}] if custom else [])
        try:
            body = self.fit({key: payload.get(key) for key in ("preset", "budget", "gpu", "sounding", "environment",
                                                               "custom")})
        except ApiError:
            return []
        return list((body.get("fit") or {}).get("customized") or [])

    def start(self, payload: Mapping[str, Any]):
        from .api import ApiError

        preset, budget, gpu = self._choice(payload)
        custom = self._custom(payload)
        given = self._given(payload) + custom
        name = payload.get("name")
        root = self.api.root
        changed = self.customized_list(payload) if bool(custom) and name in (None, "") else []
        with self._lock:
            if name in (None, ""):
                name = unique_name(preset, root.path, taken=self._starting, custom=bool(changed), customized=changed)
            try:
                require_name(str(name), "run name")
            except PathRefused as error:
                raise ApiError(400, str(error), "Use letters, digits, '.', '_' or '-'.") from error
            name = str(name)
            rundir = root.run_path(name)
            if rundir.exists() or name in self._starting:
                raise ApiError(409, f"A run called {name!r} already exists.", "Pick another name.")
            # "Show command": the two lines Start runs, the run folder and then the run.
            self.api.jobs.stop_if_dry("\n".join((C.display(C.preset_new_args(preset, budget, root.path, name, gpu,
                                                                            given)),
                                                  C.display(C.run_args(rundir, self.api.native)))))
            # Held until this Start ends, so a second Start with the same name (a double click, a second tab)
            # cannot make the folder too, nor log its own failed command into this run's commands.log.
            self._starting.add(name)
        try:
            return self._start(preset, budget, gpu, name, rundir, given, payload)
        finally:
            with self._lock:
                self._starting.discard(name)

    def _start(self, preset: str, budget: str, gpu: int | None, name: str, rundir: Path, given: list[str],
               payload: Mapping[str, Any]):
        from .api import ApiError, Reply, run_url

        root = self.api.root
        args = C.preset_new_args(preset, budget, root.path, name, gpu, given)
        # Guided mode offers nothing that fails: a storm the fit does not offer here (too big for the GPU or the
        # time, not yet checked against CM1, or no GPU that can run it) is refused before any folder is made.
        fitted = self.fit({"preset": preset, "budget": budget, "gpu": gpu,
                           "sounding": payload.get("sounding"), "environment": payload.get("environment"),
                           "custom": payload.get("custom")})
        if not fitted.get("offered"):
            raise ApiError(409, str(fitted.get("message") or "This storm does not fit this computer now."),
                           str(fitted.get("fix") or "Pick a longer time or another storm."),
                           command=C.display(args), fit=fitted.get("fit"))
        result = self.api.jobs.host(args)
        created = rundir.is_dir() and (rundir / "run.json").is_file()
        if created:
            self.api.jobs.log_command(rundir, "preset-new", result.command,
                                      "the guided fit and its run folder")
        record = result.json if isinstance(result.json, dict) else {}
        if result.rc == 1:
            raise ApiError(409, str(record.get("message") or "This storm does not fit this computer now."),
                           "Pick a longer time or another storm.", command=result.command,
                           fit=record.get("fit"))
        if result.rc != 0 or not created:
            raise ApiError(400, f"The run folder could not be made: {_reason(result)}",
                           "Switch to Advanced to see the check against CM1.", command=result.command,
                           output=result.as_dict())
        with self._lock:
            self._list = None  # the Runs list and the gallery's GPU numbers change once a run starts
            self._fits.clear()
        # A first launch is guided only while no run exists and no mode is stored; once this run exists, a reload
        # or the next launch would open in Advanced.  Someone who starts a guided storm is using guided mode, so
        # it is stored now (never over a mode the user chose).
        if self.api.prefs.load().get("mode") not in ("guided", "advanced"):
            self.api.prefs.set_mode("guided")
        reply: Reply = self.api._launch_engine("start", str(name), rundir, C.run_args(rundir, self.api.native))
        body = dict(reply.body or {})
        run_line = body.get("command")
        body.update({
            "action": "guided-start",
            "command": "\n".join(line for line in (result.command, run_line) if line),
            "commands": [result.command, *(body.get("sidecars") or []), run_line],
            "preset_command": result.command,
            "run_command": run_line,
            "fit": record.get("fit"),
            "url": run_url(str(name)),
            "message": (body.get("message") or "") if body.get("ok") else body.get("message"),
        })
        return Reply(reply.status, body)

    # ---- Results in guided mode

    def results_extra(self, rundir: Path, index: Mapping[str, Any] | None, status: Mapping[str, Any],
                      looks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        for look in looks or []:
            for frame in look.get("frames") or []:
                words = strongest_words(str(look.get("id")), frame.get("facts"))
                if words:
                    frame["strongest"] = words
        facts = run_facts(rundir, index)
        copy = load_copy()["looks"]
        filled = panels(facts, copy)
        fit = read_json(rundir / "fit.json", default=None)
        run = read_json(rundir / "run.json", default={}) or {}
        extra: dict[str, Any] = {"facts": facts, **filled, "preset": run.get("preset"), "case": run.get("case")}
        if isinstance(fit, dict):
            extra["fit"] = {key: fit.get(key) for key in ("preset", "preset_name", "budget", "budget_words", "summary",
                                                          "sentences", "facts", "customized")}
        end = _number(status.get("model_time_s")) or facts.get("duration_s")
        if end:
            until = float(end) + RUN_LONGER_S
            about = None
            estimate = (fit or {}).get("estimate") if isinstance(fit, dict) else None
            if isinstance(estimate, dict) and _number(estimate.get("wall_s")) and facts.get("duration_s"):
                about = duration_words(float(estimate["wall_s"]) * RUN_LONGER_S / float(facts["duration_s"]))
            extra["run_longer"] = {"until_s": until,
                                   "command": C.display(C.resume_args(rundir, None, until, self.api.native)),
                                   "about": about}
        return extra


def _suggest(result: Any) -> dict[str, Any]:
    """The ``suggest`` of a refused ``preset`` command's JSON (a Customize value that is taken), or nothing."""

    try:
        detail = json.loads(result.stderr)
    except (TypeError, ValueError):
        return {}
    suggest = detail.get("suggest") if isinstance(detail, dict) else None
    return {str(knob): value for knob, value in suggest.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)} if isinstance(suggest, dict) else {}


def _reason(result: Any) -> str:
    try:
        detail = json.loads(result.stderr)
        if isinstance(detail, dict) and detail.get("reason"):
            return str(detail["reason"])
    except (TypeError, ValueError):
        pass
    text = (result.stderr or result.stdout or "").strip().splitlines()
    return text[-1] if text else f"exit {result.rc}"


def unique_name(preset_id: str, root: Path, taken: set[str] | frozenset[str] = frozenset(), *,
                custom: bool = False, customized: Any = None) -> str:
    """``PRESET-YYYYMMDD-HHMM`` (``PRESET-custom-...`` for a customized storm), ``-2``, ``-3`` when taken.

    ``customized`` is the fit's list of what the user changed: with a grid spacing of the user's own the name carries
    that spacing in place of the preset's (``supercell-150m-custom-...``), as ``preset new`` names it."""

    stamp = time.strftime("%Y%m%d-%H%M", time.localtime())
    if custom and customized:
        from ..cm1run.presets import run_id_for

        preset_id = run_id_for(preset_id, customized)
    base = (f"{preset_id}-custom-{stamp}" if custom else f"{preset_id}-{stamp}")[:60]
    name, number = base, 2
    while (root / name).exists() or name in taken:
        name, number = f"{base}-{number}", number + 1
    return name


__all__ = ["BUDGETS", "Guided", "MINIMUM_DRIVER", "duration_words", "distance_words", "gpu_blocker", "load_copy",
           "panels", "presets_dir", "run_facts", "unique_name"]
