"""Guided presets: CM1 namelists the GUI's gallery offers, fitted to this GPU.

A preset is a folder ``cm1core/presets/<id>/`` holding a CM1 r21.1
``namelist.input`` and a ``preset.json`` (``gpuwm-cm1.preset/v1``): its plain
name, the one line the gallery card shows, the budget it opens on, and how
the auto-fit (:mod:`.fit`) may make it smaller.  ``picture.png`` with
``picture.json`` beside it is the card's picture from a real run of the
preset; a preset without one names another preset's picture, and the card
says whose it is.

A preset whose namelist reads a sounding file (``isnd = 7``) carries it as
``input_sounding`` beside the namelist, with ``preset.json``'s
``"sounding": {"file", "sha256", "source", "rights"}`` saying where it came
from.  ``"group"`` puts the card in one of the gallery's groups, in
:data:`GROUPS` order: ``start`` (Start here), ``explore`` (Explore),
``other`` (Other storm types) and ``famous`` (Famous environments).

Commands (each is what a guided GUI action runs):

``gpuwm-cm1 preset list [--json]``
    every preset, whether GM1 offers it (its namelist imports with no
    refusal, nothing to acknowledge and nothing not yet checked against
    CM1; otherwise the report's first reason), its grid and its picture.
``gpuwm-cm1 preset show ID [--json]``
    one preset, with its namelist and what changed from CM1's file.
``gpuwm-cm1 preset fit ID --budget 10m|1h|overnight [--sounding FILE] [--custom KNOB=VALUE ...] [--gpu N] [--root DIR] [--json]``
    the preset fitted to this GPU's free memory, the RAM, the free disk
    under DIR and the budget; exit 0 fits, 1 not offered (with the reason).
    ``--sounding FILE`` runs a preset whose namelist has ``isnd = 7`` on
    FILE instead of its own sounding; ``fit.json`` records FILE's SHA-256
    and where it came from.  ``--custom KNOB=VALUE`` (repeatable) changes one
    of the settings guided mode's "Customize this storm" panel offers
    (:mod:`.customize`: grid, levels, area, time, pictures, push, bubble,
    bumps, rotation, parcels); a changed setting is pinned, so the fit never
    changes it, and ``fit.json``'s ``customized`` list says what changed.
    ``--set KEY=VALUE`` sets a CM1 namelist word directly.  Either way the
    result must pass the same guided check as the shipped preset.
``gpuwm-cm1 preset options ID [--json]``
    the settings ``--custom`` may change for one preset, their choices and
    the preset's own values (the Customize panel draws itself from this).
``gpuwm-cm1 preset new ID --budget B --root RUNS [--name NAME] [--sounding FILE] [--custom KNOB=VALUE ...] [--gpu N] [--json]``
    the fitted run directory RUNS/NAME: ``namelist.input`` (the fitted
    namelist), ``namelist.input.original`` (the preset's), ``fit.json``, and
    everything ``gpuwm-cm1 import`` writes.  Guided Start is this command
    followed by ``gpuwm-cm1 run RUNS/NAME``.  It refuses (exit 1) anything
    that does not import with no refusal, nothing to acknowledge and nothing
    not yet checked: the guided path offers only what GM1 runs as CM1 does.

Nothing here imports CuPy or opens a GPU; the GPU is read through
``nvidia-smi`` (:mod:`.gpuinfo`).
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - the fit and the estimator load only when a preset is fitted
    from .fit import FitResult, PresetSpec

PRESET_SCHEMA = "gpuwm-cm1.preset/v1"
LIST_SCHEMA = "gpuwm-cm1.preset-list/v1"
NEW_SCHEMA = "gpuwm-cm1.preset-new/v1"
PICTURE_SCHEMA = "gpuwm-cm1.preset-picture/v1"
#: A folder of presets to use instead of the installed ones (tests, or a site's own).
PRESETS_ENV = "GPUWM_CM1_PRESETS_DIR"
FIT_JSON = "fit.json"
PROGRAM = "gpuwm-cm1"

#: The budget buttons of the guided screen, in order, in seconds (:data:`.fit.BUDGETS`).
#: ``any`` is "as long as it takes": no limit on the time, for a run that needs more than overnight's 10 hours.
BUDGET_ORDER = ("10m", "1h", "overnight", "any")
BUDGETS = {"10m": 600.0, "1h": 3600.0, "overnight": 36000.0, "any": math.inf}
BUDGET_BUTTONS = {"10m": "About 10 minutes", "1h": "About 1 hour", "overnight": "Overnight (up to 10 hours)",
                  "any": "As long as it takes"}
#: Rain and hail schemes by CM1 ptype, in plain words.
SCHEMES = {5: "Morrison", 7: "WSM6", 3: "Thompson"}
#: The gallery's groups, in order: the first card of the first group is the quick look.  Famous
#: environments come last while every card in it is greyed, so what runs comes first.
GROUPS = ("start", "explore", "other", "famous")
#: The sounding file a preset (or a run folder) carries beside its namelist, as CM1 reads it.
SOUNDING_FILE = "input_sounding"
#: Fields in fit.json that record the moment the fit was made, not what it chose.
MOMENT_FIELDS = ("elapsed_s", "created_utc")


class PresetError(ValueError):
    """An unknown preset, a folder that cannot be written, or a preset that cannot run.

    ``suggest`` is a Customize value that is taken instead of the one refused (``{"grid": 23.0}``), when there is
    one: the error's JSON carries it, and the budget screen makes a button of it."""

    def __init__(self, message: str, *, suggest: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.suggest = dict(suggest or {})


# ------------------------------------------------------------------ the library


def presets_root() -> Path:
    explicit = os.environ.get(PRESETS_ENV)
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parents[1] / "presets"


@dataclass(frozen=True)
class Preset:
    id: str
    folder: Path
    record: Mapping[str, Any]
    namelist_text: str
    #: The sounding this preset runs on (isnd = 7): its own ``input_sounding``, or the file ``--sounding`` gave.
    sounding_path: Path | None = None
    #: Where a ``--sounding`` file came from, as fit.json records it (None: the preset's own).
    sounding_given: Mapping[str, Any] | None = None
    #: A preset changed for its sounding (the frame that follows the storm) is read from this file, and keeps the
    #: preset folder's own namelist as the run's original.
    namelist_file: Path | None = None
    namelist_original: str | None = None

    @property
    def name(self) -> str:
        return str(self.record.get("name") or self.id)

    @property
    def line(self) -> str:
        return str(self.record.get("line") or "")

    @property
    def order(self) -> int:
        return int(self.record.get("order", 99))

    @property
    def group(self) -> str:
        value = str(self.record.get("group") or "start")
        return value if value in GROUPS else "other"

    @property
    def sounding_text(self) -> str | None:
        if self.sounding_path is None:
            return None
        return self.sounding_path.read_text(encoding="utf-8")

    @property
    def sounding_sha256(self) -> str | None:
        if self.sounding_path is None:
            return None
        return hashlib.sha256(self.sounding_path.read_bytes()).hexdigest()

    @property
    def budget(self) -> str:
        value = str(self.record.get("budget") or "1h")
        return value if value in BUDGETS else "1h"

    @property
    def namelist_path(self) -> Path:
        return self.namelist_file or self.folder / "namelist.input"

    @property
    def namelist_original_text(self) -> str:
        return self.namelist_original if self.namelist_original is not None else self.namelist_text

    @property
    def namelist_sha256(self) -> str:
        return hashlib.sha256(self.namelist_text.encode("ascii")).hexdigest()

    def spec(self, *, offered: bool = True, reason: str | None = None) -> "PresetSpec":
        from .fit import PresetSpec

        record = dict(self.record)
        record["offered"] = offered
        record["not_offered_reason"] = reason
        return PresetSpec.from_mapping(record, self.namelist_text, sounding_text=self.sounding_text)


def _read(folder: Path) -> Preset:
    try:
        record = json.loads((folder / "preset.json").read_text(encoding="utf-8"))
        text = (folder / "namelist.input").read_text(encoding="ascii")
    except (OSError, UnicodeError, ValueError) as error:
        raise PresetError(f"the preset folder {folder} is not readable: {error}") from error
    if record.get("schema") != PRESET_SCHEMA or record.get("id") != folder.name:
        raise PresetError(f"{folder / 'preset.json'} is not a {PRESET_SCHEMA} record for {folder.name}")
    sounding = folder / SOUNDING_FILE
    return Preset(folder.name, folder, record, text, sounding if sounding.is_file() else None)


def with_sounding(preset: Preset, path: str | Path, *, source: str | None = None,
                  environment: Mapping[str, Any] | None = None) -> Preset:
    """``preset`` running on the sounding file ``path`` (``--sounding``): only a preset whose namelist has isnd = 7.

    The file is read as CM1 reads it when the preset is examined (a file CM1 would stop on, or whose top is below
    the model's, makes the preset not offered with that reason).  ``source`` is where the file came from, in
    words (fit.json keeps it); by default its path.  ``environment`` is the generator's record when the file was
    made by ``preset environment``.

    Another sounding moves the storm another way, so the view follows it: the moving frame (umove, vmove) and the
    warm bubble become the storm-track check's recommended pair for this sounding (``estimate.storm_track``, the
    same pair the Domain screen's "Use the recommended frame and bubble" sets), each a sentence in fit.json.
    """

    from ..cm1import.namelist import NamelistError

    path = Path(path).expanduser()
    if not path.is_file():
        raise PresetError(f"--sounding {path}: there is no such file")
    try:
        from ..cm1import import coverage

        effective = coverage.import_for_report(preset.namelist_path).effective
    except (NamelistError, OSError, ValueError) as error:  # pragma: no cover - the shipped namelists parse
        raise PresetError(f"the preset {preset.id} cannot be read: {error}") from error
    isnd = int(effective.get("isnd", 0))
    if isnd != 7:
        raise PresetError(f"the preset {preset.id} makes its own sounding (isnd = {isnd}), so it takes no --sounding; "
                          "pick a preset that runs a sounding file (isnd = 7)")
    from ..cm1import.sounding import check_sounding

    # Read as CM1 reads it first, so a file CM1 would stop on (not text, not numbers, a top below the model's)
    # is refused with that plain reason instead of a later, vaguer one (the storm-track check) or a decode error.
    problems = check_sounding(path, effective).problems
    if problems:
        raise PresetError(f"--sounding {path}: {problems[0].text}; {problems[0].fix}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    given: dict[str, Any] = {"file": str(path.resolve()), "sha256": digest, "source": source or str(path.resolve()),
                             "rights": None, "given": True}
    if environment is not None:
        given["environment"] = dict(environment)
    # replace(), so a preset already changed (Customize, or the made air on a Weisman-Klemp preset) keeps its own
    # namelist: the storm-track check below must read the namelist that runs, not the shipped one.
    derived = replace(preset, sounding_path=path.resolve(), sounding_given=given)
    return _follow_storm(derived, effective)


def _follow_storm(preset: Preset, effective: Mapping[str, Any]) -> Preset:
    """The preset with the frame and bubble the storm-track check recommends for its sounding (see with_sounding)."""

    found = _follow_change(preset, effective, " this sounding makes")
    if found is None:
        return preset
    text, record, change, folder = found
    # A frame Customize worked out for the preset's own air (a smaller area, a longer storm) no longer runs: this
    # sounding's frame replaces it, from the preset's own values, so the record never states a frame that does not run.
    earlier = [entry for entry in record.get("customized") or [] if entry.get("knob") == "follow"]
    if earlier:
        record["customized"] = [entry for entry in record.get("customized") or [] if entry.get("knob") != "follow"]
        first = earlier[0].get("keys") or {}
        change = {**change, "keys": {key: [first[key][0] if key in first else pair[0], pair[1]]
                                     for key, pair in change["keys"].items()}}
    given = dict(preset.sounding_given or {})
    given["follow"] = change
    return Preset(preset.id, preset.folder, record, text, preset.sounding_path, given,
                  namelist_file=folder / "namelist.input", namelist_original=preset.namelist_original_text)


def follow_storm_if_it_leaves(preset: Preset) -> Preset:
    """A customized preset whose storm would leave its area (a longer storm time, a smaller area): the frame and
    bubble the storm-track check recommends, as the Domain screen's "Use the recommended frame and bubble" sets them,
    recorded as one of the preset's ``customized`` changes.  A storm the check keeps inside keeps the preset's own.
    The check is a prediction: the fit's edge note stays on a small area whatever it predicts."""

    track = storm_track_for(preset)
    if not track or track.get("status") != "evaluated" or track.get("meets_edge_rule", True):
        return preset
    from ..cm1import import coverage

    effective = coverage.import_for_report(preset.namelist_path).effective
    found = _follow_change(preset, effective, "")
    if found is None:
        return preset
    text, record, change, folder = found
    record["customized"] = [*list(record.get("customized") or []), {"knob": "follow", **change}]
    return replace(preset, record=record, namelist_text=text, namelist_file=folder / "namelist.input")


def _follow_change(preset: Preset, effective: Mapping[str, Any],
                   cause: str) -> tuple[str, dict[str, Any], dict[str, Any], Path] | None:
    iinit = int(effective.get("iinit", 0))
    if int(effective.get("imove", 0)) != 1 or iinit not in (1, 12):
        return None
    track = storm_track_for(preset)
    pair = (track or {}).get("recommended_pair") or {}
    frame, bubble = pair.get("frame_m_s"), pair.get("bubble_km")
    if not frame or not bubble or pair.get("uses_it"):
        return None
    from .fit import set_namelist_values

    before = (float(effective.get("umove", 0.0)), float(effective.get("vmove", 0.0)))
    settings = dict(preset.record.get("gm1_settings") or {})
    # iinit = 12 starts the storm at the updraft nudging's centre
    # (init3d.F:1372-1376), iinit = 1 at the warm bubble's.
    kx, ky = (
        ("initialization.xc_wnudge", "initialization.yc_wnudge") if iinit == 12
        else ("initialization.bubble_center_x_m", "initialization.bubble_center_y_m")
    )
    was = (settings.get(kx), settings.get(ky))
    place = [float(bubble[0]) * 1000.0, float(bubble[1]) * 1000.0]
    held = False
    if iinit == 1:
        # The whole warm bubble has to lie inside the area (the run's own start refuses a truncated one), so in a
        # small area the recommended centre is brought in to where the bubble still fits.
        radius = float(settings.get("initialization.bhrad_m", 10000.0))
        centred = int(effective.get("iorigin", 1) or 1) == 2
        for axis, cells in enumerate((int(effective.get("nx", 0)), int(effective.get("ny", 0)))):
            length = cells * float(effective.get("dx" if axis == 0 else "dy", 0.0))
            low, high = (-length / 2.0, length / 2.0) if centred else (0.0, length)
            low, high = low + radius, high - radius
            if low > high:
                return None
            inside = min(high, max(low, place[axis]))
            if abs(inside - place[axis]) > 1e-6:
                # A whole 100 m, toward the middle, as the recommendation itself is given.
                toward = math.floor if inside > (low + high) / 2.0 else math.ceil
                place[axis], held = toward(inside / 100.0) * 100.0, True
    settings[kx], settings[ky] = place
    bubble = (place[0] / 1000.0, place[1] / 1000.0)
    text = set_namelist_values(preset.namelist_text, {"umove": float(frame[0]), "vmove": float(frame[1])})
    folder = _scratch("gm1-preset-")
    (folder / "namelist.input").write_text(text, encoding="ascii")
    record = dict(preset.record)
    record["gm1_settings"] = settings
    change = {
        "keys": {"umove": [before[0], float(frame[0])], "vmove": [before[1], float(frame[1])],
                 kx: [was[0], settings[kx]],
                 ky: [was[1], settings[ky]]},
        # What is done, and that it follows a prediction: not that the storm stays inside.  On an RTX 4090 a 67 m
        # storm left a 34 km area within 45 minutes with the view moving as the storm-track check advised.
        "sentence": (f"The view moves with the storm{cause} as the storm-track check predicts it "
                     f"({float(frame[0]):g} m/s east and {float(frame[1]):g} m/s north, ground-relative), and the "
                     + ("push of rising air" if iinit == 12 else "warm bubble")
                     + f" starts at {float(bubble[0]):g}, {float(bubble[1]):g} km from the centre"
                     + (", as far that way as the bubble fits in this area." if held else ".")),
        "why": "the storm-track check's recommended frame and bubble for this sounding",
    }
    return text, record, change, folder


def _scratch(prefix: str) -> Path:
    """A temporary folder for a derived namelist or a made sounding, removed when this process ends.

    ``preset fit`` and ``preset new`` copy what they need into the run folder, and the guided sliders ask for a
    fit on every move, so nothing is left behind in the temporary directory."""

    folder = Path(tempfile.mkdtemp(prefix=prefix))
    atexit.register(shutil.rmtree, folder, True)
    return folder


def storm_track_for(preset: Preset) -> dict[str, Any] | None:
    """The estimator's storm-track check for the preset as it is (its namelist, sounding and bubble), or None."""

    from .estimate import EstimateError, _model_config, load_deck, storm_track
    from ..errors import CM1Refusal

    try:
        from ..cm1import import coverage
        from .fit import _restart_at_end

        # The run keeps one restart at its end where the namelist writes none (rstfrq < 0), as the fit prices it:
        # read as an interval instead, a storm time that is not a whole number of them (45 or 90 min of a 60 min
        # one) was refused here, so no track came back and the view never followed the storm for those times.
        settings = {**_restart_at_end(coverage.import_for_report(preset.namelist_path).effective),
                    **dict(preset.record.get("gm1_settings") or {})}
        deck = load_deck(preset.namelist_path, sounding=preset.sounding_path, gm1_only=settings or None)
        return storm_track(_model_config(deck.raw), sounding_bytes=deck.sounding_bytes)
    except (CM1Refusal, EstimateError, OSError, ValueError, KeyError, TypeError):
        return None


def with_environment(preset: Preset, instability: float, shear: float, hodograph: str = "curved") -> Preset:
    """``preset`` on a sounding made by ``preset environment`` (written to a temporary file; fit.json records how)."""

    from ..cm1import import analytic_sounding
    from .customize import made_air

    try:
        made = analytic_sounding.generate(instability, shear, hodograph)
    except analytic_sounding.EnvironmentError as error:
        raise PresetError(str(error)) from error
    # A Weisman-Klemp preset (CM1's own built-in air) takes the made air as a sounding file: isnd = 7, iwnd = 0.
    preset = made_air(preset)
    folder = _scratch("gm1-environment-")
    path = analytic_sounding.write(made, folder / SOUNDING_FILE)
    derived = with_sounding(preset, path, source=command_line(made.command()),
                            environment={**made.record(), **environment_words(made)})
    given = dict(derived.sounding_given or {})
    given["file"] = None  # made here, not a file the user has: the command makes it again, byte for byte
    return replace(derived, sounding_given=given)


def environment_words(made: Any) -> dict[str, Any]:
    """getcape's numbers for a made environment on the 1 km preset grid, and the card's sentence from them.

    Without the initializer library (a host with no wheel natives) the sentence comes from the settings' targets
    and says so (``numbers_basis``: "target")."""

    from ..cm1import import analytic_sounding

    numbers: dict[str, Any] | None = None
    reason = None
    try:
        numbers = analytic_sounding.report(made.text)
    except Exception as error:  # noqa: BLE001 - no initializer library here: the targets stand in, labelled
        reason = f"getcape could not be run here ({error}); the sentence is from the settings' targets"
    return {"numbers": numbers, "numbers_reason": reason, **analytic_sounding.describe(made, numbers)}


def variants(preset: Preset) -> list[dict[str, Any]]:
    """The measured settings of a card with several (``pictures/NAME.png`` and ``.json``), for the nearest one."""

    rows = []
    for record_path in sorted((preset.folder / "pictures").glob("*.json")):
        if not record_path.with_suffix(".png").is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        environment = (record.get("run") or {}).get("environment") or {}
        rows.append({"id": record_path.stem, "picture": f"{preset.id}.{record_path.stem}",
                     "instability": environment.get("instability"), "shear": environment.get("shear"),
                     "hodograph": environment.get("hodograph"), "caption": record.get("caption") or ""})
    return rows


def sounding_record(preset: Preset) -> dict[str, Any] | None:
    """What the run's sounding is and where it came from (the card, fit.json), or None for isnd = 5 presets."""

    if preset.sounding_path is None:
        return None
    if preset.sounding_given is not None:
        return {key: value for key, value in preset.sounding_given.items() if key != "follow"}
    stored = dict(preset.record.get("sounding") or {})
    return {"file": SOUNDING_FILE, "sha256": preset.sounding_sha256,
            "source": stored.get("source"), "rights": stored.get("rights"), "given": False,
            **({"observation": stored["observation"]} if stored.get("observation") else {})}


def load_all(root: Path | None = None) -> list[Preset]:
    root = presets_root() if root is None else Path(root)
    if not root.is_dir():
        return []
    presets = [_read(folder) for folder in sorted(root.iterdir())
               if folder.is_dir() and (folder / "preset.json").is_file()]
    return sorted(presets, key=lambda preset: (GROUPS.index(preset.group), preset.order, preset.id))


def load(preset_id: str, root: Path | None = None) -> Preset:
    presets = {preset.id: preset for preset in load_all(root)}
    if preset_id not in presets:
        known = ", ".join(presets) or "none"
        raise PresetError(f"there is no preset {preset_id!r}; the presets are {known} (gpuwm-cm1 preset list)")
    return presets[preset_id]


# ------------------------------------------------------------------ what a preset is


_EXAMINED: dict[tuple[Any, ...], dict[str, Any]] = {}
#: The effective namelist each examined preset was read as (the Customize choices start from it).
_EFFECTIVE: dict[tuple[Any, ...], dict[str, Any]] = {}


def examine(preset: Preset) -> dict[str, Any]:
    """The preset's namelist read as ``gpuwm-cm1 import`` reads it: may guided mode offer it?

    Offered means the import exits 0 and its report's guided block has no
    reason against it (no refusal, nothing to acknowledge, nothing not yet
    checked against CM1, no microphysics that measurably runs differently).
    A ``preset.json`` may also withhold a preset with ``"offered": false`` and
    its own ``not_offered_reason``.
    """

    settings = preset.record.get("gm1_settings") or None
    seed = preset.record.get("random_seed")
    key = (str(preset.folder), preset.namelist_sha256, preset.sounding_sha256,
           json.dumps(settings, sort_keys=True, default=str), seed)
    if key in _EXAMINED:
        return _EXAMINED[key]
    from ..cm1import import coverage

    imported = coverage.import_for_report(preset.namelist_path)
    report = coverage.build_report(imported, environment=False, sounding=preset.sounding_path,
                                   overrides=settings, seed=seed)
    guided = report.guided()
    reasons = list(guided.get("reasons") or [])
    if report.exit_code != 0 and not reasons:
        reasons.append(report.as_dict().get("verdict_text") or "GM1 does not run this namelist as written.")
    if preset.record.get("offered") is False:
        reasons.insert(0, str(preset.record.get("not_offered_reason") or "Not offered yet."))
    offered = not reasons and report.exit_code == 0
    effective = imported.effective
    why_not = None
    if not offered:
        first = reasons[0].strip()
        prefix = "Not offered: " if " yet" in first.lower() else "Not offered yet: "
        why_not = first if first.lower().startswith("not offered") else prefix + _lower_first(first)
    result = {
        "offered": offered,
        "why_not": why_not,
        "reasons": reasons,
        "import_exit": report.exit_code,
        "verdict": report.as_dict().get("verdict"),
        "facts": facts(effective),
    }
    _EFFECTIVE[key[:3]] = dict(effective)
    _EXAMINED[key] = result
    return result


def _lower_first(text: str) -> str:
    text = text.strip()
    return text[:1].lower() + text[1:] if text[:2] != text[:2].upper() else text


def facts(effective: Mapping[str, Any]) -> dict[str, Any]:
    """The grid and the storm time in numbers and plain words (card and budget screens)."""

    nx, ny, nz = int(effective["nx"]), int(effective["ny"]), int(effective["nz"])
    dx, dz = float(effective["dx"]), float(effective["dz"])
    timax, tapfrq = float(effective["timax"]), float(effective["tapfrq"])
    ptype = int(effective.get("ptype", 0))
    periodic = [side for side in ("wbc", "ebc", "sbc", "nbc") if int(effective.get(side, 2)) == 1]
    return {
        "nx": nx, "ny": ny, "nz": nz, "dx_m": dx, "dz_m": dz,
        "cells": nx * ny * nz,
        "domain_km": [nx * dx / 1000.0, ny * dx / 1000.0],
        "duration_s": timax,
        "output_interval_s": tapfrq,
        "scheme": SCHEMES.get(ptype, f"ptype {ptype}"),
        "ptype": ptype,
        "repeating_sides": periodic,
        "words": grid_words(nx, ny, dx, timax),
    }


def spacing_words(dx_m: float) -> str:
    return f"{dx_m / 1000.0:g} km" if dx_m >= 1000.0 else f"{dx_m:g} m"


def hours_words(seconds: float) -> str:
    hours = seconds / 3600.0
    if abs(hours - round(hours)) < 1e-9:
        return f"{int(round(hours))} h"
    minutes = int(round(seconds / 60.0))
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"


def grid_words(nx: int, ny: int, dx_m: float, duration_s: float) -> str:
    width, depth = nx * dx_m / 1000.0, ny * dx_m / 1000.0
    area = f"{width:g} × {depth:g} km"
    return f"{spacing_words(dx_m)} grid · {area} · {hours_words(duration_s)} of storm time"


def picture(preset: Preset, presets: Mapping[str, Preset] | None = None) -> dict[str, Any]:
    """The card's picture: the preset's own run, or another preset's, labelled as that run."""

    own = preset.folder / "picture.png"
    if own.is_file():
        return {"available": True, "own": True, "preset": preset.id, "path": str(own),
                **_picture_record(preset.folder)}
    # Another preset's picture only where preset.json names one: the same storm at another grid spacing,
    # never a different kind of storm.
    source_id = preset.record.get("picture_from")
    if presets is None:
        presets = {item.id: item for item in load_all(preset.folder.parent)}
    for candidate in [source_id] if source_id else []:
        other = presets.get(candidate)
        if other is not None and (other.folder / "picture.png").is_file():
            record = _picture_record(other.folder)
            return {
                "available": True, "own": False, "preset": other.id, "path": str(other.folder / "picture.png"),
                **record,
                "caption": f"From {other.name} (this storm has no run of its own yet): "
                           + str(record.get("caption") or ""),
            }
    return {"available": False, "own": False, "preset": None, "path": None,
            "caption": "No picture yet: this storm has not been run on this build."}


def _picture_record(folder: Path) -> dict[str, Any]:
    try:
        record = json.loads((folder / "picture.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    return {"caption": record.get("caption") or "", "run": record.get("run") or {}}


def summary_record(preset: Preset, presets: Mapping[str, Preset] | None = None, *,
                   check: bool = True) -> dict[str, Any]:
    examined = examine(preset) if check else None
    shown_picture = picture(preset, presets)
    return {
        "id": preset.id,
        "name": preset.name,
        "line": preset.line,
        "order": preset.order,
        "group": preset.group,
        "note": preset.record.get("note") or "",
        "budget": preset.budget,
        "budget_words": BUDGET_BUTTONS[preset.budget],
        "offered": None if examined is None else examined["offered"],
        "why_not": None if examined is None else examined["why_not"],
        "facts": None if examined is None else examined["facts"],
        "picture": {key: value for key, value in shown_picture.items() if key != "path"},
        "source": preset.record.get("source"),
        "sounding": sounding_record(preset),
        "gm1_settings_why": preset.record.get("gm1_settings_why"),
        "card": preset.record.get("card"),
        "pictures": variants(preset),
        "customize": None if examined is None else _options(preset),
    }


def _options(preset: Preset) -> dict[str, Any] | None:
    """The Customize panel's choices for the preset (:func:`.customize.options`), or None if it cannot be read."""

    from .customize import CustomError, options

    effective = _EFFECTIVE.get((str(preset.folder), preset.namelist_sha256, preset.sounding_sha256))
    try:
        return options(preset, effective)
    except (CustomError, KeyError, TypeError, ValueError, OSError):  # pragma: no cover - the shipped presets read
        return None


def list_record(root: Path | None = None, *, check: bool = True) -> dict[str, Any]:
    presets = load_all(root)
    by_id = {preset.id: preset for preset in presets}
    return {
        "schema": LIST_SCHEMA,
        "groups": list(GROUPS),
        "presets": [summary_record(preset, by_id, check=check) for preset in presets],
        # "seconds" is null for the time with no limit (JSON has no infinity).
        "budgets": [{"id": key, "words": BUDGET_BUTTONS[key],
                     "seconds": BUDGETS[key] if math.isfinite(BUDGETS[key]) else None} for key in BUDGET_ORDER],
        "command": command_line(["preset", "list", "--json"]),
    }


# ------------------------------------------------------------------ fitting


def command_line(args: Sequence[str]) -> str:
    return shlex.join([PROGRAM, *[str(arg) for arg in args]])


def fit_args(preset_id: str, budget: str, gpu: int | None = None, root: Path | None = None,
             json_output: bool = True, given: Sequence[str] = ()) -> list[str]:
    """``given``: the sounding flags (``--sounding FILE`` or ``--instability I --shear S --hodograph H``)."""

    args = ["preset", "fit", preset_id, "--budget", budget, *given]
    if gpu is not None:
        args += ["--gpu", str(int(gpu))]
    if root is not None:
        args += ["--root", str(root)]
    return args + (["--json"] if json_output else [])


def new_args(preset_id: str, budget: str, root: Path, name: str, gpu: int | None = None,
             json_output: bool = True, given: Sequence[str] = ()) -> list[str]:
    args = ["preset", "new", preset_id, "--budget", budget, "--root", str(root), "--name", name, *given]
    if gpu is not None:
        args += ["--gpu", str(int(gpu))]
    return args + (["--json"] if json_output else [])


def machine(gpu_index: int | None, root: Path | None) -> tuple[Any, Any]:
    from .gpuinfo import query_gpus, query_host

    gpu = query_gpus().select(gpu_index)
    host = query_host(root if root is not None else Path.cwd())
    return gpu, host


def fit(preset: Preset, budget: str, *, gpu_index: int | None = None, root: Path | None = None,
        gpu: Any = None, host: Any = None) -> "FitResult":
    """Fit ``preset`` to the machine and ``budget``; not offered presets are not fitted."""

    from .fit import FitError, fit_preset

    if budget not in BUDGETS:
        raise PresetError(f"--budget must be one of {', '.join(BUDGET_ORDER)}, not {budget!r}")
    examined = examine(preset)
    if gpu is None and host is None:
        gpu, host = machine(gpu_index, root)
    elif host is None:
        from .gpuinfo import query_host

        host = query_host(root if root is not None else Path.cwd())
    spec = preset.spec(offered=examined["offered"], reason=examined["why_not"])
    try:
        return fit_preset(spec, gpu, budget, host=host)
    except FitError as error:
        raise PresetError(str(error)) from error


def fit_record(preset: Preset, result: "FitResult", *, command: str) -> dict[str, Any]:
    """``fit.json`` (``gpuwm-cm1.fit/v1``) with the preset's identity and the command that made it."""

    record = result.record()
    record.update({
        "preset_line": preset.line,
        "preset_namelist_sha256": preset.namelist_sha256,
        "preset_source": preset.record.get("source"),
        "budget_button": BUDGET_BUTTONS.get(result.budget, result.budget),
        "command": command,
        "sounding": sounding_record(preset),
    })
    # What the user changed (Customize), apart from what the fit changed to fit ("changes").
    record["customized"] = [dict(change) for change in preset.record.get("customized") or []]
    record["pinned"] = list(preset.record.get("pinned") or [])
    record["reason_kind"] = reason_kind(preset, result)
    if record["reason_kind"] == "drift":
        record["reason"] = _drift_reason(preset, str(record.get("reason") or ""))
    follow = (preset.sounding_given or {}).get("follow")
    if follow:
        record["changes"] = [dict(follow), *record.get("changes", [])]
        record["sentences"] = [follow["sentence"], *record.get("sentences", [])]
    if result.namelist_text is not None:
        record["namelist_sha256"] = hashlib.sha256(result.namelist_text.encode("ascii")).hexdigest()
        record["facts"] = _fitted_facts(preset, result)
    # The preset's own settings (the bubble) and the run settings the fit chose
    # (memory budget, closure tile), apart: an advanced re-import keeps the first.
    preset_settings = dict(preset.record.get("gm1_settings") or {})
    record["preset_settings"] = preset_settings
    record["run_settings"] = {key: value for key, value in result.gm1_settings.items() if key not in preset_settings}
    return record


#: Why a preset is not offered, by kind (fit.json ``reason_kind``; the budget screen's badge and its remedies).
REASON_KINDS = ("time", "gpu", "host", "disk", "edge", "limits", "unpriced", "drift", "withheld", "check")


def reason_kind(preset: Preset, result: "FitResult") -> str | None:
    """What holds a not-offered fit back: one of :data:`REASON_KINDS` (None when it is offered)."""

    if result.offered:
        return None
    kind = (result.needs or {}).get("kind")
    if kind:
        return str(kind)
    if preset.record.get("offered") is False:
        return "withheld"
    reason = str(result.reason or "").lower()
    if "side of the" in reason and "drift" in reason:
        return "drift"
    return "check"


def _drift_reason(preset: Preset, reason: str) -> str:
    """The storm-track check's refusal in guided words, with the cause when a coarser grid is it."""

    text = "The storm would drift too close to the side of the area before the run ends."
    grid = next((entry for entry in preset.record.get("customized") or [] if entry.get("knob") == "grid"), None)
    if grid is not None and preset.record.get("start_capped"):
        # A fine grid's area starts at the most one restart file holds: a larger area is not there to pick, so the
        # way out is a coarser grid (whose file holds a larger area) or a shorter storm.
        dx = float(grid["keys"]["dx"][1])
        try:
            nx, ny = (int(value[1]) for value in (grid["keys"]["nx"], grid["keys"]["ny"]))
            held = f" (about {max(nx, ny) * dx / 1000.0:.0f} km)"
        except (KeyError, IndexError, TypeError, ValueError):
            held = ""
        return (f"At a {spacing_words(dx)} grid the largest area one restart file holds{held} is too small for this "
                "storm: it would drift too close to the side before the run ends. Use a coarser grid, which can "
                "cover a larger area, or a shorter storm time.")
    if grid is not None:
        dx0, dx = (float(value) for value in grid["keys"]["dx"])
        if dx > dx0:
            text += (f" At a {spacing_words(dx)} grid this area has too few cells to keep the storm inside: make the "
                     f"area larger, or use a finer grid.")
            return text
    if preset.record.get("customized"):
        return text + " Make the area larger, or undo a change that moves the storm."
    return text + " Make the area larger."


def _fitted_facts(preset: Preset, result: "FitResult") -> dict[str, Any]:
    estimate = result.estimate
    if estimate is None:
        return {}
    grid = estimate.grid
    return {
        "nx": grid["nx"], "ny": grid["ny"], "nz": grid["nz"], "dx_m": grid["dx_m"], "dz_m": grid["dz_m"],
        "cells": estimate.cells, "domain_km": grid["domain_km"],
        "duration_s": estimate.duration_s, "output_interval_s": estimate.history_interval_s,
        "frames": estimate.frames,
        "words": grid_words(grid["nx"], grid["ny"], grid["dx_m"], estimate.duration_s),
        "dt_s": estimate.dt_s,
        "steps": estimate.steps,
        # The area simulated, said exactly: "67 m grid, 59.0 × 59.0 km, 880 × 880 cells".
        "exact": exact_words(grid["nx"], grid["ny"], grid["dx_m"]),
    }


def exact_words(nx: int, ny: int, dx_m: float) -> str:
    """The grid and the area it covers, exactly: "67 m grid, 59.0 × 59.0 km, 880 × 880 cells"."""

    width, depth = nx * dx_m / 1000.0, ny * dx_m / 1000.0
    return f"{spacing_words(dx_m)} grid, {width:.1f} × {depth:.1f} km, {nx} × {ny} cells"


# ------------------------------------------------------------------ a new run directory


Importer = Callable[..., tuple[int, Mapping[str, Any]]]


def _real_import(namelist: Path, rundir: Path, *, overrides: Mapping[str, Any], original: bytes,
                 command: str, sounding: Path | None = None, seed: int | None = None) -> tuple[int, Mapping[str, Any]]:
    from ..cm1import.door import run_import

    code, payload, _ = run_import(namelist, rundir, sounding=sounding, overrides=overrides or None,
                                  original=original, command=command, seed=seed)
    return code, payload


def run_id_for(preset_id: str, customized: Sequence[Mapping[str, Any]] | None) -> str:
    """The preset's id with the grid spacing the run really has: ``supercell-1km`` at a 150 m grid of the user's
    own is ``supercell-150m`` (the id's own spacing replaced; added at the end of an id that names none), so a run
    is never named after a spacing it does not have."""

    grid = next((entry for entry in customized or [] if entry.get("knob") == "grid"), None)
    try:
        dx = float(grid["keys"]["dx"][1]) if grid is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        dx = None
    if dx is None or not dx > 0.0:
        return preset_id
    token = f"{dx / 1000.0:g}km" if dx >= 1000.0 else f"{dx:g}m"
    renamed, count = re.subn(r"(?<![0-9A-Za-z.])\d+(?:\.\d+)?(?:km|m)(?![0-9A-Za-z])", token, preset_id, count=1)
    return renamed if count else f"{preset_id}-{token}"


def case_words(name: str, customized: Sequence[Mapping[str, Any]] | None) -> str:
    """What the Runs list calls a run: the preset's name, and for a grid or levels of the user's own the spacing the
    run really has ("Supercell, 150 m grid (customized from Supercell, quick look, 1 km)")."""

    customized = list(customized or [])
    if not customized:
        return name
    parts = []
    for knob, key, words in (("grid", "dx", "{} grid"), ("levels", "dz", "levels every {}")):
        entry = next((item for item in customized if item.get("knob") == knob), None)
        try:
            if entry is not None:
                parts.append(words.format(spacing_words(float(entry["keys"][key][1]))))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    if not parts:
        return f"{name} (customized)"
    return f"{name.split(',')[0]}, {', '.join(parts)} (customized from {name})"


def default_name(preset: Preset, root: Path) -> str:
    """``ID-YYYYMMDD-HHMM``, or ``ID-custom-YYYYMMDD-HHMM`` for a customized preset (the id with the run's own grid
    spacing, :func:`run_id_for`); ``-2``, ``-3`` when taken."""

    stamp = time.strftime("%Y%m%d-%H%M", time.localtime())
    customized = preset.record.get("customized")
    base = f"{run_id_for(preset.id, customized)}-custom-{stamp}" if customized else f"{preset.id}-{stamp}"
    name, number = base, 2
    while (root / name).exists():
        name, number = f"{base}-{number}", number + 1
    return name


def new_run(preset: Preset, budget: str, root: Path, name: str, *, gpu_index: int | None = None,
            command: str | None = None, importer: Importer | None = None,
            gpu: Any = None, host: Any = None) -> dict[str, Any]:
    """Fit ``preset`` and write the run directory ``root/name``: the guided Start's first half."""

    from ..gui.files import require_name

    try:
        require_name(name, "run name")
    except ValueError as error:
        raise PresetError(str(error)) from error
    root = Path(root).expanduser().resolve()
    rundir = root / name
    if rundir.exists():
        raise PresetError(f"{rundir} already exists; pick another --name")
    command = command or command_line(new_args(preset.id, budget, root, name, gpu_index, json_output=False,
                                               given=given_args(preset)))
    result = fit(preset, budget, gpu_index=gpu_index, root=root, gpu=gpu, host=host)
    record = fit_record(preset, result, command=command)
    if not result.offered:
        return {"schema": NEW_SCHEMA, "ok": False, "created": False, "run": None, "fit": record,
                "message": result.reason, "command": command}
    root.mkdir(parents=True, exist_ok=True)
    rundir.mkdir()
    staged = rundir / "namelist.input"
    staged.write_text(result.namelist_text, encoding="ascii")
    # A preset with no sounding file is imported exactly as before (an importer may not take a sounding).
    given: dict[str, Any] = {"sounding": preset.sounding_path} if preset.sounding_path is not None else {}
    if preset.record.get("random_seed") is not None:
        # "Try another draw": the random bumps from GM1's own random numbers at this seed (as import --seed).
        given["seed"] = int(preset.record["random_seed"])
    code, payload = (importer or _real_import)(
        staged, rundir, overrides=dict(result.gm1_settings),
        original=preset.namelist_original_text.encode("ascii"), command=command, **given)
    guided = payload.get("guided") or {}
    _write_json(rundir / FIT_JSON, record)
    _mark_run(rundir, preset, result)
    if code != 0 or guided.get("offered") is False:
        why = guided.get("why_not") or payload.get("verdict_text") or f"the import exited {code}"
        raise PresetError(
            f"the fitted namelist does not import as a run GM1 runs as CM1 does ({why}); "
            f"the report is in {rundir / 'import-report.txt'}")
    return {
        "schema": NEW_SCHEMA, "ok": True, "created": True, "run": str(rundir), "name": name,
        "preset": preset.id, "budget": budget, "fit": record,
        "import": {"exit": code, "verdict": payload.get("verdict")},
        "message": result.summary,
        "command": command,
        "next": command_line(["run", str(rundir)]),
    }


def given_args(preset: Preset) -> list[str]:
    """The flags that give ``preset`` its sounding again: ``--sounding FILE``, the environment's, or none."""

    from .customize import custom_args

    custom = custom_args(preset.record.get("custom") or {})
    custom += [arg for key, value in (preset.record.get("custom_sets") or {}).items()
               for arg in ("--set", f"{key}={_set_text(value)}")]
    given = preset.sounding_given
    if not given:
        return custom
    made = given.get("environment")
    if made:
        return ["--instability", f"{made['instability']:g}", "--shear", f"{made['shear']:g}",
                "--hodograph", str(made["hodograph"]), *custom]
    if given.get("file"):
        return ["--sounding", str(given["file"]), *custom]
    return custom


def _set_text(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _mark_run(rundir: Path, preset: Preset, result: "FitResult") -> None:
    """run.json names the preset and the budget, so the Runs list shows the case."""

    path = rundir / "run.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    record["preset"] = preset.id
    record["case"] = preset.name
    record["guided"] = {"preset": preset.id, "budget": result.budget, "fit": FIT_JSON}
    customized = preset.record.get("customized") or []
    if customized:
        record["case"] = case_words(preset.name, customized)
        record["guided"]["customized"] = [change["knob"] for change in customized]
    _write_json(path, record)


# ------------------------------------------------------------------ the command line


def add_preset_parser(commands: Any) -> None:
    command = commands.add_parser(
        "preset",
        help="the guided presets: list them, fit one to this GPU and a time budget, or make its run folder",
        description=(
            "Guided presets are CM1 r21.1 namelists GM1 runs as CM1 does. 'fit' sizes one to this GPU's free "
            "memory, the RAM, the free disk and a time budget; 'new' writes the fitted run folder, which "
            "gpuwm-cm1 run then runs."
        ),
    )
    actions = command.add_subparsers(dest="preset_action", required=True)
    listing = actions.add_parser("list", help="every preset and whether GM1 offers it")
    listing.add_argument("--json", action="store_true")
    show = actions.add_parser("show", help="one preset: its namelist and what changed from CM1's file")
    show.add_argument("preset")
    show.add_argument("--json", action="store_true")
    choices = actions.add_parser("options", help="the settings --custom may change for one preset, and their choices")
    choices.add_argument("preset")
    choices.add_argument("--json", action="store_true")
    for name, help_text in (("fit", "fit a preset to this GPU and a time budget"),
                            ("new", "fit a preset and write its run folder")):
        sub = actions.add_parser(name, help=help_text)
        sub.add_argument("preset")
        sub.add_argument("--budget", required=True, choices=BUDGET_ORDER,
                         help="about 10 minutes, about 1 hour, overnight (up to 10 hours), or any: as long as it "
                              "takes, with no limit on the time")
        sub.add_argument("--gpu", type=int, default=None, help="nvidia-smi's index of the GPU (default: the first)")
        sub.add_argument("--sounding", type=Path, default=None, metavar="FILE",
                         help="run the preset on this input_sounding instead of its own (presets with isnd = 7)")
        sub.add_argument("--instability", type=float, default=None, metavar="0..1",
                         help="with --shear: run the preset on the environment 'preset environment' makes")
        sub.add_argument("--shear", type=float, default=None, metavar="0..1")
        sub.add_argument("--hodograph", choices=("curved", "straight"), default="curved")
        sub.add_argument("--custom", action="append", default=[], metavar="KNOB=VALUE",
                         help="change one of the preset's settings as guided mode's Customize panel does (repeat for "
                              "more; 'gpuwm-cm1 preset options ID' lists them); the fit never changes it again. The "
                              "grid spacing and the level height take any value in metres (grid=67, levels=125)")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="set_values",
                         help="set a CM1 namelist word of the preset (repeatable); it must still pass the guided check")
        sub.add_argument("--json", action="store_true")
        if name == "fit":
            sub.add_argument("--root", type=Path, default=None,
                             help="the runs folder whose free disk to fit to (default: this folder)")
        else:
            sub.add_argument("--root", type=Path, required=True, help="the runs folder")
            sub.add_argument("--name", default=None,
                             help="the run folder's name (default: the preset and the time)")
    made = actions.add_parser(
        "environment",
        help="make your own environment: a sounding with a chosen instability and wind shear (CM1 isnd = 7 file)",
        description=("Weisman and Klemp's (1982) sounding, as CM1's isnd = 5 builds it, with the boundary-layer "
                     "moisture set for a CAPE of 300-3000 J/kg (--instability 0..1) and a 0-6 km wind shear of "
                     "2-40 m/s (--shear 0..1), written as an input_sounding. The numbers reported are CM1 getcape's "
                     "on the 1 km preset grid, and the sentence says which kind of storm such air usually makes."),
    )
    made.add_argument("--instability", type=float, required=True, metavar="0..1")
    made.add_argument("--shear", type=float, required=True, metavar="0..1")
    made.add_argument("--hodograph", choices=("curved", "straight"), default="curved",
                      help="curved (CM1's quarter circle, favours the right-moving storm) or straight (storms split)")
    made.add_argument("--out", type=Path, default=None, help="write the input_sounding here (never over a file)")
    made.add_argument("--json", action="store_true")
    command.set_defaults(handler=preset_door)


def preset_door(arguments: argparse.Namespace, *, importer: Importer | None = None) -> int:
    try:
        return _door(arguments, importer)
    except PresetError as error:
        if getattr(arguments, "json", False):
            detail: dict[str, Any] = {"status": "error", "code": "GM1-PRESET", "reason": str(error),
                                      "remedy": "see the reason"}
            if getattr(error, "suggest", None):
                detail["suggest"] = dict(error.suggest)
            print(json.dumps(detail, indent=2), file=sys.stderr)
        else:
            print(f"gpuwm-cm1 preset: {error}", file=sys.stderr)
        return 2


def _door(arguments: argparse.Namespace, importer: Importer | None = None) -> int:
    action = arguments.preset_action
    if action == "list":
        record = list_record()
        if arguments.json:
            print(json.dumps(record, indent=1, default=str))
        else:
            print(_list_text(record))
        return 0
    if action == "environment":
        record = environment_record(arguments.instability, arguments.shear, arguments.hodograph, out=arguments.out)
        print(json.dumps(record, indent=1, default=str) if arguments.json else _environment_text(record))
        return 0
    preset = load(arguments.preset)
    if action == "options":
        from .customize import options

        record = options(preset)
        print(json.dumps(record, indent=1, default=str) if arguments.json else _options_text(record))
        return 0
    preset = _given_sounding(preset, arguments)
    if action == "show":
        record = summary_record(preset)
        record["namelist"] = preset.namelist_text
        record["command"] = command_line(["preset", "show", preset.id, "--json"])
        if arguments.json:
            print(json.dumps(record, indent=1, default=str))
        else:
            print(_show_text(record))
        return 0
    if action == "fit":
        command = command_line(fit_args(preset.id, arguments.budget, arguments.gpu, arguments.root,
                                        json_output=False, given=given_args(preset)))
        result = fit(preset, arguments.budget, gpu_index=arguments.gpu, root=arguments.root)
        record = fit_record(preset, result, command=command)
        if arguments.json:
            print(json.dumps(record, indent=1, default=str))
        else:
            print(_fit_text(record))
        return 0 if result.offered else 1
    root = Path(arguments.root).expanduser()
    name = arguments.name or default_name(preset, root.resolve())
    command = command_line(new_args(preset.id, arguments.budget, root.resolve(), name, arguments.gpu,
                                    json_output=False, given=given_args(preset)))
    record = new_run(preset, arguments.budget, root, name, gpu_index=arguments.gpu, command=command,
                     importer=importer)
    if arguments.json:
        print(json.dumps(record, indent=1, default=str))
    else:
        print(_new_text(record))
    return 0 if record["ok"] else 1


def _given_sounding(preset: Preset, arguments: argparse.Namespace) -> Preset:
    """The preset with its --custom and --set choices first, then its --sounding or made environment."""

    from .customize import CustomError, apply, parse, parse_sets

    try:
        preset = apply(preset, parse(getattr(arguments, "custom", None) or ()),
                       parse_sets(getattr(arguments, "set_values", None) or ()))
    except CustomError as error:
        raise PresetError(str(error), suggest=error.suggest) from error
    sounding = getattr(arguments, "sounding", None)
    instability, shear = getattr(arguments, "instability", None), getattr(arguments, "shear", None)
    if sounding is not None and (instability is not None or shear is not None):
        raise PresetError("give --sounding or --instability with --shear, not both")
    if sounding is not None:
        # A column kept from a WRF or ArWen run says where it came from, not the kept file's path.
        from ..cm1import.sounding_door import column_words

        return with_sounding(preset, sounding, source=column_words(sounding))
    if instability is not None or shear is not None:
        if instability is None or shear is None:
            raise PresetError("--instability and --shear go together")
        return with_environment(preset, instability, shear, getattr(arguments, "hodograph", "curved"))
    return preset


def environment_record(instability: float, shear: float, hodograph: str = "curved", *,
                       out: Path | None = None) -> dict[str, Any]:
    """``gpuwm-cm1 preset environment``: the made sounding, getcape's numbers for it, and the sentence."""

    from ..cm1import import analytic_sounding

    try:
        made = analytic_sounding.generate(instability, shear, hodograph)
    except analytic_sounding.EnvironmentError as error:
        raise PresetError(str(error)) from error
    words = environment_words(made)
    written = None
    if out is not None:
        out = Path(out).expanduser()
        if out.exists():
            raise PresetError(f"{out} already exists; pick another --out")
        written = str(analytic_sounding.write(made, out).resolve())
    args = made.command() + (["--out", str(out)] if out is not None else []) + ["--json"]
    return {
        **made.record(),
        **words,
        "written": written,
        "text": made.text if out is None else None,
        "command": command_line(args),
        "use": command_line(["preset", "new", "your-environment", "--budget", "10m", "--root", "RUNS",
                             "--instability", f"{made.instability:g}", "--shear", f"{made.shear:g}",
                             "--hodograph", made.hodograph]),
    }


def _environment_text(record: Mapping[str, Any]) -> str:
    numbers = record.get("numbers") or {}
    lines = [record["sentence"]]
    if numbers:
        lines.append(f"  CAPE {numbers['cape_j_kg']:.0f} J/kg, 0-6 km shear {numbers['shear_0_6km_m_s']:.1f} m/s, "
                     f"0-3 km helicity {numbers['srh_0_3km_m2_s2']:.0f} m2/s2 (CM1 getcape on the 1 km grid)")
    else:
        lines.append(f"  CAPE about {record['cape_target_j_kg']:.0f} J/kg, 0-6 km shear about "
                     f"{record['shear_0_6km_target_m_s']:.0f} m/s (the settings' targets; {record['numbers_reason']})")
    if record.get("written"):
        lines.append(f"Wrote {record['written']} (sha256 {record['sha256'][:12]}...)")
    lines.append(f"Run it: {record['use']}")
    return "\n".join(lines)


def _options_text(record: Mapping[str, Any]) -> str:
    def said(value: Any) -> str:
        return f"{value:g}" if isinstance(value, float) else str(value)

    lines = [f"Settings of {record['preset']} (gpuwm-cm1 preset fit|new {record['preset']} --custom KNOB=VALUE):"]
    for knob in record["knobs"]:
        unit = f" {knob['unit']}" if knob.get("unit") else ""
        if knob["id"] == "area":
            shown = (f"{knob['min_km']:g} to {knob['max_km']:g} km in steps of {knob['step_cells']} cells "
                     "(another width is rounded to the nearest step)")
        elif knob["id"] == "grid":
            shown = (f"any spacing up to {knob['max_m']:g} m (grid=67); the area keeps its size, rounded up to whole "
                     "cells, and the time step follows. Quick picks: "
                     + ", ".join(said(choice["value"]) for choice in knob["choices"]))
        elif knob["id"] == "levels":
            low, high = knob["levels_range"]
            shown = (f"any height from {knob['min_m']:g} to {knob['max_m']:g} m ({high} to {low} levels up to "
                     f"{knob['top_m']:g} m; the top moves by less than half a level when the height does not divide "
                     "it). Quick picks: " + ", ".join(said(choice["value"]) for choice in knob["choices"]))
        else:
            shown = ", ".join(("seed:N" if knob["id"] == "bumps" and choice["value"] == "seed" else said(choice["value"]))
                              + ("" if choice.get("available", True) else " (not offered here)")
                              for choice in knob["choices"])
        lines.append(f"  {knob['id']:<9} now {said(knob['default'])}{unit}; {shown}")
    air = record.get("air") or {}
    lines.append("  air       " + ("--instability 0..1 --shear 0..1 --hodograph curved|straight" if air.get("available")
                                   else str(air.get("why") or "not offered")))
    return "\n".join(lines)


def _list_text(record: Mapping[str, Any]) -> str:
    lines = ["Presets (gpuwm-cm1 preset fit ID --budget 10m|1h|overnight|any):"]
    for row in record["presets"]:
        state = "offered" if row["offered"] else (row["why_not"] or "not offered")
        facts = row.get("facts") or {}
        lines.append(f"  {row['id']:<20} {row['name']}")
        lines.append(f"  {'':<20} {row['line']}")
        if facts:
            lines.append(f"  {'':<20} {facts['words']}; {facts['scheme']}; opens on {row['budget_words'].lower()}")
        lines.append(f"  {'':<20} {state}")
    return "\n".join(lines)


def _show_text(record: Mapping[str, Any]) -> str:
    lines = [f"{record['name']} ({record['id']})", f"  {record['line']}"]
    if record.get("facts"):
        lines.append(f"  {record['facts']['words']}; {record['facts']['scheme']}")
    lines.append("  " + ("Offered" if record.get("offered") else (record.get("why_not") or "Not offered")))
    source = record.get("source") or {}
    if source:
        lines.append(f"  From {source.get('file')} (sha256 {str(source.get('sha256'))[:12]}...), changed:")
        for change in source.get("changes") or []:
            lines.append(f"    {change['key']} = {change['to']} (CM1's file: {change['from']}): {change['why']}")
    if record.get("gm1_settings_why"):
        lines.append(f"  {record['gm1_settings_why']}")
    lines.append("")
    lines.append(record["namelist"].rstrip())
    return "\n".join(lines)


def _fit_text(record: Mapping[str, Any]) -> str:
    gpu = record.get("gpu") or {}
    head = f"{record['preset_name']}, {record['budget_words']}"
    if gpu:
        head += f", on {gpu.get('name')}"
    lines = [head]
    if record["status"] == "fits":
        lines.append(f"  {record['summary']}")
        exact = (record.get("facts") or {}).get("exact")
        if exact and record.get("customized"):
            lines.append(f"  {exact}; time step {(record.get('facts') or {}).get('dt_s')!r} s")
        lines += _changes_lines(record)
        lines += _warning_lines(record)
    else:
        reason = str(record["reason"])
        lines.append(f"  {reason}" if reason.lower().startswith("not offered") else f"  Not offered: {reason}")
        lines += [f"  - You changed: {sentence}" for sentence in _customized_sentences(record)]
    lines.append(f"  Command: {record['command']}")
    return "\n".join(lines)


def _new_text(record: Mapping[str, Any]) -> str:
    if not record["ok"]:
        reason = str(record["message"])
        reason = reason if reason.lower().startswith("not offered") else f"Not offered: {reason}"
        return f"{reason}\nNothing was written.\nCommand: {record['command']}"
    lines = [f"Wrote {record['run']}", f"  {record['message']}"]
    lines += _changes_lines(record["fit"])
    lines += _warning_lines(record["fit"])
    lines.append(f"Start it with: {record['next']}")
    return "\n".join(lines)


def _warning_lines(record: Mapping[str, Any]) -> list[str]:
    """What an offered run should be read with (a storm near the edge, an untried size or spacing); it still runs."""

    return [f"  - Note: {sentence}" for sentence in record.get("warnings") or []]


def _customized_sentences(record: Mapping[str, Any]) -> list[str]:
    return [str(change.get("sentence")) for change in record.get("customized") or [] if change.get("sentence")]


def _changes_lines(record: Mapping[str, Any]) -> list[str]:
    """What the user changed (Customize), then what the fit changed; "as the preset is" only when neither."""

    mine = _customized_sentences(record)
    fitted = [str(sentence) for sentence in record.get("sentences") or []]
    lines = [f"  - You changed: {sentence}" for sentence in mine]
    lines += [f"  - Changed to fit: {sentence}" if mine else f"  - {sentence}" for sentence in fitted]
    if not mine and not fitted:
        lines.append("  - Runs as the preset is; nothing was changed.")
    elif not fitted:
        lines.append("  - Nothing else was changed to fit.")
    return lines


__all__ = [
    "BUDGET_BUTTONS",
    "BUDGET_ORDER",
    "FIT_JSON",
    "GROUPS",
    "LIST_SCHEMA",
    "MOMENT_FIELDS",
    "PRESETS_ENV",
    "PRESET_SCHEMA",
    "Preset",
    "PresetError",
    "add_preset_parser",
    "command_line",
    "default_name",
    "examine",
    "facts",
    "fit",
    "fit_args",
    "fit_record",
    "given_args",
    "grid_words",
    "list_record",
    "load",
    "load_all",
    "new_args",
    "new_run",
    "picture",
    "preset_door",
    "presets_root",
    "sounding_record",
    "environment_record",
    "environment_words",
    "storm_track_for",
    "variants",
    "summary_record",
    "with_environment",
    "with_sounding",
]
