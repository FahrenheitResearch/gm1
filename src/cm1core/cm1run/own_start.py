"""A guided storm on the user's own sounding: the file in whichever layout it has, and how the storm is started.

``gpuwm-cm1 preset fit|new|options ID --sounding FILE --start push|bubble|strong-bubble|auto`` goes through here
(:func:`given`), so guided mode's "Use my own sounding" card and a terminal do the same thing.

**The file.**  ``--sounding`` takes CM1's ``input_sounding`` as before, and now also the other layouts
:mod:`..cm1import.sounding_formats` reads (a University of Wyoming text listing, a SHARPpy file, a table with
named columns): such a file is converted to a scratch ``input_sounding`` and the preset runs on that, with what
was done recorded in ``fit.json`` (``sounding.own``).  A file that cannot run is refused with its first problem,
its line and what to do.  A sounding written by ``gpuwm-cm1 sounding convert`` (or kept by the card) carries its
record beside it, and the run is named from the name the user's file had.

**The start.**  A preset whose namelist starts its storm with CM1's warm bubble (``iinit = 1``) can start it with
CM1's updraft nudging instead (``iinit = 12``, the push of rising air the famous-day presets use), and the other
way round.  The storm-track check follows the warm bubble only, so its recommended frame and starting place are
worked out for the bubble on this sounding first (``presets.with_sounding``), and the push is then centred where
the bubble would have been, which is how the famous-day presets were made.  A preset that starts with the push
(a famous day) given another sounding goes the same way: the bubble for the check, then its own push again.  The
fit's sentence says the frame was worked out for a bubble, since a storm from the push can move another way.
``auto`` takes the default :func:`..cm1import.own_sounding.start_choice` picks from the sounding.

Nothing here opens a GPU.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover
    from .presets import Preset

START_CHOICES = ("auto", "push", "bubble", "strong-bubble")
#: The warm bubble's strength a "stronger warm bubble" takes (K), one of Customize's own choices.
STRONG_BUBBLE_K = 3.0
#: CM1's iinit by start, and each start's GM1 settings for where the storm begins.
IINIT = {"bubble": 1, "strong-bubble": 1, "push": 12}
CENTRE_KEYS = {
    1: ("initialization.bubble_center_x_m", "initialization.bubble_center_y_m"),
    12: ("initialization.xc_wnudge", "initialization.yc_wnudge"),
}
START_SENTENCES = {
    12: ("The storm is started by a push of rising air for its first 20 minutes (CM1's updraft nudging) instead of "
         "a warm bubble."),
    1: "The storm is started by CM1's warm bubble instead of a push of rising air.",
}
CASE_WORDS = "Your sounding: {label}"


def _error(text: str, suggest: Mapping[str, Any] | None = None) -> Exception:
    from .presets import PresetError

    return PresetError(text, suggest=suggest)


def _iinit(preset: "Preset") -> int:
    from .customize import _effective, _int

    return _int(_effective(preset), "iinit")


def switch_start(preset: "Preset", iinit: int) -> "Preset":
    """``preset`` started the other way (``iinit`` 1 or 12), from the place its storm starts now."""

    from .customize import _scratch
    from .fit import set_namelist_values

    own = _iinit(preset)
    if own == iinit:
        return preset
    if own not in CENTRE_KEYS or iinit not in CENTRE_KEYS:
        raise _error(f"the preset {preset.id} starts its storm another way (iinit = {own}), so --start does not "
                     "apply to it; pick a preset that starts from a warm bubble or a push of rising air")
    text = set_namelist_values(preset.namelist_text, {"iinit": iinit})
    folder = _scratch()
    (folder / "namelist.input").write_text(text, encoding="ascii")
    record = dict(preset.record)
    settings = dict(record.get("gm1_settings") or {})
    keys: dict[str, list[Any]] = {"iinit": [own, iinit]}
    for old, new in zip(CENTRE_KEYS[own], CENTRE_KEYS[iinit]):
        if old in settings:
            value = settings.pop(old)
            keys[old] = [value, None]
            keys[new] = [settings.get(new), value]
            settings[new] = value
    # The bubble's own strength and size mean nothing to the push, and the push's to the bubble.
    stale = ("initialization.bptpert_k", "initialization.bhrad_m") if iinit == 12 else ("initialization.wmax_wnudge",)
    for key in stale:
        if key in settings:
            keys[key] = [settings.pop(key), None]
    record["gm1_settings"] = settings
    record["customized"] = [*[entry for entry in record.get("customized") or [] if entry.get("knob") != "start"],
                            {"knob": "start", "keys": keys, "sentence": START_SENTENCES[iinit]}]
    given = preset.sounding_given
    if given and given.get("follow"):
        # The frame was worked out for the bubble on this sounding; the storm now starts at the same place.
        follow = dict(given["follow"])
        words = {1: "warm bubble", 12: "push of rising air"}
        follow["keys"] = {(CENTRE_KEYS[iinit][CENTRE_KEYS[own].index(key)] if key in CENTRE_KEYS[own] else key): pair
                          for key, pair in (follow.get("keys") or {}).items()}
        follow["sentence"] = (follow_words(follow["keys"], iinit) or
                              str(follow.get("sentence") or "").replace(words[own], words[iinit]))
        given = {**given, "follow": follow}
    return replace(preset, record=record, namelist_text=text, namelist_file=folder / "namelist.input",
                   namelist_original=preset.namelist_original_text, sounding_given=given)


def _where(x_km: float, y_km: float) -> str:
    return (f"{abs(x_km):.1f} km {'east' if x_km >= 0 else 'west'} and {abs(y_km):.1f} km "
            f"{'north' if y_km >= 0 else 'south'} of the centre")


def follow_words(keys: Mapping[str, Any], iinit: int) -> str | None:
    """The fit's sentence for the frame and the start's place on a sounding file, saying what is known of it.

    The storm-track check follows a warm bubble only: for the bubble it expects the storm to stay inside; for the
    push of rising air the same frame and place are used, worked out for a bubble, and the push's storm can move
    another way (on GM1's own runs it drifted toward an edge late in a 2-hour storm)."""

    try:
        u, v = float(keys["umove"][1]), float(keys["vmove"][1])
        x, y = (float(keys[key][1]) / 1000.0 for key in CENTRE_KEYS[iinit])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    frame = (f"The view moves with the storm at {abs(u):.1f} m/s toward the {'east' if u >= 0 else 'west'} and "
             f"{abs(v):.1f} m/s toward the {'north' if v >= 0 else 'south'} (ground-relative)")
    if iinit == 12:
        return (f"{frame}, and the push of rising air starts {_where(x, y)}. These were worked out for a warm "
                "bubble started there, since GM1's storm-track check follows a bubble only; a storm started by the "
                "push can move another way, so on a long run it may come near an edge of the area.")
    return (f"{frame}, and the warm bubble starts {_where(x, y)}, where the storm-track check expects the storm "
            "to stay inside the area.")


def prepared(preset: "Preset", sounding: Path | str) -> tuple[Path, dict[str, Any] | None]:
    """(the ``input_sounding`` the preset runs on, what is known of the user's file or None) for ``--sounding``.

    None means a plain CM1 file with no record beside it and nothing to say about it: the 0.1.2 path, unchanged.
    """

    from ..cm1import import own_sounding, sounding_formats
    from ..cm1import.sounding_door import column_words

    path = Path(sounding).expanduser()
    if not path.is_file():
        raise _error(f"--sounding {path}: there is no such file")
    if column_words(path) is not None:
        return path, None                     # a column kept from a WRF or ArWen run says its own source
    record = own_sounding.kept(path)
    if record is not None:
        return path, {**record, "original": record.get("original"), "file_given": str(path.resolve())}
    try:
        text = path.read_bytes().decode("utf-8")
        layout = sounding_formats.detect(text)
    except (UnicodeError, sounding_formats.SoundingRefused):
        return path, None                     # the preset's own check says why CM1 cannot read it
    label = None if path.name == "input_sounding" else own_sounding.label_from(path.name)
    if layout == "cm1" and sounding_formats.strict_cm1(text):
        if label is None:
            return path, None
        return path, {"name": path.name, "label": label, "format": "cm1", "converted": False,
                      "file_given": str(path.resolve())}
    # Another layout, or a CM1 file the engine would not read as it is (a blank line, commas, extra words): the
    # file the run reads is written from it, and the run keeps the user's own beside it.
    top, nz, dz = _model_levels(preset)
    record = own_sounding.examine(text, name=path.name, top_needed_m=top, nz=nz, dz_m=dz)
    if label is None:
        record["label"] = None
    if not record["ok"]:
        problem = record["problems"][0]
        hint = ""
        if problem["code"] == "top" and (problem.get("offer") or {}).get("extend_to_m") and top is not None:
            hint = (f" To add the levels: gpuwm-cm1 sounding convert {path} OUT --extend-top --top-m {top:g}, then "
                    "give OUT as --sounding.")
        elif problem["code"] == "speed-unit":
            hint = f" gpuwm-cm1 sounding convert {path} OUT --speed-unit kt (or ms), then give OUT as --sounding."
        raise _error(f"--sounding {path}: {problem['text']} {problem.get('fix') or ''}{hint}".strip())
    from .presets import SOUNDING_FILE, _scratch

    out = _scratch("gm1-sounding-") / SOUNDING_FILE
    own_sounding.write(record, out)
    kept = {key: value for key, value in record.items() if key not in ("text", "report")}
    return out, {**kept, "original": str(path.resolve()), "file_given": str(path.resolve())}


def _model_levels(preset: "Preset") -> tuple[float | None, int | None, float | None]:
    """(the highest scalar level, nz, dz) of the preset's own vertical grid; nz and dz only when it is uniform."""

    from ..cm1import.sounding import highest_mass_level_m
    from .customize import _effective, _float, _int

    effective = _effective(preset)
    top = highest_mass_level_m(effective)
    if _int(effective, "stretch_z") != 0:
        return top, None, None
    return top, _int(effective, "nz"), _float(effective, "dz")


def given(preset: "Preset", custom: Mapping[str, Any], sets: Mapping[str, Any], sounding: Path | str | None,
          start: str | None) -> "Preset":
    """``preset`` with its Customize choices, its ``--sounding`` file and its ``--start``, in the order that works."""

    from ..cm1import import own_sounding, sounding_formats
    from ..cm1import.sounding_door import column_words
    from .customize import CustomError, apply
    from .presets import with_sounding

    custom = dict(custom)
    if start is not None and start not in START_CHOICES:
        raise _error(f"--start is one of {', '.join(START_CHOICES)}, not {start!r}")
    path: Path | None = None
    own: dict[str, Any] | None = None
    if sounding is not None:
        path, own = prepared(preset, sounding)
    own_iinit = _iinit(preset)
    own_settings = dict(preset.record.get("gm1_settings") or {})
    chosen = start
    if start == "auto":
        if path is None:
            raise _error("--start auto picks the start from a sounding file, so it goes with --sounding FILE")
        try:
            profile = sounding_formats.parse_cm1(path.read_bytes().decode("utf-8"))
            chosen = own_sounding.start_choice(profile)["recommended"]
        except (UnicodeError, sounding_formats.SoundingRefused):
            chosen = "push"
    if chosen == "strong-bubble":
        custom.setdefault("push", STRONG_BUBBLE_K)
    want = IINIT.get(chosen or "")
    # A preset that starts with the push, given a sounding: its frame and place are worked out for a bubble on
    # that sounding, then it starts with its own push again (as for --start push on a bubble preset).
    round_trip = path is not None and own_iinit == 12 and want in (None, 12)
    pushed = want == 12 or round_trip
    try:
        if want == 1 or round_trip:
            preset = switch_start(preset, 1)        # the bubble first, so the track check and its knobs are the bubble's
        late = {key: custom.pop(key) for key in ("push",) if pushed and key in custom}
        if pushed and "bubble" in custom and _iinit(preset) == 1:
            raise CustomError("A push of rising air has no bubble size to set; leave out bubble=KM, or use "
                              "--start bubble")
        preset = apply(preset, custom, sets)
        if path is not None:
            preset = with_sounding(preset, path, source=(own_sounding.source_words(own) if own else
                                                         column_words(path)))
        if pushed:
            preset = switch_start(preset, 12)
            if round_trip:
                preset = _own_start_again(preset, own_settings)
            if late:
                preset = apply(preset, late)
    except CustomError as error:
        # A value Customize takes instead of the one refused (a grid spacing, a number of levels) stays a button.
        raise _error(str(error), getattr(error, "suggest", None)) from error
    if start is not None:
        record = dict(preset.record)
        record["start"] = start
        preset = replace(preset, record=record)
    if own is not None and preset.sounding_given is not None:
        marked = dict(preset.sounding_given)
        follow = marked.get("follow")
        if isinstance(follow, Mapping) and _iinit(preset) == 1:
            # The bubble's own sentence, in the words the push's has (km east and north, one decimal).
            follow = dict(follow)
            follow["sentence"] = follow_words(follow.get("keys") or {}, 1) or follow.get("sentence")
            marked["follow"] = follow
        marked["file"] = own.get("file_given") or marked.get("file")
        marked["label"] = own.get("label")
        marked["own"] = {key: own.get(key) for key in (
            "name", "label", "format", "format_words", "converted", "tidied", "conversions", "dropped", "notes",
            "warnings",
            "extension", "levels", "top_m", "numbers", "verdict", "start", "original", "shipped") if key in own}
        preset = replace(preset, sounding_given=marked)
    return preset


def _own_start_again(preset: "Preset", settings: Mapping[str, Any]) -> "Preset":
    """A push preset back on its own start after the bubble's track check: no "start" change is recorded (it
    starts as it ships), and a push setting of its own that the switch took out is put back."""

    record = dict(preset.record)
    record["customized"] = [entry for entry in record.get("customized") or [] if entry.get("knob") != "start"]
    kept = dict(record.get("gm1_settings") or {})
    for key in ("initialization.wmax_wnudge",):
        if key in settings and key not in kept:
            kept[key] = settings[key]
    record["gm1_settings"] = kept
    return replace(preset, record=record)


def label_of(preset: "Preset") -> str | None:
    """The name the user's sounding file had, as a run-name stem (None: not a file of the user's own)."""

    label = (preset.sounding_given or {}).get("label")
    return str(label) if label else None


def run_name(preset: "Preset", stamp: str) -> str | None:
    label = label_of(preset)
    return f"{label}-{stamp}"[:60] if label else None


def keep_original(preset: "Preset", rundir: Path) -> None:
    """The user's own file beside the run's ``input_sounding`` when that was converted from it."""

    own = (preset.sounding_given or {}).get("own") or {}
    original = own.get("original")
    if not (own.get("converted") or own.get("tidied")) or not original:
        return
    source = Path(str(original))
    try:
        data = source.read_bytes()
    except OSError:
        return
    (rundir / "input_sounding.original").write_bytes(data)
    own["original_sha256"] = hashlib.sha256(data).hexdigest()


__all__ = ["CASE_WORDS", "IINIT", "START_CHOICES", "STRONG_BUBBLE_K", "follow_words", "given", "keep_original",
           "label_of", "prepared", "run_name", "switch_start"]
