"""Whether a quick-look picture shows a storm yet, in plain words, and when the next picture comes.

A run starts from quiet air: the warm bubble (or the updraft nudging) takes some storm time to grow a storm, so
the first pictures are dark.  Someone watching should not have to guess whether that is a fault.  The quick-look
index (``gm1.quicklook-index/v1``) already records, for every saved time, the strongest reflectivity at 1 km and
in the whole column (the reflectivity look's facts) and the strongest updraft at 5 km (the updraft look's facts),
so the words come from the run's own numbers:

``none``     every number known is below :data:`ECHO_DBZ` and :data:`UPDRAFT_M_S`: no storm yet;
``forming``  the column or the 5 km updraft has one, but 1 km has no echo yet: the storm is starting higher up;
(nothing)    the 1 km picture shows an echo, at this time or at an earlier one.  A storm that has rained itself
             out later in the run is not "not yet", so no time after the first storm gets a line.

A run that ran to its end (or past the time its preset's storm usually shows) with no echo at 1 km at any saved
time says that no storm formed, not "not yet".  A run stopped (or failed) earlier says when it ended, before the
storm's usual time: it is not a verdict on the air.  A look drawn once per run (the environment, the parcels) is
not a moment of the storm and gets no line.

A preset whose storm forms at a known time says so (``storm_appears`` in its ``preset.json``, measured); every
other run gets the general line.  The words are guided mode's copy (``copy/guided.json``, ``storm.*``).

A hurricane (``"storm_kind": "hurricane"`` in ``preset.json``) starts from a swirl that is there from the first
picture, so "No storm yet" would be wrong for it.  Its lines are its own (``storm.hurricane.*``): the swirl has
no rain yet; rain cells are forming and the swirl looks ragged, with the time the eye is usually clear by
(``eye_forms`` in ``preset.json``, with its basis); and, for a run that ended with no rain at all, that no rain
formed.  Once the eye's usual time has passed the pictures speak for themselves and get no line.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

#: A picture with no reflectivity this strong anywhere shows no storm (5 dBZ is about the weakest echo a weather
#: radar shows as rain; a cloud with no rain is below it).
ECHO_DBZ = 5.0
#: An updraft at 5 km this strong is a growing storm, not the gentle rising of the starting air (measured on the
#: 1 km supercell: 0.2 to 0.3 m/s before the storm, 10 m/s as it forms).
UPDRAFT_M_S = 2.0


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _key(time_s: Any) -> float | None:
    value = _number(time_s)
    return None if value is None else round(value, 3)


def clock(seconds: float) -> str:
    """Storm time as T+H:MM:SS."""

    total = int(round(max(0.0, float(seconds))))
    return f"T+{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def signals(index: Mapping[str, Any] | None) -> dict[float, dict[str, float]]:
    """For every saved time, the numbers the quick-looks recorded: ``dbz_1km``, ``dbz_column`` and ``w_5km``."""

    out: dict[float, dict[str, float]] = {}
    if not isinstance(index, Mapping):
        return out
    wanted = {"reflectivity": (("dbz_max_1km", "dbz_1km"), ("dbz_max_column", "dbz_column")),
              "updraft-5km": (("w_max", "w_5km"),)}
    for product in index.get("products") or []:
        if not isinstance(product, Mapping) or product.get("id") not in wanted:
            continue
        for frame in product.get("frames") or []:
            if not isinstance(frame, Mapping) or not isinstance(frame.get("facts"), Mapping):
                continue
            key = _key(frame.get("time_s"))
            if key is None:
                continue
            for fact, name in wanted[str(product["id"])]:
                value = _number(frame["facts"].get(fact))
                if value is not None:
                    out.setdefault(key, {})[name] = value
    return out


def states(index: Mapping[str, Any] | None) -> dict[float, str]:
    """``"none"`` or ``"forming"`` for each saved time before the storm first shows at 1 km (see the module)."""

    out: dict[float, str] = {}
    for time_s, seen in sorted(signals(index).items()):
        if seen.get("dbz_1km", -math.inf) >= ECHO_DBZ:
            break
        if seen.get("dbz_column", -math.inf) >= ECHO_DBZ or seen.get("w_5km", -math.inf) >= UPDRAFT_M_S:
            out[time_s] = "forming"
        elif "dbz_1km" in seen or "dbz_column" in seen or "w_5km" in seen:
            out[time_s] = "none"
    return out


def appears(preset_id: Any, presets: Path) -> tuple[float, float] | None:
    """When the preset's storm usually first shows, ``(from_s, to_s)`` of storm time, or None if not known."""

    if not isinstance(preset_id, str) or not preset_id or "/" in preset_id or preset_id.startswith("."):
        return None
    try:
        record = json.loads((presets / preset_id / "preset.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    window = record.get("storm_appears") if isinstance(record, dict) else None
    if not isinstance(window, Mapping):
        return None
    start, end = _number(window.get("from_s")), _number(window.get("to_s"))
    if start is None or end is None or not 0.0 <= start <= end:
        return None
    return start, end


def storm_kind(preset_id: Any, presets: Path) -> str | None:
    """The preset's ``storm_kind`` (``"hurricane"``), or None for a thunderstorm preset or an unknown one."""

    record = _preset_record(preset_id, presets)
    kind = record.get("storm_kind") if record else None
    return str(kind) if isinstance(kind, str) and kind else None


def eye_forms(preset_id: Any, presets: Path) -> float | None:
    """The storm time (s) by which a hurricane preset's eye is usually clear (``eye_forms.to_s``), if recorded."""

    record = _preset_record(preset_id, presets)
    window = record.get("eye_forms") if record else None
    end = _number(window.get("to_s")) if isinstance(window, Mapping) else None
    return end if end is not None and end > 0.0 else None


def _preset_record(preset_id: Any, presets: Path) -> Mapping[str, Any] | None:
    if not isinstance(preset_id, str) or not preset_id or "/" in preset_id or preset_id.startswith("."):
        return None
    try:
        record = json.loads((presets / preset_id / "preset.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, Mapping) else None


def day_words(seconds: float) -> str:
    """A long storm time in words: "hour 48 (the end of day 2)", "hour 30"."""

    hours = int(round(float(seconds) / 3600.0))
    if hours >= 24 and hours % 24 == 0:
        return f"hour {hours} (the end of day {hours // 24})"
    return f"hour {hours}"


def hurricane_notes(index: Mapping[str, Any] | None, eye_s: float | None, strings: Mapping[str, Any],
                    ended: bool) -> dict[float, str]:
    """A hurricane run's line for every saved time that has one (see the module)."""

    def text(key: str) -> str:
        entry = strings.get(key) or {}
        return str(entry.get("text") if isinstance(entry, Mapping) else entry or "")

    seen = signals(index)
    rained = any(row.get("dbz_1km", -math.inf) >= ECHO_DBZ or row.get("dbz_column", -math.inf) >= ECHO_DBZ
                 for row in seen.values())
    out: dict[float, str] = {}
    for time_s, row in sorted(seen.items()):
        if "dbz_1km" not in row and "dbz_column" not in row:
            continue
        echo = row.get("dbz_1km", -math.inf) >= ECHO_DBZ or row.get("dbz_column", -math.inf) >= ECHO_DBZ
        if not echo:
            line = text("storm.hurricane.none_ended") if ended and not rained else text("storm.hurricane.none")
        elif eye_s is not None and time_s < eye_s - 1e-6:
            line = text("storm.hurricane.ragged").replace("{eye}", day_words(eye_s))
        else:
            line = ""
        if line:
            out[time_s] = line
    return out


def shows_storm(line: str | None, strings: Mapping[str, Any]) -> bool:
    """Whether a picture's line is said beside a storm that is in the picture (a hurricane's ragged first day),
    not in place of one ("No storm yet", "no rain yet"): such a picture still has a strongest value to say."""

    entry = strings.get("storm.hurricane.ragged") or {}
    lead = str(entry.get("text") if isinstance(entry, Mapping) else entry or "").split("{eye}")[0]
    return bool(line) and bool(lead) and str(line).startswith(lead)


def words(state: str, window: tuple[float, float] | None, strings: Mapping[str, Any],
          ended_s: float | None = None) -> str | None:
    """The line for one picture, from guided mode's copy (``strings`` is its ``strings`` table).  ``state``
    ``"stopped"``: no storm yet in a run that ended at ``ended_s``, before ``window``."""

    def text(key: str) -> str:
        entry = strings.get(key) or {}
        return str(entry.get("text") if isinstance(entry, Mapping) else entry or "")

    if state == "forming":
        return text("storm.forming") or None
    if state == "never":
        return text("storm.none_ended") or None
    if state == "stopped" and window is not None and ended_s is not None:
        line = text("storm.none_stopped")
        if line:
            return line.replace("{clock}", clock(ended_s)).replace("{from}", clock(window[0])).replace(
                "{to}", clock(window[1]))
        state = "none"
    if state != "none":
        return None
    if window is not None:
        return text("storm.none_usual").replace("{from}", clock(window[0])).replace("{to}", clock(window[1])) or None
    return text("storm.none") or None


def notes(run: Mapping[str, Any] | None, index: Mapping[str, Any] | None, presets: Path,
          strings: Mapping[str, Any]) -> dict[float, str]:
    """The line for every saved time that has one, keyed by the time (seconds, rounded to 1 ms)."""

    preset_id = (run or {}).get("preset") if isinstance(run, Mapping) else None
    if storm_kind(preset_id, presets) == "hurricane":
        ended = bool(isinstance(index, Mapping) and index.get("run_ended"))
        return hurricane_notes(index, eye_forms(preset_id, presets), strings, ended)
    found = states(index)
    if not found:
        return {}
    window = appears((run or {}).get("preset") if isinstance(run, Mapping) else None, presets)
    # A run that has ended with no storm at 1 km at any saved time: "not yet" would promise one, if it ran to its
    # end or past the storm's usual time; one stopped before then only ended early.
    seen = signals(index)
    ended = bool(isinstance(index, Mapping) and index.get("run_ended")) and len(found) == len(seen)
    last = max(seen) if seen else None
    reached = ended and (
        str(index.get("end_reason") or "") == "completed"
        or (last is not None and _ran_to_end(run, last))
        or (window is not None and last is not None and last >= window[1] - 1e-6))
    early = ended and not reached
    out = {}
    for time_s, state in found.items():
        if state == "none" and reached:
            state = "never"
        elif state == "none" and early and window is not None:
            state = "stopped"
        line = words(state, window, strings, ended_s=last)
        if line:
            out[time_s] = line
    return out


def _ran_to_end(run: Mapping[str, Any] | None, last_s: float) -> bool:
    """Whether the last saved time is the run's end time (run.json's ``duration_s``, when it has one)."""

    duration = _number((run or {}).get("duration_s")) if isinstance(run, Mapping) else None
    return duration is not None and last_s >= duration - 1e-6


def run_notes(rundir: Path, index: Any) -> dict[float, str]:
    """:func:`notes` for the run folder ``rundir`` (its ``run.json`` names the preset), in guided mode's words."""

    from .guided import load_copy, presets_dir

    if not isinstance(index, Mapping) or not index.get("products"):
        return {}
    try:
        run = json.loads((Path(rundir) / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run = {}
    return notes(run if isinstance(run, Mapping) else {}, index, presets_dir(),
                 load_copy()["guided"].get("strings") or {})


def note_for(notes_by_time: Mapping[float, str], time_s: Any) -> str | None:
    key = _key(time_s)
    return None if key is None else notes_by_time.get(key)


def schedule(admitted: Mapping[str, Any] | None, status: Mapping[str, Any] | None,
             shown_s: float | None) -> dict[str, Any] | None:
    """When pictures are saved: ``interval_s``, ``first_s`` (this segment's first saved time), ``shown_s`` (the
    newest picture drawn) and ``next_s`` (the next one to come, or None once the last is drawn)."""

    admitted = admitted if isinstance(admitted, Mapping) else {}
    status = status if isinstance(status, Mapping) else {}
    interval = _number(admitted.get("history_interval_s"))
    if interval is None or interval <= 0:
        return None
    duration = _number(status.get("duration_s")) or _number(admitted.get("duration_s"))
    step = _number(admitted.get("start_step")) or 0.0
    dt = _number(admitted.get("dt_s")) or 0.0
    start = step * dt
    # A run's first segment saves its starting state; a later one goes on from a restart already saved, so its
    # first picture is the next saved time after it.
    first = 0.0 if start <= 0.0 else (math.floor(start / interval + 1e-9) + 1) * interval
    following = first if shown_s is None else shown_s + interval
    if duration is not None and following > duration + 1e-6:
        following = None
    return {"interval_s": interval, "first_s": first, "shown_s": shown_s, "next_s": following,
            "duration_s": duration}


__all__ = ["ECHO_DBZ", "UPDRAFT_M_S", "appears", "clock", "day_words", "eye_forms", "hurricane_notes", "note_for",
           "notes", "run_notes", "schedule", "shows_storm", "signals", "states", "storm_kind", "words"]
