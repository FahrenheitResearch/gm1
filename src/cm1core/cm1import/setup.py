"""The ``setup`` block of ``import-report.json``: what the GUI's setup screens show.

The coverage report says what GM1 does with each CM1 setting.  The setup
screens also need the values themselves, in one place, with CM1's own names:

``values``        the effective CM1 settings the screens show (grid, time, frame, boundaries, start-up)
``fortran``       the values CM1 keeps in Fortran, by group (warm bubble, cold pool, line thermal,
                  parcel seed box), each with the value this run uses, CM1's own value, whether the
                  run's ``settings.toml`` changed it, and whether the namelist's ``iinit``/``iprcl``
                  reads the group at all
``output_words``  every ``&param9`` word with its value and what GM1 does with it
``user_settings`` the run's own choices from ``settings.toml`` (``usersettings``)

Written by the importer, which has the parsed namelist and the deck, so the
GUI server reads one file and never parses a namelist or imports the model.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping

from . import usersettings

SETUP_SCHEMA = "gm1.cm1-setup/v1"

#: The CM1 settings the setup screens show, in CM1's own names.
SETUP_KEYS: tuple[str, ...] = (
    "nx", "ny", "nz", "dx", "dy", "dz", "stretch_x", "stretch_y", "stretch_z", "ztop", "str_bot", "str_top",
    "dz_bot", "dz_top",
    "dtl", "timax", "adapt_dt", "cflmax", "tapfrq", "rstfrq", "statfrq", "prclfrq",
    "imove", "umove", "vmove", "iorigin",
    "wbc", "ebc", "sbc", "nbc", "bbc", "tbc", "irbc", "sfcmodel", "isfcflx", "icor", "fcor", "lspgrad",
    "isnd", "iwnd", "iinit", "irandp", "iprcl", "imoist", "ptype", "ihail", "sgsmodel", "tconfig",
    "output_format", "output_filetype",
)

#: The Settings screen's groups of values CM1 keeps in Fortran.
GROUPS: tuple[dict[str, Any], ...] = (
    {"id": "bubble", "title": "Warm bubble", "plain": "the warm bubble that starts the storm",
     "selector": "iinit = 1", "source": "init3d.F:456-476"},
    {"id": "coldpool", "title": "Cold pool", "plain": "the block of cold air that starts the storm",
     "selector": "iinit = 2", "source": "init3d.F:495-525"},
    {"id": "line", "title": "Line thermal with noise", "plain": "the line of warm air, with small random noise, "
     "that starts a line of storms", "selector": "iinit = 8", "source": "init3d.F:1264-1300"},
    {"id": "nudging", "title": "Updraft nudging", "plain": "the gentle push of rising air that starts the storm",
     "selector": "iinit = 12", "source": "init3d.F:1370-1396, misclibs.F:539-585, sound.F:564-590"},
    {"id": "convergence", "title": "Forced convergence", "plain": "the air pushed together near the ground that "
     "starts the storm", "selector": "iinit = 9", "source": "init3d.F:1303-1333, misclibs.F:456-531"},
    {"id": "uforce", "title": "Momentum forcing", "plain": "the push of wind that starts a line of storms",
     "selector": "iinit = 10", "source": "init3d.F:1334-1347, solve1.F:1245-1265"},
    {"id": "benchmark", "title": "Moist benchmark bubble", "plain": "the saturated warm bubble of Bryan and "
     "Fritsch's (2002) moist benchmark, the same across the domain in y", "selector": "iinit = 4",
     "source": "init3d.F:579-647"},
    {"id": "benchmark_sounding", "title": "Saturated neutral sounding", "plain": "the moist benchmark's saturated, "
     "neutrally stable air", "selector": "isnd = 4", "source": "base.F:263-355"},
    {"id": "parcels", "title": "Parcels", "plain": "air parcels followed through the run",
     "selector": "iprcl = 1", "source": "parcel.F, init3d.F (the seed box)"},
)

_ACTIVE = {"bubble": ("iinit", 1), "coldpool": ("iinit", 2), "line": ("iinit", 8), "nudging": ("iinit", 12),
           "convergence": ("iinit", 9), "uforce": ("iinit", 10), "benchmark": ("iinit", 4),
           "benchmark_sounding": ("isnd", 4), "parcels": ("iprcl", 1)}


def _deck_tables(report: Any) -> dict[str, Any]:
    text = getattr(report, "toml", None)
    if not text:
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _display(item: usersettings.Editable) -> tuple[str, float]:
    """(unit shown, factor from the stored unit): lengths in km, the rest as stored."""

    if item.unit == "m":
        return "km", 0.001
    return item.unit, 1.0


def _cm1_values(item: usersettings.Editable, effective: Mapping[str, Any], deck: Mapping[str, Any],
                arm: int | None = None, active: bool = True) -> tuple[Any, Any]:
    """(the value this run uses, CM1's own value), in the stored unit.

    ``arm`` is the ``iinit`` whose named parameters give CM1's own value
    (init3d.F assigns them per arm); a group the namelist does not select shows
    its own arm's values, and the deck's declared value only where it is read.
    """

    iinit = _int(effective.get("iinit"), 1) if arm is None else arm
    if item.table == "initialization":
        from ..config import _initialization_word  # CM1's per-arm named parameters (init3d.F)

        declared = dict(deck.get("initialization") or {})
        if item.name in ("bubble_center_x_m", "bubble_center_y_m", "xc_wnudge", "yc_wnudge", "xc_uforce"):
            nx, ny = _int(effective.get("nx"), 1), _int(effective.get("ny"), 1)
            dx, dy = _float(effective.get("dx"), 1.0), _float(effective.get("dy"), 1.0)
            iorigin = _int(effective.get("iorigin"), 1)
            # param.F:7408 centerx/centery: the domain centre, in the iorigin frame (param.F:7212-7213).
            centre = (0.0, 0.0) if iorigin == 2 else (0.5 * nx * dx, 0.5 * ny * dy)
            branch = _initialization_word({}, item.name, iinit)
            default = branch if branch is not None else centre[0 if item.name[0] == "x" or item.name.endswith("x_m") else 1]
            value = declared.get(item.name, default) if active else default
            return (default if value is None else value), default
        default = _initialization_word({}, item.name, iinit)
        return (declared.get(item.name, default) if active else default), default
    if item.table == "lagrangian":
        from ..lagrangian_config import _DEFAULTS as LAGRANGIAN_DEFAULTS

        declared = dict(deck.get("lagrangian") or {})
        default = LAGRANGIAN_DEFAULTS.get(item.name)
        return declared.get(item.name, default), default
    return None, None


def fortran_groups(effective: Mapping[str, Any], deck: Mapping[str, Any],
                   user: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = []
    for spec in GROUPS:
        selector, wanted = _ACTIVE[spec["id"]]
        active = _int(effective.get(selector), 0) == wanted
        fields = []
        for item in usersettings.EDITABLE:
            if spec["id"] not in item.groups:
                continue
            arm = wanted if selector == "iinit" else None
            value, default = _cm1_values(item, effective, deck, arm, active)
            unit, factor = _display(item)
            shown = value
            shown_default = default
            if item.kind == "float":
                shown = None if value is None else round(float(value) * factor, 6)
                shown_default = None if default is None else round(float(default) * factor, 6)
            fields.append({
                "key": item.key,
                "label": item.label,
                "cm1": item.cm1,
                "kind": item.kind,
                "unit": unit,
                "factor": factor,
                "value": shown,
                "cm1_value": shown_default,
                "changed": item.key in user,
            })
        groups.append({**spec, "active": active, "fields": fields})
    return groups


def output_words(effective: Mapping[str, Any], report: Any) -> list[dict[str, Any]]:
    """Every ``&param9`` output word: its value and what GM1 does with it."""

    from ..cm1_output_selection import PRODUCT, STATE, SWITCHES, UNSUPPORTED
    from .translate import UNIFORM_METRIC_ONLY

    component = str((getattr(report, "component", None) or {}).get("component") or "")
    stretched = any(_int(effective.get(key)) for key in ("stretch_x", "stretch_y", "stretch_z"))
    by_key: dict[str, Any] = {}
    for row in getattr(report, "rows", []) or []:
        for key in row.keys:
            if key.startswith("output_") and key not in by_key and row.category in (
                    "differs", "refused", "not_read", "not_checked"):
                by_key[key] = row
    words = []
    for switch in SWITCHES:
        if switch.key in ("output_format", "output_filetype"):
            continue
        value = _int(effective.get(switch.key), switch.cm1_default)
        row = by_key.get(switch.key)
        why = "" if row is None else (row.plain or row.text or "")
        if switch.disposition == STATE:
            state, label = "always", "always written"
            if not why:
                why = "GM1 writes this field in every frame: its resume and viewer read it"
        elif switch.disposition == UNSUPPORTED:
            state, label = "refused", "not available"
            why = why or switch.defect
        elif switch.key in UNIFORM_METRIC_ONLY and stretched:
            state, label = "differs", "uniform grids only"
            why = why or "written on uniform grids only"
        elif switch.key == "output_dbz" and value and component != "wsm6-mp6":
            state, label = "differs", "after each frame"
            why = why or "worked out after each saved frame with CM1's calc_refl10cm formula"
        elif switch.disposition == PRODUCT:
            state, label = ("written", "written") if value else ("off", "off")
        else:  # pragma: no cover - the three dispositions are the whole table
            state, label = "off", switch.disposition
        words.append({
            "word": switch.key,
            "value": value,
            "state": state,
            "label": label,
            "why": why,
            "fields": list(switch.fields),
            "editable": switch.disposition == PRODUCT,
            "unvalidated": switch.unvalidated or None,
        })
    return words


def setup_block(effective: Mapping[str, Any] | None, report: Any, user: Mapping[str, Any],
                *, rundir: Path | None = None) -> dict[str, Any]:
    effective = dict(effective or {})
    deck = _deck_tables(report)
    values = {key: effective.get(key) for key in SETUP_KEYS if key in effective}
    seed = None
    if rundir is not None:
        seed = (getattr(report, "settings", {}) or {}).get("initialization", {}).get("random_seed")
    generator = str((deck.get("initialization") or {}).get("random_generator", "gm1"))
    return {
        "schema": SETUP_SCHEMA,
        "values": values,
        "fortran": fortran_groups(effective, deck, user),
        "output_words": output_words(effective, report),
        "user_settings": dict(user),
        "gpu": {
            "memory_budget_mib": (deck.get("runtime") or {}).get("memory_budget_mib"),
            "chosen": user.get("gpu.memory_budget_mib"),
            "note": ("The most GPU memory the run's device registry may take (GM1's runtime.memory_budget_mib; "
                     "CM1 has no such setting). The run takes only what it needs, up to this and to what the GPU "
                     "has free; the estimate recommends one."),
        },
        "random_seed": seed,
        # "cm1": CM1's own random numbers (init3d.F:427-440), word for word;
        # "gm1": GM1's stream with random_seed.
        "random_generator": generator,
        "deck_initialization": dict(deck.get("initialization") or {}),
        "note": ("Values CM1 keeps in Fortran come from CM1 r21.1's init3d.F and parcel.F, which GM1 uses unless "
                 "this run's settings.toml changes them. Heat-flux initiation and custom Fortran perturbations "
                 "are not available."),
    }


__all__ = ["GROUPS", "SETUP_KEYS", "SETUP_SCHEMA", "fortran_groups", "output_words", "setup_block"]
