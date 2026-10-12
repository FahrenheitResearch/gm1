"""What a run needs, before a GPU is taken: ``gpuwm-cm1 estimate``.

``price(deck, gpu)`` answers for one deck on one GPU, on the host, in well
under a second:

* **GPU memory**: the door's own priced registry peak for the closure tile the
  run will use (``cartesian_door.closure_decomposition_peak_bytes``), plus the
  measured residual of ``records/GM1-GPU-RATES.json`` (context, compiled
  modules, pool slack; 0.55 GB + 80 B per cell, 0.56 GB more for WSM6).  It
  fits when it is at most 0.95 of the memory ``nvidia-smi`` says is free,
  which on a Windows desktop is what the desktop leaves;
* **locked host memory**: the two page-locked publication arenas, a history
  frame and a restart, at the plan's own payloads;
* **disk**: every frame (with ``gpuwm-cm1 run``'s t = 0 frame and no second
  envelope copy), its cm1out netCDF, every restart and the quick-looks;
* **wall time**: start-up, steps at the card's rate (this machine's own
  measurement when it has one), and the writes;
* **the storm-track check** (:func:`storm_track`): where CM1's ``iinit=1``
  bubble goes, from the wind the run integrates, and how close it comes to an
  open side.  The analytic ``isnd=5`` lane is evaluated here from base.F's own
  ``iwnd`` branch.

Every number carries a label -- measured, documented, estimated, rough, or
measured on this machine -- and its basis.  Nothing here imports CuPy.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import CM1Refusal
from . import rates as _rates
from .gpuinfo import GpuInfo, GpuQuery, HostInfo, query_gpus, query_host

ESTIMATE_SCHEMA = "gpuwm-cm1.estimate/v1"
STORM_TRACK_SCHEMA = "gpuwm-cm1.storm-track/v1"
GB = 1e9
MIB = 1024 * 1024

#: The guided fit's edge rule: the tracked
#: storm must stay this far from every open side for the whole run.
FIT_EDGE_CLEARANCE_M = 10_000.0

#: The fraction of free GPU memory a run may plan to use.
FITS_FRACTION_OF_FREE = 0.95
#: Locked host memory above this share of RAM is warned about.
HOST_LOCKED_WARN_FRACTION = 0.5
#: Disk above this share of the free space is warned about.
DISK_WARN_FRACTION = 0.8
#: The share of the free space a run may plan to write.  Start refuses a run
#: whose disk estimate exceeds free space minus 10 %, so the estimate never
#: says "fits" above it.
DISK_FITS_FRACTION = 0.9

#: A pricing budget high enough that the door's budget refusal never fires
#: while the peak itself is being priced; the deck's own budget is judged
#: separately (the ``memory-budget`` check).
_PRICING_BUDGET_MIB = 4_000_000

#: The closure tile the estimator recommends when the domain is wider than it.
RECOMMENDED_TILE = 128
#: Tile 128 is recommended for its memory alone only when its estimated step
#: time is within this factor of the whole domain's: every tile after the
#: first costs host launch time, so a ragged tiling can be much slower.
TILE_SPEED_TOLERANCE = 1.03


class EstimateError(ValueError):
    """The target cannot be priced; the message says why in plain words."""

    def __init__(self, message: str, *, refusal: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.refusal = dict(refusal) if refusal is not None else None


# ------------------------------------------------------------------ decks


@dataclass(frozen=True)
class Deck:
    """One run to price: its gpuwm-cm1 document, and where it came from.

    ``raw`` is the ``gpuwm-cm1.config/v1`` mapping (a TOML document, or the
    translation of a CM1 namelist).  ``sounding_bytes`` is the
    ``input_sounding`` the run would read, when it reads one.
    """

    raw: Mapping[str, Any]
    source: str
    kind: str
    name: str
    sounding_bytes: bytes | None = None
    sounding_label: str | None = None
    rundir: Path | None = None
    namelist_import: Any = None

    def with_runtime(self, **values: Any) -> "Deck":
        """The same deck with ``[runtime]`` words replaced (tile, budget)."""

        raw = copy.deepcopy(dict(self.raw))
        runtime = dict(raw.get("runtime", {}))
        for key, value in values.items():
            if value is None:
                runtime.pop(key, None)
            else:
                runtime[key] = value
        raw["runtime"] = runtime
        return replace(self, raw=raw)


def _looks_like_namelist(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return False
    for line in head.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        return stripped.startswith("&")
    return False


def _read_sounding(path: Path | None) -> tuple[bytes | None, str | None]:
    if path is None:
        return None, None
    return Path(path).read_bytes(), str(Path(path))


def deck_from_namelist(
    path: str | os.PathLike[str],
    *,
    sounding: str | os.PathLike[str] | None = None,
    gm1_only: Mapping[str, Any] | None = None,
) -> Deck:
    """A CM1 namelist, read and translated by the importer."""

    from ..cm1import import import_namelist
    from ..cm1import.namelist import NamelistError, NamelistToolError

    path = Path(path)
    try:
        imported = import_namelist(path, gm1_only=gm1_only)
    except NamelistToolError as error:
        raise EstimateError(f"the CM1 namelist parser is not available: {error}") from None
    except NamelistError as error:
        raise EstimateError(f"{path.name} is not a namelist the parser can read: {error}") from None
    except ValueError as error:  # TranslationError and the config's own input errors
        raise EstimateError(f"{path.name} cannot be priced: {error}") from None
    if imported.translation is None:
        why = imported.component.reason if imported.component.component is None else "; ".join(imported.cm1_stops)
        raise EstimateError(
            f"{path.name} cannot be priced: the importer does not translate it ({why}). "
            f"Run `gpuwm-cm1 import {path}` for the full report."
        )
    raw = tomllib.loads(imported.toml())
    sounding_bytes, label = _read_sounding(Path(sounding) if sounding else None)
    return Deck(
        raw=raw, source=str(path), kind="namelist", name=path.name,
        sounding_bytes=sounding_bytes, sounding_label=label, namelist_import=imported,
    )


def deck_from_mapping(raw: Mapping[str, Any], *, name: str = "deck", sounding: bytes | None = None) -> Deck:
    return Deck(raw=copy.deepcopy(dict(raw)), source=name, kind="mapping", name=name, sounding_bytes=sounding)


def load_deck(
    target: str | os.PathLike[str] | Mapping[str, Any] | Deck,
    *,
    sounding: str | os.PathLike[str] | None = None,
    gm1_only: Mapping[str, Any] | None = None,
) -> Deck:
    """A TOML deck, a CM1 namelist, or a run directory, as a :class:`Deck`.

    A run directory (``gm1.cm1-run/v1``) is priced from its ``deck.toml``, or
    from its ``namelist.input`` when the deck has not been generated yet, with
    its ``input_sounding`` when present.
    """

    if isinstance(target, Deck):
        return target
    if isinstance(target, Mapping):
        return deck_from_mapping(target)
    path = Path(target).expanduser()
    if path.is_dir():
        sounding_path = Path(sounding) if sounding else (path / "input_sounding")
        if not sounding_path.is_file():
            sounding_path = None
        if (path / "deck.toml").is_file():
            raw = tomllib.loads((path / "deck.toml").read_text(encoding="utf-8"))
            data, label = _read_sounding(sounding_path)
            return Deck(
                raw=raw, source=str(path), kind="rundir", name=path.name,
                sounding_bytes=data, sounding_label=label, rundir=path,
            )
        if (path / "namelist.input").is_file():
            deck = deck_from_namelist(path / "namelist.input", sounding=sounding_path, gm1_only=gm1_only)
            return replace(deck, source=str(path), kind="rundir", name=path.name, rundir=path)
        raise EstimateError(
            f"{path} is a folder with neither deck.toml nor namelist.input; "
            "point at a run folder, a gpuwm-cm1 deck (.toml) or a CM1 namelist"
        )
    if not path.is_file():
        raise EstimateError(f"{path} does not exist")
    if _looks_like_namelist(path):
        return deck_from_namelist(path, sounding=sounding, gm1_only=gm1_only)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeError) as error:
        raise EstimateError(f"{path} is neither a CM1 namelist nor a gpuwm-cm1 TOML deck: {error}") from None
    data, label = _read_sounding(Path(sounding) if sounding else None)
    return Deck(raw=raw, source=str(path), kind="toml", name=path.name, sounding_bytes=data, sounding_label=label)


# ------------------------------------------------------------ storm track


def storm_track(
    config: Any,
    *,
    sounding_bytes: bytes | None = None,
    min_edge_clearance_m: float = FIT_EDGE_CLEARANCE_M,
    sample_interval_s: float | None = None,
) -> dict[str, Any]:
    """Where CM1's warm bubble goes and how close it comes to an open side.

    The same host computation the door runs (``cartesian_door``'s
    ``run_geometry_hodograph`` and ``run_geometry.evaluate_run_geometry``),
    reported rather than refused, with the track sampled for a plan map.
    ``minimum_edge_clearance_km`` is the right-moving member's closest
    approach to an open side over the stated organisation band;
    ``meets_edge_rule`` is the guided fit's rule that it stays at least
    ``min_edge_clearance_m`` away and never leaves.
    """

    from .. import cartesian_door as door
    from ..run_geometry import (
        DEFAULT_TRACK_MODEL,
        MEMBER_LEFT,
        MEMBER_MEAN,
        MEMBER_RIGHT,
        evaluate_run_geometry,
        track_displacement,
    )

    grid = door.require_grid_extent(config)
    selection, _ = door.require_initializer_selection(config)
    frame = door.require_moving_frame(config)
    origin = door.horizontal_origin_m(config)
    centre = door.bubble_center_m(config)
    sides = door.open_lateral_sides(config)
    width, depth = grid.nx * grid.dx_m, grid.ny * grid.dy_m
    base: dict[str, Any] = {
        "schema": STORM_TRACK_SCHEMA,
        "domain_km": [width / 1000.0, depth / 1000.0],
        "origin_km": [origin[0] / 1000.0, origin[1] / 1000.0],
        "bubble_km": [centre[0] / 1000.0, centre[1] / 1000.0],
        "frame_m_s": [frame.u_move_m_s, frame.v_move_m_s],
        "open_sides": dict(zip(("west", "east", "south", "north"), sides)),
        "duration_s": config.run.duration_s,
        "isnd": selection.isnd,
        "iwnd": selection.iwnd,
        "iinit": selection.iinit,
        "fit_edge_rule_km": min_edge_clearance_m / 1000.0,
        "coordinates": (
            "km in the namelist's iorigin frame (param.F:7212-7213); the domain "
            "spans origin_km to origin_km + domain_km"
        ),
    }
    if not any(sides):
        return {
            **base, "status": "not-evaluated-doubly-periodic",
            "reason": "Every side repeats (wbc = ebc = sbc = nbc = 1), so a storm cannot leave the area.",
            "meets_edge_rule": True,
        }
    if selection.iinit != 1:
        return {
            **base, "status": "not-evaluated-not-a-warm-bubble",
            "reason": (
                f"The track model follows CM1's warm bubble (iinit = 1); iinit = {selection.iinit} "
                "starts the storm another way, whose motion it does not describe."
            ),
            "meets_edge_rule": True,
        }
    if selection.isnd == 7 and not sounding_bytes:
        return {
            **base, "status": "not-evaluated-no-readable-profile",
            "reason": "isnd = 7 takes its winds from input_sounding, and none was given (--sounding).",
            "meets_edge_rule": None,
        }
    try:
        hodograph, wind_source = door.run_geometry_hodograph(config, selection, sounding_bytes or b"")
        geometry = evaluate_run_geometry(
            hodograph=hodograph, nx=grid.nx, ny=grid.ny, dx_m=grid.dx_m, dy_m=grid.dy_m,
            duration_s=config.run.duration_s,
            frame_m_s=(frame.u_move_m_s, frame.v_move_m_s),
            bubble_center_xy_m=(centre[0] - origin[0], centre[1] - origin[1]),
            open_sides=sides,
        )
    except (TypeError, ValueError) as error:
        return {
            **base, "status": "not-evaluated-no-usable-profile",
            "reason": f"The wind profile cannot be used for a track: {error}",
            "meets_edge_rule": None,
        }
    right = geometry.members[MEMBER_RIGHT]
    left = geometry.members[MEMBER_LEFT]
    interval = sample_interval_s or max(60.0, min(config.run.history_interval_s, config.run.duration_s / 24.0))
    times = [0.0]
    while times[-1] + interval < config.run.duration_s - 1e-6:
        times.append(times[-1] + interval)
    times.append(float(config.run.duration_s))

    def sampled(member: str) -> list[list[float]]:
        rows = []
        for t in times:
            dx, dy = track_displacement(DEFAULT_TRACK_MODEL, geometry.steering_m_s, geometry.motion, member, t)
            x = centre[0] + dx - frame.u_move_m_s * t
            y = centre[1] + dy - frame.v_move_m_s * t
            rows.append([t, round(x / 1000.0, 3), round(y / 1000.0, 3)])
        return rows

    minimum = right.minimum_clearance_m
    admitted = selection.isnd in door.RUN_GEOMETRY_ADMITTED_ISND
    # A frame velocity must be a binary32 word (config.py refuses any other:
    # CM1-CFG-MOVING-FRAME-PRECISION), so the recommendation is rounded to
    # 1/8 m/s, which every such word near it is.  The bubble that goes with it
    # is the one recommended in that frame: the recommended bubble depends on
    # the frame the run moves in, so the pair is evaluated together and is a
    # fixed point (a run that uses it is recommended the same pair again).
    frame_rec = tuple(round(value * 8.0) / 8.0 for value in geometry.recommended_frame_m_s)
    paired = geometry
    if (abs(frame_rec[0] - frame.u_move_m_s) > 1e-9 or abs(frame_rec[1] - frame.v_move_m_s) > 1e-9):
        try:
            paired = evaluate_run_geometry(
                hodograph=hodograph, nx=grid.nx, ny=grid.ny, dx_m=grid.dx_m, dy_m=grid.dy_m,
                duration_s=config.run.duration_s, frame_m_s=frame_rec,
                bubble_center_xy_m=(centre[0] - origin[0], centre[1] - origin[1]),
                open_sides=sides,
            )
        except (TypeError, ValueError):
            paired = geometry
    paired_bubble = [
        round((paired.recommended_bubble_center_xy_m[0] + origin[0]) / 1000.0, 1) + 0.0,
        round((paired.recommended_bubble_center_xy_m[1] + origin[1]) / 1000.0, 1) + 0.0,
    ]
    try:
        # The track the pair itself predicts, for its clearance.
        with_pair = evaluate_run_geometry(
            hodograph=hodograph, nx=grid.nx, ny=grid.ny, dx_m=grid.dx_m, dy_m=grid.dy_m,
            duration_s=config.run.duration_s, frame_m_s=frame_rec,
            bubble_center_xy_m=(paired_bubble[0] * 1000.0 - origin[0], paired_bubble[1] * 1000.0 - origin[1]),
            open_sides=sides,
        )
    except (TypeError, ValueError):
        with_pair = paired
    uses_recommended = (
        abs(frame_rec[0] - frame.u_move_m_s) <= 1e-9 and abs(frame_rec[1] - frame.v_move_m_s) <= 1e-9
        and abs(paired_bubble[0] - centre[0] / 1000.0) <= 0.051 and abs(paired_bubble[1] - centre[1] / 1000.0) <= 0.051
    )
    profile = [
        [round(float(z), 1), round(float(u), 3), round(float(v), 3)]
        for z, u, v in zip(hodograph.heights_m, hodograph.u_m_s, hodograph.v_m_s)
    ]
    if len(profile) > 120:
        step = math.ceil(len(profile) / 120)
        profile = profile[::step] + ([profile[-1]] if (len(profile) - 1) % step else [])
    meets = right.stays_inside if minimum_is_infinite(minimum) else (minimum >= min_edge_clearance_m and minimum > 0.0)
    if admitted and not right.stays_inside:
        # On the file lanes the door refuses what loses its admission
        # clearance, so the fit's weaker 10 km rule must not admit it.
        meets = False
    if right.stays_inside:
        door_says = "admitted"
    else:
        door_says = "refused" if admitted else "reported (the analytic lane is not refused on the track)"
    return {
        **base,
        "status": "evaluated",
        "wind_source": wind_source["wind_source"],
        "stays_inside": right.stays_inside,
        "minimum_edge_clearance_km": None if minimum_is_infinite(minimum) else minimum / 1000.0,
        "clearance_at_end_km": None if minimum_is_infinite(right.clearance_at_end_m) else right.clearance_at_end_m / 1000.0,
        "exit_time_s": right.exit_time_s,
        "admission_clearance_km": geometry.admission_clearance_m / 1000.0,
        "meets_edge_rule": bool(meets),
        "door": door_says,
        "left_mover_leaves_at_s": left.exit_time_s,
        "steering_m_s": list(geometry.steering_m_s),
        "storm_motion": geometry.motion.record(),
        "recommended_frame_m_s": list(frame_rec),
        "recommended_frame_exact_m_s": list(geometry.recommended_frame_m_s),
        "recommended_pair": {
            "frame_m_s": list(frame_rec),
            "bubble_km": paired_bubble,
            "minimum_edge_clearance_km": (
                None if minimum_is_infinite(with_pair.members[MEMBER_RIGHT].minimum_clearance_m)
                else with_pair.members[MEMBER_RIGHT].minimum_clearance_m / 1000.0),
            "stays_inside": with_pair.members[MEMBER_RIGHT].stays_inside,
            "uses_it": bool(uses_recommended),
            "why": ("the moving frame (umove, vmove) follows the storm's predicted motion, so the storm stays "
                    "near the middle; the bubble is placed so the predicted track keeps the most room to every "
                    "open side"),
        },
        "hodograph_m_s": profile,
        "hodograph_frame": "ground-relative m/s at the run's own levels (height m, u, v)",
        "recommended_bubble_km": [
            round((geometry.recommended_bubble_center_xy_m[0] + origin[0]) / 1000.0, 1),
            round((geometry.recommended_bubble_center_xy_m[1] + origin[1]) / 1000.0, 1),
        ],
        "required_domain_km": [value / 1000.0 for value in geometry.required_domain_extent_m],
        "track_km": {
            "right": sampled(MEMBER_RIGHT),
            "left": sampled(MEMBER_LEFT),
            "mean": sampled(MEMBER_MEAN),
        },
        "model": (
            "the bubble drifts with the wind through its own depth for 30-45 min, then "
            "moves at the Bunkers et al. (2000) right-mover motion; the clearance is the "
            "worst over that band. Not CM1: a published method, stated"
        ),
    }


def minimum_is_infinite(value: float) -> bool:
    return value is None or (isinstance(value, float) and math.isinf(value))


# --------------------------------------------------------------- pricing


@dataclass(frozen=True)
class Quantity:
    """One number with its label and basis.

    ``label`` says what the number itself is (a prediction for a deck that
    has not run is ``estimated``); ``basis_label`` says what it rests on (the
    rate or the memory rows it is arithmetic on: ``measured``, ``documented``,
    ``estimated``, ``rough`` or ``measured-here``).
    """

    value: float
    label: str
    basis: str
    low: float | None = None
    high: float | None = None
    unit: str = ""
    basis_label: str | None = None

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "label": self.label, "basis": self.basis}
        if self.basis_label is not None:
            out["basis_label"] = self.basis_label
        if self.unit:
            out["unit"] = self.unit
        if self.low is not None:
            out["low"] = self.low
        if self.high is not None:
            out["high"] = self.high
        return out


@dataclass(frozen=True)
class Check:
    """One line the estimate shows under the numbers."""

    id: str
    level: str  # "ok" | "info" | "warn" | "stop"
    text: str

    def record(self) -> dict[str, str]:
        return {"id": self.id, "level": self.level, "text": self.text}


@dataclass(frozen=True)
class Estimate:
    """Everything ``gpuwm-cm1 estimate`` prints, as data."""

    deck: Deck
    config_sha256: str
    grid: Mapping[str, Any]
    cells: int
    steps: int
    dt_s: float
    duration_s: float
    history_interval_s: float
    frames: int
    restarts: int
    component: str
    acoustic_substeps: int
    tile: str
    gpu: GpuInfo | None
    host: HostInfo | None
    gpu_need: Quantity
    priced_peak_bytes: int
    host_locked: Quantity
    disk: Quantity
    disk_parts: Mapping[str, int]
    frame_bytes: int
    restart_bytes: int
    wall: Quantity
    wall_parts: Mapping[str, float]
    step: Mapping[str, Any]
    storm: Mapping[str, Any]
    recommendation: Mapping[str, Any]
    checks: tuple[Check, ...]
    command: str
    elapsed_s: float
    assumptions: tuple[str, ...] = ()

    # -- the answers the guided fit reads
    @property
    def gpu_need_bytes(self) -> int:
        return int(self.gpu_need.value)

    @property
    def gpu_free_bytes(self) -> int | None:
        return None if self.gpu is None else self.gpu.free_bytes

    @property
    def gpu_fits(self) -> bool | None:
        if self.gpu is None:
            return None
        return self.gpu_need.value <= FITS_FRACTION_OF_FREE * self.gpu.free_bytes

    @property
    def host_locked_bytes(self) -> int:
        return int(self.host_locked.value)

    @property
    def host_ram_bytes(self) -> int | None:
        return None if self.host is None else self.host.ram_total_bytes

    @property
    def host_fits(self) -> bool | None:
        if self.host_ram_bytes is None:
            return None
        return self.host_locked.value <= HOST_LOCKED_WARN_FRACTION * self.host_ram_bytes

    @property
    def disk_bytes(self) -> int:
        return int(self.disk.value)

    @property
    def disk_free_bytes(self) -> int | None:
        return None if self.host is None else self.host.disk_free_bytes

    @property
    def disk_fits(self) -> bool | None:
        if self.disk_free_bytes is None:
            return None
        return self.disk.value <= DISK_FITS_FRACTION * self.disk_free_bytes

    @property
    def wall_s(self) -> float:
        return float(self.wall.value)

    @property
    def min_edge_clearance_km(self) -> float | None:
        return self.storm.get("minimum_edge_clearance_km")

    @property
    def fits(self) -> bool | None:
        answers = [self.gpu_fits, self.host_fits, self.disk_fits]
        if any(answer is False for answer in answers):
            return False
        if any(check.level == "stop" for check in self.checks):
            return False
        if any(answer is None for answer in answers):
            return None
        return True

    def record(self) -> dict[str, Any]:
        return {
            "schema": ESTIMATE_SCHEMA,
            "target": {"source": self.deck.source, "kind": self.deck.kind, "name": self.deck.name},
            "config_sha256": self.config_sha256,
            "run": {
                "grid": dict(self.grid),
                "cells": self.cells,
                "dt_s": self.dt_s,
                "duration_s": self.duration_s,
                "steps": self.steps,
                "acoustic_substeps": self.acoustic_substeps,
                "history_interval_s": self.history_interval_s,
                "frames": self.frames,
                "restarts": self.restarts,
                "microphysics": self.component,
                "closure_tile": self.tile,
            },
            "gpu": None if self.gpu is None else self.gpu.record(),
            "host": None if self.host is None else self.host.record(),
            "needs": {
                "gpu_memory_bytes": self.gpu_need.record(),
                "priced_registry_peak_bytes": self.priced_peak_bytes,
                "host_locked_bytes": self.host_locked.record(),
                "disk_bytes": self.disk.record(),
                "disk_parts_bytes": dict(self.disk_parts),
                "frame_payload_bytes": self.frame_bytes,
                "restart_payload_bytes": self.restart_bytes,
            },
            "fits": {
                "gpu": self.gpu_fits,
                "host": self.host_fits,
                "disk": self.disk_fits,
                "all": self.fits,
                "rule": (
                    f"GPU need <= {FITS_FRACTION_OF_FREE} x free GPU memory; locked host memory "
                    f"<= {HOST_LOCKED_WARN_FRACTION:.0%} of RAM; disk <= {DISK_FITS_FRACTION} x free space"
                ),
            },
            "time": {
                "wall_s": self.wall.record(),
                "parts_s": dict(self.wall_parts),
                "step": dict(self.step),
            },
            "storm_track": dict(self.storm),
            "recommendation": dict(self.recommendation),
            "checks": [check.record() for check in self.checks],
            "assumptions": list(self.assumptions),
            "command": self.command,
            "elapsed_s": round(self.elapsed_s, 3),
        }

    def text(self) -> str:
        return render_text(self)


def _gb(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    if value >= 10e9:
        return f"{value / GB:.0f} GB"
    if value >= 1e9:
        return f"{value / GB:.1f} GB"
    return f"{value / GB:.2f} GB"


def duration_words(seconds: float) -> str:
    """Two significant figures, in the unit a person would use."""

    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min" if minutes >= 10 or abs(minutes - round(minutes)) < 0.05 else f"{minutes:.1f} min"
    hours = minutes / 60.0
    if abs(hours - round(hours)) < 0.05:
        return f"{round(hours):.0f} h"
    return f"{hours:.1f} h"


def _cells_words(cells: int) -> str:
    return f"{cells / 1e6:.2f} M cells" if cells < 10e6 else f"{cells / 1e6:.1f} M cells"


def _model_config(raw: Mapping[str, Any]) -> Any:
    from ..config import parse_mapping

    return parse_mapping(raw)


def _refusal_record(error: CM1Refusal) -> dict[str, Any]:
    return error.detail.as_dict()


def _price_tile(raw: Mapping[str, Any], tile: tuple[int, int] | None) -> tuple[Any, Any, dict[str, int]]:
    """(config, runtime config, peak rows) for one closure tile, budget lifted."""

    from .. import cartesian_door as door

    priced = copy.deepcopy(dict(raw))
    runtime = dict(priced.get("runtime", {}))
    runtime["memory_budget_mib"] = _PRICING_BUDGET_MIB
    if tile is None:
        runtime.pop("tile_nx", None)
        runtime.pop("tile_ny", None)
    else:
        runtime["tile_nx"], runtime["tile_ny"] = tile
    priced["runtime"] = runtime
    config = _model_config(priced)
    runtime_config = door.build_runtime_config(config)
    return config, runtime_config, door.closure_decomposition_peak_bytes(runtime_config)


def _device_need(priced_peak: int, cells: int, component: str) -> tuple[float, str]:
    model = _rates.load_record()["memory_model"]
    extra = float(model.get("component_extra_bytes", {}).get(component, 0.0))
    need = priced_peak + float(model["fixed_bytes"]) + float(model["per_cell_bytes"]) * cells + extra
    basis = (
        f"the door's priced registry peak {_gb(priced_peak)} + {_gb(model['fixed_bytes'])} "
        f"+ {model['per_cell_bytes']} B per cell"
        + (f" + {_gb(extra)} for {component}" if extra else "")
        + " (records/GM1-GPU-RATES.json memory_model: every measured run within 7.7 %)"
    )
    return need, basis


def device_need(priced_peak: int, cells: int, component: str) -> tuple[float, str]:
    """GPU memory one run needs on the door's priced peak, and its basis.

    The one memory model: ``price`` shows it and ``gpuwm-cm1 run`` admits a
    run against the card's free memory with it (cm1run.launch).
    """

    return _device_need(priced_peak, cells, component)


def _counts(config: Any) -> tuple[int, int, int]:
    """(steps, frames incl. t = 0, restarts) as gpuwm-cm1 run makes them."""

    run = config.run
    steps = int(round(run.duration_s / run.dt_s))
    frames = int(math.floor(run.duration_s / run.history_interval_s + 1e-9)) + 1
    interval = run.restart_interval_s
    if interval and interval > 0:
        restarts = max(1, int(math.floor(run.duration_s / interval + 1e-9)))
    else:
        restarts = 1
    return steps, frames, restarts


def price(
    deck: Deck | str | os.PathLike[str] | Mapping[str, Any],
    gpu: GpuInfo | None = None,
    *,
    host: HostInfo | None = None,
    cache: Sequence[_rates.MeasuredPoint] | None = None,
    sounding: str | os.PathLike[str] | None = None,
    command: str | None = None,
    envelope: bool = False,
) -> Estimate:
    """Price ``deck`` on ``gpu`` (``None``: no GPU known; times use the RTX 3080).

    ``host`` defaults to this machine with the disk of the deck's run folder
    (or the current folder).  ``cache`` defaults to this machine's measured
    rates.  ``envelope`` prices frames written with ``--history-envelope on``.
    Raises :class:`EstimateError` when the deck is refused before pricing.
    """

    from .. import cartesian_door as door
    from ..cartesian_restart import cartesian_restart_profile
    from ..cartesian_tornado_diagnostics import swath_restart_layout

    started = time.perf_counter()
    deck = load_deck(deck, sounding=sounding) if not isinstance(deck, Deck) else deck
    record = _rates.load_record()
    try:
        config = _model_config(deck.raw)
        door.require_cartesian_lane(config)
        door.require_level_capacity(config)
        door.require_history_cadence(config)
    except CM1Refusal as error:
        raise EstimateError(
            f"{deck.name} is refused before it can be priced: {error.detail.reason}. "
            f"What to change: {error.detail.remedy}",
            refusal=_refusal_record(error),
        ) from None
    except (TypeError, ValueError) as error:
        raise EstimateError(f"{deck.name} cannot be priced: {error}") from None
    runtime = config.runtime
    declared_tile = None if runtime is None or runtime.tile_nx is None else (runtime.tile_nx, runtime.tile_ny)
    grid = config.grid
    cells = grid.nx * grid.ny * grid.nz
    component = config.physics.component
    checks: list[Check] = []
    assumptions = [
        "Priced as `gpuwm-cm1 run` runs it: a frame at t = 0, "
        + ("two copies of each frame (--history-envelope on)" if envelope else "one copy of each frame (--history-envelope off)")
        + ", and a restart every rstfrq (at least one, at the end).",
    ]

    # --- memory, as written and with the recommended tile
    try:
        _, runtime_config, peak_rows = _price_tile(deck.raw, declared_tile)
    except CM1Refusal as error:
        raise EstimateError(
            f"{deck.name} is refused before it can be priced: {error.detail.reason}. "
            f"What to change: {error.detail.remedy}",
            refusal=_refusal_record(error),
        ) from None
    priced_peak = int(peak_rows["priced_peak_bytes"])
    if declared_tile is None:
        tile_words = f"whole domain ({grid.nx} x {grid.ny})"
    else:
        count = _rates.closure_tile_count(grid.nx, grid.ny, declared_tile)
        tile_words = f"{declared_tile[0]} x {declared_tile[1]} ({count} tile{'s' if count != 1 else ''})"
    need, need_basis = _device_need(priced_peak, cells, component)
    band = float(record["memory_model"]["band"])
    gpu_need = Quantity(
        need, "estimated", need_basis, low=need * (1 - band), high=need * (1 + band), unit="bytes",
        basis_label="measured",
    )

    # The step model is needed here as well as below: whether tile 128 is
    # worth recommending depends on what it costs in step time.
    gpu_name = None if gpu is None else gpu.name
    model = _rates.step_model(gpu_name or "NVIDIA GeForce RTX 3080", component, cache=cache)
    tiles = _rates.closure_tile_count(grid.nx, grid.ny, declared_tile)

    recommendation: dict[str, Any] = {}
    tiled_need = None
    if declared_tile is None and max(grid.nx, grid.ny) > RECOMMENDED_TILE:
        tiled_xy = (RECOMMENDED_TILE, RECOMMENDED_TILE)
        _, _, tiled_rows = _price_tile(deck.raw, tiled_xy)
        tiled_peak = int(tiled_rows["priced_peak_bytes"])
        tiled_need, _ = _device_need(tiled_peak, cells, component)
        tiled_tiles = _rates.closure_tile_count(grid.nx, grid.ny, tiled_xy)
        whole_step = model.seconds_per_step(cells, 1)
        tiled_step = model.seconds_per_step(cells, tiled_tiles)
        usable = None if gpu is None else FITS_FRACTION_OF_FREE * gpu.free_bytes
        whole_fits = usable is None or need <= usable
        tiled_fits = usable is None or tiled_need <= usable
        option = {
            "tile_nx": RECOMMENDED_TILE,
            "tile_ny": RECOMMENDED_TILE,
            "closure_tiles": tiled_tiles,
            "gpu_memory_bytes": tiled_need,
            "priced_registry_peak_bytes": tiled_peak,
            "memory_budget_mib": int(math.ceil(max(tiled_peak * 1.10, 1024 * MIB) / MIB)),
            "seconds_per_step": tiled_step,
            "whole_domain_seconds_per_step": whole_step,
        }
        why = None
        if tiled_fits and not whole_fits:
            why = (
                f"the whole domain as one closure tile needs about {_gb(need)} of GPU memory, more than your "
                f"GPU can give it; {tiled_tiles} tiles of {RECOMMENDED_TILE} need about {_gb(tiled_need)}"
            )
        elif tiled_fits and tiled_need < need and tiled_step <= TILE_SPEED_TOLERANCE * whole_step:
            why = (
                f"it needs about {_gb(need - tiled_need)} less GPU memory at about the same estimated step "
                f"time ({tiled_step:.3f} s against {whole_step:.3f} s)"
            )
        if why is not None:
            recommendation = {**option, "why": why, "rule": record["tile_rule"]["rule"]}
            surface = getattr(config.closures, "sfcmodel", 0)
            if surface and int(surface) >= 1 and int(getattr(config.closures, "cecd", 0)) in (2, 3):
                recommendation["caveat"] = record["tile_rule"]["exception"]
        else:
            recommendation = {"tile_128_option": {
                **option,
                "why_not": (
                    "the whole domain fits and tile 128 would cost more step time"
                    if tiled_fits else "tile 128 does not fit either"
                ),
            }}
    budget_mib = None if runtime is None else runtime.memory_budget_mib
    wanted_mib = int(math.ceil(max(priced_peak * 1.10, 1024 * MIB) / MIB))
    if budget_mib is not None and budget_mib * MIB < priced_peak:
        checks.append(Check(
            "memory-budget", "stop",
            (
                f"runtime.memory_budget_mib = {budget_mib} is below this run's priced peak of "
                f"{priced_peak / MIB:.0f} MiB with {tile_words}, so the run would stop at its first "
                f"step. Raise it to {wanted_mib}"
                + (f", or set runtime.tile_nx = tile_ny = {RECOMMENDED_TILE} ({_gb(recommendation['gpu_memory_bytes'])} in all)" if recommendation.get("tile_nx") else "")
                + "."
            ),
        ))
    recommendation.setdefault("memory_budget_mib", wanted_mib)

    # --- host memory and disk from the plan's own payloads
    try:
        frame_bytes = int(door.cartesian_history_plan(runtime_config)["payload_bytes"])
    except (CM1Refusal, ValueError) as error:
        # A frame past the history envelope's cap: priced at the scoping's
        # 84.5 B per cell, and the run stopped by name below.
        frame_bytes = int(84.5 * cells)
        text = error.detail.reason if isinstance(error, CM1Refusal) else str(error)
        checks.append(Check("frame-size", "stop", f"This grid cannot publish a frame: {text}."))
    try:
        # The swaths the run keeps travel in its restarts (restart_write.F:696-758),
        # the ground canvas sized for the run's whole track.
        restart_bytes = int(cartesian_restart_profile(
            runtime_config,
            swaths=swath_restart_layout(runtime_config, total_steps=_counts(config)[0]),
        ).payload_nbytes)
    except ValueError as error:
        # Past the CM1FCP1 cap there is no restart, so neither Continue nor a
        # clean pause exists; priced at 280 B per cell and stopped by name.
        restart_bytes = 280 * cells
        checks.append(Check("restart-size", "stop", f"This grid cannot write a restart: {error}."))
    steps, frames, restarts = _counts(config)
    locked = frame_bytes + restart_bytes
    host_locked = Quantity(
        float(locked), "measured",
        f"the two page-locked publication arenas: a frame ({_gb(frame_bytes)}) and a restart ({_gb(restart_bytes)}), the plan's own payloads",
        unit="bytes",
    )
    disk_record = record["disk"]
    raw_frames = frames * frame_bytes * (2 if envelope else 1)
    netcdf = frames * frame_bytes * float(disk_record["netcdf_over_raw"])
    restart_disk = restarts * restart_bytes
    quicklooks = frames * int(disk_record["quicklook_bytes_per_frame"])
    disk_parts = {
        "frames": int(raw_frames), "netcdf": int(netcdf), "restarts": int(restart_disk),
        "quicklooks": int(quicklooks),
    }
    disk_total = float(sum(disk_parts.values()))
    disk = Quantity(
        disk_total, "estimated",
        (
            f"{frames} frames of {_gb(frame_bytes)}, their cm1out netCDF at {disk_record['netcdf_over_raw']} x "
            f"(estimated), {restarts} restart(s) of {_gb(restart_bytes)} and quick-looks"
        ),
        unit="bytes",
    )

    # --- time
    if gpu is None:
        assumptions.append("No GPU was found, so the times are for the RTX 3080 the rates were measured on.")
    step_s = model.seconds_per_step(cells, tiles)
    step_label = model.label_for(cells, tiles)
    step_band = _rates.band_for_label(step_label)
    startup_model = record["startup"]
    io = record["io"]
    startup = float(startup_model["fixed_s"]) + float(startup_model["per_cell_s"]) * cells
    stepping = steps * step_s
    writes = (
        frames * (frame_bytes / (float(io["snapshot_gb_s"]) * GB) + float(io["field_scan_s"]))
        + max(0, restarts - 1) * restart_bytes / (float(io["snapshot_gb_s"]) * GB)
        + (frame_bytes + restart_bytes) / (float(io["final_write_gb_s"]) * GB)
    )
    wall_s = startup + stepping + writes
    wall = Quantity(
        wall_s, "estimated",
        (
            f"start-up {startup:.0f} s + {steps} steps x {step_s:.4f} s ({model.regime(cells, tiles)} regime: "
            f"{model.basis_for(cells, tiles)}) + writes {writes:.0f} s"
        ),
        low=wall_s * (1 - step_band), high=wall_s * (1 + step_band), unit="s",
        basis_label=step_label,
    )
    step_record = model.record(cells, tiles)

    # --- storm track
    storm = storm_track(config, sounding_bytes=deck.sounding_bytes)

    # --- checks
    if host is None:
        root = deck.rundir.parent if deck.rundir is not None else Path.cwd()
        host = query_host(root)
    _add_checks(
        checks, config=config, runtime_config=runtime_config, gpu=gpu, host=host, cells=cells,
        gpu_need=need, tiled_need=tiled_need, host_locked=locked, disk=disk_total,
        storm=storm, component=component, record=record, grid=grid,
        priced_per_cell=(priced_peak / cells if cells else 0.0),
    )
    target = deck.source
    cmd = command or "gpuwm-cm1 estimate " + shlex.quote(str(target)) + (
        "" if gpu is None else f" --gpu {gpu.index}"
    )
    runtime_record = runtime_config.record()
    return Estimate(
        deck=deck,
        config_sha256=config.sha256,
        grid={
            "nx": grid.nx, "ny": grid.ny, "nz": grid.nz, "dx_m": grid.dx_m, "dy_m": grid.dy_m,
            "dz_m": grid.dz_m, "stretch_z": grid.stretch_z,
            "domain_km": [grid.nx * grid.dx_m / 1000.0, grid.ny * grid.dy_m / 1000.0],
        },
        cells=cells,
        steps=steps,
        dt_s=config.run.dt_s,
        duration_s=config.run.duration_s,
        history_interval_s=config.run.history_interval_s,
        frames=frames,
        restarts=restarts,
        component=component,
        acoustic_substeps=int(runtime_record.get("acoustic_substeps", 0) or 0),
        tile=tile_words,
        gpu=gpu,
        host=host,
        gpu_need=gpu_need,
        priced_peak_bytes=priced_peak,
        host_locked=host_locked,
        disk=disk,
        disk_parts=disk_parts,
        frame_bytes=frame_bytes,
        restart_bytes=restart_bytes,
        wall=wall,
        wall_parts={"startup": startup, "stepping": stepping, "writes": writes},
        step=step_record,
        storm=storm,
        recommendation=recommendation,
        checks=tuple(checks),
        command=cmd,
        elapsed_s=time.perf_counter() - started,
        assumptions=tuple(assumptions),
    )


def largest_square_domain(
    *, free_bytes: float, nz: int, dx_m: float, component: str, priced_per_cell: float,
) -> dict[str, Any]:
    """The widest square box at ``dx_m`` and ``nz`` whose need fits 0.95 of ``free_bytes``.

    Uses the priced per-cell registry cost of the deck in hand (tile 128 for
    large domains, about 400 B per cell) and the memory model's residual.
    """

    model = _rates.load_record()["memory_model"]
    extra = float(model.get("component_extra_bytes", {}).get(component, 0.0))
    usable = FITS_FRACTION_OF_FREE * free_bytes - float(model["fixed_bytes"]) - extra
    per_cell = priced_per_cell + float(model["per_cell_bytes"])
    if usable <= 0 or per_cell <= 0:
        return {"cells": 0, "n": 0, "width_km": 0.0}
    cells = usable / per_cell
    n = int(math.floor(math.sqrt(cells / nz) / 8.0) * 8)
    return {"cells": n * n * nz, "n": n, "width_km": n * dx_m / 1000.0}


def _add_checks(
    checks: list[Check], *, config: Any, runtime_config: Any, gpu: GpuInfo | None, host: HostInfo,
    cells: int, gpu_need: float, tiled_need: float | None, host_locked: float, disk: float,
    storm: Mapping[str, Any], component: str, record: Mapping[str, Any], grid: Any,
    priced_per_cell: float,
) -> None:
    runtime = runtime_config.record()
    # GPU memory
    if gpu is None:
        checks.append(Check("gpu", "info", "No NVIDIA GPU was found here, so whether it fits is not known."))
    else:
        usable = FITS_FRACTION_OF_FREE * gpu.free_bytes
        if gpu_need <= usable:
            checks.append(Check(
                "gpu-memory", "ok",
                f"Needs about {_gb(gpu_need)} of GPU memory; your GPU has {_gb(gpu.free_bytes)} free.",
            ))
        else:
            text = (
                f"Needs about {_gb(gpu_need)} of GPU memory, but your GPU has only {_gb(gpu.free_bytes)} free "
                f"(it may use {FITS_FRACTION_OF_FREE:.0%} of that)."
            )
            if tiled_need is not None and tiled_need <= usable:
                text += (
                    f" With runtime.tile_nx = tile_ny = {RECOMMENDED_TILE} it needs {_gb(tiled_need)}, which fits."
                )
            elif gpu.wsl2 and not gpu.compute_apps:
                text += " Closing programs that draw on the Windows desktop frees some."
            checks.append(Check("gpu-memory", "stop", text))
        if gpu.wsl2 and not gpu.compute_apps and gpu.in_use_bytes > 1.0e9:
            checks.append(Check(
                "gpu-shared", "info",
                f"{_gb(gpu.in_use_bytes)} of your GPU is in use by other programs (under WSL2 this includes "
                "the Windows desktop, whose processes nvidia-smi cannot list).",
            ))
        others = [app for app in gpu.compute_apps]
        if others:
            used = sum(app.used_bytes or 0 for app in others)
            checks.append(Check(
                "gpu-busy", "warn",
                f"{len(others)} other program(s) are using this GPU ({_gb(used)}): "
                + ", ".join(f"{app.name} (pid {app.pid})" for app in others[:4])
                + ". A second GM1 run waits for the first.",
            ))
        sm = gpu.sm
        if sm and sm not in record["architectures_run"]:
            checks.append(Check(
                "architecture", "warn",
                f"GM1 has never run on this GPU architecture ({sm}, {gpu.name}); it has run on "
                + ", ".join(sorted(record["architectures_run"])) + ".",
            ))
    # host memory
    if host.ram_total_bytes:
        share = host_locked / host.ram_total_bytes
        level = "ok" if share <= HOST_LOCKED_WARN_FRACTION else "warn"
        checks.append(Check(
            "host-memory", level,
            f"Locks {_gb(host_locked)} of host memory for writing frames and restarts "
            f"({share:.0%} of {_gb(host.ram_total_bytes)} RAM)"
            + ("." if level == "ok" else "; above half of RAM the rest of the machine slows down."),
        ))
    # disk
    if host.disk_free_bytes:
        share = disk / host.disk_free_bytes
        if share > DISK_FITS_FRACTION:
            level = "stop"
        elif share > DISK_WARN_FRACTION:
            level = "warn"
        else:
            level = "ok"
        checks.append(Check(
            "disk", level,
            f"Writes about {_gb(disk)}; {_gb(host.disk_free_bytes)} is free under {host.disk_root} "
            f"({share:.0%})."
            + ("" if level == "ok" else " Raise the output interval (tapfrq) or pick a folder with more space."),
        ))
    # time step
    small = runtime.get("acoustic_small_step_s")
    limit = runtime.get("acoustic_small_step_limit_s")
    stop_speed = runtime.get("advective_courant_stop_speed_m_s")
    substeps = runtime.get("acoustic_substeps")
    if substeps and small and limit:
        checks.append(Check(
            "time-step", "info",
            f"Time step {config.run.dt_s:g} s with {substeps} sound-wave substeps of {small:.3g} s "
            f"(CM1's dtsmall limit {limit:.3g} s, nsound derived as CM1 does).",
        ))
    if stop_speed:
        # gm1_runtime.cm1_courant_limited_speed_m_s: 1.5 x min(dx, dy, dz) / dt,
        # the slowest wind that can reach CM1's stop, pointed along the
        # finest spacing (the vertical one on most storm decks).
        level = "warn" if stop_speed < 75.0 else "info"
        checks.append(Check(
            "courant", level,
            f"No wind slower than {stop_speed:.0f} m/s can reach CM1's Courant stop (1.5) at this time step "
            "and spacing"
            + (": strong storms reach 60-70 m/s updrafts, so a shorter time step is safer." if level == "warn" else "."),
        ))
    # a moving domain's position: the run checks CM1's running sum of it exactly
    if not getattr(runtime_config, "adaptive_timestep", False) and getattr(config, "nest", None) is None:
        steps = int(round(config.run.duration_s / config.run.dt_s)) if config.run.dt_s > 0 else 0
        stops = [
            first_inexact_step(config.run.dt_s, speed, steps)
            for speed in (getattr(runtime_config, "u_move_m_s", 0.0), getattr(runtime_config, "v_move_m_s", 0.0))
        ]
        found = [step for step in stops if step]
        if found:
            better = ""
            try:
                from .customize import snap_time_step

                step = snap_time_step(config.run.dt_s, [config.run.duration_s, config.run.history_interval_s],
                                      binary=True)
                better = f" {step!r} s works here."
            except ValueError:
                pass
            sure = [step for step in found if step > 0]
            if sure:
                checks.append(Check(
                    "moving-step", "stop",
                    f"With a moving domain (umove, vmove) a {config.run.dt_s:g} s time step stops the run after "
                    f"{min(sure)} steps: CM1's running sum of the domain's position rounds there, and the run "
                    "checks that sum exactly. A step that is a small odd number over a power of two (0.5 s, "
                    f"0.375 s, 0.75 s) never rounds.{better}",
                ))
            else:
                checks.append(Check(
                    "moving-step", "warn",
                    f"With a moving domain (umove, vmove) a {config.run.dt_s:g} s time step may stop a run this "
                    "long: CM1's running sum of the domain's position can round, and the run checks that sum "
                    f"exactly. A step that is a small odd number over a power of two never rounds.{better}",
                ))
    bounds = (8, 80) if component == "wsm6-mp6" else None
    if bounds and grid.nz > bounds[1]:
        checks.append(Check("levels", "stop", f"WSM6 runs at most {bounds[1]} levels; this deck has {grid.nz}."))
    # storm track
    status = storm.get("status")
    if status == "evaluated":
        clearance = storm.get("minimum_edge_clearance_km")
        if storm.get("stays_inside"):
            checks.append(Check(
                "storm-track", "ok",
                "The storm stays inside the area: it comes no closer than "
                + (f"{clearance:.0f} km" if clearance is not None else "any distance")
                + f" to an open side (the run needs {storm['admission_clearance_km']:.0f} km).",
            ))
        else:
            exit_s = storm.get("exit_time_s") or 0.0
            refused = storm.get("door") == "refused"
            checks.append(Check(
                "storm-track", "stop" if refused else "warn",
                # In an area too small for the storm it is short of room at once: said as that, not "at about 0 s".
                (f"The storm reaches the edge of the area at about {duration_words(exit_s)} " if exit_s >= 30.0
                 else "The storm is too close to the edge of the area from the start ")
                + f"(it comes within {clearance:.0f} km of an open side; it needs "
                f"{storm['admission_clearance_km']:.0f} km). After that, the biggest values are "
                "edge effects, not the storm. "
                f"Recommended: frame {storm['recommended_frame_m_s'][0]:.1f}, {storm['recommended_frame_m_s'][1]:.1f} m/s "
                f"(umove, vmove) and bubble at {storm['recommended_bubble_km'][0]:.0f}, "
                f"{storm['recommended_bubble_km'][1]:.0f} km, or an area of "
                f"{storm['required_domain_km'][0]:.0f} x {storm['required_domain_km'][1]:.0f} km."
                + ("" if refused else " (Reported, not refused, on the analytic sounding.)"),
            ))
    elif status:
        checks.append(Check("storm-track", "info", f"Storm track not checked: {storm.get('reason', status)}"))
    # the largest box, at 50 m or when it does not fit
    largest_cells = int(record["largest_domain_cells"]["value"])
    if cells > largest_cells:
        checks.append(Check(
            "size", "warn",
            f"{_cells_words(cells)} is larger than any domain GM1 has run ({_cells_words(largest_cells)}).",
        ))
    if gpu is not None and (grid.dx_m <= 50.0 or gpu_need > FITS_FRACTION_OF_FREE * gpu.free_bytes):
        box = largest_square_domain(
            free_bytes=gpu.free_bytes, nz=grid.nz, dx_m=grid.dx_m, component=component,
            priced_per_cell=min(priced_per_cell, 420.0) if priced_per_cell else 420.0,
        )
        if box["n"] > 0:
            checks.append(Check(
                "largest-box", "info",
                f"At {grid.dx_m:g} m and {grid.nz} levels, the largest square area that fits your GPU's free "
                f"memory is about {box['width_km']:.0f} km ({box['n']} x {box['n']} x {grid.nz}, tile {RECOMMENDED_TILE}).",
            ))
    if getattr(config, "nest", None) is not None:
        checks.append(Check("nest", "warn", "Nests are not priced by this estimator: only the parent domain is counted."))


def first_inexact_step(dt_s: float, speed_m_s: float, steps: int, *, look: int = 200_000) -> int | None:
    """The first step at which a moving domain's running sum of ``dt * speed`` stops being ``n * dt * speed``.

    CM1 adds ``dbldt*dble(umove)`` to the domain's position once a step (cm1.F:3512-3513) and the run checks the
    sum against the product exactly (``gm1_runtime.RuntimeClock``).  The two are equal at every step while no
    addition rounds: the increment's significant bits and the step count's together fit binary64's 53, which is
    so for a step that is a small odd number over a power of two and a frame speed of a few binary digits.
    Returns None when they stay equal for all ``steps``; the step that first differs; or -1 when the first
    ``look`` steps agree and the bits do not rule out a later one.
    """

    from ..gm1_runtime import cm1_real

    increment = float(dt_s) * cm1_real(float(speed_m_s))
    if increment == 0.0 or steps <= 0 or not math.isfinite(increment):
        return None
    mantissa = int(abs(math.frexp(increment)[0]) * (1 << 53))
    while mantissa and mantissa % 2 == 0:
        mantissa //= 2
    if mantissa.bit_length() + int(steps).bit_length() <= 53:
        return None
    total = 0.0
    for step in range(1, min(steps, look) + 1):
        total += increment
        if total != step * increment:
            return step
    return None if steps <= look else -1


# ------------------------------------------------------------ plain text


def render_text(estimate: Estimate) -> str:
    """The advanced-mode printout: numbers, basis, checks, command."""

    e = estimate
    g = e.grid
    lines = []
    where = e.gpu.name if e.gpu is not None else "no GPU found (times for an RTX 3080)"
    lines.append(f"Estimate for {e.deck.name} on {where}")
    lines.append(
        f"  Grid {g['nx']} x {g['ny']} x {g['nz']} at {g['dx_m']:g} m ({g['domain_km'][0]:.0f} x "
        f"{g['domain_km'][1]:.0f} km, {_cells_words(e.cells)}); {duration_words(e.duration_s)} of storm time in "
        f"{e.steps} steps of {e.dt_s:g} s; {e.component}; closure tile: {e.tile}."
    )
    fits = {True: "fits", False: "does not fit", None: "not known"}
    gpu_free = "unknown" if e.gpu is None else _gb(e.gpu.free_bytes)
    lines.append(
        f"  GPU memory  about {_gb(e.gpu_need.value)} ({_gb(e.gpu_need.low)}-{_gb(e.gpu_need.high)}, estimated); "
        f"free {gpu_free}: {fits[e.gpu_fits]}"
    )
    lines.append(f"              basis: {e.gpu_need.basis}")
    ram = "unknown" if e.host is None or e.host.ram_total_bytes is None else _gb(e.host.ram_total_bytes)
    lines.append(f"  Host memory {_gb(e.host_locked.value)} locked of {ram} RAM: {fits[e.host_fits]}")
    parts = e.disk_parts
    free_disk = "unknown" if e.disk_free_bytes is None else _gb(e.disk_free_bytes)
    lines.append(
        f"  Disk        about {_gb(e.disk.value)} (frames {_gb(parts['frames'])}, netCDF {_gb(parts['netcdf'])}, "
        f"restarts {_gb(parts['restarts'])}); free {free_disk}: {fits[e.disk_fits]}"
    )
    rate_words = {
        "measured": "estimated from a measured rate",
        "measured-here": "estimated from a rate measured on this GPU",
        "documented": "estimated from a documented rate",
        "estimated": "estimated from an estimated rate",
        "rough": "rough: no GM1 run on this card",
    }
    label = rate_words.get(e.wall.basis_label or "", "estimated")
    lines.append(
        f"  Wall time   about {duration_words(e.wall.value)} ({duration_words(e.wall.low)}-{duration_words(e.wall.high)}; {label})"
    )
    lines.append(f"              {e.wall.basis}")
    storm = e.storm
    if storm.get("status") == "evaluated":
        lines.append(
            f"  Storm track evaluated ({storm['wind_source']}): closest approach to an open side "
            f"{storm['minimum_edge_clearance_km']:.0f} km"
            + ("" if storm.get("stays_inside") else f", leaves its clearance at {duration_words(storm.get('exit_time_s') or 0)}")
        )
    else:
        lines.append(f"  Storm track {storm.get('status')}: {storm.get('reason', '')}")
    if e.recommendation.get("tile_nx"):
        lines.append(
            f"  Recommended runtime.tile_nx = tile_ny = {e.recommendation['tile_nx']}: {e.recommendation['why']}."
        )
    for check in e.checks:
        if check.id == "storm-track" and check.level == "info":
            continue  # the storm-track line above already says it
        if check.level in ("warn", "stop", "info"):
            mark = {"warn": "Warning:", "stop": "Stop:", "info": "-"}[check.level]
            lines.append(f"  {mark} {check.text}")
    for assumption in e.assumptions:
        lines.append(f"  - {assumption}")
    lines.append(f"  Command: {e.command}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------- measure speed


def measure_speed(
    deck: Deck,
    gpu: GpuInfo | None,
    *,
    steps: int = 60,
    library: str | os.PathLike[str] | None = None,
    lock: str | os.PathLike[str] | None = None,
    workdir: str | os.PathLike[str] | None = None,
    timeout_s: float = 1800.0,
    cache_file: Path | None = None,
    python: str | None = None,
) -> tuple[_rates.MeasuredPoint, Path]:
    """Run ``steps`` large steps of ``deck`` on ``gpu`` and keep the measured rate.

    The deck runs through ``gpuwm-cm1 cartesian``'s timing harness (no frames,
    no restarts) at its own grid, physics and tile, with the estimate's
    recommended tile when it declares none and one is recommended (the
    tiling the run itself should take).  The receipt's mean step time
    goes into this machine's rate cache, so the next estimate uses it.
    """

    if steps < MEASURE_MIN_STEPS:
        raise ValueError(
            f"measure at least {MEASURE_MIN_STEPS} steps ({_rates.MIN_LEARN_STEPS} are timed after the warm-up)"
        )
    raw = copy.deepcopy(dict(deck.raw))
    run = dict(raw["run"])
    dt = float(run["dt_s"])
    run["duration_s"] = dt * steps
    run["history_interval_s"] = dt * steps
    run["restart_interval_s"] = dt * steps
    raw["run"] = run
    runtime = dict(raw.get("runtime", {}))
    runtime["warmup_steps"] = min(3, steps // 5)
    if "tile_nx" not in runtime:
        # Measure the tiling the run would take: the whole domain, or tile
        # 128 where the estimate recommends it (memory, or the same speed in
        # less memory).
        advice = price(deck, gpu, host=HostInfo(None, None, ".", None, None), cache=()).recommendation
        if advice.get("tile_nx"):
            runtime["tile_nx"], runtime["tile_ny"] = int(advice["tile_nx"]), int(advice["tile_ny"])
    # The registry budget the measurement needs at the tile it runs: the
    # priced peak with the 10 % the fit leaves, never below the deck's own.
    # A deck whose budget is below its own peak (the importer writes 8192 MiB
    # whatever the grid) is raised to it, since the budget selects no
    # arithmetic; it is capped by what the GPU has free.
    tile = None if "tile_nx" not in runtime else (int(runtime["tile_nx"]), int(runtime["tile_ny"]))
    _, _, peak_rows = _price_tile({**raw, "runtime": runtime}, tile)
    peak = int(peak_rows["priced_peak_bytes"])
    needed_mib = int(math.ceil(max(peak * 1.10, 1024 * MIB) / MIB))
    budget_mib = max(int(runtime.get("memory_budget_mib", 0) or 0), needed_mib)
    if gpu is not None:
        usable_mib = int(FITS_FRACTION_OF_FREE * gpu.free_bytes / MIB)
        if needed_mib > usable_mib:
            raise EstimateError(
                f"this deck needs about {_gb(needed_mib * MIB)} of GPU memory to measure, and your GPU "
                f"has {_gb(gpu.free_bytes)} free"
            )
        budget_mib = min(budget_mib, usable_mib)
    runtime["memory_budget_mib"] = budget_mib
    raw["runtime"] = runtime
    if workdir:
        base = Path(workdir)
    else:
        root = _rates.cache_dir() / "measure"
        root.mkdir(parents=True, exist_ok=True)
        _prune_measurements(root, keep=MEASUREMENTS_KEPT - 1)
        base = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%dT%H%M%SZ-", time.gmtime()), dir=root))
    base.mkdir(parents=True, exist_ok=True)
    deck_path = base / "measure.toml"
    deck_path.write_text(dump_toml(raw), encoding="utf-8")
    receipt = base / "receipt.json"
    command = [python or sys.executable, "-m", "cm1core", "cartesian", str(deck_path), "--output", str(receipt)]
    if deck.sounding_bytes and int(raw.get("initialization", {}).get("isnd", 7)) in (7, 17):
        sounding_path = base / "input_sounding"
        sounding_path.write_bytes(deck.sounding_bytes)
        command += ["--sounding", str(sounding_path)]
    if library:
        command += ["--library", str(library)]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
        if lock:
            command += ["--gpu-uuid", gpu.uuid, "--lock", str(lock)]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_s, env=env, check=False,
    )
    (base / "measure.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not receipt.is_file():
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
        raise EstimateError(
            f"the speed measurement stopped (exit {completed.returncode}): " + " | ".join(tail)
        )
    point, _ = _rates.learn_receipt(
        receipt, source="measure", gpu_uuid=None if gpu is None else gpu.uuid, path=cache_file,
    )
    return point, receipt


#: How many "Measure speed" folders (deck, receipt, log) the cache keeps.
MEASUREMENTS_KEPT = 5
#: The fewest steps "Measure speed" runs: the warm-up (3) plus the fewest a
#: receipt must time to be learned (``rates.MIN_LEARN_STEPS``).
MEASURE_MIN_STEPS = 25


def _prune_measurements(root: Path, *, keep: int) -> None:
    import shutil

    folders = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    for old in folders[: max(0, len(folders) - keep)]:
        shutil.rmtree(old, ignore_errors=True)


def dump_toml(raw: Mapping[str, Any]) -> str:
    """A ``gpuwm-cm1.config/v1`` mapping as TOML text (tables of scalars and lists)."""

    from ..cm1import.translate import toml_scalar

    lines: list[str] = []
    scalars = {k: v for k, v in raw.items() if not isinstance(v, Mapping)}
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value, toml_scalar)}")
    for table, body in raw.items():
        if not isinstance(body, Mapping):
            continue
        lines.append("")
        lines.append(f"[{table}]")
        nested = []
        for key, value in body.items():
            if isinstance(value, Mapping):
                nested.append((key, value))
                continue
            lines.append(f"{key} = {_toml_value(value, toml_scalar)}")
        for key, value in nested:
            lines.append("")
            lines.append(f"[{table}.{key}]")
            for inner, item in value.items():
                lines.append(f"{inner} = {_toml_value(item, toml_scalar)}")
    return "\n".join(lines) + "\n"


def _toml_value(value: Any, scalar: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item, scalar) for item in value) + "]"
    return scalar(value)


# ------------------------------------------------------------- the door


def learn_from_rundir(rundir: Path, *, cache_file: Path | None = None) -> list[_rates.MeasuredPoint]:
    """Every finished segment's receipt in a run folder, into the rate cache."""

    learned = []
    for receipt in sorted(rundir.glob("segments/*/receipt.json")):
        try:
            point, added = _rates.learn_receipt(receipt, source="run", path=cache_file)
        except (OSError, ValueError, _rates.ReceiptError):
            continue
        if added:
            learned.append(point)
    return learned


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
            handle.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def estimate_door(arguments: Any) -> int:
    """``gpuwm-cm1 estimate``: exit 0 priced, 2 refused or unreadable."""

    if getattr(arguments, "each", None):
        return _estimate_each(arguments)
    learned: list[dict[str, Any]] = []
    for receipt in arguments.learn or ():
        try:
            point, added = _rates.learn_receipt(receipt, source="run")
        except (OSError, ValueError) as error:  # ReceiptError, unreadable JSON, a missing file
            return _fail(arguments, f"{receipt} was not learned: {error}", None)
        learned.append({**point.record(), "added": added})
    if arguments.target is None:
        if not learned:
            print("gpuwm-cm1 estimate: give a deck, a namelist or a run folder (or --learn RECEIPT)", file=sys.stderr)
            return 2
        payload = {"schema": "gpuwm-cm1.rates-learned/v1", "learned": learned, "cache": str(_rates.cache_path())}
        print(json.dumps(payload, indent=2, sort_keys=True) if arguments.json else "\n".join(
            f"Learned {row['seconds_per_step']:.4f} s/step at {row['cells'] / 1e6:.2f} M cells on {row['gpu_name']}"
            + ("" if row["added"] else " (already known)") for row in learned
        ))
        return 0
    try:
        deck = load_deck(arguments.target, sounding=arguments.sounding)
    except EstimateError as error:
        return _fail(arguments, str(error), getattr(error, "refusal", None))
    except (CM1Refusal,) as error:
        return _fail(arguments, str(error), error.detail.as_dict())
    learned += _learn_rundir(deck)
    query: GpuQuery = query_gpus()
    try:
        gpu = query.select(arguments.gpu)
    except ValueError as error:
        return _fail(arguments, str(error), None)
    host = _host_for(arguments, deck)
    measured = None
    if arguments.measure:
        if gpu is None:
            return _fail(arguments, "Measure speed needs an NVIDIA GPU, and none was found.", None)
        try:
            point, receipt = measure_speed(
                deck, gpu, steps=arguments.measure_steps, library=arguments.library, lock=arguments.lock,
            )
        except (EstimateError, _rates.ReceiptError, subprocess.TimeoutExpired, OSError, ValueError) as error:
            return _fail(arguments, f"Measure speed failed: {error}", None)
        measured = {**point.record(), "receipt": str(receipt)}
    try:
        estimate, payload = _priced_payload(arguments, deck, query, gpu, host, learned, measured)
    except EstimateError as error:
        return _fail(arguments, str(error), error.refusal)
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        text = estimate.text()
        if query.status != "ok":
            text = f"  ({query.message})\n" + text
        if measured:
            text += (
                f"  Measured {measured['seconds_per_step']:.4f} s/step over {measured['steps']} steps; "
                "later estimates on this GPU use it.\n"
            )
        sys.stdout.write(text)
    return 0


def _learn_rundir(deck: Deck) -> list[dict[str, Any]]:
    """A run folder's finished segments, learned into this machine's rates before it is priced."""

    if deck.rundir is None:
        return []
    return [{**point.record(), "added": True} for point in learn_from_rundir(deck.rundir)]


def _host_for(arguments: Any, deck: Deck) -> HostInfo:
    return query_host(arguments.root or (deck.rundir.parent if deck.rundir is not None else Path.cwd()))


def _priced_payload(arguments: Any, deck: Deck, query: GpuQuery, gpu: GpuInfo | None, host: HostInfo,
                    learned: list[dict[str, Any]], measured: Mapping[str, Any] | None) -> tuple[Estimate, dict[str, Any]]:
    """Price one loaded deck and write its run folder's estimate.json/geometry.json (unless --no-write).

    Shared by one target and by ``--each``, so a target priced in a batch gets the words
    ``gpuwm-cm1 estimate TARGET ...`` prints for it alone, against the same GPU answer.
    """

    command = "gpuwm-cm1 estimate " + " ".join(shlex.quote(part) for part in _command_words(arguments))
    estimate = price(deck, gpu, host=host, command=command)
    payload = estimate.record()
    payload["gpu_query"] = {k: v for k, v in query.record().items() if k != "gpus"}
    if learned:
        payload["learned"] = learned
    if measured:
        payload["measured"] = dict(measured)
    geometry = dict(estimate.storm)
    if deck.rundir is not None and (deck.rundir / "deck.toml").is_file():
        # Which deck these numbers were worked out for, by content: the GUI compares it with the run's deck.toml
        # to tell a stale estimate from a current one without trusting file times (a WSL2 clock can step).
        import hashlib

        deck_sha256 = hashlib.sha256((deck.rundir / "deck.toml").read_bytes()).hexdigest()
        payload["deck_file_sha256"] = deck_sha256
        geometry["deck_file_sha256"] = deck_sha256
    payload = _json_safe(payload)
    if deck.rundir is not None and not arguments.no_write:
        _write_json_atomic(deck.rundir / "estimate.json", payload)
        _write_json_atomic(deck.rundir / "geometry.json", _json_safe(geometry))
    return estimate, payload


#: ``gpuwm-cm1 estimate --each``: one JSON line per target.
EACH_SCHEMA = "gpuwm-cm1.estimate-each/v1"


def _estimate_each(arguments: Any) -> int:
    """``--each T1 --each T2 ...``: every target priced against one GPU answer, one JSON line each as it is done.

    Each line is ``{"schema", "target", "rc", "payload"}``: ``payload`` and ``rc`` are what
    ``gpuwm-cm1 estimate TARGET [--gpu N] [--root R] [--no-write] --json`` prints and returns for that target
    alone (its ``command`` names that call), priced against the same ``nvidia-smi`` answer, which is asked once.
    The call exits 0 when every line was written, whatever each target's ``rc``.
    """

    import argparse

    clash = [flag for flag, on in (("TARGET", arguments.target is not None), ("--measure", arguments.measure),
                                   ("--learn", bool(arguments.learn)), ("--sounding", arguments.sounding is not None))
             if on]
    if clash:
        print(f"gpuwm-cm1 estimate: --each prices run folders or decks on their own; drop {', '.join(clash)}",
              file=sys.stderr)
        return 2
    if not arguments.json:
        print("gpuwm-cm1 estimate: --each writes JSON lines; add --json", file=sys.stderr)
        return 2
    query: GpuQuery = query_gpus()
    try:
        gpu = query.select(arguments.gpu)
    except ValueError as error:
        gpu, gpu_error = None, str(error)
    else:
        gpu_error = None
    for target in arguments.each:
        one = argparse.Namespace(**{**vars(arguments), "target": target, "each": None})
        payload: dict[str, Any]
        if gpu_error is not None:
            rc, payload = 2, {"schema": ESTIMATE_SCHEMA, "status": "not-priced", "reason": gpu_error}
        else:
            try:
                deck = load_deck(target)
                learned = _learn_rundir(deck)
                _, payload = _priced_payload(one, deck, query, gpu, _host_for(one, deck), learned, None)
                rc = 0
            except EstimateError as error:
                rc, payload = 2, {"schema": ESTIMATE_SCHEMA, "status": "not-priced", "reason": str(error)}
                if error.refusal:
                    payload["refusal"] = dict(error.refusal)
            except CM1Refusal as error:
                rc, payload = 2, {"schema": ESTIMATE_SCHEMA, "status": "not-priced", "reason": str(error),
                                  "refusal": error.detail.as_dict()}
            except (OSError, ValueError) as error:  # an unreadable folder never hides the other targets
                rc, payload = 2, {"schema": ESTIMATE_SCHEMA, "status": "not-priced",
                                  "reason": f"{target} could not be read: {error}"}
            payload = json.loads(json.dumps(payload, default=str))
        line = {"schema": EACH_SCHEMA, "target": str(target), "rc": rc, "payload": payload}
        sys.stdout.write(json.dumps(line, sort_keys=True, allow_nan=False) + "\n")
        sys.stdout.flush()
    return 0


def _command_words(arguments: Any) -> list[str]:
    words = [str(arguments.target)]
    if arguments.sounding:
        words += ["--sounding", str(arguments.sounding)]
    if arguments.gpu is not None:
        words += ["--gpu", str(arguments.gpu)]
    if arguments.root:
        words += ["--root", str(arguments.root)]
    if arguments.measure:
        words += ["--measure", "--measure-steps", str(arguments.measure_steps)]
        if arguments.library:
            words += ["--library", str(arguments.library)]
        if arguments.lock:
            words += ["--lock", str(arguments.lock)]
    for receipt in arguments.learn or ():
        words += ["--learn", str(receipt)]
    if arguments.no_write:
        words.append("--no-write")
    if arguments.json:
        words.append("--json")
    return words


def _fail(arguments: Any, message: str, refusal: Mapping[str, Any] | None) -> int:
    if getattr(arguments, "json", False):
        payload = {"schema": ESTIMATE_SCHEMA, "status": "not-priced", "reason": message}
        if refusal:
            payload["refusal"] = dict(refusal)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(f"gpuwm-cm1 estimate: {message}", file=sys.stderr)
    return 2


def add_estimate_parser(commands: Any) -> None:
    """``gpuwm-cm1 estimate`` on the CLI's subparsers."""

    command = commands.add_parser(
        "estimate",
        help="price a deck, a CM1 namelist or a run folder on this machine's GPU (no GPU is opened)",
        description=(
            "GPU memory, locked host memory, disk and wall time for one run on one GPU, "
            "against the memory nvidia-smi says is free, with the storm-track check. "
            "Nothing is run unless --measure is given."
        ),
    )
    command.add_argument("target", nargs="?", help="gpuwm-cm1 deck (.toml), CM1 namelist.input, or run folder")
    command.add_argument(
        "--each", action="append", type=Path, metavar="TARGET",
        help="price several run folders or decks against one nvidia-smi answer (repeatable, with --json): one "
             "JSON line per target, as it is priced, holding what 'gpuwm-cm1 estimate TARGET --json' prints for it",
    )
    command.add_argument("--sounding", type=Path, help="input_sounding for isnd = 7 or 17")
    command.add_argument("--gpu", type=int, help="nvidia-smi's GPU index (default: the first)")
    command.add_argument("--root", type=Path, help="the folder whose free disk to price against")
    command.add_argument("--json", action="store_true", help="print the estimate as JSON")
    command.add_argument(
        "--no-write", action="store_true",
        help="for a run folder, do not write estimate.json and geometry.json into it",
    )
    command.add_argument(
        "--measure", action="store_true",
        help="run a few steps on the GPU first (Measure speed) and keep the measured rate",
    )
    command.add_argument(
        "--measure-steps", type=int, default=60,
        help=f"steps for --measure (default 60, at least {MEASURE_MIN_STEPS})",
    )
    command.add_argument("--library", type=Path, help="the Rust initializer for --measure in a source checkout")
    command.add_argument("--lock", type=Path, help="the GPU mutex directory --measure takes with its GPU's UUID")
    command.add_argument(
        "--learn", action="append", type=Path, metavar="RECEIPT",
        help="add a finished run's receipt to this machine's measured rates (repeatable)",
    )
    command.set_defaults(handler=estimate_door)


__all__ = [
    "Check",
    "DISK_FITS_FRACTION",
    "Deck",
    "ESTIMATE_SCHEMA",
    "Estimate",
    "EstimateError",
    "FIT_EDGE_CLEARANCE_M",
    "FITS_FRACTION_OF_FREE",
    "MEASURE_MIN_STEPS",
    "Quantity",
    "RECOMMENDED_TILE",
    "STORM_TRACK_SCHEMA",
    "add_estimate_parser",
    "deck_from_mapping",
    "device_need",
    "deck_from_namelist",
    "dump_toml",
    "duration_words",
    "estimate_door",
    "largest_square_domain",
    "learn_from_rundir",
    "load_deck",
    "measure_speed",
    "price",
    "render_text",
    "storm_track",
]
