"""``gpuwm-cm1 chain``: downscale a WRF or ArWen run into a finer GM1 domain (3D).

The packaged front of the ArWen chain (:mod:`cm1core.arwen_chain`,
``records/GM1-ARWEN-CHAIN.md``): WRF's ``ndown`` in GM1 terms.  A finer,
flat, unmapped GM1 Cartesian domain takes its initial state from one saved
time of a WRF or ArWen history and its lateral boundaries are specified and
relaxed toward the saved times after it.

::

    gpuwm-cm1 chain inspect FILES|DIR [--map-png P] [--json]
    gpuwm-cm1 chain check --history FILES|DIR --window I0,J0,NI,NJ --ratio R
                          [--start T] [--end T] [--every S] [--scheme S] [--json]
    gpuwm-cm1 chain new   --history ... --window ... --ratio R ... --out RUNDIR [--json]
    gpuwm-cm1 chain frames-from-gm1 RUNDIR --out DIR [--every S] [--json]

``chain new`` writes an ordinary run directory (``gm1.cm1-run/v1``): the
physics and numerics come from a CM1 namelist (the shipped 1 km supercell's
by default), imported by :mod:`cm1core.cm1import` as any namelist is, with the
grid, clock, lateral boundaries and initial condition the chain requires; the
prepared products go into ``RUNDIR/chain``, the prepared t0 column is the
run's ``input_sounding``, and ``deck.toml`` carries the ``[chain]`` table.
``gpuwm-cm1 run RUNDIR`` runs it like any run.

History files are read through :mod:`cm1core.wrf_history` (netCDF4):
classic CDF-1/CDF-2/CDF-5 and netCDF-4, one or many records per
file, ArWen or WRF.  Frames extracted by ``gm1-arwen-history-extract`` (the
Rust oracle) are read as they are.  Host only: nothing here opens a device.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from . import arwen_chain as chain
from .errors import CM1Refusal

CHAIN_DOOR_SCHEMA = "gm1.chain-door/v1"
CHAIN_RUN_SCHEMA = "gm1.chain-run/v1"
#: GM1 defaults for the two limits the document declares (records/GM1-ARWEN-CHAIN.md 2).  Neither is
#: a transcription: 50 m is about the depth of a WRF run's lowest model layer, so the ground GM1 flattens
#: lies within it; 0.01 is a 1% error in every horizontal length.  ``--terrain-limit`` and
#: ``--map-factor-limit`` declare others, and the run records what was declared.
DEFAULT_TERRAIN_LIMIT_M = 50.0
DEFAULT_MAP_FACTOR_LIMIT = 0.01
#: WRF's default specified and relaxation zone widths (namelist.input spec_zone = 1, relax_zone = 4).
DEFAULT_SPEC_ZONE = 1
DEFAULT_RELAX_ZONE = 4
DEFAULT_DZ_M = 500.0
DEFAULT_ZTOP_M = 20000.0
#: CM1's own cases step 6 s at 1 km (run/config_files supercell and squall_line); the chain's
#: step is the largest of these at most that rate that divides the saving interval exactly.
STEP_S_PER_KM = 6.0
NICE_STEPS_S = (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0, 1.2, 1.25, 1.5, 2.0, 2.4, 2.5, 3.0, 4.0,
                5.0, 6.0, 7.5, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
#: The GM1 components the chain offers, as CM1 ptype (cm1import.translate.PTYPE_COMPONENTS).
COMPONENT_PTYPE = {"morrison-mp10": 5, "wsm6-mp6": 7, "thompson-mp8": 3}
#: The GM1 schemes by the names the checks print.
COMPONENT_NAMES = {"morrison-mp10": "Morrison (10)", "wsm6-mp6": "WSM6 (6)", "thompson-mp8": "Thompson (8)"}
#: The physics and numerics a chained run takes when no namelist is given.
TEMPLATE_PRESET = "supercell-1km"
CHAIN_DIRECTORY = "chain"
#: The verification label; pinned by tests/test_gui_chain.py.
VERIFIED = (
    "Checked on one GPU (an RTX 3080) with an idealized ArWen storm run, with GM1 driving itself, and with one "
    "real-weather ArWen run (an atmospheric river over the ocean off California, water vapour only). "
    "Not compared with observations."
)
HISTORY_GLOBS = ("wrfout_*", "*.nc", "*.nc4")


class ChainDoorError(ValueError):
    """A request this command cannot act on; the message says what to do."""

    def __init__(self, message: str, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


def _np() -> Any:
    import numpy

    return numpy


def _copy() -> Mapping[str, Any]:
    path = Path(__file__).resolve().parent / "gui" / "copy" / "chain.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _minutes(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "never"
    if seconds < 90:
        return f"{seconds:g} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min" if abs(minutes - round(minutes)) < 1e-6 else f"{minutes:.1f} min"
    return f"{minutes / 60.0:.1f} h"


# ============================================================ sources


def _is_frame_dir(path: Path) -> bool:
    return (path / "frame.json").is_file() and (path / "fields.f32le").is_file()


def history_paths(sources: Sequence[Path | str]) -> list[Path]:
    """The history files and extracted frame directories named by ``sources``, expanded."""

    found: list[Path] = []
    for raw in sources:
        path = Path(raw).expanduser()
        if _is_frame_dir(path) or path.is_file():
            found.append(path)
            continue
        if not path.is_dir():
            raise ChainDoorError(f"{path} does not exist", "name a wrfout file, a folder of them, or a frame folder")
        frames = sorted(child for child in path.iterdir() if child.is_dir() and _is_frame_dir(child))
        if frames:
            found.extend(frames)
            continue
        files: list[Path] = []
        for pattern in HISTORY_GLOBS:
            files.extend(child for child in path.glob(pattern) if child.is_file())
        files = sorted(set(files))
        if not files:
            raise ChainDoorError(
                f"{path} holds no wrfout files and no extracted frames",
                "name the folder the WRF or ArWen run wrote its wrfout_d0N_* files to",
            )
        found.extend(files)
    return found


def open_frames(sources: Sequence[Path | str]) -> list[Any]:
    """Every saved time in ``sources``, oldest first (one frame per history record)."""

    frames: list[Any] = []
    for path in history_paths(sources):
        if _is_frame_dir(path):
            frames.append(chain.ArwenFrame.open(path))
            continue
        from .wrf_history import open_history

        history = open_history(path)
        for record in range(int(history.records)):
            frames.append(history.frame(record))
    domains = sorted({_domain_id(frame) for frame in frames})
    if len(domains) > 1:
        names = " or ".join(f"d{d:02d}" for d in domains)
        raise ChainDoorError(
            f"these files come from {len(domains)} domains of one run; name one domain's files only ({names})",
            "choose the domain whose spacing you want to refine and name only its files",
        )
    frames.sort(key=lambda frame: frame.valid_s)
    for a, b in zip(frames, frames[1:]):
        if a.valid_s == b.valid_s:
            raise ChainDoorError(
                f"two saved times are both {a.times} ({_label(a)} and {_label(b)})",
                "name one domain's files only (d01 or d02, not both)",
            )
    return frames


def _label(frame: Any) -> str:
    source = chain.frame_source(frame)
    where = source.get("path") or source.get("directory") or ""
    record = source.get("record")
    return f"{Path(str(where)).name}" + ("" if record in (None, 0) else f" record {record}")


def select_frames(frames: Sequence[Any], *, start: str | None, end: str | None,
                  every_s: float | None) -> list[Any]:
    """The saved times from ``start`` to ``end``, thinned to one every ``every_s`` seconds."""

    if not frames:
        raise ChainDoorError("no saved times were found", "name a wrfout file or a folder of them")
    lo = frames[0].valid_s if start is None else chain.parse_wrf_time(start)
    hi = frames[-1].valid_s if end is None else chain.parse_wrf_time(end)
    chosen = [frame for frame in frames if lo <= frame.valid_s <= hi]
    if every_s:
        chosen = [frame for frame in chosen if abs(((frame.valid_s - lo) / float(every_s)) - round((frame.valid_s - lo) / float(every_s))) < 1e-9]
    if start is not None and (not chosen or chosen[0].valid_s != lo):
        raise ChainDoorError(f"{start} is not a saved time", "choose a start from the times gpuwm-cm1 chain inspect lists")
    if end is not None and (not chosen or chosen[-1].valid_s != hi):
        raise ChainDoorError(f"{end} is not a saved time" + ("" if not every_s else f" on the {every_s:g} s spacing"),
                             "choose an end from the times gpuwm-cm1 chain inspect lists")
    return chosen


# ============================================================ inspect


def _grid_facts(frame: Any) -> dict[str, Any]:
    np = _np()
    attrs = frame.attributes
    projection = int(attrs.get("MAP_PROJ", -1))
    facts: dict[str, Any] = {
        "west_east": int(frame.dimensions["west_east"]),
        "south_north": int(frame.dimensions["south_north"]),
        "bottom_top": int(frame.dimensions["bottom_top"]),
        "dx_m": float(attrs.get("DX", float("nan"))), "dy_m": float(attrs.get("DY", float("nan"))),
        "map_proj": projection, "projection": chain.PROJECTION_NAMES.get(projection, f"MAP_PROJ {projection}"),
    }
    for key in ("TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON", "GRID_ID", "HYBRID_OPT", "USE_THETA_M"):
        if attrs.get(key) is not None:
            facts[key.lower()] = attrs.get(key)
    if frame.has("HGT"):
        hgt = frame.array("HGT").astype(np.float64)
        facts["terrain_m"] = {"min": float(hgt.min()), "max": float(hgt.max())}
    if frame.has("MAPFAC_M"):
        m = frame.array("MAPFAC_M").astype(np.float64)
        facts["map_factor"] = {"min": float(m.min()), "max": float(m.max())}
    for name in ("XLAT", "XLONG"):
        if frame.has(name):
            a = frame.array(name).astype(np.float64)
            facts[name.lower()] = {"min": float(a.min()), "max": float(a.max())}
    if frame.has("PH") and frame.has("PHB"):
        ph = frame.array("PH")[-1].astype(np.float64) + frame.array("PHB")[-1].astype(np.float64)
        top = ph / chain.WRF_GRAVITY_M_S2 - (frame.array("HGT").astype(np.float64) if frame.has("HGT") else 0.0)
        facts["model_top_m"] = {"min": float(top.min()), "max": float(top.max())}
    return facts


def inspect(frames: Sequence[Any], *, sources: Sequence[Any] = ()) -> dict[str, Any]:
    """What a history is, in the terms the chain admits it by."""

    if not frames:
        raise ChainDoorError("no saved times were found", "name a wrfout file or a folder of them")
    first = frames[0]
    attrs = first.attributes
    mp = attrs.get("MP_PHYSICS")
    try:
        component = chain.component_for_mp_physics(mp)
        scheme = {"mp_physics": mp, "scheme": chain.wrf_scheme_name(mp), "component": component,
                  "offered": True}
    except CM1Refusal as refusal:
        scheme = {"mp_physics": mp, "scheme": chain.wrf_scheme_name(mp), "component": None,
                  "offered": False, "refusal": refusal.detail.as_dict()}
    times = [frame.times for frame in frames]
    gaps = sorted({round(b.valid_s - a.valid_s, 6) for a, b in zip(frames, frames[1:])})
    return {
        "schema": CHAIN_DOOR_SCHEMA,
        "action": "inspect",
        "sources": [os.fspath(path) for path in sources],
        "producer": chain.producer_of(attrs),
        "title": str(attrs.get("TITLE", "")).strip(),
        "times": times,
        "count": len(frames),
        "intervals_s": gaps,
        "grid": _grid_facts(first),
        "microphysics": scheme,
        "use_theta_m": attrs.get("USE_THETA_M"),
        "has": {name: bool(first.has(name)) for name in (
            "U", "V", "W", "T", "P", "PB", "PH", "PHB", "HGT", "MAPFAC_M", "XLAT", "XLONG", "F", "PSFC",
            "T2", "TH2", "Q2", "THM", "QVAPOR")},
        "verified": VERIFIED,
    }


#: The updraft outline on the domain map: column-maximum w at or above this (m/s) at the time drawn.
MAP_UPDRAFT_M_S = 5.0


def domain_map_png(frame: Any, path: Path | str, *, window: Sequence[int] | None = None) -> dict[str, Any]:
    """A picture of the domain for drawing the box: ground height, updrafts, the map-scale line, the unusable edge.

    The ground is drawn where it varies, grey by height with water dark blue (a flat or idealized run is
    drawn dark); updrafts of at least
    :data:`MAP_UPDRAFT_M_S` (column maximum of W) are outlined in orange, and the three history cells
    along each side that no box may use are shaded.  Returns the pixel box of the domain inside the picture, so a page can turn a
    drag into cells.
    """

    np = _np()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ny, nx = int(frame.dimensions["south_north"]), int(frame.dimensions["west_east"])
    hgt = frame.array("HGT").astype(np.float64) if frame.has("HGT") else np.zeros((ny, nx))
    width, height = 960, 540
    void = "#0a0a0a"
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    try:
        fig.patch.set_facecolor(void)
        aspect = ny / nx
        box_h = 0.9
        box_w = min(0.96, box_h / aspect * height / width)
        box_h = box_w * aspect * width / height
        left = (1.0 - box_w) / 2.0
        bottom = (1.0 - box_h) / 2.0
        ax = fig.add_axes([left, bottom, box_w, box_h])
        ax.set_facecolor("#151515")
        relief = float(hgt.max() - hgt.min())
        if relief > 0:
            # Land as a grey ramp by height (dark low, light high); water, where the history says so
            # (LANDMASK, else ground at or below sea level), a flat dark blue.
            water = frame.array("LANDMASK").astype(np.float64) < 0.5 if frame.has("LANDMASK") else hgt <= 0.0
            top = max(1.0, float(np.percentile(hgt[~water], 99.5)) if (~water).any() else 1.0)
            shade = np.clip(hgt / top, 0.0, 1.0)
            rgb = np.empty((ny, nx, 3))
            for c, (lo, hi) in enumerate(((0.16, 0.86), (0.15, 0.84), (0.13, 0.80))):
                rgb[..., c] = lo + (hi - lo) * shade
            rgb[water] = (0.06, 0.15, 0.23)
            ax.imshow(rgb, origin="lower", extent=(0, nx, 0, ny), interpolation="nearest")
        xs, ys = np.arange(nx) + 0.5, np.arange(ny) + 0.5
        updraft = None
        if frame.has("W"):
            wmax = frame.array("W").astype(np.float64).max(axis=0)
            wmax = 0.5 * (wmax[:-1] + wmax[1:]) if wmax.shape[0] == ny + 1 else wmax
            updraft = float(wmax.max())
            if updraft >= MAP_UPDRAFT_M_S:
                ax.contourf(xs, ys, wmax, levels=[MAP_UPDRAFT_M_S, max(updraft, MAP_UPDRAFT_M_S) + 1.0],
                            colors=["#f09a3a"], alpha=0.35)
                ax.contour(xs, ys, wmax, levels=[MAP_UPDRAFT_M_S], colors=["#f09a3a"], linewidths=1.2)
        edge = 3
        for rect in ((0, 0, nx, edge), (0, ny - edge, nx, edge), (0, edge, edge, ny - 2 * edge),
                     (nx - edge, edge, edge, ny - 2 * edge)):
            ax.add_patch(plt.Rectangle(rect[:2], rect[2], rect[3], facecolor="#8a8a8a", edgecolor="none",
                                       alpha=0.45))
        if window is not None:
            i0, j0, ni, nj = window
            ax.add_patch(plt.Rectangle((i0, j0), ni, nj, fill=False, edgecolor="#22d3ee", linewidth=1.6))
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a3a")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return {
        "png": os.fspath(path), "width": width, "height": height, "time": frame.times,
        "domain_px": {"left": left * width, "top": (1.0 - bottom - box_h) * height,
                      "width": box_w * width, "height": box_h * height},
        "cells": {"west_east": nx, "south_north": ny}, "edge_cells": 3,
        "terrain_m": {"min": float(hgt.min()), "max": float(hgt.max())},
        "ground_drawn": relief > 0,
        "updraft_outline_m_s": MAP_UPDRAFT_M_S, "max_updraft_m_s": updraft,
        # GM1 takes the box's own map scale (arwen_chain.GM1_CHOICES["map_scale"]), so no line on the
        # map bounds where a box may go; the check measures how much the scale varies inside a box.
        "map_factor_line": None,
    }


# ============================================================ the request


@dataclass
class ChainChoice:
    """What the user chose, with the defaults filled in from the history."""

    window: tuple[int, int, int, int]
    ratio: int
    start: str
    end: str
    every_s: float | None
    component: str
    nz: int
    dz_m: float
    dt_s: float
    duration_s: float
    output_s: float
    spec_zone: int = DEFAULT_SPEC_ZONE
    relax_zone: int = DEFAULT_RELAX_ZONE
    terrain_limit_m: float = DEFAULT_TERRAIN_LIMIT_M
    map_factor_limit: float = DEFAULT_MAP_FACTOR_LIMIT
    coriolis: int = 0
    fcor_s_inv: float = 0.0
    dx_m: float = float("nan")
    notes: list[str] = field(default_factory=list)
    moisture: str = chain.DEFAULT_MOISTURE

    @property
    def nx(self) -> int:
        return self.window[2] * self.ratio

    @property
    def ny(self) -> int:
        return self.window[3] * self.ratio

    def as_dict(self) -> dict[str, Any]:
        return {
            "window": {"i0": self.window[0], "j0": self.window[1], "ni": self.window[2], "nj": self.window[3]},
            "ratio": self.ratio, "start": self.start, "end": self.end, "every_s": self.every_s,
            "component": self.component, "nx": self.nx, "ny": self.ny, "nz": self.nz, "dz_m": self.dz_m,
            "dt_s": self.dt_s, "duration_s": self.duration_s, "output_s": self.output_s,
            "spec_zone": self.spec_zone, "relax_zone": self.relax_zone,
            "terrain_limit_m": self.terrain_limit_m, "map_factor_limit": self.map_factor_limit,
            "coriolis": self.coriolis, "fcor_s_inv": self.fcor_s_inv, "dx_m": self.dx_m,
            "moisture": self.moisture, "notes": list(self.notes),
        }


def parse_window(text: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(part) for part in str(text).replace(" ", "").split(",")]
    except ValueError:
        parts = []
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0 or parts[0] < 0 or parts[1] < 0:
        raise ChainDoorError(f"--window {text!r} is not I0,J0,NI,NJ", "give the box as four whole numbers: first cell and size")
    return tuple(parts)  # type: ignore[return-value]


def choose_step(dx_m: float, interval_s: float) -> float:
    """The largest of :data:`NICE_STEPS_S` at most CM1's 6 s per km that divides ``interval_s`` exactly."""

    limit = STEP_S_PER_KM * dx_m / 1000.0
    for dt in sorted(NICE_STEPS_S, reverse=True):
        if dt > limit + 1e-9:
            continue
        steps = interval_s / dt
        if float(steps).is_integer() and round(steps) * dt == interval_s:
            return dt
    raise ChainDoorError(
        f"no time step up to {limit:g} s divides the {interval_s:g} s between saved times",
        "pass --dt with a step that divides it",
    )


def _history_tops(frame: Any, window: Sequence[int]) -> tuple[float, float]:
    """The lowest last mass level and the lowest top face of the history's columns in the box (m above ground)."""

    np = _np()
    i0, j0, ni, nj = window
    # The transfer reads one history cell around the box (nest_vertical.VerticalWindow).
    sl = (slice(max(j0 - 1, 0), j0 + nj + 1), slice(max(i0 - 1, 0), i0 + ni + 1))
    hgt = frame.array("HGT")[sl].astype(np.float64) if frame.has("HGT") else 0.0
    faces = []
    for k in (-2, -1):
        ph = frame.array("PH")[k][sl].astype(np.float64) + frame.array("PHB")[k][sl].astype(np.float64)
        faces.append(ph / chain.WRF_GRAVITY_M_S2 - hgt)
    return float(np.min(0.5 * (faces[0] + faces[1]))), float(np.min(faces[1]))


def _window_rotation(frame: Any, window: Sequence[int]) -> tuple[int, float]:
    np = _np()
    i0, j0, ni, nj = window
    centre = (min(j0 + nj // 2, int(frame.dimensions["south_north"]) - 1),
              min(i0 + ni // 2, int(frame.dimensions["west_east"]) - 1))
    msl = (slice(j0, j0 + nj), slice(i0, i0 + ni))
    try:
        _, f_centre, _ = chain._f_window(frame, msl, centre)
    except CM1Refusal:
        return 0, 0.0
    f_centre = float(f_centre)
    return (1, f_centre) if f_centre != 0.0 else (0, 0.0)


def make_choice(frames: Sequence[Any], *, window: str | Sequence[int], ratio: int, start: str | None = None,
                end: str | None = None, every_s: float | None = None, scheme: str | None = None,
                dz_m: float | None = None, ztop_m: float | None = None, dt_s: float | None,
                duration_s: float | None = None, output_s: float | None = None,
                terrain_limit_m: float | None = None, map_factor_limit: float | None = None,
                spec_zone: int | None = None, relax_zone: int | None = None,
                moisture: str | None = None) -> tuple[ChainChoice, list[Any]]:
    """Fill every default of a chain request from the history; returns the choice and its frames.

    ``dt_s`` has no default (the timestep rule, tests/test_storm_lane.py): pass a step, or ``None`` to have the
    chain choose one from the grid spacing and the history's interval (``choose_step``), explicitly."""

    box = parse_window(window) if isinstance(window, str) else tuple(int(v) for v in window)
    chosen = select_frames(frames, start=start, end=end, every_s=every_s)
    if len(chosen) < 2:
        interval = 0.0
    else:
        interval = float(chosen[1].valid_s - chosen[0].valid_s)
    first = chosen[0]
    notes: list[str] = []
    lane = str(moisture or chain.DEFAULT_MOISTURE)
    if lane not in chain.MOISTURE_LANES:
        raise ChainDoorError(f"--moisture {moisture!r} is not a moisture lane",
                             "use all-species or vapour-only")
    component: str
    if scheme:
        text = str(scheme)
        component = chain.CHAIN_SCHEMES.get(int(text), "") if text.isdigit() else text
        if component not in COMPONENT_PTYPE:
            raise ChainDoorError(f"--scheme {scheme!r} is not one GM1 downscales",
                                 "use wsm6-mp6, thompson-mp8 or morrison-mp10 (or 6, 8, 10)")
    else:
        mp = first.attributes.get("MP_PHYSICS")
        component = chain.CHAIN_SCHEMES.get(int(mp)) if isinstance(mp, (int, float)) or str(mp).lstrip("-").isdigit() else None  # type: ignore[assignment]
        if component is None:
            component = "wsm6-mp6"
            if lane == "vapour-only":
                notes.append(f"the history ran {chain.wrf_scheme_name(mp)}; GM1 runs WSM6 (6) and takes only the "
                             "history's water vapour (--scheme picks another GM1 scheme)")
            else:
                notes.append(f"the history ran {chain.wrf_scheme_name(mp)}; the microphysics check says why it cannot "
                             "be downscaled with every species, and --moisture vapour-only takes its water vapour alone")
    ratio = int(ratio)
    dx = float(first.attributes.get("DX", 0.0)) / ratio if ratio else float("nan")
    try:
        # The history's true spacing over the box: DX/(ratio*m), m the box's mean map factor
        # (arwen_chain.GM1_CHOICES["map_scale"]).  A box off the grid is refused by the placement check.
        i0, j0, ni, nj = box
        if ratio and 1 <= i0 and 1 <= j0 and i0 + ni + 1 <= int(first.dimensions["west_east"]) and \
                j0 + nj + 1 <= int(first.dimensions["south_north"]):
            cells = chain.box_mass_slice(i0, j0, ni, nj)
            dx = chain.chain_spacing_m(first, ratio, cells)
            if dx != float(first.attributes.get("DX", 0.0)) / ratio:
                scale, _ = chain.window_map_scale(first, cells)
                notes.append(f"the box's map scale is {scale:.5f}, so GM1's cells are {dx:.6g} m on the ground "
                             f"(your run's {float(first.attributes['DX']):g} m / {ratio} / {scale:.5f})")
    except Exception:  # noqa: BLE001 - a history without MAPFAC_M is refused by the variables check
        pass
    dz = float(dz_m or DEFAULT_DZ_M)
    top_cap = float(ztop_m or DEFAULT_ZTOP_M)
    try:
        last_mass, top_face = _history_tops(first, box)
    except Exception:  # noqa: BLE001 - a malformed window is refused by the placement check
        last_mass, top_face = top_cap, top_cap
    # GM1's highest scalar level (ztop - dz/2) must not be above the history's last mass level, nor its
    # top face above the history's (the model-top check says why).
    fitting = min(math.floor((last_mass + 0.5 * dz) / dz), math.floor(top_face / dz)) * dz
    if ztop_m is None and fitting < top_cap:
        top_cap = fitting
        notes.append(f"GM1's top lowered to {top_cap:g} m, under your run's last level in the box "
                     f"({last_mass:.0f} m)")
    nz = max(int(round(top_cap / dz)), 3)
    if dt_s is None:
        dt = choose_step(dx, interval) if interval > 0 and math.isfinite(dx) and dx > 0 else 1.0
    else:
        dt = float(dt_s)
    span = float(chosen[-1].valid_s - chosen[0].valid_s) if len(chosen) > 1 else 0.0
    coriolis, fcor = _window_rotation(first, box)
    choice = ChainChoice(
        window=box, ratio=ratio, start=first.times, end=chosen[-1].times, every_s=every_s,
        component=component, nz=nz, dz_m=dz, dt_s=dt,
        duration_s=float(duration_s) if duration_s is not None else span,
        output_s=float(output_s) if output_s is not None else (min(interval, 300.0) if interval > 0 else 300.0),
        spec_zone=int(spec_zone or DEFAULT_SPEC_ZONE),
        relax_zone=DEFAULT_RELAX_ZONE if relax_zone is None else int(relax_zone),
        terrain_limit_m=DEFAULT_TERRAIN_LIMIT_M if terrain_limit_m is None else float(terrain_limit_m),
        map_factor_limit=DEFAULT_MAP_FACTOR_LIMIT if map_factor_limit is None else float(map_factor_limit),
        coriolis=coriolis, fcor_s_inv=fcor, dx_m=dx, notes=notes, moisture=lane,
    )
    return choice, chosen


def request_of(choice: ChainChoice) -> chain.ChainRequest:
    from .cm1_vertical_grid import uniform_vertical_levels

    return chain.ChainRequest(
        nx=choice.nx, ny=choice.ny, dx_m=choice.dx_m, dy_m=choice.dx_m,
        levels=uniform_vertical_levels(choice.nz, choice.dz_m), dt_s=choice.dt_s,
        duration_s=choice.duration_s, component=choice.component, coriolis=choice.coriolis,
        fcor_s_inv=choice.fcor_s_inv, start_time=choice.start, end_time=choice.end,
        placement=chain.ChainPlacement(
            parent_i0=choice.window[0], parent_j0=choice.window[1], refinement_ratio=choice.ratio,
            spec_zone=choice.spec_zone, relax_zone=choice.relax_zone,
        ),
        terrain_relief_limit_m=choice.terrain_limit_m, map_factor_departure_limit=choice.map_factor_limit,
        moisture=choice.moisture,
    )


# ============================================================ check


def _plain(check: Mapping[str, Any], facts: Mapping[str, Any], choice: ChainChoice,
           copy: Mapping[str, Any]) -> dict[str, Any]:
    """One check as a line a person reads: its label, pass or why not, the code as a hint."""

    words = copy["checks"]
    name = check["id"]
    entry = words[name]
    status = check["status"]
    projection = facts.get("projection") or {}
    interval = facts.get("interval_s")
    coriolis = facts.get("coriolis") or {}
    f = float(coriolis.get("window_centre_f_s_inv") or 0.0)
    if f:
        lat = math.degrees(math.asin(max(-1.0, min(1.0, f / (2.0 * chain.WRF_EOMEG_S_INV)))))
        rotation = copy["rotation"]["f"].format(lat=f"{lat:.1f}")
    else:
        rotation = copy["rotation"]["none"]
    micro = facts.get("microphysics") or {}
    refusal = check.get("refusal") or {}
    values = {
        "n": facts.get("frame_count", 0),
        "interval": _minutes(interval) if interval else "?",
        "start": choice.start, "end": choice.end,
        "steps": int(round(interval / choice.dt_s)) if interval else "?", "dt": f"{choice.dt_s:g}",
        "duration": _minutes(choice.duration_s),
        "projection": projection.get("name") or f"MAP_PROJ {refusal.get('value')}",
        "cell_km": f"{(facts.get('arwen_dx_m') or 0) / 1000.0:g}",
        "ratio": choice.ratio, "cell_m": f"{choice.dx_m:g}",
        "relief": f"{facts.get('terrain_relief_m', float('nan')):.0f}",
        "limit": (f"{choice.terrain_limit_m:g}" if name == "terrain" else f"{100.0 * choice.map_factor_limit:g}"),
        "departure": f"{100.0 * float(facts.get('map_factor_departure', float('nan'))):.2f}",
        "scale": f"{float((facts.get('map_scale') or {}).get('window_mean_map_factor', 1.0)):.4f}",
        "gm1_scheme": COMPONENT_NAMES.get(choice.component, choice.component),
        "rotation": rotation,
        "scheme": micro.get("scheme") or chain.wrf_scheme_name((refusal.get("value") or [None])[0]
                                                               if isinstance(refusal.get("value"), list) else None),
        "crossing": _minutes(facts.get("crossing_time_s", float("nan"))),
    }
    if name == "microphysics" and status == "refused":
        value = refusal.get("value")
        values["scheme"] = chain.wrf_scheme_name(value[0] if isinstance(value, list) else value)
    if choice.moisture == "vapour-only" and name in ("microphysics", "species") and status == "pass":
        entry = dict(entry, **{"pass": entry["pass_vapour"]})
    if name == "map_factor" and status == "pass" and str(facts.get("map_factor_source", "")).startswith("1 ("):
        entry = dict(entry, **{"pass": entry["pass_unit"]})
    if name == "frame_grid" and status == "refused" and refusal.get("option") == "HGT":
        t0, t1, change = refusal.get("value") or ("?", "?", float("nan"))
        values.update(t0=t0, t1=t1, dhgt=f"{float(change):.2g}")
        entry = dict(entry, refused=entry["refused_ground"])
    missing = refusal.get("value") if name == "species" else refusal.get("option")
    if isinstance(missing, list):
        missing = ", ".join(str(item) for item in missing)
    values["missing"] = str(missing or "a field")
    if status == "refused" and refusal.get("code") == "GM1-CHAIN-VARIABLE":
        # A field this check reads is not in the history: say which, not the check's own numbers.
        entry = dict(entry, refused=words["variables"]["refused"])
    if name == "theta_m" and status == "refused" and "carries no THM" in str(refusal.get("reason", "")):
        entry = dict(entry, refused=entry["refused_no_thm"])
    if status == "not-checked":
        text = words["not_checked"].format(after=copy["labels"].get(check.get("after"), check.get("after")))
    else:
        try:
            text = entry["pass" if status == "pass" else "refused"].format(**values)
        except (KeyError, ValueError, IndexError):
            text = entry["pass" if status == "pass" else "refused"]
    row = {"id": name, "label": entry["label"], "status": status, "text": text}
    if refusal:
        row["code"] = refusal.get("code")
        row["detail"] = refusal.get("reason")
        row["remedy"] = refusal.get("remedy")
    return row


def prepare_estimate(choice: ChainChoice, frames: int) -> dict[str, Any]:
    """The prepare step's own cost: its arenas on disk and its peak host memory (float32 words)."""

    carriers = len(chain.carriers_of(choice.component, choice.moisture))
    cells = choice.nx * choice.ny * choice.nz
    width = choice.spec_zone + choice.relax_zone
    strip_cells = 2 * width * (choice.nx + choice.ny) * choice.nz
    initial = carriers * cells * 4
    strips = frames * carriers * strip_cells * 4
    return {
        "disk_bytes": initial + strips,
        "host_peak_bytes": 2 * carriers * cells * 4,
        "basis": "float32: every carrier on the GM1 lattice at t0, plus each saved time's side strips "
                 f"({width} rows deep); the remap holds one saved time's carriers at once",
    }


def check(frames: Sequence[Any], choice: ChainChoice, *, price_it: bool = True,
          command: str = "") -> dict[str, Any]:
    """Every admission check in plain words, and what the run would take."""

    copy = _copy()
    request = request_of(choice)
    report = chain.admission_report(frames, request)
    rows = [_plain(row, report["facts"], choice, copy) for row in report["checks"]]
    refused = [row for row in rows if row["status"] == "refused"]
    payload: dict[str, Any] = {
        "schema": CHAIN_DOOR_SCHEMA,
        "action": "check",
        "admitted": report["admitted"],
        "summary": copy["screen"]["admitted"] if report["admitted"] else
        copy["screen"]["refused"].format(n=len(refused)),
        "checks": rows,
        "facts": report["facts"],
        "choice": choice.as_dict(),
        "prepare": prepare_estimate(choice, len(frames)),
        "verified": VERIFIED,
        "command": command,
    }
    if report["admitted"] and price_it:
        try:
            payload["estimate"] = _price(frames, choice)
        except Exception as error:  # noqa: BLE001 - the admission stands without a price; say why there is none
            payload["estimate"] = {"error": str(error)}
    return payload


def _price(frames: Sequence[Any], choice: ChainChoice) -> dict[str, Any]:
    """The estimate on the child grid, from the deck ``chain new`` would write."""

    from .cm1run.estimate import price
    from .cm1run.gpuinfo import query_gpus, query_host

    with tempfile.TemporaryDirectory(prefix="gm1-chain-check-") as scratch:
        root = Path(scratch)
        deck, _ = _child_deck(frames, choice, root / "run", prepare_all=False)
        query = query_gpus()
        try:
            gpu = query.select(None)
        except ValueError:
            gpu = None
        estimate = price(deck, gpu, host=query_host(root))
        record = estimate.record()
    run = record["run"]
    needs = record["needs"]
    return {
        "cells": run["cells"], "steps": run["steps"], "dt_s": run["dt_s"], "duration_s": run["duration_s"],
        "frames": run["frames"], "microphysics": run["microphysics"], "grid": run["grid"],
        "gpu_memory_bytes": needs["gpu_memory_bytes"]["value"], "disk_bytes": needs["disk_bytes"]["value"],
        "wall_s": record["time"]["wall_s"]["value"], "fits": record["fits"],
        "gpu": record.get("gpu"), "basis": "gpuwm-cm1 estimate on the downscaled grid",
    }


# ============================================================ new


def _template_namelist(namelist: Path | None) -> Path:
    if namelist is not None:
        return Path(namelist).expanduser()
    return Path(__file__).resolve().parent / "presets" / TEMPLATE_PRESET / "namelist.input"


def namelist_values(choice: ChainChoice) -> dict[str, Any]:
    """The CM1 namelist words a chained run takes over the template's."""

    return {
        "nx": choice.nx, "ny": choice.ny, "nz": choice.nz,
        "dx": choice.dx_m, "dy": choice.dx_m, "dz": choice.dz_m,
        "dtl": choice.dt_s, "timax": choice.duration_s, "tapfrq": choice.output_s,
        "rstfrq": -abs(choice.duration_s), "statfrq": choice.output_s,
        "adapt_dt": 0, "stretch_x": 0, "stretch_y": 0, "stretch_z": 0, "terrain_flag": ".false.",
        "imove": 0, "umove": 0.0, "vmove": 0.0,
        "iinit": 0, "isnd": 7, "iwnd": 0, "irandp": 0,
        "ptype": COMPONENT_PTYPE[choice.component], "imoist": 1,
        "icor": choice.coriolis, "fcor": choice.fcor_s_inv if choice.coriolis else 0.0, "betaplane": 0,
        "wbc": 2, "ebc": 2, "sbc": 2, "nbc": 2, "roflux": 0, "nudgeobc": 0,
    }


def chain_table(choice: ChainChoice, *, history_directory: Path, domain_id: int,
                prepared_directory: Path) -> dict[str, Any]:
    return {
        "history_directory": os.fspath(history_directory),
        "domain_id": int(domain_id),
        "start_time": choice.start,
        "end_time": choice.end,
        "parent_i0": int(choice.window[0]),
        "parent_j0": int(choice.window[1]),
        "refinement_ratio": int(choice.ratio),
        "spec_zone": int(choice.spec_zone),
        "relax_zone": int(choice.relax_zone),
        "terrain_relief_limit_m": float(choice.terrain_limit_m),
        "map_factor_departure_limit": float(choice.map_factor_limit),
        "prepared_directory": os.fspath(prepared_directory),
        # Absent is the all-species lane (config: not projected), so a default run's deck is unchanged.
        **({} if choice.moisture == chain.DEFAULT_MOISTURE else {"moisture": choice.moisture}),
    }


def _domain_id(frame: Any) -> int:
    for key in ("GRID_ID",):
        value = frame.attributes.get(key)
        if isinstance(value, (int, float)) and int(value) >= 1:
            return int(value)
    name = str(chain.frame_source(frame).get("path") or "")
    stem = Path(name).name
    if stem.startswith("wrfout_d") and stem[8:10].isdigit():
        return int(stem[8:10])
    return 1


def _child_deck(frames: Sequence[Any], choice: ChainChoice, rundir: Path, *,
                prepare_all: bool, namelist: Path | None = None, memory_budget_mib: int | None = None,
                acknowledge: Sequence[str] = (), command: str | None = None,
                name: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Prepare the chain (every saved time, or t0 only for pricing), then import the namelist around it."""

    from .cm1run.fit import set_namelist_values
    from .cm1run.rundir import RunDirectory, derive_deck, sha256_file, write_json_atomic, write_text_atomic

    request = request_of(choice)
    used = list(frames) if prepare_all else list(frames[:2])
    if not prepare_all:
        # Pricing reads only t0's column and the grid; two frames satisfy the table layout.
        request = chain.ChainRequest(**{**request.__dict__, "end_time": used[-1].times,
                                        "duration_s": min(choice.duration_s, used[-1].valid_s - used[0].valid_s)})
    if rundir.exists():
        raise ChainDoorError(f"{rundir} already exists", "choose a new folder for the run")
    rundir.parent.mkdir(parents=True, exist_ok=True)
    # The import wants a new folder and the prepared t0 column as its sounding: prepare beside the run
    # folder, import, then move the prepared products into it.
    aside = Path(tempfile.mkdtemp(prefix=f".{rundir.name}.chain-", dir=rundir.parent))
    prepared = rundir / CHAIN_DIRECTORY
    try:
        receipt = chain.prepare_chain(used, request, aside / CHAIN_DIRECTORY,
                                      document={"by": "gpuwm-cm1 chain", "choice": choice.as_dict()})
        template = _template_namelist(namelist)
        text = template.read_text(encoding="ascii")
        text = set_namelist_values(text, namelist_values(choice))
        staged = aside / "namelist.input"
        staged.write_text(text, encoding="ascii")
        code, payload = _import(staged, rundir, aside / CHAIN_DIRECTORY / chain.SOUNDING, template,
                                acknowledge=acknowledge, command=command, memory_budget_mib=memory_budget_mib)
        (aside / CHAIN_DIRECTORY).rename(prepared)
        # The report names the namelist it read: the staged copy, gone once this returns.  Name the run
        # folder's own copy instead, so the report is the same wherever and whenever it was made.
        for report_name in ("import-report.json", "import-report.txt"):
            path = rundir / report_name
            if path.is_file():
                path.write_text(path.read_text(encoding="utf-8").replace(
                    os.fspath(staged), os.fspath(rundir / "namelist.input")), encoding="utf-8")
    finally:
        shutil.rmtree(aside, ignore_errors=True)
    run = RunDirectory(rundir)
    if not run.deck.is_file():
        refused = (run.load().get("import") or {}).get("refused") or [payload.get("verdict_text")]
        raise ChainDoorError("the chained run's settings were refused: " + "; ".join(str(r) for r in refused[:3]),
                             "see import-report.txt in the run folder")
    table = chain_table(
        choice, history_directory=_history_directory(frames), domain_id=_domain_id(frames[0]),
        prepared_directory=prepared,
    )
    deck_text = derive_deck(run.deck.read_text(encoding="utf-8"), {"chain": table}, header=(
        "# [chain]: driven by a WRF or ArWen history (gpuwm-cm1 chain new; records/GM1-ARWEN-CHAIN.md).",
    ))
    write_text_atomic(run.deck, deck_text)
    if prepare_all:
        _verify_binding(run.deck, prepared)
    record = {
        "schema": CHAIN_RUN_SCHEMA,
        "table": table,
        "choice": choice.as_dict(),
        "admission": receipt["admission"],
        "prepared": {"directory": os.fspath(prepared), "receipt": chain.RECEIPT,
                     "sounding_sha256": receipt["sounding"]["sha256"]},
        "sources": sorted({str(chain.frame_source(f).get("path") or chain.frame_source(f).get("directory") or "")
                           for f in used}),
        "verified": VERIFIED,
        "import_exit": code,
    }

    def mutate(existing: dict[str, Any]) -> None:
        existing["chain"] = record
        existing["deck"] = {"file": "deck.toml", "sha256": sha256_file(run.deck)}
        if name:
            existing["name"] = name
        existing["case"] = f"Downscaled {chain.producer_of(frames[0].attributes)} run"

    run.update(mutate)
    report_path = rundir / "import-report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["chain"] = {k: record[k] for k in ("table", "choice", "admission", "verified")}
        write_json_atomic(report_path, report)
    return run.deck, record


def _import(staged: Path, rundir: Path, sounding: Path, template: Path, *, acknowledge: Sequence[str],
            command: str | None, memory_budget_mib: int | None) -> tuple[int, dict[str, Any]]:
    """``gpuwm-cm1 import`` of the chained run's namelist, with the prepared t0 column as its sounding."""

    from .cm1import.door import run_import
    from .cm1import.namelist import NamelistToolError

    overrides: dict[str, Any] = {}
    if memory_budget_mib is not None:
        overrides["runtime.memory_budget_mib"] = int(memory_budget_mib)
    try:
        code, payload, _ = run_import(
            staged, rundir, sounding=sounding, acknowledge=list(acknowledge), command=command,
            overrides=overrides, original=template.read_bytes(),
        )
    except NamelistToolError as error:
        raise ChainDoorError(str(error), "run gpuwm-cm1 doctor --plain") from error
    return code, payload


def _history_directory(frames: Sequence[Any]) -> Path:
    source = chain.frame_source(frames[0])
    where = source.get("path") or source.get("directory") or "."
    path = Path(str(where))
    return path.parent if path.suffix or path.is_file() or _is_frame_dir(path) else path


def _verify_binding(deck: Path, prepared: Path) -> None:
    """The run's own check, on the host: the deck parses, and binds the prepared chain and sounding."""

    import tomllib

    from .config import parse_mapping

    config = parse_mapping(tomllib.loads(deck.read_text(encoding="utf-8")))
    chain.require_chain(config, sounding_bytes=(prepared / chain.SOUNDING).read_bytes())


def new_run(frames: Sequence[Any], choice: ChainChoice, out: Path | str, *, namelist: Path | None = None,
            memory_budget_mib: int | None = None, acknowledge_thompson: bool = False,
            name: str | None = None, command: str = "") -> dict[str, Any]:
    """``chain new``: admit, prepare and write a run directory ``gpuwm-cm1 run`` runs."""

    out = Path(out).expanduser().absolute()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ChainDoorError(f"{out} already exists and is not empty", "choose a new folder for the run")
    report = check(frames, choice, price_it=False, command=command)
    if not report["admitted"]:
        refused = [row for row in report["checks"] if row["status"] == "refused"]
        lines = "; ".join(f"{row['label']}: {row['text']}" for row in refused)
        raise ChainDoorError(
            (f"{len(refused)} checks do not pass. " if len(refused) > 1 else "") + lines,
            refused[0].get("remedy") or "run gpuwm-cm1 chain check for every check in plain words",
        )
    acknowledge = ["thompson"] if acknowledge_thompson and choice.component == "thompson-mp8" else []
    if choice.component in chain.ACKNOWLEDGED_COMPONENTS and not acknowledge:
        raise ChainDoorError(
            f"{chain.ACKNOWLEDGED_COMPONENTS[choice.component]}",
            "pass --acknowledge-thompson to run it anyway",
        )
    staging = out.parent / f".{out.name}.chain-staging"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        deck, record = _child_deck(frames, choice, staging, prepare_all=True,
                                   namelist=namelist, memory_budget_mib=memory_budget_mib,
                                   acknowledge=acknowledge, command=command, name=name or out.name)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # The prepared directory is named inside the deck: re-point it at the run folder's own path, then
    # move the staged run into place in one rename.
    _repoint(staging, out)
    if out.exists():
        out.rmdir()
    staging.rename(out)
    _verify_binding(out / "deck.toml", out / CHAIN_DIRECTORY)
    from .cm1run.rundir import RunDirectory

    run = RunDirectory(out)
    return {
        "schema": CHAIN_DOOR_SCHEMA, "action": "new", "run": os.fspath(out), "name": run.name(),
        "deck": os.fspath(out / "deck.toml"), "prepared": os.fspath(out / CHAIN_DIRECTORY),
        "checks": report["checks"], "admission": record["admission"], "choice": record["choice"],
        "start": f"gpuwm-cm1 run {shlex.quote(os.fspath(out))}", "verified": VERIFIED, "command": command,
    }


def _repoint(staging: Path, out: Path) -> None:
    """Rewrite the staged run's own absolute paths (deck, run.json, receipts) to the final folder."""

    from .cm1run.rundir import RunDirectory, derive_deck, sha256_file, write_text_atomic

    import tomllib

    run = RunDirectory(staging)
    deck_text = run.deck.read_text(encoding="utf-8")
    table = dict(tomllib.loads(deck_text)["chain"])
    table["prepared_directory"] = os.fspath(out / CHAIN_DIRECTORY)
    write_text_atomic(run.deck, derive_deck(deck_text, {"chain": table}))
    receipt_path = staging / CHAIN_DIRECTORY / chain.RECEIPT

    def mutate(record: dict[str, Any]) -> None:
        block = record.get("chain") or {}
        block.setdefault("table", {})["prepared_directory"] = os.fspath(out / CHAIN_DIRECTORY)
        block.setdefault("prepared", {})["directory"] = os.fspath(out / CHAIN_DIRECTORY)
        record["chain"] = block
        record["deck"] = {"file": "deck.toml", "sha256": sha256_file(run.deck)}
        text = json.dumps(record)
        record.clear()
        record.update(json.loads(text.replace(json.dumps(os.fspath(staging))[1:-1], json.dumps(os.fspath(out))[1:-1])))

    run.update(mutate)
    for name in ("import-report.json", "import-report.txt"):
        path = staging / name
        if path.is_file():
            path.write_text(path.read_text(encoding="utf-8").replace(os.fspath(staging), os.fspath(out)),
                            encoding="utf-8")
    assert receipt_path.is_file()


# ============================================================ GM1 frames as history frames


def _species_names(meaning: str) -> list[str]:
    import re

    match = re.search(r"current ([A-Za-z0-9_]+(?:,[A-Za-z0-9_]+)*) ", meaning)
    return match.group(1).split(",") if match else []


def frames_from_gm1(target: Path | str, out: Path | str, *, every_s: float | None = None,
                    start_date: str = "0001-01-01_00:00:00") -> dict[str, Any]:
    """A GM1 run's frames as history frames the chain reads (record 7.12).

    Written as ``gm1-arwen-history-extract`` writes (``frame.json`` +
    ``fields.f32le``), one per published GM1 frame, as an idealized
    (``MAP_PROJ = 0``) history: ``U/V/W`` are GM1's ``u3d/v3d/w3d``, ``T`` is
    ``th - 300``, ``P + PB`` is ``prs`` (``PB`` its level mean), ``PHB`` is
    ``g*zf`` with ``PH = 0``, ``HGT`` the terrain, ``F`` GM1's f-plane, and every
    species under its WRF name.  GM1 frames carry no ``PSFC``, ``T2`` or ``Q2``;
    the chain's own rule fills the surface line (``arwen_chain.surface_sources``).
    A translating GM1 frame is refused: its grid moves over the ground the chain
    would pin the window to.
    """

    import tomllib

    np = _np()
    from .cm1out.frames import open_frame
    from .cm1run.rundir import RunDirectory
    from .config import parse_mapping
    from .moisture_species import moisture_species_profile

    root = Path(target).expanduser()
    run = RunDirectory(root)
    if not run.is_run():
        raise ChainDoorError(f"{root} is not a GM1 run folder (no run.json)", "name a folder gpuwm-cm1 run wrote")
    deck = root / "deck.toml"
    segments = run.segments()
    if segments and segments[-1].deck.is_file():
        deck = segments[-1].deck
    config = parse_mapping(tomllib.loads(deck.read_text(encoding="utf-8")))
    component = str(config.physics.component)
    mp = chain.WRF_MP_PHYSICS.get(component)
    if mp is None:
        raise ChainDoorError(f"{component} has no WRF mp_physics option", "chain from a WSM6, Thompson or Morrison run")
    fcor = float(config.dynamics.fcor_s_inv) if int(config.dynamics.coriolis) == 1 else 0.0
    infos = run.frames()
    if not infos:
        raise ChainDoorError(f"{root} has no frames yet", "run it first")
    base = chain.parse_wrf_time(start_date)
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=False)
    written = []
    species = list(moisture_species_profile(component).names)
    for info in infos:
        t = float(info.model_time_s or 0.0)
        if every_s and abs(t / every_s - round(t / every_s)) > 1e-9:
            continue
        frame = open_frame(info.path)
        moving = frame.moving_frame or {}
        if int(moving.get("imove", 0) or 0) != 0 or float(moving.get("umove", 0.0) or 0.0) != 0.0 \
                or float(moving.get("vmove", 0.0) or 0.0) != 0.0:
            raise ChainDoorError(
                f"{info.path} was written by a translating GM1 frame (imove = 1): its grid moves over the ground, "
                "and the chain pins its window to fixed ground",
                "run the driving GM1 domain with imove = 0",
            )
        th = np.asarray(frame.array("th"), dtype=np.float32)
        prs = np.asarray(frame.array("prs"), dtype=np.float64)
        zf = np.asarray(frame.array("zf"), dtype=np.float64)
        hgt = np.asarray(frame.array("terrain_msl"), dtype=np.float32)
        pb = prs.mean(axis=(1, 2), keepdims=True) + 0.0 * prs
        fields: dict[str, Any] = {
            "U": frame.array("u3d"), "V": frame.array("v3d"), "W": frame.array("w3d"),
            "T": (th.astype(np.float64) - chain.WRF_T0_K),
            "P": prs - pb, "PB": pb,
            "PH": np.zeros_like(zf), "PHB": chain.WRF_GRAVITY_M_S2 * zf,
            "HGT": hgt, "F": np.full(hgt.shape, fcor),
        }
        q_names = _species_names(frame.carriers["q3d"].meaning) if frame.has("q3d") else []
        n_names = _species_names(frame.carriers["nc3d"].meaning) if frame.has("nc3d") else []
        q3d = np.asarray(frame.array("q3d")) if q_names else None
        nc3d = np.asarray(frame.array("nc3d")) if n_names else None
        for name in species:
            if name in q_names:
                fields[chain.WRF_SPECIES[name]] = q3d[q_names.index(name)]
            elif name in n_names:
                fields[chain.WRF_SPECIES[name]] = nc3d[n_names.index(name)]
            else:
                raise ChainDoorError(f"{info.path} carries no {name}", "chain from a run whose frames carry every species")
        dims = {}
        nz = th.shape[0]
        for key, value in fields.items():
            shape = np.shape(value)
            if len(shape) == 2:
                dims[key] = ["south_north", "west_east"]
            else:
                dims[key] = ["bottom_top_stag" if shape[0] == nz + 1 else "bottom_top",
                             "south_north_stag" if shape[1] == th.shape[1] + 1 else "south_north",
                             "west_east_stag" if shape[2] == th.shape[2] + 1 else "west_east"]
        attrs = {
            "TITLE": "GM1 (gpuwm-cm1) history as frames: gpuwm-cm1 chain frames-from-gm1",
            "DX": float(config.grid.dx_m), "DY": float(config.grid.dy_m), "MAP_PROJ": 0,
            "MP_PHYSICS": int(mp), "USE_THETA_M": 0, "GRID_ID": 1,
            "SIMULATION_START_DATE": start_date,
        }
        stamp = chain.wrf_time(base + t)
        written_frame = chain.write_frame(
            out / f"frame-{info.index:06d}", times=stamp, attributes=attrs, fields=fields, dims=dims,
            source={"format": "gm1-frame", "path": os.fspath(info.path), "record": 0,
                    "sha256": frame.arena_sha256 or "", "model_time_s": t},
        )
        written.append({"times": written_frame.times, "model_time_s": t, "directory": os.fspath(written_frame.directory)})
    return {
        "schema": CHAIN_DOOR_SCHEMA, "action": "frames-from-gm1", "run": os.fspath(root), "out": os.fspath(out),
        "component": component, "mp_physics": mp, "dx_m": float(config.grid.dx_m), "frames": written,
        "start_date": start_date, "surface_record": "the chain's rule (no PSFC, T2 or Q2 in GM1 frames)",
    }


# ============================================================ the command line


def _command(arguments: argparse.Namespace) -> str:
    words = ["gpuwm-cm1", "chain", arguments.chain_action]
    if arguments.chain_action in ("inspect",):
        words += [os.fspath(p) for p in arguments.sources]
        if arguments.map_png:
            words += ["--map-png", os.fspath(arguments.map_png)]
        if arguments.map_time:
            words += ["--map-time", arguments.map_time]
    elif arguments.chain_action == "frames-from-gm1":
        words += [os.fspath(arguments.run), "--out", os.fspath(arguments.out)]
        if arguments.every:
            words += ["--every", f"{arguments.every:g}"]
    else:
        for path in arguments.history:
            words += ["--history", os.fspath(path)]
        words += ["--window", arguments.window, "--ratio", str(arguments.ratio)]
        for flag in ("start", "end", "scheme", "moisture"):
            value = getattr(arguments, flag)
            if value is not None:
                words += [f"--{flag}", str(value)]
        for flag, attr in (("--every", "every"), ("--dt", "dt"), ("--dz", "dz"), ("--ztop", "ztop"),
                           ("--duration", "duration"), ("--output-every", "output_every"),
                           ("--terrain-limit", "terrain_limit"), ("--map-factor-limit", "map_factor_limit")):
            value = getattr(arguments, attr)
            if value is not None:
                words += [flag, f"{value:g}"]
        if arguments.chain_action == "new":
            words += ["--out", os.fspath(arguments.out)]
            if arguments.namelist:
                words += ["--namelist", os.fspath(arguments.namelist)]
            if arguments.memory_budget_mib is not None:
                words += ["--memory-budget-mib", str(arguments.memory_budget_mib)]
            if arguments.acknowledge_thompson:
                words.append("--acknowledge-thompson")
            if arguments.name:
                words += ["--name", arguments.name]
    if getattr(arguments, "json", False):
        words.append("--json")
    return shlex.join(words)


def add_command(commands: Any) -> None:
    """Register ``gpuwm-cm1 chain`` on the CLI's subparsers."""

    command = commands.add_parser(
        "chain",
        help="downscale a WRF or ArWen run: a finer GM1 domain driven by its history (3D)",
        description=(
            "A finer, flat GM1 domain inside a WRF or ArWen run: its initial state is one saved time "
            "of the run and its four sides follow the saved times after it (WRF's ndown in GM1 terms; "
            "records/GM1-ARWEN-CHAIN.md). " + VERIFIED
        ),
    )
    actions = command.add_subparsers(dest="chain_action", required=True)

    inspect_parser = actions.add_parser("inspect", help="what a history is: times, grid, map, scheme, terrain")
    inspect_parser.add_argument("sources", nargs="+", type=Path, metavar="FILE|DIR")
    inspect_parser.add_argument("--map-png", type=Path,
                                help="also draw the domain (ground, updrafts, the unusable edge) to this PNG")
    inspect_parser.add_argument("--map-time", help="the saved time the map's updrafts are drawn at (default: the first)")
    inspect_parser.add_argument("--json", action="store_true")

    for action, help_text in (("check", "every admission check in plain words, and what the run would take"),
                              ("new", "admit, prepare and write a run folder that gpuwm-cm1 run runs")):
        parser = actions.add_parser(action, help=help_text)
        parser.add_argument("--history", type=Path, action="append", required=True, metavar="FILE|DIR",
                            help="wrfout file(s), a folder of them, or extracted frame folders (repeatable)")
        parser.add_argument("--window", required=True, metavar="I0,J0,NI,NJ",
                            help="the box, in the history's mass cells: first cell (0-based) and size")
        parser.add_argument("--ratio", type=int, required=True, help="GM1 cells per history cell: odd, 3 or more")
        parser.add_argument("--start", help="first saved time (WRF Times); default: the first")
        parser.add_argument("--end", help="last saved time; default: the last")
        parser.add_argument("--every", type=float, help="use one saved time every S seconds (default: all)")
        parser.add_argument("--scheme", help="GM1 microphysics (default: the history's own)")
        parser.add_argument("--moisture", choices=tuple(chain.MOISTURE_LANES),
                            help="all-species (default: every species of the history's scheme, which GM1 must "
                                 "run) or vapour-only (WRF's have_bcs_moist = .false.: water vapour alone; "
                                 "clouds and rain start at zero; any history scheme)")
        parser.add_argument("--dt", type=float, help="GM1 time step, dividing the saving interval")
        parser.add_argument("--dz", type=float, help=f"GM1 layer depth in m (default {DEFAULT_DZ_M:g})")
        parser.add_argument("--ztop", type=float, help=f"GM1 model top in m (default {DEFAULT_ZTOP_M:g}, "
                                                       "lowered under the history's top)")
        parser.add_argument("--duration", type=float, help="run length in s (default: start to end)")
        parser.add_argument("--output-every", type=float, help="GM1 frames every S seconds (default: 300 s or the saving interval)")
        parser.add_argument("--terrain-limit", type=float,
                            help=f"declared terrain relief limit in m (default {DEFAULT_TERRAIN_LIMIT_M:g})")
        parser.add_argument("--map-factor-limit", type=float,
                            help=f"declared map-factor departure limit (default {DEFAULT_MAP_FACTOR_LIMIT:g})")
        parser.add_argument("--json", action="store_true")
        if action == "new":
            parser.add_argument("--out", type=Path, required=True, metavar="RUNDIR")
            parser.add_argument("--namelist", type=Path,
                                help="CM1 namelist for the physics and numerics (default: the 1 km supercell's)")
            parser.add_argument("--memory-budget-mib", type=int, help="GPU memory budget for the run")
            parser.add_argument("--acknowledge-thompson", action="store_true",
                                help="run Thompson microphysics, checked only in a 120-s column")
            parser.add_argument("--name", help="the run's name (default: the folder's)")

    frames_parser = actions.add_parser("frames-from-gm1", help="a GM1 run's frames as history frames the chain reads")
    frames_parser.add_argument("run", type=Path, metavar="RUNDIR")
    frames_parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    frames_parser.add_argument("--every", type=float, help="one frame every S seconds of model time")
    frames_parser.add_argument("--json", action="store_true")
    command.set_defaults(handler=chain_door)


def _print(payload: Mapping[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    action = payload.get("action")
    if action == "inspect":
        grid = payload["grid"]
        print(f"{payload['producer']} history, {payload['count']} saved time(s): {payload['times'][0]} .. {payload['times'][-1]}")
        print(f"  grid {grid['west_east']} x {grid['south_north']} x {grid['bottom_top']}, "
              f"DX {grid['dx_m']:g} m, {grid['projection']}")
        micro = payload["microphysics"]
        print(f"  microphysics {micro['scheme']}" + (f" -> GM1 {micro['component']}" if micro["component"] else
                                                     " (GM1 does not run it: --moisture vapour-only takes "
                                                     "its water vapour alone)"))
        if "terrain_m" in grid:
            print(f"  ground {grid['terrain_m']['min']:.0f} .. {grid['terrain_m']['max']:.0f} m")
        if "map_factor" in grid:
            print(f"  map factor {grid['map_factor']['min']:.4f} .. {grid['map_factor']['max']:.4f}")
    elif action in ("check", "new"):
        for row in payload["checks"]:
            mark = {"pass": "ok ", "refused": "NO ", "not-checked": " - "}[row["status"]]
            print(f"  {mark} {row['label']}: {row['text']}" + (f"  [{row['code']}]" if row.get("code") else ""))
        if action == "check":
            print(payload["summary"])
            estimate = payload.get("estimate") or {}
            if estimate and "error" not in estimate:
                print(f"  {estimate.get('cells')} cells, {estimate.get('steps')} steps")
        else:
            print(f"Made {payload['run']}. Start it with: {payload['start']}")
        print(payload["verified"])
    elif action == "frames-from-gm1":
        print(f"{len(payload['frames'])} frame(s) written to {payload['out']}")


def chain_door(arguments: argparse.Namespace) -> int:
    command = _command(arguments)
    try:
        if arguments.chain_action == "inspect":
            frames = open_frames(arguments.sources)
            payload = inspect(frames, sources=arguments.sources)
            if arguments.map_png:
                shown = frames[0]
                if arguments.map_time:
                    shown = select_frames(frames, start=arguments.map_time, end=arguments.map_time, every_s=None)[0]
                payload["map"] = domain_map_png(shown, arguments.map_png)
        elif arguments.chain_action == "frames-from-gm1":
            payload = frames_from_gm1(arguments.run, arguments.out, every_s=arguments.every)
        else:
            frames = open_frames(arguments.history)
            choice, chosen = make_choice(
                frames, window=arguments.window, ratio=arguments.ratio, start=arguments.start, end=arguments.end,
                every_s=arguments.every, scheme=arguments.scheme, dz_m=arguments.dz, ztop_m=arguments.ztop,
                dt_s=arguments.dt, duration_s=arguments.duration, output_s=arguments.output_every,
                terrain_limit_m=arguments.terrain_limit, map_factor_limit=arguments.map_factor_limit,
                moisture=arguments.moisture,
            )
            if arguments.chain_action == "check":
                payload = check(chosen, choice, command=command)
            else:
                payload = new_run(chosen, choice, arguments.out, namelist=arguments.namelist,
                                  memory_budget_mib=arguments.memory_budget_mib,
                                  acknowledge_thompson=arguments.acknowledge_thompson,
                                  name=arguments.name, command=command)
    except ChainDoorError as error:
        message = {"schema": CHAIN_DOOR_SCHEMA, "action": arguments.chain_action, "error": str(error),
                   "fix": error.fix, "command": command}
        print(json.dumps(message) if getattr(arguments, "json", False) else f"gpuwm-cm1 chain: {error}" +
              (f"\n  What to do: {error.fix}" if error.fix else ""), file=sys.stderr if not arguments.json else sys.stdout)
        return 2
    except CM1Refusal as refusal:
        detail = refusal.detail.as_dict()
        print(json.dumps({"schema": CHAIN_DOOR_SCHEMA, "action": arguments.chain_action, "refusal": detail,
                          "command": command}) if getattr(arguments, "json", False) else
              f"gpuwm-cm1 chain: {detail['code']}: {detail['reason']}\n  What to do: {detail['remedy']}",
              file=sys.stderr if not getattr(arguments, "json", False) else sys.stdout)
        return 2
    payload["command"] = command
    _print(payload, getattr(arguments, "json", False))
    if arguments.chain_action == "check":
        return 0 if payload["admitted"] else 3
    return 0


__all__ = [
    "CHAIN_DOOR_SCHEMA", "ChainChoice", "ChainDoorError", "DEFAULT_MAP_FACTOR_LIMIT", "DEFAULT_TERRAIN_LIMIT_M",
    "VERIFIED", "add_command", "chain_door", "check", "choose_step", "domain_map_png", "frames_from_gm1",
    "history_paths", "inspect", "make_choice", "new_run", "open_frames", "request_of", "select_frames",
]
