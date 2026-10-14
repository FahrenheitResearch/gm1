"""Compare runs: two to six runs side by side at one storm time, on one set of time-series charts, and the soundings
each started in beside the storm it grew (``gpuwm-cm1 quicklook RUNDIR --compare OTHER [OTHER ...]``).

Three looks, drawn on the CPU at nice 19 with no GPU (:mod:`.compare_draw`), written to
``RUNDIR/quicklook/compare/<set>/`` with ``compare.json`` (``gm1.quicklook-compare/v1``):

``storm``      each run's storm at one storm time, in boxes of one size, each centred on its own storm: reflectivity
               at 1 km (or, when a run has none, the clouds seen from above for every run), and the lowest level's
               θ′ and wind; a hurricane run's wind at the lowest level in place of θ′
``series``     every run's time series on one set of charts
``soundings``  each run's sounding (skew-T, hodograph, and the numbers the run itself computed before it started)
               beside the storm it grew, up to three runs to a card

The rules, beside the quick-looks' own (:mod:`.products`):

* one storm time ``T``: ``--time``, else the latest time every run reached; each run is drawn at its own saved time
  nearest ``T``, and a picture whose runs were saved at different times prints each run's own time;
* each box is centred on its own run's storm: the rotating updraft (the centre of the 2-5 km updraft helicity above a
  quarter of its peak, within 6 km of the peak, when the peak is at least :data:`UH_ROTATING`); else the strongest
  updraft at 5 km (at least :data:`W_UPDRAFT`), found the same way; else, for a storm with no updraft left, its
  strongest reflectivity at 1 km (at least :data:`DBZ_ECHO`); else the middle of the run's area.  A hurricane
  run's box is centred on its lowest pressure at the lowest level.  Each panel says which;
* one box size for every run, so the pictures share one scale: 48 km (500 km for hurricanes), or the largest run's
  shorter side when that is smaller; what lies outside a run's own area is hatched;
* runs that differ in grid, area, levels or save times are drawn as they are: each panel names its grid, its lowest
  level and its own time, and the picture lists the settings that differ between the runs;
* a run whose saved frames were deleted (Results: Keep pictures only) is drawn from its own quick-look picture of that
  look at that time, cropped around its strongest updraft at 5 km from its picture records, and its panel says so;
  its time series comes from its ``series.json`` and its sounding from its ``input_sounding`` file;
* a hurricane run is compared only with hurricane runs: their looks and ranges differ from a thunderstorm's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time as _time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..cm1out import frames as _frames
from ..cm1out import settings as _settings
from . import products as P
from . import style as S
from .facts import RunFacts, run_facts
from .frame import FrameUnreadable, QuickFrame, frames_of

SCHEMA = "gm1.quicklook-compare/v1"
PROGRESS_SCHEMA = "gm1.quicklook-compare-progress/v1"
#: The compare looks, as ``--only`` names them.
LOOKS = ("storm", "series", "soundings")
MAX_RUNS = 6
LETTERS = "ABCDEF"
#: The dataviz reference palette's categorical slots for a dark surface, in its fixed order (checked on the
#: quick-looks' black with the dataviz validator: every adjacent pair clears the colour-blind and normal-vision
#: floors).  A run keeps its colour on every compare picture.
COLOURS = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300")
STORM_BOX = 48_000.0         # m
HURRICANE_BOX = 500_000.0    # m
SMALLEST_BOX = 2_000.0       # m
#: 2-5 km updraft helicity (m² s⁻²) from which an updraft is called rotating, for the centre of its box.
UH_ROTATING = 75.0
#: The updraft at 5 km (m/s) from which a run without a rotating updraft is centred on it.
W_UPDRAFT = 5.0
#: The reflectivity at 1 km (dBZ) from which a run with no updraft left is centred on its heaviest rain.
DBZ_ECHO = 20.0
CENTRE_RADIUS = 6_000.0             # m around the peak that its centre is taken over
HURRICANE_CENTRE_RADIUS = 100_000.0  # m
HURRICANE_CENTRE_HPA = 1.0          # hPa above the lowest pressure that the centre is taken over
FAINT = (70, 80, 102)               # the quick-looks' panel outline (style token "faint")
#: CM1's iinit, in plain words (README.namelist).
STARTS = {0: "no start perturbation", 1: "warm bubble", 2: "cold pool", 3: "line of warm air", 7: "hurricane swirl",
          12: "push of rising air (updraft nudging)"}
#: Settings described in their own words when they differ; any other setting that changes the storm is named by
#: its CM1 name.
GRID_KEYS = frozenset({"nx", "ny", "nz", "dx", "dy", "dz", "stretch_x", "stretch_y", "stretch_z", "str_bot",
                       "str_top", "dz_bot", "dz_top", "ztop", "dx_inner", "dx_outer", "dy_inner", "dy_outer",
                       "nos_x_len", "nos_y_len", "tot_x_len", "tot_y_len"})
SOUNDING_KEYS = frozenset({"isnd", "iwnd"})
START_KEYS = frozenset({"iinit"})
PHYSICS_KEYS = frozenset({"ptype", "ihail"})
#: The moving domain (CM1's imove, umove, vmove), said as such.
MOTION_KEYS = frozenset({"imove", "umove", "vmove"})
#: Settings named in plain words, with their CM1 name, when they differ.
NAMED_KEYS = {"dtl": "time step (dtl)"}
#: CM1's parcels (iprcl, nparcels, prcl_*): air followed through the run, which does not change the storm.
PARCEL_KEYS = frozenset({"iprcl", "nparcels"})
PARCEL_PREFIX = "prcl_"


class CompareError(ValueError):
    """The runs cannot be compared; the message says why, in plain words."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _utc() -> str:
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())


def tag(t: float) -> str:
    return f"t{int(round(t)):06d}"


# ----------------------------------------------------------------------------------------------- one run
@dataclass(eq=False)
class Run:
    """One run of a comparison: where it is, its grid, its saved times and its records."""

    letter: str
    colour: str
    root: Path
    layout: Any                       # cm1out.frames.RunLayout, or None when its frames were deleted
    lineage: list                     # [(time, key, frame directory, segment)]
    index: dict | None                # quicklook/index.json
    series_doc: dict | None           # quicklook/series.json
    settings: Any                     # cm1out.settings.Settings, or None
    record: dict                      # run.json, or {}
    facts: RunFacts
    name: str
    dx: float
    dy: float
    nx: int
    ny: int
    nz: int
    z0: float
    stretched: bool
    times: list[float] = field(default_factory=list)

    @property
    def hurricane(self) -> bool:
        return self.facts.storm_kind == "hurricane"

    @property
    def has_frames(self) -> bool:
        return bool(self.lineage)

    @property
    def extent(self) -> tuple[float, float]:
        return self.nx * self.dx, self.ny * self.dy

    @property
    def quicklook(self) -> Path:
        return self.root / "quicklook"

    @property
    def interval(self) -> float | None:
        gaps = sorted(b - a for a, b in zip(self.times, self.times[1:]) if b > a)
        return gaps[len(gaps) // 2] if gaps else None

    def nearest(self, t: float) -> float:
        return min(self.times, key=lambda s: (abs(s - t), s))

    def frame_at(self, t: float) -> tuple[float, str, Path, int]:
        return min(self.lineage, key=lambda row: (abs(row[0] - t), row[0]))

    def setting(self, key: str) -> Any:
        if self.settings is None:
            return None
        try:
            return self.settings.values.get(key)
        except AttributeError:
            return None

    # -- words ---------------------------------------------------------------------
    def grid_words(self) -> str:
        lx, ly = self.extent
        levels = f"{self.nz} levels" + (", stretched" if self.stretched else "")
        return S.data_line(f"{S.fmt_grid(self.dx)} grid", f"{km_words(lx)} × {km_words(ly)} km", levels)

    def case_words(self) -> str:
        case = self.record.get("case") if isinstance(self.record, dict) else None
        return str(case) if case else ""

    def start_words(self) -> str:
        value = self.setting("iinit")
        try:
            number = int(value)
        except (TypeError, ValueError):
            return ""
        return STARTS.get(number, f"start iinit = {number}")

    def interval_words(self) -> str:
        return f"saves every {duration_words(self.interval)}" if self.interval else "one save"

    def sounding_identity(self) -> str:
        """What tells this run's sounding from another's: its file's SHA-256, else CM1's analytic choice."""

        sounding = self.record.get("sounding") if isinstance(self.record, dict) else None
        if isinstance(sounding, dict) and sounding.get("sha256"):
            return f"file {sounding['sha256']}"
        path = self.root / "input_sounding"
        isnd = self.setting("isnd")
        if path.is_file() and str(isnd) in ("7", "17"):
            return "file " + hashlib.sha256(path.read_bytes()).hexdigest()
        return f"isnd {isnd} iwnd {self.setting('iwnd')}"


def km_words(metres: float) -> str:
    """A length in km as a picture says it: whole km as they are (120, 1536), else three figures (49.3, 20.6),
    with one decimal from 100 km up (100.5)."""

    km = float(metres) / 1000.0
    if abs(km - round(km)) < 0.005:
        return f"{round(km):d}"
    if abs(km) >= 100.0:
        return f"{km:.1f}"
    return f"{float(f'{km:.3g}'):g}"


def duration_words(seconds: float | None) -> str:
    if not seconds:
        return ""
    s = float(seconds)
    if s >= 3600 and s % 3600 == 0:
        return f"{s / 3600:g} h"
    if s >= 60 and s % 60 == 0:
        return f"{s / 60:g} min"
    return f"{s:g} s"


def _grid_from_index(index: Mapping[str, Any]) -> tuple[int, int, int] | None:
    found = re.match(r"(\d+) × (\d+) × (\d+)", str(((index or {}).get("run") or {}).get("grid") or ""))
    return tuple(int(found.group(n)) for n in (1, 2, 3)) if found else None  # type: ignore[return-value]


def opened_cm1_output(target: Path) -> tuple[Path, str] | None:
    """CM1 output opened for pictures (Runs: Open CM1 output): (where its files are, its name), from the folder's
    ``cm1-output.json``; that folder holds only the pictures and their records.  None for anything else, or when
    the files are no longer where it says."""

    doc = _read_json(Path(target) / "cm1-output.json") if Path(target).is_dir() else None
    if not isinstance(doc, dict) or doc.get("schema") != "gm1.cm1-output/v1" or not isinstance(doc.get("source"), str):
        return None
    source = Path(doc["source"])
    return (source, str(doc.get("name") or Path(target).name)) if source.exists() else None


def load_run(target: Path | str, letter: str, colour: str) -> Run:
    """A run folder, a ``gpuwm-cm1 cartesian`` output folder, a history folder, or CM1 netCDF output (the files, or
    a folder that opened them for pictures), read for comparing."""

    from .ncread import discover

    target = Path(target)
    if not target.exists():
        raise CompareError(f"{target} does not exist.")
    opened = opened_cm1_output(target)
    layout = None
    lineage: list = []
    try:
        # a GM1 run's frames, or CM1's netCDF files read as every look reads them
        layout = discover(opened[0] if opened else target)
        lineage = frames_of(layout)
    except _frames.FrameError:
        layout = None
    if opened is not None:
        root = target
    elif layout is not None:
        root = layout.root.parent if layout.kind == "frames" else layout.root
    else:
        root = target
    index = _read_json(root / "quicklook" / "index.json")
    index = index if isinstance(index, dict) else None
    series_doc = _read_json(root / "quicklook" / "series.json")
    series_doc = series_doc if isinstance(series_doc, dict) and series_doc.get("charts") else None
    record = _read_json(root / "run.json")
    record = record if isinstance(record, dict) else {}
    try:
        settings = (layout.settings if layout is not None and layout.kind == "netcdf" else
                    _settings.resolve(layout if layout is not None else _frames.RunLayout("rundir", root, ())))
    except (_settings.SettingsError, _frames.FrameError, OSError, ValueError, KeyError, TypeError):
        settings = None
    if lineage:
        qf = QuickFrame.open(lineage[-1][2], key=lineage[-1][1], segment=lineage[-1][3])
        try:
            facts = run_facts(layout, qf.config, qf.dx, qf.dy, settings=settings)
            dx, dy, nx, ny, nz = qf.dx, qf.dy, qf.nx, qf.ny, qf.nz
            z0, stretched = float(qf.zh[0]), bool(qf.stretched)
        finally:
            qf.close()
        times = [row[0] for row in lineage]
    else:
        shown = (index or {}).get("run") or {}
        grid = _grid_from_index(index or {})
        times = sorted({float(f["time_s"]) for f in (index or {}).get("frames") or []
                        if isinstance(f, dict) and isinstance(f.get("time_s"), (int, float))})
        if grid is None or not times or not shown.get("dx_m"):
            raise CompareError(f"{target.name}: no saved frames and no quick-look pictures, so there is nothing of "
                               "its storm to compare.")
        nx, ny, nz = grid
        dx, dy = float(shown["dx_m"]), float(shown.get("dy_m") or shown["dx_m"])
        z0 = float(shown.get("lowest_level_m") or 0.0)
        stretch = (settings.values.get("stretch_z") if settings is not None else None)
        stretched = str(stretch) not in ("0", "None", "")
        facts = RunFacts(name=str(shown.get("name") or record.get("name") or root.name),
                         version=str(shown.get("gm1_version") or "(not recorded)"),
                         version_source=str(shown.get("gm1_version_source") or ""),
                         scheme=str(shown.get("microphysics") or "unknown microphysics"),
                         component=str(shown.get("component") or ""), checked=str(shown.get("checked") or ""),
                         checked_source=str(shown.get("checked_source") or ""), dx_m=dx, dy_m=dy,
                         iorigin=int(shown.get("iorigin") or 1), storm_kind=str(shown.get("storm_kind") or ""))
    name = str(record.get("name") or (opened[1] if opened else "") or facts.name or root.name)
    return Run(letter=letter, colour=colour, root=root, layout=layout, lineage=lineage, index=index,
               series_doc=series_doc, settings=settings, record=record, facts=facts, name=name, dx=dx, dy=dy,
               nx=nx, ny=ny, nz=nz, z0=z0, stretched=stretched, times=times)


def load_runs(targets: Sequence[Path | str]) -> list[Run]:
    if len(targets) < 2:
        raise CompareError("Name at least one other run to compare with.")
    if len(targets) > MAX_RUNS:
        raise CompareError(f"At most {MAX_RUNS} runs can be compared at once (one picture holds six side by side); "
                           f"{len(targets)} were named.")
    seen: dict[str, str] = {}
    for target in targets:
        key = os.path.realpath(target)
        if key in seen:
            raise CompareError(f"{Path(target).name} is named twice; compare it with other runs.")
        seen[key] = str(target)
    runs = [load_run(t, LETTERS[n], COLOURS[n]) for n, t in enumerate(targets)]
    names = [r.name for r in runs]
    for r in runs:
        if names.count(r.name) > 1:
            # two runs with one name (two folders each holding "your-environment"): the parent folder tells them apart
            r.name = f"{r.root.parent.name}/{r.name}"
    kinds = {r.hurricane for r in runs}
    if len(kinds) > 1:
        which = ", ".join(r.letter for r in runs if r.hurricane)
        raise CompareError(f"A hurricane run ({which}) is compared only with other hurricane runs: its looks and "
                           "ranges are not a thunderstorm's.")
    return runs


# ----------------------------------------------------------------------------------------------- what differs
def differences(runs: Sequence[Run]) -> tuple[list[str], list[str]]:
    """(what differs between the runs, in plain words; runs whose settings could not be read)."""

    words: list[str] = []

    def differ(values: list[Any]) -> bool:
        return len({json.dumps(v, sort_keys=True, default=str) for v in values}) > 1

    if differ([round(r.dx, 3) for r in runs]) or differ([round(r.dy, 3) for r in runs]):
        words.append("grid spacing")
    if differ([(round(r.extent[0]), round(r.extent[1])) for r in runs]):
        words.append("area")
    if differ([(r.nz, round(r.z0, 1), r.stretched) for r in runs]):
        words.append("levels")
    if differ([r.sounding_identity() for r in runs]):
        words.append("sounding")
    if differ([r.setting("iinit") for r in runs]):
        words.append("start")
    if differ([r.facts.scheme for r in runs]):
        words.append("microphysics")
    if differ([r.interval for r in runs]):
        words.append("save times")
    unread = [r.letter for r in runs if r.settings is None]
    readable = [r for r in runs if r.settings is not None]
    if len(readable) >= 2:
        from .series import storm_settings

        tables = [storm_settings(r.settings.values) for r in readable]
        keys = sorted(set().union(*tables))
        if any(differ([t.get(k) for t in tables]) for k in MOTION_KEYS):
            words.append("domain motion")
        for key, said in NAMED_KEYS.items():
            if differ([t.get(key) for t in tables]):
                words.append(said)
        covered = GRID_KEYS | SOUNDING_KEYS | START_KEYS | PHYSICS_KEYS | MOTION_KEYS | PARCEL_KEYS | set(NAMED_KEYS)
        other = [k for k in keys if k not in covered and not k.startswith(PARCEL_PREFIX)
                 and differ([t.get(k) for t in tables])]
        if other:
            shown = other[:6]
            more = f" and {len(other) - 6} more" if len(other) > 6 else ""
            words.append("settings " + ", ".join(shown) + more)
    return words, unread


# ----------------------------------------------------------------------------------------------- the storm time
def storm_time(runs: Sequence[Run], asked: float | None) -> float:
    """``T``: the time asked for, else the latest time every run reached.  A time asked for after every run ended
    (more than one save interval past the last save of any of them) is refused: no run has a save near it."""

    if asked is not None:
        if not math.isfinite(asked) or asked < 0:
            raise CompareError("--time is a storm time in seconds, 0 or more.")
        last = max(runs, key=lambda r: max(r.times))
        latest = max(last.times)
        slack = max([r.interval or 0.0 for r in runs])
        if asked > latest + slack:
            clock = S.fmt_long_clock if any(r.hurricane for r in runs) else S.fmt_clock
            raise CompareError(f"The storm time {clock(asked)} is after every run ended: the last save of any of "
                               f"them is {last.letter} at {clock(latest)}.")
        return float(asked)
    return float(min(max(r.times) for r in runs))


# ----------------------------------------------------------------------------------------------- the centres
def _centroid(field: np.ndarray, j0: int, i0: int, radius_m: float, dx: float, dy: float,
              floor: float) -> tuple[float, float]:
    """The value-weighted centre (m, the run's x and y from its south-west corner) of the cells at or above
    ``floor`` within ``radius_m`` of cell (j0, i0)."""

    ny, nx = field.shape
    ri, rj = int(math.ceil(radius_m / dx)), int(math.ceil(radius_m / dy))
    j_lo, j_hi, i_lo, i_hi = max(0, j0 - rj), min(ny, j0 + rj + 1), max(0, i0 - ri), min(nx, i0 + ri + 1)
    sub = np.asarray(field[j_lo:j_hi, i_lo:i_hi], dtype=np.float64)
    jj, ii = np.mgrid[j_lo:j_hi, i_lo:i_hi]
    near = ((ii - i0) * dx) ** 2 + ((jj - j0) * dy) ** 2 <= radius_m ** 2
    weight = np.where(near & (sub >= floor), sub, 0.0)
    total = float(weight.sum())
    if total <= 0.0:
        return (i0 + 0.5) * dx, (j0 + 0.5) * dy
    return (float((weight * (ii + 0.5)).sum() / total) * dx, float((weight * (jj + 0.5)).sum() / total) * dy)


@dataclass
class Centre:
    x: float              # m from the run's south-west corner
    y: float
    kind: str             # rotation | updraft | rain | middle | pressure | records
    words: str            # what the panel says


def storm_centre(qf: QuickFrame, hurricane: bool, dbz_1km: np.ndarray | None = None) -> Centre:
    """The centre of a run's box at one saved time (see the module's rules).  ``dbz_1km``: the frame's reflectivity
    at 1 km, for a storm with no updraft left (centred on its heaviest rain)."""

    lx, ly = qf.extent
    middle = Centre(0.5 * lx, 0.5 * ly, "middle", "")
    if hurricane:
        p = qf.pressure_level(0)
        if p is None:
            middle.words = "centre: the middle of the area (no pressure saved)"
            return middle
        j, i = np.unravel_index(int(np.argmin(p)), p.shape)
        depth = (float(p.min()) + HURRICANE_CENTRE_HPA * 100.0) - p.astype(np.float64)
        x, y = _centroid(depth, int(j), int(i), HURRICANE_CENTRE_RADIUS, qf.dx, qf.dy, 0.0)
        return Centre(x, y, "pressure", "centre: the lowest pressure at the lowest level")
    uh, _source = qf.uh()
    peak = float(uh.max())
    if peak >= UH_ROTATING:
        j, i = np.unravel_index(int(np.argmax(uh)), uh.shape)
        x, y = _centroid(uh, int(j), int(i), CENTRE_RADIUS, qf.dx, qf.dy, 0.25 * peak)
        return Centre(x, y, "rotation", f"centre: the rotating updraft (2–5 km updraft helicity, peak "
                                        f"{peak:,.0f} m² s⁻²)")
    k5 = qf.level_near(5000.0, faces=True)
    w5 = qf.get("w3d", k5)
    top = float(w5.max())
    if top >= W_UPDRAFT:
        j, i = np.unravel_index(int(np.argmax(w5)), w5.shape)
        x, y = _centroid(w5, int(j), int(i), CENTRE_RADIUS, qf.dx, qf.dy, 0.5 * top)
        return Centre(x, y, "updraft", f"centre: the strongest updraft at 5 km (helicity under {UH_ROTATING:g} m² s⁻²)")
    if dbz_1km is not None and float(np.max(dbz_1km)) >= DBZ_ECHO:
        heaviest = float(np.max(dbz_1km))
        j, i = np.unravel_index(int(np.argmax(dbz_1km)), dbz_1km.shape)
        x, y = _centroid(dbz_1km, int(j), int(i), CENTRE_RADIUS, qf.dx, qf.dy, heaviest - 10.0)
        return Centre(x, y, "rain", f"centre: the strongest reflectivity at 1 km (no updraft of {W_UPDRAFT:g} m/s "
                                    "at 5 km)")
    middle.words = f"centre: the middle of the area (no updraft of {W_UPDRAFT:g} m/s at 5 km)"
    return middle


# ----------------------------------------------------------------------------------------------- one run at T
@dataclass(eq=False)
class StormState:
    """What the storm pictures need of one run at its saved time nearest ``T``: 2-D planes from its frame, or its
    own quick-look pictures when its frames were deleted."""

    run: Run
    t: float
    centre: Centre
    source: str                                   # "frames" or "pictures"
    planes: dict[str, np.ndarray] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    pictures: dict[str, tuple[np.ndarray, P.Box]] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)   # look -> why it is not drawn


#: The quick-look each compare panel can be cropped from, for a run whose frames were deleted.
CROP_LOOK = {"reflectivity": "reflectivity", "clouds": "clouds-from-above", "surface": "surface",
             "wind": "wind-lowest"}


def picture_box(look: str, extent: tuple[float, float], hurricane: bool) -> P.Box:
    """Where a quick-look's map of the whole area sits in its own 1920 x 1080 picture (:mod:`.products`)."""

    if hurricane:
        return P.hurricane_boxes(extent)[0]
    if look == "reflectivity":
        return P.dual_boxes(extent)[0]
    return P.single_box(extent)


def outline_holds(rgb: np.ndarray, box: P.Box) -> bool:
    """Whether the picture's panel outline lies where :func:`picture_box` says (the picture's layout is this
    version's)."""

    h, w = rgb.shape[:2]
    points = ((box.x - 1, box.y + box.h // 2), (box.x + box.w, box.y + box.h // 2),
              (box.x + box.w // 2, box.y - 1), (box.x + box.w // 2, box.y + box.h))
    for x, y in points:
        if not (0 <= x < w and 0 <= y < h):
            return False
        if np.abs(rgb[y, x, :3].astype(int) - np.array(FAINT)).max() > 6:
            return False
    return True


def _index_frame(run: Run, look: str, t: float) -> dict | None:
    for product in (run.index or {}).get("products") or []:
        if isinstance(product, dict) and product.get("id") == look:
            frames = [f for f in product.get("frames") or [] if isinstance(f, dict) and f.get("file")]
            if frames:
                return min(frames, key=lambda f: (abs(float(f.get("time_s") or 0.0) - t), float(f.get("time_s") or 0)))
    return None


def state_from_pictures(run: Run, t: float, looks: Sequence[str]) -> StormState:
    """A run whose frames were deleted: its own pictures of ``looks`` at its time nearest ``t``."""

    from PIL import Image

    when = run.nearest(t)
    records = "pressure-lowest" if run.hurricane else "updraft-5km"
    row = _index_frame(run, records, when) or {}
    facts = row.get("facts") or {}
    xy = facts.get("p_min_xy_m" if run.hurricane else "w_max_xy_m")
    w_max = facts.get("w_max")
    lx, ly = run.extent
    if not run.hurricane and isinstance(w_max, (int, float)) and w_max < W_UPDRAFT:
        centre = Centre(0.5 * lx, 0.5 * ly, "middle",
                        f"centre: the middle of the area (no updraft of {W_UPDRAFT:g} m/s at 5 km)")
    elif isinstance(xy, list) and len(xy) == 2:
        centre = Centre(float(xy[0]), float(xy[1]), "records",
                        "centre: the lowest pressure, from its picture records" if run.hurricane else
                        "centre: the strongest updraft at 5 km, from its picture records")
    else:
        centre = Centre(0.5 * lx, 0.5 * ly, "middle", "centre: the middle of the area (no picture records)")
    state = StormState(run, when, centre, "pictures")
    for look in looks:
        frame = _index_frame(run, CROP_LOOK[look], when)
        if frame is None or abs(float(frame.get("time_s") or 0.0) - when) > 0.5:
            state.missing[look] = "its saved frames were deleted and it has no picture of this look at this time"
            continue
        path = run.quicklook / str(frame["file"])
        try:
            rgb = np.asarray(Image.open(path).convert("RGB"))
        except OSError:
            state.missing[look] = "its saved frames were deleted and its picture of this look cannot be read"
            continue
        box = picture_box(CROP_LOOK[look], run.extent, run.hurricane)
        if not outline_holds(rgb, box):
            state.missing[look] = ("its saved frames were deleted and its pictures were drawn in another layout, "
                                   "so they cannot be cropped")
            continue
        state.pictures[look] = (rgb, box)
        facts = frame.get("facts") or {}
        for key in ("dbz_max_1km", "theta_min", "wind_max_ground_relative"):
            if isinstance(facts.get(key), (int, float)):
                state.values[key] = float(facts[key])
    return state


def state_from_frames(run: Run, t: float, looks: Sequence[str]) -> StormState:
    """A run's frame nearest ``t``: its centre and the 2-D planes the looks draw."""

    when, key, path, segment = run.frame_at(t)
    qf = QuickFrame.open(path, key=key, segment=segment)
    try:
        L = P.Looks(qf, run.facts)
        planes: dict[str, np.ndarray] = {}
        notes: dict[str, Any] = {}
        missing: dict[str, str] = {}
        if "reflectivity" in looks:
            refl = L.reflectivity
            if refl.available:
                one, (lo, hi) = L.dbz_1km
                planes["dbz"] = one
                notes["dbz_operator"] = refl.operator
                notes["dbz_levels"] = [lo, hi]
            else:
                missing["reflectivity"] = refl.reason or "no reflectivity for this microphysics"
        centre = storm_centre(qf, run.hurricane, planes.get("dbz"))
        state = StormState(run, when, centre, "frames", planes=planes, notes=notes, missing=missing)
        if "clouds" in looks:
            from . import satellite as SA

            species = SA.species_present(qf)
            if any(name in species for name in ("qc", "qi")):
                light = SA.light(SA.extinction(qf, species), qf.zf, qf.dx, qf.dy)
                for name in ("reflect", "lit", "ground_sun"):
                    state.planes[f"cloud_{name}"] = light[name]
                state.notes["cloud_species"] = species
            else:
                state.missing["clouds"] = "this run saved no cloud water or ice"
        if "surface" in looks or "wind" in looks:
            u, v = L.uv0_ground
            state.planes["u0"], state.planes["v0"] = u, v
        if "surface" in looks:
            state.planes["theta0"] = L.theta0
            state.planes["conv0"] = L.conv0
        state.notes["domain_moves"] = bool(qf.domain_moves)
    finally:
        qf.close()
    return state


def reflectivity_possible(run: Run, t: float) -> bool:
    """Whether a run can be drawn in reflectivity at its time nearest ``t`` (without computing it)."""

    from . import reflectivity as R

    if not run.has_frames:
        return _index_frame(run, "reflectivity", run.nearest(t)) is not None
    when, key, path, segment = run.frame_at(t)
    qf = QuickFrame.open(path, key=key, segment=segment)
    try:
        return bool(R.possible(qf))
    finally:
        qf.close()


# ----------------------------------------------------------------------------------------------- the time series
#: The charts of the compare series, in order: (key, label, unit).  The lowest level is each run's own.
STORM_CHARTS: tuple[tuple[str, str, str], ...] = (
    ("wmax", "strongest updraft", "m/s"),
    ("uh_max", "max 2–5 km updraft helicity", "m² s⁻²"),
    ("wind_sfc_max", "fastest wind at the lowest level, ground-relative", "m/s"),
    ("dp_sfc_max", "deepest pressure drop at the lowest level", "hPa"),
    ("thpert_sfc_min", f"coldest θ{S.PRIME} at the lowest level", "K"),
    ("rain_max_cm", "most rain in one cell", "cm"),
)
HURRICANE_CHARTS: tuple[tuple[str, str, str], ...] = (
    ("prs_sfc_min_hpa", "lowest pressure at the lowest level (not sea level)", "hPa"),
    ("wind_sfc_max", "fastest wind at the lowest level, ground-relative", "m/s"),
    ("wmax", "strongest updraft", "m/s"),
    ("rain_max_cm", "most rain in one cell", "cm"),
    ("dbz_max", "highest reflectivity", "dBZ"),
    ("wmin", "strongest downdraft", "m/s"),
)


def pressure_drop_hpa(qf: QuickFrame) -> float | None:
    """How far the pressure at the lowest level falls below the base state there, at most (hPa), or None.

    The base state is the frame's prs0 or pi0 column when it carries one; else, cell by cell, CM1's
    ``p00 * (pi - pp3d) ** (cp/rd)`` with ``pi = (prs/p00) ** (rd/cp)`` (the frame's total pressure and its
    perturbation Exner function), which is the same base state exactly."""

    from ..cm1out.fields import _CP, _P00, _RD

    p = qf.pressure_level(0)
    if p is None:
        return None
    p = p.astype(np.float64)
    prs0, pi0 = qf.base_column("prs0"), qf.base_column("pi0")
    if prs0 is not None:
        base: Any = float(prs0[0])
    elif pi0 is not None:
        base = _P00 * float(pi0[0]) ** (_CP / _RD)
    elif qf.has("pp3d"):
        pi_total = np.power(p / _P00, _RD / _CP)
        base = _P00 * np.power(np.maximum(pi_total - qf.get("pp3d", 0).astype(np.float64), 1.0e-6), _CP / _RD)
    else:
        return None
    return float(np.max((base - p) / 100.0))


def series_points(run: Run, log: Callable[[str], None]) -> tuple[dict[str, list[list[float]]], str]:
    """A run's time series: {key: [[t, value], ...]} and where it came from.

    The run's own ``series.json`` (what its Results screen charts) when it has one; else each saved frame's domain
    extremes, computed here as the quick-look worker computes them.  The thunderstorm pressure drop, which the
    one-run series does not chart, is read from the saved frames."""

    points: dict[str, list[list[float]]] = {}
    source = ""
    if run.series_doc is not None:
        for chart in run.series_doc.get("charts") or []:
            if isinstance(chart, dict) and isinstance(chart.get("gm1"), list):
                points[str(chart.get("key"))] = [[float(p[0]), float(p[1])] for p in chart["gm1"]
                                                 if isinstance(p, list) and len(p) == 2]
        source = "its series.json"
    needs_frames = run.has_frames and (not points or not run.hurricane)
    if not needs_frames:
        return points, source or "no series"
    computed: dict[str, list[list[float]]] = {}
    for t, key, path, segment in run.lineage:
        qf = QuickFrame.open(path, key=key, segment=segment)
        try:
            if not points:
                try:
                    stats = P.Looks(qf, run.facts).statistics()
                except (FrameUnreadable, ValueError, KeyError) as error:
                    log(f"{run.letter}: t = {t:g} s statistics not read: {error}")
                    stats = {}
                for k, value in stats.items():
                    computed.setdefault(k, []).append([t, float(value)])
            if not run.hurricane:
                drop = pressure_drop_hpa(qf)
                if drop is not None:
                    computed.setdefault("dp_sfc_max", []).append([t, drop])
        finally:
            qf.close()
    if not points:
        source = "its saved frames"
    points.update(computed)
    return points, source


# ----------------------------------------------------------------------------------------------- soundings
def _profile_from_file(path: Path) -> dict[str, np.ndarray]:
    """A CM1 ``input_sounding`` (base.F isnd = 7: a surface line of pressure in hPa, theta in K and qv in g/kg, then
    z, theta, qv, u, v) as a profile: pressure from the surface pressure, integrated hydrostatically with the
    virtual potential temperature, as base.F does; the surface wind extrapolated from the first two levels
    (base.F:617-618)."""

    from .skewt import CP, P00, RD, dewpoint_k

    rows = []
    head = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if not words:
            continue
        if head is None:
            head = [float(w) for w in words[:3]]
            continue
        if len(words) >= 5:
            rows.append([float(w) for w in words[:5]])
    if head is None or len(rows) < 2:
        raise ValueError(f"{path.name} has no surface line and two levels")
    z = np.array([0.0] + [r[0] for r in rows])
    th = np.array([head[1]] + [r[1] for r in rows])
    qv = np.array([head[2]] + [r[2] for r in rows]) / 1000.0
    u = np.array([0.0] + [r[3] for r in rows])
    v = np.array([0.0] + [r[4] for r in rows])
    span = z[2] - z[1]
    u[0] = u[1] - (z[1] - z[0]) * (u[2] - u[1]) / span
    v[0] = v[1] - (z[1] - z[0]) * (v[2] - v[1]) / span
    thv = th * (1.0 + 0.61 * qv)
    pi = np.empty_like(z)
    pi[0] = (head[0] * 100.0 / P00) ** (RD / CP)
    for k in range(1, z.size):
        pi[k] = pi[k - 1] - 9.81 * (z[k] - z[k - 1]) / (CP * 0.5 * (thv[k] + thv[k - 1]))
    pi = np.maximum(pi, 1.0e-3)
    p = P00 * pi ** (CP / RD)
    t = th * pi
    return {"z": z, "p": p, "t": t, "td": dewpoint_k(p, qv), "u": u, "v": v, "qv": qv}


def profile(run: Run) -> tuple[dict[str, np.ndarray] | None, str]:
    """(the run's starting profile, where it came from): the earliest frame's corner columns, as the environment
    look draws it; else the run's ``input_sounding``; else none."""

    from . import environment as EN

    if run.has_frames:
        t, key, path, segment = run.lineage[0]
        qf = QuickFrame.open(path, key=key, segment=segment)
        try:
            prof = EN.profile(qf)
            varies = EN.air_varies(run.root)
            return prof, ("first frame, south-west corner" if varies else "first frame, undisturbed corner")
        except FrameUnreadable:
            pass
        finally:
            qf.close()
    path = run.root / "input_sounding"
    if path.is_file():
        try:
            return _profile_from_file(path), "its input_sounding file"
        except (OSError, ValueError, IndexError):
            return None, "none: its input_sounding cannot be read"
    return None, "none: frames deleted, no sounding file"


def environment(run: Run) -> tuple[dict | None, str]:
    """The numbers the run computed before it started (the environment look's)."""

    from . import environment as EN

    segments = sorted(p for p in (run.root / "segments").glob("*") if p.is_dir()) if (run.root / "segments").is_dir() \
        else []
    if run.layout is not None:
        segments = [Path(s.frames_root).parent for s in run.layout.segments] or segments
    return EN.parameters(run.root, segments)


# ----------------------------------------------------------------------------------------------- progress and the door
class Progress:
    def __init__(self, out: Path, total: int) -> None:
        self.path = out / "progress.json"
        self.doc: dict[str, Any] = {"schema": PROGRESS_SCHEMA, "state": "running", "done": 0, "total": total,
                                    "step": "", "message": "", "pid": os.getpid(), "started_utc": _utc()}

    def step(self, words: str, *, done: bool = True) -> None:
        if done and self.doc["step"]:
            self.doc["done"] = min(self.doc["total"], self.doc["done"] + 1)
        self.doc["step"] = words
        self.write()

    def finish(self, state: str, message: str = "") -> None:
        self.doc.update({"state": state, "message": message, "step": ""})
        if state == "done":
            self.doc["done"] = self.doc["total"]
        self.write()

    def write(self) -> None:
        self.doc["updated_utc"] = _utc()
        try:
            _write_json(self.path, self.doc)
        except OSError:
            pass


def compare_folder(targets: Sequence[Path | str]) -> str:
    """The comparison's own folder name under ``quicklook/compare/``: the other runs' folder names and a short hash
    of where every run is, so comparing with another set never writes over this one."""

    paths = [Path(os.path.realpath(t)) for t in targets]
    digest = hashlib.sha256(chr(10).join(os.fspath(p) for p in paths).encode()).hexdigest()[:8]
    words = re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(p.name for p in paths[1:3]))[:60].strip("-")
    more = f"-and-{len(paths) - 3}-more" if len(paths) > 3 else ""
    return f"with-{words}{more}-{digest}"


def run_root(target: Path | str) -> Path:
    """The folder that holds a run's ``quicklook/`` (a cartesian output folder for its ``frames/``)."""

    try:
        layout = _frames.discover(target)
    except _frames.FrameError:
        return Path(target)
    return layout.root.parent if layout.kind == "frames" else layout.root


def _lost_of(run: Run, rows: Mapping[str, Any]) -> int:
    from .render import lost_times

    if not rows:
        return 0
    if run.layout is None:
        return len(rows)
    return len(lost_times(run.layout, run.lineage, rows))


def _left_since(run: Run) -> bool:
    """Whether the run's own records (run.json) say it has left a branch: a resume from an earlier restart, or a
    start over, so it may rightly have fewer saved frames than before."""

    directory = run.layout.run_directory() if run.layout is not None else None
    if directory is None:
        return False
    line = directory.recorded_line()
    try:
        recorded = {row.get("number") for row in directory.load().get("segments") or [] if isinstance(row, dict)}
    except Exception:  # noqa: BLE001 - a torn run.json says nothing
        return False
    return any(until is not None for until in line.values()) or bool(recorded - set(line))


def lost_frames(run: Run) -> int:
    """How many of a run's drawn saved times (its pictures' record, else their index, and the pictures themselves:
    :func:`.render.drawn_rows`) have lost their saved frames, though the run never left them
    (:func:`.render.lost_times`)."""

    from .render import drawn_rows

    return _lost_of(run, drawn_rows(run.quicklook))


def lost_since(folder: Path, runs: Sequence[Run]) -> str | None:
    """Why a comparison drawn before is not drawn again in ``folder``, or None: one of its runs has lost saved
    frames since, and drawn again it would show less (a storm time from another frame, fewer points).  The
    comparison's own record names the frames each run had when it was drawn; a record without them says how many
    there were, and a run with fewer now has lost some, unless its pictures' records say it left those times."""

    record = _read_json(Path(folder) / "compare.json")
    if not isinstance(record, dict):
        return None
    before = {os.path.realpath(row["folder"]): row for row in record.get("runs") or []
              if isinstance(row, dict) and isinstance(row.get("folder"), str)}
    said = []
    for run in runs:
        row = before.get(os.path.realpath(run.root))
        if row is None:
            continue
        had = row.get("saved_times")
        frames = row.get("frames")
        if isinstance(frames, list):
            # the frames it was drawn from: each one gone, and not on a branch the run has left since, is lost
            drawn = {repr(float(f["time"])): {"frame": f.get("frame"), "segment": f.get("segment"),
                                              "time": float(f["time"])}
                     for f in frames if isinstance(f, dict) and isinstance(f.get("time"), (int, float))}
            lost = _lost_of(run, drawn)
        else:
            lost = lost_frames(run)
            fewer = isinstance(had, int) and not isinstance(had, bool) and had > len(run.lineage)
            if fewer and not lost and not (run.layout is not None and _left_since(run)):
                lost = had - len(run.lineage)
            if not fewer:
                lost = 0          # drawn when those frames were gone already: drawing it again shows the same
        if not lost:
            continue
        said.append(f"{run.letter} ({run.name}) has lost {lost} of its saved frames")
    if not said:
        return None
    return (f"This comparison was drawn when its runs had more saved frames: {'; '.join(said)} since. Drawn again "
            "it would show less, so nothing was drawn and the pictures it has are kept. To draw it from what is "
            "left, pass --out with a new folder.")


def make(targets: Sequence[Path | str], *, time_s: float | None = None, box_km: float | None = None,
         only: Sequence[str] | None = None, out: Path | str | None = None,
         log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Draw the comparison of ``targets`` (the first is the run whose folder holds it) and return its record.

    Its ``progress.json`` says what is being drawn, and why not when it stops, so the Results screen can say so."""

    from . import compare_draw as D

    log = log or (lambda _m: None)
    looks = tuple(only) if only else LOOKS
    unknown = [look for look in looks if look not in LOOKS]
    if unknown:
        raise CompareError(f"no compare look {', '.join(unknown)} (they are {', '.join(LOOKS)})")
    if box_km is not None and not (SMALLEST_BOX / 1000.0 <= box_km <= 2000.0):
        raise CompareError("--box is the width of each run's box in km, from 2 to 2000.")
    if not targets or not Path(targets[0]).is_dir():
        raise CompareError(f"{targets[0] if targets else 'the run'} does not exist.")
    folder = Path(out) if out is not None else run_root(targets[0]) / "quicklook" / "compare" / compare_folder(targets)
    folder.mkdir(parents=True, exist_ok=True)
    steps = (2 if "storm" in looks else 0) + (1 if "series" in looks else 0) + (1 if "soundings" in looks else 0)
    progress = Progress(folder, steps + 1)
    begun = _time.perf_counter()
    try:
        progress.step("reading the runs", done=False)
        runs = load_runs(targets)
        refused = lost_since(folder, runs)
        if refused:
            raise CompareError(refused)
        record = _make(runs, looks, folder, time_s, box_km, progress, log, D)
    except Exception as error:
        progress.finish("failed", str(error))
        raise
    record["seconds"] = round(_time.perf_counter() - begun, 2)
    record["folder"] = os.fspath(folder)
    _write_json(folder / "compare.json", record)
    progress.finish("done", f"{sum(len(v) if isinstance(v, list) else 1 for v in record['pictures'].values())} "
                            "pictures")
    return record


def _make(runs: list[Run], looks: Sequence[str], folder: Path, time_s: float | None, box_km: float | None,
          progress: Progress, log: Callable[[str], None], D: Any) -> dict[str, Any]:
    hurricane = runs[0].hurricane
    t_storm = storm_time(runs, time_s)
    span = box_km * 1000.0 if box_km is not None else min(
        HURRICANE_BOX if hurricane else STORM_BOX, max(min(r.extent) for r in runs))
    differ, unread = differences(runs)
    context = D.Context(runs=runs, t=t_storm, span=span, differ=differ, unread=unread, hurricane=hurricane)
    pictures: dict[str, Any] = {}
    not_drawn: dict[str, str] = {}
    states: list[StormState] | None = None
    first_look = None
    if "storm" in looks or "soundings" in looks:
        first_look = "reflectivity" if all(reflectivity_possible(r, t_storm) for r in runs) else "clouds"
        second = "wind" if hurricane else "surface"
        wanted = (first_look, second) if "storm" in looks else (first_look,)
        states = []
        for r in runs:
            log(f"{r.letter} {r.name}: reading its storm at {context.clock(r.nearest(t_storm))}")
            states.append(state_from_frames(r, t_storm, wanted) if r.has_frames
                          else state_from_pictures(r, t_storm, wanted))
        context.states = states
        context.first_look = first_look
    if "storm" in looks:
        for look in (first_look, "wind" if hurricane else "surface"):
            progress.step(f"drawing the side-by-side {look} picture")
            name = f"storm-{look}-{tag(t_storm)}.png"
            try:
                facts = D.storm_picture(context, look, folder / name)
                pictures[f"storm-{look}"] = {"file": name, "facts": facts}
            except D.NotDrawn as why:
                not_drawn[f"storm-{look}"] = str(why)
    if "series" in looks:
        progress.step("drawing the runs over time")
        series = []
        for r in runs:
            pts, source = series_points(r, log)
            series.append((pts, source))
        context.series = series
        try:
            facts = D.series_picture(context, folder / "series.png")
            pictures["series"] = {"file": "series.png", "facts": facts}
        except D.NotDrawn as why:
            not_drawn["series"] = str(why)
    if "soundings" in looks:
        progress.step("drawing the soundings")
        if context.series is None:
            context.series = [series_points(r, log) for r in runs]
        context.profiles = [profile(r) for r in runs]
        context.environments = [environment(r) for r in runs]
        files = []
        cards = D.card_groups(len(runs))
        for number, group in enumerate(cards, start=1):
            name = f"soundings-{tag(t_storm)}.png" if len(cards) == 1 else f"soundings-{tag(t_storm)}-{number}.png"
            facts = D.soundings_card(context, group, folder / name, number=number, of=len(cards))
            files.append({"file": name, "runs": [runs[n].letter for n in group], "facts": facts})
        pictures["soundings"] = files
    times = {r.letter: (states[n].t if states else r.nearest(t_storm)) for n, r in enumerate(runs)}
    return {
        "schema": SCHEMA,
        "made_utc": _utc(),
        "time_s": t_storm,
        "box_m": span,
        "kind": "hurricane" if hurricane else "thunderstorm",
        "runs": [{"letter": r.letter, "colour": r.colour, "name": r.name, "folder": os.fspath(r.root),
                  "grid": r.grid_words(), "dx_m": r.dx, "lowest_level_m": r.z0, "case": r.case_words(),
                  "saved_frames": r.has_frames, "saved_times": len(r.lineage), "time_s": times[r.letter],
                  # the frames it was drawn from, so it is not drawn again from fewer (:func:`lost_since`)
                  "frames": [{"time": float(t), "frame": str(key), "segment": segment}
                             for t, key, _path, segment in r.lineage],
                  "centre": None if not states else {"x_m": states[n].centre.x, "y_m": states[n].centre.y,
                                                     "kind": states[n].centre.kind,
                                                     "words": states[n].centre.words}}
                 for n, r in enumerate(runs)],
        "differ": differ,
        "settings_unread": unread,
        "pictures": pictures,
        "not_drawn": not_drawn,
        "looks": list(looks),
    }


__all__ = ["COLOURS", "CompareError", "LETTERS", "LOOKS", "MAX_RUNS", "Run", "SCHEMA", "StormState",
           "compare_folder", "differences", "load_run", "load_runs", "lost_frames", "lost_since", "make", "profile",
           "series_points", "storm_centre", "storm_time"]
