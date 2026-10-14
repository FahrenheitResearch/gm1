"""Flow near the ground: three looks for a thunderstorm run on a grid of 250 m or finer.

``storm-relative-flow``  Storm-relative flow at three heights: the wind with the storm's own motion taken away, at
                         0.5, 3 and 10 km above the ground, in three panels side by side, each a square of up to
                         40 km around the storm.  Streaks follow the flow, brighter where it is faster (a fixed,
                         stated range), and arrowed streamlines say which way it goes.
``streamwise-spin``      Streamwise spin near the ground: the horizontal spin (vorticity) on the w level nearest
                         100 m, as iron-filing streaks along its axis, amber where the axis lies along the
                         storm-relative wind (streamwise), violet where it points against it, blue-grey across it;
                         brighter where it spins faster, on a fixed logarithmic range.
``cold-pool-depth``      Cold-pool depth: how high the air at least 1 K colder than the air around it at the same
                         height reaches from the ground, in the square that holds all the cold air of this and the
                         earlier saved times (40, 60, 80, 120 km and so on: it only grows, so a film keeps its scale).

The storm's motion
    The run's own track first.  The storm is the strongest updraft at 5 km now (the w level nearest 5 km, averaged
    over 3 km squares so one gust does not move it).  Each saved time keeps its few strongest updrafts there (at
    least 8 km apart), and the track follows this one updraft back through them, each step to the updraft nearest
    the straight line through the places so far; for the first step, which has no line yet, every updraft within
    reach is tried and the one that leads furthest back along a line is kept.  So a second storm, such as the other
    half of a split, that is the strongest at some saved times is not mixed in.  The places on the ground (the
    domain's own motion added) over the last 30 minutes of saved times are fitted with a straight line.  It needs
    at least three saved times spanning 10 minutes and an updraft of at least 5 m/s; the track stops at a saved
    time with no updraft within reach (40 m/s from the later place, and 8 km from the line).  Without such a track
    the motion is the Bunkers right-mover estimate from the run's sounding: the number
    the run recorded before it started (the importer's report or the run's receipt), else the same method applied
    to the base-state wind the frame carries, or to the corner columns of the run's first saved time (as the
    environment look reads the air the storm starts in).  Every picture says which, and its index facts record it.

Where these looks are drawn
    Grids of :data:`FINEST_DX_M` (250 m) and finer.  A coarser run gets one plain reason in place of each picture,
    and a hurricane run leaves them out with the other looks made for one thunderstorm
    (:data:`.products.THUNDERSTORM_ONLY`).  The spin look also needs model levels at most 100 m apart near the
    ground (:func:`spin_face`) and at least 3 columns each way; otherwise it says why it is not drawn.

Nothing here changes the numbers beyond what is said on each picture: bilinear sampling onto the pixels, linear
interpolation in height between two levels, and the differences named in the footers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import products as P
from . import style as S
from .frame import FrameUnreadable, QuickFrame

#: The coarsest grid spacing (m, dx and dy) these looks are drawn for.
FINEST_DX_M = 250.0
#: The three heights of the flow look (m above ground) and each panel's colour.
FLOW_HEIGHTS: tuple[tuple[float, str, str], ...] = ((500.0, "#ffb347", "near the ground"),
                                                    (3000.0, "#a58bff", "mid-levels"),
                                                    (10000.0, "#4cc3ff", "anvil level"))
#: The side of the square each flow panel shows (m), centred on the storm; the area's own size when smaller.
FLOW_BOX_M = 40_000.0
#: Storm-relative speed drawn from dark (0) to full colour (this, m/s and faster).
FLOW_RANGE = 50.0
#: Streamlines start from an n x n jittered grid in each panel.
STREAMLINE_SEEDS = 16
#: The height of the spin look (m), the side of its square around the storm, and its fixed logarithmic range.
SPIN_HEIGHT_M = 100.0
SPIN_BOX_M = 20_000.0
SPIN_RANGE = (0.01, 0.5)          # 1/s, horizontal vorticity, logarithmic
#: The spin look's level rule: the w level nearest 100 m lies within this band and the two scalar levels around it
#: are at most this far apart.
SPIN_FACE_BAND = (50.0, 150.0)
SPIN_MAX_SPACING_M = 100.0
#: "Inflow" for the spin look's numbers: within this distance of the storm's updraft, where the storm-relative wind
#: blows toward it.  Streamwise: the spin's axis within 45 degrees of that wind.
INFLOW_RADIUS_M = 8000.0
STREAMWISE_DEGREES = 45.0
#: The cold-pool threshold (K below the air around it at the same height) and the depth colours' range (m).
COLD_K = 1.0
DEPTH_RANGE_M = 3000.0
DEPTH_LINES_M = 500.0
#: The cold-pool picture shows the square holding all the cold air so far, this much around it and at least this
#: wide (m), its side rounded up to the next of these steps (each doubled again past the last), so that a film of it
#: changes scale only a few times.
COLD_MARGIN_M = 5000.0
COLD_BOX_M = 40_000.0
COLD_STEPS_M = (40_000.0, 60_000.0)
COLD_FILE = "cold-pool-area.json"
COLD_SCHEMA = "gm1.quicklook-cold-pool-area/v1"
#: Above this share of the area cold at the lowest level, the median there is partly cold air; the picture says so.
CROWDED_SHARE = 0.4
#: The storm track: the window (s), the least number of saved times and span, the fastest believable step
#: (m/s, ground-relative), the smoothing square (m) and the weakest updraft (m/s) that counts as a storm.
TRACK_WINDOW_S = 1800.0
TRACK_MIN_POINTS = 3
TRACK_MIN_SPAN_S = 600.0
TRACK_MAX_STEP_M_S = 40.0
TRACK_SMOOTH_M = 3000.0
TRACK_MIN_W = 5.0
TRACK_HEIGHT_M = 5000.0
#: The updrafts each saved time keeps for the track: at most this many, each at least this far (m) from a stronger
#: one; and how far (m) an earlier place may lie from the straight line through the later ones.
TRACK_PEAKS = 6
TRACK_PEAK_GAP_M = 8000.0
TRACK_MISS_M = 8000.0
#: The Bunkers estimate reads the wind from the ground to this height (m): the 0-6 km mean wind and shear.
BUNKERS_TOP_M = 6000.0
#: The gold ring's radius on the ground (m): about the smoothing square the storm's place is found with.
RING_M = 1500.0
TRACK_FILE = "storm-track.json"
TRACK_SCHEMA = "gm1.quicklook-storm-track/v2"

IDS = ("storm-relative-flow", "streamwise-spin", "cold-pool-depth")
#: What each look would show, for the sentence that says why a coarse grid does not get it.
SHOWS = {"storm-relative-flow": "the flow around the storm's updraft",
         "streamwise-spin": "the spin near the ground that feeds the storm's rotation",
         "cold-pool-depth": "the shape of the cold pool"}

COMPASS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


# ----------------------------------------------------------------------------------------------- refusals
def refuse(L: "P.Looks", pid: str) -> None:
    """The plain reason a run does not get this look, raised as :class:`.products.NotDrawn`."""

    if L.hurricane:
        raise P.NotDrawn(P.THUNDERSTORM_ONLY_WHY)
    coarse = max(L.f.dx, L.f.dy)
    if coarse > FINEST_DX_M + 1e-6:
        raise P.NotDrawn(f"This look is drawn for grids of {FINEST_DX_M:g} m and finer; this run's grid is "
                         f"{S.fmt_grid(coarse)}, too coarse to show {SHOWS[pid]}.")


def spin_face(zh: Sequence[float], zf: Sequence[float]) -> int:
    """The w level of the spin look: the face nearest :data:`SPIN_HEIGHT_M` above the ground, with a scalar level
    below and above it at most :data:`SPIN_MAX_SPACING_M` apart.  Raises NotDrawn with the levels' own numbers
    when they are too far apart to show how the wind changes across 100 m."""

    zh = np.asarray(zh, dtype=np.float64)
    zf = np.asarray(zf, dtype=np.float64)
    k = int(np.argmin(np.abs(zf[1:len(zh)] - SPIN_HEIGHT_M))) + 1 if len(zh) > 1 else 0
    lo, hi = SPIN_FACE_BAND
    if k >= 1 and lo <= zf[k] <= hi and zh[k] - zh[k - 1] <= SPIN_MAX_SPACING_M + 1e-6:
        return k
    gap = float(zh[1] - zh[0]) if len(zh) > 1 else float("nan")
    raise P.NotDrawn(f"Not drawn: this run's lowest levels are {S.fmt_height(gap)} apart (the first at "
                     f"{S.fmt_height(float(zh[0]))}), too far apart to show how the wind changes across "
                     f"{S.fmt_height(SPIN_HEIGHT_M)}. This look needs model levels {S.fmt_height(SPIN_MAX_SPACING_M)} "
                     "apart or closer near the ground (Height levels, in Customize, for the next run).")


# ----------------------------------------------------------------------------------------------- the storm
def box_mean(a: np.ndarray, n: int) -> np.ndarray:
    """The mean of ``a`` over n x n squares (fewer cells at the edges), numpy only."""

    n = max(1, int(n))
    if n == 1:
        return np.asarray(a, dtype=np.float64)
    ny, nx = a.shape
    c = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    c[1:, 1:] = np.cumsum(np.cumsum(np.asarray(a, dtype=np.float64), axis=0), axis=1)
    h = n // 2
    j0, j1 = np.clip(np.arange(ny) - h, 0, ny), np.clip(np.arange(ny) + n - h, 0, ny)
    i0, i1 = np.clip(np.arange(nx) - h, 0, nx), np.clip(np.arange(nx) + n - h, 0, nx)
    s = c[j1][:, i1] - c[j0][:, i1] - c[j1][:, i0] + c[j0][:, i0]
    return s / ((j1 - j0)[:, None] * (i1 - i0)[None, :])


def storm_peaks(qf: QuickFrame) -> tuple[list[dict[str, float]], float]:
    """The strongest updrafts on the w level nearest 5 km, averaged over :data:`TRACK_SMOOTH_M` squares, strongest
    first: at most :data:`TRACK_PEAKS`, each at least :data:`TRACK_PEAK_GAP_M` from a stronger one and at least
    :data:`TRACK_MIN_W`; x, y in the domain (m) and on the ground (m).  Also the level's height."""

    k = qf.level_near(TRACK_HEIGHT_M, faces=True)
    w = qf.get("w3d", k)
    mean = box_mean(w, int(round(TRACK_SMOOTH_M / max(qf.dx, 1.0))))
    left = np.where(np.isfinite(mean), mean, -np.inf)
    gj = int(math.ceil(TRACK_PEAK_GAP_M / max(qf.dy, 1.0)))
    gi = int(math.ceil(TRACK_PEAK_GAP_M / max(qf.dx, 1.0)))
    peaks: list[dict[str, float]] = []
    for _ in range(TRACK_PEAKS):
        j, i = np.unravel_index(int(np.argmax(left)), left.shape)
        peak = float(left[j, i])
        if not np.isfinite(peak) or peak < TRACK_MIN_W:
            break
        x, y = (i + 0.5) * qf.dx, (j + 0.5) * qf.dy
        peaks.append({"x": float(x), "y": float(y), "xg": float(x + qf.domainlocx), "yg": float(y + qf.domainlocy),
                      "w": peak})
        # the cells within the gap of this updraft are its own, not another storm's
        j0, j1 = max(0, j - gj), min(left.shape[0], j + gj + 1)
        i0, i1 = max(0, i - gi), min(left.shape[1], i + gi + 1)
        dy = (np.arange(j0, j1)[:, None] - j) * qf.dy
        dx = (np.arange(i0, i1)[None, :] - i) * qf.dx
        left[j0:j1, i0:i1][dx ** 2 + dy ** 2 < TRACK_PEAK_GAP_M ** 2] = -np.inf
    return peaks, float(qf.zf[k])


def storm_place(qf: QuickFrame) -> dict[str, float] | None:
    """Where the storm is on this frame: the strongest updraft on the w level nearest 5 km, averaged over
    :data:`TRACK_SMOOTH_M` squares; x, y in the domain (m) and on the ground (m), or None when no updraft reaches
    :data:`TRACK_MIN_W`."""

    peaks, level = storm_peaks(qf)
    return {**peaks[0], "level_m": level} if peaks else None


def compass(u: float, v: float) -> str:
    return COMPASS[int(round(math.degrees(math.atan2(v, u)) / 45.0)) % 8]


@dataclass
class Motion:
    """The storm's motion (m/s, ground-relative) and where it came from."""

    u: float
    v: float
    source: str                   # "track", "bunkers-record", "bunkers-base-state" or "bunkers-corner"
    how: str                      # one line for the picture
    short: str                    # a few words for the footer
    points: int = 0
    span_s: float = 0.0
    why_not_track: str = ""

    @property
    def speed(self) -> float:
        return math.hypot(self.u, self.v)

    def line(self) -> str:
        return f"storm motion {self.speed:.1f} m/s ground-relative, {self.toward()}"

    def toward(self) -> str:
        return f"toward the {compass(self.u, self.v)} (u {S.fmt_num(self.u, 1)}, v {S.fmt_num(self.v, 1)})"

    def facts(self) -> dict[str, Any]:
        out: dict[str, Any] = {"u_m_s": self.u, "v_m_s": self.v, "source": self.source, "frame": "ground-relative"}
        if self.source == "track":
            out.update({"saved_times": self.points, "span_s": self.span_s})
        elif self.why_not_track:
            out["why_not_track"] = self.why_not_track
        return out


class FrameNotes:
    """A few numbers about each saved time, kept in ``state_dir/FILE`` by frame key, so that a picture can use the
    earlier saved times without reading them again.  :meth:`note` says what is kept."""

    FILE = ""
    SCHEMA = ""

    def __init__(self, state_dir: Path | None) -> None:
        self.state_dir = None if state_dir is None else Path(state_dir)
        self.rows: dict[str, dict[str, Any]] = {}
        self.changed = False
        if self.path is not None:
            try:
                doc = json.loads(self.path.read_text(encoding="utf-8"))
                if doc.get("schema") == self.SCHEMA and isinstance(doc.get("frames"), dict):
                    self.rows = {k: r for k, r in doc["frames"].items() if isinstance(r, dict)
                                 and isinstance(r.get("t"), (int, float)) and not isinstance(r.get("t"), bool)}
            except (OSError, ValueError, AttributeError):
                self.rows = {}

    @property
    def path(self) -> Path | None:
        return None if self.state_dir is None else self.state_dir / self.FILE

    def note(self, qf: QuickFrame) -> dict[str, Any]:
        raise NotImplementedError

    def add(self, qf: QuickFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"t": float(qf.t)}
        row.update(self.note(qf))
        self.rows[qf.key] = row
        self.changed = True
        return row

    def has(self, qf: QuickFrame) -> bool:
        return qf.key in self.rows and self.rows[qf.key].get("t") == float(qf.t)

    def catch_up(self, before: Sequence[tuple[float, str, Path, int]], open_frame: Callable[..., QuickFrame]) -> None:
        for t, key, path, segment in before:
            if key in self.rows:
                continue
            try:
                qf = open_frame(path, key, segment)
            except (FrameUnreadable, OSError, ValueError, KeyError):
                self.rows[key] = {"t": float(t), "unreadable": True}
                self.changed = True
                continue
            try:
                self.add(qf)
            except (FrameUnreadable, OSError, ValueError, KeyError):
                self.rows[key] = {"t": float(t), "unreadable": True}
                self.changed = True
            finally:
                qf.close()

    def save(self) -> None:
        if self.path is None or not self.changed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.partial")
        tmp.write_text(json.dumps({"schema": self.SCHEMA, "frames": self.rows}, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        self.changed = False

    def rows_until(self, keys: Sequence[str], t_now: float, since: float = -math.inf) -> list[dict[str, Any]]:
        """The rows of these frame keys from ``since`` to ``t_now``, oldest first."""

        rows = [self.rows[k] for k in keys if k in self.rows]
        return sorted((r for r in rows if since - 1e-6 <= r["t"] <= t_now + 1e-6), key=lambda r: r["t"])


def _line(points: Sequence[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    """(x0, u, y0, v): the straight line x = x0 + u t, y = y0 + v t fitted to (t, x, y) points (least squares)."""

    t = np.array([p[0] for p in points])
    u, x0 = np.polyfit(t, np.array([p[1] for p in points]), 1)
    v, y0 = np.polyfit(t, np.array([p[2] for p in points]), 1)
    return float(x0), float(u), float(y0), float(v)


class StormTrack(FrameNotes):
    """The storm's updrafts at each saved time (see :func:`storm_peaks`), and the track of the strongest one now."""

    FILE = TRACK_FILE
    SCHEMA = TRACK_SCHEMA

    def note(self, qf: QuickFrame) -> dict[str, Any]:
        peaks, level = storm_peaks(qf)
        if not peaks:
            return {}
        return {**peaks[0], "level_m": level, "peaks": [[p["xg"], p["yg"], p["w"]] for p in peaks]}

    @staticmethod
    def _peaks(row: Mapping[str, Any]) -> list[tuple[float, float]]:
        found = [(float(p[0]), float(p[1])) for p in row.get("peaks") or ()
                 if isinstance(p, (list, tuple)) and len(p) >= 2]
        return found or [(float(row["xg"]), float(row["yg"]))]

    @staticmethod
    def _back(kept: list[tuple[float, float, float]], rows: Sequence[Mapping[str, Any]]) -> float:
        """Follow ``kept`` (newest first, two places or more) back through ``rows`` (newest first): at each saved
        time the updraft nearest the straight line through the places so far, within :data:`TRACK_MISS_M` of it and
        :data:`TRACK_MAX_STEP_M_S` of the later place; stops at the first saved time with none.  Returns the sum of
        the misses."""

        total = 0.0
        for row in rows:
            t_last, x_last, y_last = kept[-1]
            t = float(row["t"])
            dt = t_last - t
            if dt <= 0:
                continue
            x0, u, y0, v = _line(kept)
            px, py = x0 + u * t, y0 + v * t
            best = None
            for xg, yg in StormTrack._peaks(row):
                if math.hypot(xg - x_last, yg - y_last) / dt > TRACK_MAX_STEP_M_S:
                    continue
                miss = math.hypot(xg - px, yg - py)
                if miss <= TRACK_MISS_M and (best is None or miss < best[0]):
                    best = (miss, xg, yg)
            if best is None:
                break
            total += best[0]
            kept.append((t, best[1], best[2]))
        return total

    def fit(self, keys: Sequence[str], t_now: float) -> tuple[tuple[float, float, int, float] | None, str]:
        """(u, v, saved times, span) of the straight-line fit to the places of the storm's updraft over the last
        :data:`TRACK_WINDOW_S`, followed back from now; or None and the plain reason.

        The step to the saved time before has no line to go by yet, and the nearest updraft there need not be this
        storm (a weak cell can sit nearer the storm's place now than the storm itself was).  So each updraft within
        reach there is tried, and the track is the one that follows back furthest along a straight line: the most
        saved times, then the longest span, then the smallest misses, then the strongest first step (the old rule,
        when nothing tells them apart)."""

        rows = self.rows_until(keys, t_now, t_now - TRACK_WINDOW_S)
        if not rows or "xg" not in rows[-1]:
            return None, "no updraft at 5 km strong enough to follow yet"
        newest = (float(rows[-1]["t"]), float(rows[-1]["xg"]), float(rows[-1]["yg"]))
        earlier: list[Mapping[str, Any]] = []
        for row in reversed(rows[:-1]):
            if "xg" not in row:
                if row.get("unreadable"):
                    continue          # a saved time whose data was removed: skip it, keep looking back
                break                 # a saved time with no updraft: the track cannot go back past it
            if float(row["t"]) < newest[0]:
                earlier.append(row)
        best: tuple[tuple[int, float, float, int], list[tuple[float, float, float]]] | None = None
        if earlier:
            first = earlier[0]
            dt = newest[0] - float(first["t"])
            for rank, (xg, yg) in enumerate(self._peaks(first)):
                if math.hypot(xg - newest[1], yg - newest[2]) / dt > TRACK_MAX_STEP_M_S:
                    continue
                kept = [newest, (float(first["t"]), xg, yg)]
                misses = self._back(kept, earlier[1:])
                score = (len(kept), kept[0][0] - kept[-1][0], -misses, -rank)
                if best is None or score > best[0]:
                    best = (score, kept)
        kept = best[1] if best is not None else [newest]
        span = kept[0][0] - kept[-1][0]
        if len(kept) < TRACK_MIN_POINTS or span < TRACK_MIN_SPAN_S - 1e-6:
            need = f"a track needs {TRACK_MIN_POINTS} saved times over {TRACK_MIN_SPAN_S / 60:.0f} min or more"
            if len(kept) == 1 and earlier:
                return None, ("the storm's updraft could not be followed back to the saved time before (no "
                              f"updraft near its place there); {need}")
            if len(kept) == 1:
                return None, (f"the storm's updraft has been seen at only 1 saved time in the last "
                              f"{TRACK_WINDOW_S / 60:.0f} min; {need}")
            return None, f"the storm's updraft has been followed at {len(kept)} saved times over {span / 60:.0f} " \
                         f"min; {need}"
        _x0, u, _y0, v = _line(kept)
        return (u, v, len(kept), float(span)), ""


def _cold_box(qf: QuickFrame, cold: np.ndarray) -> list[float] | None:
    """[x0, y0, x1, y1] (domain m) of the columns marked ``cold``, or None."""

    rows, cols = np.nonzero(cold)
    if rows.size == 0:
        return None
    return [float(cols.min() * qf.dx), float(rows.min() * qf.dy), float((cols.max() + 1) * qf.dx),
            float((rows.max() + 1) * qf.dy)]


def lowest_cold(qf: QuickFrame) -> np.ndarray:
    """theta minus the median of the whole area on the lowest level, as :func:`cold_depth` reads it."""

    d = np.asarray(qf.view("th3d")[0], np.float32)
    return d - np.float32(np.median(d))


class ColdArea(FrameNotes):
    """Where the cold air is on the lowest level at each saved time (the box around it), so that the cold-pool
    picture can show all the cold air so far."""

    FILE = COLD_FILE
    SCHEMA = COLD_SCHEMA

    def note(self, qf: QuickFrame) -> dict[str, Any]:
        return {"box": _cold_box(qf, lowest_cold(qf) <= -COLD_K)}

    def boxes(self, keys: Sequence[str], t_now: float) -> list[list[float]]:
        return [list(r["box"]) for r in self.rows_until(keys, t_now)
                if isinstance(r.get("box"), list) and len(r["box"]) == 4]


def _bunkers_record(history: Mapping[str, Any] | None) -> tuple[float, float, str] | None:
    """The Bunkers right mover the run recorded before it started (import report or receipt)."""

    layout = (history or {}).get("layout")
    if layout is None:
        return None
    from .environment import parameters

    try:
        root = Path(layout.root if layout.kind != "frames" else layout.root.parent)
        segments = [Path(s.frames_root).parent for s in layout.segments]
        env, source = parameters(root, segments)
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    if not isinstance(env, Mapping):
        return None
    u, v = env.get("bunkers_right_u_m_per_s"), env.get("bunkers_right_v_m_per_s")
    if not all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in (u, v)):
        return None
    return float(u), float(v), source


def _bunkers_of(qf: QuickFrame, u: np.ndarray, v: np.ndarray) -> tuple[float, float] | None:
    """Bunkers et al. (2000) right mover of a wind profile on the frame's scalar levels (domain-relative, so the
    domain's motion is added: CM1 init3d.F:32-33), the lowest level's wind taken down to the ground."""

    from ..run_geometry import Hodograph, bunkers_storm_motion

    z = np.asarray(qf.zh, dtype=np.float64)
    if len(u) != len(z) or len(v) != len(z) or z[-1] < BUNKERS_TOP_M:
        return None
    heights = (0.0, *[float(h) for h in z])
    ug = [float(u[0]) + qf.umove, *[float(c) + qf.umove for c in u]]
    vg = [float(v[0]) + qf.vmove, *[float(c) + qf.vmove for c in v]]
    try:
        right = bunkers_storm_motion(Hodograph(heights, tuple(ug), tuple(vg))).right_mover_m_s
    except (ValueError, TypeError):
        return None
    return float(right[0]), float(right[1])


def _bunkers_base_state(qf: QuickFrame) -> tuple[float, float] | None:
    """The Bunkers right mover of the frame's base-state wind (CM1's u0, v0)."""

    u0, v0 = qf.base_column("u0"), qf.base_column("v0")
    if u0 is None or v0 is None:
        return None
    return _bunkers_of(qf, np.asarray(u0, np.float64), np.asarray(v0, np.float64))


def _bunkers_corner(qf: QuickFrame) -> tuple[float, float] | None:
    """The Bunkers right mover of the wind in the frame's south-west corner columns (the median of a 4 x 4 block,
    as the environment look reads the air the storm starts in), for frames without the base-state wind."""

    if not (qf.has("u3d") and qf.has("v3d")):
        return None
    n = min(4, qf.nx - 1, qf.ny - 1)
    if n < 1:
        return None
    u = np.asarray(qf.view("u3d")[:, :n, :n], np.float64).reshape(qf.nz, -1)
    v = np.asarray(qf.view("v3d")[:, :n, :n], np.float64).reshape(qf.nz, -1)
    return _bunkers_of(qf, np.median(u, axis=1), np.median(v, axis=1))


def _bunkers_columns(L: "P.Looks", history: Mapping[str, Any]) -> tuple[float, float, str] | None:
    """The Bunkers estimate worked out for the picture: from the base-state wind the frame carries, else from the
    corner columns of the run's first saved time (this frame when the earlier ones are not at hand)."""

    base = _bunkers_base_state(L.f)
    if base is not None:
        return (*base, "the run's starting wind")
    before = list(history.get("before") or ())
    opener = history.get("open")
    if before and opener is not None:
        t, key, path, segment = before[0]
        try:
            first = opener(path, key, segment)
        except (FrameUnreadable, OSError, ValueError, KeyError):
            first = None
        if first is not None:
            try:
                corner = _bunkers_corner(first)
            finally:
                first.close()
            if corner is not None:
                return (*corner, f"the corner columns of the first saved time ({S.fmt_clock(t)})")
    corner = _bunkers_corner(L.f)
    if corner is not None:
        return (*corner, "this saved time's corner columns")
    return None


def motion_for(L: "P.Looks") -> Motion:
    """The storm's motion for this frame (computed once per frame and shared by the looks)."""

    cached = L.__dict__.get("_storm_motion")
    if cached is not None:
        return cached
    history = getattr(L, "history", None) or {}
    track = StormTrack(history.get("state_dir"))
    before = list(history.get("before") or ())
    opener = history.get("open")
    if opener is not None:
        track.catch_up(before, opener)
    if not track.has(L.f):
        track.add(L.f)
    track.save()
    L.__dict__["_storm_place"] = track.rows[L.f.key] if "xg" in track.rows[L.f.key] else None
    fit, why = track.fit([key for _t, key, _p, _s in before] + [L.f.key], L.f.t)
    if fit is not None:
        u, v, n, span = fit
        motion = Motion(u, v, "track",
                        f"from the run's own track: the strongest updraft at 5 km now, followed back over {n} saved "
                        f"times ({span / 60:.0f} min)",
                        "storm motion from the run's own track", n, span)
    else:
        record = _bunkers_record(history)
        if record is not None:
            u, v, source = record
            motion = Motion(u, v, "bunkers-record",
                            f"the Bunkers right-mover estimate from the sounding ({source}): {why}",
                            "storm motion: Bunkers estimate from the sounding", why_not_track=why)
        else:
            worked = _bunkers_columns(L, history)
            if worked is None:
                lack = (f"the run's wind does not reach {S.fmt_km(BUNKERS_TOP_M)}, which the Bunkers estimate needs"
                        if float(L.f.zh[-1]) < BUNKERS_TOP_M else "the run recorded no wind to estimate one from")
                raise P.NotDrawn(f"No storm motion to take away: the storm has no steady track yet ({why}), and "
                                 f"{lack}.")
            u, v, where = worked
            motion = Motion(u, v, "bunkers-base-state" if where == "the run's starting wind" else "bunkers-corner",
                            f"the Bunkers right-mover estimate from {where}, worked out for this picture: {why}",
                            "storm motion: Bunkers estimate", why_not_track=why)
    L.__dict__["_storm_motion"] = motion
    return motion


def storm_centre(L: "P.Looks") -> dict[str, Any] | None:
    """This frame's storm place (see :func:`storm_place`), shared with the motion."""

    motion_for(L)
    return L.__dict__.get("_storm_place")


# ----------------------------------------------------------------------------------------------- shared drawing
def around(extent: tuple[float, float], centre: tuple[float, float] | None,
           side: float) -> tuple[float, float, float, float]:
    """(x0, y0, width, height) of a square of ``side`` around ``centre`` kept inside the area (m), or the area's
    own size where it is smaller."""

    lx, ly = extent
    wx, wy = min(side, lx), min(side, ly)
    cx, cy = centre if centre is not None else (0.5 * lx, 0.5 * ly)
    x0 = float(np.clip(cx - 0.5 * wx, 0.0, lx - wx))
    y0 = float(np.clip(cy - 0.5 * wy, 0.0, ly - wy))
    return x0, y0, wx, wy


def panel_box(x: int, y: int, wmax: int, hmax: int, area: tuple[float, float, float, float]) -> P.Box:
    x0, y0, wx, wy = area
    mpp = max(wx / wmax, wy / hmax)
    w, h = int(round(wx / mpp)), int(round(wy / mpp))
    return P.Box(x + (wmax - w) // 2, y, w, x0, y0, w * mpp, h=h)


def uv_at(qf: QuickFrame, z_m: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Ground-relative u, v at scalar points ``z_m`` above ground, linear in height between the two levels around
    it, and those two heights."""

    zh = qf.zh
    k = int(np.searchsorted(zh, z_m)) - 1
    k = max(0, min(k, len(zh) - 2))
    w = float(np.clip((z_m - zh[k]) / (zh[k + 1] - zh[k]), 0.0, 1.0))
    ua, va = qf.uv_level(k, ground=True)
    ub, vb = qf.uv_level(k + 1, ground=True)
    return ((1 - w) * ua + w * ub).astype(np.float32), ((1 - w) * va + w * vb).astype(np.float32), \
        (float(zh[k]), float(zh[k + 1]))


def colour_bar(ov: S.Overlay, rgb: np.ndarray, box: tuple[int, int, int, int], ticks: Sequence[tuple[float, str]],
               title: str | None = None) -> None:
    """A colour strip of ``rgb`` (n x 3 floats, left to right) with ticks at positions 0..1 and a title above."""

    from PIL import Image

    x0, y0, x1, y1 = (int(v) for v in box)
    idx = np.linspace(0, len(rgb) - 1, max(2, x1 - x0)).round().astype(int)
    strip = np.repeat(S.to8(rgb[idx])[None], max(1, y1 - y0), axis=0)
    ov.draw_rgba(Image.fromarray(strip).convert("RGBA"), (x0, y0))
    ms = ov.px("micro")
    for pos, label in ticks:
        tx = x0 + pos * (x1 - x0)
        anchor = "ls" if pos <= 0.01 else "rs" if pos >= 0.99 else "ms"
        ov.text((tx, y1 + ov.U(8) + ms), label, "micro", anchor=anchor, halo=False)
    if title:
        ov.text(((x0 + x1) / 2, y0 - ov.U(9)), title, "micro", colour="data", anchor="ms", halo=False)


def hue_ramp(colour: str, values: np.ndarray, top: float, floor: float = 0.07) -> np.ndarray:
    """``colour`` scaled from near black (0) to full (``top`` and above), as float RGB."""

    b = np.clip(np.asarray(values, dtype=np.float32) / np.float32(top), 0.0, 1.0) ** 0.85
    return np.array(S.hex_rgb(colour), np.float32) * (floor + (1.0 - floor) * b)[..., None]


def regular(field: np.ndarray, box: P.Box, dx: float, dy: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, values) of a field on an n-across regular grid over ``box`` (bilinear), x and y in metres."""

    ny = max(2, int(round(n * box.span_y / box.span)))
    xs = box.mx0 + (np.arange(n) + 0.5) * box.span / n
    ys = box.my0 + (np.arange(ny) + 0.5) * box.span_y / ny
    rr = np.broadcast_to((ys / dy - 0.5)[:, None], (ny, n))
    cc = np.broadcast_to((xs / dx - 0.5)[None, :], (ny, n))
    return xs, ys, P.bilinear(field, rr, cc)


def streamlines(cv: np.ndarray, panels: Sequence[tuple[P.Box, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                colour: str = "#eef3fb", alpha: float = 0.55) -> np.ndarray:
    """Arrowed streamlines (matplotlib's streamplot) of each panel's (box, x, y, u, v) drawn over ``cv``."""

    W, H = cv.shape[1], cv.shape[0]
    fig = S.figure((W, H))
    back = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    back.imshow(cv, interpolation="none", aspect="auto", extent=(0, W, H, 0))
    back.set_axis_off()
    r, g, b = S.hex_rgb(colour)
    for box, xs, ys, us, vs in panels:
        ax = fig.add_axes([box.x / W, 1.0 - (box.y + box.h) / H, box.w / W, box.h / H])
        ax.set_xlim(box.mx0, box.mx0 + box.span)
        ax.set_ylim(box.my0, box.my0 + box.span_y)
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        speed = np.hypot(us, vs)
        if not np.isfinite(speed).all() or float(speed.max()) <= 0.0:
            continue
        # Short lines from seeds on a jittered grid (the same on every picture): matplotlib puts each line's arrow
        # at its middle, so long lines across an even flow, or seeds on a regular grid, put the arrows in a row,
        # which would read as a feature of the flow.
        n = STREAMLINE_SEEDS
        rng = np.random.default_rng(29)
        gx, gy = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
        jx = np.clip(gx + rng.uniform(-0.45, 0.45, gx.shape) / n, 0.01, 0.99)
        jy = np.clip(gy + rng.uniform(-0.45, 0.45, gy.shape) / n, 0.01, 0.99)
        seeds = np.column_stack([xs[0] + jx.ravel() * (xs[-1] - xs[0]), ys[0] + jy.ravel() * (ys[-1] - ys[0])])
        ax.streamplot(xs, ys, us, vs, density=1.15, color=(r, g, b, alpha), linewidth=0.75, arrowsize=0.9,
                      arrowstyle="-|>", minlength=0.08, maxlength=0.45, integration_direction="both",
                      broken_streamlines=True, start_points=seeds)
    return S.fig_to_rgb(fig)


def dbz_1km_around(L: "P.Looks",
                   area: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[float, float]] | None:
    """(dBZ at 1 km, the south-west corner it starts at in m) over the columns of ``area``, or None without a
    reflectivity operator.  The reflectivity look's own field when this pass has made it; otherwise the same
    operator on just these columns (CM1's calc_refl10cm works column by column, so the numbers are the same)."""

    from . import reflectivity as R

    f = L.f
    if "reflectivity" in L.__dict__ or (f.has("dbz") and tuple(f.frame.carriers["dbz"].shape) == (f.nz, f.ny, f.nx)):
        return (L.dbz_1km[0], (0.0, 0.0)) if L.reflectivity.available else None
    if str(f.config.get("microphysics") or "") not in R.MORRISON:
        return None
    calc, _where = R._calc_function()
    if calc is None:
        return None
    x0, y0, wx, wy = area
    i0, i1 = max(0, int(x0 // f.dx) - 1), min(f.nx, int(math.ceil((x0 + wx) / f.dx)) + 1)
    j0, j1 = max(0, int(y0 // f.dy) - 1), min(f.ny, int(math.ceil((y0 + wy) / f.dy)) + 1)
    try:
        volume = R.morrison_volume(f, calc, crop=(slice(j0, j1), slice(i0, i1)))
    except (ValueError, KeyError):
        return None
    one, _levels = R.at_height(volume, f.zh, P.REFLECTIVITY_HEIGHT)
    return one, (i0 * f.dx, j0 * f.dy)


#: The spin look's wind arrows: how many across the square, and the speed one arrow's reference length stands for.
ARROWS_ACROSS = 13
ARROW_REF_M_S = 20.0
ARROW_REF_PX = 46.0


def arrows(cv: np.ndarray, box: P.Box, xs: np.ndarray, ys: np.ndarray, us: np.ndarray, vs: np.ndarray,
           colour: str = "#eef3fb", alpha: float = 0.6) -> np.ndarray:
    """Arrows of the wind (us, vs on the grid xs, ys in m) over one panel; an arrow of :data:`ARROW_REF_PX` pixels
    is :data:`ARROW_REF_M_S`."""

    W, H = cv.shape[1], cv.shape[0]
    fig = S.figure((W, H))
    back = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    back.imshow(cv, interpolation="none", aspect="auto", extent=(0, W, H, 0))
    back.set_axis_off()
    ax = fig.add_axes([box.x / W, 1.0 - (box.y + box.h) / H, box.w / W, box.h / H])
    ax.set_xlim(box.mx0, box.mx0 + box.span)
    ax.set_ylim(box.my0, box.my0 + box.span_y)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    r, g, b = S.hex_rgb(colour)
    xx, yy = np.meshgrid(xs, ys)
    ax.quiver(xx, yy, us, vs, angles="xy", scale_units="xy", scale=ARROW_REF_M_S / (ARROW_REF_PX * box.mpp),
              color=(r, g, b, alpha), width=0.0024, headwidth=3.6, headlength=4.0, headaxislength=3.6, pivot="mid")
    return S.fig_to_rgb(fig)


def square_words(area: Sequence[float]) -> str:
    """'40 × 40 km' for an (x0, y0, width, height) in metres, to 0.1 km."""

    return f"{round(area[2] / 1000.0, 1):g} × {round(area[3] / 1000.0, 1):g} km"


def ground_note(spacing: float) -> str:
    return f"+ fixed to the ground every {S.fmt_km(spacing)}"


def within(a: np.ndarray, area: Sequence[float], dx: float, dy: float) -> np.ndarray:
    """The columns of ``a`` (ny, nx on the scalar grid) inside ``area`` (x0, y0, width, height, m)."""

    rows = slice(int(area[1] // dy), int((area[1] + area[3]) // dy) + 1)
    return a[rows, int(area[0] // dx):int((area[0] + area[2]) // dx) + 1]


#: The flow look's three colour bars, bottom right: each this wide, this far apart (layout units); and the least
#: room between the words on the left and the first bar.
FLOW_BAR_W = 170
FLOW_BAR_GAP = 18
FLOW_TEXT_GAP = 24


def flow_bars_left(ov: S.Overlay) -> int:
    """Where the flow look's first colour bar starts (px)."""

    return ov.x1 - 3 * (ov.U(FLOW_BAR_W) + ov.U(FLOW_BAR_GAP)) + ov.U(FLOW_BAR_GAP)


# ----------------------------------------------------------------------------------------------- the looks
def product_flow(L: "P.Looks", out: Path) -> dict[str, Any]:
    """Storm-relative flow at 0.5, 3 and 10 km, three panels around the storm."""

    refuse(L, "storm-relative-flow")
    f = L.f
    motion = motion_for(L)
    place = storm_centre(L)
    centre = (place["x"], place["y"]) if place else None
    area = around(f.extent, centre, FLOW_BOX_M)
    gap, top = 40, 205
    pw = (P.W - 160 - 2 * gap) // 3
    boxes = [panel_box(80 + n * (pw + gap), top, pw, pw, area) for n in range(3)]
    cv = P.canvas()
    found = dbz_1km_around(L, area)
    dbz, dbz_origin = found if found is not None else (None, (0.0, 0.0))
    rows, lines = [], []
    spacing = 0.0
    for n, ((height, colour, _role), box) in enumerate(zip(FLOW_HEIGHTS, boxes)):
        if height > float(f.zh[-1]):
            rows.append({"height_m": height, "above_top": True})
            lines.append(None)
            continue
        u, v, levels = uv_at(f, height)
        u -= np.float32(motion.u)
        v -= np.float32(motion.v)
        us, vs = P.sample(u, box, f.dx, f.dy), P.sample(v, box, f.dx, f.dy)
        tx, sp = P.streaks(us, vs, box, px=14, seed=7 + n)
        rgb = hue_ramp(colour, sp, FLOW_RANGE) * (0.62 + 0.38 * np.clip(0.5 + 0.5 * tx, 0.0, 1.3))[..., None]
        img = S.to8(rgb)
        if dbz is not None:
            ds = P.sample(dbz, box, f.dx, f.dy, origin=dbz_origin)
            img = P.blend(img, P.line_alpha(ds, 30.0, width=1.3), "title", 0.55)
        img, spacing = P.ground_ticks(img, box, f.domainlocx, f.domainlocy, alpha=0.16)
        P.paste(cv, img, box)
        cv = P.outline(cv, box)
        xs, ys, ug = regular(u, box, f.dx, f.dy, 70)
        _, _, vg = regular(v, box, f.dx, f.dy, 70)
        lines.append((box, xs, ys, ug, vg))
        shown = within(np.hypot(u, v), area, f.dx, f.dy)
        rows.append({"height_m": height, "levels_m": list(levels), "speed_max_storm_relative": float(shown.max()),
                     "speed_mean_storm_relative": float(shown.mean())})
    cv = streamlines(cv, [row for row in lines if row is not None])
    where = "around the storm" if place is not None else "in the middle of the area"
    data = L.data_line(f"{square_words(area)} {where}")
    dbz_note = "white line: 30 dBZ at 1 km" if dbz is not None else "no reflectivity for this run"
    ov = P.overlay(L, "storm-relative-flow", data, level="0.5, 3 and 10 km", wind="storm-relative",
                   notes=S.data_line("winds minus the storm motion below", "linear in height between levels",
                                     dbz_note, "squares move with the storm", ground_note(spacing)))
    for (height, colour, role), box, row in zip(FLOW_HEIGHTS, boxes, rows):
        ov.text((box.x, box.y - ov.U(12)), f"{S.fmt_height(height)}, {role}", "label", colour=colour, anchor="ls")
        if row.get("above_top"):
            ov.text((box.x + box.w / 2, box.y + box.h / 2), "above the model top", "label", colour="muted",
                    anchor="ms")
            continue
        ov.text((box.x + box.w, box.y - ov.U(12)), f"fastest {row['speed_max_storm_relative']:.0f} m/s storm-relative",
                "label", colour="data", anchor="rs")
        lo, hi = row["levels_m"]
        ov.text((box.x, box.y + box.h + ov.U(22)), f"between the {S.fmt_height(lo)} and {S.fmt_height(hi)} levels",
                "micro", colour="muted", anchor="ls", halo=False)
        if place is not None:
            P._ring(ov, box, place["x"], place["y"], RING_M)
    y = boxes[0].y + boxes[0].h + ov.U(66)
    ov.text((ov.x0, y), motion.line(), "label", colour="text", anchor="ls")
    # where the motion came from, in as many lines as it needs to stay clear of the colour bars on the right
    how = ov.wrap(motion.how, "micro", flow_bars_left(ov) - ov.U(FLOW_TEXT_GAP) - ov.x0)
    for n, part in enumerate(how):
        ov.text((ov.x0, y + ov.U(26 + 22 * n)), part, "micro", colour="data", anchor="ls", halo=False, vertical=True)
    below = y + ov.U(22) * (len(how) - 1)
    ring = "gold ring: the storm's updraft at 5 km" if place is not None else \
        f"no updraft of {TRACK_MIN_W:g} m/s at 5 km yet: the squares sit in the middle of the area"
    ov.text((ov.x0, below + ov.U(48)), ring, "micro", colour="gold" if place is not None else "muted", anchor="ls",
            halo=False, vertical=True)
    ramp = np.linspace(0.0, FLOW_RANGE, 256)
    bar_w, by = ov.U(FLOW_BAR_W), y + ov.U(10)
    for n, (_height, colour, _role) in enumerate(FLOW_HEIGHTS):
        bx = flow_bars_left(ov) + n * (bar_w + ov.U(FLOW_BAR_GAP))
        colour_bar(ov, hue_ramp(colour, ramp, FLOW_RANGE), (bx, by, bx + bar_w, by + max(3, ov.U(7))),
                   [(0.0, "0"), (1.0, f"{FLOW_RANGE:g} m/s")] if n == 2 else [(0.0, "0"), (1.0, f"{FLOW_RANGE:g}")])
    ov.text((ov.x1, by - ov.U(9)), "storm-relative speed, brighter is faster; arrows: the way the air goes", "micro",
            colour="data", anchor="rs", halo=False)
    P.scale_bar(ov, ov.x0, below + ov.U(76), boxes[0])
    P.save(ov, cv, out)
    facts: dict[str, Any] = {"storm_motion": motion.facts(), "heights": rows, "range_m_s": FLOW_RANGE,
                             "box_m": [area[0], area[1], area[2], area[3]], "dbz_line": dbz is not None}
    if place is not None:
        facts["storm_xy_m"] = [place["x"], place["y"]]
    return facts


def horizontal_spin(qf: QuickFrame, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """(omega_x, omega_y, u, v) on w level k at scalar columns, and its height: omega_x = dw/dy - dv/dz,
    omega_y = du/dz - dw/dx, with dz the distance between the scalar levels k-1 and k, the horizontal derivatives
    centred on the level's w, and u, v ground-relative, linear in height to the level."""

    zh, zf = qf.zh, qf.zf
    ua, va = qf.uv_level(k - 1, ground=True)
    ub, vb = qf.uv_level(k, ground=True)
    dz = np.float32(zh[k] - zh[k - 1])
    w = qf.get("w3d", k)
    dwdy, dwdx = np.gradient(w, np.float32(qf.dy), np.float32(qf.dx))
    ox = (dwdy - (vb - va) / dz).astype(np.float32)
    oy = ((ub - ua) / dz - dwdx).astype(np.float32)
    share = np.float32((zf[k] - zh[k - 1]) / (zh[k] - zh[k - 1]))
    return ox, oy, (1 - share) * ua + share * ub, (1 - share) * va + share * vb, float(zf[k])


#: The spin look's colours by the angle between the spin's axis and the storm-relative wind: along it (0),
#: across it (0.5), against it (1).
SPIN_STOPS = ((0.0, "#ffb347"), (0.5, "#8b97c2"), (1.0, "#a77dff"))


def spin_colours(t: np.ndarray) -> np.ndarray:
    pos = np.array([p for p, _ in SPIN_STOPS])
    cols = np.array([S.hex_rgb(c) for _, c in SPIN_STOPS])
    return np.stack([np.interp(t, pos, cols[:, ch]) for ch in range(3)], axis=-1).astype(np.float32)


def spin_bright(mag: np.ndarray) -> np.ndarray:
    lo, hi = SPIN_RANGE
    return np.clip((np.log10(np.maximum(mag, 1e-9)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)), 0.0, 1.0)


def product_spin(L: "P.Looks", out: Path) -> dict[str, Any]:
    """Streamwise spin near the ground: horizontal vorticity on the w level nearest 100 m."""

    refuse(L, "streamwise-spin")
    f = L.f
    k = spin_face(f.zh, f.zf)
    if min(f.nx, f.ny) < 3:
        raise P.NotDrawn(f"Not drawn: this run is {f.nx} × {f.ny} columns, too few to work out how w changes from "
                         "column to column. The spin needs 3 columns or more each way.")
    motion = motion_for(L)
    place = storm_centre(L)
    ox, oy, u, v, z = horizontal_spin(f, k)
    u = u - np.float32(motion.u)
    v = v - np.float32(motion.v)
    mag = np.hypot(ox, oy)
    sr = np.hypot(u, v)
    cosang = (ox * u + oy * v) / np.maximum(mag * sr, 1e-12)
    area = around(f.extent, (place["x"], place["y"]) if place else None, SPIN_BOX_M)
    box = panel_box(P.W // 2 - 400, 165, 800, 800, area)
    oxs, oys = P.sample(ox, box, f.dx, f.dy), P.sample(oy, box, f.dx, f.dy)
    us, vs = P.sample(u, box, f.dx, f.dy), P.sample(v, box, f.dx, f.dy)
    ms = np.hypot(oxs, oys)
    cs = (oxs * us + oys * vs) / np.maximum(ms * np.hypot(us, vs), 1e-12)
    sure = np.clip(np.hypot(us, vs) / 2.0, 0.0, 1.0)        # under 2 m/s storm-relative the wind has no clear way
    t = 0.5 + (0.5 * (1.0 - np.clip(cs, -1.0, 1.0)) - 0.5) * sure
    filings = P.lic(P.noise(box.h, box.w, 23), oxs / np.maximum(ms, 1e-12), -oys / np.maximum(ms, 1e-12),
                    max(5, int(round(10 * box.h / 800))))
    tex = np.clip(0.5 + (filings - 0.5) / 0.12 * 0.5, 0.0, 1.4)
    b = spin_bright(ms)
    rgb = spin_colours(t) * (0.05 + 0.95 * b ** 0.9)[..., None] * (0.5 + 0.5 * tex)[..., None]
    img, spacing = P.ground_ticks(S.to8(rgb), box, f.domainlocx, f.domainlocy, alpha=0.14)
    cv = P.canvas()
    P.paste(cv, img, box)
    cv = P.outline(cv, box)
    xs, ys, ug = regular(u, box, f.dx, f.dy, ARROWS_ACROSS)
    _, _, vg = regular(v, box, f.dx, f.dy, ARROWS_ACROSS)
    cv = arrows(cv, box, xs, ys, ug, vg)
    # numbers in the inflow: near the storm's updraft, where the storm-relative wind blows toward it
    jj, ii = np.mgrid[0:f.ny, 0:f.nx]
    xx, yy = (ii + 0.5) * f.dx, (jj + 0.5) * f.dy
    strongest = float(within(mag, area, f.dx, f.dy).max())       # in the square shown, as the flow's "fastest"
    facts: dict[str, Any] = {"height_m": z, "levels_m": [float(f.zh[k - 1]), float(f.zh[k])],
                             "spin_max": strongest, "spin_max_whole_area": float(mag.max()),
                             "range_per_s": list(SPIN_RANGE), "storm_motion": motion.facts(), "box_m": list(area),
                             "method": "dw/dy - dv/dz and du/dz - dw/dx, by this picture"}
    readout = [(f"strongest horizontal spin shown, at {S.fmt_height(z)}", f"{S.fmt_num(strongest, 2)} {S.PER_S}")]
    if place is not None:
        rx, ry = place["x"] - xx, place["y"] - yy
        inflow = (np.hypot(rx, ry) <= INFLOW_RADIUS_M) & (rx * u + ry * v > 0.0) & (sr >= 2.0)
        if inflow.any():
            streamwise = float((cosang[inflow] >= math.cos(math.radians(STREAMWISE_DEGREES))).mean())
            along = float(np.median((mag * cosang)[inflow]))
            facts.update({"inflow_cells": int(inflow.sum()), "streamwise_share": streamwise,
                          "streamwise_spin_median": along, "spin_median_inflow": float(np.median(mag[inflow]))})
            readout += [("streamwise share of the inflow", f"{100 * streamwise:.0f} %"),
                        ("median streamwise spin in the inflow", f"{S.fmt_num(along, 3)} {S.PER_S}")]
        facts["storm_xy_m"] = [place["x"], place["y"]]
    where = "around the storm" if place is not None else "in the middle of the area"
    data = L.data_line(f"{S.fmt_height(z)} up", f"{square_words(area)} {where}")
    ov = P.overlay(L, "streamwise-spin", data, level=f"w level {S.fmt_height(z)}", wind="storm-relative",
                   notes=S.data_line(f"spin: dw/dy − dv/dz, du/dz − dw/dx across the "
                                     f"{S.fmt_height(float(f.zh[k - 1]))} and {S.fmt_height(float(f.zh[k]))} levels, "
                                     "by this picture",
                                     "streaks along the spin's axis", ground_note(spacing)))
    x, y = ov.x0, box.y + ov.U(10)
    ov.readout((x, y), readout)
    ly = y + ov.U(70) * len(readout) + ov.U(40)
    room = box.x - ov.U(24) - x            # the words on the left stay clear of the picture
    if place is not None:
        P._ring(ov, box, place["x"], place["y"], RING_M)
        ov.text((x, ly), "gold ring: the storm's updraft at 5 km", "micro", colour="gold", anchor="ls", halo=False)
        ov.text((x, ly + ov.U(22)), f"inflow: within {S.fmt_km(INFLOW_RADIUS_M)} of it, where the wind blows toward it",
                "micro", colour="muted", anchor="ls", halo=False)
        ov.text((x, ly + ov.U(44)), f"streamwise: axis within {STREAMWISE_DEGREES:g}° of the wind", "micro",
                colour="muted", anchor="ls", halo=False)
    else:
        lonely = f"no updraft of {TRACK_MIN_W:g} m/s at 5 km yet: the square sits in the middle of the area"
        for n, part in enumerate(ov.wrap(lonely, "micro", room)):
            ov.text((x, ly + ov.U(22 * n)), part, "micro", colour="muted", anchor="ls", halo=False, vertical=True)
    ov.text((x, ly + ov.U(80)), f"storm motion {motion.speed:.1f} m/s ground-relative,", "micro", colour="data",
            anchor="ls", halo=False)
    ov.text((x, ly + ov.U(100)), motion.toward(), "micro", colour="data", anchor="ls", halo=False)
    how = ov.wrap(motion.how, "micro", room)
    for n, part in enumerate(how):
        ov.text((x, ly + ov.U(122 + 20 * n)), part, "micro", colour="muted", anchor="ls", halo=False, vertical=True)
    ov.text((x, ly + ov.U(122 + 20 * (len(how) - 1) + 38)),
            f"features smaller than about {S.fmt_km(6 * max(f.dx, f.dy))} are not resolved", "micro", colour="muted",
            anchor="ls", halo=False)
    rx0, wr = ov.x1 - ov.U(330), ov.U(330)
    by = box.y + ov.U(40)
    colour_bar(ov, spin_colours(np.linspace(0.0, 1.0, 256)), (rx0, by, rx0 + wr, by + max(3, ov.U(8))),
               [(0.0, "along"), (0.5, "across"), (1.0, "against")], title="spin's axis to the storm-relative wind")
    by2 = by + ov.U(90)
    shade = np.linspace(0.0, 1.0, 256)
    colour_bar(ov, np.repeat((0.05 + 0.95 * shade ** 0.9)[:, None], 3, axis=1) * np.array(S.hex_rgb("#c7cfdb"),
                                                                                           np.float32),
               (rx0, by2, rx0 + wr, by2 + max(3, ov.U(8))),
               [(float(spin_bright(np.array(val))), label) for val, label in
                ((0.01, "0.01"), (0.03, "0.03"), (0.1, "0.1"), (0.5, f"0.5 {S.PER_S}"))],
               title="brightness: how fast it spins, log scale")
    P.scale_bar(ov, rx0, by2 + ov.U(80), box)
    ay = by2 + ov.U(140)
    ov.text((rx0, ay), f"arrows: the storm-relative wind at {S.fmt_height(z)}", "micro", colour="data", anchor="ls",
            halo=False)
    ov.rule(rx0, ay + ov.U(18), ARROW_REF_PX, thickness=2, colour="#eef3fb", alpha=0.7)
    ov.text((rx0 + ARROW_REF_PX + ov.U(10), ay + ov.U(24)), f"{ARROW_REF_M_S:g} m/s storm-relative", "micro",
            colour="data", anchor="ls", halo=False)
    P.save(ov, cv, out)
    facts["arrow_reference_m_s"] = ARROW_REF_M_S
    return facts


def cold_depth(qf: QuickFrame, cold_k: float = COLD_K) -> tuple[np.ndarray, np.ndarray, int]:
    """(depth m, theta minus the environment at the lowest level, levels read) of the ground-attached air at least
    ``cold_k`` colder than the environment: the median theta of the whole area at each height.  Theta is the
    base state plus CM1's theta' (th3d); the base state is the same in every column, so theta minus its median is
    th3d minus its median.  The depth is where theta minus the environment rises through -cold_k above the ground's
    cold air, linear between the two levels around it; cold air above a warmer gap is not counted."""

    zh = qf.zh
    th = qf.view("th3d")
    depth = np.zeros((qf.ny, qf.nx), np.float32)
    alive = None
    prev = first = None
    used = 0
    for k in range(qf.nz):
        d = np.asarray(th[k], np.float32)
        d = d - np.float32(np.median(d))
        cold = d <= -cold_k
        used = k + 1
        if k == 0:
            first = d
            alive = cold
            depth[alive] = zh[0]
        else:
            stop = alive & ~cold
            share = np.clip((-cold_k - prev) / np.where(d - prev == 0, 1e-9, d - prev), 0.0, 1.0)
            depth[stop] = (zh[k - 1] + share * (zh[k] - zh[k - 1]))[stop]
            alive = alive & cold
            depth[alive] = zh[k]
        prev = d
        if not alive.any():
            break
    return depth, first, used


def depth_position(d: np.ndarray | float) -> np.ndarray:
    """Where a depth sits on the colours, 0..1: the square root of its share of :data:`DEPTH_RANGE_M`."""

    return np.sqrt(np.clip(np.asarray(d, dtype=np.float64) / DEPTH_RANGE_M, 0.0, 1.0))


#: Cold-pool depth colours (placed by square root): night blue to ice to white, rising in lightness.
DEPTH_STOPS = ((0.0, "#0b1030"), (0.18, "#1b2f78"), (0.38, "#2f5fc0"), (0.58, "#5f9be6"), (0.78, "#a9d3f5"),
               (1.0, "#f4fbff"))


def depth_colours(t: np.ndarray) -> np.ndarray:
    pos = np.array([p for p, _ in DEPTH_STOPS])
    cols = np.array([S.hex_rgb(c) for _, c in DEPTH_STOPS])
    return np.stack([np.interp(t, pos, cols[:, ch]) for ch in range(3)], axis=-1).astype(np.float32)


def step_up(side: float) -> float:
    """The first of 40, 60, 80, 120, 160, 240 km and so on (:data:`COLD_STEPS_M`, doubled) at least ``side``."""

    k = 0
    while True:
        for step in COLD_STEPS_M:
            if step * 2 ** k >= side - 1e-6:
                return step * 2 ** k
        k += 1


def cold_square(extent: tuple[float, float],
                boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float] | None:
    """The square (x0, y0, width, height, m) that holds all of ``boxes`` (the cold air of this and the earlier saved
    times, [x0, y0, x1, y1] each) with :data:`COLD_MARGIN_M` around it, at least :data:`COLD_BOX_M` across and
    rounded up by :func:`step_up`, kept inside the area; None when there is no cold air yet.  The boxes only add up
    over a run, so the square never shrinks from one saved time to the next."""

    if not boxes:
        return None
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    side = step_up(max(COLD_BOX_M, max(x1 - x0, y1 - y0) + 2.0 * COLD_MARGIN_M))
    return around(extent, (0.5 * (x0 + x1), 0.5 * (y0 + y1)), side)


#: The domain's edges by the side of the array they are on: (name, row slice, column slice); row 0 is the south.
EDGES = (("north", -1, slice(None)), ("south", 0, slice(None)), ("east", slice(None), -1),
         ("west", slice(None), 0))


def edges_reached(cold: np.ndarray) -> list[str]:
    """The domain's edges the cold air touches (north, south, east, west, in that order)."""

    return [name for name, rows, cols in EDGES if bool(np.any(cold[rows, cols]))]


def _and(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def product_cold_pool(L: "P.Looks", out: Path) -> dict[str, Any]:
    """Cold-pool depth, in the square that holds all the cold air so far (the whole area when there is none)."""

    refuse(L, "cold-pool-depth")
    f = L.f
    depth, first, used = cold_depth(f)
    history = getattr(L, "history", None) or {}
    before = list(history.get("before") or ())
    notes = ColdArea(history.get("state_dir"))
    if history.get("open") is not None:
        notes.catch_up(before, history["open"])
    if not notes.has(f):
        notes.rows[f.key] = {"t": float(f.t), "box": _cold_box(f, first <= -COLD_K)}
        notes.changed = True
    notes.save()
    square = cold_square(f.extent, notes.boxes([key for _t, key, _p, _s in before] + [f.key], f.t))
    view = square if square is not None else (0.0, 0.0, f.extent[0], f.extent[1])
    box = panel_box(P.W // 2 - 400, 165, 800, 800, view)
    # the words beside the picture keep their places when the area shown is wider than it is tall
    bottom = 165 + 800
    ds = P.sample(depth, box, f.dx, f.dy)
    rgb = depth_colours(depth_position(ds))
    rgb[ds <= 0.0] = np.array(S.hex_rgb("void"), np.float32)
    img = S.to8(rgb)
    img = P.blend(img, P.line_alpha(ds, every=DEPTH_LINES_M, width=1.1, far=400.0), "title", 0.35)
    img = P.blend(img, P.line_alpha(ds, 0.5, width=1.3), "muted", 0.9)
    img, spacing = P.ground_ticks(img, box, f.domainlocx, f.domainlocy)
    cv = P.canvas()
    P.paste(cv, img, box)
    cv = P.outline(cv, box)
    cold = depth > 0.0
    area = float(cold.sum() * L.cell_km2())
    deepest = float(depth.max())
    z0 = L.z0
    whole = square is None or (view[2] >= f.extent[0] and view[3] >= f.extent[1])
    shown = (f"the whole {square_words((0.0, 0.0, *f.extent))} domain" if whole
             else f"{square_words(view)} around all the cold air so far")
    data = S.data_line(L.facts.name, f"{S.fmt_grid(f.dx)} grid", f"θ at least {COLD_K:g} K below the air around it")
    ov = P.overlay(L, "cold-pool-depth", data,
                   level=f"ground up, lowest level {S.fmt_height(z0)}", wind="",
                   notes=S.data_line(f"cold: θ at least {COLD_K:g} K below the median θ of the whole area at its "
                                     "height", "from the ground up, linear between levels", L.frame_words(),
                                     ground_note(spacing)))
    rows = [("deepest column of cold air", S.fmt_height(deepest) if deepest > 0 else "none"),
            (f"area with cold air at {S.fmt_height(z0)}", f"{area:,.0f} km²")]
    reaches = edges_reached(cold)
    facts: dict[str, Any] = {"threshold_k": COLD_K, "environment": "median theta of the whole area at each height",
                             "depth_max_m": deepest, "cold_area_km2": area, "levels_read": used,
                             "range_m": DEPTH_RANGE_M, "scale": "square root", "lowest_level_m": z0,
                             "box_m": list(view), "reaches_edges": reaches}
    if cold.any():
        median = float(np.median(depth[cold]))
        coldest = float(first.min())
        rows += [("median depth of the cold air", S.fmt_height(median)),
                 (f"coldest at {S.fmt_height(z0)}, below the air around it", f"{S.fmt_num(-coldest, 1)} K")]
        facts.update({"depth_median_m": median, "depth_p95_m": float(np.percentile(depth[cold], 95.0)),
                      "coldest_minus_environment_k": coldest})
    x, y = ov.x0, box.y + ov.U(10)
    ov.readout((x, y), rows)
    share = float(cold.mean())
    facts["cold_share_of_area"] = share
    cautions: list[str] = []
    if share > CROWDED_SHARE:
        # the median is then partly the cold air itself, so "the air around it" is cooler than the storm's
        # surroundings and the depths come out too shallow: the picture says so
        cautions.append(f"the cold air covers {100 * share:.0f} % of the area at {S.fmt_height(z0)}, so the median "
                        "is partly cold air: depths here read too shallow")
    if reaches:
        # the cold air goes on past the domain's edge, where nothing is known: the area and depths stop there
        cautions.append(f"the cold air reaches the domain's {_and(reaches)} edge{'s' if len(reaches) > 1 else ''}: "
                        "the area and the depths count only what is inside")
    wy, room = y + ov.U(300), P.W // 2 - 400 - ov.U(24) - x
    for caution in cautions:
        for part in ov.wrap(caution, "micro", room):
            ov.text((x, wy), part, "micro", colour="gold", anchor="ls", halo=False)
            wy += ov.U(20)
        wy += ov.U(14)
    ov.text((x, bottom - ov.U(84)), f"shown: {shown}", "micro", colour="data", anchor="ls", halo=False)
    ov.text((x, bottom - ov.U(62)), "colder air above a warmer gap is not counted", "micro",
            colour="muted", anchor="ls", halo=False)
    ov.text((x, bottom - ov.U(40)), f"grey line: the cold air's edge; white lines every {DEPTH_LINES_M:g} m",
            "micro", colour="muted", anchor="ls", halo=False)
    P.scale_bar(ov, x, bottom - ov.U(8), box)
    wr = ov.U(400)
    by = bottom - ov.U(40)
    colour_bar(ov, depth_colours(np.linspace(0.0, 1.0, 256)), (ov.x1 - wr, by, ov.x1, by + max(3, ov.U(7))),
               [(float(depth_position(v)), label) for v, label in
                ((0.0, "0"), (250.0, "250"), (1000.0, "1000"), (DEPTH_RANGE_M, f"{DEPTH_RANGE_M:g} m"))],
               title="depth of the cold air, square-root scale")
    P.save(ov, cv, out)
    return facts


RENDERERS: dict[str, Callable[..., dict[str, Any]]] = {
    "storm-relative-flow": product_flow,
    "streamwise-spin": product_spin,
    "cold-pool-depth": product_cold_pool,
}


__all__ = ["COLD_K", "ColdArea", "FINEST_DX_M", "IDS", "Motion", "RENDERERS", "StormTrack", "cold_depth",
           "cold_square", "edges_reached", "horizontal_spin", "motion_for", "refuse", "spin_face", "storm_peaks",
           "storm_place", "step_up"]
