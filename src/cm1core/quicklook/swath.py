"""Swaths: the strongest lowest-level wind (and 2-5 km updraft helicity) each place on the ground has had.

Two sources, the better one first:

``model``
    The model's own run-long maxima, updated every step.  When the domain moves, GM1 keeps them on a ground-fixed
    canvas (CM1's ``sws2``/``shs2``; the canvas is placed exactly from ``domainlocx/domainlocy``, see
    ``tornado_diagnostics.GroundAnchor``); when it does not, the domain is the ground and ``sws``/``shs`` are
    already ground-fixed.  CM1's ``sws`` is ground-relative (solve3.F:855-866 adds umove and vmove).
``frames``
    Without those words, the quick-look builds its own: the lowest-level ground-relative wind speed (and CM1's
    ``uh``) of each saved frame, placed on the ground at the frame's own position.  Only the saved frames count,
    and the picture says so.

A CM1 netCDF file (:mod:`.ncread`) holds CM1's translated swaths (``sws2``/``shs2``, written when the domain
moves) on the domain's own cells: the ground under the domain at that time.  Each saved time's window is placed on
the ground where the domain is (``model-window``), so the swath covers the ground the domain has passed over.

Either way the quick-look keeps a running maximum over the run's frames, oldest first.  With the model's words
this changes nothing within one segment (each is already the maximum since the segment began) and it carries the
maxima across a resume even where a restart does not.  The running maximum is kept in ``.state/swath.npz`` beside
the pictures, with the frames it has seen; a run that leaves a branch (a resume from an earlier restart) is folded
again from its own frames.

A run whose frames have the model's wind maximum but not its helicity maximum (``sws``/``sws2`` without
``shs``/``shs2``: GM1 writes no ``shs`` on stretched levels, nor when ``output_shs = 0``) keeps the wind from the
model and builds the helicity from the saved frames, each frame's 2-5 km updraft helicity summed from w and the
winds by the picture tool with CM1's own calcuh method (:meth:`.frame.QuickFrame.uh`; CM1's ``uh`` when the frame
has it), and the picture says which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .frame import QuickFrame

STATE = "swath.npz"
#: The kept maxima's layout: a state written before the helicity could come from the frames beside the model's
#: wind (no ``uh_sources``) is folded again from the run's frames, so a run's helicity is never part-blank.
STATE_VERSION = 2
#: Where a helicity plane came from: the model's every-step maximum, or one saved frame's own helicity.
UH_MODEL, UH_CM1, UH_CALCUH = "model", "uh", "calcuh"
#: The words for a helicity built from the saved frames (``UH_CM1``: the frame's own CM1 uh; ``UH_CALCUH``: summed
#: from the frame's w and winds by the picture tool, CM1 r21.1 misclibs.F calcuh).
UH_WORDS = {UH_CM1: "CM1's uh of each saved frame",
            UH_CALCUH: "summed from w and the winds by the picture tool (CM1 calcuh method)"}
#: The same in fewer words, for a panel too narrow for :data:`UH_WORDS` beside its colour legend.
UH_SHORT = {UH_CM1: "CM1's uh of each saved frame", UH_CALCUH: "CM1 calcuh sum by the picture tool"}


def _round_half_away(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0.0 else -int(math.floor(-value + 0.5))


@dataclass
class Plane:
    """One frame's contribution: arrays on ground cells starting at ``origin`` (cells from the start corner).

    ``uh_origin`` is where the helicity plane starts when it is not on the wind's cells (one frame's own helicity
    beside the model's ground canvas of wind), and ``uh_source`` where it came from (``UH_*``)."""

    wind: np.ndarray
    uh: np.ndarray | None
    origin: tuple[int, int]
    source: str
    uh_origin: tuple[int, int] | None = None
    uh_source: str | None = None


@dataclass
class Swath:
    wind: np.ndarray
    uh: np.ndarray | None
    origin_cells: tuple[int, int]
    dx: float
    dy: float
    t0: float
    frames_used: int
    source: str
    method: str
    #: Set when the wind is the model's own maximum and the helicity is not (it was built from the saved frames):
    #: the helicity's own method, in one line (the record), in the two lines said under its panel, and in two
    #: shorter ones for a panel too narrow for those beside its colour legend.
    uh_method: str = ""
    uh_lines: tuple[str, ...] = ()
    uh_short: tuple[str, ...] = ()
    #: Built from translated swaths that hold only the ground under the domain at each saved time (CM1 netCDF
    #: files): the picture says what that leaves out (:data:`WINDOW_NOTE`).
    window: bool = False

    @property
    def origin_m(self) -> tuple[float, float]:
        return self.origin_cells[0] * self.dx, self.origin_cells[1] * self.dy

    @property
    def extent_m(self) -> tuple[float, float]:
        return self.wind.shape[1] * self.dx, self.wind.shape[0] * self.dy


#: What a swath or track built from CM1 netCDF files leaves out.  Each file holds the translated swath (sws2, shs2,
#: sgs2, svs2) only on the ground under the domain at its time; what the run added to ground the domain then left
#: before the next file is in no file (and all of it, for ground the domain left before the first file).
WINDOW_NOTE = ("Each file holds the swath only under the domain at its time: what the run added to ground the domain "
               "left before the next file is missing (before the first file, all of it), so edges can be stepped "
               "and areas smaller than the run's own.")


def canvas_origin(qf: QuickFrame) -> tuple[int, int] | None:
    """Ground cell of the model canvas's (0, 0) cell, from the frame's swath record (GroundAnchor.origin_cells).
    A CM1 netCDF file's translated swaths are the ground under the domain now: their (0, 0) cell is the domain's."""

    if window_swaths(qf):
        return _round_half_away(qf.domainlocx / qf.dx), _round_half_away(qf.domainlocy / qf.dy)
    swaths = ((qf.publication.get("tornado_diagnostics") or {}).get("swaths")) or {}
    anchor = swaths.get("moving_frame") or {}
    try:
        seconds = float(anchor["dt_s"]) * int(anchor["total_steps"])
        fx = _round_half_away(float(anchor["u_move_m_s"]) * seconds / qf.dx)
        fy = _round_half_away(float(anchor["v_move_m_s"]) * seconds / qf.dy)
    except (KeyError, TypeError, ValueError):
        return None
    return min(0, fx), min(0, fy)


def frame_origin(qf: QuickFrame) -> tuple[int, int]:
    """Ground cell of the frame's own (0, 0) cell: where the domain is now."""

    return _round_half_away(qf.domainlocx / qf.dx), _round_half_away(qf.domainlocy / qf.dy)


def frame_uh(qf: QuickFrame) -> tuple[np.ndarray | None, str | None]:
    """This saved frame's own 2-5 km updraft helicity and where it came from (``UH_CM1`` or ``UH_CALCUH``), or
    (None, None) when the frame has neither CM1's uh nor the w and winds to sum it from."""

    if not (qf.has("uh") or (qf.has("w3d") and qf.has("u3d") and qf.has("v3d"))):
        return None, None
    plane, source = qf.uh()
    return plane, (UH_CM1 if source == "CM1 uh" else UH_CALCUH)


def window_swaths(qf: QuickFrame) -> bool:
    """Whether the frame's translated swaths are a window under the domain (a CM1 netCDF file), not GM1's canvas."""

    return bool(getattr(qf.frame, "swath_window", False))


def plane_of(qf: QuickFrame) -> Plane | None:
    """This frame's swath contribution, from the best source it has.

    The model's wind maximum without its helicity maximum (no ``shs``/``shs2``) takes this frame's own helicity,
    placed where the domain is now."""

    if qf.has("sws2"):
        origin = canvas_origin(qf)
        if origin is not None:
            source = "model-window" if window_swaths(qf) else "model"
            if qf.has("shs2"):
                return Plane(qf.get("sws2"), qf.get("shs2"), origin, source, uh_source=UH_MODEL)
            uh, kind = frame_uh(qf)
            return Plane(qf.get("sws2"), uh, origin, source, frame_origin(qf), kind)
    if qf.has("sws") and not qf.domain_moves:
        if qf.has("shs"):
            return Plane(qf.get("sws"), qf.get("shs"), (0, 0), "model-fixed", uh_source=UH_MODEL)
        uh, kind = frame_uh(qf)
        return Plane(qf.get("sws"), uh, (0, 0), "model-fixed", frame_origin(qf), kind)
    if not (qf.has("u3d") and qf.has("v3d")):
        return None
    u, v = qf.uv_level(0, ground=True)
    uh, kind = frame_uh(qf)       # u3d and v3d are here, so: CM1's uh, else the sum when the frame has w
    return Plane(np.hypot(u, v).astype(np.float32), uh, frame_origin(qf), "frames", uh_source=kind)


def _merge(acc: np.ndarray | None, acc_origin: tuple[int, int], plane: np.ndarray,
           origin: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Running maximum of ``plane`` into ``acc`` on the ground, growing the canvas as needed (unvisited = 0)."""

    ph, pw = plane.shape
    if acc is None:
        return np.maximum(plane, 0.0).astype(np.float32), origin
    ah, aw = acc.shape
    x0, y0 = min(acc_origin[0], origin[0]), min(acc_origin[1], origin[1])
    x1 = max(acc_origin[0] + aw, origin[0] + pw)
    y1 = max(acc_origin[1] + ah, origin[1] + ph)
    if (x0, y0, x1 - x0, y1 - y0) != (acc_origin[0], acc_origin[1], aw, ah):
        grown = np.zeros((y1 - y0, x1 - x0), np.float32)
        grown[acc_origin[1] - y0:acc_origin[1] - y0 + ah, acc_origin[0] - x0:acc_origin[0] - x0 + aw] = acc
        acc, acc_origin = grown, (x0, y0)
    sl = (slice(origin[1] - acc_origin[1], origin[1] - acc_origin[1] + ph),
          slice(origin[0] - acc_origin[0], origin[0] - acc_origin[0] + pw))
    np.maximum(acc[sl], plane, out=acc[sl])
    return acc, acc_origin


def reserved_canvas(cells: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """An empty ground canvas over cells [x0, x1) x [y0, y1), and its origin."""

    x0, y0, x1, y1 = cells
    return np.zeros((y1 - y0, x1 - x0), np.float32), (x0, y0)


class SwathBuilder:
    """The running maxima over a run's frames, in time order, kept in ``state_dir/swath.npz``."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.keys: list[str] = []
        self.times: list[float] = []
        self.wind: np.ndarray | None = None
        self.uh: np.ndarray | None = None
        self.origin = (0, 0)
        self.uh_origin = (0, 0)
        self.sources: set[str] = set()
        self.uh_sources: set[str] = set()
        self.dx = self.dy = 0.0
        #: Ground cells (x0, y0, x1, y1) the canvas covers from its first frame on, when they are known before the
        #: frames are drawn (CM1 netCDF files: every saved time is there); None: the canvas grows as frames come.
        self.reserved: tuple[int, int, int, int] | None = None
        self._load()

    # -- persistence ------------------------------------------------------------
    def _load(self) -> None:
        path = self.state_dir / STATE
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                wind = np.array(data["wind"], np.float32)
                uh = np.array(data["uh"], np.float32) if "uh" in data.files else None
        except (OSError, KeyError, ValueError):
            return
        if meta.get("version") != STATE_VERSION:
            return      # kept by an earlier version: folded again from the run's frames
        self.wind, self.uh = wind, uh
        self.keys = list(meta["keys"])
        self.times = [float(t) for t in meta["times"]]
        self.origin = tuple(meta["origin"])
        self.uh_origin = tuple(meta.get("uh_origin", meta["origin"]))
        self.sources = set(meta["sources"])
        self.uh_sources = set(meta.get("uh_sources") or ())
        self.dx, self.dy = float(meta["dx"]), float(meta["dy"])

    def save(self) -> None:
        if self.wind is None:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"version": STATE_VERSION, "keys": self.keys, "times": self.times,
                           "origin": list(self.origin), "uh_origin": list(self.uh_origin),
                           "sources": sorted(self.sources), "uh_sources": sorted(self.uh_sources),
                           "dx": self.dx, "dy": self.dy})
        tmp = self.state_dir / f".{STATE}.partial.npz"
        arrays = {"wind": self.wind, "meta": np.array(meta)}
        if self.uh is not None:
            arrays["uh"] = self.uh
        np.savez(tmp, **arrays)
        os.replace(tmp, self.state_dir / STATE)

    def reset(self) -> None:
        self.keys, self.times, self.wind, self.uh, self.sources = [], [], None, None, set()
        self.uh_sources = set()
        self.origin = self.uh_origin = (0, 0)

    # -- accumulation -----------------------------------------------------------
    def add(self, qf: QuickFrame) -> None:
        plane = plane_of(qf)
        self.keys.append(qf.key)
        self.times.append(qf.t)
        self.dx, self.dy = qf.dx, qf.dy
        if plane is None:
            return
        self.sources.add(plane.source)
        if self.wind is None and self.reserved is not None:
            self.wind, self.origin = reserved_canvas(self.reserved)
        self.wind, self.origin = _merge(self.wind, self.origin, plane.wind, plane.origin)
        if plane.uh is not None:
            at = plane.uh_origin if plane.uh_origin is not None else plane.origin
            self.uh, self.uh_origin = _merge(self.uh, self.uh_origin, plane.uh, at)
            if plane.uh_source:
                self.uh_sources.add(plane.uh_source)

    def catch_up(self, before: Sequence[tuple[float, str, Path, int]], open_frame) -> None:
        """Make the running maxima cover exactly the frames ``before`` (time, key, path, segment), oldest first.

        When what is kept is not those frames' leading part (a branch the run has left), it is folded again."""

        keys = [key for _t, key, _p, _s in before]
        if self.keys != keys[:len(self.keys)] or len(self.keys) > len(keys):
            self.reset()
        for t, key, path, segment in before[len(self.keys):]:
            qf = open_frame(path, key, segment)
            try:
                self.add(qf)
            finally:
                qf.close()

    def swath(self, interval_s: float | None) -> Swath | None:
        if self.wind is None:
            return None
        uh = None
        if self.uh is not None:
            # the updraft-helicity canvas on the wind canvas's cells
            uh = np.zeros_like(self.wind)
            dx0 = self.uh_origin[0] - self.origin[0]
            dy0 = self.uh_origin[1] - self.origin[1]
            h, w = self.uh.shape
            ys, xs = max(0, dy0), max(0, dx0)
            ye, xe = min(uh.shape[0], dy0 + h), min(uh.shape[1], dx0 + w)
            if ye > ys and xe > xs:
                uh[ys:ye, xs:xe] = self.uh[ys - dy0:ye - dy0, xs - dx0:xe - dx0]
        every = f" (every {interval_s:g} s)" if interval_s else ""
        uh_method, uh_lines, uh_short = "", (), ()
        if self.sources and self.sources <= {"model", "model-fixed", "model-window"}:
            source = "model"
            suffix = "2" if self.sources & {"model", "model-window"} else ""
            # CM1 netCDF files' translated swaths hold only the ground under the domain at each saved time
            where = ("(the files' {names}, under the domain at each saved time)" if "model-window" in self.sources
                     else "on the ground (CM1's {names})")
            if self.uh is not None and self.uh_sources - {UH_MODEL}:
                # the model's wind maximum, and a helicity built from the saved frames (the frames carry no shs)
                method = "wind: the model's every-step maximum " + where.format(names=f"sws{suffix}")
                words = " and ".join(UH_WORDS[k] for k in (UH_CALCUH, UH_CM1) if k in self.uh_sources)
                why = (f", and CM1's shs{suffix} where a frame had it" if UH_MODEL in self.uh_sources
                       else f": the frames carry no CM1 shs{suffix}")
                uh_method = f"helicity {words}, at each saved frame only{every}{why}"
                uh_lines = (f"helicity {words}", f"at each saved frame only{every}{why}")
                short = " and ".join(UH_SHORT[k] for k in (UH_CALCUH, UH_CM1) if k in self.uh_sources)
                uh_short = (f"helicity: {short}",
                            f"saved frames only{every}, " + (f"CM1's shs{suffix} where saved"
                                                              if UH_MODEL in self.uh_sources
                                                              else f"no CM1 shs{suffix}"))
            else:
                names = f"sws{suffix} and shs{suffix}" if self.uh is not None else f"sws{suffix}"
                method = "the model's every-step maximum " + where.format(names=names)
        else:
            source = "frames"
            method = f"the maximum over the saved frames only{every}, placed on the ground"
        return Swath(self.wind, uh, self.origin, self.dx, self.dy, self.times[0] if self.times else 0.0,
                     len(self.keys), source, method, uh_method, uh_lines, uh_short,
                     window="model-window" in self.sources)


__all__ = ["Plane", "STATE_VERSION", "Swath", "SwathBuilder", "UH_SHORT", "UH_WORDS", "WINDOW_NOTE", "canvas_origin",
           "frame_origin", "frame_uh", "plane_of", "reserved_canvas", "window_swaths"]
