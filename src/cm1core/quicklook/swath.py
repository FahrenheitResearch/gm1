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

Either way the quick-look keeps a running maximum over the run's frames, oldest first.  With the model's words
this changes nothing within one segment (each is already the maximum since the segment began) and it carries the
maxima across a resume even where a restart does not.  The running maximum is kept in ``.state/swath.npz`` beside
the pictures, with the frames it has seen; a run that leaves a branch (a resume from an earlier restart) is folded
again from its own frames.
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


def _round_half_away(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0.0 else -int(math.floor(-value + 0.5))


@dataclass
class Plane:
    """One frame's contribution: arrays on ground cells starting at ``origin`` (cells from the start corner)."""

    wind: np.ndarray
    uh: np.ndarray | None
    origin: tuple[int, int]
    source: str


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

    @property
    def origin_m(self) -> tuple[float, float]:
        return self.origin_cells[0] * self.dx, self.origin_cells[1] * self.dy

    @property
    def extent_m(self) -> tuple[float, float]:
        return self.wind.shape[1] * self.dx, self.wind.shape[0] * self.dy


def canvas_origin(qf: QuickFrame) -> tuple[int, int] | None:
    """Ground cell of the model canvas's (0, 0) cell, from the frame's swath record (GroundAnchor.origin_cells)."""

    swaths = ((qf.publication.get("tornado_diagnostics") or {}).get("swaths")) or {}
    anchor = swaths.get("moving_frame") or {}
    try:
        seconds = float(anchor["dt_s"]) * int(anchor["total_steps"])
        fx = _round_half_away(float(anchor["u_move_m_s"]) * seconds / qf.dx)
        fy = _round_half_away(float(anchor["v_move_m_s"]) * seconds / qf.dy)
    except (KeyError, TypeError, ValueError):
        return None
    return min(0, fx), min(0, fy)


def plane_of(qf: QuickFrame) -> Plane | None:
    """This frame's swath contribution, from the best source it has."""

    if qf.has("sws2"):
        origin = canvas_origin(qf)
        if origin is not None:
            return Plane(qf.get("sws2"), qf.get("shs2") if qf.has("shs2") else None, origin, "model")
    if qf.has("sws") and not qf.domain_moves:
        return Plane(qf.get("sws"), qf.get("shs") if qf.has("shs") else None, (0, 0), "model-fixed")
    if not (qf.has("u3d") and qf.has("v3d")):
        return None
    u, v = qf.uv_level(0, ground=True)
    origin = (_round_half_away(qf.domainlocx / qf.dx), _round_half_away(qf.domainlocy / qf.dy))
    uh = qf.uh()[0] if (qf.has("uh") or qf.has("w3d")) else None
    return Plane(np.hypot(u, v).astype(np.float32), uh, origin, "frames")


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
        self.dx = self.dy = 0.0
        self._load()

    # -- persistence ------------------------------------------------------------
    def _load(self) -> None:
        path = self.state_dir / STATE
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                self.wind = np.array(data["wind"], np.float32)
                self.uh = np.array(data["uh"], np.float32) if "uh" in data.files else None
        except (OSError, KeyError, ValueError):
            return
        self.keys = list(meta["keys"])
        self.times = [float(t) for t in meta["times"]]
        self.origin = tuple(meta["origin"])
        self.uh_origin = tuple(meta.get("uh_origin", meta["origin"]))
        self.sources = set(meta["sources"])
        self.dx, self.dy = float(meta["dx"]), float(meta["dy"])

    def save(self) -> None:
        if self.wind is None:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"keys": self.keys, "times": self.times, "origin": list(self.origin),
                           "uh_origin": list(self.uh_origin), "sources": sorted(self.sources),
                           "dx": self.dx, "dy": self.dy})
        tmp = self.state_dir / f".{STATE}.partial.npz"
        arrays = {"wind": self.wind, "meta": np.array(meta)}
        if self.uh is not None:
            arrays["uh"] = self.uh
        np.savez(tmp, **arrays)
        os.replace(tmp, self.state_dir / STATE)

    def reset(self) -> None:
        self.keys, self.times, self.wind, self.uh, self.sources = [], [], None, None, set()
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
        self.wind, self.origin = _merge(self.wind, self.origin, plane.wind, plane.origin)
        if plane.uh is not None:
            self.uh, self.uh_origin = _merge(self.uh, self.uh_origin, plane.uh, plane.origin)

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
        if self.sources and self.sources <= {"model", "model-fixed"}:
            source = "model"
            suffix = "2" if "model" in self.sources else ""
            names = f"sws{suffix} and shs{suffix}" if self.uh is not None else f"sws{suffix}"
            method = f"the model's every-step maximum on the ground (CM1's {names})"
        else:
            source = "frames"
            every = f" (every {interval_s:g} s)" if interval_s else ""
            method = f"the maximum over the saved frames only{every}, placed on the ground"
        return Swath(self.wind, uh, self.origin, self.dx, self.dy, self.times[0] if self.times else 0.0,
                     len(self.keys), source, method)


__all__ = ["Plane", "Swath", "SwathBuilder", "canvas_origin", "plane_of"]
