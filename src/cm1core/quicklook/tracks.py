"""Tracks on the ground: the most hail near the ground, and the strongest near-ground spin, each place has had.

They are kept exactly as the wind swath is (:mod:`.swath`), from the best source a run has:

``model``
    The model's own run-long maxima, updated every step: ``sgs``/``svs`` (CM1's lowest-level maximum of qg and of
    vertical vorticity, solve3.F:1003-1036 and :881-906), on the ground-anchored canvas ``sgs2``/``svs2`` when
    the domain moves.  A run without ``sgs`` (``output_sgs = 0``) gets its hail track from the frames.
``frames``
    Without those words, the quick-look keeps its own: the lowest-level qg (or the lowest-level vertical
    vorticity) of each saved frame, placed on the ground at the frame's own position.  Only the saved frames
    count, and the picture says so.

The running maximum is kept in ``.state/track-<id>.npz`` beside the pictures, with the frames it has seen, and is
folded again when a run leaves a branch, as the swath is.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .frame import QuickFrame
from .swath import _merge, _round_half_away, canvas_origin

HAIL = "hail-track"
SPIN = "spin-track"
TRACK_IDS = (HAIL, SPIN)


@dataclass(frozen=True)
class TrackSpec:
    id: str
    model: str            # CM1's name: the ground canvas is this name + "2"
    what: str             # plain words for the method line


SPECS = {
    HAIL: TrackSpec(HAIL, "sgs", "lowest-level qg"),
    SPIN: TrackSpec(SPIN, "svs", "lowest-level vertical vorticity"),
}


@dataclass
class Track:
    values: np.ndarray             # on ground cells from origin_cells; unvisited = 0
    origin_cells: tuple[int, int]
    dx: float
    dy: float
    t0: float
    frames_used: int
    source: str                    # "model" or "frames"
    method: str
    interval_s: float | None

    @property
    def origin_m(self) -> tuple[float, float]:
        return self.origin_cells[0] * self.dx, self.origin_cells[1] * self.dy

    @property
    def extent_m(self) -> tuple[float, float]:
        return self.values.shape[1] * self.dx, self.values.shape[0] * self.dy


def frame_plane(qf: QuickFrame, track_id: str) -> np.ndarray | None:
    """This frame's own lowest-level field for the track (qg in kg/kg, or vertical vorticity in 1/s)."""

    if track_id == HAIL:
        try:
            if not qf.has_species("qg"):
                return None
        except ValueError:
            return None
        return qf.species("qg", 0)
    if not (qf.has("u3d") and qf.has("v3d")) and not qf.has("zvort_sfc"):
        return None
    return qf.zeta_level(0)[0]


def plane_of(qf: QuickFrame, track_id: str) -> tuple[np.ndarray, tuple[int, int], str] | None:
    """(plane, ground origin in cells, source) of this frame's contribution, from the best source it has."""

    spec = SPECS[track_id]
    if qf.has(spec.model + "2"):
        origin = canvas_origin(qf)
        if origin is not None:
            return np.maximum(qf.get(spec.model + "2"), 0.0), origin, "model"
    if qf.has(spec.model) and not qf.domain_moves:
        return np.maximum(qf.get(spec.model), 0.0), (0, 0), "model-fixed"
    plane = frame_plane(qf, track_id)
    if plane is None:
        return None
    origin = (_round_half_away(qf.domainlocx / qf.dx), _round_half_away(qf.domainlocy / qf.dy))
    return np.maximum(plane, 0.0).astype(np.float32), origin, "frames"


class TrackBuilder:
    """The running maximum of one track over a run's frames, in time order, kept in ``state_dir``."""

    def __init__(self, state_dir: Path, track_id: str) -> None:
        self.state_dir = Path(state_dir)
        self.id = track_id
        self.keys: list[str] = []
        self.times: list[float] = []
        self.values: np.ndarray | None = None
        self.origin = (0, 0)
        self.sources: set[str] = set()
        self.dx = self.dy = 0.0
        self._load()

    @property
    def path(self) -> Path:
        return self.state_dir / f"track-{self.id}.npz"

    def _load(self) -> None:
        try:
            with np.load(self.path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                values = np.array(data["values"], np.float32)
        except (OSError, KeyError, ValueError):
            return
        self.values = values
        self.keys = list(meta["keys"])
        self.times = [float(t) for t in meta["times"]]
        self.origin = tuple(meta["origin"])
        self.sources = set(meta["sources"])
        self.dx, self.dy = float(meta["dx"]), float(meta["dy"])

    def save(self) -> None:
        if self.values is None:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"keys": self.keys, "times": self.times, "origin": list(self.origin),
                           "sources": sorted(self.sources), "dx": self.dx, "dy": self.dy})
        tmp = self.state_dir / f".track-{self.id}.partial.npz"
        np.savez(tmp, values=self.values, meta=np.array(meta))
        os.replace(tmp, self.path)

    def reset(self) -> None:
        self.keys, self.times, self.values, self.sources = [], [], None, set()
        self.origin = (0, 0)

    def add(self, qf: QuickFrame) -> None:
        found = plane_of(qf, self.id)
        self.keys.append(qf.key)
        self.times.append(qf.t)
        self.dx, self.dy = qf.dx, qf.dy
        if found is None:
            return
        plane, origin, source = found
        self.sources.add(source)
        self.values, self.origin = _merge(self.values, self.origin, plane, origin)

    def catch_up(self, before: Sequence[tuple[float, str, Path, int]], open_frame: Any) -> None:
        keys = [key for _t, key, _p, _s in before]
        if self.keys != keys[:len(self.keys)] or len(self.keys) > len(keys):
            self.reset()
        for _t, key, path, segment in before[len(self.keys):]:
            qf = open_frame(path, key, segment)
            try:
                self.add(qf)
            finally:
                qf.close()

    def track(self, interval_s: float | None) -> Track | None:
        if self.values is None:
            return None
        spec = SPECS[self.id]
        if self.sources and self.sources <= {"model", "model-fixed"}:
            name = spec.model + ("2" if "model" in self.sources else "")
            source, method = "model", f"the model's every-step maximum on the ground (CM1's {name})"
        else:
            every = f" (every {interval_s:g} s)" if interval_s else ""
            source = "frames"
            method = f"the {spec.what} of the saved frames only{every}, placed on the ground"
        return Track(self.values, self.origin, self.dx, self.dy, self.times[0] if self.times else 0.0,
                     len(self.keys), source, method, interval_s)


# ----------------------------------------------------------------------------------------------- pictures
HAIL_RANGE = (0.01, 10.0)   # g/kg, fixed, drawn on a log scale (1 km runs keep a few tenths near the ground,
                             # finer ones several g/kg: one linear range would draw one of them black)
TITLE = {HAIL: "Hail near the ground so far", SPIN: "Strongest near-ground spin so far"}


def dense_word(config: dict[str, Any]) -> str:
    """'hail' or 'graupel': what the run's dense-ice category is (the frame's publication record)."""

    category = (config.get("dense_ice") or {}).get("category") if isinstance(config.get("dense_ice"), dict) else None
    return category if category in ("hail", "graupel") else "graupel"


def product_track(L: Any, out: Path, track: Track | None, track_id: str) -> dict[str, Any]:
    """The track on the ground, in the swath's layout: one panel, the domain at the start (dim box) and now
    (grey box), a fixed stated range."""

    from . import products as P
    from . import style as S

    f = L.f
    if track is None:
        if track_id == HAIL:   # a scheme without CM1's qg (nqg = 0, param.F:4728 turns output_sgs off)
            raise P.NotDrawn("This run has no hail or graupel to follow.")
        raise P.NotDrawn("This run's frames carry no wind near the ground to build the track from.")
    lx, ly = track.extent_m
    box = P.fit_box(lx, ly, P.W // 2, 190, 1180, 700)
    box = P.Box(box.x, box.y, box.w, track.origin_m[0], track.origin_m[1], box.span, h=box.h)
    z0 = L.z0
    since = "since the start" if track.source == "model" or track.t0 <= 0.0 else f"since {S.fmt_clock(track.t0)}"
    if track_id == HAIL:
        dense = dense_word(f.config)
        grams = track.values * 1000.0
        values = np.log10(np.maximum(grams, 1.0e-6)).astype(np.float32)
        lo, rng, ramp, unit = math.log10(HAIL_RANGE[0]), math.log10(HAIL_RANGE[1]), "ice", "g/kg"
        title = f"{dense.capitalize()} near the ground so far"
        label = f"most {dense} in the air at {S.fmt_height(z0)} {since}"
        ticks = [(-2.0, "0.01"), (-1.0, "0.1"), (0.0, "1"), (1.0, "10 g/kg")]
        legend = f"{dense} mixing ratio at {S.fmt_height(z0)}, log scale"
        extra = "the most in the air near the ground, not the size of the stones"
        cmax = float(grams.max())
        facts: dict[str, Any] = {"qg_max_g_kg": cmax, "dense_ice": dense}
        readout = [(f"most {dense} near the ground", f"{S.fmt_num(cmax, 2 if cmax < 1 else 1)} g/kg"),
                   ("area above 0.1 g/kg", f"{float((grams >= 0.1).sum()) * track.dx * track.dy / 1e6:,.0f} km²")]
    else:
        values = track.values
        lo, rng, ramp, unit = 0.0, P.spin_range(f.dx), "ember", S.PER_S
        title = TITLE[SPIN]
        label = f"strongest counter-clockwise spin at {S.fmt_height(z0)} {since}"
        ticks = [(0.0, "0"), (rng / 2, S.fmt_num(rng / 2, 3)), (rng, f"{rng:g} {S.PER_S}")]
        legend = f"spin at {S.fmt_height(z0)}, fixed at this dx"
        extra = f"features smaller than about {S.fmt_km(6 * f.dx)} are not resolved"
        cmax = float(values.max())
        facts = {"zeta_max": cmax, "range_per_s": rng, "resolved_m": 6.0 * f.dx}
        readout = [("strongest spin near the ground", f"{S.fmt_num(cmax, 3)} {S.PER_S}"),
                   (f"above half the range ({S.fmt_num(rng / 2, 3)})",
                    f"{float((values >= rng / 2).sum()) * track.dx * track.dy / 1e6:,.0f} km²")]
    sampled = P.sample(values, box, track.dx, track.dy, origin=track.origin_m)
    cv = P.canvas()
    P.paste(cv, S.to8(S.colours(ramp, sampled, lo, rng)), box)
    cv = P.outline(cv, box)
    from PIL import Image, ImageDraw

    im = Image.fromarray(cv)
    d = ImageDraw.Draw(im)
    dom_x, dom_y = f.extent
    for (gx, gy), colour in (((0.0, 0.0), "dim"), ((f.domainlocx, f.domainlocy), "muted")):
        xa, ya = box.to_px(gx, gy + dom_y)
        xb, yb = box.to_px(gx + dom_x, gy)
        d.rectangle((xa, ya, xb, yb), outline=S.rgb8(colour), width=1)
    cv = np.asarray(im).copy()
    ov = S.Overlay((P.W, P.H))
    ov.title(title, S.data_line(L.facts.name, f"{S.fmt_grid(f.dx)} grid",
                                f"on the ground, {lx / 1000:g} × {ly / 1000:g} km", f"strongest {since}"))
    ov.clock((getattr(L.facts, "clock", None) or S.fmt_clock)(f.t))
    ov.footer(L.facts.footer(t=f.t, level=f"lowest level {S.fmt_height(z0)}", wind="",
                             notes=S.data_line(track.method, "dim box: the domain at the start, grey box: now",
                                               extra)))
    ov.mark()
    ov.text((box.x, box.y - ov.U(10)), label, "label", colour="text", anchor="ls")
    x = ov.x0
    ov.readout((x, box.y + ov.U(10)) if box.x - x > ov.U(260) else (box.x + box.w + ov.U(30), box.y + ov.U(10)),
               readout)
    ov.ramp_bar(ramp, (box.x + box.w - ov.U(400), box.y + box.h + ov.U(14), box.x + box.w,
                       box.y + box.h + ov.U(14) + max(3, ov.U(7))), vmin=lo, vmax=rng, ticks=ticks, title=legend)
    P.scale_bar(ov, box.x, box.y + box.h + ov.U(20), box)
    P.save(ov, cv, out)
    facts.update({"source": track.source, "frames_used": track.frames_used, "since": since, "height_m": z0,
                  "unit": unit, "range": list(HAIL_RANGE) if track_id == HAIL else [0.0, rng]})
    return facts


__all__ = ["HAIL", "HAIL_RANGE", "SPECS", "SPIN", "TRACK_IDS", "Track", "TrackBuilder", "dense_word",
           "frame_plane", "plane_of", "product_track"]
