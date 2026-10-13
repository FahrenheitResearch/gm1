"""Air rising into the storm: the run's own air parcels, from above and from the side, with 10-minute trails.

The parcels are the ones the model followed (``iprcl = 1``): their positions come from the run's parcel track,
written once at the end of the run (``<frames>/lagrangian-track``, :mod:`cm1core.cm1out.pdata`), read with the same
reader ``gpuwm-cm1 cm1out`` uses to write ``cm1out_pdata``.  Nothing is interpolated between the saved parcel
times except along each trail's straight segments, and the picture says so.

* Bright: the parcels that rise more than 3 km during the run, coloured by their height now.
* Faint: every other parcel (one in four), where it is now.
* Positions are the model's own x, y (in the domain, which moves with the storm when the run moves it) and z.

A run without parcels, or one still running, has none to show, and the look says why in plain words.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import products as P
from . import style as S

RISE_M = 3000.0
TRAIL_S = 600.0
HEIGHT_TOP = 14000.0
SENTINEL = -1.0e29


class NoParcels(P.NotDrawn):
    """The run has no parcel track to draw (in plain words)."""


def load(layout: Any) -> dict[str, Any]:
    """The run's merged parcel track: times (s) and x, y, z (m, the model's own coordinates), each (times, n)."""

    from ..cm1out import pdata

    tracks = pdata.lineage_tracks(layout)
    merged = pdata.merge(tracks)
    if merged is None:
        raise NoParcels("This run did not follow air parcels.")
    names = merged.track.names
    for need in ("x", "y", "z"):
        if need not in names:
            raise NoParcels(f"This run's parcel track has no {need} position.")
    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    iw = names.index("w") if "w" in names else None
    xs, ys, zs, ws = [], [], [], []
    for position in range(len(merged.rows)):
        record = merged.record(position)
        xs.append(np.asarray(record[ix], np.float32))
        ys.append(np.asarray(record[iy], np.float32))
        zs.append(np.asarray(record[iz], np.float32))
        ws.append(np.asarray(record[iw], np.float32) if iw is not None else np.zeros_like(xs[-1]))
    return {"times": np.asarray(merged.times, np.float64), "x": np.stack(xs), "y": np.stack(ys),
            "z": np.stack(zs), "w": np.stack(ws), "notes": dict(merged.notes), "nparcels": merged.track.nparcels}


def rising(track: dict[str, Any], rise_m: float = RISE_M) -> np.ndarray:
    """Which parcels climb more than ``rise_m`` above where they started, at some saved time."""

    z = track["z"]
    placed = np.abs(z) < -SENTINEL
    zmax = np.where(placed, z, -np.inf).max(axis=0)
    first = np.where(placed[0], z[0], np.inf)
    return (zmax - first) > rise_m


def best_time(track: dict[str, Any], chosen: np.ndarray) -> int:
    """The saved parcel time with the most chosen parcels rising faster than 2 m/s (the still's moment)."""

    counts = ((track["w"][:, chosen] > 2.0) & (np.abs(track["z"][:, chosen]) < -SENTINEL)).sum(axis=1)
    return int(np.argmax(counts)) if counts.size else 0


def _splat(acc: np.ndarray, wsum: np.ndarray, px: np.ndarray, py: np.ndarray, rgb: np.ndarray,
           weight: np.ndarray) -> None:
    h, w = wsum.shape
    ix = np.floor(px).astype(np.intp)
    iy = np.floor(py).astype(np.intp)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & (weight > 0)
    ix, iy, rgb, weight = ix[ok], iy[ok], rgb[ok], weight[ok]
    flat = iy * w + ix
    np.add.at(wsum.reshape(-1), flat, weight)
    for c in range(3):
        np.add.at(acc[..., c].reshape(-1), flat, rgb[:, c] * weight)


def _layer(size: tuple[int, int], points: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """An RGBA float layer from splatted points: colour is the weighted mean, opacity grows with the weight."""

    h, w = size
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    for px, py, rgb, weight in points:
        _splat(acc, wsum, px, py, rgb, weight)
    rgb = acc / np.maximum(wsum, 1e-6)[..., None]
    alpha = 1.0 - np.exp(-1.6 * wsum)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def _compose(base: np.ndarray, layer: np.ndarray, x0: int, y0: int) -> None:
    h, w = layer.shape[:2]
    region = base[y0:y0 + h, x0:x0 + w].astype(np.float32) / 255.0
    a = layer[..., 3:4]
    base[y0:y0 + h, x0:x0 + w] = S.to8(region * (1.0 - a) + layer[..., :3] * a)


def picture(track: dict[str, Any], n: int, chosen: np.ndarray, facts: Any, extent: tuple[float, float],
            origin: tuple[float, float], out: Path, *, size: tuple[int, int] = (P.W, P.H),
            top_m: float = HEIGHT_TOP, data: str = "") -> dict[str, Any]:
    """One picture at the ``n``-th saved parcel time: plan view left, west-east side view right."""

    W, H = size
    u = W / P.W
    times = track["times"]
    t = float(times[n])
    lx, ly = extent
    plan = P.fit_box(lx, ly, int(W * 0.27), int(190 * u), int(700 * u), int(700 * u))
    sx0, sw = int(W * 0.52), int(W * 0.43)
    sy0, sh = plan.y + int(plan.h * 0.18), int(plan.h * 0.64)
    cv = np.tile(np.array(S.rgb8("void"), np.uint8), (H, W, 1))
    cv[plan.y:plan.y + plan.h, plan.x:plan.x + plan.w] = S.rgb8("#060a12")
    cv[sy0:sy0 + sh, sx0:sx0 + sw] = S.rgb8("#060a12")
    placed = np.abs(track["z"]) < -SENTINEL
    ox, oy = origin
    steps = max(1, int(round(TRAIL_S / max(1.0, float(np.median(np.diff(times))) if times.size > 1 else 60.0))))
    first = max(0, n - steps)
    plan_pts: list[Any] = []
    side_pts: list[Any] = []
    idx = np.where(chosen)[0]
    sub = 6
    for m in range(first, n):
        ok = placed[m, idx] & placed[m + 1, idx]
        if not ok.any():
            continue
        k = idx[ok]
        age = (n - (m + 1)) / max(1, steps)
        for s in range(sub):
            f = (s + 0.5) / sub
            x = track["x"][m, k] * (1 - f) + track["x"][m + 1, k] * f - ox
            y = track["y"][m, k] * (1 - f) + track["y"][m + 1, k] * f - oy
            z = track["z"][m, k] * (1 - f) + track["z"][m + 1, k] * f
            rgb = S.colours("ember", z, -0.35 * top_m, top_m)
            weight = np.full(k.size, 0.55 * (1.0 - age) ** 1.5 + 0.05, np.float32)
            plan_pts.append(((x - plan.mx0) / plan.mpp, plan.h - (y - plan.my0) / plan.mpp, rgb, weight))
            side_pts.append((x / lx * sw, sh - z / top_m * sh, rgb, weight))
    here = placed[n, idx]
    k = idx[here]
    if k.size:
        rgb = S.colours("ember", track["z"][n, k], -0.35 * top_m, top_m)
        x, y, z = track["x"][n, k] - ox, track["y"][n, k] - oy, track["z"][n, k]
        plan_pts.append(((x - plan.mx0) / plan.mpp, plan.h - (y - plan.my0) / plan.mpp, rgb, np.full(k.size, 2.0)))
        side_pts.append((x / lx * sw, sh - z / top_m * sh, rgb, np.full(k.size, 2.0)))
    others = np.where(~chosen & placed[n])[0][::4]
    faint = np.array(S.hex_rgb("faint"), np.float32)
    if others.size:
        x, y, z = track["x"][n, others] - ox, track["y"][n, others] - oy, track["z"][n, others]
        grey = np.broadcast_to(faint, (others.size, 3))
        plan_pts.insert(0, ((x - plan.mx0) / plan.mpp, plan.h - (y - plan.my0) / plan.mpp, grey,
                            np.full(others.size, 0.35)))
        side_pts.insert(0, (x / lx * sw, sh - z / top_m * sh, grey, np.full(others.size, 0.35)))
    _compose(cv, _layer((plan.h, plan.w), plan_pts), plan.x, plan.y)
    _compose(cv, _layer((sh, sw), side_pts), sx0, sy0)
    cv = P.outline(cv, plan)
    cv = P.outline(cv, P.Box(sx0, sy0, sw, h=sh))
    ov = S.Overlay((W, H))
    ov.title("Air rising into the storm", data or S.data_line(facts.name, f"{int(chosen.sum()):,} of "
                                                              f"{track['nparcels']:,} parcels rise more than 3 km"))
    ov.clock((getattr(facts, "clock", None) or S.fmt_clock)(t))
    ov.footer(facts.footer(t=t, level="the parcels' own heights", wind="",
                           notes=S.data_line("the model's own air parcels, moved by its resolved wind",
                                             "trails: the last 10 min, straight between saved parcel times",
                                             "positions in the domain")))
    ov.mark()
    ov.text((plan.x, plan.y - ov.U(10)), "from above", "label", colour="text", anchor="ls")
    ov.text((sx0, sy0 - ov.U(10)), "from the south: west–east and height", "label", colour="text", anchor="ls")
    for zk in P.nice_step(top_m, 6) * np.arange(1, 7):
        if zk < top_m:
            yy = sy0 + sh - zk / top_m * sh
            ov.text((sx0 - ov.U(8), yy), S.fmt_height(float(zk)), "micro", colour="muted", anchor="rm", halo=False)
    live = int(((track["w"][n, idx] > 2.0) & here).sum()) if idx.size else 0
    heights = track["z"][n, k] if k.size else np.zeros(1)
    ov.readout((sx0, sy0 + sh + ov.U(40)), [("rising parcels now above 5 km", f"{int((heights > 5000).sum()):,}"),
                                            ("of them climbing now", f"{live:,}")])
    ov.ramp_bar("ember", (sx0 + sw - ov.U(380), sy0 + sh + ov.U(44), sx0 + sw, sy0 + sh + ov.U(44) + max(3, ov.U(7))),
                vmin=-0.35 * top_m, vmax=top_m,
                ticks=[(0.0, "0"), (top_m / 2, S.fmt_height(top_m / 2)), (top_m, S.fmt_height(top_m))],
                title="height of a rising parcel")
    P.scale_bar(ov, plan.x, plan.y + plan.h + ov.U(20), plan)
    P.save(ov, cv, out)
    return {"time_s": t, "rising": int(chosen.sum()), "nparcels": int(track["nparcels"]),
            "above_5km_now": int((heights > 5000).sum()), "climbing_now": live}


def origin_of(track: dict[str, Any], facts: Any, extent: tuple[float, float], nxy: tuple[int, int]
              ) -> tuple[float, float]:
    """The run coordinates of the domain's south-west corner, for the parcels' x and y.

    The run's ``iorigin`` says it (:meth:`.facts.RunFacts.origin_m`); when the run's settings could not be read
    (a folder of frames alone) it is taken from the parcels themselves: the corner that puts most of the first
    saved positions inside the domain."""

    lx, ly = extent
    x, y = track["x"][0], track["y"][0]
    placed = np.abs(x) < -SENTINEL

    def inside(o: tuple[float, float]) -> int:
        return int((placed & (x >= o[0]) & (x <= o[0] + lx) & (y >= o[1]) & (y <= o[1] + ly)).sum())

    stated = facts.origin_m(*nxy)
    centred = (-0.5 * lx, -0.5 * ly)
    return max((stated, (0.0, 0.0), centred), key=inside)


def still(layout: Any, facts: Any, extent: tuple[float, float], nxy: tuple[int, int], out: Path) -> dict[str, Any]:
    track = load(layout)
    chosen = rising(track)
    if not chosen.any():
        raise P.NotDrawn("No parcel in this run rose more than 3 km.")
    n = best_time(track, chosen)
    origin = origin_of(track, facts, extent, nxy)
    record = picture(track, n, chosen, facts, extent, origin, out)
    record["times"] = int(track["times"].size)
    return record


def film_frames(layout: Any, facts: Any, extent: tuple[float, float], nxy: tuple[int, int], folder: Path, *,
                size: tuple[int, int], should_stop: Callable[[], bool], progress: Callable[[int, int], None]
                ) -> list[Path]:
    track = load(layout)
    chosen = rising(track)
    if not chosen.any():
        raise P.NotDrawn("No parcel in this run rose more than 3 km.")
    origin = origin_of(track, facts, extent, nxy)
    paths = []
    total = int(track["times"].size)
    for n in range(total):
        if should_stop():
            break
        path = folder / f"{n:05d}.png"
        picture(track, n, chosen, facts, extent, origin, path, size=size)
        paths.append(path)
        progress(n + 1, total)
    return paths


__all__ = ["NoParcels", "best_time", "film_frames", "load", "picture", "rising", "still"]
