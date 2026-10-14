"""The storm seen from the side, and the water and ice above each place: the model's own hydrometeor species.

Two looks, from the mass mixing ratios the run saves (CM1's ``q3d``, whatever its scheme carries: cloud water
``qc``, rain ``qr``, cloud ice ``qi``, snow ``qs``, graupel or hail ``qg``; Kessler carries only the first two):

``side-view``    Storm seen from the side: a west-east elevation through the storm core, seen from the south.
                 Top panel: each species' mass summed through a slab (:data:`SLAB_M` deep, at least
                 :data:`SLAB_MIN_COLUMNS` grid rows), in kg per square metre of the picture (``rho q dy`` summed
                 over the slab's rows).  Bottom panel: the mixing ratios on the plane through the core, in g/kg.
                 The species are four lights that add where they mix (cloud water, rain, cloud ice and snow
                 together, graupel or hail), each on a fixed, stated logarithmic range.  Both panels share the
                 x axis and the vertical scale: true scale for a thunderstorm; a hurricane's heights are
                 stretched and the picture says by how much.
``water-paths``  Water and ice above each place: each species' column total (its water path, ``rho q dz`` summed
                 over every level, kg/m2; 1 kg/m2 is 1 mm of water) as one small map per species and one for all
                 of them together, on one fixed logarithmic range.

The storm core is the column of strongest w on the w level nearest 5 km (where the updraft slice cuts), or, for a
hurricane, the lowest pressure at the lowest model level (the storm's centre).  ``rho`` is the frame's air
density, else CM1's own (:func:`.satellite.density`).  The pictures are sampled bilinearly along x and linearly in
height between levels (the lowest level is drawn down to the ground); nothing is smoothed.

These are picture-tool sums of the model's saved fields, not CM1 diagnostics: CM1's cloud and ice water paths
(``lwp``, ``cwp``) are not used.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import products as P
from . import style as S
from .frame import FrameUnreadable

#: The species a scheme can carry, in drawing order: (row name in q3d, words; None: graupel or hail, from the
#: run's own record of its dense ice).
SPECIES: tuple[tuple[str, str | None], ...] = (("qc", "cloud water"), ("qr", "rain"), ("qi", "cloud ice"),
                                               ("qs", "snow"), ("qg", None))
#: The side view's four lights and the water paths' colours.  Cloud ice and snow share one light.
LIGHTS: dict[str, str] = {"water": "#dff6ff", "rain": "#4f7cff", "ice": "#c38cff", "dense": "#ffad42",
                          "all": "#e9eef7"}
GROUP = {"qc": "water", "qr": "rain", "qi": "ice", "qs": "ice", "qg": "dense"}
#: How bright each light is at the top of its range (rain's blue is darker than the others, so it gets more).
GAIN = {"water": 0.85, "rain": 1.7, "ice": 1.05, "dense": 1.15}
LIGHT_POWER = 2.0          # brightness = (position on the log range) ** this: faint values stay faint

SLAB_M = 20000.0           # m: the side view's slab depth (south to north)
SLAB_MIN_COLUMNS = 5       # grid rows, at least (a hurricane's 8 km grid: 40 km)
SIDE_TOP_M = 18000.0       # m: the side view's top, or the model top when lower
#: Each light's fixed logarithmic range, three tenfold steps: in the slab (kg per m2 of the picture) and on the cut
#: (g/kg).  Rain and the dense ice are heavier than cloud and ice, so their ranges sit higher; a value above a range
#: is drawn at full brightness.  A 1 km supercell's slab holds up to about 30 kg/m2 of cloud water and rain, 13 of
#: ice and snow and 60 of hail; a hurricane's 8 km run's up to 15 of rain and graupel (measured).
SIDE_RANGES = {"water": (0.05, 50.0), "rain": (0.1, 100.0), "ice": (0.05, 50.0), "dense": (0.2, 200.0)}
CUT_RANGES = {"water": (0.01, 10.0), "rain": (0.01, 10.0), "ice": (0.01, 10.0), "dense": (0.03, 30.0)}
PATH_RANGE = (0.01, 100.0)  # kg/m2 of ground, the column path, logarithmic, every species
#: A hurricane's window, in both looks (it is hundreds of km across, so its heights are stretched in the side view).
HURRICANE_WINDOW = 480_000.0
#: Less than this much condensate anywhere (kg/kg) and there is nothing to draw yet.
NOTHING = 1.0e-6


def _hex(rgb: np.ndarray) -> str:
    return "#" + "".join(f"{int(round(255 * min(1.0, max(0.0, float(v))))):02x}" for v in rgb)


def _species_ramp(key: str) -> S.Ramp:
    """Black, through the species' colour, to near white: one sequential ramp per light (L* rises)."""

    c = np.array(S.hex_rgb(LIGHTS[key]))
    void = np.array(S.hex_rgb("void"))
    stops = ((0.0, _hex(void)), (0.3, _hex(void * 0.5 + c * 0.22)), (0.65, _hex(c * 0.7)), (0.88, _hex(c)),
             (1.0, _hex(c * 0.45 + 0.55)))
    return S.Ramp(f"species_{key}", stops, 0.0, 1.0, "", "sequential", "a hydrometeor species, logarithmic")


for _key in LIGHTS:
    S.RAMPS.setdefault(f"species_{_key}", _species_ramp(_key))


def log_position(values: np.ndarray, rng: tuple[float, float]) -> np.ndarray:
    """Where a value sits on a logarithmic range, 0..1 (0 at or below the range's bottom, and for zero)."""

    lo, hi = rng
    v = np.maximum(np.asarray(values, dtype=np.float32), np.float32(lo * 1e-3))
    return np.clip(np.log10(v / np.float32(lo)) / np.float32(math.log10(hi / lo)), 0.0, 1.0).astype(np.float32)


#: Below this a maximum is printed as "under 0.001" (a trace, not a number to read digits from).
TRACE = 0.001


def fmt_amount(v: float) -> str:
    """0.05, 0.5, 5, 50: a stated range end or a maximum, plainly; a trace as "under 0.001"."""

    if v <= 0.0:
        return "0"
    if v < TRACE:
        return f"under {TRACE:g}"
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    digits = max(1, -int(math.floor(math.log10(v))))
    return f"{v:.{digits}f}"


# ----------------------------------------------------------------------------------------------- the frame's species
class Hydrometeors:
    """A frame's hydrometeors, read once and shared by both looks."""

    def __init__(self, L: "P.Looks") -> None:
        f = L.f
        try:
            names = f.species_names("q3d") if f.has("q3d") else ()
        except (FrameUnreadable, KeyError, ValueError):
            names = ()
        self.names = tuple(names)
        self.present = [n for n, _w in SPECIES if n in self.names]
        dense = (f.config.get("dense_ice") or {}).get("category")
        self.dense = dense if dense in ("hail", "graupel") else "graupel"
        self.words = {n: (w or self.dense) for n, w in SPECIES}
        self.L = L
        self._rho: np.ndarray | None = None
        self._paths: dict[str, np.ndarray] | None = None
        self.dz = np.diff(f.zf[:f.nz + 1]).astype(np.float32)

    @property
    def groups(self) -> list[str]:
        """The lights this run has, in order."""

        out = []
        for name in self.present:
            if GROUP[name] not in out:
                out.append(GROUP[name])
        return out

    def group_words(self, group: str) -> str:
        if group == "ice":
            have = [n for n in ("qi", "qs") if n in self.present]
            return " and ".join(self.words[n] for n in have)
        return self.words[[n for n in self.present if GROUP[n] == group][0]]

    def require(self) -> None:
        if not self.present:
            raise P.NotDrawn("This run saved no cloud water, rain or ice.")

    @property
    def rho(self) -> np.ndarray:
        if self._rho is None:
            from .satellite import density

            self._rho = density(self.L.f)
        return self._rho

    def q(self, name: str) -> np.ndarray:
        """The species' saved mixing ratios (kg/kg), a view on the frame's bytes."""

        f = self.L.f
        return f.view("q3d")[self.names.index(name)]

    def paths(self) -> dict[str, np.ndarray]:
        """Each species' column total, kg/m2: rho q dz summed over every level (negative q counts as zero)."""

        if self._paths is None:
            rho = self.rho
            out = {}
            for name in self.present:
                q = self.q(name)
                acc = np.zeros(rho.shape[1:], np.float32)
                for k in range(rho.shape[0]):
                    acc += rho[k] * np.maximum(np.asarray(q[k], np.float32), 0.0) * self.dz[k]
                out[name] = acc
            self._paths = out
        return self._paths

    def most(self) -> float:
        """The largest mixing ratio of any species anywhere (kg/kg), level by level."""

        best = 0.0
        for name in self.present:
            q = self.q(name)
            for k in range(q.shape[0]):
                best = max(best, float(np.max(q[k])))
        return best


def hydrometeors(L: "P.Looks") -> Hydrometeors:
    """The frame's :class:`Hydrometeors`, made once per frame (kept on the frame's shared fields)."""

    got = L.__dict__.get("_hydrometeors")
    if got is None:
        got = L.__dict__["_hydrometeors"] = Hydrometeors(L)
    return got


def core(L: "P.Looks") -> tuple[int, int, str]:
    """(i, j, what it is) of the storm core: a hurricane's lowest pressure at the lowest level, else the column of
    strongest w on the w level nearest 5 km."""

    f = L.f
    if L.hurricane and L.pressure0_hpa is not None:
        p = L.pressure0_hpa
        j, i = np.unravel_index(int(np.argmin(p)), p.shape)
        return int(i), int(j), "its lowest pressure at the lowest level"
    w5 = L.w5
    j, i = np.unravel_index(int(np.argmax(w5)), w5.shape)
    return int(i), int(j), f"the strongest w at {S.fmt_height(float(f.zf[L.k5]))}"


def slab_rows(L: "P.Looks", j: int) -> tuple[int, int]:
    """[j0, j1) of the side view's slab around row j, inside the domain."""

    f = L.f
    n = max(SLAB_MIN_COLUMNS, int(round(SLAB_M / f.dy)))
    n = min(n, f.ny)
    j0 = min(max(0, j - n // 2), f.ny - n)
    return j0, j0 + n


def damping_base(L: "P.Looks") -> float | None:
    """The height where CM1's damping layer near the model top (``irdamp``) starts, when the run uses it."""

    f = L.f
    cfg = f.config
    try:
        if int(cfg.get("irdamp") or 0) != 1:
            return None
        depth = float(cfg.get("damping_depth_m") or 0.0)
    except (TypeError, ValueError):
        return None
    if depth <= 0.0:
        return None
    return float(f.zf[-1]) - depth


# ----------------------------------------------------------------------------------------------- drawing helpers
def _vertical_sample(field: np.ndarray, zh: np.ndarray, zs: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """A (nz, nx) field sampled at heights ``zs`` (rows) and fractional columns ``ci``: linear in height between
    levels (the lowest level is held down to the ground), bilinear along x."""

    rows = np.interp(zs, zh, np.arange(len(zh), dtype=np.float64))
    rr = np.broadcast_to(rows[:, None], (len(zs), len(ci)))
    cc = np.broadcast_to(ci[None, :], (len(zs), len(ci)))
    return P.bilinear(field, rr, cc)


def lights(planes: dict[str, np.ndarray], ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    """The species as lights that add: each brightens with its position on its log range, in its colour."""

    shape = next(iter(planes.values())).shape
    hdr = np.zeros(shape + (3,), np.float32)
    for group, values in planes.items():
        b = log_position(values, ranges[group]) ** np.float32(LIGHT_POWER)
        hdr += (b * np.float32(GAIN[group]))[..., None] * np.array(S.hex_rgb(LIGHTS[group]), np.float32)
    return S.finish(hdr, k=1.6, gamma=1.1)


def _swatch(ov: S.Overlay, x: float, y: float, colour: str, w: int = 22, h: int = 10) -> None:
    from PIL import Image

    ov.draw_rgba(Image.new("RGBA", (ov.U(w), max(3, ov.U(h))), S.rgb8(colour, 1.0)), (int(x), int(y)))


def _dashed(cv: np.ndarray, x0: int, x1: int, y: int, colour: str, dash: int = 8, gap: int = 6) -> None:
    rgb = np.array(S.rgb8(colour), np.uint8)
    x = x0
    while x < x1:
        cv[y, x:min(x + dash, x1)] = rgb
        x += dash + gap


# ----------------------------------------------------------------------------------------------- side view
def product_side_view(L: "P.Looks", out: Path) -> dict[str, Any]:
    f = L.f
    H = hydrometeors(L)
    H.require()
    if H.most() < NOTHING:
        raise P.NotDrawn("No cloud, rain or ice in the air yet.")
    i, j, core_words = core(L)
    core_name = "the storm's centre" if L.hurricane else "the storm core"
    j0, j1 = slab_rows(L, j)
    top = min(SIDE_TOP_M, float(f.zf[-1]))
    lx = f.extent[0]
    x0, pw_max, ph, gap = 150, 1690, 270, 86
    y1 = 206
    y2 = y1 + ph + gap
    rho = H.rho
    # the slab's path and the cut's mixing ratio, each (nz, nx), per light
    slab: dict[str, np.ndarray] = {}
    cut: dict[str, np.ndarray] = {}
    cut_most: dict[str, float] = {}
    for name in H.present:
        q = np.maximum(np.asarray(H.q(name)[:, j0:j1, :], np.float32), 0.0)
        g = GROUP[name]
        s = (rho[:, j0:j1, :] * q).sum(axis=1) * np.float32(f.dy)
        c = np.maximum(np.asarray(H.q(name)[:, j, :], np.float32), 0.0) * np.float32(1000.0)
        slab[g] = slab[g] + s if g in slab else s
        cut[g] = cut[g] + c if g in cut else c
        cut_most[name] = float(c.max())
    # the window along x: at true vertical scale, as wide as the panel allows (a narrower domain gets a narrower
    # panel); a hurricane wider than that is shown 480 km wide with its heights stretched, and says so
    true_window = pw_max * top / ph
    if L.hurricane and lx > true_window:
        window = min(HURRICANE_WINDOW, lx)
        pw = pw_max
    else:
        window = min(true_window, lx)
        pw = int(round(window * ph / top)) if window < true_window else pw_max
    stretch = (ph / top) / (pw / window)
    core_x = (i + 0.5) * f.dx
    centre = core_x
    if not L.hurricane:
        # centred on the slab's water and ice (the anvil spreads downwind), keeping the core well inside
        total = sum(slab.values()).sum(axis=0)
        keep = total >= 0.01 * float(total.max()) if total.max() > 0 else np.zeros_like(total, bool)
        if keep.any():
            xs = (np.arange(f.nx) + 0.5) * f.dx
            centroid = float((total * keep * xs).sum() / (total * keep).sum())
            centre = min(max(centroid, core_x - 0.3 * window), core_x + 0.3 * window)
    xa = min(max(centre - 0.5 * window, 0.0), lx - window)
    px0 = x0 + (pw_max - pw) // 2
    xs_m = xa + (np.arange(pw) + 0.5) / pw * window
    ci = xs_m / f.dx - 0.5
    zs = (1.0 - (np.arange(ph) + 0.5) / ph) * top
    zh = f.zh
    img_side = lights({g: _vertical_sample(v, zh, zs, ci) for g, v in slab.items()}, SIDE_RANGES)
    img_cut = lights({g: _vertical_sample(v, zh, zs, ci) for g, v in cut.items()}, CUT_RANGES)
    cv = P.canvas()
    cv[y1:y1 + ph, px0:px0 + pw] = img_side
    cv[y2:y2 + ph, px0:px0 + pw] = img_cut
    damp = damping_base(L)
    for y in (y1, y2):
        cv = P.outline(cv, P.Box(px0, y, pw, 0.0, 0.0, 1.0, h=ph))
        if damp is not None and damp < top:
            _dashed(cv, px0, px0 + pw, int(round(y + ph - damp / top * ph)), "dim")
    # where the core is: a gold tick on the ground of both panels
    cx = px0 + (core_x - xa) / window * pw
    ox, oy = L.facts.origin_m(f.nx, f.ny)
    xk, yk = (ox + core_x) / 1000.0, (oy + (j + 0.5) * f.dy) / 1000.0
    ya, yb = (oy + j0 * f.dy) / 1000.0, (oy + j1 * f.dy) / 1000.0
    vertical_words = ("true vertical scale" if abs(stretch - 1.0) < 0.05
                      else f"heights drawn {stretch:.1f} × the horizontal scale")
    lights_words = [H.group_words(g) for g in H.groups]
    ov = P.overlay(L, "side-view",
                   S.data_line(L.facts.name, f"{S.fmt_grid(f.dx)} grid", f"west to east through {core_name}, "
                               "seen from the south", f"at x {S.fmt_num(xk, 1)} km, y {S.fmt_num(yk, 1)} km"),
                   level=f"ground to {S.fmt_height(top)}", wind="",
                   notes=S.data_line("the model's own mixing ratios", "logarithmic colours, fixed ranges",
                                     "linear between levels", vertical_words, L.frame_words()))
    deep = (j1 - j0) * f.dy
    side_top = _cloud_top(slab, zh, SIDE_RANGES)

    # (height, left label, right label): each (words, type role, colour), drawn once their alignment is known
    rows: list[tuple[float, tuple[str, str, str], tuple[str, str, str] | None]] = []
    slab_words = (f"seen from the south: each species summed through the {S.fmt_km(deep)} slab from y "
                  f"{S.fmt_num(ya, 1)} to {S.fmt_num(yb, 1)} km")
    top_words = (f"cloud top in the slab (where a colour begins) {S.fmt_height(side_top)}" if side_top is not None
                 else None)
    most_words = "most on the cut: " + ", ".join(f"{H.words[n]} {fmt_amount(cut_most[n])}" for n in H.present)
    rows.append((y1 - ov.U(12), (slab_words, "label", "text"), (top_words, "label", "data") if top_words else None))
    rows.append((y2 - ov.U(12), (f"cut through {core_name} at y {S.fmt_num(yk, 1)} km: mixing ratio", "label", "text"),
                 (most_words + " g/kg", "micro", "data")))
    for y in (y1, y2):
        for zk in _height_ticks(top):
            yy = y + ph - zk / top * ph
            if zk > 0:
                ov.text((px0 - ov.U(14), yy), S.fmt_height(zk), "micro", colour="muted", anchor="rm", halo=False)
        ov.text((px0 - ov.U(14), y + ph), "W", "micro", colour="muted", anchor="rm", halo=False)
        ov.text((px0 + pw + ov.U(12), y + ph), "E", "micro", colour="muted", anchor="lm", halo=False)
        ov.text((cx, y + ph + ov.U(2)), "▲", "micro", colour="gold", anchor="ma", halo=False)
    if damp is not None and damp < top:
        ov.text((px0 + pw - ov.U(6), y1 + ph - damp / top * ph - ov.U(5)),
                f"the model's damping layer starts at {S.fmt_height(damp)}", "micro", colour="dim", anchor="rs",
                halo=False)
    # the x labels: up to 10, fewer on a narrow panel so they do not touch
    wide = ov.measure(S.fmt_num(-(abs(ox) + lx) / 1000.0, 1), "micro")[0] + ov.U(24)
    step = P.nice_step(window, max(2, min(10, int(pw // wide))))
    first = math.ceil((ox + xa) / step) * step
    for d in np.arange(first, ox + xa + window + 1e-6, step):
        xx = px0 + (d - ox - xa) / window * pw
        ov.text((xx, y2 + ph + ov.U(30)), S.fmt_num(d / 1000.0, 0 if step >= 1000 else 1), "micro",
                colour="muted", anchor="ms", halo=False)
    rows.append((y2 + ph + ov.U(52), (f"▲ {core_name}: {core_words}", "micro", "gold"),
                 (f"x, km; {vertical_words}", "micro", "muted")))
    # the labels over the panels' edges, or, when one pair does not fit side by side over a narrow panel, all of
    # them over the picture's edges
    fits = all(ov.measure(left[0], left[1])[0] + (ov.measure(right[0], right[1])[0] if right else 0) + ov.U(40) <= pw
               for _y, left, right in rows)
    a, b = (px0, px0 + pw) if fits else (x0, x0 + pw_max)
    for y, left, right in rows:
        ov.text((a, y), left[0], left[1], colour=left[2], anchor="ls")
        if right:
            ov.text((b, y), right[0], right[1], colour=right[2], anchor="rs")
    # the key: one swatch per light with its two ranges, then how to read them
    ky = y2 + ph + ov.U(86)
    x = x0
    for g in H.groups:
        _swatch(ov, x, ky - ov.U(11), LIGHTS[g])
        a = ov.text((x + ov.U(30), ky), H.group_words(g), "label", colour="data", anchor="ls")
        (s0, s1), (c0, c1) = SIDE_RANGES[g], CUT_RANGES[g]
        b = ov.text((x + ov.U(30), ky + ov.U(20)), f"{fmt_amount(s0)}–{fmt_amount(s1)} kg/m²  ·  "
                    f"{fmt_amount(c0)}–{fmt_amount(c1)} g/kg", "micro", colour="muted", anchor="ls", halo=False)
        x = max(a[2], b[2]) + ov.U(40)
    ov.text((x0 + pw_max, ky), "upper: kg of water or ice per m² of the picture; lower: g per kg of air",
            "micro", colour="muted", anchor="rs", halo=False)
    ov.text((x0 + pw_max, ky + ov.U(20)), "each on its own log scale; the lights add where species mix", "micro",
            colour="muted", anchor="rs", halo=False)
    P.save(ov, cv, out)
    return {"core_ij": [i + 1, j + 1], "core_km": [xk, yk], "core": core_words, "slab_km": [ya, yb],
            "slab_rows": [j0 + 1, j1], "window_km": [(ox + xa) / 1000.0, (ox + xa + window) / 1000.0],
            "top_m": top, "vertical_stretch": round(stretch, 3), "species": list(H.present),
            "lights": lights_words, "slab_path_max_kg_m2": {g: float(v.max()) for g, v in slab.items()},
            "cut_max_g_kg": cut_most, "cloud_top_m": side_top, "damping_base_m": damp,
            "ranges": {"slab_kg_m2": {g: list(SIDE_RANGES[g]) for g in H.groups},
                       "cut_g_kg": {g: list(CUT_RANGES[g]) for g in H.groups}}}


def _height_ticks(top: float) -> list[float]:
    step = P.nice_step(top, 4)
    return [float(v) for v in np.arange(0.0, top + 1e-6, step)]


def _cloud_top(slab: dict[str, np.ndarray], zh: np.ndarray, ranges: dict[str, tuple[float, float]]) -> float | None:
    """The highest level where the slab holds any species at the bottom of its range or more (where it shows)."""

    seen = np.zeros(len(zh), bool)
    for g, v in slab.items():
        seen |= (v >= ranges[g][0]).any(axis=1)
    rows = np.nonzero(seen)[0]
    return float(zh[rows.max()]) if rows.size else None


# ----------------------------------------------------------------------------------------------- water paths
def _path_region(L: "P.Looks", i: int, j: int) -> tuple[float, float, float, float]:
    """(x0, y0, width, height) in m of the maps: the whole domain, or a hurricane's window around its centre."""

    lx, ly = L.f.extent
    if not L.hurricane:
        return 0.0, 0.0, lx, ly
    wx, wy = min(HURRICANE_WINDOW, lx), min(HURRICANE_WINDOW, ly)
    cx, cy = (i + 0.5) * L.f.dx, (j + 0.5) * L.f.dy
    return min(max(cx - wx / 2, 0.0), lx - wx), min(max(cy - wy / 2, 0.0), ly - wy), wx, wy


def product_water_paths(L: "P.Looks", out: Path) -> dict[str, Any]:
    f = L.f
    H = hydrometeors(L)
    H.require()
    paths = H.paths()
    total = sum(paths.values())
    if float(total.max()) < PATH_RANGE[0]:
        raise P.NotDrawn("No cloud, rain or ice in the air yet.")
    i, j, core_words = core(L)
    core_name = "the storm's centre" if L.hurricane else "the storm core"
    j0, j1 = slab_rows(L, j)
    rx0, ry0, rw, rh = _path_region(L, i, j)
    # the columns the maps show (a hurricane's window), so each map's "most" is the most it shows
    shown = (slice(int(math.floor(ry0 / f.dy + 1e-6)), int(math.ceil((ry0 + rh) / f.dy - 1e-6))),
             slice(int(math.floor(rx0 / f.dx + 1e-6)), int(math.ceil((rx0 + rw) / f.dx - 1e-6))))
    panels = [(n, H.words[n], f"species_{GROUP[n]}", paths[n]) for n in H.present]
    panels.append(("all", "all of them", "species_all", total))
    n = len(panels)
    cols = 3 if n > 4 else n
    rows = int(math.ceil(n / cols))
    gx, gy = 44, 66
    area_w, area_h = 1760, 664
    cell_w = (area_w - (cols - 1) * gx) / cols
    cell_h = (area_h - (rows - 1) * gy) / rows
    mpp = max(rw / cell_w, rh / cell_h)
    bw, bh = int(round(rw / mpp)), int(round(rh / mpp))
    left = (P.W - cols * bw - (cols - 1) * gx) // 2
    top = 214 + (area_h - rows * bh - (rows - 1) * gy) // 2
    cv = P.canvas()
    boxes = []
    maxima: dict[str, float] = {}
    for idx, (name, words, ramp, path) in enumerate(panels):
        r, c = divmod(idx, cols)
        box = P.Box(left + c * (bw + gx), top + r * (bh + gy), bw, rx0, ry0, bw * mpp, h=bh)
        img = S.to8(S.RAMPS[ramp].colours(log_position(P.sample(path, box, f.dx, f.dy), PATH_RANGE), 0.0, 1.0))
        P.paste(cv, img, box)
        boxes.append(box)
        maxima[name] = float(path[shown].max())
    for box in boxes:
        cv = P.outline(cv, box)
    # the side view's slab, on the last panel (all of them)
    last = boxes[-1]
    ya, yb = j0 * f.dy, j1 * f.dy
    for ym in (ya, yb):
        _x, yy = last.to_px(rx0, ym)
        if last.y <= yy < last.y + last.h:
            _dashed(cv, last.x, last.x + last.w, int(yy), "muted", dash=5, gap=5)
    ox, oy = L.facts.origin_m(f.nx, f.ny)
    where = (f"the whole {rw / 1000:g} × {rh / 1000:g} km domain" if not L.hurricane
             else f"the {rw / 1000:g} × {rh / 1000:g} km around the storm's centre")
    data = S.data_line(L.facts.name, f"{S.fmt_grid(f.dx)} grid", where, "each species summed over the whole column")
    ov = P.overlay(L, "water-paths", data, level="the whole column", wind="",
                   notes=S.data_line("path: ρ q dz summed over every level", "1 kg m⁻² is 1 mm of water",
                                     f"{fmt_amount(PATH_RANGE[0])}–{fmt_amount(PATH_RANGE[1])} kg m⁻² fixed, "
                                     "logarithmic", "not satellite data", L.frame_words()))
    for box, (name, words, _ramp, _path) in zip(boxes, panels):
        ov.text((box.x, box.y - ov.U(10)), words, "label", colour="text", anchor="ls")
        ov.text((box.x + box.w, box.y - ov.U(10)), f"most {fmt_amount(maxima[name])} kg m⁻²", "micro",
                colour="data", anchor="rs")
    cxp, cyp = last.to_px((i + 0.5) * f.dx, (j + 0.5) * f.dy)
    if last.x <= cxp <= last.x + last.w and last.y <= cyp <= last.y + last.h:
        ov.ring((cxp, cyp), 9.0)
    ms = ov.px("micro")
    by = max(b.y + b.h for b in boxes)
    ov.text((last.x + last.w, by + ov.U(14) + ms),
            f"dashed lines: the side view's slab; gold ring: {core_name}", "micro", colour="muted", anchor="rs",
            halo=False)
    bar_y = by + ov.U(36)
    bx0 = boxes[0].x
    ticks = [(float(log_position(v, PATH_RANGE)), lab) for v, lab in
             ((0.01, "0.01"), (0.1, "0.1"), (1.0, "1"), (10.0, "10"), (100.0, "100"))]
    bar_w = min(ov.U(380), boxes[0].w - ov.U(6))
    ov.ramp_bar("species_all", (bx0, bar_y, bx0 + bar_w, bar_y + max(3, ov.U(7))), vmin=0.0, vmax=1.0,
                ticks=ticks, title="kg m⁻², each map in its species' colour")
    mid = boxes[min(len(boxes) - 1, cols // 2)] if cols > 1 else boxes[0]
    P.scale_bar(ov, mid.x + (ov.U(10) if cols > 1 else bar_w + ov.U(30)), by + ov.U(22), mid)
    P.save(ov, cv, out)
    return {"path_max_kg_m2": maxima, "species": list(H.present), "dense_ice": H.dense,
            "region_km": [(ox + rx0) / 1000.0, (oy + ry0) / 1000.0, rw / 1000.0, rh / 1000.0],
            "core_ij": [i + 1, j + 1], "core": core_words, "slab_rows": [j0 + 1, j1],
            "range_kg_m2": list(PATH_RANGE), "scale": "logarithmic"}


__all__ = ["CUT_RANGES", "GROUP", "Hydrometeors", "LIGHTS", "PATH_RANGE", "SIDE_RANGES", "SLAB_M", "core",
           "damping_base", "hydrometeors", "log_position", "product_side_view", "product_water_paths",
           "slab_rows"]
