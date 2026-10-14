"""The viz 2.0 look for quick-looks: colour tokens, ramps, type, the frame grammar and the honesty lint.

This is the part of the viz 2.0 design system (``viz20_style``) the quick-looks use, generalised for any run: no
run, storm, place or date names, no output folders, no film or encoding helpers.  The rules it keeps:

* one text engine (Pillow) draws every title, clock, footer, label and mark, so a matplotlib chart and a numpy
  picture are typographically identical;
* the frame grammar: title and data line top-left, the gold T+ clock top-right, the honesty footer bottom-left,
  the GM1 mark ("GM1 simulated") bottom-right;
* ``lint``: every picture says "simulated", every horizontal wind speed says which frame it is measured in, and
  "tornado" never appears (the GUI's picture rule is stricter than viz 2.0's: no tornado wording at all).

Fonts: DejaVu Sans.  The book, bold and oblique faces come with matplotlib; the condensed and extra-light faces
the grammar uses are bundled in ``fonts/`` (Bitstream Vera licence, ``fonts/LICENSE-DejaVu.txt``), so a picture
looks the same on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import re
from typing import Iterable, Sequence
import warnings

import numpy as np

# =============================================================================
# colour tokens
# =============================================================================

PALETTE: dict[str, str] = {
    "void": "#030407",      # true-black ground
    "ground": "#05070d",
    "panel": "#0a0e18",
    "rule": "#1d2436",      # hairlines
    "faint": "#465066",
    "title": "#ffffff",
    "text": "#e6edf7",
    "data": "#c7cfdb",      # the data line, in-scene labels
    "muted": "#8e97a8",     # footer, captions, ramp ticks
    "dim": "#5b6477",
    "gold": "#ffd166",      # the one accent: T+ clock, rings
    "cold": "#4cc3ff",
    "warm": "#ffb347",
    "silk": "#6a84c8",      # flow streak tint
    "cloud": "#c9d0de",
    "core": "#fff4dc",
}

MINUS = "−"
PRIME = "′"
DOT = "·"
SEP = f"  {DOT}  "
PER_S = "s⁻¹"
SIMULATED = "simulated"
#: The smallest size (of the data size) a data line too long for one row is set at (:meth:`Overlay.title`).
DATA_FLOOR = 0.75


def hex_rgb(colour: str) -> tuple[float, float, float]:
    """``'#rrggbb'`` or a palette token -> floats in 0..1."""

    c = (colour if colour.startswith("#") else PALETTE[colour]).lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb8(colour: str, alpha: float | None = None) -> tuple[int, ...]:
    r, g, b = (int(round(255 * v)) for v in hex_rgb(colour))
    return (r, g, b) if alpha is None else (r, g, b, int(round(255 * max(0.0, min(1.0, alpha)))))


def token(name: str) -> str:
    return name if name.startswith("#") else PALETTE[name]


# =============================================================================
# ramps
# =============================================================================


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lightness(rgb: np.ndarray) -> np.ndarray:
    """CIELAB L* (0-100) of sRGB colours in 0..1."""

    lin = _srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    f = np.where(y > 216 / 24389, np.cbrt(y), (24389 / 27 * y + 16) / 116)
    return 116 * f - 16


@dataclass(frozen=True)
class Ramp:
    """A colour ramp: ``stops`` are ``(position 0..1, '#rrggbb')`` over the data range ``vmin..vmax``."""

    name: str
    stops: tuple[tuple[float, str], ...]
    vmin: float
    vmax: float
    unit: str
    kind: str          # sequential or diverging (dark at ``centre``)
    use: str
    centre: float | None = None

    def _norm(self, values, vmin=None, vmax=None) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        lo = self.vmin if vmin is None else vmin
        hi = self.vmax if vmax is None else vmax
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    def colours(self, values, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
        """float32 RGB in 0..1, shape ``values.shape + (3,)``; NaN takes the ramp's zero colour."""

        t = self._norm(values, vmin, vmax)
        pos = np.array([p for p, _ in self.stops])
        cols = np.array([hex_rgb(c) for _, c in self.stops])
        t = np.nan_to_num(t, nan=0.5 if self.kind == "diverging" else 0.0)
        return np.stack([np.interp(t, pos, cols[:, ch]) for ch in range(3)], axis=-1).astype(np.float32)

    def at(self, t: float) -> str:
        pos = np.array([p for p, _ in self.stops])
        cols = np.array([hex_rgb(c) for _, c in self.stops])
        rgb = [np.interp(t, pos, cols[:, ch]) for ch in range(3)]
        return "#" + "".join(f"{int(round(255 * x)):02x}" for x in rgb)

    def lut(self, n: int = 256) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n)
        pos = np.array([p for p, _ in self.stops])
        cols = np.array([hex_rgb(c) for _, c in self.stops])
        rgb = np.stack([np.interp(t, pos, cols[:, ch]) for ch in range(3)], axis=-1)
        return np.round(rgb * 255).astype(np.uint8)

    def cmap(self):
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(f"gm1_{self.name}", list(self.stops), N=512)

    def check(self, tol: float = 0.75) -> list[str]:
        """Perceptual-order problems (empty = passes): L* rises along a sequential ramp and away from a
        diverging ramp's dark centre."""

        L = lightness(self.lut(129).astype(np.float64) / 255.0)
        if self.kind == "sequential":
            return [f"{self.name}: L* not monotonic"] if np.any(np.diff(L) < -tol) else []
        c = float(self._norm(self.centre if self.centre is not None else 0.0))
        k = int(round(c * (len(L) - 1)))
        left, right = L[:k + 1][::-1], L[k:]
        if np.any(np.diff(left) < -tol) or np.any(np.diff(right) < -tol):
            return [f"{self.name}: L* does not rise away from the dark centre"]
        return []


def value_stops(pairs: Sequence[tuple[float, str]]) -> tuple[tuple[float, str], ...]:
    lo, hi = pairs[0][0], pairs[-1][0]
    return tuple(((v - lo) / (hi - lo), c) for v, c in pairs)


RAMPS: dict[str, Ramp] = {
    # viz 2.0's signature vertical-vorticity LUT: lilac-teal anticyclonic, black at zero, crimson-amber-cream cyclonic.
    "zeta": Ramp("zeta",
                 ((0.00, "#b9a8ff"), (0.14, "#5a8ee8"), (0.30, "#138a9a"), (0.46, "#0b1d24"), (0.50, "#07080c"),
                  (0.54, "#2a0f1d"), (0.68, "#a3304f"), (0.84, "#e39a2d"), (1.00, "#fff4dc")),
                 -1.0, 1.0, PER_S, "diverging", "vertical vorticity and updraft helicity", centre=0.0),
    "ember": Ramp("ember",
                  value_stops(((0, "#0b0710"), (1, "#2a1236"), (2, "#5b1d57"), (3, "#9b2a5c"), (4, "#d1523f"),
                               (5, "#f09a3a"), (6, "#f7d27a"), (7, "#fff6df"))),
                  0.0, 1.0, "", "sequential", "magnitude: swath build-up, mixing ratios"),
    "ice": Ramp("ice",
                value_stops(((0, "#03050a"), (1, "#0b1d3a"), (2, "#1d4f8f"), (3, "#3b8fd9"), (4, "#8fd3ff"),
                             (5, "#eaf8ff"))),
                0.0, 1.0, "", "sequential", "cold magnitude: wind swath, rain"),
    # vertical velocity, symmetric: blue down, amber up, black at 0 (viz 2.0 'w' colours, stated range)
    "w_sym": Ramp("w_sym",
                  value_stops(((-50.0, "#cfe1ff"), (-35.0, "#9cc3ff"), (-20.0, "#3b82f6"), (-7.0, "#10244a"),
                               (0.0, "#07080c"), (7.0, "#3a1d06"), (20.0, "#b45309"), (35.0, "#f59e0b"),
                               (50.0, "#fcd34d"))),
                  -50.0, 50.0, "m/s", "diverging", "vertical velocity", centre=0.0),
    # theta', symmetric: blue-violet cold pool, black at 0, amber warm
    "theta_sym": Ramp("theta_sym",
                      value_stops(((-8.0, "#8a78ff"), (-6.5, "#6a55f0"), (-5.0, "#4d38c8"), (-3.5, "#35259a"),
                                   (-2.0, "#1e1664"), (-0.8, "#0d0a28"), (0.0, "#050609"), (0.8, "#1c1106"),
                                   (2.0, "#6e4212"), (5.0, "#b86b1c"), (8.0, "#ffb347"))),
                      -8.0, 8.0, "K", "diverging", "potential temperature perturbation", centre=0.0),
    # a generic signed field (u, v, pressure perturbation): ice-blue negative, black 0, amber positive
    "signed": Ramp("signed",
                   ((0.0, "#e6f6ff"), (0.2, "#4cc3ff"), (0.4, "#0f3552"), (0.5, "#07080c"), (0.6, "#4a2406"),
                    (0.8, "#ffa53a"), (1.0, "#fff1d6")),
                   -1.0, 1.0, "", "diverging", "any signed field, symmetric range", centre=0.0),
    "rain_cm": Ramp("rain_cm",
                    value_stops(((0.0, "#03050a"), (1.0, "#0b1d3a"), (2.0, "#1d4f8f"), (4.0, "#3b8fd9"),
                                 (6.0, "#8fd3ff"), (8.0, "#eaf8ff"))),
                    0.0, 8.0, "cm", "sequential", "accumulated rain"),
    # A hurricane's looks (quicklook.products): each rises in lightness, so the strongest part is the brightest.
    # Wind, 0 to 80 m/s: blue from 10 m/s, violet through gale force, red at hurricane force (33 m/s), then
    # orange, yellow and white for the eyewall.
    "hurricane_wind": Ramp("hurricane_wind",
                           value_stops(((0.0, "#04060c"), (5.0, "#0a1230"), (10.0, "#12296b"), (17.0, "#1f4fb8"),
                                        (25.0, "#6a4fd8"), (33.0, "#d0408a"), (42.0, "#f0603a"),
                                        (52.0, "#f79a2e"), (62.0, "#fbd24a"), (72.0, "#fff1a8"),
                                        (80.0, "#ffffff"))),
                           0.0, 80.0, "m/s", "sequential", "a hurricane's wind at the lowest level"),
    # Rain over days, placed logarithmically (1 mm to 400 cm): blue, teal, green, yellow, cream.
    "hurricane_rain": Ramp("hurricane_rain",
                           ((0.00, "#0b1740"), (0.20, "#14389a"), (0.40, "#1c78c8"), (0.58, "#22aab0"),
                            (0.74, "#5fcf72"), (0.88, "#e4e05a"), (1.00, "#fffbe0")),
                           0.0, 1.0, "", "sequential", "a hurricane's accumulated rain, logarithmic"),
    # Pressure below the area's edge, placed by its square root: night violet to lavender to white.
    "hurricane_pressure": Ramp("hurricane_pressure",
                               ((0.00, "#06060e"), (0.15, "#141138"), (0.32, "#2c2382"), (0.50, "#5a45c8"),
                                (0.68, "#8f7df0"), (0.84, "#c9bfff"), (1.00, "#fbf8ff")),
                               0.0, 1.0, "", "sequential", "a hurricane's pressure drop, square root"),
    # The ring-mean swirling wind by radius and height, -20 to 80 m/s: above zero the wind's own colours
    # (hurricane_wind, so a speed is the same colour in both looks), below zero green for air turning the other way.
    "hurricane_swirl": Ramp("hurricane_swirl",
                            value_stops(((-20.0, "#d4f5c8"), (-12.0, "#6cc28a"), (-6.0, "#1f6b4a"),
                                         (-2.0, "#0b2419"), (0.0, "#04060c"), (5.0, "#0a1230"), (10.0, "#12296b"),
                                         (17.0, "#1f4fb8"), (25.0, "#6a4fd8"), (33.0, "#d0408a"),
                                         (42.0, "#f0603a"), (52.0, "#f79a2e"), (62.0, "#fbd24a"),
                                         (72.0, "#fff1a8"), (80.0, "#ffffff"))),
                            -20.0, 80.0, "m/s", "diverging", "a hurricane's ring-mean swirling wind", centre=0.0),
    # The warm core, -10 to +25 K: violet colder, black at 0, amber to cream warmer; the colours change faster near
    # zero (stops closer together), so a warm core of a few kelvin shows beside one of twenty.
    "hurricane_warm": Ramp("hurricane_warm",
                           value_stops(((-10.0, "#a99bff"), (-6.0, "#6a55f0"), (-3.0, "#35259a"), (-1.0, "#120d36"),
                                        (0.0, "#050609"), (1.0, "#2a1306"), (3.0, "#6e3410"), (6.0, "#b4561a"),
                                        (10.0, "#e8862a"), (14.0, "#f7b84a"), (19.0, "#fde3a0"),
                                        (25.0, "#fff8e6"))),
                           -10.0, 25.0, "K", "diverging", "a hurricane's warm core", centre=0.0),
    # Radial velocity on a simulated radar's scan (quicklook.radar), m/s, -70 to +70: the NWS base-velocity colours
    # (green toward the radar, red away), dark grey at zero so a still gate differs from no echo, lighter with speed.
    "radial_velocity": Ramp("radial_velocity",
                            value_stops(((-70.0, "#f0fff8"), (-50.0, "#9dffe0"), (-40.0, "#00ffb4"),
                                         (-32.0, "#00ff28"), (-25.0, "#00e600"), (-18.0, "#00c800"),
                                         (-13.0, "#00aa00"), (-10.0, "#009400"), (-5.0, "#006e00"),
                                         (-1.5, "#2f4a2f"), (0.0, "#2a2a2a"), (1.5, "#4c2c2c"), (5.0, "#7c0000"),
                                         (10.0, "#a60000"), (13.0, "#bf0000"), (18.0, "#dc0000"),
                                         (25.0, "#fa0000"), (32.0, "#ff4a4a"), (40.0, "#ff8cc6"),
                                         (50.0, "#ffc8e8"), (70.0, "#fff4fa"))),
                            -70.0, 70.0, "m/s", "diverging", "radial velocity, toward and away from a radar",
                            centre=0.0),
}

# NWS base reflectivity, 5 dBZ bins from 5 dBZ; below 5 dBZ nothing is drawn.
NWS_REFLECTIVITY: tuple[tuple[float, str], ...] = (
    (5, "#04e9e7"), (10, "#019ff4"), (15, "#0300f4"), (20, "#02fd02"), (25, "#01c501"), (30, "#008e00"),
    (35, "#fdf802"), (40, "#e5bc00"), (45, "#fd9500"), (50, "#fd0000"), (55, "#d40000"), (60, "#bc0000"),
    (65, "#f800fd"), (70, "#9854c6"), (75, "#fdfdfd"))


def dbz_colours(dbz) -> np.ndarray:
    """RGBA float32 in the NWS table; alpha 0 below 5 dBZ and for NaN."""

    v = np.nan_to_num(np.asarray(dbz, dtype=np.float64), nan=-99.0)
    cols = np.array([hex_rgb(c) for _, c in NWS_REFLECTIVITY], dtype=np.float32)
    idx = np.clip(np.floor((v - 5.0) / 5.0).astype(int), 0, len(cols) - 1)
    out = np.empty(v.shape + (4,), np.float32)
    out[..., :3] = cols[idx]
    out[..., 3] = (v >= 5.0).astype(np.float32)
    return out


def colours(name: str, values, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    return RAMPS[name].colours(values, vmin, vmax)


def tonemap(hdr: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Hue-preserving tone map ``RGB * (1 - exp(-k * max)) / max`` (keeps amber amber)."""

    m = np.maximum(hdr.max(axis=-1, keepdims=True), 1e-9)
    return hdr / m * (1.0 - np.exp(-k * m))


def finish(hdr: np.ndarray, *, k: float = 1.35, gamma: float = 1.15) -> np.ndarray:
    """Tone map and display gamma to uint8.  No bloom, glow or vignette: a quick-look is a measurement."""

    x = np.clip(tonemap(hdr.astype(np.float32), k), 0.0, 1.0) ** (1.0 / gamma)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def to8(rgb01: np.ndarray) -> np.ndarray:
    return (np.clip(rgb01, 0, 1) * 255 + 0.5).astype(np.uint8)


# =============================================================================
# type
# =============================================================================

FONT_FILES: dict[str, str] = {
    "bold": "DejaVuSans-Bold.ttf",
    "book": "DejaVuSans.ttf",
    "light": "DejaVuSans-ExtraLight.ttf",
    "condensed": "DejaVuSansCondensed.ttf",
    "condensed_bold": "DejaVuSansCondensed-Bold.ttf",
}
_FONT_FALLBACK = {"light": "book", "condensed": "book", "condensed_bold": "bold"}
FONT_DIR = Path(__file__).resolve().parent / "fonts"


def _font_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("GM1_VIZ_FONT_DIR"):
        dirs.append(Path(os.environ["GM1_VIZ_FONT_DIR"]))
    dirs.append(FONT_DIR)
    try:
        import matplotlib

        dirs.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except Exception:  # pragma: no cover - matplotlib comes with gpuwm
        pass
    dirs += [Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu"), Path("/usr/share/fonts/TTF")]
    return dirs


@lru_cache(maxsize=None)
def font_path(style: str = "book") -> str:
    seen = set()
    s = style
    while s and s not in seen:
        seen.add(s)
        for d in _font_dirs():
            p = d / FONT_FILES[s]
            if p.is_file():
                return str(p)
        s = _FONT_FALLBACK.get(s, "book" if s != "book" else "")
    raise FileNotFoundError("no DejaVu Sans font found (matplotlib ships one); set GM1_VIZ_FONT_DIR")


@lru_cache(maxsize=None)
def _codepoints(path: str) -> frozenset[int]:
    try:
        from matplotlib.ft2font import FT2Font

        face = FT2Font(path)
        return frozenset(cp for cp in range(0x20, 0x2400) if face.get_char_index(cp))
    except Exception:  # pragma: no cover - matplotlib comes with gpuwm
        return frozenset()


def has_glyphs(style: str, text: str) -> bool:
    points = _codepoints(font_path(style))
    return not points or all(ord(ch) in points for ch in text if not ch.isspace() and ord(ch) < 0x2400)


@lru_cache(maxsize=256)
def pil_font(style: str, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(font_path(style), max(1, int(round(px))))


# (style, px at a 1080 px short side, colour token, uppercase?)
TYPE_ROLES: dict[str, tuple[str, float, str, bool]] = {
    "title": ("bold", 46, "title", True),
    "clock": ("book", 40, "gold", False),
    "hero_number": ("light", 64, "title", False),
    "data": ("book", 21, "data", False),
    "label": ("book", 19, "data", False),
    "footer": ("condensed", 16, "muted", False),
    "mark": ("bold", 16, "text", False),
    "micro": ("condensed", 14, "muted", False),
}


def unit(width: int, height: int) -> float:
    """Layout unit: 1 px at a 1080 px short side."""

    return min(width, height) / 1080.0


def apply_matplotlib() -> None:
    """rcParams for a matplotlib chart inside a quick-look (dark, DejaVu, no chrome)."""

    import matplotlib as mpl
    from matplotlib import font_manager

    for s in ("book", "bold", "light", "condensed"):
        try:
            font_manager.fontManager.addfont(font_path(s))
        except Exception:
            pass
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "figure.facecolor": PALETTE["void"], "savefig.facecolor": PALETTE["void"],
        "axes.facecolor": PALETTE["void"], "axes.edgecolor": PALETTE["rule"], "axes.labelcolor": PALETTE["muted"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"], "text.color": PALETTE["data"],
        "legend.frameon": False, "lines.solid_capstyle": "round",
        "axes.unicode_minus": True, "mathtext.default": "regular",
    })


def mpl_fontprops(role: str, fig=None, *, px: float | None = None):
    from matplotlib.font_manager import FontProperties

    style, base, _, _ = TYPE_ROLES[role]
    if px is None:
        if fig is None:
            px = base
        else:
            w, h = fig.get_size_inches() * fig.dpi
            px = base * unit(int(w), int(h))
    d = fig.dpi if fig is not None else 100.0
    return FontProperties(fname=font_path(style), size=px * 72.0 / d)


# =============================================================================
# formatters and the honesty lint
# =============================================================================


def fmt_num(x: float, nd: int = 0, *, sign: bool = False) -> str:
    """A number with a true minus sign."""

    s = f"{abs(x):.{nd}f}"
    if x < 0 and float(s) != 0.0:
        return MINUS + s
    return ("+" + s) if sign and x > 0 else s


def fmt_clock(model_time_s: float, *, prefix: str = "T+") -> str:
    """Model time since the start, the GUI's one form: ``T+1:20:00`` (hours, minutes, seconds)."""

    total = max(0, int(round(model_time_s)))
    return f"{prefix}{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def fmt_long_clock(model_time_s: float) -> str:
    """A hurricane's storm time as Watching and Results say it (gui/static/js/core.js stormClock for a long run):
    ``T+81 h (3 d 9 h)``, ``T+36 h (1 d 12 h)``, ``T+6 h``; a time between whole hours keeps one decimal."""

    s = max(0, int(round(model_time_s)))
    whole = s // 3600
    shown = str(whole) if s % 3600 == 0 else f"{s / 3600:.1f}"
    days = whole // 24
    return f"T+{shown} h ({days} d {whole % 24} h)" if days else f"T+{shown} h"


def fmt_km(m: float) -> str:
    return f"{fmt_num(m / 1000.0, 1 if abs(m) < 10000 and m % 1000 else 0)} km" if abs(m) >= 1000 else f"{fmt_num(m)} m"


def fmt_height(z_m: float) -> str:
    """The real height of a level above ground: '250 m', '1.25 km', '5 km'."""

    if z_m >= 1000.0:
        text = f"{z_m / 1000.0:.2f}".rstrip("0").rstrip(".")
        return f"{text} km"
    return f"{z_m:.0f} m"


def fmt_grid(dx_m: float) -> str:
    return f"{dx_m / 1000.0:g} km" if dx_m >= 1000.0 else f"{dx_m:g} m"


def upper(text: str) -> str:
    """Upper case for titles, keeping Greek symbols as the symbols they are (θ′ stays θ′, not Θ′)."""

    return "".join(ch if "\u03b1" <= ch <= "\u03c9" else ch.upper() for ch in text)


def data_line(*parts: str) -> str:
    return SEP.join(p for p in parts if p)


class Footer(list):
    """A picture's footer lines, carrying the word the GM1 mark shows beside "GM1" (:meth:`Overlay.mark`):
    "simulated" on GM1's own storms; a picture GM1 made of another model's output says "picture"."""

    def __init__(self, lines: Iterable[str] = (), mark_word: str = SIMULATED) -> None:
        super().__init__(lines)
        self.mark_word = mark_word


#: Marketing words and product names never appear on a picture.
BANNED_WORDS = ("amazing", "stunning", "revolutionary", "breakthrough", "cutting-edge", "cutting edge",
                "state-of-the-art", "state of the art", "blazing", "powerful", "world-class", "next-generation",
                "game-changing", "unprecedented", "incredible", "ultimate", "seamless", "effortless", "magic")
_WIND_NUM = re.compile(r"\d\s*m/s")
_RELATIVE = re.compile(r"(ground|vortex|frame|storm|domain)-relative|relative to the (moving )?domain", re.I)


def lint(texts: Iterable[str], *, vertical: Iterable[str] = (), require_simulated: bool = True) -> list[str]:
    """Honesty problems in a picture's text (empty = clean).

    * "simulated" is present;
    * "tornado" never appears (the GUI shows no tornado wording);
    * no marketing word appears;
    * every speed in m/s names the frame it is measured in (ground-, storm- or frame-relative), unless the text
      is listed in ``vertical``: a vertical velocity, which no frame motion changes.
    """

    texts = [t for t in texts if t]
    exempt = set(vertical)
    problems = []
    if require_simulated and "simulated" not in "\n".join(texts).lower():
        problems.append("'simulated' is missing")
    for t in texts:
        low = t.lower()
        if "tornad" in low:
            problems.append(f"tornado wording: {t!r}")
        for word in BANNED_WORDS:
            if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", low):
                problems.append(f"marketing word {word!r}: {t!r}")
        if t not in exempt and _WIND_NUM.search(t) and len(t.split()) > 2 and not _RELATIVE.search(t):
            problems.append(f"wind speed without its frame (ground-relative, frame-relative...): {t!r}")
    return problems


# =============================================================================
# the frame grammar (Pillow text engine)
# =============================================================================


class Overlay:
    """Text and marks for one picture, composited over any RGB array of the same size."""

    def __init__(self, size: tuple[int, int] = (1920, 1080), *, halo: float = 0.6):
        from PIL import Image

        self.W, self.H = int(size[0]), int(size[1])
        self.u = unit(self.W, self.H)
        # safe area: 80 / 64 layout units (16:9 landscape)
        self.x0, self.y0 = int(round(80 * self.u)), int(round(64 * self.u))
        self.x1, self.y1 = int(round(self.W - 80 * self.u)), int(round(self.H - 64 * self.u))
        self.layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        self.halo_mask = Image.new("L", (self.W, self.H), 0)
        self.halo_strength = halo
        self.texts: list[str] = []
        self.vertical: list[str] = []
        self.footer_cut: list[str] = []
        #: The word beside "GM1" in the mark (a :class:`Footer` given to :meth:`footer` may say another).
        self.mark_word = SIMULATED
        self._title_baseline: int | None = None
        self._head_bottom = self.y0
        #: How many rows the data line under the title took (:meth:`title`), and layout faults :meth:`check` reports.
        self._head_rows = 0
        self.layout_problems: list[str] = []
        self._rgba_cache = None

    def px(self, role: str) -> int:
        return max(10, int(round(TYPE_ROLES[role][1] * self.u)))

    def U(self, v: float) -> int:
        return int(round(v * self.u))

    def font(self, role: str, *, style: str | None = None, px: int | None = None, text: str = ""):
        st = style or TYPE_ROLES[role][0]
        while text and not has_glyphs(st, text) and st in _FONT_FALLBACK:
            st = _FONT_FALLBACK[st]
        return pil_font(st, px or self.px(role))

    def measure(self, text: str, role: str = "label", *, tracking: float = 0.0, px: int | None = None,
                style: str | None = None) -> tuple[int, int, int]:
        f = self.font(role, style=style, px=px, text=text)
        asc, desc = f.getmetrics()
        size = px or self.px(role)
        return int(math.ceil(f.getlength(text) + tracking * size * max(0, len(text) - 1))), asc, desc

    def text(self, xy: tuple[float, float], text: str, role: str = "label", *, colour: str | None = None,
             alpha: float = 1.0, anchor: str = "ls", tracking: float = 0.0, halo: bool = True,
             style: str | None = None, px: int | None = None, record: bool = True, vertical: bool = False,
             caps: bool | None = None) -> tuple[int, int, int, int]:
        """One line of text.  ``anchor``: horizontal l/m/r + vertical a(scender)/s(baseline)/d/m.

        ``vertical=True`` marks a text whose speeds are vertical velocities (exempt from the wind-frame rule)."""

        from PIL import Image, ImageChops, ImageDraw

        if not text or alpha <= 0:
            return (0, 0, 0, 0)
        _, _, col_tok, role_caps = TYPE_ROLES[role]
        if role_caps if caps is None else caps:
            text = upper(text)
        if record:
            self.texts.append(text)
            if vertical:
                self.vertical.append(text)
        f = self.font(role, style=style, px=px, text=text)
        size = px or self.px(role)
        asc, desc = f.getmetrics()
        width = f.getlength(text) + tracking * size * max(0, len(text) - 1)
        x, y = xy
        hx, vy = anchor[0], anchor[1]
        left = x - (width if hx == "r" else width / 2 if hx == "m" else 0)
        base = y + (asc if vy == "a" else -desc if vy == "d" else asc / 2 - desc / 2 if vy == "m" else 0)
        pad = int(size * 0.5) + 2
        bw, bh = int(math.ceil(width)) + 2 * pad, asc + desc + 2 * pad
        ox, oy = int(math.floor(left)) - pad, int(round(base)) - asc - pad
        mask = Image.new("L", (bw, bh), 0)
        d = ImageDraw.Draw(mask)
        if tracking:
            cx = left - ox
            for ch in text:
                d.text((cx, pad + asc), ch, font=f, fill=255, anchor="ls")
                cx += f.getlength(ch) + tracking * size
        else:
            d.text((left - ox, pad + asc), text, font=f, fill=255, anchor="ls")
        solid = Image.new("RGBA", (bw, bh), rgb8(token(colour or col_tok)) + (0,))
        solid.putalpha(mask.point(lambda v: int(v * alpha)))
        self.layer.alpha_composite(solid, dest=(max(ox, 0), max(oy, 0)), source=(max(-ox, 0), max(-oy, 0)))
        if halo:
            box = (max(ox, 0), max(oy, 0), min(ox + bw, self.W), min(oy + bh, self.H))
            if box[2] > box[0] and box[3] > box[1]:
                sub = mask.crop((box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)).point(lambda v: int(v * alpha))
                self.halo_mask.paste(ImageChops.lighter(self.halo_mask.crop(box), sub), box)
        self._rgba_cache = None
        return (ox + pad, oy + pad, ox + pad + int(width), oy + pad + asc + desc)

    def draw_rgba(self, patch, xy: tuple[float, float]) -> None:
        self.layer.alpha_composite(patch, dest=(int(xy[0]), int(xy[1])))
        self._rgba_cache = None

    def rule(self, x: float, y: float, length: float, *, thickness: float = 2.0, colour: str = "muted",
             alpha: float = 1.0, vertical: bool = False) -> None:
        from PIL import Image

        t = max(1, self.U(thickness))
        w, h = (t, int(length)) if vertical else (int(length), t)
        if w > 0 and h > 0:
            self.draw_rgba(Image.new("RGBA", (w, h), rgb8(token(colour), alpha)), (x, y))

    # -- grammar ----------------------------------------------------------------
    def title(self, text: str, data: str | None = None) -> "Overlay":
        """Title (at most six words, caps, bold) and the data line under it."""

        if len(text.split()) > 6:
            warnings.warn(f"viz20 title has more than 6 words: {text!r}")
        size = self.px("title")
        room = self.x1 - self.x0 - (self.measure("T+8:88:88", "clock")[0] + self.U(48))
        width = self.measure(upper(text), "title", tracking=0.02)[0]
        if width > room:
            size = max(int(size * 0.7), int(size * room / width))
        baseline = self.y0 + int(self.px("title") * 0.73)
        self.text((self.x0 - self.U(2), baseline), text, "title", anchor="ls", tracking=0.02, px=size)
        self._title_baseline = baseline
        self._head_bottom = baseline + int(self.px("title") * 0.25)
        if data:
            dsz = self.px("data")
            y = baseline + int(self.px("title") * 0.35) + dsz
            self.texts.append(data)
            room = self.x1 - self.x0 - self.U(260)
            lines, px = self.wrap(data, "data", room), None
            if len(lines) > 1:
                # A second row would reach the panels' titles under the head: the line is set smaller instead.
                lines, px = self._one_row(data, room)
            self._head_rows = len(lines)
            for line in lines:
                self.text((self.x0, y), line, "data", colour="text", record=False, px=px)
                self._head_bottom = y + int((px or dsz) * 0.35)
                y += int((px or dsz) * 1.4)
        return self

    def _one_row(self, data: str, room: int) -> tuple[list[str], int]:
        """``data`` on one row of at most ``room`` px: in smaller type, down to :data:`DATA_FLOOR` of the data size;
        below that the first clause (the run's name) gives way to an ellipsis.  Only a line that cannot be shortened
        so (a first clause of a few letters) wraps, at the smallest size."""

        size = self.px("data")
        floor = max(10, int(size * DATA_FLOOR))
        for px in range(size - 1, floor - 1, -1):
            if self.font("data", px=px, text=data).getlength(data) <= room:
                return [data], px
        parts = data.split(SEP)
        if len(parts) > 1:
            head, rest = parts[0], SEP.join(parts[1:])
            f = self.font("data", px=floor, text=data + "…")
            while len(head) > 12:
                head = head[:-1].rstrip(" ,")
                trial = f"{head}…{SEP}{rest}"
                if f.getlength(trial) <= room:
                    return [trial], floor
        return self.wrap(data, "data", room, px=floor), floor

    def clear_of_head(self, top: float, what: str) -> None:
        """A wrapped data line must end above ``top`` (the top of ``what``, drawn under the head): a picture where it
        does not is refused by :meth:`check`."""

        if self._head_rows > 1 and self._head_bottom > top:
            self.layout_problems.append(f"the data line under the title runs into {what}")

    def clock(self, value: float | str) -> "Overlay":
        s = fmt_clock(value) if isinstance(value, (int, float)) else value
        base = self._title_baseline if self._title_baseline is not None else self.y0 + int(self.px("clock") * 0.73)
        self.text((self.x1, base), s, "clock", anchor="rs")
        return self

    def footer(self, lines: str | Sequence[str], *, max_lines: int = 2, vertical: Iterable[str] = ()) -> "Overlay":
        """Grey honesty small print, bottom-left, clear of the mark; at most two lines.

        Lines listed in ``vertical`` state vertical velocities (exempt from the wind-frame rule, as ``text`` does)."""

        self.mark_word = getattr(lines, "mark_word", self.mark_word)
        if isinstance(lines, str):
            lines = [lines]
        width = self.x1 - self.x0 - self.mark_width() - self.U(48)
        lines = [line for line in lines if line]
        # Each line is kept to one row by cutting its last clauses ('  ·  '-separated), never by shrinking type;
        # what was cut is recorded, and the product tests require that nothing is.
        vertical = set(vertical)
        kept = []
        for line in lines[:max_lines]:
            parts = line.split(SEP)
            while len(parts) > 1 and len(self.wrap(SEP.join(parts), "footer", width)) > 1:
                self.footer_cut.append(parts.pop())
            kept.append(SEP.join(parts))
            if line in vertical:
                self.vertical.append(kept[-1])
        self.footer_cut.extend(lines[max_lines:])
        wrapped: list[str] = []
        self.texts.extend(kept)
        for line in kept:
            wrapped += self.wrap(line, "footer", width)
        if len(wrapped) > max_lines:
            warnings.warn(f"viz20 footer wraps to {len(wrapped)} lines (rule: <= {max_lines}); cut words")
        lh = int(self.px("footer") * 1.38)
        _, desc = self.font("footer").getmetrics()
        y = self.y1 - desc
        for line in reversed(wrapped):
            self.text((self.x0, y), line, "footer", record=False)
            y -= lh
        return self

    def mark_width(self) -> int:
        size = self.px("mark")
        return int(size * 1.25 + size * 0.45 + self.measure("GM1", "mark")[0] + size * 0.5
                   + self.measure(self.mark_word, "mark", style="book")[0])

    def mark(self, *, alpha: float = 0.85) -> "Overlay":
        """The GM1 mark bottom-right: a small gold spiral, 'GM1' and 'simulated' (or :attr:`mark_word`)."""

        size = self.px("mark")
        _, desc = self.font("mark").getmetrics()
        x, base = self.x1 - self.mark_width(), self.y1 - desc
        glyph = int(size * 1.25)
        self.draw_rgba(spiral_glyph(glyph, alpha=alpha), (x, base - int(size * 0.80 / 2) - glyph // 2))
        gx = x + glyph + int(size * 0.45)
        self.text((gx, base), "GM1", "mark", colour="text", alpha=alpha)
        sx = gx + self.measure("GM1", "mark")[0] + int(size * 0.5)
        self.text((sx, base), self.mark_word, "mark", colour="muted", alpha=alpha, style="book")
        return self

    def readout(self, xy: tuple[float, float], rows: Sequence[tuple[str, str]], *,
                vertical: Iterable[str] = ()) -> "Overlay":
        """Stacked small-caps captions over thin values.  Values listed in ``vertical`` are vertical velocities."""

        x, y = xy
        vs = self.px("data") * 1.5
        ms = self.px("micro")
        vertical = set(vertical)
        for cap, val in rows:
            self.text((x, y + ms), upper(cap), "micro", anchor="ls", tracking=0.08, vertical=val in vertical)
            self.text((x, y + ms + int(vs * 1.05)), val, "hero_number", anchor="ls", px=int(vs),
                      vertical=val in vertical)
            y += ms + int(vs * 1.55)
        return self

    def ramp_bar(self, name: str, box: tuple[int, int, int, int], *, vmin: float | None = None,
                 vmax: float | None = None, ticks: Sequence[tuple[float, str]] = (), title: str | None = None,
                 vertical: bool = False) -> "Overlay":
        from PIL import Image

        r = RAMPS[name]
        x0, y0, x1, y1 = (int(v) for v in box)
        lo = r.vmin if vmin is None else vmin
        hi = r.vmax if vmax is None else vmax
        vals = np.linspace(lo, hi, max(2, x1 - x0))
        strip = np.repeat((r.colours(vals, lo, hi) * 255).astype(np.uint8)[None], max(1, y1 - y0), axis=0)
        self.draw_rgba(Image.fromarray(strip).convert("RGBA"), (x0, y0))
        ms = self.px("micro")
        for v, lab in ticks:
            tx = x0 + (v - lo) / (hi - lo) * (x1 - x0)
            anc = "ls" if tx - x0 < 4 else "rs" if x1 - tx < 4 else "ms"
            self.text((tx, y1 + self.U(8) + ms), lab, "micro", anchor=anc, halo=False, vertical=vertical)
        if title:
            self.text(((x0 + x1) / 2, y0 - self.U(9)), title, "micro", colour="data", anchor="ms", halo=False,
                      vertical=vertical)
        return self

    def ring(self, centre: tuple[float, float], radius: float, *, width: float = 2.0, colour: str = "gold",
             alpha: float = 0.95, edge: str | None = None, edge_width: float = 1.5) -> "Overlay":
        """A ring of outer radius ``radius`` (px).  ``edge``: a colour laid ``edge_width`` wider on both sides,
        under the ring, so it stands out on a bright field as well as on a dark one."""

        from PIL import Image, ImageDraw

        ss = 4
        r = int(radius)
        lw = max(1, self.U(width))
        e = max(1, self.U(edge_width)) if edge else 0
        m = lw + 2 + e
        size = 2 * (r + m)
        img = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if edge:
            d.ellipse(((m - e) * ss, (m - e) * ss, (size - m + e) * ss, (size - m + e) * ss),
                      outline=rgb8(token(edge), alpha), width=(lw + 2 * e) * ss)
        bb = (m * ss, m * ss, (size - m) * ss, (size - m) * ss)
        d.ellipse(bb, outline=rgb8(token(colour), alpha), width=lw * ss)
        self.draw_rgba(img.resize((size, size), Image.LANCZOS),
                       (int(centre[0]) - size // 2, int(centre[1]) - size // 2))
        return self

    # -- output -----------------------------------------------------------------
    def rgba(self) -> np.ndarray:
        from PIL import Image, ImageFilter

        if self._rgba_cache is None:
            base = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
            if self.halo_strength > 0 and self.halo_mask.getbbox():
                blur = self.halo_mask.filter(ImageFilter.GaussianBlur(max(1.5, 3.2 * self.u)))
                shadow = Image.new("RGBA", (self.W, self.H), rgb8("void") + (0,))
                shadow.putalpha(blur.point(lambda v: int(v * self.halo_strength)))
                base.alpha_composite(shadow)
            base.alpha_composite(self.layer)
            self._rgba_cache = np.asarray(base, dtype=np.float32) / 255.0
        return self._rgba_cache

    def composite(self, base: np.ndarray) -> np.ndarray:
        b = np.asarray(base, dtype=np.float32)[..., :3] / 255.0
        if b.shape[:2] != (self.H, self.W):
            raise ValueError(f"base is {b.shape[1]}x{b.shape[0]}, overlay is {self.W}x{self.H}")
        o = self.rgba()
        a = o[..., 3:4]
        return (np.clip(b * (1 - a) + o[..., :3] * a, 0, 1) * 255 + 0.5).astype(np.uint8)

    def check(self) -> list[str]:
        return lint(self.texts, vertical=self.vertical) + list(self.layout_problems)

    def wrap(self, text: str, role: str, width: int, *, px: int | None = None) -> list[str]:
        words, lines, cur = text.split(), [], ""
        f = self.font(role, text=text, px=px)
        for w in words:
            trial = f"{cur} {w}".strip()
            if cur and f.getlength(trial) > width:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines


@lru_cache(maxsize=16)
def _spiral_rgba(size: int, alpha_q: int) -> bytes:
    from PIL import Image, ImageDraw

    ss = 6
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    turns, n = 1.6, 160
    r_out = S * 0.46
    pts = []
    for i in range(n + 1):
        t = i / n
        th = 2 * math.pi * turns * t
        r = r_out * (0.10 + 0.90 * t) ** 1.15
        pts.append((c + r * math.cos(-th + math.pi / 2), c - r * math.sin(-th + math.pi / 2), t))
    a = alpha_q / 255.0
    lw_max = max(ss, int(S * 0.13))
    for (xa, ya, ta), (xb, yb, _) in zip(pts[:-1], pts[1:]):
        col = RAMPS["ember"].at(1.0 - 0.30 * ta)
        width = max(ss, int(lw_max * (0.50 + 0.50 * (1 - ta))))
        d.line((xa, ya, xb, yb), fill=rgb8(col, a * (0.62 + 0.38 * (1 - ta) ** 0.6)), width=width)
    d.ellipse((c - lw_max * 0.7, c - lw_max * 0.7, c + lw_max * 0.7, c + lw_max * 0.7), fill=rgb8("core", a))
    return img.resize((size, size), Image.LANCZOS).tobytes()


def spiral_glyph(size: int, *, alpha: float = 1.0):
    from PIL import Image

    return Image.frombytes("RGBA", (size, size), _spiral_rgba(int(size), int(round(255 * alpha))))


def figure(size: tuple[int, int] = (1920, 1080), *, dpi: float = 100.0):
    """A black matplotlib figure of exactly ``size`` pixels (Agg), for charts drawn under an Overlay."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    apply_matplotlib()
    return plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=PALETTE["void"])


def fig_to_rgb(fig) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


__all__ = [
    "BANNED_WORDS", "DOT", "MINUS", "NWS_REFLECTIVITY", "Overlay", "PALETTE", "PER_S", "PRIME", "RAMPS", "Ramp",
    "Footer", "SEP", "SIMULATED", "apply_matplotlib", "has_glyphs", "upper", "colours", "data_line", "dbz_colours", "fig_to_rgb", "figure", "finish",
    "fmt_clock", "fmt_grid", "fmt_long_clock", "fmt_height", "fmt_km", "fmt_num", "font_path", "hex_rgb", "lint", "mpl_fontprops",
    "rgb8", "to8", "token", "value_stops",
]
