"""Sounding files people have, read and checked for a guided storm, and written as CM1's ``input_sounding``.

CM1 r21.1 reads one layout (``isnd = 7``, ``base.F:495-607``): a surface line with the surface pressure in hPa,
the potential temperature in K and the water-vapour mixing ratio in g/kg, then one line per level with the height
above the ground in m, the potential temperature, the mixing ratio and the two wind components in m/s.  Most
soundings people have are not in that layout: they list pressure, height, temperature, dewpoint, wind direction
and wind speed.  This module reads four layouts and writes the first:

``cm1``
    CM1's own ``input_sounding`` (kept byte for byte when nothing is wrong with it).
``wyoming``
    The University of Wyoming archive's text listing: a ``PRES HGHT TEMP DWPT ...`` header, a line of units and
    fixed columns seven characters wide.  The wind speed column is ``SKNT`` (knots) or ``SPED`` (m/s; the units
    line says which).  A saved page (HTML) is read too.
``sharppy``
    SHARPpy's text file: ``%TITLE%``, then ``%RAW%``, one ``pressure, height, temperature, dewpoint, direction,
    speed`` line per level (hPa, m above sea level, C, C, degrees, knots; ``-9999`` is a missing value), ``%END%``.
``table``
    A table with one header line that names its columns (comma, semicolon, tab or space separated): pressure,
    height, temperature, dewpoint, and the wind as direction and speed or as u and v.  A unit in a name is read
    (``speed_kt``, ``wspd (m/s)``, ``temp_K``, ``pressure_Pa``).  Pressure in Pa and temperature in K are told
    from hPa and C by their size.  A wind speed whose column does not say knots or m/s is not guessed: the
    reader asks (:class:`SoundingRefused` with ``code = "speed-unit"``), and ``speed_unit`` answers.

The conversions (:func:`convert`), with CM1 r21.1's constants (``constants.F``), are the ones the shipped balloon
soundings were made with (``data/soundings/PROVENANCE.md``):

* height above the ground = height - the first level's height (the first level is the surface line);
* potential temperature = T (1000 hPa / p) ^ (Rd / cp), T in K;
* mixing ratio = CM1's ``rslf`` (saturation mixing ratio over liquid, Bolton's formula) at the dewpoint;
* u = -speed sin(direction), v = -speed cos(direction), the direction being where the wind blows from;
  1 knot = 0.514444 m/s.

A level missing one of the six values, or not higher than the level below it, is left out and named.  Nothing is
smoothed or interpolated.  CM1's layout has no surface wind: CM1 extrapolates it from the first two levels
(``base.F:617-618``), so the first level's wind is not written.

:func:`check` says, in plain words and with the line of the user's own file, what would stop CM1 or give a wrong
run (a problem) and what is only worth knowing (a warning).  :func:`extend` adds levels above a sounding that
stops below the model top.  Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import html
import math
import re
from typing import Any, Mapping, Sequence

# CM1 r21.1 constants (src/constants.F).
G = 9.81
RD = 287.04
CP = 1005.7
RV = 461.5
P00 = 1.0e5
EPS = RD / RV
REPS = RV / RD
ROVCP = RD / CP
KNOT_M_S = 0.514444
MPH_M_S = 0.44704
FOOT_M = 0.3048

FORMATS = ("cm1", "wyoming", "sharppy", "table")
FORMAT_WORDS = {
    "cm1": "CM1 input_sounding",
    "wyoming": "University of Wyoming text listing",
    "sharppy": "SHARPpy text file",
    "table": "a table with named columns",
}
#: The largest text read as a sounding (a balloon with 7000 levels is about 400 kB).
MAX_BYTES = 4 * 1024 * 1024
#: A sounding that stops below this height is not extended: too little of it is measured.
EXTEND_FROM_M = 3000.0
#: The spacing of the levels :func:`extend` adds.
EXTEND_STEP_M = 500.0
#: A top level warmer than this (-40 C) is probably below the tropopause.
TROPOPAUSE_WARM_K = 233.15
#: The warm bubble's centre height in CM1 (``init3d.F``: zc = 1400 m), where :func:`bubble_moisture` looks.
BUBBLE_CENTRE_M = 1400.0
#: Potential temperature falling with height below this height is the afternoon surface layer (a note, not a warning)
SURFACE_LAYER_M = 500.0
#: ... and above it, a fall of more than this (K) is worth a warning.
LARGE_FALL_K = 1.0
MISSING = frozenset({"", "nan", "na", "n/a", "null", "none", "missing", "mm", "m", "--", "-", "///", "/////"})

ACCEPTED = ("GM1 reads CM1's input_sounding, a University of Wyoming text listing (PRES HGHT TEMP DWPT ...), a "
            "SHARPpy file (%RAW% ... %END%), or a table with a header line that names pressure, height, "
            "temperature, dewpoint and the wind")


class SoundingRefused(ValueError):
    """A file that cannot be read as a sounding; ``text`` says why and ``fix`` what to do, in plain words."""

    def __init__(self, code: str, text: str, fix: str, *, line: int | None = None,
                 offer: Mapping[str, Any] | None = None) -> None:
        super().__init__(text)
        self.code = code
        self.text = text
        self.fix = fix
        self.line = line
        self.offer = dict(offer or {})

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"code": self.code, "text": self.text, "fix": self.fix}
        if self.line is not None:
            row["line"] = self.line
        if self.offer:
            row["offer"] = dict(self.offer)
        return row


@dataclass(frozen=True)
class Finding:
    """One thing :func:`check` found: ``level`` is ``"problem"`` (it cannot run as it is) or ``"warning"``."""

    level: str
    code: str
    text: str
    fix: str = ""
    line: int | None = None
    offer: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"code": self.code, "text": self.text}
        if self.fix:
            row["fix"] = self.fix
        if self.line is not None:
            row["line"] = self.line
        if self.offer:
            row["offer"] = dict(self.offer)
        return row


@dataclass(frozen=True)
class Profile:
    """A sounding in CM1's own terms.  ``lines[k]`` is the line of the user's file level ``k`` came from."""

    surface_hpa: float
    surface_theta_k: float
    surface_qv_gkg: float
    z_m: tuple[float, ...]
    theta_k: tuple[float, ...]
    qv_gkg: tuple[float, ...]
    u_ms: tuple[float, ...]
    v_ms: tuple[float, ...]
    surface_line: int = 1
    lines: tuple[int, ...] = ()
    #: How many of the top levels :func:`extend` added (they are not from the file).
    extended: int = 0

    @property
    def levels(self) -> int:
        return len(self.z_m)

    @property
    def top_m(self) -> float:
        return self.z_m[-1] if self.z_m else 0.0

    def line_of(self, k: int) -> int | None:
        return self.lines[k] if 0 <= k < len(self.lines) else None


@dataclass(frozen=True)
class Reading:
    """What :func:`read` made of a file."""

    format: str
    profile: Profile
    #: The CM1 text: the user's own bytes for a CM1 file, the conversion's otherwise.
    text: str
    converted: bool
    #: Each conversion made, as a sentence (empty for a CM1 file).
    conversions: tuple[str, ...] = ()
    #: Levels left out: ``{"line", "why"}``.
    dropped: tuple[Mapping[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    station_m: float | None = None
    #: What the conversion did that changes the air the storm runs in (the ground taken higher up).
    warnings: tuple[str, ...] = ()

    @property
    def format_words(self) -> str:
        return FORMAT_WORDS[self.format]


# ------------------------------------------------------------------ thermodynamics (CM1's forms)


def rslf(p_pa: float, t_k: float) -> float:
    """CM1's rslf: saturation mixing ratio over liquid (kg/kg), with its min(es, p/2) clamp.

    Air colder than 100 K, or with no pressure, holds no vapour here (Bolton's formula would overflow there; no
    real air is that cold, and :func:`check` says so with the line)."""

    if not (t_k > 100.0 and p_pa > 0.0):
        return 0.0
    esl = min(611.2 * math.exp(17.67 * (t_k - 273.15) / (t_k - 29.65)), 0.5 * p_pa)
    return EPS * esl / (p_pa - esl)


def pressures_pa(profile: Profile) -> list[float]:
    """CM1's hydrostatic pressure at each level (``base.F:627-635``), the surface first."""

    z = [0.0, *profile.z_m]
    th = [profile.surface_theta_k, *profile.theta_k]
    qv = [profile.surface_qv_gkg / 1000.0, *(q / 1000.0 for q in profile.qv_gkg)]
    thv = [t * (1.0 + REPS * q) / (1.0 + q) for t, q in zip(th, qv)]
    pi = [(profile.surface_hpa * 100.0 / P00) ** ROVCP]
    for k in range(1, len(z)):
        pi.append(pi[-1] - G * (z[k] - z[k - 1]) / (CP * 0.5 * (thv[k] + thv[k - 1])))
    return [P00 * max(value, 1.0e-6) ** (CP / RD) for value in pi]


def temperatures_k(profile: Profile) -> list[float]:
    """Temperature at each level (the surface first), from theta and CM1's hydrostatic pressure."""

    th = [profile.surface_theta_k, *profile.theta_k]
    return [t * (p / P00) ** ROVCP for t, p in zip(th, pressures_pa(profile))]


def interpolate(z: Sequence[float], values: Sequence[float], height: float) -> float:
    """``values`` at ``height``, linear between levels, the ends held."""

    if height <= z[0]:
        return values[0]
    for k in range(1, len(z)):
        if height <= z[k]:
            weight = (height - z[k - 1]) / (z[k] - z[k - 1])
            return values[k - 1] + weight * (values[k] - values[k - 1])
    return values[-1]


def bubble_moisture(profile: Profile) -> dict[str, float] | None:
    """Water vapour at the warm bubble's centre (1.4 km above the ground) against the surface's, or None."""

    if not profile.z_m or profile.top_m < BUBBLE_CENTRE_M or profile.surface_qv_gkg <= 0.0:
        return None
    aloft = interpolate([0.0, *profile.z_m], [profile.surface_qv_gkg, *profile.qv_gkg], BUBBLE_CENTRE_M)
    return {"height_m": BUBBLE_CENTRE_M, "surface_g_kg": profile.surface_qv_gkg, "aloft_g_kg": aloft,
            "ratio": aloft / profile.surface_qv_gkg}


# ------------------------------------------------------------------ telling the layouts apart


def _number(token: str) -> float:
    # Fortran list-directed input takes a D exponent as well as an E.
    value = float(token.replace("d", "e").replace("D", "e"))
    if not math.isfinite(value):
        raise ValueError(token)
    return value


def _is_number(token: str) -> bool:
    try:
        _number(token)
    except ValueError:
        return False
    return True


def _plain(text: str) -> str:
    """The text with a saved page's tags taken out (the Wyoming archive serves its listing inside a page)."""

    if re.search(r"<\s*(pre|html|body|h\d)\b", text, re.I):
        text = html.unescape(re.sub(r"<[^>]*>", "", text))
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")


_WYOMING_HEAD = re.compile(r"^\s*PRES\s+HGHT\s+TEMP\s+DWPT\b", re.I)


def detect(text: str) -> str:
    """Which layout ``text`` is, or :class:`SoundingRefused` saying what GM1 reads."""

    plain = _plain(text)
    lines = [line for line in plain.split("\n") if line.strip()]
    if not lines:
        raise SoundingRefused("empty", "The file is empty.", "Choose a file with a sounding in it. " + ACCEPTED + ".")
    if any(line.strip().upper().startswith("%RAW%") for line in lines):
        return "sharppy"
    if any(_WYOMING_HEAD.match(line) for line in lines):
        return "wyoming"
    first = next((line for line in lines if not line.lstrip().startswith(("#", "!"))), lines[0])
    tokens = first.replace(",", " ").replace(";", " ").split()
    if tokens and all(_is_number(token) for token in tokens):
        return "cm1"
    if re.search(r"[A-Za-z]", first):
        return "table"
    raise SoundingRefused(
        "format", "This does not look like a sounding GM1 can read.",
        ACCEPTED + ". See docs/SOUNDINGS.md for each layout with an example.", line=1)


# ------------------------------------------------------------------ CM1's own layout


def _cm1_rows(text: str) -> list[tuple[int, list[str]]]:
    rows = []
    for number, line in enumerate(_plain(text).split("\n"), start=1):
        tokens = line.replace(",", " ").split()
        if tokens:
            rows.append((number, tokens))
    return rows


def parse_cm1(text: str) -> Profile:
    """CM1's ``input_sounding`` as ``base.F`` reads it, or the first line that is not what CM1 reads."""

    rows = _cm1_rows(text)
    if not rows:
        raise SoundingRefused("empty", "The file is empty.", "Choose a file with a sounding in it.")
    number, tokens = rows[0]
    if tokens[0].lstrip().startswith(("#", "!")):
        raise SoundingRefused(
            "format", f"Line {number} is a comment. CM1 reads numbers from the first line on, so it would stop here.",
            "Take the comment lines out.", line=number)
    if not all(_is_number(token) for token in tokens[:3]) or len(tokens) < 3:
        raise SoundingRefused(
            "surface-line",
            f"Line {number} should be the surface line: three numbers, the surface pressure in hPa, the potential "
            f"temperature in K and the water-vapour mixing ratio in g/kg. It has {len(tokens)} value(s).",
            "Start the file with the surface line, for example: 1000.0 300.0 14.0", line=number)
    if len(tokens) > 5 and all(_is_number(token) for token in tokens):
        # Rows of numbers with no header line: most often a listing copied without its header (a University of
        # Wyoming listing has eleven numbers on a line), or a table saved without its first line.
        raise SoundingRefused(
            "no-header",
            f"Line {number} has {len(tokens)} numbers, and there is no header line naming them, so GM1 cannot "
            "tell which number is the pressure, the height, the temperature or the wind.",
            "Copy the listing from its header line down (for a University of Wyoming listing, from the line "
            "PRES HGHT TEMP DWPT ...), or add a first line naming the columns, for example: "
            "pressure,height,temperature,dewpoint,direction,speed_kt", line=number)
    if len(tokens) >= 5 and all(_is_number(token) for token in tokens[:5]):
        raise SoundingRefused(
            "surface-line",
            f"The surface line is missing. Line {number} has five numbers, as a level line does (height, potential "
            "temperature, mixing ratio, u, v). CM1 would read its first three as the surface pressure, potential "
            "temperature and mixing ratio, and the storm would run in the wrong air.",
            "Add a first line with three numbers: the surface pressure in hPa, the potential temperature in K and "
            "the mixing ratio in g/kg at the ground.", line=number)
    surface = [_number(token) for token in tokens[:3]]
    surface_line = number
    z: list[float] = []
    th: list[float] = []
    qv: list[float] = []
    u: list[float] = []
    v: list[float] = []
    lines: list[int] = []
    for number, tokens in rows[1:]:
        bad = next((token for token in tokens[:5] if not _is_number(token)), None)
        if bad is not None:
            raise SoundingRefused(
                "format", f"Line {number} has {bad!r}, which is not a number. CM1 would stop reading here.",
                "Each level line is five numbers: height (m), potential temperature (K), mixing ratio (g/kg), "
                "u (m/s), v (m/s).", line=number)
        if len(tokens) == 3:
            raise SoundingRefused(
                "no-wind",
                f"Line {number} has three numbers: height, potential temperature and mixing ratio, with no wind. "
                "A storm needs the wind at each level.",
                "Add the two wind components to each level line: u (toward the east) and v (toward the north), "
                "in m/s.", line=number)
        if len(tokens) < 5:
            raise SoundingRefused(
                "format",
                f"Line {number} has {len(tokens)} number(s). CM1 reads five on each level line and would stop here.",
                "Each level line is five numbers: height (m), potential temperature (K), mixing ratio (g/kg), "
                "u (m/s), v (m/s).", line=number)
        values = [_number(token) for token in tokens[:5]]
        z.append(values[0])
        th.append(values[1])
        qv.append(values[2])
        u.append(values[3])
        v.append(values[4])
        lines.append(number)
    return Profile(surface[0], surface[1], surface[2], tuple(z), tuple(th), tuple(qv), tuple(u), tuple(v),
                   surface_line=surface_line, lines=tuple(lines))


_PLAIN_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")


def strict_cm1(text: str) -> bool:
    """Whether ``text`` is already in the form the engine reads (``rust/cm1-state``, ``InputSounding::parse``).

    That reader is stricter than CM1's Fortran: ASCII lines ending in LF, no blank line anywhere (a second
    newline at the end is one), exactly three words on the first line and five on every other, each a plain
    number.  A file that is not is written again by :func:`tidy_cm1` with the same numbers."""

    if not text or "\r" in text:
        return False
    try:
        text.encode("ascii")
    except UnicodeError:
        return False
    lines = text.split("\n")
    if lines[-1] == "":
        lines = lines[:-1]
    if len(lines) < 3:
        return False
    for index, line in enumerate(lines):
        words = line.split()
        if len(words) != (3 if index == 0 else 5) or not all(_PLAIN_NUMBER.match(word) for word in words):
            return False
    return True


def tidy_cm1(text: str) -> tuple[str, list[str]]:
    """(``text`` in the engine's strict form, what was taken out, in plain words) for a CM1 file
    :func:`parse_cm1` read: the numbers CM1 reads, as the user wrote them, one line each."""

    plain = _plain(text).split("\n")
    out: list[str] = []
    extra: list[int] = []
    for number, line in enumerate(plain, start=1):
        tokens = line.replace(",", " ").split()
        if not tokens:
            continue
        width = 3 if not out else 5
        if len(tokens) > width:
            extra.append(number)
        if len(tokens) == width and "," not in line and all(_PLAIN_NUMBER.match(token) for token in tokens):
            out.append(line.rstrip())         # a line already as the engine reads it is kept as it was
            continue
        out.append(" ".join(token if _PLAIN_NUMBER.match(token) else repr(_number(token)) for token in tokens[:width]))
    blank = sum(1 for line in (plain[:-1] if plain and plain[-1] == "" else plain) if not line.strip())
    said: list[str] = []
    if blank:
        said.append(f"{blank} blank line(s)")
    if "," in text:
        said.append("the commas between numbers")
    if extra:
        said.append(f"the words after the numbers CM1 reads on {len(extra)} line(s), first on line {extra[0]}")
    return "\n".join(out) + "\n", said


def render(profile: Profile) -> str:
    """The profile as CM1's ``input_sounding`` text (the layout the shipped balloon files have)."""

    lines = ["%12.4f %14.6f %14.6f" % (profile.surface_hpa, profile.surface_theta_k, profile.surface_qv_gkg)]
    for k in range(profile.levels):
        lines.append("%12.4f %14.6f %14.6f %10.4f %10.4f" % (
            profile.z_m[k], profile.theta_k[k], profile.qv_gkg[k], profile.u_ms[k], profile.v_ms[k]))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ measured levels (the other layouts)


@dataclass
class _Level:
    line: int
    p_hpa: float | None = None
    z_m: float | None = None
    t_c: float | None = None
    td_c: float | None = None
    direction: float | None = None
    speed_ms: float | None = None
    u_ms: float | None = None
    v_ms: float | None = None


def _value(token: str) -> float | None:
    """A number, or None for a missing-value mark (blank, NaN, -9999 and its kin)."""

    token = token.strip()
    if token.lower() in MISSING:
        return None
    try:
        value = float(token)
    except ValueError:
        raise ValueError(token) from None
    if not math.isfinite(value) or value <= -998.0 or value in (99999.0, 999999.0):
        return None
    return value


def _wyoming_levels(text: str) -> tuple[list[_Level], list[str]]:
    lines = _plain(text).split("\n")
    start = next(index for index, line in enumerate(lines) if _WYOMING_HEAD.match(line))
    names = lines[start].split()
    units = lines[start + 1].split() if start + 1 < len(lines) else []
    speed_name = next((name for name in names if name.upper() in ("SKNT", "SPED")), None)
    if speed_name is None or "DRCT" not in [name.upper() for name in names]:
        raise SoundingRefused(
            "columns", "The listing has no wind columns (DRCT and SKNT or SPED).",
            "Save the listing with its wind direction and speed columns.", line=start + 1)
    knots = speed_name.upper() == "SKNT"
    if len(units) == len(names):
        unit = units[[name.upper() for name in names].index(speed_name.upper())].lower()
        knots = unit.startswith("kn") or unit == "kt"
    upper = [name.upper() for name in names]
    # The archive's columns are seven characters wide, and a missing value is left blank: read by position when
    # the header itself sits in those columns, so a blank is known for the column it is in.
    width = 7
    fixed = all(lines[start][width * j:width * (j + 1)].strip().upper() == name for j, name in enumerate(upper))
    levels: list[_Level] = []
    begun = False
    def numeric_next(index: int) -> bool:
        """Whether the next line that is not blank is another level line (a blank line inside the listing)."""

        following = next((line.split() for line in lines[index + 1:] if line.strip()), [])
        return len(following) >= 2 and _is_number(following[0]) and _is_number(following[1])

    for index in range(start + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped or set(stripped) <= {"-"}:
            if begun and not stripped and not numeric_next(index):
                break
            continue
        if not begun and not _is_number(stripped.split()[0]):
            continue  # the units line
        if not _is_number(stripped.split()[0]):
            break     # the block of station numbers and indices under the listing
        begun = True
        cells: dict[str, str] = {}
        tokens = stripped.split()
        if len(tokens) == len(upper):
            cells = dict(zip(upper, tokens))
        elif fixed:
            for j, name in enumerate(upper):
                cells[name] = line[width * j:width * (j + 1)]
        else:
            # Fewer values than columns and no fixed columns to place them by: which are missing is not known.
            cells = {"PRES": tokens[0]}
        try:
            p, z, t, td = (_value(cells.get(name, "")) for name in ("PRES", "HGHT", "TEMP", "DWPT"))
            direction = _value(cells.get("DRCT", ""))
            speed = _value(cells.get(speed_name.upper(), ""))
        except ValueError as error:
            raise SoundingRefused(
                "format", f"Line {index + 1} has {str(error)!r}, which is not a number.",
                "Save the listing again as plain text, with its columns as the archive writes them.",
                line=index + 1) from None
        levels.append(_Level(index + 1, p, z, t, td, direction,
                             None if speed is None else speed * (KNOT_M_S if knots else 1.0)))
    said = [
        "Wind speed read as " + ("knots (1 knot = 0.514444 m/s)" if knots else "m/s") + f", from the {speed_name} "
        "column" + (" and its units line" if len(units) == len(names) else "") + ".",
    ]
    return levels, said


def _several(text: str, layout: str) -> str | None:
    """A note when one file holds several soundings (an archive's range of times): which one GM1 read."""

    lines = _plain(text).split("\n")
    if layout == "wyoming":
        heads = [index for index, line in enumerate(lines) if _WYOMING_HEAD.match(line)]
    elif layout == "sharppy":
        heads = [index for index, line in enumerate(lines) if line.strip().upper().startswith("%RAW%")]
    else:
        return None
    if len(heads) < 2:
        return None
    title = next((line.strip() for line in reversed(lines[:heads[0]]) if "observations at" in line.lower()), "")
    return (f"This file holds {len(heads)} soundings; GM1 read the first one" + (f" ({title[:80]})" if title else "")
            + ". To use another, save it on its own.")


def _sharppy_levels(text: str) -> tuple[list[_Level], list[str]]:
    lines = _plain(text).split("\n")
    start = next(index for index, line in enumerate(lines) if line.strip().upper().startswith("%RAW%"))
    levels: list[_Level] = []
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        if stripped.upper().startswith("%END%"):
            break
        tokens = [token.strip() for token in stripped.replace(";", ",").split(",")]
        if len(tokens) < 6:
            tokens = stripped.split()
        if len(tokens) < 6:
            raise SoundingRefused(
                "format", f"Line {index + 1} has {len(tokens)} value(s); a SHARPpy level line has six: pressure, "
                "height, temperature, dewpoint, wind direction, wind speed.",
                "Give six values on each line between %RAW% and %END%.", line=index + 1)
        try:
            p, z, t, td, direction, speed = (_value(token) for token in tokens[:6])
        except ValueError as error:
            raise SoundingRefused("format", f"Line {index + 1} has {str(error)!r}, which is not a number.",
                                  "Give six numbers on each line between %RAW% and %END%.",
                                  line=index + 1) from None
        t, td, direction, speed = (_unmarked(value) for value in (t, td, direction, speed))
        levels.append(_Level(index + 1, p, z, t, td, direction, None if speed is None else speed * KNOT_M_S))
    return levels, ["Wind speed read as knots (1 knot = 0.514444 m/s), as SHARPpy files give it."]


def _unmarked(value: float | None) -> float | None:
    """None for 9999, a missing-value mark some files use for temperature and wind (never a real one)."""

    return None if value is not None and abs(value) in (9999.0, 999.0) else value


#: Column names a table may use, by what they are.  A name is lowercased and its punctuation dropped first.
_NAMES: Mapping[str, tuple[str, ...]] = {
    "p": ("p", "pres", "press", "pressure", "prs", "level", "airpressure"),
    "z": ("z", "hght", "hgt", "height", "alt", "altitude", "gph", "geopotentialheight", "h", "heightmsl", "zmsl",
          "heightagl", "zagl"),
    "t": ("t", "temp", "tmpc", "tmpk", "temperature", "airtemperature", "tc", "tk", "airtemp", "tmp"),
    "td": ("td", "dwpt", "dwpc", "dwpk", "dewpoint", "dewpointtemperature", "dewp", "tdc", "tdk", "dew", "dpt",
           "dewpt"),
    "dir": ("wdir", "drct", "dir", "direction", "winddirection", "wd", "winddir"),
    "spd": ("wspd", "sped", "sknt", "speed", "windspeed", "ws", "spd", "wspeed", "windspd"),
    "u": ("u", "uwnd", "uwind", "ugrd", "ucomponent", "windu"),
    "v": ("v", "vwnd", "vwind", "vgrd", "vcomponent", "windv"),
}
#: Units a column name may carry, by kind: the factor to GM1's unit (hPa, m, m/s), or the temperature scale.
_UNITS: Mapping[str, Mapping[str, Any]] = {
    "p": {"hpa": 1.0, "mb": 1.0, "mbar": 1.0, "millibar": 1.0, "millibars": 1.0, "pa": 0.01, "kpa": 10.0},
    "z": {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0, "km": 1000.0, "ft": FOOT_M,
          "feet": FOOT_M, "foot": FOOT_M, "gpm": 1.0},
    "t": {"c": "c", "degc": "c", "celsius": "c", "k": "k", "kelvin": "k", "degk": "k", "f": "f", "degf": "f",
          "fahrenheit": "f"},
    "wind": {"kt": KNOT_M_S, "kts": KNOT_M_S, "knot": KNOT_M_S, "knots": KNOT_M_S, "kn": KNOT_M_S,
             "ms": 1.0, "mps": 1.0, "ms1": 1.0, "mpers": 1.0, "meterspersecond": 1.0, "metrespersecond": 1.0,
             "msec": 1.0, "msec1": 1.0, "mpersec": 1.0, "mpsec": 1.0,
             "mph": MPH_M_S, "kmh": 1.0 / 3.6, "kph": 1.0 / 3.6, "kmhr": 1.0 / 3.6},
    "dir": {"deg": 1.0, "degrees": 1.0, "degree": 1.0},
}
_IMPLIED = {"tmpc": "c", "dwpc": "c", "tc": "c", "tdc": "c", "tmpk": "k", "dwpk": "k", "tk": "k", "tdk": "k",
            "sknt": KNOT_M_S}


def _column(header: str) -> tuple[str | None, Any]:
    """(what a column is, its unit or None) from its header cell."""

    raw = header.strip().strip("\"'").lower()
    unit_text = ""
    match = re.search(r"[\(\[]([^\)\]]*)[\)\]]", raw)
    if match:
        unit_text = match.group(1)
        raw = (raw[:match.start()] + raw[match.end():]).strip()
    parts = [part for part in re.split(r"[\s_\-./]+", raw.replace("°", "deg")) if part]
    unit_key = re.sub(r"[^a-z0-9]", "", unit_text.replace("°", "deg").replace("/", "").replace("^-1", "1"))
    # The longest run of leading parts that is a known name; what is left is the unit.
    for count in range(len(parts), 0, -1):
        name = "".join(parts[:count])
        kind = next((kind for kind, names in _NAMES.items() if name in names), None)
        if kind is None:
            continue
        rest = "".join(parts[count:])
        unit_key = unit_key or rest
        family = "t" if kind in ("t", "td") else "wind" if kind in ("spd", "u", "v") else kind
        unit = _UNITS[family].get(unit_key) if unit_key else None
        if unit is None and name in _IMPLIED:
            unit = _IMPLIED[name]
        if unit_key and unit is None:
            return kind, ("unknown", unit_key)
        return kind, unit
    return None, None


def _split(line: str, delimiter: str | None) -> list[str]:
    cells = line.split(delimiter) if delimiter else line.split()
    return [cell.strip().strip("\"'").strip() for cell in cells]


def _unit_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("°", "deg").replace("/", "").replace("^-1", "1"))


def _units_row(cells: Sequence[str], kinds: Mapping[str, int], units: dict[str, Any]) -> list[str]:
    """Units from a second header line (``hPa,m,C,C,deg,kt``) for the columns whose name gave none."""

    read: list[str] = []
    for kind, index in kinds.items():
        if units.get(kind) is not None or index >= len(cells) or not cells[index]:
            continue
        family = "t" if kind in ("t", "td") else "wind" if kind in ("spd", "u", "v") else kind
        unit = _UNITS[family].get(_unit_key(cells[index]))
        if unit is not None:
            units[kind] = unit
            read.append(cells[index])
    return read


def _table_levels(text: str, speed_unit: str | None) -> tuple[list[_Level], list[str]]:
    lines = _plain(text).split("\n")
    head = next(index for index, line in enumerate(lines)
                if line.strip() and not line.lstrip().startswith(("#", "!")))
    header = lines[head]
    delimiter = next((mark for mark in (",", ";", "\t") if mark in header), None)
    cells = _split(header, delimiter)
    kinds: dict[str, int] = {}
    units: dict[str, Any] = {}
    for index, cell in enumerate(cells):
        kind, unit = _column(cell)
        if kind is None or kind in kinds:
            continue
        if isinstance(unit, tuple):
            raise SoundingRefused(
                "unit", f"The column {cell.strip()!r} names a unit GM1 does not know ({unit[1]}).",
                "Use hPa or Pa for pressure, m, km or ft for height, C or K for temperature, and kt or m/s for "
                "wind speed.", line=head + 1)
        kinds[kind] = index
        units[kind] = unit
    wanted = {"p": "pressure", "z": "height", "t": "temperature", "td": "dewpoint"}
    absent = [words for kind, words in wanted.items() if kind not in kinds]
    has_polar = "dir" in kinds and "spd" in kinds
    has_uv = "u" in kinds and "v" in kinds
    if not (has_polar or has_uv):
        absent.append("the wind (direction and speed, or u and v)")
    if absent:
        raise SoundingRefused(
            "columns",
            "The header line does not name " + _join(absent) + f". It reads: {header.strip()[:120]}",
            "Name the columns in the first line, for example: pressure,height,temperature,dewpoint,direction,"
            "speed_kt. " + ACCEPTED + ".", line=head + 1)
    said: list[str] = []
    second = next((index for index in range(head + 1, len(lines)) if lines[index].strip()
                   and not lines[index].lstrip().startswith(("#", "!"))), None)
    if second is not None:
        cells_below = _split(lines[second], delimiter)
        if not any(_is_number(cell) for cell in cells_below):
            read = _units_row(cells_below, kinds, units)
            if read:
                said.append(f"Units read from line {second + 1}: " + ", ".join(read) + ".")
    wind_kinds = ("spd",) if has_polar else ("u", "v")
    if not has_polar and units.get("u") != units.get("v"):
        named = {kind: units.get(kind) for kind in ("u", "v")}
        raise SoundingRefused(
            "unit", "The u and v columns do not say the same unit"
            + ("" if all(named.values()) else " (one of them says none)")
            + ", and GM1 does not guess: both wind components must be in one unit.",
            "Name both columns with the same unit, for example u_kt and v_kt, or u_ms and v_ms.", line=head + 1)
    factor = units.get(wind_kinds[0])
    if factor is None:
        if speed_unit is None:
            raise SoundingRefused(
                "speed-unit",
                "The wind speed column does not say whether it is in knots or metres per second, and GM1 does not "
                "guess: the wrong one would double or halve every wind.",
                "Say which it is, or name the column with its unit (speed_kt or speed_ms).", line=head + 1,
                offer={"speed_unit": ["kt", "ms"]})
        if speed_unit not in ("kt", "ms"):
            raise SoundingRefused("speed-unit", f"The wind speed unit is kt or ms, not {speed_unit!r}.",
                                  "Use kt (knots) or ms (metres per second).")
        factor = KNOT_M_S if speed_unit == "kt" else 1.0
        said.append("Wind speed read as " + ("knots (1 knot = 0.514444 m/s)" if speed_unit == "kt" else "m/s")
                    + ", as you said.")
    else:
        said.append("Wind speed read as " + ("m/s" if factor == 1.0 else f"its column's unit ({factor:g} m/s each)")
                    + ", from the column name.")
    rows: list[tuple[int, list[float | None]]] = []
    order = ["p", "z", "t", "td", *(("dir", "spd") if has_polar else ("u", "v"))]
    for index in range(head + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!")) or set(stripped) <= {"-", "=", " "}:
            continue
        tokens = _split(line, delimiter)
        if not rows and not any(_is_number(token) for token in tokens):
            continue  # a second header line of units
        if len(tokens) <= max(kinds[kind] for kind in order):
            raise SoundingRefused(
                "format", f"Line {index + 1} has {len(tokens)} value(s), fewer than the header's columns.",
                "Give every line as many values as the header has names (leave a missing value empty or write "
                "-9999).", line=index + 1)
        try:
            values = [_value(tokens[kinds[kind]]) for kind in order]
            rows.append((index + 1, [*values[:2], *(_unmarked(value) for value in values[2:])]))
        except ValueError as error:
            raise SoundingRefused("format", f"Line {index + 1} has {str(error)!r}, which is not a number.",
                                  "Leave a missing value empty or write -9999.", line=index + 1) from None

    def column(position: int) -> list[float]:
        return [values[position] for _, values in rows if values[position] is not None]

    def typical(position: int) -> float | None:
        values = sorted(column(position))
        return values[len(values) // 2] if values else None

    # Pressure in Pa and temperature in K are told from hPa and C by their size: the middle value, so one odd
    # cell (a missing-value mark such as 9999) does not decide it for the whole column.
    p_factor = units.get("p")
    if p_factor is None:
        p_factor = 0.01 if (typical(0) or 0.0) > 2000.0 else 1.0
        if p_factor != 1.0:
            said.append("Pressure read as Pa (its values are above 2000), divided by 100 for hPa.")
    z_factor = units.get("z") or 1.0
    if z_factor != 1.0:
        said.append(f"Height multiplied by {z_factor:g} for metres, from the column name.")

    def scale_of(kind: str, position: int) -> str:
        scale = units.get(kind)
        if scale is None:
            scale = "k" if (typical(position) or 0.0) > 150.0 else "c"
            if scale == "k":
                said.append(("Temperature" if kind == "t" else "Dewpoint")
                            + " read as K (its values are above 150), less 273.15 for C.")
        return scale

    scales = (scale_of("t", 2), scale_of("td", 3))

    def celsius(value: float | None, scale: str) -> float | None:
        if value is None:
            return None
        return value - 273.15 if scale == "k" else (value - 32.0) / 1.8 if scale == "f" else value

    levels = []
    for line_number, values in rows:
        level = _Level(line_number,
                       None if values[0] is None else values[0] * p_factor,
                       None if values[1] is None else values[1] * z_factor,
                       celsius(values[2], scales[0]), celsius(values[3], scales[1]))
        if has_polar:
            level.direction = values[4]
            level.speed_ms = None if values[5] is None else values[5] * factor
        else:
            level.u_ms = None if values[4] is None else values[4] * factor
            level.v_ms = None if values[5] is None else values[5] * factor
        levels.append(level)
    return levels, said


def _join(words: Sequence[str]) -> str:
    words = list(words)
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


def _missing(level: _Level) -> list[str]:
    return list(dict.fromkeys(words for words, value in (
        ("pressure", level.p_hpa), ("height", level.z_m), ("temperature", level.t_c), ("dewpoint", level.td_c),
        ("wind", level.speed_ms if level.u_ms is None else level.u_ms),
        ("wind", level.direction if level.u_ms is None else level.v_ms)) if value is None))


def convert(levels: Sequence[_Level]) -> tuple[Profile, list[dict[str, Any]], list[str], float, list[str]]:
    """(the CM1 profile, the levels left out, notes, the station height, warnings) from measured levels.

    The levels are read in order of pressure, from the ground up (a file listed from the top down, or with a
    level out of place, is put in order and the note says so).  The ground is the lowest level with a
    temperature: lines below it with none are the levels a listing gives below the station's ground.  The
    ground's own wind may be missing (CM1 does not read a surface wind); when its dewpoint is, the next whole
    level becomes the ground and a warning says how much higher it is."""

    notes: list[str] = []
    warnings: list[str] = []
    levels = list(levels)
    whole = [level for level in levels if level.p_hpa is not None]
    if len(whole) >= 2 and whole[0].p_hpa < whole[-1].p_hpa:
        levels.reverse()
        notes.append("The file lists the top of the sounding first; its levels were read from the bottom up.")
        whole.reverse()
    out_of_place = [later.line for earlier, later in zip(whole, whole[1:]) if later.p_hpa > earlier.p_hpa]
    ordered = sorted(whole, key=lambda level: -level.p_hpa)
    heights = [level.z_m for level in ordered if level.z_m is not None]
    if out_of_place and all(b > a for a, b in zip(heights, heights[1:])):
        # Only when pressure and height then agree (levels listed out of place, not a mistyped pressure).
        # Stable, so levels at one pressure keep their order; a level with no pressure is left out below.
        levels = ordered + [level for level in levels if level.p_hpa is None]
        notes.append(f"{len(out_of_place)} level(s) were out of order (first at line {out_of_place[0]}); GM1 put "
                     "the levels in order of pressure, from the ground up.")
    ground = next((index for index, level in enumerate(levels)
                   if level.p_hpa is not None and level.z_m is not None and level.t_c is not None), None)
    used: list[_Level] = []
    dropped: list[dict[str, Any]] = []
    first_without: _Level | None = None
    last = -1.0e9
    for index, level in enumerate(levels):
        missing = _missing(level)
        if not used and ground is not None and index < ground and level.t_c is None and level.p_hpa is not None:
            dropped.append({"line": level.line, "why": f"below the ground ({level.p_hpa:g} hPa): it has no "
                                                       "temperature, as a listing gives the levels under the "
                                                       "station"})
            continue
        if not used and missing == ["wind"]:
            missing = []                      # the ground's wind: CM1's layout has none
        if missing:
            dropped.append({"line": level.line, "why": "no " + " or ".join(missing)})
            if not used and first_without is None and level.t_c is not None:
                first_without = level
            continue
        if level.z_m <= last:
            dropped.append({"line": level.line, "why": f"its height ({level.z_m:g} m) is not above the level "
                                                       "below it"})
            continue
        if not 1.0 <= level.p_hpa <= 1100.0:
            raise SoundingRefused(
                "pressure-unit", f"Line {level.line} has a pressure of {level.p_hpa:g} hPa, which is not a pressure "
                "in the air.", "Give pressure in hPa (the same as mb), or name the column with its unit.",
                line=level.line)
        if not -150.0 <= level.t_c <= 70.0 or not -200.0 <= level.td_c <= 70.0:
            raise SoundingRefused(
                "temperature-unit",
                f"Line {level.line} has a temperature of {level.t_c:g} C and a dewpoint of {level.td_c:g} C, which "
                "air does not have.", "Give temperature and dewpoint in C or in K, or name the columns with "
                "their unit.", line=level.line)
        if level.u_ms is None and level.direction is not None and level.speed_ms is not None:
            if not -0.5 <= level.direction <= 360.5 or level.speed_ms < 0.0:
                raise SoundingRefused(
                    "wind", f"Line {level.line} has a wind direction of {level.direction:g} and a speed of "
                    f"{level.speed_ms:g}; a direction is 0 to 360 degrees and a speed is not negative.",
                    "Give the direction the wind blows from, in degrees, and its speed.", line=level.line)
        used.append(level)
        last = level.z_m
    if len(used) < 3:
        raise SoundingRefused(
            "levels",
            f"Only {len(used)} level(s) have all of pressure, height, temperature, dewpoint and wind"
            + (f" ({len(dropped)} were left out, the first at line {dropped[0]['line']}: {dropped[0]['why']})"
               if dropped else "") + ". CM1 needs the surface and at least two levels above it.",
            "Use a sounding with more complete levels.")
    if len(dropped) > max(3, len(levels) // 2):
        notes.append(f"{len(dropped)} of {len(levels)} levels were left out (missing values or repeated heights).")
    if first_without is not None and first_without.line != used[0].line:
        gap = (f", {used[0].z_m - first_without.z_m:.0f} m higher" if first_without.z_m is not None else "")
        warnings.append(
            f"Line {first_without.line}, the lowest level with a temperature, has {_join(_missing(first_without))} "
            f"missing, so GM1 took line {used[0].line} ({used[0].p_hpa:g} hPa{gap}) as the ground. The air below "
            "it is not in the run.")
    station = used[0].z_m
    z, th, qv, u, v, lines = [], [], [], [], [], []
    for level in used:
        p_pa = level.p_hpa * 100.0
        t_k = level.t_c + 273.15
        z.append(level.z_m - station)
        th.append(t_k * (P00 / p_pa) ** ROVCP)
        qv.append(rslf(p_pa, level.td_c + 273.15) * 1000.0)
        if level.u_ms is None and (level.direction is None or level.speed_ms is None):
            u.append(0.0)                     # the ground's own wind, which CM1's layout does not hold
            v.append(0.0)
        elif level.u_ms is None:
            angle = math.radians(level.direction)
            u.append(-level.speed_ms * math.sin(angle))
            v.append(-level.speed_ms * math.cos(angle))
        else:
            u.append(level.u_ms)
            v.append(level.v_ms)
        lines.append(level.line)
    profile = Profile(used[0].p_hpa, th[0], qv[0], tuple(z[1:]), tuple(th[1:]), tuple(qv[1:]), tuple(u[1:]),
                      tuple(v[1:]), surface_line=lines[0], lines=tuple(lines[1:]))
    return profile, dropped, notes, station, warnings


CONVERSIONS = (
    "Height above the ground = height - the first level's height ({station:g} m), which is the surface line.",
    "Potential temperature (K) = T x (1000 hPa / p) ^ 0.2854, with T in K (CM1's Rd / cp).",
    "Water-vapour mixing ratio (g/kg) = the saturation mixing ratio at the dewpoint (CM1's rslf, Bolton's formula).",
    "Wind: u = -speed x sin(direction), v = -speed x cos(direction), in m/s; the direction is where the wind "
    "blows from.",
    "CM1's layout has no surface wind: CM1 works it out from the first two levels above the ground.",
)


def read(text: str, *, speed_unit: str | None = None) -> Reading:
    """``text`` as a sounding: its layout found, its levels converted when it is not CM1's own."""

    if len(text.encode("utf-8", "replace")) > MAX_BYTES:
        raise SoundingRefused("size", "The file is too large to be a sounding (over 4 MB).",
                              "Choose the sounding's own text file.")
    if "\x00" in text:
        raise SoundingRefused("format", "The file is not text.", "Choose a plain text file. " + ACCEPTED + ".")
    layout = detect(text)
    if layout == "cm1":
        profile = parse_cm1(text)
        return Reading("cm1", profile, text, False)
    levels, said = {"wyoming": lambda: _wyoming_levels(text), "sharppy": lambda: _sharppy_levels(text),
                    "table": lambda: _table_levels(text, speed_unit)}[layout]()
    if not levels:
        raise SoundingRefused("levels", "No level lines were found under the header.",
                              "Choose a file that lists the sounding's levels.")
    profile, dropped, notes, station, warnings = convert(levels)
    several = _several(text, layout)
    if several:
        notes.append(several)
    uses_uv = any(level.u_ms is not None for level in levels)
    conversions = [line.format(station=station) for line in CONVERSIONS
                   if not (uses_uv and line.startswith("Wind: u ="))]
    if uses_uv:
        conversions.insert(3, "Wind: the file's own u (toward the east) and v (toward the north), in m/s.")
    return Reading(layout, profile, render(profile), True, tuple([*conversions, *said]), tuple(dropped),
                   tuple(notes), station, tuple(warnings))


# ------------------------------------------------------------------ checking


def check(profile: Profile, *, top_needed_m: float | None = None) -> tuple[list[Finding], list[Finding]]:
    """(problems, warnings) for a guided storm on ``profile``; lines are the user's own file's.

    A warning whose ``level`` is ``"note"`` is only worth a plain sentence (the warm layer near the ground that
    every afternoon balloon sounding has)."""

    problems: list[Finding] = []
    warnings: list[Finding] = []
    #: Plain notes (level "note"), returned with the warnings.
    notes = warnings

    def problem(code: str, text: str, fix: str, line: int | None = None, offer: Mapping[str, Any] | None = None):
        problems.append(Finding("problem", code, text, fix, line, dict(offer or {})))

    def warning(code: str, text: str, fix: str = "", line: int | None = None):
        warnings.append(Finding("warning", code, text, fix, line))

    z = profile.z_m
    if profile.levels < 2:
        problem("levels",
                f"The file has {profile.levels} level(s) above the surface line. CM1 needs at least two, and a "
                "storm needs the air up to the top of the model.",
                "Use a sounding with more levels.")
    if not 300.0 <= profile.surface_hpa <= 1100.0:
        looks = " It looks like Pa: divide it by 100." if 30000.0 <= profile.surface_hpa <= 110000.0 else ""
        problem("pressure-unit",
                f"The surface pressure on line {profile.surface_line} is {profile.surface_hpa:g}. CM1 reads it in "
                f"hPa (the same as mb), where the ground is between 300 and 1100.{looks}",
                "Give the surface pressure in hPa.", profile.surface_line)
    thetas = [(profile.surface_line, profile.surface_theta_k), *zip(profile.lines or [None] * profile.levels,
                                                                     profile.theta_k)]
    cold = next(((line, value) for line, value in thetas if value < 150.0), None)
    if cold is not None:
        celsius = -100.0 <= cold[1] <= 70.0
        problem("temperature-unit",
                f"Line {cold[0]} has {cold[1]:g} where the potential temperature in K belongs."
                + (" That looks like a temperature in C." if celsius else "")
                + " Potential temperature is about 300 K near the ground and rises with height.",
                "Give potential temperature in K: T x (1000 / p) ^ 0.2854, with T in K and p in hPa. Or give the "
                "file as a table of pressure, height, temperature and dewpoint, which GM1 converts.", cold[0])
    elif any(value > 5000.0 for _, value in thetas):
        line, value = next((line, value) for line, value in thetas if value > 5000.0)
        problem("temperature-unit", f"Line {line} has {value:g} where the potential temperature in K belongs.",
                "Give potential temperature in K.", line)
    elif profile.levels >= 2 and profile.top_m >= 5000.0 and profile.theta_k[-1] < profile.surface_theta_k - 15.0:
        problem("temperature-unit",
                f"The second column falls from {profile.surface_theta_k:g} K at the ground to "
                f"{profile.theta_k[-1]:g} K at {profile.top_m / 1000.0:.1f} km. Potential temperature rises with "
                "height, so this looks like plain temperature in K.",
                "Give potential temperature: T x (1000 / p) ^ 0.2854, with T in K and p in hPa. Or give the file "
                "as a table of pressure, height, temperature and dewpoint, which GM1 converts.",
                profile.line_of(profile.levels - 1))
    vapours = [(profile.surface_line, profile.surface_qv_gkg), *zip(profile.lines or [None] * profile.levels,
                                                                     profile.qv_gkg)]
    negative = next(((line, value) for line, value in vapours if value < 0.0), None)
    wet = next(((line, value) for line, value in vapours if value > 45.0), None)
    if negative is not None:
        problem("moisture", f"Line {negative[0]} has a mixing ratio of {negative[1]:g} g/kg. It cannot be negative.",
                "Give the water-vapour mixing ratio in g/kg, 0 or more.", negative[0])
    elif wet is not None:
        problem("moisture",
                f"Line {wet[0]} has a mixing ratio of {wet[1]:g} g/kg, more water vapour than air holds anywhere on "
                "Earth. It looks like relative humidity in percent.",
                "Give the water-vapour mixing ratio in g/kg (about 5 to 20 near the ground in storm air).", wet[0])
    elif max(value for _, value in vapours) < 0.1 and any(value > 0.0 for _, value in vapours):
        warning("moisture-unit",
                "Every mixing ratio is below 0.1 g/kg. CM1 reads g/kg, so if your file is in kg/kg, multiply the "
                "column by 1000. As it is, the air is almost dry and no cloud will form.")
    if z and z[0] <= 0.0:
        problem("heights",
                f"The first level, on line {profile.line_of(0)}, is at {z[0]:g} m. The surface line is the ground "
                "(0 m), so every level line must be above it.",
                "Give heights in metres above the ground, the first one above 0.", profile.line_of(0))
    for k in range(1, profile.levels):
        if not z[k] > z[k - 1]:
            problem("heights",
                    f"The height on line {profile.line_of(k)} ({z[k]:g} m) is not above the one before it "
                    f"({z[k - 1]:g} m). CM1 needs the levels in order, from the ground up, each height once.",
                    "Put the levels in order of height and take out repeated heights.", profile.line_of(k))
            break
    if profile.levels >= 10 and 0.0 < profile.top_m < 100.0 and not any(f.code == "heights" for f in problems):
        problem("height-unit",
                f"The heights go up to only {profile.top_m:g} m over {profile.levels} levels. They look like km; "
                "CM1 reads the height above the ground in metres.",
                "Multiply the height column (the first number on each level line) by 1000.",
                profile.line_of(profile.levels - 1))
    fast = next((k for k in range(profile.levels) if max(abs(profile.u_ms[k]), abs(profile.v_ms[k])) > 150.0), None)
    if fast is not None:
        problem("wind",
                f"Line {profile.line_of(fast)} has a wind of {profile.u_ms[fast]:g}, {profile.v_ms[fast]:g} m/s. "
                "No wind is that fast; it looks like a missing-value mark or another unit.",
                "Give u and v in m/s, and take out levels with no wind.", profile.line_of(fast))
    if problems:
        return problems, warnings
    # From here the numbers are a real column: its temperature needs CM1's hydrostatic pressure.
    pressures = pressures_pa(profile)
    temperatures = temperatures_k(profile)
    theta = [profile.surface_theta_k, *profile.theta_k]
    heights = [0.0, *z]
    line_at = [profile.surface_line, *(profile.lines or [None] * profile.levels)]
    frozen = next((k for k, value in enumerate(temperatures) if value < 150.0), None)
    if frozen is not None:
        problem("temperature",
                f"At line {line_at[frozen]} ({heights[frozen] / 1000.0:.1f} km) the air works out to "
                f"{temperatures[frozen]:.0f} K ({temperatures[frozen] - 273.15:.0f} C), colder than any real air. "
                "Potential temperature that does not rise with height, far up, gives air like this.",
                "Check the potential temperature (the second number on each line) up to that line: above the "
                "tropopause it rises by several K per km.", line_at[frozen])
        return problems, warnings
    falls = [(k, theta[k - 1] - theta[k]) for k in range(1, len(theta)) if theta[k] < theta[k - 1] - 0.01]
    # A warmer layer near the ground (a sunny afternoon), or small ones aloft, are in every balloon sounding:
    # a plain note.  A large one aloft is worth a warning.
    aloft = [(k, fall) for k, fall in falls if heights[k] >= SURFACE_LAYER_M and fall > LARGE_FALL_K]
    if falls and not aloft:
        notes.append(Finding(
            "note", "surface-layer",
            f"In {len(falls)} thin layer(s) (the lowest at {heights[falls[0][0]]:g} m) the air is a little warmer "
            f"than the air just above it, by {max(fall for _, fall in falls):.1f} K at most. Near the ground this is "
            "sun-heated air on an afternoon. Real balloon soundings have such layers; the model mixes them in its "
            "first minutes. Nothing to fix."))
    if aloft:
        k, most = max(falls, key=lambda row: row[1])
        first = falls[0][0]
        warning("superadiabatic",
                f"Potential temperature falls with height in {len(falls)} layer(s), first at line {line_at[first]} "
                f"({heights[first]:g} m) and by {most:.2f} K at most (line {line_at[k]}). Such air turns over by "
                "itself, so the model will mix it in its first minutes. Sun-heated ground does this in real "
                "soundings; it runs.")
    soaked = []
    for k in range(len(theta)):
        qv = (profile.surface_qv_gkg if k == 0 else profile.qv_gkg[k - 1]) / 1000.0
        saturation = rslf(pressures[k], temperatures[k])
        if saturation > 0.0 and qv > 1.02 * saturation:
            soaked.append((k, 100.0 * qv / saturation))
    if soaked:
        k, most = max(soaked, key=lambda row: row[1])
        first = soaked[0][0]
        warning("supersaturated",
                f"The air holds more water vapour than it can at {len(soaked)} level(s), first at line "
                f"{line_at[first]} ({heights[first]:g} m), up to {most:.0f} % relative humidity (line {line_at[k]}). "
                "The model turns the extra into cloud in its first step, so the run starts with cloud there. It "
                "runs.")
    if profile.levels < 10:
        warning("coarse", f"The file has {profile.levels} levels above the ground. CM1 draws straight lines "
                          "between them, so thin layers (a lid of warm air, a shallow moist layer) are not there.")
    if z and z[0] > 1000.0:
        warning("coarse", f"The first level above the ground is at {z[0]:g} m (line {profile.line_of(0)}): "
                          "nothing is known of the air below it but the surface line.")
    if top_needed_m is not None and z and z[-1] < top_needed_m:
        need = math.ceil(top_needed_m)
        if z[-1] >= EXTEND_FROM_M:
            problem("top",
                    f"The sounding stops at {z[-1] / 1000.0:.1f} km and the model goes up to "
                    f"{top_needed_m / 1000.0:.2f} km. CM1 stops on such a file.",
                    f"Let GM1 add levels up to the model top (a made-up layer at the top level's temperature, "
                    f"not measured), or use a sounding that reaches {need} m.",
                    profile.line_of(profile.levels - 1), {"extend_to_m": extend_target(top_needed_m)})
        else:
            problem("top",
                    f"The sounding stops at {z[-1] / 1000.0:.1f} km and the model goes up to "
                    f"{top_needed_m / 1000.0:.2f} km. CM1 stops on such a file, and too little of this one is "
                    "measured to add the rest.",
                    f"Use a sounding that reaches {need} m (or at least {EXTEND_FROM_M / 1000.0:g} km, which GM1 "
                    "can extend).", profile.line_of(profile.levels - 1))
    return problems, warnings


# ------------------------------------------------------------------ extending the top


def extend_target(top_needed_m: float) -> float:
    """The height :func:`extend` goes to for a model whose highest level is ``top_needed_m``."""

    return (math.floor(top_needed_m / EXTEND_STEP_M) + 2) * EXTEND_STEP_M


def extend(profile: Profile, to_m: float) -> tuple[Profile, dict[str, Any]]:
    """``profile`` with levels every 500 m above its top, up to ``to_m``: a simple stratosphere.

    The added air has the top level's temperature at every height (as the lower stratosphere roughly does, and as
    CM1's own Weisman-Klemp sounding has above its tropopause, ``base.F``: theta = theta_top
    exp(g (z - z_top) / (cp T_top))), and the top level's mixing ratio and wind.  It is made up, not measured, and
    the record says where it begins.
    """

    if profile.levels < 1 or to_m <= profile.top_m:
        return profile, {"extended": False}
    top = profile.top_m
    t_top = temperatures_k(profile)[-1]
    th_top = profile.theta_k[-1]
    z, th = list(profile.z_m), list(profile.theta_k)
    qv, u, v = list(profile.qv_gkg), list(profile.u_ms), list(profile.v_ms)
    height = (math.floor(top / EXTEND_STEP_M) + 1) * EXTEND_STEP_M
    if height - top < 0.25 * EXTEND_STEP_M:
        height += EXTEND_STEP_M
    added = 0
    while height <= to_m + 1.0e-6:
        z.append(height)
        th.append(th_top * math.exp(G * (height - top) / (CP * t_top)))
        qv.append(qv[profile.levels - 1])
        u.append(u[profile.levels - 1])
        v.append(v[profile.levels - 1])
        added += 1
        height += EXTEND_STEP_M
    extended = replace(profile, z_m=tuple(z), theta_k=tuple(th), qv_gkg=tuple(qv), u_ms=tuple(u), v_ms=tuple(v),
                       lines=tuple([*profile.lines, *([None] * added)]) if profile.lines else (),
                       extended=profile.extended + added)
    low = t_top > TROPOPAUSE_WARM_K
    record = {
        "extended": True, "from_m": top, "to_m": z[-1], "levels_added": added, "temperature_k": t_top,
        "likely_below_tropopause": low,
        "sentence": (
            f"Levels were added from {top / 1000.0:.1f} km to {z[-1] / 1000.0:.1f} km: air at the top level's "
            f"temperature ({t_top - 273.15:.0f} C) with its wind and moisture. That part is made up, not measured."
            + (f" Your sounding stops where it is still {t_top - 273.15:.0f} C, probably below the real tropopause, "
               f"so the made-up layer acts as a lid at {top / 1000.0:.1f} km and storm tops will be lower than in "
               "the real air." if low else "")),
    }
    return extended, record


__all__ = [
    "ACCEPTED", "BUBBLE_CENTRE_M", "CONVERSIONS", "EXTEND_FROM_M", "FORMATS", "FORMAT_WORDS", "Finding",
    "MAX_BYTES", "Profile", "Reading", "SoundingRefused", "bubble_moisture", "check", "convert", "detect",
    "extend", "extend_target", "interpolate", "parse_cm1", "pressures_pa", "read", "render", "rslf",
    "temperatures_k",
]
