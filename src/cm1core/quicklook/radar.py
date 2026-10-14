"""Radar velocity: one low scan of a simulated Doppler radar through a saved frame.

What the two radar looks show (:func:`product_radar_storm_relative`, :func:`product_radar_velocity`):

* the model's wind along a simulated radar beam, positive away from the radar (radial velocity), either as it is
  (ground-relative) or with the storm's motion taken out (storm-relative), beside the reflectivity the same beam
  sees;
* one scan at a low tilt (0.5 degrees unless set), from a radar site placed 40 km south-east of the storm for each
  picture (120 km from a hurricane's eye), or fixed on the ground where the user puts it (:class:`Settings`,
  ``quicklook/radar.json``, written by ``gpuwm-cm1 quicklook RUNDIR --radar-site X Y``).  The storm is followed
  from one saved time to the next (:func:`find_storm` with the :class:`Track` of the times before, kept by
  :class:`StormTrack`), so the radar and the panels move with it and a film does not jump between its parts.

How (each step is said on the picture or in its notes):

* beam path: the 4/3 effective-earth model (Doviak and Zrnic 1993, section 2.2).  Beam height
  ``h = sqrt(r^2 + (ke a)^2 + 2 r ke a sin e) - ke a``, ground range ``s = ke a asin(r cos e / (ke a + h))`` and
  the beam's angle above the local ground at the gate ``e + s / (ke a)``; the antenna is 20 m above the ground;
* beam width: a two-way Gaussian power pattern of half-power width 0.95 degrees, integrated with three
  Gauss-Hermite points in elevation and three in azimuth, and three points along each range gate;
* radial velocity at each sample: ``cos(e_loc) (u sin az + v cos az) + w sin(e_loc)`` with the ground-relative wind
  (``u + umove``, ``v + vmove``: CM1 init3d.F:32-33) interpolated trilinearly from the C grid (u on x faces, v on
  y faces, w on the w levels, any vertical spacing); no fall speed of the rain or hail is added;
* reflectivity: the quick-looks' own (:mod:`.reflectivity`: the scheme's own dBZ, or CM1's ``calc_refl10cm`` for
  Morrison), turned into Z (mm^6 m^-3) before it is interpolated and averaged over the beam, and the velocity is
  averaged with the same Z as weight, as a radar's is.  A scheme without a reflectivity operator in GM1
  (Thompson) shows the rain, snow and graupel or hail mass on the beam in g/kg instead, labelled as not
  reflectivity, and weights the velocity with it;
* the velocity is drawn only where the echo is 5 dBZ (0.01 g/kg) or more, since a radar sees wind only where
  something reflects;
* storm-relative: the storm's motion projected on the beam is taken out.  The motion is the one the user set, else
  the domain's own (CM1's ``umove``, ``vmove``, which the run set to follow the storm), else the Bunkers right
  mover of the run's starting air (the run's own number, as the environment look shows it), else the same formula
  (:func:`air_motion`) on the first saved time's undisturbed corner air.  Where the starting air's 0-6 km shear is
  under :data:`RIGHT_MOVER_MIN_SHEAR` the storms are single cells that move with the wind around them, and a right
  mover means nothing, so the 0-6 km mean wind is taken out instead, and the picture says so.

What it is not: no attenuation, no ground clutter, no beam blocking, no noise, no range or velocity folding, no
fall speed.  Every picture says so.

The scan is computed on the radar's own polar gates (0.5 degree radials, 250 m gates; 1 degree and 1 km for a
hurricane's scan of hundreds of kilometres), only over the area the panels show, and each pixel shows the gate
it falls in, as a radar display does.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import products as P
from . import style as S

EARTH_RADIUS_M = 6371000.0
EFFECTIVE_EARTH_FACTOR = 4.0 / 3.0
ANTENNA_M = 20.0
SETTINGS_FILE = "radar.json"
SETTINGS_SCHEMA = "gm1.quicklook-radar/v1"
STORM_RELATIVE = "radar-storm-relative"
GROUND_RELATIVE = "radar-velocity"
LOOK_IDS = (STORM_RELATIVE, GROUND_RELATIVE)
#: The fixed, stated velocity range of both looks (m/s); a colour means the same speed on every picture.
VELOCITY_RANGE = 70.0
DBZ_THRESHOLD = 5.0          # dBZ: a gate with less echo shows no velocity
MASS_THRESHOLD = 0.01        # g/kg, the same for a run with no reflectivity operator
PRECIPITATION = ("qr", "qs", "qg", "qh")
#: The words every radar picture carries.
WHAT_IT_IS = "model wind along a simulated beam; no attenuation, clutter or velocity folding"
#: The couplet test (from the virtual radar tool it was ported from): each side at least this fast and
#: (out - in) / 2 at least this, with the outbound side clockwise of the inbound one (cyclonic).
COUPLET_MIN_SIDE = 5.0
COUPLET_MIN_ROTATION = 10.0
COMPASS16 = ("north", "north-north-east", "north-east", "east-north-east", "east", "east-south-east", "south-east",
             "south-south-east", "south", "south-south-west", "south-west", "west-south-west", "west",
             "west-north-west", "north-west", "north-north-west")


class SettingsError(ValueError):
    """Radar settings that cannot be used, in plain words."""


# ----------------------------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Settings:
    """A run's radar settings (``quicklook/radar.json``).

    ``site_km``: the radar's place on the ground, in the run's own x, y (km) at the start (T+0), when the domain
    moves the radar stays on the ground; None places it beside the storm in each picture.  ``tilt_deg``: the
    scan's elevation angle.  ``motion_m_s``: the storm motion (ground-relative) the storm-relative look takes out;
    None uses the run's own (the domain's motion, else the Bunkers right mover of its starting air, or its 0-6 km
    mean wind where the shear is too weak for a right mover)."""

    site_km: tuple[float, float] | None = None
    tilt_deg: float = 0.5
    motion_m_s: tuple[float, float] | None = None

    def check(self) -> "Settings":
        if not 0.1 <= float(self.tilt_deg) <= 20.0:
            raise SettingsError("The radar tilt must be from 0.1 to 20 degrees; radars scan storms from 0.5.")
        if self.site_km is not None:
            if len(self.site_km) != 2 or not all(math.isfinite(float(v)) and abs(float(v)) <= 5000.0
                                                 for v in self.site_km):
                raise SettingsError("The radar site must be two numbers, x and y in km, each within 5000 km.")
        if self.motion_m_s is not None:
            if len(self.motion_m_s) != 2 or not all(math.isfinite(float(v)) and abs(float(v)) <= 100.0
                                                    for v in self.motion_m_s):
                raise SettingsError("The storm motion must be two numbers, east and north in m/s, each within "
                                    "100 m/s.")
        return self

    def record(self) -> dict[str, Any]:
        return {"schema": SETTINGS_SCHEMA,
                "site_km": None if self.site_km is None else [float(v) for v in self.site_km],
                "tilt_deg": float(self.tilt_deg),
                "motion_m_s": None if self.motion_m_s is None else [float(v) for v in self.motion_m_s]}


def read_settings(folder: Path | str) -> Settings:
    """The settings in ``folder/radar.json`` (a run's quicklook folder); the defaults when there are none."""

    try:
        doc = json.loads((Path(folder) / SETTINGS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    if not isinstance(doc, dict) or doc.get("schema") != SETTINGS_SCHEMA:
        return Settings()

    def pair(value: Any) -> tuple[float, float] | None:
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return float(value[0]), float(value[1])
        return None

    tilt = doc.get("tilt_deg")
    try:
        return Settings(pair(doc.get("site_km")),
                        float(tilt) if isinstance(tilt, (int, float)) and not isinstance(tilt, bool) else 0.5,
                        pair(doc.get("motion_m_s"))).check()
    except SettingsError:
        return Settings()


def write_settings(folder: Path | str, settings: Settings) -> Path:
    settings.check()
    path = Path(folder) / SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(settings.record(), indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------------------------- beam geometry
def effective_radius_m(ke: float = EFFECTIVE_EARTH_FACTOR) -> float:
    return ke * EARTH_RADIUS_M


def beam_height_m(slant_m: Any, elevation_deg: Any, *, ke: float = EFFECTIVE_EARTH_FACTOR) -> np.ndarray:
    """Height of the beam centre above the antenna (Doviak and Zrnic 1993, eq. 2.28b)."""

    r = np.asarray(slant_m, dtype=np.float64)
    e = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    big = effective_radius_m(ke)
    return np.sqrt(r * r + big * big + 2.0 * r * big * np.sin(e)) - big


def ground_range_m(slant_m: Any, elevation_deg: Any, *, ke: float = EFFECTIVE_EARTH_FACTOR) -> np.ndarray:
    """Distance along the ground under the beam (Doviak and Zrnic 1993, eq. 2.28c)."""

    r = np.asarray(slant_m, dtype=np.float64)
    e = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    big = effective_radius_m(ke)
    return big * np.arcsin(r * np.cos(e) / (big + beam_height_m(r, elevation_deg, ke=ke)))


def slant_range_m(ground_m: Any, elevation_deg: Any, *, ke: float = EFFECTIVE_EARTH_FACTOR) -> np.ndarray:
    """The slant range at which the beam is over ground range ``ground_m``: the inverse of :func:`ground_range_m`.

    In the triangle of the earth's centre, the antenna and the gate, the angle at the antenna is 90 degrees plus
    the elevation and the angle at the centre is ``s / (ke a)``, so the law of sines gives
    ``r = ke a sin(s / ke a) / cos(e + s / ke a)``."""

    big = effective_radius_m(ke)
    theta = np.asarray(ground_m, dtype=np.float64) / big
    e = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    return big * np.sin(theta) / np.cos(e + theta)


def local_elevation_rad(elevation_deg: Any, ground_m: Any, *, ke: float = EFFECTIVE_EARTH_FACTOR) -> np.ndarray:
    """The beam's angle above the local ground at the gate: the elevation plus the angle the earth has turned."""

    return (np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
            + np.asarray(ground_m, dtype=np.float64) / effective_radius_m(ke))


def radial_velocity(u: Any, v: Any, w: Any, azimuth_rad: Any, elevation_rad: Any) -> np.ndarray:
    """The wind along the beam, positive away from the radar; azimuth clockwise from north, u east, v north."""

    return (np.cos(elevation_rad) * (np.asarray(u) * np.sin(azimuth_rad) + np.asarray(v) * np.cos(azimuth_rad))
            + np.sin(elevation_rad) * np.asarray(w))


def beam_quadrature(beamwidth_deg: float, points: int) -> tuple[np.ndarray, np.ndarray]:
    """Angle offsets (degrees) and weights for a two-way Gaussian beam of half-power width ``beamwidth_deg``.

    The one-way power is ``exp(-4 ln2 t^2 / b^2)``; what comes back is its square, a Gaussian of standard
    deviation ``b / sqrt(16 ln 2)``, integrated by probabilists' Gauss-Hermite points."""

    if beamwidth_deg <= 0.0 or points <= 1:
        return np.zeros(1), np.ones(1)
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(points))
    sigma = beamwidth_deg / math.sqrt(16.0 * math.log(2.0))
    return nodes * sigma, weights / weights.sum()


def bearing_deg(dx: Any, dy: Any) -> np.ndarray:
    """Compass bearing (degrees clockwise from north) of a displacement east ``dx``, north ``dy``."""

    return np.degrees(np.arctan2(dx, dy)) % 360.0


def compass(bearing: float) -> str:
    return COMPASS16[int(round((bearing % 360.0) / 22.5)) % 16]


@dataclass(frozen=True)
class Geometry:
    """The radar: NEXRAD-like by default (0.95 degree beam, 250 m gates, 0.5 degree radials)."""

    tilt_deg: float = 0.5
    beamwidth_deg: float = 0.95
    gate_m: float = 250.0
    radial_deg: float = 0.5
    max_range_m: float = 230000.0
    beam_points: int = 3
    range_points: int = 3

    @property
    def words(self) -> str:
        return f"{self.beamwidth_deg:g}° wide, {S.fmt_km(self.gate_m)} gates, {self.radial_deg:g}° radials"


# ----------------------------------------------------------------------------------------------- sampling
def _axis(f: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.clip(f, 0.0, max(n - 1, 0))
    i0 = np.minimum(np.floor(f).astype(np.int64), max(n - 2, 0))
    i1 = np.minimum(i0 + 1, n - 1)
    t = (f - i0) if n > 1 else np.zeros_like(f)
    return i0, i1, t.astype(np.float32)


def trilinear(volume: np.ndarray, kf: np.ndarray, jf: np.ndarray, if_: np.ndarray) -> np.ndarray:
    """Trilinear samples of a ``(nk, ny, nx)`` array at fractional indices (clamped at the edges)."""

    nk, ny, nx = volume.shape
    flat = volume.reshape(-1)
    k0, k1, tk = _axis(kf, nk)
    j0, j1, tj = _axis(jf, ny)
    i0, i1, ti = _axis(if_, nx)
    sk, sj = ny * nx, nx

    def g(k: np.ndarray, j: np.ndarray, i: np.ndarray) -> np.ndarray:
        return flat[k * sk + j * sj + i]

    c0 = (g(k0, j0, i0) * (1 - ti) + g(k0, j0, i1) * ti) * (1 - tj) + (g(k0, j1, i0) * (1 - ti) + g(k0, j1, i1) * ti) * tj
    c1 = (g(k1, j0, i0) * (1 - ti) + g(k1, j0, i1) * ti) * (1 - tj) + (g(k1, j1, i0) * (1 - ti) + g(k1, j1, i1) * ti) * tj
    return c0 * (1 - tk) + c1 * tk


@dataclass
class Beamable:
    """The frame's fields a beam samples, over the levels the scan reaches.

    ``u``, ``v``, ``w`` as the frame holds them (relative to the domain, on the C grid); ``signal``: linear Z
    (mm^6 m^-3) or precipitation mass (g/kg) at scalar points.  Positions are on the ground, in metres from the
    domain's south-west corner at the start: a point is at ``x - domainlocx`` in the frame's own grid."""

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    signal: np.ndarray
    zh: np.ndarray
    zf: np.ndarray
    dx: float
    dy: float
    umove: float = 0.0
    vmove: float = 0.0
    domainlocx: float = 0.0
    domainlocy: float = 0.0
    signal_kind: str = "dbz"      # "dbz" (signal is linear Z) or "mass" (g/kg)

    @property
    def nx(self) -> int:
        return int(self.signal.shape[2])

    @property
    def ny(self) -> int:
        return int(self.signal.shape[1])

    @property
    def top_m(self) -> float:
        return float(self.zh[self.signal.shape[0] - 1])

    def sample(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, azimuth: np.ndarray,
               elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(signal, radial velocity) at ground points ``x``, ``y`` (m) and heights ``z`` above ground (m), NaN off
        the model's area or below the ground.  The velocity is ground-relative."""

        xd = np.asarray(x, np.float64) - self.domainlocx
        yd = np.asarray(y, np.float64) - self.domainlocy
        z = np.asarray(z, np.float64)
        inside = ((xd >= 0.0) & (xd <= self.nx * self.dx) & (yd >= 0.0) & (yd <= self.ny * self.dy)
                  & (z >= 0.0))
        sig = np.full(xd.shape, np.nan)
        vr = np.full(xd.shape, np.nan)
        if not inside.any():
            return sig, vr
        xd, yd, z = xd[inside], yd[inside], z[inside]
        nk = self.signal.shape[0]
        kh = np.interp(z, self.zh[:nk], np.arange(nk, dtype=np.float64))
        kf = np.interp(z, self.zf[:self.w.shape[0]], np.arange(self.w.shape[0], dtype=np.float64))
        fi = xd / self.dx - 0.5
        fj = yd / self.dy - 0.5
        u = trilinear(self.u, kh, fj, xd / self.dx) + np.float32(self.umove)
        v = trilinear(self.v, kh, yd / self.dy, fi) + np.float32(self.vmove)
        w = trilinear(self.w, kf, fj, fi)
        sig[inside] = trilinear(self.signal, kh, fj, fi)
        vr[inside] = radial_velocity(u, v, w, np.asarray(azimuth, np.float64)[inside],
                                     np.asarray(elevation, np.float64)[inside])
        return sig, vr


@dataclass
class Scan:
    """One scan on the radar's polar gates: ``[radial, gate]`` arrays."""

    site: tuple[float, float]        # ground metres
    geometry: Geometry
    az0_deg: float                   # the first radial's left edge
    r0_m: float                      # the first gate's near edge (slant range)
    signal: np.ndarray               # dBZ (or g/kg); NaN where nothing is seen
    velocity: np.ndarray             # ground-relative radial velocity where the echo is seen, else NaN
    elevation_rad: np.ndarray        # the beam's angle above the local ground at each gate centre
    signal_kind: str
    full_circle: bool

    @property
    def az_centres_deg(self) -> np.ndarray:
        return (self.az0_deg + (np.arange(self.signal.shape[0]) + 0.5) * self.geometry.radial_deg) % 360.0

    @property
    def r_centres_m(self) -> np.ndarray:
        return self.r0_m + (np.arange(self.signal.shape[1]) + 0.5) * self.geometry.gate_m

    def gate_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Ground positions (m) of the gate centres."""

        s = ground_range_m(self.r_centres_m, self.geometry.tilt_deg)[None, :]
        az = np.deg2rad(self.az_centres_deg)[:, None]
        return self.site[0] + s * np.sin(az), self.site[1] + s * np.cos(az)

    def storm_relative(self, motion: tuple[float, float]) -> np.ndarray:
        """The velocity with the storm motion (ground-relative, m/s) taken out along each beam."""

        az = np.deg2rad(self.az_centres_deg)[:, None]
        return self.velocity - np.cos(self.elevation_rad) * (motion[0] * np.sin(az) + motion[1] * np.cos(az))

    def at_pixels(self, values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The value of the gate each ground point (m) falls in, as a radar display draws it; NaN off the scan."""

        dx = np.asarray(x, np.float64) - self.site[0]
        dy = np.asarray(y, np.float64) - self.site[1]
        s = np.hypot(dx, dy)
        az = bearing_deg(dx, dy)
        r = slant_range_m(s, self.geometry.tilt_deg)
        ia = np.floor(((az - self.az0_deg) % 360.0) / self.geometry.radial_deg).astype(np.int64)
        ir = np.floor((r - self.r0_m) / self.geometry.gate_m).astype(np.int64)
        ok = (ia >= 0) & (ia < values.shape[0]) & (ir >= 0) & (ir < values.shape[1]) & np.isfinite(r)
        out = np.full(np.shape(dx), np.nan)
        out[ok] = values[ia[ok], ir[ok]]
        return out


def gate_span(site: tuple[float, float], corners: Sequence[tuple[float, float]], geometry: Geometry
              ) -> tuple[float, int, float, int, bool]:
    """(first radial's edge, radials, first gate's edge, gates, full circle) covering the rectangle spanned by
    ``corners`` (ground metres), up to the radar's longest range."""

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    t = np.linspace(0.0, 1.0, 257)
    bx = np.concatenate([x0 + (x1 - x0) * t, np.full_like(t, x1), x1 - (x1 - x0) * t, np.full_like(t, x0)]) - site[0]
    by = np.concatenate([np.full_like(t, y0), y0 + (y1 - y0) * t, np.full_like(t, y1), y1 - (y1 - y0) * t]) - site[1]
    dist = np.hypot(bx, by)
    inside = x0 <= site[0] <= x1 and y0 <= site[1] <= y1
    pad = 2.0 * geometry.gate_m
    s_max = min(float(dist.max()) + pad, ground_range_m(geometry.max_range_m, geometry.tilt_deg))
    s_min = 0.0 if inside else max(float(dist.min()) - pad, 0.0)
    r_min = float(slant_range_m(s_min, geometry.tilt_deg))
    r_max = float(slant_range_m(s_max, geometry.tilt_deg))
    g0 = int(math.floor(r_min / geometry.gate_m))
    gates = max(1, int(math.ceil(r_max / geometry.gate_m)) - g0)
    res = geometry.radial_deg
    if inside or s_min <= 0.0:
        return 0.0, int(round(360.0 / res)), g0 * geometry.gate_m, gates, True
    az = np.sort(bearing_deg(bx, by))
    gaps = np.diff(np.append(az, az[0] + 360.0))
    g = int(np.argmax(gaps))
    start = float(az[(g + 1) % az.size]) - 2.0 * geometry.beamwidth_deg
    span = 360.0 - float(gaps[g]) + 4.0 * geometry.beamwidth_deg
    first = math.floor(start / res) * res
    radials = int(math.ceil((start + span - first) / res))
    if radials * res >= 360.0:
        return 0.0, int(round(360.0 / res)), g0 * geometry.gate_m, gates, True
    return first % 360.0, radials, g0 * geometry.gate_m, gates, False


def scan(field: Beamable, site: tuple[float, float], geometry: Geometry,
         corners: Sequence[tuple[float, float]]) -> Scan:
    """One scan at ``geometry.tilt_deg`` from ``site`` (ground metres, antenna :data:`ANTENNA_M` up) over the area
    spanned by ``corners``, beam-averaged as the module says."""

    az0, naz, r0, nr, full = gate_span(site, corners, geometry)
    az_c = az0 + (np.arange(naz) + 0.5) * geometry.radial_deg
    r_c = r0 + (np.arange(nr) + 0.5) * geometry.gate_m
    el_off, el_w = beam_quadrature(geometry.beamwidth_deg, geometry.beam_points)
    az_off, az_w = beam_quadrature(geometry.beamwidth_deg, geometry.beam_points)
    nrs = max(1, geometry.range_points) if geometry.beamwidth_deg > 0 else 1
    r_off = geometry.gate_m * ((np.arange(nrs) + 0.5) / nrs - 0.5)
    shape = (naz, nr)
    total = np.zeros(shape)
    seen = np.zeros(shape)
    acc_sig = np.zeros(shape)
    acc_vel = np.zeros(shape)
    acc_vw = np.zeros(shape)
    linear = field.signal_kind == "dbz"
    for de, we in zip(el_off, el_w):
        el = geometry.tilt_deg + de
        for dr in r_off:
            r = r_c + dr
            h = ANTENNA_M + beam_height_m(r, el)
            s = ground_range_m(r, el)
            eloc = local_elevation_rad(el, s)
            for da, wa in zip(az_off, az_w):
                weight = we * wa / nrs
                azr = np.deg2rad(az_c + da)[:, None]
                x = site[0] + s[None, :] * np.sin(azr)
                y = site[1] + s[None, :] * np.cos(azr)
                z = np.broadcast_to(h[None, :], shape)
                sig, vr = field.sample(x.ravel(), y.ravel(), z.ravel(), np.broadcast_to(azr, shape).ravel(),
                                       np.broadcast_to(eloc[None, :], shape).ravel())
                sig = sig.reshape(shape)
                vr = vr.reshape(shape)
                ok = np.isfinite(vr)
                total += weight
                seen += weight * ok
                s2 = np.where(ok, sig, 0.0)
                acc_sig += weight * s2
                # The velocity is averaged with the echo as weight (a floor keeps an echo-free beam defined).
                vw = weight * ok * (s2 if linear else s2 + 1e-6)
                acc_vel += vw * np.where(ok, vr, 0.0)
                acc_vw += vw
    with np.errstate(invalid="ignore", divide="ignore"):
        covered = seen >= 0.5 * total
        mean = np.where(covered, acc_sig / seen, np.nan)
        vel = np.where(covered & (acc_vw > 0), acc_vel / acc_vw, np.nan)
        if linear:
            signal = 10.0 * np.log10(mean)
            threshold = DBZ_THRESHOLD
        else:
            signal = mean
            threshold = MASS_THRESHOLD
        echo = signal >= threshold
    signal = np.where(echo, signal, np.nan)
    vel = np.where(echo, vel, np.nan)
    s_c = ground_range_m(r_c, geometry.tilt_deg)
    eloc_c = np.broadcast_to(local_elevation_rad(geometry.tilt_deg, s_c)[None, :], shape)
    return Scan((float(site[0]), float(site[1])), geometry, float(az0), float(r0), signal, vel, eloc_c,
                field.signal_kind, full)


# ----------------------------------------------------------------------------------------------- the storm
#: Where the radar goes by default, from the storm (km, compass bearing), the width of the panels (km), and the
#: radar's gates for each kind of run: a thunderstorm is scanned as a NEXRAD radar scans one nearby (0.5 degree
#: radials, 250 m gates); a hurricane, hundreds of kilometres across, with 1 degree radials and 1 km gates.
THUNDERSTORM = {"range_km": 40.0, "bearing_deg": 135.0, "view_km": 72.0,
                "geometry": {"gate_m": 250.0, "radial_deg": 0.5}}
HURRICANE = {"range_km": 120.0, "bearing_deg": 135.0, "view_km": 320.0,
             "geometry": {"gate_m": 1000.0, "radial_deg": 1.0}}
#: How far from the storm's rotating updraft its low-level spin is looked for (m), and the depth that spin is
#: averaged over.  A supercell's low-level rotation sits a few kilometres from its mid-level one; 8 km keeps the
#: search on that storm rather than on spin along its gust front or on a neighbour.
SPIN_SEARCH_M = 8000.0
SPIN_DEPTH_M = 1000.0
#: The storm is followed from one saved time to the next: its rotating updraft is looked for within this distance
#: (m) of where the times before put it (moved on by its own motion between the last two), and over the whole
#: area only when nothing there is strong enough.  With only one time before, its motion is not known yet, so the
#: search also reaches as far as :data:`TRACK_FIRST_SPEED` (m/s) would take it, up to three times this distance.
TRACK_RADIUS_M = 10000.0
TRACK_FIRST_SPEED = 15.0
TRACK_FILE = "radar-track.json"
TRACK_SCHEMA = "gm1.quicklook-radar-track/v1"
#: A rotating updraft: at least this much 2-5 km updraft helicity (m^2/s^2, smoothed over about 2 km); else an
#: updraft of at least this much at 5 km (m/s).
MIN_UPDRAFT_HELICITY = 10.0
MIN_UPDRAFT_5KM = 2.0


def box_mean(field2d: np.ndarray, half_width: int) -> np.ndarray:
    """(2n+1)^2 moving average with the edges repeated (a summed-area table; no scipy)."""

    n = int(half_width)
    a = np.asarray(field2d, dtype=np.float64)
    if n <= 0:
        return a.copy()
    p = np.pad(a, n, mode="edge")
    c = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    c[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * n + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / float(k * k)


#: What placed the storm (:attr:`Storm.kind`): the kinds a :class:`Track` follows to the next saved time.
FOLLOWED_KINDS = ("rotating updraft", "updraft", "eye")


@dataclass
class Storm:
    """Where the storm is in the frame (the domain's own metres from its south-west corner) and how it was found.

    ``x``, ``y``: the storm's centre, from which the radar and the panels are placed and the beam's height and the
    radar's range are given: its rotating updraft (the strongest 2-5 km updraft helicity), else its strongest
    updraft at 5 km, else its echo's middle; a hurricane's eye.  ``spin_x``, ``spin_y``: where the rotation test
    looks, the storm's low-level spin near that updraft (None: at the centre)."""

    x: float
    y: float
    how: str
    spin: float | None = None         # the low-level spin (1/s) at (spin_x, spin_y), when one was found
    found: bool = True                # False: no storm (or eye) in the frame, so the middle of the area stands in
    kind: str = ""                    # "rotating updraft", "updraft", "echo", "eye" or "" (none)
    spin_x: float | None = None
    spin_y: float | None = None
    followed: bool = False            # found near where the saved times before put it
    left: bool = False                # no storm now, though the times before had one

    def words(self, hurricane: bool) -> str:
        """What the radar and the panels are placed from, in a picture's words."""

        return ("the eye" if hurricane else "the storm") if self.found else "the middle of the area"

    @property
    def look_at(self) -> tuple[float, float]:
        """Where the rotation test looks (domain metres): the low-level spin, else the storm's centre."""

        if self.spin_x is None or self.spin_y is None:
            return self.x, self.y
        return self.spin_x, self.spin_y


@dataclass(frozen=True)
class TrackPoint:
    """The storm's centre at one saved time: ground metres (the domain's own x, y at the start), and whether a
    storm was there to follow (its rotating updraft, its updraft, or a hurricane's eye)."""

    t: float
    x: float
    y: float
    found: bool


@dataclass(frozen=True)
class Track:
    """Where the saved times before this one put the storm's centre, oldest first."""

    points: tuple[TrackPoint, ...] = ()

    def predict(self, t: float, dx: float) -> tuple[float, float, float] | None:
        """(ground x, ground y, search radius in m) where the storm should be at ``t``, or None when the time before
        had no storm to follow: the last place, moved on by the motion between the last two places when both had a
        storm; with one place only, not moved but looked for further (:data:`TRACK_FIRST_SPEED`)."""

        if not self.points or not self.points[-1].found or t <= self.points[-1].t:
            return None
        last = self.points[-1]
        dt = t - last.t
        radius = max(TRACK_RADIUS_M, 4.0 * dx)
        if len(self.points) >= 2 and self.points[-2].found and self.points[-2].t < last.t:
            before = self.points[-2]
            vx = (last.x - before.x) / (last.t - before.t)
            vy = (last.y - before.y) / (last.t - before.t)
            return last.x + vx * dt, last.y + vy * dt, radius
        return last.x, last.y, radius + min(TRACK_FIRST_SPEED * dt, 3.0 * TRACK_RADIUS_M)

    @property
    def had_storm(self) -> bool:
        return any(p.found for p in self.points)


def track_point(L: "P.Looks", storm: Storm) -> TrackPoint:
    f = L.f
    return TrackPoint(float(f.t), float(storm.x + f.domainlocx), float(storm.y + f.domainlocy),
                      storm.found and storm.kind in FOLLOWED_KINDS)


def find_storm(L: "P.Looks", track: Track | None = None) -> Storm:
    """A thunderstorm: its centre is its rotating updraft, the strongest 2-5 km updraft helicity (smoothed over
    about 2 km), else its strongest updraft at 5 km, each looked for first near where ``track`` (the saved times
    before) puts it, and over the whole area when nothing there is strong enough; the rotation test then looks at
    its low-level spin, the strongest 0-1 km mean vertical vorticity within :data:`SPIN_SEARCH_M` of a rotating
    updraft.  A storm with neither is placed at its echo's middle, and a frame with no storm at the middle of the
    area.  A hurricane: its eye, the lowest pressure at the lowest level."""

    f = L.f
    if L.hurricane:
        p = L.pressure0_hpa
        if p is not None:
            j, i = np.unravel_index(int(np.argmin(p)), p.shape)
            return Storm((i + 0.5) * f.dx, (j + 0.5) * f.dy, "the eye (the lowest pressure at the lowest level)",
                         kind="eye")
        return Storm(0.5 * f.extent[0], 0.5 * f.extent[1], "the middle of the area (no pressure in the frame)",
                     found=False)
    xs = (np.arange(f.nx) + 0.5) * f.dx
    ys = (np.arange(f.ny) + 0.5) * f.dy
    guess = track.predict(float(f.t), f.dx) if track is not None else None
    near = None
    if guess is not None:
        near = np.hypot(xs[None, :] - (guess[0] - f.domainlocx), ys[:, None] - (guess[1] - f.domainlocy)) <= guess[2]

    def strongest(field2d: np.ndarray, least: float) -> tuple[int, int, bool] | None:
        if near is not None and near.any():
            there = np.where(near, field2d, -np.inf)
            if float(there.max()) >= least:
                j, i = np.unravel_index(int(np.argmax(there)), there.shape)
                return int(i), int(j), True
        if float(field2d.max()) >= least:
            j, i = np.unravel_index(int(np.argmax(field2d)), field2d.shape)
            return int(i), int(j), False
        return None

    smooth = max(1, int(round(1000.0 / f.dx)))
    kind, how = "rotating updraft", "the storm's rotating updraft (2-5 km updraft helicity)"
    found = strongest(box_mean(np.maximum(L.uh[0], 0.0), smooth), MIN_UPDRAFT_HELICITY)
    if found is None:
        kind, how = "updraft", "the storm's strongest updraft at 5 km"
        found = strongest(box_mean(L.w5, smooth), MIN_UPDRAFT_5KM)
    if found is None:
        had = track is not None and track.had_storm
        refl = L.reflectivity
        if refl.available:
            col = L.dbz_column
            echo = col >= 20.0
            if echo.any():
                jj, ii = np.nonzero(echo)
                return Storm((float(ii.mean()) + 0.5) * f.dx, (float(jj.mean()) + 0.5) * f.dy, "the echo's middle",
                             kind="echo")
        return Storm(0.5 * f.extent[0], 0.5 * f.extent[1],
                     "the middle of the area (no storm left in it)" if had else "the middle of the area (no storm yet)",
                     found=False, left=had)
    i, j, followed = found
    ax, ay = float(xs[i]), float(ys[j])
    if kind != "rotating updraft":
        return Storm(ax, ay, how, kind=kind, followed=followed)
    levels = [k for k, z in enumerate(f.zh) if z <= SPIN_DEPTH_M] or [0]
    spin = np.zeros((f.ny, f.nx), np.float64)
    for k in levels:
        spin += f.zeta_level(k)[0]
    spin = box_mean(spin / len(levels), max(0, int(round(500.0 / f.dx))))
    close = np.hypot(xs[None, :] - ax, ys[:, None] - ay) <= SPIN_SEARCH_M
    j, i = np.unravel_index(int(np.argmax(np.where(close, spin, -np.inf))), spin.shape)
    if spin[j, i] > 0.0:
        return Storm(ax, ay, how, float(spin[j, i]), kind=kind, spin_x=float(xs[i]), spin_y=float(ys[j]),
                     followed=followed)
    return Storm(ax, ay, how, kind=kind, followed=followed)


class StormTrack:
    """The storm's centre at each saved time of a run, in time order, kept in the quick-looks' state folder
    (``.state/radar-track.json``) beside the swath and the tracks, so a picture drawn later (a pass that adds the
    radar looks to a run drawn before, or new radar settings) follows the storm exactly as one drawn in order."""

    def __init__(self, state_dir: Path | str) -> None:
        self.path = Path(state_dir) / TRACK_FILE
        self.rows: list[dict[str, Any]] = []
        self._dirty = False
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(doc, dict) and doc.get("schema") == TRACK_SCHEMA and isinstance(doc.get("rows"), list):
            self.rows = [r for r in doc["rows"] if isinstance(r, dict) and {"key", "t", "x", "y", "found"} <= set(r)]

    @staticmethod
    def _point(row: Mapping[str, Any]) -> TrackPoint:
        return TrackPoint(float(row["t"]), float(row["x"]), float(row["y"]), bool(row["found"]))

    def before(self, rows: Sequence[tuple[float, str, Any, int]],
               locate: Callable[[tuple[float, str, Any, int], Track], TrackPoint]) -> Track:
        """The track of the saved times ``rows`` (the run's lineage before one time, oldest first).  A time not kept
        yet, or kept from another frame (a branch the run left), is worked out again with ``locate`` in order, and
        everything kept after it is dropped."""

        keys = [row[1] for row in rows]
        n = 0
        while n < min(len(self.rows), len(keys)) and self.rows[n]["key"] == keys[n]:
            n += 1
        if n < len(keys):
            del self.rows[n:]
            for row in rows[n:]:
                point = locate(row, Track(tuple(self._point(r) for r in self.rows)))
                self.rows.append({"key": row[1], "t": point.t, "x": point.x, "y": point.y, "found": point.found})
            self._dirty = True
        return Track(tuple(self._point(r) for r in self.rows[:len(keys)]))

    def put(self, index: int, key: str, point: TrackPoint) -> None:
        """Keep the storm's centre of the saved time at ``index`` (after :meth:`before` of the times before it)."""

        row = {"key": key, "t": point.t, "x": point.x, "y": point.y, "found": point.found}
        if index < len(self.rows) and self.rows[index] == row:
            return
        del self.rows[index:]
        self.rows.append(row)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.partial")
        tmp.write_text(json.dumps({"schema": TRACK_SCHEMA, "rows": self.rows}, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        self._dirty = False


def view_km(kind: Mapping[str, Any], dx_m: float) -> float:
    """The panels' width: a hurricane's 320 km; a thunderstorm's 72 km on a grid of 500 m or coarser, 48 km down
    to 200 m and 32 km on a finer grid, where the storm's smaller features are worth the closer look."""

    if kind is HURRICANE or dx_m >= 500.0:
        return float(kind["view_km"])
    return 48.0 if dx_m >= 200.0 else 32.0


def default_site(storm_ground: tuple[float, float], kind: Mapping[str, Any]) -> tuple[float, float]:
    b = math.radians(kind["bearing_deg"])
    r = kind["range_km"] * 1000.0
    return storm_ground[0] + r * math.sin(b), storm_ground[1] + r * math.cos(b)


#: Bunkers et al. (2000) internal dynamics method: 7.5 m/s to the right of the 0-6 km mean wind, across the shear
#: from the 0-0.5 km mean to the 5.5-6 km mean.  It is the formula the run's own sounding report uses (its recorded
#: numbers follow from its mean wind and shear this way).
BUNKERS_DEVIATION = 7.5
#: Under this 0-6 km shear (m/s, the wind at 6 km less the wind at the ground) storms are single cells that move
#: with the wind around them (the run's own environment card calls such air "single", the storm regimes' name),
#: and a right mover is a deviation from nothing: the storm-relative look takes out the 0-6 km mean wind there.
RIGHT_MOVER_MIN_SHEAR = 10.0
#: Where the storm motion came from (:func:`storm_motion`), in a picture's words: the full words first (the
#: picture's facts and the line under the panels when it fits), then shorter ones for a narrow line.
MOTION_WORDS: dict[str, tuple[str, ...]] = {
    "set": ("set by you",),
    "domain": ("the domain's motion",),
    "bunkers-recorded": ("Bunkers right mover of the starting air", "Bunkers right mover"),
    "bunkers-corner": ("Bunkers right mover of the first saved time's corner air", "Bunkers right mover"),
    "mean-recorded": ("the 0-6 km mean wind of the starting air: too little shear for a right mover",
                      "0-6 km mean wind: too little shear", "0-6 km mean wind"),
    "mean-corner": ("the 0-6 km mean wind of the first saved time's corner air: too little shear for a right mover",
                    "0-6 km mean wind: too little shear", "0-6 km mean wind"),
}


def _layer_mean(z: np.ndarray, value: np.ndarray, z0: float, z1: float) -> float:
    """The mean of ``value`` (linear between heights ``z``) over ``z0..z1``."""

    zz = np.linspace(z0, z1, 241)
    return float(np.trapezoid(np.interp(zz, z, value), zz) / (z1 - z0))


def bunkers_right(z: Any, u: Any, v: Any) -> tuple[float, float] | None:
    """The Bunkers right mover (m/s, in the winds' own frame) of a wind profile at heights ``z`` above ground, or
    None where the 0-6 km shear is too small to give a direction or the profile does not reach 6 km."""

    z = np.asarray(z, np.float64)
    u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64)
    if z.size < 2 or float(z.max()) < 6000.0:
        return None
    mean = np.array([_layer_mean(z, u, 0.0, 6000.0), _layer_mean(z, v, 0.0, 6000.0)])
    shear = (np.array([_layer_mean(z, u, 5500.0, 6000.0), _layer_mean(z, v, 5500.0, 6000.0)])
             - np.array([_layer_mean(z, u, 0.0, 500.0), _layer_mean(z, v, 0.0, 500.0)]))
    size = float(np.hypot(*shear))
    if size < 0.5:
        return None
    right = mean + BUNKERS_DEVIATION * np.array([shear[1], -shear[0]]) / size
    return float(right[0]), float(right[1])


def air_motion(z: Any, u: Any, v: Any) -> tuple[tuple[float, float], str] | None:
    """The motion of a storm in air with the wind profile ``u``, ``v`` (m/s) at heights ``z`` (m above ground), and
    its :data:`MOTION_WORDS` key: the Bunkers right mover, or the 0-6 km mean wind where the 0-6 km shear is under
    :data:`RIGHT_MOVER_MIN_SHEAR`; None when the profile does not reach 6 km."""

    z = np.asarray(z, np.float64)
    u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64)
    if z.size < 2 or float(z.max()) < 6000.0:
        return None
    shear = math.hypot(float(np.interp(6000.0, z, u) - np.interp(0.0, z, u)),
                       float(np.interp(6000.0, z, v) - np.interp(0.0, z, v)))
    if shear < RIGHT_MOVER_MIN_SHEAR:
        return (_layer_mean(z, u, 0.0, 6000.0), _layer_mean(z, v, 0.0, 6000.0)), "mean-corner"
    right = bunkers_right(z, u, v)
    return None if right is None else (right, "bunkers-corner")


def recorded_motion(env: Mapping[str, Any] | None) -> tuple[tuple[float, float], str] | None:
    """The same from the numbers the run recorded for its starting air (``gm1.sounding-parameters/v1``, the
    importer's report or the run's receipt): its Bunkers right mover, or its 0-6 km mean wind where its 0-6 km
    shear is under :data:`RIGHT_MOVER_MIN_SHEAR`; None when the record has neither."""

    def num(key: str) -> float | None:
        value = (env or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
        return None

    shear = num("shear_0_6km_m_per_s")
    mean = (num("bunkers_mean_wind_u_m_per_s"), num("bunkers_mean_wind_v_m_per_s"))
    if shear is not None and shear < RIGHT_MOVER_MIN_SHEAR and None not in mean:
        return (mean[0], mean[1]), "mean-recorded"  # type: ignore[return-value]
    right = (num("bunkers_right_u_m_per_s"), num("bunkers_right_v_m_per_s"))
    if None not in right and not (env or {}).get("bunkers_deviation_undefined"):
        return (right[0], right[1]), "bunkers-recorded"  # type: ignore[return-value]
    return None


def storm_motion(L: "P.Looks", settings: Settings) -> tuple[tuple[float, float] | None, str]:
    """The storm motion (ground-relative, m/s) the storm-relative look takes out, and where it came from (a
    :data:`MOTION_WORDS` key; "" when the run gives none)."""

    if settings.motion_m_s is not None:
        return (float(settings.motion_m_s[0]), float(settings.motion_m_s[1])), "set"
    f = L.f
    if f.domain_moves:
        # The run moves its domain with the storm (CM1's umove, vmove): that is the motion it set for the storm.
        return (f.umove, f.vmove), "domain"
    found = L.extras.get("run_motion")
    if found is not None:
        (u, v), key = found
        return (float(u), float(v)), str(key)
    return None, ""


# ----------------------------------------------------------------------------------------------- the couplet
def azimuthal_shear(values: np.ndarray, sc: Scan) -> np.ndarray:
    """Centred azimuthal derivative (1/s, positive cyclonic: the outbound side clockwise of the inbound one) of a
    3 x 3 (radial x gate) mean of ``values``."""

    ok = np.isfinite(values)
    total = np.where(ok, values, 0.0)
    count = ok.astype(np.float64)
    pad = "wrap" if sc.full_circle else "constant"
    tp = np.pad(total, ((1, 1), (0, 0)), mode=pad)
    cp = np.pad(count, ((1, 1), (0, 0)), mode=pad)
    tsum = np.pad(tp[:-2] + tp[1:-1] + tp[2:], ((0, 0), (1, 1)))
    csum = np.pad(cp[:-2] + cp[1:-1] + cp[2:], ((0, 0), (1, 1)))
    tsum = tsum[:, :-2] + tsum[:, 1:-1] + tsum[:, 2:]
    csum = csum[:, :-2] + csum[:, 1:-1] + csum[:, 2:]
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where((csum >= 5) & ok, tsum / csum, np.nan)
    ahead = np.roll(smooth, -1, axis=0)
    behind = np.roll(smooth, 1, axis=0)
    if not sc.full_circle:
        ahead[-1] = np.nan
        behind[0] = np.nan
    s = ground_range_m(sc.r_centres_m, sc.geometry.tilt_deg)[None, :]
    dtheta = math.radians(sc.geometry.radial_deg)
    with np.errstate(invalid="ignore"):
        return (ahead - behind) / (2.0 * np.maximum(s, 1.0) * dtheta)


def find_couplet(values: np.ndarray, sc: Scan, near: tuple[float, float], *, pair_m: float, search_m: float,
                 range_gates: int) -> dict[str, Any]:
    """The rotation test on the scan near ``near`` (ground metres): the strongest cyclonic azimuthal shear within
    ``search_m`` sets a radial and a gate; the strongest outbound value on the radials clockwise of it and the
    strongest inbound one counter-clockwise of it, each within ``pair_m`` of arc and ``range_gates`` gates, are the
    pair.  ``couplet`` when both sides reach :data:`COUPLET_MIN_SIDE` with opposite signs, the outbound side is
    clockwise of the inbound one, and (out - in) / 2 reaches :data:`COUPLET_MIN_ROTATION`; ``weak`` when both
    sides are found but under those limits; ``none`` otherwise."""

    result: dict[str, Any] = {"status": "none", "pair_m": pair_m, "search_m": search_m,
                              "min_side_m_s": COUPLET_MIN_SIDE, "min_rotation_m_s": COUPLET_MIN_ROTATION}
    shear = azimuthal_shear(values, sc)
    gx, gy = sc.gate_xy()
    close = (np.hypot(gx - near[0], gy - near[1]) <= search_m) & np.isfinite(shear)
    if not close.any():
        result["reason"] = "no velocity on the scan near the storm's low-level spin"
        return result
    ia, ir = np.unravel_index(int(np.argmax(np.where(close, shear, -np.inf))), shear.shape)
    result["shear_per_s"] = float(shear[ia, ir])
    if shear[ia, ir] <= 0.0:
        result["reason"] = "no cyclonic shear"
        return result
    naz, nr = values.shape
    s = float(ground_range_m(sc.r_centres_m[ir], sc.geometry.tilt_deg))
    arc = max(s * math.radians(sc.geometry.radial_deg), 1e-6)
    k = max(1, int(math.ceil(pair_m / arc)))
    r0, r1 = max(0, ir - range_gates), min(nr, ir + range_gates + 1)

    def side(offsets: range, biggest: bool) -> tuple[float, tuple[int, int]] | None:
        rows = np.array([ia + o for o in offsets])
        rows = rows % naz if sc.full_circle else rows[(rows >= 0) & (rows < naz)]
        if rows.size == 0:
            return None
        block = values[rows, r0:r1]
        if not np.isfinite(block).any():
            return None
        at = int((np.nanargmax if biggest else np.nanargmin)(block))
        bi, bj = np.unravel_index(at, block.shape)
        return float(block[bi, bj]), (int(rows[bi]), int(r0 + bj))

    out_side = side(range(0, k + 1), True)
    in_side = side(range(-k, 1), False)
    if out_side is None or in_side is None:
        result["reason"] = "no velocity on one side of the shear"
        return result
    (vout, a), (vin, b) = out_side, in_side
    sep = float(math.hypot(gx[a] - gx[b], gy[a] - gy[b]))
    az = sc.az_centres_deg
    turn = (float(az[a[0]]) - float(az[b[0]]) + 180.0) % 360.0 - 180.0
    rotation = 0.5 * (vout - vin)
    result.update({"out_m_s": vout, "in_m_s": vin, "apart_m": sep, "rotation_m_s": rotation,
                   "centre": [0.5 * float(gx[a] + gx[b]), 0.5 * float(gy[a] + gy[b])],
                   "gate": [int(ia), int(ir)]})
    if not (vout > 0.0 > vin):
        result["reason"] = "no inbound and outbound pair"
    elif turn <= 0.0:
        result["reason"] = "the pair turns the other way (anticyclonic)"
    elif vout < COUPLET_MIN_SIDE or vin > -COUPLET_MIN_SIDE or rotation < COUPLET_MIN_ROTATION:
        result["status"] = "weak"
        result["reason"] = "weak shear"
    else:
        result["status"] = "couplet"
    return result


# ----------------------------------------------------------------------------------------------- one frame's scan
@dataclass
class View:
    """What both radar looks of one frame share: the storm, the radar, the panels' area and the scan."""

    settings: Settings
    storm: Storm
    site: tuple[float, float]            # ground metres
    site_words: str
    span: float                          # the panels' width (m)
    centre: tuple[float, float]          # the panels' middle (domain metres)
    scan: Scan
    operator: str                        # where the echo comes from, in plain words
    beam_at_storm: tuple[float, float, float]   # half-power bottom, centre, top above the ground (m)
    storm_range_m: float
    mass_words: str = ""                 # what a mass echo sums ("rain, snow and graupel"); "" for dBZ


#: The precipitation a mass echo sums, in words (``qg`` is graupel or hail as the frame's microphysics says).
PRECIPITATION_WORDS = {"qr": "rain", "qs": "snow", "qh": "hail"}


def _and(words: Sequence[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _signal(L: "P.Looks", nk: int) -> tuple[np.ndarray, str, str, str]:
    """(signal over the lowest ``nk`` levels, kind, operator words, what a mass echo sums): linear Z from the
    quick-looks' reflectivity, else the precipitation mass in g/kg."""

    from .tracks import dense_word

    f = L.f
    refl = L.reflectivity
    if refl.available:
        z = np.power(np.float32(10.0), np.asarray(refl.dbz[:nk], np.float32) / np.float32(10.0))
        return z.astype(np.float32), "dbz", refl.operator, ""
    if not f.has("q3d"):
        raise P.NotDrawn("This run's frames carry no rain, snow or hail, so a radar would see nothing.")
    names = f.species_names("q3d")
    have = [s for s in PRECIPITATION if s in names]
    if not have:
        raise P.NotDrawn("This run's microphysics has no rain, snow or hail, so a radar would see nothing.")
    q = f.view("q3d")
    total = np.zeros((nk, f.ny, f.nx), np.float32)
    for s in have:
        total += np.asarray(q[names.index(s), :nk], np.float32)
    scheme = L.facts.scheme.split(" with ")[0]
    words = f"{scheme} has no reflectivity formula in GM1: its {', '.join(have)} on the beam, g/kg"
    what = []
    for s in have:
        word = dense_word(f.config) if s == "qg" else PRECIPITATION_WORDS[s]
        if word not in what:
            what.append(word)
    return total * np.float32(1000.0), "mass", words, _and(what)


def view_of(L: "P.Looks") -> View:
    """The radar view of one frame, computed once for both looks."""

    if "radar_view" in L.extras:
        return L.extras["radar_view"]
    f = L.f
    settings = L.extras.get("radar_settings") or Settings()
    kind = HURRICANE if L.hurricane else THUNDERSTORM
    geometry = Geometry(tilt_deg=float(settings.tilt_deg), **kind["geometry"])
    storm = L.extras.get("radar_storm") or find_storm(L, L.extras.get("radar_track"))
    lx, ly = f.extent
    span = min(view_km(kind, f.dx) * 1000.0, max(lx, ly))
    storm_ground = (storm.x + f.domainlocx, storm.y + f.domainlocy)
    ox, oy = L.facts.origin_m(f.nx, f.ny)
    if settings.site_km is None:
        site = default_site(storm_ground, kind)
        site_words = (f"placed {kind['range_km']:g} km {compass(kind['bearing_deg'])} of "
                      f"{storm.words(L.hurricane)} for each picture (you can set it)")
    else:
        site = (settings.site_km[0] * 1000.0 - ox, settings.site_km[1] * 1000.0 - oy)
        site_words = (f"chosen by you, at x {S.fmt_num(settings.site_km[0], 1)} km, "
                      f"y {S.fmt_num(settings.site_km[1], 1)} km")
    # The panels: centred on the storm, pulled toward the middle of its echo (at most a quarter of the panel), and
    # kept so that where the rotation test looks stays well inside them.
    cx, cy = storm.x, storm.y
    xs = (np.arange(f.nx) + 0.5) * f.dx
    ys = (np.arange(f.ny) + 0.5) * f.dy
    if not L.hurricane and L.reflectivity.available:
        one = L.dbz_1km[0]
        inside = (np.abs(xs[None, :] - cx) <= 0.5 * span) & (np.abs(ys[:, None] - cy) <= 0.5 * span)
        echo = inside & (one >= 20.0)
        if echo.any():
            jj, ii = np.nonzero(echo)
            lim = 0.25 * span
            cx += float(np.clip(0.5 * (float(xs[ii].mean()) - cx), -lim, lim))
            cy += float(np.clip(0.5 * (float(ys[jj].mean()) - cy), -lim, lim))
    qx, qy = storm.look_at
    cx = float(np.clip(cx, qx - 0.3 * span, qx + 0.3 * span))
    cy = float(np.clip(cy, qy - 0.3 * span, qy + 0.3 * span))
    corners = [(cx + sx * 0.5 * span + f.domainlocx, cy + sy * 0.5 * span + f.domainlocy)
               for sx in (-1, 1) for sy in (-1, 1)]
    # The levels the beam reaches over the panels, plus two.
    far = max(math.hypot(c[0] - site[0], c[1] - site[1]) for c in corners) + 2.0 * geometry.gate_m
    far = min(far, float(ground_range_m(geometry.max_range_m, geometry.tilt_deg)))
    top = ANTENNA_M + float(beam_height_m(slant_range_m(far, geometry.tilt_deg),
                                          geometry.tilt_deg + 1.5 * geometry.beamwidth_deg))
    nk = int(min(f.nz, np.searchsorted(f.zh, top) + 2))
    signal, signal_kind, operator, mass_words = _signal(L, nk)
    field_ = Beamable(u=f.get("u3d", slice(0, nk)), v=f.get("v3d", slice(0, nk)),
                      w=f.get("w3d", slice(0, min(nk + 1, f.nz + 1))), signal=signal,
                      zh=np.asarray(f.zh, np.float64), zf=np.asarray(f.zf, np.float64), dx=f.dx, dy=f.dy,
                      umove=f.umove, vmove=f.vmove, domainlocx=f.domainlocx, domainlocy=f.domainlocy,
                      signal_kind=signal_kind)
    sc = scan(field_, site, geometry, corners)
    d = math.hypot(storm_ground[0] - site[0], storm_ground[1] - site[1])
    r = float(slant_range_m(d, geometry.tilt_deg))
    half = 0.5 * geometry.beamwidth_deg
    beam = tuple(ANTENNA_M + float(beam_height_m(r, geometry.tilt_deg + e)) for e in (-half, 0.0, half))
    view = View(settings, storm, (float(site[0]), float(site[1])), site_words, span, (cx, cy), sc, operator,
                beam, d, mass_words)  # type: ignore[arg-type]
    L.extras["radar_view"] = view
    return view


# ----------------------------------------------------------------------------------------------- the pictures
PANEL = 650
PANEL_GAP = 48
PANEL_Y = 182


def _boxes(view: View) -> tuple["P.Box", "P.Box"]:
    x0 = (P.W - 2 * PANEL - PANEL_GAP) // 2
    mx0 = view.centre[0] - 0.5 * view.span
    my0 = view.centre[1] - 0.5 * view.span
    return (P.Box(x0, PANEL_Y, PANEL, mx0, my0, view.span),
            P.Box(x0 + PANEL + PANEL_GAP, PANEL_Y, PANEL, mx0, my0, view.span))


def _ground_grid(L: "P.Looks", box: "P.Box") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground positions (m) of every pixel of ``box``, and whether each lies on the model's area."""

    f = L.f
    xm, ym = box.axes_m()
    xd = np.broadcast_to(xm[None, :], (box.h, box.w))
    yd = np.broadcast_to(ym[:, None], (box.h, box.w))
    on = (xd >= 0.0) & (xd <= f.extent[0]) & (yd >= 0.0) & (yd <= f.extent[1])
    return xd + f.domainlocx, yd + f.domainlocy, on


def _rings(x: np.ndarray, y: np.ndarray, site: tuple[float, float], every: float, mpp: float) -> np.ndarray:
    """How strongly a thin ring of constant ground range from the radar covers each pixel (0..1)."""

    s = np.hypot(x - site[0], y - site[1])
    d = np.abs(s - np.rint(s / every) * every) / mpp
    return np.clip(1.0 - d, 0.0, 1.0) * (s > 0.5 * every)


def _panel_image(values: np.ndarray, on: np.ndarray, colour: Any) -> np.ndarray:
    """uint8 RGB: ``colour(values)`` where a value is drawn, the ground colour elsewhere on the model's area,
    a lighter panel colour off it."""

    img = np.empty(values.shape + (3,), np.float32)
    img[...] = np.array(S.hex_rgb("void"), np.float32)
    img[~on] = np.array(S.hex_rgb("panel"), np.float32)
    shown = np.isfinite(values)
    if shown.any():
        img[shown] = colour(values[shown])
    return S.to8(img)


def _dbz_rgb(v: np.ndarray) -> np.ndarray:
    c = S.dbz_colours(v)
    void = np.array(S.hex_rgb("void"), np.float32)
    return c[..., :3] * c[..., 3:4] + void * (1 - c[..., 3:4])


def _mass_position(v: Any) -> np.ndarray:
    """Where a precipitation mass (g/kg) sits on its colours, 0..1: logarithmic from 0.01 to 10 g/kg."""

    return np.clip(np.log10(np.maximum(np.asarray(v, np.float64), 1e-9) / MASS_THRESHOLD) / 3.0, 0.0, 1.0)


def _mass_rgb(v: np.ndarray) -> np.ndarray:
    return S.colours("ember", 0.15 + 0.85 * _mass_position(v), 0.0, 1.0)


def _velocity_rgb(v: np.ndarray) -> np.ndarray:
    return S.colours("radial_velocity", v, -VELOCITY_RANGE, VELOCITY_RANGE)


def _edge_point(box: "P.Box", px: float, py: float, inset: float = 16.0) -> tuple[float, float]:
    """Where the line from the panel's middle toward (px, py) leaves the panel, a little inside it."""

    cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
    dx, dy = px - cx, py - cy
    t = min((box.w / 2.0 - inset) / max(abs(dx), 1e-9), (box.h / 2.0 - inset) / max(abs(dy), 1e-9))
    return cx + dx * t, cy + dy * t


def _mark_site(ov: S.Overlay, cv: np.ndarray, box: "P.Box", L: "P.Looks", view: View, *, label: bool) -> np.ndarray:
    """The radar on a panel: a white dot where it stands, or a white arrowhead at the panel's edge pointing to it."""

    from PIL import Image, ImageDraw

    f = L.f
    px, py = box.to_px(view.site[0] - f.domainlocx, view.site[1] - f.domainlocy)
    inside = box.x + 6 <= px <= box.x + box.w - 6 and box.y + 6 <= py <= box.y + box.h - 6
    im = Image.fromarray(cv)
    d = ImageDraw.Draw(im)
    if inside:
        d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=S.rgb8("title"), outline=S.rgb8("void"))
        if label:
            ov.text((px + 10, py - 8), "radar", "micro", colour="title", anchor="ls")
    else:
        ex, ey = _edge_point(box, px, py)
        ang = math.atan2(py - ey, px - ex)
        tip = (ex + 10 * math.cos(ang), ey + 10 * math.sin(ang))
        left = (ex - 6 * math.cos(ang) + 7 * math.cos(ang + math.pi / 2), ey - 6 * math.sin(ang)
                + 7 * math.sin(ang + math.pi / 2))
        right = (ex - 6 * math.cos(ang) - 7 * math.cos(ang + math.pi / 2), ey - 6 * math.sin(ang)
                 - 7 * math.sin(ang + math.pi / 2))
        d.polygon([tip, left, right], fill=S.rgb8("title"), outline=S.rgb8("void"))
        if label:
            gap = 16
            anchor = ("r" if ex > box.x + box.w / 2 else "l") + ("d" if ey > box.y + box.h / 2 else "a")
            tx = ex - gap if anchor[0] == "r" else ex + gap
            ty = ey - 4 if anchor[1] == "d" else ey + 4
            ov.text((tx, ty), f"radar {S.fmt_km(view.storm_range_m)} this way", "micro", colour="title",
                    anchor=anchor)
    return np.asarray(im).copy()


def _where(L: "P.Looks", view: View, x: float, y: float) -> str:
    """'12 km north-east of the eye' for a ground point."""

    f = L.f
    dx = x - (view.storm.x + f.domainlocx)
    dy = y - (view.storm.y + f.domainlocy)
    d = math.hypot(dx, dy)
    if d < 1000.0:
        return "at the eye"
    return f"{S.fmt_km(d)} {compass(float(bearing_deg(dx, dy)))} of the eye"


def _draw(L: "P.Looks", out: Path, product: str) -> dict[str, Any]:
    f = L.f
    view = view_of(L)
    sc = view.scan
    storm_relative = product == STORM_RELATIVE
    motion, motion_key = storm_motion(L, view.settings)
    motion_words = MOTION_WORDS.get(motion_key, ("",))
    if storm_relative and motion is None:
        raise P.NotDrawn("This run records no storm motion to take out. Set one with gpuwm-cm1 quicklook RUNDIR "
                         "--radar-motion U V (m/s east and north, ground-relative).")
    values = sc.storm_relative(motion) if storm_relative else sc.velocity
    frame = "storm-relative" if storm_relative else "ground-relative"
    b1, b2 = _boxes(view)
    gx, gy, on = _ground_grid(L, b1)
    echo_px = sc.at_pixels(sc.signal, gx, gy)
    vel_px = sc.at_pixels(values, gx, gy)
    echo_px[~on] = np.nan
    vel_px[~on] = np.nan
    left = _panel_image(echo_px, on, _dbz_rgb if sc.signal_kind == "dbz" else _mass_rgb)
    right = _panel_image(vel_px, on, _velocity_rgb)
    every = 50000.0 if L.hurricane else 10000.0
    ring = _rings(gx, gy, view.site, every, b1.mpp)[..., None] * np.float32(0.35)
    ring_rgb = np.array(S.rgb8("data"), np.float32)
    cv = P.canvas()
    spacing = 0.0
    for box, img in ((b1, left), (b2, right)):
        img = (img.astype(np.float32) * (1 - ring) + ring_rgb * ring + 0.5).astype(np.uint8)
        img, spacing = P.ground_ticks(img, box, f.domainlocx, f.domainlocy)
        P.paste(cv, img, box)
    cv = P.outline(P.outline(cv, b1), b2)
    tilt = view.settings.tilt_deg
    lo, mid, hi = view.beam_at_storm
    where_storm = view.storm.words(L.hurricane)
    data = S.data_line(L.facts.name, f"{S.fmt_grid(f.dx)} grid",
                       f"{tilt:g}° scan from a radar {S.fmt_km(view.storm_range_m)} from {where_storm}",
                       f"{S.fmt_km(view.span)} across")
    notes = S.data_line(WHAT_IT_IS, f"radar site {view.site_words}", "4/3-earth beam path")
    ov = P.overlay(L, product, data, level=f"{tilt:g}° scan, beam {S.fmt_height(mid)} up at {where_storm}",
                   wind=frame, notes=notes)
    # The panels' words.
    # The numbers over the panels are the panels' own (the scan's sector reaches a little beyond them).
    echo_shown = np.isfinite(echo_px)
    top = float(echo_px[echo_shown].max()) if echo_shown.any() else None
    if sc.signal_kind == "dbz":
        echo_words = f"max {top:.0f} dBZ" if top is not None else "nothing above 5 dBZ"
        P.panel_label(ov, b1, "reflectivity", echo_words)
        P.dbz_legend(ov, b1.x + b1.w - ov.U(300), b1.y + b1.h + ov.U(12), ov.U(300))
    else:
        echo_words = f"max {top:.2f} g/kg" if top is not None else "nothing above 0.01 g/kg"
        title = f"{view.mass_words or 'precipitation'} (not reflectivity)"
        if ov.measure(title, "label")[0] + ov.measure(echo_words, "label")[0] + ov.U(24) > b1.w:
            title = "precipitation (not reflectivity)"
        P.panel_label(ov, b1, title, echo_words)
        ov.ramp_bar("ember", (b1.x + b1.w - ov.U(300), b1.y + b1.h + ov.U(12), b1.x + b1.w,
                              b1.y + b1.h + ov.U(12) + max(3, ov.U(7))), vmin=-0.15 / 0.85, vmax=1.0,
                    ticks=[(0.0, "0.01"), (1.0 / 3.0, "0.1"), (2.0 / 3.0, "1"), (1.0, "10 g/kg")])
    shown = np.isfinite(vel_px) & on
    vmin = float(np.nanmin(vel_px[shown])) if shown.any() else None
    vmax = float(np.nanmax(vel_px[shown])) if shown.any() else None
    vel_words = (f"{S.fmt_num(vmin, sign=True)} to {S.fmt_num(vmax, sign=True)} m/s {frame}" if shown.any()
                 else "no echo to measure")
    P.panel_label(ov, b2, f"{frame} velocity", vel_words)
    ov.ramp_bar("radial_velocity", (b2.x + b2.w - ov.U(360), b2.y + b2.h + ov.U(12), b2.x + b2.w,
                                    b2.y + b2.h + ov.U(12) + max(3, ov.U(7))),
                vmin=-VELOCITY_RANGE, vmax=VELOCITY_RANGE,
                ticks=[(-VELOCITY_RANGE, S.fmt_num(-VELOCITY_RANGE)), (-35.0, S.fmt_num(-35)), (0.0, "0"),
                       (35.0, "+35"), (VELOCITY_RANGE, f"+{VELOCITY_RANGE:g} m/s")])
    ov.text((b2.x + b2.w - ov.U(372), b2.y + b2.h + ov.U(21)), "green toward the radar, red away", "micro",
            colour="data", anchor="rs", halo=False)
    P.scale_bar(ov, b1.x, b1.y + b1.h + ov.U(16), b1)
    cv = _mark_site(ov, cv, b1, L, view, label=False)
    cv = _mark_site(ov, cv, b2, L, view, label=True)
    # The couplet (a thunderstorm), or the eye (a hurricane).
    facts: dict[str, Any] = {}
    lines_right: list[tuple[str, str]] = []
    reach_m = float(ground_range_m(sc.geometry.max_range_m, sc.geometry.tilt_deg))
    model_top = float(f.zf[-1]) if len(f.zf) else float(f.zh[-1])
    if not shown.any():
        if view.storm_range_m > reach_m:
            why = (f"the radar is {S.fmt_km(view.storm_range_m)} from {where_storm}, beyond its "
                   f"{S.fmt_km(sc.geometry.max_range_m)} range: move it closer")
        elif lo > model_top:
            why = (f"the beam is {S.fmt_height(lo)} up at {where_storm}, above the model's top "
                   f"({S.fmt_height(model_top)}): lower the tilt")
        elif view.storm.left:
            why = "no echo on this scan: the storm has left the model's area"
        else:
            why = "no echo on this scan yet: a radar sees the wind only where rain, snow or hail reflects it"
        lines_right.append((why, "data"))
        if storm_relative:
            facts["couplet"] = {"status": "none", "reason": why}
    elif L.hurricane:
        for box in (b1, b2):
            qx, qy = box.to_px(view.storm.x, view.storm.y)
            if box.x <= qx <= box.x + box.w and box.y <= qy <= box.y + box.h:
                ov.ring((qx, qy), 12.0)
        lines_right.append(("gold ring: the eye, the lowest pressure at the lowest level", "gold"))
        if shown.any():
            gxs, gys = gx[shown], gy[shown]
            vs = vel_px[shown]
            k_in, k_out = int(np.argmin(vs)), int(np.argmax(vs))
            lines_right.append((f"fastest toward the radar {_where(L, view, gxs[k_in], gys[k_in])}, away "
                                f"{_where(L, view, gxs[k_out], gys[k_out])}", "data"))
    elif storm_relative:
        footprint = view.storm_range_m * math.radians(sc.geometry.beamwidth_deg)
        pair = min(4000.0, max(2.0 * footprint, 4.0 * sc.geometry.gate_m, 3.0 * f.dx))
        search = max(1500.0, 3.0 * f.dx)
        gates = max(2, int(math.ceil(f.dx / sc.geometry.gate_m)))
        lx_, ly_ = view.storm.look_at
        at = "at the storm's low-level spin" if view.storm.spin_x is not None else f"at {where_storm}"
        couplet = find_couplet(values, sc, (lx_ + f.domainlocx, ly_ + f.domainlocy), pair_m=pair, search_m=search,
                               range_gates=gates)
        facts["couplet"] = couplet
        if couplet["status"] == "couplet":
            cxg, cyg = couplet["centre"]
            r_ring = max(10.0, 0.5 * couplet["apart_m"] / b2.mpp + 8.0)
            for box in (b1, b2):
                ov.ring(box.to_px(cxg - f.domainlocx, cyg - f.domainlocy), r_ring)
            lines_right.append((f"gold ring: rotation, (away − toward)/2 = {couplet['rotation_m_s']:.1f} m/s "
                                f"{frame}, peaks {S.fmt_km(couplet['apart_m'])} apart", "gold"))
        elif couplet["status"] == "weak":
            # One decimal, cut rather than rounded, so a value under the limit never prints as the limit.
            weak = math.floor(couplet["rotation_m_s"] * 10.0) / 10.0
            lines_right.append((f"weak shear {at}: (away − toward)/2 = {weak:.1f} m/s {frame}, not a pair",
                                "data"))
        else:
            lines_right.append((f"no rotation pair on this scan {at}", "data"))
        lines_right.append((f"a pair: each side {COUPLET_MIN_SIDE:g} m/s and (away − toward)/2 "
                            f"{COUPLET_MIN_ROTATION:g} m/s or more, {frame}", "muted"))
    else:
        lines_right.append(("the wind as the radar measures it, with the storm's own motion in it", "data"))
        lines_right.append(("a storm's rotation shows best on Radar velocity, storm-relative", "muted"))
    if storm_relative and motion is not None:
        # Where the motion came from, in its full words: on the same line when they fit within the panel, else on
        # the line under it when a line is free, else the shorter words that fit.
        head = (f"storm motion taken out: {S.fmt_num(motion[0], 1)} m/s east, {S.fmt_num(motion[1], 1)} m/s north, "
                "ground-relative")
        fits = [w for w in motion_words if ov.measure(f"{head} ({w})", "micro")[0] <= b2.w]
        free = 4 - len(lines_right) - 1 - (0 if on.all() else 1)
        if fits and fits[0] == motion_words[0]:
            lines_right.append((f"{head} ({motion_words[0]})", "data"))
        elif free >= 1 and ov.measure(f"({motion_words[0]})", "micro")[0] <= b2.w:
            lines_right.append((head, "data"))
            lines_right.append((f"({motion_words[0]})", "muted"))
        else:
            lines_right.append((f"{head} ({fits[0] if fits else motion_words[-1]})", "data"))
    elif L.hurricane:
        lines_right.append(("the wind as the radar measures it, with the storm's own motion in it", "muted"))
    if not on.all():
        lines_right.append(("lighter grey: outside the model's area", "muted"))
    seen = (f"velocity where the echo is {DBZ_THRESHOLD:g} dBZ or more" if sc.signal_kind == "dbz"
            else f"velocity where there is {MASS_THRESHOLD:g} g/kg or more")
    lines_left = [(f"echo: {view.operator}, in Z over the beam" if sc.signal_kind == "dbz"
                   else f"echo: {view.operator}", "data"),
                  (f"beam {S.fmt_height(max(lo, 0.0))} to {S.fmt_height(hi)} up at {where_storm} (half-power), "
                   f"{sc.geometry.words}", "data"),
                  (f"{seen} · arcs every {S.fmt_km(every)} from the radar", "muted"),
                  (f"+ fixed to the ground every {S.fmt_km(spacing)} · {L.frame_words()}", "muted")]
    ms = ov.px("micro")
    y0 = b1.y + b1.h + ov.U(44) + ms
    step = int(ms * 1.42)
    for n, (text, colour) in enumerate(lines_left[:4]):
        ov.text((b1.x, y0 + n * step), text, "micro", colour=colour, anchor="ls", halo=False)
    for n, (text, colour) in enumerate(lines_right[:4]):
        ov.text((b2.x, y0 + n * step), text, "micro", colour=colour, anchor="ls", halo=False)
    P.save(ov, cv, out)
    ox, oy = L.facts.origin_m(f.nx, f.ny)
    facts.update({
        "tilt_deg": float(tilt), "site_km": [(view.site[0] + ox) / 1000.0, (view.site[1] + oy) / 1000.0],
        "site_chosen": view.settings.site_km is not None,
        "storm_xy_km": [(view.storm.x + f.domainlocx + ox) / 1000.0, (view.storm.y + f.domainlocy + oy) / 1000.0],
        "storm_placed_by": view.storm.how, "storm_followed": view.storm.followed,
        "rotation_looked_for_km": [(view.storm.look_at[0] + f.domainlocx + ox) / 1000.0,
                                   (view.storm.look_at[1] + f.domainlocy + oy) / 1000.0],
        "storm_range_km": view.storm_range_m / 1000.0,
        "beam_at_storm_m": [lo, mid, hi], "echo": sc.signal_kind, "operator": view.operator,
        "echo_max": top, "velocity_min_m_s": vmin, "velocity_max_m_s": vmax, "frame": frame,
        "view_km": view.span / 1000.0, "gate_m": sc.geometry.gate_m, "radial_deg": sc.geometry.radial_deg,
        "beamwidth_deg": sc.geometry.beamwidth_deg, "range_m_s": VELOCITY_RANGE})
    if view.mass_words:
        facts["echo_sums"] = view.mass_words
    if storm_relative:
        facts.update({"storm_motion_m_s": list(motion), "storm_motion_from": motion_words[0],
                      "storm_motion_key": motion_key})
    return facts


def product_radar_storm_relative(L: "P.Looks", out: Path) -> dict[str, Any]:
    """Reflectivity and storm-relative radial velocity on one low scan (thunderstorm runs)."""

    return _draw(L, out, STORM_RELATIVE)


def product_radar_velocity(L: "P.Looks", out: Path) -> dict[str, Any]:
    """Reflectivity and ground-relative radial velocity on one low scan."""

    return _draw(L, out, GROUND_RELATIVE)


__all__ = ["ANTENNA_M", "Beamable", "GROUND_RELATIVE", "Geometry", "LOOK_IDS", "MOTION_WORDS", "STORM_RELATIVE", "Scan",
           "Settings", "SettingsError", "Storm", "StormTrack", "Track", "TrackPoint", "air_motion", "beam_height_m",
           "beam_quadrature", "bearing_deg", "find_couplet", "find_storm", "ground_range_m", "local_elevation_rad",
           "product_radar_storm_relative", "product_radar_velocity", "radial_velocity", "read_settings",
           "recorded_motion", "scan", "slant_range_m", "storm_motion", "track_point", "view_of", "write_settings"]
