"""The radar looks (:mod:`cm1core.quicklook.radar`): one low scan of a simulated Doppler radar through a frame.

CPU only.  What is held here:

1. the beam geometry and the radial velocity, checked against an independent construction: straight rays on a
   sphere of 4/3 the earth's radius, built from three-dimensional vectors, with the wind laid out in each gate's own
   local frame and dotted with the ray's direction there;
2. the sampling of the C grid (u on x faces, v on y faces, w on w levels, stretched levels, a moving domain): fields
   linear in x, y and z come back exactly, so a staggering or a level that is off by one shows;
3. whole scans of known winds: a uniform wind gives ``U sin(az) cos(e)`` at every gate and nothing storm-relative
   when the storm moves with it; a turning vortex passes the rotation test, one turning the other way does not;
4. the echo: Morrison through CM1's calc_refl10cm (the quick-looks' own), WSM6's own reflectivity, and a scheme
   without a reflectivity formula (Thompson) drawn as precipitation mass and labelled so;
5. the looks drawn by the worker for every frame, their words linted, nothing cut from a footer; the radar
   settings and ``--radar-site`` / ``--radar-tilt`` / ``--radar-motion``; the GUI's radar endpoint.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest

import importlib.util

NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")

from cm1import_support import namelist_tool  # noqa: E402
from quicklook_support import run_directory  # noqa: E402

from cm1core.quicklook import products as P  # noqa: E402
from cm1core.quicklook import radar as RA  # noqa: E402
from cm1core.quicklook import render as RD  # noqa: E402

RADAR = ("radar-storm-relative", "radar-velocity")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


# ---------------------------------------------------------------- 1. the beam, independently

def _ray(r: float, e_deg: float, az_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A straight ray on the equivalent earth: (gate position, ray direction, radar position), the earth's centre
    at the origin and the radar at its antenna height on top."""

    big = RA.effective_radius_m()
    e, a = math.radians(e_deg), math.radians(az_deg)
    radar = np.array([0.0, 0.0, big])
    d = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])
    return radar + r * d, d, radar


@pytest.mark.parametrize("r,e", [(1000.0, 0.5), (40000.0, 0.5), (100000.0, 0.5), (230000.0, 1.5), (50000.0, 10.0),
                                 (20000.0, 19.5)])
def test_the_beam_height_and_ground_range_are_those_of_a_straight_ray_on_the_four_thirds_earth(r, e):
    big = RA.effective_radius_m()
    gate, d, radar = _ray(r, e, 37.0)
    height = float(np.linalg.norm(gate)) - big
    ground = big * math.atan2(math.hypot(float(gate[0]), float(gate[1])), float(gate[2]))   # the angle at the centre
    assert float(RA.beam_height_m(r, e)) == pytest.approx(height, rel=1e-9, abs=1e-6)
    assert float(RA.ground_range_m(r, e)) == pytest.approx(ground, rel=1e-9)
    assert float(RA.slant_range_m(ground, e)) == pytest.approx(r, rel=1e-9)
    up = gate / np.linalg.norm(gate)
    assert float(RA.local_elevation_rad(e, ground)) == pytest.approx(math.asin(float(d @ up)), abs=1e-12)


def test_the_beam_heights_match_the_usual_rule_of_thumb():
    """h = r sin(e) + r^2 / (2 ke a), the textbook approximation (Doviak and Zrnic, eq. 2.28b expanded): at 0.5
    degrees the beam is 0.44 km up at 40 km, 1.46 km at 100 km and 5.12 km at 230 km."""

    big = RA.effective_radius_m()
    for r, expected_km in ((40000.0, 0.443), (100000.0, 1.461), (230000.0, 5.120)):
        approx = r * math.sin(math.radians(0.5)) + r * r / (2.0 * big)
        assert float(RA.beam_height_m(r, 0.5)) == pytest.approx(approx, abs=2.0)
        assert float(RA.beam_height_m(r, 0.5)) / 1000.0 == pytest.approx(expected_km, abs=0.002)


def test_the_radial_velocity_is_the_wind_along_the_ray_in_three_dimensions():
    """The wind (u east, v north, w up in the model's flat frame) placed in each gate's own local frame (up away
    from the earth's centre, along the ground away from the radar, and across it) and dotted with the ray."""

    rng = np.random.default_rng(3)
    for _ in range(400):
        r = float(rng.uniform(500.0, 230000.0))
        e = float(rng.uniform(0.1, 19.5))
        az = float(rng.uniform(0.0, 360.0))
        u, v, w = (float(x) for x in rng.normal(0.0, 25.0, 3))
        gate, d, radar = _ray(r, e, az)
        up = gate / np.linalg.norm(gate)
        # the ground direction away from the radar at the gate: in the plane of the radar, the gate and the centre
        toward = radar / np.linalg.norm(radar)
        along = np.cross(np.cross(toward, up), up)       # (toward . up) up - toward: horizontal, away from the radar
        along = along / np.linalg.norm(along)
        across = np.cross(along, up)   # 90 degrees clockwise of along, seen from above
        a = math.radians(az)
        # The model's plane is laid out by ground range and bearing from the radar, so at the gate "along" points to
        # bearing az and "across" to az + 90 degrees.
        wind = (u * math.sin(a) + v * math.cos(a)) * along + (u * math.cos(a) - v * math.sin(a)) * across + w * up
        expected = float(wind @ d)
        s = float(RA.ground_range_m(r, e))
        got = float(RA.radial_velocity(u, v, w, a, RA.local_elevation_rad(e, s)))
        assert got == pytest.approx(expected, abs=1e-9)


def test_the_beam_pattern_is_the_two_way_gaussian_of_its_half_power_width():
    offsets, weights = RA.beam_quadrature(0.95, 3)
    assert weights.sum() == pytest.approx(1.0)
    sigma = 0.95 / math.sqrt(16.0 * math.log(2.0))
    assert float(weights @ offsets ** 2) == pytest.approx(sigma ** 2, rel=1e-12)
    # the two-way power falls to a quarter (each way to a half) at half the beam width
    two_way = lambda t: math.exp(-0.5 * t * t / sigma ** 2)  # noqa: E731
    assert two_way(0.475) == pytest.approx(0.25, rel=1e-9)


# ---------------------------------------------------------------- 2. the C grid

def _stretched_levels(nz: int = 30) -> tuple[np.ndarray, np.ndarray]:
    dz = np.concatenate([np.full(6, 100.0), np.linspace(120.0, 600.0, nz - 6)])
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    return zf, 0.5 * (zf[1:] + zf[:-1])


def _linear_field(nx=40, ny=30, dx=500.0, dy=400.0, *, umove=12.5, vmove=3.0, loc=(7000.0, -2500.0),
                  a=0.002, b=0.004, c=-0.003, g=0.001, h=0.0015) -> tuple[RA.Beamable, dict]:
    """u = a x + b z (on x faces, scalar heights), v = c y (on y faces), w = g z + h x (w levels), relative to the
    domain; positions in the domain's own metres."""

    zf, zh = _stretched_levels()
    nz = zh.size
    xf = np.arange(nx + 1) * dx
    xc = (np.arange(nx) + 0.5) * dx
    yf = np.arange(ny + 1) * dy
    u = (a * xf[None, None, :] + b * zh[:, None, None]) * np.ones((nz, ny, nx + 1))
    v = (c * yf[None, :, None]) * np.ones((nz, ny + 1, nx))
    w = (g * zf[:, None, None] + h * xc[None, None, :]) * np.ones((nz + 1, ny, nx))
    signal = np.ones((nz, ny, nx), np.float32)
    field = RA.Beamable(u.astype(np.float32), v.astype(np.float32), w.astype(np.float32), signal, zh, zf, dx, dy,
                        umove=umove, vmove=vmove, domainlocx=loc[0], domainlocy=loc[1])
    return field, dict(a=a, b=b, c=c, g=g, h=h, umove=umove, vmove=vmove, loc=loc, nx=nx, ny=ny, dx=dx, dy=dy,
                       zf=zf, zh=zh)


def test_the_c_grid_and_stretched_levels_give_back_a_wind_that_is_linear_in_space():
    field, k = _linear_field()
    rng = np.random.default_rng(5)
    n = 2000
    xd = rng.uniform(0.6 * k["dx"], (k["nx"] - 0.6) * k["dx"], n)
    yd = rng.uniform(0.6 * k["dy"], (k["ny"] - 0.6) * k["dy"], n)
    z = rng.uniform(k["zh"][0], k["zh"][-1], n)
    az = rng.uniform(0.0, 2.0 * math.pi, n)
    el = rng.uniform(0.0, 0.3, n)
    sig, vr = field.sample(xd + k["loc"][0], yd + k["loc"][1], z, az, el)
    u = k["a"] * xd + k["b"] * z + k["umove"]
    v = k["c"] * yd + k["vmove"]
    w = k["g"] * z + k["h"] * xd
    expected = RA.radial_velocity(u, v, w, az, el)
    assert np.allclose(vr, expected, atol=2e-4), float(np.abs(vr - expected).max())
    assert np.allclose(sig, 1.0)
    # off the model's area, or below the ground: nothing
    sig, vr = field.sample(np.array([k["loc"][0] - 10.0, k["loc"][0] + 10.0]), np.array([k["loc"][1] + 10.0] * 2),
                           np.array([200.0, -1.0]), np.zeros(2), np.zeros(2))
    assert np.isnan(vr).all() and np.isnan(sig).all()


# ---------------------------------------------------------------- 3. whole scans

def _uniform(u0: float, v0: float, w0: float = 0.0, *, n=160, dx=1000.0, umove=0.0, vmove=0.0) -> RA.Beamable:
    zf, zh = _stretched_levels()
    nz = zh.size
    return RA.Beamable(np.full((nz, n, n + 1), u0 - umove, np.float32), np.full((nz, n + 1, n), v0 - vmove, np.float32),
                       np.full((nz + 1, n, n), w0, np.float32), np.full((nz, n, n), 1000.0, np.float32), zh, zf, dx,
                       dx, umove=umove, vmove=vmove)


def test_a_uniform_wind_scans_as_its_component_along_every_beam_and_vanishes_storm_relative():
    field = _uniform(15.0, -6.0, umove=10.0, vmove=2.0)
    geometry = RA.Geometry()
    site = (20000.0, 20000.0)
    sc = RA.scan(field, site, geometry, [(60000.0, 60000.0), (120000.0, 60000.0), (60000.0, 120000.0),
                                         (120000.0, 120000.0)])
    az = np.deg2rad(sc.az_centres_deg)[:, None]
    expected = np.cos(sc.elevation_rad) * (15.0 * np.sin(az) - 6.0 * np.cos(az))
    seen = np.isfinite(sc.velocity)
    assert seen.sum() > 10000 and not sc.full_circle
    # the beam's width spreads the azimuth by a fraction of a degree: cos(sigma) of the component, under 0.01 m/s
    assert np.abs(sc.velocity[seen] - expected[seen]).max() < 0.01
    assert np.abs(sc.storm_relative((15.0, -6.0))[seen]).max() < 0.01
    # 30 dBZ everywhere (1000 mm^6 m^-3) stays 30 dBZ through the beam average
    assert np.allclose(sc.signal[seen], 30.0, atol=1e-6)


def _vortex(sign: float, vmax: float = 30.0, core: float = 2500.0, centre=(80000.0, 80000.0), n=160,
            dx=1000.0) -> RA.Beamable:
    """A Rankine vortex turning counter-clockwise (sign +1, cyclonic in the north) or clockwise (-1)."""

    zf, zh = _stretched_levels()
    nz = zh.size

    def tangential(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rx, ry = x - centre[0], y - centre[1]
        r = np.hypot(rx, ry) + 1e-9
        speed = np.where(r <= core, vmax * r / core, vmax * core / r)
        return -sign * speed * ry / r, sign * speed * rx / r

    xf, yc = np.meshgrid(np.arange(n + 1) * dx, (np.arange(n) + 0.5) * dx)
    u = tangential(xf, yc)[0]
    xc, yf = np.meshgrid((np.arange(n) + 0.5) * dx, np.arange(n + 1) * dx)
    v = tangential(xc, yf)[1]
    return RA.Beamable(np.broadcast_to(u, (nz, n, n + 1)).astype(np.float32),
                       np.broadcast_to(v, (nz, n + 1, n)).astype(np.float32),
                       np.zeros((nz + 1, n, n), np.float32), np.full((nz, n, n), 1000.0, np.float32), zh, zf, dx, dx)


def _scan_of(field: RA.Beamable, centre=(80000.0, 80000.0)) -> tuple[RA.Scan, tuple[float, float]]:
    site = (centre[0] + 40000.0 * math.sin(math.radians(135.0)), centre[1] + 40000.0 * math.cos(math.radians(135.0)))
    corners = [(centre[0] + sx * 15000.0, centre[1] + sy * 15000.0) for sx in (-1, 1) for sy in (-1, 1)]
    return RA.scan(field, site, RA.Geometry(), corners), site


def test_a_turning_vortex_shows_as_a_couplet_with_its_speed_and_one_turning_the_other_way_does_not():
    centre = (80000.0, 80000.0)
    sc, _site = _scan_of(_vortex(+1.0))
    couplet = RA.find_couplet(sc.velocity, sc, centre, pair_m=3000.0, search_m=3000.0, range_gates=4)
    assert couplet["status"] == "couplet", couplet
    # the beam (0.66 km wide at 40 km) and the 3 x 3 shear smoothing blunt the 30 m/s peaks only a little
    assert 25.0 <= couplet["rotation_m_s"] <= 30.0
    assert couplet["out_m_s"] > 20.0 and couplet["in_m_s"] < -20.0
    # the pair sits across the vortex, about two core radii apart
    assert 3000.0 <= couplet["apart_m"] <= 7000.0
    assert math.hypot(couplet["centre"][0] - centre[0], couplet["centre"][1] - centre[1]) < 1500.0
    sc, _site = _scan_of(_vortex(-1.0))
    other = RA.find_couplet(sc.velocity, sc, centre, pair_m=3000.0, search_m=3000.0, range_gates=4)
    assert other["status"] == "none", other
    sc, _site = _scan_of(_vortex(+1.0, vmax=4.0))
    weak = RA.find_couplet(sc.velocity, sc, centre, pair_m=3000.0, search_m=3000.0, range_gates=4)
    assert weak["status"] == "weak" and weak["rotation_m_s"] < RA.COUPLET_MIN_ROTATION, weak


def test_the_pixels_show_the_gate_they_fall_in():
    field = _uniform(10.0, 0.0)
    sc = RA.scan(field, (0.0, 0.0), RA.Geometry(), [(20000.0, 20000.0), (60000.0, 60000.0)])
    gx, gy = sc.gate_xy()
    ia, ir = 7, 11
    values = np.arange(sc.signal.size, dtype=np.float64).reshape(sc.signal.shape)
    assert sc.at_pixels(values, np.array([gx[ia, ir]]), np.array([gy[ia, ir]]))[0] == values[ia, ir]


def test_the_bunkers_right_mover_is_the_runs_own_formula():
    """A straight hodograph, 0 to 30 m/s east over 6 km: the mean wind is 15 m/s east, the shear from the lowest
    half kilometre to the top one 27.5 m/s east, so the right mover is 7.5 m/s south of the mean.  And the 1 km
    stock supercell's own report (import-report.json, sounding.environment) holds a mean wind of (13.524, 6.117),
    a shear of (29.094, 5.690) and a right mover of (14.964, -1.243) m/s: the same formula."""

    z = np.linspace(0.0, 12000.0, 121)
    u = np.clip(z / 6000.0, 0.0, 1.0) * 30.0
    right = RA.bunkers_right(z, u, np.zeros_like(z))
    assert right == pytest.approx((15.0, -7.5), abs=0.01)
    mean, shear = np.array([13.524492, 6.117333]), np.array([29.093580, 5.690150])
    recorded = mean + RA.BUNKERS_DEVIATION * np.array([shear[1], -shear[0]]) / np.hypot(*shear)
    assert recorded == pytest.approx((14.964074, -1.243211), abs=1e-5)
    assert RA.bunkers_right(z, np.zeros_like(z), np.zeros_like(z)) is None


def test_with_too_little_shear_the_storm_relative_look_takes_out_the_mean_wind_not_a_right_mover():
    """Calm air, u = z / 3000 (2 m/s of 0-6 km shear): storms there are single cells that move with the wind around
    them, so a right mover 7.5 m/s across a 2 m/s shear means nothing; the 0-6 km mean wind (1 m/s east) is the
    motion.  Strong shear keeps the right mover.  The run's own record is read the same way: the calm pulse-storm
    run's importer report (0-6 km shear 1.96 m/s, mean wind 1.0 m/s east, "right mover" 7.5 m/s south of it)."""

    z = np.linspace(0.0, 12000.0, 121)
    motion, key = RA.air_motion(z, z / 3000.0, np.zeros_like(z))
    assert key == "mean-corner" and motion == pytest.approx((1.0, 0.0), abs=1e-3)
    assert "too little shear for a right mover" in RA.MOTION_WORDS[key][0]
    motion, key = RA.air_motion(z, np.clip(z / 6000.0, 0.0, 1.0) * 30.0, np.zeros_like(z))
    assert key == "bunkers-corner" and motion == pytest.approx((15.0, -7.5), abs=0.01)
    assert RA.air_motion(np.linspace(0.0, 5000.0, 11), np.zeros(11), np.zeros(11)) is None
    calm = {"schema": "gm1.sounding-parameters/v1", "bunkers_mean_wind_u_m_per_s": 0.9990302324295044,
            "bunkers_mean_wind_v_m_per_s": 0, "bunkers_right_u_m_per_s": 0.9990302324295044,
            "bunkers_right_v_m_per_s": -7.5, "shear_0_6km_m_per_s": 1.958299994468689}
    assert RA.recorded_motion(calm) == ((pytest.approx(0.99903), 0.0), "mean-recorded")
    supercell = {**calm, "bunkers_mean_wind_u_m_per_s": 13.524, "bunkers_mean_wind_v_m_per_s": 6.117,
                 "bunkers_right_u_m_per_s": 14.964, "bunkers_right_v_m_per_s": -1.243, "shear_0_6km_m_per_s": 31.13}
    assert RA.recorded_motion(supercell) == ((14.964, -1.243), "bunkers-recorded")
    assert RA.recorded_motion({"bunkers_right_u_m_per_s": 1.0, "bunkers_right_v_m_per_s": 2.0,
                               "bunkers_deviation_undefined": True}) is None
    assert RA.recorded_motion(None) is None


def test_a_scheme_without_a_reflectivity_formula_names_the_precipitation_it_sums():
    """Thompson's own frames (qv, qc, qr, qi, qs, qg; its dense ice recorded as no selectable branch, so graupel):
    rain, snow and graupel, not cloud ice.  A frame whose dense ice is hail says hail."""

    from types import SimpleNamespace

    names = ["qv", "qc", "qr", "qi", "qs", "qg"]
    q = np.zeros((6, 4, 3, 3), np.float32)
    q[names.index("qr")] = 0.001
    q[names.index("qi")] = 0.5     # cloud ice, which a radar look of precipitation leaves out

    def looks(category: str):
        f = SimpleNamespace(has=lambda name: name == "q3d", species_names=lambda name: names, view=lambda name: q,
                            ny=3, nx=3, config={"dense_ice": {"category": category}})
        return SimpleNamespace(f=f, reflectivity=SimpleNamespace(available=False),
                               facts=SimpleNamespace(scheme="Thompson"))

    signal, kind, operator, what = RA._signal(looks("no-selectable-dense-ice-branch"), 2)
    assert kind == "mass" and what == "rain, snow and graupel" and "qr, qs, qg" in operator
    assert signal.shape == (2, 3, 3) and np.allclose(signal, 1.0)       # g/kg, the cloud ice left out
    assert RA._signal(looks("hail"), 2)[3] == "rain, snow and hail"


# ---------------------------------------------------------------- following the storm

def _frame_looks(uh: np.ndarray, *, t: float, spin: np.ndarray | None = None, dx: float = 1000.0,
                 loc: tuple[float, float] = (0.0, 0.0)):
    """What find_storm reads of one frame: the 2-5 km updraft helicity, the 5 km updraft, the spin near the
    ground on levels at 250 m and 750 m, the frame's time and where its domain is."""

    from types import SimpleNamespace

    ny, nx = uh.shape
    zeta = np.zeros_like(uh) if spin is None else spin
    f = SimpleNamespace(dx=dx, dy=dx, nx=nx, ny=ny, extent=(nx * dx, ny * dx), zh=np.array([250.0, 750.0, 1500.0]),
                        domainlocx=loc[0], domainlocy=loc[1], t=t, zeta_level=lambda k: (zeta, "spin"))
    return SimpleNamespace(f=f, hurricane=False, uh=(uh.astype(np.float32), "uh"), w5=np.zeros_like(uh),
                           reflectivity=SimpleNamespace(available=False))


def _blob(n: int, at_km: tuple[float, float], peak: float, width_km: float = 2.0) -> np.ndarray:
    x = np.arange(n) + 0.5
    return peak * np.exp(-((x[None, :] - at_km[0]) ** 2 + (x[:, None] - at_km[1]) ** 2) / (2.0 * width_km ** 2))


def test_the_storm_is_followed_from_one_saved_time_to_the_next():
    """Two rotating updrafts: a stronger one at (45, 45) km and the one the track followed, at (15, 15) km.  With
    the track (the storm at (13, 13) then (14, 14) km, so (15, 15) km now) the followed one is kept; with no track,
    or a track whose storm is nowhere near anything strong enough, the strongest in the area is taken.  The low-level
    spin is looked for within 8 km of the updraft, so a stronger spin 11 km away is not taken."""

    uh = _blob(60, (15.5, 15.5), 60.0) + _blob(60, (45.5, 45.5), 200.0)
    spin = _blob(60, (12.5, 13.5), 0.004, 1.0) + _blob(60, (26.5, 15.5), 0.02, 1.0)
    L = _frame_looks(uh, t=600.0, spin=spin)
    track = RA.Track((RA.TrackPoint(0.0, 13000.0, 13000.0, True), RA.TrackPoint(300.0, 14000.0, 14000.0, True)))
    assert track.predict(600.0, 1000.0) == pytest.approx((15000.0, 15000.0, RA.TRACK_RADIUS_M))
    storm = RA.find_storm(L, track)
    assert (storm.x, storm.y) == (15500.0, 15500.0) and storm.followed
    assert storm.kind == "rotating updraft" and "rotating updraft" in storm.how
    assert storm.look_at == (12500.0, 13500.0)
    alone = RA.find_storm(L)
    assert (alone.x, alone.y) == (45500.0, 45500.0) and not alone.followed
    lost = RA.Track((RA.TrackPoint(300.0, 55000.0, 5000.0, True),))
    assert lost.predict(600.0, 1000.0)[2] == pytest.approx(RA.TRACK_RADIUS_M + RA.TRACK_FIRST_SPEED * 300.0)
    far = RA.find_storm(L, lost)
    assert (far.x, far.y) == (45500.0, 45500.0) and not far.followed
    # A domain that moves: the track is on the ground, the search in the frame's own metres.
    moved = RA.find_storm(_frame_looks(uh, t=600.0, spin=spin, loc=(5000.0, -2000.0)),
                          RA.Track(tuple(RA.TrackPoint(p.t, p.x + 5000.0, p.y - 2000.0, True) for p in track.points)))
    assert (moved.x, moved.y) == pytest.approx((storm.x, storm.y))
    # A time before with no storm: no guess, the whole area is searched.
    assert RA.Track((RA.TrackPoint(300.0, 0.0, 0.0, False),)).predict(600.0, 1000.0) is None


def test_the_storm_track_is_kept_and_worked_out_again_only_where_the_run_changed(tmp_path):
    calls = []

    def locate(row, track):
        calls.append((row[1], len(track.points)))
        return RA.TrackPoint(row[0], 1000.0 * row[0], 0.0, True)

    keeper = RA.StormTrack(tmp_path)
    rows = [(0.0, "a", None, 0), (300.0, "b", None, 0)]
    track = keeper.before(rows, locate)
    assert calls == [("a", 0), ("b", 1)] and [p.t for p in track.points] == [0.0, 300.0]
    keeper.put(2, "c", RA.TrackPoint(600.0, 5.0, 6.0, True))
    keeper.save()
    again = RA.StormTrack(tmp_path)
    calls.clear()
    assert again.before(rows + [(600.0, "c", None, 0)], locate).points[-1] == RA.TrackPoint(600.0, 5.0, 6.0, True)
    assert calls == []
    # A branch the run left: "b" is now another frame; it and everything after it are worked out again.
    track = again.before([(0.0, "a", None, 0), (300.0, "b2", None, 0)], locate)
    assert calls == [("b2", 1)] and len(again.rows) == 2 and track.points[-1].x == 300000.0


@NEEDS_MATPLOTLIB
def test_the_radar_follows_the_storm_and_a_picture_drawn_later_follows_it_the_same_way(tmp_path):
    """The stored storm state at three saved times in a domain moving at 12.5/3.0 m/s: the same storm on the ground
    moves 3.75 km east and 0.9 km north between them, and so does the radar beside it.  Drawing the last time alone
    (the track worked out again from the frames before it) puts the storm and the radar exactly where the pass in
    order did."""

    run = run_directory(tmp_path)
    _render(run, only=RADAR)
    frames = _products(run)["radar-velocity"]["frames"]
    sites = [f["facts"]["site_km"] for f in frames]
    assert [f["facts"]["storm_followed"] for f in frames] == [False, True, True]
    for a, b in zip(sites, sites[1:]):
        assert (b[0] - a[0], b[1] - a[1]) == pytest.approx((3.75, 0.9), abs=1e-6)
    track = json.loads((run / "quicklook" / ".state" / RA.TRACK_FILE).read_text())
    assert track["schema"] == RA.TRACK_SCHEMA and [r["found"] for r in track["rows"]] == [True, True, True]
    (run / "quicklook" / ".state" / RA.TRACK_FILE).unlink()
    renderer = RD.Renderer(run, only=RADAR, out=tmp_path / "alone")
    lineage = RD.frames_of(renderer.layout)
    record = renderer.draw(lineage[-1], lineage)
    assert record["products"]["radar-velocity"]["facts"]["site_km"] == pytest.approx(sites[-1], abs=1e-9)
    assert record["products"]["radar-velocity"]["facts"]["storm_followed"] is True


# ---------------------------------------------------------------- 4 and 5: the looks on CM1's own storm

@pytest.fixture()
def pictures(monkeypatch):
    drawn = []
    save = P.save

    def keep(ov, base, path):
        drawn.append((Path(path).parent.name, ov))
        return save(ov, base, path)

    monkeypatch.setattr(P, "save", keep)
    return drawn


def _render(run: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return RD.render(run, **kwargs)


def _products(run: Path) -> dict:
    index = json.loads((run / "quicklook" / "index.json").read_text())
    return {p["id"]: p for p in index["products"]}


@NEEDS_MATPLOTLIB
def test_morrison_wsm6_and_thompson_runs_each_get_both_radar_looks(tmp_path, pictures):
    morrison = run_directory(tmp_path, name="morrison")
    dbz = np.clip(np.load(Path(__file__).parent / "data" / "cm1-ref-24x24" / "stats-run-t1500.npz")["qr"] * 3.0e4,
                  0.0, 60.0)
    wsm6 = run_directory(tmp_path, name="wsm6", component="wsm6-mp6", dbz=dbz)
    thompson = run_directory(tmp_path, name="thompson", component="thompson-mp8")
    for run in (morrison, wsm6, thompson):
        summary = _render(run, only=RADAR)
        assert summary.problems == [], (run.name, summary.problems)
        products = _products(run)
        assert {pid: len(products[pid]["frames"]) for pid in RADAR} == {pid: 3 for pid in RADAR}, run.name
    facts = {run.name: _products(run)["radar-storm-relative"]["frames"][-1]["facts"]
             for run in (morrison, wsm6, thompson)}
    assert facts["morrison"]["echo"] == "dbz" and "calc_refl10cm" in facts["morrison"]["operator"]
    assert facts["wsm6"]["echo"] == "dbz" and facts["wsm6"]["operator"].startswith("WSM6's own reflectivity")
    assert facts["thompson"]["echo"] == "mass" and "no reflectivity formula" in facts["thompson"]["operator"]
    for f in facts.values():
        assert f["storm_motion_m_s"] == [12.5, 3.0] and f["storm_motion_from"] == "the domain's motion"
        assert f["beam_at_storm_m"][1] == pytest.approx(RA.ANTENNA_M + float(RA.beam_height_m(
            RA.slant_range_m(40000.0, 0.5), 0.5)), abs=0.5)
    for name, ov in pictures:
        assert name in RADAR
        assert ov.check() == [] and ov.footer_cut == [], name
        text = "\n".join(ov.texts)
        assert RA.WHAT_IT_IS in text and "radar site placed 40 km south-east of the storm" in text
        assert "domain moving at 12.5/3.0 m/s ground-relative" in text
    thompson_texts = ["\n".join(ov.texts) for name, ov in pictures][-2:]
    # The mass it sums, named (the fixture's frames call their dense ice hail; Thompson's own call it graupel)
    assert facts["thompson"]["echo_sums"] == "rain, snow and hail"
    assert all("rain, snow and hail (not reflectivity)" in t for t in thompson_texts)
    assert "echo_sums" not in facts["morrison"] and "echo_sums" not in facts["wsm6"]


@NEEDS_MATPLOTLIB
def test_a_fixed_domain_without_a_recorded_motion_takes_the_bunkers_mover_of_its_first_air(tmp_path, pictures):
    run = run_directory(tmp_path, move=(0.0, 0.0), swath="domain")
    _render(run, only=RADAR)
    facts = _products(run)["radar-storm-relative"]["frames"][-1]["facts"]
    assert facts["storm_motion_from"] == "Bunkers right mover of the first saved time's corner air"
    assert all("domain fixed to the ground" in "\n".join(ov.texts) for _name, ov in pictures)


@NEEDS_MATPLOTLIB
def test_a_run_whose_starting_air_has_little_shear_takes_out_its_mean_wind(tmp_path, pictures):
    """The calm pulse-storm run's own record (0-6 km shear 1.96 m/s): the 0-6 km mean wind is taken out, and the
    picture says it is the mean wind, not a right mover."""

    run = run_directory(tmp_path, move=(0.0, 0.0), swath="domain")
    calm = {"schema": "gm1.sounding-parameters/v1", "bunkers_mean_wind_u_m_per_s": 0.9990302324295044,
            "bunkers_mean_wind_v_m_per_s": 0, "bunkers_right_u_m_per_s": 0.9990302324295044,
            "bunkers_right_v_m_per_s": -7.5, "shear_0_6km_m_per_s": 1.958299994468689}
    (run / "import-report.json").write_text(json.dumps({"sounding": {"environment": calm}}), encoding="utf-8")
    _render(run, only=RADAR)
    facts = _products(run)["radar-storm-relative"]["frames"][-1]["facts"]
    assert facts["storm_motion_key"] == "mean-recorded"
    assert facts["storm_motion_m_s"] == pytest.approx([0.99903, 0.0], abs=1e-5)
    assert facts["storm_motion_from"] == RA.MOTION_WORDS["mean-recorded"][0]
    texts = ["\n".join(ov.texts) for name, ov in pictures if name == "radar-storm-relative"]
    assert texts and all("0-6 km mean wind" in t and "Bunkers" not in t for t in texts)
    for _name, ov in pictures:
        assert ov.check() == [] and ov.footer_cut == []


@NEEDS_MATPLOTLIB
def test_a_radar_set_beyond_its_range_says_so_rather_than_no_rain(tmp_path, pictures):
    run = run_directory(tmp_path, times=(1500.0,))
    RA.write_settings(run / "quicklook", RA.Settings((1000.0, 0.0), 0.5, None))
    _render(run, only=RADAR)
    facts = _products(run)["radar-storm-relative"]["frames"][-1]["facts"]
    assert "beyond its 230 km range" in facts["couplet"]["reason"]
    for _name, ov in pictures:
        text = "\n".join(ov.texts)
        assert "beyond its 230 km range" in text and "no echo on this scan" not in text


def _cli(run: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(run), *args], capture_output=True,
                          text=True, env=env)


@NEEDS_MATPLOTLIB
def test_the_radar_settings_move_the_radar_and_draw_every_frame_again(tmp_path):
    run = run_directory(tmp_path)
    _render(run)
    first = _products(run)["radar-velocity"]["frames"][-1]["facts"]
    assert first["site_chosen"] is False and first["tilt_deg"] == 0.5
    done = _cli(run, "--radar-site", "10", "-5.5", "--radar-tilt", "1.5", "--radar-motion", "11", "4", "--json")
    assert done.returncode == 0, done.stderr
    record = json.loads(done.stdout)
    assert record["settings"] == {"schema": RA.SETTINGS_SCHEMA, "site_km": [10.0, -5.5], "tilt_deg": 1.5,
                                  "motion_m_s": [11.0, 4.0]}
    assert record["frames_drawn"] == 3 and record["pictures_redrawn"] == 6
    assert RA.read_settings(run / "quicklook") == RA.Settings((10.0, -5.5), 1.5, (11.0, 4.0))
    products = _products(run)
    for pid in RADAR:
        for frame in products[pid]["frames"]:
            assert frame["facts"]["site_km"] == pytest.approx([10.0, -5.5])
            assert frame["facts"]["tilt_deg"] == 1.5 and frame["facts"]["site_chosen"] is True
    assert products["radar-storm-relative"]["frames"][-1]["facts"]["storm_motion_m_s"] == [11.0, 4.0]
    # the other looks are not drawn again
    assert len(products["reflectivity"]["frames"]) == 3
    done = _cli(run, "--radar-site", "storm", "--radar-motion", "auto", "--json")
    assert done.returncode == 0, done.stderr
    assert RA.read_settings(run / "quicklook") == RA.Settings(None, 1.5, None)
    assert _products(run)["radar-velocity"]["frames"][-1]["facts"]["site_chosen"] is False
    refused = _cli(run, "--radar-site", "10")
    assert refused.returncode == 2 and "two numbers or the word storm" in refused.stderr
    refused = _cli(run, "--radar-tilt", "45")
    assert refused.returncode == 2 and "tilt must be from 0.1 to 20" in refused.stderr


@NEEDS_MATPLOTLIB
def test_the_radar_of_cm1_output_kept_elsewhere_is_moved_on_its_files(tmp_path):
    """The radar row of CM1 output opened for pictures from where it is kept draws the files' radar looks into the
    entry's own pictures, and leaves the files' folder as it was (no follower's lock: the files are never followed)."""

    from cm1core.gui import commands as C
    from cm1core.gui.files import CM1_OUTPUT_FILE, CM1_OUTPUT_SCHEMA

    source = tmp_path / "data" / "cm1run"
    source.mkdir(parents=True)
    shutil.copy(ROOT / "tests" / "data" / "cm1-ref-24x24" / "cm1out_000001.nc", source / "cm1out_000001.nc")
    rundir = tmp_path / "runs" / "mine"
    rundir.mkdir(parents=True)
    (rundir / CM1_OUTPUT_FILE).write_text(json.dumps({"schema": CM1_OUTPUT_SCHEMA, "source": str(source)}))
    args = C.radar_args(rundir, [10.0, -5.0], 1.0)
    assert args == ["quicklook", str(source), "--out", str(rundir / "quicklook"), "--label", "mine",
                    "--radar-site", "10", "-5", "--radar-tilt", "1", "--json"]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run([sys.executable, "-m", "cm1core", *args], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["settings"]["site_km"] == [10.0, -5.0]
    assert RA.read_settings(rundir / "quicklook").site_km == (10.0, -5.0)
    assert sorted(p.name for p in source.iterdir()) == ["cm1out_000001.nc"]


def test_the_radar_waits_for_a_runs_quicklook_worker(tmp_path):
    from cm1core.cm1out.follow_lock import follow_lock

    run = run_directory(tmp_path, times=(900.0,))
    with follow_lock(run, "quicklook"):
        refused = _cli(run, "--radar-site", "storm")
    assert refused.returncode == 2 and "quick-look worker is drawing its pictures now" in refused.stderr


def test_unreadable_settings_fall_back_to_the_radar_beside_the_storm(tmp_path):
    (tmp_path / RA.SETTINGS_FILE).write_text("{not json", encoding="utf-8")
    assert RA.read_settings(tmp_path) == RA.Settings()
    (tmp_path / RA.SETTINGS_FILE).write_text(json.dumps({"schema": RA.SETTINGS_SCHEMA, "tilt_deg": 90}),
                                             encoding="utf-8")
    assert RA.read_settings(tmp_path) == RA.Settings()
    with pytest.raises(RA.SettingsError):
        RA.Settings(site_km=(1.0, float("nan"))).check()
    with pytest.raises(RA.SettingsError):
        RA.Settings(motion_m_s=(150.0, 0.0)).check()


def test_the_gui_lists_the_radar_looks_and_moves_the_radar_with_a_quicklook_line(tmp_path, monkeypatch):
    from cm1core.gui import commands as C
    from cm1core.gui import looks as GL
    from gui_support import start_server, stop_server

    root = tmp_path / "runs"
    root.mkdir()
    run_directory(root, times=(900.0, 1500.0), name="storm")
    assert C.display(C.radar_args(Path("/runs/storm"), [12.0, -3.25], 0.5)) == \
        "gpuwm-cm1 quicklook /runs/storm --radar-site 12 -3.25 --radar-tilt 0.5 --json"
    assert C.display(C.radar_args(Path("/runs/storm"), "storm", 1.0)) == \
        "gpuwm-cm1 quicklook /runs/storm --radar-site storm --radar-tilt 1 --json"
    assert {row.action: row for row in C.API}["radar"].kind == "launch"
    server, client = start_server(root, tmp_path / "config")
    try:
        results = client.get("/api/runs/storm/results").json
        assert results["radar"]["looks"] == list(RADAR)
        assert results["radar"]["settings"] == {"site_km": None, "tilt_deg": 0.5, "motion_m_s": None}
        assert set(RADAR) <= set(results["film_looks"])
        assert [row["id"] for row in results["more_looks"]][2:4] == list(RADAR)
        answer = client.post("/api/runs/storm/radar", {"site": [12, -3.25], "tilt": 0.5})
        assert answer.status == 200 and answer.json["ok"], answer.json
        assert answer.json["command"] == C.display(C.radar_args(root / "storm", [12.0, -3.25], 0.5))
        assert "gpuwm-cm1 quicklook" in (root / "storm" / "commands.log").read_text()
        for bad in ({"site": [1]}, {"site": "here"}, {"site": [1, 2], "tilt": 0}, {"site": [1, True]}):
            answer = client.post("/api/runs/storm/radar", bad)
            assert answer.status == 400 and answer.json["message"], bad
    finally:
        stop_server(server)
    (root / "storm" / "quicklook").mkdir(exist_ok=True)
    RA.write_settings(root / "storm" / "quicklook", RA.Settings((5.0, 6.0), 2.0, None))
    assert GL.radar_settings(root / "storm") == {"site_km": [5.0, 6.0], "tilt_deg": 2.0, "motion_m_s": None}


def test_the_radar_row_refuses_empty_boxes_and_speaks_of_the_eye_for_a_hurricane(tmp_path):
    """Number("") is 0 in a browser, so an empty x or y once fixed the radar at (0, 0), and every radar picture was
    drawn again: the row now refuses to post until both boxes hold numbers (and the tilt is 0.1 to 20 degrees).  A
    hurricane's row says the radar stands beside the eye, and that it draws its one radar look again."""

    from cm1core.gui import looks as GL

    parts = (ROOT / "src" / "cm1core" / "gui" / "static" / "js" / "run" / "parts.js").read_text(encoding="utf-8")
    # The Results screen builds the film bar again on every refresh: the row is kept while what it shows is the
    # same, so what is being typed in it stays.
    bar = parts[parts.index("export function filmBar("):parts.index("function keptRadarRow(")]
    assert "keptRadarRow(id, data, post, opts)" in bar and "radarRow(data" not in bar
    kept = parts[parts.index("function keptRadarRow("):parts.index("function radarRow(")]
    assert "kept.key === key) return kept.row;" in kept and "radarRows.set(id" in kept
    row = parts[parts.index("function radarRow("):parts.index("// The More looks row")]
    assert "[typed(x), typed(y)]" in row and 'at.includes(null) ? "radar_need_xy"' in row
    assert "if (missing) return;" in row and row.index("if (missing) return;") < row.index('post("radar"')
    assert 'text !== "" && Number.isFinite(value)' in row
    assert 'r.hurricane ? "radar_beside_eye" : "radar_beside"' in row
    assert 'r.hurricane ? "radar_about_one" : "radar_about"' in row
    ui = GL.ui_words()
    for key in ("radar_need_xy", "radar_need_tilt", "radar_beside_eye", "radar_about_one"):
        assert ui[key], key
    assert GL.radar_state(tmp_path, {"run": {"storm_kind": "hurricane"}})["hurricane"] is True
    assert GL.radar_state(tmp_path, {"run": {"storm_kind": "supercell"}})["hurricane"] is False
    assert GL.radar_state(tmp_path, None)["hurricane"] is False
