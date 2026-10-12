"""Synthetic WRF and ArWen history files for the reader, the column and the GUI tests (host only).

``write_wrfout`` writes a small, physically consistent history in any container netCDF4 writes (CDF-1, CDF-2,
CDF-5, netCDF-4): a horizontally uniform atmosphere whose truth is known in closed form, placed on WRF's grid
exactly as WRF lays it out (C grid, geopotential on w levels, T = dry theta - 300, P+PB, SINALPHA/COSALPHA,
the surface fields), so a test can compare what a reader extracts with the numbers that went in.

The atmosphere (heights above ground):

* theta = 300 K + 4 K/km up to 12 km, then 348 K + 20 K/km (a tropopause);
* qv = 14 g/kg * exp(-z / 2.5 km), floored at 1e-6;
* earth-relative wind u = 20 m/s * tanh(z / 3 km) + record, v = 5 m/s + 6 m/s * sin(z / 4 km);
* pressure: CM1's Exner hydrostatics (g 9.81, cp 1005.7, rd 287.04, CM1's virtual theta) integrated from the
  surface pressure on a 0.1 m grid, so CM1's own reconstruction of the written sounding must match it.

Grid-relative winds are the earth winds rotated by the file's alpha (u = ue cos a + ve sin a, v = ve cos a - ue sin a),
plus ``wind_slope`` * the staggered index, so destaggering a linear field can be checked exactly.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np

G = 9.81
RD, CP, RV, P00 = 287.04, 1005.7, 461.5, 1.0e5
REPS = RV / RD
WRF_RV_OVER_RD = 461.6 / 287.0
ZTOP = 20500.0

FORMATS = {"CDF-1": "NETCDF3_CLASSIC", "CDF-2": "NETCDF3_64BIT_OFFSET", "CDF-5": "NETCDF3_64BIT_DATA",
           "netCDF-4": "NETCDF4"}


def theta(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z < 12000.0, 300.0 + 0.004 * z, 348.0 + 0.02 * (z - 12000.0))


def qv(z):
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(0.014 * np.exp(-z / 2500.0), 1.0e-6)


def wind(z, record=0):
    z = np.asarray(z, dtype=np.float64)
    return 20.0 * np.tanh(z / 3000.0) + record, 5.0 + 6.0 * np.sin(z / 4000.0)


def thv(z):
    q = qv(z)
    return theta(z) * (1.0 + REPS * q) / (1.0 + q)


_FINE_Z = np.linspace(0.0, 25000.0, 250001)
_FINE_F: list[np.ndarray] = []


def _fine_f() -> np.ndarray:
    """F(z) = -g/cp * integral of 1/theta_v from 0 to z, on a 0.1 m grid (trapezoid), computed once."""

    if not _FINE_F:
        inv = 1.0 / thv(_FINE_Z)
        dz = np.diff(_FINE_Z)
        cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * dz)])
        _FINE_F.append(-G / CP * cumulative)
    return _FINE_F[0]


def exner(z, psfc_pa):
    """CM1's hydrostatic Exner function at heights z (m above ground): pi_sfc - g/cp * int(1/theta_v dz).

    d(pi)/dz = -g/(cp theta_v) depends only on the height above ground here, so one fine integral serves every
    surface pressure.
    """

    z = np.asarray(z, dtype=np.float64)
    return (np.asarray(psfc_pa, dtype=np.float64) / P00) ** (RD / CP) + np.interp(z, _FINE_Z, _fine_f())


def pressure(z, psfc_pa):
    return P00 * exner(z, psfc_pa) ** (CP / RD)


def stamp(start: str, seconds: float) -> str:
    base = dt.datetime.strptime(start, "%Y-%m-%d_%H:%M:%S")
    return (base + dt.timedelta(seconds=seconds)).strftime("%Y-%m-%d_%H:%M:%S")


def write_wrfout(path: Path, *, fmt: str = "CDF-2", producer: str = "ArWen", nx: int = 8, ny: int = 7, nz: int = 24,
                 records: int = 3, start: str = "2026-05-01_18:00:00", interval_s: float = 3600.0,
                 map_proj: int = 1, alpha_deg: float = 0.0, use_theta_m: int | None = 0, thm: str = "dry",
                 terrain_m: float = 0.0, wind_slope: float = 0.0, psfc_pa: float = 97000.0, dx: float = 3000.0,
                 drop: tuple[str, ...] = (), grid_id: int | None = 1, lat0: float = 35.0, lon0: float = -97.0,
                 mp_physics: int = 6, with_th2: bool = True, title: str | None = None) -> dict[str, Any]:
    """Write a synthetic history at ``path``; return the facts a test compares against.

    ``thm``: "dry" (T is dry theta and THM is consistent), "moist" (T carries moist theta: the inconsistency the
    reader must refuse), or "none" (no THM variable).  ``terrain_m``: the terrain rises linearly west to east
    from 0 to this height.  ``drop``: variables left out.
    """

    import netCDF4

    path = Path(path)
    k = np.arange(nz + 1, dtype=np.float64)
    s = (k / nz) ** 1.35
    hgt = (terrain_m * np.arange(nx) / max(nx - 1, 1))[None, :] + np.zeros((ny, 1))
    hgt32 = hgt.astype(np.float32)
    # WRF: the top face is a constant pressure surface; here a constant height above sea level.
    zmsl_w = hgt32.astype(np.float64)[None] + (ZTOP - hgt32.astype(np.float64))[None] * s[:, None, None]
    ph_pert = G * 3.0 * np.sin(np.pi * k / nz)[:, None, None] + np.zeros((1, ny, nx))
    phb = (G * zmsl_w - ph_pert).astype(np.float32)
    ph = ph_pert.astype(np.float32)
    zw = (ph.astype(np.float64) + phb.astype(np.float64)) / G - hgt32.astype(np.float64)[None]
    zm = 0.5 * (zw[:-1] + zw[1:])
    # Every column carries the same profile of height above ground; the surface pressure falls with the terrain.
    psfc = psfc_pa * np.exp(-hgt32.astype(np.float64) / 8000.0)
    p = pressure(zm, psfc[None])
    pb = P00 * np.exp(-(zm + hgt32[None]) / 7500.0)
    th = theta(zm)
    q = qv(zm)
    alpha = math.radians(alpha_deg)
    ca, sa = math.cos(alpha), math.sin(alpha)
    # u points: heights the mean of the two mass columns (edges copy); v points likewise.
    zu = np.concatenate([zm[:, :, :1], 0.5 * (zm[:, :, :-1] + zm[:, :, 1:]), zm[:, :, -1:]], axis=2)
    zv = np.concatenate([zm[:, :1, :], 0.5 * (zm[:, :-1, :] + zm[:, 1:, :]), zm[:, -1:, :]], axis=1)
    lat = (lat0 + (np.arange(ny) - (ny - 1) / 2)[:, None] * dx / 111000.0 + np.zeros((1, nx))).astype(np.float32)
    lon = (lon0 + (np.arange(nx) - (nx - 1) / 2)[None, :] * dx / (111000.0 * math.cos(math.radians(lat0)))
           + np.zeros((ny, 1))).astype(np.float32)
    t2 = theta(0.0) * (psfc / P00) ** (RD / CP)

    fmt_name = FORMATS[fmt]
    ds = netCDF4.Dataset(str(path), "w", format=fmt_name)
    try:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("west_east", nx)
        ds.createDimension("south_north", ny)
        ds.createDimension("bottom_top", nz)
        ds.createDimension("west_east_stag", nx + 1)
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("bottom_top_stag", nz + 1)
        attrs: dict[str, Any] = {
            "TITLE": title if title is not None else (
                "gpuwm synthetic history" if producer == "ArWen" else " OUTPUT FROM WRF V4.6.1 MODEL"),
            "DX": np.float32(dx), "DY": np.float32(dx), "MAP_PROJ": np.int32(map_proj),
            "TRUELAT1": np.float32(30.0), "TRUELAT2": np.float32(60.0), "STAND_LON": np.float32(lon0),
            "CEN_LAT": np.float32(lat0), "CEN_LON": np.float32(lon0), "MP_PHYSICS": np.int32(mp_physics),
            "SIMULATION_START_DATE": start, "START_DATE": start, "HYBRID_OPT": np.int32(2),
        }
        if producer == "ArWen":
            attrs["GPUWM_VERSION"] = "2.8.9"
        if use_theta_m is not None:
            attrs["USE_THETA_M"] = np.int32(use_theta_m)
        if grid_id is not None:
            attrs["GRID_ID"] = np.int32(grid_id)
        ds.setncatts(attrs)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        stamps = [stamp(start, r * interval_s) for r in range(records)]
        for r, text in enumerate(stamps):
            times[r] = np.array(list(text), dtype="S1")

        def var(name, dims, data_by_record, stagger="", units=""):
            if name in drop:
                return
            v = ds.createVariable(name, "f4", ("Time", *dims))
            v.setncattr("stagger", stagger)
            if units:
                v.setncattr("units", units)
            for r in range(records):
                v[r] = np.asarray(data_by_record(r), dtype=np.float32)

        m3 = ("bottom_top", "south_north", "west_east")
        m2 = ("south_north", "west_east")

        def grid_winds(zz, r, axis_index, stagger_axis):
            ue, ve = wind(zz, r)
            ug = ue * ca + ve * sa
            vg = ve * ca - ue * sa
            return ug, vg

        def u_field(r):
            ug, _ = grid_winds(zu, r, None, None)
            return ug + wind_slope * np.arange(nx + 1)[None, None, :]

        def v_field(r):
            _, vg = grid_winds(zv, r, None, None)
            return vg + wind_slope * np.arange(ny + 1)[None, :, None]

        var("U", ("bottom_top", "south_north", "west_east_stag"), u_field, "X", "m s-1")
        var("V", ("bottom_top", "south_north_stag", "west_east"), v_field, "Y", "m s-1")
        var("W", ("bottom_top_stag", "south_north", "west_east"), lambda r: np.zeros((nz + 1, ny, nx)), "Z")
        moist_theta = th * (1.0 + WRF_RV_OVER_RD * q)
        var("T", m3, lambda r: (moist_theta if thm == "moist" else th) - 300.0, "", "K")
        if thm in ("dry", "moist"):
            var("THM", m3, lambda r: moist_theta - 300.0, "", "K")
        var("PH", ("bottom_top_stag", "south_north", "west_east"), lambda r: ph, "Z")
        var("PHB", ("bottom_top_stag", "south_north", "west_east"), lambda r: phb, "Z")
        var("P", m3, lambda r: p - pb, "", "Pa")
        var("PB", m3, lambda r: pb, "", "Pa")
        var("QVAPOR", m3, lambda r: q, "", "kg kg-1")
        var("QCLOUD", m3, lambda r: np.zeros_like(q), "")
        var("HGT", m2, lambda r: hgt32, "", "m")
        var("XLAT", m2, lambda r: lat, "", "degree_north")
        var("XLONG", m2, lambda r: lon, "", "degree_east")
        var("MAPFAC_M", m2, lambda r: np.ones((ny, nx)))
        if map_proj in (1, 2):
            var("SINALPHA", m2, lambda r: np.full((ny, nx), sa))
            var("COSALPHA", m2, lambda r: np.full((ny, nx), ca))
        var("PSFC", m2, lambda r: psfc, "", "Pa")
        var("T2", m2, lambda r: t2, "", "K")
        if with_th2:
            var("TH2", m2, lambda r: np.full((ny, nx), theta(0.0)), "", "K")
        var("Q2", m2, lambda r: np.full((ny, nx), float(qv(0.0))), "", "kg kg-1")
        ue10, ve10 = wind(10.0, 0)
        var("U10", m2, lambda r: np.full((ny, nx), (ue10 + r) * ca + ve10 * sa))
        var("V10", m2, lambda r: np.full((ny, nx), ve10 * ca - (ue10 + r) * sa))
        # A static (non-record) variable, as real files carry.
        if "ZNU" not in drop:
            znu = ds.createVariable("ZNU", "f4", ("Time", "bottom_top"))
            for r in range(records):
                znu[r] = np.linspace(1.0, 0.0, nz).astype(np.float32)
        if "P_TOP" not in drop:
            ptop = ds.createVariable("P_TOP", "f4", ("Time",))
            for r in range(records):
                ptop[r] = np.float32(5000.0)
    finally:
        ds.close()
    return {
        "path": path, "times": stamps, "nx": nx, "ny": ny, "nz": nz, "alpha": alpha, "hgt": hgt32,
        "zw": zw, "zm": zm, "p": p, "psfc": psfc, "lat": lat, "lon": lon, "t2": t2, "wind_slope": wind_slope,
    }


def earth_winds_at_mass(facts: dict[str, Any], record: int, j: int, i: int) -> tuple[np.ndarray, np.ndarray]:
    """The exact earth-relative wind a reader must return at mass point (j, i) on flat ground."""

    zm = facts["zm"][:, j, i]
    ue, ve = wind(zm, record)
    a = facts["alpha"]
    ca, sa = math.cos(a), math.sin(a)
    ug = ue * ca + ve * sa + facts["wind_slope"] * (i + 0.5)
    vg = ve * ca - ue * sa + facts["wind_slope"] * (j + 0.5)
    return ug * ca - vg * sa, vg * ca + ug * sa


def cm1_reconstructed_pressure(sounding_text: str) -> tuple[np.ndarray, np.ndarray]:
    """base.F:627-635 on a written input_sounding: (heights incl. the surface at 0, pressures in Pa)."""

    rows = [line.split() for line in sounding_text.strip().splitlines()]
    p_sfc, th_sfc, qv_sfc = (float(x) for x in rows[0])
    z = [0.0] + [float(r[0]) for r in rows[1:]]
    th = [th_sfc] + [float(r[1]) for r in rows[1:]]
    q = [qv_sfc / 1000.0] + [float(r[2]) / 1000.0 for r in rows[1:]]
    tv = [t * (1.0 + REPS * qq) / (1.0 + qq) for t, qq in zip(th, q)]
    pi = [(p_sfc * 100.0 / P00) ** (RD / CP)]
    for n in range(1, len(z)):
        pi.append(pi[-1] - G * (z[n] - z[n - 1]) / (CP * 0.5 * (tv[n] + tv[n - 1])))
    return np.array(z), P00 * np.array(pi) ** (CP / RD)
