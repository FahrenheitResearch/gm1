"""One column of a WRF or ArWen run as CM1's ``input_sounding``: the column route of WRF/ArWen downscaling.

WHAT THIS IS, AND IS NOT
------------------------
The environment of one model column -- potential temperature, water vapour and the earth-relative wind on the
column's own levels, with its surface pressure, potential temperature and vapour -- written in CM1 r21.1's
``isnd = 7`` layout (``base.F:495-607``).  A GM1 run on it is an idealized storm: the same air everywhere in the
domain, flat ground, and no weather coming in from the sides.  It is not a downscaling of the run's weather; that
is the 3D route (:mod:`cm1core.arwen_chain`, "Downscale a WRF or ArWen run (3D)").

THE RULES (every one recorded in ``wrf-column.json``)
----------------------------------------------------
* **The point:** a latitude/longitude becomes the nearest mass point by great-circle distance over ``XLAT`` and
  ``XLONG``; a grid point is given as 1-based (i, j), i west-east and j south-north, as WRF numbers them.
* **Heights** above ground: w levels ``z_w = (PH + PHB)/g - HGT`` with WRF's g = 9.81; a mass level is the mean of
  its two faces (as :func:`cm1core.arwen_chain.column_heights` places them).
* **theta = T + 300.**  T is the dry potential-temperature perturbation in both ArWen
  (``gpuwm/io/wrfout.py``: ``T = total_theta - 300``) and WRF histories.  With ``USE_THETA_M = 1``, ``THM`` must
  equal ``(T + 300)(1 + Rv/Rd qv) - 300`` within 0.01 K, as the chain checks; otherwise the file is refused.
* **qv** from ``QVAPOR`` (kg/kg), written in g/kg; a negative value (a numerical undershoot) is written as 0 and
  counted.
* **Winds:** ``U`` and ``V`` averaged from their two faces to the mass point, then rotated from the grid to the
  earth: ``u_e = u cos(a) - v sin(a)``, ``v_e = v cos(a) + u sin(a)`` with the point's ``SINALPHA``/``COSALPHA``
  (WRF's ``module_diag_*`` rule).  Without them the grid is taken as earth-aligned only for ``MAP_PROJ`` 0
  (idealized) and 3 (Mercator); a Lambert or polar grid without them is refused.
* **Surface record:** ``PSFC`` in hPa; theta from ``TH2``, else ``T2 (p00/PSFC)^(Rd/cp)`` with CM1's constants;
  qv from ``Q2``.  Missing surface fields are filled by named rules: PSFC by CM1's Exner hydrostatics one step
  down from the lowest level (``pi_sfc = pi_1 + g z_1 / (cp thv_1)``, which needs P+PB); theta and qv by the lowest
  level's.  ``U10``/``V10`` are reported, rotated, but CM1's file has no surface wind (``base.F:617-618``
  extrapolates it from the first two levels).
* **Checks:** the heights rise; the top reaches a deck's highest scalar level when one is named.  The file carries
  no comment line: CM1 reads it list-directed and stops on any text (``base.F:546-557``), so the provenance is
  ``wrf-column.json`` beside it.
* **The hydrostatic check:** CM1's own reconstruction of pressure from the written file (``base.F:627-635``) is
  compared with the column's ``P + PB`` at every level and the largest difference recorded.

Nothing here opens a GPU; numpy and netCDF4 only (through :mod:`cm1core.wrf_history`).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..wrf_history import WrfHistory

SCHEMA = "gm1.wrf-column/v1"
GATE = "wrf-column"
#: WRF share/module_model_constants.F: g.  Heights come from WRF's own geopotential, so WRF's g divides it.
WRF_G = 9.81
WRF_RV_OVER_RD = 461.6 / 287.0
#: CM1 r21.1 constants.incl (the file is read by CM1's base.F).
CM1_G, CM1_RD, CM1_CP, CM1_RV, CM1_P00 = 9.81, 287.04, 1005.7, 461.5, 1.0e5
CM1_REPS = CM1_RV / CM1_RD
THETA_M_TOLERANCE_K = 1.0e-2
#: The projections whose grid axes are earth-aligned, so a missing SINALPHA/COSALPHA means no rotation.
EARTH_ALIGNED = {0: "idealized (no map)", 3: "Mercator"}
REQUIRED = ("U", "V", "T", "PH", "PHB", "HGT", "QVAPOR")

COVERAGE_NOTE = (
    "This uses one column of your run as the environment: the same air everywhere in the domain, flat ground, "
    "and no weather coming in from the sides. For a GM1 domain driven by your whole run, use Downscale a WRF or "
    "ArWen run (3D) in Advanced."
)

REFUSAL_CODES = {
    "GM1-WRF-POINT": "the point is outside the domain, or the file has no coordinates to find it by",
    "GM1-WRF-VARIABLE": "a field the column needs is not in the file",
    "GM1-WRF-THETA-M": "T may carry moist potential temperature",
    "GM1-WRF-ROTATION": "the grid is rotated from the earth and the file carries no SINALPHA/COSALPHA",
    "GM1-WRF-HEIGHTS": "the column's level heights do not rise",
    "GM1-WRF-TOP": "the column ends below the model's highest level",
    "GM1-WRF-SURFACE": "no surface pressure can be found or derived",
}


def _refuse(code: str, reason: str, remedy: str, *, option: str | None = None, value: object = None) -> None:
    from ..errors import refuse

    if code not in REFUSAL_CODES:
        raise KeyError(code)
    refuse(code=code, gate=GATE, reason=reason, remedy=remedy, option=option, value=value)


def _np() -> Any:
    import numpy

    return numpy


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    np = _np()
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def parse_pair(text: str, what: str) -> tuple[float, float]:
    """``"A,B"`` (or ``"A B"``) as two numbers."""

    parts = str(text).replace(",", " ").replace("\u2212", "-").split()
    try:
        if len(parts) != 2:
            raise ValueError
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"{what} must be two numbers, as in 35.2,-97.4 or 150,120; got {text!r}") from None


class _Fields:
    """Whole fields of one record, each read from the file once (a map of many columns reuses them)."""

    def __init__(self, history: WrfHistory, record: int) -> None:
        self.history, self.record, self.cache = history, record, {}

    def __call__(self, name: str) -> Any:
        if name not in self.cache:
            self.cache[name] = self.history.array(name, self.record)
        return self.cache[name]

    def point(self, name: str, j: int, i: int) -> float | None:
        return float(self(name)[j, i]) if self.history.has(name) else None


def locate_point(history: WrfHistory, record: int, *, at: tuple[float, float] | None = None,
                 ij: tuple[int, int] | None = None, fields: "_Fields | None" = None) -> dict[str, Any]:
    """The mass point: from (lat, lon) by great-circle distance, or from a 1-based (i, j)."""

    np = _np()
    get = fields or _Fields(history, record)
    dims = history.dimensions
    nx, ny = int(dims.get("west_east", 0)), int(dims.get("south_north", 0))
    has_coords = history.has("XLAT") and history.has("XLONG")
    lat = get("XLAT").astype(np.float64) if has_coords else None
    lon = get("XLONG").astype(np.float64) if has_coords else None
    if (at is None) == (ij is None):
        raise ValueError("give exactly one of at=(lat, lon) or ij=(i, j)")
    if ij is not None:
        i, j = int(ij[0]), int(ij[1])
        if not (1 <= i <= nx and 1 <= j <= ny):
            _refuse("GM1-WRF-POINT",
                    f"grid point i={i}, j={j} is outside the {nx} x {ny} mass grid (1-based, i west-east, "
                    "j south-north)", f"give i from 1 to {nx} and j from 1 to {ny}", option="ij", value=[i, j])
        row: dict[str, Any] = {"requested": {"i": i, "j": j}, "i": i, "j": j, "distance_km": 0.0}
        if lat is not None:
            row["lat"], row["lon"] = float(lat[j - 1, i - 1]), float(lon[j - 1, i - 1])
        return row
    want_lat, want_lon = float(at[0]), float(at[1])
    if lat is None or float(np.ptp(lat)) == 0.0 or float(np.ptp(lon)) == 0.0:
        _refuse("GM1-WRF-POINT",
                "the file has no usable latitudes and longitudes (an idealized run's XLAT/XLONG are one value), "
                "so a latitude and longitude cannot be found on its grid",
                "give a grid point instead (i,j, 1-based)", option="at", value=[want_lat, want_lon])
    distance = haversine_km(want_lat, want_lon, lat, lon)
    flat = int(np.argmin(distance))
    j0, i0 = divmod(flat, nx)
    nearest = float(distance[j0, i0])
    dx_km = float(history.attribute("DX", 0.0) or 0.0) / 1000.0
    dy_km = float(history.attribute("DY", dx_km * 1000.0) or 0.0) / 1000.0
    limit = max(dx_km, dy_km, 1.0e-3)
    edge = i0 in (0, nx - 1) or j0 in (0, ny - 1)
    if nearest > limit or (edge and nearest > 0.5 * limit):
        _refuse("GM1-WRF-POINT",
                f"{want_lat:.4f}, {want_lon:.4f} is outside this domain: its nearest mass point is "
                f"{nearest:.1f} km away (grid spacing {dx_km:g} km); latitudes {lat.min():.2f} to {lat.max():.2f}, "
                f"longitudes {lon.min():.2f} to {lon.max():.2f}",
                "pick a point inside the domain", option="at", value=[want_lat, want_lon])
    return {"requested": {"lat": want_lat, "lon": want_lon}, "i": i0 + 1, "j": j0 + 1,
            "lat": float(lat[j0, i0]), "lon": float(lon[j0, i0]), "distance_km": nearest}


@dataclass
class WrfColumn:
    source: dict[str, Any]
    point: dict[str, Any]
    z_m: Any                     # mass-level heights above ground
    theta_k: Any
    qv_kgkg: Any
    u_ms: Any                    # earth-relative
    v_ms: Any
    p_pa: Any | None             # P + PB at the mass levels, when the file has them
    surface: dict[str, Any]
    rotation: dict[str, Any]
    theta_m: dict[str, Any]
    model_top_face_m: float
    negative_qv_levels: int = 0
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def levels(self) -> int:
        return int(len(self.z_m))

    @property
    def top_m(self) -> float:
        return float(self.z_m[-1])

    def sounding_text(self) -> str:
        """CM1 r21.1 ``input_sounding`` (``isnd = 7``): the surface record, then z, theta, qv (g/kg), u, v."""

        s = self.surface
        lines = [f"{s['pressure_hpa']:12.4f} {s['theta_k']:14.6f} {s['qv_gkg']:13.6f}"]
        for z, th, q, u, v in zip(self.z_m, self.theta_k, self.qv_kgkg, self.u_ms, self.v_ms):
            lines.append(f"{float(z):12.4f} {float(th):14.6f} {max(float(q), 0.0) * 1000.0:13.6f} "
                         f"{float(u):10.4f} {float(v):10.4f}")
        return "\n".join(lines) + "\n"

    def sounding_bytes(self) -> bytes:
        return self.sounding_text().encode("ascii")

    def hydrostatic(self) -> dict[str, Any] | None:
        """CM1's reconstruction (``base.F:627-635``) of the written file against the column's P+PB."""

        if self.p_pa is None:
            return None
        np = _np()
        rows = [line.split() for line in self.sounding_text().strip().splitlines()]
        p_sfc, th_sfc, qv_sfc = (float(x) for x in rows[0])
        z = np.array([0.0] + [float(r[0]) for r in rows[1:]])
        th = np.array([th_sfc] + [float(r[1]) for r in rows[1:]])
        q = np.array([qv_sfc] + [float(r[2]) for r in rows[1:]]) / 1000.0
        tv = th * (1.0 + CM1_REPS * q) / (1.0 + q)
        pi = np.empty_like(z)
        pi[0] = (p_sfc * 100.0 / CM1_P00) ** (CM1_RD / CM1_CP)
        for n in range(1, z.size):
            pi[n] = pi[n - 1] - CM1_G * (z[n] - z[n - 1]) / (CM1_CP * 0.5 * (tv[n] + tv[n - 1]))
        rebuilt = CM1_P00 * pi[1:] ** (CM1_CP / CM1_RD)
        diff = (rebuilt - np.asarray(self.p_pa, dtype=np.float64)) / 100.0
        worst = int(np.argmax(np.abs(diff)))
        return {"rule": "CM1 r21.1 base.F:627-635 from the written file, against the column's P+PB",
                "max_abs_hpa": float(abs(diff[worst])), "at_m": float(z[worst + 1]),
                "at_level": worst + 1, "surface_hpa": p_sfc, "top_diff_hpa": float(diff[-1])}

    def record(self, *, sounding_file: str = "input_sounding", command: str | None = None) -> dict[str, Any]:
        data = self.sounding_bytes()
        return {
            "schema": SCHEMA,
            "source": dict(self.source),
            "point": dict(self.point),
            "levels": self.levels,
            "first_level_m": float(self.z_m[0]),
            "top_m": self.top_m,
            "model_top_face_m": self.model_top_face_m,
            "surface": dict(self.surface),
            "rotation": dict(self.rotation),
            "theta_m": dict(self.theta_m),
            "negative_qv_levels": self.negative_qv_levels,
            "hydrostatic": self.hydrostatic(),
            "checks": list(self.checks),
            "sounding": {"file": sounding_file, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                         "format": "CM1 r21.1 isnd=7 input_sounding"},
            "rules": RULES,
            "coverage_note": COVERAGE_NOTE,
            "command": command,
        }


RULES = {
    "point": "nearest mass point by great-circle distance over XLAT/XLONG, or a 1-based (i, j)",
    "heights": "z_w = (PH+PHB)/9.81 - HGT (WRF g); a mass level is the mean of its two faces",
    "theta": "T + 300 (dry potential temperature in ArWen and WRF histories); with USE_THETA_M = 1, THM is "
             "checked to equal (T+300)(1+Rv/Rd qv) - 300 within 0.01 K",
    "qv": "QVAPOR in g/kg; a negative value is written as 0",
    "winds": "U and V averaged from their faces to the mass point, rotated to the earth with SINALPHA/COSALPHA: "
             "u_e = u cos a - v sin a, v_e = v cos a + u sin a (identity only for MAP_PROJ 0 and 3 without them)",
    "surface": "PSFC (hPa); theta from TH2, else T2 (p00/PSFC)^(Rd/cp) with CM1's constants; qv from Q2. Missing: "
               "PSFC from CM1's Exner hydrostatics one step below the lowest level; theta and qv from the lowest "
               "level",
    "file": "CM1 r21.1 base.F:495-607 isnd = 7 layout, no comment line (CM1 stops on text); provenance in "
            "wrf-column.json",
}


def extract_column(history: WrfHistory, record: int, *, at: tuple[float, float] | None = None,
                   ij: tuple[int, int] | None = None, need_top_m: float | None = None,
                   fields: _Fields | None = None) -> WrfColumn:
    """The column at a point of record ``record``; refuses (``CM1Refusal``) with the reason and what to do.

    ``fields``: whole fields already read from this record (a map of many columns passes one).
    """

    np = _np()
    history.check_record(record)
    get = fields if fields is not None else _Fields(history, record)
    for name in REQUIRED:
        if not history.has(name):
            _refuse("GM1-WRF-VARIABLE",
                    f"{history.path.name} has no {name}; the column needs "
                    f"{', '.join(REQUIRED)} (a standard wrfout carries all of them)",
                    f"write {name} to the history file", option=name, value=history.path.name)
    point = locate_point(history, record, at=at, ij=ij, fields=get)
    i, j = point["i"] - 1, point["j"] - 1
    hgt = float(get("HGT")[j, i])
    ph = get("PH")[:, j, i].astype(np.float64)
    phb = get("PHB")[:, j, i].astype(np.float64)
    zw = (ph + phb) / WRF_G - hgt
    if not np.all(np.diff(zw) > 0.0) or zw[1] <= 0.0:
        _refuse("GM1-WRF-HEIGHTS",
                f"the column's level heights (PH+PHB)/g - HGT do not rise from the ground at i={i + 1}, j={j + 1} "
                f"(first faces {zw[:3].round(1).tolist()} m); CM1 brackets each model level by the first sounding "
                "height above it, so a height that does not rise interpolates between the wrong levels",
                "check that PH and PHB are the geopotential of this record", option="heights",
                value=[i + 1, j + 1])
    zm = 0.5 * (zw[:-1] + zw[1:])
    point["terrain_m"] = hgt
    theta = get("T")[:, j, i].astype(np.float64) + 300.0
    qv = get("QVAPOR")[:, j, i].astype(np.float64)
    theta_m: dict[str, Any] = {"USE_THETA_M": history.attribute("USE_THETA_M")}
    use_theta_m = theta_m["USE_THETA_M"]
    if use_theta_m is not None and int(use_theta_m) != 0:
        if not history.has("THM"):
            _refuse("GM1-WRF-THETA-M",
                    "the file declares USE_THETA_M = 1 and carries no THM, so T cannot be shown to be the dry "
                    "potential temperature CM1's sounding holds; a moist theta would warm every moist level by "
                    "about 0.6 qv theta",
                    "write THM beside T, or run with use_theta_m = 0", option="USE_THETA_M", value=use_theta_m)
        thm = get("THM")[:, j, i].astype(np.float64) + 300.0
        residual = float(np.max(np.abs(theta * (1.0 + WRF_RV_OVER_RD * qv) - thm)))
        theta_m["dry_T_residual_K"] = residual
        if residual > THETA_M_TOLERANCE_K:
            _refuse("GM1-WRF-THETA-M",
                    f"with USE_THETA_M = 1, (T+300)(1+Rv/Rd qv) differs from THM+300 by {residual:.3g} K in this "
                    "column, so T is not the dry potential temperature; the sounding would start moist levels warm",
                    "write the dry potential temperature as T", option="USE_THETA_M", value=residual)
    negative = int(np.sum(qv < 0.0))
    qv = np.maximum(qv, 0.0)

    u_faces = get("U")[:, j, i:i + 2].astype(np.float64)
    v_faces = get("V")[:, j:j + 2, i].astype(np.float64)
    ug = 0.5 * (u_faces[:, 0] + u_faces[:, 1])
    vg = 0.5 * (v_faces[:, 0] + v_faces[:, 1])
    map_proj = history.attribute("MAP_PROJ")
    if history.has("SINALPHA") and history.has("COSALPHA"):
        sa = float(get("SINALPHA")[j, i])
        ca = float(get("COSALPHA")[j, i])
        rotation = {"sinalpha": sa, "cosalpha": ca, "alpha_deg": math.degrees(math.atan2(sa, ca)),
                    "source": "SINALPHA/COSALPHA at the point"}
    elif map_proj is not None and int(map_proj) in EARTH_ALIGNED:
        sa, ca = 0.0, 1.0
        rotation = {"sinalpha": 0.0, "cosalpha": 1.0, "alpha_deg": 0.0,
                    "source": f"none: MAP_PROJ = {int(map_proj)} ({EARTH_ALIGNED[int(map_proj)]}) grid axes are "
                              "earth-aligned"}
    else:
        _refuse("GM1-WRF-ROTATION",
                f"the file's grid (MAP_PROJ = {map_proj}) is rotated from the earth and it carries no SINALPHA/"
                "COSALPHA, so its U and V cannot be turned into earth-relative winds; taking them as earth winds "
                "would turn the hodograph by the local rotation angle",
                "write SINALPHA and COSALPHA to the history (WRF writes them by default)", option="MAP_PROJ",
                value=map_proj)
    u_e = ug * ca - vg * sa
    v_e = vg * ca + ug * sa

    p = None
    if history.has("P") and history.has("PB"):
        p = (get("P")[:, j, i].astype(np.float64)
             + get("PB")[:, j, i].astype(np.float64))
    sources: dict[str, str] = {}
    psfc = get.point("PSFC", j, i)
    if psfc is not None and psfc > 0.0:
        sources["pressure"] = "PSFC"
    elif p is not None:
        thv1 = theta[0] * (1.0 + CM1_REPS * qv[0]) / (1.0 + qv[0])
        pi1 = (p[0] / CM1_P00) ** (CM1_RD / CM1_CP)
        psfc = CM1_P00 * (pi1 + CM1_G * zm[0] / (CM1_CP * thv1)) ** (CM1_CP / CM1_RD)
        sources["pressure"] = "derived: CM1 Exner hydrostatics one step below the lowest level (no PSFC)"
    else:
        _refuse("GM1-WRF-SURFACE",
                "the file has no PSFC and no P/PB, so the surface pressure CM1's sounding starts from cannot be read "
                "or derived",
                "write PSFC (or P and PB) to the history file", option="PSFC", value=None)
    th2 = get.point("TH2", j, i)
    t2 = get.point("T2", j, i)
    if th2 is not None and th2 > 0.0:
        theta_sfc = th2
        sources["theta"] = "TH2"
    elif t2 is not None and t2 > 0.0:
        theta_sfc = t2 * (CM1_P00 / psfc) ** (CM1_RD / CM1_CP)
        sources["theta"] = "T2 (p00/PSFC)^(Rd/cp), CM1 constants"
    else:
        theta_sfc = float(theta[0])
        sources["theta"] = "derived: the lowest level's theta (no TH2 or T2)"
    q2 = get.point("Q2", j, i)
    if q2 is not None and q2 >= 0.0:
        qv_sfc = q2
        sources["qv"] = "Q2"
    else:
        qv_sfc = float(qv[0])
        sources["qv"] = "derived: the lowest level's qv (no Q2)"
    surface: dict[str, Any] = {"pressure_hpa": float(psfc) / 100.0, "theta_k": float(theta_sfc),
                               "qv_gkg": float(qv_sfc) * 1000.0, "sources": sources}
    u10, v10 = get.point("U10", j, i), get.point("V10", j, i)
    if u10 is not None and v10 is not None:
        surface["u10_ms"] = u10 * ca - v10 * sa
        surface["v10_ms"] = v10 * ca + u10 * sa
        surface["wind_10m_note"] = "reported only: CM1's input_sounding has no surface wind"

    checks = [{"state": "ok", "text": f"The heights rise from {zm[0]:.0f} m to {zm[-1]:,.0f} m above the ground."}]
    if need_top_m is not None and float(zm[-1]) < float(need_top_m):
        _refuse("GM1-WRF-TOP",
                f"the column's highest level is {zm[-1]:,.0f} m above the ground and the model's highest scalar "
                f"level is {need_top_m:,.0f} m; CM1 r21.1 stops on such a file (base.F:678-683, 'zmax of sounding "
                "< zmax of grid')",
                f"lower the model top below {zm[-1]:,.0f} m (ztop or nz), or use a run whose top is higher",
                option="top", value=float(zm[-1]))
    if need_top_m is not None:
        checks.append({"state": "ok", "text": f"The column reaches {zm[-1]:,.0f} m, above the model's highest "
                                              f"level ({need_top_m:,.0f} m)."})
    stamp = history.times[record]
    source = {
        "path": str(history.path), "name": history.path.name, "sha256": history.sha256, "bytes": history.size_bytes,
        "format": history.format, "producer": history.producer, "title": history.title.strip(),
        "domain_id": history.domain_id, "record": int(record), "times": stamp,
        "valid_utc": stamp.replace("_", " ") + " UTC", "map_proj": map_proj,
        "dx_m": history.attribute("DX"), "mp_physics": history.attribute("MP_PHYSICS"),
    }
    return WrfColumn(source=source, point=point, z_m=zm, theta_k=theta, qv_kgkg=qv, u_ms=u_e, v_ms=v_e, p_pa=p,
                     surface=surface, rotation=rotation, theta_m=theta_m, model_top_face_m=float(zw[-1]),
                     negative_qv_levels=negative, checks=checks)


# ---------------------------------------------------------------------------------------------------------------
# The domain map: 0-6 km bulk shear (vectorised) and, on request, CAPE on a subsample of columns.
# ---------------------------------------------------------------------------------------------------------------
def domain_heights(history: WrfHistory, record: int) -> Any:
    """Mass-level heights above ground for every column, float64 (nz, ny, nx)."""

    np = _np()
    ph = history.array("PH", record).astype(np.float64)
    ph += history.array("PHB", record)
    ph /= WRF_G
    ph -= history.array("HGT", record)[None]
    return 0.5 * (ph[:-1] + ph[1:])


def earth_winds(history: WrfHistory, record: int) -> tuple[Any, Any]:
    """Every column's mass-point wind, earth-relative, (nz, ny, nx) each."""

    np = _np()
    u = history.array("U", record)
    v = history.array("V", record)
    ug = 0.5 * (u[:, :, :-1].astype(np.float64) + u[:, :, 1:])
    vg = 0.5 * (v[:, :-1, :].astype(np.float64) + v[:, 1:, :])
    if history.has("SINALPHA") and history.has("COSALPHA"):
        sa = history.array("SINALPHA", record).astype(np.float64)[None]
        ca = history.array("COSALPHA", record).astype(np.float64)[None]
        return ug * ca - vg * sa, vg * ca + ug * sa
    return ug, vg


def _at_height(values: Any, z: Any, target: float) -> Any:
    """values interpolated linearly in z to ``target`` m in every column (NaN where the column is shorter)."""

    np = _np()
    below = np.sum(z < target, axis=0) - 1
    inside = (below >= 0) & (below < z.shape[0] - 1)
    k = np.clip(below, 0, z.shape[0] - 2)
    z0 = np.take_along_axis(z, k[None], 0)[0]
    z1 = np.take_along_axis(z, (k + 1)[None], 0)[0]
    v0 = np.take_along_axis(values, k[None], 0)[0]
    v1 = np.take_along_axis(values, (k + 1)[None], 0)[0]
    w = (target - z0) / np.where(z1 > z0, z1 - z0, 1.0)
    out = v0 + w * (v1 - v0)
    return np.where(inside, out, np.nan)


def shear_map(history: WrfHistory, record: int, *, top_m: float = 6000.0) -> Any:
    """0-``top_m`` bulk shear, m/s: |wind at top_m - wind at the lowest level| in every column."""

    np = _np()
    z = domain_heights(history, record)
    u, v = earth_winds(history, record)
    du = _at_height(u, z, top_m) - u[0]
    dv = _at_height(v, z, top_m) - v[0]
    return np.hypot(du, dv)


def cape_map(history: WrfHistory, record: int, library: Path | str, *, max_columns: int = 4096,
             dz_m: float = 250.0) -> tuple[Any, dict[str, Any]]:
    """Surface-based CAPE (J/kg) from CM1's getcape.F (the Rust report) on at most ``max_columns`` columns.

    The columns are a regular subsample; each is written as its own sounding and reported on a uniform
    ``dz_m`` grid up to its top.  Returns the full-size map (NaN where not sampled) and how it was sampled.
    """

    np = _np()
    from ..sounding_report import SoundingReportSelection, initial_sounding_report

    ny = int(history.dimensions["south_north"])
    nx = int(history.dimensions["west_east"])
    stride = max(1, int(math.ceil(math.sqrt(nx * ny / float(max_columns)))))
    out = np.full((ny, nx), np.nan)
    fields = _Fields(history, record)
    done = 0
    failed = 0
    for j in range(stride // 2, ny, stride):
        for i in range(stride // 2, nx, stride):
            try:
                column = extract_column(history, record, ij=(i + 1, j + 1), fields=fields)
                nz = max(2, min(160, int(column.top_m / dz_m - 0.5)))
                report = initial_sounding_report(library, column.sounding_bytes(), isnd=7, iwnd=0, nz=nz,
                                                 dz_m=dz_m, selection=SoundingReportSelection())
                out[j, i] = float(report["surface_based"]["cape_j_per_kg"])
                done += 1
            except Exception:  # noqa: BLE001 - a column the report cannot take is left blank and counted
                failed += 1
    return out, {"stride": stride, "columns": done, "failed": failed, "dz_m": dz_m,
                 "rule": "CM1 r21.1 getcape.F surface parcel via GM1's Rust report, one column in every "
                         f"{stride} x {stride}"}


def environment_report(column: WrfColumn, library: Path | str | None = None, *, dz_m: float = 250.0
                       ) -> tuple[dict[str, Any] | None, str | None]:
    """CM1's getcape block and the hodograph numbers for the column, or (None, why not)."""

    from ..errors import CM1Refusal, fail_fast

    try:
        from ..cartesian_door import resolve_initializer_library
        from ..sounding_report import SoundingReportSelection, initial_sounding_report

        with fail_fast():
            path = resolve_initializer_library(None if library is None else Path(library))
            nz = max(2, min(160, int(column.top_m / dz_m - 0.5)))
            report = initial_sounding_report(path, column.sounding_bytes(), isnd=7, iwnd=0, nz=nz, dz_m=dz_m,
                                             selection=SoundingReportSelection())
    except CM1Refusal as refusal:
        return None, refusal.detail.reason
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, str(error)
    report = dict(report)
    report["grid"] = {"dz_m": dz_m, "nz": nz, "rule": f"the file on a uniform {dz_m:g} m grid up to its top"}
    return report, None


def summary_numbers(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """The five numbers a page shows first: CAPE, CIN, 0-1 and 0-6 km shear, 0-3 km SRH (and 0-1 km)."""

    if not report:
        return {}
    sb = report.get("surface_based") or {}
    mu = report.get("most_unstable") or {}
    ml = report.get("mixed_layer") or {}
    return {
        "cape_sb": sb.get("cape_j_per_kg"), "cin_sb": sb.get("cin_j_per_kg"),
        "cape_mu": mu.get("cape_j_per_kg"), "cape_ml": ml.get("cape_j_per_kg"), "cin_ml": ml.get("cin_j_per_kg"),
        "lcl_m": sb.get("lcl_m_agl"), "lfc_m": sb.get("lfc_m_agl"),
        "shear_0_1km": report.get("shear_0_1km_m_per_s"), "shear_0_6km": report.get("shear_0_6km_m_per_s"),
        "srh_0_1km": report.get("srh_0_1km_m2_per_s2"), "srh_0_3km": report.get("srh_0_3km_m2_per_s2"),
        "bunkers_right": [report.get("bunkers_right_u_m_per_s"), report.get("bunkers_right_v_m_per_s")],
    }


__all__ = [
    "COVERAGE_NOTE", "REFUSAL_CODES", "RULES", "SCHEMA", "WrfColumn", "cape_map", "domain_heights", "earth_winds",
    "environment_report", "extract_column", "haversine_km", "locate_point", "parse_pair", "shear_map",
    "summary_numbers",
]
