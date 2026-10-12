"""GM1 tornado criterion v2: the rule, applied to one 10 m wind plane and to a time series.

OFFLINE, CPU-ONLY, NUMPY-ONLY.  The rule is specified in
``docs/GM1-TORNADO-CRITERION-V2.md``; every number used here is a key of
:data:`CRITERION_V2` and its source is given there.  The constants are frozen
by the pre-registration commit (as amended before any run was scored; see the
document's Pre-registration record): :data:`CRITERION_V2_SHA256` is the
SHA-256 of their canonical JSON, and a verdict carries it together with the
SHA-256 of this file and of ``score.py`` (``FROZEN.json``).  A change to any
constant or to the code is a new criterion version (v3), never an edit of v2.

ONE SAMPLE (:func:`evaluate_plane`) reads a :class:`Plane`: the frame-relative
horizontal wind at 10 m above ground on CM1's C grid (``u`` on x faces,
``v`` on y faces), the frame velocity, the model free interior and the extent
of valid data, and the offset that puts plane coordinates on the ground.

1. First guesses: local maxima of the vertical vorticity (CM1 corner form)
   averaged over a square of fixed physical half-width, mutually separated
   (a pool of the strongest), ranked by the peak azimuthal-mean tangential
   wind about each; the best few are refined.
2. Each guess is moved to the point that maximises the peak of the
   azimuthal-mean tangential wind (a deterministic pattern search down to a
   1 m step).  The centre of an axisymmetric vortex is the unique maximum of
   that quantity, whether it is one- or two-celled, flat-topped or carries
   small satellite vortices.  Every refined candidate is kept.
3. About each centre: the azimuthal-mean tangential wind profile Vbar(r) from
   bilinear interpolation of the staggered winds on circles, its maximum
   Vmax and the radius of maximum wind RMW, whether the profile closes
   (falls to a fraction of Vmax within three RMW), the wavenumber-2 share of
   the tangential wind on the RMW circle, the mean core vorticity 2 Vmax/RMW,
   the ground-relative wind maximum within 1.5 RMW, and reported quantities.
4. Class: ``Q`` (qualifies), ``U`` (would qualify but RMW < 4 dx, or a
   threshold is met only within the discretisation uncertainty) or ``N``.

TIME SERIES (:func:`decide`): every candidate is tracked; a link between
samples is an identity link when the two cores overlap in the
storm-following frame; runs of identity-linked qualifying candidates, the
120 s duration, and the run outcome TORNADO / NO_TORNADO / UNRESOLVED /
INSUFFICIENT_CADENCE.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

CRITERION_V2_SCHEMA = "gm1.tornado-criterion/v2"

#: The criterion.  Frozen at pre-registration; see docs/GM1-TORNADO-CRITERION-V2.md.
CRITERION_V2: Mapping[str, Any] = MappingProxyType({
    # where
    "height_agl_m": 10.0,
    "max_lowest_scalar_level_m": 10.0,
    # grid-level resolvedness
    "max_grid_spacing_m": 25.0,
    "resolved_rmw_over_dx": 4.0,
    "profile_inner_radius_over_dx": 2.0,
    # a measured Vmax, mean core vorticity or point wind within this fraction
    # (x dx/RMW) below its threshold is "marginal": the grid cannot decide it
    "discretisation_uncertainty_dx_over_rmw": 0.5,
    # a nest child's samples are judged at the parent's spacing until the
    # child has had this long to make structure finer than the parent's
    "nest_spinup_s": 300.0,
    # finding the vortex
    "first_guess_box_half_width_m": 100.0,
    "first_guess_pool": 64,
    "first_guess_candidates": 16,
    "first_guess_separation_m": 200.0,
    "centre_search_steps_m": (64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0),
    "centre_search_moves_per_step": 64,
    "search_profile_dr_max_m": 4.0,
    "search_profile_dr_over_dx": 1.0,
    "search_profile_azimuths": 96,
    "edge_margin_m": 300.0,
    # the profile
    "profile_dr_max_m": 2.0,
    "profile_dr_over_dx": 0.5,
    "profile_azimuths": 360,
    "profile_max_radius_m": 1000.0,
    # rotation, closure, shape, intensity, hazard
    "mean_tangential_wind_min_m_s": 20.0,
    "closure_fraction": 0.9,
    "closure_radius_over_rmw": 3.0,
    "wavenumber2_max_fraction": 0.5,
    "mean_core_vorticity_min_s_inv": 0.1,
    "hazard_radius_over_rmw": 1.5,
    "ground_relative_wind_min_m_s": 35.0,
    "ground_relative_wind_reported_m_s": 29.0,
    "reported_fixed_disk_radius_m": 300.0,
    # time
    "centre_speed_max_m_s": 60.0,
    "centre_jump_allowance_m": 100.0,
    "identity_move_max_over_rmw_sum": 1.0,
    "bridge_gap_s": 3.0,
    "sustain_s": 120.0,
    "sustain_reported_s": 60.0,
    "max_sample_interval_s": 60.0,
    "clock_tolerance_s": 1.0e-6,
})


def _canonical(c: Mapping[str, Any]) -> str:
    return json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
                      sort_keys=True, separators=(",", ":"))


CRITERION_V2_SHA256 = hashlib.sha256(_canonical(CRITERION_V2).encode("ascii")).hexdigest()

OUTCOMES = ("TORNADO", "NO_TORNADO", "UNRESOLVED", "INSUFFICIENT_CADENCE")
CAPTIONS = {
    "TORNADO": "simulated tornado (GM1 criterion v2, {grid} grid, {cadence} output)",
    "NO_TORNADO": "vortex did not meet GM1's tornado criterion (v2)",
    "UNRESOLVED": "grid too coarse to judge (GM1 criterion v2)",
    "INSUFFICIENT_CADENCE": "output too sparse or too short to judge duration (GM1 criterion v2)",
}

SAMPLE_COLUMNS: tuple[str, ...] = (
    "sample", "model_time_s", "found", "class", "conditions", "flags",
    "centre_x_m", "centre_y_m", "centre_ground_x_m", "centre_ground_y_m",
    "vmax_m_s", "rmw_m", "rmw_over_dx", "closure_ratio", "wavenumber2_fraction", "wavenumber1_m_s",
    "profile_radius_m", "judged_dx_m", "candidates_n",
    "ground_wind_max_m_s", "ground_wind_300m_max_m_s", "frame_wind_max_m_s",
    "circulation_rmw_m2_s", "mean_zeta_core_s_inv", "inner_to_core_zeta",
    "zeta_corner_max_s_inv", "p_prime_min_hpa", "centre_move_m", "continuous",
    "q_run_s", "qu_run_s",
)


# ---------------------------------------------------------------------------
# One plane.
# ---------------------------------------------------------------------------
@dataclass
class Plane:
    """The 10 m frame-relative wind on CM1's C grid, in plane-local metres.

    ``u[j, i]`` sits at ``(i*dx, (j+1/2)*dy)``, shape ``(ny, nx+1)``;
    ``v[j, i]`` at ``((i+1/2)*dx, j*dy)``, shape ``(ny+1, nx)``.
    ``interior`` is the model's free interior (outside the lateral relaxation
    and specified zones) and ``data`` the rectangle where the winds are valid;
    both ``(x0, x1, y0, y1)`` in plane-local metres.  ``ground_offset`` added
    to a plane-local position gives the ground position (domain motion +
    placement).  ``p_prime_pa`` is optional ``(ny, nx)`` at scalar points.
    """

    u: np.ndarray
    v: np.ndarray
    dx: float
    dy: float
    umove: float
    vmove: float
    interior: tuple[float, float, float, float]
    data: tuple[float, float, float, float]
    ground_offset: tuple[float, float] = (0.0, 0.0)
    p_prime_pa: np.ndarray | None = None
    #: the spacing the resolvedness rules use when coarser than dx: a nest
    #: child's parent spacing during its spin-up (``nest_spinup_s``)
    resolution_dx: float | None = None

    @property
    def judged_dx(self) -> float:
        return max(self.dx, self.dy, float(self.resolution_dx or 0.0))

    @property
    def nx(self) -> int:
        return int(self.v.shape[1])

    @property
    def ny(self) -> int:
        return int(self.u.shape[0])

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=np.float64)
        self.v = np.asarray(self.v, dtype=np.float64)
        ny, nx1 = self.u.shape
        if self.v.shape != (ny + 1, nx1 - 1):
            raise ValueError(f"u {self.u.shape} and v {self.v.shape} are not one C grid")
        g = (0.5 * self.dx, (nx1 - 1.5) * self.dx, 0.5 * self.dy, (ny - 0.5) * self.dy)
        d = self.data
        self.data = (max(d[0], g[0]), min(d[1], g[1]), max(d[2], g[2]), min(d[3], g[3]))


def _bilinear(field: np.ndarray, fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
    nj, ni = field.shape
    i0 = np.clip(np.floor(fi).astype(np.int64), 0, ni - 2)
    j0 = np.clip(np.floor(fj).astype(np.int64), 0, nj - 2)
    a = fi - i0
    b = fj - j0
    f00 = field[j0, i0]
    f01 = field[j0, i0 + 1]
    f10 = field[j0 + 1, i0]
    f11 = field[j0 + 1, i0 + 1]
    return (1.0 - b) * ((1.0 - a) * f00 + a * f01) + b * ((1.0 - a) * f10 + a * f11)


def winds_at(plane: Plane, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame-relative (u, v) at arbitrary points, each from its own staggered lattice."""
    u = _bilinear(plane.u, x / plane.dx, y / plane.dy - 0.5)
    v = _bilinear(plane.v, x / plane.dx - 0.5, y / plane.dy)
    return u, v


_TRIG: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _trig(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n not in _TRIG:
        th = 2.0 * math.pi * (np.arange(n, dtype=np.float64) + 0.5) / n
        _TRIG[n] = (np.cos(th), np.sin(th))
    return _TRIG[n]


def available_radius(plane: Plane, cx: float, cy: float) -> float:
    d = plane.data
    return min(float(CRITERION_V2["profile_max_radius_m"]), cx - d[0], d[1] - cx, cy - d[2], d[3] - cy)


def profile_radii(r_lo: float, r_hi: float, dr: float) -> np.ndarray:
    k0 = int(math.ceil(r_lo / dr - 1e-9))
    k1 = int(math.floor(r_hi / dr + 1e-9))
    return dr * np.arange(max(k0, 1), k1 + 1, dtype=np.float64)


def tangential_profile(plane: Plane, centres: np.ndarray, radii: np.ndarray, azimuths: int) -> np.ndarray:
    """Azimuthal-mean tangential wind, shape ``(len(centres), len(radii))``."""
    c, s = _trig(azimuths)
    cx = centres[:, 0][:, None, None]
    cy = centres[:, 1][:, None, None]
    r = radii[None, :, None]
    x = cx + r * c[None, None, :]
    y = cy + r * s[None, None, :]
    u, v = winds_at(plane, x, y)
    vt = -s[None, None, :] * u + c[None, None, :] * v
    return vt.mean(axis=2)


def profile_dr(plane: Plane) -> float:
    c = CRITERION_V2
    return min(float(c["profile_dr_max_m"]), float(c["profile_dr_over_dx"]) * max(plane.dx, plane.dy))


def inner_radius(plane: Plane) -> float:
    return float(CRITERION_V2["profile_inner_radius_over_dx"]) * max(plane.dx, plane.dy)


def _objective(plane: Plane, pts: np.ndarray) -> np.ndarray:
    """Peak azimuthal-mean tangential wind of the search profile at each point (-inf when none)."""
    c = CRITERION_V2
    dr = min(float(c["search_profile_dr_max_m"]), float(c["search_profile_dr_over_dx"]) * max(plane.dx, plane.dy))
    r_lo = inner_radius(plane)
    out = np.full(len(pts), -np.inf)
    ravail = np.array([available_radius(plane, float(p[0]), float(p[1])) for p in pts])
    ok = ravail >= r_lo + dr
    if not np.any(ok):
        return out
    radii = profile_radii(r_lo, float(ravail[ok].max()), dr)
    prof = tangential_profile(plane, pts[ok], radii, int(c["search_profile_azimuths"]))
    # each point's profile is cut at its own data edge
    prof = np.where(radii[None, :] <= ravail[ok][:, None] + 1e-9, prof, -np.inf)
    out[ok] = prof.max(axis=1)
    return out


def refine_centre(plane: Plane, x0: float, y0: float) -> tuple[float, float, int]:
    """Deterministic pattern search for the maximum of the peak azimuthal-mean tangential wind."""
    c = CRITERION_V2
    x, y = float(x0), float(y0)
    f = float(_objective(plane, np.array([[x, y]]))[0])
    evals = 1
    offsets = np.array([(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)], dtype=np.float64)
    for step in c["centre_search_steps_m"]:
        for _ in range(int(c["centre_search_moves_per_step"])):
            pts = np.array([x, y])[None, :] + float(step) * offsets
            vals = _objective(plane, pts)
            evals += len(pts)
            k = int(np.argmax(vals))
            if vals[k] > f + 1e-9:
                x, y, f = float(pts[k, 0]), float(pts[k, 1]), float(vals[k])
            else:
                break
    return x, y, evals


def corner_vorticity(plane: Plane) -> np.ndarray:
    """``(ny+1, nx+1)`` CM1 corner vorticity (parcel.F:1066-1067 form); NaN on the outer ring."""
    z = np.full((plane.ny + 1, plane.nx + 1), np.nan)
    z[1:-1, 1:-1] = ((plane.v[1:-1, 1:] - plane.v[1:-1, :-1]) / plane.dx
                     - (plane.u[1:, 1:-1] - plane.u[:-1, 1:-1]) / plane.dy)
    return z


def _in_region(plane: Plane, x: float, y: float) -> bool:
    m = float(CRITERION_V2["edge_margin_m"])
    i, d = plane.interior, plane.data
    return (i[0] + m <= x <= i[1] - m and i[2] + m <= y <= i[3] - m
            and d[0] <= x <= d[1] and d[2] <= y <= d[3])


def first_guesses(plane: Plane) -> list[tuple[float, float, float]]:
    """``(x, y, box-mean zeta)`` of the strongest separated local maxima inside the centre region."""
    c = CRITERION_V2
    z = corner_vorticity(plane)
    nb = max(1, int(round(float(c["first_guess_box_half_width_m"]) / plane.dx)))
    mb = max(1, int(round(float(c["first_guess_box_half_width_m"]) / plane.dy)))
    zz = np.where(np.isfinite(z), z, 0.0)
    fin = np.isfinite(z).astype(np.float64)
    sat = np.zeros((zz.shape[0] + 1, zz.shape[1] + 1))
    sat[1:, 1:] = zz.cumsum(0).cumsum(1)
    cnt = np.zeros_like(sat)
    cnt[1:, 1:] = fin.cumsum(0).cumsum(1)
    J, I = zz.shape
    box = np.full(zz.shape, -np.inf)
    if J <= 2 * mb or I <= 2 * nb:
        return []
    j0, j1 = mb, J - mb
    i0, i1 = nb, I - nb
    s = (sat[j0 + mb + 1:j1 + mb + 1, i0 + nb + 1:i1 + nb + 1] - sat[j0 - mb:j1 - mb, i0 + nb + 1:i1 + nb + 1]
         - sat[j0 + mb + 1:j1 + mb + 1, i0 - nb:i1 - nb] + sat[j0 - mb:j1 - mb, i0 - nb:i1 - nb])
    n = (cnt[j0 + mb + 1:j1 + mb + 1, i0 + nb + 1:i1 + nb + 1] - cnt[j0 - mb:j1 - mb, i0 + nb + 1:i1 + nb + 1]
         - cnt[j0 + mb + 1:j1 + mb + 1, i0 - nb:i1 - nb] + cnt[j0 - mb:j1 - mb, i0 - nb:i1 - nb])
    full = (2 * mb + 1) * (2 * nb + 1)
    box[j0:j1, i0:i1] = np.where(n == full, s / np.maximum(n, 1), -np.inf)
    # centre region mask on corners
    xs = np.arange(I) * plane.dx
    ys = np.arange(J) * plane.dy
    m = float(c["edge_margin_m"])
    it, d = plane.interior, plane.data
    okx = (xs >= max(it[0] + m, d[0])) & (xs <= min(it[1] - m, d[1]))
    oky = (ys >= max(it[2] + m, d[2])) & (ys <= min(it[3] - m, d[3]))
    region = oky[:, None] & okx[None, :]
    b = np.where(region, box, -np.inf)
    pad = np.pad(b, 1, constant_values=-np.inf)
    neigh = np.max(np.stack([pad[1 + dj:1 + dj + J, 1 + di:1 + di + I]
                             for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]), axis=0)
    peaks = np.nonzero((b >= neigh) & (b > 0.0) & np.isfinite(b))
    order = np.lexsort((peaks[1], peaks[0], -b[peaks]))
    sep2 = float(c["first_guess_separation_m"]) ** 2
    out: list[tuple[float, float, float]] = []
    for k in order:
        j, i = int(peaks[0][k]), int(peaks[1][k])
        x, y = i * plane.dx, j * plane.dy
        if all((x - p[0]) ** 2 + (y - p[1]) ** 2 >= sep2 for p in out):
            out.append((x, y, float(b[j, i])))
            if len(out) >= int(c["first_guess_pool"]):
                break
    return out


def describe_centre(plane: Plane, cx: float, cy: float) -> dict[str, Any]:
    """The full-resolution profile and every rule quantity about one centre."""
    c = CRITERION_V2
    dxm = plane.judged_dx
    r_lo = inner_radius(plane)
    ravail = available_radius(plane, cx, cy)
    rec: dict[str, Any] = {"centre_x_m": cx, "centre_y_m": cy, "profile_radius_m": ravail,
                           "in_region": _in_region(plane, cx, cy), "flags": []}
    radii = profile_radii(r_lo, ravail, profile_dr(plane))
    if len(radii) < 2:
        rec["flags"].append("no-profile")
        rec.update(valid=False)
        return rec
    vbar = tangential_profile(plane, np.array([[cx, cy]]), radii, int(c["profile_azimuths"]))[0]
    k = int(np.argmax(vbar))
    vmax, rmw = float(vbar[k]), float(radii[k])
    rclose = min(float(c["closure_radius_over_rmw"]) * rmw, ravail)
    beyond = (radii > rmw) & (radii <= rclose + 1e-9)
    closure_ratio = float(vbar[beyond].min() / vmax) if np.any(beyond) and vmax > 0 else math.nan
    closed = bool(np.any(beyond) and vmax > 0 and closure_ratio <= float(c["closure_fraction"]))
    if not closed and float(c["closure_radius_over_rmw"]) * rmw > ravail:
        rec["flags"].append("broad" if ravail >= float(c["profile_max_radius_m"]) - 1e-9 else "extent-limited")
    if k == 0:
        rec["flags"].append("rmw-at-inner-radius")
    # shape: azimuthal wavenumbers of the tangential wind on the RMW circle.
    # Wavenumber 1 is a uniform flow across the vortex (translation relative
    # to the frame) and is reported only; wavenumber 2 is elongation, and a
    # straight vortex sheet has |V2| = (2/3) Vbar (Vt = dU/2 |sin(theta)|).
    na = int(c["profile_azimuths"])
    cs, sn = _trig(na)
    xs_c = cx + rmw * cs
    ys_c = cy + rmw * sn
    uu, vv = winds_at(plane, xs_c, ys_c)
    vt_c = -sn * uu + cs * vv
    th = 2.0 * math.pi * (np.arange(na, dtype=np.float64) + 0.5) / na
    wn1 = float(2.0 * abs(np.mean(vt_c * np.exp(-1j * th))))
    wn2 = float(2.0 * abs(np.mean(vt_c * np.exp(-2j * th))))
    wn2_fraction = wn2 / vmax if vmax > 0 else math.inf
    # winds at scalar points within the hazard disk and the fixed 300 m disk
    rh = float(c["hazard_radius_over_rmw"]) * rmw
    rfix = float(c["reported_fixed_disk_radius_m"])
    rbox = max(rh, rfix)
    i0 = max(0, int(math.floor((cx - rbox) / plane.dx)) - 1)
    i1 = min(plane.nx, int(math.ceil((cx + rbox) / plane.dx)) + 1)
    j0 = max(0, int(math.floor((cy - rbox) / plane.dy)) - 1)
    j1 = min(plane.ny, int(math.ceil((cy + rbox) / plane.dy)) + 1)
    uc = 0.5 * (plane.u[j0:j1, i0:i1] + plane.u[j0:j1, i0 + 1:i1 + 1])
    vc = 0.5 * (plane.v[j0:j1, i0:i1] + plane.v[j0 + 1:j1 + 1, i0:i1])
    xs = (np.arange(i0, i1) + 0.5) * plane.dx
    ys = (np.arange(j0, j1) + 0.5) * plane.dy
    d = plane.data
    r2 = (ys[:, None] - cy) ** 2 + (xs[None, :] - cx) ** 2
    inside = ((xs[None, :] >= d[0] - 0.5 * plane.dx) & (xs[None, :] <= d[1] + 0.5 * plane.dx)
              & (ys[:, None] >= d[2] - 0.5 * plane.dy) & (ys[:, None] <= d[3] + 0.5 * plane.dy))
    gs = np.hypot(uc + plane.umove, vc + plane.vmove)
    fs = np.hypot(uc, vc)

    def _max(a: np.ndarray, mask: np.ndarray) -> float:
        m = mask & np.isfinite(a)
        return float(a[m].max()) if np.any(m) else math.nan

    hz = inside & (r2 <= rh * rh)
    gmax = _max(gs, hz)
    # corner vorticity within RMW (reported, v1-like point value)
    zc = ((plane.v[max(j0, 1):j1, i0 + 1:i1] - plane.v[max(j0, 1):j1, i0:i1 - 1]) / plane.dx
          - (plane.u[max(j0, 1):j1, i0 + 1:i1] - plane.u[max(j0, 1) - 1:j1 - 1, i0 + 1:i1]) / plane.dy)
    xz = np.arange(i0 + 1, i1) * plane.dx
    yz = np.arange(max(j0, 1), j1) * plane.dy
    zr2 = (yz[:, None] - cy) ** 2 + (xz[None, :] - cx) ** 2
    zin = zr2 <= rmw * rmw
    pmin = math.nan
    if plane.p_prime_pa is not None:
        pmin = _max(-plane.p_prime_pa[j0:j1, i0:i1], inside & (r2 <= rmw * rmw))
        pmin = -pmin / 100.0 if math.isfinite(pmin) else math.nan
    # mean vorticity inside r is 2 Vbar(r)/r (Stokes)
    rin = 0.25 * rmw
    inner = math.nan
    if rin >= radii[0]:
        vin = float(np.interp(rin, radii, vbar))
        inner = (2.0 * vin / rin) / (2.0 * vmax / rmw) if vmax > 0 else math.nan
    rec.update(
        valid=True, vmax_m_s=vmax, rmw_m=rmw, rmw_over_dx=rmw / dxm, closure_ratio=closure_ratio,
        closed=closed, wavenumber1_m_s=wn1, wavenumber2_fraction=wn2_fraction,
        ground_wind_max_m_s=gmax,
        ground_wind_300m_max_m_s=_max(gs, inside & (r2 <= rfix * rfix)),
        frame_wind_max_m_s=_max(fs, hz),
        circulation_rmw_m2_s=2.0 * math.pi * rmw * vmax, mean_zeta_core_s_inv=2.0 * vmax / rmw,
        inner_to_core_zeta=inner,
        zeta_corner_max_s_inv=float(zc[zin].max()) if np.any(zin) else math.nan,
        p_prime_min_hpa=pmin,
    )
    return rec


CONDITION_LETTERS = "ERCAZWS"


def classify(plane: Plane, rec: Mapping[str, Any]) -> tuple[str, str]:
    """``(class, conditions)``: class Q/U/N; conditions E R C A Z W S.

    A letter is upper case when the condition holds, ``-`` when it fails, and
    lower case (R, Z, W only) when it fails by less than the discretisation
    uncertainty ``0.5 dx/RMW`` of the measured value: marginal, the grid
    cannot decide it.  Q: every condition holds.  U: E, C and A hold, R, Z and
    W hold or are marginal, and S fails or one of them is marginal.  N: else.
    """
    c = CRITERION_V2
    if not rec.get("valid"):
        return "N", "-" * len(CONDITION_LETTERS)
    dxm = plane.judged_dx
    rmw = float(rec["rmw_m"])
    delta = float(c["discretisation_uncertainty_dx_over_rmw"]) * dxm / rmw if rmw > 0 else 1.0

    def graded(value: float, threshold: float) -> str:
        if not math.isfinite(value):
            return "fail"
        if value >= threshold:
            return "ok"
        return "marginal" if value >= threshold * (1.0 - delta) else "fail"

    e = "ok" if rec["in_region"] else "fail"
    r = graded(rec["vmax_m_s"], float(c["mean_tangential_wind_min_m_s"]))
    cl = "ok" if rec["closed"] else "fail"
    a = "ok" if (math.isfinite(rec["wavenumber2_fraction"])
                 and rec["wavenumber2_fraction"] <= float(c["wavenumber2_max_fraction"])) else "fail"
    z = graded(rec["mean_zeta_core_s_inv"], float(c["mean_core_vorticity_min_s_inv"]))
    w = graded(rec["ground_wind_max_m_s"], float(c["ground_relative_wind_min_m_s"]))
    s = "ok" if rmw >= float(c["resolved_rmw_over_dx"]) * dxm - 1e-9 else "fail"
    states = (e, r, cl, a, z, w, s)
    cond = "".join(ch if st == "ok" else (ch.lower() if st == "marginal" else "-")
                   for ch, st in zip(CONDITION_LETTERS, states))
    if any(st == "fail" for st in (e, r, cl, a, z, w)):
        return "N", cond
    if s == "ok" and all(st == "ok" for st in (r, z, w)):
        return "Q", cond
    return "U", cond


_RANK = {"Q": 2, "U": 1, "N": 0}
CANDIDATE_KEYS = ("class", "conditions", "centre_x_m", "centre_y_m", "centre_ground_x_m", "centre_ground_y_m",
                  "vmax_m_s", "rmw_m", "closure_ratio", "wavenumber2_fraction", "mean_zeta_core_s_inv",
                  "ground_wind_max_m_s", "flags")


def evaluate_plane(plane: Plane) -> dict[str, Any]:
    """The sample of one plane: every refined candidate vortex, and the strongest as the sample's row.

    Every first guess is refined and classified; refined centres closer than
    ``max(2 dx, 2 m)`` to one already kept are the same vortex.  All kept
    candidates go to :func:`decide`, which tracks each through time.
    """
    pool = first_guesses(plane)
    # the pool of box-vorticity maxima is ranked by the search objective (the peak azimuthal-mean
    # tangential wind about each), so a row of shear-line maxima cannot crowd out a vortex
    if pool:
        score = _objective(plane, np.array([[g[0], g[1]] for g in pool]))
        order = sorted(range(len(pool)), key=lambda i: (-score[i], i))
        guesses = [pool[i] for i in order[:int(CRITERION_V2["first_guess_candidates"])]]
    else:
        guesses = []
    kept: list[dict[str, Any]] = []
    same = max(2.0 * max(plane.dx, plane.dy), 2.0)
    for gx, gy, bz in guesses:
        x, y, _ = refine_centre(plane, gx, gy)
        if any(math.hypot(x - k["centre_x_m"], y - k["centre_y_m"]) <= same for k in kept):
            continue
        rec = describe_centre(plane, x, y)
        klass, cond = classify(plane, rec)
        rec["class"], rec["conditions"] = klass, cond
        rec["first_guess"] = (gx, gy, bz)
        rec["found"] = bool(rec.get("valid"))
        rec["centre_ground_x_m"] = x + plane.ground_offset[0]
        rec["centre_ground_y_m"] = y + plane.ground_offset[1]
        kept.append(rec)
    base = {"umove": plane.umove, "vmove": plane.vmove, "judged_dx_m": plane.judged_dx,
            "first_guesses": len(guesses), "first_guess_pool": len(pool)}
    if not kept:
        return {**base, "found": False, "class": "N", "conditions": "-" * len(CONDITION_LETTERS),
                "flags": ["no-candidate"], "candidates": [], "candidates_n": 0}

    def key(r: Mapping[str, Any]) -> tuple[int, float]:
        return (_RANK[r["class"]], r.get("vmax_m_s", -math.inf) if r.get("valid") else -math.inf)

    best = dict(max(kept, key=key))
    best.update(base)
    best["candidates"] = [{k: r.get(k) for k in CANDIDATE_KEYS} for r in kept if r.get("valid")]
    best["candidates_n"] = len(best["candidates"])
    return best


# ---------------------------------------------------------------------------
# Time series.
# ---------------------------------------------------------------------------
@dataclass
class GridFacts:
    dx_m: float
    dy_m: float
    lowest_scalar_level_m: float
    notes: list[str] = field(default_factory=list)


def grid_gate(g: GridFacts) -> str | None:
    c = CRITERION_V2
    if max(g.dx_m, g.dy_m) > float(c["max_grid_spacing_m"]) + 1e-9:
        return (f"horizontal grid spacing {max(g.dx_m, g.dy_m):g} m exceeds "
                f"{float(c['max_grid_spacing_m']):g} m")
    if g.lowest_scalar_level_m > float(c["max_lowest_scalar_level_m"]) + 1e-9:
        return (f"lowest scalar level {g.lowest_scalar_level_m:g} m is above "
                f"{float(c['height_agl_m']):g} m AGL; 10 m winds would be extrapolated")
    return None


def _candidates(s: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """A sample's candidates; a bare sample dict (tests, old rows) is its own single candidate."""
    if "candidates" in s and isinstance(s["candidates"], list):
        return s["candidates"]
    return [s] if s.get("found") else []


def _link(p: Mapping[str, Any], q: Mapping[str, Any], dt: float, um: float, vm: float) -> tuple[bool, bool]:
    """``(plausible, identity)`` of candidate ``p`` then ``q``, ``dt`` apart.

    plausible: the ground centre moved at most ``60 m/s * dt + 100 m``.
    identity: in the storm-following frame (ground move minus the domain's
    own motion ``(umove, vmove) * dt``) the centre moved no more than the sum
    of the two RMWs, so the two cores overlap and cannot be two vortices.
    """
    c = CRITERION_V2
    dxg = q["centre_ground_x_m"] - p["centre_ground_x_m"]
    dyg = q["centre_ground_y_m"] - p["centre_ground_y_m"]
    plausible = math.hypot(dxg, dyg) <= float(c["centre_speed_max_m_s"]) * dt + float(c["centre_jump_allowance_m"])
    rel = math.hypot(dxg - um * dt, dyg - vm * dt)
    rsum = float(p.get("rmw_m") or 0.0) + float(q.get("rmw_m") or 0.0)
    identity = plausible and rel <= float(c["identity_move_max_over_rmw_sum"]) * rsum
    return plausible, identity


def _chains(samples: list[dict[str, Any]], t: list[float], classes: tuple[str, ...], identity: bool
            ) -> list[list[tuple[float, tuple[int, int] | None]]]:
    """For every candidate: ``(chain start time, predecessor)`` of the longest-lasting chain ending there.

    A chain links candidates of ``classes`` in successive samples at most
    ``max_sample_interval_s`` apart (or across a gap of samples when the two
    are at most ``bridge_gap_s`` apart), by identity links or, when
    ``identity`` is False, by plausible links.  Candidates outside
    ``classes`` get ``(nan, None)``.
    """
    c = CRITERION_V2
    tol = float(c["clock_tolerance_s"])
    maxdt = float(c["max_sample_interval_s"])
    bridge = float(c["bridge_gap_s"])
    out: list[list[tuple[float, tuple[int, int] | None]]] = []
    for k, s in enumerate(samples):
        row: list[tuple[float, tuple[int, int] | None]] = []
        um, vm = float(s.get("umove", 0.0) or 0.0), float(s.get("vmove", 0.0) or 0.0)
        for q in _candidates(s):
            if q.get("class") not in classes:
                row.append((math.nan, None))
                continue
            start, pred = t[k], None
            m = k - 1
            while m >= 0:
                dt = t[k] - t[m]
                if dt <= 0 or dt > maxdt + tol or (m < k - 1 and dt > bridge + tol):
                    break
                for ip, p in enumerate(_candidates(samples[m])):
                    st = out[m][ip][0]
                    if not math.isfinite(st) or st >= start:
                        continue
                    plaus, ident = _link(p, q, dt, um, vm)
                    if ident if identity else plaus:
                        start, pred = st, (m, ip)
                m -= 1
            row.append((start, pred))
        out.append(row)
    return out


def _cadence_span(t: list[float]) -> float:
    """The longest stretch of the record whose every interval is at most ``max_sample_interval_s``."""
    c = CRITERION_V2
    tol = float(c["clock_tolerance_s"])
    best, start = 0.0, 0
    for k in range(1, len(t)):
        if t[k] - t[k - 1] > float(c["max_sample_interval_s"]) + tol or t[k] <= t[k - 1]:
            start = k
        best = max(best, t[k] - t[start])
    return best


def decide(samples: list[dict[str, Any]], grid: GridFacts) -> dict[str, Any]:
    """Tracking, runs and the outcome.  ``samples`` need ``model_time_s`` and :func:`evaluate_plane` keys.

    Every candidate of every sample is tracked (:func:`_chains`), so a second
    vortex never breaks the first one's run and two vortices are never
    stitched into one.  Mutates each sample with ``centre_move_m`` (its row
    candidate from the previous sample's), ``continuous`` (the row candidate
    has an identity link back), ``q_run_s`` and ``qu_run_s`` (the longest
    identity-linked Q and Q-or-U chains ending at the sample, any candidate).
    """
    c = CRITERION_V2
    tol = float(c["clock_tolerance_s"])
    samples.sort(key=lambda s: s["model_time_s"])
    t = [float(s["model_time_s"]) for s in samples]
    sustain = float(c["sustain_s"])
    sustain60 = float(c["sustain_reported_s"])
    chains = {
        ("Q", True): _chains(samples, t, ("Q",), True),
        ("QU", True): _chains(samples, t, ("Q", "U"), True),
        ("Q", False): _chains(samples, t, ("Q",), False),
        ("QU", False): _chains(samples, t, ("Q", "U"), False),
    }
    ident_any = _chains(samples, t, ("Q", "U", "N"), True)

    def spans(ch) -> list[float]:
        return [max((t[k] - st for st, _ in row if math.isfinite(st)), default=0.0) for k, row in enumerate(ch)]

    per = {key: spans(ch) for key, ch in chains.items()}
    for k, s in enumerate(samples):
        s["q_run_s"] = per[("Q", True)][k]
        s["qu_run_s"] = per[("QU", True)][k]
        s["centre_move_m"] = math.nan
        s["continuous"] = False
        if k > 0 and s.get("found") and samples[k - 1].get("found"):
            s["centre_move_m"] = math.hypot(s["centre_ground_x_m"] - samples[k - 1]["centre_ground_x_m"],
                                            s["centre_ground_y_m"] - samples[k - 1]["centre_ground_y_m"])
        for ic, q in enumerate(_candidates(s)):
            if (q.get("centre_x_m"), q.get("centre_y_m")) == (s.get("centre_x_m"), s.get("centre_y_m")):
                s["continuous"] = bool(k > 0 and ident_any[k][ic][1] is not None)
    longest = {key: max(v, default=0.0) for key, v in per.items()}

    def episodes_of(ch) -> list[dict[str, Any]]:
        """Maximal chains lasting ``sustain_s``: ends that no later chain member extends."""
        extended = {pred for row in ch for _, pred in row if pred is not None}
        eps = []
        for k, row in enumerate(ch):
            for ic, (st, _) in enumerate(row):
                if math.isfinite(st) and (k, ic) not in extended and t[k] - st >= sustain - tol:
                    members = []
                    node: tuple[int, int] | None = (k, ic)
                    while node is not None:
                        members.append(node)
                        node = ch[node[0]][node[1]][1]
                    members.reverse()
                    cands = [_candidates(samples[m])[i] for m, i in members]
                    eps.append({"start_s": st, "end_s": t[k], "span_s": t[k] - st, "samples": len(members),
                                "vmax_peak_m_s": max(q["vmax_m_s"] for q in cands),
                                "ground_wind_peak_m_s": max(q["ground_wind_max_m_s"] for q in cands),
                                "rmw_range_m": [min(q["rmw_m"] for q in cands), max(q["rmw_m"] for q in cands)],
                                "centre_ground_start_m": [cands[0]["centre_ground_x_m"],
                                                          cands[0]["centre_ground_y_m"]]})
        eps.sort(key=lambda e: (e["start_s"], -e["span_s"]))
        return eps

    episodes = episodes_of(chains[("Q", True)])
    intervals = np.diff(np.asarray(t)) if len(t) > 1 else np.array([])
    record_span = (t[-1] - t[0]) if t else 0.0
    gate = grid_gate(grid)
    reason = None
    maxdt = float(c["max_sample_interval_s"])
    if gate is not None:
        outcome, reason = "UNRESOLVED", gate
    elif _cadence_span(t) < sustain - tol:
        outcome = "INSUFFICIENT_CADENCE"
        reason = (f"{len(t)} sample(s) spanning {record_span:g} s with intervals "
                  f"{sorted(set(float(x) for x in intervals))[:5]} s: no stretch of the record at <= "
                  f"{maxdt:g} s spacing lasts {sustain:g} s")
    elif episodes:
        outcome = "TORNADO"
    elif longest[("QU", True)] >= sustain - tol:
        outcome = "UNRESOLVED"
        reason = (f"one tracked vortex met every condition for >= {sustain:g} s except that its RMW was below "
                  f"{float(c['resolved_rmw_over_dx']):g} dx or a threshold was met only within the "
                  "discretisation uncertainty (marginal)")
    elif max(longest[("Q", False)], longest[("QU", False)]) >= sustain - tol:
        outcome = "INSUFFICIENT_CADENCE"
        reason = (f"qualifying vortices were present in consecutive samples for "
                  f"{max(longest[('Q', False)], longest[('QU', False)]):g} s, but between samples the centre moved "
                  "more than the two core radii in the storm-following frame: the output interval cannot show "
                  "that one vortex persisted")
    else:
        outcome = "NO_TORNADO"
    dx = max(grid.dx_m, grid.dy_m)
    cadence = float(np.median(intervals)) if len(intervals) else math.nan
    caption = CAPTIONS[outcome].format(grid=f"{dx:g} m", cadence=f"{cadence:g} s")
    counts: dict[str, int] = {}
    for s in samples:
        counts[s["class"]] = counts.get(s["class"], 0) + 1
    cond_counts = {ch: sum(1 for s in samples if ch in s["conditions"]) for ch in CONDITION_LETTERS}
    marginal_counts = {ch: sum(1 for s in samples if ch.lower() in s["conditions"]) for ch in "RZW"}
    flag_counts: dict[str, int] = {}
    for s in samples:
        for f in s.get("flags", []):
            flag_counts[f] = flag_counts.get(f, 0) + 1
    return {
        "schema": CRITERION_V2_SCHEMA,
        "criterion_v2_sha256": CRITERION_V2_SHA256,
        "outcome": outcome,
        "reason": reason,
        "caption": caption,
        "genesis_model_time_s": episodes[0]["start_s"] if episodes and outcome == "TORNADO" else None,
        "episodes": episodes,
        "longest_qualifying_run_s": longest[("Q", True)],
        "longest_qualifying_or_unresolved_run_s": longest[("QU", True)],
        "longest_qualifying_run_plausible_links_s": longest[("Q", False)],
        "reported_60s_rule": {"longest_s": longest[("Q", True)],
                              "would_be_tornado": longest[("Q", True)] >= sustain60 - tol and gate is None},
        "samples": len(samples),
        "record_start_s": t[0] if t else None,
        "record_end_s": t[-1] if t else None,
        "median_interval_s": cadence,
        "max_interval_s": float(intervals.max()) if len(intervals) else None,
        "grid": {"dx_m": grid.dx_m, "dy_m": grid.dy_m,
                 "lowest_scalar_level_m": grid.lowest_scalar_level_m,
                 "resolved_rmw_min_m": float(c["resolved_rmw_over_dx"]) * dx, "notes": grid.notes},
        "class_counts": counts,
        "condition_counts": cond_counts,
        "marginal_counts": marginal_counts,
        "flag_counts": flag_counts,
    }


def sample_row(k: int, s: Mapping[str, Any]) -> list[str]:
    def f(key: str, fmt: str = "%.4f") -> str:
        v = s.get(key)
        if v is None:
            return ""
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, float):
            return "" if not math.isfinite(v) else fmt % v
        return str(v)
    row = []
    for col in SAMPLE_COLUMNS:
        if col == "sample":
            row.append(str(k))
        elif col == "flags":
            row.append(",".join(s.get("flags", [])))
        elif col in ("model_time_s",):
            row.append("%.4f" % float(s["model_time_s"]))
        else:
            row.append(f(col))
    return row
