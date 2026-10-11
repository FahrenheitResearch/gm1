"""Morrison (``ptype = 5``) radar reflectivity on the CPU: CM1 r21.1's ``calc_refl10cm``.

GM1 runs Morrison through the ArWen seam, whose step computes no
reflectivity, so a Morrison frame carries no ``dbz``
(``CM1-CFG-DBZ-COMPONENT``).  CM1 r21.1 does compute it for ``ptype = 5``,
in ``morrison.F`` ``calc_refl10cm`` (called from ``MP_GRAUPEL`` at
``morrison.F:1246-1249`` when ``getdbz`` is set, on the step that ends at
an output time).  This module is a vectorised NumPy port of that routine and
of everything it reads, so the converter and the quick-looks share one
reflectivity:

* ``GRAUPEL_INIT`` (``morrison.F:304-658``): ``PI`` (a single-precision
  parameter), ``RHOW=997``, ``RHOSN=100``, ``RHOG`` = 400 (ihail=0) or 900
  (ihail=1), ``CS``/``CG``/``DS``/``DG``, ``CONS1``/``CONS2``, the lambda
  bounds ``LAMMINR/LAMMAXR/LAMMINS/LAMMAXS/LAMMING/LAMMAXG`` and the snow and
  graupel diameter bins ``Dds/dts`` (100 um - 2 cm) and ``Ddg/dtg``
  (100 um - 5 cm);
* ``radar_init`` (``morrison.F:5657-5705``): ``PI5``, ``lamda4``, the water
  (Ray 1972) and ice (Maetzler 1998) refractive indices at 10 cm and 0 C,
  ``K_w``, the Simpson weights and the mixing-rule strings;
* ``rayleigh_soak_wetgraupel``, ``get_m_mix_nested``, ``get_m_mix`` and
  ``m_complex_maxwellgarnett`` (Blahak's melting-particle routines) for the
  melting snow / graupel branch below the melting level;
* ``calccref`` (``misclibs.F:2259-2292``): ``cref`` is the column maximum of
  ``dbz``, starting from -1000.

A Fortran oracle test in the development repository cuts the untouched
routines out of CM1's ``morrison.F``, compiles them and holds this port to
them within 1e-5 dBZ.

Only ``RHOG`` depends on ``ihail`` among the constants ``calc_refl10cm``
reads (``AG``/``BG`` change too, but only fall speeds use them).  ``ndcnst``
sets the cloud-droplet number, which ``calc_refl10cm`` does not read: cloud
water and cloud ice contribute nothing to CM1's Morrison reflectivity.

Faithfulness notes (deliberate, matching CM1 rather than "fixing" it):

* where the slope is not clipped, CM1 forms ``N0 = rn*lam/rho`` with ``rn``
  already a per-volume number, i.e. a per-kilogram intercept, while the
  clipped branch forms a per-volume ``N0 = lam**4*r/CONS``.  The port keeps
  that (``cm1_n0_per_kg=True``); pass ``False`` to use ``rn*lam`` instead.
* the Simpson weights are used without the ``1/2`` step factor and against
  the per-bin widths of a geometric grid, exactly as CM1 does.
* REAL (single) versus DOUBLE PRECISION is emulated where the Fortran mixes
  them (constants and the column slope/intercept in float32, the melting
  integrals in float64, results rounded back to float32).
* at the initial time no microphysics step has run, so CM1's ``qdiag`` still
  holds its zero start: CM1's own t = 0 history file has ``dbz = cref = 0``
  everywhere (``tests/data/cm1-ref-24x24/cm1out_000001.nc``), and so does
  the converter's for a frame published before the first step.

Temperature only locates the melting level: ``T = (th0+th3d)*(pi0+pp3d)``
(CM1 ``mp_driver.F:108``) from whatever the frame carries
(:func:`temperature`).  Nothing here imports CuPy.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Callable, Mapping
import weakref

import numpy as np

f32 = np.float32
f64 = np.float64

#: The GM1 components whose frames this formula applies to.
MORRISON_COMPONENTS = ("morrison-mp10",)

#: CM1 r21.1 source the port follows, for provenance records.
AUTHORITY = (
    "CM1 r21.1 morrison.F calc_refl10cm (ptype=5) and misclibs.F calccref, "
    "NumPy port cm1core.cm1out.reflectivity"
)

# morrison.F:103: PI, a binary32 parameter of this value.
PI = f32(3.1415926535897932384626434)
# morrison.F:5659 PI5 = PI*PI*PI*PI*PI (REAL arithmetic, stored DOUBLE)
PI5 = f64(PI * PI * PI * PI * PI)
LAMDA_RADAR = 0.10  # morrison.F:282, metres
LAMDA4 = LAMDA_RADAR * LAMDA_RADAR * LAMDA_RADAR * LAMDA_RADAR
NBINS = 100
R1 = f32(1.0e-12)
T_0 = f32(273.15)
FLOOR_DBZ = -35.0  # calc_refl10cm initialises dBZ(k) = -35.0 (never output)
ZE_FLOOR = f32(1.0e-22)  # ze_* initial value: 10*log10(3e-22*1e18) = -35.23 dBZ
#: misclibs.F:2272: calccref starts every column at -1000.
CREF_START = f32(-1000.0)

# CM1 constants.F rd, cp, rv and p00, used only when T must be rebuilt.
RD = 287.04
CP = 1005.7
RV = 461.5
P00 = 100000.0

MASS_DEFAULT = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBER_DEFAULT = ("nci", "ncs", "ncr", "ncg")


# --------------------------------------------------------------------------
# constants (GRAUPEL_INIT)
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def morrison_constants(ihail: int = 1) -> dict[str, np.floating]:
    """The GRAUPEL_INIT values calc_refl10cm reads, as CM1's REAL words."""

    if ihail not in (0, 1):
        raise ValueError(f"ihail must be 0 or 1, not {ihail!r}")
    rhow = f32(997.0)
    rhosn = f32(100.0)
    rhog = f32(400.0) if ihail == 0 else f32(900.0)  # morrison.F:440-445
    ds = f32(3.0)
    dg = f32(3.0)
    cs = rhosn * PI / f32(6.0)  # morrison.F:474, left to right in REAL
    cg = rhog * PI / f32(6.0)  # morrison.F:476
    # CONS1 and CONS2 are gamma(1 + DS) times CS and gamma(1 + DG) times CG (morrison.F:552-553).
    # CM1's Cody GAMMA returns 6.0 for 4.0 (checked against the compiled
    # routine by the Fortran oracle).
    gamma4 = f32(math.gamma(4.0))
    return {
        "rhow": rhow,
        "rhosn": rhosn,
        "rhog": rhog,
        "ds": ds,
        "dg": dg,
        "cons1": gamma4 * cs,
        "cons2": gamma4 * cg,
        # morrison.F:486-498
        "lammaxr": f32(1.0) / f32(20.0e-6),
        "lamminr": f32(1.0) / f32(2800.0e-6),
        "lammaxs": f32(1.0) / f32(10.0e-6),
        "lammins": f32(1.0) / f32(2000.0e-6),
        "lammaxg": f32(1.0) / f32(20.0e-6),
        "lamming": f32(1.0) / f32(2000.0e-6),
    }


def _bins(d0: float, dmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Geometric diameter bins, GRAUPEL_INIT morrison.F:626-649."""

    ddx = np.empty(NBINS + 1, dtype=f64)
    ddx[0] = f64(f32(d0))  # D0s/D0g are REAL parameters times 1.0d0
    ddx[NBINS] = dmax
    n = np.arange(2, NBINS + 1, dtype=f64)
    ddx[1:NBINS] = np.exp((n - 1.0) / float(NBINS) * math.log(ddx[NBINS] / ddx[0]) + math.log(ddx[0]))
    centres = np.sqrt(ddx[:-1] * ddx[1:])
    widths = ddx[1:] - ddx[:-1]
    return centres, widths


def _m_complex_water_ray(lam: float, t: float) -> complex:
    """Ray (1972) water refractive index, morrison.F:5708-5738."""

    s = lambda v: float(f32(v))  # a REAL literal promoted into DOUBLE arithmetic  # noqa: E731
    pi = float(PI)
    epsinf = 5.27137 + 0.02164740 * t - 0.00131198 * t * t
    tt = t - 25.0
    epss = 78.54 * (1.0 - 4.579e-3 * tt + 1.190e-5 * tt * tt - 2.800e-8 * tt * tt * tt)
    alpha = -16.8129 / (t + s(273.16)) + 0.0609265
    lambdas = 0.00033836 * math.exp(2513.98 / (t + s(273.16))) * s(1e-2)
    ratio = lambdas / lam
    nenner = (1.0 + 2.0 * ratio ** (1.0 - alpha) * math.sin(alpha * pi * 0.5)
              + ratio ** (2.0 - 2.0 * alpha))
    epsr = epsinf + ((epss - epsinf) * (ratio ** (1.0 - alpha) * math.sin(alpha * pi * 0.5) + 1.0)) / nenner
    epsi = ((epss - epsinf) * (ratio ** (1.0 - alpha) * math.cos(alpha * pi * 0.5) + 0.0)) / nenner \
        + lam * s(1.25664) / s(1.88496)
    # SQRT(CMPLX(epsr,-epsi)): CMPLX without KIND is default (single) complex.
    return complex(np.sqrt(np.complex64(complex(epsr, -epsi))))


def _m_complex_ice_maetzler(lam: float, t: float) -> complex:
    """Maetzler (1998) ice refractive index, morrison.F:5742-5777."""

    s = lambda v: float(f32(v))  # noqa: E731
    c = 2.99e8
    tk = t + s(273.16)
    f = c / lam * 1e-9
    b1 = s(0.0207)
    b2 = 1.16e-11
    b = 335.0
    deltabeta = math.exp(s(-10.02) + s(0.0364) * (tk - s(273.16)))
    betam = (b1 / tk) * (math.exp(b / tk) / ((math.exp(b / tk) - 1.0) ** 2)) + b2 * f * f
    beta = betam + deltabeta
    theta = s(300.0) / tk - 1.0
    alfa = (0.00504 + 0.0062 * theta) * math.exp(-22.1 * theta)
    real = s(3.1884) + s(9.1e-4) * (tk - s(273.16))
    imag = float(f32(alfa / f + beta * f))  # CMPLX(0.0d0, ...) is single complex
    m = complex(real, imag)
    return complex(np.sqrt(np.conj(np.complex128(m))))


@lru_cache(maxsize=None)
def radar_tables() -> dict[str, object]:
    """radar_init (morrison.F:5657-5705) plus the snow/graupel bins."""

    m_w = _m_complex_water_ray(LAMDA_RADAR, 0.0)
    m_i = _m_complex_ice_maetzler(LAMDA_RADAR, 0.0)
    k_w = abs((m_w * m_w - 1.0) / (m_w * m_w + 2.0)) ** 2
    simpson = np.zeros(NBINS + 1, dtype=f64)
    for n in range(0, NBINS - 1, 2):  # Fortran n = 1, nbins-1, 2
        simpson[n] += 1.0 / 3.0
        simpson[n + 1] += 4.0 / 3.0
        simpson[n + 2] += 1.0 / 3.0
    dds, dts = _bins(100.0e-6, 0.02)  # D0s, 2 cm
    ddg, dtg = _bins(100.0e-6, 0.05)  # D0g, 5 cm
    return {
        "m_w_0": m_w, "m_i_0": m_i, "K_w": k_w,
        "simpson": simpson[:NBINS].copy(),
        "Dds": dds, "dts": dts, "Ddg": ddg, "dtg": dtg,
    }


# --------------------------------------------------------------------------
# Blahak melting-particle backscatter (vectorised)
# --------------------------------------------------------------------------


def _maxwellgarnett_spheroidal(vol1, vol2, vol3, m1, m2, m3):
    """m_complex_maxwellgarnett with inclusion='spheroidal' (morrison.F:6062)."""

    error = np.abs(vol1 + vol2 + vol3 - 1.0) > 1e-6
    m1t, m2t, m3t = m1 * m1, m2 * m2, m3 * m3
    with np.errstate(all="ignore"):
        beta2 = 2.0 * m1t / (m2t - m1t) * (m2t / (m2t - m1t) * np.log(m2t / m1t) - 1.0)
        beta3 = 2.0 * m1t / (m3t - m1t) * (m3t / (m3t - m1t) * np.log(m3t / m1t) - 1.0)
        m = np.sqrt(((1.0 - vol2 - vol3) * m1t + vol2 * beta2 * m2t + vol3 * beta3 * m3t)
                    / (1.0 - vol2 - vol3 + vol2 * beta2 + vol3 * beta3))
    return m, error


def rayleigh_soak_wetgraupel(x_g, a_geo, b_geo, fmelt, meltratio_outside, m_w, m_i):
    """Backscatter cross-section of a melting particle, morrison.F:5781-5872.

    Hard-wired to the radar_init strings CM1 uses for both snow and graupel:
    mixing rule maxwellgarnett, host air, matrix water, inclusion spheroidal,
    hostmatrix icewater, hostinclusion spheroidal.  Broadcasts over x_g/fmelt.
    """

    rho_i = 900.0
    rho_w = 1000.0
    m_air = 1.0 + 0.0j
    x_g, fmelt = np.broadcast_arrays(np.asarray(x_g, f64), np.asarray(fmelt, f64))
    fm = np.maximum(np.minimum(fmelt, 1.0), 0.0)
    mra = max(min(meltratio_outside, 1.0), 0.0)
    mra = mra + (1.0 - mra) * fm
    x_w = x_g * fm
    d_g = a_geo * x_g ** b_geo
    ok = d_g >= 1e-12
    with np.errstate(all="ignore"):
        vg = float(PI / f32(6.0)) * d_g ** 3
        rhog = np.maximum(np.minimum(x_g / vg, rho_i), 10.0)
        vg = x_g / rhog
        grenz = 1.0 - rhog / rho_w
        fmgrenz = (rho_i - rhog) / (mra * rho_i - rhog + rho_i * rhog / rho_w)
        volg = np.where(
            (mra <= grenz) | (fm <= fmgrenz),
            vg * (1.0 - mra * fm),
            (x_g - x_w) / rho_i + x_w / rho_w,
        )
        d_large = (float(f32(6.0) / PI) * volg) ** float(f32(1.0) / f32(3.0))
        volice = (x_g - x_w) / (volg * rho_i)
        volwater = x_w / (rho_w * volg)
        volair = 1.0 - volice - volwater
        # get_m_mix_nested, host='air', hostmatrix='icewater'
        vol1 = volice / np.maximum(volice + volwater, 1e-10)
        vol2 = 1.0 - vol1
        # get_m_mix(m_a, m_i, m_w, 0, vol1, vol2, matrix='water')
        #   -> maxwellgarnett(volwater, volair, volice, m_w, m_a, m_i)
        mtmp, err1 = _maxwellgarnett_spheroidal(vol2, 0.0, vol1, m_w, m_air, m_i)
        # get_m_mix(m_a, mtmp, 2*m_a, volair, 1-volair, 0, matrix='ice')
        #   -> maxwellgarnett(1-volair, volair, 0, mtmp, m_a, 2*m_a)
        m_core, err2 = _maxwellgarnett_spheroidal(1.0 - volair, volair, 0.0, mtmp, m_air, 2.0 * m_air)
        m2 = m_core * m_core
        c_back = np.abs((m2 - 1.0) / (m2 + 2.0)) ** 2 * PI5 * d_large ** 6 / LAMDA4
    return np.where(ok & ~err1 & ~err2, c_back, 0.0)


# --------------------------------------------------------------------------
# calc_refl10cm and calccref
# --------------------------------------------------------------------------


def _pow4(x):
    x2 = x * x
    return x2 * x2


def _slope_intercept(r, rn, rho, cons, d, lammin, lammax, cm1_n0_per_kg):
    """dumLAM/dumN0 with CM1's clipping, in REAL; returns (ilam, n0) DOUBLE."""

    with np.errstate(all="ignore"):
        lam = (cons * rn / r) ** (f32(1.0) / d)
        n0 = rn * lam / rho if cm1_n0_per_kg else rn * lam
        lo = lam < lammin
        hi = (~lo) & (lam > lammax)
        lam = np.where(lo, lammin, np.where(hi, lammax, lam)).astype(f32)
        n0 = np.where(lo | hi, _pow4(lam) * r / cons, n0).astype(f32)
        ilam = (f32(1.0) / lam).astype(f64)
    return ilam, n0.astype(f64)


# --------------------------------------------------------------------------
# The melting-layer integral (morrison.F:6340-6388): below the melting level,
# eta = sum over the 100 diameter bins of
#     f_d(n) * CBACK(n) * simpson(n) * dt(n)      (morrison.F:6358, 6382),
# CBACK from rayleigh_soak_wetgraupel (morrison.F:5780-5872), for snow and
# for graupel/hail at every level from k_0-1 down.  Two evaluations of it:
# --------------------------------------------------------------------------

#: Melting points integrated together by :func:`_melting_eta`: every
#: temporary of a block stays in the core's cache.
MELT_BLOCK = 8192

#: Where :func:`_melting_eta` forms |K|^2 once per point, its double integral
#: differs from the reference's by rounding: at most 3.9e-15 relative over
#: 6.4 M melting points of the stock supercell, two 5.24 M-cell frames and
#: two 39 M-cell 100 m frames.  The port
#: writes ``ze`` as float32, so a point changes a word only if its ze lies
#: that close to a float32 rounding boundary: every point whose ze is within
#: ``ROUNDING_GUARD`` (relative, about 3,800 times the largest difference
#: seen) of one is formed again by the reference loop (:func:`_melting_ze`),
#: one point in about 2,700.
ROUNDING_GUARD = 2.0 ** -36


def _species_bins(species: str, rho_p: np.floating) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """(Dd, dt, x, oc, ob) of one species, as the melting branch forms them."""

    tabs = radar_tables()
    dbin, wbin = (tabs["Dds"], tabs["dts"]) if species == "s" else (tabs["Ddg"], tabs["dtg"])
    ob = f32(1.0) / f32(3.0)
    oc = (f32(1.0) / (PI * rho_p / f32(6.0))) ** ob
    x = float(PI * rho_p / f32(6.0)) * dbin ** 3
    return dbin, wbin, x, float(oc), float(ob)


def _melting_eta_reference(species: str, rho_p: np.floating, fk: np.ndarray, n0k: np.ndarray,
                           ilamk: np.ndarray) -> np.ndarray:
    """The melting integral bin by bin over every point, :func:`rayleigh_soak_wetgraupel` per bin.

    This is the port's earlier evaluation, word for word; it computes
    CBACK's two Maxwell-Garnett mixtures (two complex logarithms, three
    complex square roots) 100 times per point.
    """

    tabs = radar_tables()
    dbin, wbin, x, oc, ob = _species_bins(species, rho_p)
    lamk = 1.0 / ilamk
    out = np.zeros(fk.size, dtype=f64)
    for n in range(NBINS):
        cback = rayleigh_soak_wetgraupel(x[n], oc, ob, fk, 0.7, tabs["m_w_0"], tabs["m_i_0"])
        if species == "s":
            f_d = n0k * np.exp(-dbin[n] / ilamk)
        else:
            f_d = n0k * np.exp(-lamk * dbin[n])
        out = out + f_d * cback * tabs["simpson"][n] * wbin[n]
    return out


@lru_cache(maxsize=None)
def _bin_constants(species: str, rho_p: float) -> tuple[tuple[bool, np.float64, np.float64, np.float64], ...]:
    """Per bin, rayleigh_soak_wetgraupel's point-independent words: (D_g >= 1e-12, vg, rhog, grenz).

    The reference forms them on a stride-0 row (``x_g`` broadcast against the
    points); they are formed here the same way, on a row of two, once.
    """

    _, _, x, oc, ob = _species_bins(species, f32(rho_p))
    rho_i, rho_w = 900.0, 1000.0
    out = []
    with np.errstate(all="ignore"):
        for n in range(NBINS):
            x_g = np.broadcast_to(np.asarray(x[n], f64), (2,))
            d_g = oc * x_g ** ob
            vg = float(PI / f32(6.0)) * d_g ** 3
            rhog = np.maximum(np.minimum(x_g / vg, rho_i), 10.0)
            vg = x_g / rhog
            grenz = 1.0 - rhog / rho_w
            out.append((bool((d_g >= 1e-12)[0]), vg[0], rhog[0], grenz[0]))
    return tuple(out)


def _mg_constants(m_w: complex, m_i: complex) -> tuple[Any, Any, Any, Any]:
    """The first Maxwell-Garnett mixture's point-independent words (water matrix, air and ice inclusions)."""

    m_air = 1.0 + 0.0j
    m1t, m2t, m3t = m_w * m_w, m_air * m_air, m_i * m_i
    with np.errstate(all="ignore"):
        beta2 = 2.0 * m1t / (m2t - m1t) * (m2t / (m2t - m1t) * np.log(m2t / m1t) - 1.0)
        beta3 = 2.0 * m1t / (m3t - m1t) * (m3t / (m3t - m1t) * np.log(m3t / m1t) - 1.0)
    return m1t, m3t, beta3, (0.0 * beta2 * m2t, 0.0 * beta2)


def _third_term_can_matter(num: np.ndarray, den: np.ndarray, q1t: np.ndarray, q3t: complex) -> bool:
    """Whether adding ``0.0 * beta3 * m3t`` (and ``0.0 * beta3``) could change a word of the second mixture.

    A signed zero added to a finite, non-zero part changes nothing; it can change the sign of a zero part, and
    ``beta3`` is not finite when ``m1t`` is ``m3t`` or zero (or not finite).
    """

    return not (np.all(np.isfinite(num)) and np.all(np.isfinite(den)) and np.all(num.real != 0.0)
                and np.all(num.imag != 0.0) and np.all(den.real != 0.0) and np.all(den.imag != 0.0)
                and np.all(q1t != q3t) and np.all(q1t != 0.0) and np.all(np.isfinite(q1t)))


def _cback_core(vol1: np.ndarray, volair: np.ndarray, mg1: tuple[Any, Any, Any, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(|K|^2 of the melting particle, CM1's mixing-error flag) from its ice and air fractions.

    The same words as ``rayleigh_soak_wetgraupel``'s two
    ``_maxwellgarnett_spheroidal`` calls and its ``|(m^2-1)/(m^2+2)|^2``,
    with one term left out: the second mixture's third inclusion has volume
    ``0.0``, so ``0.0 * beta3 * m3t`` adds a signed zero.  Where that zero
    could change a word (a zero or non-finite sum, or a ``beta3`` that is not
    finite), the block is evaluated with the term, as the reference does.
    """

    m1t, m3t, beta3_1, (zero_num1, zero_den1) = mg1
    m_air = 1.0 + 0.0j
    q2t = m_air * m_air
    vol2 = 1.0 - vol1
    err1 = np.abs(vol2 + 0.0 + vol1 - 1.0) > 1e-6
    mtmp = np.sqrt(((1.0 - 0.0 - vol1) * m1t + zero_num1 + vol1 * beta3_1 * m3t)
                   / (1.0 - 0.0 - vol1 + zero_den1 + vol1 * beta3_1))
    v1 = 1.0 - volair
    err2 = np.abs(v1 + volair + 0.0 - 1.0) > 1e-6
    q1t = mtmp * mtmp
    dq = q2t - q1t
    beta2 = 2.0 * q1t / dq * (q2t / dq * np.log(q2t / q1t) - 1.0)
    num = (1.0 - volair - 0.0) * q1t + volair * beta2 * q2t
    den = 1.0 - volair - 0.0 + volair * beta2
    q3 = 2.0 * m_air
    q3t = q3 * q3
    if _third_term_can_matter(num, den, q1t, q3t):
        beta3 = 2.0 * q1t / (q3t - q1t) * (q3t / (q3t - q1t) * np.log(q3t / q1t) - 1.0)
        num = num + 0.0 * beta3 * q3t
        den = den + 0.0 * beta3
    m_core = np.sqrt(num / den)
    m2 = m_core * m_core
    return np.abs((m2 - 1.0) / (m2 + 2.0)) ** 2, err1 | err2


def _melting_eta(species: str, rho_p: np.floating, fk: np.ndarray, n0k: np.ndarray,
                 ilamk: np.ndarray) -> np.ndarray:
    """The melting integral, fast: the reference's words, in blocks, with CBACK's mixture formed once where it can be.

    CBACK(n) = |K|^2 * PI5 * D_large(n)**6 / lamda4, and |K|^2 depends on the
    bin only through the particle's ice, water and air fractions.  Those are
    fixed by the melted fraction and the particle density ``rhog``; the bin
    enters them only through rounding (``x_g`` cancels).

    * When ``rhog`` is the same word in every bin (graupel/hail: ``x/vg`` is
      clipped to 900 with ``ihail = 1``), |K|^2 is formed once per point from
      the first bin's fractions, instead of 100 times.  Its double-precision
      words differ from the reference's by rounding (a few parts in 1e15);
      the float32 ze this port writes is still the reference's word, since
      every point near a float32 rounding boundary is formed again by the
      reference (:data:`ROUNDING_GUARD`, :func:`_melting_ze`).
    * When ``rhog`` differs between bins (snow, whose REAL ``obms = 1./ds``
      exponent, morrison.F:6348-6349, leaves ``x/vg`` drifting by about 5e-7
      across the bins, and
      unclipped graupel with ``ihail = 0``), |K|^2 is formed per bin, with
      the reference's words (:func:`_cback_core`): every double word equals
      the reference's.

    Everything else (D_large, f_d, the Simpson sum in bin order) is the
    reference's words.  Blocks of :data:`MELT_BLOCK` points change no word:
    every operation is elementwise.
    """

    tabs = radar_tables()
    dbin, wbin, x, _, _ = _species_bins(species, rho_p)
    consts = _bin_constants(species, float(rho_p))
    mg1 = _mg_constants(tabs["m_w_0"], tabs["m_i_0"])
    simpson = tabs["simpson"]
    rho_i, rho_w = 900.0, 1000.0
    c6pi = float(f32(6.0) / PI)
    c13 = float(f32(1.0) / f32(3.0))
    one_density = len({float(c[2]) for c in consts}) == 1
    out = np.empty(fk.size, dtype=f64)
    with np.errstate(all="ignore"):
        lamk_all = 1.0 / ilamk
        for s0 in range(0, fk.size, MELT_BLOCK):
            sl = slice(s0, min(s0 + MELT_BLOCK, fk.size))
            n0, il, lamk = n0k[sl], ilamk[sl], lamk_all[sl]
            fm = np.maximum(np.minimum(fk[sl], 1.0), 0.0)
            mra = 0.7 + (1.0 - 0.7) * fm
            one_minus = 1.0 - mra * fm
            mra_rho_i = mra * rho_i
            k2pi5 = bad = None
            branch_a = None
            acc = np.zeros(fm.size, dtype=f64)
            for n in range(NBINS):
                ok, vg, rhog, grenz = consts[n]
                x_g = x[n]
                x_w = x_g * fm
                if branch_a is None or not one_density:
                    # With one density the branch (and so fmgrenz) is the same in every bin.
                    fmgrenz = (rho_i - rhog) / (mra_rho_i - rhog + rho_i * rhog / rho_w)
                    branch_a = (mra <= grenz) | (fm <= fmgrenz)
                    every_a, no_a = bool(branch_a.all()), not branch_a.any()
                # np.where's words, forming only the branch a block uses when it uses one.
                if every_a:
                    volg = vg * one_minus
                elif no_a:
                    volg = (x_g - x_w) / rho_i + x_w / rho_w
                else:
                    volg = np.where(branch_a, vg * one_minus, (x_g - x_w) / rho_i + x_w / rho_w)
                d_large = (c6pi * volg) ** c13
                if k2pi5 is None or not one_density:
                    volice = (x_g - x_w) / (volg * rho_i)
                    volwater = x_w / (rho_w * volg)
                    volair = 1.0 - volice - volwater
                    vol1 = volice / np.maximum(volice + volwater, 1e-10)
                    k2, bad = _cback_core(vol1, volair, mg1)
                    k2pi5 = k2 * PI5
                cback = np.where(~bad if ok else False, k2pi5 * d_large ** 6 / LAMDA4, 0.0)
                if species == "s":
                    f_d = n0 * np.exp(-dbin[n] / il)
                else:
                    f_d = n0 * np.exp(-lamk * dbin[n])
                acc = acc + f_d * cback * simpson[n] * wbin[n]
            out[sl] = acc
    return out


def _melting_ze(species: str, rho_p: np.floating, fk: np.ndarray, n0k: np.ndarray, ilamk: np.ndarray,
                scale: float, melting: str = "fast") -> np.ndarray:
    """The melting points' float32 ze, ``(scale * eta).astype(f32)`` (CM1's
    ``SNGL(lamda4 / (pi5 * K_w) * eta)``, morrison.F:6361, 6384), eta from
    :func:`_melting_eta_reference` (``melting="reference"``) or :func:`_melting_eta`.

    With the fast integral, a point whose ze is not the same float32 word
    for every value within :data:`ROUNDING_GUARD` of it (a rounding
    boundary is that close; or ze is zero or not finite) takes the
    reference's integral instead, so each float32 word is the reference's
    as long as the two doubles differ by less than the guard.
    """

    if melting == "reference":
        return (scale * _melting_eta_reference(species, rho_p, fk, n0k, ilamk)).astype(f32)
    value = scale * _melting_eta(species, rho_p, fk, n0k, ilamk)
    ze = value.astype(f32)
    with np.errstate(all="ignore"):
        low = (value * (1.0 - ROUNDING_GUARD)).astype(f32)
        high = (value * (1.0 + ROUNDING_GUARD)).astype(f32)
    near = np.nonzero(~(low == high) | (value == 0.0) | ~np.isfinite(value))[0]
    if near.size:
        ze[near] = (scale * _melting_eta_reference(species, rho_p, fk[near], n0k[near], ilamk[near])).astype(f32)
    return ze


def calc_refl10cm(qv, qr, qs, qg, t, p, nr, ns, ng, rho, *, ihail: int,
                  cm1_n0_per_kg: bool = True, melting: str = "fast") -> np.ndarray:
    """CM1 r21.1 morrison.F calc_refl10cm over columns.

    Every input is ``[nz, ...]`` with ``k=0`` the lowest level (CM1's kts);
    trailing axes are independent columns.  Mixing ratios in kg/kg, numbers
    in kg-1, ``t`` in K, ``rho`` in kg m-3.  ``qv`` and ``p`` are accepted
    for signature parity; CM1 copies them into locals it never reads.
    Returns dBZ as float32 with the input shape.

    ``melting`` picks how the melting-layer integral is evaluated:
    ``"fast"`` (:func:`_melting_eta`, the default) or ``"reference"``
    (:func:`_melting_eta_reference`, the bin-by-bin loop this port had
    before, kept so the two can be compared on any frame).
    """

    if melting not in ("fast", "reference"):
        raise ValueError(f"melting must be 'fast' or 'reference', not {melting!r}")
    del qv, p  # read into locals by CM1, never used
    c = morrison_constants(ihail)
    arrs = np.broadcast_arrays(*[np.asarray(a, dtype=f32) for a in (qr, qs, qg, t, nr, ns, ng, rho)])
    shape = arrs[0].shape
    nz = shape[0]
    arrs = [np.ascontiguousarray(a).reshape(nz, -1) for a in arrs]
    qr, qs, qg, t, nr, ns, ng, rho = arrs

    l_qr, l_qs, l_qg = qr > R1, qs > R1, qg > R1
    rr = np.where(l_qr, qr * rho, R1).astype(f32)
    rs = np.where(l_qs, qs * rho, R1).astype(f32)
    rg = np.where(l_qg, qg * rho, R1).astype(f32)
    rnr = np.where(nr > R1, nr * rho, R1).astype(f32)
    rns = np.where(ns > R1, ns * rho, R1).astype(f32)
    rng = np.where(ng > R1, ng * rho, R1).astype(f32)

    ilams, n0_s = _slope_intercept(rs, rns, rho, c["cons1"], c["ds"], c["lammins"], c["lammaxs"], cm1_n0_per_kg)
    ilamg, n0_g = _slope_intercept(rg, rng, rho, c["cons2"], c["dg"], c["lamming"], c["lammaxg"], cm1_n0_per_kg)
    ilamr, n0_r = _slope_intercept(rr, rnr, rho, PI * c["rhow"], f32(3.0), c["lamminr"], c["lammaxr"],
                                   cm1_n0_per_kg)

    # Rayleigh: rain at all temperatures, dry snow/graupel as equivalent ice.
    rhosn, rhog = c["rhosn"], c["rhog"]
    six_pi = f32(6.0) / PI
    fac_s = (f32(0.176) / f32(0.93)) * six_pi * six_pi \
        * (PI * rhosn / f32(6.0) / f32(900.0)) * (PI * rhosn / f32(6.0) / f32(900.0))
    fac_g = (f32(0.176) / f32(0.93)) * six_pi * six_pi \
        * (PI * rhog / f32(6.0) / f32(900.0)) * (PI * rhog / f32(6.0) / f32(900.0))
    ze_rain = np.where(l_qr, (n0_r * 720.0 * ilamr ** 7).astype(f32), ZE_FLOOR)
    ze_snow = np.where(l_qs, (float(fac_s) * n0_s * 720.0 * ilams ** 7).astype(f32), ZE_FLOOR)
    ze_graupel = np.where(l_qg, (float(fac_g) * n0_g * 720.0 * ilamg ** 7).astype(f32), ZE_FLOOR)

    # Melting level: the highest k (scanning kte-1 down to kts) with T > 0 C,
    # rain > 1e-6 kg m-3 and snow+graupel > 1e-5 kg m-3 at k+1.
    cond = np.zeros_like(l_qr)
    cond[:-1] = (t[:-1] > T_0) & (rr[:-1] > f32(0.001e-3)) & ((rs[1:] + rg[1:]) > f32(0.01e-3))
    melti = cond.any(axis=0)
    k_hit = nz - 1 - np.argmax(cond[::-1], axis=0)  # highest True
    k0 = np.where(melti, k_hit + 1, -1)  # 0-based k_0; below it is the melt zone

    levels = np.arange(nz)[:, None]
    below = melti[None, :] & (levels < k0[None, :])
    if below.any():
        tabs = radar_tables()
        cols = np.arange(rs.shape[1])
        k0c = np.where(melti, k0, 0)
        rs0 = rs[k0c, cols][None, :]
        rg0 = rg[k0c, cols][None, :]
        scale = LAMDA4 / (PI5 * tabs["K_w"])
        for species in ("s", "g"):
            r = rs if species == "s" else rg
            r0 = rs0 if species == "s" else rg0
            fmelt = np.minimum(1.0 - (r / r0).astype(f32).astype(f64), 1.0)
            sel = below & (fmelt > 0.01) & (fmelt < 0.99) & (r > R1)
            kk, jj = np.nonzero(sel)
            if kk.size == 0:
                continue
            if species == "s":
                n0, ilam, rho_p = n0_s, ilams, rhosn
            else:
                n0, ilam, rho_p = n0_g, ilamg, rhog
            ze = _melting_ze(species, rho_p, fmelt[kk, jj], n0[kk, jj], ilam[kk, jj], scale, melting)
            if species == "s":
                ze_snow[kk, jj] = ze
            else:
                ze_graupel[kk, jj] = ze

    total = (ze_rain + ze_snow + ze_graupel).astype(f32)
    dbz = (10.0 * np.log10(total.astype(f64) * 1.0e18)).astype(f32)
    return dbz.reshape(shape)


def calccref(dbz: Any) -> np.ndarray:
    """``misclibs.F:2259-2292``: the column maximum of ``dbz`` from a start of -1000."""

    dbz = np.asarray(dbz, dtype=f32)
    cref = np.full(dbz.shape[1:], CREF_START, dtype=f32)
    for level in dbz:
        np.maximum(cref, level, out=cref)
    return cref


def default_chunk_rows(nx: int) -> int:
    """The rows of one column block: about 256 k columns, as the tool has always used."""

    return max(1, (1 << 18) // max(nx, 1))


def columns_dbz(qv, qr, qs, qg, t, nr, ns, ng, rho, *, ihail: int, cm1_n0_per_kg: bool = True,
                chunk_rows: int | None = None, melting: str = "fast") -> np.ndarray:
    """``calc_refl10cm`` over a ``(nz, ny, nx)`` block, a band of rows at a time.

    Every column is independent, so the bands change no word; they bound the
    melting branch's working memory on a large frame.
    """

    nz, ny, nx = np.shape(qr)
    t = np.broadcast_to(t, (nz, ny, nx))
    dbz = np.empty((nz, ny, nx), dtype=f32)
    step = chunk_rows or default_chunk_rows(nx)
    for j0 in range(0, ny, step):
        sl = (slice(None), slice(j0, min(j0 + step, ny)), slice(None))
        dbz[sl] = calc_refl10cm(qv[sl], qr[sl], qs[sl], qg[sl], t[sl], None, nr[sl], ns[sl], ng[sl],
                                rho[sl], ihail=ihail, cm1_n0_per_kg=cm1_n0_per_kg, melting=melting)
    return dbz


# --------------------------------------------------------------------------
# What a frame must hold, read through accessors so the converter's frames
# (cm1core.cm1out.frames) and the tool's own reader share every step.
# --------------------------------------------------------------------------


def _column(get: Callable[[str], Any], name: str, nz: int) -> np.ndarray:
    arr = get(name)
    if arr.ndim == 1:
        if arr.shape[0] != nz:
            raise ValueError(f"{name} column has {arr.shape[0]} levels, mass has {nz}")
        return arr[:, None, None]
    return arr


def temperature(get: Callable[[str], Any], has: Callable[[str], bool], nz: int,
                qv: Callable[[], Any] | None = None) -> tuple[np.ndarray, str]:
    """(T, source): T = (th0+th3d)*(pi0+pp3d), CM1 mp_driver.F:108.

    Built from whatever the frame carries, in this order: ``th`` or
    ``th3d``+``th0`` for theta, times ``pi`` or ``pp3d``+``pi0``/``base.pi0``
    or ``(prs/p00)**(rd/cp)``.  A frame without base-state theta falls back to
    inverting CM1's equation of state (solve2.F:1588-1591),
    ``T = prs / (rho*(rd + max(0,qv)*rv))``, from ``prs``, ``rho`` and qv
    (``qv`` returns the frame's water-vapour mixing ratio).
    """

    theta = None
    if has("th"):
        theta, tsrc = get("th"), "th"
    elif has("th3d") and has("th0"):
        theta, tsrc = get("th3d") + _column(get, "th0", nz), "th0+th3d"
    if theta is not None:
        if has("pi"):
            pi, psrc = get("pi"), "pi"
        elif has("pp3d") and (has("pi0") or has("base.pi0")):
            name = "pi0" if has("pi0") else "base.pi0"
            pi, psrc = get("pp3d") + _column(get, name, nz), f"{name}+pp3d"
        elif has("prs"):
            pi, psrc = (get("prs").astype(f64) / P00) ** (RD / CP), "(prs/p00)**(rd/cp)"
        else:
            raise KeyError("Exner function needs 'pi', 'pp3d'+'pi0'/'base.pi0', or 'prs'")
        return (theta * pi).astype(f32), f"({tsrc})*({psrc})"
    if has("prs") and has("rho") and qv is not None:
        vapour = qv()
        t = get("prs") / (get("rho") * (f32(RD) + np.maximum(f32(0.0), vapour) * f32(RV)))
        return t.astype(f32), "prs/(rho*(rd+max(0,qv)*rv)) (CM1 solve2.F:1588-1591 inverted)"
    raise KeyError(
        "temperature needs 'th', or 'th3d' with the base-state 'th0' (select "
        "output_basestate or output_th), or 'prs'+'rho'+'q3d' to invert the "
        "equation of state"
    )


def temperature_available(has: Callable[[str], bool]) -> bool:
    """Whether :func:`temperature` can be formed from carriers ``has`` names (q3d assumed present)."""

    theta = has("th") or (has("th3d") and has("th0"))
    exner = has("pi") or (has("pp3d") and (has("pi0") or has("base.pi0"))) or has("prs")
    return (theta and exner) or (has("prs") and has("rho"))


def resolve_ihail(microphysics: Any, recorded: Any, ihail: int | None, where: str) -> tuple[int, str]:
    """(ihail, source), refusing a non-Morrison frame or a conflicting ihail.

    The frame's recorded ``config.dense_ice.ihail`` is authoritative; an
    explicit ``ihail`` must agree with it.  Only a frame without a record may
    take ``ihail`` from the caller, and then the caller must give it.
    """

    if microphysics is not None and microphysics not in MORRISON_COMPONENTS:
        raise ValueError(f"frame {where} was made by microphysics {microphysics!r}, not Morrison "
                         f"({', '.join(MORRISON_COMPONENTS)}); calc_refl10cm does not apply")
    if recorded is not None:
        if recorded not in (0, 1):
            raise ValueError(f"frame {where} records ihail={recorded!r}, not 0 or 1")
        if ihail is not None and ihail != recorded:
            raise ValueError(f"frame {where} records ihail={recorded} (config.dense_ice.ihail) "
                             f"but ihail={ihail} was given; RHOG would be wrong (morrison.F:440-445)")
        return int(recorded), "history-publication.json config.dense_ice.ihail"
    if ihail is None:
        raise ValueError(f"frame {where} does not record ihail (no history-publication.json "
                         "config.dense_ice.ihail); pass the run's physics.ihail explicitly "
                         "(CLI --ihail)")
    return int(ihail), "caller (frame records none)"


def publication_physics(publication: Mapping[str, Any]) -> tuple[Any, Any]:
    """(microphysics id, ihail) from a frame's ``history-publication.json`` record."""

    config = publication.get("config") or {}
    dense = config.get("dense_ice") or {}
    ihail = dense.get("ihail") if isinstance(dense, dict) else None
    return config.get("microphysics"), ihail


# --------------------------------------------------------------------------
# The converter's frames (cm1core.cm1out.frames.Frame)
# --------------------------------------------------------------------------


class NotComputable(ValueError):
    """Why a frame's reflectivity cannot be formed, in plain words."""


def _species_rows(frame: Any) -> tuple[dict[str, int], dict[str, int]]:
    from .settings import SettingsError, species

    try:
        axis = species(frame)
    except SettingsError as error:
        raise NotComputable(str(error)) from error
    mass = {name: row for row, name in enumerate(axis.mass_names)}
    numbers = [name for name in axis.names if name not in mass]
    return mass, {name: row for row, name in enumerate(numbers)}


def _is_initial(frame: Any) -> bool:
    """A frame published before the first step: CM1's dbz there is its zero start."""

    return int(frame.step) == 0 and float(frame.model_time_s) == 0.0


def _in_model_zero_start(frame: Any) -> bool:
    """The initial frame of a run whose scheme computes reflectivity in the model (``output_dbz``).

    The runtime computes ``dbz``/``cref`` only inside the microphysics of a
    large step, so it leaves them out of the frame taken before the first one
    (``history-publication.json`` ``reflectivity.omitted_at_model_time_zero``).
    CM1 writes its ``qdiag`` there, which starts at zero (``cm1.F:2467-2468``),
    and ``calccref``'s ``max(-1000, 0)`` is zero too: its own t = 0 file holds
    ``dbz = cref = 0``.
    """

    if not _is_initial(frame):
        return False
    record = frame.publication.get("reflectivity") or {}
    return bool(record.get("output_dbz")) and not record.get("carriers")


def _check(frame: Any, ihail: int | None = None) -> tuple[int | None, str | None]:
    """(ihail, its source) when this frame's reflectivity can be formed; raises :class:`NotComputable`.

    The initial frame needs no ihail (its words are CM1's zero start), so
    its ihail comes back None.
    """

    microphysics, recorded = publication_physics(frame.publication)
    if _in_model_zero_start(frame):
        # Any scheme: nothing has been computed yet, and CM1 writes zeros.
        return None, None
    if microphysics is None:
        raise NotComputable("The frame does not say which microphysics made it.")
    if microphysics not in MORRISON_COMPONENTS:
        raise NotComputable(
            "GM1 computes reflectivity after the run only for Morrison (CM1's calc_refl10cm); "
            f"this frame was made by {microphysics}, whose own formula is not ported."
        )
    if not frame.has("q3d"):
        raise NotComputable("The frame has no q3d carrier, which CM1's calc_refl10cm reads.")
    if _is_initial(frame):
        # Nothing is read at the initial time: CM1's words there are its zero start.
        return None, None
    for need in ("q3d", "nc3d", "rho"):
        if not frame.has(need):
            raise NotComputable(f"The frame has no {need} carrier, which CM1's calc_refl10cm reads.")
        stagger = str(frame.carriers[need].stagger)
        if not stagger.startswith("mass"):
            raise NotComputable(f"The frame's {need} is on the {stagger!r} grid, not at scalar points.")
    mass, numbers = _species_rows(frame)
    missing = [name for name in ("qv", "qr", "qs", "qg") if name not in mass]
    missing += [name for name in ("ncr", "ncs", "ncg") if name not in numbers]
    if missing:
        raise NotComputable(f"The frame's species axis has no {', '.join(missing)}.")
    if not temperature_available(frame.has):
        raise NotComputable(
            "The frame cannot give the temperature that places the melting level: it needs th, or th3d "
            "with the base-state th0 and the Exner function (pi, or pp3d with pi0), or prs with rho."
        )
    try:
        return resolve_ihail(microphysics, recorded, ihail, str(frame.directory))
    except ValueError as error:
        raise NotComputable(
            "The frame does not record the hail switch (ihail) that sets the graupel density "
            "calc_refl10cm integrates with."
        ) from error


def _mass_shape(frame: Any) -> tuple[int, int, int]:
    """(nz, ny, nx) of the frame's scalar grid, from its recorded grid (or its q3d carrier)."""

    config = frame.publication.get("config") or {}
    grid = config.get("grid") or (frame.publication.get("profile") or {}).get("geometry")
    if grid:
        return int(grid["nz"]), int(grid["ny"]), int(grid["nx"])
    return tuple(int(n) for n in frame.array("q3d").shape[1:])  # type: ignore[return-value]


def frame_reflectivity(frame: Any, *, ihail: int | None = None, chunk_rows: int | None = None,
                       info: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(dbz ``[nz, ny, nx]``, cref ``[ny, nx]``), float32, for one Morrison frame.

    ``frame`` is a :class:`cm1core.cm1out.frames.Frame`.  Raises
    :class:`NotComputable` with the reason when the frame is not Morrison's or
    lacks what ``calc_refl10cm`` reads.  A frame taken before the first step
    gives CM1's zero start, for Morrison and for a scheme that computes its
    reflectivity in the model (:func:`_in_model_zero_start`).
    """

    ihail_used, ihail_source = _check(frame, ihail)
    if ihail_used is None:
        nz, ny, nx = _mass_shape(frame)
        # CM1's qdiag starts at zero and no microphysics step has filled it:
        # its own t = 0 file holds dbz = 0 and calccref's max(-1000, 0) = 0.
        if info is not None:
            info.update({"model_time_s": frame.model_time_s, "initial_time": True,
                         "note": "CM1 writes dbz = cref = 0 before its first microphysics step"})
        return np.zeros((nz, ny, nx), dtype=f32), np.zeros((ny, nx), dtype=f32)
    mass, numbers = _species_rows(frame)

    def get(name: str) -> np.ndarray:
        return np.asarray(frame.array(name), dtype=f32)

    q = frame.array("q3d")
    nz, ny, nx = q.shape[1:]
    nc = frame.array("nc3d")
    qv = np.asarray(q[mass["qv"]], dtype=f32)
    qr = np.asarray(q[mass["qr"]], dtype=f32)
    qs = np.asarray(q[mass["qs"]], dtype=f32)
    qg = np.asarray(q[mass["qg"]], dtype=f32)
    nr = np.asarray(nc[numbers["ncr"]], dtype=f32)
    ns = np.asarray(nc[numbers["ncs"]], dtype=f32)
    ng = np.asarray(nc[numbers["ncg"]], dtype=f32)
    rho = get("rho")
    for name, arr in (("ncr", nr), ("ncs", ns), ("ncg", ng), ("rho", rho)):
        if arr.shape != qr.shape:
            raise NotComputable(f"The frame's {name} has shape {arr.shape}, the mass species {qr.shape}.")
    try:
        t, how = temperature(get, frame.has, nz, qv=lambda: qv)
    except (KeyError, ValueError) as error:
        raise NotComputable(f"The frame cannot give the temperature: {error}.") from error
    dbz = columns_dbz(qv, qr, qs, qg, t, nr, ns, ng, rho, ihail=ihail_used, chunk_rows=chunk_rows)
    if info is not None:
        info.update({"model_time_s": frame.model_time_s, "temperature_source": how,
                     "ihail": ihail_used, "ihail_source": ihail_source, "authority": AUTHORITY})
    return dbz, calccref(dbz)


#: One computation per frame: ``dbz`` and ``cref`` are read one after the other.
_CACHE: "weakref.WeakKeyDictionary[Any, tuple[np.ndarray, np.ndarray]]" = weakref.WeakKeyDictionary()


def reason(frame: Any) -> str | None:
    """Why this frame's reflectivity cannot be formed, or None when it can."""

    try:
        _check(frame)
    except NotComputable as error:
        return str(error)
    return None


def reader(frame: Any, name: str) -> Callable[[Any], np.ndarray] | None:
    """The converter's hook (``cm1out.convert``): a reader of ``dbz`` or ``cref``, or None.

    None when the frame is not a Morrison frame or lacks what
    ``calc_refl10cm`` reads; the converter then names the field as not
    produced, with :func:`not_produced_reason`.
    """

    if name not in ("dbz", "cref") or reason(frame) is not None:
        return None

    def read(f: Any) -> np.ndarray:
        cached = _CACHE.get(f)
        if cached is None:
            cached = frame_reflectivity(f)
            _CACHE[f] = cached
        return cached[0] if name == "dbz" else cached[1]

    return read


def not_produced_reason(frame: Any) -> str:
    """The plain-words reason ``dbz``/``cref`` is not in a file made from this frame."""

    return reason(frame) or "The frame's reflectivity could not be formed."


__all__ = [
    "AUTHORITY",
    "MORRISON_COMPONENTS",
    "NotComputable",
    "calc_refl10cm",
    "calccref",
    "columns_dbz",
    "frame_reflectivity",
    "morrison_constants",
    "not_produced_reason",
    "publication_physics",
    "radar_tables",
    "rayleigh_soak_wetgraupel",
    "reader",
    "reason",
    "resolve_ihail",
    "temperature",
]
