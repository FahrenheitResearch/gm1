"""CM1 r21.1's two-part near-wall subgrid models, ``sgsmodel=3`` and ``=4``.

What CM1 r21.1 actually has
---------------------------

``README.namelist:334-352`` lists seven values of ``sgsmodel``: 0 (no model),
1 (Deardorff TKE), 2 (Smagorinsky), 3 (TKE plus the Sullivan et al. 1994
two-part model, "S94"), 4 (TKE plus the Bryan 2020 two-part model, "B20"),
5 and 6 (the Mirocha et al. 2010 nonlinear backscatter and anisotropy
models).  The two-part models are the only near-wall variants.  Both keep the
Deardorff prognostic-TKE closure untouched (``src/turb.F:1348`` runs ``tkekm``
for 1, 3, 4 and 5 alike) and add a near-wall piece
(``src/param.F:8861-8936``, ``dot2p``):

* ``gamwall(k) = 1-0.5*(1+tanh((zf-gam_cp)/gam_dr))`` with the hard-coded
  ``gam_cp=150 m`` and ``gam_dr=25 m`` (``src/param.F:8869-8888``), and
  ``ntwk`` the highest ``w`` level with ``gamwall > 0.01``.
* ``t2pfac``, a hard-coded linear ramp from model time 1800 s to 3600 s
  (``src/turb.F:264-270``).  Nothing two-part acts before 1800 s.

**S94 (sgsmodel=3).**  ``t2psmm`` (``src/turb.F:5694-6068``) computes the
isotropy factor ``gamk(k) = S'/(S' + gamwall*<S>)`` from the domain-mean
fluctuating and mean strain.  ``src/turb.F:1612-1628`` and
``src/solve1.F:296-310`` then hand ``gamk*kmh``/``gamk*kmv`` -- not the
coefficients themselves -- to ``gettau``, ``turbt``, ``turbu``, ``turbv`` and
``turbw``.  ``t2psmm`` also forms a mean-shear viscosity ``kmw``/``ufw``, but
in r21.1 that profile is never copied into ``ufwk``, the array
``turbtend.F:1779`` reads: ``ufwk`` is only filled for sgsmodel=4
(``src/turb.F:1497-1504``) or by ``t2pcodetavg``.  The S94 mean-shear flux
therefore adds exactly ``tem*(0-0)`` in r21.1, and this port adds exactly
that.

**B20 (sgsmodel=4).**  ``getgamk`` (``src/turb.F:6456-6615``) and
``t2pcode`` (``src/turb.F:6078-6267``) form a horizontally uniform
mean-shear eddy viscosity from the momentum budget: at each ``w`` level the
viscosity is the one whose flux divergence cancels the domain-mean tendency of
everything else (``u2pt``, saved by ``src/solve3.F:342-352``), divided by the
log-law shear of the domain-mean ``ust``/``znt``.  ``turbuz``/``turbvz``
(``src/turbtend.F:1767-1802``, ``:2424-2459``) add that flux's divergence to
the vertical stress tendency and save it back into ``u2pt``.  ``getgamk``'s
``gamk`` is a diagnostic here: nothing prognostic reads it for sgsmodel=4.

``t2p_avg`` (``README.namelist:1089-1092``) selects the domain average (1) or
CM1's per-column time average (2, ``t2pcodetavg``).  Only 1 is ported.

Neutral surface layer
---------------------

Both codes evaluate ``stabil_funcs`` only when ``|<mol>| > 1e-6``.  For
``sfcmodel=1`` -- the only surface model this lane has, and the one
``bbc=3`` requires -- ``sfcdiags`` sets ``mol=0`` everywhere
(``src/sfcphys.F:529-535``, "impose neutral sfc layer"), so r21.1 always takes
the neutral branch ``tpsim=0``.  The stability functions are therefore not a
path this lane can reach and are not transcribed.

Arithmetic
----------

Every domain sum is CM1's own serial ``j``-outer, ``i``-inner binary64
accumulation of binary32 terms, executed on one device thread per level, so the
means are raw-word the Fortran order, not a tree reduction.  Every other
statement keeps CM1's operand order and its real/double promotion.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import importlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .errors import refuse
from .nvrtc_module import compile_nvrtc_module


TWO_PART_SCHEMA = "gm1.cm1-two-part-sgs/v1"
TWO_PART_GATE = "cartesian-runtime"

#: src/param.F:8869-8871.  Not namelist words: CM1 hard-codes them.
GAM_CP_M = 150.0
GAM_DR_M = 25.0
#: src/turb.F:266.  The two-part ramp runs from 1800 s to 3600 s.
T2P_RAMP_START_S = 1800.0
#: src/constants.F:23.
KARMAN = 0.40
#: src/cm1.F:311, the single-precision build.
SMEPS = 1.0e-30

TWO_PART_AUTHORITY = MappingProxyType(
    {
        "options": "README.namelist:334-352",
        "t2p_avg": "README.namelist:1089-1092; src/param.F:8899-8908",
        "terrain_stop": "src/param.F:1405-1420",
        "gamwall_ntwk": "src/param.F:8861-8936",
        "t2pfac_ramp": "src/turb.F:264-270",
        "dispatch": "src/turb.F:1456-1537",
        "s94_gamk": "src/turb.F:5694-6068 (t2psmm)",
        "s94_scaled_coefficients": "src/turb.F:1612-1628; src/solve1.F:296-310,1136-1150",
        "b20_means": "src/turb.F:6456-6615 (getgamk)",
        "b20_kmw": "src/turb.F:6078-6267 (t2pcode)",
        "ufwk_copy": "src/turb.F:1497-1504",
        "turbuz_turbvz_flux": "src/turbtend.F:1767-1802; src/turbtend.F:2424-2459",
        "u2pt_solve3": "src/solve3.F:340-352",
        "neutral_surface_layer": "src/sfcphys.F:529-535 (sfcdiags, sfcmodel=1)",
        "restart_skip": "src/turb.F:1460-1466; src/cm1.F:2663,4404",
        "restart_inventory": "src/restart_write.F:1593-1620",
        "smeps": "src/cm1.F:311",
        "karman": "src/constants.F:23",
    }
)

TWO_PART_NAMES = MappingProxyType(
    {
        3: "Deardorff TKE + Sullivan, McWilliams and Moeng (1994) two-part model",
        4: "Deardorff TKE + Bryan (2020) two-part model",
    }
)

#: The sgsmodel values this lane integrates.
ADMITTED_SGSMODELS = (1, 3, 4)


# ---------------------------------------------------------------- selection


@dataclass(frozen=True, slots=True)
class TwoPartSelection:
    sgsmodel: int
    t2p_avg: int = 1

    @property
    def active(self) -> bool:
        return self.sgsmodel in (3, 4)

    def record(self) -> dict[str, object]:
        return {
            "schema": TWO_PART_SCHEMA,
            "sgsmodel": self.sgsmodel,
            "t2p_avg": self.t2p_avg,
            "name": TWO_PART_NAMES[self.sgsmodel],
            "average": "domain mean at each model level (t2p_avg=1)",
            "gam_cp_m": GAM_CP_M,
            "gam_dr_m": GAM_DR_M,
            "t2pfac_ramp_s": [T2P_RAMP_START_S, 2.0 * T2P_RAMP_START_S],
            "surface_layer_stability": (
                "neutral: sfcmodel=1's sfcdiags sets mol=0 "
                "(src/sfcphys.F:529-535), so |<mol>|<=1e-6 selects tpsim=0"
            ),
            "authority": dict(TWO_PART_AUTHORITY),
        }


def validate_two_part_selection(
    *,
    sgsmodel: int,
    t2p_avg: int,
    t2p_avg_declared: bool,
    terrain: bool,
    bbc: int,
    sfcmodel: int,
    option_prefix: str = "closures",
) -> TwoPartSelection:
    """Refuse every two-part selection this lane would integrate as another."""

    if sgsmodel not in ADMITTED_SGSMODELS:
        refuse(
            code="CM1-CFG-SGSMODEL-NOT-PORTED",
            gate=TWO_PART_GATE,
            reason=(
                f"sgsmodel={sgsmodel} selects "
                + {
                    0: "no subgrid model (README.namelist:334-352)",
                    2: "the Smagorinsky scheme (src/turb.F:1403-1413)",
                    5: "the TKE nonlinear backscatter and anisotropy model (src/turb.F:1391-1401)",
                    6: "the Smagorinsky nonlinear backscatter and anisotropy model (src/turb.F:1415-1426)",
                }.get(sgsmodel, "no CM1 r21.1 subgrid model (src/param.F:1386-1398 stops)")
                + "; this lane has kernels only for the Deardorff TKE closure "
                "and its two-part near-wall variants, so the run would mix "
                "every column with the TKE closure while its receipt named "
                "another model"
            ),
            remedy=f"set {option_prefix}.sgs_model to 1, 3 or 4",
            option=f"{option_prefix}.sgs_model",
            value=sgsmodel,
        )
    # Every CM1 r21.1 namelist ships t2p_avg=1 (run/config_files/*/
    # namelist.input), and CM1 ignores it unless dot2p is on, so that value
    # beside the plain TKE closure is CM1's own deck.  Any other value there is
    # a selection no run of this model reads.
    if t2p_avg_declared and sgsmodel not in (3, 4) and t2p_avg != 1:
        refuse(
            code="CM1-CFG-T2P-AVG-WITHOUT-TWO-PART",
            gate=TWO_PART_GATE,
            reason=(
                "t2p_avg selects how the two-part models average "
                "(README.namelist:1089-1092) and src/param.F:8865-8908 reads "
                f"it only for sgsmodel=3 or 4; with sgsmodel={sgsmodel}, "
                f"t2p_avg={t2p_avg} is a number nothing reads, and the receipt "
                "would name an averaging the run never applied"
            ),
            remedy=f"remove {option_prefix}.t2p_avg or set it to CM1's 1, or set {option_prefix}.sgs_model to 3 or 4",
            option=f"{option_prefix}.t2p_avg",
            value=t2p_avg,
        )
    if sgsmodel not in (3, 4):
        return TwoPartSelection(sgsmodel=sgsmodel, t2p_avg=t2p_avg)
    if t2p_avg not in (1, 2):
        refuse(
            code="CM1-CFG-T2P-AVG-RANGE",
            gate=TWO_PART_GATE,
            reason=(
                f"t2p_avg={t2p_avg} is neither CM1's spatial average (1) nor "
                "its time average (2); src/param.F:8899-8902 stops the model"
            ),
            remedy=f"set {option_prefix}.t2p_avg to 1",
            option=f"{option_prefix}.t2p_avg",
            value=t2p_avg,
        )
    if t2p_avg == 2:
        refuse(
            code="CM1-CFG-T2P-AVG-TIME-AVERAGE-NOT-PORTED",
            gate=TWO_PART_GATE,
            reason=(
                "t2p_avg=2 runs t2pcodetavg (src/turb.F:6277-6447), whose "
                "per-column kmwk/ufwk come from the running time averages "
                "timavg/sfctimavg that cm1.F:3555-3630's tavg_plus1/tavg_shift "
                "machinery maintains; this lane carries none of those arrays, "
                "so the run would integrate the domain-average profile of "
                "t2p_avg=1 while its receipt named per-column time averages"
            ),
            remedy=f"set {option_prefix}.t2p_avg to 1",
            option=f"{option_prefix}.t2p_avg",
            value=t2p_avg,
        )
    selection = TwoPartSelection(sgsmodel=sgsmodel, t2p_avg=t2p_avg)
    if not selection.active:
        return selection
    if terrain:
        refuse(
            code="CM1-CFG-TWO-PART-TERRAIN",
            gate=TWO_PART_GATE,
            reason=(
                "CM1 r21.1 stops with 'cannot use terrain with sgsmodel = "
                "3,4,5,6' (src/param.F:1405-1420): the two-part profiles read "
                "mh, mf and zf of column (1,1) for the whole domain, which is "
                "every column's metric only over flat ground"
            ),
            remedy="set grid.terrain = false, or closures.sgs_model = 1",
            option=f"{option_prefix}.sgs_model",
            value=sgsmodel,
        )
    if sgsmodel == 4 and (bbc != 3 or sfcmodel != 1):
        refuse(
            code="CM1-CFG-B20-WITHOUT-SURFACE-LAYER",
            gate=TWO_PART_GATE,
            reason=(
                "t2pcode forms the log-law wind speed "
                "(ustbar/karman)*alog(zh/zntbar) from the domain-mean friction "
                "velocity and roughness length (src/turb.F:6186-6195); without "
                "CM1's bbc=3 surface layer both are zero, alog(zh/0) is "
                "infinite, 0*inf is not a number, and kmw, the two-part flux "
                "and every near-wall momentum tendency would be NaN"
            ),
            remedy=(
                f"set {option_prefix}.bbc = 3 and {option_prefix}.sfcmodel = 1, "
                f"or {option_prefix}.sgs_model = 3 or 1"
            ),
            option=f"{option_prefix}.bbc",
            value=bbc,
        )
    return selection


# ---------------------------------------------------------------- host words


def _libm_f32(*names: str) -> tuple[Any, ...]:
    if os.name == "nt":
        library = ctypes.CDLL("ucrtbase")
    else:
        found = ctypes.util.find_library("m")
        library = ctypes.CDLL(found) if found else ctypes.CDLL(None)
    bound = []
    for name in names:
        function = getattr(library, name)
        function.argtypes = [ctypes.c_float]
        function.restype = ctypes.c_float
        bound.append(function)
    return tuple(bound)


def _f32(value: Any) -> Any:
    import numpy as np

    return np.float32(value)


def gamwall_profile(zf: Any) -> tuple[Any, int]:
    """``src/param.F:8879-8888`` from the ``nk+1`` column-(1,1) ``zf`` words.

    Returns ``(gamwall, ntwk)`` with ``gamwall[p]`` CM1's ``gamwall(p+1)`` and
    ``ntwk`` CM1's 1-based level count.  ``tanh`` is the platform libm's
    binary32 ``tanhf``, which is what the Fortran default-REAL intrinsic calls.
    """

    import numpy as np

    (tanhf,) = _libm_f32("tanhf")
    zf = np.asarray(zf, dtype=np.float32)
    gamwall = np.empty(zf.shape, np.float32)
    ntwk = 0
    one, half = np.float32(1.0), np.float32(0.5)
    for p, value in enumerate(zf):
        arg = np.float32((np.float32(value) - np.float32(GAM_CP_M)) / np.float32(GAM_DR_M))
        t = np.float32(tanhf(ctypes.c_float(float(arg))))
        gamwall[p] = one - half * (one + t)
        if gamwall[p] > np.float32(0.01):
            ntwk = p + 1
    return gamwall, max(2, ntwk)


def t2pfac_f32(model_time_s: float) -> Any:
    """``src/turb.F:266-268`` on CM1's ``rtime=sngl(mtime)`` (cm1.F:3724)."""

    import numpy as np

    rtime = np.float32(model_time_s)
    fac = np.float32((rtime - np.float32(1800.0)) / np.float32(1800.0))
    fac = fac if fac > np.float32(0.0) else np.float32(0.0)
    fac = fac if fac < np.float32(1.0) else np.float32(1.0)
    return np.float32(fac)


# ---------------------------------------------------------------- CPU references
#
# GM1 layouts, 0-based: a profile of length nk+1 holds CM1's k=1..nk+1 at
# index k-1; mass level m is CM1's k=m+1; ``u`` planes are (ny, nx+1) and
# ``v`` planes (ny+1, nx), of which CM1 sums i=1..ni, j=1..nj.


def cpu_plane_sum(plane: Any, ni: int, nj: int, *, shift: Any = None) -> float:
    """CM1's ``s = 0.0; s = s + x(i,j)`` for j=1..nj, i=1..ni, in binary64."""

    import numpy as np

    total = 0.0
    for j in range(nj):
        for i in range(ni):
            x = np.float32(plane[j, i])
            if shift is not None:
                x = np.float32(x + np.float32(shift))
            total = total + float(x)
    return total


def cpu_plane_product_sum(a: Any, b: Any, ni: int, nj: int) -> float:
    import numpy as np

    total = 0.0
    for j in range(nj):
        for i in range(ni):
            total = total + float(np.float32(np.float32(a[j, i]) * np.float32(b[j, i])))
    return total


def cpu_mean_profiles(
    ua: Any, va: Any, mf: Any, *, nx: int, ny: int, ntwk: int, rdz: Any,
    umove: Any, vmove: Any,
) -> dict[str, Any]:
    """``getgamk``/``t2psmm`` level means, ``savg`` and ``s13b``/``s23b``."""

    import numpy as np

    f = np.float32
    sqrtf, = _libm_f32("sqrtf")
    temd = 1.0 / float(nx * ny)
    uavg = np.zeros(ntwk, np.float32)
    vavg = np.zeros(ntwk, np.float32)
    savg = np.zeros(ntwk, np.float32)
    for m in range(ntwk):
        uavg[m] = f(cpu_plane_sum(ua[m], nx, ny, shift=umove) * temd)
        vavg[m] = f(cpu_plane_sum(va[m], nx, ny, shift=vmove) * temd)
        r = f(sqrtf(ctypes.c_float(float(f(f(uavg[m] * uavg[m]) + f(vavg[m] * vavg[m]))))))
        savg[m] = r if r > f(1.0e-10) else f(1.0e-10)
    s13b = np.zeros(ntwk + 1, np.float32)
    s23b = np.zeros(ntwk + 1, np.float32)
    for p in range(1, ntwk):
        s13b[p] = f(f(f(f(0.5) * f(uavg[p] - uavg[p - 1])) * f(rdz)) * f(mf[p]))
        s23b[p] = f(f(f(f(0.5) * f(vavg[p] - vavg[p - 1])) * f(rdz)) * f(mf[p]))
    return {"uavg": uavg, "vavg": vavg, "savg": savg, "s13b": s13b, "s23b": s23b}


def cpu_b20_profile(
    *, uavg: Any, vavg: Any, savg: Any, u2pt: Any, v2pt: Any, rf: Any, rr: Any,
    ust: Any, znt: Any, zh: Any, mh: Any, mf: Any, gamwall: Any,
    nx: int, ny: int, nk: int, ntwk: int, rdz: Any, t2pfac: Any,
) -> dict[str, Any]:
    """``t2pcode`` (src/turb.F:6146-6258), neutral branch, t2p_avg=1."""

    import numpy as np

    f = np.float32
    logf, = _libm_f32("logf")
    kmw = np.zeros(nk + 1, np.float32)
    ufw = np.zeros(nk + 1, np.float32)
    vfw = np.zeros(nk + 1, np.float32)
    wspa = np.zeros(ntwk + 1, np.float32)
    if not f(t2pfac) >= f(0.001):
        return {"kmw": kmw, "ufw": ufw, "vfw": vfw, "wspa": wspa}
    teme = 1.0 / float(nx * ny)
    ustbar = cpu_plane_sum(ust, nx, ny) * teme
    zntbar = cpu_plane_sum(znt, nx, ny) * teme
    for m in range(ntwk + 1):
        ratio = f(f(zh[m]) / f(zntbar))
        level = f(f(logf(ctypes.c_float(float(ratio)))) - f(0.0))
        wspa[m] = f((ustbar / float(f(KARMAN))) * float(level))
    rdz = f(rdz)
    t2pfac = f(t2pfac)
    for m in range(ntwk - 1):
        t13avg = cpu_plane_sum(u2pt[m], nx, ny) * teme
        t23avg = cpu_plane_sum(v2pt[m], nx, ny) * teme
        rfavg = cpu_plane_product_sum(rf[m + 1], rr[m], nx, ny) * teme
        d2pm1 = f((float(uavg[m]) * t13avg + float(vavg[m]) * t23avg) / float(savg[m]))
        head = f(f(uavg[m] * ufw[m]) + f(vavg[m] * vfw[m]))
        d2pm3 = f((((float(head) * rfavg) * float(rdz)) * float(mh[m])) / float(savg[m]))
        dsdz = f(f(f(wspa[m + 1] - wspa[m]) * rdz) * f(mf[m + 1]))
        bounded = f(1.0e-12) if f(1.0e-12) > dsdz else dsdz
        tem = f(float(f(t2pfac * f(gamwall[m + 1]))) / (float(f(f(rdz * f(mh[m])) * bounded)) * rfavg))
        x = f(f(f(f(0.0) - d2pm1) + d2pm3) * tem)
        kmw[m + 1] = x if x > f(0.0) else f(0.0)
        ufw[m + 1] = f(f(f(kmw[m + 1] * f(uavg[m + 1] - uavg[m])) * rdz) * f(mf[m + 1]))
        vfw[m + 1] = f(f(f(kmw[m + 1] * f(vavg[m + 1] - vavg[m])) * rdz) * f(mf[m + 1]))
    return {"kmw": kmw, "ufw": ufw, "vfw": vfw, "wspa": wspa}


def cpu_s94_point_terms(
    *, s11: Any, s22: Any, s33: Any, s12: Any, s13: Any, s23: Any,
    c1: Any, c2: Any, rf: Any, s13b: Any, s23b: Any, ni: int, nj: int, ntwk: int,
) -> Any:
    """``t2psmm``'s per-point strain term (src/turb.F:5859-5881)."""

    import numpy as np

    f = np.float32

    def sq(x: Any) -> Any:
        return f(f(x) * f(x))

    term = np.zeros((ntwk + 1, nj, ni), np.float32)
    for p in range(1, ntwk):
        for j in range(nj):
            for i in range(ni):
                a1, a2 = f(c1[p, j, i]), f(c2[p, j, i])
                tmp11 = f(f(a1 * sq(s11[p - 1, j, i])) + f(a2 * sq(s11[p, j, i])))
                tmp22 = f(f(a1 * sq(s22[p - 1, j, i])) + f(a2 * sq(s22[p, j, i])))
                tmp33 = f(f(a1 * sq(s33[p - 1, j, i])) + f(a2 * sq(s33[p, j, i])))
                lo = f(f(sq(s12[p - 1, j, i]) + sq(s12[p - 1, j + 1, i + 1])) + f(sq(s12[p - 1, j + 1, i]) + sq(s12[p - 1, j, i + 1])))
                hi = f(f(sq(s12[p, j, i]) + sq(s12[p, j + 1, i + 1])) + f(sq(s12[p, j + 1, i]) + sq(s12[p, j, i + 1])))
                tmp12 = f(f(0.25) * f(f(a1 * lo) + f(a2 * hi)))
                tmp13 = f(f(0.5) * f(sq(f(s13[p, j, i] - s13b[p])) + sq(f(s13[p, j, i + 1] - s13b[p]))))
                tmp23 = f(f(0.5) * f(sq(f(s23[p, j, i] - s23b[p])) + sq(f(s23[p, j + 1, i] - s23b[p]))))
                num = f(f(f(4.0) * f(tmp12 + f(tmp13 + tmp23))) + f(f(2.0) * f(f(tmp11 + tmp22) + tmp33)))
                term[p, j, i] = f(num / sq(rf[p, j, i]))
    return term


def cpu_s94_gamk(
    *, term: Any, s13b: Any, s23b: Any, gamwall: Any, gamk: Any,
    nx: int, ny: int, ntwk: int, t2pfac: Any,
) -> Any:
    """``t2psmm``'s gamk (src/turb.F:5883-5896), in place on a copy."""

    import numpy as np

    f = np.float32
    sqrtf, = _libm_f32("sqrtf")
    out = np.array(gamk, dtype=np.float32, copy=True)
    if not f(t2pfac) >= f(0.001):
        return out
    teme = 1.0 / float(nx * ny)
    for p in range(1, ntwk):
        spavg = math.sqrt(cpu_plane_sum(term[p], nx, ny) * teme)
        inner = f(f(4.0) * f(f(s13b[p] * s13b[p]) + f(s23b[p] * s23b[p])))
        shravg = float(f(sqrtf(ctypes.c_float(float(inner)))))
        out[p] = f(spavg / (float(f(SMEPS)) + (spavg + float(f(gamwall[p])) * shravg)))
    out[0] = out[1]
    return out


def cpu_scale_km(gamk: Any, km: Any) -> Any:
    """``gamk(k)*kmh(i,j,k)`` (src/turb.F:1624-1625; src/solve1.F:306-307)."""

    import numpy as np

    return (np.asarray(gamk, np.float32)[:, None, None] * np.asarray(km, np.float32)).astype(np.float32)


def cpu_solve3_t2pt(current: Any, baseline: Any, t2pt: Any, dt: Any) -> Any:
    """``u2pt = (u3d-ua)*rdt - u2pt`` with ``rdt=1.0/dt`` (src/solve3.F:342-349)."""

    import numpy as np

    f = np.float32
    rdt = f(f(1.0) / f(dt))
    difference = np.asarray(current, np.float32) - np.asarray(baseline, np.float32)
    return (difference * rdt - np.asarray(t2pt, np.float32)).astype(np.float32)


def cpu_turbuz_two_part_increment(
    *, rf: Any, rho: Any, mh00: Any, tfw: Any, ntwk: int, rdz: Any,
    left: Any, right: Any,
) -> tuple[Any, Any]:
    """One column of ``turbuz``'s ``dot2p`` block (src/turbtend.F:1771-1802).

    ``rf``/``rho`` are the two columns either side of the face (``left`` and
    ``right`` are the ``(rf_column, rho_column)`` pairs); returns the per-level
    ``tem*(dum2(k+1)-dum2(k))`` added to ``turbz`` and the saved ``u2pt``.
    """

    import numpy as np

    f = np.float32
    rf_l, rho_l = left
    rf_r, rho_r = right
    dum2 = np.zeros(ntwk + 1, np.float32)
    for w in range(1, ntwk):
        dum2[w] = f(f(0.5) * f(f(rf_l[w] * tfw[w]) + f(rf_r[w] * tfw[w])))
    increment = np.zeros(ntwk, np.float32)
    saved = np.zeros(ntwk, np.float32)
    for m in range(ntwk):
        tem = f(f(rdz) * f(mh00[m]))
        increment[m] = f(tem * f(dum2[m + 1] - dum2[m]))
        rru0 = f(f(1.0) / f(f(0.5) * f(rho_l[m] + rho_r[m])))
        saved[m] = f(f(tem * f(dum2[m + 1] - dum2[m])) * rru0)
    return increment, saved


# ---------------------------------------------------------------- CUDA text

CUDA_TWO_PART_KERNEL_NAMES = (
    "cm1_t2p_plane_sum_v1",
    "cm1_t2p_plane_product_sum_v1",
    "cm1_t2p_mean_profiles_v1",
    "cm1_t2p_b20_profile_v1",
    "cm1_t2p_s94_point_term_v1",
    "cm1_t2p_s94_gamk_v1",
    "cm1_t2p_scale_km_v1",
    "cm1_t2p_solve3_v1",
)

CUDA_TWO_PART_SOURCE = r'''// gpuwm-cm1.two-part-sgs-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
static_assert(sizeof(double)==8,"CM1 double precision requires binary64");
__device__ __forceinline__ float TA(float x,float y){return __fadd_rn(x,y);}
__device__ __forceinline__ float TS(float x,float y){return __fsub_rn(x,y);}
__device__ __forceinline__ float TM(float x,float y){return __fmul_rn(x,y);}
__device__ __forceinline__ float TD(float x,float y){return __fdiv_rn(x,y);}
__device__ __forceinline__ unsigned long long T3(unsigned k,unsigned j,unsigned i,unsigned nj,unsigned ni){return ((unsigned long long)k*nj+j)*ni+i;}
// CM1 s=0.0; s=s+x(i,j) for j=1..nj, i=1..ni: one thread, source order.
extern "C" __global__ void cm1_t2p_plane_sum_v1(const float*a,unsigned ni,unsigned nj,unsigned width,float shift,unsigned shifted,double*out,unsigned slot){if(blockIdx.x!=0u||threadIdx.x!=0u)return;double s=0.0;for(unsigned j=0u;j<nj;j++)for(unsigned i=0u;i<ni;i++){float x=a[(unsigned long long)j*width+i];if(shifted)x=TA(x,shift);s=s+(double)x;}out[slot]=s;}
extern "C" __global__ void cm1_t2p_plane_product_sum_v1(const float*a,const float*b,unsigned ni,unsigned nj,double*out,unsigned slot){if(blockIdx.x!=0u||threadIdx.x!=0u)return;double s=0.0;for(unsigned j=0u;j<nj;j++)for(unsigned i=0u;i<ni;i++){unsigned long long q=(unsigned long long)j*ni+i;s=s+(double)TM(a[q],b[q]);}out[slot]=s;}
// getgamk/t2psmm: uavg(k)=cavg(k,1)*temd; savg; s13b/s23b.
extern "C" __global__ void cm1_t2p_mean_profiles_v1(const double*usum,const double*vsum,const float*mf,float*uavg,float*vavg,float*savg,float*s13b,float*s23b,unsigned nx,unsigned ny,unsigned ntwk,float rdz){if(blockIdx.x!=0u||threadIdx.x!=0u)return;double temd=1.0/(double)(nx*ny);for(unsigned m=0u;m<ntwk;m++){uavg[m]=(float)(usum[m]*temd);vavg[m]=(float)(vsum[m]*temd);float r=sqrtf(TA(TM(uavg[m],uavg[m]),TM(vavg[m],vavg[m])));savg[m]=(r>1.0e-10f)?r:1.0e-10f;}for(unsigned p=0u;p<=ntwk;p++){s13b[p]=0.0f;s23b[p]=0.0f;}for(unsigned p=1u;p<ntwk;p++){s13b[p]=TM(TM(TM(0.5f,TS(uavg[p],uavg[p-1u])),rdz),mf[p]);s23b[p]=TM(TM(TM(0.5f,TS(vavg[p],vavg[p-1u])),rdz),mf[p]);}}
// t2pcode, neutral branch (sfcmodel=1 mol=0), t2p_avg=1.
extern "C" __global__ void cm1_t2p_b20_profile_v1(const float*uavg,const float*vavg,const float*savg,const double*t13sum,const double*t23sum,const double*rfsum,double ustsum,double zntsum,const float*zh,const float*mh,const float*mf,const float*gamwall,float*kmw,float*ufw,float*vfw,float*wspa,unsigned nx,unsigned ny,unsigned nk,unsigned ntwk,float rdz,float t2pfac){if(blockIdx.x!=0u||threadIdx.x!=0u)return;for(unsigned p=0u;p<=nk;p++){kmw[p]=0.0f;ufw[p]=0.0f;vfw[p]=0.0f;}for(unsigned m=0u;m<=ntwk;m++)wspa[m]=0.0f;if(!(t2pfac>=0.001f))return;double teme=1.0/(double)(nx*ny);double ustbar=ustsum*teme,zntbar=zntsum*teme;for(unsigned m=0u;m<=ntwk;m++){float level=TS(logf(TD(zh[m],(float)zntbar)),0.0f);wspa[m]=(float)((ustbar/(double)0.40f)*(double)level);}for(unsigned m=0u;m+1u<ntwk;m++){double t13avg=t13sum[m]*teme,t23avg=t23sum[m]*teme,rfavg=rfsum[m]*teme;float d2pm1=(float)(((double)uavg[m]*t13avg+(double)vavg[m]*t23avg)/(double)savg[m]);float head=TA(TM(uavg[m],ufw[m]),TM(vavg[m],vfw[m]));float d2pm3=(float)(((((double)head*rfavg)*(double)rdz)*(double)mh[m])/(double)savg[m]);float dsdz=TM(TM(TS(wspa[m+1u],wspa[m]),rdz),mf[m+1u]);float bounded=(1.0e-12f>dsdz)?1.0e-12f:dsdz;float tem=(float)((double)TM(t2pfac,gamwall[m+1u])/((double)TM(TM(rdz,mh[m]),bounded)*rfavg));float x=TM(TA(TS(0.0f,d2pm1),d2pm3),tem);kmw[m+1u]=(x>0.0f)?x:0.0f;ufw[m+1u]=TM(TM(TM(kmw[m+1u],TS(uavg[m+1u],uavg[m])),rdz),mf[m+1u]);vfw[m+1u]=TM(TM(TM(kmw[m+1u],TS(vavg[m+1u],vavg[m])),rdz),mf[m+1u]);}}
// t2psmm per-point strain term; planes p=1..ntwk-1 are CM1's k=2..ntwk.
extern "C" __global__ void cm1_t2p_s94_point_term_v1(const float*s11,const float*s22,const float*s33,const float*s12,const float*s13,const float*s23,const float*c1,const float*c2,const float*rf,const float*s13b,const float*s23b,float*term,unsigned ni,unsigned nj,unsigned ntwk){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)(ntwk+1u)*nj*ni,st=(unsigned long long)blockDim.x*gridDim.x,z=(unsigned long long)nj*ni;for(;q<n;q+=st){unsigned p=(unsigned)(q/z),j=(unsigned)((q/ni)%nj),i=(unsigned)(q%ni);if(p==0u||p>=ntwk){term[q]=0.0f;continue;}float a1=c1[T3(p,j,i,nj,ni)],a2=c2[T3(p,j,i,nj,ni)];float x;
float l11=s11[T3(p-1u,j,i,nj,ni)],h11=s11[T3(p,j,i,nj,ni)],l22=s22[T3(p-1u,j,i,nj,ni)],h22=s22[T3(p,j,i,nj,ni)],l33=s33[T3(p-1u,j,i,nj,ni)],h33=s33[T3(p,j,i,nj,ni)];
float tmp11=TA(TM(a1,TM(l11,l11)),TM(a2,TM(h11,h11)));float tmp22=TA(TM(a1,TM(l22,l22)),TM(a2,TM(h22,h22)));float tmp33=TA(TM(a1,TM(l33,l33)),TM(a2,TM(h33,h33)));
float e0=s12[T3(p-1u,j,i,nj+1u,ni+1u)],e1=s12[T3(p-1u,j+1u,i+1u,nj+1u,ni+1u)],e2=s12[T3(p-1u,j+1u,i,nj+1u,ni+1u)],e3=s12[T3(p-1u,j,i+1u,nj+1u,ni+1u)];
float g0=s12[T3(p,j,i,nj+1u,ni+1u)],g1=s12[T3(p,j+1u,i+1u,nj+1u,ni+1u)],g2=s12[T3(p,j+1u,i,nj+1u,ni+1u)],g3=s12[T3(p,j,i+1u,nj+1u,ni+1u)];
float lo=TA(TA(TM(e0,e0),TM(e1,e1)),TA(TM(e2,e2),TM(e3,e3)));float hi=TA(TA(TM(g0,g0),TM(g1,g1)),TA(TM(g2,g2),TM(g3,g3)));float tmp12=TM(0.25f,TA(TM(a1,lo),TM(a2,hi)));
x=TS(s13[T3(p,j,i,nj,ni+1u)],s13b[p]);float y=TS(s13[T3(p,j,i+1u,nj,ni+1u)],s13b[p]);float tmp13=TM(0.5f,TA(TM(x,x),TM(y,y)));
x=TS(s23[T3(p,j,i,nj+1u,ni)],s23b[p]);y=TS(s23[T3(p,j+1u,i,nj+1u,ni)],s23b[p]);float tmp23=TM(0.5f,TA(TM(x,x),TM(y,y)));
float r=rf[T3(p,j,i,nj,ni)];term[q]=TD(TA(TM(4.0f,TA(tmp12,TA(tmp13,tmp23))),TM(2.0f,TA(TA(tmp11,tmp22),tmp33))),TM(r,r));}}
extern "C" __global__ void cm1_t2p_s94_gamk_v1(const double*spsum,const float*s13b,const float*s23b,const float*gamwall,float*gamk,unsigned nx,unsigned ny,unsigned ntwk,float t2pfac,float smeps){if(blockIdx.x!=0u||threadIdx.x!=0u)return;if(!(t2pfac>=0.001f))return;double teme=1.0/(double)(nx*ny);for(unsigned p=1u;p<ntwk;p++){double spavg=sqrt(spsum[p]*teme);double shravg=(double)sqrtf(TM(4.0f,TA(TM(s13b[p],s13b[p]),TM(s23b[p],s23b[p]))));gamk[p]=(float)(spavg/((double)smeps+(spavg+(double)gamwall[p]*shravg)));}gamk[0]=gamk[1];}
extern "C" __global__ void cm1_t2p_scale_km_v1(const float*gamk,const float*km,float*out,unsigned ni,unsigned nj,unsigned nk1){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nk1*nj*ni,st=(unsigned long long)blockDim.x*gridDim.x,z=(unsigned long long)nj*ni;for(;q<n;q+=st){out[q]=TM(gamk[(unsigned)(q/z)],km[q]);}}
extern "C" __global__ void cm1_t2p_solve3_v1(const float*current,const float*baseline,float*t2pt,unsigned long long n,float dt){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)blockDim.x*gridDim.x;float rdt=TD(1.0f,dt);for(;q<n;q+=st){t2pt[q]=TS(TM(TS(current[q],baseline[q]),rdt),t2pt[q]);}}
'''
CUDA_TWO_PART_SOURCE_SHA256 = hashlib.sha256(CUDA_TWO_PART_SOURCE.encode()).hexdigest()


# ------------------------------------------- the turbuz/turbvz column kernels

TWO_PART_COLUMN_KERNEL_NAMES = (
    "cm1_sgs_turbu_terrain_t2p_v1",
    "cm1_sgs_turbv_terrain_t2p_v1",
)

_COLUMN_HELPERS = r'''
// src/turbtend.F:1779 and :2436: dum2(k)=0.5*(rf(i-1,j,k)*ufwk(i-1,j,k)+rf(i,j,k)*ufwk(i,j,k)),
// dum2(1)=dum2(ntwk+1)=0.  ufwk(i,j,k)=ufw(k) for every column (src/turb.F:1497-1504).
__device__ __forceinline__ float T2PU(const float*rf,const float*tfw,unsigned w,int j,int a,int b,unsigned ntwk,int nj,int ni){if(w==0u||w>=ntwk)return 0.0f;return M(0.5f,A(M(CELL(rf,(int)w,j,a,nj,ni),tfw[w]),M(CELL(rf,(int)w,j,b,nj,ni),tfw[w])));}
__device__ __forceinline__ float T2PV(const float*rf,const float*tfw,unsigned w,int a,int b,int i,unsigned ntwk,int nj,int ni){if(w==0u||w>=ntwk)return 0.0f;return M(0.5f,A(M(CELL(rf,(int)w,a,i,nj,ni),tfw[w]),M(CELL(rf,(int)w,b,i,nj,ni),tfw[w])));}
'''

_SIGNATURE_TAIL = "const float*rdsf,float zt){"
_SIGNATURE_TWO_PART = "const float*rdsf,float zt,const float*tfw,float*t2pt,unsigned ntwk){"
_U_TAIL = "vz=A(vz,M(S(k+1<nk?cr[k+1]:0.0f,cr[k]),M(rdz,M(0.5f,hm))));unsigned long long p=I3(k,j,f,nj,ni+1);"
_U_TWO_PART = (
    "vz=A(vz,M(S(k+1<nk?cr[k+1]:0.0f,cr[k]),M(rdz,M(0.5f,hm))));"
    "if(k<ntwk){float tm=M(rdz,CELL(mh,(int)k,0,0,nj,ni));"
    "float dd=S(T2PU(rf,tfw,k+1u,j,l,r,ntwk,nj,ni),T2PU(rf,tfw,k,j,l,r,ntwk,nj,ni));"
    "vz=A(vz,M(tm,dd));t2pt[I3(k,j,f,nj,ni+1)]=M(M(tm,dd),D(1.0f,M(0.5f,rh)));}"
    "unsigned long long p=I3(k,j,f,nj,ni+1);"
)
_V_TAIL = "vz=A(vz,M(S(k+1<nk?cr[k+1]:0.0f,cr[k]),M(rz,M(0.5f,hm))));unsigned long long p=I3(k,f,i,nj+1,ni);"
_V_TWO_PART = (
    "vz=A(vz,M(S(k+1<nk?cr[k+1]:0.0f,cr[k]),M(rz,M(0.5f,hm))));"
    "if(k<ntwk){float tm=M(rz,CELL(mh,(int)k,0,0,nj,ni));"
    "float dd=S(T2PV(rf,tfw,k+1u,s,r,i,ntwk,nj,ni),T2PV(rf,tfw,k,s,r,i,ntwk,nj,ni));"
    "vz=A(vz,M(tm,dd));t2pt[I3(k,f,i,nj+1,ni)]=M(M(tm,dd),D(1.0f,M(0.5f,rh)));}"
    "unsigned long long p=I3(k,f,i,nj+1,ni);"
)


def _kernel_text(source: str, name: str, stop: str) -> str:
    start = source.index(f'extern "C" __global__ void {name}(')
    end = source.index(stop, start)
    return source[start:end]


def _exact(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"two-part {label} transform found {text.count(old)} sites, expected 1")
    return text.replace(old, new)


def two_part_column_additions(base_source: str) -> str:
    """The two new column kernels, derived from the audited stretched text.

    The base ``turbu``/``turbv`` column kernels are copied, renamed, given
    three trailing arguments and one inserted statement block at the point
    ``turbuz``/``turbvz`` reach it: after the implicit solve and the ``dwdx``
    cross term have formed ``turbz`` and before ``turbu`` folds ``turbz`` into
    ``uten`` (src/turbtend.F:1767-1802 then :1290-1300).  The base kernels are
    left in the module untouched.
    """

    u = _kernel_text(base_source, "cm1_sgs_turbu_terrain_v1", "// v kernel uses the exact transpose")
    v = _kernel_text(base_source, "cm1_sgs_turbv_terrain_v1", 'extern "C" __global__ void cm1_sgs_turbw_terrain_v1(')
    u = _exact(u, "cm1_sgs_turbu_terrain_v1(", "cm1_sgs_turbu_terrain_t2p_v1(", "u name")
    v = _exact(v, "cm1_sgs_turbv_terrain_v1(", "cm1_sgs_turbv_terrain_t2p_v1(", "v name")
    u = _exact(u, _SIGNATURE_TAIL, _SIGNATURE_TWO_PART, "u signature")
    v = _exact(v, _SIGNATURE_TAIL, _SIGNATURE_TWO_PART, "v signature")
    u = _exact(u, _U_TAIL, _U_TWO_PART, "turbuz dot2p")
    v = _exact(v, _V_TAIL, _V_TWO_PART, "turbvz dot2p")
    return "\n// CM1_TWO_PART_COLUMN_KERNELS_V1\n" + _COLUMN_HELPERS + u + "\n" + v + "\n"


def build_two_part_column_source(*, max_nk: int) -> str:
    from .cuda_gm1_child_stretched_sgs_tke import build_cuda_gm1_child_stretched_sources

    base, _tke = build_cuda_gm1_child_stretched_sources(max_nk=max_nk)
    return base + two_part_column_additions(base)


# ---------------------------------------------------------------- compile

_NVRTC_OPTIONS = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")


@dataclass(frozen=True, slots=True)
class CompiledTwoPart:
    module: Any
    kernels: Mapping[str, Any]
    column_module: Any
    column_kernels: tuple[Any, Any]
    source_sha256: str
    column_source_sha256: str


def compile_two_part(*, max_nk: int) -> CompiledTwoPart:
    cupy = importlib.import_module("cupy")
    module = compile_nvrtc_module(cupy, code=CUDA_TWO_PART_SOURCE, options=_NVRTC_OPTIONS,
                                  name_expressions=CUDA_TWO_PART_KERNEL_NAMES, filename="gm1_two_part_sgs.cu")
    column_source = build_two_part_column_source(max_nk=max_nk)
    column_module = compile_nvrtc_module(cupy, code=column_source, options=_NVRTC_OPTIONS,
                                         name_expressions=TWO_PART_COLUMN_KERNEL_NAMES,
                                         filename="gm1_two_part_sgs_column.cu")
    return CompiledTwoPart(
        module,
        MappingProxyType({name: module.get_function(name) for name in CUDA_TWO_PART_KERNEL_NAMES}),
        column_module,
        tuple(column_module.get_function(name) for name in TWO_PART_COLUMN_KERNEL_NAMES),
        CUDA_TWO_PART_SOURCE_SHA256,
        hashlib.sha256(column_source.encode()).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class TwoPartColumnBinding:
    """What the stress launch needs to run the ``dot2p`` column kernels."""

    kernels: tuple[Any, Any]
    tfu: Any
    tfv: Any
    u2pt: Any
    v2pt: Any
    ntwk: int
    source_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "kernel_names": list(TWO_PART_COLUMN_KERNEL_NAMES),
            "source_sha256": self.source_sha256,
            "ntwk": self.ntwk,
            "authority": TWO_PART_AUTHORITY["turbuz_turbvz_flux"],
        }


# ---------------------------------------------------------------- orchestration

_THREADS = 128


class CartesianTwoPart:
    """The two-part state and its per-step operators for one Cartesian domain.

    One instance belongs to one :class:`~cm1core.cuda_cartesian_closure.CartesianClosure`
    -- the parent's or a live nest child's -- and averages over that closure's
    own integration domain, which is CM1's ``nx*ny`` for the run it is.
    """

    def __init__(self, closure: Any) -> None:
        import numpy as np

        self.closure = closure
        cp, g = closure.cp, closure.grid
        self.cp, self.grid, self.stream, self.registry = cp, g, closure.stream, closure.registry
        config = closure.config
        self.selection = validate_two_part_selection(
            sgsmodel=int(config.sgsmodel), t2p_avg=int(config.t2p_avg),
            t2p_avg_declared=False, terrain=False, bbc=int(config.bbc),
            sfcmodel=int(config.sfcmodel),
        )
        d = closure.dynamics
        nk = g.nz
        with self.stream:
            zf = cp.ascontiguousarray(d.metrics["zf"][:, 0, 0]).get(stream=self.stream)
        gamwall, ntwk = gamwall_profile(zf)
        if ntwk + 1 > nk:
            raise RuntimeError(
                f"the two-part wall layer reaches w level {ntwk} of a {nk}-level "
                "column; t2pcode reads zh(1,1,ntwk+1) (src/turb.F:6186) above "
                "the model top, so the log-law shear of the top two-part level "
                "would be taken from a word the column does not have"
            )
        self.ntwk = int(ntwk)
        self.ztwk_m = float(zf[self.ntwk - 1])
        self.gamwall_sha256 = hashlib.sha256(np.asarray(gamwall, "<f4").tobytes()).hexdigest()
        self.compiled = compile_two_part(max_nk=closure.sgs_compiled.max_nk)
        self._names: list[str] = []
        self._values: dict[str, Any] = {}
        # CM1's rdz = 1.0/dz in default REAL.
        self.rdz = np.float32(np.float32(1.0) / np.float32(g.dz_m))
        ntwk, ny, nx = self.ntwk, g.ny, g.nx
        spec0 = closure.plan.specs[0]
        cnx, cny = spec0.cnx, spec0.cny
        f64 = cp.float64
        a = self._allocate
        self.gamwall = a("gamwall", (nk + 1,))
        self.gamk = a("gamk", (nk + 1,))
        self.zero_profile = a("zero_profile", (nk + 1,))
        for name in ("kmw", "ufw", "vfw"):
            setattr(self, name, a(name, (nk + 1,)))
        self.wspa = a("wspa", (ntwk + 1,))
        for name in ("uavg", "vavg", "savg"):
            setattr(self, name, a(name, (ntwk,)))
        self.s13b = a("s13b", (ntwk + 1,))
        self.s23b = a("s23b", (ntwk + 1,))
        self.zh = a("zh", (nk,))
        self.mh = a("mh", (nk,))
        self.mf = a("mf", (nk + 1,))
        self.usum = a("usum", (ntwk,), dtype=f64)
        self.vsum = a("vsum", (ntwk,), dtype=f64)
        self.ua = a("ua_levels", (ntwk, ny, nx + 1))
        self.va = a("va_levels", (ntwk, ny + 1, nx))
        with self.stream:
            self.gamwall.set(np.asarray(gamwall, np.float32), stream=self.stream)
            self.gamk.fill(cp.float32(1.0))
            for name in ("zh", "mh", "mf"):
                cp.copyto(getattr(self, name), d.metrics[name][:, 0, 0])
        if self.selection.sgsmodel == 4:
            self.u2pt = a("u2pt", (ntwk, ny, nx + 1))
            self.v2pt = a("v2pt", (ntwk, ny + 1, nx))
            self.u3d = a("u3d_levels", (ntwk, ny, nx + 1))
            self.v3d = a("v3d_levels", (ntwk, ny + 1, nx))
            self.ust = a("ust", (ny, nx))
            self.znt = a("znt", (ny, nx))
            self.rf = a("rf_levels", (ntwk, ny, nx))
            self.rr = a("rr_levels", (ntwk, ny, nx))
            self.t13sum = a("t13sum", (ntwk,), dtype=f64)
            self.t23sum = a("t23sum", (ntwk,), dtype=f64)
            self.rfsum = a("rfsum", (ntwk,), dtype=f64)
            self.surface_sums = a("surface_sums", (2,), dtype=f64)
            self.tile_u2pt = a("tile_u2pt", (ntwk, cny, cnx + 1))
            self.tile_v2pt = a("tile_v2pt", (ntwk, cny + 1, cnx))
        else:
            self.term = a("term", (ntwk + 1, ny, nx))
            self.tile_term = a("tile_term", (ntwk + 1, cny, cnx))
            self.spsum = a("spsum", (ntwk + 1,), dtype=f64)
            self.kmh_scaled = a("kmh_scaled", (nk + 1, cny, cnx))
            self.kmv_scaled = a("kmv_scaled", (nk + 1, cny, cnx))
            # r21.1 never copies S94's ufw into ufwk, so its column flux is
            # built from zeros; the saved tendency has no reader.
            self.tile_u2pt = a("tile_u2pt", (ntwk, cny, cnx + 1))
            self.tile_v2pt = a("tile_v2pt", (ntwk, cny + 1, cnx))
        self.state_payload_bytes = sum(
            int(self.registry_value(name).nbytes) for name in self._names
        )
        self.resumed = False
        self.t2pfac = t2pfac_f32(0.0)
        self.profile_updated = False
        self._step_resumed = False

    # -- storage ----------------------------------------------------------
    def _allocate(self, name: str, shape: tuple[int, ...], *, dtype: Any = None) -> Any:
        full = f"closure.two_part.{name}"
        value = self.registry.allocate(full, shape, dtype=dtype, zero=True)
        self._names.append(full)
        self._values[full] = value
        return value

    def registry_value(self, full: str) -> Any:
        return self._values[full]

    # -- records ----------------------------------------------------------
    def record(self) -> dict[str, object]:
        return {
            **self.selection.record(),
            "ntwk": self.ntwk,
            "ztwk_m": self.ztwk_m,
            "gamwall_sha256": self.gamwall_sha256,
            "kernel_source_sha256": self.compiled.source_sha256,
            "column_kernel_source_sha256": self.compiled.column_source_sha256,
            "domain_average_extent": (
                "this closure's integration domain (a live nest child averages "
                "over its own window, as a CM1 run of that domain would)"
            ),
            "domain_sum_order": "CM1 serial binary64, j outer then i, one device thread per level",
            "state_payload_bytes": self.state_payload_bytes,
            "restart": (
                "B20 carries no two-part word across a restart: CM1 skips "
                "t2pcode on the restart step, ufwk and u2pt restart at zero and "
                "t2pcode zeroes kmw/ufw/vfw before it reads them"
                if self.selection.sgsmodel == 4
                else "S94 restart writing is refused: gamk would have to be carried "
                "(src/restart_write.F:1598) and this lane's restart carries no profile"
            ),
            "claim_limit": (
                "logf/sqrtf/tanhf are the platform's on the host and CUDA's on "
                "the device; every other statement is CM1's binary32/binary64 "
                "order"
            ),
        }

    def step_record(self) -> dict[str, object]:
        return {
            "t2pfac": float(self.t2pfac),
            "profile_updated": self.profile_updated,
            "restart_step_skipped": self._step_resumed,
        }

    def mark_resumed(self) -> None:
        """``src/cm1.F:2663``: the first step after a restart skips t2p."""

        self.resumed = True

    # -- per-step ---------------------------------------------------------
    def begin(self, model_time_s: float | None) -> bool:
        """Return whether this step computes new two-part profiles."""

        if model_time_s is None:
            raise RuntimeError(
                "the two-part ramp t2pfac is a function of model time "
                "(src/turb.F:266); a held-tendency phase without it would "
                "switch the near-wall model on or off at an unknown time"
            )
        self.t2pfac = t2pfac_f32(model_time_s)
        self._step_resumed = self.resumed
        self.resumed = False
        # src/turb.F:1460-1466.  B20's t2pcode zeroes its profiles before it
        # tests t2pfac, so an early step integrates a zero flux; S94's gamk is
        # left as it stands.
        self.profile_updated = (not self._step_resumed) and bool(self.t2pfac >= _f32(0.001))
        if not self._step_resumed and not self.profile_updated and self.selection.sgsmodel == 4:
            with self.stream:
                for name in ("kmw", "ufw", "vfw"):
                    getattr(self, name).fill(self.cp.float32(0.0))
        return self.profile_updated

    def _sum(self, plane: Any, ni: int, nj: int, width: int, out: Any, slot: int, shift: float = 0.0, shifted: bool = False) -> None:
        cp = self.cp
        self.compiled.kernels["cm1_t2p_plane_sum_v1"](
            (1,), (1,),
            (plane, cp.uint32(ni), cp.uint32(nj), cp.uint32(width), cp.float32(shift), cp.uint32(1 if shifted else 0), out, cp.uint32(slot)),
            stream=self.stream,
        )

    def means(self) -> None:
        """``getgamk``/``t2psmm``'s level means of the ground-relative wind."""

        cp, g, d = self.cp, self.grid, self.closure.dynamics
        ntwk, nx, ny = self.ntwk, g.nx, g.ny
        config = self.closure.config
        cp.copyto(self.ua, d.fields["ua"][:ntwk])
        cp.copyto(self.va, d.fields["va"][:ntwk])
        for m in range(ntwk):
            # src/turb.F:296-307: ugr=ua+umove, as the surface layer forms it.
            self._sum(self.ua[m], nx, ny, nx + 1, self.usum, m, float(config.u_move_m_s), True)
            self._sum(self.va[m], nx, ny, nx, self.vsum, m, float(config.v_move_m_s), True)
        self.compiled.kernels["cm1_t2p_mean_profiles_v1"](
            (1,), (1,),
            (self.usum, self.vsum, self.mf, self.uavg, self.vavg, self.savg, self.s13b, self.s23b,
             cp.uint32(nx), cp.uint32(ny), cp.uint32(ntwk), cp.float32(self.rdz)),
            stream=self.stream,
        )

    def prepass_tile(self, tile: Mapping[str, Any], work: Mapping[str, Any], surface: Mapping[str, Any], spec: Any) -> None:
        from .cuda_cartesian_closure import _scatter

        cp = self.cp
        if self.selection.sgsmodel == 4:
            _scatter(cp, self.ust, surface["ust"], spec)
            _scatter(cp, self.znt, surface["znt"], spec)
            return
        ni, nj = spec.cnx, spec.cny
        words = (self.ntwk + 1) * nj * ni
        self.compiled.kernels["cm1_t2p_s94_point_term_v1"](
            ((words + _THREADS - 1) // _THREADS,), (_THREADS,),
            (*(work[name] for name in ("s11", "s22", "s33", "s12", "s13", "s23", "c1", "c2")), tile["rf"],
             self.s13b, self.s23b, self.tile_term, cp.uint32(ni), cp.uint32(nj), cp.uint32(self.ntwk)),
            stream=self.stream,
        )
        _scatter(cp, self.term, self.tile_term, spec)

    def reduce(self) -> None:
        cp, g, d = self.cp, self.grid, self.closure.dynamics
        ntwk, nx, ny = self.ntwk, g.nx, g.ny
        rdz = cp.float32(self.rdz)
        if self.selection.sgsmodel == 3:
            for p in range(1, ntwk):
                self._sum(self.term[p], nx, ny, nx, self.spsum, p)
            self.compiled.kernels["cm1_t2p_s94_gamk_v1"](
                (1,), (1,),
                (self.spsum, self.s13b, self.s23b, self.gamwall, self.gamk, cp.uint32(nx), cp.uint32(ny),
                 cp.uint32(ntwk), cp.float32(self.t2pfac), cp.float32(SMEPS)),
                stream=self.stream,
            )
            return
        self._sum(self.ust, nx, ny, nx, self.surface_sums, 0)
        self._sum(self.znt, nx, ny, nx, self.surface_sums, 1)
        cp.copyto(self.rf, d.fields["rf"][1:ntwk + 1])
        cp.copyto(self.rr, d.fields["rr"][:ntwk])
        for m in range(ntwk - 1):
            self._sum(self.u2pt[m], nx, ny, nx + 1, self.t13sum, m)
            self._sum(self.v2pt[m], nx, ny, nx, self.t23sum, m)
            self.compiled.kernels["cm1_t2p_plane_product_sum_v1"](
                (1,), (1,), (self.rf[m], self.rr[m], cp.uint32(nx), cp.uint32(ny), self.rfsum, cp.uint32(m)),
                stream=self.stream,
            )
        sums = self.surface_sums
        # Two device doubles become kernel scalars: one host read per step.
        ustsum, zntsum = (float(x) for x in sums.get(stream=self.stream))
        self.compiled.kernels["cm1_t2p_b20_profile_v1"](
            (1,), (1,),
            (self.uavg, self.vavg, self.savg, self.t13sum, self.t23sum, self.rfsum,
             cp.float64(ustsum), cp.float64(zntsum), self.zh, self.mh, self.mf, self.gamwall,
             self.kmw, self.ufw, self.vfw, self.wspa, cp.uint32(nx), cp.uint32(ny), cp.uint32(g.nz),
             cp.uint32(ntwk), rdz, cp.float32(self.t2pfac)),
            stream=self.stream,
        )

    def scaled_coefficients(self, work: Mapping[str, Any], spec: Any) -> tuple[Any, Any] | None:
        """S94's ``gamk*kmh``/``gamk*kmv`` for this tile; ``None`` for B20."""

        if self.selection.sgsmodel != 3:
            return None
        cp = self.cp
        ni, nj, nk1 = spec.cnx, spec.cny, self.grid.nz + 1
        words = nk1 * nj * ni
        for name, target in (("kmh", self.kmh_scaled), ("kmv", self.kmv_scaled)):
            self.compiled.kernels["cm1_t2p_scale_km_v1"](
                ((words + _THREADS - 1) // _THREADS,), (_THREADS,),
                (self.gamk, work[name], target, cp.uint32(ni), cp.uint32(nj), cp.uint32(nk1)),
                stream=self.stream,
            )
        return self.kmh_scaled, self.kmv_scaled

    def column_binding(self) -> TwoPartColumnBinding:
        if self.selection.sgsmodel == 4 and not self._step_resumed:
            tfu, tfv = self.ufw, self.vfw
        else:
            tfu = tfv = self.zero_profile
        return TwoPartColumnBinding(
            self.compiled.column_kernels, tfu, tfv, self.tile_u2pt, self.tile_v2pt,
            self.ntwk, self.compiled.column_source_sha256,
        )

    def after_tile(self, spec: Any) -> None:
        from .cuda_cartesian_closure import _scatter

        if self.selection.sgsmodel != 4:
            return
        _scatter(self.cp, self.u2pt, self.tile_u2pt, spec, "u")
        _scatter(self.cp, self.v2pt, self.tile_v2pt, spec, "v")

    def solve3(self, *, dt_s: float) -> None:
        """``src/solve3.F:340-352``, before ``ua``/``va`` take the new winds."""

        if self.selection.sgsmodel != 4:
            return
        cp, d = self.cp, self.closure.dynamics
        ntwk = self.ntwk
        with self.stream:
            for current, baseline, name, carrier in (
                ("u3d", "ua", "u3d", self.u2pt), ("v3d", "va", "v3d", self.v2pt),
            ):
                scratch = self.u3d if name == "u3d" else self.v3d
                base = self.ua if name == "u3d" else self.va
                cp.copyto(scratch, d.fields[current][:ntwk])
                cp.copyto(base, d.fields[baseline][:ntwk])
                words = int(carrier.size)
                self.compiled.kernels["cm1_t2p_solve3_v1"](
                    ((words + _THREADS - 1) // _THREADS,), (_THREADS,),
                    (scratch, base, carrier, cp.uint64(words), cp.float32(dt_s)),
                    stream=self.stream,
                )


def two_part_receipt_hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
