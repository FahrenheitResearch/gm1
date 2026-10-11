"""CM1 r21.1 ``pdcomp.F`` pressure decomposition, inverted on the device.

Authority
---------
``src/pdcomp.F:10-420`` ``pidcomp``, called from ``src/cm1.F:3483-3503`` at
every history time when ``output_pdcomp=1`` (``src/param.F:727-729``).  Each
of its four forcings is handed to ``src/poiss.F:10-220``; the results are the
``pdiag`` slots ``writeout.F:6252-6340`` publishes as ``pipb`` / ``pipdl`` /
``pipdn`` / ``pipc`` and ``writeout.F:8478-8580`` differentiates into
``pgradb`` / ``pgraddl`` / ``pgraddn`` / ``pgradc``.

How CM1 inverts on open laterals
--------------------------------
There is no separate open-boundary solver.  ``src/param.F:5178-5205`` sets
``imirror=1, ipe=2*ni`` when ``wbc`` or ``ebc`` is 2 or 3 (``jmirror=1,
jpe=2*nj`` for ``sbc``/``nbc``), and ``poiss.F:80-114`` reflects the
right-hand side evenly into the doubled rectangle before the same Singleton
FFT (``singleton.F``) and vertical Thomas solve.  An even reflection over a
``2n`` periodic transform is a discrete cosine transform: the lateral
condition is zero normal pressure gradient at the domain faces, not
periodicity.  ``init3d.F:1668-1692`` builds the ``psolver<=3`` coefficients
on that doubled mode set, with ``mh``/``mf`` carrying ``stretch_z``, and
``pdcomp.F:165-191`` sets the gauge from the mean of ``pp3d`` on the top
level (buoyancy) or zero (the other three).

This module ports exactly that and nothing else: the forcings (their float
statement order included), the mirrored right-hand side, the doubled-mode
coefficients, the zero-mode recurrence (shared with
:mod:`cm1core.cuda_anelp_poisson_periodic`, whose kernel is the same
``poiss.F:118-185``), the gauge and the vertical gradient products.  The
nonlinear forcing calls this lane's own ``advu``/``advv``/``advw`` kernels on
pdcomp-owned buffers, as ``pdcomp.F:317-326`` calls CM1's.

It reads model carriers and writes none: every model array reaches a kernel
through a ``const`` parameter (asserted in the tests), so ``output_pdcomp``
cannot change a simulated word.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping

CM1_OPEN_NGXY = 3

PDCOMP_OPEN_SCHEMA = "gpuwm-cm1.pdcomp-open/v1"

SOURCE_ANCHORS: Mapping[str, str] = {
    "pidcomp": "src/pdcomp.F:10-420",
    "call_site": "src/cm1.F:3483-3503",
    "selector": "src/param.F:727-729",
    "buoyancy_forcing": "src/pdcomp.F:68-163",
    "buoyancy_gauge": "src/pdcomp.F:165-191",
    "linear_dynamic_forcing": "src/pdcomp.F:193-240",
    "nonlinear_dynamic_forcing": "src/pdcomp.F:266-340",
    "coriolis_forcing": "src/pdcomp.F:366-390",
    "zero_top_gauge": "src/pdcomp.F:242-264,342-364,392-416",
    "mirror_transform_shape": "src/param.F:5178-5205",
    "coefficients_psolver_le_3": "src/init3d.F:1668-1692",
    "poiss_rhs_and_mirror": "src/poiss.F:62-116",
    "poiss_thomas_and_zero_mode": "src/poiss.F:118-185",
    "poiss_inverse": "src/poiss.F:187-213",
    "thv0": "src/base.F:2127-2135",
    "rth0": "src/base.F:3209",
    "constants": "src/constants.F:100-120",
    "smeps": "src/cm1.F:311",
    "names_scalar": "src/writeout.F:6252-6340",
    "names_w": "src/writeout.F:8478-8580",
    "refusals": "src/param.F:1765-1829,2434-2446",
}

#: ``writeout.F:6252-6340`` names, in ``pdiag`` slot order.
PDCOMP_SCALAR_NAMES: tuple[str, ...] = ("pipb", "pipdl", "pipdn", "pipc")
#: ``writeout.F:8478-8580`` names, in the same order.
PDCOMP_W_NAMES: tuple[str, ...] = ("pgradb", "pgraddl", "pgraddn", "pgradc")

def _np():
    return importlib.import_module("numpy")


# constants.F:100-120, default (testcase not 4/5) branch, evaluated in REAL.
def cm1_constants() -> SimpleNamespace:
    np = _np()
    g = np.float32(9.81)
    rd = np.float32(287.04)
    rv = np.float32(461.5)
    cp = np.float32(1005.7)
    reps = np.float32(rv / rd)
    repsm1 = np.float32(np.float32(rv / rd) - np.float32(1.0))
    return SimpleNamespace(g=g, rd=rd, rv=rv, cp=cp, reps=reps, repsm1=repsm1,
                           smeps=np.float32(1.0e-30))


# --------------------------------------------------------------------------
# Selection and refusal
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PdcompSelection:
    ni: int
    nj: int
    nk: int
    dx: float
    dy: float
    dz: float
    dt: float
    wbc: int = 2
    ebc: int = 2
    sbc: int = 2
    nbc: int = 2
    icor: int = 0
    imoist: int = 1
    iice: int = 1
    nqv: int = 1
    nql1: int = 2
    nql2: int = 3
    nqs1: int = 4
    nqs2: int = 6
    psolver: int = 3
    terrain: bool = False

    @property
    def imirror(self) -> int:
        # param.F:5188-5193
        return int(self.wbc in (2, 3) or self.ebc in (2, 3))

    @property
    def jmirror(self) -> int:
        return int(self.sbc in (2, 3) or self.nbc in (2, 3))

    @property
    def ipe(self) -> int:
        return self.ni * 2 if self.imirror else self.ni

    @property
    def jpe(self) -> int:
        return self.nj * 2 if self.jmirror else self.nj

    @property
    def names(self) -> tuple[str, ...]:
        return PDCOMP_SCALAR_NAMES if self.icor == 1 else PDCOMP_SCALAR_NAMES[:3]

    @property
    def w_names(self) -> tuple[str, ...]:
        return PDCOMP_W_NAMES if self.icor == 1 else PDCOMP_W_NAMES[:3]


def pdcomp_selection_defect(selection: PdcompSelection) -> str | None:
    """Why this run cannot publish CM1's decomposition, or ``None``."""

    s = selection
    if s.terrain:
        return (
            "grid.terrain: param.F:1765-1779 stops CM1 for pdcomp with "
            "terrain_flag, because poiss.F:70-75 has no terrain metric and "
            "init3d.F:1687-1689's modes are those of a flat column"
        )
    if s.psolver > 3:
        return (
            f"dynamics.psolver={s.psolver}: init3d.F:1693-1720 builds a "
            "different cfa/cfc/d1/d2 set for psolver 4/5 that this port does "
            "not build; the psolver<=3 set would invert the wrong operator"
        )
    if float(s.dx) != float(s.dy):
        return (
            f"grid.dx_m={s.dx} != grid.dy_m={s.dy}: init3d.F:1687-1689 divides "
            "the y-mode cosine by dx*dx, so CM1's inversion is only the inverse "
            "of poiss.F:70-75's operator when dx equals dy; with dx != dy every "
            "pipb/pipdl/pipdn word would be the solution of a different equation"
        )
    if not (s.imirror and s.jmirror):
        # The nonlinear forcing's advu/advv/advw take the run's own family
        # (adv.F i1/i2, j1/j2, doitw..doitn), so an open or free-slip side is
        # the mirrored inversion this port has measured; a side without that
        # mirror is not.
        return (
            f"boundaries (wbc,ebc,sbc,nbc)={(s.wbc, s.ebc, s.sbc, s.nbc)}: "
            "param.F:5188-5205 gives this family poiss.F's un-mirrored "
            "transform (imirror or jmirror 0), and that inversion has not been "
            "compared with CM1's poiss.F/singleton.F; only the mirrored "
            "(open or free-slip rigid) inversion has"
        )
    if s.imoist == 1 and not (1 <= s.nqv and s.nql1 >= 1 and s.nql2 >= s.nql1):
        return "moisture layout has no nqv/nql1..nql2 axis for pdcomp.F:84-104"
    return None


# --------------------------------------------------------------------------
# Coefficients (init3d.F:1670-1693, base.F:2127-2135)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PdcompCoefficients:
    thv0: Any   # (nk,)
    cfa: Any    # (nk,)
    cfc: Any    # (nk,)
    d1: Any     # (nk,)  ad1
    d2: Any     # (nk,)  ad2
    cfb: Any    # (nk, jpe, ipe)
    ipe: int
    jpe: int
    sha256: str
    # duck-typed for cuda_anelp_poisson_periodic's shared Thomas solve
    stretch_z: int = 0
    vertical_grid_proof_sha256: str | None = None


def cm1_thv0_profile(th0: Any, qv0: Any, qc0: Any = None, qi0: Any = None) -> Any:
    """base.F:2130-2131; ``qc0``/``qi0`` default to CM1's zero."""

    np = _np()
    c = cm1_constants()
    th0 = np.asarray(th0, dtype=np.float32)
    qv0 = np.asarray(qv0, dtype=np.float32)
    z = np.float32(0.0)
    qc0 = np.zeros_like(qv0) if qc0 is None else np.asarray(qc0, dtype=np.float32)
    qi0 = np.zeros_like(qv0) if qi0 is None else np.asarray(qi0, dtype=np.float32)
    qvp = np.maximum(z, qv0)
    num = np.float32(1.0) + c.reps * qvp
    num = (th0 + z) * num
    den = (np.float32(1.0) + qvp) + np.maximum(z, (qc0 + qi0).astype(np.float32))
    return (num / den).astype(np.float32)


def build_pdcomp_coefficients(
    selection: PdcompSelection, *, mh: Any, mf: Any, rho0: Any, rf0: Any, thv0: Any,
) -> PdcompCoefficients:
    """``init3d.F:1675-1692`` (psolver<=3) on ``param.F:5178-5205``'s modes.

    ``mh``/``rho0``/``thv0`` are ``(nk,)`` columns (CM1 ``k=1..nk``);
    ``mf``/``rf0`` are ``(nk+1,)`` (CM1 ``k=1..nk+1``).  REAL statements are
    evaluated in binary32 left to right; the ``cfb`` line is DOUBLE PRECISION
    with ``dx*dx`` a REAL product, as Fortran evaluates it.
    """

    np = _np()
    s = selection
    nk = s.nk
    c = cm1_constants()
    mh = np.asarray(mh, dtype=np.float32)
    mf = np.asarray(mf, dtype=np.float32)
    rho0 = np.asarray(rho0, dtype=np.float32)
    rf0 = np.asarray(rf0, dtype=np.float32)
    thv0 = np.asarray(thv0, dtype=np.float32)
    for name, value, n in (("mh", mh, nk), ("rho0", rho0, nk), ("thv0", thv0, nk),
                           ("mf", mf, nk + 1), ("rf0", rf0, nk + 1)):
        if value.shape != (n,):
            raise ValueError(f"{name} column must have shape ({n},)")
    dz = np.float32(s.dz)
    dz2 = np.float32(dz * dz)
    half = np.float32(0.5)
    cfa = np.zeros(nk, dtype=np.float32)
    cfc = np.zeros(nk, dtype=np.float32)
    d1 = np.zeros(nk, dtype=np.float32)
    d2 = np.ones(nk, dtype=np.float32)
    for k in range(1, nk + 1):
        if k > 1:
            num = np.float32(np.float32(np.float32(mh[k - 1] * mf[k - 1]) * rf0[k - 1]) * half)
            num = np.float32(num * np.float32(thv0[k - 2] + thv0[k - 1]))
            den = np.float32(np.float32(dz2 * rho0[k - 1]) * thv0[k - 1])
            cfa[k - 1] = np.float32(num / den)
        if k < nk:
            num = np.float32(np.float32(np.float32(mh[k - 1] * mf[k]) * rf0[k]) * half)
            num = np.float32(num * np.float32(thv0[k - 1] + thv0[k]))
            den = np.float32(np.float32(dz2 * rho0[k - 1]) * thv0[k - 1])
            cfc[k - 1] = np.float32(num / den)
        d1[k - 1] = np.float32(np.float32(1.0) / np.float32(np.float32(c.cp * rho0[k - 1]) * thv0[k - 1]))
    # cfa(1)=0, cfc(nk)=0 (init3d.F:1680-1681, cfa(1)=0 and cfc(nk)=0): never computed above, so +0.0.
    ipe, jpe = s.ipe, s.jpe
    dpi = 4.0 * math.atan(1.0)
    dx = np.float32(s.dx)
    dxdx = float(np.float32(dx * dx))
    ci = np.array([math.cos(2.0 * dpi * float(i) / float(ipe)) for i in range(ipe)], dtype=np.float64)
    cj = np.array([math.cos(2.0 * dpi * float(j) / float(jpe)) for j in range(jpe)], dtype=np.float64)
    hmode = (2.0 * ((ci[None, :] + cj[:, None]) - 2.0)) / dxdx
    cfb = np.empty((nk, jpe, ipe), dtype=np.float32)
    for k0 in range(nk):
        cfb[k0] = ((hmode - float(cfa[k0])) - float(cfc[k0])).astype(np.float32)
    digest = hashlib.sha256()
    for value in (thv0, cfa, cfc, d1, d2, cfb):
        digest.update(np.ascontiguousarray(value).astype("<f4").tobytes())
    digest.update(repr((ipe, jpe, float(dx), float(dz))).encode())
    return PdcompCoefficients(thv0.copy(), cfa, cfc, d1, d2, cfb, ipe, jpe, digest.hexdigest())


# --------------------------------------------------------------------------
# CPU reference.  Layout: padded u (nk+2, nj+6, ni+7), v (nk+2, nj+7, ni+6),
# w (nk+3, nj+6, ni+6); CM1 (i,j,k) -> [k, j+2, i+2].  Physical scalars
# (nk, nj, ni): [k-1, j-1, i-1].  w-level metrics (nk+1, nj, ni): [k-1, ...].
# f2d (nj+2, ni+2): [j, i].
# --------------------------------------------------------------------------


def padded_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    ng = CM1_OPEN_NGXY
    return {
        "u": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "v": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "w": (nk + 3, nj + 2 * ng, ni + 2 * ng),
    }


def cpu_buoyancy_wten(s: PdcompSelection, *, th3d, q3d, th0, qv0, c1, c2, qc0=None, qi0=None) -> Any:
    """pdcomp.F:71-147."""

    np = _np()
    c = cm1_constants()
    ni, nj, nk = s.ni, s.nj, s.nk
    wten = np.zeros(padded_shapes(ni, nj, nk)["w"], dtype=np.float32)
    rth0 = (np.float32(1.0) / np.asarray(th0, dtype=np.float32))[:, None, None]
    if s.imoist == 1:
        dum1 = np.zeros((nk, nj, ni), dtype=np.float32)
        for n in range(s.nql1, s.nql2 + 1):
            dum1 = dum1 + q3d[n - 1]
        if s.iice == 1:
            for n in range(s.nqs1, s.nqs2 + 1):
                dum1 = dum1 + q3d[n - 1]
        qv0c = np.asarray(qv0, dtype=np.float32)[:, None, None]
        zero = np.zeros_like(np.asarray(qv0, dtype=np.float32))
        qc0c = (zero if qc0 is None else np.asarray(qc0, dtype=np.float32))[:, None, None]
        qi0c = (zero if qi0 is None else np.asarray(qi0, dtype=np.float32))[:, None, None]
        left = th3d * rth0 + c.repsm1 * (q3d[s.nqv - 1] - qv0c)
        dum6 = c.g * (left - ((dum1 - qc0c) - qi0c))
    else:
        dum6 = (c.g * th3d) * rth0
    dum6 = dum6.astype(np.float32)
    for k in range(2, nk + 1):
        wten[k, 3:nj + 3, 3:ni + 3] = c1[k - 1] * dum6[k - 2] + c2[k - 1] * dum6[k - 1]
    return wten


def cpu_linear_uv(s: PdcompSelection, *, w3d, rf0c, mfc, rr0c, u0, v0) -> tuple[Any, Any]:
    """pdcomp.F:196-236."""

    np = _np()
    ni, nj, nk = s.ni, s.nj, s.nk
    shapes = padded_shapes(ni, nj, nk)
    rdz = np.float32(np.float32(1.0) / np.float32(s.dz))
    half = np.float32(0.5)
    dum1 = np.zeros((nk + 2, nj + 1, ni + 1), dtype=np.float32)
    dum2 = np.zeros((nk + 2, nj + 1, ni + 1), dtype=np.float32)
    for k in range(2, nk + 1):
        dudz = -np.float32(np.float32(np.float32(np.float32(half * rf0c[k - 1]) * np.float32(u0[k - 1] - u0[k - 2])) * rdz) * mfc[k - 1])
        dvdz = -np.float32(np.float32(np.float32(np.float32(half * rf0c[k - 1]) * np.float32(v0[k - 1] - v0[k - 2])) * rdz) * mfc[k - 1])
        dum1[k] = dudz * (w3d[k, 3:nj + 4, 2:ni + 3] + w3d[k, 3:nj + 4, 3:ni + 4])
        dum2[k] = dvdz * (w3d[k, 2:nj + 3, 3:ni + 4] + w3d[k, 3:nj + 4, 3:ni + 4])
    uten = np.zeros(shapes["u"], dtype=np.float32)
    vten = np.zeros(shapes["v"], dtype=np.float32)
    for k in range(1, nk + 1):
        tem = np.float32(half * rr0c[k - 1])
        uten[k, 3:nj + 4, 3:ni + 4] = tem * (dum1[k] + dum1[k + 1])
        vten[k, 3:nj + 4, 3:ni + 4] = tem * (dum2[k] + dum2[k + 1])
    return uten, vten


def cpu_coriolis_uv(s: PdcompSelection, *, u3d, v3d, f2d) -> tuple[Any, Any]:
    """pdcomp.F:371-386."""

    np = _np()
    ni, nj, nk = s.ni, s.nj, s.nk
    shapes = padded_shapes(ni, nj, nk)
    q = np.float32(0.125)
    uten = np.zeros(shapes["u"], dtype=np.float32)
    vten = np.zeros(shapes["v"], dtype=np.float32)
    fu = q * (f2d[1:nj + 2, 1:ni + 2] + f2d[1:nj + 2, 0:ni + 1])
    fv = q * (f2d[1:nj + 2, 1:ni + 2] + f2d[0:nj + 1, 1:ni + 2])
    for k in range(1, nk + 1):
        b = (v3d[k, 3:nj + 4, 3:ni + 4] + v3d[k, 4:nj + 5, 3:ni + 4]) + (v3d[k, 3:nj + 4, 2:ni + 3] + v3d[k, 4:nj + 5, 2:ni + 3])
        uten[k, 3:nj + 4, 3:ni + 4] = fu * b
        b = (u3d[k, 3:nj + 4, 3:ni + 4] + u3d[k, 3:nj + 4, 4:ni + 5]) + (u3d[k, 2:nj + 3, 3:ni + 4] + u3d[k, 2:nj + 3, 4:ni + 5])
        vten[k, 3:nj + 4, 3:ni + 4] = -(fv * b)
    return uten, vten


def cpu_nonlinear_fluxes(s: PdcompSelection, *, u3d, v3d, w3d, rho0c, rf0c, mhc, u0, v0, uh, vh) -> dict[str, Any]:
    """pdcomp.F:271-315: rho0-weighted fluxes, perturbation winds, divx."""

    np = _np()
    ni, nj, nk = s.ni, s.nj, s.nk
    c = cm1_constants()
    shapes = padded_shapes(ni, nj, nk)
    out = {name: np.zeros(shapes[st], dtype=np.float32) for name, st in (
        ("rru", "u"), ("uten1", "u"), ("rrv", "v"), ("vten1", "v"), ("rrw", "w"))}
    for k in range(1, nk + 1):
        out["rru"][k] = rho0c[k - 1] * u3d[k]
        out["uten1"][k] = u3d[k] - u0[k - 1]
        out["rrv"][k] = rho0c[k - 1] * v3d[k]
        out["vten1"][k] = v3d[k] - v0[k - 1]
    for k in range(2, nk + 1):
        out["rrw"][k, 2:nj + 4, 2:ni + 4] = rf0c[k - 1] * w3d[k, 2:nj + 4, 2:ni + 4]
    rdx = np.float32(np.float32(1.0) / np.float32(s.dx))
    rdy = np.float32(np.float32(1.0) / np.float32(s.dy))
    rdz = np.float32(np.float32(1.0) / np.float32(s.dz))
    divx = np.zeros((nk, nj, ni), dtype=np.float32)
    rru, rrv, rrw = out["rru"], out["rrv"], out["rrw"]
    for k in range(1, nk + 1):
        a = ((rru[k, 3:nj + 3, 4:ni + 4] - rru[k, 3:nj + 3, 3:ni + 3]) * rdx) * uh[3:ni + 3]
        b = ((rrv[k, 4:nj + 4, 3:ni + 3] - rrv[k, 3:nj + 3, 3:ni + 3]) * rdy) * vh[3:nj + 3, None]
        z = ((rrw[k + 1, 3:nj + 3, 3:ni + 3] - rrw[k, 3:nj + 3, 3:ni + 3]) * rdz) * mhc[k - 1]
        d = ((a + b) + z).astype(np.float32)
        d[np.abs(d) < c.smeps] = np.float32(0.0)
        divx[k - 1] = d
    out["divx"] = divx
    return out


def cpu_poiss_rhs(s: PdcompSelection, coeff: PdcompCoefficients, *, uten, vten, wten,
                  rho0c, rf0, mh, uh, vh) -> Any:
    """poiss.F:62-114 into the (nk, jpe, ipe) mirrored real rectangle."""

    np = _np()
    ni, nj, nk = s.ni, s.nj, s.nk
    rdx = np.float32(np.float32(1.0) / np.float32(s.dx))
    rdy = np.float32(np.float32(1.0) / np.float32(s.dy))
    rdz = np.float32(np.float32(1.0) / np.float32(s.dz))
    tem = np.float32(np.float32(1.0) / np.float32(s.dt))
    divterm = np.float32(np.float32(0.0) * tem)
    rhs = np.zeros((nk, coeff.jpe, coeff.ipe), dtype=np.float32)
    for k in range(1, nk + 1):
        x = ((uten[k, 3:nj + 3, 4:ni + 4] - uten[k, 3:nj + 3, 3:ni + 3]) * rdx) * uh[3:ni + 3]
        y = ((vten[k, 4:nj + 4, 3:ni + 3] - vten[k, 3:nj + 3, 3:ni + 3]) * rdy) * vh[3:nj + 3, None]
        z = ((rf0[k] * wten[k + 1, 3:nj + 3, 3:ni + 3] - rf0[k - 1] * wten[k, 3:nj + 3, 3:ni + 3]) * rdz) * mh[k - 1]
        d = ((rho0c[k - 1] * (x + y)) + z) + divterm
        live = (d * coeff.d1[k - 1]).astype(np.float32)
        rhs[k - 1, :nj, :ni] = live
        if s.imirror:
            rhs[k - 1, :nj, coeff.ipe - ni:] = live[:, ::-1]
        if s.jmirror:
            rhs[k - 1, coeff.jpe - nj:, :ni] = live[::-1, :]
        if s.imirror and s.jmirror:
            rhs[k - 1, coeff.jpe - nj:, coeff.ipe - ni:] = live[::-1, ::-1]
    return rhs


def cpu_poiss_invert(s: PdcompSelection, coeff: PdcompCoefficients, rhs: Any) -> Any:
    """poiss.F:116-213 on the mirrored rectangle; returns (nk, nj, ni) ppi."""

    np = _np()
    from .cuda_anelp_poisson_periodic import cpu_anelp_spectral_vertical_solve

    ipe, jpe, nk = coeff.ipe, coeff.jpe, s.nk
    scale = np.float32(np.sqrt(np.float32(np.float32(1.0) / np.float32(ipe * jpe)), dtype=np.float32))
    scaled = np.empty(rhs.shape, dtype=np.complex64)
    scaled.real = np.multiply(rhs, scale, dtype=np.float32)
    scaled.imag.fill(np.float32(0.0))
    trans = np.fft.fft2(scaled, axes=(-2, -1), norm="backward").astype(np.complex64)
    solved = cpu_anelp_spectral_vertical_solve(trans, coeff, ni=ipe, nj=jpe, nk=nk)
    inverse = np.multiply(solved.pdt, np.complex64(scale), dtype=np.complex64)
    physical = np.fft.ifft2(inverse, axes=(-2, -1), norm="forward").astype(np.complex64)
    out = np.empty((nk, s.nj, s.ni), dtype=np.float32)
    for k0 in range(nk):
        out[k0] = np.multiply(physical[k0, :s.nj, :s.ni].real, coeff.d2[k0], dtype=np.float32)
    return out


def cpu_gauge(s: PdcompSelection, pdiag: Any, pp3d: Any | None) -> Any:
    """pdcomp.F:168-191 (``pp3d`` given) or :244-264 (``None``)."""

    np = _np()
    top = np.ascontiguousarray(pdiag[s.nk - 1], dtype=np.float64).reshape(-1)
    pavg2 = float(np.add.accumulate(np.concatenate(([0.0], top)))[-1]) / float(s.ni * s.nj)
    if pp3d is None:
        pavg1 = 0.0
    else:
        p = np.ascontiguousarray(pp3d[s.nk - 1], dtype=np.float64).reshape(-1)
        pavg1 = float(np.add.accumulate(np.concatenate(([0.0], p)))[-1]) / float(s.ni * s.nj)
    tem = np.float32(pavg1 - pavg2)
    return (pdiag + tem).astype(np.float32)


def cpu_pgrad(s: PdcompSelection, pdiag: Any, *, c1, c2, thv0, mf) -> Any:
    """writeout.F:8497-8505."""

    np = _np()
    c = cm1_constants()
    nk = s.nk
    rdz = np.float32(np.float32(1.0) / np.float32(s.dz))
    out = np.zeros((nk + 1, s.nj, s.ni), dtype=np.float32)
    for k in range(2, nk + 1):
        a = c2[k - 1] * thv0[k - 1] + c1[k - 1] * thv0[k - 2]
        out[k - 1] = -((((c.cp * a) * (pdiag[k - 1] - pdiag[k - 2])) * rdz) * mf[k - 1])
    return out


def cpu_pdcomp(
    s: PdcompSelection,
    coeff: PdcompCoefficients,
    *,
    u3d, v3d, w3d, th3d, pp3d, q3d, th0, qv0, u0, v0,
    rho0c, rr0c, rf0c, mhc, mfc, rf0, mh, mf, c1, c2, uh, vh,
    f2d: Any | None,
    advect: Callable[[Mapping[str, Any], Any], None],
) -> dict[str, Any]:
    """The whole of ``pidcomp`` plus ``writeout.F``'s gradients, on the host.

    ``advect(arrays, divx)`` adds CM1's advu/advv/advw increments to
    ``arrays['uten'/'vten'/'wten']`` for fields ``arrays['u3d'/'v3d'/'w3d']``
    and fluxes ``arrays['rru'/'rrv'/'rrw']``.
    """

    np = _np()
    shapes = padded_shapes(s.ni, s.nj, s.nk)
    zu = np.zeros(shapes["u"], dtype=np.float32)
    zv = np.zeros(shapes["v"], dtype=np.float32)
    zw = np.zeros(shapes["w"], dtype=np.float32)
    rhs_args = dict(rho0c=rho0c, rf0=rf0, mh=mh, uh=uh, vh=vh)
    out: dict[str, Any] = {}

    wten = cpu_buoyancy_wten(s, th3d=th3d, q3d=q3d, th0=th0, qv0=qv0, c1=c1, c2=c2)
    p = cpu_poiss_invert(s, coeff, cpu_poiss_rhs(s, coeff, uten=zu, vten=zv, wten=wten, **rhs_args))
    out["pipb"] = cpu_gauge(s, p, pp3d)

    uten, vten = cpu_linear_uv(s, w3d=w3d, rf0c=rf0c, mfc=mfc, rr0c=rr0c, u0=u0, v0=v0)
    p = cpu_poiss_invert(s, coeff, cpu_poiss_rhs(s, coeff, uten=uten, vten=vten, wten=zw, **rhs_args))
    out["pipdl"] = cpu_gauge(s, p, None)

    fl = cpu_nonlinear_fluxes(s, u3d=u3d, v3d=v3d, w3d=w3d, rho0c=rho0c, rf0c=rf0c,
                              mhc=mhc, u0=u0, v0=v0, uh=uh, vh=vh)
    arrays = {"u3d": fl["uten1"], "v3d": fl["vten1"], "w3d": w3d,
              "rru": fl["rru"], "rrv": fl["rrv"], "rrw": fl["rrw"],
              "uten": zu.copy(), "vten": zv.copy(), "wten": zw.copy(), "uh": uh, "vh": vh}
    advect(arrays, fl["divx"])
    p = cpu_poiss_invert(s, coeff, cpu_poiss_rhs(s, coeff, uten=arrays["uten"], vten=arrays["vten"], wten=arrays["wten"], **rhs_args))
    out["pipdn"] = cpu_gauge(s, p, None)

    if s.icor == 1:
        uten, vten = cpu_coriolis_uv(s, u3d=u3d, v3d=v3d, f2d=f2d)
        p = cpu_poiss_invert(s, coeff, cpu_poiss_rhs(s, coeff, uten=uten, vten=vten, wten=zw, **rhs_args))
        out["pipc"] = cpu_gauge(s, p, None)

    for name, wname in zip(s.names, s.w_names):
        out[wname] = cpu_pgrad(s, out[name], c1=c1, c2=c2, thv0=coeff.thv0, mf=mf)
    return out


def cpu_uniform_advect(metrics: Mapping[str, Any], s: PdcompSelection) -> Callable[[Mapping[str, Any], Any], None]:
    """This lane's scalar-dz advu/advv/advw CPU reference as pdcomp's advect."""

    from .cuda_momentum_advection5 import _cpu_momentum_transport5

    np = _np()

    def advect(arrays: Mapping[str, Any], divx: Any) -> None:
        _cpu_momentum_transport5(
            np, arrays, metrics, divx, ni=s.ni, nj=s.nj, nk=s.nk,
            rdx=np.float32(np.float32(1.0) / np.float32(s.dx)),
            rdy=np.float32(np.float32(1.0) / np.float32(s.dy)),
            rdz=np.float32(np.float32(1.0) / np.float32(s.dz)),
            lateral_sides=(s.wbc, s.ebc, s.sbc, s.nbc),
        )

    return advect


# --------------------------------------------------------------------------
# CUDA kernels
# --------------------------------------------------------------------------

CUDA_PDCOMP_OPEN_SOURCE = r"""// gpuwm-cm1.pdcomp-open-kernels/v1
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long pix(unsigned long long k,unsigned long long y,unsigned long long x,unsigned long long ny,unsigned long long nx){return (k*ny+y)*nx+x;}
__device__ __forceinline__ unsigned long long tid(){return (unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;}

__device__ __forceinline__ float pd_dum6(const float* th3d,const float* q3d,const float* th0,const float* qv0,const float* qc0,const float* qi0,unsigned int k,unsigned int j0,unsigned int i0,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int moist,unsigned int iice,unsigned int nqv,unsigned int nql1,unsigned int nql2,unsigned int nqs1,unsigned int nqs2,float g,float repsm1){
 unsigned long long n3=(unsigned long long)nk*nj*ni,c=pix(k-1u,j0,i0,nj,ni);
 float rth0=d32(1.0f,th0[k-1u]);
 if(moist==0u)return m32(m32(g,th3d[c]),rth0);
 float s=0.0f;
 for(unsigned int n=nql1;n<=nql2;++n)s=a32(s,q3d[(unsigned long long)(n-1u)*n3+c]);
 if(iice==1u){for(unsigned int n=nqs1;n<=nqs2;++n)s=a32(s,q3d[(unsigned long long)(n-1u)*n3+c]);}
 float left=a32(m32(th3d[c],rth0),m32(repsm1,s32(q3d[(unsigned long long)(nqv-1u)*n3+c],qv0[k-1u])));
 return m32(g,s32(left,s32(s32(s,qc0[k-1u]),qi0[k-1u])));
}

extern "C" __global__ void cm1_pdcomp_buoyancy_v1(const float* th3d,const float* q3d,const float* th0,const float* qv0,const float* qc0,const float* qi0,const float* c1,const float* c2,float* wten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int moist,unsigned int iice,unsigned int nqv,unsigned int nql1,unsigned int nql2,unsigned int nqs1,unsigned int nqs2,float g,float repsm1){
 unsigned long long q=tid(),plane=(unsigned long long)nj*ni;if(q>=(unsigned long long)(nk-1u)*plane)return;
 unsigned int k=2u+(unsigned int)(q/plane),j0=(unsigned int)((q/ni)%nj),i0=(unsigned int)(q%ni);
 float lo=pd_dum6(th3d,q3d,th0,qv0,qc0,qi0,k-1u,j0,i0,ni,nj,nk,moist,iice,nqv,nql1,nql2,nqs1,nqs2,g,repsm1);
 float hi=pd_dum6(th3d,q3d,th0,qv0,qc0,qi0,k,j0,i0,ni,nj,nk,moist,iice,nqv,nql1,nql2,nqs1,nqs2,g,repsm1);
 unsigned long long m=pix(k-1u,j0,i0,nj,ni);
 wten[pix(k,j0+3u,i0+3u,nj+6u,ni+6u)]=a32(m32(c1[m],lo),m32(c2[m],hi));
}

__device__ __forceinline__ float pd_ldum(const float* w,const float* rf0c,const float* mfc,const float* p0,unsigned int k,unsigned int j,unsigned int i,unsigned int dj,unsigned int di,unsigned int ni,unsigned int nj,unsigned int nk,float rdz){
 if(k<2u||k>nk)return 0.0f;
 float d=-m32(m32(m32(m32(0.5f,rf0c[k-1u]),s32(p0[k-1u],p0[k-2u])),rdz),mfc[k-1u]);
 unsigned long long nyw=nj+6u,nxw=ni+6u;
 return m32(d,a32(w[pix(k,j+2u-dj,i+2u-di,nyw,nxw)],w[pix(k,j+2u,i+2u,nyw,nxw)]));
}

extern "C" __global__ void cm1_pdcomp_linear_v1(const float* w3d,const float* rf0c,const float* mfc,const float* rr0c,const float* u0,const float* v0,float* uten,float* vten,unsigned int ni,unsigned int nj,unsigned int nk,float rdz){
 unsigned long long q=tid(),plane=(unsigned long long)(nj+1u)*(ni+1u);if(q>=(unsigned long long)nk*plane)return;
 unsigned int k=1u+(unsigned int)(q/plane),j=1u+(unsigned int)((q/(ni+1u))%(nj+1u)),i=1u+(unsigned int)(q%(ni+1u));
 float tem=m32(0.5f,rr0c[k-1u]);
 uten[pix(k,j+2u,i+2u,nj+6u,ni+7u)]=m32(tem,a32(pd_ldum(w3d,rf0c,mfc,u0,k,j,i,0u,1u,ni,nj,nk,rdz),pd_ldum(w3d,rf0c,mfc,u0,k+1u,j,i,0u,1u,ni,nj,nk,rdz)));
 vten[pix(k,j+2u,i+2u,nj+7u,ni+6u)]=m32(tem,a32(pd_ldum(w3d,rf0c,mfc,v0,k,j,i,1u,0u,ni,nj,nk,rdz),pd_ldum(w3d,rf0c,mfc,v0,k+1u,j,i,1u,0u,ni,nj,nk,rdz)));
}

extern "C" __global__ void cm1_pdcomp_coriolis_v1(const float* u3d,const float* v3d,const float* f2d,float* uten,float* vten,unsigned int ni,unsigned int nj,unsigned int nk){
 unsigned long long q=tid(),plane=(unsigned long long)(nj+1u)*(ni+1u);if(q>=(unsigned long long)nk*plane)return;
 unsigned int k=1u+(unsigned int)(q/plane),j=1u+(unsigned int)((q/(ni+1u))%(nj+1u)),i=1u+(unsigned int)(q%(ni+1u));
 unsigned long long fx=ni+2u,nyu=nj+6u,nxu=ni+7u,nyv=nj+7u,nxv=ni+6u;
 float fu=m32(0.125f,a32(f2d[(unsigned long long)j*fx+i],f2d[(unsigned long long)j*fx+i-1u]));
 float bv=a32(a32(v3d[pix(k,j+2u,i+2u,nyv,nxv)],v3d[pix(k,j+3u,i+2u,nyv,nxv)]),a32(v3d[pix(k,j+2u,i+1u,nyv,nxv)],v3d[pix(k,j+3u,i+1u,nyv,nxv)]));
 uten[pix(k,j+2u,i+2u,nyu,nxu)]=m32(fu,bv);
 float fv=m32(0.125f,a32(f2d[(unsigned long long)j*fx+i],f2d[(unsigned long long)(j-1u)*fx+i]));
 float bu=a32(a32(u3d[pix(k,j+2u,i+2u,nyu,nxu)],u3d[pix(k,j+2u,i+3u,nyu,nxu)]),a32(u3d[pix(k,j+1u,i+2u,nyu,nxu)],u3d[pix(k,j+1u,i+3u,nyu,nxu)]));
 vten[pix(k,j+2u,i+2u,nyv,nxv)]=-m32(fv,bu);
}

extern "C" __global__ void cm1_pdcomp_nl_uv_v1(const float* s3d,const float* rho0c,const float* s0,float* rr,float* pert,unsigned int ny,unsigned int nx,unsigned int nk){
 unsigned long long q=tid(),plane=(unsigned long long)ny*nx;if(q>=(unsigned long long)nk*plane)return;
 unsigned int k=1u+(unsigned int)(q/plane);unsigned long long x=(unsigned long long)k*plane+q%plane;
 rr[x]=m32(rho0c[k-1u],s3d[x]);pert[x]=s32(s3d[x],s0[k-1u]);
}

extern "C" __global__ void cm1_pdcomp_nl_w_v1(const float* w3d,const float* rf0c,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk){
 unsigned long long q=tid(),plane=(unsigned long long)(nj+2u)*(ni+2u);if(q>=(unsigned long long)(nk-1u)*plane)return;
 unsigned int k=2u+(unsigned int)(q/plane),y=2u+(unsigned int)((q/(ni+2u))%(nj+2u)),x=2u+(unsigned int)(q%(ni+2u));
 unsigned long long c=pix(k,y,x,nj+6u,ni+6u);rrw[c]=m32(rf0c[k-1u],w3d[c]);
}

extern "C" __global__ void cm1_pdcomp_divx_v1(const float* rru,const float* rrv,const float* rrw,const float* uh,const float* vh,const float* mhc,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,float rdx,float rdy,float rdz,float smeps){
 unsigned long long q=tid(),plane=(unsigned long long)nj*ni;if(q>=(unsigned long long)nk*plane)return;
 unsigned int k=1u+(unsigned int)(q/plane),j=1u+(unsigned int)((q/ni)%nj),i=1u+(unsigned int)(q%ni);
 float a=m32(m32(s32(rru[pix(k,j+2u,i+3u,nj+6u,ni+7u)],rru[pix(k,j+2u,i+2u,nj+6u,ni+7u)]),rdx),uh[i+2u]);
 float b=m32(m32(s32(rrv[pix(k,j+3u,i+2u,nj+7u,ni+6u)],rrv[pix(k,j+2u,i+2u,nj+7u,ni+6u)]),rdy),vh[j+2u]);
 float z=m32(m32(s32(rrw[pix(k+1u,j+2u,i+2u,nj+6u,ni+6u)],rrw[pix(k,j+2u,i+2u,nj+6u,ni+6u)]),rdz),mhc[k-1u]);
 float d=a32(a32(a,b),z);float ad=d<0.0f?-d:d;
 divx[q]=(ad<smeps)?0.0f:d;
}

extern "C" __global__ void cm1_pdcomp_poiss_rhs_v1(const float* uten,const float* vten,const float* wten,const float* rho0c,const float* rf0,const float* mh,const float* d1,const float* uh,const float* vh,float* rhs,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ipe,unsigned int jpe,unsigned int imirror,unsigned int jmirror,float rdx,float rdy,float rdz,float rdt){
 unsigned long long q=tid(),plane=(unsigned long long)nj*ni;if(q>=(unsigned long long)nk*plane)return;
 unsigned int k=1u+(unsigned int)(q/plane),j=1u+(unsigned int)((q/ni)%nj),i=1u+(unsigned int)(q%ni);
 float x=m32(m32(s32(uten[pix(k,j+2u,i+3u,nj+6u,ni+7u)],uten[pix(k,j+2u,i+2u,nj+6u,ni+7u)]),rdx),uh[i+2u]);
 float y=m32(m32(s32(vten[pix(k,j+3u,i+2u,nj+7u,ni+6u)],vten[pix(k,j+2u,i+2u,nj+7u,ni+6u)]),rdy),vh[j+2u]);
 unsigned long long m=pix(k-1u,j-1u,i-1u,nj,ni);
 float z=m32(m32(s32(m32(rf0[m+plane],wten[pix(k+1u,j+2u,i+2u,nj+6u,ni+6u)]),m32(rf0[m],wten[pix(k,j+2u,i+2u,nj+6u,ni+6u)])),rdz),mh[m]);
 float def=a32(a32(m32(rho0c[k-1u],a32(x,y)),z),m32(0.0f,rdt));
 float r=m32(def,d1[k-1u]);
 unsigned long long base=(unsigned long long)(k-1u)*jpe*ipe;
 unsigned int jj=j-1u,ii=i-1u;
 rhs[base+(unsigned long long)jj*ipe+ii]=r;
 if(imirror)rhs[base+(unsigned long long)jj*ipe+(ipe-1u-ii)]=r;
 if(jmirror)rhs[base+(unsigned long long)(jpe-1u-jj)*ipe+ii]=r;
 if(imirror&&jmirror)rhs[base+(unsigned long long)(jpe-1u-jj)*ipe+(ipe-1u-ii)]=r;
}

extern "C" __global__ void cm1_pdcomp_poiss_store_v1(const float* physical_real,const float* d2,float* out,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ipe,unsigned int jpe){
 unsigned long long q=tid(),plane=(unsigned long long)nj*ni;if(q>=(unsigned long long)nk*plane)return;
 unsigned int k=(unsigned int)(q/plane),j=(unsigned int)((q/ni)%nj),i=(unsigned int)(q%ni);
 out[q]=m32(physical_real[((unsigned long long)k*jpe+j)*ipe+i],d2[k]);
}

extern "C" __global__ void cm1_pdcomp_gauge_v1(const float* pdiag,const float* pp3d,float* tem,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int use_pp3d){
 if(blockIdx.x||threadIdx.x)return;
 unsigned long long base=(unsigned long long)(nk-1u)*nj*ni;
 double pavg1=0.0,pavg2=0.0;
 for(unsigned int j=0u;j<nj;++j)for(unsigned int i=0u;i<ni;++i){unsigned long long c=base+(unsigned long long)j*ni+i;if(use_pp3d)pavg1=pavg1+(double)pp3d[c];pavg2=pavg2+(double)pdiag[c];}
 double n=(double)((unsigned long long)ni*nj);
 if(use_pp3d)pavg1=pavg1/n;
 pavg2=pavg2/n;
 tem[0]=(float)(pavg1-pavg2);
}

extern "C" __global__ void cm1_pdcomp_add_v1(float* pdiag,float tem,unsigned long long n){
 unsigned long long q=tid();if(q>=n)return;pdiag[q]=a32(pdiag[q],tem);
}

extern "C" __global__ void cm1_pdcomp_pgrad_v1(const float* p,const float* c1,const float* c2,const float* thv0c,const float* mf,float* out,unsigned int ni,unsigned int nj,unsigned int nk,float cp,float rdz){
 unsigned long long q=tid(),plane=(unsigned long long)nj*ni;if(q>=(unsigned long long)(nk-1u)*plane)return;
 unsigned int k=2u+(unsigned int)(q/plane);unsigned long long r=q%plane,m=(unsigned long long)(k-1u)*plane+r;
 float a=a32(m32(c2[m],thv0c[k-1u]),m32(c1[m],thv0c[k-2u]));
 out[m]=-m32(m32(m32(m32(cp,a),s32(p[m],p[m-plane])),rdz),mf[m]);
}
"""
CUDA_PDCOMP_OPEN_SOURCE_SHA256 = hashlib.sha256(CUDA_PDCOMP_OPEN_SOURCE.encode("utf-8")).hexdigest()
CUDA_PDCOMP_OPEN_KERNEL_NAMES: tuple[str, ...] = (
    "cm1_pdcomp_buoyancy_v1",
    "cm1_pdcomp_linear_v1",
    "cm1_pdcomp_coriolis_v1",
    "cm1_pdcomp_nl_uv_v1",
    "cm1_pdcomp_nl_w_v1",
    "cm1_pdcomp_divx_v1",
    "cm1_pdcomp_poiss_rhs_v1",
    "cm1_pdcomp_poiss_store_v1",
    "cm1_pdcomp_gauge_v1",
    "cm1_pdcomp_add_v1",
    "cm1_pdcomp_pgrad_v1",
)
CUDA_PDCOMP_OPEN_THREADS = 128


@dataclass(frozen=True, slots=True)
class CompiledPdcompOpen:
    module_owner: Any
    kernels: Mapping[str, Any]
    source_sha256: str


def compile_cuda_pdcomp_open() -> CompiledPdcompOpen:
    from .nvrtc_module import compile_nvrtc_module

    cupy = importlib.import_module("cupy")
    module = compile_nvrtc_module(
        cupy,
        code=CUDA_PDCOMP_OPEN_SOURCE,
        filename="gm1_pdcomp_open.cu",
        options=("--std=c++14", "--fmad=false", "--ftz=false",
                 "--prec-div=true", "--prec-sqrt=true"),
        name_expressions=CUDA_PDCOMP_OPEN_KERNEL_NAMES,
    )
    return CompiledPdcompOpen(
        module,
        {name: module.get_function(name) for name in CUDA_PDCOMP_OPEN_KERNEL_NAMES},
        CUDA_PDCOMP_OPEN_SOURCE_SHA256,
    )


@dataclass(slots=True)
class DevicePdcompPlan:
    selection: PdcompSelection
    coefficients: PdcompCoefficients
    device: SimpleNamespace          # cfa/cfc/cfb/d1/d2/thv0 + metric columns
    compiled: CompiledPdcompOpen
    poisson_compiled: Any


def _launch(kernel: Any, count: int, args: tuple, stream: Any) -> None:
    t = CUDA_PDCOMP_OPEN_THREADS
    kernel(((count + t - 1) // t, 1, 1), (t, 1, 1), args, stream=stream)


def prepare_cuda_pdcomp(
    selection: PdcompSelection, coefficients: PdcompCoefficients, *,
    metrics: Mapping[str, Any], base_profiles: Mapping[str, Any],
    f2d: Any | None, stream: Any,
    compiled: CompiledPdcompOpen | None = None, poisson_compiled: Any | None = None,
) -> DevicePdcompPlan:
    """Upload the host-built coefficients and contiguous metric columns once."""

    cupy = importlib.import_module("cupy")
    from .cuda_anelp_poisson_periodic import compile_cuda_anelp_poisson

    defect = pdcomp_selection_defect(selection)
    if defect is not None:
        raise ValueError(f"output_pdcomp refused: {defect}")
    s = selection
    with stream:
        c = lambda a: cupy.ascontiguousarray(cupy.asarray(a, dtype=cupy.float32))  # noqa: E731
        device = SimpleNamespace(
            cfa=c(coefficients.cfa), cfc=c(coefficients.cfc), cfb=c(coefficients.cfb),
            d1=c(coefficients.d1), d2=c(coefficients.d2), thv0=c(coefficients.thv0),
            stretch_z=0, vertical_grid_proof_sha256=None,
            rho0c=c(metrics["rho0_3d"][:, 0, 0]), rr0c=c(metrics["rr0"][:, 0, 0]),
            rf0c=c(metrics["rf0"][:, 0, 0]), mhc=c(metrics["mh"][:, 0, 0]),
            mfc=c(metrics["mf"][:, 0, 0]),
            rf0=c(metrics["rf0"]), mh=c(metrics["mh"]), mf=c(metrics["mf"]),
            c1=c(metrics["c1"]), c2=c(metrics["c2"]),
            th0=c(base_profiles["th0"]), qv0=c(base_profiles["qv0"]),
            # base.F's qc0/qi0 (pdcomp.F:104); exact zero for a base state without cloud.
            qc0=c(base_profiles["qc0"]) if "qc0" in base_profiles else cupy.zeros_like(c(base_profiles["qv0"])),
            qi0=c(base_profiles["qi0"]) if "qi0" in base_profiles else cupy.zeros_like(c(base_profiles["qv0"])),
            u0=c(base_profiles["u0"]), v0=c(base_profiles["v0"]),
            f2d=None if f2d is None else c(f2d),
        )
    if s.icor == 1 and device.f2d is None:
        raise ValueError("icor=1 pdcomp needs f2d (param.F:8945-8969)")
    return DevicePdcompPlan(
        s, coefficients, device,
        compiled or compile_cuda_pdcomp_open(),
        poisson_compiled or compile_cuda_anelp_poisson(),
    )


def launch_cuda_pdcomp(
    plan: DevicePdcompPlan, *,
    u3d: Any, v3d: Any, w3d: Any, th3d: Any, pp3d: Any, q3d: Any, uh: Any, vh: Any,
    advect: Callable[[Mapping[str, Any], Any], None],
    stream: Any,
) -> dict[str, Any]:
    """Enqueue ``pidcomp`` and the gradients; returns device arrays by CM1 name.

    Every model carrier passed here is read through a ``const`` kernel
    parameter or by ``advect`` as a field/flux source; the writable buffers are
    all allocated here.
    """

    cupy = importlib.import_module("cupy")
    np = _np()
    from .cuda_anelp_poisson_periodic import launch_cuda_anelp_spectral_vertical_solve

    s, d, k = plan.selection, plan.device, plan.compiled.kernels
    ni, nj, nk, ipe, jpe = s.ni, s.nj, s.nk, plan.coefficients.ipe, plan.coefficients.jpe
    shapes = padded_shapes(ni, nj, nk)
    const = cm1_constants()
    u32, f32 = cupy.uint32, cupy.float32
    rdx = np.float32(np.float32(1.0) / np.float32(s.dx))
    rdy = np.float32(np.float32(1.0) / np.float32(s.dy))
    rdz = np.float32(np.float32(1.0) / np.float32(s.dz))
    rdt = np.float32(np.float32(1.0) / np.float32(s.dt))
    scale = np.float32(np.sqrt(np.float32(np.float32(1.0) / np.float32(ipe * jpe)), dtype=np.float32))
    th3d, pp3d, q3d = (cupy.ascontiguousarray(a) for a in (th3d, pp3d, q3d))
    out: dict[str, Any] = {}

    with stream:
        def zeros(st: str) -> Any:
            return cupy.zeros(shapes[st], dtype=cupy.float32)

        def invert(uten: Any, vten: Any, wten: Any, gauge_pp3d: bool) -> Any:
            rhs = cupy.zeros((nk, jpe, ipe), dtype=cupy.float32)
            _launch(k["cm1_pdcomp_poiss_rhs_v1"], nk * nj * ni, (
                uten, vten, wten, d.rho0c, d.rf0, d.mh, d.d1, uh, vh, rhs,
                u32(ni), u32(nj), u32(nk), u32(ipe), u32(jpe),
                u32(s.imirror), u32(s.jmirror), f32(rdx), f32(rdy), f32(rdz), f32(rdt)), stream)
            scaled = rhs.astype(cupy.complex64) * cupy.complex64(scale)
            trans = cupy.fft.fft2(scaled, axes=(-2, -1), norm="backward").astype(cupy.complex64)
            spectral = launch_cuda_anelp_spectral_vertical_solve(
                trans, d, ni=ipe, nj=jpe, nk=nk, stream=stream, compiled=plan.poisson_compiled)
            physical = cupy.fft.ifft2(spectral.pdt * cupy.complex64(scale), axes=(-2, -1),
                                      norm="forward").astype(cupy.complex64)
            real = cupy.ascontiguousarray(physical.real)
            p = cupy.empty((nk, nj, ni), dtype=cupy.float32)
            _launch(k["cm1_pdcomp_poiss_store_v1"], nk * nj * ni,
                    (real, d.d2, p, u32(ni), u32(nj), u32(nk), u32(ipe), u32(jpe)), stream)
            tem = cupy.zeros((1,), dtype=cupy.float32)
            k["cm1_pdcomp_gauge_v1"]((1, 1, 1), (1, 1, 1), (
                p, pp3d, tem, u32(ni), u32(nj), u32(nk), u32(1 if gauge_pp3d else 0)), stream=stream)
            stream.synchronize()
            _launch(k["cm1_pdcomp_add_v1"], p.size,
                    (p, f32(float(tem.get()[0])), cupy.uint64(p.size)), stream)
            return p

        # pdcomp.F:71-191
        wten = zeros("w")
        _launch(k["cm1_pdcomp_buoyancy_v1"], (nk - 1) * nj * ni, (
            th3d, q3d, d.th0, d.qv0, d.qc0, d.qi0, d.c1, d.c2, wten, u32(ni), u32(nj), u32(nk),
            u32(s.imoist), u32(s.iice), u32(s.nqv), u32(s.nql1), u32(s.nql2),
            u32(s.nqs1), u32(s.nqs2), f32(const.g), f32(const.repsm1)), stream)
        out["pipb"] = invert(zeros("u"), zeros("v"), wten, True)

        # pdcomp.F:196-264
        uten, vten = zeros("u"), zeros("v")
        _launch(k["cm1_pdcomp_linear_v1"], nk * (nj + 1) * (ni + 1), (
            w3d, d.rf0c, d.mfc, d.rr0c, d.u0, d.v0, uten, vten,
            u32(ni), u32(nj), u32(nk), f32(rdz)), stream)
        out["pipdl"] = invert(uten, vten, zeros("w"), False)

        # pdcomp.F:269-364
        rru, uten1, rrv, vten1, rrw = zeros("u"), zeros("u"), zeros("v"), zeros("v"), zeros("w")
        _launch(k["cm1_pdcomp_nl_uv_v1"], nk * shapes["u"][1] * shapes["u"][2], (
            u3d, d.rho0c, d.u0, rru, uten1, u32(shapes["u"][1]), u32(shapes["u"][2]), u32(nk)), stream)
        _launch(k["cm1_pdcomp_nl_uv_v1"], nk * shapes["v"][1] * shapes["v"][2], (
            v3d, d.rho0c, d.v0, rrv, vten1, u32(shapes["v"][1]), u32(shapes["v"][2]), u32(nk)), stream)
        _launch(k["cm1_pdcomp_nl_w_v1"], (nk - 1) * (nj + 2) * (ni + 2), (
            w3d, d.rf0c, rrw, u32(ni), u32(nj), u32(nk)), stream)
        divx = cupy.zeros((nk, nj, ni), dtype=cupy.float32)
        _launch(k["cm1_pdcomp_divx_v1"], nk * nj * ni, (
            rru, rrv, rrw, uh, vh, d.mhc, divx, u32(ni), u32(nj), u32(nk),
            f32(rdx), f32(rdy), f32(rdz), f32(const.smeps)), stream)
        arrays = {"u3d": uten1, "v3d": vten1, "w3d": w3d, "rru": rru, "rrv": rrv, "rrw": rrw,
                  "uten": zeros("u"), "vten": zeros("v"), "wten": zeros("w"), "uh": uh, "vh": vh}
        advect(arrays, divx)
        out["pipdn"] = invert(arrays["uten"], arrays["vten"], arrays["wten"], False)

        # pdcomp.F:369-414
        if s.icor == 1:
            uten, vten = zeros("u"), zeros("v")
            _launch(k["cm1_pdcomp_coriolis_v1"], nk * (nj + 1) * (ni + 1), (
                u3d, v3d, d.f2d, uten, vten, u32(ni), u32(nj), u32(nk)), stream)
            out["pipc"] = invert(uten, vten, zeros("w"), False)

        # writeout.F:8478-8580
        for name, wname in zip(s.names, s.w_names):
            g = cupy.zeros((nk + 1, nj, ni), dtype=cupy.float32)
            _launch(k["cm1_pdcomp_pgrad_v1"], (nk - 1) * nj * ni, (
                out[name], d.c1, d.c2, d.thv0, d.mf, g, u32(ni), u32(nj), u32(nk),
                f32(const.cp), f32(rdz)), stream)
            out[wname] = g
        stream.synchronize()
    return out


def device_advect(
    compiled: Any, metrics: Mapping[str, Any], selection: PdcompSelection, *,
    vertical: Any | None, stream: Any,
) -> Callable[[Mapping[str, Any], Any], None]:
    """This lane's own advu/advv/advw kernels on pdcomp's buffers.

    ``vertical`` is ``None`` for the scalar-dz family
    (:mod:`cm1core.cuda_momentum_advection5`), else the stretched binding
    :mod:`cm1core.cuda_gm1_child_stretched_momentum` reads: the flat
    column's ``mh``/``mf`` and ``rdz`` (terrain_flag=.false., the Cartesian
    lane) or the terrain child's ``rds``/``rdsf``.  The argument tuples are
    the ones those launchers pass.
    """

    cupy = importlib.import_module("cupy")
    from .cuda_momentum_advection5 import CUDA_MOMENTUM_ADVECTION5_THREADS as T
    from .cuda_momentum_advection5 import momentum_advection5_counts, momentum_open_boundary_code

    s = selection
    ni, nj, nk, ng = s.ni, s.nj, s.nk, CM1_OPEN_NGXY
    _, _, _, advu, advv, advw = compiled.kernels
    u32, f32 = cupy.uint32, cupy.float32
    m = metrics
    # pdcomp.F:317-326 calls advu/advv/advw with the run's own lateral family.
    sides = (s.wbc, s.ebc, s.sbc, s.nbc)
    obc = u32(momentum_open_boundary_code(sides))
    nu, nv, nw = momentum_advection5_counts(ni, nj, nk, sides)

    def grid(count: int) -> tuple[int, int, int]:
        return ((count + T - 1) // T, 1, 1)

    flat = None
    if vertical is not None and vertical.flat_column:
        from .cuda_gm1_child_stretched_metrics import require_flat_column

        flat = require_flat_column(vertical, nk=nk, cupy=cupy)

    def advect(a: Mapping[str, Any], divx: Any) -> None:
        block = (T, 1, 1)
        with stream:
            if vertical is None:
                advu(grid(nu), block, (a["u3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rho0_3d"], m["c2"], m["gzu"], a["vh"], a["uten"], u32(ni), u32(nj), u32(nk), u32(ng), f32(s.dx), f32(s.dy), f32(s.dz), obc), stream=stream)
                advv(grid(nv), block, (a["v3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rho0_3d"], m["c2"], m["gzv"], a["uh"], a["vten"], u32(ni), u32(nj), u32(nk), u32(ng), f32(s.dx), f32(s.dy), f32(s.dz), obc), stream=stream)
                advw(grid(nw), block, (a["w3d"], a["rru"], a["rrv"], a["rrw"], divx, m["c2"], m["rrf0"], m["gz"], a["uh"], a["vh"], a["wten"], u32(ni), u32(nj), u32(nk), u32(ng), f32(s.dx), f32(s.dy), f32(s.dz), obc), stream=stream)
            elif flat is not None:
                mh, mf, rdz = flat
                tail = (u32(ni), u32(nj), u32(nk), u32(ng), f32(s.dx), f32(s.dy), f32(rdz))
                advu(grid(nu), block, (a["u3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rr0"], m["c2"], a["vh"], mh, a["uten"], *tail, obc), stream=stream)
                advv(grid(nv), block, (a["v3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rr0"], m["c2"], a["uh"], mh, a["vten"], *tail, obc), stream=stream)
                advw(grid(nw), block, (a["w3d"], a["rru"], a["rrv"], a["rrw"], divx, m["c2"], m["rrf0"], a["uh"], a["vh"], mf, a["wten"], *tail, obc), stream=stream)
            else:
                advu(grid(nu), block, (a["u3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rho0_3d"], m["c2"], m["gzu"], a["vh"], vertical.rdsf, a["uten"], u32(ni), u32(nj), u32(nk), u32(ng), f32(s.dx), f32(s.dy), obc), stream=stream)
                advv(grid(nv), block, (a["v3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rho0_3d"], m["c2"], m["gzv"], a["uh"], vertical.rdsf, a["vten"], u32(ni), u32(nj), u32(nk), u32(ng), f32(s.dx), f32(s.dy), obc), stream=stream)
                advw(grid(nw), block, (a["w3d"], a["rru"], a["rrv"], a["rrw"], divx, m["c2"], m["rrf0"], m["gz"], a["uh"], a["vh"], vertical.rds, a["wten"], u32(ni), u32(nj), u32(nk), u32(ng), f32(s.dx), f32(s.dy), obc), stream=stream)

    return advect


def pdcomp_record(selection: PdcompSelection, coefficients: PdcompCoefficients) -> dict[str, Any]:
    return {
        "schema": PDCOMP_OPEN_SCHEMA,
        "fields_scalar": list(selection.names),
        "fields_w": list(selection.w_names),
        "units": {"pip*": "nondimensional (pi-prime)", "pgrad*": "m/s/s"},
        "poisson_inversion_performed": True,
        "lateral": {"imirror": selection.imirror, "jmirror": selection.jmirror,
                    "ipe": coefficients.ipe, "jpe": coefficients.jpe,
                    "condition": "even reflection (zero normal gradient) on open sides, param.F:5188-5199"},
        "coefficient_sha256": coefficients.sha256,
        "kernel_source_sha256": CUDA_PDCOMP_OPEN_SOURCE_SHA256,
        "fft": "cuFFT via cupy.fft with Singleton's unitary pre-scale; Singleton itself is not bitwise reproduced",
        "cm1_authority": dict(SOURCE_ANCHORS),
    }


__all__ = [
    "CUDA_PDCOMP_OPEN_KERNEL_NAMES", "CUDA_PDCOMP_OPEN_SOURCE", "CUDA_PDCOMP_OPEN_SOURCE_SHA256",
    "PDCOMP_OPEN_SCHEMA", "PDCOMP_SCALAR_NAMES", "PDCOMP_W_NAMES", "PdcompCoefficients",
    "PdcompSelection", "SOURCE_ANCHORS", "build_pdcomp_coefficients", "cm1_constants",
    "cm1_thv0_profile", "compile_cuda_pdcomp_open", "cpu_pdcomp", "cpu_uniform_advect",
    "device_advect", "launch_cuda_pdcomp", "padded_shapes", "pdcomp_record",
    "pdcomp_selection_defect", "prepare_cuda_pdcomp",
]
