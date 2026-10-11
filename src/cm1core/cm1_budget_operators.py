"""Array arithmetic behind CM1's per-process budgets and GM1's vorticity budget.

Every function here takes the array module ``xp`` explicitly, so the same
lines run on CuPy device arrays in a run and on NumPy host arrays in a test,
and every operation is a separate binary32 array expression -- there is no
contraction and no reordering to argue about.  Nothing in this module writes
into an argument it did not allocate: each result is a new array.

What is CM1's and what is GM1's
-------------------------------
*CM1 r21.1 transcriptions* (the formulas CM1's own budget code evaluates):

* ``nondiffusive_*``: the sixth-order flux (``adv_routines.F:6026-6037``
  ``flx6``), the lower-order vertical closures (``adv_routines.F:5518-5740``
  ``vadv_lwr_flx_2``/``vadv_lwr_flx_4``) and the ``ndiff`` expressions of the
  implicit-diffusion split -- ``adv.F:1640-1830`` (u), ``adv.F:2560-2750``
  (v), ``adv.F:3441-3600`` (w), ``adv.F:790-900`` (scalars).  CM1 publishes
  ``*_hadv = ndiff`` and ``*_hidiff = tot - ndiff``.
* ``pressure_gradient_residual``: ``solve2.F:925-956`` saves the explicit
  tendency before the acoustic steps and ``solve2.F:1196-1245`` forms
  ``(u3d-ua)*rdt - saved``.
* ``calcvort_*``: ``misclibs.F:1618-2056`` ``calcvort``, the no-terrain
  Cartesian branch with its ``mf`` vertical metric and its ``bbc``/``tbc``
  closures.

*GM1 extension* (CM1 r21.1 has no vorticity budget anywhere):

* ``curl_budget_term``: ``calcvort`` applied to a momentum *tendency* instead
  of a wind.  ``calcvort`` is linear, so the curl of the sum of CM1's momentum
  budget terms is exactly the tendency of the vorticity CM1's ``output_vort``
  publishes; the budget closes by construction, to rounding.  The lateral ring
  outside the physical rectangle is a zero-gradient copy (``bc.F:550-562``'s
  open rule), so the outermost ring is not a closure point.
* ``velocity_gradient_decomposition``: the continuous identity
  ``curl(-(V.grad)V) = -(V.grad)w + (w.grad)V - w(div V)`` discretised with
  second-order centred differences at scalar points.  Its difference from the
  curl of CM1's own advection terms is published by value as ``advres``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as _np

#: ``adv_routines.F:6033-6035``: CM1 folds these in default REAL.
FLX6_A = _np.float32(37.0) / _np.float32(60.0)
FLX6_B = _np.float32(-8.0) / _np.float32(60.0)
FLX6_C = _np.float32(1.0) / _np.float32(60.0)
#: ``adv_routines.F`` ``flx4``: ``(7/12)*(s3+s2) - (1/12)*(s4+s1)``.
FLX4_A = _np.float32(7.0) / _np.float32(12.0)
FLX4_B = _np.float32(1.0) / _np.float32(12.0)


def _f(xp: Any, value: Any) -> Any:
    return xp.float32(value)


def flx6(xp: Any, s1: Any, s2: Any, s3: Any, s4: Any, s5: Any, s6: Any) -> Any:
    """``adv_routines.F:6026-6037``, left to right as gfortran evaluates it."""

    a = xp.multiply(_f(xp, FLX6_A), xp.add(s4, s3))
    b = xp.multiply(_f(xp, FLX6_B), xp.add(s5, s2))
    c = xp.multiply(_f(xp, FLX6_C), xp.add(s6, s1))
    return xp.add(xp.add(a, b), c)


def flx4(xp: Any, s1: Any, s2: Any, s3: Any, s4: Any) -> Any:
    a = xp.multiply(_f(xp, FLX4_A), xp.add(s3, s2))
    b = xp.multiply(_f(xp, FLX4_B), xp.add(s4, s1))
    return xp.subtract(a, b)


# ---------------------------------------------------------------------------
# Divergence split.  CM1's hadv/vadv expressions (adv.F:1555-1601, 2477-2521,
# 3379-3425, 665-702) each carry their own directional share of the mass-flux
# divergence.  GM1's fused advection kernels take one divergence array; handed
# the directional share instead, they evaluate exactly CM1's expression.
# ---------------------------------------------------------------------------

def horizontal_divergence(xp: Any, rru: Any, rrv: Any, *, ni: int, nj: int, nk: int,
                          ng: int, dx: float, dy: float) -> Any:
    """``(rru(i+1)-rru(i))*rdx*uh + (rrv(j+1)-rrv(j))*rdy*vh`` on ``(nk,nj,ni)``.

    ``uh = vh = 1`` on this lane's uniform horizontal grid (param.F's
    uniform branch), so the factor is omitted rather than multiplied by one.
    """

    rdx = xp.divide(_f(xp, 1.0), _f(xp, dx))
    rdy = xp.divide(_f(xp, 1.0), _f(xp, dy))
    k = slice(1, nk + 1)
    east = rru[k, ng:ng + nj, ng + 1:ng + ni + 1]
    west = rru[k, ng:ng + nj, ng:ng + ni]
    north = rrv[k, ng + 1:ng + nj + 1, ng:ng + ni]
    south = rrv[k, ng:ng + nj, ng:ng + ni]
    return xp.add(xp.multiply(xp.subtract(east, west), rdx),
                  xp.multiply(xp.subtract(north, south), rdy))


@dataclass(frozen=True)
class FlatMetricLevels:
    """A flat column's vertical metric, ``rdz*m(1,1,k)``, as CM1 applies it.

    ``terrain_flag=.false.`` multiplies a vertical difference by ``rdz`` and
    then by the column word -- ``((d*rdz)*mh(1,1,k))`` (adv.F:562-572,
    :790-900, :1640-1830, :2560-2750) and ``((-d*rdz)*mf(1,1,k))``
    (adv.F:3252-3262, :3441-3600) -- where the terrain branch multiplies once
    by ``rdsf(k)``/``rds(k)``.  ``levels`` holds the column words at the same
    ``k`` the ``rdsf``/``rds`` level arrays it replaces are indexed by.
    """

    rdz: Any
    levels: Any


def apply_vertical_metric(xp: Any, d: Any, levels: Any, shape: tuple[int, ...]) -> Any:
    """``d*rdsf(k)`` on the uniform/terrain column, ``((d*rdz)*m(k))`` on a flat one."""

    if isinstance(levels, FlatMetricLevels):
        return xp.multiply(xp.multiply(d, levels.rdz), levels.levels.reshape(shape))
    return xp.multiply(d, levels.reshape(shape))


def apply_vertical_metric_at(xp: Any, d: Any, levels: Any, k: int) -> Any:
    """:func:`apply_vertical_metric` for one level ``k``."""

    if isinstance(levels, FlatMetricLevels):
        return xp.multiply(xp.multiply(d, levels.rdz), levels.levels[k])
    return xp.multiply(d, levels[k])


def vertical_divergence(xp: Any, rrw: Any, rdsf_levels: Any, *, ni: int, nj: int,
                        nk: int, ng: int) -> Any:
    """``(rrw(k+1)-rrw(k))*rdsf(k)``; ``rdsf_levels`` holds levels ``1..nk``.

    On ``param.F:7789-7811``'s uniform column every entry is ``1/dz``, which is
    the reciprocal GM1's uniform divergence kernel multiplies by.  A
    :class:`FlatMetricLevels` applies ``((d*rdz)*mh(1,1,k))``
    (solve2.F:584-596).
    """

    upper = rrw[2:nk + 2, ng:ng + nj, ng:ng + ni]
    lower = rrw[1:nk + 1, ng:ng + nj, ng:ng + ni]
    return apply_vertical_metric(xp, xp.subtract(upper, lower), rdsf_levels, (nk, 1, 1))


# ---------------------------------------------------------------------------
# Non-diffusive (sixth-order) advection: the reference CM1 subtracts to form
# the implicit diffusion of its odd-order schemes.
# ---------------------------------------------------------------------------

def _pad_axis(xp: Any, a: Any, axis: int, before: int, after: int) -> Any:
    widths = [(0, 0)] * a.ndim
    widths[axis] = (before, after)
    return xp.pad(a, widths, mode="edge")


def _wrap_padded(xp: Any, a: Any, axis: int, n: int, ng: int, *, faces: bool) -> Any:
    """A copy of padded carrier ``a`` whose first halo plane each side is the periodic word.

    ``solve2.F:493-617`` forms the mass fluxes and ``divx`` in the halo from
    ``bc.F``'s periodic wind copies, so across a periodic side they are the
    words at the wrapped index -- which is what the fused kernels read.  A face
    carrier (``faces``) keeps its own boundary face ``n``; its halo faces are
    ``-1 -> n-1`` and ``n+1 -> 1``.  A cell carrier's are ``-1 -> n-1`` and
    ``n -> 0``.
    """

    out = xp.array(a, copy=True)
    moved = xp.moveaxis(out, axis, -1)
    moved[..., ng - 1] = moved[..., ng + n - 1]
    if faces:
        moved[..., ng + n + 1] = moved[..., ng + 1]
    else:
        moved[..., ng + n] = moved[..., ng]
    return out


def _pad_last(xp: Any, a: Any, axis: int, *, periodic: bool) -> Any:
    """One plane each side along ``axis``: wrapped when periodic, else a clamp."""

    return xp.pad(a, [(1, 1) if d == axis else (0, 0) for d in range(a.ndim)],
                  mode="wrap" if periodic else "edge")


def nondiffusive_u(xp: Any, *, u: Any, rru: Any, rrv: Any, rrw: Any, c2: Any,
                   divh: Any, divv: Any, scale: Any, rdsf_levels: Any,
                   ni: int, nj: int, nk: int, ng: int, dx: float, dy: float,
                   periodic_x: bool = False) -> tuple[Any, Any]:
    """``adv.F:1640-1830``'s ``ndiff`` for u, horizontal and vertical.

    ``u``/``rru``/``rrv``/``rrw`` are GM1's padded carriers; ``c2`` is the
    ``(nk+1, nj, ni)`` w-level metric; ``scale`` is the ``(nk, nj, ni+1)``
    factor GM1's ``cm1_mom5_advu_v1`` multiplies by (``gzu/(0.5*(rho0+rho0))``
    on the terrain-form family).  Faces ``i = i1..i2`` are filled, CM1's own
    range (adv.F:968-978): ``2..ni`` when the x sides are open or rigid, where
    the two boundary faces stay zero as the fused kernel leaves them, and
    ``1..ni+1`` when they are periodic (``periodic_x``), where the mass
    fluxes, ``divh``/``divv`` and ``c2`` are read across the side as the
    kernel reads them.
    """

    rdx, rdy = _f(xp, 1.0) / _f(xp, dx), _f(xp, 1.0) / _f(xp, dy)
    half = _f(xp, 0.5)
    ks = slice(1, nk + 1)
    f0, f1 = (0, ni) if periodic_x else (1, ni - 1)
    if periodic_x:
        rru = _wrap_padded(xp, rru, 2, ni, ng, faces=True)
        rrv = _wrap_padded(xp, rrv, 2, ni, ng, faces=False)
        rrw = _wrap_padded(xp, rrw, 2, ni, ng, faces=False)
    # The kernel's met3 clamps c2's lateral halo to the edge column; divx is
    # read across a periodic side at the wrapped cell.
    divh = _pad_last(xp, divh, 2, periodic=periodic_x)
    divv = _pad_last(xp, divv, 2, periodic=periodic_x)
    c2 = _pad_last(xp, c2, 2, periodic=False)
    # hadv_flx6 stag=2: dumx at scalar cells c=f0-1..f1 (between faces c, c+1).
    col = lambda o: slice(ng + f0 - 1 + o, ng + f1 + 1 + o)  # noqa: E731 - index helper
    ubar = xp.multiply(half, xp.add(rru[ks, ng:ng + nj, col(0)], rru[ks, ng:ng + nj, col(1)]))
    uu = u[ks, ng:ng + nj]
    dumx = xp.multiply(ubar, flx6(xp, uu[..., col(-2)], uu[..., col(-1)], uu[..., col(0)],
                                  uu[..., col(1)], uu[..., col(2)], uu[..., col(3)]))
    # dumy at row faces g=0..nj over faces f=f0..f1.
    fs = slice(ng + f0, ng + f1 + 1)
    rows = lambda o: slice(ng + o, ng + o + nj + 1)  # noqa: E731
    vbar = xp.multiply(half, xp.add(rrv[ks, rows(0), ng + f0:ng + f1 + 1], rrv[ks, rows(0), ng + f0 - 1:ng + f1]))
    uc = u[ks, :, fs]
    dumy = xp.multiply(vbar, flx6(xp, uc[:, rows(-3)], uc[:, rows(-2)], uc[:, rows(-1)],
                                  uc[:, rows(0)], uc[:, rows(1)], uc[:, rows(2)]))
    advx = xp.multiply(xp.negative(xp.subtract(dumx[..., 1:], dumx[..., :-1])), rdx)
    advy = xp.multiply(xp.negative(xp.subtract(dumy[:, 1:], dumy[:, :-1])), rdy)
    face = u[ks, ng:ng + nj, fs]
    dh = xp.multiply(half, xp.add(divh[..., f0 + 1:f1 + 2], divh[..., f0:f1 + 1]))
    dv = xp.multiply(half, xp.add(divv[..., f0 + 1:f1 + 2], divv[..., f0:f1 + 1]))
    sc = scale[..., f0:f1 + 1]
    h = xp.zeros((nk, nj, ni + 1), dtype=xp.float32)
    v = xp.zeros((nk, nj, ni + 1), dtype=xp.float32)
    h[..., f0:f1 + 1] = xp.multiply(xp.add(xp.add(advx, advy), xp.multiply(face, dh)), sc)
    # vadv_flx6 stag=2 on w levels k=2..nk, face columns f=f0..f1.
    wbar = xp.multiply(half, xp.add(rrw[:, ng:ng + nj, ng + f0:ng + f1 + 1], rrw[:, ng:ng + nj, ng + f0 - 1:ng + f1]))
    cc2 = xp.multiply(half, xp.add(c2[:, :, f0:f1 + 1], c2[:, :, f0 + 1:f1 + 2]))
    dumz = _vertical_faces_stag23(xp, u[:, ng:ng + nj, fs], wbar, cc2, nk)
    advz = apply_vertical_metric(xp, xp.negative(xp.subtract(dumz[2:nk + 2], dumz[1:nk + 1])),
                                 rdsf_levels, (nk, 1, 1))
    v[..., f0:f1 + 1] = xp.multiply(xp.add(advz, xp.multiply(face, dv)), sc)
    return h, v


def nondiffusive_v(xp: Any, *, v: Any, rru: Any, rrv: Any, rrw: Any, c2: Any,
                   divh: Any, divv: Any, scale: Any, rdsf_levels: Any,
                   ni: int, nj: int, nk: int, ng: int, dx: float, dy: float,
                   periodic_y: bool = False) -> tuple[Any, Any]:
    """``adv.F:2560-2750``'s ``ndiff`` for v on rows ``j = j1..j2``.

    ``2..nj`` when the y sides are open or rigid; ``1..nj+1`` when they are
    periodic (``periodic_y``), read across the side as :func:`nondiffusive_u`
    reads x (adv.F:1913-1923).
    """

    rdx, rdy = _f(xp, 1.0) / _f(xp, dx), _f(xp, 1.0) / _f(xp, dy)
    half = _f(xp, 0.5)
    ks = slice(1, nk + 1)
    g0, g1 = (0, nj) if periodic_y else (1, nj - 1)
    if periodic_y:
        rru = _wrap_padded(xp, rru, 1, nj, ng, faces=False)
        rrv = _wrap_padded(xp, rrv, 1, nj, ng, faces=True)
        rrw = _wrap_padded(xp, rrw, 1, nj, ng, faces=False)
    divh = _pad_last(xp, divh, 1, periodic=periodic_y)
    divv = _pad_last(xp, divv, 1, periodic=periodic_y)
    c2 = _pad_last(xp, c2, 1, periodic=False)
    rs = slice(ng + g0, ng + g1 + 1)  # v rows g=g0..g1
    # hadv_flx6 stag=3: dumx at u faces i=0..ni over those v rows.
    fcols = lambda o: slice(ng + o, ng + o + ni + 1)  # noqa: E731
    ubar = xp.multiply(half, xp.add(rru[ks, ng + g0:ng + g1 + 1, fcols(0)], rru[ks, ng + g0 - 1:ng + g1, fcols(0)]))
    vr = v[ks, rs]
    dumx = xp.multiply(ubar, flx6(xp, vr[..., fcols(-3)], vr[..., fcols(-2)], vr[..., fcols(-1)],
                                  vr[..., fcols(0)], vr[..., fcols(1)], vr[..., fcols(2)]))
    # dumy at scalar cells c=g0-1..g1 (between v rows c, c+1).
    cells = lambda o: slice(ng + g0 - 1 + o, ng + g1 + 1 + o)  # noqa: E731
    vbar = xp.multiply(half, xp.add(rrv[ks, cells(0), ng:ng + ni], rrv[ks, cells(1), ng:ng + ni]))
    vc = v[ks, :, ng:ng + ni]
    dumy = xp.multiply(vbar, flx6(xp, vc[:, cells(-2)], vc[:, cells(-1)], vc[:, cells(0)],
                                  vc[:, cells(1)], vc[:, cells(2)], vc[:, cells(3)]))
    advx = xp.multiply(xp.negative(xp.subtract(dumx[..., 1:], dumx[..., :-1])), rdx)
    advy = xp.multiply(xp.negative(xp.subtract(dumy[:, 1:], dumy[:, :-1])), rdy)
    row = v[ks, rs, ng:ng + ni]
    dh = xp.multiply(half, xp.add(divh[:, g0 + 1:g1 + 2], divh[:, g0:g1 + 1]))
    dv = xp.multiply(half, xp.add(divv[:, g0 + 1:g1 + 2], divv[:, g0:g1 + 1]))
    sc = scale[:, g0:g1 + 1]
    h = xp.zeros((nk, nj + 1, ni), dtype=xp.float32)
    out_v = xp.zeros((nk, nj + 1, ni), dtype=xp.float32)
    h[:, g0:g1 + 1] = xp.multiply(xp.add(xp.add(advx, advy), xp.multiply(row, dh)), sc)
    wbar = xp.multiply(half, xp.add(rrw[:, ng + g0:ng + g1 + 1, ng:ng + ni], rrw[:, ng + g0 - 1:ng + g1, ng:ng + ni]))
    cc2 = xp.multiply(half, xp.add(c2[:, g0:g1 + 1, :], c2[:, g0 + 1:g1 + 2, :]))
    dumz = _vertical_faces_stag23(xp, v[:, rs, ng:ng + ni], wbar, cc2, nk)
    advz = apply_vertical_metric(xp, xp.negative(xp.subtract(dumz[2:nk + 2], dumz[1:nk + 1])),
                                 rdsf_levels, (nk, 1, 1))
    out_v[:, g0:g1 + 1] = xp.multiply(xp.add(advz, xp.multiply(row, dv)), sc)
    return h, out_v


def _vertical_faces_stag23(xp: Any, a: Any, wbar: Any, cc2: Any, nk: int) -> Any:
    """``vadv_flx6``/``vadv_lwr_flx_4``/``vadv_lwr_flx_2`` for stag 2 and 3.

    ``a`` is the staggered wind with its ``k=0..nk+1`` ghosts; ``wbar`` the
    ``(nk+3, ...)`` averaged vertical mass flux; ``cc2`` the ``(nk+1, ...)``
    averaged ``c2`` indexed by ``k-1``.  Returns ``dumz`` indexed by w level
    ``k=0..nk+1`` with ``dumz(1) = dumz(nk+1) = 0`` (adv.F:1747-1752).
    """

    shape = (nk + 2, *a.shape[1:])
    dumz = xp.zeros(shape, dtype=xp.float32)
    one = _f(xp, 1.0)
    for k in range(2, nk + 1):
        if k in (2, nk):
            c = cc2[k - 1]
            dumz[k] = xp.multiply(wbar[k], xp.add(xp.multiply(xp.subtract(one, c), a[k - 1]),
                                                  xp.multiply(c, a[k])))
        elif k in (3, nk - 1):
            dumz[k] = xp.multiply(wbar[k], flx4(xp, a[k - 2], a[k - 1], a[k], a[k + 1]))
        else:
            dumz[k] = xp.multiply(wbar[k], flx6(xp, a[k - 3], a[k - 2], a[k - 1], a[k],
                                                a[k + 1], a[k + 2]))
    return dumz


def nondiffusive_w(xp: Any, *, w: Any, rru: Any, rrv: Any, rrw: Any, c1: Any, c2: Any,
                   divh: Any, divv: Any, scale: Any, rds_levels: Any,
                   ni: int, nj: int, nk: int, ng: int, dx: float, dy: float) -> tuple[Any, Any]:
    """``adv.F:3441-3600``'s ``ndiff`` for w on levels ``k=2..nk``.

    ``c1``/``c2`` are the ``(nk+1, nj, ni)`` w-level metrics; ``scale`` is the
    ``(nk+1, nj, ni)`` ``rrf0*gz`` factor of ``cm1_mom5_advw_v1``;
    ``rds_levels`` is indexed by w level ``0..nk``.
    """

    rdx, rdy = _f(xp, 1.0) / _f(xp, dx), _f(xp, 1.0) / _f(xp, dy)
    half, one = _f(xp, 0.5), _f(xp, 1.0)
    h = xp.zeros((nk + 1, nj, ni), dtype=xp.float32)
    out_v = xp.zeros((nk + 1, nj, ni), dtype=xp.float32)
    # CM1 clamps no index here; GM1's metric reads clamp the lateral c2 halo
    # to the edge column, which is CM1's own flat-grid c2 halo value.
    c2p = _pad_axis(xp, _pad_axis(xp, c2, 2, 1, 1), 1, 1, 1)
    for kf in range(2, nk + 1):
        c2k = c2[kf - 1]
        c1k = xp.subtract(one, c2k)
        # hadv_flx6 stag=4, u faces f=0..ni.
        cc2x = xp.multiply(half, xp.add(c2p[kf - 1, 1:nj + 1, 0:ni + 1], c2p[kf - 1, 1:nj + 1, 1:ni + 2]))
        ubar = xp.add(xp.multiply(cc2x, rru[kf, ng:ng + nj, ng:ng + ni + 1]),
                      xp.multiply(xp.subtract(one, cc2x), rru[kf - 1, ng:ng + nj, ng:ng + ni + 1]))
        wr = w[kf, ng:ng + nj]
        fc = lambda o: slice(ng + o, ng + o + ni + 1)  # noqa: E731
        dumx = xp.multiply(ubar, flx6(xp, wr[:, fc(-3)], wr[:, fc(-2)], wr[:, fc(-1)],
                                      wr[:, fc(0)], wr[:, fc(1)], wr[:, fc(2)]))
        cc2y = xp.multiply(half, xp.add(c2p[kf - 1, 0:nj + 1, 1:ni + 1], c2p[kf - 1, 1:nj + 2, 1:ni + 1]))
        vbar = xp.add(xp.multiply(cc2y, rrv[kf, ng:ng + nj + 1, ng:ng + ni]),
                      xp.multiply(xp.subtract(one, cc2y), rrv[kf - 1, ng:ng + nj + 1, ng:ng + ni]))
        wc = w[kf, :, ng:ng + ni]
        fr = lambda o: slice(ng + o, ng + o + nj + 1)  # noqa: E731
        dumy = xp.multiply(vbar, flx6(xp, wc[fr(-3)], wc[fr(-2)], wc[fr(-1)],
                                      wc[fr(0)], wc[fr(1)], wc[fr(2)]))
        advx = xp.multiply(xp.negative(xp.subtract(dumx[:, 1:], dumx[:, :-1])), rdx)
        advy = xp.multiply(xp.negative(xp.subtract(dumy[1:], dumy[:-1])), rdy)
        wk = w[kf, ng:ng + nj, ng:ng + ni]
        dterm = xp.add(xp.multiply(c2k, divh[kf - 1]), xp.multiply(c1k, divh[kf - 2]))
        h[kf] = xp.multiply(xp.add(xp.add(advx, advy), xp.multiply(wk, dterm)), scale[kf - 1])
    # vadv_flx6 stag=4: dumz at scalar levels k=1..nk.
    wcol = w[:, ng:ng + nj, ng:ng + ni]
    dumz = xp.zeros((nk + 1, nj, ni), dtype=xp.float32)
    for k in range(1, nk + 1):
        wbar = xp.multiply(half, xp.add(rrw[k, ng:ng + nj, ng:ng + ni], rrw[k + 1, ng:ng + nj, ng:ng + ni]))
        if k in (1, nk):
            dumz[k] = xp.multiply(xp.multiply(wbar, half), xp.add(wcol[k], wcol[k + 1]))
        elif k in (2, nk - 1):
            dumz[k] = xp.multiply(wbar, flx4(xp, wcol[k - 1], wcol[k], wcol[k + 1], wcol[k + 2]))
        else:
            dumz[k] = xp.multiply(wbar, flx6(xp, wcol[k - 2], wcol[k - 1], wcol[k], wcol[k + 1],
                                             wcol[k + 2], wcol[k + 3]))
    for kf in range(2, nk + 1):
        c2k = c2[kf - 1]
        c1k = xp.subtract(one, c2k)
        advz = apply_vertical_metric_at(xp, xp.negative(xp.subtract(dumz[kf], dumz[kf - 1])), rds_levels, kf)
        wk = wcol[kf]
        dterm = xp.add(xp.multiply(c2k, divv[kf - 1]), xp.multiply(c1k, divv[kf - 2]))
        out_v[kf] = xp.multiply(xp.add(advz, xp.multiply(wk, dterm)), scale[kf - 1])
    return h, out_v


def nondiffusive_scalar(xp: Any, *, s: Any, rru: Any, rrv: Any, rrw: Any, c1: Any, c2: Any,
                        divh: Any, divv: Any, rr0: Any, gz: Any, rdsf_levels: Any,
                        ni: int, nj: int, nk: int, ng: int, dx: float, dy: float) -> tuple[Any, Any]:
    """``adv.F:790-900``'s scalar ``ndiff`` (``hadv_flx6``/``vadv_flx6`` stag 1).

    ``s`` is the padded scalar the stage advected, its halo filled by the
    run's own ``bcs``.  CM1 forms ``ndiff`` on every cell i=1..ni, j=1..nj
    (adv.F:785-797) from the unedited flx6 fluxes, and so does this: the
    whole-domain scalar tendency it is differenced against covers the same
    cells.
    """

    rdx, rdy = _f(xp, 1.0) / _f(xp, dx), _f(xp, 1.0) / _f(xp, dy)
    ks = slice(1, nk + 1)
    fc = lambda o: slice(ng + o, ng + o + ni + 1)  # noqa: E731
    sr = s[ks, ng:ng + nj]
    dumx = xp.multiply(rru[ks, ng:ng + nj, ng:ng + ni + 1],
                       flx6(xp, sr[..., fc(-3)], sr[..., fc(-2)], sr[..., fc(-1)],
                            sr[..., fc(0)], sr[..., fc(1)], sr[..., fc(2)]))
    fr = lambda o: slice(ng + o, ng + o + nj + 1)  # noqa: E731
    sc = s[ks, :, ng:ng + ni]
    dumy = xp.multiply(rrv[ks, ng:ng + nj + 1, ng:ng + ni],
                       flx6(xp, sc[:, fr(-3)], sc[:, fr(-2)], sc[:, fr(-1)],
                            sc[:, fr(0)], sc[:, fr(1)], sc[:, fr(2)]))
    advx = xp.multiply(xp.negative(xp.subtract(dumx[..., 1:], dumx[..., :-1])), rdx)
    advy = xp.multiply(xp.negative(xp.subtract(dumy[:, 1:], dumy[:, :-1])), rdy)
    cell = s[ks, ng:ng + nj, ng:ng + ni]
    gz2 = gz.reshape(1, nj, ni)
    hnd = xp.multiply(xp.multiply(xp.add(xp.add(advx, advy), xp.multiply(cell, divh)), rr0), gz2)
    col = s[:, ng:ng + nj, ng:ng + ni]
    flux = rrw[:, ng:ng + nj, ng:ng + ni]
    dumz = xp.zeros((nk + 2, nj, ni), dtype=xp.float32)
    for k in range(2, nk + 1):
        if k in (2, nk):
            dumz[k] = xp.multiply(flux[k], xp.add(xp.multiply(c1[k - 1], col[k - 1]),
                                                  xp.multiply(c2[k - 1], col[k])))
        elif k in (3, nk - 1):
            dumz[k] = xp.multiply(flux[k], flx4(xp, col[k - 2], col[k - 1], col[k], col[k + 1]))
        else:
            dumz[k] = xp.multiply(flux[k], flx6(xp, col[k - 3], col[k - 2], col[k - 1], col[k],
                                                col[k + 1], col[k + 2]))
    advz = apply_vertical_metric(xp, xp.negative(xp.subtract(dumz[2:nk + 2], dumz[1:nk + 1])),
                                 rdsf_levels, (nk, 1, 1))
    vnd = xp.multiply(xp.multiply(xp.add(advz, xp.multiply(cell, divv)), rr0), gz2)
    return hnd, vnd


def implicit_diffusion_split(xp: Any, total: Any, nondiffusive: Any) -> tuple[Any, Any]:
    """``*_hadv = ndiff``, ``*_hidiff = tot - ndiff`` (``adv.F:1679-1689``)."""

    return nondiffusive, xp.subtract(total, nondiffusive)


def pressure_gradient_residual(xp: Any, current: Any, baseline: Any, saved: Any,
                               dt: float) -> Any:
    """``solve2.F:1196-1245``: ``(u3d - ua)*rdt - udiag(ud_pgrad)``."""

    rdt = xp.divide(_f(xp, 1.0), _f(xp, dt))
    return xp.subtract(xp.multiply(xp.subtract(current, baseline), rdt), saved)


def process_difference(xp: Any, after: Any, before: Any, dt: float) -> Any:
    """``mp_driver.F:1670-1690``: ``(th3d - saved)*rdt``."""

    rdt = xp.divide(_f(xp, 1.0), _f(xp, dt))
    return xp.multiply(xp.subtract(after, before), rdt)


# ---------------------------------------------------------------------------
# calcvort, misclibs.F:1618-2056, no-terrain Cartesian branch with mf.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VorticityGrid:
    ni: int
    nj: int
    nk: int
    dx: float
    dy: float
    dz: float
    #: ``mf`` at w levels ``1..nk+1`` (``(nk+1,)``); all ones on a uniform column.
    mf: Any
    bbc: int = 1
    tbc: int = 1


def _lateral_pad(xp: Any, a: Any, axis: int) -> Any:
    return _pad_axis(xp, a, axis, 1, 1)


def calcvort_x(xp: Any, v: Any, w: Any, grid: VorticityGrid, *, vmove: float = 0.0) -> Any:
    """``misclibs.F:1618-1683``: xvort at scalar points from physical v and w.

    ``v`` is ``(nk, nj+1, ni)`` and ``w`` is ``(nk+1, nj, ni)``.  Pass a
    tendency and ``vmove=0`` to take the curl of a budget term.
    """

    nk, nj, ni = grid.nk, grid.nj, grid.ni
    rdy = xp.divide(_f(xp, 1.0), _f(xp, grid.dy))
    rdz = xp.divide(_f(xp, 1.0), _f(xp, grid.dz))
    mf = grid.mf.reshape(nk + 1, 1, 1)
    half = _f(xp, 0.5)
    wp = _lateral_pad(xp, w, 1)  # rows -1..nj
    tem = xp.zeros((nk + 1, nj + 1, ni), dtype=xp.float32)
    mfav = xp.multiply(half, xp.add(mf[1:nk], mf[1:nk]))
    dwdy = xp.multiply(xp.subtract(wp[1:nk, 1:nj + 2], wp[1:nk, 0:nj + 1]), rdy)
    dvdz = xp.multiply(xp.multiply(xp.subtract(v[1:nk], v[0:nk - 1]), rdz), mfav)
    tem[1:nk] = xp.subtract(dwdy, dvdz)
    two = _f(xp, 2.0)
    if grid.bbc == 1:
        tem[0] = tem[1]
    elif grid.bbc == 2:
        m0 = xp.multiply(half, xp.add(mf[0], mf[0]))
        tem[0] = xp.multiply(xp.multiply(xp.multiply(-two, xp.add(v[0], _f(xp, vmove))), rdz), m0)
    if grid.tbc == 1:
        tem[nk] = tem[nk - 1]
    elif grid.tbc == 2:
        mt = xp.multiply(half, xp.add(mf[nk], mf[nk]))
        tem[nk] = xp.multiply(xp.multiply(xp.multiply(two, v[nk - 1]), rdz), mt)
    quarter = _f(xp, 0.25)
    s = xp.add(xp.add(xp.add(tem[0:nk, 0:nj], tem[0:nk, 1:nj + 1]), tem[1:nk + 1, 0:nj]),
               tem[1:nk + 1, 1:nj + 1])
    return xp.multiply(quarter, s)


def calcvort_y(xp: Any, u: Any, w: Any, grid: VorticityGrid, *, umove: float = 0.0) -> Any:
    """``misclibs.F:1854-1925``: yvort at scalar points from physical u and w."""

    nk, nj, ni = grid.nk, grid.nj, grid.ni
    rdx = xp.divide(_f(xp, 1.0), _f(xp, grid.dx))
    rdz = xp.divide(_f(xp, 1.0), _f(xp, grid.dz))
    mf = grid.mf.reshape(nk + 1, 1, 1)
    half = _f(xp, 0.5)
    wp = _lateral_pad(xp, w, 2)  # columns -1..ni
    tem = xp.zeros((nk + 1, nj, ni + 1), dtype=xp.float32)
    mfav = xp.multiply(half, xp.add(mf[1:nk], mf[1:nk]))
    dudz = xp.multiply(xp.multiply(xp.subtract(u[1:nk], u[0:nk - 1]), rdz), mfav)
    dwdx = xp.multiply(xp.subtract(wp[1:nk, :, 1:ni + 2], wp[1:nk, :, 0:ni + 1]), rdx)
    tem[1:nk] = xp.subtract(dudz, dwdx)
    two = _f(xp, 2.0)
    if grid.bbc == 1:
        tem[0] = tem[1]
    elif grid.bbc == 2:
        m0 = xp.multiply(half, xp.add(mf[0], mf[0]))
        tem[0] = xp.multiply(xp.multiply(xp.multiply(two, xp.add(u[0], _f(xp, umove))), rdz), m0)
    if grid.tbc == 1:
        tem[nk] = tem[nk - 1]
    elif grid.tbc == 2:
        mt = xp.multiply(half, xp.add(mf[nk], mf[nk]))
        tem[nk] = xp.multiply(xp.multiply(xp.multiply(-two, u[nk - 1]), rdz), mt)
    quarter = _f(xp, 0.25)
    s = xp.add(xp.add(xp.add(tem[0:nk, :, 0:ni], tem[0:nk, :, 1:ni + 1]), tem[1:nk + 1, :, 0:ni]),
               tem[1:nk + 1, :, 1:ni + 1])
    return xp.multiply(quarter, s)


def calcvort_z(xp: Any, u: Any, v: Any, grid: VorticityGrid) -> Any:
    """``misclibs.F:2035-2056``: zvort, the corner value averaged to scalar points."""

    nk, nj, ni = grid.nk, grid.nj, grid.ni
    rdx = xp.divide(_f(xp, 1.0), _f(xp, grid.dx))
    rdy = xp.divide(_f(xp, 1.0), _f(xp, grid.dy))
    vp = _lateral_pad(xp, v, 2)  # columns -1..ni
    up = _lateral_pad(xp, u, 1)  # rows -1..nj
    dvdx = xp.multiply(xp.subtract(vp[:, :, 1:ni + 2], vp[:, :, 0:ni + 1]), rdx)
    dudy = xp.multiply(xp.subtract(up[:, 1:nj + 2], up[:, 0:nj + 1]), rdy)
    tem = xp.subtract(dvdx, dudy)
    quarter = _f(xp, 0.25)
    s = xp.add(xp.add(xp.add(tem[:, 0:nj, 0:ni], tem[:, 0:nj, 1:ni + 1]), tem[:, 1:nj + 1, 0:ni]),
               tem[:, 1:nj + 1, 1:ni + 1])
    return xp.multiply(quarter, s)


def curl_budget_term(xp: Any, tu: Any, tv: Any, tw: Any, grid: VorticityGrid) -> tuple[Any, Any, Any]:
    """The three vorticity tendencies of one momentum budget term (GM1 extension).

    Any of ``tu``/``tv``/``tw`` may be ``None`` for a term CM1 does not carry
    on that component (``wb_buoy`` has no u or v part); its contribution is
    exactly zero.
    """

    nk, nj, ni = grid.nk, grid.nj, grid.ni
    zu = xp.zeros((nk, nj, ni + 1), dtype=xp.float32) if tu is None else tu
    zv = xp.zeros((nk, nj + 1, ni), dtype=xp.float32) if tv is None else tv
    zw = xp.zeros((nk + 1, nj, ni), dtype=xp.float32) if tw is None else tw
    return calcvort_x(xp, zv, zw, grid), calcvort_y(xp, zu, zw, grid), calcvort_z(xp, zu, zv, grid)


# ---------------------------------------------------------------------------
# Velocity-gradient decomposition at scalar points (GM1 extension).
# ---------------------------------------------------------------------------

def _centred(xp: Any, a: Any, axis: int, spacing: Any) -> Any:
    """Centred difference over two cells, one-sided on the two outer cells."""

    n = a.shape[axis]
    out = xp.empty_like(a)

    def take(lo: int, hi: int) -> Any:
        index = [slice(None)] * a.ndim
        index[axis] = slice(lo, hi)
        return a[tuple(index)]

    def put(lo: int, hi: int, value: Any) -> None:
        index = [slice(None)] * a.ndim
        index[axis] = slice(lo, hi)
        out[tuple(index)] = value

    two = _f(xp, 2.0)
    put(1, n - 1, xp.divide(xp.subtract(take(2, n), take(0, n - 2)), xp.multiply(two, spacing)))
    put(0, 1, xp.divide(xp.subtract(take(1, 2), take(0, 1)), spacing))
    put(n - 1, n, xp.divide(xp.subtract(take(n - 1, n), take(n - 2, n - 1)), spacing))
    return out


def _vertical_centred(xp: Any, a: Any, zh: Any) -> Any:
    """``d/dz`` at scalar levels on a possibly stretched column ``zh`` (``(nk,)``)."""

    nk = a.shape[0]
    out = xp.empty_like(a)
    out[1:nk - 1] = xp.divide(xp.subtract(a[2:nk], a[0:nk - 2]),
                              xp.subtract(zh[2:nk], zh[0:nk - 2]).reshape(nk - 2, 1, 1))
    out[0] = xp.divide(xp.subtract(a[1], a[0]), xp.subtract(zh[1], zh[0]))
    out[nk - 1] = xp.divide(xp.subtract(a[nk - 1], a[nk - 2]), xp.subtract(zh[nk - 1], zh[nk - 2]))
    return out


def velocity_gradient_decomposition(
    xp: Any, *, u: Any, v: Any, w: Any, xvort: Any, yvort: Any, zvort: Any,
    zh: Any, zf: Any, dx: float, dy: float,
) -> dict[str, Any]:
    """Stretching, tilting, divergence and advection of vorticity at scalar points.

    ``u``/``v``/``w`` are the physical staggered winds the advection used;
    ``xvort``/``yvort``/``zvort`` are ``calcvort`` of the same winds; ``zh``
    (``(nk,)``) and ``zf`` (``(nk+1,)``) are the column heights.  Returns the
    four terms for each axis, keyed ``stretch_x`` .. ``advect_z``.
    """

    half = _f(xp, 0.5)
    nk = u.shape[0]
    us = xp.multiply(half, xp.add(u[..., :-1], u[..., 1:]))
    vs = xp.multiply(half, xp.add(v[:, :-1], v[:, 1:]))
    ws = xp.multiply(half, xp.add(w[:-1], w[1:]))
    dxf, dyf = _f(xp, dx), _f(xp, dy)
    dzc = xp.subtract(zf[1:nk + 1], zf[0:nk]).reshape(nk, 1, 1)
    dudx = xp.divide(xp.subtract(u[..., 1:], u[..., :-1]), dxf)
    dvdy = xp.divide(xp.subtract(v[:, 1:], v[:, :-1]), dyf)
    dwdz = xp.divide(xp.subtract(w[1:], w[:-1]), dzc)
    grads = {
        "dudy": _centred(xp, us, 1, dyf), "dudz": _vertical_centred(xp, us, zh),
        "dvdx": _centred(xp, vs, 2, dxf), "dvdz": _vertical_centred(xp, vs, zh),
        "dwdx": _centred(xp, ws, 2, dxf), "dwdy": _centred(xp, ws, 1, dyf),
    }
    divergence = xp.add(xp.add(dudx, dvdy), dwdz)
    out: dict[str, Any] = {}
    out["stretch_x"] = xp.multiply(xvort, dudx)
    out["tilt_x"] = xp.add(xp.multiply(yvort, grads["dudy"]), xp.multiply(zvort, grads["dudz"]))
    out["stretch_y"] = xp.multiply(yvort, dvdy)
    out["tilt_y"] = xp.add(xp.multiply(xvort, grads["dvdx"]), xp.multiply(zvort, grads["dvdz"]))
    out["stretch_z"] = xp.multiply(zvort, dwdz)
    out["tilt_z"] = xp.add(xp.multiply(xvort, grads["dwdx"]), xp.multiply(yvort, grads["dwdy"]))
    for axis, omega in (("x", xvort), ("y", yvort), ("z", zvort)):
        out[f"divg_{axis}"] = xp.negative(xp.multiply(omega, divergence))
        adv = xp.add(xp.add(xp.multiply(us, _centred(xp, omega, 2, dxf)),
                            xp.multiply(vs, _centred(xp, omega, 1, dyf))),
                     xp.multiply(ws, _vertical_centred(xp, omega, zh)))
        out[f"advect_{axis}"] = xp.negative(adv)
    return out


# ---------------------------------------------------------------------------
# Write-set guard.  A budget launch may write only arrays the budget owns.
# ---------------------------------------------------------------------------

_SIGNATURE = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(([^)]*)\)', re.S)


@dataclass(frozen=True)
class KernelParameter:
    name: str
    base: str
    pointer: bool
    const: bool


def kernel_signatures(source: str) -> dict[str, tuple[KernelParameter, ...]]:
    """Every ``__global__`` entry point of one kernel source, typed."""

    out: dict[str, tuple[KernelParameter, ...]] = {}
    for name, body in _SIGNATURE.findall(source):
        params: list[KernelParameter] = []
        for raw in body.split(","):
            text = " ".join(raw.replace("__restrict__", " ").split())
            if not text:
                continue
            pointer = "*" in text
            const = text.startswith("const ")
            words = text.replace("const ", "").replace("*", " * ").split()
            params.append(KernelParameter(
                words[-1], " ".join(w for w in words[:-1] if w != "*"), pointer, const,
            ))
        out[name] = tuple(params)
    return out


def writable_parameter_positions(source: str) -> dict[str, tuple[int, ...]]:
    """For every ``__global__`` entry point, the non-``const`` pointer slots."""

    return {
        name: tuple(i for i, p in enumerate(params) if p.pointer and not p.const)
        for name, params in kernel_signatures(source).items()
    }


_SCALAR_TYPES: Mapping[str, str] = {
    "unsigned int": "uint32", "unsigned": "uint32", "int": "int32",
    "float": "float32", "double": "float64", "unsigned long long": "uint64",
}


class BudgetWriteGuard:
    """Refuse any budget launch whose writable slot is not a budget-owned array.

    This is what makes "budgets on" unable to change a simulated word: the
    guard runs on the device path too, before the kernel is enqueued, and it
    compares object identity against the arrays the budget stage allocated.
    A launch binds its arguments by the kernel's own parameter *names*, read
    from the packaged source, so a replay that substitutes a zeroed input
    names what it substitutes and cannot silently shift a positional slot.
    """

    def __init__(self, xp: Any) -> None:
        self.xp = xp
        self._owned: dict[int, Any] = {}
        self._signatures: dict[tuple[str, str], tuple[KernelParameter, ...]] = {}
        self.launches = 0

    def own(self, array: Any) -> Any:
        self._owned[id(array)] = array
        return array

    def owns(self, array: Any) -> bool:
        return self._owned.get(id(array)) is array

    def register_source(self, family: str, source: str) -> None:
        for name, params in kernel_signatures(source).items():
            self._signatures[(family, name)] = params

    def signature(self, family: str, name: str) -> tuple[KernelParameter, ...]:
        params = self._signatures.get((family, name))
        if params is None:
            raise RuntimeError(f"budget launch {family}:{name} has no registered signature")
        return params

    def call(self, family: str, name: str, kernel: Any, grid: Sequence[int],
             block: Sequence[int], values: Mapping[str, Any], *, stream: Any = None) -> None:
        params = self.signature(family, name)
        missing = [p.name for p in params if p.name not in values]
        extra = sorted(set(values) - {p.name for p in params})
        if missing or extra:
            raise RuntimeError(
                f"budget launch {name}: missing arguments {missing}, unknown {extra}"
            )
        bound = getattr(kernel, "name", None)
        if isinstance(bound, str) and bound != name:
            raise RuntimeError(f"budget launch named {name} but the kernel is {bound}")
        args: list[Any] = []
        for p in params:
            value = values[p.name]
            if p.pointer:
                if not p.const and not self.owns(value):
                    raise RuntimeError(
                        f"budget launch {name} would write {p.name}, which is not a "
                        "budget-owned array; a diagnostic must never write a model "
                        "carrier"
                    )
                args.append(value)
            else:
                args.append(getattr(self.xp, _SCALAR_TYPES[p.base])(value))
        kernel(tuple(grid), tuple(block), tuple(args), stream=stream)
        self.launches += 1

    def call_positional(self, kernel: Any, grid: Sequence[int], block: Sequence[int],
                        args: Sequence[Any], *, writes: Sequence[int], stream: Any = None) -> None:
        """For a lateral-family fill whose bound kernel name varies by selection.

        The caller names the slots the kernel writes; every one must be owned.
        """

        for index in writes:
            if not self.owns(args[index]):
                raise RuntimeError(
                    "budget boundary fill would write an array the budget does not own"
                )
        kernel(tuple(grid), tuple(block), tuple(args), stream=stream)
        self.launches += 1


__all__ = [
    "BudgetWriteGuard", "FlatMetricLevels", "apply_vertical_metric", "apply_vertical_metric_at", "FLX4_A", "FLX4_B", "FLX6_A", "FLX6_B", "FLX6_C",
    "KernelParameter", "VorticityGrid", "calcvort_x", "calcvort_y", "calcvort_z",
    "curl_budget_term", "kernel_signatures",
    "flx4", "flx6", "horizontal_divergence", "implicit_diffusion_split",
    "nondiffusive_scalar", "nondiffusive_u", "nondiffusive_v", "nondiffusive_w",
    "pressure_gradient_residual", "process_difference", "velocity_gradient_decomposition",
    "vertical_divergence", "writable_parameter_positions",
]
