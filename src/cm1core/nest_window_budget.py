"""The vorticity budget along a trajectory, from a child-window stream.  OFFLINE.

This is the second half of the answer the trajectories are for.  A trajectory
says where a parcel came from; the budget along it says **what made its
rotation**, term by term, and whether those terms add up to the rotation the
parcel actually gained.

WHAT IS CM1'S, WHAT IS GM1'S
----------------------------
The three vorticity components are CM1's ``src/misclibs.F:1542`` ``calcvort``,
flat-Cartesian branch (``xvort`` :1618-1683, ``yvort`` :1854-1900, ``zvort``
:2035-2056), transcribed once in :func:`cm1core.vorticity_budget.cpu_vorticity_budget`
and used here unchanged.

The **budget** is not CM1's: ``grep -rn "tilt" src/`` over the r21.1 tree
returns nothing, and ``src/cm1core/vorticity_budget.py`` says so already.  It
is the standard analysis of every tornadogenesis paper.  What this module adds
over the Eulerian frame published by ``[lagrangian].vorticity_budget`` is that
the terms are evaluated **on the window stream's own records** -- one per child
step -- and sampled at the moving parcel with ``parcel.F``'s own scalar-point
stencil (``src/parcel.F:1365-1447``, ``rx = rxs, ry = rys, rz = rzs``, via
:func:`cm1core.nest_parcels.sample_mass_rectangles`).  Between two history
frames there is no way to do that; along a model-step stream there is.

THE TERMS
---------
For :math:`\\boldsymbol\\omega = \\nabla\\times\\mathbf V`,

.. math::

    \\frac{D\\boldsymbol\\omega}{Dt}
      = (\\boldsymbol\\omega\\cdot\\nabla)\\mathbf V
      - \\boldsymbol\\omega(\\nabla\\cdot\\mathbf V)
      + \\frac{1}{\\rho^{2}}\\nabla\\rho\\times\\nabla p
      + \\nabla\\times\\mathbf F

published as ``stretching`` (the like-with-like part of the first term),
``tilting`` (its cross terms), ``divergence``, ``solenoid`` and ``friction``,
per component, in :data:`cm1core.vorticity_budget.VORTICITY_BUDGET_FIELD_NAMES`
order.

FRICTION IS THE CURL OF THE HELD TENDENCY
-----------------------------------------
The stream carries the step's held momentum tendency -- CM1's
``uten1``/``vten1``/``wten1``, formed once per step in ``solve1.F`` before the
RK loop and added unchanged at every stage.  In CM1 that array holds the
Rayleigh damping (``solve1.F:414-446``), large-scale forcing
(``solve1.F:538-672``), ``idiff`` diffusion (``solve1.F:688-734``) and the
subgrid stress ``turbu``/``turbv``/``turbw`` (``solve1.F:1124-1160``,
``src/turbtend.F``).  The door (:mod:`cm1core.nest_child_window`) admits a
window only below the Rayleigh layer and with no ``hrdamp``, and admits no
large-scale forcing but ``lspgrad = 1`` (removed below, under ROTATION), so
**inside the window** the tendency is the subgrid stress plus any ``idiff``
diffusion, and its curl is published as ``friction``.  On a free-slip child it is purely internal mixing:
there is no surface stress.

What no held array carries is the implicit dissipation of the odd-order
(5th-order upwind / WENO) advection, which acts on momentum inside the
advection term.  It stays in the closure residual

    ``residual = D(omega_i)/Dt|measured - (stretching + tilting + divergence + solenoid + friction)``

together with the discretisation error of the published terms and the error
of sampling a trilinear field along a moving point.

ROTATION (``icor = 1``, a v3 stream)
-----------------------------------
CM1 adds the Coriolis force to the *stage* tendency (``solve2.F:633-653``),
never to the held ``uten1``, so no streamed word carries it.  The budget
re-forms it from the record's own advecting wind and the stream header's
``fcor`` and frame velocity, in CM1's float32 operation order on CM1's own
faces:

    ``Fu = fcor*(0.25*((v(i,j)+v(i,j+1))+(v(i-1,j)+v(i-1,j+1))) + vmove)``
    ``Fv = -fcor*(0.25*((u(i,j)+u(i+1,j))+(u(i,j-1)+u(i+1,j-1))) + umove)``

i.e. ``-f k x (V + Vmove)`` on the ground-relative wind (the momentum
budget's ``coriolis_accel_*`` rows), and publishes its curl -- by the same
centred operator the friction term uses -- as ``coriolis_*``, which is
``(f du/dz, f dv/dz, -f (du/dx + dv/dy))``: the tilting and stretching of the
planetary vorticity.  The frame velocity is constant and
drops out of the curl, but not out of the momentum rows.

The wind it is evaluated on is the record's ``uadv``/``vadv`` -- the last RK
stage's acoustic small-step average (``solve2.F:1290-1301``), the same wind
every other term here is formed from -- not the ``u3d``/``v3d`` CM1 hands
``solve2.F:633-653`` at each stage.  The operator is CM1's word for word (the
device check compares it with the model's own kernel on the same words); the
force is the budget's instantaneous one, not a copy of what any one stage
applied.

With ``lspgrad = 1`` the held tendency *does* hold ``solve1.F:590-612``'s
per-level geostrophic words ``-fcor (V0 + Vmove)`` x k; the header records
them, the budget subtracts them from the held tendency before naming the rest
friction, and publishes their own curl (``-f dU0/dz``, ``-f dV0/dz``, 0) as
``lspg_*`` -- so the large-scale pressure gradient is counted once, as itself,
and ``coriolis_* + lspg_*`` is the curl of CM1's perturbation Coriolis force.

A non-rotating (v2) stream publishes exactly the rows it always did.

The solenoid uses the total density ``rho_d (1 + qv + qhyd)``: the dry air
alone would drop the vapour and condensate loading whose horizontal gradients
are part of the baroclinic generation near a hook echo.

THE CLOSURE CHECK
-----------------
``D(zeta)/Dt|measured`` is the centred time difference of the sampled
``zvort`` along the trajectory (one-sided at the two ends).  The check is
validated against a closed form rather than against CM1, because CM1 has no
budget: the field

.. math::

    u = -\\tfrac{a}{2}x - \\Omega(t)\\,y,\\quad
    v = -\\tfrac{a}{2}y + \\Omega(t)\\,x,\\quad
    w = a z,\\quad \\Omega(t) = \\Omega_0 e^{a t}

is an exact solution of the vorticity equation: it is non-divergent,
:math:`\\zeta = 2\\Omega(t)` everywhere, tilting and the divergence term
vanish identically, and ``stretching_z`` :math:`= \\zeta\\,\\partial w/\\partial z
= 2a\\Omega` is exactly :math:`D\\zeta/Dt`.  Its trajectories are closed form
too, so the same field checks the integrator and the budget at once
(:mod:`tests.test_nest_window_budget`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .vorticity_budget import (
    VORTICITY_BUDGET_FIELD_NAMES,
    VORTICITY_COMPONENT_NAMES,
    cpu_vorticity_budget,
)

TRAJECTORY_BUDGET_SCHEMA = "gm1.nest-window-trajectory-budget/v2"
#: v3 is v2 plus the rotation rows; written only from a rotating (v3) stream.
TRAJECTORY_BUDGET_ROTATING_SCHEMA = "gm1.nest-window-trajectory-budget/v3"
TRAJECTORY_BUDGET_KIND = "offline-diagnostic"
BUDGET_ARENA_FILENAME = "trajectory-budget.f32"
BUDGET_TABLE_FILENAME = "trajectory-budget-table.tsv"
BUDGET_RECEIPT_FILENAME = "trajectory-budget.json"

SENTINEL_F32 = -1.0e30

#: Sampled straight from the record, beside the budget.  ``buoyancy`` is
#: parcel.F:768-775's ``g*(th'/th0 + repsm1*(qv - qv0) - qhyd)``.
SCALAR_ROWS: tuple[tuple[str, str], ...] = (
    ("rho", "kg/m3"), ("prs", "Pa"), ("th", "K"), ("qv", "kg/kg"), ("qhyd", "kg/kg"),
    ("buoyancy", "m/s2"),
)
#: The closure columns, one triple per component, appended after the
#: components, the terms and the scalars.
CLOSURE_ROWS: tuple[tuple[str, str], ...] = tuple(
    (f"{component}_{what}", "s-2")
    for component in ("xvort", "yvort", "zvort")
    for what in ("tendency_measured", "tendency_terms", "residual")
)

BUDGET_ROWS: tuple[tuple[str, str], ...] = (
    *((name, "s-1") for name in VORTICITY_COMPONENT_NAMES),
    *((name, "s-2") for name in VORTICITY_BUDGET_FIELD_NAMES),
    *SCALAR_ROWS,
    *CLOSURE_ROWS,
)

#: The vorticity terms a closure sums, in the order it sums them.  The last
#: two exist only on a rotating stream (``lspg`` only with lspgrad=1).
CLOSURE_TERMS: tuple[str, ...] = ("stretching", "tilting", "divergence", "solenoid", "friction")
#: A rotating stream's extra rows: the Coriolis force (momentum, at scalar
#: points) and its curl (vorticity); with lspgrad=1 the same pair for the
#: large-scale pressure gradient the held tendency carries.
CORIOLIS_ROWS: tuple[tuple[str, str], ...] = (
    ("coriolis_x", "s-2"), ("coriolis_y", "s-2"), ("coriolis_z", "s-2"),
    ("coriolis_accel_x", "m/s2"), ("coriolis_accel_y", "m/s2"),
)
LSPG_ROWS: tuple[tuple[str, str], ...] = (
    ("lspg_x", "s-2"), ("lspg_y", "s-2"), ("lspg_z", "s-2"),
    ("lspg_accel_x", "m/s2"), ("lspg_accel_y", "m/s2"),
)


def budget_rows(rotation: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """The rows a budget of a stream with this ``rotation`` header publishes.

    ``None`` (a non-rotating stream) is :data:`BUDGET_ROWS` exactly.  A
    rotating stream inserts :data:`CORIOLIS_ROWS` (and, with lspgrad=1,
    :data:`LSPG_ROWS`) after the scalars and before the closure columns.
    """

    if rotation is None:
        return BUDGET_ROWS
    extra = CORIOLIS_ROWS + (LSPG_ROWS if int(rotation.get("lspgrad", 0)) == 1 else ())
    return (
        *((name, "s-1") for name in VORTICITY_COMPONENT_NAMES),
        *((name, "s-2") for name in VORTICITY_BUDGET_FIELD_NAMES),
        *SCALAR_ROWS,
        *extra,
        *CLOSURE_ROWS,
    )


def sampled_names(rotation: Mapping[str, Any] | None) -> list[str]:
    """Every row :func:`record_budget` forms and a trajectory samples (all but the closure)."""

    closure = {name for name, _ in CLOSURE_ROWS}
    return [name for name, _ in budget_rows(rotation) if name not in closure]

SOURCE_ANCHORS: Mapping[str, str] = {
    "components": "src/misclibs.F:1542-2100 (calcvort, flat Cartesian branch)",
    "xvort_flat": "src/misclibs.F:1618-1683",
    "yvort_flat": "src/misclibs.F:1854-1900",
    "zvort_flat": "src/misclibs.F:2035-2056",
    "scalar_point_stencil": "src/parcel.F:1365-1447 (rx = rxs, ry = rys, rz = rzs)",
    "lateral_rule": "src/bc.F:180-300 (bcs), one cell deep",
    "advecting_velocity": "src/solve2.F:1290-1301; src/solve3.F:438-469",
    "friction_terms_cm1_would_use": "src/turbtend.F:1326-1347,2092-2114,2852-2871,395-420",
    "budget": "GM1 EXTENSION: CM1 r21.1 carries no vorticity budget",
}
ROTATION_SOURCE_ANCHORS: Mapping[str, str] = {
    "coriolis_force": "src/solve2.F:633-653 (f plane, frame add-back at :645,647)",
    "lspgrad_1": "src/solve1.F:590-612 (inside the held uten1/vten1)",
}

#: No term is refused.  (Schema v1 refused friction because the stream carried
#: no momentum tendency; v2 carries the held tendency and the door admits it
#: only where it is the subgrid stress.)
REFUSED_TERMS: Mapping[str, str] = {}

UNVALIDATED_MARKER = (
    "UNVALIDATED against CM1: CM1 r21.1 has no vorticity budget of any kind and no Lagrangian "
    "budget, so there is nothing to difference this against.  Validated analytically instead, "
    "against a closed-form stretching vortex whose vorticity equation closes exactly."
)

CITATIONS = (
    "CM1 r21.1 src/misclibs.F:1542-2100 calcvort (vorticity components, flat Cartesian branch)",
    "CM1 r21.1 src/parcel.F:1365-1447 (scalar-point trilinear stencil)",
    "CM1 r21.1 src/solve2.F:1290-1301 (the advecting velocity the stream carries)",
    "Orf, L., R. Wilhelmson, B. Lee, C. Finley and A. Houston, 2017: Evolution of a long-track "
    "violent tornado within a simulated supercell. Bull. Amer. Meteor. Soc., 98, 45-68",
    "Dahl, J. M. L., M. D. Parker, and L. J. Wicker, 2012: Uncertainties in trajectory "
    "calculations within near-surface mesocyclones of simulated supercells. Mon. Wea. Rev., "
    "140, 2959-2966",
    "Markowski, P. M., and Y. P. Richardson, 2014: The influence of environmental low-level "
    "shear and cold pools on tornadogenesis. J. Atmos. Sci., 71, 243-275 (the circuit method)",
)


class TrajectoryBudgetError(ValueError):
    """A stream or a trajectory set that cannot produce the budget it names."""


def _np() -> Any:
    import numpy

    return numpy


# ---------------------------------------------------------------------------
# One record's budget, on the window's own physical rectangle.
# ---------------------------------------------------------------------------
def centred_solenoid(rho_ring: Any, prs_ring: Any, *, dx: float, dy: float,
                     zh: Any) -> dict[str, Any]:
    """``(1/rho^2) grad(rho) x grad(p)`` with every derivative centred.

    ``rho_ring`` and ``prs_ring`` are the window's ``(nz+2, ny+2, nx+2)``
    blocks -- the physical rectangle with one ring of the child's own
    neighbouring cells -- so the solenoid is centred at *every* point of the
    window, including its outermost ring and its top and bottom rows.  The
    Eulerian frame (``cm1core.cuda_cartesian_runtime._fill_solenoid``) is
    one-sided there, because the whole domain has no neighbour beyond its edge;
    a window cut out of a child does, and using it is the point of cutting the
    ring.

    ``zh`` is the window deck's CM1-indexed half levels, so the vertical
    gradient is taken over the deck's own spacings and a stretched child --
    which every tornado-grade child is, 20 m at the ground under a 17 km top --
    is treated correctly rather than with one nominal ``dz``.
    """

    np = _np()
    f = np.float32
    rho = np.ascontiguousarray(rho_ring, dtype=np.float32)
    prs = np.ascontiguousarray(prs_ring, dtype=np.float32)
    if rho.shape != prs.shape or len(rho.shape) != 3:
        raise TrajectoryBudgetError("rho and prs rings must be the same (nz+2, ny+2, nx+2) block")

    span = f(np.asarray(zh[2:], dtype=np.float32) - np.asarray(zh[:-2], dtype=np.float32))
    if span.shape[0] != rho.shape[0] - 2:
        raise TrajectoryBudgetError(
            f"zh carries {span.shape[0] + 2} centred levels but the rho ring has "
            f"{rho.shape[0]}; the vertical gradient would be taken over another deck's spacings"
        )

    def grad(block: Any, spacing: float, axis: int) -> Any:
        two = f(f(2.0) * f(spacing))
        if axis == 0:
            return f(f(block[2:, 1:-1, 1:-1] - block[:-2, 1:-1, 1:-1]) / span[:, None, None])
        if axis == 1:
            return f(f(block[1:-1, 2:, 1:-1] - block[1:-1, :-2, 1:-1]) / two)
        return f(f(block[1:-1, 1:-1, 2:] - block[1:-1, 1:-1, :-2]) / two)

    drdz, drdy, drdx = (grad(rho, s, a) for s, a in ((0.0, 0), (dy, 1), (dx, 2)))
    dpdz, dpdy, dpdx = (grad(prs, s, a) for s, a in ((0.0, 0), (dy, 1), (dx, 2)))
    core = rho[1:-1, 1:-1, 1:-1]
    inv = f(f(1.0) / f(core * core))
    return {
        "solenoid_x": f(inv * f(f(drdy * dpdz) - f(drdz * dpdy))),
        "solenoid_y": f(inv * f(f(drdz * dpdx) - f(drdx * dpdz))),
        "solenoid_z": f(inv * f(f(drdx * dpdy) - f(drdy * dpdx))),
    }


def centred_curl(uten: Any, vten: Any, wten: Any, *, dx: float, dy: float,
                 zh: Any, prefix: str = "friction") -> dict[str, Any]:
    """``curl(uten, vten, wten)`` at the window's scalar points, every derivative centred.

    ``uten``/``vten``/``wten`` are a record's held momentum tendency blocks on
    their own faces (``(nz+1, ny+2, nx+3)``, ``(nz+1, ny+3, nx+2)``,
    ``(nz+2, ny+2, nx+2)``: CM1 levels ``1..nz+1`` and one ring).  Each is
    averaged to scalar points exactly as :mod:`cm1core.vorticity_budget`
    averages the winds, given the ghost level of
    :func:`cm1core.nest_child_window.fill_ring_bottom_ghost`, and differenced
    over the deck's own ``zh``.
    """

    from .nest_child_window import fill_ring_bottom_ghost

    np = _np()
    f = np.float32
    half = f(0.5)
    su = f(half * f(np.asarray(uten, np.float32)[:, :, :-1] + np.asarray(uten, np.float32)[:, :, 1:]))
    sv = f(half * f(np.asarray(vten, np.float32)[:, :-1, :] + np.asarray(vten, np.float32)[:, 1:, :]))
    sw = f(half * f(np.asarray(wten, np.float32)[:-1] + np.asarray(wten, np.float32)[1:]))
    su, sv, sw = (fill_ring_bottom_ghost(a, zh) for a in (su, sv, sw))
    span = f(np.asarray(zh[2:su.shape[0]], np.float32) - np.asarray(zh[:su.shape[0] - 2], np.float32))

    def d(block: Any, axis: int, spacing: float) -> Any:
        if axis == 0:
            return f(f(block[2:, 1:-1, 1:-1] - block[:-2, 1:-1, 1:-1]) / span[:, None, None])
        two = f(f(2.0) * f(spacing))
        if axis == 1:
            return f(f(block[1:-1, 2:, 1:-1] - block[1:-1, :-2, 1:-1]) / two)
        return f(f(block[1:-1, 1:-1, 2:] - block[1:-1, 1:-1, :-2]) / two)

    return {
        f"{prefix}_x": f(d(sw, 1, dy) - d(sv, 0, 0.0)),
        f"{prefix}_y": f(d(su, 0, 0.0) - d(sw, 2, dx)),
        f"{prefix}_z": f(d(sv, 2, dx) - d(su, 1, dy)),
    }


def coriolis_face_forces(uadv: Any, vadv: Any, *, fcor_s_inv: float, u_move_m_s: float,
                         v_move_m_s: float) -> tuple[Any, Any]:
    """CM1's f-plane Coriolis force on the held-tendency faces of a window record.

    ``uadv``/``vadv`` are the record's padded wind blocks
    (:meth:`cm1core.nest_child_window.WindowBox.shapes`); the result is on the
    ``uten``/``vten`` blocks' own faces and levels -- ``(nz+1, ny+2, nx+3)``
    and ``(nz+1, ny+3, nx+2)``, CM1 levels ``1..nz+1`` with one ring -- so its
    curl is taken by exactly the operator the friction term uses.  Every word
    is ``solve2.F:639-653``'s, in its float32 order::

        Fu = fcor*(0.25*((v(i,j)+v(i,j+1))+(v(i-1,j)+v(i-1,j+1))) + vmove)
        Fv = -(fcor*(0.25*((u(i,j)+u(i+1,j))+(u(i,j-1)+u(i+1,j-1))) + umove))

    The ``u`` face ``(k', j', i')`` of the ring block is the window's padded
    ``uadv[k'+1, j'-1+NG, i'-1+NG]``, whose four ``v`` neighbours are
    ``vadv[k'+1, j'-1+NG+{0,1}, i'-1+NG+{0,-1}]``; the ``v`` face likewise.
    """

    from .nest_child_window import NG

    np = _np()
    f = np.float32
    u = np.asarray(uadv, dtype=np.float32)
    v = np.asarray(vadv, dtype=np.float32)
    nz = u.shape[0] - 2
    ny = u.shape[1] - 2 * NG
    nx = u.shape[2] - 1 - 2 * NG
    if tuple(v.shape) != (nz + 2, ny + 1 + 2 * NG, nx + 2 * NG):
        raise TrajectoryBudgetError(
            f"uadv {tuple(u.shape)} and vadv {tuple(v.shape)} are not one window's padded "
            "blocks; the Coriolis force would average the wrong neighbours"
        )
    fcor, umove, vmove = f(fcor_s_inv), f(u_move_m_s), f(v_move_m_s)
    ks = slice(1, nz + 2)
    # u faces of the ring block: rows NG-1 .. NG+ny, columns NG-1 .. NG+nx+1.
    rows, rows1 = slice(NG - 1, NG + ny + 1), slice(NG, NG + ny + 2)
    cols, cols_w = slice(NG - 1, NG + nx + 2), slice(NG - 2, NG + nx + 1)
    first = f(v[ks, rows, cols] + v[ks, rows1, cols])
    second = f(v[ks, rows, cols_w] + v[ks, rows1, cols_w])
    fu = f(fcor * f(f(f(first + second) * f(0.25)) + vmove))
    # v faces of the ring block: rows NG-1 .. NG+ny+1, columns NG-1 .. NG+nx.
    rows, rows_s = slice(NG - 1, NG + ny + 2), slice(NG - 2, NG + ny + 1)
    cols, cols1 = slice(NG - 1, NG + nx + 1), slice(NG, NG + nx + 2)
    first = f(u[ks, rows, cols] + u[ks, rows, cols1])
    second = f(u[ks, rows_s, cols] + u[ks, rows_s, cols1])
    fv = f(f(0.0) - f(fcor * f(f(f(first + second) * f(0.25)) + umove)))
    return np.ascontiguousarray(fu), np.ascontiguousarray(fv)


def _face_to_scalar_core(fu: Any, fv: Any) -> tuple[Any, Any]:
    """Two face blocks averaged to the window's own scalar points, ``(nz, ny, nx)``."""

    np = _np()
    f = np.float32
    half = f(0.5)
    su = f(half * f(fu[:, :, :-1] + fu[:, :, 1:]))[:-1, 1:-1, 1:-1]
    sv = f(half * f(fv[:, :-1, :] + fv[:, 1:, :]))[:-1, 1:-1, 1:-1]
    return np.ascontiguousarray(su), np.ascontiguousarray(sv)


def rotation_terms(words: Mapping[str, Any], deck: Any,
                   rotation: Mapping[str, Any]) -> dict[str, Any]:
    """The rotating stream's extra rectangles: Coriolis (and lspgrad=1's lspg), both views.

    Returns ``coriolis_{x,y,z}`` (curl of the force), ``coriolis_accel_{x,y}``
    (the force at scalar points) and, with lspgrad=1, the same for ``lspg``
    plus the two per-level held-tendency blocks ``_lspg_u``/``_lspg_v`` the
    friction term subtracts.
    """

    np = _np()
    fu, fv = coriolis_face_forces(
        words["uadv"], words["vadv"], fcor_s_inv=float(rotation["fcor_s_inv"]),
        u_move_m_s=float(rotation["u_move_m_s"]), v_move_m_s=float(rotation["v_move_m_s"]))
    zero_w = np.zeros(np.asarray(words["wten"]).shape, dtype=np.float32)
    out = centred_curl(fu, fv, zero_w, dx=deck.dx_m, dy=deck.dy_m, zh=deck.zh,
                       prefix="coriolis")
    out["coriolis_accel_x"], out["coriolis_accel_y"] = _face_to_scalar_core(fu, fv)
    if int(rotation.get("lspgrad", 0)) == 1:
        uten = np.asarray(words["uten"])
        vten = np.asarray(words["vten"])
        lu = np.asarray(rotation["lspg_u_m_s2"], dtype=np.float32)
        lv = np.asarray(rotation["lspg_v_m_s2"], dtype=np.float32)
        if lu.shape[0] != uten.shape[0] or lv.shape[0] != vten.shape[0]:
            raise TrajectoryBudgetError(
                f"lspg words cover {lu.shape[0]} levels but the held tendency {uten.shape[0]}; "
                "the large-scale pressure gradient would be removed from the wrong levels"
            )
        bu = np.ascontiguousarray(np.broadcast_to(lu[:, None, None], uten.shape), np.float32)
        bv = np.ascontiguousarray(np.broadcast_to(lv[:, None, None], vten.shape), np.float32)
        out.update(centred_curl(bu, bv, zero_w, dx=deck.dx_m, dy=deck.dy_m, zh=deck.zh,
                                prefix="lspg"))
        out["lspg_accel_x"], out["lspg_accel_y"] = _face_to_scalar_core(bu, bv)
        out["_lspg_u"], out["_lspg_v"] = bu, bv
    return out


def record_budget(words: Mapping[str, Any], deck: Any, *,
                  base_columns: Mapping[str, Any] | None = None,
                  rotation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every component and term of one window record, as ``(nz, ny, nx)`` rectangles.

    ``words`` is a record's fields
    (:meth:`cm1core.nest_child_window.WindowStream.fields`).  Every vertical
    derivative is taken over the window deck's own ``zh``, because a
    tornado-grade child is stretched -- 20 m at the ground under a 17 km top --
    and one nominal ``dz`` would mis-scale the stretching and tilting terms of
    exactly the near-ground parcels this tool follows.

    The solenoid is formed with the **total** density
    ``rho_d (1 + qv + qhyd)`` -- dry air, vapour and condensate loading, the
    density CM1's own buoyancy (parcel.F:768-775, solve2's ``thv``) acts on --
    and the full pressure.  The friction term is the curl of the step's held
    momentum tendency (:func:`centred_curl`), which the door admits only where
    that tendency is the subgrid stress (and any ``idiff`` diffusion).

    ``rotation`` is a v3 stream's header (:attr:`WindowStream.rotation`):
    given, the Coriolis rows are added, and with lspgrad=1 the held tendency's
    large-scale pressure gradient is taken out of friction and published as
    ``lspg`` (see the module docstring, ROTATION).
    """

    from .nest_child_window import fill_ring_bottom_ghost

    np = _np()
    f = np.float32
    budget = cpu_vorticity_budget(
        u=words["uadv"], v=words["vadv"], w=words["wadv"],
        nx=deck.nx, ny=deck.ny, nz=deck.nz,
        dx=deck.dx_m, dy=deck.dy_m, dz=float(deck.dz_reference_m), zh=deck.zh,
    )
    out: dict[str, Any] = {
        "xvort": budget.components.xvort,
        "yvort": budget.components.yvort,
        "zvort": budget.components.zvort,
    }
    out.update({name: value for name, value in budget.terms.items()})
    zh = deck.zh
    ring = {name: fill_ring_bottom_ghost(words[name], zh)
            for name in ("rho", "prs", "th", "qv", "qhyd")}
    total = f(ring["rho"] * f(f(f(1.0) + ring["qv"]) + ring["qhyd"]))
    out.update(centred_solenoid(total, ring["prs"], dx=deck.dx_m, dy=deck.dy_m, zh=zh))
    uten, vten = words["uten"], words["vten"]
    if rotation is not None:
        extra = rotation_terms(words, deck, rotation)
        if "_lspg_u" in extra:
            f32 = np.float32
            uten = f32(np.asarray(uten, np.float32) - extra.pop("_lspg_u"))
            vten = f32(np.asarray(vten, np.float32) - extra.pop("_lspg_v"))
        out.update(extra)
    out.update(centred_curl(uten, vten, words["wten"],
                            dx=deck.dx_m, dy=deck.dy_m, zh=zh))
    core = {name: np.ascontiguousarray(block[1:-1, 1:-1, 1:-1], dtype=np.float32)
            for name, block in ring.items()}
    out.update(core)
    if base_columns and "th0" in base_columns and "qv0" in base_columns:
        nz = deck.nz
        th0 = np.asarray(base_columns["th0"], np.float32)[:nz][:, None, None]
        qv0 = np.asarray(base_columns["qv0"], np.float32)[:nz][:, None, None]
        from .nest_parcels import _cm1_constants

        const = _cm1_constants(np)       # constants.F:112-133, CM1's REAL words
        g, repsm1 = const["g"], const["repsm1"]
        out["buoyancy"] = f(g * f(f(f(core["th"] / th0) + f(repsm1 * f(core["qv"] - qv0)))
                                  - core["qhyd"]))
    else:
        out["buoyancy"] = np.full_like(core["th"], np.float32(np.nan))
    return out


def vertical_metric_record(deck: Any) -> dict[str, Any]:
    """What vertical spacings the budget's derivatives were taken over.

    Published rather than assumed: a reader comparing two budgets has to be
    able to see that both were formed on the deck's own levels, and how far
    from uniform those levels were where the trajectories ran.
    """

    np = _np()
    zh = np.asarray(deck.zh[: deck.nz + 2], dtype=np.float64)
    gaps = np.diff(zh)
    return {
        "metric": "the window deck's own zh, centred: d/dz|k = (s[k+1] - s[k-1]) / (zh[k+1] - zh[k-1])",
        "cm1_equivalent": "src/misclibs.F:1628 calcvort's rdz*mf sigma metric",
        "levels": int(deck.nz),
        "spacing_m": {"min": float(gaps.min()), "max": float(gaps.max())},
        "uniform": bool(float(gaps.max() - gaps.min()) <= 1.0e-6 * max(1.0, float(gaps[0]))),
        "lowest_mass_level_m": float(deck.zh[1]),
        "top_m": float(deck.zf[deck.nz + 1]),
    }


# ---------------------------------------------------------------------------
# The budget along trajectories.
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryBudget:
    rows: Any                  # (steps+1, len(BUDGET_ROWS), nparcels) float32
    times_s: list[float]
    sampled: Any               # (steps+1, nparcels) bool: a row that holds a sample
    closure: dict[str, Any]
    vertical_metric: dict[str, Any]
    row_spec: tuple[tuple[str, str], ...] = BUDGET_ROWS
    rotation: dict[str, Any] | None = None


def _sample_rectangles(deck: Any, rectangles: Mapping[str, Any], positions: Any) -> dict[str, Any]:
    from .nest_parcels import sample_mass_rectangles

    np = _np()
    # sample_mass_rectangles reads only ``deck`` off its carriers (the
    # rectangles are passed in); ``ng`` and ``bbc`` are on the protocol and are
    # named here so the namespace is not mistaken for a runtime.
    carriers = SimpleNamespace(deck=deck, ng=0, bbc=0)
    return sample_mass_rectangles(np, carriers, positions, rectangles)


def budget_along(stream: Any, result: Any, *, verify: bool = False,
                 time_shift_s: float = 0.0, tolerance_s: float = 1.0e-6) -> TrajectoryBudget:
    """Sample every budget term at every trajectory position that has a record.

    ``stream`` is a :class:`cm1core.nest_child_window.WindowStream` and
    ``result`` a :class:`cm1core.nest_parcel_backward.BackwardResult`
    integrated through that same stream.  ``time_shift_s`` is what the
    integrator subtracted from every record's label (the tool labels a record
    at the **midpoint** of the child step it averages when it blends records in
    time, ``child_dt/2``); the budget of a record is sampled at a trajectory
    time only where the two agree to ``tolerance_s``, so it is never
    interpolated in time.  A trajectory stepped more finely than the stream
    was written therefore carries the budget at every record time and at no
    other, and the budget's own time axis is those steps.

    A row whose parcel had already stopped, or whose position is the sentinel,
    is left as the sentinel and marked unsampled.
    """

    from .nest_parcel_backward import STATUS_ACTIVE

    np = _np()
    rows_in = result.rows
    n = rows_in.shape[2]
    rotation = getattr(stream, "rotation", None)
    spec = budget_rows(rotation)
    names = [name for name, _ in spec]
    labels = [float(r.model_time_s) - float(time_shift_s) for r in stream.records]
    order = np.argsort(np.asarray(labels))
    sorted_labels = np.asarray(labels)[order]

    def record_at(t: float) -> int | None:
        q = int(np.searchsorted(sorted_labels, t))
        for c in (q - 1, q):
            if 0 <= c < len(sorted_labels) and abs(float(sorted_labels[c]) - t) <= tolerance_s:
                return int(order[c])
        return None

    steps_used: list[tuple[int, int]] = []
    for step, t in enumerate(result.times_s):
        index = record_at(float(t))
        if index is not None:
            steps_used.append((step, index))
    if not steps_used:
        raise TrajectoryBudgetError(
            "no trajectory time falls on a record of the stream; the budget would have to be "
            "interpolated in time, which is the error this stream exists to remove"
        )
    out = np.full((len(steps_used), len(names), n), np.float32(SENTINEL_F32), dtype=np.float32)
    sampled = np.zeros((len(steps_used), n), dtype=bool)
    term_names = sampled_names(rotation)
    base = stream.receipt.get("base_columns") or {}
    for q, (step, index) in enumerate(steps_used):
        live = rows_in[step, 7] == STATUS_ACTIVE
        live &= rows_in[step, 0] > np.float32(-1.0e29)
        if not bool(live.any()):
            continue
        record = stream.records[index]
        deck = stream.deck_for(record)
        words = stream.fields(index, verify=verify)
        rectangles = record_budget(words, deck, base_columns=base, rotation=rotation)
        positions = np.ascontiguousarray(rows_in[step, 0:3][:, live], dtype=np.float32)
        # A position outside this record's own rectangle (the window moved
        # under a parcel near its edge) is not sampled: the stencil would
        # clamp it onto the window's edge cell and publish another cell's term.
        inside = _inside_rectangle(deck, positions)
        if not bool(inside.any()):
            continue
        values = _sample_rectangles(deck, {k: rectangles[k] for k in term_names},
                                    np.ascontiguousarray(positions[:, inside]))
        cols = np.flatnonzero(live)[inside]
        for r, name in enumerate(term_names):
            out[q, r, cols] = np.asarray(values[name], np.float32)
        sampled[q, cols] = True
    times = [float(result.times_s[step]) for step, _ in steps_used]
    _fill_closure(out, np.asarray(times, dtype=np.float64), sampled, names)
    metric = vertical_metric_record(stream.deck_for(stream.records[0]))
    return TrajectoryBudget(out, times, sampled, closure_report(out, sampled, names), metric,
                            spec, rotation)


def _inside_rectangle(deck: Any, positions: Any) -> Any:
    np = _np()
    x, y = np.asarray(positions[0], np.float64), np.asarray(positions[1], np.float64)
    x0, y0 = float(deck.minx_m), float(deck.miny_m)
    # Half a cell inside the scalar rectangle: the scalar stencil then has a
    # neighbour each way without reaching past the published ring.
    return ((x >= x0 + 0.5 * deck.dx_m) & (x <= x0 + (deck.nx - 0.5) * deck.dx_m)
            & (y >= y0 + 0.5 * deck.dy_m) & (y <= y0 + (deck.ny - 0.5) * deck.dy_m))


def _fill_closure(out: Any, times: Any, sampled: Any, names: Sequence[str]) -> None:
    """The measured Lagrangian ``D(omega_i)/Dt``, the terms' sum and the residual, per component."""

    np = _np()
    steps = out.shape[0] - 1
    for component, axis in (("xvort", "x"), ("yvort", "y"), ("zvort", "z")):
        zi = names.index(component)
        measured = names.index(f"{component}_tendency_measured")
        summed = names.index(f"{component}_tendency_terms")
        residual = names.index(f"{component}_residual")
        # A rotating budget's names carry coriolis_* (and lspg_*); a
        # non-rotating one's do not, and sums exactly the five it always did.
        parts = [names.index(f"{term}_{axis}")
                 for term in (*CLOSURE_TERMS, "coriolis", "lspg")
                 if f"{term}_{axis}" in names]
        total = np.zeros_like(out[:, zi])
        for part in parts:
            total = total + np.where(sampled, out[:, part], np.float32(0.0))
        out[:, summed] = np.where(sampled, total, np.float32(SENTINEL_F32))
        zeta = out[:, zi]
        for step in range(steps + 1):
            lo = max(step - 1, 0)
            hi = min(step + 1, steps)
            usable = sampled[step] & sampled[lo] & sampled[hi] & (hi > lo)
            if not bool(usable.any()):
                continue
            span = float(times[hi] - times[lo])
            rate = (np.asarray(zeta[hi], np.float64) - np.asarray(zeta[lo], np.float64)) / span
            # The term sum centred on the same interval the difference spans,
            # so an end row (one-sided difference) is compared with the mean
            # of the two rows it spans rather than with one end of it.
            if hi - lo == 2:
                terms = np.asarray(out[step, summed], np.float64)
            else:
                terms = 0.5 * (np.asarray(out[lo, summed], np.float64)
                               + np.asarray(out[hi, summed], np.float64))
            out[step, measured] = np.where(usable, rate, SENTINEL_F32).astype(np.float32)
            out[step, residual] = np.where(usable, rate - terms, SENTINEL_F32).astype(np.float32)


def closure_report(rows: Any, sampled: Any, names: Sequence[str]) -> dict[str, Any]:
    """How well the resolved terms account for the rotation the parcels gained, per component."""

    np = _np()
    report: dict[str, Any] = {}
    for component in ("xvort", "yvort", "zvort"):
        measured = rows[:, names.index(f"{component}_tendency_measured")]
        summed = rows[:, names.index(f"{component}_tendency_terms")]
        residual = rows[:, names.index(f"{component}_residual")]
        usable = sampled & (measured > np.float32(-1.0e29)) & (residual > np.float32(-1.0e29))
        if not bool(usable.any()):
            report[component] = {"samples": 0,
                                 "note": "no trajectory position had three consecutive samples"}
            continue
        m = np.abs(np.asarray(measured[usable], dtype=np.float64))
        t = np.abs(np.asarray(summed[usable], dtype=np.float64))
        r = np.abs(np.asarray(residual[usable], dtype=np.float64))
        scale = np.maximum(m, t)
        ratio = np.where(scale > 0.0, r / np.where(scale > 0.0, scale, 1.0), 0.0)
        report[component] = {
            "samples": int(usable.sum()),
            "abs_measured_tendency_s2": {"median": float(np.median(m)), "max": float(m.max())},
            "abs_term_sum_s2": {"median": float(np.median(t)), "max": float(t.max())},
            "abs_residual_s2": {"median": float(np.median(r)), "max": float(r.max())},
            "residual_over_scale": {
                "median": float(np.median(ratio)),
                "p90": float(np.quantile(ratio, 0.9)),
                "max": float(ratio.max()),
            },
        }
    report["residual_holds"] = (
        "the implicit dissipation of the odd-order advection (which no held tendency carries), "
        "the discretisation error of the published terms, and the error of sampling a "
        "trilinear field along a moving point; friction is published as its own term"
    )
    # Back-compatible top-level summary: the vertical component.
    report.update({k: v for k, v in report["zvort"].items()})
    return report


# ---------------------------------------------------------------------------
# Publication.
# ---------------------------------------------------------------------------
def write_trajectory_budget(directory: Path | str, budget: TrajectoryBudget, *,
                            stream: Any, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Arena, field table and receipt for one trajectory budget."""

    np = _np()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    spec = tuple(getattr(budget, "row_spec", BUDGET_ROWS))
    rotation = getattr(budget, "rotation", None)
    raw = np.ascontiguousarray(budget.rows, dtype="<f4").tobytes(order="C")
    with (root / BUDGET_ARENA_FILENAME).open("xb", buffering=0) as handle:
        handle.write(raw)
        os.fsync(handle.fileno())
    table = ["index\tname\tunits"] + [
        f"{q}\t{name}\t{unit}" for q, (name, unit) in enumerate(spec)
    ]
    with (root / BUDGET_TABLE_FILENAME).open("xb", buffering=0) as handle:
        handle.write(("\n".join(table) + "\n").encode("utf-8"))
        os.fsync(handle.fileno())
    receipt = {
        "schema": TRAJECTORY_BUDGET_SCHEMA,
        "kind": TRAJECTORY_BUDGET_KIND,
        "label": (
            "OFFLINE DIAGNOSTIC -- the vorticity budget sampled at every trajectory position "
            "from a cm1core.nest_child_window model-step stream; not a CM1 product"
        ),
        "arena_shape_steps_rows_nparcels": list(budget.rows.shape),
        "arena_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": [name for name, _ in spec],
        "units": [unit for _, unit in spec],
        "model_times_s": list(budget.times_s),
        "sampled_positions": int(budget.sampled.sum()),
        "closure": dict(budget.closure),
        "vertical_metric": dict(budget.vertical_metric),
        "closure_definition": (
            "per component: residual = D(omega_i)/Dt measured as the centred time difference of "
            "the sampled component along the trajectory, minus (stretching + tilting + "
            "divergence + solenoid + friction) of that component"
        ),
        "friction_definition": (
            "curl of the step's held momentum tendency (CM1 uten1/vten1/wten1): inside an "
            "admitted window, turbtend.F's subgrid stress plus any idiff diffusion"
        ),
        "refused_terms": dict(REFUSED_TERMS),
        "stream": {
            "directory": str(stream.directory),
            "schema": stream.receipt["schema"],
            # Schema v3 publishes the species, not qhyd; the budget's total
            # density and buoyancy read the loading WindowStream.fields
            # rebuilt from them (nest_child_window.rebuild_qhyd).
            "qhyd_source": ("rebuilt ((qc + qr) + qis) + qg from the record's species"
                            if stream.receipt["schema"].endswith("/v3")
                            else "the record's own qhyd word"),
            "records": len(stream.records),
            "dt_s": stream.dt_s,
            "window": stream.receipt["window"],
        },
        "cm1_authority": dict(SOURCE_ANCHORS),
        "citations": list(CITATIONS),
        "unvalidated": UNVALIDATED_MARKER,
    }
    if rotation is not None:
        lspg = int(rotation.get("lspgrad", 0)) == 1
        receipt["schema"] = TRAJECTORY_BUDGET_ROTATING_SCHEMA
        receipt["rotation"] = dict(rotation)
        receipt["closure_definition"] = (
            "per component: residual = D(omega_i)/Dt measured as the centred time difference of "
            "the sampled component along the trajectory, minus (stretching + tilting + "
            "divergence + solenoid + friction + coriolis" + (" + lspg" if lspg else "")
            + ") of that component"
        )
        receipt["coriolis_definition"] = (
            "coriolis_accel_x/y: solve2.F:639-653's f-plane force fcor*(vbar + vmove), "
            "-fcor*(ubar + umove) on the record's own faces, re-formed from uadv/vadv and the "
            "stream header (CM1 adds it to the stage tendency, not the held one), averaged to "
            "scalar points; coriolis_x/y/z: its centred curl, (f du/dz, f dv/dz, "
            "-f (du/dx + dv/dy)) -- the tilting and stretching of the planetary vorticity"
        )
        receipt["friction_definition"] = (
            "curl of the step's held momentum tendency (CM1 uten1/vten1/wten1)"
            + (" less solve1.F:590-612's per-level lspgrad=1 words, which are published as "
               "lspg" if lspg else "")
            + ": inside an admitted window, turbtend.F's subgrid stress plus any idiff diffusion"
        )
        if lspg:
            receipt["lspg_definition"] = (
                "lspg_accel_x/y: the per-level large-scale pressure gradient solve1.F:590-612 "
                "holds in uten1/vten1 (-fcor*(v0 + vmove), fcor*(u0 + umove)), from the stream "
                "header; lspg_x/y/z: its centred curl (-f dU0/dz, -f dV0/dz, 0).  coriolis + "
                "lspg is the curl of CM1's perturbation Coriolis force"
            )
        receipt["cm1_authority"] = {**receipt["cm1_authority"], **ROTATION_SOURCE_ANCHORS}
    if extra:
        receipt.update(dict(extra))
    with (root / BUDGET_RECEIPT_FILENAME).open("xb", buffering=0) as handle:
        handle.write((json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
                     .encode("utf-8"))
        os.fsync(handle.fileno())
    return receipt


# ---------------------------------------------------------------------------
# Seeding into the vortex, and material circuits.
# ---------------------------------------------------------------------------
def core_disk_positions(*, centre_x_m: float, centre_y_m: float, radius_m: float,
                        heights_m: Sequence[float], rings: int, per_ring: int) -> Any:
    """Parcels filling a disk of ``radius_m`` about a vortex centre, at each height.

    The measured failure of a lattice seeding is that a 40 m lattice does not
    sample a 40-60 m core.  A disk seeding puts every parcel inside the core by
    construction: ``rings`` concentric rings (the first is the centre itself)
    of ``per_ring`` parcels each, at every height in ``heights_m``.  The rings
    are equal-area, so the parcels are uniform over the disk rather than
    crowded at its centre.
    """

    np = _np()
    if rings < 1 or per_ring < 1:
        raise TrajectoryBudgetError("a core disk needs at least one ring of at least one parcel")
    if radius_m <= 0.0:
        raise TrajectoryBudgetError(
            "a core disk needs a positive radius; a zero radius stacks every parcel on one "
            "point and every trajectory is the same trajectory"
        )
    xs, ys, zs = [], [], []
    for z in heights_m:
        xs.append(centre_x_m)
        ys.append(centre_y_m)
        zs.append(float(z))
        for ring in range(1, rings):
            # equal-area rings: r_k = R sqrt(k / (rings - 1))
            r = float(radius_m) * math.sqrt(ring / float(rings - 1)) if rings > 1 else 0.0
            for q in range(per_ring):
                angle = 2.0 * math.pi * q / per_ring + 0.5 * ring
                xs.append(centre_x_m + r * math.cos(angle))
                ys.append(centre_y_m + r * math.sin(angle))
                zs.append(float(z))
    return np.stack([np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
                     np.asarray(zs, dtype=np.float32)])


def circuit_positions(*, centre_x_m: float, centre_y_m: float, radius_m: float,
                      height_m: float, count: int) -> Any:
    """One closed material circuit: ``count`` vertices on a circle, in order.

    The Markowski and Richardson (2014) circuit: seed a ring, carry it with the
    flow -- here backward through the stream -- and close the line integral
    about it at every record.  Kelvin's theorem says the circulation is
    conserved for barotropic inviscid flow, so what it is *not* conserved by is
    the answer to where the rotation came from.
    """

    np = _np()
    if count < 3:
        raise TrajectoryBudgetError(
            "a closed circuit needs at least three vertices; two enclose no area and their "
            "circulation is identically zero for every flow"
        )
    angles = [2.0 * math.pi * q / count for q in range(count)]
    return np.stack([
        np.asarray([centre_x_m + radius_m * math.cos(a) for a in angles], dtype=np.float32),
        np.asarray([centre_y_m + radius_m * math.sin(a) for a in angles], dtype=np.float32),
        np.full(count, np.float32(height_m), dtype=np.float32),
    ])


def circulation_along(result: Any, *, first_parcel: int, count: int,
                      name: str = "circuit") -> dict[str, Any]:
    """The circuit's circulation at every published time, from the trajectory rows.

    The velocities are the ones the integrator already published beside every
    position (the record's own wind at that position), so nothing is resampled
    and the line integral is over the model's own words.
    """

    from .circulation_circuits import CircuitDefinition, circuit_circulation
    from .nest_parcel_backward import STATUS_ACTIVE

    np = _np()
    definition = CircuitDefinition(name=name, first_parcel=int(first_parcel), count=int(count))
    lo, hi = definition.first_parcel, definition.stop
    samples = []
    for step in range(result.rows.shape[0]):
        row = result.rows[step]
        if not bool((row[7, lo:hi] == STATUS_ACTIVE).all()):
            break
        positions = np.ascontiguousarray(row[0:3], dtype=np.float64)
        velocities = np.ascontiguousarray(row[3:6], dtype=np.float64)
        # The integrator publishes the step's stage-1 wind beside the position
        # it was taken at, so the last row -- which begins no step -- carries
        # the sentinel and closes no line integral.
        if not (np.isfinite(velocities[:, lo:hi]).all()
                and bool((velocities[:, lo:hi] > -1.0e29).all())):
            break
        sample = circuit_circulation(positions, velocities, definition)
        entry = sample.record()
        entry["model_time_s"] = float(result.times_s[step])
        samples.append(entry)
    return {
        "circuit": definition.record(),
        "samples": samples,
        "kelvin": (
            "for barotropic inviscid flow the circulation of a material circuit is conserved "
            "(Kelvin); what it changes by is the baroclinic and frictional line integral "
            "(Bjerknes), and this tool publishes the circulation, not that split"
        ),
        "citation": (
            "Markowski, P. M., and Y. P. Richardson, 2014: J. Atmos. Sci., 71, 243-275"
        ),
    }


__all__ = [
    "BUDGET_ARENA_FILENAME", "BUDGET_RECEIPT_FILENAME", "BUDGET_ROWS",
    "BUDGET_TABLE_FILENAME", "CITATIONS", "CLOSURE_ROWS", "CLOSURE_TERMS", "CORIOLIS_ROWS",
    "LSPG_ROWS", "REFUSED_TERMS", "ROTATION_SOURCE_ANCHORS",
    "TRAJECTORY_BUDGET_ROTATING_SCHEMA", "budget_rows", "coriolis_face_forces",
    "rotation_terms", "sampled_names",
    "SCALAR_ROWS", "SOURCE_ANCHORS", "TRAJECTORY_BUDGET_KIND",
    "TRAJECTORY_BUDGET_SCHEMA", "TrajectoryBudget", "TrajectoryBudgetError",
    "UNVALIDATED_MARKER", "budget_along", "centred_solenoid", "circuit_positions",
    "circulation_along", "closure_report", "core_disk_positions", "record_budget",
    "vertical_metric_record", "write_trajectory_budget",
]
