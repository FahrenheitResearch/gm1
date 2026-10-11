"""Lagrangian parcels integrated inside the live nest child, on its own grid.

A nested run's ``[lagrangian]`` parcels (:mod:`cm1core.lagrangian_stage`)
advect on the parent.  A trajectory into a vortex the child resolves and the
parent does not must read the child's winds, on the child's lattice, on the
child's own vertical deck, at the child's step.  This module does that.

AUTHORITY
---------
CM1 r21.1 has no nesting (``records/CM1-NEST-SEAM.md`` section 1), so the
*placement* of parcels in a child is GM1's.  The *parcel arithmetic* is CM1's
and is transcribed, not re-derived:

* ``src/parcel.F:10-620`` ``parcel_driver`` (``terrain_flag`` false): the
  vertical ghost fill ``:59-130``, the horizontal searches ``:153-231``, the
  vertical search ``:251-260``, the u/v/w staggered weights ``:278-461``, the
  semi-slip log layer below the lowest level ``:470-515``, the RK2 update
  ``:521-568`` and the periodic exit ``:577-593``.
* ``src/parcel.F:1693-1711`` ``tri_interp``: the eight-term sum in written order;
  ``:1719-1787`` ``get2d``: the four-term horizontal sum in written order.
* ``src/solve3.F:769`` and ``src/cm1.F:4011``: both parcel routines read
  ``zntmp``, which ``src/cm1.F:2192-2194`` starts at ``1.0e-6`` everywhere and
  ``src/turb.F:1161,1233-1237`` refreshes from ``bc2d``-filled ``znt`` over
  ``i = 0..ni+1``, ``j = 0..nj+1`` only.
* ``src/solve2.F:1290-1301`` and ``src/solve3.F:438-469,768-769``: the parcel
  step reads the acoustic small-step average saved at ``nrk == nrkmax`` and
  runs once at the end of each large step -- in the child, each *child* step.
* ``src/solve3.F:498-508`` with ``src/bc.F:528-1100`` ``bcu``/``bcv``/``bcw``:
  the lateral halos of that average before ``parcel_driver`` reads it.
* ``src/parcel.F:656-1510`` ``parcel_interp``: the derived fields
  (``:732-876``, with ``src/constants.F:112-133`` and ``src/base.F:3209``),
  the corner vorticity ``:1061-1072`` and the per-attribute stencils
  ``:1134-1510``.

The child's deck is CM1's own ``zf``/``zh``/``mf`` (``param.F:8162-8209``,
:mod:`cm1core.cm1_vertical_grid`), so the search runs over the child's layer
faces exactly as ``parcel.F:253`` runs over any stretched CM1 grid.

WHAT IS GM1'S, AND WHY (every item is in :data:`GM1_CHOICES` and the receipt)
-----------------------------------------------------------------------------
* **Coordinates.**  A position is metres from the parent grid's origin --
  the parent's ``xf(1)``, ``yf(1)`` and the ground -- in the parent's
  translating frame; the frame the runtime publishes ``[lagrangian]`` parent
  parcels in when ``[initialization].iorigin = 1``, and that frame shifted by
  ``(nx*dx/2, ny*dy/2)`` when it is 2.  It is never a
  child index, so a nest move (whole parent cells, :mod:`cm1core.nest_move`)
  changes which lattice a parcel is sampled on and nothing about where it is.
* **Hand-off, not termination, at the child boundary.**  At every parent-step
  boundary -- after the child's steps, the feedback and any move, with both
  domains at the same model time -- a child-owned parcel outside the child's
  *free interior* (the lattice less ``spec_zone + relax_zone`` rows, whose
  winds are the parent's interpolated, not the child's) is handed to the
  parent, and a parent-owned parcel inside it is handed to the child.  From
  then on it takes whole CM1 steps on that grid, with that grid's step and
  that step's own advecting average.  Terminating instead would end every
  inflow trajectory at the edge of a storm-following nest, which is where the
  air a tornado ingests comes from; handing off mid-interval would need a
  partial parent step, which ``parcel.F`` does not have.  Each published
  sample names the grid it was taken on.
* **Termination** happens only where ``parcel.F`` has no defined answer: a
  child-owned parcel that leaves the child lattice *between* parent-step
  boundaries (``parcel.F:194`` would skip it: not this grid's parcel), a
  position whose vertical stencil would read past the deck's ghost levels, or
  a non-finite position.  The reason and model time are recorded.
* **Search.**  ``parcel.F``'s scans are linear.  Here a closed-form estimate
  is corrected by exact comparisons against the same face words; for a
  monotone face array that returns the index the scan returns (the largest
  admissible cell), and the g++ harness proves it on faces hit exactly.
* **Diagnostic halos and ghosts.**  ``parcel_interp`` fills no lateral halo
  of ``buoy``/``vpg``/``th``/``t`` and no vertical ghost of any scalar in
  serial CM1.  A sample that reaches one takes ``bc.F`` ``bcs``'s rule for the
  grid's lateral boundary and the nearest physical level vertically.  A
  child-owned parcel is sampled only at parent-step boundaries, inside the
  free interior, so it never reaches a lateral halo.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

#: v2 adds ``parcel_attributes``: the layout record with the switches as
#: ``param.F:6239-6282`` resolved them for this run.
NEST_PARCEL_SCHEMA = "gm1.cartesian-nest-parcels/v2"
NEST_PARCEL_KERNEL_ABI = "gm1.cartesian-nest-parcel-kernels/v2"
NEST_CHILD_TRACK_DIRECTORY = "nest-child-lagrangian-track"
NEST_PARCEL_ARENA_FILENAME = "parcel-attributes.f32"
NEST_PARCEL_TABLE_FILENAME = "parcel-attribute-table.tsv"
NEST_PARCEL_EVENTS_FILENAME = "parcel-events.tsv"
NEST_PARCEL_RECEIPT_FILENAME = "parcel-track.json"
NEST_CHILD_PLACEMENT_DIRECTORY = "placements"
NEST_CHILD_PLACEMENT_SCHEMA = "gm1.cartesian-nest-child-placement/v1"

#: The one coordinate frame a child seed box may be declared in.
SEED_FRAME_PARENT_GRID_ORIGIN = "parent-grid-origin"

OWNER_TERMINATED = 0
OWNER_CHILD = 1
OWNER_PARENT = 2
OWNER_NAMES = {OWNER_TERMINATED: "terminated", OWNER_CHILD: "child", OWNER_PARENT: "parent"}

STATUS_OK = 0
STATUS_LEFT_LATTICE = 1
STATUS_LEFT_DECK = 2
STATUS_NONFINITE = 3
STATUS_REASONS = {
    STATUS_LEFT_LATTICE: (
        "left the lattice it was owned by between parent-step boundaries; "
        "parcel.F:194 would not treat it as this grid's parcel"
    ),
    STATUS_LEFT_DECK: (
        "its vertical stencil would read past the deck's ghost levels "
        "(parcel.F:253 has no bound there)"
    ),
    STATUS_NONFINITE: "its position became non-finite",
}

SENTINEL_F32 = -1.0e30
EVENT_ROW_BUDGET = 4096
NEST_PARCEL_THREADS = 256

CM1_AUTHORITY: Mapping[str, str] = {
    "vertical_ghost_fill": "src/parcel.F:59-130",
    "horizontal_search_first_stage": "src/parcel.F:153-183",
    "horizontal_search_second_stage": "src/parcel.F:203-231",
    "vertical_search": "src/parcel.F:251-260",
    "u_point_weights": "src/parcel.F:278-338",
    "v_point_weights": "src/parcel.F:343-401",
    "w_point_weights": "src/parcel.F:406-461",
    "semi_slip_log_layer": "src/parcel.F:470-515 (parcel_driver); src/parcel.F:1515-1583 (parcel_interp)",
    "roughness_carrier": "src/solve3.F:769, src/cm1.F:4011 (zntmp); src/cm1.F:2192-2194; src/turb.F:1161,1233-1237; src/bc.F:12-172 (bc2d)",
    "get2d": "src/parcel.F:1719-1787",
    "tri_interp": "src/parcel.F:1693-1711",
    "rk2_update": "src/parcel.F:521-568",
    "periodic_exit_parent_only": "src/parcel.F:577-593",
    "advecting_velocity": "src/solve2.F:1290-1301; src/solve3.F:438-469",
    "call_site": "src/solve3.F:768-769",
    "lateral_halos": "src/solve3.F:498-508; src/bc.F:528-1100 (bcu, bcv, bcw)",
    "diagnostic_halos": "src/bc.F:180-300 (bcs)",
    "derived_fields": "src/parcel.F:732-876; src/constants.F:112-133; src/base.F:3209; src/param.F:6327",
    "corner_vorticity": "src/parcel.F:1061-1072",
    "attribute_stencils": "src/parcel.F:1134-1510",
    "deck": "src/param.F:8162-8209 (zf, zh, mf ghosts)",
}

GM1_CHOICES: Mapping[str, str] = {
    "coordinates": (
        "metres from the parent grid origin (parent xf(1), yf(1), ground) in the "
        "parent's translating frame; ground position is x + domainlocx; [lagrangian] "
        "parent parcels share it only when [initialization].iorigin = 1"
    ),
    "hand_off": (
        "at each parent-step boundary, after feedback and any nest move: child-owned "
        "parcels outside the child's free interior (lattice less spec_zone+relax_zone "
        "rows) go to the parent, parent-owned parcels inside it go to the child; each "
        "then takes whole CM1 steps on its grid"
    ),
    "termination": (
        "only for a parcel parcel.F has no answer for: left its lattice between "
        "boundaries, a vertical stencil past the deck ghosts, a non-finite position"
    ),
    "search": (
        "closed-form estimate corrected by exact face comparisons; returns parcel.F's "
        "linear-scan index for a monotone face array"
    ),
    "diagnostic_halos": (
        "a sample reaching a lateral halo parcel_interp leaves unfilled takes bc.F bcs's "
        "rule for that side; a vertical ghost of a scalar takes the nearest level"
    ),
    "frame_cadence": "prclfrq_s is a whole number of parent steps; frames sit on parent-step boundaries",
    "sampled_winds": "u, v, w from u3d/v3d/w3d, as parcel_interp:1218-1343 does",
}

UNVALIDATED_MARKER = (
    "UNVALIDATED against CM1: CM1 r21.1 has no nesting, so a parcel handed between "
    "two grids has no CM1 counterpart.  The per-step arithmetic is parcel.F's and is "
    "proved raw-word against the packaged kernel text; the hand-off rule is GM1's."
)

#: constants.F:112-133 as CM1 evaluates them: default-REAL words and REAL
#: arithmetic at run time (set_constants), not double-precision literals.
def _cm1_constants(np: Any) -> dict[str, Any]:
    rd = np.float32(287.04)
    rv = np.float32(461.5)
    reps = np.float32(rv / rd)
    return {
        "g": np.float32(9.81),
        "cp": np.float32(1005.7),
        "reps": reps,
        "repsm1": np.float32(reps - np.float32(1.0)),
    }


def _np() -> Any:
    return importlib.import_module("numpy")


def _is_cupy(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "cupy"


def _host(xp: Any, value: Any) -> Any:
    return xp.asnumpy(value) if _is_cupy(xp) else value


# ---------------------------------------------------------------------------
# One CM1 grid as a parcel sees it.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ParcelDeck:
    """The words ``parcel.F`` reads about one grid: lattice, origin, deck, laterals.

    ``minx_m``/``miny_m`` are the grid's ``xf(1)``/``yf(1)`` in the parent-origin
    frame.  ``zf`` is CM1-indexed ``0..nz+2``, ``zh`` ``0..nz+1``, ``mf``
    ``0..nz+2``.  ``lateral`` is ``(wbc, ebc, sbc, nbc)``; ``periodic`` is
    whether ``parcel.F:577-593``'s exit applies (the outermost grid only).
    """

    name: str
    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    minx_m: float
    miny_m: float
    dz_reference_m: float
    zf: Any
    zh: Any
    mf: Any
    cgs: tuple[float, float, float]
    cgt: tuple[float, float, float]
    lateral: tuple[int, int, int, int]
    periodic: bool
    vertical_sha256: str

    @property
    def maxz_m(self) -> float:
        """``param.F:8193`` ``maxz = sigmaf(nk+1)``, which is ``zf(nk+1)`` without terrain."""

        return float(self.zf[self.nz + 1])

    def face_x(self, i: int) -> float:
        np = _np()
        return float(np.float32(self.minx_m) + np.float32(i - 1) * np.float32(self.dx_m))

    def face_y(self, j: int) -> float:
        np = _np()
        return float(np.float32(self.miny_m) + np.float32(j - 1) * np.float32(self.dy_m))

    def record(self) -> dict[str, Any]:
        return {
            "grid": self.name, "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "dx_m": self.dx_m, "dy_m": self.dy_m,
            "xf1_m": self.minx_m, "yf1_m": self.miny_m,
            "maxz_m": self.maxz_m, "lateral_wbc_ebc_sbc_nbc": list(self.lateral),
            "periodic_exit": self.periodic, "vertical_sha256": self.vertical_sha256,
            "lowest_layer_m": float(self.zf[2]) - float(self.zf[1]),
        }


def _levels_words(levels: Any) -> tuple[Any, Any, Any]:
    np = _np()
    zf = np.ascontiguousarray(np.asarray(levels.zf, dtype=np.float32))
    zh = np.ascontiguousarray(np.asarray(levels.zh, dtype=np.float32))
    mf = np.ascontiguousarray(np.asarray(levels.mf, dtype=np.float32))
    nk = int(levels.nk)
    if zf.shape != (nk + 3,) or zh.shape != (nk + 2,) or mf.shape != (nk + 3,):
        raise ValueError("the vertical coordinate does not carry CM1's ghosted zf/zh/mf")
    return zf, zh, mf


def deck_from_levels(*, name: str, nx: int, ny: int, dx_m: float, dy_m: float,
                     minx_m: float, miny_m: float, levels: Any,
                     lateral: tuple[int, int, int, int], periodic: bool) -> ParcelDeck:
    np = _np()
    zf, zh, mf = _levels_words(levels)
    for side in lateral:
        if side not in (1, 2):
            raise ValueError(
                f"lateral boundary {side} has no parcel halo rule here; bc.F's rigid "
                "branches (3, 4) negate the interior and are not transcribed"
            )
    return ParcelDeck(
        name=name, nx=int(nx), ny=int(ny), nz=int(levels.nk),
        dx_m=float(np.float32(dx_m)), dy_m=float(np.float32(dy_m)),
        minx_m=float(np.float32(minx_m)), miny_m=float(np.float32(miny_m)),
        dz_reference_m=float(levels.dz_reference_m), zf=zf, zh=zh, mf=mf,
        cgs=tuple(float(np.float32(v)) for v in levels.cgs),  # type: ignore[arg-type]
        cgt=tuple(float(np.float32(v)) for v in levels.cgt),  # type: ignore[arg-type]
        lateral=tuple(int(v) for v in lateral),  # type: ignore[arg-type]
        periodic=bool(periodic), vertical_sha256=str(levels.sha256),
    )


def child_deck(geometry: Any, child_grid: Any) -> ParcelDeck:
    """The child as a CM1 grid: its lattice, its origin on the parent, its deck.

    Its laterals are open (``nest_live.child_runtime_config`` seals ``wbc..nbc
    = 2``) and it has no periodic exit: a parcel leaving it is handed off.
    """

    from .nest_vertical import levels_of_grid

    return deck_from_levels(
        name="child", nx=geometry.nx, ny=geometry.ny, dx_m=geometry.dx_m, dy_m=geometry.dy_m,
        minx_m=geometry.origin_x_m, miny_m=geometry.origin_y_m,
        levels=levels_of_grid(child_grid), lateral=(2, 2, 2, 2), periodic=False,
    )


def parent_deck(parent_config: Any) -> ParcelDeck:
    """The parent with ``xf(1) = 0`` and ``parcel.F:577-593``'s periodic exit."""

    from .nest_vertical import levels_of_grid

    g = parent_config.grid
    lateral = tuple(int(getattr(parent_config, name, 2)) for name in ("wbc", "ebc", "sbc", "nbc"))
    return deck_from_levels(
        name="parent", nx=g.nx, ny=g.ny, dx_m=g.dx_m, dy_m=g.dy_m, minx_m=0.0, miny_m=0.0,
        levels=levels_of_grid(g), lateral=lateral, periodic=True,  # type: ignore[arg-type]
    )


def free_interior_m(geometry: Any) -> tuple[float, float, float, float]:
    """``(x_lo, x_hi, y_lo, y_hi)``: the child less its specified and relaxation rows.

    Those rows carry the parent's interpolated state, not the child's own
    integration, so a parcel there would be sampling parent winds on a child
    lattice and publishing them as child samples.
    """

    width = int(geometry.spec_zone) + int(geometry.relax_zone)
    return (
        float(geometry.origin_x_m) + width * float(geometry.dx_m),
        float(geometry.origin_x_m) + (int(geometry.nx) - width) * float(geometry.dx_m),
        float(geometry.origin_y_m) + width * float(geometry.dy_m),
        float(geometry.origin_y_m) + (int(geometry.ny) - width) * float(geometry.dy_m),
    )


def inside_free_interior(np: Any, geometry: Any, x: Any, y: Any) -> Any:
    x_lo, x_hi, y_lo, y_hi = free_interior_m(geometry)
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    return (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)


# ---------------------------------------------------------------------------
# Lateral halos and vertical ghosts of an advecting-velocity snapshot.
# ---------------------------------------------------------------------------
def fill_velocity_halos(xp: Any, u: Any, v: Any, w: Any, deck: ParcelDeck, *, ng: int) -> None:
    """``bc.F`` ``bcu``/``bcv``/``bcw`` for ``wbc..nbc`` in {1, 2}, in bc.F's order.

    Carriers are GM1's padded ``u (nz+2, ny+2ng, nx+1+2ng)``, ``v (nz+2,
    ny+1+2ng, nx+2ng)``, ``w (nz+3, ny+2ng, nx+2ng)``; CM1 ``(i, j, k)`` is
    ``[k, j-1+ng, i-1+ng]``.
    """

    del xp
    nx, ny, nz = deck.nx, deck.ny, deck.nz
    wbc, ebc, sbc, nbc = deck.lateral
    ks = slice(1, nz + 1)
    # bcu (bc.F:528-720): west, east, south, north.
    jr = slice(ng - 1, ny + ng + 1)  # j = 0..nj+1
    if wbc == 1:
        for i in range(1, ng + 1):
            u[ks, :, ng - i] = u[ks, :, nx + ng - i]
    else:
        for c in range(0, ng):
            u[ks, jr, c] = u[ks, jr, ng]
    if ebc == 1:
        for i in range(1, ng + 1):
            u[ks, :, nx + ng + i] = u[ks, :, ng + i]
    else:
        for c in range(nx + ng + 1, nx + 2 * ng + 1):
            u[ks, jr, c] = u[ks, jr, nx + ng]
    ir = slice(ng - 1, nx + ng + 2)  # i = 0..ni+2
    if sbc == 1:
        for j in range(1, ng + 1):
            u[ks, ng - j, :] = u[ks, ny + ng - j, :]
    else:
        for c in range(0, ng):
            u[ks, c, ir] = u[ks, ng, ir]
    if nbc == 1:
        for j in range(1, ng + 1):
            u[ks, ny + ng - 1 + j, :] = u[ks, j - 1 + ng, :]
    else:
        for c in range(ny + ng, ny + 2 * ng):
            u[ks, c, ir] = u[ks, ny + ng - 1, ir]
    # bcv (bc.F:722-940): south, north, west, east.
    ir = slice(ng - 1, nx + ng + 1)  # i = 0..ni+1
    if sbc == 1:
        for j in range(1, ng + 1):
            v[ks, ng - j, :] = v[ks, ny + ng - j, :]
    else:
        for c in range(0, ng):
            v[ks, c, ir] = v[ks, ng, ir]
    if nbc == 1:
        for j in range(1, ng + 1):
            v[ks, ny + ng + j, :] = v[ks, ng + j, :]
    else:
        for c in range(ny + ng + 1, ny + 2 * ng + 1):
            v[ks, c, ir] = v[ks, ny + ng, ir]
    jr = slice(ng - 1, ny + ng + 2)  # j = 0..nj+2
    if wbc == 1:
        for i in range(1, ng + 1):
            v[ks, :, ng - i] = v[ks, :, nx + ng - i]
    else:
        for c in range(0, ng):
            v[ks, jr, c] = v[ks, jr, ng]
    if ebc == 1:
        for i in range(1, ng + 1):
            v[ks, :, nx + ng - 1 + i] = v[ks, :, i - 1 + ng]
    else:
        for c in range(nx + ng, nx + 2 * ng):
            v[ks, jr, c] = v[ks, jr, nx + ng - 1]
    # bcw (bc.F:942-1100): k = 2..nk; west, east, south, north.
    kw = slice(2, nz + 1)
    jr = slice(ng - 1, ny + ng + 1)
    if wbc == 1:
        for i in range(1, ng + 1):
            w[kw, :, ng - i] = w[kw, :, nx + ng - i]
    else:
        for c in range(0, ng):
            w[kw, jr, c] = w[kw, jr, ng]
    if ebc == 1:
        for i in range(1, ng + 1):
            w[kw, :, nx + ng - 1 + i] = w[kw, :, i - 1 + ng]
    else:
        for c in range(nx + ng, nx + 2 * ng):
            w[kw, jr, c] = w[kw, jr, nx + ng - 1]
    ir = slice(ng - 1, nx + ng + 1)
    if sbc == 1:
        for j in range(1, ng + 1):
            w[kw, ng - j, :] = w[kw, ny + ng - j, :]
    else:
        for c in range(0, ng):
            w[kw, c, ir] = w[kw, ng, ir]
    if nbc == 1:
        for j in range(1, ng + 1):
            w[kw, ny + ng - 1 + j, :] = w[kw, j - 1 + ng, :]
    else:
        for c in range(ny + ng, ny + 2 * ng):
            w[kw, c, ir] = w[kw, ny + ng - 1, ir]


def fill_velocity_bottom_ghost(xp: Any, u: Any, v: Any, deck: ParcelDeck, *,
                               bbc: int, u_move_m_s: float, v_move_m_s: float) -> None:
    """``parcel.F:59-96``: the ``k = 0`` ghost of ``u`` and ``v`` alone.

    The lower branch reads only levels 1, 2 and 3, so it is local to the
    bottom of whatever block it is given: a sub-volume of a grid that reaches
    the ground gets the same words here as the whole grid would
    (:mod:`cm1core.nest_child_window` depends on exactly that).  The upper
    ghost is not local in the same way -- ``cgt`` is derived from the top three
    levels of the grid it belongs to -- so it stays in
    :func:`fill_velocity_ghosts`.
    """

    f32 = xp.float32
    c1, c2, c3 = (f32(c) for c in deck.cgs)
    if bbc == 1:
        for a in (u, v):
            a[0] = (c1 * a[1] + c2 * a[2]) + c3 * a[3]
    elif bbc == 2:
        u[0] = f32(0.0) - f32(u_move_m_s)
        v[0] = f32(0.0) - f32(v_move_m_s)
    elif bbc == 3:
        # parcel.F:94-96: the level is left alone; a parcel below zh(1) takes
        # u and v from the log layer at sample time (parcel.F:470-515), so the
        # word the trilinear stencil reads there is never kept.
        pass
    else:
        raise ValueError(f"bbc={bbc} has no parcel.F lower-boundary branch")


def fill_velocity_ghosts(xp: Any, u: Any, v: Any, w: Any, deck: ParcelDeck, *,
                         bbc: int, tbc: int, u_move_m_s: float, v_move_m_s: float) -> None:
    """``parcel.F:59-130`` on whole planes, with this deck's ``cgs``/``cgt``."""

    f32 = xp.float32
    nz = deck.nz
    t1, t2, t3 = (f32(c) for c in deck.cgt)
    fill_velocity_bottom_ghost(xp, u, v, deck, bbc=bbc, u_move_m_s=u_move_m_s,
                               v_move_m_s=v_move_m_s)
    if tbc == 1:
        for a in (u, v):
            a[nz + 1] = (t1 * a[nz] + t2 * a[nz - 1]) + t3 * a[nz - 2]
    elif tbc == 2:
        u[nz + 1] = f32(0.0)
        v[nz + 1] = f32(0.0)
    else:
        raise ValueError(f"tbc={tbc} has no parcel.F upper-boundary branch")
    w[nz + 1] = f32(0.0)


# ---------------------------------------------------------------------------
# The semi-slip lower boundary: what parcel.F:470-515 reads beyond the winds.
# ---------------------------------------------------------------------------
#: ``cm1.F:2192-2194``: every ``zntmp`` word starts at ``tem = 1.0e-6``, and
#: ``turb.F:1233-1237`` refreshes only ``i = 0..ni+1``, ``j = 0..nj+1``, so a
#: wider halo keeps this word for the whole run.
CM1_ZNTMP_INITIAL_F32 = 1.0e-6


@dataclass(frozen=True)
class SemiSlip:
    """``bbc = 3``: CM1's ``zntmp`` on a padded 2-D carrier and the frame motion.

    ``zntmp`` is ``(ny + 2ng, nx + 2ng)`` with CM1 ``(i, j)`` at
    ``[j-1+ng, i-1+ng]``, built by :func:`pad_roughness`.
    """

    zntmp: Any
    imove: int
    u_move_m_s: float
    v_move_m_s: float


def pad_roughness(xp: Any, znt: Any, deck: ParcelDeck, *, ng: int) -> Any:
    """``turb.F:1161`` ``bc2d(znt)`` then ``:1233-1237``'s copy into ``zntmp``.

    ``bc.F:12-172`` fills west and east over every row, then south and north
    over every column, so the corner words are the south/north rule applied to
    the column the west/east rule just wrote.  Only the first ring is copied
    into ``zntmp``; the rest keep ``cm1.F:2194``'s ``1.0e-6``.
    """

    nx, ny = deck.nx, deck.ny
    if tuple(int(v) for v in znt.shape) != (ny, nx):
        raise ValueError(f"the roughness carrier is {tuple(znt.shape)}, not this deck's ({ny}, {nx})")
    f32 = xp.float32
    ring = xp.empty((ny + 2, nx + 2), dtype=f32)
    ring[1:ny + 1, 1:nx + 1] = znt
    wbc, ebc, sbc, nbc = deck.lateral
    ring[1:ny + 1, 0] = znt[:, nx - 1] if wbc == 1 else znt[:, 0]
    ring[1:ny + 1, nx + 1] = znt[:, 0] if ebc == 1 else znt[:, nx - 1]
    south = ring[ny, :].copy() if sbc == 1 else ring[1, :].copy()
    north = ring[1, :].copy() if nbc == 1 else ring[ny, :].copy()
    ring[0, :] = south
    ring[ny + 1, :] = north
    out = xp.full((ny + 2 * ng, nx + 2 * ng), f32(CM1_ZNTMP_INITIAL_F32), dtype=f32)
    out[ng - 1:ng + ny + 1, ng - 1:ng + nx + 1] = ring
    return out


def parcel_roughness(runtime: Any) -> Any:
    """The grid's ``closure.surface.znt`` after this step's ``getcecd``, or ``None``."""

    closure = getattr(runtime, "closure", None)
    state = getattr(closure, "surface_state", None) or {}
    return state.get("znt")


def semi_slip_for(xp: Any, config: Any, deck: ParcelDeck, roughness: Any, *, ng: int) -> "SemiSlip | None":
    """The :class:`SemiSlip` a ``bbc = 3`` grid's parcel step needs, else ``None``."""

    if int(config.bbc) != 3:
        return None
    if roughness is None:
        raise RuntimeError(
            "bbc=3 but this grid carries no surface roughness: parcel.F:470-515 takes a parcel "
            "below zh(1) from the log layer over zntmp, and without it every such parcel would "
            "take the trilinear wind through a ghost level nothing fills"
        )
    u_move, v_move = float(config.u_move_m_s), float(config.v_move_m_s)
    imove = getattr(config, "imove", None)
    if imove is None:
        imove = int(u_move != 0.0 or v_move != 0.0)
    return SemiSlip(pad_roughness(xp, roughness, deck, ng=ng), int(imove), u_move, v_move)


def _get2d(np: Any, s00: Any, s10: Any, s01: Any, s11: Any, rx: Any, ry: Any) -> Any:
    """``parcel.F:1759-1783``: ``w1, w2, w3, w7`` and the sum in written order."""

    one = np.float32(1.0)
    mx, my = one - rx, one - ry
    return ((s00 * (mx * my) + s10 * (rx * my)) + s01 * (mx * ry)) + s11 * (rx * ry)


def _get2d_padded(np: Any, s: Any, j: Any, i: Any, rx: Any, ry: Any) -> Any:
    """``get2d`` on a padded 2-D carrier at storage ``(j, i)``."""

    return _get2d(np, s[j, i], s[j, i + 1], s[j + 1, i], s[j + 1, i + 1], rx, ry)


def libm_logf(np: Any, values: Any) -> Any:
    """``alog`` on default REAL: the platform's binary32 ``logf``, element by element.

    NumPy's float32 log is its own SIMD kernel, not the ``logf`` gfortran's
    ``alog`` and the g++-compiled kernel text call.
    """

    from .cuda_surface_drag import _host_libm_f32

    logf = _host_libm_f32()[0]
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.empty_like(flat)
    for q in range(flat.size):
        out[q] = logf(np, flat[q])
    return out.reshape(np.shape(values))


def _log_layer_ratio(np: Any, z: Any, z0: Any, zh1: Any) -> Any:
    """``parcel.F:505-506``: ``rznt = 1.0/z0``; ``alog((z3d+z0)*rznt)/alog((zh(1)+z0)*rznt)``."""

    rznt = np.float32(1.0) / z0
    return libm_logf(np, (z + z0) * rznt) / libm_logf(np, (zh1 + z0) * rznt)


# ---------------------------------------------------------------------------
# Host reference: vectorized, in the kernel's IEEE order.
# ---------------------------------------------------------------------------
def _face(np: Any, origin: float, spacing: float, i: Any) -> Any:
    return np.float32(origin) + (i - 1).astype(np.float32) * np.float32(spacing)


def _half(np: Any, origin: float, spacing: float, i: Any) -> Any:
    return np.float32(origin) + (i.astype(np.float32) - np.float32(0.5)) * np.float32(spacing)


def reference_search(np: Any, c: Any, origin: float, spacing: float, n: int,
                     lo: int, hi: int) -> Any:
    """The largest ``i`` in ``[lo, hi]`` with ``face(i) <= c <= face(i+1)``, else ``-100``."""

    c = np.asarray(c, dtype=np.float32)
    q = np.floor((c - np.float32(origin)) / np.float32(spacing))
    q = np.where(q < np.float32(-4.0), np.float32(-4.0), q)
    q = np.where(q > np.float32(n + 4), np.float32(n + 4), q)
    e = q.astype(np.int64) + 1
    e = np.minimum(np.maximum(e, lo), hi)
    for _ in range(64):
        up = (e < hi) & (_face(np, origin, spacing, e + 1) <= c)
        if not bool(up.any()):
            break
        e = e + up
    else:  # pragma: no cover - a monotone face array converges in two passes
        raise RuntimeError("the face search did not converge")
    for _ in range(64):
        down = (e >= lo) & (_face(np, origin, spacing, e) > c)
        if not bool(down.any()):
            break
        e = e - down
    else:  # pragma: no cover
        raise RuntimeError("the face search did not converge")
    ok = (e >= lo) & ~(c > _face(np, origin, spacing, e + 1))
    return np.where(ok, e, -100)


def _weights(np: Any, rx: Any, ry: Any, rz: Any) -> tuple[Any, ...]:
    one = np.float32(1.0)
    mx, my, mz = one - rx, one - ry, one - rz
    return (
        (mx * my) * mz, (rx * my) * mz, (mx * ry) * mz, (mx * my) * rz,
        (rx * my) * rz, (mx * ry) * rz, (rx * ry) * mz, (rx * ry) * rz,
    )


#: ``tri_interp``'s eight corners as ``(di, dj, dk)`` in written order.
TRI_CORNERS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 1, 1),
)


def _tri(np: Any, corners: Sequence[Any], weights: Sequence[Any]) -> Any:
    total = corners[0] * weights[0]
    for q in range(1, 8):
        total = total + corners[q] * weights[q]
    return total


@dataclass(frozen=True)
class Stencils:
    """``parcel.F``'s cell flags, stencil anchors and saved ratios for many parcels."""

    status: Any
    iflag: Any
    jflag: Any
    kflag: Any
    iu: Any   # u point anchors (i, j, k)
    ju: Any
    ks: Any   # the scalar-level k used by u, v, scalar and zvort samples
    iv: Any   # v/w/scalar i anchor
    rxu: Any
    rys: Any
    rzs: Any
    rxs: Any
    ryv: Any
    rzw: Any


def reference_stencils(np: Any, deck: ParcelDeck, x: Any, y: Any, z: Any, *,
                       second_stage: bool) -> Stencils:
    """``parcel.F:153-461`` (and ``:1134-1343``): flags, anchors and ratios."""

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    nx, ny, nz = deck.nx, deck.ny, deck.nz
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    safe_x = np.where(finite, x, np.float32(deck.minx_m))
    safe_y = np.where(finite, y, np.float32(deck.miny_m))
    safe_z = np.where(finite, z, np.float32(0.0))
    shift = 1 if second_stage else 0
    iflag = reference_search(np, safe_x, deck.minx_m, deck.dx_m, nx, 1 - shift, nx + shift)
    jflag = reference_search(np, safe_y, deck.miny_m, deck.dy_m, ny, 1 - shift, ny + shift)
    zf, zh = deck.zf, deck.zh
    count = np.searchsorted(zf[: nz + 2], safe_z, side="right")
    kflag = np.maximum(count.astype(np.int64) - 1, 1)
    deck_bad = (kflag == nz + 1) & (safe_z >= zf[nz + 2])
    lattice_bad = (iflag < 0) | (jflag < 0)
    ic = np.where(lattice_bad, 1, iflag)
    jc = np.where(lattice_bad, 1, jflag)
    kc = np.where(deck_bad, nz, kflag)
    ju = np.where(safe_y < _half(np, deck.miny_m, deck.dy_m, jc), jc - 1, jc)
    ks = np.where(safe_z < zh[kc], kc - 1, kc)
    deck_bad = deck_bad | (ks > nz)
    ks = np.minimum(ks, nz)
    rzs = (safe_z - zh[ks]) / (zh[ks + 1] - zh[ks])
    xfi, xfi1 = _face(np, deck.minx_m, deck.dx_m, ic), _face(np, deck.minx_m, deck.dx_m, ic + 1)
    rxu = (safe_x - xfi) / (xfi1 - xfi)
    yhj, yhj1 = _half(np, deck.miny_m, deck.dy_m, ju), _half(np, deck.miny_m, deck.dy_m, ju + 1)
    rys = (safe_y - yhj) / (yhj1 - yhj)
    iv = np.where(safe_x < _half(np, deck.minx_m, deck.dx_m, ic), ic - 1, ic)
    xhi, xhi1 = _half(np, deck.minx_m, deck.dx_m, iv), _half(np, deck.minx_m, deck.dx_m, iv + 1)
    rxs = (safe_x - xhi) / (xhi1 - xhi)
    yfj, yfj1 = _face(np, deck.miny_m, deck.dy_m, jc), _face(np, deck.miny_m, deck.dy_m, jc + 1)
    ryv = (safe_y - yfj) / (yfj1 - yfj)
    zfk, zfk1 = zf[kc], zf[kc + 1]
    rzw = (safe_z - zfk) / (zfk1 - zfk)
    status = np.where(
        ~finite, STATUS_NONFINITE,
        np.where(lattice_bad, STATUS_LEFT_LATTICE, np.where(deck_bad, STATUS_LEFT_DECK, STATUS_OK)),
    ).astype(np.int32)
    return Stencils(status, ic, jc, kc, ic, ju, ks, iv, rxu, rys, rzs, rxs, ryv, rzw)


def _padded_gather(np: Any, array: Any, k: Any, j: Any, i: Any, ng: int) -> Any:
    return array[k, j - 1 + ng, i - 1 + ng]


def reference_sample_uvw(np: Any, u: Any, v: Any, w: Any, deck: ParcelDeck,
                         x: Any, y: Any, z: Any, *, second_stage: bool, ng: int,
                         surface: "SemiSlip | None" = None) -> tuple[Any, Any, Any, Any]:
    """``parcel.F:251-515`` on host padded carriers: ``(u, v, w, status)``.

    ``surface`` is given exactly when ``bbc = 3``.
    """

    s = reference_stencils(np, deck, x, y, z, second_stage=second_stage)
    ok = s.status == STATUS_OK
    iu = np.where(ok, s.iu, 1)
    ju = np.where(ok, s.ju, 1)
    ks = np.where(ok, s.ks, 1)
    iv = np.where(ok, s.iv, 1)
    jv = np.where(ok, s.jflag, 1)
    kw = np.where(ok, s.kflag, 1)
    out = []
    for array, (ia, ja, ka), (rx, ry, rz) in (
        (u, (iu, ju, ks), (s.rxu, s.rys, s.rzs)),
        (v, (iv, jv, ks), (s.rxs, s.ryv, s.rzs)),
        (w, (iv, ju, kw), (s.rxs, s.rys, s.rzw)),
    ):
        weights = _weights(np, rx, ry, rz)
        corners = [_padded_gather(np, array, ka + dk, ja + dj, ia + di, ng)
                   for di, dj, dk in TRI_CORNERS]
        out.append(_tri(np, corners, weights))
    if surface is not None:
        # parcel.F:472-514.  The u point is (iflag, j) with j moved south of
        # yh, the v point (i, jflag) with i moved west of xh, z0 at (i, j):
        # exactly the anchors, and so the rx/ry words, of the 3-D samples.
        zz = np.asarray(z, dtype=np.float32)
        zh1 = np.float32(deck.zh[1])
        q = np.flatnonzero(ok & (np.where(ok, zz, zh1) < zh1))
        if q.size:
            level_u, level_v = u[1], v[1]
            uval = _get2d_padded(np, level_u, ju[q] - 1 + ng, iu[q] - 1 + ng, s.rxu[q], s.rys[q])
            vval = _get2d_padded(np, level_v, jv[q] - 1 + ng, iv[q] - 1 + ng, s.rxs[q], s.ryv[q])
            z0 = _get2d_padded(np, surface.zntmp, ju[q] - 1 + ng, iv[q] - 1 + ng, s.rxs[q], s.rys[q])
            var = _log_layer_ratio(np, zz[q], z0, zh1)
            if surface.imove == 1:
                umove, vmove = np.float32(surface.u_move_m_s), np.float32(surface.v_move_m_s)
                uval = (uval + umove) * var - umove
                vval = (vval + vmove) * var - vmove
            else:
                uval = uval * var
                vval = vval * var
            out[0] = np.array(out[0], dtype=np.float32, copy=True)
            out[1] = np.array(out[1], dtype=np.float32, copy=True)
            out[0][q] = uval
            out[1][q] = vval
    return out[0], out[1], out[2], s.status


def reference_rk2(np: Any, positions: Any, owner: Any, status: Any, *, code: int,
                  u: Any, v: Any, w: Any, deck: ParcelDeck, dt_s: float, ng: int,
                  surface: "SemiSlip | None" = None) -> tuple[Any, Any]:
    """``parcel.F:137-619`` for the parcels ``owner == code`` whose ``status`` is 0.

    Returns new ``(positions, status)``; every other parcel's words are
    returned unchanged.  ``dt_s`` carries its sign (negative is backward).
    ``surface`` is given exactly when ``bbc = 3``.
    """

    out = np.array(positions, dtype=np.float32, copy=True)
    st = np.array(status, dtype=np.int32, copy=True)
    x0, y0, z0 = out[0].copy(), out[1].copy(), out[2].copy()
    active = (np.asarray(owner) == code) & (st == 0)
    finite = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(z0)
    active &= finite
    active &= ~(x0 < np.float32(-1.0e29))
    if not bool(active.any()):
        return out, st
    dt = np.float32(dt_s)
    dt2 = dt / np.float32(2.0)
    u1, v1, w1, s1 = reference_sample_uvw(np, u, v, w, deck, x0, y0, z0,
                                          second_stage=False, ng=ng, surface=surface)
    fail = active & (s1 != 0)
    st[fail] = s1[fail]
    go = active & (s1 == 0)
    x1 = x0 + dt * u1
    y1 = y0 + dt * v1
    z1 = z0 + dt * w1
    u2, v2, w2, s2 = reference_sample_uvw(np, u, v, w, deck, x1, y1, z1,
                                          second_stage=True, ng=ng, surface=surface)
    fail = go & (s2 != 0)
    st[fail] = s2[fail]
    go &= s2 == 0
    x = x0 + dt2 * (u1 + u2)
    y = y0 + dt2 * (v1 + v2)
    zz = z0 + dt2 * (w1 + w2)
    zz = np.where(zz < np.float32(0.0), np.float32(1.0e-6), zz)
    maxz = np.float32(deck.maxz_m)
    zz = np.where(zz <= maxz, zz, maxz)
    if deck.periodic:
        minx, miny = np.float32(deck.minx_m), np.float32(deck.miny_m)
        maxx = minx + np.float32(deck.nx) * np.float32(deck.dx_m)
        maxy = miny + np.float32(deck.ny) * np.float32(deck.dy_m)
        x = np.where(x < minx, x + (maxx - minx), x)
        x = np.where(x > maxx, x - (maxx - minx), x)
        y = np.where(y > maxy, y - (maxy - miny), y)
        y = np.where(y < miny, y + (maxy - miny), y)
    out[0, go] = x[go]
    out[1, go] = y[go]
    out[2, go] = zz[go]
    st[go] = STATUS_OK
    return out, st


# ---------------------------------------------------------------------------
# The device kernel.
# ---------------------------------------------------------------------------
NEST_PARCEL_KERNEL_SOURCE = r'''// gm1.cartesian-nest-parcel-kernels/v2
// CM1 r21.1 src/parcel.F:10-620 parcel_driver (terrain_flag false) on one CM1
// grid with its own lattice origin and vertical deck.  cm1core.nest_parcels.
#define GM1_NP_IDX(k, j, i, nj, ni) \
    ((((unsigned long long)(k) * (unsigned long long)(nj)) + (unsigned long long)(j)) \
     * (unsigned long long)(ni) + (unsigned long long)(i))

__device__ __forceinline__ float gm1_np_a(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float gm1_np_s(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float gm1_np_m(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float gm1_np_d(float a, float b) { return __fdiv_rn(a, b); }

// xf(i) and xh(i) in the closed form cm1core.lagrangian_parcels' kernel uses.
__device__ __forceinline__ float gm1_np_face(float origin, float spacing, int i) {
    return gm1_np_a(origin, gm1_np_m((float)(i - 1), spacing));
}
__device__ __forceinline__ float gm1_np_half(float origin, float spacing, int i) {
    return gm1_np_a(origin, gm1_np_m(gm1_np_s((float)i, 0.5f), spacing));
}

// parcel.F:161-167 (lo=1, hi=n) and :209-230 (lo=0, hi=n+1): the largest i in
// [lo, hi] with face(i) <= c <= face(i+1), else -100.
__device__ int gm1_np_search(float c, float origin, float spacing, int n, int lo, int hi) {
    float q = floorf(gm1_np_d(gm1_np_s(c, origin), spacing));
    if (q < -4.0f) q = -4.0f;
    if (q > (float)(n + 4)) q = (float)(n + 4);
    int e = (int)q + 1;
    if (e < lo) e = lo;
    if (e > hi) e = hi;
    while (e < hi && gm1_np_face(origin, spacing, e + 1) <= c) ++e;
    while (e >= lo && gm1_np_face(origin, spacing, e) > c) --e;
    if (e < lo) return -100;
    if (c > gm1_np_face(origin, spacing, e + 1)) return -100;
    return e;
}

__device__ __forceinline__ void gm1_np_weights(float rx, float ry, float rz, float* w) {
    const float mx = gm1_np_s(1.0f, rx), my = gm1_np_s(1.0f, ry), mz = gm1_np_s(1.0f, rz);
    w[0] = gm1_np_m(gm1_np_m(mx, my), mz); w[1] = gm1_np_m(gm1_np_m(rx, my), mz);
    w[2] = gm1_np_m(gm1_np_m(mx, ry), mz); w[3] = gm1_np_m(gm1_np_m(mx, my), rz);
    w[4] = gm1_np_m(gm1_np_m(rx, my), rz); w[5] = gm1_np_m(gm1_np_m(mx, ry), rz);
    w[6] = gm1_np_m(gm1_np_m(rx, ry), mz); w[7] = gm1_np_m(gm1_np_m(rx, ry), rz);
}

// parcel.F:1702-1709 with CM1 (i,j,k) already mapped to owner indices.
__device__ float gm1_np_tri(const float* s, int k, int j, int i, int nj, int ni, const float* w) {
    float t = gm1_np_m(s[GM1_NP_IDX(k, j, i, nj, ni)], w[0]);
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k, j, i + 1, nj, ni)], w[1]));
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k, j + 1, i, nj, ni)], w[2]));
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k + 1, j, i, nj, ni)], w[3]));
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k + 1, j, i + 1, nj, ni)], w[4]));
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k + 1, j + 1, i, nj, ni)], w[5]));
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k, j + 1, i + 1, nj, ni)], w[6]));
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k + 1, j + 1, i + 1, nj, ni)], w[7]));
    return t;
}

// parcel.F:1719-1787 get2d: the four-term sum in written order on level k.
__device__ float gm1_np_get2d(const float* s, int k, int j, int i, int nj, int ni, float rx, float ry) {
    const float mx = gm1_np_s(1.0f, rx), my = gm1_np_s(1.0f, ry);
    float t = gm1_np_m(s[GM1_NP_IDX(k, j, i, nj, ni)], gm1_np_m(mx, my));
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k, j, i + 1, nj, ni)], gm1_np_m(rx, my)));
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k, j + 1, i, nj, ni)], gm1_np_m(mx, ry)));
    t = gm1_np_a(t, gm1_np_m(s[GM1_NP_IDX(k, j + 1, i + 1, nj, ni)], gm1_np_m(rx, ry)));
    return t;
}

// parcel.F:251-515: u, v, w at one position.  0, or the status that stops it:
// 1 outside the lattice, 2 past the deck's ghost levels, 3 non-finite.
// semi_slip is parcel.F:472's bbc.eq.3; znt is zntmp padded like w's columns.
__device__ int gm1_np_sample(const float* u, const float* v, const float* w,
                             const float* zf, const float* zh, const float* znt,
                             int nx, int ny, int nz, int ng, int semi_slip, int imove,
                             float dx, float dy, float minx, float miny, float umove, float vmove,
                             float x, float y, float z, int second,
                             float* uo, float* vo, float* wo) {
    if (!(isfinite(x) && isfinite(y) && isfinite(z))) return 3;
    const int iflag = gm1_np_search(x, minx, dx, nx, 1 - second, nx + second);
    const int jflag = gm1_np_search(y, miny, dy, ny, 1 - second, ny + second);
    if (iflag < 0 || jflag < 0) return 1;
    int kflag = 1;
    while (kflag <= nz && z >= zf[kflag + 1]) ++kflag;
    if (kflag == nz + 1 && z >= zf[nz + 2]) return 2;
    // u points, parcel.F:278-338
    int j = jflag;
    if (y < gm1_np_half(miny, dy, j)) j = j - 1;
    int k = kflag;
    if (z < zh[k]) k = k - 1;
    if (k > nz) return 2;
    const float rzs = gm1_np_d(gm1_np_s(z, zh[k]), gm1_np_s(zh[k + 1], zh[k]));
    const float xfi = gm1_np_face(minx, dx, iflag), xfi1 = gm1_np_face(minx, dx, iflag + 1);
    const float rxu = gm1_np_d(gm1_np_s(x, xfi), gm1_np_s(xfi1, xfi));
    const float yhj = gm1_np_half(miny, dy, j), yhj1 = gm1_np_half(miny, dy, j + 1);
    const float rys = gm1_np_d(gm1_np_s(y, yhj), gm1_np_s(yhj1, yhj));
    float wt[8];
    gm1_np_weights(rxu, rys, rzs, wt);
    *uo = gm1_np_tri(u, k, j - 1 + ng, iflag - 1 + ng, ny + 2 * ng, nx + 1 + 2 * ng, wt);
    // v points, parcel.F:343-401
    int i = iflag;
    if (x < gm1_np_half(minx, dx, i)) i = i - 1;
    const float xhi = gm1_np_half(minx, dx, i), xhi1 = gm1_np_half(minx, dx, i + 1);
    const float rxs = gm1_np_d(gm1_np_s(x, xhi), gm1_np_s(xhi1, xhi));
    const float yfj = gm1_np_face(miny, dy, jflag), yfj1 = gm1_np_face(miny, dy, jflag + 1);
    const float ryv = gm1_np_d(gm1_np_s(y, yfj), gm1_np_s(yfj1, yfj));
    gm1_np_weights(rxs, ryv, rzs, wt);
    *vo = gm1_np_tri(v, k, jflag - 1 + ng, i - 1 + ng, ny + 1 + 2 * ng, nx + 2 * ng, wt);
    // w points, parcel.F:406-461
    const float zfk = zf[kflag], zfk1 = zf[kflag + 1];
    const float rzw = gm1_np_d(gm1_np_s(z, zfk), gm1_np_s(zfk1, zfk));
    gm1_np_weights(rxs, rys, rzw, wt);
    *wo = gm1_np_tri(w, kflag, j - 1 + ng, i - 1 + ng, ny + 2 * ng, nx + 2 * ng, wt);
    // parcel.F:470-515: below zh(1) on a semi-slip grid, u and v at the lowest
    // level scaled by the neutral log layer over zntmp.  get2d's u anchor is
    // (iflag, j), v's (i, jflag), z0's (i, j): the 3-D samples' own anchors.
    if (semi_slip && z < zh[1]) {
        const float us = gm1_np_get2d(u, 1, j - 1 + ng, iflag - 1 + ng, ny + 2 * ng, nx + 1 + 2 * ng, rxu, rys);
        const float vs = gm1_np_get2d(v, 1, jflag - 1 + ng, i - 1 + ng, ny + 1 + 2 * ng, nx + 2 * ng, rxs, ryv);
        const float z0 = gm1_np_get2d(znt, 0, j - 1 + ng, i - 1 + ng, ny + 2 * ng, nx + 2 * ng, rxs, rys);
        const float rznt = gm1_np_d(1.0f, z0);
        const float var = gm1_np_d(logf(gm1_np_m(gm1_np_a(z, z0), rznt)),
                                   logf(gm1_np_m(gm1_np_a(zh[1], z0), rznt)));
        if (imove == 1) {
            *uo = gm1_np_s(gm1_np_m(gm1_np_a(us, umove), var), umove);
            *vo = gm1_np_s(gm1_np_m(gm1_np_a(vs, vmove), var), vmove);
        } else {
            *uo = gm1_np_m(us, var);
            *vo = gm1_np_m(vs, var);
        }
    }
    return 0;
}

// parcel.F:137-619 for the parcels owner[p] == code whose status is still 0.
extern "C" __global__ void gm1_nest_parcel_rk2_v2(
    float* __restrict__ px, float* __restrict__ py, float* __restrict__ pz,
    const int* __restrict__ owner, int* __restrict__ status,
    const float* __restrict__ u, const float* __restrict__ v, const float* __restrict__ w,
    const float* __restrict__ zf, const float* __restrict__ zh, const float* __restrict__ znt,
    int nparcels, int code, int nx, int ny, int nz, int ng, int periodic, int semi_slip, int imove,
    float dx, float dy, float minx, float miny, float maxz, float dt, float umove, float vmove)
{
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned long long)nparcels) return;
    const int p = (int)tid;
    if (owner[p] != code || status[p] != 0) return;
    const float x0 = px[p], y0 = py[p], z0 = pz[p];
    if (!(isfinite(x0) && isfinite(y0) && isfinite(z0))) return;
    if (x0 < -1.0e29f) return;
    const float dt2 = gm1_np_d(dt, 2.0f);
    float u1, v1, w1, u2, v2, w2;
    int st = gm1_np_sample(u, v, w, zf, zh, znt, nx, ny, nz, ng, semi_slip, imove, dx, dy, minx, miny,
                           umove, vmove, x0, y0, z0, 0, &u1, &v1, &w1);
    if (st != 0) { status[p] = st; return; }
    // parcel.F:526-537: the first stage is neither wrapped nor clamped.
    const float x1 = gm1_np_a(x0, gm1_np_m(dt, u1));
    const float y1 = gm1_np_a(y0, gm1_np_m(dt, v1));
    const float z1 = gm1_np_a(z0, gm1_np_m(dt, w1));
    st = gm1_np_sample(u, v, w, zf, zh, znt, nx, ny, nz, ng, semi_slip, imove, dx, dy, minx, miny,
                       umove, vmove, x1, y1, z1, 1, &u2, &v2, &w2);
    if (st != 0) { status[p] = st; return; }
    float x = gm1_np_a(x0, gm1_np_m(dt2, gm1_np_a(u1, u2)));
    float y = gm1_np_a(y0, gm1_np_m(dt2, gm1_np_a(v1, v2)));
    float z = gm1_np_a(z0, gm1_np_m(dt2, gm1_np_a(w1, w2)));
    if (z < 0.0f) z = 1.0e-6f;
    z = z <= maxz ? z : maxz;
    if (periodic) {
        const float maxx = gm1_np_a(minx, gm1_np_m((float)nx, dx));
        const float maxy = gm1_np_a(miny, gm1_np_m((float)ny, dy));
        if (x < minx) x = gm1_np_a(x, gm1_np_s(maxx, minx));
        if (x > maxx) x = gm1_np_s(x, gm1_np_s(maxx, minx));
        if (y > maxy) y = gm1_np_s(y, gm1_np_s(maxy, miny));
        if (y < miny) y = gm1_np_a(y, gm1_np_s(maxy, miny));
    }
    px[p] = x; py[p] = y; pz[p] = z;
}
'''
NEST_PARCEL_KERNEL_SHA256 = hashlib.sha256(NEST_PARCEL_KERNEL_SOURCE.encode("utf-8")).hexdigest()


def compile_nest_parcel_module(cupy: Any) -> Any:
    from .nest_live_kernels import NEST_LIVE_NVRTC_OPTIONS
    from .nvrtc_module import compile_nvrtc_module

    return compile_nvrtc_module(
        cupy, code=NEST_PARCEL_KERNEL_SOURCE, options=NEST_LIVE_NVRTC_OPTIONS,
        filename="gm1_nest_parcels.cu",
    )


@dataclass
class DeviceDeck:
    """A deck's ``zf``/``zh`` words on the device, uploaded once."""

    zf: Any
    zh: Any


def upload_deck(xp: Any, deck: ParcelDeck) -> DeviceDeck:
    return DeviceDeck(xp.asarray(deck.zf), xp.asarray(deck.zh))


def launch_nest_parcel_rk2(cupy: Any, module: Any, *, positions: Any, owner: Any, status: Any,
                           code: int, u: Any, v: Any, w: Any, deck: ParcelDeck,
                           device_deck: DeviceDeck, dt_s: float, ng: int, stream: Any = None,
                           surface: SemiSlip | None = None) -> None:
    n = int(positions.shape[1])
    for name, array, dtype in (("positions", positions, cupy.float32), ("owner", owner, cupy.int32),
                               ("status", status, cupy.int32)):
        if array.dtype != dtype or not array.flags.c_contiguous:
            raise ValueError(f"parcel {name} must be a C-contiguous {dtype} array")
    if tuple(u.shape) != (deck.nz + 2, deck.ny + 2 * ng, deck.nx + 1 + 2 * ng):
        raise ValueError("the u carrier does not match this deck")
    if surface is None:
        znt = cupy.zeros((1,), dtype=cupy.float32)  # never read: semi_slip is 0
        imove, umove, vmove = 0, 0.0, 0.0
    else:
        znt = cupy.ascontiguousarray(surface.zntmp, dtype=cupy.float32)
        if tuple(znt.shape) != (deck.ny + 2 * ng, deck.nx + 2 * ng):
            raise ValueError("the zntmp carrier does not match this deck")
        imove, umove, vmove = surface.imove, surface.u_move_m_s, surface.v_move_m_s
    blocks = ((n + NEST_PARCEL_THREADS - 1) // NEST_PARCEL_THREADS, 1, 1)
    i32, f32 = cupy.int32, cupy.float32
    module.get_function("gm1_nest_parcel_rk2_v2")(
        blocks, (NEST_PARCEL_THREADS, 1, 1),
        (positions[0], positions[1], positions[2], owner, status, u, v, w,
         device_deck.zf, device_deck.zh, znt,
         i32(n), i32(code), i32(deck.nx), i32(deck.ny), i32(deck.nz), i32(ng),
         i32(1 if deck.periodic else 0), i32(0 if surface is None else 1), i32(imove),
         f32(deck.dx_m), f32(deck.dy_m), f32(deck.minx_m), f32(deck.miny_m),
         f32(deck.maxz_m), f32(dt_s), f32(umove), f32(vmove)),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Attribute sampling: parcel_interp on one grid's live carriers.
# ---------------------------------------------------------------------------
class ParcelCarriers(Protocol):
    """What the attribute sampler reads from one grid's runtime."""

    deck: ParcelDeck
    ng: int
    bbc: int
    tbc: int
    imove: int
    u_move_m_s: float
    v_move_m_s: float

    def roughness(self) -> Any: ...                  # physical (ny, nx) znt, or None
    def padded(self, name: str) -> Any: ...          # "u3d" | "v3d" | "w3d"
    def mass(self, name: str) -> Any: ...            # physical (nz, ny, nx)
    def species(self, index: int) -> Any: ...        # physical (nz, ny, nx)
    def face(self, name: str) -> Any: ...            # physical (nz+1, ny, nx)
    def column(self, name: str) -> Any: ...          # host column, CM1 k at [k-1]


@dataclass
class RuntimeCarriers:
    """:class:`ParcelCarriers` over a :class:`CudaCartesianRuntime`'s live buffers."""

    runtime: Any
    deck: ParcelDeck
    xp: Any
    ng: int = 3

    @property
    def bbc(self) -> int:
        return int(self.runtime.config.bbc)

    @property
    def tbc(self) -> int:
        return int(getattr(self.runtime.config, "tbc", 1))

    @property
    def u_move_m_s(self) -> float:
        return float(self.runtime.config.u_move_m_s)

    @property
    def v_move_m_s(self) -> float:
        return float(self.runtime.config.v_move_m_s)

    @property
    def imove(self) -> int:
        value = getattr(self.runtime.config, "imove", None)
        if value is None:
            value = int(self.u_move_m_s != 0.0 or self.v_move_m_s != 0.0)
        return int(value)

    def roughness(self) -> Any:
        return parcel_roughness(self.runtime)

    def padded(self, name: str) -> Any:
        return self.runtime.dynamics.arrays[name]

    def mass(self, name: str) -> Any:
        return self.runtime.dynamics.fields[name]

    def species(self, index: int) -> Any:
        return self.runtime.dynamics.fields["q3d"][index]

    def face(self, name: str) -> Any:
        return self.runtime.dynamics.tke[name]

    def column(self, name: str) -> Any:
        np = _np()
        value = self.runtime.dynamics.metrics[name]
        return np.asarray(_host(self.xp, value[:, 0, 0]), dtype=np.float32)


class _Gatherer:
    """Point values of one grid's carriers at CM1 ``(k, j, i)`` index arrays."""

    def __init__(self, np: Any, xp: Any, carriers: ParcelCarriers) -> None:
        self.np, self.xp, self.c = np, xp, carriers
        self.deck = carriers.deck

    def _take(self, array: Any, flat: Any) -> Any:
        xp, np = self.xp, self.np
        values = xp.take(array.reshape(-1), xp.asarray(flat.reshape(-1)))
        return np.asarray(_host(xp, values), dtype=np.float32).reshape(flat.shape)

    def _lateral(self, i: Any, j: Any) -> tuple[Any, Any]:
        """``bc.F`` ``bcs`` one cell deep: periodic wraps, open copies the edge."""

        np, d = self.np, self.deck
        wbc, ebc, sbc, nbc = d.lateral
        i = np.where(i < 1, d.nx + i if wbc == 1 else 1, i)
        i = np.where(i > d.nx, i - d.nx if ebc == 1 else d.nx, i)
        j = np.where(j < 1, d.ny + j if sbc == 1 else 1, j)
        j = np.where(j > d.ny, j - d.ny if nbc == 1 else d.ny, j)
        return i, j

    def mass(self, array: Any, k: Any, j: Any, i: Any) -> Any:
        np, d = self.np, self.deck
        i, j = self._lateral(i, j)
        k = np.minimum(np.maximum(k, 1), d.nz)
        flat = ((k - 1) * d.ny + (j - 1)) * d.nx + (i - 1)
        return self._take(array, flat)

    def face(self, array: Any, k: Any, j: Any, i: Any) -> Any:
        np, d = self.np, self.deck
        i, j = self._lateral(i, j)
        k = np.minimum(np.maximum(k, 1), d.nz + 1)
        flat = ((k - 1) * d.ny + (j - 1)) * d.nx + (i - 1)
        return self._take(array, flat)

    def _padded_raw(self, name: str, k: Any, j: Any, i: Any) -> Any:
        array = self.c.padded(name)
        _, nj, ni = (int(v) for v in array.shape)
        ng = self.c.ng
        flat = (k * nj + (j - 1 + ng)) * ni + (i - 1 + ng)
        return self._take(array, flat)

    def uv(self, name: str, k: Any, j: Any, i: Any) -> Any:
        """``u3d``/``v3d`` with ``parcel_interp:902-959``'s ghost levels, never written."""

        np, d, c = self.np, self.deck, self.c
        nz = d.nz
        mid = np.minimum(np.maximum(k, 1), nz)
        out = self._padded_raw(name, mid, j, i)
        low, high = k <= 0, k >= nz + 1
        if bool(low.any()):
            if c.bbc == 1:
                a1 = self._padded_raw(name, np.ones_like(k), j, i)
                a2 = self._padded_raw(name, np.full_like(k, 2), j, i)
                a3 = self._padded_raw(name, np.full_like(k, 3), j, i)
                c1, c2, c3 = (np.float32(v) for v in d.cgs)
                ghost = (c1 * a1 + c2 * a2) + c3 * a3
            elif c.bbc == 2:
                move = c.u_move_m_s if name == "u3d" else c.v_move_m_s
                ghost = np.full(k.shape, np.float32(0.0) - np.float32(move), dtype=np.float32)
            elif c.bbc == 3:
                # parcel_interp:933-935 leaves the level unfilled; the sample
                # that reads it is replaced by :1530-1583's log layer.
                ghost = self._padded_raw(name, np.zeros_like(k), j, i)
            else:
                raise ValueError(f"bbc={c.bbc} has no transcribed parcel ghost")
            out = np.where(low, ghost, out)
        if bool(high.any()):
            if c.tbc == 1:
                a1 = self._padded_raw(name, np.full_like(k, nz), j, i)
                a2 = self._padded_raw(name, np.full_like(k, nz - 1), j, i)
                a3 = self._padded_raw(name, np.full_like(k, nz - 2), j, i)
                t1, t2, t3 = (np.float32(v) for v in d.cgt)
                ghost = (t1 * a1 + t2 * a2) + t3 * a3
            else:
                ghost = np.zeros(k.shape, dtype=np.float32)
            out = np.where(high, ghost, out)
        return out

    def w(self, k: Any, j: Any, i: Any) -> Any:
        np, d = self.np, self.deck
        kk = np.minimum(np.maximum(k, 0), d.nz + 2)
        out = self._padded_raw("w3d", kk, j, i)
        return np.where(k == d.nz + 1, np.float32(0.0), out)


def _derived_at(g: _Gatherer, k: Any, j: Any, i: Any, cache: dict[str, Any]) -> dict[str, Any]:
    """``parcel_interp:737-783`` (imoist=1) at scalar points, CM1's order."""

    np, c = g.np, g.c
    key = (k.tobytes(), j.tobytes(), i.tobytes())
    if key in cache:
        return cache[key]
    const = _cm1_constants(np)
    q = [g.mass(c.species(n), k, j, i) for n in range(6)]
    th3d = g.mass(c.mass("th3d"), k, j, i)
    th0 = g.mass(c.mass("th0"), k, j, i)
    pi0 = g.mass(c.mass("pi0"), k, j, i)
    pp3d = g.mass(c.mass("pp3d"), k, j, i)
    qv0 = g.mass(c.mass("qv0"), k, j, i)
    qt = q[0]
    for n in range(1, 6):
        qt = qt + q[n]
    th = th0 + th3d
    t = th * (pi0 + pp3d)
    rth0 = np.float32(1.0) / th0
    # parcel.F:771-773.  qc0 and qi0 are this lane's zero base condensate;
    # subtracting a +0.0 word changes no word, so the two terms are omitted.
    dum7 = const["g"] * ((th3d * rth0 + const["repsm1"] * (q[0] - qv0)) - (qt - q[0]))
    dum8 = (th * (np.float32(1.0) + const["reps"] * q[0])) / (np.float32(1.0) + qt)
    values = {"th": th, "t": t, "qt": qt, "dum7": dum7, "dum8": dum8, "pp3d": pp3d,
              "qv": q[0], "qc": q[1], "qr": q[2], "qi": q[3], "qs": q[4], "qg": q[5]}
    cache[key] = values
    return values


def _buoy_vpg_at(g: _Gatherer, k: Any, j: Any, i: Any, cache: dict[str, Any]) -> tuple[Any, Any]:
    """``parcel_interp:822-876``: ``buoy`` and ``vpg`` at w points ``k = 1..nz+1``."""

    np, d, c = g.np, g.deck, g.c
    nz = d.nz
    c1col, c2col = c.column("c1"), c.column("c2")
    const = _cm1_constants(np)
    rdz = np.float32(1.0) / np.float32(d.dz_reference_m)

    def mid_buoy(kk: Any) -> Any:
        lower = _derived_at(g, kk - 1, j, i, cache)["dum7"]
        upper = _derived_at(g, kk, j, i, cache)["dum7"]
        return c1col[kk - 1] * lower + c2col[kk - 1] * upper

    kk = np.minimum(np.maximum(k, 2), nz)
    buoy = mid_buoy(kk)
    zf = d.zf
    if bool((k <= 1).any()):
        b2, b3 = mid_buoy(np.full_like(k, 2)), mid_buoy(np.full_like(k, 3))
        edge = b2 + ((b3 - b2) * (zf[1] - zf[2])) / (zf[3] - zf[2])
        buoy = np.where(k <= 1, edge, buoy)
    if bool((k >= nz + 1).any()):
        bn, bn1 = mid_buoy(np.full_like(k, nz)), mid_buoy(np.full_like(k, nz - 1))
        edge = bn + ((bn - bn1) * (zf[nz + 1] - zf[nz])) / (zf[nz] - zf[nz - 1])
        buoy = np.where(k >= nz + 1, edge, buoy)
    lower = _derived_at(g, kk - 1, j, i, cache)
    upper = _derived_at(g, kk, j, i, cache)
    tem1 = (rdz * const["cp"]) * d.mf[kk]
    vpg = -((tem1 * (upper["pp3d"] - lower["pp3d"]))
            * (c2col[kk - 1] * upper["dum8"] + c1col[kk - 1] * lower["dum8"]))
    edge = (k <= 1) | (k >= nz + 1)
    vpg = np.where(edge, -buoy, vpg)
    return buoy.astype(np.float32), vpg.astype(np.float32)


def _zvort_at(g: _Gatherer, k: Any, j: Any, i: Any) -> Any:
    """``parcel.F:1066-1067`` at one corner, ``uf = vf = 1`` on this uniform lattice."""

    np, d = g.np, g.deck
    rdx = np.float32(1.0) / np.float32(d.dx_m)
    rdy = np.float32(1.0) / np.float32(d.dy_m)
    one = np.float32(1.0)
    dv = g.uv("v3d", k, j, i) - g.uv("v3d", k, j, i - 1)
    du = g.uv("u3d", k, j, i) - g.uv("u3d", k, j - 1, i)
    return ((dv * rdx) * one) - ((du * rdy) * one)


def sample_parcel_attributes(xp: Any, carriers: ParcelCarriers, positions: Any,
                             names: Sequence[str],
                             number_rows: Mapping[str, int] | None = None) -> dict[str, Any]:
    """``parcel_interp:1134-1510`` for the named attributes, host ``float32``.

    ``positions`` is host ``(3, m)``; every parcel must lie on this grid's
    lattice (the tracker samples a parcel only on the grid that owns it).
    ``number_rows`` maps each ``prcl_nc`` attribute to its zero-based ``q3d``
    row (:func:`cm1core.lagrangian_parcels.parcel_number_concentrations`).
    """

    np = _np()
    deck = carriers.deck
    s = reference_stencils(np, deck, positions[0], positions[1], positions[2], second_stage=False)
    if bool((s.status != STATUS_OK).any()):
        raise ValueError("an attribute sample was requested off this grid's lattice")
    g = _Gatherer(np, xp, carriers)
    cache: dict[str, Any] = {}
    out: dict[str, Any] = {}

    def corners(ia: Any, ja: Any, ka: Any) -> list[tuple[Any, Any, Any]]:
        return [(ka + dk, ja + dj, ia + di) for di, dj, dk in TRI_CORNERS]

    wu = _weights(np, s.rxu, s.rys, s.rzs)
    wv = _weights(np, s.rxs, s.ryv, s.rzs)
    ww = _weights(np, s.rxs, s.rys, s.rzw)
    wz = _weights(np, s.rxu, s.ryv, s.rzs)
    if "u" in names:
        out["u"] = _tri(np, [g.uv("u3d", k, j, i) for k, j, i in corners(s.iu, s.ju, s.ks)], wu)
    if "v" in names:
        out["v"] = _tri(np, [g.uv("v3d", k, j, i) for k, j, i in corners(s.iv, s.jflag, s.ks)], wv)
    wpoints = corners(s.iv, s.ju, s.kflag)
    if "w" in names:
        out["w"] = _tri(np, [g.w(k, j, i) for k, j, i in wpoints], ww)
    if "tke" in names:
        out["tke"] = _tri(np, [g.face(carriers.face("tke3d"), k, j, i) for k, j, i in wpoints], ww)
    if "b" in names or "vpg" in names:
        pairs = [_buoy_vpg_at(g, k, j, i, cache) for k, j, i in wpoints]
        if "b" in names:
            out["b"] = _tri(np, [p[0] for p in pairs], ww)
        if "vpg" in names:
            out["vpg"] = _tri(np, [p[1] for p in pairs], ww)
    # Scalar points, parcel.F:1365-1447: rx = rxs, ry = rys, rz = rzs.
    spoints = corners(s.iv, s.ju, s.ks)
    wsc = _weights(np, s.rxs, s.rys, s.rzs)
    scalar_names = [n for n in ("th", "t", "qv", "qc", "qr", "qi", "qs", "qg") if n in names]
    if scalar_names:
        derived = [_derived_at(g, k, j, i, cache) for k, j, i in spoints]
        for name in scalar_names:
            out[name] = _tri(np, [value[name] for value in derived], wsc)
    for name in ("prs", "rho"):
        if name in names:
            out[name] = _tri(np, [g.mass(carriers.mass(name), k, j, i) for k, j, i in spoints], wsc)
    # parcel.F:1430-1431: q3d(n) for n=nnc1..nnc2 on the scalar-point stencil.
    for name, row in (number_rows or {}).items():
        if name in names:
            out[name] = _tri(np, [g.mass(carriers.species(int(row)), k, j, i) for k, j, i in spoints], wsc)
    if "zvort" in names:
        out["zvort"] = _tri(np, [_zvort_at(g, k, j, i) for k, j, i in corners(s.iu, s.jflag, s.ks)], wz)
    if carriers.bbc == 3 and any(name in names for name in ("u", "v", "zvort")):
        _semi_slip_attributes(np, g, carriers, s, positions, names, out)
    return {name: np.asarray(value, dtype=np.float32) for name, value in out.items()}


def sample_mass_rectangles(xp: Any, carriers: ParcelCarriers, positions: Any,
                           rectangles: Mapping[str, Any]) -> dict[str, Any]:
    """Named physical ``(nz, ny, nx)`` scalar-point rectangles at parcels, host ``float32``.

    The stencil, the ``bc.F`` lateral copy and the vertical clamp are
    :func:`sample_parcel_attributes`' own ``prs``/``rho`` path
    (``parcel.F:1365-1447``, ``rx = rxs, ry = rys, rz = rzs``) on this grid's
    deck, so a GM1 extension column -- the vorticity budget -- is sampled
    exactly as a CM1 scalar attribute is, stretched-deck ``zh`` search included.
    """

    np = _np()
    deck = carriers.deck
    s = reference_stencils(np, deck, positions[0], positions[1], positions[2], second_stage=False)
    if bool((s.status != STATUS_OK).any()):
        raise ValueError("an attribute sample was requested off this grid's lattice")
    g = _Gatherer(np, xp, carriers)
    spoints = [(s.ks + dk, s.ju + dj, s.iv + di) for di, dj, dk in TRI_CORNERS]
    wsc = _weights(np, s.rxs, s.rys, s.rzs)
    return {
        name: np.asarray(_tri(np, [g.mass(array, k, j, i) for k, j, i in spoints], wsc), dtype=np.float32)
        for name, array in rectangles.items()
    }


def _semi_slip_attributes(np: Any, g: "_Gatherer", carriers: ParcelCarriers, s: Stencils,
                          positions: Any, names: Sequence[str], out: dict[str, Any]) -> None:
    """``parcel_interp:1515-1583``: below ``zh(1)`` on a semi-slip grid.

    ``u`` and ``v`` are the lowest level's ``get2d`` sample times the log-layer
    ratio, *plus* ``umove`` (``:1554-1555``, which unlike ``parcel_driver:508``
    does not subtract it back), and ``zvort`` is ``get2d`` of the corner
    vorticity of the log-layer-scaled lowest-level winds (``:1557-1581``).
    """

    deck = carriers.deck
    roughness = carriers.roughness()
    if roughness is None:
        raise RuntimeError(
            "bbc=3 but this grid carries no surface roughness; parcel_interp:1515-1583 reads zntmp"
        )
    ng = carriers.ng
    zntmp = pad_roughness(np, np.asarray(_host(g.xp, roughness), dtype=np.float32), deck, ng=ng)
    zz = np.asarray(positions[2], dtype=np.float32)
    zh1 = np.float32(deck.zh[1])
    q = np.flatnonzero(zz < zh1)
    if q.size == 0:
        return
    umove, vmove = np.float32(carriers.u_move_m_s), np.float32(carriers.v_move_m_s)
    iflag, jflag, iv, ju = s.iflag[q], s.jflag[q], s.iv[q], s.ju[q]
    z3d = zz[q]
    # :1516-1524, z0 at the scalar anchor.
    z0 = _get2d_padded(np, zntmp, ju - 1 + ng, iv - 1 + ng, s.rxs[q], s.rys[q])
    one = np.ones_like(iflag)

    def level(name: str, jj: Any, ii: Any) -> Any:
        return g._padded_raw(name, one, jj, ii)

    if "u" in names or "v" in names:
        var = _log_layer_ratio(np, z3d, z0, zh1)
        if "u" in names:
            uval = _get2d(np, level("u3d", ju, iflag), level("u3d", ju, iflag + 1),
                          level("u3d", ju + 1, iflag), level("u3d", ju + 1, iflag + 1), s.rxu[q], s.rys[q])
            column = np.array(out["u"], dtype=np.float32, copy=True)
            column[q] = (uval + umove) * var
            out["u"] = column
        if "v" in names:
            vval = _get2d(np, level("v3d", jflag, iv), level("v3d", jflag, iv + 1),
                          level("v3d", jflag + 1, iv), level("v3d", jflag + 1, iv + 1), s.rxs[q], s.ryv[q])
            column = np.array(out["v"], dtype=np.float32, copy=True)
            column[q] = (vval + vmove) * var
            out["v"] = column
    if "zvort" in names:
        half = np.float32(0.5)

        def scaled(name: str, jj: Any, ii: Any, move: Any, lower: tuple[int, int]) -> Any:
            # :1560-1562 (u, averaging znt west) and :1567-1569 (v, south).
            z0c = half * (zntmp[jj - lower[0] - 1 + ng, ii - lower[1] - 1 + ng] + zntmp[jj - 1 + ng, ii - 1 + ng])
            rznt = np.float32(1.0) / z0c
            return ((level(name, jj, ii) + move) * libm_logf(np, (z3d + z0c) * rznt)) / libm_logf(
                np, (zh1 + z0c) * rznt)

        rdx = np.float32(1.0) / np.float32(deck.dx_m)
        rdy = np.float32(1.0) / np.float32(deck.dy_m)
        unit = np.float32(1.0)  # uf(i) = vf(j) = 1 on this uniform lattice
        dum7 = {}
        for dj in (0, 1):
            for di in (0, 1):
                ii, jj = iflag + di, jflag + dj
                dum4 = scaled("v3d", jj, ii, vmove, (1, 0)) - scaled("v3d", jj, ii - 1, vmove, (1, 0))
                dum3 = scaled("u3d", jj, ii, umove, (0, 1)) - scaled("u3d", jj - 1, ii, umove, (0, 1))
                dum7[(di, dj)] = ((dum4 * rdx) * unit) - ((dum3 * rdy) * unit)
        column = np.array(out["zvort"], dtype=np.float32, copy=True)
        column[q] = _get2d(np, dum7[(0, 0)], dum7[(1, 0)], dum7[(0, 1)], dum7[(1, 1)], s.rxu[q], s.ryv[q])
        out["zvort"] = column


# ---------------------------------------------------------------------------
# Extension point: extra per-parcel attributes (the vorticity budget terms).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParcelSampleContext:
    """Everything a hook may read when a frame is recorded.  Read-only by contract.

    ``carriers[code]`` is the :class:`ParcelCarriers` of the grid that owns
    the parcels ``indices[code]``; ``sample`` is the grid-aware attribute
    sampler, so a hook's terms are taken on the same stencil as CM1's
    attributes.  A hook must return host ``float32`` ``(nparcels,)`` columns
    and leave unowned parcels at :data:`SENTINEL_F32`.
    """

    step: int
    model_time_s: float
    positions: Any
    owner: Any
    indices: Mapping[int, Any]
    carriers: Mapping[int, ParcelCarriers]
    xp: Any
    sample: Callable[..., Mapping[str, Any]]


class ParcelAttributeHook(Protocol):
    """A named group of extra per-parcel columns, appended after CM1's attributes."""

    names: tuple[str, ...]
    units: tuple[str, ...]
    marker: str

    def sample(self, context: ParcelSampleContext) -> Mapping[str, Any]: ...


# ---------------------------------------------------------------------------
# Seeding and admission.
# ---------------------------------------------------------------------------
def seed_box_positions(settings: Any) -> Any:
    """``(3, nparcels)`` float32: the declared box, then ``init3d.F:363``'s sentinel."""

    np = _np()
    n = int(settings.nparcels)
    out = np.full((3, n), np.float32(SENTINEL_F32), dtype=np.float32)
    index = 0
    for k in range(int(settings.seed_nz)):
        for j in range(int(settings.seed_ny)):
            for i in range(int(settings.seed_nx)):
                out[0, index] = np.float32(settings.seed_x0_m + settings.seed_dx_m * i)
                out[1, index] = np.float32(settings.seed_y0_m + settings.seed_dy_m * j)
                out[2, index] = np.float32(settings.seed_z0_m + settings.seed_dz_m * k)
                index += 1
    return out


def seed_extent_defect(settings: Any, geometry: Any, child_top_m: float) -> str | None:
    """Name the first seed outside the child's free interior, or ``None``."""

    x_lo, x_hi, y_lo, y_hi = free_interior_m(geometry)
    np = _np()
    corners = {
        "seed_x0_m": float(np.float32(settings.seed_x0_m)),
        "last x": float(np.float32(settings.seed_x0_m + settings.seed_dx_m * (settings.seed_nx - 1))),
        "seed_y0_m": float(np.float32(settings.seed_y0_m)),
        "last y": float(np.float32(settings.seed_y0_m + settings.seed_dy_m * (settings.seed_ny - 1))),
        "seed_z0_m": float(np.float32(settings.seed_z0_m)),
        "last z": float(np.float32(settings.seed_z0_m + settings.seed_dz_m * (settings.seed_nz - 1))),
    }
    frame = (
        f"metres from the parent grid origin; the child's free interior at parent cell "
        f"({geometry.parent_i0}, {geometry.parent_j0}) is x [{x_lo}, {x_hi}], "
        f"y [{y_lo}, {y_hi}], z [0, {child_top_m}]"
    )
    for label in ("seed_x0_m", "last x"):
        if not x_lo <= corners[label] <= x_hi:
            return f"the seed box's {label} is {corners[label]} m, {frame}"
    for label in ("seed_y0_m", "last y"):
        if not y_lo <= corners[label] <= y_hi:
            return f"the seed box's {label} is {corners[label]} m, {frame}"
    for label in ("seed_z0_m", "last z"):
        if not 0.0 <= corners[label] <= child_top_m:
            return f"the seed box's {label} is {corners[label]} m, {frame}"
    return None


def nest_parcel_admission_defect(*, settings: Any, geometry: Any, child_grid: Any,
                                 parent_dt_s: float, bbc: int, tbc: int, lateral: Sequence[int],
                                 terrain: bool) -> tuple[str, str, str] | None:
    """``(code_suffix, option, reason)`` for the first inadmissible choice, or ``None``."""

    from .nest_vertical import levels_of_grid

    if terrain:
        return ("TERRAIN", "grid.terrain",
                "parcel.F's terrain branch (sigma coordinate, getparcelzs) is not transcribed "
                "for a child; a flat-deck search over terrain would place every parcel at the "
                "wrong height")
    if bbc not in (1, 2, 3):
        return ("BOUNDARY", "closures.bbc",
                f"bbc={bbc} has no parcel.F lower-boundary branch (parcel.F:59-96)")
    if tbc not in (1, 2):
        return ("BOUNDARY", "boundaries.top", f"tbc={tbc} has no parcel.F upper-boundary branch")
    for side in lateral:
        if side not in (1, 2):
            return ("BOUNDARY", "boundaries.west",
                    f"parent lateral boundary {side}: a parcel handed to the parent near a rigid "
                    "wall would read halos bc.F fills by negation, which is not transcribed")
    cadence = float(settings.prclfrq_s) / float(parent_dt_s)
    if abs(cadence - round(cadence)) > 1.0e-9 * max(1.0, cadence) or round(cadence) < 1:
        return ("CADENCE", "lagrangian.nest_child.prclfrq_s",
                f"prclfrq_s = {settings.prclfrq_s} s is not a whole number of parent steps of "
                f"{parent_dt_s} s; a child frame between parent-step boundaries would publish "
                "parcels the hand-off rule has not yet assigned to a grid")
    top = float(levels_of_grid(child_grid).zf[int(child_grid.nz) + 1])
    defect = seed_extent_defect(settings, geometry, top)
    if defect is not None:
        return ("SEED-EXTENT", "lagrangian.nest_child.seed_x0_m",
                defect + "; a seed outside it would be sampled on parent-forced rows or off the "
                "child altogether while its track claimed child winds")
    return None


NEST_PARCEL_GATE = "cartesian-runtime"


def require_nest_parcel_admission(config: Any, binding: Any) -> None:
    """Refuse ``[lagrangian.nest_child]`` where it cannot mean what it says.

    ``binding`` is the door's :class:`~cm1core.cartesian_door.NestBinding`, or
    ``None`` for a document without ``[nest]``.
    """

    from .errors import refuse

    lagrangian = getattr(config, "lagrangian", None)
    settings = None if lagrangian is None else lagrangian.nest_child
    if settings is None:
        return
    if binding is None:
        refuse(
            code="CM1-CARTESIAN-NEST-PARCELS-NO-NEST",
            gate=NEST_PARCEL_GATE,
            reason=(
                "[lagrangian.nest_child] seeds parcels in a nest child, but the document "
                "declares no [nest]; there is no child lattice to integrate them on, and "
                "running them on the parent would publish parent winds as child tracks"
            ),
            remedy="declare [nest], or move the seeding to [lagrangian] for parent parcels",
            option="lagrangian.nest_child", value=None,
        )
    b = config.boundaries
    defect = nest_parcel_admission_defect(
        settings=settings, geometry=binding.geometry, child_grid=binding.child.grid,
        parent_dt_s=float(config.run.dt_s), bbc=int(config.closures.bbc), tbc=int(b.top),
        lateral=(int(b.west), int(b.east), int(b.south), int(b.north)),
        terrain=bool(config.grid.terrain),
    )
    if defect is not None:
        suffix, option, reason = defect
        refuse(
            code=f"CM1-CARTESIAN-NEST-PARCELS-{suffix}",
            gate=NEST_PARCEL_GATE,
            reason=reason,
            remedy=(
                "declare lagrangian.nest_child in metres from the parent grid origin, inside "
                "the child's free interior, on a whole number of parent steps, over a flat "
                "free-slip, no-slip or semi-slip grid with open or periodic parent laterals"
            ),
            option=option,
            value=settings.record(),
        )


# ---------------------------------------------------------------------------
# The tracker.
# ---------------------------------------------------------------------------
def _attribute_names(layout: Any) -> tuple[str, ...]:
    return tuple(layout.names) + ("grid",)


class NestParcelSeedExtentError(ValueError):
    """The child's seed box leaves the child's free interior at seeding time.

    Its own type so the live driver refuses exactly this defect as a seed box
    to move, and lets every other ``ValueError`` reach the reader as itself.
    """


@dataclass
class NestParcelTracker:
    """Child and parent parcels of one nested run, owned per grid, handed off at boundaries."""

    settings: Any
    layout: Any
    child_deck: ParcelDeck
    parent_deck: ParcelDeck
    child_dt_s: float
    parent_dt_s: float
    xp: Any
    stream: Any = None
    ng: int = 3
    hooks: list[Any] = field(default_factory=list)
    positions: Any = None
    owner: Any = None
    status: Any = None
    events: list[dict[str, Any]] = field(default_factory=list)
    events_dropped: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    handoffs: dict[str, int] = field(default_factory=lambda: {"child_to_parent": 0, "parent_to_child": 0})
    terminations: dict[str, int] = field(default_factory=dict)
    child_steps: int = 0
    parent_steps: int = 0
    frames: list[dict[str, Any]] = field(default_factory=list)
    seeded: dict[str, Any] = field(default_factory=dict)
    _child_velocity: Any = None
    _parent_velocity: Any = None
    _child_surface: Any = None
    _parent_surface: Any = None
    _child_captured: bool = False
    _parent_captured: bool = False
    _module: Any = None
    _device_decks: dict[str, DeviceDeck] = field(default_factory=dict)
    _writer: Any = None
    _owned: dict[int, int] = field(default_factory=dict)

    # -- construction -------------------------------------------------------------
    @classmethod
    def build(cls, *, settings: Any, layout: Any, geometry: Any, parent_config: Any,
              child_config: Any, xp: Any, stream: Any = None, model_time_s: float = 0.0,
              parent_step: int = 0, seed: bool = True) -> "NestParcelTracker":
        """The child tracker, seeded.

        ``layout`` must be resolved for both grids' own words
        (:func:`cm1core.lagrangian_parcels.parcel_switch_context`), as
        :meth:`cm1core.lagrangian_stage.LagrangianStage.build` requires of the
        parent stage: the parcels integrate each grid's ``bbc``, and
        ``param.F:6280-6282`` decides ``prcl_sfc`` from it.  A seed box outside
        the child raises :class:`NestParcelSeedExtentError`.
        """

        from .cuda_psolver3_terrain_open import CM1_OPEN_NGXY
        from .lagrangian_parcels import parcel_switch_context

        for grid, config in (("parent", parent_config), ("child", child_config)):
            context = parcel_switch_context(config)
            if layout.context == context:
                continue
            if int(layout.context.bbc) != int(context.bbc):
                raise ValueError(
                    f"the parcel switches were resolved for bbc={layout.context.bbc} but the nest "
                    f"{grid} integrates bbc={context.bbc}; param.F:6280-6282 decides prcl_sfc "
                    "from the bbc the run uses"
                )
            raise ValueError(
                f"the parcel switches were resolved for words the nest {grid} does not hold "
                "(param.F:6257-6282 reads the moisture axis, closure and bbc): "
                f"{layout.context.record()} against {context.record()}"
            )
        tracker = cls(
            settings=settings, layout=layout,
            child_deck=child_deck(geometry, child_config.grid),
            parent_deck=parent_deck(parent_config),
            child_dt_s=float(child_config.dt_s), parent_dt_s=float(parent_config.dt_s),
            xp=xp, stream=stream, ng=int(CM1_OPEN_NGXY),
        )
        if seed:
            tracker.seed(geometry, model_time_s=model_time_s, parent_step=parent_step)
        # seed=False: a resumed nest's parcels are read from the child's restart
        # (cm1core.nest_restart.restore_nest_child), wherever they have travelled.
        return tracker

    def seed(self, geometry: Any, *, model_time_s: float, parent_step: int) -> dict[str, Any]:
        np = _np()
        defect = seed_extent_defect(self.settings, geometry, self.child_deck.maxz_m)
        if defect is not None:
            raise NestParcelSeedExtentError(defect)
        host = seed_box_positions(self.settings)
        placed = host[0] > np.float32(-1.0e29)
        owner = np.where(placed, OWNER_CHILD, OWNER_TERMINATED).astype(np.int32)
        self.positions = self.xp.asarray(host)
        self.owner = self.xp.asarray(owner)
        self.status = self.xp.zeros(owner.shape, dtype=self.xp.int32)
        self._owned = {OWNER_CHILD: int(placed.sum()), OWNER_PARENT: 0}
        self.seeded = {
            "model_time_s": float(model_time_s), "parent_step": int(parent_step),
            "placed_parcels": int(placed.sum()), "nparcels": int(host.shape[1]),
            "child_placement": _placement(geometry),
        }
        return dict(self.seeded)

    def owned(self, code: int) -> int:
        return int(self._owned.get(code, 0))

    # -- the advecting velocity -----------------------------------------------------
    def _snapshot(self, current: Any, arrays: Mapping[str, Any], deck: ParcelDeck,
                  config: Any) -> Any:
        xp = self.xp
        if current is None:
            current = tuple(xp.empty_like(arrays[name]) for name in ("rru", "rrv", "rrw"))
        for target, name in zip(current, ("rru", "rrv", "rrw")):
            xp.copyto(target, arrays[name])
        u, v, w = current
        fill_velocity_halos(xp, u, v, w, deck, ng=self.ng)
        fill_velocity_ghosts(xp, u, v, w, deck, bbc=int(config.bbc), tbc=int(getattr(config, "tbc", 1)),
                             u_move_m_s=float(config.u_move_m_s), v_move_m_s=float(config.v_move_m_s))
        return current

    def capture_child(self, arrays: Mapping[str, Any], config: Any, roughness: Any = None) -> None:
        """``solve2.F:1290-1301`` in the child, at its ``nrk == nrkmax``.

        ``roughness`` is the child's ``closure.surface.znt``; ``bbc = 3``
        needs it, because ``parcel.F:470-515`` reads that step's ``zntmp``.
        """

        if self.owned(OWNER_CHILD) == 0:
            return
        self._child_velocity = self._snapshot(self._child_velocity, arrays, self.child_deck, config)
        self._child_surface = semi_slip_for(self.xp, config, self.child_deck, roughness, ng=self.ng)
        self._child_captured = True

    def capture_parent(self, arrays: Mapping[str, Any], config: Any, roughness: Any = None) -> None:
        if self.owned(OWNER_PARENT) == 0:
            return
        self._parent_velocity = self._snapshot(self._parent_velocity, arrays, self.parent_deck, config)
        self._parent_surface = semi_slip_for(self.xp, config, self.parent_deck, roughness, ng=self.ng)
        self._parent_captured = True

    # -- stepping -------------------------------------------------------------------
    def _advance(self, code: int, velocity: Any, deck: ParcelDeck, dt_s: float,
                 surface: SemiSlip | None = None) -> None:
        xp = self.xp
        u, v, w = velocity
        if _is_cupy(xp):
            if self._module is None:
                self._module = compile_nest_parcel_module(xp)
            if deck.name not in self._device_decks:
                self._device_decks[deck.name] = upload_deck(xp, deck)
            launch_nest_parcel_rk2(
                xp, self._module, positions=self.positions, owner=self.owner, status=self.status,
                code=code, u=u, v=v, w=w, deck=deck, device_deck=self._device_decks[deck.name],
                dt_s=dt_s, ng=self.ng, stream=self.stream, surface=surface,
            )
        else:
            self.positions, self.status = reference_rk2(
                xp, self.positions, self.owner, self.status, code=code, u=u, v=v, w=w,
                deck=deck, dt_s=dt_s, ng=self.ng, surface=surface,
            )

    def advance_child(self) -> None:
        """``solve3.F:768-769`` in the child: once per completed child step."""

        if self.owned(OWNER_CHILD) == 0:
            return
        if not self._child_captured:
            raise RuntimeError(
                "the child parcel step has no captured advecting velocity; the child's "
                "solve2.F:1290-1301 snapshot did not run this step"
            )
        self._advance(OWNER_CHILD, self._child_velocity, self.child_deck, self.child_dt_s,
                      self._child_surface)
        self._child_captured = False
        self.child_steps += 1

    def after_parent_step(self, driver: Any) -> dict[str, Any] | None:
        """Parent parcels' step, terminations, hand-off on the current placement, frame."""

        np, xp = _np(), self.xp
        parent = driver.parent
        step = int(parent.clock.completed_steps)
        when = float(parent.clock.model_seconds)
        if self.owned(OWNER_PARENT):
            if not self._parent_captured:
                raise RuntimeError(
                    "parcels are owned by the parent but the parent's solve2.F:1290-1301 "
                    "snapshot did not run this step"
                )
            self._advance(OWNER_PARENT, self._parent_velocity, self.parent_deck, self.parent_dt_s,
                          self._parent_surface)
            self._parent_captured = False
        self.parent_steps += 1
        positions = np.asarray(_host(xp, self.positions), dtype=np.float32)
        owner = np.asarray(_host(xp, self.owner), dtype=np.int32).copy()
        status = np.asarray(_host(xp, self.status), dtype=np.int32)
        for parcel in np.flatnonzero((status != STATUS_OK) & (owner != OWNER_TERMINATED)):
            reason = int(status[parcel])
            self._event(step, when, int(parcel), int(owner[parcel]), OWNER_TERMINATED,
                        STATUS_REASONS[reason], positions[:, parcel])
            key = STATUS_REASONS[reason].split(";")[0]
            self.terminations[key] = self.terminations.get(key, 0) + 1
            owner[parcel] = OWNER_TERMINATED
        inside = inside_free_interior(np, driver.geometry, positions[0], positions[1])
        to_parent = np.flatnonzero((owner == OWNER_CHILD) & ~inside)
        to_child = np.flatnonzero((owner == OWNER_PARENT) & inside)
        for parcel in to_parent:
            self._event(step, when, int(parcel), OWNER_CHILD, OWNER_PARENT,
                        "outside the child's free interior", positions[:, parcel])
        for parcel in to_child:
            self._event(step, when, int(parcel), OWNER_PARENT, OWNER_CHILD,
                        "inside the child's free interior", positions[:, parcel])
        owner[to_parent] = OWNER_PARENT
        owner[to_child] = OWNER_CHILD
        self.handoffs["child_to_parent"] += int(to_parent.size)
        self.handoffs["parent_to_child"] += int(to_child.size)
        self.owner = xp.asarray(owner)
        self.status = xp.zeros(owner.shape, dtype=xp.int32)
        self._owned = {OWNER_CHILD: int((owner == OWNER_CHILD).sum()),
                       OWNER_PARENT: int((owner == OWNER_PARENT).sum())}
        # A move (cm1core.nest_move) re-places the lattice by whole parent
        # cells; the child's next steps search the moved faces.  Positions are
        # ground-frame metres and do not change.
        self.relocate_child(driver.geometry)
        cadence = int(round(float(self.settings.prclfrq_s) / self.parent_dt_s))
        if self._writer is not None and step % cadence == 0:
            return self.record_frame(driver, step=step, model_time_s=when,
                                     positions=positions, owner=owner)
        return None

    def relocate_child(self, geometry: Any) -> bool:
        """Put the child deck's ``xf(1)``/``yf(1)`` on ``geometry``; ``True`` if it moved."""

        from dataclasses import replace

        np = _np()
        minx = float(np.float32(geometry.origin_x_m))
        miny = float(np.float32(geometry.origin_y_m))
        if minx == self.child_deck.minx_m and miny == self.child_deck.miny_m:
            return False
        self.child_deck = replace(self.child_deck, minx_m=minx, miny_m=miny)
        return True

    # -- events ---------------------------------------------------------------------
    def _event(self, step: int, when: float, parcel: int, source: int, target: int,
               reason: str, position: Any) -> None:
        row = {
            "parent_step": step, "model_time_s": when, "parcel": parcel,
            "from": OWNER_NAMES[source], "to": OWNER_NAMES[target], "reason": reason,
            "x_m": float(position[0]), "y_m": float(position[1]), "z_m": float(position[2]),
        }
        if self._writer is not None:
            self._writer.event(row)
        if len(self.events) < EVENT_ROW_BUDGET:
            self.events.append(row)
        else:
            self.events_dropped += 1

    # -- frames ---------------------------------------------------------------------
    def attribute_names(self) -> tuple[str, ...]:
        names = list(_attribute_names(self.layout))
        for hook in self.hooks:
            names.extend(hook.names)
        return tuple(names)

    def register_hook(self, hook: Any) -> None:
        """Add extra per-parcel columns (for example vorticity-budget terms).

        Hooks are fixed before the track opens: a column added mid-run would
        change the arena's shape between frames.
        """

        if self._writer is not None:
            raise RuntimeError("attribute hooks must be registered before the track opens")
        if len(hook.names) != len(hook.units):
            raise ValueError("a parcel attribute hook needs one unit per name")
        taken = set(self.attribute_names())
        clash = [name for name in hook.names if name in taken]
        if clash:
            raise ValueError(f"parcel attribute hook column {clash[0]!r} is already published")
        self.hooks.append(hook)

    def record_frame(self, driver: Any, *, step: int, model_time_s: float,
                     positions: Any, owner: Any) -> dict[str, Any]:
        from .lagrangian_parcels import PARCEL_UNFILLED_ATTRIBUTES

        np = _np()
        names = self.attribute_names()
        n = positions.shape[1]
        rows = np.full((len(names), n), np.float32(SENTINEL_F32), dtype=np.float32)
        placed = positions[0] > np.float32(-1.0e29)
        index = {name: q for q, name in enumerate(names)}
        for axis, name in enumerate(("x", "y", "z")):
            rows[index[name], placed] = positions[axis, placed]
        rows[index["mtime"], placed] = np.float32(model_time_s)
        rows[index["grid"]] = owner.astype(np.float32)
        carriers = {
            OWNER_CHILD: RuntimeCarriers(driver.child, self.child_deck, self.xp, self.ng),
            OWNER_PARENT: RuntimeCarriers(driver.parent, self.parent_deck, self.xp, self.ng),
        }
        wanted = [name for name in self.layout.names
                  if name not in ("x", "y", "z", "mtime") and name not in PARCEL_UNFILLED_ATTRIBUTES]
        indices: dict[int, Any] = {}
        counts = {}
        for code in (OWNER_CHILD, OWNER_PARENT):
            chosen = np.flatnonzero(owner == code)
            indices[code] = chosen
            counts[OWNER_NAMES[code]] = int(chosen.size)
            if chosen.size == 0:
                continue
            values = sample_parcel_attributes(self.xp, carriers[code], positions[:, chosen], wanted,
                                              dict(self.layout.number_rows))
            for name, column in values.items():
                rows[index[name], chosen] = column
        counts["terminated"] = int(((owner == OWNER_TERMINATED) & placed).sum())
        if self.hooks:
            context = ParcelSampleContext(
                step=step, model_time_s=model_time_s, positions=positions, owner=owner,
                indices=indices, carriers=carriers, xp=self.xp,
                sample=sample_parcel_attributes,
            )
            for hook in self.hooks:
                columns = hook.sample(context)
                for name in hook.names:
                    column = np.asarray(columns[name], dtype=np.float32)
                    if column.shape != (n,):
                        raise ValueError(f"parcel attribute hook column {name!r} is not ({n},)")
                    rows[index[name]] = column
        row = {"step": int(step), "model_time_s": float(model_time_s), "counts": counts,
               "child_placement": _placement(driver.geometry)}
        self._writer.frame(rows, row)
        self.frames.append(row)
        return row

    # -- the track directory -----------------------------------------------------------
    def open_track(self, directory: Path | str) -> None:
        if self._writer is not None:
            raise RuntimeError("the child parcel track is already open")
        units = self._units()
        self._writer = NestParcelTrackWriter.open(Path(directory), names=self.attribute_names(),
                                                  units=units, nparcels=int(self.settings.nparcels))

    def _units(self) -> tuple[str, ...]:
        from .lagrangian_parcels import parcel_attribute_units

        units = [parcel_attribute_units(name) for name in self.layout.names] + ["code"]
        for hook in self.hooks:
            units.extend(hook.units)
        return tuple(units)

    def publish_track(self) -> dict[str, Any] | None:
        if self._writer is None:
            return None
        return self._writer.close(self.record())

    def record(self) -> dict[str, Any]:
        from .lagrangian_parcels import PARCEL_UNFILLED_ATTRIBUTES

        return {
            "schema": NEST_PARCEL_SCHEMA,
            "kernel_abi": NEST_PARCEL_KERNEL_ABI,
            "kernel_sha256": NEST_PARCEL_KERNEL_SHA256,
            "settings": self.settings.record(),
            "coordinate_frame": GM1_CHOICES["coordinates"],
            "child_deck": self.child_deck.record(),
            "parent_deck": self.parent_deck.record(),
            "child_dt_s": self.child_dt_s,
            "parent_dt_s": self.parent_dt_s,
            "seeded": dict(self.seeded),
            "child_steps": self.child_steps,
            "parent_steps": self.parent_steps,
            "owned_now": {"child": self.owned(OWNER_CHILD), "parent": self.owned(OWNER_PARENT)},
            "handoffs": dict(self.handoffs),
            "terminations": dict(self.terminations),
            "events_retained": len(self.events),
            "events_dropped": self.events_dropped,
            "grid_codes": {str(k): v for k, v in OWNER_NAMES.items()},
            "attributes": list(self.attribute_names()),
            "parcel_attributes": self.layout.record(),
            "unfilled_attributes": {n: PARCEL_UNFILLED_ATTRIBUTES[n] for n in self.layout.unfilled},
            "attribute_hooks": [{"names": list(h.names), "marker": h.marker} for h in self.hooks],
            "frames_recorded": len(self.frames),
            "cm1_authority": dict(CM1_AUTHORITY),
            "gm1_choices": dict(GM1_CHOICES),
            "unvalidated": UNVALIDATED_MARKER,
            "changes_the_simulated_storm": False,
        }


def _placement(geometry: Any) -> dict[str, Any]:
    return {
        "parent_i0": int(geometry.parent_i0), "parent_j0": int(geometry.parent_j0),
        "origin_x_m": float(geometry.origin_x_m), "origin_y_m": float(geometry.origin_y_m),
    }


class NestParcelTrackWriter:
    """``(frames, npvals, nparcels)`` little-endian float32, streamed; events and receipt."""

    def __init__(self, root: Path, names: tuple[str, ...], units: tuple[str, ...],
                 nparcels: int, arena: Any, events: Any) -> None:
        self.root, self.names, self.units, self.nparcels = root, names, units, nparcels
        self._arena, self._events = arena, events
        self._digest = hashlib.sha256()
        self._bytes = 0
        self._frames: list[dict[str, Any]] = []

    @classmethod
    def open(cls, root: Path, *, names: tuple[str, ...], units: tuple[str, ...],
             nparcels: int) -> "NestParcelTrackWriter":
        root.mkdir(parents=True, exist_ok=False)
        arena = (root / NEST_PARCEL_ARENA_FILENAME).open("xb", buffering=0)
        events = (root / NEST_PARCEL_EVENTS_FILENAME).open("x", encoding="utf-8")
        events.write("parent_step\tmodel_time_s\tparcel\tfrom\tto\tx_m\ty_m\tz_m\treason\n")
        return cls(root, names, units, nparcels, arena, events)

    def frame(self, rows: Any, row: Mapping[str, Any]) -> None:
        np = _np()
        raw = np.ascontiguousarray(rows, dtype="<f4").tobytes(order="C")
        self._arena.write(raw)
        self._digest.update(raw)
        self._bytes += len(raw)
        self._frames.append(dict(row))

    def event(self, row: Mapping[str, Any]) -> None:
        self._events.write(
            f"{row['parent_step']}\t{row['model_time_s']!r}\t{row['parcel']}\t{row['from']}\t"
            f"{row['to']}\t{row['x_m']!r}\t{row['y_m']!r}\t{row['z_m']!r}\t{row['reason']}\n"
        )

    def close(self, record: Mapping[str, Any]) -> dict[str, Any]:
        os.fsync(self._arena.fileno())
        self._arena.close()
        self._events.flush()
        os.fsync(self._events.fileno())
        self._events.close()
        table = ["index\tname\tunits"]
        table += [f"{q}\t{name}\t{unit}" for q, (name, unit) in enumerate(zip(self.names, self.units))]
        with (self.root / NEST_PARCEL_TABLE_FILENAME).open("xb", buffering=0) as handle:
            handle.write(("\n".join(table) + "\n").encode("utf-8"))
            os.fsync(handle.fileno())
        receipt = {
            "schema": NEST_PARCEL_SCHEMA,
            "arena_shape_frames_npvals_nparcels": [len(self._frames), len(self.names), self.nparcels],
            "arena_bytes": self._bytes,
            "arena_sha256": self._digest.hexdigest(),
            "attribute_order": list(self.names),
            "grid_row": (
                "the 'grid' row names the grid each sample was taken on: 1 child, 2 parent, "
                "0 terminated (x, y, z keep the last position; every other attribute is the "
                "-1e30 sentinel)"
            ),
            "frames": list(self._frames),
            "tracker": dict(record),
            "cm1_authority": {
                "write_layout": "src/parcel.F:1626-1690 (writepdata), transposed once as (frames, npvals, nparcels)",
            },
        }
        with (self.root / NEST_PARCEL_RECEIPT_FILENAME).open("xb", buffering=0) as handle:
            handle.write((json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            os.fsync(handle.fileno())
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return receipt


# ---------------------------------------------------------------------------
# The child frame's placement, beside each published child history frame.
# ---------------------------------------------------------------------------
def child_placement_document(*, index: int, step: int, model_time_s: float, geometry: Any,
                             child_config: Any, domainlocx_m: float, domainlocy_m: float) -> dict[str, Any]:
    """Everything an offline reader needs to put one child frame back on the ground."""

    from .nest_vertical import levels_of_grid

    deck = child_deck(geometry, child_config.grid)
    levels = levels_of_grid(child_config.grid)
    return {
        "schema": NEST_CHILD_PLACEMENT_SCHEMA,
        "index": int(index), "step": int(step), "model_time_s": float(model_time_s),
        **_placement(geometry),
        "nx": deck.nx, "ny": deck.ny, "nz": deck.nz, "dx_m": deck.dx_m, "dy_m": deck.dy_m,
        "dz_reference_m": deck.dz_reference_m,
        "zf_cm1_indexed": [float(v) for v in deck.zf],
        "zh_cm1_indexed": [float(v) for v in deck.zh],
        "mf_cm1_indexed": [float(v) for v in deck.mf],
        "cgs": list(deck.cgs), "cgt": list(deck.cgt),
        "vertical_sha256": str(levels.sha256),
        "bbc": int(child_config.bbc), "tbc": int(getattr(child_config, "tbc", 1)),
        "imove": int(getattr(child_config, "imove", int(float(child_config.u_move_m_s) != 0.0
                                                       or float(child_config.v_move_m_s) != 0.0))),
        "u_move_m_s": float(child_config.u_move_m_s), "v_move_m_s": float(child_config.v_move_m_s),
        "domainlocx_m": float(domainlocx_m), "domainlocy_m": float(domainlocy_m),
        "spec_zone": int(geometry.spec_zone), "relax_zone": int(geometry.relax_zone),
        "coordinate_frame": GM1_CHOICES["coordinates"],
        "winds_published": "u3d, v3d, w3d: the completed state at this step (not the parcel step's acoustic average)",
    }


def child_roughness_filename(index: int) -> str:
    return f"frame-{int(index):06d}.znt.f32"


def write_child_placement(child_root: Path, document: Mapping[str, Any], roughness: Any = None) -> Path:
    """The placement JSON, and on a ``bbc = 3`` child the frame's ``znt`` beside it.

    ``roughness`` is the child's host ``(ny, nx)`` ``znt`` at the frame's step:
    an offline reader of a semi-slip child needs it for ``parcel.F:470-515``,
    and it evolves with the wind over water (``sfcphys.F:155-158``), so one
    field per frame is written, little-endian float32, with its SHA-256.
    """

    np = _np()
    directory = Path(child_root) / NEST_CHILD_PLACEMENT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    if int(payload.get("bbc", 1)) == 3:
        if roughness is None:
            raise ValueError(
                "a bbc=3 child frame needs its znt beside the placement; without it an offline "
                "trajectory below zh(1) could not apply parcel.F:470-515's log layer"
            )
        words = np.ascontiguousarray(roughness, dtype="<f4")
        if words.shape != (int(payload["ny"]), int(payload["nx"])):
            raise ValueError(f"the child znt is {words.shape}, not ({payload['ny']}, {payload['nx']})")
        raw = words.tobytes(order="C")
        name = child_roughness_filename(int(payload["index"]))
        with (directory / name).open("xb", buffering=0) as handle:
            handle.write(raw)
            os.fsync(handle.fileno())
        payload["roughness"] = {"file": name, "sha256": hashlib.sha256(raw).hexdigest(),
                                "shape_yx": [int(payload["ny"]), int(payload["nx"])],
                                "carrier": "closure.surface.znt (CM1 znt; zntmp is bc2d-padded from it)"}
    path = directory / f"frame-{int(payload['index']):06d}.json"
    with path.open("xb", buffering=0) as handle:
        handle.write((json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        os.fsync(handle.fileno())
    return path


__all__ = [
    "CM1_AUTHORITY", "GM1_CHOICES", "NEST_CHILD_PLACEMENT_DIRECTORY", "NEST_CHILD_TRACK_DIRECTORY",
    "NEST_PARCEL_KERNEL_SHA256", "NEST_PARCEL_KERNEL_SOURCE", "NEST_PARCEL_SCHEMA",
    "NestParcelSeedExtentError",
    "OWNER_CHILD", "OWNER_PARENT", "OWNER_TERMINATED", "ParcelAttributeHook", "ParcelDeck",
    "ParcelSampleContext", "RuntimeCarriers", "SEED_FRAME_PARENT_GRID_ORIGIN", "STATUS_REASONS",
    "NestParcelTracker", "NestParcelTrackWriter", "UNVALIDATED_MARKER", "child_deck",
    "child_placement_document", "compile_nest_parcel_module", "deck_from_levels",
    "fill_velocity_bottom_ghost", "fill_velocity_ghosts", "fill_velocity_halos", "free_interior_m", "inside_free_interior",
    "launch_nest_parcel_rk2", "nest_parcel_admission_defect", "parent_deck", "reference_rk2",
    "reference_sample_uvw", "reference_search", "reference_stencils", "sample_parcel_attributes",
    "seed_box_positions", "seed_extent_defect", "write_child_placement",
    "CM1_ZNTMP_INITIAL_F32", "SemiSlip", "child_roughness_filename", "libm_logf", "pad_roughness",
    "parcel_roughness", "semi_slip_for",
]
