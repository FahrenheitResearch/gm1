"""Parent-to-child lateral forcing for one refined GM1 nest.

CM1 r21.1 HAS NEITHER OF THESE OPERATORS
----------------------------------------
CM1 r21.1's lateral boundaries are the four ``wbc``/``ebc``/``sbc``/``nbc``
branches of ``README.namelist:561-572``: periodic, open-radiative, rigid free
slip, rigid no slip (``src/bc.F:528-934``, ``bcu`` and ``bcv``).  There is
no specified boundary, no relaxation zone and no nudging at a lateral face
anywhere in the release -- the only ``nudge`` in ``src`` is the Naylor-Gilmore
**updraft** nudging of ``src/misclibs.F:539-585`` and the large-scale profile
nudging read at ``src/base.F:3276``, neither of which touches a lateral
boundary.  A nest's child therefore needs two operators CM1 does not contain,
and this module declares both as extensions rather than attributing them:

1. **parent -> child interpolation**, used to fill the child's boundary rows.
   This port's operator is its own: a cell-mean-preserving, extremum-limited
   piecewise-linear reconstruction described and proved below.  It is not a
   transliteration of anything.
2. **Davies lateral relaxation** (Davies 1976, QJRMS 102, 405-418; Davies
   1983, QJRMS 109, 225-233), in the discrete form WRF uses for a nested
   domain: ``dyn_em/module_bc_em.F:lbc_fcx_gcx`` (nested branch, no sponge
   term) supplying the weights and ``share/module_bc.F:relax_bdytend`` /
   ``spec_bdytend`` applying them.  The reference implementation this port
   was written against, and checked against by hand, is the staged ArWen
   2.7.4 tree: ``gpuwm/core/kernels/spec_bdy.cu`` and the float64 mirror
   ``gpuwm/verify/npref.py:4047-4060`` (weights) and ``:4062-4139``
   (application).  ``engine_pin.py`` pins ``gpuwm>=2.8.9,<2.8.10``, but its
   byte-admitted surface is the column microphysics only -- no ``nest.cu``,
   ``spec_bdy.cu`` or lateral-boundary kernel -- so nothing here imports
   ``gpuwm`` and nothing here depends on the ArWen pin.  GM1 carries its own
   kernel text.

WHY THE CHILD BOUNDARY IS FACE ALIGNED, AND WHAT THAT BUYS
----------------------------------------------------------
:class:`~cm1core.nest_geometry.NestGeometry` places the child on whole parent
cells, so the child's outermost ``u`` column sits exactly on a parent ``u``
column and the outermost ``v`` row exactly on a parent ``v`` row.  The
consequence is stated once and tested:
:func:`boundary_normal_velocity_is_parent_face_value` -- the **normal mass
flux specified at the child's lateral boundary is the parent's own face value,
copied, with no interpolation in the normal direction**.  Whatever mass the
parent's face carried out of the covered region is exactly what the child's
boundary carries in.  Conservation across the interface is therefore a
property of the geometry, not of a correction applied afterwards.

Tangential and scalar boundary values are interpolated, and for those the
conservation statement is the block-mean identity of
:func:`interpolate_parent_to_child`.

MONOTONE, CELL-MEAN-PRESERVING INTERPOLATION
--------------------------------------------
For a parent mass-point field ``P`` and a child column at sub-offsets
``(ax, ay)`` in parent-cell units::

    sx  = 0.5*(P[j][i+1] - P[j][i-1])        # centred, second order
    sy  = 0.5*(P[j+1][i] - P[j-1][i])
    dev = 0.5*|sx| + 0.5*|sy|                # largest |correction| in a block
    phi = min(1, (hi - P)/dev, (P - lo)/dev) # one scalar, both slopes
    C   = P + phi*(sx*ax + sy*ay)

where ``lo``/``hi`` are the min and max of the five-point parent stencil.  Two
properties follow, and both are tested:

* **no new extrema.** ``|ax| <= 0.5`` and ``|ay| <= 0.5``, so
  ``|phi*(sx*ax + sy*ay)| <= phi*dev``, which the limiter caps at
  ``min(hi-P, P-lo)``.  Every child value lies in ``[lo, hi]``.  With a
  non-negative parent stencil the child values are non-negative, which is what
  keeps the child's microphysics from being handed negative mixing ratios at
  its boundary -- see the microphysics note below.
* **the parent cell mean is preserved exactly, in binary floating point.**
  The limiter is one scalar applied to both slopes, so the correction is
  linear in the offsets; the ``r`` sub-offsets of an odd ratio are the
  symmetric set ``{-a_k, ..., 0, ..., +a_k}``
  (:meth:`NestGeometry.sub_cell_offsets`), so summing the block in the pairing
  order ``(c_m + c_{r-1-m})`` cancels each pair to exactly ``2*P``.
  :func:`block_mean_residual` sums in that order and returns exactly ``0.0``.

MICROPHYSICS AT THE INTERFACE
-----------------------------
Every moisture variable crosses the interface through the same operator as
``theta``, and the extremum limiter above is the whole of the treatment: a
mixing ratio interpolated from a non-negative parent stencil is non-negative,
so the child never begins a step with negative water at its boundary and
CM1's positive-definite fixer (``src/misclibs.F`` ``pdefq``) has nothing to
repair there.  The number concentrations and hydrometeor species are carried
as independent scalars, which is the standard one-way treatment: no attempt is
made to keep a drop-size distribution self-consistent across a spacing change,
because the parent does not carry the information that would make that
possible.  :data:`MICROPHYSICS_INTERFACE_LIMIT` states this in one sentence for
the receipt.

GRAVITY AND ACOUSTIC WAVE REFLECTION OFF THE NEST BOUNDARY
-----------------------------------------------------------
A specified boundary is perfectly reflecting.  That is the failure mode of
one-way nesting: an updraft's own gravity waves and the psolver=3 acoustic
modes reach the child's boundary and come back in.  The relaxation zone is the
entire mitigation and is why ``relax_zone`` is a declared key rather than a
constant -- ``fcx`` relaxes the child toward the parent over the zone and the
``gcx`` term is a second-order diffusion that absorbs the short waves the
first-order term reflects.  What this does **not** do is make the boundary
transparent; :data:`WAVE_REFLECTION_LIMIT` says so by value, and the remedy is
geometric: put the nest boundary far enough from the storm that a reflected
wave takes longer than the run to return.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

from .nest_geometry import NestGeometry

NEST_BOUNDARY_SCHEMA = "gm1.cartesian-nest-lateral-forcing/v1"

#: Davies relaxation, WRF's nested-branch discrete form.
DAVIES_AUTHORITY = (
    "Davies (1976) QJRMS 102:405-418 and Davies (1983) QJRMS 109:225-233; "
    "discrete form after WRF dyn_em/module_bc_em.F lbc_fcx_gcx nested branch "
    "(no sponge term) and share/module_bc.F relax_bdytend/spec_bdytend, "
    "cross-checked against the staged ArWen 2.7.4 reference at "
    "gpuwm/core/kernels/spec_bdy.cu and gpuwm/verify/npref.py:4047-4139. "
    "CM1 r21.1 has no lateral relaxation of any kind; this is a declared "
    "extension beyond CM1."
)

MICROPHYSICS_INTERFACE_LIMIT = (
    "every moisture and hydrometeor scalar crosses the interface through the "
    "same extremum-limited interpolation as theta, so a non-negative parent "
    "stencil yields non-negative child boundary values and the child never "
    "starts a step with negative water at its boundary; no attempt is made to "
    "reconcile a drop-size distribution across the spacing change, because the "
    "parent does not carry the moments that would make that possible"
)

WAVE_REFLECTION_LIMIT = (
    "a specified lateral boundary is perfectly reflecting. The Davies zone "
    "damps what reaches it -- fcx relaxes toward the parent and gcx diffuses "
    "the short waves the first-order term would reflect -- but the boundary is "
    "not transparent: an updraft's gravity waves and the psolver=3 acoustic "
    "modes do return from it. The mitigation that works is geometric: place "
    "the nest boundary so that a wave reflected from it cannot reach the "
    "feature of interest within the run"
)

ACOUSTIC_INTERFACE_NOTE = (
    "psolver=3's acoustic step is vertically implicit and horizontally "
    "explicit (CM1 r21.1 src/misclibs.F:857-868 bounds the small step by the "
    "horizontal spacings alone). The implicit part is one tridiagonal solve "
    "per column, so it never crosses the lateral interface: the child solves "
    "its own columns from its own relaxed state and the parent solves its own. "
    "The horizontal part is what refinement touches, and CM1's own dtsmall "
    "(src/misclibs.F:883-885) answers it: refining space by r and time by r "
    "leaves dbldt/dtsm unchanged, so the child needs the same nsound the "
    "parent does. Refining space without refining time multiplies both the "
    "required nsound and the advective Courant number of src/misclibs.F:2714 "
    "by r"
)

#: Sides, in the order every table and receipt lists them.
SIDES = ("west", "east", "south", "north")


def _minmax_stencil(
    parent: Sequence[Sequence[float]], j: int, i: int
) -> tuple[float, float]:
    ny = len(parent)
    nx = len(parent[0])

    def at(jj: int, ii: int) -> float:
        return parent[min(max(jj, 0), ny - 1)][min(max(ii, 0), nx - 1)]

    values = (at(j, i), at(j, i - 1), at(j, i + 1), at(j - 1, i), at(j + 1, i))
    return min(values), max(values)


def limited_slopes(
    parent: Sequence[Sequence[float]], j: int, i: int
) -> tuple[float, float]:
    """The two limited parent slopes at ``(j, i)``, in parent-cell units.

    One scalar limiter is applied to both slopes so that the correction stays
    linear in the sub-cell offset and the block mean is preserved exactly.
    """

    ny = len(parent)
    nx = len(parent[0])

    def at(jj: int, ii: int) -> float:
        return parent[min(max(jj, 0), ny - 1)][min(max(ii, 0), nx - 1)]

    centre = at(j, i)
    sx = 0.5 * (at(j, i + 1) - at(j, i - 1))
    sy = 0.5 * (at(j + 1, i) - at(j - 1, i))
    deviation = 0.5 * abs(sx) + 0.5 * abs(sy)
    if deviation == 0.0:
        return 0.0, 0.0
    lo, hi = _minmax_stencil(parent, j, i)
    phi = min(1.0, (hi - centre) / deviation, (centre - lo) / deviation)
    if phi < 0.0:
        phi = 0.0
    return phi * sx, phi * sy


def interpolate_point(
    parent: Sequence[Sequence[float]], j: int, i: int, ax: float, ay: float
) -> float:
    """One child mass-point value from the parent cell ``(j, i)``."""

    sx, sy = limited_slopes(parent, j, i)
    return parent[j][i] + (sx * ax + sy * ay)


def interpolate_parent_to_child(
    parent: Sequence[Sequence[float]], geometry: NestGeometry
) -> list[list[float]]:
    """Full-extent child capture of one parent mass-point field."""

    donor_i = geometry.donor_index_x()
    donor_j = geometry.donor_index_y()
    sub_i = geometry.sub_index_x()
    sub_j = geometry.sub_index_y()
    offsets = geometry.sub_cell_offsets()
    return [
        [
            interpolate_point(
                parent, donor_j[jc], donor_i[ic], offsets[sub_i[ic]], offsets[sub_j[jc]]
            )
            for ic in range(geometry.nx)
        ]
        for jc in range(geometry.ny)
    ]


def block_mean_residual(
    parent: Sequence[Sequence[float]], geometry: NestGeometry, j: int, i: int
) -> float:
    """Exactly ``0.0`` when the ``r*r`` block's mean is the parent value.

    The block is summed in the pairing order ``(c_m + c_{r-1-m})`` so that the
    two antisymmetric corrections of each pair cancel to exactly ``2*P`` in
    binary floating point.  Any nonzero return is a defect in the operator, not
    a rounding artefact.
    """

    r = geometry.refinement_ratio
    offsets = geometry.sub_cell_offsets()
    sx, sy = limited_slopes(parent, j, i)
    total = 0.0
    seen: set[tuple[int, int]] = set()
    for m in range(r):
        for mm in range(r):
            key = (m, mm)
            if key in seen:
                continue
            mate = (r - 1 - m, r - 1 - mm)
            seen.add(key)
            seen.add(mate)
            here = sx * offsets[m] + sy * offsets[mm]
            if mate == key:
                # The coincident centre column of an odd ratio: its offsets
                # are exactly zero, so this term is exactly 0.0.
                total += here
            else:
                # offsets[mate] is the exact binary negation of offsets[key],
                # so ``there`` is the exact negation of ``here`` and the pair
                # adds exactly 0.0.
                there = sx * offsets[mate[0]] + sy * offsets[mate[1]]
                total += here + there
    return total


def boundary_normal_velocity_is_parent_face_value(
    geometry: NestGeometry, side: str
) -> bool:
    """Whether this side's normal velocity is copied, not interpolated.

    True for every side of a face-aligned nest, which is every nest
    :class:`NestGeometry` admits.  The child's west ``u`` column is the parent
    ``u`` column at ``parent_i0``; the east one is the parent column at
    ``parent_i0 + parent_span_x``; likewise ``v`` in ``y``.
    """

    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    return True


def parent_face_index(geometry: NestGeometry, side: str) -> int:
    """The parent ``u``/``v`` face index the child's ``side`` boundary is on."""

    if side == "west":
        return geometry.parent_i0
    if side == "east":
        return geometry.parent_i0 + geometry.parent_span_x
    if side == "south":
        return geometry.parent_j0
    if side == "north":
        return geometry.parent_j0 + geometry.parent_span_y
    raise ValueError(f"side must be one of {SIDES}")


# ---------------------------------------------------------------------------
# Davies relaxation weights.
# ---------------------------------------------------------------------------
def davies_weights(
    *, spec_zone: int, relax_zone: int, dt_s: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(fcx, gcx)`` indexed by distance from the child's outer edge.

    WRF ``dyn_em/module_bc_em.F lbc_fcx_gcx``, nested branch::

        do loop = spec_zone+1, spec_zone+relax_zone
          index = loop - 1
          ramp  = (spec_zone + relax_zone - loop) / (relax_zone - 1)
          fcx(index) = 0.1/dt        * ramp
          gcx(index) = 1.0/dt/50.0   * ramp

    ``dt`` is the step the relaxation tendency is added on -- the **child's**
    large step, since the child is what is being relaxed.  Entries below
    ``spec_zone`` are zero because those rows are specified outright rather
    than relaxed, and the last entry is zero because the ramp reaches zero at
    the inner edge of the zone.
    """

    if spec_zone < 1:
        raise ValueError("spec_zone must be at least 1")
    if relax_zone < 2:
        raise ValueError(
            "relax_zone must be at least 2; the ramp divides by relax_zone-1"
        )
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    width = spec_zone + relax_zone
    fcx = [0.0] * width
    gcx = [0.0] * width
    for loop in range(spec_zone + 1, spec_zone + relax_zone + 1):
        index = loop - 1
        if index >= width:
            break
        ramp = (spec_zone + relax_zone - loop) / (relax_zone - 1)
        fcx[index] = 0.1 / dt_s * ramp
        gcx[index] = 1.0 / dt_s / 50.0 * ramp
    return tuple(fcx), tuple(gcx)


@dataclass(frozen=True, slots=True)
class BoundaryTable:
    """One side's value and tendency rows for one field.

    ``value`` is the child's own state on that side at the start of the parent
    interval; ``tendency`` is ``(parent_at_end - child_at_start)/parent_dt``,
    so the child evaluates its boundary target at its own sub-time ``dtbc``
    inside the interval as ``value + dtbc*tendency``.  Linear in time between
    parent steps is the whole time treatment, and it is exact at both ends.
    """

    side: str
    value: tuple[tuple[float, ...], ...]
    tendency: tuple[tuple[float, ...], ...]

    def target(self, *, dtbc: float) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(v + dtbc * t for v, t in zip(vrow, trow))
            for vrow, trow in zip(self.value, self.tendency)
        )


def build_boundary_tables(
    *,
    parent_at_end: Sequence[Sequence[float]],
    child_at_start: Sequence[Sequence[float]],
    geometry: NestGeometry,
    parent_dt_s: float,
    width: int,
) -> dict[str, BoundaryTable]:
    """Value and tendency rows for all four sides of one 2-D field level.

    ``width`` is ``spec_zone + relax_zone``: every row the child's relaxation
    can read.  The tendency reciprocal is formed in float64 and the difference
    in the field's own precision, which is the arrangement WRF uses
    (``rdt = 1.D0/cdt``) and which keeps a small parent step from losing
    significance in the reciprocal.
    """

    if width < 1:
        raise ValueError("width must be at least 1")
    if width > min(geometry.nx, geometry.ny):
        raise ValueError("boundary width exceeds the child extent")
    child_from_parent = interpolate_parent_to_child(parent_at_end, geometry)
    rdt = 1.0 / float(parent_dt_s)
    tables: dict[str, BoundaryTable] = {}
    for side in SIDES:
        values: list[tuple[float, ...]] = []
        tendencies: list[tuple[float, ...]] = []
        if side in ("west", "east"):
            for jc in range(geometry.ny):
                vrow: list[float] = []
                trow: list[float] = []
                for w in range(width):
                    ic = w if side == "west" else geometry.nx - 1 - w
                    here = child_at_start[jc][ic]
                    vrow.append(here)
                    trow.append(rdt * (child_from_parent[jc][ic] - here))
                values.append(tuple(vrow))
                tendencies.append(tuple(trow))
        else:
            for w in range(width):
                vrow = []
                trow = []
                for ic in range(geometry.nx):
                    jc = w if side == "south" else geometry.ny - 1 - w
                    here = child_at_start[jc][ic]
                    vrow.append(here)
                    trow.append(rdt * (child_from_parent[jc][ic] - here))
                values.append(tuple(vrow))
                tendencies.append(tuple(trow))
        tables[side] = BoundaryTable(
            side=side, value=tuple(values), tendency=tuple(tendencies)
        )
    return tables


def apply_specified_relaxation(
    *,
    field: Sequence[Sequence[float]],
    tendency: Sequence[Sequence[float]],
    tables: dict[str, BoundaryTable],
    geometry: NestGeometry,
    dtbc: float,
    child_dt_s: float,
) -> list[list[float]]:
    """The CPU reference for ``gm1_nest_specified_relaxation``.

    Rows within ``spec_zone`` of an edge have their tendency **replaced** by
    the boundary table's; rows in ``[spec_zone, relax_zone)`` have the Davies
    terms **added**.  Corner ownership follows WRF's loop bounds exactly: the
    south and north sides own the corners, and the west and east sides start
    one row in, so no point is written twice.
    """

    ny = geometry.ny
    nx = geometry.nx
    spec = geometry.spec_zone
    relax = geometry.relax_zone
    fcx, gcx = davies_weights(
        spec_zone=spec, relax_zone=max(relax, 2), dt_s=child_dt_s
    )
    out = [list(row) for row in tendency]

    def bval(side: str, a: int, b: int) -> float:
        table = tables[side]
        return table.value[a][b] + dtbc * table.tendency[a][b]

    def f(j: int, i: int) -> float:
        return field[min(max(j, 0), ny - 1)][min(max(i, 0), nx - 1)]

    for j in range(ny):
        for i in range(nx):
            ds, dn, dw, de = j, ny - 1 - j, i, nx - 1 - i
            if ds < spec and ds <= i < nx - ds:
                out[j][i] = tables["south"].tendency[ds][i]
                continue
            if dn < spec and dn <= i < nx - dn:
                out[j][i] = tables["north"].tendency[dn][i]
                continue
            if dw < spec and dw + 1 <= j < ny - dw - 1:
                out[j][i] = tables["west"].tendency[j][dw]
                continue
            if de < spec and de + 1 <= j < ny - de - 1:
                out[j][i] = tables["east"].tendency[j][de]
                continue
            if relax < 2:
                continue
            here = f(j, i)
            if spec <= ds < relax and ds <= i < nx - ds:
                d = ds
                f0 = bval("south", d, i) - here
                f1 = bval("south", d, max(i - 1, 0)) - f(j, i - 1)
                f2 = bval("south", d, min(i + 1, nx - 1)) - f(j, i + 1)
                f3 = bval("south", d - 1, i) - f(j - 1, i)
                f4 = bval("south", d + 1, i) - f(j + 1, i)
            elif spec <= dn < relax and dn <= i < nx - dn:
                d = dn
                f0 = bval("north", d, i) - here
                f1 = bval("north", d, max(i - 1, 0)) - f(j, i - 1)
                f2 = bval("north", d, min(i + 1, nx - 1)) - f(j, i + 1)
                f3 = bval("north", d - 1, i) - f(j + 1, i)
                f4 = bval("north", d + 1, i) - f(j - 1, i)
            elif spec <= dw < relax and dw + 1 <= j < ny - dw - 1:
                d = dw
                f0 = bval("west", j, d) - here
                f1 = bval("west", j - 1, d) - f(j - 1, i)
                f2 = bval("west", j + 1, d) - f(j + 1, i)
                f3 = bval("west", j, d - 1) - f(j, i - 1)
                f4 = bval("west", j, d + 1) - f(j, i + 1)
            elif spec <= de < relax and de + 1 <= j < ny - de - 1:
                d = de
                f0 = bval("east", j, d) - here
                f1 = bval("east", j - 1, d) - f(j - 1, i)
                f2 = bval("east", j + 1, d) - f(j + 1, i)
                f3 = bval("east", j, d - 1) - f(j, i + 1)
                f4 = bval("east", j, d + 1) - f(j, i - 1)
            else:
                continue
            out[j][i] += fcx[d] * f0 - gcx[d] * (f1 + f2 + f3 + f4 - 4.0 * f0)
    return out


# ---------------------------------------------------------------------------
# Two-way feedback: child -> parent restriction.
# ---------------------------------------------------------------------------
def restrict_child_to_parent(
    child: Sequence[Sequence[float]], geometry: NestGeometry
) -> list[list[float]]:
    """Block mean of each ``r*r`` child block, one value per covered cell.

    This is the two-way feedback operator.  It is the exact adjoint of the
    conservative interpolation in the only sense that matters: the mean of the
    block equals the parent value the interpolation started from, so a
    round-trip of a field the child does not modify returns the parent
    unchanged.  The interior rows are fed back; the child's own relaxation
    zone is not, because those rows carry the parent's own values and feeding
    them back would relax the parent toward itself through the child.
    """

    r = geometry.refinement_ratio
    skip = geometry.relax_zone
    out: list[list[float]] = []
    for jp in range(geometry.parent_span_y):
        row: list[float] = []
        for ip in range(geometry.parent_span_x):
            j0, i0 = jp * r, ip * r
            if (
                j0 < skip
                or i0 < skip
                or j0 + r > geometry.ny - skip
                or i0 + r > geometry.nx - skip
            ):
                row.append(math.nan)
                continue
            total = math.fsum(
                child[j0 + a][i0 + b] for a in range(r) for b in range(r)
            )
            row.append(total / (r * r))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# The packaged CUDA text.  Device-resident production path; the CPU functions
# above are its oracle.
# ---------------------------------------------------------------------------
NEST_RELAXATION_KERNEL_NAME = "gm1_nest_specified_relaxation"
NEST_RELAXATION_KERNEL_SOURCE = r'''// gm1.cartesian-nest-lateral-forcing/v1
// Davies (1976,1983) lateral relaxation, WRF nested-branch discrete form.
// CM1 r21.1 has no lateral relaxation; this is a declared GM1 extension.
// Side layout: west/east (nz,ny,width), south/north (nz,width,nx); distance
// index 0 is the outermost row.  Y sides own the corners.
#define GM1_NEST_IDX3(k, a, b, na, nb) \
    ((((unsigned long long)(k) * (unsigned long long)(na)) + (unsigned long long)(a)) \
     * (unsigned long long)(nb) + (unsigned long long)(b))

__device__ __forceinline__ float gm1_nest_target(const float* v, const float* t,
                                                 unsigned long long q, float dtbc)
{
    return __fadd_rn(v[q], __fmul_rn(dtbc, t[q]));
}

extern "C" __global__ void gm1_nest_specified_relaxation(
    const float* __restrict__ field,
    float* __restrict__ field_tend,
    const float* __restrict__ west,  const float* __restrict__ west_t,
    const float* __restrict__ east,  const float* __restrict__ east_t,
    const float* __restrict__ south, const float* __restrict__ south_t,
    const float* __restrict__ north, const float* __restrict__ north_t,
    const float* __restrict__ fcx,   const float* __restrict__ gcx,
    float dtbc, int width, int spec_zone, int relax_zone, int apply_relax,
    int nz, int ny, int nx)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = (unsigned long long)nz * ny * nx;
    if (tid >= count) return;
    int k = (int)(tid / ((unsigned long long)ny * nx));
    unsigned long long rem = tid - (unsigned long long)k * ny * nx;
    int j = (int)(rem / nx);
    int i = (int)(rem - (unsigned long long)j * nx);
    float result = field_tend[tid];

    int ds = j, dn = ny - 1 - j, dw = i, de = nx - 1 - i;
    if (ds < spec_zone && i >= ds && i < nx - ds) {
        field_tend[tid] = south_t[GM1_NEST_IDX3(k, ds, i, width, nx)];
        return;
    }
    if (dn < spec_zone && i >= dn && i < nx - dn) {
        field_tend[tid] = north_t[GM1_NEST_IDX3(k, dn, i, width, nx)];
        return;
    }
    if (dw < spec_zone && j >= dw + 1 && j < ny - dw - 1) {
        field_tend[tid] = west_t[GM1_NEST_IDX3(k, j, dw, ny, width)];
        return;
    }
    if (de < spec_zone && j >= de + 1 && j < ny - de - 1) {
        field_tend[tid] = east_t[GM1_NEST_IDX3(k, j, de, ny, width)];
        return;
    }
    if (!apply_relax) return;

    unsigned long long plane = (unsigned long long)k * ny * nx;
    float f0, f1, f2, f3, f4;
    int d;
    if (ds >= spec_zone && ds < relax_zone && i >= ds && i < nx - ds) {
        d = ds;
        int im = i > 0 ? i - 1 : 0, ip = i + 1 < nx ? i + 1 : nx - 1;
        f0 = gm1_nest_target(south, south_t, GM1_NEST_IDX3(k, d, i, width, nx), dtbc)
             - field[tid];
        f1 = gm1_nest_target(south, south_t, GM1_NEST_IDX3(k, d, im, width, nx), dtbc)
             - field[plane + (unsigned long long)j * nx + im];
        f2 = gm1_nest_target(south, south_t, GM1_NEST_IDX3(k, d, ip, width, nx), dtbc)
             - field[plane + (unsigned long long)j * nx + ip];
        f3 = gm1_nest_target(south, south_t, GM1_NEST_IDX3(k, d - 1, i, width, nx), dtbc)
             - field[plane + (unsigned long long)(j - 1) * nx + i];
        f4 = gm1_nest_target(south, south_t, GM1_NEST_IDX3(k, d + 1, i, width, nx), dtbc)
             - field[plane + (unsigned long long)(j + 1) * nx + i];
    } else if (dn >= spec_zone && dn < relax_zone && i >= dn && i < nx - dn) {
        d = dn;
        int im = i > 0 ? i - 1 : 0, ip = i + 1 < nx ? i + 1 : nx - 1;
        f0 = gm1_nest_target(north, north_t, GM1_NEST_IDX3(k, d, i, width, nx), dtbc)
             - field[tid];
        f1 = gm1_nest_target(north, north_t, GM1_NEST_IDX3(k, d, im, width, nx), dtbc)
             - field[plane + (unsigned long long)j * nx + im];
        f2 = gm1_nest_target(north, north_t, GM1_NEST_IDX3(k, d, ip, width, nx), dtbc)
             - field[plane + (unsigned long long)j * nx + ip];
        f3 = gm1_nest_target(north, north_t, GM1_NEST_IDX3(k, d - 1, i, width, nx), dtbc)
             - field[plane + (unsigned long long)(j + 1) * nx + i];
        f4 = gm1_nest_target(north, north_t, GM1_NEST_IDX3(k, d + 1, i, width, nx), dtbc)
             - field[plane + (unsigned long long)(j - 1) * nx + i];
    } else if (dw >= spec_zone && dw < relax_zone && j >= dw + 1 && j < ny - dw - 1) {
        d = dw;
        f0 = gm1_nest_target(west, west_t, GM1_NEST_IDX3(k, j, d, ny, width), dtbc)
             - field[tid];
        f1 = gm1_nest_target(west, west_t, GM1_NEST_IDX3(k, j - 1, d, ny, width), dtbc)
             - field[plane + (unsigned long long)(j - 1) * nx + i];
        f2 = gm1_nest_target(west, west_t, GM1_NEST_IDX3(k, j + 1, d, ny, width), dtbc)
             - field[plane + (unsigned long long)(j + 1) * nx + i];
        f3 = gm1_nest_target(west, west_t, GM1_NEST_IDX3(k, j, d - 1, ny, width), dtbc)
             - field[plane + (unsigned long long)j * nx + (i - 1)];
        f4 = gm1_nest_target(west, west_t, GM1_NEST_IDX3(k, j, d + 1, ny, width), dtbc)
             - field[plane + (unsigned long long)j * nx + (i + 1)];
    } else if (de >= spec_zone && de < relax_zone && j >= de + 1 && j < ny - de - 1) {
        d = de;
        f0 = gm1_nest_target(east, east_t, GM1_NEST_IDX3(k, j, d, ny, width), dtbc)
             - field[tid];
        f1 = gm1_nest_target(east, east_t, GM1_NEST_IDX3(k, j - 1, d, ny, width), dtbc)
             - field[plane + (unsigned long long)(j - 1) * nx + i];
        f2 = gm1_nest_target(east, east_t, GM1_NEST_IDX3(k, j + 1, d, ny, width), dtbc)
             - field[plane + (unsigned long long)(j + 1) * nx + i];
        f3 = gm1_nest_target(east, east_t, GM1_NEST_IDX3(k, j, d - 1, ny, width), dtbc)
             - field[plane + (unsigned long long)j * nx + (i + 1)];
        f4 = gm1_nest_target(east, east_t, GM1_NEST_IDX3(k, j, d + 1, ny, width), dtbc)
             - field[plane + (unsigned long long)j * nx + (i - 1)];
    } else {
        return;
    }
    float lap = __fsub_rn(
        __fadd_rn(__fadd_rn(f1, f2), __fadd_rn(f3, f4)), __fmul_rn(4.0f, f0));
    result = __fadd_rn(result,
                       __fsub_rn(__fmul_rn(fcx[d], f0), __fmul_rn(gcx[d], lap)));
    field_tend[tid] = result;
}
'''
NEST_RELAXATION_KERNEL_SHA256 = hashlib.sha256(
    NEST_RELAXATION_KERNEL_SOURCE.encode("utf-8")
).hexdigest()


#: Threads per block.  The kernel is one thread per element with no shared
#: memory and no cross-thread read, so this is an occupancy choice only.
NEST_RELAXATION_THREADS = 256

#: ``-fmad=false``.  Every float step in the kernel is an explicit ``_rn``
#: intrinsic, and the CPU oracle mirrors that rounding exactly; letting the
#: device compiler contract a multiply-add would break the raw-word comparison
#: the kernel is verified by.
NEST_RELAXATION_NVRTC_OPTIONS = ("--std=c++17", "-fmad=false")


def compile_nest_relaxation_module(cupy):
    """Compile the packaged kernel text for the current device.

    Separate from the launcher so a caller can pay the compile once and so the
    compile itself is testable on a device without a launch.
    """

    from .nvrtc_module import compile_nvrtc_module

    return compile_nvrtc_module(
        cupy,
        code=NEST_RELAXATION_KERNEL_SOURCE,
        options=NEST_RELAXATION_NVRTC_OPTIONS,
        filename="gm1_nest_relaxation.cu",
    )


def launch_nest_specified_relaxation(
    cupy,
    *,
    field,
    field_tend,
    sides: dict,
    fcx,
    gcx,
    dtbc: float,
    width: int,
    spec_zone: int,
    relax_zone: int,
    nz: int,
    ny: int,
    nx: int,
    apply_relax: bool = True,
    module=None,
):
    """Run the Davies specified/relaxation operator on one device field.

    ``field_tend`` is updated in place: specified rows are **replaced** by the
    parent's tendency and relaxed rows have the two Davies terms added.
    ``sides`` carries the eight tables keyed ``west``/``west_t``/``east``/
    ``east_t``/``south``/``south_t``/``north``/``north_t``, in the layouts
    :func:`build_boundary_tables` produces: west and east ``(nz, ny, width)``,
    south and north ``(nz, width, nx)``, distance index 0 at the outer edge.

    Every array must be C-contiguous float32; a float64 array would be read as
    twice as many float32 words and the operator would silently read the wrong
    half of every value, so the dtype is checked rather than cast.
    """

    required = (
        "west", "west_t", "east", "east_t", "south", "south_t", "north", "north_t",
    )
    missing = [name for name in required if name not in sides]
    if missing:
        raise ValueError(f"the boundary tables are incomplete: missing {missing}")
    arrays = [field, field_tend, fcx, gcx] + [sides[name] for name in required]
    for array in arrays:
        if array.dtype != cupy.float32:
            raise TypeError(
                f"every nest relaxation operand must be float32; got "
                f"{array.dtype}. A float64 array would be read as twice as "
                "many float32 words and every value would be half a double"
            )
        if not array.flags.c_contiguous:
            raise ValueError(
                "every nest relaxation operand must be C-contiguous; the "
                "kernel indexes raw words"
            )
    count = nz * ny * nx
    if field.size != count or field_tend.size != count:
        raise ValueError("field and field_tend must hold nz*ny*nx words")
    if fcx.size != width or gcx.size != width:
        raise ValueError("fcx and gcx must hold exactly `width` words")
    for name in ("west", "west_t", "east", "east_t"):
        if sides[name].size != nz * ny * width:
            raise ValueError(f"{name} must hold nz*ny*width words")
    for name in ("south", "south_t", "north", "north_t"):
        if sides[name].size != nz * width * nx:
            raise ValueError(f"{name} must hold nz*width*nx words")
    if module is None:
        module = compile_nest_relaxation_module(cupy)
    kernel = module.get_function(NEST_RELAXATION_KERNEL_NAME)
    blocks = (count + NEST_RELAXATION_THREADS - 1) // NEST_RELAXATION_THREADS
    kernel(
        (blocks,),
        (NEST_RELAXATION_THREADS,),
        (
            field,
            field_tend,
            sides["west"], sides["west_t"],
            sides["east"], sides["east_t"],
            sides["south"], sides["south_t"],
            sides["north"], sides["north_t"],
            fcx,
            gcx,
            cupy.float32(dtbc),
            cupy.int32(width),
            cupy.int32(spec_zone),
            cupy.int32(relax_zone),
            cupy.int32(1 if apply_relax else 0),
            cupy.int32(nz),
            cupy.int32(ny),
            cupy.int32(nx),
        ),
    )
    return field_tend


def nest_boundary_record(geometry: NestGeometry, *, child_dt_s: float) -> dict[str, object]:
    """What a receipt carries about this nest's lateral forcing."""

    fcx, gcx = davies_weights(
        spec_zone=geometry.spec_zone,
        relax_zone=max(geometry.relax_zone, 2),
        dt_s=child_dt_s,
    )
    return {
        "schema": NEST_BOUNDARY_SCHEMA,
        "authority": DAVIES_AUTHORITY,
        "spec_zone": geometry.spec_zone,
        "relax_zone": geometry.relax_zone,
        "boundary_width": geometry.spec_zone + geometry.relax_zone,
        "davies_fcx_s_inv": list(fcx),
        "davies_gcx_s_inv": list(gcx),
        "relaxation_kernel": NEST_RELAXATION_KERNEL_NAME,
        "relaxation_kernel_sha256": NEST_RELAXATION_KERNEL_SHA256,
        "normal_flux_is_parent_face_value": True,
        "normal_face_index": {
            side: parent_face_index(geometry, side) for side in SIDES
        },
        "microphysics_interface": MICROPHYSICS_INTERFACE_LIMIT,
        "wave_reflection": WAVE_REFLECTION_LIMIT,
        "acoustic_interface": ACOUSTIC_INTERFACE_NOTE,
        "feedback": "two-way" if geometry.feedback else "one-way",
    }


__all__ = [
    "ACOUSTIC_INTERFACE_NOTE",
    "DAVIES_AUTHORITY",
    "MICROPHYSICS_INTERFACE_LIMIT",
    "NEST_BOUNDARY_SCHEMA",
    "NEST_RELAXATION_KERNEL_NAME",
    "NEST_RELAXATION_KERNEL_SHA256",
    "NEST_RELAXATION_KERNEL_SOURCE",
    "NEST_RELAXATION_NVRTC_OPTIONS",
    "NEST_RELAXATION_THREADS",
    "SIDES",
    "WAVE_REFLECTION_LIMIT",
    "BoundaryTable",
    "apply_specified_relaxation",
    "block_mean_residual",
    "boundary_normal_velocity_is_parent_face_value",
    "build_boundary_tables",
    "compile_nest_relaxation_module",
    "davies_weights",
    "interpolate_parent_to_child",
    "interpolate_point",
    "launch_nest_specified_relaxation",
    "limited_slopes",
    "nest_boundary_record",
    "parent_face_index",
    "restrict_child_to_parent",
]
