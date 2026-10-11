"""Build a refined child's initial state from a parent's, one-way.

WHAT THIS IS, AND WHAT IT IS NOT
-------------------------------
This is **sequential downscaling**, not a live nest.  A parent run is
integrated and checkpointed; a finer child domain is then initialized from one
of those checkpoints and integrated on its own.  The parent does not see the
child, and after the initial instant the child is not forced by the parent --
it runs under CM1's own lateral boundary conditions, exactly as any other
single-domain run does.

That distinction is the whole honesty of this module, so it is stated here
rather than in a footnote.  A **live** two-way nest -- the optional ``[nest]``
table, :mod:`cm1core.nest_geometry`, :mod:`cm1core.nest_boundary` and
:mod:`cm1core.nest_coupler` -- keeps the child inside the parent for the whole
run, feeds its lateral rows from the parent every parent step and (with
``feedback = 1``) writes the child's block means back.  That capability is
admitted by ``--integrate`` for a fixed nest (see records/CM1-NEST-SEAM.md
5.4-5.5).  This module is not that: it never declares a ``[nest]`` table and
never claims to be one.  What it produces is a
child whose *initial condition* came from a parent, which is a different and
weaker claim, and the receipt says so by value.

THE VERTICAL: A COPY WHEN THE DECKS AGREE, CM1'S RULE WHEN THEY DO NOT
---------------------------------------------------------------------
A child that carries its parent's deck -- same ``sha256``, bit for bit -- is
copied in the vertical and only the horizontal operator below runs; that path
is unchanged.  A child may instead declare a finer deck over the same model top
(a 5 m lowest layer under a 20 m one).  That transfer goes through
:mod:`cm1core.nest_vertical`: CM1's own ``interp_on_restart`` rule
(``src/restart_read.F:564-609``, ``3302-3470``) applied to each parent column
of a parent-resolution window, then this module's horizontal operator, in
WRF's vertical-then-horizontal order.  :func:`require_identical_vertical_decks`
still exists for the copy path and still refuses there; a caller that wants a
refined deck checks it with ``nest_vertical.require_vertical_refinement``,
which refuses a different model top or a child coarser than its parent
anywhere.

THE HORIZONTAL OPERATOR IS THE NEST'S OWN
-----------------------------------------
:func:`cm1core.nest_boundary.interpolate_parent_to_child` is the cell-mean
preserving, extremum-limited piecewise-linear reconstruction the nest uses to
fill a child's boundary rows.  It is reused here unchanged, for the whole
child extent rather than the boundary rows, because the property that makes it
right at a boundary -- no new extrema, and a block mean equal to the parent
cell exactly in binary floating point -- is exactly what an initial condition
needs: the child must start from the parent's mass and the parent's moisture,
not from an interpolation that has invented a new maximum somewhere.

That reference is a pure-Python nest of lists and costs roughly a microsecond
per point.  One 75-level transfer onto a 500 x 500 child is 18.75 million
points, which is minutes per carrier and hours per restart.  :func:`downscale_level`
is the same arithmetic written over NumPy arrays, and
:func:`assert_matches_reference` is the proof obligation that comes with
saying "the same": it compares raw words, not values within a tolerance.
Every operation in the reference is elementwise -- there is no reduction whose
summation order could change -- so vectorizing it is bit-preserving rather
than merely close, and the test asserts that rather than assuming it.

Arrays cross this module as ``float32``, which is what a restart arena holds.
The reference promotes each word to a Python float (IEEE double) and does its
arithmetic there, so this module promotes to ``float64``, computes, and
narrows back exactly once at the end -- the same two roundings in the same two
places.
"""

from __future__ import annotations

from typing import Any, Sequence

from .errors import refuse
from .moisture_species import MoistureSpeciesProfile
from .nest_boundary import interpolate_parent_to_child
from .nest_geometry import NestGeometry

NEST_DOWNSCALE_SCHEMA = "gm1.cartesian-nest-downscale/v1"

#: Stated in the receipt so a reader of a downscaled run is never left to infer
#: it from the absence of a ``[nest]`` key.
ONE_WAY_LIMIT = (
    "this child was initialized from a parent checkpoint and is not coupled to "
    "it afterwards: its lateral boundaries are CM1's own wbc/ebc/sbc/nbc "
    "branches (src/bc.F:528-934), not a parent-fed Davies specified and "
    "relaxation zone. The parent's later evolution does not reach it, and its "
    "own evolution does not reach the parent. A storm feature that leaves the "
    "child's window is gone; one that approaches the window from outside never "
    "arrives. Frames are comparable to the parent's for as long as the "
    "interior is not yet contaminated by the lateral boundary, which for a "
    "window of W metres and a domain-relative wind of U m/s is of order W/(2U)"
)


def _numpy() -> Any:
    import numpy

    return numpy


def require_identical_vertical_decks(parent_grid: Any, child_grid: Any) -> None:
    """Refuse a transfer whose two columns are not the same column.

    GM1's nest refines the horizontal only, so this operator copies in the
    vertical.  If a child ever carries a different deck, copying would put the
    parent's level ``k`` value at a different height in the child and every
    buoyancy and pressure term after it would be computed against a profile the
    carriers have never seen.  That is a wrong answer that looks like a working
    run, so it refuses here rather than interpolating onto a deck no operator
    in this module has validated.
    """

    from .errors import refuse

    if int(parent_grid.nz) != int(child_grid.nz):
        refuse(
            code="CM1-NEST-DOWNSCALE-VERTICAL",
            gate="cartesian-runtime",
            reason=(
                f"the parent carries {int(parent_grid.nz)} levels and the child "
                f"{int(child_grid.nz)}; this operator copies in the vertical "
                "because GM1's nest refines the horizontal only, and copying "
                "between unequal decks would place each parent level at the "
                "wrong height in the child"
            ),
            remedy="give the child the parent's grid.nz, or add a vertical operator",
            option="grid.nz",
            value=int(child_grid.nz),
        )
    parent_vertical = getattr(parent_grid, "vertical", None)
    child_vertical = getattr(child_grid, "vertical", None)
    if (parent_vertical is None) != (child_vertical is None):
        refuse(
            code="CM1-NEST-DOWNSCALE-VERTICAL",
            gate="cartesian-runtime",
            reason=(
                "one of these two grids carries a stretched vertical coordinate "
                "and the other does not, so level k is at a different height in "
                "each and a vertical copy would misplace every word"
            ),
            remedy="give the child the parent's grid.stretch_z and its four lengths",
            option="grid.stretch_z",
            value=None if child_vertical is None else child_vertical.stretch_z,
        )
    if parent_vertical is None:
        return
    if parent_vertical.sha256 != child_vertical.sha256:
        refuse(
            code="CM1-NEST-DOWNSCALE-VERTICAL",
            gate="cartesian-runtime",
            reason=(
                f"the parent's vertical deck is {parent_vertical.sha256} and the "
                f"child's is {child_vertical.sha256}; the two columns are not "
                "the same column, so a vertical copy would misplace every word"
            ),
            remedy="give the child the parent's vertical-grid words exactly",
            option="grid.stretch_z",
            value=child_vertical.stretch_z,
        )


def limited_slopes_array(parent: Any) -> tuple[Any, Any]:
    """The limited parent slopes at every parent cell, in parent-cell units.

    This is ``nest_boundary.limited_slopes`` for a whole field at once, with
    the same clamped five-point stencil at the edges and the same single
    scalar limiter applied to both slopes.  Every step is elementwise.
    """

    numpy = _numpy()
    field = numpy.asarray(parent, dtype=numpy.float64)
    if field.ndim != 2:
        raise ValueError("a parent mass-point level is two-dimensional")
    ny, nx = field.shape
    rows = numpy.arange(ny)
    cols = numpy.arange(nx)
    # The reference clamps the index, so an edge cell's missing neighbour is
    # the edge cell itself and its one-sided slope is half the interior one.
    jm = numpy.clip(rows - 1, 0, ny - 1)
    jp = numpy.clip(rows + 1, 0, ny - 1)
    im = numpy.clip(cols - 1, 0, nx - 1)
    ip = numpy.clip(cols + 1, 0, nx - 1)
    centre = field
    west = field[:, im]
    east = field[:, ip]
    south = field[jm, :]
    north = field[jp, :]
    slope_x = 0.5 * (east - west)
    slope_y = 0.5 * (north - south)
    deviation = 0.5 * numpy.abs(slope_x) + 0.5 * numpy.abs(slope_y)
    low = numpy.minimum(
        numpy.minimum(numpy.minimum(centre, west), numpy.minimum(east, south)), north
    )
    high = numpy.maximum(
        numpy.maximum(numpy.maximum(centre, west), numpy.maximum(east, south)), north
    )
    flat = deviation == 0.0
    # Divide only where the reference divides; a flat stencil returns (0, 0)
    # before it ever forms the quotient.
    safe = numpy.where(flat, 1.0, deviation)
    phi = numpy.minimum(1.0, numpy.minimum((high - centre) / safe, (centre - low) / safe))
    phi = numpy.where(phi < 0.0, 0.0, phi)
    phi = numpy.where(flat, 0.0, phi)
    return phi * slope_x, phi * slope_y


def downscale_level(parent: Any, geometry: NestGeometry) -> Any:
    """One parent mass-point level captured onto the full child extent.

    Bit-for-bit what :func:`cm1core.nest_boundary.interpolate_parent_to_child`
    returns for the same inputs; :func:`assert_matches_reference` is the proof.
    """

    numpy = _numpy()
    slope_x, slope_y = limited_slopes_array(parent)
    field = numpy.asarray(parent, dtype=numpy.float64)
    donor_i = numpy.asarray(geometry.donor_index_x(), dtype=numpy.intp)
    donor_j = numpy.asarray(geometry.donor_index_y(), dtype=numpy.intp)
    offsets = numpy.asarray(geometry.sub_cell_offsets(), dtype=numpy.float64)
    ax = offsets[numpy.asarray(geometry.sub_index_x(), dtype=numpy.intp)][None, :]
    ay = offsets[numpy.asarray(geometry.sub_index_y(), dtype=numpy.intp)][:, None]
    rows = donor_j[:, None]
    cols = donor_i[None, :]
    # The reference forms ``parent[j][i] + (sx*ax + sy*ay)``: the two products
    # are added to each other first and the centre is added last.  Each of the
    # three operations is exact per element, so the association is what has to
    # match, and this is it.
    return field[rows, cols] + (slope_x[rows, cols] * ax + slope_y[rows, cols] * ay)


#: The two horizontal lattices a CM1 carrier can sit on, per axis.
#:
#: ``"mass"`` is a cell centre: child column ``c`` sits inside parent cell
#: ``c // r`` at the symmetric sub-offset ``(2m + 1 - r)/(2r)``, which is the
#: arrangement :func:`downscale_level` uses and the one the block-mean identity
#: is proved for.
#:
#: ``"face"`` is a cell edge: child face ``c`` sits at parent-face coordinate
#: ``f0 + c/r`` exactly, so it is interpolated between parent FACES rather than
#: between parent centres.  The consequence that matters is at ``c % r == 0``:
#: the offset is then exactly ``0.0`` and the child face value is the parent
#: face value copied, with no arithmetic at all.  That is the same
#: face-alignment statement
#: :func:`cm1_core.nest_boundary.boundary_normal_velocity_is_parent_face_value`
#: makes about the child's outermost normal-velocity column, extended to every
#: coincident face inside the window.
LATTICES = ("mass", "face")


def _axis_donors(
    geometry: NestGeometry, axis: str, lattice: str
) -> tuple[Any, Any]:
    """Donor parent index and sub-cell offset for each child index on one axis.

    Returns ``(donor, offset)`` as NumPy arrays, in parent-cell units, for the
    child's full extent on that axis.
    """

    numpy = _numpy()
    if lattice not in LATTICES:
        raise ValueError(f"lattice must be one of {LATTICES}, not {lattice!r}")
    ratio = geometry.refinement_ratio
    if axis == "x":
        origin, count = geometry.parent_i0, geometry.nx
    elif axis == "y":
        origin, count = geometry.parent_j0, geometry.ny
    else:
        raise ValueError("axis must be 'x' or 'y'")

    if lattice == "mass":
        child = numpy.arange(count)
        donor = origin + child // ratio
        offsets = numpy.asarray(geometry.sub_cell_offsets(), dtype=numpy.float64)
        return donor.astype(numpy.intp), offsets[child % ratio]

    # A face lattice carries one more point than the cell lattice.
    child = numpy.arange(count + 1)
    exact = child / ratio
    # Nearest parent face, and the signed distance to it in parent cells.  A
    # child face that coincides with a parent face gets offset exactly 0.0 and
    # is therefore copied rather than reconstructed.
    nearest = numpy.floor(exact + 0.5).astype(numpy.intp)
    donor = origin + nearest
    offset = exact - nearest
    # ``c % r == 0`` must give exactly zero, not a rounding of it.
    offset = numpy.where(child % ratio == 0, 0.0, offset)
    return donor.astype(numpy.intp), offset


def downscale_level_staggered(
    parent: Any, geometry: NestGeometry, *, x_lattice: str = "mass",
    y_lattice: str = "mass",
) -> Any:
    """Capture one parent level onto the child, on each axis's own lattice.

    ``downscale_level`` is this with both lattices ``"mass"``, and the two
    agree word for word in that case -- :func:`assert_matches_reference` holds
    the mass/mass case against the nest's own reference operator.

    The staggered cases are a **declared extension** of that operator, needed
    because ``u``, ``v`` and ``w`` do not sit at cell centres.  The extension
    is the smallest one available: the same limited five-point slopes,
    evaluated at the child point's own offset from its nearest parent point on
    the same lattice.

    This docstring used to say the operator was admissible for an initial
    condition and not for a re-imposed boundary tendency, on the grounds that
    the block-mean identity does not carry over to a face lattice.  That was a
    worry, not a measurement, and measuring it settled it the other way
    (``tests/test_nest_live_operators.py``): on the normal axis a child face
    that coincides with a parent face has offset exactly zero, and on the
    tangential axis the offsets are the symmetric centre set, so the ``r``
    child faces of one parent face sum -- pair-ordered, in the float64
    reconstruction -- to exactly ``r`` times the parent face value.  That holds
    for every parent face in the window, so the block-integrated divergence of
    the child equals the parent's (measured to 1e-11 of the spacing) and the
    flux a re-imposed boundary tendency carries through the child's boundary
    is the parent's own.  Child faces between parent faces carry no
    conservation obligation.  The live nest's boundary tables and the child's
    initial condition use this operator (``nest_live_kernels.gm1_nest_capture``
    is its device form, raw-word checked against this function), and the
    device consistency and reflection cases in ``CM1-NEST-SEAM.md`` section 5
    are the end-to-end evidence.
    """

    numpy = _numpy()
    slope_x, slope_y = limited_slopes_array(parent)
    field = numpy.asarray(parent, dtype=numpy.float64)
    donor_i, ax = _axis_donors(geometry, "x", x_lattice)
    donor_j, ay = _axis_donors(geometry, "y", y_lattice)
    rows = donor_j[:, None]
    cols = donor_i[None, :]
    return field[rows, cols] + (
        slope_x[rows, cols] * ax[None, :] + slope_y[rows, cols] * ay[:, None]
    )


def downscale_volume_staggered(
    parent: Any, geometry: NestGeometry, *, x_lattice: str = "mass",
    y_lattice: str = "mass",
) -> Any:
    """:func:`downscale_level_staggered` for every level of a volume."""

    numpy = _numpy()
    volume = numpy.asarray(parent)
    if volume.ndim != 3:
        raise ValueError("a parent volume is [nz, ny, nx]")
    first = downscale_level_staggered(
        volume[0], geometry, x_lattice=x_lattice, y_lattice=y_lattice
    )
    captured = numpy.empty((volume.shape[0],) + first.shape, dtype=numpy.float64)
    captured[0] = first
    for k in range(1, volume.shape[0]):
        captured[k] = downscale_level_staggered(
            volume[k], geometry, x_lattice=x_lattice, y_lattice=y_lattice
        )
    return captured


def downscale_volume(parent: Any, geometry: NestGeometry) -> Any:
    """A ``[nz, ny, nx]`` parent volume captured onto the child's columns.

    The vertical is copied level for level, which is only correct because the
    two decks are the same deck; :func:`require_identical_vertical_decks` is
    what makes that a checked precondition rather than an assumption.
    """

    numpy = _numpy()
    volume = numpy.asarray(parent)
    if volume.ndim != 3:
        raise ValueError("a parent volume is [nz, ny, nx]")
    captured = numpy.empty(
        (volume.shape[0], geometry.ny, geometry.nx), dtype=numpy.float64
    )
    for k in range(volume.shape[0]):
        captured[k] = downscale_level(volume[k], geometry)
    return captured


def assert_matches_reference(parent: Sequence[Sequence[float]], geometry: NestGeometry) -> None:
    """Raise unless the vectorized operator reproduces the reference's words.

    Raw words, not a tolerance: the reference's arithmetic is elementwise, so
    "the same operator, faster" is a claim about bits and is checkable as one.
    """

    numpy = _numpy()
    expected = numpy.asarray(
        interpolate_parent_to_child(parent, geometry), dtype=numpy.float64
    )
    measured = downscale_level(parent, geometry)
    if expected.shape != measured.shape:
        raise AssertionError(
            f"shape {measured.shape} does not match the reference's {expected.shape}"
        )
    same = expected.view(numpy.uint64) == measured.view(numpy.uint64)
    if not bool(same.all()):
        index = tuple(int(v) for v in numpy.argwhere(~same)[0])
        raise AssertionError(
            f"the vectorized capture differs from the reference at {index}: "
            f"{measured[index]!r} against {expected[index]!r}"
        )


def downscale_record(
    geometry: NestGeometry,
    *,
    parent_restart: str,
    parent_step: int,
    parent_model_time_s: float,
    carriers: Sequence[str],
    vertical: Any = None,
) -> dict[str, object]:
    """What a downscaled child says about where it came from.

    ``vertical`` is a :class:`cm1core.nest_vertical.VerticalRefinement`; absent
    or identical, the record states the level-for-level copy it always has.
    """

    record = {
        "schema": NEST_DOWNSCALE_SCHEMA,
        "coupling": "one-way-initial-condition-only",
        "limit": ONE_WAY_LIMIT,
        "operator": "cm1core.nest_boundary.interpolate_parent_to_child",
        "operator_form": "cell-mean-preserving extremum-limited piecewise linear",
        "vertical_treatment": "copied level for level; the decks are identical",
        "parent_restart": parent_restart,
        "parent_step": int(parent_step),
        "parent_model_time_s": float(parent_model_time_s),
        "refinement_ratio": geometry.refinement_ratio,
        "parent_i0": geometry.parent_i0,
        "parent_j0": geometry.parent_j0,
        "parent_span_x": geometry.parent_span_x,
        "parent_span_y": geometry.parent_span_y,
        "child_nx": geometry.nx,
        "child_ny": geometry.ny,
        "child_dx_m": geometry.dx_m,
        "child_dy_m": geometry.dy_m,
        "carriers": list(carriers),
    }
    if vertical is not None and not vertical.identical:
        refinement = vertical.record()
        record["vertical_treatment"] = refinement["vertical_treatment"]
        record["vertical_refinement"] = refinement
    return record


NEST_DOWNSCALE_SPECIES_SCHEMA = "gm1.nest-downscale-species/v1"

#: The one initialization of an absent number concentration this port admits,
#: and only for a scheme whose own code supplies the relation.  The child's
#: rows are written as exact zero and the scheme's slope bound rebuilds each
#: number from its mass on the first microphysics call: with N = 0 and
#: q >= qsmall the slope lambda = (c*N/q)^(1/d) is 0, below the scheme's
#: minimum, so the scheme sets lambda to that minimum and N from it.
SCHEME_LAMBDA_BOUND = "scheme-lambda-bound"
SCHEME_LAMBDA_BOUND_AUTHORITY: dict[str, dict[str, object]] = {
    "morrison-mp10": {
        "species": ("nci", "ncs", "ncr", "ncg"),
        "cm1": (
            "src/morrison.F:1876-1885 (rain, LAMMINR), :1947-1956 (snow, "
            "LAMMINS), :1971-1980 (graupel, LAMMING), :2796-2805 (ice, LAMMINI)"
        ),
        "arwen": "gpuwm/core/kernels/morrison.cu:99-113 morr_bound_one, called at :146-154",
        "consequence": (
            "each rebuilt number is the one at the scheme's minimum slope, i.e. "
            "the largest mean particle size the scheme admits for that mass; it "
            "is the scheme's bound, not a measured or climatological size "
            "distribution, and it holds until the scheme's own processes evolve it"
        ),
    },
}


def plan_species_transfer(
    parent: MoistureSpeciesProfile,
    child: MoistureSpeciesProfile,
    *,
    missing_numbers: str | None = None,
) -> dict[str, Any]:
    """Map a child's q3d rows onto a parent's scalars by CM1 name, or refuse.

    A row is never matched by position.  Every child mass mixing ratio must
    exist in the parent: there is no authority for inventing water.  A child
    number concentration the parent does not carry is refused unless
    ``missing_numbers`` names the child scheme's own relation
    (:data:`SCHEME_LAMBDA_BOUND`).  A parent mass species the child cannot
    carry is refused too, because dropping it removes water from the child's
    initial state; a parent number concentration the child does not carry is
    recorded as not transferred, since the child scheme has no such moment.
    """

    gate, code = "cartesian-runtime", "CM1-NEST-DOWNSCALE-SPECIES"
    if missing_numbers not in (None, SCHEME_LAMBDA_BOUND):
        refuse(
            code=code, gate=gate,
            reason=(
                f"{missing_numbers!r} is not an initialization this port admits "
                "for an absent number concentration"
            ),
            remedy=f"omit it, or pass {SCHEME_LAMBDA_BOUND!r}",
            option="missing_number_concentrations", value=missing_numbers,
        )
    parent_kind = {s.name: s.kind for s in parent.scalars}
    child_kind = {s.name: s.kind for s in child.scalars}
    for name in set(parent_kind) & set(child_kind):
        if parent_kind[name] != child_kind[name]:
            refuse(
                code=code, gate=gate,
                reason=(
                    f"{name} is a {parent_kind[name]} in {parent.component_id} and a "
                    f"{child_kind[name]} in {child.component_id}; copying it would "
                    "reinterpret a count as a mass or a mass as a count"
                ),
                remedy="run parent and child with the same physics.component",
                option="physics.component", value=child.component_id,
            )
    lost_mass = [n for n in parent.mass_names if n not in child_kind]
    if lost_mass:
        refuse(
            code=code, gate=gate,
            reason=(
                f"the parent {parent.component_id} carries {', '.join(lost_mass)} and "
                f"the child {child.component_id} has no row for them; the child's "
                "initial state would silently lose that water"
            ),
            remedy="run the child with a component that carries every parent mass species",
            option="physics.component", value=child.component_id,
        )
    absent = [n for n in child.names if n not in parent_kind]
    absent_mass = [n for n in absent if child_kind[n] != "number-concentration"]
    if absent_mass:
        refuse(
            code=code, gate=gate,
            reason=(
                f"the child {child.component_id} needs {', '.join(absent_mass)} and the "
                f"parent frame from {parent.component_id} carries none; no CM1 "
                "relation initializes a mass mixing ratio from another scheme's state"
            ),
            remedy="downscale from a parent run with the child's physics.component",
            option="physics.component", value=child.component_id,
        )
    authority = SCHEME_LAMBDA_BOUND_AUTHORITY.get(child.component_id)
    if absent and missing_numbers is None:
        remedy = "downscale from a parent run with the child's physics.component"
        if authority is not None:
            remedy += (
                f", or pass --missing-number-concentrations {SCHEME_LAMBDA_BOUND} to "
                f"write exact zero and let {child.component_id}'s own slope bound "
                f"rebuild them from mass ({authority['cm1']})"
            )
        refuse(
            code=code, gate=gate,
            reason=(
                f"the child {child.component_id} transports {', '.join(absent)} and the "
                f"parent frame from {parent.component_id} carries none; a zero-filled "
                "buffer would hand the scheme hydrometeor mass with no particles, "
                "which is a silent change to the child's microphysics, not a copy"
            ),
            remedy=remedy,
            option="physics.component", value=child.component_id,
        )
    if absent and missing_numbers == SCHEME_LAMBDA_BOUND:
        admitted = () if authority is None else tuple(authority["species"])  # type: ignore[arg-type]
        unsupported = [n for n in absent if n not in admitted]
        if unsupported:
            refuse(
                code=code, gate=gate,
                reason=(
                    f"{child.component_id} supplies no slope-bound relation for "
                    f"{', '.join(unsupported)}, so {SCHEME_LAMBDA_BOUND!r} would leave "
                    "those rows at a zero the scheme never repairs"
                ),
                remedy="downscale from a parent run with the child's physics.component",
                option="missing_number_concentrations", value=missing_numbers,
            )
    rows: list[dict[str, Any]] = []
    for child_row, name in enumerate(child.names):
        if name in parent_kind:
            rows.append({"child_row": child_row, "name": name, "source": "parent",
                         "parent_row": parent.names.index(name)})
        else:
            rows.append({"child_row": child_row, "name": name,
                         "source": "zero-then-" + SCHEME_LAMBDA_BOUND})
    record: dict[str, Any] = {
        "schema": NEST_DOWNSCALE_SPECIES_SCHEMA,
        "parent_component": parent.component_id,
        "child_component": child.component_id,
        "matched_by": "CM1 scalar name",
        "rows": rows,
        "parent_species_not_transferred": [n for n in parent.names if n not in child_kind],
    }
    if absent:
        record["missing_number_concentrations"] = {
            "policy": missing_numbers, **(authority or {}),
        }
    return record


__all__ = [
    "LATTICES",
    "NEST_DOWNSCALE_SCHEMA",
    "NEST_DOWNSCALE_SPECIES_SCHEMA",
    "SCHEME_LAMBDA_BOUND",
    "SCHEME_LAMBDA_BOUND_AUTHORITY",
    "plan_species_transfer",
    "ONE_WAY_LIMIT",
    "assert_matches_reference",
    "downscale_level",
    "downscale_level_staggered",
    "downscale_record",
    "downscale_volume",
    "downscale_volume_staggered",
    "limited_slopes_array",
    "require_identical_vertical_decks",
]
