"""Host-side geometry and clock for one refined child grid inside a parent.

WHAT CM1 r21.1 ACTUALLY HAS, STATED PLAINLY
-------------------------------------------
CM1 r21.1 has **no grid nesting**.  The release carries no ``nest`` namelist
word (``run/namelist.input`` has none), no nesting section in ``README`` or
``README.namelist``, no ``CHANGES`` entry, and no refinement, feedback or
parent/child machinery anywhere in ``src``.  The only occurrences of the
string in the whole tree are ``get_m_mix_nested`` inside the Thompson radar
reflectivity module (``src/module_mp_radar.F:362``), commented-out WRF
remnants in the imported GFDL surface layer (``src/module_sf_gfdl.F:1262``),
and two comments in the imported radiation driver
(``src/radtrns3d.F:62,318``).  None of them is a grid nest.

CM1 has exactly one moving-mesh capability, and it is not a nest: the
adaptive **domain** move of ``src/azimavg.F:4376-4528``
(``update_adapt_move``), which translates the *whole single domain* after a
tropical cyclone by changing ``umove``/``vmove`` -- the same translating
reference frame GM1 already carries, whose ground position is the
``domainlocx``/``domainlocy`` accumulator of ``src/cm1.F:3512-3513``.  Its
tracker is ``getpsmth``/``getpcent`` (``src/azimavg.F:4012-4330``).

Grid nesting is therefore a **declared extension beyond CM1 r21.1**, and
``records/CM1-NEST-SEAM.md`` says so in those words.  Everything this module
takes from CM1 is cited to CM1; everything that is not in CM1 is named as
this port's own and justified on its own terms.

WHAT IS TAKEN FROM CM1, BY LINE
-------------------------------
* the acoustic small step and its substep count:
  ``src/misclibs.F:857-868`` (``dtsm``, 0.60 times the smaller horizontal
  spacing over 360 m/s, for the vertically implicit ``psolver=3``) and
  ``src/misclibs.F:883-885`` (``nsound``, the nearest integer to
  ``dbldt/dtsm`` and at least 4, rounded up to even, incremented once
  more if the small step still exceeds ``dtsm``), with
  ``src/misclibs.F:835``'s ``max_nsound = 192``.  GM1 already ports all of
  this in :mod:`cm1core.gm1_runtime`; the child reuses that port unchanged,
  which is the whole point -- the child is a CM1 grid, so CM1's own rule
  decides its acoustic cadence.
* the advective Courant norm and stop: ``src/misclibs.F:2747-2765`` forms
  ``dt*sqrt((u/dx)^2+(v/dy)^2+(w/dz)^2)`` and ``src/misclibs.F:2714`` stops at
  1.50.  Refining space without refining time multiplies that norm by the
  space/time ratio, which is reported here by value.
* the storm tracker: ``src/azimavg.F:4071-4107`` (100 passes of the 9-point
  ``0.11111111`` box smoother on the lowest level of ``pp3d``, in CM1's own
  summation order), ``src/azimavg.F:4122-4139`` (corner interpolation to zeta
  points and the minimum), ``src/azimavg.F:4159-4163`` (the centre is the
  grid **face** ``xfref(ipmin)``/``yfref(jpmin)``, not a cell centre).
* the re-centring cadence and intent: ``src/azimavg.F:4421-4435``
  (``adapt_move_frq``; "attempt to move toward center of domain by next
  adaptmove time").

WHAT IS THIS PORT'S OWN
-----------------------
* placing a refined grid inside the parent at all;
* quantising the nest's position to whole parent cells (CM1's own tracker
  already snaps its centre to a grid face, ``azimavg.F:4159-4163``, so
  quantisation is in CM1's spirit, but nothing in CM1 moves a sub-domain);
* the conservative odd-ratio block relationship between parent and child
  cells proved by :meth:`NestGeometry.restriction_offsets_m`.

GEOMETRY CONTRACT
-----------------
The child covers a whole number of parent cells.  ``parent_i0``/``parent_j0``
are 0-based parent scalar cell indices of the child's south-west corner cell,
and the child spans ``parent_span_x``/``parent_span_y`` parent cells, so the
child's outer boundary lies exactly on parent cell **faces**.  That is a
conservation requirement, not a convenience: CM1 is an Arakawa C grid
(``src/cm1.F:744-777`` establishes the staggered allocations), the mass flux
into the child crosses the child's outermost ``u``/``v`` face column, and
those columns coincide with parent ``u``/``v`` face columns only when the
child boundary is face-aligned.  A centre-aligned child would import a flux
through a face the parent does not store, so the mass the child gains would
have no parent face to have left through.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

from .gm1_runtime import (
    CM1_COURANT_STOP,
    CM1_MAX_ACOUSTIC_SUBSTEPS,
    acoustic_small_step_defect,
    cm1_acoustic_small_step_limit_s,
    cm1_diagnostic_acoustic_substeps,
    positive_normal_f32_bits,
)

#: CM1 r21.1 has no nesting.  This string is the one sentence every receipt,
#: plan and record repeats, so the claim can never drift.
CM1_NESTING_ABSENCE = (
    "CM1 r21.1 contains no grid nesting: no nest namelist word in "
    "run/namelist.input, no nesting section in README or README.namelist, no "
    "CHANGES entry, and no refinement, parent/child or feedback code in src. "
    "Its only moving-mesh capability is the whole-domain adaptive move of "
    "src/azimavg.F:4376-4528, which translates the single domain rather than "
    "refining part of it. Grid nesting in GM1 is a declared extension beyond "
    "CM1 r21.1."
)

NEST_GEOMETRY_SCHEMA = "gm1.cartesian-nest-geometry/v1"

#: The fifth-order horizontal stencil CM1 establishes at ``cm1.F:744-777``
#: reads three cells either side.  A child boundary closer than this to the
#: parent's own boundary would build its lateral forcing from parent ghost
#: words that the parent's open radiative boundary
#: (``src/bc.F:528-934``) writes rather than from parent interior air.
PARENT_STENCIL_HALO = 3

#: One-way (``0``) and two-way (``1``) feedback, by value.
FEEDBACK_ONE_WAY = 0
FEEDBACK_TWO_WAY = 1
ADMITTED_FEEDBACK = (FEEDBACK_ONE_WAY, FEEDBACK_TWO_WAY)


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class NestGeometry:
    """One refined child grid placed on whole parent cells.

    Every length here is derived from the parent's own spacing and the integer
    refinement ratio, so the child grid is exact rather than re-declared: a
    separately declared child ``dx`` could disagree with ``parent_dx/ratio`` by
    a rounding and put the child's cells and the parent's faces permanently out
    of registration.
    """

    parent_nx: int
    parent_ny: int
    parent_dx_m: float
    parent_dy_m: float
    refinement_ratio: int
    time_refinement_ratio: int
    parent_i0: int
    parent_j0: int
    parent_span_x: int
    parent_span_y: int
    spec_zone: int
    relax_zone: int
    feedback: int

    def __post_init__(self) -> None:
        for name in ("parent_nx", "parent_ny"):
            _integer(name, getattr(self, name), minimum=2 * PARENT_STENCIL_HALO + 1)
        for name in ("parent_dx_m", "parent_dy_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        ratio = _integer("refinement_ratio", self.refinement_ratio, minimum=3)
        if ratio % 2 == 0:
            raise ValueError(
                f"refinement_ratio={ratio} is even; the C-grid face-staggered "
                "restriction block is symmetric about a parent face only for an "
                "odd ratio, so two-way feedback would write a value displaced "
                "by half a child cell into every parent u and v face"
            )
        _integer("time_refinement_ratio", self.time_refinement_ratio, minimum=1)
        for name in ("parent_i0", "parent_j0"):
            _integer(name, getattr(self, name), minimum=0)
        for name in ("parent_span_x", "parent_span_y"):
            _integer(name, getattr(self, name), minimum=1)
        _integer("spec_zone", self.spec_zone, minimum=1)
        _integer("relax_zone", self.relax_zone, minimum=0)
        if self.relax_zone and self.relax_zone <= self.spec_zone:
            raise ValueError(
                "relax_zone must exceed spec_zone; the relaxation weight is "
                "zero at the first relaxed row and a zone no wider than the "
                "specified zone applies no relaxation at all"
            )
        if self.feedback not in ADMITTED_FEEDBACK:
            raise ValueError(
                f"feedback={self.feedback!r} names neither one-way (0) nor "
                "two-way (1)"
            )
        # The child's own lateral zones must fit inside it with an interior
        # left over.  A child that is all boundary integrates nothing of its
        # own and merely re-interpolates the parent at finer spacing.
        for axis, count in (("x", self.nx), ("y", self.ny)):
            if count <= 2 * self.relax_zone:
                raise ValueError(
                    f"the child has {count} cells in {axis} and its two "
                    f"{self.relax_zone}-cell relaxation zones would meet, so "
                    "the refined grid would carry no freely evolving interior"
                )

    # -- derived grid -----------------------------------------------------
    @property
    def nx(self) -> int:
        return self.parent_span_x * self.refinement_ratio

    @property
    def ny(self) -> int:
        return self.parent_span_y * self.refinement_ratio

    @property
    def dx_m(self) -> float:
        return self.parent_dx_m / self.refinement_ratio

    @property
    def dy_m(self) -> float:
        return self.parent_dy_m / self.refinement_ratio

    @property
    def origin_x_m(self) -> float:
        """Child west boundary, on the parent ``u`` face at ``parent_i0``."""

        return self.parent_i0 * self.parent_dx_m

    @property
    def origin_y_m(self) -> float:
        return self.parent_j0 * self.parent_dy_m

    @property
    def extent_x_m(self) -> float:
        return self.parent_span_x * self.parent_dx_m

    @property
    def extent_y_m(self) -> float:
        return self.parent_span_y * self.parent_dy_m

    @property
    def centre_x_m(self) -> float:
        return self.origin_x_m + 0.5 * self.extent_x_m

    @property
    def centre_y_m(self) -> float:
        return self.origin_y_m + 0.5 * self.extent_y_m

    @property
    def cell_ratio(self) -> int:
        """Child cells per parent cell, horizontally."""

        return self.refinement_ratio * self.refinement_ratio

    # -- placement --------------------------------------------------------
    def placement_defect(self) -> str | None:
        """Name what a badly placed child would compute, or ``None``.

        The child must keep ``PARENT_STENCIL_HALO`` parent cells clear of the
        parent's own lateral boundary on every side.
        """

        low = PARENT_STENCIL_HALO
        for axis, i0, span, parent_n in (
            ("x", self.parent_i0, self.parent_span_x, self.parent_nx),
            ("y", self.parent_j0, self.parent_span_y, self.parent_ny),
        ):
            if i0 < low or i0 + span > parent_n - low:
                return (
                    f"the child spans parent {axis} cells {i0}..{i0 + span - 1} "
                    f"of 0..{parent_n - 1}, which leaves fewer than "
                    f"{PARENT_STENCIL_HALO} parent cells between the child "
                    "boundary and the parent's own; the parent air the child's "
                    "lateral forcing would sample is then written by CM1's open "
                    "radiative lateral boundary (src/bc.F:528-934) rather than "
                    "integrated, so the child would be driven by the parent's "
                    "boundary treatment instead of by the parent's storm"
                )
        return None

    # -- donor maps -------------------------------------------------------
    def donor_index_x(self) -> tuple[int, ...]:
        """Parent scalar cell index containing each child scalar column."""

        r = self.refinement_ratio
        return tuple(self.parent_i0 + ic // r for ic in range(self.nx))

    def donor_index_y(self) -> tuple[int, ...]:
        r = self.refinement_ratio
        return tuple(self.parent_j0 + jc // r for jc in range(self.ny))

    def sub_index_x(self) -> tuple[int, ...]:
        """Position of each child column inside its parent cell, ``0..r-1``."""

        return tuple(ic % self.refinement_ratio for ic in range(self.nx))

    def sub_index_y(self) -> tuple[int, ...]:
        return tuple(jc % self.refinement_ratio for jc in range(self.ny))

    def sub_cell_offsets(self) -> tuple[float, ...]:
        """Child scalar-centre offsets from the parent cell centre, in cells.

        For sub-index ``m`` the offset is ``(2m + 1 - r)/(2r)`` parent cells,
        which is algebraically ``(m + 0.5)/r - 0.5`` but is written this way on
        purpose: the numerator ``2m + 1 - r`` is an exact integer and its
        negation is the numerator of entry ``r-1-m``, so dividing both by the
        same ``2r`` makes ``offset[r-1-m]`` the **exact** binary negation of
        ``offset[m]``.  The obvious spelling does not have that property --
        for ``r = 3`` it gives ``-0.33333333333333337`` and
        ``+0.33333333333333326``, which do not cancel -- and the exact
        cancellation is what makes the block-mean identity of
        ``nest_boundary.block_mean_residual`` hold to the last bit rather than
        to a tolerance.

        For an **odd** ratio the middle entry is exactly zero, so one child
        column is coincident with the parent column.
        """

        r = self.refinement_ratio
        return tuple((2 * m + 1 - r) / (2 * r) for m in range(r))

    def restriction_offsets_m(self, *, stagger: str) -> tuple[float, ...]:
        """Where the ``r`` child points of a feedback block sit, in metres.

        ``stagger="mass"`` is the cell-centred block: offsets from the parent
        cell centre.  ``stagger="face"`` is the block the two-way feedback must
        average to produce one parent ``u`` (or ``v``) face value: the ``r``
        child faces nearest that parent face.  For an odd ratio those are at
        ``-(r-1)/2 .. +(r-1)/2`` child cells and their mean offset is exactly
        zero; for an even ratio the mean offset is half a child cell, which is
        the displacement the constructor refuses.
        """

        r = self.refinement_ratio
        d = self.dx_m
        if stagger == "mass":
            return tuple(((m + 0.5) / r - 0.5) * self.parent_dx_m for m in range(r))
        if stagger == "face":
            half = (r - 1) // 2
            if r % 2:
                return tuple((m - half) * d for m in range(r))
            return tuple((m - r // 2) * d for m in range(r))
        raise ValueError("stagger must be 'mass' or 'face'")

    def feedback_face_displacement_m(self) -> float:
        """Mean offset of the face restriction block from the parent face."""

        offsets = self.restriction_offsets_m(stagger="face")
        return math.fsum(offsets) / len(offsets)

    # -- clock ------------------------------------------------------------
    def child_dt_s(self, parent_dt_s: float) -> float:
        return parent_dt_s / self.time_refinement_ratio

    def cm1_child_acoustic_substeps(self, parent_dt_s: float) -> int:
        """``misclibs.F:883-885`` evaluated on the child's step and spacing."""

        return cm1_diagnostic_acoustic_substeps(
            dt_s=self.child_dt_s(parent_dt_s), dx_m=self.dx_m, dy_m=self.dy_m
        )

    def child_acoustic_small_step_limit_s(self) -> float:
        return cm1_acoustic_small_step_limit_s(dx_m=self.dx_m, dy_m=self.dy_m)

    def advective_courant_multiplier(self) -> float:
        """How much larger the child's ``misclibs.F:2747-2765`` norm is.

        The norm is ``dt*sqrt((u/dx)^2 + (v/dy)^2 + (w/dz)^2)``.  Refining the
        horizontal spacing by ``r`` and the step by ``q`` multiplies the
        horizontal part of the same wind's norm by ``r/q``; CM1 stops the run
        at 1.50 (``misclibs.F:2714``).
        """

        return self.refinement_ratio / self.time_refinement_ratio

    def clock_defect(self, parent_dt_s: float) -> str | None:
        """Name what this parent step would do to the child, or ``None``."""

        child_dt = self.child_dt_s(parent_dt_s)
        if positive_normal_f32_bits(child_dt) is None:
            return (
                f"the parent step {parent_dt_s!r} s divided by "
                f"time_refinement_ratio={self.time_refinement_ratio} is "
                f"{child_dt!r} s, which is not a positive normal CM1 default-REAL "
                "step; solve2.F:145 applies dt as one binary32 word, so the "
                "child's kernels would advance by zero or by a subnormal while "
                "its clock advanced by the requested value"
            )
        needed = self.cm1_child_acoustic_substeps(parent_dt_s)
        if needed > CM1_MAX_ACOUSTIC_SUBSTEPS:
            return (
                f"CM1's own dtsmall (misclibs.F:883-885) derives nsound={needed} "
                f"for the child's {self.dx_m!r} m spacing at {child_dt!r} s, "
                f"above CM1's max_nsound={CM1_MAX_ACOUSTIC_SUBSTEPS} "
                "(misclibs.F:835), where misclibs.F:887-911 stops the model "
                "rather than run that many small steps; raise "
                "time_refinement_ratio so the child's step shrinks with its "
                "spacing"
            )
        return None

    def record(self, *, parent_dt_s: float) -> dict[str, object]:
        """Everything a reader needs to place this nest and price its clock."""

        child_dt = self.child_dt_s(parent_dt_s)
        return {
            "schema": NEST_GEOMETRY_SCHEMA,
            "cm1_authority": CM1_NESTING_ABSENCE,
            "refinement_ratio": self.refinement_ratio,
            "time_refinement_ratio": self.time_refinement_ratio,
            "feedback": self.feedback,
            "feedback_name": "two-way" if self.feedback else "one-way",
            "parent_i0": self.parent_i0,
            "parent_j0": self.parent_j0,
            "parent_span_x": self.parent_span_x,
            "parent_span_y": self.parent_span_y,
            "nx": self.nx,
            "ny": self.ny,
            "dx_m": self.dx_m,
            "dy_m": self.dy_m,
            "origin_x_m": self.origin_x_m,
            "origin_y_m": self.origin_y_m,
            "extent_x_m": self.extent_x_m,
            "extent_y_m": self.extent_y_m,
            "centre_x_m": self.centre_x_m,
            "centre_y_m": self.centre_y_m,
            "spec_zone": self.spec_zone,
            "relax_zone": self.relax_zone,
            "child_dt_s": child_dt,
            "child_acoustic_substeps_cm1_would_choose": (
                self.cm1_child_acoustic_substeps(parent_dt_s)
            ),
            "child_acoustic_small_step_limit_s": (
                self.child_acoustic_small_step_limit_s()
            ),
            "child_steps_per_parent_step": self.time_refinement_ratio,
            "advective_courant_multiplier": self.advective_courant_multiplier(),
            "advective_courant_stop": CM1_COURANT_STOP,
            "feedback_face_displacement_m": self.feedback_face_displacement_m(),
            "boundary_face_aligned": True,
        }


def child_acoustic_defect(
    geometry: NestGeometry, *, parent_dt_s: float, acoustic_substeps: int
) -> str | None:
    """CM1's own ``dtsmall`` verdict on the child's declared substep count."""

    return acoustic_small_step_defect(
        dt_s=geometry.child_dt_s(parent_dt_s),
        dx_m=geometry.dx_m,
        dy_m=geometry.dy_m,
        acoustic_substeps=acoustic_substeps,
    )


# ---------------------------------------------------------------------------
# CM1's own storm tracker, transcribed, and the nest motion built on it.
# ---------------------------------------------------------------------------

#: ``src/azimavg.F:4077`` -- "by default, run del^2 diffusion 100 times".
CM1_PSMTH_PASSES = 100
#: ``src/azimavg.F:4092`` -- the literal CM1 writes, not ``1.0/9.0``.
CM1_PSMTH_WEIGHT = 0.11111111


def cm1_smoothed_pressure(
    field: Sequence[Sequence[float]], *, passes: int = CM1_PSMTH_PASSES
) -> list[list[float]]:
    """``getpsmth``'s smoother, in CM1's own summation order.

    ``src/azimavg.F:4092-4096``: with ``C`` the point, ``E``/``W``/``N``/``S``
    its edge neighbours and ``NE``/``NW``/``SE``/``SW`` its corners, the
    smoothed value is ``0.11111111 * (C + (((E + W) + (N + S)) + ((NE + NW) +
    (SE + SW))))``.

    The parenthesisation is CM1's and is preserved: it fixes the float
    summation order, and a reassociated smoother reaches a different minimum
    on a field with two nearly equal pressure minima, which is exactly the
    field a splitting supercell presents.

    The halo is filled by zero-gradient extension, which is what ``bc2d``
    (``src/bc.F:12-172``) does for CM1's open lateral boundaries.
    """

    ny = len(field)
    nx = len(field[0])
    current = [[float(v) for v in row] for row in field]
    for _ in range(passes):
        def at(j: int, i: int) -> float:
            return current[min(max(j, 0), ny - 1)][min(max(i, 0), nx - 1)]

        current = [
            [
                CM1_PSMTH_WEIGHT
                * (
                    at(j, i)
                    + (
                        ((at(j, i + 1) + at(j, i - 1)) + (at(j + 1, i) + at(j - 1, i)))
                        + (
                            (at(j + 1, i + 1) + at(j + 1, i - 1))
                            + (at(j - 1, i + 1) + at(j - 1, i - 1))
                        )
                    )
                )
                for i in range(nx)
            ]
            for j in range(ny)
        ]
    return current


def cm1_pressure_centre_cell(
    field: Sequence[Sequence[float]], *, passes: int = CM1_PSMTH_PASSES
) -> tuple[int, int]:
    """``getpsmth``'s centre, as the ``(i, j)`` grid **face** CM1 reports.

    ``src/azimavg.F:4122-4139`` interpolates the smoothed field to zeta points
    -- ``0.25*((d(i,j)+d(i-1,j-1)) + (d(i-1,j)+d(i,j-1)))``, CM1's own
    association -- takes the minimum, and ``src/azimavg.F:4159-4163`` reports
    the corresponding ``xfref``/``yfref``, i.e. the cell **face** index.  The
    returned pair is that face index in 0-based child-of-nothing parent cells:
    face ``i`` is the west face of parent cell ``i``.
    """

    smoothed = cm1_smoothed_pressure(field, passes=passes)
    ny = len(smoothed)
    nx = len(smoothed[0])

    def at(j: int, i: int) -> float:
        return smoothed[min(max(j, 0), ny - 1)][min(max(i, 0), nx - 1)]

    best = math.inf
    best_ij = (0, 0)
    for j in range(ny):
        for i in range(nx):
            zeta = 0.25 * (
                (at(j, i) + at(j - 1, i - 1)) + (at(j - 1, i) + at(j, i - 1))
            )
            if zeta < best:
                best = zeta
                best_ij = (i, j)
    return best_ij


#: The device form of ``getpsmth``'s smoother.  The tracker runs over the whole
#: parent's lowest level every ``follow_interval_s``, one hundred times; that is
#: a field operation and belongs on the device, not in the driver.
NEST_PSMTH_KERNEL_NAME = "gm1_nest_psmth_pass"
NEST_PSMTH_KERNEL_SOURCE = r'''// gm1.cartesian-nest-storm-tracker/v1
// CM1 r21.1 src/azimavg.F:4092-4096 (getpsmth's del^2 smoother) and
// src/azimavg.F:4122-4125 (interpolation to zeta points), transliterated.
// CM1's parenthesisation is preserved exactly: it fixes the float summation
// order, and a reassociated smoother reaches a different minimum on a field
// with two nearly equal pressure minima -- which is exactly the field a
// splitting supercell presents.  The halo is filled by zero-gradient
// extension, which is what src/bc.F:12-172 (bc2d) does for an open lateral
// boundary.

__device__ __forceinline__ int gm1_nest_clamp(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// azimavg.F:4092 -- the literal CM1 writes is 0.11111111, not 1.0/9.0.
#define GM1_NEST_PSMTH_WEIGHT 0.11111111f

extern "C" __global__ void gm1_nest_psmth_pass(
    const float* __restrict__ in, float* __restrict__ out, int ny, int nx)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned long long)ny * nx) return;
    int j = (int)(tid / nx);
    int i = (int)(tid - (unsigned long long)j * nx);
    int im = gm1_nest_clamp(i - 1, 0, nx - 1), ip = gm1_nest_clamp(i + 1, 0, nx - 1);
    int jm = gm1_nest_clamp(j - 1, 0, ny - 1), jp = gm1_nest_clamp(j + 1, 0, ny - 1);
    float c  = in[(unsigned long long)j * nx + i];
    float e  = in[(unsigned long long)j * nx + ip];
    float w  = in[(unsigned long long)j * nx + im];
    float n  = in[(unsigned long long)jp * nx + i];
    float s  = in[(unsigned long long)jm * nx + i];
    float ne = in[(unsigned long long)jp * nx + ip];
    float nw = in[(unsigned long long)jp * nx + im];
    float se = in[(unsigned long long)jm * nx + ip];
    float sw = in[(unsigned long long)jm * nx + im];
    // 0.11111111*( c + ( ((e+w)+(n+s)) + ((ne+nw)+(se+sw)) ) )
    float cross = __fadd_rn(__fadd_rn(e, w), __fadd_rn(n, s));
    float diag  = __fadd_rn(__fadd_rn(ne, nw), __fadd_rn(se, sw));
    out[tid] = __fmul_rn(GM1_NEST_PSMTH_WEIGHT,
                         __fadd_rn(c, __fadd_rn(cross, diag)));
}

// azimavg.F:4124-4125:
//   dum1(i,j,2) = 0.25*( (d(i,j)+d(i-1,j-1)) + (d(i-1,j)+d(i,j-1)) )
// The result is indexed by the zeta (corner/face) point, which is what
// azimavg.F:4159-4163 reports as xfref(ipmin)/yfref(jpmin).
extern "C" __global__ void gm1_nest_zeta_point(
    const float* __restrict__ in, float* __restrict__ out, int ny, int nx)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned long long)ny * nx) return;
    int j = (int)(tid / nx);
    int i = (int)(tid - (unsigned long long)j * nx);
    int im = gm1_nest_clamp(i - 1, 0, nx - 1);
    int jm = gm1_nest_clamp(j - 1, 0, ny - 1);
    float a = in[(unsigned long long)j * nx + i];
    float b = in[(unsigned long long)jm * nx + im];
    float c = in[(unsigned long long)jm * nx + i];
    float d = in[(unsigned long long)j * nx + im];
    out[tid] = __fmul_rn(0.25f, __fadd_rn(__fadd_rn(a, b), __fadd_rn(d, c)));
}
'''
NEST_PSMTH_KERNEL_SHA256 = hashlib.sha256(
    NEST_PSMTH_KERNEL_SOURCE.encode("utf-8")
).hexdigest()
NEST_PSMTH_THREADS = 256
NEST_PSMTH_NVRTC_OPTIONS = ("--std=c++17", "-fmad=false")


def compile_nest_tracker_module(cupy):
    """Compile the packaged tracker kernels for the current device."""

    from .nvrtc_module import compile_nvrtc_module

    return compile_nvrtc_module(
        cupy,
        code=NEST_PSMTH_KERNEL_SOURCE,
        options=NEST_PSMTH_NVRTC_OPTIONS,
        filename="gm1_nest_tracker.cu",
    )


def device_pressure_centre_cell(
    cupy, surface_pressure, *, passes: int = CM1_PSMTH_PASSES, module=None
) -> tuple[int, int]:
    """CM1's tracker on the device: smooth, corner-average, take the minimum.

    ``surface_pressure`` is the parent's lowest level of ``pp3d`` as one
    C-contiguous float32 ``(ny, nx)`` array.  Returns the same ``(i, j)`` grid
    **face** pair :func:`cm1_pressure_centre_cell` returns, so the host path
    and the device path are interchangeable and comparable.

    Nothing here brings a field to the host: the hundred smoother passes, the
    corner interpolation and the minimum all run on the device, and only the
    two integers the nest is moved by cross back.
    """

    if surface_pressure.dtype != cupy.float32:
        raise TypeError(
            "the tracker operand must be float32; CM1 carries pp3d as its "
            "default REAL and a float64 array would be read as twice as many "
            "float32 words"
        )
    if surface_pressure.ndim != 2:
        raise ValueError("the tracker operand must be a single (ny, nx) level")
    if not surface_pressure.flags.c_contiguous:
        raise ValueError("the tracker operand must be C-contiguous")
    if type(passes) is not int or passes < 1:
        raise ValueError("passes must be a positive integer")
    ny, nx = (int(value) for value in surface_pressure.shape)
    if module is None:
        module = compile_nest_tracker_module(cupy)
    smoother = module.get_function(NEST_PSMTH_KERNEL_NAME)
    zeta = module.get_function("gm1_nest_zeta_point")
    count = ny * nx
    blocks = (count + NEST_PSMTH_THREADS - 1) // NEST_PSMTH_THREADS
    current = surface_pressure.copy()
    scratch = cupy.empty_like(current)
    for _ in range(passes):
        smoother(
            (blocks,), (NEST_PSMTH_THREADS,),
            (current, scratch, cupy.int32(ny), cupy.int32(nx)),
        )
        current, scratch = scratch, current
    zeta(
        (blocks,), (NEST_PSMTH_THREADS,),
        (current, scratch, cupy.int32(ny), cupy.int32(nx)),
    )
    flat = int(cupy.argmin(scratch.reshape(-1)))
    return flat % nx, flat // nx


@dataclass(frozen=True, slots=True)
class NestMove:
    """One re-centring decision, in whole parent cells."""

    shift_i: int
    shift_j: int
    requested_shift_i: int
    requested_shift_j: int
    clamped: bool
    reason: str

    def record(self) -> dict[str, object]:
        return {
            "shift_i": self.shift_i,
            "shift_j": self.shift_j,
            "requested_shift_i": self.requested_shift_i,
            "requested_shift_j": self.requested_shift_j,
            "clamped": self.clamped,
            "reason": self.reason,
        }


def plan_nest_move(
    geometry: NestGeometry,
    *,
    centre_i_face: int,
    centre_j_face: int,
    max_cells_per_move: int,
) -> NestMove:
    """Move the nest toward a tracked centre, in whole parent cells.

    ``centre_i_face``/``centre_j_face`` are what ``cm1_pressure_centre_cell``
    returns: the parent grid **face** CM1's own tracker reports
    (``src/azimavg.F:4159-4163``).  Reporting the centre as a face rather than
    a position is CM1's -- ``getpcent``'s centroid branch does the same, by
    searching ``xfref``/``yfref`` for the nearest face at
    ``src/azimavg.F:4301-4315`` -- so the nest is quantised in CM1's own
    currency.

    The nest's own centre face is ``parent_i0 + parent_span_x/2``, which is a
    face exactly when the span is even; an odd span puts the nest centre at a
    parent cell centre, and the resulting half-cell requested shift is rounded
    with Fortran ``nint`` semantics.  That rounding is this port's own: CM1
    snaps by a nearest-face search over the whole axis, which gives the same
    answer on a uniform grid and is not the same code.

    The motion is quantised to whole parent cells for the reason the module
    docstring gives: a fractional shift would take the child's outer ``u``/``v``
    face columns off the parent's face columns and the interface flux would no
    longer be the parent's own.

    The shift is clamped so the child never leaves the parent's interior; the
    clamp is reported rather than hidden, because a nest that has silently
    stopped following the storm is a nest that is refining clear air.
    """

    _integer("max_cells_per_move", max_cells_per_move, minimum=1)
    nest_centre_i = geometry.parent_i0 + geometry.parent_span_x / 2.0
    nest_centre_j = geometry.parent_j0 + geometry.parent_span_y / 2.0
    requested_i = _nint(centre_i_face - nest_centre_i)
    requested_j = _nint(centre_j_face - nest_centre_j)
    shift_i = max(-max_cells_per_move, min(max_cells_per_move, requested_i))
    shift_j = max(-max_cells_per_move, min(max_cells_per_move, requested_j))
    rate_clamped = (shift_i, shift_j) != (requested_i, requested_j)

    low = PARENT_STENCIL_HALO
    lo_i = low - geometry.parent_i0
    hi_i = (geometry.parent_nx - low - geometry.parent_span_x) - geometry.parent_i0
    lo_j = low - geometry.parent_j0
    hi_j = (geometry.parent_ny - low - geometry.parent_span_y) - geometry.parent_j0
    bounded_i = max(lo_i, min(hi_i, shift_i))
    bounded_j = max(lo_j, min(hi_j, shift_j))
    edge_clamped = (bounded_i, bounded_j) != (shift_i, shift_j)

    if edge_clamped:
        reason = (
            "clamped at the parent interior: the nest cannot move further "
            f"without leaving fewer than {PARENT_STENCIL_HALO} parent cells "
            "between its boundary and the parent's own"
        )
    elif rate_clamped:
        reason = (
            f"clamped at max_cells_per_move={max_cells_per_move}; the tracked "
            "centre is moving faster than the nest is permitted to follow"
        )
    else:
        reason = "tracked centre reached"
    return NestMove(
        shift_i=bounded_i,
        shift_j=bounded_j,
        requested_shift_i=requested_i,
        requested_shift_j=requested_j,
        clamped=edge_clamped or rate_clamped,
        reason=reason,
    )


def follow_target_face(
    tracker: str,
    *,
    centre_i_face: int,
    centre_j_face: int,
    placement_offset_faces: tuple[float, float],
) -> tuple[float, float]:
    """Where a following nest's centre is moved toward, in parent faces.

    ``centre_*_face`` is CM1's own pressure centre
    (:func:`cm1_pressure_centre_cell`, ``src/azimavg.F:4071-4163``).
    ``placement_offset_faces`` is the nest centre minus that centre when the
    nest was placed.  ``cm1-pressure-held-offset`` returns the pressure centre
    plus that offset, so a window placed over the surface vortex -- 9 to 11 km
    from the mesocyclone's steady pressure centre on a mature supercell -- is
    carried with the storm instead of being pulled onto the pressure centre
    aloft; ``cm1-pressure-centre`` returns the pressure centre itself, CM1's
    own adaptive-move target.
    """

    if tracker == "cm1-pressure-held-offset":
        return (
            centre_i_face + float(placement_offset_faces[0]),
            centre_j_face + float(placement_offset_faces[1]),
        )
    if tracker == "cm1-pressure-centre":
        return float(centre_i_face), float(centre_j_face)
    raise ValueError(f"unknown nest follow tracker {tracker!r}")


def move_nest(geometry: NestGeometry, move: NestMove) -> NestGeometry:
    """The same nest, translated by one admitted whole-parent-cell move."""

    from dataclasses import replace

    return replace(
        geometry,
        parent_i0=geometry.parent_i0 + move.shift_i,
        parent_j0=geometry.parent_j0 + move.shift_j,
    )


def _nint(value: float) -> int:
    """Fortran ``nint``: round half away from zero."""

    return int(math.floor(value + 0.5)) if value >= 0 else -int(math.floor(-value + 0.5))


__all__ = [
    "ADMITTED_FEEDBACK",
    "CM1_NESTING_ABSENCE",
    "CM1_PSMTH_PASSES",
    "CM1_PSMTH_WEIGHT",
    "NEST_PSMTH_KERNEL_NAME",
    "NEST_PSMTH_KERNEL_SHA256",
    "NEST_PSMTH_KERNEL_SOURCE",
    "NEST_PSMTH_NVRTC_OPTIONS",
    "NEST_PSMTH_THREADS",
    "FEEDBACK_ONE_WAY",
    "FEEDBACK_TWO_WAY",
    "NEST_GEOMETRY_SCHEMA",
    "PARENT_STENCIL_HALO",
    "NestGeometry",
    "NestMove",
    "child_acoustic_defect",
    "cm1_pressure_centre_cell",
    "compile_nest_tracker_module",
    "device_pressure_centre_cell",
    "cm1_smoothed_pressure",
    "follow_target_face",
    "move_nest",
    "plan_nest_move",
]
