"""Vertical refinement for the live nest and the one-way downscale child.

A child may carry a finer vertical deck than its parent -- a 5 m lowest layer
under a 20 m one -- over the same model top.  CM1 r21.1 has no nesting
(``records/CM1-NEST-SEAM.md`` section 1), so what follows is a declared GM1
extension assembled from two authorities, each used for what it actually
decides.

WHAT CM1 DECIDES: THE VERTICAL INTERPOLATION RULE
-------------------------------------------------
CM1's only operator that moves a state between two vertical decks is
``interp_on_restart`` (``src/restart_read.F``):

* ``:564-575`` scalar/u/v levels: ``kold`` advances ``while sigma0(kold+1) <
  sigma(k)``; ``rz = (sigma(k)-sigma0(kold))/(sigma0(kold+1)-sigma0(kold))``
  as a default-REAL expression, over the OLD deck's ghosted ``sigma0(0:nk+1)``
  (``:450``).  An exact match stops the search with ``rz = 1``.
* ``:600-609`` the same over ``sigmaf0(0:nk+2)`` for ``w``.
* ``:3037-3039`` and ``prep_for_interp`` ``:3302-3406``: the two data planes
  are read sequentially; the first level read is duplicated into the lower
  plane (``kold1 == 0``) and, once no old level is left, the lower plane is
  set equal to the upper one -- constant extension, with the ghost-derived
  ``rz`` still applied to the two identical planes.
* ``do_interp`` ``:3414-3470``: eight REAL weight expressions (``:3437-3444``)
  assigned to DOUBLE PRECISION, summed in written order (``:3458-3465``) and
  narrowed on assignment.
* ``:1207``: ``tkea`` is not read when interpolating to a new grid.

Here the rule is evaluated at a new column that coincides with an old column
(``rx = ry = 0``): the horizontal transfer is the nest's own operator and is
applied separately.  :func:`build_cm1_vertical_map` is a literal
transliteration of the search and of the plane state machine, and the kernel
evaluates the full eight-term statement, including its six zero-weight terms.

WHAT WRF DECIDES: WHERE THE VERTICAL STEP SITS
----------------------------------------------
WRF nests vertically with ``vert_refine_method`` (``Registry/Registry.EM_COMMON:
2279``).  For boundary forcing it interpolates the parent's fields vertically
on the intermediate grid -- the parent's horizontal resolution over the
nest's footprint -- and only then horizontally onto the nest
(``external/RSL_LITE/force_domain_em_part2.F:126`` includes
``nest_forcedown_interp_vert.inc``; the comment at ``:141-147`` names the
intermediate grid "interpolated in the vertical to the resolution of the
nest"); the initial condition does the same (``interp_domain_em_part2.F:132``
includes ``nest_interpdown_interp_vert.inc``).  u, v, t and moisture go
through ``vert_interp_vert_nesting`` and w, ph through
``vert_interp_vert_nesting_w`` (``share/interp_fcn.F:1910-2013``).  GM1 takes
that ORDER -- vertical on a parent-resolution window, then the unchanged
horizontal capture -- but not WRF's interpolant: WRF fits a monotone cubic in
a log-pressure altitude (``inter_wrf_copy``, ``share/interp_fcn.F:2073-2106``)
because eta levels move with the column mass; GM1's levels are fixed heights
and CM1's own linear ``sigma`` rule is the one CM1 applies between two of
its decks.

WHERE GM1 MUST CHOOSE: FEEDBACK
-------------------------------
WRF refuses feedback with vertical refinement outright
(``share/module_check_a_mundo.F:2711-2722``, "vert_refine_method=2 only works
with feedback = 0").  GM1 admits it with a conservative restriction of its
own: each parent layer ``k`` becomes the base-density- and overlap-weighted
mean of the child layers that intersect it,
``P_k = sum_m rho0_m ov(m,k) C_m / sum_m rho0_m ov(m,k)``, applied to the
horizontal block sums of the existing restriction.  Because the child layers
tile ``[0, ztop]`` and every child layer's overlaps sum to its own thickness,
``sum_k M_k P_k = sum_m rho0_m dz_m C_m`` with ``M_k = sum_m rho0_m ov(m,k)``:
the child's column mass of every conserved carrier (dry air through ``rho0``,
water through ``q``, heat through ``theta``, momentum through ``u``/``v``) is
the parent's, in the child's mass measure.  ``w`` is restricted over dual cells
``[zh(f-1), zh(f)]`` weighted by ``rf0``; its ground and top faces are the
child's own coincident faces copied (they are CM1's boundary faces).  What is
**not** claimed: that ``M_k`` equals the parent's own ``rho0 dz`` -- the two
decks sample the same base state at different heights, so the parent's
quadrature of that mass differs from the child's by the base state's own
curvature over one parent layer.

IDENTICAL DECKS
---------------
When the two decks carry the same sealed ``sha256`` there is no vertical step
at all: every caller keeps the existing copy path, word for word.  The tests
also drive the refined path through two identical decks and require the same
raw words, which is the proof that the new composition adds no arithmetic of
its own when it has nothing to do.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

NEST_VERTICAL_SCHEMA = "gm1.cartesian-nest-vertical-refinement/v1"

CM1_AUTHORITY = {
    "scalar_level_search": "src/restart_read.F:564-575 (sigma0(0:nzold+1), :450)",
    "w_level_search": "src/restart_read.F:600-609 (sigmaf0(0:nzold+2), :452)",
    "plane_state_machine": "src/restart_read.F:3037-3039, 3302-3406 prep_for_interp",
    "weights_and_sum": "src/restart_read.F:3414-3470 do_interp (weights :3437-3444, sum :3458-3465)",
    "tke_not_interpolated": "src/restart_read.F:1207",
    "ghost_levels": "src/param.F:7952-7953 (sigmaf), 8165-8173 (zf, zh)",
    "advective_courant": "src/misclibs.F:2584-2718 calccfl (stop at 1.50, :2714)",
    "psolver3_small_step": "src/misclibs.F:857-868 (0.60*min(dx,dy)/360; no dz)",
}
WRF_AUTHORITY = {
    "switch": "Registry/Registry.EM_COMMON:2279 vert_refine_method",
    "force_order": (
        "external/RSL_LITE/force_domain_em_part2.F:126 (include) and :141-147 "
        "vertical interpolation on the intermediate (parent-resolution) grid "
        "before the horizontal interpolation; inc/nest_forcedown_interp_vert.inc"
    ),
    "initial_order": (
        "external/RSL_LITE/interp_domain_em_part2.F:132; "
        "inc/nest_interpdown_interp_vert.inc"
    ),
    "carrier_split": (
        "share/interp_fcn.F:1910-1965 vert_interp_vert_nesting (u, v, t, moist) "
        "and :1969-2013 vert_interp_vert_nesting_w (w, ph)"
    ),
    "interpolant_not_taken": (
        "share/interp_fcn.F:2073-2106 inter_wrf_copy, monotone cubic in "
        "log-pressure altitude; GM1 uses CM1's linear sigma rule instead"
    ),
    "feedback_refused_by_wrf": "share/module_check_a_mundo.F:2711-2722",
    "rebalance_required_by_wrf": "share/module_check_a_mundo.F:2724-2742",
}
GM1_CHOICES = {
    "interpolant": (
        "CM1 interp_on_restart (linear in sigma, constant extension through "
        "duplicated planes) evaluated at rx = ry = 0, not WRF's cubic"
    ),
    "order": "vertical on a parent-resolution window, then the unchanged horizontal capture (WRF's order)",
    "feedback": (
        "admitted, unlike WRF: rho0-and-overlap-weighted restriction of the "
        "horizontal block sums; w over rf0-weighted dual cells with its "
        "boundary faces copied"
    ),
    "no_rebalance": (
        "no hydrostatic rebalance of the interpolated perturbations, as CM1's "
        "interp_on_restart does none; WRF requires rebalance=1 for its "
        "mass-coordinate nest"
    ),
}


def _np() -> Any:
    import numpy

    return numpy


# ---------------------------------------------------------------------------
# CM1's search and plane state machine.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Cm1VerticalMap:
    """One new level's data planes and weight, as ``interp_on_restart`` has them.

    ``lower``/``upper`` are 0-based OLD data levels held by ``datk1``/``datk2``
    when ``do_interp`` runs for that new level; ``fraction`` is CM1's REAL
    ``rz``; ``iz`` is CM1's 0-based ``kold``.
    """

    lower: Any
    upper: Any
    fraction: Any
    iz: Any
    below_support: Any
    above_support: Any

    @property
    def levels(self) -> int:
        return int(self.lower.size)


def _f32_ghosted(np: Any, value: Any, label: str) -> Any:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 1 or array.size < 3:
        raise ValueError(f"{label} must be a float32 rank-1 CM1-indexed coordinate")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{label} carries a non-finite word")
    return array


def build_cm1_vertical_map(old_coordinate: Any, new_coordinate: Any, *, nzold: int) -> Cm1VerticalMap:
    """Transliterate ``restart_read.F:564-575`` and ``prep_for_interp``.

    ``old_coordinate`` is the OLD deck's ghosted coordinate starting at CM1
    index 0 (``sigma0(0:nzold+1)`` or ``sigmaf0(0:nzold+2)``);
    ``new_coordinate`` is the NEW deck's interior levels (``sigma(1:nk)`` or
    ``sigmaf(1:nk+1)``); ``nzold`` is the number of OLD data levels the
    carrier has (``nk`` for scalars, ``nk+1`` for ``w``).
    """

    np = _np()
    old = _f32_ghosted(np, old_coordinate, "old coordinate")
    new = _f32_ghosted(np, new_coordinate, "new coordinate")
    if nzold < 1 or old.size < nzold + 2:
        raise ValueError("the old coordinate does not carry its data levels and ghosts")
    count = int(new.size)
    iz = np.empty(count, dtype=np.int32)
    rz = np.empty(count, dtype=np.float32)
    kold = 0
    for k in range(count):
        target = new[k]
        # Advance ``kold`` while ``sigma0(kold+1)`` is below ``sigma(k)`` -- a float32 comparison.
        while True:
            if kold + 1 >= old.size:
                raise ValueError(
                    f"new level {k + 1} at {float(target)} m lies above the old "
                    "deck's last ghost level; CM1's search would read past sigma0"
                )
            if not bool(old[kold + 1] < target):
                break
            kold += 1
        iz[k] = kold
        numerator = np.float32(target - old[kold])
        denominator = np.float32(old[kold + 1] - old[kold])
        rz[k] = np.float32(numerator / denominator)

    # prep_for_interp: kold1=-1, kold2=0, alldone=.false. (:3037-3039).
    lower = np.empty(count, dtype=np.int32)
    upper = np.empty(count, dtype=np.int32)
    kold1, kold2, alldone = -1, 0, False
    datk1: int | None = None
    datk2: int | None = None
    for k in range(count):
        if int(iz[k]) + 1 > kold2:
            if int(iz[k]) + 1 <= nzold:
                for _ in range(kold2 + 1, int(iz[k]) + 2):
                    kold1 += 1
                    kold2 += 1
                    datk1 = datk2
                    datk2 = kold2  # read_binary reads the next old level
                if kold1 == 0:
                    datk1 = datk2
            else:
                if not alldone:
                    datk1 = datk2
                alldone = True
        if datk1 is None or datk2 is None:
            raise ValueError("CM1's plane state machine never read an old level")
        lower[k] = datk1 - 1
        upper[k] = datk2 - 1
    below = np.less(new, old[1])
    above = np.greater(new, old[nzold])
    for array in (lower, upper, rz, iz, below, above):
        array.flags.writeable = False
    return Cm1VerticalMap(lower, upper, rz, iz, below, above)


def cm1_vertical_weights(fraction: Any) -> tuple[Any, ...]:
    """``do_interp``'s eight weights at ``rx = ry = 0``, REAL then widened.

    ``restart_read.F:3437-3444``: each right-hand side is a default-REAL
    product, left-associated, assigned to a DOUBLE PRECISION weight.
    """

    np = _np()
    z = np.asarray(fraction, dtype=np.float32)
    x = np.float32(0.0)
    y = np.float32(0.0)
    one = np.float32(1.0)
    ox = np.float32(one - x)
    oy = np.float32(one - y)
    oz = (one - z).astype(np.float32)

    def product(a: Any, b: Any, c: Any) -> Any:
        return (np.float32(a * b) * c).astype(np.float32).astype(np.float64)

    return (
        product(ox, oy, oz),
        product(x, oy, oz),
        product(ox, y, oz),
        product(ox, oy, z),
        product(x, oy, z),
        product(ox, y, z),
        product(x, y, oz),
        product(x, y, z),
    )


def cm1_vertical_interpolate(volume: Any, vmap: Cm1VerticalMap) -> Any:
    """Every column of ``[nzold, ...]`` onto the new levels, float32 out.

    ``var = datk1*w1 + datk1*w2 + datk1*w3 + datk2*w4 + datk2*w5 + datk2*w6 +
    datk1*w7 + datk2*w8`` (``restart_read.F:3458-3465``), left to right in
    double, narrowed once.  At ``rx = ry = 0`` every horizontal neighbour is
    the column itself.
    """

    np = _np()
    data = np.asarray(volume)
    if data.dtype != np.float32:
        raise ValueError("the carriers interp_on_restart reads are default REAL words")
    w1, w2, w3, w4, w5, w6, w7, w8 = cm1_vertical_weights(vmap.fraction)
    shape = (vmap.levels,) + (1,) * (data.ndim - 1)
    lo = data[np.asarray(vmap.lower, dtype=np.intp)].astype(np.float64)
    hi = data[np.asarray(vmap.upper, dtype=np.intp)].astype(np.float64)
    result = lo * w1.reshape(shape)
    result = result + lo * w2.reshape(shape)
    result = result + lo * w3.reshape(shape)
    result = result + hi * w4.reshape(shape)
    result = result + hi * w5.reshape(shape)
    result = result + hi * w6.reshape(shape)
    result = result + lo * w7.reshape(shape)
    result = result + hi * w8.reshape(shape)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Admission.
# ---------------------------------------------------------------------------
def _faces_f64(np: Any, levels: Any) -> Any:
    return np.asarray(levels.zf[1:levels.nk + 2], dtype=np.float32).astype(np.float64)


def vertical_refinement_defect(parent: Any, child: Any) -> str | None:
    """Name why ``child`` cannot represent ``parent``'s column, or ``None``.

    Two conditions, both about the column rather than a tolerance:

    * the model top is the same binary32 word -- a different top would leave
      parent air with no child level (or child levels with no parent air), and
      CM1's constant extension would silently fill them from the last level;
    * no child layer is thicker than every parent layer it overlaps.  A child
      coarser than its parent somewhere would average away structure the
      parent resolves and hand it back through feedback as if resolved.
    """

    np = _np()
    if parent.sha256 == child.sha256:
        return None
    parent_top = np.float32(parent.zf[parent.nk + 1])
    child_top = np.float32(child.zf[child.nk + 1])
    if parent_top.tobytes() != child_top.tobytes():
        return (
            f"the child's model top is {float(child_top)} m and the parent's "
            f"{float(parent_top)} m; the child would either leave parent air "
            "above its lid or carry levels CM1's constant extension would fill "
            "from the parent's last level"
        )
    pf = _faces_f64(np, parent)
    cf = _faces_f64(np, child)
    for m in range(child.nk):
        lo, hi = cf[m], cf[m + 1]
        thickness = hi - lo
        widest = max(
            pf[k + 1] - pf[k] for k in range(parent.nk) if pf[k] < hi and pf[k + 1] > lo
        )
        if thickness > widest:
            return (
                f"child layer {m + 1} ({lo} m to {hi} m) is {thickness} m thick, "
                f"thicker than every parent layer it overlaps (at most {widest} m); "
                "the child would be coarser than its parent there, and feedback "
                "would return an average as if it were resolved"
            )
    return None


@dataclass(frozen=True, slots=True)
class VerticalRefinement:
    """A checked parent/child vertical pair and the maps between them."""

    parent: Any
    child: Any
    identical: bool
    mass: Cm1VerticalMap | None
    face: Cm1VerticalMap | None

    def map_for(self, stagger: str) -> Cm1VerticalMap:
        if self.identical:
            raise ValueError("identical decks carry no vertical map; use the copy path")
        return self.face if stagger == "w" else self.mass  # type: ignore[return-value]

    def record(self) -> dict[str, Any]:
        np = _np()
        payload: dict[str, Any] = {
            "schema": NEST_VERTICAL_SCHEMA,
            "identical_decks": self.identical,
            "parent_nz": int(self.parent.nk),
            "child_nz": int(self.child.nk),
            "parent_vertical_sha256": self.parent.sha256,
            "child_vertical_sha256": self.child.sha256,
            "parent_lowest_layer_m": float(np.float32(self.parent.zf[2]) - np.float32(self.parent.zf[1])),
            "child_lowest_layer_m": float(np.float32(self.child.zf[2]) - np.float32(self.child.zf[1])),
            "model_top_m": float(self.parent.zf[self.parent.nk + 1]),
        }
        if self.identical:
            payload["vertical_treatment"] = "copied level for level; the decks are identical"
        else:
            payload.update(
                {
                    "vertical_treatment": (
                        "CM1 interp_on_restart at rx=ry=0 on a parent-resolution "
                        "window, then the nest's horizontal capture"
                    ),
                    "cm1_authority": dict(CM1_AUTHORITY),
                    "wrf_authority": dict(WRF_AUTHORITY),
                    "gm1_choices": dict(GM1_CHOICES),
                    "child_levels_below_parent_first_scalar": int(np.count_nonzero(self.mass.below_support)),
                    "child_levels_above_parent_last_scalar": int(np.count_nonzero(self.mass.above_support)),
                }
            )
        return payload


def plan_vertical_refinement(parent: Any, child: Any) -> VerticalRefinement:
    """Check the pair and build CM1's maps; raises ``ValueError`` by name."""

    if parent.sha256 == child.sha256:
        return VerticalRefinement(parent, child, True, None, None)
    defect = vertical_refinement_defect(parent, child)
    if defect is not None:
        raise ValueError(defect)
    mass = build_cm1_vertical_map(
        parent.sigma, child.sigma[1:child.nk + 1], nzold=parent.nk
    )
    face = build_cm1_vertical_map(
        parent.sigmaf, child.sigmaf[1:child.nk + 2], nzold=parent.nk + 1
    )
    return VerticalRefinement(parent, child, False, mass, face)


def require_vertical_refinement(
    parent: Any, child: Any, *, code: str, gate: str, option: str
) -> VerticalRefinement:
    """:func:`plan_vertical_refinement`, refused through the error contract."""

    from .errors import refuse

    try:
        return plan_vertical_refinement(parent, child)
    except ValueError as error:
        refuse(
            code=code,
            gate=gate,
            reason=f"the child's vertical deck cannot represent the parent's column: {error}",
            remedy=(
                "give the child the parent's model top and a deck no coarser than "
                "the parent's at any height (for example a smaller dz_bot with the "
                "parent's dz_top), or the parent's deck exactly"
            ),
            option=option,
            value={"parent_nz": int(parent.nk), "child_nz": int(child.nk)},
        )


def levels_of_grid(grid: Any) -> Any:
    """The CM1 coordinate a runtime grid carries, the uniform one built when absent."""

    from .cm1_vertical_grid import uniform_vertical_levels

    vertical = getattr(grid, "vertical", None)
    if vertical is not None:
        return vertical
    return uniform_vertical_levels(int(grid.nz), float(grid.dz_m))


# ---------------------------------------------------------------------------
# Feedback weights.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RestrictionWeights:
    """CSR rows: parent level ``k`` takes child levels ``index[offsets[k]:offsets[k+1]]``."""

    offsets: Any
    index: Any
    weight: Any
    mass: Any  # float64 M_k, the child's mass measure of each parent level

    @property
    def parent_levels(self) -> int:
        return int(self.offsets.size - 1)


def _csr(np: Any, rows: list[list[tuple[int, float]]], masses: list[float]) -> RestrictionWeights:
    offsets = np.zeros(len(rows) + 1, dtype=np.int32)
    index: list[int] = []
    weight: list[float] = []
    for k, row in enumerate(rows):
        offsets[k + 1] = offsets[k] + len(row)
        for m, w in row:
            index.append(m)
            weight.append(w)
    return RestrictionWeights(
        offsets, np.asarray(index, dtype=np.int32), np.asarray(weight, dtype=np.float64),
        np.asarray(masses, dtype=np.float64),
    )


def _overlap_rows(np: Any, parent_edges: Any, child_edges: Any, density: Any,
                  child_ids: Any) -> tuple[list[list[tuple[int, float]]], list[float]]:
    rows: list[list[tuple[int, float]]] = []
    masses: list[float] = []
    for k in range(parent_edges.size - 1):
        lo, hi = float(parent_edges[k]), float(parent_edges[k + 1])
        terms: list[tuple[int, float]] = []
        for c in range(child_edges.size - 1):
            overlap = min(hi, float(child_edges[c + 1])) - max(lo, float(child_edges[c]))
            if overlap > 0.0:
                terms.append((int(child_ids[c]), float(density[c]) * overlap))
        total = 0.0
        for _, a in terms:
            total = total + a
        if total <= 0.0:
            raise ValueError(f"parent level {k + 1} overlaps no child mass")
        rows.append([(m, a / total) for m, a in terms])
        masses.append(total)
    return rows, masses


def feedback_weights(refinement: VerticalRefinement, *, stagger: str,
                     child_density: Any) -> RestrictionWeights:
    """The conservative vertical restriction for one lattice.

    ``child_density`` is the child's base density column: ``rho0`` on its
    ``nk`` mass levels for ``u``, ``v`` and the scalars, ``rf0`` on its
    ``nk+1`` faces for ``w``.  Weights are float64, each row ordered by child
    level and normalized by its own float64 sum.
    """

    np = _np()
    parent, child = refinement.parent, refinement.child
    density = np.asarray(child_density, dtype=np.float64)
    if stagger != "w":
        if density.shape != (child.nk,):
            raise ValueError(f"mass-level density must be ({child.nk},)")
        rows, masses = _overlap_rows(
            np, _faces_f64(np, parent), _faces_f64(np, child), density,
            np.arange(child.nk),
        )
        return _csr(np, rows, masses)
    if density.shape != (child.nk + 1,):
        raise ValueError(f"face density must be ({child.nk + 1},)")
    pzh = np.asarray(parent.zh[1:parent.nk + 1], dtype=np.float32).astype(np.float64)
    czh = np.asarray(child.zh[1:child.nk + 1], dtype=np.float32).astype(np.float64)
    if czh[0] > pzh[0] or czh[-1] < pzh[-1]:
        raise ValueError(
            "the child's first or last scalar level lies outside the parent's, so "
            "a child boundary face's dual cell would enter an interior parent one"
        )
    interior_rows, interior_masses = _overlap_rows(
        np, pzh, czh, density[1:child.nk], np.arange(1, child.nk)
    )
    top = float(parent.zf[parent.nk + 1])
    rows = [[(0, 1.0)]] + interior_rows + [[(child.nk, 1.0)]]
    masses = [float(density[0]) * float(czh[0])] + interior_masses + [
        float(density[child.nk]) * (top - float(czh[-1]))
    ]
    return _csr(np, rows, masses)


# ---------------------------------------------------------------------------
# The parent-resolution window the vertical step fills.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VerticalWindow:
    """The parent lattice rectangle every captured child point reads.

    Donor indices on a mass axis run ``parent_i0 .. parent_i0+span-1`` and on a
    face axis ``parent_i0 .. parent_i0+span``; the limited slopes read one
    parent point either side.  The window is that range widened by one point,
    so inside it the capture's clamps never engage -- exactly as they never
    engage on the whole parent lattice, which ``NestGeometry`` keeps
    ``PARENT_STENCIL_HALO`` cells clear of its edges.
    """

    j0: int
    i0: int
    nj: int
    ni: int

    @classmethod
    def for_stagger(cls, geometry: Any, stagger: str) -> "VerticalWindow":
        from .nest_live_kernels import LATTICE_OF_STAGGER

        x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
        return cls(
            j0=int(geometry.parent_j0) - 1,
            i0=int(geometry.parent_i0) - 1,
            nj=int(geometry.parent_span_y) + 2 + y_face,
            ni=int(geometry.parent_span_x) + 2 + x_face,
        )

    def require_within(self, parent_lattice: tuple[int, int, int]) -> None:
        _, nyl, nxl = parent_lattice
        if self.j0 < 0 or self.i0 < 0 or self.j0 + self.nj > nyl or self.i0 + self.ni > nxl:
            raise ValueError(
                f"the vertical window {self} leaves the {parent_lattice} parent lattice"
            )


# ---------------------------------------------------------------------------
# Device kernels.
# ---------------------------------------------------------------------------
NEST_VERTICAL_KERNEL_NAMES = ("gm1_nest_vinterp", "gm1_nest_restrict_vertical")

NEST_VERTICAL_KERNEL_SOURCE = r'''// gm1.cartesian-nest-vertical-refinement/v1
// CM1 r21.1 has no nesting; the vertical rule is CM1's interp_on_restart
// (src/restart_read.F:564-609, 3302-3465) evaluated at rx = ry = 0, placed
// in WRF's order (external/RSL_LITE/force_domain_em_part2.F:120-126).
#define GM1_NV_IDX(k, j, i, nj, ni) \
    ((((unsigned long long)(k) * (unsigned long long)(nj)) + (unsigned long long)(j)) \
     * (unsigned long long)(ni) + (unsigned long long)(i))

// One output point per thread over (nzl_out, win_nj, win_ni).  lower/upper
// are the 0-based old levels datk1/datk2 hold for new level k; rz is CM1's
// REAL rz.  The eight weights are do_interp's REAL products widened to
// double; the sum is do_interp's written order.
extern "C" __global__ void gm1_nest_vinterp(
    const float* __restrict__ parent, float* __restrict__ out,
    const int* __restrict__ lower, const int* __restrict__ upper,
    const float* __restrict__ rz,
    int p_nj_owner, int p_ni_owner, int p_k0, int p_j0, int p_i0,
    int win_j0, int win_i0, int nzl_out, int win_nj, int win_ni)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long plane = (unsigned long long)win_nj * (unsigned long long)win_ni;
    if (tid >= (unsigned long long)nzl_out * plane) return;
    int k = (int)(tid / plane);
    unsigned long long rem = tid - (unsigned long long)k * plane;
    int a = (int)(rem / (unsigned long long)win_ni);
    int b = (int)(rem - (unsigned long long)a * win_ni);
    int j = win_j0 + a + p_j0;
    int i = win_i0 + b + p_i0;
    float x = 0.0f, y = 0.0f, one = 1.0f, z = rz[k];
    float ox = __fsub_rn(one, x);
    float oy = __fsub_rn(one, y);
    float oz = __fsub_rn(one, z);
    double w1 = (double)__fmul_rn(__fmul_rn(ox, oy), oz);
    double w2 = (double)__fmul_rn(__fmul_rn(x, oy), oz);
    double w3 = (double)__fmul_rn(__fmul_rn(ox, y), oz);
    double w4 = (double)__fmul_rn(__fmul_rn(ox, oy), z);
    double w5 = (double)__fmul_rn(__fmul_rn(x, oy), z);
    double w6 = (double)__fmul_rn(__fmul_rn(ox, y), z);
    double w7 = (double)__fmul_rn(__fmul_rn(x, y), oz);
    double w8 = (double)__fmul_rn(__fmul_rn(x, y), z);
    double lo = (double)parent[GM1_NV_IDX(lower[k] + p_k0, j, i, p_nj_owner, p_ni_owner)];
    double hi = (double)parent[GM1_NV_IDX(upper[k] + p_k0, j, i, p_nj_owner, p_ni_owner)];
    double r = lo * w1;
    r = r + lo * w2;
    r = r + lo * w3;
    r = r + hi * w4;
    r = r + hi * w5;
    r = r + hi * w6;
    r = r + lo * w7;
    r = r + hi * w8;
    out[tid] = (float)r;
}

// gm1_nest_restrict with a vertical row: each covered parent point on parent
// level k becomes sum_e weight[e] * blocksum(child level index[e]) / (r*r),
// the block sum in float64 raster order and the row in CSR order.  With one
// entry of weight 1.0 per level this is gm1_nest_restrict's arithmetic.
extern "C" __global__ void gm1_nest_restrict_vertical(
    const float* __restrict__ child, float* __restrict__ parent,
    const int* __restrict__ offsets, const int* __restrict__ index,
    const double* __restrict__ weight,
    int c_nj_owner, int c_ni_owner, int c_k0, int c_j0, int c_i0,
    int p_nj_owner, int p_ni_owner, int p_k0, int p_j0, int p_i0,
    int nzl_parent, int cl_nj, int cl_ni, int pnl_j, int pnl_i,
    int ratio, int relax, int parent_i0, int parent_j0, int x_face, int y_face)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long plane = (unsigned long long)pnl_j * (unsigned long long)pnl_i;
    if (tid >= (unsigned long long)nzl_parent * plane) return;
    int k = (int)(tid / plane);
    unsigned long long rem = tid - (unsigned long long)k * plane;
    int jp = (int)(rem / (unsigned long long)pnl_i);
    int ip = (int)(rem - (unsigned long long)jp * pnl_i);
    int h = (ratio - 1) / 2;
    int i0 = x_face ? ip * ratio - h : ip * ratio;
    int j0 = y_face ? jp * ratio - h : jp * ratio;
    if (i0 < relax || j0 < relax) return;
    if (i0 + ratio > cl_ni - relax) return;
    if (j0 + ratio > cl_nj - relax) return;
    double total = 0.0;
    for (int e = offsets[k]; e < offsets[k + 1]; ++e) {
        int m = index[e];
        double acc = 0.0;
        for (int aa = 0; aa < ratio; ++aa) {
            for (int bb = 0; bb < ratio; ++bb) {
                acc = acc + (double)child[GM1_NV_IDX(m + c_k0, j0 + aa + c_j0,
                                                     i0 + bb + c_i0,
                                                     c_nj_owner, c_ni_owner)];
            }
        }
        total = total + weight[e] * acc;
    }
    parent[GM1_NV_IDX(k + p_k0, jp + parent_j0 + p_j0, ip + parent_i0 + p_i0,
                      p_nj_owner, p_ni_owner)] =
        (float)(total / (double)(ratio * ratio));
}
'''
NEST_VERTICAL_KERNEL_SHA256 = hashlib.sha256(
    NEST_VERTICAL_KERNEL_SOURCE.encode("utf-8")
).hexdigest()


def compile_nest_vertical_module(cupy: Any) -> Any:
    from .nest_live_kernels import NEST_LIVE_NVRTC_OPTIONS
    from .nvrtc_module import compile_nvrtc_module

    return compile_nvrtc_module(
        cupy, code=NEST_VERTICAL_KERNEL_SOURCE, options=NEST_LIVE_NVRTC_OPTIONS,
        filename="gm1_nest_vertical.cu",
    )


def launch_vinterp(
    cupy: Any, module: Any, *, parent: Any, parent_origin: tuple[int, int, int],
    parent_lattice: tuple[int, int, int], window: VerticalWindow, device_map: Mapping[str, Any],
    out: Any, stream: Any = None,
) -> None:
    from .nest_live_kernels import _launch_grid, _owner_layout, _require_f32

    p = _owner_layout(parent, parent_origin, parent_lattice)
    _require_f32(cupy, parent, out)
    window.require_within(parent_lattice)
    levels = int(device_map["lower"].size)
    if tuple(out.shape) != (levels, window.nj, window.ni) or not out.flags.c_contiguous:
        raise ValueError(f"the vertical window output must be {(levels, window.nj, window.ni)}")
    lower_max = int(device_map["lower_max"])
    if lower_max >= parent_lattice[0]:
        raise ValueError("the vertical map reads a parent level the lattice does not carry")
    blocks, threads = _launch_grid(int(out.size))
    i32 = cupy.int32
    module.get_function("gm1_nest_vinterp")(
        blocks, threads,
        (parent, out, device_map["lower"], device_map["upper"], device_map["rz"],
         i32(p[0]), i32(p[1]), i32(p[2]), i32(p[3]), i32(p[4]),
         i32(window.j0), i32(window.i0), i32(levels), i32(window.nj), i32(window.ni)),
        stream=stream,
    )


def upload_vertical_map(cupy: Any, vmap: Cm1VerticalMap) -> dict[str, Any]:
    np = _np()
    return {
        "lower": cupy.asarray(np.ascontiguousarray(vmap.lower, dtype=np.int32)),
        "upper": cupy.asarray(np.ascontiguousarray(vmap.upper, dtype=np.int32)),
        "rz": cupy.asarray(np.ascontiguousarray(vmap.fraction, dtype=np.float32)),
        "lower_max": int(max(int(vmap.lower.max()), int(vmap.upper.max()))),
    }


def upload_weights(cupy: Any, weights: RestrictionWeights) -> dict[str, Any]:
    np = _np()
    return {
        "offsets": cupy.asarray(np.ascontiguousarray(weights.offsets, dtype=np.int32)),
        "index": cupy.asarray(np.ascontiguousarray(weights.index, dtype=np.int32)),
        "weight": cupy.asarray(np.ascontiguousarray(weights.weight, dtype=np.float64)),
        "levels": weights.parent_levels,
        "index_max": int(weights.index.max()),
    }


def launch_restrict_vertical(
    cupy: Any, module: Any, *, child: Any, child_origin: tuple[int, int, int],
    child_lattice: tuple[int, int, int], parent: Any, parent_origin: tuple[int, int, int],
    parent_lattice: tuple[int, int, int], device_weights: Mapping[str, Any],
    span_x: int, span_y: int, ratio: int, relax_zone: int, parent_i0: int,
    parent_j0: int, stagger: str, stream: Any = None,
) -> None:
    from .nest_live_kernels import LATTICE_OF_STAGGER, _launch_grid, _owner_layout, _require_f32

    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    c = _owner_layout(child, child_origin, child_lattice)
    p = _owner_layout(parent, parent_origin, parent_lattice)
    _require_f32(cupy, child, parent)
    if int(device_weights["levels"]) != parent_lattice[0]:
        raise ValueError("the restriction rows do not cover the parent's levels")
    if int(device_weights["index_max"]) >= child_lattice[0]:
        raise ValueError("the restriction reads a child level the lattice does not carry")
    pnl_j, pnl_i = span_y + y_face, span_x + x_face
    if parent_j0 + pnl_j > parent_lattice[1] or parent_i0 + pnl_i > parent_lattice[2]:
        raise ValueError("the covered parent rectangle does not fit the parent lattice")
    count = parent_lattice[0] * pnl_j * pnl_i
    blocks, threads = _launch_grid(count)
    i32 = cupy.int32
    module.get_function("gm1_nest_restrict_vertical")(
        blocks, threads,
        (child, parent, device_weights["offsets"], device_weights["index"], device_weights["weight"],
         i32(c[0]), i32(c[1]), i32(c[2]), i32(c[3]), i32(c[4]),
         i32(p[0]), i32(p[1]), i32(p[2]), i32(p[3]), i32(p[4]),
         i32(parent_lattice[0]), i32(child_lattice[1]), i32(child_lattice[2]),
         i32(pnl_j), i32(pnl_i), i32(ratio), i32(relax_zone),
         i32(parent_i0), i32(parent_j0), i32(x_face), i32(y_face)),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# CPU references.
# ---------------------------------------------------------------------------
def reference_vinterp(parent_lattice_volume: Any, vmap: Cm1VerticalMap,
                      window: VerticalWindow) -> Any:
    """``gm1_nest_vinterp`` on the host: the window, then CM1's column rule."""

    np = _np()
    parent = np.asarray(parent_lattice_volume, dtype=np.float32)
    window.require_within(tuple(parent.shape))
    cut = np.ascontiguousarray(
        parent[:, window.j0:window.j0 + window.nj, window.i0:window.i0 + window.ni]
    )
    return cm1_vertical_interpolate(cut, vmap)


def reference_capture_refined(
    parent_lattice_volume: Any, *, vmap: Cm1VerticalMap, window: VerticalWindow,
    child_lattice: tuple[int, int, int], child_volume: Any | None, side: str,
    width: int, ratio: int, stagger: str, parent_dt_s: float,
) -> tuple[Any, Any | None]:
    """Vertical window, then ``reference_capture`` with donors inside the window."""

    from .nest_live_kernels import reference_capture

    column = reference_vinterp(parent_lattice_volume, vmap, window)
    return reference_capture(
        column, child_lattice=child_lattice, child_volume=child_volume, side=side,
        width=width, ratio=ratio, parent_i0=1, parent_j0=1, stagger=stagger,
        parent_dt_s=parent_dt_s,
    )


def reference_restrict_vertical(
    child: Any, parent: Any, *, weights: RestrictionWeights,
    child_lattice: tuple[int, int, int], span_x: int, span_y: int, ratio: int,
    relax_zone: int, parent_i0: int, parent_j0: int, stagger: str,
) -> Any:
    """``gm1_nest_restrict_vertical`` on physical lattice volumes (origin 0)."""

    np = _np()
    from .nest_live_kernels import LATTICE_OF_STAGGER

    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    out = np.array(parent, dtype=np.float32, copy=True)
    _, cl_nj, cl_ni = child_lattice
    h = (ratio - 1) // 2
    child = np.asarray(child, dtype=np.float32)
    for k in range(weights.parent_levels):
        for jp in range(span_y + y_face):
            for ip in range(span_x + x_face):
                i0 = ip * ratio - h if x_face else ip * ratio
                j0 = jp * ratio - h if y_face else jp * ratio
                if i0 < relax_zone or j0 < relax_zone:
                    continue
                if i0 + ratio > cl_ni - relax_zone or j0 + ratio > cl_nj - relax_zone:
                    continue
                total = 0.0
                for e in range(int(weights.offsets[k]), int(weights.offsets[k + 1])):
                    m = int(weights.index[e])
                    acc = 0.0
                    for aa in range(ratio):
                        for bb in range(ratio):
                            acc = acc + float(child[m, j0 + aa, i0 + bb])
                    total = total + float(weights.weight[e]) * acc
                out[k, jp + parent_j0, ip + parent_i0] = np.float32(total / float(ratio * ratio))
    return out


# ---------------------------------------------------------------------------
# Host downscale (the one-way initial condition tool).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _WindowGeometry:
    """A NestGeometry view whose donors are indexed inside a VerticalWindow."""

    base: Any
    parent_i0: int = 1
    parent_j0: int = 1

    @property
    def refinement_ratio(self) -> int:
        return int(self.base.refinement_ratio)

    @property
    def nx(self) -> int:
        return int(self.base.nx)

    @property
    def ny(self) -> int:
        return int(self.base.ny)

    def sub_cell_offsets(self) -> tuple[float, ...]:
        return tuple(self.base.sub_cell_offsets())


def downscale_volume_refined(parent_volume: Any, geometry: Any, refinement: VerticalRefinement,
                             *, x_lattice: str, y_lattice: str, vertical: str) -> Any:
    """A parent carrier onto the child's lattice through a different deck.

    ``vertical`` is ``"mass"`` for carriers on ``sigma`` levels and ``"w"``
    for carriers on ``sigmaf`` faces.  Identical decks never reach here.
    """

    np = _np()
    from .nest_downscale import downscale_level_staggered

    if refinement.identical:
        raise ValueError("identical decks take the copy path")
    stagger = "w" if vertical == "w" else {
        ("face", "mass"): "u", ("mass", "face"): "v", ("mass", "mass"): "mass"
    }[(x_lattice, y_lattice)]
    window = VerticalWindow.for_stagger(geometry, stagger)
    column = reference_vinterp(parent_volume, refinement.map_for(vertical), window)
    view = _WindowGeometry(geometry)
    first = downscale_level_staggered(column[0], view, x_lattice=x_lattice, y_lattice=y_lattice)
    captured = np.empty((column.shape[0],) + first.shape, dtype=np.float64)
    captured[0] = first
    for k in range(1, column.shape[0]):
        captured[k] = downscale_level_staggered(
            column[k], view, x_lattice=x_lattice, y_lattice=y_lattice
        )
    return captured


__all__ = [
    "CM1_AUTHORITY",
    "Cm1VerticalMap",
    "GM1_CHOICES",
    "NEST_VERTICAL_KERNEL_NAMES",
    "NEST_VERTICAL_KERNEL_SHA256",
    "NEST_VERTICAL_KERNEL_SOURCE",
    "NEST_VERTICAL_SCHEMA",
    "RestrictionWeights",
    "VerticalRefinement",
    "VerticalWindow",
    "WRF_AUTHORITY",
    "build_cm1_vertical_map",
    "cm1_vertical_interpolate",
    "cm1_vertical_weights",
    "compile_nest_vertical_module",
    "downscale_volume_refined",
    "feedback_weights",
    "launch_restrict_vertical",
    "launch_vinterp",
    "levels_of_grid",
    "plan_vertical_refinement",
    "reference_capture_refined",
    "reference_restrict_vertical",
    "reference_vinterp",
    "require_vertical_refinement",
    "upload_vertical_map",
    "upload_weights",
    "vertical_refinement_defect",
]
