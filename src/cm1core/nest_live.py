"""One refined child integrated concurrently inside a running parent, on one GPU.

CM1 r21.1 has no nesting (``records/CM1-NEST-SEAM.md`` section 1).  The
authority for **where** boundary forcing enters the time step is WRF, whose
``dyn_em/solve_em.F`` is the split-explicit RK3 integrator GM1's
``psolver=3`` step most closely resembles.  Every placement below names the
WRF line it follows and the GM1 array it lands in; section 5 of the seam
record carries the same table for a reader who has only the record.

ONE PARENT STEP (``frame/module_integrate.F:393-443``)
------------------------------------------------------
1. The parent completes its large step (``solve_interface``, ``:393``).
2. ``med_nest_force`` (``:416``) builds the child's boundary tables from the
   parent's new state: ``bdy_xs = nfld`` (the child's own value now) and
   ``bdy_txs = rdt*(interp(parent) - nfld)`` with ``rdt = 1/dt_parent``
   (``share/interp_fcn.F:2500,2583-2584``), and resets ``dtbc = 0``
   (``share/mediation_force_domain.F:213``).  The boundary target is
   therefore linear in time across the parent interval and reaches the
   interpolated parent exactly at its end.
3. The child integrates ``time_refinement_ratio`` large steps
   (``integrate``, ``:430``).  Each one first advances ``dtbc = dtbc + dt``
   (``dyn_em/solve_em.F:372``), so the relaxation target inside child step
   ``n`` (1-based) is ``bdy_xs + n*dtc*bdy_txs``.
4. ``med_nest_feedback`` (``:443``) writes the child's interior back into the
   parent when ``feedback = 1``.

INSIDE ONE CHILD STEP
---------------------
* **Relaxation zone, once per large step, into the held tendencies.**  WRF
  calls ``relax_bdy_dry`` only when ``rk_step == 1`` (``solve_em.F:933-957``)
  and writes it into ``*_save``, which ``rk_addtend_dry`` folds into the
  physics tendency ``*_tendf`` at ``rk_step == 1`` and adds to the large-step
  tendency in every stage (``dyn_em/module_em.F:1047-1080``).  GM1's analogue
  of ``*_tendf`` is the held tendency CM1 builds once per step and every RK
  stage re-reads: ``held.uten/vten/wten`` (packed into the stage momentum
  tendency by ``pack_cartesian_held_momentum``), ``held.thten1``,
  ``held.qten`` and ``arrays.ppten``.  The Davies term is added there, right
  after ``finalize_held_tendencies``, from the time-level-``n`` state.  Moist
  species follow WRF's ``relax_bdy_scalar`` (``solve_em.F:2258-2276``), also
  at ``rk_step == 1``.
* **Specified zone, inside every acoustic small step.**  WRF sets the
  specified-zone large-step tendency to the boundary tendency
  (``spec_bdy_dry``, ``solve_em.F:979``), excludes the zone from
  ``advance_uv`` (``module_small_step_em.F:734-742``) and then advances it by
  that tendency alone after each small-step advance: ``u_2``/``v_2`` after
  ``advance_uv`` (``solve_em.F:1341,1351``), ``t_2``/``mu_2`` after
  ``advance_mu_t`` (``:1457,1467``), ``ph_2``/``w_2`` after ``advance_w``
  (``:1582-1604``).  GM1's acoustic operator writes the specified rows too, so
  their words are gathered at the top of the small step and rewritten as
  ``gathered + dts*tendency`` after the horizontal momentum advance (``u3d``,
  ``v3d``) and after the implicit column step (``w3d``, ``pp3d``).
* **Specified zone for the RK scalars, after each stage's update.**  WRF's
  moist species take ``spec_bdy_scalar`` (``solve_em.F:2277``) and then
  ``rk_update_scalar`` (``:2305``), i.e. ``q = q_old + dt_rk*tend`` on the
  specified rows.  In GM1 ``psolver=3`` potential temperature is an RK scalar
  like moisture (it is not advanced in the acoustic loop), so ``th3d`` and
  every ``q3d`` species take ``field = A-level + dt_rk*tendency`` on the
  specified rows after ``scalar_moist``.

* **The specified rows at the target, at the end of every step.**  WRF ends
  ``solve_em`` with ``spec_bdy_final`` (WRF v4.6.1 ``dyn_em/solve_em.F:4531-
  4716``, ``share/module_bc.F:2066-2216``): after microphysics the specified
  rows are set to ``bdy + dtbc*bdy_tend``; a nested domain does it for u, v,
  w, t, ph, mu and every moist, tracer and scalar field.  GM1 sets the
  specified rows of ``u3d``, ``v3d``, ``w3d``, ``pp3d``, ``th3d`` and every
  ``q3d`` species to ``value + dtbc*tend`` just before ``solve3`` copies them
  into the time-level-``n`` arrays (:meth:`NestLateralForcing.
  finalize_specified`).  Without it microphysics acted on the specified rows
  after their stage update and nothing returned them before the next parent
  step's tables were cut from the child's own words.

WRF -> GM1 VARIABLES
--------------------
===========================  ==========================  ======================
WRF (mass coordinate)        GM1 (height, compressible)  translation
===========================  ==========================  ======================
``ru``, ``rv`` (mu*u, mu*v)  ``u3d``, ``v3d``            GM1's base density is
                                                         horizontally uniform
                                                         on the non-terrain
                                                         grids this lane admits,
                                                         so relaxing ``rho0*u``
                                                         and ``u`` is one term
``w`` (nested only)          ``w3d``                     same
``t`` (mu-coupled theta)     ``th3d`` (theta')           uncoupled; no mu
``mu``, ``ph``               ``pp3d`` (pi')              GM1's only pressure
                                                         prognostic; WRF's mu
                                                         and geopotential have
                                                         no separate counterpart
``moist``                    ``q3d``                     uncoupled; no mu
``tke_2``                    ``tke3d``                   not forced: WRF's tke
                                                         advance (``:2378-
                                                         2416``) has no bdy call
===========================  ==========================  ======================

WRF applies ``mu`` relaxation through ``mu_tend``, which ``rk_tendency``
rewrites each stage, so its mass relaxation acts in stage 1 only.  GM1's
``ppten`` is a held tendency CM1 builds once per step (``solve1``), so the
relaxation added to it acts in all three stages, the same way WRF's ``u``,
``v``, ``w``, ``t`` and ``ph`` relaxation does.  That is the one place the
mapping is a choice rather than a transcription, and the receipt says so.

CM1's dry-mass correction (``apmasscon``) is not run in the child: WRF has no
counterpart in a nested domain, whose mass is set through its specified zone,
and holding the child to its own initial dry mass would add a domain-uniform
pressure offset that the specified zone contradicts on every boundary row.
The parent keeps it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from . import cuda_graphs
from .nest_boundary import (
    DAVIES_AUTHORITY,
    compile_nest_relaxation_module,
    davies_weights,
    launch_nest_specified_relaxation,
)
from .nest_live_kernels import (
    NEST_LIVE_KERNEL_SHA256,
    compile_nest_live_module,
    lattice_shape,
    launch_capture,
    launch_restrict,
    launch_spec_apply,
    launch_spec_gather,
    side_shape,
)

NEST_LIVE_SCHEMA = "gm1.cartesian-nest-live/v1"
SIDES = ("west", "east", "south", "north")

WRF_PLACEMENT = {
    "parent_step_order": (
        "frame/module_integrate.F:393 solve_interface(parent) -> :416 "
        "med_nest_force -> :430 integrate(nest) -> :443 med_nest_feedback"
    ),
    "boundary_tables": (
        "share/interp_fcn.F:2500 rdt=1.D0/cdt; :2583-2584 bdy_txs=rdt*(psca-nfld), "
        "bdy_xs=nfld; share/mediation_force_domain.F:213 dtbc=0"
    ),
    "time_interpolation": (
        "linear across the parent step: target = bdy_xs + dtbc*bdy_txs with "
        "dtbc advanced by dt at the start of every child step "
        "(dyn_em/solve_em.F:372)"
    ),
    "relaxation_zone": (
        "once per child large step at rk_step==1 into the held tendencies "
        "(dyn_em/solve_em.F:933-957 relax_bdy_dry; dyn_em/module_em.F:1047-1080 "
        "rk_addtend_dry folds *_save into *_tendf; solve_em.F:2258-2276 "
        "relax_bdy_scalar); GM1 arrays held.uten/vten/wten, held.thten1, "
        "held.qten, arrays.ppten after finalize_held_tendencies"
    ),
    "specified_zone_acoustic": (
        "inside every small step after the advance: u,v after advance_uv "
        "(dyn_em/solve_em.F:1275,1341,1351), t,mu after advance_mu_t "
        "(:1388,1457,1467), ph,w after advance_w (:1495,1582-1604); spec_bdyupdate "
        "share/module_bc.F:1955-2064; GM1 u3d,v3d after the pressure-gradient "
        "advance, w3d,pp3d after the implicit column step"
    ),
    "specified_zone_scalars": (
        "spec_bdy_scalar (dyn_em/solve_em.F:2277) then rk_update_scalar (:2305): "
        "q = q_old + dt_rk*tend on the specified rows; GM1 th3d and q3d after "
        "scalar_moist, from tha/qa"
    ),
    "specified_zone_held_diabatic": (
        "CM1 eqtset=2 adds the previous step's microphysics rates to the held "
        "tendencies (solve1.F:1748-1751: ppten+=qpten, thten1+=qtten, "
        "qten(nqv)+=qvten, qten(nqc)+=qcten) and takes them back out of the "
        "final-stage state before microphysics (solve2.F:1561-1576).  The "
        "specified rows never received them -- their stage value is WRF's "
        "q_old + dt_rk*bdy_tend -- so after the final stage GM1 adds dt*rate to "
        "the specified rows of pp3d, th3d, qv and qc, and the removal returns "
        "exactly WRF's pre-physics value there instead of subtracting a rate "
        "that was never added"
    ),
    "specified_zone_final": (
        "end of every step, after microphysics: WRF v4.6.1 dyn_em/solve_em.F:4531-4716 calls "
        "spec_bdy_final (share/module_bc.F:2066-2216) on u, v, t, ph, mu and, for a nested "
        "domain, w and every moist, tracer and scalar field: the specified rows are set to "
        "bdy + dtbc*bdy_tend.  GM1 sets the specified rows of u3d, v3d, w3d, pp3d, th3d and "
        "every q3d species to value + dtbc*tend before solve3 copies them into the "
        "time-level-n arrays (gm1_nest_spec_apply modes 2 and 3, spec_bdyupdate's loop "
        "ownership); a species is held at zero or above, after pdefq"
    ),
    "davies_weights": "dyn_em/module_bc_em.F:1297-1341 lbc_fcx_gcx nested branch",
    "choice_not_transcription": (
        "WRF relaxes mu through mu_tend, which rk_tendency rewrites each stage, so "
        "its mass relaxation acts in stage 1 only; GM1 relaxes pp3d through the "
        "held ppten CM1 builds once per step, so it acts in all three stages like "
        "WRF's u, v, w, t and ph relaxation"
    ),
    "dry_mass_correction": (
        "not run in the child: a WRF nest has no mass fixer and takes its mass "
        "through the specified zone; CM1 apmasscon stays on in the parent"
    ),
    "not_forced": "tke3d (WRF's tke advance, solve_em.F:2378-2416, has no boundary call)",
}


@dataclass(frozen=True, slots=True)
class _Carrier:
    name: str
    stagger: str
    species: int | None = None


def _carriers(numq: int) -> tuple[_Carrier, ...]:
    return (
        _Carrier("u", "u"),
        _Carrier("v", "v"),
        _Carrier("w", "w"),
        _Carrier("pp", "mass"),
        _Carrier("th", "mass"),
        *(_Carrier(f"q{s}", "mass", s) for s in range(numq)),
    )


def _grid_lattice(dynamics: Any, stagger: str) -> tuple[int, int, int]:
    g = dynamics.geometry
    return lattice_shape(stagger, g.nz, g.ny, g.nx)


def _current(dynamics: Any, carrier: _Carrier) -> tuple[Any, tuple[int, int, int]]:
    """The time-level-``n+1`` owner (``*3d``) and the lattice's origin in it."""

    if carrier.name in ("u", "v", "w"):
        return dynamics.arrays[f"{carrier.name}3d"], (1, 3, 3)
    if carrier.name == "pp":
        return dynamics.fields["pp3d"], (0, 0, 0)
    if carrier.name == "th":
        return dynamics.fields["th3d"], (0, 0, 0)
    return dynamics.fields["q3d"][carrier.species], (0, 0, 0)


def _baseline(dynamics: Any, carrier: _Carrier) -> tuple[Any, tuple[int, int, int]]:
    """The time-level-``n`` owner (``*a``) and the lattice's origin in it."""

    if carrier.name in ("u", "v", "w"):
        return dynamics.arrays[f"{carrier.name}a"], (1, 3, 3)
    if carrier.name == "pp":
        return dynamics.fields["ppi"], (0, 0, 0)
    if carrier.name == "th":
        return dynamics.fields["tha"], (0, 0, 0)
    return dynamics.fields["qa"][carrier.species], (0, 0, 0)


class NestVerticalCoupling:
    """The vertical step between two different decks, on the device.

    Parent -> child (tables and the initial condition): CM1's
    ``interp_on_restart`` column rule fills a parent-resolution window on the
    child's levels (``gm1_nest_vinterp``), and the unchanged
    ``gm1_nest_capture`` reads that window -- WRF's intermediate-grid order,
    ``force_domain_em_part2.F:126,141-147``.  Child -> parent (feedback):
    ``gm1_nest_restrict_vertical`` with rho0/rf0-and-overlap weights, GM1's
    own choice because WRF refuses feedback with vertical nesting
    (``module_check_a_mundo.F:2711-2722``).  See :mod:`cm1core.nest_vertical`.
    """

    STAGGERS = ("u", "v", "w", "mass")

    def __init__(self, *, cupy: Any, stream: Any, refinement: Any, geometry: Any,
                 child: Any) -> None:
        import numpy

        from .nest_vertical import (
            VerticalWindow,
            compile_nest_vertical_module,
            feedback_weights,
            upload_vertical_map,
            upload_weights,
        )

        self.cp, self.stream, self.refinement = cupy, stream, refinement
        self.module = compile_nest_vertical_module(cupy)
        rho0 = cupy.asnumpy(child.metrics["rho0_3d"])
        rf0 = cupy.asnumpy(child.metrics["rf0"])
        for name, base in (("rho0_3d", rho0), ("rf0", rf0)):
            if not bool(numpy.all(base == base[:, :1, :1])):
                raise RuntimeError(
                    f"the child's {name} is not horizontally uniform, so one vertical "
                    "feedback weight per level would weight some columns by another "
                    "column's air"
                )
        nzc = int(refinement.child.nk)
        with stream:
            self.maps = {
                "mass": upload_vertical_map(cupy, refinement.mass),
                "w": upload_vertical_map(cupy, refinement.face),
            }
            self.windows = {s: VerticalWindow.for_stagger(geometry, s) for s in self.STAGGERS}
            self.scratch = {
                s: cupy.empty(
                    (nzc + (1 if s == "w" else 0), self.windows[s].nj, self.windows[s].ni),
                    dtype=cupy.float32,
                )
                for s in self.STAGGERS
            }
            self.weights = {
                "mass": upload_weights(cupy, feedback_weights(
                    refinement, stagger="mass", child_density=rho0[:, 0, 0])),
                "w": upload_weights(cupy, feedback_weights(
                    refinement, stagger="w", child_density=rf0[:, 0, 0])),
            }

    def window_bytes(self) -> int:
        return sum(int(a.nbytes) for a in self.scratch.values())

    def relocate(self, geometry: Any) -> None:
        """Re-place every stagger's window over a moved nest (:mod:`cm1core.nest_move`).

        A window's extent depends only on the span (``VerticalWindow.for_stagger``),
        so the scratch keeps its shape and only its parent origin moves with the
        nest; the tables, the move's exposed-strip fill and the feedback all read
        the relocated window.
        """

        from .nest_vertical import VerticalWindow

        self.windows = {s: VerticalWindow.for_stagger(geometry, s) for s in self.STAGGERS}

    def source(self, owner: Any, origin: tuple[int, int, int],
               lattice: tuple[int, int, int], stagger: str,
               ) -> tuple[Any, tuple[int, int, int], tuple[int, int, int], int, int]:
        """Fill the stagger's window from ``owner``; return it as the capture's parent."""

        from .nest_vertical import launch_vinterp

        kind = "w" if stagger == "w" else "mass"
        out = self.scratch[stagger]
        launch_vinterp(
            self.cp, self.module, parent=owner, parent_origin=origin, parent_lattice=lattice,
            window=self.windows[stagger], device_map=self.maps[kind], out=out,
            stream=self.stream,
        )
        return out, (0, 0, 0), tuple(int(v) for v in out.shape), 1, 1

    def restrict(self, *, child: Any, child_origin: tuple[int, int, int],
                 child_lattice: tuple[int, int, int], parent: Any,
                 parent_origin: tuple[int, int, int], parent_lattice: tuple[int, int, int],
                 geometry: Any, stagger: str) -> None:
        from .nest_vertical import launch_restrict_vertical

        launch_restrict_vertical(
            self.cp, self.module, child=child, child_origin=child_origin,
            child_lattice=child_lattice, parent=parent, parent_origin=parent_origin,
            parent_lattice=parent_lattice,
            device_weights=self.weights["w" if stagger == "w" else "mass"],
            span_x=geometry.parent_span_x, span_y=geometry.parent_span_y,
            ratio=geometry.refinement_ratio, relax_zone=geometry.relax_zone,
            parent_i0=geometry.parent_i0, parent_j0=geometry.parent_j0,
            stagger=stagger, stream=self.stream,
        )


class NestLateralForcing:
    """The child's Davies specified and relaxation zones, on the device."""

    def __init__(self, *, cupy: Any, stream: Any, geometry: Any, child: Any,
                 parent_dt_s: float, vertical: "NestVerticalCoupling | None" = None) -> None:
        import numpy

        self.cp, self.stream, self.geometry = cupy, stream, geometry
        self.vertical = vertical
        self.parent_dt_s = float(parent_dt_s)
        self.child_dt_s = float(child.config.dt_s)
        self.spec_zone = int(geometry.spec_zone)
        self.relax_zone = int(geometry.relax_zone)
        self.width = self.spec_zone + self.relax_zone
        self.apply_relax = self.relax_zone >= 2
        self.carriers = _carriers(int(child.numq))
        #: Every carrier the child holds; ``carriers`` are the ones this forcing's tables drive
        #: (a subclass may drive fewer, e.g. the chain's vapour-only lane).
        self.all_carriers = self.carriers
        self.module = compile_nest_live_module(cupy)
        self.relax_module = compile_nest_relaxation_module(cupy)
        fcx, gcx = davies_weights(
            spec_zone=self.spec_zone, relax_zone=max(self.relax_zone, 2),
            dt_s=self.child_dt_s,
        )
        with stream:
            self.fcx = cupy.asarray(numpy.asarray(fcx, dtype=numpy.float32))
            self.gcx = cupy.asarray(numpy.asarray(gcx, dtype=numpy.float32))
            self.tables: dict[str, dict[str, tuple[Any, Any]]] = {}
            self.snapshots: dict[str, dict[str, Any]] = {}
            for carrier in self.carriers:
                lattice = _grid_lattice(child, carrier.stagger)
                self.tables[carrier.name] = {
                    side: (
                        cupy.zeros(side_shape(side, lattice, self.width), dtype=cupy.float32),
                        cupy.zeros(side_shape(side, lattice, self.width), dtype=cupy.float32),
                    )
                    for side in SIDES
                }
                if carrier.name in ("u", "v", "w", "pp"):
                    self.snapshots[carrier.name] = {
                        side: cupy.zeros(side_shape(side, lattice, self.spec_zone), dtype=cupy.float32)
                        for side in SIDES
                    }
        self._dtbc = numpy.float32(0.0)
        self._dtc32 = numpy.float32(self.child_dt_s)
        self.refreshes = 0
        self.held_relaxations = 0
        self.acoustic_small_steps = 0
        self.scalar_specifications = 0
        self.final_specifications = 0
        self.tables_ready = False

    # -- tables -------------------------------------------------------------
    @property
    def dtbc(self) -> float:
        return float(self._dtbc)

    def table_bytes(self) -> int:
        total = 0
        for sides in self.tables.values():
            for value, tend in sides.values():
                total += int(value.nbytes) + int(tend.nbytes)
        for sides in self.snapshots.values():
            for snap in sides.values():
                total += int(snap.nbytes)
        if self.vertical is not None:
            total += self.vertical.window_bytes()
        return total + int(self.fcx.nbytes) + int(self.gcx.nbytes)

    def capture_source(self, dynamics: Any, owner: Any, origin: tuple[int, int, int],
                       stagger: str) -> tuple[Any, tuple[int, int, int], tuple[int, int, int], int, int]:
        """The parent array the capture reads: the parent itself, or its vertical window."""

        lattice = _grid_lattice(dynamics, stagger)
        if self.vertical is None:
            return owner, origin, lattice, self.geometry.parent_i0, self.geometry.parent_j0
        return self.vertical.source(owner, origin, lattice, stagger)

    def refresh(self, parent: Any, child: Any) -> None:
        """``med_nest_force``: tables from the parent's new state, ``dtbc = 0``."""

        import numpy

        g = self.geometry
        with self.stream:
            for carrier in self.carriers:
                parent_owner, parent_origin = _current(parent, carrier)
                source, source_origin, source_lattice, p_i0, p_j0 = self.capture_source(
                    parent, parent_owner, parent_origin, carrier.stagger
                )
                child_owner, child_origin = _current(child, carrier)
                for side in SIDES:
                    value, tend = self.tables[carrier.name][side]
                    launch_capture(
                        self.cp, self.module,
                        parent=source, parent_origin=source_origin,
                        parent_lattice=source_lattice,
                        child=child_owner, child_origin=child_origin,
                        child_lattice=_grid_lattice(child, carrier.stagger),
                        out_value=value, out_tend=tend, side=side, width=self.width,
                        ratio=g.refinement_ratio, parent_i0=p_i0,
                        parent_j0=p_j0, stagger=carrier.stagger,
                        parent_dt_s=self.parent_dt_s, stream=self.stream,
                    )
        self._dtbc = numpy.float32(0.0)
        self.refreshes += 1
        self.tables_ready = True

    def advance_clock(self) -> float:
        """``dtbc = dtbc + dt`` at the start of one child step (``solve_em.F:372``)."""

        import numpy

        self._dtbc = numpy.float32(self._dtbc + self._dtc32)
        return float(self._dtbc)

    def _sides(self, name: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for side in SIDES:
            value, tend = self.tables[name][side]
            out[side] = value
            out[f"{side}_t"] = tend
        return out

    # -- relaxation ---------------------------------------------------------
    def relax_held(self, dynamics: Any) -> None:
        """Davies terms added to the held tendencies from the time-``n`` state."""

        if not self.tables_ready:
            raise RuntimeError(
                "the child is being stepped before its boundary tables were built; "
                "it would be relaxed toward zeros"
            )
        cp, g = self.cp, dynamics.geometry
        with self.stream:
            for carrier in self.carriers:
                lattice = _grid_lattice(dynamics, carrier.stagger)
                owner, origin = _baseline(dynamics, carrier)
                k0, j0, i0 = origin
                field = cp.ascontiguousarray(
                    owner[k0:k0 + lattice[0], j0:j0 + lattice[1], i0:i0 + lattice[2]]
                )
                if carrier.name in ("u", "v", "w"):
                    tend = dynamics.held[f"{carrier.name}ten"]
                    writeback = None
                elif carrier.name == "pp":
                    view = dynamics.arrays["ppten"][1:1 + g.nz, 1:1 + g.ny, 1:1 + g.nx]
                    tend = cp.ascontiguousarray(view)
                    writeback = view
                elif carrier.name == "th":
                    tend = dynamics.held["thten1"]
                    writeback = None
                else:
                    tend = dynamics.held["qten"][carrier.species]
                    writeback = None
                launch_nest_specified_relaxation(
                    cp, field=field, field_tend=tend, sides=self._sides(carrier.name),
                    fcx=self.fcx, gcx=self.gcx, dtbc=self.dtbc, width=self.width,
                    spec_zone=self.spec_zone, relax_zone=self.relax_zone,
                    nz=lattice[0], ny=lattice[1], nx=lattice[2],
                    apply_relax=self.apply_relax, module=self.relax_module,
                )
                if writeback is not None:
                    cp.copyto(writeback, tend)
        cuda_graphs.bookkeeping(self._count_held_relaxation, 0)

    def _count_held_relaxation(self, _step: int) -> None:
        self.held_relaxations += 1

    def _count_acoustic_small_step(self, _step: int) -> None:
        self.acoustic_small_steps += 1

    def _count_scalar_specification(self, _step: int) -> None:
        self.scalar_specifications += 1

    def _count_final_specification(self, _step: int) -> None:
        self.final_specifications += 1

    def graph_key(self) -> tuple[str, int]:
        """What a captured child step freezes of this forcing: ``dtbc``'s bits.

        ``relax_held`` hands the kernels ``dtbc``, the time since the parent's
        tables were refreshed, which differs between the substeps of one parent
        step and repeats for the same substep of the next.
        """

        import numpy

        return ("nest-dtbc", int(numpy.float32(self._dtbc).view(numpy.uint32)))

    # -- specified zone -----------------------------------------------------
    def acoustic_hook(self, dynamics: Any) -> "_AcousticSpecifiedZone":
        return _AcousticSpecifiedZone(self, dynamics)

    def specify_scalars(self, dynamics: Any, nrk: int) -> None:
        from .cuda_sound3_column import rk_stage_dt_f32

        dt_rk = rk_stage_dt_f32(float(dynamics.config.dt_s), nrk)
        with self.stream:
            for carrier in self.carriers:
                if carrier.name in ("u", "v", "w", "pp"):
                    continue
                field, origin = _current(dynamics, carrier)
                base, _ = _baseline(dynamics, carrier)
                lattice = _grid_lattice(dynamics, carrier.stagger)
                for side in SIDES:
                    launch_spec_apply(
                        self.cp, self.module, field=field, origin=origin,
                        lattice=lattice, tend=self.tables[carrier.name][side][1],
                        side=side, spec_zone=self.spec_zone, width=self.width,
                        dt=dt_rk, base=base, stream=self.stream,
                    )
            self.after_scalar_stage(dynamics, nrk)
            if nrk == 3:
                self.restore_held_diabatic(dynamics)
        cuda_graphs.bookkeeping(self._count_scalar_specification, 0)

    def after_scalar_stage(self, dynamics: Any, nrk: int) -> None:
        """A lane's own specified-row rule for carriers the tables do not drive (none in a nest)."""

    #: Which fields WRF's ``spec_bdy_final`` sets, for this lane.
    SPEC_BDY_FINAL_LANE = "nested"

    def spec_bdy_final_carriers(self) -> tuple[_Carrier, ...]:
        """A nested domain: every field (``solve_em.F:4540-4713`` with ``config_flags%nested``)."""

        return tuple(self.carriers)

    def finalize_specified(self, dynamics: Any) -> None:
        """WRF ``spec_bdy_final``: the specified rows at ``value + dtbc*tend`` at the end of the step.

        ``dtbc`` was advanced at the start of this step (``solve_em.F:372``), so the target is the
        boundary state at the time the step ends.  Single precision, the product then the sum
        (``gm1_nest_spec_apply`` modes 2 and 3); the loop ownership is ``spec_bdyupdate``'s, the
        ``y`` sides owning the corners.  A species is held at zero or above: this runs after
        ``pdefq`` and ``value + dtbc*tend`` can round a few 1e-12 below zero where both ends hold
        none (GM1's choice, recorded).
        """

        if not self.tables_ready:
            raise RuntimeError(
                "the specified rows are being reset before the boundary tables were built; "
                "they would be set to zeros"
            )
        with self.stream:
            for carrier in self.spec_bdy_final_carriers():
                field, origin = _current(dynamics, carrier)
                lattice = _grid_lattice(dynamics, carrier.stagger)
                for side in SIDES:
                    value, tend = self.tables[carrier.name][side]
                    launch_spec_apply(
                        self.cp, self.module, field=field, origin=origin, lattice=lattice,
                        tend=tend, side=side, spec_zone=self.spec_zone, width=self.width,
                        dt=self.dtbc, value=value, nonnegative=carrier.species is not None,
                        stream=self.stream,
                    )
        cuda_graphs.bookkeeping(self._count_final_specification, 0)

    #: The (carrier, rate) pairs of CM1's eqtset=2 held diabatic estimate:
    #: solve1.F:1748-1751 adds each rate to the carrier's held tendency and
    #: solve2.F:1561-1576 subtracts ``dt*rate`` from the carrier after the
    #: final RK stage.
    HELD_DIABATIC: tuple[tuple[str, str], ...] = (
        ("pp", "qpten"), ("th", "qtten"), ("q0", "qvten"), ("q1", "qcten"),
    )

    def specified_bands(self, ny: int, nx: int) -> tuple[tuple[slice, slice], ...]:
        """The specified rows of a mass lattice as four disjoint (j, i) slices."""

        sz = self.spec_zone
        return (
            (slice(0, ny), slice(0, sz)),                  # west, every row
            (slice(0, ny), slice(nx - sz, nx)),            # east, every row
            (slice(0, sz), slice(sz, nx - sz)),            # south, corners excluded
            (slice(ny - sz, ny), slice(sz, nx - sz)),      # north, corners excluded
        )

    def restore_held_diabatic(self, dynamics: Any) -> None:
        """Add ``dt*rate`` to the specified rows before the final-stage removal.

        The specified rows advance by the boundary tendency alone
        (``q = q_old + dt_rk*bdy_tend``, WRF's ``spec_bdy_scalar`` and
        ``spec_bdyupdate``); the held diabatic estimate that every other cell
        carries through its RK stages never reaches them.  CM1's removal
        (``solve2.F:1561-1576``) nevertheless subtracts ``dt*rate`` from every
        cell, so without this the specified rows lost the previous step's
        microphysics rate once per child step.  In a condensing or depositing
        edge column that loop fed itself (the removal raised qc and qv, the
        microphysics turned the excess into rain or ice, the new rate was
        larger) and built columns of 0.4-1 kg/kg of hydrometeor on the
        outflow rows of every nested run.  Adding the rate here makes the
        removal return exactly the WRF pre-physics value on those rows.
        """

        import numpy

        cp = self.cp
        g = dynamics.geometry
        dt = numpy.float32(dynamics.config.dt_s)
        fields = dynamics.fields
        bands = self.specified_bands(int(g.ny), int(g.nx))
        for name, rate_name in self.HELD_DIABATIC:
            carrier = next(c for c in getattr(self, "all_carriers", self.carriers) if c.name == name)
            field, origin = _current(dynamics, carrier)
            if origin != (0, 0, 0):
                raise RuntimeError(f"{name}: the held diabatic estimate needs a physical carrier")
            rate = fields[rate_name]
            if tuple(rate.shape) != tuple(field.shape):
                raise RuntimeError(
                    f"{rate_name} {tuple(rate.shape)} does not share {name}'s lattice "
                    f"{tuple(field.shape)}; the removal would take another cell's rate"
                )
            for js, is_ in bands:
                view = field[:, js, is_]
                view[...] = view + cp.float32(dt) * rate[:, js, is_]

    def record(self) -> dict[str, Any]:
        return {
            "spec_zone": self.spec_zone,
            "relax_zone": self.relax_zone,
            "table_width": self.width,
            "relaxation_applied": self.apply_relax,
            "davies_authority": DAVIES_AUTHORITY,
            "forced_carriers": [c.name for c in self.carriers],
            "table_refreshes": self.refreshes,
            "held_relaxations": self.held_relaxations,
            "acoustic_small_steps_specified": self.acoustic_small_steps,
            "scalar_stage_specifications": self.scalar_specifications,
            "spec_bdy_final": {
                "authority": WRF_PLACEMENT["specified_zone_final"],
                "lane": self.SPEC_BDY_FINAL_LANE,
                "carriers": [c.name for c in self.spec_bdy_final_carriers()],
                "steps": self.final_specifications,
            },
            "table_and_snapshot_bytes": self.table_bytes(),
            "kernel_sha256": NEST_LIVE_KERNEL_SHA256,
            "vertical_kernel_sha256": (
                None if self.vertical is None else _vertical_kernel_sha256()
            ),
        }


def _vertical_kernel_sha256() -> str:
    from .nest_vertical import NEST_VERTICAL_KERNEL_SHA256

    return NEST_VERTICAL_KERNEL_SHA256


class _AcousticSpecifiedZone:
    """The per-small-step hook the acoustic launchers call."""

    def __init__(self, forcing: NestLateralForcing, dynamics: Any) -> None:
        self.forcing, self.dynamics = forcing, dynamics

    def _owner(self, name: str) -> tuple[Any, tuple[int, int, int], tuple[int, int, int]]:
        d = self.dynamics
        if name == "pp":
            return d.arrays["pp3d"], (1, 1, 1), _grid_lattice(d, "mass")
        stagger = {"u": "u", "v": "v", "w": "w"}[name]
        return d.arrays[f"{name}3d"], (1, 3, 3), _grid_lattice(d, stagger)

    def begin(self, *, nrk: int, n: int, dts: float) -> None:
        f = self.forcing
        for name in ("u", "v", "w", "pp"):
            owner, origin, lattice = self._owner(name)
            for side in SIDES:
                launch_spec_gather(
                    f.cp, f.module, field=owner, origin=origin, lattice=lattice,
                    snap=f.snapshots[name][side], side=side, spec_zone=f.spec_zone,
                    stream=f.stream,
                )

    def _apply(self, names: tuple[str, ...], dts: float) -> None:
        f = self.forcing
        for name in names:
            owner, origin, lattice = self._owner(name)
            for side in SIDES:
                launch_spec_apply(
                    f.cp, f.module, field=owner, origin=origin, lattice=lattice,
                    tend=f.tables[name][side][1], side=side, spec_zone=f.spec_zone,
                    width=f.width, dt=dts, snap=f.snapshots[name][side],
                    stream=f.stream,
                )

    def after_momentum(self, *, nrk: int, n: int, dts: float) -> None:
        self._apply(("u", "v"), dts)

    def after_pressure(self, *, nrk: int, n: int, dts: float) -> None:
        self._apply(("w", "pp"), dts)
        cuda_graphs.bookkeeping(self.forcing._count_acoustic_small_step, 0)

    def record(self) -> dict[str, Any]:
        return {
            "authority": WRF_PLACEMENT["specified_zone_acoustic"],
            "spec_zone": self.forcing.spec_zone,
            "carriers": ["u3d", "v3d", "w3d", "pp3d"],
        }


# ---------------------------------------------------------------------------
# The child runtime and the driver.
# ---------------------------------------------------------------------------
def nest_child_parcel_layout(parent_config: Any, child_config: Any) -> Any:
    """``[lagrangian.nest_child]``'s attribute layout, or ``None`` without the table.

    ``param.F:6257-6282`` resolves the one ``[lagrangian]`` table against the
    namelist's own words, and :func:`child_runtime_config` makes the child's
    contract the parent's on another grid, so both grids share one resolution.
    A pair built around that would publish one track whose parcels were sampled
    under two layouts as they are handed between the grids, so it is refused.
    """

    lagrangian = getattr(parent_config, "lagrangian", None)
    if lagrangian is None or lagrangian.nest_child is None:
        return None
    from .lagrangian_parcels import parcel_switch_context

    context = parcel_switch_context(parent_config)
    if parcel_switch_context(child_config) != context:
        raise ValueError(
            "the live nest parent and child resolve the prcl_* switches differently "
            "(param.F:6257-6282 reads the moisture axis, closure and bbc); one parcel "
            "track cannot carry both layouts"
        )
    return lagrangian.attribute_layout(context)


def child_runtime_config(parent_config: Any, child_grid: Any, *, child_dt_s: float,
                         acoustic_substeps: int) -> Any:
    """The parent's own runtime contract on the child's grid and clock.

    Every physics, closure, boundary and output selector is the parent's, so a
    child can never run a different scheme from the parent that forces it.
    The parent's Lagrangian stage is the parent's own and is not duplicated
    onto the child.  Of the parent's ``output_*`` diagnostics the child keeps
    exactly the surface swaths (:func:`child_swath_selection`).
    """

    # The child's own lateral halos are CM1's open radiative family whatever
    # the parent's are: its boundary rows are specified from the parent, a
    # periodic or rigid child would wrap or wall a window that is neither, and
    # nudgeobc/roflux would pull those rows toward the base state or rescale
    # the parent's own outflow.
    return replace(
        parent_config, grid=child_grid, dt_s=float(child_dt_s),
        acoustic_substeps=int(acoustic_substeps), lagrangian=None,
        diagnostics=child_swath_selection(getattr(parent_config, "diagnostics", None)),
        wbc=2, ebc=2, sbc=2, nbc=2, roflux=0, nudgeobc=0,
    )


def child_swath_selection(parent_selection: Any) -> Any:
    """The child's ``output_*`` selection: the parent's surface swaths only.

    ``solve3.F:855-1136`` folds ``sws``..``shs`` on every large step of the
    grid that carries them, and a child is a full CM1 grid on its own step, so
    each swath the parent keeps the child keeps too, folded on every child
    step at the child's own resolution -- the swath of the refined vortex,
    which the parent's block means cannot hold.  The instantaneous rotation
    fields (``output_uh``/``output_vort``/``output_pv``) and the GM1 tornado
    report are the parent's: the report states one run's headline numbers
    and is not duplicated per grid.  A parent that keeps no swath gives an
    all-off selection, which builds nothing on the child.

    The child grid is checked by ``RuntimeConfig``'s own
    ``validate_tornado_diagnostic_selection`` when the child contract is
    built, so a swath the child grid cannot carry (``output_sus`` under a
    child top below 5 km) is refused by name there, not published wrong.
    """

    from .tornado_diagnostics import (
        DEFAULT_TORNADO_DIAGNOSTICS,
        SWATH_SWITCH_BY_NAME,
        TornadoDiagnosticSelection,
    )

    if parent_selection is None:
        return DEFAULT_TORNADO_DIAGNOSTICS
    if not isinstance(parent_selection, TornadoDiagnosticSelection):
        raise TypeError("the parent's diagnostics must be a TornadoDiagnosticSelection")
    return TornadoDiagnosticSelection(**{
        switch: int(getattr(parent_selection, switch))
        for switch in SWATH_SWITCH_BY_NAME.values()
    })


def make_child_clock_class() -> type:
    """The child's clock: its own steps, the parent's time and ground position.

    CM1's frame integral (``cm1.F:3512-3513``) is the domain's ground position,
    and a child in the parent's frame is where the parent is.  The parent's
    own ``RuntimeClock`` keeps that integral exact and refuses the moment it is
    not (``gm1_runtime.RuntimeClock._require_exact_domain_location``).  The
    child's step is the parent's divided by the time ratio, and that quotient
    times a frame speed is often not a short binary fraction (0.1 s times
    3 m/s), so a child accumulating its own sum would drift and stop the run.
    The child therefore does not own the integral: between parent steps it is
    only required to stay finite, and at every parent-step boundary the driver
    sets it -- and the model time the child publishes -- to the parent's.
    """

    from .gm1_runtime import RuntimeClock

    @dataclass(slots=True)
    class NestChildClock(RuntimeClock):
        parent_clock: Any = None
        time_ratio: int = 1

        def _require_exact_domain_location(self, stage: str) -> None:
            for value in (self.domainlocx_m, self.domainlocy_m):
                if not math.isfinite(value):
                    raise ValueError(f"{stage} child domain location is not finite")

        @property
        def model_seconds(self) -> float:
            parent = self.parent_clock
            if parent is not None and self.completed_steps == parent.completed_steps * self.time_ratio:
                return parent.model_seconds
            return self.completed_steps * self.config.dt_s

    return NestChildClock


def make_child_runtime_class(base: type) -> type:
    """The runtime class whose schedule skips CM1's dry-mass correction."""

    from .gm1_runtime import StepPhase

    class NestChildRuntime(base):  # type: ignore[misc,valid-type]
        nest_skipped_phases = (StepPhase.MASS_BEGIN, StepPhase.MASS_FINISH)
        #: The nest's child parcels (cm1core.nest_parcels), set by NestDriver.
        nest_parcels: Any = None
        #: The vortex-following sub-volume stream (cm1core.nest_child_window),
        #: set by NestDriver.  It is captured at the same moment in the same
        #: step as the parcels' advecting velocity, because it *is* that
        #: velocity: solve2.F:1290-1301's nrk==nrkmax average.
        nest_child_window: Any = None

        def execute(self, operation: Any, *, step: int) -> None:
            if operation.phase in self.nest_skipped_phases:
                return
            if operation.phase == StepPhase.SOLVE3:
                forcing = getattr(self.dynamics, "nest_forcing", None)
                if forcing is not None:
                    # WRF's spec_bdy_final: after microphysics (and CM1's rate
                    # recapture), before solve3 copies the new state into the
                    # time-level-n arrays.
                    forcing.finalize_specified(self.dynamics)
            super().execute(operation, step=step)

        def _capture_parcel_velocity(self) -> None:
            # solve2.F:1290-1301 in the child, at its own nrk == nrkmax.
            super()._capture_parcel_velocity()
            if self.nest_parcels is not None:
                from .nest_parcels import parcel_roughness

                # parcel.F:470-515 on a semi-slip child reads the child's zntmp.
                roughness = parcel_roughness(self)
                extra = {} if roughness is None else {"roughness": roughness}
                self.nest_parcels.capture_child(self.dynamics.arrays, self.config, **extra)
            if self.nest_child_window is not None:
                from .nest_parcels import parcel_roughness

                # The clock has not advanced yet (gm1_runtime.RuntimeClock.step
                # increments after the phases), so this capture completes step
                # completed_steps + 1 and holds the wind that carried the model
                # from model_seconds to model_seconds + dt_s.  That end time is
                # the time a backward integrator steps *from* when it reverses
                # this record.
                dynamics = self.dynamics
                carriers = {name: dynamics.arrays[name] for name in ("rru", "rrv", "rrw")}
                carriers.update({name: dynamics.fields[name]
                                 for name in ("rho", "prs", "tha", "qa")})
                # The base state as its compact column: the expanded th0/qv0
                # fields are released between phases (_drop_expanded_base).
                carriers.update({f"{name}_profile": dynamics.base_profiles[name]
                                 for name in ("th0", "qv0")})
                if self.nest_child_window.icor == 1:
                    # lspgrad=1's per-level words (solve1.F:590-612) are formed
                    # from the child's own frame-relative base winds.
                    carriers.update({f"{name}_profile": dynamics.base_profiles[name]
                                     for name in ("u0", "v0")})
                carriers.update({name: dynamics.held[name] for name in ("uten", "vten", "wten")})
                self.nest_child_window.capture(
                    carriers,
                    child_step=int(self.clock.completed_steps) + 1,
                    model_time_s=float(self.clock.model_seconds) + float(self.config.dt_s),
                    roughness=parcel_roughness(self),
                )

        def graph_key(self, step: int) -> Any:
            key = super().graph_key(step)
            if isinstance(key, str):
                return key
            if self.nest_parcels is not None:
                return "nest-child-parcel-velocity-capture"
            window = self.nest_child_window
            if window is not None:
                # _capture_parcel_velocity's own arguments for this step.
                state = window.graph_state(
                    child_step=int(step),
                    model_time_s=float(self.clock.model_seconds) + float(self.config.dt_s),
                )
                if state == "due":
                    return "nest-child-window-record"
                key = key + (("nest-child-window", state),)
            forcing = getattr(self.dynamics, "nest_forcing", None)
            if forcing is not None:
                key = key + (forcing.graph_key(),)
            return key

        def _advance_lagrangian(self) -> None:
            # solve3.F:768-769 in the child: once per completed child step.
            super()._advance_lagrangian()
            if self.nest_parcels is not None:
                self.nest_parcels.advance_child()

    return NestChildRuntime


class NestDriver:
    """Parent step -> tables -> child steps -> feedback, both domains resident."""

    def __init__(self, *, parent: Any, child: Any, geometry: Any, cupy: Any,
                 stream: Any, follow: Any = None, criterion: Any = None,
                 restoring: bool = False) -> None:
        # ``restoring``: the child will be read from its own restart
        # (cm1core.nest_restart.restore_nest_child) at ``geometry``, the
        # restored placement, instead of captured from the parent; its parcels
        # are then left unseeded for the restart to fill.
        self.parent, self.child, self.geometry = parent, child, geometry
        for role, runtime in (("parent", parent), ("child", child)):
            if getattr(runtime, "budget", None) is not None:
                # config.py refuses this by name at the door
                # (CM1-OUTPUT-BUDGET); a caller that built the pair around it
                # would publish relaxation and feedback under CM1 budget names.
                raise RuntimeError(
                    f"the {role} runtime carries a budget stage, but a live nest's "
                    "Davies relaxation, specified zone and feedback have no CM1 "
                    "budget term"
                )
        self.cp, self.stream = cupy, stream
        self.q = int(geometry.time_refinement_ratio)
        pdt, cdt = float(parent.config.dt_s), float(child.config.dt_s)
        if not math.isclose(pdt / self.q, cdt, rel_tol=0.0, abs_tol=0.0) and pdt / self.q != cdt:
            raise ValueError("the child's step is not the parent's divided by the time ratio")
        # Two schemes can never share one relaxation or feedback table: the
        # species are exchanged row by row, so a parent and child on different
        # components would relax one scheme's rows toward another's (or a
        # count toward a mass) even where the axis lengths happened to agree.
        parent_scheme = getattr(parent.config, "microphysics", None)
        child_scheme = getattr(child.config, "microphysics", None)
        if parent_scheme != child_scheme:
            raise ValueError(
                f"the live nest parent runs {parent_scheme!r} and the child "
                f"{child_scheme!r}; parent and child must run the same "
                "physics.component, because the lateral relaxation and the "
                "two-way feedback exchange the q3d axis row by row"
            )
        if parent.dynamics.numq != child.dynamics.numq:
            raise ValueError("parent and child carry different moisture axes")
        from .nest_vertical import levels_of_grid, plan_vertical_refinement

        # config.py refused an inadmissible child deck by name; this raises
        # only if a caller built the pair around that check.
        self.vertical = plan_vertical_refinement(
            levels_of_grid(parent.config.grid), levels_of_grid(child.config.grid)
        )
        coupling = None
        if not self.vertical.identical:
            coupling = NestVerticalCoupling(
                cupy=cupy, stream=stream, refinement=self.vertical, geometry=geometry,
                child=child.dynamics,
            )
        self.forcing = NestLateralForcing(
            cupy=cupy, stream=stream, geometry=geometry, child=child.dynamics,
            parent_dt_s=pdt, vertical=coupling,
        )
        child.dynamics.nest_forcing = self.forcing
        self.module = self.forcing.module
        self._align_child_clock()
        self.follow = follow
        self.parent_steps = 0
        self.child_steps = 0
        self.feedback_writes = 0
        self.initialized_from_parent = False
        self.restored: dict[str, Any] | None = None
        self.child_frames: list[dict[str, Any]] = []
        #: The child's swath planes a nested restart carried
        #: (cm1core.nest_restart.restore_nest_child), held until
        #: :meth:`open_child_swaths` opens the child's swaths with the run's
        #: step count and copies them in.
        self.restored_swaths: dict[str, Any] | None = None
        # A following nest (cm1core.nest_move): WRF's med_nest_move, called
        # after med_nest_feedback (frame/module_integrate.F:442-449).
        self.mover = None
        if follow is not None:
            from .nest_move import MovingNest

            self.mover = MovingNest.build(settings=follow, driver=self, cupy=cupy, stream=stream)
        # [lagrangian.nest_child]: parcels on the child's own lattice and deck
        # (cm1core.nest_parcels), stepped by the child and handed off here.
        self.parcels = None
        lagrangian = getattr(parent.config, "lagrangian", None)
        settings = None if lagrangian is None else lagrangian.nest_child
        if settings is not None:
            from .nest_parcels import NestParcelSeedExtentError, NestParcelTracker

            # Resolved outside the seed-extent refusal below: a pair that
            # resolves the prcl_* switches differently is its own ValueError,
            # naming param.F:6257-6282, not a seed box to move.
            layout = nest_child_parcel_layout(parent.config, child.config)
            try:
                self.parcels = NestParcelTracker.build(
                    settings=settings, geometry=geometry, layout=layout,
                    parent_config=parent.config, child_config=child.config, xp=cupy,
                    stream=stream, model_time_s=float(parent.clock.model_seconds),
                    parent_step=int(parent.clock.completed_steps), seed=not restoring,
                )
            except NestParcelSeedExtentError as error:
                from .errors import refuse

                refuse(
                    code="CM1-CARTESIAN-NEST-PARCELS-SEED-EXTENT",
                    gate="cartesian-runtime",
                    reason=(
                        f"{error}; at this model time the child stands where the seed box "
                        "does not, so the parcels would be sampled off the child's own rows"
                    ),
                    remedy="move lagrangian.nest_child's box inside the child's free interior",
                    option="lagrangian.nest_child.seed_x0_m", value=settings.record(),
                )
            child.nest_parcels = self.parcels
        # [lagrangian.nest_child_window]: the vortex-following sub-volume of
        # the child, cut every child step (cm1core.nest_child_window).  It is
        # built after the criterion below only in reading order: the stream
        # reads driver.criterion at each parent step, never at construction.
        self.window = None
        window_settings = None if lagrangian is None else lagrangian.nest_child_window
        if window_settings is not None:
            from .nest_child_window import NestChildWindowStream
            from .nest_parcels import child_deck as nest_child_deck

            self.window = NestChildWindowStream.build(
                settings=window_settings, geometry=geometry,
                child_deck=nest_child_deck(geometry, child.config.grid),
                child_config=child.config, parent_dt_s=pdt, xp=cupy, stream=stream,
            )
            child.nest_child_window = self.window
        # [nest.tornado_criterion]: the fixed tornado criterion on the live
        # child (cm1core.nest_tornado_criterion), sampled every parent step in
        # after_parent_step.  It reads the child's winds and writes no word.
        self.criterion = None
        if criterion is not None:
            from .nest_tornado_criterion import NestTornadoCriterion

            self.criterion = NestTornadoCriterion(
                settings=criterion, driver=self, xp=cupy, stream=stream,
            )

    def _align_child_clock(self) -> None:
        """Put the child on the parent's clock, fresh or restored.

        ``restart_read.F:419-420`` continues a restored parent's clock; a child
        built at that moment continues from the same model time, and the
        schedulers that refuse any step but the next one are told so.
        """

        parent, child = self.parent, self.child
        steps = int(parent.clock.completed_steps) * self.q
        child.clock = make_child_clock_class()(
            child.config, completed_steps=steps,
            domainlocx_m=float(parent.clock.domainlocx_m),
            domainlocy_m=float(parent.clock.domainlocy_m),
            parent_clock=parent.clock, time_ratio=self.q,
        )
        if steps:
            child.finish.completed_physics_step = steps
            child.finish.completed_solve3_step = steps
            child.closure._held_step = steps
            child.closure._pending_step = None
            child.closure._tke_stage = 3

    # -- the child's surface swaths --------------------------------------------
    def open_child_swaths(self, *, parent_total_steps: int) -> Any:
        """Open the child's ``solve3.F:855-1136`` swaths for a run of this length.

        Called by the parent's ``open_tornado_diagnostics``, so the child's
        swaths are sized and opened by every entry point that opens the
        parent's.  The child folds them itself at the end of each of its own
        large steps (``CudaCartesianRuntime.step``), ``q`` per parent step, so
        a vortex the child resolves between two parent steps is kept.

        No ground-fixed twin is kept (``ground=False``): the child's lattice
        moves inside the parent on every nest move as well as with the frame,
        and a canvas that follows both would have to cover the whole parent
        track at the child's spacing.  The frame-relative swath is shifted
        with the child on a move (:mod:`cm1core.nest_move`) and each child
        frame's placement document puts it on the ground.
        """

        child = self.child
        selection = getattr(child.config, "diagnostics", None)
        if selection is None or not selection.swath_names:
            if self.restored_swaths is not None:
                raise RuntimeError("restored child swaths exist for a child that keeps none")
            return None
        if getattr(child, "tornado", None) is not None:
            raise RuntimeError(
                "the nest child's swaths are already open for another step count; "
                "folding a second run's steps into them would report one run's "
                "swath under the other's clock"
            )
        from .cartesian_tornado_diagnostics import CartesianTornadoDiagnostics

        tornado = CartesianTornadoDiagnostics(
            child, total_steps=int(parent_total_steps) * self.q, ground=False,
        )
        if self.restored_swaths is not None:
            tornado.restore_swaths(self.restored_swaths)
            self.restored_swaths = None
        child.tornado = tornado
        return tornado

    # -- child initial condition --------------------------------------------
    def initialize_child(self) -> dict[str, Any]:
        """``med_nest_initial``: the whole child state from the parent's."""

        cp, g = self.cp, self.geometry
        p, c = self.parent.dynamics, self.child.dynamics
        carriers = list(_carriers(int(c.numq)))
        with self.stream:
            for carrier in carriers:
                lattice = _grid_lattice(c, carrier.stagger)
                owner, owner_origin = _current(p, carrier)
                source, source_origin, source_lattice, p_i0, p_j0 = self.forcing.capture_source(
                    p, owner, owner_origin, carrier.stagger
                )
                target = cp.empty(lattice, dtype=cp.float32)
                launch_capture(
                    cp, self.module, parent=source, parent_origin=source_origin,
                    parent_lattice=source_lattice, child=None,
                    child_origin=(0, 0, 0), child_lattice=lattice, out_value=target,
                    out_tend=None, side="full", width=self.forcing.width,
                    ratio=g.refinement_ratio, parent_i0=p_i0,
                    parent_j0=p_j0, stagger=carrier.stagger,
                    parent_dt_s=float(self.parent.config.dt_s), stream=self.stream,
                )
                for owner_of in (_current, _baseline):
                    owner, (k0, j0, i0) = owner_of(c, carrier)
                    cp.copyto(owner[k0:k0 + lattice[0], j0:j0 + lattice[1], i0:i0 + lattice[2]], target)
                if carrier.name == "pp":
                    for name in ("pp3d", "ppi"):
                        cp.copyto(c.arrays[name][1:1 + lattice[0], 1:1 + lattice[1], 1:1 + lattice[2]], target)
            lattice = _grid_lattice(c, "w")
            # WRF interpolates its nest's tke like every other state field; a
            # refined child takes it through the same sigmaf column rule as w.
            # (CM1's own restart-grid path does not read tkea, restart_read.F:1207,
            # and the one-way downscale tool follows that; the live nest keeps
            # the capture it has always had.)
            source, source_origin, source_lattice, p_i0, p_j0 = self.forcing.capture_source(
                p, p.tke["tke3d"], (0, 0, 0), "w"
            )
            target = cp.empty(lattice, dtype=cp.float32)
            launch_capture(
                cp, self.module, parent=source, parent_origin=source_origin,
                parent_lattice=source_lattice, child=None,
                child_origin=(0, 0, 0), child_lattice=lattice, out_value=target,
                out_tend=None, side="full", width=self.forcing.width,
                ratio=g.refinement_ratio, parent_i0=p_i0,
                parent_j0=p_j0, stagger="w",
                parent_dt_s=float(self.parent.config.dt_s), stream=self.stream,
            )
            cp.copyto(c.tke["tke3d"], target)
            cp.copyto(c.tke["tkea"], target)
            del target
        _refill_halos_and_thermodynamics(c)
        self.stream.synchronize()
        finish = self.child.finish
        mass = finish._initial_mass()
        finish.initial_mass1_f64 = mass
        c.mass1_f64 = mass
        self.initialized_from_parent = True
        return {
            "operator": "gm1_nest_capture over the whole child lattice",
            "carriers": [k.name for k in carriers] + ["tke"],
            "child_initial_dry_mass_kg": mass,
        }

    # -- one parent step ----------------------------------------------------
    def after_parent_step(self) -> None:
        if not self.initialized_from_parent and self.restored is None:
            raise RuntimeError(
                "the child was never initialized from the parent or restored from its restart"
            )
        self.forcing.refresh(self.parent.dynamics, self.child.dynamics)
        parent_step = int(self.parent.clock.completed_steps)
        frames = getattr(self.parent, "reflectivity_frame_steps", frozenset())
        for n in range(1, self.q + 1):
            self.forcing.advance_clock()
            if parent_step in frames and n == self.q:
                self.child.reflectivity_frame_steps = frozenset(
                    {int(self.child.clock.completed_steps) + 1}
                )
            self.child.step()
            self.child_steps += 1
        self.child.synchronize()
        # The child's ground position is the parent's (see make_child_clock_class).
        self.child.clock.domainlocx_m = float(self.parent.clock.domainlocx_m)
        self.child.clock.domainlocy_m = float(self.parent.clock.domainlocy_m)
        if self.child.clock.completed_steps != self.parent.clock.completed_steps * self.q:
            raise RuntimeError("the child's clock left the parent's")
        if self.criterion is not None:
            # Both domains at one model time, the child's words its own and on
            # the placement it was integrated on: before feedback writes the
            # parent and before a move shifts the child.
            self.criterion.after_parent_step(self)
        if self.geometry.feedback:
            self.feed_back()
        if self.mover is not None:
            self.mover.after_parent_step(self)
        if self.parcels is not None:
            # After feedback and any move: both domains at one model time, the
            # child at its new placement (cm1core.nest_parcels hand-off rule).
            self.parcels.after_parent_step(self)
        if self.window is not None:
            # Last: the criterion has sampled, the feedback is written and any
            # move has happened, so the centre and the lattice the next child
            # steps are cut on are the ones they will run on.
            self.window.after_parent_step(self)
        self.parent_steps += 1

    def feed_back(self) -> None:
        cp, g = self.cp, self.geometry
        p, c = self.parent.dynamics, self.child.dynamics
        with self.stream:
            for carrier in _carriers(int(c.numq)):
                child_owner, child_origin = _current(c, carrier)
                targets = [_current(p, carrier), _baseline(p, carrier)]
                if carrier.name == "pp":
                    targets += [(p.arrays["pp3d"], (1, 1, 1)), (p.arrays["ppi"], (1, 1, 1))]
                for owner, origin in targets:
                    if self.forcing.vertical is not None:
                        self.forcing.vertical.restrict(
                            child=child_owner, child_origin=child_origin,
                            child_lattice=_grid_lattice(c, carrier.stagger), parent=owner,
                            parent_origin=origin,
                            parent_lattice=_grid_lattice(p, carrier.stagger),
                            geometry=g, stagger=carrier.stagger,
                        )
                        continue
                    launch_restrict(
                        cp, self.module, child=child_owner, child_origin=child_origin,
                        child_lattice=_grid_lattice(c, carrier.stagger), parent=owner,
                        parent_origin=origin, parent_lattice=_grid_lattice(p, carrier.stagger),
                        span_x=g.parent_span_x, span_y=g.parent_span_y,
                        ratio=g.refinement_ratio, relax_zone=g.relax_zone,
                        parent_i0=g.parent_i0, parent_j0=g.parent_j0,
                        stagger=carrier.stagger, stream=self.stream,
                    )
        _refill_halos_and_thermodynamics(p, momentum=False)
        self.feedback_writes += 1

    # -- restart ------------------------------------------------------------
    def snapshot_child_restart(self, directory: Any, *, publisher: Any,
                               published_directory: Any, restart_number: int,
                               next_history_index: int, host: Any = None) -> tuple[Any, dict[str, Any]]:
        """The child's own restart (``med_restart_out`` for this child).

        See :mod:`cm1core.nest_restart`.  Returns the child's restart snapshot
        and the declaration the parent's receipt carries.
        """

        from .nest_restart import snapshot_nest_child_restart

        return snapshot_nest_child_restart(
            self, directory, publisher=publisher, published_directory=published_directory,
            restart_number=restart_number, next_history_index=next_history_index, host=host,
        )

    def restore_child(self, restored: Any, state: Any) -> dict[str, Any]:
        """``med_nest_initial`` on a restart: read the child instead of capturing it."""

        from .nest_restart import restore_nest_child

        if self.initialized_from_parent or self.restored is not None:
            raise RuntimeError(
                "the child already holds a state; restoring over it would discard it"
            )
        self.restored = restore_nest_child(self, restored, state)
        return self.restored

    # -- output -------------------------------------------------------------
    def publish_child_frame(self, root: Any, index: int, *, publisher: Any,
                            carrier_profile: str) -> dict[str, Any]:
        part = self.snapshot_child_frame(
            root, index, publisher=publisher, carrier_profile=carrier_profile,
        )
        return self.child._commit_inline([part])[0]

    def snapshot_child_frame(self, root: Any, index: int, *, publisher: Any,
                             carrier_profile: str, host: Any = None) -> Any:
        """Take the child's frame off the device; the part writes it and its placement.

        Everything that reads the child or the nest geometry -- the carriers,
        the clock, a following nest's current placement and a semi-slip
        child's ``znt`` -- is read here, at the same completed step as the
        parent frame it belongs to.  ``commit`` touches only host memory and
        the filesystem.
        """

        from pathlib import Path

        from .background_publication import PublicationPart
        from .cartesian_history import cartesian_history_profile
        from .nest_parcels import child_placement_document, parcel_roughness, write_child_placement

        child_root = Path(root) / "nest-child"
        child_root.mkdir(parents=True, exist_ok=True)
        tornado = getattr(self.child, "tornado", None)
        profile = cartesian_history_profile(
            self.child.config, carrier_profile=carrier_profile,
            # The child's own swaths (open_child_swaths); frame-relative only.
            tornado_carriers=() if tornado is None else tornado.carrier_table(),
        )
        frame = self.child._snapshot_frame(
            child_root, index, publisher=publisher, profile=profile, host=host,
            profiled=False,
        )
        snapshot = frame.detail
        # Where the child stood when this frame was written; a following
        # nest moves between frames (cm1core.nest_move).
        placement = {
            "parent_i0": int(self.geometry.parent_i0),
            "parent_j0": int(self.geometry.parent_j0),
            "origin_x_m": float(self.geometry.origin_x_m),
            "origin_y_m": float(self.geometry.origin_y_m),
        }
        # The placement and deck beside the frame, so an offline reader
        # (cm1core.nest_parcel_backward) can put its winds back on the ground
        # without the run receipt.
        document = child_placement_document(
            index=index, step=int(snapshot.step), model_time_s=float(snapshot.model_time_s),
            geometry=self.geometry, child_config=self.child.config,
            domainlocx_m=float(self.child.clock.domainlocx_m),
            domainlocy_m=float(self.child.clock.domainlocy_m),
        )
        roughness = None
        if int(self.child.config.bbc) == 3:
            # A semi-slip child's offline reader needs this frame's znt
            # (parcel.F:470-515), which the history frame does not carry.
            roughness = self.cp.asnumpy(parcel_roughness(self.child))

        def commit(cancel: Any) -> dict[str, Any]:
            # The parent's event times this under "nest_child_frame".
            receipt, location = frame.commit(cancel)
            row = {
                "index": index,
                "step": int(receipt["step"]),
                "model_time_s": float(receipt["model_time_s"]),
                "directory": location["published_directory"],
                **placement,
            }
            row["placement_path"] = str(write_child_placement(child_root, document, roughness))
            if len(self.child_frames) < 1024:
                self.child_frames.append(row)
            return row

        return PublicationPart(
            label=f"nest-child/{frame.label}", commit=commit, discard=frame.discard,
            snapshot_seconds=frame.snapshot_seconds, detail=snapshot,
        )

    def record(self) -> dict[str, Any]:
        g = self.geometry
        return {
            "schema": NEST_LIVE_SCHEMA,
            "status": "integrated-concurrently",
            "wrf_placement": dict(WRF_PLACEMENT),
            "feedback": "two-way" if g.feedback else "one-way",
            "feedback_operator": (
                None if not g.feedback else
                "gm1_nest_restrict: float64 raster-order mean of each r*r child "
                "block; blocks reaching the relaxation zone are not fed back; "
                "parent rho/prs/rr/rf rebuilt from the fed-back state"
                if self.vertical.identical else
                "gm1_nest_restrict_vertical: the r*r block sums of the child "
                "layers overlapping each parent layer, weighted by child rho0 "
                "(rf0 for w) times overlap and normalized per parent layer; w's "
                "ground and top faces copied; relaxation-zone blocks excluded; "
                "parent rho/prs/rr/rf rebuilt from the fed-back state"
            ),
            "vertical_refinement": self.vertical.record(),
            "parent_dt_s": float(self.parent.config.dt_s),
            "child_dt_s": float(self.child.config.dt_s),
            "child_steps_per_parent_step": self.q,
            "parent_steps_coupled": self.parent_steps,
            "child_steps": self.child_steps,
            "feedback_writes": self.feedback_writes,
            "child_initialized_from_parent": self.initialized_from_parent,
            **({} if self.restored is None else {"child_restored_from_restart": {
                key: self.restored[key] for key in (
                    "restart_directory", "step", "model_time_s", "artifact_sha256",
                    "coupling", "document_changes",
                )
            }}),
            "child_skips_phases": ["dry-mass-reduction", "dry-mass-correction"],
            "lateral_forcing": self.forcing.record(),
            "child_frames": list(self.child_frames),
            "child_grid": self.child.config.grid.record(),
            "placement": {"parent_i0": int(g.parent_i0), "parent_j0": int(g.parent_j0)},
            "storm_following": None if self.mover is None else self.mover.record(),
            **({} if self.parcels is None else {"child_parcels": self.parcels.record()}),
            **({} if getattr(self.child, "tornado", None) is None else {
                "child_swaths": self.child.tornado.record().get("swaths"),
            }),
        }


def _refill_halos_and_thermodynamics(dynamics: Any, *, momentum: bool = True) -> None:
    """Rebuild what a direct write into a domain's state leaves stale.

    The open-boundary halo fills are the ones ``momentum_boundaries`` and
    ``sync_padded_from_physical`` run; ``prs``/``rho``/``rr``/``rf`` are
    ``init3d.F``'s ``getset`` through the same startup kernel a fresh run uses.
    """

    from .cuda_cartesian_startup import launch_cuda_cartesian_startup

    d, cp = dynamics, dynamics.cupy
    g = d.geometry
    with d.stream:
        if momentum:
            kfwe, kfsn, rigid = d.open_compiled.kernels[5:8]
            a, m = d.arrays, d.metrics
            words = max(int(a[name].size) for name in ("u3d", "v3d", "w3d"))
            grid, block = ((words + 255) // 256, 1, 1), (256, 1, 1)
            common = (cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz), cp.uint32(3))
            kfwe(grid, block, (a["u3d"], *common, cp.uint32(1)), stream=d.stream)
            kfsn(grid, block, (a["u3d"], *common, cp.uint32(1)), stream=d.stream)
            kfsn(grid, block, (a["v3d"], *common, cp.uint32(2)), stream=d.stream)
            kfwe(grid, block, (a["v3d"], *common, cp.uint32(2)), stream=d.stream)
            kfwe(grid, block, (a["w3d"], *common, cp.uint32(3)), stream=d.stream)
            kfsn(grid, block, (a["w3d"], *common, cp.uint32(3)), stream=d.stream)
            rigid(grid, block, (a["w3d"], *common), stream=d.stream)
            # solve2.F:1676: bcwsfc is terrain-only, and a stretched
            # Cartesian column is flat (terrain_flag=.false.).
            if not getattr(d, "stretched", False):
                edge_words = (g.nx + 2) * (g.ny + 2)
                d.boundary_compiled.bcwsfc_kernel(
                    ((edge_words + 255) // 256, 1, 1), block,
                    (a["u3d"], a["v3d"], a["w3d"], m["gz"], m["dzdx"], m["dzdy"], *common),
                    stream=d.stream,
                )
            for current, baseline in (("u3d", "ua"), ("v3d", "va"), ("w3d", "wa")):
                cp.copyto(a[baseline], a[current])
        for name in ("pp3d", "ppi"):
            value = d.arrays[name]
            for kernel in d.open_compiled.kernels[3:5]:
                kernel(((int(value.size) + 127) // 128, 1, 1), (128, 1, 1),
                       (value, cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz)),
                       stream=d.stream)
        # init3d.F getset reads q3d(nqv) only; the launcher checks the whole
        # axis against the run's numq, exactly as the dynamics' own startup.
        launch_cuda_cartesian_startup(
            d.fields, d.metrics, ni=g.nx, nj=g.ny, nk=g.nz, cupy=cp, stream=d.stream,
            compiled=d.startup_compiled, density_kernel=d.forecast_compiled.kernels[1],
            numq=int(d.numq),
        )


__all__ = [
    "NEST_LIVE_SCHEMA",
    "WRF_PLACEMENT",
    "NestDriver",
    "NestLateralForcing",
    "child_runtime_config",
    "make_child_runtime_class",
    "nest_child_parcel_layout",
]
