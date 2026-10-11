"""Source-ordered, bounded-workspace Cartesian solve1 and SGS-TKE runtime.

The physical atmosphere is supplied by the Rust initializer through
CartesianDynamics.  One tile owns each published word.  Closure coefficients
and solve1 terms are consumed together before their workspace is released;
there is no second resident atmosphere or reference execution.  The reused
CUDA operators retain their CM1 arithmetic and ordering.  This composition
does not claim native-CM1 trajectory identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from . import cuda_graphs
from .cuda_gm1_child_large_step import (
    CHILD_TERRAIN_RAYLEIGH_ARRAYS,
    GM1_CHILD_RAYLEIGH_SOURCE_SHA256,
    compile_cuda_gm1_child_terrain_rayleigh,
)
from .cuda_gm1_child_nm_def import CHILD_MOIST_ALL_NAMES, launch_cuda_child_moist_nm
from .cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics
from .cuda_gm1_flat_stretched_closure import (
    GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE_SHA256,
    compile_cuda_gm1_flat_stretched_closure,
    enqueue_flat_calcdef,
    enqueue_flat_gettau,
    enqueue_flat_momentum_stress,
    enqueue_flat_turbt,
)
from .cuda_gm1_child_stretched_scalar_turbulence import (
    CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256,
    child_stretched_scalar_turbulence_shapes,
    compile_cuda_gm1_child_stretched_scalar_turbulence,
)
from .cuda_gm1_child_stretched_sgs_tke import (
    CARTESIAN_VERTICAL_METRIC_SCHEMA,
    ChildStretchedMetricView,
    compile_cuda_gm1_child_stretched_sgs_tke,
    launch_cuda_gm1_child_stretched_sgs_stress,
)
from .cuda_gm1_child_turbsmag_init import OUTPUT_ARRAY_NAMES as FRESH_OUTPUT_NAMES
from .cuda_gm1_child_turbsmag_init_generic import CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS
from .cuda_gm1_scalar_turbulence import Gm1ScalarTurbulenceWorkspace, CUDA_GM1_SCALAR_TURBULENCE_THREADS
from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec, build_gm1_open_tile_plan
from .cuda_gm1_tilestream_solve1_full_held import solve1_full_held_tile_shapes
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE,
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_KERNEL_NAME,
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_SHA256,
)
from .cuda_gm1_tilestream_stretched_closure import (
    MOIST_SCRATCH_NAMES,
    OPERATOR_SOURCE_SHA256,
    compile_cuda_gm1_tiled_stretched_closure,
    tiled_stretched_closure_input_shapes,
    tiled_stretched_closure_workspace_shapes,
)
from .cuda_sgs_stress_tendency import SgsStressSelection, sgs_stress_shapes
from .cuda_surface_drag import (
    SURFACE_FLUX_STATE_NAMES,
    SURFACE_PERSISTENT_NAMES,
    SURFACE_STATE_NAMES,
    SurfaceDragBinding,
    compile_cuda_surface_drag,
    enqueue_surface_dissipation,
    enqueue_surface_enthalpy_flux,
    enqueue_surface_layer,
    enqueue_surface_stress,
    enqueue_surface_tke_face,
    surface_drag_shapes,
)
from .eqtset2_surface_dissipation import (
    SurfaceDissipationCarriers,
    apply_surface_dissipative_heating,
)
from .cuda_sgs_tke import (
    CUDA_SGS_TKE_SOURCE_SHA256,
    compile_cuda_sgs_tke,
    launch_cuda_sgs_tke,
)
from .cuda_gm1_solve1_idealized_radiation import (
    CUDA_GM1_RTERM_AUTHORITY,
    CUDA_GM1_RTERM_SOURCE_SHA256,
    compile_cuda_gm1_rterm,
    launch_cuda_gm1_rterm,
)
from .cuda_gm1_solve1_lspgrad1_closure import (
    CARTESIAN_LSPGRAD1_AUTHORITY,
    CUDA_GM1_LSPGRAD1_SOURCE_SHA256,
    SOURCE_ORDER_ENTRY as LSPGRAD_SOURCE_ORDER_ENTRY,
    build_cartesian_lspgrad1_profiles,
    compile_cuda_gm1_lspgrad1,
    launch_cuda_gm1_lspgrad1,
)
from .cuda_stretch_z1 import launch_cuda_stretch_z1
from .cuda_stretch_z1_terrain import CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256
from .cuda_terrain_deformation import CUDA_TERRAIN_DEFORMATION_THREADS
from .eqtset2_energy_terms import Eqtset2IdissCarriers, Eqtset2IdissConsumer
from .grid_geometry import CartesianGridGeometry


SOURCE_ORDER = (
    "source-built Cartesian metric and atmospheric tile",
    "calcnm then calcdef",
    "fresh initial TKE or retained post-solve3 TKE coefficients",
    "solve1 TKE local source then turbt diffusion",
    "Rayleigh held assignment",
    "eqtset=2 idiss theta contribution",
    "SGS momentum stress divergence",
    "scalar turbulence theta then qv/qc/qr/qi/qs/qg",
    "Rayleigh plus stress composition and exactly-once owned scatter",
)

#: src/solve1.F:1006-1053 sits between the idiss block that ends at :1001
#: and the subgrid turbulence that starts at :1116.  It is spliced into
#: SOURCE_ORDER only when rterm=1 selects it, so an rterm=0 receipt -- and
#: the operation token sealed from it -- is byte-identical to the receipt
#: this runtime published before the operator existed.
RTERM_SOURCE_ORDER_ENTRY = "rterm=1 capped idealized-radiation theta tendency"
_RTERM_SOURCE_ORDER_AFTER = "eqtset=2 idiss theta contribution"

#: src/solve1.F:590-612 sits between the Rayleigh assignment that ends at :446
#: and the difforder=2 diffusion at :686, so ahead of idiss.  It is spliced in
#: by the same rule as rterm: an lspgrad=0 receipt, and the operation token
#: sealed from it, is byte-identical to the one published before it existed.
_LSPGRAD_SOURCE_ORDER_AFTER = "Rayleigh held assignment"


def source_order(rterm: int, lspgrad: int = 0) -> tuple[str, ...]:
    """The recorded operator order for one run's selection."""

    order = SOURCE_ORDER
    if lspgrad == 1:
        index = order.index(_LSPGRAD_SOURCE_ORDER_AFTER) + 1
        order = order[:index] + (LSPGRAD_SOURCE_ORDER_ENTRY,) + order[index:]
    if rterm == 1:
        index = order.index(_RTERM_SOURCE_ORDER_AFTER) + 1
        order = order[:index] + (RTERM_SOURCE_ORDER_ENTRY,) + order[index:]
    return order


def _hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _tile_plan(grid: CartesianGridGeometry, tile_nx: int, tile_ny: int) -> Gm1TilePlan:
    if grid.nx >= tile_nx + 6 and grid.ny >= tile_ny + 6:
        return build_gm1_open_tile_plan(nx=grid.nx, ny=grid.ny, tile_nx=tile_nx, tile_ny=tile_ny)
    spec = Gm1TileSpec(0, 0, 0, grid.nx, grid.ny, grid.nx, grid.ny, 3, 0, grid.nx, 0, grid.ny, 0, 0, grid.nx, grid.ny)
    return Gm1TilePlan(grid.nx, grid.ny, grid.nx, grid.ny, 3, (spec,), _hash({"full_domain_tile": spec.record()}))


def _scalar_workspace_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    return {
        "rho_halo": (nk, nj + 2, ni + 2), "khh_halo": (nk + 1, nj + 2, ni + 2),
        "gz_halo": (nj + 2, ni + 2), "rgz_halo": (nj + 2, ni + 2),
        "scalar_halo": (nk, nj + 2, ni + 2), "scalar_w": (nk + 1, nj + 2, ni + 2),
        "flux_x": (nk, nj, ni + 1), "flux_y": (nk, nj + 1, ni),
        "flux_x_w": (nk + 1, nj, ni + 1), "flux_y_w": (nk + 1, nj + 1, ni),
        "thomas_c": (nk, nj, ni), "thomas_d": (nk, nj, ni),
    }


def cartesian_closure_allocation_plan(config: Any) -> dict[str, Any]:
    """Exact payload inventory; allocator rounding remains registry-owned."""

    g = config.grid
    spec = _tile_plan(g, config.tile_nx, config.tile_ny).specs[0]
    ni, nj, nk = spec.cnx, spec.cny, g.nz
    tile = dict(solve1_full_held_tile_shapes(ni=ni, nj=nj, nk=nk, numq=config.numq))
    tile.update(tiled_stretched_closure_input_shapes(ni, nj, nk))
    closure = dict(tiled_stretched_closure_workspace_shapes(ni, nj, nk))
    closure.pop("rds")
    closure.pop("rdsf")
    groups = {
        "tile": tile,
        "closure": closure,
        "scalar": _scalar_workspace_shapes(ni, nj, nk),
        "retained": {name: (nk + 1, nj, ni) for name in ("tk", "grdscl", "rgrdscl", "cme", "ce1", "ce2", "csm")},
        "idiss": {name: (nk, nj, ni) for name in ("ql", "qice", "cvm", "work", "epsd")},
    }
    # CM1's lower boundary owns twelve (nj, ni) surface words when it is not
    # free slip: the lowest model level wind, the exchange coefficients
    # getcecd derives from it, the surface stress magnitude and the surface
    # dissipation rate (src/turb.F:532-555, src/sfcphys.F:120-136,
    # src/turb.F:4692-4700, src/solve1.F:956-957).  A free-slip run
    # allocates none of them, so its payload is exactly what it was.
    if config.bbc != 1:
        surface = surface_drag_shapes(ni, nj, nk)
        groups["surface"] = {name: surface[name] for name in SURFACE_STATE_NAMES}
        # src/init_surface.F:40-65.  The land/water flag, the moisture
        # availability and the skin temperature are whole-domain state the
        # surface layer reads rather than derives, so each tile gets its own
        # rectangle of them; the roughness length is already in
        # SURFACE_STATE_NAMES because getcecd writes it.
        groups["surface"].update(
            {
                name: surface[name]
                for name in SURFACE_PERSISTENT_NAMES
                if name not in groups["surface"]
            }
        )
        if config.isfcflx == 1:
            # src/sfcphys.F:197-236 and :337-352.  Nothing carries the
            # enthalpy coefficients or the surface fluxes for isfcflx=0, so
            # a run without surface fluxes allocates none of them.
            groups["surface"].update(
                {name: surface[name] for name in SURFACE_FLUX_STATE_NAMES}
            )
        if config.bbc == 3:
            # src/solve1.F:932-935 starts the interior dissipative-heating
            # average at k=2 for bbc=3, so the sealed consumer runs over the
            # k=2..nk slice and its scratch is one level shorter.
            groups["idiss"] = {name: (nk - 1, nj, ni) for name in groups["idiss"]}
    # ... plus the four-byte device scalars the phase allocates outside the
    # named groups.  Two travel with every phase -- the metric/TKE
    # invalid-operand counter and the ksmax fold of misclibs.F:2846-2866 --
    # and a third only when there is a surface: src/sfcphys.F:189-191's
    # water-roughness nonconvergence stop.  Each has to be counted, because a
    # sealed allocation plan that under-counts is a budget a run can exceed.
    held_bytes = (
        sum(
            math.prod(shape) * (1 if name == "saturation_mask" else 4)
            for values in groups.values()
            for name, shape in values.items()
        )
        + 8
        + (0 if config.bbc == 1 else 4)
    )
    tke_shapes = {
        **{name: (nk + 1, nj, ni) for name in ("tkea", "tke3d", "tketen", "c1", "c2", "rrf0", "output")},
        "gz": (nj, ni), "divx": (nk, nj, ni),
        "rru": (nk + 2, nj + 6, ni + 7), "rrv": (nk + 2, nj + 7, ni + 6),
        "rrw": (nk + 3, nj + 6, ni + 6),
    }
    return {
        "schema": "gm1.cartesian-closure-allocation-plan/v1", "geometry": g.record(),
        "compute_shape_zyx": [nk, nj, ni], "held_groups": groups,
        "held_phase_payload_bytes": held_bytes,
        "tke_stage_shapes": tke_shapes,
        "tke_phase_payload_bytes": 4 * (math.prod(g.w_shape) + sum(math.prod(shape) for shape in tke_shapes.values())),
        # src/solve1.F:590-612.  lspgrad=1 holds its two per-level profiles
        # for the life of the run; lspgrad=0 holds none, so its plan -- and
        # every price sealed from it -- is exactly what it was.
        "persistent_profile_payload_bytes": 4 * (5 * (nk + 2) + 5 * (nk + 3) + 3 * nk + 1)
        + (4 * 2 * nk if int(getattr(config, "lspgrad", 0)) == 1 else 0),
        # src/init_surface.F:40-65 whole-domain surface state, held for the
        # life of the run rather than rebuilt per phase, because
        # src/sfcphys.F:155-158 carries the roughness length from one step to
        # the next.  A free-slip bottom has no surface, and its plan carries
        # no key for one, so its bytes are exactly what they always were.
        **(
            {}
            if config.bbc == 1
            else {
                "surface_state_payload_bytes": (
                    4 * len(SURFACE_PERSISTENT_NAMES) * g.ny * g.nx
                )
            }
        ),
        "claim_limit": "one workspace per compute window and one TKE stage output; no second reference atmosphere",
    }


def _copy_tile(cp: Any, target: Any, source: Any, spec: Gm1TileSpec) -> None:
    """Pack the physical rectangle, including its true staggered extra face."""

    y = slice(spec.cj0, spec.cj0 + target.shape[-2])
    x = slice(spec.ci0, spec.ci0 + target.shape[-1])
    leading = tuple(slice(0, size) for size in target.shape[:-2])
    cp.copyto(target, source[(*leading, y, x)])


def _scatter(cp: Any, target: Any, source: Any, spec: Gm1TileSpec, stagger: str = "mass") -> None:
    y0, y1, x0, x1 = spec.owned_rect(stagger)
    cp.copyto(
        target[..., y0:y1, x0:x1],
        source[..., y0 - spec.cj0:y1 - spec.cj0, x0 - spec.ci0:x1 - spec.ci0],
    )


@dataclass(slots=True)
class _PhaseStorage:
    registry: Any
    prefix: str
    names: list[str]

    def allocate(self, name: str, shape: tuple[int, ...], *, dtype: Any = None, zero: bool = False) -> Any:
        full = f"{self.prefix}.{name}"
        value = self.registry.allocate(full, shape, dtype=dtype, zero=zero)
        self.names.append(full)
        return value

    def release(self) -> None:
        if self.names:
            self.registry.release(self.names)
            self.names.clear()



def _refuse_invalid_closure_cells(value: Any) -> None:
    invalid_count = int(value[0])
    if invalid_count:
        raise ValueError(f"closure metric/TKE operands contain {invalid_count} invalid tile cells")


def _refuse_nonconvergent_roughness(value: Any) -> None:
    if int(value[0]):
        raise ValueError(
            "the wind-dependent water roughness did not converge in "
            "twenty iterations (src/sfcphys.F:189-191); CM1 stops "
            "here, because a drag coefficient taken from an "
            "unconverged 10 m wind is not the one the surface exerts"
        )

class CartesianClosure:
    # misclibs.F:2890's ``ksmax``, less the dt-dependent factor: the largest
    # kmh/khh/kmv this closure produced at k>=2, in m^2/s.  Zero until the
    # first held-tendency phase has run, which is also the first time CM1
    # itself has a number to read.
    max_eddy_coefficient_m2_s: float = 0.0

    # README.namelist:391-403.  Class defaults so a caller that builds this
    # type without the runtime table -- the operator-order tests do -- still
    # gets CM1's own supercell selection rather than an attribute error.
    irdamp: int = 1
    hrdamp: int = 0
    xhd_m: float = 0.0
    rayleigh_active: bool = True
    family_compiled: Any = None

    #: Declared on the class so an instance assembled another way -- a test
    #: double, a probe -- reads the rterm=0 identity rather than raising
    #: inside the held tile.  ``_compile`` sets it for a real closure.
    rterm_compiled: Any = None
    #: The same rule for lspgrad=0: no kernel, no profiles, no receipt key.
    lspgrad_compiled: Any = None
    lspgrad_profiles: Any = None
    lspgrad_ul: Any = None
    lspgrad_vl: Any = None
    #: A :class:`cm1core.cartesian_budgets.CartesianBudgetStage` on a budget
    #: step, read-only with respect to every closure carrier; ``None`` otherwise.
    budget_probe: Any = None
    #: The stretched Cartesian column is terrain_flag=.false.; its closure is
    #: :mod:`cm1core.cuda_gm1_flat_stretched_closure`.  ``None`` on the
    #: uniform column, which keeps the images it has always launched.
    flat_compiled: Any = None

    def _budget(self) -> Any:
        probe = self.budget_probe
        return probe if probe is not None and probe.active else None

    #: README.namelist:334-352.  ``None`` is sgsmodel=1; a closure built for
    #: sgsmodel=3/4 holds its :class:`~cm1core.cuda_gm1_two_part_sgs.CartesianTwoPart`.
    two_part: Any = None
    #: cm1.F:3724's rtime for the phase about to run, set by the runtime.
    model_time_s: float | None = None
    #: CM1 diagnoses the first step's TKE with turbsmag only when it
    #: interpolates a restart onto a new grid (``src/init3d.F:2688-2760``,
    #: ``interp_on_restart``).  A fresh ``iinit`` start runs tkekm on its zero
    #: TKE from step 1, and a restart carries the TKE it wrote.  This lane
    #: took the turbsmag branch on every step 1, which on a fresh start seeds a
    #: Smagorinsky TKE CM1 never has (1.7e-4 K of theta' after one step on
    #: CM1's supercell deck).  A caller that builds a runtime from an
    #: interpolated state sets this before the first step.
    interpolated_tke_start: bool = False
    #: The SGS start-up ramp's words for the step being prepared (see
    #: cuda_cartesian_sgs_ramp); ``None`` when the sealed cme=0.10 runs.
    sgs_ramp_step: Mapping[str, Any] | None = None
    sgs_ramp_compiled: Any = None
    #: README.namelist:1081-1087; the constructor binds the run's selection,
    #: and a shell built without it is src/cm1.F:355-356's ramp_sgs=0.
    ramp_sgs: int = 0
    ramp_time_s: float = 0.0

    def __init__(self, dynamics: Any, *, damping_depth_m: float | None = None, damping_rate_s_inv: float | None = None) -> None:
        self.dynamics, self.config = dynamics, dynamics.config
        self.grid, self.cp, self.stream, self.registry = self.config.grid, dynamics.cupy, dynamics.stream, dynamics.registry
        g = self.grid
        damping_depth_m = self.config.damping_depth_m if damping_depth_m is None else damping_depth_m
        damping_rate_s_inv = self.config.damping_rate_s_inv if damping_rate_s_inv is None else damping_rate_s_inv
        g.require_stencil(radius_xy=3, minimum_nz=8)
        if g.dx_m != g.dy_m:
            raise ValueError("SGS momentum stress uses one horizontal reciprocal and requires dx_m == dy_m")
        for name, value in (("damping_depth_m", damping_depth_m), ("damping_rate_s_inv", damping_rate_s_inv)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if damping_depth_m > g.ztop_m * 0.5:
            raise ValueError("CM1 param.F refuses Rayleigh damping over more than half the domain")
        self.damping_depth_m, self.damping_rate_s_inv = float(damping_depth_m), float(damping_rate_s_inv)
        # README.namelist:391-403.  irdamp selects which departure the upper
        # sponge damps toward and hrdamp adds the lateral one; both branches of
        # src/solve1.F:404-438 reach the same held-assignment kernel, because
        # irdamp=2 is irdamp=1 with three operands replaced.
        self.irdamp = int(getattr(self.config, "irdamp", 1))
        self.hrdamp = int(getattr(self.config, "hrdamp", 0))
        self.xhd_m = float(getattr(self.config, "xhd_m", 0.0))
        self.rayleigh_active = self.irdamp >= 1 or self.hrdamp >= 1
        # README.namelist:1081-1087, src/turb.F:240-262.
        self.ramp_sgs = int(getattr(self.config, "ramp_sgs", 0))
        self.ramp_time_s = float(getattr(self.config, "ramp_time_s", 0.0))
        self.plan = _tile_plan(g, self.config.tile_nx, self.config.tile_ny)
        self.allocation_plan = cartesian_closure_allocation_plan(self.config)
        self.last_receipts: dict[str, Mapping[str, Any]] = {}
        self._pending_step: int | None = None
        self._held_step = 0
        self._tke_stage = 3
        self.flat_stretched = int(g.stretch_z) != 0
        self._compile()
        self._vertical_profiles()
        self._lspgrad_profiles()
        self._surface_fields()
        # README.namelist:334-352.  Only sgsmodel=3/4 build the two-part
        # state; sgsmodel=1 compiles, allocates and publishes nothing extra.
        if int(getattr(self.config, "sgsmodel", 1)) in (3, 4):
            from .cuda_gm1_two_part_sgs import CartesianTwoPart

            self.two_part = CartesianTwoPart(self)

    def _surface_fields(self) -> None:
        """``src/init_surface.F:40-322``, once, before the first step.

        The four fields the surface layer reads rather than derives -- the
        roughness length, the land/water flag, the moisture availability and
        the skin temperature -- are whole-domain state.  ``znt`` in
        particular is carried from step to step: ``src/sfcphys.F:155-158``
        starts the wind-dependent water-roughness iteration from the value
        the previous step converged to, so rebuilding it per tile would
        restart that iteration from whichever tile ran last.
        """

        self.surface_state: dict[str, Any] = {}
        self.surface_field_receipt: Mapping[str, Any] | None = None
        if not self.surface_selection.active:
            return
        from .surface_fields import build_surface_fields

        cp, g = self.cp, self.grid
        fields = build_surface_fields(
            selection=self.config.surface_field_selection,
            nx=g.nx,
            ny=g.ny,
            dx_m=g.dx_m,
            dy_m=g.dy_m,
        )
        with self.stream:
            for name in SURFACE_PERSISTENT_NAMES:
                value = self.registry.allocate(
                    f"closure.surface.{name}", (g.ny, g.nx), zero=True
                )
                value.set(fields.arrays[name], stream=self.stream)
                self.surface_state[name] = value
        self.surface_field_receipt = fields.receipt
        self.last_receipts["surface_fields"] = dict(fields.receipt)
        # src/init_surface.F:207-227.  dosfcflx is what actually switches the
        # scalar lower flux on at src/turbtend.F:519; isfcflx=1 alone does
        # not, and the two disagreeing would be a silently frictionless
        # surface energy budget.
        self.surface_dosfcflx = bool(fields.dosfcflx)
        if self.surface_selection.uses_surface_enthalpy_flux and not fields.dosfcflx:
            raise RuntimeError(
                "isfcflx=1 but init_surface.F:207-227 left dosfcflx false, so "
                "src/turbtend.F:519 would keep the scalar lower flux at zero "
                "while the receipt named a surface energy budget"
            )

    @property
    def surface_pressure_weights(self) -> tuple[float, float, float]:
        """``src/param.F:9038-9040`` cgs1,cgs2,cgs3 for this vertical grid."""

        from .cm1_vertical_grid import fornberg_extrapolation_weights

        levels = self.grid.levels
        weights = fornberg_extrapolation_weights(
            levels.sigma, levels.sigmaf, nk=self.grid.nz, top=False
        )
        return (float(weights[0]), float(weights[1]), float(weights[2]))

    @property
    def surface_selection(self) -> Any:
        """CM1's lower boundary for this run, from the bound runtime table.

        ``bbc=1`` keeps every free-slip word this runtime already produced
        and compiles, allocates and launches nothing extra.
        """

        return self.config.surface_drag_selection

    @property
    def rterm(self) -> int:
        """CM1's ``rterm`` for this run, from the bound runtime table."""

        return int(getattr(self.config, "rterm", 0))

    @property
    def lspgrad(self) -> int:
        """CM1's ``lspgrad`` for this run, from the bound runtime table."""

        return int(getattr(self.config, "lspgrad", 0))

    def _lspgrad_profiles(self) -> None:
        """``src/solve1.F:590-612``'s two per-level words, once, before step 1.

        Built from this domain's own compact ``u0``/``v0`` -- the frame-relative
        base state the Rayleigh kernel damps toward (base.F:2661-2674) -- so a
        nest child, whose source builds that state on its own levels, forms
        its forcing on its own levels.  The profiles are refused unless they
        are raw-word identical to the source-exact CPU operator.
        """

        self.lspgrad_ul = self.lspgrad_vl = None
        if self.lspgrad != 1:
            return
        g, config = self.grid, self.config
        base = self.dynamics.base_profiles
        u0 = base["u0"].get(stream=self.stream)
        v0 = base["v0"].get(stream=self.stream)
        profiles = build_cartesian_lspgrad1_profiles(
            u0, v0,
            fcor_s_inv=config.fcor_s_inv,
            u_move_m_s=config.u_move_m_s,
            v_move_m_s=config.v_move_m_s,
        )
        if profiles.ul.shape != (g.nz,):
            raise ValueError(
                f"the base-state wind has {profiles.ul.shape[0]} levels but this "
                f"closure integrates {g.nz}"
            )
        with self.stream:
            self.lspgrad_ul = self.registry.allocate("closure.lspgrad.ul", (g.nz,))
            self.lspgrad_vl = self.registry.allocate("closure.lspgrad.vl", (g.nz,))
            self.lspgrad_ul.set(profiles.ul, stream=self.stream)
            self.lspgrad_vl.set(profiles.vl, stream=self.stream)
        self.lspgrad_profiles = profiles
        self.last_receipts["lspgrad_profiles"] = dict(profiles.receipt)

    @property
    def stress_selection(self) -> SgsStressSelection:
        """The gettau/turbu selection, differing only in CM1's ``bbc``."""

        return SgsStressSelection(bbc=self.surface_selection.bbc)

    def _compile(self) -> None:
        self.closure_compiled = compile_cuda_gm1_tiled_stretched_closure()
        self.sgs_compiled = compile_cuda_gm1_child_stretched_sgs_tke(max_nk=self.grid.nz)
        # A stretched Cartesian column is flat (config.py admits grid.terrain
        # only on the real-product lane): turb.F/turbtend.F/adv.F run their
        # terrain_flag=.false. statements on rdz*mh(1,1,k) and rdz*mf, not
        # the rds/rdsf terrain words the images above were derived for.
        self.flat_compiled = (
            compile_cuda_gm1_flat_stretched_closure(max_nk=self.grid.nz)
            if self.flat_stretched
            else None
        )
        self.retained_compiled = compile_cuda_sgs_tke()
        self.scalar_compiled = compile_cuda_gm1_child_stretched_scalar_turbulence()
        self.rayleigh_compiled = compile_cuda_gm1_child_terrain_rayleigh()
        self.family_compiled = None
        if self.hrdamp >= 1 or self.irdamp == 2:
            from .cuda_boundary_families import compile_cuda_boundary_families

            self.family_compiled = compile_cuda_boundary_families()
        self.surface_compiled = (
            compile_cuda_surface_drag() if self.surface_selection.active else None
        )
        # src/solve1.F:1006.  rterm=0 compiles and launches nothing extra, so
        # a run that does not ask for the Rotunno-Emanuel relaxation is
        # exactly the run it was before this operator existed.
        self.rterm_compiled = (
            compile_cuda_gm1_rterm() if self.rterm == 1 else None
        )
        # src/solve1.F:538.  lspgrad=0 compiles and launches nothing extra.
        self.lspgrad_compiled = (
            compile_cuda_gm1_lspgrad1() if self.lspgrad == 1 else None
        )
        options = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
        if hashlib.sha256(CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE.encode("ascii")).hexdigest() != CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_SHA256:
            raise RuntimeError("local TKE source differs from its pin")
        from .nvrtc_module import compile_nvrtc_module

        self.local_tke_module = compile_nvrtc_module(self.cp, code=CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE, options=options, name_expressions=(CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_KERNEL_NAME,), filename="gm1_tilestream_solve1_tke.cu")
        self.local_tke_kernel = self.local_tke_module.get_function(CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_KERNEL_NAME)

    def _vertical_profiles(self) -> None:
        from .gm1_rayleigh_profiles import build_rayleigh_profiles

        cp, g = self.cp, self.grid
        # param.F:7789-7811 for an unstretched grid, param.F:7819-7953 or
        # :8005-8137 otherwise.  There is exactly one generator: the degenerate
        # stretch_z=1 parameters this used to hard-code are now what the
        # stretch_z=0 member of that family produces, so an unstretched run
        # reaches the same kernel with the same words.
        levels = g.levels
        rayleigh = build_rayleigh_profiles(
            self.dynamics.source.library, levels=levels,
            damping_depth_m=self.damping_depth_m, damping_rate_s_inv=self.damping_rate_s_inv,
        )
        self.damping_rate_s_inv = float(rayleigh.parameters["damping_rate_s_inv"])
        with self.stream:
            if levels.stretch_z in (0, 1):
                parameters = levels.stretch_parameters()
                vertical = launch_cuda_stretch_z1(parameters, stream=self.stream)
            else:
                # param.F:8005-8132 reads its levels rather than generating
                # them, so only the param.F:8162-8209 tail runs on device.
                from .cuda_vertical_levels import launch_cuda_vertical_levels

                vertical = launch_cuda_vertical_levels(levels, stream=self.stream)
            for name in ("sigma", "sigmaf", "zh", "zf", "mh", "mf", "rmh", "rmf"):
                self.registry.adopt(f"closure.vertical.{name}", getattr(vertical.grid, name), read_only=True)
            rds = self.registry.allocate("closure.vertical.rds", (g.nz + 2,))
            rdsf = self.registry.allocate("closure.vertical.rdsf", (g.nz + 3,))
            self.closure_compiled.static.kernels[3]((1, 1, 1), (1, 1, 1), (vertical.grid.sigma, vertical.grid.sigmaf, rds, rdsf, cp.uint32(g.nz)), stream=self.stream)
            self.tauh_profile = self.registry.allocate("closure.rayleigh.mass", (g.nz,))
            self.taus_profile = self.registry.allocate("closure.rayleigh.theta", (g.nz,))
            self.tauf_profile = self.registry.allocate("closure.rayleigh.face", (g.nz + 1,))
            self.tauh_profile.set(rayleigh.tauh, stream=self.stream)
            self.taus_profile.set(rayleigh.taus, stream=self.stream)
            self.tauf_profile.set(rayleigh.tauf, stream=self.stream)
            # src/param.F:8633 builds the vertical coefficients only when
            # irdamp>=1.  With the upper damper off, a lateral sponge must act
            # on tauf alone, so the vertical profile is exactly zero rather
            # than a cosine nothing selected.
            if self.irdamp == 0:
                for profile in (self.tauh_profile, self.taus_profile, self.tauf_profile):
                    profile.fill(cp.float32(0.0))
            event = cp.cuda.Event(disable_timing=True)
            event.record(self.stream)
        identity = {
            "schema": CARTESIAN_VERTICAL_METRIC_SCHEMA[
                "uniform" if levels.stretch_z == 0 else "stretched"
            ],
            "grid_geometry": g.record(),
            "vertical_producer": dict(vertical.receipt),
            "static_source_sha256": CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256,
            "profile_indexing": "direct CM1 indices 0:nk+1 and 0:nk+2",
            "ztop_f32": g.ztop_m, "dz_reference_f32": g.dz_m,
        }
        flat_column: dict[str, Any] = {}
        if levels.stretch_z != 0:
            identity["vertical_coordinate"] = levels.record()
            # The Cartesian lane is terrain_flag=.false.: its stretched operators
            # run CM1's flat statements on param.F:8467,8476's mh(1,1,k) and
            # mf(1,1,k), so the binding carries the generated column's own.
            identity["column"] = "flat: terrain_flag=.false., rdz*mh(1,1,k) and rdz*mf(1,1,k)"
            self._require_flat_column_is_the_metric_column(vertical.grid, g.nz)
            flat_column = {"mh": vertical.grid.mh, "mf": vertical.grid.mf}
        self.vertical = Gm1ChildStretchedMetrics(vertical.grid.sigma, vertical.grid.sigmaf, rds, rdsf, g.ztop_m, g.dz_m, self.sgs_compiled.target_arch, MappingProxyType(identity), **flat_column)
        self.vertical_view = ChildStretchedMetricView(self.vertical.arrays, event, MappingProxyType(identity), False, g)
        self.dynamics.vertical, self.dynamics.vertical_launch = self.vertical, vertical
        self.vertical_launch = vertical
        self.last_receipts["rayleigh_coefficients"] = dict(rayleigh.receipt)

    def _require_flat_column_is_the_metric_column(self, grid: Any, nk: int) -> None:
        """The profile the dynamics reads is the column the Rust metrics carry.

        The flat operators read ``mh``/``mf`` from the generated device
        profile, while the closure's vertical diffusion reads the uploaded
        3-D ``mh``/``mf`` metrics.  CM1 has one array for both
        (param.F:8467, 8476), so the two must be one set of words at the
        first physical column; a disagreement would integrate two columns.
        One host read, at construction.
        """

        cp, metrics = self.cp, self.dynamics.metrics
        with self.stream:
            same = cp.logical_and(
                cp.all(cp.ascontiguousarray(grid.mh[1 : nk + 1]).view(cp.uint32)
                       == cp.ascontiguousarray(metrics["mh"][:, 0, 0]).view(cp.uint32)),
                cp.all(cp.ascontiguousarray(grid.mf[1 : nk + 2]).view(cp.uint32)
                       == cp.ascontiguousarray(metrics["mf"][:, 0, 0]).view(cp.uint32)),
            )
        if not bool(same):
            raise ValueError(
                "the generated mh/mf column differs from the uploaded metric column; "
                "param.F:8467,8476 build one mh and one mf, and the flat operators "
                "would integrate a different column than the vertical diffusion"
            )

    def checkpoint_step(self) -> int:
        """The completed large step this closure stands at, or refuse.

        ``src/cm1.F:4246`` writes a restart only after a completed large step.
        These three counters are what say the closure is there: the held
        coefficients belong to a finished step, no step is part-way through,
        and all three TKE RK stages have run.
        """

        if self._pending_step is not None or self._tke_stage != 3:
            raise RuntimeError(
                f"the closure stands at held step {self._held_step} with "
                f"pending={self._pending_step!r} and tke stage {self._tke_stage}; "
                "a checkpoint taken here would persist coefficients from one "
                "large step beside carriers from another"
            )
        return self._held_step

    def resume_checkpoint_step(self, step: int) -> None:
        """Continue a restored trajectory's closure cadence at ``step``.

        Without this the closure would refuse the first large step after a
        resume -- it expects ``step == self._held_step + 1`` -- and a closure
        that silently accepted it would recompute the held coefficients under
        a step index the rest of the runtime had already passed.
        """

        if type(step) is not int or step < 0:
            raise ValueError("a resumed closure step must be a nonnegative integer")
        if self._held_step != 0 or self._pending_step is not None or self._tke_stage != 3:
            raise RuntimeError(
                "a closure is resumed into a freshly built runtime only; this "
                f"one already stands at held step {self._held_step}"
            )
        self._held_step = step

    def closure(self, step: int) -> None:
        if type(step) is not int or step != self._held_step + 1 or self._pending_step is not None or self._tke_stage != 3:
            raise RuntimeError("closure requires the next complete large-step boundary")
        cuda_graphs.bookkeeping(self._set_pending_step, step)

    def _set_pending_step(self, step: int) -> None:
        self._pending_step = step

    def _set_held_step(self, step: int) -> None:
        self._held_step, self._pending_step, self._tke_stage = step, None, 0

    def sgs_ramp_active(self, model_time_s: float) -> bool:
        """Whether ``src/turb.F:240-262`` scales ``cme`` below ``c_m`` for this step."""

        if self.ramp_sgs != 1:
            return False
        from .cuda_cartesian_sgs_ramp import cm1_turbfac

        return bool(cm1_turbfac(model_time_s, self.ramp_time_s) < 1.0)

    def _prepare_sgs_ramp(self) -> None:
        """``src/turb.F:240-262`` for the step about to run, at cm1.F:3724's rtime."""

        self.sgs_ramp_step = None
        if self.ramp_sgs != 1:
            return
        if self.model_time_s is None:
            raise RuntimeError(
                "ramp_sgs=1 reads the model time at the start of the step "
                "(src/turb.F:242, cm1.F:3724), and the runtime set none"
            )
        from .cuda_cartesian_sgs_ramp import (
            cm1_ramped_cme,
            cm1_turbfac,
            compile_cuda_cartesian_sgs_ramp,
        )

        turbfac = cm1_turbfac(self.model_time_s, self.ramp_time_s)
        if not turbfac < 1.0:
            # cme = max(0.001, 1*c_m) = c_m: the sealed operator's own words.
            return
        if self.sgs_ramp_compiled is None:
            self.sgs_ramp_compiled = compile_cuda_cartesian_sgs_ramp()
        self.sgs_ramp_step = {
            "rtime_s": float(self.model_time_s),
            "turbfac": float(turbfac),
            "cme": cm1_ramped_cme(self.model_time_s, self.ramp_time_s),
        }

    def held_tendencies(self, step: int) -> None:
        if step != self._pending_step:
            raise RuntimeError("held tendencies require the matching closure phase")
        self._prepare_sgs_ramp()
        cp, g = self.cp, self.grid
        spec0 = self.plan.specs[0]
        ni, nj, nk = spec0.cnx, spec0.cny, g.nz
        storage = _PhaseStorage(self.registry, "closure.held", [])
        tile: dict[str, Any] = {}
        work: dict[str, Any] = {}
        surface: dict[str, Any] = {}
        retained: dict[str, Any] = {}
        scalar: Any = None
        idiss: Any = None
        invalid: Any = None
        try:
            groups = self.allocation_plan["held_groups"]
            tile = {name: storage.allocate(f"tile.{name}", shape) for name, shape in groups["tile"].items()}
            work = {name: storage.allocate(f"work.{name}", shape, dtype=cp.uint8 if name == "saturation_mask" else cp.float32) for name, shape in groups["closure"].items()}
            work.update(rds=self.vertical.rds, rdsf=self.vertical.rdsf)
            retained = {name: storage.allocate(f"retained.{name}", shape, zero=True) for name, shape in groups["retained"].items()}
            scalar_values = {name: storage.allocate(f"scalar.{name}", shape) for name, shape in groups["scalar"].items()}
            scalar = Gm1ScalarTurbulenceWorkspace(ni, nj, nk, **scalar_values)
            scalar_values.clear()
            # src/solve1.F:932-935.  bbc=3 starts the interior heating loop
            # at k=2 and heats k=1 from the surface-layer dissipation rate,
            # so the consumer is driven over the k=2..nk slice and records
            # the lower boundary the run actually integrates.
            idiss = Eqtset2IdissConsumer(
                nx=ni,
                ny=nj,
                nz=nk - 1 if self.surface_selection.bbc == 3 else nk,
                bottom_boundary=self.surface_selection.bbc,
            )
            idiss._scratch.update({name: storage.allocate(f"idiss.{name}", shape) for name, shape in groups["idiss"].items()})
            invalid = storage.allocate("invalid", (1,), dtype=cp.uint32, zero=True)
            # misclibs.F:2846-2866 takes the largest of kmh, khh and kmv over
            # k=2..nk.  On this lane's uniform horizontal grid uh=vh=1, so the
            # three expressions share one factor and the domain maximum is
            # exactly that factor times this one scalar -- see
            # cm1_adaptive_timestep.cm1_diffusion_number.  Reducing it here is
            # the only place the coefficients exist: they are tile workspace
            # and are released with this phase.
            eddy_max = storage.allocate("eddy_max", (1,), dtype=cp.float32, zero=True)
            # src/sfcphys.F:189-191 stops CM1 when the wind-dependent water
            # roughness has not converged in twenty iterations.  A kernel
            # cannot stop the model, so it raises this flag and the host
            # reads it at the same point in the step CM1 would have stopped.
            nonconvergent = (
                storage.allocate(
                    "surface_nonconvergent", (1,), dtype=cp.uint32, zero=True
                )
                if self.surface_selection.active
                else None
            )
            surface = {
                name: storage.allocate(f"surface.{name}", shape, zero=True)
                for name, shape in groups.get("surface", {}).items()
            }
            rayleigh_scratch: dict[str, Any] = {}
            if self.irdamp == 2:
                # src/solve1.F:427-438 replaces exactly three operands; the
                # held-assignment kernel is unchanged, so the tile's own u0/v0
                # and tha are left untouched for their other consumers.
                for name, shape in (
                    ("u0", (nk, nj, ni + 1)),
                    ("v0", (nk, nj + 1, ni)),
                    ("tha", (nk, nj, ni)),
                ):
                    rayleigh_scratch[name] = storage.allocate(f"rayleigh.{name}", shape)
                for name in ("uavg", "vavg", "thavg"):
                    rayleigh_scratch[name] = storage.allocate(f"rayleigh.{name}", (nk,))
            with self.stream:
                if self.irdamp == 2:
                    # get_avg_uvtq is a DOMAIN average, so it is taken once
                    # from the whole resident state, not per tile: a tile-local
                    # mean would damp each tile toward its own average and the
                    # seams would become a forcing.
                    self._rayleigh_horizontal_averages(rayleigh_scratch)
                for name, profile in (("tauh", self.tauh_profile), ("taus", self.taus_profile)):
                    cp.copyto(tile[name], cp.broadcast_to(profile[:, None, None], tile[name].shape))
                if self.two_part is not None:
                    self._two_part_prepass(tile, work, surface, nonconvergent, ni, nj, nk)
                for spec in self.plan.specs:
                    # The vertical coefficient is spec-independent; the lateral
                    # taper is not, so tauf is rebuilt per tile whenever a
                    # lateral sponge is selected (src/param.F:8736-8776).
                    cp.copyto(tile["tauf"], cp.broadcast_to(self.tauf_profile[:, None, None], tile["tauf"].shape))
                    if self.hrdamp >= 1:
                        from .cuda_boundary_families import enqueue_hrdamp_tauf

                        enqueue_hrdamp_tauf(
                            tile["tauf"], ni=ni, nj=nj, nk=nk, i0=spec.ci0, j0=spec.cj0,
                            nx=g.nx, ny=g.ny, dx=g.dx_m, dy=g.dy_m, xhd=self.xhd_m,
                            compiled=self.family_compiled, stream=self.stream,
                        )
                    if self._budget() is not None:
                        self._budget().begin_tile(spec)
                    self._gather(tile, work, spec)
                    self._surface_layer(tile, work, surface, ni, nj, nk, spec, nonconvergent)
                    self._surface_enthalpy_flux(tile, surface, ni, nj, nk)
                    self._refresh_nm_def(tile, work, ni, nj, nk)
                    self._tke_coefficients(
                        work, retained, invalid, spec,
                        first=step == 1 and self.interpolated_tke_start,
                    )
                    # src/turb.F:1378.  tkebc is a separate call after tkekm,
                    # and its bbc=2/3 branch replaces the free-slip lower face
                    # the coefficient operators just wrote.
                    self._surface_tke_face(work, surface, ni, nj, nk)
                    self._fold_eddy_coefficient_maximum(work, eddy_max, spec, nk)
                    if self.two_part is None:
                        self._tke_held(tile, work, ni, nj, nk)
                    else:
                        # src/solve1.F:296-310: S94 hands turbt gamk*kmh and
                        # gamk*kmv; B20 hands it the coefficients themselves.
                        self._two_part_scaled = self.two_part.scaled_coefficients(work, spec)
                        if self._two_part_scaled is None:
                            self._tke_held(tile, work, ni, nj, nk)
                        else:
                            self._tke_held(tile, work, ni, nj, nk, km=self._two_part_scaled)
                    self._held_tile(tile, work, surface, scalar, idiss, ni, nj, nk, rayleigh_scratch)
                    if self.two_part is not None:
                        self.two_part.after_tile(spec)
                    for name in ("tkea", "tke3d", "tketen"):
                        _scatter(cp, self.dynamics.tke[name], work[name], spec)
                    for name, stagger in (("uten", "u"), ("vten", "v"), ("wten", "w"), ("thten1", "mass"), ("qten", "mass")):
                        _scatter(cp, self.dynamics.held[name], tile[name], spec, stagger)
            # One host read per step, beside the one this phase already makes;
            # under a CUDA graph both are read after the step's launch
            # (cm1core.cuda_graphs.read_device), in this order.
            cuda_graphs.read_device(invalid, _refuse_invalid_closure_cells, stream=self.stream)
            cuda_graphs.read_device(eddy_max, self._set_eddy_maximum, stream=self.stream)
            if nonconvergent is not None:
                cuda_graphs.read_device(nonconvergent, _refuse_nonconvergent_roughness, stream=self.stream)
        finally:
            tile.clear()
            work.clear()
            retained.clear()
            surface.clear()
            scalar = idiss = invalid = eddy_max = None
            storage.release()
        receipt = {
            "schema": "gm1.cartesian-solve1/v1", "step": step, "grid_geometry": g.record(),
            "tile_count": self.plan.ntiles, "tile_plan_sha256": self.plan.ownership_sha256,
            "source_order": list(source_order(self.rterm, self.lspgrad)),
            # src/init3d.F:2688-2760: turbsmag only for interp_on_restart.
            "fresh_tke_initialization": step == 1 and self.interpolated_tke_start,
            "retained_tke_source": (
                "preceding completed solve3" if step > 1
                else None if self.interpolated_tke_start
                else "fresh start: tkekm on the zero initial TKE"
            ),
            # ramp_sgs=0 publishes no key, so its receipt is what it was.
            **(
                {"sgs_startup_ramp": (
                    {"turbfac": self.sgs_ramp_step["turbfac"], "rtime_s": self.sgs_ramp_step["rtime_s"],
                     "cme_raw_bits": int(self.sgs_ramp_step["cme"].view("uint32"))}
                    if self.sgs_ramp_step is not None
                    else {"turbfac": 1.0, "cme": "c_m (sealed operator)"}
                )}
                if self.ramp_sgs == 1
                else {}
            ),
            "invalid_operand_count": 0, "tile_workspace_released": True,
            "kernel_source_sha256": self._held_kernel_sources(),
            "lower_boundary": self.surface_selection.record(),
            "surface_fields": (
                dict(self.surface_field_receipt)
                if self.surface_field_receipt is not None
                else None
            ),
            "surface_enthalpy_flux": (
                self.surface_selection.uses_surface_enthalpy_flux
            ),
            # sgsmodel=1 publishes no two-part key, so its receipt and token
            # are exactly what they were (README.namelist:334-352).
            **(
                {
                    "two_part_sgs": {
                        **self.two_part.record(),
                        **self.two_part.step_record(),
                    }
                }
                if self.two_part is not None
                else {}
            ),
            # An rterm=0 run publishes neither key, so its receipt and the
            # operation token sealed from it are exactly what they were.
            **(
                {"rterm": self.rterm, "rterm_authority": CUDA_GM1_RTERM_AUTHORITY}
                if self.rterm == 1
                else {}
            ),
            # The same rule for lspgrad=0: neither key, the same token.
            **(
                {
                    "lspgrad": self.lspgrad,
                    "lspgrad_authority": CARTESIAN_LSPGRAD1_AUTHORITY,
                    "lspgrad_profiles_operation_token_sha256": (
                        self.lspgrad_profiles.receipt["operation_token_sha256"]
                        if self.lspgrad_profiles is not None
                        else None
                    ),
                }
                if self.lspgrad == 1
                else {}
            ),
        }
        receipt["operation_token_sha256"] = _hash(receipt)
        self.last_receipts["held"] = receipt
        cuda_graphs.bookkeeping(self._set_held_step, step)

    def _set_eddy_maximum(self, value: Any) -> None:
        self.max_eddy_coefficient_m2_s = float(value[0])

    def _held_kernel_sources(self) -> dict[str, str]:
        """The sources the held phase launched, by operator.

        The uniform column's record is exactly what it has always been.  A
        flat stretched column names the flat closure image in place of the
        terrain calcdef, stress, turbt and turbs images it does not launch.
        """

        sources: dict[str, str] = {
            **dict(OPERATOR_SOURCE_SHA256), "sgs_stress": self.sgs_compiled.sgs_source_sha256,
            "tke_diffusion": self.sgs_compiled.tke_source_sha256, "retained_sgs": CUDA_SGS_TKE_SOURCE_SHA256,
            "tke_local_source": CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_SHA256,
            "scalar_turbulence": CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256,
            "rayleigh_apply": GM1_CHILD_RAYLEIGH_SOURCE_SHA256,
            **(
                {"surface_drag": self.surface_compiled.source_sha256}
                if self.surface_compiled is not None
                else {}
            ),
            **(
                {"rterm": CUDA_GM1_RTERM_SOURCE_SHA256}
                if self.rterm_compiled is not None
                else {}
            ),
            **(
                {"lspgrad": CUDA_GM1_LSPGRAD1_SOURCE_SHA256}
                if self.lspgrad_compiled is not None
                else {}
            ),
        }
        if self.flat_compiled is not None:
            flat = self.flat_compiled.source_sha256
            sources.pop("stretched_calcdef")
            sources.pop("base_calcdef")
            sources.update(
                flat_calcdef=flat, sgs_stress=flat, tke_diffusion=flat,
                scalar_turbulence=flat,
                scalar_turbulence_pad=CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256,
            )
        return sources

    #: S94's scaled pair for the tile being integrated, or ``None``.
    _two_part_scaled: Any = None

    def _two_part_prepass(self, tile: Mapping[str, Any], work: Mapping[str, Any], surface: Mapping[str, Any], nonconvergent: Any, ni: int, nj: int, nk: int) -> None:
        """The domain means CM1's two-part models take before ``gettau``.

        ``t2psmm``/``t2pcode`` run inside ``sfc_and_turb`` after ``getsfc``,
        ``calcdef`` and ``tkekm`` and before ``gettau`` (src/turb.F:1456-1537
        then :1612), and they average over the whole domain.  A tile cannot see
        the whole domain, so every tile's ``ust``/``znt`` (B20) or strain term
        (S94) is produced by the same operators first and reduced once.  The
        roughness length those operators write back is restored, so the
        integrating pass below starts the water-roughness iteration from the
        value the previous step left, exactly as it would have.
        """

        two_part = self.two_part
        if not two_part.begin(self.model_time_s):
            return
        cp = self.cp
        two_part.means()
        snapshot = None
        if two_part.selection.sgsmodel == 4:
            snapshot = cp.array(self.surface_state["znt"], copy=True)
        for spec in self.plan.specs:
            self._gather(tile, work, spec)
            if two_part.selection.sgsmodel == 4:
                self._surface_layer(tile, work, surface, ni, nj, nk, spec, nonconvergent)
            else:
                self._refresh_nm_def(tile, work, ni, nj, nk)
            two_part.prepass_tile(tile, work, surface, spec)
        if snapshot is not None:
            cp.copyto(self.surface_state["znt"], snapshot)
        two_part.reduce()

    def _gather(self, tile: Mapping[str, Any], work: Mapping[str, Any], spec: Gm1TileSpec) -> None:
        cp, d = self.cp, self.dynamics
        for name, target in tile.items():
            if name in d.fields:
                _copy_tile(cp, target, d.fields[name], spec)
            elif name == "terrain_msl":
                _copy_tile(cp, target, d.metrics[name], spec)
        for name in ("u0", "v0"):
            cp.copyto(tile[name], cp.broadcast_to(d.base_profiles[name][:, None, None], tile[name].shape))
        for name, source in d.metrics.items():
            if name in work:
                _copy_tile(cp, work[name], source, spec)
        cp.copyto(work["ruh"], d.arrays["uh"][3 + spec.ci0:3 + spec.ci0 + spec.cnx])
        cp.copyto(work["rvh"], d.arrays["vh"][3 + spec.cj0:3 + spec.cj0 + spec.cny])

    def _refresh_nm_def(self, tile: Mapping[str, Any], work: Mapping[str, Any], ni: int, nj: int, nk: int) -> None:
        cp, g, v = self.cp, self.grid, self.vertical
        # src/turb.F:5521-5535: calcnm accumulates qt over nqv, nql1..nql2 and
        # nqs1..nqs2 only.  Number concentrations are excluded by name in CM1,
        # so a double-moment axis hands calcnm its contiguous mass prefix, not
        # its whole q3d: adding a kg-1 row into a kg/kg total water would put
        # the moist Brunt-Vaisala frequency off by twenty orders of magnitude.
        mass_rows = self.config.profile.repaired_scalars
        moist = {"c1": work["c1"], "c2": work["c2"], "mf": work["mf"], **{name: (tile[name][:mass_rows] if name == "q3d" else tile[name]) for name in ("pi0", "th0", "prs", "pp3d", "th3d", "q3d")}, "nm": work["nm"], **{name: work[name] for name in MOIST_SCRATCH_NAMES}}
        if tuple(moist) != CHILD_MOIST_ALL_NAMES:
            raise RuntimeError("calcnm input ordering changed")
        launch_cuda_child_moist_nm(moist, ni=ni, nj=nj, nk=nk, dz=g.dz_m, stream=self.stream, compiled=self.closure_compiled.calcnm)
        work["defv"].fill(cp.float32(0))
        work["defh"].fill(cp.float32(0))
        u, f = cp.uint32, cp.float32
        if self.flat_compiled is not None:
            # turb.F:4766-4895 and :5250-5388 (terrain_flag=.false.).
            enqueue_flat_calcdef(
                self.flat_compiled.kernels,
                {**{name: tile[name] for name in ("u3d", "v3d", "w3d", "rho", "rf")},
                 **{name: work[name] for name in ("mf", "c1", "c2", "s11", "s22", "s33", "s12", "s13", "s23", "defv", "defh")}},
                mh_profile=v.mh, ni=ni, nj=nj, nk=nk, rdx=f(1.0) / f(g.dx_m), rdy=f(1.0) / f(g.dy_m),
                rdz=f(v.rdz_f32), cupy=cp, stream=self.stream,
            )
            return
        common = (u(ni), u(nj), u(nk), f(1 / g.dx_m), f(1 / g.dy_m), v.sigma, v.sigmaf, v.rds, v.rdsf, f(g.ztop_m))
        kernels = self.closure_compiled.calcdef.kernels
        totals = (nk * (nj + 1) * (ni + 1), (nk - 1) * (nj + 1) * (ni + 1), (nk - 1) * nj * ni)
        threads = CUDA_TERRAIN_DEFORMATION_THREADS
        launches = tuple(((words + threads - 1) // threads,) for words in totals)
        kernels[0](launches[0], (threads,), (tile["u3d"], tile["v3d"], tile["w3d"], tile["rho"], *(work[name] for name in ("gz", "rgzu", "rgzv", "gx", "gy", "s11", "s22", "s33", "s12")), *common), stream=self.stream)
        kernels[1](launches[1], (threads,), (tile["u3d"], tile["v3d"], tile["w3d"], tile["rf"], *(work[name] for name in ("gz", "rgz", "s13", "s23")), *common), stream=self.stream)
        kernels[2](launches[2], (threads,), (tile["rf"], *(work[name] for name in ("c1", "c2", "s11", "s22", "s33", "s12", "s13", "s23", "defv", "defh")), u(ni), u(nj), u(nk)), stream=self.stream)

    def _tke_coefficients(self, work: Mapping[str, Any], retained: Mapping[str, Any], invalid: Any, spec: Gm1TileSpec, *, first: bool) -> None:
        cp, g = self.cp, self.grid
        ni, nj, nk = spec.cnx, spec.cny, g.nz
        for name in FRESH_OUTPUT_NAMES:
            work[name].fill(cp.float32(0))
        inputs = {name: work[name] for name in ("ruh", "rvh", "rmf", "nm", "defv", "defh")}
        suffix = (cp.uint32(ni), cp.uint32(nj), cp.uint32(nk), cp.float32(g.dx_m), cp.float32(g.dy_m), cp.float32(g.dz_m))
        threads = CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS
        blocks = (min(65535, (ni * nj * (nk + 1) + threads - 1) // threads), 1, 1)
        kernels = self.closure_compiled.fresh_tke.kernels
        kernels[0](blocks, (threads, 1, 1), (*inputs.values(), invalid, *suffix), stream=self.stream)
        if first:
            kernels[1](blocks, (threads, 1, 1), (*inputs.values(), *(work[name] for name in ("grdscl", "lenscl", "dissten", "kmh", "kmv", "khh", "khv", "tkea")), *suffix), stream=self.stream)
            kernels[2](blocks, (threads, 1, 1), (work["tkea"], work["tke3d"], work["tketen"], *suffix[:3]), stream=self.stream)
        else:
            _copy_tile(cp, work["tkea"], self.dynamics.tke["tkea"], spec)
            for value in retained.values():
                value.fill(cp.float32(0))
            arrays = {**inputs, **{name: work[name] for name in ("tkea", "lenscl", "dissten", "kmh", "kmv", "khh", "khv", "tketen")}, **retained}
            if self.sgs_ramp_step is not None:
                # src/turb.F:240-262: tkekm with CM1's ramped cme/ce1/ce2.
                from .cuda_cartesian_sgs_ramp import launch_cuda_cartesian_sgs_ramp

                launch_cuda_cartesian_sgs_ramp(
                    arrays, cme=self.sgs_ramp_step["cme"], ni=ni, nj=nj, nk=nk,
                    dx=g.dx_m, dy=g.dy_m, dz=g.dz_m, stream=self.stream,
                    compiled=self.sgs_ramp_compiled,
                )
            else:
                launch_cuda_sgs_tke(arrays, ni=ni, nj=nj, nk=nk, dx=g.dx_m, dy=g.dy_m, dz=g.dz_m, stream=self.stream, compiled=self.retained_compiled)
            cp.copyto(work["tke3d"], work["tkea"])

    def _tke_held(self, tile: Mapping[str, Any], work: Mapping[str, Any], ni: int, nj: int, nk: int, km: tuple[Any, Any] | None = None) -> None:
        cp, g, v = self.cp, self.grid, self.vertical
        u, f = cp.uint32, cp.float32
        # The local source keeps kmv itself (src/solve1.F:318-345); only turbt
        # takes S94's gamk-scaled pair (src/solve1.F:369-372).
        kmh, kmv = (work["kmh"], work["kmv"]) if km is None else km
        self.local_tke_kernel((((nk - 1) * nj * ni + 127) // 128,), (128,), (*(work[name] for name in ("khv", "nm", "dissten", "kmv", "defv", "defh", "tketen")), u(ni), u(nj), u(nk)), stream=self.stream)
        if self.flat_compiled is not None:
            # turbtend.F:729-1044 (terrain_flag=.false.).
            enqueue_flat_turbt(
                self.flat_compiled.kernels,
                {"tke3d": work["tke3d"], "rho": tile["rho"], "rf": tile["rf"], "mh": work["mh"], "mf": work["mf"], "tketen": work["tketen"]},
                kmh=kmh, kmv=kmv, ni=ni, nj=nj, nk=nk, rdx=f(1.0) / f(g.dx_m), rdy=f(1.0) / f(g.dy_m),
                rdz=f(v.rdz_f32), dt=f(self.config.dt_s), cupy=cp, stream=self.stream,
            )
            return
        self.sgs_compiled.tke_kernels[0](((ni * nj + 127) // 128,), (128,), (work["tke3d"], kmh, kmv, tile["rho"], tile["rf"], *(work[name] for name in ("mh", "mf", "gz", "rgz", "rgzu", "rgzv", "tketen")), u(ni), u(nj), u(nk), f(g.dx_m), f(g.dy_m), f(g.dz_m), f(self.config.dt_s), v.sigma, v.rds, f(g.ztop_m)), stream=self.stream)

    def _held_tile(self, tile: Mapping[str, Any], work: Mapping[str, Any], surface: Mapping[str, Any], scalar: Gm1ScalarTurbulenceWorkspace, idiss: Eqtset2IdissConsumer, ni: int, nj: int, nk: int, rayleigh_scratch: Mapping[str, Any] | None = None) -> None:
        cp, g = self.cp, self.grid
        for name in ("thten1", "qten", "uten", "vten", "wten"):
            tile[name].fill(cp.float32(0))
        count = max(nk * nj * (ni + 1), nk * (nj + 1) * ni)
        # src/solve1.F:440-452.  With neither damper on CM1 takes its ELSE
        # branch and writes zeros into all four held arrays; the fill above
        # covers thten1 only, so the three momentum ones are zeroed here and
        # the Rayleigh operator is not launched at all.
        if not self.rayleigh_active:
            for name in ("uten1", "vten1", "wten1", "thten1"):
                tile[name].fill(cp.float32(0))
        if self.rayleigh_active:
            operands = dict(tile)
            if self.irdamp == 2:
                from .cuda_boundary_families import enqueue_irdamp2_operands

                scratch = rayleigh_scratch or {}
                enqueue_irdamp2_operands(
                    scratch, tile["th0"], tile["tha"], scratch,
                    ni=ni, nj=nj, nk=nk, compiled=self.family_compiled, stream=self.stream,
                )
                operands.update({name: scratch[name] for name in ("u0", "v0", "tha")})
            self.rayleigh_compiled.kernel(((count + 255) // 256, 1, 1), (256, 1, 1), (*(operands[name] for name in CHILD_TERRAIN_RAYLEIGH_ARRAYS), cp.uint32(ni), cp.uint32(nj), cp.uint32(nk), cp.float32(self.damping_rate_s_inv)), shared_mem=0, stream=self.stream)
        budget = self._budget()
        if budget is not None:
            # solve1.F:489-529 records ud_rdamp before solve1.F:538 adds the
            # large-scale pressure gradient, so the damping is taken first.
            budget.after_rayleigh(tile)
        # src/solve1.F:590-612, on the held arrays the Rayleigh block just
        # assigned and before anything else adds to them: the stress below is
        # composed as ((Rayleigh - ul) + stress), CM1's uten1 accumulation.
        if self.lspgrad_compiled is not None:
            launch_cuda_gm1_lspgrad1(
                tile["uten1"],
                tile["vten1"],
                self.lspgrad_ul,
                self.lspgrad_vl,
                ni=ni,
                nj=nj,
                nk=nk,
                stream=self.stream,
                compiled=self.lspgrad_compiled,
            )
            if budget is not None:
                budget.large_scale_pressure_gradient(tile, self.lspgrad_ul, self.lspgrad_vl)
        if budget is not None:
            budget.theta_mark(tile, "diss")
        self._dissipative_heating(tile, work, surface, idiss, ni, nj, nk)
        if budget is not None:
            budget.theta_delta(tile, "diss")
            budget.theta_mark(tile, "rad")
        # src/solve1.F:1003-1054, after the idiss block and before the
        # subgrid turbulence: the capped Rotunno-Emanuel theta tendency.
        if self.rterm_compiled is not None:
            launch_cuda_gm1_rterm(
                tile["thten1"],
                tile["tha"],
                ni=ni,
                nj=nj,
                nk=nk,
                stream=self.stream,
                compiled=self.rterm_compiled,
            )
        if budget is not None:
            budget.theta_delta(tile, "rad")
        stress_arrays = {name: tile[name] if name in tile else work[name] for name in sgs_stress_shapes(ni, nj, nk)}
        two_part_kwargs: dict[str, Any] = {}
        if self.two_part is not None:
            # src/turb.F:1612-1628: S94's gettau and turbu/v/w take gamk*kmh
            # and gamk*kmv; both models run turbuz/turbvz's dot2p block.
            scaled = self._two_part_scaled
            if scaled is not None:
                stress_arrays["kmh"], stress_arrays["kmv"] = scaled
            two_part_kwargs["two_part"] = self.two_part.column_binding()
        if self.flat_compiled is not None:
            self._flat_stress(stress_arrays, surface, work, ni, nj, nk, two_part_kwargs.get("two_part"))
        else:
            launch_cuda_gm1_child_stretched_sgs_stress(stress_arrays, metrics=self.vertical_view, ni=ni, nj=nj, nk=nk, dx=g.dx_m, dy=g.dy_m, dz=g.dz_m, ztop=g.ztop_m, dt=self.config.dt_s, stream=self.stream, compiled=self.sgs_compiled, allow_uniform_idealized_validation=True, geometry=g, selection=self.stress_selection, surface=self._surface_binding(surface, work), **two_part_kwargs)
        if budget is not None:
            # The replay reads the same stress_arrays -- for sgsmodel=3/4 the
            # gamk-scaled kmh/kmv substituted above -- and the same two-part
            # column binding, so ub/vb_vturb carry turbuz/turbvz's dot2p flux
            # exactly where src/turbtend.F:1767-1802 puts it: inside turbz.
            budget.momentum_turbulence(self, stress_arrays, ni=ni, nj=nj, nk=nk, two_part=two_part_kwargs.get("two_part"))
        scalar_arrays = {name: tile[name] if name in tile else work[name] for name in child_stretched_scalar_turbulence_shapes(ni, nj, nk, self.config.numq)}
        self._scalar_held(scalar_arrays, scalar, ni, nj, nk, surface)
        cp.add(tile["uten1"], tile["uten"], out=tile["uten"])
        cp.add(tile["vten1"], tile["vten"], out=tile["vten"])
        cp.add(tile["wten1"], tile["wten"][:nk], out=tile["wten"][:nk])

    def _flat_stress(self, stress: Mapping[str, Any], surface: Mapping[str, Any], work: Mapping[str, Any], ni: int, nj: int, nk: int, two_part: Any) -> None:
        """gettau, sfcstress and turbu/turbv/turbw on a flat column.

        turb.F:4094-4433 then solve1.F:1138-1150.  ``sfcstress`` for
        bbc=2/3 replaces the free-slip lower face after the interior build
        and before the divergence reads it (turb.F:4417), exactly as on the
        uniform column.
        """

        cp, g, v = self.cp, self.grid, self.vertical
        f = cp.float32
        kernels = self.flat_compiled.kernels
        enqueue_flat_gettau(kernels, stress, ni=ni, nj=nj, nk=nk, cupy=cp, stream=self.stream)
        binding = self._surface_binding(surface, work)
        if binding is not None:
            enqueue_surface_stress(
                binding,
                {name: stress[name] for name in ("t13", "t23", "rf", "mf", "kmv", "u3d", "v3d")},
                ni=ni, nj=nj, nk=nk, stream=self.stream,
            )
        enqueue_flat_momentum_stress(
            kernels, stress, mh_profile=v.mh, ni=ni, nj=nj, nk=nk,
            rdx=f(1.0) / f(g.dx_m), rdy=f(1.0) / f(g.dy_m), rdz=f(v.rdz_f32), dt=f(self.config.dt_s),
            cupy=cp, stream=self.stream, two_part=two_part,
        )

    def _rayleigh_horizontal_averages(self, scratch: Mapping[str, Any]) -> None:
        """``src/testcase_simple_phys.F:307-375``: the area-weighted level means.

        ``get_avg_uvtq`` sums ``ua(i,j,k)*dx*dy*ruf(i)*rvh(j)`` over the whole
        domain and divides by ``(maxx-minx)*(maxy-miny)``.  ``ruh``/``rvh``/
        ``ruf``/``rvf`` are exactly one on this port's uniform horizontal grid,
        and ``dx*dy`` then cancels against the area for every level, so the
        mean is the plain level mean of the interior.  ``thavg`` is the mean of
        ``th0+tha``, the full potential temperature, not of the perturbation.

        The reduction is a device tree sum rather than CM1's serial i-then-j
        double accumulation, so this coefficient is accurate to the reduction
        and not raw-word identical to the Fortran order; every run receipt that
        selects ``irdamp=2`` says so.
        """

        cp, d, g = self.cp, self.dynamics, self.grid
        nk, nj, ni = g.nz, g.ny, g.nx
        # src/testcase_simple_phys.F:333-336 sums i=1..ni and j=1..nj, so the
        # extra staggered face of each wind is outside the average.
        for name, source in (("uavg", d.fields["ua"]), ("vavg", d.fields["va"])):
            interior = source[:nk, :nj, :ni]
            mean = cp.mean(interior.astype(cp.float64), axis=(1, 2)).astype(cp.float32)
            cp.copyto(scratch[name], mean)
        total = cp.add(d.fields["th0"][:nk, :nj, :ni], d.fields["tha"][:nk, :nj, :ni], dtype=cp.float32)
        cp.copyto(
            scratch["thavg"], cp.mean(total.astype(cp.float64), axis=(1, 2)).astype(cp.float32)
        )

    def _surface_binding(
        self, surface: Mapping[str, Any], work: Mapping[str, Any] | None = None
    ) -> SurfaceDragBinding | None:
        """Bind this step's surface carriers to CM1's lower-boundary selection.

        ``zf`` travels with the surface state because ``src/solve1.F:956``
        needs the depth of the lowest model level,
        ``zf(i,j,2)-zf(i,j,1)``, to form the surface dissipation rate, and
        that rate is written from this binding inside the dissipative-heating
        block rather than by any carrier the phase already holds.
        """

        if not self.surface_selection.active:
            return None
        arrays = dict(surface)
        if work is not None and "zf" in work:
            arrays["zf"] = work["zf"]
        compiled = self.surface_compiled
        if compiled is None:
            raise RuntimeError(
                "the surface drag kernels were never compiled for a "
                "non-free-slip lower boundary"
            )
        return SurfaceDragBinding(
            selection=self.surface_selection,
            compiled=compiled,
            arrays=arrays,
            dz=self.grid.dz_m,
            surface_pressure_weights=(
                self.surface_pressure_weights
                if self.surface_selection.uses_surface_enthalpy_flux
                else None
            ),
        )

    def _surface_layer(self, tile: Mapping[str, Any], work: Mapping[str, Any], surface: Mapping[str, Any], ni: int, nj: int, nk: int, spec: Gm1TileSpec, nonconvergent: Any) -> None:
        """CM1's ``getsfc`` block: the lowest-level wind and ``getcecd``.

        ``src/turb.F:296-307`` forms the ground-relative wind from the frame
        wind and ``umove``/``vmove``, ``src/turb.F:539-543`` averages it to
        the scalar point, and ``src/turb.F:614-616`` calls ``getcecd``, which
        turns the declared constant drag coefficient into the roughness
        length and friction velocity the lower boundary needs
        (``src/sfcphys.F:120-136``).  ``tkebc`` is a separate CM1 call after
        ``tkekm`` and is enqueued by :meth:`_surface_tke_face`.  Nothing runs
        for a free-slip bottom.
        """

        binding = self._surface_binding(surface, work)
        if binding is None:
            return
        cp = self.cp
        # src/init_surface.F:40-65 wrote these for the whole domain before the
        # first step; this tile reads its own rectangle of them, and writes
        # the roughness length back because getcecd is allowed to change it
        # (src/sfcphys.F:108,131,170).
        for name in SURFACE_PERSISTENT_NAMES:
            _copy_tile(cp, surface[name], self.surface_state[name], spec)
        enqueue_surface_layer(
            binding,
            {
                "u3d": tile["u3d"],
                "v3d": tile["v3d"],
                "zh": work["zh"],
                "terrain_msl": tile["terrain_msl"],
            },
            ni=ni,
            nj=nj,
            nk=nk,
            stream=self.stream,
            nonconvergent=nonconvergent,
        )
        _scatter(cp, self.surface_state["znt"], surface["znt"], spec)

    def _surface_enthalpy_flux(
        self,
        tile: Mapping[str, Any],
        surface: Mapping[str, Any],
        ni: int,
        nj: int,
        nk: int,
    ) -> None:
        """``src/turb.F:645-649``: sfcflux, immediately after getcecd.

        CM1 calls it inside the same ``sfcmodel.eq.1`` block, before
        ``tkekm``, so the heat and moisture fluxes the scalar diffusion
        consumes later in the step are the ones this step's exchange
        coefficients produced.  ``divx`` is CM1's carrier for ``qv`` here
        (``src/turb.F:558``), which is ``qa`` index 0 in this port.
        """

        if not self.surface_selection.uses_surface_enthalpy_flux:
            return
        binding = self._surface_binding(surface)
        enqueue_surface_enthalpy_flux(
            binding,
            {
                "prs": tile["prs"],
                "th0": tile["th0"],
                "tha": tile["tha"],
                "qva": tile["qa"][0],
            },
            ni=ni,
            nj=nj,
            nk=nk,
            moisture=True,
            stream=self.stream,
        )

    def _fold_eddy_coefficient_maximum(
        self, work: Mapping[str, Any], accumulator: Any, spec: Gm1TileSpec, nk: int
    ) -> None:
        """``calcksquick`` (misclibs.F:2846-2866) reduced to one scalar.

        CM1 forms three expressions per cell at ``k >= 2`` -- one each from
        ``kmh``, ``khh`` and ``kmv`` -- and keeps the largest over the whole
        domain.  With ``uh = vh = 1`` every one of them is the same constant
        ``sqrt(dtdx**2 + dtdy**2)`` times its own coefficient, so their joint
        maximum is that constant times the maximum of the three coefficients.
        The constant carries ``dt``, so it is applied by the controller at the
        step being considered rather than baked in here.

        ``k >= 2`` is CM1's own loop bound and is GM1 plane index ``1``
        upward: ``gm1_tiled_solve1_tke_local_source_f32_v1`` writes exactly
        planes ``1..nk-1`` of these ``(nk+1)``-plane carriers for CM1's
        ``k=2..nk``.  Only the cells this tile owns are reduced, so a halo
        gathered from a neighbour cannot be counted twice or counted from a
        previous tile's leftovers.
        """

        cp = self.cp
        y0, y1, x0, x1 = spec.owned_rect("mass")
        rows = slice(y0 - spec.cj0, y1 - spec.cj0)
        columns = slice(x0 - spec.ci0, x1 - spec.ci0)
        for name in ("kmh", "khh", "kmv"):
            window = work[name][1:nk, rows, columns]
            cp.maximum(accumulator, window.max(), out=accumulator)

    def _surface_tke_face(self, work: Mapping[str, Any], surface: Mapping[str, Any], ni: int, nj: int, nk: int) -> None:
        """``tkebc``'s ``bbc=2/3`` lower face (``src/turb.F:1378``).

        CM1 calls ``tkebc`` once per step, after ``tkekm``
        (``src/turb.F:1351``) has written every interior eddy coefficient,
        and its ``bbc=2/3`` branch then overwrites the free-slip lower face
        those operators just produced.  It reads ``khh``/``khv``/``kmh``/
        ``kmv`` at ``k=2``, so it runs here and nowhere else: enqueued
        before the coefficient operators it would read uninitialized
        workspace, and enqueued twice it would be one wasted launch per tile
        per step.
        """

        binding = self._surface_binding(surface)
        if binding is None:
            return
        enqueue_surface_tke_face(
            binding,
            {name: work[name] for name in ("tkea", "kmh", "kmv", "khh", "khv")},
            ni=ni,
            nj=nj,
            nk=nk,
            stream=self.stream,
        )

    def _dissipative_heating(self, tile: Mapping[str, Any], work: Mapping[str, Any], surface: Mapping[str, Any], idiss: Eqtset2IdissConsumer, ni: int, nj: int, nk: int) -> None:
        """``src/solve1.F:917-988`` dissipative heating, with CM1's ``k1``.

        For a free-slip or no-slip bottom ``k1`` is 1 and the sealed
        supplied-dissten consumer owns every level.  For ``bbc=3`` CM1 sets
        ``k1=2`` (``src/solve1.F:932-935``) and heats ``k=1`` with the
        surface-layer dissipation rate instead, so the same sealed consumer
        runs over the ``k=2..nk`` slice and the lowest level is heated by
        :mod:`cm1core.eqtset2_surface_dissipation`.

        ``src/solve1.F:955-957`` forms that surface rate from ``ust`` and
        ``zntmp`` inside this very block, immediately before it is consumed.
        This port materializes it in a carrier, so the launch that writes it
        is enqueued here, ahead of the read -- not in ``sfcstress``, which is
        enqueued later in the chain and would leave the lowest model level
        heated by a carrier that is still zero.
        """

        species = {name: tile["qa"][index] for index, name in enumerate(("qv", "qc", "qr", "qi", "qs", "qg"))}
        if self.surface_selection.bbc != 3:
            idiss.apply(Eqtset2IdissCarriers(thten1=tile["thten1"], dissten=work["dissten"], pi0=tile["pi0"], ppi=tile["ppi"], **species))
            return
        # bbc=3 is active by construction, so the binding exists here.
        enqueue_surface_dissipation(
            self._surface_binding(surface, work),
            {},
            ni=ni,
            nj=nj,
            nk=nk,
            stream=self.stream,
        )
        idiss.apply(
            Eqtset2IdissCarriers(
                thten1=tile["thten1"][1:],
                dissten=work["dissten"][1:],
                pi0=tile["pi0"][1:],
                ppi=tile["ppi"][1:],
                **{name: value[1:] for name, value in species.items()},
            )
        )
        apply_surface_dissipative_heating(
            SurfaceDissipationCarriers(
                thten1=tile["thten1"],
                epsd_surface=surface["epsd_sfc"],
                pi0=tile["pi0"],
                ppi=tile["ppi"],
                **species,
            ),
            nx=ni,
            ny=nj,
            nz=nk,
            equation_set=2,
            stream=self.stream,
        )

    def _scalar_held(self, a: Mapping[str, Any], w: Gm1ScalarTurbulenceWorkspace, ni: int, nj: int, nk: int, surface: Mapping[str, Any]) -> None:
        cp, g, v = self.cp, self.grid, self.vertical
        u, f = cp.uint32, cp.float32
        kpad, kpack, krecon, kflux, kinterp, kapply = self.scalar_compiled.kernels
        # src/solve1.F:1208,1217 pass thflux for theta and qvflux for qv, and
        # src/solve1.F:1222,1233 pass iflux=0 for every other scalar, so only
        # those two get a nonzero lower flux at src/turbtend.F:595.  Without
        # a surface energy budget every scalar keeps the source's zero, and
        # the pointer -- which the kernel never dereferences then -- is the
        # (nj,ni) metric array the launch already carries.
        enthalpy = self.surface_selection.uses_surface_enthalpy_flux
        lower_flux = {
            0xFFFFFFFF: surface["thflux"] if enthalpy else a["gz"],
            0: surface["qvflux"] if enthalpy else a["gz"],
        }
        largest = max((nk + 1) * (nj + 2) * (ni + 2), nk * nj * (ni + 1), nk * (nj + 1) * ni)
        threads = CUDA_GM1_SCALAR_TURBULENCE_THREADS
        grid, block = ((largest + threads - 1) // threads, 1, 1), (threads, 1, 1)
        columns = ((ni * nj + threads - 1) // threads, 1, 1)
        dims = (u(ni), u(nj), u(nk))
        kpad(grid, block, (a["rho"], a["khh"], a["gz"], a["rgz"], w.rho_halo, w.khh_halo, w.gz_halo, w.rgz_halo, *dims), stream=self.stream)
        flat = self.flat_compiled
        # src/solve1.F:1213-1231: turbs is called for theta and then once
        # per n=1,numq, so a double-moment axis diffuses its number
        # concentrations exactly as CM1 does.
        for species, target in ((0xFFFFFFFF, a["thten1"]), *((index, a["qten"][index]) for index in range(self.config.numq))):
            kpack(grid, block, (a["th0"], a["tha"], a["qa"], w.scalar_halo, *dims, u(species)), stream=self.stream)
            if flat is not None:
                self._flat_scalar_species(a, w, species, target, lower_flux, enthalpy, grid, block, columns, dims)
                continue
            krecon(grid, block, (w.scalar_halo, v.sigma, v.sigmaf, v.rds, w.scalar_w, *dims), stream=self.stream)
            kflux(grid, block, (w.scalar_halo, w.scalar_w, w.rho_halo, w.khh_halo, w.gz_halo, w.rgz_halo, a["rgzu"], a["rgzv"], a["gxu"], a["gyv"], v.rdsf, w.flux_x, w.flux_y, *dims, f(g.dx_m), f(g.dy_m)), stream=self.stream)
            kinterp(grid, block, (w.flux_x, w.flux_y, v.sigma, v.sigmaf, v.rds, w.flux_x_w, w.flux_y_w, *dims), stream=self.stream)
            sflux = lower_flux.get(species, a["gz"])
            iflux = u(1 if enthalpy and species in lower_flux else 0)
            kapply(columns, block, (w.scalar_halo, a["rho"], a["rr"], a["rf"], a["mh"], a["mf"], a["khv"], a["gz"], a["rgzu"], a["rgzv"], a["gx"], a["gy"], v.rdsf, w.flux_x, w.flux_y, w.flux_x_w, w.flux_y_w, sflux, w.thomas_c, w.thomas_d, target, *dims, f(g.dx_m), f(g.dy_m), f(g.dz_m), f(self.config.dt_s), iflux), stream=self.stream)
            budget = self._budget()
            if budget is not None and species in (0xFFFFFFFF, 0):
                budget.scalar_turbulence(self, species=species, kernel=kapply, arguments={
                    "sh": w.scalar_halo, "rho": a["rho"], "rr": a["rr"], "rf": a["rf"], "mh": a["mh"],
                    "mf": a["mf"], "khv": a["khv"], "gz": a["gz"], "rgzu": a["rgzu"], "rgzv": a["rgzv"],
                    "gx": a["gx"], "gy": a["gy"], "rdsf": v.rdsf, "fx": w.flux_x, "fy": w.flux_y,
                    "fxw": w.flux_x_w, "fyw": w.flux_y_w, "sflux": sflux, "cp": w.thomas_c,
                    "dp": w.thomas_d, "target": target, "ni": ni, "nj": nj, "nk": nk,
                    "dx": g.dx_m, "dy": g.dy_m, "dz": g.dz_m, "dt": self.config.dt_s, "iflux": int(iflux),
                })

    def _flat_scalar_species(self, a: Mapping[str, Any], w: Gm1ScalarTurbulenceWorkspace, species: int, target: Any, lower_flux: Mapping[int, Any], enthalpy: bool, grid: Any, block: Any, columns: Any, dims: tuple[Any, ...]) -> None:
        """turbs' flat fluxes and turbsz for one scalar (turbtend.F:60-434)."""

        cp, g = self.cp, self.grid
        u, f = cp.uint32, cp.float32
        kernels = self.flat_compiled.kernels
        kernels["gm1_flat_turbs_xy_flux_v1"](grid, block, (w.scalar_halo, w.rho_halo, w.khh_halo, w.flux_x, w.flux_y, *dims, f(g.dx_m), f(g.dy_m)), stream=self.stream)
        sflux = lower_flux.get(species, a["gz"])
        iflux = u(1 if enthalpy and species in lower_flux else 0)
        arguments = {
            "sh": w.scalar_halo, "rho": a["rho"], "rr": a["rr"], "rf": a["rf"], "mh": a["mh"],
            "mf": a["mf"], "khv": a["khv"], "fx": w.flux_x, "fy": w.flux_y, "sflux": sflux,
            "cp": w.thomas_c, "dp": w.thomas_d, "target": target,
            "ni": int(dims[0]), "nj": int(dims[1]), "nk": int(dims[2]),
            "dx": g.dx_m, "dy": g.dy_m, "dz": g.dz_m, "dt": self.config.dt_s, "iflux": int(iflux),
        }
        kapply = kernels["gm1_flat_turbs_implicit_apply_v1"]
        kapply(columns, block, (w.scalar_halo, a["rho"], a["rr"], a["rf"], a["mh"], a["mf"], a["khv"], w.flux_x, w.flux_y, sflux, w.thomas_c, w.thomas_d, target, *dims, f(g.dx_m), f(g.dy_m), f(g.dz_m), f(self.config.dt_s), iflux), stream=self.stream)
        budget = self._budget()
        if budget is not None and species in (0xFFFFFFFF, 0):
            budget.scalar_turbulence(self, species=species, kernel=kapply, arguments=arguments, flat=True)

    def tke_rk(self, nrk: int) -> None:
        if type(nrk) is not int or nrk != self._tke_stage + 1 or nrk not in (1, 2, 3) or self._held_step < 1:
            raise RuntimeError("TKE RK requires the next stage after source-ordered held tendencies")
        cp, g, d = self.cp, self.grid, self.dynamics
        storage = _PhaseStorage(self.registry, "closure.tke-stage", [])
        tile: dict[str, Any] = {}
        next_state: Any = None
        try:
            tile = {name: storage.allocate(name, shape) for name, shape in self.allocation_plan["tke_stage_shapes"].items()}
            next_state = storage.allocate("next", g.w_shape)
            with self.stream:
                for spec in self.plan.specs:
                    for name in ("tkea", "tke3d", "tketen"):
                        _copy_tile(cp, tile[name], d.tke[name], spec)
                    for name in ("c1", "c2", "rrf0", "gz"):
                        _copy_tile(cp, tile[name], d.metrics[name], spec)
                    _copy_tile(cp, tile["divx"], d.moist_bridge.divx, spec)
                    for name in ("rru", "rrv", "rrw"):
                        target = tile[name]
                        cp.copyto(target, d.arrays[name][:, spec.cj0:spec.cj0 + target.shape[-2], spec.ci0:spec.ci0 + target.shape[-1]])
                    words = int(tile["output"].size)
                    f, u = cp.float32, cp.uint32
                    dttmp = f(f(self.config.dt_s) / f(4 - nrk))
                    weps = float(f(f(100) * f(1e-18)).item())
                    if self.flat_compiled is not None:
                        # adv.F:3252-3355 (iamw=.false., terrain_flag=.false.)
                        # and solve2.F:1708-1803.
                        self.flat_compiled.tke_kernel(((words + 127) // 128,), (128,), (*(tile[name] for name in ("tke3d", "tkea", "tketen", "c1", "c2", "rrf0", "rru", "rrv", "rrw", "divx", "output")), u(spec.cnx), u(spec.cny), u(g.nz), u(3), f(g.dx_m), f(g.dy_m), dttmp, u(nrk == 3), cp.float64(weps), self.vertical.mf, f(self.vertical.rdz_f32)), stream=self.stream)
                    else:
                        self.sgs_compiled.tke_kernels[1](((words + 127) // 128,), (128,), (*(tile[name] for name in ("tke3d", "tkea", "tketen", "c1", "c2", "rrf0", "gz", "rru", "rrv", "rrw", "divx", "output")), u(spec.cnx), u(spec.cny), u(g.nz), u(3), f(g.dx_m), f(g.dy_m), f(g.dz_m), dttmp, u(nrk == 3), cp.float64(weps), self.vertical.rds), stream=self.stream)
                    _scatter(cp, next_state, tile["output"], spec)
                cp.copyto(d.tke["tke3d"], next_state)
        finally:
            tile.clear()
            next_state = None
            storage.release()
        cuda_graphs.bookkeeping(lambda _step, nrk=nrk: setattr(self, "_tke_stage", nrk), self._held_step)
        self.last_receipts["tke_rk"] = {
            "schema": "gm1.cartesian-tke-rk/v1", "step": self._held_step, "nrk": nrk,
            "source_sha256": (
                GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE_SHA256
                if self.flat_compiled is not None
                else self.sgs_compiled.tke_source_sha256
            ),
            "column_workspace_max_nk": self.sgs_compiled.max_nk,
            "fixed_tkea_baseline": True, "immutable_current_during_tile_scatter": True,
            "uses_weno": nrk == 3, "tile_workspace_released": True,
        }
